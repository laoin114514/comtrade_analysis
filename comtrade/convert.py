"""数值转换：原始采样值 → 系统内部的工程值。

合同依据
    第一条 2.(1)② "录波数据由原始文件到系统内部分析数据的自动转换"
对应需求
    P-06（统一内部数据格式）、E-01/E-02（有效值精度）、NF-32（通道数据兼容）

转换链（顺序不可调换）
    1. 识别缺失值哨兵   → 标记为无效（必须在换算之前，否则哨兵会被换算成"看似正常"的值）
    2. 取值范围校验     → 记录诊断（字节序错误、量程不符会在这里暴露）
    3. a/b 换算         → ``工程值 = a × 原始值 + b``
    4. 单位归一化       → kV→V、kA→A
    5. 一次/二次值换算  → 按 PS 字段与变比换算到一次值
    6. 无效点置 NaN

    **换算在解析层只做一次。** 如果留给波形、算法模块各自做，
    同一个系数会散落在多个模块里，出错时无法定位是谁算错的。

关于 ASCII / FLOAT32 是否已换算的二义性
    标准与主流实现（python-comtrade / comtrade-rs / GSF / pycomtrade）
    都把 ASCII 的模拟量当作原始值，同样施加 a/b 换算。
    但现场确实存在直接写工程量的 ASCII 文件，盲目换算会得到静默错误的结果。
    本模块按 :class:`~comtrade.options.AsciiScaling` 策略处理，默认 AUTO：
    用数据跨度与 cfg 声明的量程跨度比对来自动判定。
    **判定结果按文件聚合后一次性报告** —— 一个根因只出一条诊断，
    避免 7 个通道刷 7 条同样的警告把界面淹掉。
"""
from __future__ import annotations

import numpy as np

from .diagnostics import Code, DiagnosticCollector
from .models import AnalogChannel, ComtradeVersion, DataFileType
from .options import AsciiScaling, ParseOptions, DEFAULT_OPTIONS
from .units import parse_unit

#: 缺失值哨兵超过该占比时，判定为"哨兵值猜错了"，放弃屏蔽。
#: 用于防止 1991 版二进制把 -1 当哨兵、而该文件恰好大量出现 -1 的误伤。
MAX_MISSING_FRACTION = 0.2

#: 绝对数量下限：出现次数不超过该值时不看占比，直接屏蔽。
#: 录制点很少（如测试文件）时占比没有统计意义；
#: 而在正常规模的录波里，一个值只出现几次，也不可能是"正常数据值"。
MIN_MISSING_ABSOLUTE = 8

#: 判定依据标签
_R_A_ZERO = "a_zero"
_R_BINARY = "binary"
_R_IDENTITY = "identity"
_R_ALWAYS = "always"
_R_NEVER = "never"
_R_PRESCALED = "prescaled"
_R_FLOAT32 = "float32"


def missing_sentinel(data_type: DataFileType, version: ComtradeVersion) -> float | None:
    """返回该版本 + 编码约定下的缺失值哨兵。

    1991 版 ASCII 用空字段表示缺失（由解析器转成 NaN），此处返回 None。
    """
    if data_type is DataFileType.ASCII:
        return 99999.0 if version >= ComtradeVersion.V1999 else None
    if data_type is DataFileType.BINARY:
        # 1991: 0xFFFF；1999 起: 0x8000。按 int16 读取时分别是 -1 与 -32768。
        # 注意 1991 的 -1 落在正常量程内，存在误判可能，由占比守卫兜底。
        return -1.0 if version is ComtradeVersion.V1991 else -32768.0
    if data_type is DataFileType.BINARY32:
        return float(np.iinfo(np.int32).min)
    if data_type is DataFileType.FLOAT32:
        return float(-np.finfo(np.float32).max)
    return None


def convert_analog_channels(
    channels: list[AnalogChannel],
    analog_raw: np.ndarray | None,
    *,
    data_type: DataFileType,
    version: ComtradeVersion,
    diagnostics: DiagnosticCollector,
    options: ParseOptions = DEFAULT_OPTIONS,
    location: str = "",
) -> None:
    """就地把原始矩阵转换成工程量，写回每个通道的 ``values`` 等字段。

    就地写入是有意为之：整个解析流程只产出一次 Recording，
    在这里做一次深拷贝等于白白翻倍内存。
    """
    if analog_raw is None or analog_raw.shape[0] == 0:
        return

    sentinel = missing_sentinel(data_type, version)
    out_dtype = options.resolved_value_dtype()

    # 按文件聚合的判定结果：一个根因只报一条诊断
    prescaled: list[str] = []
    scale_skipped: list[str] = []
    float32_untouched: list[str] = []
    missing_hits: list[tuple[str, int]] = []
    out_of_range: list[tuple[str, int]] = []

    for ch in channels:
        if ch.index >= analog_raw.shape[0]:
            continue

        raw = np.asarray(analog_raw[ch.index], dtype=np.float64)
        # 注：skew（通道时偏）在标准里是"该通道采样时刻相对基准的偏移"，
        # 现场文件几乎恒为 0。这里不做平移 —— 按错误的理解去平移数据，
        # 比不平移的危害大得多。skew 原值保留在通道对象里供后续按需处理。

        if options.keep_raw_values:
            ch.raw_values = raw.copy()

        # ---------------------------------------------------- 1. 缺失值哨兵
        invalid = np.zeros(raw.shape, dtype=bool)
        if options.mask_missing_values and sentinel is not None:
            hit = raw == sentinel
            count = int(hit.sum())
            if count:
                fraction = count / raw.size
                if count <= MIN_MISSING_ABSOLUTE or fraction <= MAX_MISSING_FRACTION:
                    invalid |= hit
                    missing_hits.append((ch.name, count))
                else:
                    # 占比过高，说明这个值属于正常数据而不是缺失标记
                    diagnostics.warn(
                        Code.DAT_MISSING_VALUE,
                        f"通道「{ch.name}」有 {fraction:.1%} 的采样点等于候选缺失值 "
                        f"{sentinel:g}，占比过高，判断为该值属于正常数据，未做屏蔽",
                        location=location,
                    )

        # -------------------------------------------------- 2. 取值范围校验
        if ch.raw_max > ch.raw_min:
            oor = (raw < ch.raw_min) | (raw > ch.raw_max)
            oor &= ~invalid  # 哨兵本身必然越界，不重复计数
            count = int(oor.sum())
            if count:
                out_of_range.append((ch.name, count))
                if options.mask_out_of_range:
                    invalid |= oor

        # ------------------------------------------------------ 3. a/b 换算
        apply_scale, reason = _decide_scaling(ch, raw, data_type, options)
        if reason == _R_PRESCALED:
            prescaled.append(ch.name)
        elif reason == _R_FLOAT32:
            float32_untouched.append(ch.name)
        elif reason == _R_A_ZERO:
            scale_skipped.append(ch.name)

        if apply_scale:
            values = raw * ch.a + ch.b
            ch.scaling_applied = True
        else:
            values = raw.copy()
            ch.scaling_applied = False

        # --------------------------------------------------- 4. 单位归一化
        unit_info = parse_unit(ch.unit_raw)
        if unit_info.recognized and unit_info.scale != 1.0:
            values *= unit_info.scale
        ch.unit = unit_info.canonical or ch.unit_raw

        # ------------------------------------- 5. 一次值 / 二次值换算（PS）
        _apply_ratio(ch, values, diagnostics, options, location)

        # ------------------------------------------------------ 6. 无效点置 NaN
        if invalid.any():
            values[invalid] = np.nan
        ch.invalid_mask = invalid if invalid.any() else None

        ch.values = values.astype(out_dtype, copy=False)

    # ------------------------------------------------------------ 汇总诊断
    _report(diagnostics, location, missing_hits, out_of_range,
            prescaled, scale_skipped, float32_untouched)


def _report(
    diagnostics: DiagnosticCollector,
    location: str,
    missing_hits: list[tuple[str, int]],
    out_of_range: list[tuple[str, int]],
    prescaled: list[str],
    scale_skipped: list[str],
    float32_untouched: list[str],
) -> None:
    """把逐通道的判定结果聚合成按文件维度的诊断。"""
    if missing_hits:
        total = sum(n for _, n in missing_hits)
        preview = "、".join(f"{name}({n})" for name, n in missing_hits[:6])
        diagnostics.info(
            Code.DAT_MISSING_VALUE,
            f"检测到 {total} 个缺失值采样点，已标记为无效（NaN）：{preview}"
            + ("…" if len(missing_hits) > 6 else ""),
            location=location,
        )

    if out_of_range:
        total = sum(n for _, n in out_of_range)
        preview = "、".join(f"{name}({n})" for name, n in out_of_range[:6])
        diagnostics.info(
            Code.DAT_OUT_OF_RANGE,
            f"有 {total} 个采样点超出 cfg 声明的原始值范围（{preview}"
            + ("…" if len(out_of_range) > 6 else "")
            + "）。若是大量通道普遍越界，通常说明字节序判断或通道数量有误",
            location=location,
        )

    if prescaled:
        diagnostics.warn(
            Code.DAT_ASCII_PRESCALED,
            f"有 {len(prescaled)} 个通道的数据跨度远小于 cfg 声明的量程跨度，"
            "判定数据已经是工程量、已跳过 a/b 换算："
            + "、".join(prescaled[:8])
            + ("…" if len(prescaled) > 8 else "")
            + "。若判定有误，请将 ascii_scaling 设为 always",
            location=location,
        )

    if float32_untouched:
        diagnostics.info(
            Code.DAT_FLOAT32_SCALED,
            f"FLOAT32 编码的 {len(float32_untouched)} 个通道按工程量直接读取（未施加 a/b 换算），"
            "这是该编码的常规做法",
            location=location,
        )

    if scale_skipped:
        diagnostics.warn(
            Code.DAT_SCALE_SKIPPED,
            f"有 {len(scale_skipped)} 个通道的比例系数 a 为 0，"
            "已跳过 a/b 换算、输出原始值（这些通道的数值不可用于计算）："
            + "、".join(scale_skipped[:8]),
            location=location,
        )


def _decide_scaling(
    ch: AnalogChannel,
    raw: np.ndarray,
    data_type: DataFileType,
    options: ParseOptions,
) -> tuple[bool, str | None]:
    """判定是否对某通道施加 a/b 换算，并给出判定依据。"""
    if ch.a == 0.0:
        return False, _R_A_ZERO

    # 恒等映射：走哪条路径结果都一样，不必判定
    if abs(ch.a - 1.0) < 1e-15 and abs(ch.b) < 1e-15:
        return True, _R_IDENTITY

    # 二进制整型必然是 ADC 原始计数，一定要换算
    if data_type in (DataFileType.BINARY, DataFileType.BINARY32):
        return True, _R_BINARY

    if options.ascii_scaling is AsciiScaling.ALWAYS:
        return True, _R_ALWAYS
    if options.ascii_scaling is AsciiScaling.NEVER:
        return False, _R_PRESCALED

    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return True, None

    data_span = float(finite.max() - finite.min())
    raw_span = ch.raw_max - ch.raw_min
    if raw_span > 0 and data_span > 0:
        if data_span / raw_span < options.prescaled_span_ratio:
            # ASCII 里出现这种情况是异常、需要提醒；
            # FLOAT32 里这是常规做法，仅作提示
            reason = (
                _R_FLOAT32 if data_type is DataFileType.FLOAT32 else _R_PRESCALED
            )
            return False, reason

    return True, None


def _apply_ratio(
    ch: AnalogChannel,
    values: np.ndarray,
    diagnostics: DiagnosticCollector,
    options: ParseOptions,
    location: str,
) -> None:
    """按 PS 字段把数值统一到一次值。

    ``PS = S`` → 文件中是二次值，需乘 ``primary / secondary``；
    ``PS = P`` → 已是一次值，不动。

    注意
        1991 版没有 PS 与变比字段，无论哪种情况都无法做这一步换算，
        结果保持在 a/b 换算后的量级（通常是二次值）。这一点必须让下游知道，
        否则算法模块会把二次值当成一次值去和额定值比较。
    """
    if not options.apply_ratio_conversion:
        return

    if ch.ps is None:
        # 1991 版：无变比信息。只在通道确实有单位时提示一次，
        # 避免无意义地刷屏（无单位通道本身就不可用于定量比较）。
        if ch.unit_raw:
            diagnostics.info(
                Code.CHN_RATIO_MISSING,
                f"通道「{ch.name}」所在文件为 1991 版，没有一次/二次变比字段，"
                "数值为 a/b 换算结果（通常为二次值），未做一次值换算",
                location=location,
            )
        return

    if ch.ps != "S":
        return

    if not ch.primary or not ch.secondary or ch.secondary == 0:
        diagnostics.warn(
            Code.CHN_RATIO_MISSING,
            f"通道「{ch.name}」标记为二次值（PS=S）但缺少有效变比"
            f"（primary={ch.primary}, secondary={ch.secondary}），未做一次值换算；"
            "该通道的绝对值不可与一次值通道直接比较",
            location=location,
        )
        return

    values *= ch.primary / ch.secondary
    ch.ratio_applied = True


__all__ = ["missing_sentinel", "convert_analog_channels", "MAX_MISSING_FRACTION"]
