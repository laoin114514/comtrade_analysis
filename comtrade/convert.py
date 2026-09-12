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

关于 ASCII 是否已换算的二义性
    标准与全部主流实现（python-comtrade / comtrade-rs / GSF / pycomtrade）
    都对 ASCII 的模拟量施加 a/b 换算。现场确实也存在直接写工程量的 ASCII 文件，
    但**两者无法可靠区分** —— 真实文件里"min/max 声明满量程、实际原始计数很小"
    是正常现象（SEL 等装置的 ASCII 样例即如此）。

    本模块早期版本曾用"数据跨度 vs 声明量程跨度"自动判定并跳过换算，
    在 11 组真实公开样例（186 个通道）上实测误判 10 个通道、且全部错向"跳过"。
    因此现在：**默认按标准换算**，跨度判定降级为纯提示（DAT-009），
    需要跳过的场景由使用者显式设置 ``ascii_scaling="never"``。

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

#: 绝对数量下限：样本量不足时，出现次数不超过该值即按缺失值屏蔽。
#: 而在正常规模的录波里，一个值只出现几次，也不可能是"正常数据值"。
MIN_MISSING_ABSOLUTE = 8

#: 占比守卫的启用门槛（样本点数）。
#: 样本量低于该值时占比没有统计意义（3 个点里有 1 个哨兵就是 33%），
#: 此时只按 ``MIN_MISSING_ABSOLUTE`` 判断；达到门槛后占比守卫无条件生效。
#:
#: 早先的实现把两条判据写成"或"（次数少 **或** 占比低即屏蔽），
#: 等于让绝对次数规则覆盖了占比守卫 —— 16~39 点的短录波里，
#: 即使一半数值都等于哨兵（很像真实数据）也会被整片抹成 NaN。
MIN_SAMPLES_FOR_FRACTION = 16

#: 判定依据标签
_R_A_ZERO = "a_zero"
_R_SCALED = "scaled"
_R_IDENTITY = "identity"
_R_NEVER = "never"


def looks_like_missing_value(count: int, fraction: float, size: int) -> bool:
    """判断"等于候选哨兵"的采样点该按缺失值屏蔽，还是视为正常数据。

    样本量足够时以占比为准；样本量不足时占比没有统计意义，只看出现次数。
    两条判据的适用边界是显式的 —— 见 ``MIN_SAMPLES_FOR_FRACTION`` 的说明。
    """
    if size >= MIN_SAMPLES_FOR_FRACTION:
        return fraction <= MAX_MISSING_FRACTION
    return count <= MIN_MISSING_ABSOLUTE


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
    never_skipped: list[str] = []
    float32_nonidentity: list[str] = []
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
                if looks_like_missing_value(count, fraction, raw.size):
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
        apply_scale, reason = _decide_scaling(ch, options)
        if reason == _R_A_ZERO:
            scale_skipped.append(ch.name)
        elif reason == _R_NEVER:
            never_skipped.append(ch.name)
        # 提示：只在"确实施加了换算"的前提下，提示"若文件其实已换算则应改用 never"
        if apply_scale and _prescaled_suspicion(ch, raw, data_type, options):
            prescaled.append(ch.name)
        if (
            data_type is DataFileType.FLOAT32
            and apply_scale
            and reason == _R_SCALED
        ):
            float32_nonidentity.append(ch.name)

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
            prescaled, scale_skipped, never_skipped, float32_nonidentity)


def _report(
    diagnostics: DiagnosticCollector,
    location: str,
    missing_hits: list[tuple[str, int]],
    out_of_range: list[tuple[str, int]],
    prescaled: list[str],
    scale_skipped: list[str],
    never_skipped: list[str],
    float32_nonidentity: list[str],
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
            f"有 {len(prescaled)} 个通道的数据跨度远小于 cfg 声明的量程跨度："
            + "、".join(prescaled[:8])
            + ("…" if len(prescaled) > 8 else "")
            + "。已按标准施加 a/b 换算；"
            "若这些通道存的确实已经是工程量，请将 ascii_scaling 设为 never",
            location=location,
        )

    if float32_nonidentity:
        diagnostics.info(
            Code.DAT_FLOAT32_SCALED,
            f"FLOAT32 编码的 {len(float32_nonidentity)} 个通道带有非单位比例系数并已施加换算："
            + "、".join(float32_nonidentity[:8])
            + "。FLOAT32 通常直接存工程量（a=1, b=0），若数值异常请核对 cfg",
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

    if never_skipped:
        # 使用者显式要求跳过，属预期行为，给 INFO —— 但必须能看见是哪几个通道，
        # 否则"选项有没有生效"无从确认（DAT-009 的提示语在这条路径上不适用）。
        diagnostics.info(
            Code.DAT_SCALING_DISABLED,
            f"按 ascii_scaling=never 跳过 a/b 换算的 {len(never_skipped)} 个通道"
            "（数值为文件中的原始值，未经比例系数换算）："
            + "、".join(never_skipped[:8])
            + ("…" if len(never_skipped) > 8 else ""),
            location=location,
        )


def _decide_scaling(
    ch: AnalogChannel,
    options: ParseOptions,
) -> tuple[bool, str | None]:
    """判定是否对某通道施加 a/b 换算。

    **只做两件事：a 为 0 时跳过（无法换算），以及 NEVER 策略下跳过。**
    其余情况一律按标准施加换算 —— 真实文件里"原始计数很小"是正常现象，
    不能据此判定数据已经是工程量（11 组真实样例实测该判定 100% 误判）。
    """
    if ch.a == 0.0:
        return False, _R_A_ZERO
    # 这里用身份比较是安全的：ParseOptions.__post_init__ 已把字符串取值
    # 归一化成枚举成员。若绕过该入口直接构造对象，务必先归一化 ——
    # AsciiScaling 是 str 混入枚举，字符串与原成员 `is` 不相等。
    if options.ascii_scaling is AsciiScaling.NEVER:
        return False, _R_NEVER
    if ch.a == 1.0 and ch.b == 0.0:
        return True, _R_IDENTITY
    return True, _R_SCALED


def _prescaled_suspicion(
    ch: AnalogChannel,
    raw: np.ndarray,
    data_type: DataFileType,
    options: ParseOptions,
) -> bool:
    """数据跨度远小于声明量程跨度 —— 仅为提示，不改变解析行为。

    只对 ASCII 判定：FLOAT32 的数值本来就是工程量，跨度小属正常，
    由 DAT-011 单独提示。
    """
    if data_type is not DataFileType.ASCII:
        return False
    if ch.a in (0.0, 1.0):
        return False
    raw_span = ch.raw_max - ch.raw_min
    if raw_span <= 0:
        return False
    finite = raw[np.isfinite(raw)]
    if finite.size == 0:
        return False
    data_span = float(finite.max() - finite.min())
    if data_span <= 0:
        return False
    return (data_span / raw_span) < options.prescaled_span_ratio


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


__all__ = [
    "missing_sentinel",
    "looks_like_missing_value",
    "convert_analog_channels",
    "MAX_MISSING_FRACTION",
    "MIN_MISSING_ABSOLUTE",
    "MIN_SAMPLES_FOR_FRACTION",
]
