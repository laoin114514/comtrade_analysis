"""COMTRADE 数据文件（.dat）解析。

合同依据
    第一条 2.(1)① "支持 .dat 数据文件读取……建立统一内部数据模型"
    第一条 2.(1)② "录波数据由原始文件到系统内部分析数据的自动转换"
对应需求
    P-02（数据文件读取）、P-03（模拟量通道解析）、P-05（时间轴生成）、
    P-07（异常解析处理）、NF-31（1991/1999/2013 兼容）

支持四种编码
    ==========  ==================  ==========================================
    类型        模拟量元素           说明
    ==========  ==================  ==========================================
    ASCII       文本                一行一个采样点，开关量不打包
    BINARY      2 字节有符号整数     现场最常见
    BINARY32    4 字节有符号整数     2013 版新增
    FLOAT32     4 字节 IEEE 754     2013 版新增，通常已是工程量
    ==========  ==================  ==========================================

记录布局（所有二进制类型一致，小端）
    ==========  ======  ==================================================
    字段        长度    说明
    ==========  ======  ==================================================
    采样号      4 字节   uint32
    采样时标    4 字节   uint32，单位由 time_mult 与 time_base_seconds 决定
    模拟量      A×S     S = 2（BINARY）或 4（BINARY32/FLOAT32）
    开关量      W×2     W = ceil(D/16)，每 16 个通道打包进一个 uint16
    ==========  ======  ==================================================

    开关量位序：每 16 个通道一组，**最低位（bit 0）对应该组第 1 个通道**。
    这是最容易出错的地方 —— 开关量不是一通道一字节。

字节序
    标准规定低字节在前（little-endian）。本模块固定按小端解析，
    解析后由 dat_parser 做取值范围校验 —— 字节序判断错误时，
    解出的值会大幅超出 cfg 声明的 min/max，能立刻被发现。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .diagnostics import Code, DiagnosticCollector, ParseAbort
from .encoding import read_text
from .models import ComtradeVersion, DataFileType
from .options import ParseOptions, DEFAULT_OPTIONS

#: 二进制模拟量元素的小端 dtype
_BINARY_DTYPE = {
    DataFileType.BINARY: "<i2",
    DataFileType.BINARY32: "<i4",
    DataFileType.FLOAT32: "<f4",
}


@dataclass(slots=True)
class ParsedDat:
    """dat 解析产物（尚未做 a/b 换算）。"""

    sample_numbers: np.ndarray
    """采样号，形状 ``(N,)``。"""

    timestamps: np.ndarray
    """采样时标，形状 ``(N,)``。单位见 ``Metadata.time_mult`` / ``time_base_seconds``。"""

    analog_raw: np.ndarray | None
    """原始模拟量矩阵，形状 ``(模拟量通道数, N)``。

    通道为主序（channel-major）—— 下游按通道做有效值、序分量计算时，
    同一通道的数据连续存放，缓存命中率显著更高。
    """

    digital: np.ndarray | None
    """开关量矩阵，``bool``，形状 ``(开关量通道数, N)``。True 表示动作/闭合。"""

    record_count: int = 0
    declared_count: int = 0
    digital_word_count: int = 0


def record_size(analog_count: int, digital_count: int, data_type: DataFileType) -> int:
    """计算二进制记录的字节长度。ASCII 类型返回 0。"""
    if not data_type.is_binary:
        return 0
    words = -(-digital_count // 16)  # ceil(D/16)
    return 8 + analog_count * data_type.value_size + words * 2


def parse_dat(
    dat_path: Path,
    *,
    analog_count: int,
    digital_count: int,
    data_type: DataFileType,
    declared_count: int,
    version: ComtradeVersion,
    diagnostics: DiagnosticCollector,
    options: ParseOptions = DEFAULT_OPTIONS,
) -> ParsedDat:
    """解析 .dat 文件，返回原始采样矩阵（未做 a/b 换算）。

    Raises:
        ParseAbort: 数据文件为空或完全无法解析。
    """
    if not dat_path.exists():
        diagnostics.fatal(
            Code.FILE_NOT_FOUND, f"数据文件不存在：{dat_path}", location=dat_path.name
        )
        raise ParseAbort("dat 不存在")

    size = dat_path.stat().st_size
    if size == 0:
        diagnostics.fatal(Code.DAT_EMPTY, "数据文件为空", location=dat_path.name)
        raise ParseAbort("dat 为空")

    if data_type is DataFileType.ASCII:
        return _parse_ascii(dat_path, analog_count, digital_count, declared_count, diagnostics)

    return _parse_binary(
        dat_path,
        analog_count,
        digital_count,
        data_type,
        declared_count,
        diagnostics,
    )


# ---------------------------------------------------------------------------
# ASCII
# ---------------------------------------------------------------------------

def _parse_ascii(
    dat_path: Path,
    analog_count: int,
    digital_count: int,
    declared_count: int,
    diagnostics: DiagnosticCollector,
) -> ParsedDat:
    """解析 ASCII 数据文件。"""
    expected_columns = 2 + analog_count + digital_count

    # 快路径：交给 numpy 的 C 解析器；只有它失败时才走逐行容错解析
    data = _try_fast_load(dat_path, diagnostics)

    if data is not None and data.ndim == 2 and data.shape[1] == expected_columns:
        # 采样号/采样时标含 nan/inf 字面量时不能让快路径通过：这两列会被
        # astype(int64) 静默转成 INT64_MIN，且快路径无法定位到具体行。
        # 交给容错解析逐行处理并报告（见 _parse_ascii_tolerant）。
        if np.isfinite(data[:, :2]).all():
            diagnostics.info(
                Code.DAT_OK,
                f"ASCII 数据解析完成：{data.shape[0]} 个采样点 × {data.shape[1]} 列",
                location=dat_path.name,
            )
            return _assemble_ascii(
                data,
                analog_count,
                digital_count,
                declared_count,
                diagnostics,
                location=dat_path.name,
            )

    # 列数与声明不符时不在这里单独告警：loadtxt 能读出结果就说明**所有行**列数一致，
    # 那样容错解析一行也匹配不上、直接致命中断 —— 结论统一由容错解析给出。
    # 否则使用者会先看到"列数不符"的警告、再看到"没有可解析的采样行"的致命错误，
    # 真正的原因被埋在两条互相矛盾的信息后面（回归案例见
    # test_ascii_all_rows_column_mismatch_fatal_names_root_cause）。
    return _parse_ascii_tolerant(
        dat_path, analog_count, digital_count, declared_count, expected_columns, diagnostics
    )


def _try_fast_load(dat_path: Path, diagnostics: DiagnosticCollector) -> np.ndarray | None:
    """尝试用 numpy 快速加载；失败返回 None（由容错解析接管）。"""
    try:
        head = dat_path.open("rb").read(8192).decode("ascii", errors="ignore")
    except OSError:
        return None
    delimiter = "," if "," in head else None  # 有些厂家用空白分隔
    try:
        data = np.loadtxt(dat_path, delimiter=delimiter, ndmin=2)
    except (ValueError, OSError):
        return None
    if data.size == 0:
        return None
    return data


def _parse_ascii_tolerant(
    dat_path: Path,
    analog_count: int,
    digital_count: int,
    declared_count: int,
    expected_columns: int,
    diagnostics: DiagnosticCollector,
) -> ParsedDat:
    """逐行容错解析。跳过无法解析的行并记录诊断，保证"坏文件不崩溃"（NF-12）。

    列数不符分两种结局，必须给出不同的结论：

    * **部分行不符** —— 跳过坏行后仍能产出数据，记 DAT-007 警告继续（容错的本意）；
    * **所有行都不符** —— 一行都产不出，此时根因是 cfg 声明与 dat 实际列数不一致，
      致命诊断必须直接说明这一点，而不是含糊地报"没有可解析的采样行"。
    """
    text = read_text(dat_path, diagnostics)
    rows: list[list[float]] = []
    bad_rows = 0
    column_mismatch = 0
    key_missing = 0
    widths: dict[int, int] = {}
    first_bad_line: int | None = None
    first_bad_detail: str | None = None
    first_key_line: int | None = None
    first_key_detail: str | None = None

    # 分隔符按首个非空行判定，避免逐行反复探测
    delim = ","
    for probe in text.splitlines():
        if probe.strip():
            delim = "," if "," in probe else None
            break

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(delim)] if delim else line.split()
        widths[len(parts)] = widths.get(len(parts), 0) + 1
        if len(parts) != expected_columns:
            column_mismatch += 1
            if first_bad_line is None:
                first_bad_line, first_bad_detail = line_no, line[:120]
            continue
        try:
            # 空字段是 1991 版 ASCII 表示"该点未采到"的方式，转为 NaN 而非丢弃整行
            row = [float(p) if p else math.nan for p in parts]
        except ValueError:
            bad_rows += 1
            if first_bad_line is None:
                first_bad_line, first_bad_detail = line_no, line[:120]
            continue

        # 采样号（第 0 列）与采样时标（第 1 列）必须能定位成一个整数时刻，
        # 不能像模拟量那样用 NaN 表示"没采到"：NaN 经 astype(int64) 会**静默**
        # 变成 -9223372036854775808（实测），采样号随之失去意义，下游按采样号
        # 索引会算出无意义结果；空值落在首行还会让记录数校验算出天文数字、
        # 误报"文件被截断"。因此这里显式判定并跳过该行，让问题以诊断的形式出现。
        # 除空字段外，"nan"/"inf" 这类字面量同样落在这里（float() 能解析但不有限）。
        if not (math.isfinite(row[0]) and math.isfinite(row[1])):
            key_missing += 1
            if first_key_line is None:
                first_key_line, first_key_detail = line_no, line[:120]
            continue

        rows.append(row)

    if not rows:
        # 先判根因：列数问题与"内容本身不可解析"要给出不同的致命原因
        if column_mismatch:
            diagnostics.fatal(
                Code.DAT_COLUMN_MISMATCH,
                f"数据文件的列数为 {_describe_widths(widths)}，与 cfg 声明的 "
                f"{expected_columns} 列（2 + {analog_count} 个模拟量 + "
                f"{digital_count} 个开关量）不符，没有任何一行的列数与声明一致。"
                "多出或缺失的是哪个通道无法确定，因此不按猜测的通道映射解析。"
                "请核对 cfg 声明的通道数量是否与 dat 一致，或确认 dat 是否被截断",
                location=f"{dat_path.name}:{first_bad_line}" if first_bad_line else dat_path.name,
                detail=first_bad_detail or "",
            )
            raise ParseAbort("dat 列数与声明不符")

        if key_missing:
            diagnostics.fatal(
                Code.DAT_KEY_FIELD_MISSING,
                f"全部 {key_missing} 行的采样号或采样时标为空/非法，没有任何采样点能定位，"
                "无法产出数据。请检查该列是否被误删或被别的字段占用",
                location=f"{dat_path.name}:{first_key_line}",
                detail=first_key_detail or "",
            )
            raise ParseAbort("采样号/采样时标全部缺失")

        # 列数没问题，那是内容本身转不成数值。把行数写进文案 —— 否则
        # "没有可解析的采样行"会让人以为文件是空的，而它可能有一堆内容，
        # 只是没有一行能解析。
        reason = f"（{bad_rows} 行内容无法转换为数值）" if bad_rows else "（文件中没有数据行）"
        diagnostics.fatal(
            Code.DAT_EMPTY,
            f"数据文件中没有可解析的采样行{reason}",
            location=dat_path.name,
            detail=first_bad_detail or "",
        )
        raise ParseAbort("dat 无可解析内容")

    if column_mismatch:
        diagnostics.warn(
            Code.DAT_COLUMN_MISMATCH,
            f"{column_mismatch} 行的字段数与声明不符（期望 {expected_columns} 列），已跳过",
            location=f"{dat_path.name}:{first_bad_line}",
            detail=first_bad_detail or "",
        )
    if key_missing:
        diagnostics.warn(
            Code.DAT_KEY_FIELD_MISSING,
            f"{key_missing} 行的采样号或采样时标为空/非法，这些采样点无法定位，已跳过"
            "（其余采样点不受影响；若记录数与 cfg 声明不符，原因就在这里）",
            location=f"{dat_path.name}:{first_key_line}",
            detail=first_key_detail or "",
        )
    if bad_rows:
        diagnostics.warn(
            Code.DAT_VALUE_UNPARSABLE,
            f"{bad_rows} 行含无法转换为数值的内容，已跳过",
            location=f"{dat_path.name}:{first_bad_line}",
            detail=first_bad_detail or "",
        )

    data = np.asarray(rows, dtype=np.float64)
    return _assemble_ascii(
        data,
        analog_count,
        digital_count,
        declared_count,
        diagnostics,
        location=dat_path.name,
    )


def _assemble_ascii(
    data: np.ndarray,
    analog_count: int,
    digital_count: int,
    declared_count: int,
    diagnostics: DiagnosticCollector,
    *,
    location: str = "",
) -> ParsedDat:
    """把 ASCII 二维数组拆成采样号、时标、模拟量矩阵、开关量矩阵。"""
    n = data.shape[0]
    columns = data.shape[1]

    # 采样号/采样时标必须能转成整数。astype 对非有限值是**静默转换**：
    # NaN → INT64_MIN（-9223372036854775808），采样号就此失去意义且无任何提示，
    # 下游按采样号索引会算出无意义结果。上游两条路径都已拦掉这类行
    # （快路径让位、容错解析逐行跳过），这里再守一道：将来任何新路径往这两列
    # 塞进非有限值，都会立刻以"不应出现的缺陷"形式暴露，而不是变成垃圾值流向下游。
    keys = data[:, :2]
    if not np.isfinite(keys).all():
        bad = np.flatnonzero(~np.isfinite(keys).all(axis=1))
        diagnostics.fatal(
            Code.SYS_UNEXPECTED_ERROR,
            f"有 {bad.size} 行的采样号或采样时标不是有效数值（首个在第 {int(bad[0]) + 1} 行），"
            "这些采样点的位置无法确定。这属于解析实现缺陷，请保留该文件反馈",
            location=location,
        )
        raise ParseAbort("采样号/采样时标非有限值")

    sample_numbers = data[:, 0].astype(np.int64)
    _check_record_count(
        n, declared_count, int(sample_numbers[0]) if n else None, diagnostics, location
    )
    timestamps = data[:, 1].astype(np.int64)

    a_hi = min(2 + analog_count, columns)
    analog = data[:, 2 : a_hi].T.copy() if a_hi > 2 else None
    if analog is not None and analog.shape[0] < analog_count:
        # 列数不足时补 NaN，保证下游拿到的通道数与 cfg 声明一致
        pad = np.full((analog_count - analog.shape[0], n), np.nan)
        analog = np.vstack([analog, pad])

    d_lo = 2 + analog_count
    digital = None
    if digital_count > 0 and columns > d_lo:
        d_hi = min(d_lo + digital_count, columns)
        digital = (data[:, d_lo:d_hi] != 0).T.copy()
        if digital.shape[0] < digital_count:
            pad = np.zeros((digital_count - digital.shape[0], n), dtype=bool)
            digital = np.vstack([digital, pad])

    return ParsedDat(
        sample_numbers=sample_numbers,
        timestamps=timestamps,
        analog_raw=analog,
        digital=digital,
        record_count=n,
        declared_count=declared_count,
        digital_word_count=0,
    )


# ---------------------------------------------------------------------------
# BINARY / BINARY32 / FLOAT32
# ---------------------------------------------------------------------------

def _parse_binary(
    dat_path: Path,
    analog_count: int,
    digital_count: int,
    data_type: DataFileType,
    declared_count: int,
    diagnostics: DiagnosticCollector,
) -> ParsedDat:
    """解析二进制数据文件。

    用 numpy 结构化 dtype 一次性映射整条记录，避免逐字段手工切片 ——
    既快，也让"记录布局"这件事集中在一处定义，不易与文档脱节。
    """
    words = -(-digital_count // 16)
    fields: list[tuple] = [("samp", "<u4"), ("ts", "<u4")]
    if analog_count > 0:
        fields.append(("analog", _BINARY_DTYPE[data_type], (analog_count,)))
    if words > 0:
        fields.append(("digital", "<u2", (words,)))

    rec_dtype = np.dtype(fields)
    rec_size = rec_dtype.itemsize

    file_size = dat_path.stat().st_size
    full_records = file_size // rec_size
    remainder = file_size - full_records * rec_size

    if remainder:
        diagnostics.info(
            Code.DAT_TRAILING_BYTES,
            f"文件末尾有 {remainder} 字节残余（不足一条 {rec_size} 字节的记录），已忽略；"
            "通常是无意义的换行或填充字符",
            location=dat_path.name,
        )

    if full_records == 0:
        diagnostics.fatal(
            Code.DAT_EMPTY,
            f"数据文件不足以构成一条完整记录（{file_size} 字节 < {rec_size} 字节）",
            location=dat_path.name,
        )
        raise ParseAbort("dat 记录不完整")

    # 记录数交叉校验推迟到读完采样号之后（见 _check_record_count）：
    # cfg 的 endsamp 是**采样号**不是条数，必须减去首采样号才能与条数比较。
    # 实测有 0 起编号的现场文件（wisp_example2：采样号 0~19679、endsamp=19679），
    # 不减去首采样号会误报一条"记录数不符"。

    try:
        arr = np.fromfile(dat_path, dtype=rec_dtype, count=full_records)
    except (OSError, ValueError) as exc:
        diagnostics.fatal(
            Code.FILE_UNREADABLE, f"数据文件读取失败：{exc}", location=dat_path.name
        )
        raise ParseAbort("dat 读取失败") from exc

    if arr.size != full_records:
        diagnostics.warn(
            Code.DAT_RECORD_SHORT,
            f"实际读取到 {arr.size} 条记录，少于按文件大小推算的 {full_records} 条",
            location=dat_path.name,
        )

    # ------------------------------------------------------------ 通道为主序
    analog = None
    if analog_count > 0:
        analog = np.ascontiguousarray(arr["analog"].T)

    digital = None
    if words > 0 and digital_count > 0:
        digital = _unpack_digital(arr["digital"], digital_count, diagnostics, dat_path.name)

    if arr.size:
        _check_record_count(
            int(arr.size), declared_count, int(arr["samp"][0]), diagnostics, dat_path.name
        )

    diagnostics.info(
        Code.DAT_OK,
        f"{data_type.value} 数据解析完成：{arr.size} 个采样点，记录长度 {rec_size} 字节",
        location=dat_path.name,
    )

    return ParsedDat(
        sample_numbers=arr["samp"].astype(np.int64),
        timestamps=arr["ts"].astype(np.int64),
        analog_raw=analog,
        digital=digital,
        record_count=int(arr.size),
        declared_count=declared_count,
        digital_word_count=words,
    )


def declared_expected_count(declared_end_sample: int, first_sample_number: int | None) -> int:
    """把 cfg 的"结束采样号"换算成期望的记录**条数**。

    ``endsamp`` 是采样号（含），不是条数：标准写法采样号从 1 起，
    条数 = endsamp；实测也有从 0 起的（wisp_example2），条数 = endsamp + 1。
    统一按 ``endsamp - 首采样号 + 1`` 计算。
    """
    base = 1 if first_sample_number is None else int(first_sample_number)
    return int(declared_end_sample) - base + 1


#: 记录条数的容差。真实文件在"endsamp 是末采样号还是总条数"上并不统一，
#: 有的装置就是差一条（实测 wisp_example4：endsamp=2399 但有 2400 条；
#: wisp_example5：0 起编号、endsamp=202 恰为条数）。这条校验的价值在于发现
#: **量级性**的错误（通道数解析错、版本判错、文件被截断），±1 条没有意义。
_RECORD_COUNT_TOLERANCE = 1


def _describe_widths(widths: dict[int, int]) -> str:
    """把观察到的列数说成人话，供"列数不符"的致命诊断定位根因。

    列数一致时报具体值（最常见的情况：cfg 多声明或少声明了通道）；
    不一致时报范围 —— 那说明文件本身是拼凑或半截的，与通道数声明无关。
    """
    if not widths:
        return "0"
    if len(widths) == 1:
        width, count = next(iter(widths.items()))
        return f"{width}（{count} 行）"
    return (
        f"{min(widths)}~{max(widths)}"
        f"（{len(widths)} 种，共 {sum(widths.values())} 行）"
    )


def _check_record_count(
    actual: int,
    declared_end_sample: int,
    first_sample_number: int | None,
    diagnostics: DiagnosticCollector,
    location: str,
) -> None:
    """记录条数与 cfg 声明的一致性校验。

    这是**最有价值的诊断之一**：一旦出现量级性偏差，通常说明通道数量解析错误、
    版本判定错误或文件被截断。±1 条以内的偏差按容差忽略（见上文说明）。
    """
    if not declared_end_sample:
        return
    expected = declared_expected_count(declared_end_sample, first_sample_number)
    diff = actual - expected
    if abs(diff) <= _RECORD_COUNT_TOLERANCE:
        return
    diagnostics.warn(
        Code.DAT_SIZE_MISMATCH,
        f"实际记录数 {actual} 与 cfg 声明推算的 {expected} 不一致（相差 {diff:+d} 条；"
        f"cfg 结束采样号 {declared_end_sample}、首采样号 {first_sample_number}）。"
        "请检查通道数量、版本判定或文件是否损坏",
        location=location,
    )
    if diff < 0:
        diagnostics.warn(
            Code.DAT_RECORD_SHORT,
            "实际记录数明显少于声明值，数据可能被截断",
            location=location,
        )


def _unpack_digital(
    words: np.ndarray,
    digital_count: int,
    diagnostics: DiagnosticCollector,
    location: str,
) -> np.ndarray:
    """把打包的 16 位字拆成逐通道的布尔矩阵。

    位序：每 16 个通道一组，最低位对应该组第 1 个通道。
    实现上先按字节展开（小端下字节 0 即最低 8 位），再用 ``unpackbits``
    的 ``little`` 位序取位，最后截取前 ``digital_count`` 位。
    """
    n, nwords = words.shape
    contiguous = np.ascontiguousarray(words)
    as_bytes = contiguous.view(np.uint8).reshape(n, nwords * 2)
    # bitorder='little' → 每个字节内 bit0 在前，与 COMTRADE 的位序一致
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")
    out = bits[:, :digital_count].astype(bool).T.copy()

    unused = nwords * 16 - digital_count
    if unused:
        # 标准要求未使用的位为 0；若非 0 说明位序或通道数理解有误
        padding = bits[:, digital_count:]
        if padding.any():
            diagnostics.info(
                Code.DAT_DIGITAL_PADDING,
                f"开关量打包的 {unused} 个未使用位中存在非 0 值，"
                "通常无害，但若通道状态异常请检查开关量通道数是否正确",
                location=location,
            )
    return out


__all__ = ["ParsedDat", "parse_dat", "record_size"]
