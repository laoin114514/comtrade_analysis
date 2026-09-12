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
        diagnostics.info(
            "DAT-OK",
            f"ASCII 数据解析完成：{data.shape[0]} 个采样点 × {data.shape[1]} 列",
            location=dat_path.name,
        )
        return _assemble_ascii(data, analog_count, digital_count, declared_count, diagnostics)

    if data is not None and data.ndim == 2 and data.shape[1] != expected_columns:
        diagnostics.warn(
            Code.DAT_COLUMN_MISMATCH,
            f"ASCII 数据列数为 {data.shape[1]}，与 cfg 声明不符"
            f"（期望 2 + {analog_count} 模拟 + {digital_count} 开关 = {expected_columns}）",
            location=dat_path.name,
        )

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
    """逐行容错解析。跳过无法解析的行并记录诊断，保证"坏文件不崩溃"（NF-12）。"""
    text = read_text(dat_path, diagnostics)
    rows: list[list[float]] = []
    bad_rows = 0
    column_mismatch = 0
    first_bad_line: int | None = None
    first_bad_detail: str | None = None

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
        if len(parts) != expected_columns:
            column_mismatch += 1
            if first_bad_line is None:
                first_bad_line, first_bad_detail = line_no, line[:120]
            continue
        try:
            # 空字段是 1991 版 ASCII 表示"该点未采到"的方式，转为 NaN 而非丢弃整行
            rows.append([float(p) if p else math.nan for p in parts])
        except ValueError:
            bad_rows += 1
            if first_bad_line is None:
                first_bad_line, first_bad_detail = line_no, line[:120]

    if column_mismatch:
        diagnostics.warn(
            Code.DAT_COLUMN_MISMATCH,
            f"{column_mismatch} 行的字段数与声明不符（期望 {expected_columns} 列），已跳过",
            location=f"{dat_path.name}:{first_bad_line}",
            detail=first_bad_detail or "",
        )
    if bad_rows:
        diagnostics.warn(
            Code.DAT_VALUE_UNPARSABLE,
            f"{bad_rows} 行含无法转换为数值的内容，已跳过",
            location=f"{dat_path.name}:{first_bad_line}",
            detail=first_bad_detail or "",
        )

    if not rows:
        diagnostics.fatal(
            Code.DAT_EMPTY,
            "数据文件中没有可解析的采样行",
            location=dat_path.name,
        )
        raise ParseAbort("dat 无可解析内容")

    data = np.asarray(rows, dtype=np.float64)
    return _assemble_ascii(data, analog_count, digital_count, declared_count, diagnostics)


def _assemble_ascii(
    data: np.ndarray,
    analog_count: int,
    digital_count: int,
    declared_count: int,
    diagnostics: DiagnosticCollector,
) -> ParsedDat:
    """把 ASCII 二维数组拆成采样号、时标、模拟量矩阵、开关量矩阵。"""
    n = data.shape[0]
    columns = data.shape[1]

    if n < declared_count:
        diagnostics.warn(
            Code.DAT_RECORD_SHORT,
            f"实际采样点数 {n} 少于 cfg 声明的 {declared_count}，数据可能被截断",
            location="",
        )

    sample_numbers = data[:, 0].astype(np.int64)
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

    # ------------------------------------------------------ 记录数交叉校验
    if declared_count and full_records != declared_count:
        diff = full_records - declared_count
        diagnostics.warn(
            Code.DAT_SIZE_MISMATCH,
            f"按记录长度推算的采样点数 {full_records} 与 cfg 声明的 {declared_count} "
            f"不一致（相差 {diff:+d}）。请检查通道数量、版本判定或文件是否损坏",
            location=dat_path.name,
            detail=f"文件 {file_size} 字节 / 记录 {rec_size} 字节",
        )
        if diff < 0:
            diagnostics.warn(
                Code.DAT_RECORD_SHORT,
                "实际记录数少于声明值，数据可能被截断",
                location=dat_path.name,
            )

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

    diagnostics.info(
        "DAT-OK",
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
                "DAT-DIG-PAD",
                f"开关量打包的 {unused} 个未使用位中存在非 0 值，"
                "通常无害，但若通道状态异常请检查开关量通道数是否正确",
                location=location,
            )
    return out


__all__ = ["ParsedDat", "parse_dat", "record_size"]
