"""COMTRADE 配置文件（.cfg）解析。

合同依据
    第一条 2.(1)① "支持 .cfg 配置文件解析……自动识别录波通道信息、
                     获取采样频率及时间轴信息……提供异常文件检测及错误提示机制"
    第一条 2.(1)① "COMTRADE（符合 IEEE C37.111 标准的 1991、1999、2013 版本）"
对应需求
    P-01（配置文件解析）、P-03（模拟量通道解析）、P-04（采样频率识别）、
    P-07（异常解析处理）、NF-31（标准 COMTRADE 格式兼容）

解析策略：一次解析，版本自适应
    cfg 的行数**不是固定的** —— 第 2 行声明了通道数量，通道定义行数随之变化，
    其后的采样率、时间、数据类型的行号全部浮动。因此必须从第 1 行起顺序推进游标，
    不能按固定行号取值。

    版本判定的唯一可靠依据是**首行字段个数**：
    2 个字段 → 1991 版；3 个字段 → 第 3 个是版本年份。
    此后各行的字段个数差异（模拟通道 10 vs 13、开关量 3 vs 5）用于交叉校验。
    本解析器对两种字段数都接受 —— 现场有版本年份写错、但字段数正确的情况，
    不能因为元数据写错就判定文件非法。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .diagnostics import Code, DiagnosticCollector, ParseAbort
from .encoding import read_text
from .models import (
    AnalogChannel,
    ComtradeVersion,
    DataFileType,
    DigitalChannel,
    Metadata,
    SampleRateSegment,
)
from .options import ParseOptions, DEFAULT_OPTIONS
from .units import parse_unit


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

_COUNT_RE = re.compile(r"^\s*[AaDd]?\s*(\d+)\s*[AaDd]?\s*$")
_INT_RE = re.compile(r"^\s*[+-]?\d+\s*$")


def _split(line: str) -> list[str]:
    """按逗号切分并去除字段两端空白。

    保留字段**内部**的空格 —— 通道名 ``"I a"`` 是合法的，
    不能整行 strip 后再按空白处理。
    """
    return [part.strip() for part in line.split(",")]


def _as_float(text: str, default: float | None = None) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _as_int(text: str, default: int | None = None) -> int | None:
    t = (text or "").strip()
    if _INT_RE.match(t):
        return int(t)
    f = _as_float(t)
    return int(f) if f is not None else default


def _parse_channel_count(text: str) -> int | None:
    """解析形如 ``6A`` / ``4D`` / ``6`` / ``A6`` 的通道数量字段。"""
    m = _COUNT_RE.match(text or "")
    return int(m.group(1)) if m else None


class _LineCursor:
    """按行推进的游标，自动跳过空行并保留原始行号用于诊断定位。"""

    def __init__(self, text: str, filename: str, diagnostics: DiagnosticCollector) -> None:
        self._lines: list[tuple[int, str]] = []
        for i, raw in enumerate(text.splitlines(), start=1):
            stripped = raw.strip()
            if stripped:
                self._lines.append((i, stripped))
        self._pos = 0
        self._filename = filename
        self._diag = diagnostics
        self.skipped_blank = 0

    @property
    def filename(self) -> str:
        return self._filename

    def location(self, line_no: int | None = None) -> str:
        return f"{self._filename}:{line_no or (self._lines[self._pos - 1][0] if self._pos else 0)}"

    def next(self, what: str) -> tuple[int, str]:
        """取下一行，取不到时记录致命诊断并中断。"""
        if self._pos >= len(self._lines):
            self._diag.fatal(
                Code.CFG_UNEXPECTED_END,
                f"配置文件提前结束，缺少「{what}」",
                location=self.location(),
            )
            raise ParseAbort(what)
        self._pos += 1
        return self._lines[self._pos - 1]

    def rewind(self, count: int = 1) -> None:
        """回退游标。

        用于"先探查再决定"的场景：例如 2013 版的可选行可能不存在，
        探查后需要把它让给后面的步骤解析。
        """
        self._pos = max(0, self._pos - count)

    def peek_next(self) -> tuple[int, str] | None:
        return self._lines[self._pos] if self._pos < len(self._lines) else None

    def remaining(self) -> list[tuple[int, str]]:
        return self._lines[self._pos :]

    def consume_rest(self) -> None:
        """把游标推到末尾，表示剩余内容已被处理（不再算作"多余内容"）。"""
        self._pos = len(self._lines)

    @property
    def consumed_all(self) -> bool:
        return self._pos >= len(self._lines)

    @property
    def total_lines(self) -> int:
        return len(self._lines)


# ---------------------------------------------------------------------------
# 时间解析
# ---------------------------------------------------------------------------

def parse_datetime(
    raw: str,
    diagnostics: DiagnosticCollector,
    location: str,
) -> tuple[datetime | None, bool]:
    """解析 COMTRADE 的日期时间字段。

    返回 ``(datetime, 是否纳秒精度)``。

    标准格式为 ``日/月/年,时:分:秒.微秒``，但现场存在两种偏差：
    一是写成 ``月/日/年``，二是 2013 版允许 9 位小数（纳秒）。
    两者都会处理并记录诊断。
    """
    text = (raw or "").strip()
    if not text:
        diagnostics.error(Code.CFG_DATETIME_UNPARSABLE, "时间字段为空", location=location)
        return None, False

    date_part, _, time_part = text.partition(",")

    # ---------------------------------------------------------------- 时间部分
    nano = False
    micro = 0
    time_part = time_part.strip()
    if time_part:
        hms, _, frac = time_part.partition(".")
        if frac:
            digits = re.sub(r"\D", "", frac)
            if len(digits) > 6:
                nano = len(digits) >= 9
                if nano:
                    diagnostics.info(
                        Code.CFG_TIME_PRECISION_TRUNCATED,
                        "时间精度为纳秒（9 位小数），已截断至微秒存储；"
                        "如需纳秒级时标请使用 time_mult 与 time_base_seconds",
                        location=location,
                        detail=text,
                    )
                digits = digits[:6]
            micro = int(digits.ljust(6, "0")) if digits else 0
    else:
        hms = ""

    try:
        hh, mm, ss = (int(x) for x in hms.split(":"))
    except (ValueError, TypeError):
        diagnostics.error(
            Code.CFG_DATETIME_UNPARSABLE, "时间部分无法解析", location=location, detail=text
        )
        return None, nano

    # ---------------------------------------------------------------- 日期部分
    parts = [p.strip() for p in date_part.split("/")]
    if len(parts) != 3:
        diagnostics.error(
            Code.CFG_DATETIME_UNPARSABLE, "日期部分字段数不为 3", location=location, detail=text
        )
        return None, nano

    nums: list[int] = []
    for p in parts:
        n = _as_int(p)
        if n is None:
            diagnostics.error(
                Code.CFG_DATETIME_UNPARSABLE, "日期部分含非数字", location=location, detail=text
            )
            return None, nano
        nums.append(n)

    a, b, c = nums
    if len(parts[0].strip()) == 4:
        # 非标准但常见：yyyy/mm/dd
        year, month, day = a, b, c
    elif a > 12 and b <= 12:
        day, month, year = a, b, c  # 标准：dd/mm/yyyy
    elif b > 12 and a <= 12:
        month, day, year = a, b, c  # 非标准：mm/dd/yyyy
    else:
        # 两个数都 ≤12，无法判定；按标准取 dd/mm，但必须让用户知道
        day, month, year = a, b, c
        diagnostics.warn(
            Code.CFG_DATE_AMBIGUOUS,
            f"日期「{date_part}」无法判定是 日/月 还是 月/日，已按标准取 日/月；"
            "若结果有误请检查录波器设置",
            location=location,
            detail=text,
        )

    if year < 100:
        year += 2000 if year < 70 else 1900
        diagnostics.info(
            Code.CFG_DATE_AMBIGUOUS, f"年份为两位数字，已按 {year} 解析", location=location
        )

    try:
        return datetime(year, month, day, hh, mm, ss, micro), nano
    except ValueError as exc:
        diagnostics.error(
            Code.CFG_DATETIME_UNPARSABLE,
            f"时间数值非法：{exc}",
            location=location,
            detail=text,
        )
        return None, nano


# ---------------------------------------------------------------------------
# 解析结果
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ParsedCfg:
    """cfg 解析产物：元数据 + 通道骨架（尚无采样数据）。"""

    meta: Metadata
    analog_channels: list[AnalogChannel] = field(default_factory=list)
    digital_channels: list[DigitalChannel] = field(default_factory=list)
    declared_sample_count: int = 0
    """cfg 声明的采样点数（各采样率段 endsamp 的最大值）。"""

    version: ComtradeVersion = ComtradeVersion.UNKNOWN
    nanosecond_precision: bool = False


# ---------------------------------------------------------------------------
# 主解析流程
# ---------------------------------------------------------------------------

def parse_cfg(
    cfg_path: Path,
    diagnostics: DiagnosticCollector,
    options: ParseOptions = DEFAULT_OPTIONS,
) -> ParsedCfg:
    """解析 .cfg 文件。

    Raises:
        ParseAbort: 遇到无法继续的致命问题（由 reader 统一转换为对外异常）。
    """
    text = read_text(cfg_path, diagnostics)
    filename = cfg_path.name

    if not text.strip():
        diagnostics.fatal(Code.CFG_EMPTY, "配置文件为空", location=filename)
        raise ParseAbort("cfg 为空")

    cursor = _LineCursor(text, filename, diagnostics)

    version, revision_year = _parse_header(cursor, diagnostics)
    total, analog_count, digital_count = _parse_channel_counts(cursor, diagnostics, options)

    if analog_count == 0 and digital_count == 0:
        diagnostics.fatal(Code.CFG_NO_CHANNEL, "配置文件声明了 0 个通道", location=filename)
        raise ParseAbort("无通道")

    analog_channels = _parse_analog_channels(cursor, analog_count, version, diagnostics)
    digital_channels = _parse_digital_channels(cursor, digital_count, version, diagnostics)

    line_freq = _parse_line_frequency(cursor, diagnostics)
    segments, declared_samples = _parse_sample_rates(cursor, diagnostics)

    start_time, nano_start = _parse_time_line(cursor, diagnostics, "起始时刻")
    trigger_time, nano_trigger = _parse_time_line(cursor, diagnostics, "触发时刻")

    data_type = _parse_data_type(cursor, diagnostics)
    time_mult, time_zone, local_code, tmq, leapsec = _parse_tail(cursor, version, diagnostics)

    _warn_trailing_content(cursor, diagnostics)

    nanosecond_precision = nano_start or nano_trigger

    if options.time_base_seconds is not None:
        time_base = options.time_base_seconds
    elif nanosecond_precision:
        time_base = 1e-9
    else:
        time_base = 1e-6

    station_name, device_id = _header_names(text)

    meta = Metadata(
        station_name=station_name,
        device_id=device_id,
        revision_year=revision_year,
        version=version,
        source_cfg=cfg_path,
        source_dat=None,
        line_frequency=line_freq,
        sample_rate_segments=segments,
        sample_count=declared_samples,
        start_time=start_time,
        trigger_time=trigger_time,
        data_type=data_type,
        time_mult=time_mult,
        time_base_seconds=time_base,
        timezone_code=time_zone,
        local_code=local_code,
        time_quality_code=tmq,
        leap_second=leapsec,
        total_channels=total,
        analog_count=len(analog_channels),
        digital_count=len(digital_channels),
    )

    diagnostics.info(
        "CFG-OK",
        f"配置文件解析完成：COMTRADE {version.label}，"
        f"{len(analog_channels)} 个模拟量 / {len(digital_channels)} 个开关量",
        location=filename,
    )

    return ParsedCfg(
        meta=meta,
        analog_channels=analog_channels,
        digital_channels=digital_channels,
        declared_sample_count=declared_samples,
        version=version,
        nanosecond_precision=nanosecond_precision,
    )


def _header_names(text: str) -> tuple[str, str]:
    """从原文首行取厂站名与装置标识（保留原始大小写，不做小写化）。"""
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    parts = _split(first)
    return (parts[0] if parts else "", parts[1] if len(parts) > 1 else "")


def _parse_header(
    cursor: _LineCursor, diagnostics: DiagnosticCollector
) -> tuple[ComtradeVersion, int | None]:
    """解析首行，判定 COMTRADE 版本。

    判定依据是**字段个数**，不是版本年份的数值 ——
    现场存在版本年份写错的文件（如 2000、2001），
    字段个数才是格式差异的真实来源。
    """
    line_no, line = cursor.next("版本标识行")
    fields = _split(line)

    if len(fields) < 2:
        diagnostics.warn(
            Code.CFG_HEADER_UNPARSABLE,
            f"首行仅 {len(fields)} 个字段，按 1991 版规则解析",
            location=cursor.location(line_no),
            detail=line,
        )
        return ComtradeVersion.V1991, None

    if len(fields) == 2:
        # 1991 版没有版本年份字段
        return ComtradeVersion.V1991, None

    raw_year = fields[2]
    year = _as_int(raw_year)
    if year is None:
        diagnostics.warn(
            Code.CFG_VERSION_NONSTANDARD,
            f"版本年份「{raw_year}」不是数字，按 1999 版规则解析",
            location=cursor.location(line_no),
            detail=line,
        )
        return ComtradeVersion.V1999, None

    if year == 1991:
        return ComtradeVersion.V1991, year
    if 1999 <= year < 2013:
        if year != 1999:
            diagnostics.info(
                Code.CFG_VERSION_NONSTANDARD,
                f"版本年份为 {year}（非标准值），按 1999 版规则解析",
                location=cursor.location(line_no),
            )
        return ComtradeVersion.V1999, year
    if year >= 2013:
        return ComtradeVersion.V2013, year

    diagnostics.warn(
        Code.CFG_VERSION_NONSTANDARD,
        f"版本年份 {year} 早于 1991，按 1991 版规则解析",
        location=cursor.location(line_no),
    )
    return ComtradeVersion.V1991, year


def _parse_channel_counts(
    cursor: _LineCursor, diagnostics: DiagnosticCollector, options: ParseOptions
) -> tuple[int, int, int]:
    """解析第 2 行：``总通道数, 模拟量数, 开关量数``。"""
    line_no, line = cursor.next("通道数量行")
    fields = _split(line)

    if len(fields) < 3:
        diagnostics.fatal(
            Code.CFG_CHANNEL_COUNT_LINE,
            f"通道数量行应有 3 个字段，实际 {len(fields)} 个",
            location=cursor.location(line_no),
            detail=line,
        )
        raise ParseAbort("通道数量行非法")

    total = _parse_channel_count(fields[0])
    analog = _parse_channel_count(fields[1])
    digital = _parse_channel_count(fields[2])

    if total is None or analog is None or digital is None:
        diagnostics.fatal(
            Code.CFG_CHANNEL_COUNT_LINE,
            "通道数量行含无法解析的数值",
            location=cursor.location(line_no),
            detail=line,
        )
        raise ParseAbort("通道数量行非法")

    if total != analog + digital:
        message = (
            f"声明总通道数 {total} 与 模拟量 {analog} + 开关量 {digital} = "
            f"{analog + digital} 不一致"
        )
        if options.strict_channel_count:
            diagnostics.fatal(
                Code.CFG_CHANNEL_COUNT_MISMATCH, message, location=cursor.location(line_no)
            )
            raise ParseAbort("通道数量不一致")
        # 以模拟量+开关量为准：后面的行数按这两个数排布，这是唯一能继续解析的选择
        diagnostics.warn(
            Code.CFG_CHANNEL_COUNT_MISMATCH,
            message + "，已按 模拟量+开关量 继续解析",
            location=cursor.location(line_no),
            detail=line,
        )

    return total, analog, digital


# ------------------------------------------------ 通道定义行

def _parse_analog_channels(
    cursor: _LineCursor,
    count: int,
    version: ComtradeVersion,
    diagnostics: DiagnosticCollector,
) -> list[AnalogChannel]:
    """解析模拟量通道定义行。

    1991 版 10 个字段，1999/2013 版 13 个字段（多出 primary/secondary/PS）。
    对两种字段数都接受：现场存在版本年份与字段数不匹配的文件。
    """
    channels: list[AnalogChannel] = []

    for i in range(count):
        line_no, line = cursor.next(f"第 {i + 1} 个模拟量通道定义")
        fields = _split(line)
        loc = cursor.location(line_no)
        n = len(fields)

        if n not in (10, 13):
            if n < 10:
                diagnostics.warn(
                    Code.CFG_ANALOG_FIELD_COUNT,
                    f"模拟通道「{fields[1] if n > 1 else i + 1}」字段数 {n}，"
                    "少于 1991 版要求的 10 个，缺失字段按默认值处理",
                    location=loc,
                    detail=line,
                )
            else:
                diagnostics.info(
                    Code.CFG_ANALOG_FIELD_COUNT,
                    f"模拟通道字段数 {n}（非 10/13），多余字段已忽略",
                    location=loc,
                )
        elif n == 10 and version >= ComtradeVersion.V1999:
            diagnostics.info(
                Code.CFG_ANALOG_FIELD_COUNT,
                "声明为 1999/2013 版但模拟通道只有 10 个字段（缺变比信息），"
                "已按 1991 版格式解析",
                location=loc,
            )

        def get_field(idx: int, default: str = "") -> str:
            return fields[idx] if idx < n else default

        name = get_field(1)
        if not name:
            diagnostics.warn(
                Code.CFG_CHANNEL_NAME_EMPTY,
                f"第 {i + 1} 个模拟量通道名称为空",
                location=loc,
            )
            name = f"ANALOG_{i + 1}"

        unit_raw = get_field(4)
        unit_info = parse_unit(unit_raw)
        if not unit_raw:
            diagnostics.info(
                Code.CFG_UNIT_EMPTY,
                f"通道「{name}」的单位字段为空，未做单位归一化",
                location=loc,
            )
        elif not unit_info.recognized:
            diagnostics.info(
                Code.CHN_UNIT_UNRECOGNIZED,
                f"通道「{name}」的单位「{unit_raw}」无法识别，未做单位归一化",
                location=loc,
            )

        a = _as_float(get_field(5), 1.0) or 0.0
        b = _as_float(get_field(6), 0.0) or 0.0
        if a == 0.0:
            diagnostics.warn(
                Code.CFG_SCALE_A_ZERO,
                f"通道「{name}」的比例系数 a 为 0，无法构成有效换算，将跳过 a/b 换算",
                location=loc,
                detail=line,
            )

        primary = _as_float(get_field(10)) if n >= 13 else None
        secondary = _as_float(get_field(11)) if n >= 13 else None
        ps_raw = get_field(12).upper() if n >= 13 else ""
        ps = ps_raw if ps_raw in ("P", "S") else None

        channels.append(
            AnalogChannel(
                index=i,
                declared_no=_as_int(get_field(0)),
                name=name,
                phase_raw=get_field(2),
                circuit=get_field(3),
                unit_raw=unit_raw,
                a=a,
                b=b,
                skew_us=_as_float(get_field(7), 0.0) or 0.0,
                raw_min=_as_float(get_field(8), 0.0) or 0.0,
                raw_max=_as_float(get_field(9), 0.0) or 0.0,
                primary=primary,
                secondary=secondary,
                ps=ps,
                unit=unit_info.canonical,
                role_confidence=0.0,
            )
        )

    return channels


def _parse_digital_channels(
    cursor: _LineCursor,
    count: int,
    version: ComtradeVersion,
    diagnostics: DiagnosticCollector,
) -> list[DigitalChannel]:
    """解析开关量通道定义行。

    1991 版 3 个字段（``序号, 名称, 常态值``），
    1999/2013 版 5 个字段（多出 ``ph`` 与 ``ccbm``）。
    """
    channels: list[DigitalChannel] = []

    for i in range(count):
        line_no, line = cursor.next(f"第 {i + 1} 个开关量通道定义")
        fields = _split(line)
        loc = cursor.location(line_no)
        n = len(fields)

        if n not in (3, 5):
            diagnostics.info(
                Code.CFG_DIGITAL_FIELD_COUNT,
                f"开关量通道字段数 {n}（非 3/5），缺失字段按默认值处理，多余字段已忽略",
                location=loc,
                detail=line,
            )

        def get_field(idx: int, default: str = "") -> str:
            return fields[idx] if idx < n else default

        name = get_field(1)
        if not name:
            diagnostics.warn(
                Code.CFG_CHANNEL_NAME_EMPTY, f"第 {i + 1} 个开关量通道名称为空", location=loc
            )
            name = f"DIGITAL_{i + 1}"

        if n >= 5:
            phase_raw, circuit, y_idx = get_field(2), get_field(3), 4
        else:
            # 1991 版：序号,名称,常态值
            phase_raw, circuit, y_idx = "", "", 2

        channels.append(
            DigitalChannel(
                index=i,
                declared_no=_as_int(get_field(0)),
                name=name,
                phase_raw=phase_raw,
                circuit=circuit,
                normal_state=_as_int(fields[y_idx], 0) if y_idx < n else 0,
            )
        )

    return channels


# ------------------------------------------------ 采样相关

def _parse_line_frequency(cursor: _LineCursor, diagnostics: DiagnosticCollector) -> float:
    line_no, line = cursor.next("系统频率")
    freq = _as_float(line, 0.0) or 0.0
    if freq <= 0:
        diagnostics.info(
            Code.CFG_NO_SAMPLE_RATE,
            "系统频率未声明或为 0，不影响解析（特征计算需要时由算法模块按 50Hz 处理）",
            location=cursor.location(line_no),
        )
    return freq


def _parse_sample_rates(
    cursor: _LineCursor, diagnostics: DiagnosticCollector
) -> tuple[list[SampleRateSegment], int]:
    """解析采样率段：``nrates`` 行 + ``nrates`` 组 ``采样率,结束采样号``。"""
    line_no, line = cursor.next("采样率段数")
    nrates = _as_int(line)

    if nrates is None or nrates < 0:
        diagnostics.warn(
            Code.CFG_SAMPLE_RATE_LINE,
            f"采样率段数「{line}」无法解析，按 0 处理（时间轴改用采样时标推算）",
            location=cursor.location(line_no),
            detail=line,
        )
        nrates = 0

    if nrates == 0:
        diagnostics.warn(
            Code.CFG_NO_SAMPLE_RATE,
            "文件声明的采样率段数为 0，时间轴将改用采样点自带时标推算",
            location=cursor.location(line_no),
        )
        return [], 0

    segments: list[SampleRateSegment] = []
    declared_max = 0

    for i in range(nrates):
        seg_no, seg_line = cursor.next(f"第 {i + 1} 段采样率")
        fields = _split(seg_line)
        loc = cursor.location(seg_no)

        if len(fields) < 2:
            diagnostics.warn(
                Code.CFG_SAMPLE_RATE_LINE,
                f"第 {i + 1} 段采样率字段数不足 2",
                location=loc,
                detail=seg_line,
            )
            continue

        rate = _as_float(fields[0], 0.0) or 0.0
        end_sample = _as_int(fields[1], 0) or 0

        if rate <= 0:
            diagnostics.warn(
                Code.CFG_SAMPLE_RATE_LINE,
                f"第 {i + 1} 段采样率 {rate:g} 非正数，该段将依赖采样时标",
                location=loc,
            )
        if end_sample <= 0:
            diagnostics.warn(
                Code.CFG_SAMPLE_RATE_LINE,
                f"第 {i + 1} 段结束采样号 {end_sample} 非正数",
                location=loc,
            )

        segments.append(SampleRateSegment(rate_hz=rate, end_sample=end_sample))
        declared_max = max(declared_max, end_sample)

    # 段的结束采样号必须递增，否则采样率分段无意义
    ends = [s.end_sample for s in segments]
    if ends != sorted(ends):
        diagnostics.warn(
            Code.CFG_SAMPLE_RATE_LINE,
            f"采样率段的结束采样号不是递增的（{ends}），时间轴可能不正确",
            location=cursor.location(),
        )

    return segments, declared_max


def _parse_time_line(
    cursor: _LineCursor, diagnostics: DiagnosticCollector, what: str
) -> tuple[datetime | None, bool]:
    line_no, line = cursor.next(what)
    return parse_datetime(line, diagnostics, cursor.location(line_no))


def _parse_data_type(cursor: _LineCursor, diagnostics: DiagnosticCollector) -> DataFileType:
    line_no, line = cursor.next("数据文件类型")
    raw = line.strip().upper()
    try:
        return DataFileType(raw)
    except ValueError:
        diagnostics.fatal(
            Code.CFG_DATA_TYPE_UNKNOWN,
            f"无法识别的数据文件类型「{line}」，"
            f"支持的类型：{'/'.join(t.value for t in DataFileType)}",
            location=cursor.location(line_no),
            detail=line,
        )
        raise ParseAbort("数据类型非法")


def _parse_tail(
    cursor: _LineCursor,
    version: ComtradeVersion,
    diagnostics: DiagnosticCollector,
) -> tuple[float, str | None, str | None, str | None, str | None]:
    """解析 ``ft`` 之后的尾部字段。

    1991 版：无任何尾部字段。
    1999 版：``time_mult``。
    2013 版：``time_mult`` + ``time_code,local_code`` + ``tmq_code,leapsec``。

    标准把 2013 版的新字段**追加在文件末尾**（而不是插在 ft 之后），
    以免破坏老解析器 —— 老解析器读到 time_mult 就停下了。
    但现场两种顺序都出现过，因此这里不按位置硬取，而是**按字段形状**归类：

    * 只有一个字段且能解析成数字 → ``time_mult``
    * 两个字段 → 依次是 ``time_code`` 与 ``tmq_code`` 行

    Returns:
        ``(time_mult, time_code, local_code, tmq_code, leapsec)``
    """
    time_mult = 1.0
    time_zone: str | None = None
    local_code: str | None = None
    tmq: str | None = None
    leapsec: str | None = None
    saw_time_mult = False
    pair_lines: list[tuple[int, str, list[str]]] = []

    for line_no, line in cursor.remaining():
        fields = _split(line)
        # 单字段且为数值 → 时间倍数
        if len(fields) == 1 and _as_float(fields[0]) is not None:
            value = _as_float(fields[0], 1.0) or 1.0
            if value <= 0:
                diagnostics.warn(
                    Code.CFG_TIME_MULT_INVALID,
                    f"时间倍数「{line}」非法，按 1 处理",
                    location=cursor.location(line_no),
                    detail=line,
                )
            else:
                time_mult = value
            saw_time_mult = True
            continue
        if len(fields) >= 2:
            pair_lines.append((line_no, line, fields))
            continue
        # 其它形状：忽略
        diagnostics.info(
            Code.CFG_TRAILING_CONTENT,
            f"忽略无法识别的尾部行：{line[:60]}",
            location=cursor.location(line_no),
        )

    # 两字段行按出现顺序对应 time_code 行、tmq_code 行
    if pair_lines:
        _, line, fields = pair_lines[0]
        time_zone, local_code = fields[0], fields[1]
        if version is not ComtradeVersion.V2013:
            diagnostics.info(
                Code.CFG_TRAILING_CONTENT,
                f"非 2013 版文件出现 time_code 行（{line[:40]}），已按 2013 版规则读取",
                location=cursor.location(pair_lines[0][0]),
            )
    if len(pair_lines) >= 2:
        tmq, leapsec = pair_lines[1][2][0], pair_lines[1][2][1]

    if not saw_time_mult and version is not ComtradeVersion.V1991:
        diagnostics.info(
            Code.CFG_TIME_MULT_INVALID,
            "文件未提供时间倍数 time_mult，按 1 处理",
            location=cursor.location(),
        )

    # 消费掉尾部，避免被当成"多余内容"重复告警
    cursor.consume_rest()
    return time_mult, time_zone, local_code, tmq, leapsec


def _warn_trailing_content(cursor: _LineCursor, diagnostics: DiagnosticCollector) -> None:
    leftovers = cursor.remaining()
    if leftovers:
        diagnostics.info(
            Code.CFG_TRAILING_CONTENT,
            f"文件末尾有 {len(leftovers)} 行多余内容（已忽略）",
            location=cursor.location(leftovers[0][0]),
            detail=leftovers[0][1][:80],
        )


def compute_file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    """计算文件内容摘要，用于案例去重与解析结果一致性校验（NF-14）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


__all__ = ["ParsedCfg", "parse_cfg", "parse_datetime", "compute_file_sha256"]
