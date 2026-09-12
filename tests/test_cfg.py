"""cfg 解析器测试：版本自适应、字段数容错、时间解析、异常处理。"""
from __future__ import annotations

import tempfile
from pathlib import Path

from comtrade.cfg_parser import parse_cfg, parse_datetime
from comtrade.diagnostics import Code, DiagnosticCollector, ParseAbort
from comtrade.models import ComtradeVersion, DataFileType

from tests._util import AnalogDef, CfgSpec, DigitalDef, render_cfg


def _parse_cfg_text(tmpdir: Path, text: str, encoding: str = "utf-8"):
    path = tmpdir / "t.cfg"
    path.write_text(text, encoding=encoding)
    diag = DiagnosticCollector()
    parsed = parse_cfg(path, diag)
    return parsed, diag


# ---------------------------------------------------------------------------
# 版本判定
# ---------------------------------------------------------------------------

def test_version_from_header_field_count():
    """版本判定的唯一可靠依据是首行字段个数。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # 2 字段 → 1991
        parsed, _ = _parse_cfg_text(tmp, render_cfg(CfgSpec(version=None)))
        assert parsed.version is ComtradeVersion.V1991
        assert parsed.meta.revision_year is None

        # 3 字段 → 1999
        parsed, _ = _parse_cfg_text(tmp, render_cfg(CfgSpec(version=1999)))
        assert parsed.version is ComtradeVersion.V1999
        assert parsed.meta.revision_year == 1999

        # 3 字段 → 2013
        parsed, _ = _parse_cfg_text(
            tmp, render_cfg(CfgSpec(version=2013, tail_lines=["8h00,8h00", "B,0"]))
        )
        assert parsed.version is ComtradeVersion.V2013


def test_nonstandard_revision_year():
    """现场有 2000/2001 这类非标准年份，必须按最接近的版本规则解析而不是报错。"""
    for year in (2000, 2001, 2010):
        with tempfile.TemporaryDirectory() as td:
            parsed, diag = _parse_cfg_text(Path(td), render_cfg(CfgSpec(version=year)))
            assert parsed.version is ComtradeVersion.V1999, year
            # 版本年份非标准应留下痕迹
            assert Code.CFG_VERSION_NONSTANDARD in diag.codes()


def test_version_year_written_wrong_but_field_count_correct():
    """声明 2013 但字段数是 1991 格式：应能解析并提示，而不是崩溃。"""
    spec = CfgSpec(version=2013, tail_lines=["8h00,8h00", "B,0"])
    text = render_cfg(spec)
    # 把模拟通道行裁成 10 字段
    lines = text.splitlines()
    lines[2] = ",".join(lines[2].split(",")[:10])
    with tempfile.TemporaryDirectory() as td:
        parsed, diag = _parse_cfg_text(Path(td), "\n".join(lines) + "\n")
        assert len(parsed.analog_channels) == 1
        assert parsed.analog_channels[0].primary is None
        assert Code.CFG_ANALOG_FIELD_COUNT in diag.codes()


# ---------------------------------------------------------------------------
# 通道定义
# ---------------------------------------------------------------------------

def test_analog_field_count_1991_vs_1999():
    """1991 版 10 字段（无变比），1999 版 13 字段（含变比）。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        parsed, _ = _parse_cfg_text(tmp, render_cfg(CfgSpec(version=None)))
        ch = parsed.analog_channels[0]
        assert ch.primary is None and ch.secondary is None and ch.ps is None

        parsed, _ = _parse_cfg_text(tmp, render_cfg(CfgSpec(version=1999)))
        ch = parsed.analog_channels[0]
        assert ch.primary == 400.0
        assert ch.secondary == 1.0
        assert ch.ps == "S"


def test_digital_field_count_1991_vs_1999():
    """1991 版 3 字段，1999 版 5 字段（多出 ph 与 ccbm）。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        parsed, _ = _parse_cfg_text(tmp, render_cfg(CfgSpec(version=None)))
        d = parsed.digital_channels[0]
        assert d.phase_raw == "" and d.circuit == ""
        assert d.normal_state == 0

        parsed, _ = _parse_cfg_text(tmp, render_cfg(CfgSpec(version=1999)))
        d = parsed.digital_channels[0]
        assert d.phase_raw == "A"
        assert d.circuit == "LINE1"


def test_declared_channel_number_is_not_trusted():
    """通道序号字段现场会乱写（跳号、从 0 起），一律以读取顺序为准。"""
    spec = CfgSpec(version=1999, analog=[AnalogDef(name="Ia"), AnalogDef(name="Ib")])
    text = render_cfg(spec)
    lines = text.splitlines()
    lines[2] = lines[2].replace("1,Ia", "7,Ia")  # 序号写成 7
    lines[3] = lines[3].replace("2,Ib", "0,Ib")  # 序号写成 0
    with tempfile.TemporaryDirectory() as td:
        parsed, _ = _parse_cfg_text(Path(td), "\n".join(lines) + "\n")
        assert [c.index for c in parsed.analog_channels] == [0, 1]
        assert parsed.analog_channels[0].declared_no == 7
        assert parsed.analog_channels[1].declared_no == 0


def test_channel_name_empty_gets_placeholder_and_warning():
    spec = CfgSpec(version=1999, analog=[AnalogDef(name="")])
    with tempfile.TemporaryDirectory() as td:
        parsed, diag = _parse_cfg_text(Path(td), render_cfg(spec))
        assert parsed.analog_channels[0].name == "ANALOG_1"
        assert Code.CFG_CHANNEL_NAME_EMPTY in diag.codes()


def test_channel_count_mismatch_is_recoverable():
    """总通道数与 A+D 之和不一致时，按 A+D 继续（否则后面的行全错位）。"""
    spec = CfgSpec(version=1999, analog=[AnalogDef(), AnalogDef(name="Ib")])
    text = render_cfg(spec).replace("3,2A,1D", "9,2A,1D", 1)
    with tempfile.TemporaryDirectory() as td:
        parsed, diag = _parse_cfg_text(Path(td), text)
        assert len(parsed.analog_channels) == 2
        assert Code.CFG_CHANNEL_COUNT_MISMATCH in diag.codes()


def test_channel_count_format_tolerates_missing_letters():
    """有些厂家把 ``2A,1D`` 写成 ``2,1``。"""
    spec = CfgSpec(version=1999, analog=[AnalogDef()], digital=[DigitalDef()])
    text = render_cfg(spec).replace("2,1A,1D", "2,1,1")
    with tempfile.TemporaryDirectory() as td:
        parsed, _ = _parse_cfg_text(Path(td), text)
        assert len(parsed.analog_channels) == 1


def test_scale_a_zero_is_flagged():
    spec = CfgSpec(version=1999, analog=[AnalogDef(a=0.0)])
    with tempfile.TemporaryDirectory() as td:
        _, diag = _parse_cfg_text(Path(td), render_cfg(spec))
        assert Code.CFG_SCALE_A_ZERO in diag.codes()


def test_unit_empty_is_flagged_but_not_fatal():
    spec = CfgSpec(version=1999, analog=[AnalogDef(unit="")])
    with tempfile.TemporaryDirectory() as td:
        _, diag = _parse_cfg_text(Path(td), render_cfg(spec))
        assert Code.CFG_UNIT_EMPTY in diag.codes()


# ---------------------------------------------------------------------------
# 采样率与数据格式
# ---------------------------------------------------------------------------

def test_sample_rate_segments_preserved():
    spec = CfgSpec(version=1999, segments=[(1000.0, 200), (4000.0, 1600)])
    with tempfile.TemporaryDirectory() as td:
        parsed, _ = _parse_cfg_text(Path(td), render_cfg(spec))
        segs = parsed.meta.sample_rate_segments
        assert len(segs) == 2
        assert segs[0].rate_hz == 1000.0 and segs[0].end_sample == 200
        assert segs[1].rate_hz == 4000.0 and segs[1].end_sample == 1600
        # 声明采样点数取各段 endsamp 的最大值
        assert parsed.declared_sample_count == 1600


def test_zero_sample_rate_segments_falls_back_to_timestamps():
    spec = CfgSpec(version=1999, segments=[])
    text = render_cfg(spec)
    with tempfile.TemporaryDirectory() as td:
        parsed, diag = _parse_cfg_text(Path(td), text)
        assert parsed.meta.sample_rate_segments == []
        assert Code.CFG_NO_SAMPLE_RATE in diag.codes()


def test_all_four_data_types_recognized():
    for dtype in ("ASCII", "BINARY", "BINARY32", "FLOAT32"):
        with tempfile.TemporaryDirectory() as td:
            parsed, _ = _parse_cfg_text(Path(td), render_cfg(CfgSpec(data_type=dtype)))
            assert parsed.meta.data_type is DataFileType(dtype)


def test_unknown_data_type_is_fatal():
    with tempfile.TemporaryDirectory() as td:
        diag = DiagnosticCollector()
        path = Path(td) / "t.cfg"
        path.write_text(render_cfg(CfgSpec(data_type="XML")), encoding="utf-8")
        try:
            parse_cfg(path, diag)
        except ParseAbort:
            pass
        else:
            raise AssertionError("非法数据类型应当中断解析")
        assert diag.has_fatal()
        assert Code.CFG_DATA_TYPE_UNKNOWN in diag.codes()


# ---------------------------------------------------------------------------
# 尾部字段
# ---------------------------------------------------------------------------

def test_time_mult_read_in_1999():
    with tempfile.TemporaryDirectory() as td:
        parsed, _ = _parse_cfg_text(Path(td), render_cfg(CfgSpec(version=1999, time_mult=0.5)))
        assert parsed.meta.time_mult == 0.5


def test_time_mult_absent_defaults_to_one():
    with tempfile.TemporaryDirectory() as td:
        parsed, diag = _parse_cfg_text(Path(td), render_cfg(CfgSpec(version=1999, time_mult=None)))
        assert parsed.meta.time_mult == 1.0


def test_2013_tail_standard_order():
    """标准把 2013 新字段追加在末尾：ft → time_mult → time_code → tmq_code。"""
    spec = CfgSpec(version=2013, time_mult=1.0, tail_lines=["8h00,8h00", "B,3"])
    with tempfile.TemporaryDirectory() as td:
        parsed, _ = _parse_cfg_text(Path(td), render_cfg(spec))
        assert parsed.meta.timezone_code == "8h00"
        assert parsed.meta.local_code == "8h00"
        assert parsed.meta.time_quality_code == "B"
        assert parsed.meta.leap_second == "3"
        assert parsed.meta.time_mult == 1.0


def test_2013_tail_missing_is_tolerated():
    spec = CfgSpec(version=2013, time_mult=1.0, tail_lines=[])
    with tempfile.TemporaryDirectory() as td:
        parsed, _ = _parse_cfg_text(Path(td), render_cfg(spec))
        assert parsed.meta.timezone_code is None
        assert parsed.meta.time_mult == 1.0


# ---------------------------------------------------------------------------
# 时间解析
# ---------------------------------------------------------------------------

def test_datetime_day_month_order():
    diag = DiagnosticCollector()
    dt, _ = parse_datetime("25/12/2026,10:20:30.123456", diag, "x")
    assert dt.year == 2026 and dt.month == 12 and dt.day == 25
    assert dt.microsecond == 123456


def test_datetime_detects_month_day_order():
    """第一个数 >12 时才能确定是 dd/mm；反过来则说明是 mm/dd。"""
    diag = DiagnosticCollector()
    dt, _ = parse_datetime("12/25/2026,00:00:00.000000", diag, "x")
    assert dt.month == 12 and dt.day == 25


def test_datetime_ambiguous_warns():
    diag = DiagnosticCollector()
    dt, _ = parse_datetime("01/02/2026,00:00:00.000000", diag, "x")
    # 按标准取 日/月
    assert dt.day == 1 and dt.month == 2
    assert Code.CFG_DATE_AMBIGUOUS in diag.codes()


def test_datetime_iso_like_year_first():
    diag = DiagnosticCollector()
    dt, _ = parse_datetime("2026/07/20,08:00:00.000000", diag, "x")
    assert dt.year == 2026 and dt.month == 7 and dt.day == 20


def test_datetime_nanosecond_is_truncated_with_diagnostic():
    """2013 版允许 9 位小数（纳秒），Python datetime 只到微秒。"""
    diag = DiagnosticCollector()
    dt, nano = parse_datetime("20/07/2026,08:00:00.123456789", diag, "x")
    assert nano is True
    assert dt.microsecond == 123456
    assert Code.CFG_TIME_PRECISION_TRUNCATED in diag.codes()


def test_datetime_garbage_is_reported_not_raised():
    diag = DiagnosticCollector()
    dt, _ = parse_datetime("不是时间", diag, "x")
    assert dt is None
    assert Code.CFG_DATETIME_UNPARSABLE in diag.codes()


# ---------------------------------------------------------------------------
# 异常文件
# ---------------------------------------------------------------------------

def test_empty_cfg_is_fatal():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.cfg"
        path.write_text("", encoding="utf-8")
        diag = DiagnosticCollector()
        try:
            parse_cfg(path, diag)
        except ParseAbort:
            pass
        else:
            raise AssertionError("空 cfg 应当中断")
        assert Code.CFG_EMPTY in diag.codes()


def test_truncated_cfg_is_fatal():
    """通道声明了多个但文件在通道定义中途结束 —— 必须报致命错误而不是崩溃。"""
    text = render_cfg(CfgSpec(version=1999, analog=[AnalogDef(), AnalogDef(name="Ib")]))
    lines = text.splitlines()[:3]  # 只保留 首行 + 通道数量行 + 第 1 个模拟通道行
    with tempfile.TemporaryDirectory() as td:
        diag = DiagnosticCollector()
        path = Path(td) / "t.cfg"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            parse_cfg(path, diag)
        except ParseAbort:
            pass
        else:
            raise AssertionError("cfg 提前结束应当中断")
        assert diag.has_fatal()
        assert Code.CFG_UNEXPECTED_END in diag.codes()


def test_gbk_encoded_chinese_names():
    """国内录波器常把中文按 GBK 写入 cfg，UTF-8 读取会失败。"""
    spec = CfgSpec(version=1999, station="110kV某某变", analog=[AnalogDef(name="A相电流")],
                   encoding="gbk")
    with tempfile.TemporaryDirectory() as td:
        parsed, _ = _parse_cfg_text(Path(td), render_cfg(spec), encoding="gbk")
        assert parsed.meta.station_name == "110kV某某变"
        assert parsed.analog_channels[0].name == "A相电流"


def test_extra_trailing_lines_are_ignored():
    text = render_cfg(CfgSpec(version=1999)) + "\n\n# 备注\n"
    with tempfile.TemporaryDirectory() as td:
        parsed, _ = _parse_cfg_text(Path(td), text)
        assert len(parsed.analog_channels) == 1
