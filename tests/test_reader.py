"""主入口测试：文件配对、完整性检查、非法文件处理、端到端。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from comtrade import ChannelRole, ComtradeParseError, ComtradeVersion, load_recording, try_load_recording
from comtrade.diagnostics import Code
from comtrade.reader import resolve_pair

from tests._util import AnalogDef, CfgSpec, DigitalDef, TempRecording, simple_sine


def _standard_case(tmp: Path, name="case", *, version=1999, data_type="BINARY",
                   extra_tail=(), segments=((4000.0, 400),)):
    """构造一份内容正确的录波：Ia/Ib/Ic/Ua/Ub/Uc + 2 个开关量。"""
    rec = TempRecording(tmp, name)
    spec = CfgSpec(
        version=version,
        analog=[
            AnalogDef(name="Ia", phase="A", unit="A"),
            AnalogDef(name="Ib", phase="B", unit="A"),
            AnalogDef(name="Ic", phase="C", unit="A"),
            AnalogDef(name="Ua", phase="A", unit="V", primary=110000.0, secondary=100.0),
            AnalogDef(name="Ub", phase="B", unit="V", primary=110000.0, secondary=100.0),
            AnalogDef(name="Uc", phase="C", unit="V", primary=110000.0, secondary=100.0),
        ],
        digital=[DigitalDef(name="TRIP"), DigitalDef(name="52a")],
        data_type=data_type,
        segments=[(r, e) for r, e in segments],
        tail_lines=list(extra_tail),
    )
    rec.write_cfg(spec)

    n = segments[-1][1]
    analog = np.vstack([simple_sine(n=n, amplitude=1000.0, phase=i * 1.0) for i in range(6)])
    digital = np.zeros((2, n), dtype=bool)
    digital[0, n // 2:] = True
    if data_type == "ASCII":
        rec.write_ascii(analog, digital)
    else:
        rec.write_binary(analog, digital, data_type=data_type)
    return rec


# ---------------------------------------------------------------------------
# 文件配对（F-03）
# ---------------------------------------------------------------------------

def test_pair_resolution_from_cfg_and_dat():
    with tempfile.TemporaryDirectory() as td:
        rec = _standard_case(Path(td))
        cfg, dat = resolve_pair(rec.cfg, __import__("comtrade").DiagnosticCollector())
        assert cfg == rec.cfg and dat == rec.dat

        cfg2, dat2 = resolve_pair(rec.dat, __import__("comtrade").DiagnosticCollector())
        assert cfg2 == rec.cfg and dat2 == rec.dat


def test_pair_resolution_is_case_insensitive():
    """扩展名大小写不一致（.CFG/.DAT）也必须能配对。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp, "up")
        rec.write_cfg(CfgSpec(version=1999))
        rec.write_binary(np.zeros((1, 10), dtype=np.int32), np.zeros((1, 10), dtype=bool))
        upper_cfg = tmp / "up.CFG"
        rec.cfg.rename(upper_cfg)
        cfg, dat = resolve_pair(upper_cfg, __import__("comtrade").DiagnosticCollector())
        assert cfg == upper_cfg and dat == rec.dat


def test_missing_pair_is_fatal():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp, "lonely")
        rec.write_cfg(CfgSpec(version=1999))
        recording, diags = try_load_recording(rec.cfg)
        assert recording is None
        assert any(d.code == Code.PAIR_MISSING for d in diags)


def test_missing_file_is_fatal():
    recording, diags = try_load_recording("C:/不存在的路径/nothing.cfg")
    assert recording is None
    assert any(d.code == Code.FILE_NOT_FOUND for d in diags)


def test_unsupported_extension_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.txt"
        path.write_text("hello", encoding="utf-8")
        recording, diags = try_load_recording(path)
        assert recording is None
        assert any(d.code == Code.UNSUPPORTED_EXTENSION for d in diags)


def test_cff_is_reported_as_unsupported():
    """2013 版单文件格式本期不支持，必须给出明确提示而不是莫名崩溃。"""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.cff"
        path.write_text("[CFG]\n...", encoding="utf-8")
        recording, diags = try_load_recording(path)
        assert recording is None
        assert any(d.code == Code.CFF_NOT_SUPPORTED for d in diags)


def test_empty_file_is_rejected():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "empty.cfg"
        path.write_bytes(b"")
        recording, diags = try_load_recording(path)
        assert recording is None
        assert any(d.code == Code.FILE_EMPTY for d in diags)


def test_cfg_dat_swapped_is_detected():
    """把 cfg 和 dat 的内容传反了 —— 应检测出来并提示，而不是抛未处理异常。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = _standard_case(tmp, "swap")
        cfg_bytes = rec.cfg.read_bytes()
        dat_bytes = rec.dat.read_bytes()
        rec.cfg.write_bytes(dat_bytes)
        rec.dat.write_bytes(cfg_bytes)
        recording, diags = try_load_recording(rec.cfg)
        assert recording is None
        assert any(
            d.code in (Code.FILE_LOOKS_BINARY, Code.CFG_HEADER_UNPARSABLE,
                       Code.CFG_CHANNEL_COUNT_LINE, Code.CFG_UNEXPECTED_END,
                       Code.CFG_DATA_TYPE_UNKNOWN, "SYS-001")
            for d in diags
        )


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------

def test_end_to_end_binary():
    with tempfile.TemporaryDirectory() as td:
        rec = _standard_case(Path(td))
        recording = load_recording(rec.cfg)

        assert recording.meta.version is ComtradeVersion.V1999
        assert recording.meta.station_name == "TEST-STATION"
        assert recording.sample_count == 400
        assert recording.duration > 0
        assert recording.meta.cfg_sha256 and recording.meta.dat_sha256

        roles = recording.by_role_map()
        for role in (ChannelRole.IA, ChannelRole.IB, ChannelRole.IC,
                     ChannelRole.UA, ChannelRole.UB, ChannelRole.UC):
            assert role in roles, role

        ia = recording.require_role(ChannelRole.IA)
        assert ia.values.shape == (400,)
        assert np.all(np.isfinite(ia.values))
        assert ia.unit == "A"


def test_end_to_end_all_data_types():
    for dtype in ("ASCII", "BINARY", "BINARY32", "FLOAT32"):
        with tempfile.TemporaryDirectory() as td:
            rec = _standard_case(Path(td), name=f"c_{dtype}", data_type=dtype)
            recording = load_recording(rec.cfg)
            assert recording.meta.data_type.value == dtype
            assert recording.sample_count == 400
            assert recording.require_role(ChannelRole.IA).values.shape == (400,)


def test_end_to_end_1991_and_2013():
    for version, tail in ((1991, ()), (2013, ("8h00,8h00", "B,0"))):
        with tempfile.TemporaryDirectory() as td:
            rec = _standard_case(Path(td), name=f"v{version}", version=version, extra_tail=tail)
            recording = load_recording(rec.cfg)
            assert recording.meta.version is ComtradeVersion(version)
            assert recording.sample_count == 400


def test_digital_channels_and_transitions():
    with tempfile.TemporaryDirectory() as td:
        rec = _standard_case(Path(td))
        recording = load_recording(rec.cfg)
        trip = recording.digital_channels[0]
        assert trip.values.dtype == bool
        assert trip.transitions().tolist() == [200]
        assert recording.digital_channels[1].transitions().tolist() == []


def test_time_axis_matches_declared_rate():
    with tempfile.TemporaryDirectory() as td:
        rec = _standard_case(Path(td))
        recording = load_recording(rec.cfg)
        assert abs(recording.duration - 399 / 4000.0) < 1e-9
        assert np.all(np.diff(recording.time_axis) > 0)


# ---------------------------------------------------------------------------
# 结果查询接口
# ---------------------------------------------------------------------------

def test_by_role_helpers():
    with tempfile.TemporaryDirectory() as td:
        rec = _standard_case(Path(td))
        recording = load_recording(rec.cfg)

        assert len(recording.by_role(ChannelRole.IA, ChannelRole.IB)) == 2
        assert recording.first_by_role(ChannelRole.UC) is not None
        assert recording.first_by_role(ChannelRole.I0) is None

        try:
            recording.require_role(ChannelRole.I0)
        except KeyError as exc:
            assert "I0" in str(exc)
        else:
            raise AssertionError("缺失角色应当抛 KeyError")


def test_index_range_by_time():
    with tempfile.TemporaryDirectory() as td:
        rec = _standard_case(Path(td))
        recording = load_recording(rec.cfg)
        start, end = recording.index_range(0.0, 0.0005)
        assert start == 0
        assert end == 3  # 0, 0.00025, 0.0005 三个点


def test_summary_is_renderable():
    with tempfile.TemporaryDirectory() as td:
        rec = _standard_case(Path(td))
        recording = load_recording(rec.cfg)
        text = recording.summary()
        assert "COMTRADE 版本" in text
        assert "Ia" in text


# ---------------------------------------------------------------------------
# 异常数据的健壮性（NF-12）
# ---------------------------------------------------------------------------

def test_truncated_dat_does_not_crash():
    """数据文件被截断 —— 必须给出可用结果 + 明确诊断，而不是抛异常。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = _standard_case(tmp, "trunc")
        data = rec.dat.read_bytes()
        rec.dat.write_bytes(data[: len(data) // 2])
        recording, diags = try_load_recording(rec.cfg)
        assert recording is not None
        assert any(d.code == Code.DAT_SIZE_MISMATCH for d in diags)


def test_single_side_file_does_not_crash():
    """只有 cfg 没有 dat。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp, "half")
        rec.write_cfg(CfgSpec(version=1999))
        recording, diags = try_load_recording(rec.cfg)
        assert recording is None
        assert any(d.code == Code.PAIR_MISSING for d in diags)


def test_load_recording_raises_with_diagnostics():
    """单文件导入场景：抛异常，但异常里带着完整诊断供界面展示。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp, "half2")
        rec.write_cfg(CfgSpec(version=1999))
        try:
            load_recording(rec.cfg)
        except ComtradeParseError as exc:
            assert exc.diagnostics
            assert str(exc)
        else:
            raise AssertionError("应当抛 ComtradeParseError")


def test_garbage_dat_content_does_not_crash():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = _standard_case(tmp, "garbage")
        rec.dat.write_bytes(b"\x00" * 1000)
        recording, diags = try_load_recording(rec.cfg)
        # 能读出数据（内容无意义），但必须留下诊断痕迹
        assert recording is not None
        assert any(d.code == Code.DAT_SIZE_MISMATCH for d in diags)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def test_cli_ascii_scaling_never_takes_effect():
    """回归：``--ascii-scaling never`` 曾经完全不生效。

    根因是命令行把字符串传进 ``ParseOptions``，而 ``AsciiScaling`` 作为
    str 混入枚举与原成员 ``is`` 不相等，判定被静默绕过。
    这里走完整的命令行路径 —— 只看 API 层会漏掉这类"接线"错误。
    """
    import io
    import json
    from contextlib import redirect_stdout

    from comtrade.__main__ import main

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp, "cli")
        rec.write_cfg(CfgSpec(
            analog=[AnalogDef(name="Ia", phase="A", unit="A", a=0.5, b=0.0, ps="P")],
            digital=[],
            data_type="ASCII",
            segments=[(4000.0, 10)],
        ))
        rec.write_ascii(np.ones((1, 10)) * 100.0, None)

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = main([str(rec.cfg), "--json", "--ascii-scaling", "never"])
        payload = json.loads(buf.getvalue())

    assert code == 0
    codes = [d["code"] for d in payload["diagnostics"]]
    assert Code.DAT_SCALING_DISABLED in codes, (
        f"--ascii-scaling never 未生效（若是 a/b 换算被照常施加，说明选项被忽略）。诊断：{codes}"
    )


# ---------------------------------------------------------------------------
# 伴随文件（.hdr / .inf）
# ---------------------------------------------------------------------------

def test_companion_file_is_loaded_when_readable():
    """.hdr 里常有人工填写的故障简报，报告模块（R-02）需要它。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = _standard_case(tmp, "companion")
        rec.cfg.with_suffix(".hdr").write_text("故障简报：A相接地", encoding="utf-8")

        recording, diags = try_load_recording(rec.cfg)

    assert recording is not None
    assert recording.meta.header_text == "故障简报：A相接地"
    assert Code.COMPANION_UNREADABLE not in [d.code for d in diags]


def test_unreadable_companion_file_is_reported_not_swallowed():
    """回归：.hdr/.inf 读取失败曾经被静默吞掉。

    静默的后果是"报告里少了故障简报"变成无法解释的现象，
    而这两个文件恰恰承载人工结论。
    """
    from comtrade import reader as reader_mod

    original = reader_mod.read_text

    def boom(path, diagnostics):
        raise OSError("模拟读取失败")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = _standard_case(tmp, "companion_bad")
        rec.cfg.with_suffix(".hdr").write_text("故障简报：A相接地", encoding="utf-8")

        reader_mod.read_text = boom
        try:
            recording, diags = try_load_recording(rec.cfg)
        finally:
            reader_mod.read_text = original

    # 伴随文件读不到不影响解析本身
    assert recording is not None
    assert recording.meta.header_text is None

    hits = [d for d in diags if d.code == Code.COMPANION_UNREADABLE]
    assert hits, "伴随文件读取失败必须留痕"
    assert hits[0].location == "companion_bad.hdr", hits[0].location


# ---------------------------------------------------------------------------
# 端到端：PS 字段为空
# ---------------------------------------------------------------------------

def test_blank_ps_field_is_flagged_end_to_end():
    """端到端回归：1999 版文件里 PS 字段为空时，必须给出警告级诊断。

    这条链路上的缺陷曾经是**静默**的：PS 字段为空被当作"1991 版没有该字段"处理，
    于是数值不乘变比（CT 400/1 就偏小 400 倍，实测 1131.4 A 变成 2.83 A），
    诊断却说"所在文件为 1991 版"（与文件头声明矛盾）、等级只有 INFO ——
    界面若按 WARNING 及以上过滤，就完全看不到这件事，
    而二次值的数值本身看起来完全正常。算法模块拿它去比一次值定值必然漏判。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp, "blank_ps")
        rec.write_cfg(CfgSpec(
            version=1999,
            analog=[AnalogDef(name="Ia", phase="A", unit="A",
                              primary=400.0, secondary=1.0, ps="")],
            digital=[],
            segments=[(4000.0, 20)],
        ))
        rec.write_binary(np.ones((1, 20)) * 100.0, None)

        recording, diags = try_load_recording(rec.cfg)

    assert recording is not None
    ia = recording.require_role(ChannelRole.IA)
    # 不猜测数值基准：保持 a/b 换算结果（400 倍偏差由人工核对后修正）
    assert ia.values[0] == 100.0
    assert ia.ratio_applied is False
    assert ia.ps is None and ia.ps_raw == ""

    hits = [d for d in diags if d.code == Code.CHN_PS_UNREADABLE]
    assert hits, f"PS 字段为空必须告警，实际诊断：{[d.code for d in diags]}"
    assert hits[0].severity.value == "warning", "必须是警告级，否则界面不会展示"
    assert "1991" not in hits[0].message, "不得再把它说成 1991 版文件"

    # 数值基准不明 ≠ 数据损坏：结果仍可用，界面标黄并提示人工确认即可
    assert recording.is_reliable is True


# ---------------------------------------------------------------------------
# 端到端：ASCII 关键列为空
# ---------------------------------------------------------------------------

def test_blank_sample_number_does_not_fake_truncation_alarm():
    """端到端回归：首行采样号为空，不再被误报成"文件被截断"。

    原实现让空采样号经 astype 静默变成 INT64_MIN，而它又被当作"首采样号"
    参与记录数校验：期望条数被算成 9.2e18，于是报 DAT-002 + DAT-004
    （"实际记录数明显少于声明值，数据可能被截断"）。这两个码都在
    不可信码集合里，一份完好的文件就这样被判成 is_reliable=False。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp, "blank_sn")
        rec.write_cfg(CfgSpec(
            version=1999,
            analog=[AnalogDef(name="Ia", phase="A", unit="A", ps="P")],
            digital=[],
            data_type="ASCII",
            segments=[(4000.0, 3)],
        ))
        # 首行采样号为空；其余两行正常
        rec.dat.write_text(",0,100\n2,250,101\n3,500,102\n", encoding="ascii")

        recording, diags = try_load_recording(rec.cfg)

    assert recording is not None
    assert recording.sample_count == 2, "无法定位的那一行应当被跳过"
    assert not (recording.sample_numbers == np.iinfo(np.int64).min).any()

    codes = [d.code for d in diags]
    assert Code.DAT_KEY_FIELD_MISSING in codes, f"必须留痕，实际诊断：{codes}"
    assert Code.DAT_SIZE_MISMATCH not in codes, "不得再把根因报成'记录数与声明不符'"
    assert Code.DAT_RECORD_SHORT not in codes, "不得再误报'数据可能被截断'"
    assert recording.is_reliable is True, "一份完好的文件不该被判为不可信"
