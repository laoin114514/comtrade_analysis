"""dat 解析器测试：四种编码、记录布局、开关量位序、大小校验。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from comtrade.dat_parser import parse_dat, record_size
from comtrade.diagnostics import Code, DiagnosticCollector, ParseAbort, Severity
from comtrade.models import ComtradeVersion, DataFileType

from tests._util import TempRecording


def _parse(tmpdir: Path, analog, digital, *, data_type="BINARY", version=ComtradeVersion.V1999,
           declared=None, name="t"):
    rec = TempRecording(tmpdir, name)
    if data_type == "ASCII":
        rec.write_ascii(analog, digital)
    else:
        rec.write_binary(analog, digital, data_type=data_type)
    if declared is None:
        declared = analog.shape[1] if analog is not None else digital.shape[1]
    diag = DiagnosticCollector()
    parsed = parse_dat(
        rec.dat,
        analog_count=analog.shape[0] if analog is not None else 0,
        digital_count=digital.shape[0] if digital is not None else 0,
        data_type=DataFileType(data_type),
        declared_count=declared,
        version=version,
        diagnostics=diag,
    )
    return parsed, diag


# ---------------------------------------------------------------------------
# 记录长度
# ---------------------------------------------------------------------------

def test_record_size_formula():
    """记录长度 = 8 + 每模拟量字节数 × A + 2 × ceil(D/16)。"""
    assert record_size(7, 4, DataFileType.BINARY) == 8 + 14 + 2
    assert record_size(7, 16, DataFileType.BINARY) == 8 + 14 + 2
    assert record_size(7, 17, DataFileType.BINARY) == 8 + 14 + 4
    assert record_size(7, 0, DataFileType.BINARY) == 8 + 14
    assert record_size(3, 4, DataFileType.BINARY32) == 8 + 12 + 2
    assert record_size(3, 4, DataFileType.FLOAT32) == 8 + 12 + 2
    assert record_size(3, 4, DataFileType.ASCII) == 0


# ---------------------------------------------------------------------------
# 二进制
# ---------------------------------------------------------------------------

def test_binary_roundtrip_values():
    with tempfile.TemporaryDirectory() as td:
        analog = np.array([[100, -200, 300], [400, -500, 600]], dtype=np.int32)
        digital = np.zeros((2, 3), dtype=bool)
        parsed, _ = _parse(Path(td), analog, digital)
        np.testing.assert_array_equal(parsed.analog_raw, analog)
        assert parsed.record_count == 3


def test_binary_digital_bit_order():
    """开关量每 16 个打包进一个 uint16，最低位对应该组第 1 个通道。

    这是最容易出错的地方：开关量不是一通道一字节。
    用一个"只有一个通道为真"的矩阵逐个验证每个位的位置。
    """
    n = 3
    d_count = 4
    with tempfile.TemporaryDirectory() as td:
        for target in range(d_count):
            digital = np.zeros((d_count, n), dtype=bool)
            digital[target, :] = True
            parsed, _ = _parse(Path(td), np.zeros((1, n), dtype=np.int32), digital)
            for ch in range(d_count):
                expected = ch == target
                assert bool(parsed.digital[ch, 0]) is expected, (
                    f"通道 {target} 为真时，通道 {ch} 的状态应为 {expected}"
                )


def test_binary_digital_packing_across_word_boundary():
    """超过 16 个开关量时要用第二个字，位序同样从 LSB 开始。"""
    n = 2
    d_count = 20
    with tempfile.TemporaryDirectory() as td:
        digital = np.zeros((d_count, n), dtype=bool)
        digital[16, :] = True  # 第 17 个通道 → 第二个字的 bit 0
        parsed, _ = _parse(Path(td), np.zeros((1, n), dtype=np.int32), digital)
        assert bool(parsed.digital[16, 0]) is True
        assert bool(parsed.digital[15, 0]) is False
        assert bool(parsed.digital[17, 0]) is False


def test_binary32_and_float32_layout():
    for dtype, values in (
        ("BINARY32", np.array([[100000, -200000]], dtype=np.int64)),
        ("FLOAT32", np.array([[1.5, -2.25]], dtype=np.float64)),
    ):
        with tempfile.TemporaryDirectory() as td:
            parsed, _ = _parse(Path(td), values, np.zeros((1, 2), dtype=bool), data_type=dtype)
            np.testing.assert_allclose(parsed.analog_raw[0], values[0], rtol=1e-6)


def test_binary_size_mismatch_is_reported():
    """按记录长度推算的点数与 cfg 声明不符时必须报警告 —— 这是发现
    "通道数解析错/版本判错/文件损坏"最省力的手段。"""
    with tempfile.TemporaryDirectory() as td:
        analog = np.zeros((2, 10), dtype=np.int32)
        parsed, diag = _parse(Path(td), analog, np.zeros((1, 10), dtype=bool), declared=999)
        assert Code.DAT_SIZE_MISMATCH in diag.codes()
        assert parsed.record_count == 10


def test_binary_trailing_bytes_are_ignored():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        analog = np.zeros((2, 5), dtype=np.int32)
        rec = TempRecording(tmp)
        rec.write_binary(analog, np.zeros((1, 5), dtype=bool))
        with rec.dat.open("ab") as fh:
            fh.write(b"\r\n")  # 模拟文件末尾多余的换行
        diag = DiagnosticCollector()
        parsed = parse_dat(
            rec.dat, analog_count=2, digital_count=1, data_type=DataFileType.BINARY,
            declared_count=5, version=ComtradeVersion.V1999, diagnostics=diag,
        )
        assert parsed.record_count == 5
        assert Code.DAT_TRAILING_BYTES in diag.codes()


def test_binary_too_small_is_fatal():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "t.dat"
        path.write_bytes(b"\x01\x02\x03")
        diag = DiagnosticCollector()
        try:
            parse_dat(
                path, analog_count=7, digital_count=4, data_type=DataFileType.BINARY,
                declared_count=100, version=ComtradeVersion.V1999, diagnostics=diag,
            )
        except ParseAbort:
            pass
        else:
            raise AssertionError("不足一条记录应当中断")
        assert Code.DAT_EMPTY in diag.codes()


# ---------------------------------------------------------------------------
# ASCII
# ---------------------------------------------------------------------------

def test_ascii_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        analog = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        digital = np.array([[True, False, True], [False, True, False]])
        parsed, _ = _parse(Path(td), analog, digital, data_type="ASCII")
        np.testing.assert_allclose(parsed.analog_raw, analog)
        np.testing.assert_array_equal(parsed.digital, digital)


def test_ascii_digital_not_packed():
    """ASCII 里开关量是一列一个通道，不打包。"""
    with tempfile.TemporaryDirectory() as td:
        analog = np.zeros((1, 4))
        digital = np.array([[True, True, False, False]])
        parsed, _ = _parse(Path(td), analog, digital, data_type="ASCII")
        np.testing.assert_array_equal(parsed.digital[0], [True, True, False, False])


def test_ascii_column_mismatch_falls_back_to_tolerant_parse():
    """列数不符时走容错解析，跳过坏行并记录诊断，而不是整份文件失败。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp)
        rec.dat.write_text(
            "1,0,10,20,0\n"
            "2,250,11,21,0\n"
            "3,500,12\n"  # 坏行：列数不足
            "4,750,13,23,1\n",
            encoding="ascii",
        )
        diag = DiagnosticCollector()
        parsed = parse_dat(
            rec.dat, analog_count=2, digital_count=1, data_type=DataFileType.ASCII,
            declared_count=4, version=ComtradeVersion.V1999, diagnostics=diag,
        )
        assert parsed.record_count == 3
        assert Code.DAT_COLUMN_MISMATCH in diag.codes()


def test_ascii_all_rows_column_mismatch_fatal_names_root_cause():
    """所有行列数都不符时，致命诊断必须直接说出"cfg 声明与 dat 列数不符"。

    回归 —— 原实现先警告"列数不符"、再报 DAT-001"数据文件中没有可解析的采样行"，
    而实际那 1600 行数据完好无损。使用者会顺着 DAT-001 去怀疑 dat 损坏，
    真正的原因（cfg 多声明了一个通道）被埋在两条互相矛盾的信息后面。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp)
        # cfg 侧声明 2 模拟 + 1 开关 = 期望 5 列；dat 实际每行 4 列
        rec.dat.write_text(
            "1,0,10,20\n"
            "2,250,11,21\n",
            encoding="ascii",
        )
        diag = DiagnosticCollector()
        try:
            parse_dat(
                rec.dat, analog_count=2, digital_count=1, data_type=DataFileType.ASCII,
                declared_count=2, version=ComtradeVersion.V1999, diagnostics=diag,
            )
        except ParseAbort:
            pass
        else:
            raise AssertionError("所有行列数都不符应当中断")

    assert Code.DAT_COLUMN_MISMATCH in diag.codes()
    assert Code.DAT_EMPTY not in diag.codes(), (
        "不得再把根因报成'没有可解析的采样行'——数据行本身是可解析的"
    )
    fatal = [d for d in diag if d.severity is Severity.FATAL]
    assert fatal and fatal[0].code == Code.DAT_COLUMN_MISMATCH
    # 实际列数与声明列数都要出现在文案里，否则使用者无法判断差在哪
    assert "4" in fatal[0].message and "5" in fatal[0].message
    assert fatal[0].location, "致命诊断必须带定位信息"


def test_ascii_empty_key_field_is_skipped_with_diagnostic():
    """采样号/时标列为空时必须显式判定，不能让 astype 静默转成 INT64_MIN。

    回归 —— 原实现把空字段统一转成 NaN，再对这两列做 ``astype(np.int64)``：
    NaN 会静默变成 -9223372036854775808（只留一个 numpy RuntimeWarning，
    应用层通常关掉了），采样号随之失去意义；空值落在首行还会让记录数校验
    推算出天文数字的期望条数，误报"文件可能被截断"。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp)
        rec.dat.write_text(
            "1,0,10,20,0\n"
            "2,250,11,21,0\n"
            ",500,12,22,0\n"  # 采样号列为空
            "4,750,13,23,0\n",
            encoding="ascii",
        )
        diag = DiagnosticCollector()
        parsed = parse_dat(
            rec.dat, analog_count=2, digital_count=1, data_type=DataFileType.ASCII,
            declared_count=4, version=ComtradeVersion.V1999, diagnostics=diag,
        )

        assert parsed.record_count == 3, "无法定位的行应当被跳过"
        assert not (parsed.sample_numbers == np.iinfo(np.int64).min).any(), (
            "INT64_MIN 说明 NaN 又被 astype 静默转换了"
        )
        entry = [d for d in diag if d.code == Code.DAT_KEY_FIELD_MISSING]
        assert entry, f"跳过行必须留痕，实际诊断：{diag.codes()}"
        assert entry[0].severity is Severity.WARNING


def test_ascii_nan_literal_in_key_column_is_rejected():
    """除空字段外，"nan"/"inf" 字面量也不能进这两列（float() 能解析但不有限）。

    这条走的是快路径让位的分支：np.loadtxt 能正常读出含 nan 的矩阵，
    若直接交给 _assemble_ascii，nan 同样会被 astype 静默转成 INT64_MIN。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp)
        rec.dat.write_text(
            "1,0,10,20,0\n"
            "2,nan,11,21,0\n"  # 时标是字面量 nan
            "3,500,12,22,0\n",
            encoding="ascii",
        )
        diag = DiagnosticCollector()
        parsed = parse_dat(
            rec.dat, analog_count=2, digital_count=1, data_type=DataFileType.ASCII,
            declared_count=3, version=ComtradeVersion.V1999, diagnostics=diag,
        )

        assert parsed.record_count == 2
        assert not np.isnan(parsed.timestamps.astype(np.float64)).any()
        assert Code.DAT_KEY_FIELD_MISSING in diag.codes()


def test_ascii_all_key_fields_missing_is_fatal():
    """采样号/时标整列都为空时一个点也定位不了 —— 必须致命中断并说明原因。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp)
        rec.dat.write_text("1,,10,20,0\n2,,11,21,0\n", encoding="ascii")
        diag = DiagnosticCollector()
        try:
            parse_dat(
                rec.dat, analog_count=2, digital_count=1, data_type=DataFileType.ASCII,
                declared_count=2, version=ComtradeVersion.V1999, diagnostics=diag,
            )
        except ParseAbort:
            pass
        else:
            raise AssertionError("关键列整列为空应当中断")

    fatal = [d for d in diag if d.severity is Severity.FATAL]
    assert fatal and fatal[0].code == Code.DAT_KEY_FIELD_MISSING
    assert Code.DAT_EMPTY not in diag.codes(), "不得报成'没有可解析的采样行'"


def test_assemble_ascii_guard_rejects_non_finite_key_columns():
    """astype 是静默转换的地雷，解析层最后一道守卫必须拦住它。

    上游两条路径（快路径让位、容错解析逐行跳过）都已拦掉这类输入，
    因此只能直接调用 ``_assemble_ascii`` 验证守卫本身：将来若有新路径往这两列
    塞进非有限值，必须立刻以"不应出现的缺陷"形式暴露，而不是变成垃圾值流向下游。
    """
    from comtrade.dat_parser import _assemble_ascii

    data = np.array([[1.0, 0.0, 10.0], [2.0, np.nan, 11.0]])
    diag = DiagnosticCollector()
    try:
        _assemble_ascii(
            data, analog_count=1, digital_count=0, declared_count=2,
            diagnostics=diag, location="t.dat",
        )
    except ParseAbort:
        pass
    else:
        raise AssertionError("关键列含非有限值应当中断")

    assert Code.SYS_UNEXPECTED_ERROR in diag.codes()


def test_ascii_empty_field_becomes_nan():
    """1991 版用空字段表示"该点未采到"。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp)
        rec.dat.write_text("1,0,10,20,0\n2,250,,21,0\n", encoding="ascii")
        diag = DiagnosticCollector()
        parsed = parse_dat(
            rec.dat, analog_count=2, digital_count=1, data_type=DataFileType.ASCII,
            declared_count=2, version=ComtradeVersion.V1991, diagnostics=diag,
        )
        assert parsed.record_count == 2
        assert np.isnan(parsed.analog_raw[0, 1])


def test_ascii_garbage_lines_are_skipped():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        rec = TempRecording(tmp)
        rec.dat.write_text("1,0,10,20,0\nGARBAGE LINE\n2,250,11,21,0\n", encoding="ascii")
        diag = DiagnosticCollector()
        parsed = parse_dat(
            rec.dat, analog_count=2, digital_count=1, data_type=DataFileType.ASCII,
            declared_count=2, version=ComtradeVersion.V1999, diagnostics=diag,
        )
        assert parsed.record_count == 2


def test_empty_dat_is_fatal():
    for dtype in ("ASCII", "BINARY"):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "t.dat"
            path.write_bytes(b"")
            diag = DiagnosticCollector()
            try:
                parse_dat(
                    path, analog_count=1, digital_count=0, data_type=DataFileType(dtype),
                    declared_count=10, version=ComtradeVersion.V1999, diagnostics=diag,
                )
            except ParseAbort:
                pass
            else:
                raise AssertionError("空 dat 应当中断")
            assert Code.DAT_EMPTY in diag.codes()


# ---------------------------------------------------------------------------
# 诊断定位信息
# ---------------------------------------------------------------------------

def test_ascii_record_count_mismatch_carries_location():
    """回归：ASCII 路径的 DAT-002/DAT-004 曾经丢失文件名。

    记录数不符是最有价值的诊断之一，它的价值就在于指出"哪个文件对不上"；
    没有定位信息等于废掉一半，而 ASCII 恰恰是最容易被手工编辑或截断的格式。
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, diag = _parse(tmp, np.ones((1, 3)), None, data_type="ASCII", declared=10)

        hits = [d for d in diag if d.code == Code.DAT_SIZE_MISMATCH]
        assert hits, "记录数明显不符应当报警"
        assert hits[0].location == "t.dat", f"定位信息缺失：{hits[0].location!r}"

        short = [d for d in diag if d.code == Code.DAT_RECORD_SHORT]
        assert short and short[0].location == "t.dat"


def test_binary_record_count_mismatch_carries_location():
    """二进制路径同时校验，防止两边行为再次分叉。"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _, diag = _parse(tmp, np.ones((1, 3)), None, data_type="BINARY", declared=10)

        hits = [d for d in diag if d.code == Code.DAT_SIZE_MISMATCH]
        assert hits and hits[0].location == "t.dat"
