"""dat 解析器测试：四种编码、记录布局、开关量位序、大小校验。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from comtrade.dat_parser import parse_dat, record_size
from comtrade.diagnostics import Code, DiagnosticCollector, ParseAbort
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
