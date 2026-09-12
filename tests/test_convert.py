"""数值转换测试：a/b 换算、单位归一、PS 变比、缺失值哨兵。"""
from __future__ import annotations

import numpy as np

from comtrade.convert import convert_analog_channels, missing_sentinel
from comtrade.diagnostics import Code, DiagnosticCollector
from comtrade.models import AnalogChannel, ComtradeVersion, DataFileType
from comtrade.options import AsciiScaling, ParseOptions


def _channel(**kw) -> AnalogChannel:
    base = dict(
        index=0, declared_no=1, name="Ia", phase_raw="A", circuit="LINE1",
        unit_raw="A", a=1.0, b=0.0, skew_us=0.0, raw_min=-32767.0, raw_max=32767.0,
        primary=400.0, secondary=1.0, ps="P",
    )
    base.update(kw)
    return AnalogChannel(**base)


def _run(channels, raw, *, data_type=DataFileType.BINARY, version=ComtradeVersion.V1999,
         options=None, diagnostics=None):
    diag = diagnostics or DiagnosticCollector()
    convert_analog_channels(
        channels, raw, data_type=data_type, version=version,
        diagnostics=diag, options=options or ParseOptions(),
    )
    return diag


# ---------------------------------------------------------------------------
# 缺失值哨兵
# ---------------------------------------------------------------------------

def test_sentinel_per_version_and_type():
    """缺失值哨兵随版本与编码不同，必须分别处理。"""
    assert missing_sentinel(DataFileType.ASCII, ComtradeVersion.V1991) is None
    assert missing_sentinel(DataFileType.ASCII, ComtradeVersion.V1999) == 99999.0
    assert missing_sentinel(DataFileType.BINARY, ComtradeVersion.V1991) == -1.0
    assert missing_sentinel(DataFileType.BINARY, ComtradeVersion.V1999) == -32768.0
    assert missing_sentinel(DataFileType.BINARY32, ComtradeVersion.V1999) == float(np.iinfo(np.int32).min)
    assert missing_sentinel(DataFileType.FLOAT32, ComtradeVersion.V2013) < -1e38


def test_missing_values_become_nan():
    """缺失值必须在换算前被屏蔽 —— 否则会被 a/b 换算成"看似正常"的值。"""
    raw = np.array([[100.0, -32768.0, 300.0]])
    ch = _channel(a=2.0, b=10.0)
    diag = _run([ch], raw)
    assert np.isnan(ch.values[1])
    assert ch.values[0] == 210.0
    assert ch.invalid_count == 1
    assert Code.DAT_MISSING_VALUE in diag.codes()


def test_missing_masking_can_be_disabled():
    raw = np.array([[100.0, -32768.0]])
    ch = _channel(a=1.0, b=0.0)
    _run([ch], raw, options=ParseOptions(mask_missing_values=False))
    assert not np.isnan(ch.values[1])


def test_implausible_sentinel_share_is_not_masked():
    """1991 二进制把 -1 当哨兵，若该值大量出现在正常数据里，不应误伤。"""
    raw = np.zeros((1, 100))
    raw[0, :50] = -1.0  # 50% 都是 -1，明显是正常数据
    ch = _channel()
    diag = _run([ch], raw, version=ComtradeVersion.V1991)
    assert ch.invalid_mask is None
    assert Code.DAT_MISSING_VALUE in diag.codes()


# ---------------------------------------------------------------------------
# a/b 换算
# ---------------------------------------------------------------------------

def test_binary_always_scaled():
    raw = np.array([[100.0, 200.0]], dtype=np.int32)
    ch = _channel(a=0.5, b=10.0)
    _run([ch], raw, data_type=DataFileType.BINARY)
    np.testing.assert_allclose(ch.values, [60.0, 110.0])
    assert ch.scaling_applied is True


def test_ascii_scales_by_default():
    """默认按标准施加 a/b 换算 —— 全行业一致的行为。

    早期版本会在这里"自动判定"并跳过换算，但在 11 组真实公开样例
    （186 个通道）上实测误判 10 个通道、且全部错向"跳过"，
    因此改为默认换算，跨度判定降级为纯提示。
    """
    raw = np.array([[1.0, 2.0, 3.0]])
    ch = _channel(a=0.01, b=0.0)
    _run([ch], raw, data_type=DataFileType.ASCII)
    np.testing.assert_allclose(ch.values, [0.01, 0.02, 0.03])
    assert ch.scaling_applied is True


def test_ascii_small_span_raises_advisory_but_still_scales():
    """数据跨度远小于声明量程时只给提示，不改变换算行为。"""
    raw = np.array([[1.0, 2.0, 3.0]])  # 跨度 2，声明量程跨度 65534
    ch = _channel(a=0.01, b=0.0)
    diag = _run([ch], raw, data_type=DataFileType.ASCII)
    np.testing.assert_allclose(ch.values, [0.01, 0.02, 0.03])
    assert ch.scaling_applied is True
    hits = [d for d in diag if d.code == Code.DAT_ASCII_PRESCALED]
    assert hits, "应当给出疑似已是工程量的提示"
    assert "never" in hits[0].message


def test_span_advisory_not_raised_for_binary():
    """二进制整型必然是原始计数，不该给"疑似已换算"提示。"""
    raw = np.array([[1, 2, 3]], dtype=np.int32)
    ch = _channel(a=0.01, b=0.0)
    diag = _run([ch], raw, data_type=DataFileType.BINARY)
    assert Code.DAT_ASCII_PRESCALED not in diag.codes()


def test_span_advisory_not_raised_for_identity_scaling():
    raw = np.array([[1.0, 2.0, 3.0]])
    ch = _channel(a=1.0, b=0.0)
    diag = _run([ch], raw, data_type=DataFileType.ASCII)
    assert Code.DAT_ASCII_PRESCALED not in diag.codes()


def test_ascii_scaling_override():
    raw = np.array([[1.0, 2.0]])
    ch = _channel(a=10.0, b=0.0)
    _run([ch], raw, data_type=DataFileType.ASCII,
         options=ParseOptions(ascii_scaling=AsciiScaling.ALWAYS))
    np.testing.assert_allclose(ch.values, [10.0, 20.0])

    ch = _channel(a=10.0, b=0.0)
    _run([ch], raw, data_type=DataFileType.ASCII,
         options=ParseOptions(ascii_scaling=AsciiScaling.NEVER))
    np.testing.assert_allclose(ch.values, [1.0, 2.0])


def test_a_zero_skips_scaling():
    raw = np.array([[100.0, 200.0]])
    ch = _channel(a=0.0)
    diag = _run([ch], raw)
    np.testing.assert_allclose(ch.values, [100.0, 200.0])
    assert ch.scaling_applied is False
    assert Code.DAT_SCALE_SKIPPED in diag.codes()


def test_identity_scaling_is_not_flagged():
    raw = np.array([[100.0, 200.0]])
    ch = _channel(a=1.0, b=0.0)
    diag = _run([ch], raw, data_type=DataFileType.ASCII)
    assert Code.DAT_ASCII_PRESCALED not in diag.codes()


# ---------------------------------------------------------------------------
# 单位归一化
# ---------------------------------------------------------------------------

def test_unit_normalization_kv_to_v():
    """kV 必须换算成 V —— 否则报告会写 0.4 而专工期待 400。"""
    raw = np.array([[1.0, 2.0]])
    ch = _channel(name="Ua", unit_raw="kV", a=1.0, b=0.0, primary=110000.0,
                  secondary=100.0, ps="P")
    _run([ch], raw, data_type=DataFileType.ASCII)
    np.testing.assert_allclose(ch.values, [1000.0, 2000.0])
    assert ch.unit == "V"


def test_unit_normalization_kilovolt_chinese_style():
    raw = np.array([[1.0]])
    ch = _channel(unit_raw="kA", a=1.0, b=0.0, ps="P")
    _run([ch], raw, data_type=DataFileType.ASCII)
    assert ch.values[0] == 1000.0
    assert ch.unit == "A"


# ---------------------------------------------------------------------------
# 一次值 / 二次值换算
# ---------------------------------------------------------------------------

def test_ps_secondary_converted_to_primary():
    """PS=S：文件里是二次值，需乘 primary/secondary。"""
    raw = np.array([[1.0, 2.0]])
    ch = _channel(a=1.0, b=0.0, primary=400.0, secondary=1.0, ps="S")
    _run([ch], raw, data_type=DataFileType.ASCII)
    np.testing.assert_allclose(ch.values, [400.0, 800.0])
    assert ch.ratio_applied is True


def test_ps_primary_left_alone():
    raw = np.array([[1.0, 2.0]])
    ch = _channel(a=1.0, b=0.0, primary=400.0, secondary=1.0, ps="P")
    _run([ch], raw, data_type=DataFileType.ASCII)
    np.testing.assert_allclose(ch.values, [1.0, 2.0])
    assert ch.ratio_applied is False


def test_ps_secondary_without_ratio_warns():
    raw = np.array([[1.0]])
    ch = _channel(a=1.0, b=0.0, primary=None, secondary=None, ps="S")
    diag = _run([ch], raw, data_type=DataFileType.ASCII)
    assert Code.CHN_RATIO_MISSING in diag.codes()


def test_1991_without_ratio_fields_is_reported():
    """1991 版根本没有变比字段，必须让下游知道数值不是一次值。"""
    raw = np.array([[1.0]])
    ch = _channel(a=1.0, b=0.0, primary=None, secondary=None, ps=None)
    diag = _run([ch], raw, data_type=DataFileType.ASCII, version=ComtradeVersion.V1991)
    assert Code.CHN_RATIO_MISSING in diag.codes()


def test_ratio_conversion_can_be_disabled():
    raw = np.array([[1.0]])
    ch = _channel(a=1.0, b=0.0, primary=400.0, secondary=1.0, ps="S")
    _run([ch], raw, data_type=DataFileType.ASCII,
         options=ParseOptions(apply_ratio_conversion=False))
    assert ch.values[0] == 1.0


# ---------------------------------------------------------------------------
# 取值范围与原始值保留
# ---------------------------------------------------------------------------

def test_out_of_range_is_reported_but_not_masked_by_default():
    """min/max 是文件元数据，现场常写错，按它删数据是破坏性的。"""
    raw = np.array([[100.0, 40000.0]])
    ch = _channel(a=1.0, b=0.0, raw_max=32767.0)
    diag = _run([ch], raw, data_type=DataFileType.ASCII)
    assert Code.DAT_OUT_OF_RANGE in diag.codes()
    assert not np.isnan(ch.values[1])


def test_out_of_range_masking_when_enabled():
    raw = np.array([[100.0, 40000.0]])
    ch = _channel(a=1.0, b=0.0, raw_max=32767.0)
    _run([ch], raw, data_type=DataFileType.ASCII,
         options=ParseOptions(mask_out_of_range=True))
    assert np.isnan(ch.values[1])


def test_keep_raw_values_option():
    raw = np.array([[100.0, 200.0]])
    ch = _channel(a=2.0)
    _run([ch], raw, options=ParseOptions(keep_raw_values=True))
    np.testing.assert_allclose(ch.raw_values, [100.0, 200.0])
    np.testing.assert_allclose(ch.values, [200.0, 400.0])

    ch = _channel(a=2.0)
    _run([ch], raw, options=ParseOptions(keep_raw_values=False))
    assert ch.raw_values is None


# ---------------------------------------------------------------------------
# 诊断聚合
# ---------------------------------------------------------------------------

def test_multiple_channels_produce_one_aggregated_diagnostic():
    """一个根因只报一条诊断 —— 7 个通道刷 7 条同样的警告会把界面淹掉。"""
    raw = np.tile(np.array([[1.0, 2.0, 3.0]]), (5, 1))
    channels = [_channel(index=i, name=f"CH{i}", a=0.01) for i in range(5)]
    diag = _run(channels, raw, data_type=DataFileType.ASCII)
    hits = [d for d in diag if d.code == Code.DAT_ASCII_PRESCALED]
    assert len(hits) == 1
    assert "5 个通道" in hits[0].message


def test_float32_nonidentity_scaling_is_reported_as_info():
    """FLOAT32 通常直接存工程量（a=1,b=0）；带非单位系数时要给出提示。"""
    from comtrade.diagnostics import Severity

    raw = np.array([[1.5, 2.5, 3.5]])
    ch = _channel(a=0.01)
    diag = _run([ch], raw, data_type=DataFileType.FLOAT32)
    hits = [d for d in diag if d.code == Code.DAT_FLOAT32_SCALED]
    assert hits and hits[0].severity is Severity.INFO
    np.testing.assert_allclose(ch.values, [0.015, 0.025, 0.035])
