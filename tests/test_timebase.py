"""时间轴生成测试：采样率分段、时标兜底、交叉校验、单调性。"""
from __future__ import annotations

import numpy as np

from comtrade.diagnostics import Code, DiagnosticCollector
from comtrade.models import Metadata, SampleRateSegment
from comtrade.timebase import build_time_axis


def _meta(segments, *, time_mult=1.0, time_base=1e-6):
    return Metadata(
        station_name="S", device_id="D", revision_year=1999,
        version=__import__("comtrade").ComtradeVersion.V1999,
        source_cfg=None, source_dat=None,
        sample_rate_segments=[SampleRateSegment(r, e) for r, e in segments],
        time_mult=time_mult, time_base_seconds=time_base,
    )


def _build(segments, n, *, timestamps=None, time_mult=1.0, tolerance=0.05):
    diag = DiagnosticCollector()
    if timestamps is None:
        timestamps = np.zeros(n, dtype=np.int64)
    axis = build_time_axis(
        _meta(segments, time_mult=time_mult),
        np.arange(1, n + 1),
        timestamps,
        diag,
        rate_tolerance=tolerance,
    )
    return axis, diag


# ---------------------------------------------------------------------------
# 采样率分段
# ---------------------------------------------------------------------------

def test_single_rate_axis():
    axis, _ = _build([(4000.0, 400)], 400)
    assert axis[0] == 0.0
    assert abs(axis[1] - 0.00025) < 1e-12
    assert abs(axis[-1] - 399 / 4000.0) < 1e-12


def test_multirate_segments_are_continuous():
    """endsamp 是**累计编号**而不是段内点数 —— 这是最容易算错的地方。"""
    axis, _ = _build([(1000.0, 400), (4000.0, 1600)], 1600)
    # 第 1 段：0 ~ 0.399
    assert abs(axis[0] - 0.0) < 1e-12
    assert abs(axis[399] - 0.399) < 1e-12
    # 第 2 段必须从 0.400 接上，不能重新从 0 开始
    assert abs(axis[400] - 0.400) < 1e-12
    assert abs(axis[401] - 0.40025) < 1e-12
    assert abs(axis[-1] - (0.4 + 1199 / 4000.0)) < 1e-12


def test_segment_order_is_normalized():
    """分段乱序时按 endsamp 排序后再应用。"""
    axis, _ = _build([(4000.0, 1600), (1000.0, 400)], 1600)
    assert abs(axis[399] - 0.399) < 1e-12
    assert abs(axis[400] - 0.400) < 1e-12


def test_segments_not_covering_all_samples_warns_and_extrapolates():
    axis, diag = _build([(1000.0, 100)], 300)
    assert axis.size == 300
    assert Code.CFG_SAMPLE_RATE_LINE in diag.codes()
    assert abs(axis[-1] - 0.299) < 1e-9


def test_zero_sample_number_start_warns():
    diag = DiagnosticCollector()
    meta = _meta([(1000.0, 100)])
    axis = build_time_axis(meta, np.arange(0, 100), np.zeros(100, dtype=np.int64), diag)
    assert Code.TIM_SAMPLE_NO_ANOMALY in diag.codes()
    assert axis.size == 100


# ---------------------------------------------------------------------------
# 时标兜底
# ---------------------------------------------------------------------------

def test_timestamps_used_when_no_sample_rate():
    """nrates = 0 时只能靠采样时标建时间轴。"""
    ts = np.arange(0, 1000, 250, dtype=np.int64)  # 250us 步长 → 4000Hz
    axis, _ = _build([], 4, timestamps=ts)
    assert abs(axis[1] - 0.00025) < 1e-12
    assert abs(axis[-1] - 0.00075) < 1e-12


def test_time_mult_scales_timestamps():
    ts = np.arange(0, 4000, 1000, dtype=np.int64)
    axis, _ = _build([], 4, timestamps=ts, time_mult=2.0)
    assert abs(axis[1] - 0.002) < 1e-12


def test_nanosecond_base_unit():
    diag = DiagnosticCollector()
    meta = _meta([], time_base=1e-9)
    ts = np.array([0, 250_000], dtype=np.int64)  # 250us = 250000 ns
    axis = build_time_axis(meta, np.arange(1, 3), ts, diag)
    assert abs(axis[1] - 0.00025) < 1e-12


def test_all_zero_timestamps_are_rejected():
    axis, diag = _build([], 100, timestamps=np.zeros(100, dtype=np.int64))
    # 退化：按索引生成，且给出提示
    assert axis.size == 100
    assert "TIM-TS-ZERO" in diag.codes()


def test_timestamp_sentinel_is_interpolated():
    ts = np.array([0, 250, 0xFFFFFFFF, 750], dtype=np.int64)
    axis, diag = _build([], 4, timestamps=ts)
    assert np.all(np.isfinite(axis))
    assert abs(axis[2] - 0.0005) < 1e-9
    assert Code.DAT_MISSING_VALUE in diag.codes()


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def test_rate_and_timestamp_cross_check_warns_on_mismatch():
    """采样率推算与采样时标推算差异过大时必须报警 —— 这能暴露 endsamp 理解错误。"""
    ts = np.arange(0, 800, 200, dtype=np.int64)  # 200us 步长 → 5000Hz，与声明的 1000Hz 不符
    _, diag = _build([(1000.0, 4)], 4, timestamps=ts)
    assert Code.TIM_RATE_TS_INCONSISTENT in diag.codes()


def test_rate_and_timestamp_agreeing_is_silent():
    ts = np.arange(0, 1000, 250, dtype=np.int64)
    _, diag = _build([(4000.0, 4)], 4, timestamps=ts)
    assert Code.TIM_RATE_TS_INCONSISTENT not in diag.codes()


def test_non_monotonic_timestamps_are_flagged():
    ts = np.array([0, 250, 200, 750], dtype=np.int64)  # 第 3 点回退
    _, diag = _build([], 4, timestamps=ts)
    assert Code.TIM_NOT_MONOTONIC in diag.codes()


def test_empty_input_returns_empty_axis():
    diag = DiagnosticCollector()
    axis = build_time_axis(_meta([(4000.0, 0)]), np.array([]), np.array([]), diag)
    assert axis.size == 0


def test_time_axis_is_monotonic_for_normal_input():
    axis, diag = _build([(4000.0, 1000)], 1000)
    assert np.all(np.diff(axis) > 0)
    assert Code.TIM_NOT_MONOTONIC not in diag.codes()
