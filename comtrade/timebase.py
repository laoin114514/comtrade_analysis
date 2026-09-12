"""时间轴生成。

合同依据
    第一条 2.(1)① "……获取采样频率及时间轴信息……"
对应需求
    P-04（采样频率识别）、P-05（根据采样间隔生成正确时间序列）、W-06（故障时间定位的基础）

两种机制与优先级
    方式一（优先）：**按采样率分段推算**。
        标准与主流实现（GSF / comtrade-rs / python-comtrade）都以采样率分段为准，
        因为它定义在 cfg 里、不依赖 dat 中逐点的时标是否可靠。
    方式二（兜底）：**按采样点时标推算**，``t = 时标 × time_mult × time_base``。
        当 ``nrates = 0``（文件不声明采样率）、采样率非正、
        或某采样点的时标为缺失值哨兵时使用。

    两者都可用时做交叉校验，不一致则记录 TIM-002 ——
    这能有效暴露"采样率段理解错误"这一类问题（例如把累计编号当成段内点数）。

时间轴定义
    ``time_axis[0] = 0``，单位为秒，相对第一个采样点。
    绝对时刻 = ``Metadata.start_time + time_axis``。
"""
from __future__ import annotations

import numpy as np

from .diagnostics import Code, DiagnosticCollector
from .models import Metadata


def build_time_axis(
    meta: Metadata,
    sample_numbers: np.ndarray,
    timestamps: np.ndarray,
    diagnostics: DiagnosticCollector,
    *,
    location: str = "",
    rate_tolerance: float = 0.05,
    source: str = "rates",
) -> np.ndarray:
    """生成时间轴（秒）。

    Args:
        meta: 元数据，提供采样率分段与时间倍数。
        sample_numbers: dat 中解析出的采样号。
        timestamps: dat 中解析出的采样时标。
        diagnostics: 诊断收集器。
        location: 诊断定位信息。
        rate_tolerance: 两种机制的一致性容差（相对值）。
        source: 优先依据，``"rates"``（采样率分段优先）或 ``"timestamps"``。

    Returns:
        形状 ``(N,)`` 的 ``float64`` 时间轴。
    """
    count = len(sample_numbers)
    if count == 0:
        return np.empty(0, dtype=np.float64)

    rate_axis = _build_from_rates(meta, sample_numbers, count, diagnostics, location)
    time_axis_from_ts = _build_from_timestamps(meta, timestamps, count, diagnostics, location)

    # ------------------------------------------------------------ 选择与校验
    if rate_axis is None and time_axis_from_ts is None:
        diagnostics.warn(
            Code.CFG_NO_SAMPLE_RATE,
            "既没有可用的采样率分段，也没有可用的采样时标，"
            "时间轴将按索引退化生成（仅能表示相对顺序）",
            location=location,
        )
        return np.arange(count, dtype=np.float64)

    if rate_axis is None:
        axis, origin = time_axis_from_ts, "采样时标"
    elif time_axis_from_ts is None:
        axis, origin = rate_axis, "采样率分段"
    elif source == "timestamps":
        # 时标优先：反映录波器实际采样时刻，与标称采样率存在微小漂移
        axis, origin = time_axis_from_ts, "采样时标（配置指定优先）"
        _cross_check(time_axis_from_ts, rate_axis, rate_tolerance, diagnostics, location)
    else:
        axis, origin = rate_axis, "采样率分段"
        _cross_check(rate_axis, time_axis_from_ts, rate_tolerance, diagnostics, location)

    _check_monotonic(axis, diagnostics, location)
    diagnostics.info("TIM-OK", f"时间轴已生成（依据：{origin}），时长 {axis[-1] - axis[0]:.6f} s")
    return axis


# ---------------------------------------------------------------------------
# 方式一：采样率分段
# ---------------------------------------------------------------------------

def _build_from_rates(
    meta: Metadata,
    sample_numbers: np.ndarray,
    count: int,
    diagnostics: DiagnosticCollector,
    location: str,
) -> np.ndarray | None:
    """按采样率分段推算时间轴。

    分段语义：``(rate, end_sample)`` 表示第 ``prev_end+1`` 到第 ``end_sample``
    个采样点（**累计编号**）按 ``rate`` 采样。
    """
    segments = [s for s in meta.sample_rate_segments if s.rate_hz > 0 and s.end_sample > 0]
    if not segments:
        return None

    _check_sample_numbers(sample_numbers, diagnostics, location)

    axis = np.zeros(count, dtype=np.float64)
    filled = 0
    elapsed = 0.0

    for seg in sorted(segments, key=lambda s: s.end_sample):
        end = min(seg.end_sample, count)
        if end <= filled:
            continue
        block = end - filled
        axis[filled:end] = elapsed + np.arange(block, dtype=np.float64) / seg.rate_hz
        elapsed += block / seg.rate_hz
        filled = end

    if filled >= count:
        return axis

    # 分段未覆盖全部采样点：用最后一段的采样率外推，并告知用户
    tail_rate = sorted(segments, key=lambda s: s.end_sample)[-1].rate_hz
    block = count - filled
    axis[filled:] = elapsed + np.arange(block, dtype=np.float64) / tail_rate
    diagnostics.warn(
        Code.CFG_SAMPLE_RATE_LINE,
        f"采样率分段只覆盖到第 {filled} 点，剩余 {block} 个采样点按最后一段的 "
        f"{tail_rate:g}Hz 外推，时间轴末端可能不准确",
        location=location,
    )
    return axis


def _check_sample_numbers(
    sample_numbers: np.ndarray, diagnostics: DiagnosticCollector, location: str
) -> None:
    """采样率分段是按"累计编号"定义的，因此采样号异常时时间轴会错位。"""
    count = len(sample_numbers)
    if count == 0:
        return
    if sample_numbers[0] != 1:
        diagnostics.warn(
            Code.TIM_SAMPLE_NO_ANOMALY,
            f"首个采样号是 {sample_numbers[0]}，标准要求从 1 开始；"
            "采样率分段按文件顺序应用，时间轴可能与实际不符",
            location=location,
        )
        return
    expected = np.arange(1, count + 1, dtype=np.int64)
    if not np.array_equal(sample_numbers, expected):
        gaps = int(np.count_nonzero(np.diff(sample_numbers) != 1))
        diagnostics.warn(
            Code.TIM_SAMPLE_NO_ANOMALY,
            f"采样号不连续（{gaps} 处跳变），采样率分段按文件顺序应用",
            location=location,
        )


# ---------------------------------------------------------------------------
# 方式二：采样时标
# ---------------------------------------------------------------------------

def _build_from_timestamps(
    meta: Metadata,
    timestamps: np.ndarray,
    count: int,
    diagnostics: DiagnosticCollector,
    location: str,
) -> np.ndarray | None:
    """按采样点时标推算时间轴。"""
    if timestamps is None or len(timestamps) != count:
        return None
    if not np.any(timestamps):
        # 时标全为 0：字段存在但无意义，不能用来建时间轴
        diagnostics.info(
            "TIM-TS-ZERO",
            "采样时标全为 0，未采用时标推算时间轴",
            location=location,
        )
        return None

    ts = timestamps.astype(np.float64)
    sentinel = _timestamp_sentinel(ts)
    if sentinel is not None:
        valid = ts != float(sentinel)
        if not valid.any():
            diagnostics.warn(
                "TIM-TS-INVALID",
                "所有采样时标均为缺失值哨兵，未采用时标推算时间轴",
                location=location,
            )
            return None
        if not valid.all():
            count_invalid = int((~valid).sum())
            diagnostics.warn(
                Code.DAT_MISSING_VALUE,
                f"有 {count_invalid} 个采样时标为缺失值哨兵，"
                "时间轴将按有效时标线性插值补齐",
                location=location,
            )
            ts = np.interp(np.arange(count), np.flatnonzero(valid), ts[valid])

    axis = (ts - ts[0]) * meta.time_mult * meta.time_base_seconds
    return axis


def _timestamp_sentinel(ts: np.ndarray) -> int | None:
    """识别时标字段的缺失值哨兵（32 位无符号的上限附近）。"""
    if ts.size == 0:
        return None
    for candidate in (0xFFFFFFFF, 0x80000000):
        if np.any(ts == float(candidate)):
            return candidate
    return None


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def _cross_check(
    rate_axis: np.ndarray,
    ts_axis: np.ndarray,
    tolerance: float,
    diagnostics: DiagnosticCollector,
    location: str,
) -> None:
    """两种机制的一致性交叉校验。"""
    span_rate = float(rate_axis[-1] - rate_axis[0])
    span_ts = float(ts_axis[-1] - ts_axis[0])
    if span_rate <= 0 or span_ts <= 0:
        return
    relative = abs(span_rate - span_ts) / max(span_rate, span_ts)
    if relative > tolerance:
        diagnostics.warn(
            Code.TIM_RATE_TS_INCONSISTENT,
            f"按采样率推算的时长 {span_rate:.6f}s 与按采样时标推算的 "
            f"{span_ts:.6f}s 相差 {relative:.2%}，已采用采样率推算结果；"
            "请检查采样率分段（endsamp 是累计编号）或 time_mult 是否正确",
            location=location,
        )


def _check_monotonic(
    axis: np.ndarray, diagnostics: DiagnosticCollector, location: str
) -> None:
    """时间轴必须单调递增，否则后续按时间定位故障会全部错位。"""
    if axis.size < 2:
        return
    diff = np.diff(axis)
    if not np.all(diff > 0):
        bad = int(np.count_nonzero(diff <= 0))
        diagnostics.warn(
            Code.TIM_NOT_MONOTONIC,
            f"时间轴存在 {bad} 处非递增（回退或跳变），时间信息不可信；"
            "波形定位与按时间切片的结果可能错误",
            location=location,
        )


__all__ = ["build_time_axis"]
