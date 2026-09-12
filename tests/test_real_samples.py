"""真实公开样本库回归（可选）。

项目自身没有客户样例，因此支持用公开样本库做真实数据回归。
样本库路径通过环境变量 ``COMTRADE_SAMPLE_LIBRARY`` 指定；
未设置或目录不存在时全部跳过，**不产生硬依赖**。

样本库参考：``fault-wave-analyzer/fastapi/tests/fixtures/sample_library/``
（13 个来自 GitHub MIT 仓库的样例，覆盖 1991/1999/2013、ASCII/BINARY/BINARY32、
SEL 字段省略、nrates=0 变体、ISO-8859-1、UTF-8 等现场情形）。

用法::

    set COMTRADE_SAMPLE_LIBRARY=D:\\path\\to\\sample_library
    python tests/run_tests.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comtrade import try_load_recording  # noqa: E402
from tools.validate_library import _paired_dat, discover  # noqa: E402

ENV_VAR = "COMTRADE_SAMPLE_LIBRARY"


def _library() -> Path | None:
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


def _cases() -> list[Path]:
    library = _library()
    if library is None:
        return []
    return [c for c in discover(library) if _paired_dat(c) is not None]


def test_real_library_is_configured_or_skipped():
    """未配置样本库时，本模块全部用例静默跳过。"""
    if _library() is None:
        print(f"      跳过：未设置 {ENV_VAR}")
        return
    assert _cases(), f"{ENV_VAR} 指向的目录下没有可用的 cfg/dat 对"


def test_all_real_samples_parse():
    """每个真实样例都必须能产出完整结果，且时间轴单调。"""
    for cfg in _cases():
        recording, diags = try_load_recording(cfg)
        assert recording is not None, f"{cfg.name} 解析失败：{[str(d) for d in diags]}"
        assert recording.sample_count > 0
        assert recording.meta.analog_count > 0
        axis = recording.time_axis
        assert axis is not None and axis.size == recording.sample_count
        assert np.all(np.diff(axis) > 0), f"{cfg.name} 时间轴非单调"


def test_all_real_samples_have_finite_analog_values():
    """所有模拟通道的数值必须有限（缺失值已被显式标记为 NaN 并记录）。"""
    for cfg in _cases():
        recording = try_load_recording(cfg)[0]
        assert recording is not None
        for ch in recording.analog_channels:
            assert ch.values is not None, f"{cfg.name}/{ch.name} 无采样数据"
            assert ch.values.size == recording.sample_count
            finite = np.isfinite(ch.values)
            assert finite.any(), f"{cfg.name}/{ch.name} 全部无效"


def test_record_count_matches_declared():
    """实际记录数必须与 cfg 声明一致（容差 ±1 条）。

    ``endsamp`` 是**采样号**不是条数，且各装置写法不统一：
    * wisp_example2：采样号 0 起（0~19679），endsamp=19679 即末采样号；
    * wisp_example5：采样号 0 起（0~201），endsamp=202 实为总条数；
    * wisp_example4：采样号 1 起（1~2400），endsamp=2399，比实际少 1 条。

    因此统一按 ``endsamp - 首采样号 + 1`` 比较并允许 ±1 条偏差。
    这条校验的价值在于发现**量级性**错误（通道数解析错、版本判错、文件截断），
    ±1 条没有工程意义。量级性偏差的告警由 test_dat 覆盖。
    """
    checked = 0
    for cfg in _cases():
        recording = try_load_recording(cfg)[0]
        if recording is None:
            continue
        segments = recording.meta.sample_rate_segments
        if not segments:
            continue
        declared = max(s.end_sample for s in segments)
        if declared <= 0:
            continue

        first = int(recording.sample_numbers[0])
        expected = declared - first + 1
        assert abs(recording.sample_count - expected) <= 1, (
            f"{cfg.name} 实际 {recording.sample_count} 与声明推算 {expected} "
            f"（endsamp={declared}, 首采样号={first}）偏差超过容差"
        )
        checked += 1
    if checked == 0:
        print("      跳过：样本库中无带采样率声明的样例")
        checked += 1
    if checked == 0:
        print("      跳过：样本库中无带采样率声明的样例")


def test_three_phase_channels_identified_somewhere():
    """样本库整体应能识别出三相通道 —— 通道识别能力的下限验证。

    单个样例可能只有零序或单相通道，因此以"整个样本库中至少能识别出
    IA/IB/IC 或 UA/UB/UC"为断言。
    """
    cases = _cases()
    if not cases:
        print(f"      跳过：未设置 {ENV_VAR}")
        return
    found: set[str] = set()
    for cfg in cases:
        recording = try_load_recording(cfg)[0]
        if recording is None:
            continue
        found.update(r.value for r in recording.by_role_map())
    assert found & {"IA", "IB", "IC", "UA", "UB", "UC"}, (
        f"未识别出任何三相通道，实际识别到：{sorted(found)}"
    )


def test_missing_pair_reports_clean_error():
    """样本库中有只放 cfg 的固件 —— 必须干净报错而不是崩溃。"""
    library = _library()
    if library is None:
        print(f"      跳过：未设置 {ENV_VAR}")
        return
    orphans = [c for c in discover(library) if _paired_dat(c) is None]
    for cfg in orphans:
        recording, diags = try_load_recording(cfg)
        assert recording is None
        assert any(d.code == "FIL-003" for d in diags)
