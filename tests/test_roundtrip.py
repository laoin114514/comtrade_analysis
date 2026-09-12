"""回归测试：对 tools/make_samples.py 生成的全部样例做端到端解析。

这一组是"D 岗位交付物 · 兼容性测试样例库"的自动化部分。
样例文件不存在时会自动生成，因此可重复执行、结果稳定（NF-14）。

样例覆盖矩阵
    * 版本：1991 / 1999 / 2013
    * 编码：ASCII / BINARY / BINARY32 / FLOAT32
    * 特性：定频 / 变频 / 缺失值 / 中文名 GBK / 已换算 ASCII / 加噪
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "generated"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comtrade import ChannelRole, load_recording, try_load_recording  # noqa: E402

#: 合成波形的期望值（见 tools/make_samples.py 的电气参数）
EXPECTED_I_PRE = 160.0
EXPECTED_I_POST = 800.0
EXPECTED_U_PRE = 110_000.0 / np.sqrt(3.0)
EXPECTED_U_POST = EXPECTED_U_PRE * 0.30
#: 故障采样点：定频样例在第 400 点，变频样例（前段 1000Hz）在第 100 点
FAULT_SAMPLE = {"v1999_multirate": 100}


def _ensure_fixtures() -> list[Path]:
    if not FIXTURES.exists() or not list(FIXTURES.glob("*.cfg")):
        sys.path.insert(0, str(ROOT / "tools"))
        import make_samples

        make_samples.main(["--out", str(FIXTURES)])
    return sorted(FIXTURES.glob("*.cfg"))


def _rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0


def test_all_samples_parse_without_fatal():
    for cfg in _ensure_fixtures():
        recording, diags = try_load_recording(cfg)
        assert recording is not None, f"{cfg.name} 解析失败：{[str(d) for d in diags]}"
        assert recording.sample_count > 0
        assert recording.time_axis is not None
        assert np.all(np.diff(recording.time_axis) > 0), f"{cfg.name} 时间轴非单调"


def test_all_samples_identify_three_phase_channels():
    """每个样例都应能识别出完整的三相电流与三相电压 —— 算法模块的前提。"""
    for cfg in _ensure_fixtures():
        recording = load_recording(cfg)
        roles = recording.by_role_map()
        for role in (ChannelRole.IA, ChannelRole.IB, ChannelRole.IC,
                     ChannelRole.UA, ChannelRole.UB, ChannelRole.UC):
            assert role in roles, f"{cfg.name} 缺少角色 {role.value}"


def test_all_samples_have_physically_correct_magnitudes():
    """数值必须与合成时设定的电气量一致 —— 这是"解析正确"最硬的证据。

    覆盖了 a/b 换算、单位归一化、PS 一次/二次变比三条链路。
    """
    for cfg in _ensure_fixtures():
        name = cfg.stem
        recording = load_recording(cfg)
        fault = FAULT_SAMPLE.get(name, 400)

        ia = recording.require_role(ChannelRole.IA).values
        ua = recording.require_role(ChannelRole.UA).values

        i_pre, i_post = _rms(ia[:fault]), _rms(ia[fault:])
        u_pre, u_post = _rms(ua[:fault]), _rms(ua[fault:])

        # 含噪与含缺失值的样例允许 1% 偏差
        tol = 0.01
        assert abs(i_pre - EXPECTED_I_PRE) / EXPECTED_I_PRE < tol, f"{name} Ia 故障前 {i_pre}"
        assert abs(i_post - EXPECTED_I_POST) / EXPECTED_I_POST < tol, f"{name} Ia 故障后 {i_post}"
        assert abs(u_pre - EXPECTED_U_PRE) / EXPECTED_U_PRE < tol, f"{name} Ua 故障前 {u_pre}"
        assert abs(u_post - EXPECTED_U_POST) / EXPECTED_U_POST < tol, f"{name} Ua 故障后 {u_post}"


def test_fault_current_is_five_times_load():
    """A 相接地故障的核心特征：电流突增 5 倍、电压跌落至 30%。"""
    for cfg in _ensure_fixtures():
        recording = load_recording(cfg)
        fault = FAULT_SAMPLE.get(cfg.stem, 400)
        ia = recording.require_role(ChannelRole.IA).values
        ua = recording.require_role(ChannelRole.UA).values
        assert abs(_rms(ia[fault:]) / _rms(ia[:fault]) - 5.0) < 0.05
        assert abs(_rms(ua[fault:]) / _rms(ua[:fault]) - 0.30) < 0.05


def test_unaffected_phases_stay_at_load_current():
    for cfg in _ensure_fixtures():
        recording = load_recording(cfg)
        fault = FAULT_SAMPLE.get(cfg.stem, 400)
        ib = recording.require_role(ChannelRole.IB).values
        assert abs(_rms(ib[:fault]) - EXPECTED_I_PRE) / EXPECTED_I_PRE < 0.01
        assert abs(_rms(ib[fault:]) - EXPECTED_I_PRE) / EXPECTED_I_PRE < 0.01


def test_multirate_time_axis_is_correct():
    """变频样例：endsamp 是累计编号，两段必须连续衔接。"""
    recording = load_recording(FIXTURES / "v1999_multirate.cfg")
    axis = recording.time_axis
    assert axis.size == 1600
    assert abs(axis[0]) < 1e-12
    assert abs(axis[399] - 0.399) < 1e-9  # 400 点 @1000Hz → 0 ~ 0.399
    assert abs(axis[400] - 0.400) < 1e-9  # 第二段接着 0.400
    assert abs(axis[-1] - (0.4 + 1199 / 4000.0)) < 1e-9
    # 两段的采样间隔必须不同
    assert abs((axis[1] - axis[0]) - 1e-3) < 1e-12
    assert abs((axis[401] - axis[400]) - 2.5e-4) < 1e-12


def test_missing_sample_is_masked():
    recording = load_recording(FIXTURES / "v1999_missing.cfg")
    for ch in recording.analog_channels:
        assert ch.has_invalid, f"{ch.name} 应当有无效采样点"
        indices = np.flatnonzero(ch.invalid_mask).tolist()
        assert indices == [5, 500]
        assert np.isnan(ch.values[5])


def test_chinese_gbk_names_are_decoded_and_identified():
    recording = load_recording(FIXTURES / "v1999_chinese_gbk.cfg")
    assert "变" in recording.meta.station_name
    roles = recording.by_role_map()
    assert ChannelRole.IA in roles and ChannelRole.I0 in roles


def test_ascii_small_span_sample_is_scaled_and_advisory_fires():
    """ASCII 原始计数相对声明量程偏小 —— 真实文件里很常见（SEL 装置即如此）。

    必须仍按标准施加 a/b 换算；只给出 DAT-009 提示，
    绝不能据此跳过换算（早期版本这样做，在真实样例上 100% 误判）。
    """
    recording, diags = try_load_recording(FIXTURES / "v1999_ascii_smallspan.cfg")
    assert recording is not None
    ia = recording.require_role(ChannelRole.IA).values
    assert abs(_rms(ia[:400]) - EXPECTED_I_PRE) / EXPECTED_I_PRE < 0.01
    assert any(d.code == "DAT-009" for d in diags)


def test_float32_sample_uses_identity_coefficients():
    """FLOAT32 按现场惯例在 cfg 里写 a=1,b=0，数值即工程量。"""
    recording = load_recording(FIXTURES / "v2013_float32.cfg")
    ch = recording.analog_channels[0]
    assert ch.a == 1.0 and ch.b == 0.0
    ia = recording.require_role(ChannelRole.IA).values
    assert abs(_rms(ia[:400]) - EXPECTED_I_PRE) / EXPECTED_I_PRE < 0.01


def test_1991_sample_has_no_ratio_fields():
    recording = load_recording(FIXTURES / "v1991_ascii.cfg")
    ch = recording.analog_channels[0]
    assert ch.primary is None and ch.secondary is None and ch.ps is None
    # 1991 版把变比烘焙进 a/b，因此数值仍应是一次值
    ia = recording.require_role(ChannelRole.IA).values
    assert abs(_rms(ia[:400]) - EXPECTED_I_PRE) / EXPECTED_I_PRE < 0.01


def test_2013_samples_carry_tail_fields():
    recording = load_recording(FIXTURES / "v2013_binary32.cfg")
    assert recording.meta.timezone_code == "8h00"
    assert recording.meta.time_quality_code == "B"
    assert recording.meta.leap_second == "0"


def test_repeated_parsing_is_stable():
    """同一文件解析两次结果必须一致 —— NF-14 的基础。"""
    cfg = FIXTURES / "v1999_binary.cfg"
    first = load_recording(cfg)
    second = load_recording(cfg)
    assert first.meta.cfg_sha256 == second.meta.cfg_sha256
    assert first.meta.dat_sha256 == second.meta.dat_sha256
    np.testing.assert_array_equal(
        first.require_role(ChannelRole.IA).values,
        second.require_role(ChannelRole.IA).values,
    )
    np.testing.assert_array_equal(first.time_axis, second.time_axis)
