"""数值转换测试：a/b 换算、单位归一、PS 变比、缺失值哨兵。"""
from __future__ import annotations

import numpy as np

from comtrade.convert import convert_analog_channels, missing_sentinel
from comtrade.diagnostics import Code, DiagnosticCollector, Severity
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


def test_non_finite_values_are_marked_invalid():
    """解析期产生的 NaN（ASCII 空字段）也必须进无效掩码。

    回归 —— NaN 既不匹配缺失值哨兵，也不会被范围校验捕获（与任何数比较都是 False），
    于是 values 里明明有 NaN，invalid_mask / invalid_count / has_invalid 却都说没有：
    下游据 has_invalid 判断"该通道能否参与计算"就会被误导，算出 NaN 而无提示。
    """
    raw = np.array([[100.0, np.nan, 300.0]])
    ch = _channel(a=2.0, b=10.0)
    diag = _run([ch], raw, data_type=DataFileType.ASCII)

    assert ch.has_invalid is True
    assert ch.invalid_count == 1
    assert bool(ch.invalid_mask[1]) is True
    assert np.isnan(ch.values[1])
    assert ch.values[0] == 210.0, "其余采样点照常换算"
    assert Code.DAT_MISSING_VALUE in diag.codes()


def test_non_finite_stays_invalid_when_sentinel_masking_is_off():
    """关掉哨兵屏蔽也照样算无效点。

    NaN 本来就是无效数据，不需要"转换"，也就不存在"误伤正常值"的风险 ——
    哨兵那套占比守卫在这里不适用。
    """
    raw = np.array([[100.0, np.nan]])
    ch = _channel(a=1.0, b=0.0)
    _run([ch], raw, data_type=DataFileType.ASCII,
         options=ParseOptions(mask_missing_values=False))

    assert ch.invalid_count == 1
    assert bool(ch.invalid_mask[1]) is True


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


def test_ps_field_blank_warns_and_keeps_raw_values():
    """PS 字段存在但为空：变比信息其实在文件里，只是数值基准读不出来。

    回归 —— 早期实现把它与"1991 版没有该字段"混为一谈，于是同时出现三个问题：
    * 数值静默偏小变比倍数（CT 400/1 就差 400 倍），而二次值本身看起来正常；
    * 诊断说"所在文件为 1991 版"，与文件头声明矛盾；
    * 等级只有 INFO，界面按 WARNING 过滤时完全看不到。
    """
    raw = np.array([[1.0, 2.0]])
    ch = _channel(a=1.0, b=0.0, primary=400.0, secondary=1.0, ps=None, ps_raw="")
    diag = _run([ch], raw, data_type=DataFileType.ASCII)

    # 不猜测数值基准：保持 a/b 换算结果，但必须报警，由人工核对后决定
    np.testing.assert_allclose(ch.values, [1.0, 2.0])
    assert ch.ratio_applied is False

    entry = [d for d in diag if d.code == Code.CHN_PS_UNREADABLE]
    assert entry, f"PS 为空必须告警，实际诊断：{diag.codes()}"
    assert entry[0].severity is Severity.WARNING
    assert "1991" not in entry[0].message, "不得再说成 1991 版（文件头声明是 1999）"
    assert "400" in entry[0].message, "应给出变比，便于核对数值是否恰好差了这么多倍"
    assert Code.CHN_RATIO_MISSING not in diag.codes(), "两种情况不应混用同一个码"


def test_ps_field_absent_is_info_not_warning():
    """1991 版没有 PS 字段是正常情况 —— 只提示，不算警告。

    与上一条的区别就是"信息在不在文件里"：没有字段时无换算可做，
    按 INFO 如实告知即可，不该让界面标黄。
    """
    raw = np.array([[1.0]])
    ch = _channel(a=1.0, b=0.0, primary=None, secondary=None, ps=None, ps_raw=None)
    diag = _run([ch], raw, data_type=DataFileType.ASCII, version=ComtradeVersion.V1991)

    entry = [d for d in diag if d.code == Code.CHN_RATIO_MISSING]
    assert entry and entry[0].severity is Severity.INFO
    assert Code.CHN_PS_UNREADABLE not in diag.codes()


def test_ratio_diagnostics_are_aggregated_per_file():
    """一个根因只出一条诊断，不逐通道刷屏（README §10 第 6 条）。

    回归：原实现逐通道记诊断，7 个通道的同一问题会刷 7 条，
    既淹掉界面，也让人误以为存在多个不同的问题。
    """
    raw = np.zeros((3, 4))
    channels = [
        _channel(index=i, name=name, a=1.0, b=0.0,
                 primary=400.0, secondary=1.0, ps=None, ps_raw="")
        for i, name in enumerate(("Ia", "Ib", "Ic"))
    ]
    diag = _run(channels, raw, data_type=DataFileType.ASCII)

    hits = [d for d in diag if d.code == Code.CHN_PS_UNREADABLE]
    assert len(hits) == 1, f"应当聚合成一条诊断，实际 {len(hits)} 条"
    for name in ("Ia", "Ib", "Ic"):
        assert name in hits[0].message


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


# ---------------------------------------------------------------------------
# 解析选项取值（回归：字符串形式曾被静默忽略）
# ---------------------------------------------------------------------------

def test_ascii_scaling_accepts_plain_string():
    """命令行与配置文件传进来的是字符串，必须与枚举完全等价。

    回归：``AsciiScaling`` 是 str 混入枚举，``"never" is AsciiScaling.NEVER``
    为假，而命令行传的正是字符串 —— 修复前 ``--ascii-scaling never``
    被静默忽略，开关看起来存在但完全不生效。
    """
    assert ParseOptions(ascii_scaling="never").ascii_scaling is AsciiScaling.NEVER
    assert ParseOptions(ascii_scaling="always").ascii_scaling is AsciiScaling.ALWAYS

    raw = np.array([[1.0, 2.0]])
    for value in (AsciiScaling.NEVER, "never"):
        ch = _channel(a=10.0, b=0.0)
        _run([ch], raw, data_type=DataFileType.ASCII,
             options=ParseOptions(ascii_scaling=value))
        np.testing.assert_allclose(ch.values, [1.0, 2.0])
        assert ch.scaling_applied is False

    for value in (AsciiScaling.ALWAYS, "always"):
        ch = _channel(a=10.0, b=0.0)
        _run([ch], raw, data_type=DataFileType.ASCII,
             options=ParseOptions(ascii_scaling=value))
        np.testing.assert_allclose(ch.values, [10.0, 20.0])


def test_invalid_option_values_are_rejected():
    """写错的取值必须报错，不能静默退回默认行为。

    静默退回的后果是"上游以为换了依据、实际没换"，
    表现为同一份录波两处算出不同时长这类难查的问题。
    """
    for bad in ("alway", "", None, "NEVER1"):
        try:
            ParseOptions(ascii_scaling=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"ascii_scaling={bad!r} 应当被拒绝")

    try:
        ParseOptions(time_axis_source="timestamp")  # 少一个 s
    except ValueError:
        pass
    else:
        raise AssertionError("非法的 time_axis_source 应当被拒绝")

    assert ParseOptions(time_axis_source="timestamps").time_axis_source == "timestamps"


def test_never_scaling_reports_which_channels_were_skipped():
    """显式跳过换算时必须能看见影响范围，否则无从确认选项是否生效。

    同时这条路径不应再给 DAT-009 —— 那句提示说"已按标准施加换算、建议改用
    never"，在已经设成 never 时自相矛盾。
    """
    raw = np.array([[1.0, 2.0]])
    ch = _channel(a=10.0, b=0.0)
    diag = _run([ch], raw, data_type=DataFileType.ASCII,
                options=ParseOptions(ascii_scaling="never"))
    hits = [d for d in diag if d.code == Code.DAT_SCALING_DISABLED]
    assert hits, "跳过换算的通道必须列出"
    assert "Ia" in hits[0].message
    assert Code.DAT_ASCII_PRESCALED not in diag.codes()


# ---------------------------------------------------------------------------
# 缺失值占比守卫的适用边界
# ---------------------------------------------------------------------------

def test_sentinel_fraction_guard_applies_to_short_records():
    """短录波里"占比过高"不能被绝对次数规则覆盖。

    回归：早先的判据是"次数少 **或** 占比低即屏蔽"，等于让绝对次数规则
    压过了占比守卫 —— 16~39 点的录波即使一半数值等于哨兵（很像真实数据）
    也会被整片抹成 NaN，整通道变成"全部采样点无效"。
    """
    # 16 点里 8 个等于 1991 版哨兵 -1（占 50%）—— 判为正常数据，不屏蔽
    raw = np.zeros((1, 16))
    raw[0, :8] = -1.0
    ch = _channel()
    diag = _run([ch], raw, version=ComtradeVersion.V1991)
    assert ch.invalid_mask is None
    assert Code.DAT_MISSING_VALUE in diag.codes()

    # 同一占比放在长录波里结论一致
    raw = np.zeros((1, 1600))
    raw[0, :800] = -1.0
    ch = _channel()
    _run([ch], raw, version=ComtradeVersion.V1991)
    assert ch.invalid_mask is None

    # "少量出现"仍照常屏蔽 —— 守卫没有把功能一起关掉
    raw = np.zeros((1, 1600))
    raw[0, [5, 500]] = -1.0
    ch = _channel()
    _run([ch], raw, version=ComtradeVersion.V1991)
    assert ch.invalid_count == 2


def test_short_record_still_masks_when_sentinel_is_rare():
    """样本量不足时按出现次数判断：短录波里的个别哨兵照样屏蔽。"""
    raw = np.array([[100.0, -32768.0, 300.0]])
    ch = _channel()
    diag = _run([ch], raw)
    assert ch.invalid_count == 1
    assert Code.DAT_MISSING_VALUE in diag.codes()
