"""单位归一化与通道角色识别的测试。"""
from __future__ import annotations

from comtrade.channels import identify, normalize_name
from comtrade.models import ChannelRole
from comtrade.units import parse_unit


# ---------------------------------------------------------------------------
# 单位
# ---------------------------------------------------------------------------

def test_unit_recognizes_common_units():
    assert parse_unit("V").canonical == "V"
    assert parse_unit("V").scale == 1.0
    assert parse_unit("kV").canonical == "V"
    assert parse_unit("kV").scale == 1000.0
    assert parse_unit("A").canonical == "A"
    assert parse_unit("kA").canonical == "A"
    assert parse_unit("kA").scale == 1000.0
    assert parse_unit("mA").scale == 1e-3


def test_unit_handles_case_and_noise():
    """现场写法五花八门：大小写混写、带括号、带空格。"""
    for raw in ("KV", "kv", " kV ", "(kV)", "kV.", "kV一次"):
        info = parse_unit(raw)
        assert info.canonical == "V", raw
        assert info.scale == 1000.0, raw


def test_unit_empty_is_unrecognized():
    info = parse_unit("")
    assert info.recognized is False
    assert info.scale == 1.0
    assert parse_unit("   ").recognized is False


def test_unit_unknown_is_passed_through():
    info = parse_unit("W")
    assert info.recognized is True
    assert info.kind == "power"
    assert info.scale == 1.0


def test_unit_kind_classification():
    assert parse_unit("kV").kind == "voltage"
    assert parse_unit("A").kind == "current"
    assert parse_unit("Hz").kind == "frequency"


#: SI 词头里 M(10^6) 与 m(10^-3) 只差大小写，µ/μ 又容易被当排版字符删掉。
#: 这张表按 SI 的规范写法逐个核对 —— 期望值写死，对不上就报错。
_SI_SCALE_EXPECTED: dict[str, float] = {
    # 电压
    "V": 1.0, "mV": 1e-3, "kV": 1e3, "MV": 1e6,
    "uV": 1e-6, "µV": 1e-6, "μV": 1e-6,          # ASCII u / U+00B5 / U+03BC
    # 电流
    "A": 1.0, "mA": 1e-3, "kA": 1e3, "MA": 1e6,
    "uA": 1e-6, "µA": 1e-6, "μA": 1e-6,
    # 功率
    "W": 1.0, "mW": 1e-3, "kW": 1e3, "MW": 1e6,
    "var": 1.0, "kvar": 1e3, "Mvar": 1e6,
    "VA": 1.0, "kVA": 1e3, "MVA": 1e6,
    # 大小写无关的写法（KV/KA 这类现场常见）
    "KV": 1e3, "KA": 1e3, "kv": 1e3, "ka": 1e3,
    "kW一次": 1e3, "(kV)": 1e3, " kV ": 1e3,
}


def test_unit_si_case_is_significant():
    """回归：大小写有语义的写法必须按 SI 解析，不能被小写化抹平。

    原实现先 lower() 再查表，于是 MV（兆伏）撞上 mV（毫伏）、MA 撞上 mA，
    静默差 10^9 倍；µV/µA 里的 µ 被清洗正则当排版字符删掉，退化成 V/A，差 10^6 倍。
    两者都检测不到的原因是 recognized 仍为 True —— 一条诊断都不会发出。
    """
    wrong: list[str] = []
    for raw, expected in _SI_SCALE_EXPECTED.items():
        info = parse_unit(raw)
        if not info.recognized or info.scale != expected:
            wrong.append(
                f"{raw}: scale={info.scale:g} recognized={info.recognized}（应为 {expected:g}）"
            )
    assert not wrong, "以下写法的倍率与 SI 不一致：" + "；".join(wrong)


def test_unit_milli_and_mega_never_collide():
    """同一字母的大小写两义必须产生不同的倍率 —— 撞在一起就是 10^9 倍偏差。"""
    for milli, mega in (("mV", "MV"), ("mA", "MA"), ("mW", "MW")):
        low, high = parse_unit(milli), parse_unit(mega)
        assert low.recognized and high.recognized, f"{milli}/{mega} 都应识别"
        assert high.scale / low.scale == 1e9, (
            f"{milli}({low.scale:g}) 与 {mega}({high.scale:g}) 的倍率关系不对"
        )


def test_unit_micro_sign_is_not_stripped():
    """µ/μ 参与语义，不能被当排版字符丢掉（丢掉会让 µV 退化成 V，差 10^6 倍）。"""
    for raw in ("uV", "µV", "μV", "uA", "µA", "μA"):
        info = parse_unit(raw)
        assert info.recognized is True, raw
        assert info.scale == 1e-6, f"{raw} 的倍率应为 1e-6，实际 {info.scale:g}"


def test_unit_unresolvable_spellings_fall_back_to_unrecognized():
    """确实无法判定的写法宁可标"未识别"（不缩放 + CHN-003），也不要猜。

    与通道角色识别同一条原则："宁可标不出来让用户手工映射，也不要标错。"
    """
    for raw in ("mva", "毫伏", "微伏", "兆伏", "kWh"):
        info = parse_unit(raw)
        assert info.recognized is False, (
            f"{raw} 无法判定意图，应当落到未识别分支；实际 scale={info.scale:g}"
        )
        assert info.scale == 1.0, "未识别时不得缩放"


def test_unit_sloppy_mv_ma_mw_keep_documented_reading():
    """全小写/随意大小写的 mv、ma、mw 保留"现场最可能"的取值。

    这是**刻意保留的兼容行为**，不是 SI 判定：这三个写法在现场很常见，
    改成"未识别"会让原本按毫伏/毫安正确处理的通道反过来完全不缩放（差 1000 倍），
    风险比"把兆当成毫"更大。取值依据见 units._UNIT_MAP 的注释。
    大小写规范的写法由 _CASE_SIGNIFICANT_UNITS 精确判定，不受本策略影响。
    """
    assert parse_unit("mv").scale == 1e-3
    assert parse_unit("Mv").scale == 1e-3
    assert parse_unit("ma").scale == 1e-3
    assert parse_unit("mw").scale == 1e6  # 功率取兆：毫瓦在电力录波里没有意义

    # 与规范写法区分开：这正是本次修好的部分
    assert parse_unit("MV").scale == 1e6
    assert parse_unit("MA").scale == 1e6
    assert parse_unit("mW").scale == 1e-3


# ---------------------------------------------------------------------------
# 通道角色识别
# ---------------------------------------------------------------------------

def test_role_from_unit_and_phase_field():
    """结构化字段（uu + ph）是最可靠的依据，置信度应为满值。"""
    r = identify("任意名字", "A", "current")
    assert r.role is ChannelRole.IA
    assert r.confidence == 1.0
    assert r.source == "unit+phase"

    r = identify("xx", "C", "voltage")
    assert r.role is ChannelRole.UC


def test_role_from_common_names():
    """常见命名习惯都能识别（英文、IEC、中文）。"""
    cases = [
        ("Ia", "A", "current", ChannelRole.IA),
        ("IA", "", "current", ChannelRole.IA),
        ("I a", "", "current", ChannelRole.IA),
        ("Ia1", "", "current", ChannelRole.IA),
        ("IL1", "", "current", ChannelRole.IA),
        ("Ib", "", "current", ChannelRole.IB),
        ("IL2", "", "current", ChannelRole.IB),
        ("Ic", "", "current", ChannelRole.IC),
        ("IL3", "", "current", ChannelRole.IC),
        ("Ua", "", "voltage", ChannelRole.UA),
        ("UA", "", "voltage", ChannelRole.UA),
        ("Uan", "", "voltage", ChannelRole.UA),
        ("UL1", "", "voltage", ChannelRole.UA),
        ("Ub", "", "voltage", ChannelRole.UB),
        ("A相电流", "", "current", ChannelRole.IA),
        ("B相电流", "B", "current", ChannelRole.IB),
        ("A相电压", "", "voltage", ChannelRole.UA),
    ]
    for name, phase, kind, expected in cases:
        r = identify(name, phase, kind)
        assert r.role is expected, f"{name!r} -> {r.role}，期望 {expected}"


def test_role_zero_sequence():
    """零序通道名的各种写法。"""
    for name in ("3I0", "I0", "i0", "3I0A", "零序电流", "零序I", "中性线电流"):
        r = identify(name, "", "current")
        assert r.role is ChannelRole.I0, name
    for name in ("3U0", "U0", "零序电压", "开口三角电压"):
        r = identify(name, "", "voltage")
        assert r.role is ChannelRole.U0, name


def test_role_line_voltage():
    assert identify("Uab", "", "voltage").role is ChannelRole.UAB
    assert identify("Ubc", "", "voltage").role is ChannelRole.UBC
    assert identify("Uca", "", "voltage").role is ChannelRole.UCA


def test_role_unknown_is_not_guessed():
    """识别不出来必须返回 UNKNOWN，不能猜 —— 猜错比不识别危害大得多。"""
    for name in ("通道1", "CH1", "", "备用", "I1", "I2", "U1"):
        r = identify(name, "", "")
        assert r.role is ChannelRole.UNKNOWN, name
        assert r.confidence == 0.0


def test_role_quantity_from_name_when_unit_missing():
    """单位字段为空时靠名称推断，置信度应低于结构化字段。"""
    r = identify("Ia", "", "")
    assert r.role is ChannelRole.IA
    assert 0.0 < r.confidence < 1.0


def test_normalize_name_strips_separators_and_fullwidth():
    assert normalize_name("I a") == "ia"
    assert normalize_name("I_A") == "ia"
    assert normalize_name("Ia-1") == "ia1"
    assert normalize_name("Ａ相电流") == "a相电流"  # 全角 A


# ---------------------------------------------------------------------------
# 运行入口（无 pytest 时也能跑）
# ---------------------------------------------------------------------------

def _run_all() -> tuple[int, int]:
    from tests.run_tests import run_module

    return run_module(__name__)

def test_role_phase_suffix_variants():
    """相别后面可跟任意后缀（绕组/支路标识），真实样例里大量出现。

    实测公开样例：IAX/IBX/ICX/IAY/IBY/ICY/IAT/IBT/ICT（不同绕组的三相电流）、
    VA(kV)/VB(kV)/VC(kV)（单位写在名称里的相电压）。
    """
    cases = [
        ("IAX", ChannelRole.IA), ("IBX", ChannelRole.IB), ("ICX", ChannelRole.IC),
        ("IAY", ChannelRole.IA), ("IBY", ChannelRole.IB), ("ICY", ChannelRole.IC),
        ("IAT", ChannelRole.IA), ("IBT", ChannelRole.IB), ("ICT", ChannelRole.IC),
        ("VA(kV)", ChannelRole.UA), ("VB(kV)", ChannelRole.UB), ("VC(kV)", ChannelRole.UC),
        ("Ia1", ChannelRole.IA), ("Uan", ChannelRole.UA),
    ]
    for name, expected in cases:
        r = identify(name, "", "current" if expected.value.startswith("I") else "voltage")
        assert r.role is expected, f"{name!r} -> {r.role}，期望 {expected}"


def test_role_phase_to_phase_is_not_guessed():
    """相别后面紧跟另一个相别字母 → 相间量，含义不确定，不猜。"""
    for name in ("IAB", "IBC", "ICA", "UABX"):
        assert identify(name, "", "").role is ChannelRole.UNKNOWN, name


def test_role_zero_sequence_voltage_aliases():
    """部分厂家用 Vo 表示零序电压。"""
    for name in ("Vo", "V0", "Uo", "UN"):
        assert identify(name, "", "voltage").role is ChannelRole.U0, name


def test_duplicate_roles_are_reported():
    """多个通道占用同一角色时必须告警。

    真实样例（SEL-651R 趋势记录）里同时存在 IARMS / SDIA / SDIAREF / dA
    四组 A/B/C 电流，ph 与 uu 字段完全一样，角色识别只能都标成 IA/IB/IC。
    这不是识别错误，但下游按角色取通道会有歧义，必须让人知道。
    """
    from comtrade.channels import identify_all
    from comtrade.diagnostics import Code, DiagnosticCollector

    channels = [
        _mk("IARMS", "A"), _mk("IBRMS", "B"), _mk("ICRMS", "C"),
        _mk("SDIA", "A"), _mk("SDIB", "B"), _mk("SDIC", "C"),
    ]
    diag = DiagnosticCollector()
    identify_all(channels, diag, location="t.cfg")

    hits = [d for d in diag if d.code == Code.CHN_ROLE_DUPLICATE]
    assert hits, "应当报告角色重复"
    assert "IA 有 2 个" in hits[0].message
    assert "SDIA" in hits[0].message


def test_single_role_per_phase_is_silent():
    """每个角色只被一个通道占用时不应报警。"""
    from comtrade.channels import identify_all
    from comtrade.diagnostics import Code, DiagnosticCollector

    channels = [_mk(n, p) for n, p in (("Ia", "A"), ("Ib", "B"), ("Ic", "C"))]
    diag = DiagnosticCollector()
    identify_all(channels, diag, location="t.cfg")
    assert Code.CHN_ROLE_DUPLICATE not in diag.codes()


def _mk(name: str, phase: str):
    from comtrade.models import AnalogChannel

    return AnalogChannel(
        index=0, declared_no=1, name=name, phase_raw=phase, circuit="",
        unit_raw="A", a=1.0, b=0.0, skew_us=0.0, raw_min=-32767.0, raw_max=32767.0,
        primary=1000.0, secondary=1.0, ps="P",
    )
