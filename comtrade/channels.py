"""通道角色识别与归一化。

合同依据
    第一条 2.(1)① "……自动识别录波通道信息……"
对应需求
    P-03（模拟量通道解析）、NF-32（正确识别主要模拟量通道）、D-04（结合序分量辅助判断的前提）
对应分工方案
    D 岗位交付物「通道映射方案」

为什么需要这一层
    同一个"A 相电流"在不同录波器里可能叫 ``Ia`` / ``IA`` / ``I a`` / ``A相电流`` /
    ``IL1`` / ``Ia1``。如果算法模块靠名称字符串去取通道，遇到任何一个新厂家都要改代码。
    这一层把名称差异一次性吸收掉，输出稳定的 :class:`~comtrade.models.ChannelRole`。

判定优先级
    1. **单位 + 相别字段** —— cfg 里 ``uu`` 与 ``ph`` 是结构化字段，最可靠。
    2. **名称模式** —— 结构化字段缺失时按命名习惯推断，置信度降低。
    3. **无法判定 → UNKNOWN** —— 宁可标不出来让用户手工映射，
       也不要标错。标错的通道会被算法模块当成 A 相电流参与故障判断，
       后果比"少一个通道"严重得多。

已知的命名冲突（刻意不做映射）
    ``I1`` / ``I2``：既可能是 IEC 的 1/2/3 号电流，也可能是"正序/负序"分量。
    两者语义完全不同，无法从名称区分，一律标为 UNKNOWN。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .diagnostics import Code, DiagnosticCollector
from .models import AnalogChannel, ChannelRole, Quantity
from .units import parse_unit

# ---------------------------------------------------------------------------
# 名称归一化
# ---------------------------------------------------------------------------

_SEPARATORS = re.compile(r"[\s\-_.()\[\]{}（）【】、,，;；/\\|]+")


def normalize_name(name: str) -> str:
    """归一化通道名称：全角转半角、去除分隔符、统一小写。

    只用于模式匹配，不改变通道对象里保存的原始名称。
    """
    text = unicodedata.normalize("NFKC", name or "")
    text = _SEPARATORS.sub("", text)
    return text.lower()


# ---------------------------------------------------------------------------
# 名称模式表
# ---------------------------------------------------------------------------

#: 零序 / 中性点电压
_ZERO_VOLTAGE_RES = (
    re.compile(r"^(?:3)?u0+[a-z0-9]*$"),  # u0 / 3u0 / u00 / 3U0A
    re.compile(r"^(?:u|v)o+\d*$"),  # uo / vo（部分厂家用 Vo 表示零序电压）
    re.compile(r"^un\d*$"),
    re.compile(r"^v_?0\d*$"),
    re.compile(r"零序.*(?:电压|压|u)"),
    re.compile(r"中性(?:点)?电压"),
    re.compile(r"开口三角"),
)

#: 零序 / 中性线电流
_ZERO_CURRENT_RES = (
    re.compile(r"^(?:3)?i0+[a-z0-9]*$"),  # i0 / 3i0 / i00 / 3I0A
    re.compile(r"^io+\d*$"),
    re.compile(r"^in\d*$"),
    re.compile(r"零序"),  # 兜底：出现"零序"但不是电压的，按电流处理
    re.compile(r"中性(?:点|线)?电流"),
)

#: 线电压（相间电压）
_LINE_VOLTAGE_RES: tuple[tuple[re.Pattern[str], ChannelRole], ...] = (
    (re.compile(r"^(?:u|v)ab\d*$"), ChannelRole.UAB),
    (re.compile(r"^(?:u|v)bc\d*$"), ChannelRole.UBC),
    (re.compile(r"^(?:u|v)ca\d*$"), ChannelRole.UCA),
    (re.compile(r"^ab线电压"), ChannelRole.UAB),
    (re.compile(r"^bc线电压"), ChannelRole.UBC),
    (re.compile(r"^ca线电压"), ChannelRole.UCA),
)

#: 相电压。
#: 后缀放宽为任意字母数字，但**紧跟相别之后的字符不能再是相别字母** ——
#: 这样 ``IAX``/``IBY``/``IAT``（不同绕组/支路的 A/B/C 相）能识别，
#: 而 ``IAB``/``IBC``（相间量，相别含义不确定）不会被误判成单相量。
#: ``(kV)`` 这类单位后缀在归一化时已被去掉，因此 ``VA(kV)`` → ``vakv`` 也能命中。
_PHASE_VOLTAGE_RES = re.compile(r"^(?:u|v|pt)([abc])(?![abc])[a-z0-9]*$")
#: IEC 习惯：UL1/UL2/UL3 对应 A/B/C
_VOLTAGE_IEC_L123 = re.compile(r"^(?:u|v)l([123])\d*$")

#: 相电流，后缀规则同相电压
_PHASE_CURRENT_RES = re.compile(r"^(?:i|ct|il)([abc])(?![abc])[a-z0-9]*$")
#: IEC 习惯：IL1/IL2/IL3 对应 A/B/C
_CURRENT_IEC_L123 = re.compile(r"^il([123])\d*$")

#: 中文相别：A相电流 / A相电压 / A相
_CN_PHASE = re.compile(r"([abc])相")
_CN_VOLTAGE = "电压"
_CN_CURRENT = "电流"

_IEC_L_MAP = {"1": "A", "2": "B", "3": "C"}

#: cfg 的 ph 字段到相别的映射
_PH_FIELD_MAP = {
    "A": "A",
    "B": "B",
    "C": "C",
    "N": "N",
    "AB": "AB",
    "BC": "BC",
    "CA": "CA",
    "ABC": "ABC",
}


@dataclass(frozen=True, slots=True)
class ChannelIdentification:
    """通道识别结果。"""

    role: ChannelRole
    quantity: Quantity
    confidence: float
    source: str
    """判定依据：``unit+phase`` / ``unit`` / ``phase+name`` / ``name`` / ``unresolved``"""


def identify(
    name: str,
    phase_field: str,
    unit_kind: str,
) -> ChannelIdentification:
    """识别通道角色。

    Args:
        name: 通道原始名称。
        phase_field: cfg 里的 ``ph`` 字段原文。
        unit_kind: :func:`comtrade.units.parse_unit` 给出的类别
            （``voltage`` / ``current`` / ``other``）。

    Returns:
        :class:`ChannelIdentification`
    """
    normalized = normalize_name(name)

    # ------------------------------------------------ 结构化字段：单位与相别
    unit_quantity = _quantity_from_kind(unit_kind)
    ph_phase = _PH_FIELD_MAP.get((phase_field or "").strip().upper())

    # ---------------------------------------------------------- 零序优先判定
    zero_role = _detect_zero_sequence(normalized, unit_quantity, ph_phase)
    if zero_role is not None:
        quantity = (
            Quantity.VOLTAGE if zero_role is ChannelRole.U0 else Quantity.CURRENT
        )
        # 零序通道的名称模式很特征化（3I0/零序电流），但 ph 字段常为 N 或空，
        # 因此名称证据是主要依据
        confidence = 0.9 if unit_quantity is not None else 0.7
        source = "name+unit" if unit_quantity is not None else "name"
        return ChannelIdentification(zero_role, quantity, confidence, source)

    # ------------------------------------------------------------ 名称推断
    name_quantity, name_phase, name_line = _parse_name(normalized)

    quantity = unit_quantity or name_quantity
    phase = ph_phase or name_phase

    if phase is None and name_line is not None:
        # 线电压名称已经给出了线别
        return ChannelIdentification(
            name_line,
            Quantity.VOLTAGE,
            0.75 if unit_quantity is not None else 0.6,
            "unit+name" if unit_quantity is not None else "name",
        )

    if quantity is None or phase is None:
        return ChannelIdentification(
            ChannelRole.UNKNOWN, Quantity.UNKNOWN, 0.0, "unresolved"
        )

    role = _compose_role(quantity, phase)
    if role is None:
        return ChannelIdentification(
            ChannelRole.UNKNOWN, Quantity.UNKNOWN, 0.0, "unresolved"
        )

    # 置信度：结构化字段越多越高
    if ph_phase is not None and unit_quantity is not None:
        confidence, source = 1.0, "unit+phase"
    elif unit_quantity is not None:
        confidence, source = 0.7, "unit+name"
    elif ph_phase is not None:
        confidence, source = 0.6, "phase+name"
    else:
        confidence, source = 0.5, "name"

    return ChannelIdentification(role, quantity, confidence, source)


def _quantity_from_kind(unit_kind: str) -> Quantity | None:
    if unit_kind == "voltage":
        return Quantity.VOLTAGE
    if unit_kind == "current":
        return Quantity.CURRENT
    return None


def _detect_zero_sequence(
    normalized: str, unit_quantity: Quantity | None, ph_phase: str | None
) -> ChannelRole | None:
    """识别零序/中性通道。"""
    if not normalized:
        return None

    # 先判电压：名称里同时出现"零序"和"电压/压"时，必须归到 U0 而不是 I0
    for pattern in _ZERO_VOLTAGE_RES:
        if pattern.search(normalized):
            return ChannelRole.U0
    for pattern in _ZERO_CURRENT_RES:
        if pattern.search(normalized):
            return ChannelRole.I0

    # ph 字段为 N 且无其他线索时，按单位判断是零序电流还是零序电压
    if ph_phase == "N" and unit_quantity in (Quantity.VOLTAGE, Quantity.CURRENT):
        # N 相在录波里通常就是中性线/零序通道；但名称里若另有相别线索则不采纳
        if not _PHASE_VOLTAGE_RES.search(normalized) and not _PHASE_CURRENT_RES.search(
            normalized
        ):
            return ChannelRole.U0 if unit_quantity is Quantity.VOLTAGE else ChannelRole.I0
    return None


def _parse_name(normalized: str) -> tuple[Quantity | None, str | None, ChannelRole | None]:
    """从名称推断 (物理量, 相别, 线电压角色)。"""
    quantity: Quantity | None = None
    phase: str | None = None
    line_role: ChannelRole | None = None

    if not normalized:
        return None, None, None

    # ------------------------------------------------------------ 线电压
    for pattern, role in _LINE_VOLTAGE_RES:
        if pattern.search(normalized):
            return Quantity.VOLTAGE, None, role

    # ------------------------------------------------------------ 英文/拼音命名
    if m := _PHASE_VOLTAGE_RES.match(normalized):
        quantity, phase = Quantity.VOLTAGE, m.group(1).upper()
    elif m := _VOLTAGE_IEC_L123.match(normalized):
        quantity, phase = Quantity.VOLTAGE, _IEC_L_MAP[m.group(1)]
    elif m := _PHASE_CURRENT_RES.match(normalized):
        quantity, phase = Quantity.CURRENT, m.group(1).upper()
    elif m := _CURRENT_IEC_L123.match(normalized):
        quantity, phase = Quantity.CURRENT, _IEC_L_MAP[m.group(1)]

    # ------------------------------------------------------------ 中文命名
    if phase is None or quantity is None:
        cn_phase = _CN_PHASE.search(normalized)
        cn_phase_letter = cn_phase.group(1).upper() if cn_phase else None
        has_u = _CN_VOLTAGE in normalized
        has_i = _CN_CURRENT in normalized

        if has_u and not has_i:
            quantity = Quantity.VOLTAGE
            phase = phase or cn_phase_letter
        elif has_i and not has_u:
            quantity = Quantity.CURRENT
            phase = phase or cn_phase_letter
        elif has_u and has_i:
            # "电流电压"这类复合名称无法确定，放弃
            return None, None, None
        elif cn_phase_letter and phase is None:
            phase = cn_phase_letter

    return quantity, phase, line_role


def _compose_role(quantity: Quantity, phase: str) -> ChannelRole | None:
    """把 (物理量, 相别) 组合成角色。"""
    if quantity is Quantity.VOLTAGE:
        return {
            "A": ChannelRole.UA,
            "B": ChannelRole.UB,
            "C": ChannelRole.UC,
            "N": ChannelRole.U0,
            "AB": ChannelRole.UAB,
            "BC": ChannelRole.UBC,
            "CA": ChannelRole.UCA,
        }.get(phase)

    if quantity is Quantity.CURRENT:
        return {
            "A": ChannelRole.IA,
            "B": ChannelRole.IB,
            "C": ChannelRole.IC,
            "N": ChannelRole.I0,
        }.get(phase)

    return None


def identify_all(
    channels: list[AnalogChannel],
    diagnostics: DiagnosticCollector,
    location: str = "",
) -> None:
    """对一批模拟通道就地执行角色识别，并汇总诊断。

    识别失败的通道统一记录一条 CHN-001，界面据此提示用户手工映射。
    """
    unresolved: list[str] = []

    for ch in channels:
        unit_info = parse_unit(ch.unit_raw)
        result = identify(ch.name, ch.phase_raw, unit_info.kind)
        ch.role = result.role
        ch.quantity = result.quantity
        ch.role_confidence = result.confidence
        ch.role_source = result.source
        if result.role in (ChannelRole.UNKNOWN, ChannelRole.OTHER):
            unresolved.append(ch.name)

    if unresolved:
        diagnostics.warn(
            Code.CHN_ROLE_UNRESOLVED,
            f"有 {len(unresolved)} 个通道的角色无法自动识别，需人工确认映射后才能参与算法计算："
            + "、".join(unresolved[:10])
            + ("…" if len(unresolved) > 10 else ""),
            location=location,
        )

    _warn_duplicate_names(channels, diagnostics, location)
    _warn_duplicate_roles(channels, diagnostics, location)


def _warn_duplicate_roles(channels, diagnostics, location: str) -> None:
    """检测"多个通道占用同一角色"。

    真实文件里很常见：一份 SEL 装置的趋势记录里同时有 ``IARMS``（有效值）、
    ``SDIA``、``SDIAREF``、``dA`` 四组 A/B/C 电流，它们的 ``ph`` 与 ``uu``
    字段完全一样，角色识别只能都标成 IA/IB/IC。

    这不是识别错误，而是文件本身有多组同相别通道。但下游按角色取通道时
    只能拿到其中一个（见 :meth:`Recording.by_role_map` 的取值规则），
    因此必须显式告警，提示人工确认该用哪一组。
    """
    from .diagnostics import Code
    from .models import ChannelRole

    by_role: dict[ChannelRole, list[str]] = {}
    for ch in channels:
        if ch.role in (ChannelRole.UNKNOWN, ChannelRole.OTHER):
            continue
        by_role.setdefault(ch.role, []).append(ch.name)

    duplicated = {role: names for role, names in by_role.items() if len(names) > 1}
    if not duplicated:
        return

    preview = "；".join(
        f"{role.value} 有 {len(names)} 个（{'/'.join(names[:4])}{'…' if len(names) > 4 else ''}）"
        for role, names in list(duplicated.items())[:4]
    )
    diagnostics.warn(
        Code.CHN_ROLE_DUPLICATE,
        f"有 {len(duplicated)} 个角色被多个通道占用，按角色取通道时会取文件顺序中的第一个，"
        f"结果可能不是期望的那组量，请人工确认：{preview}",
        location=location,
    )


def _warn_duplicate_names(
    channels: list[AnalogChannel],
    diagnostics: DiagnosticCollector,
    location: str,
) -> None:
    """检测重名通道 —— 名称不能作为唯一键。"""
    seen: dict[str, int] = {}
    for ch in channels:
        seen[ch.name] = seen.get(ch.name, 0) + 1
    duplicates = {name: n for name, n in seen.items() if n > 1}
    if duplicates:
        preview = "、".join(f"{name}×{n}" for name, n in list(duplicates.items())[:5])
        diagnostics.warn(
            Code.CHN_NAME_DUPLICATE,
            f"存在 {len(duplicates)} 组重复的通道名称（{preview}），"
            "请以下标或角色而非名称作为通道唯一标识",
            location=location,
        )


__all__ = [
    "ChannelIdentification",
    "identify",
    "identify_all",
    "normalize_name",
]
