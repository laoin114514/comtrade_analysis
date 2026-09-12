"""统一内部数据模型（解析层的对外契约）。

合同依据
    第一条 2.(1)① "……建立统一内部数据模型"
    第一条 2.(1)② "录波数据由原始文件到系统内部分析数据的自动转换"
对应需求
    P-06（建立统一内部数据格式，供波形显示与计算模块调用）
对应分工方案
    "统一内部数据格式" 模块：录波对象、通道对象、时间序列、元数据、单位、错误码、模块调用接口

设计约定（下游模块必须知道的四条）
    1. **只读**。解析结果一经产出不再修改。下游做滤波、重采样、抽稀等任何变换，
       都必须生成新对象。这是 K-01（自动保存）与 NF-14（重复分析结果稳定）的前提。
    2. **单位已归一化**。电压一律为 V，电流一律为 A（见 units 模块）。
       原始单位字符串保留在 ``unit_raw`` 里供报告展示。
    3. **数值默认是一次值**。当 cfg 提供 PS 与变比时，``values`` 已按 PS 换算到一次侧；
       未提供（1991 版）时保持 a/b 换算结果不变，并记录 CHN-004。
    4. **不承担业务**。这里没有有效值、峰值、序分量、故障类型 —— 那些属于算法模块（E）。
       解析层只回答"录波里有什么数据"，不回答"这意味着什么"。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .diagnostics import Diagnostic, Severity


class ComtradeVersion(enum.IntEnum):
    """COMTRADE 版本（对应 IEEE C37.111 各修订年份）。"""

    UNKNOWN = 0
    V1991 = 1991
    V1999 = 1999
    V2013 = 2013

    @property
    def label(self) -> str:
        return {0: "未知", 1991: "1991", 1999: "1999", 2013: "2013"}[int(self)]


class DataFileType(enum.Enum):
    """数据文件（.dat）编码类型。"""

    ASCII = "ASCII"
    BINARY = "BINARY"
    BINARY32 = "BINARY32"
    FLOAT32 = "FLOAT32"

    @property
    def is_binary(self) -> bool:
        return self is not DataFileType.ASCII

    @property
    def value_size(self) -> int:
        """单个模拟量采样值占用的字节数。"""
        return {"ASCII": 0, "BINARY": 2, "BINARY32": 4, "FLOAT32": 4}[self.value]

    @property
    def dtype_str(self) -> str:
        """二进制模拟量元素类型（小端由解析器统一处理）。"""
        return {"ASCII": "f8", "BINARY": "i2", "BINARY32": "i4", "FLOAT32": "f4"}[self.value]


class ChannelRole(enum.Enum):
    """归一化通道角色。

    下游模块（波形、算法、报告）应当通过角色取通道，而不是匹配通道名称字符串 ——
    同一个 A 相电流在不同录波器里可能叫 ``Ia`` / ``IA`` / ``I a`` / ``A相电流`` / ``IL1``。

    关于零序量（**实现前必须与电力专家确认的一点**）
        现场对"零序电流"有两种口径：严格定义的 ``I0 = (Ia+Ib+Ic)/3``，
        以及工程上常用的 ``3I0 = Ia+Ib+Ic``（即中性线/接地线上实际流过的电流），
        两者相差 3 倍。本模块只负责把通道标成 ``I0`` 角色，
        **不对数值做任何 /3 或 ×3 处理**；规则阈值按哪种口径整定，
        由算法模块（E）与电力专家确认后在规则配置里明确。
    """

    UA = "UA"
    UB = "UB"
    UC = "UC"
    UAB = "UAB"
    UBC = "UBC"
    UCA = "UCA"
    U0 = "U0"
    IA = "IA"
    IB = "IB"
    IC = "IC"
    I0 = "I0"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"

    @property
    def is_voltage(self) -> bool:
        return self.value.startswith("U")

    @property
    def is_current(self) -> bool:
        return self.value.startswith("I")

    @property
    def is_zero_sequence(self) -> bool:
        return self in (ChannelRole.U0, ChannelRole.I0)


class Quantity(enum.Enum):
    """物理量类别。"""

    VOLTAGE = "voltage"
    CURRENT = "current"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class SampleRateSegment:
    """一段采样率。

    Attributes
        rate_hz: 采样率（Hz）。0 表示该段未声明采样率。
        end_sample: 该采样率覆盖到的**最后一个采样点的累计编号**（含）。

    注意
        ``end_sample`` 是累计编号而非段内点数。
        例：``[(1000, 2000), (4000, 10000)]`` 表示第 1~2000 点按 1000Hz，
        第 2001~10000 点按 4000Hz。这是最容易算错的地方。
    """

    rate_hz: float
    end_sample: int


@dataclass(slots=True)
class AnalogChannel:
    """模拟量通道（电压/电流等连续量）。

    字段分三类：文件元数据（原样保留，供追溯与报告）、
    归一化信息（供下游使用）、解析结果（采样数据）。
    """

    # ------------------------------------------------------------ 文件元数据
    index: int
    """通道在文件中的出现顺序（0 起）。这是唯一可信的索引。"""

    declared_no: int | None
    """文件里声明的通道序号。**不可信** —— 现场存在从 0 起、跳号、重复的情况，
    仅原样保留供排查，下游一律用 ``index``。"""

    name: str
    """通道名称，原样保留（可能含空格、可能重复）。"""

    phase_raw: str
    """cfg 里的 ``ph`` 字段原文（A/B/C/N/空）。"""

    circuit: str
    """cfg 里的 ``ccbm`` 字段（所属线路/间隔）。"""

    unit_raw: str
    """cfg 里的 ``uu`` 字段原文（kV/A/空）。"""

    a: float
    """换算系数：``工程值 = a * 原始值 + b``。"""

    b: float
    """换算偏移。"""

    skew_us: float
    """通道时偏（微秒），一般为 0。"""

    raw_min: float
    """原始值下限（未换算）。"""

    raw_max: float
    """原始值上限（未换算）。"""

    primary: float | None
    """一次侧额定值（CT/PT 变比分子）。1991 版无此字段。"""

    secondary: float | None
    """二次侧额定值（CT/PT 变比分母）。1991 版无此字段。"""

    ps: str | None
    """``P``（文件中的值已是一次值）或 ``S``（已换算为二次值）。1991 版为 None。"""

    # -------------------------------------------------------------- 归一化信息
    unit: str = ""
    """归一化后的单位符号（``V`` / ``A``）。"""

    quantity: Quantity = Quantity.UNKNOWN
    """物理量类别。"""

    role: ChannelRole = ChannelRole.UNKNOWN
    """归一化通道角色。"""

    role_confidence: float = 0.0
    """角色识别置信度（0~1）。偏低时界面应提示人工确认映射。"""

    role_source: str = "unresolved"
    """角色判定依据：``unit+phase`` / ``name`` / ``unresolved``。"""

    # -------------------------------------------------------------- 解析结果
    values: np.ndarray | None = None
    """工程量数组，形状 ``(采样点数,)``。

    已按 :mod:`comtrade.units` 归一化到基准单位，并按 PS 换算到一次值
    （受 :class:`~comtrade.options.ParseOptions` 控制）。
    被判定为无效的采样点为 ``NaN``。
    """

    raw_values: np.ndarray | None = None
    """原始采样值（仅在 ``ParseOptions.keep_raw_values`` 打开时保留）。"""

    invalid_mask: np.ndarray | None = None
    """无效采样点掩码（缺失值哨兵等），True 表示该点无效。"""

    scaling_applied: bool = True
    """是否应用了 a/b 换算。False 表示判定为"已是工程量"。"""

    ratio_applied: bool = False
    """是否应用了一次/二次变比换算。"""

    # ---------------------------------------------------------------- 便捷属性
    @property
    def is_usable(self) -> bool:
        """该通道是否可以参与后续计算。"""
        return self.values is not None and self.values.size > 0

    @property
    def has_invalid(self) -> bool:
        return self.invalid_mask is not None and bool(self.invalid_mask.any())

    @property
    def invalid_count(self) -> int:
        return int(self.invalid_mask.sum()) if self.invalid_mask is not None else 0

    @property
    def unit_source(self) -> str:
        """报告用：展示"原始单位 → 归一化单位"的换算关系。"""
        if self.unit_raw and self.unit and self.unit != self.unit_raw:
            return f"{self.unit_raw} → {self.unit}"
        return self.unit or self.unit_raw or "-"

    def __repr__(self) -> str:  # pragma: no cover
        n = self.values.size if self.values is not None else 0
        return (
            f"AnalogChannel(index={self.index}, name={self.name!r}, "
            f"role={self.role.value}, unit={self.unit or '-'}, samples={n})"
        )


@dataclass(slots=True)
class DigitalChannel:
    """开关量通道（保护动作、断路器位置等状态量）。"""

    index: int
    """在开关量列表中的出现顺序（0 起）。"""

    declared_no: int | None
    """文件里声明的序号，不可信，仅保留原文。"""

    name: str
    """通道名称。"""

    phase_raw: str
    """cfg 里的 ``ph`` 字段原文。1991 版无此字段，为空串。"""

    circuit: str
    """cfg 里的 ``ccbm`` 字段。1991 版无此字段，为空串。"""

    normal_state: int
    """常态值（0/1），即正常运行时该开关量的状态，用于判断变位方向。"""

    values: np.ndarray | None = None
    """状态数组，``bool`` 类型，形状 ``(采样点数,)``，True 表示"动作/闭合"。"""

    @property
    def is_usable(self) -> bool:
        return self.values is not None and self.values.size > 0

    def transitions(self) -> np.ndarray:
        """返回状态发生变化的采样点索引。

        归解析层提供，是因为它纯粹是数据的直接派生，不含业务判断
        （"哪次变位意味着跳闸"属于算法模块 E）。
        """
        if self.values is None or self.values.size < 2:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.values[1:] != self.values[:-1]) + 1

    def __repr__(self) -> str:  # pragma: no cover
        n = self.values.size if self.values is not None else 0
        return f"DigitalChannel(index={self.index}, name={self.name!r}, samples={n})"


@dataclass(slots=True)
class Metadata:
    """录波元数据。对应需求 K-02"保存文件名称、时间等信息"。"""

    # -------------------------------------------------------------- 来源与版本
    station_name: str
    """厂站名称（可能为中文，编码见解析诊断）。"""

    device_id: str
    """录波装置标识。"""

    revision_year: int | None
    """cfg 首行声明的版本年份。1991 版无此字段，为 None。"""

    version: ComtradeVersion
    """判定出的版本。"""

    source_cfg: Path | None
    """来源 .cfg 文件路径。"""

    source_dat: Path | None
    """来源 .dat 文件路径。"""

    cfg_sha256: str = ""
    """cfg 内容摘要，用于案例去重与"重复分析结果稳定"校验（NF-14）。"""

    dat_sha256: str = ""
    """dat 内容摘要。"""

    # -------------------------------------------------------------- 采样参数
    line_frequency: float = 0.0
    """系统频率（Hz），50 或 60；0 表示文件未声明。"""

    sample_rate_segments: list[SampleRateSegment] = field(default_factory=list)
    """采样率分段。为空表示文件未声明采样率。"""

    sample_count: int = 0
    """实际解析出的采样点数。"""

    start_time: datetime | None = None
    """第一个采样点对应的绝对时刻。"""

    trigger_time: datetime | None = None
    """触发（故障）时刻。"""

    data_type: DataFileType | None = None
    """数据文件编码类型。"""

    time_mult: float = 1.0
    """时间倍数（微秒基准）。1991 版无此字段，按 1 处理。"""

    time_base_seconds: float = 1e-6
    """采样时标的基准单位（秒）。

    1e-6 = 微秒（1991/1999 版）；1e-9 = 纳秒（2013 版声明 9 位小数时）。
    时间轴换算：``t = 采样时标 * time_mult * time_base_seconds``。
    """

    # ------------------------------------------------------ 2013 版附加字段
    timezone_code: str | None = None
    """2013 版 time_code（如 ``-5h30``，``x`` 表示不适用）。"""

    local_code: str | None = None
    """2013 版 local_code。"""

    time_quality_code: str | None = None
    """2013 版 tmq_code。"""

    leap_second: str | None = None
    """2013 版 leapsec。"""

    # ---------------------------------------------------------------- 通道统计
    total_channels: int = 0
    analog_count: int = 0
    digital_count: int = 0

    # ------------------------------------------------------------ 可选伴随文件
    header_text: str | None = None
    """``.hdr`` 文件内容（人工填写的故障简报，可能含关键信息）。"""

    info_text: str | None = None
    """``.inf`` 文件内容（1999 版起）。"""

    # ---------------------------------------------------------------- 便捷属性
    @property
    def primary_sample_rate(self) -> float:
        """主采样率（取最高的一段），用于展示与抽稀参考。"""
        if not self.sample_rate_segments:
            return 0.0
        return max(seg.rate_hz for seg in self.sample_rate_segments)

    @property
    def has_uniform_rate(self) -> bool:
        return len(self.sample_rate_segments) <= 1

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Metadata(station={self.station_name!r}, version={self.version.label}, "
            f"channels={self.analog_count}A/{self.digital_count}D, "
            f"samples={self.sample_count})"
        )


@dataclass(slots=True)
class Recording:
    """一次录波的统一内部数据对象 —— 解析模块的最终产物。

    这是解析层与下游（波形 W、算法 E、报告 R、案例 K）之间的唯一契约。
    所有下游模块只依赖这个对象，不感知 COMTRADE 的版本差异与文件格式差异。
    """

    meta: Metadata
    analog_channels: list[AnalogChannel] = field(default_factory=list)
    digital_channels: list[DigitalChannel] = field(default_factory=list)

    time_axis: np.ndarray | None = None
    """时间轴（秒），相对第一个采样点，形状 ``(采样点数,)``，单调递增。"""

    sample_numbers: np.ndarray | None = None
    """采样点编号，形状 ``(采样点数,)``。"""

    diagnostics: list[Diagnostic] = field(default_factory=list)
    """解析过程中的全部诊断信息（含提示、警告、致命）。"""

    # ---------------------------------------------------------------- 便捷属性
    @property
    def sample_count(self) -> int:
        return self.meta.sample_count

    @property
    def duration(self) -> float:
        """录波时长（秒）。"""
        if self.time_axis is None or self.time_axis.size == 0:
            return 0.0
        return float(self.time_axis[-1] - self.time_axis[0])

    @property
    def all_channels(self) -> list[AnalogChannel | DigitalChannel]:
        return [*self.analog_channels, *self.digital_channels]

    # ---------------------------------------------------------------- 查询接口
    def by_role(self, *roles: ChannelRole) -> list[AnalogChannel]:
        """按角色取通道，例如 ``rec.by_role(ChannelRole.IA, ChannelRole.IB, ChannelRole.IC)``。"""
        wanted = set(roles)
        return [ch for ch in self.analog_channels if ch.role in wanted]

    def first_by_role(self, role: ChannelRole) -> AnalogChannel | None:
        """取第一个匹配角色的通道，找不到返回 None。"""
        for ch in self.analog_channels:
            if ch.role is role:
                return ch
        return None

    def require_role(self, role: ChannelRole) -> AnalogChannel:
        """取通道，找不到时抛 KeyError 并给出可读提示。"""
        hit = self.first_by_role(role)
        if hit is None:
            raise KeyError(
                f"未找到角色为 {role.value} 的通道；"
                f"可用通道：{[c.name for c in self.analog_channels]}"
            )
        return hit

    def all_by_role(self, role: ChannelRole) -> list[AnalogChannel]:
        """取占用该角色的**全部**通道。

        一份录波里可能有多个通道同角色（例如同时存在有效值电流 IARMS 与
        瞬时电流 IA、或多组绕组的三相电流）。解析时会记录 CHN-005 提示歧义，
        算法模块若发现该提示，应当用本方法列出候选再人工/规则确认。
        """
        return [ch for ch in self.analog_channels if ch.role is role]

    def by_role_map(self) -> dict[ChannelRole, AnalogChannel]:
        """角色 → 通道的映射，只包含识别成功的通道。

        算法模块（E）应当基于这个映射判断"三相电流是否齐全"，
        而不是挨个 try/except。

        取值规则
            同一角色有多个通道时，取**文件中出现顺序最靠前**的那个
            （即索引最小者）。这是确定性的，但**不保证是期望的那一组** ——
            当诊断中出现 CHN-005 时，说明存在歧义，应当改用
            :meth:`all_by_role` 列出候选后确认。
        """
        out: dict[ChannelRole, AnalogChannel] = {}
        for ch in self.analog_channels:
            if ch.role is not ChannelRole.UNKNOWN:
                out.setdefault(ch.role, ch)
        return out

    def by_name(self, name: str) -> AnalogChannel | DigitalChannel | None:
        """按名称精确查找（名称可能重复，返回第一个）。"""
        for ch in self.all_channels:
            if ch.name == name:
                return ch
        return None

    def unresolved_channels(self) -> list[AnalogChannel]:
        """返回角色未能识别、需要人工映射的通道（对应 CHN-001）。"""
        return [ch for ch in self.analog_channels if ch.role in (ChannelRole.UNKNOWN, ChannelRole.OTHER)]

    # ---------------------------------------------------------------- 数据切片
    def index_range(self, time_from: float, time_to: float) -> tuple[int, int]:
        """按时间区间求采样点索引区间 ``[start, end)``。

        仅做索引计算，不复制数据 —— 抽稀/降采样属于显示模块（W）的职责，
        解析层不为了让界面画得动而丢弃数据精度。
        """
        if self.time_axis is None or self.time_axis.size == 0:
            return (0, 0)
        start = int(np.searchsorted(self.time_axis, time_from, side="left"))
        end = int(np.searchsorted(self.time_axis, time_to, side="right"))
        return (start, end)

    def slice_time(self, time_from: float, time_to: float) -> np.ndarray:
        """时间切片对应的采样点索引数组（可直接用于数组花式索引）。"""
        start, end = self.index_range(time_from, time_to)
        return np.arange(start, end, dtype=np.int64)

    # ---------------------------------------------------------------- 诊断查询
    def diagnostics_by_severity(self, severity: Severity) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is severity]

    @property
    def is_reliable(self) -> bool:
        """结果是否可信。

        界面可以据此决定：正常出结果 / 出结果但标黄提示 /
        不允许生成报告并要求用户检查文件。
        """
        return not any(
            d.severity in (Severity.WARNING, Severity.FATAL)
            and d.code.startswith(("DAT-002", "DAT-004", "TIM-001"))
            for d in self.diagnostics
        )

    def summary(self) -> str:
        """生成一段可读摘要，便于日志输出与命令行排查。"""
        m = self.meta
        lines = [
            f"厂站        : {m.station_name or '-'}   装置: {m.device_id or '-'}",
            f"COMTRADE 版本: {m.version.label}"
            + (f"（声明年份 {m.revision_year}）" if m.revision_year else ""),
            f"数据格式    : {m.data_type.value if m.data_type else '-'}"
            f"   时间倍数: {m.time_mult:g}",
            f"通道        : 模拟 {m.analog_count} / 开关量 {m.digital_count}",
            "采样率      : "
            + (
                ", ".join(
                    (f"{seg.rate_hz:g}Hz→{seg.end_sample}点" if seg.rate_hz > 0
                     else f"未声明(仅给出结束采样号 {seg.end_sample})")
                    for seg in m.sample_rate_segments
                )
                or "未声明（时间轴按采样时标推算）"
            ),
            f"采样点数    : {m.sample_count}   时长: {self.duration:.6f} s",
            f"起始时刻    : {m.start_time}" if m.start_time else "起始时刻    : -",
            f"触发时刻    : {m.trigger_time}" if m.trigger_time else "触发时刻    : -",
            "",
            "通道明细:",
        ]
        for ch in self.analog_channels:
            conf = f"{ch.role_confidence:.2f}"
            lines.append(
                f"  A{ch.index:<3} {ch.name:<16} 角色={ch.role.value:<6} "
                f"单位={ch.unit_source:<12} 置信度={conf} 依据={ch.role_source}"
                + (f" 无效点={ch.invalid_count}" if ch.has_invalid else "")
            )
        for ch in self.digital_channels:
            lines.append(f"  D{ch.index:<3} {ch.name:<16} 常态={ch.normal_state}")

        if self.diagnostics:
            lines.append("")
            counts = {s: len(self.diagnostics_by_severity(s)) for s in Severity}
            lines.append(
                f"诊断: 共 {len(self.diagnostics)} 条 "
                f"(致命 {counts[Severity.FATAL]} / 警告 {counts[Severity.WARNING]} / "
                f"提示 {counts[Severity.INFO]})"
            )
            for d in self.diagnostics:
                if d.severity is not Severity.INFO:
                    lines.append(f"  {d}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Recording({self.meta.station_name!r}, {self.meta.version.label}, "
            f"{self.meta.analog_count}A/{self.meta.digital_count}D, "
            f"{self.meta.sample_count} samples)"
        )


__all__ = [
    "ComtradeVersion",
    "DataFileType",
    "ChannelRole",
    "Quantity",
    "SampleRateSegment",
    "AnalogChannel",
    "DigitalChannel",
    "Metadata",
    "Recording",
]
