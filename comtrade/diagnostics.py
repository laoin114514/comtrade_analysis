"""COMTRADE 解析诊断体系：错误码、分级、诊断记录与异常类型。

合同依据
    第一条 2.(1)① "……提供异常文件检测及错误提示机制"
    第一条 2.(1)① "……建立统一内部数据模型"
对应需求编号
    P-07（异常解析处理，输出错误原因）、F-04（文件完整性检查）、
    F-05（非法文件格式检测）、NF-12（异常数据不崩溃）、NF-22（运行日志）

设计原则
    1. 解析器遇到"可恢复问题"时不抛异常，而是记录一条 Diagnostic 后继续工作；
       只有"无法产出任何有效数据"时才抛 ComtradeParseError（FATAL）。
       这样 NF-12「错误文件不崩溃」由架构保证，而不是靠每一处 try/except。
    2. 每条诊断都带稳定错误码。界面可以直接按码组织提示语，
       测试可以用码做断言，不依赖中文文案。
    3. 诊断信息随解析结果一起返回（Recording.diagnostics），
       便于 K-01 保存分析记录时一并落库、NF-22 输出运行日志。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class Severity(enum.Enum):
    """诊断等级。"""

    INFO = "info"
    """提示：不影响结果，仅记录（例如版本年份写法非标准）。"""

    WARNING = "warning"
    """可恢复：解析继续，结果可能受影响，建议人工确认（例如单位字段为空）。"""

    FATAL = "fatal"
    """不可恢复：无法产出有效数据，解析终止（例如缺少 .dat 文件）。"""


class Code:
    """错误码常量表。

    命名规则：``模块前缀-三位序号``。前缀含义：

    ======  ==========================================
    前缀    含义
    ======  ==========================================
    FIL     文件层面（存在性、配对、扩展名、可读性）
    CFG     .cfg 配置文件解析
    DAT     .dat 数据文件解析
    CHN     通道识别与归一化
    TIM     时间轴生成
    ======  ==========================================

    错误码一旦发布不应修改含义（测试用例与界面提示都依赖它）。
    """

    # ---------------------------------------------------------------- 文件层
    FILE_NOT_FOUND = "FIL-001"
    """文件不存在或不可访问。"""

    FILE_UNREADABLE = "FIL-002"
    """文件无法读取（权限、占用、IO 错误）。"""

    PAIR_MISSING = "FIL-003"
    """cfg/dat 未配对，缺少另一半文件。"""

    FILE_EMPTY = "FIL-004"
    """文件为空或过小，不足以构成有效录波。"""

    UNSUPPORTED_EXTENSION = "FIL-005"
    """扩展名不是 .cfg/.dat/.cff。"""

    CFF_NOT_SUPPORTED = "FIL-006"
    """检测到 2013 版 .cff 单文件格式，本期尚未支持。"""

    FILE_LOOKS_BINARY = "FIL-007"
    """本应是文本的文件内容疑似二进制（很可能 cfg/dat 传反了）。"""

    # --------------------------------------------------------------- cfg 层
    CFG_EMPTY = "CFG-001"
    """配置文件为空或无有效行。"""

    CFG_HEADER_UNPARSABLE = "CFG-002"
    """首行字段数异常，无法判定 COMTRADE 版本。"""

    CFG_VERSION_NONSTANDARD = "CFG-003"
    """版本年份非标准值（如 2000/2001），已按最接近的版本规则解析。"""

    CFG_CHANNEL_COUNT_LINE = "CFG-004"
    """通道数量行（第 2 行）无法解析。"""

    CFG_CHANNEL_COUNT_MISMATCH = "CFG-005"
    """声明通道总数与模拟量、开关量之和不一致。"""

    CFG_ANALOG_FIELD_COUNT = "CFG-006"
    """模拟通道行字段数异常（应为 10 或 13）。"""

    CFG_DIGITAL_FIELD_COUNT = "CFG-007"
    """开关量通道行字段数异常（应为 3 或 5）。"""

    CFG_NO_CHANNEL = "CFG-008"
    """配置文件声明了 0 个通道。"""

    CFG_NO_SAMPLE_RATE = "CFG-009"
    """采样率段数为 0，时间轴将改用采样时标推算。"""

    CFG_SAMPLE_RATE_LINE = "CFG-010"
    """采样率段行无法解析。"""

    CFG_DATETIME_UNPARSABLE = "CFG-011"
    """起始时刻或触发时刻无法解析。"""

    CFG_DATE_AMBIGUOUS = "CFG-012"
    """日期存在 日/月 与 月/日 二义性，已按标准取 日/月。"""

    CFG_TIME_PRECISION_TRUNCATED = "CFG-013"
    """时间精度超过微秒（纳秒），已截断至微秒。"""

    CFG_DATA_TYPE_UNKNOWN = "CFG-014"
    """数据文件类型无法识别。"""

    CFG_TIME_MULT_INVALID = "CFG-015"
    """时间倍数 time_mult 缺失或非法，已按 1 处理。"""

    CFG_UNEXPECTED_END = "CFG-016"
    """配置文件提前结束，行数不足以覆盖声明的内容。"""

    CFG_SCALE_A_ZERO = "CFG-017"
    """模拟通道比例系数 a 为 0，无法构成有效换算。"""

    CFG_UNIT_EMPTY = "CFG-018"
    """模拟通道单位字段为空。"""

    CFG_CHANNEL_NAME_EMPTY = "CFG-019"
    """通道名称为空。"""

    CFG_TRAILING_CONTENT = "CFG-020"
    """配置文件在预期结束位置之后仍有内容（已忽略）。"""

    CFG_NUMERIC_FIELD_INVALID = "CFG-021"
    """模拟通道的数值字段（a/b/skew/min/max）为空或无法解析，已按默认值处理。

    a/b 取默认值会让该通道的工程量按恒等映射输出（数值看似正常但不可用）；
    min/max 取默认值会让取值范围校验被静默跳过 —— 该校验是字节序判断错误、
    通道数量解析错误的主要发现手段。
    """

    # --------------------------------------------------------------- dat 层
    DAT_EMPTY = "DAT-001"
    """数据文件为空。"""

    DAT_SIZE_MISMATCH = "DAT-002"
    """数据文件大小与 cfg 声明的记录数不符。"""

    DAT_TRAILING_BYTES = "DAT-003"
    """数据文件末尾存在不足以构成一条记录的残余字节（已忽略）。"""

    DAT_RECORD_SHORT = "DAT-004"
    """实际记录条数少于声明值，数据可能被截断。"""

    DAT_MISSING_VALUE = "DAT-005"
    """检测到缺失值哨兵，对应采样点已标记为无效。"""

    DAT_OUT_OF_RANGE = "DAT-006"
    """采样原始值超出 cfg 声明的 min/max 范围。"""

    DAT_COLUMN_MISMATCH = "DAT-007"
    """ASCII 行字段数与通道声明不符。"""

    DAT_VALUE_UNPARSABLE = "DAT-008"
    """ASCII 采样值无法转换为数值（该行已跳过）。"""

    DAT_ASCII_PRESCALED = "DAT-009"
    """ASCII 数据疑似已经是工程量（数据跨度远小于声明量程）。

    **仅提示，不改变换算行为**：仍按标准施加 a/b 换算。确认某文件确实直接存
    工程量时，把 ``ascii_scaling`` 设为 ``never`` 显式跳过。
    早期版本曾据此自动跳过换算，在真实样例上误判率 100%，已废弃。
    """

    DAT_SCALE_SKIPPED = "DAT-010"
    """比例系数 a 为 0，已跳过 a/b 换算。"""

    DAT_FLOAT32_SCALED = "DAT-011"
    """FLOAT32 数据带有非单位比例系数，已按 a/b 换算（通常不应发生）。"""

    DAT_SCALING_DISABLED = "DAT-012"
    """按 ascii_scaling=never 显式跳过了 a/b 换算，数值为文件中的原始值。"""

    # --------------------------------------------------------------- 通道层
    CHN_ROLE_UNRESOLVED = "CHN-001"
    """通道角色无法自动识别，需人工映射后算法模块才能使用。"""

    CHN_NAME_DUPLICATE = "CHN-002"
    """存在重复的通道名称，不能以名称作为唯一键。"""

    CHN_UNIT_UNRECOGNIZED = "CHN-003"
    """单位无法识别，未做单位归一化。"""

    CHN_RATIO_MISSING = "CHN-004"
    """缺少一次/二次变比信息（1991 版无此字段），未做一次值换算。"""

    CHN_ROLE_DUPLICATE = "CHN-005"
    """同一个角色被多个通道占用（如多组三相电流），按角色取通道时有歧义。"""

    # --------------------------------------------------------------- 时间轴
    TIM_NOT_MONOTONIC = "TIM-001"
    """时间轴非单调递增，时间信息不可信。"""

    TIM_RATE_TS_INCONSISTENT = "TIM-002"
    """采样率推算的时间与采样时标推算的时间不一致。"""

    TIM_SAMPLE_NO_ANOMALY = "TIM-003"
    """采样号不连续或不从 1 开始，采样率分段按行序应用。"""


@dataclass(slots=True)
class Diagnostic:
    """一条诊断记录。

    Attributes
        code: 错误码，见 :class:`Code`。
        severity: 等级，见 :class:`Severity`。
        message: 面向人的中文说明，界面可直接展示。
        location: 定位信息，形如 ``fault1.cfg:12``，便于排查。
        detail: 补充细节（原始行内容、实际数值等），可为空。
    """

    code: str
    severity: Severity
    message: str
    location: str | None = None
    detail: str | None = None

    def __str__(self) -> str:  # pragma: no cover - 仅用于日志与调试输出
        parts = [f"[{self.severity.value.upper()}] {self.code} {self.message}"]
        if self.location:
            parts.append(f" @ {self.location}")
        if self.detail:
            parts.append(f" ({self.detail})")
        return "".join(parts)


class DiagnosticCollector:
    """诊断收集器。解析过程中所有问题都汇总到这里。"""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: list[Diagnostic] = []

    # ------------------------------------------------------------------ 记录
    def info(self, code: str, message: str, **kw) -> None:
        self._items.append(Diagnostic(code, Severity.INFO, message, **kw))

    def warn(self, code: str, message: str, **kw) -> None:
        self._items.append(Diagnostic(code, Severity.WARNING, message, **kw))

    def error(self, code: str, message: str, **kw) -> None:
        """记录可恢复的严重问题：解析继续，但结果不应被信任。"""
        self._items.append(Diagnostic(code, Severity.WARNING, message, **kw))

    def fatal(self, code: str, message: str, **kw) -> None:
        self._items.append(Diagnostic(code, Severity.FATAL, message, **kw))

    # ------------------------------------------------------------------ 读取
    @property
    def items(self) -> list[Diagnostic]:
        return list(self._items)

    def has_fatal(self) -> bool:
        return any(d.severity is Severity.FATAL for d in self._items)

    def count(self, severity: Severity | None = None) -> int:
        if severity is None:
            return len(self._items)
        return sum(1 for d in self._items if d.severity is severity)

    def codes(self) -> list[str]:
        """返回出现过的错误码（去重、保持顺序），便于测试断言。"""
        seen: list[str] = []
        for d in self._items:
            if d.code not in seen:
                seen.append(d.code)
        return seen

    def extend(self, other: "DiagnosticCollector") -> None:
        self._items.extend(other._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)


class ComtradeParseError(Exception):
    """致命解析错误。

    只有"无法产出任何有效数据"时才抛这个异常（对应 Severity.FATAL）。
    调用方（UI 层）应当捕获它并把 ``diagnostics`` 里的信息展示给用户，
    而不是让程序崩溃 —— 对应 NF-12。
    """

    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None):
        super().__init__(message)
        self.diagnostics: list[Diagnostic] = list(diagnostics or [])

    def __str__(self) -> str:  # pragma: no cover
        if not self.diagnostics:
            return super().__str__()
        lines = [super().__str__()]
        lines.extend(f"  - {d}" for d in self.diagnostics)
        return "\n".join(lines)


class ParseAbort(Exception):
    """内部信号：用于在深层解析函数中提前终止，由 reader 统一转换为 ComtradeParseError。

    不对外暴露，业务代码不应捕获。
    """


__all__ = [
    "Severity",
    "Code",
    "Diagnostic",
    "DiagnosticCollector",
    "ComtradeParseError",
    "ParseAbort",
]
