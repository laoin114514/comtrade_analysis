"""解析选项：把"现场文件的坑"变成可配置项，而不是硬编码的猜测。

对应需求：NF-28（规则/参数可配置）、NF-32（通道数据兼容）
设计意图：解析层对可疑数据一律"给出结论 + 记录诊断 + 允许覆盖"，
不静默猜测，也不因为个别文件异常就写死特殊分支。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class AsciiScaling(str, enum.Enum):
    """ASCII 模拟量的换算策略。

    背景：标准与全部主流实现（python-comtrade / comtrade-rs / GSF / pycomtrade）
    都对 ASCII 的模拟量施加 ``a * raw + b`` 换算。现场确实存在直接写工程量的
    ASCII 文件，但也有大量"min/max 声明满量程、实际原始计数很小"的正常文件，
    两者无法可靠区分。

    早期版本默认用"数据跨度 vs 声明量程跨度"自动判定并跳过换算，
    在 11 组真实公开样例（186 个通道）上实测误判 10 个通道、全部错向"跳过"，
    因此**默认改为按标准换算**，跨度判定降级为纯提示（见 ``prescaled_span_ratio``）。
    """

    ALWAYS = "always"
    """按标准施加 a/b 换算（默认）。"""

    NEVER = "never"
    """不施加 a/b 换算。仅在确认客户文件写的是工程量时使用。"""


@dataclass(slots=True)
class ParseOptions:
    """解析选项。默认值适用于绝大多数标准合规文件。"""

    # ------------------------------------------------------------ 数值与内存
    value_dtype: str = "float64"
    """换算后工程量数组的数据类型。``float32`` 可显著降低大文件内存占用。"""

    keep_raw_values: bool = False
    """是否在结果中保留原始采样值。

    关闭（默认）可省一半内存；打开便于追溯换算过程、复核 a/b 是否正确。
    无论开关，a/b 都会保留在通道对象里，必要时可重新换算。
    """

    # -------------------------------------------------------------- 换算策略
    ascii_scaling: AsciiScaling = AsciiScaling.ALWAYS
    """ASCII 数据的 a/b 换算策略，见 :class:`AsciiScaling`。"""

    prescaled_span_ratio: float = 0.05
    """跨度提示的阈值（**纯提示，不改变解析行为**）。

    ASCII 通道的数据跨度小于 min/max 声明跨度该比例时，记录一条 DAT-009 提示：
    "这个通道看起来可能已经是工程量了，若确认如此请把 ascii_scaling 设为 never"。

    注意：真实文件中"min/max 声明满量程、实际原始计数很小"是**正常现象**
    （SEL 等装置的 ASCII 样例即如此），因此该判定只作为提示，
    不再像早期版本那样据此跳过换算。
    """

    apply_ratio_conversion: bool = True
    """是否按 PS 字段与一次/二次变比，把数值统一换算成一次值。"""

    # ---------------------------------------------------------------- 缺失值
    mask_missing_values: bool = True
    """是否把缺失值哨兵对应的采样点标记为无效（NaN）。"""

    mask_out_of_range: bool = False
    """是否把超出 min/max 范围的采样点也标记为无效。

    默认关闭：min/max 是文件里的元数据，现场常常写错，
    按它删除数据是破坏性的。开启后仅在确认文件元数据可靠时使用。
    """

    # ---------------------------------------------------------------- 时间轴
    time_base_seconds: float | None = None
    """采样时标的基准单位（秒）。

    ``None`` 表示自动判定：起始时刻小数位为 9 位（2013 版纳秒精度）时取 1e-9，
    否则取 1e-6（微秒，1991/1999 版）。
    """

    rate_tolerance: float = 0.05
    """采样率推算时间与采样时标推算时间的相对容差，超出则记录 TIM-002。"""

    time_axis_source: str = "rates"
    """时间轴的优先依据。

    ``"rates"``（默认）
        采样率分段优先，采样时标兜底。标准定义的方式，
        也是 GSF / comtrade-rs / python-comtrade 的一致做法。

    ``"timestamps"``
        采样时标优先，采样率分段兜底。反映录波器**实际**的采样时刻，
        单频文件里与本机标称采样率会有微小漂移（实测样例差 0.04%~0.1%）。

    两者都会做交叉校验与单调性检查。同一份文件两种取值可能给出
    略有差异的时间轴，**用客户真实文件确定后应固定下来**，
    否则同一份录波在两处（如本模块与已有解析服务）会算出不同的时长。
    """

    # ---------------------------------------------------------------- 严格度
    strict_channel_count: bool = False
    """通道总数与模拟+开关量之和不一致时，是否直接判定为致命错误。

    默认关闭 —— 现场文件偶有声明错误，按实际通道行解析更稳妥。
    """

    def resolved_value_dtype(self):
        import numpy as np

        return np.dtype(self.value_dtype)


DEFAULT_OPTIONS = ParseOptions()


__all__ = ["AsciiScaling", "ParseOptions", "DEFAULT_OPTIONS"]
