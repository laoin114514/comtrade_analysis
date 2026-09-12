"""COMTRADE 解析模块（故障录波智能辅助分析验证平台 · 一期）。

本模块的职责边界（严格按合同第一条 2.(1) 划分）
================================================

**做**（合同第一条 2.(1)①② 明确要求）
    * ``.cfg`` 配置文件解析（COMTRADE 1991 / 1999 / 2013 三个版本自适应）
    * ``.dat`` 数据文件读取（ASCII / BINARY / BINARY32 / FLOAT32 四种编码）
    * 自动识别录波通道信息（含通道角色归一化 —— 见 :mod:`comtrade.channels`）
    * 获取采样频率及生成时间轴（见 :mod:`comtrade.timebase`）
    * 建立统一内部数据模型（见 :mod:`comtrade.models`）
    * 异常文件检测及错误提示机制（见 :mod:`comtrade.diagnostics`）
    * 录波数据由原始文件到系统内部分析数据的自动转换（见 :mod:`comtrade.convert`）

**不做**（属于其它模块，本合同条款未授权解析层承担）
    * 电气特征计算（有效值、峰值、变化率、序分量）→ 算法模块（E）
    * 故障类型与相别判断、规则库 → 算法模块（E）
    * 波形抽稀与显示 → 客户端模块（F）。解析层**不为了让界面画得动而丢弃数据精度**
    * PDF 报告生成 → 测试交付模块（G）
    * 历史案例库读写 → 案例模块（F）
    * 数据修复（插值、补零、滤波）→ 一律不做。畸形数据如实记录并报警，
      不修改原始信息

快速开始
--------

>>> from comtrade_analysis.comtrade import load_recording, ChannelRole
>>> rec = load_recording(r"D:\\cases\\fault1.cfg")     # doctest: +SKIP
>>> rec.meta.station_name                              # doctest: +SKIP
'110kV 某某变'
>>> ia = rec.require_role(ChannelRole.IA)              # doctest: +SKIP
>>> ia.values.shape                                    # doctest: +SKIP
(8000,)
>>> rec.duration                                       # doctest: +SKIP
2.0

不抛异常的版本（批量处理、回归测试用）：

>>> rec, diags = try_load_recording(path)               # doctest: +SKIP
>>> if rec is None:
...     for d in diags:
...         print(d)
"""
from __future__ import annotations

from .channels import ChannelIdentification, identify, normalize_name
from .cfg_parser import ParsedCfg, parse_cfg
from .convert import convert_analog_channels, missing_sentinel
from .dat_parser import ParsedDat, parse_dat, record_size
from .diagnostics import (
    Code,
    ComtradeParseError,
    Diagnostic,
    DiagnosticCollector,
    Severity,
)
from .models import (
    AnalogChannel,
    ChannelRole,
    ComtradeVersion,
    DataFileType,
    DigitalChannel,
    Metadata,
    Quantity,
    Recording,
    SampleRateSegment,
)
from .options import AsciiScaling, ParseOptions
from .reader import (
    SUPPORTED_EXTENSIONS,
    load_recording,
    resolve_pair,
    try_load_recording,
)
from .timebase import build_time_axis
from .units import parse_unit

__version__ = "0.1.0"

__all__ = [
    # 主入口
    "load_recording",
    "try_load_recording",
    "resolve_pair",
    "SUPPORTED_EXTENSIONS",
    # 数据模型
    "Recording",
    "Metadata",
    "AnalogChannel",
    "DigitalChannel",
    "SampleRateSegment",
    "ComtradeVersion",
    "DataFileType",
    "ChannelRole",
    "Quantity",
    # 诊断
    "Diagnostic",
    "DiagnosticCollector",
    "ComtradeParseError",
    "Severity",
    "Code",
    # 选项
    "ParseOptions",
    "AsciiScaling",
    # 分步接口（测试与专项排查用）
    "parse_cfg",
    "ParsedCfg",
    "parse_dat",
    "ParsedDat",
    "record_size",
    "convert_analog_channels",
    "missing_sentinel",
    "build_time_axis",
    "identify",
    "ChannelIdentification",
    "normalize_name",
    "parse_unit",
    "__version__",
]
