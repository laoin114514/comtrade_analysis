"""单位归一化。

合同依据：第一条 2.(1)① "……建立统一内部数据模型"
对应需求：E-01/E-02（有效值计算精度的前提）、NF-32（正确识别主要模拟量通道）

为什么必须在解析层做
    cfg 里的 ``uu`` 字段现场可能是 V / kV / A / kA / mV / 空值 / 大小写混写。
    如果不在解析层归一化，同一个"0.4"到底是 0.4kV 还是 0.4V
    要靠每个下游模块各自猜，必然出现"报告写 0.4 而专工期待 400"的问题。
    这里统一到基准单位（V / A），原始单位字符串保留在通道对象里供报告展示。
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class UnitInfo:
    """单位解析结果。"""

    raw: str
    """文件里原始的 ``uu`` 字符串。"""

    canonical: str
    """归一化后的单位符号，如 ``V`` / ``A``；无法识别时保留去空格的原文。"""

    scale: float
    """把该单位的数值换算到基准单位所需的乘数（kV → 1000）。"""

    kind: str
    """物理量类别：``voltage`` / ``current`` / ``power`` / ``frequency`` / ``other``。"""

    recognized: bool
    """是否被识别。未识别时不缩放，只记录诊断。"""


# 基准单位：电压 → V，电流 → A
_UNIT_MAP: dict[str, tuple[str, float, str]] = {
    # --- 电压 ---
    "v": ("V", 1.0, "voltage"),
    "volt": ("V", 1.0, "voltage"),
    "volts": ("V", 1.0, "voltage"),
    "kv": ("V", 1_000.0, "voltage"),
    "kilovolt": ("V", 1_000.0, "voltage"),
    "mv": ("V", 1e-3, "voltage"),
    "uv": ("V", 1e-6, "voltage"),
    "伏": ("V", 1.0, "voltage"),
    "千伏": ("V", 1_000.0, "voltage"),
    # --- 电流 ---
    "a": ("A", 1.0, "current"),
    "amp": ("A", 1.0, "current"),
    "amps": ("A", 1.0, "current"),
    "ampere": ("A", 1.0, "current"),
    "ka": ("A", 1_000.0, "current"),
    "kiloamp": ("A", 1_000.0, "current"),
    "ma": ("A", 1e-3, "current"),
    "ua": ("A", 1e-6, "current"),
    "安": ("A", 1.0, "current"),
    "千安": ("A", 1_000.0, "current"),
    # --- 其他（不缩放，但要归类，避免被误判成电压/电流）---
    "w": ("W", 1.0, "power"),
    "kw": ("W", 1_000.0, "power"),
    "mw": ("W", 1e6, "power"),
    "var": ("var", 1.0, "power"),
    "kvar": ("var", 1_000.0, "power"),
    "va": ("VA", 1.0, "power"),
    "kva": ("VA", 1_000.0, "power"),
    "hz": ("Hz", 1.0, "frequency"),
    "deg": ("deg", 1.0, "angle"),
    "°": ("deg", 1.0, "angle"),
    "c": ("°C", 1.0, "temperature"),
    "°c": ("°C", 1.0, "temperature"),
    "%": ("%", 1.0, "ratio"),
    "pu": ("pu", 1.0, "ratio"),
}

# 归一化前先剥掉这些字符：厂家写法五花八门，"(kV)"、"kV."、"U/kV" 都出现过
_STRIP_RE = re.compile(r"[\s\(\)\[\]\.\,、]")
# 形如 "kV/1000" 或 "A(一次)" 这类带说明的写法，取斜杠或括号前的主体
_SPLIT_RE = re.compile(r"[/（(]")

_UNRECOGNIZED = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff°%]+")

#: 单位字段里常见的后缀说明（不是单位本身），匹配失败时逐个剥掉再试
_ANNOTATIONS = ("一次", "二次", "侧", "主", "备", "进线", "出线", "线路", "母线", "值")


def _lookup(cleaned: str):
    """在单位表里查一次（含复数形式）。"""
    hit = _UNIT_MAP.get(cleaned)
    if hit is None and cleaned.endswith("s"):
        hit = _UNIT_MAP.get(cleaned[:-1])
    return hit


def _strip_annotations(cleaned: str) -> str:
    """剥掉后缀说明词，例如 ``kV一次`` → ``kV``。"""
    text = cleaned
    changed = True
    while changed and text:
        changed = False
        for suffix in _ANNOTATIONS:
            if text.endswith(suffix) and len(text) > len(suffix):
                text = text[: -len(suffix)]
                changed = True
    return text


def _clean(raw: str) -> str:
    text = (raw or "").strip()
    if not text:
        return ""
    text = _STRIP_RE.sub("", text)
    # 形如 "(kV)" / "kV一次" / "kV/1000" 的写法：取第一个非空主体
    parts = [p for p in _SPLIT_RE.split(text) if p]
    text = parts[0] if parts else text
    text = _UNRECOGNIZED.sub("", text)
    return text.lower()


def parse_unit(raw: str) -> UnitInfo:
    """解析 ``uu`` 字段。

    >>> parse_unit("kV").canonical, parse_unit("kV").scale
    ('V', 1000.0)
    >>> parse_unit("").recognized
    False
    >>> parse_unit("A").kind
    'current'
    """
    cleaned = _clean(raw)
    if not cleaned:
        return UnitInfo(raw=raw or "", canonical="", scale=1.0, kind="other", recognized=False)

    hit = _lookup(cleaned)
    if hit is None:
        # 现场会写成 "kV一次"、"A侧" 这类带说明的写法，剥掉说明再试一次
        stripped = _strip_annotations(cleaned)
        if stripped != cleaned:
            hit = _lookup(stripped)
    if hit is None:
        return UnitInfo(
            raw=raw or "",
            canonical=cleaned,
            scale=1.0,
            kind="other",
            recognized=False,
        )

    canonical, scale, kind = hit
    return UnitInfo(raw=raw or "", canonical=canonical, scale=scale, kind=kind, recognized=True)


def quantity_from_unit(kind: str) -> str | None:
    """把单位类别映射到通道物理量类别（供 channels 模块使用）。"""
    if kind == "voltage":
        return "voltage"
    if kind == "current":
        return "current"
    return None


__all__ = ["UnitInfo", "parse_unit", "quantity_from_unit"]
