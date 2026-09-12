"""测试辅助：构造最小可用的 COMTRADE 文件对。

不依赖 pytest fixture，用标准库的临时目录 —— 这样测试既能被 pytest 收集，
也能用 ``tests/run_tests.py`` 直接运行（客户/同事机器上没装 pytest 时用得上）。
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AnalogDef:
    """测试用模拟通道定义。"""

    name: str = "Ia"
    phase: str = "A"
    circuit: str = "LINE1"
    unit: str = "A"
    a: float = 1.0
    b: float = 0.0
    skew: float = 0.0
    vmin: float = -32767.0
    vmax: float = 32767.0
    primary: float = 400.0
    secondary: float = 1.0
    ps: str = "S"


@dataclass
class DigitalDef:
    """测试用开关量通道定义。"""

    name: str = "TRIP"
    phase: str = "A"
    circuit: str = "LINE1"
    normal: int = 0


@dataclass
class CfgSpec:
    """测试用 cfg 描述。"""

    version: int | None = 1999
    station: str = "TEST-STATION"
    device: str = "DFR-001"
    analog: list[AnalogDef] = field(default_factory=lambda: [AnalogDef()])
    digital: list[DigitalDef] = field(default_factory=lambda: [DigitalDef()])
    line_freq: float = 50.0
    segments: list[tuple[float, int]] = field(default_factory=lambda: [(4000.0, 100)])
    start: str = "01/07/2026,09:15:00.000000"
    trigger: str = "01/07/2026,09:15:00.010000"
    data_type: str = "BINARY"
    time_mult: float | None = 1.0
    tail_lines: list[str] = field(default_factory=list)
    encoding: str = "utf-8"
    extra_before_type: list[str] = field(default_factory=list)


def render_cfg(spec: CfgSpec) -> str:
    """把 :class:`CfgSpec` 渲染成 cfg 文本。"""
    lines: list[str] = []

    # 行 1
    if spec.version is None:
        lines.append(f"{spec.station},{spec.device}")
    else:
        lines.append(f"{spec.station},{spec.device},{spec.version}")

    # 行 2
    lines.append(f"{len(spec.analog) + len(spec.digital)},{len(spec.analog)}A,{len(spec.digital)}D")

    # 模拟通道
    for i, ch in enumerate(spec.analog, start=1):
        row = (
            f"{i},{ch.name},{ch.phase},{ch.circuit},{ch.unit},"
            f"{ch.a:.10g},{ch.b:.10g},{ch.skew:g},{ch.vmin:g},{ch.vmax:g}"
        )
        if spec.version is not None and spec.version >= 1999:
            row += f",{ch.primary:.10g},{ch.secondary:.10g},{ch.ps}"
        lines.append(row)

    # 开关量通道
    for i, d in enumerate(spec.digital, start=1):
        if spec.version is not None and spec.version >= 1999:
            lines.append(f"{i},{d.name},{d.phase},{d.circuit},{d.normal}")
        else:
            lines.append(f"{i},{d.name},{d.normal}")

    lines.append(f"{spec.line_freq:g}")
    lines.append(f"{len(spec.segments)}")
    for rate, end in spec.segments:
        lines.append(f"{rate:g},{end}")
    lines.append(spec.start)
    lines.append(spec.trigger)
    lines.extend(spec.extra_before_type)
    lines.append(spec.data_type)
    if spec.time_mult is not None:
        lines.append(f"{spec.time_mult:g}")
    lines.extend(spec.tail_lines)
    return "\n".join(lines) + "\n"


def write_binary_dat(
    path: Path,
    analog: np.ndarray | None,
    digital: np.ndarray | None,
    *,
    data_type: str = "BINARY",
    sample_numbers: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> None:
    """写出二进制 dat。

    Args:
        analog: ``(A, N)`` 原始值矩阵。
        digital: ``(D, N)`` 布尔矩阵。
    """
    n = (analog.shape[1] if analog is not None else (digital.shape[1] if digital is not None else 0))
    a_count = analog.shape[0] if analog is not None else 0
    d_count = digital.shape[0] if digital is not None else 0
    words = -(-d_count // 16)

    if sample_numbers is None:
        sample_numbers = np.arange(1, n + 1)
    if timestamps is None:
        timestamps = np.arange(n) * 250  # 4000Hz → 250us

    value_dtype = {"BINARY": "<i2", "BINARY32": "<i4", "FLOAT32": "<f4"}[data_type]
    fields: list[tuple] = [("samp", "<u4"), ("ts", "<u4")]
    if a_count:
        fields.append(("analog", value_dtype, (a_count,)))
    if words:
        fields.append(("digital", "<u2", (words,)))

    arr = np.zeros(n, dtype=np.dtype(fields))
    arr["samp"] = sample_numbers
    arr["ts"] = timestamps
    if a_count:
        arr["analog"] = analog.T.astype(arr["analog"].dtype)
    if words:
        packed = np.zeros((n, words), dtype=np.uint16)
        for d in range(d_count):
            packed[:, d // 16] |= (digital[d].astype(np.uint16) << (d % 16))
        arr["digital"] = packed
    arr.tofile(path)


def write_ascii_dat(
    path: Path,
    analog: np.ndarray | None,
    digital: np.ndarray | None,
    *,
    timestamps: np.ndarray | None = None,
) -> None:
    """写出 ASCII dat。"""
    n = analog.shape[1] if analog is not None else digital.shape[1]
    if timestamps is None:
        timestamps = np.arange(n) * 250
    rows = []
    for s in range(n):
        cols = [str(s + 1), str(int(timestamps[s]))]
        if analog is not None:
            cols.extend(f"{v:.10g}" for v in analog[:, s])
        if digital is not None:
            cols.extend("1" if digital[d, s] else "0" for d in range(digital.shape[0]))
        rows.append(",".join(cols))
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


class TempRecording:
    """在临时目录里构造一组 cfg/dat，供测试直接解析。"""

    def __init__(self, tmpdir: Path, name: str = "case") -> None:
        self.dir = tmpdir
        self.name = name
        self.cfg = tmpdir / f"{name}.cfg"
        self.dat = tmpdir / f"{name}.dat"

    def write_cfg(self, spec: CfgSpec) -> Path:
        self.cfg.write_text(render_cfg(spec), encoding=spec.encoding)
        return self.cfg

    def write_binary(self, analog, digital, **kw) -> Path:
        write_binary_dat(self.dat, analog, digital, **kw)
        return self.dat

    def write_ascii(self, analog, digital, **kw) -> Path:
        write_ascii_dat(self.dat, analog, digital, **kw)
        return self.dat


def simple_sine(freq: float = 50.0, fs: float = 4000.0, n: int = 400, amplitude: float = 1000.0,
                offset: float = 0.0, phase: float = 0.0) -> np.ndarray:
    """生成一段正弦原始值，用于需要真实波形的测试。"""
    t = np.arange(n) / fs
    return offset + amplitude * np.sin(2.0 * np.pi * freq * t + phase)


__all__ = [
    "AnalogDef",
    "DigitalDef",
    "CfgSpec",
    "TempRecording",
    "render_cfg",
    "write_binary_dat",
    "write_ascii_dat",
    "simple_sine",
    "struct",
]
