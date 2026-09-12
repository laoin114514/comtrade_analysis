"""COMTRADE 样例文件生成器。

用途
    1. 为解析模块提供可重复生成的测试数据（项目目前没有客户真实样例）；
    2. 作为 D 岗位交付物「兼容性测试样例库」的雏形 ——
       覆盖 1991/1999/2013 三个版本 × ASCII/BINARY/BINARY32/FLOAT32 四种编码；
    3. 客户样例到位后，可用同样的结构做"解析结果一致性比对"。

生成的数据是**合成的 A 相单相接地故障录波**：故障前正常负荷，
故障后 A 相电流突增 5 倍、A 相电压骤降至 30%，非故障相电压略升，
并出现零序电流。波形参数按 110kV 线路、CT 400/1、PT 110kV/100V 设置，
量纲与真实录波一致，便于算法模块（E）后续直接用来验证判据。

用法::

    python tools/make_samples.py --out tests/fixtures/generated
    python tools/make_samples.py --out ./samples --only v1999_binary
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 电气参数（110kV 线路典型值）
# ---------------------------------------------------------------------------

F_LINE = 50.0  # 系统频率 Hz
I_RATED = 400.0  # CT 一次额定电流 A
I_PRIMARY = 400.0  # CT 一次额定
I_SECONDARY = 1.0  # CT 二次额定
I_RATIO = I_PRIMARY / I_SECONDARY  # CT 变比 400/1
U_RATED = 110_000.0  # PT 一次额定电压 V（线电压）
U_PRIMARY = 110_000.0  # PT 一次额定
U_SECONDARY = 100.0  # PT 二次额定
U_RATIO = U_PRIMARY / U_SECONDARY  # PT 变比 1100

I_LOAD_PU = 0.4  # 故障前负荷电流（标幺）
I_FAULT_PU = 2.0  # 故障时 A 相电流（5 倍负荷）
U_FAULT_PU = 0.30  # 故障时 A 相电压
U_HEALTHY_PU = 1.15  # 故障时非故障相电压

RAW_FULL_SCALE = 32767.0
I_SECONDARY_FS = 4.0  # 电流通道二次侧满量程 ±4A（可覆盖 5 倍故障）
U_SECONDARY_FS = 120.0  # 电压通道二次侧满量程 ±120V

A_CURRENT = I_SECONDARY_FS / RAW_FULL_SCALE
A_VOLTAGE = U_SECONDARY_FS / RAW_FULL_SCALE


def primary_to_raw(value_primary: float, ratio: float, a: float) -> float:
    """一次值 → ADC 原始整数。"""
    return (value_primary / ratio) / a


#: 每通道的元数据（名称、相别、单位、a、b、变比、PS）
@dataclass(frozen=True)
class ChannelSpec:
    name: str
    phase: str
    unit: str
    a: float
    b: float
    primary: float
    secondary: float
    ps: str
    quantity: str  # 'I' 或 'U'
    ratio: float
    scale_pu: float  # 正常运行时该通道的标幺值


ANALOG_SPECS: tuple[ChannelSpec, ...] = (
    ChannelSpec("Ia", "A", "A", A_CURRENT, 0.0, I_PRIMARY, I_SECONDARY, "S", "I", I_RATIO, I_LOAD_PU),
    ChannelSpec("Ib", "B", "A", A_CURRENT, 0.0, I_PRIMARY, I_SECONDARY, "S", "I", I_RATIO, I_LOAD_PU),
    ChannelSpec("Ic", "C", "A", A_CURRENT, 0.0, I_PRIMARY, I_SECONDARY, "S", "I", I_RATIO, I_LOAD_PU),
    ChannelSpec("3I0", "N", "A", A_CURRENT, 0.0, I_PRIMARY, I_SECONDARY, "S", "I", I_RATIO, 0.0),
    ChannelSpec("Ua", "A", "V", A_VOLTAGE, 0.0, U_PRIMARY, U_SECONDARY, "S", "U", U_RATIO, U_RATED / math.sqrt(3.0)),
    ChannelSpec("Ub", "B", "V", A_VOLTAGE, 0.0, U_PRIMARY, U_SECONDARY, "S", "U", U_RATIO, U_RATED / math.sqrt(3.0)),
    ChannelSpec("Uc", "C", "V", A_VOLTAGE, 0.0, U_PRIMARY, U_SECONDARY, "S", "U", U_RATIO, U_RATED / math.sqrt(3.0)),
)

DIGITAL_SPECS: tuple[tuple[str, str], ...] = (
    ("保护动作", "A"),
    ("断路器跳闸", "A"),
    ("断路器合位", "A"),
    ("备用1", "A"),
)


# ---------------------------------------------------------------------------
# 波形生成
# ---------------------------------------------------------------------------

@dataclass
class WaveformSet:
    """合成的一组录波数据。"""

    time_axis: np.ndarray
    analog_primary: np.ndarray  # (A, N) 一次值
    digital: np.ndarray  # (D, N) bool
    fault_sample: int


def synthesize(
    sample_rate: float = 4000.0,
    duration: float = 0.4,
    fault_time: float = 0.1,
    jitter: float = 0.0,
    seed: int = 20260720,
) -> WaveformSet:
    """合成一段 A 相单相接地故障录波（定频采样）。"""
    n = int(round(sample_rate * duration))
    t = np.arange(n, dtype=np.float64) / sample_rate
    fault_sample = int(round(fault_time * sample_rate))
    return _build(t, fault_sample, jitter, seed)


def synthesize_multirate(
    pre_rate: float = 1000.0,
    post_rate: float = 4000.0,
    pre_duration: float = 0.4,
    post_duration: float = 0.3,
    fault_time: float = 0.1,
    jitter: float = 0.0,
    seed: int = 20260720,
) -> tuple[WaveformSet, list[tuple[float, int]]]:
    """合成一段变频采样录波：故障前后使用不同采样率。

    用于验证时间轴的分段推算逻辑（endsamp 是**累计编号**，最容易算错的地方）。

    Returns:
        ``(波形, 采样率分段)``，分段形如 ``[(1000.0, 400), (4000.0, 1600)]``。
    """
    n_pre = int(round(pre_rate * pre_duration))
    n_post = int(round(post_rate * post_duration))
    t_pre = np.arange(n_pre, dtype=np.float64) / pre_rate
    t_post = t_pre[-1] + 1.0 / pre_rate + np.arange(n_post, dtype=np.float64) / post_rate
    t = np.concatenate([t_pre, t_post])
    fault_sample = int(round(fault_time * pre_rate))
    segments = [(pre_rate, n_pre), (post_rate, n_pre + n_post)]
    return _build(t, fault_sample, jitter, seed), segments


def _build(t: np.ndarray, fault_sample: int, jitter: float, seed: int) -> WaveformSet:
    n = t.size
    rng = np.random.default_rng(seed)

    after = np.arange(n) >= fault_sample
    omega = 2.0 * math.pi * F_LINE * t

    # ---------------------------------------------------------------- 电流
    ia_pu = np.where(after, I_FAULT_PU, I_LOAD_PU)
    ib_pu = np.full(n, I_LOAD_PU)
    ic_pu = np.full(n, I_LOAD_PU)

    sqrt2 = math.sqrt(2.0)
    ia = ia_pu * I_RATED * sqrt2 * np.sin(omega)
    ib = ib_pu * I_RATED * sqrt2 * np.sin(omega - 2.0 * math.pi / 3.0)
    ic = ic_pu * I_RATED * sqrt2 * np.sin(omega + 2.0 * math.pi / 3.0)

    # 零序电流（接地故障的特征量）：故障后出现，量值约为故障相电流的 1/3
    i0 = np.where(after, ia * 0.33, 0.0)

    # ---------------------------------------------------------------- 电压
    u_nom = U_RATED / math.sqrt(3.0)
    ua_pu = np.where(after, U_FAULT_PU, 1.0)
    u_healthy_pu = np.where(after, U_HEALTHY_PU, 1.0)

    ua = ua_pu * u_nom * sqrt2 * np.sin(omega)
    ub = u_healthy_pu * u_nom * sqrt2 * np.sin(omega - 2.0 * math.pi / 3.0)
    uc = u_healthy_pu * u_nom * sqrt2 * np.sin(omega + 2.0 * math.pi / 3.0)

    analog = np.vstack([ia, ib, ic, i0, ua, ub, uc])

    if jitter:
        analog = analog + rng.normal(0.0, jitter, size=analog.shape) * np.abs(analog)

    # -------------------------------------------------------------- 开关量
    digital = np.zeros((len(DIGITAL_SPECS), n), dtype=bool)
    trip = min(fault_sample + 2, n - 1)
    digital[0, fault_sample:] = True  # 保护动作
    digital[1, trip:] = True  # 断路器跳闸
    digital[2, :] = True  # 断路器合位（故障前在合位）
    digital[2, trip:] = False  # 跳闸后分位

    return WaveformSet(time_axis=t, analog_primary=analog, digital=digital, fault_sample=fault_sample)


# ---------------------------------------------------------------------------
# 文件写出
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float, base_epoch: tuple[int, int, int, int, int, int] = (2026, 7, 20, 9, 15, 0)) -> str:
    """把"从起始时刻起的秒数"格式化为 COMTRADE 的 日/月/年,时:分:秒.微秒。"""
    import datetime as _dt

    start = _dt.datetime(*base_epoch)
    moment = start + _dt.timedelta(seconds=seconds)
    return moment.strftime("%d/%m/%Y,%H:%M:%S.%f")


@dataclass
class SampleSpec:
    """一份样例文件的完整描述。"""

    name: str
    version: int = 1999
    data_type: str = "BINARY"
    station: str = "110kV测试变"
    device: str = "DFR-01"
    line_freq: float = F_LINE
    multirate: bool = False
    inject_missing: bool = False
    chinese_names: bool = False
    encoding: str = "utf-8"
    fault_time: float = 0.1
    duration: float = 0.4
    sample_rate: float = 4000.0
    jitter: float = 0.0
    raw_full_scale: float | None = None
    """原始计数实际占用的满量程（None 表示占满 ±32767）。

    真实现场常见「cfg 声明 ±32767、实际计数只用了几百」的情况（SEL 装置即如此）。
    这个参数用于复现该现象，验证解析器不会据此误判为"已是工程量"。
    """
    notes: str = ""


def _rescale(specs: tuple[ChannelSpec, ...], raw_full_scale: float) -> tuple[ChannelSpec, ...]:
    """按「实际占用的满量程」重算 a。

    cfg 声明 ±32767，但装置实际只用了其中一小段，于是 a 必须相应放大，
    才能把原始计数换算回正确的二次值。
    """
    out = []
    for ch in specs:
        secondary_fs = I_SECONDARY_FS if ch.quantity == "I" else U_SECONDARY_FS
        out.append(
            ChannelSpec(
                ch.name, ch.phase, ch.unit, secondary_fs / raw_full_scale, ch.b,
                ch.primary, ch.secondary, ch.ps, ch.quantity, ch.ratio, ch.scale_pu,
            )
        )
    return tuple(out)


def _analog_specs_for(spec: SampleSpec) -> tuple[ChannelSpec, ...]:
    specs = _build_analog_specs(spec)
    if spec.raw_full_scale:
        specs = _rescale(specs, spec.raw_full_scale)
    return specs


def _build_analog_specs(spec: SampleSpec) -> tuple[ChannelSpec, ...]:
    if not spec.chinese_names:
        return ANALOG_SPECS
    replacements = {
        "Ia": ("A相电流", "A"),
        "Ib": ("B相电流", "B"),
        "Ic": ("C相电流", "C"),
        "3I0": ("零序电流", "N"),
        "Ua": ("A相电压", "A"),
        "Ub": ("B相电压", "B"),
        "Uc": ("C相电压", "C"),
    }
    out = []
    for ch in ANALOG_SPECS:
        name, phase = replacements.get(ch.name, (ch.name, ch.phase))
        out.append(
            ChannelSpec(
                name, phase, ch.unit, ch.a, ch.b, ch.primary, ch.secondary, ch.ps,
                ch.quantity, ch.ratio, ch.scale_pu,
            )
        )
    return tuple(out)


def write_sample(spec: SampleSpec, out_dir: Path) -> tuple[Path, Path]:
    """按描述生成一对 .cfg / .dat 文件，返回两个路径。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    analog_specs = _analog_specs_for(spec)

    if spec.multirate:
        wave, segments = synthesize_multirate(
            fault_time=spec.fault_time, jitter=spec.jitter
        )
    else:
        wave = synthesize(
            sample_rate=spec.sample_rate,
            duration=spec.duration,
            fault_time=spec.fault_time,
            jitter=spec.jitter,
        )
        segments = [(spec.sample_rate, wave.time_axis.size)]

    # 一次值 → 原始整数
    raw = np.empty_like(wave.analog_primary)
    for i, ch in enumerate(analog_specs):
        raw[i] = primary_to_raw(wave.analog_primary[i], ch.ratio, ch.a)
    raw = np.rint(raw).astype(np.int32)

    if spec.inject_missing:
        # 在第 5 与第 500 个采样点注入缺失值哨兵
        sentinel = 99999 if spec.data_type == "ASCII" else (
            -32768 if spec.version >= 1999 else -1
        )
        for idx in (5, 500):
            if idx < raw.shape[1]:
                raw[:, idx] = sentinel

    cfg_path = out_dir / f"{spec.name}.cfg"
    dat_path = out_dir / f"{spec.name}.dat"

    _write_cfg(cfg_path, spec, analog_specs, wave, segments)
    _write_dat(dat_path, spec, analog_specs, raw, wave)

    (out_dir / f"{spec.name}.notes.txt").write_text(
        f"{spec.name}\n{spec.notes}\n"
        f"版本={spec.version} 编码={spec.data_type} "
        f"采样率={segments} 采样点={wave.time_axis.size}\n"
        f"故障类型=单相接地(A相) 故障时刻={spec.fault_time}s "
        f"(第 {wave.fault_sample} 个采样点)\n",
        encoding="utf-8",
        newline="\n",
    )
    return cfg_path, dat_path


def _write_cfg(
    path: Path,
    spec: SampleSpec,
    analog_specs: tuple[ChannelSpec, ...],
    wave: WaveformSet,
    segments: list[tuple[float, int]],
) -> None:
    lines: list[str] = []

    # 行 1：1991 版没有版本年份字段
    if spec.version >= 1999:
        lines.append(f"{spec.station},{spec.device},{spec.version}")
    else:
        lines.append(f"{spec.station},{spec.device}")

    # 行 2
    lines.append(f"{len(analog_specs) + len(DIGITAL_SPECS)},{len(analog_specs)}A,{len(DIGITAL_SPECS)}D")

    # 模拟通道
    for i, ch in enumerate(analog_specs, start=1):
        if spec.data_type == "FLOAT32":
            # FLOAT32 通常直接存工程量，cfg 的 a/b 按惯例置为单位映射。
            # 若此处仍写原始计数换算系数，解析器会被误导（真实文件不会这样写）。
            a_effective, b_effective = 1.0, 0.0
        elif spec.version == 1991:
            # 1991 版没有 primary/secondary/PS 字段，录波器只能把变比直接烘焙进 a/b，
            # 否则解析出来的就只是二次值。这里按真实现场做法生成。
            a_effective, b_effective = ch.a * ch.ratio, ch.b
        else:
            a_effective, b_effective = ch.a, ch.b
        base = (
            f"{i},{ch.name},{ch.phase},LINE1,{ch.unit},"
            f"{a_effective:.10g},{b_effective:.10g},0,-32767,32767"
        )
        if spec.version >= 1999:
            base += f",{ch.primary:.10g},{ch.secondary:.10g},{ch.ps}"
        lines.append(base)

    # 开关量通道
    for i, (name, phase) in enumerate(DIGITAL_SPECS, start=1):
        normal = 0 if i <= 2 else 1
        if spec.version >= 1999:
            lines.append(f"{i},{name},{phase},LINE1,{normal}")
        else:
            lines.append(f"{i},{name},{normal}")

    # 频率与采样率
    lines.append(f"{spec.line_freq:.0f}")
    lines.append(f"{len(segments)}")
    for rate, end_sample in segments:
        lines.append(f"{rate:.0f},{end_sample}")

    # 起始与触发时刻
    total_duration = float(wave.time_axis[-1] - wave.time_axis[0])
    trigger_offset = spec.fault_time if not spec.multirate else spec.fault_time
    lines.append(_fmt_time(0.0))
    lines.append(_fmt_time(min(trigger_offset, total_duration)))

    lines.append(spec.data_type)

    if spec.version >= 2013:
        lines.append("1")  # time_mult
        lines.append("8h00,8h00")  # time_code, local_code
        lines.append("B,0")  # tmq_code, leapsec
    elif spec.version >= 1999:
        lines.append("1")  # time_mult

    path.write_text("\n".join(lines) + "\n", encoding=spec.encoding, newline="\n")


def _write_dat(
    path: Path,
    spec: SampleSpec,
    analog_specs: tuple[ChannelSpec, ...],
    raw: np.ndarray,
    wave: WaveformSet,
) -> None:
    """写出数据文件。"""
    n = raw.shape[1]
    timestamps = np.rint((wave.time_axis - wave.time_axis[0]) * 1e6).astype(np.int64)
    sample_numbers = np.arange(1, n + 1, dtype=np.int64)

    if spec.data_type == "ASCII":
        _write_ascii(path, spec, analog_specs, raw, wave, timestamps, sample_numbers)
        return

    value_dtype = {"BINARY": "<i2", "BINARY32": "<i4", "FLOAT32": "<f4"}[spec.data_type]
    words = -(-len(DIGITAL_SPECS) // 16)

    fields: list[tuple] = [("samp", "<u4"), ("ts", "<u4")]
    if len(analog_specs):
        fields.append(("analog", value_dtype, (len(analog_specs),)))
    if words:
        fields.append(("digital", "<u2", (words,)))

    arr = np.zeros(n, dtype=np.dtype(fields))
    arr["samp"] = sample_numbers
    arr["ts"] = timestamps

    if len(analog_specs):
        values = raw.astype(np.float64)
        if spec.data_type == "FLOAT32":
            # FLOAT32 按工程量写入（标准做法：a/b 置为单位映射）
            values = np.empty_like(raw, dtype=np.float64)
            for i, ch in enumerate(analog_specs):
                values[i] = raw[i].astype(np.float64) * ch.a + ch.b
        arr["analog"] = values.T.astype(arr["analog"].dtype)

    if words:
        packed = np.zeros((n, words), dtype=np.uint16)
        for d in range(len(DIGITAL_SPECS)):
            for s in range(n):
                if wave.digital[d, s]:
                    packed[s, d // 16] |= np.uint16(1 << (d % 16))
        arr["digital"] = packed

    arr.tofile(path)


def _write_ascii(
    path: Path,
    spec: SampleSpec,
    analog_specs: tuple[ChannelSpec, ...],
    raw: np.ndarray,
    wave: WaveformSet,
    timestamps: np.ndarray,
    sample_numbers: np.ndarray,
) -> None:
    n = raw.shape[1]
    out: list[str] = []
    for s in range(n):
        cols: list[str] = [str(sample_numbers[s]), str(timestamps[s])]
        for i, ch in enumerate(analog_specs):
            # 标准要求 ASCII 存原始计数，换算由解析器按 cfg 的 a/b 完成
            cols.append(str(int(raw[i, s])))
        cols.extend("1" if wave.digital[d, s] else "0" for d in range(len(DIGITAL_SPECS)))
        out.append(",".join(cols))
    path.write_text("\n".join(out) + "\n", encoding="ascii", newline="\n")


# ---------------------------------------------------------------------------
# 样例清单
# ---------------------------------------------------------------------------

def default_samples() -> list[SampleSpec]:
    """生成覆盖版本与编码组合的样例清单。"""
    return [
        SampleSpec(
            name="v1999_binary",
            version=1999,
            data_type="BINARY",
            notes="标准 1999 版二进制：模拟通道 13 字段、开关量 5 字段、含 time_mult",
        ),
        SampleSpec(
            name="v1991_ascii",
            version=1991,
            data_type="ASCII",
            notes="标准 1991 版 ASCII：模拟通道 10 字段（无变比）、开关量 3 字段、无 time_mult",
        ),
        SampleSpec(
            name="v2013_binary32",
            version=2013,
            data_type="BINARY32",
            notes="2013 版 int32 编码：含 time_code/local_code 与 tmq_code/leapsec 行",
        ),
        SampleSpec(
            name="v2013_float32",
            version=2013,
            data_type="FLOAT32",
            notes="2013 版 float32 编码：模拟量为工程量而非原始计数",
        ),
        SampleSpec(
            name="v1999_multirate",
            version=1999,
            data_type="BINARY",
            multirate=True,
            notes="变频采样：故障前 1000Hz、故障后 4000Hz，验证 endsamp 累计编号语义",
        ),
        SampleSpec(
            name="v1999_missing",
            version=1999,
            data_type="BINARY",
            inject_missing=True,
            notes="在第 5、500 点注入缺失值哨兵 0x8000，验证缺失值识别",
        ),
        SampleSpec(
            name="v1999_chinese_gbk",
            version=1999,
            data_type="BINARY",
            chinese_names=True,
            encoding="gbk",
            notes="中文通道名 + GBK 编码，验证编码兜底与中文名称的通道识别",
        ),
        SampleSpec(
            name="v1999_ascii_smallspan",
            version=1999,
            data_type="ASCII",
            raw_full_scale=100.0,
            notes="ASCII 原始计数只用到 ±100（cfg 却声明 ±32767），复现 SEL 等装置的现场情形；"
                  "验证仍按标准施加 a/b 换算，只给出 DAT-009 提示而不跳过换算",
        ),
        SampleSpec(
            name="v1999_noisy",
            version=1999,
            data_type="BINARY",
            jitter=0.03,
            notes="叠加 3% 随机噪声，验证算法模块的误差容忍度",
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 COMTRADE 样例录波文件")
    parser.add_argument("--out", default="tests/fixtures/generated", help="输出目录")
    parser.add_argument("--only", default=None, help="只生成指定名称的样例")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    specs = default_samples()
    if args.only:
        specs = [s for s in specs if s.name == args.only]
        if not specs:
            print(f"未找到样例「{args.only}」")
            return 1

    for spec in specs:
        cfg, dat = write_sample(spec, out_dir)
        print(f"已生成 {cfg.name} / {dat.name}  ({dat.stat().st_size} 字节)")
    print(f"\n共 {len(specs)} 组样例，输出目录：{out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
