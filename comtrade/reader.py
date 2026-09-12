"""解析模块对外主入口。

合同依据
    第一条 2.(1)① "支持 .cfg 配置文件解析、支持 .dat 数据文件读取……
                     提供异常文件检测及错误提示机制"
对应需求
    F-01（选择本地 COMTRADE 文件导入）、F-03（cfg/dat 自动匹配）、
    F-04（文件完整性检查）、F-05（非法文件格式检测）、F-06（显示导入文件信息）、
    P-06（建立统一内部数据格式供后续模块调用）

对外只暴露两个函数
    :func:`load_recording`      —— 失败时抛 :class:`~comtrade.diagnostics.ComtradeParseError`
    :func:`try_load_recording`  —— 永不抛异常，返回 ``(Recording | None, 诊断列表)``

    界面做单文件导入用前者（配合 try/except 展示错误）；
    批量处理、回归测试、客户样例批量验证用后者。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cfg_parser import compute_file_sha256, parse_cfg
from .channels import identify_all
from .convert import convert_analog_channels
from .dat_parser import parse_dat
from .diagnostics import (
    Code,
    ComtradeParseError,
    Diagnostic,
    DiagnosticCollector,
    ParseAbort,
    Severity,
)
from .encoding import read_text, sniff_is_text
from .models import Recording
from .options import ParseOptions, DEFAULT_OPTIONS
from .timebase import build_time_axis

#: 模块支持的文件扩展名
SUPPORTED_EXTENSIONS = (".cfg", ".dat")
#: 2013 版引入但本期未实现的单文件格式
KNOWN_UNSUPPORTED_EXTENSIONS = (".cff",)


# ---------------------------------------------------------------------------
# 文件配对与完整性检查
# ---------------------------------------------------------------------------

def resolve_pair(path: str | Path, diagnostics: DiagnosticCollector) -> tuple[Path, Path]:
    """由任意一个文件定位配对的 .cfg 与 .dat。

    对应 F-03"支持 cfg/dat 文件自动匹配"：用户拖入任意一个文件都能找到另一半，
    并处理大小写不一致的扩展名（``.CFG`` / ``.Dat``）。

    Raises:
        ParseAbort: 文件不存在、扩展名不支持或缺失配对文件。
    """
    given = Path(path)

    if not given.exists():
        diagnostics.fatal(
            Code.FILE_NOT_FOUND, f"文件不存在：{given}", location=given.name
        )
        raise ParseAbort("文件不存在")

    # 目录直接拒绝，避免后续异常信息难以理解
    if given.is_dir():
        diagnostics.fatal(
            Code.UNSUPPORTED_EXTENSION, f"路径是目录而非文件：{given}", location=given.name
        )
        raise ParseAbort("路径是目录")

    suffix = given.suffix.lower()

    if suffix in KNOWN_UNSUPPORTED_EXTENSIONS:
        diagnostics.fatal(
            Code.CFF_NOT_SUPPORTED,
            f"检测到 2013 版单文件格式 {given.suffix}，本期尚未支持；"
            "请提供配套的 .cfg + .dat 文件",
            location=given.name,
        )
        raise ParseAbort("cff 暂不支持")

    if suffix not in SUPPORTED_EXTENSIONS:
        diagnostics.fatal(
            Code.UNSUPPORTED_EXTENSION,
            f"不支持的文件类型「{given.suffix}」，仅支持 "
            + " / ".join(SUPPORTED_EXTENSIONS),
            location=given.name,
        )
        raise ParseAbort("扩展名不支持")

    if given.stat().st_size == 0:
        diagnostics.fatal(Code.FILE_EMPTY, "文件为空", location=given.name)
        raise ParseAbort("文件为空")

    cfg_path = given if suffix == ".cfg" else _find_sibling(given, ".cfg")
    dat_path = given if suffix == ".dat" else _find_sibling(given, ".dat")

    missing: list[str] = []
    if cfg_path is None:
        missing.append(".cfg")
    if dat_path is None:
        missing.append(".dat")
    if missing:
        diagnostics.fatal(
            Code.PAIR_MISSING,
            f"缺少配对的 {' 和 '.join(missing)} 文件。"
            "COMTRADE 的一组的录波必须同时包含 .cfg（配置）与 .dat（数据），"
            "两者缺一不可，且需放在同一目录、使用相同的主文件名",
            location=given.name,
        )
        raise ParseAbort("配对文件缺失")

    assert cfg_path is not None and dat_path is not None
    return cfg_path, dat_path


def _find_sibling(anchor: Path, extension: str) -> Path | None:
    """在同一目录下查找主文件名相同、扩展名匹配的文件（大小写不敏感）。"""
    stem = anchor.stem.lower()
    try:
        entries = list(anchor.parent.iterdir())
    except OSError:
        entries = []

    exact = anchor.with_suffix(extension)
    for entry in entries:
        if (
            entry.is_file()
            and entry.stem.lower() == stem
            and entry.suffix.lower() == extension
        ):
            return entry
    # 目录列举失败或未命中时，退化为直接拼接
    return exact if exact.exists() else None


def _check_cfg_looks_textual(cfg_path: Path, diagnostics: DiagnosticCollector) -> None:
    """cfg 必须是文本；若不是，最可能的原因是用户把 cfg 与 dat 传反了。"""
    if not sniff_is_text(cfg_path):
        diagnostics.warn(
            Code.FILE_LOOKS_BINARY,
            "配置文件内容疑似二进制，可能 .cfg 与 .dat 文件的内容被互换；"
            "解析将继续，但结果很可能不可用",
            location=cfg_path.name,
        )


def _load_companion_text(base: Path, extension: str, diagnostics: DiagnosticCollector) -> str | None:
    """读取可选的伴随文件（.hdr / .inf）。

    这两个文件可能含有人工填写的故障简报与线路信息，
    对报告模块（R-02 基础信息）有价值，且读取成本极低。
    """
    sibling = _find_sibling(base.with_suffix(extension), extension)
    if sibling is None:
        return None
    try:
        text = read_text(sibling, diagnostics)
    except OSError as exc:
        # 静默失败会让"报告里少了故障简报"变成无法解释的现象
        diagnostics.info(
            Code.COMPANION_UNREADABLE,
            f"伴随文件 {sibling.name} 存在但读取失败（{exc}），"
            "本次解析不包含该文件内容；报告中的故障简报等信息可能缺失",
            location=sibling.name,
        )
        return None
    return text.strip() or None


# ---------------------------------------------------------------------------
# 主解析流程
# ---------------------------------------------------------------------------

def load_recording(
    path: str | Path,
    options: ParseOptions | None = None,
) -> Recording:
    """解析一组 COMTRADE 录波文件。

    Args:
        path: ``.cfg`` 或 ``.dat`` 的路径，均可（自动配对）。
        options: 解析选项，默认 :data:`~comtrade.options.DEFAULT_OPTIONS`。

    Returns:
        :class:`~comtrade.models.Recording`

    Raises:
        ComtradeParseError: 无法产出有效数据。异常的 ``diagnostics`` 属性
            包含全部诊断信息，界面应展示给用户而不是让程序崩溃（NF-12）。
    """
    opts = options or DEFAULT_OPTIONS
    diagnostics = DiagnosticCollector()
    try:
        cfg_path, dat_path = resolve_pair(Path(path), diagnostics)
        return _load_pair(cfg_path, dat_path, opts, diagnostics)
    except ParseAbort as exc:
        raise ComtradeParseError(f"录波文件解析失败：{exc}", diagnostics.items) from exc
    except ComtradeParseError:
        raise
    except Exception as exc:  # noqa: BLE001 - 兜底：任何未预期异常都转为可控错误
        diagnostics.fatal(
            Code.SYS_UNEXPECTED_ERROR,
            f"解析过程中发生未预期的错误：{type(exc).__name__}: {exc}",
        )
        raise ComtradeParseError(f"解析失败：{exc}", diagnostics.items) from exc


def try_load_recording(
    path: str | Path,
    options: ParseOptions | None = None,
) -> tuple[Recording | None, list[Diagnostic]]:
    """永不抛异常的解析入口。

    Returns:
        ``(Recording 或 None, 诊断列表)``。批量处理与回归测试使用。
    """
    try:
        recording = load_recording(path, options)
        return recording, recording.diagnostics
    except ComtradeParseError as exc:
        return None, exc.diagnostics


def _load_pair(
    cfg_path: Path,
    dat_path: Path,
    options: ParseOptions,
    diagnostics: DiagnosticCollector,
) -> Recording:
    """完整解析流程：cfg → dat → 识别 → 换算 → 时间轴 → 组装。"""
    _check_cfg_looks_textual(cfg_path, diagnostics)

    # ------------------------------------------------------- 1. 解析配置文件
    parsed = parse_cfg(cfg_path, diagnostics, options)

    # --------------------------------------------------------- 2. 解析数据文件
    parsed_dat = parse_dat(
        dat_path,
        analog_count=len(parsed.analog_channels),
        digital_count=len(parsed.digital_channels),
        data_type=parsed.meta.data_type,
        declared_count=parsed.declared_sample_count,
        version=parsed.version,
        diagnostics=diagnostics,
        options=options,
    )

    # ----------------------------------------------------- 3. 通道角色识别
    identify_all(parsed.analog_channels, diagnostics, location=cfg_path.name)

    # --------------------------------------------------------- 4. 数值转换
    convert_analog_channels(
        parsed.analog_channels,
        parsed_dat.analog_raw,
        data_type=parsed.meta.data_type,
        version=parsed.version,
        diagnostics=diagnostics,
        options=options,
        location=dat_path.name,
    )

    # ------------------------------------------------------------ 5. 时间轴
    time_axis = build_time_axis(
        parsed.meta,
        parsed_dat.sample_numbers,
        parsed_dat.timestamps,
        diagnostics,
        location=dat_path.name,
        rate_tolerance=options.rate_tolerance,
        source=options.time_axis_source,
    )

    # ------------------------------------------------------- 6. 开关量数据
    if parsed_dat.digital is not None:
        for ch in parsed.digital_channels:
            if ch.index < parsed_dat.digital.shape[0]:
                ch.values = parsed_dat.digital[ch.index]

    # ------------------------------------------------------------ 7. 元数据收尾
    meta = parsed.meta
    meta.source_dat = dat_path
    meta.sample_count = parsed_dat.record_count
    meta.cfg_sha256 = compute_file_sha256(cfg_path)
    meta.dat_sha256 = compute_file_sha256(dat_path)
    meta.header_text = _load_companion_text(cfg_path, ".hdr", diagnostics)
    meta.info_text = _load_companion_text(cfg_path, ".inf", diagnostics)

    # 开关量通道为空但 dat 里解析出了数据，说明 cfg 声明有误
    if parsed_dat.digital is not None and not parsed.digital_channels:
        diagnostics.warn(
            Code.CFG_CHANNEL_COUNT_MISMATCH,
            "数据文件中存在开关量数据，但配置文件中未声明开关量通道",
            location=cfg_path.name,
        )

    recording = Recording(
        meta=meta,
        analog_channels=parsed.analog_channels,
        digital_channels=parsed.digital_channels,
        time_axis=time_axis,
        sample_numbers=parsed_dat.sample_numbers,
        diagnostics=diagnostics.items,
    )

    # ------------------------------------------------------------ 8. 结果自检
    _final_sanity_check(recording, diagnostics)
    recording.diagnostics = diagnostics.items

    return recording


def _final_sanity_check(recording: Recording, diagnostics: DiagnosticCollector) -> None:
    """解析完成后的整体自检 —— 把"静默错误"变成"显式诊断"。"""
    meta = recording.meta

    for ch in recording.analog_channels:
        if ch.values is None:
            continue
        values = ch.values
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            diagnostics.warn(
                Code.CHK_CHANNEL_ALL_INVALID,
                f"通道「{ch.name}」的全部采样点都无效（NaN），无法参与计算",
                location=meta.source_dat.name if meta.source_dat else "",
            )
            continue
        if not np.any(finite):
            continue
        # 常值通道：多半是通道没用、或 a/b 有问题
        if finite.size > 1 and float(finite.max() - finite.min()) == 0.0:
            diagnostics.info(
                Code.CHK_CHANNEL_CONSTANT,
                f"通道「{ch.name}」的数值恒定（{finite[0]:g}），请确认该通道是否有实际接线",
                location=meta.source_dat.name if meta.source_dat else "",
            )

    # 三相对称性提示：仅在通道齐全且没有其它严重问题时给出，避免噪音
    if meta.data_type is not None and not any(
        d.severity is Severity.FATAL for d in diagnostics
    ):
        _hint_missing_three_phase(recording, diagnostics)


def _hint_missing_three_phase(recording: Recording, diagnostics: DiagnosticCollector) -> None:
    """提示三相电流/电压是否齐全 —— 算法模块判断故障类型的前提。"""
    from .models import ChannelRole

    groups = {
        "三相电流": (ChannelRole.IA, ChannelRole.IB, ChannelRole.IC),
        "三相电压": (ChannelRole.UA, ChannelRole.UB, ChannelRole.UC),
    }
    present = recording.by_role_map()
    for label, roles in groups.items():
        found = [r.value for r in roles if r in present]
        missing = [r.value for r in roles if r not in present]
        if missing and found:
            diagnostics.info(
                Code.CHK_THREE_PHASE_INCOMPLETE,
                f"{label}不完整：已识别 {'/'.join(found)}，缺少 {'/'.join(missing)}；"
                "依赖三相量的故障判据（如三相短路、两相短路）将无法执行",
                location=recording.meta.source_cfg.name if recording.meta.source_cfg else "",
            )
        elif not found:
            diagnostics.info(
                Code.CHK_THREE_PHASE_MISSING,
                f"未识别到任何{label}通道，请确认通道命名或人工指定映射",
                location=recording.meta.source_cfg.name if recording.meta.source_cfg else "",
            )


__all__ = [
    "load_recording",
    "try_load_recording",
    "resolve_pair",
    "SUPPORTED_EXTENSIONS",
]
