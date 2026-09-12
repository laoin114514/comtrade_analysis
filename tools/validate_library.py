"""用真实 COMTRADE 样本库验证解析器。

项目自身没有客户样例，因此用公开样本库（如 fault-wave-analyzer 的
``fastapi/tests/fixtures/sample_library/``，12 组来自 GitHub MIT 仓库的样例，
其中 1 组只有 cfg 无 dat，见 README「实测结果」一节）
做兼容性回归。本工具不硬编码任何样本库路径，由参数指定。

用法::

    python tools/validate_library.py --library <样本库目录>
    python tools/validate_library.py --library <目录> --cross-check <另一解析模块的父目录>

``--cross-check`` 会把另一个解析器的输出与本品逐通道对比，
确认数值差异只来自「单位归一化 × PS 变比」这两步（本模块相对标准实现多做的工作）。
被对比的模块必须只依赖标准库（fault-wave-analyzer 的 ``app.comtrade`` 满足）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comtrade import try_load_recording  # noqa: E402
from comtrade.units import parse_unit  # noqa: E402


def discover(library: Path) -> list[Path]:
    """在样本库目录下递归找出所有 .cfg 文件。"""
    return sorted(
        p for p in library.rglob("*")
        if p.is_file() and p.suffix.lower() == ".cfg"
    )


def _paired_dat(cfg: Path) -> Path | None:
    """找出与 cfg 配对的 dat（扩展名大小写不敏感）。"""
    exact = cfg.with_suffix(".dat")
    if exact.exists():
        return exact
    for p in cfg.parent.iterdir():
        if p.is_file() and p.suffix.lower() == ".dat" and p.stem.lower() == cfg.stem.lower():
            return p
    return None


def _rms(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else 0.0


def validate(library: Path) -> int:
    """逐个解析样本，报告结果。返回失败数。"""
    cases = discover(library)
    if not cases:
        print(f"在 {library} 下没有找到 .cfg 文件")
        return 1

    failures = 0
    print("=" * 100)
    print(f"{'样本':<40}{'版本':<10}{'编码':<10}{'通道':<12}{'点数':<10}{'时长(s)':<12}判定")
    print("=" * 100)

    for cfg in cases:
        rel = cfg.relative_to(library)
        label = str(rel.parent) if rel.parent != Path(".") else cfg.stem

        if _paired_dat(cfg) is None:
            # 样本库里有只放 cfg 的固件（用于验证"缺少 dat 时的报错"），不算失败
            print(f"{label[:39]:<40}{'—':<10}{'—':<10}{'—':<12}{'—':<10}{'—':<12}"
                  "跳过（无配对 dat）")
            continue

        recording, diags = try_load_recording(cfg)

        if recording is None:
            failures += 1
            print(f"{label[:39]:<40}{'—':<10}{'—':<10}{'—':<12}{'—':<10}{'—':<12}失败")
            for d in diags:
                print(f"      {d}")
            continue

        m = recording.meta
        warn_count = sum(1 for d in diags if d.severity.value == "warning")
        channels = f"{m.analog_count}A/{m.digital_count}D"
        print(
            f"{label[:39]:<40}{m.version.label:<10}"
            f"{(m.data_type.value if m.data_type else '-'):<10}"
            f"{channels:<12}{recording.sample_count:<10}"
            f"{recording.duration:<12.6f}"
            f"OK（警告 {warn_count}）"
        )

        # 时间轴必须单调，否则按时间定位会全错
        axis = recording.time_axis
        if axis is not None and axis.size > 1 and not np.all(np.diff(axis) > 0):
            failures += 1
            print("      ★ 时间轴非单调")

        # 至少识别出三相电流或三相电压之一，否则算法模块无法工作
        if recording.by_role_map() and not any(
            r.value in {"IA", "IB", "IC", "UA", "UB", "UC"} for r in recording.by_role_map()
        ):
            print("      注意：未识别出任何三相电流/电压通道")

    print("=" * 100)
    print(f"样本 {len(cases)} 个，失败 {failures} 个")
    return failures


def cross_check(library: Path, module_parent: Path) -> int:
    """与另一个 COMTRADE 解析模块逐通道对比数值。"""
    sys.path.insert(0, str(module_parent))
    try:
        from app.comtrade import parse_recording  # type: ignore
    except ImportError as exc:
        print(f"无法导入对照解析器（{module_parent}）：{exc}")
        return 1

    total = matched = both_zero = mismatched = 0
    unexplained: list[tuple] = []

    for cfg in discover(library):
        dat = _paired_dat(cfg)
        if dat is None:
            continue

        mine, _ = try_load_recording(cfg)
        if mine is None:
            continue
        theirs = parse_recording(cfg.read_bytes(), dat.read_bytes())

        for i, ch in enumerate(mine.analog_channels):
            if i >= len(theirs.analog_values):
                break
            total += 1
            mine_v = np.asarray(ch.values, dtype=np.float64)
            theirs_v = np.asarray(theirs.analog_values[i], dtype=np.float64)
            n = min(mine_v.size, theirs_v.size)
            mask = (
                np.isfinite(mine_v[:n])
                & np.isfinite(theirs_v[:n])
                & (np.abs(theirs_v[:n]) > 1e-9)
            )
            if not mask.any():
                both_zero += 1
                continue

            ratio = float(np.median(mine_v[:n][mask] / theirs_v[:n][mask]))
            expected = parse_unit(ch.unit_raw).scale * (
                (ch.primary / ch.secondary)
                if (ch.ps == "S" and ch.primary and ch.secondary)
                else 1.0
            )
            if abs(ratio - expected) <= max(1e-6, abs(expected) * 1e-9):
                matched += 1
            else:
                mismatched += 1
                unexplained.append(
                    (cfg.parent.name, ch.name.strip(), round(ratio, 6), round(expected, 6))
                )

    print("=" * 100)
    print(f"通道 {total} 个：差异符合「单位×PS变比」的 {matched}，"
          f"双方均为零的 {both_zero}，不符 {mismatched}")
    for item in unexplained[:20]:
        print(f"   ★ {item}")
    return mismatched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="用真实样本库验证 COMTRADE 解析器")
    parser.add_argument("--library", required=True, help="样本库根目录")
    parser.add_argument("--cross-check", default=None,
                        help="对照解析模块的父目录（其下应有 app/comtrade 包）")
    args = parser.parse_args(argv)

    library = Path(args.library)
    if not library.is_dir():
        print(f"目录不存在：{library}")
        return 1

    failures = validate(library)
    if args.cross_check:
        print()
        failures += cross_check(library, Path(args.cross_check))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
