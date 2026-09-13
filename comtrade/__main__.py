"""命令行入口：快速查看一份录波的解析结果与诊断信息。

用于开发调试与现场排查 —— 客户发来一个"打不开"的文件时，
先跑这个命令，看诊断信息里给出了什么。

用法::

    python -m comtrade <文件路径>            # 摘要
    python -m comtrade <文件路径> --json     # 结构化输出（供其它工具消费）
    python -m comtrade <文件路径> --quiet    # 只输出诊断，不出摘要
"""
from __future__ import annotations

import argparse
import json
import sys

from .models import ChannelRole
from .reader import try_load_recording


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m comtrade",
        description="COMTRADE 录波文件解析与诊断（.cfg / .dat，1991/1999/2013）",
    )
    parser.add_argument("path", help=".cfg 或 .dat 文件路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结构化结果")
    parser.add_argument("--quiet", action="store_true", help="只输出诊断信息")
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="保留原始采样值（便于核对 a/b 换算，内存占用翻倍）",
    )
    parser.add_argument(
        "--ascii-scaling",
        choices=["always", "never"],
        default="always",
        help="ASCII 数据的 a/b 换算策略（默认 always，按标准施加 a/b 换算）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from .options import ParseOptions

    options = ParseOptions(keep_raw_values=args.keep_raw, ascii_scaling=args.ascii_scaling)
    recording, diagnostics = try_load_recording(args.path, options)

    if args.json:
        payload = {
            "ok": recording is not None,
            "diagnostics": [
                {
                    "code": d.code,
                    "severity": d.severity.value,
                    "message": d.message,
                    "location": d.location,
                    "detail": d.detail,
                }
                for d in diagnostics
            ],
        }
        if recording is not None:
            meta = recording.meta
            payload["meta"] = {
                "station_name": meta.station_name,
                "device_id": meta.device_id,
                "version": meta.version.label,
                "revision_year": meta.revision_year,
                "data_type": meta.data_type.value if meta.data_type else None,
                "time_mult": meta.time_mult,
                "line_frequency": meta.line_frequency,
                "sample_count": meta.sample_count,
                "duration": recording.duration,
                "start_time": meta.start_time.isoformat() if meta.start_time else None,
                "trigger_time": meta.trigger_time.isoformat() if meta.trigger_time else None,
                "sample_rate_segments": [
                    {"rate_hz": s.rate_hz, "end_sample": s.end_sample}
                    for s in meta.sample_rate_segments
                ],
                "analog_count": meta.analog_count,
                "digital_count": meta.digital_count,
                "cfg_sha256": meta.cfg_sha256,
                "dat_sha256": meta.dat_sha256,
                "cfg_size_bytes": meta.cfg_size_bytes,
                "dat_size_bytes": meta.dat_size_bytes,
            }
            payload["analog_channels"] = [
                {
                    "index": ch.index,
                    "name": ch.name,
                    "role": ch.role.value,
                    "role_confidence": ch.role_confidence,
                    "role_source": ch.role_source,
                    "unit": ch.unit,
                    "unit_raw": ch.unit_raw,
                    "a": ch.a,
                    "b": ch.b,
                    "ps": ch.ps,
                    "ps_raw": ch.ps_raw,
                    "primary": ch.primary,
                    "secondary": ch.secondary,
                    "invalid_count": ch.invalid_count,
                }
                for ch in recording.analog_channels
            ]
            payload["digital_channels"] = [
                {
                    "index": ch.index,
                    "name": ch.name,
                    "normal_state": ch.normal_state,
                    "transitions": ch.transitions().tolist()[:50],
                }
                for ch in recording.digital_channels
            ]
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0 if recording is not None else 1

    if recording is None:
        print("解析失败：", file=sys.stderr)
        for d in diagnostics:
            print(f"  {d}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(recording.summary())
        roles = recording.by_role_map()
        missing = [
            r.value
            for r in (ChannelRole.IA, ChannelRole.IB, ChannelRole.IC)
            if r not in roles
        ]
        if missing:
            print(f"\n注意：未识别到 {'/'.join(missing)}，算法模块无法执行三相相关判据")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
