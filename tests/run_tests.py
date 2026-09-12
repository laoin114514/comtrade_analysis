"""轻量测试运行器（不依赖 pytest）。

交付给客户的源码不应强制依赖 pytest —— 但测试又必须能随时跑。
这个运行器用标准库实现收集与执行，``python tests/run_tests.py`` 即可。

如果机器上装了 pytest，``python -m pytest tests`` 同样可用
（测试函数都是普通的 ``test_*`` 函数，没有使用 pytest 专有 fixture）。
"""
from __future__ import annotations

import importlib
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = (
    "tests.test_units_and_channels",
    "tests.test_cfg",
    "tests.test_dat",
    "tests.test_convert",
    "tests.test_timebase",
    "tests.test_reader",
    "tests.test_roundtrip",
)


def run_module(module_name: str) -> tuple[int, int]:
    """运行单个模块内的全部 ``test_*`` 函数，返回 ``(通过, 失败)``。"""
    module = importlib.import_module(module_name)
    passed = failed = 0
    for name in sorted(dir(module)):
        if not name.startswith("test_"):
            continue
        func = getattr(module, name)
        if not callable(func):
            continue
        try:
            func()
        except Exception:  # noqa: BLE001 - 测试运行器需要捕获一切
            failed += 1
            print(f"  FAIL  {module_name}.{name}")
            traceback.print_exc(limit=3)
        else:
            passed += 1
    return passed, failed


def main() -> int:
    total_pass = total_fail = 0
    started = time.time()

    for module_name in MODULES:
        print(f"\n=== {module_name} ===")
        try:
            passed, failed = run_module(module_name)
        except Exception:  # noqa: BLE001 - 模块导入失败
            print(f"  模块导入/执行失败：{module_name}")
            traceback.print_exc()
            total_fail += 1
            continue
        total_pass += passed
        total_fail += failed
        marker = "OK" if failed == 0 else "FAIL"
        print(f"  [{marker}] 通过 {passed} / 失败 {failed}")

    elapsed = time.time() - started
    print("\n" + "=" * 60)
    print(f"合计：通过 {total_pass}，失败 {total_fail}，耗时 {elapsed:.2f}s")
    print("=" * 60)
    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
