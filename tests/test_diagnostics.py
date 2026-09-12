"""诊断体系自身的测试：错误码登记完整性。

对应需求：P-07（异常解析处理）、NF-12（异常不崩溃）

这一组测的不是某个解析分支，而是"诊断体系本身"。

为什么需要它
    错误码是界面提示语与测试断言的共同依据，README 也明确写着
    "完整错误码表见 comtrade/diagnostics.py 的 Code 类"。
    如果某个码只以字符串字面量散落在业务代码里，界面按码枚举时就
    会漏掉它 —— 回归案例：CHK-001~004 与 SYS-001 曾经就是这样，
    而 CHK-003（三相量不完整）恰恰是算法模块判断故障类型的前置条件。
"""
from __future__ import annotations

import re
from pathlib import Path

from comtrade.diagnostics import Code, Diagnostic, Severity
from comtrade.models import ComtradeVersion, Metadata, Recording

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "comtrade"

#: 只匹配带引号的、符合命名规则的错误码，避免命中 "UTF-8" 这类普通字符串
_CODE_LITERAL = re.compile(r"""["']((?:FIL|CFG|DAT|CHN|TIM|CHK|SYS)-[A-Z0-9-]+)["']""")


def registered_codes() -> set[str]:
    """Code 类中已登记的全部错误码。"""
    return {
        value
        for name, value in vars(Code).items()
        if not name.startswith("_") and isinstance(value, str)
    }


def _recording(*diagnostics: Diagnostic) -> Recording:
    meta = Metadata(
        station_name="S",
        device_id="D",
        revision_year=1999,
        version=ComtradeVersion.V1999,
        source_cfg=None,
        source_dat=None,
    )
    return Recording(meta=meta, diagnostics=list(diagnostics))


# ---------------------------------------------------------------------------
# 登记表本身
# ---------------------------------------------------------------------------

def test_registry_values_are_unique():
    """一个错误码只能有一个名字。

    重复定义意味着改含义时必然漏改一处，而"错误码一旦发布不应修改含义"
    是本模块对界面层的承诺。
    """
    seen: dict[str, str] = {}
    for name, value in vars(Code).items():
        if name.startswith("_") or not isinstance(value, str):
            continue
        assert value not in seen, (
            f"{value} 被 {seen[value]} 与 {name} 重复定义"
        )
        seen[value] = name


def test_every_emitted_code_literal_is_registered():
    """守卫：业务代码里出现的错误码必须已登记在 Code 表。

    这是本次补漏的核心断言 —— 任何人在 comtrade/ 里新写一个
    ``diagnostics.warn("XXX-999", ...)`` 都会在这里失败。
    """
    registry = registered_codes()
    unregistered: list[str] = []

    for path in sorted(PACKAGE_DIR.glob("*.py")):
        if path.name == "diagnostics.py":
            continue  # 这里就是登记表本身
        source = path.read_text(encoding="utf-8")
        for literal in _CODE_LITERAL.findall(source):
            if literal not in registry:
                unregistered.append(f"{path.name}: {literal}")

    assert not unregistered, (
        "以下错误码未登记在 Code 表中（界面按码组织提示语时会漏掉）："
        + "、".join(unregistered)
    )


def test_check_and_sys_codes_are_registered():
    """CHK（结果自检）与 SYS（兜底异常）两组码必须在表内。

    回归：这 5 个码曾经只以字符串字面量存在于 reader.py。
    """
    registry = registered_codes()
    for expected in ("CHK-001", "CHK-002", "CHK-003", "CHK-004", "SYS-001"):
        assert expected in registry, f"{expected} 未登记在 Code 表中"


def test_success_markers_are_registered():
    """正常路径的 INFO 级标记也要登记。

    回归：CFG-OK / DAT-OK / TIM-OK / CFG-ENC / DAT-DIG-PAD /
    TIM-TS-ZERO / TIM-TS-INVALID 曾经是散落的字面量。
    """
    registry = registered_codes()
    for expected in (
        "CFG-OK", "DAT-OK", "TIM-OK",
        "CFG-ENC", "DAT-DIG-PAD", "TIM-TS-ZERO", "TIM-TS-INVALID",
    ):
        assert expected in registry, f"{expected} 未登记在 Code 表中"


# ---------------------------------------------------------------------------
# is_reliable 的判据
# ---------------------------------------------------------------------------

def test_is_reliable_is_false_only_for_data_integrity_codes():
    """只有数据完整性问题才判定为"不可信"，可容忍的元数据瑕疵不算。"""
    # 数据完整性问题 → 不可信
    for code in (Code.DAT_SIZE_MISMATCH, Code.DAT_RECORD_SHORT, Code.TIM_NOT_MONOTONIC):
        assert _recording(Diagnostic(code, Severity.WARNING, "m")).is_reliable is False, code

    # 元数据瑕疵 → 仍然可信（出结果、标黄提示即可）
    for code in (Code.CHN_RATIO_MISSING, Code.DAT_ASCII_PRESCALED, Code.CHK_THREE_PHASE_INCOMPLETE):
        assert _recording(Diagnostic(code, Severity.WARNING, "m")).is_reliable is True, code

    # INFO 级一律不影响可信度
    assert _recording(
        Diagnostic(Code.TIM_OK, Severity.INFO, "m"),
        Diagnostic(Code.CHK_THREE_PHASE_INCOMPLETE, Severity.INFO, "m"),
    ).is_reliable is True


def test_no_diagnostics_means_reliable():
    assert _recording().is_reliable is True
