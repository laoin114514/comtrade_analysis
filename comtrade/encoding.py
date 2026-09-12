"""文本读取的编码兜底。

对应需求：NF-31（文件兼容）、NF-12（异常文件不崩溃）

背景
    COMTRADE 标准未强制编码：
    * 1991/1999 名义上是 ASCII，但国内录波器普遍把中文按 **GBK/GB18030** 写入；
    * 2013 版引入 UTF-8 的 ``.cff``，但 ``.cfg`` 仍可能是本地编码；
    * 进口装置与国外公开样例常见 **CP1251（俄文）** 与 **Latin-1（西欧）**。

    关键难点：**GB18030 几乎能解码任意字节序列**，如果把它放在兜底链中间，
    俄文/葡文文件都会被"成功"解码成乱码汉字而不报错。实测公开样例：

    * 俄文 ``Неизвестный регистратор`` → 误读为 ``Íåèçâåñòíûé ðåãèñòðàòîð``
    * 葡文 ``Estação de Medição`` → 误读为 ``Esta玢o de Medi玢o``

判定顺序
    1. UTF-8（含 BOM）；
    2. 非 ASCII 字节数够多且西里尔字母占比高 → **CP1251**；
    3. GB18030 解码成功，且结果不像"拉丁文本被误读成汉字" → **中文**；
    4. 兜底 **Latin-1**（单字节编码，永不失败）。

    第 2 步必须先于第 3 步：GB18030 能把西里尔字节解成汉字且不报错。
"""
from __future__ import annotations

import re
from pathlib import Path

from .diagnostics import Code, DiagnosticCollector

#: 尝试顺序：带 BOM 的 UTF-8 → 无 BOM UTF-8
_UTF8_ENCODINGS = ("utf-8-sig", "utf-8")

#: 出现这些字符基本可以判定文件不是文本
_BINARY_PROBE = ("\x00",)

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_CJK = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
#: CJK 字符被夹在两个 ASCII 字母之间 —— 拉丁文本被误读成汉字的典型特征
_SANDWICHED_CJK = re.compile(r"[A-Za-z][\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF][A-Za-z]")

#: 非 ASCII 字节达到该数量才考虑 CP1251 俄文
_MIN_CYRILLIC_BYTES = 20
_CYRILLIC_RATIO = 0.7
#: cp1251 解码结果里"非西里尔字母、非空白、非 ASCII"的字符占比上限。
#: 真正的俄文文本高字节全部落在 0xC0-0xFF（西里尔字母区）；
#: GBK 中文字节落在 0x80-0xBF 的部分会解成标点符号，
#: 据此把"中文被误判成俄文"挡掉（实测俄文 0%、GBK 中文 26%）。
_SYMBOL_RATIO_LIMIT = 0.05
#: 被夹在字母之间的汉字占比超过该值，判定为"拉丁文本被误读"
_MOJIBAKE_RATIO = 0.3


class TextDecodeResult:
    """解码结果。"""

    __slots__ = ("text", "encoding", "had_decode_errors")

    def __init__(self, text: str, encoding: str, had_decode_errors: bool) -> None:
        self.text = text
        self.encoding = encoding
        self.had_decode_errors = had_decode_errors


def _non_ascii_count(raw: bytes) -> int:
    return sum(1 for b in raw if b >= 0x80)


def _looks_like_latin_mojibake(text: str) -> bool:
    """判断"能解码成汉字"的文本其实是拉丁字母文本被误读。

    依据：真正的中文里汉字很少被夹在两个英文字母之间
    （``A相电流`` 中汉字是连续的一段）；而被误读的拉丁文本会产生
    ``Esta玢o`` 这种"字母-汉字-字母"的三明治结构。
    """
    cjk = _CJK.findall(text)
    if not cjk:
        return False
    sandwiched = len(_SANDWICHED_CJK.findall(text))
    return sandwiched / len(cjk) > _MOJIBAKE_RATIO


def decode_bytes(
    raw: bytes, diagnostics: DiagnosticCollector, location: str
) -> TextDecodeResult:
    """把字节流解码为文本，自动挑选编码。"""
    # ------------------------------------------------------------ 1. UTF-8
    for enc in _UTF8_ENCODINGS:
        try:
            return TextDecodeResult(raw.decode(enc), enc, had_decode_errors=False)
        except UnicodeDecodeError:
            continue

    if any(marker in raw.decode("latin-1") for marker in _BINARY_PROBE):
        diagnostics.warn(
            Code.FILE_LOOKS_BINARY,
            "文件内容疑似二进制，可能 cfg 与 dat 文件传反了",
            location=location,
        )

    non_ascii = _non_ascii_count(raw)
    if non_ascii == 0:
        # 纯 ASCII 却走不到 UTF-8 分支，只可能是 BOM 异常
        return TextDecodeResult(raw.decode("ascii", errors="replace"), "ascii", True)

    # --------------------------------------------------- 2. CP1251（俄文）
    # 必须先于 GB18030：GB18030 能把西里尔字节解成汉字，且不会报错。
    # 但 GBK 中文字节用 cp1251 解出来也大多是西里尔字母，因此仅看"西里尔占比"
    # 会把中文误判成俄文 —— 还要看有没有解出符号（见 _SYMBOL_RATIO_LIMIT）。
    cp1251_text = raw.decode("cp1251", errors="replace")
    if non_ascii >= _MIN_CYRILLIC_BYTES:
        high = [ch for ch in cp1251_text if ord(ch) > 0x7F]
        if high:
            cyrillic = sum(1 for ch in high if 0x0400 <= ord(ch) <= 0x04FF)
            symbols = sum(1 for ch in high if 0x0400 > ord(ch) or ord(ch) > 0x04FF)
            if (
                cyrillic >= len(high) * _CYRILLIC_RATIO
                and symbols <= len(high) * _SYMBOL_RATIO_LIMIT
            ):
                diagnostics.info(
                    Code.CFG_ENCODING_FALLBACK,
                    "文件不是 UTF-8 编码，已按 cp1251（西里尔/俄文）解码",
                    location=location,
                )
                return TextDecodeResult(cp1251_text, "cp1251", False)

    # ------------------------------------------------------ 3. GB18030（中文）
    try:
        gb_text = raw.decode("gb18030")
    except UnicodeDecodeError:
        pass
    else:
        if not _looks_like_latin_mojibake(gb_text):
            diagnostics.info(
                Code.CFG_ENCODING_FALLBACK,
                "文件不是 UTF-8 编码，已按 gb18030 解码（常见于国内录波器的中文通道名）",
                location=location,
            )
            return TextDecodeResult(gb_text, "gb18030", False)

    # --------------------------------------------------------- 4. Latin-1 兜底
    diagnostics.info(
        Code.CFG_ENCODING_FALLBACK,
        "无法确定编码（非 UTF-8 / 非中文 / 非俄文），已按 latin-1 解码；"
        "若厂站名或通道名显示为乱码，请提供该文件的正确编码",
        location=location,
    )
    return TextDecodeResult(raw.decode("latin-1"), "latin-1", False)


def read_text(path: Path, diagnostics: DiagnosticCollector) -> str:
    """读取文本文件（带编码兜底），返回解码后的全文。"""
    return decode_bytes(path.read_bytes(), diagnostics, location=path.name).text


def sniff_is_text(path: Path, probe_bytes: int = 4096) -> bool:
    """粗略判断文件是否为文本，用于识别"cfg/dat 传反了"。"""
    try:
        head = path.open("rb").read(probe_bytes)
    except OSError:
        return False
    if not head:
        return True
    if b"\x00" in head:
        return False
    # 控制字符占比过高也判定为非文本
    control = sum(1 for b in head if b < 0x09 or (0x0E <= b < 0x20))
    return control / len(head) < 0.05


__all__ = ["TextDecodeResult", "decode_bytes", "read_text", "sniff_is_text"]
