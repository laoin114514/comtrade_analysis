"""文本读取的编码兜底。

对应需求：NF-31（文件兼容）、NF-12（异常文件不崩溃）

背景
    COMTRADE 1991/1999 的 .cfg 名义上是 ASCII，但国内录波器普遍把
    中文厂站名/通道名按 GBK 写入；2013 版才引入 UTF-8 的 .cff。
    用 UTF-8 硬读 GBK 文件会得到乱码或直接抛 UnicodeDecodeError。
    这里按"UTF-8 优先、GB18030 兜底、latin-1 保底"的顺序尝试，
    并把实际使用的编码记入诊断，便于排查。
"""
from __future__ import annotations

from pathlib import Path

from .diagnostics import Code, DiagnosticCollector

# 尝试顺序：带 BOM 的 UTF-8 → 无 BOM UTF-8 → 简体中文 → 中文全字库 → 永不失败的保底
_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "big5", "latin-1")

# 出现这些字符基本可以判定文件不是文本
_BINARY_PROBE = ("\x00",)


class TextDecodeResult:
    """解码结果。"""

    __slots__ = ("text", "encoding", "had_decode_errors")

    def __init__(self, text: str, encoding: str, had_decode_errors: bool) -> None:
        self.text = text
        self.encoding = encoding
        self.had_decode_errors = had_decode_errors


def decode_bytes(raw: bytes, diagnostics: DiagnosticCollector, location: str) -> TextDecodeResult:
    """把字节流解码为文本，自动挑选编码。"""
    for enc in _ENCODINGS:
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue

        # latin-1 永不失败，用它兜底时几乎必然有乱码，需要额外判断
        if enc == "latin-1" and any(ch in text for ch in _BINARY_PROBE):
            diagnostics.warn(
                Code.FILE_LOOKS_BINARY,
                "文件内容疑似二进制，可能 cfg 与 dat 文件传反了",
                location=location,
            )
        if enc in ("gb18030", "big5", "latin-1"):
            diagnostics.info(
                "CFG-ENC",
                f"文件不是 UTF-8 编码，已按 {enc} 解码（常见于国内录波器的中文通道名）",
                location=location,
            )
        return TextDecodeResult(text, enc, had_decode_errors=False)

    # 理论上不可达（latin-1 不会失败），保留以防输入被替换
    text = raw.decode("latin-1", errors="replace")
    return TextDecodeResult(text, "latin-1/replace", had_decode_errors=True)


def read_text(path: Path, diagnostics: DiagnosticCollector) -> str:
    """读取文本文件（带编码兜底），返回解码后的全文。"""
    raw = path.read_bytes()
    return decode_bytes(raw, diagnostics, location=path.name).text


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
