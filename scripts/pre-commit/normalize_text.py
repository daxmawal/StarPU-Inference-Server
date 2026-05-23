#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


UTF8_BOM = b"\xef\xbb\xbf"
TRAILING_WHITESPACE = " \t\f\v"


def is_binary(data: bytes) -> bool:
    return b"\x00" in data


def normalize_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(
        line.rstrip(TRAILING_WHITESPACE) for line in normalized.split("\n")
    )
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def normalize_file(path: Path) -> None:
    raw = path.read_bytes()
    if is_binary(raw):
        return

    had_bom = raw.startswith(UTF8_BOM)
    if had_bom:
        raw = raw[len(UTF8_BOM) :]

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return

    normalized = normalize_text(text)
    if had_bom or normalized != text:
        path.write_text(normalized, encoding="utf-8", newline="\n")


def main(argv: list[str]) -> int:
    for name in argv[1:]:
        path = Path(name)
        if not path.is_file():
            continue
        normalize_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
