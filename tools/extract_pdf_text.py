from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        parts.append(f"\n\n--- PAGE {i+1} ---\n\n{text}")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = extract_pdf_text(args.pdf)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
