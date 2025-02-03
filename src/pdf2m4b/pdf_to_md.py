#!/usr/bin/env python3
"""
pdf_to_md.py

Converts a PDF to Markdown using pymupdf4llm.
"""

import sys
from pathlib import Path
import pymupdf4llm
from .logger_config import logger

def pdf_to_md(pdf_path: Path, output_md: Path) -> None:
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as e:
        logger.error("Error converting PDF to Markdown", pdf=str(pdf_path), error=str(e))
        raise

    try:
        output_md.write_text(md_text, encoding="utf-8")
    except Exception as e:
        logger.error("Error writing Markdown file", output=str(output_md), error=str(e))
        raise

    logger.info("Markdown written", output=str(output_md))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        logger.error("Usage: {} <input_pdf> <output_md>".format(sys.argv[0]))
        sys.exit(1)
    pdf_to_md(Path(sys.argv[1]), Path(sys.argv[2]))
