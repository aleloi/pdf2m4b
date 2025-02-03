#!/usr/bin/env python3
"""
main.py

End-to-end audiobook generation:
  1. Convert the input PDF to Markdown.
  2. Convert the Markdown to a folder structure.
  3. Run TTS synthesis (AWS Polly) on each chapter Markdown.
  4. Combine the resulting audio segments into a final M4B audiobook.
"""

import argparse
from pathlib import Path
from .logger_config import logger
from .pdf_to_md import pdf_to_md
from .md_to_folders import convert_md
from .tts_polly import run_tts
from .make_m4b import create_m4b

def main():
    parser = argparse.ArgumentParser(description="Create an audiobook from a PDF.")
    parser.add_argument("--pdf", required=True, help="Path to the input PDF file")
    parser.add_argument("--out_dir", default="output", help="Output directory for intermediate files")
    parser.add_argument("--audiobook", default="output.m4b", help="Filename for the final M4B audiobook")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_dir = Path(args.out_dir)
    try:
        out_dir.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logger.error("Error creating output directory", directory=str(out_dir), exc_info=e)
        raise

    # Step 1: Convert PDF to Markdown.
    md_file = out_dir / "output.md"
    logger.info("Converting PDF to Markdown", pdf=str(pdf_path), markdown=str(md_file))
    pdf_to_md(pdf_path, md_file)

    # Step 2: Convert Markdown to folder structure.
    chapters_dir = out_dir / "chapters"
    try:
        chapters_dir.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        logger.error("Error creating chapters directory", directory=str(chapters_dir), exc_info=e)
        raise
    logger.info("Converting Markdown to folder structure", markdown=str(md_file), chapters=str(chapters_dir))
    convert_md(md_file, chapters_dir)

    # Step 3: Run TTS synthesis.
    logger.info("Starting TTS synthesis", chapters=str(chapters_dir))
    run_tts(chapters_dir)

    # Step 4: Combine audio segments into the final audiobook.
    logger.info("Combining audio segments into M4B", audiobook=args.audiobook)
    create_m4b(chapters_dir, output_file=args.audiobook)

    logger.info("Audiobook creation complete")

if __name__ == "__main__":
    main()
