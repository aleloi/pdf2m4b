#!/usr/bin/env python3
"""
make_m4b.py

Collects audio segments from the folder hierarchy, generates chapter metadata,
and calls FFmpeg to create an M4B audiobook.
"""

import os
import subprocess
import sys
import re
from dataclasses import dataclass
from typing import List
from pathlib import Path

from .logger_config import logger

OUTPUT_M4B_DEFAULT = "output.m4b"
CONCAT_LIST_FILENAME = "concat_list.txt"
METADATA_FILENAME = "chapters.txt"
AUDIO_EXT = ".ogg"  # Adjust if using a different format

@dataclass
class Segment:
    file_path: str  # Full path to the audio file
    title: str      # Chapter title (the folder’s relative path)

def get_duration_ms(file_path: str) -> int:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        duration_ms = int(float(result.stdout.strip()) * 1000)
        return duration_ms
    except Exception as e:
        logger.error("Error getting duration", file_path=file_path, error=str(e))
        sys.exit(1)

def sanitize_title(raw_title: str) -> str:
    parts = raw_title.split(os.sep)
    sanitized_parts = []
    for part in parts:
        part = re.sub(r'^\d+_+', '', part)
        part = part.replace('_', ' ').strip().title()
        sanitized_parts.append(part)
    return ": ".join(sanitized_parts)

def collect_segments(root: str, rel: str = "") -> List[Segment]:
    segments: List[Segment] = []
    current_dir = os.path.join(root, rel) if rel else root
    try:
        entries = sorted(os.listdir(current_dir))
    except Exception as e:
        logger.error("Error reading directory", directory=current_dir, error=str(e))
        sys.exit(1)

    ogg_files = sorted(
        f for f in entries
        if f.lower().endswith(AUDIO_EXT) and os.path.isfile(os.path.join(current_dir, f))
    )
    if ogg_files:
        audio_file = os.path.join(current_dir, ogg_files[0])
        title = rel if rel else os.path.basename(os.path.abspath(root))
        segments.append(Segment(audio_file, title))

    for entry in entries:
        full_path = os.path.join(current_dir, entry)
        if os.path.isdir(full_path):
            new_rel = os.path.join(rel, entry) if rel else entry
            segments.extend(collect_segments(root, new_rel))
    return segments

def create_m4b(book_root: Path, output_file: str = OUTPUT_M4B_DEFAULT) -> None:
    logger.info("Scanning for OGG files...")
    segments = collect_segments(str(book_root))
    if not segments:
        logger.error("No OGG audio files found in the folder hierarchy")
        sys.exit(1)
    logger.info("Found segments", segments_count=len(segments))

    concat_list = Path(CONCAT_LIST_FILENAME)
    try:
        with concat_list.open("w", encoding="utf-8") as clist:
            for seg in segments:
                abs_path = os.path.abspath(seg.file_path)
                clist.write(f"file '{abs_path}'\n")
        logger.info("Wrote concat list", filename=CONCAT_LIST_FILENAME)
    except Exception as e:
        logger.error("Error writing concat list", filename=CONCAT_LIST_FILENAME, error=str(e))
        sys.exit(1)

    metadata_lines = [";FFMETADATA1"]
    current_start = 0
    for seg in segments:
        dur = get_duration_ms(seg.file_path)
        current_end = current_start + dur
        clean_title = sanitize_title(seg.title)
        metadata_lines.extend([
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={current_start}",
            f"END={current_end}",
            f"title={clean_title}",
            ""
        ])
        logger.info("Added chapter", title=clean_title, start_ms=current_start, end_ms=current_end)
        current_start = current_end

    meta_file = Path(METADATA_FILENAME)
    try:
        with meta_file.open("w", encoding="utf-8") as mf:
            mf.write("\n".join(metadata_lines))
        logger.info("Wrote chapter metadata", filename=METADATA_FILENAME)
    except Exception as e:
        logger.error("Error writing metadata", filename=METADATA_FILENAME, error=str(e))
        sys.exit(1)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "quiet",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-i", str(meta_file),
        "-map_metadata", "1",
        "-c:a", "aac",
        output_file
    ]
    logger.info("Running FFmpeg", command=" ".join(ffmpeg_cmd))
    try:
        subprocess.run(ffmpeg_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg error", error=str(e))
        sys.exit(1)
    logger.info("Successfully created M4B file", output_file=output_file)

    # Clean up temporary files
    if concat_list.exists():
        concat_list.unlink()
    if meta_file.exists():
        meta_file.unlink()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python make_m4b.py /path/to/book_root")
        sys.exit(1)
    book_root = Path(sys.argv[1])
    if not book_root.is_dir():
        logger.error("Provided path is not a directory", path=str(book_root))
        sys.exit(1)
    create_m4b(book_root)
