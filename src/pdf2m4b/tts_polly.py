#!/usr/bin/env python3
"""
tts_polly.py

Recursively traverses a folder structure to find files named "00.md" and
synthesizes speech using AWS Polly. If the text is too long, it is split
into chunks and (if necessary) combined via ffmpeg.
"""

import re
import sys
import time
import subprocess
from pathlib import Path
from typing import List
import boto3
from .logger_config import logger

OUTPUT_FORMAT: str = "ogg_vorbis"  # Change to "mp3" if preferred.
AUDIO_EXTENSION: str = "ogg" if OUTPUT_FORMAT == "ogg_vorbis" else "mp3"
MAX_CHARS: int = 3000

def clean_text(text: str) -> str:
    text = re.sub(r'(^|[^0-9A-Za-z])_(.+?)_(?=$|[^0-9A-Za-z])', r'\1\2', text)
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str, max_length: int = MAX_CHARS) -> List[str]:
    if len(text) <= max_length:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_length:
            if current:
                chunks.append(current.strip())
                current = ""
            forced = [sentence[i:i+max_length] for i in range(0, len(sentence), max_length)]
            chunks.extend(forced)
            continue
        if current and (len(current) + 1 + len(sentence)) > max_length:
            chunks.append(current.strip())
            current = sentence
        else:
            current = sentence if not current else current + " " + sentence
    if current:
        chunks.append(current.strip())
    return chunks

def process_md_file(md_file: Path, polly_client: boto3.client) -> None:
    folder = md_file.parent
    output_audio = folder / f"00.{AUDIO_EXTENSION}"
    if output_audio.exists():
        logger.info("Audio already exists; skipping TTS",
                    md_file=str(md_file), output=str(output_audio))
        return

    try:
        raw_text = md_file.read_text(encoding="utf-8")
    except Exception as err:
        logger.error("Error reading file", md_file=str(md_file), exc_info=err)
        raise

    if not raw_text.strip():
        logger.info("Empty markdown file; skipping", md_file=str(md_file))
        return

    cleaned = clean_text(raw_text)
    chunks = chunk_text(cleaned)
    logger.info("Chunked text", md_file=str(md_file), chunks=len(chunks))

    total_words = 0
    total_tts_time = 0.0
    temp_files: List[Path] = []

    for idx, chunk in enumerate(chunks, start=1):
        word_count = len(chunk.split())
        total_words += word_count
        logger.info("Processing TTS chunk", md_file=str(md_file),
                    chunk=idx, total_chunks=len(chunks), words=word_count)
        start = time.monotonic()
        try:
            response = polly_client.synthesize_speech(
                Text=chunk,
                OutputFormat=OUTPUT_FORMAT,
                VoiceId="Amy",
                Engine="neural"
            )
        except Exception as err:
            logger.error("Error synthesizing speech", md_file=str(md_file),
                         chunk=idx, exc_info=err)
            raise

        if "AudioStream" not in response:
            logger.error("No AudioStream in Polly response", md_file=str(md_file), chunk=idx)
            raise Exception("No audio!")
            continue

        audio_data = response["AudioStream"].read()
        elapsed = time.monotonic() - start
        total_tts_time += elapsed
        logger.debug("Chunk synthesized", md_file=str(md_file), chunk=idx, time=elapsed)
        chunk_file = folder / f"00_chunk{idx:02d}.{AUDIO_EXTENSION}"
        try:
            chunk_file.write_bytes(audio_data)
        except Exception as err:
            logger.error("Error writing audio chunk", file=str(chunk_file), exc_info=err)
            raise
        temp_files.append(chunk_file)

    if not temp_files:
        logger.error("No audio chunks generated", md_file=str(md_file))
        return

    tts_speed = total_words / (total_tts_time / 60) if total_tts_time > 0 else 0.0
    estimated_minutes = total_words / 150.0
    logger.info("TTS stats", md_file=str(md_file), words=total_words,
                tts_time=total_tts_time, speed=tts_speed, estimated_duration=estimated_minutes)

    if len(temp_files) == 1:
        try:
            temp_files[0].rename(output_audio)
            logger.info("Single chunk renamed to final output", output=str(output_audio))
        except Exception as err:
            logger.error("Error renaming audio file", source=str(temp_files[0]),
                         target=str(output_audio), exc_info=err)
            raise
    else:
        filelist = folder / "chunks.txt"
        try:
            with filelist.open("w", encoding="utf-8") as f:
                for fpath in temp_files:
                    f.write(f"file '{fpath.name}'\n")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(filelist),
                "-c", "copy", str(output_audio)
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                logger.error("ffmpeg error", md_file=str(md_file), stderr=result.stderr)
            else:
                logger.info("Combined audio saved", output=str(output_audio))
        except Exception as err:
            logger.error("Error during audio combination", md_file=str(md_file), exc_info=err)
            raise 
        finally:
            if filelist.exists():
                filelist.unlink()
            for fpath in temp_files:
                if fpath.exists():
                    fpath.unlink()

def run_tts(base_folder: Path) -> None:
    if not base_folder.is_dir():
        logger.error("Base folder is not a directory", folder=str(base_folder))
        return
    polly = boto3.client("polly", region_name="us-east-1")
    md_files = list(base_folder.rglob("00.md"))
    if not md_files:
        logger.warning("No markdown files found for TTS", base=str(base_folder))
        return
    for md in md_files:
        process_md_file(md, polly)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python tts_polly.py <base_folder>")
        sys.exit(1)
    run_tts(Path(sys.argv[1]))
