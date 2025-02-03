import pytest
import subprocess
from pathlib import Path
from pdf2m4b.make_m4b import get_duration_ms, sanitize_title, collect_segments

# A dummy CompletedProcess to simulate ffprobe.
class DummyCompletedProcess:
    def __init__(self, stdout):
        self.stdout = stdout
        self.returncode = 0

def dummy_run(cmd, stdout, stderr, text, check):
    # Simulate ffprobe output returning a duration of 1.23 seconds.
    return DummyCompletedProcess("1.23")

def test_get_duration_ms(monkeypatch, tmp_path):
    dummy_audio = tmp_path / "dummy.ogg"
    dummy_audio.write_bytes(b"dummy audio")
    
    monkeypatch.setattr(subprocess, "run", dummy_run)
    duration = get_duration_ms(str(dummy_audio))
    # 1.23 seconds * 1000 = 1230 ms
    assert duration == 1230

def test_sanitize_title():
    raw_title = "01_chapter_one"
    sanitized = sanitize_title(raw_title)
    # Expect underscores replaced and title-cased output.
    assert "Chapter One" in sanitized

def test_collect_segments(monkeypatch, tmp_path):
    # Create a fake folder structure with .ogg files.
    root = tmp_path / "book"
    root.mkdir()
    
    # Create a dummy ogg file in the root.
    dummy_file = root / "chapter.ogg"
    dummy_file.write_bytes(b"dummy audio")
    
    # Create a subfolder with another dummy ogg file.
    subfolder = root / "subchapter"
    subfolder.mkdir()
    dummy_file2 = subfolder / "subchapter.ogg"
    dummy_file2.write_bytes(b"dummy audio")
    
    segments = collect_segments(str(root))
    # We expect two segments found.
    assert len(segments) == 2
    for seg in segments:
        assert seg.file_path.endswith(".ogg")
