import pytest
from pathlib import Path
import io
import subprocess
from pdf2m4b.tts_polly import clean_text, chunk_text, process_md_file, AUDIO_EXTENSION

# A dummy Polly client for testing.
class DummyPollyClient:
    def synthesize_speech(self, Text, OutputFormat, VoiceId, Engine):
        # Return a dummy response containing an AudioStream as BytesIO.
        return {"AudioStream": io.BytesIO(b"dummy audio data")}

@pytest.fixture
def temp_md(tmp_path):
    md_file = tmp_path / "00.md"
    # Create a markdown file with enough text for one chunk.
    md_file.write_text("This is a test. " * 10, encoding="utf-8")
    return md_file

def test_clean_text():
    raw = "This _should_ be cleaned.\nNew line."
    cleaned = clean_text(raw)
    assert "_" not in cleaned
    assert "\n" not in cleaned

def test_chunk_text_short():
    text = "Short text."
    chunks = chunk_text(text, max_length=50)
    assert len(chunks) == 1

def test_chunk_text_long():
    text = "Sentence. " * 100  # creates a long string
    chunks = chunk_text(text, max_length=50)
    assert len(chunks) > 1

# Dummy subprocess.run to simulate a successful ffmpeg call.
def dummy_subprocess_run(cmd, stdout, stderr, text):
    class DummyResult:
        returncode = 0
        stderr = ""
    return DummyResult()

def test_process_md_file_single_chunk(monkeypatch, temp_md):
    dummy_polly = DummyPollyClient()
    folder = temp_md.parent
    output_audio = folder / f"00.{AUDIO_EXTENSION}"
    
    # Remove output file if it already exists.
    if output_audio.exists():
        output_audio.unlink()
    
    # Patch subprocess.run used when combining chunks.
    monkeypatch.setattr(subprocess, "run", dummy_subprocess_run)
    
    process_md_file(temp_md, dummy_polly)
    
    # Verify that the final output audio file was created.
    assert output_audio.exists()
    
    # Clean up after test.
    output_audio.unlink()
