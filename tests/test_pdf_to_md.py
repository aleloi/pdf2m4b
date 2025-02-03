import pytest
from pathlib import Path
from pdf2m4b.pdf_to_md import pdf_to_md
import pymupdf4llm

# Dummy replacement for the actual PDF-to-Markdown conversion.
def dummy_to_markdown(pdf_path: str) -> str:
    return "## Dummy Markdown\nContent generated for testing."

@pytest.fixture(autouse=True)
def patch_pymupdf(monkeypatch):
    monkeypatch.setattr(pymupdf4llm, "to_markdown", dummy_to_markdown)

def test_pdf_to_md_creates_file(tmp_path):
    # Create a dummy PDF file.
    pdf_path = tmp_path / "dummy.pdf"
    pdf_path.write_bytes(b"dummy pdf content")
    
    output_md = tmp_path / "output.md"
    pdf_to_md(pdf_path, output_md)
    
    content = output_md.read_text(encoding="utf-8")
    assert "Dummy Markdown" in content
