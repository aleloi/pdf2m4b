import pytest
from pathlib import Path
from pdf2m4b.md_to_folders import convert_md

@pytest.fixture
def sample_md(tmp_path):
    md_content = (
        "\n"
        "preamble content outside main section\n"
        "# Chapter One\n"
        "Content for chapter one.\n\n"
        "## Section 1.1\n"
        "Details of section 1.1.\n\n"
        "# Chapter Two\n"
        "Content for chapter two."
    )
    md_file = tmp_path / "sample.md"
    md_file.write_text(md_content, encoding="utf-8")
    return md_file

def test_convert_md_creates_folders(tmp_path, sample_md):
    output_folder = tmp_path / "output"
    convert_md(sample_md, output_folder)
    
    # Check that a preamble file is created if there is root-level content.
    preamble = output_folder / "00_preamble.md"
    assert preamble.exists()
    
    # Look for at least one chapter folder (folders are prefixed with "01_" etc.).
    chapter_folders = list(output_folder.glob("01_*"))
    assert len(chapter_folders) >= 1
