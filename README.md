# pdf2m4b

**pdf2m4b** is a command-line tool that converts PDF documents into M4B audiobooks using an end-to-end pipeline. It extracts text from PDFs, organizes the content into a structured hierarchy (chapters/sections), synthesizes speech via AWS Polly, and finally combines the audio segments into an M4B audiobook file using FFmpeg.

## Features

- **PDF to Markdown:** Extract text from PDFs using [pymupdf4llm](https://pypi.org/project/pymupdf4llm/).
- **Structured Chapters:** Parse Markdown into a hierarchical folder structure.
- **Text-to-Speech:** Generate audio for each chapter with AWS Polly.
- **Audiobook Creation:** Combine audio segments into a single M4B audiobook with chapter metadata.
- **Flexible Logging:** Uses `structlog` for logging with options for colorized terminal output or JSON logging.
- **Easy Installation:** Available on PyPI and installable via pip.

## TODO
- usage instructions
- tests
