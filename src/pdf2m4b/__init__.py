"""
pdf2m4b - A Python library for converting PDFs to M4B audiobook format.

Features:
- Extracts text from PDFs
- Converts text to speech using TTS engines
- Packages output as an M4B audiobook

Usage:

```bash
    $ python -m pdf2m4b.main --help
    usage: main.py [-h] --pdf PDF [--out_dir OUT_DIR] [--audiobook AUDIOBOOK]

    $ python -m pdf2m4b.main --pdf /path/to/pdf/file.pdf
    # ... diagnostic output ...
    # by default writes audio to `output.m4b`

    # Alternatively:
    $ python

    >>> import pdf2m4b
    >>> pdf2m4b.main()
```


For detailed documentation, visit: https://github.com/your-repo/pdf2m4b
"""

__version__ = "0.1.0"
