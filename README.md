# pdf2m4b

**pdf2m4b** is a command-line tool that converts PDF documents into M4B audiobooks using an end-to-end pipeline. It extracts text from PDFs, organizes the content into a structured hierarchy (chapters/sections), synthesizes speech via AWS Polly, and finally combines the audio segments into an M4B audiobook file using FFmpeg.

## Features

- **PDF to Markdown:** Extract text from PDFs using [pymupdf4llm](https://pypi.org/project/pymupdf4llm/).
- **Structured Chapters:** Parse Markdown into a hierarchical folder structure.
- **Text-to-Speech:** Generate audio for each chapter with AWS Polly.
- **Audiobook Creation:** Combine audio segments into a single M4B audiobook with chapter metadata.
- **Flexible Logging:** Uses `structlog` for logging with options for colorized terminal output or JSON logging.
- **Easy Installation:** Available on PyPI and installable via pip.

## Usage example
```bash
$ export AWS_ACCESS_KEY_ID=AKI...
$ export AWS_SECRET_ACCESS_KEY=7SR...
$ pip install pdf2m4b
$ python -m pdf2m4b.main --pdf ../extracted.pdf
2025-02-03 07:27:58 [info     ] Converting PDF to Markdown     func_name=main markdown=output/output.md module=main pdf=../extracted.pdf
Processing ../extracted.pdf...
[========================================]
2025-02-03 07:28:01 [info     ] Markdown written               func_name=pdf_to_md module=pdf_to_md output=output/output.md
2025-02-03 07:28:01 [info     ] Converting Markdown to folder structure chapters=output/chapters func_name=main markdown=output/output.md module=main
2025-02-03 07:28:01 [info     ] Folder structure created       func_name=convert_md module=md_to_folders output=output/chapters
2025-02-03 07:28:01 [info     ] Starting TTS synthesis         chapters=output/chapters func_name=main module=main
2025-02-03 07:28:01 [info     ] Chunked text                   chunks=3 func_name=process_md_file md_file=output/chapters/02_32_multiplexing_and_demultiplexing/00.md module=tts_polly
2025-02-03 07:28:01 [info     ] Processing TTS chunk           chunk=1 func_name=process_md_file md_file=output/chapters/02_32_multiplexing_and_demultiplexing/00.md module=tts_polly total_chunks=3 words=407
2025-02-03 07:28:06 [debug    ] Chunk synthesized              chunk=1 func_name=process_md_file md_file=output/chapters/02_32_multiplexing_and_demultiplexing/00.md module=tts_polly time=5.158603383999434

...

...

2025-02-03 07:28:58 [info     ] Added chapter                  end_ms=1567077 func_name=create_m4b module=make_m4b start_ms=1327299 title='32 Multiplexing And Demultiplexing: Connectionless Multiplexing And Demultiplexing'
2025-02-03 07:28:58 [info     ] Added chapter                  end_ms=1767232 func_name=create_m4b module=make_m4b start_ms=1567077 title='32 Multiplexing And Demultiplexing: Connection-Oriented Multiplexing And Demultiplexing'
2025-02-03 07:28:58 [info     ] Wrote chapter metadata         filename=chapters.txt func_name=create_m4b module=make_m4b
2025-02-03 07:28:58 [info     ] Running FFmpeg                 command='ffmpeg -y -loglevel quiet -f concat -safe 0 -i concat_list.txt -i chapters.txt -map_metadata 1 -c:a aac output.m4b' func_name=create_m4b module=make_m4b
2025-02-03 07:29:26 [info     ] Successfully created M4B file  func_name=create_m4b module=make_m4b output_file=output.m4b
2025-02-03 07:29:26 [info     ] Audiobook creation complete    func_name=main module=main
```

## AWS costs
As of February 2025, it's around $10 per book. The project uses the **generative** setting of AWS Polly; check an example below.

## Example output
See [here](example_outputs/bishop_mcmc.m4b) for an output audiobook, which is the end result.
[Here](example_outputs/bishop_mcmc.pdf) is the corresponding input. [Here](example_outputs/bishop_mcmc/output.md) is an intermediate markdown file, and [here](example_outputs/bishop_mcmc/) is a file tree with hierarchical text and audio snippets. It's a rather *adversarial* example; there are lots of formulas that the initial parsing does wrong, and which are then filtered out before passing the text to the model. It performs quite well if you use it on an ordinary plain text without formulas.


## TODOs
- example PDFs
- run more pdfs through it
