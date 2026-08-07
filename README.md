# FileFinder v3

Cross-platform (Windows / Linux / Android) **semantic file search + RAG chat** desktop
application built with [Flet](https://flet.dev) (Flutter for Python). Everything runs
in a single process — no local HTTP server.

## Features

- **Hybrid search** — semantic vector search (BGE embeddings, ONNX) combined with
  SQLite FTS5 keyword search (`0.7 * semantic + 0.3 * keyword`).
- **RAG chat** — chat with your files through a local Ollama model or the DeepSeek
  API; retrieved chunks are injected as context.
- **Broad file support** — txt, md, docx, xlsx, pptx, pdf (with OCR fallback), and
  images (jpg/jpeg/png/bmp) via Tesseract OCR.
- **Background indexing** — watchdog-based live re-indexing, cooperative cancel,
  per-file failure isolation.
- **Auto-update** — checks GitHub Releases, downloads and installs updates in-app.
- **i18n** — Chinese (default), Uzbek, English.
- **Cross-platform** — Windows and Linux in Phase 1; Android behind abstractions
  (sqlite-vec vector store) in Phase 2.

## Install

```bash
# with Poetry (recommended)
poetry install

# or plain pip (editable install)
pip install -e .
```

## Run

```bash
flet run
# or
python -m filefinder.main
```

## Build

```bash
flet build windows   # Windows installer/app
flet build linux     # Linux bundle
flet build apk       # Android (Phase 2)
```

## Tech stack

| Component        | Technology                                   |
|------------------|----------------------------------------------|
| UI               | Flet (Flutter for Python)                    |
| Embeddings       | BGE-M3 / bge-small-zh via ONNX Runtime       |
| Vector store     | ChromaDB (desktop), sqlite-vec (Android)     |
| Keyword search   | SQLite FTS5 (bm25)                           |
| OCR              | Tesseract (pytesseract) + Pillow/pdf2image   |
| File parsing     | pdfplumber, python-docx, openpyxl, python-pptx |
| File watching    | watchdog                                     |
| LLM providers    | Ollama (local), DeepSeek (API)               |
| Updates          | GitHub Releases API                          |
| Config security  | cryptography (Fernet, machine-bound key)     |

## Author

Botir Bakhtiyarov — https://github.com/BotirBakhtiyarov/Filefinder
