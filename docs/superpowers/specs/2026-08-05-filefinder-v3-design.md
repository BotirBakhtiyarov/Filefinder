# FileFinder v3 — Design Spec

**Date:** 2026-08-05
**Status:** Approved (user directive: build and finish)
**Rewrite of:** FileFinder v2 (CustomTkinter + Flask) → v3 (Flet, in-process core, cross-platform)

## 1. Goals & Scope

Build FileFinder v3: a cross-platform (Windows / Linux / Android) semantic file search
and RAG chat application in a **single process** — no Flask server, no Chinese-CLIP.

**Phase 1 (this spec):** fully working **Windows desktop** app — core engine, Flet UI,
auto-update, tests, CI workflow skeleton. Linux works via the same code path (CI matrix).
Android-specific modules (sqlite-vec impl, SAF, onnxruntime-mobile) are designed behind
abstractions but implemented in Phase 2.

**Hard constraints:**
- NO Flask / HTTP server for the core — everything in-process.
- NO Chinese-CLIP — OCR (Tesseract) + text embeddings (BGE, ONNX) only.
- Heavy work (indexing, embedding, search) in background threads; UI never blocks.
- All user-facing strings via i18n JSON (zh default, uz, en).

## 2. Architecture

```
Flet UI (ui/)  ──calls──▶  Core Engine (core/)  ──uses──▶  SQLite + Vector store + ONNX + Tesseract
      ▲                          │
      └──── progress callbacks ──┘   (threading + queue, page.pubsub-free: callbacks post via page.run_task)
Updater (updater/) ── GitHub Releases API ──▶ background download ──▶ zip extract + restart
```

- **UI ↔ Core:** UI calls async methods; core runs blocking work via `asyncio.to_thread`
  and reports progress through plain Python callbacks (the UI wraps them with
  `page.run_task` to marshal back to the UI thread).
- **Platform detection** in `core/config.py`: `is_android()`, `is_windows()`, `is_linux()`.
- **Dependency injection is avoided**; modules expose module-level singletons
  (`get_config()`, `get_db()`, `get_embedder()`, `get_search_engine()`).

## 3. Project Layout

```
filefinder_v3/
├── pyproject.toml               # Poetry
├── README.md
├── .gitignore
├── .github/workflows/release.yml
├── src/filefinder/
│   ├── __init__.py              # __version__
│   ├── main.py                  # flet entry: ft.app(main)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   ├── config.py
│   │   ├── db.py
│   │   ├── embeddings.py
│   │   ├── file_parser.py
│   │   ├── ocr.py
│   │   ├── indexer.py
│   │   ├── search.py
│   │   ├── chat.py
│   │   └── watcher.py
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── app.py
│   │   ├── views/{__init__,search_view,chat_view,index_view,settings_view}.py
│   │   └── components/{__init__,file_card,chat_bubble,search_bar,sidebar,index_progress}.py
│   └── updater/{__init__,github_checker,downloader,installer}.py
├── assets/
│   ├── icons/ (placeholder .gitkeep)
│   ├── fonts/ (placeholder .gitkeep)
│   └── i18n/{zh.json,uz.json,en.json}
├── models/.gitkeep
└── tests/{__init__,test_embeddings,test_search,test_parser,test_indexer}.py
```

## 4. Core Module Contracts

### 4.1 constants.py
- `APP_NAME = "FileFinder v3"`, `APP_ID = "filefinder"`
- `CHUNK_SIZE = 512` (tokens approx by chars*), `CHUNK_OVERLAP = 64`, `TOP_K = 10`, `RAG_TOP_K = 5`
- `SUPPORTED_EXTENSIONS = {".txt",".md",".docx",".pdf",".xlsx",".pptx",".jpg",".jpeg",".png",".bmp"}`
- `SEMANTIC_WEIGHT = 0.7`, `KEYWORD_WEIGHT = 0.3`
- `EMBEDDING_DIM_DESKTOP = 1024` (bge-m3), `EMBEDDING_DIM_ANDROID = 512` (bge-small-zh)
- `GITHUB_REPO = "BotirBakhtiyarov/Filefinder"`
- `DEBOUNCE_SECONDS = 2.0`

### 4.2 config.py
- `get_app_data_dir() -> Path`: `%APPDATA%/FileFinder` (win), `~/.config/filefinder` (linux),
  Flet client storage path (android).
- `Config` dataclass: `monitored_dirs: list[str]`, `theme: str("dark")`,
  `language: str("zh")`, `llm_provider: str("ollama"|"deepseek"|"off")`,
  `llm_api_url`, `llm_api_key` (encrypted-at-rest), `llm_model`,
  `embedding_model` (auto by platform), `auto_update_check: bool = True`.
- `load_config() / save_config(cfg)` → `config.json` in app data dir.
- `t(key: str, **kwargs) -> str`: i18n lookup, fallback current-lang → en → key itself.
  `load_i18n(lang)`, `set_language(lang)`.
- API key encryption: Fernet (cryptography) with key derived from machine id
  (`uuid.getnode()` + platform node), stored in `secret.key` file.
- `get_model_dir() -> Path` (app data `models/`), model name per platform:
  desktop `bge-m3-onnx`, android `bge-small-zh-v1.5-onnx`.
- Path safety: `is_safe_path(p)` rejects `..` traversal outside monitored roots.

### 4.3 db.py
- SQLite (`files.db`) via stdlib `sqlite3`, WAL mode, thread-local connections.
- Tables:
  - `files(id, path UNIQUE, filename, extension, size, modified_time, content_hash, text_preview, indexed_at)`
  - `files_fts` FTS5 virtual table on `(filename, content)` for keyword search.
  - `file_chunks(id, file_id, chunk_index, content, embedding_id)` — embedding_id = vector-store doc id.
  - `chat_history(id, conversation_id, role, content, timestamp)`
  - `kv(key, value)` — misc (schema_version, last_indexed, api_key).
- **VectorStore abstract base:** `add(ids, embeddings, metadatas)`, `delete(ids)`,
  `query(embedding, top_k) -> list[(id, score)]`, `count()`, `clear()`.
  - `ChromaStore` (desktop): chromadb PersistentClient in app data `chroma/`.
  - `SqliteVecStore` (android): sqlite-vec vec0 table; imported lazily, Phase 2.
  - `get_vector_store()` picks by platform.
- `Database` facade: `upsert_file(...)`, `remove_file(path)`, `get_file_by_path`,
  `keyword_search(query, limit)`, `add_chunks(file_id, chunks, embeddings)`,
  `get_stats() -> dict`, `clear_index()`, chat history CRUD.

### 4.4 embeddings.py
- `Embedder` singleton (`get_embedder()`), lazy ONNX session.
- Model: ONNX `model.onnx` + HF tokenizer files in `models/<model_name>/`.
  Desktop default BAAI/bge-m3 (quantized onnx, dim 1024); android bge-small-zh-v1.5 (dim 512).
- `embed_text(text: str) -> list[float]`, `embed_batch(texts: list[str]) -> list[list[float]]`
  — tokenize (max_length=512, padding, truncation), run session, mean-pool + L2 normalize.
- `is_model_available() -> bool`; `download_model(progress_cb)` — pulls from HuggingFace
  (`huggingface_hub.snapshot_download` filtered to onnx+tokenizer files) with progress.
- If model missing and download declined → `ModelNotAvailableError`; search degrades to
  keyword-only with a UI notice.

### 4.5 file_parser.py
- `extract_text(path: str) -> str` dispatch by extension:
  - `.txt/.md`: read with encoding detection (utf-8 → gbk fallback).
  - `.docx`: python-docx paragraphs+tables. `.xlsx`: openpyxl cells. `.pptx`: python-pptx shapes.
  - `.pdf`: pdfplumber text; if <20 chars/page → OCR fallback via ocr.extract_text_from_pdf.
  - images: `ocr.extract_text_from_image`.
- Raises `UnsupportedFileError`; returns "" on empty (indexer skips with log).

### 4.6 ocr.py
- `extract_text_from_image(path) -> str`, `extract_text_from_pdf(path) -> str`
  (pdf2image → per-page Tesseract).
- Preprocessing (Pillow): grayscale → autocontrast → optional threshold. OpenCV deskew
  only if cv2 importable (optional dep).
- Languages `chi_sim+eng`, `tesseract` binary from PATH or `TESSERACT_CMD` env /
  common Windows install path; tessdata dir checked for `chi_sim.traineddata`,
  `is_ocr_available() -> bool`.

### 4.7 indexer.py
- `scan_directory(path, extensions) -> list[str]` (os.walk, skip hidden/symlink-loops).
- `process_file(path) -> dict` — parser + metadata + md5 content hash.
- `index_file(path)`: skip unchanged (hash match), chunk text (CHUNK_SIZE chars w/ overlap),
  `embed_batch`, `db.upsert_file` + `add_chunks`.
- `index_directories(dirs, progress_cb(current,total,filename), cancel_event)` —
  runs in worker thread; returns stats dict. Cooperative cancel via `threading.Event`.
- `remove_file(path)`; `reindex_all(progress_cb)` = clear + full index.

### 4.8 search.py
- `SearchEngine.search(query, filters=None, top_k=TOP_K) -> list[SearchResult]`
- `SearchResult` dataclass: `path, filename, extension, snippet, score, modified_time, size`.
- Semantic: embed query → vector query → map chunk→file, max score per file, snippet =
  best chunk. Keyword: FTS5 → bm25 normalized. Hybrid: `0.7*sem + 0.3*kw`.
- Filters dataclass: `file_types: list[str]|None`, `date_from/date_to: float|None`,
  `directory: str|None`, `min_size/max_size: int|None` — applied post-hoc on candidates.
- Graceful degradation: no model → keyword only; no FTS hits → semantic only.

### 4.9 chat.py
- `ChatEngine.send_message(text, conversation_id, use_rag=True) -> str`
- RAG: search top RAG_TOP_K → context block (path + snippet) → prompt template (i18n-agnostic, English scaffold + user text).
- LLM router: `provider == "ollama"` → POST `{api_url}/api/generate` (default
  `http://localhost:11434`, model default `qwen2.5:3b`); `"deepseek"` → OpenAI-compatible
  POST `{api_url}/chat/completions` (default `https://api.deepseek.com`, model `deepseek-chat`);
  `"off"` or failure → localized "LLM unavailable" string.
- History persisted in `chat_history`; `new_conversation() -> str` (uuid),
  `get_history(conversation_id)`, last N=10 messages sent as context.

### 4.10 watcher.py
- watchdog `Observer` per monitored dir (desktop only; `is_available()` False on android).
- Events → debounce 2s (per-path timers) → `index_file` / `remove_file` in worker thread.
- `start(dirs, on_change_cb)`, `stop()`.

## 5. UI Design (Flet)

- `main.py`: `ft.app(target=main, assets_dir="assets")`.
- `app.py` `FileFinderApp`: page config (title, theme dark default, window 1100x720),
  left `Sidebar` (NavigationRail) + content area; on mobile-width (<700px) switch to
  bottom NavigationBar. Views: Search / Chat / Index / Settings. Snackbar helper
  `show_message(msg, error=False)`. Language switch rebuilds view labels from `t()`.
- First run: if model missing → dialog offering download (progress bar) or
  keyword-only mode; welcome overlay (5 short slides, skippable) once per install
  (`kv` flag).
- **SearchView:** SearchBar component (TextField, filter chips for file type, search
  button), results ListView of FileCard; empty state with "go to Index" button;
  search runs via `asyncio.to_thread`, spinner while running.
- **FileCard:** icon by extension, filename, truncated path, snippet, score badge,
  "Open" button (`os.startfile` on win / `xdg-open` linux).
- **ChatView:** ListView of ChatBubble (user right/accent, assistant left), input row
  (multiline TextField + send), Dropdown model selector (Ollama / DeepSeek / Off),
  RAG Switch, "New chat" button, typing indicator while waiting.
- **IndexView:** monitored folder list w/ remove buttons, FilePicker add-folder,
  stats row (total/indexed/last indexed/db size), LinearProgressIndicator + status +
  Cancel, buttons: Start Indexing / Reindex All / Clear Index (confirm dialog).
- **SettingsView:** language Dropdown, theme Switch, LLM fields (url, key password
  field, model name), model management (size on disk, download/delete), update section
  (current version, "Check for updates" → dialog w/ notes → download progress →
  install+restart), About (version, GitHub link, author).
- System tray (desktop): pystray in separate thread — Show/Quit; window close event
  minimizes to tray (Windows). Global hotkey Alt+Q via pynput listener toggling window
  visibility. Both optional-guarded (import errors → disabled, log only).

## 6. Updater

- `github_checker.check_update(current_version) -> dict|None` — GitHub latest release
  API, compare semver, pick asset matching `FileFinder-v{ver}-windows.zip` (platform
  suffix). Returns `{version, url, notes, published_at, sha256?}` (hash from
  accompanying `.sha256` asset if present).
- `downloader.download(url, dest, progress_cb)` — streamed chunks, resume via Range
  header if temp file exists, verify sha256 when provided.
- `installer.install(zip_path)` — backup current dir to `_backup/`, extract over app
  dir, relaunch `sys.executable` (frozen: the exe path) via subprocess and exit;
  restore backup on extraction failure. Windows-only in Phase 1.

## 7. CI/CD

`.github/workflows/release.yml`: trigger on tag `v*`; jobs build-windows
(`flet build windows`), build-linux (`flet build linux`), build-android
(`flet build apk`, allowed to fail in Phase 1), then `create-release` collecting
artifacts + generating `.sha256` files.

## 8. Testing

- `tests/test_parser.py` — extract_text on generated tmp docx/xlsx/txt/pdf.
- `tests/test_embeddings.py` — mean-pool/normalize math on fake session; tokenizer-free
  unit of `_normalize`; skip model-dependent tests if model absent.
- `tests/test_search.py` — hybrid score merge, filters, snippet pick (fake vector store).
- `tests/test_indexer.py` — integration: temp dir with txt/docx → index (fake embedder
  if no model) → search finds it → modify → reindex → remove.
- `python -m pytest tests/` must pass without model/network/OCR (auto-skip).

## 9. Error Handling & Logging

- `logging` to `<appdata>/filefinder.log` (RotatingFileHandler 2MB x3) + console.
- Core raises typed errors (`ModelNotAvailableError`, `UnsupportedFileError`,
  `LLMUnavailableError`); UI catches → snackbar with localized message.
- Per-file indexing failures are logged and skipped, counted in final stats.

## 10. Phase 2 (out of scope now, noted for future)

Android build: SAF file access, sqlite-vec store impl, onnxruntime-mobile,
tesseract4android, foreground-service indexing, OTA update path, WiFi-only model
download dialog.

## 11. Addendum (2026-08-06, implemented)

- **Auto-indexing**: default monitored dirs = user standard folders + all fixed drives (D:...);
  startup auto-index (size+mtime fast-path skip) + watchdog live sync.
- **Image content search**: CLIP pipeline (`core/image_embeddings.py`) — multilingual text tower
  (sentence-transformers/clip-ViT-B-32-multilingual-v1 ONNX + 2_Dense 768→512 projection via
  safetensors) + vision tower (Xenova/clip-vit-base-patch32 quantized), separate Chroma
  `images` collection; merged into search results. One-time CLIP reindex migration
  (kv `clip_reindex_v1`). Images without text are indexed by filename as fallback.
- **Self-installing deps**: `core/deps.py` auto-installs Tesseract (winget → NSIS fallback) and
  downloads eng+chi_sim traineddata into `~/.filefinder/tessdata` (MS Store Python virtualizes
  %APPDATA%, so tessdata must live outside it); first-run OCR install dialog in app.py.
- flet 0.86 API migration applied throughout (show_dialog/pop_dialog, ft.Alignment.CENTER,
  ft.BorderRadius/ft.Padding, Dropdown on_select, FilePicker service API, ft.run).
