"""Application-wide constants for FileFinder v3 (spec section 4.1)."""

APP_NAME = "FileFinder"
APP_ID = "filefinder"

# Chunking (approximate token counts, measured in characters).
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Search.
TOP_K = 10
RAG_TOP_K = 5
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

# Image (CLIP) search relevance gate: CLIP cosine scores cluster in a
# narrow band, so weak/irrelevant matches are filtered out below this
# absolute floor and below IMAGE_RELATIVE_RATIO * best-score.
IMAGE_MIN_SCORE = 0.24
IMAGE_RELATIVE_RATIO = 0.9

# CLIP scores max out around ~0.35 while BGE text scores reach ~0.9, so
# images would never rank in merged results. Boost filtered image scores
# onto a comparable scale (relevance gate is applied BEFORE boosting).
IMAGE_SCORE_BOOST = 2.5

# Directories never descended into during scans (app caches, VCS, system,
# virtual environments). Compared case-insensitively by directory name.
EXCLUDED_DIR_NAMES = {
    ".git", ".svn", ".hg", ".idea", ".vscode",
    "node_modules", "__pycache__", ".venv", "venv", "dist", "build",
    "AppData", "Application Data", "Local Settings", "LocalLow",
    "$RECYCLE.BIN", "System Volume Information", "Recovery",
    "Windows", "Program Files", "Program Files (x86)", "ProgramData",
    "Tencent Files", "WeChat Files", "xwechat_files",
}

# Texts shorter than this (filename fallbacks, OCR noise from icons) get
# no semantic vectors — they stay keyword-searchable only.
MIN_SEMANTIC_TEXT_LENGTH = 16

# Supported file extensions for indexing.
SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".docx",
    ".pdf",
    ".xlsx",
    ".pptx",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}

# Embedding dimensions per platform model.
EMBEDDING_DIM_DESKTOP = 1024  # bge-m3
EMBEDDING_DIM_ANDROID = 512  # bge-small-zh

# Image-content (CLIP) search.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGE_EMBEDDING_DIM = 512
# Multilingual text tower (uz/zh/en) and quantized vision tower.
CLIP_TEXT_REPO = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
CLIP_VISION_REPO = "Xenova/clip-vit-base-patch32"

# GitHub repository used by the auto-updater.
GITHUB_REPO = "BotirBakhtiyarov/Filefinder"

# File-watcher debounce interval in seconds.
DEBOUNCE_SECONDS = 2.0

# Free-tier limits (premium removes them; see core/license.py LIMITS).
# Indexing, search results and OCR are unlimited on the free tier; only
# chat queries, RAG, export and watched folders are gated.
FREE_WATCH_DIRS = 1
FREE_CHAT_QUERIES = 20
