# Smart File Finder & Chat Assistant v2 🇨🇳📂

[![GitHub stars](https://img.shields.io/github/stars/BotirBakhtiyarov/Filefinder?style=social)](https://github.com/BotirBakhtiyarov/Filefinder/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/BotirBakhtiyarov/Filefinder?style=social)](https://github.com/BotirBakhtiyarov/Filefinder/network/members)
[![GitHub license](https://img.shields.io/github/license/BotirBakhtiyarov/Filefinder)](https://github.com/BotirBakhtiyarov/Filefinder/blob/main/LICENSE)
[![Python version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Smart File Finder & Chat Assistant v2** is a powerful Python-based desktop application designed to streamline file management and information retrieval for Chinese-speaking users. With a sleek, Chinese-localized interface, this tool leverages advanced AI to search, summarize, and interact with documents and images.

Built with `customtkinter` for a modern GUI and `Flask` for a robust API backend, it’s perfect for professionals, researchers, and teams handling large file collections in Chinese environments. 🚀

---

## ✨ Features

- **Semantic File Search 🔍**: Find documents (DOCX, PDF, TXT) and images (JPG, PNG) using natural language queries in Chinese.
- **AI-Driven Insights 🤖**: Generate embeddings for text and images with models optimized for Chinese content, powered by SentenceTransformers and ChineseCLIP.
- **OCR Excellence 📸**: Extract and summarize text from images and PDFs using Tesseract OCR, tailored for Chinese characters.
- **Smart Chat Assistant 💬**: Engage in contextual conversations or summarize files with a Chinese-focused AI, supporting both Normal and RAG modes.
- **Chinese Interface 🇨🇳**: Fully localized UI with intuitive navigation, designed for native Chinese speakers.
- **Dark/Light Mode 🌗**: Toggle themes for comfortable use in any lighting.
- **System Tray Integration 🖥️**: Minimize to tray with `Alt+Q` hotkey to restore.
- **Continuous Indexing 🔄**: Automatically updates file indexes for monitored directories.

---

## 📸 Screenshots

*To be updated with v2’s Chinese interface.*

- **Search Tab**: Browse files with Chinese queries.
- **Chat Assistant**: Chat and summarize in Chinese.
- **OCR Summary**: View extracted Chinese text summaries.

---

## 🚀 Installation

### Prerequisites

- **Python 🐍**: 3.10 or higher
- **Tesseract OCR 📖**: For Chinese text extraction  
  - **Windows**: Download, add to PATH, ensure `chi_sim` data is installed  
  - **Linux**: `sudo apt install tesseract-ocr tesseract-ocr-chi-sim`  
  - **macOS**: `brew install tesseract`
- **Poppler 📄**: For PDF processing  
  - **Windows**: Download, add to PATH  
  - **Linux**: `sudo apt install poppler-utils`  
  - **macOS**: `brew install poppler`

### Setup Steps

```bash
# Clone the Repository 📥
git clone https://github.com/BotirBakhtiyarov/Filefinder.git
cd Filefinder

# Set Up Virtual Environment 🛠️
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install Dependencies 📦
pip install -r requirements.txt
```

### Configure AI Models ⚙️

Update `MODEL_DIR` in `api_server.py`:

```python
MODEL_DIR = "models/"  # Path to Chinese-optimized models
```

Make sure models like:

- `paraphrase-multilingual-MiniLM-L12-v2`
- `chinese-clip-vit-base-patch16`

are available or downloaded.

### Launch the App

```bash
# Start the API server 🌐
python api_server.py

# Run the main application 🎉
python app.py
```

---

## 🖱️ Usage

### First Launch 🚀

A Chinese-language setup wizard will guide you to:

- Enter API keys/URLs
- Set document/image directories
- Choose theme (Dark/Light)

### Search Files 🔎

- Go to the **“搜索”** tab.
- Enter Chinese queries (e.g., `年度报告概述`).
- View results and:
  - **打开**: Open file
  - **摘要**: AI-generated summary
  - **OCR 摘要**: OCR-based summary for images

### Chat Assistant 💬

- Go to the **“聊天”** tab
- Select:
  - **Normal Mode**: For general chat
  - **RAG Mode**: Upload and ask about specific files
- Ask questions in Chinese and get summarized answers

### System Tray 🖥️

- Minimize to tray by closing window
- Press **Alt+Q** to restore

---

## 📁 Project Structure

```
FileFinder_v2/
├── .venv/              # Virtual environment
├── assets/
│   ├── ui/             # Chinese-localized GUI
│   │   ├── chat_frame.py
│   │   ├── main_app.py
│   │   ├── search_frame.py
│   │   ├── setup_wizard.py
│   │   └── summary_window.py
│   └── utils/
│       └── file_utils.py
├── app.py              # App entry point
├── api_server.py       # Flask backend
├── requirements.txt    # Dependencies
└── myicon.ico          # App icon
```

---

## 🌐 API Endpoints

`api_server.py` includes:

- `/embed_text`: Embed Chinese text
- `/embed_image`: Embed image
- `/embed_clip_text`: CLIP-based text-image embedding
- `/extract_pdf_with_ocr`: OCR for Chinese PDFs
- `/extract_image_ocr`: OCR for Chinese images

---

## 🤝 Contributing

We welcome contributions! 🌟

```bash
# Fork and branch
git checkout -b feature/YourFeature

# Commit and push
git commit -m "Add YourFeature"
git push origin feature/YourFeature
```

- Follow PEP 8
- Maintain Chinese localization

---

## ⚠️ Troubleshooting

- **File Access Errors**: Verify file paths and network drives.
- **OCR Issues**: Make sure Tesseract is installed with `chi_sim` support.
- **API Failures**: Confirm `api_server.py` is running and keys are set.
- **UI Bugs**: Check `customtkinter` compatibility.

Open an [issue on GitHub](https://github.com/BotirBakhtiyarov/Filefinder/issues) with logs if needed.

---

## 📜 License

MIT License – see [LICENSE](https://github.com/BotirBakhtiyarov/Filefinder/blob/main/LICENSE) for details.

---

## 🙌 Acknowledgments

- **CustomTkinter**: Modern GUI framework
- **SentenceTransformers**: Chinese text embeddings
- **Transformers**: ChineseCLIP integration
- **Tesseract OCR**: Robust OCR engine

---

**Built with ❤️ for Chinese-speaking users, Smart File Finder & Chat Assistant v2 is your ultimate file management companion!**

