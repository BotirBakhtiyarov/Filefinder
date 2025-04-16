import os
import json
import sys
import customtkinter as ctk
from tkinter import messagebox
from pynput import keyboard
from .search_frame import SearchFrame
from .chat_frame import ChatFrame
from .setup_wizard import SetupWizard
from .settings import SettingsWindow
from pathlib import Path

local_path = str(Path(__file__).parent.resolve())

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Set base path for resources
        if getattr(sys, 'frozen', False):
            self.base_path = sys._MEIPASS  # Use temporary directory for bundled files
        else:
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        # Load translations
        translation_path = os.path.join(self.base_path, 'translations.json')
        default_translations = {
            "zh": {
                "search_placeholder": "搜索文档和图像...",
                "search_button": "搜索",
                "status_ready": "准备就绪",
                "status_indexing": "正在索引文件...",
                "send_button": "发送",
                "new_chat_button": "新聊天",
                "document_dir": "文档目录:",
                "image_dir": "图像目录：",
                "search": "搜索",
                "chat": "聊天",
                "open": "打开",
                "summary": "摘要",
                "index_folder": "索引文件夹",
                "dark_mode": "暗模式",
                "save": "保存",
                "settings": "设置",
                "upload_files": "上传文件"
            }
        }
        try:
            with open(translation_path, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            messagebox.showwarning("警告", f"无法加载翻译文件，使用默认中文翻译: {str(e)}")
            self.translations = default_translations

        # Set up appdata directory
        appdata = os.getenv('APPDATA')
        self.appdata_path = os.path.join(appdata, "FileFinder")
        if not os.path.exists(self.appdata_path):
            os.makedirs(self.appdata_path)

        self.config_file = os.path.join(self.appdata_path, "config.json")
        self.index_file = os.path.join(self.appdata_path, "file_index.json")

        # Window setup
        self.title("工一文件查找器和聊天助手")
        self.geometry("1200x800")
        icon_path = os.path.join(local_path, 'img', 'logo.ico')
        try:
            self.iconbitmap(icon_path)
        except Exception as e:
            messagebox.showwarning("警告", f"无法加载图标: {str(e)}")
        self.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)

        # Initialize variables
        self.current_language = ctk.StringVar(value="ZH")
        self.dark_mode = ctk.BooleanVar(value=True)
        self.chat_models = ["Regular", "OpenAI", "DeepSeek", "Ollama"]
        self.chat_model = ctk.StringVar(value="Regular")
        self.api_url = ""
        self.api_key = ""
        self.chat_api_url = ""
        self.chat_api_key = ""
        self.document_dir = ""
        self.image_dir = ""

        if not os.path.exists(self.config_file):
            self.show_setup_wizard()
        else:
            self.load_config()
            self.initialize_ui()

        self.hotkey_listener = keyboard.GlobalHotKeys({'<alt>+q': self.show_window})
        self.hotkey_listener.start()

        self.is_quitting = False

    def show_setup_wizard(self):
        self.wizard = SetupWizard(self, self.on_wizard_complete)
        self.wizard.grab_set()

    def on_wizard_complete(self, config):
        self.config = config
        try:
            self.save_config()
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败: {str(e)}")

        self.api_url = config["api_url"]
        self.api_key = config["api_key"]
        self.chat_api_url = config["chat_api_url"]
        self.chat_api_key = config["chat_api_key"]
        self.document_dir = config["document_dir"]
        self.image_dir = config["image_dir"]
        self.current_language.set(config["language"])
        self.chat_model.set(config["chat_model"])
        self.dark_mode.set(config["dark_mode"])
        self.wizard.destroy()
        self.initialize_ui()

    def load_config(self):
        default_config = {
            "api_url": "http://localhost:5000",
            "api_key": "",
            "chat_api_url": "http://10.20.1.213/v1/chat-messages",
            "chat_api_key": "",
            "document_dir": "",
            "image_dir": "",
            "language": "ZH",
            "chat_model": "Regular",
            "dark_mode": True
        }

        if not os.path.exists(self.config_file):
            self.config = default_config
        else:
            try:
                with open(self.config_file, "r", encoding='utf-8') as f:
                    loaded_config = json.load(f)
                self.config = {**default_config, **loaded_config}
            except json.JSONDecodeError:
                self.config = default_config
            except Exception as e:
                self.config = default_config

        self.api_url = self.config["api_url"]
        self.api_key = self.config["api_key"]
        self.chat_api_url = self.config["chat_api_url"]
        self.chat_api_key = self.config["chat_api_key"]
        self.document_dir = self.config["document_dir"]
        self.image_dir = self.config["image_dir"]
        self.current_language.set(self.config["language"])
        self.chat_model.set(self.config.get("chat_model", "Regular"))
        self.dark_mode.set(self.config["dark_mode"])

    def save_config(self):
        self.config = {
            "api_url": self.api_url,
            "api_key": self.api_key,
            "chat_api_url": self.chat_api_url,
            "chat_api_key": self.chat_api_key,
            "document_dir": self.document_dir,
            "image_dir": self.image_dir,
            "language": self.current_language.get(),
            "chat_model": self.chat_model.get(),
            "dark_mode": self.dark_mode.get()
        }
        try:
            with open(self.config_file, "w", encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            raise Exception(f"保存配置失败: {str(e)}")

    def initialize_ui(self):
        ctk.set_appearance_mode("dark" if self.dark_mode.get() else "light")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=0, pady=0)
        self.sidebar.grid_remove()

        ctk.CTkButton(
            self.sidebar,
            text=self.get_translation("🔍搜索"),
            command=lambda: self.switch_mode("search"),
            fg_color="#1f6aa8",
            hover_color="#14487f"
        ).pack(pady=(50,10), padx=10, fill="x")
        ctk.CTkButton(
            self.sidebar,
            text=self.get_translation("💬聊天"),
            command=lambda: self.switch_mode("chat"),
            fg_color="#1f6aa8",
            hover_color="#14487f"
        ).pack(pady=10, padx=10, fill="x")
        ctk.CTkButton(
            self.sidebar,
            text=self.get_translation("🛠设置"),
            command=self.show_settings,
            fg_color="#1f6aa8",
            hover_color="#14487f"
        ).pack(pady=10, padx=10, fill="x")

        self.content_frame = ctk.CTkFrame(self, corner_radius=10)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.burger_btn = ctk.CTkButton(
            self,
            text="☰",
            width=40,
            command=self.toggle_sidebar,
            fg_color="#1f6aa8",
            hover_color="#14487f"
        )
        self.burger_btn.grid(row=0, column=0, sticky="nw", padx=5, pady=5)

        self.search_frame = SearchFrame(self, self.index_file)
        self.search_frame.place(in_=self.content_frame, x=0, y=0, relwidth=1, relheight=1)
        self.chat_frame = ChatFrame(self)
        self.switch_mode("search")

    def toggle_sidebar(self):
        if self.sidebar.winfo_ismapped():
            self.sidebar.grid_remove()
        else:
            self.sidebar.grid()

    def switch_mode(self, mode_key):
        for frame in [self.search_frame, self.chat_frame]:
            frame.place_forget()
        if mode_key == "search":
            self.search_frame.place(in_=self.content_frame, x=0, y=0, relwidth=1, relheight=1)
        elif mode_key == "chat":
            self.chat_frame.place(in_=self.content_frame, x=0, y=0, relwidth=1, relheight=1)
        self.update_texts()

    def update_texts(self):
        self.search_frame.update_texts()
        self.chat_frame.update_texts()

    def show_settings(self):
        if hasattr(self, "settings_window") and self.settings_window.winfo_exists():
            self.settings_window.lift()
        else:
            self.settings_window = SettingsWindow(self)

    def minimize_to_tray(self):
        self.withdraw()

    def show_window(self):
        if self.state() == "withdrawn":
            self.deiconify()
            self.lift()
            self.focus_force()

    def quit_app(self):
        self.is_quitting = True
        self.hotkey_listener.stop()
        if hasattr(self, 'search_frame'):
            self.search_frame.stop_continuous_indexing()
        self.save_config()
        self.destroy()
        os._exit(0)

    def get_translation(self, key):
        lang = self.current_language.get().lower()
        return self.translations.get(lang, {}).get(key, key)