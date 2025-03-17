import os
import json
import customtkinter as ctk
from tkinter import messagebox
from pynput import keyboard
from .search_frame import SearchFrame
from .chat_frame import ChatFrame
from .setup_wizard import SetupWizard

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Smart File Finder & Chat Assistant")
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)

        self.current_language = ctk.StringVar(value="en")
        self.dark_mode = ctk.BooleanVar(value=True)
        self.api_url = ""
        self.api_key = ""
        self.chat_api_url = ""
        self.chat_api_key = ""
        self.document_dir = ""
        self.image_dir = ""

        self.config_file = "config.json"
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
        self.save_config()
        self.api_url = config["api_url"]
        self.api_key = config["api_key"]
        self.chat_api_url = config["chat_api_url"]
        self.chat_api_key = config["chat_api_key"]
        self.document_dir = config["document_dir"]
        self.image_dir = config["image_dir"]
        self.current_language.set(config["language"])
        self.dark_mode.set(config["dark_mode"])
        self.wizard.destroy()
        self.initialize_ui()

    def load_config(self):
        with open(self.config_file, "r") as f:
            self.config = json.load(f)
        self.api_url = self.config["api_url"]
        self.api_key = self.config["api_key"]
        self.chat_api_url = self.config["chat_api_url"]
        self.chat_api_key = self.config["chat_api_key"]
        self.document_dir = self.config["document_dir"]
        self.image_dir = self.config["image_dir"]
        self.current_language.set(self.config["language"])
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
            "dark_mode": self.dark_mode.get()
        }
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

    def initialize_ui(self):
        ctk.set_appearance_mode("dark" if self.dark_mode.get() else "light")
        ctk.set_default_color_theme("blue")

        self.menu_frame = ctk.CTkFrame(self, height=50)
        self.menu_frame.pack(fill="x", pady=5)

        self.lang_segmented = ctk.CTkSegmentedButton(
            self.menu_frame, values=["EN", "ZH"], variable=self.current_language, command=self.update_language
        )
        self.lang_segmented.pack(side="left", padx=10)

        self.mode_var = ctk.StringVar(value="Search")
        self.mode_segmented = ctk.CTkSegmentedButton(
            self.menu_frame, values=["Search", "Chat"], variable=self.mode_var, command=self.switch_mode
        )
        self.mode_segmented.pack(side="left", padx=10)

        self.settings_btn = ctk.CTkButton(self.menu_frame, text="⚙", width=40, command=self.show_settings)
        self.settings_btn.pack(side="right", padx=10)

        self.search_frame = SearchFrame(self)
        self.chat_frame = ChatFrame(self)
        self.switch_mode()

    def switch_mode(self, *args):
        mode = self.mode_var.get()
        for frame in [self.search_frame, self.chat_frame]:
            frame.pack_forget()
        if mode == "Search":
            self.search_frame.pack(fill="both", expand=True)
        else:
            self.chat_frame.pack(fill="both", expand=True)

    def update_language(self, *args):
        self.search_frame.update_texts()
        self.chat_frame.update_texts()

    def show_settings(self):
        if hasattr(self, "settings_window") and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        self.settings_window = ctk.CTkToplevel(self)
        self.settings_window.title("Settings")
        self.settings_window.geometry("400x500")

        fields = [
            ("API URL", "api_url"), ("API Key", "api_key"),
            ("Chat API URL", "chat_api_url"), ("Chat API Key", "chat_api_key")
        ]
        self.entries = {}
        for label, key in fields:
            ctk.CTkLabel(self.settings_window, text=f"{label}:").pack(pady=5)
            entry = ctk.CTkEntry(self.settings_window, width=300)
            entry.insert(0, getattr(self, key))
            entry.pack(pady=5)
            self.entries[key] = entry

        ctk.CTkLabel(self.settings_window, text="Document Directory:").pack(pady=5)
        self.doc_btn = ctk.CTkButton(self.settings_window, text=self.document_dir, command=self.select_doc_dir)
        self.doc_btn.pack(pady=5)
        ctk.CTkLabel(self.settings_window, text="Image Directory:").pack(pady=5)
        self.img_btn = ctk.CTkButton(self.settings_window, text=self.image_dir, command=self.select_img_dir)
        self.img_btn.pack(pady=5)

        ctk.CTkSwitch(self.settings_window, text="Dark Mode", variable=self.dark_mode, command=self.update_appearance).pack(pady=10)
        ctk.CTkButton(self.settings_window, text="Save", command=self.save_settings).pack(pady=10)

    def select_doc_dir(self):
        dir = ctk.filedialog.askdirectory(initialdir=self.document_dir)
        if dir:
            self.document_dir = dir
            self.doc_btn.configure(text=dir)

    def select_img_dir(self):
        dir = ctk.filedialog.askdirectory(initialdir=self.image_dir)
        if dir:
            self.image_dir = dir
            self.img_btn.configure(text=dir)

    def save_settings(self):
        for key, entry in self.entries.items():
            setattr(self, key, entry.get().strip())
        self.save_config()
        messagebox.showinfo("Success", "Settings saved successfully!")
        self.settings_window.destroy()

    def update_appearance(self):
        ctk.set_appearance_mode("dark" if self.dark_mode.get() else "light")

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
        self.search_frame.stop_continuous_indexing()
        self.save_config()
        self.destroy()
        os._exit(0)

    def get_translation(self, key):
        translations = {
            "en": {
                "search_placeholder": "Search documents and images...",
                "search_button": "Search",
                "status_ready": "Ready",
                "status_indexing": "Indexing files...",
                "send_button": "Send",
                "new_chat_button": "New Chat"
            },
            "zh": {
                "search_placeholder": "搜索文档和图像...",
                "search_button": "搜索",
                "status_ready": "准备就绪",
                "status_indexing": "正在索引文件...",
                "send_button": "发送",
                "new_chat_button": "新聊天"
            }
        }
        return translations[self.current_language.get().lower()].get(key, key)