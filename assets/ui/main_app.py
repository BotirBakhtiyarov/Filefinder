import os
import json
import sys
import customtkinter as ctk
from tkinter import messagebox
from pynput import keyboard
from .search_frame import SearchFrame
from .chat_frame import ChatFrame
from .setup_wizard import SetupWizard

class MainApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        # Set base path and appdata path
        if getattr(sys, 'frozen', False):  # Running as .exe
            self.base_path = os.path.dirname(sys.executable)
        else:  # Running as .py
            self.base_path = os.path.dirname(os.path.abspath(__file__))

        # Load translations
        translation_path = os.path.join(self.base_path, 'translations.json')
        try:
            with open(translation_path, 'r', encoding='utf-8') as f:
                self.translations = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load translations: {str(e)}")
            self.translations = {}

        # Use APPDATA for writable storage
        appdata = os.getenv('APPDATA')
        self.appdata_path = os.path.join(appdata, "FileFinder")
        if not os.path.exists(self.appdata_path):
            os.makedirs(self.appdata_path)


        self.config_file = os.path.join(self.appdata_path, "config.json")
        self.index_file = os.path.join(self.appdata_path, "file_index.json")

        self.title("工一智能文件查找器和聊天助手")
        self.geometry("1200x800")
        icon_path = os.path.join(self.base_path,'img', 'logo.ico')
        self.iconbitmap(icon_path)
        self.protocol("WM_DELETE_WINDOW", self.minimize_to_tray)

        self.current_language = ctk.StringVar(value="EN")
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
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

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
        # Define default configuration with all expected keys
        default_config = {
            "api_url": "",
            "api_key": "",
            "chat_api_url": "",
            "chat_api_key": "",
            "document_dir": "",
            "image_dir": "",
            "language": "ZH",
            "chat_model": "Regular",
            "dark_mode": True
        }

        if not os.path.exists(self.config_file):
            # If config file doesn't exist, use defaults
            self.config = default_config

        else:
            # Load existing config file with error handling
            try:
                with open(self.config_file, "r", encoding='utf-8') as f:
                    loaded_config = json.load(f)
                # Merge default config with loaded config
                self.config = {**default_config, **loaded_config}

            except json.JSONDecodeError:
                # If file is corrupted, fall back to defaults

                self.config = default_config
            except Exception as e:

                self.config = default_config

        # Assign config values to instance variables
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
                json.dump(self.config, f, indent=4)

        except Exception as e:

            raise

    def initialize_ui(self):
        ctk.set_appearance_mode("dark" if self.dark_mode.get() else "light")
        ctk.set_default_color_theme("blue")

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="ns", padx=0, pady=0)
        self.sidebar.grid_remove()  # Initially hidden

        # Sidebar buttons
        ctk.CTkButton(self.sidebar, text=self.get_translation("🔍搜索"), command=lambda: self.switch_mode("search"),
                      fg_color="#1f6aa8", hover_color="#14487f").pack(pady=(50,10), padx=10, fill="x")
        ctk.CTkButton(self.sidebar, text=self.get_translation("💬聊天"), command=lambda: self.switch_mode("chat"),
                      fg_color="#1f6aa8", hover_color="#14487f").pack(pady=10, padx=10, fill="x")
        ctk.CTkButton(self.sidebar, text=self.get_translation("🛠设置"), command=self.show_settings,
                      fg_color="#1f6aa8", hover_color="#14487f").pack(pady=10, padx=10, fill="x")

        # Content frame
        self.content_frame = ctk.CTkFrame(self, corner_radius=10)
        self.content_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        # Burger menu button
        self.burger_btn = ctk.CTkButton(self, text="☰", width=40, command=self.toggle_sidebar, fg_color="#1f6aa8",
                                        hover_color="#14487f")
        self.burger_btn.grid(row=0, column=0, sticky="nw", padx=5, pady=5)

        # Initialize SearchFrame and ChatFrame with MainApp as master
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
            return
        self.settings_window = ctk.CTkToplevel(self)
        self.settings_window.title(self.get_translation("settings"))
        self.settings_window.geometry("400x700")
        self.settings_window.resizable(False, False)
        ico_path = os.path.join(self.base_path, 'img', 'logo.ico')
        self.settings_window.iconbitmap(ico_path)


        # Language selection
        ctk.CTkLabel(self.settings_window, text="语言:").pack(pady=5)
        self.lang_option = ctk.CTkOptionMenu(self.settings_window, values=["ZH"], #["EN", "ZH", "RU", "UZ"]
                                             variable=self.current_language)
        self.lang_option.pack(pady=5)

        # Chat AI model selection
        ctk.CTkLabel(self.settings_window, text="聊天AI模型:").pack(pady=5)
        self.chat_model_option = ctk.CTkOptionMenu(self.settings_window, values=self.chat_models,
                                                   variable=self.chat_model)
        self.chat_model_option.pack(pady=5)

        fields = [
            ("API URL (Embedding)", "api_url"), ("API Key (Embedding)", "api_key"),
            ("Chat API URL", "chat_api_url"), ("Chat API Key", "chat_api_key")
        ]
        self.entries = {}
        for label, key in fields:
            ctk.CTkLabel(self.settings_window, text=f"{label}:").pack(pady=5)
            entry = ctk.CTkEntry(self.settings_window, width=300)
            entry.insert(0, getattr(self, key))
            entry.pack(pady=5)
            self.entries[key] = entry

        ctk.CTkLabel(self.settings_window, text=self.get_translation("document_dir")).pack(pady=5)
        self.doc_btn = ctk.CTkButton(self.settings_window, text=self.document_dir or "Select",
                                     command=self.select_doc_dir)
        self.doc_btn.pack(pady=5)
        ctk.CTkLabel(self.settings_window, text=self.get_translation("image_dir")).pack(pady=5)
        self.img_btn = ctk.CTkButton(self.settings_window, text=self.image_dir or "Select", command=self.select_img_dir)
        self.img_btn.pack(pady=5)

        ctk.CTkSwitch(self.settings_window, text=self.get_translation("dark_mode"), variable=self.dark_mode,
                      command=self.update_appearance).pack(pady=10)
        ctk.CTkButton(self.settings_window, text=self.get_translation("save"), command=self.save_settings,
                      fg_color="#1f6aa8", hover_color="#14487f").pack(pady=10)

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
        self.update_texts()
        self.update_appearance()
        messagebox.showinfo("Success", "设置已成功保存!")
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
        lang = self.current_language.get().lower()
        return self.translations.get(lang, {}).get(key, key)
        return self.translations.get(lang, {}).get(key, key)