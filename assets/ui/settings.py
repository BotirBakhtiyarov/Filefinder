import os
import customtkinter as ctk
from tkinter import messagebox
import json

class SettingsWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.title(self.master.get_translation("settings"))
        self.geometry("400x700")
        self.resizable(False, True)
        ico_path = os.path.join(self.master.base_path, 'img', 'logo.ico')
        self.iconbitmap(ico_path)

        self.entries = {}
        self.create_widgets()

    def create_widgets(self):
        ctk.CTkLabel(self, text="聊天AI模型:").pack(pady=5)
        self.chat_model_option = ctk.CTkOptionMenu(
            self,
            values=self.master.chat_models,
            variable=self.master.chat_model,
            command=self.on_ai_model_change
        )
        self.chat_model_option.pack(pady=5)

        fields = [
            ("嵌入API URL", "api_url"),
            ("嵌入API Key", "api_key"),
            ("聊天API URL", "chat_api_url"),
            ("聊天API Key", "chat_api_key")
        ]
        for label, key in fields:
            ctk.CTkLabel(self, text=f"{label}:").pack(pady=5)
            entry = ctk.CTkEntry(self, width=300)
            entry.insert(0, getattr(self.master, key))
            entry.pack(pady=5)
            self.entries[key] = entry

        ctk.CTkLabel(self, text=self.master.get_translation("document_dir")).pack(pady=5)
        self.doc_btn = ctk.CTkButton(
            self,
            text=self.master.document_dir or "选择",
            command=self.select_doc_dir
        )
        self.doc_btn.pack(pady=5)

        ctk.CTkLabel(self, text=self.master.get_translation("image_dir")).pack(pady=5)
        self.img_btn = ctk.CTkButton(
            self,
            text=self.master.image_dir or "选择",
            command=self.select_img_dir
        )
        self.img_btn.pack(pady=5)

        ctk.CTkSwitch(
            self,
            text=self.master.get_translation("dark_mode"),
            variable=self.master.dark_mode,
            command=self.update_appearance
        ).pack(pady=10)

        ctk.CTkButton(
            self,
            text=self.master.get_translation("save"),
            command=self.save_settings,
            fg_color="#1f6aa8",
            hover_color="#14487f"
        ).pack(pady=10)

    def on_ai_model_change(self, selected_model):
        chat_api_urls = {
            "Regular": "http://10.20.1.213/v1/chat-messages",
            "OpenAI": "https://api.openai.com/v1/chat/completions",
            "DeepSeek": "https://api.deepseek.com/chat/completions",
            "Ollama": "http://localhost:11434/api/generate"
        }
        new_url = chat_api_urls.get(selected_model, "http://10.20.1.213/v1/chat-messages")
        self.entries["chat_api_url"].delete(0, "end")
        self.entries["chat_api_url"].insert(0, new_url)

    def select_doc_dir(self):
        dir_path = ctk.filedialog.askdirectory(initialdir=self.master.document_dir)
        if dir_path:
            self.master.document_dir = dir_path
            self.doc_btn.configure(text=dir_path)

    def select_img_dir(self):
        dir_path = ctk.filedialog.askdirectory(initialdir=self.master.image_dir)
        if dir_path:
            self.master.image_dir = dir_path
            self.img_btn.configure(text=dir_path)

    def save_settings(self):
        for key, entry in self.entries.items():
            setattr(self.master, key, entry.get().strip())
        try:
            self.master.save_config()
            self.master.update_texts()
            self.update_appearance()
            messagebox.showinfo("成功", "设置已成功保存!")
            self.destroy()
        except Exception as e:
            messagebox.showerror("错误", f"保存设置失败: {str(e)}")

    def update_appearance(self):
        ctk.set_appearance_mode("dark" if self.master.dark_mode.get() else "light")