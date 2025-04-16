import customtkinter as ctk
from tkinter import filedialog, messagebox

class SetupWizard(ctk.CTkToplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.master = master
        self.callback = callback
        self.title("初始设置")
        self.geometry("500x750")
        self.resizable(False, True)

        self.config = {
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

        ctk.CTkLabel(self, text="设置配置", font=("Arial", 16)).pack(pady=10)

        self.lang_var = ctk.StringVar(value="ZH")
        self.chat_model_var = ctk.StringVar(value="Regular")
        ctk.CTkLabel(self, text="聊天AI模型:").pack(pady=5)
        self.chat_model_option = ctk.CTkOptionMenu(self, values=self.master.chat_models, variable=self.chat_model_var, command=self.on_ai_model_change)
        self.chat_model_option.pack(pady=5)

        ctk.CTkLabel(self, text="嵌入API URL:").pack(pady=5)
        self.embed_api_url_entry = ctk.CTkEntry(self, width=400)
        self.embed_api_url_entry.insert(0, "http://localhost:5000")
        self.embed_api_url_entry.pack(pady=5)
        ctk.CTkLabel(self, text="嵌入API Key:").pack(pady=5)
        self.embed_api_key_entry = ctk.CTkEntry(self, width=400)
        self.embed_api_key_entry.pack(pady=5)

        ctk.CTkLabel(self, text="聊天API URL:").pack(pady=5)
        self.chat_api_url_entry = ctk.CTkEntry(self, width=400)
        self.chat_api_url_entry.insert(0, "http://10.20.1.213/v1/chat-messages")
        self.chat_api_url_entry.pack(pady=5)
        ctk.CTkLabel(self, text="聊天API Key:").pack(pady=5)
        self.chat_api_key_entry = ctk.CTkEntry(self, width=400)
        self.chat_api_key_entry.pack(pady=5)

        ctk.CTkLabel(self, text="选择目录", font=("Arial", 16)).pack(pady=10)
        self.doc_btn = ctk.CTkButton(self, text="选择文档目录", command=self.select_doc_dir, fg_color="#1f6aa8", hover_color="#14487f")
        self.doc_btn.pack(pady=5)
        self.img_btn = ctk.CTkButton(self, text="选择图像目录", command=self.select_img_dir, fg_color="#1f6aa8", hover_color="#14487f")
        self.img_btn.pack(pady=5)

        self.dark_mode_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self, text="深色模式", variable=self.dark_mode_var).pack(pady=10)

        ctk.CTkButton(self, text="完成", command=self.finish_setup, fg_color="#1f6aa8", hover_color="#14487f").pack(pady=20)

    def select_doc_dir(self):
        dir = filedialog.askdirectory()
        if dir:
            self.config["document_dir"] = dir
            self.doc_btn.configure(text=dir)

    def select_img_dir(self):
        dir = filedialog.askdirectory()
        if dir:
            self.config["image_dir"] = dir
            self.img_btn.configure(text=dir)

    def finish_setup(self):
        self.config["api_url"] = self.embed_api_url_entry.get().strip()
        self.config["api_key"] = self.embed_api_key_entry.get().strip()
        self.config["chat_api_url"] = self.chat_api_url_entry.get().strip()
        self.config["chat_api_key"] = self.chat_api_key_entry.get().strip()
        self.config["language"] = self.lang_var.get()
        self.config["chat_model"] = self.chat_model_var.get()
        self.config["dark_mode"] = self.dark_mode_var.get()

        if not self.config["document_dir"] or not self.config["image_dir"]:
            messagebox.showwarning("警告", "未选择一个或两个目录。您可以稍后在设置中设置它们。")

        try:
            self.callback(self.config)
        except Exception as e:
            messagebox.showerror("错误", f"保存配置失败： {str(e)}")

    def on_ai_model_change(self, selected_model):
        if selected_model == "Regular":
            self.chat_api_url_entry.delete(0, "end")
            self.chat_api_url_entry.insert(0, "http://10.20.1.213/v1/chat-messages")
        elif selected_model == "OpenAI":
            self.chat_api_url_entry.delete(0, "end")
            self.chat_api_url_entry.insert(0, "https://api.openai.com/v1/chat/completions")
        elif selected_model == "DeepSeek":
            self.chat_api_url_entry.delete(0, "end")
            self.chat_api_url_entry.insert(0, "https://api.deepseek.com/chat/completions")
        elif selected_model == "Ollama":
            self.chat_api_url_entry.delete(0, "end")
            self.chat_api_url_entry.insert(0, "http://localhost:11434/api/generate")