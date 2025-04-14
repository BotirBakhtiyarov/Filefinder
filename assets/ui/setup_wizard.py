import customtkinter as ctk
from tkinter import filedialog, messagebox
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SetupWizard(ctk.CTkToplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.master = master
        self.callback = callback
        self.title("Initial Setup - v2")
        self.geometry("500x650")
        # self.resizable(False, False)

        self.config = {
            "api_url": "",
            "api_key": "",
            "chat_api_url": "",
            "chat_api_key": "",
            "document_dir": "",
            "image_dir": "",
            "language": "EN",
            "chat_model": "Regular",
            "dark_mode": True
        }

        ctk.CTkLabel(self, text="Setup Configuration", font=("Arial", 16)).pack(pady=10)

        # Language selection
        self.lang_var = ctk.StringVar(value="EN")
        ctk.CTkLabel(self, text="Language:").pack(pady=5)
        self.lang_option = ctk.CTkOptionMenu(self, values=["EN", "ZH", "RU", "UZ"], variable=self.lang_var)
        self.lang_option.pack(pady=5)

        # Chat AI model selection
        self.chat_model_var = ctk.StringVar(value="Regular")
        ctk.CTkLabel(self, text="Chat AI Model:").pack(pady=5)
        self.chat_model_option = ctk.CTkOptionMenu(self, values=self.master.chat_models, variable=self.chat_model_var, command=self.on_ai_model_change)
        self.chat_model_option.pack(pady=5)


        ctk.CTkLabel(self, text="Embed API URL:").pack(pady=5)
        self.embed_api_url_entry = ctk.CTkEntry(self, width=400)
        self.embed_api_url_entry.pack(pady=5)
        ctk.CTkLabel(self, text="Embed API Key:").pack(pady=5)
        self.embed_api_key_entry = ctk.CTkEntry(self, width=400)
        self.embed_api_key_entry.pack(pady=5)

        # API fields
        ctk.CTkLabel(self, text="Chat API URL:").pack(pady=5)
        self.chat_api_url_entry = ctk.CTkEntry(self, width=400)
        self.chat_api_url_entry.insert(0, "http://10.20.1.213/v1/chat-messages")
        self.chat_api_url_entry.pack(pady=5)
        ctk.CTkLabel(self, text="Chat API Key:").pack(pady=5)
        self.chat_api_key_entry = ctk.CTkEntry(self, width=400)
        self.chat_api_key_entry.pack(pady=5)

        ctk.CTkLabel(self, text="Select Directories", font=("Arial", 16)).pack(pady=10)
        self.doc_btn = ctk.CTkButton(self, text="Select Document Directory", command=self.select_doc_dir, fg_color="#1f6aa8", hover_color="#14487f")
        self.doc_btn.pack(pady=5)
        self.img_btn = ctk.CTkButton(self, text="Select Image Directory", command=self.select_img_dir, fg_color="#1f6aa8", hover_color="#14487f")
        self.img_btn.pack(pady=5)

        self.dark_mode_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self, text="Dark Mode", variable=self.dark_mode_var).pack(pady=10)

        ctk.CTkButton(self, text="Finish", command=self.finish_setup, fg_color="#1f6aa8", hover_color="#14487f").pack(pady=20)


    def select_doc_dir(self):
        dir = filedialog.askdirectory()
        if dir:
            self.config["document_dir"] = dir
            self.doc_btn.configure(text=dir)
            logging.debug(f"Selected document directory: {dir}")

    def select_img_dir(self):
        dir = filedialog.askdirectory()
        if dir:
            self.config["image_dir"] = dir
            self.img_btn.configure(text=dir)
            logging.debug(f"Selected image directory: {dir}")

    def finish_setup(self):
        self.config["api_url"] = self.embed_api_url_entry.get().strip()
        self.config["api_key"] = self.embed_api_key_entry.get().strip()
        self.config["chat_api_url"] = self.chat_api_url_entry.get().strip()
        self.config["chat_api_key"] = self.chat_api_key_entry.get().strip()
        self.config["language"] = self.lang_var.get()
        self.config["chat_model"] = self.chat_model_var.get()
        self.config["dark_mode"] = self.dark_mode_var.get()

        # Log the config before saving
        logging.debug(f"SetupWizard config: {self.config}")

        # Optional: Warn about missing directories but proceed
        if not self.config["document_dir"] or not self.config["image_dir"]:
            messagebox.showwarning("Warning", "One or both directories are not selected. You can set them later in settings.")

        try:
            self.callback(self.config)
            logging.debug("SetupWizard callback executed successfully")
        except Exception as e:
            logging.error(f"Error in callback: {str(e)}")
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def on_ai_model_change(self, selected_model):
        if selected_model == "Regular":
            self.chat_api_url_entry.delete(0, "end")
            self.chat_api_url_entry.insert(0, "http://10.20.1.213/v1/chat-messages")
        elif selected_model == "ChatGPT":
            self.chat_api_url_entry.delete(0, "end")
            self.chat_api_url_entry.insert(0, "https://api.openai.com/v1/chat/completions")
        elif selected_model == "DeepSeek":
            self.chat_api_url_entry.delete(0, "end")
            self.chat_api_url_entry.insert(0, "https://api.deepseek.com/v1/chat/completions")
        elif selected_model == "Ollama":
            self.chat_api_url_entry.delete(0, "end")
            self.chat_api_url_entry.insert(0, "http://localhost:11434/api/generate")