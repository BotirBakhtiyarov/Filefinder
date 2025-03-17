import customtkinter as ctk
from tkinter import filedialog, messagebox

class SetupWizard(ctk.CTkToplevel):
    def __init__(self, master, callback):
        super().__init__(master)
        self.master = master
        self.callback = callback
        self.title("Initial Setup")
        self.geometry("500x600")
        self.resizable(False, False)

        self.config = {
            "api_url": "",
            "api_key": "",
            "chat_api_url": "",
            "chat_api_key": "",
            "document_dir": "",
            "image_dir": "",
            "language": "en",
            "dark_mode": True
        }

        ctk.CTkLabel(self, text="API Configuration", font=("Arial", 16)).pack(pady=10)
        self.entries = {}
        for label, key in [("API URL", "api_url"), ("API Key", "api_key"),
                           ("Chat API URL", "chat_api_url"), ("Chat API Key", "chat_api_key")]:
            ctk.CTkLabel(self, text=f"{label}:").pack(pady=5)
            entry = ctk.CTkEntry(self, width=400)
            entry.insert(0, self.config[key])
            entry.pack(pady=5)
            self.entries[key] = entry

        ctk.CTkLabel(self, text="Select Directories", font=("Arial", 16)).pack(pady=10)
        self.doc_btn = ctk.CTkButton(self, text="Select Document Directory", command=self.select_doc_dir)
        self.doc_btn.pack(pady=5)
        self.img_btn = ctk.CTkButton(self, text="Select Image Directory", command=self.select_img_dir)
        self.img_btn.pack(pady=5)

        self.dark_mode_var = ctk.BooleanVar(value=True)
        ctk.CTkSwitch(self, text="Dark Mode", variable=self.dark_mode_var).pack(pady=10)

        ctk.CTkButton(self, text="Finish", command=self.finish_setup).pack(pady=20)

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
        for key, entry in self.entries.items():
            self.config[key] = entry.get().strip()
        if not self.config["document_dir"] or not self.config["image_dir"]:
            messagebox.showerror("Error", "Please select both directories!")
            return
        self.config["dark_mode"] = self.dark_mode_var.get()
        self.callback(self.config)