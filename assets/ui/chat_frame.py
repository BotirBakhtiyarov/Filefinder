import queue
import threading
import requests
import json
import customtkinter as ctk
from tkinter import filedialog, messagebox
from assets.utils.file_utils import extract_text_content
from assets.utils.ai_utils import get_chat_payload, parse_streaming_response
import os

class ChatFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.history = []
        self.queue = queue.Queue()
        self.uploaded_docs = []
        self.uploaded_filenames = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.text_widget = ctk.CTkTextbox(self, wrap="word")
        self.text_widget.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")

        self.mode_var = ctk.StringVar(value="普通模式")
        self.mode_menu = ctk.CTkOptionMenu(
            self,
            values=["普通模式", "RAG模式"],
            variable=self.mode_var,
            command=self.toggle_mode
        )
        self.mode_menu.grid(row=1, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="ew")

        self.uploaded_files_label = ctk.CTkLabel(
            self,
            text="未上传文件",
            font=("Arial", 12),
            wraplength=600
        )
        self.uploaded_files_label.grid(row=2, column=0, columnspan=2, padx=10, pady=(0, 5), sticky="w")

        self.entry = ctk.CTkEntry(self)
        self.entry.grid(row=3, column=0, padx=10, pady=5, sticky="ew")

        self.send_btn = ctk.CTkButton(
            self,
            text=self.master.get_translation("send_button"),
            command=self.send_message,
            fg_color="#1f6aa8",
            hover_color="#14487f"
        )
        self.send_btn.grid(row=3, column=1, padx=5, pady=5)

        self.new_chat_btn = ctk.CTkButton(
            self,
            text=self.master.get_translation("new_chat_button"),
            command=self.new_chat,
            fg_color="#1f6aa8",
            hover_color="#14487f"
        )
        self.new_chat_btn.grid(row=4, column=0, padx=10, pady=10, sticky="w")

        self.upload_btn = ctk.CTkButton(
            self,
            text="+",
            width=40,
            command=self.upload_files,
            fg_color="#1f6aa8",
            hover_color="#14487f"
        )
        self.upload_btn.grid(row=4, column=1, padx=5, pady=10)
        self.upload_btn.grid_remove()

        self.text_widget.tag_config("user", foreground="#0084FF")
        self.text_widget.tag_config("assistant", foreground="#00B894")
        self.text_widget.tag_config("error", foreground="#FF5555")
        self.text_widget.configure(state="disabled")

        self.entry.bind("<Return>", lambda e: self.send_message())
        self.after(100, self.process_queue)

    def update_texts(self):
        self.send_btn.configure(text=self.master.get_translation("send_button"))
        self.new_chat_btn.configure(text=self.master.get_translation("new_chat_button"))

    def toggle_mode(self, mode):
        if mode == "RAG模式":
            self.upload_btn.grid()
            self.update_uploaded_files_label()
        else:
            self.upload_btn.grid_remove()
            self.uploaded_docs.clear()
            self.uploaded_filenames.clear()
            self.uploaded_files_label.configure(text="未上传文件")
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", f"\n模式切换至: {mode}\n", "assistant")
        self.text_widget.configure(state="disabled")
        self.text_widget.see("end")

    def update_uploaded_files_label(self):
        if self.uploaded_filenames:
            filenames = ", ".join(os.path.basename(f) for f in self.uploaded_filenames)
            self.uploaded_files_label.configure(text=f"已上传: {filenames}")
        else:
            self.uploaded_files_label.configure(text="未上传文件")

    def send_message(self):
        prompt = self.entry.get().strip()
        if not prompt:
            return
        self.entry.delete(0, "end")
        self.history.append({"role": "user", "content": prompt})
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", f"\n你: {prompt}\n", "user")
        self.text_widget.configure(state="disabled")
        self.text_widget.see("end")
        threading.Thread(target=self.get_response, args=(prompt,), daemon=True).start()

    def get_response(self, prompt):
        model = self.master.chat_model.get()
        headers = {"Content-Type": "application/json"}
        if self.master.chat_api_key:
            headers["Authorization"] = f"Bearer {self.master.chat_api_key}"

        if self.mode_var.get() == "RAG模式" and self.uploaded_docs:
            doc_text = "\n\n".join(self.uploaded_docs)
            full_query = f"基于以下文档:\n{doc_text}\n\n{prompt}"
        else:
            full_query = prompt

        data = get_chat_payload(model, full_query)

        try:
            response = requests.post(self.master.chat_api_url, headers=headers, json=data, stream=True, timeout=10)
            response.raise_for_status()
            self.queue.put(("assistant_prefix", "助手: "))

            content = []
            for chunk in parse_streaming_response(model, response):
                self.queue.put(("chunk", chunk))
                content.append(chunk)

            self.history.append({"role": "assistant", "content": "".join(content)})
        except Exception as e:
            self.queue.put(("error", f"错误: {str(e)}"))

    def process_queue(self):
        while not self.queue.empty():
            item_type, content = self.queue.get_nowait()
            self.text_widget.configure(state="normal")
            if item_type == "assistant_prefix":
                self.text_widget.insert("end", content, "assistant")
            elif item_type == "chunk":
                self.text_widget.insert("end", content, "assistant")
            elif item_type == "error":
                self.text_widget.insert("end", f"\n{content}\n", "error")
            self.text_widget.configure(state="disabled")
            self.text_widget.see("end")
        self.after(100, self.process_queue)

    def new_chat(self):
        self.history.clear()
        self.uploaded_docs.clear()
        self.uploaded_filenames.clear()
        self.text_widget.configure(state="normal")
        self.text_widget.delete("1.0", "end")
        self.text_widget.configure(state="disabled")
        self.update_uploaded_files_label()

    def upload_files(self):
        files = filedialog.askopenfilenames(filetypes=[("文档", "*.docx *.pdf *.txt *.pptx *.xlsx")])
        for file in files:
            content = extract_text_content(file)
            if content:
                self.uploaded_docs.append(content)
                self.uploaded_filenames.append(file)
        if files:
            self.update_uploaded_files_label()
            messagebox.showinfo("信息", f"已上传 {len(files)} 个文件用于RAG模式。")