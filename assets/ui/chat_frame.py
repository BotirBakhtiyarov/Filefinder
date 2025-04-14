import queue
import threading
import requests
import json
import customtkinter as ctk
from tkinter import filedialog, messagebox
from assets.utils.file_utils import extract_text_content
from assets.utils.ai_utils import get_chat_payload, parse_streaming_response

class ChatFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.history = []
        self.queue = queue.Queue()
        self.uploaded_docs = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.text_widget = ctk.CTkTextbox(self, wrap="word")
        self.text_widget.grid(row=0, column=0, columnspan=3, padx=10, pady=10, sticky="nsew")
        self.entry = ctk.CTkEntry(self)
        self.entry.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.send_btn = ctk.CTkButton(self, text=self.master.get_translation("send_button"), command=self.send_message, fg_color="#1f6aa8", hover_color="#14487f")
        self.send_btn.grid(row=1, column=1, padx=5, pady=10)
        self.new_chat_btn = ctk.CTkButton(self, text=self.master.get_translation("new_chat_button"), command=self.new_chat, fg_color="#1f6aa8", hover_color="#14487f")
        self.new_chat_btn.grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.upload_btn = ctk.CTkButton(self, text="+", width=40, command=self.upload_files, fg_color="#1f6aa8", hover_color="#14487f")
        self.upload_btn.grid(row=2, column=1, padx=5, pady=10)

        self.text_widget.tag_config("user", foreground="#0084FF")
        self.text_widget.tag_config("assistant", foreground="#00B894")
        self.text_widget.tag_config("error", foreground="#FF5555")
        self.text_widget.configure(state="disabled")

        self.entry.bind("<Return>", lambda e: self.send_message())
        self.after(100, self.process_queue)

    def update_texts(self):
        self.send_btn.configure(text=self.master.get_translation("send_button"))
        self.new_chat_btn.configure(text=self.master.get_translation("new_chat_button"))

    def send_message(self):
        prompt = self.entry.get().strip()
        if not prompt:
            return
        self.entry.delete(0, "end")
        self.history.append({"role": "user", "content": prompt})
        self.text_widget.configure(state="normal")
        self.text_widget.insert("end", f"\nYou: {prompt}\n", "user")
        self.text_widget.configure(state="disabled")
        self.text_widget.see("end")
        threading.Thread(target=self.get_response, args=(prompt,), daemon=True).start()

    def get_response(self, prompt):
        model = self.master.chat_model.get()
        headers = {"Content-Type": "application/json"}
        if self.master.chat_api_key:
            headers["Authorization"] = f"Bearer {self.master.chat_api_key}"

        if self.uploaded_docs:
            doc_text = "\n\n".join(self.uploaded_docs)
            full_query = f"Based on the following documents:\n{doc_text}\n\n{prompt}"
        else:
            full_query = prompt

        data = get_chat_payload(model, full_query)

        try:
            response = requests.post(self.master.chat_api_url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            self.queue.put(("assistant_prefix", "Assistant: "))

            content = []
            for chunk in parse_streaming_response(model, response):
                self.queue.put(("chunk", chunk))
                content.append(chunk)

            self.history.append({"role": "assistant", "content": "".join(content)})
        except Exception as e:
            self.queue.put(("error", f"Error: {str(e)}"))

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
        self.uploaded_docs = []
        self.text_widget.configure(state="normal")
        self.text_widget.delete("1.0", "end")
        self.text_widget.configure(state="disabled")

    def upload_files(self):
        files = filedialog.askopenfilenames(filetypes=[("Documents", "*.docx *.pdf *.txt")])
        for file in files:
            content = extract_text_content(file)
            if content:
                self.uploaded_docs.append(content)
        if files:
            messagebox.showinfo("Info", f"{len(files)} files uploaded for RAG mode.")