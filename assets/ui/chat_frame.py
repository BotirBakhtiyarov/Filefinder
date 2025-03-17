import queue
import threading
import requests
import json
import customtkinter as ctk

class ChatFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.history = []
        self.queue = queue.Queue()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.text_widget = ctk.CTkTextbox(self, wrap="word")
        self.text_widget.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.entry = ctk.CTkEntry(self)
        self.entry.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        self.send_btn = ctk.CTkButton(self, text=self.master.get_translation("send_button"), command=self.send_message)
        self.send_btn.grid(row=1, column=1, padx=10, pady=10)
        self.new_chat_btn = ctk.CTkButton(self, text=self.master.get_translation("new_chat_button"), command=self.new_chat)
        self.new_chat_btn.grid(row=2, column=0, padx=10, pady=10, sticky="w")

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
        headers = {"Authorization": f"Bearer {self.master.chat_api_key}", "Content-Type": "application/json"}
        data = {
            "inputs": {}, "query": prompt, "response_mode": "streaming", "conversation_id": "", "user": "abc-123", "files": []
        }
        try:
            response = requests.post(self.master.chat_api_url, headers=headers, json=data, stream=True)
            response.raise_for_status()
            self.queue.put(("assistant_prefix", "Assistant: "))
            content = []
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    data = json.loads(line[6:].decode("utf-8"))
                    if "answer" in data:
                        self.queue.put(("chunk", data["answer"]))
                        content.append(data["answer"])
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
        self.text_widget.configure(state="normal")
        self.text_widget.delete("1.0", "end")
        self.text_widget.configure(state="disabled")