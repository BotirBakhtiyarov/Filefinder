import queue
import threading
import requests
import json
from customtkinter import CTkTextbox
from assets.utils.file_utils import extract_text_content
import customtkinter as ctk
import os
import base64

class SummaryWindow(ctk.CTkToplevel):
    def __init__(self, master, file_path, use_ocr=False):
        super().__init__(master)
        self.master = master
        self.file_path = file_path
        self.use_ocr = use_ocr  # Flag to determine if OCR should be used
        self.title(f"{'OCR ' if use_ocr else ''}Summary of {os.path.basename(file_path)}")
        self.geometry("600x400")
        self.queue = queue.Queue()

        self.text_widget = CTkTextbox(self, wrap="word", height=300)
        self.text_widget.pack(padx=10, pady=10, fill="both", expand=True)
        self.text_widget.tag_config("assistant", foreground="#00B894")
        self.text_widget.tag_config("error", foreground="#FF5555")
        self.text_widget.configure(state="disabled")

        self.generate_summary()
        self.after(100, self.process_queue)

    def generate_summary(self):
        content = ""
        
        if self.use_ocr:
            # Extract text from image using OCR API
            content = self.extract_ocr_from_image()
        else:
            # Extract text from the file (non-image content)
            content = extract_text_content(self.file_path)
            if not content or content.strip() == "":
                content = ""

            # If it's a PDF, also extract text from images
            if self.file_path.lower().endswith(".pdf"):
                ocr_content = self.extract_ocr_from_pdf()
                content += "\n" + ocr_content if ocr_content else ""

        if not content.strip():
            self.queue.put(("error", "Error: No content extracted from the file."))
            return

        prompt = f"Summarize the following content:\n\n{content}"
        threading.Thread(target=self.stream_summary, args=(prompt,), daemon=True).start()

    def extract_ocr_from_image(self):
        """Extract text from an image using the OCR API."""
        try:
            with open(self.file_path, "rb") as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode("utf-8")
            headers = {"Authorization": f"Bearer {self.master.master.api_key}", "Content-Type": "application/json"}
            data = {"image": encoded_image}
            response = requests.post(
                f"{self.master.master.api_url}/extract_image_ocr", headers=headers, json=data, timeout=10
            )
            response.raise_for_status()
            text = response.json().get("text", "")
            return text if text.strip() else "No text detected in the image."
        except Exception as e:
            self.queue.put(("error", f"OCR Extraction Error: {str(e)}"))
            return ""

    def extract_ocr_from_pdf(self):
        """Extract text from PDF images using the API."""
        try:
            with open(self.file_path, "rb") as pdf_file:
                encoded_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
            headers = {"Authorization": f"Bearer {self.master.master.api_key}", "Content-Type": "application/json"}
            data = {"pdf": encoded_pdf}
            response = requests.post(
                f"{self.master.master.api_url}/extract_pdf_with_ocr", headers=headers, json=data, timeout=10
            )
            response.raise_for_status()
            return response.json().get("text", "")
        except Exception as e:
            self.queue.put(("error", f"OCR Extraction Error: {str(e)}"))
            return ""

    def stream_summary(self, prompt):
        headers = {"Authorization": f"Bearer {self.master.master.chat_api_key}", "Content-Type": "application/json"}
        data = {
            "inputs": {},
            "query": prompt,
            "response_mode": "streaming",
            "conversation_id": "",
            "user": "abc-123",
            "files": []
        }
        try:
            response = requests.post(self.master.master.chat_api_url, headers=headers, json=data, stream=True, timeout=10)
            response.raise_for_status()
            self.queue.put(("assistant_prefix", "Summary: "))
            for line in response.iter_lines():
                if line and line.startswith(b"data: "):
                    data = json.loads(line[6:].decode("utf-8"))
                    if "answer" in data:
                        self.queue.put(("chunk", data["answer"]))
        except Exception as e:
            self.queue.put(("error", f"Chat API Error: {str(e)}"))

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