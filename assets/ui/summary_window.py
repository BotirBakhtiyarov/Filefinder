import queue
import threading
import requests
import json
from customtkinter import CTkTextbox
from assets.utils.file_utils import extract_text_content
from assets.utils.ai_utils import get_chat_payload, parse_streaming_response
import customtkinter as ctk
import os
import base64

class SummaryWindow(ctk.CTkToplevel):
    def __init__(self, master, file_path, use_ocr=False):
        super().__init__(master)
        self.master = master
        self.file_path = file_path
        self.use_ocr = use_ocr
        self.title(f"{'OCR ' if use_ocr else ''}摘要: {os.path.basename(file_path)}")
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
            content = self.extract_ocr_from_image()
        else:
            content = extract_text_content(self.file_path)
            if not content or content.strip() == "":
                content = ""
            if self.file_path.lower().endswith(".pdf"):
                ocr_content = self.extract_ocr_from_pdf()
                content += "\n" + ocr_content if ocr_content else ""

        if not content.strip():
            self.queue.put(("error", "错误: 无法从文件中提取内容。"))
            return

        prompt = f"总结以下内容:\n\n{content}"
        threading.Thread(target=self.stream_summary, args=(prompt,), daemon=True).start()

    def extract_ocr_from_image(self):
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
            return text if text.strip() else "图像中未检测到文本。"
        except Exception as e:
            self.queue.put(("error", f"OCR提取错误: {str(e)}"))
            return ""

    def extract_ocr_from_pdf(self):
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
            self.queue.put(("error", f"OCR提取错误: {str(e)}"))
            return ""

    def stream_summary(self, prompt):
        model = self.master.master.chat_model.get()
        headers = {"Content-Type": "application/json"}
        if self.master.master.chat_api_key:
            headers["Authorization"] = f"Bearer {self.master.master.chat_api_key}"

        data = get_chat_payload(model, prompt)

        try:
            response = requests.post(self.master.master.chat_api_url, headers=headers, json=data, stream=True, timeout=10)
            response.raise_for_status()
            self.queue.put(("assistant_prefix", "摘要: "))

            for chunk in parse_streaming_response(model, response):
                self.queue.put(("chunk", chunk))
        except Exception as e:
            self.queue.put(("error", f"聊天API错误: {str(e)}"))

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