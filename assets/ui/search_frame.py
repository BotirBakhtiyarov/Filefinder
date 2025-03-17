import os
import json
import threading
import base64
import requests
import numpy as np
from tkinter import filedialog
from customtkinter import CTkImage
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from .summary_window import SummaryWindow
from assets.utils.file_utils import extract_text_content, scan_files
import time
import customtkinter as ctk

class SearchFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.index_file = "file_index.json"
        self.indexing_in_progress = False
        self.stop_indexing = threading.Event()
        self.load_index()
        self.create_widgets()
        self.start_continuous_indexing()

    def create_widgets(self):
        self.notebook = ctk.CTkTabview(self)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=5)
        self.text_tab = self.notebook.add("Documents")
        self.text_scroll = ctk.CTkScrollableFrame(self.text_tab)
        self.text_scroll.pack(expand=True, fill="both")
        self.image_tab = self.notebook.add("Images")
        self.image_scroll = ctk.CTkScrollableFrame(self.image_tab)
        self.image_scroll.pack(expand=True, fill="both")

        self.search_frame = ctk.CTkFrame(self)
        self.search_frame.pack(pady=10, padx=10, fill="x")
        self.search_entry = ctk.CTkEntry(
            self.search_frame, placeholder_text=self.master.get_translation("search_placeholder"), width=600
        )
        self.search_entry.pack(side="left", expand=True, padx=5)
        self.search_entry.bind("<Return>", lambda e: self.perform_search())
        self.search_btn = ctk.CTkButton(
            self.search_frame, text=self.master.get_translation("search_button"), command=self.perform_search
        )
        self.search_btn.pack(side="left", padx=5)
        self.index_btn = ctk.CTkButton(
            self.search_frame, text="Index Folders", command=self.select_folders_to_index
        )
        self.index_btn.pack(side="left", padx=5)

        self.status_frame = ctk.CTkFrame(self)
        self.status_frame.pack(side="bottom", fill="x")
        self.status_var = ctk.StringVar(value=self.master.get_translation("status_ready"))
        self.status_label = ctk.CTkLabel(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side="left", fill="x", expand=True)
        self.progress_bar = ctk.CTkProgressBar(self.status_frame, mode="indeterminate")
        self.progress_bar.pack(side="right", padx=10)
        self.progress_bar.stop()

    def update_texts(self):
        self.search_entry.configure(placeholder_text=self.master.get_translation("search_placeholder"))
        self.search_btn.configure(text=self.master.get_translation("search_button"))
        if not self.indexing_in_progress:
            self.status_var.set(self.master.get_translation("status_ready"))

    def load_index(self):
        self.index_data = {"text": {}, "images": {}}
        if os.path.exists(self.index_file):
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.index_data = json.load(f)

    def save_index(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index_data, f)

    def select_folders_to_index(self):
        folders = []
        while True:
            folder = filedialog.askdirectory(title="Select Folder to Index")
            if not folder:
                break
            if folder not in folders:
                folders.append(folder)
        if folders:
            self.status_var.set("Indexing selected folders...")
            self.progress_bar.start()
            threading.Thread(target=self.index_selected_folders, args=(folders,), daemon=True).start()

    def index_selected_folders(self, folders):
        headers = {"Authorization": f"Bearer {self.master.api_key}"} if self.master.api_key else {}
        text_extensions = {"docx", "xlsx", "pptx", "pdf", "txt"}
        image_extensions = {"jpg", "jpeg", "png", "gif", "bmp"}

        for folder in folders:
            text_files = scan_files(folder, text_extensions)
            image_files = scan_files(folder, image_extensions)

            for file in text_files:
                current_mtime = os.path.getmtime(file)
                if file not in self.index_data["text"] or self.index_data["text"][file]["mtime"] != current_mtime:
                    content = extract_text_content(file)
                    if content:
                        response = requests.post(
                            f"{self.master.api_url}/embed_text", json={"text": content}, headers=headers
                        )
                        if response.status_code == 200:
                            self.index_data["text"][file] = {
                                "embedding": response.json()["embedding"], "mtime": current_mtime
                            }

            for file in image_files:
                current_mtime = os.path.getmtime(file)
                if file not in self.index_data["images"] or self.index_data["images"][file]["mtime"] != current_mtime:
                    with open(file, "rb") as img_file:
                        encoded = base64.b64encode(img_file.read()).decode("utf-8")
                    response = requests.post(
                        f"{self.master.api_url}/embed_image", json={"image": encoded}, headers=headers
                    )
                    if response.status_code == 200:
                        self.index_data["images"][file] = {
                            "embedding": response.json()["embedding"], "mtime": current_mtime
                        }

        self.save_index()
        self.progress_bar.stop()
        self.status_var.set(self.master.get_translation("status_ready"))

    def start_continuous_indexing(self):
        threading.Thread(target=self.continuous_indexing, daemon=True).start()

    def continuous_indexing(self):
        while not self.stop_indexing.is_set():
            if not self.indexing_in_progress:
                self.indexing_in_progress = True
                self.update_index()
                self.indexing_in_progress = False
            time.sleep(60)

    def stop_continuous_indexing(self):
        self.stop_indexing.set()

    def update_index(self):
        text_files = scan_files(self.master.document_dir, {"docx", "xlsx", "pptx", "pdf", "txt"})
        image_files = scan_files(self.master.image_dir, {"jpg", "jpeg", "png", "gif", "bmp"})
        headers = {"Authorization": f"Bearer {self.master.api_key}"} if self.master.api_key else {}

        for file in text_files:
            current_mtime = os.path.getmtime(file)
            if file not in self.index_data["text"] or self.index_data["text"][file]["mtime"] != current_mtime:
                content = extract_text_content(file)
                if content:
                    response = requests.post(
                        f"{self.master.api_url}/embed_text", json={"text": content}, headers=headers
                    )
                    if response.status_code == 200:
                        self.index_data["text"][file] = {
                            "embedding": response.json()["embedding"], "mtime": current_mtime
                        }

        for file in image_files:
            current_mtime = os.path.getmtime(file)
            if file not in self.index_data["images"] or self.index_data["images"][file]["mtime"] != current_mtime:
                with open(file, "rb") as img_file:
                    encoded = base64.b64encode(img_file.read()).decode("utf-8")
                response = requests.post(
                    f"{self.master.api_url}/embed_image", json={"image": encoded}, headers=headers
                )
                if response.status_code == 200:
                    self.index_data["images"][file] = {
                        "embedding": response.json()["embedding"], "mtime": current_mtime
                    }

        for key in ["text", "images"]:
            existing = text_files if key == "text" else image_files
            for path in list(self.index_data[key].keys()):
                if path not in existing:
                    del self.index_data[key][path]

        self.save_index()
        if self.winfo_exists():
            self.after(0, self.progress_bar.stop)
            self.after(0, lambda: self.status_var.set(self.master.get_translation("status_ready")))

    def perform_search(self):
        query = self.search_entry.get().strip()
        if len(query) < 2:
            return
        for widget in self.text_scroll.winfo_children() + self.image_scroll.winfo_children():
            widget.destroy()
        self.status_var.set("Searching...")
        self.progress_bar.start()
        threading.Thread(target=self.run_search, args=(query,), daemon=True).start()

    def run_search(self, query):
        text_results = self.search_text(query)
        image_results = self.search_images(query)
        self.after(0, lambda: self.display_results(text_results, image_results))

    def search_text(self, query):
        headers = {"Authorization": f"Bearer {self.master.api_key}"} if self.master.api_key else {}
        response = requests.post(
            f"{self.master.api_url}/embed_text", json={"text": query}, headers=headers
        )
        if response.status_code != 200:
            raise Exception(f"Embedding error: {response.text}")
        query_embed = np.array(response.json()["embedding"])
        scores = []
        for path, data in self.index_data["text"].items():
            similarity = cosine_similarity([query_embed], [np.array(data["embedding"])])[0][0]
            if similarity >= 0.2:
                scores.append((path, similarity))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:10]

    def search_images(self, query):
        headers = {"Authorization": f"Bearer {self.master.api_key}"} if self.master.api_key else {}
        response = requests.post(
            f"{self.master.api_url}/embed_clip_text", json={"text": query}, headers=headers
        )
        if response.status_code != 200:
            raise Exception(f"Embedding error: {response.text}")
        query_embed = np.array(response.json()["embedding"]).reshape(1, -1)
        scores = []
        for path, data in self.index_data["images"].items():
            img_embed = np.array(data["embedding"]).reshape(1, -1)
            score = cosine_similarity(query_embed, img_embed)[0][0]
            if score >= 0.4:
                scores.append((path, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:5]

    def display_results(self, text_results, image_results):
        # Display text results (unchanged)
        for path, score in text_results:
            frame = ctk.CTkFrame(self.text_scroll)
            frame.pack(fill="x", pady=2)
            norm_path = norm_path = os.path.normpath(path)
            ctk.CTkLabel(frame, text=f"{os.path.basename(path)} ({score:.1%})").pack(side="left", padx=5)
            ctk.CTkButton(frame, text="Open", width=50, command=lambda p=norm_path: os.startfile(p)).pack(side="right", padx=5)
            ctk.CTkButton(frame, text="Summary", width=70, command=lambda p=norm_path: self.show_summary(p)).pack(side="right", padx=5)

        # Display image results with Open and OCR Summary buttons
        row, col = 0, 0
        for path, score in image_results:
            img = Image.open(path).resize((200, 200))
            photo = CTkImage(light_image=img, size=(200, 200))
            frame = ctk.CTkFrame(self.image_scroll)
            frame.grid(row=row, column=col, padx=5, pady=5)

            # Image display
            ctk.CTkLabel(frame, image=photo, text="").pack()

            # File name and score
            ctk.CTkLabel(frame, text=f"{os.path.basename(path)}\n({score:.1%})").pack()

            # Button frame for Open and OCR Summary
            btn_frame = ctk.CTkFrame(frame)
            btn_frame.pack(pady=5)
            norm_path = os.path.normpath(path)
            # Open button
            ctk.CTkButton(btn_frame, text="Open", width=70, command=lambda p=norm_path: os.startfile(p)).pack(side="left", padx=5)

            # OCR Summary button
            ctk.CTkButton(btn_frame, text="OCR+ AI Summary", width=100, command=lambda p=norm_path: self.show_ocr_summary(p)).pack(side="left", padx=5)

            col = (col + 1) % 3
            row += 1 if col == 0 else 0

        self.progress_bar.stop()
        self.status_var.set(self.master.get_translation("status_ready"))
    
    def show_ocr_summary(self, image_path):
        """Display a summary window with OCR-extracted text from the image."""
        # Create a new SummaryWindow instance specifically for OCR
        SummaryWindow(self, image_path, use_ocr=True)    
        
        
    def show_summary(self, file_path):
        SummaryWindow(self, file_path)