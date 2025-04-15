from flask import Flask, request, jsonify
import os
from sentence_transformers import SentenceTransformer
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
from PIL import Image
import base64
import io
import logging
import pytesseract
from pdf2image import convert_from_bytes
import torch

app = Flask(__name__)

# Configure logging to show all requests and errors
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('api_server.log')  # Save logs to a file for debugging
    ]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"使用设备: {device}")

# Configure Tesseract path (update to your actual path if different)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Model path (update to your actual path)
MODEL_DIR = r"X:\projects\Ollama\models"
try:
    text_model = SentenceTransformer(os.path.join(MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2"))
    clip_model = ChineseCLIPModel.from_pretrained(os.path.join(MODEL_DIR, "chinese-clip-vit-base-patch16"))
    clip_processor = ChineseCLIPProcessor.from_pretrained(os.path.join(MODEL_DIR, "chinese-clip-vit-base-patch16"))
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Failed to load models: {str(e)}")
    raise

@app.route('/embed_text', methods=['POST'])
def embed_text():
    logging.debug("Received request for /embed_text")
    try:
        data = request.json
        if not data or 'text' not in data:
            logging.error("Missing 'text' field in request")
            return jsonify({'error': "Missing 'text' field"}), 400
        text = data['text']
        embedding = text_model.encode(text).tolist()
        logging.info(f"Generated embedding for text: {text[:50]}...")
        return jsonify({'embedding': embedding})
    except Exception as e:
        logging.error(f"Error embedding text: {str(e)}")
        return jsonify({'error': f"Failed to embed text: {str(e)}"}), 500

@app.route('/embed_image', methods=['POST'])
def embed_image():
    logging.debug("Received request for /embed_image")
    try:
        data = request.json
        if not data or 'image' not in data:
            logging.error("Missing 'image' field in request")
            return jsonify({'error': "Missing 'image' field"}), 400
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        inputs = clip_processor(images=image, return_tensors="pt")
        embedding = clip_model.get_image_features(**inputs).detach().numpy().tolist()
        logging.info("Generated embedding for image")
        return jsonify({'embedding': embedding})
    except Exception as e:
        logging.error(f"Error embedding image: {str(e)}")
        return jsonify({'error': f"Failed to embed image: {str(e)}"}), 500

@app.route('/embed_clip_text', methods=['POST'])
def embed_clip_text():
    logging.debug("Received request for /embed_clip_text")
    try:
        data = request.json
        if not data or 'text' not in data:
            logging.error("Missing 'text' field in request")
            return jsonify({'error': "Missing 'text' field"}), 400
        text = data['text']
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
        embedding = clip_model.get_text_features(**inputs).detach().numpy().tolist()
        logging.info(f"Generated CLIP embedding for text: {text[:50]}...")
        return jsonify({'embedding': embedding})
    except Exception as e:
        logging.error(f"Error embedding CLIP text: {str(e)}")
        return jsonify({'error': f"Failed to embed CLIP text: {str(e)}"}), 500

@app.route('/extract_pdf_with_ocr', methods=['POST'])
def extract_pdf_with_ocr():
    logging.debug("Received request for /extract_pdf_with_ocr")
    try:
        data = request.json
        if not data or 'pdf' not in data:
            logging.error("Missing 'pdf' field in request")
            return jsonify({'error': "Missing 'pdf' field"}), 400
        pdf_data = base64.b64decode(data['pdf'])
        pdf_file = io.BytesIO(pdf_data)
        images = convert_from_bytes(pdf_file.read())
        ocr_text = ""
        for img in images:
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            ocr_text += text + "\n"
        embedding = text_model.encode(ocr_text if ocr_text.strip() else "No text detected").tolist()
        logging.info("Extracted OCR text from PDF")
        return jsonify({'text': ocr_text, 'embedding': embedding})
    except Exception as e:
        logging.error(f"Error extracting PDF OCR: {str(e)}")
        return jsonify({'error': f"Failed to extract PDF OCR: {str(e)}"}), 500

@app.route('/extract_image_ocr', methods=['POST'])
def extract_image_ocr():
    logging.debug("Received request for /extract_image_ocr")
    try:
        data = request.json
        if not data or 'image' not in data:
            logging.error("Missing 'image' field in request")
            return jsonify({'error': "Missing 'image' field"}), 400
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        embedding = text_model.encode(ocr_text if ocr_text.strip() else "No text detected").tolist()
        logging.info("Extracted OCR text from image")
        return jsonify({'text': ocr_text, 'embedding': embedding})
    except Exception as e:
        logging.error(f"Error extracting image OCR: {str(e)}")
        return jsonify({'error': f"Failed to extract image OCR: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)