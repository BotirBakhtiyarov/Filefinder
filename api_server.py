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

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# Update this to your actual model directory
MODEL_DIR = r"X:\projects\Ollama\models"  # Replace with actual path
text_model = SentenceTransformer(os.path.join(MODEL_DIR, "paraphrase-multilingual-MiniLM-L12-v2"))
clip_model = ChineseCLIPModel.from_pretrained(os.path.join(MODEL_DIR, "chinese-clip-vit-base-patch16"))
clip_processor = ChineseCLIPProcessor.from_pretrained(os.path.join(MODEL_DIR, "chinese-clip-vit-base-patch16"))

@app.route('/embed_text', methods=['POST'])
def embed_text():
    logging.info("Received embed_text request")
    try:
        data = request.json
        text = data['text']
        embedding = text_model.encode(text).tolist()
        return jsonify({'embedding': embedding})
    except Exception as e:
        logging.error(f"Embed Text Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/embed_image', methods=['POST'])
def embed_image():
    logging.info("Received embed_image request")
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        inputs = clip_processor(images=image, return_tensors="pt")
        embedding = clip_model.get_image_features(**inputs).detach().numpy().tolist()
        return jsonify({'embedding': embedding})
    except Exception as e:
        logging.error(f"Embed Image Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/embed_clip_text', methods=['POST'])
def embed_clip_text():
    logging.info("Received embed_clip_text request")
    try:
        data = request.json
        text = data['text']
        inputs = clip_processor(text=[text], return_tensors="pt", padding=True)
        embedding = clip_model.get_text_features(**inputs).detach().numpy().tolist()
        return jsonify({'embedding': embedding})
    except Exception as e:
        logging.error(f"Embed Clip Text Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/extract_pdf_with_ocr', methods=['POST'])
def extract_pdf_with_ocr():
    logging.info("Received extract_pdf_with_ocr request")
    try:
        data = request.json
        pdf_data = base64.b64decode(data['pdf'])  # Expecting base64-encoded PDF
        pdf_file = io.BytesIO(pdf_data)

        # Convert PDF to images
        images = convert_from_bytes(pdf_file.read())

        # Perform OCR on each image
        ocr_text = ""
        for img in images:
            text = pytesseract.image_to_string(img, lang='chi_sim+eng')
            ocr_text += text + "\n"

        # Optionally, you can embed the text here or return it for summarization
        embedding = text_model.encode(ocr_text).tolist()
        return jsonify({'text': ocr_text, 'embedding': embedding})
    except Exception as e:
        logging.error(f"Extract PDF with OCR Error: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/extract_image_ocr', methods=['POST'])
def extract_image_ocr():
    logging.info("Received extract_image_ocr request")
    try:
        data = request.json
        image_data = base64.b64decode(data['image'])  # Expecting base64-encoded image
        image = Image.open(io.BytesIO(image_data))

        # Perform OCR on the image
        ocr_text = pytesseract.image_to_string(image, lang='chi_sim+eng')

        # Optionally, embed the text
        embedding = text_model.encode(ocr_text if ocr_text.strip() else "No text detected").tolist()
        return jsonify({'text': ocr_text, 'embedding': embedding})
    except Exception as e:
        logging.error(f"Extract Image OCR Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)