import os
import io
import json
import time
import traceback
import tempfile
from collections import defaultdict

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

import pickle
import numpy as np

from PIL import Image, ImageEnhance
import pytesseract
import pdfplumber
import docx

# CONFIG
MODEL_PKL = "model.pkl"
TFIDF_PKL = "tfidf.pkl"
CLASSES_JSON = "classes.json"
LOG_FILE = "logs_api.jsonl"
ALLOWED_EXT = {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".tiff"}
PORT = int(os.environ.get("PORT", 5000))

# Tesseract path (Windows / Linux)
TESSERACT_WIN = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_LINUX = "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_LINUX if os.name != "nt" else TESSERACT_WIN

# Flask app
app = Flask(__name__)

# -------------------------
# Helpers: logs
# -------------------------
def save_log(entry: dict):
    try:
        entry["ts"] = int(time.time())
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        traceback.print_exc()

# -------------------------
# Load artifacts
# -------------------------
if not os.path.exists(MODEL_PKL) or not os.path.exists(TFIDF_PKL) or not os.path.exists(CLASSES_JSON):
    raise FileNotFoundError("Artefatos (model.pkl, tfidf.pkl, classes.json) não encontrados no diretório atual.")

with open(MODEL_PKL, "rb") as f:
    MODEL = pickle.load(f)

with open(TFIDF_PKL, "rb") as f:
    VECT = pickle.load(f)

with open(CLASSES_JSON, "r", encoding="utf-8") as f:
    CLASS_LIST = json.load(f)

# model: espera-se sklearn estimator com predict/prob
MODEL_NAME = getattr(MODEL, "__class__", type(MODEL)).__name__

# -------------------------
# Text extraction utils
# -------------------------
def melhorar_imagem_para_ocr(img: Image.Image) -> Image.Image:
    try:
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img); img = enhancer.enhance(1.8)
        enhancer = ImageEnhance.Sharpness(img); img = enhancer.enhance(1.3)
    except Exception:
        pass
    return img

def extract_text_from_image_path(path: str) -> str:
    try:
        img = Image.open(path)
        img = melhorar_imagem_para_ocr(img)
        text = pytesseract.image_to_string(img, lang="por", config="--psm 6")
        img.close()
        return text or ""
    except Exception:
        traceback.print_exc()
        return ""

def extract_text_from_pdf_path(path: str) -> str:
    # Try digital text first
    try:
        with pdfplumber.open(path) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            text = "\n".join(pages).strip()
            if text:
                return text
    except Exception:
        # fail silently to try OCR fallback below
        traceback.print_exc()

    # Fallback: OCR with pdf2image if available
    try:
        from pdf2image import convert_from_path
    except Exception:
        convert_from_path = None

    if convert_from_path:
        try:
            images = convert_from_path(path, dpi=200, first_page=1, last_page=2)
            parts = []
            for im in images:
                im = melhorar_imagem_para_ocr(im)
                parts.append(pytesseract.image_to_string(im, lang="por"))
            return "\n".join(parts).strip()
        except Exception:
            traceback.print_exc()
            return ""
    else:
        return ""  # no fallback lib installed

def extract_text_from_docx_path(path: str) -> str:
    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        traceback.print_exc()
        return ""

def extract_text_from_txt_path(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        traceback.print_exc()
        return ""

def extract_text_from_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".tiff"}:
        return extract_text_from_image_path(path)
    if ext == ".pdf":
        return extract_text_from_pdf_path(path)
    if ext in {".docx", ".doc"}:
        return extract_text_from_docx_path(path)
    if ext == ".txt":
        return extract_text_from_txt_path(path)
    return ""

# -------------------------
# Text cleaning (light)
# -------------------------
import re
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-úçâêîôûãõ\s\w]")

def clean_text(t: str) -> str:
    if not t:
        return ""
    s = t.lower()
    s = RE_EMAIL.sub(" ", s)
    s = RE_NUM.sub(" ", s)
    s = RE_CARACT.sub(" ", s)
    s = " ".join(s.split())
    return s

# -------------------------
# Prediction routine
# -------------------------
def predict_text_return(text: str):
    txt_clean = clean_text(text)
    if not txt_clean.strip() or len(txt_clean.split()) < 5:
        return None
    X = VECT.transform([txt_clean])
    # model must support predict_proba
    try:
        probs = MODEL.predict_proba(X)[0]
        # ensure alignment: classes order from model or classes.json
        # sklearn models provide classes_ corresponding to columns of predict_proba
        if hasattr(MODEL, "classes_"):
            model_classes = [str(c) for c in MODEL.classes_]
            # if model_classes are indices or numbers and CLASS_LIST are strings, try to map
            # we will produce prob dict using CLASS_LIST if lengths match, otherwise model_classes
            if len(model_classes) == len(probs) and len(CLASS_LIST) == len(probs):
                probs_map = {cls: float(p) for cls, p in zip(CLASS_LIST, probs)}
            else:
                probs_map = {cls: float(p) for cls, p in zip(model_classes, probs)}
        else:
            probs_map = {f"class_{i}": float(p) for i, p in enumerate(probs)}
    except Exception as e:
        # fallback: predict only
        try:
            pred = MODEL.predict(X)[0]
            return {"prediction": str(pred), "confidence": None, "probs": {}}
        except Exception:
            raise

    # top
    top_idx = int(np.argmax(probs))
    try:
        if len(CLASS_LIST) == len(probs):
            pred_label = CLASS_LIST[top_idx]
        else:
            # try model.classes_
            pred_label = str(MODEL.classes_[top_idx]) if hasattr(MODEL, "classes_") else list(probs_map.keys())[top_idx]
    except Exception:
        pred_label = list(probs_map.keys())[top_idx]

    top_prob = float(probs[top_idx])
    return {"prediction": pred_label, "confidence": top_prob, "probs": probs_map}

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def health():
    info = {
        "status": "ok",
        "model": MODEL_NAME,
        "n_classes": len(CLASS_LIST)
    }
    return jsonify(info)

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "campo 'file' ausente"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"success": False, "error": "arquivo sem nome"}), 400

        filename = secure_filename(uploaded.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return jsonify({"success": False, "error": f"extensão não permitida: {ext}"}), 400

        # optional fields
        subject = request.form.get("subject") or request.form.get("assunto") or ""
        body = request.form.get("body") or request.form.get("corpo") or ""

        with tempfile.TemporaryDirectory(prefix="cv_api_") as tmp:
            tmp_path = os.path.join(tmp, filename)
            uploaded.save(tmp_path)

            # extract text
            text = extract_text_from_file(tmp_path)
            combined = " ".join([subject, body, text]).strip()

            # validate
            if not combined or len(combined.split()) < 5:
                save_log({
                    "status": "rejected_insufficient",
                    "filename": filename,
                    "extracted_len": len(text),
                    "extraction": "ok" if text else "empty"
                })
                return jsonify({"success": False, "error": "conteúdo insuficiente para classificar", "extracted_length": len(text)}), 422

            pred = predict_text_return(combined)
            if pred is None:
                save_log({
                    "status": "rejected_cleaning",
                    "filename": filename,
                    "extracted_len": len(text)
                })
                return jsonify({"success": False, "error": "não foi possível extrair conteúdo válido"}), 422

            # build response
            resp = {
                "success": True,
                "prediction": pred["prediction"],
                "confidence": pred["confidence"],
                "probs": pred["probs"]
            }

            # save log
            save_log({
                "status": "classified",
                "filename": filename,
                "prediction": pred["prediction"],
                "confidence": pred["confidence"],
                "extracted_len": len(text)
            })

            return jsonify(resp)

    except Exception as e:
        traceback.print_exc()
        save_log({"status": "error", "error": str(e)})
        return jsonify({"success": False, "error": "erro interno", "detail": str(e)}), 500

@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        if not os.path.exists(LOG_FILE):
            return jsonify({"logs": []})
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f.readlines() if l.strip()]
        return jsonify({"logs": lines[-200:]})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Run
if __name__ == "__main__":
    # threaded for handling multiple requests concurrently (simple)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
