import os
import re
import tempfile
import zipfile
import pickle
import traceback
import base64
import gzip

import docx
import pdfplumber
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack, csr_matrix
import nltk
from nltk.corpus import stopwords

# Configura tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# --- Preparar stopwords ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("portuguese"))

# --- Regex pré-compiladas ---
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-ú\s]")

# --- Função de limpeza ---
def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

# --- Processamento de arquivos ---
def processar_item(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".zip":
        with zipfile.ZipFile(filepath, "r") as z:
            with tempfile.TemporaryDirectory() as tmpdir:
                z.extractall(tmpdir)
                for name in z.namelist():
                    yield os.path.join(tmpdir, name)
    else:
        yield filepath

# --- Extração de texto ---
def extrair_texto_arquivo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg", ".tiff"):
            if os.path.getsize(filepath) > 5 * 1024 * 1024:  # >5MB
                return ""
            img = Image.open(filepath).convert("L")
            img.thumbnail((1024, 1024))
            return pytesseract.image_to_string(img, lang="por", config="--psm 6")
        elif ext == ".pdf":
            texts = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        texts.append(t)
            return " ".join(texts)
        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            return " ".join(p.text for p in doc.paragraphs)
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        print(f"[ERRO] {filepath}\n{traceback.format_exc()}")
        return ""
    return ""

# --- Carrega modelo ---
with open("modelo_curriculos_xgb_oversampling.pkl", "rb") as f:
    data = pickle.load(f)

clf = data["clf"]
word_v = data["word_vectorizer"]
char_v = data["char_vectorizer"]
palavras_chave_dict = data["palavras_chave_dict"]
selector = data["selector"]
le = data["label_encoder"]

# --- Features de palavras-chave ---
def extrair_features_chave(texto):
    return [int(any(p.lower() in texto for p in palavras)) for palavras in palavras_chave_dict.values()]

# --- API Flask ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower().lstrip(".") in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"error": "Nome de arquivo vazio"}), 400

        filename = secure_filename(uploaded.filename)
        if not allowed_file(filename):
            return jsonify({"error": f"Tipo de arquivo não suportado: {filename}"}), 400

        with tempfile.TemporaryDirectory(prefix="cv_api_") as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            uploaded.save(filepath)

            textos = []
            for pfile in processar_item(filepath):
                txt = extrair_texto_arquivo(pfile)
                if txt:
                    textos.append(limpar_texto(txt))

            if not textos:
                return jsonify({"error": "Não foi possível extrair texto"}), 400

            full_text = " ".join(textos)
            if len(full_text) > 30000:
                full_text = full_text[:30000]

            compressed_text = None
            if len(full_text.encode("utf-8")) > 20000:
                compressed_text = base64.b64encode(gzip.compress(full_text.encode("utf-8"))).decode("utf-8")

            Xw = word_v.transform([full_text])
            Xc = char_v.transform([full_text])
            Xchaves = csr_matrix([extrair_features_chave(full_text)])
            Xfull = hstack([Xw, Xc, Xchaves])
            Xsel = selector.transform(Xfull)

            pred = clf.predict(Xsel)[0]
            classe = le.inverse_transform([pred])[0]

            resp = {
                "success": True,
                "prediction": classe,
                "tokens": len(full_text.split())
            }
            if compressed_text:
                resp["compressed_text"] = compressed_text[:200] + "... (compactado)"

            return jsonify(resp)
    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Falha interna no processamento"}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos rodando 🚀"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
