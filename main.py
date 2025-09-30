import os
import tempfile
import shutil
import zipfile
import pickle
import re
import docx
import pdfplumber
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords

# Baixa stopwords se não tiver
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# Configuração Flask
app = Flask(__name__)

# Extensões suportadas
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'jpg', 'jpeg', 'png'}

# Carrega modelo
with open("modelo_curriculos_avancado.pkl", "rb") as f:
    clf, word_v, char_v = pickle.load(f)

# Funções utilitárias
def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"[^a-zá-ú0-9\s]", " ", texto)
    stop = set(stopwords.words("portuguese"))
    palavras = [p for p in texto.split() if p not in stop]
    return " ".join(palavras)

def processar_item(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".zip":
        with zipfile.ZipFile(filepath, "r") as z:
            for name in z.namelist():
                tmp_path = os.path.join(tempfile.mkdtemp(), name)
                z.extract(name, os.path.dirname(tmp_path))
                yield tmp_path, os.path.splitext(name)[1].lower()
    else:
        yield filepath, ext

def extrair_texto_arquivo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        with pdfplumber.open(filepath) as pdf:
            return " ".join([p.extract_text() or "" for p in pdf.pages])
    elif ext in (".docx", ".doc"):
        doc = docx.Document(filepath)
        return " ".join([p.text for p in doc.paragraphs])
    elif ext == ".txt":
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            return f.read()
    elif ext in (".jpg", ".jpeg", ".png"):
        return pytesseract.image_to_string(Image.open(filepath), lang="por")
    return ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# ENDPOINTS
# =========================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename)

        if filename == "":
            return jsonify({"success": False, "error": "Nome de arquivo vazio"}), 400

        if not allowed_file(filename):
            return jsonify({"success": False, "error": "Tipo de arquivo não suportado"}), 400

        # Pasta temporária
        tmpdir = tempfile.mkdtemp(prefix="cv_api_")
        filepath = os.path.join(tmpdir, filename)
        uploaded.save(filepath)

        # Extrai texto
        textos = []
        for pfile, ext in processar_item(filepath):
            txt = extrair_texto_arquivo(pfile)
            if txt:
                textos.append(limpar_texto(txt))

        shutil.rmtree(tmpdir, ignore_errors=True)

        if not textos:
            return jsonify({"success": False, "error": "Não foi possível extrair texto"}), 400

        full_text = " ".join(textos)

        # Vetorização
        Xw = word_v.transform([full_text])
        Xc = char_v.transform([full_text])
        Xv = hstack([Xw, Xc])

        # Predição
        pred = clf.predict(Xv)[0]
        return jsonify({"success": True, "prediction": pred}), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# =========================
# RUN LOCAL
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
