import os
import re
import tempfile
import shutil
import zipfile
import pickle
import docx
import pdfplumber
import pytesseract
from PIL import Image
import nltk
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
import logging

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

# --- Funções auxiliares ---
def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"\S+@\S+", " ", texto)  # emails
    texto = re.sub(r"\d+", " ", texto)      # números
    texto = re.sub(r"[^a-zá-ú\s]", " ", texto)
    palavras = [p for p in texto.split() if p not in STOPWORDS]
    return " ".join(palavras)

def processar_item(filepath):
    """Processa arquivos, inclusive dentro de ZIP"""
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
    """Extrai texto de diversos formatos"""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                return " ".join([p.extract_text() or "" for p in pdf.pages if p.extract_text()])
        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            img = Image.open(filepath).convert("L")
            return pytesseract.image_to_string(img, lang="por", config="--psm 6")
    except Exception as e:
        logging.error(f"Erro ao extrair texto de {filepath}: {e}")
        return ""
    return ""

# --- Carregar modelo ---
with open("modelo_curriculos_xgb_oversampling.pkl", "rb") as f:
    data = pickle.load(f)

clf = data["clf"]
word_v = data["word_vectorizer"]
char_v = data["char_vectorizer"]
palavras_chave_dict = data["palavras_chave_dict"]
selector = data["selector"]
le = data["label_encoder"]

# --- Features palavras-chave ---
def extrair_features_chave(texto):
    encontradas = []
    features = []
    for area, palavras in palavras_chave_dict.items():
        hit = any(p.lower() in texto for p in palavras)
        features.append(int(hit))
        if hit:
            encontradas.extend([p for p in palavras if p.lower() in texto])
    return features, list(set(encontradas))

# --- Flask API ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"success": False, "error": "Nome de arquivo vazio"}), 400

        filename = secure_filename(uploaded.filename)
        tmpdir = tempfile.mkdtemp(prefix="cv_api_")
        filepath = os.path.join(tmpdir, filename)
        uploaded.save(filepath)

        textos = []
        for pfile, _ in processar_item(filepath):
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
        Xchaves, palavras_encontradas = extrair_features_chave(full_text)
        Xchaves = csr_matrix([Xchaves])
        Xfull = hstack([Xw, Xc, Xchaves])
        Xsel = selector.transform(Xfull)

        # Predição
        pred = clf.predict(Xsel)[0]
        probas = clf.predict_proba(Xsel)[0]
        classe = le.inverse_transform([pred])[0]
        confianca = float(probas[pred])

        return jsonify({
            "success": True,
            "prediction": classe,
            "confidence": round(confianca, 3),
            "tokens": len(full_text.split()),
            "keywords_found": palavras_encontradas
        })

    except Exception as e:
        logging.exception("Erro interno")
        return jsonify({"success": False, "error": f"Falha interna: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos rodando 🚀"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
