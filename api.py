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

# --- Preparar NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("portuguese"))

# --- Funções auxiliares ---
def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"\S+@\S+", " ", texto)  # remove emails
    texto = re.sub(r"\d+", " ", texto)      # remove números
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
    """Extrai texto de PDF, DOCX, TXT e imagens"""
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
            img = Image.open(filepath)
            img = img.convert("L")
            return pytesseract.image_to_string(img, lang="por", config="--psm 6")
    except Exception as e:
        print(f"[ERRO] Falha ao extrair texto de {filepath}: {e}")
        return ""
    return ""

# --- Carrega modelo (dicionário ou tupla) ---
with open("modelo_curriculos_xgb_oversampling.pkl", "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict):
    clf = data["clf"]
    word_v = data["word_vectorizer"]
    char_v = data["char_vectorizer"]
    palavras_chave_dict = data["palavras_chave_dict"]
    selector = data["selector"]
    le = data["label_encoder"]
elif isinstance(data, (tuple, list)):
    clf = data[0]
    word_v = data[1]
    char_v = data[2]
    palavras_chave_dict = data[3]
    selector = data[4]
    le = data[5]
else:
    raise ValueError("Formato do pickle desconhecido. Esperado dict ou tuple.")

# --- Extrair features de palavras-chave ---
def extrair_features_chave(texto):
    return [int(any(p.lower() in texto for p in palavras)) for palavras in palavras_chave_dict.values()]

# --- Flask API ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado."}), 400

        # --- Extrair texto ---
        filename = secure_filename(file.filename)
        ext = filename.split(".")[-1].lower()
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(temp_path)

        full_text = ""

        if ext in ["pdf"]:
            with pdfplumber.open(temp_path) as pdf:
                for page in pdf.pages:
                    full_text += page.extract_text() or ""
        elif ext in ["docx"]:
            doc = docx.Document(temp_path)
            full_text = " ".join([p.text for p in doc.paragraphs])
        elif ext in ["jpg", "jpeg", "png"]:
            img = Image.open(temp_path)
            full_text = pytesseract.image_to_string(img)
        else:
            return jsonify({"success": False, "error": f"Extensão não suportada: {ext}"}), 400

        os.remove(temp_path)

        if not full_text.strip():
            return jsonify({"success": False, "error": "Texto vazio extraído."}), 400

        # --- Pré-processar texto ---
        clean = re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", full_text.lower())
        clean = " ".join([w for w in clean.split() if w not in stopwords_port])

        # --- Vetorização ---
        Xvec = vectorizer.transform([clean])
        Xsel = selector.transform(Xvec)

        # --- Predição ---
        pred = clf.predict(Xsel)[0]
        classe = le.inverse_transform([pred])[0]

        # 🔹 Calcular probabilidade da classe
        if hasattr(clf, "predict_proba"):
            probas = clf.predict_proba(Xsel)[0]
            conf = round(float(probas[pred]) * 100, 2)
        else:
            conf = None

        # 🔹 Extrair palavras-chave mais relevantes
        palavras_encontradas = []
        for categoria, palavras in palavras_chave_dict.items():
            for p in palavras:
                if p.lower() in full_text.lower():
                    palavras_encontradas.append(p)

        # 🔹 Retorno final para o Apps Script
        return jsonify({
            "success": True,
            "prediction": classe,
            "confidence": conf,
            "keywords": list(set(palavras_encontradas)),
            "tokens": len(full_text.split())
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos rodando 🚀"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
