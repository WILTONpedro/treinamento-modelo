import os
import re
import tempfile
import shutil
import zipfile
import pickle
import logging
import subprocess

import docx
import pdfplumber
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords

# OCR
from PIL import Image
import pytesseract

# --------------------------------------------------
# Configuração de logs
# --------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_api")

# --------------------------------------------------
# Download de stopwords se não existirem
# --------------------------------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("portuguese"))

# --------------------------------------------------
# Inicialização do Flask
# --------------------------------------------------
app = Flask(__name__)

# Extensões suportadas
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'jpg', 'jpeg', 'png'}

# --------------------------------------------------
# Funções auxiliares
# --------------------------------------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def limpar_texto(texto: str) -> str:
    """Limpa o texto para o modelo (lowercase + remove símbolos + remove stopwords)."""
    texto = texto.lower()
    texto = re.sub(r"[^a-zá-ú0-9\s]", " ", texto)
    palavras = [p for p in texto.split() if p not in STOPWORDS]
    return " ".join(palavras)


def processar_item(filepath):
    """Descompacta ZIP ou retorna o arquivo único."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".zip":
        with zipfile.ZipFile(filepath, "r") as z:
            for name in z.namelist():
                tmp_path = os.path.join(tempfile.mkdtemp(), name)
                z.extract(name, os.path.dirname(tmp_path))
                yield tmp_path, os.path.splitext(name)[1].lower()
    else:
        yield filepath, ext


def extrair_doc(filepath):
    """Extrai texto de arquivos .doc usando antiword."""
    try:
        output = subprocess.check_output(["antiword", filepath])
        return output.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Erro ao extrair .doc com antiword: {e}")
        return ""


def extrair_texto_arquivo(filepath):
    """Extrai texto de diferentes formatos suportados (PDF, DOCX, TXT, JPG, PNG, DOC)."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                return " ".join([p.extract_text() or "" for p in pdf.pages])
        elif ext == ".docx":
            doc = docx.Document(filepath)
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == ".doc":
            return extrair_doc(filepath)
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext in (".jpg", ".jpeg", ".png"):
            img = Image.open(filepath)
            texto = pytesseract.image_to_string(img, lang="por")
            return texto
    except Exception as e:
        logger.error(f"Erro ao extrair texto de {filepath}: {e}")
    return ""

# --------------------------------------------------
# Carrega modelo treinado
# --------------------------------------------------
MODEL_PATH = "modelo_curriculos_avancado.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    clf, word_v, char_v = pickle.load(f)

logger.info("✅ Modelo carregado com sucesso.")

# --------------------------------------------------
# Rotas da API
# --------------------------------------------------
@app.route("/ping", methods=["GET"])
def ping():
    """Healthcheck endpoint"""
    return jsonify({"status": "pong"})


@app.route("/predict", methods=["POST"])
def predict():
    """Recebe arquivo, extrai texto, limpa, vetoriza e retorna a predição."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename)

        if filename == "":
            return jsonify({"error": "Nome de arquivo vazio"}), 400

        if not allowed_file(filename):
            return jsonify({"error": "Tipo de arquivo não suportado"}), 400

        # salva em uma pasta temporária
        tmpdir = tempfile.mkdtemp(prefix="cv_api_")
        filepath = os.path.join(tmpdir, filename)
        uploaded.save(filepath)

        textos = []
        for pfile, ext in processar_item(filepath):
            txt = extrair_texto_arquivo(pfile)
            if txt:
                textos.append(limpar_texto(txt))

        shutil.rmtree(tmpdir, ignore_errors=True)

        if not textos:
            return jsonify({"error": "Não foi possível extrair texto"}), 400

        full_text = " ".join(textos)

        # vetorização
        Xw = word_v.transform([full_text])
        Xc = char_v.transform([full_text])
        Xv = hstack([Xw, Xc])

        pred = clf.predict(Xv)[0]
        logger.info(f"Arquivo '{filename}' classificado como: {pred}")

        return jsonify({"prediction": pred})

    except Exception as e:
        logger.exception("Erro inesperado no /predict")
        return jsonify({"error": str(e)}), 500

# --------------------------------------------------
# Inicialização local (debug)
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
