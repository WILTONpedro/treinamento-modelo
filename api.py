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
from nltk.corpus import stopwords
import nltk

# Garante que o stopwords está baixado
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ----------- Funções utilitárias -----------

def limpar_texto(texto: str) -> str:
    """Limpa texto e remove stopwords"""
    texto = texto.lower()
    texto = re.sub(r"[^a-zá-ú0-9\s]", " ", texto)
    stop = set(stopwords.words("portuguese"))
    palavras = [p for p in texto.split() if p not in stop]
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
                return " ".join([p.extract_text() or "" for p in pdf.pages])

        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            return " ".join([p.text for p in doc.paragraphs])

        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
    img = Image.open(filepath)
    img = img.convert("L")  # converte para tons de cinza
    texto = pytesseract.image_to_string(img, lang="por", config="--psm 6")
    return texto

    except Exception as e:
        print(f"[ERRO] Falha ao extrair texto de {filepath}: {e}")
        return ""

    return ""

# ----------- Flask API -----------

app = Flask(__name__)

# Extensões suportadas
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Carrega modelo e vetorizadores
with open("modelo_curriculos_avancado.pkl", "rb") as f:
    clf, word_v, char_v = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename)

        if filename == "":
            return jsonify({"error": "Nome de arquivo vazio"}), 400

        if not allowed_file(filename):
            return jsonify({"error": f"Tipo de arquivo não suportado: {filename}"}), 400

        # salva em uma pasta temporária
        tmpdir = tempfile.mkdtemp(prefix="cv_api_")
        filepath = os.path.join(tmpdir, filename)
        uploaded.save(filepath)

        # extrai texto de todos os itens (incluindo dentro de .zip)
        textos = []
        for pfile, ext in processar_item(filepath):
            txt = extrair_texto_arquivo(pfile)
            if txt:
                textos.append(limpar_texto(txt))

        # remove arquivos temporários
        shutil.rmtree(tmpdir, ignore_errors=True)

        if not textos:
            return jsonify({"error": "Não foi possível extrair texto"}), 400

        full_text = " ".join(textos)

        # vetorização
        Xw = word_v.transform([full_text])
        Xc = char_v.transform([full_text])
        Xv = hstack([Xw, Xc])

        # predição
        pred = clf.predict(Xv)[0]

        return jsonify({
            "success": True,
            "prediction": pred,
            "tokens": len(full_text.split())
        })

    except Exception as e:
        # Sempre retorna JSON mesmo em erros
        return jsonify({"error": f"Falha interna: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos rodando 🚀"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
