import re
import os
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
import time

# --- NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

# --- Funções auxiliares ---
def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-zá-ú\s]", " ", texto)
    palavras = [p for p in texto.split() if p not in STOPWORDS]
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

# --- Carrega modelo ---
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
    raise ValueError("Formato do pickle desconhecido.")

def extrair_features_chave(texto):
    return [int(any(p.lower() in texto for p in palavras)) for palavras in palavras_chave_dict.values()]

# --- Flask API ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}
MAX_MB = 5
MAX_BYTES = MAX_MB * 1024 * 1024
app.config['MAX_CONTENT_LENGTH'] = MAX_BYTES

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"success": False, "error": "Nome de arquivo vazio"}), 400

        if uploaded.content_length > MAX_BYTES:
            return jsonify({"success": False, "error": f"Arquivo maior que {MAX_MB} MB"}), 400

        filename = secure_filename(uploaded.filename)
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"success": False, "error": f"Tipo de arquivo não suportado: {uploaded.filename}"}), 400

        tmpdir = tempfile.mkdtemp(prefix="cv_api_")
        filepath = os.path.join(tmpdir, filename)
        uploaded.save(filepath)

        textos = []
        for pfile, pext in processar_item(filepath):
            txt = extrair_texto_arquivo(pfile)
            if txt:
                textos.append(limpar_texto(txt))

        shutil.rmtree(tmpdir, ignore_errors=True)

        if not textos:
            return jsonify({"success": False, "error": "Não foi possível extrair texto"}), 400

        full_text = " ".join(textos)

        Xw = word_v.transform([full_text])
        Xc = char_v.transform([full_text])
        Xchaves = csr_matrix([extrair_features_chave(full_text)])
        Xfull = hstack([Xw, Xc, Xchaves])
        Xsel = selector.transform(Xfull)

        pred = clf.predict(Xsel)[0]
        classe = le.inverse_transform([pred])[0]

        tempo = round(time.time() - start_time, 2)
        return jsonify({"success": True, "prediction": classe, "tokens": len(full_text.split()), "processing_time": tempo})

    except Exception as e:
        return jsonify({"success": False, "error": f"Falha interna: {str(e)}"}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos rodando 🚀"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
