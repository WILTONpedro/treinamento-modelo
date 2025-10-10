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
import unicodedata
import nltk
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
import numpy as np

# --- Configuração NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

# --- Pesos configuráveis ---
PESO_ASSUNTO = 3
PESO_CURRICULO = 1

# --- Funções auxiliares ---
def limpar_texto(texto: str) -> str:
    texto = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    texto = texto.lower()
    texto = re.sub(r'\S+@\S+', ' ', texto)
    texto = re.sub(r'\d+', ' ', texto)
    texto = re.sub(r'http\S+|www\S+', ' ', texto)
    texto = re.sub(r'[^a-z\s]', ' ', texto)
    return " ".join([w for w in texto.split() if w not in STOPWORDS and len(w) > 2])

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
            img = Image.open(filepath).convert("L")
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
    clf, word_v, char_v, palavras_chave_dict, selector, le = data
else:
    raise ValueError("Formato do pickle desconhecido. Esperado dict ou tuple.")

# --- Extração de features ponderadas ---
def extrair_features_chave(texto, assunto=""):
    texto = texto.lower()
    assunto = assunto.lower()
    features = []
    for palavras in palavras_chave_dict.values():
        score = 0
        for p in palavras:
            if p.lower() in texto:
                score += PESO_CURRICULO
            if p.lower() in assunto:
                score += PESO_ASSUNTO
        features.append(score)
    return features

# --- Função de análise combinada ---
def analisar_curriculo(assunto="", corpo="", texto=""):
    texto_completo = f"{assunto}\n{corpo}\n\n{texto}"
    clean_text = limpar_texto(texto_completo)

    Xw = word_v.transform([clean_text])
    Xc = char_v.transform([clean_text])
    Xchaves = csr_matrix([extrair_features_chave(clean_text, assunto)])
    Xfull = hstack([Xw, Xc, Xchaves])
    Xsel = selector.transform(Xfull)

    pred = clf.predict(Xsel)[0]
    classe = le.inverse_transform([pred])[0]  # sempre dentro das classes treinadas

    try:
        probas = clf.predict_proba(Xsel)[0]
        conf = round(float(np.max(probas)) * 100, 2)
    except Exception:
        conf = None

    keywords = []
    for categoria, palavras in palavras_chave_dict.items():
        for p in palavras:
            if p.lower() in texto_completo.lower() or p.lower() in assunto.lower():
                keywords.append(p)
    keywords = list(set(keywords))

    return {
        "success": True,
        "prediction": classe,
        "confidence": conf,
        "keywords": keywords,
        "tokens": len(texto_completo.split())
    }

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

        assunto = request.form.get("assunto", "")
        corpo = request.form.get("corpo", "")

        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        full_text = ""
        for pfile, pext in processar_item(temp_path):
            txt = extrair_texto_arquivo(pfile)
            if txt:
                full_text += txt + " "

        if not full_text.strip():
            shutil.rmtree(temp_dir, ignore_errors=True)
            return jsonify({"success": False, "error": "Não foi possível extrair texto."}), 400

        resultado = analisar_curriculo(assunto, corpo, full_text)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify(resultado)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos rodando 🚀"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
