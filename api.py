import os
import tempfile
import pickle
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
from PIL import Image
import pytesseract
import pdfplumber
import docx
import re
import nltk

# ---------------------------
# Configurações iniciais
# ---------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# NLTK stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("portuguese"))

# Regex limpeza
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-úçâêîôûãõ\s]")

def limpar_texto(texto: str) -> str:
    texto = (texto or "").lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

# ---------------------------
# Entrada / extração de arquivo
# ---------------------------
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower().lstrip(".") in ALLOWED_EXTENSIONS

def extrair_texto_arquivo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg", ".tiff"):
            img = Image.open(filepath).convert("L")
            img.thumbnail((1024, 1024))
            return pytesseract.image_to_string(img, lang="por", config="--psm 6")
        elif ext == ".pdf":
            textos = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        textos.append(t)
            return " ".join(textos)
        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            return " ".join(p.text for p in doc.paragraphs)
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        traceback.print_exc()
        return ""
    return ""

# ---------------------------
# Carregar modelo novo
# ---------------------------
MODEL_PATH = "modelo_curriculos_otimizado.pkl"  # ajuste conforme seu novo modelo

with open(MODEL_PATH, "rb") as f:
    modelo_dict = pickle.load(f)

clf = modelo_dict["clf"]
word_vectorizer = modelo_dict["word_vectorizer"]
char_vectorizer = modelo_dict["char_vectorizer"]
selector = modelo_dict.get("selector", None)
le = modelo_dict["label_encoder"]

# ---------------------------
# Função de predição
# ---------------------------
def prever_texto(texto_raw: str):
    texto_limpo = limpar_texto(texto_raw)

    # Vetorização
    Xw = word_vectorizer.transform([texto_limpo])
    Xc = char_vectorizer.transform([texto_limpo])
    X_full = hstack([Xw, Xc])

    # Seleção de features
    if selector is not None:
        try:
            X_sel = selector.transform(X_full)
        except Exception:
            X_sel = X_full
    else:
        X_sel = X_full

    # Predição
    pred = clf.predict(X_sel)
    probs = clf.predict_proba(X_sel)[0]
    classe = le.inverse_transform(pred)[0]

    return {
        "prediction": classe,
        "confidence": float(max(probs)),
        "probabilities": dict(zip(le.classes_, map(float, probs))),
        "texto_limpo": texto_limpo
    }

# ---------------------------
# Flask app + endpoint /predict
# ---------------------------
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado (field 'file')"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"success": False, "error": "Nome de arquivo vazio"}), 400

        filename = secure_filename(uploaded.filename)
        if not allowed_file(filename):
            return jsonify({"success": False, "error": f"Tipo de arquivo não suportado: {filename}"}), 400

        with tempfile.TemporaryDirectory(prefix="cv_api_") as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            uploaded.save(filepath)

            texto_total = extrair_texto_arquivo(filepath)
            if not texto_total.strip():
                return jsonify({"success": False, "error": "Não foi possível extrair texto do arquivo."}), 400

            resultado = prever_texto(texto_total)

            response = {
                "success": True,
                "prediction": resultado["prediction"],
                "confidence": resultado["confidence"],
                "probabilities": resultado["probabilities"]
            }
            return jsonify(response)

    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Falha interna no processamento"}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API de Currículos ativa"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
