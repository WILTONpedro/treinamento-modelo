import os
import tempfile
import zipfile
import pickle
import traceback
import re
import nltk
import pdfplumber
import pytesseract
import docx
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
from nltk.corpus import stopwords

# --- Configura Tesseract ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# --- Preparar NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("portuguese"))

# --- Expressões regulares para limpeza ---
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-ú\s]")

def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

# --- Função para processar arquivos ZIP e individuais ---
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

# --- Função para extrair texto de vários formatos ---
def extrair_texto_arquivo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg", ".tiff"):
            if os.path.getsize(filepath) > 5 * 1024 * 1024:
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

# ============================
# 🔹 CARREGAR OS DOIS MODELOS
# ============================

with open("modelo_curriculos_otimizado.pkl", "rb") as f:
    data_base = pickle.load(f)

with open("modelo_curriculos_xgb_oversampling.pkl", "rb") as f:
    data_over = pickle.load(f)

clf_base = data_base["clf"]
clf_over = data_over["clf"]

word_v = data_base["word_vectorizer"]
char_v = data_base["char_vectorizer"]
selector = data_base["selector"]
le = data_base["label_encoder"]

# --- Função para prever com fusão ---
def prever_fusao(texto):
    Xw = word_v.transform([texto])
    Xc = char_v.transform([texto])
    Xfull = hstack([Xw, Xc])
    Xsel = selector.transform(Xfull)

    # Probabilidades individuais
    probs_base = clf_base.predict_proba(Xsel)
    probs_over = clf_over.predict_proba(Xsel)

    # Média das probabilidades
    probs_final = (probs_base + probs_over) / 2

    # Classe final
    idx_final = probs_final.argmax()
    classe_final = le.inverse_transform([idx_final])[0]
    confianca = float(probs_final[0, idx_final])

    # Também retorna previsões individuais
    pred_base = le.inverse_transform([probs_base.argmax()])[0]
    pred_over = le.inverse_transform([probs_over.argmax()])[0]

    return {
        "classe_final": classe_final,
        "confianca_final": confianca,
        "pred_base": pred_base,
        "conf_base": float(probs_base[0].max()),
        "pred_over": pred_over,
        "conf_over": float(probs_over[0].max())
    }

# ============================
# 🔹 CONFIGURAR FLASK
# ============================
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower().lstrip(".") in ALLOWED_EXTENSIONS

# ============================
# 🔹 ENDPOINT /predict
# ============================
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

            full_text = " ".join(textos)
            if not full_text.strip():
                return jsonify({"error": "Não foi possível extrair texto"}), 400

            # Limite de tamanho
            if len(full_text) > 30000:
                full_text = full_text[:30000]

            # --- Predição combinada ---
            resultado = prever_fusao(full_text)

            resp = {
                "success": True,
                "tokens": len(full_text.split()),
                "prediction_final": resultado["classe_final"],
                "confidence_final": resultado["confianca_final"],
                "modelo_base_prediction": resultado["pred_base"],
                "modelo_base_conf": resultado["conf_base"],
                "modelo_oversampling_prediction": resultado["pred_over"],
                "modelo_oversampling_conf": resultado["conf_over"]
            }

            return jsonify(resp)

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Falha interna no processamento"}), 500

# ============================
# 🔹 HEALTHCHECK
# ============================
@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos com fusão rodando 🚀"})

# ============================
# 🔹 RUN
# ============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
