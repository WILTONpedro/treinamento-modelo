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

# =====================================
# 🔹 CONFIGURAÇÕES INICIAIS
# =====================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# Baixar stopwords se necessário
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("portuguese"))

# Expressões regulares para limpeza
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-ú\s]")

def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

# =====================================
# 🔹 EXTRAÇÃO DE TEXTO
# =====================================

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
        print(f"[ERRO ao extrair texto de] {filepath}\n{traceback.format_exc()}")
        return ""
    return ""


# =====================================
# 🔹 CARREGAMENTO DOS MODELOS
# =====================================

with open("modelo_curriculos_otimizado.pkl", "rb") as f:
    data_base = pickle.load(f)

with open("modelo_curriculos_xgb_oversampling.pkl", "rb") as f:
    data_over = pickle.load(f)

# Componentes do modelo base
clf_base = data_base["clf"]
word_v_base = data_base["word_vectorizer"]
char_v_base = data_base["char_vectorizer"]
selector_base = data_base["selector"]
le_base = data_base["label_encoder"]

# Componentes do modelo oversampling
clf_over = data_over["clf"]
word_v_over = data_over["word_vectorizer"]
char_v_over = data_over["char_vectorizer"]
selector_over = data_over["selector"]
le_over = data_over["label_encoder"]

# =====================================
# 🔹 PREDIÇÃO COM FUSÃO DE MODELOS
# =====================================

def prever_fusao(texto):
    # --- Modelo Base ---
    Xw_b = word_v_base.transform([texto])
    Xc_b = char_v_base.transform([texto])
    Xsel_b = selector_base.transform(hstack([Xw_b, Xc_b]))
    probs_base = clf_base.predict_proba(Xsel_b)

    # --- Modelo Oversampling ---
    Xw_o = word_v_over.transform([texto])
    Xc_o = char_v_over.transform([texto])
    Xover = hstack([Xw_o, Xc_o])

    try:
        # tenta usar selector se for compatível
        Xsel_o = selector_over.transform(Xover)
    except Exception as e:
        print(f"[AVISO] Selector oversampling incompatível, usando vetor completo. Detalhes: {e}")
        Xsel_o = Xover  # fallback

    probs_over = clf_over.predict_proba(Xsel_o)

    # --- Validação ---
    if list(le_base.classes_) != list(le_over.classes_):
        raise ValueError("As classes dos modelos são diferentes! Reentreine com o mesmo label encoder.")

    # --- Fusão de probabilidades ---
    probs_final = (probs_base + probs_over) / 2

    idx_final = probs_final.argmax()
    classe_final = le_base.inverse_transform([idx_final])[0]
    confianca = float(probs_final[0, idx_final])

    return {
        "classe_final": classe_final,
        "confianca_final": confianca,
        "pred_base": le_base.inverse_transform([probs_base.argmax()])[0],
        "conf_base": float(probs_base[0].max()),
        "pred_over": le_over.inverse_transform([probs_over.argmax()])[0],
        "conf_over": float(probs_over[0].max())
    }

# =====================================
# 🔹 FLASK CONFIG
# =====================================

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower().lstrip(".") in ALLOWED_EXTENSIONS


# =====================================
# 🔹 ENDPOINT /predict
# =====================================

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

            if len(full_text) > 30000:
                full_text = full_text[:30000]

            resultado = prever_fusao(full_text)

            return jsonify({
                "success": True,
                "tokens": len(full_text.split()),
                "prediction_final": resultado["classe_final"],
                "confidence_final": resultado["confianca_final"],
                "modelo_base_prediction": resultado["pred_base"],
                "modelo_base_conf": resultado["conf_base"],
                "modelo_oversampling_prediction": resultado["pred_over"],
                "modelo_oversampling_conf": resultado["conf_over"]
            })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Falha interna no processamento"}), 500


# =====================================
# 🔹 HEALTHCHECK
# =====================================

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos com fusão rodando 🚀"})


# =====================================
# 🔹 MAIN
# =====================================

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
