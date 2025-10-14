import os
import tempfile
import zipfile
import pickle
import traceback

import docx
import pdfplumber
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords

# --- Configura Tesseract ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

# --- Preparar NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

# --- Função de limpeza ---
import re
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-ú\s]")

def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

# --- Processamento de arquivos ---
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

# --- Extração de texto ---
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

# --- Carregar dois modelos para ensemble ---

def carregar_modelo_dict(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

model_opt = carregar_modelo_dict("modelo_curriculos_otimizado.pkl")
model_over = carregar_modelo_dict("modelo_curriculos_xgb_oversampling.pkl")

clf_opt = model_opt["clf"]
clf_over = model_over["clf"]
word_v_opt = model_opt["word_vectorizer"]
char_v_opt = model_opt["char_vectorizer"]
selector_opt = model_opt["selector"]
le_opt = model_opt["label_encoder"]

word_v_ov = model_over["word_vectorizer"]
char_v_ov = model_over["char_vectorizer"]
selector_ov = model_over["selector"]
le_ov = model_over["label_encoder"]

# Verificar que os label_encoders são compatíveis
if list(le_opt.classes_) != list(le_ov.classes_):
    raise RuntimeError("Classes dos modelos diferentes — não é possível fazer ensemble.")

# --- Função para transformar texto para cada modelo ---
def transformar_para_modelo(model_dict, word_v, char_v, selector, texto_limpo):
    # Extrai features
    Xw = word_v.transform([texto_limpo])
    Xc = char_v.transform([texto_limpo])
    Xfull = hstack([Xw, Xc])
    Xsel = selector.transform(Xfull)
    return Xsel

# --- Configurar Flask ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower().lstrip(".") in ALLOWED_EXTENSIONS

# --- Endpoint de predição com ensemble ---
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

            # Transformar para cada modelo
            Xsel_opt = transformar_para_modelo(model_opt, word_v_opt, char_v_opt, selector_opt, full_text)
            Xsel_ov = transformar_para_modelo(model_over, word_v_ov, char_v_ov, selector_ov, full_text)

            # Prever probabilidades
            probs_opt = None
            if hasattr(clf_opt, "predict_proba"):
                probs_opt = clf_opt.predict_proba(Xsel_opt)
            else:
                # converter predição para “probabilidade” trivial
                pred_o = clf_opt.predict(Xsel_opt)[0]
                probs_opt = np.zeros((1, len(le_opt.classes_)))
                probs_opt[0, pred_o] = 1.0

            probs_ov = None
            if hasattr(clf_over, "predict_proba"):
                probs_ov = clf_over.predict_proba(Xsel_ov)
            else:
                pred_o2 = clf_over.predict(Xsel_ov)[0]
                probs_ov = np.zeros((1, len(le_opt.classes_)))
                probs_ov[0, pred_o2] = 1.0

            # Ensemble: média simples (soft voting)
            # peso igual para os dois modelos — você pode ajustar pesos
            peso1 = 0.5
            peso2 = 0.5
            probs_ensemble = peso1 * probs_opt + peso2 * probs_ov
            idx = int(probs_ensemble[0].argmax())
            classe = le_opt.inverse_transform([idx])[0]
            confidence = float(probs_ensemble[0, idx])

            resp = {
                "success": True,
                "prediction": classe,
                "confidence": confidence,
                "tokens": len(full_text.split())
            }

            return jsonify(resp)

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Falha interna no processamento"}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos rodando 🚀"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
