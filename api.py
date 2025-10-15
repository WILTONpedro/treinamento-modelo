# api.py
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
from scipy.sparse import hstack, csr_matrix
import scipy.sparse as sp
import numpy as np

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
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    return os.path.splitext(filename)[1].lower().lstrip(".") in ALLOWED_EXTENSIONS

def processar_item(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".zip":
        try:
            with zipfile.ZipFile(filepath, "r") as z:
                with tempfile.TemporaryDirectory() as td:
                    z.extractall(td)
                    for name in z.namelist():
                        yield os.path.join(td, name)
        except Exception:
            return
    else:
        yield filepath

def extrair_texto_arquivo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg", ".tiff"):
            if os.path.getsize(filepath) > 6 * 1024 * 1024:
                return ""
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
# Carregar modelos (.pkl)
# ---------------------------
MODEL_BASE_PATH = "modelo_curriculos_otimizado.pkl"
MODEL_OVER_PATH = "modelo_curriculos_xgb_oversampling.pkl"

def carregar_modelo(path):
    with open(path, "rb") as f:
        return pickle.load(f)

model_base = carregar_modelo(MODEL_BASE_PATH)
model_over = carregar_modelo(MODEL_OVER_PATH)

# Componentes model_base
clf_base = model_base["clf"]
word_v_base = model_base["word_vectorizer"]
char_v_base = model_base["char_vectorizer"]
selector_base = model_base.get("selector", None)
le_base = model_base["label_encoder"]

# Componentes model_over
clf_over = model_over["clf"]
word_v_over = model_over["word_vectorizer"]
char_v_over = model_over["char_vectorizer"]
selector_over = model_over.get("selector", None)
le_over = model_over["label_encoder"]

# palavras chave do oversampling (opcional)
palavras_chave_dict = model_over.get("palavras_chave_dict", {})

# ---------------------------
# Utils
# ---------------------------
def _adjust_sparse_cols(X_sparse, expected_features):
    """Corta ou preenche colunas da sparse matrix X_sparse até expected_features."""
    X = X_sparse.tocsr()
    n_rows, n_cols = X.shape
    if n_cols == expected_features:
        return X
    if n_cols > expected_features:
        return X[:, :expected_features].tocsr()
    # n_cols < expected_features -> pad zeros
    diff = expected_features - n_cols
    zeros = sp.csr_matrix((n_rows, diff))
    return sp.hstack([X, zeros]).tocsr()

def align_probs_to_common_classes(probs, model_le, common_classes):
    """Alinha vetor de probabilidades de model_le.classes_ para common_classes order."""
    aligned = np.zeros(len(common_classes), dtype=float)
    model_classes = list(model_le.classes_)
    for i, c in enumerate(model_classes):
        if c in common_classes:
            j = common_classes.index(c)
            aligned[j] = probs[i]
    return aligned

# ---------------------------
# Predição e fusão
# ---------------------------
def prever_fusao_texto(full_text_raw):
    texto = limpar_texto(full_text_raw or "")
    if not texto.strip():
        raise ValueError("Texto vazio após extração/limpeza.")

    # MODEL BASE pipeline
    Xw_b = word_v_base.transform([texto])
    Xc_b = char_v_base.transform([texto])
    X_b = hstack([Xw_b, Xc_b])
    # aplica selector_base se existir, senão ajusta para o esperado
    if selector_base is not None:
        try:
            Xsel_b = selector_base.transform(X_b)
        except Exception:
            try:
                expected_b = clf_base.get_booster().num_features()
            except Exception:
                expected_b = X_b.shape[1]
            Xsel_b = _adjust_sparse_cols(X_b, expected_b)
    else:
        try:
            expected_b = clf_base.get_booster().num_features()
        except Exception:
            expected_b = X_b.shape[1]
        Xsel_b = _adjust_sparse_cols(X_b, expected_b)

    probs_b = clf_base.predict_proba(Xsel_b)[0]

    # MODEL OVER pipeline
    Xw_o = word_v_over.transform([texto])
    Xc_o = char_v_over.transform([texto])
    X_o = hstack([Xw_o, Xc_o])

    if selector_over is not None:
        try:
            Xsel_o = selector_over.transform(X_o)
        except Exception as e:
            # selector incompatível -> ajustar cols para o que clf_over espera
            try:
                expected_over = clf_over.get_booster().num_features()
            except Exception:
                expected_over = X_o.shape[1]
            Xsel_o = _adjust_sparse_cols(X_o, expected_over)
    else:
        try:
            expected_over = clf_over.get_booster().num_features()
        except Exception:
            expected_over = X_o.shape[1]
        Xsel_o = _adjust_sparse_cols(X_o, expected_over)

    probs_o = clf_over.predict_proba(Xsel_o)[0]

    # --- Common classes (união preservando ordem do base)
    classes_base = list(le_base.classes_)
    classes_over = list(le_over.classes_)
    common_classes = classes_base[:]
    for c in classes_over:
        if c not in common_classes:
            common_classes.append(c)

    # align
    probs_b_al = align_probs_to_common_classes(probs_b, le_base, common_classes)
    probs_o_al = align_probs_to_common_classes(probs_o, le_over, common_classes)

    # confidences
    conf_b = float(np.max(probs_b_al))
    conf_o = float(np.max(probs_o_al))

    # pesos dinâmicos pela confiança (se soma zero, pesos iguais)
    w_b = conf_b
    w_o = conf_o
    if w_b + w_o <= 0:
        w_b = w_o = 0.5
    else:
        # normalize so w_b + w_o == 1
        s = w_b + w_o
        w_b = w_b / s
        w_o = w_o / s

    # fusão final
    probs_final = probs_b_al * w_b + probs_o_al * w_o
    idx_final = int(np.argmax(probs_final))
    class_final = common_classes[idx_final]
    conf_final = float(probs_final[idx_final])

    # preds individuais (origem)
    idx_b = int(np.argmax(probs_b))
    pred_b = classes_base[idx_b] if idx_b < len(classes_base) else None
    idx_o = int(np.argmax(probs_o))
    pred_o = classes_over[idx_o] if idx_o < len(classes_over) else None

    # palavras-chave encontradas (somente as que aparecem no texto)
    keywords_found = []
    for area, palavras in palavras_chave_dict.items():
        for p in palavras:
            if p and p.lower() in texto:
                keywords_found.append(p)
    keywords_found = list(dict.fromkeys(keywords_found))  # unique preserve order

    return {
        "prediction_base": pred_b,
        "confidence_base": float(np.max(probs_b)),
        "prediction_over": pred_o,
        "confidence_over": float(np.max(probs_o)),
        "prediction_final": class_final,
        "confidence_final": conf_final,
        "common_classes": common_classes,
        "palavras_chave": keywords_found
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

        # opcional: receber assunto e corpo via form-data
        assunto = request.form.get("assunto", "") or ""
        corpo = request.form.get("corpo", "") or ""

        with tempfile.TemporaryDirectory(prefix="cv_api_") as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            uploaded.save(filepath)

            textos = []
            for p in processar_item(filepath):
                txt = extrair_texto_arquivo(p)
                if txt:
                    textos.append(txt)

            full_text_raw = " ".join(textos + [assunto, corpo]).strip()
            if not full_text_raw:
                return jsonify({"success": False, "error": "Não foi possível extrair texto do arquivo."}), 400

            if len(full_text_raw) > 30000:
                full_text_raw = full_text_raw[:30000]

            resultado = prever_fusao_texto(full_text_raw)

            # For compatibility with your Apps Script, return top-level fields 'prediction' and 'confidence'
            response = {
                "success": True,
                "prediction": resultado["prediction_final"],
                "confidence": resultado["confidence_final"],
                # extras useful for debug/reporting
                "prediction_base": resultado["prediction_base"],
                "confidence_base": resultado["confidence_base"],
                "prediction_over": resultado["prediction_over"],
                "confidence_over": resultado["confidence_over"],
                "palavras_chave": resultado["palavras_chave"],
                "common_classes": resultado["common_classes"]
            }

            return jsonify(response)

    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Falha interna no processamento"}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "API de Currículos (ensemble) ativa"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
