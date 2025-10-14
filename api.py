import os
import tempfile
import zipfile
import pickle
import traceback
import platform
import shutil

import docx
import pdfplumber
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords
import re
import numpy as np

# --- Setup NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

# --- Regex ---
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-ú\s]")

def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

# --- Configurar pytesseract ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
else:
    tess_path = shutil.which("tesseract")
    if tess_path:
        pytesseract.pytesseract.tesseract_cmd = tess_path
    else:
        print("⚠️ Tesseract não encontrado no ambiente — OCR de imagens ficará desativado")

# --- Funções de processamento ---
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
            if not shutil.which("tesseract"):
                print(f"[IGNORADO - OCR indisponível] {filepath}")
                return ""
            if os.path.getsize(filepath) > 10 * 1024 * 1024:
                print(f"[IGNORADO - imagem muito grande] {filepath}")
                return ""
            img = Image.open(filepath).convert("L")
            img.thumbnail((1800, 1800))
            texto = pytesseract.image_to_string(img, lang="por", config="--psm 6")
            print(f"[IMG OCR] {filepath}: {len(texto.strip())} chars extraídos")
            return texto

        elif ext == ".pdf":
            texts = []
            with pdfplumber.open(filepath) as pdf:
                for i, page in enumerate(pdf.pages):
                    t = page.extract_text()
                    if t and len(t.strip()) > 10:
                        texts.append(t)
                        continue
                    if shutil.which("tesseract"):
                        pil_img = page.to_image(resolution=300).original.convert("L")
                        ocr_text = pytesseract.image_to_string(pil_img, lang="por", config="--psm 6")
                        if ocr_text.strip():
                            print(f"[OCR PDF pág {i+1}] extraídos {len(ocr_text.strip())} chars")
                            texts.append(ocr_text)
                        else:
                            print(f"[OCR FALHOU pág {i+1}]")
            full_text = " ".join(texts)
            print(f"[PDF FINAL] {len(full_text.strip())} chars totais extraídos")
            return full_text

        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            texto = " ".join(p.text for p in doc.paragraphs)
            print(f"[DOCX] {filepath}: {len(texto.strip())} chars extraídos")
            return texto

        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                texto = f.read()
            print(f"[TXT] {filepath}: {len(texto.strip())} chars extraídos")
            return texto

        else:
            print(f"[ARQUIVO NÃO SUPORTADO] {filepath}")
            return ""

    except Exception:
        print(f"[ERRO EXTRAÇÃO] {filepath}\n{traceback.format_exc()}")
        return ""

def carregar_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# --- Carregar modelos ---
model_opt = carregar_model("modelo_curriculos_otimizado.pkl")
model_over = carregar_model("modelo_curriculos_xgb_oversampling.pkl")

clf_opt = model_opt["clf"]
clf_over = model_over["clf"]
wv_opt = model_opt["word_vectorizer"]
cv_opt = model_opt["char_vectorizer"]
sel_opt = model_opt["selector"]
le_opt = model_opt["label_encoder"]

wv_ov = model_over["word_vectorizer"]
cv_ov = model_over["char_vectorizer"]
sel_ov = model_over["selector"]
le_ov = model_over["label_encoder"]

# --- Classes unificadas ---
classes_opt = list(le_opt.classes_)
classes_ov = list(le_ov.classes_)
classes_unificadas = sorted(set(classes_opt) | set(classes_ov))
map_opt_to_uni = {cls: classes_unificadas.index(cls) for cls in classes_opt}
map_ov_to_uni = {cls: classes_unificadas.index(cls) for cls in classes_ov}

def transformar(model_dict, wv, cv, sel, texto_limpo):
    Xw = wv.transform([texto_limpo])
    Xc = cv.transform([texto_limpo])
    Xfull = hstack([Xw, Xc])
    Xsel = sel.transform(Xfull)
    return Xsel

# --- Flask API ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(fname):
    return os.path.splitext(fname)[1].lower().lstrip(".") in ALLOWED_EXTENSIONS

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
            for pf in processar_item(filepath):
                t = extrair_texto_arquivo(pf)
                if t:
                    textos.append(limpar_texto(t))
            full = " ".join(textos)
            if not full.strip():
                return jsonify({"error": "Não foi possível extrair texto"}), 400
            if len(full) > 30000:
                full = full[:30000]

            Xopt = transformar(model_opt, wv_opt, cv_opt, sel_opt, full)
            Xov = transformar(model_over, wv_ov, cv_ov, sel_ov, full)

            def get_probs(clf, X, le_src, map_to_uni):
                if hasattr(clf, "predict_proba"):
                    p = clf.predict_proba(X)[0]
                else:
                    idx = clf.predict(X)[0]
                    p = np.zeros(len(le_src.classes_))
                    p[idx] = 1.0
                p_uni = np.zeros(len(classes_unificadas))
                for cls_src, prob_val in zip(le_src.classes_, p):
                    tgt_idx = map_to_uni[cls_src]
                    p_uni[tgt_idx] = prob_val
                return p_uni

            probs_opt = get_probs(clf_opt, Xopt, le_opt, map_opt_to_uni)
            probs_ov = get_probs(clf_over, Xov, le_ov, map_ov_to_uni)

            ps = 0.5 * probs_opt + 0.5 * probs_ov
            idx_uni = int(np.argmax(ps))
            classe_pred = classes_unificadas[idx_uni]
            confidence = float(ps[idx_uni])

            return jsonify({
                "success": True,
                "prediction": classe_pred,
                "confidence": confidence,
                "tokens": len(full.split())
            })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Falha interna no processamento"}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API rodando"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
