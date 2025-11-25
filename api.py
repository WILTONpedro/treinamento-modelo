import os
import re
import tempfile
import traceback
import pickle
import time
import gc
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import numpy as np
from scipy.sparse import hstack, csr_matrix

# Text extraction
import pdfplumber
import docx
from PIL import Image, ImageEnhance
import pytesseract

# ---------------- CONFIG ----------------
MODEL_PATH = "modelo_unificado.pkl"  # caminho para seu modelo
PORT = int(os.environ.get("PORT", 5000))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# Tesseract OCR
TESSERACT_WIN = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
TESSERACT_LINUX = "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_WIN if os.name == "nt" else TESSERACT_LINUX

app = Flask(__name__)

# ---------------- Helpers OCR / Extraction ----------------
def melhorar_imagem_para_ocr(img):
    try:
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img); img = enhancer.enhance(1.8)
        enhancer = ImageEnhance.Sharpness(img); img = enhancer.enhance(1.3)
    except Exception:
        pass
    return img

def extrair_texto_imagem(path):
    try:
        img = Image.open(path)
        img = melhorar_imagem_para_ocr(img)
        txt = pytesseract.image_to_string(img, lang="por", config="--psm 6")
        img.close()
        return txt or ""
    except Exception:
        traceback.print_exc()
        return ""

def extrair_texto_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            paginas = [p.extract_text() or "" for p in pdf.pages]
            texto = "\n".join(paginas).strip()
            if texto:
                return texto
    except Exception:
        traceback.print_exc()
    # fallback OCR se PDF for scan
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(path, dpi=200, first_page=1, last_page=2)
        partes = []
        for im in images:
            im = melhorar_imagem_para_ocr(im)
            partes.append(pytesseract.image_to_string(im, lang="por"))
        return "\n".join(partes).strip()
    except Exception:
        return ""

def extrair_texto_docx(path):
    try:
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        traceback.print_exc()
        return ""

def extrair_texto_txt(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        traceback.print_exc()
        return ""

def extrair_texto_arquivo(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return extrair_texto_imagem(path)
    if ext == ".pdf":
        return extrair_texto_pdf(path)
    if ext in {".docx", ".doc"}:
        return extrair_texto_docx(path)
    if ext == ".txt":
        return extrair_texto_txt(path)
    return ""

# ---------------- Limpeza simples ----------------
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-úçâêîôûãõ0-9\s]")

def limpar_texto(texto: str) -> str:
    t = (texto or "").lower()
    t = RE_EMAIL.sub(" ", t)
    t = RE_NUM.sub(" ", t)
    t = RE_CARACT.sub(" ", t)
    t = " ".join(t.split())
    return t

# ---------------- Carregar modelo ----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

clf = MODEL.get("clf")
wv = MODEL.get("word_vectorizer")
cv = MODEL.get("char_vectorizer")
vect_cursos = MODEL.get("vect_cursos")
vect_exp = MODEL.get("vect_exp")
selector = MODEL.get("selector")
le = MODEL.get("label_encoder")
palavras_chave = MODEL.get("palavras_chave_dict", {})
feature_dim = int(MODEL.get("feature_dim", 0))

# ---------------- Funções utilitárias ----------------
def ensure_feature_dim(X, expected_dim):
    if expected_dim is None or expected_dim == 0:
        return X
    if X.shape[1] == expected_dim:
        return X
    if X.shape[1] < expected_dim:
        zeros = csr_matrix((X.shape[0], expected_dim - X.shape[1]))
        return hstack([X, zeros]).tocsr()
    return X[:, :expected_dim]

def montar_features_na_ordem_do_treino(texto_limpo):
    Xw = wv.transform([texto_limpo]) if wv is not None else csr_matrix((1,0))
    Xc = cv.transform([texto_limpo]) if cv is not None else csr_matrix((1,0))
    Xcu = vect_cursos.transform([texto_limpo]) if vect_cursos is not None else csr_matrix((1,0))
    Xex = vect_exp.transform([texto_limpo]) if vect_exp is not None else csr_matrix((1,0))
    Xtime = csr_matrix(np.zeros((1,1)))

    def pad_trunc(X_block, expected_dim):
        if expected_dim is None: return X_block
        n = X_block.shape[1]
        if n == expected_dim: return X_block
        elif n < expected_dim:
            zeros = csr_matrix((1, expected_dim - n))
            return hstack([X_block, zeros])
        else:
            return X_block[:, :expected_dim]

    Xw = pad_trunc(Xw, wv.max_features if hasattr(wv, "max_features") else None)
    Xc = pad_trunc(Xc, cv.max_features if hasattr(cv, "max_features") else None)
    Xcu = pad_trunc(Xcu, vect_cursos.max_features if hasattr(vect_cursos, "max_features") else None)
    Xex = pad_trunc(Xex, vect_exp.max_features if hasattr(vect_exp, "max_features") else None)

    X_full = hstack([Xw, Xc, Xcu, Xex, Xtime])
    return X_full.tocsr()

def predict_with_text(texto):
    texto_limpo = limpar_texto(texto)
    if not texto_limpo or len(texto_limpo.split()) < 3:
        return None

    X_full = montar_features_na_ordem_do_treino(texto_limpo)

    if selector is not None:
        X_full = ensure_feature_dim(X_full, selector.n_features_in_)
        X_sel = selector.transform(X_full)
    else:
        X_sel = X_full

    if feature_dim:
        X_sel = ensure_feature_dim(X_sel, feature_dim)

    probs = clf.predict_proba(X_sel)[0]
    pred_pos = int(np.argmax(probs))
    label = le.inverse_transform([pred_pos])[0] if le is not None else str(pred_pos)

    prob_map = {lab: float(p) for lab, p in zip(le.classes_, probs)} if le is not None else {str(i): float(p) for i,p in enumerate(probs)}
    top = float(probs[pred_pos])
    second = float(sorted(probs, reverse=True)[1]) if len(probs) > 1 else 0.0

    return {"pred_label": label, "top_prob": top, "gap": top-second, "probs": prob_map}

# ---------------- Endpoints ----------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "feature_dim": feature_dim,
        "n_classes": len(le.classes_) if le else None
    })

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "campo 'file' ausente"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"success": False, "error": "arquivo sem nome"}), 400

        filename = secure_filename(uploaded.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"success": False, "error": f"extensão não permitida: {ext}"}), 400

        subject = request.form.get("subject") or request.form.get("assunto") or ""
        body = request.form.get("body") or request.form.get("corpo") or ""

        with tempfile.TemporaryDirectory(prefix="cv_api_") as tmpdir:
            tmp_path = os.path.join(tmpdir, filename)
            uploaded.save(tmp_path)
            texto_extraido = extrair_texto_arquivo(tmp_path)
            combined = " ".join([subject, body, texto_extraido]).strip()

            if not combined or len(combined.split()) < 6:
                return jsonify({"success": False, "error": "conteúdo insuficiente para classificar"}), 422

            res = predict_with_text(combined)
            if res is None:
                return jsonify({"success": False, "error": "falha na predição"}), 500

            response = {
                "success": True,
                "prediction": res["pred_label"],
                "confidence": round(res["top_prob"], 4),
                "detail": {"gap": round(res["gap"], 4), "probs": res["probs"]}
            }

            try: os.remove(tmp_path)
            except Exception: pass
            gc.collect()
            return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/admin/reload-model", methods=["POST"])
def reload_model():
    try:
        global MODEL, clf, wv, cv, vect_cursos, vect_exp, selector, le, palavras_chave, feature_dim
        with open(MODEL_PATH, "rb") as f:
            MODEL = pickle.load(f)
        clf = MODEL.get("clf")
        wv = MODEL.get("word_vectorizer")
        cv = MODEL.get("char_vectorizer")
        vect_cursos = MODEL.get("vect_cursos")
        vect_exp = MODEL.get("vect_exp")
        selector = MODEL.get("selector")
        le = MODEL.get("label_encoder")
        palavras_chave = MODEL.get("palavras_chave_dict", {})
        feature_dim = int(MODEL.get("feature_dim", 0))
        return jsonify({"success": True, "message": "modelo recarregado"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
