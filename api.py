import os
import re
import tempfile
import traceback
import pickle
import json
import time
import gc

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import numpy as np
from scipy.sparse import hstack, csr_matrix, issparse

# text extraction libs
import pdfplumber
import docx
from PIL import Image, ImageEnhance
import pytesseract

# Config
MODEL_PATH = "modelo_unificado.pkl"   # coloque aqui seu pickle gerado
PORT = int(os.environ.get("PORT", 5000))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

# Tesseract: ajusta automaticamente entre Windows/Linux
TESSERACT_WIN = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
TESSERACT_LINUX = "/usr/bin/tesseract"
pytesseract.pytesseract.tesseract_cmd = TESSERACT_WIN if os.name == "nt" else TESSERACT_LINUX

app = Flask(__name__)

# ----------------- util imagem / ocr -----------------
def melhorar_imagem_para_ocr(img: Image.Image) -> Image.Image:
    try:
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img); img = enhancer.enhance(1.8)
        enhancer = ImageEnhance.Sharpness(img); img = enhancer.enhance(1.3)
    except Exception:
        pass
    return img

def extrair_texto_imagem(path: str) -> str:
    try:
        img = Image.open(path)
        img = melhorar_imagem_para_ocr(img)
        txt = pytesseract.image_to_string(img, lang="por", config="--psm 6")
        img.close()
        return txt or ""
    except Exception:
        traceback.print_exc()
        return ""

def extrair_texto_pdf(path: str) -> str:
    try:
        with pdfplumber.open(path) as pdf:
            paginas = [p.extract_text() or "" for p in pdf.pages]
            texto = "\n".join(paginas).strip()
            if texto:
                return texto
    except Exception:
        traceback.print_exc()

    # fallback simples: tenta o OCR nas primeiras 2 páginas se pdf2image disponível
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(path, dpi=200, first_page=1, last_page=2)
        partes = []
        for im in images:
            im = melhorar_imagem_para_ocr(im)
            partes.append(pytesseract.image_to_string(im, lang="por"))
        return "\n".join(partes).strip()
    except Exception:
        # se não houver pdf2image ou falhar, retorna ""
        return ""

def extrair_texto_docx(path: str) -> str:
    try:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception:
        traceback.print_exc()
        return ""

def extrair_texto_txt(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        traceback.print_exc()
        return ""

def extrair_texto_arquivo(path: str) -> str:
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

# ----------------- limpeza de texto leve -----------------
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-úçâêîôûãõ0-9\s]")

def limpar_texto(texto: str) -> str:
    t = (texto or "").lower()
    t = RE_EMAIL.sub(" ", t)
    t = RE_NUM.sub(" ", t)
    t = RE_CARACT.sub(" ", t)
    # normaliza espaços
    t = " ".join(t.split())
    return t

# ----------------- carregar modelo -----------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Arquivo de modelo não encontrado: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)

# extrai componentes (nomes conforme seu treino)
clf = MODEL.get("clf")
wv = MODEL.get("word_vectorizer")
cv = MODEL.get("char_vectorizer")
vect_cursos = MODEL.get("vect_cursos")
vect_exp = MODEL.get("vect_exp")
selector = MODEL.get("selector")
le = MODEL.get("label_encoder")
palavras_chave = MODEL.get("palavras_chave_dict", {})
feature_dim = int(MODEL.get("feature_dim", 0))

# ---------- helpers para features e predição ----------
def ensure_feature_dim(X, expected_dim):
    """Pad (com zeros) ou trunc columns para expected_dim. X é sparse."""
    if expected_dim is None or expected_dim == 0:
        return X
    if X.shape[1] == expected_dim:
        return X
    if X.shape[1] < expected_dim:
        n_extra = expected_dim - X.shape[1]
        zeros = csr_matrix((X.shape[0], n_extra))
        return hstack([X, zeros]).tocsr()
    # truncar
    return X[:, :expected_dim]

def normalize_probs(probs):
    """Retorna dict class_name -> prob. Usa label_encoder quando possível."""
    # probs correspondem à ordem de classes do clf (clf.classes_) ou 0..n-1
    try:
        if hasattr(clf, "classes_") and clf.classes_ is not None:
            model_classes = list(clf.classes_)
            # se classes_ são índices numéricos (0..n-1), mapeia via label encoder
            if le is not None:
                try:
                    # converter model_classes (p.ex [0,1,2]) para labels
                    model_classes_int = [int(x) for x in model_classes]
                    human = list(le.inverse_transform(model_classes_int))
                    return {lab: float(p) for lab, p in zip(human, probs)}
                except Exception:
                    # fallback: usar str(classes_)
                    return {str(c): float(p) for c, p in zip(model_classes, probs)}
            else:
                return {str(c): float(p) for c, p in zip(model_classes, probs)}
        else:
            # fallback: usar label_encoder.classes_ if exists and length matches
            if le is not None and len(le.classes_) == len(probs):
                return {lab: float(p) for lab, p in zip(le.classes_, probs)}
            return {f"class_{i}": float(p) for i, p in enumerate(probs)}
    except Exception:
        return {f"class_{i}": float(p) for i, p in enumerate(probs)}

def predict_with_text(texto: str):
    text_clean = limpar_texto(texto)
    if not text_clean.strip():
        return None
    # transformar
    Xw = wv.transform([text_clean]) if wv is not None else csr_matrix((1,0))
    Xc = cv.transform([text_clean]) if cv is not None else csr_matrix((1,0))
    Xcu = vect_cursos.transform([text_clean]) if vect_cursos is not None else csr_matrix((1,0))
    Xex = vect_exp.transform([text_clean]) if vect_exp is not None else csr_matrix((1,0))
    Xtime = csr_matrix(np.zeros((1,1)))
    Xfull = hstack([Xw, Xc, Xcu, Xex, Xtime])
    # aplicar selector se houver
    try:
        Xsel = selector.transform(Xfull) if selector is not None else Xfull
    except Exception:
        # tentativa segura: sem selector
        Xsel = Xfull
    # garantir dimensões esperadas
    if feature_dim:
        Xsel = ensure_feature_dim(Xsel, feature_dim)
    # prever
    probs = clf.predict_proba(Xsel)[0]
    pred_pos = int(np.argmax(probs))
    # mapear para label humano via clf.classes_ -> le.inverse_transform
    try:
        if hasattr(clf, "classes_") and clf.classes_ is not None:
            clf_classes = np.array(clf.classes_, dtype=int)
            mapped = int(clf_classes[pred_pos])
            label = le.inverse_transform([mapped])[0]
        else:
            label = le.inverse_transform([pred_pos])[0]
    except Exception:
        # fallback: usar ordem de le.classes_
        try:
            label = le.inverse_transform([pred_pos])[0]
        except Exception:
            label = str(pred_pos)
    prob_map = normalize_probs(probs)
    top = float(probs[pred_pos]) if len(probs) > 0 else 0.0
    second = float(sorted(probs, reverse=True)[1]) if len(probs) > 1 else 0.0
    return {"pred_label": label, "top_prob": top, "gap": top-second, "probs": prob_map}

# ----------------- endpoints -----------------
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "feature_dim": feature_dim,
        "n_classes": len(le.classes_) if le is not None else None
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

        # save temporário
        with tempfile.TemporaryDirectory(prefix="cv_api_") as tmpdir:
            tmp_path = os.path.join(tmpdir, filename)
            uploaded.save(tmp_path)

            texto_extraido = extrair_texto_arquivo(tmp_path)
            combined = " ".join([subject, body, texto_extraido]).strip()

            if not combined or len(combined.split()) < 6:
                return jsonify({
                    "success": False,
                    "error": "conteúdo insuficiente para classificar",
                    "extracted_length": len(texto_extraido)
                }), 422

            res = predict_with_text(combined)
            if res is None:
                return jsonify({"success": False, "error": "falha na predição"}), 500

            response = {
                "success": True,
                "prediction": res["pred_label"],
                "confidence": round(res["top_prob"], 4),
                "detail": {
                    "gap": round(res["gap"], 4),
                    "probs": res["probs"]
                }
            }

            # cleanup e gc
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            gc.collect()
            return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

# rota opcional para recarregar modelo (útil em deploy)
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
    # threaded para aceitar múltiplas requisições simultâneas
    app.run(host="0.0.0.0", port=PORT, threaded=True)
