import os
import tempfile
import zipfile
import pickle
import traceback
import re

import docx
import pdfplumber
import pytesseract
from PIL import Image
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
import nltk
from nltk.corpus import stopwords

# --- Setup Tesseract e NLTK ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-ú\s]")


def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)


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
    """Extrai texto de PDFs, imagens, DOCX e TXT, com fallback OCR para PDFs escaneados."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        # === Imagens ===
        if ext in (".png", ".jpg", ".jpeg", ".tiff"):
            if os.path.getsize(filepath) > 5 * 1024 * 1024:
                print(f"[AVISO] Imagem muito grande ignorada: {filepath}")
                return ""
            img = Image.open(filepath).convert("L")
            img.thumbnail((1024, 1024))
            return pytesseract.image_to_string(img, lang="por", config="--psm 6")

        # === PDFs ===
        elif ext == ".pdf":
            texts = []
            with pdfplumber.open(filepath) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    t = page.extract_text()
                    if t and len(t.strip()) > 20:
                        texts.append(t)
                    else:
                        # Fallback OCR (para páginas escaneadas)
                        try:
                            pil_img = page.to_image(resolution=200).original.convert("L")
                            ocr_text = pytesseract.image_to_string(pil_img, lang="por", config="--psm 6")
                            if ocr_text.strip():
                                texts.append(ocr_text)
                                print(f"[INFO] OCR aplicado na página {page_num} do PDF.")
                        except Exception:
                            print(f"[ERRO OCR] Falha ao aplicar OCR na página {page_num}.")
            return " ".join(texts)

        # === Word (DOCX/DOC) ===
        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            return " ".join(p.text for p in doc.paragraphs)

        # === Texto simples ===
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()

    except Exception:
        print(f"[ERRO] Falha ao extrair texto de {filepath}\n{traceback.format_exc()}")
        return ""

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

            # --- Transformar para cada modelo ---
            Xopt = transformar(model_opt, wv_opt, cv_opt, sel_opt, full)
            Xov = transformar(model_over, wv_ov, cv_ov, sel_ov, full)

            import numpy as np

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

            # --- Combinar previsões (soft voting) ---
            peso_opt = 0.5
            peso_ov = 0.5
            ps = peso_opt * probs_opt + peso_ov * probs_ov

            idx_uni = int(np.argmax(ps))
            classe_pred = classes_unificadas[idx_uni]
            confidence = float(ps[idx_uni])

            resp = {
                "success": True,
                "prediction": classe_pred,
                "confidence": confidence,
                "tokens": len(full.split())
            }
            return jsonify(resp)

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Falha interna no processamento"}), 500


@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API rodando"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
