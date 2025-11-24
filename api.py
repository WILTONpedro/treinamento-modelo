import os
import tempfile
import traceback
import pickle
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
from PIL import Image
import pytesseract
import pdfplumber
import docx
import re
import numpy as np

# -------------------------
# CONFIG - ajuste conforme necessário
# -------------------------
MODEL_UNIFICADO_PATH = "modelo_unificado.pkl"         # modelo que junta currículos + folders
MODEL_HASHING_PATH = "modelo_hashing_foco.pkl"        # seu outro modelo (hashing)
TESSERACT_CMD = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
PORT = int(os.environ.get("PORT", 5000))

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'tiff'}

# thresholds (ajustáveis)
GAP_SIGNIFICANT = 0.10   # diferença de gap para considerar vencedor claramente mais confiante
GAP_CONFIDENT = 0.15     # gap absoluto que indica alta confiança do modelo
INSECURE_GAP = 0.10      # se ambos abaixo => inseguros -> média

# -------------------------
# Tesseract config
# -------------------------
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# -------------------------
# util limpeza de texto
# -------------------------
import nltk
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("portuguese"))

RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-úçâêîôûãõ\s]")

def limpar_texto(texto: str) -> str:
    texto = (texto or "").lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

# -------------------------
# extrair texto dos arquivos suportados
# -------------------------
def extrair_texto_arquivo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in (".png", ".jpg", ".jpeg", ".tiff"):
            img = Image.open(filepath).convert("L")
            img.thumbnail((1200, 1200))
            text = pytesseract.image_to_string(img, lang="por", config="--psm 6")
            img.close()
            return text or ""
        elif ext == ".pdf":
            texts = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text() or ""
                    if t.strip():
                        texts.append(t)
            return "\n".join(texts)
        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception:
        traceback.print_exc()
        return ""
    return ""

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------
# carregar modelos
# -------------------------
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    # espera d ser dict com keys: clf, word_vectorizer, char_vectorizer, selector (opcional), label_encoder
    return d

print("Carregando modelos...")
MODEL_UNIFICADO = load_model(MODEL_UNIFICADO_PATH)
MODEL_HASHING = load_model(MODEL_HASHING_PATH)

clf_u = MODEL_UNIFICADO["clf"]
wv_u = MODEL_UNIFICADO["word_vectorizer"]
cv_u = MODEL_UNIFICADO["char_vectorizer"]
sel_u = MODEL_UNIFICADO.get("selector", None)
le_u = MODEL_UNIFICADO["label_encoder"]

clf_h = MODEL_HASHING["clf"]
wv_h = MODEL_HASHING["word_vectorizer"]
cv_h = MODEL_HASHING["char_vectorizer"]
sel_h = MODEL_HASHING.get("selector", None)
le_h = MODEL_HASHING["label_encoder"]

# Basic check: classes should be same set or similar — we will map by class names (strings)
classes_u = list(le_u.classes_)
classes_h = list(le_h.classes_)

# Create a unified class set (intersection or union). We will use union but keep consistent naming.
all_classes = list(dict.fromkeys(list(classes_u) + list(classes_h)))  # preserves order, union

# Helper: normalize class probabilities to the same class order (all_classes)
def normalize_probs(probs, le, model_classes):
    # probs: array-like of shape (n_classes,) in model_classes order
    # return dictionary mapping class_name -> prob (over all_classes)
    prob_map = {}
    for cls_name, p in zip(model_classes, probs):
        prob_map[cls_name] = float(p)
    # classes missing in model get 0
    for cls in all_classes:
        if cls not in prob_map:
            prob_map[cls] = 0.0
    return prob_map

# -------------------------
# predição por modelo
# -------------------------
def predict_with_model(text, clf, wv, cv, selector, le):
    text_clean = limpar_texto(text)
    Xw = wv.transform([text_clean])
    Xc = cv.transform([text_clean])
    Xfull = hstack([Xw, Xc])
    if selector is not None:
        try:
            Xsel = selector.transform(Xfull)
        except Exception:
            Xsel = Xfull
    else:
        Xsel = Xfull
    probs = clf.predict_proba(Xsel)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_idx])[0]
    # also prepare ordered probs with model's classes
    model_classes = list(le.classes_)
    prob_map = normalize_probs(probs, le, model_classes)
    # compute gap (top - second)
    sorted_probs = sorted(probs, reverse=True)
    top = float(sorted_probs[0]) if len(sorted_probs) > 0 else 0.0
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    gap = top - second
    return {
        "pred_label": pred_label,
        "pred_index": pred_idx,
        "probs": prob_map,
        "top_prob": top,
        "gap": gap,
        "model_classes": model_classes
    }

# -------------------------
# lógica de fusão (consenso + gaps + média)
# -------------------------
def merge_predictions(res_u, res_h):
    # if both agree
    if res_u["pred_label"] == res_h["pred_label"]:
        final = {
            "final_prediction": res_u["pred_label"],
            "final_confidence": max(res_u["top_prob"], res_h["top_prob"]),
            "reason": "agreement",
            "winner_model": "both"
        }
        return final

    # compute gaps
    gap_u = res_u["gap"]
    gap_h = res_h["gap"]

    # prefer model with much larger gap (sign of confident decision)
    if (gap_u - gap_h) >= GAP_SIGNIFICANT and gap_u >= GAP_CONFIDENT:
        return {
            "final_prediction": res_u["pred_label"],
            "final_confidence": res_u["top_prob"],
            "reason": f"unificado_gap_better (gap_u {gap_u:.3f} > gap_h {gap_h:.3f})",
            "winner_model": "unificado"
        }
    if (gap_h - gap_u) >= GAP_SIGNIFICANT and gap_h >= GAP_CONFIDENT:
        return {
            "final_prediction": res_h["pred_label"],
            "final_confidence": res_h["top_prob"],
            "reason": f"hashing_gap_better (gap_h {gap_h:.3f} > gap_u {gap_u:.3f})",
            "winner_model": "hashing"
        }

    # if both insecure (gaps small) -> average probabilities
    if gap_u < INSECURE_GAP and gap_h < INSECURE_GAP:
        # average probs over all_classes
        avg_probs = {}
        for cls in all_classes:
            p = (res_u["probs"].get(cls, 0.0) + res_h["probs"].get(cls, 0.0)) / 2.0
            avg_probs[cls] = p
        final_cls = max(avg_probs.items(), key=lambda x: x[1])[0]
        return {
            "final_prediction": final_cls,
            "final_confidence": float(avg_probs[final_cls]),
            "reason": "both_insecure_avg",
            "winner_model": "combined_avg",
            "combined_probs": avg_probs
        }

    # fallback: take the model with larger top probability (more absolutely confident)
    if res_u["top_prob"] > res_h["top_prob"]:
        return {
            "final_prediction": res_u["pred_label"],
            "final_confidence": res_u["top_prob"],
            "reason": "unificado_higher_top_prob",
            "winner_model": "unificado"
        }
    else:
        return {
            "final_prediction": res_h["pred_label"],
            "final_confidence": res_h["top_prob"],
            "reason": "hashing_higher_top_prob",
            "winner_model": "hashing"
        }

# -------------------------
# Flask app
# -------------------------
from flask import Flask
app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "message": "API de triagem rodando com dois modelos"}

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        # Expect 'file' field containing the resume AND optional 'assunto' and 'corpo' fields
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado (campo 'file' ausente)"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"success": False, "error": "Arquivo sem nome"}), 400

        filename = secure_filename(uploaded.filename)
        if not allowed_file(filename):
            return jsonify({"success": False, "error": f"Tipo de arquivo não suportado: {filename}"}), 400

        assunto = request.form.get("assunto") or request.form.get("subject") or ""
        corpo = request.form.get("corpo") or request.form.get("body") or ""

        with tempfile.TemporaryDirectory(prefix="cv_api_") as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            uploaded.save(filepath)

            texto_extraido = extrair_texto_arquivo(filepath)
            if not texto_extraido.strip() and not (assunto.strip() or corpo.strip()):
                return jsonify({"success": False, "error": "Não foi possível extrair texto do arquivo e assunto/corpo vazios"}), 400

            # concatenar assunto/corpo antes de predizer (se existirem)
            combined_text = " ".join([assunto, corpo, texto_extraido]).strip()

            # predições individuais
            res_u = predict_with_model(combined_text, clf_u, wv_u, cv_u, sel_u, le_u)
            res_h = predict_with_model(combined_text, clf_h, wv_h, cv_h, sel_h, le_h)

            # merge
            merged = merge_predictions(res_u, res_h)

            # build debug/audit info
            detail = {
                "unificado": {
                    "pred": res_u["pred_label"],
                    "top_prob": res_u["top_prob"],
                    "gap": res_u["gap"],
                    "probs": res_u["probs"]
                },
                "hashing": {
                    "pred": res_h["pred_label"],
                    "top_prob": res_h["top_prob"],
                    "gap": res_h["gap"],
                    "probs": res_h["probs"]
                },
                "merged": merged,
                "all_classes": all_classes
            }

            response = {
                "success": True,
                "prediction": merged["final_prediction"],
                "confidence": merged["final_confidence"],
                "detail": detail
            }
            return jsonify(response)

    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "error": "Falha interna no servidor"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
