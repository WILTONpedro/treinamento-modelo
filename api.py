import os
import tempfile
import traceback
import pickle
import gc
import re
import json
import time
import shutil
from collections import Counter
import numpy as np
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
from PIL import Image, ImageEnhance
import pytesseract
import pdfplumber
import docx
import nltk

# -------------------------
# CONFIGURAÇÕES
# -------------------------
MODEL_UNIFICADO_PATH = "modelo_unificado.pkl"
MODEL_HASHING_PATH = "modelo_hashing_foco.pkl"
PORT = int(os.environ.get("PORT", 5000))

# Tesseract: Windows (default) or Linux path
TESSERACT_WIN = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
TESSERACT_LINUX = "/usr/bin/tesseract"
TESSERACT_CMD = TESSERACT_LINUX if os.name != "nt" else TESSERACT_WIN

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg', 'tiff'}
LOG_FILE = "logs_api.jsonl"
LOW_CONF_DIR = "low_confidence_examples"
LOW_CONF_THRESHOLD = 0.15   # ajustar conforme necessidade
GAP_SIGNIFICANT = 0.10

# configura Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# garantir diretórios
os.makedirs(LOW_CONF_DIR, exist_ok=True)

# -------------------------
# NLTK / STOPWORDS
# -------------------------
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("portuguese"))

# regexs
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-úçâêîôûãõ\s]")

# -------------------------
# UTIL: imagem / ocr / extração
# -------------------------
def melhorar_imagem_para_ocr(img):
    """Melhora contraste e nitidez para OCR (retorna PIL image em L)."""
    try:
        img = img.convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.5)
    except Exception:
        try:
            img = img.convert("L")
        except Exception:
            pass
    return img

def validar_texto_util(texto: str, min_len: int = 50) -> bool:
    """Verifica se o texto extraído tem aparência natural (stopwords presentes)."""
    if not texto or len(texto) < min_len:
        return False
    palavras = texto.lower().split()
    contagem = sum(1 for p in palavras if p in STOPWORDS)
    ratio = contagem / len(palavras) if len(palavras) > 0 else 0
    return ratio > 0.02

def extrair_texto_smart(filepath):
    """
    Extrai texto de arquivos:
    - imagens: OCR
    - pdf: tenta texto digital via pdfplumber; se insuficiente, tenta OCR com pdf2image (se disponível)
    - docx: lê parágrafos
    - txt: lê raw
    Retorna: texto, metodo
    """
    ext = os.path.splitext(filepath)[1].lower()
    texto = ""
    metodo = "none"
    try:
        if ext in (".png", ".jpg", ".jpeg", ".tiff"):
            img = Image.open(filepath)
            img = melhorar_imagem_para_ocr(img)
            texto = pytesseract.image_to_string(img, lang="por", config="--psm 6")
            img.close()
            metodo = "ocr_image"

        elif ext == ".pdf":
            try:
                with pdfplumber.open(filepath) as pdf:
                    paginas = [p.extract_text() or "" for p in pdf.pages]
                    texto = "\n".join(paginas)
                metodo = "pdf_text"
                if not validar_texto_util(texto):
                    # tentar OCR fallback se pdf2image disponível
                    try:
                        from pdf2image import convert_from_path
                        images = convert_from_path(filepath, dpi=200, first_page=1, last_page=2)
                        texto_ocr = []
                        for im in images:
                            im = melhorar_imagem_para_ocr(im)
                            texto_ocr.append(pytesseract.image_to_string(im, lang="por"))
                        texto = "\n".join(texto_ocr)
                        metodo = "ocr_pdf_fallback"
                    except ImportError:
                        metodo = "pdf_text_no_pdf2image"
                    except Exception:
                        metodo = "pdf_text_ocr_error"
            except Exception:
                metodo = "pdf_read_error"

        elif ext in (".docx", ".doc"):
            try:
                doc = docx.Document(filepath)
                texto = "\n".join([p.text for p in doc.paragraphs])
                metodo = "docx"
            except Exception:
                metodo = "docx_error"

        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                texto = f.read()
            metodo = "txt"
    except Exception:
        traceback.print_exc()
        return "", "error"

    return (texto or ""), metodo

def limpar_texto(texto: str) -> str:
    texto = (texto or "").lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

def extrair_keywords(texto: str, top: int = 10):
    if not texto:
        return []
    tokens = [t for t in texto.lower().split() if t not in STOPWORDS and len(t) > 2]
    ctr = Counter(tokens)
    return [w for w,_ in ctr.most_common(top)]

def save_log_entry(entry: dict):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        traceback.print_exc()

def save_low_confidence_example(src_filepath: str, filename: str, meta: dict):
    try:
        ts = int(time.time())
        dest_dir = os.path.join(LOW_CONF_DIR, f"{ts}_{secure_filename(filename)}")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src_filepath, os.path.join(dest_dir, filename))
        with open(os.path.join(dest_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        traceback.print_exc()
        
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    # compatibilidade com pickles diferentes
    if isinstance(d, dict):
        if "modelo" in d and "clf" not in d:
            d["clf"] = d["modelo"]
        if "word_vectorizer" not in d and "vect_texto" in d:
            d["word_vectorizer"] = d["vect_texto"]
        if "char_vectorizer" not in d and "vect_exp" in d:
            d["char_vectorizer"] = d["vect_exp"]
        if "label_encoder" not in d:
            d["label_encoder"] = d.get("le") or d.get("labelenc")
    return d

print("⏳ Carregando modelos...")
MODEL_UNIFICADO = load_model(MODEL_UNIFICADO_PATH)
MODEL_HASHING = load_model(MODEL_HASHING_PATH)

clf_u = MODEL_UNIFICADO.get("clf")
wv_u = MODEL_UNIFICADO.get("word_vectorizer")
cv_u = MODEL_UNIFICADO.get("char_vectorizer")
sel_u = MODEL_UNIFICADO.get("selector", None)
le_u = MODEL_UNIFICADO.get("label_encoder")

clf_h = MODEL_HASHING.get("clf")
wv_h = MODEL_HASHING.get("word_vectorizer")
cv_h = MODEL_HASHING.get("char_vectorizer")
sel_h = MODEL_HASHING.get("selector", None)
le_h = MODEL_HASHING.get("label_encoder")

# union das classes
all_classes = list(dict.fromkeys(list(getattr(le_u, "classes_", [])) + list(getattr(le_h, "classes_", []))))

# -------------------------
# Helpers para mapear probs -> nomes (ordem real do clf)
# -------------------------
def model_classes_names_from_clf(clf, le):
    if clf is None:
        return []
    if hasattr(clf, "classes_") and clf.classes_ is not None:
        arr = np.array(clf.classes_)
        try:
            idxs = arr.astype(int)
            return list(le.inverse_transform(idxs))
        except Exception:
            return [str(x) for x in arr]
    if le is not None and hasattr(le, "classes_"):
        try:
            return list(le.inverse_transform(list(range(len(le.classes_)))))
        except Exception:
            return list(le.classes_)
    return []

def normalize_probs_from_clf(probs, clf, le):
    prob_map = {}
    names = model_classes_names_from_clf(clf, le)
    if len(names) != len(probs):
        if le is not None and hasattr(le, "classes_") and len(le.classes_) == len(probs):
            names = list(le.classes_)
        else:
            names = [f"class_{i}" for i in range(len(probs))]
    for name, p in zip(names, probs):
        prob_map[name] = float(p)
    for c in all_classes:
        if c not in prob_map:
            prob_map[c] = 0.0
    return prob_map

def predict_with_model(text, clf, wv, cv, selector, le):
    text_clean = limpar_texto(text)
    if not text_clean.strip():
        return None
    Xw = wv.transform([text_clean])
    Xc = cv.transform([text_clean])
    Xfull = hstack([Xw, Xc])
    Xsel = selector.transform(Xfull) if selector is not None else Xfull
    probs = clf.predict_proba(Xsel)[0]
    prob_map = normalize_probs_from_clf(probs, clf, le)
    model_names = model_classes_names_from_clf(clf, le)
    if len(model_names) == len(probs):
        pred_pos = int(np.argmax(probs))
        pred_label = model_names[pred_pos]
    else:
        try:
            pred_pos = int(np.argmax(probs))
            pred_label = le.inverse_transform([pred_pos])[0]
        except Exception:
            pred_label = max(prob_map.items(), key=lambda x: x[1])[0]
    sorted_probs = sorted(probs, reverse=True)
    top = float(sorted_probs[0]) if len(sorted_probs) > 0 else 0.0
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    return {
        "pred_label": pred_label,
        "top_prob": top,
        "gap": top - second,
        "probs": prob_map
    }

def merge_predictions(res_u, res_h, gap_significant=GAP_SIGNIFICANT):
    if not res_u and not res_h:
        return None
    if not res_u:
        return {**res_h, "winner_model": "hashing_only", "reason": "hashing_only"}
    if not res_h:
        return {**res_u, "winner_model": "unificado_only", "reason": "unificado_only"}
    if res_u["pred_label"] == res_h["pred_label"]:
        return {
            "final_prediction": res_u["pred_label"],
            "final_confidence": max(res_u["top_prob"], res_h["top_prob"]),
            "reason": "agreement",
            "winner_model": "both"
        }
    if (res_u["gap"] - res_h["gap"]) >= gap_significant and res_u["top_prob"] >= res_h["top_prob"]:
        return {
            "final_prediction": res_u["pred_label"],
            "final_confidence": res_u["top_prob"],
            "reason": "unificado_strong_gap",
            "winner_model": "unificado"
        }
    if (res_h["gap"] - res_u["gap"]) >= gap_significant and res_h["top_prob"] >= res_u["top_prob"]:
        return {
            "final_prediction": res_h["pred_label"],
            "final_confidence": res_h["top_prob"],
            "reason": "hashing_strong_gap",
            "winner_model": "hashing"
        }
    avg_probs = {}
    for cls in all_classes:
        avg_probs[cls] = (res_u["probs"].get(cls, 0.0) + res_h["probs"].get(cls, 0.0)) / 2.0
    final_cls = max(avg_probs.items(), key=lambda x: x[1])[0]
    return {
        "final_prediction": final_cls,
        "final_confidence": avg_probs[final_cls],
        "reason": "weighted_average",
        "winner_model": "average",
        "combined_probs": avg_probs
    }

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET"])
def health():
    return {"status": "ok", "message": "API de triagem (reformulada) em execução", "tesseract_cmd": pytesseract.pytesseract.tesseract_cmd}

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    filepath = None
    filename = None
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Arquivo não enviado - campo 'file' ausente"}), 400
        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"success": False, "error": "Arquivo sem nome"}), 400
        filename = secure_filename(uploaded.filename)
        if not allowed_file(filename):
            return jsonify({"success": False, "error": f"Extensão não permitida: {filename}"}), 400
        assunto = request.form.get("assunto") or request.form.get("subject") or ""
        corpo = request.form.get("corpo") or request.form.get("body") or ""
        with tempfile.TemporaryDirectory(prefix="cv_processing_") as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            uploaded.save(filepath)
            texto_extraido, metodo_extracao = extrair_texto_smart(filepath)
            texto_final_raw = f"{assunto} {corpo} {texto_extraido}".strip()
            if len(texto_final_raw) < 40 or not validar_texto_util(texto_extraido):
                log_entry = {
                    "timestamp": int(time.time()),
                    "filename": filename,
                    "status": "rejected_low_quality",
                    "extracted_length": len(texto_extraido),
                    "extraction_method": metodo_extracao
                }
                save_log_entry(log_entry)
                return jsonify({
                    "success": False,
                    "error": "Conteúdo insuficiente/qualidade baixa para classificação",
                    "details": {"extracted_length": len(texto_extraido), "method": metodo_extracao}
                }), 422
            res_u = predict_with_model(texto_final_raw, clf_u, wv_u, cv_u, sel_u, le_u)
            res_h = predict_with_model(texto_final_raw, clf_h, wv_h, cv_h, sel_h, le_h)
            merged = merge_predictions(res_u, res_h)
            if merged is None:
                return jsonify({"success": False, "error": "Nenhuma predição gerada"}), 500
            # baixa confiança -> guarda exemplo e retorna 409
            if merged.get("final_confidence", 0.0) < LOW_CONF_THRESHOLD:
                meta = {
                    "timestamp": int(time.time()),
                    "filename": filename,
                    "extraction_method": metodo_extracao,
                    "text_length": len(texto_final_raw),
                    "merged": merged,
                    "unificado_top": res_u["top_prob"] if res_u else None,
                    "hashing_top": res_h["top_prob"] if res_h else None,
                    "keywords": extrair_keywords(texto_final_raw, top=12)
                }
                save_low_confidence_example(filepath, filename, meta)
                save_log_entry({**meta, "status": "low_confidence"})
                return jsonify({
                    "success": False,
                    "error": "classificacao_incerta",
                    "details": {"final_confidence": merged.get("final_confidence"), "reason": merged.get("reason")},
                    "meta": {"extraction_method": metodo_extracao, "text_length": len(texto_final_raw)}
                }), 409
            response = {
                "success": True,
                "prediction": merged["final_prediction"],
                "confidence": round(float(merged["final_confidence"]), 4),
                "meta": {
                    "extraction_method": metodo_extracao,
                    "text_length": len(texto_final_raw),
                    "model_reason": merged.get("reason"),
                    "winner_model": merged.get("winner_model")
                },
                "debug": {
                    "prob_unificado": res_u["top_prob"] if res_u else 0,
                    "prob_hashing": res_h["top_prob"] if res_h else 0,
                    "top_keywords": extrair_keywords(texto_final_raw, top=8)
                }
            }
            log_entry = {
                "timestamp": int(time.time()),
                "filename": filename,
                "status": "classified",
                "prediction": merged["final_prediction"],
                "final_confidence": merged.get("final_confidence"),
                "reason": merged.get("reason"),
                "extraction_method": metodo_extracao,
                "text_length": len(texto_final_raw),
                "keywords": extrair_keywords(texto_final_raw, top=8)
            }
            save_log_entry(log_entry)
            del texto_extraido, texto_final_raw, res_u, res_h, merged
            gc.collect()
            return jsonify(response)
    except Exception as e:
        traceback.print_exc()
        try:
            save_log_entry({"timestamp": int(time.time()), "filename": filename, "status": "error", "error": str(e)})
        except Exception:
            pass
        return jsonify({"success": False, "error": f"Erro interno: {str(e)}"}), 500

@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        n = int(request.args.get("n", 200))
        if not os.path.exists(LOG_FILE):
            return jsonify({"logs": []})
        lines = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line))
        return jsonify({"logs": lines[-n:]})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/low_conf_examples", methods=["GET"])
def list_low_conf_examples():
    try:
        itens = []
        for name in sorted(os.listdir(LOW_CONF_DIR), reverse=True):
            d = os.path.join(LOW_CONF_DIR, name)
            meta_file = os.path.join(d, "meta.json")
            if os.path.isdir(d) and os.path.exists(meta_file):
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                itens.append({"dir": name, "meta": meta})
        return jsonify({"examples": itens})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/low_conf_examples/<dir_name>/file", methods=["GET"])
def get_low_conf_file(dir_name):
    try:
        d = os.path.join(LOW_CONF_DIR, secure_filename(dir_name))
        if not os.path.isdir(d):
            return jsonify({"success": False, "error": "Diretório não existe"}), 404
        # pega primeiro arquivo que não seja meta.json
        for f in os.listdir(d):
            if f == "meta.json": continue
            path = os.path.join(d, f)
            return send_file(path, as_attachment=True)
        return jsonify({"success": False, "error": "Arquivo original não encontrado"}), 404
    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, threaded=True)
