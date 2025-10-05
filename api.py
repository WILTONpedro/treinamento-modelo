import os
import re
import tempfile
import shutil
import pickle
import gc
import traceback
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import stopwords
import pdfplumber
import docx
from pdf2image import convert_from_path
import concurrent.futures

# --- Setup NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

def limpar_texto(txt):
    txt = txt.lower()
    txt = re.sub(r"\S+@\S+", " ", txt)
    txt = re.sub(r"\d+", " ", txt)
    txt = re.sub(r"[^a-zá-ú\s]", " ", txt)
    return " ".join([t for t in txt.split() if t not in STOPWORDS])

# --- Timeout OCR ---
TIMEOUT_OCR = 15  # segundos

def extrair_texto_pdf_ocr(fp, dpi=150):
    texto = ""
    try:
        pages = convert_from_path(fp, dpi=dpi)
    except Exception as e:
        print(f"[ERRO conversão PDF] {fp}: {e}")
        return ""
    def process_page(page_img):
        try:
            return pytesseract.image_to_string(page_img, lang="por", config="--psm 6")
        except Exception:
            return ""
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_page, p) for p in pages]
            for f in concurrent.futures.as_completed(futures, timeout=TIMEOUT_OCR):
                try:
                    texto += f.result()
                except concurrent.futures.TimeoutError:
                    print("[TIMEOUT OCR] página ignorada")
                finally:
                    del f
                    gc.collect()
    except concurrent.futures.TimeoutError:
        print(f"[TIMEOUT OCR] arquivo ignorado {fp}")
    for p in pages:
        del p
    gc.collect()
    return texto.strip()

def extrair_texto_imagem_ocr(fp):
    texto = ""
    try:
        img = Image.open(fp).convert("L")
        img = img.point(lambda x: 0 if x < 140 else 255, '1')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(pytesseract.image_to_string, img, "por", "--psm 3")
            texto = future.result(timeout=TIMEOUT_OCR)
        img.close()
    except concurrent.futures.TimeoutError:
        print(f"[TIMEOUT OCR] {fp}")
    except Exception as e:
        print(f"[ERRO OCR] {fp}: {e}")
    finally:
        gc.collect()
    return texto.strip()

def extrair_texto_arquivo(fp):
    ext = os.path.splitext(fp)[1].lower()
    try:
        if ext == ".pdf":
            texto = ""
            with pdfplumber.open(fp) as pdf:
                for p in pdf.pages:
                    if p.extract_text():
                        texto += p.extract_text() + " "
            if not texto.strip():
                texto += extrair_texto_pdf_ocr(fp)
            return texto.strip() or ""
        elif ext in (".docx", ".doc"):
            doc = docx.Document(fp)
            return " ".join(p.text for p in doc.paragraphs) or ""
        elif ext == ".txt":
            with open(fp, encoding="utf-8", errors="ignore") as f:
                return f.read() or ""
        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            return extrair_texto_imagem_ocr(fp) or ""
    except Exception as e:
        print(f"[ERRO extração] {fp}: {e}")
        return ""

# --- Carregar modelos ---
with open("modelo_curriculos_xgb_oversampling.pkl", "rb") as f:
    data = pickle.load(f)
clf = data["clf"]
word_v = data["word_vectorizer"]
char_v = data["char_vectorizer"]
palavras_chave_dict = data["palavras_chave_dict"]
selector = data["selector"]
le = data["label_encoder"]

def extrair_features_chave(texto):
    return [int(any(p.lower() in texto for p in palavras)) for palavras in palavras_chave_dict.values()]

# --- Fallback BERT (lazy-loaded) ---
bert_model = None
clf_bert = None
le_bert = le
if os.path.exists("modelo_bert_fallback.pkl"):
    with open("modelo_bert_fallback.pkl", "rb") as f:
        clf_bert, le_bert = pickle.load(f)

# --- Config Flask ---
app = Flask(__name__)
ALLOWED_EXT = {"pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "tiff"}
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"success": False, "error": "Arquivo muito grande. Máximo permitido: 20MB"}), 413

@app.route("/predict", methods=["POST"])
def predict():
    tmpdir = None
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename)
        if not filename:
            return jsonify({"success": False, "error": "Nome vazio"}), 400

        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in ALLOWED_EXT:
            return jsonify({"success": False, "error": "Extensão não permitida"}), 400

        tmpdir = tempfile.mkdtemp(prefix="tmp_api_")
        path = os.path.join(tmpdir, filename)
        uploaded.save(path)

        texto_raw = extrair_texto_arquivo(path)
        texto = limpar_texto(texto_raw)

        if not texto.strip():
            return jsonify({"success": False, "error": "Não foi possível extrair texto do arquivo"}), 400

        # --- Inferência TF-IDF ---
        Xw = word_v.transform([texto])
        Xc = char_v.transform([texto])
        Xch = csr_matrix([extrair_features_chave(texto)])
        X = selector.transform(hstack([Xw, Xc, Xch]))

        probs = clf.predict_proba(X)[0]
        idx = np.argmax(probs)
        conf = float(probs[idx])
        classe = le.inverse_transform([idx])[0]
        origem = "tfidf"

        LIMIAR = 0.65
        if conf < LIMIAR and clf_bert is not None:
            origem = "bert"
            try:
                global bert_model
                if bert_model is None:
                    from sentence_transformers import SentenceTransformer
                    bert_model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")
                emb = bert_model.encode([texto])
                pb = clf_bert.predict_proba(emb)[0]
                ib = np.argmax(pb)
                classe = le_bert.inverse_transform([ib])[0]
                conf = float(pb[ib])
            except Exception:
                classe, conf = "INDEFINIDO", 0.0
                print(f"[ERRO BERT] {traceback.format_exc()}")

        return jsonify({
            "success": True,
            "prediction": classe,
            "confidence": round(conf, 3),
            "origin": origem
        })

    except Exception:
        print("[ERRO API]", traceback.format_exc())
        return jsonify({"success": False, "error": "Erro interno ao processar arquivo"}), 500

    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
