import os, re, tempfile, shutil, zipfile, pickle, gc, traceback, multiprocessing, time
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
import nltk
import docx
import pdfplumber
import pytesseract
from PIL import Image
import concurrent.futures

# --- NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

# --- Config ---
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}
MAX_FILE_SIZE = 20 * 1024 * 1024   # 20MB
MAX_PAGES = 80                      # ajuste se quiser menos
OCR_PAGE_TIMEOUT = 12               # segundos por página em OCR subprocess
EXTRACTION_TIMEOUT = 40             # tempo máximo (s) para toda extração em subprocess
REQUEST_TIMEOUT = 90                # timeout global (não forçado, só referência)

# --- Util helpers ---
def limpar_texto(texto: str) -> str:
    texto = (texto or "").lower()
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-zá-ú\s]", " ", texto)
    return " ".join([p for p in texto.split() if p not in STOPWORDS])

def sanitize_filename(filename):
    filename = secure_filename(filename or "")
    parts = filename.split(".")
    if len(parts) > 2:
        filename = parts[0] + "." + parts[-1]
    return filename

# --- Extração pesada (rodada em subprocesso) ---
def _worker_extract(filepath, max_pages, queue):
    """Worker que roda em subprocesso — devolve texto ou ''."""
    try:
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            texto = ""
            with pdfplumber.open(filepath) as pdf:
                pages = pdf.pages[:max_pages]
                for p in pages:
                    t = p.extract_text()
                    if t:
                        texto += t + " "
            texto = texto.strip()
            if not texto:
                # OCR por página, respeitando tempo por página
                try:
                    from pdf2image import convert_from_path
                    pages_img = convert_from_path(filepath, dpi=150, fmt='jpeg')[:max_pages]
                    for img in pages_img:
                        try:
                            t = pytesseract.image_to_string(img, lang="por", config="--psm 6")
                            texto += t + " "
                        except Exception:
                            pass
                        finally:
                            try:
                                img.close()
                            except Exception:
                                pass
                except Exception:
                    pass
            queue.put((True, (texto or "").strip()))
            return
        elif ext in (".docx", ".doc"):
            try:
                doc = docx.Document(filepath)
                texto = " ".join([p.text for p in doc.paragraphs]) or ""
                queue.put((True, texto.strip()))
                return
            except Exception:
                queue.put((True, ""))
                return
        elif ext == ".txt":
            try:
                with open(filepath, encoding="utf-8", errors="ignore") as f:
                    queue.put((True, f.read()))
                    return
            except Exception:
                queue.put((True, ""))
                return
        elif ext in ("png", "jpg", "jpeg", "tiff"):
            try:
                img = Image.open(filepath).convert("L")
                texto = pytesseract.image_to_string(img, lang="por", config="--psm 3")
                queue.put((True, (texto or "").strip()))
                try: img.close()
                except: pass
                return
            except Exception:
                queue.put((True, ""))
                return
        elif ext == ".zip":
            # sinalizamos vazio — processamento do zip no processo principal (mais seguro)
            queue.put((True, "ZIP"))
            return
        else:
            queue.put((True, ""))
            return
    except Exception as e:
        # Nunca deixar crashar sem resposta
        queue.put((False, str(e)))
        return

def run_extraction_with_timeout(filepath, max_pages=MAX_PAGES, timeout=EXTRACTION_TIMEOUT):
    """Roda extração em subprocesso; retorna (ok, texto_or_error_flag)."""
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker_extract, args=(filepath, max_pages, queue))
    p.start()
    p.join(timeout)
    if p.is_alive():
        # timeout: termina o processo
        p.terminate()
        p.join(2)
        return (False, "TIMEOUT_EXTRACTION")
    try:
        ok, payload = queue.get_nowait()
        return (ok, payload)
    except Exception:
        return (False, "NO_RESPONSE_FROM_WORKER")

# --- ZIP handling (process in main process but cautious) ---
def process_zip(filepath, max_pages=MAX_PAGES):
    textos = []
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            tmpdir = tempfile.mkdtemp(prefix="zip_extract_")
            for name in z.namelist():
                safe_name = os.path.basename(name)
                if not safe_name:
                    continue
                z.extract(name, tmpdir)
                fpath = os.path.join(tmpdir, name)
                ok, res = run_extraction_with_timeout(fpath, max_pages=max_pages, timeout=EXTRACTION_TIMEOUT/2)
                if ok and res and res != "ZIP":
                    textos.append(limpar_texto(res))
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        print("[ERRO ZIP]", e)
    return textos

# --- Load model (assume dict as before) ---
with open("modelo_curriculos_xgb_oversampling.pkl", "rb") as f:
    data = pickle.load(f)
if not isinstance(data, dict):
    raise RuntimeError("Formato de pickle inesperado")
clf = data["clf"]
word_v = data["word_vectorizer"]
char_v = data["char_vectorizer"]
palavras_chave_dict = data["palavras_chave_dict"]
selector = data["selector"]
le = data["label_encoder"]

def extrair_features_chave(texto):
    return [int(any(p.lower() in texto for p in palavras)) for palavras in palavras_chave_dict.values()]

# --- BERT fallback lazy load (unchanged) ---
bert_model = None
clf_bert = None
le_bert = le
if os.path.exists("modelo_bert_fallback.pkl"):
    with open("modelo_bert_fallback.pkl", "rb") as f:
        clf_bert, le_bert = pickle.load(f)

# --- Flask app ---
from flask import Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"success": False, "error": "Arquivo muito grande. Máximo permitido: 20MB"}), 413

@app.errorhandler(Exception)
def handle_all_errors(e):
    print("[ERRO GLOBAL]", traceback.format_exc())
    return jsonify({"success": False, "error": "Erro interno do servidor"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    tmpdir = None
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        filename = sanitize_filename(uploaded.filename)
        if not filename:
            return jsonify({"success": False, "error": "Nome de arquivo vazio"}), 400

        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"success": False, "error": f"Tipo de arquivo não suportado: {filename}"}), 400

        tmpdir = tempfile.mkdtemp(prefix="cv_api_")
        path = os.path.join(tmpdir, filename)
        uploaded.save(path)

        textos = []

        # ZIP
        if ext == "zip":
            textos.extend(process_zip(path))
        else:
            ok, res = run_extraction_with_timeout(path, max_pages=MAX_PAGES, timeout=EXTRACTION_TIMEOUT)
            if not ok:
                # res can be "TIMEOUT_EXTRACTION" or an error message
                print("[EXTRACTION ISSUE]", res)
                # return controlled JSON so client never sees HTML
                return jsonify({"success": False, "error": "Não foi possível extrair (timeout ou erro interno)"}), 400
            if res and res != "ZIP":
                textos.append(limpar_texto(res))

        if not textos:
            return jsonify({"success": False, "error": "Não foi possível extrair texto do arquivo"}), 400

        full_text = " ".join(textos)

        # TF-IDF pipeline
        Xw = word_v.transform([full_text])
        Xc = char_v.transform([full_text])
        Xchaves = csr_matrix([extrair_features_chave(full_text)])
        Xfull = hstack([Xw, Xc, Xchaves])
        Xsel = selector.transform(Xfull)

        probs = clf.predict_proba(Xsel)[0]
        idx = probs.argmax()
        conf = float(probs[idx])
        classe = le.inverse_transform([idx])[0]
        origem = "tfidf"

        # BERT fallback if needed
        LIMIAR = 0.65
        if conf < LIMIAR and clf_bert is not None:
            origem = "bert"
            try:
                global bert_model
                if bert_model is None:
                    from sentence_transformers import SentenceTransformer
                    bert_model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")
                emb = bert_model.encode([full_text])
                pb = clf_bert.predict_proba(emb)[0]
                ib = pb.argmax()
                classe = le_bert.inverse_transform([ib])[0]
                conf = float(pb[ib])
            except Exception:
                print("[ERRO BERT]", traceback.format_exc())
                classe, conf = "INDEFINIDO", 0.0

        return jsonify({
            "success": True,
            "prediction": classe,
            "confidence": round(conf,3),
            "origin": origem,
            "tokens": len(full_text.split())
        })

    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
