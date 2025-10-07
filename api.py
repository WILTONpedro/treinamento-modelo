import os, re, tempfile, shutil, zipfile, pickle, gc, traceback, threading
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
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_PAGES = 100  # Limite de páginas PDF
OCR_TIMEOUT = 15  # segundos por página
REQUEST_TIMEOUT = 60  # Timeout global por requisição

# --- Funções ---
def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-zá-ú\s]", " ", texto)
    return " ".join([p for p in texto.split() if p not in STOPWORDS])

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext in ALLOWED_EXTENSIONS

def sanitize_filename(filename):
    filename = secure_filename(filename)
    parts = filename.split(".")
    if len(parts) > 2:
        filename = parts[0] + "." + parts[-1]
    return filename

def extrair_texto_pdf_ocr(fp):
    texto = ""
    try:
        from pdf2image import convert_from_path
        pages = convert_from_path(fp, dpi=150)[:MAX_PAGES]
        def process_page(page_img):
            try:
                return pytesseract.image_to_string(page_img, lang="por", config="--psm 6")
            except Exception:
                return ""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_page, p) for p in pages]
            for f in concurrent.futures.as_completed(futures):
                try:
                    texto += f.result(timeout=OCR_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    continue
    except Exception as e:
        print(f"[ERRO OCR PDF] {fp}: {e}")
    finally:
        gc.collect()
    return texto.strip()

def extrair_texto_imagem_ocr(fp):
    texto = ""
    try:
        img = Image.open(fp).convert("L")
        img = img.point(lambda x: 0 if x < 140 else 255, '1')
        texto = pytesseract.image_to_string(img, lang="por", config="--psm 3")
        img.close()
    except Exception as e:
        print(f"[ERRO OCR IMG] {fp}: {e}")
    finally:
        gc.collect()
    return texto.strip()

def extrair_texto_arquivo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                texto = " ".join([p.extract_text() or "" for p in pdf.pages[:MAX_PAGES] if p.extract_text()])
            if not texto.strip():
                texto = extrair_texto_pdf_ocr(filepath)
            return texto.strip()
        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext in ("png","jpg","jpeg","tiff"):
            return extrair_texto_imagem_ocr(filepath)
        elif ext == ".zip":
            return ""  # tratado separadamente
    except Exception as e:
        print(f"[ERRO EXTRAÇÃO] {filepath}: {e}")
    finally:
        gc.collect()
    return ""

def processar_zip(filepath):
    textos = []
    try:
        with zipfile.ZipFile(filepath, "r") as z:
            tmpdir = tempfile.mkdtemp()
            for name in z.namelist():
                z.extract(name, tmpdir)
                txt = extrair_texto_arquivo(os.path.join(tmpdir, name))
                if txt:
                    textos.append(limpar_texto(txt))
            shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception as e:
        print(f"[ERRO ZIP] {filepath}: {e}")
    return textos

# --- Carregar modelo ---
with open("modelo_curriculos_super_avancado.pkl", "rb") as f:
    data = pickle.load(f)
clf = data["clf"]
word_v = data["word_vectorizer"]
char_v = data["char_vectorizer"]
palavras_chave_dict = data["palavras_chave_dict"]
selector = data["selector"]
le = data["label_encoder"]

def extrair_features_chave(texto):
    return [int(any(p.lower() in texto for p in palavras)) for palavras in palavras_chave_dict.values()]

# --- BERT fallback ---
bert_model = None
clf_bert = None
le_bert = le
if os.path.exists("modelo_bert_fallback.pkl"):
    with open("modelo_bert_fallback.pkl", "rb") as f:
        clf_bert, le_bert = pickle.load(f)

# --- Flask ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"success": False, "error": "Arquivo muito grande. Máximo permitido: 20MB"}), 413

@app.errorhandler(Exception)
def handle_all_errors(e):
    print("[ERRO GLOBAL]", traceback.format_exc())
    return jsonify({"success": False, "error": "Erro interno do servidor"}), 500

def run_with_timeout(func, args=(), timeout=REQUEST_TIMEOUT):
    result = {}
    thread = threading.Thread(target=lambda r: r.update({"value": func(*args)}), args=(result,))
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None
    return result.get("value")

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
        filepath = os.path.join(tmpdir, filename)
        uploaded.save(filepath)

        textos = []
        if ext == "zip":
            textos.extend(processar_zip(filepath))
        else:
            txt = run_with_timeout(extrair_texto_arquivo, (filepath,))
            if txt:
                textos.append(limpar_texto(txt))

        if not textos:
            return jsonify({"success": False, "error": "Não foi possível extrair texto"}), 400

        full_text = " ".join(textos)

        # TF-IDF
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

        # Fallback BERT
        LIMIAR = 0.20
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
                classe, conf = "INDEFINIDO", 0.0
                print(f"[ERRO BERT] {traceback.format_exc()}")

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
def
