import os
import re
import tempfile
import shutil
import zipfile
import pickle
import traceback
import gc

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from scipy.sparse import hstack, csr_matrix

# --- Configurações ---
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
OCR_ENABLED = os.getenv("ENABLE_OCR", "false").lower() == "true"
LIMIAR = 0.50

# --- Carregar modelo leve inicial ---
with open("modelo_curriculos_super_avancado.pkl", "rb") as f:
    model_data = pickle.load(f)

clf = model_data["clf"]
word_v = model_data["word_vectorizer"]
char_v = model_data["char_vectorizer"]
palavras_chave_dict = model_data["palavras_chave_dict"]
selector = model_data["selector"]
le = model_data["label_encoder"]

# --- Variáveis de fallback (BERT) ---
bert_model = None
clf_bert = None
le_bert = None

# --- Funções auxiliares ---
def allowed_file(filename):
    return os.path.splitext(filename)[1].lower().lstrip(".") in ALLOWED_EXTENSIONS

def limpar_texto(texto: str) -> str:
    import nltk
    from nltk.corpus import stopwords
    texto = texto.lower()
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-zá-ú\s]", " ", texto)
    STOPWORDS = set(stopwords.words("portuguese"))
    return " ".join([p for p in texto.split() if p not in STOPWORDS])

def extrair_texto_arquivo(fp):
    import pdfplumber, docx, pytesseract
    from PIL import Image
    texto = ""
    ext = os.path.splitext(fp)[1].lower()
    try:
        if ext == ".pdf":
            with pdfplumber.open(fp) as pdf:
                texto = " ".join([p.extract_text() or "" for p in pdf.pages if p.extract_text()])
            if not texto.strip() and OCR_ENABLED:
                texto = extrair_texto_pdf_ocr(fp)
        elif ext in (".docx", ".doc"):
            doc = docx.Document(fp)
            texto = " ".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            with open(fp, encoding="utf-8", errors="ignore") as f:
                texto = f.read()
        elif ext in ("png","jpg","jpeg","tiff") and OCR_ENABLED:
            texto = extrair_texto_imagem_ocr(fp)
    except Exception as e:
        print(f"[WARN] Falha ao extrair texto de {fp}: {e}")
    finally:
        gc.collect()
    return texto.strip()

def extrair_texto_pdf_ocr(fp):
    try:
        from pdf2image import convert_from_path
        import pytesseract
        pages = convert_from_path(fp, dpi=150)
        texto = ""
        for page in pages:
            texto += pytesseract.image_to_string(page, lang="por", config="--psm 6") or ""
        return texto.strip()
    except Exception as e:
        print(f"[WARN] OCR PDF falhou: {e}")
        return ""
    finally:
        gc.collect()

def extrair_texto_imagem_ocr(fp):
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(fp).convert("L")
        texto = pytesseract.image_to_string(img, lang="por", config="--psm 3") or ""
        img.close()
        return texto.strip()
    except Exception as e:
        print(f"[WARN] OCR imagem falhou: {e}")
        return ""
    finally:
        gc.collect()

def processar_zip(fp):
    textos = []
    try:
        with zipfile.ZipFile(fp, "r") as z:
            tmpd = tempfile.mkdtemp()
            for name in z.namelist():
                try:
                    extracted = z.extract(name, tmpd)
                    txt = extrair_texto_arquivo(os.path.join(tmpd, name))
                    if txt:
                        textos.append(txt)
                except Exception as e:
                    print(f"[WARN] Falha ao processar {name} no ZIP: {e}")
            shutil.rmtree(tmpd, ignore_errors=True)
    except Exception as e:
        print(f"[WARN] Falha ao abrir ZIP: {e}")
    finally:
        gc.collect()
    return textos

def extrair_features_chave(texto: str):
    return [int(any(p.lower() in texto for p in lst)) for lst in palavras_chave_dict.values()]

# --- Flask ---
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return jsonify({"success": False, "error": "Arquivo muito grande"}), 413

@app.route("/predict", methods=["POST"])
def predict():
    tmpdir = None
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename or "")
        if not filename:
            return jsonify({"success": False, "error": "Nome de arquivo vazio"}), 400
        if not allowed_file(filename):
            return jsonify({"success": False, "error": "Tipo de arquivo não suportado"}), 400

        tmpdir = tempfile.mkdtemp(prefix="cv_api_")
        fp = os.path.join(tmpdir, filename)
        uploaded.save(fp)

        # Extrair textos
        textos = []
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext == "zip":
            textos.extend(processar_zip(fp))
        else:
            txt = extrair_texto_arquivo(fp)
            if txt:
                textos.append(txt)

        if not textos:
            return jsonify({"success": False, "error": "Não foi possível extrair texto"}), 400

        full_text = " ".join(textos)
        cleaned = limpar_texto(full_text)

        # Vetorização + features
        Xw = word_v.transform([cleaned])
        Xc = char_v.transform([cleaned])
        Xk = csr_matrix([extrair_features_chave(cleaned)])
        Xfull = hstack([Xw, Xc, Xk])
        Xsel = selector.transform(Xfull)

        # Predição principal
        probs = clf.predict_proba(Xsel)[0]
        idx = int(probs.argmax())
        conf = float(probs[idx])
        classe = le.inverse_transform([idx])[0]
        origem = "tfidf"

        # Fallback BERT (só se necessário)
        if conf < LIMIAR:
            global clf_bert, le_bert, bert_model
            try:
                if clf_bert is None and os.path.exists("modelo_bert_fallback.pkl"):
                    with open("modelo_bert_fallback.pkl", "rb") as f:
                        clf_bert, le_bert = pickle.load(f)
                if clf_bert is not None:
                    if bert_model is None:
                        from sentence_transformers import SentenceTransformer
                        bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                    emb = bert_model.encode([cleaned])
                    from sklearn.preprocessing import normalize
                    emb = normalize(emb)
                    pb = clf_bert.predict_proba(emb)[0]
                    ib = int(pb.argmax())
                    classe = le_bert.inverse_transform([ib])[0]
                    conf = float(pb[ib])
                    origem = "bert"
            except Exception as e:
                print(f"[WARN] Fallback BERT falhou: {e}")
                classe, conf = "INDEFINIDO", 0.0

        result = {
            "success": True,
            "prediction": classe,
            "confidence": round(conf, 3),
            "origin": origem,
            "tokens": len(cleaned.split())
        }

        return jsonify(result)

    except Exception:
        traceback_str = traceback.format_exc()
        print("[ERRO API]", traceback_str)
        return jsonify({"success": False, "error": "Erro interno"}), 500

    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API rodando"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
