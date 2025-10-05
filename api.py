import os, re, tempfile, shutil, pickle, gc
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import stopwords
import pdfplumber
import docx
from pdf2image import convert_from_path
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

from pdf2image import convert_from_path
from PIL import Image
import pytesseract, pdfplumber, docx, os, gc

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
                # Converte e processa página por página, liberando memória
                for page_img in convert_from_path(fp, dpi=150):
                    texto += pytesseract.image_to_string(page_img, lang="por", config="--psm 6")
                    del page_img
                    gc.collect()
            return texto.strip()

        elif ext in (".docx", ".doc"):
            doc = docx.Document(fp)
            return " ".join(p.text for p in doc.paragraphs)

        elif ext == ".txt":
            return open(fp, encoding="utf-8", errors="ignore").read()

        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            img = Image.open(fp).convert("L")
            img = img.point(lambda x: 0 if x < 140 else 255, '1')
            texto = pytesseract.image_to_string(img, lang="por", config="--psm 3")
            img.close()
            return texto.strip()

    except Exception as e:
        print(f"[ERRO] {fp}: {e}")
        return ""

# --- Carregar modelos leves ---
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

# --- Fallback BERT opcional (carregado só se realmente necessário) ---
bert_model = None
clf_bert = None
le_bert = le
if os.path.exists("modelo_bert_fallback.pkl"):
    with open("modelo_bert_fallback.pkl", "rb") as f:
        clf_bert, le_bert = pickle.load(f)

app = Flask(__name__)
ALLOWED_EXT = {"pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "tiff"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # checks iniciais
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400
        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename)
        if not filename:
            return jsonify({"error": "Nome vazio"}), 400
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in ALLOWED_EXT:
            return jsonify({"error": "Extensão não permitida"}), 400

        # salvar temporário
        tmpdir = tempfile.mkdtemp(prefix="tmp_api_")
        path = os.path.join(tmpdir, filename)
        uploaded.save(path)

        texto_raw = extrair_texto_arquivo(path)
        texto = limpar_texto(texto_raw)

        shutil.rmtree(tmpdir, ignore_errors=True)
        gc.collect()

        if not texto.strip():
            return jsonify({"error": "Não foi possível extrair texto"}), 400

        # inferência TF-IDF
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
            if bert_model is None:
                from sentence_transformers import SentenceTransformer
                bert_model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")
            emb = bert_model.encode([texto])
            try:
                pb = clf_bert.predict_proba(emb)[0]
                ib = np.argmax(pb)
                classe = le_bert.inverse_transform([ib])[0]
                conf = float(pb[ib])
            except Exception:
                classe, conf = "INDEFINIDO", 0.0

        return jsonify({
            "success": True,
            "prediction": classe,
            "confidence": round(conf, 3),
            "origin": origem
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
