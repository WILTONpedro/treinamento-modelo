import os, re, tempfile, shutil, pickle
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack, csr_matrix
import numpy as np
import pytesseract
from PIL import Image
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import pdfplumber
import docx
from pdf2image import convert_from_path

# --- NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")
STOPWORDS = set(stopwords.words("portuguese"))

# --- Função de limpeza ---
def limpar_texto(txt):
    txt = txt.lower()
    txt = re.sub(r"\S+@\S+", " ", txt)
    txt = re.sub(r"\d+", " ", txt)
    txt = re.sub(r"[^a-zá-ú\s]", " ", txt)
    return " ".join([t for t in txt.split() if t not in STOPWORDS])

# --- Função de extração de texto (PDF/Word/TXT/Imagens) ---
def extrair_texto_arquivo(fp):
    ext = os.path.splitext(fp)[1].lower()
    try:
        if ext == ".pdf":
            texto = ""
            with pdfplumber.open(fp) as pdf:
                texto = " ".join(p.extract_text() or "" for p in pdf.pages)
            if not texto.strip():  # aplica OCR se não houver texto
                imagens = convert_from_path(fp, dpi=300)
                texto = " ".join(
                    pytesseract.image_to_string(img, lang="por", config="--psm 6")
                    for img in imagens
                )
            return texto

        elif ext in (".docx", ".doc"):
            doc = docx.Document(fp)
            return " ".join(p.text for p in doc.paragraphs)

        elif ext == ".txt":
            with open(fp, encoding="utf-8", errors="ignore") as f:
                return f.read()

        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            img = Image.open(fp).convert("L")
            img = img.point(lambda x: 0 if x < 140 else 255, '1')
            return pytesseract.image_to_string(img, lang="por", config="--psm 3")

    except Exception as e:
        print(f"[ERRO] Falha ao extrair texto de {fp}: {e}")
        return ""

# --- Carrega modelo TF-IDF + XGBoost ---
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

# --- BERTimbau como fallback ---
bert_model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")
if os.path.exists("modelo_bert_fallback.pkl"):
    with open("modelo_bert_fallback.pkl", "rb") as f:
        clf_bert, le_bert = pickle.load(f)
else:
    clf_bert, le_bert = LogisticRegression(), le

# --- Configura API Flask ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "tiff"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        filename = secure_filename(uploaded.filename)
        if not filename:
            return jsonify({"error": "Nome de arquivo vazio"}), 400

        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify({"error": "Tipo de arquivo não suportado"}), 400

        tmpdir = tempfile.mkdtemp(prefix="cv_api_")
        path = os.path.join(tmpdir, filename)
        uploaded.save(path)

        texto = limpar_texto(extrair_texto_arquivo(path))
        shutil.rmtree(tmpdir, ignore_errors=True)

        if not texto.strip():
            return jsonify({"error": "Não foi possível extrair texto"}), 400

        # --- Inferência TF-IDF ---
        Xw = word_v.transform([texto])
        Xc = char_v.transform([texto])
        Xchaves = csr_matrix([extrair_features_chave(texto)])
        Xfull = selector.transform(hstack([Xw, Xc, Xchaves]))
        probs = clf.predict_proba(Xfull)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx])
        classe = le.inverse_transform([pred_idx])[0]
        origem = "tfidf"

        # --- Limiar de confiança para fallback BERT ---
        LIMIAR = 0.6
        if confidence < LIMIAR:
            origem = "bert_fallback"
            emb = bert_model.encode([texto])
            try:
                probs_b = clf_bert.predict_proba(emb)[0]
                idx_b = np.argmax(probs_b)
                classe = le_bert.inverse_transform([idx_b])[0]
                confidence = float(probs_b[idx_b])
            except Exception:
                classe, confidence = "INDEFINIDO", 0.0

        return jsonify({
            "success": True,
            "prediction": classe,
            "confidence": round(confidence, 3),
            "origin": origem
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API híbrida rodando 🚀"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
