<<<<<<< HEAD
import os
import re
import tempfile
import shutil
import pickle
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack, csr_matrix
import docx
import pdfplumber
from PIL import Image
import pytesseract
=======
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

def extrair_texto_arquivo(fp):
    ext = os.path.splitext(fp)[1].lower()
    try:
        if ext == ".pdf":
            texto = ""
            with pdfplumber.open(fp) as pdf:
                texto = " ".join(p.extract_text() or "" for p in pdf.pages)
            if not texto.strip():
                # OCR fallback
                imagens = convert_from_path(fp, dpi=150)
                partes = []
                for img in imagens:
                    partes.append(pytesseract.image_to_string(img, lang="por", config="--psm 6"))
                return " ".join(partes)
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
        print("erro extrair:", e)
        return ""
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
>>>>>>> 1f90bbe (Salvar alterações pendentes antes do rebase)

# --- Configurações ---
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# --- Funções auxiliares ---
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-zá-ú\s]", " ", texto)
    palavras = [p for p in texto.split()]
    return " ".join(palavras)

def extrair_texto_arquivo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                return " ".join([p.extract_text() or "" for p in pdf.pages])
        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            return " ".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext in ("png", "jpg", "jpeg", "tiff"):
            img = Image.open(filepath)
            img = img.convert("L")
            texto = pytesseract.image_to_string(img, lang="por", config="--psm 6")
            img.close()
            return texto
    except Exception as e:
        print(f"[ERRO] {filepath}: {e}")
    return ""

# --- Carrega modelo ---
with open("modelo_curriculos_super_avancado.pkl", "rb") as f:
    data = pickle.load(f)

clf = data["clf"]
word_vectorizer = data["word_vectorizer"]
char_vectorizer = data["char_vectorizer"]
palavras_chave_dict = data["palavras_chave_dict"]
selector = data["selector"]
le = data["label_encoder"]

def extrair_features_chave(texto):
    return [int(any(p.lower() in texto for p in palavras)) for palavras in palavras_chave_dict.values()]

# --- Flask API ---
app = Flask(__name__)
ALLOWED_EXT = {"pdf", "docx", "doc", "txt", "png", "jpg", "jpeg", "tiff"}

<<<<<<< HEAD
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Rota de teste (GET) ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "message": "API de triagem ativa e rodando!"})

# --- Rota principal de predição ---
@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    arquivos = request.files.getlist("files")
    resultados = []

    for uploaded in arquivos:
        nome = secure_filename(uploaded.filename)
        if not nome:
            resultados.append({"filename": nome, "error": "Nome de arquivo vazio"})
            continue

        if not allowed_file(nome):
            resultados.append({"filename": nome, "error": "Tipo de arquivo não suportado"})
            continue

        uploaded.seek(0, os.SEEK_END)
        tamanho = uploaded.tell()
        uploaded.seek(0)
        if tamanho > MAX_FILE_SIZE:
            resultados.append({"filename": nome, "error": "Arquivo maior que 10MB"})
            continue

        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, nome)
        uploaded.save(filepath)

        texto = extrair_texto_arquivo(filepath)
        shutil.rmtree(tmpdir, ignore_errors=True)

        if not texto.strip():
            resultados.append({"filename": nome, "error": "Não foi possível extrair texto"})
            continue

        # Vetorização e features
        Xw = word_vectorizer.transform([texto])
        Xc = char_vectorizer.transform([texto])
        Xch = csr_matrix([extrair_features_chave(texto)])
        Xfull = selector.transform(hstack([Xw, Xc, Xch]))

        pred = clf.predict(Xfull)[0]
        classe = le.inverse_transform([pred])[0]

        resultados.append({
        "filename": nome,
        "prediction": classe,
        "tokens": len(texto.split()),
        "preview": texto[:200]  # mostra só os primeiros 200 caracteres
})
    return jsonify({"success": True, "results": resultados})

# --- Rota auxiliar para o Apps Script (POST simples) ---
@app.route("/analisar", methods=["POST"])
def analisar():
    try:
        dados = request.get_json()
        return jsonify({
            "mensagem": "Recebido com sucesso!",
            "dados": dados,
            "status": "ok"
        })
    except Exception as e:
        return jsonify({"erro": str(e)}), 400

# --- Execução ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

=======
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
>>>>>>> 1f90bbe (Salvar alterações pendentes antes do rebase)
