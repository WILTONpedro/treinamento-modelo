import os
import re
import tempfile
import shutil
import zipfile
import pickle
import docx
import pdfplumber
import pytesseract
from PIL import Image
import nltk
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords
import traceback

# --- Preparar NLTK ---
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("portuguese"))

# --- Regex pré-compiladas ---
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-ú\s]")

# --- Função de limpeza de texto ---
def limpar_texto(texto: str) -> str:
    texto = texto.lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

# --- Processamento de arquivos e ZIPs ---
def processar_item(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".zip":
        with zipfile.ZipFile(filepath, "r") as z:
            with tempfile.TemporaryDirectory() as tmpdir:
                z.extractall(tmpdir)
                for name in z.namelist():
                    yield os.path.join(tmpdir, name), os.path.splitext(name)[1].lower()
    else:
        yield filepath, ext

# --- Extração de texto ---
def extrair_texto_arquivo(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            texts = []
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        texts.append(t)
            return " ".join(texts)
        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            return " ".join(p.text for p in doc.paragraphs)
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext in (".png", ".jpg", ".jpeg", ".tiff"):
            img = Image.open(filepath).convert("L")
            img.thumbnail((1024, 1024))  # reduz memória no OCR
            return pytesseract.image_to_string(img, lang="por", config="--psm 6")
    except Exception:
        print(f"[ERRO] Falha ao extrair texto de {filepath}:\n{traceback.format_exc()}")
        return ""
    return ""

# --- Carrega modelo ---
with open("modelo_curriculos_xgb_oversampling.pkl", "rb") as f:
    data = pickle.load(f)

if isinstance(data, dict):
    clf = data["clf"]
    word_v = data["word_vectorizer"]
    char_v = data["char_vectorizer"]
    palavras_chave_dict = data["palavras_chave_dict"]
    selector = data["selector"]
    le = data["label_encoder"]
elif isinstance(data, (tuple, list)):
    clf, word_v, char_v, palavras_chave_dict, selector, le = data
else:
    raise ValueError("Formato do pickle desconhecido. Esperado dict ou tuple.")

# --- Extração de features de palavras-chave ---
def extrair_features_chave(texto):
    return [int(any(p.lower() in texto for p in palavras)) for palavras in palavras_chave_dict.values()]

# --- Flask API ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower().lstrip(".")
    return ext in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "Nenhum arquivo enviado"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"error": "Nome de arquivo vazio"}), 400

        filename = secure_filename(uploaded.filename)
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        if ext not in ALLOWED_EXTENSIONS:
            content_type_ext = uploaded.content_type.split("/")[-1].lower()
            if content_type_ext in ALLOWED_EXTENSIONS:
                ext = content_type_ext
                filename = f"{filename}.{ext}"
            else:
                return jsonify({"error": f"Tipo de arquivo não suportado: {uploaded.filename}"}), 400

        # Cria diretório temporário seguro
        with tempfile.TemporaryDirectory(prefix="cv_api_") as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            uploaded.save(filepath)

            textos = []
            for pfile, _ in processar_item(filepath):
                txt = extrair_texto_arquivo(pfile)
                if txt:
                    textos.append(limpar_texto(txt))

            if not textos:
                return jsonify({"error": "Não foi possível extrair texto"}), 400

            full_text = " ".join(textos)

            # --- Vetorização ---
            Xw = word_v.transform([full_text])
            Xc = char_v.transform([full_text])
            Xchaves = csr_matrix([extrair_features_chave(full_text)])
            Xfull = hstack([Xw, Xc, Xchaves])

            # --- Seleção de features ---
            Xsel = selector.transform(Xfull)

            # --- Predição ---
            pred = clf.predict(Xsel)[0]
            classe = le.inverse_transform([pred])[0]

            return jsonify({
                "success": True,
                "prediction": classe,
                "tokens": len(full_text.split())
            })

    except Exception:
        print(traceback.format_exc())
        return jsonify({"error": "Falha interna no processamento"}), 500

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos rodando 🚀"})

# --- Execução ---
if __name__ == "__main__":
    # Debug desligado para produção no Render
    app.run(host="0.0.0.0", port=5000)port)
