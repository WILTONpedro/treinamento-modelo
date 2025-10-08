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

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            "tokens": len(texto.split())
        })

    return jsonify({"success": True, "results": resultados})

@app.route("/", methods=["GET"])
def healthcheck():
    return jsonify({"status": "ok", "message": "API de Currículos rodando 🚀"})

port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)

