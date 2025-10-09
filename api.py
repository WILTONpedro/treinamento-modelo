import os
import re
import tempfile
import shutil
import pickle
import gc
import torch
import pdfplumber
import docx
from PIL import Image
import pytesseract
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer

# --- Configurações ---
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip', 'png', 'jpg', 'jpeg', 'tiff'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# --- Inicialização do Flask ---
app = Flask(__name__)

# --- Funções auxiliares ---
def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"\S+@\S+", " ", texto)
    texto = re.sub(r"\d+", " ", texto)
    texto = re.sub(r"[^a-zá-ú\s]", " ", texto)
    return " ".join(p for p in texto.split() if len(p) > 2)

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
        print(f"[ERRO EXTRAÇÃO] {e}")
    return ""

# --- Lazy loading global (evita travar o Render na inicialização) ---
clf = None
word_vectorizer = None
char_vectorizer = None
selector = None
le = None
emb_model = None
palavras_chave_dict = None

def carregar_modelos():
    global clf, word_vectorizer, char_vectorizer, selector, le, emb_model, palavras_chave_dict
    if clf is None:
        print("🔹 Carregando modelos na memória...")
        with open("modelo_curriculos_super_avancado.pkl", "rb") as f:
            data = pickle.load(f)

        clf = data["clf"]
        word_vectorizer = data["word_vectorizer"]
        char_vectorizer = data["char_vectorizer"]
        selector = data["selector"]
        le = data["label_encoder"]
        palavras_chave_dict = data["palavras_chave_dict"]

        # Load SentenceTransformer (com torch leve)
        device = "cpu"
        emb_model = SentenceTransformer("modelo_emb_dir", device=device)
        emb_model.max_seq_length = 128  # corte curto para RAM de 512MB
        print("✅ Modelos carregados com sucesso.")

def extrair_features_chave(texto):
    return [int(any(p.lower() in texto for p in palavras)) for palavras in palavras_chave_dict.values()]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Rotas ---
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "online", "message": "API de triagem otimizada para Render"})

@app.route("/predict", methods=["POST"])
def predict():
    carregar_modelos()

    if "files" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    arquivos = request.files.getlist("files")
    resultados = []

    for uploaded in arquivos:
        nome = secure_filename(uploaded.filename)
        if not nome or not allowed_file(nome):
            resultados.append({"filename": nome, "error": "Tipo de arquivo inválido"})
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
            resultados.append({"filename": nome, "error": "Texto vazio"})
            continue

        texto_limpo = limpar_texto(texto)

        # --- Vetorização tradicional ---
        Xw = word_vectorizer.transform([texto_limpo])
        Xc = char_vectorizer.transform([texto_limpo])
        Xch = csr_matrix([extrair_features_chave(texto_limpo)])

        # --- Embeddings (CPU only) ---
        with torch.no_grad():
            Xemb = emb_model.encode([texto_limpo], convert_to_tensor=False, show_progress_bar=False)
        Xemb = csr_matrix(Xemb)

        # --- Combinação e seleção ---
        Xfull = selector.transform(hstack([Xw, Xc, Xch, Xemb]))

        # --- Predição ---
        pred = clf.predict(Xfull)[0]
        classe = le.inverse_transform([pred])[0]

        resultados.append({
            "filename": nome,
            "prediction": classe,
            "tokens": len(texto_limpo.split())
        })

        # Limpeza para evitar memory leak
        del Xw, Xc, Xch, Xemb, Xfull
        gc.collect()

    return jsonify({"success": True, "results": resultados})

# --- Execução ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
