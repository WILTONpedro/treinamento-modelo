from flask import Flask, request, jsonify
import os
import pickle
import re
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
import unicodedata
import logging

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Usuario\PROJETOpypy\.venv\Scripts\pytesseract.exe"
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

app = Flask(__name__)

# -----------------------
# Carregar modelo
# -----------------------
with open("modelo_curriculos_avancado.pkl", "rb") as f:
    clf, vectorizer = pickle.load(f)
logging.info("Modelo carregado com sucesso.")

# -----------------------
# Funções utilitárias
# -----------------------
def remover_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'(http[s]?://\S+)|(\S+@\S+)|([^\w\s])', ' ', texto)
    texto = remover_acentos(texto)
    return texto.strip()

def extrair_texto_pdf(path):
    texto = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if not txt or txt.strip() == "":
                imagem = page.to_image(resolution=300).original
                txt = pytesseract.image_to_string(imagem, lang="por")
            texto += txt + " "
    return texto

def extrair_texto_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def extrair_texto_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extrair_texto_arquivo(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extrair_texto_pdf(path)
    elif ext in [".docx", ".doc"]:
        return extrair_texto_docx(path)
    elif ext == ".txt":
        return extrair_texto_txt(path)
    else:
        return None

# -----------------------
# Rota principal
# -----------------------
@app.route("/classificar", methods=["POST"])
def classificar_curriculo():
    if 'arquivo' not in request.files:
        return jsonify({"erro": "Nenhum arquivo enviado"}), 400

    arquivo = request.files['arquivo']
    if arquivo.filename == "":
        return jsonify({"erro": "Nome do arquivo vazio"}), 400

    # Salvar temporariamente
    temp_path = os.path.join("temp", arquivo.filename)
    os.makedirs("temp", exist_ok=True)
    arquivo.save(temp_path)

    try:
        texto = extrair_texto_arquivo(temp_path)
        if not texto or texto.strip() == "":
            return jsonify({"erro": "Não foi possível extrair texto do arquivo"}), 400
        texto = limpar_texto(texto)
        X_vect = vectorizer.transform([texto])
        predicao = clf.predict(X_vect)[0]
        return jsonify({"vaga": predicao})
    except Exception as e:
        logging.error(f"Erro na classificação: {e}")
        return jsonify({"erro": str(e)}), 500
    finally:
        os.remove(temp_path)

# -----------------------
# Inicialização
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
