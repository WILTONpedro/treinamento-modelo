from flask import Flask, request, jsonify
import pickle
import os
import re
from docx import Document
import pdfplumber
import pytesseract
from PIL import Image
import io

app = Flask(__name__)

# Carregar modelo treinado
with open("modelo_curriculos.pkl", "rb") as f:
    clf, vectorizer = pickle.load(f)

def extrair_texto_pdf(file_bytes):
    texto = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if not txt:
                # PDF como imagem, usa OCR
                imagem = page.to_image(resolution=300).original
                txt = pytesseract.image_to_string(imagem, lang="por")
            texto += txt + " "
    return texto

def extrair_texto_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    texto = "\n".join([p.text for p in doc.paragraphs])
    return texto

def limpar_texto(texto):
    texto = texto.lower().strip()
    texto = re.sub(r'\s+', ' ', texto)         # Remove múltiplos espaços
    texto = re.sub(r'[^\w\s]', '', texto)      # Remove caracteres especiais
    return texto

@app.route("/classificar", methods=["POST"])
def classificar():
    data = request.get_json()

    pdf_base64 = data.get("pdf_base64")
    if not pdf_base64:
        return jsonify({"erro": "Nenhum arquivo recebido"}), 400

    # Decodifica PDF ou DOCX do Base64
    file_bytes = io.BytesIO(base64.b64decode(pdf_base64)).read()

    # Detecta tipo (PDF ou DOCX)
    tipo = data.get("tipo", "pdf").lower()

    try:
        if tipo == "pdf":
            texto = extrair_texto_pdf(file_bytes)
        elif tipo == "docx":
            texto = extrair_texto_docx(file_bytes)
        else:
            return jsonify({"erro": "Formato não suportado"}), 400

        texto_limpo = limpar_texto(texto)

        if not texto_limpo.strip():
            return jsonify({"erro": "Texto vazio"}), 400

        x = vectorizer.transform([texto_limpo])
        vaga = clf.predict(x)[0]

        return jsonify({"vaga": vaga})

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
