from flask import Flask, request, jsonify
import pickle
import pdfplumber
import pytesseract
from PIL import Image
import os
app = Flask(__name__)
# Carregar modelo treinado
with open("modelo_curriculos.pkl", "rb") as f:
    clf, vectorizer = pickle.load(f)

# Caminho do Tesseract (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Função para extrair texto de PDF
def extrair_texto(pdf_path):
    texto = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texto += page_text + "\n"
        if not texto.strip():  # se pdf for imagem, usar OCR
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    pil_image = page.to_image(resolution=300).original
                    texto += pytesseract.image_to_string(pil_image)
    except Exception as e:
        return f"ERRO: {e}"
    return texto
@app.route("/classificar", methods=["POST"])
def classificar():
    data = request.get_json()
    pdf_base64 = data.get("pdf_base64")
    if not pdf_base64:
        return jsonify({"erro": "PDF não enviado"}), 400
    import base64
    from io import BytesIO
    pdf_bytes = BytesIO(base64.b64decode(pdf_base64))
    texto = extrair_texto(pdf_bytes)
    if not texto.strip():
        return jsonify({"erro": "Texto vazio"}), 400
    x = vectorizer.transform([texto])
    vaga = clf.predict(x)[0]
    return jsonify({"vaga": vaga})
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
