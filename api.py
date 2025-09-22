from flask import Flask, request, jsonify
import pickle
import pdfplumber
from io import BytesIO
import base64

app = Flask(__name__)

# Carregar modelo treinado
with open("modelo_curriculos.pkl", "rb") as f:
    clf, vectorizer = pickle.load(f)

# Função para extrair texto do PDF
def extrair_texto(pdf_bytesio):
    texto = ""
    try:
        with pdfplumber.open(pdf_bytesio) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    texto += page_text + "\n"
    except Exception as e:
        return f"ERRO: {e}"
    return texto

@app.route("/classificar", methods=["POST"])
def classificar():
    data = request.get_json()
    pdf_base64 = data.get("pdf_base64")

    if not pdf_base64:
        return jsonify({"erro": "PDF não enviado"}), 400

    try:
        pdf_bytes = BytesIO(base64.b64decode(pdf_base64))
    except Exception as e:
        return jsonify({"erro": f"Erro ao decodificar PDF: {e}"}), 400

    texto = extrair_texto(pdf_bytes)

    if not texto.strip():
        return jsonify({"erro": "Texto vazio"}), 400

    x = vectorizer.transform([texto])
    vaga = clf.predict(x)[0]

    return jsonify({"vaga": vaga})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
