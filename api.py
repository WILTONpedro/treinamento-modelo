import os
import tempfile
import shutil
import zipfile
import pickle
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack

# Importe suas funções de extração/limpeza do script anterior
from meu_script import (
    limpar_texto,
    processar_item,
    extrair_texto_arquivo
)

app = Flask(__name__)

# Extensões suportadas
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'doc', 'zip'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Carrega modelo e vetorizadores
with open("modelo_curriculos_avancado.pkl", "rb") as f:
    clf, word_v, char_v = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    # validação básica do upload
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    uploaded = request.files["file"]
    filename = secure_filename(uploaded.filename)

    if filename == "":
        return jsonify({"error": "Nome de arquivo vazio"}), 400

    if not allowed_file(filename):
        return jsonify({"error": "Tipo de arquivo não suportado"}), 400

    # salva em uma pasta temporária
    tmpdir = tempfile.mkdtemp(prefix="cv_api_")
    filepath = os.path.join(tmpdir, filename)
    uploaded.save(filepath)

    # extrai texto de todos os itens (incluindo dentro de .zip)
    textos = []
    for pfile, ext in processar_item(filepath):
        txt = extrair_texto_arquivo(pfile)
        if txt:
            textos.append(limpar_texto(txt))

    # remove arquivos temporários
    shutil.rmtree(tmpdir, ignore_errors=True)

    if not textos:
        return jsonify({"error": "Não foi possível extrair texto"}), 400

    full_text = " ".join(textos)

    # vetorização
    Xw = word_v.transform([full_text])
    Xc = char_v.transform([full_text])
    Xv = hstack([Xw, Xc])

    # predição
    pred = clf.predict(Xv)[0]
    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
