from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Carregar modelo treinado
with open("modelo_curriculos.pkl", "rb") as f:
    clf, vectorizer = pickle.load(f)

@app.route("/classificar", methods=["POST"])
def classificar():
    data = request.get_json()
    texto = data.get("texto", "")
    if not texto.strip():
        return jsonify({"erro": "Texto vazio"}), 400
    
    x = vectorizer.transform([texto])
    vaga = clf.predict(x)[0]
    
    return jsonify({"vaga": vaga})

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)