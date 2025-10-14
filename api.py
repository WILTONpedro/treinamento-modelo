import pickle
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import scipy.sparse as sp

# === Inicialização da API ===
app = Flask(__name__)
CORS(app)

# === Carrega os modelos ===
def carregar_modelo(caminho):
    with open(caminho, "rb") as f:
        return pickle.load(f)

print("🔹 Carregando modelos...")
modelo_padrao = carregar_modelo("modelo_curriculos_xgb.pkl")
modelo_over = carregar_modelo("modelo_curriculos_xgb_oversampling.pkl")
print("✅ Modelos carregados com sucesso.")

# === Extrai componentes ===
clf = modelo_padrao["clf"]
word_vectorizer = modelo_padrao["word_vectorizer"]
char_vectorizer = modelo_padrao["char_vectorizer"]
selector = modelo_padrao["selector"]
le = modelo_padrao["label_encoder"]

clf_over = modelo_over["clf"]
word_vectorizer_over = modelo_over["word_vectorizer"]
char_vectorizer_over = modelo_over["char_vectorizer"]
selector_over = modelo_over["selector"]
le_over = modelo_over["label_encoder"]

# Palavras-chave (se existirem)
palavras_chave_dict = modelo_over.get("palavras_chave_dict", {})

# === Função para processar texto ===
def preprocessar_texto(texto):
    texto = texto.lower()
    texto = texto.replace("\n", " ").replace("\r", " ")
    return texto

# === Função principal de predição ===
def prever_fusao(texto):
    texto_proc = preprocessar_texto(texto)
    
    # Vetorização base
    Xw = word_vectorizer.transform([texto_proc])
    Xc = char_vectorizer.transform([texto_proc])
    X = sp.hstack([Xw, Xc])

    # === Modelo principal ===
    Xsel = selector.transform(X)
    probs_main = clf.predict_proba(Xsel)[0]
    pred_main = np.argmax(probs_main)
    conf_main = np.max(probs_main)
    classe_main = le.inverse_transform([pred_main])[0]

    # === Modelo oversampling ===
    Xwo = word_vectorizer_over.transform([texto_proc])
    Xco = char_vectorizer_over.transform([texto_proc])
    Xover = sp.hstack([Xwo, Xco])

    try:
        # tenta aplicar selector normalmente
        Xsel_o = selector_over.transform(Xover)
    except Exception as e:
        print(f"[AVISO] Selector oversampling incompatível, ajustando manualmente. Detalhes: {e}")
        try:
            expected_features = clf_over.get_booster().num_features()
        except Exception:
            expected_features = 5000  # fallback

        current_features = Xover.shape[1]

        if current_features > expected_features:
            Xsel_o = Xover[:, :expected_features]
        elif current_features < expected_features:
            diff = expected_features - current_features
            zeros = sp.csr_matrix((Xover.shape[0], diff))
            Xsel_o = sp.hstack([Xover, zeros])
        else:
            Xsel_o = Xover

    probs_over = clf_over.predict_proba(Xsel_o)[0]
    pred_over = np.argmax(probs_over)
    conf_over = np.max(probs_over)
    classe_over = le_over.inverse_transform([pred_over])[0]

    # === Fusão dos resultados ===
    if conf_over >= 0.7:
        classe_final = classe_over
        conf_final = conf_over
        origem = "oversampling"
    else:
        classe_final = classe_main
        conf_final = conf_main
        origem = "modelo_principal"

    # === Palavras-chave ===
    palavras_chave_encontradas = []
    for classe, palavras in palavras_chave_dict.items():
        for p in palavras:
            if p.lower() in texto_proc:
                palavras_chave_encontradas.append(p)
    palavras_chave_encontradas = list(set(palavras_chave_encontradas))

    return {
        "prediction_main": classe_main,
        "confidence_main": float(conf_main),
        "prediction_over": classe_over,
        "confidence_over": float(conf_over),
        "prediction_final": classe_final,
        "confidence_final": float(conf_final),
        "origem_resultado": origem,
        "palavras_chave": palavras_chave_encontradas
    }

# === Rota de predição ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        full_text = data.get("text", "")

        if not full_text.strip():
            return jsonify({"error": "Texto vazio recebido."}), 400

        resultado = prever_fusao(full_text)
        return jsonify(resultado)

    except Exception as e:
        print("❌ ERRO INTERNO:", traceback.format_exc())
        return jsonify({"error": "Falha interna no processamento"}), 500

# === Rota de status ===
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API de Classificação de Currículos operacional ✅"})

# === Execução local ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
