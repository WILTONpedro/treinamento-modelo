import os
import tempfile
import traceback
import pickle
import gc
import re
import numpy as np
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from scipy.sparse import hstack
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import pdfplumber
import docx
import nltk

# -------------------------
# CONFIGURAÇÕES
# -------------------------
MODEL_UNIFICADO_PATH = "modelo_unificado.pkl"
MODEL_HASHING_PATH = "modelo_hashing_foco.pkl"

# Caminho do Tesseract (ajuste se mudar para Linux/Render, ex: '/usr/bin/tesseract')
TESSERACT_CMD = r"C:\Users\Usuario\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
if os.name != 'nt': # Se estiver no Render (Linux)
    TESSERACT_CMD = '/usr/bin/tesseract'

PORT = int(os.environ.get("PORT", 5000))
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt', 'png', 'jpg', 'jpeg'}

# Configuração Tesseract
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Baixar stopwords se necessário
try:
    nltk.data.find("corpora/stopwords")
except Exception:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("portuguese"))

# Regex para limpeza
RE_EMAIL = re.compile(r"\S+@\S+")
RE_NUM = re.compile(r"\d+")
RE_CARACT = re.compile(r"[^a-zá-úçâêîôûãõ\s]")

# -------------------------
# 1. FUNÇÕES DE MELHORIA DE IMAGEM (NOVO)
# -------------------------
def melhorar_imagem_para_ocr(img):
    """
    Aplica filtros para facilitar a leitura do Tesseract.
    Converte para cinza, aumenta contraste e nitidez.
    """
    img = img.convert("L") # Escala de cinza
    
    # Aumentar contraste (ajuda a separar letra do fundo)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.8)
    
    # Aumentar nitidez
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)
    
    # Binarização simples (Opcional, remove ruído cinza claro)
    # img = img.point(lambda x: 0 if x < 128 else 255, '1')
    
    return img

def validar_texto_util(texto):
    """Verifica se o texto extraído parece ser linguagem natural ou lixo (encoding erro)."""
    if not texto or len(texto) < 50:
        return False
    
    # Conta quantas palavras reais (stopwords) existem no texto
    # Se não tiver "de", "para", "com", provavelmente é lixo ou imagem mal convertida
    palavras = texto.lower().split()
    contagem = sum(1 for p in palavras if p in STOPWORDS)
    
    # Se menos de 2% das palavras forem conectivos comuns, suspeite da extração
    ratio = contagem / len(palavras) if len(palavras) > 0 else 0
    return ratio > 0.02

# -------------------------
# 2. EXTRAÇÃO DE TEXTO INTELIGENTE (ATUALIZADO)
# -------------------------
def extrair_texto_smart(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    texto = ""
    metodo = "nativo"

    try:
        # IMAGENS
        if ext in (".png", ".jpg", ".jpeg", ".tiff"):
            img = Image.open(filepath)
            img = melhorar_imagem_para_ocr(img)
            texto = pytesseract.image_to_string(img, lang="por", config="--psm 6")
            img.close()
            metodo = "ocr_imagem"

        # PDF (Lógica Híbrida)
        elif ext == ".pdf":
            try:
                # Tentativa 1: Texto Digital
                with pdfplumber.open(filepath) as pdf:
                    paginas = [p.extract_text() or "" for p in pdf.pages]
                    texto = "\n".join(paginas)
                
                # Validação: O texto digital presta?
                if not validar_texto_util(texto):
                    print(f"⚠️ Texto digital do PDF ruim ou vazio. Forçando OCR em: {filepath}")
                    # Tentativa 2: Forçar OCR no PDF (Converte paginas em imagens)
                    # Nota: Requer poppler instalado. Se não tiver, essa parte pula.
                    try:
                        from pdf2image import convert_from_path
                        images = convert_from_path(filepath, dpi=200, last_page=2) # Só 2 paginas para poupar RAM
                        texto_ocr = []
                        for img in images:
                            img = melhorar_imagem_para_ocr(img)
                            texto_ocr.append(pytesseract.image_to_string(img, lang="por"))
                        texto = "\n".join(texto_ocr)
                        metodo = "ocr_pdf_fallback"
                    except ImportError:
                        metodo = "falha_ocr_lib_missing"
                    except Exception as e:
                        print(f"Erro no OCR do PDF: {e}")
            except Exception as e:
                print(f"Erro leitura PDF: {e}")

        # DOCX / DOC
        elif ext in (".docx", ".doc"):
            doc = docx.Document(filepath)
            texto = "\n".join([p.text for p in doc.paragraphs])
            metodo = "docx_xml"

        # TXT
        elif ext == ".txt":
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                texto = f.read()
            metodo = "txt_raw"

    except Exception:
        traceback.print_exc()
        return "", "erro"

    return (texto or ""), metodo

# -------------------------
# LIMPEZA E MODELOS (MANTIDOS DA SUA VERSÃO)
# -------------------------
def limpar_texto(texto: str) -> str:
    texto = (texto or "").lower()
    texto = RE_EMAIL.sub(" ", texto)
    texto = RE_NUM.sub(" ", texto)
    texto = RE_CARACT.sub(" ", texto)
    return " ".join(w for w in texto.split() if w not in STOPWORDS)

def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

print("⏳ Carregando modelos...")
MODEL_UNIFICADO = load_model(MODEL_UNIFICADO_PATH)
MODEL_HASHING = load_model(MODEL_HASHING_PATH)

# Extração dos componentes dos pickles
clf_u, wv_u, cv_u = MODEL_UNIFICADO["clf"], MODEL_UNIFICADO["word_vectorizer"], MODEL_UNIFICADO["char_vectorizer"]
sel_u, le_u = MODEL_UNIFICADO.get("selector", None), MODEL_UNIFICADO["label_encoder"]

clf_h, wv_h, cv_h = MODEL_HASHING["clf"], MODEL_HASHING["word_vectorizer"], MODEL_HASHING["char_vectorizer"]
sel_h, le_h = MODEL_HASHING.get("selector", None), MODEL_HASHING["label_encoder"]

all_classes = list(dict.fromkeys(list(le_u.classes_) + list(le_h.classes_)))

# -------------------------
# LÓGICA DE PREDIÇÃO
# -------------------------
def predict_with_model(text, clf, wv, cv, selector, le):
    text_clean = limpar_texto(text)
    # Proteção para texto vazio pós-limpeza
    if not text_clean.strip():
        return None

    Xw = wv.transform([text_clean])
    Xc = cv.transform([text_clean])
    Xfull = hstack([Xw, Xc])
    
    Xsel = selector.transform(Xfull) if selector else Xfull
    
    probs = clf.predict_proba(Xsel)[0]
    pred_idx = int(np.argmax(probs))
    
    # Métricas de gap
    sorted_probs = sorted(probs, reverse=True)
    top = float(sorted_probs[0])
    second = float(sorted_probs[1]) if len(sorted_probs) > 1 else 0.0
    
    return {
        "pred_label": le.inverse_transform([pred_idx])[0],
        "top_prob": top,
        "gap": top - second,
        "probs": {cls: float(p) for cls, p in zip(le.classes_, probs)}
    }

def merge_predictions(res_u, res_h):
    # Se algum modelo falhou (texto vazio), usa o outro
    if not res_u and not res_h: return None
    if not res_u: return {**res_h, "winner_model": "hashing_only"}
    if not res_h: return {**res_u, "winner_model": "unificado_only"}

    # Lógica Original (Consenso e Gaps)
    GAP_SIGNIFICANT = 0.10
    
    if res_u["pred_label"] == res_h["pred_label"]:
        return {
            "final_prediction": res_u["pred_label"],
            "final_confidence": max(res_u["top_prob"], res_h["top_prob"]),
            "reason": "agreement",
            "winner_model": "both"
        }

    # Desempate por confiança agressiva
    if (res_u["gap"] - res_h["gap"]) >= GAP_SIGNIFICANT:
        return {"final_prediction": res_u["pred_label"], "final_confidence": res_u["top_prob"], "reason": "unificado_strong_gap", "winner_model": "unificado"}
    
    if (res_h["gap"] - res_u["gap"]) >= GAP_SIGNIFICANT:
        return {"final_prediction": res_h["pred_label"], "final_confidence": res_h["top_prob"], "reason": "hashing_strong_gap", "winner_model": "hashing"}

    # Empate técnico: Média das probabilidades
    avg_probs = {}
    for cls in all_classes:
        p_u = res_u["probs"].get(cls, 0.0)
        p_h = res_h["probs"].get(cls, 0.0)
        avg_probs[cls] = (p_u + p_h) / 2.0
    
    final_cls = max(avg_probs.items(), key=lambda x: x[1])[0]
    
    return {
        "final_prediction": final_cls,
        "final_confidence": avg_probs[final_cls],
        "reason": "weighted_average",
        "winner_model": "average"
    }

# -------------------------
# FLASK APP
# -------------------------
app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    filepath = None
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "Arquivo não enviado"}), 400

        uploaded = request.files["file"]
        if uploaded.filename == "":
            return jsonify({"success": False, "error": "Arquivo sem nome"}), 400

        filename = secure_filename(uploaded.filename)
        if not allowed_file(filename):
            return jsonify({"success": False, "error": "Extensão não permitida"}), 400

        # Cria pasta temp e processa
        with tempfile.TemporaryDirectory(prefix="cv_processing_") as tmpdir:
            filepath = os.path.join(tmpdir, filename)
            uploaded.save(filepath)

            # 1. Extração Melhorada
            texto_extraido, metodo_extracao = extrair_texto_smart(filepath)
            
            # Recuperar campos extras
            assunto = request.form.get("assunto") or ""
            corpo = request.form.get("corpo") or ""
            texto_final = f"{assunto} {corpo} {texto_extraido}".strip()

            # 2. Validação de Conteúdo (Evitar classificar lixo)
            if len(texto_final) < 20:
                return jsonify({
                    "success": False, 
                    "error": "Conteúdo insuficiente para classificação",
                    "details": {"extracted_length": len(texto_extraido), "method": metodo_extracao}
                }), 422

            # 3. Predição
            res_u = predict_with_model(texto_final, clf_u, wv_u, cv_u, sel_u, le_u)
            res_h = predict_with_model(texto_final, clf_h, wv_h, cv_h, sel_h, le_h)
            
            merged = merge_predictions(res_u, res_h)

            # 4. Resposta Enriquecida
            response = {
                "success": True,
                "prediction": merged["final_prediction"],
                "confidence": round(merged["final_confidence"], 4),
                "meta": {
                    "extraction_method": metodo_extracao,
                    "text_length": len(texto_final),
                    "model_reason": merged["reason"]
                },
                "debug": {
                    "prob_unificado": res_u["top_prob"] if res_u else 0,
                    "prob_hashing": res_h["top_prob"] if res_h else 0
                }
            }
            
            # Limpeza de memória explícita
            del texto_extraido, texto_final, res_u, res_h, merged
            gc.collect()
            
            return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"Erro interno: {str(e)}"}), 500

if __name__ == "__main__":
    # threaded=True ajuda a não bloquear requisições enquanto processa OCR
    app.run(host="0.0.0.0", port=PORT, threaded=True)
