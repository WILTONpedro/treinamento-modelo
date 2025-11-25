import os
import re
import pickle
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# --- SERVER & API ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- PROCESSAMENTO ---
import pdfplumber
import docx
from PIL import Image
import pytesseract
import nltk
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

# --- MACHINE LEARNING (Necessário para carregar o pickle) ---
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

# Configurações Iniciais
warnings.filterwarnings("ignore")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("rslp", quiet=True)

STEMMER = RSLPStemmer()
STOPWORDS = set(stopwords.words("portuguese"))

class TextCleaner:
    @staticmethod
    def clean(text, stem=True):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'\S*@\S*\s?', '', text) 
        text = re.sub(r'http\S+', '', text)    
        text = re.sub(r'[^\w\s]', ' ', text)   
        text = re.sub(r'\d+', ' ', text)       
        tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
        if stem:
            tokens = [STEMMER.stem(t) for t in tokens]
        return " ".join(tokens)

class FolderIntelligence:
    # Mesmo não usando no predict, precisa estar aqui se foi salvo no pickle
    pass 

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = pd.DataFrame({'texto': X})
        df['tempo_carreira'] = df['texto'].apply(self._extract_career_years)
        df['last_exp_text'] = df['texto'].apply(self._extract_last_exp).apply(lambda x: TextCleaner.clean(x))
        df['cursos_text'] = df['texto'].apply(self._extract_courses).apply(lambda x: TextCleaner.clean(x))
        return df[['tempo_carreira', 'last_exp_text', 'cursos_text']]

    def _extract_career_years(self, text):
        matches = re.findall(r"(\d{4})\s*(?:-|at[ée]|presente|atual)", text, re.IGNORECASE)
        if not matches: return 0
        try:
            years = sorted([int(m) for m in matches if 1980 < int(m) <= datetime.now().year])
            if len(years) >= 2: return min(years[-1] - years[0], 40)
        except: pass
        return 0

    def _extract_last_exp(self, text):
        split = re.split(r"experi[êe]ncia|profissional", text, flags=re.IGNORECASE)
        return split[1][:800] if len(split) > 1 else ""

    def _extract_courses(self, text):
        matches = re.findall(r".{0,40}(?:curso|certificação|formação).{0,80}", text, flags=re.IGNORECASE)
        return " ".join(matches)

class SpreadsheetEnforcer(BaseEstimator, TransformerMixin):
    def __init__(self, excel_path=None):
        self.excel_path = excel_path
        self.keywords_map = {} 
    def fit(self, X, y=None): return self
    def transform(self, X):
        # A mágica: O keywords_map já vem preenchido de dentro do pickle!
        if not self.keywords_map:
            return pd.DataFrame(np.zeros((len(X), 1)), columns=['dummy_score'])
        data = []
        for text in X:
            text_lower = text.lower()
            row = {}
            for setor, palavras in self.keywords_map.items():
                score = sum(1 for p in palavras if p in text_lower)
                row[f'score_manual_{setor}'] = score
            data.append(row)
        return pd.DataFrame(data)

# ==============================================================================
# 2. FUNÇÃO DE EXTRAÇÃO DE TEXTO (Para ler o arquivo que chega)
# ==============================================================================
def extract_text_robust(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    text = ""
    try:
        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                text = " ".join([p.extract_text() or "" for p in pdf.pages])
        elif ext in [".docx", ".doc"]:
            doc = docx.Document(filepath)
            text = " ".join([p.text for p in doc.paragraphs])
        elif ext in [".jpg", ".png", ".jpeg"]:
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img, lang="por")
        elif ext == ".txt":
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    except Exception as e:
        print(f"Erro leitura: {e}")
    return text

# ==============================================================================
# 3. CONFIGURAÇÃO DA API FASTAPI
# ==============================================================================

app = FastAPI(title="API Triagem de Currículos")

# Permitir conexões de qualquer lugar (útil para AppScript)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis globais para o modelo
MODEL = None
PREPROCESSOR = None
ENCODER = None
ENGINEER = None

    @app.on_event("startup")
    def load_brain():
    """Carrega o cérebro na memória quando a API liga."""
    global MODEL, PREPROCESSOR, ENCODER, ENGINEER
    
    # --- INICIO DA CORREÇÃO ---
    import sys
    # Truque: Faz o Pickle achar que este arquivo (api.py) é o criador original (__main__)
    sys.modules['__main__'] = sys.modules[__name__]
    # --- FIM DA CORREÇÃO ---

    pickle_path = "cerebro_final.pkl" 
    
    if os.path.exists(pickle_path):
    """Carrega o cérebro na memória quando a API liga."""
    global MODEL, PREPROCESSOR, ENCODER, ENGINEER
    
    pickle_path = "cerebro_final.pkl" # O arquivo gerado pelo brain_auto.py
    
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            artifacts = pickle.load(f)
        
        MODEL = artifacts['model']
        PREPROCESSOR = artifacts['preprocessor']
        ENCODER = artifacts['encoder']
        # Instancia a classe de engenharia
        ENGINEER = FeatureEngineer()
        
        print("✅ Cérebro carregado com sucesso! Pronto para triagem.")
    else:
        print("❌ ERRO: 'cerebro_final.pkl' não encontrado. Treine o modelo primeiro!")

@app.post("/triagem")
async def triar_curriculo(file: UploadFile = File(...)):
    """Recebe um arquivo e retorna a qual setor ele pertence."""
    
    if not MODEL:
        raise HTTPException(status_code=500, detail="Modelo não carregado.")

    # 1. Salvar arquivo temporariamente
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Extrair Texto
        raw_text = extract_text_robust(temp_filename)
        clean_text = TextCleaner.clean(raw_text)
        
        if len(raw_text) < 10:
             return {"status": "erro", "mensagem": "Arquivo vazio ou ilegível (Imagem ruim? PDF protegido?)"}

        # 3. Engenharia de Features (Exatamente como no treino)
        # O engineer extrai as colunas auxiliares (exp, cursos, tempo)
        features_df = ENGINEER.transform([raw_text])
        
        # 4. Montar o DataFrame Final para o Preprocessor
        # A ordem e nome das colunas devem ser IDÊNTICOS ao treino
        input_df = pd.DataFrame({
            'main_text': [clean_text],          
            'raw_text_for_excel': [raw_text], # O Enforcer usa isso para buscar palavras-chave
            'last_exp': features_df['last_exp_text'],
            'courses': features_df['cursos_text'],
            'career_time': features_df['tempo_carreira']
        })
        
        # 5. Transformação e Predição
        X_input = PREPROCESSOR.transform(input_df)
        
        # Pega a probabilidade de cada classe
        probs = MODEL.predict_proba(X_input)[0]
        # Pega o índice da maior probabilidade
        pred_idx = np.argmax(probs)
        # Pega o nome da classe usando o Encoder
        setor_sugerido = ENCODER.inverse_transform([pred_idx])[0]
        confianca = float(probs[pred_idx])
        
        # 6. Resultado
        return {
            "arquivo": file.filename,
            "setor_sugerido": setor_sugerido,
            "confianca": f"{confianca:.2%}",
            "detalhes": {
                "tempo_estimado": int(features_df['tempo_carreira'][0]),
                "tem_cursos": bool(features_df['cursos_text'][0])
            }
        }

    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}
    
    finally:
        # Limpa o arquivo temporário para não encher o disco
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    # Roda o servidor localmente na porta 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
