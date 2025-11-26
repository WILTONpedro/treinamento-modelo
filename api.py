import os
import re
import pickle
import io  # <--- Para manipular arquivos na memória
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
from contextlib import asynccontextmanager # <--- Para o novo ciclo de vida

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

# --- MACHINE LEARNING ---
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Configuração NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords", quiet=True)
    nltk.download("rslp", quiet=True)

STEMMER = RSLPStemmer()
STOPWORDS = set(stopwords.words("portuguese"))

# ==============================================================================
# 1. CLASSES (CÉREBRO)
# ==============================================================================
# Mantivemos idêntico para não quebrar o pickle

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
# 2. LEITURA DE ARQUIVOS (Versão em Memória - Sem Disco)
# ==============================================================================
def extract_text_from_memory(file_bytes, filename):
    """Lê o arquivo direto da memória RAM, sem salvar no disco."""
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    
    # Cria um objeto de arquivo na memória
    file_stream = io.BytesIO(file_bytes)
    
    try:
        if ext == ".pdf":
            with pdfplumber.open(file_stream) as pdf:
                text = " ".join([p.extract_text() or "" for p in pdf.pages])
        elif ext in [".docx", ".doc"]:
            doc = docx.Document(file_stream)
            text = " ".join([p.text for p in doc.paragraphs])
        elif ext in [".jpg", ".png", ".jpeg"]:
            img = Image.open(file_stream)
            text = pytesseract.image_to_string(img, lang="por")
        elif ext == ".txt":
            text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Erro leitura memória: {e}")
    return text

# ==============================================================================
# 3. CONFIGURAÇÃO DA API (Ciclo de Vida Moderno)
# ==============================================================================

# Dicionário global para guardar o modelo
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executa ao ligar e desligar a API."""
    # --- 1. Carregar Modelo ---
    sys.modules['__main__'] = sys.modules[__name__] # Correção do Pickle
    pickle_path = "cerebro_final.pkl"
    
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                artifacts = pickle.load(f)
            ml_models['model'] = artifacts['model']
            ml_models['preprocessor'] = artifacts['preprocessor']
            ml_models['encoder'] = artifacts['encoder']
            ml_models['engineer'] = FeatureEngineer() # Instancia limpa
            print("✅ CÉREBRO CARREGADO NA MEMÓRIA!")
        except Exception as e:
            print(f"❌ Erro ao carregar pickle: {e}")
    else:
        print("❌ Pickle não encontrado.")
    
    yield # A API roda aqui
    
    # --- 2. Limpeza (ao desligar) ---
    ml_models.clear()

app = FastAPI(title="API Triagem PRO", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 4. ROTA DE TRIAGEM (Sem 'async' para não travar CPU)
# ==============================================================================

@app.post("/triagem")
def triar_curriculo(file: UploadFile = File(...)): # Tirei o 'async' propositalmente
    """
    Processa o currículo. 
    Nota: Função síncrona (def) roda em ThreadPool para não bloquear o servidor.
    """
    
    if 'model' not in ml_models:
        raise HTTPException(status_code=500, detail="Cérebro não está ativo.")

    try:
        # 1. Lê bytes da memória (rápido)
        content = file.file.read()
        
        # 2. Extrai texto sem criar arquivo temp
        raw_text = extract_text_from_memory(content, file.filename)
        clean_text = TextCleaner.clean(raw_text)
        
        if len(raw_text) < 10:
             return {"status": "erro", "mensagem": "Arquivo ilegível ou vazio."}

        # 3. Engenharia de Features
        engineer = ml_models['engineer']
        features_df = engineer.transform([raw_text])
        
        # 4. DataFrame
        input_df = pd.DataFrame({
            'main_text': [clean_text],          
            'raw_text_for_excel': [raw_text], 
            'last_exp': features_df['last_exp_text'],
            'courses': features_df['cursos_text'],
            'career_time': features_df['tempo_carreira']
        })
        
        # 5. Predição
        preprocessor = ml_models['preprocessor']
        model = ml_models['model']
        encoder = ml_models['encoder']
        
        X_input = preprocessor.transform(input_df)
        probs = model.predict_proba(X_input)[0]
        pred_idx = np.argmax(probs)
        
        setor = encoder.inverse_transform([pred_idx])[0]
        confianca = float(probs[pred_idx])
        
        return {
            "arquivo": file.filename,
            "setor_sugerido": setor,
            "confianca": f"{confianca:.2%}",
            "detalhes": {
                "tempo_estimado": int(features_df['tempo_carreira'][0]),
                "tem_cursos": bool(features_df['cursos_text'][0])
            }
        }

    except Exception as e:
        return {"status": "erro", "mensagem": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
