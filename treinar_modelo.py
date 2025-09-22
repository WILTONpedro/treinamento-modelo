import os
import pickle
import re
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# -----------------------
# Funções utilitárias
# -----------------------
def extrair_texto_pdf(path):
    texto = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if not txt:
                # PDF como imagem, usa OCR
                imagem = page.to_image(resolution=300).original
                txt = pytesseract.image_to_string(imagem, lang="por")
            texto += txt + " "
    return texto

def extrair_texto_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def limpar_texto(texto):
    texto = texto.lower().strip()
    texto = re.sub(r'\s+', ' ', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    return texto

def extrair_texto_arquivo(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extrair_texto_pdf(path)
    elif ext == ".docx":
        return extrair_texto_docx(path)
    else:
        return None

# -----------------------
# Carregar dados
# -----------------------
# Estrutura: pasta "dados" com subpastas por vaga, contendo currículos
DATA_DIR = "dados"  # Substitua pelo caminho da sua pasta de dados
X, y = [], []

for vaga in os.listdir(DATA_DIR):
    pasta_vaga = os.path.join(DATA_DIR, vaga)
    if not os.path.isdir(pasta_vaga):
        continue
    for arquivo in os.listdir(pasta_vaga):
        path = os.path.join(pasta_vaga, arquivo)
        try:
            texto = extrair_texto_arquivo(path)
            if texto:
                texto = limpar_texto(texto)
                if texto.strip():
                    X.append(texto)
                    y.append(vaga.upper())  # Sempre maiúscula para consistência
        except Exception as e:
            print(f"Erro ao processar {path}: {e}")

# -----------------------
# Treinamento
# -----------------------
vectorizer = TfidfVectorizer()
X_vect = vectorizer.fit_transform(X)

clf = LogisticRegression(max_iter=500)
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# Avaliação
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# -----------------------
# Salvar modelo
# -----------------------
with open("modelo_curriculos.pkl", "wb") as f:
    pickle.dump((clf, vectorizer), f)

print("Treinamento concluído e modelo salvo em modelo_curriculos.pkl")
