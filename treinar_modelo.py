import os
import pickle
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
import nltk
import pdfplumber
import logging
import pytesseract
from PIL import Image
logging.getLogger("pdfminer").setLevel(logging.ERROR)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Baixar stopwords do NLTK (só precisa rodar uma vez)
nltk.download('stopwords')

PASTA_CURRICULOS = r"C:\Users\user\CURRICULOS"
PASTA_IGNORADA = "OUTROS"
ARQUIVO_MODELO = "modelo_curriculos.pkl"

def ler_pdf_arquivo(caminho):
    texto_total = ""
    with pdfplumber.open(caminho) as pdf:
        for pagina in pdf.pages:
            pagina_texto = pagina.extract_text()
            
            if pagina_texto and pagina_texto.strip():  # se tem texto embutido
                texto_total += pagina_texto.strip() + "\n"
            else:  # tenta OCR
                try:
                    imagem = pagina.to_image(resolution=300)
                    texto_ocr = pytesseract.image_to_string(imagem.original, lang="por")
                    if texto_ocr.strip():
                        texto_total += texto_ocr.strip() + "\n"
                except Exception as e:
                    print(f"Erro ao aplicar OCR em {caminho}: {e}")
    return texto_total.strip()

def ler_docx_arquivo(caminho):
    doc = Document(caminho)
    texto = "\n".join([p.text for p in doc.paragraphs])
    return texto

def treinar_modelo():
    textos = []
    labels = []

    for setor in os.listdir(PASTA_CURRICULOS):
        if setor == PASTA_IGNORADA:
            continue
        caminho_setor = os.path.join(PASTA_CURRICULOS, setor)
        if os.path.isdir(caminho_setor):
            for arquivo in os.listdir(caminho_setor):
                caminho_arquivo = os.path.join(caminho_setor, arquivo)
                if arquivo.lower().endswith(".pdf"):
                    texto = ler_pdf_arquivo(caminho_arquivo)
                elif arquivo.lower().endswith(".docx"):
                    texto = ler_docx_arquivo(caminho_arquivo)
                else:
                    continue
                textos.append(texto)
                labels.append(setor)

    if not textos:
        print("Nenhum currículo encontrado para treino.")
        return

    # Lista de stopwords em português
    stop_words_pt = stopwords.words('portuguese')

    # Vetorizador com stopwords personalizadas
    vectorizer = TfidfVectorizer(stop_words=stop_words_pt)
    X = vectorizer.fit_transform(textos)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, labels)

    with open(ARQUIVO_MODELO, "wb") as f:
        pickle.dump((clf, vectorizer), f)

    print(f"✅ Modelo treinado com {len(textos)} currículos e salvo em {ARQUIVO_MODELO}!")

if __name__ == "__main__":
    treinar_modelo()