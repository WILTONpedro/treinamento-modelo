# Imagem base com Python
FROM python:3.13-slim

# Instalar dependências do sistema (Tesseract + poppler para pdfplumber)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-por \
        poppler-utils \
        libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Definir diretório de trabalho
WORKDIR /app

# Copiar arquivos do projeto
COPY requirements.txt .
COPY api.py .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Variável para pytesseract encontrar o executável
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

# Expor porta
EXPOSE 5000

# Comando para rodar a API
CMD ["python", "api.py"]
