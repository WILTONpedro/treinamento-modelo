# ---- BASE ----
FROM python:3.13-slim

# ---- DEPENDÊNCIAS DO SISTEMA ----
RUN apt-get update && \
    apt-get install -y tesseract-ocr tesseract-ocr-por poppler-utils && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- DIRETÓRIO DE TRABALHO ----
WORKDIR /app

# ---- COPIAR CÓDIGO ----
COPY . .

# ---- INSTALAR DEPENDÊNCIAS PYTHON ----
RUN pip install --no-cache-dir -r requirements.txt

# ---- EXPOR PORTA ----
EXPOSE 5000

# ---- RODAR APP ----
CMD ["python", "api.py"]
