FROM python:3.11-slim

# Instalar dependências básicas do sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Variáveis para não gerar cache
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copiar dependências e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar a aplicação
COPY . .

EXPOSE 8080

# Iniciar com gunicorn (usado pelo Koyeb)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api:app"]
