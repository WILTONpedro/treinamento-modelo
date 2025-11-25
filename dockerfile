FROM python:3.11-slim

# 1. Instalar Tesseract (OCR) e dependências do sistema
# O 'build-essential' ajuda a compilar bibliotecas Python se necessário
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Configurar diretório
WORKDIR /app

# 3. Copiar e instalar requisitos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar o restante da aplicação
# ISSO É IMPORTANTE: O arquivo 'cerebro_final.pkl' DEVE estar na pasta antes de você fazer o deploy
COPY . .

# 5. Comando de Inicialização (Ajustado para Render)
# O Render injeta a variável $PORT. O sh -c permite ler essa variável.
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
