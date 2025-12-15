FROM python:3.11-slim

# 1. Configurar diretório de trabalho
WORKDIR /app

# 2. Copiar e instalar requisitos
# (Removemos o apt-get install tesseract pois não usamos mais OCR local)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiar o restante da aplicação
COPY . .

# 4. Comando de Inicialização
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
