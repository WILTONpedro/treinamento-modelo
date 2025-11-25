FROM python:3.11-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-por \
    poppler-utils \
    && apt-get clean

# Criar diretório de trabalho
WORKDIR /app

# Copiar requisitos e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar sua aplicação
COPY . .

# Expor porta e rodar Flask
EXPOSE 5000
CMD ["python", "api.py"]
