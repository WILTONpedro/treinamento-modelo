# Imagem de produção: API + escritório (front) num só processo.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PASTA_DADOS=/dados PORT=8000

WORKDIR /srv
COPY backend/requirements.txt backend/requirements.txt
# Rede corporativa com inspeção TLS? passe o certificado sem gravá-lo na imagem:
#   docker build --secret id=ca,src=/caminho/ca-da-empresa.crt .
RUN --mount=type=secret,id=ca,required=false \
    if [ -s /run/secrets/ca ]; then export PIP_CERT=/run/secrets/ca; fi; \
    pip install -r backend/requirements.txt

COPY backend/app backend/app
COPY backend/demo backend/demo
COPY frontend frontend

# usuário sem privilégios; /dados guarda os currículos cifrados (monte um volume aqui)
RUN useradd --system --create-home --uid 10001 hub && mkdir -p /dados && chown hub:hub /dados
USER hub
WORKDIR /srv/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/api/saude' % os.environ.get('PORT','8000'), timeout=4)"

# um único processo: a fila de trabalho e a supervisão do Atlas rodam dentro da API.
# --forwarded-allow-ips: de quais endereços o X-Forwarded-For é aceito. NUNCA '*' (confiaria no IP que o
# cliente mandar, burlando o limite de login e a auditoria). No docker-compose, FORWARDED_ALLOW_IPS aponta
# para o Caddy; sem compose, o padrão 127.0.0.1 não confia em ninguém de fora.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips=\"${FORWARDED_ALLOW_IPS:-127.0.0.1}\""]
