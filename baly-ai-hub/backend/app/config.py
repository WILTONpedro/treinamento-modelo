"""Configuração lida de variáveis de ambiente (arquivo .env opcional).

Nada sensível fica no código: chaves, senhas e segredos vêm do ambiente.
"""
import os
import secrets
import logging
from pathlib import Path

log = logging.getLogger("baly.config")

BASE = Path(__file__).resolve().parent.parent


def _carregar_env():
    arq = BASE / ".env"
    if not arq.exists():
        return
    for linha in arq.read_text(encoding="utf-8").splitlines():
        linha = linha.strip()
        if not linha or linha.startswith("#") or "=" not in linha:
            continue
        k, v = linha.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_carregar_env()


def _bool(nome, padrao=False):
    return os.environ.get(nome, str(padrao)).strip().lower() in ("1", "true", "sim", "yes", "on")


def _int(nome, padrao):
    try:
        return int(os.environ.get(nome, padrao))
    except ValueError:
        return padrao


PASTA_DADOS = Path(os.environ.get("PASTA_DADOS", BASE / "data"))
PASTA_DADOS.mkdir(parents=True, exist_ok=True)
PASTA_ARQUIVOS = PASTA_DADOS / "arquivos"
PASTA_ARQUIVOS.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{PASTA_DADOS / 'baly.db'}")

# --- IA -------------------------------------------------------------------
# IA_MODO=gemini usa a API do Google; IA_MODO=demo usa heurísticas locais
# (sem enviar dados para fora) — útil para testar o sistema sem chave.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
IA_MODO = os.environ.get("IA_MODO", "gemini" if GEMINI_API_KEY else "demo").lower()
# A triagem continua no modelo que a API original já usava.
GEMINI_MODELO_TRIAGEM = os.environ.get("GEMINI_MODELO_TRIAGEM", "gemini-2.5-flash")
# Para análises e chat (projeto novo), o Google recomenda os modelos atuais.
GEMINI_MODELO_AGENTES = os.environ.get("GEMINI_MODELO_AGENTES", "gemini-3.8-flash")
IA_CONCORRENCIA = _int("IA_CONCORRENCIA", 3)
DEMO_DURACAO_ANALISE = float(os.environ.get("DEMO_DURACAO_ANALISE", "4"))

# --- Segurança --------------------------------------------------------------
SEGREDO_JWT = os.environ.get("SEGREDO_JWT", "")
if not SEGREDO_JWT:
    SEGREDO_JWT = secrets.token_urlsafe(48)
    log.warning("SEGREDO_JWT não definido: gerado um segredo temporário (sessões caem a cada reinício).")
SESSAO_HORAS = _int("SESSAO_HORAS", 8)
COOKIE_SECURE = _bool("COOKIE_SECURE", False)  # True em produção (HTTPS)
TRIAGEM_API_KEY = os.environ.get("TRIAGEM_API_KEY", "")
ORIGENS_CORS = [o.strip() for o in os.environ.get("ORIGENS_CORS", "").split(",") if o.strip()]

# Chave de criptografia dos arquivos de currículo (Fernet, base64 urlsafe 32 bytes)
CHAVE_ARQUIVOS = os.environ.get("CHAVE_ARQUIVOS", "")

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@baly.local")
ADMIN_SENHA = os.environ.get("ADMIN_SENHA", "")
ADMIN_NOME = os.environ.get("ADMIN_NOME", "Administrador")

# --- LGPD -------------------------------------------------------------------
RETENCAO_DIAS = _int("RETENCAO_DIAS", 180)
TAMANHO_MAX_MB = _int("TAMANHO_MAX_MB", 10)

# --- Comportamento dos agentes ------------------------------------------------
ATIVACAO_AUTOMATICA = _bool("ATIVACAO_AUTOMATICA", False)
SEGUNDOS_PARA_DESCANSO = _int("SEGUNDOS_PARA_DESCANSO", 300)
SEGUNDOS_CONCLUIDO = _int("SEGUNDOS_CONCLUIDO", 8)

# --- Leitura de e-mail (opcional; desligada por padrão) -----------------------
IMAP_ATIVO = _bool("IMAP_ATIVO", False)
IMAP_HOST = os.environ.get("IMAP_HOST", "")
IMAP_USUARIO = os.environ.get("IMAP_USUARIO", "")
IMAP_SENHA = os.environ.get("IMAP_SENHA", "")
IMAP_PASTA = os.environ.get("IMAP_PASTA", "INBOX")
IMAP_INTERVALO = _int("IMAP_INTERVALO", 120)

# --- Demonstração -------------------------------------------------------------
SEED_DEMO = _bool("SEED_DEMO", False)
DEMO_INTERVALO = float(os.environ.get("DEMO_INTERVALO", "5"))
