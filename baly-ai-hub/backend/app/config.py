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

SEED_DEMO = _bool("SEED_DEMO", False)  # demonstração: nunca em produção

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
# Pensamentos e conversas entre agentes (podem citar o primeiro nome do candidato) são apagados depois disto
RETENCAO_CONVERSAS_HORAS = _int("RETENCAO_CONVERSAS_HORAS", 72)
TAMANHO_MAX_MB = _int("TAMANHO_MAX_MB", 10)

# --- Comportamento dos agentes ------------------------------------------------
# O Atlas contrata (ativa) especialistas sozinho quando falta alguém, quando a fila cresce
# ou quando a equipe pede ajuda de uma área que ainda não existe. false = ele só propõe,
# e o supervisor aprova no quadro geral. (ATIVACAO_AUTOMATICA é o nome antigo.)
CONTRATACAO_AUTOMATICA = _bool("CONTRATACAO_AUTOMATICA", _bool("ATIVACAO_AUTOMATICA", True))
# Média de currículos na fila por agente da mesma especialidade que faz o Atlas reforçar o time
LIMITE_FILA_POR_AGENTE = _int("LIMITE_FILA_POR_AGENTE", 2 if SEED_DEMO else 5)
MAX_AGENTES_POR_ESPECIALIDADE = _int("MAX_AGENTES_POR_ESPECIALIDADE", 3)
# Espera mínima entre duas contratações para a mesma especialidade
INTERVALO_CONTRATACAO = _int("INTERVALO_CONTRATACAO", 30 if SEED_DEMO else 600)
# Quantos pedidos de ajuda sem especialista (em 7 dias) fazem o Atlas contratar alguém da área
DEMANDA_PARA_CONTRATAR = _int("DEMANDA_PARA_CONTRATAR", 2)
# Analistas consultam colegas de outra especialidade quando têm dúvida sobre o currículo
COLABORACAO = _bool("COLABORACAO", True)
# Currículos sem pasta (OUTROS): o Atlas lê e encaminha sozinho quando combinam com uma vaga aberta
ENCAMINHAMENTO_AUTOMATICO = _bool("ENCAMINHAMENTO_AUTOMATICO", True)
# Falhas de IA: o Atlas tenta de novo sozinho depois deste tempo (até 2 vezes) antes de depender do recrutador
REPROCESSAR_APOS_MIN = _int("REPROCESSAR_APOS_MIN", 10)
# Boletim do Atlas para o time (0 desliga)
BOLETIM_MINUTOS = _int("BOLETIM_MINUTOS", 60)
SUPERVISAO_SEGUNDOS = float(os.environ.get("SUPERVISAO_SEGUNDOS", "2" if SEED_DEMO else "5"))
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
DEMO_INTERVALO = float(os.environ.get("DEMO_INTERVALO", "5"))
# Velocidade da "digitação" do chat em modo demonstração (ms por palavra)
DEMO_DIGITACAO_MS = _int("DEMO_DIGITACAO_MS", 18)
