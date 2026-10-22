"""Configuração lida de variáveis de ambiente (arquivo .env opcional).

Nada sensível fica no código: chaves, senhas e segredos vêm do ambiente.
"""
import os
import secrets
import logging
from pathlib import Path

log = logging.getLogger("hub.config")

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


def _float(nome, padrao=None):
    v = os.environ.get(nome, "").strip().replace(",", ".")
    try:
        return float(v) if v else padrao
    except ValueError:
        return padrao


def _int(nome, padrao):
    try:
        return int(os.environ.get(nome, padrao))
    except ValueError:
        return padrao


PASTA_DADOS = Path(os.environ.get("PASTA_DADOS", BASE / "data"))
PASTA_DADOS.mkdir(parents=True, exist_ok=True)
PASTA_ARQUIVOS = PASTA_DADOS / "arquivos"
PASTA_ARQUIVOS.mkdir(parents=True, exist_ok=True)



def _url_banco(url: str) -> str:
    """Aceita a URL como o painel do Supabase/Postgres mostra (postgresql://...): usa o driver psycopg 3 e,
    no Supabase, exige conexão cifrada."""
    if url.startswith(("postgres://", "postgresql://")):
        url = "postgresql+psycopg://" + url.split("://", 1)[1]
    if url.startswith("postgresql+psycopg://") and "supabase" in url and "sslmode=" not in url:
        url += ("&" if "?" in url else "?") + "sslmode=require"
    return url


DATABASE_URL = _url_banco(os.environ.get("DATABASE_URL", f"sqlite:///{PASTA_DADOS / 'hub.db'}"))
# Schema do PostgreSQL usado pelo sistema (no Supabase: "hub", fora da API pública). Vazio = padrão do usuário.
DB_SCHEMA = os.environ.get("DB_SCHEMA", "").strip()
# Onde ficam os arquivos dos currículos (sempre cifrados): disco | banco (recomendado com Supabase)
ARMAZENAMENTO = os.environ.get("ARMAZENAMENTO", "disco").strip().lower()

SEED_DEMO = _bool("SEED_DEMO", False)  # demonstração: nunca em produção
# producao = o sistema se recusa a subir com configuração insegura (veja problemas_producao)
AMBIENTE = os.environ.get("AMBIENTE", "desenvolvimento").strip().lower()

# --- Identidade (sem marca fixa: ajuste para a sua empresa) -------------------
APP_NOME = os.environ.get("APP_NOME", "").strip()  # vazio: nenhum nome na tela
APP_SUBTITULO = os.environ.get("APP_SUBTITULO", "Escritório de Recrutamento")
# Nome da empresa contratante, usado nos prompts ("Recrutador Sênior da empresa X") e na LGPD
EMPRESA_NOME = os.environ.get("EMPRESA_NOME", "").strip()
DA_EMPRESA = "da empresa" + (f" {EMPRESA_NOME}" if EMPRESA_NOME else "")

# --- IA -------------------------------------------------------------------
# IA_MODO=gemini usa a API do Google; IA_MODO=demo usa heurísticas locais
# (sem enviar dados para fora) — útil para testar o sistema sem chave.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
IA_MODO = os.environ.get("IA_MODO", "gemini" if GEMINI_API_KEY else "demo").lower()
# Modelo econômico e atual para tudo (triagem, análises, consultas e chat). Os 2.5 estão sendo desligados
# pelo Google (out/2026). Alternativa mais barata: gemini-3.1-flash-lite; mais forte: gemini-3.8-flash.
GEMINI_MODELO = os.environ.get("GEMINI_MODELO", "gemini-3.5-flash-lite")
GEMINI_MODELO_TRIAGEM = os.environ.get("GEMINI_MODELO_TRIAGEM", GEMINI_MODELO)
GEMINI_MODELO_AGENTES = os.environ.get("GEMINI_MODELO_AGENTES", GEMINI_MODELO)
# "Pensamento" do modelo é cobrado como saída: minimal (padrão) | low | medium | high | padrao (deixa o do modelo)
GEMINI_PENSAMENTO = os.environ.get("GEMINI_PENSAMENTO", "minimal").strip().lower()
# Preço em US$ por 1M tokens, se o do modelo mudar ou não estiver na tabela de uso_ia.py
GEMINI_PRECO_ENTRADA = _float("GEMINI_PRECO_ENTRADA")
GEMINI_PRECO_SAIDA = _float("GEMINI_PRECO_SAIDA")
# Teto de gasto estimado por dia (US$). Ao atingir, o trabalho de IA pausa até o dia seguinte. 0 = sem teto.
IA_LIMITE_DIARIO_USD = _float("IA_LIMITE_DIARIO_USD", 3.0)
IA_CONCORRENCIA = _int("IA_CONCORRENCIA", 3)
FUSO_HORARIO = os.environ.get("FUSO_HORARIO", "America/Sao_Paulo")
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

ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@hub.local")
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


def problemas_producao() -> list[str]:
    """O que impede de subir com AMBIENTE=producao (segurança e LGPD)."""
    p = []
    if len(os.environ.get("SEGREDO_JWT", "")) < 32:
        p.append("SEGREDO_JWT ausente ou curto (gere com python -m app.preparar)")
    if not CHAVE_ARQUIVOS:
        p.append("CHAVE_ARQUIVOS ausente (sem ela os currículos cifrados ficam presos a este servidor)")
    if not TRIAGEM_API_KEY:
        p.append("TRIAGEM_API_KEY ausente (POST /triagem ficaria aberto)")
    if not COOKIE_SECURE:
        p.append("COOKIE_SECURE precisa ser true (acesso só por HTTPS)")
    if SEED_DEMO:
        p.append("SEED_DEMO precisa ser false (cria usuário e dados de demonstração)")
    if ARMAZENAMENTO not in ("disco", "banco"):
        p.append("ARMAZENAMENTO precisa ser disco ou banco")
    if IA_MODO != "gemini":
        p.append("IA_MODO precisa ser gemini (o modo demonstração é heurístico)")
    elif not GEMINI_API_KEY:
        p.append("GEMINI_API_KEY ausente")
    return p
