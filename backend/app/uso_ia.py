"""Uso da IA: tokens e custo estimado por chamada, limite diário de gasto e pausa automática
quando o provedor recusa (sem créditos, chave inválida, modelo indisponível).

Os preços são estimativas em US$ por 1 milhão de tokens. Confira a tabela do Google e, se mudarem,
ajuste GEMINI_PRECO_ENTRADA / GEMINI_PRECO_SAIDA.
"""
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from sqlalchemy import func, select

from . import config
from .db import sessao
from .models import UsoIA, agora

log = logging.getLogger("hub.uso_ia")

# US$ por 1M tokens: (entrada, saída — o "pensamento" é cobrado como saída). Referência: out/2026.
PRECOS = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}
DESCONTO_CACHE = 0.1  # tokens reaproveitados do cache implícito custam ~10% da entrada


class IAIndisponivel(Exception):
    """A IA está pausada (limite diário, sem créditos, chave ou modelo recusados). O trabalho espera."""


def preco(modelo: str) -> tuple[float, float] | None:
    if config.GEMINI_PRECO_ENTRADA is not None and config.GEMINI_PRECO_SAIDA is not None:
        return config.GEMINI_PRECO_ENTRADA, config.GEMINI_PRECO_SAIDA
    return PRECOS.get(modelo)


def custo(modelo: str, entrada: int, saida: int, cache: int = 0) -> float:
    p = preco(modelo)
    if not p:
        return 0.0
    cobrada = max(0, entrada - cache) + cache * DESCONTO_CACHE
    return (cobrada * p[0] + saida * p[1]) / 1_000_000


def _tokens(uso) -> tuple[int, int, int, int]:
    """(entrada, saída visível, pensamento, cache) a partir do usage_metadata do Gemini."""
    if uso is None:
        return 0, 0, 0, 0
    g = lambda nome: int(getattr(uso, nome, 0) or 0)  # noqa: E731
    return g("prompt_token_count"), g("candidates_token_count"), g("thoughts_token_count"), g("cached_content_token_count")


def registrar(finalidade: str, modelo: str, uso) -> dict:
    entrada, saida, pensamento, cache = _tokens(uso)
    valor = custo(modelo, entrada, saida + pensamento, cache)
    try:
        with sessao() as db:
            db.add(UsoIA(finalidade=finalidade, modelo=modelo, entrada=entrada, saida=saida, pensamento=pensamento,
                         cache=cache, custo_usd=valor))
    except Exception:  # contabilidade nunca derruba o trabalho
        log.exception("Falha ao registrar uso de IA")
    _estado["gasto_cache"] = None
    sucesso()
    return {"entrada": entrada, "saida": saida, "pensamento": pensamento, "cache": cache, "custo_usd": valor}


# --------------------------------------------------------------------------- dia (no fuso da operação)
def inicio_do_dia() -> datetime:
    """Meia-noite de hoje no fuso configurado, em UTC sem fuso (como o banco guarda)."""
    tz = ZoneInfo(config.FUSO_HORARIO)
    local = datetime.now(tz).replace(hour=0, minute=0, second=0, microsecond=0)
    return local.astimezone(timezone.utc).replace(tzinfo=None)


def resumo(db, desde: datetime) -> dict:
    linha = db.execute(select(func.count(UsoIA.id), func.coalesce(func.sum(UsoIA.entrada), 0),
                              func.coalesce(func.sum(UsoIA.saida), 0), func.coalesce(func.sum(UsoIA.pensamento), 0),
                              func.coalesce(func.sum(UsoIA.cache), 0), func.coalesce(func.sum(UsoIA.custo_usd), 0.0))
                       .where(UsoIA.ts >= desde)).one()
    return {"chamadas": linha[0], "entrada": int(linha[1]), "saida": int(linha[2]), "pensamento": int(linha[3]),
            "cache": int(linha[4]), "custo_usd": round(float(linha[5]), 4)}


def por_finalidade(db, desde: datetime) -> list[dict]:
    q = (select(UsoIA.finalidade, func.count(UsoIA.id), func.sum(UsoIA.entrada + UsoIA.saida + UsoIA.pensamento),
                func.sum(UsoIA.custo_usd)).where(UsoIA.ts >= desde).group_by(UsoIA.finalidade))
    return sorted(({"finalidade": f, "chamadas": n, "tokens": int(t or 0), "custo_usd": round(float(c or 0), 4)}
                   for f, n, t, c in db.execute(q).all()), key=lambda x: -x["tokens"])


# --------------------------------------------------------------------------- pausa automática
_trava = threading.Lock()
_estado = {"ate": 0.0, "motivo": "", "falhas": 0, "gasto_cache": None, "gasto_em": 0.0, "teve_sucesso": False}


def gasto_hoje() -> float:
    """Gasto estimado de hoje (com cache de 10 s: é consultado a cada ciclo da fila)."""
    agora_s = time.monotonic()
    if _estado["gasto_cache"] is not None and agora_s - _estado["gasto_em"] < 10:
        return _estado["gasto_cache"]
    with sessao() as db:
        valor = resumo(db, inicio_do_dia())["custo_usd"]
    _estado.update(gasto_cache=valor, gasto_em=agora_s)
    return valor


def bloqueio() -> str | None:
    """Motivo pelo qual a IA não deve ser chamada agora, ou None."""
    if config.IA_MODO != "gemini":
        return None
    with _trava:
        if time.monotonic() < _estado["ate"]:
            return _estado["motivo"]
    limite = config.IA_LIMITE_DIARIO_USD
    if limite and limite > 0:
        gasto = gasto_hoje()
        if gasto >= limite:
            return (f"limite diário de IA atingido (US$ {gasto:.2f} de US$ {limite:.2f}); o trabalho continua amanhã "
                    "ou aumente IA_LIMITE_DIARIO_USD")
    return None


def classificar(erro) -> str | None:
    """Mensagem para o recrutador quando o erro do provedor exige ação humana (e não só uma nova tentativa)."""
    codigo = getattr(erro, "code", None)
    texto = str(erro).lower()
    if codigo == 402 or "prepayment" in texto or "credits are depleted" in texto or "billing" in texto:
        return ("a conta do Gemini está sem créditos: recarregue em https://ai.studio/projects "
                "(o trabalho recomeça sozinho depois)")
    if codigo in (401, 403) or "api key not valid" in texto or "api_key_invalid" in texto or "permission" in texto:
        return "a chave do Gemini foi recusada (inválida ou sem permissão): confira GEMINI_API_KEY"
    if codigo == 404 or "no longer available" in texto or "is not found" in texto:
        return (f"o modelo de IA configurado não está disponível para esta conta: ajuste GEMINI_MODELO "
                f"(hoje: {config.GEMINI_MODELO_AGENTES})")
    if codigo == 429 and ("per day" in texto or "perday" in texto or "daily" in texto):
        return "a cota diária do Gemini acabou: o trabalho recomeça quando a cota renovar"
    return None


def falha_provedor(erro) -> bool:
    """Registra uma recusa do provedor. True = a IA foi pausada (não adianta tentar de novo agora)."""
    motivo = classificar(erro)
    if not motivo:
        return False
    with _trava:
        _estado["falhas"] += 1
        minutos = min(60, 5 * 2 ** (_estado["falhas"] - 1))
        _estado.update(ate=time.monotonic() + minutos * 60, motivo=motivo, teve_sucesso=False)
    log.warning("IA pausada por %d min: %s", minutos, motivo)
    return True


def sucesso():
    with _trava:
        if _estado["falhas"] or _estado["ate"]:
            log.info("IA respondeu de novo: pausa encerrada")
        _estado.update(ate=0.0, motivo="", falhas=0, teve_sucesso=True)


def pausa_do_provedor() -> str | None:
    """Só a pausa por recusa do provedor (sem o limite diário)."""
    with _trava:
        return _estado["motivo"] if time.monotonic() < _estado["ate"] or (_estado["falhas"] and not _estado["teve_sucesso"]) else None


def zerar():
    """Para testes."""
    with _trava:
        _estado.update(ate=0.0, motivo="", falhas=0, gasto_cache=None, gasto_em=0.0, teve_sucesso=False)
