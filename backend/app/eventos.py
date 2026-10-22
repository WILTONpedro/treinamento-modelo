"""Registro de eventos + transmissão em tempo real (Server-Sent Events).

Todo evento é gravado no banco (fonte da verdade) e depois enviado aos
navegadores conectados. O front só anima o que chega por aqui.
"""
import asyncio
import json
import logging
import threading
from collections import OrderedDict

from sqlalchemy import event as sa_event, select
from sqlalchemy.orm import Session

from .db import Sessao
from .models import Evento, iso

log = logging.getLogger("hub.eventos")

_loop: asyncio.AbstractEventLoop | None = None
_ouvintes: set[asyncio.Queue] = set()


def configurar_loop(loop):
    global _loop
    _loop = loop


def serializar(e: Evento) -> dict:
    return {"id": e.id, "ts": iso(e.ts), "tipo": e.tipo, "agente_id": e.agente_id,
            "mensagem": e.mensagem, "dados": e.dados or {}}


def registrar(db: Session, tipo: str, mensagem: str, agente_id: int | None = None, **dados) -> Evento:
    ev = Evento(tipo=tipo, mensagem=mensagem, agente_id=agente_id, candidato_id=dados.get("candidato_id"), dados=dados)
    db.add(ev)
    db.flush()
    payload = serializar(ev)
    # só transmite depois do commit, para o front nunca ver algo que foi desfeito
    pend = db.info.setdefault("eventos_pendentes", [])
    pend.append(payload)
    return ev


def transmitir_pendentes(db: Session):
    for payload in db.info.pop("eventos_pendentes", []):
        if _primeira_vez(payload["id"]):
            _transmitir(payload)


# --------------------------------------------------------------------------- vários computadores (PostgreSQL)
# Cada computador acompanha a tabela de eventos e leva às suas telas o que os outros gravaram. Por consulta
# (e não LISTEN/NOTIFY): funciona no pooler do Supabase em modo transação, que atende muitos computadores.
# Transações paralelas podem gravar eventos fora da ordem dos números; a janela revê os últimos números, então
# nada se perde nem aparece duas vezes.
JANELA = 200
_vistos: OrderedDict[int, None] = OrderedDict()
_trava_vistos = threading.Lock()


def _primeira_vez(evento_id: int) -> bool:
    with _trava_vistos:
        if evento_id in _vistos:
            return False
        _vistos[evento_id] = None
        if len(_vistos) > 20 * JANELA:
            _vistos.popitem(last=False)
        return True


def acompanhar_banco(intervalo: float = 1.0) -> threading.Event:
    """Liga o acompanhamento numa thread própria; devolve o sinal que o desliga ao encerrar o programa."""
    parar = threading.Event()
    threading.Thread(target=_acompanhar, args=(parar, intervalo), name="eventos-do-banco", daemon=True).start()
    return parar


def _acompanhar(parar: threading.Event, intervalo: float):
    ultimo = None  # maior número já visto; começa no presente (o histórico não é reenviado)
    while not parar.wait(intervalo):
        try:
            with Sessao() as s:
                if ultimo is None:
                    recentes = s.scalars(select(Evento.id).order_by(Evento.id.desc()).limit(JANELA)).all()
                    for i in recentes:
                        _primeira_vez(i)
                    ultimo = max(recentes, default=0)
                    continue
                ids = s.scalars(select(Evento.id).where(Evento.id > ultimo - JANELA).order_by(Evento.id)).all()
                novos = [i for i in ids if _primeira_vez(i)]
                ultimo = max([ultimo, *ids])
                if novos and _ouvintes:
                    for ev in s.scalars(select(Evento).where(Evento.id.in_(novos)).order_by(Evento.id)):
                        _transmitir(serializar(ev))
        except Exception as e:
            log.warning("Não consegui buscar os eventos dos outros computadores: %s", str(e).splitlines()[0][:160])
            parar.wait(5)


def descartar_pendentes(db: Session):
    db.info.pop("eventos_pendentes", None)


def _transmitir(payload: dict):
    if not _loop:
        return
    dado = json.dumps(payload, ensure_ascii=False)
    for q in list(_ouvintes):
        try:
            _loop.call_soon_threadsafe(q.put_nowait, dado)
        except Exception:  # loop encerrado
            pass


@sa_event.listens_for(Sessao, "after_commit")
def _apos_commit(s):
    transmitir_pendentes(s)


@sa_event.listens_for(Sessao, "after_rollback")
def _apos_rollback(s):
    descartar_pendentes(s)


def novo_ouvinte() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _ouvintes.add(q)
    return q


def remover_ouvinte(q):
    _ouvintes.discard(q)
