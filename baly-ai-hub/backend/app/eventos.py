"""Registro de eventos + transmissão em tempo real (Server-Sent Events).

Todo evento é gravado no banco (fonte da verdade) e depois enviado aos
navegadores conectados. O front só anima o que chega por aqui.
"""
import asyncio
import json
import logging

from sqlalchemy import event as sa_event
from sqlalchemy.orm import Session

from .db import Sessao
from .models import Evento, iso

log = logging.getLogger("baly.eventos")

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
        _transmitir(payload)


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
