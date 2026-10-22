"""Registro de eventos + transmissão em tempo real (Server-Sent Events).

Todo evento é gravado no banco (fonte da verdade) e depois enviado aos
navegadores conectados. O front só anima o que chega por aqui.
"""
import asyncio
import json
import logging
import threading

from sqlalchemy import event as sa_event, select, text
from sqlalchemy.orm import Session

from .db import Sessao, engine
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
    pend = db.info.pop("eventos_pendentes", [])
    if not pend:
        return
    if _escuta["ok"]:  # avisa todos os computadores (inclusive este) pelo banco; cada um repassa às suas telas
        try:
            with engine.begin() as con:
                con.execute(text("SELECT pg_notify(:c, :ids)"), {"c": CANAL, "ids": ",".join(str(p["id"]) for p in pend)})
            return
        except Exception as e:  # sem o aviso, ao menos as telas deste computador recebem
            log.warning("Aviso de eventos pelo banco falhou: %s", str(e).splitlines()[0][:160])
    for payload in pend:
        _transmitir(payload)


# --------------------------------------------------------------------------- vários computadores (PostgreSQL)
CANAL = "hub_eventos"
_escuta = {"ok": False}  # sem a escuta no ar, cada computador mostra ao menos os próprios eventos


def iniciar_escuta() -> threading.Event:
    """Liga a escuta numa thread própria; devolve o sinal que a desliga ao encerrar o programa."""
    parar = threading.Event()
    threading.Thread(target=escutar_banco, args=(parar,), name="escuta-eventos", daemon=True).start()
    return parar


def escutar_banco(_parar: threading.Event):
    """Recebe os avisos de eventos de todos os computadores ligados ao mesmo banco e repassa às telas deste
    (com uma conexão só para isso; reconecta sozinha se a rede cair)."""
    while not _parar.is_set():
        bruta = None
        try:
            bruta = engine.raw_connection()
            con = bruta.driver_connection
            bruta.detach()  # conexão exclusiva da escuta, fora do pool
            con.autocommit = True
            con.execute(f"LISTEN {CANAL}")
            _escuta["ok"] = True
            while not _parar.is_set():
                ids = []
                for aviso in con.notifies(timeout=15):
                    ids += [int(i) for i in aviso.payload.split(",") if i.isdigit()]
                    if len(ids) > 200:
                        break
                if ids:
                    _repassar(ids)
                else:
                    con.execute("SELECT 1")  # mantém a conexão viva e percebe se ela caiu
        except Exception as e:
            _escuta["ok"] = False
            if not _parar.is_set():
                log.warning("Escuta de eventos interrompida (%s); tentando de novo", str(e).splitlines()[0][:120])
                _parar.wait(5)
        finally:
            if bruta is not None:
                try:
                    bruta.close()
                except Exception:
                    pass


def _repassar(ids: list[int]):
    if not _ouvintes:
        return
    with Sessao() as s:
        for ev in s.scalars(select(Evento).where(Evento.id.in_(ids)).order_by(Evento.id)):
            _transmitir(serializar(ev))


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
