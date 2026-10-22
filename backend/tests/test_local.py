"""ACESSO=local: sem login, só o próprio computador; e vários computadores no mesmo banco."""
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from app import config, seguranca
from app.db import engine, sessao
from app.lideranca import Lideranca
from app.models import Usuario

PG = engine.dialect.name == "postgresql"


def pedido(cliente="127.0.0.1", host="127.0.0.1:8765", metodo="GET", cabecalhos=()):
    h = [(b"host", host.encode())] + [(k.encode(), v.encode()) for k, v in cabecalhos]
    return Request({"type": "http", "method": metodo, "path": "/auth/eu", "headers": h, "client": (cliente, 50000),
                    "query_string": b""})


def test_so_o_proprio_computador():
    assert seguranca.pedido_local(pedido())
    assert seguranca.pedido_local(pedido(host="localhost:8765"))
    assert seguranca.pedido_local(pedido(cliente="::1", host="[::1]:8765"))
    assert not seguranca.pedido_local(pedido(cliente="192.168.0.20"))  # outro computador da rede
    assert not seguranca.pedido_local(pedido(host="rh.exemplo.com:8765"))  # site apontando para 127.0.0.1


def test_recrutador_e_o_usuario_do_windows(cli, monkeypatch):
    monkeypatch.setattr(config, "LOCAL", True)
    monkeypatch.setattr(config, "RECRUTADOR_NOME", "Maria Souza")
    monkeypatch.setattr(seguranca, "identidade_local", lambda: ("maria.souza", "rh-02"))
    monkeypatch.setattr(seguranca, "_id_local", None)
    with sessao() as db:
        u = seguranca.usuario_atual(pedido(), db)
        assert (u.nome, u.email, u.papel) == ("Maria Souza", "maria.souza@rh-02.local", "admin")
        assert not seguranca.conferir_senha("", u.senha_hash)  # sem senha: não dá para entrar por login
        with pytest.raises(HTTPException) as e:
            seguranca.usuario_atual(pedido(cliente="10.0.0.7"), db)
        assert e.value.status_code == 403
        with pytest.raises(HTTPException):  # escrita sem o cabeçalho do próprio escritório
            seguranca.usuario_atual(pedido(metodo="POST"), db)
        assert seguranca.usuario_atual(pedido(metodo="POST", cabecalhos=[("x-hub", "1")]), db).id == u.id
    with sessao() as db:  # o administrador pode desativar um computador
        db.get(Usuario, u.id).ativo = False
    with sessao() as db, pytest.raises(HTTPException):
        seguranca.usuario_atual(pedido(), db)
    monkeypatch.setattr(seguranca, "_id_local", None)


@pytest.mark.skipif(not PG, reason="vários computadores só com PostgreSQL/Supabase")
def test_um_computador_comanda_por_vez():
    a, b = Lideranca(engine, "teste"), Lideranca(engine, "teste")
    try:
        assert a.tentar() and a.tentar()
        assert not b.tentar()
        a.soltar()  # o computador que comandava fechou o programa
        assert b.tentar()
        assert not a.tentar()
    finally:
        a.soltar()
        b.soltar()
