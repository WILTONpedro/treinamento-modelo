"""ACESSO=local: sem login, só o próprio computador; e vários computadores no mesmo banco."""
import pytest
from fastapi import HTTPException
from sqlalchemy import text
from starlette.requests import Request

from app import config, seguranca
from app import db as banco
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
def test_um_computador_comanda_por_vez(cli):
    a, b = Lideranca(engine, "teste"), Lideranca(engine, "teste")
    try:
        assert a.tentar() and a.tentar()  # assume e renova
        assert not b.tentar()
        a.soltar()  # o computador que comandava fechou o programa: a vez fica livre na hora
        assert b.tentar() and not a.tentar()
        with engine.begin() as con:  # o computador de b desligou sem avisar: o prazo vence
            con.execute(text("UPDATE comando SET ate = now() - interval '1 second' WHERE nome = 'teste'"))
        assert a.tentar() and not b.tentar()
    finally:
        a.soltar()
        b.soltar()


def test_acha_o_pooler_do_projeto():
    """O pooler do Supabase fica em aws-0 ou aws-1 conforme o projeto: o programa tenta os dois."""
    assert banco._outro_pooler("aws-0-sa-east-1.pooler.supabase.com") == "aws-1-sa-east-1.pooler.supabase.com"
    assert banco._outro_pooler("aws-1-sa-east-1.pooler.supabase.com") == "aws-0-sa-east-1.pooler.supabase.com"
    assert banco._outro_pooler("db.ref.supabase.co") is None and banco._outro_pooler("127.0.0.1") is None
    tentados = []

    class Driver:
        @staticmethod
        def connect(**kw):
            tentados.append(kw["host"][:5])
            if kw["host"].startswith("aws-0") or kw["password"] == "errada":
                raise Exception("connection failed: FATAL:  Tenant or user not found" if kw["password"] != "errada"
                                else "connection failed: FATAL:  password authentication failed")
            return "conectado"

    class Dialeto:
        loaded_dbapi = Driver

    banco._pooler_certo.clear()
    params = {"host": "aws-0-sa-east-1.pooler.supabase.com", "user": "hub_app.ref", "password": "certa"}
    assert banco._conectar_no_pooler_certo(Dialeto, None, [], params) == "conectado"
    assert tentados == ["aws-0", "aws-1"]
    tentados.clear()
    assert banco._conectar_no_pooler_certo(Dialeto, None, [], params) == "conectado"
    assert tentados == ["aws-1"]  # lembra o endereço certo
    with pytest.raises(Exception, match="password authentication failed"):  # senha errada não vira troca de pooler
        banco._conectar_no_pooler_certo(Dialeto, None, [], {**params, "password": "errada"})
    assert banco._conectar_no_pooler_certo(Dialeto, None, [], {"host": "127.0.0.1"}) is None  # outros bancos: normal
    banco._pooler_certo.clear()
    # a mensagem das versões novas do pooler também vale
    assert banco._projeto_desconhecido(Exception("FATAL: (ENOTFOUND) tenant/user hub_app.ref not found"))
    assert not banco._projeto_desconhecido(Exception("FATAL: password authentication failed for user hub_app"))
    # vários computadores: o pooler em modo transação (6543) no lugar do modo sessão (5432)
    url = "postgresql+psycopg://hub_app.ref:x@aws-0-sa-east-1.pooler.supabase.com:5432/postgres?sslmode=require"
    assert ":6543/postgres" in config._modo_transacao(url)
