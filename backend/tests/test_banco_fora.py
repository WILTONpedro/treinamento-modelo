"""Banco fora do ar ou acesso cortado: a tela recebe um aviso claro (503) e o programa tenta de novo sozinho."""
import asyncio

import pytest
from sqlalchemy.exc import OperationalError

from app import db as banco, main
from app.db import obter_db, problema_do_banco
from conftest import H


class ErroDoDriver(Exception):
    def __init__(self, msg, sqlstate=None):
        super().__init__(msg)
        self.sqlstate = sqlstate


def erro(msg, sqlstate=None):
    return OperationalError("SELECT 1", {}, ErroDoDriver(msg, sqlstate))


@pytest.fixture
def postgres(monkeypatch):
    """A classificação vale para o PostgreSQL (no SQLite, "database is locked" é passageiro e não vira aviso)."""
    monkeypatch.setattr(banco, "_sqlite", False)


@pytest.mark.parametrize("msg, sqlstate, esperado", [
    ('FATAL: password authentication failed for user "hub_app"', "28P01", "recusado"),
    ('FATAL: role "hub_clinica" is not permitted to log in', "28000", "recusado"),
    ('server login failed: FATAL: role "hub_clinica" is not permitted to log in', "08P01", "recusado"),
    ("FATAL: Tenant or user not found", None, "recusado"),
    ("(ENOTFOUND) tenant/user hub_app.abc not found", None, "recusado"),
    ('FATAL: role "hub_x" does not exist', None, "recusado"),
    ('connection to server at "127.0.0.2", port 6543 failed: Connection refused', None, "fora"),
    ("server closed the connection unexpectedly", None, "fora"),
    ("FATAL: too many connections for role", "53300", "fora"),
    ("canceling statement due to statement timeout", "57014", None),  # consulta lenta: não é falta de banco
    ("deadlock detected", "40P01", None),
])
def test_classifica_o_problema(postgres, msg, sqlstate, esperado):
    p = problema_do_banco(erro(msg, sqlstate))
    assert (p or {}).get("banco") == esperado
    if esperado:
        assert p["detail"]


def test_outros_erros_e_sqlite_nao_viram_aviso(monkeypatch):
    assert problema_do_banco(ValueError("x")) is None
    monkeypatch.setattr(banco, "_sqlite", True)
    assert problema_do_banco(erro("database is locked")) is None


def test_rota_responde_503_com_o_aviso(cli, postgres):
    def falha():
        raise erro('FATAL: role "hub_clinica" is not permitted to log in', "28000")
        yield  # noqa
    main.app.dependency_overrides[obter_db] = falha
    try:
        r = cli.get("/api/vagas")
    finally:
        main.app.dependency_overrides.pop(obter_db)
    assert r.status_code == 503 and r.json()["banco"] == "recusado" and "configuração nova" in r.json()["detail"]
    assert cli.get("/api/vagas").status_code == 200  # voltou


def test_saude(cli, postgres, monkeypatch):
    assert cli.get("/api/saude").json() == {"ok": True, "banco": "ok", "ia_modo": "demo", "ia_pausada": False}

    def caiu():
        raise erro("connection refused")
    monkeypatch.setattr(main, "_testar_banco", caiu)
    r = cli.get("/api/saude")
    assert r.status_code == 503 and r.json()["banco"] == "fora"


def test_banco_ainda_nao_pronto(cli, monkeypatch):
    """Na abertura sem banco: dados respondem 503 com o aviso; a tela e a marca abrem normalmente."""
    monkeypatch.setitem(main.ESTADO_BANCO, "pronto", False)
    monkeypatch.setitem(main.ESTADO_BANCO, "problema", {"banco": "recusado", "detail": banco.SEM_ACESSO})
    r = cli.get("/api/vagas")
    assert r.status_code == 503 and r.json()["banco"] == "recusado"
    assert cli.get("/auth/eu").status_code == 503  # a tela não pede login: espera o banco
    assert cli.get("/api/marca").status_code == 200 and cli.get("/").status_code == 200
    assert cli.post("/api/vagas", headers=H, json={"titulo": "X", "setores": ["TI"]}).status_code == 503


def test_insiste_ate_o_banco_voltar(postgres, monkeypatch):
    tentativas, comecou = [], []

    def preparar():
        tentativas.append(1)
        if len(tentativas) < 3:
            raise erro("connection refused")
    monkeypatch.setattr(main, "_preparar_banco", preparar)

    async def roda():
        main._acordar = asyncio.Event()
        await asyncio.wait_for(main._insistir_no_banco(lambda: comecou.append(1), espera=0.01), timeout=5)
    monkeypatch.setitem(main.ESTADO_BANCO, "problema", None)
    asyncio.run(roda())
    assert len(tentativas) == 3 and comecou == [1]


def test_pooler_fora_de_sincronia_e_passageiro(postgres):
    """Logo depois de o banco voltar, o pooler pode responder em dobro a um COMMIT: aviso passageiro, não erro 500."""
    from sqlalchemy.exc import InternalError
    p = problema_do_banco(InternalError("COMMIT", {}, ErroDoDriver("received 2 results from command 'COMMIT'")))
    assert p and p["banco"] == "fora"
    assert problema_do_banco(InternalError("x", {}, ErroDoDriver("cache lookup failed for type 1"))) is None


def test_commit_que_falha_nao_responde_ok(cli, postgres, monkeypatch):
    """O commit acontece antes da resposta: se a conexão cair bem nessa hora, a tela recebe o aviso, não um "ok"
    para algo que não foi gravado."""
    from sqlalchemy.orm import Session
    original, cair = Session.commit, {"agora": False}

    def cai_no_commit(self):
        if cair["agora"]:
            raise erro("server closed the connection unexpectedly")
        return original(self)
    monkeypatch.setattr(Session, "commit", cai_no_commit)
    cair["agora"] = True
    try:
        r = cli.post("/api/vagas", headers=H, json={"titulo": "Vaga que não foi gravada", "setores": ["TI"]})
    finally:
        cair["agora"] = False
    assert r.status_code == 503 and r.json()["banco"] == "fora"
    monkeypatch.setattr(Session, "commit", original)
    assert all(v["titulo"] != "Vaga que não foi gravada" for v in cli.get("/api/vagas").json())
