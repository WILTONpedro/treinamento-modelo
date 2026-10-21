"""Ambiente comum dos testes: modo demonstração (sem IA externa) e banco temporário.

Rode com:  cd backend && python -m pytest -q
"""
import os
import sys
import tempfile
import time
from pathlib import Path

PASTA = tempfile.mkdtemp(prefix="baly-teste-")
os.environ.update({
    "PASTA_DADOS": PASTA, "IA_MODO": "demo", "DEMO_DURACAO_ANALISE": "0", "DEMO_DIGITACAO_MS": "0", "SEED_DEMO": "false",
    "ADMIN_EMAIL": "admin@teste.local", "ADMIN_SENHA": "senha-admin-123", "TRIAGEM_API_KEY": "chave-teste",
    "SEGREDO_JWT": "segredo-de-teste-com-tamanho-suficiente", "SEGUNDOS_CONCLUIDO": "1",
    # cada teste liga a contratação automática quando precisa dela
    "CONTRATACAO_AUTOMATICA": "false", "SUPERVISAO_SEGUNDOS": "0.2", "BOLETIM_MINUTOS": "0",
})
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

DEMO = Path(__file__).resolve().parent.parent / "demo"
H = {"X-Baly": "1"}


def esperar(cond, limite=20, msg="a condição não foi atendida"):
    fim = time.time() + limite
    while time.time() < fim:
        r = cond()
        if r:
            return r
        time.sleep(0.15)
    raise AssertionError(msg)


def esperar_fila(c, limite=30):
    def vazia():
        q = c.get("/api/escritorio").json()["quadro"]
        return q if q["tarefas_pendentes"] == 0 else None
    return esperar(vazia, limite, "a fila não esvaziou")


def enviar(c, nome, chave="chave-teste", conteudo=None):
    dado = conteudo.encode() if conteudo is not None else (DEMO / nome).read_bytes()
    return c.post("/triagem", files={"file": (nome, dado, "text/plain")}, headers={"X-API-Key": chave} if chave else {})


@pytest.fixture(scope="module")
def cli():
    """Banco limpo e sessão de admin: cada módulo de teste começa do zero."""
    from app import seguranca
    from app.db import Base, engine
    from app.main import app
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    seguranca._tentativas.clear()
    for arq in Path(PASTA, "arquivos").glob("*.bin"):
        arq.unlink()
    with TestClient(app) as c:
        assert c.post("/auth/login", json={"email": "admin@teste.local", "senha": "senha-admin-123"}).status_code == 200
        yield c
