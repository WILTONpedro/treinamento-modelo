"""A tela nunca fica velha depois de uma atualização: versão nos endereços e nada guardado sem conferir."""
from app.main import VERSAO


def test_pagina_com_versao_e_sem_cache_velho(cli):
    r = cli.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    for arq in ("app.js", "cena.js", "estilo.css"):
        assert f'"{arq}?v={VERSAO}"' in r.text
        a = cli.get(f"/{arq}?v={VERSAO}")
        assert a.status_code == 200 and a.headers["cache-control"] == "no-cache"
    assert cli.get("/index.html").text == r.text
    assert cli.get("/api/marca").json()["versao"] == VERSAO == cli.get("/api/escritorio").json()["versao"]
    assert len(VERSAO) == 10
