"""Banco de currículos: pastas agrupadas, o que a IA disse de cada um, vaga sugerida, encaminhar e pedir sugestão."""
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import sessao
from app.main import app
from app.models import Auditoria
from conftest import DEMO, H, enviar, esperar_fila


def _preparar(cli):
    r = cli.post("/api/vagas", headers=H, json={"titulo": "Assistente administrativo", "setores": ["ADMINISTRATIVO"],
                                                "requisitos": ["Excel", "rotinas administrativas"]})
    assert r.status_code == 200, r.text
    for arq in sorted(DEMO.glob("*.txt")):
        enviar(cli, arq.name)
    esperar_fila(cli, 60)
    return r.json()["id"]


def test_banco_mostra_pastas_parecer_e_vaga_sugerida(cli):
    vaga_id = _preparar(cli)
    r = cli.get("/api/banco")
    assert r.status_code == 200
    d = r.json()
    grupos = {g["codigo"]: g for g in d["resumo"]["grupos"]}
    assert "administrativo" in grupos and "outros" in grupos
    assert d["resumo"]["total"] == sum(g["total"] for g in grupos.values()) > 0
    assert all(p["setor"] for g in grupos.values() for p in g["pastas"])
    itens = d["itens"]
    assert itens and all(i["situacao"] for i in itens)
    na_vaga = [i for i in itens if i["vaga_sugerida"] and i["vaga_sugerida"]["vaga_id"] == vaga_id]
    assert na_vaga, "algum currículo administrativo devia estar na vaga"
    melhor = na_vaga[0]["vaga_sugerida"]
    assert melhor["analista"] and melhor["compatibilidade"] in ("ALTA", "MEDIA", "BAIXA") and melhor["parecer"]

    # filtros: banco de talentos (sem vaga), por pasta e por grupo, e busca no texto
    talentos = cli.get("/api/banco?situacao=talentos").json()["itens"]
    assert talentos and all(i["vagas"] == 0 and i["situacao"] == "banco de talentos" for i in talentos)
    adm = cli.get("/api/banco?grupo=administrativo").json()["itens"]
    assert adm and all(i["grupo"] == "administrativo" for i in adm)
    pasta = adm[0]["setor"]
    assert all(i["setor"] == pasta for i in cli.get("/api/banco", params={"pasta": pasta}).json()["itens"])
    achou = cli.get("/api/banco", params={"busca": "inglês"}).json()["itens"]
    assert any("Tiago" in i["nome"] for i in achou)
    assert cli.get("/api/banco?situacao=qualquer").status_code == 400

    # o escritório recebe o resumo para desenhar as pastas do balcão
    assert cli.get("/api/escritorio").json()["banco"]["total"] == d["resumo"]["total"]


def test_detalhe_sugestao_do_atlas_e_encaminhamento(cli):
    vaga_id = next(v["id"] for v in cli.get("/api/vagas").json() if v["titulo"] == "Assistente administrativo")
    tiago = next(i for i in cli.get("/api/banco", params={"busca": "Tiago"}).json()["itens"] if "Tiago" in i["nome"])
    d = cli.get(f"/api/banco/{tiago['curriculo_id']}").json()
    assert d["triagem"]["por"] == "Bia" and d["triagem"]["resumo"] and d["texto"]
    assert any(v["id"] == vaga_id for v in d["vagas_abertas"])
    # caiu em OUTROS: o Atlas já leu na distribuição e a sugestão ficou guardada
    if d["triagem"]["setor"] == "OUTROS" and not d["candidaturas"]:
        assert d["sugestao_atlas"] and d["sugestao_atlas"]["motivo"]

    # pedir de novo ao Atlas grava a sugestão (com a vaga ou dizendo que nenhuma combina)
    r = cli.post(f"/api/banco/{tiago['curriculo_id']}/sugerir", headers=H)
    assert r.status_code == 200, r.text
    assert r.json()["sugestao_atlas"]["motivo"]

    # encaminhar à vaga: vira candidatura e o analista da vaga analisa
    r = cli.post(f"/api/banco/{tiago['curriculo_id']}/encaminhar", headers=H, json={"vaga_id": vaga_id})
    assert r.status_code == 200, r.text
    assert any(c["vaga_id"] == vaga_id and c["origem"] == "banco" for c in r.json()["candidaturas"])
    assert cli.post(f"/api/banco/{tiago['curriculo_id']}/encaminhar", headers=H, json={"vaga_id": vaga_id}).status_code == 409
    esperar_fila(cli, 60)
    d = cli.get(f"/api/banco/{tiago['curriculo_id']}").json()
    c = next(c for c in d["candidaturas"] if c["vaga_id"] == vaga_id)
    assert c["analises"] and c["analises"][0]["justificativa"] and d["vaga_sugerida"]["vaga_id"] == vaga_id
    assert cli.get("/api/banco/999999").status_code == 404

    # cada abertura fica na auditoria (LGPD)
    with sessao() as db:
        acoes = set(db.scalars(select(Auditoria.acao)).all())
    assert {"listar_banco", "ver_curriculo", "encaminhar_do_banco", "sugerir_vaga_banco"} <= acoes


def test_recrutador_ve_mas_nao_encaminha(cli):
    r = cli.post("/api/usuarios", headers=H, json={"nome": "Rui RH", "email": "rui@teste.local", "papel": "recrutador"})
    temp = r.json()["senha_temporaria"]
    cur_id = cli.get("/api/banco").json()["itens"][0]["curriculo_id"]
    vaga_id = cli.get("/api/vagas").json()[0]["id"]
    with TestClient(app) as rui:
        rui.post("/auth/login", json={"email": "rui@teste.local", "senha": temp})
        rui.post("/auth/senha", headers=H, json={"senha_atual": temp, "nova_senha": "rui-senha-forte-1"})
        assert rui.get("/api/banco").status_code == 200 and rui.get(f"/api/banco/{cur_id}").status_code == 200
        assert rui.post(f"/api/banco/{cur_id}/encaminhar", headers=H, json={"vaga_id": vaga_id}).status_code == 403
        assert rui.post(f"/api/banco/{cur_id}/sugerir", headers=H).status_code == 403
