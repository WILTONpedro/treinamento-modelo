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


# --------------------------------------------------------------------------- critério do Atlas ao sugerir vaga
SEGURANCA_QUE_ESTUDA_TI = """Carlos Teste
Campinas - SP
EXPERIÊNCIA
Segurança — Supermercado Bom Preço (2025–atual): controle de acesso e rondas.
Operador de caixa — Mercado Central (2024–2025).
FORMAÇÃO
Curso técnico em informática (em andamento). Conhecimentos básicos em programação."""
VAGAS = [{"id": 2, "titulo": "ANALISTA DE TI", "area": "TI", "descricao": "SER DO TI", "requisitos": ["SER BOM EM TI"],
          "setores": ["TI"]},
         {"id": 5, "titulo": "Auxiliar de suporte técnico", "area": "TI", "descricao": "", "requisitos": [], "setores": ["TI"]}]


def test_nivel_pelo_titulo():
    from app import ia
    assert ia.nivel_da_vaga("ANALISTA DE TI") == "experiente" and ia.nivel_da_vaga("Gerente de Produção") == "experiente"
    assert ia.nivel_da_vaga("Auxiliar de suporte técnico") == "entrada" and ia.nivel_da_vaga("Estágio em TI") == "entrada"
    assert ia.nivel_da_vaga("Vendedor") is None


def test_atlas_sem_evidencia_de_experiencia_nao_sugere(monkeypatch):
    """A IA disse MÉDIA para Analista de TI sem citar experiência profissional: a trava rebaixa para BAIXA."""
    from app import config, ia
    pedido = {}

    def falso(finalidade, sistema, conteudos, schema, temperatura=0.2, max_saida=1500):
        pedido.update(sistema=sistema, conteudos="\n".join(conteudos))
        return {"vaga_id": 2, "confianca": "MEDIA", "motivo": "Tem curso técnico na área.", "evidencias": [],
                "lacunas": [], "perfil_indicado": "segurança patrimonial, atendimento de caixa"}
    monkeypatch.setattr(config, "IA_MODO", "gemini")
    monkeypatch.setattr(ia, "_gemini_json", falso)
    r = ia.sugerir_vaga(SEGURANCA_QUE_ESTUDA_TI, VAGAS, "VIGIA", 1)
    assert r["confianca"] == "BAIXA" and "sem experiência profissional" in r["lacunas"][0]
    assert r["perfil_indicado"].startswith("segurança")
    # o pedido leva os critérios de recrutador, a pasta da triagem e o nível e as pastas de cada vaga
    assert "NÃO são experiência" in pedido["sistema"] and "nivel=experiente" in pedido["sistema"]
    assert "PASTA DA TRIAGEM: VIGIA" in pedido["conteudos"] and '"nivel":"experiente"' in pedido["conteudos"]
    assert '"setores":["TI"]' in pedido["conteudos"]

    # com evidência de experiência profissional, a sugestão vale
    monkeypatch.setattr(ia, "_gemini_json", lambda *a, **k: {
        "vaga_id": 5, "confianca": "MEDIA", "motivo": "Atendimento e noções de suporte.", "perfil_indicado": "suporte júnior",
        "evidencias": ["Operador de caixa — Mercado Central (2024–2025)"], "lacunas": ["experiência formal com suporte"]})
    r = ia.sugerir_vaga(SEGURANCA_QUE_ESTUDA_TI, VAGAS, "VIGIA", 1)
    assert r["confianca"] == "MEDIA" and r["vaga_id"] == 5


def test_demonstracao_nao_sugere_vaga_experiente_de_outra_area():
    from app import ia
    # requisitos vagos ("ser bom em TI") não dão sugestão
    r = ia.sugerir_vaga("Segurança, controle de acesso. Estudo TI e sou bom em TI.", VAGAS[:1], "VIGIA", 1)
    assert r["confianca"] == "BAIXA" and any("vagos" in x for x in r["lacunas"])
    # vaga de nível experiente em outra área: termos em comum não bastam
    vaga = [{"id": 7, "titulo": "Analista de suporte", "area": "TI", "descricao": "Suporte técnico a usuários",
             "requisitos": ["Redes e infraestrutura", "Atendimento a chamados"], "setores": ["TI"]}]
    r = ia.sugerir_vaga("Vigilante. Estudo suporte técnico, redes e infraestrutura; atendimento a chamados no curso.",
                        vaga, "VIGIA", 1)
    assert r["confianca"] == "BAIXA" and any("outra área" in x for x in r["lacunas"])


def test_sugestao_antiga_pede_de_novo(cli):
    from app.models import Curriculo
    cur_id = cli.get("/api/banco?situacao=talentos").json()["itens"][0]["curriculo_id"]
    with sessao() as db:
        cur = db.get(Curriculo, cur_id)
        cur.triagem_json = {**(cur.triagem_json or {}), "sugestao_atlas": {"vaga_id": 1, "vaga": "Qualquer", "confianca": "MEDIA",
                                                                        "motivo": "critério antigo"}}
    d = cli.get(f"/api/banco/{cur_id}").json()
    assert d["sugestao_atlas"]["antiga"] and d["sugestao_atlas"]["vaga"] is None
    r = cli.post(f"/api/banco/{cur_id}/sugerir", headers=H)
    assert r.status_code == 200 and not r.json()["sugestao_atlas"].get("antiga")
    assert r.json()["sugestao_atlas"]["criterio"] == 2
