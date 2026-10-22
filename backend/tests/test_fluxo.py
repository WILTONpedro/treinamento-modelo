"""Teste ponta a ponta no modo demonstração (sem IA externa), com contratação manual.

Rode com:  cd backend && python -m pytest -q
"""
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from conftest import H, PASTA, entrar, enviar, esperar_fila


def test_permissoes_e_csrf(cli):
    # sem o cabeçalho X-Hub, escritas são recusadas
    assert cli.post("/api/vagas", json={"titulo": "X", "setores": ["TI"]}).status_code == 403
    r = cli.post("/api/usuarios", headers=H, json={"nome": "Rec", "email": "rec@teste.local", "senha": "senha-rec-1234", "papel": "recrutador"})
    assert r.status_code == 200
    with TestClient(app) as outro:
        entrar(outro, "rec@teste.local", "senha-rec-1234")
        r = outro.post("/api/vagas", headers=H, json={"titulo": "Vaga X", "setores": ["TI"]})
        assert r.status_code == 403 and "permissão" in r.json()["detail"]
        assert outro.get("/api/auditoria").status_code == 403
    assert TestClient(app).get("/api/escritorio").status_code == 401


def test_fluxo_completo(cli):
    for v in [
        {"titulo": "Assistente Administrativo", "setores": ["ADMINISTRATIVO"], "local": "Campinas - SP",
         "requisitos": ["Ensino médio completo", "Excel intermediário", "Experiência em rotinas administrativas", "Controle de documentos"]},
        {"titulo": "Vendedor Externo", "setores": ["VENDAS", "PROMOTOR DE VENDAS"],
         "requisitos": ["CNH categoria B", "Experiência em vendas externas"]},
    ]:
        assert cli.post("/api/vagas", headers=H, json=v).status_code == 200

    # a API de triagem exige a chave e mantém o contrato original
    assert enviar(cli, "01_ana_martins.txt", chave=None).status_code == 401
    r = enviar(cli, "01_ana_martins.txt").json()
    assert r["setor_sugerido"] == "ADMINISTRATIVO" and r["nome_identificado"] == "Ana Martins"
    assert {"arquivo", "confianca", "detalhes"} <= r.keys()
    # reenvio do mesmo arquivo não gera nova candidatura
    r2 = enviar(cli, "09_ana_martins_reenvio.txt").json()
    assert r2["curriculo_id"] == r["curriculo_id"]
    for nome in ["05_carta_apresentacao_lucia.txt", "06_paula_souza.txt", "08_renan_alves.txt", "12_fernanda_costa.txt"]:
        enviar(cli, nome)
    esperar_fila(cli)

    esc = cli.get("/api/escritorio").json()
    ag = {a["nome"]: a for a in esc["agentes"]}
    vagas = {v["titulo"]: v for v in esc["vagas"]}
    assert vagas["Assistente Administrativo"]["recebidos"] == 2  # Ana + Fernanda (sem duplicar a Ana)
    assert vagas["Vendedor Externo"]["sem_responsavel"]
    assert any(e["tipo"] == "arquivo_invalido" for e in esc["eventos"])

    # candidatos da Sofia, com análise fundamentada em trechos reais do currículo
    cands = cli.get(f'/api/agentes/{ag["Sofia"]["id"]}/candidatos').json()
    ana = next(c for c in cands if c["nome"] == "Ana Martins")
    assert ana["recomendado"] and ana["compatibilidade"] == "ALTA"
    det = cli.get(f'/api/candidaturas/{ana["candidatura_id"]}').json()
    for ev in det["analises"][0]["atendidos"]:
        assert ev["evidencia"] in det["curriculo"]["texto"]
    fer = next(c for c in cands if c["nome"] == "Fernanda Costa")
    assert fer["avaliacao_humana"] and fer["a_verificar"]

    # o recrutador decide (a IA nunca decide)
    assert cli.post(f'/api/candidaturas/{ana["candidatura_id"]}/decisao', headers=H,
                    json={"decisao": "entrevista"}).status_code == 200

    # o Atlas sinalizou que falta especialista comercial; o recrutador ativa um agente
    q = cli.get("/api/quadro").json()
    p = next(p for p in q["pendencias"] if p["tipo"] == "SEM_ESPECIALISTA")
    assert cli.post(f'/api/pendencias/{p["id"]}/resolver', headers=H, json={"acao": "ativar"}).status_code == 200
    esperar_fila(cli)
    esc = cli.get("/api/escritorio").json()
    novo = next(a for a in esc["agentes"] if a["especialidade_codigo"] == "comercial")
    assert novo["vagas"]
    assert len(cli.get(f'/api/agentes/{novo["id"]}/candidatos').json()) == 2
    # o currículo do promotor fez o analista pedir ajuda a promotores; como não há ninguém, o Atlas anotou a necessidade
    evs = [e["mensagem"] for e in cli.get(f'/api/agentes/{ag["Atlas"]["id"]}').json()["historico"]]
    assert any("precisou de alguém de Promotores" in m for m in evs)
    renan = next(c for c in cli.get(f'/api/agentes/{novo["id"]}/candidatos').json() if c["nome"] == "Renan Alves")
    assert any("promotores" in x for x in renan["a_verificar"])

    # chat responde a partir dos registros
    resp = cli.post(f'/api/agentes/{ag["Sofia"]["id"]}/chat', headers=H, json={"mensagem": "Quantos currículos faltam?"}).json()
    assert "0 currículo" in resp["resposta"]
    resp = cli.post(f'/api/agentes/{ag["Sofia"]["id"]}/chat', headers=H, json={"mensagem": "Por que recomendou a Ana?"}).json()
    assert "Ana Martins" in resp["resposta"]

    # LGPD: acesso e eliminação
    tit = cli.get("/api/lgpd/titulares", params={"busca": "ana.martins"}).json()
    assert len(tit) == 1
    exp = cli.get(f'/api/lgpd/titulares/{tit[0]["id"]}/exportar')
    assert exp.status_code == 200 and "Ana Martins" in exp.text
    assert cli.post(f'/api/lgpd/titulares/{tit[0]["id"]}/eliminar', headers=H, json={}).status_code == 200
    assert cli.get(f'/api/candidaturas/{ana["candidatura_id"]}').status_code == 404
    assert cli.get(f'/api/curriculos/{r["curriculo_id"]}/arquivo').status_code == 404
    todos = cli.get("/api/escritorio").json()["eventos"]
    assert not any("Ana M." in e["mensagem"] for e in todos)
    # o mesmo arquivo pode voltar a ser enviado depois da eliminação (novo processo)
    assert enviar(cli, "01_ana_martins.txt").json()["curriculo_id"] != r["curriculo_id"]

    acoes = {a["acao"] for a in cli.get("/api/auditoria").json()}
    assert {"ver_curriculo", "decidir", "exportar_titular", "eliminar_titular"} <= acoes


def test_arquivos_cifrados_em_disco(cli):
    for arq in Path(PASTA, "arquivos").glob("*.bin"):
        assert b"Excel" not in arq.read_bytes()
