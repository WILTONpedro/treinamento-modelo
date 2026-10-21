"""Equipe viva: personalidades, pensamentos, consultas entre colegas, contratações do Atlas,
dispensas, canal da equipe e automações. Modo demonstração (sem IA externa)."""
from datetime import timedelta

from sqlalchemy import create_engine, inspect, select, text

from app import config
from app import servicos as S
from app.db import migrar, sessao
from app.models import Curriculo, Evento, Pendencia, agora
from conftest import H, enviar, esperar, esperar_fila

VAGAS = [
    {"titulo": "Operador de Produção", "setores": ["PRODUÇÃO", "LOGÍSTICA"],
     "requisitos": ["Ensino fundamental completo", "Experiência em linha de produção", "Disponibilidade para turnos"]},
    {"titulo": "Assistente Administrativo", "setores": ["ADMINISTRATIVO"],
     "requisitos": ["Ensino médio completo", "Excel intermediário", "Experiência em rotinas administrativas", "Controle de documentos"]},
    {"titulo": "Vendedor Externo", "setores": ["VENDAS", "COMERCIAL", "PROMOTOR DE VENDAS"],
     "requisitos": ["CNH categoria B", "Experiência em vendas externas"]},
]


def agentes(c):
    return {a["nome"]: a for a in c.get("/api/escritorio").json()["agentes"]}


def da_especialidade(c, codigo):
    return [a for a in c.get("/api/escritorio").json()["agentes"] if a["especialidade_codigo"] == codigo]


def eventos_do_tipo(tipo):
    with sessao() as db:
        return [(e.agente_id, e.mensagem, e.dados) for e in db.scalars(select(Evento).where(Evento.tipo == tipo)).all()]


def test_personalidades(cli):
    ags = agentes(cli)
    assert {"Atlas", "Bia", "Sofia", "Flávio"} <= ags.keys()
    assert ags["Atlas"]["perfil"]["emoji"] == "♟️"
    for a in ags.values():
        p = a["perfil"]
        assert p["resumo"] and p["bordoes"] and p["papo"] and p["pensando"]
    assert ags["Sofia"]["perfil"]["tom"] != ags["Flávio"]["perfil"]["tom"]
    det = cli.get(f'/api/agentes/{ags["Flávio"]["id"]}').json()
    assert det["perfil_completo"]["apresentacao"]


def test_atlas_contrata_quem_falta_e_colegas_se_ajudam(cli, monkeypatch):
    monkeypatch.setattr(config, "CONTRATACAO_AUTOMATICA", True)
    for v in VAGAS:
        assert cli.post("/api/vagas", headers=H, json=v).status_code == 200

    # ninguém cobre vendas: o Atlas contrata sozinho na rodada de supervisão
    comercial = esperar(lambda: da_especialidade(cli, "comercial"), msg="o Atlas não contratou o especialista comercial")[0]
    assert comercial["vagas"] and comercial["perfil"]["resumo"]
    contr = cli.get("/api/contratacoes").json()
    assert any(c["acao"] == "contratacao" and c["motivo"] == "sem_especialista" and comercial["nome"] in c["descricao"] for c in contr)
    assert not [p for p in cli.get("/api/quadro").json()["pendencias"] if p["tipo"] == "SEM_ESPECIALISTA"]
    falas = [m for _, m, d in eventos_do_tipo("conversa")]
    assert any(f"Boas-vindas ao time, {comercial['nome']}" in m for m in falas)  # o Atlas recebe o novo colega
    assert any(m.startswith(f"Oi, time! {comercial['nome']} aqui") for m in falas)  # que se apresenta

    # Priscila trabalha em almoxarifado (pasta LOGÍSTICA → Flávio), mas tem perfil administrativo:
    # Flávio pede ajuda à Sofia, que responde e a inclui também na vaga administrativa
    assert enviar(cli, "13_priscila_almeida.txt").json()["setor_sugerido"] == "LOGÍSTICA"
    esperar_fila(cli)
    ags = agentes(cli)
    pedidos = eventos_do_tipo("consulta_pedida")
    assert any(a == ags["Flávio"]["id"] and d["destino_id"] == ags["Sofia"]["id"] for a, _, d in pedidos)
    assert eventos_do_tipo("consulta_respondida") and eventos_do_tipo("indicacao")

    flavio = cli.get(f'/api/agentes/{ags["Flávio"]["id"]}/candidatos').json()
    pri = next(c for c in flavio if c["nome"] == "Priscila Almeida")
    assert pri["colaboracoes"] and pri["colaboracoes"][0]["colega"] == "Sofia"
    assert "Assistente Administrativo" in pri["colaboracoes"][0]["resposta"]
    sofia = cli.get(f'/api/agentes/{ags["Sofia"]["id"]}/candidatos').json()
    pri_s = next(c for c in sofia if c["nome"] == "Priscila Almeida")
    assert pri_s["origem"] == "indicacao" and pri_s["recomendado"]
    assert not pri_s["colaboracoes"]  # indicada pelo Flávio: a Sofia não devolve a pergunta para ele
    det = cli.get(f'/api/candidaturas/{pri_s["candidatura_id"]}').json()
    assert det["indicado_por"] == "Flávio"

    # pensamentos reais da leitura e conversa registrada no canal da equipe
    pens = [m for a, m, _ in eventos_do_tipo("pensamento") if a == ags["Flávio"]["id"]]
    assert any("Priscila A." in m for m in pens)
    canal = cli.get("/api/equipe/conversas").json()
    conversa = [e for e in canal["eventos"] if e["tipo"] == "conversa"]
    assert any(e["agente_id"] == ags["Flávio"]["id"] and e["dados"]["destino_id"] == ags["Sofia"]["id"] for e in conversa)
    assert str(ags["Sofia"]["id"]) in {str(k) for k in canal["agentes"]}
    assert cli.get("/api/quadro").json()["resumo"]["consultas_entre_colegas"] >= 1
    # o feed de atividade não fica poluído com pensamentos
    assert not any(e["tipo"] == "pensamento" for e in cli.get("/api/escritorio").json()["eventos"])


def test_atlas_reforca_o_time_quando_a_fila_cresce(cli, monkeypatch):
    monkeypatch.setattr(config, "CONTRATACAO_AUTOMATICA", True)
    monkeypatch.setattr(config, "LIMITE_FILA_POR_AGENTE", 1)
    monkeypatch.setattr(config, "INTERVALO_CONTRATACAO", 0)
    monkeypatch.setattr(config, "DEMO_DURACAO_ANALISE", 0.6)
    for nome in ["14_lote_wesley_rocha.txt", "15_lote_kelly_matos.txt", "16_lote_andre_luz.txt", "17_lote_bruna_dias.txt"]:
        enviar(cli, nome)
    ops = esperar(lambda: len(da_especialidade(cli, "operacional")) >= 2 and da_especialidade(cli, "operacional"),
                  msg="o Atlas não reforçou o operacional")
    novo = next(a for a in ops if a["nome"] != "Flávio")
    esperar_fila(cli)
    assert any(c["motivo"] == "sobrecarga" for c in cli.get("/api/contratacoes").json())
    det = cli.get(f'/api/agentes/{novo["id"]}').json()
    assert any(v["titulo"] == "Operador de Produção" for v in det["vagas_detalhe"])  # divide a vaga com o Flávio
    assert det["estatisticas"]["analisados"] >= 1  # e pegou parte da fila


def test_dispensas(cli, monkeypatch):
    ags = agentes(cli)
    reforcos = [a for a in ags.values() if a["especialidade_codigo"] == "operacional" and a["nome"] != "Flávio"]
    assert reforcos
    for a in reforcos:
        assert cli.post(f'/api/agentes/{a["id"]}/dispensar', headers=H).status_code == 200
    # o Flávio ficou sozinho com a vaga: não pode sair sem alguém para assumir
    r = cli.post(f'/api/agentes/{ags["Flávio"]["id"]}/dispensar', headers=H)
    assert r.status_code == 400 and "Operador de Produção" in r.json()["detail"]
    assert not {a["nome"] for a in reforcos} & agentes(cli).keys()
    assert any(c["acao"] == "dispensa" for c in cli.get("/api/contratacoes").json())

    # com todas as estações ocupadas, o Atlas libera quem está sem vagas para contratar quem falta
    monkeypatch.setattr(config, "CONTRATACAO_AUTOMATICA", True)
    esps = {e["codigo"]: e["id"] for e in cli.get("/api/especialidades").json()}
    for codigo in ["ti", "marketing", "promotores", "inclusao"] * 4:
        if agentes(cli) and sum(1 for a in agentes(cli).values() if a["funcao"] == "analista") >= S.MAX_ESTACOES:
            break
        assert cli.post("/api/agentes", headers=H, json={"especialidade_id": esps[codigo]}).status_code == 200
    analistas = [a for a in agentes(cli).values() if a["funcao"] == "analista"]
    assert len(analistas) == S.MAX_ESTACOES
    r = cli.post("/api/agentes", headers=H, json={"especialidade_id": esps["ti"]})
    assert r.status_code == 400 and "estações" in r.json()["detail"]  # o recrutador não força a saída de ninguém
    assert cli.post("/api/vagas", headers=H, json={"titulo": "Gerente de Produção", "setores": ["GERENTE PRODUÇÃO"]}).status_code == 200
    lider = esperar(lambda: da_especialidade(cli, "lideranca"), msg="o Atlas não liberou estação para liderança")[0]
    assert lider["vagas"]
    assert sum(1 for a in agentes(cli).values() if a["funcao"] == "analista") == S.MAX_ESTACOES
    assert any(d.get("por") == "Atlas" for _, _, d in eventos_do_tipo("agente_dispensado"))


def test_canal_da_equipe(cli):
    ags = agentes(cli)
    r = cli.post("/api/equipe/chat", headers=H, json={"mensagem": "Como está o dia?"}).json()
    quem = [x["agente_id"] for x in r["respostas"]]
    assert quem[0] == ags["Atlas"]["id"] and len(quem) >= 2  # o Atlas abre e chama quem tem vagas
    assert all(x["texto"] and x["nome"] for x in r["respostas"])
    r = cli.post("/api/equipe/chat", headers=H, json={"mensagem": "Bia, quantos currículos chegaram?"}).json()
    assert r["respostas"][0]["agente_id"] == ags["Bia"]["id"] and "currículo" in r["respostas"][0]["texto"]
    r = cli.post("/api/equipe/chat", headers=H, json={"mensagem": "Precisamos contratar alguém?"}).json()
    assert r["respostas"][0]["agente_id"] == ags["Atlas"]["id"] and "contrat" in r["respostas"][0]["texto"]
    r = cli.post("/api/equipe/chat", headers=H, json={"mensagem": "E a Priscila?"}).json()
    assert r["respostas"][0]["agente_id"] in (ags["Flávio"]["id"], ags["Sofia"]["id"])


def test_chat_em_streaming_com_personalidade(cli):
    ags = agentes(cli)
    r = cli.post(f'/api/agentes/{ags["Sofia"]["id"]}/chat/stream', headers=H, json={"mensagem": "Oi!"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    assert r.text == ags["Sofia"]["perfil"]["saudacao"]
    r = cli.post(f'/api/agentes/{ags["Flávio"]["id"]}/chat/stream', headers=H, json={"mensagem": "Você pediu ajuda a algum colega?"})
    assert "Priscila" in r.text and "Sofia" in r.text
    r = cli.post(f'/api/agentes/{ags["Atlas"]["id"]}/chat', headers=H, json={"mensagem": "Quem você contratou?"}).json()
    assert "contrat" in r["resposta"]


def test_atlas_encaminha_curriculo_sem_pasta(cli):
    assert cli.post("/api/vagas", headers=H, json={
        "titulo": "Instrutor de Idiomas", "setores": ["RECURSOS HUMANOS"], "descricao": "Aulas de inglês para equipes internas.",
        "requisitos": ["Experiência com aulas de inglês", "Licenciatura em Letras"]}).status_code == 200
    r = enviar(cli, "10_tiago_moreira.txt").json()
    assert r["setor_sugerido"] == "OUTROS"
    esperar_fila(cli)
    assert any("Instrutor de Idiomas" in m for _, m, _ in eventos_do_tipo("encaminhamento_automatico"))
    assert not [p for p in cli.get("/api/quadro").json()["pendencias"] if p["tipo"] == "TRIAGEM_MANUAL"]
    # sem vaga parecida, continua pedindo ajuda ao recrutador
    enviar(cli, "x.txt", conteudo="Marta Lima\nAstrônoma\nmarta@example.com\nObservação de estrelas e telescópios desde 2010–2024")
    esperar_fila(cli)
    assert [p for p in cli.get("/api/quadro").json()["pendencias"] if p["tipo"] == "TRIAGEM_MANUAL"]


def test_atlas_sugere_pastas_da_vaga(cli):
    r = cli.post("/api/vagas/sugerir", headers=H, json={"titulo": "Operador de empilhadeira", "descricao": "Movimentação no armazém, NR-11"}).json()
    assert "EMPILHADEIRA" in r["setores"]
    v = cli.post("/api/vagas", headers=H, json={"titulo": "Motorista entregador", "descricao": "Entregas com CNH categoria D"}).json()
    assert "MOTORISTA" in v["setores"]
    assert cli.post("/api/vagas", headers=H, json={"titulo": "Vaga misteriosa"}).status_code == 400


def test_atlas_tenta_de_novo_sozinho(cli, monkeypatch):
    with sessao() as db:
        cur = Curriculo(sha256="teste-erro-ia", arquivo_nome="x.txt", status="erro", texto="Fulano\nauxiliar de produção")
        db.add(cur)
        db.flush()
        db.add(Pendencia(tipo="ERRO_IA", descricao="falha simulada", curriculo_id=cur.id,
                         criado_em=agora() - timedelta(hours=1)))
        cid = cur.id
    assert S.reprocessar_automatico() == 1
    with sessao() as db:
        assert db.get(Curriculo, cid).status in ("recebido", "triado")
        for _ in range(2):  # mais duas falhas: na terceira, o Atlas deixa para o recrutador
            db.add(Pendencia(tipo="ERRO_IA", descricao="falha simulada", curriculo_id=cid, criado_em=agora() - timedelta(hours=1)))
            db.flush()
            db.commit()
            S.reprocessar_automatico()
    assert S.reprocessar_automatico() == 0
    assert any(p["descricao"] == "falha simulada" for p in cli.get("/api/quadro").json()["pendencias"])


def test_boletim_do_atlas(cli, monkeypatch):
    monkeypatch.setattr(config, "BOLETIM_MINUTOS", 1)
    assert S.boletim() is True
    assert S.boletim() is False  # um por período
    canal = cli.get("/api/equipe/conversas").json()["eventos"]
    assert any(e["mensagem"].startswith("Boletim:") for e in canal)


def test_lgpd_apaga_pensamentos_e_conversas(cli):
    tit = cli.get("/api/lgpd/titulares", params={"busca": "priscila"}).json()
    exp = cli.get(f'/api/lgpd/titulares/{tit[0]["id"]}/exportar').json()
    assert exp["conversas_dos_agentes"]  # acesso do titular inclui o que os agentes falaram sobre ele
    assert cli.post(f'/api/lgpd/titulares/{tit[0]["id"]}/eliminar', headers=H, json={}).status_code == 200
    with sessao() as db:
        assert not db.scalars(select(Evento).where(Evento.candidato_id == tit[0]["id"], Evento.tipo.in_(S.FALAS))).all()
        assert not [e for e in db.scalars(select(Evento)).all() if "Priscila" in e.mensagem]
    assert "Priscila" not in str(cli.get("/api/equipe/conversas").json())


def test_conversas_antigas_sao_apagadas(cli):
    with sessao() as db:
        db.add(Evento(tipo="conversa", mensagem="antiga", ts=agora() - timedelta(hours=config.RETENCAO_CONVERSAS_HORAS + 1)))
    assert S.limpar_conversas() >= 1
    with sessao() as db:
        assert not db.scalars(select(Evento).where(Evento.mensagem == "antiga")).all()


def test_migracao_de_banco_antigo(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'antigo.db'}")
    with eng.begin() as con:
        con.execute(text("CREATE TABLE eventos (id INTEGER PRIMARY KEY, ts DATETIME, tipo VARCHAR(40), agente_id INTEGER, "
                         "mensagem TEXT, dados JSON)"))
        con.execute(text("INSERT INTO eventos (tipo, mensagem, dados) VALUES ('x', 'ok', '{}')"))
    migrar(eng)
    cols = {c["name"] for c in inspect(eng).get_columns("eventos")}
    assert "candidato_id" in cols
    assert any(i["column_names"] == ["candidato_id"] for i in inspect(eng).get_indexes("eventos"))
    migrar(eng)  # idempotente
