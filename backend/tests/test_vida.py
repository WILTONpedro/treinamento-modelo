"""A equipe com mais vida: personalidades humanas, segunda leitura entre colegas e momentos sociais."""
from sqlalchemy import select

from app import config, servicos as S
from app.db import sessao
from app.models import Agente, Analise, Evento
from app.personalidade import ARQUETIPOS, PERFIS, VERSAO_PERFIL, perfil_para
from conftest import H, enviar, esperar_fila

CV_PLANILHAS = """Joana Teste
Assistente administrativo · Campinas - SP
joana.teste@example.com
EXPERIÊNCIA
Assistente administrativo — Escritório Contábil Silva (2021–2024): controle de planilhas e organização de documentos.
Atendimento a clientes no balcão e por telefone.
FORMAÇÃO
Ensino médio completo."""


def eventos(tipo):
    with sessao() as db:
        return [(e.agente_id, e.mensagem, e.dados) for e in db.scalars(select(Evento).where(Evento.tipo == tipo)).all()]


def test_perfis_tem_historia_e_jeito():
    for p in list(PERFIS.values()) + ARQUETIPOS:
        for campo in ("historia", "jeito", "expressoes", "gostos", "saudacoes", "pedir_segunda"):
            assert p.get(campo), (p.get("emoji"), campo)
    assert perfil_para("Sofia")["versao"] == VERSAO_PERFIL and perfil_para("Nina")["historia"]


def test_perfil_antigo_no_banco_e_atualizado(cli):
    with sessao() as db:
        bia = db.scalar(select(Agente).where(Agente.nome == "Bia"))
        bia.perfil = {"emoji": "📎", "resumo": "perfil antigo"}
    with sessao() as db:
        S.semear_basico(db)
    with sessao() as db:
        assert db.scalar(select(Agente).where(Agente.nome == "Bia")).perfil["historia"]


def test_chat_de_colega(cli):
    ags = {a["nome"]: a for a in cli.get("/api/escritorio").json()["agentes"]}
    r = cli.post(f'/api/agentes/{ags["Flávio"]["id"]}/chat', headers=H, json={"mensagem": "tudo bem?"}).json()["resposta"]
    assert "?" in r  # devolve a pergunta, como gente
    r = cli.post(f'/api/agentes/{ags["Bia"]["id"]}/chat', headers=H, json={"mensagem": "valeu!"}).json()["resposta"]
    assert r and "registro" not in r.lower()
    r = cli.post(f'/api/agentes/{ags["Sofia"]["id"]}/chat', headers=H, json={"mensagem": "você é uma pessoa?"}).json()["resposta"]
    assert "agente de IA" in r
    r = cli.post(f'/api/agentes/{ags["Atlas"]["id"]}/chat', headers=H, json={"mensagem": "o que você faz no fim de semana?"}).json()["resposta"]
    assert "xadrez" in r.lower() or "café" in r.lower() or "bonsai" in r.lower()


def test_prompt_do_chat_fala_como_gente(cli):
    from app import ia
    ags = {a["nome"]: a for a in cli.get("/api/escritorio").json()["agentes"]}
    with sessao() as db:
        ctx = S.contexto_chat(db, db.get(Agente, ags["Flávio"]["id"]))
    sistema, prompt = ia._prompt_chat(ctx, "e aí, como tá?", [])
    assert "Quem você é:" in sistema and "churrasco" in sistema and "colega de trabalho" in sistema
    assert "AGORA:" in sistema and "agente de IA" in sistema
    assert '"historia"' not in prompt  # a personalidade vai no sistema, não repetida no contexto


def test_segunda_leitura_acha_o_que_passou_batido(cli, monkeypatch):
    monkeypatch.setattr(config, "SEGUNDA_LEITURA_CHANCE", 1.0)
    r = cli.post("/api/vagas", headers=H, json={"titulo": "Assistente administrativo", "setores": ["ADMINISTRATIVO"],
                                                "requisitos": ["Excel avançado", "Ensino médio completo"]})
    assert r.status_code == 200
    enviar(cli, "joana.txt", conteudo=CV_PLANILHAS)
    esperar_fila(cli, 60)
    pedidos = [d for _, _, d in eventos("consulta_pedida") if d.get("segunda_leitura")]
    assert pedidos, "o analista devia pedir uma segunda leitura (Excel ficou a verificar)"
    with sessao() as db:
        an = db.scalars(select(Analise).order_by(Analise.id.desc())).first()
        colab = (an.resultado or {}).get("colaboracoes", [])
    assert colab and colab[0]["tipo"] == "segunda_leitura"
    achados = colab[0]["achados"]
    assert any(a["tipo"] == "evidencia" and "planilha" in a["trecho"].lower() for a in achados)
    # o requisito saiu de "a verificar" e virou atendido com o trecho que o colega achou
    assert any(x["requisito"] == "Excel avançado" for x in an.resultado["requisitos_atendidos"])
    assert any("segunda leitura" in m for _, m, _ in eventos("consulta_respondida"))
