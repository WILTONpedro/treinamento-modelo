"""A equipe com mais vida: personalidades humanas, segunda leitura entre colegas e momentos sociais."""
import pytest
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
        for campo in ("historia", "jeito", "expressoes", "gostos", "saudacoes", "pedir_segunda", "convida", "comemora", "perde"):
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



def test_segunda_leitura_nao_apaga_anotacao_de_ajuda():
    """A anotação "ponto para alguém de promotores" não é requisito: a segunda leitura não pode dar como resolvida."""
    from app import ia
    vaga = {"titulo": "Vendedor", "requisitos": ["Experiência com vendas"]}
    prelim = {"a_verificar": ["Ponto para alguém de promotores: experiência como promotor em supermercado"],
              "requisitos_atendidos": [{"requisito": "Experiência com vendas", "evidencia": "Vendedor — Loja X (2020–2023)"}]}
    r = ia._segunda_leitura_demo({"nome": "Bia"}, vaga, "Renan\nPromotor de vendas — Mercado Y (2019–2020)", prelim, "Renan A.")
    assert not r["reforca_requisito"] and not [a for a in r["achados"] if a["tipo"] == "evidencia"]
    final = ia._revisar_demo(prelim, r | {"colega": "Bia"})
    assert any("promotores" in x for x in final["a_verificar"])


# --------------------------------------------------------------------------- momentos da equipe
NOS_VALIDOS = {"ping0", "ping1", "comer0", "comer1", "comer2", "comer3", "comer4", "comer5", "bebedouro", "papoAgua",
               "sofa1", "sofa2", "leitura0", "leitura1", "alonga0", "alonga1", "alonga2", "alonga3", "alonga4",
               "reuniao0", "reuniao1", "reuniao2", "reuniao3"}


def _falas_do(mid):
    return sorted(((d["momento"]["ordem"], ag, m) for ag, m, d in eventos("conversa")
                   if (d.get("momento") or {}).get("id") == mid), key=lambda x: x[0])


@pytest.fixture
def momentos(cli):
    from app import momentos as M
    M._estado["ocupado_ate"].clear()
    esperar_fila(cli, 60)
    yield M
    M._estado["ocupado_ate"].clear()


def test_momento_de_pingue_pongue(momentos):
    m = momentos.talvez_um_momento(forcar="pingue")
    assert m and m["tipo"] == "pingue" and m["duracao"] >= 30
    assert sorted(m["lugares"].values()) == ["ping0", "ping1"]
    falas = _falas_do(m["id"])
    assert len(falas) >= 3 and {ag for _, ag, _ in falas} == {int(k) for k in m["lugares"]}
    assert all(txt and "{" not in txt for _, _, txt in falas)
    # quem está no momento não entra em outro enquanto ele dura
    ocupados = {int(k) for k in m["lugares"]}
    m2 = momentos.talvez_um_momento(forcar="cafe")
    assert not m2 or not ocupados & {int(k) for k in m2["lugares"]}


def test_todos_os_momentos_tem_lugar_e_papo(momentos):
    for tipo in momentos.TIPOS:
        momentos._estado["ocupado_ate"].clear()
        m = momentos.talvez_um_momento(forcar=tipo)
        assert m, tipo
        assert set(m["lugares"].values()) <= NOS_VALIDOS, (tipo, m["lugares"])
        assert len(set(m["lugares"].values())) == len(m["lugares"])  # um lugar por pessoa
        assert len(_falas_do(m["id"])) >= 2, tipo
        if tipo == "aniversario":
            assert "aniversário de" in m["rotulo"]


def test_momento_precisa_de_gente_livre(momentos, monkeypatch):
    with sessao() as db:
        um = db.scalars(select(Agente).where(Agente.ativo.is_(True))).first()
    monkeypatch.setattr(momentos, "_livres", lambda db: [db.get(Agente, um.id)])
    assert momentos.talvez_um_momento(forcar="pingue") is None
    monkeypatch.setattr(momentos, "_livres", lambda db: [])
    assert momentos.talvez_um_momento(forcar="cafe") is None


def test_ronda_do_atlas(momentos, monkeypatch):
    with sessao() as db:
        analista = db.scalars(select(Agente).where(Agente.funcao == "analista", Agente.estacao.is_not(None))).first()
        aid = analista.id
    monkeypatch.setattr(momentos, "_trabalhando", lambda db: [db.get(Agente, aid)])
    m = momentos.talvez_um_momento(forcar="ronda")
    assert m and m["tipo"] == "ronda"
    with sessao() as db:
        atlas = S.atlas(db)
        assert m["lugares"] == {str(atlas.id): [f"c{db.get(Agente, aid).estacao}"]}
    falas = _falas_do(m["id"])
    assert falas and falas[0][1] == atlas.id


def test_momento_com_papo_da_ia(momentos, monkeypatch):
    from app import ia
    pedido = {}

    def falso(situacao, pessoas, fatos):
        pedido.update(situacao=situacao, pessoas=pessoas, fatos=fatos)
        a, b = pessoas[0]["nome"], pessoas[1]["nome"]
        return [{"agente": a.upper(), "texto": "Saque cortado, cuidado!"}, {"agente": b, "texto": "Pode vir, tô aquecido."},
                {"agente": "Fulano", "texto": "não é do time"}, {"agente": a, "texto": "Onze a sete. Revanche?"}]
    monkeypatch.setattr(config, "IA_MODO", "gemini")
    monkeypatch.setattr(config, "MOMENTOS_IA_CHANCE", 1.0)
    monkeypatch.setattr(ia, "momento_conversa", falso)
    m = momentos.talvez_um_momento(forcar="pingue")
    assert m and "pingue-pongue" in pedido["situacao"]
    assert all(p.get("historia") and p.get("jeito") for p in pedido["pessoas"])  # cada um com o seu jeito
    assert "hora" in pedido["fatos"] and "vagas_abertas" in pedido["fatos"]
    assert [t for _, _, t in _falas_do(m["id"])] == ["Saque cortado, cuidado!", "Pode vir, tô aquecido.", "Onze a sete. Revanche?"]

    # a IA falhou: fica o roteiro
    monkeypatch.setattr(ia, "momento_conversa", lambda *a: (_ for _ in ()).throw(RuntimeError("fora do ar")))
    momentos._estado["ocupado_ate"].clear()
    m = momentos.talvez_um_momento(forcar="bebedouro")
    assert m and len(_falas_do(m["id"])) >= 2
