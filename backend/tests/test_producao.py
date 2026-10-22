"""Pronto para produção: usuários e senhas, configuração segura, preparo do .env e uma IA real
(Gemini) simulada — consumo de tokens, pensamento mínimo e pausa quando o Google recusa."""
import json
import os
import stat
import time

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from google.genai import errors
from sqlalchemy import select

from app import config, ia, preparar, triagem, uso_ia
from app.db import sessao
from app.main import app, verificar_producao
from app.models import Pendencia, Tarefa, UsoIA
from conftest import H, entrar, enviar, esperar


# --------------------------------------------------------------------------- usuários e senhas
def test_usuario_novo_recebe_senha_temporaria_e_troca_no_primeiro_acesso(cli):
    r = cli.post("/api/usuarios", headers=H, json={"nome": "Rita RH", "email": "rita@teste.local", "papel": "recrutador"})
    assert r.status_code == 200
    temp, uid = r.json()["senha_temporaria"], r.json()["id"]
    with TestClient(app) as rita:
        assert rita.post("/auth/login", json={"email": "rita@teste.local", "senha": temp}).json()["trocar_senha"]
        r = rita.get("/api/escritorio")
        assert r.status_code == 403 and "senha temporária" in r.json()["detail"]  # nada antes de trocar
        assert rita.post("/auth/senha", headers=H, json={"senha_atual": temp, "nova_senha": "1234567890"}).status_code == 400
        assert rita.post("/auth/senha", headers=H, json={"senha_atual": "errada", "nova_senha": "rita-senha-forte-1"}).status_code == 400
        assert rita.post("/auth/senha", headers=H, json={"senha_atual": temp, "nova_senha": "rita-senha-forte-1"}).status_code == 200
        assert rita.get("/api/escritorio").status_code == 200

        # o admin redefine a senha: a sessão antiga cai na hora
        nova_temp = cli.post(f"/api/usuarios/{uid}/senha", headers=H).json()["senha_temporaria"]
        assert rita.get("/api/escritorio").status_code == 401
        entrar(rita, "rita@teste.local", nova_temp, nova="rita-senha-forte-2")
        assert rita.get("/api/escritorio").status_code == 200

        # desativada: perde o acesso imediatamente
        assert cli.patch(f"/api/usuarios/{uid}", headers=H, json={"ativo": False}).status_code == 200
        assert rita.get("/api/escritorio").status_code == 401
    lista = {u["email"]: u for u in cli.get("/api/usuarios").json()}
    assert lista["rita@teste.local"]["ativo"] is False and lista["rita@teste.local"]["ultimo_acesso_em"]


def test_sempre_sobra_um_administrador(cli):
    eu = cli.get("/auth/eu").json()
    r = cli.patch(f'/api/usuarios/{eu["id"]}', headers=H, json={"ativo": False})
    assert r.status_code == 400
    r = cli.patch(f'/api/usuarios/{eu["id"]}', headers=H, json={"papel": "recrutador"})
    assert r.status_code == 400
    outro = cli.post("/api/usuarios", headers=H, json={"nome": "Outro Admin", "email": "admin2@teste.local", "papel": "admin"}).json()
    assert cli.patch(f'/api/usuarios/{outro["id"]}', headers=H, json={"papel": "supervisor"}).status_code == 200
    assert cli.post("/api/usuarios", headers=H, json={"nome": "X", "email": "nao-e-email", "papel": "admin"}).status_code == 422


# --------------------------------------------------------------------------- configuração de produção
def test_producao_recusa_configuracao_insegura(monkeypatch):
    monkeypatch.setattr(config, "COOKIE_SECURE", False)
    monkeypatch.setattr(config, "CHAVE_ARQUIVOS", "")
    monkeypatch.setattr(config, "SEED_DEMO", True)
    problemas = " | ".join(config.problemas_producao())
    for trecho in ("COOKIE_SECURE", "CHAVE_ARQUIVOS", "SEED_DEMO", "IA_MODO"):
        assert trecho in problemas
    monkeypatch.setattr(config, "AMBIENTE", "producao")
    with pytest.raises(SystemExit, match="Configuração insegura"):
        verificar_producao()  # roda no início do ciclo de vida da API


def test_preparar_gera_env_com_segredos_fortes(tmp_path):
    destino = tmp_path / ".env"
    assert preparar.main(["--docker", "--dominio", "rh.exemplo.com.br", "--empresa", "Exemplo SA",
                          "--gemini-key", "chave-x", "--arquivo", str(destino)]) == 0
    env = dict(l.split("=", 1) for l in destino.read_text().splitlines() if l and not l.startswith("#") and "=" in l)
    assert env["AMBIENTE"] == "producao" and env["COOKIE_SECURE"] == "true" and env["IA_MODO"] == "gemini"
    assert len(env["SEGREDO_JWT"]) >= 48 and len(env["TRIAGEM_API_KEY"]) >= 32
    Fernet(env["CHAVE_ARQUIVOS"].encode())  # chave válida
    assert env["DATABASE_URL"].startswith("postgresql+psycopg://hub:") and env["POSTGRES_PASSWORD"] in env["DATABASE_URL"]
    assert env["DOMINIO"] == "rh.exemplo.com.br" and env["EMPRESA_NOME"] == "Exemplo SA"
    assert env["COMPOSE_PROFILES"] == "banco-local"  # PostgreSQL próprio sobe junto
    assert stat.S_IMODE(os.stat(destino).st_mode) == 0o600
    assert preparar.main(["--arquivo", str(destino)]) == 1  # não sobrescreve


def test_preparar_com_supabase(tmp_path):
    destino, url = tmp_path / ".env", "postgresql://hub_app.ref:senha@aws-0-sa-east-1.pooler.supabase.com:5432/postgres"
    assert preparar.main(["--docker", "--supabase", url, "--gemini-key", "k", "--arquivo", str(destino)]) == 0
    env = dict(l.split("=", 1) for l in destino.read_text().splitlines() if l and not l.startswith("#") and "=" in l)
    assert env["DATABASE_URL"] == url and env["DB_SCHEMA"] == "hub" and env["ARMAZENAMENTO"] == "banco"
    assert "COMPOSE_PROFILES" not in env and "POSTGRES_PASSWORD" not in env  # sem banco local
    assert config._url_banco(url) == url.replace("postgresql://", "postgresql+psycopg://") + "?sslmode=require"
    assert config._url_banco("postgres://u:p@h:5432/db?x=1") == "postgresql+psycopg://u:p@h:5432/db?x=1"


def test_arquivos_cifrados_no_banco(cli, monkeypatch):
    from app import arquivos
    from app.models import ArquivoCurriculo, Curriculo
    monkeypatch.setattr(config, "ARMAZENAMENTO", "banco")
    original = "Currículo de Teste Banco\nexperiência: auxiliar administrativo, Excel avançado".encode()
    r = enviar(cli, "teste_banco.txt", conteudo=original.decode())
    cid = r.json()["curriculo_id"]
    with sessao() as db:
        cur = db.get(Curriculo, cid)
        assert cur.caminho == "banco:" + cur.sha256
        guardado = db.get(ArquivoCurriculo, cur.sha256).conteudo
        assert original not in guardado and len(guardado) < len(original) * 2 + 120  # cifrado e sem base64
    assert cli.get(f"/api/curriculos/{cid}/arquivo").content == original
    with sessao() as db:  # eliminar o titular apaga o arquivo do banco
        cur = db.get(Curriculo, cid)
        sha = cur.sha256
        arquivos.apagar_arquivo(cur.caminho, db)
    with sessao() as db:
        assert db.get(ArquivoCurriculo, sha) is None


def test_marca_configuravel(cli, monkeypatch):
    monkeypatch.setattr(config, "APP_NOME", "Talentos")
    monkeypatch.setattr(config, "DA_EMPRESA", "da empresa Acme")
    assert cli.get("/api/marca").json()["nome"] == "Talentos"
    assert "Recrutador Sênior da empresa Acme" in triagem.montar_prompt()
    pagina = cli.get("/").text  # o nome vem de /api/marca: nada fixo na página
    assert 'data-marca="nome"' in pagina and "Talentos" not in pagina


# --------------------------------------------------------------------------- IA real (Gemini) simulada
class _Uso:
    def __init__(self, entrada, saida, pensamento=0, cache=0):
        self.prompt_token_count, self.candidates_token_count = entrada, saida
        self.thoughts_token_count, self.cached_content_token_count = pensamento, cache


class _Resp:
    def __init__(self, texto, uso):
        self.text, self.usage_metadata, self.candidates = texto, uso, [object()]


class GeminiFalso:
    """Imita o cliente google-genai: responde conforme o esquema pedido e registra o que recebeu."""

    def __init__(self):
        self.erro = None          # exceção a lançar em toda chamada (ex.: 402 sem créditos)
        self.niveis_recusados = set()
        self.chamadas = []
        self.models = self

    def generate_content(self, model, contents, config):
        nivel = getattr(getattr(config, "thinking_config", None), "thinking_level", None)
        nivel = getattr(nivel, "value", nivel)
        self.chamadas.append({"modelo": model, "nivel": nivel, "max": config.max_output_tokens,
                              "sistema": config.system_instruction, "conteudos": contents})
        if self.erro:
            raise self.erro
        if nivel and nivel.lower() in self.niveis_recusados:
            raise errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                                     "message": f"Thinking level {nivel} is not supported for this model."}})
        sch = getattr(config, "response_schema", None)
        nome = getattr(sch, "__name__", "")
        if nome == "CurriculoSchema":
            d = {"nome": "Wesley Rocha", "email": "wesley.rocha@example.com", "numero": "(19) 90000-1414", "cidade": "Sumaré - SP",
                 "setor": "PRODUÇÃO", "confianca": "ALTA", "anos_experiencia": 4, "resumo": "Auxiliar de produção."}
        elif nome == "AnaliseSchema":
            d = {"compatibilidade": "ALTA", "recomendado": True, "avaliacao_humana": False, "resumo_profissional": "r",
                 "experiencias_relevantes": [], "competencias": [], "requisitos_atendidos": [
                     {"requisito": "Experiência em linha de produção", "evidencia": "Abastecimento e operação de linha de produção"}],
                 "a_verificar": [], "incompatibilidades": [], "justificativa": "j", "pensamentos": ["Linha de produção confere."],
                 "pedir_ajuda": False, "ajuda_especialidade": "", "ajuda_pergunta": ""}
        else:
            d = {"ok": True}
        return _Resp("OK" if sch is None else json.dumps(d, ensure_ascii=False), _Uso(1200, 300, 40, 200))

    def generate_content_stream(self, model, contents, config):
        r = self.generate_content(model, contents, config)
        yield _Resp("Olá, ", None)
        yield _Resp("tudo certo.", r.usage_metadata)


@pytest.fixture
def gemini(monkeypatch):
    falso = GeminiFalso()
    monkeypatch.setattr(config, "IA_MODO", "gemini")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "chave-de-teste")
    monkeypatch.setattr(ia, "_cliente", falso)
    ia._nivel_aceito.clear()
    uso_ia.zerar()
    yield falso
    uso_ia.zerar()
    ia._nivel_aceito.clear()


def test_chamadas_economicas_e_contabilizadas(cli, gemini):
    r = ia.testar()
    assert r["ok"] and r["pensamento"] == "minimal"
    c = gemini.chamadas[-1]
    assert c["modelo"] == "gemini-3.5-flash-lite" and c["nivel"] == "MINIMAL" and c["max"] == 32
    # triagem de PDF/texto vai como texto (mais barato) e com o prompt fixo como instrução de sistema
    entrada = triagem.preparar_entrada_gemini(b"%PDF", "cv.pdf", "application/pdf", "x" * 400)
    assert entrada == "x" * 400
    assert triagem.preparar_entrada_gemini(b"%PDF", "cv.pdf", "application/pdf", "pouco")["mime_type"] == "application/pdf"
    res = triagem.triar(b"Wesley", "cv.txt", "text/plain", "Wesley Rocha\nAuxiliar de produção")
    assert res["setor"] == "PRODUÇÃO"
    c = gemini.chamadas[-1]
    assert "Recrutador Sênior da empresa" in c["sistema"] and c["conteudos"][0].startswith("CURRÍCULO:")
    uso = cli.get("/api/ia/uso").json()
    assert uso["hoje"]["chamadas"] >= 2 and uso["hoje"]["custo_usd"] > 0
    assert {f["finalidade"] for f in uso["por_finalidade_hoje"]} >= {"teste", "triagem"}
    # 1200 entrada (200 do cache a 10%) + 340 saída no preço do 3.5 Flash-Lite
    esperado = ((1000 + 200 * 0.1) * 0.30 + 340 * 2.50) / 1e6
    with sessao() as db:
        assert abs(db.scalars(select(UsoIA).order_by(UsoIA.id.desc())).first().custo_usd - esperado) < 1e-9


def test_modelo_que_recusa_pensamento_minimo_usa_o_proximo_nivel(cli, gemini):
    gemini.niveis_recusados = {"minimal"}
    assert ia.testar()["ok"]
    assert [c["nivel"] for c in gemini.chamadas] == ["MINIMAL", "LOW"]
    assert ia._nivel_aceito["gemini-3.5-flash-lite"] == "low"
    ia.testar()
    assert gemini.chamadas[-1]["nivel"] == "LOW"  # lembra o ajuste aceito


def test_sem_creditos_pausa_a_fila_e_avisa_uma_vez(cli, gemini):
    gemini.erro = errors.ClientError(402, {"error": {"code": 402, "status": "RESOURCE_EXHAUSTED",
                                                     "message": "Your prepayment credits are depleted."}})
    inicio = time.monotonic()
    r = cli.post("/triagem", files={"file": ("wesley.txt", b"Wesley Rocha\nAuxiliar de producao\nwesley@example.com", "text/plain")},
                 headers={"X-API-Key": "chave-teste"})
    cid = r.json()["curriculo_id"]
    assert r.json()["status"] == "na_fila" and time.monotonic() - inicio < 30  # o fluxo de e-mail não fica pendurado
    r2 = cli.post("/triagem", files={"file": ("w2.txt", b"Outro\nAuxiliar", "text/plain")}, headers={"X-API-Key": "chave-teste"})
    assert "IA voltar" in r2.json()["mensagem"]
    pend = esperar(lambda: [p for p in cli.get("/api/quadro").json()["pendencias"] if p["tipo"] == "IA_INDISPONIVEL"],
                   msg="o Atlas não avisou que a IA parou")
    assert len(pend) == 1 and "sem créditos" in pend[0]["descricao"] and "ai.studio" in pend[0]["descricao"]
    assert not [p for p in cli.get("/api/quadro").json()["pendencias"] if p["tipo"] == "ERRO_IA"]  # nada de erro por currículo
    with sessao() as db:
        t = db.scalar(select(Tarefa).where(Tarefa.curriculo_id == cid, Tarefa.tipo == "TRIAGEM"))
        assert t.status == "pendente" and t.tentativas == 0  # espera na fila, sem gastar tentativas
    n = len(gemini.chamadas)
    time.sleep(1.2)
    assert len(gemini.chamadas) == n  # pausada: não fica martelando o Google
    assert "sem créditos" in cli.get("/api/ia/uso").json()["pausa"]
    resp = cli.post(f'/api/agentes/{cli.get("/api/escritorio").json()["agentes"][0]["id"]}/chat/stream', headers=H,
                    json={"mensagem": "oi"})
    assert "pausada" in resp.text

    # créditos recarregados: o teste de conexão encerra a pausa e a fila anda sozinha
    gemini.erro = None
    assert cli.post("/api/ia/testar", headers=H).json()["ok"]
    esperar(lambda: not [p for p in cli.get("/api/quadro").json()["pendencias"] if p["tipo"] == "IA_INDISPONIVEL"],
            msg="a pendência da IA não foi resolvida")
    esperar(lambda: cli.get("/api/escritorio").json()["quadro"]["curriculos_triados"] >= 1, msg="a triagem não retomou")


def test_limite_diario_de_gasto(cli, gemini, monkeypatch):
    monkeypatch.setattr(config, "IA_LIMITE_DIARIO_USD", 0.01)
    with sessao() as db:
        db.add(UsoIA(finalidade="analise", modelo="gemini-3.5-flash-lite", entrada=1, saida=1, custo_usd=0.02))
    uso_ia.zerar()
    motivo = uso_ia.bloqueio()
    assert motivo and "limite diário" in motivo
    with pytest.raises(ia.IAIndisponivel):
        ia._gerar("analise", "s", ["c"])
    assert ia.testar()["ok"]  # o teste de conexão ignora o limite
    with sessao() as db:
        for u in db.scalars(select(UsoIA)).all():
            db.delete(u)
    uso_ia.zerar()
    assert uso_ia.bloqueio() is None


def test_chave_invalida_e_modelo_indisponivel_tem_mensagens_claras():
    chave = errors.ClientError(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid."}})
    modelo = errors.ClientError(404, {"error": {"code": 404, "status": "NOT_FOUND",
                                                "message": "This model is no longer available to new users."}})
    assert "GEMINI_API_KEY" in uso_ia.classificar(chave)
    assert "GEMINI_MODELO" in uso_ia.classificar(modelo)
    assert uso_ia.classificar(errors.ServerError(503, {"error": {"code": 503, "message": "overloaded"}})) is None
