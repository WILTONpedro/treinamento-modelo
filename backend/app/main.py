"""API do escritório de recrutamento multiagente."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from . import config, eventos, servicos as S, uso_ia
from .catalogo import CATEGORIAS_DISPONIVEIS
from .db import Base, engine, migrar, obter_db, sessao
from .models import (Agente, Auditoria, Candidato, Candidatura, Curriculo, Decisao, Especialidade, Pendencia,
                     SolicitacaoTitular, Usuario, Vaga, agora, iso)
from .arquivos import ler_arquivo
from .lideranca import Lideranca, exclusivo
from .seguranca import (COOKIE, PAPEIS, auditar, conferir_chave_triagem, conferir_senha, criar_token, exigir, gerar_hash,
                        limitar_login, pedido_local, senha_temporaria, usuario_atual, validar_senha)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hub")
FRONT = Path(__file__).resolve().parent.parent.parent / "frontend"


# =========================================================================== trabalhador em segundo plano
async def trabalhador():
    """Executa a fila: uma tarefa por vez para a recepção, para o Atlas e para cada analista."""
    ativos: dict[str, asyncio.Task] = {}
    while True:
        try:
            for raia in [r for r, t in ativos.items() if t.done()]:
                ativos.pop(raia)
            for raia, tid, tipo in await asyncio.to_thread(S.reservar_tarefas, set(ativos)):
                ativos[raia] = asyncio.create_task(asyncio.to_thread(S.executar, tid, tipo))
        except Exception:
            log.exception("Falha no ciclo do trabalhador")
        await asyncio.sleep(0.5)


async def supervisao():
    """O Atlas acompanha o time sozinho: contrata, reforça, tenta de novo o que falhou e manda o boletim."""
    ultimo_reproc = ultimo_boletim = ultima_limpeza = 0.0
    while True:
        t = asyncio.get_running_loop().time()
        try:
            await asyncio.to_thread(S.supervisionar)
            if t - ultimo_reproc > 60:
                ultimo_reproc = t
                await asyncio.to_thread(S.reprocessar_automatico)
            if t - ultimo_boletim > 60:
                ultimo_boletim = t
                await asyncio.to_thread(S.boletim)
            if t - ultima_limpeza > 600:
                ultima_limpeza = t
                await asyncio.to_thread(S.limpar_conversas)
        except Exception:
            log.exception("Falha na supervisão")
        await asyncio.sleep(max(0.2, config.SUPERVISAO_SEGUNDOS))


async def comandar():
    """A equipe de agentes (fila, supervisão do Atlas, e-mail e retenção) roda só na cópia que tem o comando
    do escritório; com vários computadores no mesmo banco, as outras assumem se ela sair (veja lideranca.py)."""
    vez = Lideranca(engine)
    equipe: list[asyncio.Task] = []
    avisado = False
    try:
        while True:
            lider = await asyncio.to_thread(vez.tentar)
            if lider and not equipe:
                if vez.pg:
                    log.info("Esta cópia passou a comandar a equipe de agentes")
                await asyncio.to_thread(S.recuperar_fila)  # o que a cópia anterior deixou pela metade volta à fila
                equipe = [asyncio.create_task(trabalhador()), asyncio.create_task(rotinas()),
                          asyncio.create_task(supervisao())]
            elif not lider:
                for t in equipe:
                    t.cancel()
                equipe = []
                if not avisado:
                    log.info("Outro computador comanda a equipe de agentes; este mostra o escritório e assume se ele sair")
                    avisado = True
            await asyncio.sleep(5)
    finally:
        for t in equipe:
            t.cancel()
        vez.soltar()


def _lotes_demo() -> list[list[Path]]:
    """Currículos fictícios em ordem; arquivos com "_lote_" chegam juntos (como um lote de e-mails)."""
    lotes: list[list[Path]] = []
    for arq in sorted((Path(__file__).resolve().parent.parent / "demo").glob("*.txt")):
        if "_lote_" in arq.name and lotes and "_lote_" in lotes[-1][0].name:
            lotes[-1].append(arq)
        else:
            lotes.append([arq])
    return lotes


async def rotinas():
    """Retenção (LGPD), leitura de e-mail opcional e alimentação da demonstração."""
    from . import email_imap
    ultimo_imap = ultimo_ret = 0.0
    demo = _lotes_demo() if config.SEED_DEMO else []
    proximo_demo = 2.0
    while True:
        t = asyncio.get_running_loop().time()
        try:
            if t - ultimo_ret > 6 * 3600:
                ultimo_ret = t
                n = await asyncio.to_thread(S.aplicar_retencao)
                if n:
                    log.info("Retenção: %d titular(es) anonimizado(s)", n)
            if config.IMAP_ATIVO and t - ultimo_imap > config.IMAP_INTERVALO:
                ultimo_imap = t
                await asyncio.to_thread(email_imap.ler_caixa)
            if demo and t > proximo_demo:
                proximo_demo = t + config.DEMO_INTERVALO
                lote = demo.pop(0)

                def _registrar(arqs=lote):
                    with sessao() as db:
                        for a in arqs:
                            S.registrar_curriculo(db, a.read_bytes(), a.name, "text/plain", "demo")
                await asyncio.to_thread(_registrar)
        except Exception:
            log.exception("Falha nas rotinas")
        await asyncio.sleep(1)


def _semear_demo(db):
    if db.scalar(select(func.count()).select_from(Vaga)):
        return
    for v in [
        {"titulo": "Operador de Produção", "area": "Produção", "local": "Campinas - SP", "setores": ["PRODUÇÃO", "LOGÍSTICA"],
         "descricao": "Operação de linha de envase e embalagem em turnos.",
         "requisitos": ["Ensino fundamental completo", "Experiência em linha de produção", "Disponibilidade para turnos"]},
        {"titulo": "Assistente Administrativo", "area": "Administrativo", "local": "Campinas - SP", "setores": ["ADMINISTRATIVO"],
         "descricao": "Rotinas administrativas, controle de documentos e atendimento interno.",
         "requisitos": ["Ensino médio completo", "Excel intermediário", "Experiência em rotinas administrativas", "Controle de documentos"]},
        {"titulo": "Vendedor Externo", "area": "Comercial", "local": "Interior - SP", "setores": ["VENDAS", "COMERCIAL", "PROMOTOR DE VENDAS"],
         "descricao": "Visitas a clientes do varejo e prospecção na região.",
         "requisitos": ["CNH categoria B", "Experiência em vendas externas", "Ensino médio completo"]},
    ]:
        S.criar_vaga(db, v)


def verificar_producao():
    """Com AMBIENTE=producao, o sistema não sobe com configuração insegura."""
    if config.AMBIENTE != "producao":
        return
    problemas = config.problemas_producao()
    if problemas:
        for p in problemas:
            log.error("Configuração de produção: %s", p)
        raise SystemExit("Configuração insegura para produção: " + "; ".join(problemas))


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    verificar_producao()
    with exclusivo(engine, "inicio"):  # vários computadores abrindo juntos: um prepara o banco, os outros esperam
        Base.metadata.create_all(engine)
        migrar()
        with sessao() as db:
            S.semear_basico(db)
            if not db.scalar(select(func.count()).select_from(Usuario)) and not config.SEED_DEMO and not config.LOCAL:
                log.warning("Nenhum usuário cadastrado: defina ADMIN_EMAIL e ADMIN_SENHA para criar o primeiro administrador.")
            if config.SEED_DEMO:
                if not config.ADMIN_SENHA and not db.scalar(select(Usuario)):
                    db.add(Usuario(nome="Recrutador Demo", email="admin@demo.local", senha_hash=gerar_hash("demo1234"), papel="admin"))
                    log.warning("SEED_DEMO: usuário de demonstração admin@demo.local / demo1234 — NÃO use em produção")
                _semear_demo(db)
    eventos.configurar_loop(asyncio.get_running_loop())
    if not config.TRIAGEM_API_KEY and not config.LOCAL:
        log.warning("TRIAGEM_API_KEY não definida: POST /triagem aceita chamadas sem chave. Defina em produção.")
    log.info("IA em modo %s%s", config.IA_MODO, " · acesso local (só este computador, sem login)" if config.LOCAL else "")
    comando = asyncio.create_task(comandar())
    # PostgreSQL: os eventos de todos os computadores ligados ao banco chegam a este escritório
    escuta = eventos.iniciar_escuta() if engine.dialect.name == "postgresql" else None
    yield
    comando.cancel()
    if escuta:
        escuta.set()


app = FastAPI(title=config.APP_NOME or "Escritório de Recrutamento", lifespan=ciclo_de_vida, docs_url="/api/docs", redoc_url=None)
if config.ORIGENS_CORS:
    app.add_middleware(CORSMiddleware, allow_origins=config.ORIGENS_CORS, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])


if config.LOCAL:
    @app.middleware("http")
    async def so_este_computador(request: Request, call_next):
        """ACESSO=local: nada responde a outro computador, nem a página, nem a entrada de currículos."""
        if not pedido_local(request):
            return JSONResponse({"detail": "Este escritório só abre no próprio computador."}, status_code=403)
        return await call_next(request)


@app.middleware("http")
async def cabecalhos_seguranca(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if not request.url.path.startswith("/api/docs"):
        resp.headers.setdefault("Content-Security-Policy",
                                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data: blob:; frame-src 'self'; object-src 'self'; base-uri 'none'")
    if request.url.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


@app.exception_handler(S.ErroNegocio)
async def _erro_negocio(_, e: S.ErroNegocio):
    return JSONResponse({"detail": str(e)}, status_code=400)


# =========================================================================== autenticação
class Login(BaseModel):
    email: str
    senha: str


def _usuario_json(u: Usuario):
    return {"id": u.id, "nome": u.nome, "email": u.email, "papel": u.papel, "trocar_senha": bool(u.trocar_senha)}


def _gravar_sessao(response: Response, u: Usuario):
    response.set_cookie(COOKIE, criar_token(u), httponly=True, samesite="strict", secure=config.COOKIE_SECURE,
                        max_age=config.SESSAO_HORAS * 3600, path="/")


@app.post("/auth/login")
def login(dados: Login, request: Request, response: Response, db: Session = Depends(obter_db)):
    if config.LOCAL:
        raise HTTPException(400, "Neste computador o escritório abre sem login.")
    limitar_login(request.client.host if request.client else "?")
    u = db.scalar(select(Usuario).where(Usuario.email == dados.email.strip().lower()))
    if not u or not u.ativo or not conferir_senha(dados.senha, u.senha_hash):
        raise HTTPException(401, "E-mail ou senha incorretos.")
    u.ultimo_acesso_em = agora()
    _gravar_sessao(response, u)
    auditar(db, u, "login", "usuario", u.id, request)
    return _usuario_json(u)


class TrocaSenha(BaseModel):
    senha_atual: str
    nova_senha: str = Field(max_length=200)


@app.post("/auth/senha")
def trocar_senha(d: TrocaSenha, request: Request, response: Response, u: Usuario = Depends(usuario_atual),
                 db: Session = Depends(obter_db)):
    if config.LOCAL:
        raise HTTPException(400, "Neste computador o escritório abre sem senha.")
    limitar_login(request.client.host if request.client else "?")
    if not conferir_senha(d.senha_atual, u.senha_hash):
        raise HTTPException(400, "Senha atual incorreta.")
    erro = validar_senha(d.nova_senha)
    if erro:
        raise HTTPException(400, erro)
    if conferir_senha(d.nova_senha, u.senha_hash):
        raise HTTPException(400, "A nova senha precisa ser diferente da atual.")
    u.senha_hash, u.trocar_senha = gerar_hash(d.nova_senha), False
    u.versao_sessao = (u.versao_sessao or 0) + 1  # encerra as outras sessões abertas com a senha antiga
    _gravar_sessao(response, u)
    auditar(db, u, "trocar_senha", "usuario", u.id, request)
    return _usuario_json(u)


@app.post("/auth/sair")
def sair(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/auth/eu")
def eu(u: Usuario = Depends(usuario_atual)):
    return _usuario_json(u)


@app.get("/api/saude")
def saude(db: Session = Depends(obter_db)):
    db.execute(select(1))  # banco respondendo
    return {"ok": True, "ia_modo": config.IA_MODO, "ia_pausada": bool(uso_ia.bloqueio())}


@app.get("/api/marca")
def marca():
    """Identidade configurável (APP_NOME etc.) — pública, para a tela de login."""
    return {"nome": config.APP_NOME, "subtitulo": config.APP_SUBTITULO, "empresa": config.EMPRESA_NOME,
            "local": config.LOCAL}


# =========================================================================== entrada de currículos
@app.post("/triagem", dependencies=[Depends(conferir_chave_triagem)])
async def triagem_compativel(file: UploadFile = File(...)):
    """Mesmo contrato da API de triagem original: o fluxo de e-mail continua funcionando.

    Agora o currículo também é registrado no sistema, distribuído pelo Atlas e analisado.
    """
    conteudo = await file.read()

    try:
        cid, _, _ = await asyncio.to_thread(S.registrar_com_seguranca, conteudo, file.filename or "arquivo", file.content_type, "email")
    except S.ErroNegocio as e:
        return {"status": "erro", "mensagem": str(e)}
    for _ in range(240):  # aguarda a triagem (até ~2 min), como a API original respondia
        pausa = uso_ia.bloqueio()
        if pausa:  # IA pausada: responde já (o fluxo de e-mail não fica pendurado); a triagem sai quando ela voltar
            return {"arquivo": file.filename, "status": "na_fila", "curriculo_id": cid,
                    "mensagem": f"Registrado; a triagem acontece quando a IA voltar ({pausa})."}
        with sessao() as db:
            cur = db.get(Curriculo, cid)
            if cur.status != "recebido":
                analise = cur.triagem_json or {}
                nome = analise.get("nome") or "Candidato"
                return {"arquivo": file.filename, "nome_identificado": nome if len(nome) >= 3 else "Candidato",
                        "setor_sugerido": analise.get("setor", "OUTROS"), "confianca": analise.get("confianca", "BAIXA"),
                        "detalhes": analise, "curriculo_id": cid}
        await asyncio.sleep(0.5)
    return {"arquivo": file.filename, "status": "na_fila", "curriculo_id": cid,
            "mensagem": "Registrado; a triagem ainda está em andamento."}


@app.post("/api/curriculos")
async def enviar_curriculos(request: Request, arquivos: list[UploadFile] = File(...),
                            u: Usuario = Depends(exigir("supervisor"))):
    saida = []
    for f in arquivos:
        conteudo = await f.read()

        def _auditar(db, cur, novo):
            auditar(db, u, "enviar_curriculo", "curriculo", cur.id, request)
        try:
            cid, novo, _ = await asyncio.to_thread(S.registrar_com_seguranca, conteudo, f.filename or "arquivo",
                                                   f.content_type, "upload", _auditar)
            saida.append({"arquivo": f.filename, "curriculo_id": cid, "novo": novo})
        except S.ErroNegocio as e:
            saida.append({"arquivo": f.filename, "erro": str(e)})
    return saida


# =========================================================================== escritório
@app.get("/api/escritorio")
def escritorio(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    return S.escritorio(db) | {"usuario": _usuario_json(u)}


@app.get("/api/eventos/stream")
async def stream(request: Request, u: Usuario = Depends(usuario_atual)):
    fila = eventos.novo_ouvinte()

    async def gerar():
        try:
            yield "retry: 3000\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    dado = await asyncio.wait_for(fila.get(), timeout=15)
                    yield f"data: {dado}\n\n"
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            eventos.remover_ouvinte(fila)

    return StreamingResponse(gerar(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


def _agente(db, ag_id) -> Agente:
    ag = db.get(Agente, ag_id)
    if not ag:
        raise HTTPException(404, "Agente não encontrado.")
    return ag


@app.get("/api/agentes/{ag_id}")
def agente(ag_id: int, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    return S.detalhe_agente(db, _agente(db, ag_id))


@app.get("/api/agentes/{ag_id}/candidatos")
def candidatos(ag_id: int, request: Request, vaga_id: int | None = None,
               u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    _agente(db, ag_id)
    auditar(db, u, "listar_candidatos", "agente", ag_id, request, vaga_id=vaga_id)
    return S.candidatos_do_agente(db, ag_id, vaga_id)


class Mensagem(BaseModel):
    mensagem: str = Field(min_length=1, max_length=1000)
    historico: list[dict] = []


def _ctx_chat(ag_id):
    with sessao() as db:
        return S.contexto_chat(db, _agente(db, ag_id))


@app.post("/api/agentes/{ag_id}/chat")
async def chat(ag_id: int, m: Mensagem, u: Usuario = Depends(usuario_atual)):
    ctx = await asyncio.to_thread(_ctx_chat, ag_id)
    try:
        resp = await asyncio.to_thread(S.ia.responder_chat, ctx, m.mensagem, m.historico[-6:])
    except Exception as e:
        log.warning("Chat falhou: %s", type(e).__name__)
        resp = "Não consegui consultar a IA agora. Tente de novo em instantes."
    return {"resposta": resp, "modo": config.IA_MODO}  # o conteúdo da conversa não é armazenado


@app.post("/api/agentes/{ag_id}/chat/stream")
async def chat_stream(ag_id: int, m: Mensagem, u: Usuario = Depends(usuario_atual)):
    """Mesmo chat, entregue aos pedaços: o agente aparece 'digitando' no painel."""
    ctx = await asyncio.to_thread(_ctx_chat, ag_id)

    def gerar():
        enviou = False
        try:
            for parte in S.ia.responder_chat_stream(ctx, m.mensagem, m.historico[-6:]):
                enviou = True
                yield parte
        except Exception as e:
            log.warning("Chat (stream) falhou: %s", type(e).__name__)
            yield ("\n" if enviou else "") + "Não consegui consultar a IA agora. Tente de novo em instantes."

    return StreamingResponse(gerar(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/equipe/conversas")
def conversas(limite: int = 60, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    """Canal da equipe: o que os agentes disseram uns aos outros (consultas, contratações, boletins)."""
    return S.conversas_equipe(db, min(max(limite, 1), 200))


@app.post("/api/equipe/chat")
async def chat_equipe(m: Mensagem, u: Usuario = Depends(usuario_atual)):
    """O recrutador fala com o time; quem tem a informação responde (o conteúdo não é armazenado)."""
    def _ctx():
        with sessao() as db:
            return S.contexto_equipe(db, por_agente=config.IA_MODO != "gemini")
    ctx = await asyncio.to_thread(_ctx)
    try:
        falas = await asyncio.to_thread(S.ia.conversa_equipe, ctx, m.mensagem, m.historico[-8:])
    except uso_ia.IAIndisponivel as e:
        atlas = next(a for a in ctx["equipe"] if a["funcao"] == "supervisor")
        falas = [{"agente_id": atlas["id"], "texto": f"A IA está pausada agora: {e}."}]
    except Exception as e:
        log.warning("Chat da equipe falhou: %s", type(e).__name__)
        atlas = next(a for a in ctx["equipe"] if a["funcao"] == "supervisor")
        falas = [{"agente_id": atlas["id"], "texto": "Não consegui consultar a IA agora. Tente de novo em instantes."}]
    nomes = {a["id"]: a["nome"] for a in ctx["equipe"]}
    return {"respostas": [f | {"nome": nomes.get(f["agente_id"], "")} for f in falas], "modo": config.IA_MODO}


class NovoAgente(BaseModel):
    especialidade_id: int
    nome: str | None = Field(None, max_length=40)


@app.post("/api/agentes")
def novo_agente(d: NovoAgente, request: Request, u: Usuario = Depends(exigir("supervisor"))):
    with S.trava_equipe, sessao() as db:  # estações de trabalho: uma contratação de cada vez
        esp = db.get(Especialidade, d.especialidade_id)
        if not esp:
            raise HTTPException(404, "Especialidade não encontrada.")
        ag = S.contratar(db, esp, "manual", por=u.nome, nome=(d.nome or "").strip() or None)
        auditar(db, u, "contratar_agente", "agente", ag.id, request)
        return S.agente_json(db, ag)


@app.post("/api/agentes/{ag_id}/dispensar")
def dispensar(ag_id: int, request: Request, u: Usuario = Depends(exigir("supervisor"))):
    with S.trava_equipe, sessao() as db:
        ag = _agente(db, ag_id)
        S.dispensar_agente(db, ag, por=u.nome)
        auditar(db, u, "dispensar_agente", "agente", ag.id, request)
        return {"ok": True}


@app.get("/api/ia/uso")
def ia_uso(u: Usuario = Depends(exigir("supervisor")), db: Session = Depends(obter_db)):
    """Tokens e custo estimado da IA (hoje e últimos 30 dias), por finalidade, e se ela está pausada."""
    hoje = uso_ia.inicio_do_dia()
    preco = uso_ia.preco(config.GEMINI_MODELO_AGENTES)
    return {"modo": config.IA_MODO, "modelo": config.GEMINI_MODELO_AGENTES, "modelo_triagem": config.GEMINI_MODELO_TRIAGEM,
            "pensamento": config.GEMINI_PENSAMENTO, "preco_por_milhao_usd": {"entrada": preco[0], "saida": preco[1]} if preco else None,
            "limite_diario_usd": config.IA_LIMITE_DIARIO_USD, "pausa": uso_ia.bloqueio() or uso_ia.pausa_do_provedor(),
            "hoje": uso_ia.resumo(db, hoje), "ultimos_30_dias": uso_ia.resumo(db, hoje - timedelta(days=30)),
            "por_finalidade_hoje": uso_ia.por_finalidade(db, hoje)}


@app.post("/api/ia/testar")
async def ia_testar(request: Request, u: Usuario = Depends(exigir("admin"))):
    """Uma chamada mínima ao modelo configurado: confirma chave, créditos e modelo (e encerra a pausa se der certo)."""
    r = await asyncio.to_thread(S.ia.testar)
    with sessao() as db:
        auditar(db, u, "testar_ia", "ia", None, request, ok=r.get("ok"))
    return r


@app.get("/api/contratacoes")
def contratacoes(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    return S.contratacoes_recentes(db, 30)


@app.get("/api/especialidades")
def especialidades(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    return [{"id": e.id, "codigo": e.codigo, "nome": e.nome, "descricao": e.descricao, "setores": e.setores}
            for e in db.scalars(select(Especialidade).order_by(Especialidade.id)).all()]


class NovaEspecialidade(BaseModel):
    codigo: str = Field(min_length=2, max_length=40)
    nome: str
    descricao: str = ""
    setores: list[str]
    nomes_sugeridos: list[str] = []


@app.post("/api/especialidades")
def nova_especialidade(d: NovaEspecialidade, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    invalidos = [s for s in d.setores if s not in CATEGORIAS_DISPONIVEIS]
    if invalidos:
        raise HTTPException(400, f"Pastas inexistentes: {', '.join(invalidos)}")
    e = Especialidade(codigo=d.codigo, nome=d.nome, descricao=d.descricao, setores=d.setores, nomes_sugeridos=d.nomes_sugeridos)
    db.add(e)
    db.flush()
    return {"id": e.id}


@app.get("/api/categorias")
def categorias(u: Usuario = Depends(usuario_atual)):
    return CATEGORIAS_DISPONIVEIS


# =========================================================================== candidaturas e currículos
def _candidatura(db, cid) -> Candidatura:
    c = db.get(Candidatura, cid)
    if not c or (c.candidato and c.candidato.anonimizado_em):
        raise HTTPException(404, "Candidatura não encontrada.")
    return c


@app.get("/api/candidaturas/{cid}")
def candidatura(cid: int, request: Request, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    c = _candidatura(db, cid)
    auditar(db, u, "ver_curriculo", "candidatura", cid, request)
    return S.detalhe_candidatura(db, c)


@app.get("/api/curriculos/{cur_id}/arquivo")
def arquivo(cur_id: int, request: Request, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    cur = db.get(Curriculo, cur_id)
    if not cur or not cur.caminho:
        raise HTTPException(404, "Arquivo indisponível.")
    auditar(db, u, "abrir_arquivo", "curriculo", cur_id, request)
    nome = "curriculo-%d%s" % (cur.id, Path(cur.arquivo_nome).suffix.lower())
    return Response(ler_arquivo(cur.caminho, db), media_type=cur.mime,
                    headers={"Content-Disposition": f'inline; filename="{nome}"'})


class NovaDecisao(BaseModel):
    decisao: str
    observacao: str = Field("", max_length=1000)


@app.post("/api/candidaturas/{cid}/decisao")
def decidir(cid: int, d: NovaDecisao, request: Request, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    if d.decisao not in ("entrevista", "banco_talentos", "descartado"):
        raise HTTPException(400, "Decisão inválida.")
    c = _candidatura(db, cid)
    db.add(Decisao(candidatura_id=cid, decisao=d.decisao, observacao=d.observacao, usuario_id=u.id))
    if c.candidato:
        c.candidato.ultima_atividade_em = agora()
    rot = {"entrevista": "chamar para entrevista", "banco_talentos": "manter no banco de talentos", "descartado": "não seguir"}[d.decisao]
    eventos.registrar(db, "decisao_humana", f"{u.nome} decidiu {rot}: {S.primeiro_nome(c.candidato.nome if c.candidato else '')} (“{c.vaga.titulo}”)",
                      None, candidatura_id=cid)
    auditar(db, u, "decidir", "candidatura", cid, request, decisao=d.decisao)
    return {"ok": True}


# =========================================================================== vagas, quadro e pendências
@app.get("/api/vagas")
def vagas(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    return [S.vaga_json(db, v, completo=True) for v in db.scalars(select(Vaga).order_by(Vaga.id)).all()]


class NovaVaga(BaseModel):
    titulo: str = Field(min_length=3, max_length=160)
    area: str = ""
    local: str = ""
    descricao: str = ""
    requisitos: list[str] = []
    setores: list[str] = []  # vazio: o Atlas deduz as pastas pelo título e pela descrição


class RascunhoVaga(BaseModel):
    titulo: str = Field(min_length=3, max_length=160)
    descricao: str = Field("", max_length=4000)


@app.post("/api/vagas/sugerir")
async def sugerir_vaga(d: RascunhoVaga, u: Usuario = Depends(exigir("supervisor"))):
    """O Atlas sugere as pastas da triagem (e, com Gemini, requisitos) para uma vaga nova."""
    try:
        return await asyncio.to_thread(S.ia.sugerir_setores, d.titulo, d.descricao)
    except uso_ia.IAIndisponivel as e:
        raise HTTPException(503, f"A IA está pausada: {e}")
    except Exception as e:
        log.warning("Sugestão de vaga falhou: %s", type(e).__name__)
        raise HTTPException(503, "Não consegui consultar a IA agora.")


@app.post("/api/vagas")
def criar_vaga(d: NovaVaga, request: Request, u: Usuario = Depends(exigir("supervisor")), db: Session = Depends(obter_db)):
    invalidos = [s for s in d.setores if s not in CATEGORIAS_DISPONIVEIS]
    if invalidos:
        raise HTTPException(400, f"Pastas inexistentes: {', '.join(invalidos)}")
    v = S.criar_vaga(db, d.model_dump())
    auditar(db, u, "criar_vaga", "vaga", v.id, request)
    return S.vaga_json(db, v, completo=True)


class StatusVaga(BaseModel):
    status: str


@app.patch("/api/vagas/{vid}")
def status_vaga(vid: int, d: StatusVaga, request: Request, u: Usuario = Depends(exigir("supervisor")), db: Session = Depends(obter_db)):
    v = db.get(Vaga, vid)
    if not v:
        raise HTTPException(404, "Vaga não encontrada.")
    S.mudar_status_vaga(db, v, d.status)
    auditar(db, u, "status_vaga", "vaga", vid, request, status=d.status)
    return S.vaga_json(db, v, completo=True)


@app.get("/api/quadro")
def quadro(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db)):
    pend = db.scalars(select(Pendencia).where(Pendencia.status == "aberta").order_by(Pendencia.id.desc())).all()
    nomes = {a.id: a.nome for a in db.scalars(select(Agente)).all()}
    return {"resumo": S.quadro(db),
            "equipe": [S.agente_json(db, a) for a in db.scalars(select(Agente).where(Agente.ativo.is_(True)).order_by(Agente.id)).all()],
            "pendencias": [{"id": p.id, "tipo": p.tipo, "descricao": p.descricao, "agente": nomes.get(p.agente_id),
                            "curriculo_id": p.curriculo_id, "candidatura_id": p.candidatura_id, "vaga_id": p.vaga_id,
                            "especialidade_id": p.especialidade_id, "criado_em": iso(p.criado_em)} for p in pend],
            "pode_gerir": u.papel in ("supervisor", "admin")}


class Resolucao(BaseModel):
    acao: str = "resolver"
    setor: str | None = None


@app.post("/api/pendencias/{pid}/resolver")
def resolver(pid: int, d: Resolucao, request: Request, u: Usuario = Depends(exigir("supervisor"))):
    with S.trava_equipe, sessao() as db:  # "ativar" contrata: mesma trava das contratações do Atlas
        p = db.get(Pendencia, pid)
        if not p:
            raise HTTPException(404, "Pendência não encontrada.")
        S.resolver_pendencia(db, p, d.acao, u.id, d.setor)
        auditar(db, u, "resolver_pendencia", "pendencia", pid, request, tipo_acao=d.acao)
        return {"ok": True}


# =========================================================================== administração e LGPD
class NovoUsuario(BaseModel):
    nome: str = Field(min_length=2, max_length=120)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$", max_length=200)
    senha: str | None = Field(None, max_length=200)  # vazia: o sistema gera uma senha temporária
    papel: str = "recrutador"


def _usuario_admin_json(x: Usuario):
    return _usuario_json(x) | {"ativo": x.ativo, "criado_em": iso(x.criado_em), "ultimo_acesso_em": iso(x.ultimo_acesso_em)}


@app.get("/api/usuarios")
def usuarios(u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    return [_usuario_admin_json(x) for x in db.scalars(select(Usuario).order_by(Usuario.id)).all()]


@app.post("/api/usuarios")
def novo_usuario(d: NovoUsuario, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    if d.papel not in PAPEIS:
        raise HTTPException(400, "Papel inválido.")
    if db.scalar(select(Usuario).where(Usuario.email == d.email.strip().lower())):
        raise HTTPException(400, "E-mail já cadastrado.")
    senha = d.senha or senha_temporaria()
    erro = validar_senha(senha)
    if erro:
        raise HTTPException(400, erro)
    novo = Usuario(nome=d.nome.strip(), email=d.email.strip().lower(), senha_hash=gerar_hash(senha), papel=d.papel,
                   trocar_senha=True)  # quem recebe a conta escolhe a própria senha no primeiro acesso
    db.add(novo)
    db.flush()
    auditar(db, u, "criar_usuario", "usuario", novo.id, request, papel=d.papel)
    return _usuario_admin_json(novo) | ({"senha_temporaria": senha} if not d.senha else {})


class EdicaoUsuario(BaseModel):
    nome: str | None = Field(None, min_length=2, max_length=120)
    papel: str | None = None
    ativo: bool | None = None


def _admins_ativos(db) -> int:
    return db.scalar(select(func.count()).select_from(Usuario).where(Usuario.papel == "admin", Usuario.ativo.is_(True))) or 0


@app.patch("/api/usuarios/{uid}")
def editar_usuario(uid: int, d: EdicaoUsuario, request: Request, u: Usuario = Depends(exigir("admin")),
                   db: Session = Depends(obter_db)):
    alvo = db.get(Usuario, uid)
    if not alvo:
        raise HTTPException(404, "Usuário não encontrado.")
    if d.papel is not None and d.papel not in PAPEIS:
        raise HTTPException(400, "Papel inválido.")
    if alvo.id == u.id and (d.ativo is False or (d.papel and d.papel != "admin")):
        raise HTTPException(400, "Você não pode desativar nem rebaixar a própria conta.")
    deixa_de_ser_admin = alvo.papel == "admin" and alvo.ativo and (d.ativo is False or (d.papel and d.papel != "admin"))
    if deixa_de_ser_admin and _admins_ativos(db) <= 1:
        raise HTTPException(400, "Mantenha pelo menos um administrador ativo.")
    mudou_acesso = (d.ativo is not None and d.ativo != alvo.ativo) or (d.papel is not None and d.papel != alvo.papel)
    if d.nome:
        alvo.nome = d.nome.strip()
    if d.papel:
        alvo.papel = d.papel
    if d.ativo is not None:
        alvo.ativo = d.ativo
    if mudou_acesso:
        alvo.versao_sessao = (alvo.versao_sessao or 0) + 1  # o novo acesso vale já: sessões antigas caem
    auditar(db, u, "editar_usuario", "usuario", alvo.id, request, papel=alvo.papel, ativo=alvo.ativo)
    return _usuario_admin_json(alvo)


@app.post("/api/usuarios/{uid}/senha")
def redefinir_senha(uid: int, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    alvo = db.get(Usuario, uid)
    if not alvo:
        raise HTTPException(404, "Usuário não encontrado.")
    temp = senha_temporaria()
    alvo.senha_hash, alvo.trocar_senha = gerar_hash(temp), True
    alvo.versao_sessao = (alvo.versao_sessao or 0) + 1
    auditar(db, u, "redefinir_senha", "usuario", alvo.id, request)
    return {"senha_temporaria": temp}


@app.get("/api/lgpd/titulares")
def buscar_titulares(busca: str, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    termo = f"%{busca.strip().lower()}%"
    achados = db.scalars(select(Candidato).where(Candidato.anonimizado_em.is_(None),
                                                 or_(func.lower(Candidato.email).like(termo), func.lower(Candidato.nome).like(termo))).limit(20)).all()
    auditar(db, u, "buscar_titular", "candidato", None, request)
    return [{"id": c.id, "nome": c.nome, "email": c.email, "cidade": c.cidade} for c in achados]


def _titular(db, cid) -> Candidato:
    c = db.get(Candidato, cid)
    if not c or c.anonimizado_em:
        raise HTTPException(404, "Titular não encontrado.")
    return c


@app.get("/api/lgpd/titulares/{cid}/exportar")
def exportar(cid: int, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    c = _titular(db, cid)
    db.add(SolicitacaoTitular(candidato_id=c.id, tipo="acesso", usuario_id=u.id, concluido_em=agora()))
    auditar(db, u, "exportar_titular", "candidato", cid, request)
    return Response(json.dumps(S.exportar_titular(db, c), ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="titular-{cid}.json"'})


class Eliminacao(BaseModel):
    observacao: str = ""


@app.post("/api/lgpd/titulares/{cid}/eliminar")
def eliminar(cid: int, d: Eliminacao, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    c = _titular(db, cid)
    S.anonimizar_titular(db, c, "pedido do titular")
    db.add(SolicitacaoTitular(candidato_id=c.id, tipo="eliminacao", observacao=d.observacao, usuario_id=u.id, concluido_em=agora()))
    auditar(db, u, "eliminar_titular", "candidato", cid, request)
    return {"ok": True}


@app.get("/api/auditoria")
def auditoria(limite: int = 200, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    nomes = {x.id: x.nome for x in db.scalars(select(Usuario)).all()}
    return [{"ts": iso(a.ts), "usuario": nomes.get(a.usuario_id), "acao": a.acao, "recurso": a.recurso,
             "recurso_id": a.recurso_id, "ip": a.ip, "detalhes": a.detalhes}
            for a in db.scalars(select(Auditoria).order_by(Auditoria.id.desc()).limit(min(limite, 1000))).all()]


# =========================================================================== front-end
if FRONT.exists():
    app.mount("/", StaticFiles(directory=FRONT, html=True), name="front")
