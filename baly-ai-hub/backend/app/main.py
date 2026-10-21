"""Baly AI Hub — API do escritório de recrutamento multiagente."""
import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from . import config, eventos, servicos as S
from .catalogo import CATEGORIAS_DISPONIVEIS
from .db import Base, engine, obter_db, sessao
from .models import (Agente, Auditoria, Candidato, Candidatura, Curriculo, Decisao, Especialidade, Pendencia,
                     SolicitacaoTitular, Usuario, Vaga, agora, iso)
from .seguranca import (COOKIE, auditar, conferir_chave_triagem, conferir_senha, criar_token, exigir, gerar_hash,
                        ler_arquivo, limitar_login, usuario_atual)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("baly")
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


async def rotinas():
    """Retenção (LGPD), leitura de e-mail opcional e alimentação da demonstração."""
    from . import email_imap
    ultimo_imap = ultimo_ret = 0.0
    demo = sorted((Path(__file__).resolve().parent.parent / "demo").glob("*.txt")) if config.SEED_DEMO else []
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
                arq = demo.pop(0)

                def _registrar(a=arq):
                    with sessao() as db:
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


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    Base.metadata.create_all(engine)
    eventos.configurar_loop(asyncio.get_running_loop())
    with sessao() as db:
        S.semear_basico(db)
        if config.SEED_DEMO:
            if not config.ADMIN_SENHA and not db.scalar(select(Usuario)):
                db.add(Usuario(nome="Recrutador Demo", email="admin@baly.local", senha_hash=gerar_hash("demo1234"), papel="admin"))
                log.warning("SEED_DEMO: usuário de demonstração admin@baly.local / demo1234 — NÃO use em produção")
            _semear_demo(db)
    S.recuperar_fila()
    if not config.TRIAGEM_API_KEY:
        log.warning("TRIAGEM_API_KEY não definida: POST /triagem aceita chamadas sem chave. Defina em produção.")
    log.info("IA em modo %s", config.IA_MODO)
    tarefas = [asyncio.create_task(trabalhador()), asyncio.create_task(rotinas())]
    yield
    for t in tarefas:
        t.cancel()


app = FastAPI(title="Baly AI Hub", lifespan=ciclo_de_vida, docs_url="/api/docs", redoc_url=None)
if config.ORIGENS_CORS:
    app.add_middleware(CORSMiddleware, allow_origins=config.ORIGENS_CORS, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])


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
    return {"id": u.id, "nome": u.nome, "email": u.email, "papel": u.papel}


@app.post("/auth/login")
def login(dados: Login, request: Request, response: Response, db: Session = Depends(obter_db)):
    limitar_login(request.client.host if request.client else "?")
    u = db.scalar(select(Usuario).where(Usuario.email == dados.email.strip().lower()))
    if not u or not u.ativo or not conferir_senha(dados.senha, u.senha_hash):
        raise HTTPException(401, "E-mail ou senha incorretos.")
    response.set_cookie(COOKIE, criar_token(u), httponly=True, samesite="strict", secure=config.COOKIE_SECURE,
                        max_age=config.SESSAO_HORAS * 3600, path="/")
    auditar(db, u, "login", "usuario", u.id, request)
    return _usuario_json(u)


@app.post("/auth/sair")
def sair(response: Response):
    response.delete_cookie(COOKIE, path="/")
    return {"ok": True}


@app.get("/auth/eu")
def eu(u: Usuario = Depends(usuario_atual)):
    return _usuario_json(u)


@app.get("/api/saude")
def saude():
    return {"ok": True, "ia_modo": config.IA_MODO}


# =========================================================================== entrada de currículos
@app.post("/triagem", dependencies=[Depends(conferir_chave_triagem)])
async def triagem_compativel(file: UploadFile = File(...)):
    """Mesmo contrato da API de triagem original: o fluxo de e-mail continua funcionando.

    Agora o currículo também é registrado no sistema, distribuído pelo Atlas e analisado.
    """
    conteudo = await file.read()

    def _registrar():
        with sessao() as db:
            cur, novo = S.registrar_curriculo(db, conteudo, file.filename or "arquivo", file.content_type, "email")
            return cur.id

    try:
        cid = await asyncio.to_thread(_registrar)
    except S.ErroNegocio as e:
        return {"status": "erro", "mensagem": str(e)}
    for _ in range(240):  # aguarda a triagem (até ~2 min), como a API original respondia
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

        def _reg(c=conteudo, f=f):
            with sessao() as db:
                cur, novo = S.registrar_curriculo(db, c, f.filename or "arquivo", f.content_type, "upload")
                auditar(db, u, "enviar_curriculo", "curriculo", cur.id, request)
                return {"arquivo": f.filename, "curriculo_id": cur.id, "novo": novo}
        try:
            saida.append(await asyncio.to_thread(_reg))
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


@app.post("/api/agentes/{ag_id}/chat")
async def chat(ag_id: int, m: Mensagem, u: Usuario = Depends(usuario_atual)):
    def _ctx():
        with sessao() as db:
            return S.contexto_chat(db, _agente(db, ag_id))
    ctx = await asyncio.to_thread(_ctx)
    try:
        resp = await asyncio.to_thread(S.ia.responder_chat, ctx, m.mensagem, m.historico[-6:])
    except Exception as e:
        log.warning("Chat falhou: %s", type(e).__name__)
        resp = "Não consegui consultar a IA agora. Tente de novo em instantes."
    return {"resposta": resp, "modo": config.IA_MODO}  # o conteúdo da conversa não é armazenado


class NovoAgente(BaseModel):
    especialidade_id: int
    nome: str | None = None


@app.post("/api/agentes")
def novo_agente(d: NovoAgente, request: Request, u: Usuario = Depends(exigir("supervisor")), db: Session = Depends(obter_db)):
    esp = db.get(Especialidade, d.especialidade_id)
    if not esp:
        raise HTTPException(404, "Especialidade não encontrada.")
    ag = S.ativar_agente(db, esp, por="O recrutador", nome=(d.nome or "").strip() or None)
    auditar(db, u, "ativar_agente", "agente", ag.id, request)
    return S.agente_json(db, ag)


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
    return Response(ler_arquivo(cur.caminho), media_type=cur.mime,
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
    setores: list[str] = Field(min_length=1)


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
def resolver(pid: int, d: Resolucao, request: Request, u: Usuario = Depends(exigir("supervisor")), db: Session = Depends(obter_db)):
    p = db.get(Pendencia, pid)
    if not p:
        raise HTTPException(404, "Pendência não encontrada.")
    S.resolver_pendencia(db, p, d.acao, u.id, d.setor)
    auditar(db, u, "resolver_pendencia", "pendencia", pid, request, tipo_acao=d.acao)
    return {"ok": True}


# =========================================================================== administração e LGPD
class NovoUsuario(BaseModel):
    nome: str
    email: str
    senha: str = Field(min_length=10)
    papel: str = "recrutador"


@app.get("/api/usuarios")
def usuarios(u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    return [_usuario_json(x) | {"ativo": x.ativo} for x in db.scalars(select(Usuario).order_by(Usuario.id)).all()]


@app.post("/api/usuarios")
def novo_usuario(d: NovoUsuario, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db)):
    if d.papel not in ("recrutador", "supervisor", "admin"):
        raise HTTPException(400, "Papel inválido.")
    if db.scalar(select(Usuario).where(Usuario.email == d.email.strip().lower())):
        raise HTTPException(400, "E-mail já cadastrado.")
    novo = Usuario(nome=d.nome, email=d.email.strip().lower(), senha_hash=gerar_hash(d.senha), papel=d.papel)
    db.add(novo)
    db.flush()
    auditar(db, u, "criar_usuario", "usuario", novo.id, request, papel=d.papel)
    return _usuario_json(novo)


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
