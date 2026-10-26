"""API do escritório de recrutamento multiagente."""
import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select, text
from sqlalchemy.exc import InterfaceError, InternalError, OperationalError
from sqlalchemy.orm import Session

from . import acervo, config, eventos, momentos, servicos as S, uso_ia
from .catalogo import CATEGORIAS_DISPONIVEIS
from .db import Base, engine, migrar, obter_db, problema_do_banco, sessao
from .models import (Agente, Auditoria, Candidato, Candidatura, Curriculo, Decisao, Especialidade, Pendencia,
                     SolicitacaoTitular, Usuario, Vaga, agora, iso)
from .arquivos import ler_arquivo
from .lideranca import Lideranca, esperar_vez
from .seguranca import (COOKIE, PAPEIS, auditar, conferir_chave_triagem, conferir_senha, criar_token, exigir, gerar_hash,
                        limitar_conta, limitar_login, login_ok, pedido_local, senha_temporaria, usuario_atual,
                        validar_senha)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hub")
FRONT = (config.RECURSOS if config.CONGELADO else config.BASE.parent) / "frontend"
ARQUIVOS_DA_TELA = ("index.html", "app.js", "cena.js", "estilo.css")


def _versao_da_tela() -> tuple[str, str]:
    """Impressão digital dos arquivos da tela: muda a cada atualização do programa. Vai no endereço de app.js,
    cena.js e estilo.css (o navegador nunca usa os antigos guardados) e no /api/marca e /api/escritorio: uma
    janela que ficou aberta durante a atualização percebe a diferença e se recarrega sozinha."""
    h, quando = hashlib.sha1(), 0.0
    for nome in ARQUIVOS_DA_TELA:
        arq = FRONT / nome
        if arq.exists():
            h.update(arq.read_bytes())
            quando = max(quando, arq.stat().st_mtime)
    return h.hexdigest()[:10], (datetime.fromtimestamp(quando).strftime("%d/%m/%Y %H:%M") if quando else "")


VERSAO, VERSAO_DATA = _versao_da_tela()


# =========================================================================== trabalhador em segundo plano
_avisado_em: dict[str, float] = {}


def _falha_no_ciclo(nome: str, e: Exception):
    """Sem banco (fora do ar ou acesso cortado): um aviso por minuto no registro, não um erro a cada volta."""
    if problema_do_banco(e):
        if time.monotonic() - _avisado_em.get(nome, 0.0) > 60:
            _avisado_em[nome] = time.monotonic()
            log.warning("%s: sem acesso ao banco (%s)", nome, str(getattr(e, "orig", e)).splitlines()[0][:160])
        return
    log.exception("Falha: %s", nome)


async def trabalhador():
    """Executa a fila: uma tarefa por vez para a recepção, para o Atlas e para cada analista."""
    ativos: dict[str, asyncio.Task] = {}
    while True:
        try:
            for raia in [r for r, t in ativos.items() if t.done()]:
                ativos.pop(raia)
            for raia, tid, tipo in await asyncio.to_thread(S.reservar_tarefas, set(ativos)):
                ativos[raia] = asyncio.create_task(asyncio.to_thread(S.executar, tid, tipo))
        except Exception as e:
            _falha_no_ciclo("fila de trabalho", e)
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
            await asyncio.to_thread(momentos.talvez_um_momento)
        except Exception as e:
            _falha_no_ciclo("supervisão", e)
        await asyncio.sleep(max(0.2, config.SUPERVISAO_SEGUNDOS))


async def comandar(vez: Lideranca):
    """A equipe de agentes (fila, supervisão do Atlas, e-mail e retenção) roda só na cópia que tem o comando
    do escritório; com vários computadores no mesmo banco, as outras assumem se ela sair (veja lideranca.py)."""
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
                avisado = False
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


def _lotes_demo() -> list[list[Path]]:
    """Currículos fictícios em ordem; arquivos com "_lote_" chegam juntos (como um lote de e-mails)."""
    lotes: list[list[Path]] = []
    for arq in sorted((config.RECURSOS / "demo").glob("*.txt")):
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
                await asyncio.to_thread(email_imap.ler_e_avisar)
            if demo and t > proximo_demo:
                proximo_demo = t + config.DEMO_INTERVALO
                lote = demo.pop(0)

                def _registrar(arqs=lote):
                    with sessao() as db:
                        for a in arqs:
                            S.registrar_curriculo(db, a.read_bytes(), a.name, "text/plain", "demo")
                await asyncio.to_thread(_registrar)
        except Exception as e:
            _falha_no_ciclo("rotinas (e-mail, retenção)", e)
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


# Banco pronto? Até ficar, as rotas de dados respondem 503 com o aviso para a tela (veja banco_pronto).
ESTADO_BANCO: dict = {"pronto": False, "problema": None}
_acordar: asyncio.Event | None = None  # a tela viu o banco voltar: tenta de novo na hora, sem esperar o intervalo


def _preparar_banco():
    with engine.begin() as con:  # vários computadores abrindo juntos: um prepara o banco, os outros esperam a vez
        esperar_vez(con, "tabelas")
        Base.metadata.create_all(con)
    migrar()
    with sessao() as db:
        esperar_vez(db, "equipe-inicial")  # a equipe inicial nasce uma vez só
        S.semear_basico(db)
        if not db.scalar(select(func.count()).select_from(Usuario)) and not config.SEED_DEMO and not config.LOCAL:
            log.warning("Nenhum usuário cadastrado: defina ADMIN_EMAIL e ADMIN_SENHA para criar o primeiro administrador.")
        if config.SEED_DEMO:
            if not config.ADMIN_SENHA and not db.scalar(select(Usuario)):
                db.add(Usuario(nome="Recrutador Demo", email="admin@demo.local", senha_hash=gerar_hash("demo1234"), papel="admin"))
                log.warning("SEED_DEMO: usuário de demonstração admin@demo.local / demo1234 — NÃO use em produção")
            _semear_demo(db)


async def _insistir_no_banco(comecar, espera: float = 5.0):
    """O banco não respondeu na abertura (computador ligou antes da internet, acesso suspenso...): tenta de novo
    com espera crescente (5 s até 1 min) ou na hora em que a tela perceber que ele voltou."""
    while True:
        try:
            await asyncio.wait_for(_acordar.wait(), timeout=espera)
        except asyncio.TimeoutError:
            pass
        _acordar.clear()
        try:
            await asyncio.to_thread(_preparar_banco)
        except Exception as e:
            p = problema_do_banco(e)
            if not p:
                log.exception("Falha ao preparar o banco")
            ESTADO_BANCO["problema"] = p or ESTADO_BANCO["problema"]
            espera = min(60.0, espera * 2)
            continue
        log.info("Banco de dados disponível: o escritório volta ao normal")
        comecar()
        return


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    global _acordar
    verificar_producao()
    eventos.configurar_loop(asyncio.get_running_loop())
    _acordar = asyncio.Event()
    if not config.TRIAGEM_API_KEY and not config.LOCAL:
        log.warning("TRIAGEM_API_KEY não definida: POST /triagem aceita chamadas sem chave. Defina em produção.")
    log.info("IA em modo %s%s", config.IA_MODO, " · acesso local (só este computador, sem login)" if config.LOCAL else "")
    vez = Lideranca(engine)
    tarefas: dict = {}

    def comecar():
        ESTADO_BANCO.update(pronto=True, problema=None)
        tarefas["comando"] = asyncio.create_task(comandar(vez))
        # PostgreSQL: os eventos gravados por qualquer computador ligado ao banco chegam a este escritório
        if engine.dialect.name == "postgresql":
            tarefas["acompanhamento"] = eventos.acompanhar_banco()

    try:
        _preparar_banco()
        comecar()
    except Exception as e:
        p = problema_do_banco(e)
        if not p:
            raise
        ESTADO_BANCO.update(pronto=False, problema=p)
        log.warning("Sem acesso ao banco ao abrir (%s): o escritório abre, avisa e tenta de novo sozinho",
                    str(getattr(e, "orig", e)).splitlines()[0][:160])
        tarefas["insistir"] = asyncio.create_task(_insistir_no_banco(comecar))
    yield
    for nome in ("insistir", "comando"):
        if tarefas.get(nome):
            tarefas[nome].cancel()
    if tarefas.get("acompanhamento"):
        tarefas["acompanhamento"].set()
    await asyncio.to_thread(vez.soltar)  # fechando: outro computador assume a equipe na hora


# Em produção, a documentação interativa (OpenAPI/Swagger) fica desligada: ela lista todos os endpoints e
# formatos para qualquer um, sem login. Em desenvolvimento continua em /api/docs.
_EM_PRODUCAO = config.AMBIENTE == "producao"
app = FastAPI(title=config.APP_NOME or "Escritório de Recrutamento", lifespan=ciclo_de_vida,
              docs_url=None if _EM_PRODUCAO else "/api/docs", redoc_url=None,
              openapi_url=None if _EM_PRODUCAO else "/openapi.json")
if config.ORIGENS_CORS and "*" not in config.ORIGENS_CORS:
    app.add_middleware(CORSMiddleware, allow_origins=config.ORIGENS_CORS, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])
elif config.ORIGENS_CORS:  # "*" com credenciais aceitaria qualquer site: recusado (em produção, nem sobe)
    log.warning("ORIGENS_CORS contém '*': ignorado (com credenciais, liste os domínios exatos). Veja problemas_producao.")


_SEMPRE = ("/api/marca", "/api/saude")


@app.middleware("http")
async def banco_pronto(request: Request, call_next):
    """Enquanto o banco não fica pronto, as rotas de dados devolvem o aviso (a tela mostra e espera)."""
    caminho = request.url.path
    if not ESTADO_BANCO["pronto"] and caminho.startswith(("/api/", "/auth/")) and caminho not in _SEMPRE:
        return JSONResponse(ESTADO_BANCO["problema"] or {"banco": "fora", "detail": "Preparando o banco de dados…"},
                            status_code=503)
    return await call_next(request)


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
    resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=(), usb=()")
    if config.COOKIE_SECURE:  # só faz sentido sob HTTPS (o Caddy também põe; repetir é inofensivo)
        resp.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    if not request.url.path.startswith("/api/docs"):
        resp.headers.setdefault("Content-Security-Policy",
                                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data: blob:; frame-src 'self'; frame-ancestors 'self'; "
                                "object-src 'self'; base-uri 'none'; form-action 'self'")
    if request.url.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
    else:  # a tela: o navegador sempre confere com o programa antes de usar o que guardou
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


@app.exception_handler(S.ErroNegocio)
async def _erro_negocio(_, e: S.ErroNegocio):
    return JSONResponse({"detail": str(e)}, status_code=400)


@app.exception_handler(OperationalError)
@app.exception_handler(InterfaceError)
@app.exception_handler(InternalError)
async def _erro_de_banco(request: Request, e: Exception):
    """O banco caiu ou o acesso foi cortado no meio do dia: a tela recebe o aviso (503) e confere /api/saude."""
    p = problema_do_banco(e)
    if not p:
        log.error("Erro do banco em %s: %s", request.url.path, str(getattr(e, "orig", e)).splitlines()[0][:200])
        return JSONResponse({"detail": "Erro interno no banco de dados."}, status_code=500)
    ESTADO_BANCO["problema"] = p
    return JSONResponse(p, status_code=503)


def _testar_banco() -> bool:
    with engine.connect() as con:
        con.execute(text("select 1"))
    return bool(uso_ia.bloqueio())  # o teto de gasto do dia também lê o banco


@app.get("/api/saude")
async def saude():
    """Verificação de saúde: banco e situação da IA. A tela confere aqui se o banco voltou (a cada poucos segundos,
    só enquanto mostra o aviso)."""
    try:
        ia_pausada = await asyncio.to_thread(_testar_banco)
    except Exception as e:
        p = problema_do_banco(e) or {"banco": "fora", "detail": "Sem conexão com o banco de dados."}
        ESTADO_BANCO["problema"] = p
        return JSONResponse(p, status_code=503)
    if not ESTADO_BANCO["pronto"]:  # voltou, mas o escritório ainda está se preparando: acorda a nova tentativa
        if _acordar:
            _acordar.set()
        return JSONResponse({"banco": "preparando", "detail": "O banco voltou: preparando o escritório…"}, status_code=503)
    ESTADO_BANCO["problema"] = None
    return {"ok": True, "banco": "ok", "ia_modo": config.IA_MODO, "ia_pausada": ia_pausada}


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
def login(dados: Login, request: Request, response: Response, db: Session = Depends(obter_db, scope="function")):
    if config.LOCAL:
        raise HTTPException(400, "Neste computador o escritório abre sem login.")
    email = dados.email.strip().lower()
    limitar_login(request.client.host if request.client else "?")
    limitar_conta(email)
    u = db.scalar(select(Usuario).where(Usuario.email == email))
    if not u or not u.ativo or not conferir_senha(dados.senha, u.senha_hash):
        raise HTTPException(401, "E-mail ou senha incorretos.")
    login_ok(email)
    u.ultimo_acesso_em = agora()
    _gravar_sessao(response, u)
    auditar(db, u, "login", "usuario", u.id, request)
    return _usuario_json(u)


class TrocaSenha(BaseModel):
    senha_atual: str
    nova_senha: str = Field(max_length=200)


@app.post("/auth/senha")
def trocar_senha(d: TrocaSenha, request: Request, response: Response, u: Usuario = Depends(usuario_atual),
                 db: Session = Depends(obter_db, scope="function")):
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


@app.get("/api/marca")
def marca():
    """Identidade configurável (APP_NOME etc.) — pública, para a tela de login."""
    return {"nome": config.APP_NOME, "subtitulo": config.APP_SUBTITULO, "empresa": config.EMPRESA_NOME,
            "local": config.LOCAL, "versao": VERSAO, "versao_data": VERSAO_DATA}


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
def escritorio(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    return S.escritorio(db) | {"usuario": _usuario_json(u), "versao": VERSAO}


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
def agente(ag_id: int, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    return S.detalhe_agente(db, _agente(db, ag_id))


@app.get("/api/agentes/{ag_id}/candidatos")
def candidatos(ag_id: int, request: Request, vaga_id: int | None = None,
               u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
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
        resp = await asyncio.to_thread(S.ia.responder_chat, ctx, m.mensagem, m.historico[-10:])
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
            for parte in S.ia.responder_chat_stream(ctx, m.mensagem, m.historico[-10:]):
                enviou = True
                yield parte
        except Exception as e:
            log.warning("Chat (stream) falhou: %s", type(e).__name__)
            yield ("\n" if enviou else "") + "Não consegui consultar a IA agora. Tente de novo em instantes."

    return StreamingResponse(gerar(), media_type="text/plain; charset=utf-8",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/api/equipe/conversas")
def conversas(limite: int = 60, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
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
def ia_uso(u: Usuario = Depends(exigir("supervisor")), db: Session = Depends(obter_db, scope="function")):
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
def contratacoes(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    return S.contratacoes_recentes(db, 30)


@app.get("/api/especialidades")
def especialidades(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    return [{"id": e.id, "codigo": e.codigo, "nome": e.nome, "descricao": e.descricao, "setores": e.setores}
            for e in db.scalars(select(Especialidade).order_by(Especialidade.id)).all()]


class NovaEspecialidade(BaseModel):
    codigo: str = Field(min_length=2, max_length=40)
    nome: str
    descricao: str = ""
    setores: list[str]
    nomes_sugeridos: list[str] = []


@app.post("/api/especialidades")
def nova_especialidade(d: NovaEspecialidade, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db, scope="function")):
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
def candidatura(cid: int, request: Request, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    c = _candidatura(db, cid)
    auditar(db, u, "ver_curriculo", "candidatura", cid, request)
    return S.detalhe_candidatura(db, c)


@app.get("/api/curriculos/{cur_id}/arquivo")
def arquivo(cur_id: int, request: Request, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    cur = db.get(Curriculo, cur_id)
    if not cur or not cur.caminho:
        raise HTTPException(404, "Arquivo indisponível.")
    auditar(db, u, "abrir_arquivo", "curriculo", cur_id, request)
    nome = "curriculo-%d%s" % (cur.id, Path(cur.arquivo_nome).suffix.lower())
    return Response(ler_arquivo(cur.caminho, db), media_type=cur.mime,
                    headers={"Content-Disposition": f'inline; filename="{nome}"'})


# =========================================================================== banco de currículos (arquivo do escritório)
def _curriculo_do_banco(db, cur_id) -> Curriculo:
    cur = db.get(Curriculo, cur_id)
    if not cur or cur.status == "anonimizado" or (cur.candidato and cur.candidato.anonimizado_em):
        raise HTTPException(404, "Currículo não encontrado.")
    return cur


@app.get("/api/banco")
def banco(request: Request, grupo: str | None = None, pasta: str | None = None, situacao: str | None = None,
          busca: str = "", pagina: int = 1, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    if situacao and situacao not in acervo.SITUACOES:
        raise HTTPException(400, "Situação inválida.")
    auditar(db, u, "listar_banco", "curriculo", None, request, grupo=grupo, pasta=pasta, situacao=situacao,
            busca=bool(busca.strip()))
    return acervo.listar(db, grupo, pasta, situacao, busca, pagina)


@app.get("/api/banco/{cur_id}")
def banco_curriculo(cur_id: int, request: Request, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    cur = _curriculo_do_banco(db, cur_id)
    auditar(db, u, "ver_curriculo", "curriculo", cur_id, request)
    return acervo.detalhe(db, cur)


class Encaminhamento(BaseModel):
    vaga_id: int


@app.post("/api/banco/{cur_id}/encaminhar")
def banco_encaminhar(cur_id: int, d: Encaminhamento, request: Request, u: Usuario = Depends(exigir("supervisor")),
                     db: Session = Depends(obter_db, scope="function")):
    """O recrutador tira um currículo do banco e o põe numa vaga aberta: o analista da vaga analisa."""
    cur = _curriculo_do_banco(db, cur_id)
    vaga = db.get(Vaga, d.vaga_id)
    if not vaga or vaga.status != "aberta":
        raise HTTPException(400, "Escolha uma vaga aberta.")
    if cur.status != "triado":
        raise HTTPException(400, "Este currículo ainda não passou pela triagem.")
    if not S._encaminhar(db, cur, vaga, origem="banco"):
        raise HTTPException(409, "Este currículo já está nessa vaga.")
    if cur.candidato:
        cur.candidato.ultima_atividade_em = agora()
    nome = S.primeiro_nome(cur.candidato.nome if cur.candidato else "Candidato")
    eventos.registrar(db, "encaminhamento_manual", f"Atlas buscou no arquivo o currículo de {nome} e o pôs na vaga "
                      f"“{vaga.titulo}” (pedido de {u.nome})", S.atlas(db).id, curriculo_id=cur.id,
                      candidato_id=cur.candidato_id, vaga_id=vaga.id)
    auditar(db, u, "encaminhar_do_banco", "curriculo", cur_id, request, vaga_id=vaga.id)
    return acervo.detalhe(db, cur)


@app.post("/api/banco/{cur_id}/sugerir")
async def banco_sugerir(cur_id: int, request: Request, u: Usuario = Depends(exigir("supervisor"))):
    """O Atlas lê o currículo e diz qual vaga aberta combina (uma chamada à IA; a resposta fica guardada)."""
    def ler():
        with sessao() as db:
            cur = _curriculo_do_banco(db, cur_id)
            return acervo.dados_para_sugestao(db, cur)
    texto, abertas, pasta, anos = await asyncio.to_thread(ler)
    if not abertas:
        raise HTTPException(400, "Não há vagas abertas para sugerir.")
    if not texto:
        raise HTTPException(400, "Este currículo não tem texto legível para o Atlas comparar com as vagas.")
    try:
        sugestao = await asyncio.to_thread(S.ia.sugerir_vaga, texto, abertas, pasta, anos)
    except uso_ia.IAIndisponivel as e:
        raise HTTPException(503, f"A IA está pausada: {e}")
    except Exception as e:
        log.warning("Sugestão do banco falhou: %s", type(e).__name__)
        raise HTTPException(503, "Não consegui consultar a IA agora.")

    def gravar():
        with sessao() as db:
            cur = _curriculo_do_banco(db, cur_id)
            acervo.registrar_sugestao(db, cur, sugestao)
            auditar(db, u, "sugerir_vaga_banco", "curriculo", cur_id, request)
            return acervo.detalhe(db, cur)
    return await asyncio.to_thread(gravar)


class NovaDecisao(BaseModel):
    decisao: str
    observacao: str = Field("", max_length=1000)


@app.post("/api/candidaturas/{cid}/decisao")
def decidir(cid: int, d: NovaDecisao, request: Request, u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
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
def vagas(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    lote = S.Lote(db)
    return [S.vaga_json(db, v, completo=True, lote=lote) for v in db.scalars(select(Vaga).order_by(Vaga.id)).all()]


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
def criar_vaga(d: NovaVaga, request: Request, u: Usuario = Depends(exigir("supervisor")), db: Session = Depends(obter_db, scope="function")):
    invalidos = [s for s in d.setores if s not in CATEGORIAS_DISPONIVEIS]
    if invalidos:
        raise HTTPException(400, f"Pastas inexistentes: {', '.join(invalidos)}")
    v = S.criar_vaga(db, d.model_dump())
    auditar(db, u, "criar_vaga", "vaga", v.id, request)
    return S.vaga_json(db, v, completo=True)


class StatusVaga(BaseModel):
    status: str


@app.patch("/api/vagas/{vid}")
def status_vaga(vid: int, d: StatusVaga, request: Request, u: Usuario = Depends(exigir("supervisor")), db: Session = Depends(obter_db, scope="function")):
    v = db.get(Vaga, vid)
    if not v:
        raise HTTPException(404, "Vaga não encontrada.")
    S.mudar_status_vaga(db, v, d.status)
    auditar(db, u, "status_vaga", "vaga", vid, request, status=d.status)
    return S.vaga_json(db, v, completo=True)


@app.get("/api/quadro")
def quadro(u: Usuario = Depends(usuario_atual), db: Session = Depends(obter_db, scope="function")):
    pend = db.scalars(select(Pendencia).where(Pendencia.status == "aberta").order_by(Pendencia.id.desc())).all()
    nomes = {a.id: a.nome for a in db.scalars(select(Agente)).all()}
    lote = S.Lote(db)
    return {"resumo": S.quadro(db, lote),
            "equipe": [S.agente_json(db, a, lote) for a in lote.ativos],
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
def usuarios(u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db, scope="function")):
    return [_usuario_admin_json(x) for x in db.scalars(select(Usuario).order_by(Usuario.id)).all()]


@app.post("/api/usuarios")
def novo_usuario(d: NovoUsuario, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db, scope="function")):
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
                   db: Session = Depends(obter_db, scope="function")):
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
def redefinir_senha(uid: int, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db, scope="function")):
    alvo = db.get(Usuario, uid)
    if not alvo:
        raise HTTPException(404, "Usuário não encontrado.")
    temp = senha_temporaria()
    alvo.senha_hash, alvo.trocar_senha = gerar_hash(temp), True
    alvo.versao_sessao = (alvo.versao_sessao or 0) + 1
    auditar(db, u, "redefinir_senha", "usuario", alvo.id, request)
    return {"senha_temporaria": temp}


@app.get("/api/lgpd/titulares")
def buscar_titulares(busca: str, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db, scope="function")):
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
def exportar(cid: int, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db, scope="function")):
    c = _titular(db, cid)
    db.add(SolicitacaoTitular(candidato_id=c.id, tipo="acesso", usuario_id=u.id, concluido_em=agora()))
    auditar(db, u, "exportar_titular", "candidato", cid, request)
    return Response(json.dumps(S.exportar_titular(db, c), ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="titular-{cid}.json"'})


class Eliminacao(BaseModel):
    observacao: str = ""


@app.post("/api/lgpd/titulares/{cid}/eliminar")
def eliminar(cid: int, d: Eliminacao, request: Request, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db, scope="function")):
    c = _titular(db, cid)
    S.anonimizar_titular(db, c, "pedido do titular")
    db.add(SolicitacaoTitular(candidato_id=c.id, tipo="eliminacao", observacao=d.observacao, usuario_id=u.id, concluido_em=agora()))
    auditar(db, u, "eliminar_titular", "candidato", cid, request)
    return {"ok": True}


@app.get("/api/auditoria")
def auditoria(limite: int = 200, u: Usuario = Depends(exigir("admin")), db: Session = Depends(obter_db, scope="function")):
    nomes = {x.id: x.nome for x in db.scalars(select(Usuario)).all()}
    return [{"ts": iso(a.ts), "usuario": nomes.get(a.usuario_id), "acao": a.acao, "recurso": a.recurso,
             "recurso_id": a.recurso_id, "ip": a.ip, "detalhes": a.detalhes}
            for a in db.scalars(select(Auditoria).order_by(Auditoria.id.desc()).limit(min(limite, 1000))).all()]


# =========================================================================== front-end
@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def pagina():
    """A página com a versão nos endereços dos arquivos (e sem ficar guardada no navegador)."""
    arq = FRONT / "index.html"
    if not arq.exists():
        raise HTTPException(404, "Tela não encontrada.")
    html = arq.read_text(encoding="utf-8")
    for nome in ARQUIVOS_DA_TELA[1:]:
        html = html.replace(f'"{nome}"', f'"{nome}?v={VERSAO}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


if FRONT.exists():
    app.mount("/", StaticFiles(directory=FRONT, html=True), name="front")
