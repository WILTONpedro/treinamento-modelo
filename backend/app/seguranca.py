"""Autenticação, papéis, criptografia de arquivos e auditoria."""
import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import timedelta

import jwt
from cryptography.fernet import Fernet
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from . import config
from .db import obter_db
from .models import Usuario, Auditoria, agora

log = logging.getLogger("hub.seguranca")
COOKIE = "hub_sessao"
PAPEIS = {"recrutador": 1, "supervisor": 2, "admin": 3}


# --------------------------------------------------------------------------- senhas (scrypt, stdlib)
def gerar_hash(senha: str) -> str:
    sal = os.urandom(16)
    h = hashlib.scrypt(senha.encode(), salt=sal, n=2 ** 14, r=8, p=1, dklen=32)
    return "scrypt$" + base64.b64encode(sal).decode() + "$" + base64.b64encode(h).decode()


def conferir_senha(senha: str, armazenado: str) -> bool:
    try:
        _, sal, h = armazenado.split("$")
        calc = hashlib.scrypt(senha.encode(), salt=base64.b64decode(sal), n=2 ** 14, r=8, p=1, dklen=32)
        return hmac.compare_digest(calc, base64.b64decode(h))
    except Exception:
        return False


# --------------------------------------------------------------------------- sessão (JWT em cookie httpOnly)
def criar_token(u: Usuario) -> str:
    exp = agora() + timedelta(hours=config.SESSAO_HORAS)
    return jwt.encode({"sub": str(u.id), "papel": u.papel, "v": u.versao_sessao or 0, "exp": exp},
                      config.SEGREDO_JWT, algorithm="HS256")


def validar_senha(senha: str) -> str | None:
    """Mensagem de erro, ou None se a senha serve."""
    if len(senha) < 10:
        return "A senha precisa ter pelo menos 10 caracteres."
    if senha.isdigit() or len(set(senha)) < 4:
        return "Escolha uma senha menos previsível (misture letras, números ou símbolos)."
    return None


def senha_temporaria() -> str:
    return secrets.token_urlsafe(9)  # 12 caracteres; o usuário troca no primeiro acesso


ROTAS_LIVRES_DE_TROCA = ("/auth/eu", "/auth/senha", "/auth/sair", "/api/marca")


_tentativas: dict[str, deque] = defaultdict(deque)


def limitar_login(ip: str):
    """No máximo 8 tentativas por IP a cada 5 minutos."""
    agora_s = time.time()
    fila = _tentativas[ip]
    while fila and agora_s - fila[0] > 300:
        fila.popleft()
    if len(fila) >= 8:
        raise HTTPException(429, "Muitas tentativas. Aguarde alguns minutos.")
    fila.append(agora_s)


def usuario_atual(request: Request, db: Session = Depends(obter_db)) -> Usuario:
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, "Sessão expirada. Entre novamente.")
    try:
        dados = jwt.decode(token, config.SEGREDO_JWT, algorithms=["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Sessão inválida. Entre novamente.")
    u = db.get(Usuario, int(dados["sub"]))
    if not u or not u.ativo:
        raise HTTPException(401, "Usuário inativo.")
    if (dados.get("v") or 0) != (u.versao_sessao or 0):
        raise HTTPException(401, "Sua sessão foi encerrada (senha ou acesso alterados). Entre novamente.")
    if u.trocar_senha and request.url.path not in ROTAS_LIVRES_DE_TROCA:
        raise HTTPException(403, "Troque a senha temporária para continuar.")
    # Proteção CSRF: toda escrita precisa do cabeçalho enviado pelo front (além do SameSite=Strict)
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get("X-Hub") != "1":
        raise HTTPException(403, "Requisição sem cabeçalho de origem.")
    return u


def exigir(papel_minimo: str):
    def dep(u: Usuario = Depends(usuario_atual)) -> Usuario:
        if PAPEIS.get(u.papel, 0) < PAPEIS[papel_minimo]:
            raise HTTPException(403, "Seu perfil não tem permissão para esta ação.")
        return u
    return dep


def conferir_chave_triagem(request: Request):
    if not config.TRIAGEM_API_KEY:
        return  # aviso emitido no início do servidor
    enviada = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(enviada, config.TRIAGEM_API_KEY):
        raise HTTPException(401, "Chave da API de triagem inválida.")


# --------------------------------------------------------------------------- auditoria
def auditar(db: Session, usuario: Usuario | None, acao: str, recurso: str, recurso_id=None,
            request: Request | None = None, **detalhes):
    db.add(Auditoria(usuario_id=usuario.id if usuario else None, acao=acao, recurso=recurso,
                     recurso_id=recurso_id, ip=(request.client.host if request and request.client else None),
                     detalhes=detalhes))


# --------------------------------------------------------------------------- chave dos arquivos (ver arquivos.py)
_fernet: Fernet | None = None


def cifra() -> Fernet:
    global _fernet
    if _fernet:
        return _fernet
    chave = config.CHAVE_ARQUIVOS
    if not chave:
        arq = config.PASTA_DADOS / ".chave_arquivos"
        if arq.exists():
            chave = arq.read_text().strip()
        else:
            chave = Fernet.generate_key().decode()
            arq.write_text(chave)
            os.chmod(arq, 0o600)
            log.warning("CHAVE_ARQUIVOS não definida: chave gerada em %s. Guarde-a com segurança — "
                        "sem ela os currículos não podem ser lidos.", arq)
    _fernet = Fernet(chave.encode() if isinstance(chave, str) else chave)
    return _fernet
