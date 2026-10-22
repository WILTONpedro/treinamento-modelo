"""Arquivos originais dos currículos, sempre cifrados (Fernet) antes de serem guardados.

ARMAZENAMENTO=disco  (padrão) um arquivo por currículo em PASTA_DADOS/arquivos
ARMAZENAMENTO=banco  no próprio banco (tabela arquivos_curriculo): com Supabase/PostgreSQL os currículos,
                     os dados e os arquivos ficam no mesmo lugar e entram juntos nos backups

O `caminho` gravado no currículo diz onde o arquivo está ("banco:<sha256>" ou o caminho no disco), então
currículos antigos continuam legíveis depois de trocar o armazenamento.
"""
import base64
import os
import secrets

from sqlalchemy import delete

from . import config
from .db import sessao
from .models import ArquivoCurriculo
from .seguranca import cifra

PREFIXO_BANCO = "banco:"


def _no_banco(db, acao):
    if db is not None:
        return acao(db)
    with sessao() as s:
        return acao(s)


def salvar_arquivo(sha: str, conteudo: bytes, db=None) -> str:
    token = cifra().encrypt(conteudo)
    if config.ARMAZENAMENTO == "banco":
        bruto = base64.urlsafe_b64decode(token)  # o token em base64 ocupa 1/3 a mais

        def _gravar(s):
            if not s.get(ArquivoCurriculo, sha):
                s.add(ArquivoCurriculo(sha256=sha, conteudo=bruto))
        _no_banco(db, _gravar)
        return PREFIXO_BANCO + sha
    caminho = config.PASTA_ARQUIVOS / f"{sha}.bin"
    if not caminho.exists():
        caminho.write_bytes(token)
    return str(caminho)


def ler_arquivo(caminho: str, db=None) -> bytes:
    if caminho.startswith(PREFIXO_BANCO):
        reg = _no_banco(db, lambda s: s.get(ArquivoCurriculo, caminho[len(PREFIXO_BANCO):]))
        if reg is None:
            raise FileNotFoundError("Arquivo do currículo não encontrado no banco.")
        return cifra().decrypt(base64.urlsafe_b64encode(reg.conteudo))
    with open(caminho, "rb") as f:
        return cifra().decrypt(f.read())


def apagar_arquivo(caminho: str | None, db=None):
    if not caminho:
        return
    if caminho.startswith(PREFIXO_BANCO):
        _no_banco(db, lambda s: s.execute(delete(ArquivoCurriculo).where(
            ArquivoCurriculo.sha256 == caminho[len(PREFIXO_BANCO):])))
    elif os.path.exists(caminho):
        # sobrescreve antes de remover (melhor esforço em discos comuns)
        tam = os.path.getsize(caminho)
        with open(caminho, "r+b") as f:
            f.write(secrets.token_bytes(tam))
        os.remove(caminho)
