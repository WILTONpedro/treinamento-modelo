"""Leitura opcional de uma caixa de e-mail (IMAP).

Por padrão fica DESLIGADA: o fluxo de e-mail que a empresa já tem continua chamando
POST /triagem. Ligue com IMAP_ATIVO=true se quiser que o próprio sistema leia a caixa.

Com vários computadores no mesmo banco, só o que comanda a equipe lê a caixa (veja lideranca.py): o mesmo
arquivo de configuração, com o e-mail ligado, serve para todos.

A conexão confere o certificado do servidor (como o navegador faz). Sem isso, um servidor falso no caminho
(Wi‑Fi público, rede comprometida) poderia se passar pelo provedor de e-mail e receber a senha da caixa.
"""
import email
import imaplib
import logging
import os
import ssl
import time
from email.header import decode_header

from sqlalchemy.exc import DBAPIError

from . import config
from .db import sessao
from .servicos import ErroNegocio, registrar_curriculo
from .triagem import EXTENSOES

log = logging.getLogger("hub.imap")
_avisado_em = 0.0


class FalhaEmail(Exception):
    """Não deu para ler a caixa (senha, certificado, rede): vira aviso no escritório, não só uma linha no registro."""


def _nome(valor):
    partes = []
    for txt, enc in decode_header(valor or ""):
        partes.append(txt.decode(enc or "utf-8", errors="ignore") if isinstance(txt, bytes) else txt)
    return "".join(partes)


def ler_caixa() -> int:
    if not (config.IMAP_HOST and config.IMAP_USUARIO and config.IMAP_SENHA):
        log.warning("IMAP ativo, mas IMAP_HOST/IMAP_USUARIO/IMAP_SENHA não foram definidos")
        return 0
    n = 0
    contexto = ssl.create_default_context()  # confere certificado e nome do servidor (o padrão do imaplib não confere)
    try:
        caixa = imaplib.IMAP4_SSL(config.IMAP_HOST, config.IMAP_PORTA, ssl_context=contexto, timeout=60)
    except ssl.SSLCertVerificationError as e:
        raise FalhaEmail(f"o certificado de {config.IMAP_HOST} não é válido (endereço errado ou servidor falso no "
                         "caminho): a senha não foi enviada") from e
    except OSError as e:
        raise FalhaEmail(f"sem conexão com o servidor de e-mail {config.IMAP_HOST}") from e
    with caixa as m:
        try:
            m.login(config.IMAP_USUARIO, config.IMAP_SENHA)
        except imaplib.IMAP4.error as e:
            raise FalhaEmail(f"a caixa {config.IMAP_USUARIO} recusou o usuário e a senha (a senha de app mudou?)") from e
        m.select(config.IMAP_PASTA)
        _, ids = m.search(None, "UNSEEN")
        for num in ids[0].split():
            # PEEK: a mensagem só fica "lida" depois que os anexos estão no banco. Se o banco cair no meio, ela volta na
            # próxima leitura (o arquivo que já entrou não é registrado de novo).
            _, dados = m.fetch(num, "(BODY.PEEK[])")
            msg = email.message_from_bytes(dados[0][1])
            try:
                for parte in msg.walk():
                    if parte.get_content_disposition() != "attachment":
                        continue
                    nome = _nome(parte.get_filename())
                    if os.path.splitext(nome)[1].lower() not in EXTENSOES:
                        continue
                    try:
                        with sessao() as db:
                            _, novo = registrar_curriculo(db, parte.get_payload(decode=True) or b"", nome,
                                                          parte.get_content_type(), "imap")
                            n += int(novo)
                    except ErroNegocio as e:
                        log.info("Anexo ignorado: %s", e)
            except DBAPIError:
                raise  # banco fora: para aqui e tenta de novo na próxima leitura
            except Exception:  # um e-mail estranho não trava os seguintes
                log.exception("E-mail: mensagem %s ignorada por um erro inesperado", num.decode() if isinstance(num, bytes) else num)
            m.store(num, "+FLAGS", "\\Seen")
    if n:
        log.info("E-mail: %d currículo(s) novo(s) na caixa %s", n, config.IMAP_USUARIO)
    return n


def ler_e_avisar() -> int:
    """Lê a caixa. Se não der, a recepção abre uma pendência com o motivo (aparece no escritório e no quadro); quando
    a leitura volta a funcionar, a pendência se fecha sozinha. No registro, uma linha a cada meia hora, sem repetir."""
    global _avisado_em
    from .servicos import _abrir_pendencia, _resolver_pendencias, recepcao
    try:
        n = ler_caixa()
    except FalhaEmail as e:
        if time.monotonic() - _avisado_em > 1800:
            _avisado_em = time.monotonic()
            log.warning("E-mail: %s", e)
        with sessao() as db:
            _abrir_pendencia(db, "EMAIL", f"não consegui ler o e-mail dos currículos: {e}", recepcao(db))
        return 0
    _avisado_em = 0.0
    with sessao() as db:
        _resolver_pendencias(db, "EMAIL")
    return n
