"""Leitura opcional de uma caixa de e-mail (IMAP).

Por padrão fica DESLIGADA: o fluxo de e-mail que a empresa já tem continua chamando
POST /triagem. Ligue com IMAP_ATIVO=true se quiser que o próprio sistema leia a caixa.
"""
import email
import imaplib
import logging
import os
from email.header import decode_header

from . import config
from .db import sessao
from .servicos import ErroNegocio, registrar_curriculo
from .triagem import EXTENSOES

log = logging.getLogger("hub.imap")


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
    with imaplib.IMAP4_SSL(config.IMAP_HOST) as m:
        m.login(config.IMAP_USUARIO, config.IMAP_SENHA)
        m.select(config.IMAP_PASTA)
        _, ids = m.search(None, "UNSEEN")
        for num in ids[0].split():
            _, dados = m.fetch(num, "(RFC822)")
            msg = email.message_from_bytes(dados[0][1])
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
            m.store(num, "+FLAGS", "\\Seen")
    return n
