"""Leitura da caixa de e-mail (IMAP): confere o certificado do servidor e registra os anexos como currículos."""
import imaplib
import ssl
from email.message import EmailMessage

import pytest
from sqlalchemy import select

from app import config, email_imap
from app.db import sessao
from app.models import Curriculo

CV = """Rita Fictícia
Auxiliar de cozinha · Campinas - SP
EXPERIÊNCIA
Auxiliar de cozinha — Restaurante Sabor Caseiro (2022–2025): preparo de massas e organização do estoque.
FORMAÇÃO
Ensino médio completo."""


def _mensagem(cv: str = CV, arquivo: str = "rita_ficticia.txt") -> bytes:
    m = EmailMessage()
    m["From"], m["To"], m["Subject"] = "rita@example.com", "vagas@empresa.sim", "Currículo"
    m.set_content("Segue meu currículo.")
    m.add_attachment(cv.encode(), maintype="text", subtype="plain", filename=arquivo)
    m.add_attachment(b"\x89PNG", maintype="application", subtype="x-msdownload", filename="virus.exe")  # ignorado
    return m.as_bytes()


class CaixaFalsa:
    """Imita o servidor IMAP: guarda como foi chamado e entrega uma mensagem não lida."""
    chamadas, vistas = [], set()

    def __init__(self, host, port=993, *, ssl_context=None, timeout=None):
        CaixaFalsa.chamadas.append({"host": host, "port": port, "ssl_context": ssl_context, "timeout": timeout})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def login(self, usuario, senha):
        assert (usuario, senha) == ("vagas@empresa.sim", "senha-de-app")

    def select(self, pasta):
        assert pasta == "INBOX"

    def search(self, _, criterio):
        return "OK", [b"" if "1" in self.vistas else b"1"]

    def fetch(self, num, partes):
        if "PEEK" not in partes:  # como um servidor de verdade: ler o corpo sem PEEK já marca como lida
            self.vistas.add(num.decode() if isinstance(num, bytes) else num)
        return "OK", [(b"1 (BODY[] {999}", self.mensagem())]

    def mensagem(self):
        return _mensagem()

    def store(self, num, _, flag):
        assert flag == "\\Seen"
        self.vistas.add(num.decode() if isinstance(num, bytes) else num)


def test_le_a_caixa_conferindo_o_certificado(cli, monkeypatch):
    monkeypatch.setattr(imaplib, "IMAP4_SSL", CaixaFalsa)
    monkeypatch.setattr(config, "IMAP_HOST", "imap.empresa.sim")
    monkeypatch.setattr(config, "IMAP_USUARIO", "vagas@empresa.sim")
    monkeypatch.setattr(config, "IMAP_SENHA", "senha-de-app")
    assert email_imap.ler_caixa() == 1
    c = CaixaFalsa.chamadas[-1]
    assert c["host"] == "imap.empresa.sim" and c["port"] == 993 and c["timeout"]
    # sem isto, um servidor falso no caminho receberia a senha da caixa
    assert c["ssl_context"].verify_mode == ssl.CERT_REQUIRED and c["ssl_context"].check_hostname
    with sessao() as db:
        cvs = db.scalars(select(Curriculo).where(Curriculo.origem == "imap")).all()
    assert [x.arquivo_nome for x in cvs] == ["rita_ficticia.txt"]  # o .exe ficou de fora
    # a mensagem foi marcada como lida: a próxima leitura não traz nada de novo
    assert email_imap.ler_caixa() == 0


def test_porta_configuravel(cli, monkeypatch):
    monkeypatch.setattr(imaplib, "IMAP4_SSL", CaixaFalsa)
    monkeypatch.setattr(config, "IMAP_HOST", "imap.empresa.sim")
    monkeypatch.setattr(config, "IMAP_USUARIO", "vagas@empresa.sim")
    monkeypatch.setattr(config, "IMAP_SENHA", "senha-de-app")
    monkeypatch.setattr(config, "IMAP_PORTA", 1993)
    email_imap.ler_caixa()
    assert CaixaFalsa.chamadas[-1]["port"] == 1993


def test_caixa_que_falha_vira_pendencia_e_se_resolve_sozinha(cli, monkeypatch):
    from app.models import Pendencia

    class SenhaTrocada(CaixaFalsa):
        def login(self, usuario, senha):
            raise imaplib.IMAP4.error(b"[AUTHENTICATIONFAILED] Invalid credentials")

    monkeypatch.setattr(config, "IMAP_HOST", "imap.empresa.sim")
    monkeypatch.setattr(config, "IMAP_USUARIO", "vagas@empresa.sim")
    monkeypatch.setattr(config, "IMAP_SENHA", "senha-de-app")
    monkeypatch.setattr(imaplib, "IMAP4_SSL", SenhaTrocada)

    def abertas():
        from app.models import Agente
        with sessao() as db:
            return [(p.descricao, db.get(Agente, p.agente_id).nome if p.agente_id else None) for p in db.scalars(
                select(Pendencia).where(Pendencia.tipo == "EMAIL", Pendencia.status == "aberta")).all()]

    assert email_imap.ler_e_avisar() == 0 and email_imap.ler_e_avisar() == 0
    (desc, quem), = abertas()  # uma só, mesmo falhando várias vezes
    assert "recusou o usuário e a senha" in desc and quem == "Bia"

    def certificado_falso(*a, **k):
        raise ssl.SSLCertVerificationError("certificate verify failed")
    monkeypatch.setattr(imaplib, "IMAP4_SSL", certificado_falso)
    with pytest.raises(email_imap.FalhaEmail, match="a senha não foi enviada"):
        email_imap.ler_caixa()

    monkeypatch.setattr(imaplib, "IMAP4_SSL", CaixaFalsa)  # a senha certa voltou
    email_imap.ler_e_avisar()
    assert abertas() == []


def test_banco_que_cai_no_meio_nao_perde_o_email(cli, monkeypatch):
    """A mensagem só fica "lida" depois que o currículo está no banco: se o banco cair no meio, ela volta depois."""
    from sqlalchemy.exc import OperationalError

    class OutraCaixa(CaixaFalsa):
        vistas = set()

        def mensagem(self):
            return _mensagem(CV.replace("Rita", "Joana"), "joana_ficticia.txt")

    monkeypatch.setattr(imaplib, "IMAP4_SSL", OutraCaixa)
    monkeypatch.setattr(config, "IMAP_HOST", "imap.empresa.sim")
    monkeypatch.setattr(config, "IMAP_USUARIO", "vagas@empresa.sim")
    monkeypatch.setattr(config, "IMAP_SENHA", "senha-de-app")
    registrar = email_imap.registrar_curriculo

    def banco_caiu(*a, **k):
        raise OperationalError("INSERT", {}, Exception("server closed the connection unexpectedly"))
    monkeypatch.setattr(email_imap, "registrar_curriculo", banco_caiu)
    with pytest.raises(OperationalError):
        email_imap.ler_caixa()
    assert OutraCaixa.vistas == set()  # continua não lida no servidor

    monkeypatch.setattr(email_imap, "registrar_curriculo", registrar)  # o banco voltou
    assert email_imap.ler_caixa() == 1 and OutraCaixa.vistas == {"1"}
    with sessao() as db:
        assert db.scalar(select(Curriculo).where(Curriculo.arquivo_nome == "joana_ficticia.txt"))
