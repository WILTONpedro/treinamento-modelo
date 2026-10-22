"""Solidez do banco: regras de integridade, chegada simultânea e (no PostgreSQL/Supabase) proteções e busca."""
import hashlib
from pathlib import Path

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from app import servicos as S
from app import triagem
from app.banco_pg import script
from app.db import Base, engine, sessao
from app.models import Auditoria, Candidato, Curriculo, Tarefa

PG = engine.dialect.name == "postgresql"
so_pg = pytest.mark.skipif(not PG, reason="proteções do PostgreSQL (rode com DATABASE_URL do Postgres/Supabase)")


def test_banco_recusa_estado_invalido(cli):
    with pytest.raises(IntegrityError):
        with sessao() as db:
            db.add(Tarefa(tipo="TRIAGEM", status="feito"))
    with pytest.raises(IntegrityError):
        with sessao() as db:
            db.add(Curriculo(sha256="x" * 64, arquivo_nome="a.txt", origem="fax"))


def test_um_email_um_candidato_ativo(cli):
    with sessao() as db:
        db.add(Candidato(nome="Primeira Pessoa", email="mesmo@exemplo.com"))
    with pytest.raises(IntegrityError):
        with sessao() as db:
            db.add(Candidato(nome="Outra Pessoa", email="mesmo@exemplo.com"))
    with sessao() as db:  # anonimizado não conta: o titular pode voltar a se candidatar
        c = db.scalar(select(Candidato).where(Candidato.email == "mesmo@exemplo.com"))
        c.email, c.anonimizado_em = None, c.criado_em
        db.add(Candidato(nome="Voltou", email="mesmo@exemplo.com"))


def test_mesmo_curriculo_chegando_ao_mesmo_tempo(cli, monkeypatch):
    conteudo = "Joana Duplicada\njoana.dup@exemplo.com\nExperiência: auxiliar administrativa, Excel.".encode()
    sha = hashlib.sha256(conteudo).hexdigest()
    real, vez = triagem.extrair_texto, {"n": 0}

    def extrair(c, nome):  # na primeira leitura, "outro pedido" com o mesmo arquivo grava antes deste
        vez["n"] += 1
        if vez["n"] == 1:
            with sessao() as outro:
                S.registrar_curriculo(outro, c, nome, "text/plain", "email")
        return real(c, nome)
    monkeypatch.setattr(triagem, "extrair_texto", extrair)
    cid, novo, _ = S.registrar_com_seguranca(conteudo, "duplo.txt", "text/plain", "email")
    assert novo is False and vez["n"] >= 2
    with sessao() as db:
        assert db.scalar(select(func.count()).select_from(Curriculo).where(Curriculo.sha256 == sha)) == 1
        assert db.get(Curriculo, cid).sha256 == sha


@so_pg
def test_postgres_protege_dados_e_auditoria(cli):
    with sessao() as db:
        cid = db.scalar(select(Curriculo.id).limit(1))
        db.add(Auditoria(acao="teste", recurso="banco"))
    assert cid, "a suíte envia currículos antes deste teste"
    with pytest.raises(Exception, match="Exclusão bloqueada"):
        with engine.begin() as con:
            con.execute(text("DELETE FROM curriculos WHERE id = :i"), {"i": cid})
    with pytest.raises(Exception, match="somente inclus|só aceita inclus"):
        with engine.begin() as con:
            con.execute(text("UPDATE auditoria SET acao = 'alterada'"))
    with pytest.raises(Exception, match="Exclusão bloqueada"):
        with engine.begin() as con:
            con.execute(text("TRUNCATE candidatos CASCADE"))
    with engine.begin() as con:  # manutenção planejada continua possível, de forma explícita
        con.execute(text("SET LOCAL hub.permitir_exclusao = on"))
        con.execute(text("UPDATE auditoria SET detalhes = detalhes WHERE acao = 'teste'"))


@so_pg
def test_postgres_busca_visoes_e_rls(cli):
    with sessao() as db:
        S.registrar_curriculo(db, "Marcelo Busca\nOperador de empilhadeira com NR-11 e experiência em logística.".encode(),
                              "busca.txt", "text/plain", "upload")
    with engine.connect() as con:
        achados = con.execute(text("SELECT candidato, trecho FROM buscar_curriculos('empilhadeira logística')")).all()
        assert achados and "empilhadeira" in achados[0].trecho.lower()
        for v in ("v_curriculos", "v_vagas", "v_fila", "v_uso_ia_diario", "v_saude"):
            con.execute(text(f"SELECT * FROM {v} LIMIT 1")).all()
        sem_rls = con.execute(text("SELECT relname FROM pg_class WHERE relnamespace = current_schema()::regnamespace "
                                   "AND relkind = 'r' AND NOT relrowsecurity")).all()
        assert not sem_rls


def test_script_do_supabase_em_dia():
    """supabase/reforco.sql sai do código; se mudar as regras, rode: python -m app.banco_pg > ../supabase/reforco.sql"""
    arquivo = Path(__file__).resolve().parents[2] / "supabase" / "reforco.sql"
    assert arquivo.read_text(encoding="utf-8") == script(Base.metadata.sorted_tables)


def test_mesmo_email_em_triagens_paralelas(cli):
    with sessao() as outra:  # outra triagem grava o mesmo e-mail entre a consulta e a inclusão desta
        outra.add(Candidato(nome="Paralela Um", email="paralela@exemplo.com"))
    with sessao() as db:
        real, vez = db.scalar, {"n": 0}

        def scalar(*a, **k):
            vez["n"] += 1
            return None if vez["n"] == 1 else real(*a, **k)
        db.scalar = scalar
        cand = S._vincular_candidato(db, {"nome": "Paralela Dois", "email": "Paralela@exemplo.com"})
        assert cand.id and cand.nome == "Paralela Dois"
    with sessao() as db:
        assert db.scalar(select(func.count()).select_from(Candidato).where(Candidato.email == "paralela@exemplo.com")) == 1
