import logging
import re
from contextlib import contextmanager
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from . import config


class Base(DeclarativeBase):
    pass


_sqlite = config.DATABASE_URL.startswith("sqlite")
_args = {"check_same_thread": False, "timeout": 30} if _sqlite else {"connect_timeout": 15, "application_name": "escritorio-recrutamento"}
if ":6543/" in config.DATABASE_URL:  # pooler do Supabase em modo transação: sem prepared statements
    _args["prepare_threshold"] = None
engine = create_engine(
    config.DATABASE_URL,
    connect_args=_args,
    pool_pre_ping=True,
    **({} if _sqlite else {"pool_size": 5, "max_overflow": 5, "pool_recycle": 300}),
)

# O pooler do Supabase fica em aws-0-<região> ou aws-1-<região>, conforme o projeto, e o painel é o único lugar
# que diz qual. Se o endereço configurado não conhecer o projeto ("Tenant or user not found"), tenta o outro e
# passa a usá-lo: a configuração funciona de primeira nos dois casos.
_POOLER = re.compile(r"^aws-(\d+)-([a-z0-9-]+\.pooler\.supabase\.com)$")
_pooler_certo: dict[str, str] = {}


def _outro_pooler(host: str) -> str | None:
    m = _POOLER.match(host or "")
    return f"aws-{1 if m.group(1) == '0' else 0}-{m.group(2)}" if m else None


def _projeto_desconhecido(e: Exception) -> bool:
    """O pooler não conhece o projeto: "Tenant or user not found" ou, nas versões novas,
    "(ENOTFOUND) tenant/user <usuário>.<projeto> not found"."""
    msg = str(e).lower()
    return "tenant or user not found" in msg or ("tenant/user" in msg and "not found" in msg)


@event.listens_for(engine, "do_connect")
def _conectar_no_pooler_certo(dialect, conn_rec, cargs, cparams):
    host = cparams.get("host") or ""
    if not _outro_pooler(host):
        return None  # conexão normal
    tentativas = [_pooler_certo.get(host, host)]
    tentativas.append(_outro_pooler(tentativas[0]))
    erro = None
    for h in tentativas:
        try:
            con = dialect.loaded_dbapi.connect(*cargs, **{**cparams, "host": h})
        except Exception as e:
            if not _projeto_desconhecido(e):
                raise
            erro = e
            continue
        if h != host and _pooler_certo.get(host) != h:
            log.warning("O pooler deste projeto é %s (não %s): ajuste a DATABASE_URL quando puder", h, host)
        _pooler_certo[host] = h
        return con
    raise erro


if config.DB_SCHEMA and not _sqlite:
    @event.listens_for(engine, "connect")
    def _schema(con, _):
        cur = con.cursor()
        cur.execute('SET search_path TO "%s"' % config.DB_SCHEMA.replace('"', ''))
        cur.close()
        con.commit()

if _sqlite:
    @event.listens_for(engine, "connect")
    def _pragmas(con, _):
        cur = con.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

Sessao = sessionmaker(bind=engine, expire_on_commit=False)
log = logging.getLogger("hub.db")


def migrar(eng=None):
    """Acrescenta colunas e índices novos em bancos já existentes.

    `create_all` cria tabelas que faltam, mas não altera as que já existem; sem isto,
    quem atualiza o sistema perderia os dados ou teria de recriar o banco.
    """
    eng = eng or engine
    insp = inspect(eng)
    with eng.begin() as con:
        for tabela in Base.metadata.sorted_tables:
            if not insp.has_table(tabela.name):
                continue
            existentes = {c["name"] for c in insp.get_columns(tabela.name)}
            novas = [c for c in tabela.columns if c.name not in existentes]
            for col in novas:
                tipo = col.type.compile(dialect=eng.dialect)
                # IF NOT EXISTS: dois computadores atualizados ao mesmo tempo não brigam pela mesma coluna
                se_faltar = "IF NOT EXISTS " if eng.dialect.name == "postgresql" else ""
                con.execute(text(f"ALTER TABLE {tabela.name} ADD COLUMN {se_faltar}{col.name} {tipo}"))
                log.info("Migração: coluna %s.%s criada", tabela.name, col.name)
    for tabela in Base.metadata.sorted_tables:  # índices novos também chegam a bancos antigos
        for indice in tabela.indexes:
            try:
                with eng.begin() as con:
                    indice.create(bind=con, checkfirst=True)
            except Exception as e:  # ex.: dados antigos duplicados impedem um índice único; o sistema sobe assim mesmo
                log.warning("Índice %s não criado: %s", indice.name, str(e).splitlines()[0][:160])
    if eng.dialect.name == "postgresql":  # Supabase: RLS, proteção contra exclusão, busca e visões
        from .banco_pg import ajustar
        ajustar(eng, Base.metadata.sorted_tables)


@contextmanager
def sessao():
    s = Sessao()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def obter_db():
    """Dependência do FastAPI."""
    s = Sessao()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
