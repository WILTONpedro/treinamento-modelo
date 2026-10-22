import logging
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
                con.execute(text(f"ALTER TABLE {tabela.name} ADD COLUMN {col.name} {tipo}"))
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
