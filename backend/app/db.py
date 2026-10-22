import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from . import config


class Base(DeclarativeBase):
    pass


_sqlite = config.DATABASE_URL.startswith("sqlite")
engine = create_engine(
    config.DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if _sqlite else {},
    pool_pre_ping=True,
)

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
            if novas:
                for indice in tabela.indexes:
                    indice.create(bind=con, checkfirst=True)


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
