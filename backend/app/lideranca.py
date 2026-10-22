"""Vários computadores, um escritório só.

Com ACESSO=local cada recrutador roda a própria cópia do programa, todas no mesmo banco (Supabase). Para a
equipe de agentes não trabalhar em dobro, só uma cópia por vez "comanda": roda a fila de IA, a supervisão do
Atlas, a leitura do e-mail e a retenção. A vez é decidida por uma trava do próprio PostgreSQL, presa a uma
conexão: se esse computador fechar o programa, desligar ou perder a rede, a trava se solta sozinha e outra
cópia assume em poucos segundos. As outras continuam abrindo o escritório, enviando currículos e decidindo.

No SQLite (um computador só) esta cópia comanda sempre.
"""
import logging
import time
from contextlib import contextmanager

from sqlalchemy import text

log = logging.getLogger("hub.lideranca")


def _chave(nome: str) -> str:
    return f"escritorio-recrutamento:{nome}"


class Lideranca:
    def __init__(self, eng, nome: str = "comando"):
        self.eng, self.nome = eng, nome
        self.pg = eng.dialect.name == "postgresql"
        self._con = None

    @property
    def lider(self) -> bool:
        return self._con is not None or not self.pg

    def tentar(self) -> bool:
        """Confere se esta cópia ainda comanda ou, se a vez estiver livre, assume. Seguro chamar sempre."""
        if not self.pg:
            return True
        if self._con is not None:
            try:
                self._con.execute(text("SELECT 1"))  # a conexão (e com ela a trava) continua viva?
                return True
            except Exception:
                log.warning("Perdi a conexão que segurava o comando do escritório; outra cópia pode assumir")
                self.soltar()
        con = None
        try:
            con = self.eng.connect().execution_options(isolation_level="AUTOCOMMIT")
            schema = con.execute(text("SELECT current_schema()")).scalar()
            chave = _chave(f"{self.nome}:{schema}")
            if con.execute(text("SELECT pg_try_advisory_lock(hashtext(:k))"), {"k": chave}).scalar():
                self._con = con
                return True
        except Exception as e:
            log.warning("Não consegui verificar quem comanda o escritório: %s", str(e).splitlines()[0][:160])
        if con is not None:
            con.close()
        return False

    def soltar(self):
        if self._con is not None:
            try:
                self._con.invalidate()  # fecha a conexão de verdade (não volta ao pool): a trava se solta
            except Exception:
                pass
            self._con = None


@contextmanager
def exclusivo(eng, nome: str, limite: float = 120):
    """Só uma cópia por vez executa o bloco (ex.: criar tabelas na primeira inicialização de vários
    computadores ao mesmo tempo). As outras esperam a vez, até `limite` segundos."""
    if eng.dialect.name != "postgresql":
        yield
        return
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as con:
        k = _chave(nome)
        fim = time.monotonic() + limite
        while not con.execute(text("SELECT pg_try_advisory_lock(hashtext(:k))"), {"k": k}).scalar():
            if time.monotonic() > fim:
                raise RuntimeError(f"outra cópia do programa ficou tempo demais em: {nome}")
            time.sleep(0.5)
        try:
            yield
        finally:
            con.execute(text("SELECT pg_advisory_unlock(hashtext(:k))"), {"k": k})
