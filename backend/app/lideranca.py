"""Vários computadores, um escritório só.

Com ACESSO=local cada recrutador roda a própria cópia do programa, todas no mesmo banco (Supabase). Para a
equipe de agentes não trabalhar em dobro, só uma cópia por vez "comanda": roda a fila de IA, a supervisão do
Atlas, a leitura do e-mail e a retenção. A vez fica numa linha da tabela `comando`, com prazo: quem comanda
renova a cada poucos segundos; se o programa fechar (ele devolve a vez na hora) ou o computador desligar ou
perder a rede (o prazo vence), outra cópia assume. As outras continuam abrindo o escritório, enviando
currículos e decidindo.

Tudo em transações curtas, sem conexão presa: funciona no pooler do Supabase em modo transação, que atende
muitos computadores ao mesmo tempo. No SQLite (um computador só) esta cópia comanda sempre.
"""
import logging
import os
import socket
import time
import uuid

from sqlalchemy import text

from . import config

log = logging.getLogger("hub.lideranca")

PRAZO = 20  # segundos sem renovar até outro computador poder assumir


class Lideranca:
    def __init__(self, eng, nome: str = "equipe"):
        self.eng, self.nome = eng, nome
        self.pg = eng.dialect.name == "postgresql"
        self.eu = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"[:200]
        self._valido_ate = 0.0  # relógio deste computador: até quando a vez está garantida
        self._avisado_em = 0.0  # sem banco por muito tempo: um aviso por minuto no registro, não a cada 5 s

    def tentar(self) -> bool:
        """Renova a vez (se já comanda) ou assume (se ninguém comanda ou o prazo venceu). Seguro chamar sempre."""
        if not self.pg:
            return True
        try:
            with self.eng.begin() as con:
                dono = con.execute(text(
                    "INSERT INTO comando (nome, dono, ate) VALUES (:n, :eu, now() + make_interval(secs => :p)) "
                    "ON CONFLICT (nome) DO UPDATE SET dono = EXCLUDED.dono, ate = EXCLUDED.ate "
                    "WHERE comando.dono = EXCLUDED.dono OR comando.ate < now() RETURNING dono"),
                    {"n": self.nome, "eu": self.eu, "p": PRAZO}).scalar()
        except Exception as e:  # sem banco: segue só até o fim do prazo que já tinha (depois, outro pode assumir)
            if time.monotonic() - self._avisado_em > 60:
                self._avisado_em = time.monotonic()
                log.warning("Não consegui renovar o comando da equipe: %s", str(e).splitlines()[0][:160])
            return time.monotonic() < self._valido_ate
        if dono == self.eu:
            self._valido_ate = time.monotonic() + PRAZO - 5
            return True
        self._valido_ate = 0.0
        return False

    def soltar(self):
        """Devolve a vez na hora (programa fechando), para outro computador assumir sem esperar o prazo."""
        if self.pg and self._valido_ate:
            try:
                with self.eng.begin() as con:
                    con.execute(text("DELETE FROM comando WHERE nome = :n AND dono = :eu"), {"n": self.nome, "eu": self.eu})
            except Exception:
                pass
        self._valido_ate = 0.0


def esperar_vez(con, nome: str):
    """Só uma transação por vez passa daqui (as outras esperam) até o fim dela; ex.: vários computadores
    abrindo juntos às 8h não criam a equipe inicial em dobro. Trava de transação: segura no pooler."""
    dialeto = con.get_bind().dialect if hasattr(con, "get_bind") else con.dialect  # Session ou Connection
    if dialeto.name != "postgresql":
        return
    con.execute(text("SET LOCAL lock_timeout = '120s'"))
    # a chave leva o schema: empresas que dividem o mesmo projeto do Supabase não esperam umas pelas outras
    con.execute(text("SELECT pg_advisory_xact_lock(hashtext(:k))"),
                {"k": f"escritorio-recrutamento:{config.DB_SCHEMA}:{nome}"})
