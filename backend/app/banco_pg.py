"""Ajustes que só existem no PostgreSQL (Supabase), aplicados a cada inicialização e sem efeito se já existem.

- regras de validação (CHECK) também em bancos criados antes delas;
- RLS em todas as tabelas, sem políticas: o sistema conecta como dono das tabelas (não é afetado); qualquer
  outro papel, como o da API pública do Supabase, não lê linha alguma;
- proteção contra perda: candidatos, currículos, candidaturas, análises, decisões, vagas, usuários e pedidos de
  titulares não são apagados (a LGPD é atendida por anonimização); a auditoria só aceita inclusões;
- arquivos cifrados guardados sem tentativa de compressão (cifrado não comprime);
- busca por texto nos currículos, em português (índice + função buscar_curriculos);
- visões para consultar pelo painel do Supabase (v_curriculos, v_vagas, v_fila, v_uso_ia_diario, v_saude).

Para uma manutenção que precise mesmo apagar algo: `SET hub.permitir_exclusao = on;` na mesma sessão.
"""
import logging

from sqlalchemy import CheckConstraint, text

log = logging.getLogger("hub.banco")

PROTEGIDAS = ("candidatos", "curriculos", "candidaturas", "analises", "decisoes", "vagas", "usuarios",
              "solicitacoes_titular", "auditoria")

FUNCOES = """
CREATE OR REPLACE FUNCTION {s}.impedir_exclusao() RETURNS trigger LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF current_setting('hub.permitir_exclusao', true) = 'on' THEN
    RETURN CASE WHEN TG_LEVEL = 'ROW' THEN OLD END;
  END IF;
  RAISE EXCEPTION 'Exclusão bloqueada em %: dados de candidatos são anonimizados (LGPD), não apagados', TG_TABLE_NAME
    USING ERRCODE = 'insufficient_privilege',
          HINT = 'Em manutenção planejada: SET hub.permitir_exclusao = on; na mesma sessão.';
END $$;

CREATE OR REPLACE FUNCTION {s}.auditoria_somente_inclusao() RETURNS trigger LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF current_setting('hub.permitir_exclusao', true) = 'on' THEN
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
  END IF;
  RAISE EXCEPTION 'A auditoria só aceita inclusões (art. 37 da LGPD)' USING ERRCODE = 'insufficient_privilege';
END $$;

CREATE OR REPLACE FUNCTION {s}.buscar_curriculos(termo text, limite integer DEFAULT 20)
RETURNS TABLE (curriculo_id integer, candidato text, setor text, recebido_em timestamp, relevancia real, trecho text)
LANGUAGE sql STABLE SET search_path = {s}, pg_catalog AS $$
  SELECT c.id, coalesce(k.nome, 'Candidato')::text, c.setor::text, c.recebido_em,
         ts_rank(to_tsvector('portuguese', coalesce(c.texto, '')), q),
         ts_headline('portuguese', coalesce(c.texto, ''), q, 'MaxFragments=2, MaxWords=18, MinWords=6, StartSel=«, StopSel=»')
  FROM curriculos c
  LEFT JOIN candidatos k ON k.id = c.candidato_id
  CROSS JOIN websearch_to_tsquery('portuguese', termo) q
  WHERE c.status <> 'anonimizado' AND to_tsvector('portuguese', coalesce(c.texto, '')) @@ q
  ORDER BY 5 DESC, c.recebido_em DESC
  LIMIT greatest(1, least(limite, 200))
$$;
"""

VISOES = {
    "v_curriculos": """
SELECT c.id, c.recebido_em, c.origem, c.status, c.setor, c.confianca, c.anos_experiencia,
       k.nome AS candidato, k.email, k.cidade,
       round(c.tamanho / 1024.0, 1) AS tamanho_kb, (c.caminho LIKE 'banco:%') AS arquivo_no_banco,
       (SELECT count(*) FROM candidaturas a WHERE a.curriculo_id = c.id) AS vagas,
       (SELECT min(CASE an.compatibilidade WHEN 'ALTA' THEN 1 WHEN 'MEDIA' THEN 2 ELSE 3 END) FROM analises an
          JOIN candidaturas a ON a.id = an.candidatura_id WHERE a.curriculo_id = c.id) AS melhor_compatibilidade,
       (SELECT d.decisao FROM decisoes d JOIN candidaturas a ON a.id = d.candidatura_id
          WHERE a.curriculo_id = c.id ORDER BY d.id DESC LIMIT 1) AS ultima_decisao
FROM curriculos c LEFT JOIN candidatos k ON k.id = c.candidato_id""",
    "v_vagas": """
SELECT v.id, v.titulo, v.status, v.area, v.local, v.criado_em,
       count(a.id) AS recebidos,
       count(a.id) FILTER (WHERE a.status = 'analisada') AS analisados,
       count(DISTINCT an.candidatura_id) FILTER (WHERE an.recomendado) AS recomendados,
       count(DISTINCT d.candidatura_id) FILTER (WHERE d.decisao = 'entrevista') AS chamados_para_entrevista
FROM vagas v
LEFT JOIN candidaturas a ON a.vaga_id = v.id
LEFT JOIN analises an ON an.candidatura_id = a.id
LEFT JOIN decisoes d ON d.candidatura_id = a.id
GROUP BY v.id""",
    "v_fila": """
SELECT tipo, status, count(*) AS tarefas, min(criada_em) AS mais_antiga,
       max(tentativas) AS max_tentativas
FROM tarefas WHERE status IN ('pendente', 'executando', 'erro')
GROUP BY tipo, status""",
    "v_uso_ia_diario": """
SELECT (ts AT TIME ZONE 'UTC' AT TIME ZONE 'America/Sao_Paulo')::date AS dia, finalidade, modelo,
       count(*) AS chamadas, sum(entrada + saida + pensamento) AS tokens, round(sum(custo_usd)::numeric, 4) AS custo_usd
FROM uso_ia GROUP BY 1, 2, 3""",
    "v_saude": """
SELECT pg_size_pretty(pg_database_size(current_database())) AS banco,
       pg_size_pretty(pg_total_relation_size('arquivos_curriculo')) AS arquivos_dos_curriculos,
       (SELECT count(*) FROM curriculos WHERE status <> 'anonimizado') AS curriculos_ativos,
       (SELECT count(*) FROM candidatos WHERE anonimizado_em IS NULL) AS candidatos_ativos,
       (SELECT count(*) FROM tarefas WHERE status IN ('pendente', 'executando')) AS fila,
       (SELECT count(*) FROM tarefas WHERE status = 'erro') AS tarefas_com_erro,
       (SELECT count(*) FROM pendencias WHERE status = 'aberta') AS pendencias_abertas,
       (SELECT max(recebido_em) FROM curriculos) AS ultimo_curriculo""",
}


def estado(con) -> dict:
    """O que já existe no schema atual, para só mexer (e travar tabelas) no que falta."""
    q = lambda sql: set(con.execute(text(sql)).scalars())  # noqa: E731
    return {
        "schema": con.execute(text("SELECT current_schema()")).scalar(),
        "sem_rls": q("SELECT relname FROM pg_class WHERE relnamespace = current_schema()::regnamespace "
                     "AND relkind = 'r' AND NOT relrowsecurity"),
        "gatilhos": q("SELECT c.relname || '.' || t.tgname FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid "
                      "WHERE c.relnamespace = current_schema()::regnamespace AND NOT t.tgisinternal"),
        "regras": q("SELECT conname FROM pg_constraint WHERE connamespace = current_schema()::regnamespace AND contype = 'c'"),
        "regras_pendentes": q("SELECT conname FROM pg_constraint WHERE connamespace = current_schema()::regnamespace "
                              "AND contype = 'c' AND NOT convalidated"),
        "objetos": q("SELECT relname FROM pg_class WHERE relnamespace = current_schema()::regnamespace"),
        "externo": con.execute(text("SELECT attstorage = 'e' FROM pg_attribute WHERE attname = 'conteudo' "
                                    "AND attrelid = to_regclass('arquivos_curriculo')")).scalar(),
    }


def passos(tabelas, e: dict, script: bool = False) -> list[tuple[str, str]]:
    """Comandos que faltam, em ordem. `tabelas` são as Table do SQLAlchemy; `e` vem de estado().
    Com `script`, as regras saem numa forma que pode ser rodada de novo à mão."""
    s = '"%s"' % e["schema"].replace('"', '')
    p = [(f"RLS em {t.name}", f'ALTER TABLE {s}."{t.name}" ENABLE ROW LEVEL SECURITY')
         for t in tabelas if t.name in e["sem_rls"]]
    for t in tabelas:  # regras novas em bancos antigos: NOT VALID vale já para o que chega; VALIDATE confere o passado
        for ck in sorted((c for c in t.constraints if isinstance(c, CheckConstraint)), key=lambda c: c.name):
            if ck.name not in e["regras"]:
                if script:
                    p.append((f"regra {ck.name}", f'ALTER TABLE {s}."{t.name}" DROP CONSTRAINT IF EXISTS {ck.name}, '
                                                  f'ADD CONSTRAINT {ck.name} CHECK ({ck.sqltext})'))
                    continue
                p.append((f"regra {ck.name}", f'ALTER TABLE {s}."{t.name}" ADD CONSTRAINT {ck.name} CHECK ({ck.sqltext}) NOT VALID'))
            if ck.name not in e["regras"] or ck.name in e["regras_pendentes"]:  # tenta de novo a cada início, até os dados serem corrigidos
                p.append((f"regra {ck.name} (dados antigos)", f'ALTER TABLE {s}."{t.name}" VALIDATE CONSTRAINT {ck.name}'))
    if e["externo"] is False:
        p.append(("armazenamento dos arquivos", f"ALTER TABLE {s}.arquivos_curriculo ALTER COLUMN conteudo SET STORAGE EXTERNAL"))
    p.append(("funções", FUNCOES.format(s=s)))
    for t in PROTEGIDAS:
        if f"{t}.impedir_exclusao" not in e["gatilhos"]:
            p.append((f"proteção de {t}", f"CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON {s}.{t} "
                                          f"FOR EACH ROW EXECUTE FUNCTION {s}.impedir_exclusao()"))
        if f"{t}.impedir_truncate" not in e["gatilhos"]:
            p.append((f"proteção de {t} (truncate)", f"CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON {s}.{t} "
                                                     f"FOR EACH STATEMENT EXECUTE FUNCTION {s}.impedir_exclusao()"))
    if "auditoria.somente_inclusao" not in e["gatilhos"]:
        p.append(("auditoria somente-inclusão", f"CREATE OR REPLACE TRIGGER somente_inclusao BEFORE UPDATE ON {s}.auditoria "
                                               f"FOR EACH ROW EXECUTE FUNCTION {s}.auditoria_somente_inclusao()"))
    if "ix_curriculos_busca" not in e["objetos"]:
        p.append(("busca nos currículos", f"CREATE INDEX IF NOT EXISTS ix_curriculos_busca ON {s}.curriculos "
                                          "USING gin (to_tsvector('portuguese', coalesce(texto, '')))"))
    for nome, corpo in VISOES.items():
        p.append((f"visão {nome}", f"CREATE OR REPLACE VIEW {s}.{nome} WITH (security_invoker = true) AS {corpo}"))
    return p


def ajustar(eng, tabelas):
    """Aplica o que falta; um passo que falhar (ex.: sem permissão) é registrado e não impede o sistema de subir."""
    with eng.connect() as con:
        e = estado(con)
    s = '"%s"' % e["schema"].replace('"', '')
    lista, falhas = passos(tabelas, e), 0
    for rotulo, sql in lista:
        try:
            with eng.begin() as con:
                con.execute(text(sql))
        except Exception as erro:
            if rotulo.startswith("visão"):  # visão mudou de colunas: recria
                try:
                    with eng.begin() as con:
                        con.execute(text(f"DROP VIEW IF EXISTS {s}.{rotulo.split()[-1]}"))
                        con.execute(text(sql))
                    continue
                except Exception as e2:
                    erro = e2
            falhas += 1
            log.warning("Banco: %s não aplicado: %s", rotulo, str(erro).splitlines()[0][:160])
    log.info("Banco PostgreSQL (schema %s): %d ajuste(s) aplicado(s)%s", e["schema"], len(lista) - falhas,
             f", {falhas} com falha" if falhas else "; tudo em dia")


def script(tabelas, schema: str = "hub") -> str:
    """O mesmo que ajustar() faz num banco recém-criado, em SQL para o SQL Editor do Supabase."""
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex
    e = {"schema": schema, "sem_rls": {t.name for t in tabelas}, "gatilhos": set(), "regras": set(),
         "regras_pendentes": set(), "objetos": set(), "externo": False}
    indices = [(f"índice {i.name}", str(CreateIndex(i, if_not_exists=True).compile(dialect=postgresql.dialect())))
               for t in tabelas for i in sorted(t.indexes, key=lambda i: i.name)]
    corpo = "\n\n".join(f"-- {rotulo}\n{sql.strip().rstrip(';')};" for rotulo, sql in indices + passos(tabelas, e, script=True))
    return (f"-- Reforço do banco do escritório de recrutamento (schema {schema}).\n"
            "-- Gerado por: cd backend && python -m app.banco_pg > ../supabase/reforco.sql\n"
            "-- O sistema aplica tudo isto sozinho a cada início; este arquivo serve para aplicar ou conferir pelo\n"
            "-- SQL Editor do Supabase (rode como postgres: os objetos ficam com o dono certo, hub_app). Pode ser repetido.\n"
            f"begin;\nset local role hub_app;\nset local search_path = {schema};\n\n{corpo}\n\ncommit;\n")


if __name__ == "__main__":
    from .models import Base  # importar os modelos registra as tabelas
    print(script(Base.metadata.sorted_tables), end="")
