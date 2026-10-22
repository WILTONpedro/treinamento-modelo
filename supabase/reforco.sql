-- Reforço do banco do escritório de recrutamento (schema hub).
-- Gerado por: cd backend && python -m app.banco_pg > ../supabase/reforco.sql
-- O sistema aplica tudo isto sozinho a cada início; este arquivo serve para aplicar ou conferir pelo
-- SQL Editor do Supabase (rode como postgres: os objetos ficam com o dono certo, hub_app). Pode ser repetido.
begin;
set local role hub_app;
set local search_path = hub;

-- índice ix_candidatos_email
CREATE INDEX IF NOT EXISTS ix_candidatos_email ON candidatos (email);

-- índice uq_candidatos_email_ativo
CREATE UNIQUE INDEX IF NOT EXISTS uq_candidatos_email_ativo ON candidatos (email) WHERE email IS NOT NULL AND anonimizado_em IS NULL;

-- índice ix_uso_ia_ts
CREATE INDEX IF NOT EXISTS ix_uso_ia_ts ON uso_ia (ts);

-- índice ix_agentes_especialidade_id
CREATE INDEX IF NOT EXISTS ix_agentes_especialidade_id ON agentes (especialidade_id);

-- índice ix_auditoria_ts
CREATE INDEX IF NOT EXISTS ix_auditoria_ts ON auditoria (ts);

-- índice ix_auditoria_usuario_id
CREATE INDEX IF NOT EXISTS ix_auditoria_usuario_id ON auditoria (usuario_id);

-- índice ix_curriculos_candidato_id
CREATE INDEX IF NOT EXISTS ix_curriculos_candidato_id ON curriculos (candidato_id);

-- índice ix_solicitacoes_titular_candidato_id
CREATE INDEX IF NOT EXISTS ix_solicitacoes_titular_candidato_id ON solicitacoes_titular (candidato_id);

-- índice ix_solicitacoes_titular_usuario_id
CREATE INDEX IF NOT EXISTS ix_solicitacoes_titular_usuario_id ON solicitacoes_titular (usuario_id);

-- índice ix_candidaturas_candidato_id
CREATE INDEX IF NOT EXISTS ix_candidaturas_candidato_id ON candidaturas (candidato_id);

-- índice ix_candidaturas_vaga_id
CREATE INDEX IF NOT EXISTS ix_candidaturas_vaga_id ON candidaturas (vaga_id);

-- índice ix_contratacoes_agente_id
CREATE INDEX IF NOT EXISTS ix_contratacoes_agente_id ON contratacoes (agente_id);

-- índice ix_contratacoes_criado_em
CREATE INDEX IF NOT EXISTS ix_contratacoes_criado_em ON contratacoes (criado_em);

-- índice ix_contratacoes_especialidade_id
CREATE INDEX IF NOT EXISTS ix_contratacoes_especialidade_id ON contratacoes (especialidade_id);

-- índice ix_eventos_agente_id
CREATE INDEX IF NOT EXISTS ix_eventos_agente_id ON eventos (agente_id);

-- índice ix_eventos_candidato_id
CREATE INDEX IF NOT EXISTS ix_eventos_candidato_id ON eventos (candidato_id);

-- índice ix_eventos_ts
CREATE INDEX IF NOT EXISTS ix_eventos_ts ON eventos (ts);

-- índice ix_vaga_agentes_agente_id
CREATE INDEX IF NOT EXISTS ix_vaga_agentes_agente_id ON vaga_agentes (agente_id);

-- índice ix_analises_agente_id
CREATE INDEX IF NOT EXISTS ix_analises_agente_id ON analises (agente_id);

-- índice ix_decisoes_candidatura_id
CREATE INDEX IF NOT EXISTS ix_decisoes_candidatura_id ON decisoes (candidatura_id);

-- índice ix_decisoes_usuario_id
CREATE INDEX IF NOT EXISTS ix_decisoes_usuario_id ON decisoes (usuario_id);

-- índice ix_pendencias_agente_id
CREATE INDEX IF NOT EXISTS ix_pendencias_agente_id ON pendencias (agente_id);

-- índice ix_pendencias_candidatura_id
CREATE INDEX IF NOT EXISTS ix_pendencias_candidatura_id ON pendencias (candidatura_id);

-- índice ix_pendencias_curriculo_id
CREATE INDEX IF NOT EXISTS ix_pendencias_curriculo_id ON pendencias (curriculo_id);

-- índice ix_pendencias_especialidade_id
CREATE INDEX IF NOT EXISTS ix_pendencias_especialidade_id ON pendencias (especialidade_id);

-- índice ix_pendencias_resolvido_por_id
CREATE INDEX IF NOT EXISTS ix_pendencias_resolvido_por_id ON pendencias (resolvido_por_id);

-- índice ix_pendencias_vaga_id
CREATE INDEX IF NOT EXISTS ix_pendencias_vaga_id ON pendencias (vaga_id);

-- índice ix_tarefa_fila
CREATE INDEX IF NOT EXISTS ix_tarefa_fila ON tarefas (status, tipo, agente_id);

-- índice ix_tarefas_agente_id
CREATE INDEX IF NOT EXISTS ix_tarefas_agente_id ON tarefas (agente_id);

-- índice ix_tarefas_candidatura_id
CREATE INDEX IF NOT EXISTS ix_tarefas_candidatura_id ON tarefas (candidatura_id);

-- índice ix_tarefas_curriculo_id
CREATE INDEX IF NOT EXISTS ix_tarefas_curriculo_id ON tarefas (curriculo_id);

-- RLS em arquivos_curriculo
ALTER TABLE "hub"."arquivos_curriculo" ENABLE ROW LEVEL SECURITY;

-- RLS em candidatos
ALTER TABLE "hub"."candidatos" ENABLE ROW LEVEL SECURITY;

-- RLS em comando
ALTER TABLE "hub"."comando" ENABLE ROW LEVEL SECURITY;

-- RLS em configuracoes
ALTER TABLE "hub"."configuracoes" ENABLE ROW LEVEL SECURITY;

-- RLS em especialidades
ALTER TABLE "hub"."especialidades" ENABLE ROW LEVEL SECURITY;

-- RLS em uso_ia
ALTER TABLE "hub"."uso_ia" ENABLE ROW LEVEL SECURITY;

-- RLS em usuarios
ALTER TABLE "hub"."usuarios" ENABLE ROW LEVEL SECURITY;

-- RLS em vagas
ALTER TABLE "hub"."vagas" ENABLE ROW LEVEL SECURITY;

-- RLS em agentes
ALTER TABLE "hub"."agentes" ENABLE ROW LEVEL SECURITY;

-- RLS em auditoria
ALTER TABLE "hub"."auditoria" ENABLE ROW LEVEL SECURITY;

-- RLS em curriculos
ALTER TABLE "hub"."curriculos" ENABLE ROW LEVEL SECURITY;

-- RLS em solicitacoes_titular
ALTER TABLE "hub"."solicitacoes_titular" ENABLE ROW LEVEL SECURITY;

-- RLS em candidaturas
ALTER TABLE "hub"."candidaturas" ENABLE ROW LEVEL SECURITY;

-- RLS em contratacoes
ALTER TABLE "hub"."contratacoes" ENABLE ROW LEVEL SECURITY;

-- RLS em eventos
ALTER TABLE "hub"."eventos" ENABLE ROW LEVEL SECURITY;

-- RLS em vaga_agentes
ALTER TABLE "hub"."vaga_agentes" ENABLE ROW LEVEL SECURITY;

-- RLS em analises
ALTER TABLE "hub"."analises" ENABLE ROW LEVEL SECURITY;

-- RLS em decisoes
ALTER TABLE "hub"."decisoes" ENABLE ROW LEVEL SECURITY;

-- RLS em pendencias
ALTER TABLE "hub"."pendencias" ENABLE ROW LEVEL SECURITY;

-- RLS em tarefas
ALTER TABLE "hub"."tarefas" ENABLE ROW LEVEL SECURITY;

-- regra ck_arquivos_curriculo_tamanho
ALTER TABLE "hub"."arquivos_curriculo" DROP CONSTRAINT IF EXISTS ck_arquivos_curriculo_tamanho, ADD CONSTRAINT ck_arquivos_curriculo_tamanho CHECK (length(conteudo) <= 52428800);

-- regra ck_uso_ia_cache
ALTER TABLE "hub"."uso_ia" DROP CONSTRAINT IF EXISTS ck_uso_ia_cache, ADD CONSTRAINT ck_uso_ia_cache CHECK (cache >= 0);

-- regra ck_uso_ia_custo_usd
ALTER TABLE "hub"."uso_ia" DROP CONSTRAINT IF EXISTS ck_uso_ia_custo_usd, ADD CONSTRAINT ck_uso_ia_custo_usd CHECK (custo_usd >= 0);

-- regra ck_uso_ia_entrada
ALTER TABLE "hub"."uso_ia" DROP CONSTRAINT IF EXISTS ck_uso_ia_entrada, ADD CONSTRAINT ck_uso_ia_entrada CHECK (entrada >= 0);

-- regra ck_uso_ia_pensamento
ALTER TABLE "hub"."uso_ia" DROP CONSTRAINT IF EXISTS ck_uso_ia_pensamento, ADD CONSTRAINT ck_uso_ia_pensamento CHECK (pensamento >= 0);

-- regra ck_uso_ia_saida
ALTER TABLE "hub"."uso_ia" DROP CONSTRAINT IF EXISTS ck_uso_ia_saida, ADD CONSTRAINT ck_uso_ia_saida CHECK (saida >= 0);

-- regra ck_usuarios_papel
ALTER TABLE "hub"."usuarios" DROP CONSTRAINT IF EXISTS ck_usuarios_papel, ADD CONSTRAINT ck_usuarios_papel CHECK (papel IN ('recrutador', 'supervisor', 'admin'));

-- regra ck_vagas_status
ALTER TABLE "hub"."vagas" DROP CONSTRAINT IF EXISTS ck_vagas_status, ADD CONSTRAINT ck_vagas_status CHECK (status IN ('aberta', 'suspensa', 'encerrada', 'cancelada'));

-- regra ck_agentes_funcao
ALTER TABLE "hub"."agentes" DROP CONSTRAINT IF EXISTS ck_agentes_funcao, ADD CONSTRAINT ck_agentes_funcao CHECK (funcao IN ('supervisor', 'recepcao', 'analista'));

-- regra ck_curriculos_origem
ALTER TABLE "hub"."curriculos" DROP CONSTRAINT IF EXISTS ck_curriculos_origem, ADD CONSTRAINT ck_curriculos_origem CHECK (origem IN ('email', 'upload', 'imap', 'demo'));

-- regra ck_curriculos_status
ALTER TABLE "hub"."curriculos" DROP CONSTRAINT IF EXISTS ck_curriculos_status, ADD CONSTRAINT ck_curriculos_status CHECK (status IN ('recebido', 'triado', 'invalido', 'erro', 'anonimizado'));

-- regra ck_curriculos_tamanho
ALTER TABLE "hub"."curriculos" DROP CONSTRAINT IF EXISTS ck_curriculos_tamanho, ADD CONSTRAINT ck_curriculos_tamanho CHECK (tamanho >= 0);

-- regra ck_solicitacoes_titular_tipo
ALTER TABLE "hub"."solicitacoes_titular" DROP CONSTRAINT IF EXISTS ck_solicitacoes_titular_tipo, ADD CONSTRAINT ck_solicitacoes_titular_tipo CHECK (tipo IN ('acesso', 'correcao', 'eliminacao'));

-- regra ck_candidaturas_origem
ALTER TABLE "hub"."candidaturas" DROP CONSTRAINT IF EXISTS ck_candidaturas_origem, ADD CONSTRAINT ck_candidaturas_origem CHECK (origem IN ('triagem', 'banco', 'atlas', 'indicacao') OR origem IS NULL);

-- regra ck_candidaturas_status
ALTER TABLE "hub"."candidaturas" DROP CONSTRAINT IF EXISTS ck_candidaturas_status, ADD CONSTRAINT ck_candidaturas_status CHECK (status IN ('na_fila', 'em_analise', 'analisada', 'erro'));

-- regra ck_contratacoes_acao
ALTER TABLE "hub"."contratacoes" DROP CONSTRAINT IF EXISTS ck_contratacoes_acao, ADD CONSTRAINT ck_contratacoes_acao CHECK (acao IN ('contratacao', 'retorno', 'dispensa', 'sem_estacao'));

-- regra ck_analises_compatibilidade
ALTER TABLE "hub"."analises" DROP CONSTRAINT IF EXISTS ck_analises_compatibilidade, ADD CONSTRAINT ck_analises_compatibilidade CHECK (compatibilidade IN ('ALTA', 'MEDIA', 'BAIXA'));

-- regra ck_analises_duracao_ms
ALTER TABLE "hub"."analises" DROP CONSTRAINT IF EXISTS ck_analises_duracao_ms, ADD CONSTRAINT ck_analises_duracao_ms CHECK (duracao_ms >= 0);

-- regra ck_analises_tipo
ALTER TABLE "hub"."analises" DROP CONSTRAINT IF EXISTS ck_analises_tipo, ADD CONSTRAINT ck_analises_tipo CHECK (tipo IN ('principal', 'segunda_opiniao'));

-- regra ck_decisoes_decisao
ALTER TABLE "hub"."decisoes" DROP CONSTRAINT IF EXISTS ck_decisoes_decisao, ADD CONSTRAINT ck_decisoes_decisao CHECK (decisao IN ('entrevista', 'banco_talentos', 'descartado'));

-- regra ck_pendencias_status
ALTER TABLE "hub"."pendencias" DROP CONSTRAINT IF EXISTS ck_pendencias_status, ADD CONSTRAINT ck_pendencias_status CHECK (status IN ('aberta', 'resolvida'));

-- regra ck_tarefas_status
ALTER TABLE "hub"."tarefas" DROP CONSTRAINT IF EXISTS ck_tarefas_status, ADD CONSTRAINT ck_tarefas_status CHECK (status IN ('pendente', 'executando', 'concluida', 'erro', 'cancelada'));

-- regra ck_tarefas_tentativas
ALTER TABLE "hub"."tarefas" DROP CONSTRAINT IF EXISTS ck_tarefas_tentativas, ADD CONSTRAINT ck_tarefas_tentativas CHECK (tentativas >= 0);

-- regra ck_tarefas_tipo
ALTER TABLE "hub"."tarefas" DROP CONSTRAINT IF EXISTS ck_tarefas_tipo, ADD CONSTRAINT ck_tarefas_tipo CHECK (tipo IN ('TRIAGEM', 'DISTRIBUICAO', 'ANALISE', 'CONSULTA', 'REVISAO'));

-- armazenamento dos arquivos
ALTER TABLE "hub".arquivos_curriculo ALTER COLUMN conteudo SET STORAGE EXTERNAL;

-- funções
CREATE OR REPLACE FUNCTION "hub".impedir_exclusao() RETURNS trigger LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF current_setting('hub.permitir_exclusao', true) = 'on' THEN
    RETURN CASE WHEN TG_LEVEL = 'ROW' THEN OLD END;
  END IF;
  RAISE EXCEPTION 'Exclusão bloqueada em %: dados de candidatos são anonimizados (LGPD), não apagados', TG_TABLE_NAME
    USING ERRCODE = 'insufficient_privilege',
          HINT = 'Em manutenção planejada: SET hub.permitir_exclusao = on; na mesma sessão.';
END $$;

CREATE OR REPLACE FUNCTION "hub".auditoria_somente_inclusao() RETURNS trigger LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  IF current_setting('hub.permitir_exclusao', true) = 'on' THEN
    RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
  END IF;
  RAISE EXCEPTION 'A auditoria só aceita inclusões (art. 37 da LGPD)' USING ERRCODE = 'insufficient_privilege';
END $$;

CREATE OR REPLACE FUNCTION "hub".buscar_curriculos(termo text, limite integer DEFAULT 20)
RETURNS TABLE (curriculo_id integer, candidato text, setor text, recebido_em timestamp, relevancia real, trecho text)
LANGUAGE sql STABLE SET search_path = "hub", pg_catalog AS $$
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

-- proteção de candidatos
CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON "hub".candidatos FOR EACH ROW EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de candidatos (truncate)
CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON "hub".candidatos FOR EACH STATEMENT EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de curriculos
CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON "hub".curriculos FOR EACH ROW EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de curriculos (truncate)
CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON "hub".curriculos FOR EACH STATEMENT EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de candidaturas
CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON "hub".candidaturas FOR EACH ROW EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de candidaturas (truncate)
CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON "hub".candidaturas FOR EACH STATEMENT EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de analises
CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON "hub".analises FOR EACH ROW EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de analises (truncate)
CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON "hub".analises FOR EACH STATEMENT EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de decisoes
CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON "hub".decisoes FOR EACH ROW EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de decisoes (truncate)
CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON "hub".decisoes FOR EACH STATEMENT EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de vagas
CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON "hub".vagas FOR EACH ROW EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de vagas (truncate)
CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON "hub".vagas FOR EACH STATEMENT EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de usuarios
CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON "hub".usuarios FOR EACH ROW EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de usuarios (truncate)
CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON "hub".usuarios FOR EACH STATEMENT EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de solicitacoes_titular
CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON "hub".solicitacoes_titular FOR EACH ROW EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de solicitacoes_titular (truncate)
CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON "hub".solicitacoes_titular FOR EACH STATEMENT EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de auditoria
CREATE OR REPLACE TRIGGER impedir_exclusao BEFORE DELETE ON "hub".auditoria FOR EACH ROW EXECUTE FUNCTION "hub".impedir_exclusao();

-- proteção de auditoria (truncate)
CREATE OR REPLACE TRIGGER impedir_truncate BEFORE TRUNCATE ON "hub".auditoria FOR EACH STATEMENT EXECUTE FUNCTION "hub".impedir_exclusao();

-- auditoria somente-inclusão
CREATE OR REPLACE TRIGGER somente_inclusao BEFORE UPDATE ON "hub".auditoria FOR EACH ROW EXECUTE FUNCTION "hub".auditoria_somente_inclusao();

-- busca nos currículos
CREATE INDEX IF NOT EXISTS ix_curriculos_busca ON "hub".curriculos USING gin (to_tsvector('portuguese', coalesce(texto, '')));

-- visão v_curriculos
CREATE OR REPLACE VIEW "hub".v_curriculos WITH (security_invoker = true) AS 
SELECT c.id, c.recebido_em, c.origem, c.status, c.setor, c.confianca, c.anos_experiencia,
       k.nome AS candidato, k.email, k.cidade,
       round(c.tamanho / 1024.0, 1) AS tamanho_kb, (c.caminho LIKE 'banco:%') AS arquivo_no_banco,
       (SELECT count(*) FROM candidaturas a WHERE a.curriculo_id = c.id) AS vagas,
       (SELECT min(CASE an.compatibilidade WHEN 'ALTA' THEN 1 WHEN 'MEDIA' THEN 2 ELSE 3 END) FROM analises an
          JOIN candidaturas a ON a.id = an.candidatura_id WHERE a.curriculo_id = c.id) AS melhor_compatibilidade,
       (SELECT d.decisao FROM decisoes d JOIN candidaturas a ON a.id = d.candidatura_id
          WHERE a.curriculo_id = c.id ORDER BY d.id DESC LIMIT 1) AS ultima_decisao
FROM curriculos c LEFT JOIN candidatos k ON k.id = c.candidato_id;

-- visão v_vagas
CREATE OR REPLACE VIEW "hub".v_vagas WITH (security_invoker = true) AS 
SELECT v.id, v.titulo, v.status, v.area, v.local, v.criado_em,
       count(a.id) AS recebidos,
       count(a.id) FILTER (WHERE a.status = 'analisada') AS analisados,
       count(DISTINCT an.candidatura_id) FILTER (WHERE an.recomendado) AS recomendados,
       count(DISTINCT d.candidatura_id) FILTER (WHERE d.decisao = 'entrevista') AS chamados_para_entrevista
FROM vagas v
LEFT JOIN candidaturas a ON a.vaga_id = v.id
LEFT JOIN analises an ON an.candidatura_id = a.id
LEFT JOIN decisoes d ON d.candidatura_id = a.id
GROUP BY v.id;

-- visão v_fila
CREATE OR REPLACE VIEW "hub".v_fila WITH (security_invoker = true) AS 
SELECT tipo, status, count(*) AS tarefas, min(criada_em) AS mais_antiga,
       max(tentativas) AS max_tentativas
FROM tarefas WHERE status IN ('pendente', 'executando', 'erro')
GROUP BY tipo, status;

-- visão v_uso_ia_diario
CREATE OR REPLACE VIEW "hub".v_uso_ia_diario WITH (security_invoker = true) AS 
SELECT (ts AT TIME ZONE 'UTC' AT TIME ZONE 'America/Sao_Paulo')::date AS dia, finalidade, modelo,
       count(*) AS chamadas, sum(entrada + saida + pensamento) AS tokens, round(sum(custo_usd)::numeric, 4) AS custo_usd
FROM uso_ia GROUP BY 1, 2, 3;

-- visão v_saude
CREATE OR REPLACE VIEW "hub".v_saude WITH (security_invoker = true) AS 
SELECT pg_size_pretty(pg_database_size(current_database())) AS banco,
       pg_size_pretty(pg_total_relation_size('arquivos_curriculo')) AS arquivos_dos_curriculos,
       (SELECT count(*) FROM curriculos WHERE status <> 'anonimizado') AS curriculos_ativos,
       (SELECT count(*) FROM candidatos WHERE anonimizado_em IS NULL) AS candidatos_ativos,
       (SELECT count(*) FROM tarefas WHERE status IN ('pendente', 'executando')) AS fila,
       (SELECT count(*) FROM tarefas WHERE status = 'erro') AS tarefas_com_erro,
       (SELECT count(*) FROM pendencias WHERE status = 'aberta') AS pendencias_abertas,
       (SELECT max(recebido_em) FROM curriculos) AS ultimo_curriculo;

commit;
