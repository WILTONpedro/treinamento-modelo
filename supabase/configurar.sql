-- Prepara um projeto do Supabase para o escritório de recrutamento.
-- Rode uma vez no SQL Editor do projeto (Supabase → SQL Editor → New query).
--
-- Cria um papel só do sistema (hub_app), dono de um schema próprio (hub) que fica fora da API pública
-- do Supabase. Tabelas, índices, regras, proteções, busca e visões são criados pelo próprio sistema na
-- primeira inicialização (tudo, menos as tabelas, também está em reforco.sql, para conferir pelo SQL Editor).
--
-- 1) Troque a senha abaixo por uma senha longa e aleatória (gerenciador de senhas) e guarde-a.
create role hub_app with login password 'TROQUE-POR-UMA-SENHA-LONGA-E-ALEATORIA';
grant hub_app to postgres;                       -- o dono do projeto continua administrando pelo painel
create schema if not exists hub authorization hub_app;
alter role hub_app set search_path = hub;
revoke all on schema hub from public;

-- Limites de segurança: uma consulta presa, uma espera por bloqueio ou uma transação esquecida aberta
-- não seguram o banco (nem as outras chegadas de currículos) indefinidamente.
alter role hub_app set statement_timeout = '60s';
alter role hub_app set lock_timeout = '15s';
alter role hub_app set idle_in_transaction_session_timeout = '10min';

-- 2) DATABASE_URL do sistema: Supabase → Connect → Session pooler (porta 5432), trocando o usuário
--    "postgres.<ref>" por "hub_app.<ref>" e a senha pela do passo 1:
--    postgresql://hub_app.<ref>:<senha>@aws-0-<regiao>.pooler.supabase.com:5432/postgres
--
-- Para trocar a senha depois:  alter role hub_app password 'nova-senha';
