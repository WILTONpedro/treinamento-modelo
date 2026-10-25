#!/usr/bin/env bash
# Cópia de segurança do banco do escritório (o schema da empresa: hub, ou o de DB_SCHEMA): dados, regras, gatilhos,
# visões e arquivos cifrados. Num projeto com várias empresas, cada uma tem o seu schema: a cópia leva só o dela.
#
# Uso:   supabase/backup.sh [pasta]     (padrão: ./backups; lê DATABASE_URL e DB_SCHEMA do ambiente, do .env ou do
#                                        backend/.env; ex.: DB_SCHEMA=rh_padaria supabase/backup.sh backups/padaria)
#        agendado todo dia às 3h (crontab -e):  0 3 * * * /caminho/do/projeto/supabase/backup.sh /caminho/backups
# Volta: pg_restore --no-owner --role=<papel> -d "<url de um banco preparado com o 1-banco.sql da empresa>" <arquivo>.dump
#        (o papel é o da empresa: hub_app, ou hub_<empresa> no projeto compartilhado; o único erro esperado é no
#        "CREATE SCHEMA", que o 1-banco.sql já criou: "already exists" ou "permission denied for database")
#
# Precisa de pg_dump 17 ou mais novo (o Supabase usa PostgreSQL 17); sem ele, usa o Docker.
# A cópia tem dados pessoais (o texto dos currículos fica legível): guarde-a com o mesmo cuidado que o banco.
set -euo pipefail

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${DATABASE_URL:-}" ]; then  # sem a variável, usa a do sistema (.env do Docker ou backend/.env)
  for arq in "$RAIZ/.env" "$RAIZ/backend/.env"; do
    [ -f "$arq" ] && DATABASE_URL=$(grep -m1 '^DATABASE_URL=' "$arq" | cut -d= -f2- | tr -d '"'"'"'\r') && [ -n "$DATABASE_URL" ] && break
  done
fi
: "${DATABASE_URL:?defina DATABASE_URL (a mesma do sistema) ou rode a partir do projeto com o .env preenchido}"
if [ -z "${DB_SCHEMA:-}" ]; then
  for arq in "$RAIZ/.env" "$RAIZ/backend/.env"; do
    [ -f "$arq" ] && DB_SCHEMA=$(grep -m1 '^DB_SCHEMA=' "$arq" | cut -d= -f2- | tr -d '"'"'"'\r') && [ -n "$DB_SCHEMA" ] && break
  done
fi
SCHEMA="${DB_SCHEMA:-hub}"
[[ "$SCHEMA" =~ ^[a-z_][a-z0-9_]*$ ]] || { echo "DB_SCHEMA inválido: $SCHEMA" >&2; exit 1; }
URL="${DATABASE_URL/postgresql+psycopg:/postgresql:}"
URL="${URL/.pooler.supabase.com:6543\//.pooler.supabase.com:5432/}"  # o pg_dump precisa do pooler em modo sessão
PASTA="${1:-backups}"
GUARDAR_DIAS="${GUARDAR_DIAS:-30}"
IMAGEM="${PG_IMAGEM:-postgres:17-alpine}"

mkdir -p "$PASTA"
chmod 700 "$PASTA"
NOME="$SCHEMA-$(date +%Y-%m-%d_%H%M).dump"
umask 077

versao=$(pg_dump --version 2>/dev/null | grep -oE '[0-9]+' | head -1 || true)
copiar() {  # $1 = url do banco
  if [ "${versao:-0}" -ge 17 ] && [ -z "${PG_IMAGEM:-}" ]; then
    pg_dump "$1" --schema="$SCHEMA" --format=custom --no-owner --file="$PASTA/$NOME"
  else
    docker run --rm --network host -e PGURL="$1" -v "$(cd "$PASTA" && pwd)":/saida "$IMAGEM" \
      sh -c "pg_dump \"\$PGURL\" --schema=$SCHEMA --format=custom --no-owner --file=/saida/$NOME"
  fi
}
if ! copiar "$URL" 2>"$PASTA/.erro"; then
  # o pooler do Supabase fica em aws-0-… ou aws-1-…, conforme o projeto: se este não conhece o projeto, tenta o outro
  if grep -qi "tenant or user not found" "$PASTA/.erro" && [[ "$URL" =~ @aws-([01])- ]]; then
    n="${BASH_REMATCH[1]}"
    URL="${URL/@aws-$n-/@aws-$((1 - n))-}"
    copiar "$URL"
  else
    cat "$PASTA/.erro" >&2
    rm -f "$PASTA/.erro"
    exit 1
  fi
fi
rm -f "$PASTA/.erro"
chmod 600 "$PASTA/$NOME"

# confere se a cópia abre e tem as tabelas principais antes de considerar o backup feito
conteudo=$( (pg_restore --list "$PASTA/$NOME" 2>/dev/null) || docker run --rm -v "$(cd "$PASTA" && pwd)":/saida "$IMAGEM" pg_restore --list "/saida/$NOME")
for t in curriculos candidatos arquivos_curriculo auditoria; do
  grep -q "TABLE DATA $SCHEMA $t " <<<"$conteudo" || { echo "Cópia incompleta: falta a tabela $t" >&2; exit 1; }
done

find "$PASTA" -name "$SCHEMA-*.dump" -mtime +"$GUARDAR_DIAS" -delete
echo "Cópia salva em $PASTA/$NOME ($(du -h "$PASTA/$NOME" | cut -f1)); cópias com mais de $GUARDAR_DIAS dias removidas."
