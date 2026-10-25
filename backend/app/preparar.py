"""Gera o arquivo de configuração de produção com segredos fortes.

    cd backend
    python -m app.preparar --docker --dominio rh.suaempresa.com.br --empresa "Sua Empresa" \\
        --admin-email rh@suaempresa.com.br --gemini-key SUA_CHAVE --supabase "postgresql://..."

--docker    escreve ../.env (lido pelo docker-compose: app + HTTPS automático)
sem ele     escreve ./.env (servidor simples, para rodar com uvicorn)
--supabase  banco no Supabase (URL do pooler, modo sessão): dados e arquivos dos currículos ficam lá
sem ele     com --docker, um PostgreSQL próprio no servidor; sem --docker, SQLite

--lacrar escritorio.env   gera escritorio.escritorio (cifrado) e mostra o código de ativação: é o que vai para a
                          empresa, sem nenhum arquivo legível (veja app/lacre.py)

Para vender a uma empresa nova, tudo de uma vez (banco, e-mail, configuração e pacote lacrado):

    python -m app.preparar --nova-empresa "Padaria Pão Dourado" --supabase "<URL do Connect, Transaction pooler>" \
        --gemini-key SUA_CHAVE --imap-usuario curriculos@padaria.com.br --imap-senha "senha de app"
    python -m app.preparar --nova-empresa              pergunta tudo, passo a passo (é o que o NovaEmpresa.exe faz)
    python -m app.preparar --trocar-senha empresas/padaria_pao_dourado    senha vazou: senha e pacote novos

Gera empresas/<nome>/ com 1-banco.sql (para o SQL Editor do Supabase), escritorio.env (legível: fica só com você),
<nome>.escritorio (vai para a empresa) e LEIA-ME.txt (passo a passo, suspender, reativar, backup).
--compartilhado põe a empresa num projeto que já atende outras (papel e schema próprios); sem ele, o projeto é
só dela (papel hub_app, schema hub, como em supabase/configurar.sql).

Nunca sobrescreve um .env existente (use --forcar). Mostra a senha inicial do administrador e a
chave da API de triagem uma única vez: guarde-as num cofre de senhas.
"""
import argparse
import getpass
import os
import re
import secrets
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

from cryptography.fernet import Fernet

BACKEND = Path(__file__).resolve().parent.parent


def montar(a: argparse.Namespace) -> tuple[str, dict]:
    segredos = {
        "SEGREDO_JWT": secrets.token_urlsafe(48),
        "CHAVE_ARQUIVOS": Fernet.generate_key().decode(),
        "TRIAGEM_API_KEY": secrets.token_urlsafe(32),
        "ADMIN_SENHA": secrets.token_urlsafe(12),
    }
    demo = getattr(a, "demonstracao", False)
    linhas = [
        "# Gerado por python -m app.preparar — NUNCA versione este arquivo. Guarde uma cópia num cofre de senhas:",
        "# sem CHAVE_ARQUIVOS os currículos cifrados não podem ser lidos.",
        "AMBIENTE=desenvolvimento" if demo else "AMBIENTE=producao",
        "",
        "# --- Identidade (APP_NOME vazio = sem nome na tela)",
        f"APP_NOME={a.nome}",
        "APP_SUBTITULO=Escritório de Recrutamento",
        f"EMPRESA_NOME={a.empresa}",
        "",
        "# --- IA (use o plano pago: no gratuito o Google pode usar o conteúdo para melhorar produtos)",
        *(["# Pacote de demonstração: análises por regras, sem custo de IA (troque por gemini para valer)",
           "IA_MODO=demo"] if demo else ["IA_MODO=gemini"]),
        f"GEMINI_API_KEY={a.gemini_key}",
        "GEMINI_MODELO=gemini-3.5-flash-lite",
        "GEMINI_PENSAMENTO=minimal",
        f"IA_LIMITE_DIARIO_USD={getattr(a, 'limite_diario', None) or 3}",
        "",
        "# --- Segurança",
        f"SEGREDO_JWT={segredos['SEGREDO_JWT']}",
        f"CHAVE_ARQUIVOS={segredos['CHAVE_ARQUIVOS']}",
    ]
    if a.local:
        linhas += [
            "# Cada computador roda a própria cópia, aberta só para ele e sem login (o recrutador é o usuário",
            "# do Windows). Use ESTE MESMO arquivo em todos os computadores: a CHAVE_ARQUIVOS precisa ser igual.",
            "ACESSO=local",
            "PAPEL_LOCAL=admin",
            "",
            "# --- Currículos chegando por e-mail (o programa busca na caixa; nada fica aberto na internet).",
            "# Com vários computadores, só o que comanda a equipe lê a caixa: pode deixar ligado em todos.",
            f"IMAP_ATIVO={'true' if getattr(a, 'imap_usuario', '') else 'false'}",
            f"IMAP_HOST={getattr(a, 'imap_host', '') or ''}",
            f"IMAP_USUARIO={getattr(a, 'imap_usuario', '') or ''}",
            f"IMAP_SENHA={getattr(a, 'imap_senha', '') or ''}",
            f"IMAP_PASTA={getattr(a, 'imap_pasta', '') or 'INBOX'}",
        ]
        segredos.pop("TRIAGEM_API_KEY")
        segredos.pop("ADMIN_SENHA")
    else:
        linhas += [
            f"TRIAGEM_API_KEY={segredos['TRIAGEM_API_KEY']}",
            "COOKIE_SECURE=true",
            f"ADMIN_EMAIL={a.admin_email}",
            f"ADMIN_SENHA={segredos['ADMIN_SENHA']}",
            "ADMIN_NOME=Administrador RH",
        ]
    linhas += [
        "",
        "# --- LGPD",
        "RETENCAO_DIAS=180",
        "RETENCAO_CONVERSAS_HORAS=72",
    ]
    if a.supabase:
        linhas += ["", f"# --- Banco no Supabase (schema {getattr(a, 'schema', '') or 'hub'}, fora da API pública; "
                   "arquivos cifrados no banco)",
                   f"DATABASE_URL={a.supabase}",
                   f"DB_SCHEMA={getattr(a, 'schema', '') or 'hub'}",
                   "ARMAZENAMENTO=banco"]
    if a.docker:
        linhas += ["", "# --- Docker (docker-compose.yml)", f"DOMINIO={a.dominio}", "PASTA_DADOS=/dados"]
        if not a.supabase:
            senha_banco = secrets.token_urlsafe(24)
            segredos["POSTGRES_PASSWORD"] = senha_banco
            linhas += ["# sobe o PostgreSQL junto com o app (sem esta linha: banco externo, ex. Supabase)",
                       "COMPOSE_PROFILES=banco-local",
                       f"POSTGRES_PASSWORD={senha_banco}",
                       f"DATABASE_URL=postgresql+psycopg://hub:{senha_banco}@db:5432/hub"]
    return "\n".join(linhas) + "\n", segredos


def lacrar(origem: Path, saida: Path | None, forcar: bool) -> int:
    from .lacre import lacrar as _lacrar
    if not origem.exists():
        print(f"{origem} não existe.", file=sys.stderr)
        return 1
    saida = saida or origem.with_suffix(".escritorio")
    if saida.exists() and not forcar:
        print(f"{saida} já existe; nada foi alterado (use --forcar para gerar outro).", file=sys.stderr)
        return 1
    pacote, codigo = _lacrar(origem.read_bytes())
    saida.write_bytes(pacote)
    print(f"Pacote lacrado: {saida}")
    print(f"  Código de ativação: {codigo}")
    print("  Mande o arquivo por um canal (e-mail) e o código por outro (WhatsApp, ligação). O código não fica")
    print("  guardado em lugar nenhum: anote-o agora ou gere outro pacote depois.")
    return 0


# --------------------------------------------------------------------------- empresa nova (o kit do fornecedor)
PROVEDORES_SEM_SENHA = ("outlook", "hotmail", "live.com", "office365", "microsoft")


def nome_curto(nome: str) -> str:
    """'Metalúrgica Aço Forte' -> 'metalurgica_aco_forte' (pasta, papel e schema no banco)."""
    s = unicodedata.normalize("NFD", nome).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:40] or "empresa"


_URL = re.compile(r"^(postgres(?:ql)?(?:\+psycopg)?)://(?:([^:@/\s]*)(?::[^@/\s]*)?@)?([^/?#@\s]+)(/[^?#\s]*)?(\?[^#\s]*)?$")


def url_da_empresa(url: str, papel: str, senha: str) -> str:
    """A URL copiada do Supabase (Connect → Transaction pooler, com [YOUR-PASSWORD]) passa a entrar com o papel da
    empresa e a senha nova: postgres.<ref> vira <papel>.<ref> (no Supabase, o pooler sabe o projeto pelo <ref>)."""
    m = _URL.match((url or "").strip())
    if not m:
        raise ValueError("O endereço do banco precisa começar com postgresql:// (Supabase → Connect → Transaction pooler).")
    usuario, host, caminho, consulta = m.group(2) or "postgres", m.group(3), m.group(4) or "/postgres", m.group(5) or ""
    ref = usuario.split(".", 1)[1] if "." in usuario else ""
    return f"postgresql://{papel}{'.' + ref if ref else ''}:{senha}@{host}{caminho}{consulta}"


def _sql_do_banco(empresa: str, papel: str, schema: str, senha: str, compartilhado: bool) -> str:
    quando = datetime.now().strftime("%d/%m/%Y %H:%M")
    return f"""-- {empresa}: banco do Escritório de Recrutamento (gerado em {quando} por app.preparar --nova-empresa).
-- Rode UMA vez no projeto do Supabase: SQL Editor → New query → cole tudo → Run.
-- {"Projeto compartilhado: esta empresa ganha papel e schema próprios; as outras do projeto não enxergam nada dela."
   if compartilhado else "Projeto só desta empresa."}
-- O papel {papel} é dono só do schema {schema}, que fica fora da API pública do Supabase.
do $$ begin
  if exists (select from pg_roles where rolname = '{papel}') then
    raise exception 'O papel {papel} já existe neste projeto (empresa já preparada?). Para trocar a senha, use --trocar-senha.';
  end if;
end $$;
create role {papel} with login password '{senha}';
grant {papel} to postgres;                       -- o dono do projeto continua administrando pelo painel
create schema {schema} authorization {papel};
alter role {papel} set search_path = {schema};
revoke all on schema {schema} from public;
-- limites: uma consulta presa ou uma transação esquecida aberta não seguram o banco
alter role {papel} set statement_timeout = '60s';
alter role {papel} set lock_timeout = '15s';
alter role {papel} set idle_in_transaction_session_timeout = '10min';
"""


def _leia_me(empresa: str, curto: str, papel: str, schema: str, compartilhado: bool, com_banco: bool, email: str,
             demo: bool) -> str:
    quando = datetime.now().strftime("%d/%m/%Y %H:%M")
    linhas = [
        f"{empresa} — Escritório de Recrutamento",
        f"Preparado em {quando}." + (f" Banco: {'projeto compartilhado' if compartilhado else 'projeto só dela'} "
                                       f"(papel {papel}, schema {schema})." if com_banco else
                                       " Sem Supabase: o banco fica num computador só."),
        "Pacote de DEMONSTRAÇÃO: análises por regras, sem custo de IA." if demo else "",
        "",
        "ARQUIVOS DESTA PASTA",
        *(["  1-banco.sql           prepara o banco da empresa (uma vez, no Supabase → SQL Editor)"] if com_banco else []),
        f"  {curto}.escritorio  configuração lacrada: é o que vai para a empresa (por e-mail)",
        "  escritorio.env        a mesma configuração, legível: NUNCA envie; guarde num cofre de senhas",
        "",
        "PASSO A PASSO",
        *([f"  1. Supabase → projeto → SQL Editor → New query → cole 1-banco.sql → Run."] if com_banco else []),
        f"  {2 if com_banco else 1}. Mande {curto}.escritorio por e-mail e o código de ativação por WhatsApp ou ligação.",
        "     O código apareceu na tela quando esta pasta foi criada. Perdeu? Gere outro pacote:",
        f"     python -m app.preparar --lacrar escritorio.env --saida {curto}.escritorio --forcar",
        f"  {3 if com_banco else 2}. Em cada computador do RH: EscritorioRecrutamento-Setup.exe → escolha o .escritorio →"
        " digite o código.",
        *([f"  {4 if com_banco else 3}. Teste: mande um currículo para {email} e veja chegar no escritório (o computador"
           " que comanda a equipe lê a caixa a cada 2 minutos)."] if email else []),
    ]
    if com_banco:
        linhas += [
            "",
            "OPERAÇÃO (Supabase → SQL Editor)",
            "  Suspender (ex.: falta de pagamento). O escritório mostra \"acesso recusado\" em segundos; os dados ficam:",
            f"    alter role {papel} nologin;",
            f"    select pg_terminate_backend(pid) from pg_stat_activity where usename = '{papel}';",
            "  Reativar. O escritório volta sozinho, sem reinstalar nada:",
            f"    alter role {papel} login;",
            "  A senha vazou. Gere senha e pacote novos e reinstale nos computadores:",
            f"    python -m app.preparar --trocar-senha \"<esta pasta>\"",
            "  Backup só desta empresa (no computador do fornecedor, com o projeto do Supabase):",
            f"    DATABASE_URL=\"<a URL do escritorio.env>\" DB_SCHEMA={schema} supabase/backup.sh backups/{curto}",
            "  Restaurou um backup diário do Supabase? Ele não guarda a senha deste papel, e o escritório mostra",
            "  \"acesso recusado\" até ela voltar. Use a mesma senha da DATABASE_URL do escritorio.env (nada muda nos",
            "  computadores):",
            f"    alter role {papel} with password '<a senha da DATABASE_URL do escritorio.env>';",
        ]
    return "\n".join(linhas).replace("\n\n\n", "\n\n") + "\n"


def _gravar_pacote(pasta: Path, curto: str, env: str) -> str:
    from .lacre import lacrar as _lacrar
    pacote, codigo = _lacrar(env.encode("utf-8"))
    (pasta / f"{curto}.escritorio").write_bytes(pacote)
    return codigo


def nova_empresa(a: argparse.Namespace) -> int:
    empresa = (a.nova_empresa or "").strip()
    curto = nome_curto(empresa)
    base = Path(a.pasta) if a.pasta else _pasta_padrao()
    pasta = base / curto
    if pasta.exists():
        print(f"{pasta} já existe: essa empresa já foi preparada. Nada foi alterado.", file=sys.stderr)
        return 1
    papel, schema = (f"hub_{curto}", f"rh_{curto}") if a.compartilhado else ("hub_app", "hub")
    senha = secrets.token_urlsafe(24)
    try:
        url = url_da_empresa(a.supabase, papel, senha) if a.supabase else ""
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1
    if a.imap_usuario and not a.imap_host:
        a.imap_host = "imap.gmail.com"
    if a.imap_usuario and not a.imap_senha:
        print("Falta --imap-senha (a senha de app da caixa de e-mail).", file=sys.stderr)
        return 1
    if not a.gemini_key and not a.demonstracao:
        print("Falta a chave do Gemini (--gemini-key). Para um pacote de teste sem IA real, use --demonstracao.",
              file=sys.stderr)
        return 1
    ns = argparse.Namespace(**{**vars(a), "local": True, "docker": False, "empresa": empresa, "supabase": url,
                               "schema": schema})
    env, _ = montar(ns)
    pasta.mkdir(parents=True)
    if url:
        (pasta / "1-banco.sql").write_text(_sql_do_banco(empresa, papel, schema, senha, a.compartilhado), encoding="utf-8")
    arq_env = pasta / "escritorio.env"
    arq_env.write_text(env, encoding="utf-8")
    os.chmod(arq_env, 0o600)
    codigo = _gravar_pacote(pasta, curto, env)
    (pasta / "LEIA-ME.txt").write_text(_leia_me(empresa, curto, papel, schema, a.compartilhado, bool(url),
                                                a.imap_usuario or "", a.demonstracao), encoding="utf-8")
    print(f"Empresa preparada: {empresa}")
    print(f"  Pasta: {pasta}")
    if url:
        print(f"  Banco: {'projeto compartilhado' if a.compartilhado else 'projeto só dela'} — papel {papel}, schema {schema}")
        print("  1) Rode 1-banco.sql no Supabase → SQL Editor (uma vez).")
    else:
        print("  Sem --supabase: o banco fica num computador só (SQLite). Para vários computadores, use o Supabase.")
    print(f"  {2 if url else 1}) Mande {curto}.escritorio por e-mail e o código por outro canal (WhatsApp, ligação).")
    print(f"  Código de ativação: {codigo}")
    print("  O código não fica guardado em lugar nenhum: anote-o agora (ou gere outro pacote depois).")
    if any(x in (a.imap_host or "").lower() or x in (a.imap_usuario or "").lower() for x in PROVEDORES_SEM_SENHA):
        print("  Atenção: a Microsoft (Outlook, Hotmail, 365) não aceita mais senha no IMAP. Use Gmail ou Google "
              "Workspace com senha de app, ou um provedor que aceite IMAP com senha.")
    return 0


def trocar_senha(pasta: Path) -> int:
    """A senha do banco vazou (ou um computador foi roubado): senha nova, SQL para o Supabase e pacote novo."""
    arq_env = pasta / "escritorio.env"
    if not arq_env.exists():
        print(f"{arq_env} não existe: indique a pasta da empresa (a que tem o escritorio.env).", file=sys.stderr)
        return 1
    env = arq_env.read_text(encoding="utf-8")
    m = re.search(r"^DATABASE_URL=postgresql://([^:@/]+):([^@/]*)@(.+)$", env, re.M)
    if not m:
        print("Este escritorio.env não usa banco no Supabase (DATABASE_URL): não há senha de banco para trocar.",
              file=sys.stderr)
        return 1
    papel = m.group(1).split(".", 1)[0]
    senha = secrets.token_urlsafe(24)
    env = env[:m.start()] + f"DATABASE_URL=postgresql://{m.group(1)}:{senha}@{m.group(3)}" + env[m.end():]
    quando = datetime.now().strftime("%d/%m/%Y %H:%M")
    (pasta / "2-trocar-senha.sql").write_text(
        f"-- Senha nova do banco ({quando}). Rode no Supabase → SQL Editor. Os computadores com o pacote antigo param\n"
        "-- na hora (aviso \"acesso recusado\"); instale o pacote novo neles e tudo volta, com os dados.\n"
        f"alter role {papel} with password '{senha}';\n"
        f"select pg_terminate_backend(pid) from pg_stat_activity where usename = '{papel}';\n", encoding="utf-8")
    arq_env.write_text(env, encoding="utf-8")
    os.chmod(arq_env, 0o600)
    codigo = _gravar_pacote(pasta, pasta.name, env)
    print(f"Senha nova gerada para {papel}.")
    print("  1) Rode 2-trocar-senha.sql no Supabase → SQL Editor.")
    print(f"  2) Mande o pacote novo ({pasta.name}.escritorio) e reinstale nos computadores da empresa.")
    print(f"  Código de ativação: {codigo}")
    return 0


def _pasta_padrao() -> Path:
    if getattr(sys, "frozen", False):  # NovaEmpresa.exe: na pasta Documentos do fornecedor
        docs = Path.home() / "Documents"
        return (docs if docs.exists() else Path.home()) / "Empresas do Escritório"
    return BACKEND.parent / "empresas"


def _perguntar(texto: str, padrao: str = "", segredo: bool = False) -> str:
    rotulo = f"{texto}{f' [{padrao}]' if padrao else ''}: "
    valor = (getpass.getpass(rotulo) if segredo else input(rotulo)).strip()
    return valor or padrao


def interativo(a: argparse.Namespace) -> int:
    """Sem argumentos (duplo clique no NovaEmpresa.exe): pergunta tudo, em português, e prepara a empresa."""
    print("Escritório de Recrutamento — preparar uma empresa nova")
    print("(Enter aceita o que está entre colchetes)\n")
    while not a.nova_empresa:
        a.nova_empresa = _perguntar("Nome da empresa")
    while True:
        a.supabase = _perguntar("Endereço do banco (Supabase → Connect → Transaction pooler; vazio = um computador só)")
        try:
            if a.supabase:
                url_da_empresa(a.supabase, "teste", "x")
            break
        except ValueError as e:
            print(f"  {e}")
    if a.supabase:
        a.compartilhado = _perguntar("Esse projeto do Supabase também atende outras empresas? (s/N)", "n").lower().startswith("s")
    a.gemini_key = _perguntar("Chave do Gemini (vazio = pacote de demonstração, sem IA real)",
                              os.environ.get("GEMINI_API_KEY", ""), segredo=True)
    a.demonstracao = not a.gemini_key
    a.imap_usuario = _perguntar("E-mail que recebe os currículos (vazio = sem leitura de e-mail)")
    if a.imap_usuario:
        a.imap_host = _perguntar("  Servidor IMAP", "imap.gmail.com")
        a.imap_senha = _perguntar("  Senha de app do e-mail (não aparece enquanto digita)", segredo=True)
    a.pasta = _perguntar("Pasta onde guardar", str(_pasta_padrao()))
    print()
    r = nova_empresa(a)
    if r == 0 and getattr(sys, "frozen", False) and sys.platform == "win32":
        os.startfile(str(Path(a.pasta) / nome_curto(a.nova_empresa)))  # noqa: S606 (abre a pasta no Explorer)
    return r


def main(argv=None) -> int:
    for fluxo in (sys.stdout, sys.stderr):  # console antigo do Windows (ou saída redirecionada): "→" vira "?", não erro
        if hasattr(fluxo, "reconfigure"):
            fluxo.reconfigure(errors="replace")
    ap = argparse.ArgumentParser(description="Gera o .env de produção com segredos fortes.")
    ap.add_argument("--docker", action="store_true", help="escreve ../.env para o docker-compose (PostgreSQL + HTTPS)")
    ap.add_argument("--local", action="store_true",
                    help="computadores Windows do RH, sem login: escreve ../escritorio.env para o instalador")
    ap.add_argument("--dominio", default="localhost", help="domínio público, ex.: rh.suaempresa.com.br (só com --docker)")
    ap.add_argument("--empresa", default="", help="nome da empresa (aparece nos prompts e na LGPD)")
    ap.add_argument("--nome", default="", help="nome do sistema exibido na tela (vazio: nenhum)")
    ap.add_argument("--supabase", default="", help="URL do banco no Supabase (Connect → Session pooler), com usuário e senha")
    ap.add_argument("--admin-email", default="admin@suaempresa.com.br")
    ap.add_argument("--gemini-key", default=os.environ.get("GEMINI_API_KEY", ""), help="chave do Gemini (ou defina depois)")
    ap.add_argument("--arquivo", help="caminho do arquivo a gerar (padrão: ../.env com --docker, ./.env sem)")
    ap.add_argument("--forcar", action="store_true", help="sobrescreve um arquivo existente")
    ap.add_argument("--lacrar", metavar="ENV", help="lacra uma configuração pronta para entregar à empresa (.escritorio)")
    ap.add_argument("--saida", help="com --lacrar: caminho do pacote (padrão: mesmo nome, extensão .escritorio)")
    g = ap.add_argument_group("empresa nova (kit do fornecedor)")
    g.add_argument("--nova-empresa", nargs="?", const="", metavar="NOME",
                   help="prepara banco, configuração e pacote lacrado de uma empresa; sem NOME, pergunta tudo")
    g.add_argument("--compartilhado", action="store_true",
                   help="o projeto do Supabase atende várias empresas: papel e schema próprios para esta")
    g.add_argument("--demonstracao", action="store_true", help="pacote de teste: análises por regras, sem custo de IA")
    g.add_argument("--imap-host", default="", help="servidor IMAP (padrão com e-mail: imap.gmail.com)")
    g.add_argument("--imap-usuario", default="", help="e-mail que recebe os currículos")
    g.add_argument("--imap-senha", default="", help="senha de app do e-mail")
    g.add_argument("--imap-pasta", default="INBOX")
    g.add_argument("--limite-diario", type=float, default=3, help="teto de gasto de IA por dia, em US$ (padrão 3)")
    g.add_argument("--pasta", help="onde criar a pasta da empresa (padrão: empresas/ no projeto)")
    g.add_argument("--trocar-senha", metavar="PASTA", help="senha do banco vazou: senha, SQL e pacote novos")
    a = ap.parse_args(argv)

    if a.lacrar:
        return lacrar(Path(a.lacrar), Path(a.saida) if a.saida else None, a.forcar)
    if a.trocar_senha:
        return trocar_senha(Path(a.trocar_senha))
    if a.nova_empresa is not None:
        return nova_empresa(a) if a.nova_empresa.strip() else interativo(a)

    if a.local and a.docker:
        print("Use --local (computadores do RH) ou --docker (servidor), não os dois.", file=sys.stderr)
        return 1
    destino = Path(a.arquivo) if a.arquivo else (BACKEND.parent / "escritorio.env" if a.local
                                                 else BACKEND.parent / ".env" if a.docker else BACKEND / ".env")
    if destino.exists() and not a.forcar:
        print(f"{destino} já existe; nada foi alterado (use --forcar para gerar outro).", file=sys.stderr)
        return 1
    conteudo, segredos = montar(a)
    destino.write_text(conteudo, encoding="utf-8")
    os.chmod(destino, 0o600)
    print(f"Configuração gravada em {destino} (permissão 600).")
    if a.local:
        print("  Copie este arquivo, junto com a pasta do projeto, para cada computador do RH e rode windows\\instalar.bat.")
        print("  Guarde uma cópia num cofre de senhas: sem a CHAVE_ARQUIVOS os currículos não podem ser lidos.")
        if not a.supabase:
            print("  Sem --supabase o banco fica só neste computador (SQLite): para vários computadores, use o Supabase.")
    else:
        print(f"  Administrador: {a.admin_email}  senha inicial: {segredos['ADMIN_SENHA']}  (troca obrigatória no 1º acesso)")
        print(f"  Chave da API de triagem (cabeçalho X-API-Key do fluxo de e-mail): {segredos['TRIAGEM_API_KEY']}")
    if not a.gemini_key:
        print("  Falta GEMINI_API_KEY: preencha no arquivo antes de subir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
