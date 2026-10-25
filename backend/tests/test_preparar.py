"""Kit do fornecedor: preparar uma empresa nova (banco, e-mail, configuração e pacote lacrado) e trocar a senha."""
import re

import pytest

from app import lacre, preparar

URL_SUPABASE = "postgresql://postgres.abcdefghij:[YOUR-PASSWORD]@aws-1-sa-east-1.pooler.supabase.com:6543/postgres"


def _codigo(saida: str) -> str:
    return re.search(r"Código de ativação: ([A-Z0-9-]{24})", saida).group(1)


def _env(texto: str) -> dict:
    return dict(linha.split("=", 1) for linha in texto.splitlines() if linha and not linha.startswith("#") and "=" in linha)


def test_nome_curto_e_url():
    assert preparar.nome_curto("Metalúrgica Aço Forte Ltda.") == "metalurgica_aco_forte_ltda"
    assert preparar.nome_curto("  ") == "empresa"
    assert (preparar.url_da_empresa(URL_SUPABASE, "hub_x", "s3nh4") ==
            "postgresql://hub_x.abcdefghij:s3nh4@aws-1-sa-east-1.pooler.supabase.com:6543/postgres")
    # sem o <ref> do Supabase (um Postgres comum), o usuário é só o papel
    assert preparar.url_da_empresa("postgresql://postgres@127.0.0.2:6543/postgres", "hub_y", "s") == \
        "postgresql://hub_y:s@127.0.0.2:6543/postgres"
    with pytest.raises(ValueError):
        preparar.url_da_empresa("https://supabase.com/dashboard", "hub_x", "s")


def test_empresa_com_projeto_proprio(tmp_path, capsys):
    assert preparar.main(["--nova-empresa", "Padaria Pão Dourado", "--supabase", URL_SUPABASE, "--gemini-key", "chave-x",
                          "--imap-usuario", "curriculos@padaria.com.br", "--imap-senha", "abcd efgh", "--pasta",
                          str(tmp_path)]) == 0
    pasta = tmp_path / "padaria_pao_dourado"
    assert sorted(p.name for p in pasta.iterdir()) == ["1-banco.sql", "LEIA-ME.txt", "escritorio.env",
                                                       "padaria_pao_dourado.escritorio"]
    texto = (pasta / "escritorio.env").read_text(encoding="utf-8")
    env = _env(texto)
    senha = re.search(r"hub_app\.abcdefghij:([^@]+)@", env["DATABASE_URL"]).group(1)
    assert len(senha) >= 30 and env["DB_SCHEMA"] == "hub" and env["ARMAZENAMENTO"] == "banco"
    assert env["ACESSO"] == "local" and env["IA_MODO"] == "gemini" and env["AMBIENTE"] == "producao"
    assert env["IMAP_ATIVO"] == "true" and env["IMAP_HOST"] == "imap.gmail.com" and env["IMAP_SENHA"] == "abcd efgh"
    assert env["EMPRESA_NOME"] == "Padaria Pão Dourado" and len(env["CHAVE_ARQUIVOS"]) == 44
    sql = (pasta / "1-banco.sql").read_text(encoding="utf-8")
    assert f"create role hub_app with login password '{senha}';" in sql  # a mesma senha no banco e na configuração
    assert "create schema hub authorization hub_app;" in sql and "revoke all on schema hub from public;" in sql
    assert "alter role hub_app set search_path = hub;" in sql and "statement_timeout" in sql
    # o pacote abre com o código que apareceu na tela e é a mesma configuração; nada legível dentro dele
    pacote = (pasta / "padaria_pao_dourado.escritorio").read_bytes()
    assert senha.encode() not in pacote and b"abcd efgh" not in pacote
    assert lacre.abrir(pacote, _codigo(capsys.readouterr().out)).decode() == texto
    leia = (pasta / "LEIA-ME.txt").read_text(encoding="utf-8")
    assert "alter role hub_app nologin;" in leia and "alter role hub_app login;" in leia and "--trocar-senha" in leia
    assert "alter role hub_app with password" in leia and senha not in leia  # backup do Supabase: a senha não vai escrita
    # a mesma empresa de novo: não mexe em nada
    assert preparar.main(["--nova-empresa", "Padaria Pão Dourado", "--supabase", URL_SUPABASE, "--gemini-key", "x",
                          "--pasta", str(tmp_path)]) == 1


def test_empresa_em_projeto_compartilhado_e_demonstracao(tmp_path, capsys):
    assert preparar.main(["--nova-empresa", "Clínica Bem Viver", "--compartilhado", "--demonstracao", "--supabase",
                          URL_SUPABASE, "--pasta", str(tmp_path)]) == 0
    pasta = tmp_path / "clinica_bem_viver"
    env = _env((pasta / "escritorio.env").read_text(encoding="utf-8"))
    assert env["DATABASE_URL"].startswith("postgresql://hub_clinica_bem_viver.abcdefghij:")
    assert env["DB_SCHEMA"] == "rh_clinica_bem_viver" and env["IA_MODO"] == "demo"
    assert env["AMBIENTE"] == "desenvolvimento" and env["IMAP_ATIVO"] == "false"
    sql = (pasta / "1-banco.sql").read_text(encoding="utf-8")
    assert "create schema rh_clinica_bem_viver authorization hub_clinica_bem_viver;" in sql
    assert "já existe neste projeto" in sql  # não reaproveita (nem troca a senha de) um papel de outra empresa


def test_sem_supabase_e_validacoes(tmp_path, capsys):
    # sem banco no Supabase: um computador só (SQLite), sem 1-banco.sql
    assert preparar.main(["--nova-empresa", "Loja Teste", "--demonstracao", "--pasta", str(tmp_path)]) == 0
    assert not (tmp_path / "loja_teste" / "1-banco.sql").exists()
    assert "DATABASE_URL" not in (tmp_path / "loja_teste" / "escritorio.env").read_text(encoding="utf-8")
    # sem chave do Gemini e sem --demonstracao; e-mail sem senha; endereço do banco errado
    assert preparar.main(["--nova-empresa", "A", "--pasta", str(tmp_path)]) == 1
    assert preparar.main(["--nova-empresa", "B", "--demonstracao", "--imap-usuario", "x@y.com", "--pasta", str(tmp_path)]) == 1
    assert preparar.main(["--nova-empresa", "C", "--demonstracao", "--supabase", "supabase.com", "--pasta", str(tmp_path)]) == 1
    # Microsoft não aceita senha no IMAP: avisa
    capsys.readouterr()
    assert preparar.main(["--nova-empresa", "D", "--demonstracao", "--imap-usuario", "rh@d.com", "--imap-senha", "s",
                          "--imap-host", "outlook.office365.com", "--pasta", str(tmp_path)]) == 0
    assert "Microsoft" in capsys.readouterr().out


def test_trocar_senha(tmp_path, capsys):
    preparar.main(["--nova-empresa", "Metalúrgica Aço Forte", "--compartilhado", "--demonstracao", "--supabase",
                   URL_SUPABASE, "--pasta", str(tmp_path)])
    pasta = tmp_path / "metalurgica_aco_forte"
    antes = _env((pasta / "escritorio.env").read_text(encoding="utf-8"))
    pacote_antigo = (pasta / "metalurgica_aco_forte.escritorio").read_bytes()
    capsys.readouterr()
    assert preparar.main(["--trocar-senha", str(pasta)]) == 0
    depois_txt = (pasta / "escritorio.env").read_text(encoding="utf-8")
    depois = _env(depois_txt)
    assert depois["DATABASE_URL"] != antes["DATABASE_URL"] and depois["CHAVE_ARQUIVOS"] == antes["CHAVE_ARQUIVOS"]
    nova = re.search(r":([^:@]+)@", depois["DATABASE_URL"]).group(1)
    sql = (pasta / "2-trocar-senha.sql").read_text(encoding="utf-8")
    assert f"alter role hub_metalurgica_aco_forte with password '{nova}';" in sql and "pg_terminate_backend" in sql
    pacote = (pasta / "metalurgica_aco_forte.escritorio").read_bytes()
    assert pacote != pacote_antigo
    assert lacre.abrir(pacote, _codigo(capsys.readouterr().out)).decode() == depois_txt
    assert preparar.main(["--trocar-senha", str(tmp_path / "nao-existe")]) == 1


def test_modo_interativo(tmp_path, capsys, monkeypatch):
    respostas = iter(["Clínica Bem Viver", "endereço errado", URL_SUPABASE, "s", "rh@clinica.com.br", "", str(tmp_path)])
    monkeypatch.setattr("builtins.input", lambda _: next(respostas))
    segredos = iter(["", "senha de app"])  # sem chave do Gemini: pacote de demonstração
    monkeypatch.setattr(preparar.getpass, "getpass", lambda _: next(segredos))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert preparar.main(["--nova-empresa"]) == 0
    saida = capsys.readouterr().out
    assert "postgresql://" in saida  # explicou o endereço errado e perguntou de novo
    env = _env((tmp_path / "clinica_bem_viver" / "escritorio.env").read_text(encoding="utf-8"))
    assert env["DB_SCHEMA"] == "rh_clinica_bem_viver" and env["IA_MODO"] == "demo"
    assert env["IMAP_HOST"] == "imap.gmail.com" and env["IMAP_SENHA"] == "senha de app"
    assert _codigo(saida)
