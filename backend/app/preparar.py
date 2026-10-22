"""Gera o arquivo de configuração de produção com segredos fortes.

    cd backend
    python -m app.preparar --docker --dominio rh.suaempresa.com.br --empresa "Sua Empresa" \\
        --admin-email rh@suaempresa.com.br --gemini-key SUA_CHAVE

--docker   escreve ../.env (lido pelo docker-compose: app + PostgreSQL + HTTPS automático)
sem ele    escreve ./.env (servidor simples com SQLite, para rodar com uvicorn)

Nunca sobrescreve um .env existente (use --forcar). Mostra a senha inicial do administrador e a
chave da API de triagem uma única vez: guarde-as num cofre de senhas.
"""
import argparse
import os
import secrets
import sys
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
    linhas = [
        "# Gerado por python -m app.preparar — NUNCA versione este arquivo. Guarde uma cópia num cofre de senhas:",
        "# sem CHAVE_ARQUIVOS os currículos cifrados não podem ser lidos.",
        "AMBIENTE=producao",
        "",
        "# --- Identidade",
        f"APP_NOME={a.nome}",
        "APP_SUBTITULO=Escritório de Recrutamento",
        f"EMPRESA_NOME={a.empresa}",
        "",
        "# --- IA (use o plano pago: no gratuito o Google pode usar o conteúdo para melhorar produtos)",
        "IA_MODO=gemini",
        f"GEMINI_API_KEY={a.gemini_key}",
        "GEMINI_MODELO=gemini-3.5-flash-lite",
        "GEMINI_PENSAMENTO=minimal",
        "IA_LIMITE_DIARIO_USD=3",
        "",
        "# --- Segurança",
        f"SEGREDO_JWT={segredos['SEGREDO_JWT']}",
        f"CHAVE_ARQUIVOS={segredos['CHAVE_ARQUIVOS']}",
        f"TRIAGEM_API_KEY={segredos['TRIAGEM_API_KEY']}",
        "COOKIE_SECURE=true",
        f"ADMIN_EMAIL={a.admin_email}",
        f"ADMIN_SENHA={segredos['ADMIN_SENHA']}",
        "ADMIN_NOME=Administrador RH",
        "",
        "# --- LGPD",
        "RETENCAO_DIAS=180",
        "RETENCAO_CONVERSAS_HORAS=72",
    ]
    if a.docker:
        senha_banco = secrets.token_urlsafe(24)
        segredos["POSTGRES_PASSWORD"] = senha_banco
        linhas += ["", "# --- Docker (docker-compose.yml)",
                   f"DOMINIO={a.dominio}",
                   f"POSTGRES_PASSWORD={senha_banco}",
                   f"DATABASE_URL=postgresql+psycopg://hub:{senha_banco}@db:5432/hub",
                   "PASTA_DADOS=/dados"]
    return "\n".join(linhas) + "\n", segredos


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Gera o .env de produção com segredos fortes.")
    ap.add_argument("--docker", action="store_true", help="escreve ../.env para o docker-compose (PostgreSQL + HTTPS)")
    ap.add_argument("--dominio", default="localhost", help="domínio público, ex.: rh.suaempresa.com.br (só com --docker)")
    ap.add_argument("--empresa", default="", help="nome da empresa (aparece nos prompts e na LGPD)")
    ap.add_argument("--nome", default="AI Hub", help="nome do sistema exibido na tela")
    ap.add_argument("--admin-email", default="admin@suaempresa.com.br")
    ap.add_argument("--gemini-key", default=os.environ.get("GEMINI_API_KEY", ""), help="chave do Gemini (ou defina depois)")
    ap.add_argument("--arquivo", help="caminho do arquivo a gerar (padrão: ../.env com --docker, ./.env sem)")
    ap.add_argument("--forcar", action="store_true", help="sobrescreve um arquivo existente")
    a = ap.parse_args(argv)

    destino = Path(a.arquivo) if a.arquivo else (BACKEND.parent / ".env" if a.docker else BACKEND / ".env")
    if destino.exists() and not a.forcar:
        print(f"{destino} já existe; nada foi alterado (use --forcar para gerar outro).", file=sys.stderr)
        return 1
    conteudo, segredos = montar(a)
    destino.write_text(conteudo, encoding="utf-8")
    os.chmod(destino, 0o600)
    print(f"Configuração gravada em {destino} (permissão 600).")
    print(f"  Administrador: {a.admin_email}  senha inicial: {segredos['ADMIN_SENHA']}  (troca obrigatória no 1º acesso)")
    print(f"  Chave da API de triagem (cabeçalho X-API-Key do fluxo de e-mail): {segredos['TRIAGEM_API_KEY']}")
    if not a.gemini_key:
        print("  Falta GEMINI_API_KEY: preencha no arquivo antes de subir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
