"""NovaEmpresa.exe: o kit do fornecedor para vender a uma empresa nova (não vai para as empresas).

Duplo clique: pergunta tudo, em português, e cria a pasta da empresa em Documentos\\Empresas do Escritório.
Com argumentos, é o mesmo que python -m app.preparar --nova-empresa ... (ou --trocar-senha / --lacrar).
"""
import sys

from app.preparar import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or not args[0].startswith("--"):
        args = ["--nova-empresa", *args]
    codigo = main(args)
    if getattr(sys, "frozen", False) and len(sys.argv) == 1:  # duplo clique: a janela não fecha antes de ler
        input("\nPressione Enter para fechar.")
    raise SystemExit(codigo)
