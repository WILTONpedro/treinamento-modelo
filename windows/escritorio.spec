# -*- mode: python -*-
# PyInstaller: gera o programa do Escritório de Recrutamento (uma pasta com EscritorioRecrutamento.exe e o que
# ele precisa). Sem segredos dentro: a configuração de cada empresa (escritorio.env) é escolhida na instalação.
#
#   pip install -r backend/requirements.txt pyinstaller
#   pyinstaller windows/escritorio.spec
import os

from PyInstaller.utils.hooks import collect_all

raiz = os.path.abspath(os.path.join(SPECPATH, ".."))  # noqa: F821 (SPECPATH vem do PyInstaller)
datas, binaries, hiddenimports = [], [], ["app.main"]
# pacotes que carregam partes sob demanda (driver do banco, protocolos do servidor, modelos de documento, fusos)
for pacote in ("psycopg", "psycopg_binary", "uvicorn", "tzdata", "docx"):
    d, b, h = collect_all(pacote)
    datas, binaries, hiddenimports = datas + d, binaries + b, hiddenimports + h
datas += [(os.path.join(raiz, "frontend"), "frontend"), (os.path.join(raiz, "backend", "demo"), "demo")]

a = Analysis(  # noqa: F821
    [os.path.join(raiz, "backend", "escritorio.py")],
    pathex=[os.path.join(raiz, "backend")],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "pytest"],
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="EscritorioRecrutamento",
    console=False,  # sem janela preta: o escritório abre no Edge/Chrome em modo aplicativo
    icon=os.path.join(raiz, "windows", "icone.ico"),
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="EscritorioRecrutamento", upx=False)  # noqa: F821

# Ativador.exe (um arquivo só, pequeno): o instalador o extrai para uma pasta temporária e o roda na página do
# código de ativação, para conferir o código e guardar a configuração lacrada antes de instalar (app/lacre.py).
b = Analysis(  # noqa: F821
    [os.path.join(raiz, "backend", "ativador.py")],
    pathex=[os.path.join(raiz, "backend")],
    excludes=["tkinter", "pytest"],
)
EXE(  # noqa: F821
    PYZ(b.pure), b.scripts, b.binaries, b.datas, [],  # noqa: F821
    name="Ativador",
    console=True,  # o instalador o roda escondido e só lê o código de saída
    icon=os.path.join(raiz, "windows", "icone.ico"),
    upx=False,
)

# NovaEmpresa.exe (um arquivo só): o kit do FORNECEDOR para preparar cada empresa nova (banco, e-mail, configuração
# e pacote lacrado). Fica fora do instalador das empresas: vai num artefato separado ("Ferramentas-do-fornecedor").
c = Analysis(  # noqa: F821
    [os.path.join(raiz, "backend", "nova_empresa.py")],
    pathex=[os.path.join(raiz, "backend")],
    excludes=["tkinter", "pytest", "fastapi", "uvicorn", "sqlalchemy", "psycopg"],
)
EXE(  # noqa: F821
    PYZ(c.pure), c.scripts, c.binaries, c.datas, [],  # noqa: F821
    name="NovaEmpresa",
    console=True,  # janela de texto: pergunta e mostra o código de ativação
    icon=os.path.join(raiz, "windows", "icone.ico"),
    upx=False,
)
