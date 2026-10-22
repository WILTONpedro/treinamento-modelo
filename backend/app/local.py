"""Abre o escritório neste computador (o atalho da área de trabalho chama isto).

    python -m app.local              sobe o programa (se ainda não estiver no ar) e abre a janela do escritório
    python -m app.local --sem-janela só sobe o programa (inicialização com o Windows: a equipe já trabalha)

O programa fica aberto só para este computador (127.0.0.1) e sem login: quem usa é o usuário do Windows.
Fechar a janela não para o programa; ele segue em segundo plano até o Windows ser desligado.
"""
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

os.environ.setdefault("ACESSO", "local")  # antes de importar a configuração

from . import config  # noqa: E402

URL = f"http://127.0.0.1:{config.PORTA_LOCAL}/"


def no_ar() -> bool:
    try:
        with urllib.request.urlopen(URL + "api/marca", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def abrir_janela():
    """Abre como aplicativo (janela própria, sem barra de endereço) no Edge ou no Chrome; senão, no navegador."""
    if sys.platform == "win32":
        candidatos = [r"%ProgramFiles(x86)%\Microsoft\Edge\Application\msedge.exe",
                      r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe",
                      r"%ProgramFiles%\Google\Chrome\Application\chrome.exe",
                      r"%LocalAppData%\Google\Chrome\Application\chrome.exe"]
        for c in candidatos:
            exe = os.path.expandvars(c)
            if os.path.exists(exe):
                subprocess.Popen([exe, f"--app={URL}", "--start-maximized"])
                return
    webbrowser.open(URL)


def _registrar_em_arquivo():
    """Sem console (pythonw), o que o programa conta vai para um arquivo na pasta de dados."""
    arq = Path(config.PASTA_DADOS) / "escritorio.log"
    logging.basicConfig(filename=arq, level=logging.INFO, encoding="utf-8",
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if sys.stdout is None:
        sys.stdout = sys.stderr = open(arq, "a", encoding="utf-8", buffering=1)
    return arq


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    janela = "--sem-janela" not in argv
    if not config.LOCAL:
        print("Este atalho é para ACESSO=local (no .env).", file=sys.stderr)
        return 1
    if no_ar():  # já está rodando (inicialização com o Windows ou outro clique): só abre a janela
        if janela:
            abrir_janela()
        return 0
    _registrar_em_arquivo()
    if janela:
        def _quando_subir():
            for _ in range(240):
                if no_ar():
                    abrir_janela()
                    return
                time.sleep(0.5)
        threading.Thread(target=_quando_subir, daemon=True).start()
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=config.PORTA_LOCAL, log_config=None, access_log=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
