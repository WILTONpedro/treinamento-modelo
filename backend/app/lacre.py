"""Configuração lacrada: a empresa recebe a configuração sem nunca ter um arquivo legível no computador.

    python -m app.preparar --lacrar escritorio.env      gera escritorio.escritorio e mostra o código de ativação
    python -m app.lacre empresa.escritorio              guarda a configuração neste computador (o instalador chama
                                                        isto pelo Ativador.exe, com o código em ESCRITORIO_CODIGO)

O pacote (.escritorio) vai cifrado com uma chave tirada do código de ativação (scrypt + AES-GCM): mande o arquivo
por um canal (e-mail) e o código por outro (WhatsApp). Na instalação, o programa abre o pacote com o código e
guarda a configuração cifrada pelo próprio Windows (DPAPI): só aquele usuário, naquele computador, consegue lê-la.

Isso evita o vazamento casual (abrir no Bloco de Notas, copiar para outro computador, repassar o arquivo). Quem
tem controle total do computador ainda consegue tirar os segredos da memória do programa: por isso cada empresa
tem o próprio banco, a própria chave do Gemini e um teto de gasto, e tudo pode ser trocado (veja o README).
"""
import hashlib
import os
import secrets
import sys
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGICO = b"ESCR1"
_ALFABETO = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # sem 0/O, 1/I/L: fácil de ditar e digitar
_ENTROPIA = b"EscritorioRecrutamento/config/v1"


class ErroLacre(Exception):
    pass


class CodigoErrado(ErroLacre):
    pass


def gerar_codigo() -> str:
    letras = "".join(secrets.choice(_ALFABETO) for _ in range(20))  # ~99 bits
    return "-".join(letras[i:i + 4] for i in range(0, 20, 4))


def _normalizar(codigo: str) -> str:
    return "".join(c for c in (codigo or "").upper() if c.isalnum())


def _chave(codigo: str, sal: bytes) -> bytes:
    return hashlib.scrypt(_normalizar(codigo).encode(), salt=sal, n=2 ** 15, r=8, p=1, maxmem=64 * 2 ** 20, dklen=32)


def lacrar(texto: bytes, codigo: str | None = None) -> tuple[bytes, str]:
    codigo = codigo or gerar_codigo()
    sal, nonce = os.urandom(16), os.urandom(12)
    return MAGICO + sal + nonce + AESGCM(_chave(codigo, sal)).encrypt(nonce, texto, MAGICO), codigo


def abrir(pacote: bytes, codigo: str) -> bytes:
    if not pacote.startswith(MAGICO) or len(pacote) < len(MAGICO) + 28 + 16:
        raise ErroLacre("Este arquivo não é uma configuração do Escritório (.escritorio).")
    sal, nonce, cifrado = pacote[5:21], pacote[21:33], pacote[33:]
    try:
        return AESGCM(_chave(codigo, sal)).decrypt(nonce, cifrado, MAGICO)
    except InvalidTag:
        raise CodigoErrado("Código de ativação errado (ou arquivo alterado).") from None


# --- DPAPI do Windows: cifra para o usuário atual deste computador -------------------------------------------
def _dpapi(dados: bytes, cifrar: bool) -> bytes:
    if sys.platform != "win32":
        raise ErroLacre("A configuração lacrada só pode ser guardada no Windows.")
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    def blob(b):
        buf = ctypes.create_string_buffer(b, len(b))
        return Blob(len(b), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))), buf

    entrada, _b1 = blob(dados)
    entropia, _b2 = blob(_ENTROPIA)
    saida = Blob()
    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    funcao = crypt32.CryptProtectData if cifrar else crypt32.CryptUnprotectData
    ok = funcao(ctypes.byref(entrada), None, ctypes.byref(entropia), None, None, 0x1,  # 0x1 = sem janelas
                ctypes.byref(saida))
    if not ok:
        raise ErroLacre("O Windows não conseguiu " + ("guardar" if cifrar else "ler") + " a configuração deste "
                        "computador. Instale de novo com o arquivo .escritorio e o código de ativação.")
    try:
        return ctypes.string_at(saida.pbData, saida.cbData)
    finally:
        kernel32.LocalFree(saida.pbData)


def proteger(dados: bytes) -> bytes:
    return _dpapi(dados, True)


def desproteger(dados: bytes) -> bytes:
    return _dpapi(dados, False)


def ativar(pacote: Path, codigo: str, pasta: Path) -> Path:
    """Abre o pacote com o código e guarda a configuração cifrada pelo Windows em pasta/escritorio.cfg."""
    texto = abrir(Path(pacote).read_bytes(), codigo)
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "escritorio.cfg"
    temporario = destino.with_suffix(".tmp")
    temporario.write_bytes(proteger(texto))
    os.replace(temporario, destino)
    legivel = pasta / "escritorio.env"
    if legivel.exists():  # a cópia legível de uma instalação antiga sai: vale a lacrada
        legivel.unlink()
    return destino


def main(argv=None) -> int:
    """Ativação pelo instalador: 0 = ok, 2 = código errado, 3 = arquivo inválido ou o Windows não guardou."""
    from .config import PASTA_CONFIG
    argv = sys.argv[1:] if argv is None else argv
    codigo = os.environ.pop("ESCRITORIO_CODIGO", "")
    if len(argv) != 1:
        print("uso: python -m app.lacre empresa.escritorio  (código em ESCRITORIO_CODIGO)", file=sys.stderr)
        return 3
    try:
        ativar(Path(argv[0]), codigo, PASTA_CONFIG)
    except ErroLacre as e:
        print(e, file=sys.stderr)
        return 2 if isinstance(e, CodigoErrado) else 3
    except OSError as e:
        print(f"Não consegui ler ou gravar a configuração: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
