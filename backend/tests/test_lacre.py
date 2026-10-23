"""Configuração lacrada: a empresa recebe o pacote .escritorio e o código, nunca um arquivo legível."""
import os
import sys

import pytest

from app import config, lacre, preparar

TEXTO = "ACESSO=local\nDATABASE_URL=postgresql://hub_app.ref:segredo@host:6543/postgres\nCHAVE_LACRE_TESTE=sim\n".encode()


def test_pacote_abre_so_com_o_codigo_certo():
    pacote, codigo = lacre.lacrar(TEXTO)
    assert b"segredo" not in pacote and pacote.startswith(lacre.MAGICO)
    assert len(codigo) == 24 and codigo.count("-") == 4
    assert lacre.abrir(pacote, codigo) == TEXTO
    assert lacre.abrir(pacote, codigo.lower().replace("-", " ")) == TEXTO  # digitado de qualquer jeito
    with pytest.raises(lacre.CodigoErrado):
        lacre.abrir(pacote, lacre.gerar_codigo())
    adulterado = pacote[:-1] + bytes([pacote[-1] ^ 1])
    with pytest.raises(lacre.CodigoErrado):
        lacre.abrir(adulterado, codigo)
    with pytest.raises(lacre.ErroLacre) as e:
        lacre.abrir(b"ACESSO=local\n", codigo)  # um .env renomeado não é pacote
    assert not isinstance(e.value, lacre.CodigoErrado)
    assert lacre.lacrar(TEXTO, codigo)[0] != pacote  # sal e nonce novos a cada pacote


@pytest.fixture
def windows_simulado(monkeypatch):
    """Fora do Windows não há DPAPI: simula com uma troca reversível só para conferir o caminho dos dados."""
    monkeypatch.setattr(lacre, "proteger", lambda d: b"DPAPI" + d[::-1])

    def desproteger(d):
        if not d.startswith(b"DPAPI"):
            raise lacre.ErroLacre("O Windows não conseguiu ler a configuração deste computador.")
        return d[5:][::-1]
    monkeypatch.setattr(lacre, "desproteger", desproteger)


def test_ativacao_guarda_cifrado_e_o_programa_le(tmp_path, monkeypatch, windows_simulado):
    pasta = tmp_path / "EscritorioRecrutamento"
    pasta.mkdir()
    (pasta / "escritorio.env").write_text("ANTIGA=1\n")  # instalação antiga, legível
    pacote = tmp_path / "empresa.escritorio"
    bruto, codigo = lacre.lacrar(TEXTO)
    pacote.write_bytes(bruto)
    monkeypatch.setattr(config, "PASTA_CONFIG", pasta)

    monkeypatch.setenv("ESCRITORIO_CODIGO", lacre.gerar_codigo())
    assert lacre.main([str(pacote)]) == 2 and not (pasta / "escritorio.cfg").exists()
    assert "ESCRITORIO_CODIGO" not in os.environ  # o código não fica no ambiente
    monkeypatch.setenv("ESCRITORIO_CODIGO", codigo)
    (tmp_path / "falso.escritorio").write_text("ACESSO=local\n")
    assert lacre.main([str(tmp_path / "falso.escritorio")]) == 3
    monkeypatch.setenv("ESCRITORIO_CODIGO", codigo)
    assert lacre.main([str(pacote)]) == 0

    cfg = pasta / "escritorio.cfg"
    assert b"segredo" not in cfg.read_bytes() and not (pasta / "escritorio.env").exists()
    monkeypatch.setattr(config, "arquivo_de_configuracao", lambda: cfg)
    config._carregar_env()
    assert os.environ.pop("CHAVE_LACRE_TESTE") == "sim" and config.ERRO_CONFIGURACAO == ""

    cfg.write_bytes(b"copiado de outro computador")  # o DPAPI de outro usuário não abre
    config._carregar_env()
    assert "Windows" in config.ERRO_CONFIGURACAO
    monkeypatch.setattr(config, "ERRO_CONFIGURACAO", "")


def test_configuracao_lacrada_vem_antes_da_legivel(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CONGELADO", True)
    monkeypatch.setattr(config, "PASTA_CONFIG", tmp_path)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "programa" / "EscritorioRecrutamento.exe"))
    assert config.arquivo_de_configuracao() is None
    (tmp_path / "escritorio.env").write_text("X=1\n")
    assert config.arquivo_de_configuracao().name == "escritorio.env"
    (tmp_path / "escritorio.cfg").write_bytes(b"x")
    assert config.arquivo_de_configuracao().name == "escritorio.cfg"


def test_preparar_lacra_a_configuracao(tmp_path, capsys):
    env = tmp_path / "escritorio.env"
    env.write_bytes(TEXTO)
    assert preparar.main(["--lacrar", str(env)]) == 0
    saida = capsys.readouterr().out
    codigo = next(p for p in saida.split() if p.count("-") == 4 and len(p) == 24)
    assert lacre.abrir((tmp_path / "escritorio.escritorio").read_bytes(), codigo) == TEXTO
    assert preparar.main(["--lacrar", str(env)]) == 1  # não sobrescreve
    assert preparar.main(["--lacrar", str(tmp_path / "nao-existe.env")]) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="DPAPI só existe no Windows")
def test_dpapi_de_verdade():
    assert lacre.desproteger(lacre.proteger(TEXTO)) == TEXTO
    assert b"segredo" not in lacre.proteger(TEXTO)
