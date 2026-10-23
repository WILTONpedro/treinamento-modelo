"""Momentos da equipe: quem está livre faz alguma coisa junto — pingue-pongue, café, papo no bebedouro, videogame,
clube do livro, alongamento, o aniversário de alguém — e o Atlas dá uma volta pelas mesas de quem está trabalhando.

Roda só na cópia que comanda o escritório (junto com a supervisão do Atlas). Cada momento vira uma sequência de
eventos "conversa" com os dados do momento (quem vai aonde e por quanto tempo): todas as telas mostram a mesma cena,
e o papo aparece também no canal da equipe. Só entra quem está de fato livre (sem tarefa, sem fila, sem pendência);
se chegar trabalho no meio, a tela manda a pessoa de volta para a mesa.

O diálogo é escrito pela IA (com a história e o jeito de cada um) até MOMENTOS_IA_POR_DIA vezes por dia; fora isso,
e na demonstração, vem dos roteiros abaixo, temperados com as frases de cada personalidade.
"""
import logging
import random
import time
from datetime import date

from sqlalchemy import select

from . import config, eventos, ia, uso_ia
from .db import sessao
from .models import Agente, Vaga
from .personalidade import PADRAO, bordao

log = logging.getLogger("hub.momentos")

TIPOS = {
    "pingue": {"n": (2, 2), "lugares": [["ping0", "ping1"]], "duracao": (55, 80), "peso": 4,
               "situacao": "uma partida de pingue-pongue na área de descanso", "rotulo": "jogando pingue-pongue"},
    "cafe": {"n": (2, 3), "lugares": [["comer0", "comer1", "comer2"], ["comer3", "comer4", "comer5"]], "duracao": (45, 70),
             "peso": 4, "situacao": "um café juntos na copa", "rotulo": "tomando um café com a equipe"},
    "bebedouro": {"n": (2, 2), "lugares": [["bebedouro", "papoAgua"]], "duracao": (30, 45), "peso": 3,
                  "situacao": "um papo rápido no bebedouro", "rotulo": "conversando no bebedouro"},
    "videogame": {"n": (2, 2), "lugares": [["sofa1", "sofa2"]], "duracao": (55, 80), "peso": 3,
                  "situacao": "uma partida de videogame no sofá", "rotulo": "jogando videogame", "itens": {"sofa2": "controle"}},
    "leitura": {"n": (2, 2), "lugares": [["leitura0", "leitura1"]], "duracao": (45, 65), "peso": 2,
                "situacao": "o clube do livro no canto de leitura", "rotulo": "no clube do livro"},
    "alongamento": {"n": (3, 5), "lugares": [["alonga0", "alonga1", "alonga2", "alonga3", "alonga4"]], "duracao": (35, 50),
                    "peso": 2, "situacao": "uma pausa para alongar", "rotulo": "alongando", "acao": "alonga"},
    "aniversario": {"n": (3, 4), "lugares": [["reuniao0", "reuniao1", "reuniao2", "reuniao3"]], "duracao": (60, 80),
                    "peso": 0.6, "situacao": "o aniversário de {homenageado}, com bolo na mesa de reunião",
                    "rotulo": "no aniversário de {homenageado}", "itens": {"reuniao0": "lanche", "reuniao1": "xicara", "reuniao3": "lanche"}},
}

# Roteiros da demonstração (e reserva quando a IA não está disponível). Cada fala: (quem, texto); quem é o índice
# do participante. Campos: {a} {b} {c} nomes; {h} o homenageado; e, da personalidade de cada um, frases inteiras:
# {papo_x} um assunto, {reacao_x} a resposta a um assunto, {convida_x}, {comemora_x}, {perde_x}, {agora_x}
# (como está o dia), {bordao_x} e {gosto_x} (um gosto, no meio da frase).
ROTEIROS = {
    "pingue": [
        [(0, "{convida_a}"), (1, "Topo! Mas quem perder lava as xícaras."), (0, "Fechado. Saque meu."), (1, "{comemora_b}"),
         (0, "{perde_a}"), (1, "Melhor de três?")],
        [(0, "{b}, valendo o último pedaço de bolo?"), (1, "Valendo! Mas sem aquele saque torto."),
         (0, "Saque torto é estratégia, viu?"), (0, "{comemora_a}"), (1, "{perde_b}")],
        [(1, "Treinei o saque no fim de semana, {a}."), (0, "Então mostra aí."), (1, "{comemora_b}"), (0, "{perde_a}"),
         (0, "Agora é sério. Dez a dez, ninguém respira.")],
        [(0, "{papo_a}"), (1, "{reacao_b}"), (1, "Agora presta atenção na bola, que eu vou cortar!"), (0, "{perde_a}")],
    ],
    "cafe": [
        [(0, "{papo_a}"), (1, "{reacao_b}"), (1, "{papo_b}"), (0, "{reacao_a}"), (2, "Cheguei no meio, mas já gostei do papo.")],
        [(0, "Café passado agora, quem quer?"), (1, "Eu! Puro, por favor."), (2, "Pra mim com leite, se tiver."),
         (0, "{papo_a}"), (1, "{reacao_b}")],
        [(1, "{a}, como foi o fim de semana?"), (0, "Foi ótimo, teve {gosto_a}. E o seu?"),
         (1, "Tranquilo: {gosto_b} e descanso."), (2, "{papo_c}"), (0, "{reacao_a}")],
        [(0, "Esse cheirinho de café chamou todo mundo, né?"), (1, "Chamou! {papo_b}"), (0, "{reacao_a}"), (2, "{reacao_c}")],
    ],
    "bebedouro": [
        [(0, "E aí, {b}, tudo certo?"), (1, "{agora_b}"), (0, "Bebe água, hein! Hidratação é tudo."), (1, "Pode deixar.")],
        [(1, "{a}, você viu que esfriou hoje?"), (0, "Vi! Até troquei o café por chá."), (1, "Eu ainda não abro mão do café.")],
        [(0, "Pausa pra hidratar."), (1, "Boa ideia. {papo_b}"), (0, "{reacao_a}")],
        [(1, "Falta pouco pro fim de semana, hein?"), (0, "Já tô contando as horas. Tem {gosto_a} me esperando."),
         (1, "Que delícia. Eu quero é descansar.")],
    ],
    "videogame": [
        [(0, "Revanche?"), (1, "Sempre. Hoje eu ganho."), (0, "Ih, caí no buraco de novo!"), (1, "{comemora_b}"),
         (0, "{perde_a}"), (0, "Mais uma fase e eu volto, juro.")],
        [(1, "{a}, pega a moeda ali em cima!"), (0, "Peguei! E a estrela também."), (1, "Isso é que é parceria de fase."),
         (0, "{comemora_a}")],
        [(0, "{convida_a}"), (1, "Só se for no modo cooperativo."), (0, "Fechado. Eu vou na frente."),
         (1, "Cuidado com o chefão!"), (0, "{perde_a}")],
    ],
    "leitura": [
        [(0, "Terminei aquele livro que você indicou, {b}."), (1, "E aí, o que achou?"), (0, "O final me pegou de surpresa."),
         (1, "Né? Tenho outro pra te emprestar.")],
        [(1, "O que você está lendo agora, {a}?"), (0, "Um romance policial. Ainda não sei quem foi."),
         (1, "Aposto no mordomo."), (0, "Sempre o mordomo!")],
        [(0, "Esse capítulo tá bom demais."), (1, "Sem spoiler, hein!"), (0, "Prometo. Só digo que vale a pena.")],
        [(0, "{papo_a}"), (1, "{reacao_b}"), (1, "Anota aí pro próximo clube do livro.")],
    ],
    "alongamento": [
        [(0, "Gente, pausa de cinco minutos pra alongar!"), (1, "Bora, minhas costas agradecem."),
         (2, "Respira fundo… e solta."), (0, "Agora o pescoço, devagar."), (1, "Já me sinto outra pessoa.")],
        [(1, "Muito tempo na cadeira, hora de esticar!"), (0, "Braços pra cima!"), (2, "Ai, estalou tudo."),
         (1, "Mais um minutinho e voltamos.")],
        [(0, "Bora esticar as costas um pouquinho?"), (1, "Bora, que no pingue-pongue eu já perdi hoje."),
         (2, "Ombros pra trás… isso."), (0, "Pronto. Novinhos em folha.")],
    ],
    "aniversario": [
        [(0, "Parabéns, {h}! 🎂"), (1, "Feliz aniversário, {h}! Muita saúde!"), (2, "Valeu, gente! Que surpresa boa!"),
         (1, "Agora corta o bolo!"), (0, "Esse bolo tá ótimo. Quem fez?")],
        [(1, "Vamos cantar? Parabéns pra você…"), (0, "Nesta data querida! 🎉"),
         (2, "Ai, gente, assim eu fico sem graça. Valeu pelo carinho!"), (0, "Discurso! Discurso!"),
         (2, "Discurso curto: vocês são o melhor time. Agora bolo!")],
        [(0, "Surpresa! Parabéns, {h}!"), (2, "Gente! Vocês lembraram!"), (1, "Claro! Até vela tem."),
         (0, "Faz um pedido antes de assoprar."), (2, "Pedido feito. E não conto qual é!")],
    ],
    "ronda": [
        [(0, "Oi, {b}! Tudo certo por aí? Precisa de alguma coisa?"), (1, "{agora_b}"),
         (0, "Não vou te atrapalhar. Qualquer coisa, me chama.")],
        [(0, "Passando pra ver como vocês estão. Como tá aí, {b}?"), (1, "{agora_b}"), (0, "Boa. Qualquer coisa, tô por aqui.")],
        [(0, "{b}, quer um café? Vou passar na copa."), (1, "Aceito! Valeu, {a}."), (0, "Deixa comigo.")],
    ],
}

_estado = {"proximo": 0.0, "ocupado_ate": {}, "ia_hoje": 0, "dia": None}


def _intervalo() -> float:
    base = max(0.5, config.MOMENTOS_MINUTOS) * 60
    return random.uniform(base * 0.6, base * 1.4)


def _livres(db) -> list[Agente]:
    """Quem está de fato livre agora: sem tarefa, sem fila e sem pendência (e fora de outro momento)."""
    from .servicos import Lote
    lote, agora_s = Lote(db), time.monotonic()
    return [a for a in lote.ativos if not a.tarefa_atual_id and not lote.carga.get(a.id) and not lote.pendencias.get(a.id)
            and _estado["ocupado_ate"].get(a.id, 0) < agora_s]


def _trabalhando(db) -> list[Agente]:
    from .servicos import Lote
    lote = Lote(db)
    return [a for a in lote.ativos if a.funcao == "analista" and a.estacao is not None
            and (a.tarefa_atual_id or lote.carga.get(a.id))]


def _escolher_tipo(livres: list[Agente]) -> str | None:
    opcoes = [(t, d["peso"]) for t, d in TIPOS.items() if len(livres) >= d["n"][0]]
    if not opcoes:
        return None
    return random.choices([t for t, _ in opcoes], weights=[p for _, p in opcoes])[0]


def _campos(participantes: list[Agente], homenageado: str = "") -> dict:
    from .servicos import perfil
    c = {"h": homenageado}
    for a, x in zip(participantes, "abc"):
        p = perfil(a)
        estado = "trabalhando" if a.tarefa_atual_id else "aguardando"
        frase = lambda k: random.choice(p.get(k) or PADRAO[k])  # noqa: E731
        c |= {x: a.nome, "papo_" + x: frase("papo"), "reacao_" + x: frase("reacoes"), "convida_" + x: frase("convida"),
              "comemora_" + x: frase("comemora"), "perde_" + x: frase("perde"), "bordao_" + x: bordao(p),
              "gosto_" + x: random.choice(p.get("gostos") or ["um bom café"]),
              "agora_" + x: (p.get("agora") or {}).get(estado) or PADRAO["agora"][estado]}
    return c


def _roteiro(tipo: str, participantes: list[Agente], homenageado: str = "") -> list[tuple[int, str]]:
    campos = _campos(participantes, homenageado)
    falas = []
    for quem, txt in random.choice(ROTEIROS[tipo]):
        if quem >= len(participantes):
            continue
        try:
            falas.append((quem, txt.format(**campos).strip()))
        except (KeyError, IndexError):
            continue
    return falas


def _dialogo_ia(tipo: str, situacao: str, participantes: list[dict], fatos: dict) -> list[tuple[int, str]] | None:
    """A IA escreve o papo, cada um com a sua história e o seu jeito. None se não der (usa o roteiro)."""
    hoje = date.today()
    if _estado["dia"] != hoje:
        _estado.update(dia=hoje, ia_hoje=0)
    if (config.IA_MODO != "gemini" or _estado["ia_hoje"] >= config.MOMENTOS_IA_POR_DIA or uso_ia.bloqueio()
            or random.random() > config.MOMENTOS_IA_CHANCE):
        return None
    _estado["ia_hoje"] += 1
    try:
        falas = ia.momento_conversa(situacao, participantes, fatos)
    except Exception as e:  # sem IA agora: fica o roteiro
        log.info("Momento sem IA (%s): usando roteiro", type(e).__name__)
        return None
    idx = {ia._sem_acento(p["nome"]).strip(): i for i, p in enumerate(participantes)}
    saida = [(idx[ia._sem_acento(f["agente"]).strip()], f["texto"]) for f in falas
             if ia._sem_acento(f.get("agente", "")).strip() in idx and f.get("texto")]
    return saida if len(saida) >= 2 else None


def _fatos(db) -> dict:
    from .servicos import agora_local
    vagas = db.scalars(select(Vaga.titulo).where(Vaga.status == "aberta")).all()
    return agora_local() | {"vagas_abertas": vagas[:4]}


def _planejar(forcar: str | None):
    """Quem participa, onde e de quê. Tudo lido numa sessão curta (a IA escreve o papo depois, fora dela)."""
    from .servicos import atlas, perfil
    with sessao() as db:
        livres = _livres(db)
        sup = atlas(db)
        quem_trabalha = _trabalhando(db)
        ronda = sup is not None and sup in livres and quem_trabalha and (forcar == "ronda" or (not forcar and random.random() < 0.25))
        tipo = "ronda" if ronda else (forcar if forcar in TIPOS else None) if forcar else _escolher_tipo(livres)
        if not tipo:
            return None
        homenageado = ""
        if tipo == "ronda":
            visitados = random.sample(quem_trabalha, min(2, len(quem_trabalha)))
            participantes = [sup] + visitados
            lugares = {sup.id: [f"c{a.estacao}" for a in visitados]}
            situacao = f"{sup.nome}, que coordena o time, passando nas mesas para ver como estão os colegas que trabalham"
            rotulo, duracao = "dando uma volta pelas mesas", 16 * len(visitados)
        else:
            d = TIPOS[tipo]
            if len(livres) < d["n"][0]:
                return None
            n = random.randint(d["n"][0], min(d["n"][1], len(livres)))
            participantes = random.sample(livres, n)
            nos = random.choice(d["lugares"])
            lugares = {a.id: nos[i] for i, a in enumerate(participantes)}
            if tipo == "aniversario":
                homenageado = participantes[2].nome
            situacao = d["situacao"].format(homenageado=homenageado)
            rotulo = d["rotulo"].format(homenageado=homenageado)
            duracao = random.randint(*d["duracao"])
        roteiro = _roteiro(tipo, participantes, homenageado)
        pessoas = [{"nome": a.nome, **{k: perfil(a).get(k) for k in ("historia", "jeito", "gostos", "manias")}}
                   for a in participantes]
        return {"tipo": tipo, "ids": [a.id for a in participantes], "lugares": lugares, "situacao": situacao,
                "rotulo": rotulo, "duracao": duracao, "roteiro": roteiro, "pessoas": pessoas, "fatos": _fatos(db)}


def talvez_um_momento(forcar: str | None = None) -> dict | None:
    """Chamado a cada rodada da supervisão: de tempos em tempos, junta quem está livre num momento."""
    if config.MOMENTOS_MINUTOS <= 0 and not forcar:
        return None
    agora_s = time.monotonic()
    if not forcar and agora_s < _estado["proximo"]:
        return None
    _estado["proximo"] = agora_s + _intervalo()
    plano = _planejar(forcar)
    if not plano:
        return None
    tipo = plano["tipo"]
    falas = _dialogo_ia(tipo, plano["situacao"], plano["pessoas"], plano["fatos"]) or plano["roteiro"]
    if not falas:
        return None
    from .servicos import Lote
    with sessao() as db:
        lote = Lote(db)
        por_id = {a.id: a for a in lote.ativos}
        participantes = [por_id.get(i) for i in plano["ids"]]
        # enquanto a IA escrevia, alguém pode ter recebido trabalho (ou saído): o momento fica para outra hora
        livres = {a.id for a in _livres(db)}
        if None in participantes or any(i not in livres for i in plano["ids"][:1 if tipo == "ronda" else None]):
            return None
        duracao = max(plano["duracao"], 8 + len(falas) * 6)
        momento = {"id": f"{tipo}-{int(time.time() * 1000)}", "tipo": tipo, "duracao": duracao, "rotulo": plano["rotulo"],
                   "lugares": {str(k): v for k, v in plano["lugares"].items()},
                   "itens": TIPOS.get(tipo, {}).get("itens", {}), "acao": TIPOS.get(tipo, {}).get("acao")}
        for i, (quem, texto) in enumerate(falas):
            falante = participantes[quem]
            ouvinte = participantes[1 if quem == 0 else 0]
            eventos.registrar(db, "conversa", texto[:220], falante.id, destino_id=ouvinte.id, momento=momento | {"ordem": i},
                              de=falante.nome, para=ouvinte.nome)
    for i in plano["ids"][:1] if tipo == "ronda" else plano["ids"]:
        _estado["ocupado_ate"][i] = time.monotonic() + duracao
    log.info("Momento %s com %s", tipo, ", ".join(p["nome"] for p in plano["pessoas"]))
    return momento
