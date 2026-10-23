"""IA dos agentes: análise de currículo × vaga, consulta entre colegas, revisão, sugestões do Atlas,
chat individual (com streaming) e conversa em grupo. Gemini ou modo demonstração.

No modo demonstração nada sai do computador: as respostas vêm de regras locais sobre os
registros reais e vêm marcadas como tal. Os "pensamentos" dos agentes são sempre derivados
do que a leitura encontrou — nunca inventados para enfeitar o escritório.
"""
import json
import logging
import random
import re
import threading
import time
import unicodedata
from typing import Callable, Iterator

from pydantic import BaseModel

from . import config, uso_ia
from .catalogo import CATEGORIAS_DISPONIVEIS
from .personalidade import bordao
from .uso_ia import IAIndisponivel  # noqa: F401  (reexportado para quem chama)

log = logging.getLogger("hub.ia")

_cliente = None
_semaforo = threading.Semaphore(max(1, config.IA_CONCORRENCIA))
Pensar = Callable[[str], None] | None


def cliente_gemini():
    global _cliente
    if _cliente is None:
        from google import genai
        _cliente = genai.Client(api_key=config.GEMINI_API_KEY)
    return _cliente


def sem_bloqueio():
    from google.genai import types
    cats = ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]
    return [types.SafetySetting(category=c, threshold="BLOCK_NONE") for c in cats]


# --------------------------------------------------------------------------- esquemas
class RequisitoAtendido(BaseModel):
    requisito: str
    evidencia: str


class AnaliseSchema(BaseModel):
    compatibilidade: str
    recomendado: bool
    avaliacao_humana: bool
    resumo_profissional: str
    experiencias_relevantes: list[str]
    competencias: list[str]
    requisitos_atendidos: list[RequisitoAtendido]
    a_verificar: list[str]
    incompatibilidades: list[str]
    justificativa: str
    pensamentos: list[str]
    pedir_ajuda: bool
    ajuda_especialidade: str
    ajuda_pergunta: str


class ConsultaSchema(BaseModel):
    resposta: str
    evidencias: list[str]
    parecer: str  # reforca | enfraquece | neutro
    indicar_vaga: str  # título exato de uma vaga sua que combina melhor, ou ""
    pensamentos: list[str]


class SugestaoVagaSchema(BaseModel):
    vaga_id: int
    confianca: str
    motivo: str
    evidencias: list[str]
    lacunas: list[str]
    perfil_indicado: str


class SugestaoSetoresSchema(BaseModel):
    setores: list[str]
    requisitos: list[str]


class Fala(BaseModel):
    agente: str
    texto: str


class ConversaSchema(BaseModel):
    falas: list[Fala]


REGRAS_ETICAS = """
- Você APOIA o recrutador humano. Nunca decida contratação; sua recomendação é uma sugestão fundamentada.
- Use SOMENTE informações presentes no currículo e na descrição da vaga. Não invente dados.
- Quando não houver informação suficiente para avaliar um requisito, coloque-o em "a_verificar".
- Em "requisitos_atendidos.evidencia", copie um trecho LITERAL e curto (até 20 palavras) do currículo que comprove o requisito.
- NÃO considere nem mencione características protegidas: idade, gênero, cor/raça, etnia, religião, estado civil,
  gravidez ou filhos, orientação sexual, deficiência, aparência ou foto, origem (LGPD art. 6º, IX; Lei 9.029/95).
  Exceção: em vaga exclusiva PCD, apenas registre que o próprio candidato declarou a condição, sem avaliá-la.
- Localização e disponibilidade só contam quando forem requisitos da vaga.
- compatibilidade: ALTA = atende os requisitos principais com evidência; MEDIA = atende parte ou há lacunas
  relevantes; BAIXA = não atende requisitos obrigatórios.
- Estudo, curso em andamento, "conhecimentos básicos", projetos pessoais e interesse NÃO equivalem a experiência
  profissional. Se o título da vaga indica um nível que pede experiência (analista, especialista, técnico,
  supervisor, coordenador, gerente, sênior) e o currículo não mostra experiência profissional na área, a
  compatibilidade é BAIXA e o ponto vai para "incompatibilidades" (ou "a_verificar", se o currículo for ambíguo).
- avaliacao_humana = true quando o perfil é promissor mas há dúvida importante que só uma pessoa pode resolver.
"""

REGRAS_FALA = """
- Fale em português, no seu jeito de falar, em frases curtas. Sem markdown.
- Refira-se ao candidato pelo rótulo informado (primeiro nome e inicial) ou como "o candidato"; nunca pelo nome
  completo, e-mail ou telefone. Não comente características protegidas.
"""


def _sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", (s or "").lower()) if unicodedata.category(c) != "Mn")


def _voz(agente: dict) -> str:
    p = agente.get("perfil") or {}
    if not p:
        return ""
    return (f'Sua personalidade: {p.get("resumo", "")}. Jeito de falar: {p.get("tom", "")}. '
            f'Bordões que você às vezes usa: {", ".join(p.get("bordoes", [])[:3])}.')


# --------------------------------------------------------------------------- chamada ao Gemini (econômica)
# O "pensamento" do modelo é cobrado como saída: pedimos o mínimo e recuamos se o modelo não aceitar o ajuste.
_NIVEIS_PENSAMENTO = ("minimal", "low", "medium", "high")
_nivel_aceito: dict[str, str | None] = {}  # modelo -> ajuste que a API aceitou (None = padrão do modelo)


def _pensamento(modelo: str, nivel: str | None):
    from google.genai import types
    if not nivel:
        return None
    if modelo.startswith("gemini-2.5"):  # família 2.5: orçamento em tokens
        if "pro" in modelo:
            return types.ThinkingConfig(thinking_budget=128)  # o 2.5 Pro não desliga o raciocínio
        return types.ThinkingConfig(thinking_budget=0 if nivel == "minimal" else 1024)
    return types.ThinkingConfig(thinking_level=nivel.upper())  # 3.x em diante: nível


def _niveis_para(modelo: str) -> list[str | None]:
    base = config.GEMINI_PENSAMENTO
    if base not in _NIVEIS_PENSAMENTO:
        return [None]
    if modelo in _nivel_aceito:
        return [_nivel_aceito[modelo]]
    i = _NIVEIS_PENSAMENTO.index(base)
    return list(_NIVEIS_PENSAMENTO[i:i + 2]) + [None]  # ex.: minimal -> low -> padrão do modelo


def _config(modelo, sistema, schema, temperatura, max_saida, nivel):
    from google.genai import types
    cfg = dict(system_instruction=sistema, temperature=temperatura, max_output_tokens=max_saida,
               safety_settings=sem_bloqueio())
    if schema is not None:
        cfg.update(response_mime_type="application/json", response_schema=schema)
    pens = _pensamento(modelo, nivel)
    if pens:
        cfg["thinking_config"] = pens
    return types.GenerateContentConfig(**cfg)


def _rejeita_pensamento(e) -> bool:
    return getattr(e, "code", None) == 400 and "think" in str(e).lower()


def _gerar(finalidade: str, sistema: str, conteudos: list, schema=None, temperatura: float = 0.2,
           max_saida: int = 1024, modelo: str | None = None, forcar: bool = False):
    """Uma chamada ao Gemini: pensamento mínimo, teto de saída, registro de tokens/custo e pausa automática
    quando o provedor recusa. `forcar` ignora a pausa (usado pelo teste de conexão)."""
    modelo = modelo or config.GEMINI_MODELO_AGENTES
    motivo = None if forcar else uso_ia.bloqueio()
    if motivo:
        raise IAIndisponivel(motivo)
    niveis = _niveis_para(modelo)
    with _semaforo:
        tentativa = 0
        while True:
            nivel = niveis[0]
            try:
                r = cliente_gemini().models.generate_content(
                    model=modelo, contents=conteudos, config=_config(modelo, sistema, schema, temperatura, max_saida, nivel))
                _nivel_aceito.setdefault(modelo, nivel)
                uso_ia.registrar(finalidade, modelo, getattr(r, "usage_metadata", None))
                return r
            except Exception as e:
                if _rejeita_pensamento(e) and len(niveis) > 1:
                    log.info("O modelo %s não aceitou o pensamento '%s'; usando o próximo ajuste", modelo, nivel)
                    niveis = niveis[1:]
                    continue
                if uso_ia.falha_provedor(e):
                    raise IAIndisponivel(uso_ia.bloqueio() or uso_ia.classificar(e)) from e
                tentativa += 1
                limite = getattr(e, "code", None) == 429 or "429" in str(e)
                log.warning("Gemini (%s): tentativa %d/3 falhou (%s)", finalidade, tentativa, type(e).__name__)
                if tentativa >= 3:
                    raise
                time.sleep(5 * tentativa if limite else 2)


def _json_de(r) -> dict:
    txt = getattr(r, "text", None) or ""
    try:
        return json.loads(txt)
    except (json.JSONDecodeError, TypeError):
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        fim = getattr((getattr(r, "candidates", None) or [None])[0], "finish_reason", None)
        raise ValueError(f"resposta da IA sem JSON válido (fim: {fim})")


def _gemini_json(finalidade, sistema, conteudos, schema, temperatura=0.2, max_saida=1500):
    return _json_de(_gerar(finalidade, sistema, conteudos, schema, temperatura, max_saida))


def _js(obj) -> str:
    """JSON compacto: menos tokens."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def testar() -> dict:
    """Diagnóstico: uma chamada mínima ao modelo configurado (ignora a pausa; se der certo, encerra a pausa)."""
    if config.IA_MODO != "gemini":
        return {"ok": True, "modo": "demo", "mensagem": "Modo demonstração: nenhuma IA externa é chamada."}
    if not config.GEMINI_API_KEY:
        return {"ok": False, "modo": "gemini", "mensagem": "GEMINI_API_KEY não está definida."}
    inicio = time.monotonic()
    try:
        r = _gerar("teste", "Responda exatamente: OK", ["Teste de conexão."], temperatura=0, max_saida=32, forcar=True)
        entrada, saida, pens, _ = uso_ia._tokens(getattr(r, "usage_metadata", None))
        return {"ok": True, "modo": "gemini", "modelo": config.GEMINI_MODELO_AGENTES,
                "ms": int((time.monotonic() - inicio) * 1000), "resposta": (getattr(r, "text", "") or "").strip()[:40],
                "tokens": {"entrada": entrada, "saida": saida, "pensamento": pens},
                "pensamento": _nivel_aceito.get(config.GEMINI_MODELO_AGENTES) or "padrão do modelo",
                "mensagem": "A IA respondeu normalmente."}
    except IAIndisponivel as e:
        return {"ok": False, "modo": "gemini", "modelo": config.GEMINI_MODELO_AGENTES, "mensagem": str(e)}
    except Exception as e:
        return {"ok": False, "modo": "gemini", "modelo": config.GEMINI_MODELO_AGENTES,
                "mensagem": uso_ia.classificar(e) or f"{type(e).__name__}: {str(e)[:200]}"}


def _curriculo(texto, arquivo) -> list:
    from google.genai import types
    if texto:
        return ["CURRÍCULO (texto extraído):\n" + texto[:30000]]
    if arquivo:
        return ["CURRÍCULO (arquivo anexo):", types.Part.from_bytes(data=arquivo[0], mime_type=arquivo[1])]
    raise ValueError("Currículo sem conteúdo legível")


def _normalizar_analise(r: dict) -> dict:
    r["compatibilidade"] = str(r.get("compatibilidade", "BAIXA")).upper().replace("É", "E")
    if r["compatibilidade"] not in ("ALTA", "MEDIA", "BAIXA"):
        r["compatibilidade"] = "BAIXA"
    r["pensamentos"] = [str(p)[:120] for p in (r.get("pensamentos") or [])][:4]
    return r


def _ritmo(pensamentos: list[str], duracao: float, ao_pensar: Pensar):
    """Modo demonstração: espalha os pensamentos (reais) ao longo da leitura simulada."""
    passos = len(pensamentos) + 1
    for p in pensamentos:
        time.sleep(duracao / passos)
        if ao_pensar:
            ao_pensar(p)
    time.sleep(duracao / passos)


def _emitir(pensamentos, ao_pensar: Pensar):
    for p in pensamentos or []:
        if ao_pensar and p:
            ao_pensar(p)


# --------------------------------------------------------------------------- utilidades da demonstração
_PARADAS = set("a o e de da do das dos em no na nos nas para com por um uma ou ao aos se que ter tem ser como "
               "experiencia experiencias conhecimento conhecimentos minimo desejavel obrigatorio".split())


def _tokens(s):
    return [w for w in re.findall(r"[a-z0-9+#-]{3,}", _sem_acento(s)) if w not in _PARADAS]


def _linhas(texto):
    return [l.strip(" •-\t") for l in (texto or "").splitlines() if len(l.strip()) > 3]


def _avaliar_requisitos(linhas, reqs):
    atendidos, verificar = [], []
    for req in reqs or []:
        tk = _tokens(req)
        if not tk:
            continue
        melhor, nota = None, 0.0
        for l in linhas:
            lt = _tokens(l)
            hits = sum(1 for t in tk if any(x.startswith(t[:5]) or t.startswith(x[:5]) for x in lt))
            if hits / len(tk) > nota:
                melhor, nota = l, hits / len(tk)
        if melhor and nota >= 0.5:
            atendidos.append({"requisito": req, "evidencia": melhor[:160]})
        else:
            verificar.append(f"{req}: não consta no currículo")
    return atendidos, verificar


def _compat(atendidos, verificar):
    taxa = len(atendidos) / max(1, len(atendidos) + len(verificar))
    return "ALTA" if taxa >= 0.66 else "MEDIA" if taxa >= 0.34 else "BAIXA"


def termos_da_area(setores) -> list[str]:
    """Palavras que indicam experiência numa área (mesmas regras da triagem de demonstração)."""
    from .triagem import _PALAVRAS
    termos = [re.escape(_sem_acento(s)) for s in setores or [] if s not in ("OUTROS",)]
    for cat, chaves in _PALAVRAS:
        if cat in (setores or []):
            termos.extend(chaves)
    return termos


def _tem_termos(texto_norm, termos):
    return next((t for t in termos if re.search(t, texto_norm)), None)


def _curto(s, n=70):
    s = (s or "").strip()
    return s if len(s) <= n else s[:n - 1].rstrip() + "…"


# =========================================================================== análise
def analisar(agente: dict, vaga: dict, curriculo_texto: str | None, arquivo: tuple[bytes, str] | None,
             especialidades: list[dict], setor_triagem: str | None, colegas: list[dict] = (),
             evitar: set[str] = frozenset(), rotulo: str = "o candidato", ao_pensar: Pensar = None) -> tuple[dict, str]:
    """Retorna (resultado, modelo_usado). `ao_pensar` recebe os pensamentos do agente durante a leitura."""
    if config.IA_MODO == "gemini":
        r = _analisar_gemini(agente, vaga, curriculo_texto, arquivo, especialidades, colegas, evitar, rotulo)
        _emitir(r.get("pensamentos"), ao_pensar)
        return r, config.GEMINI_MODELO_AGENTES
    r = _analisar_demo(agente, vaga, curriculo_texto, especialidades, setor_triagem, colegas, evitar, rotulo)
    _ritmo(r["pensamentos"], config.DEMO_DURACAO_ANALISE, ao_pensar)  # só na demonstração, para ser observável
    return r, "demo-heuristica"


def _analisar_gemini(agente, vaga, texto, arquivo, especialidades, colegas, evitar, rotulo):
    proprio = agente.get("especialidade_codigo")
    outras = [f'{e["codigo"]} ({e["nome"]})' for e in especialidades if e["codigo"] not in ({proprio} | set(evitar))]
    disponiveis = ", ".join(f'{c["nome"]} ({c["especialidade"]})' for c in colegas) or "nenhum"
    sistema = (f'Você é {agente["nome"]}, agente de IA da equipe de recrutamento {config.DA_EMPRESA}, especialista em '
               f'{agente.get("especialidade") or "recrutamento"}. {_voz(agente)}\n'
               f"Analise o currículo para a vaga indicada e responda em português, de forma concisa.\n{REGRAS_ETICAS}\n"
               f'- "pensamentos": 2 a 4 frases curtas (até 90 caracteres), em primeira pessoa e no seu jeito de falar, '
               f'mostrando o raciocínio da leitura (o que procurou e o que encontrou). Candidato = "{rotulo}". '
               "Sem nome completo, contato ou características protegidas.\n"
               "- \"pedir_ajuda\": true SOMENTE se houver uma dúvida concreta sobre o currículo que um colega de outra "
               "especialidade responde melhor (termo técnico, certificação, experiência de outra área, ou se o perfil "
               "parece servir melhor a outra área). Informe \"ajuda_especialidade\" (um código entre: "
               f'{", ".join(outras) or "nenhum"}) e "ajuda_pergunta" (objetiva, até 200 caracteres, dirigida ao colega). '
               f"Colegas ativos agora: {disponiveis}. Se não houver dúvida, pedir_ajuda = false.")
    conteudos = ["VAGA:\n" + _js(vaga)] + _curriculo(texto, arquivo)
    return _normalizar_analise(_gemini_json("analise", sistema, conteudos, AnaliseSchema, max_saida=2500))


def _analisar_demo(agente, vaga, texto, especialidades, setor, colegas, evitar, rotulo):
    linhas = _linhas(texto)
    atendidos, verificar = _avaliar_requisitos(linhas, vaga.get("requisitos") or [])
    compat = _compat(atendidos, verificar)
    total = max(1, len(atendidos) + len(verificar))
    comp_linha = next((l for l in linhas if _sem_acento(l).startswith(("competencias", "habilidades"))), "")
    competencias = [c.strip() for c in re.split(r"[·,;|]", comp_linha.split(":", 1)[-1]) if c.strip()][:6] if comp_linha else []
    exp = [l for l in linhas if re.search(r"(19|20)\d{2}\s*[–-]", l)][:3]
    sem_contato = [l for l in linhas[1:] if "@" not in l and not re.search(r"\d{4,}[-\s]?\d{3,}", l)]
    i_res = next((i for i, l in enumerate(sem_contato) if _sem_acento(l).startswith(("resumo", "perfil", "objetivo"))), None)
    resumo = (sem_contato[i_res + 1] if i_res is not None and i_res + 1 < len(sem_contato)
              else next((l for l in sem_contato if len(l) > 40), ""))

    # dúvidas que um colega de outra área responde melhor
    proprio = agente.get("especialidade_codigo")
    pedir, esp_ajuda, pergunta = False, "", ""
    if setor:
        alvo = next((e for e in especialidades if setor in e.get("setores", [])), None)
        if alvo and alvo["codigo"] not in ({proprio} | set(evitar)):
            pedir, esp_ajuda = True, alvo["codigo"]
            pergunta = (f"a triagem colocou {rotulo} em {setor}. Pela sua experiência em {alvo['nome'].lower()}, "
                        f"isso conta para “{vaga['titulo']}”?")
    if not pedir and compat != "ALTA":
        txt = _sem_acento(texto)
        for col in colegas:
            if col["especialidade_codigo"] in ({proprio} | set(evitar)) or not col.get("vagas"):
                continue
            achado = _tem_termos(txt, termos_da_area(col.get("setores")))
            if achado:
                pedir, esp_ajuda = True, col["especialidade_codigo"]
                pergunta = (f"o currículo de {rotulo} tem bastante coisa de {col['especialidade'].lower()}. "
                            "Combina com alguma vaga sua?")
                break

    pensamentos = [f"Lendo o currículo de {rotulo} para “{_curto(vaga['titulo'], 40)}”…"]
    pensamentos += [f"✔ {_curto(a['requisito'], 50)}: está no currículo." for a in atendidos[:2]]
    pensamentos += [f"✘ {_curto(v.split(':')[0], 50)}: não encontrei." for v in verificar[:1]]
    if pedir:
        pensamentos.append("Hmm… melhor ouvir um colega de outra área.")
    return {
        "compatibilidade": compat,
        "recomendado": compat == "ALTA",
        "avaliacao_humana": compat == "MEDIA",
        "resumo_profissional": resumo[:240],
        "experiencias_relevantes": exp,
        "competencias": competencias,
        "requisitos_atendidos": atendidos,
        "a_verificar": verificar,
        "incompatibilidades": [],
        "justificativa": (f"[análise heurística — modo demonstração] {len(atendidos)} de {total} requisitos "
                          f"encontrados no currículo." + (" Recomendo avaliação do recrutador." if compat == "MEDIA" else "")),
        "pensamentos": pensamentos,
        "pedir_ajuda": pedir,
        "ajuda_especialidade": esp_ajuda,
        "ajuda_pergunta": pergunta,
    }


# =========================================================================== consulta entre colegas
def consultar(colega: dict, solicitante: dict, vaga: dict, texto: str | None, arquivo, pergunta: str,
              preliminar: dict, rotulo: str = "o candidato", ao_pensar: Pensar = None) -> tuple[dict, str]:
    """O colega lê o currículo e responde à dúvida; pode indicar o candidato para uma vaga dele."""
    if config.IA_MODO == "gemini":
        r = _consultar_gemini(colega, solicitante, vaga, texto, arquivo, pergunta, preliminar, rotulo)
        _emitir(r.get("pensamentos"), ao_pensar)
        return r, config.GEMINI_MODELO_AGENTES
    r = _consultar_demo(colega, vaga, texto, preliminar, rotulo)
    _ritmo(r["pensamentos"], config.DEMO_DURACAO_ANALISE / 2, ao_pensar)
    return r, "demo-heuristica"


def _consultar_gemini(colega, solicitante, vaga, texto, arquivo, pergunta, preliminar, rotulo):
    suas = [v["titulo"] for v in colega.get("vagas", [])]
    sistema = (f'Você é {colega["nome"]}, agente de IA de recrutamento {config.DA_EMPRESA}, especialista em {colega.get("especialidade")}. '
               f'{_voz(colega)}\nSeu colega {solicitante["nome"]} ({solicitante.get("especialidade")}) está analisando '
               f'o currículo de "{rotulo}" para a vaga "{vaga["titulo"]}" e pediu sua ajuda. Leia o currículo e responda '
               "do ponto de vista da sua especialidade.\n"
               f"{REGRAS_ETICAS}{REGRAS_FALA}"
               '- "resposta": até 3 frases, dirigidas ao colega, no seu jeito de falar.\n'
               '- "evidencias": trechos LITERAIS e curtos do currículo que sustentam a resposta.\n'
               '- "parecer": "reforca" se o que você viu aumenta a compatibilidade com a vaga do colega, '
               '"enfraquece" se diminui, "neutro" caso contrário.\n'
               f'- "indicar_vaga": o título EXATO de uma vaga sua ({"; ".join(suas) or "você não tem vagas"}) se o '
               'candidato combinar claramente com ela; senão "".\n'
               '- "pensamentos": 1 ou 2 frases curtas sobre a sua leitura.')
    conteudos = [f"PERGUNTA DE {solicitante['nome']}: {pergunta}",
                 "VAGA DO COLEGA:\n" + _js(vaga),
                 "ANÁLISE PRELIMINAR DO COLEGA:\n" + _js(_resumo_preliminar(preliminar)),
                 "SUAS VAGAS:\n" + _js(colega.get("vagas", []))] + _curriculo(texto, arquivo)
    r = _gemini_json("consulta", sistema, conteudos, ConsultaSchema, temperatura=0.3, max_saida=900)
    r["parecer"] = r.get("parecer") if r.get("parecer") in ("reforca", "enfraquece", "neutro") else "neutro"
    r["pensamentos"] = [str(p)[:120] for p in (r.get("pensamentos") or [])][:2]
    return r


def _resumo_preliminar(p: dict) -> dict:
    return {k: p.get(k) for k in ("compatibilidade", "requisitos_atendidos", "a_verificar", "incompatibilidades", "justificativa")}


def _consultar_demo(colega, vaga, texto, preliminar, rotulo):
    linhas = _linhas(texto)
    termos = termos_da_area(colega.get("setores"))
    # prefere linhas de experiência (com período) ao cabeçalho do currículo
    achadas = sorted((l for l in linhas[1:] if _tem_termos(_sem_acento(l), termos)),
                     key=lambda l: (not re.search(r"(19|20)\d{2}", l), "·" in l))[:3]
    # algum requisito que o colega não comprovou aparece nas linhas da minha área?
    reforco = None
    pendentes = [x.split(":")[0].strip() for x in preliminar.get("a_verificar", [])]
    for req in pendentes:
        at, _ = _avaliar_requisitos(achadas, [req])
        if at:
            reforco = at[0]
            break
    # alguma vaga minha combina melhor?
    indicar, melhor = "", None
    for v in colega.get("vagas", []):
        at, ver = _avaliar_requisitos(linhas, v.get("requisitos") or [])
        taxa = len(at) / max(1, len(at) + len(ver))
        if taxa >= 0.5 and (not melhor or taxa > melhor[0]):
            melhor = (taxa, v["titulo"], len(at), len(at) + len(ver))
    partes = [f"vi experiência da minha área: “{_curto(achadas[0], 80)}”." if achadas
              else "não achei nada específico da minha área."]
    if reforco:
        partes.append(f"Isso comprova “{reforco['requisito']}”.")
    if melhor:
        indicar = melhor[1]
        partes.append(f"E bate {melhor[2]} de {melhor[3]} requisitos da minha vaga “{indicar}” — vou incluir lá também.")
    pensamentos = [f"Lendo o currículo de {rotulo} com olhos de {colega.get('especialidade', 'especialista').lower()}…"]
    if achadas:
        pensamentos.append(f"Achei: “{_curto(achadas[0], 50)}”.")
    return {"resposta": " ".join(partes), "evidencias": [reforco["evidencia"]] if reforco else achadas[:1],
            "parecer": "reforca" if reforco else "neutro", "indicar_vaga": indicar,
            "reforca_requisito": reforco["requisito"] if reforco else "", "pensamentos": pensamentos}


def revisar(agente: dict, vaga: dict, texto: str | None, arquivo, preliminar: dict, consulta: dict,
            rotulo: str = "o candidato", ao_pensar: Pensar = None) -> tuple[dict, str]:
    """O analista revisa a própria análise com a contribuição do colega."""
    if config.IA_MODO == "gemini":
        r = _revisar_gemini(agente, vaga, texto, arquivo, preliminar, consulta, rotulo)
        _emitir(r.get("pensamentos"), ao_pensar)
        return r, config.GEMINI_MODELO_AGENTES
    r = _revisar_demo(preliminar, consulta)
    _ritmo(r["pensamentos"], config.DEMO_DURACAO_ANALISE / 3, ao_pensar)
    return r, "demo-heuristica"


def _revisar_gemini(agente, vaga, texto, arquivo, preliminar, consulta, rotulo):
    # Economia: o currículo não é reenviado; a análise preliminar já traz as evidências e o colega traz as dele.
    sistema = (f'Você é {agente["nome"]}, agente de IA de recrutamento {config.DA_EMPRESA}, especialista em '
               f'{agente.get("especialidade") or "recrutamento"}. {_voz(agente)}\n'
               f'Você fez uma análise preliminar e consultou {consulta.get("colega")} ({consulta.get("especialidade")}). '
               "Revise a análise considerando a contribuição do colega e devolva-a completa, no mesmo formato. "
               "Só use como evidência trechos que já estão na sua análise ou na resposta do colega.\n"
               f"{REGRAS_ETICAS}"
               f'- "pensamentos": 1 ou 2 frases curtas sobre o que mudou (ou não) com a ajuda. Candidato = "{rotulo}".\n'
               '- "pedir_ajuda" deve ser false.')
    conteudos = ["VAGA:\n" + _js(vaga),
                 "SUA ANÁLISE PRELIMINAR:\n" + _js({k: v for k, v in preliminar.items() if k != "pensamentos"}),
                 "CONSULTA AO COLEGA:\n" + _js({k: consulta.get(k) for k in ("pergunta", "resposta", "evidencias", "parecer")})]
    r = _normalizar_analise(_gemini_json("revisao", sistema, conteudos, AnaliseSchema, max_saida=2500))
    r["pedir_ajuda"] = False
    return r


def _revisar_demo(preliminar, consulta):
    r = json.loads(json.dumps(preliminar))  # cópia
    req = (consulta or {}).get("reforca_requisito")
    colega = (consulta or {}).get("colega", "o colega")
    if req and consulta.get("evidencias"):
        r["a_verificar"] = [x for x in r.get("a_verificar", []) if x.split(":")[0].strip() != req]
        r["requisitos_atendidos"] = r.get("requisitos_atendidos", []) + [{"requisito": req, "evidencia": consulta["evidencias"][0]}]
        compat = _compat(r["requisitos_atendidos"], r["a_verificar"])
        r.update(compatibilidade=compat, recomendado=compat == "ALTA", avaliacao_humana=compat == "MEDIA")
        n = len(r["requisitos_atendidos"])
        r["justificativa"] = (f"[análise heurística — modo demonstração] {n} de {n + len(r['a_verificar'])} requisitos "
                              f"encontrados no currículo, um deles com a ajuda de {colega}.")
        r["pensamentos"] = [f"Com a dica de {colega}, “{_curto(req, 40)}” está comprovado."]
    else:
        r["pensamentos"] = [f"Conversei com {colega}; mantenho a minha leitura."]
    r["pedir_ajuda"] = False
    return r


# =========================================================================== sugestões do Atlas
_EXPERIENTE = re.compile(r"\b(analista|especialista|tecnico|supervisor|supervisora|coordenador|coordenadora|gerente|"
                         r"senior|sr|lider|consultor|consultora|engenheiro|engenheira|desenvolvedor|desenvolvedora)\b")
_ENTRADA = re.compile(r"\b(assistente|auxiliar|ajudante|junior|jr|trainee|estagio|estagiario|estagiaria|aprendiz|"
                      r"atendente|operador|operadora)\b")


def nivel_da_vaga(titulo: str) -> str | None:
    """'entrada' (aceita estudo e pouca experiência), 'experiente' (pede experiência na área) ou None."""
    t = _sem_acento(titulo)
    return "entrada" if _ENTRADA.search(t) else "experiente" if _EXPERIENTE.search(t) else None


def _conferir_sugestao(r: dict, vagas: list[dict], pasta: str | None) -> dict | None:
    """Trava que não depende só da IA: sem evidência de experiência profissional não há sugestão (fica BAIXA)."""
    if not r:
        return None
    vaga = next((v for v in vagas if v["id"] == r.get("vaga_id")), None)
    r["confianca"] = str(r.get("confianca") or "BAIXA").upper().replace("É", "E")
    if r["confianca"] not in ("ALTA", "MEDIA", "BAIXA"):
        r["confianca"] = "BAIXA"
    r["evidencias"] = [str(x)[:200] for x in (r.get("evidencias") or []) if str(x).strip()][:4]
    r["lacunas"] = [str(x)[:200] for x in (r.get("lacunas") or []) if str(x).strip()][:4]
    r["perfil_indicado"] = str(r.get("perfil_indicado") or "")[:160]
    if not vaga:
        r.update(vaga_id=0, confianca="BAIXA")
        return r
    if r["confianca"] != "BAIXA" and not r["evidencias"]:
        r["confianca"] = "BAIXA"
        r["lacunas"].insert(0, "sem experiência profissional comprovada no currículo para esta vaga")
    return r


def sugerir_vaga(texto: str, vagas: list[dict], pasta: str | None = None, anos: int | None = None) -> dict | None:
    """O Atlas, como recrutador, diz qual vaga aberta o currículo sustenta: {vaga_id (0 = nenhuma), confianca,
    motivo, evidencias, lacunas, perfil_indicado}. Experiência profissional vale; estudo e interesse, não."""
    if not texto or not vagas:
        return None
    vagas = [{**v, "nivel": nivel_da_vaga(v.get("titulo", ""))} for v in vagas]
    if config.IA_MODO == "gemini":
        sistema = (
            f"Você é Atlas, coordenador da equipe de recrutamento {config.DA_EMPRESA}, com olhar de recrutador experiente. "
            "Diga se o currículo SUSTENTA alguma das vagas abertas (use o id) ou se nenhuma combina (vaga_id 0).\n"
            "Critérios, nesta ordem:\n"
            "1. Experiência PROFISSIONAL na função ou na área da vaga é o que mais pesa. Estudo, curso em andamento, "
            "\"conhecimentos básicos\", projetos pessoais e interesse NÃO são experiência.\n"
            "2. O título indica o nível. nivel=experiente (analista, especialista, técnico, supervisor, coordenador, "
            "gerente, sênior...) exige experiência profissional comprovada na área; nivel=entrada (assistente, auxiliar, "
            "júnior, trainee, estágio, aprendiz...) aceita formação ou estudo com pouca experiência.\n"
            "3. A pasta da triagem é a área principal do candidato. Vaga de outra área só se o currículo mostrar "
            "experiência profissional nela.\n"
            "4. Se os requisitos da vaga forem vagos, julgue pelo título e pela área e diga isso no motivo.\n"
            "confianca: ALTA = experiência profissional na função (ou muito próxima) que atende os principais requisitos; "
            "MEDIA = experiência profissional relacionada, com lacunas que uma entrevista resolve; BAIXA = só estudo ou "
            "interesse, área diferente sem experiência, ou abaixo do nível da vaga. Na dúvida, BAIXA.\n"
            "evidencias: até 3 trechos LITERAIS e curtos do currículo que mostrem a experiência profissional que sustenta "
            "a vaga (vazio se não houver). lacunas: até 3 coisas que faltam para a vaga. perfil_indicado: em poucas "
            "palavras, para que tipo de vaga a experiência real do candidato serve (mesmo que não esteja aberta). "
            "motivo: até 2 frases, sem nome do candidato. Use só o que está no currículo e nas vagas; não considere "
            "idade, gênero, cor/raça, religião, estado civil, filhos, deficiência, aparência ou origem.")
        sobre = f"PASTA DA TRIAGEM: {pasta or 'não informada'}" + (f" · ANOS DE EXPERIÊNCIA (triagem): {anos}" if anos is not None else "")
        r = _gemini_json("encaminhamento", sistema, [sobre, "VAGAS ABERTAS:\n" + _js(vagas), "CURRÍCULO:\n" + texto[:12000]],
                         SugestaoVagaSchema, max_saida=600)
        return _conferir_sugestao(r, vagas, pasta)
    # demonstração: termos em comum, com as mesmas regras de nível e de área
    cv = set(_tokens(texto))
    melhor = None
    for v in vagas:
        alvo = set(_tokens(" ".join([v.get("titulo", ""), v.get("descricao", "")] + list(v.get("requisitos") or []))))
        if not alvo:
            continue
        comuns = {t for t in alvo if any(x.startswith(t[:5]) or t.startswith(x[:5]) for x in cv)}
        nota = len(comuns) / len(alvo)
        if not melhor or nota > melhor[0]:
            melhor = (nota, v, comuns)
    if not melhor:
        return None
    nota, v, comuns = melhor
    conf = "ALTA" if nota >= 0.5 else "MEDIA" if nota >= 0.3 else "BAIXA"
    lacunas = []
    alvo = set(_tokens(" ".join([v.get("titulo", ""), v.get("descricao", "")] + list(v.get("requisitos") or []))))
    if conf != "BAIXA" and len(alvo) < 4:  # "ser bom em TI": pouco para comparar palavra a palavra
        conf = "BAIXA"
        lacunas.append("requisitos da vaga vagos demais para comparar (demonstração)")
    outra_area = pasta and pasta != "OUTROS" and pasta not in (v.get("setores") or [])
    if conf != "BAIXA" and outra_area and v["nivel"] == "experiente":
        conf = "BAIXA"
        lacunas.append(f"vaga de nível experiente em outra área (o currículo está na pasta {pasta.lower()})")
    r = {"vaga_id": v["id"], "confianca": conf, "evidencias": sorted(comuns)[:3], "lacunas": lacunas,
         "perfil_indicado": f"vagas da pasta {pasta.lower()}" if pasta and pasta != "OUTROS" else "",
         "motivo": f"termos em comum com a vaga: {', '.join(sorted(comuns)[:5]) or 'nenhum'} (demonstração)"}
    return _conferir_sugestao(r, vagas, pasta)


def sugerir_setores(titulo: str, descricao: str = "") -> dict:
    """Pastas da triagem (e, com Gemini, requisitos) sugeridas para uma vaga nova."""
    if config.IA_MODO == "gemini":
        sistema = (f"Você é Atlas, coordenador de recrutamento {config.DA_EMPRESA}. Para a vaga descrita, escolha de 1 a 4 pastas da "
                   f"triagem que devem alimentá-la, SOMENTE entre: {json.dumps(CATEGORIAS_DISPONIVEIS, ensure_ascii=False)}. "
                   "Sugira também de 3 a 6 requisitos objetivos e verificáveis em currículo (sem idade, gênero ou outras "
                   "características protegidas).")
        r = _gemini_json("sugestao_vaga", sistema, [f"TÍTULO: {titulo}\nDESCRIÇÃO: {descricao[:3000]}"],
                         SugestaoSetoresSchema, temperatura=0.3, max_saida=600)
        setores = [s for s in r.get("setores", []) if s in CATEGORIAS_DISPONIVEIS and s != "OUTROS"][:4]
        return {"setores": setores, "requisitos": [str(x)[:160] for x in r.get("requisitos", [])][:6], "fonte": "gemini"}
    from .triagem import _PALAVRAS
    t = _sem_acento(f"{titulo} {descricao}")
    setores = []
    for cat in CATEGORIAS_DISPONIVEIS:
        if cat != "OUTROS" and re.search(rf"\b{re.escape(_sem_acento(cat))}\b", t):
            setores.append(cat)
    for cat, chaves in _PALAVRAS:
        if cat not in setores and any(re.search(c, t) for c in chaves):
            setores.append(cat)
    extras = {"operador": "PRODUÇÃO", "assistente administrativ": "ADMINISTRATIVO", "auxiliar administrativ": "ADMINISTRATIVO",
              "vendedor": "VENDAS", "almoxarif": "ALMOXARIFADO", "manutencao": "MECANICA INDUSTRIAL"}
    for chave, cat in extras.items():
        if chave in t and cat not in setores:
            setores.append(cat)
    return {"setores": setores[:4], "requisitos": [], "fonte": "demo"}


# =========================================================================== chat individual
def _prompt_chat(ctx, pergunta, historico):
    ag = ctx["agente"]
    colegas = ", ".join(f'{c["nome"]} ({c["especialidade"]})' for c in ctx.get("colegas", []))
    sistema = (f'Você é {ag["nome"]}, agente de IA de recrutamento {config.DA_EMPRESA} ({ag.get("especialidade") or ag["funcao"]}). '
               f'{_voz({"perfil": ctx.get("personalidade")})}\n'
               "Responda em português, de forma curta, natural e no seu jeito de falar, usando SOMENTE os dados do CONTEXTO "
               "(seus registros reais no sistema). Cite vaga e candidato quando fizer sentido. "
               "Se a informação não estiver no contexto, diga com franqueza que não tem esse registro e, se for assunto "
               f"de um colega ({colegas or 'nenhum'}), sugira falar com essa pessoa pelo nome. "
               "Nunca decida contratações de candidatos nem comente características protegidas. Sem markdown.")
    conv = "\n".join(f'{"Recrutador" if m.get("autor") == "usuario" else ag["nome"]}: {m.get("texto", "")}'
                     for m in historico[-6:])
    prompt = f"CONTEXTO:\n{_js(ctx)}\n\nCONVERSA:\n{conv}\nRecrutador: {pergunta}"
    return sistema, prompt


def responder_chat(ctx: dict, pergunta: str, historico: list[dict]) -> str:
    return "".join(responder_chat_stream(ctx, pergunta, historico, digitar=False)).strip() or \
        "Não consegui formular uma resposta agora."


def responder_chat_stream(ctx: dict, pergunta: str, historico: list[dict], digitar: bool = True) -> Iterator[str]:
    """Resposta em pedaços, para o front mostrar o agente 'digitando'."""
    if config.IA_MODO == "gemini":
        motivo = uso_ia.bloqueio()
        if motivo:
            yield f"A IA está pausada agora: {motivo}."
            return
        sistema, prompt = _prompt_chat(ctx, pergunta, historico)
        modelo, uso, niveis = config.GEMINI_MODELO_AGENTES, None, _niveis_para(config.GEMINI_MODELO_AGENTES)
        for i, nivel in enumerate(niveis):
            try:
                for parte in cliente_gemini().models.generate_content_stream(
                        model=modelo, contents=[prompt], config=_config(modelo, sistema, None, 0.5, 800, nivel)):
                    uso = getattr(parte, "usage_metadata", None) or uso
                    if parte.text:
                        yield parte.text
                _nivel_aceito.setdefault(modelo, nivel)
                break
            except Exception as e:
                if _rejeita_pensamento(e) and i + 1 < len(niveis):
                    continue
                if uso_ia.falha_provedor(e):
                    yield f"A IA está pausada agora: {uso_ia.bloqueio()}."
                    return
                raise
        uso_ia.registrar("chat", modelo, uso)
        return
    texto = _chat_demo(ctx, pergunta)
    if not digitar:
        yield texto
        return
    for palavra in re.findall(r"\S+\s*", texto):
        if config.DEMO_DIGITACAO_MS:
            time.sleep(config.DEMO_DIGITACAO_MS / 1000)
        yield palavra


def _com_voz(ctx, texto, chance=0.4):
    p = ctx.get("personalidade") or {}
    if p.get("bordoes") and random.random() < chance:
        return f"{bordao(p)} {texto}"
    return texto


def _chat_demo(ctx, pergunta):
    q = _sem_acento(pergunta)
    ag = ctx["agente"]
    p = ctx.get("personalidade") or {}
    if re.fullmatch(r"\s*(oi|ola|bom dia|boa tarde|boa noite|e ai|tudo bem)\W*", q):
        return p.get("saudacao") or f"Oi! {ag['nome']} aqui."
    if re.search(r"quem (e|eh) voce|se apresent|personalidade|como voce e", q):
        return f"{ag['nome']} aqui — {p.get('resumo', 'faço parte da equipe de recrutamento')}. {bordao(p)}"
    vagas, cands = ctx.get("vagas", []), ctx.get("candidatos", [])
    if ag["funcao"] == "recepcao":
        return _com_voz(ctx, f'Tenho {ctx.get("fila_recepcao", 0)} currículo(s) aguardando triagem e já registrei '
                             f'{ctx.get("curriculos_registrados", 0)} no total.')
    if ag["funcao"] == "supervisor":
        return _com_voz(ctx, _atlas_demo(ctx, q))
    if re.search(r"curricul.*(cheg|receb)|triagem|recepcao|e-?mail", q):
        return "Isso é com a Bia, da recepção — pergunte no canal da equipe que ela responde."
    if not vagas:
        return "Ainda não tenho vagas atribuídas, então não tenho análises para comentar."
    nomeado = next((c for c in cands if _sem_acento(c["nome"].split(" ")[0]) in q), None)
    if nomeado and re.search(r"por ?que|justific|recomend|motivo|ajuda|colega", q):
        extra = ""
        if nomeado.get("colaboracoes"):
            c0 = nomeado["colaboracoes"][0]
            extra = f' Consultei {c0.get("colega")}: {c0.get("resposta", "")}'
        return f'{nomeado["nome"]} ({nomeado["vaga"]}): {nomeado["justificativa"]}{extra}'
    if re.search(r"andamento|como esta|progresso", q):
        return _com_voz(ctx, "; ".join(f'{v["titulo"]}: {v["analisados"]} de {v["recebidos"]} currículos analisados'
                                       for v in vagas) + ".")
    if re.search(r"falta|pendente|ainda preciso|quantos", q):
        pend = sum(v["recebidos"] - v["analisados"] for v in vagas)
        return _com_voz(ctx, f"Ainda tenho {pend} currículo(s) para analisar nas minhas vagas.")
    if re.search(r"humana|entrevista|duvida", q):
        h = [c for c in cands if c["avaliacao_humana"]]
        return ("Precisam de avaliação humana: " + "; ".join(f'{c["nome"]} ({", ".join(c["a_verificar"][:2]) or "ver análise"})' for c in h) + "."
                if h else "Nenhum candidato analisado por mim está marcado para avaliação humana.")
    if re.search(r"verificar|requisito", q):
        com = [c for c in cands if c["a_verificar"]]
        return "; ".join(f'{c["nome"]}: {", ".join(c["a_verificar"][:3])}' for c in com[:5]) + "." if com else "Não há requisitos pendentes de verificação."
    if re.search(r"colega|ajuda|consult", q):
        com = [c for c in cands if c.get("colaboracoes")]
        return ("Pedi ajuda nestes casos: " + "; ".join(f'{c["nome"]} ({c["colaboracoes"][0].get("colega")})' for c in com[:5]) + "."
                if com else "Até agora não precisei consultar colegas.")
    if re.search(r"compat|melhor|candidat|recomend", q):
        r = [c for c in cands if c["recomendado"]]
        return _com_voz(ctx, "Mais compatíveis: " + "; ".join(f'{c["nome"]} ({c["vaga"]}, {c["compatibilidade"]})' for c in r[:5]) + "."
                        if r else "Ainda não recomendei nenhum candidato.")
    return ("Não tenho esse registro. Posso responder sobre andamento das vagas, candidatos recomendados, "
            "justificativas, consultas a colegas e requisitos a verificar. (Modo demonstração: respostas por regras, sem IA externa.)")


def _atlas_demo(ctx, q):
    q_ = ctx.get("quadro", {})
    equipe = ctx.get("equipe", [])
    if re.search(r"contrat|reforc|dispens|estac", q):
        c = ctx.get("contratacoes", [])
        txt = ("Últimas mudanças no time: " + "; ".join(x["descricao"] for x in c[:4]) + "."
               if c else "Ainda não precisei contratar ninguém.")
        return (f"{txt} Eu contrato quando falta especialista para uma vaga, quando a fila passa de "
                f"{ctx.get('limite_fila', 5)} currículos por pessoa ou quando o time pede ajuda de uma área que não temos.")
    if re.search(r"equipe|time|quem|carga|ocupad|livre", q) and equipe:
        return "Agora: " + "; ".join(f'{a["nome"]} {a["estado"]}' + (f' ({a["carga"]} na fila)' if a.get("carga") else "")
                                     for a in equipe) + "."
    return (f'Vagas abertas: {q_.get("vagas_abertas", 0)}; currículos recebidos: {q_.get("curriculos_recebidos", 0)}; '
            f'análises concluídas: {q_.get("analises_concluidas", 0)}; pendências abertas: {q_.get("pendencias", 0)}.')


# =========================================================================== conversa em grupo
def conversa_equipe(ctx: dict, pergunta: str, historico: list[dict]) -> list[dict]:
    """Quem do time responde e o quê: [{agente_id, texto}] (1 a 3 falas)."""
    equipe = ctx.get("equipe", [])
    if config.IA_MODO == "gemini":
        nomes = {_sem_acento(a["nome"]): a for a in equipe}
        sistema = (f"Você escreve as falas de uma conversa em grupo da equipe de recrutamento {config.DA_EMPRESA}, formada por agentes "
                   "de IA com personalidades próprias (veja EQUIPE). O recrutador escreveu no canal da equipe. Responda como "
                   "1 a 3 membros, na ordem natural: quem tem a informação responde; Atlas organiza quando precisa; um colega "
                   "pode complementar ou citar o outro pelo nome. Cada fala com no máximo 2 frases, na voz da personalidade. "
                   "Use SOMENTE os dados do CONTEXTO; se ninguém tiver o registro, diga isso. Nunca decida contratações de "
                   f"candidatos nem comente características protegidas. Sem markdown.{REGRAS_FALA}")
        conv = "\n".join(f'{m.get("nome") or ("Recrutador" if m.get("autor") == "usuario" else "Equipe")}: {m.get("texto", "")}'
                         for m in historico[-8:])
        r = _gemini_json("equipe", sistema, [f"CONTEXTO:\n{_js(ctx)}\n\nCONVERSA:\n{conv}\nRecrutador: {pergunta}"],
                         ConversaSchema, temperatura=0.6, max_saida=900)
        saida = []
        for f in r.get("falas", [])[:3]:
            a = nomes.get(_sem_acento(f.get("agente", "")).strip())
            if a and f.get("texto"):
                saida.append({"agente_id": a["id"], "texto": str(f["texto"])[:600]})
        return saida or [{"agente_id": _atlas(equipe)["id"], "texto": "Não temos esse registro agora."}]
    return _equipe_demo(ctx, pergunta)


def _atlas(equipe):
    return next(a for a in equipe if a["funcao"] == "supervisor")


def _equipe_demo(ctx, pergunta):
    q = _sem_acento(pergunta)
    equipe = ctx.get("equipe", [])
    ctxs = ctx.get("agentes_ctx", {})
    atlas = _atlas(equipe)
    bia = next((a for a in equipe if a["funcao"] == "recepcao"), None)

    def responde(a, texto=None):
        return {"agente_id": a["id"], "texto": texto or _chat_demo(ctxs[a["id"]], pergunta)}

    mencionados = [a for a in equipe if re.search(rf"\b{re.escape(_sem_acento(a['nome']))}\b", q)]
    if mencionados:
        return [responde(a) for a in mencionados[:3]]
    if re.search(r"contrat|reforc|dispens|estac|carga|sobrecarg|ocupad|quem esta|livre", q):
        return [responde(atlas)]
    if bia and re.search(r"curricul.*(cheg|receb)|triagem|recepcao|e-?mail|arquivo|duplicad", q):
        return [responde(bia)]
    # candidato citado pelo primeiro nome -> quem analisou
    for c in ctx.get("candidatos", []):
        primeiro = _sem_acento(c["nome"].split(" ")[0])
        if len(primeiro) >= 3 and re.search(rf"\b{re.escape(primeiro)}\b", q):
            a = next((x for x in equipe if x["nome"] == c.get("analista")), None)
            if a:
                return [responde(a)]
    # vaga citada -> responsável
    for v in ctx.get("vagas", []):
        tk = [t for t in _tokens(v["titulo"]) if len(t) >= 4]
        if tk and all(t[:5] in q for t in tk[:2]):
            resp = [a for a in equipe if a["nome"] in v.get("responsaveis", [])]
            if resp:
                return [responde(resp[0], f'{v["titulo"]}: {v["analisados"]} de {v["recebidos"]} analisados, '
                                          f'{v["pendentes"]} na fila. ' + _chat_demo(ctxs[resp[0]["id"]], "quais candidatos são mais compatíveis?"))]
    # visão geral: Atlas abre e chama quem tem vagas
    analistas = [a for a in equipe if a["funcao"] == "analista" and a.get("vagas")][:2]
    falas = [responde(atlas, _atlas_demo(ctxs[atlas["id"]], q) + (
        " " + ", ".join(a["nome"] for a in analistas) + ", algo a destacar?" if analistas else ""))]
    for a in analistas:
        falas.append(responde(a, _chat_demo(ctxs[a["id"]], "como está o andamento?")))
    return falas
