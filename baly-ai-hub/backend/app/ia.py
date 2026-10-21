"""IA dos agentes: análise de currículo × vaga e chat. Gemini ou modo demonstração."""
import json
import logging
import re
import threading
import time
import unicodedata

from pydantic import BaseModel

from . import config

log = logging.getLogger("baly.ia")

_cliente = None
_semaforo = threading.Semaphore(max(1, config.IA_CONCORRENCIA))


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


# --------------------------------------------------------------------------- esquema da análise
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
    pedir_segunda_opiniao: bool
    segunda_opiniao_especialidade: str
    segunda_opiniao_motivo: str


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
- avaliacao_humana = true quando o perfil é promissor mas há dúvida importante que só uma pessoa pode resolver.
"""


def _sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", (s or "").lower()) if unicodedata.category(c) != "Mn")


def _gemini_json(modelo, sistema, conteudos, schema):
    from google.genai import types
    with _semaforo:
        for tentativa in range(3):
            try:
                r = cliente_gemini().models.generate_content(
                    model=modelo, contents=conteudos,
                    config=types.GenerateContentConfig(
                        system_instruction=sistema, response_mime_type="application/json",
                        response_schema=schema, temperature=0.2, safety_settings=sem_bloqueio()))
                txt = r.text or ""
                m = re.search(r"\{.*\}", txt, re.DOTALL)
                return json.loads(m.group(0) if m else txt)
            except Exception as e:
                limite = getattr(e, "code", None) == 429 or "429" in str(e)
                log.warning("Gemini: tentativa %d/3 falhou (%s)", tentativa + 1, type(e).__name__)
                if tentativa == 2:
                    raise
                time.sleep(5 * (tentativa + 1) if limite else 2)


# --------------------------------------------------------------------------- análise
def analisar(agente: dict, vaga: dict, curriculo_texto: str | None, arquivo: tuple[bytes, str] | None,
             especialidades: list[dict], setor_triagem: str | None) -> tuple[dict, str]:
    """Retorna (resultado, modelo_usado)."""
    if config.IA_MODO == "gemini":
        return _analisar_gemini(agente, vaga, curriculo_texto, arquivo, especialidades), config.GEMINI_MODELO_AGENTES
    time.sleep(config.DEMO_DURACAO_ANALISE)  # só no modo demonstração, para o escritório ser observável
    return _analisar_demo(agente, vaga, curriculo_texto, especialidades, setor_triagem), "demo-heuristica"


def _analisar_gemini(agente, vaga, texto, arquivo, especialidades):
    from google.genai import types
    outras = [f'{e["codigo"]} ({e["nome"]})' for e in especialidades if e["codigo"] != agente.get("especialidade_codigo")]
    sistema = (f'Você é {agente["nome"]}, agente de IA da equipe de recrutamento da Baly, especialista em '
               f'{agente.get("especialidade") or "recrutamento"}. {agente.get("personalidade", "")}\n'
               f"Analise o currículo para a vaga indicada e responda em português.\n{REGRAS_ETICAS}\n"
               "- Se o currículo tiver característica que justifique avaliação complementar de outra especialidade, "
               f"marque pedir_segunda_opiniao e informe o código entre: {', '.join(outras) or 'nenhuma'}.")
    conteudos = ["VAGA:\n" + json.dumps(vaga, ensure_ascii=False)]
    if texto:
        conteudos.append("CURRÍCULO (texto extraído):\n" + texto[:30000])
    elif arquivo:
        conteudos.append("CURRÍCULO (arquivo anexo):")
        conteudos.append(types.Part.from_bytes(data=arquivo[0], mime_type=arquivo[1]))
    else:
        raise ValueError("Currículo sem conteúdo legível")
    r = _gemini_json(config.GEMINI_MODELO_AGENTES, sistema, conteudos, AnaliseSchema)
    r["compatibilidade"] = str(r.get("compatibilidade", "BAIXA")).upper()
    if r["compatibilidade"] not in ("ALTA", "MEDIA", "BAIXA"):
        r["compatibilidade"] = "BAIXA"
    return r


_PARADAS = set("a o e de da do das dos em no na nos nas para com por um uma ou ao aos se que ter tem ser como "
               "experiencia experiencias conhecimento conhecimentos minimo desejavel obrigatorio".split())


def _tokens(s):
    return [w for w in re.findall(r"[a-z0-9+#-]{3,}", _sem_acento(s)) if w not in _PARADAS]


def _analisar_demo(agente, vaga, texto, especialidades, setor):
    linhas = [l.strip(" •-\t") for l in (texto or "").splitlines() if len(l.strip()) > 3]
    reqs = vaga.get("requisitos") or []
    atendidos, verificar = [], []
    for req in reqs:
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
    total = max(1, len(atendidos) + len(verificar))
    taxa = len(atendidos) / total
    compat = "ALTA" if taxa >= 0.66 else "MEDIA" if taxa >= 0.34 else "BAIXA"
    comp_linha = next((l for l in linhas if _sem_acento(l).startswith(("competencias", "habilidades"))), "")
    competencias = [c.strip() for c in re.split(r"[·,;|]", comp_linha.split(":", 1)[-1]) if c.strip()][:6] if comp_linha else []
    exp = [l for l in linhas if re.search(r"(19|20)\d{2}\s*[–-]", l)][:3]
    sem_contato = [l for l in linhas[1:] if "@" not in l and not re.search(r"\d{4,}[-\s]?\d{3,}", l)]
    i_res = next((i for i, l in enumerate(sem_contato) if _sem_acento(l).startswith(("resumo", "perfil", "objetivo"))), None)
    resumo = (sem_contato[i_res + 1] if i_res is not None and i_res + 1 < len(sem_contato)
              else next((l for l in sem_contato if len(l) > 40), ""))
    pedir, esp_seg, motivo = False, "", ""
    if setor:
        alvo = next((e for e in especialidades if setor in e.get("setores", [])), None)
        if alvo and alvo["codigo"] != agente.get("especialidade_codigo"):
            pedir, esp_seg = True, alvo["codigo"]
            motivo = f"Triagem classificou o currículo em {setor}, área da especialidade {alvo['nome']}."
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
        "pedir_segunda_opiniao": pedir,
        "segunda_opiniao_especialidade": esp_seg,
        "segunda_opiniao_motivo": motivo,
    }


# --------------------------------------------------------------------------- chat
def responder_chat(ctx: dict, pergunta: str, historico: list[dict]) -> str:
    if config.IA_MODO == "gemini":
        sistema = (f'Você é {ctx["agente"]["nome"]}, agente de IA de recrutamento da Baly '
                   f'({ctx["agente"].get("especialidade") or ctx["agente"]["funcao"]}). '
                   "Responda em português, de forma curta e direta, usando SOMENTE os dados do CONTEXTO "
                   "(seus registros reais no sistema). Cite vaga e candidato quando fizer sentido. "
                   "Se a informação não estiver no contexto, diga claramente que não tem esse registro. "
                   "Nunca decida contratações nem comente características protegidas dos candidatos.")
        conv = "\n".join(f'{"Recrutador" if m.get("autor") == "usuario" else ctx["agente"]["nome"]}: {m.get("texto", "")}'
                         for m in historico[-6:])
        prompt = f"CONTEXTO:\n{json.dumps(ctx, ensure_ascii=False)}\n\nCONVERSA:\n{conv}\nRecrutador: {pergunta}"
        from google.genai import types
        with _semaforo:
            r = cliente_gemini().models.generate_content(
                model=config.GEMINI_MODELO_AGENTES, contents=[prompt],
                config=types.GenerateContentConfig(system_instruction=sistema, temperature=0.3))
        return (r.text or "").strip() or "Não consegui formular uma resposta agora."
    return _chat_demo(ctx, pergunta)


def _chat_demo(ctx, pergunta):
    q = _sem_acento(pergunta)
    vagas, cands = ctx.get("vagas", []), ctx.get("candidatos", [])
    if ctx["agente"]["funcao"] != "analista":
        if ctx["agente"]["funcao"] == "recepcao":
            return f'Tenho {ctx.get("fila_recepcao", 0)} currículo(s) aguardando triagem e já registrei {ctx.get("curriculos_registrados", 0)} no total.'
        q_ = ctx.get("quadro", {})
        return (f'Vagas abertas: {q_.get("vagas_abertas", 0)}; currículos recebidos: {q_.get("curriculos_recebidos", 0)}; '
                f'análises concluídas: {q_.get("analises_concluidas", 0)}; pendências abertas: {q_.get("pendencias", 0)}.')
    if not vagas:
        return "Ainda não tenho vagas atribuídas, então não tenho análises para comentar."
    nomeado = next((c for c in cands if _sem_acento(c["nome"].split(" ")[0]) in q), None)
    if nomeado and re.search(r"por ?que|justific|recomend|motivo", q):
        return f'{nomeado["nome"]} ({nomeado["vaga"]}): {nomeado["justificativa"]}'
    if re.search(r"andamento|como esta|progresso", q):
        return "; ".join(f'{v["titulo"]}: {v["analisados"]} de {v["recebidos"]} currículos analisados' for v in vagas) + "."
    if re.search(r"falta|pendente|ainda preciso|quantos", q):
        pend = sum(v["recebidos"] - v["analisados"] for v in vagas)
        return f"Ainda tenho {pend} currículo(s) para analisar nas minhas vagas."
    if re.search(r"humana|entrevista|duvida", q):
        h = [c for c in cands if c["avaliacao_humana"]]
        return ("Precisam de avaliação humana: " + "; ".join(f'{c["nome"]} ({", ".join(c["a_verificar"][:2]) or "ver análise"})' for c in h) + "."
                if h else "Nenhum candidato analisado por mim está marcado para avaliação humana.")
    if re.search(r"verificar|requisito", q):
        com = [c for c in cands if c["a_verificar"]]
        return "; ".join(f'{c["nome"]}: {", ".join(c["a_verificar"][:3])}' for c in com[:5]) + "." if com else "Não há requisitos pendentes de verificação."
    if re.search(r"compat|melhor|candidat|recomend", q):
        r = [c for c in cands if c["recomendado"]]
        return ("Mais compatíveis: " + "; ".join(f'{c["nome"]} ({c["vaga"]}, {c["compatibilidade"]})' for c in r[:5]) + "."
                if r else "Ainda não recomendei nenhum candidato.")
    return ("Não tenho esse registro. Posso responder sobre andamento das vagas, candidatos recomendados, "
            "justificativas e requisitos a verificar. (Modo demonstração: respostas por regras, sem IA externa.)")
