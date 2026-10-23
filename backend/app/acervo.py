"""Banco de currículos: o arquivo do escritório, com as pastas da triagem agrupadas pelas áreas da equipe.

Cada currículo mostra o que os agentes disseram dele (a triagem da recepção e as análises dos analistas) e a vaga
sugerida: a melhor candidatura analisada, a que está em análise ou, sem nenhuma, a sugestão que o Atlas der quando
o recrutador pedir. Tudo sai em poucas consultas agrupadas (o banco pode estar longe, no Supabase).
"""
from collections import defaultdict

from sqlalchemy import false, func, literal_column, or_, select
from sqlalchemy.orm import Session

from . import eventos
from .catalogo import CATEGORIAS_DISPONIVEIS
from .models import Agente, Analise, Candidato, Candidatura, Curriculo, Decisao, Especialidade, Vaga, agora, iso

OUTROS = {"codigo": "outros", "nome": "Outras pastas"}
CRITERIO = 2  # versão das regras do Atlas ao sugerir vaga (2: experiência profissional x estudo, nível, evidências)
ORDEM_COMPAT = {"ALTA": 0, "MEDIA": 1, "BAIXA": 2}
SITUACOES = ("talentos", "analise", "recomendados", "humana", "decididos", "triagem", "descartados")
POR_PAGINA = 40


def grupos(db: Session) -> list[dict]:
    """Áreas da equipe (especialidades) com as pastas da triagem que cada uma atende; o resto fica em "Outras"."""
    saida, usados = [], set()
    for e in db.scalars(select(Especialidade).order_by(Especialidade.id)).all():
        setores = [s for s in (e.setores or []) if s not in usados]
        usados.update(setores)
        saida.append({"codigo": e.codigo, "nome": e.nome, "setores": setores})
    saida.append({**OUTROS, "setores": [c for c in CATEGORIAS_DISPONIVEIS if c not in usados]})
    return saida


def grupo_de(setor: str | None, gs: list[dict]) -> str:
    return next((g["codigo"] for g in gs if setor in g["setores"]), OUTROS["codigo"])


def resumo(db: Session, gs: list[dict] | None = None) -> dict:
    """Quantos currículos há em cada pasta (para as pastas do balcão e a tela do banco)."""
    gs = gs or grupos(db)
    por_setor, na_triagem, descartados = defaultdict(int), 0, 0
    for setor, status, n in db.execute(select(Curriculo.setor, Curriculo.status, func.count())
                                       .where(Curriculo.status != "anonimizado")
                                       .group_by(Curriculo.setor, Curriculo.status)).all():
        if status == "triado":
            por_setor[setor or "OUTROS"] += n
        elif status == "invalido":
            descartados += n
        else:  # recebido (na fila) ou erro na triagem (tem pendência própria)
            na_triagem += n
    saida = []
    for g in gs:
        pastas = [{"setor": s, "n": por_setor.get(s, 0)} for s in g["setores"]]
        pastas += [{"setor": s, "n": n} for s, n in por_setor.items()  # pasta fora do catálogo (dado antigo)
                   if g["codigo"] == OUTROS["codigo"] and s not in CATEGORIAS_DISPONIVEIS]
        saida.append({"codigo": g["codigo"], "nome": g["nome"], "total": sum(p["n"] for p in pastas), "pastas": pastas})
    return {"total": sum(por_setor.values()), "grupos": saida, "na_triagem": na_triagem, "descartados": descartados}


def _sugestao(cands: list[dict]) -> dict | None:
    """A vaga que o currículo mais sustenta: recomendado > avaliação humana > compatibilidade; sem análise, a da fila."""
    analisadas = [c for c in cands if c.get("compatibilidade")]
    if analisadas:
        return min(analisadas, key=lambda c: (not c["recomendado"], not c["avaliacao_humana"],
                                              ORDEM_COMPAT.get(c["compatibilidade"], 3)))
    return cands[0] if cands else None


def _situacao(cur: Curriculo, cands: list[dict]) -> str:
    if cur.status == "invalido":
        return "descartado"
    if cur.status != "triado":
        return "erro na triagem" if cur.status == "erro" else "na triagem"
    if not cands:
        return "banco de talentos"
    melhor = _sugestao(cands)
    if melhor.get("decisao"):
        return {"entrevista": "chamado para entrevista", "banco_talentos": "guardado no banco de talentos",
                "descartado": "não seguir"}.get(melhor["decisao"], melhor["decisao"])
    if not melhor.get("compatibilidade"):
        return "em análise"
    return "recomendado" if melhor["recomendado"] else "avaliação humana" if melhor["avaliacao_humana"] else "analisado"


def _candidaturas(db: Session, cur_ids: list[int], completo=False) -> dict[int, list[dict]]:
    """Candidaturas de cada currículo com a análise principal e a decisão atual (4 consultas para a página toda)."""
    if not cur_ids:
        return {}
    cands = db.execute(select(Candidatura, Vaga.titulo, Vaga.status).join(Vaga, Vaga.id == Candidatura.vaga_id)
                       .where(Candidatura.curriculo_id.in_(cur_ids)).order_by(Candidatura.id)).all()
    ids = [c.id for c, _, _ in cands]
    analises = defaultdict(list)
    for a in db.scalars(select(Analise).where(Analise.candidatura_id.in_(ids or [-1])).order_by(Analise.id)).all():
        analises[a.candidatura_id].append(a)
    decisoes = {}
    for cid, dec in db.execute(select(Decisao.candidatura_id, Decisao.decisao).where(
            Decisao.candidatura_id.in_(ids or [-1])).order_by(Decisao.id)).all():
        decisoes[cid] = dec
    nomes = dict(db.execute(select(Agente.id, Agente.nome)).all())
    saida = defaultdict(list)
    for c, titulo, status_vaga in cands:
        lista = analises.get(c.id, [])
        a = next((x for x in lista if x.tipo == "principal"), lista[0] if lista else None)
        r = (a.resultado or {}) if a else {}
        item = {"candidatura_id": c.id, "vaga_id": c.vaga_id, "vaga": titulo, "vaga_status": status_vaga,
                "status": c.status, "origem": c.origem or "triagem", "decisao": decisoes.get(c.id),
                "analista": nomes.get(a.agente_id) if a else None, "analista_id": a.agente_id if a else None,
                "compatibilidade": a.compatibilidade if a else None,
                "recomendado": bool(a and a.recomendado), "avaliacao_humana": bool(a and a.avaliacao_humana),
                "parecer": r.get("justificativa", "")}
        if completo:
            item["analises"] = [{"agente": nomes.get(x.agente_id, "?"), "tipo": x.tipo, "compatibilidade": x.compatibilidade,
                                 "recomendado": x.recomendado, "avaliacao_humana": x.avaliacao_humana,
                                 "justificativa": (x.resultado or {}).get("justificativa", ""),
                                 "resumo": (x.resultado or {}).get("resumo_profissional", ""),
                                 "atendidos": [f'{y.get("requisito")}' for y in (x.resultado or {}).get("requisitos_atendidos", [])][:6],
                                 "a_verificar": (x.resultado or {}).get("a_verificar", [])[:6],
                                 "em": iso(x.criado_em)} for x in lista]
        saida[c.curriculo_id].append(item)
    return saida


def _filtro_busca(db: Session, busca: str):
    termo = f"%{busca.lower()}%"
    campos = [func.lower(Candidato.nome).like(termo), func.lower(func.coalesce(Candidato.cidade, "")).like(termo),
              func.lower(func.coalesce(Candidato.email, "")).like(termo), func.lower(func.coalesce(Curriculo.setor, "")).like(termo)]
    if db.get_bind().dialect.name == "postgresql":  # busca em português no texto (usa o índice do banco_pg)
        port = literal_column("'portuguese'")
        campos.append(func.to_tsvector(port, func.coalesce(Curriculo.texto, literal_column("''")))
                      .bool_op("@@")(func.websearch_to_tsquery(port, busca)))
    else:
        campos.append(func.lower(func.coalesce(Curriculo.texto, "")).like(termo))
    return or_(*campos)


def listar(db: Session, grupo: str | None = None, pasta: str | None = None, situacao: str | None = None,
           busca: str = "", pagina: int = 1) -> dict:
    gs = grupos(db)
    q = (select(Curriculo, Candidato).outerjoin(Candidato, Candidato.id == Curriculo.candidato_id)
         .where(Curriculo.status != "anonimizado"))
    if pasta:
        q = q.where(Curriculo.setor == pasta, Curriculo.status == "triado")
    elif grupo:
        setores = next((g["setores"] for g in gs if g["codigo"] == grupo), [])
        sem_pasta = Curriculo.setor.is_(None) if grupo == OUTROS["codigo"] else false()
        q = q.where(Curriculo.status == "triado", or_(Curriculo.setor.in_(setores or ["-"]), sem_pasta))
    if situacao == "triagem":
        q = q.where(Curriculo.status.in_(("recebido", "erro")))
    elif situacao == "descartados":
        q = q.where(Curriculo.status == "invalido")
    elif situacao == "talentos":
        q = q.where(Curriculo.status == "triado", ~select(Candidatura.id).where(
            Candidatura.curriculo_id == Curriculo.id).exists())
    elif situacao in SITUACOES:
        q = q.where(Curriculo.status == "triado", select(Candidatura.id).where(Candidatura.curriculo_id == Curriculo.id).exists())
    elif not situacao:
        q = q.where(Curriculo.status != "invalido")
    if (busca or "").strip():
        q = q.where(_filtro_busca(db, busca.strip()[:120]))
    pagina = max(1, pagina)
    # os filtros que dependem da análise (recomendados, avaliação humana, decididos) são aplicados depois: pede-se mais
    pos = situacao in ("analise", "recomendados", "humana", "decididos")
    limite = POR_PAGINA * (5 if pos else 1)
    linhas = db.execute(q.order_by(Curriculo.recebido_em.desc(), Curriculo.id.desc())
                        .offset((pagina - 1) * POR_PAGINA if not pos else 0).limit(limite + 1)).all()
    tem_mais = len(linhas) > limite
    linhas = linhas[:limite]
    cands = _candidaturas(db, [c.id for c, _ in linhas])
    itens = []
    for cur, cand in linhas:
        lista = cands.get(cur.id, [])
        melhor = _sugestao(lista)
        sit = _situacao(cur, lista)
        if situacao == "analise" and sit != "em análise":
            continue
        if situacao == "recomendados" and sit != "recomendado":
            continue
        if situacao == "humana" and sit != "avaliação humana":
            continue
        if situacao == "decididos" and not (melhor and melhor.get("decisao")):
            continue
        sug = sugestao_de(cur)
        itens.append({
            "curriculo_id": cur.id, "nome": cand.nome if cand else "Candidato", "cidade": cand.cidade if cand else None,
            "setor": cur.setor, "grupo": grupo_de(cur.setor, gs), "status": cur.status, "situacao": sit,
            "anos_experiencia": cur.anos_experiencia, "recebido_em": iso(cur.recebido_em), "origem": cur.origem,
            "resumo_triagem": cur.resumo_triagem or "", "confianca": cur.confianca,
            "vagas": len(lista), "vaga_sugerida": melhor, "sugestao_atlas": sug,
        })
    if pos:
        tem_mais = len(itens) > pagina * POR_PAGINA
        itens = itens[(pagina - 1) * POR_PAGINA:pagina * POR_PAGINA]
    return {"itens": itens, "pagina": pagina, "tem_mais": tem_mais, "por_pagina": POR_PAGINA,
            "resumo": resumo(db, gs)}


def detalhe(db: Session, cur: Curriculo) -> dict:
    gs = grupos(db)
    cand = cur.candidato
    lista = _candidaturas(db, [cur.id], completo=True).get(cur.id, [])
    rec = db.scalar(select(Agente).where(Agente.funcao == "recepcao"))
    abertas = db.execute(select(Vaga.id, Vaga.titulo, Vaga.setores).where(Vaga.status == "aberta").order_by(Vaga.id)).all()
    ja = {c["vaga_id"] for c in lista}
    return {
        "curriculo_id": cur.id, "status": cur.status, "situacao": _situacao(cur, lista),
        "arquivo": cur.arquivo_nome, "tem_arquivo": bool(cur.caminho), "origem": cur.origem,
        "recebido_em": iso(cur.recebido_em), "texto": cur.texto,
        "candidato": {"nome": cand.nome if cand else "Candidato", "cidade": cand.cidade if cand else None,
                      "email": cand.email if cand else None, "telefone": cand.telefone if cand else None,
                      "dado_sensivel": bool(cand and cand.dado_sensivel)},
        "triagem": {"por": rec.nome if rec else "Recepção", "por_id": rec.id if rec else None, "setor": cur.setor,
                    "grupo": grupo_de(cur.setor, gs), "confianca": cur.confianca,
                    "anos_experiencia": cur.anos_experiencia, "resumo": cur.resumo_triagem or ""},
        "candidaturas": lista, "vaga_sugerida": _sugestao(lista),
        "sugestao_atlas": sugestao_de(cur),
        "vagas_abertas": [{"id": i, "titulo": t, "da_pasta": (cur.setor in (s or []))} for i, t, s in abertas if i not in ja],
    }


def guardar_sugestao(cur: Curriculo, sugestao: dict | None, vaga: Vaga | None):
    """A sugestão do Atlas fica no currículo (não precisa perguntar de novo à IA); sai junto na anonimização.
    Com confiança baixa não é sugestão: fica registrado que nenhuma vaga combina bem, citando a mais próxima."""
    s = sugestao or {}
    conf = str(s.get("confianca") or "BAIXA").upper()
    boa = bool(vaga) and conf in ("ALTA", "MEDIA")
    motivo = s.get("motivo") or ("" if boa else "nenhuma vaga aberta combina com este currículo agora")
    cur.triagem_json = {**(cur.triagem_json or {}), "sugestao_atlas": {
        "vaga_id": vaga.id if boa else None, "vaga": vaga.titulo if boa else None,
        "mais_proxima": vaga.titulo if vaga and not boa else None,
        "confianca": conf, "motivo": motivo, "evidencias": s.get("evidencias") or [], "lacunas": s.get("lacunas") or [],
        "perfil_indicado": s.get("perfil_indicado") or "", "criterio": CRITERIO, "em": iso(agora())}}
    return boa


def sugestao_de(cur: Curriculo) -> dict | None:
    """A sugestão guardada; a de um critério antigo (antes de experiência x estudo) não vale como sugestão."""
    s = (cur.triagem_json or {}).get("sugestao_atlas")
    if s and s.get("criterio") != CRITERIO:
        return {**s, "antiga": True, "vaga_id": None, "vaga": None}
    return s


def vagas_para_sugestao(db: Session) -> list[dict]:
    return [{"id": v.id, "titulo": v.titulo, "area": v.area, "descricao": v.descricao, "requisitos": v.requisitos,
             "setores": v.setores} for v in db.scalars(select(Vaga).where(Vaga.status == "aberta")).all()]


def dados_para_sugestao(db: Session, cur: Curriculo) -> tuple[str, list[dict], str | None, int | None]:
    """Texto, vagas abertas, pasta e anos de experiência para o Atlas sugerir (a IA é chamada fora da transação)."""
    return cur.texto or "", vagas_para_sugestao(db), cur.setor, cur.anos_experiencia


def registrar_sugestao(db: Session, cur: Curriculo, sugestao: dict | None):
    vaga = db.get(Vaga, sugestao["vaga_id"]) if sugestao and sugestao.get("vaga_id") else None
    if vaga and vaga.status != "aberta":
        vaga = None
    boa = guardar_sugestao(cur, sugestao, vaga)
    from .servicos import atlas, primeiro_nome
    nome = primeiro_nome(cur.candidato.nome if cur.candidato else "Candidato")
    eventos.registrar(db, "sugestao_banco", (f"Atlas sugeriu a vaga “{vaga.titulo}” para {nome}" if boa else
                                             f"Atlas não achou vaga aberta que combine com {nome} por enquanto"),
                      atlas(db).id, curriculo_id=cur.id, candidato_id=cur.candidato_id, vaga_id=vaga.id if boa else None)
    return cur.triagem_json["sugestao_atlas"]
