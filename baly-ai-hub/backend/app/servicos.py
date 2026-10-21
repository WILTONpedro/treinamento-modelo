"""Regras do negócio: recepção, triagem, Atlas (supervisor), análises, estados e LGPD.

Cada etapa é uma Tarefa real na fila. O estado de cada agente é derivado das tarefas
e registros — o escritório nunca mostra trabalho que não está acontecendo.
"""
import hashlib
import logging
import os
from datetime import timedelta

from sqlalchemy import func, select, or_
from sqlalchemy.orm import Session

from . import config, eventos, ia, triagem
from .catalogo import CATEGORIAS_SENSIVEIS, ESPECIALIDADES, aparencia_para
from .db import sessao
from .models import (Agente, Analise, Candidato, Candidatura, Curriculo, Decisao, Especialidade, Evento,
                     Pendencia, Tarefa, Vaga, VagaAgente, agora, iso)
from .seguranca import apagar_arquivo, ler_arquivo, salvar_arquivo

log = logging.getLogger("baly.servicos")
MAX_ESTACOES = 8
STATUS_VAGA = ("aberta", "suspensa", "encerrada", "cancelada")


class ErroNegocio(Exception):
    pass


# =========================================================================== utilidades
def primeiro_nome(nome):
    partes = (nome or "Candidato").split()
    return partes[0] + (f" {partes[-1][0]}." if len(partes) > 1 else "")


def recepcao(db) -> Agente:
    return db.scalar(select(Agente).where(Agente.funcao == "recepcao"))


def atlas(db) -> Agente:
    return db.scalar(select(Agente).where(Agente.funcao == "supervisor"))


def _ocupar(ag: Agente, tarefa: Tarefa, atividade: str):
    ag.tarefa_atual_id = tarefa.id
    ag.atividade = atividade
    ag.ultima_atividade_em = agora()


def _liberar(db, ag: Agente):
    ag.tarefa_atual_id = None
    ag.ultima_atividade_em = agora()
    restantes = db.scalar(select(func.count()).select_from(Tarefa).where(
        Tarefa.agente_id == ag.id, Tarefa.status.in_(("pendente", "executando"))))
    if not restantes:
        ag.concluido_em = agora()
        ag.atividade = ""


def _abrir_pendencia(db, tipo, descricao, agente=None, **refs) -> Pendencia | None:
    filtro = [Pendencia.tipo == tipo, Pendencia.status == "aberta"]
    for k, v in refs.items():
        filtro.append(getattr(Pendencia, k) == v)
    if db.scalar(select(Pendencia).where(*filtro)):
        return None
    p = Pendencia(tipo=tipo, descricao=descricao, agente_id=agente.id if agente else None, **refs)
    db.add(p)
    db.flush()
    eventos.registrar(db, "pendencia_aberta", f"{agente.nome if agente else 'Sistema'} sinalizou: {descricao}",
                      agente.id if agente else None, pendencia_id=p.id)
    return p


# =========================================================================== recepção
def registrar_curriculo(db: Session, conteudo: bytes, nome_arquivo: str, mime: str | None, origem: str):
    """Ponto de entrada de todo currículo. Retorna (curriculo, novo?)."""
    ext = os.path.splitext(nome_arquivo or "")[1].lower()
    if ext not in triagem.EXTENSOES:
        raise ErroNegocio(f"Formato não suportado ({ext or 'sem extensão'}). Use PDF, DOCX, TXT ou imagem.")
    if len(conteudo) > config.TAMANHO_MAX_MB * 1024 * 1024:
        raise ErroNegocio(f"Arquivo maior que {config.TAMANHO_MAX_MB} MB.")
    if not conteudo:
        raise ErroNegocio("Arquivo vazio.")
    sha = hashlib.sha256(conteudo).hexdigest()
    rec = recepcao(db)
    existente = db.scalar(select(Curriculo).where(Curriculo.sha256 == sha))
    if existente:
        eventos.registrar(db, "curriculo_duplicado",
                          f"{rec.nome} reconheceu um currículo já registrado (#{existente.id}) e não o reprocessou",
                          rec.id, curriculo_id=existente.id)
        return existente, False
    cur = Curriculo(sha256=sha, arquivo_nome=os.path.basename(nome_arquivo)[:250], mime=triagem.mime_de(nome_arquivo, mime),
                    tamanho=len(conteudo), caminho=salvar_arquivo(sha, conteudo),
                    texto=triagem.extrair_texto(conteudo, nome_arquivo), origem=origem)
    db.add(cur)
    db.flush()
    db.add(Tarefa(tipo="TRIAGEM", curriculo_id=cur.id, agente_id=rec.id))
    eventos.registrar(db, "curriculo_recebido", f"{rec.nome} recebeu um novo currículo (#{cur.id}) e o registrou na fila",
                      rec.id, curriculo_id=cur.id)
    return cur, True


def executar_triagem(tarefa_id: int):
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        cur = db.get(Curriculo, t.curriculo_id)
        rec = recepcao(db)
        _ocupar(rec, t, f"triando o currículo #{cur.id}")
        dados = (ler_arquivo(cur.caminho), cur.arquivo_nome, cur.mime, cur.texto)
    resultado = triagem.triar(*dados)  # fora da transação: a IA pode demorar
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        cur = db.get(Curriculo, t.curriculo_id)
        rec = recepcao(db)
        setor = resultado.get("setor") or "OUTROS"
        cur.triagem_json = resultado
        cur.setor, cur.confianca = setor, resultado.get("confianca")
        cur.resumo_triagem = resultado.get("resumo")
        try:
            cur.anos_experiencia = int(resultado.get("anos_experiencia") or 0)
        except (TypeError, ValueError):
            cur.anos_experiencia = None
        if resultado.get("confianca") == "ERRO_IA":
            cur.status = "erro"
            _abrir_pendencia(db, "ERRO_IA", f"não consegui triar o currículo #{cur.id} ({resultado.get('resumo', '')[:80]})",
                             rec, curriculo_id=cur.id)
        elif setor == "ARQUIVO_INVALIDO":
            cur.status = "invalido"
            eventos.registrar(db, "arquivo_invalido",
                              f"{rec.nome} descartou o arquivo #{cur.id}: não é um currículo ({resultado.get('motivo') or resultado.get('resumo') or 'inválido'})",
                              rec.id, curriculo_id=cur.id)
        else:
            cand = _vincular_candidato(db, resultado)
            cand.dado_sensivel = cand.dado_sensivel or setor in CATEGORIAS_SENSIVEIS
            cur.candidato_id = cand.id
            cur.status = "triado"
            eventos.registrar(db, "triagem_concluida",
                              f"{rec.nome} registrou o currículo de {primeiro_nome(cand.nome)} na pasta {setor}",
                              rec.id, curriculo_id=cur.id, candidato_id=cand.id, setor=setor)
            db.add(Tarefa(tipo="DISTRIBUICAO", curriculo_id=cur.id, agente_id=atlas(db).id))
        t.status, t.concluida_em = "concluida", agora()
        _liberar(db, rec)
        return resultado


def _vincular_candidato(db, r) -> Candidato:
    email = (r.get("email") or "").strip().lower() or None
    cand = db.scalar(select(Candidato).where(Candidato.email == email, Candidato.anonimizado_em.is_(None))) if email else None
    if not cand:
        cand = Candidato()
        db.add(cand)
    nome = (r.get("nome") or "").strip()
    cand.nome = nome if len(nome) >= 3 else (cand.nome or "Candidato")
    cand.email = email or cand.email
    cand.telefone = (r.get("numero") or "").strip() or cand.telefone
    cidade = (r.get("cidade") or "").strip()
    cand.cidade = cidade if cidade and cidade != "Não informado" else cand.cidade
    cand.ultima_atividade_em = agora()
    db.flush()
    return cand


# =========================================================================== Atlas: equipe e distribuição
def analistas_ativos(db):
    return db.scalars(select(Agente).where(Agente.funcao == "analista", Agente.ativo.is_(True))).all()


def carga(db, ag: Agente) -> int:
    return db.scalar(select(func.count()).select_from(Tarefa).where(
        Tarefa.agente_id == ag.id, Tarefa.status.in_(("pendente", "executando")))) or 0


def especialidade_para(db, setores: list[str]) -> Especialidade | None:
    melhor, pontos = None, 0
    for e in db.scalars(select(Especialidade)).all():
        p = len(set(setores) & set(e.setores or []))
        if p > pontos:
            melhor, pontos = e, p
    return melhor


def responsaveis(db, vaga: Vaga) -> list[Agente]:
    return db.scalars(select(Agente).join(VagaAgente, VagaAgente.agente_id == Agente.id)
                      .where(VagaAgente.vaga_id == vaga.id, Agente.ativo.is_(True))).all()


def garantir_responsavel(db, vaga: Vaga) -> Agente | None:
    atuais = responsaveis(db, vaga)
    if atuais:
        return min(atuais, key=lambda a: carga(db, a))
    sup = atlas(db)
    esp = especialidade_para(db, vaga.setores or [])
    candidatos = [a for a in analistas_ativos(db) if esp and a.especialidade_id == esp.id]
    if not candidatos and esp and config.ATIVACAO_AUTOMATICA:
        try:
            novo = ativar_agente(db, esp, por="Atlas")
        except ErroNegocio:
            novo = None
        if novo and responsaveis(db, vaga):
            return novo
        candidatos = [novo] if novo else []
    if not candidatos:
        _abrir_pendencia(db, "SEM_ESPECIALISTA",
                         f"nenhum agente da equipe cobre {esp.nome if esp else 'a área'} para a vaga “{vaga.titulo}”",
                         sup, vaga_id=vaga.id, especialidade_id=esp.id if esp else None)
        return None
    ag = min(candidatos, key=lambda a: carga(db, a))
    db.add(VagaAgente(vaga_id=vaga.id, agente_id=ag.id))
    db.flush()
    eventos.registrar(db, "vaga_atribuida", f"Atlas atribuiu a vaga “{vaga.titulo}” a {ag.nome}", ag.id, vaga_id=vaga.id)
    return ag


def _encaminhar(db, cur: Curriculo, vaga: Vaga) -> bool:
    """Cria a candidatura (uma por currículo × vaga) e a tarefa de análise."""
    if db.scalar(select(Candidatura).where(Candidatura.curriculo_id == cur.id, Candidatura.vaga_id == vaga.id)):
        return False
    cand = Candidatura(curriculo_id=cur.id, vaga_id=vaga.id, candidato_id=cur.candidato_id)
    db.add(cand)
    db.flush()
    ag = garantir_responsavel(db, vaga)
    nome = primeiro_nome(cur.candidato.nome if cur.candidato else "Candidato")
    if ag:
        db.add(Tarefa(tipo="ANALISE", candidatura_id=cand.id, agente_id=ag.id))
        eventos.registrar(db, "candidatura_encaminhada", f"Atlas encaminhou {nome} para a vaga “{vaga.titulo}” ({ag.nome})",
                          atlas(db).id, vaga_id=vaga.id, candidatura_id=cand.id, destino_id=ag.id)
    else:
        eventos.registrar(db, "candidatura_em_espera", f"{nome} aguarda um agente para a vaga “{vaga.titulo}”",
                          atlas(db).id, vaga_id=vaga.id, candidatura_id=cand.id)
    return True


def executar_distribuicao(tarefa_id: int):
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        cur = db.get(Curriculo, t.curriculo_id)
        sup = atlas(db)
        _ocupar(sup, t, f"distribuindo o currículo #{cur.id}")
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        cur = db.get(Curriculo, t.curriculo_id)
        sup = atlas(db)
        nome = primeiro_nome(cur.candidato.nome if cur.candidato else "Candidato")
        if cur.setor == "OUTROS":
            _abrir_pendencia(db, "TRIAGEM_MANUAL", f"o currículo de {nome} (#{cur.id}) não se encaixou em nenhuma pasta; defina a área",
                             sup, curriculo_id=cur.id)
        else:
            vagas = [v for v in db.scalars(select(Vaga).where(Vaga.status == "aberta")).all() if cur.setor in (v.setores or [])]
            novas = sum(_encaminhar(db, cur, v) for v in vagas)
            if not vagas:
                eventos.registrar(db, "banco_talentos",
                                  f"Atlas guardou {nome} no banco de talentos: nenhuma vaga aberta para {cur.setor}",
                                  sup.id, curriculo_id=cur.id)
            elif not novas:
                eventos.registrar(db, "sem_duplicidade", f"Atlas viu que {nome} já está nas vagas compatíveis",
                                  sup.id, curriculo_id=cur.id)
        t.status, t.concluida_em = "concluida", agora()
        _liberar(db, sup)


def ativar_agente(db, esp: Especialidade, por="Atlas", nome: str | None = None) -> Agente | None:
    inativo = db.scalar(select(Agente).where(Agente.funcao == "analista", Agente.especialidade_id == esp.id,
                                             Agente.ativo.is_(False)))
    ocupadas = {a.estacao for a in analistas_ativos(db)}
    livres = [i for i in range(MAX_ESTACOES) if i not in ocupadas]
    if not livres:
        raise ErroNegocio("Todas as estações de trabalho estão ocupadas.")
    if inativo and not nome:
        ag = inativo
        ag.ativo, ag.estacao, ag.ativado_em = True, livres[0], agora()
    else:
        usados = {a.nome for a in db.scalars(select(Agente)).all()}
        nome = nome or next((n for n in (esp.nomes_sugeridos or []) if n not in usados), f"Agente {len(usados) + 1}")
        ag = Agente(nome=nome, funcao="analista", especialidade_id=esp.id, estacao=livres[0],
                    aparencia=aparencia_para(nome), ativado_em=agora(),
                    personalidade=f"Analista dedicado a {esp.nome.lower()}, objetivo e cuidadoso com evidências.")
        db.add(ag)
    db.flush()
    eventos.registrar(db, "agente_ativado", f"{por} ativou {ag.nome} ({esp.nome})", ag.id, especialidade=esp.codigo)
    # assume vagas sem responsável dessa especialidade e resolve as pendências correspondentes
    for p in db.scalars(select(Pendencia).where(Pendencia.tipo == "SEM_ESPECIALISTA", Pendencia.status == "aberta",
                                                Pendencia.especialidade_id == esp.id)).all():
        p.status, p.resolvido_em = "resolvida", agora()
        vaga = db.get(Vaga, p.vaga_id)
        if vaga and not responsaveis(db, vaga):
            db.add(VagaAgente(vaga_id=vaga.id, agente_id=ag.id))
            db.flush()
            eventos.registrar(db, "vaga_atribuida", f"Atlas atribuiu a vaga “{vaga.titulo}” a {ag.nome}", ag.id, vaga_id=vaga.id)
            for c in db.scalars(select(Candidatura).where(Candidatura.vaga_id == vaga.id, Candidatura.status == "na_fila")).all():
                if not db.scalar(select(Tarefa).where(Tarefa.candidatura_id == c.id, Tarefa.status.in_(("pendente", "executando")))):
                    db.add(Tarefa(tipo="ANALISE", candidatura_id=c.id, agente_id=ag.id))
    return ag


# =========================================================================== análises
def _esp_dicts(db):
    return [{"codigo": e.codigo, "nome": e.nome, "setores": e.setores} for e in db.scalars(select(Especialidade)).all()]


def executar_analise(tarefa_id: int):
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        c = db.get(Candidatura, t.candidatura_id)
        ag = db.get(Agente, t.agente_id)
        if c.vaga.status != "aberta":
            t.status = "cancelada"
            return
        if db.scalar(select(Analise).where(Analise.candidatura_id == c.id, Analise.agente_id == ag.id)):
            t.status, t.concluida_em = "concluida", agora()  # já analisado por este agente: não duplica
            return
        ocioso = not ag.ultima_atividade_em or agora() - ag.ultima_atividade_em > timedelta(seconds=20)
        if ocioso:
            eventos.registrar(db, "curriculos_retirados", f"{ag.nome} foi à recepção buscar currículos", ag.id)
        nome = primeiro_nome(c.candidato.nome if c.candidato else "Candidato")
        _ocupar(ag, t, f"analisando {nome} para “{c.vaga.titulo}”")
        if t.motivo == "" and c.status == "na_fila":
            c.status = "em_analise"
        cur = c.curriculo
        agente = {"nome": ag.nome, "personalidade": ag.personalidade,
                  "especialidade": ag.especialidade.nome if ag.especialidade else None,
                  "especialidade_codigo": ag.especialidade.codigo if ag.especialidade else None}
        vaga = {"titulo": c.vaga.titulo, "area": c.vaga.area, "local": c.vaga.local,
                "descricao": c.vaga.descricao, "requisitos": c.vaga.requisitos}
        texto, setor = cur.texto, cur.setor
        arquivo = None if texto else (ler_arquivo(cur.caminho), cur.mime)
        esps = _esp_dicts(db)
        inicio = agora()
    try:
        resultado, modelo = ia.analisar(agente, vaga, texto, arquivo, esps, setor)
        falha = None
    except Exception as e:
        resultado, modelo, falha = None, "", e
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        c = db.get(Candidatura, t.candidatura_id)
        ag = db.get(Agente, t.agente_id)
        nome = primeiro_nome(c.candidato.nome if c.candidato else "Candidato")
        if falha:
            t.erro = f"{type(falha).__name__}: {str(falha)[:200]}"
            if t.tentativas < 3:
                t.status = "pendente"
            else:
                t.status = "erro"
                c.status = "erro"
                _abrir_pendencia(db, "ERRO_IA", f"não consegui analisar o currículo de {nome} para “{c.vaga.titulo}”",
                                 ag, candidatura_id=c.id)
            _liberar(db, ag)
            return
        segunda = bool(t.motivo)
        db.add(Analise(candidatura_id=c.id, agente_id=ag.id, tipo="segunda_opiniao" if segunda else "principal",
                       compatibilidade=resultado["compatibilidade"], recomendado=bool(resultado.get("recomendado")),
                       avaliacao_humana=bool(resultado.get("avaliacao_humana")), resultado=resultado, modelo=modelo,
                       duracao_ms=int((agora() - inicio).total_seconds() * 1000)))
        if not segunda:
            c.status = "analisada"
        if c.candidato:
            c.candidato.ultima_atividade_em = agora()
        rotulo = {"ALTA": "alta", "MEDIA": "média", "BAIXA": "baixa"}[resultado["compatibilidade"]]
        extra = " · recomendado" if resultado.get("recomendado") else (" · avaliação humana" if resultado.get("avaliacao_humana") else "")
        if segunda:
            eventos.registrar(db, "colaboracao", f"{ag.nome} deu a segunda opinião sobre {nome} (“{c.vaga.titulo}”): compatibilidade {rotulo}",
                              ag.id, candidatura_id=c.id)
        else:
            eventos.registrar(db, "analise_concluida", f"{ag.nome} concluiu a análise de {nome} para “{c.vaga.titulo}”: compatibilidade {rotulo}{extra}",
                              ag.id, candidatura_id=c.id, vaga_id=c.vaga_id)
        t.status, t.concluida_em = "concluida", agora()
        if not segunda and resultado.get("pedir_segunda_opiniao"):
            _segunda_opiniao(db, c, ag, resultado.get("segunda_opiniao_especialidade", ""), resultado.get("segunda_opiniao_motivo", ""))
        _liberar(db, ag)


def _segunda_opiniao(db, c: Candidatura, solicitante: Agente, codigo: str, motivo: str):
    """O agente pede apoio ao Atlas; o Atlas decide se outro especialista participa."""
    nome = primeiro_nome(c.candidato.nome if c.candidato else "Candidato")
    eventos.registrar(db, "colaboracao", f"{solicitante.nome} pediu ao Atlas uma segunda opinião sobre {nome}: {motivo or 'perfil de outra área'}",
                      solicitante.id, candidatura_id=c.id)
    esp = db.scalar(select(Especialidade).where(Especialidade.codigo == codigo))
    apoio = [a for a in analistas_ativos(db) if esp and a.especialidade_id == esp.id and a.id != solicitante.id]
    sup = atlas(db)
    if not apoio:
        eventos.registrar(db, "colaboracao", f"Atlas registrou o pedido, mas não há agente ativo de {esp.nome if esp else codigo or 'outra área'}",
                          sup.id, candidatura_id=c.id)
        return
    ag = min(apoio, key=lambda a: carga(db, a))
    ja = db.scalar(select(Analise).where(Analise.candidatura_id == c.id, Analise.agente_id == ag.id)) or \
        db.scalar(select(Tarefa).where(Tarefa.candidatura_id == c.id, Tarefa.agente_id == ag.id))
    if ja:
        eventos.registrar(db, "colaboracao", f"Atlas viu que {ag.nome} já avaliou {nome} nesta vaga e reaproveitou a análise",
                          sup.id, candidatura_id=c.id)
        return
    db.add(Tarefa(tipo="ANALISE", candidatura_id=c.id, agente_id=ag.id, motivo=motivo or "segunda opinião"))
    eventos.registrar(db, "colaboracao", f"Atlas encaminhou o pedido de {solicitante.nome} a {ag.nome}", sup.id,
                      candidatura_id=c.id, destino_id=ag.id)


# =========================================================================== fila (usada pelo trabalhador)
def reservar_tarefas(ocupadas: set[str]) -> list[tuple[str, int, str]]:
    """Reserva no máximo uma tarefa por 'raia' livre (recepção, Atlas e cada analista)."""
    saida = []
    with sessao() as db:
        pend = db.scalars(select(Tarefa).where(Tarefa.status == "pendente").order_by(Tarefa.id)).all()
        for t in pend:
            raia = {"TRIAGEM": "recepcao", "DISTRIBUICAO": "atlas"}.get(t.tipo, f"ag{t.agente_id}")
            if raia in ocupadas or any(r == raia for r, _, _ in saida):
                continue
            if t.tentativas and t.iniciada_em and agora() - t.iniciada_em < timedelta(seconds=10 * t.tentativas):
                continue  # espera antes de tentar de novo
            t.status, t.iniciada_em = "executando", agora()
            t.tentativas += 1
            saida.append((raia, t.id, t.tipo))
    return saida


def executar(tarefa_id: int, tipo: str):
    try:
        {"TRIAGEM": executar_triagem, "DISTRIBUICAO": executar_distribuicao, "ANALISE": executar_analise}[tipo](tarefa_id)
    except Exception:
        log.exception("Tarefa %s (%s) falhou", tarefa_id, tipo)
        with sessao() as db:
            t = db.get(Tarefa, tarefa_id)
            t.status = "pendente" if t.tentativas < 3 else "erro"
            ag = db.get(Agente, t.agente_id) if t.agente_id else None
            if ag:
                _liberar(db, ag)


def recuperar_fila():
    """Ao reiniciar: tarefas interrompidas voltam para a fila e ninguém fica 'trabalhando' sem tarefa."""
    with sessao() as db:
        for t in db.scalars(select(Tarefa).where(Tarefa.status == "executando")).all():
            t.status = "pendente"
        for a in db.scalars(select(Agente)).all():
            a.tarefa_atual_id = None


# =========================================================================== estados e leituras
def estado_de(db, ag: Agente) -> tuple[str, str]:
    if not ag.ativo:
        return "inativo", ""
    if ag.tarefa_atual_id:
        return "trabalhando", ag.atividade
    p = db.scalar(select(Pendencia).where(Pendencia.agente_id == ag.id, Pendencia.status == "aberta"))
    if p:
        return "pendencia", p.descricao
    if carga(db, ag):
        return "trabalhando", "pegando o próximo currículo"
    t = agora()
    if ag.concluido_em and t - ag.concluido_em < timedelta(seconds=config.SEGUNDOS_CONCLUIDO):
        return "concluido", {"recepcao": "terminou a triagem da fila", "supervisor": "distribuiu todos os currículos"}.get(
            ag.funcao, "terminou as análises da fila")
    tem_vagas = db.scalar(select(func.count()).select_from(VagaAgente).where(VagaAgente.agente_id == ag.id))
    if ag.funcao == "analista":
        ocioso = t - (ag.ultima_atividade_em or ag.ativado_em or t)
        if ocioso >= timedelta(seconds=config.SEGUNDOS_PARA_DESCANSO):
            return "descansando", ""
        if not tem_vagas:
            return "disponivel", "disponível para novas vagas"
        return "aguardando", "aguardando novos currículos"
    return "aguardando", "aguardando novos currículos" if ag.funcao == "recepcao" else "acompanhando a equipe"


def agente_json(db, ag: Agente) -> dict:
    est, ativ = estado_de(db, ag)
    vagas = db.scalars(select(VagaAgente.vaga_id).where(VagaAgente.agente_id == ag.id)).all()
    return {"id": ag.id, "nome": ag.nome, "funcao": ag.funcao,
            "funcao_rotulo": {"supervisor": "Supervisor da equipe", "recepcao": "Recepção de currículos",
                              "analista": "Analista de currículos"}[ag.funcao],
            "especialidade": ag.especialidade.nome if ag.especialidade else
            ("Supervisão e coordenação" if ag.funcao == "supervisor" else "Recebimento e triagem de currículos"),
            "especialidade_codigo": ag.especialidade.codigo if ag.especialidade else None,
            "estado": est, "atividade": ativ, "estacao": ag.estacao, "aparencia": ag.aparencia,
            "vagas": list(vagas),
            "pendencias": db.scalar(select(func.count()).select_from(Pendencia).where(
                Pendencia.agente_id == ag.id, Pendencia.status == "aberta"))}


def _contagem_vaga(db, v: Vaga) -> dict:
    rec = db.scalar(select(func.count()).select_from(Candidatura).where(Candidatura.vaga_id == v.id)) or 0
    ana = db.scalar(select(func.count()).select_from(Candidatura).where(Candidatura.vaga_id == v.id,
                                                                         Candidatura.status == "analisada")) or 0
    return {"recebidos": rec, "analisados": ana, "pendentes": rec - ana}


def vaga_json(db, v: Vaga, completo=False) -> dict:
    resp = responsaveis(db, v)
    d = {"id": v.id, "titulo": v.titulo, "area": v.area, "local": v.local, "status": v.status,
         "setores": v.setores, "responsaveis": [{"id": a.id, "nome": a.nome} for a in resp],
         "sem_responsavel": v.status == "aberta" and not resp, **_contagem_vaga(db, v)}
    d["situacao"] = ("sem responsável" if d["sem_responsavel"] else v.status if v.status != "aberta" else
                     "em análise" if d["pendentes"] else "aguardando currículos" if not d["recebidos"] else "em dia")
    if completo:
        d.update(descricao=v.descricao, requisitos=v.requisitos)
    return d


def quadro(db) -> dict:
    vagas = db.scalars(select(Vaga)).all()
    abertas = [v for v in vagas if v.status == "aberta"]
    hoje = agora().replace(hour=0, minute=0, second=0, microsecond=0)
    vj = [vaga_json(db, v) for v in abertas]
    return {
        "vagas_abertas": len(abertas),
        "vagas_sem_responsavel": sum(1 for v in vj if v["sem_responsavel"]),
        "vagas_em_processamento": sum(1 for v in vj if v["pendentes"]),
        "curriculos_recebidos": db.scalar(select(func.count()).select_from(Curriculo).where(Curriculo.status != "anonimizado")) or 0,
        "curriculos_triados": db.scalar(select(func.count()).select_from(Curriculo).where(Curriculo.status == "triado")) or 0,
        "analises_concluidas": db.scalar(select(func.count()).select_from(Analise)) or 0,
        "analises_hoje": db.scalar(select(func.count()).select_from(Analise).where(Analise.criado_em >= hoje)) or 0,
        "tarefas_pendentes": db.scalar(select(func.count()).select_from(Tarefa).where(Tarefa.status.in_(("pendente", "executando")))) or 0,
        "pendencias": db.scalar(select(func.count()).select_from(Pendencia).where(Pendencia.status == "aberta")) or 0,
    }


def escritorio(db) -> dict:
    ags = db.scalars(select(Agente).where(Agente.ativo.is_(True)).order_by(Agente.id)).all()
    fila_triagem = db.scalar(select(func.count()).select_from(Tarefa).where(
        Tarefa.tipo == "TRIAGEM", Tarefa.status.in_(("pendente", "executando")))) or 0
    fila_analise = db.scalar(select(func.count()).select_from(Tarefa).where(
        Tarefa.tipo == "ANALISE", Tarefa.status == "pendente")) or 0
    evs = db.scalars(select(Evento).order_by(Evento.id.desc()).limit(25)).all()
    return {"agentes": [agente_json(db, a) for a in ags],
            "fila_triagem": fila_triagem, "fila_analise": fila_analise,
            "vagas": [vaga_json(db, v) for v in db.scalars(select(Vaga).order_by(Vaga.id)).all()],
            "quadro": quadro(db),
            "eventos": [eventos.serializar(e) for e in evs],
            "ia_modo": config.IA_MODO}


def _analise_json(a: Analise, c: Candidatura) -> dict:
    r = a.resultado or {}
    return {"analise_id": a.id, "candidatura_id": c.id, "nome": c.candidato.nome if c.candidato else "Candidato",
            "vaga_id": c.vaga_id, "vaga": c.vaga.titulo, "tipo": a.tipo,
            "compatibilidade": a.compatibilidade, "recomendado": a.recomendado, "avaliacao_humana": a.avaliacao_humana,
            "resumo": r.get("resumo_profissional", ""), "experiencias": r.get("experiencias_relevantes", []),
            "competencias": r.get("competencias", []), "atendidos": r.get("requisitos_atendidos", []),
            "a_verificar": r.get("a_verificar", []), "incompatibilidades": r.get("incompatibilidades", []),
            "justificativa": r.get("justificativa", ""), "analisado_em": iso(a.criado_em), "modelo": a.modelo}


def _decisao_atual(db, cand_id):
    d = db.scalar(select(Decisao).where(Decisao.candidatura_id == cand_id).order_by(Decisao.id.desc()))
    return d.decisao if d else None


def candidatos_do_agente(db, ag_id: int, vaga_id: int | None = None) -> list[dict]:
    q = select(Analise, Candidatura).join(Candidatura, Analise.candidatura_id == Candidatura.id).where(Analise.agente_id == ag_id)
    if vaga_id:
        q = q.where(Candidatura.vaga_id == vaga_id)
    ordem = {"ALTA": 0, "MEDIA": 1, "BAIXA": 2}
    saida = []
    for a, c in db.execute(q).all():
        if c.candidato and c.candidato.anonimizado_em:
            continue
        j = _analise_json(a, c)
        j["decisao"] = _decisao_atual(db, c.id)
        saida.append(j)
    saida.sort(key=lambda j: (not j["recomendado"], not j["avaliacao_humana"], ordem.get(j["compatibilidade"], 3), j["nome"]))
    return saida


def detalhe_agente(db, ag: Agente) -> dict:
    base = agente_json(db, ag)
    vagas = [db.get(Vaga, i) for i in base["vagas"]]
    vj = []
    for v in vagas:
        d = vaga_json(db, v, completo=True)
        d["descricao"] = (d["descricao"] or "")[:220]
        vj.append(d)
    principais = select(func.count()).select_from(Analise).where(Analise.agente_id == ag.id)
    base["estatisticas"] = {
        "recebidos": sum(v["recebidos"] for v in vj),
        "analisados": db.scalar(principais.where(Analise.tipo == "principal")) or 0,
        "segundas_opinioes": db.scalar(principais.where(Analise.tipo == "segunda_opiniao")) or 0,
        "pendentes": carga(db, ag),
        "recomendados": db.scalar(principais.where(Analise.recomendado.is_(True))) or 0,
        "vagas_em_andamento": sum(1 for v in vj if v["status"] == "aberta"),
    }
    base["vagas_detalhe"] = vj
    base["historico"] = [eventos.serializar(e) for e in db.scalars(
        select(Evento).where(Evento.agente_id == ag.id).order_by(Evento.id.desc()).limit(8)).all()]
    base["pendencias_abertas"] = [{"id": p.id, "tipo": p.tipo, "descricao": p.descricao} for p in db.scalars(
        select(Pendencia).where(Pendencia.agente_id == ag.id, Pendencia.status == "aberta")).all()]
    return base


def contexto_chat(db, ag: Agente) -> dict:
    d = detalhe_agente(db, ag)
    cands = candidatos_do_agente(db, ag.id)[:40]
    return {
        "agente": {k: d[k] for k in ("nome", "funcao", "especialidade", "estado", "atividade")},
        "estatisticas": d["estatisticas"],
        "vagas": [{k: v[k] for k in ("titulo", "status", "recebidos", "analisados", "pendentes", "requisitos", "situacao")}
                  for v in d["vagas_detalhe"]],
        "candidatos": [{k: c[k] for k in ("nome", "vaga", "compatibilidade", "recomendado", "avaliacao_humana",
                                          "a_verificar", "justificativa", "decisao")} | {
            "evidencias": [f'{x.get("requisito")}: “{x.get("evidencia")}”' for x in c["atendidos"]][:5]} for c in cands],
        "pendencias": d["pendencias_abertas"],
        "atividade_recente": [f'{e["ts"]} {e["mensagem"]}' for e in d["historico"]],
        "fila_recepcao": db.scalar(select(func.count()).select_from(Tarefa).where(
            Tarefa.tipo == "TRIAGEM", Tarefa.status.in_(("pendente", "executando")))) or 0,
        "curriculos_registrados": db.scalar(select(func.count()).select_from(Curriculo)) or 0,
        "quadro": quadro(db),
    }


def detalhe_candidatura(db, c: Candidatura) -> dict:
    cand, cur = c.candidato, c.curriculo
    analises = db.scalars(select(Analise).where(Analise.candidatura_id == c.id).order_by(Analise.id)).all()
    nomes = {a.id: a.nome for a in db.scalars(select(Agente)).all()}
    return {
        "candidatura_id": c.id, "status": c.status, "decisao": _decisao_atual(db, c.id),
        "candidato": {"nome": cand.nome if cand else "Candidato", "cidade": cand.cidade if cand else None,
                      "email": cand.email if cand else None, "telefone": cand.telefone if cand else None,
                      "dado_sensivel": bool(cand and cand.dado_sensivel)},
        "vaga": {"id": c.vaga.id, "titulo": c.vaga.titulo, "requisitos": c.vaga.requisitos},
        "curriculo": {"id": cur.id, "arquivo": cur.arquivo_nome, "mime": cur.mime, "texto": cur.texto,
                      "setor": cur.setor, "confianca": cur.confianca, "anos_experiencia": cur.anos_experiencia,
                      "resumo_triagem": cur.resumo_triagem, "recebido_em": iso(cur.recebido_em), "origem": cur.origem},
        "analises": [_analise_json(a, c) | {"agente": nomes.get(a.agente_id, "?")} for a in analises],
        "decisoes": [{"decisao": d.decisao, "observacao": d.observacao, "em": iso(d.criado_em)} for d in db.scalars(
            select(Decisao).where(Decisao.candidatura_id == c.id).order_by(Decisao.id)).all()],
    }


# =========================================================================== vagas e pendências
def criar_vaga(db, dados: dict) -> Vaga:
    v = Vaga(titulo=dados["titulo"].strip(), area=dados.get("area", ""), local=dados.get("local", ""),
             descricao=dados.get("descricao", ""), requisitos=[r for r in dados.get("requisitos", []) if r.strip()],
             setores=dados.get("setores", []))
    db.add(v)
    db.flush()
    eventos.registrar(db, "vaga_criada", f"Nova vaga aberta: “{v.titulo}”", atlas(db).id, vaga_id=v.id)
    garantir_responsavel(db, v)
    _reaproveitar_banco(db, v)
    return v


def _reaproveitar_banco(db, v: Vaga):
    limite = agora() - timedelta(days=config.RETENCAO_DIAS)
    curs = db.scalars(select(Curriculo).where(Curriculo.status == "triado", Curriculo.setor.in_(v.setores or ["-"]),
                                              Curriculo.recebido_em >= limite)).all()
    n = sum(_encaminhar(db, c, v) for c in curs)
    if n:
        eventos.registrar(db, "banco_talentos", f"Atlas encontrou {n} currículo(s) do banco de talentos para “{v.titulo}”",
                          atlas(db).id, vaga_id=v.id)


def mudar_status_vaga(db, v: Vaga, status: str):
    if status not in STATUS_VAGA:
        raise ErroNegocio("Status inválido.")
    anterior, v.status = v.status, status
    ids = select(Candidatura.id).where(Candidatura.vaga_id == v.id)
    if status != "aberta":
        for t in db.scalars(select(Tarefa).where(Tarefa.candidatura_id.in_(ids), Tarefa.status == "pendente")).all():
            t.status = "cancelada"
    elif anterior != "aberta":
        ag = garantir_responsavel(db, v)
        for c in db.scalars(select(Candidatura).where(Candidatura.vaga_id == v.id, Candidatura.status.in_(("na_fila", "em_analise")))).all():
            if ag and not db.scalar(select(Tarefa).where(Tarefa.candidatura_id == c.id, Tarefa.status.in_(("pendente", "executando")))):
                db.add(Tarefa(tipo="ANALISE", candidatura_id=c.id, agente_id=ag.id))
    eventos.registrar(db, "vaga_status", f"A vaga “{v.titulo}” agora está {status}", atlas(db).id, vaga_id=v.id, status=status)


def resolver_pendencia(db, p: Pendencia, acao: str, usuario_id: int, setor: str | None = None):
    if p.status != "aberta":
        raise ErroNegocio("Pendência já resolvida.")
    if acao == "ativar":
        esp = db.get(Especialidade, p.especialidade_id) if p.especialidade_id else None
        if not esp:
            raise ErroNegocio("Pendência sem especialidade associada.")
        ativar_agente(db, esp, por="O recrutador")  # resolve a pendência internamente
        p.resolvido_por_id = usuario_id
        return
    if acao == "reprocessar":
        t = db.scalar(select(Tarefa).where(or_(Tarefa.curriculo_id == p.curriculo_id if p.curriculo_id else False,
                                               Tarefa.candidatura_id == p.candidatura_id if p.candidatura_id else False),
                                           Tarefa.status == "erro").order_by(Tarefa.id.desc()))
        if p.curriculo_id and not t:
            db.add(Tarefa(tipo="TRIAGEM", curriculo_id=p.curriculo_id, agente_id=recepcao(db).id))
        elif t:
            t.status, t.tentativas, t.erro = "pendente", 0, None
    elif acao == "definir_setor":
        cur = db.get(Curriculo, p.curriculo_id)
        if setor not in triagem.CATEGORIAS_DISPONIVEIS or setor == "OUTROS":
            raise ErroNegocio("Escolha uma pasta válida.")
        cur.setor = setor
        db.add(Tarefa(tipo="DISTRIBUICAO", curriculo_id=cur.id, agente_id=atlas(db).id))
    elif acao != "resolver":
        raise ErroNegocio("Ação desconhecida.")
    p.status, p.resolvido_em, p.resolvido_por_id = "resolvida", agora(), usuario_id
    eventos.registrar(db, "pendencia_resolvida", f"Pendência resolvida pelo recrutador: {p.descricao}", p.agente_id, pendencia_id=p.id)


# =========================================================================== LGPD
def exportar_titular(db, cand: Candidato) -> dict:
    curs = db.scalars(select(Curriculo).where(Curriculo.candidato_id == cand.id)).all()
    cs = db.scalars(select(Candidatura).where(Candidatura.candidato_id == cand.id)).all()
    return {
        "titular": {"nome": cand.nome, "email": cand.email, "telefone": cand.telefone, "cidade": cand.cidade,
                    "registrado_em": iso(cand.criado_em), "ultima_atividade_em": iso(cand.ultima_atividade_em)},
        "finalidade": "Recrutamento e seleção para vagas da Baly",
        "retencao_dias": config.RETENCAO_DIAS,
        "compartilhamento": "Provedor de IA (Google Gemini) como operador, quando IA_MODO=gemini" if config.IA_MODO == "gemini" else "Nenhum",
        "curriculos": [{"id": c.id, "arquivo": c.arquivo_nome, "recebido_em": iso(c.recebido_em), "origem": c.origem,
                        "pasta": c.setor, "texto": c.texto, "triagem": c.triagem_json} for c in curs],
        "candidaturas": [detalhe_candidatura(db, c) for c in cs],
    }


def anonimizar_titular(db, cand: Candidato, motivo: str):
    nome_antigo = cand.nome
    for c in db.scalars(select(Curriculo).where(Curriculo.candidato_id == cand.id)).all():
        apagar_arquivo(c.caminho)
        c.caminho, c.texto, c.triagem_json, c.resumo_triagem = None, None, None, None
        c.arquivo_nome, c.status, c.sha256 = "removido", "anonimizado", f"anon-{c.id}"
    ids = select(Candidatura.id).where(Candidatura.candidato_id == cand.id)
    for a in db.scalars(select(Analise).where(Analise.candidatura_id.in_(ids))).all():
        a.resultado = {}
    for d in db.scalars(select(Decisao).where(Decisao.candidatura_id.in_(ids))).all():
        d.observacao = ""
    for t in db.scalars(select(Tarefa).where(Tarefa.candidatura_id.in_(ids), Tarefa.status == "pendente")).all():
        t.status = "cancelada"
    if nome_antigo and len(nome_antigo) >= 3:
        for chave in {nome_antigo, primeiro_nome(nome_antigo)}:
            for e in db.scalars(select(Evento).where(Evento.mensagem.contains(chave))).all():
                e.mensagem = e.mensagem.replace(chave, "titular anonimizado")
    cand.nome, cand.email, cand.telefone, cand.cidade = "Titular anonimizado", None, None, None
    cand.anonimizado_em = agora()
    eventos.registrar(db, "lgpd", f"Dados de um titular foram anonimizados ({motivo})", None, candidato_id=cand.id)


def aplicar_retencao() -> int:
    limite = agora() - timedelta(days=config.RETENCAO_DIAS)
    with sessao() as db:
        vencidos = db.scalars(select(Candidato).where(Candidato.anonimizado_em.is_(None),
                                                      Candidato.ultima_atividade_em < limite)).all()
        for c in vencidos:
            anonimizar_titular(db, c, f"prazo de retenção de {config.RETENCAO_DIAS} dias")
        return len(vencidos)


# =========================================================================== carga inicial
def semear_basico(db):
    from .seguranca import gerar_hash
    from .models import Usuario
    for e in ESPECIALIDADES:
        if not db.scalar(select(Especialidade).where(Especialidade.codigo == e["codigo"])):
            db.add(Especialidade(codigo=e["codigo"], nome=e["nome"], descricao=e["descricao"],
                                 setores=e["setores"], nomes_sugeridos=e["nomes"]))
    db.flush()
    if not db.scalar(select(Agente).where(Agente.funcao == "supervisor")):
        esp = {e.codigo: e for e in db.scalars(select(Especialidade)).all()}
        db.add_all([
            Agente(nome="Atlas", funcao="supervisor", personalidade="Coordenador calmo e objetivo; distribui o trabalho pela especialidade e pela carga de cada agente.",
                   aparencia=aparencia_para("Atlas", pele="#D9A27A", cabelo="#1E1A16", estilo="curto", camisa="#FFCD00", oculos=True, barba=True)),
            Agente(nome="Bia", funcao="recepcao", personalidade="Recepcionista organizada; registra cada currículo e faz a triagem inicial.",
                   aparencia=aparencia_para("Bia", pele="#B97D55", cabelo="#1E1A16", estilo="black", camisa="#8E6BBF", oculos=False, barba=False)),
            Agente(nome="Sofia", funcao="analista", especialidade_id=esp["administrativo"].id, estacao=0,
                   personalidade="Analista atenta a detalhes e a rotinas administrativas.",
                   aparencia=aparencia_para("Sofia", pele="#F6D1B1", cabelo="#6E3C1E", estilo="longo", camisa="#3B6EC8", oculos=False, barba=False)),
            Agente(nome="Flávio", funcao="analista", especialidade_id=esp["operacional"].id, estacao=1,
                   personalidade="Analista prático, conhece bem a rotina de produção e logística.",
                   aparencia=aparencia_para("Flávio", pele="#EEC49C", cabelo="#3B2618", estilo="topete", camisa="#3CAA6E", oculos=False, barba=True)),
        ])
    if config.ADMIN_SENHA and not db.scalar(select(Usuario).where(Usuario.email == config.ADMIN_EMAIL.lower())):
        db.add(Usuario(nome=config.ADMIN_NOME, email=config.ADMIN_EMAIL.lower(), senha_hash=gerar_hash(config.ADMIN_SENHA), papel="admin"))
