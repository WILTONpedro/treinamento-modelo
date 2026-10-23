"""Regras do negócio: recepção, triagem, Atlas (supervisor e contratações), análises,
consultas entre colegas, estados e LGPD.

Cada etapa é uma Tarefa real na fila. O estado de cada agente é derivado das tarefas
e registros — o escritório nunca mostra trabalho que não está acontecendo. Os pensamentos
e falas dos agentes são eventos gerados a partir desse trabalho (o que a leitura encontrou,
a dúvida enviada ao colega, a resposta dele, a contratação feita pelo Atlas).
"""
import hashlib
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from sqlalchemy import case, delete, func, select, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, joinedload

from . import acervo, config, eventos, ia, triagem, uso_ia
from .catalogo import CATEGORIAS_SENSIVEIS, ESPECIALIDADES, NOMES_EXTRAS, aparencia_para
from .db import sessao
from .models import (Agente, Analise, Candidato, Candidatura, Configuracao, Contratacao, Curriculo, Decisao,
                     Especialidade, Evento, Pendencia, Tarefa, Vaga, VagaAgente, agora, iso)
from .personalidade import bordao, falar, maiuscula, perfil_para, publico
from .arquivos import apagar_arquivo, ler_arquivo, salvar_arquivo

log = logging.getLogger("hub.servicos")
MAX_ESTACOES = 12
STATUS_VAGA = ("aberta", "suspensa", "encerrada", "cancelada")
TAREFAS_DE_ANALISTA = ("ANALISE", "CONSULTA", "REVISAO")
TAREFAS_COM_IA = ("TRIAGEM", "ANALISE", "CONSULTA", "REVISAO")
FALAS = ("pensamento", "conversa")  # eventos com retenção curta (podem citar o primeiro nome do candidato)
ROTULO_COMPAT = {"ALTA": "alta", "MEDIA": "média", "BAIXA": "baixa"}
# contratações e dispensas mexem nas estações de trabalho: uma de cada vez
trava_equipe = threading.Lock()


class ErroNegocio(Exception):
    pass


# =========================================================================== utilidades
def primeiro_nome(nome):
    partes = (nome or "Candidato").split()
    return partes[0] + (f" {partes[-1][0]}." if len(partes) > 1 else "")


def rotulo(c: Candidatura) -> str:
    return primeiro_nome(c.candidato.nome if c.candidato else "Candidato")


def recepcao(db) -> Agente:
    return db.scalar(select(Agente).where(Agente.funcao == "recepcao"))


def atlas(db) -> Agente:
    return db.scalar(select(Agente).where(Agente.funcao == "supervisor"))


def perfil(ag: Agente) -> dict:
    return ag.perfil or perfil_para(ag.nome, ag.especialidade.nome if ag.especialidade else None)


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


def _pensar(db, ag: Agente | None, texto: str, **dados):
    if ag and texto:
        eventos.registrar(db, "pensamento", texto[:160], ag.id, **dados)


def pensar(agente_id: int, texto: str, **dados):
    """Pensamento emitido durante a leitura da IA (fora da transação principal)."""
    if texto:
        with sessao() as db:
            eventos.registrar(db, "pensamento", texto[:160], agente_id, **dados)


def _dizer(db, ag: Agente, texto: str, para: Agente | None = None, **dados):
    """Fala de um agente (para um colega ou para o time) — vira balão no escritório e mensagem no canal da equipe."""
    if ag and texto:  # os nomes ficam no evento: o feed continua legível depois de uma dispensa
        eventos.registrar(db, "conversa", texto[:500], ag.id, destino_id=para.id if para else None,
                          de=ag.nome, para=para.nome if para else None, **dados)


def _devolver_a_fila(tarefa_id: int):
    """IA pausada (limite diário, sem créditos...): a tarefa volta para a fila sem contar como tentativa."""
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        t.status, t.tentativas, t.iniciada_em = "pendente", max(0, t.tentativas - 1), None
        ag = db.get(Agente, t.agente_id) if t.agente_id else None
        if ag:
            _liberar(db, ag)


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


def _resolver_pendencias(db, tipo, **refs):
    filtro = [Pendencia.tipo == tipo, Pendencia.status == "aberta"] + [getattr(Pendencia, k) == v for k, v in refs.items()]
    for p in db.scalars(select(Pendencia).where(*filtro)).all():
        p.status, p.resolvido_em = "resolvida", agora()


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
                    tamanho=len(conteudo), caminho=salvar_arquivo(sha, conteudo, db),
                    texto=triagem.extrair_texto(conteudo, nome_arquivo), origem=origem)
    db.add(cur)
    db.flush()
    db.add(Tarefa(tipo="TRIAGEM", curriculo_id=cur.id, agente_id=rec.id))
    eventos.registrar(db, "curriculo_recebido", f"{rec.nome} recebeu um novo currículo (#{cur.id}) e o registrou na fila",
                      rec.id, curriculo_id=cur.id)
    return cur, True


def registrar_com_seguranca(conteudo: bytes, nome_arquivo: str, mime: str | None, origem: str, depois=None):
    """registrar_curriculo numa transação própria. Se o mesmo arquivo chegou ao mesmo tempo por outro caminho
    (o banco recusa o segundo pela unicidade do sha256), tenta de novo e devolve o registro que já existe."""
    for tentativa in range(2):
        try:
            with sessao() as db:
                cur, novo = registrar_curriculo(db, conteudo, nome_arquivo, mime, origem)
                extra = depois(db, cur, novo) if depois else None
                return cur.id, novo, extra
        except IntegrityError:
            if tentativa:
                raise
            log.info("O mesmo currículo chegou duas vezes ao mesmo tempo: usando o registro existente")


def executar_triagem(tarefa_id: int):
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        cur = db.get(Curriculo, t.curriculo_id)
        rec = recepcao(db)
        _ocupar(rec, t, f"triando o currículo #{cur.id}")
        _pensar(db, rec, falar(perfil(rec), "pensando"))
        dados = (ler_arquivo(cur.caminho, db), cur.arquivo_nome, cur.mime, cur.texto)
    try:
        resultado = triagem.triar(*dados)  # fora da transação: a IA pode demorar
    except ia.IAIndisponivel:
        _devolver_a_fila(tarefa_id)
        return None
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
            _pensar(db, rec, "Isso não é um currículo… vai para o descarte.")
            eventos.registrar(db, "arquivo_invalido",
                              f"{rec.nome} descartou o arquivo #{cur.id}: não é um currículo ({resultado.get('motivo') or resultado.get('resumo') or 'inválido'})",
                              rec.id, curriculo_id=cur.id)
        else:
            cand = _vincular_candidato(db, resultado)
            cand.dado_sensivel = cand.dado_sensivel or setor in CATEGORIAS_SENSIVEIS
            cur.candidato_id = cand.id
            cur.status = "triado"
            _pensar(db, rec, falar(perfil(rec), "concluir", rotulo=setor), candidato_id=cand.id)
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
        cand = Candidato(email=email)
        try:
            with db.begin_nested():
                db.add(cand)
        except IntegrityError:  # outra triagem em paralelo acabou de registrar o mesmo e-mail
            cand = db.scalar(select(Candidato).where(Candidato.email == email, Candidato.anonimizado_em.is_(None)))
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


def _vagas_abertas_de(db, ag: Agente) -> list[Vaga]:
    return db.scalars(select(Vaga).join(VagaAgente, VagaAgente.vaga_id == Vaga.id)
                      .where(VagaAgente.agente_id == ag.id, Vaga.status == "aberta")).all()


def garantir_responsavel(db, vaga: Vaga) -> Agente | None:
    """Escolhe quem cuida da vaga. Sem especialista na equipe, o Atlas sinaliza — e, com
    CONTRATACAO_AUTOMATICA, contrata na próxima rodada de supervisão."""
    atuais = responsaveis(db, vaga)
    if atuais:
        return min(atuais, key=lambda a: carga(db, a))
    esp = especialidade_para(db, vaga.setores or [])
    candidatos = [a for a in analistas_ativos(db) if esp and a.especialidade_id == esp.id]
    if not candidatos:
        sup = atlas(db)
        auto = bool(config.CONTRATACAO_AUTOMATICA and esp)
        desc = (f"nenhum agente da equipe cobre {esp.nome if esp else 'a área'} para a vaga “{vaga.titulo}”"
                + ("; vou contratar alguém" if auto else ""))
        if _abrir_pendencia(db, "SEM_ESPECIALISTA", desc, sup, vaga_id=vaga.id,
                            especialidade_id=esp.id if esp else None) and auto:
            _pensar(db, sup, f"Ninguém cobre {esp.nome.lower()}… vou contratar.")
        return None
    ag = min(candidatos, key=lambda a: carga(db, a))
    db.add(VagaAgente(vaga_id=vaga.id, agente_id=ag.id))
    db.flush()
    eventos.registrar(db, "vaga_atribuida", f"Atlas atribuiu a vaga “{vaga.titulo}” a {ag.nome}", ag.id, vaga_id=vaga.id)
    return ag


def _encaminhar(db, cur: Curriculo, vaga: Vaga, origem: str = "triagem", indicado_por: Agente | None = None) -> bool:
    """Cria a candidatura (uma por currículo × vaga) e a tarefa de análise."""
    if db.scalar(select(Candidatura).where(Candidatura.curriculo_id == cur.id, Candidatura.vaga_id == vaga.id)):
        return False
    cand = Candidatura(curriculo_id=cur.id, vaga_id=vaga.id, candidato_id=cur.candidato_id, origem=origem,
                       indicado_por_id=indicado_por.id if indicado_por else None)
    db.add(cand)
    db.flush()
    ag = garantir_responsavel(db, vaga)
    nome = primeiro_nome(cur.candidato.nome if cur.candidato else "Candidato")
    if ag:
        db.add(Tarefa(tipo="ANALISE", candidatura_id=cand.id, agente_id=ag.id))
        eventos.registrar(db, "candidatura_encaminhada", f"Atlas encaminhou {nome} para a vaga “{vaga.titulo}” ({ag.nome})",
                          atlas(db).id, vaga_id=vaga.id, candidatura_id=cand.id, destino_id=ag.id, candidato_id=cur.candidato_id)
    else:
        eventos.registrar(db, "candidatura_em_espera", f"{nome} aguarda um agente para a vaga “{vaga.titulo}”",
                          atlas(db).id, vaga_id=vaga.id, candidatura_id=cand.id, candidato_id=cur.candidato_id)
    return True


def executar_distribuicao(tarefa_id: int):
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        cur = db.get(Curriculo, t.curriculo_id)
        sup = atlas(db)
        _ocupar(sup, t, f"distribuindo o currículo #{cur.id}")
        ler = cur.setor == "OUTROS" and config.ENCAMINHAMENTO_AUTOMATICO and bool(cur.texto)
        abertas = [{"id": v.id, "titulo": v.titulo, "area": v.area, "descricao": v.descricao, "requisitos": v.requisitos}
                   for v in db.scalars(select(Vaga).where(Vaga.status == "aberta")).all()] if ler else []
        if abertas:
            _pensar(db, sup, "Esse não caiu em nenhuma pasta… deixa eu ler com calma.")
        texto = cur.texto
    sugestao = None
    if abertas:
        try:
            sugestao = ia.sugerir_vaga(texto, abertas)  # fora da transação: a IA pode demorar
        except ia.IAIndisponivel:
            _devolver_a_fila(tarefa_id)
            return
        except Exception as e:
            log.warning("Atlas não conseguiu sugerir vaga: %s", type(e).__name__)
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        cur = db.get(Curriculo, t.curriculo_id)
        sup = atlas(db)
        nome = primeiro_nome(cur.candidato.nome if cur.candidato else "Candidato")
        if cur.setor == "OUTROS":
            vaga = db.get(Vaga, sugestao["vaga_id"]) if sugestao and sugestao.get("confianca") in ("ALTA", "MEDIA") else None
            if abertas:  # o banco de currículos mostra o que o Atlas achou, sem perguntar de novo à IA
                acervo.guardar_sugestao(cur, sugestao, db.get(Vaga, sugestao["vaga_id"]) if sugestao else None)
            if vaga and vaga.status == "aberta":
                _pensar(db, sup, f"Lembra a vaga “{vaga.titulo}”. Vou encaminhar.", candidato_id=cur.candidato_id)
                eventos.registrar(db, "encaminhamento_automatico",
                                  f"Atlas leu o currículo de {nome} e o encaminhou para “{vaga.titulo}” ({sugestao.get('motivo', '')})",
                                  sup.id, vaga_id=vaga.id, curriculo_id=cur.id, candidato_id=cur.candidato_id)
                _encaminhar(db, cur, vaga, origem="atlas")
            else:
                _abrir_pendencia(db, "TRIAGEM_MANUAL", f"o currículo de {nome} (#{cur.id}) não se encaixou em nenhuma "
                                 "pasta nem em uma vaga aberta; defina a área", sup, curriculo_id=cur.id)
        else:
            vagas = [v for v in db.scalars(select(Vaga).where(Vaga.status == "aberta")).all() if cur.setor in (v.setores or [])]
            novas = sum(_encaminhar(db, cur, v) for v in vagas)
            if not vagas:
                _pensar(db, sup, f"Sem vaga aberta para {cur.setor.lower()}… banco de talentos.", candidato_id=cur.candidato_id)
                eventos.registrar(db, "banco_talentos",
                                  f"Atlas guardou {nome} no banco de talentos: nenhuma vaga aberta para {cur.setor}",
                                  sup.id, curriculo_id=cur.id, candidato_id=cur.candidato_id)
            elif not novas:
                eventos.registrar(db, "sem_duplicidade", f"Atlas viu que {nome} já está nas vagas compatíveis",
                                  sup.id, curriculo_id=cur.id)
        t.status, t.concluida_em = "concluida", agora()
        _liberar(db, sup)


# =========================================================================== Atlas: contratações e dispensas
def _estacoes_livres(db) -> list[int]:
    ocupadas = {a.estacao for a in analistas_ativos(db)}
    return [i for i in range(MAX_ESTACOES) if i not in ocupadas]


def _admitir(db, esp: Especialidade, nome: str | None = None) -> tuple[Agente, bool]:
    """Ocupa uma estação: chama de volta alguém da especialidade que saiu ou cria um novo agente."""
    livres = _estacoes_livres(db)
    if not livres:
        raise ErroNegocio("Todas as estações de trabalho estão ocupadas.")
    inativo = None if nome else db.scalar(select(Agente).where(
        Agente.funcao == "analista", Agente.especialidade_id == esp.id, Agente.ativo.is_(False)).order_by(Agente.id))
    if inativo:
        ag = inativo
        ag.ativo, ag.estacao, ag.ativado_em = True, livres[0], agora()
        ag.ultima_atividade_em, ag.concluido_em, ag.tarefa_atual_id = agora(), None, None
        ag.perfil = ag.perfil or perfil_para(ag.nome, esp.nome)
        db.flush()
        return ag, True
    usados = {a.nome for a in db.scalars(select(Agente)).all()}
    if nome and nome in usados:
        raise ErroNegocio(f"Já existe um agente chamado {nome}.")
    nome = nome or next((n for n in list(esp.nomes_sugeridos or []) + NOMES_EXTRAS if n not in usados),
                        f"Agente {len(usados) + 1}")
    p = perfil_para(nome, esp.nome)
    ag = Agente(nome=nome, funcao="analista", especialidade_id=esp.id, estacao=livres[0],
                aparencia=aparencia_para(nome), ativado_em=agora(), ultima_atividade_em=agora(), perfil=p,
                personalidade=f"Analista de {esp.nome.lower()}; {p['resumo']}.")
    db.add(ag)
    db.flush()
    return ag, False


def _assumir_vagas_sem_responsavel(db, ag: Agente) -> int:
    """O novo agente assume as vagas abertas da sua especialidade que estão sem ninguém."""
    n = 0
    for vaga in db.scalars(select(Vaga).where(Vaga.status == "aberta")).all():
        if responsaveis(db, vaga):
            continue
        esp = especialidade_para(db, vaga.setores or [])
        if not esp or esp.id != ag.especialidade_id:
            continue
        db.add(VagaAgente(vaga_id=vaga.id, agente_id=ag.id))
        db.flush()
        n += 1
        eventos.registrar(db, "vaga_atribuida", f"Atlas atribuiu a vaga “{vaga.titulo}” a {ag.nome}", ag.id, vaga_id=vaga.id)
        _resolver_pendencias(db, "SEM_ESPECIALISTA", vaga_id=vaga.id)
        _enfileirar_espera(db, vaga, ag)
    return n


def _enfileirar_espera(db, vaga: Vaga, ag: Agente):
    """Candidaturas que esperavam alguém para a vaga entram na fila do agente."""
    for c in db.scalars(select(Candidatura).where(Candidatura.vaga_id == vaga.id,
                                                  Candidatura.status.in_(("na_fila", "em_analise")))).all():
        if not db.scalar(select(Tarefa).where(Tarefa.candidatura_id == c.id, Tarefa.status.in_(("pendente", "executando")))):
            db.add(Tarefa(tipo="ANALISE", candidatura_id=c.id, agente_id=ag.id))


def _reforcar(db, novo: Agente) -> int:
    """Contratação por sobrecarga: o novo agente divide as vagas e parte da fila dos colegas."""
    colegas = [a for a in analistas_ativos(db) if a.especialidade_id == novo.especialidade_id and a.id != novo.id]
    for vaga in {v.id: v for a in colegas for v in _vagas_abertas_de(db, a)}.values():
        if not db.get(VagaAgente, (vaga.id, novo.id)):
            db.add(VagaAgente(vaga_id=vaga.id, agente_id=novo.id))
    db.flush()
    total = sum(carga(db, a) for a in colegas)
    alvo = total // (len(colegas) + 1)
    movidas = 0
    while movidas < alvo:
        mais = max(colegas, key=lambda a: carga(db, a), default=None)
        t = mais and db.scalar(select(Tarefa).where(Tarefa.agente_id == mais.id, Tarefa.tipo == "ANALISE",
                                                     Tarefa.status == "pendente", Tarefa.tentativas == 0)
                               .order_by(Tarefa.id.desc()))
        if not t:
            break
        t.agente_id = novo.id
        db.flush()
        movidas += 1
    if movidas:
        sup = atlas(db)
        eventos.registrar(db, "fila_redistribuida", f"Atlas passou {movidas} currículo(s) da fila para {novo.nome}", sup.id,
                          destino_id=novo.id)
        _dizer(db, sup, f"{novo.nome}, separei {movidas} currículo(s) da fila para você começar.", para=novo)
    return movidas


MOTIVOS = {
    "sem_especialista": "faltava alguém de {esp} para uma vaga aberta",
    "sobrecarga": "a fila de {esp} passou de {limite} currículos por pessoa",
    "consulta": "o time pediu ajuda de {esp} mais de uma vez",
    "manual": "pedido do recrutador",
}


def contratar(db, esp: Especialidade, motivo: str = "manual", por: str = "Atlas", nome: str | None = None) -> Agente:
    """Atlas (ou o recrutador) traz um agente para o time. Libera a estação de alguém ocioso se precisar."""
    if not _estacoes_livres(db) and por == "Atlas":
        _dispensar_ocioso(db, f"liberar uma estação para {esp.nome.lower()}")
    ag, voltou = _admitir(db, esp, nome)
    sup = atlas(db)
    txt_motivo = MOTIVOS.get(motivo, motivo).format(esp=esp.nome.lower(), limite=config.LIMITE_FILA_POR_AGENTE)
    desc = f"{por} {'chamou de volta' if voltou else 'contratou'} {ag.nome} ({esp.nome}): {txt_motivo}"
    db.add(Contratacao(especialidade_id=esp.id, agente_id=ag.id, acao="retorno" if voltou else "contratacao",
                       motivo=motivo, descricao=desc, por=por))
    eventos.registrar(db, "contratacao", desc, ag.id, especialidade=esp.codigo, motivo=motivo, por=por, retorno=voltou)
    p = perfil(ag)
    if voltou:
        _dizer(db, sup, f"Que bom ter você de volta, {ag.nome}! {txt_motivo.capitalize()}.", para=ag)
    else:
        _dizer(db, sup, f"Boas-vindas ao time, {ag.nome}! Você cuida de {esp.nome.lower()}. {bordao(perfil(sup))}", para=ag)
    _dizer(db, ag, f"Oi, time! {ag.nome} aqui — {p.get('apresentacao', p.get('resumo', ''))}. {bordao(p)}", para=sup)
    _assumir_vagas_sem_responsavel(db, ag)
    if motivo == "sobrecarga":
        _reforcar(db, ag)
    _resolver_pendencias(db, "PROPOSTA_CONTRATACAO", especialidade_id=esp.id)
    _resolver_pendencias(db, "ESTACOES_LOTADAS", especialidade_id=esp.id)
    return ag


def _dispensar(db, ag: Agente, por: str, motivo: str):
    ag.ativo, ag.estacao, ag.tarefa_atual_id, ag.atividade = False, None, None, ""
    db.execute(delete(VagaAgente).where(VagaAgente.agente_id == ag.id))
    desc = f"{por} liberou {ag.nome} da equipe: {motivo}"
    db.add(Contratacao(especialidade_id=ag.especialidade_id, agente_id=ag.id, acao="dispensa", motivo="estacao", descricao=desc, por=por))
    eventos.registrar(db, "agente_dispensado", desc, ag.id, por=por)
    _dizer(db, ag, "Combinado! Se precisarem, é só chamar.", para=atlas(db))


def _dispensar_ocioso(db, motivo: str) -> Agente | None:
    """Libera a estação de quem está sem vagas e sem fila (quem está há mais tempo parado sai primeiro)."""
    ociosos = [a for a in analistas_ativos(db) if not a.tarefa_atual_id and not carga(db, a)
               and not db.scalar(select(func.count()).select_from(VagaAgente).where(VagaAgente.agente_id == a.id))]
    if not ociosos:
        return None
    ag = min(ociosos, key=lambda a: a.ultima_atividade_em or a.ativado_em or agora())
    _pensar(db, atlas(db), f"{ag.nome} está sem vagas… vou liberar a estação.")
    _dispensar(db, ag, "Atlas", f"sem vagas no momento ({motivo})")
    return ag


def dispensar_agente(db, ag: Agente, por: str):
    """Dispensa pedida pelo recrutador. As vagas e a fila passam para colegas da mesma especialidade."""
    if ag.funcao != "analista":
        raise ErroNegocio("Só analistas podem ser dispensados.")
    if not ag.ativo:
        raise ErroNegocio(f"{ag.nome} já está fora da equipe.")
    if ag.tarefa_atual_id or db.scalar(select(Tarefa).where(Tarefa.agente_id == ag.id, Tarefa.tipo.in_(("CONSULTA", "REVISAO")),
                                                             Tarefa.status.in_(("pendente", "executando")))):
        raise ErroNegocio(f"{ag.nome} está no meio de uma análise; tente de novo em instantes.")
    colegas = [a for a in analistas_ativos(db) if a.especialidade_id == ag.especialidade_id and a.id != ag.id]
    vagas = _vagas_abertas_de(db, ag)
    if vagas and not colegas:
        raise ErroNegocio(f"{ag.nome} é quem cuida de “{vagas[0].titulo}”. Contrate outro agente da área ou "
                          "suspenda a vaga antes de dispensar.")
    fila = db.scalars(select(Tarefa).where(Tarefa.agente_id == ag.id, Tarefa.status == "pendente")).all()
    _dispensar(db, ag, por, "decisão do recrutador")
    for t in fila:
        c = db.get(Candidatura, t.candidatura_id) if t.candidatura_id else None
        novo = garantir_responsavel(db, c.vaga) if c else None
        if novo:
            t.agente_id = novo.id
        else:
            t.status = "cancelada"


# =========================================================================== análises e consultas entre colegas
def _esp_dicts(db):
    return [{"codigo": e.codigo, "nome": e.nome, "setores": e.setores} for e in db.scalars(select(Especialidade)).all()]


def _agente_dict(db, ag: Agente) -> dict:
    return {"nome": ag.nome, "personalidade": ag.personalidade, "perfil": publico(perfil(ag)),
            "especialidade": ag.especialidade.nome if ag.especialidade else None,
            "especialidade_codigo": ag.especialidade.codigo if ag.especialidade else None,
            "setores": ag.especialidade.setores if ag.especialidade else [],
            "vagas": [{"titulo": v.titulo, "requisitos": v.requisitos} for v in _vagas_abertas_de(db, ag)]}


def _vaga_dict(v: Vaga) -> dict:
    return {"titulo": v.titulo, "area": v.area, "local": v.local, "descricao": v.descricao, "requisitos": v.requisitos}


def _leitura(cur: Curriculo):
    """(texto, arquivo) — o arquivo só é lido quando não há texto extraído (PDF escaneado, imagem)."""
    return cur.texto, (None if cur.texto else (ler_arquivo(cur.caminho), cur.mime))


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
        nome = rotulo(c)
        _ocupar(ag, t, f"analisando {nome} para “{c.vaga.titulo}”")
        if c.status == "na_fila":
            c.status = "em_analise"
        agente, vaga = _agente_dict(db, ag), _vaga_dict(c.vaga)
        colegas = [_agente_dict(db, a) for a in analistas_ativos(db) if a.id != ag.id]
        indicador = db.get(Agente, c.indicado_por_id) if c.indicado_por_id else None
        evitar = {indicador.especialidade.codigo} if indicador and indicador.especialidade else set()
        texto, arquivo = _leitura(c.curriculo)
        setor, esps, ag_id, cand_id = c.curriculo.setor, _esp_dicts(db), ag.id, c.candidato_id
        inicio = time.time()
    try:
        resultado, modelo = ia.analisar(agente, vaga, texto, arquivo, esps, setor, colegas, evitar, nome,
                                        lambda p: pensar(ag_id, p, candidato_id=cand_id))
        falha = None
    except ia.IAIndisponivel:
        _devolver_a_fila(tarefa_id)
        return
    except Exception as e:
        resultado, modelo, falha = None, "", e
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        c = db.get(Candidatura, t.candidatura_id)
        ag = db.get(Agente, t.agente_id)
        if falha:
            t.erro = f"{type(falha).__name__}: {str(falha)[:200]}"
            if t.tentativas < 3:
                t.status = "pendente"
            else:
                t.status = "erro"
                c.status = "erro"
                _abrir_pendencia(db, "ERRO_IA", f"não consegui analisar o currículo de {rotulo(c)} para “{c.vaga.titulo}”",
                                 ag, candidatura_id=c.id)
            _liberar(db, ag)
            return
        codigo = resultado.get("ajuda_especialidade") or ""
        if config.COLABORACAO and resultado.get("pedir_ajuda") and codigo not in evitar:
            if _pedir_ajuda(db, c, ag, resultado, modelo, inicio):
                t.status, t.concluida_em = "concluida", agora()
                _liberar(db, ag)
                return
        _finalizar_analise(db, c, ag, resultado, modelo, inicio)
        t.status, t.concluida_em = "concluida", agora()
        _liberar(db, ag)


def _finalizar_analise(db, c: Candidatura, ag: Agente, resultado: dict, modelo: str, inicio: float,
                       consulta: dict | None = None):
    resultado.pop("pedir_ajuda", None)
    if consulta:
        resultado["colaboracoes"] = [consulta]
    db.add(Analise(candidatura_id=c.id, agente_id=ag.id, tipo="principal",
                   compatibilidade=resultado["compatibilidade"], recomendado=bool(resultado.get("recomendado")),
                   avaliacao_humana=bool(resultado.get("avaliacao_humana")), resultado=resultado, modelo=modelo,
                   duracao_ms=int((time.time() - inicio) * 1000)))
    c.status = "analisada"
    if c.candidato:
        c.candidato.ultima_atividade_em = agora()
    rot = ROTULO_COMPAT[resultado["compatibilidade"]]
    extra = " · recomendado" if resultado.get("recomendado") else (" · avaliação humana" if resultado.get("avaliacao_humana") else "")
    ajuda = f" (com a ajuda de {consulta['colega']})" if consulta and consulta.get("resposta_ok") else ""
    _pensar(db, ag, falar(perfil(ag), "concluir", rotulo=rot), candidato_id=c.candidato_id)
    eventos.registrar(db, "analise_concluida",
                      f"{ag.nome} concluiu a análise de {rotulo(c)} para “{c.vaga.titulo}”: compatibilidade {rot}{extra}{ajuda}",
                      ag.id, candidatura_id=c.id, vaga_id=c.vaga_id, candidato_id=c.candidato_id)


def _pedir_ajuda(db, c: Candidatura, ag: Agente, resultado: dict, modelo: str, inicio: float) -> bool:
    """O analista leva a dúvida a um colega de outra especialidade. Retorna False se não houver quem ajude."""
    esp = db.scalar(select(Especialidade).where(Especialidade.codigo == resultado.get("ajuda_especialidade")))
    if not esp or esp.id == ag.especialidade_id:
        return False
    pergunta = maiuscula((resultado.get("ajuda_pergunta") or "pode dar uma olhada neste currículo?")[:240])
    nome = rotulo(c)
    colegas = [a for a in analistas_ativos(db) if a.especialidade_id == esp.id and a.id != ag.id]
    if not colegas:
        sup = atlas(db)
        _dizer(db, ag, falar(perfil(ag), "pedir_ajuda", colega="Atlas", pergunta=f"Temos alguém de {esp.nome.lower()}? {pergunta}"),
               para=sup, candidato_id=c.candidato_id)
        _dizer(db, sup, f"{ag.nome}, ainda não temos ninguém de {esp.nome.lower()}. Anotei a necessidade; siga com a sua "
                        "leitura e deixe o ponto para o recrutador verificar.", para=ag)
        eventos.registrar(db, "consulta_sem_especialista", f"Atlas anotou: {ag.nome} precisou de alguém de {esp.nome} ({nome})",
                          sup.id, especialidade_id=esp.id, especialidade=esp.codigo, candidatura_id=c.id)
        resultado.setdefault("a_verificar", []).append(f"Ponto para alguém de {esp.nome.lower()}: {pergunta}")
        return False
    colega = min(colegas, key=lambda a: carga(db, a))
    db.add(Tarefa(tipo="CONSULTA", candidatura_id=c.id, agente_id=colega.id, motivo=pergunta[:200],
                  dados={"solicitante_id": ag.id, "pergunta": pergunta, "preliminar": resultado, "modelo": modelo,
                         "inicio": inicio}))
    eventos.registrar(db, "consulta_pedida", f"{ag.nome} pediu ajuda a {colega.nome} com o currículo de {nome} (“{c.vaga.titulo}”)",
                      ag.id, destino_id=colega.id, candidatura_id=c.id, candidato_id=c.candidato_id)
    _dizer(db, ag, falar(perfil(ag), "pedir_ajuda", colega=colega.nome, pergunta=pergunta), para=colega, candidato_id=c.candidato_id)
    return True


def executar_consulta(tarefa_id: int):
    """O colega lê o currículo e responde. Depois o solicitante revisa a análise (tarefa REVISAO)."""
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        c = db.get(Candidatura, t.candidatura_id)
        colega = db.get(Agente, t.agente_id)
        d = t.dados or {}
        sol = db.get(Agente, d.get("solicitante_id"))
        if c.vaga.status != "aberta" or not sol:
            t.status = "cancelada"
            return
        nome = rotulo(c)
        _ocupar(colega, t, f"ajudando {sol.nome} com o currículo de {nome}")
        args = (_agente_dict(db, colega), _agente_dict(db, sol), _vaga_dict(c.vaga), *_leitura(c.curriculo),
                d.get("pergunta", ""), d.get("preliminar", {}), nome)
        colega_id, cand_id = colega.id, c.candidato_id
    try:
        resp, _ = ia.consultar(*args, ao_pensar=lambda p: pensar(colega_id, p, candidato_id=cand_id))
        falha = None
    except ia.IAIndisponivel:
        _devolver_a_fila(tarefa_id)
        return
    except Exception as e:
        resp, falha = None, e
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        c = db.get(Candidatura, t.candidatura_id)
        colega = db.get(Agente, t.agente_id)
        d = dict(t.dados or {})
        sol = db.get(Agente, d.get("solicitante_id"))
        nome = rotulo(c)
        if falha and t.tentativas < 3:
            t.status, t.erro = "pendente", f"{type(falha).__name__}: {str(falha)[:200]}"
            _liberar(db, colega)
            return
        consulta = {"colega_id": colega.id, "colega": colega.nome,
                    "especialidade": colega.especialidade.nome if colega.especialidade else "",
                    "pergunta": d.get("pergunta", ""), "resposta_ok": bool(resp),
                    "resposta": (resp or {}).get("resposta") or "não consegui ler o currículo agora.",
                    "evidencias": (resp or {}).get("evidencias", [])[:3], "parecer": (resp or {}).get("parecer", "neutro"),
                    "reforca_requisito": (resp or {}).get("reforca_requisito", "")}
        if resp:
            eventos.registrar(db, "consulta_respondida", f"{colega.nome} ajudou {sol.nome} com o currículo de {nome}",
                              colega.id, destino_id=sol.id, candidatura_id=c.id, candidato_id=c.candidato_id)
            _dizer(db, colega, falar(perfil(colega), "ajudar", colega=sol.nome, resposta=maiuscula(consulta["resposta"])),
                   para=sol, candidato_id=c.candidato_id)
            _indicar(db, c, colega, sol, resp.get("indicar_vaga") or "")
        else:
            t.erro = f"{type(falha).__name__}: {str(falha)[:200]}"
            _dizer(db, colega, f"{sol.nome}, não consegui ler esse currículo agora. Siga com a sua análise.", para=sol)
        d["consulta"] = consulta
        db.add(Tarefa(tipo="REVISAO", candidatura_id=c.id, agente_id=sol.id, dados=d))
        t.status, t.concluida_em = "concluida", agora()
        _liberar(db, colega)


def _indicar(db, c: Candidatura, colega: Agente, sol: Agente, titulo: str):
    """O colega inclui o candidato numa vaga dele, se a conversa mostrou que combina."""
    alvo = (titulo or "").strip().lower()
    if not alvo or not c.curriculo.candidato_id:
        return
    vaga = next((v for v in _vagas_abertas_de(db, colega) if v.titulo.strip().lower() == alvo), None)
    if not vaga or vaga.id == c.vaga_id:
        return
    if _encaminhar(db, c.curriculo, vaga, origem="indicacao", indicado_por=sol):
        eventos.registrar(db, "indicacao", f"{colega.nome} incluiu {rotulo(c)} também na vaga “{vaga.titulo}”, "
                                           f"a partir da conversa com {sol.nome}",
                          colega.id, vaga_id=vaga.id, destino_id=sol.id, candidato_id=c.candidato_id)


def executar_revisao(tarefa_id: int):
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        c = db.get(Candidatura, t.candidatura_id)
        ag = db.get(Agente, t.agente_id)
        d = t.dados or {}
        consulta = d.get("consulta") or {}
        if c.vaga.status != "aberta":
            t.status = "cancelada"
            return
        if db.scalar(select(Analise).where(Analise.candidatura_id == c.id, Analise.agente_id == ag.id)):
            t.status, t.concluida_em = "concluida", agora()
            return
        nome = rotulo(c)
        _ocupar(ag, t, f"fechando a análise de {nome} com a ajuda de {consulta.get('colega', 'um colega')}")
        args = (_agente_dict(db, ag), _vaga_dict(c.vaga), *_leitura(c.curriculo), d.get("preliminar", {}), consulta, nome)
        ag_id, cand_id = ag.id, c.candidato_id
    try:
        resultado, modelo = ia.revisar(*args, ao_pensar=lambda p: pensar(ag_id, p, candidato_id=cand_id))
        falha = None
    except ia.IAIndisponivel:
        _devolver_a_fila(tarefa_id)
        return
    except Exception as e:
        resultado, modelo, falha = None, "", e
    with sessao() as db:
        t = db.get(Tarefa, tarefa_id)
        c = db.get(Candidatura, t.candidatura_id)
        ag = db.get(Agente, t.agente_id)
        d = t.dados or {}
        if falha:
            if t.tentativas < 3:
                t.status, t.erro = "pendente", f"{type(falha).__name__}: {str(falha)[:200]}"
                _liberar(db, ag)
                return
            resultado, modelo = dict(d.get("preliminar") or {}), d.get("modelo", "")  # fica a leitura preliminar
        consulta = d.get("consulta") or {}
        _finalizar_analise(db, c, ag, resultado, modelo, d.get("inicio") or time.time(), consulta)
        colega = db.get(Agente, consulta.get("colega_id")) if consulta.get("colega_id") else None
        if colega and consulta.get("resposta_ok"):
            _dizer(db, ag, falar(perfil(ag), "agradecer", colega=colega.nome), para=colega, candidato_id=c.candidato_id)
        t.status, t.concluida_em = "concluida", agora()
        _liberar(db, ag)


# =========================================================================== fila (usada pelo trabalhador)
def reservar_tarefas(ocupadas: set[str]) -> list[tuple[str, int, str]]:
    """Reserva no máximo uma tarefa por 'raia' livre (recepção, Atlas e cada analista)."""
    saida = []
    pausa = uso_ia.bloqueio()
    with sessao() as db:
        # FOR UPDATE SKIP LOCKED: mesmo com duas instâncias no ar, a mesma tarefa nunca é pega duas vezes
        pend = db.scalars(select(Tarefa).where(Tarefa.status == "pendente").order_by(Tarefa.id)
                          .with_for_update(skip_locked=True)).all()
        for t in pend:
            raia = {"TRIAGEM": "recepcao", "DISTRIBUICAO": "atlas"}.get(t.tipo, f"ag{t.agente_id}")
            if raia in ocupadas or any(r == raia for r, _, _ in saida):
                continue
            if pausa and (t.tipo in TAREFAS_COM_IA or (t.tipo == "DISTRIBUICAO" and _distribuicao_usa_ia(db, t))):
                continue  # IA pausada: o trabalho espera na fila (o Atlas já avisou o recrutador)
            if t.tentativas and t.iniciada_em and agora() - t.iniciada_em < timedelta(seconds=10 * t.tentativas):
                continue  # espera antes de tentar de novo
            t.status, t.iniciada_em = "executando", agora()
            t.tentativas += 1
            saida.append((raia, t.id, t.tipo))
    return saida


def _distribuicao_usa_ia(db, t: Tarefa) -> bool:
    cur = db.get(Curriculo, t.curriculo_id) if t.curriculo_id else None
    return bool(cur and cur.setor == "OUTROS" and cur.texto and config.ENCAMINHAMENTO_AUTOMATICO)


EXECUTORES = {"TRIAGEM": executar_triagem, "DISTRIBUICAO": executar_distribuicao, "ANALISE": executar_analise,
              "CONSULTA": executar_consulta, "REVISAO": executar_revisao}


def executar(tarefa_id: int, tipo: str):
    try:
        EXECUTORES[tipo](tarefa_id)
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


# =========================================================================== supervisão automática do Atlas
def supervisionar():
    """Rodada do Atlas: cobre vagas sem responsável, reforça o time quando a fila cresce e
    contrata a área que os colegas pediram. Roda em segundo plano a cada SUPERVISAO_SEGUNDOS."""
    with trava_equipe, sessao() as db:
        _verificar_ia(db)
        _cobrir_vagas(db)
        _verificar_carga(db)
        _verificar_demanda(db)


def _verificar_ia(db):
    """IA pausada vira uma pendência única para o recrutador, que some sozinha quando a IA volta a responder."""
    if config.IA_MODO != "gemini":
        return
    motivo = uso_ia.bloqueio() or uso_ia.pausa_do_provedor()
    aberta = db.scalar(select(Pendencia).where(Pendencia.tipo == "IA_INDISPONIVEL", Pendencia.status == "aberta"))
    sup = atlas(db)
    if motivo and not aberta:
        _abrir_pendencia(db, "IA_INDISPONIVEL", f"IA pausada: {motivo}", sup)
        _pensar(db, sup, "A IA parou de responder… avisei o recrutador e segurei a fila.")
    elif motivo and aberta and aberta.descricao != f"IA pausada: {motivo}":
        aberta.descricao = f"IA pausada: {motivo}"
    elif aberta and not motivo:
        aberta.status, aberta.resolvido_em = "resolvida", agora()
        eventos.registrar(db, "pendencia_resolvida", "A IA voltou a responder e o trabalho da fila seguiu sozinho",
                          sup.id, pendencia_id=aberta.id)


def _ultima_contratacao(db, esp_id):
    return db.scalar(select(func.max(Contratacao.criado_em)).where(
        Contratacao.especialidade_id == esp_id, Contratacao.acao.in_(("contratacao", "retorno", "sem_estacao"))))


def _pode_contratar(db, esp: Especialidade) -> bool:
    ultima = _ultima_contratacao(db, esp.id)
    return not ultima or agora() - ultima >= timedelta(seconds=config.INTERVALO_CONTRATACAO)


def _tentar_contratar(db, esp: Especialidade, motivo: str, descricao_proposta: str, pensamento: str = "") -> Agente | None:
    """Com CONTRATACAO_AUTOMATICA o Atlas contrata; sem ela, propõe ao recrutador (uma vez)."""
    sup = atlas(db)
    if not config.CONTRATACAO_AUTOMATICA:
        if not db.scalar(select(Pendencia).where(Pendencia.tipo == "PROPOSTA_CONTRATACAO", Pendencia.status == "aberta",
                                                 Pendencia.especialidade_id == esp.id)):
            _pensar(db, sup, pensamento)
            _abrir_pendencia(db, "PROPOSTA_CONTRATACAO", descricao_proposta, sup, especialidade_id=esp.id)
        return None
    _pensar(db, sup, pensamento)
    try:
        return contratar(db, esp, motivo)
    except ErroNegocio:
        # registra a tentativa para respeitar o intervalo e pede ajuda ao recrutador uma vez
        db.add(Contratacao(especialidade_id=esp.id, acao="sem_estacao", motivo=motivo, por="Atlas",
                           descricao=f"Atlas queria contratar alguém de {esp.nome}, mas todas as estações estão ocupadas"))
        _abrir_pendencia(db, "ESTACOES_LOTADAS", f"preciso de alguém de {esp.nome.lower()}, mas as {MAX_ESTACOES} estações "
                                                 "estão ocupadas; dispense um agente ou encerre vagas", sup, especialidade_id=esp.id)
        return None


def _cobrir_vagas(db):
    for vaga in db.scalars(select(Vaga).where(Vaga.status == "aberta")).all():
        if responsaveis(db, vaga):
            continue
        esp = especialidade_para(db, vaga.setores or [])
        if not esp:
            continue
        ag = next((a for a in analistas_ativos(db) if a.especialidade_id == esp.id), None)
        if ag:  # alguém da área já está no time (ex.: depois de uma dispensa): assume
            garantir_responsavel(db, vaga)
            _resolver_pendencias(db, "SEM_ESPECIALISTA", vaga_id=vaga.id)
            _enfileirar_espera(db, vaga, min(responsaveis(db, vaga), key=lambda a: carga(db, a)))
        elif config.CONTRATACAO_AUTOMATICA and _pode_contratar(db, esp):
            # sem contratação automática, a pendência SEM_ESPECIALISTA já é a proposta ao recrutador
            _tentar_contratar(db, esp, "sem_especialista", "")


def _verificar_carga(db):
    por_esp: dict[int, list[Agente]] = {}
    for a in analistas_ativos(db):
        if a.especialidade_id:
            por_esp.setdefault(a.especialidade_id, []).append(a)
    for esp_id, ags in por_esp.items():
        fila = sum(carga(db, a) for a in ags)
        if fila / len(ags) <= config.LIMITE_FILA_POR_AGENTE or len(ags) >= config.MAX_AGENTES_POR_ESPECIALIDADE:
            continue
        esp = db.get(Especialidade, esp_id)
        if not _pode_contratar(db, esp):
            continue
        _tentar_contratar(db, esp, "sobrecarga",
                          f"a fila de {esp.nome.lower()} está com {fila} currículo(s) para {len(ags)} agente(s); "
                          "proponho contratar mais um",
                          f"A fila de {esp.nome.lower()} está com {fila} currículos para {len(ags)}… hora de reforçar.")


def _verificar_demanda(db):
    limite = agora() - timedelta(days=7)
    evs = [e for e in db.scalars(select(Evento).where(Evento.tipo == "consulta_sem_especialista", Evento.ts >= limite)).all()
           if not (e.dados or {}).get("atendida")]
    for esp_id, n in Counter((e.dados or {}).get("especialidade_id") for e in evs).items():
        esp = db.get(Especialidade, esp_id) if esp_id else None
        if not esp or n < config.DEMANDA_PARA_CONTRATAR:
            continue
        atendida = any(a.especialidade_id == esp.id for a in analistas_ativos(db))  # alguém da área já chegou
        if not atendida:
            if not _pode_contratar(db, esp):
                continue
            novo = _tentar_contratar(db, esp, "consulta",
                                     f"o time pediu ajuda de {esp.nome.lower()} {n} vez(es); proponho contratar alguém da área",
                                     f"Já pediram ajuda de {esp.nome.lower()} {n} vezes. Vou trazer alguém.")
            # contratou, ou deixou a proposta para o recrutador; se faltou estação, tenta de novo depois
            atendida = bool(novo) or not config.CONTRATACAO_AUTOMATICA
        if atendida:
            for e in evs:
                if (e.dados or {}).get("especialidade_id") == esp_id:
                    e.dados = {**e.dados, "atendida": True}


def reprocessar_automatico() -> int:
    """Falhas de IA: o Atlas tenta de novo sozinho (até 2 vezes por item) antes de depender do recrutador."""
    limite = agora() - timedelta(minutes=config.REPROCESSAR_APOS_MIN)
    n = 0
    with sessao() as db:
        for p in db.scalars(select(Pendencia).where(Pendencia.tipo == "ERRO_IA", Pendencia.status == "aberta",
                                                    Pendencia.criado_em <= limite)).all():
            ref = (Pendencia.curriculo_id == p.curriculo_id) if p.curriculo_id else (Pendencia.candidatura_id == p.candidatura_id)
            feitas = db.scalar(select(func.count()).select_from(Pendencia).where(
                Pendencia.tipo == "ERRO_IA", Pendencia.status == "resolvida", Pendencia.resolvido_por_id.is_(None), ref)) or 0
            if feitas >= 2:
                continue
            _reprocessar(db, p)
            p.status, p.resolvido_em = "resolvida", agora()
            eventos.registrar(db, "reprocessamento", f"Atlas tentou de novo, sozinho: {p.descricao}", atlas(db).id, pendencia_id=p.id)
            n += 1
    return n


def _reprocessar(db, p: Pendencia):
    t = db.scalar(select(Tarefa).where(or_(Tarefa.curriculo_id == p.curriculo_id if p.curriculo_id else False,
                                           Tarefa.candidatura_id == p.candidatura_id if p.candidatura_id else False),
                                       Tarefa.status == "erro").order_by(Tarefa.id.desc()))
    if t:
        t.status, t.tentativas, t.erro = "pendente", 0, None
        if t.candidatura_id:
            c = db.get(Candidatura, t.candidatura_id)
            if c and c.status == "erro":
                c.status = "em_analise"
    elif p.curriculo_id:
        cur = db.get(Curriculo, p.curriculo_id)
        if cur and cur.status == "erro":
            cur.status = "recebido"
        db.add(Tarefa(tipo="TRIAGEM", curriculo_id=p.curriculo_id, agente_id=recepcao(db).id))


def boletim() -> bool:
    """Boletim periódico do Atlas para o time, só quando houve movimento desde o último."""
    if config.BOLETIM_MINUTOS <= 0:
        return False
    with sessao() as db:
        reg = db.get(Configuracao, "ultimo_boletim")
        desde = None
        if reg:
            try:
                desde = datetime.fromisoformat(reg.valor)
            except ValueError:
                desde = None
        if desde and agora() - desde < timedelta(minutes=config.BOLETIM_MINUTOS):
            return False
        inicio = desde or agora() - timedelta(minutes=config.BOLETIM_MINUTOS)
        novos = db.scalar(select(func.count()).select_from(Curriculo).where(Curriculo.recebido_em >= inicio)) or 0
        analises = db.scalar(select(func.count()).select_from(Analise).where(Analise.criado_em >= inicio)) or 0
        recomendados = db.scalar(select(func.count()).select_from(Analise).where(
            Analise.criado_em >= inicio, Analise.recomendado.is_(True))) or 0
        if reg:
            reg.valor = agora().isoformat()
        else:
            db.add(Configuracao(chave="ultimo_boletim", valor=agora().isoformat()))
        if not (novos or analises):
            return False
        q = quadro(db)
        sup = atlas(db)
        _dizer(db, sup, f"Boletim: {novos} currículo(s) novo(s), {analises} análise(s) ({recomendados} recomendado(s)). "
                        f"Na fila: {q['tarefas_pendentes']}. Pendências: {q['pendencias']}. {bordao(perfil(sup))}",
               boletim=True)
        return True


def limpar_conversas() -> int:
    """Pensamentos e conversas têm retenção curta (podem citar o primeiro nome de candidatos)."""
    limite = agora() - timedelta(hours=config.RETENCAO_CONVERSAS_HORAS)
    with sessao() as db:
        r = db.execute(delete(Evento).where(Evento.tipo.in_(FALAS), Evento.ts < limite))
        return r.rowcount or 0


# =========================================================================== estados e leituras
class Lote:
    """Contagens da equipe e das vagas lidas de uma vez, em poucas consultas agrupadas, em vez de várias por
    agente e por vaga: com o banco longe (Supabase), cada consulta é uma ida e volta pela internet."""

    def __init__(self, db):
        db.scalars(select(Especialidade)).all()  # no mapa de identidade: ag.especialidade não consulta de novo
        self.ativos = db.scalars(select(Agente).where(Agente.ativo.is_(True)).order_by(Agente.id)).all()
        self.carga = dict(db.execute(select(Tarefa.agente_id, func.count()).where(
            Tarefa.status.in_(("pendente", "executando")), Tarefa.agente_id.is_not(None)).group_by(Tarefa.agente_id)).all())
        self.pendencias, self.pendencia = Counter(), {}
        for ag_id, desc in db.execute(select(Pendencia.agente_id, Pendencia.descricao).where(
                Pendencia.status == "aberta").order_by(Pendencia.id)).all():
            self.pendencias[ag_id] += 1
            self.pendencia.setdefault(ag_id, desc)
        self.total_pendencias = sum(self.pendencias.values())
        self.vagas_de, self.responsaveis = defaultdict(list), defaultdict(list)
        por_id = {a.id: a for a in self.ativos}
        for ag_id, vaga_id in db.execute(select(VagaAgente.agente_id, VagaAgente.vaga_id).order_by(VagaAgente.vaga_id, VagaAgente.agente_id)).all():
            self.vagas_de[ag_id].append(vaga_id)
            if ag_id in por_id:
                self.responsaveis[vaga_id].append(por_id[ag_id])
        self.contagem = {vid: (rec, int(ana or 0)) for vid, rec, ana in db.execute(
            select(Candidatura.vaga_id, func.count(), func.sum(case((Candidatura.status == "analisada", 1), else_=0)))
            .group_by(Candidatura.vaga_id)).all()}


def estado_de(db, ag: Agente, lote: Lote | None = None) -> tuple[str, str]:
    if not ag.ativo:
        return "inativo", ""
    if ag.tarefa_atual_id:
        return "trabalhando", ag.atividade
    if lote:
        desc = lote.pendencia.get(ag.id)
    else:
        p = db.scalar(select(Pendencia).where(Pendencia.agente_id == ag.id, Pendencia.status == "aberta"))
        desc = p.descricao if p else None
    if desc is not None:
        return "pendencia", desc
    if (lote.carga.get(ag.id, 0) if lote else carga(db, ag)):
        if uso_ia.bloqueio() or (config.IA_MODO == "gemini" and uso_ia.pausa_do_provedor()):
            return "aguardando", "esperando a IA voltar para seguir com a fila"  # a fila não anda sem a IA
        return "trabalhando", "pegando o próximo currículo"
    t = agora()
    if ag.concluido_em and t - ag.concluido_em < timedelta(seconds=config.SEGUNDOS_CONCLUIDO):
        return "concluido", {"recepcao": "terminou a triagem da fila", "supervisor": "distribuiu todos os currículos"}.get(
            ag.funcao, "terminou as análises da fila")
    tem_vagas = (len(lote.vagas_de.get(ag.id, [])) if lote else
                 db.scalar(select(func.count()).select_from(VagaAgente).where(VagaAgente.agente_id == ag.id)))
    if ag.funcao == "analista":
        ocioso = t - (ag.ultima_atividade_em or ag.ativado_em or t)
        if ocioso >= timedelta(seconds=config.SEGUNDOS_PARA_DESCANSO):
            return "descansando", ""
        if not tem_vagas:
            return "disponivel", "disponível para novas vagas"
        return "aguardando", "aguardando novos currículos"
    return "aguardando", "aguardando novos currículos" if ag.funcao == "recepcao" else "acompanhando a equipe"


def agente_json(db, ag: Agente, lote: Lote | None = None) -> dict:
    est, ativ = estado_de(db, ag, lote)
    vagas = lote.vagas_de.get(ag.id, []) if lote else db.scalars(select(VagaAgente.vaga_id).where(VagaAgente.agente_id == ag.id)).all()
    return {"id": ag.id, "nome": ag.nome, "funcao": ag.funcao,
            "funcao_rotulo": {"supervisor": "Supervisor da equipe", "recepcao": "Recepção de currículos",
                              "analista": "Analista de currículos"}[ag.funcao],
            "especialidade": ag.especialidade.nome if ag.especialidade else
            ("Supervisão e coordenação" if ag.funcao == "supervisor" else "Recebimento e triagem de currículos"),
            "especialidade_codigo": ag.especialidade.codigo if ag.especialidade else None,
            "estado": est, "atividade": ativ, "estacao": ag.estacao, "aparencia": ag.aparencia,
            "perfil": publico(perfil(ag)), "carga": lote.carga.get(ag.id, 0) if lote else carga(db, ag),
            "vagas": list(vagas),
            "pendencias": lote.pendencias.get(ag.id, 0) if lote else db.scalar(select(func.count()).select_from(Pendencia).where(
                Pendencia.agente_id == ag.id, Pendencia.status == "aberta"))}


def _contagem_vaga(db, v: Vaga, lote: Lote | None = None) -> dict:
    if lote:
        rec, ana = lote.contagem.get(v.id, (0, 0))
        return {"recebidos": rec, "analisados": ana, "pendentes": rec - ana}
    rec = db.scalar(select(func.count()).select_from(Candidatura).where(Candidatura.vaga_id == v.id)) or 0
    ana = db.scalar(select(func.count()).select_from(Candidatura).where(Candidatura.vaga_id == v.id,
                                                                         Candidatura.status == "analisada")) or 0
    return {"recebidos": rec, "analisados": ana, "pendentes": rec - ana}


def vaga_json(db, v: Vaga, completo=False, lote: Lote | None = None) -> dict:
    resp = lote.responsaveis.get(v.id, []) if lote else responsaveis(db, v)
    d = {"id": v.id, "titulo": v.titulo, "area": v.area, "local": v.local, "status": v.status,
         "setores": v.setores, "responsaveis": [{"id": a.id, "nome": a.nome} for a in resp],
         "sem_responsavel": v.status == "aberta" and not resp, **_contagem_vaga(db, v, lote)}
    d["situacao"] = ("sem responsável" if d["sem_responsavel"] else v.status if v.status != "aberta" else
                     "em análise" if d["pendentes"] else "aguardando currículos" if not d["recebidos"] else "em dia")
    if completo:
        d.update(descricao=v.descricao, requisitos=v.requisitos)
    return d


def _soma(cond):
    return func.coalesce(func.sum(case((cond, 1), else_=0)), 0)


def quadro(db, lote: Lote | None = None, vagas: list[Vaga] | None = None) -> dict:
    lote = lote or Lote(db)
    vagas = vagas if vagas is not None else db.scalars(select(Vaga)).all()
    abertas = [v for v in vagas if v.status == "aberta"]
    hoje = agora().replace(hour=0, minute=0, second=0, microsecond=0)
    vj = [vaga_json(db, v, lote=lote) for v in abertas]
    recebidos, triados = db.execute(select(_soma(Curriculo.status != "anonimizado"), _soma(Curriculo.status == "triado"))).one()
    analises, analises_hoje = db.execute(select(func.count(), _soma(Analise.criado_em >= hoje)).select_from(Analise)).one()
    consultas, pendentes = db.execute(select(_soma((Tarefa.tipo == "CONSULTA") & (Tarefa.status == "concluida")),
                                             _soma(Tarefa.status.in_(("pendente", "executando"))))).one()
    return {
        "vagas_abertas": len(abertas),
        "vagas_sem_responsavel": sum(1 for v in vj if v["sem_responsavel"]),
        "vagas_em_processamento": sum(1 for v in vj if v["pendentes"]),
        "curriculos_recebidos": int(recebidos), "curriculos_triados": int(triados),
        "analises_concluidas": analises or 0, "analises_hoje": int(analises_hoje),
        "consultas_entre_colegas": int(consultas), "tarefas_pendentes": int(pendentes),
        "pendencias": lote.total_pendencias,
        "agentes_ativos": sum(1 for a in lote.ativos if a.funcao == "analista"),
        "estacoes": MAX_ESTACOES,
    }


def _eventos_recentes(db, limite=25, agente_id=None):
    q = select(Evento).where(Evento.tipo != "pensamento")
    if agente_id:
        q = q.where(Evento.agente_id == agente_id)
    return [eventos.serializar(e) for e in db.scalars(q.order_by(Evento.id.desc()).limit(limite)).all()]


def escritorio(db) -> dict:
    lote = Lote(db)
    vagas = db.scalars(select(Vaga).order_by(Vaga.id)).all()
    fila_triagem, fila_analise = db.execute(select(
        _soma((Tarefa.tipo == "TRIAGEM") & Tarefa.status.in_(("pendente", "executando"))),
        _soma((Tarefa.tipo == "ANALISE") & (Tarefa.status == "pendente")))).one()
    return {"agentes": [agente_json(db, a, lote) for a in lote.ativos],
            "fila_triagem": int(fila_triagem), "fila_analise": int(fila_analise),
            "vagas": [vaga_json(db, v, lote=lote) for v in vagas],
            "quadro": quadro(db, lote, vagas),
            "banco": acervo.resumo(db),
            "eventos": _eventos_recentes(db),
            "ia_modo": config.IA_MODO,
            "ia_pausa": (uso_ia.bloqueio() or uso_ia.pausa_do_provedor()) if config.IA_MODO == "gemini" else None,
            "automacao": {"contratacao": config.CONTRATACAO_AUTOMATICA, "colaboracao": config.COLABORACAO,
                          "encaminhamento": config.ENCAMINHAMENTO_AUTOMATICO, "limite_fila": config.LIMITE_FILA_POR_AGENTE}}


def _analise_json(a: Analise, c: Candidatura) -> dict:
    r = a.resultado or {}
    return {"analise_id": a.id, "candidatura_id": c.id, "nome": c.candidato.nome if c.candidato else "Candidato",
            "vaga_id": c.vaga_id, "vaga": c.vaga.titulo, "tipo": a.tipo, "origem": c.origem or "triagem",
            "compatibilidade": a.compatibilidade, "recomendado": a.recomendado, "avaliacao_humana": a.avaliacao_humana,
            "resumo": r.get("resumo_profissional", ""), "experiencias": r.get("experiencias_relevantes", []),
            "competencias": r.get("competencias", []), "atendidos": r.get("requisitos_atendidos", []),
            "a_verificar": r.get("a_verificar", []), "incompatibilidades": r.get("incompatibilidades", []),
            "justificativa": r.get("justificativa", ""),
            "colaboracoes": [{k: x.get(k) for k in ("colega", "especialidade", "pergunta", "resposta", "evidencias", "parecer")}
                             for x in r.get("colaboracoes", [])],
            "analisado_em": iso(a.criado_em), "modelo": a.modelo}


def _decisao_atual(db, cand_id):
    d = db.scalar(select(Decisao).where(Decisao.candidatura_id == cand_id).order_by(Decisao.id.desc()))
    return d.decisao if d else None


def _decisoes_atuais(db, ids) -> dict:
    atual = {}
    for cid, dec in db.execute(select(Decisao.candidatura_id, Decisao.decisao).where(
            Decisao.candidatura_id.in_(list(ids) or [-1])).order_by(Decisao.id)).all():
        atual[cid] = dec  # a mais recente fica
    return atual


def candidatos_do_agente(db, ag_id: int, vaga_id: int | None = None) -> list[dict]:
    q = (select(Analise, Candidatura).join(Candidatura, Analise.candidatura_id == Candidatura.id)
         .where(Analise.agente_id == ag_id).options(joinedload(Candidatura.candidato), joinedload(Candidatura.vaga)))
    if vaga_id:
        q = q.where(Candidatura.vaga_id == vaga_id)
    ordem = {"ALTA": 0, "MEDIA": 1, "BAIXA": 2}
    linhas = [(a, c) for a, c in db.execute(q).all() if not (c.candidato and c.candidato.anonimizado_em)]
    decisoes = _decisoes_atuais(db, {c.id for _, c in linhas})
    saida = []
    for a, c in linhas:
        j = _analise_json(a, c)
        j["decisao"] = decisoes.get(c.id)
        saida.append(j)
    saida.sort(key=lambda j: (not j["recomendado"], not j["avaliacao_humana"], ordem.get(j["compatibilidade"], 3), j["nome"]))
    return saida


def detalhe_agente(db, ag: Agente, lote: Lote | None = None) -> dict:
    lote = lote or Lote(db)
    base = agente_json(db, ag, lote)
    vagas = db.scalars(select(Vaga).where(Vaga.id.in_(base["vagas"] or [-1]))).all()
    por_id = {v.id: v for v in vagas}
    vj = []
    for i in base["vagas"]:
        d = vaga_json(db, por_id[i], completo=True, lote=lote)
        d["descricao"] = (d["descricao"] or "")[:220]
        vj.append(d)
    analisados, segundas, recomendados = db.execute(select(
        _soma(Analise.tipo == "principal"), _soma(Analise.tipo == "segunda_opiniao"), _soma(Analise.recomendado.is_(True)))
        .where(Analise.agente_id == ag.id)).one()
    base["estatisticas"] = {
        "recebidos": sum(v["recebidos"] for v in vj),
        "analisados": int(analisados), "segundas_opinioes": int(segundas),
        "ajudas_dadas": db.scalar(select(func.count()).select_from(Tarefa).where(
            Tarefa.agente_id == ag.id, Tarefa.tipo == "CONSULTA", Tarefa.status == "concluida")) or 0,
        "pendentes": lote.carga.get(ag.id, 0),
        "recomendados": int(recomendados),
        "vagas_em_andamento": sum(1 for v in vj if v["status"] == "aberta"),
    }
    base["vagas_detalhe"] = vj
    base["historico"] = _eventos_recentes(db, 8, ag.id)
    base["pendencias_abertas"] = [{"id": p.id, "tipo": p.tipo, "descricao": p.descricao} for p in db.scalars(
        select(Pendencia).where(Pendencia.agente_id == ag.id, Pendencia.status == "aberta")).all()]
    base["perfil_completo"] = {k: perfil(ag).get(k) for k in ("emoji", "resumo", "tom", "bordoes", "apresentacao")}
    return base


def contratacoes_recentes(db, limite=10) -> list[dict]:
    return [{"id": c.id, "acao": c.acao, "motivo": c.motivo, "descricao": c.descricao, "por": c.por,
             "agente_id": c.agente_id, "em": iso(c.criado_em)}
            for c in db.scalars(select(Contratacao).order_by(Contratacao.id.desc()).limit(limite)).all()]


def contexto_chat(db, ag: Agente, lote: Lote | None = None) -> dict:
    lote = lote or Lote(db)
    d = detalhe_agente(db, ag, lote)
    cands = candidatos_do_agente(db, ag.id)[:15]  # contexto enxuto: menos tokens por pergunta
    ctx = {
        "agente": {k: d[k] for k in ("nome", "funcao", "especialidade", "estado", "atividade")},
        "personalidade": publico(perfil(ag)),
        "estatisticas": d["estatisticas"],
        "vagas": [{k: v[k] for k in ("titulo", "status", "recebidos", "analisados", "pendentes", "requisitos", "situacao")}
                  for v in d["vagas_detalhe"]],
        "candidatos": [{k: c[k] for k in ("nome", "vaga", "compatibilidade", "recomendado", "avaliacao_humana",
                                          "a_verificar", "justificativa", "decisao", "colaboracoes", "origem")} | {
            "evidencias": [f'{x.get("requisito")}: “{x.get("evidencia")}”' for x in c["atendidos"]][:3]} for c in cands],
        "pendencias": d["pendencias_abertas"],
        "atividade_recente": [f'{e["ts"]} {e["mensagem"]}' for e in d["historico"]],
        "colegas": [{"nome": a.nome, "funcao": a.funcao,
                     "especialidade": a.especialidade.nome if a.especialidade else
                     ("coordenação" if a.funcao == "supervisor" else "recepção e triagem")}
                    for a in lote.ativos if a.id != ag.id],
        "fila_recepcao": db.scalar(select(func.count()).select_from(Tarefa).where(
            Tarefa.tipo == "TRIAGEM", Tarefa.status.in_(("pendente", "executando")))) or 0,
        "curriculos_registrados": db.scalar(select(func.count()).select_from(Curriculo)) or 0,
        "quadro": quadro(db, lote),
    }
    if ag.funcao == "supervisor":
        ctx["equipe"] = [{k: x[k] for k in ("nome", "funcao", "especialidade", "estado", "atividade", "carga")}
                         for x in (agente_json(db, a, lote) for a in lote.ativos)]
        ctx["contratacoes"] = contratacoes_recentes(db, 6)
        ctx["limite_fila"] = config.LIMITE_FILA_POR_AGENTE
    return ctx


def contexto_equipe(db, por_agente: bool = False) -> dict:
    """Contexto do canal da equipe: quem é quem, como está cada um, vagas e destaques."""
    lote = Lote(db)
    ags = lote.ativos
    vagas = db.scalars(select(Vaga).order_by(Vaga.id)).all()
    titulo_aberta = {v.id: v.titulo for v in vagas if v.status == "aberta"}
    equipe = []
    for a in ags:
        j = agente_json(db, a, lote)
        equipe.append({"id": a.id, "nome": a.nome, "funcao": a.funcao, "especialidade": j["especialidade"],
                       "estado": j["estado"], "atividade": j["atividade"], "carga": j["carga"],
                       "vagas": [titulo_aberta[i] for i in lote.vagas_de.get(a.id, []) if i in titulo_aberta],
                       "personalidade": {k: perfil(a).get(k) for k in ("resumo", "tom", "bordoes")}})
    candidatos = []
    for a in (x for x in ags if x.funcao == "analista"):
        for c in candidatos_do_agente(db, a.id)[:5]:
            candidatos.append({"nome": primeiro_nome(c["nome"]), "vaga": c["vaga"], "compatibilidade": c["compatibilidade"],
                               "recomendado": c["recomendado"], "avaliacao_humana": c["avaliacao_humana"],
                               "decisao": c["decisao"], "analista": a.nome,
                               "consultou": [x["colega"] for x in c["colaboracoes"]]})
    ctx = {"quadro": quadro(db, lote, vagas), "equipe": equipe,
           "vagas": [{"titulo": v["titulo"], "status": v["status"], "responsaveis": [r["nome"] for r in v["responsaveis"]],
                      "recebidos": v["recebidos"], "analisados": v["analisados"], "pendentes": v["pendentes"]}
                     for v in (vaga_json(db, v, lote=lote) for v in vagas if v.status == "aberta")],
           "candidatos": candidatos[:20],
           "conversas_recentes": [e["mensagem"] for e in conversas_equipe(db, 6)["eventos"] if e["tipo"] == "conversa"],
           "contratacoes": contratacoes_recentes(db, 5),
           "fila_recepcao": db.scalar(select(func.count()).select_from(Tarefa).where(
               Tarefa.tipo == "TRIAGEM", Tarefa.status.in_(("pendente", "executando")))) or 0}
    if por_agente:
        ctx["agentes_ctx"] = {a.id: contexto_chat(db, a, lote) for a in ags}
    return ctx


TIPOS_CANAL = ("conversa", "contratacao", "agente_dispensado", "indicacao", "encaminhamento_automatico", "fila_redistribuida")


def conversas_equipe(db, limite=60) -> dict:
    evs = db.scalars(select(Evento).where(Evento.tipo.in_(TIPOS_CANAL)).order_by(Evento.id.desc()).limit(limite)).all()
    ids = {e.agente_id for e in evs} | {(e.dados or {}).get("destino_id") for e in evs}
    ags = db.scalars(select(Agente).where(Agente.id.in_([i for i in ids if i]))).all() if ids else []
    return {"eventos": [eventos.serializar(e) for e in reversed(evs)],
            "agentes": {a.id: {"nome": a.nome, "aparencia": a.aparencia, "funcao": a.funcao, "ativo": a.ativo,
                               "emoji": perfil(a).get("emoji", "")} for a in ags}}


def detalhe_candidatura(db, c: Candidatura) -> dict:
    cand, cur = c.candidato, c.curriculo
    analises = db.scalars(select(Analise).where(Analise.candidatura_id == c.id).order_by(Analise.id)).all()
    nomes = {a.id: a.nome for a in db.scalars(select(Agente)).all()}
    return {
        "candidatura_id": c.id, "status": c.status, "decisao": _decisao_atual(db, c.id),
        "origem": c.origem or "triagem", "indicado_por": nomes.get(c.indicado_por_id),
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
    setores = dados.get("setores") or []
    if not setores:  # o Atlas sugere as pastas a partir do título e da descrição
        try:
            setores = ia.sugerir_setores(dados["titulo"], dados.get("descricao", "")).get("setores", [])
        except ia.IAIndisponivel as e:
            raise ErroNegocio(f"Não consegui sugerir as pastas agora ({e}); escolha ao menos uma.")
        if not setores:
            raise ErroNegocio("Não consegui deduzir as pastas da triagem para esta vaga; escolha ao menos uma.")
    v = Vaga(titulo=dados["titulo"].strip(), area=dados.get("area", ""), local=dados.get("local", ""),
             descricao=dados.get("descricao", ""), requisitos=[r for r in dados.get("requisitos", []) if r.strip()],
             setores=setores)
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
    n = sum(_encaminhar(db, c, v, origem="banco") for c in curs)
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
        if ag:
            _enfileirar_espera(db, v, ag)
    eventos.registrar(db, "vaga_status", f"A vaga “{v.titulo}” agora está {status}", atlas(db).id, vaga_id=v.id, status=status)


def resolver_pendencia(db, p: Pendencia, acao: str, usuario_id: int, setor: str | None = None):
    if p.status != "aberta":
        raise ErroNegocio("Pendência já resolvida.")
    if acao == "ativar":
        esp = db.get(Especialidade, p.especialidade_id) if p.especialidade_id else None
        if not esp:
            raise ErroNegocio("Pendência sem especialidade associada.")
        motivo = {"PROPOSTA_CONTRATACAO": "sobrecarga" if "fila" in p.descricao else "consulta"}.get(p.tipo, "sem_especialista")
        contratar(db, esp, motivo, por="O recrutador")  # resolve as pendências da especialidade
        p.status, p.resolvido_em, p.resolvido_por_id = "resolvida", agora(), usuario_id
        return
    if acao == "reprocessar":
        _reprocessar(db, p)
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
        "finalidade": f"Recrutamento e seleção para vagas {config.DA_EMPRESA}",
        "retencao_dias": config.RETENCAO_DIAS,
        "compartilhamento": "Provedor de IA (Google Gemini) como operador, quando IA_MODO=gemini" if config.IA_MODO == "gemini" else "Nenhum",
        "curriculos": [{"id": c.id, "arquivo": c.arquivo_nome, "recebido_em": iso(c.recebido_em), "origem": c.origem,
                        "pasta": c.setor, "texto": c.texto, "triagem": c.triagem_json} for c in curs],
        "candidaturas": [detalhe_candidatura(db, c) for c in cs],
        "conversas_dos_agentes": [{"em": iso(e.ts), "tipo": e.tipo, "texto": e.mensagem} for e in db.scalars(
            select(Evento).where(Evento.candidato_id == cand.id, Evento.tipo.in_(FALAS)).order_by(Evento.id)).all()],
    }


def anonimizar_titular(db, cand: Candidato, motivo: str):
    nome_antigo = cand.nome
    curs = db.scalars(select(Curriculo).where(Curriculo.candidato_id == cand.id)).all()
    for c in curs:
        apagar_arquivo(c.caminho, db)
        c.caminho, c.texto, c.triagem_json, c.resumo_triagem = None, None, None, None
        c.arquivo_nome, c.status, c.sha256 = "removido", "anonimizado", f"anon-{c.id}"
    ids = select(Candidatura.id).where(Candidatura.candidato_id == cand.id)
    for a in db.scalars(select(Analise).where(Analise.candidatura_id.in_(ids))).all():
        a.resultado = {}
    for d in db.scalars(select(Decisao).where(Decisao.candidatura_id.in_(ids))).all():
        d.observacao = ""
    for t in db.scalars(select(Tarefa).where(or_(Tarefa.candidatura_id.in_(ids),
                                                 Tarefa.curriculo_id.in_([c.id for c in curs] or [-1])))).all():
        t.dados = None  # análises preliminares e respostas de colegas
        if t.status == "pendente":
            t.status = "cancelada"
    # pensamentos e conversas sobre o titular são apagados; os demais eventos têm o nome substituído
    db.execute(delete(Evento).where(Evento.candidato_id == cand.id, Evento.tipo.in_(FALAS)))
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
COR_ATLAS = "#2F3E66"  # azul-marinho do gerente


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
            Agente(nome="Atlas", funcao="supervisor", personalidade="Coordenador calmo e objetivo; distribui o trabalho pela especialidade e pela carga de cada agente, e contrata reforço quando precisa.",
                   aparencia=aparencia_para("Atlas", pele="#D9A27A", cabelo="#1E1A16", estilo="curto", camisa=COR_ATLAS, oculos=True, barba=True)),
            Agente(nome="Bia", funcao="recepcao", personalidade="Recepcionista organizada; registra cada currículo e faz a triagem inicial.",
                   aparencia=aparencia_para("Bia", pele="#B97D55", cabelo="#1E1A16", estilo="black", camisa="#8E6BBF", oculos=False, barba=False)),
            Agente(nome="Sofia", funcao="analista", especialidade_id=esp["administrativo"].id, estacao=0,
                   personalidade="Analista atenta a detalhes e a rotinas administrativas.",
                   aparencia=aparencia_para("Sofia", pele="#F6D1B1", cabelo="#6E3C1E", estilo="longo", camisa="#3B6EC8", oculos=False, barba=False)),
            Agente(nome="Flávio", funcao="analista", especialidade_id=esp["operacional"].id, estacao=1,
                   personalidade="Analista prático, conhece bem a rotina de produção e logística.",
                   aparencia=aparencia_para("Flávio", pele="#EEC49C", cabelo="#3B2618", estilo="topete", camisa="#3CAA6E", oculos=False, barba=True)),
        ])
        db.flush()
    for ag in db.scalars(select(Agente)).all():  # bancos antigos: cada agente ganha sua personalidade
        if not ag.perfil:
            ag.perfil = perfil_para(ag.nome, ag.especialidade.nome if ag.especialidade else None)
        if ag.funcao == "supervisor" and (ag.aparencia or {}).get("camisa") == "#FFCD00":  # identidade visual antiga
            ag.aparencia = {**ag.aparencia, "camisa": COR_ATLAS}
    if config.ADMIN_SENHA and not db.scalar(select(Usuario).where(Usuario.email == config.ADMIN_EMAIL.lower())):
        db.add(Usuario(nome=config.ADMIN_NOME, email=config.ADMIN_EMAIL.lower(), senha_hash=gerar_hash(config.ADMIN_SENHA),
                       papel="admin", trocar_senha=True))  # a senha do .env serve só para o primeiro acesso
