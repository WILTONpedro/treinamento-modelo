"""Modelo de dados. Datas em UTC (naive) — o front converte para o fuso local."""
from datetime import datetime, timezone
from sqlalchemy import (String, Integer, Boolean, DateTime, Text, JSON, ForeignKey, LargeBinary,
                        UniqueConstraint, Index, Float, CheckConstraint, text)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def agora():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def iso(dt):
    return dt.isoformat(timespec="seconds") + "Z" if dt else None


def valores(tabela: str, coluna: str, *permitidos: str, nulo: bool = False) -> CheckConstraint:
    """Regra no banco: a coluna só aceita estes valores (o banco recusa um estado digitado errado)."""
    lista = ", ".join(f"'{v}'" for v in permitidos)
    return CheckConstraint(f"{coluna} IN ({lista})" + (f" OR {coluna} IS NULL" if nulo else ""), name=f"ck_{tabela}_{coluna}")


def minimo(tabela: str, *colunas: str) -> tuple[CheckConstraint, ...]:
    return tuple(CheckConstraint(f"{c} >= 0", name=f"ck_{tabela}_{c}") for c in colunas)


# --------------------------------------------------------------------------- pessoas
class Usuario(Base):
    __tablename__ = "usuarios"
    __table_args__ = (valores("usuarios", "papel", "recrutador", "supervisor", "admin"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    nome: Mapped[str] = mapped_column(String(120))
    email: Mapped[str] = mapped_column(String(200), unique=True)
    senha_hash: Mapped[str] = mapped_column(String(300))
    papel: Mapped[str] = mapped_column(String(20), default="recrutador")  # admin|supervisor|recrutador
    ativo: Mapped[bool] = mapped_column(Boolean, default=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)
    trocar_senha: Mapped[bool | None] = mapped_column(Boolean, default=False)  # senha temporária: troca no próximo acesso
    versao_sessao: Mapped[int | None] = mapped_column(Integer, default=0)  # muda ao trocar senha/desativar: derruba sessões
    ultimo_acesso_em: Mapped[datetime | None] = mapped_column(DateTime)


# --------------------------------------------------------------------------- equipe de agentes
class Especialidade(Base):
    """Especialidades são dados: novas podem ser cadastradas sem mudar o código."""
    __tablename__ = "especialidades"
    id: Mapped[int] = mapped_column(primary_key=True)
    codigo: Mapped[str] = mapped_column(String(40), unique=True)
    nome: Mapped[str] = mapped_column(String(120))
    descricao: Mapped[str] = mapped_column(Text, default="")
    setores: Mapped[list] = mapped_column(JSON, default=list)  # categorias da triagem atendidas
    nomes_sugeridos: Mapped[list] = mapped_column(JSON, default=list)


class Agente(Base):
    __tablename__ = "agentes"
    __table_args__ = (valores("agentes", "funcao", "supervisor", "recepcao", "analista"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    nome: Mapped[str] = mapped_column(String(60))
    funcao: Mapped[str] = mapped_column(String(20))  # supervisor|recepcao|analista
    especialidade_id: Mapped[int | None] = mapped_column(ForeignKey("especialidades.id"), index=True)
    personalidade: Mapped[str] = mapped_column(Text, default="")
    perfil: Mapped[dict | None] = mapped_column(JSON, default=dict)  # jeito de falar, bordões, papo (personalidade.py)
    aparencia: Mapped[dict] = mapped_column(JSON, default=dict)
    estacao: Mapped[int | None] = mapped_column(Integer)
    ativo: Mapped[bool] = mapped_column(Boolean, default=True)  # cadastrado ≠ ativo
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)
    ativado_em: Mapped[datetime | None] = mapped_column(DateTime, default=agora)
    # estado operacional (o estado exibido é derivado destes campos)
    tarefa_atual_id: Mapped[int | None] = mapped_column(Integer)
    atividade: Mapped[str] = mapped_column(String(200), default="")
    ultima_atividade_em: Mapped[datetime | None] = mapped_column(DateTime)
    concluido_em: Mapped[datetime | None] = mapped_column(DateTime)

    especialidade: Mapped[Especialidade | None] = relationship()


# --------------------------------------------------------------------------- vagas
class Vaga(Base):
    __tablename__ = "vagas"
    __table_args__ = (valores("vagas", "status", "aberta", "suspensa", "encerrada", "cancelada"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    titulo: Mapped[str] = mapped_column(String(160))
    area: Mapped[str] = mapped_column(String(120), default="")
    local: Mapped[str] = mapped_column(String(160), default="")
    descricao: Mapped[str] = mapped_column(Text, default="")
    requisitos: Mapped[list] = mapped_column(JSON, default=list)
    setores: Mapped[list] = mapped_column(JSON, default=list)  # categorias da triagem que alimentam a vaga
    status: Mapped[str] = mapped_column(String(20), default="aberta")  # aberta|suspensa|encerrada|cancelada
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)
    atualizado_em: Mapped[datetime] = mapped_column(DateTime, default=agora, onupdate=agora)


class VagaAgente(Base):
    __tablename__ = "vaga_agentes"
    vaga_id: Mapped[int] = mapped_column(ForeignKey("vagas.id"), primary_key=True)
    agente_id: Mapped[int] = mapped_column(ForeignKey("agentes.id"), primary_key=True, index=True)
    atribuido_em: Mapped[datetime] = mapped_column(DateTime, default=agora)


# --------------------------------------------------------------------------- candidatos e currículos
class Candidato(Base):
    __tablename__ = "candidatos"
    # um e-mail = um candidato ativo (o mesmo titular mandando outro currículo cai no mesmo cadastro)
    __table_args__ = (Index("uq_candidatos_email_ativo", "email", unique=True,
                            postgresql_where=text("email IS NOT NULL AND anonimizado_em IS NULL"),
                            sqlite_where=text("email IS NOT NULL AND anonimizado_em IS NULL")),)
    id: Mapped[int] = mapped_column(primary_key=True)
    nome: Mapped[str] = mapped_column(String(200), default="Candidato")
    email: Mapped[str | None] = mapped_column(String(200), index=True)
    telefone: Mapped[str | None] = mapped_column(String(60))
    cidade: Mapped[str | None] = mapped_column(String(120))
    dado_sensivel: Mapped[bool] = mapped_column(Boolean, default=False)  # ex.: menção a deficiência (art. 11 LGPD)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)
    ultima_atividade_em: Mapped[datetime] = mapped_column(DateTime, default=agora)
    anonimizado_em: Mapped[datetime | None] = mapped_column(DateTime)


class Curriculo(Base):
    __tablename__ = "curriculos"
    __table_args__ = (valores("curriculos", "status", "recebido", "triado", "invalido", "erro", "anonimizado"),
                      valores("curriculos", "origem", "email", "upload", "imap", "demo"),
                      *minimo("curriculos", "tamanho"))
    id: Mapped[int] = mapped_column(primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True)  # evita reprocessar o mesmo arquivo
    candidato_id: Mapped[int | None] = mapped_column(ForeignKey("candidatos.id"), index=True)
    arquivo_nome: Mapped[str] = mapped_column(String(260))
    mime: Mapped[str] = mapped_column(String(100), default="application/octet-stream")
    tamanho: Mapped[int] = mapped_column(Integer, default=0)
    caminho: Mapped[str | None] = mapped_column(String(300))  # onde está o arquivo cifrado (disco ou "banco:<sha>")
    texto: Mapped[str | None] = mapped_column(Text)
    origem: Mapped[str] = mapped_column(String(20), default="email")
    recebido_em: Mapped[datetime] = mapped_column(DateTime, default=agora)
    status: Mapped[str] = mapped_column(String(20), default="recebido")  # recebido|triado|invalido|erro|anonimizado
    setor: Mapped[str | None] = mapped_column(String(60))
    confianca: Mapped[str | None] = mapped_column(String(20))
    anos_experiencia: Mapped[int | None] = mapped_column(Integer)
    resumo_triagem: Mapped[str | None] = mapped_column(Text)
    triagem_json: Mapped[dict | None] = mapped_column(JSON)

    candidato: Mapped[Candidato | None] = relationship()


class ArquivoCurriculo(Base):
    """Arquivo original do currículo, cifrado (Fernet), quando ARMAZENAMENTO=banco (ver arquivos.py)."""
    __tablename__ = "arquivos_curriculo"
    __table_args__ = (CheckConstraint("length(conteudo) <= 52428800", name="ck_arquivos_curriculo_tamanho"),)  # 50 MB
    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    conteudo: Mapped[bytes] = mapped_column(LargeBinary)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)


class Candidatura(Base):
    """Um currículo considerado em uma vaga. Única por (currículo, vaga)."""
    __tablename__ = "candidaturas"
    __table_args__ = (UniqueConstraint("curriculo_id", "vaga_id", name="uq_curriculo_vaga"),
                      valores("candidaturas", "status", "na_fila", "em_analise", "analisada", "erro"),
                      valores("candidaturas", "origem", "triagem", "banco", "atlas", "indicacao", nulo=True))
    id: Mapped[int] = mapped_column(primary_key=True)
    curriculo_id: Mapped[int] = mapped_column(ForeignKey("curriculos.id"))
    vaga_id: Mapped[int] = mapped_column(ForeignKey("vagas.id"), index=True)
    candidato_id: Mapped[int | None] = mapped_column(ForeignKey("candidatos.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="na_fila")  # na_fila|em_analise|analisada|erro
    # como chegou à vaga: triagem (pasta) | banco (banco de talentos) | atlas (encaminhado) | indicacao (colega)
    origem: Mapped[str | None] = mapped_column(String(20), default="triagem")
    indicado_por_id: Mapped[int | None] = mapped_column(Integer)  # agente cuja conversa originou a indicação
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)

    curriculo: Mapped[Curriculo] = relationship()
    vaga: Mapped[Vaga] = relationship()
    candidato: Mapped[Candidato | None] = relationship()


class Analise(Base):
    __tablename__ = "analises"
    __table_args__ = (UniqueConstraint("candidatura_id", "agente_id", name="uq_analise_agente"),
                      valores("analises", "tipo", "principal", "segunda_opiniao"),
                      valores("analises", "compatibilidade", "ALTA", "MEDIA", "BAIXA"),
                      *minimo("analises", "duracao_ms"))
    id: Mapped[int] = mapped_column(primary_key=True)
    candidatura_id: Mapped[int] = mapped_column(ForeignKey("candidaturas.id"))
    agente_id: Mapped[int] = mapped_column(ForeignKey("agentes.id"), index=True)
    tipo: Mapped[str] = mapped_column(String(20), default="principal")  # principal|segunda_opiniao
    compatibilidade: Mapped[str] = mapped_column(String(10), default="BAIXA")
    recomendado: Mapped[bool] = mapped_column(Boolean, default=False)
    avaliacao_humana: Mapped[bool] = mapped_column(Boolean, default=False)
    resultado: Mapped[dict] = mapped_column(JSON, default=dict)
    modelo: Mapped[str] = mapped_column(String(60), default="")
    duracao_ms: Mapped[int] = mapped_column(Integer, default=0)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)

    candidatura: Mapped[Candidatura] = relationship()


class Decisao(Base):
    """Decisão humana — a única que conta para o processo seletivo."""
    __tablename__ = "decisoes"
    __table_args__ = (valores("decisoes", "decisao", "entrevista", "banco_talentos", "descartado"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    candidatura_id: Mapped[int] = mapped_column(ForeignKey("candidaturas.id"), index=True)
    decisao: Mapped[str] = mapped_column(String(20))  # entrevista|banco_talentos|descartado
    observacao: Mapped[str] = mapped_column(Text, default="")
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id"), index=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)


# --------------------------------------------------------------------------- fila, pendências e registros
class Tarefa(Base):
    __tablename__ = "tarefas"
    __table_args__ = (Index("ix_tarefa_fila", "status", "tipo", "agente_id"),
                      valores("tarefas", "tipo", "TRIAGEM", "DISTRIBUICAO", "ANALISE", "CONSULTA", "REVISAO"),
                      valores("tarefas", "status", "pendente", "executando", "concluida", "erro", "cancelada"),
                      *minimo("tarefas", "tentativas"))
    id: Mapped[int] = mapped_column(primary_key=True)
    tipo: Mapped[str] = mapped_column(String(20))  # TRIAGEM|DISTRIBUICAO|ANALISE|CONSULTA|REVISAO
    curriculo_id: Mapped[int | None] = mapped_column(ForeignKey("curriculos.id"), index=True)
    candidatura_id: Mapped[int | None] = mapped_column(ForeignKey("candidaturas.id"), index=True)
    agente_id: Mapped[int | None] = mapped_column(ForeignKey("agentes.id"), index=True)
    motivo: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="pendente")  # pendente|executando|concluida|erro
    tentativas: Mapped[int] = mapped_column(Integer, default=0)
    erro: Mapped[str | None] = mapped_column(Text)
    dados: Mapped[dict | None] = mapped_column(JSON)  # consulta entre colegas: pergunta, análise preliminar, resposta
    criada_em: Mapped[datetime] = mapped_column(DateTime, default=agora)
    iniciada_em: Mapped[datetime | None] = mapped_column(DateTime)
    concluida_em: Mapped[datetime | None] = mapped_column(DateTime)


class Pendencia(Base):
    __tablename__ = "pendencias"
    __table_args__ = (valores("pendencias", "status", "aberta", "resolvida"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    # ARQUIVO_ILEGIVEL|SEM_ESPECIALISTA|TRIAGEM_MANUAL|ERRO_IA|PROPOSTA_CONTRATACAO|ESTACOES_LOTADAS
    tipo: Mapped[str] = mapped_column(String(40))
    descricao: Mapped[str] = mapped_column(Text)
    agente_id: Mapped[int | None] = mapped_column(ForeignKey("agentes.id"), index=True)
    curriculo_id: Mapped[int | None] = mapped_column(ForeignKey("curriculos.id"), index=True)
    candidatura_id: Mapped[int | None] = mapped_column(ForeignKey("candidaturas.id"), index=True)
    vaga_id: Mapped[int | None] = mapped_column(ForeignKey("vagas.id"), index=True)
    especialidade_id: Mapped[int | None] = mapped_column(ForeignKey("especialidades.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="aberta")
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)
    resolvido_em: Mapped[datetime | None] = mapped_column(DateTime)
    resolvido_por_id: Mapped[int | None] = mapped_column(ForeignKey("usuarios.id"), index=True)


class Evento(Base):
    """Linha do tempo do escritório: alimenta animações, balões, histórico e o chat.

    `pensamento` e `conversa` são as falas dos agentes; ficam só RETENCAO_CONVERSAS_HORAS.
    """
    __tablename__ = "eventos"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=agora, index=True)
    tipo: Mapped[str] = mapped_column(String(40))
    agente_id: Mapped[int | None] = mapped_column(ForeignKey("agentes.id"), index=True)
    candidato_id: Mapped[int | None] = mapped_column(Integer, index=True)  # permite apagar tudo de um titular (LGPD)
    mensagem: Mapped[str] = mapped_column(Text)
    dados: Mapped[dict] = mapped_column(JSON, default=dict)


class Contratacao(Base):
    """Histórico de contratações (ativações) e dispensas feitas pelo Atlas ou pelo recrutador."""
    __tablename__ = "contratacoes"
    __table_args__ = (valores("contratacoes", "acao", "contratacao", "retorno", "dispensa", "sem_estacao"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    especialidade_id: Mapped[int | None] = mapped_column(ForeignKey("especialidades.id"), index=True)
    agente_id: Mapped[int | None] = mapped_column(ForeignKey("agentes.id"), index=True)
    acao: Mapped[str] = mapped_column(String(20), default="contratacao")  # contratacao|retorno|dispensa|sem_estacao
    motivo: Mapped[str] = mapped_column(String(30), default="manual")  # sem_especialista|sobrecarga|consulta|manual|estacao
    descricao: Mapped[str] = mapped_column(Text, default="")
    por: Mapped[str] = mapped_column(String(120), default="Atlas")
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora, index=True)


class Auditoria(Base):
    """Registro das operações com dados pessoais (art. 37 LGPD)."""
    __tablename__ = "auditoria"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=agora, index=True)
    usuario_id: Mapped[int | None] = mapped_column(ForeignKey("usuarios.id"), index=True)
    acao: Mapped[str] = mapped_column(String(60))
    recurso: Mapped[str] = mapped_column(String(40))
    recurso_id: Mapped[int | None] = mapped_column(Integer)
    ip: Mapped[str | None] = mapped_column(String(64))
    detalhes: Mapped[dict] = mapped_column(JSON, default=dict)


class SolicitacaoTitular(Base):
    """Pedidos de titulares (art. 18 LGPD): acesso, correção, eliminação."""
    __tablename__ = "solicitacoes_titular"
    __table_args__ = (valores("solicitacoes_titular", "tipo", "acesso", "correcao", "eliminacao"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    candidato_id: Mapped[int] = mapped_column(ForeignKey("candidatos.id"), index=True)
    tipo: Mapped[str] = mapped_column(String(20))
    observacao: Mapped[str] = mapped_column(Text, default="")
    usuario_id: Mapped[int] = mapped_column(ForeignKey("usuarios.id"), index=True)
    criado_em: Mapped[datetime] = mapped_column(DateTime, default=agora)
    concluido_em: Mapped[datetime | None] = mapped_column(DateTime)


class UsoIA(Base):
    """Tokens e custo estimado de cada chamada à IA (sem conteúdo de currículo)."""
    __tablename__ = "uso_ia"
    __table_args__ = minimo("uso_ia", "entrada", "saida", "pensamento", "cache", "custo_usd")
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime, default=agora, index=True)
    finalidade: Mapped[str] = mapped_column(String(30))  # triagem|analise|consulta|revisao|chat|equipe|...
    modelo: Mapped[str] = mapped_column(String(60))
    entrada: Mapped[int] = mapped_column(Integer, default=0)
    saida: Mapped[int] = mapped_column(Integer, default=0)
    pensamento: Mapped[int] = mapped_column(Integer, default=0)
    cache: Mapped[int] = mapped_column(Integer, default=0)
    custo_usd: Mapped[float] = mapped_column(Float, default=0.0)


class Configuracao(Base):
    __tablename__ = "configuracoes"
    chave: Mapped[str] = mapped_column(String(60), primary_key=True)
    valor: Mapped[str] = mapped_column(Text)


class Comando(Base):
    """Vários computadores no mesmo banco: quem comanda a equipe de agentes agora. Vale até `ate`; quem comanda
    renova a cada poucos segundos e, se parar (programa fechado, sem rede), outro computador assume."""
    __tablename__ = "comando"
    nome: Mapped[str] = mapped_column(String(40), primary_key=True)
    dono: Mapped[str] = mapped_column(String(200))
    ate: Mapped[datetime] = mapped_column(DateTime)
