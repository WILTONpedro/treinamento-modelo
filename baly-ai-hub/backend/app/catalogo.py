"""Categorias da triagem (as mesmas da API original) e especialidades iniciais dos agentes."""
import hashlib

# LISTA DE CATEGORIAS — idêntica à API de triagem da Baly
CATEGORIAS_DISPONIVEIS = [
    "ADMINISTRATIVO", "ALMOXARIFADO", "AREA INDUSTRIAL", "COMERCIAL", "COMERCIO EXTERIOR",
    "COMPRAS", "CONTABILIDADE", "COORDENADOR DE EXPEDIÇÃO", "COORDENADOR DE MERCHANDISING",
    "EMPILHADEIRA", "EVENTOS", "FINANCEIRO", "GERENTE COMERCIAL", "GERENTE FINANCEIRO",
    "GERENTE DE GRANDES CONTAS", "GERENTE LOGISTICA", "GERENTE MARKETING", "GERENTE PRODUÇÃO",
    "GERENTE QUALIDADE", "GERENTE DE RH", "GERENTE VENDAS", "HIGIENIZAÇÃO", "JOVEM APRENDIZ",
    "KEY ACCOUNT", "LIDER DE PRODUÇÃO", "LOGÍSTICA", "MARKETING", "MECANICA INDUSTRIAL", "EXECUTIVO DE VENDAS",
    "MERCHANDISING", "MOTORISTA", "PCD", "PCP", "PRODUÇÃO", "PROJETOS", "PROMOTOR DE VENDAS",
    "QUALIDADE", "RECURSOS HUMANOS", "SUPERVISOR DE MERCHANDISING", "SUPERVISOR DE VENDAS", "TI", "VENDAS", "VIGIA", "OUTROS"
]

# Categorias que envolvem dado sensível ou de adolescente — tratamento reforçado (LGPD arts. 11 e 14)
CATEGORIAS_SENSIVEIS = {"PCD", "JOVEM APRENDIZ"}

# Especialidades iniciais. São gravadas no banco e podem ser editadas/ampliadas.
ESPECIALIDADES = [
    {"codigo": "operacional", "nome": "Recrutamento operacional",
     "descricao": "Produção, logística, armazém, manutenção e funções de base operacional.",
     "setores": ["PRODUÇÃO", "ALMOXARIFADO", "LOGÍSTICA", "EMPILHADEIRA", "MOTORISTA", "HIGIENIZAÇÃO",
                 "MECANICA INDUSTRIAL", "VIGIA", "LIDER DE PRODUÇÃO", "AREA INDUSTRIAL",
                 "COORDENADOR DE EXPEDIÇÃO", "PCP", "QUALIDADE"],
     "nomes": ["Flávio", "Rafael", "Joana"]},
    {"codigo": "administrativo", "nome": "Vagas administrativas e de escritório",
     "descricao": "Rotinas administrativas, financeiro, compras, contabilidade e RH.",
     "setores": ["ADMINISTRATIVO", "FINANCEIRO", "COMPRAS", "CONTABILIDADE", "RECURSOS HUMANOS"],
     "nomes": ["Sofia", "Helena", "Clara"]},
    {"codigo": "comercial", "nome": "Vagas comerciais e de vendas",
     "descricao": "Vendas, representação, key account e comércio exterior.",
     "setores": ["COMERCIAL", "VENDAS", "EXECUTIVO DE VENDAS", "KEY ACCOUNT", "SUPERVISOR DE VENDAS",
                 "COMERCIO EXTERIOR"],
     "nomes": ["Marina", "Lucas", "Bianca"]},
    {"codigo": "promotores", "nome": "Promotores e merchandising",
     "descricao": "Promotores de vendas, merchandising e equipes de ponto de venda.",
     "setores": ["PROMOTOR DE VENDAS", "MERCHANDISING", "COORDENADOR DE MERCHANDISING",
                 "SUPERVISOR DE MERCHANDISING"],
     "nomes": ["Caio", "Larissa"]},
    {"codigo": "marketing", "nome": "Vagas de marketing",
     "descricao": "Marketing, comunicação e eventos.",
     "setores": ["MARKETING", "EVENTOS"],
     "nomes": ["Júlia", "Pedro"]},
    {"codigo": "lideranca", "nome": "Liderança e gestão",
     "descricao": "Gerências e coordenação de áreas.",
     "setores": ["GERENTE COMERCIAL", "GERENTE FINANCEIRO", "GERENTE DE GRANDES CONTAS", "GERENTE LOGISTICA",
                 "GERENTE MARKETING", "GERENTE PRODUÇÃO", "GERENTE QUALIDADE", "GERENTE DE RH",
                 "GERENTE VENDAS", "PROJETOS"],
     "nomes": ["Renato", "Beatriz"]},
    {"codigo": "ti", "nome": "Tecnologia da informação",
     "descricao": "Suporte, infraestrutura, desenvolvimento e redes.",
     "setores": ["TI"], "nomes": ["Lia", "Thiago"]},
    {"codigo": "inclusao", "nome": "Inclusão e aprendizagem",
     "descricao": "Vagas PCD e Jovem Aprendiz, com cuidado reforçado no tratamento de dados.",
     "setores": ["PCD", "JOVEM APRENDIZ"], "nomes": ["Bruno", "Aline"]},
]

# Nomes de reserva quando os sugeridos da especialidade já estão em uso
NOMES_EXTRAS = ["Nina", "Otávio", "Yara", "Igor", "Lívia", "Caetano", "Tainá", "Heitor", "Elis", "Davi", "Rosa", "Ícaro"]

PELES = ["#F6D1B1", "#EEC49C", "#D9A27A", "#B97D55", "#8D5A3B", "#6B4128"]
CABELOS = ["#1E1A16", "#3B2618", "#6E3C1E", "#A0612E", "#C9A15B", "#8A8A8A", "#D9D9D9"]
CAMISAS = ["#3B6EC8", "#3CAA6E", "#E68C3C", "#8E6BBF", "#E678AA", "#2F9C9C", "#D64545", "#5A6270", "#F0F0F0"]
ESTILOS = ["curto", "cacheado", "longo", "coque", "topete", "careca", "black"]


def aparencia_para(nome: str, **forcar) -> dict:
    """Aparência determinística a partir do nome (pode ser editada depois)."""
    h = hashlib.sha256(nome.encode()).digest()
    ap = {
        "pele": PELES[h[0] % len(PELES)],
        "cabelo": CABELOS[h[1] % len(CABELOS)],
        "estilo": ESTILOS[h[2] % len(ESTILOS)],
        "camisa": CAMISAS[h[3] % len(CAMISAS)],
        "oculos": h[4] % 4 == 0,
        "barba": h[5] % 5 == 0,
    }
    ap.update(forcar)
    return ap
