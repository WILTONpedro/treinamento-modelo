"""Triagem de currículos — porte da API de triagem da Baly para o SDK atual do Gemini.

O prompt, as categorias e as regras de classificação são os mesmos da API original.
Mudanças: SDK `google-genai` (o `google-generativeai` foi descontinuado), tipo MIME
inferido pela extensão, nova tentativa com espera crescente em caso de limite (429)
e nenhum dado pessoal nos logs.
"""
import io
import json
import logging
import mimetypes
import os
import re
import time
import unicodedata
from datetime import datetime

from pydantic import BaseModel

from . import config
from .catalogo import CATEGORIAS_DISPONIVEIS

log = logging.getLogger("baly.triagem")

EXTENSOES = {".pdf", ".docx", ".txt", ".jpg", ".jpeg", ".png", ".webp"}


class CurriculoSchema(BaseModel):
    nome: str
    email: str
    numero: str
    cidade: str
    setor: str
    confianca: str
    anos_experiencia: int
    resumo: str


# --------------------------------------------------------------------------- utilidades de arquivo
def mime_de(nome: str, informado: str | None = None) -> str:
    adivinhado = mimetypes.guess_type(nome)[0]
    if informado and informado not in ("application/octet-stream", "binary/octet-stream"):
        return informado
    return adivinhado or "application/octet-stream"


def extrair_texto(conteudo: bytes, nome: str) -> str | None:
    """Texto do currículo para exibição e para destacar as evidências da análise."""
    ext = os.path.splitext(nome)[1].lower()
    try:
        if ext == ".txt":
            return conteudo.decode("utf-8", errors="ignore").strip() or None
        if ext == ".docx":
            import docx
            d = docx.Document(io.BytesIO(conteudo))
            partes = [p.text for p in d.paragraphs]
            for t in d.tables:
                for linha in t.rows:
                    partes.append(" | ".join(c.text for c in linha.cells))
            return "\n".join(partes).strip() or None
        if ext == ".pdf":
            from pypdf import PdfReader
            r = PdfReader(io.BytesIO(conteudo))
            txt = "\n".join((p.extract_text() or "") for p in r.pages)
            return txt.strip() or None  # vazio = PDF escaneado (a IA ainda lê o arquivo)
    except Exception as e:
        log.warning("Falha ao extrair texto (%s): %s", ext, type(e).__name__)
    return None


def preparar_entrada_gemini(file_bytes, filename, mime_type):
    # FILTRAGEM POR NOME DE ARQUIVO
    nome_lower = filename.lower()
    if "carta" in nome_lower and "apresenta" in nome_lower:
        return {"ignorar": True, "motivo": "Carta de Apresentação (Filtro Nome)"}
    if "cover" in nome_lower and "letter" in nome_lower:
        return {"ignorar": True, "motivo": "Cover Letter (Filtro Nome)"}

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".docx":
        return extrair_texto(file_bytes, filename)
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    elif ext in [".pdf", ".jpg", ".jpeg", ".png", ".webp"]:
        return {"mime_type": mime_de(filename, mime_type), "data": file_bytes}
    return None


def limpar_json(text):
    """Tenta extrair e limpar JSON de uma string."""
    text = text.strip()
    if "```" in text:
        text = re.sub(r"```(?:json)?(.*?)```", r"\1", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return text


def montar_prompt():
    ano_atual = datetime.now().year
    ano_limite_menor = ano_atual - 18
    return f"""
    Você é um Recrutador Sênior da Baly. Sua tarefa é analisar um currículo e categorizá-lo corretamente em uma das pastas disponíveis.
    Data de Hoje: {datetime.now().strftime('%d/%m/%Y')} (Considere esta data para cálculos de idade).
    Estamos no ano de {ano_atual}.

    <categorias_permitidas>
    {json.dumps(CATEGORIAS_DISPONIVEIS, ensure_ascii=False)}
    </categorias_permitidas>

    <instrucoes_extracao>
    1. **Nome**: Identifique o nome completo do candidato (geralmente no topo).
    2. **Contato**: Extraia o telefone (campo 'numero') e email.
    3. **Localização (NOVO)**: Identifique a **CIDADE e ESTADO** de residência atual do candidato.
       - Procure no cabeçalho ou dados pessoais.
       - Formato desejado: "Cidade - UF" (ex: "Ribeirão Preto - SP").
       - Se não tiver o nome da cidade, apenas a Sigla, retorne apenas a sigla por extenso.
       - Se não encontrar, retorne "Não informado".
       - Se o candidato disser "Disponível para mudança para X", registre a cidade onde ele mora HOJE.
    </instrucoes_extracao>

    <regras_categorizacao>
    1. **Hierarquia e Liderança (MUITO IMPORTANTE)**:
       - **ÚLTIMA EXPERIÊNCIA É A QUE CONTA**: Se o candidato atuava como COORDENADOR, GERENTE ou SUPERVISOR no emprego mais recente, ele DEVE ir para a pasta correspondente (GERENTE [AREA] ou SUPERVISOR [AREA]).
       - **NÃO REBAIXE O CANDIDATO**: Um Gerente não deve ser colocado como Analista/Assistente.
       - **PROIBIDO** colocar líderes na pasta "ADMINISTRATIVO".

    2. **Cartas de Apresentação (IGNORAR)**:
       - Se o arquivo for apenas uma Carta de Apresentação (Cover Letter) sem os detalhes completos de um currículo, classifique como **ARQUIVO_INVALIDO**.
       - Não tente "adivinhar" a área baseada apenas na carta. Se não for um CV completo -> LIXO.

    3. **REGRA CRÍTICA: Jovem Aprendiz (CÁLCULO DE IDADE)**:
       - O candidato DEVE ter MENOS de 18 anos completos hoje.
       - **Matemática**: Para ter menos de 18 anos, o candidato deve ter nascido DEPOIS de {ano_limite_menor}.
       - **Exemplos**:
         - Nascidos em {ano_limite_menor - 2} ({ano_atual - 20} anos) -> NÃO É JOVEM APRENDIZ.
         - Nascidos em {ano_limite_menor - 1} ({ano_atual - 19} anos) -> NÃO É JOVEM APRENDIZ.
         - Nascidos em 2006 ou 2007 -> NÃO É JOVEM APRENDIZ (são maiores de idade).
       - Se o candidato tiver nascido em {ano_limite_menor} ou antes, classifique em OUTRA área..

    4. **Operacional vs Especialista**:
       - **Empilhadeira**: Só com curso/NR-11 explícito. Senão -> LOGÍSTICA.
       - **Motorista**: Só com CNH C, D ou E. CNH B/Moto -> LOGÍSTICA ou OUTROS.
       - **Vigia**: Só com curso de vigilante/reciclagem.
       - **Área Industrial**: Exclusiva para Segurança do Trabalho/SESMT. Operadores de máquina -> PRODUÇÃO.

    5. **Comercial e Vendas**:
       - Vendedor, Balconista, Consultor -> VENDAS.
       - Representante -> VENDAS ou COMERCIAL.
       - Gerente/Supervisor de Vendas -> GERENTE VENDAS / SUPERVISOR DE VENDAS.
       - Promotor de Vendas -> Apenas se tiver experiência prévia como promotor.

    6. **Outras Regras Específicas**:
       - **TI**: Suporte, Infra, Dev, Redes.
       - **PCP**: Planejamento e Controle de Produção.
       - **PCD**: Apenas se mencionar explicitamente Deficiência/CID.
       - **Comércio Exterior**: Importação/Exportação.

    7. **PRODUÇÃO**
        -Coloque todos que tiveram alguma experiência mais braçal como prevalente no currículo, não precisa ser especificamente como AUXILIAR DE PRODUÇÃO, mas chapeiro, zelador, etc...

    8. **Qualidade**
        -Pessoas que trabalham/tem cursos com enfermagem, farmácia ou áreas da saúde e qualidade.

    9. **GERENTE DE RH**
        -Coloque todos que você achar que se encaixaria como GERENTE ADMINISTRATIVO.

    10. **Arquivos Inválidos (LIXO)**:
       - Se for foto aleatória, boleto, carta de apresentação isolada ou o próprio anúncio da vaga ("Anti-Espelho") -> setor: "ARQUIVO_INVALIDO".
       - Apresentações (PPT), cartas soltas -> Ignorar.
    </regras_categorizacao>

    <saida_esperada>
    Responda EXCLUSIVAMENTE com um objeto JSON seguindo este schema:
    {{
        "nome": "Nome Sobrenome",
        "email": "email@exemplo.com",
        "numero": "Telefone",
        "cidade": "Cidade - UF",
        "setor": "CATEGORIA_ESCOLHIDA",
        "confianca": "ALTA/MEDIA/BAIXA",
        "anos_experiencia": 0,
        "resumo": "Breve justificativa"
    }}
    Se não se encaixar em nenhuma categoria específica, use "OUTROS".
    </saida_esperada>
    """


def _erro(resumo, nome="Desconhecido"):
    return {"setor": "OUTROS", "confianca": "ERRO_IA", "resumo": resumo, "nome": nome, "email": "", "numero": ""}


def analisar_com_gemini(conteudo_processado):
    if isinstance(conteudo_processado, dict) and conteudo_processado.get("ignorar"):
        return {"setor": "ARQUIVO_INVALIDO", "confianca": "ALTA", "motivo": conteudo_processado.get("motivo"),
                "nome": "Ignorado", "email": "", "numero": "", "anos_experiencia": 0,
                "resumo": conteudo_processado.get("motivo")}
    if not conteudo_processado:
        return {"setor": "ARQUIVO_INVALIDO", "confianca": "BAIXA", "motivo": "Vazio", "resumo": "Arquivo vazio ou formato não suportado"}

    from google.genai import types
    from .ia import cliente_gemini, sem_bloqueio

    if isinstance(conteudo_processado, dict):
        entrada = types.Part.from_bytes(data=conteudo_processado["data"], mime_type=conteudo_processado["mime_type"])
    else:
        entrada = conteudo_processado

    for tentativa in range(3):
        try:
            response = cliente_gemini().models.generate_content(
                model=config.GEMINI_MODELO_TRIAGEM,
                contents=[montar_prompt(), entrada],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CurriculoSchema,
                    temperature=0.2,
                    safety_settings=sem_bloqueio(),
                ),
            )
            if not response.candidates:
                log.warning("Gemini não retornou candidatos (bloqueio de segurança)")
                return _erro("Bloqueado pelo Google (Dados Sensíveis)", "Candidato")
            try:
                dados = json.loads(response.text)
            except (json.JSONDecodeError, TypeError):
                dados = json.loads(limpar_json(response.text or ""))
            if isinstance(dados, list):
                dados = dados[0]
            if dados.get("setor") not in CATEGORIAS_DISPONIVEIS and dados.get("setor") != "ARQUIVO_INVALIDO":
                dados["setor"] = "OUTROS"
            return dados
        except Exception as e:
            limite = getattr(e, "code", None) == 429 or "429" in str(e)
            log.warning("Triagem: tentativa %d/3 falhou (%s)", tentativa + 1, "limite 429" if limite else type(e).__name__)
            if tentativa == 2:
                return _erro(f"Falha na IA: {type(e).__name__}")
            time.sleep(5 * (tentativa + 1) if limite else 2)
    return _erro("Timeout")


# --------------------------------------------------------------------------- modo demonstração (sem IA externa)
def _sem_acento(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


_PALAVRAS = [
    ("PCD", ["deficiencia", "pcd", "cid "]),
    ("GERENTE VENDAS", ["gerente de vendas", "gerente comercial"]),
    ("SUPERVISOR DE VENDAS", ["supervisor de vendas"]),
    ("EMPILHADEIRA", ["nr-11", "nr11", "operador de empilhadeira"]),
    ("MOTORISTA", [r"cnh (categoria )?[cde]\b", "motorista"]),
    ("TI", ["suporte tecnico", "desenvolvedor", "programador", "redes de computadores", "infraestrutura de ti"]),
    ("PROMOTOR DE VENDAS", ["promotor de vendas", "promotora de vendas", "repositor"]),
    ("MARKETING", ["marketing", "midias sociais", "social media"]),
    ("VENDAS", ["vendedor", "vendedora", "vendas externas", "representante comercial", "consultor de vendas", "balconista"]),
    ("LOGÍSTICA", ["logistica", "conferente", "estoque", "separacao de pedidos"]),
    ("PRODUÇÃO", ["producao", "operador de maquina", "linha de", "auxiliar de producao", "zelador", "chapeiro"]),
    ("FINANCEIRO", ["contas a pagar", "contas a receber", "financeiro"]),
    ("ADMINISTRATIVO", ["administrativ", "secretari", "recepcionista", "arquivo", "escritorio", "assistente de arquivo"]),
]


def triagem_demo(conteudo_processado, texto: str | None):
    """Classificação por palavras-chave, só para demonstração local (marcada como tal)."""
    if isinstance(conteudo_processado, dict) and conteudo_processado.get("ignorar"):
        return analisar_com_gemini(conteudo_processado)  # mesmo retorno de arquivo ignorado
    if not texto:
        return {"setor": "ARQUIVO_INVALIDO" if not conteudo_processado else "OUTROS", "confianca": "BAIXA",
                "nome": "Candidato", "email": "", "numero": "", "cidade": "Não informado", "anos_experiencia": 0,
                "resumo": "Sem texto legível (no modo demonstração não há leitura de imagem/PDF escaneado)"}
    t = _sem_acento(texto)
    linhas = [l.strip() for l in texto.splitlines() if l.strip()]
    nome = linhas[0][:80] if linhas else "Candidato"
    email = (re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", texto) or [""])[0]
    fone = (re.search(r"\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}", texto) or [""])[0]
    cid = re.search(r"([A-ZÁ-Úa-zá-ú ]{3,40})\s*[-/]\s*([A-Z]{2})\b", texto)
    cidade = f"{cid.group(1).strip()} - {cid.group(2)}" if cid else "Não informado"
    anos = 0
    for a, b in re.findall(r"(19\d{2}|20\d{2})\s*[–-]\s*(19\d{2}|20\d{2}|atual)", texto, flags=re.I):
        fim = datetime.now().year if b.lower() == "atual" else int(b)
        anos += max(0, fim - int(a))
    setor, achou = "OUTROS", None
    for cat, chaves in _PALAVRAS:
        hit = next((c for c in chaves if re.search(c, t)), None)
        if hit:
            setor, achou = cat, hit
            break
    return {"nome": nome, "email": email, "numero": fone, "cidade": cidade, "setor": setor,
            "confianca": "MEDIA" if achou else "BAIXA", "anos_experiencia": min(anos, 50),
            "resumo": (f"[modo demonstração] Termo encontrado: “{achou}”." if achou
                       else "[modo demonstração] Nenhum termo característico encontrado.")}


def triar(conteudo: bytes, nome_arquivo: str, mime: str | None, texto: str | None) -> dict:
    entrada = preparar_entrada_gemini(conteudo, nome_arquivo, mime)
    if config.IA_MODO == "gemini":
        return analisar_com_gemini(entrada)
    return triagem_demo(entrada, texto)
