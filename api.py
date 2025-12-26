import os
import io
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import docx
import google.generativeai as genai
from typing_extensions import TypedDict
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- CONFIGURAÇÃO ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("api")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
NOME_MODELO_GEMINI = "gemini-2.5-flash"

class CurriculoSchema(TypedDict):
    nome: str
    email: str
    numero: str
    setor: str
    confianca: str
    anos_experiencia: int
    resumo: str

# LISTA DE CATEGORIAS
CATEGORIAS_DISPONIVEIS = [
    "ADMINISTRATIVO", "ALMOXARIFADO", "AREA INDUSTRIAL", "COMERCIAL", "COMERCIO EXTERIOR",
    "COMPRAS", "CONTABILIDADE", "COORDENADOR DE EXPEDIÇÃO", "COORDENADOR DE MERCHANDISING",
    "EMPILHADEIRA", "EVENTOS", "FINANCEIRO", "GERENTE COMERCIAL", "GERENTE FINANCEIRO",
    "GERENTE DE GRANDES CONTAS", "GERENTE LOGISTICA", "GERENTE MARKETING", "GERENTE PRODUÇÃO",
    "GERENTE QUALIDADE", "GERENTE DE RH", "GERENTE VENDAS", "HIGIENIZAÇÃO", "JOVEM APRENDIZ",
    "KEY ACCOUNT", "LIDER DE PRODUÇÃO", "LOGÍSTICA", "MARKETING", "MECANICA INDUSTRIAL",
    "MERCHANDISING", "MOTORISTA", "PCD", "PCP", "PRODUÇÃO", "PROJETOS", "PROMOTOR DE VENDAS",
    "QUALIDADE", "RECURSOS HUMANOS", "SUPERVISOR DE MERCHANDISING", "SUPERVISOR DE VENDAS", "TI", "VENDAS", "VIGIA", "OUTROS"
]

def preparar_entrada_gemini(file_bytes, filename, mime_type):
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".docx":
        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            return "\n".join([p.text for p in doc.paragraphs])
        except: return None
    elif ext == ".txt":
        return file_bytes.decode("utf-8", errors="ignore")
    elif ext in [".pdf", ".jpg", ".jpeg", ".png", ".webp"]:
        return {"mime_type": mime_type, "data": file_bytes}
    return None

def limpar_json(text):
    """Tenta extrair e limpar JSON de uma string."""
    text = text.strip()
    
    # Remove blocos de código markdown
    if "```" in text:
        text = re.sub(r"```(?:json)?(.*?)```", r"\1", text, flags=re.DOTALL).strip()
    
    # Tenta encontrar o objeto JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
        
    return text

def analisar_com_gemini(conteudo_processado):
    if not conteudo_processado:
        return {"setor": "ARQUIVO_INVALIDO", "confianca": "BAIXA", "motivo": "Vazio"}

    prompt = f"""
    Você é um Recrutador Sênior da Baly. Sua tarefa é analisar um currículo e categorizá-lo corretamente em uma das pastas disponíveis.

    <categorias_permitidas>
    {json.dumps(CATEGORIAS_DISPONIVEIS, ensure_ascii=False)}
    </categorias_permitidas>

    <instrucoes_extracao>
    1. **Nome**: Identifique o nome completo do candidato (geralmente no topo).
    2. **Contato**: Extraia o telefone (campo 'numero') e email.
    </instrucoes_extracao>

    <regras_categorizacao>
    1. **Hierarquia e Liderança**:
       - Candidatos com experiência em Gestão, Liderança, Coordenação ou MBA devem ir para pastas de GERENTE [AREA] ou SUPERVISOR [AREA].
       - **PROIBIDO** colocar líderes na pasta "ADMINISTRATIVO".
       - Respeite a última experiência. Se era Coordenador, não rebaixe.

    2. **Jovem Aprendiz**:
       - Apenas se < 18 anos E ensino médio em curso ou concluído recentemente.
       - Se tiver ensino superior ou > 18 anos, NÃO é Jovem Aprendiz.

    3. **Operacional vs Especialista**:
       - **Empilhadeira**: Só com curso/NR-11 explícito. Senão -> LOGÍSTICA.
       - **Motorista**: Só com CNH C, D ou E. CNH B/Moto -> LOGÍSTICA ou OUTROS.
       - **Vigia**: Só com curso de vigilante/reciclagem.
       - **Área Industrial**: Exclusiva para Segurança do Trabalho/SESMT. Operadores de máquina -> PRODUÇÃO.
    
    4. **Comercial e Vendas**:
       - Vendedor, Balconista, Consultor -> VENDAS.
       - Representante -> VENDAS ou COMERCIAL.
       - Gerente/Supervisor de Vendas -> GERENTE VENDAS / SUPERVISOR DE VENDAS.
       - Promotor de Vendas -> Apenas se tiver experiência prévia como promotor.
       
    5. **Outras Regras Específicas**:
       - **TI**: Suporte, Infra, Dev, Redes.
       - **PCP**: Planejamento e Controle de Produção.
       - **PCD**: Apenas se mencionar explicitamente Deficiência/CID.
       - **Comércio Exterior**: Importação/Exportação.

    6. **Arquivos Inválidos (LIXO)**:
       - Se for foto aleatória, boleto, ou o próprio anúncio da vaga ("Anti-Espelho") -> setor: "ARQUIVO_INVALIDO".
       - Apresentações (PPT), cartas soltas -> Ignorar.
    </regras_categorizacao>

    <saida_esperada>
    Responda EXCLUSIVAMENTE com um objeto JSON seguindo este schema:
    {{
        "nome": "Nome Sobrenome",
        "email": "email@exemplo.com",
        "numero": "Telefone",
        "setor": "CATEGORIA_ESCOLHIDA",
        "confianca": "ALTA/MEDIA/BAIXA",
        "anos_experiencia": 0,
        "resumo": "Breve justificativa"
    }}
    Se não se encaixar em nenhuma categoria específica, use "OUTROS".
    </saida_esperada>
    """

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    for tentativa in range(3):
        try:
            model = genai.GenerativeModel(NOME_MODELO_GEMINI)
            response = model.generate_content(
                [prompt, conteudo_processado], 
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": CurriculoSchema,
                    "temperature": 0.2
                },
                safety_settings=safety_settings
            )
            
            if not response.candidates:
                logger.warning("⚠️ Bloqueio de Segurança Gemini (Mesmo com filtro desligado)")
                return {"setor": "OUTROS", "confianca": "ERRO_IA", "resumo": "Bloqueado pelo Google (Dados Sensíveis)", "nome": "Candidato", "email":"", "numero":""}

            try:
                # Tentativa direta de parsing
                dados = json.loads(response.text)
            except json.JSONDecodeError:
                # Tentativa com limpeza
                try:
                    texto_limpo = limpar_json(response.text)
                    dados = json.loads(texto_limpo)
                except Exception as e:
                    logger.error(f"Erro ao fazer parse do JSON: {e} | Raw: {response.text}")
                    raise e

            if isinstance(dados, list): dados = dados[0]

            if dados.get("setor") not in CATEGORIAS_DISPONIVEIS:
                if dados.get("setor") != "ARQUIVO_INVALIDO": # Mantém ARQUIVO_INVALIDO se for o caso
                    dados["setor"] = "OUTROS"

            return dados

        except Exception as e:
            if "429" in str(e):
                logger.warning(f"Rate limit (429). Tentativa {tentativa+1}/3. Aguardando...")
                time.sleep(5)
            else:
                logger.error(f"Erro geral na análise: {e}")
                # Se for a última tentativa, retorna erro
                if tentativa == 2:
                    return {"setor": "OUTROS", "confianca": "ERRO_IA", "resumo": str(e), "nome": "Desconhecido", "email":"", "numero":""}
    
    return {"setor": "OUTROS", "confianca": "ERRO_IA", "resumo": "Timeout", "nome": "Desconhecido", "email":"", "numero":""}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 SERVIDOR INICIADO")
    yield

app = FastAPI(title="API Triagem", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/triagem")
async def triar_curriculo(file: UploadFile = File(...)):
    try:
        content = await file.read()
        dados = preparar_entrada_gemini(content, file.filename, file.content_type)
        analise = analisar_com_gemini(dados)
        
        # Garante que campos existam e tenham valores padrão
        nome = analise.get("nome") or "Candidato"
        if len(nome) < 3: nome = "Candidato"
        
        logger.info(f"🏁 {file.filename} -> {analise.get('setor')} ({nome})")

        return {
            "arquivo": file.filename,
            "nome_identificado": nome,
            "setor_sugerido": analise.get("setor", "OUTROS"),
            "confianca": analise.get("confianca", "BAIXA"),
            "detalhes": analise
        }
    except Exception as e:
        logger.error(f"Erro rota: {e}")
        return {"status": "erro", "mensagem": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
