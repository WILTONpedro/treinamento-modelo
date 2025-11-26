import os
import io
import json
import logging
import re
from contextlib import asynccontextmanager

# --- SERVER ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- PROCESSAMENTO ---
import pdfplumber
import docx
from PIL import Image
import pytesseract

# --- IA (GOOGLE GEMINI) ---
import google.generativeai as genai

# ==============================================================================
# ⚙️ CONFIGURAÇÃO E LOGS
# ==============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("uvicorn")

# Pega a chave do Render
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)

# Modelo atualizado
NOME_MODELO_GEMINI = "gemini-2.0-flash"

# LISTA EXATA DE PASTAS DO DRIVE
CATEGORIAS_DISPONIVEIS = [
    "ADMINISTRITIVO",
    "ALMOXARIFADO",
    "AREA INDUSTRIAL",
    "COMERCIAL",
    "COMERCIO EXTERIOR",
    "COMPRAS",
    "CONTABILIDADE",
    "COORDENADOR DE EXPEDIÇÃO",
    "COORDENADOR DE MERCHANDISING",
    "EMPILHADEIRA",
    "EVENTOS",
    "FINANCEIRO",
    "GERENTE COMERCIAL",
    "GERENTE FINANCEIRO",
    "GERENTE GRANDES CONTAS",
    "GERENTE LOGISTICA",
    "GERENTE MARKETING",
    "GERENTE PRODUÇÃO",
    "GERENTE QUALIDADE",
    "GERENTE DE RH",
    "GERENTE VENDAS",
    "HIGIENIZAÇÃO",
    "JOVEM APRENDIZ",
    "KEY ACCOUNT",
    "LIDER DE PRODUÇÃO",
    "LOGÍSTICA",
    "MARKETING",
    "MECANICA INDUSTRIAL",
    "MERCHANDISING",
    "MOTORISTA",
    "PCD",
    "PCP",
    "PRODUÇÃO",
    "PROJETOS",
    "PROMOTOR DE VENDAS",
    "QUALIDADE",
    "RECURSOS HUMANOS",
    "SUPERVISOR DE MERCHANDISING",
    "TI",
    "VENDAS",
    "VIGIA",
    "OUTROS"
]

# ==============================================================================
# 1. LEITURA DE ARQUIVOS (MEMÓRIA RAM)
# ==============================================================================
def extract_text_from_memory(file_bytes, filename):
    """Extrai texto de PDF, DOCX, TXT ou Imagens diretamente da memória."""
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    file_stream = io.BytesIO(file_bytes)
    
    try:
        if ext == ".pdf":
            with pdfplumber.open(file_stream) as pdf:
                text = " ".join([p.extract_text() or "" for p in pdf.pages])
        elif ext == ".docx":
            doc = docx.Document(file_stream)
            text = " ".join([p.text for p in doc.paragraphs])
        elif ext in [".jpg", ".png", ".jpeg"]:
            img = Image.open(file_stream)
            text = pytesseract.image_to_string(img, lang="por")
        elif ext == ".txt":
            text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Erro leitura arquivo ({filename}): {e}")
    return text

# ==============================================================================
# 2. CÉREBRO (GEMINI COM REGRAS DE NEGÓCIO)
# ==============================================================================
def analisar_com_gemini(texto_curriculo):
    # Validação básica de conteúdo
    if not texto_curriculo or len(texto_curriculo) < 20:
        return {"setor": "ARQUIVO_INVALIDO", "confianca": "BAIXA", "motivo": "Texto insuficiente/Arquivo vazio"}

    # --- O PROMPT REFINADO ---
    prompt = f"""
    Você é um Recrutador Sênior Especialista da empresa Baly. Sua missão é triar currículos para as pastas corretas.
    
    LISTA DE PASTAS DISPONÍVEIS (Escolha apenas uma):
    {json.dumps(CATEGORIAS_DISPONIVEIS)}

    ⚠️ REGRAS ELIMINATÓRIAS DE NEGÓCIO (IMPORTANTE):
    
    1. **HIERARQUIA (GERENTES vs OPERACIONAIS)IMPORTANTE!!!**:
       - Se o cargo for de Liderança Estratégica (Gerente, Head, Diretor), use as pastas que começam com "GERENTE ...".
       - Exemplo: Um "Gerente de Marketing" vai para "GERENTE MARKETING". Um "Analista de Marketing" vai para "MARKETING".
       - Exemplo: "Coordenador" e "Supervisor" têm pastas específicas na lista (ex: SUPERVISOR DE MERCHANDISING). Se não tiver pasta específica de coordenação, jogue na área geral.

    2. **HIERARQUIA NA EXPERIÊNCIA**: Levar a serio o criterio de ultima experiência do colaborador, Exemplo: Ele tem experiência como supervisor de merchandising mas também como coordenador(Cargo acima) não à motivos para colocar ele em um cargo abaixo.

    3. **GERENTE DE GRANDES CONTAS**:Essa pasta é específica, então ela é uma vaga para o trade marketing e vai fazer uma ponte com o comercial cuidando de nossas grandes redes. Então o rapaz tem que já ter experiência com esse assunto.

    4. **KEY ACCOUNT**: Aqui nesta empresa, essa pasta é especifica para o pessoal mais comercial focado em VENDAS para as grandes redes
    
    5. **EMPILHADEIRA**: O candidato SÓ vai para esta pasta se citar explicitamente "Curso de Empilhadeira", "Operador de Empilhadeira" ou "NR-11". Se tiver experiência em logística mas não tiver o curso, jogue em "LOGÍSTICA" ou "ALMOXARIFADO".
    
    6. **MOTORISTA**: Exige CNH categorias C, D ou E (Caminhão/Carreta). Se tiver apenas CNH B ou Moto, NÃO coloque aqui (jogue em LOGÍSTICA ou OUTROS).
    
    7. **VIGIA**: Obrigatório ter "Curso de Vigilante", "Reciclagem em dia" ou experiência comprovada em segurança patrimonial.
    
    8. **COMERCIO EXTERIOR**: O candidato deve ter experiência com Importação/Exportação, trâmites aduaneiros ou vendas internacionais.
    
    9. **PCP**: Significa "Planejamento e Controle da Produção". Se o currículo falar de planejar fábrica, cronograma de produção ou ordens de serviço, é aqui.
    
    10. **PROMOTOR DE VENDAS**: Só será colocado nesta pasta caso a pessoa já tenha experiência como promotor antes.

    11. **LIXO/INVALIDO**: Se o arquivo for foto de pessoa, print de tela, boleto ou não for um currículo, responda "ARQUIVO_INVALIDO".

    12. **ADMINISTRATIVO**: Essa pasta é para aqueles currículos de pessoas jovens que sejam acima dos 18 e que não tenham nenhuma experiência, mas tenham cursos de áreas importantes.

    13. **PCD**: Se o currículo mencionar explicitamente "PCD", "Deficiência", "CID" ou "Laudo Médico", jogue aqui.

    14. **JOVEM APRENDIZ (Prioridade Máxima)**: Se o candidato tiver MENOS de 18 anos (ex: 14, 15, 16, 17 anos), OBRIGATORIAMENTE jogue nesta pasta, independente da experiência.

    15. **AREA INDUSTRIAL**: ATENÇÃO! Nesta empresa, esta pasta é EXCLUSIVA para "Técnico em Segurança do Trabalho", "Engenheiro de Segurança" ou "SESMT". Não jogue operadores de máquina aqui (jogue em PRODUÇÃO).

    16. **MECANICA INDUSTRIAL**: Aqui não são só colocados currículos de mecânicos, mas de tudo que envolve essa área, como eletricistas

    17. **QUALIDADE**: A vaga aqui pode ser alocada o pessoal que tenha experiência ou tenha feito alguma especialização mais laboral, como biomedicina e áreas da saúde.

    18. **ANTI-ESPELHO (O PRÓPRIO ANÚNCIO)**: 
        - Se o texto extraído contiver instruções de como se candidatar (ex: "Como participar", "Envie seu currículo para", "Vem ser time amarelo", "WhatsApp para envio"), isso NÃO É UM CURRÍCULO, é a imagem da vaga.
        - Neste caso, responda OBRIGATORIAMENTE: "ARQUIVO_INVALIDO".

    TEXTO DO CURRÍCULO:
    {texto_curriculo[:9000]}

    Responda APENAS um JSON neste formato:
    {{
        "setor": "NOME_DA_PASTA_ESCOLHIDA",
        "confianca": "ALTA",
        "anos_experiencia": 0,
        "resumo": "Explique em 1 frase por que escolheu essa pasta"
    }}
    """

    try:
        model = genai.GenerativeModel(NOME_MODELO_GEMINI)
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        logger.error(f"Erro na chamada do Gemini: {e}")
        return {"setor": "OUTROS", "confianca": "ERRO_IA", "resumo": str(e)}

# ==============================================================================
# 3. CICLO DE VIDA (DIAGNÓSTICO DE INICIALIZAÇÃO)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executa ao ligar a API. Faz diagnóstico dos modelos disponíveis."""
    
    logger.info("🚀 INICIANDO SERVIDOR... VERIFICANDO MODELOS GOOGLE...")
    
    try:
        if not GEMINI_API_KEY:
            logger.warning("⚠️ AVISO CRÍTICO: Variável GEMINI_API_KEY não encontrada!")
        
        # Lista modelos disponíveis para confirmar que a chave funciona e o modelo existe
        modelos_disponiveis = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                modelos_disponiveis.append(m.name)
                logger.info(f"   ✅ Modelo disponível: {m.name}")
        
        # Verifica se o modelo escolhido está na lista
        target_model = f"models/{NOME_MODELO_GEMINI}"
        if target_model in modelos_disponiveis:
            logger.info(f"🎉 SUCESSO: O modelo '{NOME_MODELO_GEMINI}' foi encontrado e está pronto!")
        else:
            logger.warning(f"⚠️ ATENÇÃO: O modelo '{NOME_MODELO_GEMINI}' não apareceu na lista padrão. Pode dar erro 404.")
            
    except Exception as e:
        logger.error(f"❌ ERRO AO LISTAR MODELOS (Verifique sua API KEY): {e}")

    yield # A API roda aqui
    logger.info("🛑 Desligando servidor...")

# ==============================================================================
# 4. APLICAÇÃO API
# ==============================================================================

app = FastAPI(title="API Triagem Inteligente", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/triagem")
def triar_curriculo(file: UploadFile = File(...)):
    # 1. Validação de Tamanho (Máx 5MB para não travar o Render Free)
    file.file.seek(0, 2)
    tamanho = file.file.tell()
    file.file.seek(0)
    
    if tamanho > 5 * 1024 * 1024:
        return {"status": "erro", "mensagem": "Arquivo muito grande (>5MB)"}

    try:
        # 2. Leitura
        content = file.file.read()
        raw_text = extract_text_from_memory(content, file.filename)
        
        # 3. Análise IA
        analise = analisar_com_gemini(raw_text)
        
        setor = analise.get("setor", "OUTROS")
        
        # 4. Mapeamento de confiança para o AppScript entender
        conf_map = {"ALTA": 0.98, "MEDIA": 0.75, "BAIXA": 0.45, "ERRO_IA": 0.0}
        conf_val = conf_map.get(analise.get("confianca"), 0.5)

        logger.info(f"Processado: {file.filename} -> {setor} ({analise.get('resumo')})")

        return {
            "arquivo": file.filename,
            "setor_sugerido": setor,
            "confianca": f"{conf_val:.2%}",
            "detalhes": {
                "tempo_estimado": analise.get("anos_experiencia", 0),
                "tem_cursos": True,
                "motivo_rejeicao": analise.get("resumo")
            }
        }

    except Exception as e:
        logger.error(f"Erro fatal no processamento: {e}")
        return {"status": "erro", "mensagem": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
