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

# --- IA ---
import google.generativeai as genai

# ==============================================================================
# ⚙️ CONFIGURAÇÃO
# ==============================================================================

# Tenta pegar a chave do Render, senão usa a local (para testes)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDhoW9xcIr2Rr6hnhWfWr09wr4BfP_JLLwL")
genai.configure(api_key=GEMINI_API_KEY)

# SUA LISTA EXATA DE PASTAS NO DRIVE
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
    "PROMOTOR DE VENDAS",
    "PCP",
    "PRODUÇÃO",
    "PROJETOS",
    "QUALIDADE",
    "RECURSOS HUMANOS",
    "SUPERVISOR DE MERCHANDISING",
    "TI",
    "VENDAS",
    "VIGIA",
    "OUTROS"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# 1. LEITURA DE ARQUIVOS (Em memória RAM)
# ==============================================================================
def extract_text_from_memory(file_bytes, filename):
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
        logger.error(f"Erro leitura: {e}")
    return text

# ==============================================================================
# 2. CÉREBRO (GEMINI COM REGRAS DE NEGÓCIO)
# ==============================================================================
def analisar_com_gemini(texto_curriculo):
    if not texto_curriculo or len(texto_curriculo) < 20:
        return {"setor": "ARQUIVO_INVALIDO", "confianca": "BAIXA", "motivo": "Texto insuficiente"}

    # --- O PROMPT REFINADO ---
    prompt = f"""
    Você é um Recrutador Sênior Especialista da empresa Baly. Sua missão é triar currículos para as pastas corretas.
    
    LISTA DE PASTAS DISPONÍVEIS (Escolha apenas uma):
    {json.dumps(CATEGORIAS_DISPONIVEIS)}

    ⚠️ REGRAS ELIMINATÓRIAS DE NEGÓCIO (IMPORTANTE):
    
    1. **EMPILHADEIRA**: O candidato SÓ vai para esta pasta se citar explicitamente "Curso de Empilhadeira", "Operador de Empilhadeira" ou "NR-11". Se tiver experiência em logística mas não tiver o curso, jogue em "LOGÍSTICA" ou "ALMOXARIFADO".
    
    2. **MOTORISTA**: Exige CNH categorias C, D ou E (Caminhão/Carreta). Se tiver apenas CNH B ou Moto, NÃO coloque aqui (jogue em LOGÍSTICA ou OUTROS).
    
    3. **VIGIA**: Obrigatório ter "Curso de Vigilante", "Reciclagem em dia" ou experiência comprovada em segurança patrimonial.
    
    4. **COMERCIO EXTERIOR**: O candidato deve ter experiência com Importação/Exportação, trâmites aduaneiros ou vendas internacionais.
    
    5. **PCP**: Significa "Planejamento e Controle da Produção". Se o currículo falar de planejar fábrica, cronograma de produção ou ordens de serviço, é aqui.
    
    6. **HIERARQUIA (GERENTES vs OPERACIONAIS)**:
       - Se o cargo for de Liderança Estratégica (Gerente, Head, Diretor), use as pastas que começam com "GERENTE ...".
       - Exemplo: Um "Gerente de Marketing" vai para "GERENTE MARKETING". Um "Analista de Marketing" vai para "MARKETING".
       - Exemplo: "Coordenador" e "Supervisor" têm pastas específicas na lista (ex: SUPERVISOR DE MERCHANDISING). Se não tiver pasta específica de coordenação, jogue na área geral.

    7. **PROMOTOR DE VENDAS**: Só será colocado nesta pasta caso a pessoa já tenha experiência como promotor antes.

    8. **LIXO/INVALIDO**: Se o arquivo for foto de pessoa, print de tela, boleto ou não for um currículo, responda "ARQUIVO_INVALIDO".

    9. **ADMINISTRATIVO**: Essa pasta é para aqueles currículos de pessoas jovens que sejam acima dos 18 e que não tenham nenhuma experiência, mas tenham cursos de áreas importantes.

    10. **PCD**: Se o currículo mencionar explicitamente "PCD", "Deficiência", "CID" ou "Laudo Médico", jogue aqui.

    11. **JOVEM APRENDIZ (Prioridade Máxima)**: Se o candidato tiver MENOS de 18 anos (ex: 14, 15, 16, 17 anos), OBRIGATORIAMENTE jogue nesta pasta, independente da experiência.

    12. **AREA INDUSTRIAL**: ATENÇÃO! Nesta empresa, esta pasta é EXCLUSIVA para "Técnico em Segurança do Trabalho", "Engenheiro de Segurança" ou "SESMT". Não jogue operadores de máquina aqui (jogue em PRODUÇÃO).

    13. **MECANICA INDUSTRIAL**: Aqui não são só colocados currículos de mecânicos, mas de tudo que envolve essa área, como eletricistas

    TEXTO DO CURRÍCULO:
    {texto_curriculo[:9000]}

    Responda APENAS um JSON neste formato:
    {{
        "setor": "NOME_DA_PASTA_ESCOLHIDA",
        "confianca": "ALTA",
        "anos_experiencia": 0,
        "resumo": "Explique em 1 frase por que escolheu essa pasta baseado nas regras acima"
    }}
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
    except Exception as e:
        logger.error(f"Erro Gemini: {e}")
        return {"setor": "OUTROS", "confianca": "ERRO_IA", "resumo": str(e)}

# ==============================================================================
# 3. API
# ==============================================================================
app = FastAPI(title="API Triagem Inteligente")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/triagem")
def triar_curriculo(file: UploadFile = File(...)):
    # Validação de Tamanho (5MB)
    file.file.seek(0, 2)
    if file.file.tell() > 5 * 1024 * 1024:
        return {"status": "erro", "mensagem": "Arquivo > 5MB"}
    file.file.seek(0)

    try:
        content = file.file.read()
        raw_text = extract_text_from_memory(content, file.filename)
        
        # Análise IA
        analise = analisar_com_gemini(raw_text)
        
        setor = analise.get("setor", "OUTROS")
        
        # Tratamento para o AppScript
        conf_map = {"ALTA": 0.98, "MEDIA": 0.75, "BAIXA": 0.45, "ERRO_IA": 0.0}
        conf_val = conf_map.get(analise.get("confianca"), 0.5)

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
        logger.error(f"Erro fatal: {e}")
        return {"status": "erro", "mensagem": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
