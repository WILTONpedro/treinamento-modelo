import os
import io
import json
import logging
import re
import sys
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pdfplumber
import docx
from PIL import Image
import pytesseract
import google.generativeai as genai
from notion_client import Client

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("api")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
WEBHOOK_GOOGLE_URL = os.environ.get("WEBHOOK_GOOGLE_URL", "")

genai.configure(api_key=GEMINI_API_KEY)
NOME_MODELO_GEMINI = "gemini-2.0-flash"

http_session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
http_session.mount('https://', HTTPAdapter(max_retries=retries))

CATEGORIAS_DISPONIVEIS = [
    "ADMINISTRITIVO", "ALMOXARIFADO", "AREA INDUSTRIAL", "COMERCIAL", "COMERCIO EXTERIOR",
    "COMPRAS", "CONTABILIDADE", "COORDENADOR DE EXPEDIÇÃO", "COORDENADOR DE MERCHANDISING",
    "EMPILHADEIRA", "EVENTOS", "FINANCEIRO", "GERENTE COMERCIAL", "GERENTE FINANCEIRO",
    "GERENTE GRANDES CONTAS", "GERENTE LOGISTICA", "GERENTE MARKETING", "GERENTE PRODUÇÃO",
    "GERENTE QUALIDADE", "GERENTE DE RH", "GERENTE VENDAS", "HIGIENIZAÇÃO", "JOVEM APRENDIZ",
    "KEY ACCOUNT", "LIDER DE PRODUÇÃO", "LOGÍSTICA", "MARKETING", "MECANICA INDUSTRIAL",
    "MERCHANDISING", "MOTORISTA", "PCD", "PCP", "PRODUÇÃO", "PROJETOS", "PROMOTOR DE VENDAS",
    "QUALIDADE", "RECURSOS HUMANOS", "SUPERVISOR DE MERCHANDISING", "TI", "VENDAS", "VIGIA", "OUTROS"
]
def sanitize_filename(filename):
    """Remove caracteres perigosos e emojis do nome do arquivo."""
    clean = re.sub(r'[^a-zA-Z0-9 \-\.]', '', filename)
    return clean.strip() or "arquivo_sem_nome"

def extract_text_from_memory(file_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()
    text = ""
    file_stream = io.BytesIO(file_bytes)
    
    try:
        if ext == ".pdf":
            with pdfplumber.open(file_stream) as pdf:
                pages = pdf.pages[:5]
                text = "\n".join([p.extract_text() or "" for p in pages])
        elif ext == ".docx":
            doc = docx.Document(file_stream)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext in [".jpg", ".png", ".jpeg", ".tiff"]:
            img = Image.open(file_stream)
            if img.width > 2000:
                img.thumbnail((2000, 2000))
            text = pytesseract.image_to_string(img, lang="por")
        elif ext == ".txt":
            text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Erro leitura arquivo ({filename}): {e}")
        return ""
    
    return text.replace("\x00", "")

def analisar_com_gemini(texto_curriculo):
    if not texto_curriculo or len(texto_curriculo.strip()) < 20:
        return {"setor": "ARQUIVO_INVALIDO", "confianca": "BAIXA", "motivo": "Texto insuficiente"}
        
    prompt = f"""
    Você é um Recrutador Sênior Especialista da empresa Baly. Sua missão é triar currículos para as pastas corretas.
    
    TAREFA 2 (CRUCIAL): O texto abaixo veio de uma extração bruta do LinkedIn ou PDF e está sujo.
    Você deve REESCREVER e ESTRUTURAR as informações em formato de Currículo Profissional Limpo.
    - Remova: Botões ("Conectar", "Enviar mensagem"), propagandas, menus, "Pessoas também viram", textos de interface.
    - Mantenha: Nome, Resumo, Experiência (Empresas, Cargos, Datas), Formação, Idiomas e Competências.
    - Formato: Texto corrido bem organizado (Markdown simples).

    ⚠️ REGRAS ELIMINATÓRIAS DE NEGÓCIO (IMPORTANTE):

    1. **JOVEM APRENDIZ (Cuidado!)**:
       - APENAS se o candidato tiver MENOS de 18 anos.
       - COMO SABER A IDADE? Olhe a data de conclusão do Ensino Médio. Se concluiu o ensino médio antes de 2025, ele JÁ É MAIOR DE IDADE (tem 19+ anos), então NÃO coloque aqui.
       - Se ele já tiver Ensino Superior ou estiver na faculdade há mais de 1 ano, ele NÃO é Jovem Aprendiz.
       - Na dúvida sobre a idade, considere MAIOR de 18 e use a regra 3.
    
    2. **HIERARQUIA (GERENTES vs OPERACIONAIS)IMPORTANTE!!!**:
       - Se o cargo for de Liderança Estratégica (Gerente, Head, Diretor), use as pastas que começam com "GERENTE ...".
       - Exemplo: Um "Gerente de Marketing" vai para "GERENTE MARKETING". Um "Analista de Marketing" vai para "MARKETING".
       - Exemplo: "Coordenador" e "Supervisor" têm pastas específicas na lista (ex: SUPERVISOR DE MERCHANDISING). Se não tiver pasta específica de coordenação, jogue na área geral.

    3. **HIERARQUIA NA EXPERIÊNCIA**: Levar a serio o criterio de ultima experiência do colaborador, Exemplo: Ele tem experiência como supervisor de merchandising mas também como coordenador(Cargo acima) não à motivos para colocar ele em um cargo abaixo.

    4. **GERENTE DE GRANDES CONTAS**:Essa pasta é específica, então ela é uma vaga para o trade marketing e vai fazer uma ponte com o comercial cuidando de nossas grandes redes. Então o rapaz tem que já ter experiência com esse assunto.

    5. **KEY ACCOUNT**: Aqui nesta empresa, essa pasta é especifica para o pessoal mais comercial focado em VENDAS para as grandes redes
    
    6. **EMPILHADEIRA**: O candidato SÓ vai para esta pasta se citar explicitamente "Curso de Empilhadeira", "Operador de Empilhadeira" ou "NR-11". Se tiver experiência em logística mas não tiver o curso, jogue em "LOGÍSTICA" ou "ALMOXARIFADO".
    
    7. **MOTORISTA**: Exige CNH categorias C, D ou E (Caminhão/Carreta). Se tiver apenas CNH B ou Moto, NÃO coloque aqui (jogue em LOGÍSTICA ou OUTROS).
    
    8. **VIGIA**: Obrigatório ter "Curso de Vigilante", "Reciclagem em dia" ou experiência comprovada em segurança patrimonial.
    
    9. **COMERCIO EXTERIOR**: O candidato deve ter experiência com Importação/Exportação, trâmites aduaneiros ou vendas internacionais.
    
    10. **PCP**: Significa "Planejamento e Controle da Produção". Se o currículo falar de planejar fábrica, cronograma de produção ou ordens de serviço, é aqui.
    
    11. **PROMOTOR DE VENDAS**: Só será colocado nesta pasta caso a pessoa já tenha experiência como promotor antes.

    12. **LIXO/INVALIDO**: Se o arquivo for foto de pessoa, print de tela, boleto ou não for um currículo, responda "ARQUIVO_INVALIDO".

    13. **ADMINISTRATIVO**: Essa pasta é para aqueles currículos de pessoas jovens que sejam acima dos 18 e que não tenham nenhuma experiência, mas tenham cursos de áreas importantes.

    14. **PCD**: Se o currículo mencionar explicitamente "PCD", "Deficiência", "CID" ou "Laudo Médico", jogue aqui.

    15. **AREA INDUSTRIAL**: ATENÇÃO! Nesta empresa, esta pasta é EXCLUSIVA para "Técnico em Segurança do Trabalho", "Engenheiro de Segurança" ou "SESMT". Não jogue operadores de máquina aqui (jogue em PRODUÇÃO).

    16. **MECANICA INDUSTRIAL**: Aqui não são só colocados currículos de mecânicos, mas de tudo que envolve essa área, como eletricistas

    17. **QUALIDADE**: A vaga aqui pode ser alocada o pessoal que tenha experiência ou tenha feito alguma especialização mais laboral, como biomedicina e áreas da saúde.

    18. **ANTI-ESPELHO (O PRÓPRIO ANÚNCIO)**: 
        - Se o texto extraído contiver instruções de como se candidatar (ex: "Como participar", "Envie seu currículo para", "Vem ser time amarelo", "WhatsApp para envio"), isso NÃO É UM CURRÍCULO, é a imagem da vaga.
        - Neste caso, responda OBRIGATORIAMENTE: "ARQUIVO_INVALIDO".

    ENTRADA BRUTA:
    {texto_curriculo[:12000]}

    RESPONDA JSON:
    {{
        "setor": "NOME_DA_CATEGORIA",
        "confianca": "ALTA",
        "anos_experiencia": 0,
        "resumo": "Motivo",
        "cv_limpo": "TEXTO DO CV REESCRITO..."
    }}
    """

    try:
        model = genai.GenerativeModel(NOME_MODELO_GEMINI)
        response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Erro Gemini: {e}")
        return {"setor": "OUTROS", "confianca": "ERRO_IA", "resumo": str(e), "cv_limpo": texto_curriculo}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 INICIANDO SERVIDOR...")
    sys.modules['__main__'] = sys.modules[__name__]
    yield
    logger.info("🛑 DESLIGANDO SERVIDOR...")
    http_session.close()

app = FastAPI(title="API Triagem", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/triagem")
def triar_curriculo(file: UploadFile = File(...)):
    file.file.seek(0, 2)
    if file.file.tell() > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Arquivo muito grande (>5MB)")
    file.file.seek(0)

    try:
        content = file.file.read()
        raw_text = extract_text_from_memory(content, file.filename)
        analise = analisar_com_gemini(raw_text)
        setor = analise.get("setor", "OUTROS")
        cv_final = analise.get("cv_limpo") if analise.get("cv_limpo") and len(analise.get("cv_limpo")) > 50 else raw_text

        conf_map = {"ALTA": 0.98, "MEDIA": 0.75, "BAIXA": 0.45, "ERRO_IA": 0.0}
        conf_val = conf_map.get(analise.get("confianca"), 0.5)
        is_extension = "FONTE: LINKEDIN" in raw_text.upper()
        
        if WEBHOOK_GOOGLE_URL and setor != "ARQUIVO_INVALIDO" and is_extension:
            try:
                nome_base = file.filename.replace("perfil_linkedin_auto", "").replace(".txt", "").strip()
                nome_candidato = sanitize_filename(nome_base) or "Candidato LinkedIn"

                payload = {
                    "nome": nome_candidato,
                    "texto": cv_final,
                    "setor": setor,
                    "confianca": f"{conf_val:.2%}",
                    "url_perfil": "Via Extensão Chrome",
                    "detalhes": analise
                }
                http_session.post(WEBHOOK_GOOGLE_URL, json=payload, timeout=10)
                logger.info(f"✅ Webhook enviado: {nome_candidato}")
                
            except Exception as w_err:
                logger.error(f"⚠️ Erro Webhook: {w_err}")

        logger.info(f"🏁 {file.filename} -> {setor}")

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
        logger.critical(f"🔥 Erro Fatal: {e}")
        return {"status": "erro", "mensagem": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
