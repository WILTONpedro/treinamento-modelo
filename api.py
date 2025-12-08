import os
import io
import json
import logging
import re
import sys
from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import pdfplumber
import docx
from PIL import Image
import pytesseract
import google.generativeai as genai

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("api")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
NOME_MODELO_GEMINI = "gemini-2.0-flash"

CATEGORIAS_DISPONIVEIS = [
    "ADMINISTRITIVO", "ALMOXARIFADO", "AREA INDUSTRIAL", "COMERCIAL", "COMERCIO EXTERIOR",
    "COMPRAS", "CONTABILIDADE", "COORDENADOR DE EXPEDIÇÃO", "COORDENADOR DE MERCHANDISING",
    "EMPILHADEIRA", "EVENTOS", "FINANCEIRO", "GERENTE COMERCIAL", "GERENTE FINANCEIRO",
    "GERENTE GRANDES CONTAS", "GERENTE LOGISTICA", "GERENTE MARKETING", "GERENTE PRODUÇÃO",
    "GERENTE QUALIDADE", "GERENTE DE RH", "GERENTE VENDAS", "HIGIENIZAÇÃO", "JOVEM APRENDIZ",
    "KEY ACCOUNT", "LIDER DE PRODUÇÃO", "LOGÍSTICA", "MARKETING", "MECANICA INDUSTRIAL",
    "MERCHANDISING", "MOTORISTA", "PCD", "PCP", "PRODUÇÃO", "PROJETOS", "PROMOTOR DE VENDAS",
    "QUALIDADE", "RECURSOS HUMANOS", "SUPERVISOR DE MERCHANDISING", "SUPERVISOR DE VENDAS", "TI", "VENDAS", "VIGIA", "OUTROS"
]

def sanitize_filename(filename):
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
        elif ext in [".jpg", ".png", ".jpeg"]:
            img = Image.open(file_stream)
            if img.width > 2000: img.thumbnail((2000, 2000))
            text = pytesseract.image_to_string(img, lang="por")
        elif ext == ".txt":
            text = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Erro leitura: {e}")
        return ""
    return text.replace("\x00", "")

def analisar_com_gemini(texto_curriculo):
    if not texto_curriculo or len(texto_curriculo.strip()) < 20:
        return {"setor": "ARQUIVO_INVALIDO", "confianca": "BAIXA", "motivo": "Texto insuficiente"}

    prompt = f"""
    Você é um Recrutador Sênior da Baly.
    LISTA PERMITIDA: {json.dumps(CATEGORIAS_DISPONIVEIS)}
    
    TAREFA 2 (CRUCIAL): O texto abaixo veio de uma extração bruta do LinkedIn ou PDF e está sujo.
    Você deve REESCREVER e ESTRUTURAR as informações em formato de Currículo Profissional Limpo.
    - Remova: Botões ("Conectar", "Enviar mensagem"), propagandas, menus, "Pessoas também viram", textos de interface.
    - Mantenha: Nome, Resumo, Experiência (Empresas, Cargos, Datas), Formação, Idiomas e Competências.
    - Formato: Texto corrido bem organizado (Markdown simples).

    TAREFA 3 (SUPER IMPORTANTE): Sempre tente capturar o nome da pessoa no currículo.
    - Geralmente fica na parte de cima do currículo.
    - Geralmente é um nome composto (Exemplo: Wilton Pedro Silva Souza), pegue apenas o nome e sobrenome. (Exemplo: Wilton Pedro)

    REGRA SUPREMA: EVITAR AO MÁXIMO CRIAR PASTAS NOVAS.
    - Se no currículo do candidato tiver coisas que não foge tanto das categorias listadas, NÃO crie outras pastas.
    - Leve em consideração a hierarquia que está listado nas regras, então se o candidato for um EXECUTIVO, não é para colocar em uma pasta abaixo, coloque na SUPERVISOR DE VENDAS e por aí vai.

    ⚠️ REGRA DE OURO (HIERARQUIA):
    - Se o candidato tem experiência em **GESTÃO, LIDERANÇA, COORDENAÇÃO ou MBA**, ele é **PROIBIDO** de entrar na pasta "ADMINISTRATIVO".
    - Ele deve ir para a pasta de Gerência/Supervisão da área dele (Ex: Lucas tem MBA em Liderança -> GERENTE COMERCIAL ou SUPERVISOR DE VENDAS).


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

    19. **SUPERVISOR DE VENDAS**: Nesta pasta é para colocar todos que estão acima da pasta "VENDAS", então, executivos, gerentes, etc... Tudo aqui.

    20. **ATENÇÃO!! Nossas pastas e como elas funcionam**:
        - Nossas pastas Funcionam assim: Uma para GERENTE do setor, e a outra para que varia de Analista ate auxiliar, ou seja: A de GERENTE MARKETING vai os perfis mais adequados para está pasta, com experiências mais relevantes, e a de MARKETING vai o pessoal que tem experiência como Analista pra baixo
        - Não tente ir criando novas pastas como você estava criando (Ex: Executivo de vendas, representante, etc), tente encaixar os currículos nas pastas já existentes, sem criar novas.
        - Tente procurar similaridades de experiências com as pastas do drive que já temos (Ex: Você criou a pasta PROPAGANDISTA, porém quem faz propaganda geralmente é vinculado a parte do marketing).

    21. **ANTI ARQUIVO INUTIL**:
        - Geralmente o pessoal envia junto ao currículo, uma apresentação por powerpoint, cartas de apresentação, diplomas, cartas de indicação, etc...
        - Ao ver arquivos nesse tipo, não salve no drive, apenas pule para o próximo.

    ⚠️ REGRAS DE AGRUPAMENTO (EVITE CRIAR PASTAS REDUNDANTES):
    
    1. **VENDEDORES / COMERCIAL**:
       - Se for "Vendedor", "Vendedor Interno", "Balconista", "Consultor de Vendas" -> Use a pasta **VENDAS**. (Não crie pasta "Vendedor").
       - Se for "Representante Comercial" -> Use a pasta **VENDAS** ou **COMERCIAL**.
       - Se for "Vendedor Externo" -> Use a pasta **VENDAS** (ou PROMOTOR DE VENDAS se for focado em merchandising).
    
    2. **LIDERANÇA DE VENDAS**:
       - Supervisores, Coordenadores, Líderes de vendas -> Use **SUPERVISOR DE VENDAS**.
       - Gerentes -> Use **GERENTE VENDAS**.
    
    3. **TI / SUPORTE**:
       - Dev, Suporte, Infra, Redes -> Use **TI**

    4. **NOVAS PASTAS**:
       - Você pode sugerir uma pasta nova APENAS se o cargo for totalmente diferente de tudo que existe na lista (Ex: "Médico", "Advogado"). 
       - Mas para variações comuns (Vendedor x Vendas), USE A PASTA EXISTENTE NA LISTA.

    ENTRADA: {texto_curriculo[:15000]}

    RESPONDA JSON:
    {{
        "nome": "Nome Sobrenome",
        "setor": "UMA_DAS_CATEGORIAS_DA_LISTA", 
        "confianca": "ALTA", 
        "anos_experiencia": 0, 
        "resumo": "Explique por que escolheu este setor (se tiver liderança, cite aqui)", 
        "cv_limpo": "Texto..."
    }}
    """
    for tentativa in range(3):
        try:
            model = genai.GenerativeModel(NOME_MODELO_GEMINI)
            response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
            
            # --- FIX: Handle List vs Dict Response ---
            try:
                dados = json.loads(response.text)
                if isinstance(dados, list):
                    dados = dados[0]  # Take the first item if it's a list
            except json.JSONDecodeError:
                 # Fallback: try to find JSON block if raw text has extra chars
                 match = re.search(r'\{.*\}', response.text, re.DOTALL)
                 if match:
                     dados = json.loads(match.group())
                 else:
                     raise ValueError("Could not parse JSON from Gemini")

            setor_ia = dados.get("setor", "OUTROS").upper()
            if setor_ia not in CATEGORIAS_DISPONIVEIS:
                dados["setor"] = "OUTROS"
                dados["resumo"] += f" (IA tentou '{setor_ia}', movido para OUTROS)"
            
            return dados

        except Exception as e:
            if "429" in str(e):
                time.sleep(10)
            else:
                return {"setor": "OUTROS", "confianca": "ERRO_IA", "resumo": str(e)}
    
    return {"setor": "OUTROS", "confianca": "ERRO_IA", "resumo": "Timeout 429"}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 SERVIDOR INICIADO")
    yield

app = FastAPI(title="API Triagem", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/triagem")
def triar_curriculo(file: UploadFile = File(...)):
    file.file.seek(0)
    try:
        content = file.file.read()
        raw_text = extract_text_from_memory(content, file.filename)
        analise = analisar_com_gemini(raw_text)
        
        setor = analise.get("setor", "OUTROS")
        nome_ia = analise.get("nome", "Candidato")
        
        if nome_ia == "Desconhecido" or len(nome_ia) < 3:
             nome_ia = sanitize_filename(file.filename.replace(".pdf", "").replace(".docx", ""))

        logger.info(f"🏁 {file.filename} -> {setor} ({nome_ia})")

        return {
            "arquivo": file.filename,
            "nome_identificado": nome_ia,
            "setor_sugerido": setor,
            "confianca": analise.get("confianca", "BAIXA"),
            "detalhes": analise
        }

    except Exception as e:
        logger.error(f"Erro: {e}")
        return {"status": "erro", "mensagem": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
