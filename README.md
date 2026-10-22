# AI Hub — Escritório de Recrutamento Multiagente

Escritório isométrico em que agentes de IA apoiam o RH: a **Bia** (recepção) recebe e tria os currículos que
chegam por e-mail, o **Atlas** (gerente) distribui o trabalho — e **contrata reforço quando precisa** — e os
**analistas** (Sofia, Flávio e quem o Atlas contratar) comparam cada currículo com os requisitos da vaga e
recomendam candidatos com evidências. Cada agente tem **personalidade própria**, mostra o que está pensando em
**balões** e **pede ajuda a colegas** de outra área quando tem dúvida sobre um currículo.
**A decisão final é sempre do recrutador.**

O escritório só mostra o que acontece de verdade: cada personagem sentado no computador tem uma tarefa real na
fila; os números vêm do banco; quem descansa está ocioso no sistema; os balões de trabalho mostram o que a
leitura do currículo encontrou.

```
 e-mail ──► fluxo de e-mail atual ──► POST /triagem ─┐
                                                      ▼
            ┌──────────────────── fila de tarefas (banco) ─────────────────────┐
            │ TRIAGEM (Bia) → DISTRIBUIÇÃO (Atlas) → ANÁLISE (analistas)       │
            │                         ANÁLISE ⇄ CONSULTA (colega) → REVISÃO    │
            └──────────────────────────────────────────────────────────────────┘
               ▲ supervisão do Atlas: contrata, reforça, dispensa, tenta de novo, boletim, pausa da IA
               │ eventos (SSE, tempo real): pensamentos, falas, contratações
               ▼
        escritório isométrico (frontend/) ── balões, painéis, currículo, chat, canal da equipe, administração
```

## Estrutura

```
backend/
  app/main.py         API (FastAPI), trabalhador da fila, rotinas de retenção, usuários
  app/servicos.py     regras: recepção, Atlas (distribuição, contratações, supervisão), análises,
                      consultas entre colegas, estados, LGPD
  app/personalidade.py  jeito de falar, bordões, pensamentos e papo de cada agente
  app/triagem.py      triagem dos currículos por pasta (mesmo contrato da API de triagem anterior)
  app/ia.py           análise, consulta e revisão, sugestões do Atlas, chat (streaming) e conversa em grupo
                      (Gemini ou modo demonstração)
  app/uso_ia.py       tokens e custo de cada chamada, limite diário e pausa automática da IA
  app/preparar.py     gera o .env de produção com segredos fortes
  app/models.py       modelo de dados
  app/seguranca.py    login, papéis, senhas, auditoria, arquivos cifrados
  app/email_imap.py   leitura de caixa de e-mail (opcional, desligada)
  demo/               currículos fictícios para demonstração
  tests/              testes ponta a ponta
frontend/             escritório (HTML + JS puro, sem dependências)
Dockerfile, docker-compose.yml, Caddyfile   produção: app + PostgreSQL + HTTPS automático
LGPD.md               como o sistema atende a LGPD e o que validar com o jurídico
```

## Experimentar (demonstração, sem IA externa)

Requer Python 3.11+. Nenhum dado sai do computador.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
SEED_DEMO=true IA_MODO=demo uvicorn app.main:app --port 8000
```

Abra `http://localhost:8000` e entre com `admin@demo.local` / `demo1234`. A demonstração cria três vagas de
exemplo e alimenta 17 currículos fictícios aos poucos, para ver a equipe trabalhando. As análises nesse modo
são heurísticas e vêm marcadas como tal. O que dá para ver acontecer sozinho:

1. ninguém cobre a vaga de vendas: o Atlas contrata alguém da área comercial, que chega, se apresenta e assume a vaga;
2. o currículo de um promotor de vendas cai na vaga de vendedor: quem analisa pede ajuda a "promotores", não há ninguém,
   o Atlas anota a necessidade e o ponto fica para o recrutador verificar;
3. a Priscila (almoxarifado, pasta LOGÍSTICA) cai com o Flávio; ele vai até a mesa da Sofia pedir ajuda,
   ela responde e inclui a Priscila também na vaga administrativa — onde ela é recomendada;
4. o Tiago (professor) não cai em pasta nenhuma: o Atlas lê, não acha vaga parecida e pede ajuda ao recrutador;
5. chega um lote de currículos de produção: a fila do Flávio passa do limite, o Atlas contrata reforço e
   divide a fila;
6. quem fica ocioso vai descansar (pingue-pongue, sofá, pufes, leitura, café) e puxa papo com os colegas.

**Testes:** `cd backend && pip install -r requirements-dev.txt && python -m pytest -q`

## Colocar em produção

O caminho recomendado é um servidor (VPS) com Docker: a aplicação, o PostgreSQL e o Caddy (HTTPS automático
com Let's Encrypt) sobem juntos.

**1. Domínio.** Crie um registro DNS `A` (ex.: `rh.suaempresa.com.br`) apontando para o IP do servidor e
libere as portas 80 e 443.

**2. Configuração com segredos fortes.** No servidor, dentro do repositório:

```bash
cd backend
python3 -m pip install cryptography          # só para o gerador; o resto roda no Docker
python3 -m app.preparar --docker --dominio rh.suaempresa.com.br --empresa "Sua Empresa" \
    --admin-email rh@suaempresa.com.br --gemini-key SUA_CHAVE_DO_GEMINI
cd ..
```

Isso grava `.env` na raiz (permissão 600, fora do Git) com `AMBIENTE=producao`, segredo de sessão, chave de
criptografia dos currículos, chave da API de triagem, senha do banco e senha inicial do administrador. A senha
inicial e a chave da triagem aparecem **uma única vez** na tela: guarde-as num cofre de senhas, junto com uma
cópia do `.env` (sem a `CHAVE_ARQUIVOS` os currículos cifrados não podem ser lidos).

**3. Subir.**

```bash
docker compose up -d --build
docker compose logs -f app          # acompanhe a primeira subida
```

Abra `https://rh.suaempresa.com.br`, entre com o e-mail do administrador e a senha inicial: o sistema pede
uma senha nova no primeiro acesso. Em **⚙ Administração → IA e custos**, use **Testar conexão com a IA** para
conferir chave, créditos e modelo.

Com `AMBIENTE=producao` o sistema **se recusa a subir** se faltar algo essencial (segredo fraco, chave de
arquivos ou da triagem ausente, cookie sem HTTPS, dados de demonstração ligados, IA sem chave) e mostra o
motivo no log.

**Backups** (agende diariamente e guarde fora do servidor):

```bash
docker compose exec -T db pg_dump -U hub hub | gzip > backup-$(date +%F).sql.gz        # banco
docker run --rm -v "$(basename "$PWD")_dados:/dados" -v "$PWD:/b" alpine \
    tar czf /b/arquivos-$(date +%F).tgz -C /dados .                                    # currículos cifrados
```

**Atualizar:** `git pull && docker compose up -d --build` (o banco é migrado na inicialização).

**Sem Docker** (servidor simples, SQLite): `python -m app.preparar` (grava `backend/.env`), depois
`uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers` atrás de um proxy com HTTPS
(Caddy/Nginx) que **não** faça buffer de `/api/eventos/stream` e `/api/agentes/*/chat/stream`. Mantenha um
único processo: o trabalhador da fila roda dentro da API.

### Usuários e senhas

- Em **⚙ Administração → Usuários** (administrador) você cria acessos, muda o papel, desativa e gera uma nova
  senha temporária. A senha temporária aparece só uma vez; a pessoa escolhe a dela no primeiro acesso.
- **Trocar senha** fica no topo da tela. Ao trocar, as outras sessões abertas com a senha antiga caem; mudar o
  papel ou desativar alguém também encerra as sessões dessa pessoa na hora.
- Senhas com pelo menos 10 caracteres, não só números; login com limite de tentativas. Sempre fica pelo menos
  um administrador ativo.

| Papel | Pode |
|---|---|
| recrutador | ver o escritório, agentes, candidatos e currículos; conversar com agentes e com a equipe; registrar decisões |
| supervisor | + criar/suspender/encerrar vagas, enviar currículos, contratar e dispensar agentes, aprovar propostas do Atlas, resolver pendências, ver uso e custos da IA |
| admin | + usuários, teste da IA e pedidos de titulares (LGPD) |

## IA: modelo, custo e controle

**Modelo.** Por padrão tudo usa o **`gemini-3.5-flash-lite`** com raciocínio **mínimo**: é o modelo leve atual
do Google, bom para extrair e comparar informações de currículos, com preço baixo (≈ US$ 0,30 por milhão de
tokens de entrada e US$ 2,50 de saída). A família 2.5 está sendo desligada pelo Google e não aceita projetos
novos. Alternativas configuráveis em `GEMINI_MODELO` (ou separadamente `GEMINI_MODELO_TRIAGEM` e
`GEMINI_MODELO_AGENTES`):

| Modelo | US$ / 1M tokens (entrada / saída) | Quando usar |
|---|---|---|
| `gemini-3.1-flash-lite` | 0,25 / 1,50 | o mais barato; bom o bastante para a triagem |
| `gemini-3.5-flash-lite` (padrão) | 0,30 / 2,50 | equilíbrio entre qualidade e custo |
| `gemini-3.8-flash` | 0,75 / 3,75 | análises mais difíceis (preço promocional; confira a tabela do Google) |

Os preços são de referência (out/2026). Se mudarem, ajuste `GEMINI_PRECO_ENTRADA` e `GEMINI_PRECO_SAIDA`.

**Quanto custa.** Um currículo típico (triagem + análise) gasta cerca de US$ 0,004 com o modelo padrão — perto
de **US$ 4 a cada mil currículos**; consultas entre colegas, chat e canal da equipe somam por uso.

**Economia de tokens já embutida:**
- raciocínio ("thinking") no mínimo — ele é cobrado como saída; o sistema desce para o nível aceito se o modelo
  recusar o mínimo (`GEMINI_PENSAMENTO`);
- PDF com texto é enviado como texto (bem menos tokens que o arquivo); só imagem e PDF escaneado vão como arquivo;
- respostas em JSON compacto com teto de tamanho; a revisão depois de consultar um colega não reenvia o currículo;
- instruções fixas separadas do conteúdo (aproveitam o cache automático do Gemini, cobrado a ~10%);
- contexto do chat e do canal da equipe enxuto (só os registros necessários).

**Controle de gasto e pausa automática.**
- Cada chamada registra tokens (entrada, cache, saída, raciocínio) e custo estimado. Veja em **⚙ Administração
  → IA e custos** (gasto do dia, últimos 30 dias e para onde vão os tokens) e no quadro geral.
- `IA_LIMITE_DIARIO_USD` (padrão US$ 3) é o teto por dia, no fuso `FUSO_HORARIO`. Ao atingir, a fila de IA
  pausa até o dia seguinte; nada se perde.
- Se o Google recusar (conta **sem créditos**, chave inválida, modelo indisponível, cota diária esgotada), a IA
  entra em pausa com espera crescente, o Atlas abre **uma** pendência explicando o motivo e o selo "IA pausada"
  aparece no topo. As tarefas voltam para a fila sem gastar tentativas e recomeçam sozinhas quando a IA responde.
- **Sem créditos (erro 402):** a API do Gemini usa créditos pré-pagos. Recarregue em
  <https://ai.studio/projects> (projeto da chave → faturamento) e clique em **Testar IA agora**.

**Use o plano pago da API.** No gratuito, o Google pode usar o conteúdo para melhorar produtos, com revisão
humana — incompatível com currículos. Detalhes em `LGPD.md`.

## Integração com o fluxo de e-mail

O endpoint `POST /triagem` mantém **o mesmo contrato** da API de triagem anterior (campo `file`; resposta com
`arquivo`, `nome_identificado`, `setor_sugerido`, `confianca`, `detalhes`). O fluxo que hoje lê o e-mail e
chama a API continua igual; só é preciso:

1. apontar para o novo endereço (`https://seu-dominio/triagem`);
2. enviar o cabeçalho `X-API-Key` com o valor de `TRIAGEM_API_KEY`.

```bash
curl -F "file=@curriculo.pdf" -H "X-API-Key: $TRIAGEM_API_KEY" https://rh.suaempresa.com.br/triagem
```

Além de responder, o currículo é registrado, deduplicado, distribuído pelo Atlas e analisado. Um mesmo arquivo
reenviado não é reprocessado; um currículo pode entrar em várias vagas, uma análise por vaga. Se a IA estiver
pausada, a resposta vem na hora com `"status": "na_fila"` e a triagem acontece quando ela voltar.

Se preferir que o próprio sistema leia a caixa de entrada, ligue `IMAP_ATIVO=true` e preencha os dados IMAP.

## O escritório

Planta maior, em seis ambientes ligados por um corredor: **recepção** (Bia e o balcão de entrada), **vagas em
aberto** (quadro que os agentes consultam), **sala do gerente** (Atlas), **alimentação** (mesas, café e
geladeira), **análise de currículos** (12 estações em três fileiras) e **descanso** (pingue-pongue, sofás com
TV, pufes e cantinho de leitura). Os personagens têm corpo inteiro com volume e sombra, viram para onde andam,
caminham pelas portas e sentam para digitar; móveis à frente de alguém são redesenhados por cima, como num
cenário 3D. Clique em uma sala ou personagem para abrir o painel; role ou use os botões para dar zoom.

## A equipe viva

**Personalidades.** Atlas (calmo, estrategista), Bia (organizada e animada), Sofia (meticulosa, vai aos fatos)
e Flávio (prático, do chão de fábrica) têm perfis próprios em `app/personalidade.py`: tom, bordões,
pensamentos típicos e papo de descanso. Quem o Atlas contrata recebe um de oito arquétipos (escolhido pelo
nome), então cada novo colega fala do seu jeito. Com Gemini, a personalidade entra no prompt da análise, do
chat e do canal da equipe; na demonstração, ela aparece nas frases montadas a partir dos registros.

**Balões.** Nuvem = pensamento; balão = fala. Os de trabalho vêm de eventos reais: o que a leitura do currículo
encontrou ("✔ Excel intermediário: está no currículo"), a dúvida levada ao colega, a resposta, as boas-vindas a
quem chegou. O papo no descanso e os pensamentos de quem está à toa são só ambientação, tirados da
personalidade. O botão 💭 (ao lado do zoom) liga e desliga os balões.

**Colegas que se ajudam.** Na análise, o agente pode concluir que outra especialidade responde melhor a uma
dúvida (termo técnico, certificação, perfil de outra área). Então:

1. ele vai até a mesa do colega menos ocupado daquela área e pergunta (tarefa `CONSULTA` na fila do colega);
2. o colega lê o currículo, responde com trechos do currículo e, se o candidato combinar claramente com uma
   vaga dele, **inclui o candidato nessa vaga** (candidatura de origem "indicação");
3. o analista revisa a própria análise com a resposta (tarefa `REVISAO`) e agradece.

A conversa fica registrada na análise ("Conversa com colega") e no canal da equipe. Se não houver ninguém da
área, o Atlas anota a necessidade e o ponto vai para "A verificar". Quem indicou não é consultado de volta
sobre o mesmo candidato (evita pingue-pongue).

**O Atlas contrata e dispensa.** A cada rodada de supervisão (`SUPERVISAO_SEGUNDOS`), ele:

| Situação | O que o Atlas faz |
|---|---|
| vaga aberta sem ninguém da área | contrata um especialista (ou chama de volta quem saiu), que assume a vaga e a fila que esperava |
| fila média da área acima de `LIMITE_FILA_POR_AGENTE` | contrata reforço (até `MAX_AGENTES_POR_ESPECIALIDADE`), que divide as vagas e metade da fila |
| `DEMANDA_PARA_CONTRATAR` pedidos de ajuda a uma área inexistente em 7 dias | contrata alguém dessa área |
| as 12 estações ocupadas | libera a estação de quem está sem vagas e sem fila; se não houver, abre pendência para o recrutador |
| IA pausada (sem créditos, limite diário…) | segura a fila, avisa numa pendência única e retoma sozinho quando a IA volta |

Contratações respeitam `INTERVALO_CONTRATACAO` por área. Com `CONTRATACAO_AUTOMATICA=false`, o Atlas só
**propõe** no quadro geral (botão "Aprovar contratação"). O recrutador também contrata ("Contratar agente") e
dispensa (painel do agente → "Dispensar da equipe"); a dispensa é recusada se o agente for o único da área com
vaga aberta. Todo o histórico aparece em "Contratações e dispensas" (painel do Atlas e quadro geral).

**Outras automações**

- **Encaminhamento de currículos sem pasta** (`ENCAMINHAMENTO_AUTOMATICO`): o Atlas lê o currículo "OUTROS" e
  o encaminha para a vaga aberta que combinar; se nenhuma combinar, pede ajuda ao recrutador.
- **Nova tentativa automática**: falhas de IA são reprocessadas pelo Atlas depois de `REPROCESSAR_APOS_MIN`
  (até 2 vezes por item) antes de dependerem do recrutador.
- **Vaga nova**: o botão "✨ Pedir sugestão ao Atlas" escolhe as pastas da triagem (e, com Gemini, sugere
  requisitos). Sem pastas marcadas, o Atlas deduz sozinho ao criar a vaga.
- **Boletim** (`BOLETIM_MINUTOS`): o Atlas resume o período no canal da equipe quando houve movimento.

**Conversas**

- **Chat com um agente**: resposta em streaming (aparece sendo digitada), na voz do agente, a partir dos registros
  dele; se o assunto for de um colega, ele diz com quem falar.
- **Canal da equipe** (botão 💬 Equipe): você escreve para o time e quem tem a informação responde — o Atlas
  organiza, chama quem tem vagas, e um colega pode complementar o outro. O mesmo canal mostra, em tempo real, o
  que os agentes dizem entre si (pedidos de ajuda, contratações, boletins). O que você escreve não é armazenado.

## Configuração

Todas as variáveis estão comentadas em `backend/.env.example`. As principais:

| Variável | Padrão | Para quê |
|---|---|---|
| `AMBIENTE` | `desenvolvimento` | `producao` recusa subir com configuração insegura |
| `APP_NOME`, `APP_SUBTITULO`, `EMPRESA_NOME` | `AI Hub`, `Escritório de Recrutamento`, vazio | nome na tela e empresa citada nos prompts e na LGPD |
| `GEMINI_MODELO` | `gemini-3.5-flash-lite` | modelo de tudo (ou `GEMINI_MODELO_TRIAGEM` / `GEMINI_MODELO_AGENTES`) |
| `GEMINI_PENSAMENTO` | `minimal` | nível de raciocínio (`minimal`, `low`, `medium`, `high`, `padrao`) |
| `IA_LIMITE_DIARIO_USD` | `3` | teto de gasto estimado por dia (0 = sem teto) |
| `CONTRATACAO_AUTOMATICA` | `true` | Atlas contrata/dispensa sozinho; `false` = só propõe |
| `LIMITE_FILA_POR_AGENTE` | `5` (`2` na demo) | média de currículos na fila por agente que dispara reforço |
| `MAX_AGENTES_POR_ESPECIALIDADE` | `3` | teto de agentes por área |
| `INTERVALO_CONTRATACAO` | `600` s (`30` na demo) | espera entre contratações da mesma área |
| `DEMANDA_PARA_CONTRATAR` | `2` | pedidos de ajuda sem especialista (7 dias) que levam a contratar |
| `COLABORACAO` | `true` | consultas entre colegas |
| `ENCAMINHAMENTO_AUTOMATICO` | `true` | Atlas encaminha currículos sem pasta |
| `REPROCESSAR_APOS_MIN` | `10` | espera antes da nova tentativa automática |
| `BOLETIM_MINUTOS` | `60` | boletim do Atlas (0 desliga) |
| `RETENCAO_DIAS` / `RETENCAO_CONVERSAS_HORAS` | `180` / `72` | anonimização de candidatos inativos / guarda das conversas entre agentes |

## Estados dos agentes

| Estado | Origem no sistema | No escritório |
|---|---|---|
| Trabalhando | tarefa em execução ou na fila do agente | sentado, digitando, tela acesa |
| Aguardando | tem vagas, sem tarefas | na estação, tela apagada |
| Disponível | ativo, sem vaga atribuída | na estação |
| Descansando | ocioso há `SEGUNDOS_PARA_DESCANSO` | pingue-pongue, sofá/TV, pufe, leitura, alimentação |
| Com pendência | pendência aberta em seu nome | “!” vermelho; detalhes no quadro geral |
| Concluído | acabou de esvaziar a fila | tela verde por alguns segundos |
| Em deslocamento | caminhada visual disparada por evento real (buscar currículos, consultar vagas, pedir ajuda a um colega, chegar contratado, sair dispensado, ir/voltar do descanso) | anda pelos corredores e portas |
| Conversando | fala com um colega (consulta, boas-vindas, papo no descanso) | balão de fala |

O deslocamento é só visual: a tarefa já começou no servidor, o descanso nunca atrasa o processamento.

## Endpoints

| Método | Caminho | Uso |
|---|---|---|
| POST | `/triagem` | entrada do fluxo de e-mail (cabeçalho `X-API-Key`) |
| GET | `/api/saude` | verificação de saúde (banco e situação da IA) |
| GET | `/api/marca` | nome do sistema exibido na tela (público) |
| POST | `/auth/login`, `/auth/senha`, `/auth/sair` | sessão e troca da própria senha |
| GET/POST/PATCH | `/api/usuarios`, `/api/usuarios/{id}`, `/api/usuarios/{id}/senha` | gestão de acessos (admin) |
| GET | `/api/ia/uso` | tokens, custo estimado e pausa da IA (supervisor) |
| POST | `/api/ia/testar` | chamada mínima de diagnóstico ao modelo (admin) |
| POST | `/api/agentes/{id}/chat/stream` | chat com o agente, resposta em pedaços (`text/plain`) |
| GET / POST | `/api/equipe/conversas`, `/api/equipe/chat` | canal da equipe |
| POST | `/api/agentes`, `/api/agentes/{id}/dispensar` | contratar e dispensar agentes |
| GET | `/api/contratacoes` | histórico de contratações e dispensas |
| POST | `/api/vagas`, `/api/vagas/sugerir` | vagas e sugestões do Atlas |

## Limitações conhecidas

- Até 12 analistas ativos (12 estações na sala de análise); o Atlas libera estações de quem está sem vagas.
- Leitura de PDF escaneado/imagem depende do Gemini (o modo demonstração não faz OCR).
- O chat (individual e do canal da equipe) não guarda o que você escreve no servidor (minimização de dados);
  some ao recarregar a página. As falas entre agentes ficam `RETENCAO_CONVERSAS_HORAS`.
- Cada consulta entre colegas é mais uma leitura do currículo pela IA (custo e latência maiores com Gemini);
  desligue com `COLABORACAO=false` se preferir.
- Os custos exibidos são estimativas a partir dos tokens informados pela API; a cobrança oficial é a do Google.
- Bancos criados por versões anteriores são atualizados na inicialização (colunas novas são acrescentadas).
