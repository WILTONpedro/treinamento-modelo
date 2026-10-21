# Baly AI Hub — Escritório de Recrutamento Multiagente

Escritório isométrico em que agentes de IA apoiam o RH da Baly: a **Bia** (recepção) recebe e tria os
currículos que chegam por e-mail, o **Atlas** (supervisor) distribui o trabalho — e **contrata reforço quando
precisa** — e os **analistas** (Sofia, Flávio e quem o Atlas contratar) comparam cada currículo com os
requisitos da vaga e recomendam candidatos com evidências. Cada agente tem **personalidade própria**, mostra
o que está pensando em **balões** e **pede ajuda a colegas** de outra área quando tem dúvida sobre um currículo.
**A decisão final é sempre do recrutador.**

O escritório só mostra o que acontece de verdade: cada personagem sentado no computador tem uma tarefa
real na fila; os números vêm do banco; quem descansa está ocioso no sistema; os balões de trabalho
mostram o que a leitura do currículo encontrou.

```
 e-mail ──► fluxo de e-mail atual ──► POST /triagem ─┐
                                                      ▼
            ┌──────────────────── fila de tarefas (banco) ─────────────────────┐
            │ TRIAGEM (Bia) → DISTRIBUIÇÃO (Atlas) → ANÁLISE (analistas)       │
            │                         ANÁLISE ⇄ CONSULTA (colega) → REVISÃO    │
            └──────────────────────────────────────────────────────────────────┘
               ▲ supervisão do Atlas: contrata, reforça, dispensa, tenta de novo, boletim
               │ eventos (SSE, tempo real): pensamentos, falas, contratações
               ▼
        escritório isométrico (frontend/) ── balões, painéis, currículo, chat, canal da equipe
```

## Estrutura

```
backend/
  app/main.py        API (FastAPI), trabalhador da fila, rotinas de retenção
  app/servicos.py    regras: recepção, Atlas (distribuição, contratações, supervisão), análises,
                     consultas entre colegas, estados, LGPD
  app/personalidade.py  jeito de falar, bordões, pensamentos e papo de cada agente
  app/triagem.py     sua API de triagem, portada para o SDK novo do Gemini (mesmo prompt e categorias)
  app/ia.py          análise, consulta e revisão, sugestões do Atlas, chat (streaming) e conversa em grupo
                     (Gemini ou modo demonstração)
  app/models.py      modelo de dados
  app/seguranca.py   login, papéis, auditoria, arquivos cifrados
  app/email_imap.py  leitura de caixa de e-mail (opcional, desligada)
  demo/              currículos fictícios para demonstração
  tests/             teste ponta a ponta
frontend/            escritório (HTML + JS puro, sem dependências)
LGPD.md              como o sistema atende a LGPD e o que validar com o jurídico
```

## Como rodar

Requer Python 3.11+.

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # preencha as chaves (veja abaixo)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Abra `http://localhost:8000` e entre com `ADMIN_EMAIL` / `ADMIN_SENHA` do `.env`.

**Demonstração sem IA externa** (nenhum dado sai do computador):

```bash
SEED_DEMO=true IA_MODO=demo uvicorn app.main:app --port 8000
```

Entra com `admin@baly.local` / `demo1234`, cria três vagas de exemplo e alimenta 17 currículos fictícios
aos poucos, para ver a equipe trabalhando. As análises nesse modo são heurísticas e vêm marcadas como tal.
O que dá para ver acontecer sozinho na demonstração:

1. ninguém cobre a vaga de vendas: o Atlas contrata alguém da área comercial, que chega, se apresenta e assume a vaga;
2. o currículo de um promotor de vendas cai na vaga de vendedor: quem analisa pede ajuda a "promotores", não há ninguém,
   o Atlas anota a necessidade e o ponto fica para o recrutador verificar;
3. a Priscila (almoxarifado, pasta LOGÍSTICA) cai com o Flávio; ele vai até a mesa da Sofia pedir ajuda,
   ela responde e inclui a Priscila também na vaga administrativa — onde ela é recomendada;
4. o Tiago (professor) não cai em pasta nenhuma: o Atlas lê, não acha vaga parecida e pede ajuda ao recrutador;
5. chega um lote de currículos de produção: a fila do Flávio passa do limite, o Atlas contrata reforço e
   divide a fila;
6. quem fica ocioso vai descansar e puxa papo com os colegas.

**Testes:** `cd backend && python -m pytest -q`

## Integração com o fluxo de e-mail

O endpoint `POST /triagem` mantém **o mesmo contrato** da API original (mesmo campo `file`, mesma resposta:
`arquivo`, `nome_identificado`, `setor_sugerido`, `confianca`, `detalhes`). O fluxo que hoje lê o e-mail
e chama a API continua igual; só é preciso:

1. apontar para o novo endereço do servidor;
2. enviar o cabeçalho `X-API-Key` com o valor de `TRIAGEM_API_KEY`.

Além de responder, agora o currículo é registrado, deduplicado, distribuído pelo Atlas e analisado.
Um mesmo arquivo reenviado não é reprocessado; um currículo pode entrar em várias vagas, uma análise por vaga.

Se preferir que o próprio sistema leia a caixa de entrada, ligue `IMAP_ATIVO=true` e preencha os dados IMAP.

## Gemini

- O SDK `google-generativeai` usado na API original foi descontinuado pelo Google; o código agora usa
  `google-genai`. Prompt, categorias e regras da triagem foram mantidos.
- `GEMINI_MODELO_TRIAGEM` continua `gemini-2.5-flash` (o Google mantém os modelos 2.5 para quem já os usa).
  Para análises e chat, `GEMINI_MODELO_AGENTES` usa `gemini-3.8-flash`, recomendado para projetos novos.
  Os dois são configuráveis — confira a página de descontinuação do Google periodicamente.
- **Use o plano pago da API.** No gratuito, o Google pode usar o conteúdo para melhorar produtos, com
  revisão humana — incompatível com currículos. Detalhes em `LGPD.md`.

Outras correções em relação à API original: a chamada à IA não bloqueia mais o servidor; o tipo do arquivo é
inferido pela extensão; nova tentativa com espera crescente em caso de limite (429); logs sem nome de candidato;
CORS restrito; chave de acesso no endpoint.

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
| as 8 estações ocupadas | libera a estação de quem está sem vagas e sem fila; se não houver, abre pendência para o recrutador |

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

## Configuração da equipe

| Variável | Padrão | Para quê |
|---|---|---|
| `CONTRATACAO_AUTOMATICA` | `true` | Atlas contrata/dispensa sozinho; `false` = só propõe |
| `LIMITE_FILA_POR_AGENTE` | `5` (`2` na demo) | média de currículos na fila por agente que dispara reforço |
| `MAX_AGENTES_POR_ESPECIALIDADE` | `3` | teto de agentes por área |
| `INTERVALO_CONTRATACAO` | `600` s (`30` na demo) | espera entre contratações da mesma área |
| `DEMANDA_PARA_CONTRATAR` | `2` | pedidos de ajuda sem especialista (7 dias) que levam a contratar |
| `COLABORACAO` | `true` | consultas entre colegas |
| `ENCAMINHAMENTO_AUTOMATICO` | `true` | Atlas encaminha currículos sem pasta |
| `REPROCESSAR_APOS_MIN` | `10` | espera antes da nova tentativa automática |
| `BOLETIM_MINUTOS` | `60` | boletim do Atlas (0 desliga) |
| `RETENCAO_CONVERSAS_HORAS` | `72` | guarda de pensamentos e conversas entre agentes |

## Papéis

| Papel | Pode |
|---|---|
| recrutador | ver o escritório, agentes, candidatos e currículos; conversar com agentes e com a equipe; registrar decisões |
| supervisor | + criar/suspender/encerrar vagas, enviar currículos, contratar e dispensar agentes, aprovar propostas do Atlas, resolver pendências |
| admin | + usuários, auditoria e pedidos de titulares (LGPD) |

## Estados dos agentes

| Estado | Origem no sistema | No escritório |
|---|---|---|
| Trabalhando | tarefa em execução ou na fila do agente | sentado, tela acesa |
| Aguardando | tem vagas, sem tarefas | na estação, tela apagada |
| Disponível | ativo, sem vaga atribuída | na estação |
| Descansando | ocioso há `SEGUNDOS_PARA_DESCANSO` | pingue-pongue, sofá/TV, pufe, alimentação |
| Com pendência | pendência aberta em seu nome | “!” vermelho; detalhes no quadro geral |
| Concluído | acabou de esvaziar a fila | tela verde por alguns segundos |
| Em deslocamento | caminhada visual disparada por evento real (buscar currículos, consultar vagas, pedir ajuda a um colega, chegar contratado, sair dispensado, ir/voltar do descanso) | anda pelos corredores |
| Conversando | fala com um colega (consulta, boas-vindas, papo no descanso) | balão de fala e etiqueta amarela |

O deslocamento é só visual: a tarefa já começou no servidor, o descanso nunca atrasa o processamento.

## Endpoints novos

| Método | Caminho | Uso |
|---|---|---|
| POST | `/api/agentes/{id}/chat/stream` | chat com o agente, resposta em pedaços (`text/plain`) |
| GET | `/api/equipe/conversas` | canal da equipe: falas entre agentes, contratações, indicações |
| POST | `/api/equipe/chat` | mensagem para o time; devolve 1 a 3 respostas (`agente_id`, `nome`, `texto`) |
| POST | `/api/agentes/{id}/dispensar` | dispensa pedida pelo supervisor |
| GET | `/api/contratacoes` | histórico de contratações e dispensas |
| POST | `/api/vagas/sugerir` | pastas (e requisitos, com Gemini) sugeridos pelo Atlas |

`POST /api/agentes` agora contrata (mesma chamada de antes) e `POST /api/vagas` aceita `setores` vazio.

## Produção — checklist

- HTTPS na frente (Nginx/Caddy) e `COOKIE_SECURE=true`; desligue o buffer para `/api/eventos/stream` (SSE)
  e para `/api/agentes/*/chat/stream` (chat em streaming).
- PostgreSQL (`DATABASE_URL`) e backups; a `CHAVE_ARQUIVOS` guardada à parte, em cofre.
- **Um único processo** (`--workers 1`): o trabalhador da fila roda dentro da API.
- `TRIAGEM_API_KEY`, `SEGREDO_JWT` e `ADMIN_SENHA` fortes; troque a senha do admin após o primeiro acesso.
- Revise `LGPD.md` com o jurídico/encarregado antes de colocar candidatos reais.

## Limitações conhecidas

- Até 8 analistas ativos (8 estações na sala de análise); o Atlas libera estações de quem está sem vagas.
- Leitura de PDF escaneado/imagem depende do Gemini (o modo demonstração não faz OCR).
- O chat (individual e do canal da equipe) não guarda o que você escreve no servidor (minimização de dados);
  some ao recarregar a página. As falas entre agentes ficam `RETENCAO_CONVERSAS_HORAS`.
- Cada consulta entre colegas é mais uma leitura do currículo pela IA (custo e latência maiores com Gemini);
  desligue com `COLABORACAO=false` se preferir.
- Bancos criados por versões anteriores são atualizados na inicialização (colunas novas são acrescentadas).
