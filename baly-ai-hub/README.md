# Baly AI Hub — Escritório de Recrutamento Multiagente

Escritório isométrico em que agentes de IA apoiam o RH da Baly: a **Bia** (recepção) recebe e tria os
currículos que chegam por e-mail, o **Atlas** (supervisor) distribui o trabalho e os **analistas**
(Sofia, Flávio e os que forem ativados) comparam cada currículo com os requisitos da vaga e recomendam
candidatos com evidências. **A decisão final é sempre do recrutador.**

O escritório só mostra o que acontece de verdade: cada personagem sentado no computador tem uma tarefa
real na fila; os números vêm do banco; quem descansa está ocioso no sistema.

```
 e-mail ──► fluxo de e-mail atual ──► POST /triagem ─┐
                                                      ▼
            ┌──────────── fila de tarefas (banco) ───────────────┐
            │ TRIAGEM (Bia) → DISTRIBUIÇÃO (Atlas) → ANÁLISE (analistas) │
            └────────────────────────────────────────────────────┘
                     │ eventos (SSE, tempo real)
                     ▼
        escritório isométrico (frontend/) ── painéis, currículo, chat, decisões
```

## Estrutura

```
backend/
  app/main.py        API (FastAPI), trabalhador da fila, rotinas de retenção
  app/servicos.py    regras: recepção, Atlas, análises, estados, LGPD
  app/triagem.py     sua API de triagem, portada para o SDK novo do Gemini (mesmo prompt e categorias)
  app/ia.py          análise currículo × vaga e chat dos agentes (Gemini ou modo demonstração)
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

Entra com `admin@baly.local` / `demo1234`, cria três vagas de exemplo e alimenta 12 currículos fictícios
aos poucos, para ver a equipe trabalhando. As análises nesse modo são heurísticas e vêm marcadas como tal.

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

## Papéis

| Papel | Pode |
|---|---|
| recrutador | ver o escritório, agentes, candidatos e currículos; conversar com agentes; registrar decisões |
| supervisor | + criar/suspender/encerrar vagas, enviar currículos, ativar agentes, resolver pendências |
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
| Em deslocamento | caminhada visual disparada por evento real (buscar currículos, consultar vagas, ir/voltar do descanso) | anda pelos corredores |

O deslocamento é só visual: a tarefa já começou no servidor, o descanso nunca atrasa o processamento.

## Produção — checklist

- HTTPS na frente (Nginx/Caddy) e `COOKIE_SECURE=true`; desligue o buffer para `/api/eventos/stream` (SSE).
- PostgreSQL (`DATABASE_URL`) e backups; a `CHAVE_ARQUIVOS` guardada à parte, em cofre.
- **Um único processo** (`--workers 1`): o trabalhador da fila roda dentro da API.
- `TRIAGEM_API_KEY`, `SEGREDO_JWT` e `ADMIN_SENHA` fortes; troque a senha do admin após o primeiro acesso.
- Revise `LGPD.md` com o jurídico/encarregado antes de colocar candidatos reais.

## Limitações conhecidas

- Até 8 analistas ativos (8 estações na sala de análise).
- Leitura de PDF escaneado/imagem depende do Gemini (o modo demonstração não faz OCR).
- O chat não guarda histórico no servidor (minimização de dados); ele some ao recarregar a página.
