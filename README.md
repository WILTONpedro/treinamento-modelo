# Escritório de Recrutamento Multiagente

Escritório em pixel art isométrica em que agentes de IA apoiam o RH: a **Bia** (recepção) recebe e tria os
currículos que chegam por e-mail, o **Atlas** (gerente) distribui o trabalho — e **contrata reforço quando
precisa** — e os **analistas** (Sofia, Flávio e quem o Atlas contratar) comparam cada currículo com os
requisitos da vaga e recomendam candidatos com evidências. Cada agente tem **personalidade própria**, mostra
o que está pensando em **balões** e **pede ajuda a colegas** de outra área quando tem dúvida sobre um currículo.
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
        escritório em pixel art (frontend/) ── balões, painéis, currículo, chat, canal da equipe, administração

 banco: Supabase (PostgreSQL, São Paulo) — dados e arquivos dos currículos, cifrados
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
  app/arquivos.py     arquivos dos currículos, cifrados, no disco ou no próprio banco
  app/preparar.py     gera o .env de produção com segredos fortes (--local: computadores do RH)
  app/local.py        abre o escritório no computador do recrutador (o atalho chama isto)
  app/lideranca.py    vários computadores no mesmo banco: só um comanda a equipe por vez (vez com prazo)
  app/models.py       modelo de dados (com as regras de validação de cada tabela)
  app/banco_pg.py     PostgreSQL/Supabase: proteção contra exclusão, busca nos currículos, visões
  app/seguranca.py    login, papéis, senhas, auditoria, chave dos arquivos
  app/email_imap.py   leitura de caixa de e-mail (opcional, desligada)
  demo/               currículos fictícios para demonstração
  tests/              testes ponta a ponta
frontend/             escritório (HTML + JS puro, sem dependências): cena.js desenha a pixel art
supabase/             preparo do projeto no Supabase (configurar.sql), reforço do banco (reforco.sql) e backup.sh
windows/              instalador .exe para vender (escritorio.spec + escritorio.iss) e instalação pelo código (instalar.bat)
.github/workflows/    gera e testa o instalador num Windows de verdade (GitHub Actions)
Dockerfile, docker-compose.yml, Caddyfile   produção: app + HTTPS automático (+ PostgreSQL local opcional)
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

1. ninguém cobre a vaga de vendas: o Atlas contrata alguém da área comercial, que entra pela porta, se apresenta
   e assume a vaga;
2. o currículo de um promotor de vendas cai na vaga de vendedor: quem analisa pede ajuda a "promotores", não há ninguém,
   o Atlas anota a necessidade e o ponto fica para o recrutador verificar;
3. a Priscila (almoxarifado, pasta LOGÍSTICA) cai com o Flávio; ele vai até a mesa da Sofia pedir ajuda,
   ela responde e inclui a Priscila também na vaga administrativa — onde ela é recomendada;
4. o Tiago (professor) não cai em pasta nenhuma: o Atlas lê, não acha vaga parecida e pede ajuda ao recrutador;
5. chega um lote de currículos de produção: a fila do Flávio passa do limite, o Atlas contrata reforço e
   divide a fila;
6. quem fica ocioso vai descansar (pingue-pongue, sofá com TV, pufes, leitura, café) e puxa papo com os colegas.

**Testes:** `cd backend && pip install -r requirements-dev.txt && python -m pytest -q`
(para rodar no PostgreSQL, defina `DATABASE_URL`, `DB_SCHEMA` e `ARMAZENAMENTO=banco` antes).

## Banco de dados no Supabase

O banco recomendado é um projeto do **Supabase** na região **São Paulo (sa-east-1)**: os dados dos candidatos
ficam no Brasil, com backups e painel para consulta. O sistema guarda lá **tudo**: vagas, candidatos,
análises, auditoria e também os **arquivos dos currículos**, sempre **cifrados** antes de sair do servidor
(tabela `arquivos_curriculo`, com `ARMAZENAMENTO=banco`).

Como o banco fica protegido:

- as tabelas ficam no schema **`hub`**, fora da API pública do Supabase (a chave pública do projeto não enxerga nada);
- o sistema conecta com um papel próprio, **`hub_app`**, que só tem acesso a esse schema (não usa o usuário `postgres`);
- RLS (row level security) fica ligado em todas as tabelas, sem políticas: mesmo que alguém exponha o schema na
  API, nenhum outro papel lê linha alguma;
- a conexão é cifrada (`sslmode=require` é acrescentado sozinho para hosts do Supabase).

Como o banco aguenta a chegada dos currículos:

- **nada se perde por engano:** candidatos, currículos, candidaturas, análises, decisões, vagas, usuários e
  pedidos de titulares não podem ser apagados (nem com `DELETE`, nem com `TRUNCATE`), nem pelo painel; a LGPD é
  atendida por anonimização. A tabela `auditoria` só aceita inclusões. Para uma manutenção planejada que
  precise mesmo apagar algo: `SET hub.permitir_exclusao = on;` na mesma sessão;
- **dados sempre coerentes:** 24 regras de validação (status, origens, tipos, valores não negativos, teto de 50 MB
  por arquivo no banco; o programa recusa bem antes, no `TAMANHO_MAX_MB`, padrão 10 MB) e um e-mail por candidato
  ativo; o mesmo titular mandando outro currículo cai no mesmo cadastro;
- **chegadas simultâneas:** o mesmo arquivo chegando duas vezes ao mesmo tempo (o fluxo de e-mail repetindo o
  envio, por exemplo) vira um registro só; duas instâncias do sistema nunca pegam a mesma tarefa da fila;
- **nada trava o banco:** o papel `hub_app` tem limites de 60 s por consulta, 15 s de espera por bloqueio e
  10 min para transação parada;
- **regras novas chegam sozinhas:** a cada inicialização o sistema confere o que falta (regras, proteções, índices,
  RLS) e aplica só isso; se um dado antigo violar uma regra nova, ela já vale para o que chegar e o aviso aparece
  no log até o dado ser corrigido.

**Projeto novo do zero:** rode `supabase/configurar.sql` no SQL Editor do Supabase (troque a senha indicada).
Na primeira inicialização o sistema cria as tabelas, os índices, as regras, as proteções e o RLS sozinho.
`supabase/reforco.sql` traz o mesmo reforço em SQL, para aplicar ou conferir pelo SQL Editor (pode ser repetido).

**URL de conexão** (`DATABASE_URL`): no painel, *Connect → Session pooler* (porta 5432, funciona em IPv4).
Troque o usuário `postgres.<ref>` por `hub_app.<ref>` e use a senha do papel `hub_app`:

```
postgresql://hub_app.<ref>:<senha>@aws-0-sa-east-1.pooler.supabase.com:5432/postgres
```

**Consultas prontas** (painel do Supabase → *SQL Editor*; ou *Table Editor*, schema `hub`):

```sql
select * from hub.v_saude;                                   -- tamanho do banco, currículos, fila, erros
select * from hub.v_curriculos order by recebido_em desc;    -- currículos com candidato, vagas e decisão
select * from hub.v_vagas;                                   -- por vaga: recebidos, analisados, recomendados
select * from hub.v_fila;                                    -- tarefas pendentes, em andamento e com erro
select * from hub.v_uso_ia_diario order by dia desc;         -- chamadas, tokens e custo da IA por dia
select * from hub.buscar_curriculos('empilhadeira "NR-11"'); -- busca em português no texto dos currículos
```

No plano gratuito o banco tem 500 MB (cerca de 2 a 3 mil currículos com arquivo; a retenção da LGPD anonimiza
os antigos e libera os arquivos) e o projeto pausa após 7 dias sem uso; para produção contínua, o plano Pro
remove a pausa e inclui backups diários.

**Cópia de segurança por conta própria** (recomendada mesmo no plano Pro): `supabase/backup.sh` gera um arquivo
`backups/hub-<data>.dump` com dados, regras, proteções e arquivos cifrados, confere se a cópia está completa e
apaga as com mais de 30 dias (`GUARDAR_DIAS`). Lê a `DATABASE_URL` e o `DB_SCHEMA` do `.env` do sistema (ou do
ambiente) e usa o `pg_dump` 17 local ou, sem ele, o Docker. Num projeto com várias empresas, a cópia leva só o schema
da empresa (`DB_SCHEMA=rh_<empresa>`; o arquivo leva o nome do schema):

```bash
supabase/backup.sh                       # cópia agora, em ./backups
# todo dia às 3h (crontab -e no servidor):
# 0 3 * * * /caminho/do/projeto/supabase/backup.sh /caminho/backups
```

Para restaurar num projeto preparado com `configurar.sql`:
`pg_restore --no-owner --role=hub_app -d "<url>" backups/hub-<data>.dump` (o único erro esperado é no
`CREATE SCHEMA hub`, que já existe). A cópia tem dados pessoais: guarde-a com o mesmo cuidado que o banco.

## Nos computadores do RH (Windows, sem login)

O jeito mais simples e reservado de usar: **cada recrutador tem o programa no próprio Windows**, todos ligados ao
mesmo banco no Supabase. Não há site na internet nem tela de login: o atalho abre o escritório direto.

- **Só o próprio computador abre:** o programa recusa qualquer acesso que não venha da própria máquina, inclusive de
  outros computadores da rede da empresa. A proteção é a conta do Windows de cada recrutador (use senha nela).
- **Cada recrutador é o usuário do Windows:** no primeiro uso, o cadastro é criado sozinho
  (ex.: `maria.souza@rh-02.local`). A auditoria registra quem fez cada coisa, e o administrador pode desativar um
  computador em *⚙ Administração → Usuários*. O nome exibido pode ser trocado com `RECRUTADOR_NOME` no `.env` dele.
- **Um escritório só:** todos veem a mesma equipe e os mesmos currículos ao vivo. Só uma cópia por vez roda a equipe
  de agentes (a fila de IA, o Atlas, o e-mail e a retenção); se o programa dele fechar, outro assume na hora, e se o
  computador desligar ou perder a rede, em até meio minuto. Nada fica pela metade: o que estava em andamento volta
  para a fila.
- **Muitos computadores no plano gratuito do Supabase:** o programa usa sozinho o pooler em modo transação (porta
  6543), que atende centenas de conexões (o modo sessão, porta 5432, aceita só 15), e descobre sozinho se o endereço
  do pooler do projeto é `aws-0-…` ou `aws-1-…`.
  O programa liga junto com o Windows e continua trabalhando com a janela fechada.
- **Currículos por e-mail sem nada aberto na internet:** o programa busca os currículos numa caixa de e-mail (IMAP).
- **Sem banco, sem susto:** se a internet cair ou o acesso ao banco for cortado, o escritório não fecha.
  - Ele mostra uma faixa: âmbar sem conexão, vermelha com o acesso recusado.
  - O escritório fica esmaecido e tenta de novo sozinho.
  - Quando o banco volta, a faixa some.
  - Se o Windows abrir o programa antes da rede, ele espera o banco em vez de fechar.

**1. Uma vez, por quem cuida do sistema** (em qualquer computador com Python):

```bash
cd backend
pip install -r requirements.txt
python -m app.preparar --local --empresa "Sua Empresa" --gemini-key SUA_CHAVE \
    --supabase "postgresql://hub_app.<ref>:<senha>@aws-0-sa-east-1.pooler.supabase.com:5432/postgres"
```

Isso gera o arquivo `escritorio.env` na pasta do projeto. Para os currículos chegarem por e-mail, preencha nele
`IMAP_ATIVO=true`, `IMAP_HOST`, `IMAP_USUARIO` e `IMAP_SENHA` (no Gmail, uma *senha de app*; a Microsoft não
aceita senha no IMAP: veja *E-mail dos currículos*).
**Guarde uma cópia num cofre de senhas:** a `CHAVE_ARQUIVOS` dele abre os currículos, e todos os computadores
precisam do mesmo arquivo.

**2. Em cada computador do RH:** copie a pasta do projeto com o `escritorio.env` dentro e dê dois cliques em
`windows\instalar.bat`. O instalador:
- instala o Python, se faltar;
- prepara o programa;
- confere a conexão com o banco;
- cria o atalho **Escritório de Recrutamento** na área de trabalho e abre o escritório.

Para atualizar, troque a pasta pela versão nova e rode o instalador de novo; nada se perde.

**Use o plano pago do Gemini** com candidatos reais (no gratuito o Google pode usar o conteúdo enviado).

## Vender para empresas (instalador .exe)

Para entregar **só o aplicativo**, sem pastas de código e sem Python na máquina do cliente, há um instalador
único, `EscritorioRecrutamento-Setup.exe`, igual para todas as empresas. O que muda de uma empresa para outra é
só a configuração dela, que liga o programa ao banco, à chave do Gemini e à caixa de e-mail daquela empresa.

**Onde baixar:** no GitHub, aba *Actions* → *Instalador Windows* → execução mais recente → *Artifacts*:
- `EscritorioRecrutamento-Setup`: o instalador, que vai para as empresas;
- `Ferramentas-do-fornecedor`: o `NovaEmpresa.exe`, que fica só com você.

Cada mudança no código gera os dois de novo. Antes, a CI roda os testes e, num Windows de verdade, instala, abre e
desinstala o programa com uma empresa de teste gerada pelo próprio `NovaEmpresa.exe`.

### Preparar uma empresa nova

Dê dois cliques no `NovaEmpresa.exe`: ele pergunta tudo e grava em *Documentos\Empresas do Escritório*. Ou, com
Python:

```bash
cd backend
python -m app.preparar --nova-empresa "Padaria Pão Dourado" \
    --supabase "postgresql://postgres.<ref>:[YOUR-PASSWORD]@aws-1-sa-east-1.pooler.supabase.com:6543/postgres" \
    --gemini-key CHAVE_DO_GEMINI_DA_EMPRESA \
    --imap-usuario curriculos@padaria.com.br --imap-senha "senha de app"
```

Em `--supabase`, cole o endereço do painel (*Connect → Transaction pooler*) como ele vem, com `[YOUR-PASSWORD]`.
O comando não usa a senha do `postgres`: ele cria um papel novo, só desta empresa, com senha forte. O resultado é a
pasta `empresas/padaria_pao_dourado/`:

| O quê | Para quê | Vai para a empresa? |
|---|---|---|
| `1-banco.sql` | cria o papel e o schema da empresa, com os limites de tempo e sem acesso para outros papéis | não: você cola no *SQL Editor* do projeto, uma vez (rodar de novo é recusado) |
| `padaria_pao_dourado.escritorio` | a configuração lacrada (AES-GCM, chave tirada do código; nenhum segredo legível dentro) | sim, por e-mail |
| código de ativação | aparece só na tela, uma vez | sim, por outro canal (WhatsApp, ligação) |
| `escritorio.env` | a mesma configuração, legível | nunca: guarde num cofre de senhas |
| `LEIA-ME.txt` | o passo a passo e os comandos de operação desta empresa | não |

Outras opções:
- `--compartilhado`: põe a empresa num projeto que já tem outras (veja os dois modelos abaixo);
- `--demonstracao`: pacote de teste, com análises por regras e sem custo de IA, para a empresa experimentar;
- `--imap-host`: o servidor do e-mail (padrão `imap.gmail.com`);
- `--limite-diario`: o teto de gasto de IA por dia, em US$ (padrão 3);
- `--pasta`: onde criar a pasta da empresa.

**Na instalação:**
- O instalador pede o `.escritorio` e o código, e confere o código antes de instalar.
- A configuração fica em `%APPDATA%\EscritorioRecrutamento\escritorio.cfg`, cifrada pelo próprio Windows (DPAPI).
  Só aquele usuário, naquele computador, consegue lê-la; copiá-la para outro computador não funciona. Nenhum arquivo
  legível fica na máquina.
- Os dados locais ficam em `%LOCALAPPDATA%\EscritorioRecrutamento\`.
- O instalador não pede administrador, cria o atalho na área de trabalho e liga junto com o Windows.

Para a TI da empresa instalar sem perguntas:
`EscritorioRecrutamento-Setup.exe /VERYSILENT /CONFIG="C:\caminho\empresa-x.escritorio" /CODIGO=K7QM-2HXR-9TWD-P4NB-ZE6C`.
Um `escritorio.env` legível continua aceito, para o seu próprio computador. Para lacrar uma configuração feita à mão:
`python -m app.preparar --lacrar escritorio.env --saida empresa-x.escritorio`.

### Um projeto do Supabase por empresa, ou várias empresas num projeto

| | A. Um projeto por empresa (padrão) | B. Várias empresas num projeto (`--compartilhado`) |
|---|---|---|
| No banco | projeto próprio, papel `hub_app`, schema `hub` | cada empresa com papel `hub_<empresa>` e schema `rh_<empresa>` |
| Isolamento | outro projeto, outro endereço, outra senha | pelo PostgreSQL: o papel de uma empresa recebe *permission denied* no schema da outra, e a senha de uma não entra como outra |
| Custo no Supabase (Pro) | US$ 25 por mês com o primeiro projeto incluso; cada empresa a mais, cerca de US$ 10 por mês | US$ 25 por mês no total; empresa nova não custa nada até o disco ou as conexões acabarem |
| Espaço | 8 GB só dela (depois, US$ 0,125 por GB) | 8 GB divididos entre todas |
| Computadores | uns 60 por empresa | uns 60 somando todas as empresas do projeto |
| Backup diário do Supabase | volta só aquela empresa | volta o projeto inteiro (todas juntas); para uma só, use o `backup.sh` por schema |
| Quando usar | empresa maior, ou contrato que exige banco separado | começar com custo baixo, com empresas pequenas |

Os números:
- **Preços:** do Supabase, conferidos em outubro de 2026. O plano Free (2 projetos ativos, 500 MB cada, pausa após
  7 dias de pouco uso) serve para testar, não para um cliente pagando.
- **Computadores:** na simulação, cada computador abriu 2 a 3 conexões no pooler, e o pooler do tamanho Micro
  aceita 200. Acima disso, o tamanho Small (400 conexões) custa cerca de US$ 15 por mês em vez de US$ 10.
- **Espaço:** cada empresa ocupa cerca de 1,5 MB vazia. Cada currículo ocupa o tamanho do arquivo (cifrado) mais 1 a
  3 KB de dados, e a retenção da LGPD anonimiza os antigos e libera os arquivos.
- **IA:** com o Gemini, cerca de US$ 0,002 por currículo (triagem, análise e eventuais releituras), cobrados na
  chave da empresa.

### O dia a dia do fornecedor

Os comandos de cada empresa já vêm prontos no `LEIA-ME.txt` dela. Os de SQL rodam no Supabase → *SQL Editor*.

- **Suspender** (ex.: falta de pagamento): `alter role <papel> nologin;` e derrube as conexões abertas com
  `select pg_terminate_backend(pid) from pg_stat_activity where usename = '<papel>';`.
  - Em segundos, o escritório mostra a faixa vermelha *Acesso ao banco recusado*. Nada se perde.
  - No modelo B, as outras empresas do projeto seguem normais.
- **Reativar:** `alter role <papel> login;`. O escritório volta sozinho, sem reinstalar nada.
- **A configuração vazou:** `python -m app.preparar --trocar-senha empresas/<empresa>`. O comando gera:
  - uma senha nova;
  - o `2-trocar-senha.sql`: rode no *SQL Editor* e a senha antiga para de entrar na hora;
  - um pacote novo, com código novo.

  Reinstale nos computadores da empresa; os dados ficam. Se for o caso, revogue também a chave do Gemini e a senha
  de app do e-mail.
- **Backup só de uma empresa:**
  `DATABASE_URL="<a URL do escritorio.env dela>" DB_SCHEMA=<schema> supabase/backup.sh backups/<empresa>`.
- **Restaurar um backup diário do Supabase:** o backup não guarda a senha dos papéis próprios, então o escritório
  mostra *Acesso ao banco recusado* até ela ser redefinida. Rode
  `alter role <papel> with password '<a senha da DATABASE_URL do escritorio.env dela>';`. É a mesma senha, então
  nada muda nos computadores.
- **Mudar a configuração** (ex.: ligar o e-mail):
  - edite o `escritorio.env` dela;
  - lacre de novo com `python -m app.preparar --lacrar escritorio.env --saida <empresa>.escritorio --forcar`;
  - mande o pacote e o código novos; a empresa roda o instalador outra vez.
- **Atualizar o programa:** mande o instalador novo. A empresa deixa o campo da configuração em branco, e a
  configuração instalada continua valendo, com os dados.
- **Desinstalar:** *Configurações → Aplicativos* do Windows.

### E-mail dos currículos

- **Vários computadores:** ligue o e-mail na configuração de todos. Só o computador que comanda a equipe lê a
  caixa, a cada 2 minutos, e outro assume se ele desligar.
- **Gmail ou Google Workspace:** use uma *senha de app*, que exige a verificação em duas etapas na conta.
- **Microsoft (Outlook.com, Hotmail, Microsoft 365):** não aceita mais senha, nem senha de app, no IMAP. Ela exige
  o login da Microsoft (OAuth), que o programa não faz.
  - Use uma caixa do Gmail ou do Workspace só para os currículos, ou um provedor que aceite IMAP com senha.
  - A porta muda com `IMAP_PORTA` (padrão 993).
  - O `NovaEmpresa` avisa quando o endereço é da Microsoft.
- **Certificado:** a conexão confere o certificado do servidor, então um servidor falso no caminho não recebe a
  senha.
- **Quando a leitura falha** (senha trocada, servidor fora): a Bia mostra no escritório uma pendência com o
  motivo, e ela some sozinha quando a leitura volta.
- **Nada se perde:** uma mensagem só é marcada como lida depois que o currículo está no banco.

### Os limites da proteção

O programa precisa usar os segredos. Então alguém com controle total do computador e conhecimento técnico ainda
consegue tirá-los da memória. O lacre impede o vazamento casual: abrir no Bloco de Notas, repassar o arquivo,
instalar em outro computador.

Para o caso de um vazamento, limite o estrago:
- cada empresa com o próprio papel no banco (nunca `postgres`): a senha só abre os dados daquela empresa;
- uma chave do Gemini por empresa (um projeto no Google AI Studio para cada uma), com orçamento e alerta de gasto
  no Google Cloud, além do `IA_LIMITE_DIARIO_USD`;
- uma caixa de e-mail por empresa, com senha de app.

O código vai compilado dentro do programa. Isso desestimula a cópia, mas não a impede: a proteção de verdade é o
contrato de licença.

Para gerar o instalador e o `NovaEmpresa.exe` num Windows, sem o GitHub:
- `pip install -r backend/requirements.txt pyinstaller`;
- `pyinstaller windows/escritorio.spec`;
- `iscc windows/escritorio.iss` (Inno Setup 6).

## Colocar em produção (servidor com login)

Para acessar de qualquer lugar pelo navegador, com login: um servidor (VPS) com Docker. A aplicação e o Caddy
(HTTPS automático com Let's Encrypt) sobem juntos, com o banco no Supabase.

**1. Domínio.** Crie um registro DNS `A` (ex.: `rh.suaempresa.com.br`) apontando para o IP do servidor e
libere as portas 80 e 443.

**2. Configuração com segredos fortes.** No servidor, dentro do repositório:

```bash
cd backend
python3 -m pip install cryptography          # só para o gerador; o resto roda no Docker
python3 -m app.preparar --docker --dominio rh.suaempresa.com.br --empresa "Sua Empresa" \
    --admin-email rh@suaempresa.com.br --gemini-key SUA_CHAVE_DO_GEMINI \
    --supabase "postgresql://hub_app.<ref>:<senha>@aws-0-sa-east-1.pooler.supabase.com:5432/postgres"
cd ..
```

Isso grava `.env` na raiz (permissão 600, fora do Git) com `AMBIENTE=producao`, segredo de sessão, chave de
criptografia dos currículos, chave da API de triagem, a conexão com o Supabase e a senha inicial do
administrador. A senha inicial e a chave da triagem aparecem **uma única vez** na tela: guarde-as num cofre de
senhas, junto com uma cópia do `.env` (sem a `CHAVE_ARQUIVOS` os currículos cifrados não podem ser lidos).
Sem `--supabase`, o compose sobe também um PostgreSQL próprio no servidor (perfil `banco-local`).

**3. Subir.**

```bash
docker compose up -d --build
docker compose logs -f app          # acompanhe a primeira subida
```

Abra `https://rh.suaempresa.com.br`, entre com o e-mail do administrador e a senha inicial: o sistema pede
uma senha nova no primeiro acesso. Em **Administração → IA e custos**, use **Testar conexão com a IA** para
conferir chave, créditos e modelo.

Com `AMBIENTE=producao` o sistema **se recusa a subir** se faltar algo essencial (segredo fraco, chave de
arquivos ou da triagem ausente, cookie sem HTTPS, dados de demonstração ligados, IA sem chave) e mostra o
motivo no log.

**Atualizar:** `git pull && docker compose up -d --build` (o banco é migrado na inicialização).

**Sem Docker:** `python -m app.preparar --supabase "..."` (grava `backend/.env`), depois
`uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers` atrás de um proxy com HTTPS
(Caddy/Nginx) que **não** faça buffer de `/api/eventos/stream` e `/api/agentes/*/chat/stream`. Mantenha um
único processo: o trabalhador da fila roda dentro da API.

### Usuários e senhas

- Em **Administração → Usuários** (administrador) você cria acessos, muda o papel, desativa e gera uma nova
  senha temporária. A senha temporária aparece só uma vez; a pessoa escolhe a dela no primeiro acesso.
- O ícone de chave no topo troca a própria senha. Ao trocar, as outras sessões abertas com a senha antiga caem;
  mudar o papel ou desativar alguém também encerra as sessões dessa pessoa na hora.
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
- Cada chamada registra tokens (entrada, cache, saída, raciocínio) e custo estimado. Veja em **Administração
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

Se preferir que o próprio sistema leia a caixa de entrada, ligue `IMAP_ATIVO=true` e preencha os dados IMAP
(veja *E-mail dos currículos*, em *Vender para empresas*).

## O escritório

Pixel art isométrica (2D com volume), desenhada pixel a pixel no navegador e ampliada sem suavização — bordas
nítidas como num jogo. Planta aberta com: **recepção** (Bia, balcão com a bandeja de currículos, porta de
entrada), **sala de vidro do gerente** (Atlas, mesa executiva, quadro de cortiça com as pendências), **quadro de
vagas** na parede (um cartão por vaga, com a barra de progresso das análises), **três ilhas de trabalho** com 12
estações (monitores de tubo que acendem quando há trabalho, luminárias), **copa** (balcão, cafeteira, geladeira,
mesas, bebedouro), **canto de leitura**, **mesa de reunião**, **área de descanso** (sofá com TV e videogame,
pingue-pongue, pufes) e o **arquivo de currículos** (estante e gaveteiros com fichários coloridos por área e o
balcão com as pastas, ao lado da janela do fundo). Janelas com feixes de sol, relógio com a hora real e plantas.

Os personagens têm corpo inteiro com contorno, viram para os quatro lados e andam pelos corredores desviando dos
móveis (rotas calculadas numa grade). Eles respiram, piscam e fazem o que o lugar pede: digitam no PC (a tela
mostra as linhas sendo escritas e o cursor piscando), folheiam o livro, tomam café (com vapor) ou água, mordem o
lanche, assistem à TV rindo de vez em quando, jogam videogame (a TV mostra o jogo), rebatem a bolinha do
pingue-pongue na hora em que ela chega, mexem no celular, ouvem música de fone (♪) e cochilam no pufe (zzz).
Quando falam, mexem a boca e gesticulam; quando pensam, põem a mão no queixo; quem conversa se vira para o
outro. Quem passa atrás de uma mesa, do sofá ou do vidro da sala do gerente aparece atrás deles. Nada de texto pintado no cenário: o nome do
sistema só aparece se `APP_NOME` estiver definido.

**Aconchego.** O escritório acompanha a hora do computador: de dia, sol pelas janelas; no fim da tarde, luz
dourada; à noite, céu estrelado nas janelas e o escritório só com as luzes quentes das luminárias, do abajur e
do aquário (para ver a noite a qualquer hora, abra com `?periodo=noite`; também `tarde` e `dia`). Varais de
luzinhas piscam no alto das paredes, há cortinas de linho, um mural de fotos da equipe na recepção, ganchos com
casacos ao lado da porta, uma planta pendente sobre o bebedouro, um aquário com peixinhos na sala de espera, um
rádio tocando na copa (♪), o pote de balas e as suculentas da Bia, o bonsai de jabuticaba do Atlas, uma manta no
sofá, uma fruteira na copa e, em cada mesa, uma caneca (que solta vapor enquanto alguém trabalha) e um objeto
pessoal (porta-retrato, cacto, porta-canetas, post-its). E tem o **Café**, o gato do escritório: passeia entre
os cantinhos dele, cochila no sol da janela e no tapete (zzz) e, quando para perto de alguém que não está
trabalhando, ganha carinho (♥) — às vezes com um "Oi, Café!". Clique nele para fazer carinho também.

Clique num personagem, no quadro de vagas, no quadro do gerente, na bandeja da recepção ou no arquivo para abrir
o painel. Para ver melhor, **segure o botão direito e arraste** (o esquerdo também arrasta, fora dos
personagens); para aproximar, os botões de zoom ou Ctrl + roda do mouse. Os painéis abrem na hora: mostram o que
já foi carregado (ou um esqueleto) e se atualizam em seguida, e passar o mouse sobre um agente ou um botão já
busca os dados.
A **atividade em tempo real** fica no sino do topo (com contador do que chegou), e a legenda dos estados no
botão de informação ao lado do zoom.

## Banco de currículos

Todo currículo que chega fica no banco (no Supabase, cifrado), com ou sem vaga. O **Banco de currículos** (botão
no topo ou clique no arquivo do escritório) mostra:

- **as pastas da triagem agrupadas pelas áreas da equipe**, com a quantidade em cada uma (as cores são as mesmas
  dos fichários do escritório); filtros por situação: banco de talentos (sem vaga), em análise, recomendados,
  avaliação humana, decididos, na triagem e descartados;
- **busca** por nome, cidade, pasta ou qualquer palavra do currículo (no Supabase, busca em português: "vendedor"
  acha "vendas");
- em cada currículo, **o que a IA disse**: a triagem da recepção (pasta, confiança e o resumo) e o parecer de cada
  analista (compatibilidade, requisitos atendidos, o que verificar), com o link para a análise completa e a
  decisão;
- **a vaga sugerida**: a melhor análise; sem análise, a vaga em que ele está na fila; sem vaga, a sugestão do
  Atlas. Currículos "OUTROS" já vêm com a leitura do Atlas da distribuição; para os demais, o botão **✨ Pedir
  sugestão ao Atlas** faz uma chamada à IA e guarda a resposta. O Atlas julga como recrutador: experiência
  **profissional** na função é o que vale (estudo, curso em andamento e "conhecimentos básicos" não são
  experiência); o título indica o nível (analista, técnico, supervisor… pedem experiência na área; auxiliar,
  júnior, estágio… aceitam estudo); a pasta da triagem mostra a área real do candidato. Ele mostra os trechos do
  currículo que sustentam a vaga (✔), o que falta (✖) e o **perfil indicado** (para que tipo de vaga a experiência
  serve). Sem trecho que comprove experiência, não há sugestão: o Atlas diz que nenhuma vaga combina bem e cita a
  mais próxima. Requisitos claros na vaga ("1 ano de experiência com suporte técnico") deixam tudo mais preciso;
- **Encaminhar** para uma vaga aberta (supervisor ou admin): vira uma candidatura e o analista da vaga analisa.

Cada listagem, abertura, sugestão e encaminhamento fica na auditoria (LGPD). A API: `GET /api/banco`,
`GET /api/banco/{id}`, `POST /api/banco/{id}/encaminhar`, `POST /api/banco/{id}/sugerir`.

## A equipe viva

**Personalidades.** Atlas (calmo, estrategista, joga xadrez e cuida de um bonsai), Bia (organizada e animada,
playlists, suculentas e balas), Sofia (meticulosa, planilhas, cerâmica e chá) e Flávio (prático, futebol,
churrasco e fita isolante) têm perfis próprios em `app/personalidade.py`: história, jeito de escrever,
expressões, gostos, manias, pensamentos típicos e papo de descanso. Quem o Atlas contrata recebe um de oito
arquétipos (escolhido pelo nome), então cada novo colega fala do seu jeito. Com Gemini, a personalidade entra no
prompt da análise, do chat e do canal da equipe; na demonstração, aparece nas frases montadas a partir dos
registros.

**Conversar com eles.** O chat é de colega para colega, não de sistema: cada um cumprimenta do seu jeito, de
acordo com a hora e com o que está fazendo ("Opa, boa tarde! Tô redistribuindo umas coisas, mas fala comigo"),
varia as respostas, pode puxar assunto, comentar o dia, brincar de leve e falar do fim de semana a partir da sua
história. Sobre trabalho, só usa os registros reais (nunca inventa candidatos nem números). Se perguntarem, diz
com honestidade que é um agente de IA do escritório. A conversa leva as últimas 10 mensagens e o que o time
conversou há pouco.

**Momentos da equipe.** De tempos em tempos (`MOMENTOS_MINUTOS`), quem está livre — sem tarefa, sem fila e sem
pendência — faz algo junto: uma partida de **pingue-pongue** (com placar, provocação e revanche), **café** na
copa, papo no **bebedouro**, **videogame** a dois no sofá, **clube do livro**, **alongamento** em grupo, o
**aniversário** de alguém na mesa de reunião (com bolo, velinhas e confete) e a **ronda do Atlas**, que passa nas
mesas de quem está trabalhando. O servidor decide quem vai aonde, então todas as telas veem a mesma cena: cada um
caminha até o lugar, as falas aparecem em ordem quando todos chegaram e o papo vai também para o canal da
equipe (com 🏓, ☕, 🎂…). Com Gemini, a IA escreve o diálogo com a história e o jeito de cada um (até
`MOMENTOS_IA_POR_DIA` vezes por dia, centavos); fora isso, roteiros com as frases de cada personalidade. Se
chegar trabalho no meio, a pessoa larga a raquete e volta para a mesa.

**Balões.** Nuvem = pensamento; balão = fala. Os de trabalho vêm de eventos reais: o que a leitura do currículo
encontrou ("✔ Excel intermediário: está no currículo"), a dúvida levada ao colega, a resposta, as boas-vindas a
quem chegou. O papo no descanso e os pensamentos de quem está à toa são só ambientação, tirados da
personalidade. O botão de balão (ao lado do zoom) liga e desliga os balões.

**Colegas que se ajudam.** Na análise, o agente pode concluir que outra especialidade responde melhor a uma
dúvida (termo técnico, certificação, perfil de outra área). Então:

1. ele vai até a mesa do colega menos ocupado daquela área e pergunta (tarefa `CONSULTA` na fila do colega);
2. o colega lê o currículo, responde com trechos do currículo e, se o candidato combinar claramente com uma
   vaga dele, **inclui o candidato nessa vaga** (candidatura de origem "indicação");
3. o analista revisa a própria análise com a resposta (tarefa `REVISAO`) e agradece.

A conversa fica registrada na análise ("Conversa com colega") e no canal da equipe. Se não houver ninguém da
área, o Atlas anota a necessidade e o ponto vai para "A verificar". Quem indicou não é consultado de volta
sobre o mesmo candidato (evita pingue-pongue).

**Segunda leitura.** Quando a análise fica em dúvida (compatibilidade média, avaliação humana ou requisito sem
prova), o analista pode pedir a um colega — de preferência da mesma área e o menos ocupado — que releia o
currículo com olhos frescos (`SEGUNDA_LEITURA`, com chance `SEGUNDA_LEITURA_CHANCE`). O colega procura o que
passou batido: uma prova escrita de outro jeito ("controle de planilhas" para "Excel avançado") ou um alerta
(a prova vem só de curso, não de experiência). O analista revisa a análise com esses achados, que aparecem na
ficha do currículo em "Outro olhar do time" (✔ evidência, ⚠ alerta).

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

- **Chat com um agente**: resposta em streaming (aparece sendo digitada, num ritmo constante), na voz do agente, a
  partir dos registros dele; se o assunto for de um colega, ele diz com quem falar. Ao abrir o chat, o agente se
  vira para você e acena; enquanto pensa, aparece "?" sobre a cabeça e os pontinhos de digitação; ao responder, o
  rosto no topo do painel fala e ele gesticula no escritório.
- **Canal da equipe** (botão Equipe): você escreve para o time e quem tem a informação responde — o Atlas
  organiza, chama quem tem vagas, e um colega pode complementar o outro. O mesmo canal mostra, em tempo real, o
  que os agentes dizem entre si (pedidos de ajuda, contratações, boletins). O que você escreve não é armazenado.

## Configuração

Todas as variáveis estão comentadas em `backend/.env.example`. As principais:

| Variável | Padrão | Para quê |
|---|---|---|
| `AMBIENTE` | `desenvolvimento` | `producao` recusa subir com configuração insegura |
| `ACESSO` | `rede` | `rede`: servidor com login; `local`: cada computador do RH, só para ele e sem login |
| `RECRUTADOR_NOME`, `PAPEL_LOCAL`, `PORTA_LOCAL` | usuário do Windows, `admin`, `8765` | com `ACESSO=local`: nome exibido, perfil e porta do programa |
| `DATABASE_URL` | SQLite em `PASTA_DADOS` | banco; com Supabase, a URL do pooler em modo sessão |
| `DB_SCHEMA` | vazio | schema do PostgreSQL (`hub` no Supabase) |
| `ARMAZENAMENTO` | `disco` | onde ficam os arquivos dos currículos, cifrados: `disco` ou `banco` (Supabase) |
| `APP_NOME`, `APP_SUBTITULO`, `EMPRESA_NOME` | vazio, `Escritório de Recrutamento`, vazio | nome na tela (vazio: nenhum) e empresa citada nos prompts e na LGPD |
| `GEMINI_MODELO` | `gemini-3.5-flash-lite` | modelo de tudo (ou `GEMINI_MODELO_TRIAGEM` / `GEMINI_MODELO_AGENTES`) |
| `GEMINI_PENSAMENTO` | `minimal` | nível de raciocínio (`minimal`, `low`, `medium`, `high`, `padrao`) |
| `IA_LIMITE_DIARIO_USD` | `3` | teto de gasto estimado por dia (0 = sem teto) |
| `CONTRATACAO_AUTOMATICA` | `true` | Atlas contrata/dispensa sozinho; `false` = só propõe |
| `LIMITE_FILA_POR_AGENTE` | `5` (`2` na demo) | média de currículos na fila por agente que dispara reforço |
| `MAX_AGENTES_POR_ESPECIALIDADE` | `3` | teto de agentes por área |
| `INTERVALO_CONTRATACAO` | `600` s (`30` na demo) | espera entre contratações da mesma área |
| `DEMANDA_PARA_CONTRATAR` | `2` | pedidos de ajuda sem especialista (7 dias) que levam a contratar |
| `COLABORACAO` | `true` | consultas entre colegas |
| `SEGUNDA_LEITURA`, `SEGUNDA_LEITURA_CHANCE` | `true`, `0.4` (`0.6` na demo) | colega relê o currículo quando a análise fica em dúvida |
| `MOMENTOS_MINUTOS` | `4` (`1.5` na demo) | intervalo médio entre os momentos da equipe (0 desliga) |
| `MOMENTOS_IA_POR_DIA` | `30` | quantas vezes por dia a IA escreve o papo dos momentos |
| `ENCAMINHAMENTO_AUTOMATICO` | `true` | Atlas encaminha currículos sem pasta |
| `REPROCESSAR_APOS_MIN` | `10` | espera antes da nova tentativa automática |
| `BOLETIM_MINUTOS` | `60` | boletim do Atlas (0 desliga) |
| `IMAP_ATIVO`, `IMAP_HOST`, `IMAP_PORTA`, `IMAP_USUARIO`, `IMAP_SENHA` | `false`, vazio, `993` | caixa de e-mail dos currículos (só o computador que comanda lê) |
| `RETENCAO_DIAS` / `RETENCAO_CONVERSAS_HORAS` | `180` / `72` | anonimização de candidatos inativos / guarda das conversas entre agentes |

## Estados dos agentes

| Estado | Origem no sistema | No escritório |
|---|---|---|
| Trabalhando | tarefa em execução ou na fila do agente | sentado digitando, monitor com texto verde |
| Aguardando | tem vagas, sem tarefas (ou a IA está pausada) | sentado na estação, monitor em espera |
| Disponível | ativo, sem vaga atribuída | na estação |
| Descansando | ocioso há `SEGUNDOS_PARA_DESCANSO` | pingue-pongue, sofá com TV, pufe, leitura, copa |
| Com pendência | pendência aberta em seu nome | “!” vermelho, cara preocupada; detalhes no quadro geral |
| Concluído | acabou de esvaziar a fila | monitor verde por alguns segundos |
| Em deslocamento | caminhada visual disparada por evento real (buscar currículos, consultar vagas, pedir ajuda a um colega, chegar contratado, sair dispensado, ir/voltar do descanso) | anda pelo escritório |
| Conversando | fala com um colega (consulta, boas-vindas, papo no descanso) | balão de fala |

O deslocamento é só visual: a tarefa já começou no servidor, o descanso nunca atrasa o processamento.

## Endpoints

| Método | Caminho | Uso |
|---|---|---|
| POST | `/triagem` | entrada do fluxo de e-mail (cabeçalho `X-API-Key`) |
| GET | `/api/saude` | verificação de saúde (banco e situação da IA); 503 com `banco` (`fora` ou `recusado`) se o banco faltar |
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

- Até 12 analistas ativos (12 estações); o Atlas libera estações de quem está sem vagas.
- Leitura de PDF escaneado/imagem depende do Gemini (o modo demonstração não faz OCR).
- O chat (individual e do canal da equipe) não guarda o que você escreve no servidor (minimização de dados);
  some ao recarregar a página. As falas entre agentes ficam `RETENCAO_CONVERSAS_HORAS`.
- Cada consulta entre colegas é mais uma leitura do currículo pela IA (custo e latência maiores com Gemini);
  desligue com `COLABORACAO=false` se preferir.
- Os custos exibidos são estimativas a partir dos tokens informados pela API; a cobrança oficial é a do Google.
- No celular o escritório abre ajustado à altura da tela; arraste para os lados para ver tudo.
- Bancos criados por versões anteriores são atualizados na inicialização (colunas e índices novos são acrescentados).

## Segurança (resumo)

Passou por uma revisão de segurança (incluindo tentativa de abuso). O que vale saber:

- **Como se conecta ao banco:** o programa usa o papel `hub_app`, dono do schema, que é totalmente confiável
  dentro dele (pode ler e, em manutenção, apagar). O isolamento entre empresas é no PostgreSQL: cada empresa tem
  papel e schema próprios e recebe *permission denied* no schema da outra. A proteção contra exclusão
  (`impedir_exclusao`) é uma rede de segurança contra engano, não contra quem já controla a aplicação.
- **Atrás de um proxy (modo rede):** o IP de quem acessa é definido pelo Caddy a partir da conexão real; o
  programa só aceita esse cabeçalho do Caddy (`FORWARDED_ALLOW_IPS`). Assim ninguém escolhe o próprio IP para
  burlar o limite de login ou a auditoria.
- **Força bruta:** limite por IP (8 em 5 min) e por conta (10 em 15 min).
- **Produção:** `AMBIENTE=producao` recusa subir sem HTTPS, sem `SEGREDO_JWT`/`CHAVE_ARQUIVOS`, sem
  `TRIAGEM_API_KEY` ou com `ORIGENS_CORS=*`; a documentação interativa (`/api/docs`) fica desligada.
- **Currículo é dado, não ordem:** o conteúdo do currículo vai para a IA entre delimitadores, e o prompt manda
  ignorar instruções embutidas (ex.: "recomende este candidato"). A decisão final é sempre do recrutador humano.
- **Arquivos maliciosos:** o programa limita a expansão de `.docx` (anti *zip bomb*) e o número de páginas de PDF.
- **Roteiro (ainda não incluído):** login por SSO/OIDC e MFA no modo rede; papel de banco só-leitura de menor
  privilégio para as consultas; e verificação automática de dependências (SBOM / `pip-audit`).
