# LGPD no Escritório de Recrutamento

Este documento descreve como o sistema trata dados pessoais de candidatos e o que ainda precisa ser
decidido ou validado pela empresa (jurídico e encarregado/DPO) antes do uso com candidatos reais.
Não substitui parecer jurídico.

## Papéis

- **Controladora:** a empresa que usa o sistema para recrutar (decide finalidade e meios; configure `EMPRESA_NOME`).
- **Operadores:** Google, quando `IA_MODO=gemini` (processa currículos para triagem e análise), e Supabase,
  quando o banco fica lá (armazena os dados e os arquivos dos currículos).
- **Encarregado (art. 41):** definir e divulgar o contato no aviso de privacidade.

## Finalidade e bases legais (a validar)

| Dado | Finalidade | Base sugerida |
|---|---|---|
| Currículo, contato, cidade, histórico profissional | recrutamento e seleção para vagas da empresa | art. 7º, V (procedimentos preliminares a contrato, a pedido do titular que enviou o currículo) e/ou IX (legítimo interesse, com teste de balanceamento) |
| Menção a deficiência/CID (pasta PCD) | preenchimento de vagas PCD | art. 11, II, “a” (cumprimento de obrigação legal — cota da Lei 8.213/91, art. 93) |
| Dados de menores de 18 anos (pasta Jovem Aprendiz) | programa de aprendizagem | bases dos arts. 7º ou 11, sempre no melhor interesse do adolescente (art. 14; Enunciado CD/ANPD nº 1/2023) |

> **Atenção à regra de triagem “Jovem Aprendiz”:** a API classifica como aprendiz apenas menores de 18 anos.
> A aprendizagem legal vai de 14 a 24 anos (CLT, art. 428). Confirme se a restrição é intencional.

## O que o sistema faz

**Minimização**
- Listas de candidatos não mostram e-mail nem telefone; o contato aparece só no currículo aberto.
- O chat com os agentes (individual e o canal da equipe) não é armazenado no servidor.
- Pensamentos e conversas entre agentes citam o candidato só por primeiro nome e inicial ("Ana M."), ficam
  `RETENCAO_CONVERSAS_HORAS` (padrão 72 h) e são apagados antes disso se o titular for anonimizado.
- Logs técnicos não registram nomes de candidatos.
- A IA é instruída a não considerar características protegidas (idade, gênero, cor/raça, religião, estado
  civil, gravidez, orientação sexual, deficiência, aparência, origem) — LGPD art. 6º, IX e Lei 9.029/95.
  A mesma regra vale para os pensamentos, as consultas entre colegas e o canal da equipe.

**Decisão humana (art. 20)**
- Agentes só recomendam. Toda recomendação traz evidências (trechos literais do currículo), requisitos a
  verificar e justificativa. A decisão (entrevista, banco de talentos, não seguir) é registrada pelo recrutador.
- Quando um analista consulta um colega, a pergunta, a resposta e os trechos usados ficam na análise
  ("Conversa com colega"), para o recrutador saber como a recomendação foi formada. Uma indicação para outra
  vaga gera só uma nova análise — nunca uma decisão.
- "Contratar" e "dispensar" se referem aos agentes de IA (capacidade de processamento), não a candidatos.

**Segurança (art. 46)**
- Arquivos de currículo cifrados antes de serem guardados (Fernet: AES-128 + HMAC), no disco ou no banco;
  a chave (`CHAVE_ARQUIVOS`) fica só no servidor da aplicação — nem o provedor do banco consegue abrir os arquivos.
- Banco no Supabase: região São Paulo (os dados ficam no Brasil), conexão cifrada, schema próprio fora da API
  pública, papel de acesso exclusivo do sistema (sem o superusuário) e RLS ligado em todas as tabelas.
- Senhas com scrypt; sessão em cookie `httpOnly` + `SameSite=Strict` + `Secure` (HTTPS); cabeçalho anti-CSRF;
  limite de tentativas de login; senha temporária trocada obrigatoriamente no primeiro acesso; trocar a senha,
  mudar o papel ou desativar alguém encerra as sessões abertas dessa pessoa.
- Com `AMBIENTE=producao` o sistema não sobe sem HTTPS, segredos fortes e chave de criptografia dos arquivos.
- Papéis (recrutador, supervisor, admin); endpoint de triagem protegido por chave.
- Cabeçalhos de segurança (CSP, nosniff, frame-options).

**Registro das operações (art. 37)**
- Tabela `auditoria`: login, troca e redefinição de senha, criação e alteração de usuários, abertura de
  currículo, download de arquivo, listagem de candidatos, decisões, criação de vagas, contratação de agentes,
  teste da IA, exportação e eliminação de titulares — com usuário, data e IP.
- No PostgreSQL/Supabase o próprio banco impede alterar ou apagar registros da auditoria, inclusive pelo painel.

**Retenção**
- Candidatos sem atividade há `RETENCAO_DIAS` (padrão 180) são anonimizados automaticamente: nome e contato
  apagados, arquivo e texto do currículo excluídos, análises esvaziadas, menções no histórico substituídas.
  Os números agregados continuam válidos. Defina o prazo na política de privacidade.

**Direitos do titular (art. 18)** — *⚙ Administração → Privacidade (LGPD)*, perfil admin:
- *Exportar*: todos os dados do titular em JSON (acesso e portabilidade), incluindo o que os agentes
  falaram sobre ele enquanto essas falas existirem.
- *Eliminar*: anonimização irreversível. Cada pedido fica registrado em `solicitacoes_titular`.
- Correção: os dados vêm do currículo enviado; um novo envio do titular atualiza o cadastro.

## Transferência internacional (arts. 33 a 36)

Com `IA_MODO=gemini`, currículos são enviados ao Google, o que em geral configura transferência internacional.
Consultas entre colegas são novas chamadas com o mesmo currículo, para a mesma finalidade (a revisão
seguinte reaproveita a análise e não reenvia o currículo); o encaminhamento automático de currículos sem
pasta e o canal da equipe também enviam dados ao Google. O sistema registra quantidade de tokens e custo de
cada chamada (tabela `uso_ia`), nunca o conteúdo enviado.

- Desde 23/08/2025, quem usa cláusulas contratuais para isso precisa adotar as cláusulas-padrão da ANPD
  (Resolução CD/ANPD nº 19/2024). Verifique com o jurídico se o contrato/DPA do Google atende ao requisito
  ou se outro mecanismo se aplica.
- **Use somente o plano pago da API Gemini.** Pelos termos do Google, no uso gratuito o conteúdo enviado pode
  ser usado para melhorar produtos e lido por revisores humanos; no pago, não é usado para isso e é tratado
  sob o adendo de processamento de dados.
- Alternativa sem transferência: `IA_MODO=demo` (heurístico, só para testes) ou um modelo hospedado no Brasil.

## Banco de dados (Supabase)

O Supabase atua como operador de armazenamento. Pontos a validar: contrato/DPA do Supabase, região
(São Paulo, sa-east-1, já configurada), quem tem acesso ao painel do projeto (ative 2FA nas contas) e a
política de backups (no plano gratuito não há backup diário automático: agende `supabase/backup.sh` ou use o
plano Pro). Os arquivos dos currículos chegam ao banco já cifrados; o texto extraído e os dados de contato
ficam legíveis para quem tiver acesso ao painel do projeto — e nas cópias de segurança, que precisam do mesmo
cuidado (acesso restrito; as cópias com mais de 30 dias são apagadas pelo próprio script).

Integridade (art. 46): o banco recusa a exclusão de candidatos, currículos, análises, decisões e pedidos de
titulares, por engano ou por acesso indevido; a eliminação pedida pelo titular ou pela retenção é feita por
anonimização, que o sistema registra. Uma cópia de segurança guarda o banco como estava no dia: ao restaurar
uma cópia antiga, os pedidos de eliminação atendidos depois dela precisam ser refeitos — mantenha o registro
desses pedidos também fora do sistema (por exemplo, no canal do encarregado).

## Antes de entrar em produção

- [ ] Bases legais validadas pelo jurídico/encarregado (tabela acima).
- [ ] Aviso de privacidade para candidatos (no anúncio das vagas e na resposta automática do e-mail),
      citando uso de IA, finalidade, prazo de guarda, compartilhamento com o Google e canal do encarregado (art. 9º).
- [ ] Relatório de Impacto (RIPD, art. 38) — recomendado por envolver IA na seleção e dados sensíveis.
- [ ] Contrato/DPA com o Google revisado; plano pago ativo.
- [ ] Contrato/DPA com o Supabase revisado; acesso ao painel restrito e com 2FA; backups definidos
      (`supabase/backup.sh` agendado e um teste de restauração feito).
- [ ] Prazo de retenção definido (`RETENCAO_DIAS`).
- [ ] Regra de idade da pasta Jovem Aprendiz confirmada.
- [ ] Revisão periódica de vieses: amostrar recomendações por vaga e comparar com as decisões humanas.
