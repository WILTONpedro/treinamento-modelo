# LGPD no Baly AI Hub

Este documento descreve como o sistema trata dados pessoais de candidatos e o que ainda precisa ser
decidido ou validado pela Baly (jurídico e encarregado/DPO) antes do uso com candidatos reais.
Não substitui parecer jurídico.

## Papéis

- **Controladora:** Baly (decide finalidade e meios).
- **Operador:** Google, quando `IA_MODO=gemini` (processa currículos para triagem e análise).
- **Encarregado (art. 41):** definir e divulgar o contato no aviso de privacidade.

## Finalidade e bases legais (a validar)

| Dado | Finalidade | Base sugerida |
|---|---|---|
| Currículo, contato, cidade, histórico profissional | recrutamento e seleção para vagas da Baly | art. 7º, V (procedimentos preliminares a contrato, a pedido do titular que enviou o currículo) e/ou IX (legítimo interesse, com teste de balanceamento) |
| Menção a deficiência/CID (pasta PCD) | preenchimento de vagas PCD | art. 11, II, “a” (cumprimento de obrigação legal — cota da Lei 8.213/91, art. 93) |
| Dados de menores de 18 anos (pasta Jovem Aprendiz) | programa de aprendizagem | bases dos arts. 7º ou 11, sempre no melhor interesse do adolescente (art. 14; Enunciado CD/ANPD nº 1/2023) |

> **Atenção à regra de triagem “Jovem Aprendiz”:** a API classifica como aprendiz apenas menores de 18 anos.
> A aprendizagem legal vai de 14 a 24 anos (CLT, art. 428). Confirme se a restrição é intencional.

## O que o sistema faz

**Minimização**
- Listas de candidatos não mostram e-mail nem telefone; o contato aparece só no currículo aberto.
- O chat com os agentes não é armazenado no servidor.
- Logs técnicos não registram nomes de candidatos.
- A IA é instruída a não considerar características protegidas (idade, gênero, cor/raça, religião, estado
  civil, gravidez, orientação sexual, deficiência, aparência, origem) — LGPD art. 6º, IX e Lei 9.029/95.

**Decisão humana (art. 20)**
- Agentes só recomendam. Toda recomendação traz evidências (trechos literais do currículo), requisitos a
  verificar e justificativa. A decisão (entrevista, banco de talentos, não seguir) é registrada pelo recrutador.

**Segurança (art. 46)**
- Arquivos de currículo cifrados em disco (Fernet: AES-128 + HMAC); exclusão sobrescreve o arquivo.
- Senhas com scrypt; sessão em cookie `httpOnly` + `SameSite=Strict`; cabeçalho anti-CSRF; limite de tentativas de login.
- Papéis (recrutador, supervisor, admin); endpoint de triagem protegido por chave.
- Cabeçalhos de segurança (CSP, nosniff, frame-options).

**Registro das operações (art. 37)**
- Tabela `auditoria`: login, abertura de currículo, download de arquivo, listagem de candidatos, decisões,
  criação de vagas, ativação de agentes, exportação e eliminação de titulares — com usuário, data e IP.

**Retenção**
- Candidatos sem atividade há `RETENCAO_DIAS` (padrão 180) são anonimizados automaticamente: nome e contato
  apagados, arquivo e texto do currículo excluídos, análises esvaziadas, menções no histórico substituídas.
  Os números agregados continuam válidos. Defina o prazo na política de privacidade.

**Direitos do titular (art. 18)** — menu *Privacidade (LGPD)*, perfil admin:
- *Exportar*: todos os dados do titular em JSON (acesso e portabilidade).
- *Eliminar*: anonimização irreversível. Cada pedido fica registrado em `solicitacoes_titular`.
- Correção: os dados vêm do currículo enviado; um novo envio do titular atualiza o cadastro.

## Transferência internacional (arts. 33 a 36)

Com `IA_MODO=gemini`, currículos são enviados ao Google, o que em geral configura transferência internacional.

- Desde 23/08/2025, quem usa cláusulas contratuais para isso precisa adotar as cláusulas-padrão da ANPD
  (Resolução CD/ANPD nº 19/2024). Verifique com o jurídico se o contrato/DPA do Google atende ao requisito
  ou se outro mecanismo se aplica.
- **Use somente o plano pago da API Gemini.** Pelos termos do Google, no uso gratuito o conteúdo enviado pode
  ser usado para melhorar produtos e lido por revisores humanos; no pago, não é usado para isso e é tratado
  sob o adendo de processamento de dados.
- Alternativa sem transferência: `IA_MODO=demo` (heurístico, só para testes) ou um modelo hospedado no Brasil.

## Antes de entrar em produção

- [ ] Bases legais validadas pelo jurídico/encarregado (tabela acima).
- [ ] Aviso de privacidade para candidatos (no anúncio das vagas e na resposta automática do e-mail),
      citando uso de IA, finalidade, prazo de guarda, compartilhamento com o Google e canal do encarregado (art. 9º).
- [ ] Relatório de Impacto (RIPD, art. 38) — recomendado por envolver IA na seleção e dados sensíveis.
- [ ] Contrato/DPA com o Google revisado; plano pago ativo.
- [ ] Prazo de retenção definido (`RETENCAO_DIAS`).
- [ ] Regra de idade da pasta Jovem Aprendiz confirmada.
- [ ] Revisão periódica de vieses: amostrar recomendações por vaga e comparar com as decisões humanas.
