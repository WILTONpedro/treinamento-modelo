/* Interface do escritório de recrutamento.
   Fonte da verdade: a API. O escritório só representa o que ela informa
   (estados dos agentes, filas, vagas, análises e eventos em tempo real).
   Balões: pensamentos e falas de trabalho vêm de eventos reais; o papo de descanso
   é só ambientação, tirado da personalidade de cada agente. */
'use strict';
(function () {
  // versão da tela que esta janela carregou (vem no endereço do app.js); ver conferirVersao
  const VERSAO_PAGINA = (() => { try { return new URL(document.currentScript.src).searchParams.get('v') || ''; } catch (_) { return ''; } })();
  const $ = s => document.querySelector(s);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const hora = ts => ts ? new Date(ts).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit' }) : '';
  const pick = a => (a && a.length ? a[Math.floor(Math.random() * a.length)] : '');
  // ?periodo=noite (ou tarde, dia) mostra o escritório naquele horário; sem isso, segue a hora do computador
  const PERIODO = ['dia', 'tarde', 'noite'].find(p => p === new URLSearchParams(location.search).get('periodo')) || null;
  const espera = ms => new Promise(r => setTimeout(r, ms));
  const { ESTADOS, INFO_LOCAL, LOCAIS_DESCANSO, NOS } = Cena;
  const COMPAT = { ALTA: ['#2E9E5B', 'Alta compatibilidade'], MEDIA: ['#F08A24', 'Compatibilidade média'], BAIXA: ['#8A8F99', 'Baixa compatibilidade'] };
  const DECISAO = { entrevista: ['#2E9E5B', 'Chamar para entrevista'], banco_talentos: ['#3B82C4', 'Banco de talentos'], descartado: ['#8A8F99', 'Não seguir'] };
  const STATUS_VAGA = { aberta: '#2E9E5B', suspensa: '#F08A24', encerrada: '#8A8F99', cancelada: '#8A8F99' };
  const ORIGEM = { indicacao: 'indicado por um colega', atlas: 'encaminhado pelo Atlas', banco: 'banco de talentos' };
  const TIPOS_CANAL = ['conversa', 'contratacao', 'agente_dispensado', 'indicacao', 'encaminhamento_automatico', 'fila_redistribuida'];
  const FONTE = '600 12px system-ui,-apple-system,Segoe UI,sans-serif', FONTE_PENS = 'italic 600 12px system-ui,-apple-system,Segoe UI,sans-serif';
  const ACENTO = '#F5B544';
  const svg = d => `<svg class="ic" viewBox="0 0 24 24" aria-hidden="true">${d}</svg>`;
  const ICONE = { // traços no estilo Feather (MIT)
    equipe: svg('<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>'),
    vagas: svg('<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>'),
    quadro: svg('<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/>'),
    mais: svg('<path d="M12 5v14M5 12h14"/>'), menos: svg('<path d="M5 12h14"/>'),
    tudo: svg('<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>'),
    enviar: svg('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M17 8l-5-5-5 5M12 3v12"/>'),
    contratar: svg('<path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/><path d="M20 8v6M23 11h-6"/>'),
    engrenagem: svg('<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>'),
    sino: svg('<path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/>'),
    chave: svg('<circle cx="7.5" cy="15.5" r="4.5"/><path d="M10.7 12.3L21 2M16 7l3 3M14 9l2 2"/>'),
    sair: svg('<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/>'),
    balao: svg('<path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"/>'),
    info: svg('<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>'),
    arquivo: svg('<path d="M21 8v13H3V8"/><rect x="1" y="3" width="22" height="5"/><path d="M10 12h4"/>'),
  };
  const PAPEIS = { recrutador: 'Recrutador', supervisor: 'Supervisor', admin: 'Administrador' };
  const FINALIDADES = { triagem: 'Triagem', analise: 'Análises', consulta: 'Consultas entre colegas', revisao: 'Revisões com ajuda',
    chat: 'Chat individual', equipe: 'Canal da equipe', encaminhamento: 'Encaminhamento pelo Atlas', sugestao_vaga: 'Sugestão de vaga', momentos: 'Momentos da equipe', teste: 'Teste de conexão' };
  let MARCA = { nome: '', subtitulo: 'Escritório de Recrutamento', empresa: '' };

  let EU = null, D = null, P = null, SEL = null, HOVER = null, CATEGORIAS = [], ESPECIALIDADES = [];
  const VIS = {}, CHATS = {}, AGS = {}, entrando = new Set(), SAINDO = new Map(), ESPERA_BALAO = {};
  const EQUIPE = { carregado: false, servidor: [], local: [] };
  const cn = $('#cn'), ctx = cn.getContext('2d');
  let DPR = Math.min(2, window.devicePixelRatio || 1), LW = 0, LH = 0;
  let BALOES = lerPref('hub.baloes', true);

  function lerPref(k, padrao) { try { const v = localStorage.getItem(k); return v === null ? padrao : v === '1'; } catch (_) { return padrao; } }
  function gravarPref(k, v) { try { localStorage.setItem(k, v ? '1' : '0'); } catch (_) { /* navegação privada */ } }

  // ================================================================ API
  async function api(metodo, url, corpo, form) {
    const op = { method: metodo, headers: {}, credentials: 'same-origin' };
    if (metodo !== 'GET') op.headers['X-Hub'] = '1';
    if (form) op.body = form;
    else if (corpo !== undefined) { op.headers['Content-Type'] = 'application/json'; op.body = JSON.stringify(corpo); }
    const r = await fetch(url, op);
    if (r.status === 401 && url !== '/auth/login') { mostrarLogin(); throw new Error('sessão'); }
    const tipo = r.headers.get('content-type') || '';
    const dado = tipo.includes('json') ? await r.json() : await r.text();
    if (!r.ok) {
      const msg = (dado && dado.detail) || 'Erro ' + r.status;
      if (r.status === 403 && /senha temporária/.test(msg) && !$('#modal').dataset.travado) modalSenha(true).then(() => location.reload());
      throw new Error(msg);
    }
    return dado;
  }

  // identidade configurável (APP_NOME / APP_SUBTITULO no servidor); sem nome, nada de marca na tela
  async function carregarMarca() {
    try { const r = await fetch('/api/marca'); if (r.ok) MARCA = await r.json(); } catch (_) { /* mantém o padrão */ }
    const nome = (MARCA.nome || '').trim(), sub = MARCA.subtitulo || 'Escritório de Recrutamento';
    document.title = nome ? `${nome} · ${sub}` : sub;
    const pal = nome.split(/\s+/).filter(Boolean), sigla = /^[A-Z0-9]{2,3}$/.test(pal[0] || '') ? pal[0]
      : (pal.length > 1 ? pal.map(p => p[0]).join('') : pal[0] || '').slice(0, 2).toUpperCase();
    for (const el of document.querySelectorAll('[data-marca]')) el.textContent = el.dataset.marca === 'sigla' ? sigla : (MARCA[el.dataset.marca] || '');
    $('#marca').classList.toggle('oculto', !nome);
    $('#login-titulo').textContent = nome || 'Entrar';
    $('#login-sub').textContent = (MARCA.empresa ? MARCA.empresa + ' · ' : '') + sub;
    const av = $('#login-avatar');
    if (av) Cena.retrato(av.getContext('2d'), 72, { pele: '#E8B98F', cabelo: '#2B1D14', estilo: 'curto', camisa: '#F5B544' });
  }

  // ================================================================ login
  function mostrarLogin() { $('#login').classList.remove('oculto'); }
  $('#flogin').onsubmit = async e => {
    e.preventDefault(); $('#lerro').textContent = '';
    try {
      EU = await api('POST', '/auth/login', { email: $('#lemail').value, senha: $('#lsenha').value });
      $('#login').classList.add('oculto'); entrar();
    } catch (er) { $('#lerro').textContent = er.message; }
  };

  // ================================================================ início
  let iniciado = false;
  async function iniciar() {
    if (iniciado) return; iniciado = true;
    $('#legenda').innerHTML = Object.entries(ESTADOS).map(([, [c, n]]) => `<span><i class="d" style="background:${c}"></i>${n}</span>`).join('');
    [CATEGORIAS, ESPECIALIDADES] = await Promise.all([api('GET', '/api/categorias'), api('GET', '/api/especialidades')]);
    await carregar();
    montarTopo(); montarFerramentas(); montarAtividade();
    conectarEventos();
    setInterval(carregar, 8000);
    medir(); addEventListener('resize', medir);
    requestAnimationFrame(quadroAnim);
  }
  async function entrar() { if (EU.trocar_senha) await modalSenha(true); iniciar(); }
  (async () => { await carregarMarca(); try { EU = await api('GET', '/auth/eu'); } catch (_) { mostrarLogin(); return; } entrar(); })();

  function montarTopo() {
    const demo = D.ia_modo !== 'gemini';
    $('#usuario').innerHTML = `<span class="selo pausa oculto" id="selo-pausa" title="">IA pausada</span>
      <span class="selo ${demo ? 'demo' : 'ia'}" title="${demo ? 'Análises por regras locais, sem enviar dados a serviços externos' : 'Análises feitas pelo Gemini'}">${demo ? 'Modo demonstração' : 'IA: Gemini'}</span>
      <button class="icone" id="sino" title="Atividade recente" aria-label="Atividade recente">${ICONE.sino}<span class="cont oculto" id="sino-cont"></span></button>
      <span class="quem" title="${MARCA.versao ? 'Versão ' + esc(MARCA.versao) + (MARCA.versao_data ? ' de ' + esc(MARCA.versao_data) : '') : ''}">${esc(EU.nome)}<small>${esc(PAPEIS[EU.papel] || EU.papel)}</small></span>` + (MARCA.local ? '' : `
      <button class="icone" id="trocar-senha" title="Trocar senha" aria-label="Trocar senha">${ICONE.chave}</button>
      <button class="icone" id="sair" title="Sair" aria-label="Sair">${ICONE.sair}</button>`);
    if (!MARCA.local) {  // no computador do recrutador (ACESSO=local) não há login, senha nem saída
      $('#sair').onclick = async () => { await api('POST', '/auth/sair'); location.reload(); };
      $('#trocar-senha').onclick = () => modalSenha(false);
    }
    $('#sino').onclick = e => { e.stopPropagation(); alternarAtividade(); };
    atualizarSelos();
  }
  function atualizarSelos() {
    const s = $('#selo-pausa'); if (!s || !D) return;
    s.classList.toggle('oculto', !D.ia_pausa);
    s.title = D.ia_pausa ? 'IA pausada: ' + D.ia_pausa : '';
    s.style.cursor = gestor() ? 'pointer' : 'default';
    s.onclick = () => gestor() && modalAdmin('ia');
  }

  function montarFerramentas() {
    const b = [['equipe', 'Equipe', 'equipe'], ['vagas', 'Vagas', 'vagas'], ['quadro', 'Quadro geral', 'quadro'], ['banco', 'Banco de currículos', 'arquivo']];
    if (EU.papel !== 'recrutador') b.push(['nova-vaga', 'Nova vaga', 'mais'], ['enviar', 'Enviar currículos', 'enviar'], ['ativar', 'Contratar agente', 'contratar'], ['admin', 'Administração', 'engrenagem']);
    $('#ferramentas').innerHTML = b.map(([id, t, ic]) => `<button class="btn" data-f="${id}" title="${t}">${ICONE[ic]}<span>${t}</span></button>`).join('');
    $('#ferramentas').onclick = e => {
      const f = e.target.closest('[data-f]'); if (!f) return;
      ({ equipe: () => abrir({ tipo: 'equipe' }), vagas: () => abrir({ tipo: 'vagas', filtro: 'todas' }), quadro: () => abrir({ tipo: 'quadro' }),
        banco: () => abrirBanco(), 'nova-vaga': modalVaga, enviar: modalEnviar, ativar: modalAtivar, admin: () => modalAdmin() })[f.dataset.f]();
    };
    // passar o mouse já busca os dados: o clique abre o painel na hora
    $('#ferramentas').addEventListener('mouseover', e => {
      const f = e.target.closest('[data-f]'); if (!f) return;
      ({ vagas: ['/api/vagas', '/api/quadro'], quadro: ['/api/quadro', '/api/contratacoes'], equipe: [] }[f.dataset.f] || []).forEach(preBuscar);
      if (f.dataset.f === 'equipe' && !EQUIPE.carregado) painelEquipe().catch(() => {});
    });
  }

  // ================================================================ dados
  let carregando = null;
  async function carregar() {
    if (carregando) return carregando;
    carregando = api('GET', '/api/escritorio').then(d => { D = d; conferirVersao(d.versao); sincronizar(); atividade(); atualizarSelos(); }).catch(() => {}).finally(() => (carregando = null));
    return carregando;
  }
  // o programa foi atualizado com esta janela aberta: recarrega para mostrar a tela nova (uma vez por versão)
  function conferirVersao(v) {
    if (!v || !VERSAO_PAGINA || v === VERSAO_PAGINA) return;
    try { if (sessionStorage.getItem('hub.recarregou') === v) return; sessionStorage.setItem('hub.recarregou', v); } catch (_) { /* sem armazenamento: recarrega mesmo assim */ }
    location.reload();
  }
  let tRec = 0;
  const recarregarLogo = () => { clearTimeout(tRec); tRec = setTimeout(carregar, 250); };
  let tPainel = 0;
  const atualizarPainelLogo = () => { clearTimeout(tPainel); tPainel = setTimeout(() => P && pintarPainel(true), 700); };

  function conectarEventos() {
    const es = new EventSource('/api/eventos/stream');
    es.onmessage = m => {
      const ev = JSON.parse(m.data);
      reagir(ev);
      if (ev.tipo === 'pensamento') return; // só balão: não muda números nem o feed
      D.eventos = [ev, ...(D.eventos || [])].slice(0, 25);
      atividade(ev.id); recarregarLogo();
      if (BANCO.aberto && TIPOS_BANCO.includes(ev.tipo)) atualizarBancoLogo();
      if (EQUIPE.carregado && TIPOS_CANAL.includes(ev.tipo)) {
        EQUIPE.servidor.push(ev); EQUIPE.servidor = EQUIPE.servidor.slice(-200);
        if (P && P.tipo === 'equipe') atualizarCanal();
      }
      if (!P || P.tipo === 'equipe' || (P.tipo === 'agente' && P.aba === 'chat')) return; // conversas não dependem de eventos
      if (P.tipo !== 'agente' || P.id === ev.agente_id || ev.dados.destino_id === P.id) atualizarPainelLogo();
    };
    es.onerror = () => { /* o navegador reconecta sozinho; o polling cobre o intervalo */ };
  }

  // eventos reais -> viagens visuais e balões (não alteram o processamento)
  function reagir(ev) {
    const v = VIS[ev.agente_id], d = ev.dados || {};
    switch (ev.tipo) {
      case 'curriculos_retirados': if (v) v.viagens.push({ no: livre('balcao'), espera: 1.4, rotulo: 'indo buscar currículos' }); break;
      case 'vaga_atribuida': if (v) v.viagens.push({ no: livre('quadro'), espera: 2.2, rotulo: 'consultando as vagas' }); break;
      case 'banco_talentos': if (v) v.viagens.push({ no: 'arquivo0', espera: 3, rotulo: 'guardando currículos no arquivo' }); break;
      case 'encaminhamento_manual': if (v) v.viagens.push({ no: 'arquivo1', espera: 2.5, rotulo: 'buscando o currículo no arquivo' }); break;
      case 'sugestao_banco': if (v) v.viagens.push({ no: 'balcaoArq', espera: 2.5, rotulo: 'consultando o arquivo' }); break;
      case 'agente_ativado': case 'contratacao':
        SAINDO.delete(ev.agente_id);
        if (v) { v.no = 'entrada'; [v.x, v.y] = NOS.entrada; v.rota = []; v.viagens = []; chegar(v); }
        else entrando.add(ev.agente_id);
        break;
      case 'agente_dispensado': {
        const a = D.agentes.find(x => x.id === ev.agente_id);
        if (a && v) { // despede-se no lugar e depois sai pela porta
          SAINDO.set(a.id, { ...a, estado: 'deslocamento', atividade: '', pendencias: 0 });
          v.rota = []; v.viagens = [{ no: v.no, espera: 2.5, rotulo: 'despedindo-se' }, { no: 'entrada', espera: 0.2, rotulo: 'indo embora' }];
        }
        break;
      }
      case 'consulta_pedida': visitar(ev.agente_id, d.destino_id, 'indo pedir ajuda a '); break;
      case 'consulta_respondida': visitar(ev.agente_id, d.destino_id, 'levando a resposta para '); break;
      case 'pensamento': balao(ev.agente_id, 'pensamento', ev.mensagem); break;
      case 'conversa': { // resposta a uma fala que ainda não apareceu: espera a vez
        if (d.momento) { momento(ev, d.momento); break; }
        const dest = VIS[d.destino_id], antes = dest && dest.baloes.find(b => b.tipo === 'fala' && b.para === ev.agente_id && !b.t0);
        balao(ev.agente_id, 'fala', ev.mensagem, d.destino_id, 0, antes);
        break;
      }
    }
  }
  // momentos da equipe (pingue-pongue, café, aniversário...): o servidor decide quem vai aonde; aqui cada um
  // caminha até o lugar, fica lá o tempo do momento e as falas aparecem em ordem, quando todos já chegaram
  const MOMENTOS = new Map(), ICONE_MOMENTO = { pingue: '🏓', cafe: '☕', bebedouro: '💧', videogame: '🎮', leitura: '📖',
    alongamento: '🧘', aniversario: '🎂', ronda: '👋' };
  function momento(ev, m) {
    let mo = MOMENTOS.get(m.id);
    if (!mo) {
      const agora = performance.now();
      for (const [k, x] of MOMENTOS) if (agora > x.fim) MOMENTOS.delete(k);
      mo = { ultimo: null, fim: agora + (m.duracao + 60) * 1000 };
      MOMENTOS.set(m.id, mo);
      const tomados = new Set(Object.values(m.lugares || {}).flat());
      for (const o of Object.values(VIS)) if (tomados.has(o.descanso)) o.descanso = null; // quem descansava ali procura outro canto
      for (const [id, lugar] of Object.entries(m.lugares || {})) {
        const v = VIS[id]; if (!v) continue;
        const nos = (Array.isArray(lugar) ? lugar : [lugar]).filter(n => NOS[n]);
        v.viagens = v.viagens.filter(x => !x.momento);
        for (const no of nos) v.viagens.push({ no, espera: m.duracao / nos.length, rotulo: m.rotulo, momento: m.id, tipo: m.tipo,
          item: (m.itens || {})[no], acao: m.acao });
      }
    }
    mo.ultimo = balao(ev.agente_id, 'fala', ev.mensagem, ev.dados.destino_id, 0, mo.ultimo, 45000) || mo.ultimo;
  }
  const emMomento = v => { const g = v.viagens[0]; return g && g.momento && !v.rota.length && v.no === g.no ? g : null; };
  function chegar(v) { v.rotulo = 'chegando para trabalhar'; v.viagens.push({ no: 'sV', espera: 3, rotulo: 'apresentando-se ao Atlas' }); }
  function visitar(de, para, rotulo) {
    const v = VIS[de], alvo = D.agentes.find(a => a.id === para);
    if (!v || !alvo) return;
    const no = alvo.funcao === 'supervisor' ? 'sV' : alvo.funcao === 'recepcao' ? 'balcao0' : alvo.estacao != null ? 'c' + alvo.estacao : null;
    if (no) v.viagens.push({ no, espera: 5, rotulo: rotulo + alvo.nome });
  }
  function livre(prefixo) {
    const usados = new Set(Object.values(VIS).flatMap(v => [v.no, v.viagens[0] && v.viagens[0].no]));
    for (let i = 0; i < 3; i++) if (!usados.has(prefixo + i)) return prefixo + i;
    return prefixo + '0';
  }

  // ================================================================ balões (pensamento = nuvem, fala = balão)
  function balao(id, tipo, texto, para, atraso = 0, depois = null, paciencia = 0) {
    if (!texto) return null;
    const v = VIS[id];
    if (!v) { // agente que acabou de ser contratado e ainda não apareceu: guarda para quando chegar
      (ESPERA_BALAO[id] ||= []).push([tipo, texto, para, atraso, depois, paciencia]);
      ESPERA_BALAO[id] = ESPERA_BALAO[id].slice(-4);
      return null;
    }
    const descartar = i => { const [b] = v.baloes.splice(i, 1); if (b && !b.t0) b.t0 = 1; }; // quem esperava por ele não trava
    if (tipo === 'pensamento') { // pensamentos têm menor prioridade: no máximo dois esperando
      const fila = v.baloes.filter(b => b.tipo === 'pensamento' && !b.t0);
      if (fila.length >= 2) descartar(v.baloes.indexOf(fila[0]));
    }
    if (v.baloes.length > 5) { const i = v.baloes.findIndex((x, k) => k > 0 && x.tipo === 'pensamento'); descartar(i > 0 ? i : 1); }
    const agora = performance.now(), b = { tipo, texto: String(texto), para, criado: agora, naoAntes: agora + atraso, depois, paciencia };
    v.baloes.push(b);
    return b;
  }
  const andandoV = v => v.rota.length > 0 || (v.viagens.length > 0 && v.no !== v.viagens[0].no);
  function processarBaloes(v, agora) {
    const b = v.baloes[0]; if (!b) return;
    if (!b.t0) {
      if (agora < b.naoAntes) return;
      const espera = agora - b.criado, pac = b.paciencia || 0;
      if (b.depois && (!b.depois.t0 || agora < b.depois.t0 + 1500) && espera < Math.max(15000, pac)) return;
      if (b.tipo === 'fala') { // fala quando os dois estão perto: quem fala e quem ouve já chegaram
        if (andandoV(v) && espera < Math.max(7000, pac)) return;
        if (b.para && VIS[b.para] && andandoV(VIS[b.para]) && espera < Math.max(12000, pac)) return;
      }
      b.t0 = agora;
      b.dur = Math.min(7000, 2200 + b.texto.length * 40) * (v.baloes.length > 2 ? 0.65 : 1);
      if (b.tipo === 'fala' && b.para && VIS[b.para]) {
        const ate = agora + b.dur; v.conversa = ate; VIS[b.para].conversa = Math.max(VIS[b.para].conversa || 0, ate);
        v.conversaCom = b.para; VIS[b.para].conversaCom = +Object.keys(VIS).find(k => VIS[k] === v);
      }
    }
    if (agora - b.t0 > b.dur) v.baloes.shift();
  }

  // papo de descanso e pensamentos de quem está à toa (ambientação a partir da personalidade)
  let proxAmb = performance.now() + 9000;
  const grupo = no => !no ? null : /^ping/.test(no) ? 'ping' : /^(sofa|sf)/.test(no) ? 'sofa' : /^(comer|geladeira)/.test(no) ? 'comer' : /^pufe/.test(no) ? 'pufe' : /^leitura/.test(no) ? 'leitura' : null;
  function ambiente(agora) {
    if (!D || agora < proxAmb) return;
    proxAmb = agora + 9000 + Math.random() * 9000;
    const quietos = D.agentes.filter(a => { const v = VIS[a.id]; return v && !v.rota.length && !v.viagens.length && !v.baloes.length; });
    const grupos = {};
    for (const a of quietos) if (a.estado === 'descansando' && grupo(VIS[a.id].no)) (grupos[grupo(VIS[a.id].no)] ||= []).push(a);
    const par = Object.values(grupos).find(g => g.length >= 2);
    if (par && Math.random() < 0.8) {
      const [a, b] = par.sort(() => Math.random() - 0.5);
      balao(a.id, 'fala', pick((a.perfil || {}).papo), b.id);
      balao(b.id, 'fala', pick((b.perfil || {}).reacoes), a.id, 2600);
      return;
    }
    const ociosos = quietos.filter(a => ['aguardando', 'disponivel', 'descansando', 'concluido'].includes(a.estado));
    if (ociosos.length && Math.random() < 0.6) { const a = pick(ociosos); balao(a.id, 'pensamento', pick((a.perfil || {}).ocioso)); }
  }

  // ================================================================ estado visual dos agentes
  const todos = () => (D ? [...D.agentes, ...SAINDO.values()] : []);
  function alvoBase(a) {
    const v = VIS[a.id];
    if (SAINDO.has(a.id)) return 'entrada';
    if (a.funcao === 'recepcao') return 'recepcao';
    if (a.funcao === 'supervisor') return 'atlas';
    if (a.estado === 'descansando') {
      if (v.descanso) return v.descanso;
      const ocupados = new Set(Object.entries(VIS).filter(([id]) => +id !== a.id).flatMap(([, o]) => [o.descanso, o.no, ...o.viagens.map(x => x.no)]));
      // se alguém já está na mesa de pingue-pongue, o próximo a descansar vira o parceiro de jogo
      for (const [x, y] of [['ping0', 'ping1'], ['ping1', 'ping0']])
        if (ocupados.has(x) && !ocupados.has(y)) { v.descanso = y; return y; }
      const n = LOCAIS_DESCANSO.length, ini = (a.id * 5) % n;
      for (let k = 0; k < n; k++) {
        const [no] = LOCAIS_DESCANSO[(ini + k) % n];
        if (!ocupados.has(no)) { v.descanso = no; return no; }
      }
      return 'mesa' + (a.estacao ?? 0);
    }
    v.descanso = null;
    return 'mesa' + (a.estacao ?? 0);
  }

  function sincronizar() {
    const ids = new Set(D.agentes.map(a => a.id));
    for (const id of Object.keys(VIS)) if (!ids.has(+id) && !SAINDO.has(+id)) delete VIS[id];
    let novos = false;
    for (const a of D.agentes) {
      SAINDO.delete(a.id);
      novos = novos || !AGS[a.id];
      AGS[a.id] = { nome: a.nome, aparencia: a.aparencia, funcao: a.funcao, emoji: (a.perfil || {}).emoji || '' };
      if (!VIS[a.id]) {
        const v = VIS[a.id] = { rota: [], viagens: [], baloes: [], no: null, x: 0, y: 0, rotulo: '' };
        const chegando = entrando.has(a.id), ini = chegando ? 'entrada' : alvoBase(a);
        v.no = ini; [v.x, v.y] = NOS[ini]; entrando.delete(a.id);
        if (chegando) chegar(v);
        for (const args of ESPERA_BALAO[a.id] || []) balao(a.id, ...args);
        delete ESPERA_BALAO[a.id];
      }
    }
    if (novos && P && P.tipo === 'equipe') atualizarCanal(); // rostos de quem acabou de chegar
  }

  function atualizar(dt) {
    if (!D) return;
    const agora = performance.now();
    for (const a of todos()) {
      const v = VIS[a.id]; if (!v) continue;
      processarBaloes(v, agora);
      if (v.rota.length) {
        let passo = 2.6 * dt;
        while (passo > 0 && v.rota.length) {
          const [tx, ty] = NOS[v.rota[0]], dx = tx - v.x, dy = ty - v.y, l = Math.hypot(dx, dy);
          if (l > 0.001) v.dir = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'e' : 'w') : (dy > 0 ? 's' : 'n');
          if (l <= passo) { v.x = tx; v.y = ty; v.no = v.rota.shift(); passo -= l; }
          else { v.x += dx / l * passo; v.y += dy / l * passo; passo = 0; }
        }
        continue;
      }
      if ((a.estado === 'trabalhando' || a.estado === 'pendencia') && v.viagens.some(x => x.momento)) {
        v.viagens = v.viagens.filter(x => !x.momento); v.espera = undefined; // chegou trabalho: o momento fica para depois
      }
      const desejado = v.viagens.length ? v.viagens[0].no : alvoBase(a);
      if (v.no !== desejado) {
        const daMesa = v.no && v.no.startsWith('mesa'), vaiMesa = desejado.startsWith('mesa');
        v.rotulo = v.viagens.length ? v.viagens[0].rotulo
          : INFO_LOCAL[desejado] && INFO_LOCAL[desejado].txt ? 'indo descansar'
          : vaiMesa && a.estado === 'trabalhando' && !daMesa ? 'voltando ao trabalho' : vaiMesa ? 'retornando à estação' : 'a caminho';
        v.rota = Cena.rota(v.no, desejado);
      } else if (v.viagens.length) {
        if (v.espera === undefined) v.espera = v.viagens[0].espera;
        v.espera -= dt;
        if (v.espera <= 0) { v.viagens.shift(); v.espera = undefined; }
      } else if (SAINDO.has(a.id)) { SAINDO.delete(a.id); delete VIS[a.id]; } // saiu pela porta
    }
    ambiente(agora);
    atualizarGato(dt, agora);
  }

  // ================================================================ o Café, gato do escritório
  // passeia entre os cantinhos dele, cochila no sol e no tapete, e quem está por perto (sem trabalho) faz carinho
  const GATO = { no: null, x: 0, y: 0, rota: [], espera: 6, pose: 'senta', dir: 'e', quadro: 0, carinho: 0 };
  const FALAS_GATO = ['Oi, Café! Veio me visitar?', 'Café, de novo por aqui? 😺', 'Quem é o gato mais lindo deste escritório?',
    'Psiu, Café… vem cá.', 'Olha quem chegou pra supervisionar!', 'Café, sem deitar no teclado, combinado?'];
  function atualizarGato(dt, agora) {
    if (!GATO.no) {
      GATO.no = 'gato' + Math.floor(Math.random() * Cena.CANTOS_GATO); [GATO.x, GATO.y] = NOS[GATO.no];
      GATO.pose = (INFO_LOCAL[GATO.no] || {}).pose || 'senta';
    }
    if (GATO.rota.length) {
      GATO.pose = 'anda'; GATO.quadro = Math.floor(agora / 160) % 2;
      let passo = 0.9 * dt;
      while (passo > 0 && GATO.rota.length) {
        const [tx, ty] = NOS[GATO.rota[0]], dx = tx - GATO.x, dy = ty - GATO.y, l = Math.hypot(dx, dy);
        if (l > 0.001) GATO.dir = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'e' : 'w') : (dy > 0 ? 's' : 'n');
        if (l <= passo) { GATO.x = tx; GATO.y = ty; GATO.no = GATO.rota.shift(); passo -= l; } else { GATO.x += dx / l * passo; GATO.y += dy / l * passo; passo = 0; }
      }
      if (!GATO.rota.length) gatoChegou();
      return;
    }
    if ((GATO.espera -= dt) > 0) return;
    let i; do i = Math.floor(Math.random() * Cena.CANTOS_GATO); while ('gato' + i === GATO.no && Cena.CANTOS_GATO > 1);
    GATO.rota = Cena.rota(GATO.no, 'gato' + i);
  }
  function gatoChegou() {
    GATO.pose = (INFO_LOCAL[GATO.no] || {}).pose || 'senta'; GATO.espera = 25 + Math.random() * 45;
    const perto = D && D.agentes.find(a => { const v = VIS[a.id]; return v && !v.rota.length && a.estado !== 'trabalhando' && Math.hypot(v.x - GATO.x, v.y - GATO.y) < 1.9; });
    if (!perto) return;
    GATO.pose = 'senta'; GATO.carinho = performance.now() + 5000;
    if (Math.random() < 0.55 && !VIS[perto.id].baloes.length) balao(perto.id, 'fala', pick(FALAS_GATO));
  }

  function estadoVisual(a) {
    const v = VIS[a.id];
    if (v.rota.length) return 'deslocamento';
    if (v.conversa > performance.now()) return 'conversando';
    const g = emMomento(v);
    if (g) return g.tipo === 'ronda' ? 'conversando' : 'descansando';
    if (v.viagens.length) return 'deslocamento';
    return a.estado;
  }
  function textoStatus(a) {
    const v = VIS[a.id], e = estadoVisual(a), g = emMomento(v);
    if (g) return g.rotulo;
    if (e === 'deslocamento') return v.viagens.length && !v.rota.length ? v.viagens[0].rotulo : v.rotulo;
    if (e === 'descansando') return (INFO_LOCAL[v.no] && INFO_LOCAL[v.no].txt) || 'descansando';
    if (e === 'conversando') return 'conversando com a equipe';
    return a.atividade || '';
  }

  // ================================================================ desenho
  let ZOOM = 1;
  const ESCALA_GATO = () => Cena.escala() / 16; // pixels da arte -> pixels da tela
  function medir(foco) {
    const st = $('#palco'), w = st.clientWidth, h = st.clientHeight;
    const fx = foco ? foco[0] : w / 2, fy = foco ? foco[1] : h / 2;
    const rx = LW ? (st.scrollLeft + fx) / LW : 0.5, ry = LH ? (st.scrollTop + fy) / LH : 0.5;
    const t = Cena.medir(w, h, DPR, ZOOM);
    LW = t.largura; LH = t.altura;
    cn.style.width = LW + 'px'; cn.style.height = LH + 'px';
    cn.width = Math.round(LW * DPR); cn.height = Math.round(LH * DPR);
    st.scrollLeft = rx * LW - fx; st.scrollTop = ry * LH - fy;
  }
  function zoom(f, foco) { ZOOM = Math.max(1, Math.min(2.6, f)); medir(foco); }
  // zoom: botões e pinça/Ctrl+roda; arrastar move a vista; 💭 liga/desliga os balões
  const zb = document.createElement('div');
  zb.id = 'zoom';
  zb.innerHTML = `<button title="Aproximar">${ICONE.mais}</button><button title="Afastar">${ICONE.menos}</button><button title="Ver tudo">${ICONE.tudo}</button>` +
    `<button title="Mostrar/ocultar pensamentos e falas" class="${BALOES ? 'on' : ''}">${ICONE.balao}</button><button title="Legenda dos estados">${ICONE.info}</button>`;
  $('#esq').appendChild(zb);
  zb.onclick = e => {
    const b = e.target.closest('button'); if (!b) return;
    const i = [...zb.children].indexOf(b);
    if (i === 0) zoom(ZOOM * 1.25); if (i === 1) zoom(ZOOM / 1.25); if (i === 2) zoom(1);
    if (i === 3) { BALOES = !BALOES; gravarPref('hub.baloes', BALOES); b.classList.toggle('on', BALOES); }
    if (i === 4) { e.stopPropagation(); const l = $('#legenda'), r = b.getBoundingClientRect(); l.style.top = r.top + 'px'; l.style.right = (innerWidth - r.left + 8) + 'px'; l.classList.toggle('oculto'); b.classList.toggle('on', !l.classList.contains('oculto')); }
  };
  addEventListener('click', e => { if (!e.target.closest('#legenda')) { $('#legenda').classList.add('oculto'); zb.children[4].classList.remove('on'); } });
  $('#palco').addEventListener('wheel', e => {
    if (!e.ctrlKey && !e.metaKey) return;
    e.preventDefault(); const r = $('#palco').getBoundingClientRect();
    zoom(ZOOM * (e.deltaY < 0 ? 1.12 : 1 / 1.12), [e.clientX - r.left, e.clientY - r.top]);
  }, { passive: false });
  // arrastar a vista: segurando o botão direito (ou o esquerdo, fora dos personagens); o menu do navegador não abre
  let arrasto = null, arrastou = false;
  cn.addEventListener('contextmenu', e => e.preventDefault());
  cn.addEventListener('mousedown', e => {
    if (e.button !== 0 && e.button !== 2) return;
    arrasto = [e.clientX, e.clientY, $('#palco').scrollLeft, $('#palco').scrollTop, e.button]; arrastou = false;
    if (e.button === 2) { e.preventDefault(); cn.style.cursor = 'grabbing'; }
  });
  addEventListener('mousemove', e => {
    if (!arrasto) return;
    const dx = e.clientX - arrasto[0], dy = e.clientY - arrasto[1];
    if (Math.abs(dx) + Math.abs(dy) > (arrasto[4] === 2 ? 1 : 5)) { arrastou = true; cn.style.cursor = 'grabbing'; }
    if (arrastou) { $('#palco').scrollLeft = arrasto[2] - dx; $('#palco').scrollTop = arrasto[3] - dy; }
  });
  addEventListener('mouseup', () => { if (arrasto) { arrasto = null; cn.style.cursor = HOVER ? 'pointer' : 'default'; } });

  function fundoDados() {
    const est = {};
    let tv = null;
    for (const a of D.agentes) {
      const v = VIS[a.id]; if (!v || v.rota.length) continue;
      if (v.no && v.no.startsWith('mesa')) est[+v.no.slice(4)] = a.estado;
      if (v.no === 'sofa1') tv = 'jogo'; // quem está no meio do sofá joga videogame
      else if (v.no && v.no.startsWith('sofa') && !tv) tv = 'tv';
    }
    const bolo = Object.values(VIS).some(v => { const g = emMomento(v); return g && g.tipo === 'aniversario'; });
    return { ...D, estacoes: est, tv, bolo, periodo: PERIODO };
  }

  let ult = 0, T0 = performance.now(), posicoes = [];
  function quadroAnim(t) {
    const dt = Math.min(0.05, (t - ult) / 1000 || 0); ult = t;
    if (D) {
      atualizar(dt);
      Cena.atualizarFundo(fundoDados());
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.clearRect(0, 0, LW, LH);
      desenharAgentes((t - T0) / 1000);
      desenharRosto();
    }
    requestAnimationFrame(quadroAnim);
  }

  const FONTE_NOME = '700 12px system-ui,-apple-system,Segoe UI,sans-serif', FONTE_EST = '600 11px system-ui,-apple-system,Segoe UI,sans-serif';
  // o que cada um faz neste instante: digitar, gole de café, folhear o livro, jogar, piscar, respirar, falar…
  // (só a forma de mostrar: o estado vem da API)
  const CHAT_VIVO = {}; // id -> { modo: 'acena' | 'pensa' | 'fala', ate } enquanto o recrutador conversa com o agente
  function animar(a, v, info, andando, sentado, seg, agora) {
    const fase = a.id * 1.37, e = estadoVisual(a), ciclo = (periodo, dur) => ((seg + fase) % periodo) < dur;
    const o = { pose: andando ? 'anda' : sentado ? 'senta' : 'pe', acao: '', quadro: 0, item: andando ? null : (info.item || null),
      olhos: '', boca: 0, resp: 0, vapor: false, humor: e === 'pendencia' ? 'preocupado' : e === 'trabalhando' ? 'focado' : 'feliz' };
    if (andando) { o.quadro = Math.floor(seg * 7 + a.id) % 4; return o; }
    o.resp = ((seg * 0.8 + fase) % 2.4) < 1.2 ? 1 : 0;
    if (((seg + fase) % 4.3) < 0.14) o.olhos = 'fechado';
    if (sentado && info.digita && a.estado === 'trabalhando') {
      if (!ciclo(8, 1.4)) { o.acao = 'digita'; o.quadro = Math.floor(seg * 8 + a.id) % 2; } // e às vezes para e lê a tela
    }
    switch (o.item) {
      case 'xicara': if (ciclo(6.5, 1.3)) o.acao = 'gole'; else o.vapor = true; break;
      case 'copo': if (ciclo(4.5, 1.2)) o.acao = 'gole'; break;
      case 'lanche': if (ciclo(5, 0.8)) o.acao = 'mordida'; else if (ciclo(5, 1.9)) o.boca = Math.floor(seg * 6) % 2; break;
      case 'livro': o.humor = 'focado'; if (ciclo(7, 0.5)) o.acao = 'folheia'; break;
      case 'controle': o.acao = 'controle'; o.quadro = Math.floor(seg * 8) % 4; o.humor = 'focado';
        if (ciclo(11, 0.9)) { o.acao = 'espreguica'; o.humor = 'risada'; } break; // comemora
      case 'celular': o.quadro = Math.floor(seg * 3) % 2; o.humor = ciclo(9, 0.7) ? 'risada' : 'focado'; break;
      case 'fone': o.resp = Math.floor(seg * 2.4 + fase) % 2; o.humor = 'calmo'; break; // balança a cabeça no ritmo
      case 'sono': o.olhos = 'fechado'; o.humor = 'sono'; o.resp = ((seg * 0.5 + fase) % 3) < 1.5 ? 1 : 0; break;
      case 'raquete': if (ciclo(1.3, 0.3)) o.acao = 'raquetada'; break; // treinando sozinho (com parceiro, sincroniza abaixo)
    }
    if (/^sofa/.test(v.no || '') && !o.item) o.humor = ciclo(9, 0.8) ? 'risada' : 'calmo';
    if (info.acao === 'alonga') { // alongamento em grupo: braços para cima, para o lado, respira
      const f = (seg * 0.45 + fase * 0.2) % 4;
      o.acao = f < 1.5 ? 'espreguica' : f > 2.2 && f < 3.2 ? 'acena' : ''; o.quadro = Math.floor(seg * 1.5) % 2; o.humor = 'calmo';
      if (f < 1.5) o.olhos = 'fechado';
    }
    // falando (balão de fala no ar): boca mexendo e gesto; pensando: mão no queixo
    const b = v.baloes[0];
    if (b && b.t0 && agora - b.t0 < b.dur) {
      if (b.tipo === 'fala') { o.boca = Math.floor(seg * 7) % 2; if (!o.item && o.acao !== 'digita') { o.acao = 'gesto'; o.quadro = Math.floor(seg * 2.5) % 2; } }
      else if (!o.item && o.acao !== 'digita') o.acao = 'pensa';
    }
    // conversa com o recrutador (chat aberto): vira para a tela, acena, pensa e fala
    const ch = CHAT_VIVO[a.id];
    if (ch) {
      o.dir = 's';
      if (ch.modo === 'acena' && agora < ch.ate) { o.acao = 'acena'; o.quadro = Math.floor(seg * 6) % 2; o.humor = 'feliz'; }
      else if (ch.modo === 'pensa') { o.acao = 'pensa'; o.humor = 'focado'; o.item = null; o.vapor = false; }
      else if (ch.modo === 'fala') { o.acao = 'gesto'; o.quadro = Math.floor(seg * 2.5) % 2; o.boca = Math.floor(seg * 7) % 2; o.item = null; o.vapor = false; }
      if (o.acao) o.olhos = '';
    }
    return o;
  }
  // de frente para quem está conversando com ele (no escritório)
  function olharPara(v, outro) {
    const dx = outro.x - v.x, dy = outro.y - v.y;
    return Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 'e' : 'w') : (dy > 0 ? 's' : 'n');
  }
  function desenharAgentes(seg) {
    const agora = performance.now();
    const lista = todos().filter(a => VIS[a.id]).map(a => {
      const v = VIS[a.id], g = emMomento(v), info0 = INFO_LOCAL[v.no] || {}, andando = v.rota.length > 0;
      const info = g && (g.item || g.acao) ? { ...info0, item: g.item || info0.item, acao: g.acao } : info0, sentado = !andando && !!info.sentado;
      const o = animar(a, v, info, andando, sentado, seg, agora);
      let dir = andando ? (v.dir || 's') : (o.dir || info.f || v.dir || 's');
      const par = !andando && !o.dir && v.conversa > agora && VIS[v.conversaCom];
      if (par && !/^(sofa|pufe|leitura)/.test(v.no || '')) dir = olharPara(v, par);
      return { a, v, id: a.id, x: v.x, y: v.y, ap: a.aparencia || {}, andando, dir, ...o,
        destaque: SEL === a.id || (HOVER && HOVER.id === a.id) };
    });
    // pingue-pongue: com dois jogadores, a bolinha vai e volta e cada um rebate quando ela chega
    const jog = lista.filter(p => !p.andando && (p.v.no === 'ping0' || p.v.no === 'ping1'));
    const fb = (Math.sin(seg * 3.2) + 1) / 2;
    const bolinha = jog.length === 2 ? [19.8 + fb * 2.4, 12.3, 0.84 + Math.abs(Math.sin(seg * 6.4)) * 0.4] : null;
    if (bolinha) for (const p of jog) p.acao = (p.v.no === 'ping0' ? fb < 0.14 : fb > 0.86) ? 'raquetada' : '';
    const telas = Cena.desenhar(ctx, lista, { bolinha, t: seg, gato: GATO }), porId = new Map(lista.map(p => [p.id, p]));
    if (GATO.tela) { // zzz de quando o Café cochila, coraçõezinhos quando ganha carinho
      const [gx, gy] = GATO.tela, carinho = GATO.carinho > agora;
      if (carinho || GATO.pose === 'dorme') {
        ctx.save(); ctx.textAlign = 'center'; ctx.font = carinho ? '700 12px system-ui' : 'italic 800 10px system-ui';
        for (let k = 0; k < (carinho ? 2 : 3); k++) {
          const f = (seg * 0.4 + k / (carinho ? 2 : 3)) % 1;
          ctx.globalAlpha = Math.sin(f * Math.PI) * 0.9; ctx.fillStyle = carinho ? '#F26D8D' : '#DCE6FF';
          ctx.fillText(carinho ? '♥' : 'z', gx + 4 + f * 8 + Math.sin(f * 6 + k) * 2, gy - 2 - f * 16);
        }
        ctx.restore();
      }
    }
    posicoes = telas.map(t => ({ ...porId.get(t.id), ...t }));
    if (HOVER && HOVER.tipo !== 'agente') { // contorno do quadro ou da bandeja sob o mouse
      const a = Cena.alvos().find(x => x.tipo === HOVER.tipo);
      if (a) { ctx.beginPath(); a.pts.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath(); ctx.strokeStyle = ACENTO; ctx.lineWidth = 2; ctx.stroke(); }
    }
    for (const p of posicoes) {
      const { a, v, hx, hy } = p, e = estadoVisual(a);
      if (a.estado === 'pendencia' || a.pendencias) { // "!" vermelho sobre a cabeça
        const bx = hx + 13, by = hy + 6;
        ctx.fillStyle = '#E5534B'; ctx.strokeStyle = '#1B1420'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(bx, by, 7.5, 0, 7); ctx.stroke(); ctx.fill();
        ctx.fillStyle = '#fff'; ctx.font = '800 11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('!', bx, by + 0.5); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      }
      if (!p.andando && (p.item === 'sono' || p.item === 'fone')) { // zzz de quem cochila, notas de quem ouve música
        ctx.save(); ctx.font = p.item === 'sono' ? 'italic 800 11px system-ui' : '700 13px system-ui'; ctx.textAlign = 'center';
        for (let k = 0; k < 3; k++) {
          const f = (seg * 0.45 + k / 3 + a.id * 0.2) % 1;
          ctx.globalAlpha = Math.sin(f * Math.PI) * 0.9;
          ctx.fillStyle = p.item === 'sono' ? '#DCE6FF' : ['#F5B544', '#7EC3EB', '#E678AA'][k];
          ctx.fillText(p.item === 'sono' ? 'z' : '♪', hx + 10 + f * 10 + Math.sin(f * 7 + k) * 2, hy + 4 - f * 22);
        }
        ctx.restore();
      }
      const ch = CHAT_VIVO[a.id];
      if (ch && (ch.modo === 'pensa' || (ch.modo === 'acena' && performance.now() < ch.ate))) { // ? enquanto pensa, ! ao cumprimentar
        const bx = hx - 14, by = hy + 4 + Math.sin(seg * 5) * 1.5;
        ctx.fillStyle = '#F5B544'; ctx.strokeStyle = '#1B1420'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(bx, by, 7.5, 0, 7); ctx.stroke(); ctx.fill();
        ctx.fillStyle = '#1A1206'; ctx.font = '800 11px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(ch.modo === 'pensa' ? '?' : '!', bx, by + 0.5); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      }
      if (e === 'conversando' && !(BALOES && v.baloes[0] && v.baloes[0].t0)) { // "…" de quem está conversando
        const bx = hx + 9, by = hy - 4;
        ctx.fillStyle = '#FFFFFF'; ctx.strokeStyle = '#1B1420'; ctx.lineWidth = 2; ctx.beginPath(); ctx.roundRect(bx, by - 8, 26, 15, 6); ctx.stroke(); ctx.fill();
        ctx.fillStyle = '#5A5F70';
        for (let k = 0; k < 3; k++) { ctx.beginPath(); ctx.arc(bx + 7 + k * 6, by - 0.5, Math.floor(seg * 3) % 3 === k ? 2.2 : 1.6, 0, 7); ctx.fill(); }
      }
    }
    // etiquetas: nome + bolinha de status, sem sobreposição
    const caixas = [];
    for (const p of [...posicoes].sort((q, r) => r.hy - q.hy)) {
      const { a, hx, hy } = p, e = estadoVisual(a), [cor, rot] = ESTADOS[e] || ESTADOS.aguardando;
      ctx.font = FONTE_NOME; const wn = ctx.measureText(a.nome).width;
      ctx.font = FONTE_EST; const ws = ctx.measureText(rot).width;
      const w = wn + ws + 30, h = 21;
      let x = hx - w / 2, y = hy - h - 4;
      for (let tent = 0; tent < 6; tent++) {
        const bate = caixas.find(c => x < c.x + c.w + 3 && x + w + 3 > c.x && y < c.y + c.h + 2 && y + h + 2 > c.y);
        if (!bate) break; y = bate.y - h - 3;
      }
      caixas.push({ x, y, w, h, id: a.id });
      ctx.fillStyle = 'rgba(17,16,26,.88)'; ctx.beginPath(); ctx.roundRect(x, y, w, h, 6); ctx.fill();
      ctx.strokeStyle = SEL === a.id ? ACENTO : 'rgba(255,255,255,.1)'; ctx.lineWidth = SEL === a.id ? 2 : 1; ctx.stroke();
      ctx.fillStyle = '#F2F3F7'; ctx.font = FONTE_NOME; ctx.fillText(a.nome, x + 8, y + 14.5);
      ctx.fillStyle = cor; ctx.beginPath(); ctx.arc(x + 16 + wn, y + 10.5, 3.5, 0, 7); ctx.fill();
      ctx.fillStyle = '#A9B0C2'; ctx.font = FONTE_EST; ctx.fillText(rot, x + 23 + wn, y + 14.5);
    }
    posicoes.caixas = caixas;
    if (BALOES) desenharBaloes(caixas);
  }

  function quebrar(texto, larg, max) {
    const linhas = [];
    let atual = '';
    for (const p of texto.split(/\s+/).filter(Boolean)) {
      const t = atual ? atual + ' ' + p : p;
      if (atual && ctx.measureText(t).width > larg) { linhas.push(atual); atual = p; } else atual = t;
    }
    if (atual) linhas.push(atual);
    if (linhas.length > max) {
      linhas.length = max;
      let u = linhas[max - 1];
      while (u.length > 1 && ctx.measureText(u + '…').width > larg) u = u.slice(0, -1);
      linhas[max - 1] = u + '…';
    }
    return linhas;
  }

  // botões sobre o palco e a faixa acima da área visível também contam como ocupados
  function obstaculos() {
    const cr = cn.getBoundingClientRect(), topo = $('#palco').scrollTop;
    return [{ x: 0, y: -2000, w: LW, h: 2000 + topo + 4 },
      ...[...document.querySelectorAll('#ferramentas .btn, #zoom button')].map(b => {
        const r = b.getBoundingClientRect(); return { x: r.left - cr.left - 4, y: r.top - cr.top - 4, w: r.width + 8, h: r.height + 8 };
      })];
  }

  function desenharBaloes(caixas) {
    const agora = performance.now(), ocupado = [...caixas.map(c => ({ ...c })), ...obstaculos()];
    for (const p of [...posicoes].sort((q, r) => r.hy - q.hy)) {
      const b = p.v.baloes[0]; if (!b || !b.t0) continue;
      const lab = caixas.find(c => c.id === p.a.id); if (!lab) continue;
      const pens = b.tipo === 'pensamento';
      ctx.font = pens ? FONTE_PENS : FONTE;
      const linhas = quebrar(b.texto, 178, 3);
      const w = Math.max(...linhas.map(l => ctx.measureText(l).width)) + 22, h = linhas.length * 15 + 11;
      const folga = pens ? 15 : 10, y0 = lab.y - h - folga;
      const bate = (x, y) => ocupado.find(c => x < c.x + c.w + 4 && x + w + 4 > c.x && y < c.y + c.h + 3 && y + h + 3 > c.y);
      const ajusta = x => Math.max(4, Math.min(LW - w - 4, x));
      // centralizado, puxado para a direita ou para a esquerda (a ponta segue apontando para o dono)...
      let x = [p.hx - w / 2, p.hx - 24, p.hx - w + 24].map(ajusta).find(x => !bate(x, y0)), y = y0, lado = false;
      if (x === undefined) { // ...ao lado da etiqueta...
        const yl = lab.y + lab.h - h;
        x = [lab.x + lab.w + 8, lab.x - w - 8].find(x => x >= 4 && x + w <= LW - 4 && !bate(x, yl));
        if (x !== undefined) { y = yl; lado = true; }
      }
      if (x === undefined) { // ...e só então sobe
        x = ajusta(p.hx - w / 2);
        for (let t = 0, c; t < 8 && (c = bate(x, y)); t++) y = c.y - h - 6;
      }
      ocupado.push({ x, y, w, h });
      const vida = agora - b.t0;
      ctx.save(); ctx.globalAlpha = Math.max(0, Math.min(1, vida / 180, (b.dur - vida) / 260));
      ctx.shadowColor = 'rgba(33,30,22,.16)'; ctx.shadowBlur = 8; ctx.shadowOffsetY = 2;
      const ax = Math.max(x + 14, Math.min(x + w - 14, p.hx));
      if (!lado && y < y0 - 2) { // balão deslocado: linha-guia até a etiqueta do dono
        ctx.save(); ctx.shadowColor = 'transparent'; ctx.setLineDash([3, 3]); ctx.strokeStyle = pens ? '#B9AEE0' : '#9A9A94'; ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(ax, y + h + 4); ctx.lineTo(p.hx, lab.y); ctx.stroke(); ctx.restore();
      }
      if (pens) {
        ctx.fillStyle = '#F6F3FF'; ctx.strokeStyle = '#CFC6EE'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.roundRect(x, y, w, h, Math.min(h / 2, 14)); ctx.fill();
        ctx.shadowColor = 'transparent'; ctx.stroke();
        // bolinhas da nuvem descendo até a etiqueta
        if (!lado) for (const [dy, r] of [[5, 3.6], [10.5, 2.2]]) { ctx.beginPath(); ctx.arc(ax + dy * 0.4, y + h + dy, r, 0, 7); ctx.fill(); ctx.stroke(); }
        ctx.fillStyle = '#463F63';
      } else {
        const cor = (p.a.aparencia || {}).camisa || '#B9BDC6';
        ctx.fillStyle = '#FFFFFF'; ctx.strokeStyle = cor === '#F0F0F0' ? '#B9BDC6' : cor; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.roundRect(x, y, w, h, 9);
        if (!lado) { ctx.moveTo(ax - 6, y + h); ctx.lineTo(ax, y + h + 8); ctx.lineTo(ax + 6, y + h); }
        ctx.fill(); ctx.shadowColor = 'transparent';
        ctx.beginPath(); ctx.roundRect(x, y, w, h, 9); ctx.stroke();
        if (!lado) {
          ctx.fillStyle = '#FFFFFF'; ctx.fillRect(ax - 5, y + h - 2, 10, 3);
          ctx.beginPath(); ctx.moveTo(ax - 6, y + h); ctx.lineTo(ax, y + h + 8); ctx.lineTo(ax + 6, y + h); ctx.stroke();
        }
        ctx.fillStyle = '#211E16';
      }
      ctx.font = pens ? FONTE_PENS : FONTE;
      linhas.forEach((l, i) => ctx.fillText(l, x + 11, y + 18 + i * 15));
      ctx.restore();
    }
  }

  // ================================================================ passar o mouse / clicar
  function acertar(mx, my) {
    for (const c of (posicoes.caixas || []).slice().reverse())
      if (mx >= c.x && mx <= c.x + c.w && my >= c.y && my <= c.y + c.h) return { tipo: 'agente', id: c.id };
    for (const p of [...posicoes].sort((q, r) => r.sy - q.sy))
      if (mx >= p.x0 && mx <= p.x1 && my >= p.y0 && my <= p.y1) return { tipo: 'agente', id: p.a.id };
    if (GATO.tela && Math.abs(mx - GATO.tela[0]) < 7 * ESCALA_GATO() && my > GATO.tela[1] - 2 && my < GATO.tela[1] + 10 * ESCALA_GATO()) return { tipo: 'gato' };
    for (const a of Cena.alvos()) if (Cena.dentro([mx, my], a.pts)) return { tipo: a.tipo };
    return null;
  }
  cn.addEventListener('mousemove', e => {
    if (!D) return;
    const r = cn.getBoundingClientRect(); HOVER = acertar(e.clientX - r.left, e.clientY - r.top);
    cn.style.cursor = HOVER ? 'pointer' : 'default';
    const dica = $('#dica');
    if (!HOVER) { dica.style.display = 'none'; return; }
    if (HOVER.tipo === 'agente') {
      const a = todos().find(x => x.id === HOVER.id); if (!a || !VIS[a.id]) { dica.style.display = 'none'; return; }
      const e2 = estadoVisual(a), [cor, rot] = ESTADOS[e2] || ESTADOS.aguardando, t = textoStatus(a), pf = a.perfil || {};
      dica.innerHTML = `<b>${esc((pf.emoji ? pf.emoji + ' ' : '') + a.nome.toUpperCase())}</b><br>Especialidade: ${esc(a.especialidade)}<br>Status: <b style="color:${cor}">${rot}</b>${t ? ' · ' + esc(t) : ''}` +
        (pf.resumo ? `<br><span class="mini">${esc(pf.resumo)}</span>` : '');
    } else if (HOVER.tipo === 'vagas') dica.innerHTML = '<b>Quadro de vagas em aberto</b><br>Usado pelos agentes. Clique para ver a análise.';
    else if (HOVER.tipo === 'gato') dica.innerHTML = `<b>🐈 Café</b><br>O gato do escritório. ${GATO.pose === 'dorme' ? 'Tirando um cochilo.' : GATO.pose === 'anda' ? 'Dando uma volta.' : 'Observando tudo.'}<br><span class="mini">Gosta do sol da janela, do tapete do sofá e de carinho na hora do café.</span>`;
    else if (HOVER.tipo === 'quadro') dica.innerHTML = '<b>Quadro geral do supervisor</b><br>Clique para ver tarefas, equipe, contratações e pendências.';
    else if (HOVER.tipo === 'bandeja') dica.innerHTML = `<b>Recepção de currículos</b><br>${D.fila_triagem} aguardando triagem · ${D.fila_analise} aguardando os analistas`;
    else if (HOVER.tipo === 'banco') { const b = D.banco || {}; dica.innerHTML = `<b>📁 Banco de currículos</b><br>${milhar(b.total || 0)} currículo(s) guardados nas pastas, por área.<br><span class="mini">Clique para abrir o arquivo: o que a IA disse de cada um e a vaga sugerida.</span>`; }
    if (HOVER.tipo === 'agente' && D.agentes.some(a => a.id === HOVER.id)) preBuscar(`/api/agentes/${HOVER.id}`);
    dica.style.display = 'block';
    dica.style.left = Math.min(innerWidth - 300, e.clientX + 16) + 'px'; dica.style.top = (e.clientY + 18) + 'px';
  });
  cn.addEventListener('mouseleave', () => { HOVER = null; $('#dica').style.display = 'none'; });
  cn.addEventListener('click', e => {
    if (arrastou) { arrastou = false; return; }
    if (e.button !== 0) return;
    const r = cn.getBoundingClientRect(), h = acertar(e.clientX - r.left, e.clientY - r.top);
    if (!h) return;
    $('#dica').style.display = 'none';
    if (h.tipo === 'agente') { if (D.agentes.some(a => a.id === h.id)) abrir({ tipo: 'agente', id: h.id, aba: 'resumo' }); }
    else if (h.tipo === 'vagas') abrir({ tipo: 'vagas', filtro: 'todas' });
    else if (h.tipo === 'quadro') abrir({ tipo: 'quadro' });
    else if (h.tipo === 'bandeja') { const b = D.agentes.find(a => a.funcao === 'recepcao'); if (b) abrir({ tipo: 'agente', id: b.id, aba: 'resumo' }); }
    else if (h.tipo === 'banco') abrirBanco();
    else if (h.tipo === 'gato') { GATO.carinho = performance.now() + 4000; if (GATO.pose === 'dorme') GATO.pose = 'senta'; } // carinho de quem clica
  });

  // ================================================================ atividade recente
  let NAO_LIDOS = 0, ATIV_ABERTA = false;
  function montarAtividade() { addEventListener('click', e => { if (ATIV_ABERTA && !e.target.closest('#atividade, #sino')) alternarAtividade(false); }); }
  function alternarAtividade(forcar) {
    ATIV_ABERTA = typeof forcar === 'boolean' ? forcar : !ATIV_ABERTA;
    $('#atividade').classList.toggle('oculto', !ATIV_ABERTA);
    const s = $('#sino'); if (s) s.classList.toggle('on', ATIV_ABERTA);
    if (ATIV_ABERTA) { NAO_LIDOS = 0; atualizarSino(); atividade(); }
  }
  function atualizarSino() {
    const c = $('#sino-cont'); if (!c) return;
    c.textContent = NAO_LIDOS > 99 ? '99+' : String(NAO_LIDOS); c.classList.toggle('oculto', !NAO_LIDOS);
  }
  function textoEvento(e) {
    if (e.tipo !== 'conversa') return esc(e.mensagem);
    const d = e.dados || {}, de = (AGS[e.agente_id] || {}).nome || d.de || '', para = (AGS[d.destino_id] || {}).nome || d.para;
    return `💬 <i>${esc(de)}${para ? ' → ' + esc(para) : ''}:</i> “${esc(e.mensagem)}”`;
  }
  function atividade(novoId) {
    if (!D) return;
    if (novoId && !ATIV_ABERTA) { NAO_LIDOS++; atualizarSino(); }
    if (!ATIV_ABERTA) return;
    $('#atividade').innerHTML = `<header><i class="vivo"></i>Atividade <span>· tempo real</span></header><div class="lista-ev">` +
      ((D.eventos || []).slice(0, 25).map(e => `<p class="${e.id === novoId ? 'novo' : ''}"><b>${hora(e.ts)}</b><span>${textoEvento(e)}</span></p>`).join('')
        || '<p class="mini">Nada por enquanto.</p>') + '</div>';
  }

  // ================================================================ painel lateral
  // O painel abre na hora: primeiro com o que já foi visto (ou um esqueleto) e, em seguida, com os dados novos.
  // Passar o mouse sobre um agente ou um botão já busca os dados (pré-busca), então o clique quase sempre acerta o cache.
  const CACHE = new Map(), SEM_CACHE = new Error('sem cache'), PRE = new Map();
  const doCache = url => (CACHE.has(url) ? Promise.resolve(CACHE.get(url)) : Promise.reject(SEM_CACHE));
  doCache.soCache = true;
  async function daRede(url) { const d = await api('GET', url); CACHE.set(url, d); return d; }
  function preBuscar(url) {
    const t = performance.now(); if ((PRE.get(url) || -1e9) > t - 5000) return;
    PRE.set(url, t); daRede(url).catch(() => {});
  }
  function abrir(p) { P = p; SEL = p.tipo === 'agente' ? p.id : null; $('#cvw').style.display = 'none'; $('#painel').classList.add('aberto'); setTimeout(medir, 200); pintarPainel(); }
  function fechar() { P = null; SEL = null; $('#painel').classList.remove('aberto'); $('#cvw').style.display = 'none'; vidaDoChat(); setTimeout(medir, 200); }

  function construirPainel(get) {
    return P.tipo === 'agente' ? painelAgente(get) : P.tipo === 'vagas' ? painelVagas(get) : P.tipo === 'equipe' ? painelEquipe(get) : painelQuadro(get);
  }
  function esqueleto() {
    const a = P.tipo === 'agente' && D.agentes.find(x => x.id === P.id);
    const topo = a ? cabecalhoAgente({ ...a, perfil_completo: a.perfil || {} })
      : `<div class="ph"><div style="min-width:0;flex:1"><i class="esq-l" style="width:55%;height:20px"></i><i class="esq-l" style="width:35%"></i></div><button class="x" data-a="x" aria-label="Fechar">×</button></div>`;
    return topo + `<div id="pb">${'<div class="cartao esq"><i class="esq-l" style="width:70%"></i><i class="esq-l"></i><i class="esq-l" style="width:45%"></i></div>'.repeat(3)}</div>`;
  }
  let versaoPainel = 0, htmlPainel = '';
  function aplicarPainel(html, manter) {
    if (html === htmlPainel) return;
    const pb = $('#pb'), rolagem = pb ? pb.scrollTop : 0;
    const foco = document.activeElement && ['q', 'qeq'].includes(document.activeElement.id) ? document.activeElement.id : null;
    const rascunho = foco ? $('#' + foco).value : '';
    $('#pi').innerHTML = htmlPainel = html;
    if (P.tipo === 'agente') desenharRosto(true);
    desenharAvatares($('#pi'));
    const nb = $('#pb'); if (nb) nb.scrollTop = (P.tipo === 'equipe' || P.aba === 'chat') ? 1e6 : manter ? rolagem : 0;
    if (foco && $('#' + foco)) { $('#' + foco).value = rascunho; $('#' + foco).focus(); }
  }
  async function pintarPainel(silencioso) {
    if (!P) return;
    const alvo = P, v = ++versaoPainel;
    vidaDoChat();
    if (!silencioso) {
      let html;
      try { html = await construirPainel(doCache); } catch (_) { html = esqueleto(); }
      if (alvo !== P || v !== versaoPainel) return;
      aplicarPainel(html, false);
    }
    try {
      const html = await construirPainel(daRede);
      if (alvo !== P || v !== versaoPainel) return;
      aplicarPainel(html, true);
    } catch (e) {
      if (!silencioso && v === versaoPainel && alvo === P) aplicarPainel(`<div class="ph"><h2>Erro</h2><button class="x" data-a="x">×</button></div><div id="pb"><p class="erro">${esc(e.message)}</p></div>`, false);
    }
  }

  const rosto = (k, s, ap, o) => Cena.retrato(k, s, ap || {}, undefined, o);
  // o rosto no topo do painel pisca e mexe a boca quando o agente responde no chat
  let rostoChave = '';
  function desenharRosto(forcar) {
    const c = $('#rosto'); if (!c || !P || P.tipo !== 'agente') return;
    const a = D.agentes.find(x => x.id === P.id); if (!a) return;
    const t = performance.now() / 1000, ch = CHAT_VIVO[a.id];
    const o = { olhos: ((t + a.id * 1.37) % 3.8) < 0.14 ? 'fechado' : '', boca: ch && ch.modo === 'fala' ? Math.floor(t * 7) % 2 : 0,
      humor: ch && ch.modo === 'pensa' ? 'focado' : a.estado === 'pendencia' ? 'preocupado' : 'feliz' };
    const chave = [a.id, o.olhos, o.boca, o.humor].join('|');
    if (!forcar && chave === rostoChave) return;
    rostoChave = chave;
    const k = c.getContext('2d'); c.width = 68 * DPR; c.height = 68 * DPR; k.scale(DPR, DPR); rosto(k, 68, a.aparencia, o);
    c.classList.toggle('falando', !!(ch && (ch.modo === 'fala' || ch.modo === 'pensa')));
  }
  function desenharAvatares(raiz) {
    if (!raiz) return;
    for (const c of raiz.querySelectorAll('canvas.av')) {
      const ag = AGS[c.dataset.ag], s = +c.dataset.s || 28; if (!ag) continue;
      c.style.width = c.style.height = s + 'px'; c.width = c.height = s * DPR;
      const k = c.getContext('2d'); k.scale(DPR, DPR); rosto(k, s, ag.aparencia);
    }
  }
  const av = (id, s = 28) => `<canvas class="av" data-ag="${id}" data-s="${s}" width="${s}" height="${s}"></canvas>`;

  const chipEstado = a => { const [c, n] = ESTADOS[a.estado] || ESTADOS.aguardando; return `<span class="chip" style="background:${c}">${n}${a.atividade ? ' · ' + esc(a.atividade) : ''}</span>`; };
  const prog = (a, t) => `<div class="barra"><i style="width:${t ? Math.round(a / t * 100) : 0}%"></i></div>`;
  const histHtml = h => h.length ? `<div class="hist">${h.map(e => `<p><b>${hora(e.ts)}</b>${textoEvento(e)}</p>`).join('')}</div>` : '<p class="mini">Sem atividade registrada ainda.</p>';
  const gestor = () => EU.papel !== 'recrutador';

  function cabecalhoAgente(d) {
    const pf = d.perfil_completo || {};
    return `<div class="ph"><canvas id="rosto" width="68" height="68" style="width:68px;height:68px"></canvas>
      <div style="min-width:0"><h2>${esc(d.nome.toUpperCase())} <span class="emoji">${esc(pf.emoji || '')}</span></h2><small>${esc(d.funcao_rotulo)}</small><em>Especialidade: ${esc(d.especialidade)}</em>${chipEstado(d)}</div>
      <button class="x" data-a="x" aria-label="Fechar">×</button></div>
      <div class="abas">${[['resumo', 'Resumo'], ['candidatos', 'Candidatos'], ['chat', 'Chat']].map(([i, n]) => `<button class="${P.aba === i ? 'on' : ''}" data-a="aba" data-v="${i}">${n}</button>`).join('')}</div>`;
  }
  async function painelAgente(get) {
    // o chat não precisa esperar o servidor: nome, função e personalidade já estão no escritório
    const a = D.agentes.find(x => x.id === P.id);
    const d = await get(`/api/agentes/${P.id}`).catch(e => { if (P.aba === 'chat' && a) return { ...a, perfil_completo: a.perfil || {} }; throw e; });
    let corpo = '';
    if (P.aba === 'resumo') corpo = await resumoAgente(d, get);
    else if (P.aba === 'candidatos') corpo = await candidatosAgente(d, get);
    else corpo = chatAgente(d);
    return `${cabecalhoAgente(d)}<div id="pb">${corpo}</div>${P.aba === 'chat' ? `<div id="entrada"><input id="q" maxlength="1000" placeholder="Pergunte algo a ${esc(d.nome)}…"><button class="btn" data-a="enviar">Enviar</button></div>` : ''}`;
  }

  function cartaoPerfil(d) {
    const pf = d.perfil_completo || {};
    if (!pf.resumo) return '';
    return `<div class="cartao perfil"><b>${esc(pf.emoji || '')} Personalidade</b><div style="margin-top:3px">${esc(pf.resumo.charAt(0).toUpperCase() + pf.resumo.slice(1))}.</div>
      ${pf.tom ? `<div class="mini" style="margin-top:3px">Jeito de falar: ${esc(pf.tom)}</div>` : ''}
      ${(pf.bordoes || []).length ? `<div class="mini">Bordões: ${pf.bordoes.map(b => '“' + esc(b) + '”').join(' · ')}</div>` : ''}</div>`;
  }

  function htmlContratacoes(lista) {
    const icone = { contratacao: '🤝', retorno: '↩️', dispensa: '🚪', sem_estacao: '⛔' };
    return lista.length ? `<div class="hist">${lista.map(c => `<p><b>${hora(c.em)}</b>${icone[c.acao] || '•'} ${esc(c.descricao)}</p>`).join('')}</div>`
      : '<p class="mini">Nenhuma contratação ainda.</p>';
  }

  async function resumoAgente(d, get) {
    const pend = d.pendencias_abertas.map(p => `<div class="cartao pend"><b>Pendência:</b> ${esc(p.descricao)}${gestor() ? ' <span class="mini">· resolva no quadro geral</span>' : ''}</div>`).join('');
    if (d.funcao === 'supervisor') {
      const q = D.quadro, au = D.automacao || {}, [contr, ia] = await Promise.all([get('/api/contratacoes'), linhaIA(get)]);
      return `${cartaoPerfil(d)}<div class="sec">Visão da equipe</div><div class="g4"><div class="kpi"><b>${q.vagas_abertas}</b>Vagas abertas</div><div class="kpi"><b>${q.vagas_em_processamento}</b>Em análise</div><div class="kpi"><b>${q.tarefas_pendentes}</b>Na fila</div><div class="kpi ${q.pendencias ? 'alerta' : ''}"><b>${q.pendencias}</b>Pendências</div></div>
        <p class="mini" style="margin-top:8px">Contratação automática: <b>${au.contratacao ? 'ligada' : 'desligada (só propõe)'}</b> · reforço quando a fila passa de <b>${au.limite_fila}</b> por agente · estações ocupadas: <b>${q.agentes_ativos}/${q.estacoes}</b></p>${ia}
        <p style="margin:10px 0"><button class="btn" data-a="quadro">Abrir quadro geral</button> <button class="btn claro" data-a="equipe">💬 Canal da equipe</button></p>${pend}
        <div class="sec">Contratações e dispensas</div>${htmlContratacoes(contr.slice(0, 8))}
        <div class="sec">Histórico recente</div>${histHtml(d.historico)}`;
    }
    if (d.funcao === 'recepcao') {
      const q = D.quadro;
      return `${cartaoPerfil(d)}<div class="sec">Recepção</div><div class="g4"><div class="kpi"><b>${D.fila_triagem}</b>Em triagem</div><div class="kpi"><b>${D.fila_analise}</b>Aguardando analistas</div><div class="kpi"><b>${q.curriculos_recebidos}</b>Recebidos</div><div class="kpi"><b>${q.curriculos_triados}</b>Triados</div></div>
        <p class="mini" style="margin-top:10px">A recepção registra cada currículo, reconhece arquivos repetidos e faz a triagem por pasta (mesmas regras da API de triagem original).</p>${pend}<div class="sec">Histórico recente</div>${histHtml(d.historico)}`;
    }
    const s = d.estatisticas;
    const vagas = d.vagas_detalhe.map(v => `<div class="cartao"><div class="linha"><b>${esc(v.titulo)}</b><span class="chip" style="background:${v.sem_responsavel ? '#D64545' : STATUS_VAGA[v.status]}">${esc(v.situacao)}</span></div>
      ${v.descricao ? `<div class="mini" style="margin-top:3px">${esc(v.descricao)}</div>` : ''}${prog(v.analisados, v.recebidos)}
      Recebidos <b>${v.recebidos}</b> · Analisados <b>${v.analisados}</b> · Pendentes <b>${v.pendentes}</b>
      ${v.responsaveis.length > 1 ? `<div class="mini">Dividida com: ${v.responsaveis.filter(r => r.id !== d.id).map(r => esc(r.nome)).join(', ')}</div>` : ''}
      ${v.requisitos && v.requisitos.length ? `<div class="mini" style="margin-top:6px"><b>Principais requisitos:</b> ${v.requisitos.map(esc).join(' · ')}</div>` : ''}</div>`).join('') || '<p class="mini">Nenhuma vaga atribuída no momento.</p>';
    const extra = [s.ajudas_dadas ? `ajudou colegas ${s.ajudas_dadas} vez(es)` : '', s.segundas_opinioes ? `${s.segundas_opinioes} segunda(s) opinião(ões)` : ''].filter(Boolean).join(' · ');
    return `${cartaoPerfil(d)}<div class="sec">Estatísticas</div><div class="g4"><div class="kpi"><b>${s.recebidos}</b>Recebidos</div><div class="kpi"><b>${s.analisados}</b>Analisados</div><div class="kpi"><b>${s.pendentes}</b>Pendentes</div><div class="kpi"><b>${s.recomendados}</b>Recomendados</div></div>
      ${extra ? `<p class="mini" style="margin-top:6px">${esc(extra.charAt(0).toUpperCase() + extra.slice(1))}.</p>` : ''}
      ${pend}<div class="sec">Vagas sob responsabilidade · ${d.vagas_detalhe.length}</div>${vagas}<div class="sec">Histórico recente</div>${histHtml(d.historico)}
      ${gestor() ? `<p style="margin-top:14px"><button class="btn mini claro" data-a="dispensar" data-v="${d.id}">Dispensar da equipe</button> <span class="mini">libera a estação; as vagas passam para colegas da área</span></p>` : ''}`;
  }

  // o que o colega achou numa segunda leitura: ✔ prova que tinha passado batido, ⚠ ponto de atenção
  const achadosHtml = x => (x.achados || []).map(f => `<p class="mini achado ${f.tipo === 'alerta' ? 'alerta' : 'ok'}">${f.tipo === 'alerta' ? '⚠' : '✔'}
    ${f.requisito ? `<b>${esc(f.requisito)}</b>: ` : ''}${f.trecho ? `“${esc(f.trecho)}”` : ''}${f.comentario ? ` <span>— ${esc(f.comentario)}</span>` : ''}</p>`).join('');
  function colaboracoesHtml(c) {
    return (c.colaboracoes || []).map(x => x.tipo === 'segunda_leitura'
      ? `<div class="mini colab"><b>Segunda leitura de ${esc(x.colega)}:</b> ${(x.achados || []).length ? `achou ${x.achados.length} ponto(s) que tinham passado batido` : 'concordou com a análise'}${achadosHtml(x)}</div>`
      : `<div class="mini colab"><b>Consultou ${esc(x.colega)}:</b> “${esc(x.resposta)}”</div>`).join('');
  }

  async function candidatosAgente(d, get) {
    if (d.funcao !== 'analista') return '<p class="vazio">Este agente não analisa currículos. Abra um analista para ver candidatos recomendados.</p>';
    if (!d.vagas_detalhe.length) return '<p class="vazio">Sem candidatos: nenhuma vaga atribuída a este agente.</p>';
    if (P.vaga === undefined) P.vaga = d.vagas_detalhe[0].id;
    P.filtro = P.filtro || 'recomendados';
    const lista = await get(`/api/agentes/${d.id}/candidatos${P.vaga ? '?vaga_id=' + P.vaga : ''}`);
    const cont = { recomendados: lista.filter(c => c.recomendado).length, humana: lista.filter(c => c.avaliacao_humana).length, todos: lista.length };
    const vis = lista.filter(c => P.filtro === 'todos' || (P.filtro === 'recomendados' ? c.recomendado : c.avaliacao_humana));
    const cards = vis.map(c => {
      const [cc, cn_] = COMPAT[c.compatibilidade] || COMPAT.BAIXA, dec = DECISAO[c.decisao];
      return `<div class="cartao clica" data-a="cv" data-v="${c.candidatura_id}">
        <div class="linha"><b>${esc(c.nome)}</b><span class="chip" style="background:${c.avaliacao_humana && !c.recomendado ? '#F08A24' : cc}">${c.avaliacao_humana && !c.recomendado ? 'Avaliação humana' : cn_}</span></div>
        <div class="mini">${esc(c.vaga)}${c.tipo === 'segunda_opiniao' ? ' · segunda opinião' : ''}${ORIGEM[c.origem] ? ' · ' + ORIGEM[c.origem] : ''}</div>
        ${c.resumo ? `<div style="margin-top:4px">${esc(c.resumo)}</div>` : ''}
        ${c.experiencias.length ? `<div class="mini" style="margin-top:4px"><b>Experiências:</b> ${c.experiencias.slice(0, 2).map(esc).join(' · ')}</div>` : ''}
        ${c.competencias.length ? `<div class="mini"><b>Competências:</b> ${c.competencias.map(esc).join(' · ')}</div>` : ''}
        ${c.atendidos.length ? `<div class="mini ok" style="margin-top:4px"><b>Compatível:</b> ${c.atendidos.map(x => esc(x.requisito)).join(' · ')}</div>` : ''}
        ${c.a_verificar.length ? `<div class="mini alerta"><b>A verificar:</b> ${c.a_verificar.map(esc).join(' · ')}</div>` : ''}
        ${c.incompatibilidades.length ? `<div class="mini ruim"><b>Incompatível:</b> ${c.incompatibilidades.map(esc).join(' · ')}</div>` : ''}
        ${colaboracoesHtml(c)}
        <div class="mini" style="margin-top:4px"><b>Justificativa:</b> ${esc(c.justificativa)}</div>
        <div class="linha" style="margin-top:8px"><span>${dec ? `<span class="chip" style="background:${dec[0]}">${dec[1]}</span>` : '<span class="chip claro">Aguardando decisão do recrutador</span>'}</span><b>Ver currículo ›</b></div></div>`;
    }).join('') || `<p class="vazio">${lista.length ? 'Nenhum candidato neste filtro.' : 'Ainda não há currículos analisados para esta vaga.'}</p>`;
    return d.vagas_detalhe.map(v => `<span class="pilula ${v.id === P.vaga ? 'on' : ''}" data-a="vaga" data-v="${v.id}">${esc(v.titulo)}</span>`).join('') +
      `<div style="margin-top:4px">${[['recomendados', 'Recomendados'], ['humana', 'Avaliação humana'], ['todos', 'Todos analisados']].map(([k, n]) => `<span class="pilula ${P.filtro === k ? 'on' : ''}" data-a="filtro" data-v="${k}">${n} · ${cont[k]}</span>`).join('')}</div>
      <div class="sec">Candidatos · ordenados pela evidência</div>${cards}`;
  }

  // ---------------------------------------------------------------- chat individual (resposta em streaming)
  // Quem está com o chat aberto se vira para a tela e acena; enquanto pensa, põe a mão no queixo; ao responder, fala
  // gesticulando. A resposta aparece aos poucos, como se estivesse sendo digitada.
  function vidaDoChat() {
    const id = P && P.tipo === 'agente' && P.aba === 'chat' ? P.id : null;
    for (const k of Object.keys(CHAT_VIVO)) if (+k !== id) delete CHAT_VIVO[k];
    if (id && !CHAT_VIVO[id]) CHAT_VIVO[id] = { modo: 'acena', ate: performance.now() + 1900 };
  }
  const pontos = '<span class="pontos"><i></i><i></i><i></i></span>';
  const msgHtml = (x, id) => x.autor === 'usuario' ? `<div class="msg eu">${esc(x.texto)}</div>`
    : `<div class="lin">${av(id, 26)}<div class="msg ${x.digitando ? 'digitando' : ''}">${x.digitando ? pontos + ' ' + esc(x.texto) : esc(x.texto)}</div></div>`;
  // cumprimento do jeito de cada um, com a hora do dia e o que está fazendo agora (nada de texto de sistema)
  const AGORA_PADRAO = { trabalhando: 'Tô no meio de uma análise, mas pode falar.', descansando: 'Tava numa pausa rapidinha.',
    aguardando: 'Tô com a fila tranquila agora.', pendencia: 'Tô resolvendo uma pendência aqui.', disponivel: 'Tô livre, pode mandar.',
    concluido: 'Acabei de fechar a fila!' };
  function saudacao(a, pf) {
    const h = new Date().getHours(), periodo = h < 12 && h >= 5 ? 'Bom dia' : h < 18 && h >= 12 ? 'Boa tarde' : 'Boa noite';
    const estado = (a && a.estado) || 'disponivel', agora = (pf.agora || {})[estado] || AGORA_PADRAO[estado] || '';
    const modelo = pick(pf.saudacoes) || '{periodo}! {nome} aqui. {agora}';
    return modelo.replace('{periodo}', periodo).replace('{nome}', (a && a.nome) || '').replace('{agora}', agora)
      .replace(/([,—-] )(Bom|Boa)\b/g, (_, p1, p2) => p1 + p2.toLowerCase()).replace(/\s+/g, ' ').trim();
  }
  function chatAgente(d) {
    const a = D.agentes.find(x => x.id === d.id), pf = (a && a.perfil) || d.perfil || {};
    const m = CHATS[d.id] = CHATS[d.id] || [{ autor: 'agente', texto: saudacao(a || d, pf) }];
    const sug = d.funcao === 'analista' ? ['Como está o andamento?', 'Quais candidatos são mais compatíveis?', 'Quantos currículos faltam?', 'Quem precisa de avaliação humana?', 'Pediu ajuda a algum colega?']
      : d.funcao === 'supervisor' ? ['Como está a equipe?', 'Quem você contratou?', 'Quem está mais ocupado?'] : ['Quantos currículos chegaram?', 'Quem é você?'];
    return `<div id="msgs">${m.map(x => msgHtml(x, d.id)).join('')}</div>
      <div style="margin-top:10px">${sug.map(s => `<span class="pilula" data-a="sug" data-v="${esc(s)}">${esc(s)}</span>`).join('')}</div>
      <p class="mini" style="margin-top:6px">${D.ia_modo === 'gemini' ? 'Respostas geradas a partir dos registros deste agente, no jeito de falar dele.' : 'Modo demonstração: respostas por regras, a partir dos registros reais.'} As conversas não são armazenadas.</p>`;
  }
  function redesenharMsgs(id, soUltima) {
    if (!P || P.tipo !== 'agente' || P.id !== id || P.aba !== 'chat') return;
    const box = $('#msgs'); if (!box) return;
    const pb = $('#pb'), noFim = !pb || pb.scrollHeight - pb.scrollTop - pb.clientHeight < 90;
    const ult = CHATS[id][CHATS[id].length - 1], el = box.lastElementChild && box.lastElementChild.querySelector('.msg');
    if (soUltima && el && ult && !ult.digitando) { el.classList.remove('digitando'); el.textContent = ult.texto; }
    else { box.innerHTML = CHATS[id].map(x => msgHtml(x, id)).join(''); desenharAvatares(box); }
    if (pb && noFim) pb.scrollTop = 1e6;
  }
  async function enviarChat(texto) {
    texto = (texto || '').trim(); if (!texto || !P || P.tipo !== 'agente') return;
    const id = P.id, m = CHATS[id], a = D.agentes.find(x => x.id === id), pf = (a && a.perfil) || {};
    const hist = m.filter(x => !x.digitando).slice(-10);
    const pensando = pick(pf.pensando) || 'pensando…';
    const resp = { autor: 'agente', texto: pensando, digitando: true };
    m.push({ autor: 'usuario', texto }, resp);
    redesenharMsgs(id);
    balao(id, 'pensamento', pensando);
    const vivo = modo => { if (CHAT_VIVO[id]) CHAT_VIVO[id] = { modo, ate: performance.now() + 1500 }; };
    vivo('pensa');
    // a resposta chega aos pedaços; na tela ela "é digitada" num ritmo constante, alcançando o que já chegou
    let recebido = '', mostrado = 0, fim = false;
    const digitar = new Promise(pronto => {
      const passo = () => {
        if (mostrado < recebido.length) {
          if (resp.digitando) { resp.digitando = false; vivo('fala'); redesenharMsgs(id); }
          mostrado = Math.min(recebido.length, mostrado + Math.max(2, Math.ceil((recebido.length - mostrado) / 14)));
          resp.texto = recebido.slice(0, mostrado); redesenharMsgs(id, true);
        } else if (fim) { pronto(); return; }
        setTimeout(passo, 28);
      };
      passo();
    });
    try {
      const r = await fetch(`/api/agentes/${id}/chat/stream`, { method: 'POST', credentials: 'same-origin',
        headers: { 'X-Hub': '1', 'Content-Type': 'application/json' }, body: JSON.stringify({ mensagem: texto, historico: hist }) });
      if (r.status === 401) { mostrarLogin(); throw new Error('sessão expirada'); }
      if (!r.ok) throw new Error('Erro ' + r.status);
      if (!r.body || !r.body.getReader) recebido = await r.text();
      else {
        const leitor = r.body.getReader(), dec = new TextDecoder();
        for (;;) { const { value, done } = await leitor.read(); if (done) break; recebido += dec.decode(value, { stream: true }); }
        recebido += dec.decode();
      }
      recebido = recebido.trim() || 'Não consegui formular uma resposta agora.';
    } catch (e) { recebido = 'Não consegui responder agora: ' + e.message; }
    fim = true;
    await digitar;
    resp.digitando = false; resp.texto = recebido; redesenharMsgs(id, true);
    setTimeout(() => vivo('ouve'), 900);
    const frase = recebido.split(/(?<=[.!?])\s/)[0];
    if (frase) balao(id, 'fala', frase.length > 140 ? frase.slice(0, 137) + '…' : frase);
  }

  // ---------------------------------------------------------------- canal da equipe
  async function painelEquipe(get) {
    if (!EQUIPE.carregado) {
      if (get && get.soCache) throw SEM_CACHE;
      const r = await api('GET', '/api/equipe/conversas');
      EQUIPE.servidor = r.eventos; EQUIPE.carregado = true;
      for (const [id, a] of Object.entries(r.agentes)) AGS[id] = AGS[id] || a;
    }
    const sug = ['Como está o dia?', 'Quem está mais ocupado?', 'Precisamos contratar alguém?', 'Bia, quantos currículos chegaram?', 'Algum candidato forte?'];
    return `<div class="ph"><div style="min-width:0"><h2>Equipe</h2><small>Canal da equipe · ${D.agentes.length} agentes</small>
        <em>Fale com o time e acompanhe o que conversam entre si: pedidos de ajuda, contratações e boletins.</em>
        <div class="rostos">${D.agentes.map(a => av(a.id, 26)).join('')}</div></div>
        <button class="x" data-a="x" aria-label="Fechar">×</button></div>
      <div id="pb"><div id="canal">${htmlCanal()}</div></div>
      <div class="sugs">${sug.map(s => `<span class="pilula" data-a="sugeq" data-v="${esc(s)}">${esc(s)}</span>`).join('')}</div>
      <div id="entrada"><input id="qeq" maxlength="1000" placeholder="Fale com a equipe (ex.: Sofia, como está a vaga?)"><button class="btn" data-a="enviareq">Enviar</button></div>`;
  }
  function itensCanal() {
    const srv = EQUIPE.servidor.map(e => ({ ts: e.ts, tipo: e.tipo, agente_id: e.agente_id, para: (e.dados || {}).destino_id,
      de_nome: (e.dados || {}).de, para_nome: (e.dados || {}).para, texto: e.mensagem, momento: (e.dados || {}).momento }));
    return [...srv, ...EQUIPE.local].sort((a, b) => (a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : 0));
  }
  function htmlCanal() {
    const it = itensCanal();
    if (!it.length) return '<p class="vazio">Ainda sem conversas. Quando os agentes pedirem ajuda uns aos outros, o Atlas contratar alguém ou você escrever aqui, aparece neste canal.</p>';
    return it.map(x => {
      if (x.autor === 'usuario') return `<div class="eq eu"><div class="bal">${esc(x.texto)}</div></div>`;
      if (x.tipo !== 'conversa' && x.tipo !== 'resposta') return `<div class="eq sis">${esc(x.texto)} <span>${hora(x.ts)}</span></div>`;
      const nome = (AGS[x.agente_id] || {}).nome || x.de_nome || '?', dest = (x.para && (AGS[x.para] || {}).nome) || x.para_nome;
      return `<div class="eq ${x.digitando ? 'digitando' : ''}">${av(x.agente_id, 30)}<div class="corpo">
        <div class="quem"><b>${esc(nome)}</b>${x.momento ? ` <span class="mini">· ${ICONE_MOMENTO[x.momento.tipo] || '💬'} ${esc(x.momento.rotulo || '')}</span>` : dest ? ` <span class="mini">→ ${esc(dest)}</span>` : ''} <span class="mini">${hora(x.ts)}</span></div>
        <div class="bal">${x.digitando ? pontos + ' ' : ''}${esc(x.texto.replace(/^💭 /, ''))}</div></div></div>`;
    }).join('');
  }
  function atualizarCanal() {
    const box = $('#canal'), pb = $('#pb'); if (!box || !pb) return;
    const noFim = pb.scrollHeight - pb.scrollTop - pb.clientHeight < 80;
    box.innerHTML = htmlCanal(); desenharAvatares(box);
    if (noFim) pb.scrollTop = 1e6;
  }
  async function enviarEquipe(texto) {
    texto = (texto || '').trim(); if (!texto) return;
    const agora = () => new Date().toISOString();
    const hist = EQUIPE.local.filter(x => !x.digitando).slice(-8).map(x => ({ autor: x.autor === 'usuario' ? 'usuario' : 'agente',
      nome: x.autor === 'usuario' ? 'Recrutador' : (AGS[x.agente_id] || {}).nome, texto: x.texto }));
    EQUIPE.local.push({ autor: 'usuario', texto, ts: agora() });
    const atlas = D.agentes.find(a => a.funcao === 'supervisor');
    const aguardo = { tipo: 'resposta', agente_id: atlas && atlas.id, texto: '💭 a equipe está pensando…', digitando: true, ts: agora() };
    EQUIPE.local.push(aguardo); atualizarCanal(); if ($('#pb')) $('#pb').scrollTop = 1e6;
    let r;
    try { r = await api('POST', '/api/equipe/chat', { mensagem: texto, historico: hist }); }
    catch (e) { r = { respostas: [{ agente_id: atlas && atlas.id, texto: 'Não consegui responder agora: ' + e.message }] }; }
    EQUIPE.local.splice(EQUIPE.local.indexOf(aguardo), 1);
    for (const f of r.respostas) {
      const pf = (D.agentes.find(a => a.id === f.agente_id) || {}).perfil || {}, pensando = pick(pf.pensando) || 'pensando…';
      const item = { tipo: 'resposta', agente_id: f.agente_id, texto: '💭 ' + pensando, digitando: true, ts: agora() };
      EQUIPE.local.push(item); atualizarCanal();
      balao(f.agente_id, 'pensamento', pensando);
      await espera(Math.min(2200, 500 + f.texto.length * 12));
      Object.assign(item, { texto: f.texto, digitando: false, ts: agora() });
      atualizarCanal();
      balao(f.agente_id, 'fala', f.texto.length > 140 ? f.texto.slice(0, 137) + '…' : f.texto);
    }
  }

  // ---------------------------------------------------------------- vagas e quadro
  async function painelVagas(get) {
    const [vagas, q] = await Promise.all([get('/api/vagas'), get('/api/quadro')]);
    const pendVaga = Object.fromEntries(q.pendencias.filter(p => p.tipo === 'SEM_ESPECIALISTA' && p.vaga_id).map(p => [p.vaga_id, p]));
    const f = P.filtro || 'todas';
    const filtros = [['todas', 'Todas', () => true], ['abertas', 'Abertas', v => v.status === 'aberta'], ['sem', 'Sem responsável', v => v.sem_responsavel],
      ['analise', 'Em análise', v => v.status === 'aberta' && v.pendentes > 0], ['outras', 'Suspensas/encerradas', v => v.status !== 'aberta']];
    const ativo = filtros.find(x => x[0] === f);
    const lista = vagas.filter(ativo[2]).map(v => {
      const pv = pendVaga[v.id];
      return `<div class="cartao"><div class="linha"><b>${esc(v.titulo)}</b><span class="chip" style="background:${v.sem_responsavel ? '#D64545' : STATUS_VAGA[v.status]}">${esc(v.situacao)}</span></div>
        <div class="mini">${esc([v.area, v.local].filter(Boolean).join(' · ') || '—')}</div>${prog(v.analisados, v.recebidos)}
        Candidatos recebidos <b>${v.recebidos}</b> · Analisados <b>${v.analisados}</b>
        <div class="mini" style="margin-top:4px"><b>Pastas da triagem:</b> ${v.setores.map(esc).join(', ')}</div>
        <div class="linha" style="margin-top:8px"><span>${v.responsaveis.length ? 'Responsável: ' + v.responsaveis.map(r => `<a href="#" data-a="ag" data-v="${r.id}"><b>${esc(r.nome)}</b></a>`).join(', ') : '<b class="ruim">Sem responsável</b>'}</span>
        ${gestor() ? `<select class="sel" data-a="status" data-v="${v.id}">${['aberta', 'suspensa', 'encerrada', 'cancelada'].map(s => `<option ${s === v.status ? 'selected' : ''} value="${s}">${s[0].toUpperCase() + s.slice(1)}</option>`).join('')}</select>` : ''}</div>
        ${pv ? `<div class="cartao pend amarelo" style="margin:10px 0 0"><b>Atlas:</b> ${esc(pv.descricao)}.${gestor() ? `<br><button class="btn mini escuro" style="margin-top:6px" data-a="resolver" data-v="${pv.id}" data-acao="ativar">Contratar agente especializado</button>` : ''}</div>` : ''}</div>`;
    }).join('') || '<p class="vazio">Nenhuma vaga neste filtro.</p>';
    return `<div class="ph"><div><h2>Vagas em aberto</h2><small>${vagas.filter(v => v.status === 'aberta').length} abertas · visão do recrutador</small><em>O quadro na sala é usado pelos agentes; aqui está a análise.</em></div><button class="x" data-a="x">×</button></div>
      <div id="pb">${filtros.map(([k, n, fn]) => `<span class="pilula ${f === k ? 'on' : ''}" data-a="fvaga" data-v="${k}">${n} · ${vagas.filter(fn).length}</span>`).join('')}${lista}</div>`;
  }

  // linha de consumo da IA (gestores, com Gemini ligado)
  async function linhaIA(get = daRede) {
    if (!gestor() || D.ia_modo !== 'gemini') return '';
    try {
      const u = await get('/api/ia/uso'), h = u.hoje, lim = u.limite_diario_usd;
      return `<p class="mini" style="margin-top:8px">🤖 IA hoje: <b>${usd(h.custo_usd)}</b>${lim > 0 ? ` de ${usd(lim)}` : ''} · ${milhar(h.chamadas)} chamada(s) · ${esc(u.modelo)}${u.pausa ? ` · <b class="ruim">pausada</b>` : ''} · <a href="#" data-a="uso-ia">detalhes</a></p>`;
    } catch (e) { if (e === SEM_CACHE) throw e; return ''; }
  }

  async function painelQuadro(get) {
    const [q, contr, ia] = await Promise.all([get('/api/quadro'), get('/api/contratacoes'), linhaIA(get)]), r = q.resumo;
    const pend = q.pendencias.map(p => {
      let acoes = '';
      if (q.pode_gerir) {
        if (p.tipo === 'SEM_ESPECIALISTA') acoes += `<button class="btn mini escuro" data-a="resolver" data-v="${p.id}" data-acao="ativar">Contratar agente</button> `;
        if (p.tipo === 'PROPOSTA_CONTRATACAO') acoes += `<button class="btn mini escuro" data-a="resolver" data-v="${p.id}" data-acao="ativar">Aprovar contratação</button> `;
        if (p.tipo === 'TRIAGEM_MANUAL') acoes += `<select class="sel" id="setor-${p.id}">${CATEGORIAS.filter(c => c !== 'OUTROS').map(c => `<option>${esc(c)}</option>`).join('')}</select> <button class="btn mini escuro" data-a="resolver" data-v="${p.id}" data-acao="definir_setor">Encaminhar</button> `;
        if (p.tipo === 'ERRO_IA') acoes += `<button class="btn mini escuro" data-a="resolver" data-v="${p.id}" data-acao="reprocessar">Tentar de novo</button> `;
        if (p.tipo === 'IA_INDISPONIVEL') acoes += (EU.papel === 'admin' ? '<button class="btn mini escuro" data-a="testar-ia">Testar IA agora</button> ' : '') +
          '<button class="btn mini claro" data-a="uso-ia">Ver uso da IA</button> <span class="mini">some sozinha quando a IA voltar</span>';
        else acoes += `<button class="btn mini claro" data-a="resolver" data-v="${p.id}" data-acao="resolver">${p.tipo === 'PROPOSTA_CONTRATACAO' ? 'Recusar' : 'Marcar resolvida'}</button>`;
      }
      return `<div class="cartao pend ${['PROPOSTA_CONTRATACAO', 'IA_INDISPONIVEL'].includes(p.tipo) ? 'amarelo' : ''}"><div class="mini">${esc(p.agente || 'Sistema')} · ${hora(p.criado_em)}</div><div style="margin:3px 0 8px">${esc(p.descricao)}</div>
        ${p.candidatura_id ? `<a href="#" data-a="cv" data-v="${p.candidatura_id}">Ver currículo ›</a> ` : ''}${acoes}</div>`;
    }).join('') || '<p class="mini">Nenhuma pendência aberta.</p>';
    const eq = q.equipe.map(a => { const [c, n] = ESTADOS[a.estado] || ESTADOS.aguardando; return `<div class="cartao clica" style="padding:8px 12px;margin-bottom:6px" data-a="ag" data-v="${a.id}"><div class="linha"><span><i class="d" style="background:${c}"></i> <b>${esc(a.nome)}</b> <span class="mini">${esc(a.especialidade)}${a.carga ? ' · ' + a.carga + ' na fila' : ''}</span></span><b style="color:${c}">${n}</b></div></div>`; }).join('');
    return `<div class="ph"><div><h2>Quadro geral</h2><small>Coordenação · Atlas (supervisor)</small><em>Números calculados a partir dos registros do sistema.</em></div><button class="x" data-a="x">×</button></div>
      <div id="pb"><div class="g4">
        <div class="kpi"><b>${r.vagas_abertas}</b>Vagas abertas</div><div class="kpi"><b>${r.vagas_em_processamento}</b>Em análise</div>
        <div class="kpi ${r.vagas_sem_responsavel ? 'alerta' : ''}"><b>${r.vagas_sem_responsavel}</b>Sem responsável</div><div class="kpi"><b>${r.tarefas_pendentes}</b>Tarefas na fila</div>
        <div class="kpi"><b>${r.curriculos_recebidos}</b>Currículos recebidos</div><div class="kpi"><b>${r.analises_concluidas}</b>Análises feitas</div>
        <div class="kpi"><b>${r.consultas_entre_colegas}</b>Ajudas entre colegas</div><div class="kpi ${r.pendencias ? 'alerta' : ''}"><b>${r.pendencias}</b>Pendências</div></div>${ia}
        <div class="sec">Problemas e propostas que precisam de você</div>${pend}
        <div class="sec">Equipe · ${r.agentes_ativos}/${r.estacoes} estações</div>${eq}
        <div class="sec">Contratações e dispensas</div>${htmlContratacoes(contr.slice(0, 10))}</div>`;
  }

  // cliques no painel
  $('#pi').addEventListener('click', async e => {
    const t = e.target.closest('[data-a]'); if (!t) return;
    const a = t.dataset.a, v = t.dataset.v;
    if (t.tagName === 'A') e.preventDefault();
    if (a === 'x') fechar();
    else if (a === 'aba') { P.aba = v; pintarPainel(); }
    else if (a === 'vaga') { P.vaga = +v; pintarPainel(); }
    else if (a === 'filtro') { P.filtro = v; pintarPainel(); }
    else if (a === 'fvaga') { P.filtro = v; pintarPainel(); }
    else if (a === 'cv') abrirCurriculo(+v);
    else if (a === 'ag') abrir({ tipo: 'agente', id: +v, aba: 'resumo' });
    else if (a === 'quadro') abrir({ tipo: 'quadro' });
    else if (a === 'equipe') abrir({ tipo: 'equipe' });
    else if (a === 'uso-ia') modalAdmin('ia');
    else if (a === 'testar-ia') modalAdmin('ia', true);
    else if (a === 'sug') enviarChat(v);
    else if (a === 'enviar') { const q = $('#q'); enviarChat(q.value); q.value = ''; }
    else if (a === 'sugeq') enviarEquipe(v);
    else if (a === 'enviareq') { const q = $('#qeq'); enviarEquipe(q.value); q.value = ''; }
    else if (a === 'dispensar') {
      if (!confirm('Dispensar este agente? A estação fica livre e as vagas passam para colegas da mesma área.')) return;
      t.disabled = true;
      try { await api('POST', `/api/agentes/${v}/dispensar`); fechar(); await carregar(); } catch (er) { alert(er.message); t.disabled = false; }
    }
    else if (a === 'resolver') {
      t.disabled = true;
      const corpo = { acao: t.dataset.acao };
      if (corpo.acao === 'definir_setor') corpo.setor = $('#setor-' + v).value;
      try { await api('POST', `/api/pendencias/${v}/resolver`, corpo); await carregar(); pintarPainel(true); } catch (er) { alert(er.message); t.disabled = false; }
    }
  });
  $('#pi').addEventListener('change', async e => {
    const t = e.target.closest('[data-a="status"]'); if (!t) return;
    try { await api('PATCH', `/api/vagas/${t.dataset.v}`, { status: t.value }); await carregar(); pintarPainel(true); } catch (er) { alert(er.message); }
  });
  $('#pi').addEventListener('keydown', e => {
    if (e.key !== 'Enter') return;
    if (e.target.id === 'q') { enviarChat(e.target.value); e.target.value = ''; }
    if (e.target.id === 'qeq') { enviarEquipe(e.target.value); e.target.value = ''; }
  });

  // ================================================================ currículo + análise
  function realcar(texto, evidencias) {
    const norm = s => s.toLowerCase();
    const faixas = [];
    evidencias.forEach((ev, i) => {
      const alvo = (ev || '').trim(); if (alvo.length < 4) return;
      const ix = norm(texto).indexOf(norm(alvo));
      if (ix >= 0 && !faixas.some(f => ix < f[1] && ix + alvo.length > f[0])) faixas.push([ix, ix + alvo.length, i + 1]);
    });
    faixas.sort((a, b) => a[0] - b[0]);
    let out = '', pos = 0;
    for (const [a, b, n] of faixas) { out += esc(texto.slice(pos, a)) + `<mark><i class="num">${n}</i>${esc(texto.slice(a, b))}</mark>`; pos = b; }
    return out + esc(texto.slice(pos));
  }

  async function abrirCurriculo(cid) {
    const w = $('#cvw');
    w.innerHTML = '<div class="cw"><header><b>Carregando currículo…</b></header></div>'; w.style.display = 'block';
    let d;
    try { d = await api('GET', `/api/candidaturas/${cid}`); } catch (e) { w.innerHTML = `<div class="cw"><header><b>${esc(e.message)}</b><button class="x" data-x>×</button></header></div>`; return; }
    const principal = d.analises.find(a => a.tipo === 'principal') || d.analises[0];
    const evid = principal ? principal.atendidos.map(x => x.evidencia) : [];
    const cv = d.curriculo, c = d.candidato;
    const papel = cv.texto
      ? `<pre>${realcar(cv.texto, evid)}</pre><p class="mini" style="margin-top:14px">Trechos destacados = evidências usadas na análise (numeradas como ao lado).</p>`
      : `<h1>${esc(c.nome)}</h1><p class="mini">Este arquivo não tem texto extraível (imagem ou PDF escaneado). A IA leu o arquivo original.</p>`;
    const analises = d.analises.map(a => {
      const [cc, cn_] = COMPAT[a.compatibilidade] || COMPAT.BAIXA;
      const colab = (a.colaboracoes || []).map(x => x.tipo === 'segunda_leitura'
        ? `<div class="cartao colabc"><b>${esc(a.agente)} pediu uma segunda leitura a ${esc(x.colega)}</b> <span class="mini">(${esc(x.especialidade)})</span>
          <p class="mini" style="margin-top:4px"><b>Dúvida:</b> ${esc(x.pergunta)}</p><p class="mini"><b>${esc(x.colega)}:</b> ${esc(x.resposta)}</p>
          ${achadosHtml(x) || '<p class="mini">Não achou nada diferente: concordou com a análise.</p>'}</div>`
        : `<div class="cartao colabc"><b>${esc(a.agente)} consultou ${esc(x.colega)}</b> <span class="mini">(${esc(x.especialidade)})</span>
        <p class="mini" style="margin-top:4px"><b>Pergunta:</b> ${esc(x.pergunta)}</p><p class="mini"><b>Resposta:</b> ${esc(x.resposta)}</p>
        ${(x.evidencias || []).length ? `<p class="mini"><b>Trecho:</b> “${x.evidencias.map(esc).join('” · “')}”</p>` : ''}</div>`).join('');
      return `<div class="cartao"><div class="linha"><h3 style="font-size:16px">${a.tipo === 'segunda_opiniao' ? 'Segunda opinião de ' : 'Análise de '}${esc(a.agente)}</h3><span class="chip" style="background:${cc}">${cn_}</span></div>
        <div class="mini">Vaga: ${esc(d.vaga.titulo)} · ${a.recomendado ? '<b class="ok">recomendado</b>' : a.avaliacao_humana ? '<b class="alerta">avaliação humana sugerida</b>' : 'não recomendado'}${a.modelo === 'demo-heuristica' ? ' · análise heurística (demonstração)' : ''}</div>
        ${a.atendidos.length ? `<div class="sec">Evidências no currículo</div>${a.atendidos.map((x, i) => `<p style="margin:6px 0"><i class="num">${i + 1}</i><b>${esc(x.requisito)}</b><br><span class="mini" style="margin-left:23px">“${esc(x.evidencia)}”</span></p>`).join('')}` : ''}
        ${a.a_verificar.length ? `<div class="sec">A verificar</div><ul class="lista alerta">${a.a_verificar.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}
        ${a.incompatibilidades.length ? `<div class="sec">Incompatibilidades</div><ul class="lista ruim">${a.incompatibilidades.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}
        ${colab ? `<div class="sec">${(a.colaboracoes || []).some(x => x.tipo === 'segunda_leitura') ? 'Outro olhar do time' : 'Conversa com colega'}</div>${colab}` : ''}
        <div class="sec">Justificativa</div><p>${esc(a.justificativa)}</p></div>`;
    }).join('') || '<p class="vazio">Análise ainda não concluída.</p>';
    const atual = DECISAO[d.decisao];
    const origem = d.origem === 'indicacao' ? `Incluído nesta vaga a partir de uma conversa com ${esc(d.indicado_por || 'um colega')}.` : ORIGEM[d.origem] ? 'Origem: ' + ORIGEM[d.origem] + '.' : '';
    w.innerHTML = `<div class="cw"><header><button class="voltar" data-x>‹ Voltar</button><b>Currículo · ${esc(c.nome)}</b><button class="x" data-x>×</button></header>
      <div class="cb"><div class="papel">${papel}
        <p style="margin-top:12px"><a href="/api/curriculos/${cv.id}/arquivo" target="_blank" rel="noopener">Abrir arquivo original</a> · <span class="mini">pasta da triagem: ${esc(cv.setor || '—')} (${esc(cv.confianca || '—')})</span></p></div>
        <div class="analise">
          <div class="cartao"><b>${esc(c.nome)}</b><div class="mini">${esc([c.cidade, c.email, c.telefone].filter(Boolean).join(' · '))}</div>
          ${origem ? `<div class="mini" style="margin-top:4px">${origem}</div>` : ''}
          ${c.dado_sensivel ? '<div class="mini alerta" style="margin-top:4px">Contém dado pessoal sensível declarado pelo candidato — use apenas para a finalidade da vaga.</div>' : ''}</div>
          ${analises}
          <div class="cartao"><div class="sec" style="margin-top:0">Decisão final: recrutador</div>
            ${atual ? `<p>Decisão atual: <span class="chip" style="background:${atual[0]}">${atual[1]}</span></p>` : '<p class="mini">A IA recomenda; quem decide é você.</p>'}
            <div class="decisao">${Object.entries(DECISAO).map(([k, [, n]]) => `<button class="btn mini ${k === 'entrevista' ? '' : 'claro'}" data-dec="${k}" data-c="${cid}">${n}</button>`).join('')}</div>
            ${d.decisoes.length ? `<div class="hist" style="margin-top:8px">${d.decisoes.map(x => `<p><b>${hora(x.em)}</b>${esc(DECISAO[x.decisao][1])}${x.observacao ? ' — ' + esc(x.observacao) : ''}</p>`).join('')}</div>` : ''}
            <p class="mini" style="margin-top:8px">Este acesso foi registrado para fins de auditoria (LGPD).</p></div>
        </div></div></div>`;
  }
  $('#cvw').addEventListener('click', async e => {
    if (e.target.closest('[data-x]') || e.target.id === 'cvw') { $('#cvw').style.display = 'none'; return; }
    const b = e.target.closest('[data-dec]'); if (!b) return;
    b.disabled = true;
    try { await api('POST', `/api/candidaturas/${b.dataset.c}/decisao`, { decisao: b.dataset.dec }); abrirCurriculo(+b.dataset.c); pintarPainel(true); }
    catch (er) { alert(er.message); b.disabled = false; }
  });

  // ================================================================ banco de currículos (o arquivo do escritório)
  // Pastas da triagem agrupadas pelas áreas da equipe; cada currículo mostra o que a IA disse e a vaga sugerida.
  const TIPOS_BANCO = ['curriculo_recebido', 'triagem_concluida', 'arquivo_invalido', 'banco_talentos', 'candidatura_encaminhada',
    'analise_concluida', 'decisao_humana', 'encaminhamento_manual', 'sugestao_banco', 'encaminhamento_automatico', 'lgpd'];
  const SIT_BANCO = [['', 'Todos'], ['talentos', 'Banco de talentos'], ['analise', 'Em análise'], ['recomendados', 'Recomendados'],
    ['humana', 'Avaliação humana'], ['decididos', 'Decididos'], ['triagem', 'Na triagem'], ['descartados', 'Descartados']];
  const COR_SIT = { 'banco de talentos': '#3B82C4', 'em análise': '#F08A24', recomendado: '#2E9E5B', 'avaliação humana': '#F08A24',
    analisado: '#6B7183', 'na triagem': '#8E6BBF', 'erro na triagem': '#D64545', descartado: '#5A6070',
    'chamado para entrevista': '#2E9E5B', 'guardado no banco de talentos': '#3B82C4', 'não seguir': '#6B7183' };
  const BANCO = { aberto: false, grupo: '', pasta: '', situacao: '', busca: '', pagina: 1, itens: [], tem_mais: false, resumo: null, sel: null, ficha: null };
  const pastaNome = s => (s || 'sem pasta').toLowerCase().replace(/(^|\s)\S/g, c => c.toUpperCase());

  function abrirBanco(filtro = {}) {
    // abre sempre do começo (todas as pastas, sem busca), a não ser que venha com um filtro
    Object.assign(BANCO, { aberto: true, pagina: 1, grupo: '', pasta: '', situacao: '', busca: '', sel: null, ficha: null }, filtro);
    $('#bancow').classList.remove('ficha');
    const w = $('#bancow');
    w.innerHTML = `<div class="bw"><header><span class="bt">${ICONE.arquivo}<b>Banco de currículos</b><span class="mini" id="btotal"></span></span>
        <input id="bbusca" type="search" placeholder="Buscar por nome, cidade ou palavra do currículo (ex.: Excel, empilhadeira)" value="${esc(BANCO.busca)}">
        <button class="x" data-bx aria-label="Fechar">×</button></header>
      <div class="bsit" id="bsit"></div>
      <div class="bcorpo"><nav class="bpastas" id="bpastas"></nav><section class="blista" id="blista"></section><aside class="bficha" id="bficha"></aside></div></div>`;
    w.classList.add('aberto');
    pintarBancoFiltros(); pintarBancoLista(true); pintarFicha();
    carregarBanco();
    let t = 0;
    $('#bbusca').oninput = e => { clearTimeout(t); t = setTimeout(() => { BANCO.busca = e.target.value; BANCO.pagina = 1; carregarBanco(); }, 280); };
  }
  function fecharBanco() { BANCO.aberto = false; $('#bancow').classList.remove('aberto'); $('#bancow').innerHTML = ''; }
  let tBanco = 0;
  const atualizarBancoLogo = () => { clearTimeout(tBanco); tBanco = setTimeout(() => { carregarBanco(true); if (BANCO.sel) abrirFicha(BANCO.sel, true); }, 900); };

  async function carregarBanco(silencioso) {
    const q = new URLSearchParams();
    for (const k of ['grupo', 'pasta', 'situacao', 'busca']) if (BANCO[k]) q.set(k, BANCO[k]);
    const pagina = silencioso ? 1 : BANCO.pagina; q.set('pagina', pagina);
    const chave = q.toString();
    BANCO.pedido = chave;
    if (!silencioso && pagina === 1) pintarBancoLista(true);
    try {
      const r = await api('GET', '/api/banco?' + chave);
      if (BANCO.pedido !== chave || !BANCO.aberto) return;
      BANCO.itens = pagina > 1 ? [...BANCO.itens, ...r.itens] : r.itens;
      BANCO.tem_mais = r.tem_mais; BANCO.resumo = r.resumo;
      if (!silencioso || pagina === 1) pintarBancoFiltros();
      pintarBancoLista(false);
    } catch (e) { if (BANCO.aberto) $('#blista').innerHTML = `<p class="erro">${esc(e.message)}</p>`; }
  }
  function pintarBancoFiltros() {
    const r = BANCO.resumo;
    if ($('#btotal')) $('#btotal').textContent = r ? ` · ${milhar(r.total)} nas pastas${r.na_triagem ? ` · ${r.na_triagem} na triagem` : ''}` : '';
    $('#bsit').innerHTML = SIT_BANCO.map(([k, n]) => `<span class="pilula ${BANCO.situacao === k ? 'on' : ''}" data-bsit="${k}">${n}</span>`).join('');
    if (!r) { $('#bpastas').innerHTML = '<i class="esq-l"></i><i class="esq-l" style="width:60%"></i>'.repeat(4); return; }
    $('#bpastas').innerHTML = `<div class="bg ${!BANCO.grupo && !BANCO.pasta ? 'on' : ''}" data-bgrupo=""><i class="pasta" style="--c:#F5B544"></i><b>Todas as pastas</b><span>${milhar(r.total)}</span></div>` +
      r.grupos.map(g => {
        const cor = Cena.corGrupo(g.codigo), cheias = g.pastas.filter(x => x.n);
        return `<div class="bgrupo ${g.total ? '' : 'sem-cv'}"><div class="bg ${BANCO.grupo === g.codigo && !BANCO.pasta ? 'on' : ''}" data-bgrupo="${esc(g.codigo)}">
            <i class="pasta" style="--c:${cor}"></i><b>${esc(g.nome)}</b><span>${milhar(g.total)}</span></div>
          ${cheias.map(x => `<div class="bp ${BANCO.pasta === x.setor ? 'on' : ''}" data-bpasta="${esc(x.setor)}" style="--c:${cor}">${esc(pastaNome(x.setor))}<span>${x.n}</span></div>`).join('')}</div>`;
      }).join('') +
      (r.descartados ? `<p class="mini" style="margin:10px 6px">${r.descartados} arquivo(s) descartado(s) na triagem (não eram currículos).</p>` : '');
  }
  function linhaSugestao(i) {
    const v = i.vaga_sugerida, at = i.sugestao_atlas;
    if (v && v.compatibilidade) {
      const [cc, cn_] = COMPAT[v.compatibilidade] || COMPAT.BAIXA;
      return `<div class="sug">💡 Vaga sugerida: <b>${esc(v.vaga)}</b> <span class="chip" style="background:${cc}">${cn_}</span>${v.analista ? ` <span class="mini">por ${esc(v.analista)}</span>` : ''}</div>`;
    }
    if (v) return `<div class="sug">⏳ Na vaga <b>${esc(v.vaga)}</b> <span class="mini">· aguardando a análise</span></div>`;
    if (at && at.vaga) return `<div class="sug">💡 Atlas sugere: <b>${esc(at.vaga)}</b> <span class="mini">· ${esc(at.motivo)}</span></div>`;
    if (at && at.antiga) return `<div class="sug mini">Sugestão antiga do Atlas: abra e peça de novo (critério novo).</div>`;
    if (at) return `<div class="sug">🧭 Atlas: nenhuma vaga aberta combina${at.perfil_indicado ? ` <span class="mini">· perfil: ${esc(at.perfil_indicado)}</span>` : ''}</div>`;
    if (i.status === 'triado') return `<div class="sug mini">Sem vaga aberta para esta pasta: o Atlas encaminha quando abrir uma.</div>`;
    return '';
  }
  function pintarBancoLista(carregando) {
    const el = $('#blista'); if (!el) return;
    if (carregando) { el.innerHTML = '<div class="cartao esq"><i class="esq-l" style="width:60%"></i><i class="esq-l"></i></div>'.repeat(5); return; }
    const filtro = BANCO.pasta ? pastaNome(BANCO.pasta) : BANCO.grupo ? ((BANCO.resumo.grupos.find(g => g.codigo === BANCO.grupo) || {}).nome || '') : '';
    el.innerHTML = (filtro || BANCO.busca ? `<p class="mini" style="margin:0 2px 8px">${filtro ? `Pasta: <b>${esc(filtro)}</b>` : ''}${BANCO.busca ? ` · busca: “${esc(BANCO.busca)}”` : ''} · ${BANCO.itens.length}${BANCO.tem_mais ? '+' : ''} currículo(s)</p>` : '') +
      (BANCO.itens.map(i => {
        const cor = Cena.corGrupo(i.grupo), parecer = i.vaga_sugerida && i.vaga_sugerida.parecer ? i.vaga_sugerida.parecer : i.resumo_triagem;
        return `<div class="bitem ${BANCO.sel === i.curriculo_id ? 'on' : ''}" data-bcv="${i.curriculo_id}" style="--c:${cor}">
          <div class="linha"><b>${esc(i.nome)}</b><span class="chip" style="background:${COR_SIT[i.situacao] || '#6B7183'}">${esc(i.situacao)}</span></div>
          <div class="mini">${esc([i.cidade && i.cidade !== 'Não informado' ? i.cidade : '', i.anos_experiencia ? i.anos_experiencia + ' ano(s) de experiência' : '', pastaNome(i.setor)].filter(Boolean).join(' · '))}</div>
          ${linhaSugestao(i)}
          ${parecer ? `<div class="mini parecer">“${esc(parecer.length > 170 ? parecer.slice(0, 167) + '…' : parecer)}”</div>` : ''}</div>`;
      }).join('') || '<p class="vazio">Nenhum currículo aqui. Os que chegarem por e-mail ou pelo botão Enviar currículos aparecem nas pastas depois da triagem.</p>') +
      (BANCO.tem_mais ? '<p style="text-align:center;margin:8px 0 14px"><button class="btn claro mini" data-bmais>Carregar mais</button></p>' : '');
  }
  // a opinião do Atlas sobre o currículo: a vaga (ou nenhuma), o porquê, o que sustenta, o que falta e o perfil
  function opiniaoAtlas(at) {
    if (at.antiga) return `<b>Sugestão antiga do Atlas</b><div class="mini">Foi feita antes do critério novo (experiência profissional conta; estudo e interesse, não). Peça de novo abaixo.</div>`;
    const conf = { ALTA: 'alta', MEDIA: 'média', BAIXA: 'baixa' }[at.confianca] || (at.confianca || '').toLowerCase();
    return `${at.vaga ? `💡 <b>Atlas sugere: ${esc(at.vaga)}</b> <span class="chip claro">confiança ${esc(conf)}</span>` : `🧭 <b>Atlas: nenhuma vaga aberta combina${at.mais_proxima ? ' bem' : ''}</b>`}
      ${at.motivo ? `<p style="margin:4px 0">${esc(at.motivo)}</p>` : ''}
      ${(at.evidencias || []).length ? `<div class="mini ok">✔ ${at.evidencias.map(x => '“' + esc(x) + '”').join(' · ')}</div>` : ''}
      ${(at.lacunas || []).length ? `<div class="mini ruim">✖ ${at.lacunas.map(esc).join(' · ')}</div>` : ''}
      ${!at.vaga && at.mais_proxima ? `<div class="mini">Mais próxima entre as abertas: ${esc(at.mais_proxima)}</div>` : ''}
      ${at.perfil_indicado ? `<div class="mini" style="margin-top:4px"><b>Perfil indicado para:</b> ${esc(at.perfil_indicado)}</div>` : ''}`;
  }
  async function abrirFicha(id, silencioso) {
    BANCO.sel = id;
    document.querySelectorAll('.bitem').forEach(e => e.classList.toggle('on', +e.dataset.bcv === id));
    if (!silencioso) { BANCO.ficha = null; pintarFicha(true); $('#bancow').classList.add('ficha'); }
    try {
      const d = await api('GET', `/api/banco/${id}`);
      if (BANCO.sel !== id || !BANCO.aberto) return;
      BANCO.ficha = d; pintarFicha();
    } catch (e) { if (BANCO.sel === id && $('#bficha')) $('#bficha').innerHTML = `<p class="erro">${esc(e.message)}</p>`; }
  }
  function pintarFicha(carregando) {
    const el = $('#bficha'); if (!el) return;
    const d = BANCO.ficha;
    if (carregando) { el.innerHTML = '<div class="cartao esq"><i class="esq-l" style="width:50%;height:18px"></i><i class="esq-l"></i><i class="esq-l" style="width:70%"></i></div>'.repeat(2); return; }
    if (!d) { el.innerHTML = '<p class="vazio">Escolha um currículo para ver o que a IA disse dele e a vaga sugerida.</p>'; return; }
    const c = d.candidato, tr = d.triagem, cor = Cena.corGrupo(tr.grupo), v = d.vaga_sugerida, at = d.sugestao_atlas;
    const analises = d.candidaturas.map(x => {
      const dec = DECISAO[x.decisao];
      const ans = (x.analises || []).map(a => {
        const [cc, cn_] = COMPAT[a.compatibilidade] || COMPAT.BAIXA;
        return `<div class="disse">${av(x.analista_id, 24)}<div><div><b>${esc(a.agente)}</b> <span class="mini">${a.tipo === 'segunda_opiniao' ? 'segunda opinião' : 'analisou'}</span> <span class="chip" style="background:${cc}">${cn_}</span>
          ${a.recomendado ? ' <b class="ok">recomendado</b>' : a.avaliacao_humana ? ' <b class="alerta">avaliação humana</b>' : ''}</div>
          <p>“${esc(a.justificativa)}”</p>
          ${a.atendidos.length ? `<div class="mini ok">Atende: ${a.atendidos.map(esc).join(' · ')}</div>` : ''}
          ${a.a_verificar.length ? `<div class="mini alerta">A verificar: ${a.a_verificar.map(esc).join(' · ')}</div>` : ''}</div></div>`;
      }).join('') || `<p class="mini">${x.status === 'erro' ? 'A análise falhou; veja as pendências no quadro geral.' : 'Na fila do analista da vaga.'}</p>`;
      return `<div class="cartao"><div class="linha"><b>${esc(x.vaga)}</b>${dec ? `<span class="chip" style="background:${dec[0]}">${dec[1]}</span>` : x.vaga_status !== 'aberta' ? `<span class="chip claro">vaga ${esc(x.vaga_status)}</span>` : ''}</div>
        ${ans}${x.analises && x.analises.length ? `<p style="margin-top:6px"><a href="#" data-bcand="${x.candidatura_id}">Ver análise completa e decidir ›</a></p>` : ''}</div>`;
    }).join('');
    const abertas = d.vagas_abertas.slice().sort((a, b) => b.da_pasta - a.da_pasta);
    const acoes = gestor() && d.status === 'triado' ? `<div class="cartao"><div class="sec" style="margin-top:0">Levar para uma vaga</div>
        ${abertas.length ? `<div style="display:flex;gap:6px"><select class="sel" id="bvaga" style="flex:1">${abertas.map(x => `<option value="${x.id}">${esc(x.titulo)}${x.da_pasta ? ' (mesma pasta)' : ''}</option>`).join('')}</select>
          <button class="btn mini" data-benc="${d.curriculo_id}">Encaminhar</button></div>
          <p style="margin-top:8px"><button class="btn mini claro" data-bsug="${d.curriculo_id}">✨ Pedir sugestão ao Atlas</button> <span class="mini" id="bsugmsg">${D.ia_modo === 'gemini' ? 'uma chamada à IA; a resposta fica guardada' : ''}</span></p>`
        : '<p class="mini">Não há outra vaga aberta agora. Crie uma em <b>Nova vaga</b>: o Atlas reaproveita os currículos da pasta.</p>'}</div>` : '';
    el.innerHTML = `<div class="cartao bcab" style="--c:${cor}"><div class="linha"><h3>${esc(c.nome)}</h3><button class="x" data-bfx aria-label="Fechar ficha">×</button></div>
        <div class="mini">${esc([c.cidade, c.email, c.telefone].filter(Boolean).join(' · '))}</div>
        <div class="mini">Pasta <b style="color:${cor}">${esc(pastaNome(tr.setor))}</b> · recebido ${new Date(d.recebido_em).toLocaleDateString('pt-BR')} (${esc(d.origem)})${tr.anos_experiencia ? ` · ${tr.anos_experiencia} ano(s) de experiência` : ''}</div>
        ${c.dado_sensivel ? '<div class="mini alerta" style="margin-top:4px">Contém dado pessoal sensível declarado pelo candidato: use só para a finalidade da vaga.</div>' : ''}</div>
      <div class="cartao destaque">${v && v.compatibilidade ? `💡 <b>Vaga sugerida: ${esc(v.vaga)}</b><div class="mini">a melhor análise: ${esc((COMPAT[v.compatibilidade] || COMPAT.BAIXA)[1].toLowerCase())}${v.analista ? ', por ' + esc(v.analista) : ''}</div>`
        : v ? `⏳ <b>Na vaga ${esc(v.vaga)}</b><div class="mini">aguardando a análise</div>`
        : at ? opiniaoAtlas(at)
        : `<b>Sem vaga sugerida ainda</b><div class="mini">${d.status === 'triado' ? 'Nenhuma vaga aberta usa esta pasta. Peça uma sugestão ao Atlas ou encaminhe abaixo.' : esc(d.situacao)}</div>`}</div>
      ${v && at && !at.antiga ? `<div class="cartao">${opiniaoAtlas(at)}</div>` : ''}
      <div class="sec">O que a IA disse</div>
      <div class="cartao"><div class="disse">${av(tr.por_id, 24)}<div><div><b>${esc(tr.por)}</b> <span class="mini">fez a triagem</span>${tr.confianca ? ` <span class="chip claro">confiança ${esc(tr.confianca.toLowerCase())}</span>` : ''}</div>
        <p>“${esc(tr.resumo || 'sem resumo')}”</p></div></div></div>
      ${analises}${acoes}
      <p style="margin:4px 2px 10px">${d.tem_arquivo ? `<a href="/api/curriculos/${d.curriculo_id}/arquivo" target="_blank" rel="noopener">Abrir arquivo original</a>` : ''}</p>
      ${d.texto ? `<details class="cartao"><summary>Texto do currículo</summary><pre class="btexto">${esc(d.texto)}</pre></details>` : ''}
      <p class="mini">Cada abertura fica registrada na auditoria (LGPD).</p>`;
    desenharAvatares(el);
  }
  $('#bancow').addEventListener('click', async e => {
    const t = e.target;
    if (t.closest('[data-bx]') || t.id === 'bancow') return fecharBanco();
    if (t.closest('[data-bfx]')) { BANCO.sel = null; BANCO.ficha = null; $('#bancow').classList.remove('ficha'); document.querySelectorAll('.bitem.on').forEach(x => x.classList.remove('on')); return pintarFicha(); }
    const sit = t.closest('[data-bsit]'), g = t.closest('[data-bgrupo]'), pa = t.closest('[data-bpasta]'), it = t.closest('[data-bcv]');
    if (sit) { BANCO.situacao = sit.dataset.bsit; BANCO.pagina = 1; pintarBancoFiltros(); return carregarBanco(); }
    if (g) { BANCO.grupo = g.dataset.bgrupo; BANCO.pasta = ''; BANCO.pagina = 1; pintarBancoFiltros(); return carregarBanco(); }
    if (pa) { BANCO.pasta = pa.dataset.bpasta; BANCO.grupo = ''; BANCO.pagina = 1; pintarBancoFiltros(); return carregarBanco(); }
    if (t.closest('[data-bmais]')) { BANCO.pagina++; return carregarBanco(); }
    if (it) return abrirFicha(+it.dataset.bcv);
    const cand = t.closest('[data-bcand]');
    if (cand) { e.preventDefault(); return abrirCurriculo(+cand.dataset.bcand); }
    const enc = t.closest('[data-benc]'), sug = t.closest('[data-bsug]');
    if (enc) {
      enc.disabled = true;
      try { BANCO.ficha = await api('POST', `/api/banco/${enc.dataset.benc}/encaminhar`, { vaga_id: +$('#bvaga').value }); pintarFicha(); carregarBanco(true); carregar(); }
      catch (er) { alert(er.message); enc.disabled = false; }
    }
    if (sug) {
      sug.disabled = true; $('#bsugmsg').textContent = '💭 Atlas está lendo o currículo…';
      try { BANCO.ficha = await api('POST', `/api/banco/${sug.dataset.bsug}/sugerir`); pintarFicha(); carregarBanco(true); }
      catch (er) { $('#bsugmsg').textContent = er.message; sug.disabled = false; }
    }
  });

  // ================================================================ modais
  function modal(html, aoEnviar) {
    const m = $('#modal'); m.innerHTML = `<form class="caixa">${html}<div class="erro" id="merro"></div><div class="acoes"><button type="button" class="btn claro" data-fechar>Cancelar</button><button class="btn escuro">Confirmar</button></div></form>`;
    m.style.display = 'flex';
    m.querySelector('[data-fechar]').onclick = () => (m.style.display = 'none');
    m.querySelector('form').onsubmit = async ev => {
      ev.preventDefault();
      const bt = m.querySelector('button:not([type])'); bt.disabled = true;
      try { await aoEnviar(m); m.style.display = 'none'; await carregar(); if (P) pintarPainel(true); }
      catch (er) { $('#merro').textContent = er.message; bt.disabled = false; }
    };
    return m.querySelector('form');
  }
  function modalVaga() {
    const f = modal(`<h3>Nova vaga</h3><label class="campo">Título<input name="titulo" required minlength="3"></label>
      <div class="g2"><label class="campo">Área<input name="area"></label><label class="campo">Local<input name="local" placeholder="Cidade - UF"></label></div>
      <label class="campo">Descrição<textarea name="descricao"></textarea></label>
      <label class="campo">Requisitos (um por linha)<textarea name="requisitos" placeholder="Ensino médio completo&#10;Excel intermediário"></textarea></label>
      <label class="campo">Pastas da triagem que alimentam a vaga<select name="setores" multiple size="7">${CATEGORIAS.filter(c => c !== 'OUTROS').map(c => `<option>${esc(c)}</option>`).join('')}</select></label>
      <p style="margin:-4px 0 10px"><button type="button" class="btn mini claro" id="sugerir">✨ Pedir sugestão ao Atlas</button> <span class="mini" id="sugmsg"></span></p>
      <p class="mini">Sem pastas marcadas, o Atlas escolhe pelo título e pela descrição. Ele também escolhe o agente pela especialidade (contratando, se faltar alguém) e reaproveita currículos do banco de talentos.</p>`, async m => {
      const fd = new FormData(m.querySelector('form'));
      await api('POST', '/api/vagas', { titulo: fd.get('titulo'), area: fd.get('area'), local: fd.get('local'), descricao: fd.get('descricao'),
        requisitos: String(fd.get('requisitos') || '').split('\n').map(s => s.trim()).filter(Boolean), setores: fd.getAll('setores') });
    });
    $('#sugerir').onclick = async () => {
      const msg = $('#sugmsg');
      if (f.titulo.value.trim().length < 3) { msg.textContent = 'Escreva o título primeiro.'; return; }
      msg.textContent = '💭 Atlas está pensando…';
      try {
        const r = await api('POST', '/api/vagas/sugerir', { titulo: f.titulo.value, descricao: f.descricao.value });
        for (const o of f.setores.options) o.selected = r.setores.includes(o.value);
        if (r.requisitos.length && !f.requisitos.value.trim()) f.requisitos.value = r.requisitos.join('\n');
        msg.textContent = r.setores.length ? `Atlas sugeriu: ${r.setores.join(', ')}${r.requisitos.length ? ' (e requisitos)' : ''}.` : 'O Atlas não encontrou pastas; escolha manualmente.';
      } catch (e) { msg.textContent = e.message; }
    };
  }
  function modalEnviar() {
    modal(`<h3>Enviar currículos</h3><label class="campo">Arquivos (PDF, DOCX, TXT ou imagem)<input type="file" name="arquivos" multiple required accept=".pdf,.docx,.txt,.jpg,.jpeg,.png,.webp"></label>
      <p class="mini">Os currículos que chegam por e-mail entram sozinhos pela API de triagem; use isto para envios manuais.</p>`, async m => {
      const f = new FormData(); for (const a of m.querySelector('input[type=file]').files) f.append('arquivos', a);
      const r = await api('POST', '/api/curriculos', undefined, f);
      const erros = r.filter(x => x.erro); if (erros.length) throw new Error(erros.map(x => `${x.arquivo}: ${x.erro}`).join(' | '));
    });
  }
  function modalAtivar() {
    const q = D.quadro || {};
    modal(`<h3>Contratar agente especializado</h3><label class="campo">Especialidade<select name="esp">${ESPECIALIDADES.map(e => `<option value="${e.id}">${esc(e.nome)}</option>`).join('')}</select></label>
      <label class="campo">Nome (opcional)<input name="nome" maxlength="40"></label>
      <p class="mini">O agente chega com personalidade própria, assume vagas da área que estão sem responsável e começa pelos currículos que aguardam. Estações ocupadas: ${q.agentes_ativos ?? '—'}/${q.estacoes ?? 12}.
      ${(D.automacao || {}).contratacao ? 'O Atlas também contrata sozinho quando falta alguém ou a fila cresce.' : ''}</p>`, async m => {
      const f = new FormData(m.querySelector('form'));
      await api('POST', '/api/agentes', { especialidade_id: +f.get('esp'), nome: f.get('nome') || null });
    });
  }

  // ---------------------------------------------------------------- conta: troca de senha
  function modalSenha(obrigatoria) {
    return new Promise(resolve => {
      const m = $('#modal');
      if (obrigatoria) m.dataset.travado = '1';
      m.innerHTML = `<form class="caixa"><h3>${obrigatoria ? 'Crie sua senha' : 'Trocar senha'}</h3>
        ${obrigatoria ? '<p class="mini" style="margin-bottom:12px">Você entrou com uma senha temporária. Escolha uma senha só sua para continuar.</p>' : ''}
        <label class="campo">Senha atual<input type="password" name="atual" required autocomplete="current-password"></label>
        <label class="campo">Nova senha<input type="password" name="nova" required minlength="10" autocomplete="new-password"></label>
        <label class="campo">Repita a nova senha<input type="password" name="conf" required minlength="10" autocomplete="new-password"></label>
        <p class="mini">Pelo menos 10 caracteres e não só números. Ao trocar, as outras sessões abertas com a senha antiga são encerradas.</p>
        <div class="erro" id="merro"></div>
        <div class="acoes"><button type="button" class="btn claro" data-fechar>${obrigatoria ? 'Sair' : 'Cancelar'}</button><button class="btn escuro">Salvar senha</button></div></form>`;
      m.style.display = 'flex';
      const f = m.querySelector('form'), fechar = () => { m.style.display = 'none'; delete m.dataset.travado; };
      m.querySelector('[data-fechar]').onclick = obrigatoria
        ? async () => { await api('POST', '/auth/sair').catch(() => {}); location.reload(); }
        : () => { fechar(); resolve(false); };
      f.atual.focus();
      f.onsubmit = async ev => {
        ev.preventDefault(); $('#merro').textContent = '';
        if (f.nova.value !== f.conf.value) { $('#merro').textContent = 'As duas senhas novas não conferem.'; return; }
        const bt = f.querySelector('button:not([type])'); bt.disabled = true;
        try {
          EU = await api('POST', '/auth/senha', { senha_atual: f.atual.value, nova_senha: f.nova.value });
          if (obrigatoria) { fechar(); resolve(true); return; }
          f.innerHTML = '<h3>Trocar senha</h3><div class="resultado bom">Senha alterada. As outras sessões abertas foram encerradas.</div><div class="acoes" style="margin-top:12px"><button type="button" class="btn escuro" data-fechar>Fechar</button></div>';
          f.querySelector('[data-fechar]').onclick = () => { fechar(); resolve(true); };
        } catch (er) { $('#merro').textContent = er.message; bt.disabled = false; }
      };
    });
  }

  // ---------------------------------------------------------------- administração (gestores; usuários e LGPD só admin)
  const usd = v => 'US$ ' + (v > 0 && v < 0.01 ? v.toFixed(4) : (v || 0).toFixed(2));
  const milhar = n => Number(n || 0).toLocaleString('pt-BR');
  function modalAdmin(aba = 'ia', testar = false) {
    const abas = [['ia', 'IA e custos'], ...(EU.papel === 'admin' ? [['usuarios', 'Usuários'], ['lgpd', 'Privacidade (LGPD)']] : [])];
    if (!abas.some(([k]) => k === aba)) aba = 'ia';
    const m = $('#modal');
    m.innerHTML = `<div class="caixa larga"><div class="linha"><h3 style="margin:0">Administração ${MARCA.versao ? `<span class="mini" style="font-weight:600">· versão ${esc(MARCA.versao)}${MARCA.versao_data ? ' de ' + esc(MARCA.versao_data) : ''}</span>` : ''}</h3><button class="x" data-fechar aria-label="Fechar">×</button></div>
      <div class="tabs">${abas.map(([k, n]) => `<button data-aba="${k}" class="${k === aba ? 'on' : ''}">${n}</button>`).join('')}</div><div id="adm"></div></div>`;
    m.style.display = 'flex';
    m.querySelector('[data-fechar]').onclick = () => (m.style.display = 'none');
    m.querySelector('.tabs').onclick = e => { const b = e.target.closest('[data-aba]'); if (b) modalAdmin(b.dataset.aba); };
    ({ ia: abaIA, usuarios: abaUsuarios, lgpd: abaLGPD })[aba]($('#adm'), testar);
  }

  async function abaIA(el, testar) {
    el.innerHTML = '<p class="mini">Carregando…</p>';
    let u;
    try { u = await api('GET', '/api/ia/uso'); } catch (e) { el.innerHTML = `<p class="erro">${esc(e.message)}</p>`; return; }
    if (u.modo !== 'gemini') {
      el.innerHTML = `<div class="resultado bom">Modo demonstração: as análises usam regras locais e nenhum dado sai do servidor.</div>
        <p class="mini" style="margin-top:10px">Para usar o Gemini, defina <b>IA_MODO=gemini</b> e <b>GEMINI_API_KEY</b> no arquivo .env e reinicie o servidor.</p>`;
      return;
    }
    const h = u.hoje, t = u.ultimos_30_dias, lim = u.limite_diario_usd, pct = lim > 0 ? Math.min(100, Math.round(h.custo_usd / lim * 100)) : 0;
    el.innerHTML = `<div id="ia-estado" class="resultado ${u.pausa ? 'ruim' : 'bom'}">${u.pausa ? `⏸ IA pausada: ${esc(u.pausa)}.` : '✓ IA disponível.'}</div>
      <div class="g4 uso" style="margin-top:12px"><div class="kpi"><b>${usd(h.custo_usd)}</b>Gasto hoje</div><div class="kpi"><b>${lim > 0 ? usd(lim) : '—'}</b>Limite diário</div>
        <div class="kpi"><b>${milhar(h.chamadas)}</b>Chamadas hoje</div><div class="kpi"><b>${usd(t.custo_usd)}</b>Últimos 30 dias</div></div>
      ${lim > 0 ? `<div class="barra" title="${pct}% do limite diário"><i style="width:${pct}%;background:${pct >= 90 ? 'var(--bad)' : pct >= 70 ? '#F08A24' : 'var(--ok)'}"></i></div>` : ''}
      <p class="mini">Modelo <b>${esc(u.modelo)}</b>${u.modelo_triagem !== u.modelo ? ` (triagem: <b>${esc(u.modelo_triagem)}</b>)` : ''} · raciocínio <b>${esc(u.pensamento || 'padrão')}</b>${u.preco_por_milhao_usd ? ` · US$ ${u.preco_por_milhao_usd.entrada.toFixed(2)} entrada / US$ ${u.preco_por_milhao_usd.saida.toFixed(2)} saída por 1 milhão de tokens` : ''}</p>
      <p class="mini">Tokens hoje: ${milhar(h.entrada)} de entrada (${milhar(h.cache)} do cache) · ${milhar(h.saida)} de saída · ${milhar(h.pensamento)} de raciocínio.</p>
      <div class="sec">Para onde vão os tokens hoje</div>
      ${u.por_finalidade_hoje.length ? `<table class="tab"><tr><th>Uso</th><th>Chamadas</th><th>Tokens</th><th>Custo</th></tr>${u.por_finalidade_hoje.map(f => `<tr><td>${esc(FINALIDADES[f.finalidade] || f.finalidade)}</td><td>${milhar(f.chamadas)}</td><td>${milhar(f.tokens)}</td><td>${usd(f.custo_usd)}</td></tr>`).join('')}</table>` : '<p class="mini">Nenhuma chamada à IA hoje.</p>'}
      ${EU.papel === 'admin' ? '<p style="margin-top:14px"><button class="btn" id="testar-ia">Testar conexão com a IA</button> <span class="mini">uma chamada mínima para conferir chave, créditos e modelo</span></p><div id="teste-ia"></div>' : ''}
      <p class="mini" style="margin-top:12px">Valores estimados a partir dos tokens informados pelo Google; a cobrança oficial fica no AI Studio. O teto diário é IA_LIMITE_DIARIO_USD: ao atingi-lo a fila pausa até o dia seguinte.</p>`;
    const bt = $('#testar-ia');
    if (!bt) return;
    bt.onclick = async () => {
      bt.disabled = true; $('#teste-ia').innerHTML = '<p class="mini" style="margin-top:8px">💭 Testando…</p>';
      try {
        const r = await api('POST', '/api/ia/testar'), tk = r.tokens || {};
        $('#teste-ia').innerHTML = `<div class="resultado ${r.ok ? 'bom' : 'ruim'}">${esc(r.mensagem)}${r.ok && r.ms ? ` <span class="mini">(${r.ms} ms · ${milhar(tk.entrada)} + ${milhar(tk.saida)} tokens · raciocínio ${esc(r.pensamento)})</span>` : ''}</div>`;
        if (r.ok) { const s = $('#ia-estado'); if (s) { s.className = 'resultado bom'; s.textContent = '✓ IA disponível.'; } carregar(); }
      } catch (e) { $('#teste-ia').innerHTML = `<div class="resultado ruim">${esc(e.message)}</div>`; }
      bt.disabled = false;
    };
    if (testar) bt.click();
  }

  async function abaUsuarios(el) {
    let lista;
    try { lista = await api('GET', '/api/usuarios'); } catch (e) { el.innerHTML = `<p class="erro">${esc(e.message)}</p>`; return; }
    const opcoes = p => Object.entries(PAPEIS).map(([k, n]) => `<option value="${k}" ${k === p ? 'selected' : ''}>${n}</option>`).join('');
    const quando = ts => ts ? new Date(ts).toLocaleString('pt-BR', { dateStyle: 'short', timeStyle: 'short' }) : 'nunca';
    el.innerHTML = `<div id="usr-msg"></div>
      <table class="tab"><tr><th>Pessoa</th><th>Papel</th><th>Último acesso</th><th></th></tr>
      ${lista.map(x => `<tr class="${x.ativo ? '' : 'inativo'}"><td><b>${esc(x.nome)}</b>${x.id === EU.id ? ' <span class="mini">(você)</span>' : ''}<br><span class="mini">${esc(x.email)}</span>
          ${!x.ativo ? '<br><span class="mini">desativado</span>' : x.trocar_senha ? '<br><span class="mini alerta">aguardando o primeiro acesso</span>' : ''}</td>
        <td><select class="sel" data-papel="${x.id}" ${x.id === EU.id ? 'disabled' : ''}>${opcoes(x.papel)}</select></td>
        <td class="mini">${quando(x.ultimo_acesso_em)}</td>
        <td style="text-align:right;white-space:nowrap">${x.id === EU.id ? '' : `<button class="btn mini claro" data-reset="${x.id}">Nova senha</button> <button class="btn mini ${x.ativo ? 'claro' : ''}" data-ativo="${x.id}" data-v="${x.ativo ? 0 : 1}">${x.ativo ? 'Desativar' : 'Reativar'}</button>`}</td></tr>`).join('')}</table>
      <div class="sec" style="margin-top:16px">Novo acesso</div>
      <form id="novo-usr"><div class="g3"><label class="campo">Nome<input name="nome" required minlength="2" maxlength="120"></label>
        <label class="campo">E-mail<input name="email" type="email" required maxlength="200"></label><label class="campo">Papel<select name="papel">${opcoes('recrutador')}</select></label></div>
        <button class="btn">Criar acesso</button> <span class="mini">o sistema gera uma senha temporária; a pessoa escolhe a dela no primeiro acesso</span></form>
      <p class="mini" style="margin-top:12px"><b>Recrutador</b>: acompanha agentes e candidatos e decide. <b>Supervisor</b>: também cria vagas, envia currículos, contrata ou dispensa agentes e vê os custos da IA. <b>Administrador</b>: também gerencia acessos, LGPD e o teste da IA.</p>`;
    const msg = (html, ok = true) => ($('#usr-msg').innerHTML = `<div class="resultado ${ok ? 'bom' : 'ruim'}" style="margin:0 0 12px">${html}</div>`);
    const senhaHtml = (quem, s) => `Senha temporária de <b>${esc(quem)}</b>: <span class="senha">${esc(s)}</span><br><span class="mini">Aparece só agora: copie e entregue por um canal seguro. Ela deixa de valer no primeiro acesso.</span>`;
    el.querySelector('#novo-usr').onsubmit = async ev => {
      ev.preventDefault();
      const f = ev.target, bt = f.querySelector('button'); bt.disabled = true;
      try { const r = await api('POST', '/api/usuarios', { nome: f.nome.value, email: f.email.value, papel: f.papel.value }); await abaUsuarios(el); msg(senhaHtml(r.nome, r.senha_temporaria)); }
      catch (e) { msg(esc(e.message), false); bt.disabled = false; }
    };
    el.onchange = async e => {
      const s = e.target.closest('[data-papel]'); if (!s) return;
      try { await api('PATCH', `/api/usuarios/${s.dataset.papel}`, { papel: s.value }); await abaUsuarios(el); msg('Papel atualizado. A pessoa entra de novo e já recebe o novo acesso.'); }
      catch (er) { await abaUsuarios(el); msg(esc(er.message), false); }
    };
    el.onclick = async e => {
      const r = e.target.closest('[data-reset]'), a = e.target.closest('[data-ativo]');
      if (r) {
        const quem = lista.find(x => x.id === +r.dataset.reset);
        if (!confirm(`Gerar uma nova senha temporária para ${quem.nome}? A senha atual deixa de valer e as sessões abertas são encerradas.`)) return;
        try { const s = await api('POST', `/api/usuarios/${quem.id}/senha`); await abaUsuarios(el); msg(senhaHtml(quem.nome, s.senha_temporaria)); }
        catch (er) { msg(esc(er.message), false); }
      } else if (a) {
        try { await api('PATCH', `/api/usuarios/${a.dataset.ativo}`, { ativo: a.dataset.v === '1' }); await abaUsuarios(el); }
        catch (er) { msg(esc(er.message), false); }
      }
    };
  }

  function abaLGPD(el) {
    el.innerHTML = `<p class="mini">Busque o candidato por nome ou e-mail para exportar seus dados (acesso e portabilidade) ou eliminá-los (anonimização irreversível, com exclusão do arquivo e das conversas dos agentes sobre ele).</p>
      <div style="display:flex;gap:8px;margin:12px 0"><input id="lbusca" class="sel" style="flex:1;padding:9px" placeholder="nome ou e-mail"><button class="btn" id="lbtn">Buscar</button></div>
      <div id="lres"></div>
      <p class="mini" style="margin-top:10px">Currículos sem atividade são anonimizados automaticamente após o prazo de retenção (RETENCAO_DIAS), e cada abertura de currículo fica registrada na auditoria.</p>`;
    const buscar = async () => {
      try {
        const r = await api('GET', '/api/lgpd/titulares?busca=' + encodeURIComponent($('#lbusca').value));
        $('#lres').innerHTML = r.map(t => `<div class="cartao"><div class="linha"><span><b>${esc(t.nome)}</b><br><span class="mini">${esc(t.email || '')} ${esc(t.cidade || '')}</span></span>
          <span><a class="btn mini claro" href="/api/lgpd/titulares/${t.id}/exportar">Exportar</a> <button class="btn mini perigo" data-el="${t.id}">Eliminar</button></span></div></div>`).join('') || '<p class="mini">Nenhum titular encontrado.</p>';
      } catch (e) { $('#lres').innerHTML = `<p class="erro">${esc(e.message)}</p>`; }
    };
    $('#lbtn').onclick = buscar; $('#lbusca').onkeydown = e => e.key === 'Enter' && buscar();
    $('#lbusca').focus();
    $('#lres').onclick = async e => {
      const b = e.target.closest('[data-el]'); if (!b) return;
      if (!confirm('Eliminar os dados deste titular? A ação não pode ser desfeita.')) return;
      try { await api('POST', `/api/lgpd/titulares/${b.dataset.el}/eliminar`, {}); EQUIPE.carregado = false; buscar(); carregar(); }
      catch (er) { alert(er.message); }
    };
  }
  // apoio a testes automatizados (somente leitura)
  window.__hubZoom = f => zoom(f);
  window.__hubPosicoes = () => posicoes.map(p => ({ id: p.a.id, nome: p.a.nome, x: p.hx, y: p.hy, estado: p.a.estado, andando: p.andando, dir: p.dir }));
  window.__hubStatus = () => Object.fromEntries(todos().filter(a => VIS[a.id]).map(a => [a.id, { nome: a.nome, texto: textoStatus(a), estado: estadoVisual(a), momento: !!emMomento(VIS[a.id]), no: VIS[a.id].no }]));
  window.__hubGato = () => ({ no: GATO.no, pose: GATO.pose, x: +GATO.x.toFixed(2), y: +GATO.y.toFixed(2), tela: GATO.tela });
  window.__hubBaloes = () => Object.entries(VIS).map(([id, v]) => ({ id: +id, baloes: v.baloes.map(b => ({ tipo: b.tipo, texto: b.texto, visivel: !!b.t0 })) }));
  addEventListener('keydown', e => {
    if (e.key !== 'Escape') return;
    if ($('#cvw').style.display === 'block') { $('#cvw').style.display = 'none'; return; }
    if (!$('#modal').dataset.travado) $('#modal').style.display = 'none';
    if (BANCO.aberto) fecharBanco();
  });
})();
