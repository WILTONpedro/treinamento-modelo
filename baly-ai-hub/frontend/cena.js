/* Baly AI Hub — cena isométrica do escritório.
   Desenha a planta, os móveis e os agentes; calcula rotas pelos corredores.
   Não inventa estados: só anima o que o app (dados da API) informa. */
'use strict';
(function () {
  const C = Math.cos(Math.PI / 6), S = 0.5;
  let T = 30, OX = 0, OY = 0, DPR = 1, W = 0, H = 0;
  const P = (x, y, z = 0) => [OX + (x - y) * C * T, OY + (x + y) * S * T - z * T];

  // ------------------------------------------------------------------ cores
  const COR = {
    base: '#E9EAEE', piso: '#F4F4F6', corredor: '#F8F8F9', parede: '#FFFFFF', paredeE: '#E6E7EC', paredeD: '#D6D8DF',
    recepcao: '#F6F3EE', vagas: '#EEF1F6', supervisor: '#F4F0EA', alimentacao: '#EEF4EF', analise: '#F1F2F6', descanso: '#DCE8FB',
    mesa: '#FBFBFC', cadeira: '#A9AFBA', estofado: '#CBD4DE', planta: '#7DB38B', vaso: '#F2F2F2', rotulo: '#A3A7B0',
  };
  const AMARELO = '#FFCD00', ESCURO = '#211E16';
  const ESTADOS = {
    trabalhando: ['#2E9E5B', 'Trabalhando'], aguardando: ['#3B82C4', 'Aguardando'], descansando: ['#8E6BBF', 'Descansando'],
    deslocamento: ['#F08A24', 'Em deslocamento'], pendencia: ['#D64545', 'Com pendência'], disponivel: ['#1FB5A8', 'Disponível'],
    concluido: ['#7A8290', 'Concluído'], conversando: ['#E0A800', 'Conversando'],
  };

  function sombra(cor, f) {
    let r, g, b;
    if (cor[0] === '#') { const n = parseInt(cor.slice(1), 16); r = (n >> 16) & 255; g = (n >> 8) & 255; b = n & 255; }
    else[r, g, b] = cor.match(/\d+/g).map(Number);
    const h = v => Math.max(0, Math.min(255, Math.round(v * f))).toString(16).padStart(2, '0');
    return '#' + h(r) + h(g) + h(b);
  }

  // ------------------------------------------------------------------ primitivas
  function poli(c, pts, cor, linha) {
    c.beginPath(); c.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) c.lineTo(pts[i][0], pts[i][1]);
    c.closePath(); c.fillStyle = cor; c.fill();
    if (linha) { c.strokeStyle = linha; c.lineWidth = 1; c.stroke(); }
  }
  function sombraChao(c, x0, y0, x1, y1, a = 0.07) {
    for (const [d, al] of [[0.18, a * 0.6], [0.09, a]]) {
      poli(c, [P(x0 + d, y0 + d), P(x1 + d, y0 + d), P(x1 + d, y1 + d), P(x0 + d, y1 + d)], `rgba(30,35,50,${al})`);
    }
  }
  function caixa(c, x0, y0, x1, y1, z0, z1, cor, o = {}) {
    if (o.sombra && z0 === 0) sombraChao(c, x0, y0, x1, y1);
    poli(c, [P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)], o.esq || sombra(cor, 0.93));
    poli(c, [P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1)], o.dir || sombra(cor, 0.84));
    poli(c, [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)], o.topo || cor, o.linha);
  }
  function cilindro(c, x, y, r, z0, z1, cor) {
    const [cx, ct] = P(x, y, z1), cb = P(x, y, z0)[1], rx = r * Math.SQRT2 * C * T, ry = r * Math.SQRT2 * S * T;
    c.fillStyle = 'rgba(30,35,50,.08)'; c.beginPath(); c.ellipse(P(x + 0.12, y + 0.12)[0], P(x + 0.12, y + 0.12)[1], rx, ry, 0, 0, 7); c.fill();
    c.fillStyle = sombra(cor, 0.86); c.beginPath(); c.moveTo(cx - rx, ct); c.lineTo(cx - rx, cb);
    c.ellipse(cx, cb, rx, ry, 0, Math.PI, 0, true); c.lineTo(cx + rx, ct); c.closePath(); c.fill();
    c.fillStyle = cor; c.beginPath(); c.ellipse(cx, ct, rx, ry, 0, 0, 7); c.fill();
  }
  function bola(c, x, y, z, r, cor) {
    const [sx, sy] = P(x, y, z), g = c.createRadialGradient(sx - r * T * 0.35, sy - r * T * 0.4, r * T * 0.1, sx, sy, r * T);
    g.addColorStop(0, sombra(cor, 1.25)); g.addColorStop(1, sombra(cor, 0.82));
    c.fillStyle = g; c.beginPath(); c.arc(sx, sy, r * T, 0, 7); c.fill();
  }
  function planta(c, x, y, esc = 1) {
    cilindro(c, x, y, 0.22 * esc, 0, 0.36 * esc, COR.vaso);
    for (const [dx, dy, dz, r] of [[-0.12, 0.05, 0.62, 0.2], [0.12, -0.05, 0.66, 0.2], [0, 0.1, 0.78, 0.2], [0.02, -0.08, 0.86, 0.17]])
      bola(c, x + dx * esc, y + dy * esc, dz * esc, r * esc, COR.planta);
  }
  function cadeira(c, x, y, face, cor = COR.cadeira) {
    cilindro(c, x, y, 0.08, 0, 0.36, '#8D939E');
    caixa(c, x - 0.26, y - 0.26, x + 0.26, y + 0.26, 0.36, 0.46, cor);
    const t = 0.07, a = 0.26;
    const enc = { n: [x - a, y + a - t, x + a, y + a], s: [x - a, y - a, x + a, y - a + t], e: [x - a, y - a, x - a + t, y + a], w: [x + a - t, y - a, x + a, y + a] }[face];
    caixa(c, enc[0], enc[1], enc[2], enc[3], 0.46, 0.95, sombra(cor, 0.94));
  }
  function textoChao(c, txt, x, y, tam = 0.38, cor = COR.rotulo, eixo = 'x') {
    const [sx, sy] = P(x, y), k = T / 100;
    c.save();
    if (eixo === 'x') c.setTransform(DPR * C * k, DPR * S * k, -DPR * C * k, DPR * S * k, DPR * sx, DPR * sy);
    else c.setTransform(-DPR * C * k, DPR * S * k, -DPR * C * k, -DPR * S * k, DPR * sx, DPR * sy);
    c.font = `700 ${tam * 100}px system-ui,-apple-system,"Segoe UI",sans-serif`; c.fillStyle = cor;
    c.textBaseline = 'middle'; c.fillText(txt, 0, 0); c.restore();
  }
  function textoParede(c, txt, x, y, z, tam, cor) {
    const [sx, sy] = P(x, y, z), k = T / 100;
    c.save(); c.setTransform(DPR * C * k, DPR * S * k, 0, DPR * k, DPR * sx, DPR * sy);
    c.font = `800 ${tam * 100}px system-ui,-apple-system,"Segoe UI",sans-serif`; c.fillStyle = cor;
    c.textBaseline = 'top'; c.fillText(txt, 0, 0); c.restore();
  }

  // ------------------------------------------------------------------ planta baixa (unidades = 1 m aprox.)
  const SALAS = [
    { id: 'recepcao', nome: 'RECEPÇÃO', r: [0.3, 0.3, 7.5, 8], rot: [3.4, 6.55] },
    { id: 'vagas', nome: 'VAGAS EM ABERTO', r: [7.8, 0.3, 14.5, 8], rot: [8.3, 6.55] },
    { id: 'supervisor', nome: 'SUPERVISOR', r: [14.8, 0.3, 21, 8], rot: [16.0, 6.55] },
    { id: 'alimentacao', nome: 'ALIMENTAÇÃO', r: [21.3, 0.3, 29.85, 8], rot: [22.9, 6.55] },
    { id: 'corredor', nome: '', r: [0.3, 8.3, 29.85, 10.5] },
    { id: 'analise', nome: 'ANÁLISE DE CURRÍCULOS', r: [0.3, 10.8, 17, 19.85], rot: [0.9, 19.4] },
    { id: 'descanso', nome: 'DESCANSO', r: [17.3, 10.8, 29.85, 19.85], rot: [17.8, 19.4] },
  ];
  // paredes: [x0,y0,x1,y1,altura]
  const PAREDES = [];
  function parede(x0, y0, x1, y1, h, vaos = []) {
    const horiz = x1 - x0 > y1 - y0, a = horiz ? x0 : y0, b = horiz ? x1 : y1;
    const cortes = [[a, b]];
    for (const [v0, v1] of vaos) {
      for (let i = cortes.length - 1; i >= 0; i--) {
        const [c0, c1] = cortes[i];
        if (v1 <= c0 || v0 >= c1) continue;
        cortes.splice(i, 1, ...[[c0, v0], [v1, c1]].filter(([p, q]) => q - p > 0.01));
      }
    }
    for (const [p, q] of cortes) {
      for (let s = p; s < q - 0.001; s += 1) {
        const e = Math.min(q, s + 1);
        PAREDES.push(horiz ? [s, y0, e, y1, h] : [x0, s, x1, e, h]);
      }
    }
  }
  parede(0, 0, 30, 0.3, 1.6);
  parede(0, 0.3, 0.3, 20, 1.6, [[4.9, 7.1]]);
  parede(0.3, 8, 30, 8.3, 1.0, [[3.5, 5.5], [10, 12], [16.5, 18.5], [24.5, 26.5]]);
  parede(0.3, 10.5, 30, 10.8, 1.0, [[3.5, 5.5], [10, 12], [21, 23]]);
  parede(7.5, 0.3, 7.8, 8, 1.0); parede(14.5, 0.3, 14.8, 8, 1.0); parede(21, 0.3, 21.3, 8, 1.0);
  parede(17, 10.8, 17.3, 20, 1.0);
  const JANELAS_X = [[1.2, 3.2], [8.6, 13.8], [15.6, 18.6], [22.8, 23.2]], JANELAS_Y = [[1.5, 3.8], [11.5, 13.5], [16.5, 19]];

  // estações de trabalho na sala de análise (8)
  const ESTACOES = [];
  for (const [fy, sy] of [[12.7, 14.25], [16.5, 18.05]]) for (const cx of [3.1, 6.3, 10.5, 13.7]) ESTACOES.push({ x: cx, mesaY: fy, y: sy });

  // ------------------------------------------------------------------ grafo de circulação
  const NOS = {}, ARESTAS = [];
  const no = (n, x, y) => (NOS[n] = [x, y]);
  const liga = (...ns) => { for (let i = 1; i < ns.length; i++) ARESTAS.push([ns[i - 1], ns[i]]); };
  // corredor
  [['h1', 4.5], ['h2', 11], ['h3', 17.5], ['h4', 22], ['h5', 25.5]].forEach(([n, x]) => no(n, x, 9.4));
  liga('h1', 'h2', 'h3', 'h4', 'h5');
  // recepção
  no('rD', 4.5, 8.15); no('rIn', 4.5, 7.0); no('ent0', 0.75, 6.0); no('entrada', -1.4, 6.0); no('rIn2', 0.75, 2.6);
  no('recepcao', 1.9, 2.6); no('balcao0', 4.1, 2.9); no('balcao1', 4.1, 2.15); no('balcao2', 4.1, 3.65);
  liga('h1', 'rD', 'rIn', 'balcao0'); liga('rIn', 'balcao1'); liga('rIn', 'balcao2');
  liga('rIn', 'ent0', 'entrada'); liga('ent0', 'rIn2', 'recepcao');
  // vagas
  no('vD', 11, 8.15); no('vIn', 11, 7.0); no('v1', 13.95, 6.3); no('v2', 13.95, 2.2);
  no('quadro0', 11.6, 1.6); no('quadro1', 10.4, 1.6); no('quadro2', 12.8, 1.6);
  liga('h2', 'vD', 'vIn', 'v1', 'v2'); liga('v2', 'quadro0'); liga('v2', 'quadro1'); liga('v2', 'quadro2');
  // supervisor
  no('sD', 17.5, 8.15); no('sIn', 17.5, 7.0); no('atlas', 17.8, 4.15); liga('h3', 'sD', 'sIn', 'atlas');
  no('sV', 18.9, 5.3); liga('sIn', 'sV'); // onde quem chega (ou vem pedir ajuda) conversa com o Atlas
  // alimentação
  no('fD', 25.5, 8.15); no('fIn', 25.5, 7.0); no('f1', 22.2, 6.6); no('f2', 25.8, 5.7); no('f3', 29.3, 6.4); no('f4', 29.3, 4.4);
  no('geladeira', 22.2, 1.95); no('comer0', 24.0, 5.55); no('comer1', 27.6, 5.55); no('comer2', 22.85, 4.4); no('comer3', 28.75, 4.4); no('comer4', 25.15, 4.4);
  liga('h5', 'fD', 'fIn', 'comer0'); liga('fIn', 'comer1'); liga('fIn', 'f1', 'geladeira'); liga('f1', 'comer2');
  liga('fIn', 'f3', 'f4', 'comer3'); liga('fIn', 'f2', 'comer4');
  // análise
  no('aD1', 4.5, 10.65); no('aIn1', 4.5, 11.6); no('aD2', 11, 10.65); no('aIn2', 11, 11.6);
  no('am', 8.4, 11.6); no('al', 16.2, 11.6);
  liga('h1', 'aD1', 'aIn1', 'am', 'aIn2', 'aD2', 'h2'); liga('aIn2', 'al');
  for (const [lin, y] of [['A', 15.5], ['B', 19.1]]) {
    no('m' + lin, 8.4, y); no('l' + lin, 16.2, y);
  }
  liga('am', 'mA', 'mB'); liga('al', 'lA', 'lB');
  ESTACOES.forEach((e, i) => {
    const lin = i < 4 ? 'A' : 'B', ay = lin === 'A' ? 15.5 : 19.1;
    no('c' + i, e.x, ay); no('mesa' + i, e.x, e.y); liga('c' + i, 'mesa' + i);
    liga(e.x < 8.4 ? 'm' + lin : 'm' + lin, 'c' + i);
  });
  // corredores entre colunas de estações (mesma linha)
  liga('c0', 'c1', 'mA', 'c2', 'c3', 'lA'); liga('c4', 'c5', 'mB', 'c6', 'c7', 'lB');
  // descanso
  no('dD', 22, 10.65); no('dIn', 22, 11.6); no('d1', 18.0, 11.6); no('d2', 23.7, 11.6); no('d3', 23.7, 13.25);
  no('d4', 23.7, 15.2); no('d5', 23.7, 17.6); no('d6', 25.9, 17.6);
  no('ping0', 17.95, 15.2); no('ping1', 22.85, 15.2);
  no('sofa0', 25.5, 13.95); no('sofa1', 26.5, 13.95); no('sofa2', 27.5, 13.95);
  no('sf0', 25.5, 13.25); no('sf1', 26.5, 13.25); no('sf2', 27.5, 13.25);
  no('pufe0', 27.0, 18.6); no('pufe1', 28.6, 17.4);
  liga('h4', 'dD', 'dIn', 'd1', 'ping0'); liga('dIn', 'd2', 'd3', 'sf0', 'sf1', 'sf2'); liga('sf0', 'sofa0'); liga('sf1', 'sofa1'); liga('sf2', 'sofa2');
  liga('d3', 'd4', 'ping1'); liga('d4', 'd5', 'd6', 'pufe0'); liga('d6', 'pufe1');

  const LOCAIS_DESCANSO = [
    ['ping0', 'jogando pingue-pongue', 'e', 0], ['ping1', 'jogando pingue-pongue', 'w', 0],
    ['sofa0', 'assistindo TV', 'n', 1], ['sofa1', 'assistindo TV', 'n', 1], ['sofa2', 'assistindo TV', 'n', 1],
    ['comer0', 'fazendo um lanche', 'n', 1], ['comer1', 'tomando um café', 'n', 1], ['geladeira', 'pegando algo na geladeira', 'n', 0],
    ['pufe0', 'descansando no pufe', 's', 1], ['pufe1', 'descansando no pufe', 's', 1],
    ['comer2', 'fazendo um lanche', 'e', 1], ['comer3', 'tomando um café', 'w', 1], ['comer4', 'fazendo um lanche', 'w', 1],
  ];
  const INFO_LOCAL = Object.fromEntries(LOCAIS_DESCANSO.map(([n, txt, f, s]) => [n, { txt, f, sentado: !!s }]));
  ESTACOES.forEach((_, i) => (INFO_LOCAL['mesa' + i] = { f: 'n', sentado: true }));
  INFO_LOCAL.recepcao = { f: 'e', sentado: true }; INFO_LOCAL.atlas = { f: 'n', sentado: true }; INFO_LOCAL.sV = { f: 'w' };
  for (let i = 0; i < 3; i++) { INFO_LOCAL['balcao' + i] = { f: 'w' }; INFO_LOCAL['quadro' + i] = { f: 'n' }; }

  function rota(de, para) {
    if (de === para) return [];
    const dist = { [de]: 0 }, ant = {}, aberto = new Set(Object.keys(NOS));
    while (aberto.size) {
      let u = null;
      for (const n of aberto) if (dist[n] !== undefined && (u === null || dist[n] < dist[u])) u = n;
      if (u === null || u === para) break;
      aberto.delete(u);
      for (const [a, b] of ARESTAS) {
        const v = a === u ? b : b === u ? a : null;
        if (!v || !aberto.has(v)) continue;
        const d = dist[u] + Math.hypot(NOS[u][0] - NOS[v][0], NOS[u][1] - NOS[v][1]);
        if (dist[v] === undefined || d < dist[v]) { dist[v] = d; ant[v] = u; }
      }
    }
    if (dist[para] === undefined) return [para];
    const r = [para];
    while (r[0] !== de) r.unshift(ant[r[0]]);
    return r.slice(1);
  }

  // ------------------------------------------------------------------ desenho da sala (camada estática)
  const fundo = document.createElement('canvas');
  let alvos = [];          // áreas clicáveis (quadros, bandeja)
  let assinatura = '';

  function desenharFundo(d) {
    const c = fundo.getContext('2d');
    fundo.width = W * DPR; fundo.height = H * DPR;
    c.setTransform(DPR, 0, 0, DPR, 0, 0);
    c.clearRect(0, 0, W, H);
    alvos = [];
    // laje
    caixa(c, 0, 0, 30, 20, -0.35, 0, COR.piso, { esq: '#E2E4EA', dir: '#D3D6DE' });
    for (const s of SALAS) {
      const [x0, y0, x1, y1] = s.r;
      poli(c, [P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1)], s.id === 'corredor' ? COR.corredor : COR[s.id]);
    }
    // texturas sutis
    c.strokeStyle = 'rgba(120,130,150,.07)'; c.lineWidth = 1;
    for (let x = 22; x < 29.85; x += 1) { const a = P(x, 0.3), b = P(x, 8); c.beginPath(); c.moveTo(...a); c.lineTo(...b); c.stroke(); }
    for (let y = 1; y < 8; y += 1) { const a = P(21.3, y), b = P(29.85, y); c.beginPath(); c.moveTo(...a); c.lineTo(...b); c.stroke(); }
    for (let y = 1; y < 8; y += 0.5) { const a = P(14.8, y), b = P(21, y); c.beginPath(); c.moveTo(...a); c.lineTo(...b); c.stroke(); }
    // tapetes e marcações
    poli(c, [P(5.4, 0.6), P(7.3, 0.6), P(7.3, 3.8), P(5.4, 3.8)], '#ECE6DA');
    poli(c, [P(9.0, 3.0), P(13.5, 3.0), P(13.5, 5.8), P(9.0, 5.8)], '#E3E8F0');
    poli(c, [P(15.9, 2.2), P(19.8, 2.2), P(19.8, 5.0), P(15.9, 5.0)], '#ECE4D8');
    poli(c, [P(24.2, 11.6), P(28.9, 11.6), P(28.9, 15.0), P(24.2, 15.0)], '#C9DBF6');
    poli(c, [P(0.35, 4.9), P(1.3, 4.9), P(1.3, 7.1), P(0.35, 7.1)], '#E7E2D6');
    // rótulos no chão (como na planta de referência)
    for (const s of SALAS) if (s.nome) textoChao(c, s.nome, s.rot[0], s.rot[1], 0.36);
    textoChao(c, 'ENTRADA', 0.55, 6.8, 0.26, '#B4AFA2', 'y');

    const itens = [];
    const add = (k, f) => itens.push({ k, f });
    // paredes
    for (const [x0, y0, x1, y1, h] of PAREDES) add(x0 + x1 + y0 + y1 - 0.2, () => {
      caixa(c, x0, y0, x1, y1, 0, h, COR.parede, { esq: COR.paredeE, dir: COR.paredeD });
      if (h > 1.2 && y0 === 0) for (const [a, b] of JANELAS_X) { const i0 = Math.max(a, x0), i1 = Math.min(b, x1); if (i1 > i0) janela(c, 'x', i0, i1, y1); }
      if (h > 1.2 && x0 === 0) for (const [a, b] of JANELAS_Y) { const i0 = Math.max(a, y0), i1 = Math.min(b, y1); if (i1 > i0) janela(c, 'y', i0, i1, x1); }
    });
    // ---- recepção
    add(13.5, () => {
      caixa(c, 1.2, 4.0, 2.6, 4.6, 0, 0.95, '#FFFFFF', { sombra: 1 });
      caixa(c, 2.6, 1.2, 3.3, 4.6, 0, 0.95, '#FFFFFF', { sombra: 1 });
      caixa(c, 2.55, 1.15, 3.35, 4.65, 0.95, 1.0, '#F3F1EC');
      caixa(c, 2.66, 2.25, 2.74, 3.0, 1.0, 1.42, '#2B2F38');
      const fila = Math.min(14, (d.fila_triagem || 0) + (d.fila_analise || 0));
      caixa(c, 2.75, 3.55, 3.22, 4.3, 1.0, 1.03, '#D9DCE3');
      for (let i = 0; i < fila; i++) caixa(c, 2.8 + (i % 2) * 0.02, 3.6, 3.18, 4.22, 1.03 + i * 0.03, 1.05 + i * 0.03, i % 2 ? '#FFFFFF' : '#F4F4F4');
      const topo = 1.06 + fila * 0.03;
      alvos.push({ tipo: 'bandeja', pts: [P(2.7, 3.5, topo + 0.25), P(3.3, 3.5, topo + 0.25), P(3.3, 4.35, 1.0), P(2.7, 4.35, 1.0)] });
    });
    add(1.75 * 2 + 2.6 * 2 - 0.3, () => cadeira(c, 1.75, 2.6, 'e'));
    add(6.35 * 2 + 1.1 * 2, () => { caixa(c, 5.6, 0.7, 7.1, 1.5, 0, 0.42, COR.estofado, { sombra: 1 }); caixa(c, 5.6, 0.7, 7.1, 0.95, 0.42, 0.85, sombra(COR.estofado, 0.95)); });
    add(6.6 * 2 + 3.0 * 2, () => { caixa(c, 6.2, 2.6, 7.0, 3.4, 0, 0.42, COR.estofado, { sombra: 1 }); caixa(c, 6.78, 2.6, 7.0, 3.4, 0.42, 0.85, sombra(COR.estofado, 0.95)); });
    add(5.9 * 2 + 2.9 * 2, () => cilindro(c, 5.9, 2.25, 0.32, 0, 0.42, '#FFFFFF'));
    add(1.6, () => planta(c, 0.8, 0.8, 1.2));
    add(14.4, () => planta(c, 7.0, 7.4));
    add(8.3, () => planta(c, 0.8, 7.5));
    // ---- vagas: quadro grande na parede (só visual; detalhes no painel)
    add(2 * 13.8 + 2 * 0.42, () => {
      caixa(c, 8.6, 0.3, 13.8, 0.42, 0.45, 1.45, '#FFFFFF', { linha: '#DADDE4' });
      const f = (x0, x1, z0, z1, cor) => poli(c, [P(x0, 0.43, z0), P(x1, 0.43, z0), P(x1, 0.43, z1), P(x0, 0.43, z1)], cor);
      f(8.6, 13.8, 1.33, 1.45, AMARELO);
      const vs = (d.vagas || []).filter(v => v.status !== 'cancelada').slice(0, 12);
      vs.forEach((v, i) => {
        const col = i % 6, lin = Math.floor(i / 6), x0 = 8.8 + col * 0.82, z1 = 1.25 - lin * 0.42;
        const cor = v.status !== 'aberta' ? '#B9BDC6' : v.sem_responsavel ? '#E06A5F' : v.pendentes ? '#F2B33D' : '#5DB37E';
        f(x0, x0 + 0.68, z1 - 0.34, z1, '#F1F3F6'); f(x0, x0 + 0.68, z1 - 0.07, z1, cor);
        for (let k = 0; k < 3; k++) f(x0 + 0.06, x0 + 0.06 + (0.5 - k * 0.12), z1 - 0.14 - k * 0.06, z1 - 0.11 - k * 0.06, '#C9CDD6');
      });
      alvos.push({ tipo: 'vagas', pts: [P(8.6, 0.42, 1.45), P(13.8, 0.42, 1.45), P(13.8, 0.42, 0.45), P(8.6, 0.42, 0.45)] });
    });
    add(9.4 + 13.0 + 3.6 + 5.0, () => caixa(c, 9.4, 3.6, 13.0, 5.0, 0, 0.74, COR.mesa, { sombra: 1, linha: '#E3E5EA' }));
    for (const x of [10.2, 11.2, 12.2]) { add(x * 2 + 3.0 * 2, () => cadeira(c, x, 3.0, 's')); add(x * 2 + 5.6 * 2, () => cadeira(c, x, 5.6, 'n')); }
    add(8.3 * 2 + 7.4 * 2, () => planta(c, 8.3, 7.4)); add(14.1 * 2 + 7.4 * 2, () => planta(c, 14.1, 7.4));
    // ---- supervisor
    add(16 + 19.6 + 2.6 + 3.4, () => {
      caixa(c, 16.0, 2.6, 19.6, 3.4, 0, 0.74, COR.mesa, { sombra: 1, linha: '#E3E5EA' });
      const tela = d.atlas_trabalhando ? '#8FD3FF' : '#5E6B7D';
      for (const x of [16.8, 18.0]) {
        caixa(c, x + 0.35, 2.82, x + 0.45, 2.92, 0.74, 0.9, '#9AA0AA');
        caixa(c, x, 2.78, x + 0.8, 2.86, 0.88, 1.36, '#2B2F38');
        poli(c, [P(x + 0.05, 2.87, 0.92), P(x + 0.75, 2.87, 0.92), P(x + 0.75, 2.87, 1.32), P(x + 0.05, 2.87, 1.32)], tela);
      }
      caixa(c, 17.4, 3.05, 18.2, 3.25, 0.74, 0.77, '#E6E8EC');
    });
    add(17.8 * 2 + 4.15 * 2 - 0.4, () => cadeira(c, 17.8, 4.15, 'n', '#7D8492'));
    add(2 * 15.42 + 2 * 5.0, () => {
      caixa(c, 15.3, 1.7, 15.42, 1.82, 0, 0.3, '#9AA0AA'); caixa(c, 15.3, 4.7, 15.42, 4.82, 0, 0.3, '#9AA0AA');
      caixa(c, 15.3, 1.5, 15.42, 5.0, 0.3, 1.5, '#FFFFFF', { linha: '#DADDE4' });
      const f = (y0, y1, z0, z1, cor) => poli(c, [P(15.43, y0, z0), P(15.43, y1, z0), P(15.43, y1, z1), P(15.43, y0, z1)], cor);
      f(1.5, 5.0, 1.38, 1.5, AMARELO);
      const q = d.quadro || {}, notas = [];
      for (let i = 0; i < Math.min(5, q.vagas_abertas || 0); i++) notas.push('#BFE3C9');
      for (let i = 0; i < Math.min(4, q.vagas_em_processamento || 0); i++) notas.push('#FFE69A');
      for (let i = 0; i < Math.min(4, q.pendencias || 0); i++) notas.push('#F6B8B1');
      notas.forEach((cor, i) => { const y0 = 1.7 + (i % 5) * 0.62, z1 = 1.28 - Math.floor(i / 5) * 0.3; f(y0, y0 + 0.5, z1 - 0.24, z1, cor); });
      alvos.push({ tipo: 'quadro', pts: [P(15.42, 1.5, 1.5), P(15.42, 5.0, 1.5), P(15.42, 5.0, 0.3), P(15.42, 1.5, 0.3)] });
    });
    add(2 * 20.7 + 2 * 0.85, () => {
      caixa(c, 19.0, 0.35, 20.7, 0.85, 0, 1.4, '#FFFFFF', { sombra: 1 });
      const cores = ['#C9D3E3', '#E9D6B8', '#B8D2C2', '#DAC3D9', '#D7DAE0'];
      for (let p = 0; p < 3; p++) for (let i = 0; i < 9; i++) {
        const x = 19.1 + i * 0.17, z = 0.15 + p * 0.42;
        poli(c, [P(x, 0.86, z), P(x + 0.13, 0.86, z), P(x + 0.13, 0.86, z + 0.3 - (i % 3) * 0.04), P(x, 0.86, z + 0.3 - (i % 3) * 0.04)], cores[(i + p) % 5]);
      }
    });
    add(20.6 * 2 + 7.4 * 2, () => planta(c, 20.6, 7.4)); add(15.6 * 2 + 7.3 * 2, () => planta(c, 15.6, 7.3));
    // ---- alimentação
    add(2 * 22.7 + 2 * 1.35, () => {
      caixa(c, 21.7, 0.45, 22.7, 1.35, 0, 2.0, '#F4F6F8', { sombra: 1 });
      const f = (x0, x1, z0, z1, cor) => poli(c, [P(x0, 1.36, z0), P(x1, 1.36, z0), P(x1, 1.36, z1), P(x0, 1.36, z1)], cor);
      f(21.7, 22.7, 1.28, 1.3, '#D2D6DD'); f(22.55, 22.6, 1.4, 1.8, '#A7AEB9'); f(22.55, 22.6, 0.6, 1.1, '#A7AEB9');
    });
    add(2 * 29.4 + 2 * 1.1, () => {
      caixa(c, 23.4, 0.35, 29.4, 1.1, 0, 0.95, '#FFFFFF', { sombra: 1 });
      caixa(c, 24.4, 0.5, 25.4, 0.95, 0.93, 0.96, '#C9CED6');
      caixa(c, 27.2, 0.45, 27.75, 0.9, 0.95, 1.42, '#3A3F4A'); caixa(c, 28.2, 0.45, 29.1, 0.95, 0.95, 1.35, '#E9EBEF');
      for (let i = 0; i < 4; i++) caixa(c, 23.6 + i * 1.45, 0.36, 24.9 + i * 1.45, 0.4, 0.6, 0.62, '#E2E4EA');
    });
    for (const [tx, ty] of [[24.0, 4.4], [27.6, 4.4]]) {
      for (const [dx, dy, f] of [[0, -1.15, 's'], [-1.15, 0, 'e'], [1.15, 0, 'w'], [0, 1.15, 'n']])
        add((tx + dx) * 2 + (ty + dy) * 2, () => cadeira(c, tx + dx, ty + dy, f, '#C3C8D0'));
      add(tx * 2 + ty * 2 + 0.1, () => { cilindro(c, tx, ty, 0.12, 0, 0.7, '#B8BEC8'); cilindro(c, tx, ty, 0.75, 0.7, 0.76, '#FFFFFF'); bola(c, tx + 0.2, ty - 0.1, 0.8, 0.07, '#F2C14E'); });
    }
    add(29.5 * 2 + 7.4 * 2, () => planta(c, 29.5, 7.4, 1.1));
    // ---- análise de currículos: 8 estações
    for (const fy of [12.7, 16.5]) for (const [x0, x1] of [[1.6, 7.8], [9.0, 15.2]]) {
      const k = x0 + x1 + fy * 2 + 0.8;
      const est = ESTACOES.map((e, i) => [e, i]).filter(([e]) => e.mesaY === fy && e.x > x0 && e.x < x1);
      add(k, () => {
        caixa(c, x0, fy, x1, fy + 0.8, 0, 0.72, '#FFFFFF', { sombra: 1, linha: '#E3E5EA' });
        caixa(c, x0, fy - 0.06, x1, fy + 0.02, 0.72, 1.12, '#E4E7EC');
        for (const [e, i] of est) {
          const oc = (d.estacoes || {})[i];
          const tela = !oc ? '#4A505C' : oc === 'trabalhando' ? '#8FD3FF' : oc === 'pendencia' ? '#F2A39B' : oc === 'concluido' ? '#A8E0BC' : '#6F7C8F';
          caixa(c, e.x - 0.05, e.mesaY + 0.12, e.x + 0.05, e.mesaY + 0.22, 0.72, 0.88, '#9AA0AA');
          caixa(c, e.x - 0.45, e.mesaY + 0.08, e.x + 0.45, e.mesaY + 0.16, 0.86, 1.32, '#2B2F38');
          poli(c, [P(e.x - 0.4, e.mesaY + 0.17, 0.9), P(e.x + 0.4, e.mesaY + 0.17, 0.9), P(e.x + 0.4, e.mesaY + 0.17, 1.28), P(e.x - 0.4, e.mesaY + 0.17, 1.28)], tela);
          if (oc === 'trabalhando') for (let q = 0; q < 3; q++)
            poli(c, [P(e.x - 0.32, e.mesaY + 0.18, 1.2 - q * 0.08), P(e.x + 0.1 - q * 0.08, e.mesaY + 0.18, 1.2 - q * 0.08), P(e.x + 0.1 - q * 0.08, e.mesaY + 0.18, 1.22 - q * 0.08), P(e.x - 0.32, e.mesaY + 0.18, 1.22 - q * 0.08)], '#FFFFFF');
          caixa(c, e.x - 0.35, e.mesaY + 0.42, e.x + 0.35, e.mesaY + 0.6, 0.72, 0.75, '#E6E8EC');
          if (oc === 'trabalhando') caixa(c, e.x - 0.75, e.mesaY + 0.3, e.x - 0.45, e.mesaY + 0.66, 0.72, 0.76, '#FFFFFF');
        }
      });
      for (const [e] of est) add(k + 0.5 + e.x * 0.01, () => cadeira(c, e.x, e.y, 'n'));
    }
    add(2 * 0.9 + 2 * 16.4, () => { caixa(c, 0.35, 14.6, 0.9, 16.4, 0, 1.05, '#FFFFFF', { sombra: 1 }); caixa(c, 0.4, 15.0, 0.85, 15.8, 1.05, 1.25, '#D9DCE3'); });
    add(0.75 * 2 + 11.3 * 2, () => planta(c, 0.75, 11.3)); add(0.75 * 2 + 19.3 * 2, () => planta(c, 0.75, 19.3, 1.2)); add(16.6 * 2 + 19.4 * 2, () => planta(c, 16.6, 19.4));
    // ---- descanso e entretenimento
    add(2 * 28.2 + 2 * 11.35, () => {
      caixa(c, 24.8, 10.85, 28.0, 11.35, 0, 0.4, '#FFFFFF', { sombra: 1 });
      caixa(c, 24.6, 10.8, 28.2, 10.9, 0.45, 1.0, '#22252C');
      poli(c, [P(24.68, 10.91, 0.49), P(28.12, 10.91, 0.49), P(28.12, 10.91, 0.96), P(24.68, 10.91, 0.96)], d.tv_ligada ? '#6FA8E8' : '#343944');
      if (d.tv_ligada) poli(c, [P(25.2, 10.92, 0.55), P(26.6, 10.92, 0.55), P(26.6, 10.92, 0.85), P(25.2, 10.92, 0.85)], '#9FD0F5');
    });
    add(25.4 + 27.4 + 12.4 + 13.0, () => caixa(c, 25.4, 12.4, 27.4, 13.0, 0, 0.38, '#FFFFFF', { sombra: 1, linha: '#E3E5EA' }));
    add(24.6 + 28.4 + 13.6 + 14.5, () => {
      caixa(c, 24.6, 13.6, 28.4, 14.5, 0, 0.42, '#C2CEDA', { sombra: 1 });
      caixa(c, 24.6, 14.22, 28.4, 14.5, 0.42, 0.88, '#B6C3D0');
      caixa(c, 24.6, 13.6, 24.86, 14.22, 0.42, 0.62, '#B6C3D0'); caixa(c, 28.14, 13.6, 28.4, 14.22, 0.42, 0.62, '#B6C3D0');
    });
    add(18.6 + 22.2 + 14.2 + 16.2, () => {
      for (const [x, y] of [[18.9, 14.4], [21.8, 14.4], [18.9, 15.85], [21.8, 15.85]]) caixa(c, x, y, x + 0.1, y + 0.1, 0, 0.72, '#8D939E');
      caixa(c, 18.6, 14.2, 22.2, 16.2, 0.72, 0.78, '#4E8E77', { dir: '#3F735F', esq: '#46806B' });
      poli(c, [P(18.65, 15.18, 0.781), P(22.15, 15.18, 0.781), P(22.15, 15.22, 0.781), P(18.65, 15.22, 0.781)], '#E9F2EE');
      caixa(c, 20.38, 14.1, 20.42, 16.3, 0.78, 0.94, '#F4F4F4');
    });
    for (const [x, y] of [[27.0, 18.6], [28.6, 17.4]]) add(x * 2 + y * 2 - 0.2, () => { cilindro(c, x, y, 0.5, 0, 0.32, '#B9C6E6'); bola(c, x, y, 0.38, 0.32, '#B9C6E6'); });
    add(29.5 * 2 + 11.3 * 2, () => planta(c, 29.5, 11.3, 1.1)); add(29.5 * 2 + 19.4 * 2, () => planta(c, 29.5, 19.4, 1.2)); add(17.8 * 2 + 19.4 * 2, () => planta(c, 17.8, 19.4));
    add(1.1 * 2 + 10.0 * 2, () => planta(c, 1.0, 9.0, 0.9)); add(29.4 * 2 + 9.2 * 2, () => planta(c, 29.3, 9.0, 0.9));

    itens.sort((a, b) => a.k - b.k).forEach(i => i.f());
    textoParede(c, 'BALY', 4.0, 0.31, 1.45, 0.42, AMARELO);
  }

  function janela(c, eixo, a, b, face) {
    const q = eixo === 'x'
      ? [P(a + 0.08, face + 0.001, 0.55), P(b - 0.08, face + 0.001, 0.55), P(b - 0.08, face + 0.001, 1.4), P(a + 0.08, face + 0.001, 1.4)]
      : [P(face + 0.001, a + 0.08, 0.55), P(face + 0.001, b - 0.08, 0.55), P(face + 0.001, b - 0.08, 1.4), P(face + 0.001, a + 0.08, 1.4)];
    const g = c.createLinearGradient(q[0][0], q[3][1], q[1][0], q[0][1]);
    g.addColorStop(0, '#D7EAF8'); g.addColorStop(1, '#BFDDF2');
    poli(c, q, g, '#E6EEF5');
  }

  // ------------------------------------------------------------------ avatares (estilo memoji)
  function avatar(c, sx, sy, R, ap, o) {
    const pele = ap.pele || '#EEC49C', cab = ap.cabelo || '#3B2618', est = ap.estilo || 'curto', cam = ap.camisa || '#3B6EC8';
    c.save();
    // corpo/ombros
    const corpoAlt = o.sentado ? R * 0.95 : R * 1.45;
    c.fillStyle = cam;
    c.beginPath(); c.moveTo(sx - R * 0.95, sy + R * 0.7 + corpoAlt); c.lineTo(sx - R * 0.95, sy + R * 1.15);
    c.quadraticCurveTo(sx - R * 0.9, sy + R * 0.62, sx, sy + R * 0.6); c.quadraticCurveTo(sx + R * 0.9, sy + R * 0.62, sx + R * 0.95, sy + R * 1.15);
    c.lineTo(sx + R * 0.95, sy + R * 0.7 + corpoAlt); c.closePath(); c.fill();
    c.fillStyle = 'rgba(0,0,0,.08)'; c.fillRect(sx + R * 0.3, sy + R * 0.95, R * 0.65, corpoAlt - R * 0.3);
    c.fillStyle = sombra(pele.length === 7 ? pele : '#EEC49C', 0.9); c.beginPath(); c.ellipse(sx, sy + R * 0.66, R * 0.26, R * 0.14, 0, 0, 7); c.fill();
    // cabelo de trás
    if (est === 'longo') { c.fillStyle = cab; c.beginPath(); c.ellipse(sx, sy + R * 0.25, R * 1.05, R * 1.1, 0, 0, 7); c.fill(); }
    if (est === 'black') { c.fillStyle = cab; c.beginPath(); c.arc(sx, sy - R * 0.15, R * 1.22, 0, 7); c.fill(); }
    // orelhas
    c.fillStyle = sombra(pele, 0.93);
    c.beginPath(); c.arc(sx - R * 0.93, sy + R * 0.08, R * 0.2, 0, 7); c.arc(sx + R * 0.93, sy + R * 0.08, R * 0.2, 0, 7); c.fill();
    // cabeça
    const g = c.createRadialGradient(sx - R * 0.35, sy - R * 0.4, R * 0.15, sx, sy, R * 1.05);
    g.addColorStop(0, sombra(pele, 1.12)); g.addColorStop(1, sombra(pele, 0.9));
    c.fillStyle = g; c.beginPath(); c.ellipse(sx, sy, R * 0.92, R, 0, 0, 7); c.fill();
    // cabelo da frente
    c.fillStyle = cab;
    if (est === 'curto' || est === 'longo' || est === 'coque' || est === 'topete') {
      c.beginPath(); c.moveTo(sx - R * 0.95, sy - R * 0.05);
      c.quadraticCurveTo(sx - R * 1.0, sy - R * 1.12, sx, sy - R * 1.1);
      c.quadraticCurveTo(sx + R * 1.0, sy - R * 1.12, sx + R * 0.95, sy - R * 0.05);
      c.quadraticCurveTo(sx + R * 0.7, sy - R * 0.62, sx + R * 0.1, sy - R * 0.55);
      c.quadraticCurveTo(sx - R * 0.6, sy - R * 0.5, sx - R * 0.95, sy - R * 0.05); c.fill();
      if (est === 'topete') { c.beginPath(); c.ellipse(sx + R * 0.15, sy - R * 0.95, R * 0.62, R * 0.32, -0.25, 0, 7); c.fill(); }
      if (est === 'coque') { c.beginPath(); c.arc(sx, sy - R * 1.22, R * 0.36, 0, 7); c.fill(); }
      if (est === 'longo') { c.fillRect(sx - R * 1.0, sy - R * 0.2, R * 0.22, R * 1.1); c.fillRect(sx + R * 0.78, sy - R * 0.2, R * 0.22, R * 1.1); }
    } else if (est === 'cacheado' || est === 'black') {
      const n = est === 'black' ? 9 : 7;
      for (let i = 0; i < n; i++) { const a = Math.PI * (1.05 + i * 0.9 / (n - 1)); c.beginPath(); c.arc(sx + Math.cos(a) * R * 0.82, sy - R * 0.2 + Math.sin(a) * R * 0.85, R * 0.34, 0, 7); c.fill(); }
      c.beginPath(); c.ellipse(sx, sy - R * 0.72, R * 0.75, R * 0.38, 0, 0, 7); c.fill();
    } else if (est === 'careca') {
      c.fillStyle = 'rgba(255,255,255,.18)'; c.beginPath(); c.ellipse(sx - R * 0.3, sy - R * 0.6, R * 0.3, R * 0.15, -0.5, 0, 7); c.fill();
    }
    // barba
    if (ap.barba) { c.fillStyle = sombra(cab, 1.05); c.globalAlpha = 0.85; c.beginPath(); c.ellipse(sx, sy + R * 0.55, R * 0.62, R * 0.38, 0, 0, Math.PI); c.fill(); c.globalAlpha = 1; }
    // rosto
    const olhoY = sy + R * 0.02, dx = R * 0.34, dormindo = o.humor === 'calmo';
    c.fillStyle = '#FFFFFF';
    if (!dormindo) { c.beginPath(); c.ellipse(sx - dx, olhoY, R * 0.17, R * 0.2, 0, 0, 7); c.ellipse(sx + dx, olhoY, R * 0.17, R * 0.2, 0, 0, 7); c.fill(); }
    c.fillStyle = '#2A2320';
    const ol = o.olhar || 0;
    if (dormindo) {
      c.strokeStyle = '#2A2320'; c.lineWidth = Math.max(1, R * 0.08);
      c.beginPath(); c.arc(sx - dx, olhoY - R * 0.02, R * 0.12, 0.15 * Math.PI, 0.85 * Math.PI); c.stroke();
      c.beginPath(); c.arc(sx + dx, olhoY - R * 0.02, R * 0.12, 0.15 * Math.PI, 0.85 * Math.PI); c.stroke();
    } else {
      c.beginPath(); c.arc(sx - dx + ol, olhoY + R * 0.04, R * 0.1, 0, 7); c.arc(sx + dx + ol, olhoY + R * 0.04, R * 0.1, 0, 7); c.fill();
      c.fillStyle = '#FFFFFF'; c.beginPath(); c.arc(sx - dx + ol + R * 0.04, olhoY, R * 0.035, 0, 7); c.arc(sx + dx + ol + R * 0.04, olhoY, R * 0.035, 0, 7); c.fill();
    }
    c.strokeStyle = sombra(cab, 0.9); c.lineWidth = Math.max(1, R * 0.07); c.lineCap = 'round';
    const pr = o.humor === 'preocupado' ? 0.1 : 0;
    c.beginPath(); c.moveTo(sx - dx - R * 0.16, olhoY - R * 0.3 + pr * R); c.lineTo(sx - dx + R * 0.14, olhoY - R * 0.33 - pr * R);
    c.moveTo(sx + dx - R * 0.14, olhoY - R * 0.33 - pr * R); c.lineTo(sx + dx + R * 0.16, olhoY - R * 0.3 + pr * R); c.stroke();
    c.fillStyle = 'rgba(232,120,110,.22)'; c.beginPath(); c.arc(sx - R * 0.55, sy + R * 0.32, R * 0.15, 0, 7); c.arc(sx + R * 0.55, sy + R * 0.32, R * 0.15, 0, 7); c.fill();
    c.strokeStyle = '#8A4B3C'; c.lineWidth = Math.max(1, R * 0.08);
    c.beginPath();
    if (o.humor === 'preocupado') c.arc(sx, sy + R * 0.58, R * 0.16, 1.15 * Math.PI, 1.85 * Math.PI);
    else if (o.humor === 'focado') { c.moveTo(sx - R * 0.14, sy + R * 0.45); c.lineTo(sx + R * 0.14, sy + R * 0.45); }
    else c.arc(sx, sy + R * 0.32, R * 0.22, 0.2 * Math.PI, 0.8 * Math.PI);
    c.stroke();
    if (ap.oculos) {
      c.strokeStyle = '#2A2F3A'; c.lineWidth = Math.max(1, R * 0.08);
      c.beginPath(); c.roundRect(sx - dx - R * 0.26, olhoY - R * 0.2, R * 0.5, R * 0.4, R * 0.12); c.roundRect(sx + dx - R * 0.24, olhoY - R * 0.2, R * 0.5, R * 0.4, R * 0.12);
      c.moveTo(sx - R * 0.1, olhoY - R * 0.04); c.lineTo(sx + R * 0.1, olhoY - R * 0.04); c.stroke();
    }
    c.restore();
  }

  // ------------------------------------------------------------------ API pública
  const Cena = {
    ESTADOS, INFO_LOCAL, LOCAIS_DESCANSO, NOS, rota, avatar,
    medir(w, h, dpr, zoom = 1) {
      DPR = dpr;
      T = Math.max(17, Math.min((w - 24) / 43.4, (h - 24) / 27.2)) * zoom;
      W = Math.max(w, 43.4 * T + 24); H = Math.max(h, 27.2 * T + 24);
      const minX = -20 * C, maxX = 30 * C, minY = -1.7, maxY = 25.2;
      OX = W / 2 - ((minX + maxX) / 2) * T;
      OY = H / 2 - ((minY + maxY) / 2) * T;
      assinatura = '';
      return { largura: W, altura: H };
    },
    escala: () => T,
    projetar: P,
    atualizarFundo(d) {
      const ass = JSON.stringify([W, H, DPR, d.fila_triagem, d.fila_analise, d.estacoes, d.tv_ligada, d.atlas_trabalhando,
        (d.vagas || []).map(v => [v.status, v.sem_responsavel, v.pendentes > 0]), d.quadro && [d.quadro.vagas_abertas, d.quadro.vagas_em_processamento, d.quadro.pendencias]]);
      if (ass !== assinatura) { assinatura = ass; desenharFundo(d); }
    },
    fundo: () => fundo,
    alvos: () => alvos,
    dentro(pt, pts) {
      let ok = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const [xi, yi] = pts[i], [xj, yj] = pts[j];
        if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) ok = !ok;
      }
      return ok;
    },
  };
  window.Cena = Cena;
})();
