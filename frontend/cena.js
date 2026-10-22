/* AI Hub — cena isométrica do escritório.
   Desenha a planta, os móveis e as pessoas (corpo inteiro, com volume); calcula rotas pelos corredores
   e redesenha por cima o que fica na frente de cada pessoa (balcões, mesas, encostos, paredes).
   Não inventa estados: só anima o que o app (dados da API) informa. */
'use strict';
(function () {
  const C = Math.cos(Math.PI / 6), S = 0.5;
  const LARG = 36, PROF = 24; // planta: 36 × 24 (1 unidade ≈ 1 m)
  let T = 30, OX = 0, OY = 0, DPR = 1, W = 0, H = 0;
  const P = (x, y, z = 0) => [OX + (x - y) * C * T, OY + (x + y) * S * T - z * T];

  // ------------------------------------------------------------------ cores
  const COR = {
    base: '#E9EAEE', piso: '#F4F4F6', corredor: '#F8F8F9', parede: '#FFFFFF', paredeE: '#E6E7EC', paredeD: '#D6D8DF',
    recepcao: '#F5F3F0', vagas: '#EEF1F6', supervisor: '#F3F1EE', alimentacao: '#EEF4EF', analise: '#F1F2F6', descanso: '#E1E8FB',
    mesa: '#FBFBFC', cadeira: '#A9AFBA', estofado: '#CBD4DE', planta: '#7DB38B', vaso: '#F2F2F2', rotulo: '#A3A7B0',
  };
  const ACENTO = '#5B5BD6';
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
    if (o.sombra && z0 === 0 && !REDESENHO) sombraChao(c, x0, y0, x1, y1);
    poli(c, [P(x0, y1, z0), P(x1, y1, z0), P(x1, y1, z1), P(x0, y1, z1)], o.esq || sombra(cor, 0.93));
    poli(c, [P(x1, y0, z0), P(x1, y1, z0), P(x1, y1, z1), P(x1, y0, z1)], o.dir || sombra(cor, 0.84));
    poli(c, [P(x0, y0, z1), P(x1, y0, z1), P(x1, y1, z1), P(x0, y1, z1)], o.topo || cor, o.linha);
  }
  function cilindro(c, x, y, r, z0, z1, cor) {
    const [cx, ct] = P(x, y, z1), cb = P(x, y, z0)[1], rx = r * Math.SQRT2 * C * T, ry = r * Math.SQRT2 * S * T;
    if (!REDESENHO) { c.fillStyle = 'rgba(30,35,50,.08)'; c.beginPath(); c.ellipse(P(x + 0.12, y + 0.12)[0], P(x + 0.12, y + 0.12)[1], rx, ry, 0, 0, 7); c.fill(); }
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
  // encosto da cadeira conforme o lado para onde a pessoa sentada olha
  const ENCOSTO = (x, y, face, a = 0.26, t = 0.07) => ({
    n: [x - a, y + a - t, x + a, y + a], s: [x - a, y - a, x + a, y - a + t], e: [x - a, y - a, x - a + t, y + a], w: [x + a - t, y - a, x + a, y + a],
  }[face]);
  function cadeira(c, x, y, face, cor = COR.cadeira) {
    cilindro(c, x, y, 0.08, 0, 0.36, '#8D939E');
    caixa(c, x - 0.26, y - 0.26, x + 0.26, y + 0.26, 0.36, 0.46, cor);
    const e = ENCOSTO(x, y, face);
    caixa(c, e[0], e[1], e[2], e[3], 0.46, 0.95, sombra(cor, 0.94));
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
  function janela(c, eixo, a, b, face) {
    const q = eixo === 'x'
      ? [P(a + 0.08, face + 0.001, 0.55), P(b - 0.08, face + 0.001, 0.55), P(b - 0.08, face + 0.001, 1.4), P(a + 0.08, face + 0.001, 1.4)]
      : [P(face + 0.001, a + 0.08, 0.55), P(face + 0.001, b - 0.08, 0.55), P(face + 0.001, b - 0.08, 1.4), P(face + 0.001, a + 0.08, 1.4)];
    const g = c.createLinearGradient(q[0][0], q[3][1], q[1][0], q[0][1]);
    g.addColorStop(0, '#D7EAF8'); g.addColorStop(1, '#BFDDF2');
    poli(c, q, g, '#E6EEF5');
  }

  // ------------------------------------------------------------------ planta baixa
  const SALAS = [
    { id: 'recepcao', nome: 'RECEPÇÃO', r: [0.3, 0.3, 8.0, 8.5], rot: [1.7, 8.05] },
    { id: 'vagas', nome: 'VAGAS EM ABERTO', r: [8.3, 0.3, 15.6, 8.5], rot: [8.8, 8.05] },
    { id: 'supervisor', nome: 'SUPERVISOR', r: [15.9, 0.3, 23.0, 8.5], rot: [16.4, 8.05] },
    { id: 'alimentacao', nome: 'ALIMENTAÇÃO', r: [23.3, 0.3, 35.85, 8.5], rot: [23.8, 8.05] },
    { id: 'corredor', nome: '', r: [0.3, 8.8, 35.85, 11.0] },
    { id: 'analise', nome: 'ANÁLISE DE CURRÍCULOS', r: [0.3, 11.3, 21.0, 23.85], rot: [0.9, 23.45] },
    { id: 'descanso', nome: 'DESCANSO', r: [21.3, 11.3, 35.85, 23.85], rot: [21.8, 23.45] },
  ];
  const PAREDES = []; // [x0,y0,x1,y1,altura]
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
  parede(0, 0, LARG, 0.3, 1.6);
  parede(0, 0.3, 0.3, PROF, 1.6, [[5.2, 7.4]]);
  parede(0.3, 8.5, LARG, 8.8, 1.0, [[3.8, 5.8], [11.5, 13.5], [18.4, 20.4], [27.5, 29.5]]);
  parede(0.3, 11.0, LARG, 11.3, 1.0, [[3.8, 5.8], [11.5, 13.5], [27.5, 29.5]]);
  parede(8.0, 0.3, 8.3, 8.5, 1.0); parede(15.6, 0.3, 15.9, 8.5, 1.0); parede(23.0, 0.3, 23.3, 8.5, 1.0);
  parede(21.0, 11.3, 21.3, PROF, 1.0);
  const JANELAS_X = [[1.2, 3.4], [16.8, 19.0], [24.4, 26.2], [32.0, 35.2]], JANELAS_Y = [[1.4, 4.4], [12.6, 15.2], [18.4, 21.6]];

  // estações de trabalho na sala de análise: 3 fileiras × 4
  const ESTACOES = [];
  for (const fy of [13.0, 16.7, 20.4]) for (const x of [3.2, 6.6, 11.4, 14.8]) ESTACOES.push({ x, mesaY: fy, y: fy + 1.55 });
  const BLOCOS = [[1.6, 8.2], [9.8, 16.4]];

  // ------------------------------------------------------------------ grafo de circulação
  const NOS = {}, ARESTAS = [];
  const no = (n, x, y) => (NOS[n] = [x, y]);
  const liga = (...ns) => { for (let i = 1; i < ns.length; i++) ARESTAS.push([ns[i - 1], ns[i]]); };
  // corredor
  [['h1', 4.8], ['h2', 12.5], ['h3', 19.4], ['h4', 28.5], ['h5', 34.6]].forEach(([n, x]) => no(n, x, 9.9));
  liga('h1', 'h2', 'h3', 'h4', 'h5');
  // recepção
  no('rD', 4.8, 8.65); no('rIn', 4.8, 7.4); no('ent0', 0.8, 6.3); no('entrada', -1.4, 6.3); no('rIn2', 0.8, 2.9);
  no('recepcao', 2.0, 2.9); no('balcao0', 4.3, 3.0); no('balcao1', 4.3, 2.2); no('balcao2', 4.3, 3.8);
  liga('h1', 'rD', 'rIn', 'balcao0'); liga('rIn', 'balcao1'); liga('rIn', 'balcao2');
  liga('rIn', 'ent0', 'entrada'); liga('ent0', 'rIn2', 'recepcao');
  // vagas
  no('vD', 12.5, 8.65); no('vIn', 12.5, 7.4); no('v1', 15.0, 6.9); no('v2', 15.0, 2.3);
  no('quadro0', 12.0, 1.6); no('quadro1', 10.6, 1.6); no('quadro2', 13.4, 1.6);
  liga('h2', 'vD', 'vIn', 'v1', 'v2'); liga('v2', 'quadro0'); liga('v2', 'quadro1'); liga('v2', 'quadro2');
  // supervisor (o Atlas senta de frente para a sala; quem o visita fica do outro lado da mesa)
  no('sD', 19.4, 8.65); no('sIn', 19.4, 7.4); no('sV', 19.3, 5.15); no('s1', 22.3, 6.4); no('s2', 22.3, 2.2); no('atlas', 19.3, 2.6);
  liga('h3', 'sD', 'sIn', 'sV'); liga('sIn', 's1', 's2', 'atlas');
  // alimentação
  no('fD', 28.5, 8.65); no('fIn', 28.5, 7.6);
  [['f1', 24.6], ['w0', 26.3], ['f2', 28.05], ['w1', 29.8], ['f3', 31.55], ['w2', 33.3], ['f4', 35.2]].forEach(([n, x]) => no(n, x, 6.9));
  liga('h4', 'fD', 'fIn', 'f2'); liga('f1', 'w0', 'f2', 'w1', 'f3', 'w2', 'f4');
  no('comer0', 26.3, 5.85); no('comer1', 29.8, 5.85); no('comer2', 33.3, 5.85);
  liga('w0', 'comer0'); liga('w1', 'comer1'); liga('w2', 'comer2');
  no('f1b', 24.6, 4.7); no('f1c', 24.6, 2.6); no('geladeira', 24.2, 1.95); no('comer3', 25.15, 4.7);
  liga('f1', 'f1b', 'comer3'); liga('f1b', 'f1c', 'geladeira');
  no('f2b', 28.05, 4.7); no('comer4', 28.65, 4.7); liga('f2', 'f2b', 'comer4');
  no('f4b', 35.2, 4.7); no('comer5', 34.45, 4.7); liga('f4', 'f4b', 'comer5');
  // análise
  no('aD1', 4.8, 11.15); no('aIn1', 4.8, 12.15); no('aD2', 12.5, 11.15); no('aIn2', 12.5, 12.15);
  no('am', 9.0, 12.15); no('al', 18.6, 12.15);
  liga('h1', 'aD1', 'aIn1', 'am', 'aIn2', 'aD2', 'h2'); liga('aIn2', 'al');
  const LINHAS = [['A', 15.75], ['B', 19.45], ['C', 23.2]];
  for (const [l, y] of LINHAS) { no('m' + l, 9.0, y); no('l' + l, 18.6, y); }
  liga('am', 'mA', 'mB', 'mC'); liga('al', 'lA', 'lB', 'lC');
  ESTACOES.forEach((e, i) => { no('c' + i, e.x, LINHAS[Math.floor(i / 4)][1]); no('mesa' + i, e.x, e.y); liga('c' + i, 'mesa' + i); });
  for (let r = 0; r < 3; r++) { const b = r * 4, l = LINHAS[r][0]; liga('c' + b, 'c' + (b + 1), 'm' + l, 'c' + (b + 2), 'c' + (b + 3), 'l' + l); }
  // descanso
  no('dD', 28.5, 11.15); no('dIn', 28.5, 12.0); no('d1', 22.3, 12.0); no('dL', 22.0, 14.8);
  no('ping0', 21.95, 16.4); no('ping1', 26.85, 16.4);
  no('d2', 27.7, 13.4); no('d3', 27.4, 16.4); no('d4', 27.4, 19.0); no('d5', 31.5, 19.0); no('d6', 34.3, 19.0); no('d7', 24.4, 21.8);
  for (let i = 0; i < 3; i++) { no('sf' + i, 29.3 + i, 13.4); no('sofa' + i, 29.3 + i, 14.05); liga('sf' + i, 'sofa' + i); }
  no('pufe0', 30.5, 21.3); no('pufe1', 32.6, 20.0); no('pufe2', 34.3, 21.6);
  no('leitura0', 23.0, 21.0); no('leitura1', 23.0, 22.6);
  liga('h4', 'dD', 'dIn', 'd1', 'dL', 'ping0'); liga('dIn', 'd2', 'sf0', 'sf1', 'sf2');
  liga('d2', 'd3', 'ping1'); liga('d3', 'd4', 'd5', 'pufe0'); liga('d5', 'pufe1'); liga('d5', 'd6', 'pufe2');
  liga('d4', 'd7', 'leitura0'); liga('d7', 'leitura1');

  // locais de descanso: [nó, o que faz, para onde olha, sentado, objeto na mão]
  const LOCAIS_DESCANSO = [
    ['ping0', 'jogando pingue-pongue', 'e', 0, 'raquete'], ['ping1', 'jogando pingue-pongue', 'w', 0, 'raquete'],
    ['sofa0', 'assistindo TV', 'n', 1], ['sofa1', 'assistindo TV', 'n', 1], ['sofa2', 'assistindo TV', 'n', 1],
    ['comer0', 'fazendo um lanche', 'n', 1], ['comer1', 'tomando um café', 'n', 1, 'xicara'], ['comer2', 'fazendo um lanche', 'n', 1],
    ['comer3', 'tomando um café', 'e', 1, 'xicara'], ['comer4', 'fazendo um lanche', 'e', 1], ['comer5', 'tomando um café', 'w', 1, 'xicara'],
    ['geladeira', 'pegando algo na geladeira', 'n', 0],
    ['pufe0', 'descansando no pufe', 's', 1], ['pufe1', 'descansando no pufe', 's', 1], ['pufe2', 'descansando no pufe', 's', 1],
    ['leitura0', 'lendo um livro', 'e', 1, 'livro'], ['leitura1', 'lendo um livro', 'e', 1, 'livro'],
  ];
  const INFO_LOCAL = Object.fromEntries(LOCAIS_DESCANSO.map(([n, txt, f, s, item]) => [n, { txt, f, sentado: !!s, item }]));
  ESTACOES.forEach((_, i) => { INFO_LOCAL['mesa' + i] = { f: 'n', sentado: true, digita: true }; INFO_LOCAL['c' + i] = { f: 'n' }; });
  INFO_LOCAL.recepcao = { f: 'e', sentado: true, digita: true }; INFO_LOCAL.atlas = { f: 's', sentado: true };
  INFO_LOCAL.sV = { f: 'n' };
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

  // ------------------------------------------------------------------ desenho da sala (camada estática) + oclusores
  const fundo = document.createElement('canvas');
  let alvos = [];          // áreas clicáveis (quadros, bandeja)
  let OCLUSORES = [];      // o que é redesenhado por cima de quem está atrás
  let assinatura = '';
  let REDESENHO = false;   // true ao redesenhar um oclusor sobre as pessoas (sem sombras no chão)

  function telaDe(x0, y0, x1, y1, z1) {
    const pts = [];
    for (const x of [x0, x1]) for (const y of [y0, y1]) for (const z of [0, z1]) pts.push(P(x, y, z));
    return { x0: Math.min(...pts.map(p => p[0])), x1: Math.max(...pts.map(p => p[0])), y0: Math.min(...pts.map(p => p[1])), y1: Math.max(...pts.map(p => p[1])) };
  }

  function desenharFundo(d) {
    const cf = fundo.getContext('2d');
    fundo.width = W * DPR; fundo.height = H * DPR;
    cf.setTransform(DPR, 0, 0, DPR, 0, 0);
    cf.clearRect(0, 0, W, H);
    alvos = []; OCLUSORES = [];
    caixa(cf, 0, 0, LARG, PROF, -0.35, 0, COR.piso, { esq: '#E2E4EA', dir: '#D3D6DE' });
    for (const s of SALAS) {
      const [x0, y0, x1, y1] = s.r;
      poli(cf, [P(x0, y0), P(x1, y0), P(x1, y1), P(x0, y1)], s.id === 'corredor' ? COR.corredor : COR[s.id]);
    }
    // texturas e tapetes
    cf.strokeStyle = 'rgba(120,130,150,.07)'; cf.lineWidth = 1;
    for (let x = 24; x < 35.85; x += 1) { const a = P(x, 0.3), b = P(x, 8.5); cf.beginPath(); cf.moveTo(...a); cf.lineTo(...b); cf.stroke(); }
    for (let y = 1; y < 8.5; y += 1) { const a = P(23.3, y), b = P(35.85, y); cf.beginPath(); cf.moveTo(...a); cf.lineTo(...b); cf.stroke(); }
    for (let y = 1; y < 8.5; y += 0.5) { const a = P(15.9, y), b = P(23.0, y); cf.beginPath(); cf.moveTo(...a); cf.lineTo(...b); cf.stroke(); }
    poli(cf, [P(5.6, 0.6), P(7.7, 0.6), P(7.7, 4.0), P(5.6, 4.0)], '#ECE6DA');
    poli(cf, [P(9.4, 3.4), P(14.6, 3.4), P(14.6, 5.9), P(9.4, 5.9)], '#E3E8F0');
    poli(cf, [P(17.0, 1.8), P(21.6, 1.8), P(21.6, 4.6), P(17.0, 4.6)], '#ECE4D8');
    poli(cf, [P(27.9, 12.1), P(32.8, 12.1), P(32.8, 15.4), P(27.9, 15.4)], '#C9DBF6');
    poli(cf, [P(0.35, 5.2), P(1.3, 5.2), P(1.3, 7.4), P(0.35, 7.4)], '#E7E2D6');
    poli(cf, [P(22.2, 20.2), P(24.6, 20.2), P(24.6, 23.4), P(22.2, 23.4)], '#E9E1F5');
    for (const s of SALAS) if (s.nome) textoChao(cf, s.nome, s.rot[0], s.rot[1], 0.36);
    textoChao(cf, 'ENTRADA', 0.55, 7.25, 0.26, '#B4AFA2', 'y');

    const itens = [];
    const add = (k, f) => itens.push({ k, f });
    // oclusor: caixa no chão [x0,y0,x1,y1,altura] desenhada por f (também redesenhada sobre quem estiver atrás)
    const ocl = (k, box, f) => { add(k, f); OCLUSORES.push({ k, box, f, tela: telaDe(...box) }); };
    const cad = (x, y, face, cor) => {
      add(x * 2 + y * 2, c => cadeira(c, x, y, face, cor));
      if (face === 'n' || face === 'w') { // encosto entre a pessoa e quem olha a cena
        const e = ENCOSTO(x, y, face);
        OCLUSORES.push({ k: x * 2 + y * 2 + 0.01, box: [e[0], e[1], e[2], e[3], 0.95],
          f: c => caixa(c, e[0], e[1], e[2], e[3], 0.46, 0.95, sombra(cor || COR.cadeira, 0.94)), tela: telaDe(e[0], e[1], e[2], e[3], 0.95) });
      }
    };

    // paredes (as internas escondem as pernas de quem passa atrás delas)
    for (const [x0, y0, x1, y1, h] of PAREDES) {
      const f = c => {
        caixa(c, x0, y0, x1, y1, 0, h, COR.parede, { esq: COR.paredeE, dir: COR.paredeD });
        if (h > 1.2 && y0 === 0) for (const [a, b] of JANELAS_X) { const i0 = Math.max(a, x0), i1 = Math.min(b, x1); if (i1 > i0) janela(c, 'x', i0, i1, y1); }
        if (h > 1.2 && x0 === 0) for (const [a, b] of JANELAS_Y) { const i0 = Math.max(a, y0), i1 = Math.min(b, y1); if (i1 > i0) janela(c, 'y', i0, i1, x1); }
      };
      if (h > 1.2) add(x0 + x1 + y0 + y1 - 0.2, f); else ocl(x0 + x1 + y0 + y1 - 0.2, [x0, y0, x1, y1, h], f);
    }

    // ---- recepção
    const fila = Math.min(14, (d.fila_triagem || 0) + (d.fila_analise || 0));
    ocl(2 * 2.8 + 2 * 4.9, [1.3, 4.6, 2.8, 5.2, 0.95], c => caixa(c, 1.3, 4.6, 2.8, 5.2, 0, 0.95, '#FFFFFF', { sombra: 1 }));
    ocl(2.8 + 3.55 + 1.35 + 5.25, [2.75, 1.35, 3.55, 5.25, 1.45], c => {
      caixa(c, 2.8, 1.4, 3.5, 5.2, 0, 0.95, '#FFFFFF', { sombra: 1 });
      caixa(c, 2.75, 1.35, 3.55, 5.25, 0.95, 1.0, '#F3F1EC');
      caixa(c, 2.86, 2.5, 2.94, 3.3, 1.0, 1.42, '#2B2F38');
      caixa(c, 2.95, 4.1, 3.42, 4.85, 1.0, 1.03, '#D9DCE3');
      for (let i = 0; i < fila; i++) caixa(c, 3.0 + (i % 2) * 0.02, 4.15, 3.38, 4.8, 1.03 + i * 0.03, 1.05 + i * 0.03, i % 2 ? '#FFFFFF' : '#F4F4F4');
    });
    const topo = 1.06 + fila * 0.03;
    alvos.push({ tipo: 'bandeja', pts: [P(2.9, 4.05, topo + 0.25), P(3.5, 4.05, topo + 0.25), P(3.5, 4.9, 1.0), P(2.9, 4.9, 1.0)] });
    cad(2.0, 2.9, 'e');
    add(6.75 * 2 + 1.1 * 2, c => { caixa(c, 5.9, 0.7, 7.6, 1.5, 0, 0.42, COR.estofado, { sombra: 1 }); caixa(c, 5.9, 0.7, 7.6, 0.95, 0.42, 0.85, sombra(COR.estofado, 0.95)); });
    add(7.1 * 2 + 3.4 * 2, c => { caixa(c, 6.6, 3.0, 7.6, 3.8, 0, 0.42, COR.estofado, { sombra: 1 }); caixa(c, 7.38, 3.0, 7.6, 3.8, 0.42, 0.85, sombra(COR.estofado, 0.95)); });
    add(6.3 * 2 + 2.5 * 2, c => cilindro(c, 6.3, 2.5, 0.32, 0, 0.42, '#FFFFFF'));
    add(1.6, c => planta(c, 0.8, 0.8, 1.2)); add(7.5 * 2 + 7.9 * 2, c => planta(c, 7.5, 7.9)); add(0.8 * 2 + 4.4 * 2, c => planta(c, 0.8, 4.3, 0.9));

    // ---- vagas: quadro grande na parede (o painel lateral mostra os detalhes)
    add(2 * 15.0 + 2 * 0.42, c => {
      caixa(c, 9.0, 0.3, 15.0, 0.42, 0.45, 1.45, '#FFFFFF', { linha: '#DADDE4' });
      const f = (x0, x1, z0, z1, cor) => poli(c, [P(x0, 0.43, z0), P(x1, 0.43, z0), P(x1, 0.43, z1), P(x0, 0.43, z1)], cor);
      f(9.0, 15.0, 1.33, 1.45, ACENTO);
      const vs = (d.vagas || []).filter(v => v.status !== 'cancelada').slice(0, 14);
      vs.forEach((v, i) => {
        const col = i % 7, lin = Math.floor(i / 7), x0 = 9.15 + col * 0.83, z1 = 1.25 - lin * 0.42;
        const cor = v.status !== 'aberta' ? '#B9BDC6' : v.sem_responsavel ? '#E06A5F' : v.pendentes ? '#F2B33D' : '#5DB37E';
        f(x0, x0 + 0.7, z1 - 0.34, z1, '#F1F3F6'); f(x0, x0 + 0.7, z1 - 0.07, z1, cor);
        for (let k = 0; k < 3; k++) f(x0 + 0.06, x0 + 0.06 + (0.52 - k * 0.12), z1 - 0.14 - k * 0.06, z1 - 0.11 - k * 0.06, '#C9CDD6');
      });
      alvos.push({ tipo: 'vagas', pts: [P(9.0, 0.42, 1.45), P(15.0, 0.42, 1.45), P(15.0, 0.42, 0.45), P(9.0, 0.42, 0.45)] });
    });
    ocl(9.8 + 14.2 + 3.9 + 5.4, [9.8, 3.9, 14.2, 5.4, 0.74], c => caixa(c, 9.8, 3.9, 14.2, 5.4, 0, 0.74, COR.mesa, { sombra: 1, linha: '#E3E5EA' }));
    for (const x of [10.6, 12.0, 13.4]) { cad(x, 3.3, 's'); cad(x, 6.0, 'n'); }
    add(8.8 * 2 + 7.9 * 2, c => planta(c, 8.8, 7.9)); add(15.1 * 2 + 7.9 * 2, c => planta(c, 15.1, 7.9));

    // ---- supervisor
    ocl(17.4 + 21.2 + 3.4 + 4.2, [17.4, 3.4, 21.2, 4.2, 1.36], c => {
      caixa(c, 17.4, 3.4, 21.2, 4.2, 0, 0.74, COR.mesa, { sombra: 1, linha: '#E3E5EA' });
      for (const x of [18.0, 19.6]) { // monitores virados para o Atlas: daqui vemos a parte de trás
        caixa(c, x + 0.35, 3.62, x + 0.45, 3.72, 0.74, 0.9, '#9AA0AA');
        caixa(c, x, 3.48, x + 0.8, 3.58, 0.88, 1.36, '#3A3F4A');
        if (d.atlas_trabalhando) poli(c, [P(x, 3.48, 1.36), P(x + 0.8, 3.48, 1.36), P(x + 0.8, 3.48, 1.39), P(x, 3.48, 1.39)], '#8FD3FF');
      }
      caixa(c, 18.9, 3.42, 19.6, 3.6, 0.74, 0.77, '#E6E8EC');
    });
    cad(19.3, 2.6, 's', '#7D8492');
    add(2 * 16.42 + 2 * 5.5, c => {
      caixa(c, 16.3, 1.7, 16.42, 1.82, 0, 0.3, '#9AA0AA'); caixa(c, 16.3, 5.2, 16.42, 5.32, 0, 0.3, '#9AA0AA');
      caixa(c, 16.3, 1.5, 16.42, 5.5, 0.3, 1.5, '#FFFFFF', { linha: '#DADDE4' });
      const f = (y0, y1, z0, z1, cor) => poli(c, [P(16.43, y0, z0), P(16.43, y1, z0), P(16.43, y1, z1), P(16.43, y0, z1)], cor);
      f(1.5, 5.5, 1.38, 1.5, ACENTO);
      const q = d.quadro || {}, notas = [];
      for (let i = 0; i < Math.min(6, q.vagas_abertas || 0); i++) notas.push('#BFE3C9');
      for (let i = 0; i < Math.min(4, q.vagas_em_processamento || 0); i++) notas.push('#FFE69A');
      for (let i = 0; i < Math.min(4, q.pendencias || 0); i++) notas.push('#F6B8B1');
      notas.forEach((cor, i) => { const y0 = 1.7 + (i % 6) * 0.62, z1 = 1.28 - Math.floor(i / 6) * 0.3; f(y0, y0 + 0.5, z1 - 0.24, z1, cor); });
      alvos.push({ tipo: 'quadro', pts: [P(16.42, 1.5, 1.5), P(16.42, 5.5, 1.5), P(16.42, 5.5, 0.3), P(16.42, 1.5, 0.3)] });
    });
    add(2 * 22.7 + 2 * 0.85, c => {
      caixa(c, 21.0, 0.35, 22.7, 0.85, 0, 1.4, '#FFFFFF', { sombra: 1 });
      const cores = ['#C9D3E3', '#E9D6B8', '#B8D2C2', '#DAC3D9', '#D7DAE0'];
      for (let p = 0; p < 3; p++) for (let i = 0; i < 9; i++) {
        const x = 21.1 + i * 0.17, z = 0.15 + p * 0.42;
        poli(c, [P(x, 0.86, z), P(x + 0.13, 0.86, z), P(x + 0.13, 0.86, z + 0.3 - (i % 3) * 0.04), P(x, 0.86, z + 0.3 - (i % 3) * 0.04)], cores[(i + p) % 5]);
      }
    });
    add(22.5 * 2 + 7.9 * 2, c => planta(c, 22.5, 7.9)); add(16.5 * 2 + 0.9 * 2, c => planta(c, 16.6, 0.9));

    // ---- alimentação
    add(2 * 24.7 + 2 * 1.35, c => {
      caixa(c, 23.7, 0.45, 24.7, 1.35, 0, 2.0, '#F4F6F8', { sombra: 1 });
      const f = (x0, x1, z0, z1, cor) => poli(c, [P(x0, 1.36, z0), P(x1, 1.36, z0), P(x1, 1.36, z1), P(x0, 1.36, z1)], cor);
      f(23.7, 24.7, 1.28, 1.3, '#D2D6DD'); f(24.55, 24.6, 1.4, 1.8, '#A7AEB9'); f(24.55, 24.6, 0.6, 1.1, '#A7AEB9');
    });
    add(2 * 31.4 + 2 * 1.1, c => {
      caixa(c, 25.4, 0.35, 31.4, 1.1, 0, 0.95, '#FFFFFF', { sombra: 1 });
      caixa(c, 26.4, 0.5, 27.4, 0.95, 0.93, 0.96, '#C9CED6');
      caixa(c, 29.2, 0.45, 29.75, 0.9, 0.95, 1.42, '#3A3F4A'); caixa(c, 30.2, 0.45, 31.1, 0.95, 0.95, 1.35, '#E9EBEF');
      for (let i = 0; i < 4; i++) caixa(c, 25.6 + i * 1.45, 0.36, 26.9 + i * 1.45, 0.4, 0.6, 0.62, '#E2E4EA');
    });
    for (const tx of [26.3, 29.8, 33.3]) {
      const ty = 4.7;
      for (const [dx, dy, f] of [[0, -1.15, 's'], [-1.15, 0, 'e'], [1.15, 0, 'w'], [0, 1.15, 'n']]) cad(tx + dx, ty + dy, f, '#C3C8D0');
      ocl(tx * 2 + ty * 2 + 0.1, [tx - 0.75, ty - 0.75, tx + 0.75, ty + 0.75, 0.76], c => {
        cilindro(c, tx, ty, 0.12, 0, 0.7, '#B8BEC8'); cilindro(c, tx, ty, 0.75, 0.7, 0.76, '#FFFFFF'); bola(c, tx + 0.2, ty - 0.1, 0.8, 0.07, '#F2C14E');
      });
    }
    add(35.4 * 2 + 7.9 * 2, c => planta(c, 35.4, 7.9, 1.1)); add(23.8 * 2 + 7.9 * 2, c => planta(c, 23.8, 7.9));

    // ---- análise de currículos: 12 estações
    ESTACOES.forEach(e => cad(e.x, e.y, 'n')); // o encosto fica na frente de quem trabalha
    for (const fy of [13.0, 16.7, 20.4]) for (const [x0, x1] of BLOCOS) {
      const est = ESTACOES.map((e, i) => [e, i]).filter(([e]) => e.mesaY === fy && e.x > x0 && e.x < x1);
      ocl(x0 + x1 + fy * 2 + 0.8, [x0, fy - 0.06, x1, fy + 0.8, 1.32], c => {
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
    }
    add(2 * 0.9 + 2 * 17.6, c => { caixa(c, 0.35, 16.0, 0.9, 17.6, 0, 1.05, '#FFFFFF', { sombra: 1 }); caixa(c, 0.4, 16.4, 0.85, 17.2, 1.05, 1.25, '#D9DCE3'); });
    add(20.4 * 2 + 11.9 * 2, c => { cilindro(c, 20.4, 11.9, 0.22, 0, 1.0, '#E6EBF2'); cilindro(c, 20.4, 11.9, 0.16, 1.0, 1.42, '#BFDDF2'); });
    add(0.75 * 2 + 11.8 * 2, c => planta(c, 0.75, 11.8)); add(0.75 * 2 + 23.3 * 2, c => planta(c, 0.75, 23.3, 1.2)); add(20.5 * 2 + 23.4 * 2, c => planta(c, 20.5, 23.4));

    // ---- descanso e entretenimento
    add(2 * 32.2 + 2 * 11.85, c => {
      caixa(c, 28.4, 11.35, 32.0, 11.85, 0, 0.4, '#FFFFFF', { sombra: 1 });
      caixa(c, 28.2, 11.3, 32.2, 11.4, 0.45, 1.05, '#22252C');
      poli(c, [P(28.28, 11.41, 0.49), P(32.12, 11.41, 0.49), P(32.12, 11.41, 1.01), P(28.28, 11.41, 1.01)], d.tv_ligada ? '#6FA8E8' : '#343944');
      if (d.tv_ligada) poli(c, [P(28.9, 11.42, 0.56), P(30.6, 11.42, 0.56), P(30.6, 11.42, 0.9), P(28.9, 11.42, 0.9)], '#9FD0F5');
    });
    add(29.2 + 31.2 + 12.6 + 13.1, c => caixa(c, 29.2, 12.6, 31.2, 13.1, 0, 0.38, '#FFFFFF', { sombra: 1, linha: '#E3E5EA' }));
    add(28.4 + 32.2 + 13.7 + 14.6 - 0.2, c => {
      caixa(c, 28.4, 13.7, 32.2, 14.6, 0, 0.42, '#C2CEDA', { sombra: 1 });
      caixa(c, 28.4, 13.7, 28.66, 14.32, 0.42, 0.62, '#B6C3D0'); caixa(c, 31.94, 13.7, 32.2, 14.32, 0.42, 0.62, '#B6C3D0');
    });
    ocl(28.4 + 32.2 + 14.32 + 14.6, [28.4, 14.32, 32.2, 14.6, 0.88], c => caixa(c, 28.4, 14.32, 32.2, 14.6, 0.42, 0.88, '#B6C3D0'));
    ocl(22.6 + 26.2 + 15.4 + 17.4, [22.6, 15.4, 26.2, 17.4, 0.94], c => {
      for (const [x, y] of [[22.9, 15.6], [25.8, 15.6], [22.9, 17.05], [25.8, 17.05]]) caixa(c, x, y, x + 0.1, y + 0.1, 0, 0.72, '#8D939E');
      caixa(c, 22.6, 15.4, 26.2, 17.4, 0.72, 0.78, '#4E8E77', { dir: '#3F735F', esq: '#46806B' });
      poli(c, [P(22.65, 16.38, 0.781), P(26.15, 16.38, 0.781), P(26.15, 16.42, 0.781), P(22.65, 16.42, 0.781)], '#E9F2EE');
      caixa(c, 24.38, 15.3, 24.42, 17.5, 0.78, 0.94, '#F4F4F4');
    });
    for (const [x, y] of [[30.5, 21.3], [32.6, 20.0], [34.3, 21.6]]) add(x * 2 + y * 2 - 0.2, c => { cilindro(c, x, y, 0.5, 0, 0.32, '#B9C6E6'); bola(c, x, y, 0.38, 0.32, '#B9C6E6'); });
    // canto de leitura: estante na parede, duas poltronas e um abajur
    add(2 * 21.9 + 2 * 22.8, c => {
      caixa(c, 21.35, 19.4, 21.85, 23.0, 0, 1.7, '#FFFFFF', { sombra: 1 });
      const cores = ['#C9D3E3', '#E9D6B8', '#B8D2C2', '#DAC3D9', '#F2C9B6'];
      for (let p = 0; p < 4; p++) for (let i = 0; i < 16; i++) {
        const y = 19.5 + i * 0.21, z = 0.12 + p * 0.4;
        poli(c, [P(21.86, y, z), P(21.86, y + 0.16, z), P(21.86, y + 0.16, z + 0.3 - (i % 3) * 0.05), P(21.86, y, z + 0.3 - (i % 3) * 0.05)], cores[(i + p) % 5]);
      }
    });
    for (const y of [21.0, 22.6]) {
      add(23.0 * 2 + y * 2, c => {
        caixa(c, 22.7, y - 0.4, 23.4, y + 0.4, 0, 0.42, '#D8C8E8', { sombra: 1 });
        caixa(c, 22.6, y - 0.4, 22.82, y + 0.4, 0.42, 0.95, '#C9B6DC');
        caixa(c, 22.7, y - 0.48, 23.4, y - 0.36, 0.42, 0.62, '#C9B6DC'); caixa(c, 22.7, y + 0.36, 23.4, y + 0.48, 0.42, 0.62, '#C9B6DC');
      });
    }
    add(23.9 * 2 + 21.8 * 2, c => { cilindro(c, 23.9, 21.8, 0.22, 0, 0.5, '#FFFFFF'); cilindro(c, 23.9, 21.8, 0.03, 0.5, 1.35, '#8D939E'); bola(c, 23.9, 21.8, 1.42, 0.16, '#FFE7A8'); });
    add(35.3 * 2 + 11.9 * 2, c => planta(c, 35.3, 11.9, 1.1)); add(35.3 * 2 + 23.3 * 2, c => planta(c, 35.3, 23.3, 1.2)); add(27.0 * 2 + 23.4 * 2, c => planta(c, 27.0, 23.4));
    add(1.1 * 2 + 9.2 * 2, c => planta(c, 1.0, 9.3, 0.9)); add(35.3 * 2 + 9.5 * 2, c => planta(c, 35.3, 9.4, 0.9));

    itens.sort((a, b) => a.k - b.k).forEach(i => i.f(cf));
    OCLUSORES.sort((a, b) => a.k - b.k);
    textoParede(cf, (d.marca || 'AI HUB').toUpperCase().slice(0, 18), 4.2, 0.31, 1.45, 0.42, ACENTO);
  }

  // redesenha, por cima da pessoa em (x, y), o que está na frente dela e a cobre na tela
  function ocluir(c, x, y, tela) {
    REDESENHO = true;
    try {
      for (const o of OCLUSORES) {
        const [x0, y0, x1, y1] = o.box;
        if (!((x < x0 && y < y1) || (y < y0 && x < x1))) continue;
        const t = o.tela;
        if (t.x1 < tela.x0 || t.x0 > tela.x1 || t.y1 < tela.y0 || t.y0 > tela.y1) continue;
        o.f(c);
      }
    } finally { REDESENHO = false; }
  }

  // ------------------------------------------------------------------ pessoas (corpo inteiro, com volume)
  const CALCAS = ['#2F3B52', '#3A3A3A', '#5A4632', '#4B5D7A', '#6B6F76', '#2E4A3E', '#283447'];
  const SAPATOS = ['#2B2B2B', '#EDEDED', '#5A3E2B', '#30405F'];
  function escolha(chave, lista) { let h = 7; for (const ch of String(chave)) h = (h * 31 + ch.charCodeAt(0)) >>> 0; return lista[h % lista.length]; }

  // cilindro em pé na tela, a partir da junta (x, y), girado por ang; luz vinda da esquerda
  function membro(c, x, y, ang, comp, larg, cor, mao) {
    c.save(); c.translate(x, y); c.rotate(ang);
    const g = c.createLinearGradient(-larg / 2, 0, larg / 2, 0);
    g.addColorStop(0, sombra(cor, 0.96)); g.addColorStop(0.35, sombra(cor, 1.16)); g.addColorStop(1, sombra(cor, 0.7));
    c.fillStyle = g; c.beginPath(); c.roundRect(-larg / 2, -larg * 0.3, larg, comp + larg * 0.3, larg / 2); c.fill();
    c.strokeStyle = 'rgba(25,25,40,.25)'; c.lineWidth = Math.max(0.6, larg * 0.1); c.stroke();
    if (mao) {
      const gm = c.createRadialGradient(-larg * 0.15, comp - larg * 0.1, larg * 0.05, 0, comp, larg * 0.6);
      gm.addColorStop(0, sombra(mao, 1.1)); gm.addColorStop(1, sombra(mao, 0.85));
      c.fillStyle = gm; c.beginPath(); c.arc(0, comp, larg * 0.55, 0, 7); c.fill();
    }
    c.restore();
  }
  const angulo = (dx, dy) => Math.atan2(-dx, dy);
  function sapato(c, x, y, u, cor, lado) {
    const g = c.createRadialGradient(x - u * 0.04, y - u * 0.03, u * 0.01, x, y, u * 0.13);
    g.addColorStop(0, sombra(cor, 1.25)); g.addColorStop(1, sombra(cor, 0.8));
    c.fillStyle = g; c.beginPath(); c.ellipse(x + lado * u * 0.03, y, u * 0.11, u * 0.06, 0, 0, 7); c.fill();
  }
  function tronco(c, sx, yOmbro, yQuadril, largO, largQ, cor, frente, pele) {
    const r = largO * 0.3;
    c.beginPath();
    c.moveTo(sx - largQ / 2, yQuadril);
    c.lineTo(sx - largO / 2, yOmbro + r); c.quadraticCurveTo(sx - largO / 2, yOmbro, sx - largO / 2 + r, yOmbro);
    c.lineTo(sx + largO / 2 - r, yOmbro); c.quadraticCurveTo(sx + largO / 2, yOmbro, sx + largO / 2, yOmbro + r);
    c.lineTo(sx + largQ / 2, yQuadril); c.closePath();
    const g = c.createLinearGradient(sx - largO / 2, yOmbro, sx + largO / 2, yQuadril);
    g.addColorStop(0, sombra(cor, 1.12)); g.addColorStop(0.45, cor); g.addColorStop(1, sombra(cor, 0.72));
    c.fillStyle = g; c.fill();
    c.strokeStyle = 'rgba(25,25,40,.28)'; c.lineWidth = Math.max(0.7, largO * 0.04); c.stroke();
    if (frente) { // gola em V
      c.fillStyle = sombra(pele, 0.95); c.beginPath(); c.moveTo(sx - largO * 0.14, yOmbro); c.lineTo(sx, yOmbro + largO * 0.2); c.lineTo(sx + largO * 0.14, yOmbro); c.fill();
    } else {
      c.strokeStyle = 'rgba(0,0,0,.12)'; c.beginPath(); c.moveTo(sx, yOmbro + largO * 0.12); c.lineTo(sx, yQuadril - largO * 0.1); c.stroke();
    }
  }

  function cabecaFrente(c, sx, sy, R, ap, o) {
    const pele = ap.pele || '#EEC49C', cab = ap.cabelo || '#3B2618', est = ap.estilo || 'curto';
    const vf = (o.virar || 0) * R * 0.13; // rosto levemente virado para onde a pessoa olha
    c.save();
    if (est === 'longo') { c.fillStyle = cab; c.beginPath(); c.ellipse(sx, sy + R * 0.25, R * 1.05, R * 1.1, 0, 0, 7); c.fill(); }
    if (est === 'black') { c.fillStyle = cab; c.beginPath(); c.arc(sx, sy - R * 0.15, R * 1.22, 0, 7); c.fill(); }
    c.fillStyle = sombra(pele, 0.93);
    c.beginPath(); c.arc(sx - R * 0.93 + vf * 0.3, sy + R * 0.08, R * 0.2, 0, 7); c.arc(sx + R * 0.93 + vf * 0.3, sy + R * 0.08, R * 0.2, 0, 7); c.fill();
    const g = c.createRadialGradient(sx - R * 0.35, sy - R * 0.4, R * 0.15, sx, sy, R * 1.05);
    g.addColorStop(0, sombra(pele, 1.12)); g.addColorStop(1, sombra(pele, 0.88));
    c.fillStyle = g; c.beginPath(); c.ellipse(sx, sy, R * 0.92, R, 0, 0, 7); c.fill();
    c.strokeStyle = 'rgba(25,25,40,.18)'; c.lineWidth = Math.max(0.6, R * 0.05); c.stroke();
    c.fillStyle = cab;
    if (est === 'curto' || est === 'longo' || est === 'coque' || est === 'topete') {
      c.beginPath(); c.moveTo(sx - R * 0.95, sy - R * 0.05);
      c.quadraticCurveTo(sx - R * 1.0, sy - R * 1.12, sx, sy - R * 1.1);
      c.quadraticCurveTo(sx + R * 1.0, sy - R * 1.12, sx + R * 0.95, sy - R * 0.05);
      c.quadraticCurveTo(sx + R * 0.7, sy - R * 0.62, sx + R * 0.1 + vf, sy - R * 0.55);
      c.quadraticCurveTo(sx - R * 0.6, sy - R * 0.5, sx - R * 0.95, sy - R * 0.05); c.fill();
      if (est === 'topete') { c.beginPath(); c.ellipse(sx + R * 0.15, sy - R * 0.95, R * 0.62, R * 0.32, -0.25, 0, 7); c.fill(); }
      if (est === 'coque') { c.beginPath(); c.arc(sx, sy - R * 1.22, R * 0.36, 0, 7); c.fill(); }
      if (est === 'longo') { c.fillRect(sx - R * 1.0, sy - R * 0.2, R * 0.22, R * 1.1); c.fillRect(sx + R * 0.78, sy - R * 0.2, R * 0.22, R * 1.1); }
    } else if (est === 'cacheado' || est === 'black') {
      const n = est === 'black' ? 9 : 7;
      for (let i = 0; i < n; i++) { const a = Math.PI * (1.05 + i * 0.9 / (n - 1)); c.beginPath(); c.arc(sx + Math.cos(a) * R * 0.82, sy - R * 0.2 + Math.sin(a) * R * 0.85, R * 0.34, 0, 7); c.fill(); }
      c.beginPath(); c.ellipse(sx, sy - R * 0.72, R * 0.75, R * 0.38, 0, 0, 7); c.fill();
    } else if (est === 'careca') {
      c.fillStyle = 'rgba(255,255,255,.2)'; c.beginPath(); c.ellipse(sx - R * 0.3, sy - R * 0.6, R * 0.3, R * 0.15, -0.5, 0, 7); c.fill();
    }
    // brilho no cabelo (volume)
    if (est !== 'careca') { c.fillStyle = 'rgba(255,255,255,.13)'; c.beginPath(); c.ellipse(sx - R * 0.35, sy - R * 0.82, R * 0.32, R * 0.12, -0.35, 0, 7); c.fill(); }
    if (ap.barba) { c.fillStyle = sombra(cab, 1.05); c.globalAlpha = 0.85; c.beginPath(); c.ellipse(sx + vf, sy + R * 0.55, R * 0.62, R * 0.38, 0, 0, Math.PI); c.fill(); c.globalAlpha = 1; }
    const olhoY = sy + R * 0.02, dx = R * 0.34, x0 = sx + vf, dormindo = o.humor === 'calmo';
    c.fillStyle = '#FFFFFF';
    if (!dormindo) { c.beginPath(); c.ellipse(x0 - dx, olhoY, R * 0.17, R * 0.2, 0, 0, 7); c.ellipse(x0 + dx, olhoY, R * 0.17, R * 0.2, 0, 0, 7); c.fill(); }
    c.fillStyle = '#2A2320';
    const ol = (o.olhar || 0) + vf * 0.4;
    if (dormindo) {
      c.strokeStyle = '#2A2320'; c.lineWidth = Math.max(1, R * 0.08);
      c.beginPath(); c.arc(x0 - dx, olhoY - R * 0.02, R * 0.12, 0.15 * Math.PI, 0.85 * Math.PI); c.stroke();
      c.beginPath(); c.arc(x0 + dx, olhoY - R * 0.02, R * 0.12, 0.15 * Math.PI, 0.85 * Math.PI); c.stroke();
    } else {
      c.beginPath(); c.arc(x0 - dx + ol, olhoY + R * 0.04, R * 0.1, 0, 7); c.arc(x0 + dx + ol, olhoY + R * 0.04, R * 0.1, 0, 7); c.fill();
      c.fillStyle = '#FFFFFF'; c.beginPath(); c.arc(x0 - dx + ol + R * 0.04, olhoY, R * 0.035, 0, 7); c.arc(x0 + dx + ol + R * 0.04, olhoY, R * 0.035, 0, 7); c.fill();
    }
    c.strokeStyle = sombra(cab, 0.9); c.lineWidth = Math.max(1, R * 0.07); c.lineCap = 'round';
    const pr = o.humor === 'preocupado' ? 0.1 : 0;
    c.beginPath(); c.moveTo(x0 - dx - R * 0.16, olhoY - R * 0.3 + pr * R); c.lineTo(x0 - dx + R * 0.14, olhoY - R * 0.33 - pr * R);
    c.moveTo(x0 + dx - R * 0.14, olhoY - R * 0.33 - pr * R); c.lineTo(x0 + dx + R * 0.16, olhoY - R * 0.3 + pr * R); c.stroke();
    c.fillStyle = 'rgba(232,120,110,.22)'; c.beginPath(); c.arc(x0 - R * 0.55, sy + R * 0.32, R * 0.15, 0, 7); c.arc(x0 + R * 0.55, sy + R * 0.32, R * 0.15, 0, 7); c.fill();
    c.strokeStyle = '#8A4B3C'; c.lineWidth = Math.max(1, R * 0.08);
    c.beginPath();
    if (o.humor === 'preocupado') c.arc(x0, sy + R * 0.58, R * 0.16, 1.15 * Math.PI, 1.85 * Math.PI);
    else if (o.humor === 'focado') { c.moveTo(x0 - R * 0.14, sy + R * 0.45); c.lineTo(x0 + R * 0.14, sy + R * 0.45); }
    else c.arc(x0, sy + R * 0.32, R * 0.22, 0.2 * Math.PI, 0.8 * Math.PI);
    c.stroke();
    if (ap.oculos) {
      c.strokeStyle = '#2A2F3A'; c.lineWidth = Math.max(1, R * 0.08);
      c.beginPath(); c.roundRect(x0 - dx - R * 0.26, olhoY - R * 0.2, R * 0.5, R * 0.4, R * 0.12); c.roundRect(x0 + dx - R * 0.24, olhoY - R * 0.2, R * 0.5, R * 0.4, R * 0.12);
      c.moveTo(x0 - R * 0.1, olhoY - R * 0.04); c.lineTo(x0 + R * 0.1, olhoY - R * 0.04); c.stroke();
    }
    c.restore();
  }

  function cabecaCostas(c, sx, sy, R, ap) {
    const pele = ap.pele || '#EEC49C', cab = ap.cabelo || '#3B2618', est = ap.estilo || 'curto';
    c.save();
    c.fillStyle = sombra(pele, 0.9);
    c.beginPath(); c.arc(sx - R * 0.93, sy + R * 0.1, R * 0.19, 0, 7); c.arc(sx + R * 0.93, sy + R * 0.1, R * 0.19, 0, 7); c.fill();
    const g = c.createRadialGradient(sx - R * 0.35, sy - R * 0.4, R * 0.15, sx, sy, R * 1.05);
    if (est === 'careca') {
      g.addColorStop(0, sombra(pele, 1.12)); g.addColorStop(1, sombra(pele, 0.86));
      c.fillStyle = g; c.beginPath(); c.ellipse(sx, sy, R * 0.92, R, 0, 0, 7); c.fill();
      c.fillStyle = 'rgba(255,255,255,.2)'; c.beginPath(); c.ellipse(sx - R * 0.3, sy - R * 0.55, R * 0.3, R * 0.15, -0.5, 0, 7); c.fill();
    } else {
      g.addColorStop(0, sombra(cab, 1.35)); g.addColorStop(1, sombra(cab, 0.85));
      c.fillStyle = sombra(pele, 0.92); c.beginPath(); c.ellipse(sx, sy + R * 0.2, R * 0.55, R * 0.75, 0, 0, 7); c.fill(); // nuca
      c.fillStyle = g;
      if (est === 'black') { c.beginPath(); c.arc(sx, sy - R * 0.15, R * 1.22, 0, 7); c.fill(); }
      else if (est === 'cacheado') {
        for (let i = 0; i < 9; i++) { const a = Math.PI * (0.95 + i * 1.1 / 8); c.beginPath(); c.arc(sx + Math.cos(a) * R * 0.78, sy - R * 0.1 + Math.sin(a) * R * 0.8, R * 0.36, 0, 7); c.fill(); }
        c.beginPath(); c.ellipse(sx, sy - R * 0.2, R * 0.85, R * 0.75, 0, 0, 7); c.fill();
      } else {
        c.beginPath(); c.ellipse(sx, sy - R * 0.08, R * 0.95, R * 0.98, 0, Math.PI * 0.92, Math.PI * 2.08); c.closePath(); c.fill();
        if (est === 'longo') { c.beginPath(); c.roundRect(sx - R * 0.95, sy - R * 0.2, R * 1.9, R * 1.75, R * 0.5); c.fill(); }
        if (est === 'coque') { c.beginPath(); c.arc(sx, sy - R * 1.18, R * 0.36, 0, 7); c.fill(); }
        if (est === 'topete') { c.beginPath(); c.ellipse(sx + R * 0.1, sy - R * 0.92, R * 0.6, R * 0.3, -0.2, 0, 7); c.fill(); }
      }
      c.fillStyle = 'rgba(255,255,255,.13)'; c.beginPath(); c.ellipse(sx - R * 0.35, sy - R * 0.7, R * 0.34, R * 0.14, -0.35, 0, 7); c.fill();
    }
    c.restore();
  }

  function item(c, tipo, x, y, u) {
    if (tipo === 'raquete') {
      c.fillStyle = '#8A5A3C'; c.fillRect(x - u * 0.02, y - u * 0.02, u * 0.04, u * 0.12);
      c.fillStyle = '#D64545'; c.beginPath(); c.ellipse(x, y - u * 0.1, u * 0.1, u * 0.12, 0, 0, 7); c.fill();
    } else if (tipo === 'xicara') {
      c.fillStyle = '#FFFFFF'; c.beginPath(); c.roundRect(x - u * 0.06, y - u * 0.1, u * 0.12, u * 0.12, u * 0.02); c.fill();
      c.strokeStyle = '#C9CED6'; c.lineWidth = Math.max(0.6, u * 0.02); c.stroke();
      c.fillStyle = '#6B4128'; c.fillRect(x - u * 0.045, y - u * 0.095, u * 0.09, u * 0.025);
    } else if (tipo === 'livro') {
      c.fillStyle = '#5B5BD6'; c.beginPath(); c.roundRect(x - u * 0.17, y - u * 0.14, u * 0.34, u * 0.22, u * 0.02); c.fill();
      c.fillStyle = '#FFFFFF'; c.fillRect(x - u * 0.15, y - u * 0.12, u * 0.14, u * 0.18); c.fillRect(x + u * 0.01, y - u * 0.12, u * 0.14, u * 0.18);
    }
  }

  // Pessoa inteira com os pés em (sx, sy). o: {dir, sentado, andando, fase, humor, olhar, digitando, item, t}
  function pessoa(c, sx, sy, u, ap, o = {}) {
    const dir = o.dir || 's', frente = dir === 's' || dir === 'e', lado = (dir === 'e' || dir === 'n') ? 1 : -1;
    const pele = ap.pele || '#EEC49C', cam = ap.camisa || '#3B6EC8';
    const calca = ap.calca || escolha((ap.cabelo || '') + cam, CALCAS), sap = ap.sapato || escolha(cam + pele, SAPATOS);
    const R = u * 0.4, t = o.t || 0;
    const bob = o.andando ? -Math.abs(Math.sin(o.fase || 0)) * u * 0.05 : Math.sin(t * 1.6) * u * 0.012;
    c.save();
    if (o.sentado) {
      const yQ = sy - u * 0.46 + bob, yO = yQ - u * 0.56, yC = yO - u * 0.37;
      if (frente) { // coxas para frente (na direção do olhar) e canelas até o chão
        for (const k of [-1, 1]) {
          const hx = sx + k * u * 0.1, jx = hx + lado * u * 0.34, jy = yQ + u * 0.2;
          membro(c, hx, yQ, angulo(jx - hx, jy - yQ), Math.hypot(jx - hx, jy - yQ), u * 0.17, calca);
          membro(c, jx, jy, 0, sy + u * 0.2 - jy, u * 0.15, calca);
          sapato(c, jx, sy + u * 0.2, u, sap, lado);
        }
      }
      tronco(c, sx, yO, yQ, u * 0.58, u * 0.48, cam, frente, pele);
      const dig = o.digitando ? Math.sin(t * 14) * u * 0.02 : 0;
      for (const k of [-1, 1]) {
        const ox = sx + k * u * 0.28, oy = yO + u * 0.06;
        if (frente) { const hx = sx + k * u * 0.16 + lado * u * 0.22, hy = yQ + u * 0.08; membro(c, ox, oy, angulo(hx - ox, hy - oy), Math.hypot(hx - ox, hy - oy), u * 0.13, cam, pele); }
        else membro(c, ox, oy, k * -0.35, u * 0.36 + k * dig, u * 0.13, cam, pele);
      }
      if (frente) cabecaFrente(c, sx, yC, R, ap, { ...o, virar: lado }); else cabecaCostas(c, sx, yC, R, ap);
      if (o.item && frente) item(c, o.item, sx + lado * u * 0.1, yQ - u * 0.05, u);
    } else {
      const yQ = sy - u * 0.62 + bob, yO = yQ - u * 0.56, yC = yO - u * 0.36, f = o.fase || 0;
      for (const k of [-1, 1]) { // pernas: no andar, uma vai e a outra vem
        const sw = o.andando ? Math.sin(f + (k > 0 ? Math.PI : 0)) : 0;
        const hx = sx + k * u * 0.1, px = hx + sw * u * 0.09 * lado, py = sy - Math.max(0, sw) * u * 0.07 + bob * 0.2;
        membro(c, hx, yQ, angulo(px - hx, py - yQ), Math.hypot(px - hx, py - yQ), u * 0.16, calca);
        sapato(c, px, py, u, sap, lado);
      }
      tronco(c, sx, yO, yQ, u * 0.58, u * 0.46, cam, frente, pele);
      c.fillStyle = sombra(calca, 0.8); c.fillRect(sx - u * 0.23, yQ - u * 0.06, u * 0.46, u * 0.05); // cinto
      for (const k of [-1, 1]) {
        const sw = o.andando ? Math.sin(f + (k > 0 ? 0 : Math.PI)) * 0.3 : 0;
        const segura = o.item === 'raquete' && k === lado;
        membro(c, sx + k * u * 0.28, yO + u * 0.06, segura ? k * -0.9 : k * -0.12 + sw, u * (segura ? 0.36 : 0.48), u * 0.13, cam, pele);
        if (segura) item(c, 'raquete', sx + k * u * 0.6, yO + u * 0.26, u);
      }
      if (frente) cabecaFrente(c, sx, yC, R, ap, { ...o, virar: lado }); else cabecaCostas(c, sx, yC, R, ap);
    }
    c.restore();
  }

  // retrato (busto) para painéis e para o canal da equipe
  function avatar(c, sx, sy, R, ap, o = {}) {
    const pele = ap.pele || '#EEC49C', cam = ap.camisa || '#3B6EC8';
    const corpoAlt = o.sentado ? R * 0.95 : R * 1.45;
    c.save();
    const g = c.createLinearGradient(sx - R, sy + R * 0.6, sx + R, sy + R * 2);
    g.addColorStop(0, sombra(cam, 1.12)); g.addColorStop(1, sombra(cam, 0.78));
    c.fillStyle = g;
    c.beginPath(); c.moveTo(sx - R * 0.95, sy + R * 0.7 + corpoAlt); c.lineTo(sx - R * 0.95, sy + R * 1.15);
    c.quadraticCurveTo(sx - R * 0.9, sy + R * 0.62, sx, sy + R * 0.6); c.quadraticCurveTo(sx + R * 0.9, sy + R * 0.62, sx + R * 0.95, sy + R * 1.15);
    c.lineTo(sx + R * 0.95, sy + R * 0.7 + corpoAlt); c.closePath(); c.fill();
    c.fillStyle = sombra(pele, 0.9); c.beginPath(); c.ellipse(sx, sy + R * 0.66, R * 0.26, R * 0.14, 0, 0, 7); c.fill();
    c.restore();
    cabecaFrente(c, sx, sy, R, ap, o);
  }

  // ------------------------------------------------------------------ API pública
  const Cena = {
    ESTADOS, INFO_LOCAL, LOCAIS_DESCANSO, NOS, ESTACOES, rota, avatar, pessoa, ocluir,
    medir(w, h, dpr, zoom = 1) {
      DPR = dpr;
      // contorno da planta em projeção (com folga para cabeças e etiquetas no fundo)
      const largura = (LARG + PROF) * C + 1.2, altura = (LARG + PROF) * S + 0.35 + 2.6;
      T = Math.max(14, Math.min((w - 16) / largura, (h - 16) / altura)) * zoom;
      W = Math.max(w, largura * T + 16); H = Math.max(h, altura * T + 16);
      const minX = -PROF * C - 0.6, maxX = LARG * C + 0.6, minY = -2.6, maxY = (LARG + PROF) * S + 0.35;
      OX = W / 2 - ((minX + maxX) / 2) * T;
      OY = H / 2 - ((minY + maxY) / 2) * T;
      assinatura = '';
      return { largura: W, altura: H };
    },
    escala: () => T,
    projetar: P,
    atualizarFundo(d) {
      const ass = JSON.stringify([W, H, DPR, d.fila_triagem, d.fila_analise, d.estacoes, d.tv_ligada, d.atlas_trabalhando, d.marca,
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
