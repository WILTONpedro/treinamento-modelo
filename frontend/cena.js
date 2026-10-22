/* Cena do escritório em pixel art isométrica (projeção 2:1).
   Tudo é rasterizado num quadro de baixa resolução, com bordas nítidas (sem suavização), e ampliado em
   pixels inteiros quando possível; etiquetas e balões ficam nítidos por cima (app.js).
   Móveis e pessoas são desenhados em ordem de profundidade a cada quadro, então quem passa atrás de uma
   mesa, de um sofá ou do vidro da sala do gerente aparece por trás dele.
   Não inventa estados: só anima o que o app (dados da API) informa. */
'use strict';
(function () {
  // ------------------------------------------------------------------ planta e projeção (1 unidade = 1 m)
  const LARG = 24, PROF = 16, K = 16, KZ = 20, ALT_PAREDE = 2.6, M = 6;
  const AW = (LARG + PROF) * K + 2 * M;
  const AH = (LARG + PROF) * K / 2 + Math.round(ALT_PAREDE * KZ) + 10 + 2 * M;
  const OX = M + PROF * K, OY = M + Math.round(ALT_PAREDE * KZ);
  const Pa = (x, y, z = 0) => [OX + (x - y) * K, OY + (x + y) * K / 2 - z * KZ];

  const ESTADOS = {
    trabalhando: ['#3FB56B', 'Trabalhando'], aguardando: ['#4C8FD6', 'Aguardando'], descansando: ['#A07BE0', 'Descansando'],
    deslocamento: ['#F29A38', 'Em deslocamento'], pendencia: ['#E5534B', 'Com pendência'], disponivel: ['#2CC0B0', 'Disponível'],
    concluido: ['#8A93A6', 'Concluído'], conversando: ['#F2C14E', 'Conversando'],
  };
  const CONTORNO = '#1B1420';

  // ------------------------------------------------------------------ cores
  const RGB = {};
  function rgb(hex) { return RGB[hex] || (RGB[hex] = [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)]); }
  const empac = (r, g, b, a = 255) => ((a << 24) | (b << 16) | (g << 8) | r) >>> 0;
  const PK = {};
  const pk = hex => PK[hex] ?? (PK[hex] = empac(...rgb(hex)));
  function tom(hex, f) {
    const [r, g, b] = rgb(hex), h = v => Math.max(0, Math.min(255, Math.round(v * f))).toString(16).padStart(2, '0');
    return '#' + h(r) + h(g) + h(b);
  }
  function escolha(chave, lista) { let h = 7; for (const ch of String(chave)) h = (h * 31 + ch.charCodeAt(0)) >>> 0; return lista[h % lista.length]; }
  const ruido = (i, j) => { const s = Math.sin(i * 127.1 + j * 311.7) * 43758.5453; return s - Math.floor(s); };

  // ------------------------------------------------------------------ rasterizador (pixels inteiros, sem suavização)
  let A = null; // alvo: { buf: Uint32Array, w, h, ox, oy, sujo }
  function alvo(buf, w, h, ox = 0, oy = 0) { A = { buf, w, h, ox, oy, sujo: [1e9, 1e9, -1e9, -1e9] }; }
  function misturar(i, cor, a) {
    const d = A.buf[i], da = (d >>> 24) / 255, [r, g, b] = rgb(cor), oa = a + da * (1 - a);
    if (oa <= 0) return;
    const k = da * (1 - a);
    A.buf[i] = empac(Math.round((r * a + (d & 255) * k) / oa), Math.round((g * a + ((d >>> 8) & 255) * k) / oa),
      Math.round((b * a + ((d >>> 16) & 255) * k) / oa), Math.round(oa * 255));
  }
  function ponto(x, y, cor, alfa) {
    x -= A.ox; y -= A.oy;
    if (x < 0 || y < 0 || x >= A.w || y >= A.h) return;
    const i = y * A.w + x;
    if (alfa === undefined || alfa >= 1) A.buf[i] = pk(cor); else misturar(i, cor, alfa);
    const s = A.sujo; if (x < s[0]) s[0] = x; if (y < s[1]) s[1] = y; if (x > s[2]) s[2] = x; if (y > s[3]) s[3] = y;
  }
  function escurecer(x, y, f) { // sombra: multiplica o que já está desenhado
    x -= A.ox; y -= A.oy;
    if (x < 0 || y < 0 || x >= A.w || y >= A.h) return;
    const i = y * A.w + x, d = A.buf[i];
    if (!(d >>> 24)) return;
    A.buf[i] = empac(Math.round((d & 255) * f), Math.round(((d >>> 8) & 255) * f), Math.round(((d >>> 16) & 255) * f), d >>> 24);
  }
  function varrer(pts, cada) { // preenche o polígono pelos centros dos pixels (bordas nítidas)
    const n = pts.length; if (n < 3) return;
    const xs = [], ys = [];
    let y0 = 1e9, y1 = -1e9;
    for (const [x, y] of pts) { const yy = Math.round(y); xs.push(Math.round(x)); ys.push(yy); if (yy < y0) y0 = yy; if (yy > y1) y1 = yy; }
    const cortes = [];
    for (let y = y0; y < y1; y++) {
      const yc = y + 0.5; cortes.length = 0;
      for (let i = 0, j = n - 1; i < n; j = i++) if ((ys[i] <= yc) !== (ys[j] <= yc)) cortes.push(xs[i] + (yc - ys[i]) * (xs[j] - xs[i]) / (ys[j] - ys[i]));
      cortes.sort((a, b) => a - b);
      for (let k = 0; k + 1 < cortes.length; k += 2) for (let x = Math.ceil(cortes[k] - 0.5); x <= Math.floor(cortes[k + 1] - 0.5); x++) cada(x, y);
    }
  }
  const poli = (pts, cor, alfa) => varrer(pts, (x, y) => ponto(x, y, cor, alfa));
  const sombrear = (pts, f) => varrer(pts, (x, y) => escurecer(x, y, f));
  function linha(x0, y0, x1, y1, cor, alfa) {
    x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1);
    const dx = Math.abs(x1 - x0), dy = -Math.abs(y1 - y0), sx = x0 < x1 ? 1 : -1, sy = y0 < y1 ? 1 : -1;
    let e = dx + dy;
    for (let n = 0; n < 4000; n++) {
      ponto(x0, y0, cor, alfa);
      if (x0 === x1 && y0 === y1) break;
      const e2 = 2 * e;
      if (e2 >= dy) { e += dy; x0 += sx; }
      if (e2 <= dx) { e += dx; y0 += sy; }
    }
  }
  function disco(cx, cy, rx, ry, cor, alfa) {
    for (let y = Math.floor(cy - ry); y <= Math.ceil(cy + ry); y++)
      for (let x = Math.floor(cx - rx); x <= Math.ceil(cx + rx); x++) {
        const u = (x + 0.5 - cx) / rx, v = (y + 0.5 - cy) / ry;
        if (u * u + v * v <= 1) ponto(x, y, cor, alfa);
      }
  }
  const ret = (x, y, w, h, cor, alfa) => { for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) ponto(x + i, y + j, cor, alfa); };

  // ------------------------------------------------------------------ primitivas isométricas
  // caixa: topo claro, lado esquerdo médio, lado direito escuro (luz vinda das janelas da esquerda) e contorno
  function caixa(x0, y0, x1, y1, z0, z1, cor, o = {}) {
    const a = Pa(x0, y0, z1), b = Pa(x1, y0, z1), c = Pa(x1, y1, z1), d = Pa(x0, y1, z1);
    const e = Pa(x1, y0, z0), f = Pa(x1, y1, z0), g = Pa(x0, y1, z0);
    const topo = o.topo || cor;
    if (z1 > z0) { poli([d, c, f, g], o.esq || tom(cor, 0.84), o.alfa); poli([c, b, e, f], o.dir || tom(cor, 0.68), o.alfa); }
    poli([a, b, c, d], topo, o.alfa);
    if (o.brilho !== false && z1 > z0 && !o.alfa) { linha(d[0], d[1], c[0], c[1], tom(topo, 1.12)); linha(c[0], c[1], b[0], b[1], tom(topo, 1.06)); }
    if (o.contorno !== false) {
      const k = o.contorno || tom(cor, 0.4), lados = z1 > z0 ? [[a, b], [b, e], [e, f], [f, g], [g, d], [d, a]] : [[a, b], [b, c], [c, d], [d, a]];
      for (const [p, q] of lados) linha(p[0], p[1], q[0], q[1], k, o.alfa ? Math.min(1, o.alfa * 2.5) : undefined);
    }
  }
  // retângulo pintado numa face vertical: eixo 'x' = parede/face no plano y = c; eixo 'y' = plano x = c
  function painel(eixo, c, a0, a1, z0, z1, cor, alfa) {
    const q = eixo === 'x' ? [Pa(a0, c, z1), Pa(a1, c, z1), Pa(a1, c, z0), Pa(a0, c, z0)] : [Pa(c, a0, z1), Pa(c, a1, z1), Pa(c, a1, z0), Pa(c, a0, z0)];
    poli(q, cor, alfa);
  }
  function piso(x0, y0, x1, y1, cor, alfa, z = 0) { poli([Pa(x0, y0, z), Pa(x1, y0, z), Pa(x1, y1, z), Pa(x0, y1, z)], cor, alfa); }
  function cilindro(x, y, r, z0, z1, cor, o = {}) {
    const N = 24, topo = [], baseC = [];
    for (let i = 0; i < N; i++) { const t = i / N * Math.PI * 2; topo.push(Pa(x + Math.cos(t) * r, y + Math.sin(t) * r, z1)); baseC.push(Pa(x + Math.cos(t) * r, y + Math.sin(t) * r, z0)); }
    if (z1 > z0) { // lateral visível: de -45° a 135° (metade voltada para quem olha), clara à esquerda e escura à direita
      const arco = (a, b) => { const r1 = []; for (let i = 0; i <= N; i++) { const t = a + (b - a) * i / N; r1.push(t); } return r1; };
      const pt = (t, z) => Pa(x + Math.cos(t) * r, y + Math.sin(t) * r, z);
      const lado = (a, b, cor2) => { const ts = arco(a, b); poli([...ts.map(t => pt(t, z1)), ...ts.reverse().map(t => pt(t, z0))], cor2); };
      lado(Math.PI / 4, Math.PI * 3 / 4, o.esq || tom(cor, 0.86)); lado(-Math.PI / 4, Math.PI / 4, o.dir || tom(cor, 0.7));
    }
    poli(topo, o.topo || cor);
    if (o.contorno !== false) {
      const k = o.contorno || tom(cor, 0.42);
      for (let i = 0; i < N; i++) { const p = topo[i], q = topo[(i + 1) % N]; linha(p[0], p[1], q[0], q[1], k); }
      if (z1 > z0) {
        const l = Pa(x + Math.cos(Math.PI * 3 / 4) * r, y + Math.sin(Math.PI * 3 / 4) * r, 0), rr = Pa(x + Math.cos(-Math.PI / 4) * r, y + Math.sin(-Math.PI / 4) * r, 0);
        linha(l[0], l[1] - z1 * KZ, l[0], l[1] - z0 * KZ, k); linha(rr[0], rr[1] - z1 * KZ, rr[0], rr[1] - z0 * KZ, k);
        for (let i = 0; i < N; i++) { const t = i / N * Math.PI * 2; if (Math.cos(t) + Math.sin(t) <= -0.05) continue; const p = baseC[i], q = baseC[(i + 1) % N]; linha(p[0], p[1], q[0], q[1], k); }
      }
    }
  }
  function brilhoChao(x, y, r, cor, alfa) { // poça de luz em anéis (luminárias)
    for (const [f, a] of [[1, alfa * 0.5], [0.7, alfa * 0.75], [0.42, alfa]]) {
      const [cx, cy] = Pa(x, y, 0); disco(cx, cy, r * f * K * 1.41, r * f * K * 0.71, cor, a);
    }
  }

  // ------------------------------------------------------------------ paleta do ambiente
  const COR = {
    piso1: '#D8B285', piso2: '#CDA679', madeira: '#B47C4C', tabua: '#9C673B', cozinha1: '#E6DBC6', cozinha2: '#D8CBB0',
    lajeE: '#5C3D29', lajeD: '#45301F', paredeE: '#2C3253', paredeD: '#363D66', paredeTopo: '#4B5384', rodape: '#4A3121', moldura: '#525B8E',
    movel: '#A46A3B', movelEsc: '#6E4225', crt: '#E3D7BA', cadeira: '#3A3F57', vidro: '#B7E0F5', metal: '#2B2F3D',
    folha1: '#3F9147', folha2: '#5DBA5F', folha3: '#2C6A33', vaso: '#B95A2E',
  };
  const LIVROS = ['#C0504D', '#4F81BD', '#9BBB59', '#F2C14E', '#8064A2', '#E07B39', '#2E8B88', '#D9D2C0'];

  // ------------------------------------------------------------------ estações, locais e rotas
  const ESTACOES = [];   // 12 estações: 3 ilhas × 2 fileiras × 2 mesas
  const ILHAS = [6.6, 11.2, 15.8];
  ILHAS.forEach(x0 => {
    for (const fila of ['A', 'B']) for (const col of [0, 1]) {
      const x = x0 + 0.7 + col * 1.4;
      ESTACOES.push(fila === 'A' ? { x, y: 5.55, face: 's', fila, x0 } : { x, y: 8.65, face: 'n', fila, x0 });
    }
  });
  // ordem: 0-1 fileira A da 1ª ilha, 2-3 fileira B da 1ª ilha, ... (os primeiros analistas aparecem de frente)
  const NOS = {}, INFO_LOCAL = {};
  const no = (n, x, y, info) => { NOS[n] = [x, y]; if (info) INFO_LOCAL[n] = info; };
  ESTACOES.forEach((e, i) => {
    no('mesa' + i, e.x, e.y, { f: e.face, sentado: true, digita: true });
    no('c' + i, e.x, e.fila === 'A' ? 4.65 : 9.55, { f: e.face });
  });
  no('recepcao', 1.3, 11.1, { f: 'e', sentado: true, digita: true });
  no('atlas', 3.0, 1.4, { f: 's', sentado: true });
  no('sV', 3.0, 3.75, { f: 'n' });
  no('entrada', 0.55, 13.9, { f: 'e' });
  [10.4, 11.3, 12.2].forEach((y, i) => no('balcao' + i, 3.35, y, { f: 'w' }));
  [7.9, 9.3, 10.7].forEach((x, i) => no('quadro' + i, x, 1.45, { f: 'n' }));
  // locais de descanso: [nó, o que faz, para onde olha, sentado, objeto na mão]
  const LOCAIS_DESCANSO = [
    ['ping0', 'jogando pingue-pongue', 'e', 0, 'raquete'], ['ping1', 'jogando pingue-pongue', 'w', 0, 'raquete'],
    ['sofa0', 'assistindo TV', 'n', 1], ['sofa1', 'assistindo TV', 'n', 1], ['sofa2', 'assistindo TV', 'n', 1],
    ['comer0', 'tomando um café', 'e', 1, 'xicara'], ['comer1', 'fazendo um lanche', 'w', 1], ['comer2', 'tomando um café', 'n', 1, 'xicara'],
    ['comer3', 'fazendo um lanche', 'e', 1], ['comer4', 'tomando um café', 'w', 1, 'xicara'], ['comer5', 'fazendo um lanche', 'n', 1],
    ['geladeira', 'pegando algo na geladeira', 'n', 0],
    ['pufe0', 'descansando no pufe', 'n', 1], ['pufe1', 'descansando no pufe', 'n', 1], ['pufe2', 'descansando no pufe', 'n', 1],
    ['leitura0', 'lendo um livro', 's', 1, 'livro'], ['leitura1', 'lendo um livro', 's', 1, 'livro'],
  ];
  const POS_DESCANSO = {
    ping0: [19.0, 12.3], ping1: [23.0, 12.3], sofa0: [15.5, 13.75], sofa1: [16.3, 13.75], sofa2: [17.1, 13.75],
    comer0: [18.2, 2.9], comer1: [19.8, 2.9], comer2: [19.0, 3.7], comer3: [21.0, 2.9], comer4: [22.6, 2.9], comer5: [21.8, 3.7],
    geladeira: [23.1, 1.45], pufe0: [19.8, 14.95], pufe1: [21.2, 15.1], pufe2: [22.6, 14.95], leitura0: [7.8, 13.05], leitura1: [9.4, 13.05],
  };
  for (const [n, txt, f, s, item] of LOCAIS_DESCANSO) no(n, ...POS_DESCANSO[n], { txt, f, sentado: !!s, item });

  // ------------------------------------------------------------------ obstáculos e rotas (A* numa grade de 25 cm)
  const GR = 0.25, GW = LARG / GR, GH = PROF / GR, BLOQ = new Uint8Array(GW * GH);
  function bloquear(x0, y0, x1, y1, folga = 0.1) {
    for (let j = Math.max(0, Math.floor((y0 - folga) / GR)); j < Math.min(GH, Math.ceil((y1 + folga) / GR)); j++)
      for (let i = Math.max(0, Math.floor((x0 - folga) / GR)); i < Math.min(GW, Math.ceil((x1 + folga) / GR)); i++) BLOQ[j * GW + i] = 1;
  }
  const livre = (i, j) => i >= 0 && j >= 0 && i < GW && j < GH && !BLOQ[j * GW + i];
  function aEstrela(a, b) {
    const ci = p => [Math.min(GW - 1, Math.max(0, Math.floor(p[0] / GR))), Math.min(GH - 1, Math.max(0, Math.floor(p[1] / GR)))];
    const [si, sj] = ci(a), [ti, tj] = ci(b), S = sj * GW + si, T = tj * GW + ti;
    const g = new Float32Array(GW * GH).fill(Infinity), de = new Int32Array(GW * GH).fill(-1), fechado = new Uint8Array(GW * GH);
    const h = k => { const dx = Math.abs(k % GW - ti), dy = Math.abs(((k / GW) | 0) - tj); return Math.max(dx, dy) + 0.414 * Math.min(dx, dy); };
    const heap = [[h(S), S]];
    const sobe = () => { let i = heap.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (heap[p][0] <= heap[i][0]) break; [heap[p], heap[i]] = [heap[i], heap[p]]; i = p; } };
    const desce = () => { let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < heap.length && heap[l][0] < heap[m][0]) m = l; if (r < heap.length && heap[r][0] < heap[m][0]) m = r; if (m === i) break; [heap[m], heap[i]] = [heap[i], heap[m]]; i = m; } };
    g[S] = 0;
    const pode = (i, j, k) => k === T || k === S || livre(i, j);
    while (heap.length) {
      const [, u] = heap[0]; const ult = heap.pop(); if (heap.length) { heap[0] = ult; desce(); }
      if (fechado[u]) continue;
      if (u === T) break;
      fechado[u] = 1;
      const ui = u % GW, uj = (u / GW) | 0;
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        if (!di && !dj) continue;
        const vi = ui + di, vj = uj + dj; if (vi < 0 || vj < 0 || vi >= GW || vj >= GH) continue;
        const v = vj * GW + vi;
        if (!pode(vi, vj, v) || fechado[v]) continue;
        if (di && dj && !(pode(ui + di, uj, uj * GW + ui + di) && pode(ui, uj + dj, (uj + dj) * GW + ui))) continue; // sem cortar quina
        const ng = g[u] + (di && dj ? 1.414 : 1);
        if (ng < g[v]) { g[v] = ng; de[v] = u; heap.push([ng + h(v), v]); sobe(); }
      }
    }
    if (g[T] === Infinity) return null;
    const cel = [];
    for (let k = T; k !== -1 && k !== S; k = de[k]) cel.unshift(k);
    return cel.map(k => [(k % GW + 0.5) * GR, (((k / GW) | 0) + 0.5) * GR]);
  }
  function visivel(p, q) { // linha reta livre entre p e q
    const d = Math.hypot(q[0] - p[0], q[1] - p[1]), n = Math.ceil(d / 0.08);
    for (let s = 1; s < n; s++) {
      const x = p[0] + (q[0] - p[0]) * s / n, y = p[1] + (q[1] - p[1]) * s / n;
      if (!livre(Math.floor(x / GR), Math.floor(y / GR))) return false;
    }
    return true;
  }
  const ROTAS = {};
  function rota(de, para) {
    if (de === para || !NOS[de] || !NOS[para]) return NOS[para] && de !== para ? [para] : [];
    const chave = de + '>' + para, nomeados = de[0] !== '~';
    if (nomeados && ROTAS[chave]) return ROTAS[chave].slice();
    const a = NOS[de], b = NOS[para], cam = aEstrela(a, b);
    if (!cam) return [para];
    cam[cam.length - 1] = b;
    const pontos = [a, ...cam], suave = [a];
    let i = 0;
    while (i < pontos.length - 1) { // puxa o barbante: pula pontos enquanto a reta estiver livre
      let j = pontos.length - 1;
      while (j > i + 1 && !visivel(pontos[i], pontos[j])) j--;
      suave.push(pontos[j]); i = j;
    }
    const nomes = suave.slice(1, -1).map(([x, y]) => { const n = '~' + x.toFixed(2) + ',' + y.toFixed(2); NOS[n] = [x, y]; return n; });
    nomes.push(para);
    if (nomeados) ROTAS[chave] = nomes.slice();
    return nomes;
  }

  // ------------------------------------------------------------------ itens do cenário (cada um vira um sprite)
  let ITENS = [], ALVOS_ARTE = [];
  // box: pegada no chão [x0, y0, x1, y1]; z0/z1: altura; f(): desenha (nas coordenadas da arte)
  const item = (box, z0, z1, f, o = {}) => ITENS.push({ box, z0, z1, f, ...o });
  const BLOQUEIOS = [];
  const obst = (x0, y0, x1, y1) => BLOQUEIOS.push([x0, y0, x1, y1]);

  function mesaTrabalho(x0, y0, x1, y1, cor = COR.movel) {
    const pe = (x, y) => caixa(x, y, x + 0.07, y + 0.07, 0, 0.7, tom(cor, 0.55), { brilho: false });
    item([x0, y0, x1, y1], 0, 0.76, () => {
      pe(x0 + 0.05, y0 + 0.05); pe(x1 - 0.12, y0 + 0.05); pe(x0 + 0.05, y1 - 0.12); pe(x1 - 0.12, y1 - 0.12);
      caixa(x0, y0, x1, y1, 0.7, 0.76, cor);
    }, { sombra: true });
    obst(x0, y0, x1, y1);
  }
  function crt(x, y, z, face, tela) { // monitor de tubo; face: para onde a tela aponta
    const w = 0.22, p = 0.2, h = 0.4, alongX = face === 'n' || face === 's';
    const [bx0, by0, bx1, by1] = alongX ? [x - w, y - p, x + w, y + p] : [x - p, y - w, x + p, y + w];
    caixa(x - 0.12, y - 0.1, x + 0.12, y + 0.1, z, z + 0.04, tom(COR.crt, 0.8), { brilho: false });
    caixa(bx0, by0, bx1, by1, z + 0.04, z + h, COR.crt);
    const corTela = !tela ? '#232A28' : tela === 'trabalhando' ? '#14402A' : tela === 'pendencia' ? '#5A1F1C' : tela === 'concluido' ? '#1C5A33' : '#1D3442';
    if (face === 's' || face === 'e') { // tela visível
      const eixo = face === 's' ? 'x' : 'y', c = face === 's' ? by1 + 0.001 : bx1 + 0.001, a0 = (face === 's' ? bx0 : by0) + 0.05, a1 = (face === 's' ? bx1 : by1) - 0.05;
      painel(eixo, c, a0, a1, z + 0.1, z + h - 0.06, corTela);
      if (tela === 'trabalhando') for (let k = 0; k < 3; k++) {
        const l1 = a0 + 0.04, l2 = a0 + 0.04 + (a1 - a0 - 0.08) * [0.8, 0.5, 0.65][k];
        painel(eixo, c + 0.001, l1, l2, z + h - 0.13 - k * 0.07, z + h - 0.11 - k * 0.07, '#7CE39A');
      } else if (tela === 'pendencia') painel(eixo, c + 0.001, (a0 + a1) / 2 - 0.02, (a0 + a1) / 2 + 0.02, z + 0.16, z + h - 0.12, '#FF8A80');
      else if (tela === 'concluido') painel(eixo, c + 0.001, a0 + 0.08, a1 - 0.08, z + 0.2, z + 0.24, '#A8F0BE');
      else if (tela) painel(eixo, c + 0.001, (a0 + a1) / 2 - 0.03, (a0 + a1) / 2 + 0.03, z + 0.2, z + 0.24, '#5FA8C8');
    } else { // costas do monitor: aberturas de ventilação
      const eixo = face === 'n' ? 'x' : 'y', c = face === 'n' ? by1 + 0.001 : bx1 + 0.001, a0 = face === 'n' ? bx0 : by0;
      for (let k = 0; k < 3; k++) painel(eixo, c, a0 + 0.1, a0 + 0.34, z + 0.14 + k * 0.07, z + 0.16 + k * 0.07, tom(COR.crt, 0.62));
    }
  }
  function teclado(x, y, z, alongX) {
    const [x0, y0, x1, y1] = alongX ? [x - 0.2, y - 0.07, x + 0.2, y + 0.07] : [x - 0.07, y - 0.2, x + 0.07, y + 0.2];
    caixa(x0, y0, x1, y1, z, z + 0.025, '#D6CDB8', { brilho: false, contorno: '#8C8370' });
  }
  function luminaria(x, y, z, cor = '#F2C14E') {
    caixa(x - 0.07, y - 0.07, x + 0.07, y + 0.07, z, z + 0.03, '#3A3A44', { brilho: false });
    caixa(x - 0.015, y - 0.015, x + 0.015, y + 0.015, z, z + 0.32, '#3A3A44', { brilho: false, contorno: false });
    cilindro(x, y, 0.11, z + 0.3, z + 0.42, cor, { topo: tom(cor, 1.1) });
  }
  function cadeiraEscritorio(x, y, face, cor = COR.cadeira) {
    item([x - 0.22, y - 0.22, x + 0.22, y + 0.22], 0, 0.5, () => {
      caixa(x - 0.2, y - 0.04, x + 0.2, y + 0.04, 0.02, 0.06, '#24262F', { brilho: false });
      caixa(x - 0.04, y - 0.2, x + 0.04, y + 0.2, 0.02, 0.06, '#24262F', { brilho: false });
      caixa(x - 0.03, y - 0.03, x + 0.03, y + 0.03, 0.06, 0.42, '#5B6070', { brilho: false, contorno: false });
      caixa(x - 0.22, y - 0.22, x + 0.22, y + 0.22, 0.42, 0.5, cor);
    });
    const e = { n: [x - 0.22, y + 0.16, x + 0.22, y + 0.24], s: [x - 0.22, y - 0.24, x + 0.22, y - 0.16], e: [x - 0.24, y - 0.22, x - 0.16, y + 0.22], w: [x + 0.16, y - 0.22, x + 0.24, y + 0.22] }[face];
    item(e, 0.5, 0.95, () => caixa(e[0], e[1], e[2], e[3], 0.5, 0.95, tom(cor, 1.08)));
  }
  function banqueta(x, y, cor = '#7A5236') {
    item([x - 0.18, y - 0.18, x + 0.18, y + 0.18], 0, 0.48, () => {
      caixa(x - 0.03, y - 0.03, x + 0.03, y + 0.03, 0, 0.42, '#3A3A44', { brilho: false, contorno: false });
      cilindro(x, y, 0.18, 0.42, 0.48, cor);
    });
  }
  function planta(x, y, esc = 1, tipo = 0) {
    item([x - 0.22 * esc, y - 0.22 * esc, x + 0.22 * esc, y + 0.22 * esc], 0, 1.2 * esc, () => {
      cilindro(x, y, 0.2 * esc, 0, 0.32 * esc, COR.vaso, { topo: '#4A3022' });
      const [cx, cy] = Pa(x, y, 0.32 * esc), u = K * esc;
      if (tipo === 1) { // folhas altas (espada)
        const fs = [[-0.32, 1.0], [-0.12, 1.25], [0.08, 1.15], [0.26, 0.95], [-0.02, 0.8]];
        for (const [dx, hh] of fs) { const x1 = cx + dx * u * 0.9, y1 = cy - hh * u * 1.1; poli([[cx - 2 + dx * 3, cy], [x1 - 1, y1], [x1 + 2, y1 + 2], [cx + 2 + dx * 3, cy]], COR.folha1); linha(cx + dx * 3, cy, x1, y1, COR.folha3); }
        return;
      }
      const bolas = [[0, -0.42, 0.42], [-0.3, -0.25, 0.32], [0.3, -0.27, 0.33], [-0.12, -0.7, 0.3], [0.18, -0.62, 0.3]];
      for (const [dx, dy, r] of bolas) disco(cx + dx * u, cy + dy * u, r * u + 1, r * u * 0.9 + 1, COR.folha3);
      for (const [dx, dy, r] of bolas) disco(cx + dx * u, cy + dy * u, r * u, r * u * 0.9, COR.folha1);
      for (const [dx, dy, r] of bolas) disco(cx + dx * u - r * u * 0.3, cy + dy * u - r * u * 0.3, r * u * 0.45, r * u * 0.38, COR.folha2);
    }, { sombra: true });
    obst(x - 0.22 * esc, y - 0.22 * esc, x + 0.22 * esc, y + 0.22 * esc);
  }
  function estante(x0, y0, x1, y1, h, frenteEixo, cor = COR.movelEsc) {
    item([x0, y0, x1, y1], 0, h, () => {
      caixa(x0, y0, x1, y1, 0, h, cor);
      const prateleiras = Math.max(2, Math.round(h / 0.42));
      for (let p = 0; p < prateleiras; p++) {
        const z0 = 0.08 + p * (h - 0.12) / prateleiras, z1 = z0 + (h - 0.12) / prateleiras - 0.06;
        if (frenteEixo === 'y') { // frente no plano x = x1
          painel('y', x1 + 0.001, y0 + 0.05, y1 - 0.05, z0, z1 + 0.04, tom(cor, 0.45));
          for (let a = y0 + 0.07, n = 0; a < y1 - 0.1; n++) { const w = 0.07 + ruido(n, p) * 0.06, alt = (z1 - z0) * (0.6 + ruido(p, n) * 0.4); painel('y', x1 + 0.002, a, Math.min(y1 - 0.06, a + w), z0, z0 + alt, LIVROS[(n * 3 + p) % LIVROS.length]); a += w + 0.012; }
        } else { // frente no plano y = y1
          painel('x', y1 + 0.001, x0 + 0.05, x1 - 0.05, z0, z1 + 0.04, tom(cor, 0.45));
          for (let a = x0 + 0.07, n = 0; a < x1 - 0.1; n++) { const w = 0.07 + ruido(n, p) * 0.06, alt = (z1 - z0) * (0.6 + ruido(p, n) * 0.4); painel('x', y1 + 0.002, a, Math.min(x1 - 0.06, a + w), z0, z0 + alt, LIVROS[(n * 5 + p) % LIVROS.length]); a += w + 0.012; }
        }
      }
    }, { sombra: true });
    obst(x0, y0, x1, y1);
  }
  function vidro(x0, y0, x1, y1, h = 2.2) { // parede de vidro em trechos de ~1 m (cada trecho é ordenado com as pessoas)
    const horiz = x1 - x0 > y1 - y0, a = horiz ? x0 : y0, b = horiz ? x1 : y1;
    for (let s = a; s < b - 0.01; s += 1.5) {
      const e = Math.min(b, s + 1.5), box = horiz ? [s, y0, e, y1] : [x0, s, x1, e];
      item(box, 0, h, () => {
        caixa(box[0], box[1], box[2], box[3], 0.0, h, COR.vidro, { alfa: 0.22, brilho: false, contorno: false });
        // brilhos diagonais no vidro
        const m = [[s + 0.35, s + 0.47], [s + 0.6, s + 0.66]];
        for (const [p, q] of m) {
          if (q > e) continue;
          const pts = horiz ? [Pa(p, y1, h - 0.3), Pa(q, y1, h - 0.3), Pa(q - 0.25, y1, 0.4), Pa(p - 0.25, y1, 0.4)] : [Pa(x1, p, h - 0.3), Pa(x1, q, h - 0.3), Pa(x1, q - 0.25, 0.4), Pa(x1, p - 0.25, 0.4)];
          poli(pts, '#FFFFFF', 0.18);
        }
        const aco = '#566079';
        caixa(box[0], box[1], box[2], box[3], h - 0.05, h, aco, { brilho: false });
        caixa(box[0], box[1], box[2], box[3], 0, 0.05, aco, { brilho: false });
        if (horiz) caixa(s, y0, s + 0.04, y1, 0, h, aco, { brilho: false }); else caixa(x0, s, x1, s + 0.04, 0, h, aco, { brilho: false });
      });
      obst(box[0], box[1], box[2], box[3]);
    }
  }

  // ------------------------------------------------------------------ montagem do cenário (depende dos dados)
  function montarItens(d) {
    ITENS = []; BLOQUEIOS.length = 0; ALVOS_ARTE = [];
    const est = d.estacoes || {};

    // ---- sala do gerente (vidro)
    vidro(0, 4.96, 3.9, 5.04); vidro(5.1, 4.96, 6.0, 5.04); vidro(5.96, 0, 6.04, 5.04);
    item([1.5, 2.1, 4.5, 2.9], 0, 0.78, () => {
      caixa(1.5, 2.1, 4.5, 2.9, 0, 0.72, COR.movelEsc);
      painel('x', 2.901, 1.7, 4.3, 0.12, 0.6, tom(COR.movelEsc, 0.9));
      caixa(1.45, 2.05, 4.55, 2.95, 0.72, 0.78, tom(COR.movelEsc, 1.2));
    }, { sombra: true });
    obst(1.5, 2.1, 4.5, 2.9);
    item([1.75, 2.2, 2.25, 2.62], 0.78, 1.2, () => crt(2.0, 2.4, 0.78, 'n', d.atlas_trabalhando ? 'trabalhando' : 'ligado'));
    item([3.75, 2.2, 4.25, 2.62], 0.78, 1.2, () => crt(4.0, 2.4, 0.78, 'n', d.atlas_trabalhando ? 'trabalhando' : 'ligado'));
    item([2.6, 2.3, 3.4, 2.6], 0.78, 0.82, () => { caixa(2.65, 2.32, 3.0, 2.58, 0.78, 0.8, '#F4F1E8', { brilho: false }); caixa(3.05, 2.35, 3.35, 2.55, 0.78, 0.81, '#E9E3D3', { brilho: false }); });
    item([4.3, 2.75, 4.48, 2.9], 0.78, 1.25, () => luminaria(4.39, 2.82, 0.78, '#3E8E5B'));
    cadeiraEscritorio(3.0, 1.4, 's', '#5A3B2E');
    estante(0.0, 0.5, 0.42, 2.3, 2.0, 'y');
    planta(0.65, 4.35, 1.25);
    planta(5.45, 4.35, 0.9, 1);

    // ---- quadro de vagas e janela
    planta(6.6, 0.55, 1.0, 1);
    estante(15.2, 0, 16.8, 0.42, 1.85, 'x');
    item([17.0, 0.3, 17.36, 0.66], 0, 1.35, () => { // bebedouro
      caixa(17.0, 0.3, 17.36, 0.66, 0, 0.95, '#E8ECF0');
      cilindro(17.18, 0.48, 0.15, 0.95, 1.35, '#7EC3EB', { topo: '#A9DAF5' });
    }, { sombra: true });
    obst(17.0, 0.3, 17.36, 0.66);

    // ---- copa
    item([17.6, 0, 22.2, 0.7], 0, 1.05, () => {
      caixa(17.6, 0, 22.2, 0.68, 0, 0.88, '#E9E4DA');
      for (let x = 17.6; x < 22.1; x += 0.9) painel('x', 0.681, x + 0.06, x + 0.84, 0.08, 0.8, '#DCD5C7');
      caixa(17.55, 0, 22.25, 0.72, 0.88, 0.93, '#4C4A55');
      caixa(19.6, 0.1, 20.4, 0.55, 0.925, 0.93, '#9AA3AD', { brilho: false }); // pia
      caixa(18.0, 0.1, 18.45, 0.5, 0.93, 1.3, '#2E2E36'); caixa(18.1, 0.42, 18.35, 0.5, 0.95, 1.02, '#C0504D', { contorno: false }); // cafeteira
      caixa(21.2, 0.08, 21.95, 0.55, 0.93, 1.25, '#C9CED6'); painel('x', 0.551, 21.3, 21.75, 0.98, 1.2, '#2B3640'); // micro-ondas
    }, { sombra: true });
    obst(17.6, 0, 22.2, 0.7);
    item([22.6, 0, 23.6, 0.8], 0, 2.0, () => {
      caixa(22.6, 0, 23.6, 0.8, 0, 2.0, '#CED4DC');
      painel('x', 0.801, 22.62, 23.58, 1.28, 1.3, '#8E98A4');
      painel('x', 0.802, 23.4, 23.45, 1.4, 1.8, '#6C7480'); painel('x', 0.802, 23.4, 23.45, 0.6, 1.1, '#6C7480');
    }, { sombra: true });
    obst(22.6, 0, 23.6, 0.8);
    for (const [tx, ty, base] of [[19.0, 2.9, 0], [21.8, 2.9, 3]]) {
      item([tx - 0.45, ty - 0.45, tx + 0.45, ty + 0.45], 0, 0.78, () => {
        caixa(tx - 0.05, ty - 0.05, tx + 0.05, ty + 0.05, 0, 0.72, '#3A3A44', { brilho: false });
        cilindro(tx, ty, 0.45, 0.72, 0.78, '#F1ECE2');
        cilindro(tx + 0.12, ty - 0.08, 0.06, 0.78, 0.85, '#FFFFFF', { contorno: '#9A8F80' });
      }, { sombra: true });
      obst(tx - 0.45, ty - 0.45, tx + 0.45, ty + 0.45);
      banqueta(...POS_DESCANSO['comer' + base]); banqueta(...POS_DESCANSO['comer' + (base + 1)]); banqueta(...POS_DESCANSO['comer' + (base + 2)]);
    }

    // ---- recepção
    item([2.2, 9.6, 2.8, 12.6], 0, 1.06, () => {
      caixa(2.25, 9.6, 2.8, 12.6, 0, 1.0, '#7B4A2A');
      for (let y = 9.75; y < 12.5; y += 0.6) painel('y', 2.801, y, y + 0.45, 0.15, 0.85, '#6A3F23');
      caixa(2.15, 9.55, 2.85, 12.65, 1.0, 1.06, '#C9A27A');
    }, { sombra: true });
    obst(2.2, 9.6, 2.8, 12.6);
    item([2.25, 10.75, 2.65, 11.25], 1.06, 1.5, () => crt(2.45, 11.0, 1.06, 'w', 'ligado'));
    item([2.3, 11.5, 2.55, 11.9], 1.06, 1.1, () => teclado(2.42, 11.6, 1.06, false));
    const fila = Math.min(16, (d.fila_triagem || 0) + (d.fila_analise || 0));
    item([2.3, 12.0, 2.75, 12.5], 1.06, 1.1 + fila * 0.025, () => {
      caixa(2.3, 12.0, 2.75, 12.5, 1.06, 1.1, '#5B5F6E', { brilho: false });
      for (let i = 0; i < fila; i++) caixa(2.34 + (i % 2) * 0.01, 12.04, 2.71, 12.46, 1.1 + i * 0.025, 1.12 + i * 0.025, i % 2 ? '#FFFFFF' : '#F1EEE6', { brilho: false, contorno: '#B8B2A4' });
    });
    ALVOS_ARTE.push({ tipo: 'bandeja', pts: [Pa(2.3, 12.0, 1.2 + fila * 0.025), Pa(2.75, 12.0, 1.2 + fila * 0.025), Pa(2.75, 12.5, 1.06), Pa(2.3, 12.5, 1.06)] });
    cadeiraEscritorio(1.3, 11.1, 'e', '#7A4F7E');
    banqueta(0.55, 15.35, '#4F81BD'); banqueta(1.35, 15.35, '#4F81BD');
    planta(0.5, 8.9, 1.15); planta(0.55, 12.6, 0.85, 1); planta(3.4, 15.45, 1.0);

    // ---- ilhas de trabalho (12 estações)
    ILHAS.forEach((x0, k) => {
      mesaTrabalho(x0, 6.3, x0 + 2.8, 7.1); mesaTrabalho(x0, 7.1, x0 + 2.8, 7.9);
      for (let i = k * 4; i < k * 4 + 4; i++) {
        const e = ESTACOES[i], fA = e.fila === 'A', tela = est[i] || null;
        const my = fA ? 6.86 : 7.34;
        item([e.x - 0.22, my - 0.2, e.x + 0.22, my + 0.2], 0.76, 1.16, () => crt(e.x, my, 0.76, fA ? 'n' : 's', tela));
        item([e.x - 0.2, (fA ? 6.4 : 7.66) - 0.07, e.x + 0.2, (fA ? 6.4 : 7.66) + 0.07], 0.76, 0.79, () => teclado(e.x, fA ? 6.4 : 7.66, 0.76, true));
        if (tela === 'trabalhando') item([e.x + 0.3, my - 0.15, e.x + 0.6, my + 0.15], 0.76, 0.8, () => caixa(e.x + 0.32, my - 0.14, e.x + 0.58, my + 0.14, 0.76, 0.79, '#FFFFFF', { brilho: false, contorno: '#B8B2A4' }));
        cadeiraEscritorio(e.x, e.y, e.face);
      }
      item([x0 + 0.08, 6.32, x0 + 0.26, 6.5], 0.76, 1.2, () => luminaria(x0 + 0.17, 6.41, 0.76));
      item([x0 + 2.54, 7.5, x0 + 2.72, 7.68], 0.76, 1.2, () => luminaria(x0 + 2.63, 7.59, 0.76));
      item([x0 + 1.25, 6.95, x0 + 1.55, 7.25], 0.76, 0.95, () => cilindro(x0 + 1.4, 7.1, 0.12, 0.76, 0.9, '#B95A2E', { topo: COR.folha1 }));
    });
    planta(10.3, 5.0, 0.9); planta(14.9, 5.0, 0.9); planta(19.4, 6.0, 1.2, 1);

    // ---- canto de leitura
    estante(6.8, 11.6, 10.4, 12.0, 1.0, 'x', COR.movel);
    for (const n of ['leitura0', 'leitura1']) {
      const [x, y] = POS_DESCANSO[n];
      item([x - 0.35, y - 0.35, x + 0.35, y + 0.35], 0, 0.45, () => caixa(x - 0.33, y - 0.3, x + 0.33, y + 0.33, 0, 0.42, '#C2703F'));
      item([x - 0.35, y - 0.42, x + 0.35, y - 0.28], 0.42, 0.95, () => caixa(x - 0.35, y - 0.42, x + 0.35, y - 0.28, 0, 0.95, tom('#C2703F', 0.92)));
      item([x + 0.28, y - 0.35, x + 0.4, y + 0.35], 0, 0.62, () => caixa(x + 0.28, y - 0.35, x + 0.4, y + 0.35, 0, 0.62, tom('#C2703F', 0.88)));
      item([x - 0.4, y - 0.35, x - 0.28, y + 0.35], 0, 0.62, () => caixa(x - 0.4, y - 0.35, x - 0.28, y + 0.35, 0, 0.62, tom('#C2703F', 0.88)));
      obst(x - 0.4, y - 0.42, x + 0.4, y - 0.28); obst(x + 0.28, y - 0.35, x + 0.4, y + 0.35); obst(x - 0.4, y - 0.35, x - 0.28, y + 0.35);
    }
    item([8.45, 12.95, 8.75, 13.25], 0, 0.55, () => { cilindro(8.6, 13.1, 0.16, 0.48, 0.54, COR.movel); caixa(8.57, 13.07, 8.63, 13.13, 0, 0.48, COR.movelEsc, { brilho: false }); });
    item([10.3, 12.5, 10.55, 12.75], 0, 1.65, () => {
      caixa(10.36, 12.56, 10.49, 12.69, 0, 0.04, '#3A3A44', { brilho: false });
      caixa(10.41, 12.61, 10.44, 12.64, 0, 1.4, '#3A3A44', { brilho: false, contorno: false });
      cilindro(10.425, 12.625, 0.16, 1.38, 1.62, '#F6E3B4', { topo: '#FFF1CC' });
    });
    obst(10.3, 12.5, 10.55, 12.75);
    planta(11.3, 12.1, 1.25); planta(13.2, 15.4, 1.0, 1);
    item([12.05, 13.45, 13.15, 14.55], 0, 0.78, () => {
      caixa(12.55, 13.95, 12.65, 14.05, 0, 0.72, '#3A3A44', { brilho: false });
      cilindro(12.6, 14.0, 0.55, 0.72, 0.78, COR.movel);
      caixa(12.35, 13.8, 12.7, 14.05, 0.78, 0.8, '#F4F1E8', { brilho: false, contorno: '#B8B2A4' });
    }, { sombra: true });
    obst(12.05, 13.45, 13.15, 14.55);
    cadeiraEscritorio(12.6, 13.15, 's', '#4B5068'); cadeiraEscritorio(12.6, 14.85, 'n', '#4B5068');
    cadeiraEscritorio(11.75, 14.0, 'e', '#4B5068'); cadeiraEscritorio(13.45, 14.0, 'w', '#4B5068');

    // ---- área de descanso
    item([15.0, 10.6, 17.6, 11.0], 0, 0.5, () => caixa(15.0, 10.6, 17.6, 11.0, 0, 0.48, COR.movelEsc), { sombra: true });
    item([15.4, 10.68, 17.2, 10.82], 0.5, 1.32, () => {
      caixa(16.2, 10.7, 16.4, 10.8, 0.5, 0.6, '#1E2026', { brilho: false });
      caixa(15.4, 10.68, 17.2, 10.82, 0.6, 1.32, '#1E2026');
      painel('x', 10.821, 15.46, 17.14, 0.65, 1.27, d.tv_ligada ? '#5C9BD8' : '#2A2F3A');
      if (d.tv_ligada) { painel('x', 10.822, 15.7, 16.5, 0.7, 0.92, '#8DC97A'); painel('x', 10.822, 16.6, 16.95, 1.0, 1.2, '#FFE08A'); painel('x', 10.822, 15.55, 16.2, 1.1, 1.17, '#E8F4FF'); }
    });
    obst(15.0, 10.6, 17.6, 11.0);
    item([15.5, 12.1, 17.1, 12.6], 0, 0.42, () => { caixa(15.5, 12.1, 17.1, 12.6, 0.36, 0.42, COR.movel); for (const [x, y] of [[15.55, 12.15], [17.0, 12.15], [15.55, 12.5], [17.0, 12.5]]) caixa(x, y, x + 0.05, y + 0.05, 0, 0.36, COR.movelEsc, { brilho: false }); }, { sombra: true });
    obst(15.5, 12.1, 17.1, 12.6);
    const SOFA = '#A33F3F';
    item([14.9, 13.4, 17.7, 14.1], 0, 0.42, () => caixa(15.1, 13.4, 17.5, 14.1, 0, 0.42, SOFA), { sombra: true });
    item([14.9, 14.0, 17.7, 14.3], 0, 0.9, () => caixa(14.9, 14.0, 17.7, 14.3, 0, 0.9, tom(SOFA, 0.92)));
    item([14.9, 13.4, 15.1, 14.0], 0, 0.62, () => caixa(14.9, 13.4, 15.1, 14.0, 0, 0.62, tom(SOFA, 0.95)));
    item([17.5, 13.4, 17.7, 14.0], 0, 0.62, () => caixa(17.5, 13.4, 17.7, 14.0, 0, 0.62, tom(SOFA, 0.95)));
    obst(14.9, 13.95, 17.7, 14.3); obst(14.9, 13.4, 15.1, 14.3); obst(17.5, 13.4, 17.7, 14.3);
    item([19.6, 11.6, 22.4, 13.0], 0, 0.95, () => {
      for (const [x, y] of [[19.8, 11.75], [22.15, 11.75], [19.8, 12.8], [22.15, 12.8]]) caixa(x, y, x + 0.07, y + 0.07, 0, 0.72, '#30343F', { brilho: false });
      caixa(19.6, 11.6, 22.4, 13.0, 0.72, 0.78, '#2E7D5B', { topo: '#33895F' });
      piso(19.62, 12.28, 22.38, 12.32, '#EAF5EF', undefined, 0.781);
      piso(19.62, 11.62, 22.38, 11.65, '#EAF5EF', undefined, 0.781); piso(19.62, 12.95, 22.38, 12.98, '#EAF5EF', undefined, 0.781);
      caixa(20.98, 11.5, 21.02, 13.1, 0.78, 0.92, '#F4F4F4', { brilho: false, contorno: '#9DA3AE' });
    }, { sombra: true });
    obst(19.6, 11.6, 22.4, 13.0);
    for (const [n, cor] of [['pufe0', '#E07B39'], ['pufe1', '#4F81BD'], ['pufe2', '#9BBB59']]) {
      const [x, y] = POS_DESCANSO[n];
      item([x - 0.4, y - 0.4, x + 0.4, y + 0.4], 0, 0.4, () => {
        const [cx, cy] = Pa(x, y, 0.16);
        disco(cx, cy, 13, 8, tom(cor, 0.4)); disco(cx, cy - 1, 12, 7, tom(cor, 0.85)); disco(cx - 2, cy - 3, 9, 5, cor); disco(cx - 4, cy - 4, 4, 2, tom(cor, 1.2));
      }, { sombra: true });
    }
    planta(23.5, 10.4, 1.1); planta(23.5, 15.5, 1.2, 1); planta(14.0, 15.5, 0.95);

    // pessoas não atravessam móveis (folga de 10 cm); só calculado uma vez
    if (!BLOQ.preenchido) { for (const b of BLOQUEIOS) bloquear(...b); BLOQ.preenchido = true; }
  }

  // ------------------------------------------------------------------ camada fixa: piso, paredes, janelas, luz
  function desenharBase(d) {
    // sombra do bloco do escritório e laje
    poli([Pa(0.5, 0.5, -0.3), Pa(LARG + 0.5, 0.5, -0.3), Pa(LARG + 0.5, PROF + 0.5, -0.3), Pa(0.5, PROF + 0.5, -0.3)], '#000000', 0.28);
    caixa(0, 0, LARG, PROF, -0.3, 0, COR.piso1, { esq: COR.lajeE, dir: COR.lajeD, contorno: '#2A1C12', brilho: false });
    // piso: ladrilho xadrez quente (área geral), tábuas (sala do gerente), ladrilho claro (copa)
    for (let j = 0; j < PROF; j++) for (let i = 0; i < LARG; i++) {
      const base = (i + j) % 2 ? COR.piso1 : COR.piso2, v = 0.97 + ruido(i, j) * 0.05;
      piso(i, j, i + 1, j + 1, tom(base, v));
    }
    piso(0, 0, 6, 5, COR.madeira);
    for (let y = 0.25; y < 5; y += 0.25) { const a = Pa(0, y), b = Pa(6, y); linha(a[0], a[1], b[0], b[1], COR.tabua); }
    for (let r = 0; r < 20; r++) { const y = r * 0.25, x = ((r * 7) % 5) * 0.9 + 0.4; if (x < 6) { const a = Pa(x, y), b = Pa(x, y + 0.25); linha(a[0], a[1], b[0], b[1], COR.tabua); } }
    for (let j = 0; j < 9; j++) for (let i = 0; i < 14; i++) piso(17.2 + i * 0.5, j * 0.5, 17.2 + i * 0.5 + 0.5, j * 0.5 + 0.5, (i + j) % 2 ? COR.cozinha1 : COR.cozinha2);
    // tapetes
    const tapete = (x0, y0, x1, y1, cor, borda) => { piso(x0, y0, x1, y1, borda); piso(x0 + 0.15, y0 + 0.15, x1 - 0.15, y1 - 0.15, cor); piso(x0 + 0.3, y0 + 0.3, x1 - 0.3, y1 - 0.3, tom(cor, 1.12)); piso(x0 + 0.45, y0 + 0.45, x1 - 0.45, y1 - 0.45, cor); };
    tapete(1.0, 0.9, 5.0, 4.3, '#7A2E2E', '#C9A15B');
    tapete(6.9, 12.3, 10.3, 14.5, '#3E5C76', '#D9C08A');
    tapete(14.4, 11.2, 18.2, 14.9, '#2F6E6A', '#E8D6A8');
    piso(0.05, 13.25, 1.1, 14.55, '#5A4A3A'); piso(0.15, 13.35, 1.0, 14.45, '#6B5A48');

    // paredes do fundo (faces internas, topo e acabamentos)
    caixa(-0.22, -0.22, LARG, 0, 0, ALT_PAREDE, COR.paredeD, { esq: COR.paredeD, dir: tom(COR.paredeD, 0.8), topo: COR.paredeTopo, contorno: '#151A30', brilho: false });
    caixa(-0.22, 0, 0, PROF, 0, ALT_PAREDE, COR.paredeE, { esq: tom(COR.paredeE, 0.85), dir: COR.paredeE, topo: COR.paredeTopo, contorno: '#151A30', brilho: false });
    painel('x', 0.001, 0, LARG, 0, 0.12, COR.rodape); painel('y', 0.001, 0, PROF, 0, 0.12, COR.rodape);
    painel('x', 0.001, 0, LARG, ALT_PAREDE - 0.1, ALT_PAREDE, COR.moldura); painel('y', 0.001, 0, PROF, ALT_PAREDE - 0.1, ALT_PAREDE, COR.moldura);
    for (let x = 1; x < LARG; x += 1) { const a = Pa(x, 0.002, 0.12), b = Pa(x, 0.002, ALT_PAREDE - 0.1); linha(a[0], a[1], b[0], b[1], tom(COR.paredeD, 0.93)); }
    for (let y = 1; y < PROF; y += 1) { const a = Pa(0.002, y, 0.12), b = Pa(0.002, y, ALT_PAREDE - 0.1); linha(a[0], a[1], b[0], b[1], tom(COR.paredeE, 0.92)); }

    // janelas com feixes de sol
    const janela = (eixo, a0, a1) => {
      const z0 = 0.9, z1 = 2.2, f = '#C99A6B';
      painel(eixo, 0.003, a0 - 0.08, a1 + 0.08, z0 - 0.08, z1 + 0.08, f);
      painel(eixo, 0.004, a0, a1, z0, z1, '#8FC3E3');
      painel(eixo, 0.005, a0, a1, z0 + (z1 - z0) * 0.55, z1, '#B5DCF2');
      painel(eixo, 0.006, a0 + (a1 - a0) * 0.62, a1, z0 + 0.15, z0 + 0.45, '#D8EEF9');
      painel(eixo, 0.007, (a0 + a1) / 2 - 0.03, (a0 + a1) / 2 + 0.03, z0, z1, f);
      painel(eixo, 0.007, a0, a1, (z0 + z1) / 2 - 0.03, (z0 + z1) / 2 + 0.03, f);
      const sol = '#FFE6A6';
      const chao = eixo === 'y' ? [[1.64, a0 + 0.57], [4.0, a0 + 1.4], [4.0, a1 + 1.4], [1.64, a1 + 0.57]] : [[a0 + 0.57, 1.64], [a0 + 1.4, 4.0], [a1 + 1.4, 4.0], [a1 + 0.57, 1.64]];
      poli(chao.map(([x, y]) => Pa(x, y)), sol, 0.16);
      const jan = eixo === 'y' ? [Pa(0, a0, z1), Pa(0, a1, z1), Pa(0, a1, z0), Pa(0, a0, z0)] : [Pa(a0, 0, z1), Pa(a1, 0, z1), Pa(a1, 0, z0), Pa(a0, 0, z0)];
      const todos = [...jan, ...chao.map(([x, y]) => Pa(x, y))];
      poli(envoltoria(todos), sol, 0.07);
    };
    janela('y', 6.0, 8.2); janela('x', 12.4, 14.8); janela('x', 4.9, 5.75);

    // porta de entrada (parede esquerda)
    painel('y', 0.003, 13.15, 14.65, 0, 2.18, '#3B2416');
    painel('y', 0.004, 13.25, 14.55, 0, 2.1, '#7A4A2A');
    painel('y', 0.005, 13.38, 14.42, 1.15, 1.95, '#86532F'); painel('y', 0.005, 13.38, 14.42, 0.15, 1.0, '#86532F');
    painel('y', 0.006, 14.3, 14.4, 1.0, 1.1, '#F2C14E');
    // quadro com paisagem (parede esquerda)
    painel('y', 0.003, 9.35, 10.65, 1.05, 1.95, '#6B4A2A');
    painel('y', 0.004, 9.45, 10.55, 1.13, 1.87, '#9FD3F0');
    painel('y', 0.005, 9.45, 10.55, 1.13, 1.4, '#5DA35E'); painel('y', 0.006, 9.75, 10.4, 1.13, 1.52, '#4A8F52');
    painel('y', 0.006, 9.6, 9.75, 1.62, 1.75, '#FFE08A');

    // quadro de vagas: um cartão por vaga, colorido pela situação, com barra de progresso
    painel('x', 0.003, 6.95, 11.65, 0.9, 2.2, '#8A8F99');
    painel('x', 0.004, 7.02, 11.58, 0.97, 2.13, '#F4F2EA');
    const vs = (d.vagas || []).filter(v => v.status !== 'cancelada').slice(0, 12);
    vs.forEach((v, i) => {
      const col = i % 6, lin = Math.floor(i / 6), x0 = 7.14 + col * 0.74, z1 = 2.02 - lin * 0.56;
      const cor = v.status !== 'aberta' ? '#AEB4BF' : v.sem_responsavel ? '#E5655A' : v.pendentes ? '#F2B53D' : '#5DBB7E';
      painel('x', 0.005, x0, x0 + 0.62, z1 - 0.46, z1, '#FFFFFF');
      painel('x', 0.006, x0, x0 + 0.62, z1 - 0.1, z1, cor);
      for (let k = 0; k < 2; k++) painel('x', 0.006, x0 + 0.05, x0 + 0.05 + (0.44 - k * 0.14), z1 - 0.19 - k * 0.08, z1 - 0.16 - k * 0.08, '#C9CDD6');
      const prog = v.recebidos ? v.analisados / v.recebidos : 0;
      painel('x', 0.006, x0 + 0.05, x0 + 0.57, z1 - 0.38, z1 - 0.34, '#E3E5EA');
      if (prog > 0) painel('x', 0.007, x0 + 0.05, x0 + 0.05 + 0.52 * prog, z1 - 0.38, z1 - 0.34, '#3FB56B');
    });
    if (!vs.length) painel('x', 0.005, 8.6, 10.0, 1.4, 1.48, '#D9DCE3');
    ALVOS_ARTE.push({ tipo: 'vagas', pts: [Pa(6.95, 0, 2.2), Pa(11.65, 0, 2.2), Pa(11.65, 0, 0.9), Pa(6.95, 0, 0.9)] });

    // quadro geral do gerente (cortiça com notas: vagas abertas, em análise, pendências)
    painel('x', 0.003, 1.35, 4.65, 1.1, 2.1, '#6B4A2A');
    painel('x', 0.004, 1.45, 4.55, 1.18, 2.02, '#C49563');
    const q = d.quadro || {}, notas = [];
    for (let i = 0; i < Math.min(5, q.vagas_abertas || 0); i++) notas.push('#BFE8C9');
    for (let i = 0; i < Math.min(4, q.vagas_em_processamento || 0); i++) notas.push('#FFE69A');
    for (let i = 0; i < Math.min(4, q.pendencias || 0); i++) notas.push('#F6B1A8');
    notas.forEach((cor, i) => {
      const x0 = 1.6 + (i % 7) * 0.42, z1 = 1.92 - Math.floor(i / 7) * 0.4;
      painel('x', 0.005, x0, x0 + 0.32, z1 - 0.3, z1, cor); painel('x', 0.006, x0 + 0.13, x0 + 0.19, z1 - 0.05, z1, '#D9534F');
    });
    ALVOS_ARTE.push({ tipo: 'quadro', pts: [Pa(1.35, 0, 2.1), Pa(4.65, 0, 2.1), Pa(4.65, 0, 1.1), Pa(1.35, 0, 1.1)] });

    // armários da copa (presos à parede) e relógio com a hora real
    caixa(17.6, 0, 22.2, 0.38, 1.55, 2.2, '#E9E4DA', { contorno: '#7D766A' });
    for (let x = 17.6; x < 22.1; x += 1.15) { painel('x', 0.381, x + 0.06, x + 1.09, 1.6, 2.15, '#DDD6C8'); painel('x', 0.382, x + 0.5, x + 0.6, 1.62, 1.68, '#8C8578'); }
    const agora = new Date(), [rx, rz] = [16.0, 2.2];
    const circ = (r, cor) => { const pts = []; for (let k = 0; k < 20; k++) { const t = k / 20 * Math.PI * 2; pts.push(Pa(rx + Math.cos(t) * r, 0.005, rz + Math.sin(t) * r)); } poli(pts, cor); };
    circ(0.17, '#2B2F3D'); circ(0.14, '#F4F2EA');
    const ponteiro = (ang, l, cor) => { const a = Pa(rx, 0.006, rz), b = Pa(rx + Math.sin(ang) * l, 0.006, rz + Math.cos(ang) * l); linha(a[0], a[1], b[0], b[1], cor); };
    ponteiro((agora.getHours() % 12 + agora.getMinutes() / 60) / 12 * Math.PI * 2, 0.08, '#2B2F3D');
    ponteiro(agora.getMinutes() / 60 * Math.PI * 2, 0.12, '#C0504D');

    // sombras de contato dos móveis
    for (const it of ITENS) if (it.sombra) { const [x0, y0, x1, y1] = it.box; sombrear([Pa(x0 + 0.06, y0 + 0.06), Pa(x1 + 0.1, y0 + 0.06), Pa(x1 + 0.1, y1 + 0.1), Pa(x0 + 0.06, y1 + 0.1)], 0.82); }
    // luz das luminárias sobre as mesas
    ILHAS.forEach(x0 => { brilhoChao(x0 + 0.4, 6.7, 0.9, '#FFD98A', 0.1); brilhoChao(x0 + 2.4, 7.4, 0.9, '#FFD98A', 0.1); });
    brilhoChao(4.2, 3.2, 1.0, '#FFE2A6', 0.09); brilhoChao(10.4, 13.0, 1.3, '#FFE2A6', 0.12);
  }
  function envoltoria(pts) { // casco convexo (feixes de luz)
    const p = pts.map(([x, y]) => [x, y]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    const cruz = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const inf = [], sup = [];
    for (const q of p) { while (inf.length >= 2 && cruz(inf[inf.length - 2], inf[inf.length - 1], q) <= 0) inf.pop(); inf.push(q); }
    for (const q of p.slice().reverse()) { while (sup.length >= 2 && cruz(sup[sup.length - 2], sup[sup.length - 1], q) <= 0) sup.pop(); sup.push(q); }
    return inf.slice(0, -1).concat(sup.slice(0, -1));
  }

  // ------------------------------------------------------------------ sprites dos móveis e ordem de desenho
  let base = new Uint32Array(AW * AH);
  const rascunho = new Uint32Array(AW * AH);
  function construirSprites() {
    for (const it of ITENS) {
      alvo(rascunho, AW, AH);
      it.f();
      const [x0, y0, x1, y1] = A.sujo;
      if (x1 < x0) { it.spr = null; continue; }
      const w = x1 - x0 + 1, h = y1 - y0 + 1, buf = new Uint32Array(w * h);
      for (let y = 0; y < h; y++) { const o = (y0 + y) * AW + x0; buf.set(rascunho.subarray(o, o + w), y * w); rascunho.fill(0, o, o + w); }
      it.spr = { buf, w, h, ox: x0, oy: y0 };
      it.tela = [x0, y0, x0 + w - 1, y0 + h - 1];
    }
    ordenarItens();
  }
  const E = 1e-6;
  // -1: a desenha antes de b; 1: b antes de a; 0: indiferente
  function ordem(a, b) {
    const [ax0, ay0, ax1, ay1] = a.box, [bx0, by0, bx1, by1] = b.box;
    const ax = ax1 <= bx0 + E, bx = bx1 <= ax0 + E, ay = ay1 <= by0 + E, by = by1 <= ay0 + E;
    if ((ax || ay) && !(bx || by)) return -1;
    if ((bx || by) && !(ax || ay)) return 1;
    if (ax || ay || bx || by) return 0; // separados na diagonal: não se cobrem na tela
    if (a.z1 <= b.z0 + E) return -1;   // um em cima do outro (monitor sobre a mesa)
    if (b.z1 <= a.z0 + E) return 1;
    return (ax0 + ax1 + ay0 + ay1) - (bx0 + bx1 + by0 + by1) <= 0 ? -1 : 1;
  }
  const cruza = (s, t) => s[0] <= t[2] && t[0] <= s[2] && s[1] <= t[3] && t[1] <= s[3];
  function ordenarItens() {
    const its = ITENS.filter(i => i.spr), n = its.length, antes = Array.from({ length: n }, () => []), grau = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      if (!cruza(its[i].tela, its[j].tela)) continue;
      const o = ordem(its[i], its[j]);
      if (o < 0) { antes[i].push(j); grau[j]++; } else if (o > 0) { antes[j].push(i); grau[i]++; }
    }
    const prof = i => its[i].box[0] + its[i].box[2] + its[i].box[1] + its[i].box[3];
    const prontos = [], feitos = new Uint8Array(n), saida = [];
    for (let i = 0; i < n; i++) if (!grau[i]) prontos.push(i);
    while (saida.length < n) {
      if (!prontos.length) { let m = -1; for (let i = 0; i < n; i++) if (!feitos[i] && (m < 0 || prof(i) < prof(m))) m = i; prontos.push(m); grau[m] = 0; } // ciclo: quebra pelo mais ao fundo
      prontos.sort((a, b) => prof(a) - prof(b));
      const i = prontos.shift();
      if (feitos[i]) continue;
      feitos[i] = 1; saida.push(its[i]);
      for (const j of antes[i]) if (!feitos[j] && --grau[j] === 0) prontos.push(j);
    }
    ITENS = saida;
  }

  // ------------------------------------------------------------------ pessoas (sprites 24 × 40, pés em (12, 38))
  const CALCAS = ['#2F3B52', '#3A3A3A', '#5A4632', '#4B5D7A', '#6B6F76', '#2E4A3E', '#283447'];
  const SAPATOS = ['#2B2B2B', '#EDEDED', '#5A3E2B', '#30405F'];
  const SW = 24, SH = 40, PX = 12, PY = 38;
  const SPRITES = new Map();
  function sprite(ap, vista, pose, quadro, humor, objeto) {
    const chave = [ap.pele, ap.cabelo, ap.estilo, ap.camisa, ap.oculos, ap.barba, ap.calca, ap.sapato, vista, pose, quadro, humor, objeto].join('|');
    let s = SPRITES.get(chave);
    if (!s) { s = gerarSprite(ap, vista, pose, quadro, humor, objeto); SPRITES.set(chave, s); }
    return s;
  }
  function gerarSprite(ap, vista, pose, quadro, humor, objeto) {
    const buf = new Uint32Array(SW * SH);
    alvo(buf, SW, SH);
    const pele = ap.pele || '#EEC49C', cab = ap.cabelo || '#3B2618', est = ap.estilo || 'curto', cam = ap.camisa || '#3B6EC8';
    const calca = ap.calca || escolha((ap.cabelo || '') + cam, CALCAS), sap = ap.sapato || escolha(cam + pele, SAPATOS);
    const peleS = tom(pele, 0.84), cabS = tom(cab, 0.72), cabC = tom(cab, 1.4), camS = tom(cam, 0.76), camC = tom(cam, 1.16), calcaS = tom(calca, 0.74);
    const r = (x, y, w, h, cor) => ret(x, y, w, h, cor), p = (x, y, cor) => ponto(x, y, cor);
    const frente = vista === 'frente', sentado = pose === 'senta' || pose === 'digita', anda = pose === 'anda';
    const bob = anda && (quadro === 1 || quadro === 3) ? -1 : 0;

    // pernas e sapatos
    if (!sentado) {
      const passo = anda ? (quadro === 1 ? 1 : quadro === 3 ? -1 : 0) : 0;
      const perna = (x, dx, lev, cor) => { r(x + dx, 30 + bob, 4, 36 - lev - (30 + bob), cor); r(x + dx - (frente ? 0 : 1), 36 - lev, 5, 2, sap); };
      perna(8, frente ? -passo : passo, passo > 0 ? 1 : 0, calca);
      perna(12, frente ? passo : -passo, passo < 0 ? 1 : 0, calcaS);
      r(12, 31 + bob, 1, 4, tom(calca, 0.6));
    } else if (frente) { // coxas para frente (direita da tela) e canelas para baixo
      r(9, 29, 9, 3, calcaS); r(15, 31, 3, 5, calcaS); r(15, 36, 5, 2, sap);
      r(8, 30, 9, 3, calca); r(13, 32, 3, 5, calca); r(13, 37, 5, 1, sap);
    }
    // tronco
    const t0 = 19 + bob;
    r(8, t0, 8, 1, cam); r(7, t0 + 1, 10, 10, cam);
    r(15, t0 + 1, 2, 10, camS); r(7, t0 + 1, 1, 9, camC); r(7, t0 + 10, 10, 1, camS);
    r(7, t0 + 10, 10, 1, tom(calca, 0.6)); // cinto
    if (frente) { r(11, t0, 2, 2, peleS); p(11, t0 + 2, peleS); } else r(10, t0, 4, 1, camS);
    // braços
    const digita = pose === 'digita', sobe = digita ? (quadro % 2) : 0;
    if (digita && frente) { // antebraços sobre a mesa, para a frente
      r(5, t0 + 1, 2, 4, cam); r(17, t0 + 1, 2, 4, camS);
      r(14, t0 + 6 - sobe, 6, 2, cam); r(20, t0 + 6 - sobe, 2, 2, pele);
      r(10, t0 + 7 + sobe - 1, 6, 2, camS); r(16, t0 + 6 + sobe, 2, 2, peleS);
    } else if (objeto === 'xicara' && sentado && frente) {
      r(5, t0 + 1, 2, 7, cam); r(5, t0 + 8, 2, 2, pele);
      r(17, t0 + 1, 2, 3, camS); r(14, t0 + 4, 4, 2, camS); r(14, t0 + 6, 2, 2, peleS);
      r(14, t0 + 2, 3, 3, '#FFFFFF'); r(14, t0 + 2, 3, 1, '#6B4128'); p(17, t0 + 3, '#D6D0C4');
    } else if (objeto === 'livro' && sentado && frente) {
      r(5, t0 + 1, 2, 4, cam); r(17, t0 + 1, 2, 4, camS);
      r(9, t0 + 4, 8, 5, '#3E5C9A'); r(10, t0 + 4, 3, 4, '#FFFFFF'); r(13, t0 + 4, 3, 4, '#F1EEE6'); p(9, t0 + 9, pele); p(16, t0 + 9, peleS);
    } else {
      const bal = anda ? (quadro === 1 ? 1 : quadro === 3 ? -1 : 0) : 0;
      r(5, t0 + 1 + bal, 2, 5, cam); r(5, t0 + 6 + bal, 2, 3, pele);
      r(17, t0 + 1 - bal, 2, 5, camS); r(17, t0 + 6 - bal, 2, 3, peleS);
      if (objeto === 'raquete' && frente) { r(18, t0 + 4, 4, 5, '#D64545'); r(19, t0 + 5, 2, 3, '#E86A5C'); r(19, t0 + 9, 2, 2, '#7A4A2A'); }
    }
    // cabeça
    const c0 = 6 + bob;
    const cabeca = (cor) => { r(8, c0, 8, 1, cor); r(7, c0 + 1, 10, 1, cor); r(6, c0 + 2, 12, 9, cor); r(7, c0 + 11, 10, 1, cor); };
    if (est === 'longo') { r(5, c0 + 3, 4, 13, cabS); if (!frente) r(15, c0 + 3, 4, 13, cabS); } // cabelo comprido atrás
    if (est === 'black') { disco(12, c0 + 5, 8.5, 7.5, cab); }
    cabeca(pele);
    r(16, c0 + 3, 2, 8, peleS); r(7, c0 + 11, 10, 1, peleS);
    if (frente) {
      // orelha à esquerda; olhos e boca deslocados para a direita (rosto em 3/4)
      r(8, c0 + 5, 2, 3, peleS); p(8, c0 + 6, tom(pele, 0.7));
      const olho = '#231A16', ey = c0 + 6;
      if (humor === 'calmo') { r(11, ey + 1, 2, 1, olho); r(14, ey + 1, 2, 1, olho); }
      else { r(12, ey, 1, 2, olho); r(15, ey, 1, 2, olho); }
      if (ap.oculos) { const k = '#1E1E26'; r(11, ey - 1, 3, 1, k); r(14, ey - 1, 3, 1, k); r(11, ey + 2, 3, 1, k); r(14, ey + 2, 3, 1, k); p(11, ey, k); p(11, ey + 1, k); p(17, ey, k); p(17, ey + 1, k); p(13, ey, k); }
      const by = c0 + 9, bc = '#7A3B2E';
      if (humor === 'preocupado') { p(12, by + 1, bc); r(13, by, 2, 1, bc); p(15, by + 1, bc); }
      else if (humor === 'focado') r(13, by, 2, 1, bc);
      else { p(12, by - 1, bc); r(13, by, 2, 1, bc); p(15, by - 1, bc); }
      if (humor === 'preocupado') { r(11, ey - 2, 2, 1, cabS); r(15, ey - 2, 2, 1, cabS); }
      if (ap.barba) { r(9, c0 + 8, 9, 3, cabS); r(10, c0 + 11, 7, 1, cabS); r(13, by, 2, 1, tom(pele, 0.7)); }
    } else {
      r(10, c0 + 10, 5, 2, peleS); // nuca
    }
    // cabelo
    if (est !== 'careca') {
      if (frente) {
        if (est === 'black') { /* já desenhado atrás da cabeça */ r(7, c0, 10, 2, cab); r(6, c0 + 1, 3, 4, cab); }
        else if (est === 'cacheado') { for (let i = 0; i < 6; i++) disco(7 + i * 2, c0 + 1 + (i % 2), 2, 2, cab); r(6, c0 + 1, 12, 3, cab); r(6, c0 + 4, 2, 5, cab); p(8, c0 + 4, cab); }
        else {
          r(8, c0 - 1, 8, 1, cab); r(7, c0, 10, 2, cab); r(6, c0 + 2, 12, 2, cab); r(6, c0 + 4, 2, 5, cab); r(8, c0 + 4, 2, 1, cab);
          r(14, c0 + 4, 3, 1, cab);
          if (est === 'topete') { r(10, c0 - 3, 7, 3, cab); r(15, c0 - 2, 3, 2, cab); }
          if (est === 'coque') { disco(9, c0 - 2, 2.6, 2.6, cab); }
          if (est === 'longo') { r(5, c0 + 2, 2, 12, cab); r(17, c0 + 3, 1, 6, cab); }
        }
        r(8, c0, 3, 1, cabC); p(7, c0 + 1, cabC);
      } else {
        if (est === 'black') { /* já desenhado */ } else if (est === 'cacheado') { for (let i = 0; i < 6; i++) disco(7 + i * 2, c0 + 1 + (i % 2), 2, 2, cab); r(6, c0 + 1, 12, 9, cab); r(7, c0 + 10, 10, 1, cab); }
        else {
          r(8, c0 - 1, 8, 1, cab); r(7, c0, 10, 1, cab); r(6, c0 + 1, 12, 9, cab); r(7, c0 + 10, 8, 1, cab);
          r(15, c0 + 5, 2, 3, peleS); p(16, c0 + 6, tom(pele, 0.7)); // orelha à direita (de costas, virado para a direita)
          if (est === 'topete') r(9, c0 - 3, 7, 3, cab);
          if (est === 'coque') disco(12, c0 - 1, 2.8, 2.6, cab);
          if (est === 'longo') { r(6, c0 + 10, 12, 8, cab); r(7, c0 + 18, 10, 1, cabS); }
        }
        r(9, c0, 4, 1, cabC); r(8, c0 + 1, 2, 1, cabC);
      }
    } else if (!frente) { r(9, c0 + 1, 3, 1, tom(pele, 1.1)); }
    else { r(9, c0 + 1, 3, 1, tom(pele, 1.12)); }

    // contorno escuro em volta da silhueta
    const fora = new Uint8Array(SW * SH);
    for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
      if (buf[y * SW + x] >>> 24) continue;
      if ((x > 0 && buf[y * SW + x - 1] >>> 24) || (x < SW - 1 && buf[y * SW + x + 1] >>> 24) || (y > 0 && buf[(y - 1) * SW + x] >>> 24) || (y < SH - 1 && buf[(y + 1) * SW + x] >>> 24)) fora[y * SW + x] = 1;
    }
    const k = pk(CONTORNO);
    for (let i = 0; i < fora.length; i++) if (fora[i]) buf[i] = k;
    return { buf, w: SW, h: SH };
  }

  // ------------------------------------------------------------------ quadro final
  let frame = null, img = null, arte = null, actx = null;
  let S = 2, DX = 0, DY = 0, W = 0, H = 0, assinatura = '', alvosTela = [];
  function preparar() {
    arte = document.createElement('canvas'); arte.width = AW; arte.height = AH;
    actx = arte.getContext('2d'); img = actx.createImageData(AW, AH); frame = new Uint32Array(img.data.buffer);
  }
  function colar(spr, ox, oy, espelho = false) {
    const { buf, w, h } = spr;
    for (let y = 0; y < h; y++) {
      const fy = oy + y; if (fy < 0 || fy >= AH) continue;
      for (let x = 0; x < w; x++) {
        const fx = ox + x; if (fx < 0 || fx >= AW) continue;
        const v = buf[y * w + (espelho ? w - 1 - x : x)], a = v >>> 24;
        if (!a) continue;
        const i = fy * AW + fx;
        if (a === 255) { frame[i] = v; continue; }
        const d = frame[i], f = a / 255, g = 1 - f;
        frame[i] = empac(Math.round((v & 255) * f + (d & 255) * g), Math.round(((v >>> 8) & 255) * f + ((d >>> 8) & 255) * g),
          Math.round(((v >>> 16) & 255) * f + ((d >>> 16) & 255) * g), Math.max(a, d >>> 24));
      }
    }
  }
  function marcaChao(x, y, cor, alfa, rx = 9, ry = 4.5) { // sombra / destaque no chão sob a pessoa (direto no quadro)
    const [cx, cy] = Pa(x, y);
    alvo(frame, AW, AH);
    disco(Math.round(cx), Math.round(cy), rx, ry, cor, alfa);
  }

  const Cena = {
    ESTADOS, INFO_LOCAL, LOCAIS_DESCANSO, NOS, ESTACOES, rota,
    medir(w, h, dpr, zoom = 1) {
      const ajuste = Math.min((w - 24) / AW, (h - 24) / AH);
      // tela estreita (celular em pé): ajusta pela altura e deixa arrastar para os lados
      let s = (ajuste >= 1 ? ajuste : Math.max(1, Math.min(2, (h - 24) / AH * 0.85))) * zoom;
      const dev = s * dpr, inteiro = Math.floor(dev);
      if (inteiro >= 1 && inteiro >= dev * 0.86) s = inteiro / dpr; // pixels inteiros: arte nítida
      S = s;
      W = Math.max(w, Math.ceil(AW * S + 24)); H = Math.max(h, Math.ceil(AH * S + 24));
      DX = Math.round((W - AW * S) / 2 * dpr) / dpr; DY = Math.round((H - AH * S) / 2 * dpr) / dpr;
      return { largura: W, altura: H };
    },
    escala: () => K * S,
    projetar: (x, y, z = 0) => { const [a, b] = Pa(x, y, z); return [DX + a * S, DY + b * S]; },
    atualizarFundo(d) {
      const agora = new Date();
      const ass = JSON.stringify([d.fila_triagem, d.fila_analise, d.estacoes, d.tv_ligada, d.atlas_trabalhando, agora.getHours(), agora.getMinutes(),
        (d.vagas || []).map(v => [v.status, v.sem_responsavel, v.pendentes > 0, v.recebidos, v.analisados]), d.quadro && [d.quadro.vagas_abertas, d.quadro.vagas_em_processamento, d.quadro.pendencias]]);
      if (ass === assinatura) return;
      assinatura = ass;
      if (!frame) preparar();
      montarItens(d);
      base = new Uint32Array(AW * AH);
      alvo(base, AW, AH);
      desenharBase(d);
      construirSprites();
    },
    // pessoas: [{ id, x, y, ap, dir, pose, quadro, humor, item, destaque }]; extra: { bolinha: [x, y, z] | null }
    desenhar(ctx, pessoas, extra = {}) {
      if (!frame) return [];
      frame.set(base);
      const dyn = pessoas.map(p => {
        const espelho = p.dir === 's' || p.dir === 'w', vista = p.dir === 's' || p.dir === 'e' ? 'frente' : 'costas';
        const spr = sprite(p.ap || {}, vista, p.pose, p.quadro, p.humor, p.item);
        const [ax, ay] = Pa(p.x, p.y), ox = Math.round(ax) - (espelho ? SW - 1 - PX : PX), oy = Math.round(ay) - PY;
        return { p, spr, espelho, ox, oy, box: [p.x - 0.12, p.y - 0.12, p.x + 0.12, p.y + 0.12], z0: 0, z1: 1.7, tela: [ox, oy, ox + SW - 1, oy + SH - 1], prof: p.x + p.y };
      });
      if (extra.bolinha) {
        const [bx, by, bz] = extra.bolinha, [sx, sy] = Pa(bx, by, bz);
        dyn.push({ bola: [Math.round(sx), Math.round(sy)], sombraBola: Pa(bx, by, 0.78), box: [bx - 0.03, by - 0.03, bx + 0.03, by + 0.03], z0: 0.78, z1: 1.5, tela: [sx - 2, sy - 2, sx + 2, sy + 2], prof: bx + by });
      }
      // posição de cada pessoa na ordem dos móveis: depois de tudo que está atrás e antes do que está na frente
      const n = ITENS.length;
      dyn.sort((a, b) => a.prof - b.prof);
      for (const e of dyn) {
        let frenteMin = n, atrasMax = -1;
        for (let i = 0; i < n; i++) {
          const it = ITENS[i];
          if (!cruza(it.tela, e.tela)) continue;
          const [ix0, iy0, ix1, iy1] = it.box, [ex0, ey0, ex1, ey1] = e.box;
          const sobrepoe = !(ix1 <= ex0 + E || ex1 <= ix0 + E || iy1 <= ey0 + E || ey1 <= iy0 + E);
          const o = sobrepoe ? (it.z1 <= 0.65 ? -1 : 1) : ordem(it, e);
          if (o < 0) atrasMax = i; else if (o > 0 && i < frenteMin) frenteMin = i;
        }
        e.pos = atrasMax >= frenteMin ? atrasMax + 1 : frenteMin;
      }
      for (let i = 0; i < dyn.length; i++) for (let j = 0; j < i; j++) if (cruza(dyn[i].tela, dyn[j].tela) && dyn[i].pos < dyn[j].pos) dyn[i].pos = dyn[j].pos;
      const porPos = new Map();
      for (const e of dyn) { if (!porPos.has(e.pos)) porPos.set(e.pos, []); porPos.get(e.pos).push(e); }
      const desenhaDyn = e => {
        if (e.bola) {
          alvo(frame, AW, AH); disco(Math.round(e.sombraBola[0]), Math.round(e.sombraBola[1]), 2, 1, '#000000', 0.3);
          ret(e.bola[0] - 1, e.bola[1] - 1, 2, 2, '#FFF6D6'); ponto(e.bola[0], e.bola[1], '#E7B400');
          return;
        }
        const { p } = e;
        marcaChao(p.x, p.y, '#000000', 0.22);
        if (p.destaque) { marcaChao(p.x, p.y, '#F5B544', 0.55, 11, 5.5); marcaChao(p.x, p.y, '#000000', 0.12, 8, 4); }
        colar(e.spr, e.ox, e.oy, e.espelho);
      };
      for (let i = 0; i <= n; i++) {
        for (const e of porPos.get(i) || []) desenhaDyn(e);
        if (i < n && ITENS[i].spr) { const s = ITENS[i].spr; colar(s, s.ox, s.oy); }
      }
      for (const [pos, lista] of porPos) if (pos > n) lista.forEach(desenhaDyn);
      actx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(arte, 0, 0, AW, AH, DX, DY, AW * S, AH * S);
      alvosTela = ALVOS_ARTE.map(a => ({ tipo: a.tipo, pts: a.pts.map(([x, y]) => [DX + x * S, DY + y * S]) }));
      return dyn.filter(e => e.p).map(e => {
        const [fx, fy] = Pa(e.p.x, e.p.y);
        return { id: e.p.id, sx: DX + fx * S, sy: DY + fy * S, hx: DX + fx * S, hy: DY + (fy - (PY - 4)) * S,
          x0: DX + (e.ox + 4) * S, y0: DY + (e.oy + 2) * S, x1: DX + (e.ox + SW - 4) * S, y1: DY + (e.oy + SH) * S };
      });
    },
    alvos: () => alvosTela,
    dentro(pt, pts) {
      let ok = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const [xi, yi] = pts[i], [xj, yj] = pts[j];
        if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) ok = !ok;
      }
      return ok;
    },
    // retrato em pixel art para os painéis: cabeça e ombros, de frente, ampliados sem suavização
    retrato(k, tam, ap, fundo = '#2A2540') {
      const s = sprite(ap || {}, 'frente', 'pe', 0, 'feliz', null), c = document.createElement('canvas');
      c.width = SW; c.height = SH;
      const ic = c.getContext('2d'), id = ic.createImageData(SW, SH);
      new Uint32Array(id.data.buffer).set(s.buf); ic.putImageData(id, 0, 0);
      k.save(); k.beginPath(); k.arc(tam / 2, tam / 2, tam / 2, 0, 7); k.clip();
      k.fillStyle = fundo; k.fillRect(0, 0, tam, tam);
      k.imageSmoothingEnabled = false;
      const recorte = 20, esc = tam / recorte;
      k.drawImage(c, 2, 3, recorte, recorte, 0, tam * 0.08, recorte * esc, recorte * esc);
      k.restore();
    },
  };
  window.Cena = Cena;
})();
