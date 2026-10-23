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
  // cor das pastas de cada área da equipe (banco de currículos); áreas novas ganham uma cor estável
  const COR_GRUPO = { operacional: '#E07B39', administrativo: '#4F81BD', comercial: '#3FB56B', promotores: '#E678AA',
    marketing: '#8E6BBF', lideranca: '#C0504D', ti: '#2F9C9C', inclusao: '#F2C14E', outros: '#8A8F99' };
  const corGrupo = c => COR_GRUPO[c] || escolha(c || '?', ['#B47C4C', '#5DA35E', '#D9534F', '#7E8CE0', '#C9A15B', '#3E8E9E']);
  // o que muda a cada quadro sem redesenhar o cenário (telas, TV): atualizado pelo app a cada quadro
  const VIVO = { estacoes: {}, tv: null, atlas: false, bolo: false, t: 0, periodo: 'dia' };
  // manhã e tarde com sol; fim de tarde dourado; à noite céu estrelado e o escritório só com as luzes quentes
  const periodoDe = h => (h >= 19 || h < 6 ? 'noite' : h >= 17 ? 'tarde' : 'dia');

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
  no('atlas', 3.0, 1.4, { f: 's', sentado: true, digita: true });
  // arquivo (banco de currículos): atrás do balcão, de frente para a sala, e diante dele
  no('arquivo0', 13.0, 1.35, { f: 's', item: 'pasta' }); no('arquivo1', 14.2, 1.35, { f: 's', item: 'pasta' });
  no('balcaoArq', 13.6, 3.05, { f: 'n' });
  no('sV', 3.0, 3.75, { f: 'n' });
  no('entrada', 0.55, 13.9, { f: 'e' });
  [10.4, 11.3, 12.2].forEach((y, i) => no('balcao' + i, 3.35, y, { f: 'w' }));
  [7.9, 9.3, 10.7].forEach((x, i) => no('quadro' + i, x, 1.45, { f: 'n' }));
  // locais de descanso: [nó, o que faz, para onde olha, sentado, objeto na mão]
  const LOCAIS_DESCANSO = [
    ['ping0', 'jogando pingue-pongue', 'e', 0, 'raquete'], ['ping1', 'jogando pingue-pongue', 'w', 0, 'raquete'],
    ['sofa1', 'jogando videogame', 'n', 1, 'controle'], ['sofa0', 'assistindo TV', 'n', 1], ['sofa2', 'assistindo TV', 'n', 1],
    ['comer0', 'tomando um café', 'e', 1, 'xicara'], ['comer1', 'fazendo um lanche', 'w', 1, 'lanche'], ['comer2', 'tomando um café', 'n', 1, 'xicara'],
    ['comer3', 'fazendo um lanche', 'e', 1, 'lanche'], ['comer4', 'tomando um café', 'w', 1, 'xicara'], ['comer5', 'fazendo um lanche', 'n', 1, 'lanche'],
    ['geladeira', 'pegando algo na geladeira', 'n', 0], ['bebedouro', 'tomando água', 'n', 0, 'copo'],
    ['pufe0', 'cochilando no pufe', 's', 1, 'sono'], ['pufe1', 'ouvindo música', 's', 1, 'fone'], ['pufe2', 'mexendo no celular', 's', 1, 'celular'],
    ['leitura0', 'lendo um livro', 's', 1, 'livro'], ['leitura1', 'lendo um livro', 's', 1, 'livro'],
  ];
  const POS_DESCANSO = {
    ping0: [19.0, 12.3], ping1: [23.0, 12.3], sofa0: [15.5, 13.75], sofa1: [16.3, 13.75], sofa2: [17.1, 13.75],
    comer0: [18.2, 2.9], comer1: [19.8, 2.9], comer2: [19.0, 3.7], comer3: [21.0, 2.9], comer4: [22.6, 2.9], comer5: [21.8, 3.7],
    geladeira: [23.1, 1.45], bebedouro: [17.18, 1.05], pufe0: [19.8, 14.95], pufe1: [21.2, 15.1], pufe2: [22.6, 14.95],
    leitura0: [7.8, 13.05], leitura1: [9.4, 13.05],
  };
  for (const [n, txt, f, s, item] of LOCAIS_DESCANSO) no(n, ...POS_DESCANSO[n], { txt, f, sentado: !!s, item });
  // lugares dos momentos da equipe: papo no bebedouro, alongamento em grupo e aniversário na mesa de reunião
  no('papoAgua', 17.85, 1.4, { f: 'w', item: 'copo' });
  [[20.5, 6.9], [21.7, 6.9], [22.9, 6.9], [21.1, 8.1], [22.3, 8.1]].forEach(([x, y], i) => no('alonga' + i, x, y, { f: 's' }));
  [[12.6, 13.15, 's'], [12.6, 14.85, 'n'], [11.75, 14.0, 'e'], [13.45, 14.0, 'w']].forEach(([x, y, f], i) => no('reuniao' + i, x, y, { f, sentado: true }));
  // cantinhos do Café, o gato do escritório: tapete do sofá, sol da janela, canto de leitura, copa, pufes, recepção, sala do Atlas, arquivo
  const CANTOS_GATO = [[16.3, 12.95, 'dorme'], [2.9, 7.3, 'dorme'], [8.6, 13.75, 'senta'], [20.4, 1.5, 'senta'], [21.9, 14.3, 'dorme'],
    [1.75, 12.3, 'senta'], [2.3, 3.7, 'dorme'], [11.9, 3.3, 'senta'], [10.9, 9.9, 'senta'], [19.6, 9.4, 'dorme']];
  CANTOS_GATO.forEach(([x, y, pose], i) => no('gato' + i, x, y, { pose }));

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
  function crt(x, y, z, face, tela, t) { // monitor de tubo; face: para onde a tela aponta; t: segundos (animação)
    const w = 0.22, p = 0.2, h = 0.4, alongX = face === 'n' || face === 's';
    const [bx0, by0, bx1, by1] = alongX ? [x - w, y - p, x + w, y + p] : [x - p, y - w, x + p, y + w];
    caixa(x - 0.12, y - 0.1, x + 0.12, y + 0.1, z, z + 0.04, tom(COR.crt, 0.8), { brilho: false });
    caixa(bx0, by0, bx1, by1, z + 0.04, z + h, COR.crt);
    const corTela = !tela ? '#232A28' : tela === 'trabalhando' ? '#14402A' : tela === 'pendencia' ? '#5A1F1C' : tela === 'concluido' ? '#1C5A33' : '#1D3442';
    if (face === 's' || face === 'e') { // tela visível
      const eixo = face === 's' ? 'x' : 'y', c = face === 's' ? by1 + 0.001 : bx1 + 0.001, a0 = (face === 's' ? bx0 : by0) + 0.05, a1 = (face === 's' ? bx1 : by1) - 0.05;
      painel(eixo, c, a0, a1, z + 0.1, z + h - 0.06, corTela);
      const vivo = t !== undefined && t >= 0, sem = x * 7.3 + y * 3.1;
      if (tela === 'trabalhando') { // linhas sendo digitadas, uma a uma, e o cursor piscando
        const ciclo = vivo ? Math.floor((t + sem) / 5.5) : 0, fase = vivo ? ((t + sem) % 5.5) / 1.2 : 3;
        for (let k = 0; k < 4; k++) {
          if (k > fase) break;
          const cheio = 0.35 + ruido(ciclo, k) * 0.6, parte = k + 1 <= fase ? 1 : fase - k;
          const l1 = a0 + 0.04, l2 = l1 + (a1 - a0 - 0.08) * cheio * parte, zz = z + h - 0.12 - k * 0.058;
          painel(eixo, c + 0.001, l1, l2, zz, zz + 0.022, k === 0 ? '#B6F5C8' : '#7CE39A');
          if (vivo && parte < 1 && Math.floor(t * 3) % 2) painel(eixo, c + 0.001, l2 + 0.01, l2 + 0.035, zz, zz + 0.03, '#E9FFF0');
        }
      } else if (tela === 'pendencia') {
        if (!vivo || Math.floor(t * 2) % 2) painel(eixo, c + 0.001, (a0 + a1) / 2 - 0.02, (a0 + a1) / 2 + 0.02, z + 0.16, z + h - 0.12, '#FF8A80');
      } else if (tela === 'concluido') painel(eixo, c + 0.001, a0 + 0.08, a1 - 0.08, z + 0.2, z + 0.24, '#A8F0BE');
      else if (tela) { // descanso de tela: um quadradinho passeando
        const u = vivo ? Math.abs(((t * 0.23 + sem) % 2) - 1) : 0.5, v = vivo ? Math.abs(((t * 0.31 + sem * 1.7) % 2) - 1) : 0.5;
        const pa = a0 + 0.03 + (a1 - a0 - 0.12) * u, pz = z + 0.13 + (h - 0.3) * v;
        painel(eixo, c + 0.001, pa, pa + 0.06, pz, pz + 0.05, '#5FA8C8');
      }
    } else { // costas do monitor: aberturas de ventilação
      const eixo = face === 'n' ? 'x' : 'y', c = face === 'n' ? by1 + 0.001 : bx1 + 0.001, a0 = face === 'n' ? bx0 : by0;
      for (let k = 0; k < 3; k++) painel(eixo, c, a0 + 0.1, a0 + 0.34, z + 0.14 + k * 0.07, z + 0.16 + k * 0.07, tom(COR.crt, 0.62));
    }
  }
  function teclado(x, y, z, alongX) {
    const [x0, y0, x1, y1] = alongX ? [x - 0.2, y - 0.07, x + 0.2, y + 0.07] : [x - 0.07, y - 0.2, x + 0.07, y + 0.2];
    caixa(x0, y0, x1, y1, z, z + 0.025, '#D6CDB8', { brilho: false, contorno: '#8C8370' });
  }
  function tela_tv(modo, t) { // TV da área de descanso: desligada, um programa ou o videogame de quem joga
    const c = 10.821, a0 = 15.46, a1 = 17.14, z0 = 0.65, z1 = 1.27;
    if (!modo) { painel('x', c, a0, a1, z0, z1, '#2A2F3A'); painel('x', c + 0.001, a0 + 0.1, a0 + 0.5, z1 - 0.12, z1 - 0.06, '#3A404D'); return; }
    if (modo === 'jogo') { // plataforma rolando, moedas e o bonequinho pulando
      painel('x', c, a0, a1, z0, z1, '#6FB3F0');
      painel('x', c + 0.001, a0, a1, z0, z0 + 0.12, '#5DA35E'); painel('x', c + 0.002, a0, a1, z0 + 0.1, z0 + 0.12, '#8DC97A');
      for (let k = 0; k < 4; k++) {
        const bx = a0 + ((k * 0.55 - t * 0.6) % 1.68 + 1.68) % 1.68;
        if (bx < a1 - 0.18) painel('x', c + 0.002, bx, bx + 0.16, z0 + 0.24 + (k % 2) * 0.14, z0 + 0.32 + (k % 2) * 0.14, '#C0703F');
        const mx = a0 + ((k * 0.42 + 0.2 - t * 0.6) % 1.68 + 1.68) % 1.68;
        if (mx < a1 - 0.06 && Math.floor(t * 4 + k) % 2) painel('x', c + 0.003, mx, mx + 0.05, z0 + 0.42, z0 + 0.47, '#FFE08A');
      }
      const pulo = Math.abs(Math.sin(t * 3.1)) * 0.2;
      painel('x', c + 0.003, a0 + 0.4, a0 + 0.5, z0 + 0.12 + pulo, z0 + 0.24 + pulo, '#E5534B');
      painel('x', c + 0.004, a0 + 0.42, a0 + 0.48, z0 + 0.2 + pulo, z0 + 0.23 + pulo, '#F6D1B1');
      painel('x', c + 0.003, a1 - 0.5, a1 - 0.08, z1 - 0.1, z1 - 0.05, '#FFFFFF');
      return;
    }
    // programa: céu mudando de cor, apresentador e a faixa de notícias correndo
    const hue = Math.floor(t / 6) % 3;
    painel('x', c, a0, a1, z0, z1, ['#5C9BD8', '#4F7FC0', '#6A8FD0'][hue]);
    painel('x', c + 0.001, a0, a1, z0, z0 + 0.22, '#8DC97A');
    painel('x', c + 0.002, a0 + 0.5, a0 + 0.78, z0 + 0.22, z0 + 0.44, '#3B6EC8');
    painel('x', c + 0.003, a0 + 0.56, a0 + 0.72, z0 + 0.44, z0 + 0.58 + (Math.floor(t * 5) % 2) * 0.01, '#E8B98F');
    painel('x', c + 0.002, a0, a1, z0 + 0.02, z0 + 0.1, '#1E2A44');
    const fx = a0 + ((-t * 0.35) % 1.68 + 1.68) % 1.68;
    painel('x', c + 0.003, fx, Math.min(a1, fx + 0.6), z0 + 0.045, z0 + 0.075, '#F2F3F7');
    painel('x', c + 0.003, a1 - 0.36, a1 - 0.08, z1 - 0.18, z1 - 0.06, '#FFE08A');
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
  // lombadas de fichários numa face (plano y = c), agrupadas pela área e coloridas como as pastas da tela do banco
  function lombadas(c, a0, a1, z0, z1, fila) {
    let a = a0, n = 0;
    for (const [cor, cheia] of fila) {
      if (!cor) { a += 0.035; continue; } // espaço entre uma área e outra
      const w = 0.075 + ruido(n, 3) * 0.02;
      if (a + w > a1) break;
      const alt = (z1 - z0) * (cheia ? 0.9 + ruido(n, 7) * 0.08 : 0.82);
      painel('x', c, a, a + w - 0.008, z0, z0 + alt, cheia ? cor : tom(cor, 0.62));
      painel('x', c + 0.0005, a + 0.012, a + w - 0.02, z0 + alt * 0.62, z0 + alt * 0.8, cheia ? '#F4F1E8' : tom('#F4F1E8', 0.7));
      painel('x', c + 0.0005, a + w - 0.016, a + w - 0.008, z0, z0 + alt, tom(cor, 0.7));
      a += w; n++;
    }
  }
  // sequência de fichários a partir das quantidades reais: mais currículos na área, mais fichários dela
  function filaFicharios(grupos) {
    const fila = [];
    for (const g of grupos) {
      const cor = corGrupo(g.codigo), k = g.total ? Math.min(9, 1 + Math.ceil(Math.log2(1 + g.total) * 1.6)) : 1;
      for (let i = 0; i < k; i++) fila.push([cor, g.total > 0]);
      fila.push([null, false]);
    }
    return fila;
  }
  function arquivoCurriculos(banco) {
    const grupos = (banco && banco.grupos) || [];
    const fila = filaFicharios(grupos.length ? grupos : [{ codigo: 'outros', total: 0 }]);
    // as prateleiras ficam cheias: a sequência das áreas continua de uma prateleira para a outra
    const pedaco = (de, ate) => Array.from({ length: ate - de }, (_, i) => fila[(de + i) % fila.length]);
    // estante alta, à esquerda da janela
    item([11.72, 0, 12.36, 0.46], 0, 1.95, () => {
      caixa(11.72, 0, 12.36, 0.46, 0, 1.95, COR.movelEsc);
      for (let p = 0; p < 4; p++) {
        const z0 = 0.1 + p * 0.46;
        painel('x', 0.461, 11.77, 12.31, z0, z0 + 0.42, tom(COR.movelEsc, 0.45));
        lombadas(0.462, 11.79, 12.3, z0, z0 + 0.38, pedaco(p * 7, p * 7 + 7));
      }
    }, { sombra: true });
    obst(11.72, 0, 12.36, 0.46);
    // gaveteiros baixos sob a janela, com fichários em duas prateleiras
    item([12.42, 0, 15.12, 0.46], 0, 0.84, () => {
      caixa(12.42, 0, 15.12, 0.46, 0, 0.84, COR.movel);
      for (let p = 0; p < 2; p++) {
        const z0 = 0.06 + p * 0.39;
        painel('x', 0.461, 12.47, 15.07, z0, z0 + 0.35, tom(COR.movel, 0.5));
        lombadas(0.462, 12.5, 15.05, z0, z0 + 0.32, pedaco(28 + p * 34, 28 + p * 34 + 34));
      }
    }, { sombra: true });
    obst(12.42, 0, 15.12, 0.46);
    // balcão de atendimento do arquivo
    item([12.3, 2.05, 14.9, 2.55], 0, 1.04, () => {
      caixa(12.35, 2.05, 14.9, 2.55, 0, 0.98, '#5C6B8A');
      for (let x = 12.45; x < 14.8; x += 0.62) painel('x', 2.551, x, x + 0.5, 0.12, 0.86, '#52607E');
      // placa âmbar com "linhas de texto" e as cores das áreas
      painel('x', 2.552, 13.1, 14.1, 0.58, 0.84, '#F5B544');
      painel('x', 2.553, 13.18, 14.02, 0.74, 0.77, '#3A2A10'); painel('x', 2.553, 13.3, 13.9, 0.66, 0.69, '#7A5A20');
      grupos.slice(0, 9).forEach((g, i) => painel('x', 2.553, 13.12 + i * 0.106, 13.2 + i * 0.106, 0.6, 0.63, corGrupo(g.codigo)));
      caixa(12.3, 2.0, 14.95, 2.6, 0.98, 1.04, '#C9D3E6');
    }, { sombra: true });
    obst(12.3, 2.05, 14.9, 2.55);
    // pilhas de pastas sobre o balcão: uma por área com currículos (altura pela quantidade)
    const comCv = grupos.filter(g => g.total > 0).slice(0, 7);
    comCv.forEach((g, i) => {
      const x = 12.5 + i * 0.34, cor = corGrupo(g.codigo), k = Math.min(7, Math.ceil(Math.log2(1 + g.total) * 1.5));
      item([x, 2.15, x + 0.28, 2.47], 1.04, 1.04 + k * 0.028 + 0.02, () => {
        for (let j = 0; j < k; j++) {
          const z = 1.04 + j * 0.028, dx = (ruido(i, j) - 0.5) * 0.03;
          caixa(x + dx, 2.17, x + 0.26 + dx, 2.45, z, z + 0.012, '#F4F1E8', { brilho: false, contorno: false });
          caixa(x + dx, 2.17, x + 0.26 + dx, 2.45, z + 0.012, z + 0.026, cor, { brilho: false, contorno: tom(cor, 0.5) });
        }
      });
    });
    if (!comCv.length) item([13.3, 2.15, 13.9, 2.47], 1.04, 1.08, () => caixa(13.35, 2.17, 13.85, 2.45, 1.04, 1.06, '#AEB4BF', { brilho: false }));
    item([14.45, 2.12, 14.8, 2.48], 1.04, 1.5, () => luminaria(14.62, 2.3, 1.04, '#7EC3EB'));
    ALVOS_ARTE.push({ tipo: 'banco', pts: envoltoria([Pa(11.72, 0, 1.95), Pa(12.36, 0, 1.95), Pa(15.12, 0, 0.84), Pa(15.12, 0.46, 0),
      Pa(14.95, 2.0, 1.1), Pa(14.95, 2.6, 0), Pa(12.3, 2.6, 0), Pa(11.72, 0.46, 0), Pa(12.3, 2.0, 1.1)]) });
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

  // ------------------------------------------------------------------ pequenos confortos do escritório
  const NOTA = [[1, 0], [1, 1], [1, 2], [0, 3], [1, 3], [0, 2]]; // colcheia de 2 × 4 pixels
  function nota(x, y, cor, alfa) { for (const [dx, dy] of NOTA) ponto(x + dx, y + dy, cor, alfa); ponto(x + 2, y, cor, alfa); ponto(x + 2, y + 1, cor, alfa * 0.6); }
  function varal(pts) { // fio com lâmpadas coloridas que piscam devagar (mais vivas à noite)
    const cores = ['#FFD36E', '#FF8C6B', '#8FE3B0', '#7EC3EB', '#F5A3C7'], noite = VIVO.periodo !== 'dia';
    for (let i = 0; i < pts.length - 1; i++) { const a = Pa(...pts[i]), b = Pa(...pts[i + 1]); linha(Math.round(a[0]), Math.round(a[1]), Math.round(b[0]), Math.round(b[1]), '#3A2A20'); }
    for (let i = 1; i < pts.length - 1; i += 2) {
      const [px, py] = Pa(...pts[i]).map(Math.round), cor = cores[i % cores.length];
      const brilho = 0.55 + 0.45 * Math.sin(VIVO.t * 1.6 + i * 1.9);
      if (noite) { ponto(px - 1, py + 1, cor, 0.3 * brilho); ponto(px + 2, py + 1, cor, 0.3 * brilho); ponto(px, py + 3, cor, 0.3 * brilho); ponto(px + 1, py - 1, cor, 0.2 * brilho); }
      ret(px, py + 1, 2, 2, brilho > 0.35 ? cor : tom(cor, 0.55)); ponto(px, py, '#3A2A20');
    }
  }
  function curva(eixo, a0, a1, c, zTopo, queda, passo = 0.18, vao = 1.6) { // pontos de um varal preso de tanto em tanto
    const pts = [];
    for (let a = a0; a <= a1 + 1e-6; a += passo) {
      const u = ((a - a0) % vao) / vao, z = zTopo - queda * Math.sin(Math.PI * u);
      pts.push(eixo === 'x' ? [a, c, z] : [c, a, z]);
    }
    return pts;
  }
  function aconchego() {
    // varais de luzinhas no alto das paredes
    const fundo = curva('x', 6.2, 23.8, 0.03, 2.5, 0.1), lateral = curva('y', 8.7, 15.9, 0.03, 2.5, 0.1);
    item([6.2, 0, 23.8, 0.06], 2.36, 2.56, () => varal(fundo), { anim: true });
    item([0, 8.7, 0.06, 15.9], 2.36, 2.56, () => varal(lateral), { anim: true });

    // aquário na sala de espera: móvel de madeira, água, areia, plantinhas, peixes nadando e bolhas
    item([2.0, 15.45, 2.9, 15.9], 0, 0.62, () => {
      caixa(2.0, 15.45, 2.9, 15.9, 0, 0.6, COR.movelEsc);
      painel('x', 15.901, 2.08, 2.42, 0.08, 0.52, tom(COR.movelEsc, 1.15)); painel('x', 15.901, 2.48, 2.82, 0.08, 0.52, tom(COR.movelEsc, 1.15));
      caixa(1.98, 15.43, 2.92, 15.92, 0.6, 0.62, '#4A3022', { brilho: false });
    }, { sombra: true });
    obst(2.0, 15.45, 2.9, 15.9);
    item([2.04, 15.5, 2.86, 15.86], 0.62, 1.12, t => {
      caixa(2.04, 15.5, 2.86, 15.86, 0.62, 1.08, '#5FB3D9', { topo: '#8FD3F0', contorno: '#2E6E8E' });
      painel('x', 15.861, 2.05, 2.85, 0.62, 0.68, '#E3C98F'); painel('y', 2.861, 15.51, 15.85, 0.62, 0.68, '#D2B57A');
      for (const [u, h, cor] of [[2.15, 0.22, COR.folha1], [2.2, 0.16, COR.folha2], [2.72, 0.26, COR.folha1], [2.77, 0.14, COR.folha3]])
        painel('x', 15.862, u, u + 0.03, 0.66, 0.66 + h + Math.sin(Math.max(0, t) * 1.5 + u * 9) * 0.015, cor);
      const tt = Math.max(0, t);
      [['#F28C38', 0.6, 0.0], ['#F2C14E', 0.45, 2.1], ['#4F81BD', 0.75, 4.0]].forEach(([cor, vel, fase], k) => {
        const ida = Math.sin(tt * vel * 0.5 + fase), fx = 2.42 + ida * 0.3, fz = 0.76 + k * 0.08 + Math.sin(tt * 1.3 + fase) * 0.02;
        const dir = Math.cos(tt * vel * 0.5 + fase) >= 0 ? 1 : -1;
        painel('x', 15.863, fx - 0.05, fx + 0.05, fz, fz + 0.045, cor);
        painel('x', 15.863, dir > 0 ? fx - 0.09 : fx + 0.05, dir > 0 ? fx - 0.05 : fx + 0.09, fz - 0.01, fz + 0.055, tom(cor, 0.8));
        painel('x', 15.864, dir > 0 ? fx + 0.025 : fx - 0.04, dir > 0 ? fx + 0.04 : fx - 0.025, fz + 0.022, fz + 0.035, '#1B1420');
      });
      for (let k = 0; k < 3; k++) { const f = (tt * 0.4 + k / 3) % 1; painel('x', 15.864, 2.7 + Math.sin(f * 9) * 0.015, 2.725 + Math.sin(f * 9) * 0.015, 0.7 + f * 0.33, 0.725 + f * 0.33, '#E6F7FF', 0.8 * (1 - f)); }
      caixa(2.03, 15.49, 2.87, 15.87, 1.08, 1.12, '#2B2F3D', { brilho: false });
    }, { anim: true });

    // rádio na bancada da copa, com notas musicais subindo
    item([20.55, 0.12, 21.05, 0.42], 0.93, 1.75, t => {
      caixa(20.58, 0.15, 21.02, 0.4, 0.93, 1.15, '#B5523B', { topo: '#C96A50' });
      painel('x', 0.401, 20.62, 20.82, 0.97, 1.11, '#3A2A20');
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) painel('x', 0.402, 20.645 + i * 0.06, 20.67 + i * 0.06, 0.995 + j * 0.04, 1.015 + j * 0.04, '#6B4A3A');
      painel('x', 0.401, 20.86, 20.98, 1.04, 1.1, '#F5E6B8'); painel('x', 0.402, 20.9, 20.92, 1.04, 1.1, '#C0504D');
      const [ax, ay] = Pa(20.95, 0.2, 1.15), [bx, by] = Pa(21.08, 0.2, 1.5); linha(Math.round(ax), Math.round(ay), Math.round(bx), Math.round(by), '#8A8F99');
      if (t < 0) { for (let k = 0; k < 3; k++) { const [nx, ny] = Pa(20.8 + k * 0.1, 0.28, 1.2 + k * 0.15); nota(Math.round(nx), Math.round(ny), '#F5B544', 1); } return; }
      for (let k = 0; k < 3; k++) {
        const f = (t * 0.3 + k / 3) % 1, [nx, ny] = Pa(20.8 + Math.sin(f * 5 + k) * 0.12, 0.28, 1.2 + f * 0.5);
        nota(Math.round(nx), Math.round(ny), ['#F5B544', '#7EC3EB', '#E678AA'][k], Math.sin(f * Math.PI) * 0.9);
      }
    }, { anim: true });

    // recepção: as suculentas e o pote de balas da Bia
    item([2.36, 9.66, 2.64, 9.94], 1.06, 1.32, () => {
      caixa(2.38, 9.68, 2.62, 9.92, 1.06, 1.16, '#E9DCC0', { topo: '#6B4A3A' });
      for (const [dx, dy, r, cor] of [[-0.05, -0.04, 3, '#7FB77E'], [0.05, 0.03, 3, '#5DA37A'], [0, 0, 2, '#A8D5A0'], [0.04, -0.05, 2, '#E8A0B4']]) {
        const [cx, cy] = Pa(2.5 + dx, 9.8 + dy, 1.2); disco(Math.round(cx), Math.round(cy), r, r - 1, cor); ponto(Math.round(cx), Math.round(cy) - 1, tom(cor, 1.25));
      }
    });
    item([2.38, 10.08, 2.62, 10.32], 1.06, 1.36, () => {
      cilindro(2.5, 10.2, 0.11, 1.06, 1.28, '#DDEFF5', { topo: '#EAF6FA', contorno: '#9DB8C4' });
      const cores = ['#E5534B', '#F2C14E', '#9BBB59', '#E678AA', '#7EC3EB', '#F28C38'];
      for (let k = 0; k < 16; k++) { const [cx, cy] = Pa(2.42 + ruido(k, 1) * 0.16, 10.12 + ruido(2, k) * 0.16, 1.08 + ruido(k, 5) * 0.14); ret(Math.round(cx), Math.round(cy), 2, 1, cores[k % cores.length]); }
      cilindro(2.5, 10.2, 0.08, 1.28, 1.33, '#E5534B');
    });

    // sala do Atlas: o bonsai de jabuticaba (entre os papéis e o monitor)
    item([3.4, 2.62, 3.68, 2.86], 0.78, 1.1, () => {
      const bx = 3.54, by = 2.74;
      caixa(bx - 0.12, by - 0.08, bx + 0.12, by + 0.08, 0.78, 0.84, '#3E5C76', { topo: '#6B4A3A' });
      const [tx, ty] = Pa(bx, by, 0.84).map(Math.round), [ux, uy] = Pa(bx - 0.03, by, 0.98).map(Math.round);
      linha(tx, ty, ux, uy, '#5A3B2E'); linha(tx + 1, ty, ux + 1, uy, '#6E4A35');
      for (const [dx, dz, r] of [[-0.06, 0.98, 3], [0.06, 1.0, 3], [0, 1.04, 3]]) { const [cx, cy] = Pa(bx + dx, by, dz).map(Math.round); disco(cx, cy, r, r - 1, COR.folha3); disco(cx - 1, cy - 1, r - 1, r - 2, COR.folha1); }
      for (let k = 0; k < 4; k++) { const [cx, cy] = Pa(bx - 0.06 + k * 0.04, by, 0.97 + (k % 2) * 0.04).map(Math.round); ponto(cx, cy, '#3B1F4A'); }
    });

    // copa: fruteira na mesa de lanche
    item([21.62, 2.72, 21.98, 3.08], 0.78, 0.98, () => {
      cilindro(21.8, 2.9, 0.17, 0.78, 0.84, '#E9E4DA', { topo: '#CFC6B4' });
      for (const [dx, dy, dz, cor] of [[-0.06, -0.02, 0.86, '#F28C38'], [0.06, 0.02, 0.86, '#C0392B'], [0, -0.07, 0.9, '#F2C14E'], [0.02, 0.07, 0.88, '#9BBB59'], [-0.02, 0.01, 0.93, '#F28C38']]) {
        const [cx, cy] = Pa(21.8 + dx, 2.9 + dy, dz); disco(Math.round(cx), Math.round(cy), 2, 2, cor); ponto(Math.round(cx) - 1, Math.round(cy) - 1, tom(cor, 1.3));
      }
    });

    // abajur ao lado do sofá
    item([14.42, 13.42, 14.7, 13.7], 0, 1.62, () => {
      cilindro(14.56, 13.56, 0.13, 0, 0.04, '#3A3A44');
      caixa(14.545, 13.545, 14.575, 13.575, 0.04, 1.32, '#3A3A44', { brilho: false, contorno: false });
      cilindro(14.56, 13.56, 0.17, 1.3, 1.58, '#F6D9A0', { topo: '#FFF0C8' });
    });
    obst(14.42, 13.42, 14.7, 13.7);
    // canecas e objetos pessoais em cada estação (a caneca solta vapor enquanto alguém trabalha nela)
    const pessoais = ['foto', 'cacto', 'canetas', 'postit', 'foto', 'cacto'];
    const canecas = ['#E5534B', '#4F81BD', '#F2C14E', '#3FB56B', '#E678AA', '#F4F1E8', '#8E6BBF', '#E07B39'];
    ESTACOES.forEach((e, i) => {
      const fA = e.fila === 'A', mx = fA ? e.x + 0.45 : e.x - 0.45, my = fA ? 6.47 : 7.63, cor = canecas[i % canecas.length];
      item([mx - 0.07, my - 0.07, mx + 0.07, my + 0.07], 0.76, 1.2, t => {
        cilindro(mx, my, 0.05, 0.76, 0.87, cor, { topo: '#4A2E1E' });
        const [hx, hy] = Pa(mx + 0.06, my, 0.83); ponto(Math.round(hx) + 1, Math.round(hy), tom(cor, 0.7)); ponto(Math.round(hx) + 1, Math.round(hy) - 1, tom(cor, 0.7));
        if (t >= 0 && VIVO.estacoes[i] !== 'trabalhando') return;
        for (let k = 0; k < 3; k++) { const f = ((t < 0 ? 0.3 : t) * 0.7 + k / 3) % 1, [vx, vy] = Pa(mx, my, 0.9 + f * 0.25); ponto(Math.round(vx + Math.sin(f * 7 + k) * 1.5), Math.round(vy), '#FFFFFF', 0.5 * (1 - f)); }
      }, { anim: true });
      const px = fA ? e.x - 0.42 : e.x - 0.42, py = fA ? 6.86 : 7.3, tipo = pessoais[(i * 5 + 1) % pessoais.length];
      item([px - 0.08, py - 0.08, px + 0.08, py + 0.08], 0.76, 1.05, () => {
        if (tipo === 'foto') { // porta-retrato de pé, virado para quem senta
          caixa(px - 0.07, py - 0.015, px + 0.07, py + 0.015, 0.76, 0.94, '#6B4A2A');
          const foto = escolha(i, ['#7EC3EB', '#9BBB59', '#F2C14E']); // a foto fica para quem senta; quem olha a sala vê o verso nas mesas de cima
          if (fA) painel('x', py + 0.016, px - 0.05, px + 0.05, 0.78, 0.92, tom('#6B4A2A', 0.8));
          else { painel('x', py + 0.016, px - 0.05, px + 0.05, 0.78, 0.92, foto); painel('x', py + 0.017, px - 0.02, px + 0.03, 0.8, 0.86, '#F4D7B0'); }
        } else if (tipo === 'cacto') {
          cilindro(px, py, 0.045, 0.76, 0.82, '#C2703F');
          caixa(px - 0.02, py - 0.02, px + 0.02, py + 0.02, 0.82, 0.96, '#5DA35E', { brilho: false });
          caixa(px + 0.02, py - 0.01, px + 0.05, py + 0.01, 0.87, 0.9, '#5DA35E', { brilho: false, contorno: false });
          const [fx, fy] = Pa(px, py, 0.97); ponto(Math.round(fx), Math.round(fy), '#E678AA');
        } else if (tipo === 'canetas') {
          cilindro(px, py, 0.045, 0.76, 0.86, '#3E5C76');
          for (const [dx, cor] of [[-0.02, '#E5534B'], [0, '#2B2F3D'], [0.02, '#4F81BD']]) { const [ax, ay] = Pa(px + dx, py, 0.86), [bx, by] = Pa(px + dx * 1.6, py, 0.97); linha(Math.round(ax), Math.round(ay), Math.round(bx), Math.round(by), cor); }
        } else {
          caixa(px - 0.06, py - 0.06, px + 0.06, py + 0.06, 0.76, 0.79, '#FFE66D', { brilho: false });
          caixa(px - 0.04, py - 0.07, px + 0.08, py + 0.05, 0.79, 0.8, '#FF9DBB', { brilho: false });
        }
      });
    });
  }

  // ------------------------------------------------------------------ montagem do cenário (depende dos dados)
  function montarItens(d) {
    ITENS = []; BLOQUEIOS.length = 0; ALVOS_ARTE = [];

    // ---- sala do gerente (vidro)
    vidro(0, 4.96, 3.9, 5.04); vidro(5.1, 4.96, 6.0, 5.04); vidro(5.96, 0, 6.04, 5.04);
    item([1.5, 2.1, 4.5, 2.9], 0, 0.78, () => {
      caixa(1.5, 2.1, 4.5, 2.9, 0, 0.72, COR.movelEsc);
      painel('x', 2.901, 1.7, 4.3, 0.12, 0.6, tom(COR.movelEsc, 0.9));
      caixa(1.45, 2.05, 4.55, 2.95, 0.72, 0.78, tom(COR.movelEsc, 1.2));
    }, { sombra: true });
    obst(1.5, 2.1, 4.5, 2.9);
    item([1.75, 2.2, 2.25, 2.62], 0.78, 1.2, () => crt(2.0, 2.4, 0.78, 'n', 'ligado')); // de costas para a sala
    item([3.75, 2.2, 4.25, 2.62], 0.78, 1.2, () => crt(4.0, 2.4, 0.78, 'n', 'ligado'));
    item([2.6, 2.3, 3.4, 2.6], 0.78, 0.82, () => { caixa(2.65, 2.32, 3.0, 2.58, 0.78, 0.8, '#F4F1E8', { brilho: false }); caixa(3.05, 2.35, 3.35, 2.55, 0.78, 0.81, '#E9E3D3', { brilho: false }); });
    item([4.3, 2.75, 4.48, 2.9], 0.78, 1.25, () => luminaria(4.39, 2.82, 0.78, '#3E8E5B'));
    cadeiraEscritorio(3.0, 1.4, 's', '#5A3B2E');
    estante(0.0, 0.5, 0.42, 2.3, 2.0, 'y');
    planta(0.65, 4.35, 1.25);
    planta(5.45, 4.35, 0.9, 1);

    // ---- quadro de vagas e janela
    planta(6.6, 0.55, 1.0, 1);
    arquivoCurriculos(d.banco);
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
        const e = ESTACOES[i], fA = e.fila === 'A';
        const my = fA ? 6.86 : 7.34;
        // a tela (e a folha ao lado) acompanha o estado de quem está na estação, quadro a quadro
        item([e.x - 0.22, my - 0.2, e.x + 0.22, my + 0.2], 0.76, 1.16, t => crt(e.x, my, 0.76, fA ? 'n' : 's', VIVO.estacoes[i] || null, t), { anim: !fA });
        item([e.x - 0.2, (fA ? 6.4 : 7.66) - 0.07, e.x + 0.2, (fA ? 6.4 : 7.66) + 0.07], 0.76, 0.79, () => teclado(e.x, fA ? 6.4 : 7.66, 0.76, true));
        item([e.x + 0.3, my - 0.15, e.x + 0.6, my + 0.15], 0.76, 0.8, t => {
          if (t < 0 || VIVO.estacoes[i] === 'trabalhando') caixa(e.x + 0.32, my - 0.14, e.x + 0.58, my + 0.14, 0.76, 0.79, '#FFFFFF', { brilho: false, contorno: '#B8B2A4' });
        }, { anim: true });
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
    item([12.5, 13.9, 12.85, 14.25], 0.78, 1.12, t => { // bolo de aniversário (só durante a festa)
      if (t >= 0 && !VIVO.bolo) return;
      cilindro(12.67, 14.07, 0.2, 0.78, 0.8, '#F4F1E8', { contorno: '#B8B2A4' });
      cilindro(12.67, 14.07, 0.15, 0.8, 0.92, '#7B4A2A', { topo: '#F7C6D9' });
      for (const [dx, dy, cor] of [[-0.06, -0.03, '#7EC3EB'], [0.05, 0.02, '#F2C14E'], [0, 0.07, '#E678AA']]) {
        caixa(12.67 + dx - 0.01, 14.07 + dy - 0.01, 12.67 + dx + 0.01, 14.07 + dy + 0.01, 0.92, 1.0, cor, { brilho: false, contorno: false });
        const [fx, fy] = Pa(12.67 + dx, 14.07 + dy, 1.03 + (Math.floor(Math.max(0, t) * 8 + dx * 50) % 2) * 0.012);
        ponto(Math.round(fx), Math.round(fy), '#FFB23E'); ponto(Math.round(fx), Math.round(fy) - 1, '#FFF1A8');
      }
    }, { anim: true });
    item([12.1, 13.5, 13.1, 14.5], 0.78, 0.79, t => { // confete na mesa
      if (t >= 0 && !VIVO.bolo) return;
      const cores = ['#E678AA', '#7EC3EB', '#F2C14E', '#9BBB59', '#E07B39'];
      for (let k = 0; k < 14; k++) {
        const [cx, cy] = Pa(12.15 + ruido(k, 3) * 0.9, 13.55 + ruido(4, k) * 0.9, 0.785);
        ponto(Math.round(cx), Math.round(cy), cores[k % cores.length]);
      }
    }, { anim: true });
    cadeiraEscritorio(12.6, 13.15, 's', '#4B5068'); cadeiraEscritorio(12.6, 14.85, 'n', '#4B5068');
    cadeiraEscritorio(11.75, 14.0, 'e', '#4B5068'); cadeiraEscritorio(13.45, 14.0, 'w', '#4B5068');

    // ---- área de descanso
    item([15.0, 10.6, 17.6, 11.0], 0, 0.5, () => caixa(15.0, 10.6, 17.6, 11.0, 0, 0.48, COR.movelEsc), { sombra: true });
    item([15.4, 10.68, 17.2, 10.82], 0.5, 1.32, () => {
      caixa(16.2, 10.7, 16.4, 10.8, 0.5, 0.6, '#1E2026', { brilho: false });
      caixa(15.4, 10.68, 17.2, 10.82, 0.6, 1.32, '#1E2026');
    });
    item([15.46, 10.82, 17.14, 10.84], 0.65, 1.27, t => tela_tv(t < 0 ? 'tv' : VIVO.tv, Math.max(0, t)), { anim: true });
    obst(15.0, 10.6, 17.6, 11.0);
    item([15.5, 12.1, 17.1, 12.6], 0, 0.42, () => { caixa(15.5, 12.1, 17.1, 12.6, 0.36, 0.42, COR.movel); for (const [x, y] of [[15.55, 12.15], [17.0, 12.15], [15.55, 12.5], [17.0, 12.5]]) caixa(x, y, x + 0.05, y + 0.05, 0, 0.36, COR.movelEsc, { brilho: false }); }, { sombra: true });
    obst(15.5, 12.1, 17.1, 12.6);
    const SOFA = '#A33F3F';
    item([14.9, 13.4, 17.7, 14.1], 0, 0.42, () => caixa(15.1, 13.4, 17.5, 14.1, 0, 0.42, SOFA), { sombra: true });
    item([14.9, 14.0, 17.7, 14.3], 0, 0.9, () => caixa(14.9, 14.0, 17.7, 14.3, 0, 0.9, tom(SOFA, 0.92)));
    item([14.9, 13.4, 15.1, 14.0], 0, 0.62, () => caixa(14.9, 13.4, 15.1, 14.0, 0, 0.62, tom(SOFA, 0.95)));
    item([17.5, 13.4, 17.7, 14.0], 0, 0.66, () => {
      caixa(17.5, 13.4, 17.7, 14.0, 0, 0.62, tom(SOFA, 0.95));
      caixa(17.48, 13.48, 17.72, 13.9, 0.62, 0.66, '#E9DCC0', { brilho: false }); // manta dobrada no braço, caindo pelo lado
      for (let k = 0; k < 4; k++) painel('y', 17.721, 13.5 + k * 0.1, 13.55 + k * 0.1, 0.22, 0.66, k % 2 ? '#C0504D' : '#E9DCC0');
      painel('y', 17.722, 13.5, 13.9, 0.4, 0.44, '#8E2F2C');
    });
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

    aconchego();

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
      const [c0, c1, c2] = { dia: ['#8FC3E3', '#B5DCF2', '#D8EEF9'], tarde: ['#E58A55', '#F4B26E', '#FFE0A6'], noite: ['#141C38', '#1E2A50', '#2C3B68'] }[per];
      painel(eixo, 0.004, a0, a1, z0, z1, c0);
      painel(eixo, 0.005, a0, a1, z0 + (z1 - z0) * 0.55, z1, c1);
      if (per !== 'noite') painel(eixo, 0.006, a0 + (a1 - a0) * 0.62, a1, z0 + 0.15, z0 + 0.45, c2); // nuvem
      else {
        for (let k = 0; k < Math.round((a1 - a0) * 6); k++) { // estrelas e a lua
          const u = a0 + 0.05 + ruido(k, a0) * (a1 - a0 - 0.1), z = z0 + 0.1 + ruido(a0, k) * (z1 - z0 - 0.2);
          painel(eixo, 0.006, u, u + 0.03, z, z + 0.03, ruido(k, k) > 0.7 ? '#FFF1B8' : '#DCE6FF');
        }
        const m = a0 + (a1 - a0) * 0.72, mz = z1 - 0.3;
        painel(eixo, 0.006, m - 0.07, m + 0.07, mz - 0.1, mz + 0.1, '#F4EBC8'); painel(eixo, 0.0065, m - 0.11, m - 0.03, mz - 0.05, mz + 0.05, '#F4EBC8');
        painel(eixo, 0.007, m + 0.01, m + 0.08, mz - 0.04, mz + 0.08, c1);
      }
      painel(eixo, 0.007, (a0 + a1) / 2 - 0.03, (a0 + a1) / 2 + 0.03, z0, z1, f);
      painel(eixo, 0.007, a0, a1, (z0 + z1) / 2 - 0.03, (z0 + z1) / 2 + 0.03, f);
      const [sol, forte, fraco] = { dia: ['#FFE6A6', 0.16, 0.07], tarde: ['#FFB36B', 0.22, 0.09], noite: ['#9FB4FF', 0.05, 0.025] }[per];
      const chao = eixo === 'y' ? [[1.64, a0 + 0.57], [4.0, a0 + 1.4], [4.0, a1 + 1.4], [1.64, a1 + 0.57]] : [[a0 + 0.57, 1.64], [a0 + 1.4, 4.0], [a1 + 1.4, 4.0], [a1 + 0.57, 1.64]];
      poli(chao.map(([x, y]) => Pa(x, y)), sol, forte);
      const jan = eixo === 'y' ? [Pa(0, a0, z1), Pa(0, a1, z1), Pa(0, a1, z0), Pa(0, a0, z0)] : [Pa(a0, 0, z1), Pa(a1, 0, z1), Pa(a1, 0, z0), Pa(a0, 0, z0)];
      const todos = [...jan, ...chao.map(([x, y]) => Pa(x, y))];
      poli(envoltoria(todos), sol, fraco);
      // cortinas de linho dos dois lados, presas com laço, e o varão
      if (a1 - a0 > 1.5) {
        const pano = '#C8743F';
        for (const [c0c, c1c] of [[a0 - 0.42, a0 - 0.02], [a1 + 0.02, a1 + 0.42]]) {
          for (let u = c0c, k = 0; u < c1c - 0.01; u += 0.1, k++) painel(eixo, 0.008, u, Math.min(c1c, u + 0.1), 0.72, 2.32, k % 2 ? tom(pano, 0.86) : pano);
          painel(eixo, 0.009, c0c, c1c, 1.28, 1.34, '#E9C46A');
        }
        painel(eixo, 0.009, a0 - 0.5, a1 + 0.5, 2.33, 2.37, '#5A3B2E');
      }
    };
    const per = VIVO.periodo;
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
    decorarParedes();

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
    const luz = per === 'noite' ? 2.3 : per === 'tarde' ? 1.4 : 1;
    ILHAS.forEach(x0 => { brilhoChao(x0 + 0.4, 6.7, 0.9, '#FFD98A', 0.1 * luz); brilhoChao(x0 + 2.4, 7.4, 0.9, '#FFD98A', 0.1 * luz); });
    brilhoChao(4.2, 3.2, 1.0, '#FFE2A6', 0.09 * luz); brilhoChao(10.4, 13.0, 1.3, '#FFE2A6', 0.12 * luz);
    brilhoChao(14.6, 13.3, 1.1, '#FFD08A', 0.08 * luz); brilhoChao(2.45, 15.2, 0.6, '#8FD3F0', 0.07 * luz); // abajur do sofá e aquário
  }
  // ------------------------------------------------------------------ aconchego nas paredes (planas: fazem parte do fundo)
  function moldura(eixo, a0, a1, z0, z1, cor = '#6B4A2A') { painel(eixo, 0.003, a0 - 0.05, a1 + 0.05, z0 - 0.05, z1 + 0.05, cor); }
  function decorarParedes() {
    // sala do Atlas: diplomas e um pôster de xadrez
    for (const [y0, z0] of [[2.6, 1.55], [2.6, 1.05]]) {
      moldura('y', y0, y0 + 0.5, z0, z0 + 0.36); painel('y', 0.004, y0, y0 + 0.5, z0, z0 + 0.36, '#F4EFE0');
      painel('y', 0.005, y0 + 0.08, y0 + 0.42, z0 + 0.24, z0 + 0.27, '#8A7A5A'); painel('y', 0.005, y0 + 0.12, y0 + 0.38, z0 + 0.16, z0 + 0.18, '#B7A98A');
      painel('y', 0.005, y0 + 0.36, y0 + 0.44, z0 + 0.03, z0 + 0.11, '#C0504D');
    }
    moldura('y', 3.45, 4.35, 1.15, 2.05, '#2B2F3D');
    for (let i = 0; i < 6; i++) for (let j = 0; j < 6; j++) painel('y', 0.004, 3.5 + i * 0.133, 3.633 + i * 0.133, 1.2 + j * 0.133, 1.333 + j * 0.133, (i + j) % 2 ? '#3A3F57' : '#E9E3D3');
    painel('y', 0.005, 3.83, 3.93, 1.6, 1.8, '#1B1420'); painel('y', 0.005, 3.8, 3.96, 1.6, 1.64, '#1B1420'); // rei
    // recepção: mural de cortiça com fotos da equipe
    moldura('y', 11.0, 12.7, 1.3, 2.05); painel('y', 0.004, 11.0, 12.7, 1.3, 2.05, '#C49563');
    const fotos = ['#7EC3EB', '#F2C14E', '#9BBB59', '#E678AA', '#E07B39', '#8E6BBF', '#4F81BD'];
    fotos.forEach((cor, i) => {
      const y = 11.08 + (i % 4) * 0.4 + (i > 3 ? 0.2 : 0), z = i > 3 ? 1.38 : 1.7;
      painel('y', 0.005, y, y + 0.3, z, z + 0.3, '#FBF8F0'); painel('y', 0.006, y + 0.04, y + 0.26, z + 0.08, z + 0.26, cor);
      painel('y', 0.0065, y + 0.1, y + 0.18, z + 0.12, z + 0.2, tom(cor, 0.7)); painel('y', 0.007, y + 0.13, y + 0.16, z + 0.27, z + 0.31, '#D9534F');
    });
    // ganchos de casacos ao lado da porta: casaco, guarda-chuva e uma ecobag
    painel('y', 0.003, 14.85, 15.85, 1.55, 1.62, '#5A3B2E');
    for (const y of [15.0, 15.35, 15.7]) painel('y', 0.004, y, y + 0.04, 1.5, 1.58, '#C9CED6');
    painel('y', 0.005, 14.88, 15.18, 0.95, 1.55, '#3E6E8E'); painel('y', 0.006, 14.98, 15.08, 1.3, 1.55, '#2E5470');
    painel('y', 0.005, 15.32, 15.4, 0.7, 1.52, '#C0504D'); painel('y', 0.005, 15.28, 15.44, 0.66, 0.72, '#8E2F2C');
    painel('y', 0.005, 15.58, 15.84, 1.05, 1.5, '#E9DCC0'); painel('y', 0.006, 15.64, 15.78, 1.2, 1.3, '#9BBB59');
    // fundo: quadro abstrato acima da planta, e uma planta pendente sobre o bebedouro
    moldura('x', 6.15, 6.85, 1.3, 2.05, '#E9E3D3');
    painel('x', 0.004, 6.15, 6.85, 1.3, 2.05, '#F6EFE0');
    painel('x', 0.005, 6.22, 6.55, 1.62, 1.98, '#E07B39'); painel('x', 0.005, 6.45, 6.78, 1.38, 1.7, '#2F6E6A'); painel('x', 0.006, 6.3, 6.42, 1.4, 1.55, '#F2C14E');
    painel('x', 0.003, 16.9, 17.5, 2.2, 2.24, '#5A3B2E'); painel('x', 0.004, 17.18, 17.22, 2.0, 2.22, '#5A3B2E');
    painel('x', 0.005, 16.92, 17.48, 1.74, 2.02, '#B95A2E'); painel('x', 0.006, 16.92, 17.48, 1.97, 2.02, '#8E4322');
    for (let k = 0; k < 10; k++) { // folhas que caem do vaso
      const u = 16.88 + k * 0.065, comp = 0.12 + ruido(k, 9) * 0.38;
      painel('x', 0.007, u, u + 0.05, 1.82 - comp, 2.06, k % 2 ? COR.folha1 : COR.folha2);
      painel('x', 0.008, u - 0.015, u + 0.065, 1.82 - comp, 1.87 - comp, COR.folha3);
    }
    for (let k = 0; k < 6; k++) { const u = 16.95 + k * 0.1; painel('x', 0.009, u, u + 0.08, 2.0, 2.12, k % 2 ? COR.folha2 : COR.folha1); }
  }
  // fim de tarde: o escritório fica dourado; à noite, azulado, com poças de luz quente nas luminárias e luzinhas
  const LUZES = () => [
    ...ILHAS.flatMap(x0 => [[x0 + 0.17, 6.41, 1.15, 1.6], [x0 + 2.63, 7.59, 1.15, 1.6]]),
    [4.39, 2.82, 1.15, 1.7], [10.42, 12.62, 1.5, 2.4], [14.56, 13.56, 1.45, 2.2], [14.62, 2.3, 1.4, 1.4], [2.45, 15.7, 0.9, 1.2, '#8FD3F0'],
    [16.3, 11.4, 0.6, VIVO.tv ? 1.8 : 0, '#9FC8FF'],
  ];
  function luzDaNoite(ctx) {
    const noite = VIVO.periodo === 'noite', p = (x, y, z) => { const [a, b] = Pa(x, y, z); return [DX + a * S, DY + b * S]; };
    ctx.save();
    ctx.beginPath();
    [p(-0.22, PROF, ALT_PAREDE), p(-0.22, -0.22, ALT_PAREDE), p(LARG, -0.22, ALT_PAREDE), p(LARG, -0.22, -0.3), p(LARG, PROF, -0.3), p(-0.22, PROF, -0.3)]
      .forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.fillStyle = noite ? 'rgba(22,26,70,0.34)' : 'rgba(255,150,70,0.07)';
    ctx.fill();
    if (noite) {
      ctx.globalCompositeOperation = 'lighter';
      for (const [x, y, z, r, cor = '#FFC979'] of LUZES()) {
        if (!r) continue;
        const [cx, cy] = p(x, y, z * 0.35), raio = r * K * S;
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, raio);
        g.addColorStop(0, cor + '48'); g.addColorStop(0.5, cor + '1E'); g.addColorStop(1, cor + '00');
        ctx.fillStyle = g; ctx.beginPath(); ctx.ellipse(cx, cy, raio, raio * 0.7, 0, 0, 7); ctx.fill();
      }
    }
    ctx.restore();
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
      it.f(-1); // itens animados desenham aqui a versão mais completa: é ela que define o espaço e a ordem
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

  // ------------------------------------------------------------------ o gato (13 × 10, pés em (6, 9))
  const GATO_COR = { k: '#1B1420', C: '#9A6440', c: '#6E4229', b: '#EAD7B7', o: '#1B1420', p: '#E89BA0', g: '#7FB77E' };
  const GATO = {
    senta: ['.........k.k.', '........kCkCk', '........kCCCk', '..k.....kgCgk', '.kCk....kbpbk', '.kCk...kCCbk.', '..kCk.kCCCbk.', '...kCkCCcCbk.', '....kkCCCCCk.', '.....kkbkkbk.'],
    anda0: ['..........k.k', '.........kCkC', 'kk.......kCCC', '.kCk.....kgCg', '..kCkkkkkCbpb', '...kCCcCCcbk.', '...kCCCCCCCk.', '...kCk.kk.kCk', '...kk......kk', '.............'],
    anda1: ['..........k.k', '.........kCkC', '.kk......kCCC', 'kCk......kgCg', '.kCkkkkkkCbpb', '..kCCcCCcCbk.', '..kCCCCCCCCk.', '...kkCk.kCk..', '....kk...kk..', '.............'],
    dorme: ['.............', '.............', '.............', '....kkkkkk...', '..kkCCcCCCkk.', '.kCcCCCcCCCCk', 'kCCCCCCCCkCCk', 'kbCCCCCCkbbCk', '.kkkkkkkkkkk.', '.............'],
  };
  function desenharGato(x, y, g, alfaSombra = 0.22) {
    const [sx, sy] = Pa(x, y).map(Math.round), m = GATO[g.pose === 'anda' ? 'anda' + (g.quadro % 2) : g.pose] || GATO.senta;
    disco(sx, sy, 5, 2, '#000000', alfaSombra);
    const espelho = g.dir === 'w' || g.dir === 'n', ox = sx - 6, oy = sy - 9;
    for (let j = 0; j < m.length; j++) for (let i = 0; i < 13; i++) {
      let ch = m[j][espelho ? 12 - i : i];
      if (ch === '.' || !ch) continue;
      if (ch === 'g' && (g.pose === 'dorme' || g.piscando)) ch = 'C';
      ponto(ox + i, oy + j, GATO_COR[ch]);
    }
    if (g.pose === 'senta' && Math.floor(VIVO.t * 2) % 6 === 0) ponto(ox + (espelho ? 11 : 1), oy + 3, GATO_COR.C); // rabo balançando
  }

  // ------------------------------------------------------------------ pessoas (sprites 24 × 40, pés em (12, 38))
  const CALCAS = ['#2F3B52', '#3A3A3A', '#5A4632', '#4B5D7A', '#6B6F76', '#2E4A3E', '#283447'];
  const SAPATOS = ['#2B2B2B', '#EDEDED', '#5A3E2B', '#30405F'];
  const SW = 24, SH = 40, PX = 12, PY = 38;
  const SPRITES = new Map();
  // o: { vista: 'frente'|'costas', pose: 'pe'|'anda'|'senta', acao, quadro, humor, objeto, olhos, boca, resp }
  // acao: digita | acena | pensa | gesto | gole | mordida | folheia | controle | raquetada | espreguica
  function sprite(ap, o) {
    const chave = [ap.pele, ap.cabelo, ap.estilo, ap.camisa, ap.oculos, ap.barba, ap.calca, ap.sapato, o.vista, o.pose,
      o.acao || '', o.quadro || 0, o.humor || '', o.objeto || '', o.olhos || '', o.boca || 0, o.resp || 0].join('|');
    let s = SPRITES.get(chave);
    if (!s) { s = gerarSprite(ap, o); SPRITES.set(chave, s); }
    return s;
  }
  function gerarSprite(ap, o) {
    const { vista, pose, acao = '', quadro = 0, humor, objeto, olhos, boca, resp } = o;
    const buf = new Uint32Array(SW * SH);
    alvo(buf, SW, SH);
    const pele = ap.pele || '#EEC49C', cab = ap.cabelo || '#3B2618', est = ap.estilo || 'curto', cam = ap.camisa || '#3B6EC8';
    const calca = ap.calca || escolha((ap.cabelo || '') + cam, CALCAS), sap = ap.sapato || escolha(cam + pele, SAPATOS);
    const peleS = tom(pele, 0.84), cabS = tom(cab, 0.72), cabC = tom(cab, 1.4), camS = tom(cam, 0.76), camC = tom(cam, 1.16), calcaS = tom(calca, 0.74);
    const r = (x, y, w, h, cor) => ret(x, y, w, h, cor), p = (x, y, cor) => ponto(x, y, cor);
    const frente = vista === 'frente', sentado = pose === 'senta', anda = pose === 'anda';
    const bob = anda && (quadro === 1 || quadro === 3) ? -1 : 0;
    const sobe = resp && !anda ? -1 : 0; // respiração: ombros e cabeça sobem 1 px; as pernas ficam

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
    // tronco (a cintura fica no lugar; com a respiração o peito sobe)
    const t0 = 19 + bob + sobe, cint = 29 + bob, alt = cint - t0;
    r(8, t0, 8, 1, cam); r(7, t0 + 1, 10, alt, cam);
    r(15, t0 + 1, 2, alt, camS); r(7, t0 + 1, 1, alt - 1, camC); r(7, cint, 10, 1, camS);
    r(7, cint, 10, 1, tom(calca, 0.6)); // cinto
    if (frente) { r(11, t0, 2, 2, peleS); p(11, t0 + 2, peleS); } else r(10, t0, 4, 1, camS);
    const c0 = 6 + bob + sobe; // topo da cabeça

    // braços: A = o da esquerda da tela (iluminado), B = o da direita (na sombra)
    const bracoA = (dy = 0) => { r(5, t0 + 1 + dy, 2, 5, cam); r(5, t0 + 6 + dy, 2, 3, pele); };
    const bracoB = (dy = 0) => { r(17, t0 + 1 + dy, 2, 5, camS); r(17, t0 + 6 + dy, 2, 3, peleS); };
    const levaBoca = (desenhaObjeto) => { // braço B dobrado, mão (e o que segura) na boca
      r(5, t0 + 1, 2, 7, cam); r(5, t0 + 8, 2, 2, pele);
      r(17, t0 + 1, 2, 3, camS); r(15, t0 - 1, 3, 3, camS); r(14, t0 - 3, 2, 2, peleS);
      desenhaObjeto(13, c0 + 8);
    };
    const xicara = (x, y) => { r(x, y, 3, 3, '#FFFFFF'); r(x, y, 3, 1, '#6B4128'); p(x + 3, y + 1, '#D6D0C4'); };
    const sanduiche = (x, y) => { r(x, y, 4, 1, '#E0B26A'); r(x, y + 1, 4, 1, '#5DBA5F'); r(x, y + 2, 4, 1, '#C0504D'); r(x, y + 3, 4, 1, '#D49A55'); };
    const naFrente = frente && sentado;
    if (acao === 'acena') { // um braço para cima, a mão balançando
      bracoA();
      r(17, t0 - 4, 2, 6, camS); r(17 + (quadro % 2), t0 - 7, 3, 3, peleS);
    } else if (acao === 'espreguica') {
      r(5, t0 - 6, 2, 7, cam); r(17, t0 - 6, 2, 7, camS); r(5, t0 - 8, 2, 2, pele); r(17, t0 - 8, 2, 2, peleS);
    } else if (acao === 'pensa' && frente) { // mão no queixo
      bracoB();
      r(5, t0 + 1, 2, 4, cam); p(6, t0 + 5, cam); r(7, t0 + 3, 2, 2, cam); r(9, t0 + 1, 2, 2, cam); r(10, t0 - 1, 2, 2, cam);
      r(11, t0 - 2, 3, 2, pele);
    } else if (acao === 'gesto' && frente && !objeto) { // fala gesticulando
      bracoA();
      r(17, t0 + 1, 2, 3, camS); r(18, t0 + 3 - (quadro % 2), 3, 2, camS); r(21, t0 + 2 - (quadro % 2), 2, 2, peleS);
    } else if (acao === 'digita' && frente) { // antebraços sobre a mesa, para a frente; as mãos alternam
      const sa = quadro % 2, sb = (quadro + 1) % 2;
      r(5, t0 + 1, 2, 4, cam); r(17, t0 + 1, 2, 4, camS);
      r(14, t0 + 6 - sa, 6, 2, cam); r(20, t0 + 6 - sa, 2, 2, pele);
      r(10, t0 + 6 + sb, 6, 2, camS); r(16, t0 + 6 + sb, 2, 2, peleS);
    } else if (objeto === 'xicara' && naFrente) {
      if (acao === 'gole') levaBoca(xicara);
      else { r(5, t0 + 1, 2, 7, cam); r(5, t0 + 8, 2, 2, pele); r(17, t0 + 1, 2, 3, camS); r(14, t0 + 4, 4, 2, camS); r(14, t0 + 6, 2, 2, peleS); xicara(14, t0 + 2); }
    } else if (objeto === 'lanche' && naFrente) {
      if (acao === 'mordida') levaBoca((x, y) => sanduiche(x - 1, y));
      else { r(5, t0 + 1, 2, 4, cam); r(17, t0 + 1, 2, 4, camS); r(9, t0 + 5, 3, 2, cam); r(14, t0 + 5, 3, 2, camS); sanduiche(11, t0 + 4); p(10, t0 + 6, pele); p(15, t0 + 6, peleS); }
    } else if (objeto === 'livro' && naFrente) {
      r(5, t0 + 1, 2, 4, cam); r(17, t0 + 1, 2, 4, camS);
      r(9, t0 + 4, 8, 5, '#3E5C9A'); r(10, t0 + 4, 3, 4, '#FFFFFF'); r(13, t0 + 4, 3, 4, '#F1EEE6'); p(9, t0 + 9, pele); p(16, t0 + 9, peleS);
      for (let k = 0; k < 3; k++) { p(10 + (k % 2), t0 + 5 + k, '#B8B2A4'); p(14, t0 + 5 + k, '#B8B2A4'); }
      if (acao === 'folheia') { r(13, t0 + 1, 3, 3, '#FFFFFF'); p(13, t0 + 1, '#D6D0C4'); p(16, t0 + 2, '#D6D0C4'); }
    } else if (objeto === 'controle' && frente) { // as duas mãos no controle; os botões piscam com os dedos
      r(5, t0 + 1, 2, 4, cam); r(6, t0 + 5, 4, 2, cam); r(17, t0 + 1, 2, 4, camS); r(14, t0 + 5, 4, 2, camS);
      r(9, t0 + 5, 7, 3, '#2B2F3D'); p(9, t0 + 5, pele); p(15, t0 + 5, peleS);
      p(10, t0 + 6, quadro % 2 ? '#E5534B' : '#6C7480'); p(13, t0 + 6, quadro % 4 < 2 ? '#3FB56B' : '#6C7480'); p(14, t0 + 7, quadro % 3 ? '#4C8FD6' : '#6C7480');
    } else if (objeto === 'celular' && frente) {
      r(5, t0 + 1, 2, 4, cam); r(6, t0 + 5, 5, 2, cam); r(17, t0 + 1, 2, 3, camS); r(13, t0 + 4, 5, 2, camS);
      r(11, t0 + 3, 3, 5, '#1E2026'); r(12, t0 + 4, 1, 3, quadro % 2 ? '#7EC3EB' : '#A9DAF5'); p(11, t0 + 6, pele); p(13, t0 + 3, peleS);
    } else if (objeto === 'copo' && !anda) {
      bracoA(); r(17, t0 + 1, 2, 3, camS); r(16, t0 + 3, 3, 2, camS);
      r(acao === 'gole' ? 14 : 15, acao === 'gole' ? c0 + 9 : t0 + 1, 2, 3, '#BEE3F8'); p(acao === 'gole' ? 15 : 16, acao === 'gole' ? c0 + 9 : t0 + 1, '#7EC3EB');
    } else if (objeto === 'pasta' && !anda && frente) { // segura uma pasta do arquivo junto ao peito
      r(5, t0 + 1, 2, 4, cam); r(17, t0 + 1, 2, 4, camS); r(9, t0 + 4, 3, 2, cam); r(14, t0 + 4, 3, 2, camS);
      r(10, t0 + 2, 6, 6, '#F5B544'); r(10, t0 + 2, 6, 1, '#C98A2A'); r(11, t0 + 3, 4, 1, '#FFF3D6');
    } else {
      const bal = anda ? (quadro === 1 ? 1 : quadro === 3 ? -1 : 0) : 0;
      r(5, t0 + 1 + bal, 2, 5, cam); r(5, t0 + 6 + bal, 2, 3, pele);
      if (objeto === 'raquete') {
        if (acao === 'raquetada') { // raquete para a frente e para cima, na hora da batida
          r(17, t0 + 1, 2, 3, camS); r(18, t0 + 2, 2, 2, peleS);
          r(19, t0 - 4, 4, 5, '#D64545'); r(20, t0 - 3, 2, 3, '#E86A5C'); r(19, t0 + 1, 2, 2, '#7A4A2A');
        } else {
          r(17, t0 + 1 - bal, 2, 5, camS); r(17, t0 + 6 - bal, 2, 3, peleS);
          r(18, t0 + 4, 4, 5, '#D64545'); r(19, t0 + 5, 2, 3, '#E86A5C'); r(19, t0 + 9, 2, 2, '#7A4A2A');
        }
      } else { r(17, t0 + 1 - bal, 2, 5, camS); r(17, t0 + 6 - bal, 2, 3, peleS); }
    }
    // cabeça
    const cabeca = (cor) => { r(8, c0, 8, 1, cor); r(7, c0 + 1, 10, 1, cor); r(6, c0 + 2, 12, 9, cor); r(7, c0 + 11, 10, 1, cor); };
    if (est === 'longo') { r(5, c0 + 3, 4, 13, cabS); if (!frente) r(15, c0 + 3, 4, 13, cabS); } // cabelo comprido atrás
    if (est === 'black') { disco(12, c0 + 5, 8.5, 7.5, cab); }
    cabeca(pele);
    r(16, c0 + 3, 2, 8, peleS); r(7, c0 + 11, 10, 1, peleS);
    if (frente) {
      // orelha à esquerda; olhos e boca deslocados para a direita (rosto em 3/4)
      r(8, c0 + 5, 2, 3, peleS); p(8, c0 + 6, tom(pele, 0.7));
      const olho = '#231A16', ey = c0 + 6;
      if (olhos === 'fechado' || humor === 'calmo' || humor === 'sono') { r(11, ey + 1, 2, 1, olho); r(14, ey + 1, 2, 1, olho); }
      else { r(12, ey, 1, 2, olho); r(15, ey, 1, 2, olho); p(12, ey, '#4A3A32'); }
      if (ap.oculos) { const k = '#1E1E26'; r(11, ey - 1, 3, 1, k); r(14, ey - 1, 3, 1, k); r(11, ey + 2, 3, 1, k); r(14, ey + 2, 3, 1, k); p(11, ey, k); p(11, ey + 1, k); p(17, ey, k); p(17, ey + 1, k); p(13, ey, k); }
      const by = c0 + 9, bc = '#7A3B2E';
      if (boca) { r(13, by - 1, 2, 2, '#5A2320'); p(13, by, '#C25B4E'); } // falando
      else if (humor === 'preocupado') { p(12, by + 1, bc); r(13, by, 2, 1, bc); p(15, by + 1, bc); }
      else if (humor === 'focado' || humor === 'sono') r(13, by, 2, 1, bc);
      else if (humor === 'risada') { r(12, by - 1, 4, 2, '#5A2320'); r(12, by - 1, 4, 1, '#FFFFFF'); }
      else { p(12, by - 1, bc); r(13, by, 2, 1, bc); p(15, by - 1, bc); }
      if (humor === 'preocupado') { r(11, ey - 2, 2, 1, cabS); r(15, ey - 2, 2, 1, cabS); }
      if (ap.barba) { r(9, c0 + 8, 9, 3, cabS); r(10, c0 + 11, 7, 1, cabS); r(13, by, 2, 1, boca ? '#5A2320' : tom(pele, 0.7)); }
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
    if (objeto === 'fone') { // fone de ouvido: arco por cima e conchas nas orelhas
      r(7, c0 - 1 - (est === 'black' || est === 'topete' ? 2 : 0), 10, 1, '#30343F');
      r(6, c0 + 1, 1, 4, '#30343F'); r(17, c0 + 1, 1, 4, '#30343F');
      r(5, c0 + 4, 3, 4, '#E5534B'); r(16, c0 + 4, 3, 4, '#E5534B'); p(5, c0 + 4, '#FF8A80');
    }

    // contorno escuro em volta da silhueta
    const fora = new Uint8Array(SW * SH);
    for (let y = 0; y < SH; y++) for (let x = 0; x < SW; x++) {
      if (buf[y * SW + x] >>> 24) continue;
      if ((x > 0 && buf[y * SW + x - 1] >>> 24) || (x < SW - 1 && buf[y * SW + x + 1] >>> 24) || (y > 0 && buf[(y - 1) * SW + x] >>> 24) || (y < SH - 1 && buf[(y + 1) * SW + x] >>> 24)) fora[y * SW + x] = 1;
    }
    const k = pk(CONTORNO);
    for (let i = 0; i < fora.length; i++) if (fora[i]) buf[i] = k;
    return { buf, w: SW, h: SH, t0 };
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
    ESTADOS, INFO_LOCAL, LOCAIS_DESCANSO, NOS, ESTACOES, rota, corGrupo, CANTOS_GATO: CANTOS_GATO.length,
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
      VIVO.estacoes = d.estacoes || {}; VIVO.tv = d.tv || null; VIVO.bolo = !!d.bolo; // mudam sem redesenhar o cenário
      VIVO.periodo = d.periodo || periodoDe(agora.getHours());
      const ass = JSON.stringify([VIVO.periodo, d.fila_triagem, d.fila_analise, agora.getHours(), agora.getMinutes(),
        d.banco && d.banco.grupos.map(g => [g.codigo, g.total]),
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
    // pessoas: [{ id, x, y, ap, dir, pose, acao, quadro, humor, item, olhos, boca, resp, vapor, destaque }]
    // extra: { t: segundos, bolinha: [x, y, z] | null }
    desenhar(ctx, pessoas, extra = {}) {
      if (!frame) return [];
      VIVO.t = extra.t || 0;
      frame.set(base);
      const dyn = pessoas.map(p => {
        const espelho = p.dir === 's' || p.dir === 'w', vista = p.dir === 's' || p.dir === 'e' ? 'frente' : 'costas';
        const spr = sprite(p.ap || {}, { vista, pose: p.pose, acao: p.acao, quadro: p.quadro, humor: p.humor, objeto: p.item,
          olhos: p.olhos, boca: p.boca, resp: p.resp });
        const [ax, ay] = Pa(p.x, p.y), ox = Math.round(ax) - (espelho ? SW - 1 - PX : PX), oy = Math.round(ay) - PY;
        return { p, spr, espelho, ox, oy, box: [p.x - 0.12, p.y - 0.12, p.x + 0.12, p.y + 0.12], z0: 0, z1: 1.7, tela: [ox, oy, ox + SW - 1, oy + SH - 1], prof: p.x + p.y };
      });
      if (extra.gato) {
        const g = extra.gato, [sx, sy] = Pa(g.x, g.y);
        dyn.push({ gato: g, box: [g.x - 0.15, g.y - 0.1, g.x + 0.15, g.y + 0.1], z0: 0, z1: 0.4, tela: [sx - 7, sy - 10, sx + 7, sy + 2], prof: g.x + g.y });
      }
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
        if (e.gato) { alvo(frame, AW, AH); desenharGato(e.gato.x, e.gato.y, e.gato); return; }
        if (e.bola) {
          alvo(frame, AW, AH); disco(Math.round(e.sombraBola[0]), Math.round(e.sombraBola[1]), 2, 1, '#000000', 0.3);
          ret(e.bola[0] - 1, e.bola[1] - 1, 2, 2, '#FFF6D6'); ponto(e.bola[0], e.bola[1], '#E7B400');
          return;
        }
        const { p } = e;
        marcaChao(p.x, p.y, '#000000', 0.22);
        if (p.destaque) { marcaChao(p.x, p.y, '#F5B544', 0.55, 11, 5.5); marcaChao(p.x, p.y, '#000000', 0.12, 8, 4); }
        colar(e.spr, e.ox, e.oy, e.espelho);
        if (p.vapor) { // vapor subindo da xícara
          alvo(frame, AW, AH);
          for (let k = 0; k < 3; k++) {
            const f = (VIVO.t * 0.9 + k / 3 + p.id * 0.37) % 1, sx = 15 + Math.round(Math.sin(f * 6 + k)), sy = e.spr.t0 + 1 - Math.round(f * 7);
            ponto(e.ox + (e.espelho ? SW - 1 - sx : sx), e.oy + sy, '#FFFFFF', 0.55 * (1 - f));
          }
        }
      };
      for (let i = 0; i <= n; i++) {
        for (const e of porPos.get(i) || []) desenhaDyn(e);
        if (i < n && ITENS[i].anim) { alvo(frame, AW, AH); ITENS[i].f(VIVO.t); }
        else if (i < n && ITENS[i].spr) { const s = ITENS[i].spr; colar(s, s.ox, s.oy); }
      }
      for (const [pos, lista] of porPos) if (pos > n) lista.forEach(desenhaDyn);
      actx.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(arte, 0, 0, AW, AH, DX, DY, AW * S, AH * S);
      alvosTela = ALVOS_ARTE.map(a => ({ tipo: a.tipo, pts: a.pts.map(([x, y]) => [DX + x * S, DY + y * S]) }));
      if (extra.gato) { const [gx, gy] = Pa(extra.gato.x, extra.gato.y, 0.45); extra.gato.tela = [DX + gx * S, DY + gy * S]; }
      if (VIVO.periodo !== 'dia') luzDaNoite(ctx);
      return dyn.filter(e => e.p).map(e => {
        const [fx, fy] = Pa(e.p.x, e.p.y);
        return { id: e.p.id, sx: DX + fx * S, sy: DY + fy * S, hx: DX + fx * S, hy: DY + (fy - (PY - 4)) * S,
          x0: DX + (e.ox + 4) * S, y0: DY + (e.oy + 2) * S, x1: DX + (e.ox + SW - 4) * S, y1: DY + (e.oy + SH) * S };
      });
    },
    alvos: () => alvosTela,
    amostra: (ap, o) => sprite(ap || {}, o), // apoio a testes automatizados: um quadro de personagem
    dentro(pt, pts) {
      let ok = false;
      for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
        const [xi, yi] = pts[i], [xj, yj] = pts[j];
        if ((yi > pt[1]) !== (yj > pt[1]) && pt[0] < (xj - xi) * (pt[1] - yi) / (yj - yi) + xi) ok = !ok;
      }
      return ok;
    },
    // retrato em pixel art para os painéis: cabeça e ombros, de frente, ampliados sem suavização
    retrato(k, tam, ap, fundo = '#2A2540', o = {}) { // o: { humor, olhos, boca } (o rosto do chat pisca e fala)
      const s = sprite(ap || {}, { vista: 'frente', pose: 'pe', humor: o.humor || 'feliz', olhos: o.olhos, boca: o.boca }), c = document.createElement('canvas');
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
