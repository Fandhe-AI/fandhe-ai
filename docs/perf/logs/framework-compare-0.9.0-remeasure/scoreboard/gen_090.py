#!/usr/bin/env python3
"""fandhe-ai 0.9.0 対戦成績スコアボード（HTML）を framework-compare JSONL から生成する。

gen.py（0.8.0 版・0.6.0 比）の改変版。0.9.0 の結果を、直前版 0.8.0 と比較する。

- M4 Max: --m4（fandhe/candle/burn・2026-09-16・5 ラウンド中央値）＋ Python 3 FW（PyTorch/
  TensorFlow/SciPy）は 0.9.0 でも未再計測のため M4_PY（2026-09-12 転記値）をそのまま流用。
- GB10: --gb ＋ --gb-extra（fandhe-ai/candle/burn・2026-09-16 UTC・専有 1 セッション）＋
  --gb-py（Python 3 FW。0.9.0 では未再計測のため 2026-09-16 計測の HEAD 参考系列を流用）。
- 0.8.0 比: --m4-prev（既定 results-m4max-0.8.0.jsonl）／--gb-prev（既定 0.8.0 の
  results-dgx-0.8.0{,-extra,-py-0.8.0}.jsonl 3 本）。
"""
import argparse
import json
import html
from pathlib import Path

REPO = Path('<repo>')
RAW = REPO / 'scripts/bench/framework-compare/results/raw'
DGX08 = REPO / 'docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx'

PREV_VER = '0.8.0'  # 比較元バージョン（この版との差分を「{PREV_VER} 比」として表示する）


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m4', required=True, help='0.9.0 M4 Max（fandhe/candle/burn。5 ラウンド中央値）JSONL')
    p.add_argument('--m4-prev', default=str(RAW / 'results-m4max-0.8.0.jsonl'), help='0.8.0 M4 Max JSONL')
    p.add_argument('--gb', required=True, help='0.9.0 GB10（fandhe/candle/burn 本体）JSONL')
    p.add_argument('--gb-extra', required=True, help='0.9.0 GB10 追加計測（CPU GEMM reuse 等）JSONL')
    p.add_argument('--gb-py', required=True, help='0.9.0 GB10 Python 3 FW（PyTorch/TensorFlow/SciPy）JSONL')
    p.add_argument('--gb-prev', nargs='+', default=[
        str(DGX08 / 'results-dgx-0.8.0.jsonl'),
        str(DGX08 / 'results-dgx-0.8.0-extra.jsonl'),
        str(DGX08 / 'results-dgx-py-0.8.0.jsonl'),
    ], help='0.8.0 GB10 JSONL 群（複数可。既定は 0.8.0 の本体＋extra＋py の 3 本）')
    p.add_argument('--out', default='fandhe-ai-0.9.0-scoreboard.html', help='出力 HTML パス')
    p.add_argument('--m4-load', default='', help='M4 Max の負荷条件注記文（計測条件節に埋め込む）')
    p.add_argument('--gb-load', default='', help='GB10 の負荷条件注記文（計測条件節に埋め込む）')
    return p.parse_args()


ARGS = parse_args()


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def key(r):
    return (r['framework'], r['task'], r['device'], int(r['size']), r['mode'])


def index(rows):
    d = {}
    for r in rows:
        if r['task'] in ('train_phases', 'infer_phases'):
            continue
        d[key(r)] = r  # 後勝ち（追記分を優先）
    return d


m4 = index(load(ARGS.m4))
m4_prev = index(load(ARGS.m4_prev))
gb = index(load(ARGS.gb) + load(ARGS.gb_extra) + load(ARGS.gb_py))
gb_prev_rows = []
for f in ARGS.gb_prev:
    gb_prev_rows += load(f)
gb_prev = index(gb_prev_rows)


def gemm_ms(n, gf):
    return 2.0 * n ** 3 / (gf * 1e9) * 1e3


# 2026-09-12 ページからの転記（M4 Max・Python 3 FW。元 JSONL はリポジトリ未収録）。
# 0.9.0 でも Python 3 FW（PyTorch/TensorFlow/SciPy）は再計測していないため、0.8.0 版と
# 同じ転記値をそのまま流用する（fandhe-ai 側のみが変わるフレームワーク横並び比較のため、
# 比較対象 FW 自体の数値は版が進んでも変わらない）。
# gemm は GFLOP/s から ms を逆算（表示 2 桁より精度が高い）、infer は µs から。
M4_PY = {
    'pytorch': {'version': '2.14.0', 'gemm': {'metal': {256: 73, 512: 315, 1024: 705, 2048: 1170, 4096: 2663},
                                              'cpu': {256: 207, 512: 414, 1024: 749, 2048: 1142}},
                'train': {'metal': 0.40, 'cpu': 0.20}, 'infer_us': {'metal': 295, 'cpu': 33}, 'resc': {}},
    'tensorflow': {'version': '2.16.2', 'gemm': {'metal': {256: 60, 512: 250, 1024: 533, 2048: 1080, 4096: 2530},
                                                 'cpu': {256: 129, 512: 282, 1024: 471, 2048: 699}},
                   'train': {'metal': 1.44, 'cpu': 0.96}, 'infer_us': {'metal': 665, 'cpu': 266},
                   'resc': {('gemm', 'cpu', 2048): 4}},
    'scipy': {'version': '1.18.1', 'gemm': {'cpu': {256: 121, 512: 227, 1024: 372, 2048: 470}},
              'train': {'cpu': 0.31}, 'infer_us': {'cpu': 164}, 'resc': {}},
}
for fw, d in M4_PY.items():
    for dev, sizes in d['gemm'].items():
        for n, gf in sizes.items():
            m4[(fw, 'gemm', dev, n, 'fresh')] = {'framework': fw, 'version': d['version'], 'task': 'gemm', 'device': dev, 'size': n, 'mode': 'fresh',
                                                 'median_s': gemm_ms(n, gf) / 1e3, 'gflops': gf, 'parity_fail_count': 0,
                                                 'parity_scaled_abs_rescued': d['resc'].get(('gemm', dev, n), 0), 'transcribed': True}
    for dev, ms in d['train'].items():
        m4[(fw, 'train', dev, 64, 'fresh')] = {'framework': fw, 'version': d['version'], 'task': 'train', 'device': dev, 'size': 64, 'mode': 'fresh', 'median_s': ms / 1e3, 'transcribed': True}
    for dev, us in d['infer_us'].items():
        m4[(fw, 'infer', dev, 64, 'fresh')] = {'framework': fw, 'version': d['version'], 'task': 'infer', 'device': dev, 'size': 64, 'mode': 'fresh', 'median_s': us / 1e6, 'transcribed': True}

# M4 Max burn Metal N>=512 は結果テンソル全ゼロで記録拒否（skipped-m4max-0.9.0.log。0.8.0 と同一の
# upstream 既知バグによる継続現象）
M4_SKIP = {('burn', 'gemm', 'metal', n): '結果テンソルが全ゼロ（upstream 既知バグ）のため記録拒否' for n in (512, 1024, 2048, 4096)}

FW_ORDER = ['candle', 'burn', 'pytorch', 'tensorflow', 'scipy']

def fmt_ms(ms):
    if ms >= 100: return f'{ms:,.2f}'
    if ms < 0.1: return f'{ms:.3f}'
    return f'{ms:.2f}'

def cell_value(r, task, size):
    """(表示値, 有効か, 注記) を返す。"""
    ms = r['median_s'] * 1e3
    fail = r.get('parity_fail_count') or 0
    resc = r.get('parity_scaled_abs_rescued') or 0
    valid = fail == 0
    if task == 'gemm':
        gf = r.get('gflops') or (2.0 * size ** 3 / r['median_s'] / 1e9)
        txt = f'{fmt_ms(ms)} <small>/ {gf:,.0f} GF</small>'
    elif task == 'train':
        txt = f'{fmt_ms(ms)} <small>ms</small>'
    else:
        tp = 1.0 / r['median_s']
        txt = f'{tp:,.0f} <small>/ {ms*1e3:,.0f} µs</small>'
    return txt, valid, fail, resc

def metric(r, task):
    """比較用スカラ。時間は小さいほど良い・推論はスループット大きいほど良い。"""
    return (1.0 / r['median_s']) if task == 'infer' else r['median_s']

def better(a, b, task):
    return a > b if task == 'infer' else a < b

def build_row(data, data_prev, machine, task, device, size, skip):
    label = {'gemm': f'gemm {DEVNAME[device]} N={size}', 'train': f'train {DEVNAME[device]}', 'infer': f'infer {DEVNAME[device]}'}[task]
    desc = {'gemm': 'f32 正方 GEMM・reuse', 'train': '784→256→10 MLP・バッチ 64・1 step・reuse（デバイス常駐 SGD）', 'infer': '同 MLP forward・バッチ 64・reuse'}[task]
    me = data.get(('fandhe-ai', task, device, size, 'reuse'))
    me_fresh = data.get(('fandhe-ai', task, device, size, 'fresh'))
    assert me is not None, (machine, task, device, size)
    mine = metric(me, task)
    comp = []  # (fw, r, valid)
    best = None
    cells = []
    for fw in FW_ORDER:
        r = data.get((fw, task, device, size, 'fresh'))
        if r is None:
            reason = skip.get((fw, task, device, size))
            title = f' title="{html.escape(reason)}"' if reason else ''
            cells.append((fw, f'<td class="num"><span class="na"{title}>—</span></td>', None))
            continue
        txt, valid, fail, resc = cell_value(r, task, size)
        if not valid:
            tot = r.get('parity_total')
            cells.append((fw, f'<td class="num inv" title="要素検証超過 {fail:,}/{tot:,}（判定不能・順位に含めない）">{txt}</td>', None))
            continue
        mark = ''
        if resc:
            mark = f' <span class="resc" title="スケール付き絶対誤差で救済 {resc:,} 要素（#1241 承認契約・有効セル扱い）">△{resc:,}</span>'
        v = metric(r, task)
        comp.append((fw, v))
        if best is None or better(v, best[1], task):
            best = (fw, v)
        cells.append((fw, txt + mark, v))
    # 順位
    n_better = sum(1 for fw, v in comp if better(v, mine, task))
    rank = 1 + n_better
    if task == 'infer':
        ratio = mine / best[1]
    else:
        ratio = best[1] / mine
    if rank == 1:
        verdict, cls = '1 位', 'win'
    elif ratio >= 0.90:
        verdict, cls = '僅差', 'near'
    else:
        verdict, cls = f'{rank} 位', 'loss'
    # {PREV_VER} 比
    old = data_prev.get(('fandhe-ai', task, device, size, 'reuse'))
    fresh_note = ''
    if old is None:
        old = data_prev.get(('fandhe-ai', task, device, size, 'fresh'))
        base = me_fresh if me_fresh is not None else me
        fresh_note = '（fresh 同士）'
    else:
        base = me
    if old is not None:
        vprev = (metric(base, task) / metric(old, task)) if task == 'infer' else (base['median_s'] / old['median_s'])
        vprevs = f'{PREV_VER} 比 {vprev:.2f}×{fresh_note}'
    else:
        vprev, vprevs = None, f'{PREV_VER} 記録なし'
    # bar
    if ratio < 1:
        bar = f'<div class="bar"><i class="down" style="width:{ratio*100:.0f}%"></i></div>'
    else:
        w = min(100, max(10, (ratio - 1) * 100))
        bar = f'<div class="bar"><i class="up" style="width:{w:.0f}%"></i></div>'
    me_txt = cell_value(me, task, size)[0]
    fresh_txt = cell_value(me_fresh, task, size)[0] if me_fresh else '<span class="na">—</span>'
    cells_html = ''.join(
        (f'<td class="num{" best" if best and fw == best[0] else ""}">{c}</td>' if v is not None else c)
        for fw, c, v in cells)
    tr = (f'<tr class="v-{cls}"><th scope="row"><span class="ph">{label}</span><span class="ds">{desc}</span></th>'
          f'<td><span class="pill {cls}">{verdict}</span><span class="vs">vs {FWNAME[best[0]]}</span><span class="vs">{vprevs}</span></td>'
          f'<td class="ratio">{bar}<span class="rt">{ratio:.2f}×</span></td>'
          f'<td class="num me">{me_txt} <small>reuse</small></td>'
          f'<td class="num me2">{fresh_txt}{" <small>fresh</small>" if me_fresh else ""}</td>'
          f'{cells_html}</tr>')
    return dict(tr=tr, cls=cls, verdict=verdict, ratio=ratio, label=label, machine=machine, best=best[0], vprev=vprev, vprevs=vprevs,
                me=me, me_fresh=me_fresh, old=old, fresh_note=fresh_note)

DEVNAME = {'metal': 'Metal', 'cpu': 'CPU', 'cuda': 'CUDA'}
FWNAME = {'candle': 'candle', 'burn': 'burn', 'pytorch': 'PyTorch', 'tensorflow': 'TensorFlow', 'scipy': 'SciPy'}

M4_ROWS = [('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)] + \
          [('train', 'metal', 64), ('infer', 'metal', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]
GB_ROWS = [('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)] + \
          [('train', 'cuda', 64), ('infer', 'cuda', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]

m4_rows = [build_row(m4, m4_prev, 'M4 Max', *r, M4_SKIP) for r in M4_ROWS]
gb_rows = [build_row(gb, gb_prev, 'GB10', *r, {}) for r in GB_ROWS]
allrows = m4_rows + gb_rows
wins = [r for r in allrows if r['cls'] == 'win']
nears = [r for r in allrows if r['cls'] == 'near']
losses = [r for r in allrows if r['cls'] == 'loss']
# 無効セル（判定不能）数
def count_invalid(data, rows):
    n = 0
    for task, dev, size in rows:
        for fw in FW_ORDER:
            r = data.get((fw, task, dev, size, 'fresh'))
            if r is not None and (r.get('parity_fail_count') or 0) > 0:
                n += 1
    return n
invalid_n = count_invalid(m4, M4_ROWS) + count_invalid(gb, GB_ROWS)

def head(cols):
    return '<thead><tr><th>対象</th><th>勝敗</th><th>最速他 FW ÷ fandhe-ai</th><th>fandhe-ai 0.9.0（判定）</th><th>fandhe-ai 別モード（参考）</th>' + ''.join(f'<th>{c}</th>' for c in cols) + '</tr></thead>'

def lst(rows):
    return '、'.join(f'{r["machine"]} {r["label"]}（{r["ratio"]:.2f}×）' for r in rows)

def msprev(data, data_prev, task, dev, size):
    """{PREV_VER}→0.9.0 の同一モード（reuse 優先）中央値ペアを返す。"""
    old = data_prev.get(('fandhe-ai', task, dev, size, 'reuse'))
    mode = 'reuse'
    if old is None:
        old = data_prev.get(('fandhe-ai', task, dev, size, 'fresh')); mode = 'fresh'
    new = data.get(('fandhe-ai', task, dev, size, mode))
    return old, new, mode

def card_gemm(data, data_prev, dev, sizes, title):
    parts = []; ratios = []
    for n in sizes:
        old, new, mode = msprev(data, data_prev, 'gemm', dev, n)
        if old is None or new is None: continue
        parts.append(f'N={n} {old["median_s"]*1e3:.2f}→{new["median_s"]*1e3:.2f} ms')
        ratios.append(new['median_s'] / old['median_s'])
    mode_lbl = mode
    return f'<div><h3>{title}（{mode_lbl}）</h3><span class="v">{PREV_VER} 比 {min(ratios):.2f}〜{max(ratios):.2f}（&lt;1 が高速化）</span><p>{"・".join(parts)}</p></div>'

def card_train(data, data_prev, devs, title):
    parts = []; ratios = []
    for dev in devs:
        old, new, mode = msprev(data, data_prev, 'train', dev, 64)
        parts.append(f'{DEVNAME[dev]} {old["median_s"]*1e3:.2f}→{new["median_s"]*1e3:.2f} ms（{mode}）')
        ratios.append(new['median_s'] / old['median_s'])
    return f'<div><h3>{title}</h3><span class="v">{PREV_VER} 比 {min(ratios):.2f}〜{max(ratios):.2f}（&lt;1 が高速化）</span><p>{"・".join(parts)}</p></div>'

def card_infer(data, data_prev, devs, title):
    parts = []; ratios = []
    for dev in devs:
        old, new, mode = msprev(data, data_prev, 'infer', dev, 64)
        parts.append(f'{DEVNAME[dev]} {1/old["median_s"]:,.0f}→{1/new["median_s"]:,.0f} /s（{mode}）')
        ratios.append(old['median_s'] / new['median_s'])
    return f'<div><h3>{title}</h3><span class="v">{PREV_VER} 比 {min(ratios):.2f}〜{max(ratios):.2f}（&gt;1 が改善）</span><p>{"・".join(parts)}</p></div>'

CSS = open(Path(__file__).with_name('style.css')).read()
BODY = open(Path(__file__).with_name('body_090.html')).read()

out = BODY.format(
    css=CSS,
    n_win=len(wins), n_near=len(nears), n_loss=len(losses), n_invalid=invalid_n, n_total=len(allrows),
    m4_head=head(['candle-core 0.11.0', 'burn 0.21.0', 'PyTorch 2.14.0（MPS）', 'TensorFlow 2.16.2（Metal プラグイン）', 'SciPy 1.18.1']),
    m4_rows=''.join(r['tr'] for r in m4_rows),
    gb_head=head(['candle-core 0.11.0', 'burn 0.21.0', 'PyTorch 2.14.0+cu130', 'TensorFlow 2.21.0（CPU のみ）', 'SciPy 1.18.1']),
    gb_rows=''.join(r['tr'] for r in gb_rows),
    win_list=lst(wins), near_list=lst(nears), loss_list=lst(losses),
    m4_load=ARGS.m4_load, gb_load=ARGS.gb_load,
    m4_cards=card_gemm(m4, m4_prev, 'metal', (256, 512, 1024, 2048, 4096), 'M4 Max GEMM Metal') + card_gemm(m4, m4_prev, 'cpu', (256, 512, 1024, 2048), 'M4 Max GEMM CPU')
             + card_train(m4, m4_prev, ('metal', 'cpu'), 'M4 Max MLP 学習 1 step') + card_infer(m4, m4_prev, ('metal', 'cpu'), 'M4 Max 推論スループット'),
    gb_cards=card_gemm(gb, gb_prev, 'cuda', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CUDA') + card_gemm(gb, gb_prev, 'cpu', (256, 512, 1024, 2048, 4096), 'GB10 GEMM CPU')
             + card_train(gb, gb_prev, ('cuda', 'cpu'), 'GB10 MLP 学習 1 step') + card_infer(gb, gb_prev, ('cuda', 'cpu'), 'GB10 推論スループット'),
)
Path(ARGS.out).write_text(out)

# 検証出力
for r in allrows:
    print(f"{r['machine']:7} {r['label']:22} {r['verdict']:4} vs {r['best']:10} {r['ratio']:.2f}x  {r['vprevs']}")
print('tally', len(wins), len(nears), len(losses), 'invalid', invalid_n, 'total', len(allrows))
