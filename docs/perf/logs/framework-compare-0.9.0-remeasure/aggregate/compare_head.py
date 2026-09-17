#!/usr/bin/env python3
"""0.8.0（前版スコアボード）と HEAD 再計測の判定を並べる。gen.py と同じ判定規則。"""
import json, re, sys
from pathlib import Path
REPO = Path('<repo>')
RAW = REPO / 'scripts/bench/framework-compare/results/raw'
DGX = REPO / 'docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx'
SB = Path(__file__).resolve().parent.parent / 'scoreboard'
src = SB.joinpath('gen.py').read_text()
ns = {'m4': {}}
exec(src[src.index('def gemm_ms'):src.index('M4_SKIP =')], ns)   # gemm_ms + M4_PY（転記値）→ ns['m4']
M4_PY = ns['m4']
FW = ['candle', 'burn', 'pytorch', 'tensorflow', 'scipy']

def load(p):
    p = Path(p)
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []
def index(rows):
    d = {}
    for r in rows:
        if r['task'] in ('train_phases', 'infer_phases'): continue
        d[(r['framework'], r['task'], r['device'], int(r['size']), r['mode'])] = r
    return d
def metric(r, task): return (1.0 / r['median_s']) if task == 'infer' else r['median_s']
def better(a, b, task): return a > b if task == 'infer' else a < b

def judge(data, task, dev, size):
    me = data.get(('fandhe-ai', task, dev, size, 'reuse'))
    if me is None: return None
    mine = metric(me, task); best = None
    for fw in FW:
        r = data.get((fw, task, dev, size, 'fresh'))
        if r is None or (r.get('parity_fail_count') or 0) > 0: continue
        v = metric(r, task)
        if best is None or better(v, best[1], task): best = (fw, v)
    n_better = sum(1 for fw in FW if (r := data.get((fw, task, dev, size, 'fresh'))) and not (r.get('parity_fail_count') or 0) and better(metric(r, task), mine, task))
    rank = 1 + n_better
    ratio = (mine / best[1]) if task == 'infer' else (best[1] / mine)
    verdict = '1位' if rank == 1 else ('僅差' if ratio >= 0.90 else f'{rank}位')
    return dict(verdict=verdict, ratio=ratio, best=best[0], me_ms=me['median_s'] * 1e3, me_fail=me.get('parity_fail_count') or 0)

M4_ROWS = [('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)] + \
          [('train', 'metal', 64), ('infer', 'metal', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]
GB_ROWS = [('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)] + \
          [('train', 'cuda', 64), ('infer', 'cuda', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]

m4_old = index(load(RAW / 'results-m4max-0.8.0.jsonl')); m4_old.update(M4_PY)
gb_old = index(load(DGX / 'results-dgx-0.8.0.jsonl') + load(DGX / 'results-dgx-0.8.0-extra.jsonl') + load(DGX / 'results-dgx-py-0.8.0.jsonl'))
m4_new = index(load(sys.argv[1])) if len(sys.argv) > 1 else {}
m4_new.update(M4_PY)  # M4 Python 3 FW は再計測環境なし・前版転記のまま
gb_new = index(sum((load(p) for p in sys.argv[2:]), [])) if len(sys.argv) > 2 else {}

def show(machine, rows, old, new):
    print(f'\n### {machine}')
    print('| セル | 0.8.0 | HEAD | fandhe ms 0.8.0 → HEAD | 変化 |'); print('|---|---|---|---|---|')
    t = {'w': 0, 'n': 0, 'l': 0}
    for task, dev, size in rows:
        a = judge(old, task, dev, size); b = judge(new, task, dev, size)
        lab = f'{task} {dev}' + (f' N={size}' if task == 'gemm' else '')
        if b is None: print(f'| {lab} | {a["verdict"]} {a["ratio"]:.2f}× vs {a["best"]} | （未計測） | | |'); continue
        rk = lambda v: 1 if v == '1位' else 1.5 if v == '僅差' else int(v[:-1])
        chg = '' if a['verdict'] == b['verdict'] else ('↑' if rk(b['verdict']) < rk(a['verdict']) else '↓')
        t['w' if b['verdict'] == '1位' else 'n' if b['verdict'] == '僅差' else 'l'] += 1
        ff = f' ⚠fail={b["me_fail"]}' if b['me_fail'] else ''
        print(f'| {lab} | {a["verdict"]} {a["ratio"]:.2f}× vs {a["best"]} | {b["verdict"]} {b["ratio"]:.2f}× vs {b["best"]}{ff} | {a["me_ms"]:.3f} → {b["me_ms"]:.3f} ({b["me_ms"]/a["me_ms"]:.2f}×) | {chg} |')
    print(f'HEAD 集計: 1位 {t["w"]} 僅差 {t["n"]} 負け {t["l"]}')
show('M4 Max', M4_ROWS, m4_old, m4_new)
show('GB10', GB_ROWS, gb_old, gb_new)
