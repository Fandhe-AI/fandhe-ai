#!/usr/bin/env python3
"""複数 round の JSONL からセルごとに median_s の中央値 round の行を選び 1 本の JSONL へ集約する。"""
import json, sys, statistics
from collections import defaultdict
out = sys.argv[1]; files = sys.argv[2:]
cells = defaultdict(list)
for f in files:
    for l in open(f):
        if not l.strip(): continue
        r = json.loads(l)
        if r['task'] in ('train_phases', 'infer_phases'): continue
        cells[(r['framework'], r['task'], r['device'], int(r['size']), r['mode'])].append(r)
with open(out, 'w') as w:
    for k, rs in sorted(cells.items()):
        rs.sort(key=lambda r: r['median_s'])
        pick = rs[len(rs) // 2] if len(rs) % 2 else rs[len(rs) // 2 - 1]  # 偶数なら小さい側（保守的）
        pick = dict(pick); pick['rounds'] = len(rs)
        pick['round_spread'] = (rs[-1]['median_s'] / rs[0]['median_s']) if rs[0]['median_s'] else None
        pick['checksums_agree'] = len({r.get('checksum') for r in rs}) == 1
        w.write(json.dumps(pick) + '\n')
print(f'{out}: {len(cells)} cells, rounds per cell: {sorted({len(v) for v in cells.values()})}')
