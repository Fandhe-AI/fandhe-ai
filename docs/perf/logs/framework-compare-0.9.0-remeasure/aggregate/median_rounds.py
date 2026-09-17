#!/usr/bin/env python3
"""複数 round の JSONL からセルごとに median_s の中央値 round の行を選び 1 本の JSONL へ集約する。

出力行には 2 種類の比が併存するので混同しない（PR #2023 codex-review 指摘）:
- 行の選択（`median_s` の中央値 round）は framework ごとに独立に行う。したがって出力行同士から
  算出する比（例: candle fresh 行 / fandhe-ai reuse 行）は「別々に選んだ中央値同士の比」であり、
  RULE.txt が主判定とする「同一 run 内の対戦相手比」ではない。scoreboard/gen_090.py・
  compare_head.py はこの「中央値同士の比」を用いる（参考値）。
- `gemm`・`mode=reuse` の fandhe-ai 行には、同一 run 内で `candle fresh median_s / fandhe-ai reuse
  median_s` を run ごとに計算した `candle_ratio_runs`（run 順）と、その中央値 `candle_ratio_median`
  を付与する。candle 比ゲートの追補（docs/perf/{cpu,metal}-gemm-candle-gate-remeasurement.md の
  #1967 節）はこの `candle_ratio_median` を主判定に用いる。
"""
import json, sys, statistics
from collections import defaultdict
out = sys.argv[1]; files = sys.argv[2:]
cells = defaultdict(list)
per_run = []  # run ごとの索引（同一 run 内比の算出用。files の順序 = run 順）
for f in files:
    idx = {}
    for l in open(f):
        if not l.strip(): continue
        r = json.loads(l)
        if r['task'] in ('train_phases', 'infer_phases'): continue
        key = (r['framework'], r['task'], r['device'], int(r['size']), r['mode'])
        cells[key].append(r)
        idx[key] = r
    per_run.append(idx)
with open(out, 'w') as w:
    for k, rs in sorted(cells.items()):
        rs.sort(key=lambda r: r['median_s'])
        pick = rs[len(rs) // 2] if len(rs) % 2 else rs[len(rs) // 2 - 1]  # 偶数なら小さい側（保守的）
        pick = dict(pick); pick['rounds'] = len(rs)
        pick['round_spread'] = (rs[-1]['median_s'] / rs[0]['median_s']) if rs[0]['median_s'] else None
        pick['checksums_agree'] = len({r.get('checksum') for r in rs}) == 1
        fw, task, dev, size, mode = k
        if fw == 'fandhe-ai' and task == 'gemm' and mode == 'reuse':
            ratios = []
            for idx in per_run:
                me = idx.get(k); c = idx.get(('candle', task, dev, size, 'fresh'))
                if me and c and me['median_s']:
                    ratios.append(c['median_s'] / me['median_s'])
            if ratios:
                pick['candle_ratio_runs'] = [round(x, 6) for x in ratios]
                pick['candle_ratio_median'] = round(statistics.median(ratios), 6)
        w.write(json.dumps(pick) + '\n')
print(f'{out}: {len(cells)} cells, rounds per cell: {sorted({len(v) for v in cells.values()})}')
