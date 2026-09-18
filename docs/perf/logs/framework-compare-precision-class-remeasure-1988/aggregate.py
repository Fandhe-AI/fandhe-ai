#!/usr/bin/env python3
"""イシュー #1988: 精度クラス反映後の GB10 5 run（run{1..5}/results.jsonl・py.jsonl）を RULE.txt の規則で集計する。

python3 標準ライブラリのみ。出力:
- aggregate.md: 対象 6 セルの run 別 parity_fail_count／rescued／bound／median_s・判定（解消／残存）・決定性・
  採用 run（median_s の中央値 run）・同一 run 内比（candle fresh／burn fresh ÷ fandhe-ai reuse）・専有ゲート結果
- --base-gb／--base-py が与えられた場合、採用行を末尾へ追記した派生 JSONL（--out-gb／--out-py）を書く
  （gen_1988.py の index は後勝ちなので、追記行が 0.9.0 の同キー行を置き換える）

使い方: python3 aggregate.py --logs <dir> [--base-gb ... --out-gb ... --base-py ... --out-py ...] [--md aggregate.md]
        python3 aggregate.py --self-test
"""
import argparse, json, os, statistics, sys, tempfile

SIZES = (256, 512, 1024, 2048, 4096)
TARGET_BURN = [('burn', 'gemm', 'cuda', n, 'fresh') for n in SIZES]
TARGET_PY = ('pytorch', 'gemm', 'cpu', 4096, 'fresh')
EXPECTED_RESCUED = {256: 10538, 512: 42361, 1024: 169929, 2048: 681454, 4096: 2729050}  # 検算用（合否条件ではない）。u の比は 2^-11 / 2^-24 = 2^13 = 8192（RULE.txt の「2048 倍」は誤記。README 参照）
EXPECTED_BOUND_090 = {256: 1.907255e-6, 512: 3.814663e-6, 1024: 7.629375e-6, 2048: 1.525878e-5, 4096: 3.051757e-5}


def key(r):
    return (r['framework'], r['task'], r['device'], int(r['size']), r.get('mode', 'fresh'))


def load_run(path):
    idx = {}
    if not os.path.exists(path):
        return idx
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            idx[key(r)] = (r, line)
    return idx


def pick_median(rows):
    """median_s の中央値 run の (run_no, row, line) を返す（median_rounds.py と同じ規則。偶数なら小さい側）。"""
    rs = sorted(rows, key=lambda t: t[1]['median_s'])
    return rs[len(rs) // 2] if len(rs) % 2 else rs[len(rs) // 2 - 1]


def verdict(rows):
    fails = [int(r.get('parity_fail_count') or 0) for _, r, _ in rows]
    if len(rows) == 5 and all(f == 0 for f in fails):
        return '解消'
    return '残存'


def parse_gate(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        parts = line.split()
        if len(parts) >= 5 and parts[0].startswith('run='):
            r = int(parts[0][4:])
            out[r] = (parts[-1], parts[2], parts[3])  # (pass|wait|fail, load1=, gpu_util=)
    return out


def aggregate(logs):
    runs = []
    for i in range(1, 6):
        idx = load_run(os.path.join(logs, f'run{i}', 'results.jsonl'))
        idx.update(load_run(os.path.join(logs, f'run{i}', 'py.jsonl')))
        runs.append((i, idx))
    gate = parse_gate(os.path.join(logs, 'load_gate.log'))
    cells = {}
    for k in TARGET_BURN + [TARGET_PY]:
        rows = [(i, idx[k][0], idx[k][1]) for i, idx in runs if k in idx]
        cells[k] = rows
    # 同一 run 内比
    ratios = {}
    for n in SIZES:
        for i, idx in runs:
            me = idx.get(('fandhe-ai', 'gemm', 'cuda', n, 'reuse'))
            for fw in ('candle', 'burn'):
                o = idx.get((fw, 'gemm', 'cuda', n, 'fresh'))
                if me and o and me[0]['median_s']:
                    ratios.setdefault((n, fw), []).append(o[0]['median_s'] / me[0]['median_s'])
    return runs, gate, cells, ratios


def fmt_bound(x):
    return f'{x:.6e}'


def render(runs, gate, cells, ratios):
    L = ['# #1988 集計（RULE.txt 規則）', '']
    L.append('## 専有ゲート（load1 < 1.0 かつ gpu_util 0%）')
    L.append('')
    L.append('| run | 判定 | load1 | gpu_util |')
    L.append('|---|---|---|---|')
    for i, _ in runs:
        g = gate.get(i)
        L.append(f'| {i} | {g[0] if g else "記録なし"} | {g[1][6:] if g else "-"} | {g[2][9:] if g else "-"} |')
    L.append('')
    L.append('## 対象 6 セル（run 別 parity_fail_count／rescued／bound／median_s）')
    L.append('')
    L.append('| セル | run | parity_fail_count | rescued | bound | median_s | 採用 |')
    L.append('|---|---|---|---|---|---|---|')
    summary = []
    for k, rows in cells.items():
        fw, _, dev, n, _ = k
        name = f'{fw} {dev} gemm N={n}'
        pick = pick_median(rows) if rows else None
        for i, r, _ in rows:
            L.append(f'| {name} | {i} | {int(r.get("parity_fail_count") or 0):,} | {int(r.get("parity_scaled_abs_rescued") or 0):,} | '
                     f'{fmt_bound(float(r.get("parity_scaled_abs_bound") or 0))} | {r["median_s"]:.6f} | {"採用" if pick and pick[0] == i else ""} |')
        v = verdict(rows)
        det_r = len({int(r.get('parity_scaled_abs_rescued') or 0) for _, r, _ in rows}) == 1 if rows else False
        det_b = len({float(r.get('parity_scaled_abs_bound') or 0) for _, r, _ in rows}) == 1 if rows else False
        note = ''
        if fw == 'burn' and rows:
            resc = int(rows[0][1].get('parity_scaled_abs_rescued') or 0)
            b = float(rows[0][1].get('parity_scaled_abs_bound') or 0)
            ratio = b / EXPECTED_BOUND_090[n]
            note = (f'rescued 期待値 {EXPECTED_RESCUED[n]:,} と{"一致" if resc == EXPECTED_RESCUED[n] else "相違"}・'
                    f'bound は 0.9.0 記録の {ratio:.1f} 倍')
        elif fw == 'pytorch' and rows:
            fails = sorted({int(r.get('parity_fail_count') or 0) for _, r, _ in rows})
            note = f'fail_count={fails}（T1 現状維持: spec REQ-2 (b-1) 上正当な判定不能・係数不変）'
        summary.append((name, v, len(rows), det_r and det_b, pick[0] if pick else None, note))
    L.append('')
    L.append('## セル判定')
    L.append('')
    L.append('| セル | 判定 | run 数 | 決定性（rescued・bound が 5 run 同値） | 採用 run | 備考 |')
    L.append('|---|---|---|---|---|---|')
    for name, v, nr, det, pr, note in summary:
        L.append(f'| {name} | **{v}** | {nr} | {"同値" if det else "相違"} | {pr} | {note} |')
    L.append('')
    L.append('## 同一 run 内比（他 FW fresh median_s ÷ fandhe-ai reuse median_s。>1 で fandhe-ai が高速。参考記録）')
    L.append('')
    L.append('| N | 相手 | run1 | run2 | run3 | run4 | run5 | 中央値 |')
    L.append('|---|---|---|---|---|---|---|---|')
    for n in SIZES:
        for fw in ('candle', 'burn'):
            rs = ratios.get((n, fw), [])
            cellsr = ' | '.join(f'{x:.4f}' for x in rs) + ' | ' * (5 - len(rs))
            med = f'{statistics.median(rs):.4f}' if rs else '-'
            L.append(f'| {n} | {fw} | {cellsr} | {med} |')
    L.append('')
    L.append('採用 run の行を 0.9.0 本体 JSONL の末尾へ追記した派生ファイルが gen_1988.py の --gb／--gb-py 入力である（RULE.txt）。')
    return '\n'.join(L) + '\n'


def write_derived(base, out, picks):
    with open(base) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    with open(out, 'w') as w:
        for l in lines:
            w.write(l + '\n')
        for _, _, line in picks:
            w.write(line + '\n')


def self_test():
    with tempfile.TemporaryDirectory() as td:
        for i in range(1, 6):
            os.makedirs(os.path.join(td, f'run{i}'))
            with open(os.path.join(td, f'run{i}', 'results.jsonl'), 'w') as w:
                for n in SIZES:
                    w.write(json.dumps({'framework': 'fandhe-ai', 'task': 'gemm', 'device': 'cuda', 'size': n, 'mode': 'reuse', 'median_s': 0.01 * i, 'parity_fail_count': 0}) + '\n')
                    w.write(json.dumps({'framework': 'candle', 'task': 'gemm', 'device': 'cuda', 'size': n, 'mode': 'fresh', 'median_s': 0.005 * i, 'parity_fail_count': 0}) + '\n')
                    w.write(json.dumps({'framework': 'burn', 'task': 'gemm', 'device': 'cuda', 'size': n, 'mode': 'fresh', 'median_s': 0.004 * i,
                                        'parity_fail_count': 0, 'parity_scaled_abs_rescued': EXPECTED_RESCUED[n],
                                        'parity_scaled_abs_bound': EXPECTED_BOUND_090[n] * 8192, 'tf32': True}) + '\n')
            with open(os.path.join(td, f'run{i}', 'py.jsonl'), 'w') as w:
                w.write(json.dumps({'framework': 'pytorch', 'task': 'gemm', 'device': 'cpu', 'size': 4096, 'mode': 'fresh', 'median_s': 0.27, 'parity_fail_count': 1, 'parity_scaled_abs_rescued': 643}) + '\n')
        with open(os.path.join(td, 'load_gate.log'), 'w') as w:
            for i in range(1, 6):
                w.write(f'run={i} attempt=1 load1=0.10 gpu_util=0 pass\n')
        runs, gate, cells, ratios = aggregate(td)
        assert all(verdict(cells[k]) == '解消' for k in TARGET_BURN)
        assert verdict(cells[TARGET_PY]) == '残存'
        assert pick_median(cells[TARGET_BURN[0]])[0] == 3, pick_median(cells[TARGET_BURN[0]])[0]
        assert abs(statistics.median(ratios[(1024, 'burn')]) - 0.4) < 1e-12
        md = render(runs, gate, cells, ratios)
        assert '**解消**' in md and '**残存**' in md and '一致' in md and '8192.0 倍' in md
        # 4 run しかないセルは残存
        os.remove(os.path.join(td, 'run5', 'results.jsonl'))
        _, _, cells4, _ = aggregate(td)
        assert verdict(cells4[TARGET_BURN[0]]) == '残存'
        base = os.path.join(td, 'base.jsonl')
        with open(base, 'w') as w:
            w.write(json.dumps({'framework': 'x'}) + '\n')
        out = os.path.join(td, 'out.jsonl')
        write_derived(base, out, cells[TARGET_BURN[0]][:1])
        assert len(open(out).read().splitlines()) == 2
    print('self-test ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs')
    ap.add_argument('--md', default='aggregate.md')
    ap.add_argument('--base-gb'); ap.add_argument('--out-gb')
    ap.add_argument('--base-py'); ap.add_argument('--out-py')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test(); return
    if not a.logs:
        ap.error('--logs が必要')
    runs, gate, cells, ratios = aggregate(a.logs)
    md = render(runs, gate, cells, ratios)
    with open(a.md, 'w') as w:
        w.write(md)
    print(md)
    if a.base_gb and a.out_gb:
        picks = [pick_median(cells[k]) for k in TARGET_BURN if cells[k]]
        write_derived(a.base_gb, a.out_gb, picks)
        print(f'wrote {a.out_gb} (+{len(picks)} rows)')
    if a.base_py and a.out_py:
        picks = [pick_median(cells[TARGET_PY])] if cells[TARGET_PY] else []
        write_derived(a.base_py, a.out_py, picks)
        print(f'wrote {a.out_py} (+{len(picks)} rows)')


if __name__ == '__main__':
    main()
