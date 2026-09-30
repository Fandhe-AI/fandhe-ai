#!/usr/bin/env python3
"""イシュー #2120: framework-compare 両機体再計測（3 腕 × 5 run）を RULE.txt の規則で集計する。

役割: orchestrate.sh が LOGD へ書いた run{1..5}/{A,B,C,others}.jsonl を読み、
  - セル別の腕ごと 5 run 中央値・同一 run 内比 B→C（主）／A→C（参考）とその中央値・非後退判定（<= 1.00）
  - checksum の run 間・腕間の文字列完全一致
  - 専有ゲート結果（gate.log）
  を aggregate.md へ書き、scoreboard/gen_2120.py の入力となる派生 JSONL（腕ごと・機体別）を出力する。
python3 標準ライブラリのみ。#1988 aggregate.py の fail-closed 教訓（parity_fail_count の厳格読み・
5 run 完備の強制・run 欠損の空欄保持）を踏襲する。

使い方:
  python3 aggregate.py --machine gb10 --logs <dir> [--md aggregate.md] [--out-prefix results-gb10]
     -> <out-prefix>-{A,B,C}-full.jsonl（腕の fandhe-ai 採用行 + candle／burn 採用行）を出力
  python3 aggregate.py --self-test
"""
import argparse
import json
import os
import re
import statistics
import sys
import tempfile

ARMS = ('A', 'B', 'C')
NRUN = 5
PHASE_TASKS = ('train_phases', 'infer_phases', 'gemm_phases')


def judged_cells(machine):
    """判定対象の fandhe-ai セル（orchestrate.sh の fandhe_cells から --phases を除いたもの）。"""
    if machine == 'gb10':
        gdev, gsz, csz, devs = 'cuda', (256, 512, 1024, 2048, 4096), (256, 512, 1024, 2048, 4096), ('cuda', 'cpu')
    elif machine == 'm4max':
        gdev, gsz, csz, devs = 'metal', (256, 512, 1024, 2048, 4096), (256, 512, 1024, 2048), ('cpu', 'metal')
    else:
        raise ValueError(f'unknown machine: {machine}')
    cells = []
    for n in gsz:
        for m in ('fresh', 'reuse'):
            cells.append(('gemm', gdev, n, m))
    for n in csz:
        for m in ('fresh', 'reuse'):
            cells.append(('gemm', 'cpu', n, m))
    for d in devs:
        for t in ('train', 'infer'):
            for m in ('fresh', 'reuse'):
                cells.append((t, d, 64, m))
    return cells


def others_expected(machine):
    """candle／burn の期待セル（RULE.txt セル範囲・orchestrate.sh の others_cells と同一。fresh 行のみ）。
    全 run 欠損でも検出できるよう、観測キーではなく期待キーから完備性を検査する。"""
    exp = set()
    for (t, d, n, m) in judged_cells(machine):
        if m == 'fresh':
            for fw in ('candle', 'burn'):
                exp.add((fw, t, d, n, m))
    return exp - others_exceptions(machine)


def others_exceptions(machine):
    """明示例外: M4 Max の burn Metal GEMM N>=512 は結果テンソル全ゼロ（upstream 既知バグ）で記録拒否される
    （gen_2120.py の M4_SKIP と同一）。ここへ足す変更は RULE.txt 判定 1 の緩和にあたりレビュー必須。"""
    if machine == 'm4max':
        return {('burn', 'gemm', 'metal', n, 'fresh') for n in (512, 1024, 2048, 4096)}
    return set()


def key(r):
    return (r['framework'], r['task'], r['device'], int(r['size']), r.get('mode', 'fresh'))


def raw_checksum(line):
    """JSON 行から checksum の値を文字列のまま取り出す（浮動小数の再整形で不一致を見逃さない）。"""
    m = re.search(r'"checksum":\s*([^,}\s]+)', line)
    return m.group(1) if m else None


def fail_count(r):
    """parity_fail_count を非負整数として厳格に読む（欠損・null・負数・非整数は入力不正。#1988 教訓）。
    gemm 以外（train／infer）は元来この項目を持たないため、欠損のみ 0 とみなし、存在すれば同じ検証を行う。"""
    if 'parity_fail_count' not in r:
        if r.get('task') == 'gemm':
            raise ValueError(f'parity_fail_count が欠損: {r.get("framework")} {r.get("device")} N={r.get("size")}')
        return 0
    v = r['parity_fail_count']
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f'parity_fail_count が非負整数ではない: {v!r} ({r.get("framework")} {r.get("device")} N={r.get("size")})')
    return v


def load_jsonl(path):
    """{key: (row, line)}。--phases 行は判定対象外のため除外。後勝ち。"""
    idx = {}
    if not os.path.exists(path):
        return idx
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get('task') in PHASE_TASKS or 'phase' in r:
                continue
            idx[key(r)] = (r, line)
    return idx


def pick_median(rows):
    """[(run, row, line)] から median_s の中央値 run を返す（median_rounds.py と同規則。偶数なら小さい側）。"""
    rs = sorted(rows, key=lambda t: t[1]['median_s'])
    return rs[len(rs) // 2] if len(rs) % 2 else rs[len(rs) // 2 - 1]


def parse_gate(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        p = line.split()
        if len(p) >= 5 and p[0].startswith('run='):
            out[int(p[0][4:])] = (p[-1], p[2][6:], p[3][9:])  # 最後に書かれた試行を採用
    return out


def load_all(logs):
    """runs[r] = {'A': idx, 'B': idx, 'C': idx, 'O': idx}。"""
    runs = {}
    for r in range(1, NRUN + 1):
        d = os.path.join(logs, f'run{r}')
        runs[r] = {a: load_jsonl(os.path.join(d, f'{a}.jsonl')) for a in ARMS}
        runs[r]['O'] = load_jsonl(os.path.join(d, 'others.jsonl'))
    return runs


def validate(runs, machine):
    """fail-closed 検証。問題があれば ValueError（派生出力を作らない）。"""
    errs = []
    cells = judged_cells(machine)
    for a in ARMS:
        for r in range(1, NRUN + 1):
            for k, (row, _) in runs[r][a].items():
                if row['framework'] != 'fandhe-ai':
                    errs.append(f'run{r}/{a}.jsonl に別 FW の行が混入: {k}')
                    continue
                try:
                    fail_count(row)
                except ValueError as e:
                    errs.append(f'run{r}/{a}: {e}')
    for r in range(1, NRUN + 1):
        for k, (row, _) in runs[r]['O'].items():
            if row['framework'] not in ('candle', 'burn'):
                errs.append(f'run{r}/others.jsonl に想定外 FW: {k}')
                continue
            try:
                fail_count(row)
            except ValueError as e:
                errs.append(f'run{r}/others: {e}')
    for (t, d, n, m) in cells:
        k = ('fandhe-ai', t, d, n, m)
        for a in ARMS:
            # RULE.txt 判定 4: 判定対象行の checksum 欠損は一致扱いにせず入力不正として拒否する。
            nock = [r for r in range(1, NRUN + 1) if k in runs[r][a] and raw_checksum(runs[r][a][k][1]) is None]
            if nock:
                errs.append(f'腕 {a} の {t} {d} N={n} {m} の checksum が run {nock} で欠損')
            miss = [r for r in range(1, NRUN + 1) if k not in runs[r][a]]
            if miss:
                errs.append(f'腕 {a} の {t} {d} N={n} {m} が run {miss} で欠損（5 run × 3 腕完備が必須）')
    okeys = {k for r in runs for k in runs[r]['O']} | others_expected(machine)
    for k in okeys:
        have = [r for r in range(1, NRUN + 1) if k in runs[r]['O']]
        if len(have) != NRUN:
            errs.append(f'others の {k} が {len(have)}/5 run のみ（部分欠損。原因を調査してから再計測すること）')
    if errs:
        raise ValueError('入力検証に失敗（派生出力を作らない）:\n  ' + '\n  '.join(errs))


def cell_stats(runs, machine):
    """セルごとの腕別 run 値・比・checksum 一致を返す。validate 済みが前提。"""
    out = []
    for (t, d, n, m) in judged_cells(machine):
        k = ('fandhe-ai', t, d, n, m)
        med = {a: [runs[r][a][k][0]['median_s'] for r in range(1, NRUN + 1)] for a in ARMS}
        bc = {r: runs[r]['C'][k][0]['median_s'] / runs[r]['B'][k][0]['median_s'] for r in range(1, NRUN + 1)}
        ac = {r: runs[r]['C'][k][0]['median_s'] / runs[r]['A'][k][0]['median_s'] for r in range(1, NRUN + 1)}
        cks = {a: [raw_checksum(runs[r][a][k][1]) for r in range(1, NRUN + 1)] for a in ARMS}
        within = all(len(set(v)) == 1 for v in cks.values())
        across = len({c for v in cks.values() for c in v}) == 1
        bc_med = statistics.median(bc.values())
        out.append(dict(cell=(t, d, n, m), med=med, bc=bc, ac=ac, bc_med=bc_med, ac_med=statistics.median(ac.values()),
                        nonreg=bc_med <= 1.00, cks=cks, ck_within=within, ck_across=across,
                        ck_none=all(c is None for v in cks.values() for c in v)))
    return out


def fmt(x, p=4):
    return '' if x is None else f'{x:.{p}f}'


def render(runs, machine, stats):
    L = [f'# #2120 集計（{machine}・RULE.txt 規則）', '']
    L.append('## 専有ゲート')
    L.append('')
    L.append('| run | 判定 | load1 | gpu_util |')
    L.append('|---|---|---|---|')
    for r in range(1, NRUN + 1):
        g = GATE.get(r)
        L.append(f'| {r} | {g[0] if g else "記録なし"} | {g[1] if g else "-"} | {g[2] if g else "-"} |')
    L.append('')
    L.append('## Phase 3 前後比 B→C（主）・A→C（参考）。比 = C.median_s / B.median_s（同一 run 内・<1 で C が高速。非後退 ⇔ 5 run 中央値 <= 1.00）')
    L.append('')
    L.append('| セル | B 中央値[s] | C 中央値[s] | run1 | run2 | run3 | run4 | run5 | B→C 中央値 | 判定 | A→C 中央値 | checksum（run 間／腕間） |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    nreg = 0
    for s in stats:
        t, d, n, m = s['cell']
        runs5 = ' | '.join(fmt(s['bc'].get(r)) for r in range(1, NRUN + 1))
        v = '非後退' if s['nonreg'] else '**後退**'
        nreg += 0 if s['nonreg'] else 1
        ck = 'n/a' if s['ck_none'] else f'{"一致" if s["ck_within"] else "**不一致**"}／{"一致" if s["ck_across"] else "**不一致**"}'
        L.append(f'| {t} {d} N={n} {m} | {statistics.median(s["med"]["B"]):.6g} | {statistics.median(s["med"]["C"]):.6g} | {runs5} | '
                 f'{s["bc_med"]:.4f} | {v} | {s["ac_med"]:.4f} | {ck} |')
    L.append('')
    L.append(f'後退セル数（B→C 中央値 > 1.00）: {nreg} / {len(stats)}')
    bad = [s for s in stats if not s['ck_none'] and not (s['ck_within'] and s['ck_across'])]
    L.append(f'checksum 不一致セル数: {len(bad)}（不一致は是正せず記録。RULE.txt 判定 4）')
    return '\n'.join(L) + '\n'


GATE = {}


def write_full(runs, out_prefix, machine):
    """腕ごとの派生 JSONL: fandhe-ai 採用行 + candle／burn 採用行（gen_2120.py の入力）。"""
    written = []
    for a in ARMS:
        lines = []
        for (t, d, n, m) in judged_cells(machine):
            k = ('fandhe-ai', t, d, n, m)
            rows = [(r, runs[r][a][k][0], runs[r][a][k][1]) for r in range(1, NRUN + 1)]
            lines.append(pick_median(rows)[2])
        okeys = sorted({k for r in runs for k in runs[r]['O']})
        for k in okeys:
            rows = [(r, runs[r]['O'][k][0], runs[r]['O'][k][1]) for r in range(1, NRUN + 1) if k in runs[r]['O']]
            lines.append(pick_median(rows)[2])
        p = f'{out_prefix}-{a}-full.jsonl'
        with open(p, 'w') as w:
            w.write('\n'.join(lines) + '\n')
        written.append(p)
    return written


def self_test():
    with tempfile.TemporaryDirectory() as td:
        def mk(ratio_c=1.0, drop=None, bad_parity='__unset__', ck_diff=False, no_ck=False, others_all=True):
            for r in range(1, NRUN + 1):
                os.makedirs(os.path.join(td, f'run{r}'), exist_ok=True)
                for a in ARMS:
                    with open(os.path.join(td, f'run{r}', f'{a}.jsonl'), 'w') as w:
                        for (t, d, n, m) in judged_cells('gb10'):
                            if drop == (r, a, t, d, n, m):
                                continue
                            base = 0.001 * r
                            ms = base * (ratio_c if a == 'C' else 1.0)
                            row = {'framework': 'fandhe-ai', 'task': t, 'device': d, 'size': n, 'mode': m, 'median_s': ms}
                            if t == 'gemm':
                                row['parity_fail_count'] = 0
                            ck = '1.500000' if not (ck_diff and a == 'C' and r == 2) else '1.500001'
                            line = json.dumps(row)[:-1] + f', "checksum":{ck}' + '}'
                            if no_ck:
                                line = json.dumps(row)
                            if bad_parity != '__unset__' and t == 'gemm':
                                row2 = dict(row); row2['parity_fail_count'] = bad_parity
                                line = json.dumps(row2)[:-1] + f', "checksum":{ck}' + '}'
                            w.write(line + '\n')
                with open(os.path.join(td, f'run{r}', 'others.jsonl'), 'w') as w:
                    keys = others_expected('gb10') if others_all else {('candle', 'gemm', 'cuda', 256, 'fresh')}
                    for (fw, t, d, n, m) in sorted(keys):
                        row = {'framework': fw, 'task': t, 'device': d, 'size': n, 'mode': m, 'median_s': 0.01 * r}
                        if t == 'gemm':
                            row['parity_fail_count'] = 0
                        w.write(json.dumps(row) + '\n')

        def go():
            runs = load_all(td)
            validate(runs, 'gb10')
            return runs, cell_stats(runs, 'gb10')

        # 正常系: 比 = 1.00 境界は非後退
        mk(1.0)
        runs, st = go()
        assert all(s['nonreg'] and abs(s['bc_med'] - 1.0) < 1e-12 for s in st)
        assert all(s['ck_within'] and s['ck_across'] for s in st)
        # 1.0001 は後退
        mk(1.0001)
        _, st = go()
        assert all(not s['nonreg'] for s in st)
        # 改善
        mk(0.9)
        _, st = go()
        assert all(s['nonreg'] for s in st)
        # 採用 run: 5 run の中央値 run は run3
        runs = load_all(td)
        k = ('fandhe-ai', 'gemm', 'cuda', 256, 'fresh')
        assert pick_median([(r, runs[r]['C'][k][0], runs[r]['C'][k][1]) for r in range(1, 6)])[0] == 3
        # 偶数（4 run）は小さい側
        assert pick_median([(r, {'median_s': v}, '') for r, v in ((1, 4.0), (2, 1.0), (3, 3.0), (4, 2.0))])[0] == 4
        # 5 run 欠け・腕欠けは停止
        mk(1.0, drop=(4, 'B', 'gemm', 'cuda', 256, 'fresh'))
        try:
            go()
        except ValueError:
            pass
        else:
            raise AssertionError('欠損セルを受理した')
        # checksum 欠損は停止
        mk(1.0, no_ck=True)
        try:
            go()
        except ValueError:
            pass
        else:
            raise AssertionError('checksum 欠損を受理した')
        # 比較対象セルの全 run 欠損は停止（明示例外の m4max burn metal は除く）
        mk(1.0, others_all=False)
        try:
            go()
        except ValueError:
            pass
        else:
            raise AssertionError('others 全 run 欠損を受理した')
        assert ('burn', 'gemm', 'metal', 512, 'fresh') not in others_expected('m4max')
        assert ('burn', 'gemm', 'cuda', 512, 'fresh') in others_expected('gb10')
        # parity_fail_count 不正 4 種は停止
        for bad in (None, -1, 'x', True):
            mk(1.0, bad_parity=bad)
            try:
                go()
            except ValueError:
                pass
            else:
                raise AssertionError(f'不正な parity_fail_count を受理: {bad!r}')
        # checksum 不一致の検出（是正せず記録）
        mk(1.0, ck_diff=True)
        _, st = go()
        assert all(not (s['ck_within'] and s['ck_across']) for s in st)
        # 派生 JSONL と md
        mk(1.0)
        runs, st = go()
        GATE.clear()
        for r in range(1, 6):
            GATE[r] = ('pass', '0.10', '0')
        md = render(runs, 'gb10', st)
        assert '非後退' in md and '後退セル数（B→C 中央値 > 1.00）: 0' in md
        ps = write_full(runs, os.path.join(td, 'out'), 'gb10')
        assert len(ps) == 3 and all(len(open(p).read().splitlines()) == len(judged_cells('gb10')) + len(others_expected('gb10')) for p in ps)
        # run 欠損の空欄保持（左詰めしない）
        st2 = cell_stats(runs, 'gb10')
        del st2[0]['bc'][2]
        row = [l for l in render(runs, 'gb10', st2).splitlines() if l.startswith('| gemm cuda N=256 fresh |')][0]
        assert row.split('|')[5].strip() == '' , row
    print('self-test ok')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', choices=('gb10', 'm4max'))
    ap.add_argument('--logs')
    ap.add_argument('--md', default='aggregate.md')
    ap.add_argument('--out-prefix', help='派生 JSONL の接頭辞（<prefix>-{A,B,C}-full.jsonl）')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not (a.machine and a.logs):
        ap.error('--machine と --logs が必要')
    runs = load_all(a.logs)
    try:
        validate(runs, a.machine)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    GATE.update(parse_gate(os.path.join(a.logs, 'gate.log')))
    stats = cell_stats(runs, a.machine)
    md = render(runs, a.machine, stats)
    with open(a.md, 'w') as w:
        w.write(md)
    print(md)
    if a.out_prefix:
        for p in write_full(runs, a.out_prefix, a.machine):
            print('wrote', p)


if __name__ == '__main__':
    main()
