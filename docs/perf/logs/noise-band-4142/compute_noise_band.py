#!/usr/bin/env python3
"""0.14.0 の run 別比から「ノイズ帯」候補 NB-RANGE5・NB-IQR を試算する（#4142。計測キットではなく試算の再現部品）。

役割: スコアボード 0.14.0 の判定一覧は系列 A（ゲートなしの対照）を置かないため tally が `noise na` のままである。
本スクリプトは追加計測なしで、既存の 5 run の JSONL から次を出す（docs/perf/noise-band-decision.md の試算表の出典）。
  - NB-RANGE5: セルごとの run 別対戦相手比の最小〜最大（推奨案）。区分境界 1.0（1 位／僅差）・0.90（僅差／負け）を含めば「帯が境界をまたぐ」
  - NB-IQR: fandhe-ai reuse の iter 内 q1/q3 から作る比の帯（run 間の中央値。参考・判定には使わない）
呼び出し文脈: 手動実行のみ（CI 非接続）。比の式は `framework-compare-0.14.0-remeasure/scoreboard/gen_0140.py::paired_ratio`
（Mac mini は `...-macmini-remeasure/aggregate/judge_macmini_0140.py` の同名関数）と同じ。load・key・index は gen_0140.py の逐語コピーで、
`--check-verbatim` が字句一致を確かめる。

fail-closed: 判定側（fandhe-ai reuse）の parity_fail_count > 0・run 欠損・有効な相手の不在・版違いは SystemExit で止める。
相手側の parity_fail_count > 0 の行は最速に含めない（paired_ratio と同じ）。依存は Python 標準ライブラリのみ。
"""
import argparse
import ast
import json
import statistics
import tempfile
from pathlib import Path

VER = '0.14.0'
NRUNS = 5
FW_ORDER = ['candle', 'burn', 'pytorch', 'tensorflow', 'scipy']
BOUNDS = (1.0, 0.90)  # 区分境界: 1 位／僅差 = 1.0、僅差／負け = 0.90（RULE.txt の判定）
VERBATIM_FUNCS = ('load', 'key', 'index')

PROFILES = {
    'macmini': {
        'files': (f'results-macmini-{VER}.jsonl', f'results-macmini-py-{VER}.jsonl'),
        'cells': [('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)] +
                 [('train', 'metal', 64), ('infer', 'metal', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)],
    },
    'gb10': {
        'files': (f'results-dgx-{VER}.jsonl', f'results-dgx-{VER}-extra.jsonl', f'results-dgx-py-{VER}.jsonl'),
        'cells': [('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)] +
                 [('train', 'cuda', 64), ('infer', 'cuda', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)],
    },
}


# ---- gen_0140.py から逐語でコピーした関数（--check-verbatim の対象）
def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def key(r):
    return (r['framework'], r['task'], r['device'], int(r['size']), r['mode'])


def index(rows):
    d = {}
    for r in rows:
        if r['task'] in ('train_phases', 'infer_phases'):
            continue
        # TF32 opt-in スイープ行（fandhe-ai／candle）は f32 判定から外す。burn は TF32 既定で
        # その行しか存在しないため残す（0.10.0 版と同じ扱い）
        if r.get('tf32') and r['framework'] != 'burn':
            continue
        k = key(r)
        assert k not in d, f'重複行: {k}'
        d[k] = r
    return d


# ---- ここから下は本スクリプト固有
def best_rival(d, i, task, device, size):
    """run i の有効な相手（parity 合格）の最速 median_s。paired_ratio と同じ選び方。"""
    comp = [r['median_s'] for fw in FW_ORDER
            for r in [d.get((fw, task, device, size, 'fresh'))]
            if r is not None and (r.get('parity_fail_count') or 0) == 0]
    if not comp:
        raise SystemExit(f'run{i}: 有効な比較相手がない {task} {device} N={size}')
    return min(comp)


def run_ratios(runs, task, device, size):
    """paired_ratio と同じ式で run 別比 rs と、iter 内 q1/q3 から作る比の帯の run 別リスト (los, his) を返す。

    比 = 相手最速 median_s ÷ fandhe-ai reuse median_s。iter 内の遅い側 q3 が比の下側、速い側 q1 が上側になる。"""
    rs, los, his = [], [], []
    for i, d in enumerate(runs, 1):
        me = d.get(('fandhe-ai', task, device, size, 'reuse'))
        if me is None:
            raise SystemExit(f'run{i}: fandhe-ai reuse 欠損 {task} {device} N={size}')
        if (me.get('parity_fail_count') or 0) > 0:
            raise SystemExit(f'run{i}: fandhe-ai reuse の要素検証が不合格 {task} {device} N={size}')
        riv = best_rival(d, i, task, device, size)
        rs.append(riv / me['median_s'])
        los.append(riv / me['q3_s'])
        his.append(riv / me['q1_s'])
    return rs, los, his


def crossed(lo, hi):
    """帯 [lo, hi] が区分境界を含む（境界ちょうども含む）なら、またぐ境界の一覧を返す。"""
    return [b for b in BOUNDS if lo <= b <= hi]


def analyze(runs, cells):
    out = []
    for task, device, size in cells:
        rs, los, his = run_ratios(runs, task, device, size)
        lo, hi = min(rs), max(rs)
        out.append({'cell': f'{task} {device} {size}', 'rs': rs, 'median': statistics.median(rs), 'min': lo, 'max': hi,
                    'cross': crossed(lo, hi), 'iqr_lo': statistics.median(los), 'iqr_hi': statistics.median(his)})
    return out


def lines(rows):
    out = ['cell\truns\tmedian\tmin\tmax\tcross\tiqr_lo\tiqr_hi']
    for r in rows:
        runs_s = ' '.join(f'{x:.3f}' for x in r['rs'])
        cross = ','.join(f'{b:.2f}' for b in r['cross']) or '-'
        out.append(f"{r['cell']}\t{runs_s}\t{r['median']:.3f}\t{r['min']:.3f}\t{r['max']:.3f}\t{cross}\t{r['iqr_lo']:.3f}\t{r['iqr_hi']:.3f}")
    return out


def load_runs(parent, names):
    """run1〜run5 の索引と fandhe-ai の版検査。ファイル欠損は open() が例外で止める（fail-closed）。"""
    runs = []
    for i in range(1, NRUNS + 1):
        d = index(sum((load(Path(parent) / f'run{i}' / n) for n in names), []))
        vs = {r.get('version') for k, r in d.items() if k[0] == 'fandhe-ai'}
        if vs != {VER}:
            raise SystemExit(f'run{i}: fandhe-ai 行の version が {VER} でない: {sorted(map(str, vs))}')
        runs.append(d)
    return runs


def check_verbatim(src):
    """VERBATIM_FUNCS の字句が gen_0140.py と同じことを確かめる（コピーの取り違えを検出）。"""
    def funcs(text):
        lines_ = text.splitlines()
        return {n.name: lines_[n.lineno - 1:n.end_lineno] for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    mine, theirs = funcs(Path(__file__).read_text(encoding='utf-8')), funcs(Path(src).read_text(encoding='utf-8'))
    bad = [f for f in VERBATIM_FUNCS if f not in theirs or mine[f] != theirs[f]]
    if bad:
        raise SystemExit(f'gen_0140.py と字句が違う関数: {bad}')
    print('verbatim ok: ' + ' '.join(VERBATIM_FUNCS))


def self_test():
    """合成データで帯の算出・境界ちょうどの扱い・fail-closed を確かめる。"""
    def row(fw, mode, sec, fail=0, ver=VER):
        return {'framework': fw, 'task': 'gemm', 'device': 'cpu', 'size': 1, 'mode': mode, 'median_s': sec, 'q1_s': sec * 0.9,
                'q3_s': sec * 1.1, 'parity_fail_count': fail, 'version': ver if fw == 'fandhe-ai' else 'x'}

    def mk(me_secs, rival=1.0, me_fail_run=None, riv_fail=False):
        return [index([row('fandhe-ai', 'reuse', s, fail=7 if i == me_fail_run else 0),
                       row('candle', 'fresh', rival), row('pytorch', 'fresh', 0.1 if riv_fail else 5.0, fail=5 if riv_fail else 0)])
                for i, s in enumerate(me_secs, 1)]

    cell = [('gemm', 'cpu', 1)]
    # 比 = 1/me。最小 0.97・最大 1.12 → 1.0 をまたぐ
    r = analyze(mk([1 / 0.98, 1 / 1.0, 1 / 1.12, 1 / 0.97, 1 / 0.97]), cell)[0]
    assert abs(r['min'] - 0.97) < 1e-9 and abs(r['max'] - 1.12) < 1e-9 and r['cross'] == [1.0], r
    # 帯が 0.90 と 1.0 の間に収まる（安定した僅差）→ またがない
    r = analyze(mk([1 / 0.94] * 5), cell)[0]
    assert r['cross'] == [] and abs(r['median'] - 0.94) < 1e-9, r
    # 境界ちょうどは「含む」扱い
    assert crossed(0.95, 1.0) == [1.0] and crossed(0.90, 0.95) == [0.90] and crossed(0.91, 0.99) == []
    # 相手側 parity 不合格の行は最速に含めない（pytorch 0.1 を除外して candle 1.0 が相手）
    r = analyze(mk([1.0] * 5, riv_fail=True), cell)[0]
    assert abs(r['median'] - 1.0) < 1e-9, r
    # 判定側の不合格は停止
    try:
        analyze(mk([1.0] * 5, me_fail_run=3), cell)
    except SystemExit as e:
        assert '要素検証が不合格' in str(e)
    else:
        raise AssertionError('判定側の不合格で止まらなかった')
    # 有効な相手の不在は停止
    try:
        analyze([index([row('fandhe-ai', 'reuse', 1.0), row('candle', 'fresh', 1.0, fail=1)])], cell)
    except SystemExit as e:
        assert '有効な比較相手がない' in str(e)
    else:
        raise AssertionError('相手の不在で止まらなかった')
    with tempfile.TemporaryDirectory() as t:
        for i in range(1, NRUNS + 1):
            (Path(t) / f'run{i}').mkdir()
            rows = [row('fandhe-ai', 'reuse', 1.0, ver='0.0.0'), row('candle', 'fresh', 1.0)]
            (Path(t) / f'run{i}' / 'a.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))
        try:
            load_runs(t, ('a.jsonl',))
        except SystemExit as e:
            assert 'version' in str(e)
        else:
            raise AssertionError('版違いで止まらなかった')
        try:
            load_runs(t, ('missing.jsonl',))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('ファイル欠損で止まらなかった')
    print('self-test ok')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--check-verbatim', help='gen_0140.py のパス。VERBATIM_FUNCS の字句一致を確かめて終わる')
    p.add_argument('--profile', choices=sorted(PROFILES), help='セル集合と入力ファイル名の組')
    p.add_argument('--runs-dir', help='run1〜run5 を含むディレクトリ')
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if a.check_verbatim:
        check_verbatim(a.check_verbatim)
        return
    if not a.profile or not a.runs_dir:
        p.error('--profile と --runs-dir が必要')
    prof = PROFILES[a.profile]
    print('\n'.join(lines(analyze(load_runs(a.runs_dir, prof['files']), prof['cells']))))


if __name__ == '__main__':
    main()
