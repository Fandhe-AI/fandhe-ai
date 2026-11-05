#!/usr/bin/env python3
"""0.14.0 の Mac mini（Apple M4）13 セルの判定一覧を 5 run の JSONL から出す（#3874。計測キット）。

役割: 0.13.0 の判定一覧は GB10 側の `gen_0130.py`（`--gb` 必須・HTML も生成）が 27 セルをまとめて出していた。
0.14.0 は Mac mini（#3874）と GB10（#3875）が別セッション・別 PR で、スコアボード生成（#3876）は範囲外なので、
Mac mini だけの判定一覧を本スクリプトで出す。HTML は作らない。tally の前版との比較は入力の判定一覧を読むだけ。

判定規則は 0.13.0 から変えていない: 判定経路の関数（VERBATIM_FUNCS）は
`../../framework-compare-0.13.0-remeasure/scoreboard/gen_0130.py` の逐語コピーで、`--check-verbatim` が字句一致を確かめる。
`build_row` だけは HTML 部分を落とした縮約版（判定一覧の 1 行に要るものと前版比の計算だけ。前版比の式は同じ）。
出力の行形式も `print_verdicts` と同じ。0.13.0 のデータで流すと `gen_0130.out` の Mac mini 13 行と byte 一致する（--run-ver 0.13.0 で再現）。

fail-closed: 判定側（fandhe-ai reuse）の要素検証不合格・run 欠損・有効な相手の不在・版違い・前版入力の sha256 不一致は SystemExit で止める。
ノイズ帯は対照系列がないので tally 行は `noise na`（0 ではない）。前版比は表示のみで勝敗に使わない。
"""
import argparse
import ast
import hashlib
import json
import re
import statistics
import sys
import tempfile
from pathlib import Path

VER = '0.14.0'
PREV_VER = '0.13.0'
M4_NAME = 'Mac mini（M4）'
NRUNS = 5
# 前版比の参照ファイルの sha256（RULE.txt に固定した値）。一致しなければ止める
PREV_SHA256 = (
    'e288fd21d72b9d375e89aaed12b4195d1907b2bae71b9c541ff19013366c1f38',
    'f79622dd0e5bdc4d6de2a6ce18d1e84dba6c2fd74441768a8b0cf0b1644b7eaa',
)
# gen_0130.py から逐語でコピーした関数（--check-verbatim の対象）
VERBATIM_FUNCS = ('load', 'key', 'index', 'metric', 'better', 'paired_ratio', 'judge')

FW_ORDER = ['candle', 'burn', 'pytorch', 'tensorflow', 'scipy']
CLS_JA = {'win': '1 位', 'near': '僅差', 'loss': '負け'}
DEVNAME = {'metal': 'Metal', 'cpu': 'CPU', 'cuda': 'CUDA'}
M4_ROWS = [('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)] + \
          [('train', 'metal', 64), ('infer', 'metal', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]


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


def metric(r, task):
    """比較用スカラ。時間は小さいほど良い・推論はスループット大きいほど良い。"""
    return (1.0 / r['median_s']) if task == 'infer' else r['median_s']

def better(a, b, task):
    return a > b if task == 'infer' else a < b

def paired_ratio(runs, task, device, size):
    """同一 run 内の対戦相手比の 5 run 中央値と run 別の値を返す（RULE.txt の主判定）。

    run ごとに「その run の有効な相手最速 median_s ÷ fandhe-ai reuse median_s」を求める。推論は
    スループット比（fandhe ÷ 相手最速）だが、スループット = 1/median_s のため同じ式になる。
    run 欠損・判定側の要素検証不合格・有効な相手の不在は生成を止める（fail-closed）。"""
    rs = []
    for i, d in enumerate(runs, 1):
        me = d.get(('fandhe-ai', task, device, size, 'reuse'))
        if me is None:
            raise SystemExit(f'run{i}: fandhe-ai reuse 欠損 {task} {device} N={size}')
        if (me.get('parity_fail_count') or 0) > 0:
            raise SystemExit(f'run{i}: fandhe-ai reuse の要素検証が不合格 {task} {device} N={size}')
        comp = [r['median_s'] for fw in FW_ORDER
                for r in [d.get((fw, task, device, size, 'fresh'))]
                if r is not None and (r.get('parity_fail_count') or 0) == 0]
        if not comp:
            raise SystemExit(f'run{i}: 有効な比較相手がない {task} {device} N={size}')
        rs.append(min(comp) / me['median_s'])
    return statistics.median(rs), rs

def judge(data, task, device, size, runs):
    """(区分 cls, 表示 verdict, ratio, best fw, rank, 中央値同士比, run 別比) を返す。

    区分と ratio は同一 run 内比の中央値（paired_ratio）で決める。順位・相手名は 5 ラウンド中央値
    （data）で数える。build_row と対照系列の反転判定で共用する。"""
    me = data.get(('fandhe-ai', task, device, size, 'reuse'))
    assert me is not None, (task, device, size)
    # 判定側（fandhe-ai）の要素検証が不合格なら順位を出さない。判定不能行を集計
    # （判定対象 = 勝ち＋僅差＋負け）へ黙って混ぜないよう、生成自体を止める（fail-closed）
    me_fail = me.get('parity_fail_count') or 0
    if me_fail > 0:
        raise SystemExit(f'fandhe-ai reuse の要素検証が不合格（{me_fail} 要素）: {task} {device} N={size}。判定不能のため勝敗を出さない')
    mine = metric(me, task)
    comp = []
    for fw in FW_ORDER:
        r = data.get((fw, task, device, size, 'fresh'))
        if r is None or (r.get('parity_fail_count') or 0) > 0:
            continue
        comp.append((fw, metric(r, task)))
    if not comp:
        raise SystemExit(f'有効な比較相手がない: {task} {device} N={size}')
    best = None
    for fw, v in comp:
        if best is None or better(v, best[1], task):
            best = (fw, v)
    rank = 1 + sum(1 for fw, v in comp if better(v, mine, task))
    mm_ratio = (mine / best[1]) if task == 'infer' else (best[1] / mine)
    ratio, runs_r = paired_ratio(runs, task, device, size)
    if ratio > 1:
        return 'win', '1 位', ratio, best[0], rank, mm_ratio, runs_r
    if ratio >= 0.90:
        return 'near', '僅差', ratio, best[0], rank, mm_ratio, runs_r
    if rank == 1:
        # 中央値では最速なのに同一 run 内比が 0.90 未満という表示不能な組み合わせ。黙って丸めない
        raise SystemExit(f'順位 1 位だが同一 run 内比 {ratio:.2f}: {task} {device} N={size}')
    return 'loss', f'{rank} 位', ratio, best[0], rank, mm_ratio, runs_r


# ---- ここから下は本スクリプト固有（gen_0130.py の HTML・GB10 部分を落とした縮約）

def build_row(data, data_prev, prev_ver, task, device, size, runs):
    """判定一覧の 1 行分。前版比の式は gen_0130.py の build_row と同じ（reuse 同士。前版に reuse が無ければ fresh 同士）。"""
    label = {'gemm': f'gemm {DEVNAME[device]} N={size}', 'train': f'train {DEVNAME[device]}', 'infer': f'infer {DEVNAME[device]}'}[task]
    me = data.get(('fandhe-ai', task, device, size, 'reuse'))
    me_fresh = data.get(('fandhe-ai', task, device, size, 'fresh'))
    cls, verdict, ratio, best_fw, rank, mm_ratio, runs_r = judge(data, task, device, size, runs)
    fresh_note = ''
    old = data_prev.get(('fandhe-ai', task, device, size, 'reuse'))
    if old is None:
        old = data_prev.get(('fandhe-ai', task, device, size, 'fresh'))
        base = me_fresh if me_fresh is not None else me
        fresh_note = '（fresh 同士）'
    else:
        base = me
    if old is not None:
        vprev = (metric(base, task) / metric(old, task)) if task == 'infer' else (base['median_s'] / old['median_s'])
        vprevs = f'{prev_ver} 比 {vprev:.2f}×{fresh_note}'
    else:
        vprev, vprevs = None, f'{prev_ver} 記録なし'
    return {'machine': M4_NAME, 'label': label, 'cls': cls, 'verdict': verdict, 'ratio': ratio, 'best': best_fw,
            'rank': rank, 'mm_ratio': mm_ratio, 'runs_r': runs_r, 'vprev': vprev, 'vprevs': vprevs}


def count_invalid(data, rows):
    n = 0
    for task, dev, size in rows:
        for fw in FW_ORDER:
            r = data.get((fw, task, dev, size, 'fresh'))
            if r is not None and (r.get('parity_fail_count') or 0) > 0:
                n += 1
    return n


def verdict_lines(rows, invalid_n):
    """gen_0130.py の print_verdicts と同じ行形式。tally 行は末尾。"""
    out = []
    for r in rows:
        mm_cls = 'win' if r['mm_ratio'] > 1 else ('near' if r['mm_ratio'] >= 0.90 else 'loss')
        mm = f'  [中央値同士比 {r["mm_ratio"]:.2f}x で区分が {CLS_JA[mm_cls]}]' if mm_cls != r['cls'] else ''
        runs_s = ' '.join(f'{x:.2f}' for x in r['runs_r'])
        out.append(f"{r['machine']:7} {r['label']:22} {r['verdict']:4} vs {r['best']:10} {r['ratio']:.2f}x  run別 [{runs_s}]  {r['vprevs']}{mm}")
    n = {c: sum(1 for r in rows if r['cls'] == c) for c in ('win', 'near', 'loss')}
    out.append(' '.join(map(str, ['tally', n['win'], n['near'], n['loss'], 'invalid', invalid_n, 'noise na', 'total', len(rows)])))
    return out


def check_prev_sha(files, shas):
    if len(files) != len(shas):
        raise SystemExit(f'前版入力の数 {len(files)} と sha256 の数 {len(shas)} が違う')
    for f, want in zip(files, shas):
        got = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f'前版入力の sha256 が RULE.txt の値と一致しない: {f}: {got}')


def check_version(name, idx, ver):
    vs = {r.get('version') for k, r in idx.items() if k[0] == 'fandhe-ai'}
    if vs != {ver}:
        raise SystemExit(f'{name}: fandhe-ai 行の version が {ver} でない: {sorted(map(str, vs))}')


def compute(m4_dir, prev_files, prev_shas, ver, prev_ver):
    """--m4-dir の median5 と run1〜run5 から判定行と無効セル数を返す。"""
    check_prev_sha(prev_files, prev_shas)
    d = Path(m4_dir)
    m4 = index(load(d / f'results-macmini-{ver}-median5.jsonl') + load(d / f'results-macmini-py-{ver}-median5.jsonl'))
    check_version('median5', m4, ver)
    runs = []
    for i in range(1, NRUNS + 1):
        idx = index(sum((load(d / f'run{i}' / n) for n in (f'results-macmini-{ver}.jsonl', f'results-macmini-py-{ver}.jsonl')), []))
        check_version(f'run{i}', idx, ver)
        runs.append(idx)
    prev = index(sum((load(f) for f in prev_files), []))
    rows = [build_row(m4, prev, prev_ver, *r, runs) for r in M4_ROWS]
    return rows, count_invalid(m4, M4_ROWS)


ROW_RE = re.compile(r'^Mac mini（M4） (.{22}) (\S+(?: 位)?)\s+vs (\S+)\s+([0-9.]+)x ')


def prev_compare_lines(rows, prev_out):
    """前版の判定一覧（gen_0130.out）と 13 セルを突き合わせた表示用の行を返す。tally 行が読めなければ止める。"""
    lines = Path(prev_out).read_text(encoding='utf-8').splitlines()
    tl = [l for l in lines if l.startswith('tally ')]
    if len(tl) != 1 or tl[0] != lines[-1]:
        raise SystemExit(f'{prev_out}: tally 行が末尾に 1 行だけではない')
    prev = {}
    for l in lines:
        m = ROW_RE.match(l)
        if m:
            prev[m.group(1).strip()] = (m.group(2), m.group(4))
    if len(prev) != len(rows):
        raise SystemExit(f'{prev_out}: Mac mini の行が {len(prev)} 件（期待 {len(rows)}）')
    def cls_of(v):
        return {'1 位': 'win', '僅差': 'near'}.get(v, 'loss')
    n = {'win': 0, 'near': 0, 'loss': 0}
    out = [f'prev_tally_all {tl[0]}']
    for r in rows:
        pv, pr = prev[r['label']]
        n[cls_of(pv)] += 1
        same = '' if cls_of(pv) == r['cls'] else '  区分変化'
        out.append(f"{r['label']:22} 前版 {pv:4} {pr}x → {r['verdict']:4} {r['ratio']:.2f}x{same}")
    out.append(f"prev_macmini_subset {n['win']} {n['near']} {n['loss']} total {len(rows)}")
    return out


def check_verbatim(src):
    """VERBATIM_FUNCS の字句が gen_0130.py と同じことを確かめる（コピーの取り違えを検出）。"""
    def funcs(text):
        tree = ast.parse(text)
        lines = text.splitlines()
        return {n.name: lines[n.lineno - 1:n.end_lineno] for n in tree.body if isinstance(n, ast.FunctionDef)}
    mine, theirs = funcs(Path(__file__).read_text(encoding='utf-8')), funcs(Path(src).read_text(encoding='utf-8'))
    bad = [f for f in VERBATIM_FUNCS if f not in theirs or mine[f] != theirs[f]]
    if bad:
        raise SystemExit(f'gen_0130.py と字句が違う関数: {bad}')
    print('verbatim ok: ' + ' '.join(VERBATIM_FUNCS))


def self_test():
    """fail-closed の自己検査（合成データ）。止まるべき入力で止まり、通るべき入力で tally が合うことを確かめる。"""
    def row(fw, task, dev, size, mode, sec, fail=0, ver=VER):
        return {'framework': fw, 'task': task, 'device': dev, 'size': size, 'mode': mode, 'median_s': sec,
                'parity_fail_count': fail, 'version': ver if fw == 'fandhe-ai' else 'x'}
    def build(tmp, me_sec=1.0, me_fail_run=None, comp_fail=False, drop_run=None):
        d = Path(tmp)
        rows = []
        for t, dv, s in M4_ROWS:
            rows.append(row('fandhe-ai', t, dv, s, 'reuse', me_sec))
            rows.append(row('fandhe-ai', t, dv, s, 'fresh', me_sec))
            rows.append(row('candle', t, dv, s, 'fresh', 2.0))
            rows.append(row('pytorch', t, dv, s, 'fresh', 3.0, fail=5 if comp_fail else 0))
        for i in range(1, NRUNS + 1):
            (d / f'run{i}').mkdir()
            if i == drop_run:
                continue
            rr = [dict(r) for r in rows]
            if i == me_fail_run:
                rr[0]['parity_fail_count'] = 7
            (d / f'run{i}' / f'results-macmini-{VER}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rr))
            (d / f'run{i}' / f'results-macmini-py-{VER}.jsonl').write_text('')
        (d / f'results-macmini-{VER}-median5.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (d / f'results-macmini-py-{VER}-median5.jsonl').write_text('')
        p = d / 'prev.jsonl'
        p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        return tmp, [str(p)], [hashlib.sha256(p.read_bytes()).hexdigest()]
    def stops(**kw):
        with tempfile.TemporaryDirectory() as t:
            dd, pf, ps = build(t, **{k: v for k, v in kw.items() if k != 'badsha'})
            try:
                compute(dd, pf, ['0' * 64] if kw.get('badsha') else ps, VER, PREV_VER)
            except SystemExit as e:
                return str(e)
            except FileNotFoundError as e:
                return str(e)
        raise AssertionError(f'止まらなかった: {kw}')
    with tempfile.TemporaryDirectory() as t:
        dd, pf, ps = build(t)
        rows, inv = compute(dd, pf, ps, VER, PREV_VER)
        tally = verdict_lines(rows, inv)[-1]
        # infer は 1/median_s 比較でも同じ比（2.0 倍）なので 13 セルすべて 1 位
        assert tally == 'tally 13 0 0 invalid 0 noise na total 13', tally
    with tempfile.TemporaryDirectory() as t:
        dd, pf, ps = build(t, comp_fail=True)
        rows, inv = compute(dd, pf, ps, VER, PREV_VER)
        assert inv == 13, inv                       # 相手側の不合格は判定不能に数え、順位に含めない
        assert verdict_lines(rows, inv)[-1] == 'tally 13 0 0 invalid 13 noise na total 13'
    assert '要素検証が不合格' in stops(me_fail_run=3)  # 判定側の不合格は生成停止
    assert stops(drop_run=2)                         # run 欠損は生成停止
    assert 'sha256' in stops(badsha=True)            # 前版入力の sha256 不一致
    with tempfile.TemporaryDirectory() as t:
        dd, pf, ps = build(t, me_sec=2.5)
        rows, inv = compute(dd, pf, ps, VER, PREV_VER)
        assert verdict_lines(rows, inv)[-1] == 'tally 0 0 13 invalid 0 noise na total 13'  # 比 0.80 は負け
    with tempfile.TemporaryDirectory() as t:
        dd, pf, ps = build(t, me_sec=2.1)
        rows, inv = compute(dd, pf, ps, VER, PREV_VER)
        assert verdict_lines(rows, inv)[-1] == 'tally 0 13 0 invalid 0 noise na total 13'  # 比 0.95 は僅差
    print('self-test ok')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--check-verbatim', help='gen_0130.py のパス。VERBATIM_FUNCS の字句一致を確かめて終わる')
    p.add_argument('--m4-dir', help='run1〜run5 と median5 を含むディレクトリ（macmini-series-b）')
    p.add_argument('--prev', nargs='+', help='前版の median5 JSONL 2 本（Rust・Python の順）')
    p.add_argument('--prev-sha256', nargs='+', default=list(PREV_SHA256), help='前版入力の sha256。既定は RULE.txt の値（0.13.0 データでの再現時だけ指定する）')
    p.add_argument('--prev-out', help='前版の判定一覧（gen_0130.out）。--compare-out と組で使う')
    p.add_argument('--compare-out', help='前版との区分の突き合わせ（表示用）の出力先')
    p.add_argument('--run-ver', default=VER)
    p.add_argument('--prev-ver', default=PREV_VER)
    a = p.parse_args()
    if a.self_test:
        self_test(); return
    if a.check_verbatim:
        check_verbatim(a.check_verbatim); return
    if not a.m4_dir or not a.prev:
        p.error('--m4-dir と --prev が必要')
    rows, inv = compute(a.m4_dir, a.prev, a.prev_sha256, a.run_ver, a.prev_ver)
    lines = verdict_lines(rows, inv)
    if a.compare_out:
        if not a.prev_out:
            p.error('--compare-out には --prev-out が必要')
        Path(a.compare_out).write_text('\n'.join(prev_compare_lines(rows, a.prev_out)) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
