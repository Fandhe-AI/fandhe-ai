#!/usr/bin/env python3
"""GB10 の判定一覧を 5 run の JSONL から作る（0.14.0 計測キット・#3875）。

役割: 0.13.0 では判定一覧（tally を含む）を `scoreboard/gen_0130.py` が HTML 生成と一緒に出していたが、この生成器は Mac mini の
データを必須とし、スコアボードの生成は #3876 の範囲である。そこで GB10 の 14 セルの判定だけをここへ切り出す。
呼び出し元は計測後の集計（README の再現手順）。出力の `gb10/judge_0140.out` は標準出力をそのまま保存し、手で書き換えない。
両機体を合わせた判定一覧・HTML は #3876 が作る（本ファイルは HTML を出さない）。

判定規則は変えていない。`index`・`key`・`metric`・`better`・`paired_ratio`・`judge`・`count_invalid` と前版比の計算、
判定一覧の 1 行の書式は `gen_0130.py` と字句で同じ（FW_ORDER・GB_ROWS も同じ）。同じであることの検証は 2 段:
  --regress-0130  0.13.0 の GB10 データ（前版 0.12.0）へ当てて、`gen_0130.out` の GB10 14 行と byte 一致することを確かめる。
  --self-test     判定側の要素検証不合格・欠損 run・有効な相手の不在などで止まることを確かめる。

gen_0130.py に無い入口検査（判定関数の外に足す。fail-closed にする方向だけ）:
  - gemm の fandhe-ai 行（5 run と中央値）の parity_fail_count は非負整数を必須にする（判定関数の `or 0` で欠損を 0 と読まない）。
    train／infer の行には要素検証の欄がない（0.13.0 と同じ。bench-fandhe の出力仕様）ので対象にしない。
  - 各 run の env_info.txt の compute-apps が空であることを確かめる（専有未達のデータから判定一覧を作らない）。
  - 前版の参照ファイルの sha256 を RULE.txt の値と照合する。前版 tally は RULE.txt の値と一致しなければ止める。

使い方（aggregate/ の親ディレクトリで）:
  python3 aggregate/judge_gb10_0140.py --emit-verify gb10/verify_0140.txt --emit-md gb10/tables_0140.md > gb10/judge_0140.out
  python3 aggregate/judge_gb10_0140.py --regress-0130
  python3 aggregate/judge_gb10_0140.py --self-test
"""
import argparse
import copy
import hashlib
import json
import re
import statistics
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]  # framework-compare-0.14.0-remeasure
LOGS = HERE.parent                          # docs/perf/logs

VER = '0.14.0'
PREV_VER = '0.13.0'
P130 = 'framework-compare-0.13.0-remeasure'
P120 = 'framework-compare-0.12.0-remeasure'
# 前版（0.13.0）の参照ファイル。sha256 は RULE.txt と同じ値
PREV_PINS = {
    f'{P130}/gb10/results-dgx-0.13.0-median5.jsonl': '6a0e3c7530b353dcb82662f0c08affa72b126533978084b053970239b1e22896',
    f'{P130}/gb10/results-dgx-0.13.0-extra-median5.jsonl': '12f8c0c607a9b3770e9e860b8b374c095893d52e5778019aff20613078623828',
    f'{P130}/gb10/results-dgx-py-0.13.0-median5.jsonl': '4a7416222d8cbe8b56d9aa96f9cf4825b871acaf2d5a8f270663f35dd1e0008a',
}
PREV_OUT = f'{P130}/scoreboard/gen_0130.out'
PREV_TALLY = 'tally 13 3 11 invalid 1 noise na total 27'  # 両機体の値（RULE.txt に固定）

FW_ORDER = ['candle', 'burn', 'pytorch', 'tensorflow', 'scipy']
DEVNAME = {'metal': 'Metal', 'cpu': 'CPU', 'cuda': 'CUDA'}
GB_ROWS = [('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)] + \
          [('train', 'cuda', 64), ('infer', 'cuda', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]
NRUNS = 5
CLS_JA = {'win': '1 位', 'near': '僅差', 'loss': '負け'}


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


def check_fandhe_version(name, idx, rv):
    """fandhe-ai 行の version が rv でなければ止める。別版の値を載せないための入口検査。"""
    vs = {r.get('version') for k, r in idx.items() if k[0] == 'fandhe-ai'}
    if vs != {rv}:
        raise SystemExit(f'{name}: fandhe-ai 行の version が {rv} でない: {sorted(map(str, vs))}')


def load_runs(parent, names):
    """run1〜run5 の索引を run 順に返す。ファイル欠損は open() が例外で止める。"""
    return [index(sum((load(Path(parent) / f'run{i}' / n) for n in names), [])) for i in range(1, NRUNS + 1)]


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
    （data）で数える。"""
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


def build_row(data, data_prev, prev_ver, machine, task, device, size, runs):
    """判定 1 行分の値を返す。前版比の計算は gen_0130.py の build_row と同じ（表示のみ・勝敗に使わない）。"""
    label = {'gemm': f'gemm {DEVNAME[device]} N={size}', 'train': f'train {DEVNAME[device]}', 'infer': f'infer {DEVNAME[device]}'}[task]
    me = data.get(('fandhe-ai', task, device, size, 'reuse'))
    me_fresh = data.get(('fandhe-ai', task, device, size, 'fresh'))
    cls, verdict, ratio, best_fw, rank, mm_ratio, runs_r = judge(data, task, device, size, runs)
    old = None
    fresh_note = ''
    if data_prev is not None:
        old = data_prev.get(('fandhe-ai', task, device, size, 'reuse'))
        if old is None:
            old = data_prev.get(('fandhe-ai', task, device, size, 'fresh'))
            base = me_fresh if me_fresh is not None else me
            fresh_note = '（fresh 同士）'
        else:
            base = me
    if data_prev is None:
        vprev, vprevs = None, f'{prev_ver} 比なし（機体変更）'
    elif old is not None:
        vprev = (metric(base, task) / metric(old, task)) if task == 'infer' else (base['median_s'] / old['median_s'])
        vprevs = f'{prev_ver} 比 {vprev:.2f}×{fresh_note}'
    else:
        vprev, vprevs = None, f'{prev_ver} 記録なし'
    return dict(cls=cls, verdict=verdict, ratio=ratio, mm_ratio=mm_ratio, runs_r=runs_r, label=label, machine=machine, best=best_fw,
                vprev=vprev, vprevs=vprevs, me=me, me_fresh=me_fresh, old=old, fresh_note=fresh_note, key=(task, device, size))


# 無効セル（判定不能）数
def count_invalid(data, rows):
    n = 0
    for task, dev, size in rows:
        for fw in FW_ORDER:
            r = data.get((fw, task, dev, size, 'fresh'))
            if r is not None and (r.get('parity_fail_count') or 0) > 0:
                n += 1
    return n


def format_lines(allrows, invalid_n):
    """判定一覧の行を返す（gen_0130.py の print と同じ書式。tally 行は末尾。ノイズ帯は対照系列がないので `noise na`）。"""
    wins = [r for r in allrows if r['cls'] == 'win']
    nears = [r for r in allrows if r['cls'] == 'near']
    losses = [r for r in allrows if r['cls'] == 'loss']
    out = []
    for r in allrows:
        mm_cls = 'win' if r['mm_ratio'] > 1 else ('near' if r['mm_ratio'] >= 0.90 else 'loss')
        mm = f'  [中央値同士比 {r["mm_ratio"]:.2f}x で区分が {CLS_JA[mm_cls]}]' if mm_cls != r['cls'] else ''
        runs_s = ' '.join(f'{x:.2f}' for x in r['runs_r'])
        out.append(f"{r['machine']:7} {r['label']:22} {r['verdict']:4} vs {r['best']:10} {r['ratio']:.2f}x  run別 [{runs_s}]  {r['vprevs']}{mm}")
    out.append(' '.join(str(x) for x in ('tally', len(wins), len(nears), len(losses), 'invalid', invalid_n, 'noise na', 'total', len(allrows))))
    return out


# ---- gen_0130.py に無い入口検査（fail-closed にする方向だけ）

def strict_parity(name, idx, rows):
    """gemm の fandhe-ai 行の parity_fail_count を非負整数として読み、合計を返す（欠損・不正値は止める）。"""
    total, n = 0, 0
    for task, dev, size in rows:
        if task != 'gemm':
            continue
        for mode in ('reuse', 'fresh'):
            r = idx.get(('fandhe-ai', task, dev, size, mode))
            if r is None:
                raise SystemExit(f'{name}: fandhe-ai {mode} 行がない gemm {dev} N={size}')
            v = r.get('parity_fail_count')
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                raise SystemExit(f'{name}: fandhe-ai {mode} の parity_fail_count が非負整数でない: {v!r} gemm {dev} N={size}')
            total += v
            n += 1
    return total, n


def compute_apps_of(path):
    """env_info.txt の compute-apps の行（空なら []）を返す。節が無ければ止める。"""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    if 'compute-apps:' not in lines:
        raise SystemExit(f'{path}: compute-apps: の節がない')
    out = []
    for l in lines[lines.index('compute-apps:') + 1:]:
        if l == 'resident-services:':
            break
        if l.strip():
            out.append(l)
    return out


def gb_env(parent):
    """run1〜run5 の env_info.txt 先頭行（`<GPU 名>, <driver>`）が一致することを確かめて返す（gen_0130.py の gb_env と同じ検査）。"""
    seen = set()
    for i in range(1, NRUNS + 1):
        first = open(Path(parent) / f'run{i}' / 'env_info.txt', encoding='utf-8').readline().strip()
        if not re.fullmatch(r'[^,]+, \d+(\.\d+)*', first):
            raise SystemExit(f'run{i}/env_info.txt の先頭行が `<GPU 名>, <driver>` 形式でない: {first!r}')
        seen.add(first)
    if len(seen) != 1:
        raise SystemExit(f'GB10 の env_info.txt が run 間で一致しない: {sorted(seen)}')
    return seen.pop()


def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pins(logs, pins):
    for rel, want in pins.items():
        got = sha256_of(Path(logs) / rel)
        if got != want:
            raise SystemExit(f'{rel}: sha256 が RULE.txt の値と一致しない（{got}）')


def read_prev_out(path, want_tally):
    """前版の判定一覧を読み、tally が want_tally と一致することを確かめて GB10 の行を label → (verdict, cls, ratio, best, runs) で返す。"""
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    tl = [l for l in lines if l.startswith('tally ')]
    if tl != [want_tally] or lines[-1] != want_tally:
        raise SystemExit(f'{path}: tally 行が {want_tally!r} でない: {tl}')
    out = {}
    pat = re.compile(r'^GB10\s+(.+?)\s+(1 位|僅差|\d+ 位)\s+vs (\S+)\s+([0-9.]+)x  run別 \[([0-9. ]+)\]  ')
    for l in lines:
        if not l.startswith('GB10'):
            continue
        m = pat.match(l)
        if not m:
            raise SystemExit(f'{path}: GB10 行の形式が違う: {l!r}')
        v = m.group(2)
        cls = 'win' if v == '1 位' else ('near' if v == '僅差' else 'loss')
        out[m.group(1)] = (v, cls, float(m.group(4)), m.group(3), m.group(5))
    if len(out) != len(GB_ROWS):
        raise SystemExit(f'{path}: GB10 行が {len(out)} 行（期待 {len(GB_ROWS)}）')
    return out


def collect(gb_dir, run_ver, prev_files):
    """判定の入力を読み、入口検査を通した (median 索引, run 索引, 前版索引) を返す。"""
    gb_dir = Path(gb_dir)
    gb = index(load(gb_dir / f'results-dgx-{run_ver}-median5.jsonl') + load(gb_dir / f'results-dgx-{run_ver}-extra-median5.jsonl')
               + load(gb_dir / f'results-dgx-py-{run_ver}-median5.jsonl'))
    check_fandhe_version('median5', gb, run_ver)
    runs = load_runs(gb_dir, (f'results-dgx-{run_ver}.jsonl', f'results-dgx-{run_ver}-extra.jsonl', f'results-dgx-py-{run_ver}.jsonl'))
    for i, d in enumerate(runs, 1):
        check_fandhe_version(f'GB10 run{i}', d, run_ver)
    prev_rows = []
    for f in prev_files:
        prev_rows += load(f)
    return gb, runs, index(prev_rows)


def evaluate(gb_dir, run_ver, prev_ver, prev_files):
    gb, runs, prev = collect(gb_dir, run_ver, prev_files)
    return gb, runs, [build_row(gb, prev, prev_ver, 'GB10', *r, runs) for r in GB_ROWS], count_invalid(gb, GB_ROWS)


def emit_verify(path, gb, runs, gb_dir, env_first, invalid_n):
    """要素検証と専有の集計を書く（README の根拠。parity_fail_count は生の行から数える）。"""
    lines = [f'env_info 先頭行（5 run 一致）: {env_first}']
    t, n = strict_parity('median5', gb, GB_ROWS)
    lines.append(f'fandhe-ai gemm reuse/fresh の parity_fail_count: median5 合計 {t}（{n} 行）')
    for i, d in enumerate(runs, 1):
        t, n = strict_parity(f'run{i}', d, GB_ROWS)
        apps = compute_apps_of(Path(gb_dir) / f'run{i}' / 'env_info.txt')
        lines.append(f'run{i}: fandhe-ai gemm reuse/fresh の parity_fail_count 合計 {t}（{n} 行）・compute-apps {len(apps)} 件')
    lines.append(f'相手側の判定不能（parity_fail_count > 0。5 run 中央値の行）: {invalid_n} 件')
    for task, dev, size in GB_ROWS:
        for fw in FW_ORDER:
            r = gb.get((fw, task, dev, size, 'fresh'))
            if r is not None and (r.get('parity_fail_count') or 0) > 0:
                lines.append(f'  {fw} {task} {dev} N={size}: parity_fail_count {r["parity_fail_count"]}／{r.get("parity_total")}')
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def emit_md(path, rows, prev_lines, prev, ver, prev_ver):
    """README に貼る表（セル表・区分変化・GB10 CPU gemm の reuse／fresh）を生成する。"""
    o = [f'<!-- aggregate/judge_gb10_0140.py の --emit-md の出力。手で書き換えない -->', '', '#### セル表', '',
         f'| セル | {ver} | {prev_ver} | {ver} の run 別比 | {prev_ver} 比（fandhe-ai 中央値同士） |', '|---|---|---|---|---|']
    for r in rows:
        p = prev_lines[r['label']]
        star = ' ★' if r['ratio'] <= 1 else ''
        runs_s = ' '.join(f'{x:.2f}' for x in r['runs_r'])
        o.append(f"| {r['label']}{star} | {r['verdict']} {r['ratio']:.2f}×（vs {r['best']}） | {p[0]} {p[2]:.2f}×（vs {p[3]}） | {runs_s} | "
                 f"{r['vprevs'].replace(f'{prev_ver} 比 ', '')} |")
    o += ['', '#### 区分変化', '', f'| セル | {prev_ver} | {ver} |', '|---|---|---|']
    changed = 0
    for r in rows:
        p = prev_lines[r['label']]
        if p[1] != r['cls']:
            changed += 1
            o.append(f"| {r['label']} | {p[0]} {p[2]:.2f}× | {r['verdict']} {r['ratio']:.2f}× |")
    o += ['', f'区分が変わったセル: {changed} 件（14 セル中）', '', f'#### GB10 CPU gemm の reuse／fresh（fandhe-ai の 5 run 中央値・ms）', '',
          f'| N | {ver} fresh | {ver} reuse | {ver} reuse/fresh | {prev_ver} fresh | {prev_ver} reuse | {prev_ver} reuse/fresh |', '|---|---|---|---|---|---|---|']
    for r in rows:
        task, dev, size = r['key']
        if task != 'gemm' or dev != 'cpu':
            continue
        pf = prev[('fandhe-ai', task, dev, size, 'fresh')]['median_s']
        pr = prev[('fandhe-ai', task, dev, size, 'reuse')]['median_s']
        f, u = r['me_fresh']['median_s'], r['me']['median_s']
        o.append(f'| {size} | {f*1e3:.3f} | {u*1e3:.3f} | {u/f:.2f} | {pf*1e3:.3f} | {pr*1e3:.3f} | {pr/pf:.2f} |')
    Path(path).write_text('\n'.join(o) + '\n', encoding='utf-8')


def regress_0130():
    """0.13.0 の GB10 データへ当て、gen_0130.out の GB10 14 行と byte 一致することを確かめる。"""
    gb_dir = LOGS / P130 / 'gb10'
    prev = [LOGS / P120 / 'gb10' / f for f in ('results-dgx-0.12.0-median5.jsonl', 'results-dgx-0.12.0-extra-median5.jsonl', 'results-dgx-py-0.12.0-median5.jsonl')]
    gb, runs, rows, inv = evaluate(gb_dir, '0.13.0', '0.12.0', prev)
    got = format_lines(rows, inv)
    want = [l for l in (LOGS / PREV_OUT).read_text(encoding='utf-8').splitlines() if l.startswith('GB10')]
    if got[:-1] != want:
        for a, b in zip(got, want):
            if a != b:
                print('差:', repr(a), repr(b))
        raise SystemExit('回帰: gen_0130.out の GB10 行と一致しない')
    if got[-1] != 'tally 10 0 4 invalid 1 noise na total 14':
        raise SystemExit(f'回帰: GB10 の tally が想定と違う: {got[-1]}')
    print(f'regress-0130 ok: GB10 {len(want)} 行が gen_0130.out と byte 一致・{got[-1]}')


def self_test():
    """止まるべき入力で止まることを確かめる（入力は 0.13.0 の実データを複製して壊す）。"""
    gb_dir = LOGS / P130 / 'gb10'
    prev = [LOGS / P120 / 'gb10' / 'results-dgx-0.12.0-median5.jsonl']
    gb, runs, _ = collect(gb_dir, '0.13.0', prev)

    def must_stop(f, what):
        try:
            f()
        except SystemExit:
            return
        raise AssertionError(f'止まらなかった: {what}')

    cell = ('gemm', 'cuda', 256)
    me = ('fandhe-ai',) + cell + ('reuse',)
    judge(gb, *cell, runs)  # 壊す前は通る
    g2 = copy.deepcopy(gb); g2[me]['parity_fail_count'] = 1
    must_stop(lambda: judge(g2, *cell, runs), '判定側の要素検証不合格（中央値）')
    r2 = copy.deepcopy(runs); r2[2][me]['parity_fail_count'] = 1
    must_stop(lambda: judge(gb, *cell, r2), '判定側の要素検証不合格（run3）')
    r3 = copy.deepcopy(runs); del r3[1][me]
    must_stop(lambda: judge(gb, *cell, r3), '欠損 run')
    r4 = copy.deepcopy(runs)
    for fw in FW_ORDER:
        r4[0].pop((fw,) + cell + ('fresh',), None)
    must_stop(lambda: judge(gb, *cell, r4), '有効な相手の不在（run1）')
    must_stop(lambda: check_fandhe_version('t', gb, '0.14.0'), '版の不一致')
    for bad in (None, -1, '0', True):
        g3 = copy.deepcopy(gb); g3[me]['parity_fail_count'] = bad
        must_stop(lambda: strict_parity('t', g3, GB_ROWS), f'parity_fail_count={bad!r}')
    assert strict_parity('t', gb, GB_ROWS)[0] == 0
    # 相手の parity fail は判定不能として数え、その相手を最速に含めない
    assert count_invalid(gb, GB_ROWS) == 1
    # 前版 tally・sha256・env_info
    must_stop(lambda: read_prev_out(LOGS / PREV_OUT, 'tally 1 1 1 invalid 1 noise na total 27'), '前版 tally の不一致')
    assert len(read_prev_out(LOGS / PREV_OUT, PREV_TALLY)) == 14
    must_stop(lambda: verify_pins(LOGS, {PREV_OUT: '0' * 64}), 'sha256 の不一致')
    verify_pins(LOGS, PREV_PINS)
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / 'env_info.txt'
        p.write_text('NVIDIA GB10, 580.173.02\ncompute-apps:\n', encoding='utf-8')
        assert compute_apps_of(p) == []
        p.write_text('NVIDIA GB10, 580.173.02\ncompute-apps:\nresident-services:\nvllm=0\n', encoding='utf-8')
        assert compute_apps_of(p) == []
        p.write_text('NVIDIA GB10, 580.173.02\ncompute-apps:\n123, python\nresident-services:\nvllm=1\n', encoding='utf-8')
        assert compute_apps_of(p) == ['123, python']
        p.write_text('NVIDIA GB10, 580.173.02\n', encoding='utf-8')
        must_stop(lambda: compute_apps_of(p), 'compute-apps 節なし')
    print('self-test ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--regress-0130', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--gb-dir', default=str(HERE / 'gb10'))
    ap.add_argument('--emit-verify', default='')
    ap.add_argument('--emit-md', default='')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.regress_0130:
        regress_0130()
        return
    verify_pins(LOGS, PREV_PINS)
    prev_lines = read_prev_out(LOGS / PREV_OUT, PREV_TALLY)
    prev_files = [LOGS / rel for rel in PREV_PINS]
    gb, runs, _ = collect(a.gb_dir, VER, prev_files)
    # 入口検査（専有・要素検証の欠損）を判定より先に行う
    env_first = gb_env(a.gb_dir)
    strict_parity('median5', gb, GB_ROWS)
    for i, d in enumerate(runs, 1):
        strict_parity(f'run{i}', d, GB_ROWS)
        apps = compute_apps_of(Path(a.gb_dir) / f'run{i}' / 'env_info.txt')
        if apps:
            raise SystemExit(f'run{i}: compute プロセスあり（専有未達）: {apps}')
    gb, runs, rows, inv = evaluate(a.gb_dir, VER, PREV_VER, prev_files)
    print('\n'.join(format_lines(rows, inv)))
    if a.emit_verify:
        emit_verify(a.emit_verify, gb, runs, a.gb_dir, env_first, inv)
    if a.emit_md:
        _, _, prev = collect(a.gb_dir, VER, prev_files)
        emit_md(a.emit_md, rows, prev_lines, prev, VER, PREV_VER)


if __name__ == '__main__':
    sys.exit(main())
