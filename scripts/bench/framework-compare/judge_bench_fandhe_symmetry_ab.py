#!/usr/bin/env python3
"""bench-fandhe 是正（#3265）の前後を相手フレームワーク同席で併記する集計器（イシュー #3266）。

役割: `run_ab_bench_fandhe_symmetry.sh` が出力する 1 系列（run1〜run5）を読み、旧条件腕（old）と
新条件腕（new）それぞれの「同一 run 内の最速相手比」と区分（1 位／僅差／負け）を並べる。
判定式は `docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/scoreboard/gen_0110_macmini.py` の
`index`・`paired_ratio`・`judge`（RULE の主判定）と同一で、本ファイルは式を変えない。
#3247 のような ADOPT／REJECT の二値判定は行わない（腕比は参考表示のみ。合格ラインなし）。
規則の正本: `docs/perf/logs/bench-fandhe-symmetry-ab-3266/RULE.txt`。

入力（`--series-dir`）: `run{1..5}/results-{old,new,opp,py}.jsonl`。腕ごとに別ファイルなので行内容では
腕を区別しない。寛容パースはしない（欠損・重複・run 不足・ゲート不成立・sha 不一致は undetermined）。

終了コード: 0 = 全セル判定可能かつ hard 条件成立 / 2 = undetermined あり / 3 = hard 条件の不一致あり
（3 が優先。hard 条件の不一致は「正しさの失敗」として所有者へ報告する。tolerance は変えない）。
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

NRUNS = 5
FW_ORDER = ['candle', 'burn', 'pytorch', 'tensorflow', 'scipy']
# hard 条件で完全一致を要求する項目（欠落・非数値は undetermined。値の不一致のみ hard 失敗）
HARD_FIELDS = ('checksum', 'parity_total', 'parity_fail_count', 'parity_max_abs_err',
               'parity_max_rel_err', 'parity_scaled_abs_bound', 'parity_scaled_abs_rescued')

# 判定対象セル（0.11.0 スコアボードと同じ 27 行 = Mac mini 13 + GB10 14）
CELLS = {
    'macmini': ([('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)]
                + [('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)]
                + [('train', 'cpu', 64), ('train', 'metal', 64), ('infer', 'cpu', 64), ('infer', 'metal', 64)]),
    'gb10': ([('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)]
             + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)]
             + [('train', 'cpu', 64), ('train', 'cuda', 64), ('infer', 'cpu', 64), ('infer', 'cuda', 64)]),
}
# 0.11.0 正式系列の参考列用ファイル名（run ディレクトリ内）
REF_FILES = {
    'macmini': ('results-macmini-0.11.0.jsonl', 'results-macmini-py-0.11.0.jsonl'),
    'gb10': ('results-dgx-0.11.0.jsonl', 'results-dgx-0.11.0-extra.jsonl', 'results-dgx-py-0.11.0.jsonl'),
}


def expected_opponents(profile, task, device, size):
    """セルごとに必ず存在すべき比較相手（RULE の期待セル構成。ランナーの EXP_OPP／EXP_PY と同じ内訳）。

    cpu は 5 相手、加速器は candle・burn・pytorch（macmini は tensorflow も）。除外してよいのは burn の
    Metal GEMM N=512〜4096 の既知の記録拒否（0.11.0 と同じ 4 件）だけで、ここで明示的に除く。"""
    fws = ['candle', 'burn', 'pytorch']
    if profile == 'macmini' or device == 'cpu':
        fws.append('tensorflow')
    if device == 'cpu':
        fws.append('scipy')
    if profile == 'macmini' and task == 'gemm' and device == 'metal' and size >= 512:
        fws.remove('burn')
    return fws


class Undetermined(Exception):
    """系列・セルを判定できない理由（握りつぶさず表に出す）。"""


def load(path):
    p = Path(path)
    if not p.is_file():
        raise Undetermined(f'ファイル欠損: {p}')
    rows = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise Undetermined(f'JSON 不正 {p}:{i}: {e}')
        _validate_row(r, p, i)
        rows.append(r)
    return rows


def _validate_row(r, p, i):
    """索引作成前の構造検証。オブジェクト型・必須キー・size の型が不正なら Undetermined（ファイル名・行番号付き）。"""
    if not isinstance(r, dict):
        raise Undetermined(f'行がオブジェクトでない {p}:{i}')
    missing = [k for k in ('framework', 'task', 'device', 'size', 'mode') if k not in r]
    if missing:
        raise Undetermined(f'必須キー欠落（{",".join(missing)}） {p}:{i}')
    if any(not isinstance(r[k], str) for k in ('framework', 'task', 'device', 'mode')):
        raise Undetermined(f'framework/task/device/mode が文字列でない {p}:{i}')
    sz = r['size']
    if isinstance(sz, bool) or not isinstance(sz, (int, float)) or sz != sz or sz in (float('inf'), float('-inf')) or sz != int(sz):
        raise Undetermined(f'size が整数でない（{sz!r}） {p}:{i}')


def key(r):
    return (r['framework'], r['task'], r['device'], int(r['size']), r['mode'])


def index(rows):
    """gen_0110_macmini.py の index と同じ。phases 行と（burn 以外の）TF32 行を除き、重複は例外。"""
    d = {}
    for r in rows:
        if r['task'] in ('train_phases', 'infer_phases'):
            continue
        if r.get('tf32') and r['framework'] != 'burn':
            continue
        k = key(r)
        if k in d:
            raise Undetermined(f'重複行: {k}')
        d[k] = r
    return d


def load_runs(series_dir, arms, nruns=NRUNS):
    """run ごとに {arm: 索引} を返す。run 不足・ファイル欠損は Undetermined。"""
    runs = []
    for i in range(1, nruns + 1):
        rd = Path(series_dir) / f'run{i}'
        if not rd.is_dir():
            raise Undetermined(f'run{i} ディレクトリ欠損（run 数不足）')
        opp = index(load(rd / 'results-opp.jsonl') + load(rd / 'results-py.jsonl'))
        entry = {'opp': opp}
        for arm in arms:
            entry[arm] = index(load(rd / f'results-{arm}.jsonl'))
        runs.append(entry)
    return runs


def _valid_median(row, what):
    """median_s が有限の正数であることを検証して返す。NaN・0・負・欠落・非数値は Undetermined。"""
    v = row.get('median_s')
    if not (isinstance(v, (int, float)) and not isinstance(v, bool) and v == v
            and v not in (float('inf'), float('-inf')) and v > 0):
        raise Undetermined(f'{what}: median_s が有限の正数でない（{v!r}）')
    return v


def _parity_fail(row, task, what):
    """GEMM 行の parity_fail_count（非負整数）を返す。欠落・null・不正値は Undetermined。
    train／infer の行は parity 項目を出力しないので 0 扱い。"""
    if task != 'gemm':
        return row.get('parity_fail_count') or 0
    v = row.get('parity_fail_count')
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise Undetermined(f'{what}: parity_fail_count が非負整数でない（{v!r}）')
    return v


def paired_ratio(per_run_me, per_run_opp, task, device, size, required=()):
    """gen_0110_macmini.py の paired_ratio と同式: run ごとの（有効な相手最速 ÷ fandhe reuse）の中央値。"""
    rs, best = [], {}
    for i, (me_idx, opp_idx) in enumerate(zip(per_run_me, per_run_opp), 1):
        me = me_idx.get(('fandhe-ai', task, device, size, 'reuse'))
        if me is None:
            raise Undetermined(f'run{i}: fandhe-ai reuse 欠損 {task} {device} N={size}')
        me_s = _valid_median(me, f'run{i}: fandhe-ai reuse {task} {device} N={size}')
        if _parity_fail(me, task, f'run{i}: fandhe-ai reuse {task} {device} N={size}') > 0:
            raise Undetermined(f'run{i}: fandhe-ai reuse の要素検証が不合格（判定不能）{task} {device} N={size}')
        missing = [fw for fw in required if (fw, task, device, size, 'fresh') not in opp_idx]
        if missing:
            raise Undetermined(f'run{i}: 比較相手の行が欠損（{",".join(missing)}）{task} {device} N={size}')
        comp = []
        for fw in FW_ORDER:
            r = opp_idx.get((fw, task, device, size, 'fresh'))
            if r is None:
                continue
            what = f'run{i}: {fw} fresh {task} {device} N={size}'
            if _parity_fail(r, task, what) == 0:
                comp.append((_valid_median(r, what), fw))
        if not comp:
            raise Undetermined(f'run{i}: 有効な比較相手がない {task} {device} N={size}')
        m, fw = min(comp)
        best[fw] = best.get(fw, 0) + 1
        rs.append(m / me_s)
    top = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return statistics.median(rs), rs, top


def classify(ratio):
    """gen_0110_macmini.py の judge と同じ閾値（1 超 = 1 位、0.90 以上 = 僅差、他 = 負け）。"""
    if ratio > 1:
        return '1 位'
    if ratio >= 0.90:
        return '僅差'
    return '負け'


def arm_stats(runs, arm, task, device, size, required=()):
    me = [r[arm] for r in runs]
    opp = [r['opp'] for r in runs]
    ratio, rs, top = paired_ratio(me, opp, task, device, size, required)
    ms = [_valid_median(r[arm][('fandhe-ai', task, device, size, 'reuse')], f'{arm} reuse {task} {device} N={size}')
          for r in runs]
    fresh = [r[arm].get(('fandhe-ai', task, device, size, 'fresh')) for r in runs]
    fresh_ms = None if any(f is None for f in fresh) else statistics.median(
        _valid_median(f, f'fandhe-ai fresh {task} {device} N={size}') for f in fresh)
    return dict(ratio=ratio, runs=rs, top=top, cls=classify(ratio), ms=ms,
                med_ms=statistics.median(ms) * 1e3,
                fresh_ms=None if fresh_ms is None else fresh_ms * 1e3)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float('inf'), float('-inf'))


def _required_fields(task):
    """checksum は全タスクで必須、parity 項目は出力対象の GEMM で必須（train／infer の行には出力されない）。"""
    return HARD_FIELDS if task == 'gemm' else ('checksum',)


def hard_check(runs, task, device, size):
    """checksum・parity 項目が全 run・両腕・両モードで完全一致するか。

    (mismatch, missing) を返す。missing = 行欠損・必須項目の欠落／非数値（判定不能 = 終了コード 2）、
    mismatch = 存在し妥当な行どうしの値の不一致のみ（正しさの失敗 = 終了コード 3）。"""
    mismatch, missing = [], []
    req = _required_fields(task)
    for mode in ('reuse', 'fresh'):
        base = None
        for i, r in enumerate(runs, 1):
            for arm in ('old', 'new'):
                row = r[arm].get(('fandhe-ai', task, device, size, mode))
                if row is None:
                    missing.append(f'{mode} run{i} {arm}: 行欠損')
                    continue
                bad = [f for f in req if not _is_num(row.get(f))]
                if bad:
                    missing.append(f'{mode} run{i} {arm}: 必須項目の欠落／不正値 {",".join(bad)}')
                    continue
                pf = row.get('parity_fail_count')
                if isinstance(pf, (int, float)) and not isinstance(pf, bool) and pf != 0:
                    # fresh 側を含め fandhe の要素検証不合格は判定不能（RULE: parity_fail_count > 0）
                    # 判定不能として記録したうえで一致比較は継続する（checksum 等の不一致を落とさない。hard 不一致が優先）
                    missing.append(f'{mode} run{i} {arm}: parity_fail_count が 0 でない（{pf!r}。判定不能）')
                sig = tuple(row[f] for f in req)
                if base is None:
                    base = sig
                elif sig != base:
                    diff = [f for f, a, b in zip(req, sig, base) if a != b]
                    mismatch.append(f'{mode} run{i} {arm}: 不一致 {",".join(diff)}')
    return mismatch, missing


def check_series_meta(rule_path, env_info_path, gate_log, require_gate, profile=''):
    """RULE sha256・env_info・ゲートの整合。成立しなければ理由リストを返す（空なら成立）。

    ゲート検証は --require-gate、gb10（ランナーが AB_GATE=1 を必須とする）、env_info の gate=1 の
    いずれかで有効になる（フラグの付け忘れで gate.log 欠損・GATE-TIMEOUT を見逃さない）。"""
    issues = []
    require_gate = require_gate or profile == 'gb10'
    rule = Path(rule_path)
    if not rule.is_file():
        return [f'RULE 欠損: {rule}']
    rule_sha = hashlib.sha256(rule.read_bytes()).hexdigest()
    env = Path(env_info_path)
    if not env.is_file():
        issues.append(f'env_info 欠損: {env}')
    else:
        recorded = [l.split('=', 1)[1].strip() for l in env.read_text().splitlines() if l.startswith('rule_sha256=')]
        if recorded != [rule_sha]:
            issues.append(f'env_info の rule_sha256 が現 RULE と不一致（env={recorded} rule={rule_sha}）')
        if any(l.strip() == 'gate=1' for l in env.read_text().splitlines()):
            require_gate = True
    if require_gate:
        g = Path(gate_log) if gate_log else None
        if g is None or not g.is_file():
            issues.append('gate.log 欠損（ゲート付き系列は必須）')
        else:
            text = g.read_text()
            if 'GATE-INVALID' in text or 'GATE-TIMEOUT' in text:
                issues.append('gate.log に GATE-INVALID／GATE-TIMEOUT')
            passed = {l.split()[0] for l in text.splitlines() if 'gate=pass' in l}
            for r in range(1, NRUNS + 1):
                if f'round{r}' not in passed:
                    issues.append(f'gate.log に round{r} の gate=pass が無い')
    return issues


def ref_ratio(ref_dir, profile, task, device, size):
    """0.11.0 正式系列（別日・別ツリー）の参考値。取れなければ None（参考列なので判定には使わない）。"""
    try:
        runs = []
        for i in range(1, NRUNS + 1):
            rows = []
            for f in REF_FILES[profile]:
                rows += load(Path(ref_dir) / f'run{i}' / f)
            idx = index(rows)
            runs.append(idx)
        ratio, _, _ = paired_ratio(runs, runs, task, device, size)
        ms = statistics.median(r[('fandhe-ai', task, device, size, 'reuse')]['median_s'] for r in runs) * 1e3
        return ms, classify(ratio), ratio
    except (Undetermined, KeyError):
        return None


def fmt_runs(rs):
    return '[' + ' '.join(f'{x:.2f}' for x in rs) + ']'


def build(args):
    """(Markdown 文字列, 終了コード) を返す。"""
    cells = CELLS[args.profile]
    out, undetermined, hard = [], [], []
    meta = check_series_meta(args.rule, args.env_info, args.gate_log, args.require_gate, args.profile)
    out.append(f'# bench-fandhe 是正前後の併記（#3266）: {args.profile} / {args.label}\n')
    out.append('旧条件腕 = #3265 直前、新条件腕 = #3265。判定式は 0.11.0 スコアボードと同一（同一 run 内の最速相手比の 5 run 中央値）。'
               '腕比 new/old は参考で、合格ラインはない。\n')
    for m in meta:
        undetermined.append(f'系列: {m}')
    try:
        runs = load_runs(args.series_dir, ('old', 'new'))
    except Undetermined as e:
        undetermined.append(f'系列: {e}')
        runs = None
    out.append('| セル | 0.11.0 参考 (ms／区分) | 旧 ms | 旧 相手比／区分 | 新 ms | 新 相手比／区分 | 腕比 new/old（run 別） | 最速相手 | hard | fresh 旧→新 ms |')
    out.append('|---|---|---|---|---|---|---|---|---|---|')
    for task, device, size in cells:
        name = f'{task} {device}' + (f' N={size}' if task == 'gemm' else '')
        ref = ref_ratio(args.ref_dir, args.profile, task, device, size) if args.ref_dir else None
        refs = '—' if ref is None else f'{ref[0]:.3f} / {ref[1]} ({ref[2]:.2f})'
        if runs is None:
            out.append(f'| {name} | {refs} | 判定不能 | | | | | | | |')
            continue
        # hard 条件は勝敗の判定可否から独立に評価する（判定不能セルでも不一致を落とさない。終了コード 3 が優先）。
        probs, miss = hard_check(runs, task, device, size)
        if probs:
            hard.append(f'{name}: ' + '; '.join(probs))
        if miss:
            undetermined.append(f'{name}: hard 検証項目が判定不能: ' + '; '.join(miss))
        hs = 'NG' if probs else ('判定不能' if miss else 'OK')
        req = expected_opponents(args.profile, task, device, size)
        try:
            o = arm_stats(runs, 'old', task, device, size, req)
            n = arm_stats(runs, 'new', task, device, size, req)
        except Undetermined as e:
            undetermined.append(f'{name}: {e}')
            out.append(f'| {name} | {refs} | 判定不能: {e} | | | | | | {hs} | |')
            continue
        arm_rs = [nm / om for nm, om in zip(n['ms'], o['ms'])]
        arm_med = statistics.median(n['ms']) / statistics.median(o['ms'])
        top = o['top'] if o['top'] == n['top'] else f'{o["top"]}→{n["top"]}'
        fresh = '—' if o['fresh_ms'] is None or n['fresh_ms'] is None else f'{o["fresh_ms"]:.3f}→{n["fresh_ms"]:.3f}'
        out.append(f'| {name} | {refs} | {o["med_ms"]:.3f} | {o["ratio"]:.2f} {o["cls"]} {fmt_runs(o["runs"])} '
                   f'| {n["med_ms"]:.3f} | {n["ratio"]:.2f} {n["cls"]} {fmt_runs(n["runs"])} '
                   f'| {arm_med:.3f} {fmt_runs(arm_rs)} | {top} | {hs} | {fresh} |')
    out.append('')
    if hard:
        out.append('## hard 条件の不一致（正しさの失敗。所有者へ報告。tolerance は変えない）\n')
        out.extend(f'- {h}' for h in hard)
        out.append('')
    if undetermined:
        out.append('## undetermined\n')
        out.extend(f'- {u}' for u in undetermined)
        out.append('')
    code = 3 if hard else (2 if undetermined else 0)
    out.append(f'exit={code}')
    return '\n'.join(out) + '\n', code


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--profile', required=True, choices=sorted(CELLS))
    p.add_argument('--label', required=True, help='系列名（表の見出し用。例: series-b）')
    p.add_argument('--series-dir', required=True)
    p.add_argument('--env-info', required=True)
    p.add_argument('--rule', required=True)
    p.add_argument('--gate-log', default='')
    p.add_argument('--require-gate', action='store_true', help='ゲート付き系列。gate.log の 5 ラウンド pass を必須にする（gb10 と env_info の gate=1 では指定なしでも適用）')
    p.add_argument('--ref-dir', default='', help='0.11.0 正式系列の run ディレクトリ群の親（参考列。任意）')
    args = p.parse_args(argv)
    text, code = build(args)
    sys.stdout.write(text)
    return code


if __name__ == '__main__':
    sys.exit(main())
