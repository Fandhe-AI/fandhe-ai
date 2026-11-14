#!/usr/bin/env python3
"""G-CUDA-TRN の stream 種別 4 腕 A/B（イシュー #4299）の判定器。標準ライブラリのみ（`python3 -I` で動く）。

役割: `run_diag_4299_cudatrn_stream.sh` が出力する 1 系列（round1〜round16 の <腕>-<mode>.jsonl と launch.tsv）を読み、
ラウンドごとの対の比（reuse の median_s の時間比 B/A・C/A・B/C・D/A）から、issue 本文の R1（再現）・R2（原因）と
hard 条件（同じ腕の 16 ラウンドで checksum が完全一致）を判定して Markdown を出す。fresh は参考として記録だけする。
規則の正本: `docs/perf/logs/diag-4299-cudatrn-stream/RULE.txt`（計測より前に固定する。事後に緩めない）。
判定式・腕・ラウンド数は issue #4299 の本文のとおりで、本ファイルはそれを逐語に実装する（変えるときは RULE ごと別名で取り直す）。

腕: A = registry の fandhe-ai =0.14.0（環境変数なし）、B = BASE の path patch（なし）、C = B と同じバイナリ +
FANDHE_AI_CUDA_GRAPH_INFER=0、D = A と同じバイナリ + FANDHE_AI_CUDA_GRAPH_INFER=1。
カウントは「比 > 1 のラウンド数」だけを数える（同値の 1 は > 1 に入れない）。

終了コード: 0 = 判定済み（R1・R2 の結論が「再現せず」「判定できない」でも 0）/ 3 = 同じ腕の中で checksum が一致しない
（reuse・fresh のどちらでも。2 より優先）/ 2 = 行の欠け・起動記録の不一致・RULE／道具／ゲートの不成立などで判定できない。
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from pathlib import Path

NROUNDS = 16
ARMS = ('A', 'B', 'C', 'D')
MODES = ('reuse', 'fresh')
CELL = dict(framework='fandhe-ai', task='train', device='cuda', size=64)
EXPECT_VERSION = '0.14.0'  # A・B とも 0.14.0 を名乗る（BASE の workspace.version も 0.14.0）。腕の同定はパスとバイナリの sha256 で行う
GRAPH_ENV = 'FANDHE_AI_CUDA_GRAPH_INFER'
ARM_ENV = {'A': 'none', 'B': 'none', 'C': f'{GRAPH_ENV}=0', 'D': f'{GRAPH_ENV}=1'}
ARM_BUILD = {'A': 'before', 'B': 'after', 'C': 'after', 'D': 'before'}
# 4×4 のラテン方格（Williams 型）。round r は (r-1) mod 4 行目。ランナーの LATIN・RULE の [ラテン方格] と同一。
LATIN = (('A', 'B', 'D', 'C'), ('B', 'C', 'A', 'D'), ('C', 'D', 'B', 'A'), ('D', 'A', 'C', 'B'))
THRESH = 12  # 12/16 以上（片側符号検定で p≈0.038）
LOW = 4      # 4/16 以下で「逆向き」
PLACEHOLDER = '<要記入'
RATIOS = (('B/A', 'B', 'A'), ('C/A', 'C', 'A'), ('B/C', 'B', 'C'), ('D/A', 'D', 'A'))
# RULE が固定する道具（リポジトリルート相対。順序も含めランナーの TOOL_FILES と同一）。
TOOL_FILES = (
    'scripts/bench/framework-compare/run_diag_4299_cudatrn_stream.sh',
    'scripts/bench/framework-compare/judge_diag_4299.py',
    'docs/perf/logs/diag-4299-cudatrn-stream/collect.sh',
)
JUDGE_REL = 'scripts/bench/framework-compare/judge_diag_4299.py'
DEFAULT_ROOT = Path(__file__).resolve().parents[3]

R1_REPRO = '後退を再現'
R1_REVERSE = '逆向き'
R1_NONE = '再現せず（ノイズの範囲）'
R2_STREAM = 'stream の切り替えが原因（新旧どちらのコードでも再現）'
R2_COMBO = 'stream の切り替えと新しいコードの組み合わせ'
R2_OTHER_DIFF = 'stream 以外（stream 以外の差分が原因）'
R2_OTHER_UNDET = 'stream 以外（判定できない）'


class Undetermined(Exception):
    """系列を判定できない理由（握りつぶさず表に出す）。"""


# ---- 判定式（issue #4299 本文の R1・R2。数え方だけを入力に取る純粋関数）

def decide_r1(n_ba):
    """B/A > 1 のラウンド数から R1 を返す。"""
    if n_ba >= THRESH:
        return R1_REPRO
    if n_ba <= LOW:
        return R1_REVERSE
    return R1_NONE


def decide_r2(r1, n_bc, n_da, n_ca):
    """R1 が「後退を再現」のときだけ R2 を返す（それ以外は None = 判定しない）。"""
    if r1 != R1_REPRO:
        return None
    if n_bc >= THRESH:
        return R2_STREAM if n_da >= THRESH else R2_COMBO
    return R2_OTHER_DIFF if n_ca >= THRESH else R2_OTHER_UNDET


# ---- 読み込み（寛容パースをしない）

def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_row(path):
    """1 ファイル 1 行。欠損・空・複数行・JSON 不正・セル違い・版違い・diag 行は Undetermined。"""
    p = Path(path)
    if not p.is_file():
        raise Undetermined(f'ファイル欠損: {p.parent.name}/{p.name}')
    lines = [l for l in p.read_text().splitlines() if l.strip()]
    if len(lines) != 1:
        raise Undetermined(f'行数が 1 でない（{len(lines)}）: {p.parent.name}/{p.name}')
    try:
        r = json.loads(lines[0])
    except json.JSONDecodeError as e:
        raise Undetermined(f'JSON 不正 {p.parent.name}/{p.name}: {e}')
    if not isinstance(r, dict):
        raise Undetermined(f'行がオブジェクトでない {p.parent.name}/{p.name}')
    for k, v in CELL.items():
        if r.get(k) != v or (k == 'size' and isinstance(r.get(k), bool)):
            raise Undetermined(f'{k} が {v!r} でない（{r.get(k)!r}）: {p.parent.name}/{p.name}')
    if r.get('version') != EXPECT_VERSION:
        raise Undetermined(f'version が {EXPECT_VERSION} でない（{r.get("version")!r}）: {p.parent.name}/{p.name}')
    if any(k.startswith('diag') or k in ('graph', 'phases', 'tf32') for k in r):
        raise Undetermined(f'計測に使わない印（diag・graph・phases・tf32）付きの行: {p.parent.name}/{p.name}')
    ms = r.get('median_s')
    if not is_num(ms) or ms <= 0:
        raise Undetermined(f'median_s が正の有限数でない（{ms!r}）: {p.parent.name}/{p.name}')
    if not is_num(r.get('checksum')):
        raise Undetermined(f'checksum が有限数でない（{r.get("checksum")!r}）: {p.parent.name}/{p.name}')
    return r


def check_launch(rd, rnd):
    """launch.tsv（起動ごとに 1 行）が、方格の順・腕と環境変数・ビルドの対応・rc 0 のとおりかを照合する。"""
    p = Path(rd) / 'launch.tsv'
    if not p.is_file():
        raise Undetermined(f'round{rnd}: launch.tsv 欠損')
    rows = [l.split('\t') for l in p.read_text().splitlines() if l.strip()]
    order = LATIN[(rnd - 1) % 4]
    want = [(m, a) for m in MODES for a in order]
    if len(rows) != len(want):
        raise Undetermined(f'round{rnd}: launch.tsv の行数が {len(want)} でない（{len(rows)}）')
    for i, (f, (mode, arm)) in enumerate(zip(rows, want), 1):
        if len(f) != 9:
            raise Undetermined(f'round{rnd}: launch.tsv {i} 行目の列数が 9 でない')
        if f[0] != str(rnd) or f[1] != str(i):
            raise Undetermined(f'round{rnd}: launch.tsv {i} 行目の round／順位が違う（{f[0]}／{f[1]}）')
        if (f[2], f[3]) != (arm, mode):
            raise Undetermined(f'round{rnd}: 起動順が方格と違う（{i} 番目: 期待 {arm}-{mode}、記録 {f[2]}-{f[3]}）')
        if f[4] != ARM_ENV[arm]:
            raise Undetermined(f'round{rnd}: {arm}-{mode} に渡した環境変数が違う（期待 {ARM_ENV[arm]}、記録 {f[4]}）')
        if f[5] != ARM_BUILD[arm]:
            raise Undetermined(f'round{rnd}: {arm}-{mode} のバイナリが違う（期待 {ARM_BUILD[arm]}、記録 {f[5]}）')
        if f[6] != '0':
            raise Undetermined(f'round{rnd}: {arm}-{mode} の rc が 0 でない（{f[6]}）')


def load_series(series_dir):
    """{(round, arm, mode): row}。round 集合は round1〜round16 ちょうど（欠損・追加は Undetermined）。"""
    sd = Path(series_dir)
    if not sd.is_dir():
        raise Undetermined(f'系列ディレクトリ欠損: {sd.name}')
    expected = {f'round{i}' for i in range(1, NROUNDS + 1)}
    found = {c.name for c in sd.iterdir() if c.is_dir() and re.fullmatch(r'round\d+', c.name)}
    if found != expected:
        raise Undetermined(f'round 集合が round1〜round{NROUNDS} と一致しない（欠損 {sorted(expected - found)} 追加 {sorted(found - expected)}）')
    rows, issues = {}, []
    for rnd in range(1, NROUNDS + 1):
        rd = sd / f'round{rnd}'
        try:
            check_launch(rd, rnd)
        except Undetermined as e:
            issues.append(str(e))
        for arm in ARMS:
            for mode in MODES:
                try:
                    r = load_row(rd / f'{arm}-{mode}.jsonl')
                except Undetermined as e:
                    issues.append(str(e))
                    continue
                if r.get('mode') != mode:
                    issues.append(f'round{rnd} {arm}-{mode}: mode が違う（{r.get("mode")!r}）')
                    continue
                rows[(rnd, arm, mode)] = r
    return rows, issues


# ---- RULE・道具・env_info・ゲートの整合

def rule_block(text, name):
    """RULE の `# >>> <name>` と `# <<< <name>` の間の行（ランナーの rule_block と同じ切り出し）。マーカーが無ければ None。"""
    lines = text.split('\n')
    try:
        b = lines.index(f'# >>> {name}')
        e = lines.index(f'# <<< {name}', b + 1)
    except ValueError:
        return None
    return lines[b + 1:e]


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_tool_sha256(block, root, judge_file=None):
    """[道具の sha256] ブロック（`<sha256>  <相対パス>`。順序・集合とも TOOL_FILES と一致）と実ファイルの照合。理由リストを返す。"""
    if not block:
        return ['RULE に [道具の sha256] ブロック（# >>> tool-sha256）が無い']
    if any(not l for l in block):
        return ['RULE の [道具の sha256] ブロックに空行がある']
    want = [l.split('  ', 1) for l in block]
    if [w[1] if len(w) == 2 else '' for w in want] != list(TOOL_FILES):
        return [f'RULE の [道具の sha256] の対象が TOOL_FILES と一致しない（RULE={[w[-1] for w in want]}）']
    issues = []
    for sha, rel in want:
        f = Path(root) / rel
        if not f.is_file():
            issues.append(f'道具ファイルが無い: {rel}')
        elif sha256_file(f) != sha:
            issues.append(f'道具の sha256 が RULE と不一致: {rel}')
    if judge_file is not None:
        pinned = dict((rel, sha) for sha, rel in want).get(JUDGE_REL)
        if not Path(judge_file).is_file() or sha256_file(judge_file) != pinned:
            issues.append('実行中の判定器が RULE の sha256 と不一致')
    return issues


def env_values(lines, key):
    return [l.split('=', 1)[1].strip() for l in lines if l.startswith(key + '=')]


def check_meta(rule_path, env_info, gate_log, compute_log, root=None, judge_file=None):
    """RULE（未記入・道具の sha256・ラテン方格・BASE SHA）・env_info・gate.log・compute_apps.log の整合。理由リストを返す。

    root = 道具の実ファイルを探すリポジトリルート（既定は本ファイルの 3 つ上。self-test は一時ツリーを渡す）。
    judge_file = 実行中の判定器の実体（CLI 経路では常に本ファイル。self-test は None）。
    """
    issues = []
    rule = Path(rule_path)
    if not rule.is_file():
        return ['RULE.txt 欠損']
    rt = rule.read_text()
    if PLACEHOLDER in rt:
        issues.append(f'RULE に {PLACEHOLDER} が残っている')
    issues += check_tool_sha256(rule_block(rt, 'tool-sha256'), DEFAULT_ROOT if root is None else root, judge_file)
    latin = rule_block(rt, 'latin-square')
    if latin != [' '.join(r) for r in LATIN]:
        issues.append('RULE の [ラテン方格] が判定器の LATIN と一致しない')
    m = re.search(r'^- BASE = ([0-9a-f]{40})$', rt, re.M)
    base_sha = m.group(1) if m else None
    if base_sha is None:
        issues.append("RULE から BASE SHA（'- BASE = <40hex>'）を読めない")
    rule_sha = hashlib.sha256(rule.read_bytes()).hexdigest()
    env = Path(env_info)
    if not env.is_file():
        issues.append('env_info.txt 欠損')
    else:
        lines = env.read_text().splitlines()
        want = {'rule_sha256': rule_sha, 'fandhe_ai_rust_ai_env_vars': '0', 'rayon_num_threads': 'unset',
                'tool_sha256_check': 'ok', 'latin_square_check': 'ok', 'dry_run': '0', 'rounds': str(NROUNDS),
                'profile': 'gb10', 'runner_base_diff_crates_bench_fandhe': '0',
                'env_A': ARM_ENV['A'], 'env_B': ARM_ENV['B'], 'env_C': ARM_ENV['C'], 'env_D': ARM_ENV['D']}
        if base_sha:
            want['base_sha'] = base_sha
            want['rev_stamp_base'] = base_sha
        for k, v in want.items():
            got = env_values(lines, k)
            if got != [v]:
                issues.append(f'env_info の {k} が期待と違う（期待 {v}、記録 {got}）')
        shas = {a: env_values(lines, f'bin_sha256_{a}') for a in ARMS}
        if any(len(v) != 1 or not re.fullmatch(r'[0-9a-f]{64}', v[0]) for v in shas.values()):
            issues.append('env_info の bin_sha256_A〜D が 64 桁の 16 進でない')
        else:
            s = {a: v[0] for a, v in shas.items()}
            if s['A'] == s['B']:
                issues.append('A と B のバイナリが同一（腕が分かれていない）')
            if s['C'] != s['B'] or s['D'] != s['A']:
                issues.append('C は B、D は A と同じバイナリでない')
    g = Path(gate_log) if gate_log else None
    if g is None or not g.is_file():
        issues.append('gate.log 欠損（ゲートは必須）')
    else:
        text = g.read_text()
        for bad in ('GATE-INVALID', 'GATE-TIMEOUT', 'EXCLUSIVE-INVALID', 'EXCLUSIVE-UNMET'):
            if bad in text:
                issues.append(f'gate.log に {bad}')
        passed = [l.split()[0] for l in text.splitlines() if ' gate=pass ' in f'{l} ']
        for r in range(1, NROUNDS + 1):
            if passed.count(f'round{r}') != 1:
                issues.append(f'gate.log の round{r} の gate=pass が 1 件でない（{passed.count(f"round{r}")} 件）')
    c = Path(compute_log) if compute_log else None
    if c is None or not c.is_file():
        issues.append('compute_apps.log 欠損（GB10 の専有確認は必須）')
    else:
        blocks = [b for b in c.read_text().split('\n') if b.strip()]
        heads = [b for b in blocks if re.search(r' rc=\d+$', b)]
        if len(heads) != NROUNDS:
            issues.append(f'compute_apps.log の記録が {len(heads)} 件（期待 {NROUNDS}）')
        if any(not b.endswith(' rc=0') for b in heads):
            issues.append('compute_apps.log に rc != 0 の記録')
        if len(blocks) != len(heads):
            issues.append('compute_apps.log に compute プロセスの行がある（専有未達）')
    return issues


# ---- 集計

def hard_check(rows):
    """同じ腕の 16 ラウンドで checksum が完全一致するか（reuse・fresh それぞれ）。(不一致の理由リスト, {(arm, mode): checksum}) を返す。"""
    mism, base = [], {}
    for arm in ARMS:
        for mode in MODES:
            vals = [(rnd, rows[(rnd, arm, mode)]['checksum']) for rnd in range(1, NROUNDS + 1) if (rnd, arm, mode) in rows]
            if not vals:
                continue
            base[(arm, mode)] = vals[0][1]
            bad = [rnd for rnd, v in vals if v != vals[0][1]]
            if bad:
                mism.append(f'{arm}-{mode}: round{vals[0][0]} の checksum {vals[0][1]!r} と一致しないラウンド {bad}')
    return mism, base


def cross_notes(base):
    """腕の間の checksum の差。B と C・A と D は同じライブラリなので差は報告事項、A と B は記録だけ。"""
    report, record = [], []
    for mode in MODES:
        for x, y, sink in (('B', 'C', report), ('A', 'D', report), ('A', 'B', record)):
            bx, by = base.get((x, mode)), base.get((y, mode))
            if bx is None or by is None:
                continue
            sink.append((f'{x}/{y}', mode, bx == by, bx, by))
    return report, record


def round_ratios(rows, mode):
    """[(round, {'B/A': x, ...})]。全腕が揃ったラウンドだけ。"""
    out = []
    for rnd in range(1, NROUNDS + 1):
        if not all((rnd, a, mode) in rows for a in ARMS):
            continue
        ms = {a: rows[(rnd, a, mode)]['median_s'] for a in ARMS}
        out.append((rnd, {name: ms[x] / ms[y] for name, x, y in RATIOS}))
    return out


def count_gt1(rr, name):
    return sum(1 for _, d in rr if d[name] > 1.0)


def build(args, root=None, judge_file=None):
    """(Markdown, 終了コード, ratios TSV 行) を返す。"""
    out, undet = [], []
    out.append('# G-CUDA-TRN の stream 種別 4 腕 A/B（#4299）: gb10\n')
    out.append('A = registry の fandhe-ai =0.14.0、B = BASE の path patch、C = B + `FANDHE_AI_CUDA_GRAPH_INFER=0`、'
               'D = A + `FANDHE_AI_CUDA_GRAPH_INFER=1`。比は同じラウンドの reuse の median_s の時間比（> 1 は分子の腕が遅い）。'
               '腕と stream 種別の対応はコードを読んだ結果で、実行時には観測していない。fresh は参考。\n')
    undet += [f'系列: {m}' for m in check_meta(args.rule, args.env_info, args.gate_log, args.compute_log, root, judge_file)]
    try:
        rows, issues = load_series(args.series_dir)
    except Undetermined as e:
        rows, issues = {}, [str(e)]
    undet += [f'行: {m}' for m in issues]
    mism, base = hard_check(rows)
    rr = round_ratios(rows, 'reuse')
    rf = round_ratios(rows, 'fresh')
    tsv = []

    out.append('## ラウンドごとの比（reuse・median_s の時間比）\n')
    out.append('| round | 順番 | A µs | B µs | C µs | D µs | B/A | C/A | B/C | D/A |')
    out.append('|---|---|---|---|---|---|---|---|---|---|')
    rrd = dict(rr)
    for rnd in range(1, NROUNDS + 1):
        order = ''.join(LATIN[(rnd - 1) % 4])
        if rnd not in rrd:
            out.append(f'| {rnd} | {order} | 欠け | | | | | | | |')
            continue
        us = [f"{rows[(rnd, a, 'reuse')]['median_s'] * 1e6:.3f}" for a in ARMS]
        rs = [f'{rrd[rnd][n]:.5f}' for n, _, _ in RATIOS]
        out.append(f'| {rnd} | {order} | ' + ' | '.join(us + rs) + ' |')
        tsv.append('\t'.join([str(rnd), order] + rs))
    out.append('')

    complete = not undet and len(rr) == NROUNDS
    out.append('## 集計（16 ラウンドの中央値と「> 1」のラウンド数）\n')
    out.append('| 比 | 中央値 | > 1 のラウンド数 |')
    out.append('|---|---|---|')
    for n, _, _ in RATIOS:
        if rr:
            out.append(f'| {n} | {statistics.median(d[n] for _, d in rr):.5f} | {count_gt1(rr, n)}/{len(rr)} |')
    out.append('')

    out.append('## 判定\n')
    if complete:
        counts = {n: count_gt1(rr, n) for n, _, _ in RATIOS}
        r1 = decide_r1(counts['B/A'])
        r2 = decide_r2(r1, counts['B/C'], counts['D/A'], counts['C/A'])
        out.append(f"- R1（再現）: **{r1}**（B/A > 1 が {counts['B/A']}/{NROUNDS}。{THRESH} 以上で再現、{LOW} 以下で逆向き）")
        if r2 is None:
            out.append('- R2（原因）: 判定しない（R1 で再現したときだけ判定する）')
        else:
            out.append(f"- R2（原因）: **{r2}**（B/C > 1 が {counts['B/C']}/{NROUNDS}、D/A > 1 が {counts['D/A']}/{NROUNDS}、"
                       f"C/A > 1 が {counts['C/A']}/{NROUNDS}）")
    else:
        out.append('- R1・R2: 判定しない（下の undetermined を参照。16 ラウンドの全行が揃わない系列は採らない）')
    out.append('')

    out.append('## hard 条件（checksum）\n')
    if mism:
        out.append('- **同じ腕の中で checksum が一致しない（正しさの失敗として所有者へ報告。tolerance は変えない）**')
        out.extend(f'  - {m}' for m in mism)
    elif rows:
        out.append('- 同じ腕の中（reuse・fresh それぞれ）: 揃った行はすべて一致')
    report, record = cross_notes(base)
    for pair, mode, same, bx, by in report:
        out.append(f"- {pair}（同じライブラリ・{mode}）: {'一致' if same else f'**差あり（報告事項）** {bx!r} / {by!r}'}")
    for pair, mode, same, bx, by in record:
        out.append(f"- {pair}（記録のみ・{mode}）: {'一致' if same else f'差あり {bx!r} / {by!r}'}")
    out.append('')

    out.append('## fresh（参考。判定には使わない）\n')
    if rf:
        out.append('| 比 | 中央値 | > 1 のラウンド数 | ラウンドごと |')
        out.append('|---|---|---|---|')
        for n, _, _ in RATIOS:
            vals = ' '.join(f'{d[n]:.4f}' for _, d in rf)
            out.append(f'| {n} | {statistics.median(d[n] for _, d in rf):.5f} | {count_gt1(rf, n)}/{len(rf)} | {vals} |')
    out.append('')

    if undet:
        out.append('## undetermined\n')
        out.extend(f'- {u}' for u in undet)
        out.append('')
    code = 3 if mism else (2 if not complete else 0)
    out.append(f'exit={code}')
    return '\n'.join(out) + '\n', code, tsv


# ---- self-test（合成データ。R1・R2 の全分岐・境界・hard の失敗・行の欠け・起動記録・ゲート・RULE の不整合）

def self_test():
    import unittest

    base_sha = 'b' * 40

    def rule_text(tool_block, latin=None):
        latin = latin if latin is not None else [' '.join(r) for r in LATIN]
        return ('RULE fixed-at-utc: 2026-01-01T00:00:00Z\n'
                f'- BASE = {base_sha}\n'
                '# >>> latin-square\n' + '\n'.join(latin) + '\n# <<< latin-square\n'
                '# >>> tool-sha256\n' + '\n'.join(tool_block) + '\n# <<< tool-sha256\n')

    class Tree:
        """一時ツリー（道具の偽ファイル・RULE・系列）。ms(rnd, arm) で reuse の median_s を与える。"""

        def __init__(self, tmp, ms, fresh_ms=None, checksum=None):
            self.root = Path(tmp)
            for rel in TOOL_FILES:
                (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
                (self.root / rel).write_text(f'fake {rel}\n')
            block = [f'{sha256_file(self.root / rel)}  {rel}' for rel in TOOL_FILES]
            self.rule = self.root / 'RULE.txt'
            self.rule.write_text(rule_text(block))
            self.sd = self.root / 'raw'
            self.sd.mkdir()
            checksum = checksum or (lambda rnd, arm, mode: 0.080541)
            fresh_ms = fresh_ms or (lambda rnd, arm: 5e-4)
            for rnd in range(1, NROUNDS + 1):
                rd = self.sd / f'round{rnd}'
                rd.mkdir()
                lines, pos = [], 0
                for mode in MODES:
                    for arm in LATIN[(rnd - 1) % 4]:
                        pos += 1
                        lines.append('\t'.join([str(rnd), str(pos), arm, mode, ARM_ENV[arm], ARM_BUILD[arm], '0',
                                                '2026-01-01T00:00:00Z', '2026-01-01T00:00:01Z']))
                        m = ms(rnd, arm) if mode == 'reuse' else fresh_ms(rnd, arm)
                        row = dict(CELL, version=EXPECT_VERSION, median_s=m, checksum=checksum(rnd, arm, mode),
                                   warmup=20, iters=80, mode=mode)
                        (rd / f'{arm}-{mode}.jsonl').write_text(json.dumps(row) + '\n')
                (rd / 'launch.tsv').write_text('\n'.join(lines) + '\n')
            self.env = self.root / 'env_info.txt'
            sa, sb = 'a' * 64, 'c' * 64
            self.env.write_text('\n'.join([
                'profile=gb10', 'dry_run=0', f'rule_sha256={sha256_file(self.rule)}', 'tool_sha256_check=ok',
                'latin_square_check=ok', 'fandhe_ai_rust_ai_env_vars=0', 'rayon_num_threads=unset',
                f'base_sha={base_sha}', f'rev_stamp_base={base_sha}', 'runner_base_diff_crates_bench_fandhe=0',
                f'bin_sha256_A={sa}', f'bin_sha256_B={sb}', f'bin_sha256_C={sb}', f'bin_sha256_D={sa}',
                f'env_A={ARM_ENV["A"]}', f'env_B={ARM_ENV["B"]}', f'env_C={ARM_ENV["C"]}', f'env_D={ARM_ENV["D"]}',
                f'rounds={NROUNDS}']) + '\n')
            self.gate = self.root / 'gate.log'
            self.gate.write_text(''.join(f'round{r} gate=pass threshold=1.0 waited_s=0 load1=0.5\n' for r in range(1, NROUNDS + 1)))
            self.compute = self.root / 'compute_apps.log'
            self.compute.write_text(''.join(f'2026-01-01T00:00:{r:02d}Z rc=0\n\n' for r in range(1, NROUNDS + 1)))

        def run(self):
            a = argparse.Namespace(rule=str(self.rule), env_info=str(self.env), gate_log=str(self.gate),
                                   compute_log=str(self.compute), series_dir=str(self.sd))
            return build(a, root=self.root, judge_file=None)

    def by_count(n, slow=1.01, fast=0.99):
        """先頭 n ラウンドだけ slow、残りは fast を返す係数関数。"""
        return lambda rnd: slow if rnd <= n else fast

    class T(unittest.TestCase):
        def mk(self, ms, **kw):
            self._td = tempfile.TemporaryDirectory()
            self.addCleanup(self._td.cleanup)
            return Tree(self._td.name, ms, **kw)

        # R1 の全分岐と境界（12/11、4/5、0/16）
        def test_r1_branches(self):
            for n, want in ((16, R1_REPRO), (12, R1_REPRO), (11, R1_NONE), (8, R1_NONE), (5, R1_NONE), (4, R1_REVERSE), (0, R1_REVERSE)):
                self.assertEqual(decide_r1(n), want, n)

        # R2 の全分岐（12/11 の境界を含む）と、R1 が再現でないときは判定しないこと
        def test_r2_branches(self):
            self.assertEqual(decide_r2(R1_REPRO, 12, 12, 0), R2_STREAM)
            self.assertEqual(decide_r2(R1_REPRO, 16, 11, 16), R2_COMBO)
            self.assertEqual(decide_r2(R1_REPRO, 11, 16, 12), R2_OTHER_DIFF)
            self.assertEqual(decide_r2(R1_REPRO, 11, 16, 11), R2_OTHER_UNDET)
            for r1 in (R1_NONE, R1_REVERSE):
                self.assertIsNone(decide_r2(r1, 16, 16, 16))

        # 端から端まで: R1 再現 + R2 stream（B/A・B/C・D/A が 12/16）
        def test_e2e_stream(self):
            f = by_count(12)
            t = self.mk(lambda rnd, arm: {'A': 1e-4, 'B': 1e-4 * f(rnd), 'C': 1e-4, 'D': 1e-4 * f(rnd)}[arm])
            md, code, tsv = t.run()
            self.assertEqual(code, 0, md)
            self.assertIn(f'**{R1_REPRO}**', md)
            self.assertIn(f'**{R2_STREAM}**', md)
            self.assertEqual(len(tsv), NROUNDS)

        # 端から端まで: R1 再現 + R2 組み合わせ（D/A は 11/16）
        def test_e2e_combo(self):
            fb, fd = by_count(12), by_count(11)
            t = self.mk(lambda rnd, arm: {'A': 1e-4, 'B': 1e-4 * fb(rnd), 'C': 1e-4, 'D': 1e-4 * fd(rnd)}[arm])
            md, code, _ = t.run()
            self.assertEqual(code, 0, md)
            self.assertIn(f'**{R2_COMBO}**', md)

        # 端から端まで: R1 再現 + R2 stream 以外（B/C 0/16・C/A 16/16）と判定できない（C/A 11/16）
        def test_e2e_other(self):
            t = self.mk(lambda rnd, arm: {'A': 1e-4, 'B': 1.02e-4, 'C': 1.02e-4, 'D': 1e-4}[arm])
            md, code, _ = t.run()
            self.assertEqual(code, 0, md)
            self.assertIn(f'**{R2_OTHER_DIFF}**', md)
            # B/A 16/16・C/A 11/16・B/C 5/16（C が B より遅いラウンドが 11）
            fc = by_count(11, slow=1.03, fast=0.99)
            t = self.mk(lambda rnd, arm: {'A': 1e-4, 'B': 1.02e-4, 'C': 1e-4 * fc(rnd), 'D': 1e-4}[arm])
            md, code, _ = t.run()
            self.assertEqual(code, 0, md)
            self.assertIn(f'**{R2_OTHER_UNDET}**', md)

        # 端から端まで: 再現せず（11/16）・逆向き（4/16）では R2 を判定しない
        def test_e2e_no_repro_and_reverse(self):
            for n, want in ((11, R1_NONE), (4, R1_REVERSE)):
                f = by_count(n)
                t = self.mk(lambda rnd, arm: {'A': 1e-4, 'B': 1e-4 * f(rnd), 'C': 1e-4, 'D': 1e-4}[arm])
                md, code, _ = t.run()
                self.assertEqual(code, 0, md)
                self.assertIn(f'**{want}**', md)
                self.assertIn('R2（原因）: 判定しない', md)

        # 同値（比がちょうど 1）は「> 1」に数えない
        def test_tie_is_not_gt1(self):
            t = self.mk(lambda rnd, arm: 1e-4)
            md, code, _ = t.run()
            self.assertEqual(code, 0, md)
            self.assertIn(f'**{R1_REVERSE}**', md)
            self.assertIn('| B/A | 1.00000 | 0/16 |', md)

        # hard: 同じ腕の中の checksum 不一致（reuse・fresh のどちらでも）は exit 3。行の欠けより優先
        def test_hard_mismatch(self):
            for bad_mode in MODES:
                cs = lambda rnd, arm, mode, bm=bad_mode: 0.5 if (arm == 'C' and mode == bm and rnd == 7) else 0.080541
                t = self.mk(lambda rnd, arm: 1e-4, checksum=cs)
                md, code, _ = t.run()
                self.assertEqual(code, 3, md)
                self.assertIn(f'C-{bad_mode}', md)
            cs = lambda rnd, arm, mode: 0.5 if (arm == 'A' and rnd == 2) else 0.080541
            t = self.mk(lambda rnd, arm: 1e-4, checksum=cs)
            (t.sd / 'round9' / 'B-reuse.jsonl').unlink()
            md, code, _ = t.run()
            self.assertEqual(code, 3, md)

        # 腕の間: B と C・A と D の差は報告事項（exit は 0 のまま）、A と B の差は記録だけ
        def test_cross_arm_checksum_notes(self):
            cs = lambda rnd, arm, mode: {'A': 1.0, 'B': 2.0, 'C': 3.0, 'D': 1.0}[arm]
            t = self.mk(lambda rnd, arm: 1e-4, checksum=cs)
            md, code, _ = t.run()
            self.assertEqual(code, 0, md)
            self.assertIn('B/C（同じライブラリ・reuse）: **差あり（報告事項）**', md)
            self.assertIn('A/D（同じライブラリ・reuse）: 一致', md)
            self.assertIn('A/B（記録のみ・reuse）: 差あり', md)

        # 行の欠け・空・複数行・セル違い・追加ラウンドは exit 2（R1・R2 を判定しない）
        def test_missing_and_bad_rows(self):
            def mutate(fn):
                t = self.mk(lambda rnd, arm: 1.01e-4 if arm == 'B' else 1e-4)
                fn(t)
                md, code, _ = t.run()
                self.assertEqual(code, 2, md)
                self.assertIn('R1・R2: 判定しない', md)
            mutate(lambda t: (t.sd / 'round5' / 'D-reuse.jsonl').unlink())
            mutate(lambda t: (t.sd / 'round5' / 'A-fresh.jsonl').write_text(''))
            mutate(lambda t: (t.sd / 'round3' / 'A-reuse.jsonl').write_text(
                (t.sd / 'round3' / 'A-reuse.jsonl').read_text() * 2))
            mutate(lambda t: (t.sd / 'round3' / 'A-reuse.jsonl').write_text(
                (t.sd / 'round3' / 'A-reuse.jsonl').read_text().replace('"size": 64', '"size": 128')))
            mutate(lambda t: (t.sd / 'round3' / 'A-reuse.jsonl').write_text(
                (t.sd / 'round3' / 'A-reuse.jsonl').read_text().replace('"mode": "reuse"', '"mode": "fresh"')))
            mutate(lambda t: (t.sd / 'round17').mkdir())
            mutate(lambda t: __import__('shutil').rmtree(t.sd / 'round16'))

        # 起動記録: 渡した環境変数・起動順・rc の食い違いは exit 2
        def test_launch_log(self):
            def mutate(old, new):
                t = self.mk(lambda rnd, arm: 1e-4)
                p = t.sd / 'round2' / 'launch.tsv'
                txt = p.read_text()
                self.assertIn(old, txt)
                p.write_text(txt.replace(old, new, 1))
                md, code, _ = t.run()
                self.assertEqual(code, 2, md)
            mutate(f'C\treuse\t{GRAPH_ENV}=0', 'C\treuse\tnone')
            mutate(f'D\tfresh\t{GRAPH_ENV}=1', f'D\tfresh\t{GRAPH_ENV}=0')
            mutate('\tA\treuse\tnone\tbefore\t0\t', '\tA\treuse\tnone\tbefore\t1\t')
            mutate('\tB\treuse\tnone\tafter\t', '\tB\treuse\tnone\tbefore\t')
            mutate('2\t1\tB\treuse', '2\t1\tC\treuse')

        # RULE・道具・env_info・ゲートの不成立は exit 2
        def test_meta(self):
            def mutate(fn, needle):
                t = self.mk(lambda rnd, arm: 1e-4)
                fn(t)
                md, code, _ = t.run()
                self.assertEqual(code, 2, md)
                self.assertIn(needle, md)
            mutate(lambda t: (t.root / TOOL_FILES[2]).write_text('changed\n'), '道具の sha256 が RULE と不一致')
            mutate(lambda t: t.rule.write_text(t.rule.read_text() + '<要記入>\n'), '<要記入')
            mutate(lambda t: t.rule.write_text(t.rule.read_text().replace('A B D C', 'A B C D')), '[ラテン方格]')
            mutate(lambda t: t.env.write_text(t.env.read_text().replace('dry_run=0', 'dry_run=1')), 'dry_run')
            mutate(lambda t: t.env.write_text(t.env.read_text().replace('fandhe_ai_rust_ai_env_vars=0', 'fandhe_ai_rust_ai_env_vars=1')),
                   'fandhe_ai_rust_ai_env_vars')
            mutate(lambda t: t.env.write_text(t.env.read_text().replace('bin_sha256_B=' + 'c' * 64, 'bin_sha256_B=' + 'a' * 64)),
                   'A と B のバイナリが同一')
            mutate(lambda t: t.gate.write_text(t.gate.read_text().replace('round16 gate=pass', 'round16 GATE-TIMEOUT')), 'GATE-TIMEOUT')
            mutate(lambda t: t.gate.write_text(t.gate.read_text().replace('round16 gate=pass threshold=1.0 waited_s=0 load1=0.5\n', '')),
                   'round16 の gate=pass')
            mutate(lambda t: t.compute.write_text(t.compute.read_text() + '2026-01-01T00:01:00Z rc=0\n123, python\n'), '専有未達')
            mutate(lambda t: t.compute.write_text(t.compute.read_text().replace('rc=0', 'rc=9', 1)), 'rc != 0')

        # 実行中の判定器の照合（CLI 経路は常に照合する）
        def test_running_judge_pinned(self):
            t = self.mk(lambda rnd, arm: 1e-4)
            issues = check_tool_sha256(rule_block(t.rule.read_text(), 'tool-sha256'), t.root, judge_file=__file__)
            self.assertIn('実行中の判定器が RULE の sha256 と不一致', issues)
            issues = check_tool_sha256(rule_block(t.rule.read_text(), 'tool-sha256'), t.root,
                                       judge_file=t.root / JUDGE_REL)
            self.assertEqual(issues, [])

    res = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(T))
    if not res.wasSuccessful():
        print('self-test 失敗')
        return 1
    print(f'self-test ok（{res.testsRun} 件）')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--series-dir')
    p.add_argument('--env-info')
    p.add_argument('--rule')
    p.add_argument('--gate-log', default='')
    p.add_argument('--compute-log', default='')
    p.add_argument('--ratios-out', default='', help='ラウンドごとの比（reuse）の TSV の出力先')
    args = p.parse_args(argv)
    if args.self_test:
        return self_test()
    for need in ('series_dir', 'env_info', 'rule'):
        if not getattr(args, need):
            p.error(f'--{need.replace("_", "-")} が必要')
    # CLI 経路では道具の実ファイルを本ファイルの 3 つ上で探し、実行中の判定器自身も RULE の sha256 と照合する。
    text, code, tsv = build(args, root=None, judge_file=__file__)
    sys.stdout.write(text)
    if args.ratios_out and tsv:
        Path(args.ratios_out).write_text('round\torder\t' + '\t'.join(n for n, _, _ in RATIOS) + '\n' + '\n'.join(tsv) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
