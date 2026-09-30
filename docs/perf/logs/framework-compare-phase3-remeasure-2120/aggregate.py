#!/usr/bin/env python3
"""イシュー #2120: framework-compare 両機体再計測（3 腕 × 5 run）を RULE.txt の規則で集計する。

役割: orchestrate.sh が LOGD へ書いた run{1..5}/{A,B,C,others}.jsonl・gate.log・skipped.log・env_info.txt・
  switches-{B,C}.txt を読み、
  - 前提ゲート（check_prerequisites。RULE.txt「前提」）: 専有ゲート・計測条件・入力の厳格性・完備性を 1 関数で評価し、
    1 つでも不成立なら判定不能として aggregate.md・派生 JSONL・集計 JSON を作らない（既存の派生出力も消す）
  - セル別の腕ごと 5 run 採用行（判定 2）・同一 run 内比 B→C（主）／A→C（参考）の 5 run 中央値と q1／q3・
    非後退判定（<= 1.00。判定 3）
  - checksum の run 間・腕間の文字列完全一致（判定 4）
  を aggregate.md と集計 JSON（<out-prefix>-aggregate.json。機械可読の単一情報源）へ書き、
  scoreboard/gen_2120.py の入力となる派生 JSONL（腕ごと・機体別）を出力する。
  gen_2120.py は「前比」表示・22 セル転記の B→C 比を本 JSON の bc_med から取り、自前で再計算しない（判定 3 追記）。
  派生値の有限性（PR #2466 レビュー）: 同一 run 内比・中央値・q1／q3・採用行とスコアボードの表示用導出値を
  derived_value_errors で、どの出力を書くよりも前に一括検証する（不成立は前提不成立と同じく何も出力しない）。
  出力はメモリ上で生成・検証した後に commit_outputs で一時ファイル → os.replace によりまとめて確定し、既存の派生出力は
  冒頭で消す（途中の例外でも部分出力・前回の出力を残さない）。
python3 標準ライブラリのみ。#1988 aggregate.py の fail-closed 教訓（parity_fail_count の厳格読み・
5 run 完備の強制・run 欠損の空欄保持）を踏襲する。

使い方:
  python3 aggregate.py --machine gb10 --logs <dir> [--md aggregate.md] [--out-prefix results-gb10]
     -> <out-prefix>-{A,B,C}-full.jsonl（腕の fandhe-ai 採用行 + candle／burn 採用行）と
        <out-prefix>-aggregate.json（前提・ゲート・セル別の比・派生 JSONL の sha256）を出力
  python3 aggregate.py --self-test   （RULE.txt の条項ごとの表駆動テスト）
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile

ARMS = ('A', 'B', 'C')
NRUN = 5
PHASE_TASKS = ('train_phases', 'infer_phases', 'gemm_phases')
SCHEMA = 'fandhe-ai-2120-aggregate/1'
OTHER_FWS = ('candle', 'burn')
# 腕 B の rev（RULE.txt「計測腕」。orchestrate.sh の PRE_TREE .rev-stamp 検査と同値）
PRE_TREE_PREFIX = '65035979'
# 腕 B の完全な commit（PR #2466 レビュー。orchestrate.sh が PRE_TREE の git rev-parse HEAD と完全一致を照合し
# env_info.txt の rev_B_head に記録する値。前提 P1 で再確認する）
PRE_TREE_FULL = '650359799aa5e8e6d493bf3a38d6e59732062863'
HEX40_RE = re.compile(r'^[0-9a-f]{40}$')
# 専有ゲート（RULE.txt「計測」。orchestrate.sh gate() と同値。値の変更は規則の変更でありレビュー必須）。
#   precondition=True（GB10）: 5 run すべての pass が前提。fail は判定不能。
#   precondition=False（M4 Max・record_only）: fail も計測・採用し「不通過」を併記する。
GATE_SPEC = {
    'gb10': dict(precondition=True, load1_lt=1.0, gpu_util_eq=0, max_attempts=20),
    'm4max': dict(precondition=False, load1_lt=8.0, gpu_util_eq=None, max_attempts=30),
}
GATE_RE = re.compile(r'^run=(\d+) attempt=(\d+) load1=(\S+) gpu_util=(\S+) (pass|wait|fail|smoke-skip)$')
SKIP_RE = re.compile(r'^(bench-fandhe-([ABC])|bench-candle|bench-burn) (\S+) (\S+) (\d+) (fresh|reuse)( --phases)?: FAIL$')
REQUIRED_KEYS = ('framework', 'task', 'device', 'size', 'mode', 'median_s')


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
            for fw in OTHER_FWS:
                exp.add((fw, t, d, n, m))
    return exp - others_exceptions(machine)


def others_exceptions(machine):
    """明示例外: M4 Max の burn Metal GEMM N>=512 は結果テンソル全ゼロ（upstream 既知バグ）で記録拒否される
    （gen_2120.py の M4_SKIP と同一。gen は集計 JSON の others_exceptions と突合する）。
    ここへ足す変更は RULE.txt 判定 1 の緩和にあたりレビュー必須。"""
    if machine == 'm4max':
        return {('burn', 'gemm', 'metal', n, 'fresh') for n in (512, 1024, 2048, 4096)}
    return set()


def key(r):
    return (r['framework'], r['task'], r['device'], r['size'], r['mode'])


# --- 厳格な JSON 読み取り（NaN／Infinity・1e999 等の非有限値・オブジェクト内の重複キーを拒否）---
def _reject_constant(name):
    raise ValueError(f'非有限値 {name} は受理しない')


def _finite_float(s):
    v = float(s)
    if not math.isfinite(v):
        raise ValueError(f'非有限の数値 {s!r} は受理しない')
    return v


def _no_dup_keys(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError(f'JSON オブジェクト内でキー {k!r} が重複（後勝ちにしない）')
        d[k] = v
    return d


def parse_json_line(line):
    """1 行の JSON を厳格に読む。json.loads 既定は NaN／Infinity と重複キー（後勝ち）を黙って受理するため使わない。"""
    return json.loads(line, parse_constant=_reject_constant, parse_float=_finite_float, object_pairs_hook=_no_dup_keys)


def _positive_finite(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f'{name} が数値ではない: {v!r}')
    try:
        f = float(v)
    except OverflowError:
        raise ValueError(f'{name} が float に収まらない巨大整数') from None
    if not (math.isfinite(f) and f > 0):
        raise ValueError(f'{name} が正の有限値ではない: {v!r}')
    return f


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


def check_row(r):
    """計測行の型検査（必須キー・文字列キー・size は int〈bool・文字列・小数を拒否〉・median_s は正の有限値・parity）。"""
    if not isinstance(r, dict):
        raise ValueError(f'行が JSON オブジェクトではない: {r!r}')
    miss = [k for k in REQUIRED_KEYS if k not in r]
    if miss:
        raise ValueError(f'必須キー {miss} が欠損')
    for k in ('framework', 'task', 'device', 'mode'):
        if not isinstance(r[k], str):
            raise ValueError(f'{k} が文字列ではない: {r[k]!r}')
    if isinstance(r['size'], bool) or not isinstance(r['size'], int):
        raise ValueError(f'size が整数ではない: {r["size"]!r}')
    _positive_finite(r['median_s'], 'median_s')
    fail_count(r)


def load_rows(path):
    """JSONL を厳格に読み [(row, line)] を返す（--phases 行を含む。gen_2120.py の Python FW 行の検査にも使う）。"""
    if not os.path.isfile(path):
        raise ValueError(f'{path}: ファイルが無い')
    out = []
    with open(path) as f:
        for no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = parse_json_line(line)
                if isinstance(r, dict) and (r.get('task') in PHASE_TASKS or 'phase' in r):
                    out.append((r, line))  # --phases 行は判定対象外（型検査もしない）
                    continue
                check_row(r)
            except ValueError as e:
                raise ValueError(f'{path}:{no}: {e}') from None
            out.append((r, line))
    return out


def load_jsonl(path):
    """{key: (row, line)}。--phases 行は判定対象外のため除外。
    同一セルキーの重複行は採用値が曖昧になり 5 run × 3 腕の完備性も検証できないため、
    後勝ちにせず ValueError で入力不正として拒否する（orchestrate.sh は run ごとに truncate するため正常系で重複は出ない）。"""
    idx = {}
    for r, line in load_rows(path):
        if r.get('task') in PHASE_TASKS or 'phase' in r:
            continue
        k = key(r)
        if k in idx:
            raise ValueError(f'{path}: セルキーが重複: {k}（採用値が曖昧なため拒否。原因を調査して再計測すること）')
        idx[k] = (r, line)
    return idx


def raw_checksum(line):
    """JSON 行から checksum の値を文字列のまま取り出す（浮動小数の再整形で不一致を見逃さない）。"""
    m = re.search(r'"checksum":\s*([^,}\s]+)', line)
    return m.group(1) if m else None


def checksum_of(row, line):
    """判定 4 の比較に使う checksum 文字列。欠損・null・非数値・解析値との不一致は None（validate が拒否する）。"""
    raw = raw_checksum(line)
    v = row.get('checksum')
    if raw is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        if float(raw) != float(v):
            return None
    except (ValueError, OverflowError):
        return None
    return raw


def pick_median(rows):
    """[(run, row, line)] から median_s の中央値 run を返す（median_rounds.py と同規則。偶数なら小さい側）。"""
    rs = sorted(rows, key=lambda t: t[1]['median_s'])
    return rs[len(rs) // 2] if len(rs) % 2 else rs[len(rs) // 2 - 1]


# --- 前提ゲートの各検査（reasons へ不成立理由を積む。notes は判定を止めない併記事項）---
def check_env_info(logs, machine, reasons):
    """env_info.txt（orchestrate.sh が全 run 完了後に最後に書く完了記録）で計測条件を確認する。"""
    p = os.path.join(logs, 'env_info.txt')
    if not os.path.isfile(p):
        reasons.append('env_info.txt が無い（orchestrate.sh が完走していない）')
        return {}
    kv = {}
    dup = set()
    for line in open(p):
        if '=' not in line:
            continue
        k, v = line.rstrip('\n').split('=', 1)
        if k in kv:
            dup.add(k)
        kv[k] = v
    need = {
        'machine': lambda v: v == machine,
        'runs': lambda v: v == str(NRUN),
        'smoke': lambda v: v == '0',
        'FANDHE_AI_*': lambda v: v.startswith('未設定'),
        'rev_B': lambda v: v.startswith(PRE_TREE_PREFIX),
        'rev_C': lambda v: bool(v.strip()),
        # 出典の実体照合（PR #2466 レビュー・RULE.txt P1 追記）: orchestrate.sh が計測前と全 run 後に各ツリーの
        # git rev-parse HEAD と .rev-stamp・作業ツリーのクリーンさを照合した結果。stamp（rev_B／rev_C）との整合もここで再確認する。
        'rev_B_head': lambda v: v == PRE_TREE_FULL and v.startswith(kv.get('rev_B') or '\0'),
        'rev_C_head': lambda v: bool(HEX40_RE.match(v)) and v.startswith(kv.get('rev_C') or '\0'),
        'tree_B_clean': lambda v: v == 'yes',
        'tree_C_clean': lambda v: v == 'yes',
    }
    for k, ok in need.items():
        if k in dup:
            reasons.append(f'env_info.txt の {k} が重複')
        elif k not in kv:
            reasons.append(f'env_info.txt に {k} が無い')
        elif not ok(kv[k]):
            reasons.append(f'env_info.txt の {k}={kv[k]!r} が計測条件を満たさない（machine={machine}・runs={NRUN}・smoke=0・'
                           f'FANDHE_AI_* 未設定・rev_B は {PRE_TREE_PREFIX} 始まり・rev_B_head は {PRE_TREE_FULL}・'
                           'rev_C_head は rev_C で始まる 40 桁・tree_B_clean／tree_C_clean は yes）')
    return kv


def check_switches(logs, reasons):
    """switches-{B,C}.txt（採否スナップショット）の存在。C は全項目が解決されていること（<absent> 不可）。"""
    for arm in ('B', 'C'):
        p = os.path.join(logs, f'switches-{arm}.txt')
        if not os.path.isfile(p) or not open(p).read().strip():
            reasons.append(f'switches-{arm}.txt が無い／空')
            continue
        if arm == 'C' and '<absent>' in open(p).read():
            reasons.append('switches-C.txt に <absent> がある（腕 C は全スイッチの既定値を解決できること）')


def check_gate(logs, machine, reasons):
    """gate.log を厳格に読み、run ごとの最終判定を返す（RULE.txt「専有ゲート」）。
    - 全行が orchestrate.sh の書式に一致・run は 1..5 の昇順ブロック・各 run に終端（pass／fail）が 1 つ・終端後の行なし
    - smoke-skip は正式計測ではないため不成立
    - pass は記録値で閾値を再検証（ログ改変・閾値ずれの検出）
    - GB10（前提条件）は 5 run すべて pass。M4 Max（record_only）は fail も採用し不通過として併記する"""
    spec = GATE_SPEC[machine]
    p = os.path.join(logs, 'gate.log')
    gate = {}
    if not os.path.isfile(p):
        reasons.append('gate.log が無い（専有ゲートの記録を確認できない）')
        return gate
    cur, attempts, done = 0, 0, set()
    for no, line in enumerate(open(p), 1):
        line = line.rstrip('\n')
        if not line.strip():
            continue
        m = GATE_RE.match(line)
        if not m:
            reasons.append(f'gate.log:{no}: 書式不正: {line!r}')
            continue
        r, att, l1, gu, st = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4), m.group(5)
        if st == 'smoke-skip':
            reasons.append(f'gate.log:{no}: SMOKE の記録（正式計測ではない）')
            continue
        if not 1 <= r <= NRUN:
            reasons.append(f'gate.log:{no}: run={r} は範囲外')
            continue
        if r in done:
            reasons.append(f'gate.log:{no}: run={r} の終端（pass／fail）後に行がある')
            continue
        if r != cur:
            if r < cur or (cur and cur not in done):
                reasons.append(f'gate.log:{no}: run の順序が不正（run={cur} が未終端のまま run={r}）')
            cur, attempts = r, 0
        attempts += 1 if st == 'wait' else 0
        if att > spec['max_attempts'] or attempts > spec['max_attempts']:
            reasons.append(f'gate.log:{no}: 試行回数が上限 {spec["max_attempts"]} を超える')
        # 値は数値か NA（orchestrate.sh が取得失敗・非数値を NA と記録し通過扱いにしない）。pass は数値必須で閾値を再検証する。
        if not (re.fullmatch(r'\d+(\.\d+)?', l1) or l1 == 'NA'):
            reasons.append(f'gate.log:{no}: load1={l1!r} が数値でも NA でもない')
            continue
        if spec['gpu_util_eq'] is None:
            if gu != '-':
                reasons.append(f'gate.log:{no}: gpu_util={gu!r}（{machine} は "-"）')
                continue
        elif not (re.fullmatch(r'\d+', gu) or gu == 'NA'):
            reasons.append(f'gate.log:{no}: gpu_util={gu!r} が整数でも NA でもない')
            continue
        if st == 'wait':
            continue
        done.add(r)
        if st == 'pass':
            ok = (l1 != 'NA' and float(l1) < spec['load1_lt']
                  and (spec['gpu_util_eq'] is None or (gu != 'NA' and int(gu) == spec['gpu_util_eq'])))
            if not ok:
                reasons.append(f'gate.log:{no}: run={r} の pass 記録が閾値を満たさない（load1={l1} gpu_util={gu}）')
        elif spec['precondition']:
            reasons.append(f'gate.log:{no}: run={r} は専有ゲート不通過（{machine} は前提条件。判定不能）')
        gate[r] = dict(run=r, status=st, load1=l1, gpu_util=gu)
    for r in range(1, NRUN + 1):
        if r not in done:
            reasons.append(f'gate.log: run={r} の終端記録（pass／fail）が無い')
    return gate


def check_skipped(logs, machine, reasons, notes):
    """各 run の skipped.log（ベンチの非 0 終了）を検査する。行が出力済みでも終了コードが非 0 なら採用しない。
    判定対象・比較対象セルの FAIL は不成立。--phases（判定対象外）と明示例外の FAIL は notes へ併記する。"""
    judged = {('fandhe-ai',) + c for c in judged_cells(machine)}
    exp, exc = others_expected(machine), others_exceptions(machine)
    for r in range(1, NRUN + 1):
        p = os.path.join(logs, f'run{r}', 'skipped.log')
        if not os.path.isfile(p):
            reasons.append(f'run{r}/skipped.log が無い（終了コードを確認できない）')
            continue
        for no, line in enumerate(open(p), 1):
            line = line.rstrip('\n')
            if not line.strip():
                continue
            m = SKIP_RE.match(line)
            if not m:
                reasons.append(f'run{r}/skipped.log:{no}: 書式不正: {line!r}')
                continue
            fw = 'fandhe-ai' if m.group(2) else m.group(1)[len('bench-'):]
            k = (fw, m.group(3), m.group(4), int(m.group(5)), m.group(6))
            phases = bool(m.group(7))
            if fw == 'fandhe-ai' and phases:
                notes.append(f'run{r}: --phases 行の実行失敗（判定対象外）: {line}')
            elif k in exc and not phases:
                notes.append(f'run{r}: 明示例外の記録拒否: {line}')
            elif (k in judged or k in exp) and not phases:
                reasons.append(f'run{r}/skipped.log:{no}: 判定・比較対象セルの実行失敗（終了コード非 0）: {line}')
            else:
                reasons.append(f'run{r}/skipped.log:{no}: 想定外セルの記録: {line}')


def validate(runs, machine):
    """読み込んだ計測行の検証（判定 1 完備性・判定 4 checksum 欠損・未知セル・別 FW 混入）。不成立理由のリストを返す。"""
    errs = []
    cells = judged_cells(machine)
    judged = {('fandhe-ai',) + c for c in cells}
    okset = others_expected(machine) | others_exceptions(machine)
    for a in ARMS:
        for r in range(1, NRUN + 1):
            for k, (row, _) in runs[r][a].items():
                if row['framework'] != 'fandhe-ai':
                    errs.append(f'run{r}/{a}.jsonl に別 FW の行が混入: {k}')
                elif k not in judged:
                    errs.append(f'run{r}/{a}.jsonl に想定外セル: {k}')
    for r in range(1, NRUN + 1):
        for k, (row, _) in runs[r]['O'].items():
            if row['framework'] not in OTHER_FWS:
                errs.append(f'run{r}/others.jsonl に想定外 FW: {k}')
            elif k not in okset:
                errs.append(f'run{r}/others.jsonl に想定外セル: {k}')
    for (t, d, n, m) in cells:
        k = ('fandhe-ai', t, d, n, m)
        for a in ARMS:
            # RULE.txt 判定 4: 判定対象行の checksum 欠損は一致扱いにせず入力不正として拒否する。
            nock = [r for r in range(1, NRUN + 1) if k in runs[r][a] and checksum_of(*runs[r][a][k]) is None]
            if nock:
                errs.append(f'腕 {a} の {t} {d} N={n} {m} の checksum が run {nock} で欠損／不正')
            miss = [r for r in range(1, NRUN + 1) if k not in runs[r][a]]
            if miss:
                errs.append(f'腕 {a} の {t} {d} N={n} {m} が run {miss} で欠損（5 run × 3 腕完備が必須）')
    okeys = {k for r in runs for k in runs[r]['O']} | others_expected(machine)
    for k in sorted(okeys):
        have = [r for r in range(1, NRUN + 1) if k in runs[r]['O']]
        if len(have) != NRUN:
            errs.append(f'others の {k} が {len(have)}/5 run のみ（部分欠損。原因を調査してから再計測すること）')
    return errs


def check_prerequisites(logs, machine):
    """RULE.txt「前提」の単一ゲート。専有ゲート・計測条件・終了コード・入力の厳格性・完備性をまとめて評価する。
    戻り値 dict(reasons, notes, runs, gate, env)。reasons が空のときに限り判定・派生出力を行う。"""
    reasons, notes = [], []
    env = check_env_info(logs, machine, reasons)
    check_switches(logs, reasons)
    gate = check_gate(logs, machine, reasons)
    check_skipped(logs, machine, reasons, notes)
    runs, load_errs = {}, []
    for r in range(1, NRUN + 1):
        d = os.path.join(logs, f'run{r}')
        runs[r] = {}
        for a, fn in (('A', 'A.jsonl'), ('B', 'B.jsonl'), ('C', 'C.jsonl'), ('O', 'others.jsonl')):
            try:
                runs[r][a] = load_jsonl(os.path.join(d, fn))
            except ValueError as e:
                load_errs.append(str(e))
                runs[r][a] = {}
    reasons += load_errs
    if not load_errs:
        reasons += validate(runs, machine)
    return dict(reasons=reasons, notes=notes, runs=runs, gate=gate, env=env)


def _quartiles(vals):
    q = statistics.quantiles(vals, n=4, method='inclusive')
    return q[0], q[2]


def _ratio_runs(runs, k, num, den):
    """同一 run 内比 num.median_s / den.median_s の 5 run 値。有限値同士でも極端な値の除算は inf（例 1e300 / 1e-300）や
    0.0（アンダーフロー）になりうるため、値はそのまま返し（例外は None）、正の有限値でない run を含むセルは cell_stats が
    中央値・四分位を計算せず、どの出力を書くよりも前の一括検証（derived_value_errors）で判定不能にする（PR #2466 レビュー）。"""
    out = []
    for r in range(1, NRUN + 1):
        try:
            out.append(runs[r][num][k][0]['median_s'] / runs[r][den][k][0]['median_s'])
        except (OverflowError, ZeroDivisionError):
            out.append(None)
    return out


def cell_stats(runs, machine):
    """セルごとの腕別 run 値・比・checksum 一致・採用行を返す。前提成立（check_prerequisites）が前提。
    比の run 値に正の有限値でないものがあるセルは中央値・四分位を計算せず None とする（derived_value_errors が拒否する）。"""
    out = []
    for (t, d, n, m) in judged_cells(machine):
        k = ('fandhe-ai', t, d, n, m)
        med = {a: [runs[r][a][k][0]['median_s'] for r in range(1, NRUN + 1)] for a in ARMS}
        bc = _ratio_runs(runs, k, 'C', 'B')
        ac = _ratio_runs(runs, k, 'C', 'A')
        cks = {a: [checksum_of(*runs[r][a][k]) for r in range(1, NRUN + 1)] for a in ARMS}
        picked = {}
        for a in ARMS:
            pr, prow, _ = pick_median([(r, runs[r][a][k][0], runs[r][a][k][1]) for r in range(1, NRUN + 1)])
            picked[a] = dict(run=pr, median_s=prow['median_s'])
        bc_ok, ac_ok = all(_pos_finite(v) for v in bc), all(_pos_finite(v) for v in ac)
        bc_med = statistics.median(bc) if bc_ok else None
        q1, q3 = _quartiles(bc) if bc_ok else (None, None)
        out.append(dict(task=t, device=d, size=n, mode=m, med=med, bc_runs=bc, ac_runs=ac,
                        bc_med=bc_med, bc_q1=q1, bc_q3=q3, ac_med=statistics.median(ac) if ac_ok else None,
                        nonreg=(bc_med is not None and bc_med <= 1.00), cks=cks,
                        ck_within=all(len(set(v)) == 1 for v in cks.values()),
                        ck_across=len({c for v in cks.values() for c in v}) == 1,
                        picked=picked))
    return out


def _pos_finite(v):
    return (not isinstance(v, bool)) and isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def _shown_values(row):
    """スコアボード（gen_2120.py の cell_value・metric）が採用行から導出して表示・比較する値 [(名前, 値)]。
    time: median_s・ms 表示（median_s×1e3）、infer はスループット 1/median_s と µs 表示、gemm は gflops（記録値。無ければ
    2N³/median_s/1e9。gen と同じ規則）。いずれも正の有限値でなければならない。"""
    ms = row['median_s']
    vals = [('median_s', ms)]
    try:
        vals.append(('median_s×1e3', ms * 1e3))
        if row.get('task') == 'infer':
            vals.append(('1/median_s', 1.0 / ms))
            vals.append(('median_s×1e6', ms * 1e6))
        if row.get('task') == 'gemm':
            vals.append(('gflops', row['gflops'] if row.get('gflops') else 2.0 * row['size'] ** 3 / ms / 1e9))
    except (OverflowError, ZeroDivisionError) as e:
        vals.append((f'導出値（{e}）', None))
    return vals


def derived_value_errors(stats, derived_rows):
    """計算で派生するすべての数値の一括検証（PR #2466 レビュー・RULE.txt 前提の追記「派生値の有限性」）。
    どの出力（aggregate.md・派生 JSONL・集計 JSON）を書くよりも前に呼ぶ。対象:
      - セルごと: 腕別 5 run の median_s（med）・同一 run 内比 B→C／A→C の各 run 値・B→C 中央値・q1・q3・A→C 中央値・
        採用行の median_s（picked）・aggregate.md の B／C 中央値
      - 派生 JSONL に書く採用行（fandhe-ai 3 腕・candle／burn）: スコアボードが導出する値（_shown_values）
    いずれも正の有限値であること。不成立理由のリストを返す（空なら成立）。"""
    errs = []
    for s in stats:
        cid = f'{s["task"]} {s["device"]} N={s["size"]} {s["mode"]}'
        vals = []
        for a in ARMS:
            vals += [(f'腕 {a} run{i} median_s', v) for i, v in enumerate(s['med'][a], 1)]
            vals.append((f'腕 {a} 採用行 median_s', s['picked'][a]['median_s']))
        vals += [(f'B→C 比 run{i}', v) for i, v in enumerate(s['bc_runs'], 1)]
        vals += [(f'A→C 比 run{i}', v) for i, v in enumerate(s['ac_runs'], 1)]
        vals += [('B→C 中央値', s['bc_med']), ('B→C q1', s['bc_q1']), ('B→C q3', s['bc_q3']), ('A→C 中央値', s['ac_med'])]
        vals += [('B 中央値（md）', statistics.median(s['med']['B'])), ('C 中央値（md）', statistics.median(s['med']['C']))]
        bad = [f'{name}={v!r}' for name, v in vals if not _pos_finite(v)]
        if bad:
            errs.append(f'派生値が正の有限値ではない（{cid}）: {"・".join(bad)}（極端な値の比は inf／0 になりうる。判定不能）')
    for a, rows in derived_rows.items():
        for row in rows:
            bad = [f'{name}={v!r}' for name, v in _shown_values(row) if not _pos_finite(v)]
            if bad:
                errs.append(f'派生 JSONL（腕 {a}）の {row["framework"]} {row["task"]} {row["device"]} N={row["size"]} {row["mode"]} の'
                            f'表示用導出値が正の有限値ではない: {"・".join(bad)}')
    return errs


def fmt(x, p=4):
    return '' if x is None else f'{x:.{p}f}'


def render(result):
    """集計 JSON（build_result）から aggregate.md を描く（md と JSON・スコアボードの値を同じ情報源にする）。"""
    machine = result['machine']
    L = [f'# #2120 集計（{machine}・RULE.txt 規則）', '']
    L.append('前提（check_prerequisites）: 成立。専有ゲート・計測条件・終了コード・入力の厳格性・完備性を満たす入力のみを集計した。')
    L.append('')
    L.append('## 専有ゲート')
    L.append('')
    g = result['gate']
    L.append(f'区分: {"前提条件（5 run すべて pass が必須）" if g["precondition"] else "record_only（不通過でも計測・採用し不通過を併記）"}')
    L.append('')
    L.append('| run | 判定 | load1 | gpu_util |')
    L.append('|---|---|---|---|')
    for x in g['runs']:
        st = '通過' if x['status'] == 'pass' else '**不通過**'
        L.append(f'| {x["run"]} | {st} | {x["load1"]} | {x["gpu_util"]} |')
    L.append('')
    L.append('## Phase 3 前後比 B→C（主）・A→C（参考）。比 = C.median_s / B.median_s（同一 run 内・<1 で C が高速。非後退 ⇔ 5 run 中央値 <= 1.00）')
    L.append('')
    L.append('| セル | B 中央値[s] | C 中央値[s] | run1 | run2 | run3 | run4 | run5 | B→C 中央値 | B→C q1 | B→C q3 | 判定 | A→C 中央値 | checksum（run 間／腕間） |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for s in result['cells']:
        runs5 = ' | '.join(fmt(v) for v in s['bc_runs'])
        v = '非後退' if s['nonreg'] else '**後退**'
        ck = f'{"一致" if s["ck_within"] else "**不一致**"}／{"一致" if s["ck_across"] else "**不一致**"}'
        L.append(f'| {s["task"]} {s["device"]} N={s["size"]} {s["mode"]} | {statistics.median(s["med"]["B"]):.6g} | '
                 f'{statistics.median(s["med"]["C"]):.6g} | {runs5} | '
                 f'{s["bc_med"]:.4f} | {fmt(s["bc_q1"])} | {fmt(s["bc_q3"])} | {v} | {s["ac_med"]:.4f} | {ck} |')
    L.append('')
    sm = result['summary']
    L.append(f'後退セル数（B→C 中央値 > 1.00）: {sm["n_regressed"]} / {sm["n_cells"]}')
    L.append(f'checksum 不一致セル数: {sm["n_ck_mismatch"]}（不一致は是正せず記録。RULE.txt 判定 4）')
    if result['notes']:
        L.append('')
        L.append('## 併記事項（判定を止めない記録）')
        L.append('')
        L += [f'- {n}' for n in result['notes']]
    return '\n'.join(L) + '\n'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 16), b''):
            h.update(b)
    return h.hexdigest()


def derived_jsonl(runs, machine):
    """腕ごとの派生 JSONL（fandhe-ai 採用行 + candle／burn 採用行。gen_2120.py の入力）をメモリ上で作る。
    {arm: (本文 bytes, [採用行 dict])} を返す。書き出しは run_aggregate が検証後にまとめて行う（出力の原子性）。"""
    out = {}
    for a in ARMS:
        lines, rows_used = [], []
        for (t, d, n, m) in judged_cells(machine):
            k = ('fandhe-ai', t, d, n, m)
            rows = [(r, runs[r][a][k][0], runs[r][a][k][1]) for r in range(1, NRUN + 1)]
            _, prow, pline = pick_median(rows)
            lines.append(pline)
            rows_used.append(prow)
        okeys = sorted({k for r in runs for k in runs[r]['O']})
        for k in okeys:
            rows = [(r, runs[r]['O'][k][0], runs[r]['O'][k][1]) for r in range(1, NRUN + 1) if k in runs[r]['O']]
            _, prow, pline = pick_median(rows)
            lines.append(pline)
            rows_used.append(prow)
        out[a] = (('\n'.join(lines) + '\n').encode('utf-8'), rows_used)
    return out


def build_result(machine, pre, stats, derived):
    """集計 JSON（機械可読の単一情報源）。gen_2120.py はこの bc_med・採用行・ゲート・派生 JSONL の sha256 を読む。
    derived は {arm: (出力パス, 本文 bytes)}（sha256 は書き出す予定の bytes から計算する。書き出しは検証後）。"""
    spec = GATE_SPEC[machine]
    cells = []
    for s in stats:
        c = {k: s[k] for k in ('task', 'device', 'size', 'mode', 'bc_runs', 'ac_runs', 'bc_med', 'bc_q1', 'bc_q3', 'ac_med',
                               'nonreg', 'ck_within', 'ck_across', 'picked', 'med')}
        cells.append(c)
    return dict(
        schema=SCHEMA, machine=machine, status='formal',
        gate=dict(precondition=spec['precondition'], load1_lt=spec['load1_lt'], gpu_util_eq=spec['gpu_util_eq'],
                  runs=[pre['gate'][r] for r in range(1, NRUN + 1)]),
        rev_B=pre['env'].get('rev_B', ''), rev_C=pre['env'].get('rev_C', ''),
        rev_B_head=pre['env'].get('rev_B_head', ''), rev_C_head=pre['env'].get('rev_C_head', ''),
        notes=pre['notes'],
        others_exceptions=sorted(list(k) for k in others_exceptions(machine)),
        cells=cells,
        summary=dict(n_cells=len(stats), n_regressed=sum(1 for s in stats if not s['nonreg']),
                     n_ck_mismatch=sum(1 for s in stats if not (s['ck_within'] and s['ck_across']))),
        derived={a: dict(file=os.path.basename(p), sha256=hashlib.sha256(body).hexdigest()) for a, (p, body) in derived.items()},
    )


def output_paths(md, out_prefix):
    ps = [md] if md else []
    if out_prefix:
        ps += [f'{out_prefix}-{a}-full.jsonl' for a in ARMS] + [f'{out_prefix}-aggregate.json']
    return ps


def _remove_outputs(paths):
    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def commit_outputs(files):
    """[(path, bytes)] を原子的にまとめて確定する（PR #2466 レビュー）。各ファイルを同じディレクトリの一時ファイルへ
    すべて書いてから os.replace で置き換える（順序は files の順。集計 JSON は最後）。途中で例外が出たら一時ファイルと
    今回の出力先をすべて消してから例外を送出する（部分出力を残さない。前回の出力は run_aggregate が冒頭で消している）。"""
    tmps = []
    try:
        for p, body in files:
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(p)), prefix=f'.{os.path.basename(p)}.tmp-')
            tmps.append(tmp)
            with os.fdopen(fd, 'wb') as w:
                w.write(body)
        for (p, _), tmp in zip(files, tmps):
            os.replace(tmp, p)
    except BaseException:
        _remove_outputs(tmps + [p for p, _ in files])
        raise


def run_aggregate(logs, machine, md, out_prefix, quiet=False):
    """集計本体。前提不成立・派生値の検証不成立なら 1 を返し何も出力しない。成立なら (0, result)。
    既存の派生出力は冒頭で消す（古い正式結果の取り違え防止。前提不成立・派生値不正・途中の例外のいずれでも残さない）。
    出力はすべてメモリ上で生成・検証してから commit_outputs で一括して確定する。"""
    paths = output_paths(md, out_prefix)
    _remove_outputs(paths)
    pre = check_prerequisites(logs, machine)
    reasons = list(pre['reasons'])
    if not reasons:
        stats = cell_stats(pre['runs'], machine)
        dj = derived_jsonl(pre['runs'], machine)
        # どの出力を書くよりも前に、計算で派生するすべての数値を一括検証する（判定・派生出力の前提。RULE.txt 前提の追記）
        reasons += derived_value_errors(stats, {a: rows for a, (_, rows) in dj.items()})
    if reasons:
        print('前提不成立（RULE.txt「前提」）: 判定不能とし、aggregate.md・派生 JSONL・集計 JSON を作らない:\n  '
              + '\n  '.join(reasons), file=sys.stderr)
        return 1, None
    derived = {a: (f'{out_prefix}-{a}-full.jsonl', dj[a][0]) for a in ARMS} if out_prefix else {}
    result = build_result(machine, pre, stats, derived)
    md_text = render(result)
    # allow_nan=False は明示検証の取りこぼしに対する最後の防波堤（ここで失敗してもまだ何も書いていない）
    json_text = json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False) + '\n'
    files = [(p, body) for (p, body) in derived.values()]
    if md:
        files.append((md, md_text.encode('utf-8')))
    if out_prefix:
        files.append((f'{out_prefix}-aggregate.json', json_text.encode('utf-8')))  # 完了記録（派生 JSONL の sha256）は最後
    commit_outputs(files)
    if not quiet:
        print(md_text)
        if out_prefix:
            for a in ARMS:
                print('wrote', derived[a][0])
            print('wrote', f'{out_prefix}-aggregate.json')
    return 0, result


# --- self-test（RULE.txt の条項ごとの表駆動テスト。fixture は gen_2120.py の self-test からも使う）---
def make_fixture(td, machine, ratio_c=1.0, c_factors=None):
    """前提をすべて満たす正常系の LOGD を td に作る。各ケースはここから 1 条項だけを壊す。
    c_factors（run 1..5 の腕 C 倍率）を与えると同一 run 内比の中央値と採用行同士の比が食い違う入力を作れる
    （gen_2120.py の self-test が「前比は bc_med であり採用行の比ではない」ことを確かめるのに使う）。"""
    spec = GATE_SPEC[machine]
    for r in range(1, NRUN + 1):
        rd = os.path.join(td, f'run{r}')
        os.makedirs(rd, exist_ok=True)
        for a in ARMS:
            with open(os.path.join(rd, f'{a}.jsonl'), 'w') as w:
                for (t, d, n, m) in judged_cells(machine):
                    cf = (c_factors[r - 1] if c_factors else ratio_c) if a == 'C' else 1.0
                    ms = 0.001 * r * cf * (1.0 + n / 4096.0)
                    row = {'framework': 'fandhe-ai', 'version': '0.9.0', 'task': t, 'device': d, 'size': n, 'median_s': ms}
                    if t == 'gemm':
                        row['gflops'] = 2.0 * n ** 3 / ms / 1e9
                    row['checksum'] = 1.5
                    row['mode'] = m
                    if t == 'gemm':
                        row['parity_total'] = n * n
                        row['parity_fail_count'] = 0
                    w.write(json.dumps(row).replace('"checksum": 1.5', '"checksum": 1.500000') + '\n')
                # --phases 行（判定対象外）
                w.write(json.dumps({'framework': 'fandhe-ai', 'task': 'train_phases', 'device': 'cpu', 'size': 64, 'mode': 'fresh'}) + '\n')
        with open(os.path.join(rd, 'others.jsonl'), 'w') as w:
            for (fw, t, d, n, m) in sorted(others_expected(machine)):
                ms = 0.002 * r * (1.0 + n / 4096.0)
                row = {'framework': fw, 'version': 'x', 'task': t, 'device': d, 'size': n, 'median_s': ms, 'checksum': 1.5, 'mode': m}
                if t == 'gemm':
                    row['parity_total'] = n * n
                    row['parity_fail_count'] = 0
                w.write(json.dumps(row) + '\n')
        open(os.path.join(rd, 'skipped.log'), 'w').close()
        open(os.path.join(rd, 'err.log'), 'w').close()
    with open(os.path.join(td, 'gate.log'), 'w') as w:
        for r in range(1, NRUN + 1):
            gu = '0' if spec['gpu_util_eq'] is not None else '-'
            if r == 2:
                w.write(f'run={r} attempt=1 load1={spec["load1_lt"] + 0.5:.2f} gpu_util={gu} wait\n')
                w.write(f'run={r} attempt=2 load1={spec["load1_lt"] - 0.5:.2f} gpu_util={gu} pass\n')
            else:
                w.write(f'run={r} attempt=1 load1={spec["load1_lt"] - 0.5:.2f} gpu_util={gu} pass\n')
    with open(os.path.join(td, 'env_info.txt'), 'w') as w:
        w.write(f'date_utc=2026-09-30T00:00:00Z\nmachine={machine}\nrev_A=registry fandhe-ai =0.9.0\n'
                f'rev_B={PRE_TREE_PREFIX}\nrev_C=0a25a9c0\nruns={NRUN}\nsmoke=0\nFANDHE_AI_*=未設定（起動時に検査）\n'
                f'rev_B_head={PRE_TREE_FULL}\nrev_C_head=0a25a9c021dcaa6dae86aef03850fc55f8f40d00\n'
                'tree_B_clean=yes\ntree_C_clean=yes\n')
    for arm in ('B', 'C'):
        with open(os.path.join(td, f'switches-{arm}.txt'), 'w') as w:
            w.write('SME_PRODUCTION_ENABLED=false crates/x.rs:1\n' if arm == 'C' else 'SME_PRODUCTION_ENABLED=<absent> -\n')


def _edit(path, fn):
    lines = open(path).read().splitlines()
    with open(path, 'w') as w:
        w.write(''.join(l + '\n' for l in fn(lines)))


def _sub_first(path, old, new):
    _edit(path, lambda ls: [ls[0].replace(old, new, 1)] + ls[1:])


def _set_env(td, k, v):
    _edit(os.path.join(td, 'env_info.txt'), lambda ls: [f'{k}={v}' if l.startswith(k + '=') else l for l in ls])


def _append(path, line):
    with open(path, 'a') as w:
        w.write(line + '\n')


def _gate_lines(td, fn):
    _edit(os.path.join(td, 'gate.log'), fn)


def _map_rows(path, pred, upd):
    """JSONL の各行のうち pred(row) が真の行へ upd(row) を適用して書き直す（値の極端化ケース用。checksum の表記は fixture と同じに保つ）。"""
    def f(ls):
        out = []
        for l in ls:
            r = json.loads(l)
            if pred(r):
                upd(r)
                l = json.dumps(r).replace('"checksum": 1.5', '"checksum": 1.500000')
            out.append(l)
        return out
    _edit(path, f)


def _cell(t, d, n, m):
    return lambda r: (r.get('task'), r.get('device'), r.get('size'), r.get('mode')) == (t, d, n, m)


def self_test_cases():
    """(条項 ID, 内容, 機体, ratio_c, 変更関数, 期待) の表。期待は 'reject'（前提不成立・派生出力なし）か検査関数。"""
    j = os.path.join
    A1 = lambda td: j(td, 'run2', 'A.jsonl')  # noqa: E731
    O1 = lambda td: j(td, 'run2', 'others.jsonl')  # noqa: E731

    def all_nonreg(res):
        assert all(c['nonreg'] and abs(c['bc_med'] - 1.0) < 1e-12 for c in res['cells'])
        assert all(c['ck_within'] and c['ck_across'] for c in res['cells'])

    def all_reg(res):
        assert all(not c['nonreg'] for c in res['cells']) and res['summary']['n_regressed'] == len(res['cells'])

    def improve(res):
        assert all(c['nonreg'] and c['bc_med'] < 1.0 for c in res['cells'])

    def ck_mismatch(res):
        assert all(not (c['ck_within'] and c['ck_across']) for c in res['cells'])

    def m4_gate_fail_recorded(res):
        assert [x['status'] for x in res['gate']['runs']] == ['pass', 'pass', 'fail', 'pass', 'pass'], res['gate']
        assert not res['gate']['precondition']

    def note_phases(res):
        assert any('--phases' in n for n in res['notes']), res['notes']

    def note_exc(res):
        assert any('明示例外' in n for n in res['notes']), res['notes']

    def judged_fail_line(td):
        _append(j(td, 'run3', 'skipped.log'), 'bench-fandhe-B gemm cuda 256 reuse: FAIL')

    def gate_fail_gb10(td):
        _gate_lines(td, lambda ls: [l.replace('run=2 attempt=2', 'run=2 attempt=20').replace(' pass', ' fail')
                                    if l.startswith('run=2 attempt=2') else l for l in ls])

    def gate_fail_m4(td):
        _gate_lines(td, lambda ls: [l.replace('load1=7.50', 'load1=9.10').replace(' pass', ' fail')
                                    if l.startswith('run=3 ') else l for l in ls])

    def dup_key(td):
        _sub_first(A1(td), '"median_s": ', '"median_s": 0.5, "median_s": ')

    # 派生値の有限性（PR #2466 レビュー）: 入力は各々正の有限値だが、派生値が inf／0 になる極端値
    def ratio_inf(td):  # run2 の gemm cuda N=256 fresh: C/B = 1e300 / 1e-300 = inf
        c = _cell('gemm', 'cuda', 256, 'fresh')
        _map_rows(j(td, 'run2', 'B.jsonl'), c, lambda r: r.update(median_s=1e-300, gflops=1.0))
        _map_rows(j(td, 'run2', 'C.jsonl'), c, lambda r: r.update(median_s=1e300, gflops=1.0))

    def ratio_zero(td):  # run2 の train cpu 64 reuse: C/B = 1e-300 / 1e300 = 0.0（アンダーフロー）
        c = _cell('train', 'cpu', 64, 'reuse')
        _map_rows(j(td, 'run2', 'B.jsonl'), c, lambda r: r.update(median_s=1e300))
        _map_rows(j(td, 'run2', 'C.jsonl'), c, lambda r: r.update(median_s=1e-300))

    def tput_inf(td):  # infer cuda 64 fresh の全 run・全腕が 1e-320（非正規数）: 比は 1 だが 1/median_s = inf
        c = _cell('infer', 'cuda', 64, 'fresh')
        for r in range(1, NRUN + 1):
            for a in ARMS:
                _map_rows(j(td, f'run{r}', f'{a}.jsonl'), c, lambda x: x.update(median_s=1e-320))

    def gflops_inf(td):  # gemm cpu 4096 reuse の全 run・全腕: gflops 記録なし・2N³/median_s が inf
        c = _cell('gemm', 'cpu', 4096, 'reuse')
        for r in range(1, NRUN + 1):
            for a in ARMS:
                _map_rows(j(td, f'run{r}', f'{a}.jsonl'), c, lambda x: (x.update(median_s=1e-300), x.pop('gflops')))

    def gflops_bad(td):  # candle の gemm cuda 512 fresh の gflops 記録値が負
        c = _cell('gemm', 'cuda', 512, 'fresh')
        for r in range(1, NRUN + 1):
            _map_rows(j(td, f'run{r}', 'others.jsonl'), lambda x: c(x) and x['framework'] == 'candle', lambda x: x.update(gflops=-1.0))

    return [
        # 判定 3（非後退 ⇔ B→C 5 run 中央値 <= 1.00。境界 1.00 は非後退）
        ('J3-境界', '比 1.00 は非後退', 'gb10', 1.0, None, all_nonreg),
        ('J3-後退', '比 1.0001 は後退', 'gb10', 1.0001, None, all_reg),
        ('J3-改善', '比 0.9 は非後退', 'gb10', 0.9, None, improve),
        # 判定 4（checksum。不一致は是正せず記録・欠損／非有限は入力不正）
        ('J4-不一致記録', 'C 腕 run2 だけ checksum が異なる → 採用し不一致を記録', 'gb10', 1.0,
         lambda td: _edit(j(td, 'run2', 'C.jsonl'), lambda ls: [l.replace('"checksum": 1.500000', '"checksum": 1.500001') for l in ls]),
         ck_mismatch),
        ('J4-欠損', 'checksum キー欠損', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"checksum": 1.500000, ', ''), 'reject'),
        ('J4-NaN', 'checksum が NaN', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"checksum": 1.500000', '"checksum": NaN'), 'reject'),
        ('J4-null', 'checksum が null', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"checksum": 1.500000', '"checksum": null'), 'reject'),
        # 判定 1（完備性。欠けがあれば派生出力を作らない）
        ('J1-セル欠損', 'run2 の A 腕 1 行を削除', 'gb10', 1.0, lambda td: _edit(A1(td), lambda ls: ls[1:]), 'reject'),
        ('J1-ファイル欠損', 'run3/B.jsonl を削除', 'gb10', 1.0, lambda td: os.remove(j(td, 'run3', 'B.jsonl')), 'reject'),
        ('J1-比較対象全欠損', 'candle の 1 セルを全 run から削除', 'gb10', 1.0,
         lambda td: [_edit(j(td, f'run{r}', 'others.jsonl'), lambda ls: [l for l in ls if not ('"candle"' in l and '"size": 256' in l and '"cuda"' in l)])
                     for r in range(1, NRUN + 1)], 'reject'),
        ('J1-比較対象部分欠損', 'burn の 1 セルを run4 だけ削除', 'gb10', 1.0,
         lambda td: _edit(j(td, 'run4', 'others.jsonl'), lambda ls: [l for l in ls if not ('"burn"' in l and '"size": 512' in l and '"cuda"' in l)]),
         'reject'),
        # 入力の厳格性（前提）
        ('IN-重複行-腕', 'A 腕の同一セルが 2 行', 'gb10', 1.0, lambda td: _append(A1(td), open(A1(td)).readline().strip()), 'reject'),
        ('IN-重複行-比較対象', 'others の同一セルが 2 行', 'gb10', 1.0, lambda td: _append(O1(td), open(O1(td)).readline().strip()), 'reject'),
        ('IN-重複キー', '1 行内で median_s が 2 回', 'gb10', 1.0, dup_key, 'reject'),
        ('IN-median NaN', 'median_s が NaN', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"median_s": ', '"median_s": NaN, "x": '), 'reject'),
        ('IN-median 1e999', 'median_s が 1e999（inf）', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"median_s": ', '"median_s": 1e999, "x": '), 'reject'),
        ('IN-median 0', 'median_s が 0（比の分母）', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"median_s": ', '"median_s": 0, "x": '), 'reject'),
        ('IN-median 巨大整数', 'median_s が 10**400', 'gb10', 1.0,
         lambda td: _sub_first(A1(td), '"median_s": ', '"median_s": ' + '1' + '0' * 400 + ', "x": '), 'reject'),
        ('IN-size 文字列', 'size が "256"', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"size": 256', '"size": "256"'), 'reject'),
        ('IN-size 巨大整数', 'size が 10**30（未知セル）', 'gb10', 1.0,
         lambda td: _append(A1(td), open(A1(td)).readline().strip().replace('"size": 256', '"size": ' + '1' + '0' * 30)), 'reject'),
        ('IN-未知セル', 'A 腕に gemm cuda N=8192', 'gb10', 1.0,
         lambda td: _append(A1(td), open(A1(td)).readline().strip().replace('"size": 256', '"size": 8192')), 'reject'),
        ('IN-未知 FW', 'others に tch 行', 'gb10', 1.0,
         lambda td: _append(O1(td), open(O1(td)).readline().strip().replace('"burn"', '"tch"').replace('"candle"', '"tch"')), 'reject'),
        ('IN-別 FW 混入', 'A 腕に candle 行', 'gb10', 1.0,
         lambda td: _edit(A1(td), lambda ls: [ls[0].replace('"fandhe-ai"', '"candle"')] + ls[1:]), 'reject'),
        ('IN-mode 欠損', 'mode キー欠損', 'gb10', 1.0, lambda td: _sub_first(A1(td), ', "mode": "fresh"', ''), 'reject'),
        ('IN-JSON 不正', '壊れた行', 'gb10', 1.0, lambda td: _append(A1(td), '{"framework": '), 'reject'),
        ('IN-parity null', 'parity_fail_count が null', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"parity_fail_count": 0', '"parity_fail_count": null'), 'reject'),
        ('IN-parity 負数', 'parity_fail_count が -1', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"parity_fail_count": 0', '"parity_fail_count": -1'), 'reject'),
        ('IN-parity 文字列', 'parity_fail_count が "x"', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"parity_fail_count": 0', '"parity_fail_count": "x"'), 'reject'),
        ('IN-parity bool', 'parity_fail_count が true', 'gb10', 1.0, lambda td: _sub_first(A1(td), '"parity_fail_count": 0', '"parity_fail_count": true'), 'reject'),
        ('IN-parity 欠損', 'gemm 行の parity_fail_count 欠損', 'gb10', 1.0, lambda td: _sub_first(A1(td), ', "parity_fail_count": 0', ''), 'reject'),
        # 専有ゲート（GB10 は前提条件・M4 Max は record_only）
        ('GATE-GB10 不通過', 'run2 が 20 回とも不通過（fail）', 'gb10', 1.0, gate_fail_gb10, 'reject'),
        ('GATE-GB10 pass 改変', 'pass 行の load1 が 1.20', 'gb10', 1.0,
         lambda td: _gate_lines(td, lambda ls: [l.replace('load1=0.50', 'load1=1.20') if l.startswith('run=4 ') else l for l in ls]), 'reject'),
        ('GATE-GB10 GPU 使用中', 'pass 行の gpu_util が 3', 'gb10', 1.0,
         lambda td: _gate_lines(td, lambda ls: [l.replace('gpu_util=0', 'gpu_util=3') if l.startswith('run=5 ') else l for l in ls]), 'reject'),
        ('GATE-run 記録欠損', 'run5 の行が無い', 'gb10', 1.0, lambda td: _gate_lines(td, lambda ls: [l for l in ls if not l.startswith('run=5 ')]), 'reject'),
        ('GATE-ファイル欠損', 'gate.log が無い', 'gb10', 1.0, lambda td: os.remove(j(td, 'gate.log')), 'reject'),
        ('GATE-書式不正', '未知の行', 'gb10', 1.0, lambda td: _gate_lines(td, lambda ls: ls + ['hello']), 'reject'),
        ('GATE-SMOKE', 'smoke-skip 行', 'gb10', 1.0,
         lambda td: _gate_lines(td, lambda ls: ['run=1 attempt=0 load1=- gpu_util=- smoke-skip'] + ls[1:]), 'reject'),
        ('GATE-終端後の行', 'run1 の pass 後に同 run の行', 'gb10', 1.0,
         lambda td: _gate_lines(td, lambda ls: ls[:1] + [ls[0]] + ls[1:]), 'reject'),
        ('GATE-NA の待機は許容', 'run1 の待機行が NA（取得失敗）で、その後 pass', 'gb10', 1.0,
         lambda td: _gate_lines(td, lambda ls: ['run=1 attempt=1 load1=NA gpu_util=NA wait',
                                                ls[0].replace('attempt=1', 'attempt=2')] + ls[1:]), all_nonreg),
        ('GATE-NA の pass', 'pass 行の gpu_util が NA', 'gb10', 1.0,
         lambda td: _gate_lines(td, lambda ls: [l.replace('gpu_util=0', 'gpu_util=NA') if l.startswith('run=3 ') else l for l in ls]), 'reject'),
        ('GATE-M4 不通過は記録', 'M4 Max run3 が fail（record_only）→ 採用し不通過を併記', 'm4max', 1.0, gate_fail_m4, m4_gate_fail_recorded),
        ('GATE-M4 pass 改変', 'M4 Max の pass 行の load1 が 9.00', 'm4max', 1.0,
         lambda td: _gate_lines(td, lambda ls: [l.replace('load1=7.50', 'load1=9.00') if l.startswith('run=1 ') else l for l in ls]), 'reject'),
        # 終了コード（skipped.log）
        ('EXIT-判定対象 FAIL', '行はあるが判定対象セルが非 0 終了', 'gb10', 1.0, judged_fail_line, 'reject'),
        ('EXIT-比較対象 FAIL', 'candle の期待セルが非 0 終了', 'gb10', 1.0,
         lambda td: _append(j(td, 'run1', 'skipped.log'), 'bench-candle gemm cuda 256 fresh: FAIL'), 'reject'),
        ('EXIT-phases FAIL は併記', '--phases 行の失敗は判定対象外として notes へ', 'gb10', 1.0,
         lambda td: _append(j(td, 'run1', 'skipped.log'), 'bench-fandhe-A train cpu 64 fresh --phases: FAIL'), note_phases),
        ('EXIT-明示例外は併記', 'M4 Max burn Metal N=512 の記録拒否', 'm4max', 1.0,
         lambda td: _append(j(td, 'run1', 'skipped.log'), 'bench-burn gemm metal 512 fresh: FAIL'), note_exc),
        ('EXIT-ファイル欠損', 'run2/skipped.log が無い', 'gb10', 1.0, lambda td: os.remove(j(td, 'run2', 'skipped.log')), 'reject'),
        ('EXIT-書式不正', 'skipped.log に未知の行', 'gb10', 1.0, lambda td: _append(j(td, 'run1', 'skipped.log'), 'oops'), 'reject'),
        # 計測条件（env_info.txt・switches）
        ('ENV-欠損', 'env_info.txt が無い（未完走）', 'gb10', 1.0, lambda td: os.remove(j(td, 'env_info.txt')), 'reject'),
        ('ENV-機体違い', 'machine=m4max のログを gb10 として集計', 'gb10', 1.0, lambda td: _set_env(td, 'machine', 'm4max'), 'reject'),
        ('ENV-SMOKE', 'smoke=1', 'gb10', 1.0, lambda td: _set_env(td, 'smoke', '1'), 'reject'),
        ('ENV-run 数', 'runs=1', 'gb10', 1.0, lambda td: _set_env(td, 'runs', '1'), 'reject'),
        ('ENV-FANDHE_AI', 'FANDHE_AI_* が設定済み', 'gb10', 1.0, lambda td: _set_env(td, 'FANDHE_AI_*', 'FANDHE_AI_X'), 'reject'),
        ('ENV-腕 B rev', 'rev_B が 65035979 始まりでない', 'gb10', 1.0, lambda td: _set_env(td, 'rev_B', 'deadbeef'), 'reject'),
        ('SW-C 欠損', 'switches-C.txt が無い', 'gb10', 1.0, lambda td: os.remove(j(td, 'switches-C.txt')), 'reject'),
        ('SW-C absent', 'switches-C.txt に <absent>', 'gb10', 1.0,
         lambda td: _append(j(td, 'switches-C.txt'), 'X=<absent> -'), 'reject'),
        # 出典の実体照合（PR #2466 レビュー・RULE.txt P1 追記。orchestrate.sh が照合した結果の再確認）
        ('ENV-腕 B HEAD 欠損', 'rev_B_head が無い', 'gb10', 1.0,
         lambda td: _edit(j(td, 'env_info.txt'), lambda ls: [l for l in ls if not l.startswith('rev_B_head=')]), 'reject'),
        ('ENV-腕 B HEAD 不一致', 'rev_B_head が 65035979 始まりだが完全な commit と異なる', 'gb10', 1.0,
         lambda td: _set_env(td, 'rev_B_head', PRE_TREE_PREFIX + '0' * 32), 'reject'),
        ('ENV-腕 C HEAD と stamp 不一致', 'rev_C_head が rev_C（stamp）で始まらない', 'gb10', 1.0,
         lambda td: _set_env(td, 'rev_C_head', 'f' * 40), 'reject'),
        ('ENV-腕 C HEAD 短縮', 'rev_C_head が 40 桁でない', 'gb10', 1.0, lambda td: _set_env(td, 'rev_C_head', '0a25a9c0'), 'reject'),
        ('ENV-腕 B dirty', 'tree_B_clean=no', 'gb10', 1.0, lambda td: _set_env(td, 'tree_B_clean', 'no'), 'reject'),
        ('ENV-腕 C clean 欠損', 'tree_C_clean が無い', 'gb10', 1.0,
         lambda td: _edit(j(td, 'env_info.txt'), lambda ls: [l for l in ls if not l.startswith('tree_C_clean=')]), 'reject'),
        # 派生値の有限性（PR #2466 レビュー。極端値の比で inf／0 → 判定不能・出力なし・既存出力も除去）
        ('DV-比 inf', '1e300 / 1e-300 の同一 run 内比が inf', 'gb10', 1.0, ratio_inf, 'reject'),
        ('DV-比 0', '1e-300 / 1e300 の同一 run 内比が 0（アンダーフロー）', 'gb10', 1.0, ratio_zero, 'reject'),
        ('DV-スループット inf', 'median_s 1e-320 の 1/median_s が inf', 'gb10', 1.0, tput_inf, 'reject'),
        ('DV-gflops inf', 'gflops 記録なし・2N³/median_s が inf', 'gb10', 1.0, gflops_inf, 'reject'),
        ('DV-gflops 負', '比較対象の gflops 記録値が負', 'gb10', 1.0, gflops_bad, 'reject'),
    ]


# reject ケースが「その条項」の理由で拒否されたことを確かめる理由文の断片（別条項の巻き添えで通るのを防ぐ）
REJECT_HINTS = {
    'J4-欠損': 'checksum が run', 'J4-NaN': '非有限値 NaN', 'J4-null': 'checksum が run',
    'J1-セル欠損': '5 run × 3 腕完備', 'J1-ファイル欠損': 'ファイルが無い', 'J1-比較対象全欠損': '0/5 run',
    'J1-比較対象部分欠損': '4/5 run', 'IN-重複行-腕': 'セルキーが重複', 'IN-重複行-比較対象': 'セルキーが重複',
    'IN-重複キー': "キー 'median_s' が重複", 'IN-median NaN': '非有限値 NaN', 'IN-median 1e999': '非有限の数値',
    'IN-median 0': '正の有限値ではない', 'IN-median 巨大整数': '巨大整数', 'IN-size 文字列': 'size が整数ではない',
    'IN-size 巨大整数': '想定外セル', 'IN-未知セル': '想定外セル', 'IN-未知 FW': '想定外 FW', 'IN-別 FW 混入': '別 FW の行が混入',
    'IN-mode 欠損': "必須キー ['mode']", 'IN-JSON 不正': 'Expecting value', 'IN-parity null': '非負整数ではない',
    'IN-parity 負数': '非負整数ではない', 'IN-parity 文字列': '非負整数ではない', 'IN-parity bool': '非負整数ではない',
    'IN-parity 欠損': 'parity_fail_count が欠損', 'GATE-GB10 不通過': '専有ゲート不通過',
    'GATE-GB10 pass 改変': '閾値を満たさない', 'GATE-GB10 GPU 使用中': '閾値を満たさない', 'GATE-run 記録欠損': 'run=5 の終端記録',
    'GATE-ファイル欠損': 'gate.log が無い', 'GATE-書式不正': '書式不正', 'GATE-SMOKE': 'SMOKE の記録',
    'GATE-終端後の行': '終端（pass／fail）後', 'GATE-NA の pass': '閾値を満たさない', 'GATE-M4 pass 改変': '閾値を満たさない',
    'EXIT-判定対象 FAIL': '終了コード非 0', 'EXIT-比較対象 FAIL': '終了コード非 0', 'EXIT-ファイル欠損': 'skipped.log が無い',
    'EXIT-書式不正': '書式不正', 'ENV-欠損': 'env_info.txt が無い', 'ENV-機体違い': "machine='m4max'", 'ENV-SMOKE': "smoke='1'",
    'ENV-run 数': "runs='1'", 'ENV-FANDHE_AI': 'FANDHE_AI_*=', 'ENV-腕 B rev': "rev_B='deadbeef'",
    'SW-C 欠損': 'switches-C.txt が無い', 'SW-C absent': '<absent>',
    'ENV-腕 B HEAD 欠損': 'rev_B_head が無い', 'ENV-腕 B HEAD 不一致': "rev_B_head='65035979",
    'ENV-腕 C HEAD と stamp 不一致': "rev_C_head='ffff", 'ENV-腕 C HEAD 短縮': "rev_C_head='0a25a9c0'",
    'ENV-腕 B dirty': "tree_B_clean='no'", 'ENV-腕 C clean 欠損': 'tree_C_clean が無い',
    'DV-比 inf': 'B→C 比 run2=inf', 'DV-比 0': 'B→C 比 run2=0.0', 'DV-スループット inf': '1/median_s=inf',
    'DV-gflops inf': 'gflops=inf', 'DV-gflops 負': 'gflops=-1.0',
}


def self_test():
    n = 0
    for cid, desc, machine, ratio, mut, expect in self_test_cases():
        with tempfile.TemporaryDirectory() as td:
            logs = os.path.join(td, 'logs')
            os.makedirs(logs)
            make_fixture(logs, machine, ratio)
            if mut:
                mut(logs)
            md, prefix = os.path.join(td, 'aggregate.md'), os.path.join(td, 'out')
            # 前回の正式出力が残っていても前提不成立なら消えることも確認する
            for p in output_paths(md, prefix):
                open(p, 'w').write('stale')
            import io
            import contextlib
            with contextlib.redirect_stderr(io.StringIO()) as err:
                rc, res = run_aggregate(logs, machine, md, prefix, quiet=True)
            if expect == 'reject':
                assert rc == 1 and res is None, f'{cid}: 前提不成立を受理した（{desc}）'
                assert not any(os.path.exists(p) for p in output_paths(md, prefix)), f'{cid}: 派生出力が残った'
                assert REJECT_HINTS[cid] in err.getvalue(), f'{cid}: 想定した条項の理由で拒否されていない: {err.getvalue()}'
            else:
                assert rc == 0, f'{cid}: 正常系を拒否した（{desc}）: {err.getvalue()}'
                expect(res)
                on_disk = json.load(open(f'{prefix}-aggregate.json'))
                assert on_disk['status'] == 'formal' and on_disk['schema'] == SCHEMA
                for a in ARMS:
                    assert on_disk['derived'][a]['sha256'] == sha256_file(f'{prefix}-{a}-full.jsonl'), f'{cid}: sha256 不一致'
                    nl = len(open(f'{prefix}-{a}-full.jsonl').read().splitlines())
                    assert nl == len(judged_cells(machine)) + len(others_expected(machine)), f'{cid}: 派生 JSONL 行数'
                md_text = open(md).read()
                assert 'B→C q1' in md_text and '前提（check_prerequisites）: 成立' in md_text
                if cid == 'GATE-M4 不通過は記録':
                    assert '| 3 | **不通過** | 9.10 | - |' in md_text, md_text
            n += 1
    # 判定 2（採用行 = median_s が中央値の run。偶数なら小さい側）
    rows = [(r, {'median_s': v}, '') for r, v in ((1, 5.0), (2, 1.0), (3, 3.0), (4, 2.0), (5, 4.0))]
    assert pick_median(rows)[0] == 3
    assert pick_median([(r, {'median_s': v}, '') for r, v in ((1, 4.0), (2, 1.0), (3, 3.0), (4, 2.0))])[0] == 4
    # 判定 2 と JSON の採用行の一致（fixture は run 番号に比例するので run3 が採用行）
    with tempfile.TemporaryDirectory() as td:
        make_fixture(td, 'gb10')
        rc, res = run_aggregate(td, 'gb10', '', os.path.join(td, 'o'), quiet=True)
        assert rc == 0 and all(c['picked'][a]['run'] == 3 for c in res['cells'] for a in ARMS)
        # 判定 3: q1／q3 は同一 run 内比の四分位（inclusive）で中央値と別に記録
        assert all(c['bc_q1'] <= c['bc_med'] <= c['bc_q3'] for c in res['cells'])
        # run 欠損の空欄保持（render は左詰めしない）
        res['cells'][0]['bc_runs'][1] = None
        row = [ln for ln in render(res).splitlines() if ln.startswith('| gemm cuda N=256 fresh |')][0]
        assert row.split('|')[5].strip() == '', row
    # 出力の原子性（PR #2466 レビュー）: 生成途中・確定途中の例外で部分出力も前回の出力も残らない
    def _no_residue(td, md, prefix, label):
        assert not any(os.path.exists(p) for p in output_paths(md, prefix)), f'{label}: 出力が残った'
        left = [f for f in os.listdir(td) if '.tmp-' in f]
        assert not left, f'{label}: 一時ファイルが残った: {left}'

    g = globals()  # run_aggregate が参照する render をこのモジュールの名前空間で差し替える
    for label, phase in (('ATOM-生成途中の例外', 'render'), ('ATOM-確定途中の例外（3 件目の置換で失敗）', 'replace')):
        with tempfile.TemporaryDirectory() as td:
            logs = os.path.join(td, 'logs')
            os.makedirs(logs)
            make_fixture(logs, 'gb10')
            md, prefix = os.path.join(td, 'aggregate.md'), os.path.join(td, 'out')
            for p in output_paths(md, prefix):
                open(p, 'w').write('stale')
            orig_render, orig_replace, calls = g['render'], os.replace, []

            def boom_render(result):
                raise RuntimeError('injected')

            def boom_replace(a, b):
                calls.append(b)
                if len(calls) == 3:
                    raise OSError('injected')
                return orig_replace(a, b)
            try:
                if phase == 'render':
                    g['render'] = boom_render
                else:
                    os.replace = boom_replace
                try:
                    run_aggregate(logs, 'gb10', md, prefix, quiet=True)
                    raise AssertionError(f'{label}: 例外が伝播しない')
                except (RuntimeError, OSError) as e:
                    assert 'injected' in str(e), e
            finally:
                g['render'], os.replace = orig_render, orig_replace
            if phase == 'replace':
                assert len(calls) == 3, calls  # 2 件は確定済みだった（それも消えることを確かめる）
            _no_residue(td, md, prefix, label)
            n += 1
    # 明示例外の範囲（判定 1 の例外は M4 Max burn Metal N>=512 のみ）
    assert ('burn', 'gemm', 'metal', 512, 'fresh') not in others_expected('m4max')
    assert ('burn', 'gemm', 'cuda', 512, 'fresh') in others_expected('gb10')
    print(f'self-test ok（表駆動 {n} ケース + 単体）')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--machine', choices=('gb10', 'm4max'))
    ap.add_argument('--logs')
    ap.add_argument('--md', default='aggregate.md')
    ap.add_argument('--out-prefix', help='派生 JSONL・集計 JSON の接頭辞（<prefix>-{A,B,C}-full.jsonl・<prefix>-aggregate.json）')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if not (a.machine and a.logs):
        ap.error('--machine と --logs が必要')
    rc, _ = run_aggregate(a.logs, a.machine, a.md, a.out_prefix)
    sys.exit(rc)


if __name__ == '__main__':
    main()
