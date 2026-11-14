#!/usr/bin/env python3
"""結線後（Phase 4 後）の main HEAD と v0.14.0 の前後比較の集計器（イシュー #4168）。標準ライブラリのみ。

役割: `run_ab_prepost_0150_4168.sh` が出力する 1 系列（run1〜run5 の results-{before,after,opp,py}.jsonl）を読み、
before 腕（registry の fandhe-ai =0.14.0）と after 腕（main HEAD の path patch）それぞれの「同一 run 内の最速相手比」と
区分（1 位／僅差／負け）を、`unreached_0140.tsv` の 14 セルとガード 13 セルについて並べる。ADOPT／REJECT の二値判定は行わない。
判定経路の関数（VERBATIM_FUNCS）は `docs/perf/logs/framework-compare-0.14.0-remeasure/scoreboard/gen_0140.py` の逐語コピー
（`--check-verbatim` が字句一致を確かめる）。`--regress-0140` は 0.14.0 の実データを単腕として流し、`gen_0140.out` の 27 行と
比・区分・run 別が一致することを確かめる（判定規則を変えていない証拠）。
規則の正本: `docs/perf/logs/prepost-0150-4168/RULE.txt`（RULE.template.txt から実体化し計測前に単独コミットする）。

fail-closed: 欠損・重複・run 不足・追加ラウンド・diag 行・版違い・RULE 不一致・ゲート不成立は undetermined。
NB-RANGE5・NB-SEP の定義は承認済み（#4142 issuecomment-6757497844・#4209）。本 issue では記録だけで、判定と分類には使わない。
終了コード: 0 = 全セル判定可能 / 2 = undetermined あり / 3 = 同じ腕の中の hard 不一致（3 が 2 より優先）/
4 = 判定側（fandhe）の parity_fail_count > 0 で停止。
"""
import argparse
import ast
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
from pathlib import Path

NRUNS = 5
EXPECT_VERSION = '0.14.0'
FW_ORDER = ['candle', 'burn', 'pytorch', 'tensorflow', 'scipy']
VERBATIM_FUNCS = ('load', 'key', 'index', 'metric', 'better', 'paired_ratio', 'judge')
HARD_FIELDS = ('checksum', 'parity_total', 'parity_fail_count', 'parity_max_abs_err',
               'parity_max_rel_err', 'parity_scaled_abs_bound', 'parity_scaled_abs_rescued')
PLACEHOLDER = '<要記入'
# RULE が固定する道具（リポジトリルート相対。順序も含めランナー run_ab_prepost_0150_4168.sh の TOOL_FILES と同一）。
TOOL_FILES = (
    'scripts/bench/framework-compare/run_ab_prepost_0150_4168.sh',
    'scripts/bench/framework-compare/judge_prepost_0150_4168.py',
    'docs/perf/logs/prepost-0150-4168/orchestrate.sh',
    'docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py',
    'scripts/bench/framework-compare/ab_foreign_gate.sh',
)
JUDGE_REL = 'scripts/bench/framework-compare/judge_prepost_0150_4168.py'
DEFAULT_ROOT = Path(__file__).resolve().parents[3]

# 0.14.0 スコアボードと同じ 27 判定行（Mac mini 13 + GB10 14）。重点 = unreached_0140.tsv の 14 セル。
UNREACHED = frozenset({'M-MTL-G1024', 'M-MTL-G2048', 'M-MTL-G4096', 'M-CPU-G256', 'M-CPU-G512', 'M-CPU-G1024', 'M-CPU-G2048',
                       'M-MTL-TRN', 'M-CPU-TRN', 'M-CPU-INF', 'G-CPU-G256', 'G-CUDA-INF', 'G-CPU-TRN', 'G-CPU-INF'})
DEVNAME = {'metal': 'Metal', 'cpu': 'CPU', 'cuda': 'CUDA'}
CELLS = {
    'macmini': ([('gemm', 'metal', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048)]
                + [('train', 'metal', 64), ('infer', 'metal', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]),
    'gb10': ([('gemm', 'cuda', n) for n in (256, 512, 1024, 2048, 4096)] + [('gemm', 'cpu', n) for n in (256, 512, 1024, 2048, 4096)]
             + [('train', 'cuda', 64), ('infer', 'cuda', 64), ('train', 'cpu', 64), ('infer', 'cpu', 64)]),
}
MACHINE = {'macmini': 'Mac mini（M4）', 'gb10': 'GB10'}


def cell_id(profile, task, device, size):
    """gen_0140.py の cell_id と同じ規則（M-MTL-G1024・G-CPU-INF など）。"""
    mach = 'M' if profile == 'macmini' else 'G'
    dev = {'metal': 'MTL', 'cpu': 'CPU', 'cuda': 'CUDA'}[device]
    sfx = {'gemm': f'G{size}', 'train': 'TRN', 'infer': 'INF'}[task]
    return f'{mach}-{dev}-{sfx}'


def cell_label(task, device, size):
    return f'gemm {DEVNAME[device]} N={size}' if task == 'gemm' else f'{task} {DEVNAME[device]}'


# ---- ここから VERBATIM_FUNCS（gen_0140.py の逐語コピー。変更しない。--check-verbatim で照合）
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


# ---- ここから本スクリプト固有

class Undetermined(Exception):
    """系列・セルを判定できない理由（握りつぶさず表に出す）。"""


class ParityStop(Exception):
    """判定側（fandhe）の要素検証不合格。系列を止める（終了コード 4）。"""


def strict_load(path):
    """寛容パースをしない読み込み。欠損・JSON 不正・必須キー欠落・diag マーカー行は Undetermined。"""
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
        if not isinstance(r, dict):
            raise Undetermined(f'行がオブジェクトでない {p}:{i}')
        miss = [k for k in ('framework', 'task', 'device', 'size', 'mode') if k not in r]
        if miss:
            raise Undetermined(f'必須キー欠落（{",".join(miss)}） {p}:{i}')
        if any(not isinstance(r[k], str) for k in ('framework', 'task', 'device', 'mode')):
            raise Undetermined(f'framework/task/device/mode が文字列でない {p}:{i}')
        sz = r['size']
        if isinstance(sz, bool) or not isinstance(sz, (int, float)) or sz != sz or sz in (float('inf'), float('-inf')) or sz != int(sz):
            raise Undetermined(f'size が整数でない（{sz!r}） {p}:{i}')
        if any(k.startswith('diag') for k in r):
            raise Undetermined(f'diag マーカー付きの行（--diag-* は計測に使わない） {p}:{i}')
        if r['task'] not in ('train_phases', 'infer_phases'):
            ms = r.get('median_s')
            if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
                raise Undetermined(f'median_s が欠落または正の有限数でない（{ms!r}） {p}:{i}')
        # parity_fail_count: GEMM 行は必須、他の行も存在するなら検証する（欠落を「検証成功」扱いにしない）
        if r['task'] == 'gemm' or 'parity_fail_count' in r:
            pf = r.get('parity_fail_count')
            if isinstance(pf, bool) or not isinstance(pf, int) or pf < 0:
                raise Undetermined(f'parity_fail_count が欠落または非負整数でない（{pf!r}） {p}:{i}')
        rows.append(r)
    return rows


# 既知の欠測（RULE の想定内）: macmini の burn Metal gemm 512〜4096 は checksum 縮退で計測不能（ランナーの期待行数 22 と一致）。
KNOWN_MISSING = frozenset({('burn', 'gemm', 'metal', n) for n in (512, 1024, 2048, 4096)})


def required_opponents(profile, task, device, size):
    """ランナー（run_ab_prepost_0150_4168.sh の opp_cell・py_cell）が必ず出す比較相手 fresh 行の集合。"""
    need = {'candle', 'burn'}
    if profile == 'macmini':
        need |= {'pytorch', 'tensorflow'}
        if device == 'cpu':
            need.add('scipy')
    elif device == 'cpu':
        need |= {'pytorch', 'tensorflow', 'scipy'}
    else:
        need.add('pytorch')
    return {fw for fw in need if (fw, task, device, size) not in KNOWN_MISSING}


def missing_opponents(profile, runs, task, device, size):
    """必須の比較相手が欠けた run を列挙する（空なら揃っている）。欠損は undetermined（RULE の欠損時契約）。"""
    out = []
    for i, d in enumerate(runs, 1):
        for fw in sorted(required_opponents(profile, task, device, size)):
            if (fw, task, device, size, 'fresh') not in d:
                out.append(f'run{i}:{fw}')
    return out


def strict_index(rows, what):
    """verbatim の index に渡す前に重複を Undetermined にする（verbatim 側は assert のため）。"""
    seen = set()
    for r in rows:
        if r['task'] in ('train_phases', 'infer_phases'):
            continue
        if r.get('tf32') and r['framework'] != 'burn':
            continue
        k = key(r)
        if k in seen:
            raise Undetermined(f'重複行: {k} （{what}）')
        seen.add(k)
    return index(rows)


def check_run_set(series_dir, nruns=NRUNS):
    sd = Path(series_dir)
    if not sd.is_dir():
        raise Undetermined(f'系列ディレクトリ欠損: {sd}')
    expected = {f'run{i}' for i in range(1, nruns + 1)}
    found = {c.name for c in sd.iterdir() if re.fullmatch(r'run\d+', c.name)}
    if found != expected:
        raise Undetermined(f'run 集合が run1〜run{nruns} と一致しない（欠損 {sorted(expected - found)} 追加 {sorted(found - expected)}）')


def load_arm_runs(series_dir, arms=('before', 'after')):
    """{arm: [run ごとの索引(相手込み)]}、{arm: [run ごとの fandhe 行のみ]} を返す。相手の入力が読めなければ全体が判定不能。"""
    check_run_set(series_dir)
    per_arm = {a: [] for a in arms}
    own = {a: [] for a in arms}
    for i in range(1, NRUNS + 1):
        rd = Path(series_dir) / f'run{i}'
        opp = strict_load(rd / 'results-opp.jsonl') + strict_load(rd / 'results-py.jsonl')
        for a in arms:
            mine = strict_load(rd / f'results-{a}.jsonl')
            vs = {r.get('version') for r in mine if r['framework'] == 'fandhe-ai'}
            if vs != {EXPECT_VERSION}:
                raise Undetermined(f'run{i} {a}: fandhe-ai 行の version が {EXPECT_VERSION} でない: {sorted(map(str, vs))}')
            per_arm[a].append(strict_index(mine + opp, f'run{i} {a}'))
            own[a].append(strict_index(mine, f'run{i} {a} fandhe'))
    return per_arm, own


def median5(runs):
    """セルごとに median_s が中央の run の行を選ぶ（median_rounds.py と同じ。全 run に揃わないセルは含めない）。"""
    keys = set(runs[0])
    for d in runs[1:]:
        keys &= set(d)
    out = {}
    for k in keys:
        rows = sorted((d[k] for d in runs), key=lambda r: r['median_s'])
        out[k] = rows[len(rows) // 2]
    return out


def scan_parity_stop(own):
    for a, runs in own.items():
        for i, d in enumerate(runs, 1):
            for k, r in d.items():
                pf = r.get('parity_fail_count', 0)
                if isinstance(pf, (int, float)) and pf > 0:
                    raise ParityStop(f'{a} run{i}: fandhe-ai の parity_fail_count={pf} {k}')


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float('inf'), float('-inf'))


def hard_check(own_arm, task, device, size):
    """同じ腕の 5 run で checksum・parity 項目が完全一致するか（reuse・fresh）。(mismatch, missing, base) を返す。"""
    req = HARD_FIELDS if task == 'gemm' else ('checksum',)
    mismatch, missing, base_all = [], [], {}
    for mode in ('reuse', 'fresh'):
        base = {}
        for i, d in enumerate(own_arm, 1):
            row = d.get(('fandhe-ai', task, device, size, mode))
            if row is None:
                missing.append(f'{mode} run{i}: 行欠損')
                continue
            for f in req:
                if not is_num(row.get(f)):
                    missing.append(f'{mode} run{i}: 必須項目の欠落／不正値 {f}')
                elif f not in base:
                    base[f] = row[f]
                elif row[f] != base[f]:
                    mismatch.append(f'{mode} run{i}: 不一致 {f}')
        base_all[mode] = base
    return mismatch, missing, base_all


def arm_cell(data, runs, task, device, size):
    """verbatim の judge を呼ぶ。SystemExit（欠損・有効な相手の不在など）は Undetermined に変える。"""
    try:
        cls, verdict, ratio, best, rank, mm, rs = judge(data, task, device, size, runs)
    except SystemExit as e:
        raise Undetermined(str(e))
    ms = [r[('fandhe-ai', task, device, size, 'reuse')]['median_s'] for r in runs]
    return dict(cls=cls, verdict=verdict, ratio=ratio, best=best, rank=rank, mm=mm, runs=rs, ms=ms)


def fmt_runs(rs):
    return '[' + ' '.join(f'{x:.2f}' for x in rs) + ']'


def verdict_line(machine, label, a):
    """gen_0140.py の print_verdicts と同じ行形式（前版比の列なし）。"""
    mm_cls = 'win' if a['mm'] > 1 else ('near' if a['mm'] >= 0.90 else 'loss')
    cj = {'win': '1 位', 'near': '僅差', 'loss': '負け'}
    mm = f'  [中央値同士比 {a["mm"]:.2f}x で区分が {cj[mm_cls]}]' if mm_cls != a['cls'] else ''
    return f"{machine:7} {label:22} {a['verdict']:4} vs {a['best']:10} {a['ratio']:.2f}x  run別 [{' '.join(f'{x:.2f}' for x in a['runs'])}]{mm}"


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
    issues = []
    if any(not l for l in block):
        return ['RULE の [道具の sha256] ブロックに空行がある']
    want = [l.split('  ', 1) for l in block]
    if [w[1] if len(w) == 2 else '' for w in want] != list(TOOL_FILES):
        return [f'RULE の [道具の sha256] の対象が TOOL_FILES と一致しない（RULE={[w[-1] for w in want]}）']
    for sha, rel in want:
        f = Path(root) / rel
        if not f.is_file():
            issues.append(f'道具ファイルが無い: {rel}')
        elif sha256_file(f) != sha:
            issues.append(f'道具の sha256 が RULE と不一致: {rel}')
    if judge_file is not None and Path(judge_file).is_file():
        pinned = dict((rel, sha) for sha, rel in want).get(JUDGE_REL)
        if sha256_file(judge_file) != pinned:
            issues.append('実行中の判定器が RULE の sha256 と不一致')
    return issues


def check_series_meta(rule_path, env_info, gate_log, profile, compute_log='', root=None, judge_file=None):
    """RULE sha256・RULE の未記入・道具の sha256・env_info・ゲート（Mac mini は他ジョブ、GB10 は専有）の整合。成立しなければ理由リスト。

    root = 道具の実ファイルを探すリポジトリルート（既定は本ファイルの 3 つ上。テストは一時ツリーを渡す）。
    judge_file = 実行中の判定器の実体（既定は本ファイル。root を差し替えるテストでは None で省略）。
    """
    issues = []
    if root is None:
        root = DEFAULT_ROOT
        if judge_file is None:
            judge_file = __file__
    rule = Path(rule_path)
    if not rule.is_file():
        return [f'RULE 欠損: {rule}']
    rt = rule.read_text()
    if PLACEHOLDER in rt:
        issues.append(f'RULE に {PLACEHOLDER} が残っている')
    issues += check_tool_sha256(rule_block(rt, 'tool-sha256'), root, judge_file)
    consts_block = rule_block(rt, 'expected-consts')
    if not consts_block:
        issues.append('RULE に [期待するゲート定数] ブロック（# >>> expected-consts）が無い')
    elif any(not l for l in consts_block):
        issues.append('RULE の [期待するゲート定数] ブロックに空行がある')
    rule_sha = hashlib.sha256(rule.read_bytes()).hexdigest()
    env = Path(env_info)
    if not env.is_file():
        issues.append(f'env_info 欠損: {env}')
    else:
        lines = env.read_text().splitlines()
        rec = [l.split('=', 1)[1].strip() for l in lines if l.startswith('rule_sha256=')]
        if rec != [rule_sha]:
            issues.append(f'env_info の rule_sha256 が現 RULE と不一致（env={rec} rule={rule_sha}）')
        # 環境変数の混入検査（ランナーが FANDHE_AI_*／RUST_AI_* の存在を拒否し、0 件だったことと RAYON_NUM_THREADS の unset を記録する）
        for key, want in (('fandhe_ai_rust_ai_env_vars', '0'), ('rayon_num_threads', 'unset'), ('tool_sha256_check', 'ok'),
                          ('smoke', '0')):
            got = [l.split('=', 1)[1].strip() for l in lines if l.startswith(key + '=')]
            if got != [want]:
                issues.append(f'env_info の {key} が {want} でない（{got}）')
        if consts_block:
            want_sha = hashlib.sha256(('\n'.join(consts_block) + '\n').encode()).hexdigest()
            got = [l.split('=', 1)[1].strip() for l in lines if l.startswith('expected_consts_sha256=')]
            if got != [want_sha]:
                issues.append(f'env_info の expected_consts_sha256 が RULE の [期待するゲート定数] と不一致（env={got} rule={want_sha}）')
    g = Path(gate_log) if gate_log else None
    if g is None or not g.is_file():
        issues.append('gate.log 欠損（ゲートは必須）')
    else:
        text = g.read_text()
        for bad in ('GATE-INVALID', 'GATE-TIMEOUT', 'EXCLUSIVE-INVALID', 'EXCLUSIVE-UNMET'):
            if bad in text:
                issues.append(f'gate.log に {bad}')
        passed = {l.split()[0] for l in text.splitlines() if 'gate=pass' in l}
        for r in range(1, NRUNS + 1):
            if f'round{r}' not in passed:
                issues.append(f'gate.log に round{r} の gate=pass が無い')
        if profile == 'macmini':
            # 他ジョブゲート: 各ラウンドの前後が CLEAN（行頭一致。load1 の `gate=pass` 行とは別の語で記録される）
            have = set(text.splitlines())
            for r in range(1, NRUNS + 1):
                for tok in ('foreign-before=clean', 'foreign-after=clean'):
                    if f'round{r} {tok}' not in have:
                        issues.append(f'gate.log に round{r} の {tok} が無い（他ジョブゲート）')
    if profile == 'gb10':
        c = Path(compute_log) if compute_log else None
        if c is None or not c.is_file():
            issues.append('compute_apps.log 欠損（GB10 の専有確認は必須）')
        else:
            blocks = [b for b in c.read_text().split('\n') if b.strip()]
            heads = [b for b in blocks if re.search(r' rc=\d+$', b)]
            if len(heads) < NRUNS:
                issues.append(f'compute_apps.log の記録が {len(heads)} 件（期待 {NRUNS} 以上）')
            if any(not b.endswith('rc=0') for b in heads):
                issues.append('compute_apps.log に rc != 0 の記録')
            if len(blocks) != len(heads):
                issues.append('compute_apps.log に compute プロセスの行がある（専有未達）')
    return issues


def evaluate(profile, series_dir):
    """セルごとの結果リストと problem 情報を返す。"""
    per_arm, own = load_arm_runs(series_dir)
    scan_parity_stop(own)
    data = {a: median5(per_arm[a]) for a in per_arm}
    cells = []
    for task, device, size in CELLS[profile]:
        c = dict(id=cell_id(profile, task, device, size), label=cell_label(task, device, size), err=None)
        c['key'] = (task, device, size)
        for a in ('before', 'after'):
            mism, miss, base = hard_check(own[a], task, device, size)
            c['hard_' + a] = (mism, miss, base)
            lacking = missing_opponents(profile, per_arm[a], task, device, size)
            if lacking:
                c[a] = None
                c['err'] = f'{a}: 必須の比較相手が欠損 ' + ','.join(lacking)
                continue
            try:
                c[a] = arm_cell(data[a], per_arm[a], task, device, size)
            except Undetermined as e:
                c[a] = None
                c['err'] = f'{a}: {e}'
        cells.append(c)
    return cells


def arm_ratio(c):
    b, a = c['before'], c['after']
    if not b or not a:
        return None
    rs = [x / y for x, y in zip(a['ms'], b['ms'])]
    return statistics.median(a['ms']) / statistics.median(b['ms']), rs


def arm_checksum_note(c):
    """腕間の checksum 差（ライブラリのコードが違うため即失敗にはしない。差ありとして記録する）。"""
    d = []
    for mode in ('reuse', 'fresh'):
        bb = c['hard_before'][2].get(mode, {}).get('checksum')
        aa = c['hard_after'][2].get(mode, {}).get('checksum')
        if bb is not None and aa is not None and bb != aa:
            d.append(mode)
    return d


def build(args):
    """(Markdown, 終了コード, ratios TSV 行) を返す。"""
    out, undet, hard = [], [], []
    out.append(f'# 結線後の前後比較（#4168）: {args.profile} / {args.label}\n')
    out.append('before = registry の fandhe-ai =0.14.0、after = main HEAD（path patch）。判定式は 0.14.0 スコアボードと同一'
               '（同一 run 内の最速相手比の 5 run 中央値）。after の比は 0.15.0 の見込みであり、公開版には Phase 5 以降の変更が入りうる。'
               '勝敗の正は Phase 6（#4177〜#4179）の測り直し。腕比は参考で合格ラインはない。'
               'NB-RANGE5・NB-SEP の定義は承認済み（#4142 issuecomment-6757497844・#4209）。本 issue では帯を記録するだけで、判定と分類には使わない。\n')
    # CLI 経路では --tool-root の有無にかかわらず、実行中の判定器自身も RULE の sha256 と照合する（--tool-root で照合を外せないようにする）。
    meta = check_series_meta(args.rule, args.env_info, args.gate_log, args.profile, args.compute_log, args.tool_root, judge_file=__file__)
    undet += [f'系列: {m}' for m in meta]
    # --smoke（行数スモーク）の出力は計測の代わりにならない（ディレクトリ名と env_info の smoke=1 の両方で拒否）
    if Path(args.series_dir).name.startswith('ab4168-smoke-'):
        undet.append('系列: --smoke の出力ディレクトリは判定に使えない（行数スモークは計測の代わりにならない）')
    tsv, regress = [], []
    try:
        cells = evaluate(args.profile, args.series_dir)
    except ParityStop as e:
        out.append(f'## 停止\n\n- 判定側の要素検証不合格: {e}\n\nexit=4')
        return '\n'.join(out) + '\n', 4, []
    except Undetermined as e:
        undet.append(f'系列: {e}')
        cells = []
    out.append('| セル | 重点 | before 比／区分 | after 比／区分（run 別） | after の帯（記録のみ） | 腕比 after/before（run 別） | 最速相手 | hard | 腕間 checksum | 見込み |')
    out.append('|---|---|---|---|---|---|---|---|---|---|')
    for c in cells:
        mark = '重点' if c['id'] in UNREACHED else 'ガード'
        hs = []
        for a in ('before', 'after'):
            mism, miss, _ = c['hard_' + a]
            if mism:
                hard.append(f"{c['id']} {a}: " + '; '.join(mism))
            if miss:
                undet.append(f"{c['id']} {a}: hard 検証が判定不能: " + '; '.join(miss))
            hs.append('NG' if mism else ('判定不能' if miss else 'OK'))
        hard_s = '/'.join(hs)
        if c['err'] or not c['before'] or not c['after']:
            undet.append(f"{c['id']}: {c['err']}")
            out.append(f"| {c['id']} | {mark} | 判定不能: {c['err']} | | | | | {hard_s} | | |")
            continue
        b, a = c['before'], c['after']
        am, ars = arm_ratio(c)
        band = f'{min(a["runs"]):.2f}〜{max(a["runs"]):.2f}'
        cs = arm_checksum_note(c)
        cs_s = '差あり(' + ','.join(cs) + ')' if cs else '一致'
        top = b['best'] if b['best'] == a['best'] else f"{b['best']}→{a['best']}"
        note = ''
        if a['ratio'] > 1 and c['id'] in UNREACHED:
            note = '見込みで 1 位（#4168 after 腕）'
        out.append(f"| {c['id']} | {mark} | {b['ratio']:.2f} {b['verdict']} | {a['ratio']:.2f} {a['verdict']} {fmt_runs(a['runs'])} "
                   f"| {band} | {am:.3f} {fmt_runs(ars)} | {top} | {hard_s} | {cs_s} | {note} |")
        tsv.append(f"{c['id']}\t{b['ratio']:.2f}\t{a['ratio']:.2f}\t{a['verdict']}")
        if all(x > 1.0 for x in ars):
            regress.append(f"{c['id']}: 腕比が 5/5 run で 1 を超えた（after が遅い方向の一貫。所有者への報告事項。自律的な revert・再計測はしない）")
    out.append('')
    if regress:
        out.append('## 後退の一貫（記録のみ）\n')
        out.extend(f'- {r}' for r in regress)
        out.append('')
    if hard:
        out.append('## hard 条件の不一致（同じ腕の中。正しさの失敗として所有者へ報告。tolerance は変えない）\n')
        out.extend(f'- {h}' for h in hard)
        out.append('')
    if undet:
        out.append('## undetermined\n')
        out.extend(f'- {u}' for u in undet)
        out.append('')
    code = 3 if hard else (2 if undet else 0)
    out.append(f'exit={code}')
    return '\n'.join(out) + '\n', code, tsv


# ---- 0.14.0 の実データを単腕で流す回帰検査と、原稿 TSV の比の列の補完

def regress_0140(root):
    """GB10・Mac mini の 0.14.0 実データを verbatim 経路で流し、gen_0140.out の 27 行と一致するか確かめる。"""
    root = Path(root)
    gb_dir = root / 'docs/perf/logs/framework-compare-0.14.0-remeasure/gb10'
    mm_dir = root / 'docs/perf/logs/framework-compare-0.14.0-macmini-remeasure/macmini-series-b'
    want = (root / 'docs/perf/logs/framework-compare-0.14.0-remeasure/scoreboard/gen_0140.out').read_text().splitlines()
    got = []
    for profile, d, names in (('macmini', mm_dir, ('results-macmini-0.14.0.jsonl', 'results-macmini-py-0.14.0.jsonl')),
                              ('gb10', gb_dir, ('results-dgx-0.14.0.jsonl', 'results-dgx-0.14.0-extra.jsonl', 'results-dgx-py-0.14.0.jsonl'))):
        runs = [index(sum((load(d / f'run{i}' / n) for n in names), [])) for i in range(1, NRUNS + 1)]
        data = median5(runs)
        for task, device, size in CELLS[profile]:
            got.append(verdict_line(MACHINE[profile], cell_label(task, device, size), arm_cell(data, runs, task, device, size)))
    pat = re.compile(r'^(.*?run別 \[[^\]]*\])')
    want_rows = [pat.match(l).group(1) for l in want if pat.match(l)]
    got_rows = [pat.match(l).group(1) for l in got]
    # 並び（Mac mini 13 → GB10 14）は gen_0140.out と同じ。行ごとに突合する。
    if len(want_rows) != 27 or want_rows != got_rows:
        diff = [(w, g) for w, g in zip(want_rows, got_rows) if w != g]
        raise SystemExit(f'regress-0140 不一致: want={len(want_rows)} got={len(got_rows)} diff={diff[:3]}')
    print('regress-0140 ok: 27 行一致')


def fill_draft(draft, ratios, out):
    """原稿 TSV の比の列（3 列目）だけを after の比で埋める。分類・理由・出典には触れない。14 行すべてが揃わなければ止める。"""
    got = {}
    for rp in ratios:
        for l in Path(rp).read_text().splitlines():
            f = l.split('\t')
            if len(f) == 4:
                got[f[0]] = f[2]
    lines = Path(draft).read_text().splitlines()
    res, seen = [], set()
    for l in lines:
        if l.startswith('#') or not l.strip():
            res.append(l)
            continue
        f = l.split('\t')
        if len(f) != 6:
            raise SystemExit(f'原稿の列数が 6 でない: {l[:40]}')
        if f[0] not in got:
            raise SystemExit(f'ratios にセルが無い: {f[0]}')
        f[2] = got[f[0]]
        seen.add(f[0])
        res.append('\t'.join(f))
    if seen != set(UNREACHED):
        raise SystemExit(f'原稿のセル集合が 14 セルと一致しない: 不足 {sorted(UNREACHED - seen)}')
    Path(out).write_text('\n'.join(res) + '\n')
    print(f'fill-draft ok: {len(seen)} 行')


def check_verbatim(src):
    def funcs(text):
        tree = ast.parse(text)
        lines = text.splitlines()
        return {n.name: lines[n.lineno - 1:n.end_lineno] for n in tree.body if isinstance(n, ast.FunctionDef)}
    mine, theirs = funcs(Path(__file__).read_text(encoding='utf-8')), funcs(Path(src).read_text(encoding='utf-8'))
    bad = [f for f in VERBATIM_FUNCS if f not in theirs or f not in mine or mine[f] != theirs[f]]
    if bad:
        raise SystemExit(f'gen_0140.py と字句が違う関数: {bad}')
    print('verbatim ok: ' + ' '.join(VERBATIM_FUNCS))


def self_test():
    import unittest
    here = Path(__file__).with_name('judge_prepost_0150_4168_test.py')
    if not here.is_file():
        raise SystemExit('テストファイルが無い')
    suite = unittest.defaultTestLoader.discover(str(here.parent), pattern=here.name)
    res = unittest.TextTestRunner(verbosity=1).run(suite)
    if not res.wasSuccessful():
        raise SystemExit('self-test 失敗')
    print('self-test ok')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--check-verbatim', help='gen_0140.py のパス。VERBATIM_FUNCS の字句一致を確かめて終わる')
    p.add_argument('--regress-0140', help='リポジトリルート。0.14.0 の実データを単腕で流し gen_0140.out と照合して終わる')
    p.add_argument('--fill-draft', help='原稿 TSV。--ratios と --out が必要')
    p.add_argument('--ratios', nargs='*', default=[], help='--ratios-out の出力（機体ごと）')
    p.add_argument('--out', default='')
    p.add_argument('--profile', choices=sorted(CELLS))
    p.add_argument('--label', default='')
    p.add_argument('--series-dir')
    p.add_argument('--env-info')
    p.add_argument('--rule')
    p.add_argument('--gate-log', default='')
    p.add_argument('--compute-log', default='')
    p.add_argument('--tool-root', default=None, help='道具の実ファイルを探すリポジトリルート（既定は本ファイルの 3 つ上）')
    p.add_argument('--ratios-out', default='')
    args = p.parse_args(argv)
    if args.self_test:
        self_test(); return 0
    if args.check_verbatim:
        check_verbatim(args.check_verbatim); return 0
    if args.regress_0140:
        regress_0140(args.regress_0140); return 0
    if args.fill_draft:
        if not args.out or not args.ratios:
            p.error('--fill-draft には --ratios と --out が必要')
        fill_draft(args.fill_draft, args.ratios, args.out); return 0
    for need in ('profile', 'label', 'series_dir', 'env_info', 'rule'):
        if not getattr(args, need):
            p.error(f'--{need.replace("_", "-")} が必要')
    text, code, tsv = build(args)
    sys.stdout.write(text)
    if args.ratios_out and tsv:
        Path(args.ratios_out).write_text('\n'.join(tsv) + '\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
