#!/usr/bin/env python3
"""イシュー #3176: Mac mini の SME ON／OFF A/B の厳密集計（R1 到達セル・R2 checksum・R4 格子・長方形・R5 PyTorch 比）。

役割: orchestrate_macmini.sh が収録した `macmini/` 配下のログと、run_ab_sme_macmini_cpu.sh の
JSONL（`macmini/r1r2/`）を読み、RULE.txt の評価順 FAIL > REJECT > ADOPT 候補 > undetermined を
機械的に当てはめた結果と各表を Markdown で標準出力へ出す。本ツールは RULE の適用器であり、
規則そのものは RULE.txt が正。本番定数（SME_PRODUCTION_ENABLED）は切り替えない。

入力の厳格性: 重複行・非有限値・未知のキー・点数の過不足・m≠n の R4 行・欠損は捨てず、
前提ゲート不成立（undetermined）へ写像する。JSONL は読み取り時に各行がオブジェクト（重複キーなし）で、
識別フィールド task（str）・size（int。bool は不可）・mode（str。省略時 fresh）の型を満たすことを
検証し、null・配列・数値の行や size が配列の行は AttributeError／TypeError ではなく AggError にする
（PR #3501 codex P2）。python3 標準ライブラリのみ。
PyTorch の計測失敗は run_ab_sme_macmini_cpu.sh が A/B 本体の終了コードから切り離して専用ログ
`torch-failed-3176-*-cpu-<label>.log` に記録する（RULE §7「記録のみ」。PR #3501 codex P2）。本ツールは
このログを厳格に読み（欠損・形式不正・件数不一致は AggError）、結果も異常も注記にだけ出して判定には使わない。

使い方:
  python3 -I aggregate.py <macmini-dir> [<label>]   # 既定 label は 3176-r1
  python3 -I aggregate.py --self-test
"""
import glob
import json
import math
import os
import re
import statistics
import sys
import tempfile

ROUNDS = 5
GEMM_SIZES = (512, 1024, 2048)
MODES = ("fresh", "reuse")
CELLS = [("gemm", s, m) for s in GEMM_SIZES for m in MODES] + [
    (t, 64, m) for t in ("train", "infer") for m in MODES
]
CELLS_SET = set(CELLS)
# 到達セル（m=n=k>=512 の gemm 6 + train size=64 の 2。RULE.txt §2）。infer は m=64 で非到達
REACHED = [c for c in CELLS if c[0] == "gemm" or c[0] == "train"]
R4_MN = (64, 128, 256, 512)
R4_K = (32, 64, 128, 256)
RECT_SHAPES = ((64, 256, 784), (784, 256, 64), (784, 64, 256), (64, 10, 256))
# 現行しきい値（sme_shape_eligible: m,n>=256 かつ k>=64）。参照のみで本番値は変えない
CUR_MIN_MN, CUR_MIN_K = 256, 64
LINE = re.compile(
    r"^variant=(NEON\(TwoDDynamic\)|SME\(TwoDDynamicSme\)) size=\((\d+),(\d+),(\d+)\) median_gflops=(\S+)$"
)


class AggError(Exception):
    pass


def read_lines(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]
    except UnicodeDecodeError as e:
        raise AggError("%s: UTF-8 として読めない: %s" % (path, e))


def parse_variant_log(path):
    """1 run のログ -> {(m,n,k): {"NEON": g, "SME": g}}。重複・非有限は AggError。"""
    pts = {}
    for ln in read_lines(path):
        if ln.startswith("num_threads="):
            continue
        mt = LINE.match(ln)
        if not mt:
            raise AggError("%s: 未知の行: %r" % (path, ln))
        kind = "SME" if mt.group(1).startswith("SME") else "NEON"
        shape = (int(mt.group(2)), int(mt.group(3)), int(mt.group(4)))
        try:
            g = float(mt.group(5))
        except ValueError:
            raise AggError("%s: 数値でない GFLOPS: %r" % (path, ln))
        if not math.isfinite(g) or g <= 0:
            raise AggError("%s: 非有限の GFLOPS: %r" % (path, ln))
        d = pts.setdefault(shape, {})
        if kind in d:
            raise AggError("%s: 重複行 %s %s" % (path, kind, shape))
        d[kind] = g
    return pts


def load_series(d, prefix, expected):
    """5 run 分を読み、各 run が expected 形状集合をちょうど 1 回ずつ（NEON・SME 両方）含むか検証する。"""
    runs = []
    for r in range(1, ROUNDS + 1):
        p = os.path.join(d, "%s_run%d.log" % (prefix, r))
        if not os.path.isfile(p):
            raise AggError("欠損: %s" % p)
        pts = parse_variant_log(p)
        if set(pts) != set(expected) or any(set(v) != {"NEON", "SME"} for v in pts.values()):
            raise AggError("%s: 形状集合が期待と一致しない（%d 点）" % (p, len(pts)))
        runs.append(pts)
    return runs


def r4_candidates(ok):
    """ok[(mn, k)] -> bool から R4 の採用候補を返す（RULE §3。#1587 の
    `cpu-gemm-sme-fmopa-1587/aggregate.py::candidates` と同じ定義）。

    しきい値 (a, b) は mn>=a かつ k>=b の全格子点が 5/5 で SME>=NEON のときだけ採用可能とし、
    採用可能なしきい値のパレート極小を候補とする。孤立した勝ち点（右上に負け点がある）は候補にしない。
    格子点の欠損は #1587（KeyError で停止）に合わせて判定不能（AggError）とする。"""
    missing = [(mn, k) for mn in R4_MN for k in R4_K if (mn, k) not in ok]
    if missing:
        raise AggError("R4 格子点の欠損 %s（採用候補を判定できない）" % missing)
    feas = [(a, b) for a in R4_MN for b in R4_K
            if all(ok[(mn, k)] for mn in R4_MN for k in R4_K if mn >= a and k >= b)]
    return sorted(
        (a, b) for (a, b) in feas
        if not any(c <= a and d <= b and (c, d) != (a, b) for (c, d) in feas)
    )


def r4_analysis(runs):
    """格子点ごとの SME/NEON 比・5/5 で SME>=NEON か・パレート極小・K 別の参考領域。"""
    table = {}
    for mn in R4_MN:
        for k in R4_K:
            ratios = [r[(mn, mn, k)]["SME"] / r[(mn, mn, k)]["NEON"] for r in runs]
            table[(mn, k)] = dict(median=statistics.median(ratios), all_ge=all(x >= 1.0 for x in ratios))
    ok = {key for key, v in table.items() if v["all_ge"]}
    minimal = r4_candidates({key: v["all_ge"] for key, v in table.items()})
    region = {}
    for kk in (64, 128, 256):
        pts = [(mn, k) for mn in (256, 512) for k in R4_K if k >= kk]
        region[kk] = all(p in ok for p in pts)
    current_ok = all((mn, k) in ok for mn in R4_MN if mn >= CUR_MIN_MN for k in R4_K if k >= CUR_MIN_K)
    return table, minimal, region, current_ok


def rect_analysis(runs):
    out = []
    for sh in RECT_SHAPES:
        ratios = [r[sh]["SME"] / r[sh]["NEON"] for r in runs]
        elig = sh[0] >= CUR_MIN_MN and sh[1] >= CUR_MIN_MN and sh[2] >= CUR_MIN_K
        out.append((sh, statistics.median(ratios), sum(1 for x in ratios if x >= 1.0), elig))
    return out


def _no_dup_keys(pairs):
    """json.loads の object_pairs_hook。同じキーが 2 回現れる行（後勝ちで黙って潰れる）を拒否する。"""
    d = {}
    for k, v in pairs:
        if k in d:
            raise AggError("重複キー %r" % k)
        d[k] = v
    return d


def validate_row(o, where):
    """1 行の構造と識別フィールドの型を検証する（不正は AggError）。

    行はオブジェクトであること（null・配列・数値・文字列の行を拒否）、task は str、size は
    bool を除く int（配列・float・文字列・true/false を拒否）、mode は省略または str。
    値域（期待セル集合・device・parity）は checked_cells、数値フィールドは fnum が検証する。"""
    if not isinstance(o, dict):
        raise AggError("%s: 行がオブジェクトではない（%s）" % (where, type(o).__name__))
    t, sz, m = o.get("task"), o.get("size"), o.get("mode", "fresh")
    if not isinstance(t, str) or isinstance(sz, bool) or not isinstance(sz, int) or not isinstance(m, str):
        raise AggError("%s: 識別フィールドの型が不正: task=%r size=%r mode=%r" % (where, t, sz, m))
    return o


def read_jsonl(path, skip_bad=False):
    """JSONL の全行を検証して返す。JSON 不正・構造不正は AggError。

    skip_bad=True は checksum_mismatch 専用で、不正な行だけを飛ばして残りを返す（FAIL 判定を
    他の行の異常から独立させるため。不正行そのものは本流の checked_cells 側で undetermined になる）。"""
    if not os.path.isfile(path):
        raise AggError("ファイルがありません: %s" % path)
    rows = []
    for i, ln in enumerate(read_lines(path), 1):
        ln = ln.strip()
        if not ln:
            continue
        where = "%s:%d" % (path, i)
        try:
            try:
                o = json.loads(ln, object_pairs_hook=_no_dup_keys)
            except (ValueError, RecursionError) as e:
                raise AggError("%s JSON 不正: %s" % (where, e))
            rows.append(validate_row(o, where))
        except AggError:
            if not skip_bad:
                raise
    return rows


def fnum(v, what):
    try:
        ok = not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)
    except OverflowError:  # float へ変換できない巨大な JSON 整数
        ok = False
    if not ok:
        raise AggError("非数値の %s: %r" % (what, v))
    return float(v)


def cells_of(rows, torch=False):
    out = {}
    for r in rows:
        key = (r.get("task"), r.get("size"), "fresh" if torch else r.get("mode", "fresh"))
        out.setdefault(key, []).append(r)
    return out


def series(cells, key, field, what):
    rows = cells.get(key)
    if rows is None or len(rows) != ROUNDS:
        raise AggError("%s: セル %s が %s 件（%d 件を要求）" % (what, key, "欠損" if rows is None else len(rows), ROUNDS))
    return [fnum(r.get(field), "%s %s.%s" % (what, key, field)) for r in rows]


def checked_cells(rows, expected, what, torch=False):
    """行を device／parity で検証し、セル集合が expected と完全一致（余分・欠損なし）か確かめて返す（RULE P-CELLS）。"""
    for r in rows:
        if r.get("device") != "cpu":
            raise AggError("%s: device が cpu ではない行: %r" % (what, r.get("device")))
        pf = r.get("parity_fail_count")
        # PyTorch は記録専用（RULE §7）。Accelerate を FMA 参照と比べる parity は非 0 になりうるので判定に使わない
        if not torch and pf is not None and (isinstance(pf, bool) or not isinstance(pf, int) or pf != 0):
            raise AggError("%s: parity_fail_count が 0 でない: %r" % (what, pf))
    cells = cells_of(rows, torch=torch)
    if set(cells) != set(expected):
        raise AggError("%s: セル集合が期待と完全一致しない（余分: %s／欠損: %s）" % (
            what, sorted(map(str, set(cells) - set(expected))), sorted(map(str, set(expected) - set(cells)))))
    for key, rs in cells.items():
        if len(rs) != ROUNDS:
            raise AggError("%s: セル %s が %d 件（%d 件を要求）" % (what, key, len(rs), ROUNDS))
    return cells


TORCH_FAIL_LINE = re.compile(r"^torch round=([1-9][0-9]*) task=(gemm|train|infer) size=([1-9][0-9]*) : (.*)$")
TORCH_FAIL_TOTAL = re.compile(r"^torch_failed_total=(0|[1-9][0-9]*)$")


def read_torch_failures(d, label):
    """run_ab_sme_macmini_cpu.sh の PyTorch 失敗ログを厳格に読み、[(round, task, size, 理由)] を返す。

    形式: 失敗 1 件ごとに `torch round=<r> task=<task> size=<n> : <stderr 末尾>`、最後に
    `torch_failed_total=<n>` を 1 行。ファイル欠損・複数・未知の行・round／セルの範囲外・合計行の欠損や
    重複・件数不一致はすべて AggError（呼び出し側で注記へ回す。PyTorch は判定に使わない）。"""
    hits = glob.glob(os.path.join(d, "torch-failed-3176-*-cpu-%s.log" % label))
    if len(hits) != 1:
        raise AggError("PyTorch 失敗記録が 1 件ではない: %r" % hits)
    lines = [ln for ln in read_lines(hits[0]) if ln.strip()]
    if not lines:
        raise AggError("PyTorch 失敗記録が空（torch_failed_total 行なし）: %s" % hits[0])
    m = TORCH_FAIL_TOTAL.match(lines[-1])
    if not m:
        raise AggError("PyTorch 失敗記録の最終行が torch_failed_total=<n> ではない: %r" % lines[-1])
    torch_cells = {(t, s) for t, s, _ in CELLS}
    fails = []
    for i, ln in enumerate(lines[:-1], 1):
        f = TORCH_FAIL_LINE.match(ln)
        if not f:
            raise AggError("PyTorch 失敗記録の %d 行目が形式外: %r" % (i, ln))
        rnd, task, size = int(f.group(1)), f.group(2), int(f.group(3))
        if rnd > ROUNDS or (task, size) not in torch_cells:
            raise AggError("PyTorch 失敗記録の %d 行目が範囲外: round=%d task=%s size=%d" % (i, rnd, task, size))
        fails.append((rnd, task, size, f.group(4).strip()))
    if int(m.group(1)) != len(fails):
        raise AggError("PyTorch 失敗記録の件数不一致: total=%s 行数=%d" % (m.group(1), len(fails)))
    return fails


def ab_analysis(d, label):
    before, after = {}, {}
    for task in ("gemm", "train", "infer"):
        exp = [k for k in CELLS if k[0] == task]
        before[task] = checked_cells(read_jsonl(os.path.join(d, "results-before-%s-cpu-%s.jsonl" % (label, task))), exp, "before/" + task)
        after[task] = checked_cells(read_jsonl(os.path.join(d, "results-after-%s-cpu-%s.jsonl" % (label, task))), exp, "after/" + task)
    # PyTorch は記録専用。欠損・不正は A/B 判定（特に checksum FAIL）から切り離して注記のみにする（RULE §6・§7）
    torch, torch_note = None, None
    try:
        hits = glob.glob(os.path.join(d, "results-torch-3176-*-%s-cpu.jsonl" % label))
        if len(hits) != 1:
            raise AggError("torch の JSONL が 1 件ではない: %r" % hits)
        texp = sorted({(t, s, "fresh") for t, s, _ in CELLS})
        torch = checked_cells(read_jsonl(hits[0]), texp, "torch", torch=True)
    except (OSError, AggError) as e:
        torch_note = str(e)
    # PyTorch の失敗記録（A/B 本体の終了コードとは独立）。失敗の有無・記録の異常はどちらも注記に足すだけ
    try:
        fails = read_torch_failures(d, label)
        if fails:
            fn = "PyTorch 計測失敗 %d 件: %s" % (len(fails), "；".join(
                "round%d %s %d（%s）" % (r, t, s, why) for r, t, s, why in fails))
            torch_note = fn if not torch_note else torch_note + "／" + fn
    except (OSError, AggError) as e:
        torch_note = str(e) if not torch_note else torch_note + "／" + str(e)
    res = []
    for key in CELLS:
        task, size, _ = key
        b = series(before[task], key, "median_s", "before")
        a = series(after[task], key, "median_s", "after")
        bc = series(before[task], key, "checksum", "before")
        ac = series(after[task], key, "checksum", "after")
        if any(x <= 0 for x in a + b):
            raise AggError("median_s が正でない: %s" % (key,))
        t = None
        if torch is not None:
            try:
                t = series(torch, (task, size, "fresh"), "median_s", "torch")
                if any(x <= 0 for x in t):
                    t = None
            except AggError as e:
                torch_note = torch_note or str(e)
        rr = [x / y for x, y in zip(a, b)]
        res.append(dict(
            key=key, ratio=statistics.median(a) / statistics.median(b), rounds=rr,
            n_le=sum(1 for x in rr if x <= 1.0), n_gt=sum(1 for x in rr if x > 1.0),
            ck_ok=len(set(bc) | set(ac)) == 1,
            t_before=statistics.median([x / y for x, y in zip(t, b)]) if t else None,
            t_after=statistics.median([x / y for x, y in zip(t, a)]) if t else None,
        ))
    return res, torch_note


def checksum_mismatch(d, label):
    """before／after の checksum 不一致を、他セルの欠損・median_s 不正と独立に判定する（RULE §6: FAIL 最優先）。
    読めない行・ファイルは飛ばし、読めた値だけで同一セル内に 2 値以上あれば True。行の構造・識別フィールドの
    検証は read_jsonl と同じ（validate_row）で、不正な行は飛ばす（本流では P-CELLS の undetermined になる）。
    対象セル（CELLS）の行で checksum が null・文字列・非有限値・欠落の場合も「数値として完全一致しない」異常として
    True を返す（RULE §6。実行スクリプトが非数値を FAIL と記録するのと整合させる）。"""
    seen = {}
    for task in ("gemm", "train", "infer"):
        for arm in ("before", "after"):
            path = os.path.join(d, "results-%s-%s-cpu-%s.jsonl" % (arm, label, task))
            try:
                rows = read_jsonl(path, skip_bad=True)
            except (OSError, AggError):
                continue
            for r in rows:
                v = r.get("checksum")
                try:
                    v = fnum(v, "checksum")
                except AggError:
                    if (r.get("task"), r.get("size"), r.get("mode", "fresh")) in CELLS_SET:
                        return True
                    continue
                seen.setdefault((r.get("task"), r.get("size"), r.get("mode", "fresh")), set()).add(v)
    return any(len(vs) > 1 for vs in seen.values())


GATE_ROUND = re.compile(r"^round([0-9]+) (gate=\S+(?: [A-Za-z_0-9]+=\S+)*)$")


def gate_series(d):
    """load_gate.log の round 行から正式系列（5/5 gate_ok=1）かを返す。"""
    paths = glob.glob(os.path.join(d, "r1r2", "load_gate-3176-*.log"))
    if len(paths) != 1:
        return False
    seen = {}
    try:
        lines = read_lines(paths[0])
    except (OSError, AggError):
        return False
    for ln in lines:
        mt = GATE_ROUND.match(ln)
        if mt:
            # run_ab_sme_macmini_cpu.sh は gate_ok の後ろに threshold・load1・waited_s・at を続けて出力する。
            # 後続フィールドは許容し、gate_ok はキー完全一致の値だけを見る（欠落は "" で不合格）
            seen.setdefault(int(mt.group(1)), []).append(kv(mt.group(2)).get("gate_ok", ""))
    # round1〜ROUNDS が各 1 件ずつ・gate_ok がフィールド完全一致で 1 のものだけを正式系列とする
    return sorted(seen) == list(range(1, ROUNDS + 1)) and all(v == ["1"] for v in seen.values())


SHA_LINE = re.compile(r"^on-arm\.patch sha256: ([0-9a-f]{64})$")
LOAD1 = re.compile(r"^[0-9]+(\.[0-9]+)?$")


def rule_patch_sha():
    """同ディレクトリ RULE.txt の on-arm.patch sha256 固定値（取得できなければ AggError）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    for ln in read_lines(os.path.join(here, "RULE.txt")):
        m = SHA_LINE.match(ln)
        if m:
            return m.group(1)
    raise AggError("RULE.txt から patch の sha256 を取得できない")


def kv(line):
    return dict(p.split("=", 1) for p in line.split() if "=" in p)


def orchestration_checks(d):
    """macmini/load_gate.log を検証する（RULE §4 P-R4-EXEC・P-RUNAB・P-COLLECT、§5 R4／rect の負荷ゲート、
    全 run の開始・終了記録）。欠損・重複・非ゼロ終了・ゲート不通過は理由リストへ返す。"""
    why = []
    try:
        lines = read_lines(os.path.join(d, "load_gate.log"))
    except (OSError, AggError):
        return ["P-R4-EXEC: load_gate.log なし（R4・rect の負荷ゲート／開始・終了記録を検証できない）"]
    starts, ends, singles = {}, {}, {}
    for ln in lines:
        f = kv(ln)
        if ln.startswith("START "):
            starts.setdefault((f.get("series"), f.get("run")), []).append(f)
        elif ln.startswith("END "):
            ends.setdefault((f.get("series"), f.get("run")), []).append(f)
        elif ln.startswith("run_ab exit="):
            singles.setdefault("run_ab", []).append(f.get("exit"))
        elif ln.startswith("collect exit="):
            singles.setdefault("collect", []).append(f.get("exit"))
    for series in ("r4", "rect"):
        for r in range(1, ROUNDS + 1):
            key = (series, str(r))
            ss, es = starts.get(key, []), ends.get(key, [])
            if len(ss) != 1 or len(es) != 1:
                why.append("%s run%d: 開始 %d 件・終了 %d 件の記録（各 1 件を要求）" % (series, r, len(ss), len(es)))
                continue
            l1 = ss[0].get("load1", "")
            if ss[0].get("gate") != "pass" or not LOAD1.match(l1) or float(l1) >= 3.0:
                why.append("%s run%d: 負荷ゲート不通過（gate=%s load1=%s。record_only）" % (series, r, ss[0].get("gate"), l1))
            if es[0].get("exit") != "0" or es[0].get("grep_exit") != "0":
                why.append("%s run%d: 診断の終了コードが 0 でない（exit=%s grep_exit=%s）" % (series, r, es[0].get("exit"), es[0].get("grep_exit")))
    extra = sorted(k for k in set(starts) | set(ends) if k[0] not in ("r4", "rect") or k[1] not in [str(i) for i in range(1, ROUNDS + 1)])
    if extra:
        why.append("想定外の series／run の記録: %s" % extra)
    for name, label in (("run_ab", "P-RUNAB"), ("collect", "P-COLLECT")):
        v = singles.get(name, [])
        if v != ["0"]:
            why.append("%s: %s の終了コード記録が exit=0 の 1 件ではない（%r）" % (label, name, v))
    return why


def preconditions(d, label):
    """前提ゲートを 1 か所で評価する。戻り値: (理由リスト, 計算済みデータ)。"""
    why, data = [], {}
    try:
        tv = read_lines(os.path.join(d, "tree_verify.txt"))
        if tv != ["./crates/backend-cpu/src/gemm_blis/mod.rs"]:
            why.append("P-TREE: 差分ファイルが mod.rs の 1 件だけではない")
        if not any("SME_PRODUCTION_ENABLED: bool = true;" in ln for ln in read_lines(os.path.join(d, "gate_constant.txt"))):
            why.append("P-TREE: after の定数が true でない")
    except (OSError, AggError):
        why.append("P-TREE: 記録なし")
    try:
        for arm in ("before", "after"):
            if not any("kernel_enabled: true" in ln for ln in read_lines(os.path.join(d, "sme_probe_%s.log" % arm))):
                why.append("P-R0: %s で kernel_enabled が true でない" % arm)
    except (OSError, AggError):
        why.append("P-R0: 記録なし")
    try:
        sha = read_lines(os.path.join(d, "patch_sha256.txt"))
        if sha != [rule_patch_sha()]:
            why.append("P-TREE: patch の sha256 が RULE 記載と一致しない")
    except (OSError, AggError) as e:
        why.append("P-TREE: patch sha256 を検証できない（%s）" % e)
    why.extend(orchestration_checks(d))
    r4_shapes = [(mn, mn, k) for mn in R4_MN for k in R4_K] + [(mn, mn, k) for mn in R4_MN for k in R4_K]
    try:
        data["r4"] = r4_analysis(load_series(d, "r4", set(r4_shapes)))
    except (OSError, AggError) as e:
        why.append("P-R4-LOG: %s" % e)
    try:
        data["rect"] = rect_analysis(load_series(d, "rect", set(RECT_SHAPES)))
    except (OSError, AggError) as e:
        why.append("P-RECT-LOG: %s" % e)
    try:
        data["ab"], data["torch_note"] = ab_analysis(os.path.join(d, "r1r2"), label)
    except (OSError, AggError) as e:
        why.append("P-CELLS: %s" % e)
    data["ck_fail"] = checksum_mismatch(os.path.join(d, "r1r2"), label)
    data["official"] = gate_series(d)
    if not data["official"]:
        why.append("系列: 5/5 round が gate_ok=1 ではない（record_only）")
    return why, data


def verdict(why, data):
    ab = data.get("ab")
    if data.get("ck_fail") or (ab and any(not r["ck_ok"] for r in ab)):
        return "FAIL（checksum 不一致。ADOPT／REJECT に数えない）"
    if why:
        return "undetermined（" + "；".join(why) + "）"
    reached = [r for r in data["ab"] if r["key"] in REACHED]
    if any(r["n_gt"] == ROUNDS for r in reached):
        return "REJECT（到達セルに 5/5 一貫の後退あり）"
    nonreached = [r for r in data["ab"] if r["key"] not in REACHED]
    if any(r["n_gt"] == ROUNDS for r in nonreached):
        return "REJECT（非到達セルに 5/5 一貫の後退あり）"
    if data["r4"][3] and all(r["ratio"] <= 1.0 and r["n_le"] == ROUNDS for r in reached):
        return "ADOPT 候補（本番切替は #3230、しきい値は #3231。本イシューでは切り替えない）"
    return "undetermined（ADOPT 条件が成立せず REJECT 条件にも当たらない）"


def render(d, label):
    why, data = preconditions(d, label)
    L = ["# 集計（イシュー #3176）", "", "総合判定: **%s**" % verdict(why, data), ""]
    if "ab" in data:
        L += ["## R1／R2／R5（10 セル）", "",
              "| セル | 到達 | after/before 中央値比 | round 別 | <=1.00 の round | checksum | torch÷fandhe(before) | torch÷fandhe(after) |",
              "|---|---|---|---|---|---|---|---|"]
        for r in data["ab"]:
            t, s, m = r["key"]
            L.append("| %s %d %s | %s | %.4f | %s | %d/5 | %s | %s | %s |" % (
                t, s, m, "到達" if r["key"] in REACHED else "非到達", r["ratio"],
                " ".join("%.3f" % x for x in r["rounds"]), r["n_le"],
                "一致" if r["ck_ok"] else "**不一致**",
                "%.3f" % r["t_before"] if r["t_before"] is not None else "n/a",
                "%.3f" % r["t_after"] if r["t_after"] is not None else "n/a"))
        if data.get("torch_note"):
            L += ["", "注記（PyTorch は記録専用で判定に影響しない）: " + data["torch_note"]]
        reached = [r for r in data["ab"] if r["key"] in REACHED]
        ac1 = all(r["ratio"] <= 1.0 and r["n_le"] == ROUNDS for r in reached) and all(r["ck_ok"] for r in data["ab"])
        L += ["", "AC1 直接判定（到達セルの中央値 <=1.00 かつ 5/5、checksum 一致）: %s" % ("成立" if ac1 else "不成立"), ""]
    if "r4" in data:
        table, minimal, region, cur = data["r4"]
        L += ["## R4 格子（SME/NEON 中央値比。5/5 で >=1.0 の点に *）", "", "| min(m,n) \\ k | " + " | ".join(map(str, R4_K)) + " |",
              "|---|" + "---|" * len(R4_K)]
        for mn in R4_MN:
            L.append("| %d | " % mn + " | ".join("%.3f%s" % (table[(mn, k)]["median"], "*" if table[(mn, k)]["all_ge"] else "") for k in R4_K) + " |")
        L += ["", "パレート極小の採用候補 (min(m,n), k): %s" % (minimal or "なし"),
              "参考: K>=Kmin かつ min(m,n)∈{256,512} の全点が SME>=NEON: " + ", ".join("Kmin=%d: %s" % (k, "成立" if v else "不成立") for k, v in sorted(region.items())),
              "現行しきい値領域（min>=256 かつ k>=64）が全点成立: %s" % ("成立" if cur else "不成立"), ""]
    if "rect" in data:
        L += ["## 長方形 4 形状（記録のみ。判定に使わない）", "", "| 形状 (m,n,k) | SME/NEON 中央値比 | 5 run 中 >=1.0 | 現行 sme_shape_eligible |", "|---|---|---|---|"]
        for sh, med, n, el in data["rect"]:
            L.append("| %s | %.3f | %d/5 | %s |" % (sh, med, n, "可" if el else "否"))
        L.append("")
    L += ["## 前提ゲートの不成立理由", ""] + (["- " + w for w in why] or ["なし"]) + [""]
    return "\n".join(L)


# ---- self-test（合成入力）----
def _w(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _variant_log(path, shapes, sme_scale):
    with open(path, "w") as f:
        f.write("num_threads=10\n")
        for sh in shapes:
            f.write("variant=NEON(TwoDDynamic) size=(%d,%d,%d) median_gflops=100.0\n" % sh)
            f.write("variant=SME(TwoDDynamicSme) size=(%d,%d,%d) median_gflops=%.1f\n" % (sh + (100.0 * sme_scale,)))


def _fixture(d, after_scale=0.9, ck_after=1.5, sme_scale=1.5, probe_true=True, gate_ok=True, r4_gate_ok=True, run_ab_exit=0):
    os.makedirs(os.path.join(d, "r1r2"))
    with open(os.path.join(d, "tree_verify.txt"), "w") as f:
        f.write("./crates/backend-cpu/src/gemm_blis/mod.rs\n")
    with open(os.path.join(d, "gate_constant.txt"), "w") as f:
        f.write("1:const SME_PRODUCTION_ENABLED: bool = true;\n")
    for arm in ("before", "after"):
        with open(os.path.join(d, "sme_probe_%s.log" % arm), "w") as f:
            f.write("sme_report=Report { kernel_enabled: %s }\n" % ("true" if probe_true else "false"))
    with open(os.path.join(d, "patch_sha256.txt"), "w") as f:
        f.write(rule_patch_sha() + "\n")
    with open(os.path.join(d, "load_gate.log"), "w") as f:
        for series in ("r4", "rect"):
            for r in range(1, ROUNDS + 1):
                f.write("START series=%s run=%d gate=%s load1=%s waited_s=0 at=x\n" % (series, r, "pass" if r4_gate_ok else "unavailable", "0.5" if r4_gate_ok else "9.0"))
                f.write("END series=%s run=%d exit=0 grep_exit=0 end_load1=0.5\n" % (series, r))
        f.write("run_ab exit=%d\n" % run_ab_exit)
        f.write("collect exit=0\n")
    r4 = [(mn, mn, k) for mn in R4_MN for k in R4_K]
    for r in range(1, ROUNDS + 1):
        _variant_log(os.path.join(d, "r4_run%d.log" % r), r4, sme_scale)
        _variant_log(os.path.join(d, "rect_run%d.log" % r), RECT_SHAPES, sme_scale)
    for arm, sc in (("before", 1.0), ("after", after_scale)):
        for task in ("gemm", "train", "infer"):
            rows = []
            for rnd in range(ROUNDS):
                for t, s, m in CELLS:
                    if t == task:
                        ck = ck_after if (arm == "after" and t == "train" and rnd == 0) else 1.5
                        rows.append(dict(task=t, size=s, mode=m, device="cpu", median_s=0.01 * sc * (1 + rnd * 0.01), checksum=ck))
            _w(os.path.join(d, "r1r2", "results-%s-L-cpu-%s.jsonl" % (arm, task)), rows)
    _w(os.path.join(d, "r1r2", "results-torch-3176-sme-L-cpu.jsonl"),
       [dict(task=t, size=s, device="cpu", median_s=0.02, checksum=1.0) for _ in range(ROUNDS) for t, s, m in CELLS if m == "fresh"])
    with open(os.path.join(d, "r1r2", "torch-failed-3176-sme-cpu-L.log"), "w") as f:
        f.write("torch_failed_total=0\n")
    with open(os.path.join(d, "r1r2", "load_gate-3176-sme-cpu-L.log"), "w") as f:
        f.write("threshold=3.0 rule_threshold=3.0 series=official\n")
        for i in range(1, ROUNDS + 1):
            # run_ab_sme_macmini_cpu.sh の実出力形式（gate_ok の後ろに threshold・load1・waited_s・at が続く）
            f.write("round%d gate=%s gate_ok=%d threshold=3.0 load1=0.5 waited_s=0 at=2026-10-21T00:00:00Z\n" % (i, "pass" if gate_ok else "timeout", 1 if gate_ok else 0))


def _drop_line(path, prefix):
    keep = [ln for ln in read_lines(path) if not ln.startswith(prefix)]
    with open(path, "w") as f:
        f.write("\n".join(keep) + "\n")


def _edit_jsonl(d, fn):
    p = os.path.join(d, "r1r2", "results-after-L-cpu-gemm.jsonl")
    rows = read_jsonl(p)
    fn(rows[0])
    _w(p, rows)


def _append_jsonl(d, row):
    p = os.path.join(d, "r1r2", "results-after-L-cpu-gemm.jsonl")
    _w(p, read_jsonl(p) + [row])


def _r4_candidate_tests():
    """r4_candidates の単体ケース（Cursor Bugbot 指摘: 孤立した勝ちを候補にしない）。"""
    grid = [(mn, k) for mn in R4_MN for k in R4_K]
    # 孤立した勝ち (256, 64): (512, 64) が負けなので (256, 64) は採用可能でない
    ok = {p: False for p in grid}
    ok[(256, 64)] = True
    assert r4_candidates(ok) == [], r4_candidates(ok)
    ok = {p: (p[0] >= 256 and p[1] >= 64) for p in grid}
    ok[(512, 64)] = False
    assert (256, 64) not in r4_candidates(ok), r4_candidates(ok)
    assert r4_candidates(ok) == [(256, 128)], r4_candidates(ok)
    # 現行しきい値の領域だけ勝つ: 候補は (256, 64) のみ
    ok = {p: (p[0] >= 256 and p[1] >= 64) for p in grid}
    assert r4_candidates(ok) == [(256, 64)], r4_candidates(ok)
    # 全勝: 最小の格子点 1 つ
    assert r4_candidates({p: True for p in grid}) == [(R4_MN[0], R4_K[0])]
    # 全敗: 候補なし
    assert r4_candidates({p: False for p in grid}) == []
    # 欠損: 判定不能（AggError）
    ok = {p: True for p in grid}
    del ok[(512, 256)]
    try:
        r4_candidates(ok)
    except AggError:
        pass
    else:
        raise AssertionError("欠損で AggError にならない")


def self_test():
    _r4_candidate_tests()
    cases = [
        (dict(), "ADOPT 候補"),
        (dict(after_scale=1.2), "REJECT"),
        (dict(ck_after=2.5), "FAIL"),
        (dict(probe_true=False), "undetermined"),
        (dict(gate_ok=False), "undetermined"),
        (dict(sme_scale=0.5), "undetermined"),
        (dict(r4_gate_ok=False), "undetermined"),
        (dict(run_ab_exit=1), "undetermined"),
    ]
    for kw, want in cases:
        with tempfile.TemporaryDirectory() as d:
            _fixture(d, **kw)
            text = render(d, "L")
            assert want in text.split("\n")[2], (kw, text.split("\n")[2])
    with tempfile.TemporaryDirectory() as d:  # torch 欠損 + checksum 不一致でも FAIL を失わない
        _fixture(d, ck_after=2.5)
        os.remove(os.path.join(d, "r1r2", "results-torch-3176-sme-L-cpu.jsonl"))
        assert "FAIL" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:  # 別タスクの記録欠損・median_s 不正で ab_analysis が例外でも確認済み checksum 不一致は FAIL
        _fixture(d)
        _edit_jsonl(d, lambda r: r.update(checksum=9.5))
        os.remove(os.path.join(d, "r1r2", "results-after-L-cpu-infer.jsonl"))
        assert "FAIL" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        _edit_jsonl(d, lambda r: r.update(checksum=9.5, median_s=-1))
        assert "FAIL" in render(d, "L").split("\n")[2]
    for bad in (None, "x", float("nan"), float("inf")):  # 非数値 checksum は undetermined でなく FAIL
        with tempfile.TemporaryDirectory() as d:
            _fixture(d)
            _edit_jsonl(d, lambda r, bad=bad: r.update(checksum=bad))
            assert "FAIL" in render(d, "L").split("\n")[2], bad
    with tempfile.TemporaryDirectory() as d:  # torch の parity_fail_count 非 0 は記録のみで ADOPT を妨げない
        _fixture(d)
        p = os.path.join(d, "r1r2", "results-torch-3176-sme-L-cpu.jsonl")
        rows = read_jsonl(p)
        for r in rows:
            r["parity_fail_count"] = 3
        _w(p, rows)
        assert "ADOPT 候補" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:  # 異常入力（重複行）は捨てず undetermined
        _fixture(d)
        with open(os.path.join(d, "r4_run1.log"), "a") as f:
            f.write("variant=SME(TwoDDynamicSme) size=(64,64,32) median_gflops=1.0\n")
        assert "undetermined" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:  # 数値でない GFLOPS は集計を中断せず、他セルの checksum FAIL も出す
        _fixture(d, ck_after=2.5)
        with open(os.path.join(d, "r4_run1.log"), "a") as f:
            f.write("variant=SME(TwoDDynamicSme) size=(1,2,3) median_gflops=broken\n")
        assert "FAIL" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:  # round1 の 5 重複・round2〜5 欠損は正式系列でない
        _fixture(d)
        with open(os.path.join(d, "r1r2", "load_gate-3176-sme-cpu-L.log"), "w") as f:
            f.write("threshold=3.0 rule_threshold=3.0 series=official\n")
            for _ in range(ROUNDS):
                f.write("round1 gate=pass gate_ok=1\n")
        assert "undetermined" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:  # 実形式の行（後続フィールド付き）なら正式系列になり、1 round 欠落・gate_ok 欠落は不可
        _fixture(d)
        assert gate_series(d), "実形式の 5/5 round が正式系列と判定されない"
        gp = os.path.join(d, "r1r2", "load_gate-3176-sme-cpu-L.log")
        good = [ln for ln in open(gp).read().splitlines() if ln.startswith("round")]
        with open(gp, "w") as f:
            f.write("\n".join(good[:-1]) + "\n")
        assert not gate_series(d)
        with open(gp, "w") as f:
            f.write("\n".join(good[:-1] + ["round5 gate=pass threshold=3.0 load1=0.5"]) + "\n")
        assert not gate_series(d)
    with tempfile.TemporaryDirectory() as d:  # gate_ok=10 のような部分一致は不可
        _fixture(d)
        with open(os.path.join(d, "r1r2", "load_gate-3176-sme-cpu-L.log"), "w") as f:
            for i in range(1, ROUNDS + 1):
                f.write("round%d gate=pass gate_ok=10\n" % i)
        assert "undetermined" in render(d, "L").split("\n")[2]
    with tempfile.TemporaryDirectory() as d:  # 非有限値
        _fixture(d)
        with open(os.path.join(d, "rect_run2.log"), "a") as f:
            f.write("variant=SME(TwoDDynamicSme) size=(1,2,3) median_gflops=nan\n")
        assert "undetermined" in render(d, "L").split("\n")[2]
    for name, mut in (
        ("欠損 END", lambda d: _drop_line(os.path.join(d, "load_gate.log"), "END series=rect run=3")),
        ("patch sha 不一致", lambda d: open(os.path.join(d, "patch_sha256.txt"), "w").write("0" * 64 + "\n")),
        ("patch sha 欠損", lambda d: os.remove(os.path.join(d, "patch_sha256.txt"))),
        ("device 不一致", lambda d: _edit_jsonl(d, lambda r: r.update(device="gpu"))),
        ("parity 失敗", lambda d: _edit_jsonl(d, lambda r: r.update(parity_fail_count=2))),
        ("余分セル", lambda d: _append_jsonl(d, dict(task="gemm", size=4096, mode="fresh", device="cpu", median_s=0.01, checksum=1.5))),
        ("median_s 巨大整数", lambda d: _edit_jsonl(d, lambda r: r.update(median_s=10 ** 400))),
    ):
        with tempfile.TemporaryDirectory() as d:
            _fixture(d)
            mut(d)
            assert "undetermined" in render(d, "L").split("\n")[2], name
    # JSONL の構造・識別フィールドの型の不正（PR #3501 codex P2）。例外で落ちず undetermined になること。
    # 本流（checked_cells）でも独立確認（checksum_mismatch）でも AttributeError／TypeError を出さない
    good = dict(task="gemm", size=512, mode="fresh", device="cpu", median_s=0.01, checksum=1.5)
    gemm_after = os.path.join("r1r2", "results-after-L-cpu-gemm.jsonl")
    torch_f = os.path.join("r1r2", "results-torch-3176-sme-L-cpu.jsonl")
    bad_lines = [
        ("null 行", gemm_after, "null"),
        ("配列行", gemm_after, "[1, 2]"),
        ("数値行", gemm_after, "3.5"),
        ("文字列行", gemm_after, '"gemm"'),
        ("size が配列", gemm_after, json.dumps(dict(good, size=[512]))),
        ("size が bool", gemm_after, json.dumps(dict(good, size=True))),
        ("size が float", gemm_after, json.dumps(dict(good, size=512.0))),
        ("size が文字列", gemm_after, json.dumps(dict(good, size="512"))),
        ("task が配列", gemm_after, json.dumps(dict(good, task=["gemm"]))),
        ("mode が配列", gemm_after, json.dumps(dict(good, mode=["fresh"]))),
        ("重複キー", gemm_after, '{"task": "gemm", "size": 512, "size": 1024, "mode": "fresh", "device": "cpu", "median_s": 0.01, "checksum": 1.5}'),
        ("深い入れ子", gemm_after, "[" * 100000),
    ]
    for name, rel, line in bad_lines:
        with tempfile.TemporaryDirectory() as d:
            _fixture(d)
            with open(os.path.join(d, rel), "a") as f:
                f.write(line + "\n")
            head = render(d, "L").split("\n")[2]
            assert "undetermined" in head and "P-CELLS" in head, (name, head)
            assert not checksum_mismatch(os.path.join(d, "r1r2"), "L"), name
    with tempfile.TemporaryDirectory() as d:  # 不正行が混ざっても、同じファイルの読める行の checksum 不一致は FAIL のまま
        _fixture(d)
        _edit_jsonl(d, lambda r: r.update(checksum=9.5))
        with open(os.path.join(d, gemm_after), "a") as f:
            f.write("null\n" + json.dumps(dict(good, size=[512])) + "\n")
        assert "FAIL" in render(d, "L").split("\n")[2]
    for name, line in (("torch の null 行", "null"), ("torch の size が配列", json.dumps(dict(good, size=[512])))):
        with tempfile.TemporaryDirectory() as d:  # PyTorch は記録専用。不正行は注記に回り ADOPT を妨げない（落ちない）
            _fixture(d)
            with open(os.path.join(d, torch_f), "a") as f:
                f.write(line + "\n")
            text = render(d, "L")
            assert "ADOPT 候補" in text.split("\n")[2] and "注記" in text, name
    # PyTorch だけが失敗した run（run_ab は exit 0・torch JSONL の当該行が欠損・失敗ログに記録あり）は
    # A/B 判定を保ち、失敗は注記に出る（PR #3501 codex P2。RULE §7「PyTorch は記録のみ」）
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        p = os.path.join(d, torch_f)
        rows = read_jsonl(p)
        _w(p, [r for i, r in enumerate(rows) if not (i < 5 and r["task"] == "gemm" and r["size"] == 1024)])
        with open(os.path.join(d, "r1r2", "torch-failed-3176-sme-cpu-L.log"), "w") as f:
            f.write("torch round=1 task=gemm size=1024 : RuntimeError: boom\ntorch_failed_total=1\n")
        text = render(d, "L")
        head = text.split("\n")[2]
        assert "ADOPT 候補" in head, head
        assert "注記" in text and "PyTorch 計測失敗 1 件" in text and "round1 gemm 1024" in text, text
        assert "P-RUNAB" not in text, text
    for name, body in (("合計行なし", "torch round=1 task=gemm size=512 : x\n"),
                       ("件数不一致", "torch round=1 task=gemm size=512 : x\ntorch_failed_total=2\n"),
                       ("範囲外 round", "torch round=6 task=gemm size=512 : x\ntorch_failed_total=1\n"),
                       ("未知セル", "torch round=1 task=gemm size=4096 : x\ntorch_failed_total=1\n"),
                       ("形式外の行", "garbage\ntorch_failed_total=0\n"),
                       ("ファイル欠損", None)):
        with tempfile.TemporaryDirectory() as d:  # 失敗記録の異常は厳格に検出するが、PyTorch は判定外のため注記のみ
            _fixture(d)
            fp = os.path.join(d, "r1r2", "torch-failed-3176-sme-cpu-L.log")
            if body is None:
                os.remove(fp)
            else:
                with open(fp, "w") as f:
                    f.write(body)
            try:
                read_torch_failures(os.path.join(d, "r1r2"), "L")
                raise AssertionError("AggError にならない: " + name)
            except AggError:
                pass
            text = render(d, "L")
            assert "ADOPT 候補" in text.split("\n")[2] and "注記" in text, name
    # SME/NEON 本体の失敗（after 腕のセル欠損 → run_ab が非 0）は、PyTorch が全件成功でも undetermined、
    # 同じ run に checksum 不一致があれば FAIL（PyTorch 側の切り離しで本体の失敗を弱めない）
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, run_ab_exit=1)
        p = os.path.join(d, gemm_after)
        _w(p, read_jsonl(p)[1:])
        head = render(d, "L").split("\n")[2]
        assert "undetermined" in head and "P-RUNAB" in head and "P-CELLS" in head, head
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, run_ab_exit=1, ck_after=2.5)
        assert "FAIL" in render(d, "L").split("\n")[2]
    for name, data in (("UTF-8 不正の JSONL", b"\xff\xfe\n"),):
        with tempfile.TemporaryDirectory() as d:
            _fixture(d)
            with open(os.path.join(d, gemm_after), "ab") as f:
                f.write(data)
            assert "undetermined" in render(d, "L").split("\n")[2], name
    with tempfile.TemporaryDirectory() as d:  # UTF-8 不正の load_gate.log・round ログでも落ちない
        _fixture(d)
        for rel in ("load_gate.log", os.path.join("r1r2", "load_gate-3176-sme-cpu-L.log"), "sme_probe_after.log"):
            with open(os.path.join(d, rel), "ab") as f:
                f.write(b"\xff\n")
        assert "undetermined" in render(d, "L").split("\n")[2]
    print("self-test ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) not in (2, 3):
        print(__doc__, file=sys.stderr)
        return 2
    print(render(argv[1], argv[2] if len(argv) == 3 else "3176-r1"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
