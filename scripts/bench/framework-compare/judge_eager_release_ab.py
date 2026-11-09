#!/usr/bin/env python3
"""イシュー #3226: eager matmul 自動解放 A/B の判定ツール（fail-closed）。

`docs/perf/logs/cpu-gemm-reuse-release-3226/RULE.txt` の [判定] を機械的に適用する。
`run_ab_eager_release_cpu.sh` が機体ごとに出す before/after の JSONL
（gemm・train・infer）を読み、ADOPT / REJECT / undetermined を返す。手書きの判定表を
作らないための単一ソース（寛容パース禁止: 欠損・不正行・件数不足は終了コード 2）。

終了コード: 0 = ADOPT / 1 = REJECT / 2 = undetermined（判定不能・データ不正を含む）。

round の対応は JSONL の行順（スクリプトが round ごとに +1 行を保証）で i 番目同士を組にする。
ratio = after / before。セルごとの before・after は各ちょうど 5 件。
判定:
- 処置セル = GB10 の gemm reuse N=256/512/1024/2048。
- ADOPT: 処置 4 セルの中央値 ratio <= 0.95 かつ 5/5 round で ratio < 1.00、
  両機体の非処置セルの中央値 ratio <= 1.00、checksum 全一致、両機体が揃っている。
- REJECT: 全 round で ratio > 1.00 のセル／処置セルの中央値 ratio > 1.00／checksum 不一致。
- 負荷ゲートの証跡: 機体ごとに `run_ab_eager_release_cpu.sh` が出す `load_gate-*.log` を
  `--<machine>-gate-log` で必須入力とする（JSONL には系列の属性が残らないため）。ヘッダが
  `series=official`・`threshold` と `rule_threshold` がともに機体の規則値（GB10=1.0・Mac mini=3.0）・`precheck=assert`、かつ round1〜5 がすべて
  `gate=pass gate_ok=1` でなければ undetermined（欠落・参考／smoke 系列・不通過 round を含む）。
  この検査は ADOPT・REJECT のどちらを出す経路よりも前に必ず通す。
- それ以外は undetermined。`--record-only` 指定（負荷ゲート不通過の系列）は ADOPT も REJECT も出さず undetermined とし、後退は参考情報として残す。
"""
# `--rule-set 4158`（イシュー #4158・`docs/perf/logs/cpu-gemm-reuse-release-sizegate-4158/RULE.txt`）:
# サイズゲート付き eager release の GB10 単独判定。診断経路（`--diag-legacy-tape-reuse`。JSONL の
# `diag` タグ）と既定経路のセルを別セルとして扱い、処置 = 診断経路の gemm reuse N=256/512/1024/2048、
# ガード = それ以外の 16 セル（診断経路の N=4096 と train reuse を含む）。ADOPT／REJECT の形は
# 3226 と同一（0.95・5/5・中央値 <= 1.00）で緩めない。GB10 単独で判定する（--require-both 不要）。
# 負荷ゲートログは round ごとに `compute_procs=0` も要求する（GB10 専有の証跡）。
# `--rule-set 4159`（イシュー #4159・`docs/perf/logs/cpu-gemm-reuse-release-sizegate-macmini-4159/RULE.txt`）:
# 同じサイズゲートの Mac mini 非後退判定（macmini 単独）。判定セル = 診断経路の 5 セル（gemm reuse
# N=256/512/1024/2048 と train reuse。機構が実際に働く）、A/A 参照セル = 既定経路の 12 セル（常駐経路に
# eager release は届かず構造上 A/A。後退の判定には数えない）。N=4096 は構造上 A/A のため含めない。
#   REGRESS      : 判定セルのどれかが 5/5 round で ratio > 1.00、または checksum 不一致（17 セル）
#   NONREGRESS   : 判定セル 5 つすべての中央値 ratio <= 1.00 かつ A/A 参照セルに 5/5 後退がなく checksum 全一致
#   undetermined : 上記以外。A/A 参照セルの 5/5 後退は計測系の系統差の疑い（REGRESS にせず undetermined）
# ADOPT は出さない（既定の反転は両機体の判定を見て #3227 で判断する）。NB-RANGE5 は参考列で判定に使わない。
import argparse
import json
import math
import statistics
import sys

ROUNDS = 5
ADOPT_RATIO = 0.95
PROFILES = {
    "gb10": {"gemm_sizes": [256, 512, 1024, 2048, 4096]},
    "macmini": {"gemm_sizes": [256, 512, 1024, 2048]},
}
TREATED_SIZES = (256, 512, 1024, 2048)
RULE_SETS = ("3226", "4158", "4159")
DIAG_TAG = "legacy_tape_reuse"
# 機体ごとの専有ゲート規則値（RULE.txt の load1 閾値。run_ab_eager_release_cpu.sh の RULE_LOAD_GATE と同値）。
RULE_LOAD_THRESHOLD = {"gb10": 1.0, "macmini": 3.0}


class Undetermined(Exception):
    pass


def expected_cells(profile):
    cells = []
    for s in PROFILES[profile]["gemm_sizes"]:
        for m in ("fresh", "reuse"):
            cells.append(("gemm", s, m))
    for t in ("train", "infer"):
        for m in ("fresh", "reuse"):
            cells.append((t, 64, m))
    return cells


def expected_cells_4158():
    """4158 のセル集合。キーは (task, size, mode, diag)。diag は '' か DIAG_TAG。"""
    cells = []
    for s in (256, 512, 1024, 2048, 4096):
        for m in ("fresh", "reuse"):
            cells.append(("gemm", s, m, ""))
        cells.append(("gemm", s, "reuse", DIAG_TAG))
    for t in ("train", "infer"):
        for m in ("fresh", "reuse"):
            cells.append((t, 64, m, ""))
    cells.append(("train", 64, "reuse", DIAG_TAG))
    return cells


def expected_cells_4159():
    """4159 のセル集合（17 セル）。キーは (task, size, mode, diag)。N=4096 は含めない。"""
    cells = []
    for s in (256, 512, 1024, 2048):
        for m in ("fresh", "reuse"):
            cells.append(("gemm", s, m, ""))
        cells.append(("gemm", s, "reuse", DIAG_TAG))
    for t in ("train", "infer"):
        for m in ("fresh", "reuse"):
            cells.append((t, 64, m, ""))
    cells.append(("train", 64, "reuse", DIAG_TAG))
    return cells


def load(paths, with_diag=False):
    """JSONL 群を (task,size,mode) -> 行リスト（出現順）へ。1 行でも不正なら Undetermined。
    with_diag（4158）ではキーに diag タグを足す。3226 では diag 行を受理しない。"""
    cells = {}
    meta = {}
    for p in paths:
        try:
            fh = open(p, encoding="utf-8")
        except OSError as e:
            raise Undetermined(f"{p}: 開けない ({e})")
        with fh:
            for n, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    o = json.loads(line)
                except ValueError:
                    raise Undetermined(f"{p}:{n}: JSON として解析できない")
                if not isinstance(o, dict):
                    raise Undetermined(f"{p}:{n}: トップレベルが object でない")
                for k in ("framework", "device", "task", "mode", "size"):
                    if k not in o:
                        raise Undetermined(f"{p}:{n}: '{k}' 欠落")
                if o["device"] != "cpu":
                    raise Undetermined(f"{p}:{n}: device != cpu")
                if o["framework"] != "fandhe-ai":
                    raise Undetermined(f"{p}:{n}: framework != fandhe-ai")
                v = o.get("median_s")
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
                    raise Undetermined(f"{p}:{n}: median_s が有限正数でない ({v!r})")
                c = o.get("checksum")
                if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(c):
                    raise Undetermined(f"{p}:{n}: checksum が欠落または非有限 ({c!r})")
                diag = o.get("diag", "")
                if with_diag:
                    if diag not in ("", DIAG_TAG):
                        raise Undetermined(f"{p}:{n}: 未知の diag タグ")
                    key = (o["task"], o["size"], o["mode"], diag)
                else:
                    if diag != "":
                        raise Undetermined(f"{p}:{n}: 診断行（diag）は 3226 の判定に混ぜられない")
                    key = (o["task"], o["size"], o["mode"])
                cells.setdefault(key, []).append(o)
    return cells


def check_gate_log(profile, path, require_excl=False):
    """負荷ゲートログが「正式系列・5/5 round が gate_ok=1・precheck=assert」を示すか検証する。"""
    if path is None:
        raise Undetermined(f"{profile}: 負荷ゲートログ（--{profile}-gate-log）が未指定")
    try:
        with open(path, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
    except OSError as e:
        raise Undetermined(f"{profile}: 負荷ゲートログを開けない ({e})")
    if not lines:
        raise Undetermined(f"{profile}: 負荷ゲートログが空")

    def kv(line):
        return dict(tok.split("=", 1) for tok in line.split() if "=" in tok)

    head = kv(lines[0])
    if head.get("series") != "official":
        raise Undetermined(f"{profile}: 系列が official でない (series={head.get('series')!r})。参考系列は正式判定不可")
    if head.get("precheck") != "assert":
        raise Undetermined(f"{profile}: precheck が assert でない (precheck={head.get('precheck')!r})")
    want = RULE_LOAD_THRESHOLD[profile]
    for key in ("threshold", "rule_threshold"):
        try:
            got = float(head.get(key, ""))
        except ValueError:
            got = None
        if got is None or not math.isfinite(got) or got != want:
            raise Undetermined(
                f"{profile}: 負荷閾値が機体の規則値 {want} と不一致 ({key}={head.get(key)!r})。他機体のログは受理しない")
    if not head.get("threshold"):
        raise Undetermined(f"{profile}: threshold が欠落")
    round_lines = [ln for ln in lines[1:] if ln.startswith("round")]
    names = [ln.split()[0] for ln in round_lines]
    if names != [f"round{i}" for i in range(1, ROUNDS + 1)]:
        raise Undetermined(f"{profile}: round 行が round1〜{ROUNDS} と一致しない ({names})")
    for n, ln in zip(names, round_lines):
        r = kv(ln)
        if require_excl and r.get("compute_procs") != "0":
            raise Undetermined(f"{profile} {n}: GPU compute プロセスが 0 件でない (compute_procs={r.get('compute_procs')!r})")
        if r.get("gate") != "pass" or r.get("gate_ok") != "1" or r.get("threshold") != head.get("threshold"):
            raise Undetermined(
                f"{profile} {n}: 負荷ゲート不通過または閾値不一致 (gate={r.get('gate')!r} gate_ok={r.get('gate_ok')!r})")


def analyse(profile, before_paths, after_paths, rule_set="3226"):
    r4159 = rule_set == "4159"
    r4158 = rule_set == "4158"
    with_diag = r4158 or r4159
    before, after = load(before_paths, with_diag), load(after_paths, with_diag)
    want = set(expected_cells_4159() if r4159 else expected_cells_4158() if r4158 else expected_cells(profile))
    for label, d in (("before", before), ("after", after)):
        if set(d) != want:
            raise Undetermined(
                f"{profile} {label}: セル集合が期待と不一致 "
                f"(余剰={sorted(set(d) - want)} 欠落={sorted(want - set(d))})"
            )
        for k, rows in d.items():
            if len(rows) != ROUNDS:
                raise Undetermined(f"{profile} {label} {k}: {len(rows)} 件（{ROUNDS} 件必須）")
    out = []
    for k in sorted(want):
        b, a = before[k], after[k]
        for f in ("warmup", "iters"):
            if any(r.get(f) != b[0].get(f) for r in b + a):
                raise Undetermined(f"{profile} {k}: {f} が腕・round 間で不一致")
        ratios = [a[i]["median_s"] / b[i]["median_s"] for i in range(ROUNDS)]
        bm = statistics.median(r["median_s"] for r in b)
        am = statistics.median(r["median_s"] for r in a)
        sums = [r["checksum"] for r in b + a]
        out.append({
            "machine": profile, "cell": k, "before": bm, "after": am,
            "ratio": am / bm, "rounds": ratios,
            "all_lt1": all(r < 1.0 for r in ratios),
            "all_gt1": all(r > 1.0 for r in ratios),
            "checksum_exact": all(c == sums[0] for c in sums),
            "treated": (not r4159) and profile == "gb10" and k[0] == "gemm" and k[2] == "reuse"
            and k[1] in TREATED_SIZES and (not r4158 or k[3] == DIAG_TAG),
            # 4159: 判定セル = 診断経路。それ以外は A/A 参照（後退の判定に数えない）。
            "judged": r4159 and k[3] == DIAG_TAG,
            "rmin": min(ratios), "rmax": max(ratios),
        })
    return out


def verdict(rows, machines, record_only=False, single_machine=False):
    """rows は analyse の結果の連結。machines は取り込めた機体集合。"""
    reasons = []
    reject = False
    for r in rows:
        if not r["checksum_exact"]:
            reject = True
            reasons.append(f"{r['machine']} {r['cell']}: checksum 不一致")
        if r["all_gt1"]:
            reject = True
            reasons.append(f"{r['machine']} {r['cell']}: 全 round 後退")
        if r["treated"] and r["ratio"] > 1.0:
            reject = True
            reasons.append(f"{r['machine']} {r['cell']}: 処置セルの中央値 ratio > 1.00")
    if reject:
        if record_only:
            # 負荷ゲート不通過・smoke の系列は正式判定にしない（ADOPT／REJECT に数えない）。
            # 観測された後退は参考情報として reasons に残す。
            return "undetermined", ["参考（record_only のため REJECT に数えない）: " + x for x in reasons]
        return "REJECT", reasons
    ok = ({"gb10"} <= set(machines) if single_machine else {"gb10", "macmini"} <= set(machines)) and not record_only
    if not ok:
        reasons.append("片方の機体が欠落、または record_only のため ADOPT 不可")
    for r in rows:
        if r["treated"] and not (r["ratio"] <= ADOPT_RATIO and r["all_lt1"]):
            ok = False
            reasons.append(f"{r['machine']} {r['cell']}: 処置セルが ADOPT 条件未達")
        if not r["treated"] and r["ratio"] > 1.0:
            ok = False
            reasons.append(f"{r['machine']} {r['cell']}: 非処置セルの中央値 ratio > 1.00（後退）")
    return ("ADOPT", reasons) if ok else ("undetermined", reasons)


def verdict_4159(rows, machines, record_only=False):
    """4159 の判定（macmini 単独・ADOPT なし）。判定セルと A/A 参照セルの区分は analyse の judged。"""
    reasons = []
    regress = False
    aa_anomaly = False
    judged_fail = False
    for r in rows:
        if not r["checksum_exact"]:
            regress = True
            reasons.append(f"{r['machine']} {r['cell']}: checksum 不一致")
        if r["judged"]:
            if r["all_gt1"]:
                regress = True
                reasons.append(f"{r['machine']} {r['cell']}: 判定セルが全 round 後退")
            elif r["ratio"] > 1.0:
                judged_fail = True
                reasons.append(f"{r['machine']} {r['cell']}: 判定セルの中央値 ratio > 1.00（5/5 ではない）")
        elif r["all_gt1"]:
            aa_anomaly = True
            reasons.append(f"{r['machine']} {r['cell']}: A/A 参照セルが全 round 後退（計測系の系統差か到達性の読解誤り）")
    if regress:
        if record_only:
            return "undetermined", ["参考（record_only のため REGRESS に数えない）: " + x for x in reasons]
        return "REGRESS", reasons
    if record_only:
        reasons.append("record_only のため NONREGRESS 不可")
        return "undetermined", reasons
    if "macmini" not in machines:
        reasons.append("macmini の入力が欠落")
        return "undetermined", reasons
    if judged_fail or aa_anomaly:
        return "undetermined", reasons
    return "NONREGRESS", reasons


def render(rows, v, reasons, rule_set="3226"):
    r4159 = rule_set == "4159"
    if r4159:
        lines = ["| 機体 | task | size | mode | 区分 | before 中央値 s | after 中央値 s | ratio | round 別 ratio | round 比 min〜max（NB-RANGE5・記録のみ） | 後退 |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            t, s, m = r["cell"][:3]
            if r["cell"][3]:
                t = f"{t}[{r['cell'][3]}]"
            reg = "5/5 後退" if r["all_gt1"] else ("中央値後退" if r["ratio"] > 1.0 else "-")
            lines.append(
                f"| {r['machine']} | {t} | {s} | {m} | {'判定' if r['judged'] else 'A/A 参照'} | "
                f"{r['before']:.6g} | {r['after']:.6g} | {r['ratio']:.3f} | "
                f"{' '.join(f'{x:.3f}' for x in r['rounds'])} | {r['rmin']:.3f}〜{r['rmax']:.3f} | {reg} |")
        lines.append("")
        sec = {(r["cell"][1], r["cell"][2]): r for r in rows if r["cell"][0] == "gemm" and not r["cell"][3]}
        for s in (256, 512, 1024, 2048):
            if (s, "reuse") in sec and (s, "fresh") in sec:
                f, u = sec[(s, "fresh")], sec[(s, "reuse")]
                lines.append(f"- 副次（判定外）macmini gemm N={s}: reuse/fresh before={u['before']/f['before']:.3f} "
                             f"after={u['after']/f['after']:.3f}")
        lines.append(f"\nverdict: {v}")
        lines += [f"- {x}" for x in reasons]
        return "\n".join(lines)
    lines = ["| 機体 | task | size | mode | 区分 | before 中央値 s | after 中央値 s | ratio | round 別 ratio | 後退 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, s, m = r["cell"][:3]
        if len(r["cell"]) > 3 and r["cell"][3]:
            t = f"{t}[{r['cell'][3]}]"
        reg = "5/5 後退" if r["all_gt1"] else ("中央値後退" if r["ratio"] > 1.0 else "-")
        lines.append(
            f"| {r['machine']} | {t} | {s} | {m} | {'処置' if r['treated'] else '非処置'} | "
            f"{r['before']:.6g} | {r['after']:.6g} | {r['ratio']:.3f} | "
            f"{' '.join(f'{x:.3f}' for x in r['rounds'])} | {reg} |")
    lines.append("")
    # 副次（判定外）: 解放後の reuse/fresh 比（GB10 gemm）
    sec = {(r["cell"][1], r["cell"][2]): r for r in rows
           if r["machine"] == "gb10" and r["cell"][0] == "gemm" and (len(r["cell"]) < 4 or not r["cell"][3])}
    for s in TREATED_SIZES:
        if (s, "reuse") in sec and (s, "fresh") in sec:
            f, u = sec[(s, "fresh")], sec[(s, "reuse")]
            lines.append(f"- 副次（判定外）GB10 gemm N={s}: reuse/fresh before={u['before']/f['before']:.3f} "
                         f"after={u['after']/f['after']:.3f}")
    lines.append(f"\nverdict: {v}")
    lines += [f"- {x}" for x in reasons]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    for m in PROFILES:
        ap.add_argument(f"--{m}-before", nargs="+", default=None)
        ap.add_argument(f"--{m}-after", nargs="+", default=None)
        ap.add_argument(f"--{m}-gate-log", default=None)
    ap.add_argument("--record-only", action="store_true")
    ap.add_argument("--rule-set", choices=RULE_SETS, default="3226",
                    help="3226（既定・両機体）／4158（GB10 単独・診断経路セルを含む）／4159（macmini 単独・診断経路セルを含む）")
    ap.add_argument("--require-both", action="store_true",
                    help="（--combine 相当）片方の機体が欠けたら判定不能")
    a = ap.parse_args(argv)
    rows, machines = [], []
    r4158 = a.rule_set == "4158"
    r4159 = a.rule_set == "4159"
    try:
        if r4158 and (a.macmini_before is not None or a.macmini_after is not None):
            raise Undetermined("4158 は GB10 単独の判定。macmini 入力は受理しない")
        if r4159 and (a.gb10_before is not None or a.gb10_after is not None):
            raise Undetermined("4159 は macmini 単独の判定。gb10 入力は受理しない")
        for m in PROFILES:
            b, af = getattr(a, f"{m}_before"), getattr(a, f"{m}_after")
            if (b is None) != (af is None):
                raise Undetermined(f"{m}: before/after の片方だけが指定された")
            if b is not None:
                check_gate_log(m, getattr(a, f"{m}_gate_log"), require_excl=r4158)
                rows += analyse(m, b, af, a.rule_set)
                machines.append(m)
        if not machines:
            raise Undetermined("入力なし")
        if r4159:
            v, reasons = verdict_4159(rows, machines, a.record_only)
        else:
            v, reasons = verdict(rows, machines, a.record_only, single_machine=r4158)
        if a.require_both and len(machines) < 2:
            v = "undetermined"
    except Undetermined as e:
        print(f"verdict: undetermined\n- {e}")
        return 2
    print(render(rows, v, reasons, a.rule_set))
    return {"ADOPT": 0, "REJECT": 1, "NONREGRESS": 0, "REGRESS": 1}.get(v, 2)


if __name__ == "__main__":
    sys.exit(main())
