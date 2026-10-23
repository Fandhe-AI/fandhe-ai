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
- それ以外は undetermined。`--record-only` 指定（負荷ゲート不通過の系列）は ADOPT も REJECT も出さず undetermined とし、後退は参考情報として残す。
"""
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


def load(paths):
    """JSONL 群を (task,size,mode) -> 行リスト（出現順）へ。1 行でも不正なら Undetermined。"""
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
                cells.setdefault((o["task"], o["size"], o["mode"]), []).append(o)
    return cells


def analyse(profile, before_paths, after_paths):
    before, after = load(before_paths), load(after_paths)
    want = set(expected_cells(profile))
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
            "treated": profile == "gb10" and k[0] == "gemm" and k[2] == "reuse" and k[1] in TREATED_SIZES,
        })
    return out


def verdict(rows, machines, record_only=False):
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
    ok = {"gb10", "macmini"} <= set(machines) and not record_only
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


def render(rows, v, reasons):
    lines = ["| 機体 | task | size | mode | 区分 | before 中央値 s | after 中央値 s | ratio | round 別 ratio | 後退 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        t, s, m = r["cell"]
        reg = "5/5 後退" if r["all_gt1"] else ("中央値後退" if r["ratio"] > 1.0 else "-")
        lines.append(
            f"| {r['machine']} | {t} | {s} | {m} | {'処置' if r['treated'] else '非処置'} | "
            f"{r['before']:.6g} | {r['after']:.6g} | {r['ratio']:.3f} | "
            f"{' '.join(f'{x:.3f}' for x in r['rounds'])} | {reg} |")
    lines.append("")
    # 副次（判定外）: 解放後の reuse/fresh 比（GB10 gemm）
    sec = {(r["cell"][1], r["cell"][2]): r for r in rows if r["machine"] == "gb10" and r["cell"][0] == "gemm"}
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
    ap.add_argument("--record-only", action="store_true")
    ap.add_argument("--require-both", action="store_true",
                    help="（--combine 相当）片方の機体が欠けたら判定不能")
    a = ap.parse_args(argv)
    rows, machines = [], []
    try:
        for m in PROFILES:
            b, af = getattr(a, f"{m}_before"), getattr(a, f"{m}_after")
            if (b is None) != (af is None):
                raise Undetermined(f"{m}: before/after の片方だけが指定された")
            if b is not None:
                rows += analyse(m, b, af)
                machines.append(m)
        if not machines:
            raise Undetermined("入力なし")
        v, reasons = verdict(rows, machines, a.record_only)
        if a.require_both and len(machines) < 2:
            v = "undetermined"
    except Undetermined as e:
        print(f"verdict: undetermined\n- {e}")
        return 2
    print(render(rows, v, reasons))
    return {"ADOPT": 0, "REJECT": 1}.get(v, 2)


if __name__ == "__main__":
    sys.exit(main())
