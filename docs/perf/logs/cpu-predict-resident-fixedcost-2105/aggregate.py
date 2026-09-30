#!/usr/bin/env python3
"""イシュー #2105: 独立 5 run の JSONL を RULE.txt の規則で集計する（標準ライブラリのみ）。

使い方: aggregate.py <dir>   （dir は run1.jsonl..run5.jsonl を持つ）
        aggregate.py --self-test
欠落セル・不正 JSON・checksum 不一致は fail-closed（exit 1）。判定閾値は
RULE.txt（実測前に固定）が正で、本スクリプトは変更しない。
"""
import json
import math
import os
import statistics
import sys

RUNS = 5
EXPECTED = {
    "fresh": ["predict", "host_copy", "checksum", "iter_total"],
    "reuse": ["predict", "host_copy", "checksum", "iter_total"],
    "reuse_decomposed": [
        "tape_new", "snapshot", "tape_build", "upload",
        "l1_linear_forward_device", "l2_linear_forward_device",
        "forward_resident", "readout", "host_copy", "checksum", "iter_total",
        "chain_public",
    ],
    "fresh_layers": ["l1_linear_relu_fused", "l2_linear_gemm_add"],
    "ablation": ["l1_host_unfused_gemm_add_relu", "input_to_vec_copy"],
}
EXPECTED_N = 80  # 計測 20 反復 x 4 ラウンド（RULE.txt 実行方法。テスト側 ROUNDS x ITERS_PER_ROUND）
GATE_FILE = "load_gate_status.txt"  # orchestrate.sh が run ごとに 1 行 `run<N> gate=<pass|unpassed|record_only>` を書く
ATTRIB_THRESHOLD = 0.5  # RULE.txt 帰属規則（事後に変えない）


class AggError(Exception):
    pass


def load_run(path):
    cells = {}
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["arm"], rec["phase"])
                    rec["median_s"] = float(rec["median_s"])
                    n = rec["n"]
                except (ValueError, KeyError, TypeError) as e:
                    raise AggError(f"{path}:{i}: 不正な行 ({e})")
                if key[1] not in EXPECTED.get(key[0], []):
                    raise AggError(f"{path}:{i}: 想定外のセル {key}")
                if not math.isfinite(rec["median_s"]) or rec["median_s"] <= 0.0:
                    raise AggError(f"{path}:{i}: median_s が有限の正値でない {key} ({rec['median_s']})")
                if isinstance(n, bool) or n != EXPECTED_N:
                    raise AggError(f"{path}:{i}: 計測回数 n={n!r} が {EXPECTED_N} でない {key}")
                if key in cells:
                    raise AggError(f"{path}:{i}: セル重複 {key}")
                cells[key] = rec
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    for arm, phases in EXPECTED.items():
        for ph in phases:
            if (arm, ph) not in cells:
                raise AggError(f"{path}: セル欠落 {arm}/{ph}")
    return cells


def load_gate(d):
    """load_gate_status.txt を読み、負荷ゲート未通過の run 名一覧を返す（欠落・不正は fail-closed）。"""
    path = os.path.join(d, GATE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    seen = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) != 2 or not parts[1].startswith("gate="):
            raise AggError(f"{path}: 不正な行 {ln!r}")
        status = parts[1][5:]
        if status not in ("pass", "unpassed", "record_only") or parts[0] in seen:
            raise AggError(f"{path}: 不正または重複した行 {ln!r}")
        seen[parts[0]] = status
    if set(seen) != {f"run{n}" for n in range(1, RUNS + 1)}:
        raise AggError(f"{path}: run1..run{RUNS} の全行が必要 ({sorted(seen)})")
    return sorted(r for r, st in seen.items() if st == "unpassed")


def med(runs, arm, ph):
    vals = [r[(arm, ph)]["median_s"] for r in runs]
    return statistics.median(vals), min(vals), max(vals)


def check_checksums(runs):
    for arm in ("fresh", "reuse", "reuse_decomposed"):
        sums = {r[(arm, "iter_total")].get("checksum_bits") for r in runs}
        if None in sums or len(sums) != 1:
            raise AggError(f"checksum が run 間で不一致または欠落: {arm} {sorted(map(str, sums))}")
    reuse = runs[0][("reuse", "iter_total")]["checksum_bits"]
    dec = runs[0][("reuse_decomposed", "iter_total")]["checksum_bits"]
    if reuse != dec:
        raise AggError("reuse_decomposed と reuse の checksum が不一致")
    fresh = runs[0][("fresh", "iter_total")]["checksum_bits"]
    return fresh == reuse


def dist_cols(runs, arm, ph):
    """H5（順序・ばらつき）確認用の分布指標（q1_s・q3_s・min_s・max_s）を run 間 median/極値で要約する。"""
    def col(k):
        return [r[(arm, ph)].get(k) for r in runs]
    q1, q3, mn, mx = col("q1_s"), col("q3_s"), col("min_s"), col("max_s")
    if any(v is None for v in q1 + q3 + mn + mx):
        return "null | null"
    return (f"{us(statistics.median(q1))}/{us(statistics.median(q3))} | "
            f"{us(min(mn))}-{us(max(mx))}")


def us(x):
    return f"{x * 1e6:.1f}"


def aggregate(runs, unpassed=()):
    fresh_eq_reuse = check_checksums(runs)
    out = []
    if unpassed:
        out.append(f"【参考扱い】GB10 負荷ゲート未通過の run: {', '.join(unpassed)}（RULE.txt ゲート節。通常判定として扱わない）")
        out.append("")
    out.append("| arm | phase | median(us) | min-max(us) | q1/q3 median(us) | run 内 min-max(us) | minflt/iter(median) |")
    out.append("|---|---|---|---|---|---|---|")
    for arm, phases in EXPECTED.items():
        for ph in phases:
            m, lo, hi = med(runs, arm, ph)
            fl = [r[(arm, ph)].get("minflt_delta_per_iter") for r in runs]
            fl_s = "null" if any(v is None for v in fl) else f"{statistics.median(fl):.3f}"
            dist = dist_cols(runs, arm, ph)
            out.append(f"| {arm} | {ph} | {us(m)} | {us(lo)}-{us(hi)} | {dist} | {fl_s} |")
    f_tot = med(runs, "fresh", "iter_total")[0]
    r_tot = med(runs, "reuse", "iter_total")[0]
    d_tot = med(runs, "reuse_decomposed", "iter_total")[0]
    ratio = r_tot / f_tot
    diff = r_tot - f_tot
    out.append("")
    out.append(f"checksum: run 間一致 OK / reuse_decomposed==reuse OK / fresh==reuse: {fresh_eq_reuse}")
    out.append(f"ratio = reuse/fresh iter_total = {ratio:.4f}（非後退基準 <= 1.00: {'満たす' if ratio <= 1.0 else '逆転が再現'}）")
    out.append(f"diff = {us(diff)} us")
    if diff <= 0:
        out.append("帰属: 逆転非再現のため帰属を行わない")
        return "\n".join(out)
    # 分解が省略する公開経路の検証・tracked 呼び出し分（chain_public は
    # iter_total の外側で実測した predict_device_chain 全体）。
    chain_extra = (
        med(runs, "reuse_decomposed", "chain_public")[0]
        - med(runs, "reuse_decomposed", "forward_resident")[0]
        - med(runs, "reuse_decomposed", "readout")[0]
    )
    seg = {
        "H1 カーネル差(L1)": med(runs, "reuse_decomposed", "l1_linear_forward_device")[0]
        - med(runs, "fresh_layers", "l1_linear_relu_fused")[0],
        "L2 差": med(runs, "reuse_decomposed", "l2_linear_forward_device")[0]
        - med(runs, "fresh_layers", "l2_linear_gemm_add")[0],
        "H2 コピー/確保(upload+readout)": med(runs, "reuse_decomposed", "upload")[0]
        + med(runs, "reuse_decomposed", "readout")[0],
        "H4 tape 固定費(tape_build+残差-H6)": med(runs, "reuse_decomposed", "tape_build")[0]
        + (r_tot - d_tot)
        - chain_extra,
        "H6 chain 検証・tracked 差": chain_extra,
    }
    out.append("")
    out.append("| 区間 | 差分(us) | diff 比 | 判定 |")
    out.append("|---|---|---|---|")
    supported = []
    for name, v in seg.items():
        share = v / diff
        ok = share >= ATTRIB_THRESHOLD
        if ok:
            supported.append(name)
        out.append(f"| {name} | {us(v)} | {share:.2f} | {'支持' if ok else '-'} |")
    out.append("")
    out.append("帰属: " + (", ".join(supported) if supported else "未確定（どの区間も 50% 未満）"))
    return "\n".join(out)


def synth_line(arm, ph, med_s, chk=None):
    return {"arm": arm, "phase": ph, "median_s": med_s, "n": EXPECTED_N, "checksum_bits": chk,
            "q1_s": med_s * 0.9, "q3_s": med_s * 1.1, "min_s": med_s * 0.8, "max_s": med_s * 1.3,
            "minflt_delta_per_iter": None}


def synth_cells(scale=1.0):
    lines = []
    for arm, phases in EXPECTED.items():
        for ph in phases:
            t = 1e-4 * scale
            if ph == "iter_total":
                t = {"fresh": 1.0e-4, "reuse": 1.2e-4, "reuse_decomposed": 1.19e-4}[arm]
            chk = "00ff" if ph == "iter_total" and arm != "fresh" else ("00ff" if ph == "iter_total" else None)
            lines.append(synth_line(arm, ph, t, chk))
    return lines


def write_gate(d, statuses):
    with open(os.path.join(d, GATE_FILE), "w", encoding="utf-8") as f:
        for n, st in enumerate(statuses, 1):
            f.write(f"run{n} gate={st}\n")


def expect_fail(d, label):
    try:
        run_dir(d)
    except AggError:
        return
    raise SystemExit(f"self-test 失敗: {label} を検出できない")


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for n in range(1, RUNS + 1):
            with open(os.path.join(d, f"run{n}.jsonl"), "w", encoding="utf-8") as f:
                for rec in synth_cells():
                    f.write(json.dumps(rec) + "\n")
        write_gate(d, ["pass"] * RUNS)
        text = run_dir(d)
        assert "ratio" in text and "帰属" in text and "参考扱い" not in text, text
        assert "q1/q3" in text and "80.0-130.0" in text, text
        # 負荷ゲート未通過は参考扱いとして出力へ反映される
        write_gate(d, ["pass", "unpassed", "pass", "pass", "pass"])
        text = run_dir(d)
        assert "【参考扱い】" in text and "run2" in text, text
        os.remove(os.path.join(d, GATE_FILE))
        expect_fail(d, "ゲート状態ファイル欠落")
        write_gate(d, ["pass"] * RUNS)
        # n 不足・NaN・Inf・負値・ゼロ
        p = os.path.join(d, "run1.jsonl")
        orig = open(p, encoding="utf-8").read()
        for label, mut in (
            ("n=1", lambda r: r.__setitem__("n", 1)),
            ("median NaN", lambda r: r.__setitem__("median_s", float("nan"))),
            ("median Inf", lambda r: r.__setitem__("median_s", float("inf"))),
            ("median 負", lambda r: r.__setitem__("median_s", -1e-4)),
            ("median 0", lambda r: r.__setitem__("median_s", 0.0)),
        ):
            with open(p, "w", encoding="utf-8") as f:
                for i, ln in enumerate(orig.splitlines()):
                    rec = json.loads(ln)
                    if i == 0:
                        mut(rec)
                    f.write(json.dumps(rec) + "\n")
            expect_fail(d, label)
        open(p, "w", encoding="utf-8").write(orig)
        # 欠落セル
        p = os.path.join(d, "run3.jsonl")
        lines = open(p, encoding="utf-8").read().splitlines()
        open(p, "w", encoding="utf-8").write("\n".join(lines[:-1]) + "\n")
        try:
            run_dir(d)
            raise SystemExit("self-test 失敗: 欠落を検出できない")
        except AggError:
            pass
        # 不正 JSON
        open(p, "w", encoding="utf-8").write("{not json\n")
        try:
            run_dir(d)
            raise SystemExit("self-test 失敗: 不正 JSON を検出できない")
        except AggError:
            pass
        # checksum 不一致
        with open(p, "w", encoding="utf-8") as f:
            for rec in synth_cells():
                if (rec["arm"], rec["phase"]) == ("reuse", "iter_total"):
                    rec["checksum_bits"] = "dead"
                f.write(json.dumps(rec) + "\n")
        try:
            run_dir(d)
            raise SystemExit("self-test 失敗: checksum 不一致を検出できない")
        except AggError:
            pass
    print("self-test OK")


def run_dir(d):
    runs = [load_run(os.path.join(d, f"run{n}.jsonl")) for n in range(1, RUNS + 1)]
    return aggregate(runs, load_gate(d))


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        print(run_dir(argv[1]))
    except AggError as e:
        print(f"集計失敗（fail-closed）: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
