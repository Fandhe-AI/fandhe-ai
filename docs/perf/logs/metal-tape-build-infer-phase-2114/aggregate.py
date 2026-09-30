#!/usr/bin/env python3
"""イシュー #2114: 独立 5 run の JSONL を RULE.txt の規則で集計する（標準ライブラリのみ）。

使い方: aggregate.py <dir>   （dir は run1.jsonl..run5.jsonl と load_gate_status.txt を持つ）
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
    "fresh_off": ["tape_build", "leaf_register", "forward", "to_tensor", "host_copy", "checksum", "iter_total"],
    "fresh_on": ["tape_build", "leaf_register", "forward", "to_tensor", "host_copy", "checksum", "iter_total"],
    "reuse_off": ["predict_resident", "host_copy", "checksum", "iter_total"],
    "reuse_on": ["predict_resident", "host_copy", "checksum", "iter_total"],
    "reuse_decomposed_off": ["provider_select", "tape_new", "tape_build", "snapshot", "chain",
                             "host_copy", "checksum", "iter_total"],
    "reuse_decomposed_on": ["provider_select", "tape_new", "tape_build", "snapshot", "chain",
                            "host_copy", "checksum", "iter_total"],
    "tape_build_micro": ["probe_gpu_core_count", "provider_select", "tape_new", "tape_for"],
}
# counters 行を持つ arm（micro は持たない）
COUNTER_ARMS = [a for a in EXPECTED if a != "tape_build_micro"]
COUNTER_KEYS = ["encode_per_iter", "command_buffers_per_iter", "wait_per_iter",
                "verify_probe_per_iter", "verify_hit_per_iter", "uploads_per_iter",
                "upload_bytes_per_iter", "downloads_per_iter", "download_bytes_per_iter"]
EXPECTED_N = 80  # 計測 20 反復 x 4 ラウンド（RULE.txt。テスト側 ROUNDS x ITERS_PER_ROUND）
GATE_FILE = "load_gate_status.txt"
H1_THRESHOLD = 0.5  # RULE.txt 仮説判定（事後に変えない）
H2_LIMIT_S = 1e-6


class AggError(Exception):
    pass


def load_run(path):
    cells = {}
    counters = {}
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    arm, phase, n = rec["arm"], rec["phase"], rec["n"]
                except (ValueError, KeyError, TypeError) as e:
                    raise AggError(f"{path}:{i}: 不正な行 ({e})")
                if arm not in EXPECTED:
                    raise AggError(f"{path}:{i}: 想定外の arm {arm!r}")
                if isinstance(n, bool) or n != EXPECTED_N:
                    raise AggError(f"{path}:{i}: 計測回数 n={n!r} が {EXPECTED_N} でない ({arm}/{phase})")
                if phase == "counters":
                    if arm not in COUNTER_ARMS or arm in counters:
                        raise AggError(f"{path}:{i}: 想定外または重複した counters 行 ({arm})")
                    vals = {}
                    for k in COUNTER_KEYS:
                        try:
                            v = float(rec[k])
                        except (KeyError, TypeError, ValueError):
                            raise AggError(f"{path}:{i}: counters の {k} が不正 ({arm})")
                        if not math.isfinite(v) or v < 0.0:
                            raise AggError(f"{path}:{i}: counters の {k} が有限の非負値でない ({arm})")
                        vals[k] = v
                    counters[arm] = vals
                    continue
                if phase not in EXPECTED[arm]:
                    raise AggError(f"{path}:{i}: 想定外のセル {(arm, phase)}")
                try:
                    rec["median_s"] = float(rec["median_s"])
                except (KeyError, TypeError, ValueError) as e:
                    raise AggError(f"{path}:{i}: median_s が不正 ({e})")
                if not math.isfinite(rec["median_s"]) or rec["median_s"] <= 0.0:
                    raise AggError(f"{path}:{i}: median_s が有限の正値でない {(arm, phase)} ({rec['median_s']})")
                if (arm, phase) in cells:
                    raise AggError(f"{path}:{i}: セル重複 {(arm, phase)}")
                cells[(arm, phase)] = rec
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    for arm, phases in EXPECTED.items():
        for ph in phases:
            if (arm, ph) not in cells:
                raise AggError(f"{path}: セル欠落 {arm}/{ph}")
    for arm in COUNTER_ARMS:
        if arm not in counters:
            raise AggError(f"{path}: counters 欠落 {arm}")
    return {"cells": cells, "counters": counters}


def load_gate(d):
    """load_gate_status.txt を読み検証する（欠落・不正は fail-closed）。m4max は record_only 固定。"""
    path = os.path.join(d, GATE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    seen = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) != 2 or parts[1] != "gate=record_only" or parts[0] in seen:
            raise AggError(f"{path}: 不正または重複した行 {ln!r}")
        seen[parts[0]] = parts[1]
    if set(seen) != {f"run{n}" for n in range(1, RUNS + 1)}:
        raise AggError(f"{path}: run1..run{RUNS} の全行が必要 ({sorted(seen)})")


def med(runs, arm, ph):
    vals = [r["cells"][(arm, ph)]["median_s"] for r in runs]
    return statistics.median(vals), min(vals), max(vals)


def check_checksums(runs):
    def sums(arm):
        s = {r["cells"][(arm, "iter_total")].get("checksum_bits") for r in runs}
        if None in s or len(s) != 1:
            raise AggError(f"checksum が run 間で不一致または欠落: {arm} {sorted(map(str, s))}")
        return next(iter(s))
    got = {arm: sums(arm) for arm in COUNTER_ARMS}
    for kind in ("fresh", "reuse", "reuse_decomposed"):
        if got[f"{kind}_off"] != got[f"{kind}_on"]:
            raise AggError(f"{kind}: キャッシュ off と on の checksum が不一致")
    for x in ("off", "on"):
        if got[f"reuse_decomposed_{x}"] != got[f"reuse_{x}"]:
            raise AggError(f"reuse_decomposed_{x} と reuse_{x} の checksum が不一致")
    return got["fresh_off"] == got["reuse_off"]


def us(x):
    return f"{x * 1e6:.1f}"


def ratio_line(name, on, off):
    r = on / off
    return f"{name} = {r:.4f}（非後退基準 <= 1.00: {'満たす' if r <= 1.0 else '後退'}。記録のみ）"


def aggregate(runs):
    fresh_eq_reuse = check_checksums(runs)
    out = ["| arm | phase | median(us) | min-max(us) |", "|---|---|---|---|"]
    for arm, phases in EXPECTED.items():
        for ph in phases:
            m, lo, hi = med(runs, arm, ph)
            out.append(f"| {arm} | {ph} | {us(m)} | {us(lo)}-{us(hi)} |")
    out.append("")
    out.append("| arm | counter | run 間 median | run 間 min-max |")
    out.append("|---|---|---|---|")
    for arm in COUNTER_ARMS:
        for k in COUNTER_KEYS:
            vals = [r["counters"][arm][k] for r in runs]
            out.append(f"| {arm} | {k} | {statistics.median(vals):.3f} | {min(vals):.3f}-{max(vals):.3f} |")
    out.append("")
    out.append(f"checksum: run 間一致 OK / off==on OK / reuse_decomposed==reuse OK / fresh==reuse: {fresh_eq_reuse}")
    out.append(ratio_line("ratio_tape_build_fresh (on/off)", med(runs, "fresh_on", "tape_build")[0],
                          med(runs, "fresh_off", "tape_build")[0]))
    out.append(ratio_line("ratio_tape_build_decomposed (on/off)", med(runs, "reuse_decomposed_on", "tape_build")[0],
                          med(runs, "reuse_decomposed_off", "tape_build")[0]))
    out.append(ratio_line("ratio_iter_fresh (on/off)", med(runs, "fresh_on", "iter_total")[0],
                          med(runs, "fresh_off", "iter_total")[0]))
    out.append(ratio_line("ratio_iter_reuse (on/off)", med(runs, "reuse_on", "iter_total")[0],
                          med(runs, "reuse_off", "iter_total")[0]))
    out.append("")
    ps = med(runs, "tape_build_micro", "provider_select")[0]
    tb = med(runs, "fresh_off", "tape_build")[0]
    share = ps / tb
    out.append(f"H1: micro.provider_select / fresh_off.tape_build = {share:.2f} -> "
               f"{'支持' if share >= H1_THRESHOLD else '未確定'}（閾値 {H1_THRESHOLD}）")
    pg = med(runs, "tape_build_micro", "probe_gpu_core_count")[0]
    out.append(f"H1a（補助）: probe_gpu_core_count / provider_select = {pg / ps:.2f}"
               f"（残り {1 - pg / ps:.2f} が MTLCopyAllDevices・name 等の推定）")
    tn = med(runs, "tape_build_micro", "tape_new")[0]
    out.append(f"H2: micro.tape_new = {us(tn)} us -> {'支持' if tn < H2_LIMIT_S else '未確定'}（閾値 1 us）")
    exp = {"encode_per_iter": 2.0, "command_buffers_per_iter": 1.0, "wait_per_iter": 1.0}
    ok = all(abs(r["counters"]["reuse_off"][k] - v) < 1e-9 for r in runs for k, v in exp.items())
    out.append(f"H3: reuse_off の encode=2・command_buffers=1・wait=1（全 run） -> {'支持' if ok else '未確定'}")
    out.append("H4: (c) の wall - GPU busy は手動転記で判定（集計対象外）")
    return "\n".join(out)


def synth_run(scale=1.0, chk="00ff"):
    cells = {}
    counters = {}
    for arm, phases in EXPECTED.items():
        for ph in phases:
            t = 1e-5 * scale
            if arm == "tape_build_micro" and ph == "provider_select":
                t = 9e-6
            if arm == "tape_build_micro" and ph == "tape_new":
                t = 5e-7
            if ph == "tape_build" and arm.startswith("fresh_off"):
                t = 1.5e-5
            c = chk if ph == "iter_total" else None
            cells[(arm, ph)] = {"arm": arm, "phase": ph, "median_s": t, "n": EXPECTED_N, "checksum_bits": c}
    for arm in COUNTER_ARMS:
        counters[arm] = {k: 0.0 for k in COUNTER_KEYS}
        counters[arm].update(encode_per_iter=2.0, command_buffers_per_iter=1.0, wait_per_iter=1.0)
    return {"cells": cells, "counters": counters}


def write_run(path, run):
    with open(path, "w", encoding="utf-8") as f:
        for (arm, ph), rec in run["cells"].items():
            f.write(json.dumps(rec) + "\n")
        for arm, vals in run["counters"].items():
            f.write(json.dumps({"arm": arm, "phase": "counters", "n": EXPECTED_N, **vals}) + "\n")


def write_gate(d, statuses=None):
    with open(os.path.join(d, GATE_FILE), "w", encoding="utf-8") as f:
        for n in range(1, RUNS + 1):
            f.write(f"run{n} gate={(statuses or {}).get(n, 'record_only')}\n")


def expect_fail(d, label):
    try:
        run_dir(d)
    except AggError:
        return
    raise SystemExit(f"self-test 失敗: {label} を検出できない")


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        def reset(mutate=None):
            for n in range(1, RUNS + 1):
                run = synth_run()
                if mutate and n == 1:
                    mutate(run)
                write_run(os.path.join(d, f"run{n}.jsonl"), run)
            write_gate(d)

        reset()
        text = run_dir(d)
        assert "ratio_tape_build_fresh" in text and "H1: " in text and "支持" in text, text
        assert "H3" in text and "H3: reuse_off の encode=2・command_buffers=1・wait=1（全 run） -> 支持" in text, text

        def set_cell(arm, ph, key, val):
            return lambda run: run["cells"][(arm, ph)].__setitem__(key, val)

        for label, mut in (
            ("n=1", set_cell("fresh_off", "forward", "n", 1)),
            ("median NaN", set_cell("fresh_off", "forward", "median_s", float("nan"))),
            ("median Inf", set_cell("fresh_off", "forward", "median_s", float("inf"))),
            ("median 負", set_cell("fresh_off", "forward", "median_s", -1e-5)),
            ("median 0", set_cell("fresh_off", "forward", "median_s", 0.0)),
            ("checksum run 間不一致", set_cell("reuse_off", "iter_total", "checksum_bits", "dead")),
            ("checksum off/on 不一致", lambda run: [
                run["cells"][("reuse_on", "iter_total")].__setitem__("checksum_bits", "beef"),
                run["cells"][("reuse_decomposed_on", "iter_total")].__setitem__("checksum_bits", "beef")]),
            ("decomposed と reuse の checksum 不一致",
             set_cell("reuse_decomposed_off", "iter_total", "checksum_bits", "cafe")),
            ("counters 欠落", lambda run: run["counters"].pop("reuse_off")),
            ("counters 負値", lambda run: run["counters"]["reuse_off"].__setitem__("wait_per_iter", -1.0)),
        ):
            reset(mut)
            expect_fail(d, label)
        # セル欠落・不正 JSON・ゲート状態不正
        reset(lambda run: run["cells"].pop(("tape_build_micro", "tape_for")))
        expect_fail(d, "セル欠落")
        reset()
        with open(os.path.join(d, "run3.jsonl"), "w", encoding="utf-8") as f:
            f.write("{not json\n")
        expect_fail(d, "不正 JSON")
        reset()
        write_gate(d, {2: "unpassed"})
        expect_fail(d, "ゲート状態不正")
        reset()
        os.remove(os.path.join(d, GATE_FILE))
        expect_fail(d, "ゲート状態ファイル欠落")
        # H3 不成立は例外ではなく「未確定」として出力される
        reset(lambda run: run["counters"]["reuse_off"].__setitem__("wait_per_iter", 2.0))
        assert "H3: reuse_off の encode=2・command_buffers=1・wait=1（全 run） -> 未確定" in run_dir(d)
    print("self-test OK")


def run_dir(d):
    runs = [load_run(os.path.join(d, f"run{n}.jsonl")) for n in range(1, RUNS + 1)]
    load_gate(d)
    return aggregate(runs)


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
