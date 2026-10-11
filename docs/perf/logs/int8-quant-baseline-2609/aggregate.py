#!/usr/bin/env python3
"""イシュー #2997: int8 格上げ条件 (c) 実測ログの厳格な集計（python3 標準ライブラリのみ）。

役割: orchestrate_cuda.sh / orchestrate_mac.sh が回収した <arm>-run{1..5}.json から、
RULE.txt R1（5 run の median_secs の中央値・Q1/Q3・対 PyTorch 比）どおりに Markdown 表を出す。
README.md §2.1 / §2.3 へはこの出力を手で直さず転記する。

使い方:
  python3 -I aggregate.py <ログdir> --arms prefusion,fused,pytorch [--parity] [--baseline pytorch]
  python3 -I aggregate.py --self-test

厳格方針（寛容パース禁止）: JSON 欠損・パース不能・run 欠損・非有限値・負値・
warmup/iters < 20 はすべて例外（exit 1）。parity ログに pass マーカーが無ければ fail と記録する。
"""
import json
import math
import os
import re
import sys
import tempfile

RUNS = (1, 2, 3, 4, 5)
MIN_ITERS = 20


def pick(sorted_vals, p):
    """bench_harness::stats::median_q1_q3 と同じ: idx = round-half-away(p*(n-1))。"""
    n = len(sorted_vals)
    idx = min(int(math.floor(p * (n - 1) + 0.5)), n - 1)
    return sorted_vals[idx]


def median_q1_q3(vals):
    s = sorted(vals)
    return pick(s, 0.5), pick(s, 0.25), pick(s, 0.75)


def load_run(d, arm, run):
    path = os.path.join(d, "%s-run%d.json" % (arm, run))
    if not os.path.isfile(path):
        raise ValueError("run 欠損: %s" % path)
    # 異種プロセス混入で無効化された run は集計に含めない（RULE.txt DISCARD_CUDA）。
    occ = os.path.join(d, "gpu-occupancy-%s-run%d.txt" % (arm, run))
    if os.path.isfile(occ):
        with open(occ, encoding="utf-8", errors="replace") as f:
            if re.search(r"^RUN_INVALID", f.read(), re.M):
                raise ValueError("無効 run（RUN_INVALID）: %s" % occ)
    with open(path, encoding="utf-8") as f:
        try:
            j = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError("JSON 不正: %s: %s" % (path, e))
    for k in ("median_secs", "q1_secs", "q3_secs"):
        v = j.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("%s: %s が数値でない" % (path, k))
        if not math.isfinite(v) or v < 0:
            raise ValueError("%s: %s が有限の非負値でない: %r" % (path, k, v))
    for k in ("warmup", "iters"):
        v = j.get(k)
        if not isinstance(v, int) or isinstance(v, bool) or v < MIN_ITERS:
            raise ValueError("%s: %s=%r が %d 未満または不正" % (path, k, v, MIN_ITERS))
    return j["median_secs"]


def arm_stats(d, arm):
    per_run = {r: load_run(d, arm, r) for r in RUNS}
    med, q1, q3 = median_q1_q3(list(per_run.values()))
    return per_run, med, q1, q3


def parity_results(d):
    out = {}
    for r in RUNS:
        path = os.path.join(d, "parity-run%d.log" % r)
        if not os.path.isfile(path):
            raise ValueError("parity ログ欠損: %s" % path)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        ok = re.search(r"^test result: ok\. 1 passed; 0 failed", text, re.M) is not None
        bad = re.search(r"FAILED|panicked", text) is not None
        out[r] = "pass" if (ok and not bad) else "FAIL"
    return out


def ms(x):
    return "%.3f" % (x * 1000.0)


def render(d, arms, baseline, parity):
    stats = {a: arm_stats(d, a) for a in arms}
    lines = ["| 腕 | " + " | ".join("run%d" % r for r in RUNS)
             + " | 中央値 | Q1 | Q3 | 対 PyTorch 比 |",
             "|---|" + "---|" * (len(RUNS) + 4)]
    for a in arms:
        per_run, med, q1, q3 = stats[a]
        if baseline is None:
            ratio = "-"
        elif a == baseline:
            ratio = "1.00（基準）"
        else:
            ratio = "%.2f" % (med / stats[baseline][1])
        lines.append("| %s | " % a + " | ".join(ms(per_run[r]) for r in RUNS)
                     + " | %s | %s | %s | %s |" % (ms(med), ms(q1), ms(q3), ratio))
    lines.append("")
    lines.append("単位: ms（各 run の median_secs を 1000 倍。中央値・Q1・Q3 は 5 run の値から算出）。"
                 "対 PyTorch 比は Rust 腕の中央値 / PyTorch 腕の中央値（1 より大きければ遅い）。")
    if parity:
        res = parity_results(d)
        lines.append("")
        lines.append("parity（run1..5）: " + ", ".join("run%d=%s" % (r, res[r]) for r in RUNS))
    return "\n".join(lines)


def self_test():
    def write(d, arm, run, med, **kw):
        j = {"median_secs": med, "q1_secs": med, "q3_secs": med, "warmup": 20, "iters": 20}
        j.update(kw)
        with open(os.path.join(d, "%s-run%d.json" % (arm, run)), "w") as f:
            json.dump(j, f)

    def must_fail(fn, what):
        try:
            fn()
        except ValueError:
            return
        raise AssertionError("拒否されるべきケースが通った: " + what)

    with tempfile.TemporaryDirectory() as d:
        for r in RUNS:
            write(d, "a", r, 0.010 * r)
            write(d, "pytorch", r, 0.005)
        out = render(d, ["a", "pytorch"], "pytorch", False)
        assert "30.000" in out and "6.00" in out, out  # 中央値 0.030s / 0.005s = 6.00
        # run 欠損
        os.remove(os.path.join(d, "a-run3.json"))
        must_fail(lambda: arm_stats(d, "a"), "run 欠損")
        # 無効 run（occupancy に RUN_INVALID）
        write(d, "a", 3, 0.03)
        with open(os.path.join(d, "gpu-occupancy-a-run3.txt"), "w") as f:
            f.write("pre util=0%\nRUN_INVALID: foreign process detected\n")
        must_fail(lambda: arm_stats(d, "a"), "無効 run")
        os.remove(os.path.join(d, "gpu-occupancy-a-run3.txt"))
        # 負値
        write(d, "a", 3, -1.0)
        must_fail(lambda: arm_stats(d, "a"), "負値")
        # NaN（json.dump は NaN を許す）
        write(d, "a", 3, float("nan"))
        must_fail(lambda: arm_stats(d, "a"), "NaN")
        # iters < 20
        write(d, "a", 3, 0.03, iters=5)
        must_fail(lambda: arm_stats(d, "a"), "iters 不足")
        # パース不能
        with open(os.path.join(d, "a-run3.json"), "w") as f:
            f.write("{broken")
        must_fail(lambda: arm_stats(d, "a"), "JSON 不正")
        # parity: pass マーカー無しは FAIL
        for r in RUNS:
            with open(os.path.join(d, "parity-run%d.log" % r), "w") as f:
                f.write("test result: ok. 1 passed; 0 failed; 0 ignored\n" if r != 2 else "nothing\n")
        res = parity_results(d)
        assert res[1] == "pass" and res[2] == "FAIL", res
    print("self-test OK")


def main(argv):
    if "--self-test" in argv:
        self_test()
        return 0
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = argv[1]
    arms = None
    baseline = None
    parity = "--parity" in argv
    for i, a in enumerate(argv):
        if a == "--arms":
            arms = argv[i + 1].split(",")
        if a == "--baseline":
            baseline = argv[i + 1]
    if not arms:
        raise SystemExit("--arms が必要")
    if baseline is not None and baseline not in arms:
        raise SystemExit("--baseline は --arms に含める")
    try:
        print(render(d, arms, baseline, parity))
    except ValueError as e:
        print("集計失敗: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
