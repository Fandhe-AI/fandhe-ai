#!/usr/bin/env python3
"""イシュー #2997: int8 格上げ条件 (c) 実測ログの厳格な集計（python3 標準ライブラリのみ）。

役割: orchestrate_cuda.sh / orchestrate_mac.sh が回収した <arm>-run{1..5}.json から、
RULE.txt R1（5 run の median_secs の中央値・Q1/Q3・対 PyTorch 比）どおりに Markdown 表を出す。
README.md §2.1 / §2.3 へはこの出力を手で直さず転記する。

使い方:
  python3 -I aggregate.py <ログdir> --arms prefusion,fused,pytorch [--parity] [--baseline pytorch] [--require-occupancy]
  （CUDA 腕は --require-occupancy 必須: gpu-occupancy-*.txt 欠損の run を拒否する。
   <arm>-run<N>.status が非ゼロの run は常に拒否する）
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


def check_run_status(d, stem):
    """orchestrate_*.sh が書く <stem>.status（計測コマンドの終了コード）が非ゼロなら run を拒否する。
    status ファイルが無い run（スクリプト改修前の記録済みログ）は検査できないため通す。"""
    sp = os.path.join(d, stem + ".status")
    if os.path.isfile(sp):
        with open(sp, encoding="utf-8") as f:
            v = f.read().strip()
        if v != "0":
            raise ValueError("失敗終了した run（status=%s）: %s" % (v, sp))


def check_occupancy(d, arm, run, require_occupancy):
    """異種プロセス混入で無効化された run を拒否する（RULE.txt DISCARD_CUDA）。
    require_occupancy（CUDA 腕）では監視ログ欠損も拒否する（監視なしの run を有効扱いにしない）。"""
    occ = os.path.join(d, "gpu-occupancy-%s-run%d.txt" % (arm, run))
    if not os.path.isfile(occ):
        if require_occupancy:
            raise ValueError("occupancy 監視ログ欠損: %s" % occ)
        return
    with open(occ, encoding="utf-8", errors="replace") as f:
        if re.search(r"^RUN_INVALID", f.read(), re.M):
            raise ValueError("無効 run（RUN_INVALID）: %s" % occ)


def load_run(d, arm, run, require_occupancy=False):
    path = os.path.join(d, "%s-run%d.json" % (arm, run))
    if not os.path.isfile(path):
        raise ValueError("run 欠損: %s" % path)
    check_run_status(d, "%s-run%d" % (arm, run))
    check_occupancy(d, arm, run, require_occupancy)
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


def arm_stats(d, arm, require_occupancy=False):
    per_run = {r: load_run(d, arm, r, require_occupancy) for r in RUNS}
    med, q1, q3 = median_q1_q3(list(per_run.values()))
    return per_run, med, q1, q3


def parity_results(d, require_occupancy=False):
    out = {}
    for r in RUNS:
        path = os.path.join(d, "parity-run%d.log" % r)
        if not os.path.isfile(path):
            raise ValueError("parity ログ欠損: %s" % path)
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        ok = re.search(r"^test result: ok\. 1 passed; 0 failed", text, re.M) is not None
        bad = re.search(r"FAILED|panicked", text) is not None
        # 失敗終了・無効 run・監視ログ欠損の parity は pass と表示させない（RULE.txt DISCARD_CUDA）。
        check_run_status(d, "parity-run%d" % r)
        check_occupancy(d, "parity", r, require_occupancy)
        out[r] = "pass" if (ok and not bad) else "FAIL"
    return out


def reference_only_runs(d, arm):
    """load ログに REFERENCE_ONLY_load1_high を持つ run 番号（RULE.txt DISCARD_MAC: 参考扱い）。
    load ログは Mac 腕のみ存在するため、ファイルが無い腕（CUDA）は空とする。"""
    out = []
    for r in RUNS:
        path = os.path.join(d, "load-%s-run%d.txt" % (arm, r))
        if os.path.isfile(path):
            with open(path, encoding="utf-8", errors="replace") as f:
                if re.search(r"note=REFERENCE_ONLY_load1_high", f.read()):
                    out.append(r)
    return out


def ms(x):
    return "%.3f" % (x * 1000.0)


def render(d, arms, baseline, parity, require_occupancy=False):
    stats = {a: arm_stats(d, a, require_occupancy) for a in arms}
    ref = {a: reference_only_runs(d, a) for a in arms}
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
        label = a + ("（参考扱い・load1 高）" if ref[a] else "")
        lines.append("| %s | " % label + " | ".join(ms(per_run[r]) for r in RUNS)
                     + " | %s | %s | %s | %s |" % (ms(med), ms(q1), ms(q3), ratio))
    lines.append("")
    lines.append("単位: ms（各 run の median_secs を 1000 倍。中央値・Q1・Q3 は 5 run の値から算出）。"
                 "対 PyTorch 比は Rust 腕の中央値 / PyTorch 腕の中央値（1 より大きければ遅い）。")
    for a in arms:
        if ref[a]:
            lines.append("")
            lines.append("注記: %s の " % a + ", ".join("run%d" % r for r in ref[a])
                         + " は開始前 load1 が基準（< 2.0）を満たさず参考扱い（load1 高）。"
                         "この腕の中央値・Q1・Q3 も参考扱いで、(c) の充足要件に数えない。")
    if parity:
        res = parity_results(d, require_occupancy)
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
        # 参考扱い（load1 高）の注記
        with open(os.path.join(d, "load-a-run2.txt"), "w") as f:
            f.write("pre load1=3.1 waited=1800s note=REFERENCE_ONLY_load1_high\n")
        with open(os.path.join(d, "load-pytorch-run1.txt"), "w") as f:
            f.write("pre load1=0.5 waited=0s note=load1_ok\n")
        out = render(d, ["a", "pytorch"], "pytorch", False)
        assert "参考扱い（load1 高）" in out and "a の run2" in out and "pytorch（参考" not in out, out
        os.remove(os.path.join(d, "load-a-run2.txt"))
        # run 欠損
        os.remove(os.path.join(d, "a-run3.json"))
        must_fail(lambda: arm_stats(d, "a"), "run 欠損")
        # 無効 run（occupancy に RUN_INVALID）
        write(d, "a", 3, 0.03)
        with open(os.path.join(d, "gpu-occupancy-a-run3.txt"), "w") as f:
            f.write("pre util=0%\nRUN_INVALID: foreign process detected\n")
        must_fail(lambda: arm_stats(d, "a"), "無効 run")
        # CUDA 腕: 監視ログ欠損は --require-occupancy で拒否、Mac 腕（既定）では通す
        os.remove(os.path.join(d, "gpu-occupancy-a-run3.txt"))
        arm_stats(d, "a")
        must_fail(lambda: arm_stats(d, "a", True), "occupancy 監視ログ欠損")
        # 失敗終了 run（status 非ゼロ）
        with open(os.path.join(d, "a-run3.status"), "w") as f:
            f.write("125\n")
        must_fail(lambda: arm_stats(d, "a"), "失敗終了 run")
        with open(os.path.join(d, "a-run3.status"), "w") as f:
            f.write("0\n")
        arm_stats(d, "a")
        os.remove(os.path.join(d, "a-run3.status"))
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
        # parity の無効 run（occupancy に RUN_INVALID）は拒否
        with open(os.path.join(d, "gpu-occupancy-parity-run1.txt"), "w") as f:
            f.write("RUN_INVALID: foreign process detected\n")
        must_fail(lambda: parity_results(d), "parity 無効 run")
        os.remove(os.path.join(d, "gpu-occupancy-parity-run1.txt"))
        must_fail(lambda: parity_results(d, True), "parity 監視ログ欠損")
        with open(os.path.join(d, "parity-run1.status"), "w") as f:
            f.write("1\n")
        must_fail(lambda: parity_results(d), "parity 失敗終了 run")
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
    require_occupancy = "--require-occupancy" in argv
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
        print(render(d, arms, baseline, parity, require_occupancy))
    except ValueError as e:
        print("集計失敗: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
