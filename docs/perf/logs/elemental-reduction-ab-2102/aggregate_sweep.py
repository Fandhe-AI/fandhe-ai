#!/usr/bin/env python3
"""縮約しきい値スイープ（`reduction_threshold_sweep`）の機械集計（イシュー #2102）。

役割: 2101 の RULE.txt（Phase 0）と 2102 の RULE.txt（利得判定）を、生ログ
（機体ごとの run1.log..run5.log）から機械的に計算する。事前登録規則の再実装であり、
規則側を変える場合は RULE.txt を正とする（事後緩和はしない）。stdlib のみ。

入力行: `op=<name> n=<int> arm=seq|par median_ns=<int> threads=<int> checksum=<hex>`
終了コード: 0=Stage 1 へ進める / 2=入力不正 / 4=checksum 不一致（FAIL）
            5=REJECT（T <= 640）/ 6=undetermined（利得が立たない機体がある）
"""
import argparse
import os
import re
import statistics
import sys
import tempfile

OPS = ("sum_all", "sum_axis0", "sum_axis1", "max_all", "mean_all")
SIZES = (640, 2560, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
RUNS = 5
R_MAX = 1.05
DEFAULT_T = 1 << 18
REJECT_T = 640

LINE_RE = re.compile(
    r"^op=(?P<op>[a-z0-9_]+) n=(?P<n>\d+) arm=(?P<arm>seq|par) "
    r"median_ns=(?P<ns>\d+) threads=(?P<th>\d+) checksum=(?P<ck>0x[0-9a-f]+)$"
)


class InputError(Exception):
    pass


def parse_run(path):
    """1 run のログを {(op, n, arm): (median_ns, checksum)} に読む。欠落・重複・不正行は InputError。"""
    if not os.path.isfile(path):
        raise InputError(f"missing file: {path}")
    cells = {}
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.startswith("op="):
                continue
            m = LINE_RE.match(line)
            if not m:
                raise InputError(f"{path}:{lineno}: malformed op= line")
            op, n = m["op"], int(m["n"])
            if op not in OPS or n not in SIZES:
                raise InputError(f"{path}:{lineno}: unexpected cell op={op} n={n}")
            key = (op, n, m["arm"])
            if key in cells:
                raise InputError(f"{path}:{lineno}: duplicate cell {key}")
            ns = int(m["ns"])
            if ns <= 0:
                raise InputError(f"{path}:{lineno}: non-positive median_ns")
            cells[key] = (ns, m["ck"])
    expected = {(o, n, a) for o in OPS for n in SIZES for a in ("seq", "par")}
    missing = expected - set(cells)
    if missing:
        raise InputError(f"{path}: {len(missing)} missing cells (e.g. {sorted(missing)[0]})")
    return cells


def load_machine(directory):
    return [parse_run(os.path.join(directory, f"run{i}.log")) for i in range(1, RUNS + 1)]


def analyze_machine(runs):
    """機体 1 台分の r(n)・checksum 不一致・利得の元データを返す。"""
    table = {}
    mismatches = []
    for op in OPS:
        for n in SIZES:
            seq = [r[(op, n, "seq")] for r in runs]
            par = [r[(op, n, "par")] for r in runs]
            cks = {c for _, c in seq} | {c for _, c in par}
            if len(cks) != 1:
                mismatches.append((op, n))
            med_seq = statistics.median(v for v, _ in seq)
            med_par = statistics.median(v for v, _ in par)
            per_run = [s[0] / p[0] for s, p in zip(seq, par)]
            table[(op, n)] = (med_seq / med_par, per_run)
    return table, mismatches


def violation_point(table, op):
    for n in SIZES:
        if table[(op, n)][0] > R_MAX:
            return n
    return DEFAULT_T


def evaluate(machines):
    """machines: {name: runs}。判定結果の dict を返す。"""
    analyses = {name: analyze_machine(runs) for name, runs in machines.items()}
    result = {"analyses": analyses, "checksum_fail": [], "viol": {}, "gain": {}}
    for name, (_, mism) in analyses.items():
        result["checksum_fail"] += [(name, op, n) for op, n in mism]
    if result["checksum_fail"]:
        result["verdict"] = "FAIL"
        return result
    for name, (table, _) in analyses.items():
        for op in OPS:
            result["viol"][(name, op)] = violation_point(table, op)
    t = min(result["viol"].values())
    result["T"] = t
    # T 未満の全セルで max r <= R_MAX の再検証（規則の構成上の保証を生ログから確認する）
    result["max_r_below_T"] = max(
        (tbl[(op, n)][0] for tbl, _ in analyses.values() for op in OPS for n in SIZES if n < t),
        default=0.0,
    )
    if t <= REJECT_T:
        result["verdict"] = "REJECT"
        return result
    for name, (table, _) in analyses.items():
        cells = [
            (op, n)
            for op in OPS
            for n in SIZES
            if n < t and table[(op, n)][0] < 1.00 and all(x < 1.00 for x in table[(op, n)][1])
        ]
        result["gain"][name] = cells
    result["verdict"] = "PROCEED" if all(result["gain"].values()) else "UNDETERMINED"
    return result


def render(result):
    out = ["# 縮約しきい値スイープ集計（#2102）", ""]
    for name, (table, _) in result["analyses"].items():
        out += [f"## {name}", "", "| op | n | r(n)=seq/par | 5 run 個別 r |", "|---|---|---|---|"]
        for op in OPS:
            for n in SIZES:
                r, per = table[(op, n)]
                out.append(f"| {op} | {n} | {r:.4f} | {' '.join(f'{x:.3f}' for x in per)} |")
        out.append("")
    out.append(f"## 判定: {result['verdict']}")
    if result["checksum_fail"]:
        out.append(f"- checksum 不一致: {result['checksum_fail']}")
    else:
        out.append(f"- T = {result['T']}（T 未満の全セルの max r = {result['max_r_below_T']:.4f} / R_MAX = {R_MAX}）")
        for (name, op), v in sorted(result["viol"].items()):
            out.append(f"- 違反点 v: {name} {op} = {v}")
        for name, cells in result["gain"].items():
            out.append(f"- 利得（5/5 run で r<1.00・T 未満）{name}: {cells if cells else 'なし'}")
    return "\n".join(out) + "\n"


EXIT = {"PROCEED": 0, "FAIL": 4, "REJECT": 5, "UNDETERMINED": 6}


def _synth_log(path, fn, ck="0x1"):
    """fn(op, n, arm) -> ns から合成ログを書く。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("threads=8\n")
        for op in OPS:
            for n in SIZES:
                for arm in ("seq", "par"):
                    f.write(f"op={op} n={n} arm={arm} median_ns={fn(op, n, arm)} threads=8 checksum={ck}\n")


def _machine(d, fn, ck_fn=None):
    for i in range(1, RUNS + 1):
        _synth_log(os.path.join(d, f"run{i}.log"), fn)


def self_test():
    def gain(op, n, arm):  # 小 n で seq が速く、n>=8192 で par が速い
        if arm == "par":
            return 1000
        return 500 if n < 8192 else 2000

    def no_gain(op, n, arm):  # 常に par が速い（r>1.05）
        return 1000 if arm == "par" else 3000

    def flat(op, n, arm):  # r=1.0 ちょうど（利得なし・違反なし）
        return 1000

    with tempfile.TemporaryDirectory() as base:
        def mk(name, fn):
            d = os.path.join(base, name)
            os.makedirs(d)
            _machine(d, fn)
            return load_machine(d)

        a, b = mk("a", gain), mk("b", gain)
        res = evaluate({"a": a, "b": b})
        assert res["T"] == 8192 and res["verdict"] == "PROCEED", res["verdict"]
        assert res["max_r_below_T"] <= R_MAX

        res = evaluate({"a": a, "b": mk("f", flat)})
        assert res["verdict"] == "UNDETERMINED" and res["T"] == 8192, res["verdict"]

        res = evaluate({"a": a, "b": mk("n", no_gain)})
        assert res["T"] == 640 and res["verdict"] == "REJECT"

        # 機体 b だけが 2560 で違反（640 は r=1.0）-> T は全体の最小値 2560
        res = evaluate({"a": a, "b": mk("l", lambda o, n, ar: 1000 if ar == "par" else (900 if n == 640 else 3000 if n == 2560 else 500))})
        assert res["T"] == 2560, res["T"]

        # checksum 不一致 -> FAIL
        d = os.path.join(base, "ck")
        os.makedirs(d)
        _machine(d, gain)
        with open(os.path.join(d, "run3.log"), encoding="utf-8") as f:
            text = f.read()
        text = text.replace("arm=par median_ns=1000 threads=8 checksum=0x1", "arm=par median_ns=1000 threads=8 checksum=0x2", 1)
        with open(os.path.join(d, "run3.log"), "w", encoding="utf-8") as f:
            f.write(text)
        assert evaluate({"a": a, "b": load_machine(d)})["verdict"] == "FAIL"

        # 欠落行・不正行・重複行 -> InputError
        for label, mutate in (
            ("missing", None),
            ("malformed", lambda t: t + "op=sum_all n=640 arm=seq median_ns=abc\n"),
            ("dup", lambda t: t + t.splitlines()[1] + "\n"),
        ):
            d2 = os.path.join(base, label)
            os.makedirs(d2)
            _machine(d2, gain)
            p = os.path.join(d2, "run1.log")
            with open(p, encoding="utf-8") as f:
                t = f.read()
            if label == "missing":
                t = "\n".join(l for l in t.splitlines() if not l.startswith("op=max_all n=640 arm=seq")) + "\n"
            else:
                t = mutate(t)
            with open(p, "w", encoding="utf-8") as f:
                f.write(t)
            try:
                load_machine(d2)
            except InputError:
                pass
            else:
                raise AssertionError(f"{label}: expected InputError")
    print("self-test OK")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m4max", help="M4 Max の run1..5.log のあるディレクトリ")
    ap.add_argument("--gb10", help="GB10 の run1..5.log のあるディレクトリ")
    ap.add_argument("--out", help="aggregate.md の出力先（絶対パス）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv[1:])
    if args.self_test:
        self_test()
        return 0
    if not args.m4max or not args.gb10:
        print("error: --m4max と --gb10 の両方が必要（両機体の判定規則のため）", file=sys.stderr)
        return 2
    try:
        machines = {"m4max": load_machine(args.m4max), "gb10": load_machine(args.gb10)}
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    result = evaluate(machines)
    text = render(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return EXIT[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
