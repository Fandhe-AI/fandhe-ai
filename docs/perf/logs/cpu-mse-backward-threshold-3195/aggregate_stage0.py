#!/usr/bin/env python3
"""MSE backward しきい値スイープ（`mse_backward_threshold_sweep`）の機械集計（イシュー #3195）。

役割: RULE.txt §3（Stage 0 の記録と前提ゲート）を、機体ごとの stage0-run1..5.log から
機械的に適用する。事前登録規則の再実装であり、規則側を変える場合は RULE.txt を正とする
（事後緩和はしない）。`docs/perf/logs/elemental-reduction-ab-2102/aggregate_sweep.py` の同型。
python3 標準ライブラリのみ。パースは厳格（欠損・重複・不正行・run 数不足は例外）。

入力行: `n=<int> arm=seq|par median_ns=<int> first_call_ns=<int> threads=<int> checksum=<hex>`
終了コード: 0=前提ゲート成立 / 2=入力不正 / 4=checksum 不一致（FAIL）/ 6=ゲート (b) 不成立（undetermined）
"""
import argparse
import os
import re
import statistics
import sys
import tempfile

SIZES = (640, 2560, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
GATE_SIZES = (640, 2560)
REPORT_SIZES = (640, 2560, 4096)
RUNS = 5
LINE_RE = re.compile(
    r"^n=(?P<n>\d+) arm=(?P<arm>seq|par) median_ns=(?P<ns>\d+) "
    r"first_call_ns=(?P<fc>\d+) threads=(?P<th>\d+) checksum=(?P<ck>0x[0-9a-f]+)$"
)


class InputError(Exception):
    pass


def parse_run(path):
    """1 run のログを {(n, arm): (median_ns, checksum)} に読む。"""
    if not os.path.isfile(path):
        raise InputError(f"missing file: {path}")
    cells = {}
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.startswith("n="):
                continue
            m = LINE_RE.match(line)
            if not m:
                raise InputError(f"{path}:{lineno}: malformed n= line")
            n = int(m["n"])
            if n not in SIZES:
                raise InputError(f"{path}:{lineno}: unexpected n={n}")
            key = (n, m["arm"])
            if key in cells:
                raise InputError(f"{path}:{lineno}: duplicate cell {key}")
            ns = int(m["ns"])
            if ns <= 0:
                raise InputError(f"{path}:{lineno}: non-positive median_ns")
            cells[key] = (ns, m["ck"])
    missing = {(n, a) for n in SIZES for a in ("seq", "par")} - set(cells)
    if missing:
        raise InputError(f"{path}: {len(missing)} missing cells (e.g. {sorted(missing)[0]})")
    return cells


def load_machine(directory):
    return [parse_run(os.path.join(directory, f"stage0-run{i}.log")) for i in range(1, RUNS + 1)]


def analyze(runs):
    """{n: (seq 中央値, par 中央値, r(n) の 5 プロセス中央値, 個別 r)} と checksum 不一致 n の一覧。"""
    table, mismatches = {}, []
    for n in SIZES:
        seq = [r[(n, "seq")] for r in runs]
        par = [r[(n, "par")] for r in runs]
        if len({c for _, c in seq} | {c for _, c in par}) != 1:
            mismatches.append(n)
        per = [s[0] / p[0] for s, p in zip(seq, par)]
        table[n] = (
            statistics.median(v for v, _ in seq),
            statistics.median(v for v, _ in par),
            statistics.median(per),
            per,
        )
    return table, mismatches


def evaluate(machines):
    an = {name: analyze(runs) for name, runs in machines.items()}
    res = {"an": an, "fail": [(m, n) for m, (_, mm) in an.items() for n in mm], "gate_b": {}}
    if res["fail"]:
        res["verdict"] = "FAIL"
        return res
    for m, (tbl, _) in an.items():
        res["gate_b"][m] = all(all(x < 1.00 for x in tbl[n][3]) for n in GATE_SIZES)
    res["verdict"] = "PROCEED" if all(res["gate_b"].values()) else "UNDETERMINED"
    return res


def render(res):
    out = ["# MSE backward スイープ集計（#3195 Stage 0）", ""]
    for m, (tbl, _) in res["an"].items():
        out += [f"## {m}", "", "| n | seq 中央値 ns | par 中央値 ns | r(n) 中央値 | 5 プロセス個別 r |", "|---|---|---|---|---|"]
        for n in SIZES:
            s, p, r, per = tbl[n]
            mark = " *" if n in REPORT_SIZES else ""
            out.append(f"| {n}{mark} | {s:.0f} | {p:.0f} | {r:.4f} | {' '.join(f'{x:.3f}' for x in per)} |")
        out.append("")
    out.append(f"## 判定: {res['verdict']}")
    if res["fail"]:
        out.append(f"- (a) checksum 不一致: {res['fail']}")
    else:
        out.append("- (a) checksum 全一致")
        for m, ok in res["gate_b"].items():
            out.append(f"- (b) {m}: n in {GATE_SIZES} で 5/5 プロセス r<1.00 -> {'成立' if ok else '不成立'}")
    return "\n".join(out) + "\n"


EXIT = {"PROCEED": 0, "FAIL": 4, "UNDETERMINED": 6}


def _write(d, fn, ck="0x1"):
    for i in range(1, RUNS + 1):
        with open(os.path.join(d, f"stage0-run{i}.log"), "w", encoding="utf-8") as f:
            f.write("threads=8\n")
            for n in SIZES:
                for arm in ("seq", "par"):
                    f.write(f"n={n} arm={arm} median_ns={fn(n, arm)} first_call_ns=1 threads=8 checksum={ck}\n")


def self_test():
    good = lambda n, a: 100 if a == "seq" else 1000
    bad = lambda n, a: 1000 if a == "seq" else 100
    with tempfile.TemporaryDirectory() as base:
        def mk(name, fn):
            d = os.path.join(base, name)
            os.makedirs(d)
            _write(d, fn)
            return d

        g, b = mk("g", good), mk("b", bad)
        assert evaluate({"a": load_machine(g), "b": load_machine(g)})["verdict"] == "PROCEED"
        assert evaluate({"a": load_machine(g), "b": load_machine(b)})["verdict"] == "UNDETERMINED"
        p = os.path.join(g, "stage0-run3.log")
        with open(p, encoding="utf-8") as f:
            t = f.read().replace("arm=par median_ns=1000 first_call_ns=1 threads=8 checksum=0x1", "arm=par median_ns=1000 first_call_ns=1 threads=8 checksum=0x2", 1)
        with open(p, "w", encoding="utf-8") as f:
            f.write(t)
        assert evaluate({"a": load_machine(g)})["verdict"] == "FAIL"
        os.remove(os.path.join(b, "stage0-run5.log"))
        try:
            load_machine(b)
        except InputError:
            pass
        else:
            raise AssertionError("run 欠損で例外にならない")
    print("self-test OK")


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--macmini")
    ap.add_argument("--gb10")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv[1:])
    if a.self_test:
        self_test()
        return 0
    if not a.macmini or not a.gb10:
        print("error: --macmini と --gb10 の両方が必要", file=sys.stderr)
        return 2
    try:
        machines = {"macmini": load_machine(a.macmini), "gb10": load_machine(a.gb10)}
    except InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    res = evaluate(machines)
    text = render(res)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return EXIT[res["verdict"]]


if __name__ == "__main__":
    sys.exit(main(sys.argv))
