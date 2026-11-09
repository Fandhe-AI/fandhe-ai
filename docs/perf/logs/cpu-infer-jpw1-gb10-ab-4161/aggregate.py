#!/usr/bin/env python3
"""#4161 GB10 推論限定 jpw1 A/B の厳格集計（標準ライブラリのみ。`python3 -I` で実行する）。

役割: round{i}/raw/{arm}-{task}-{mode}.jsonl と round{i}/gate.txt を読み、RULE.txt 項 8 の判定を出す。
欠損・重複・解釈不能・フィールド不一致・プレースホルダ残り・自身の sha256 不一致は例外で止める
（fail-closed）。判定閾値は RULE.txt 固定で、このスクリプトは動かさない。

使い方: python3 -I aggregate.py <ログディレクトリ> / python3 -I aggregate.py --self-test
出力: <ログディレクトリ>/aggregate.md・aggregate.tsv
"""
import hashlib
import json
import math
import os
import re
import statistics
import sys
import tempfile

ARMS = ("off", "inf-jpw1")
CELLS = (("train", "fresh"), ("train", "reuse"), ("infer", "fresh"), ("infer", "reuse"))
INFER_CELLS = (("infer", "fresh"), ("infer", "reuse"))
TRAIN_CELLS = (("train", "fresh"), ("train", "reuse"))


class AggError(Exception):
    pass


def parse_params(rule_path, self_sha=None):
    params = {}
    in_params = False
    with open(rule_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("PARAMS"):
                in_params = True
                continue
            if in_params:
                if not line.startswith("  "):
                    if line.strip() == "":
                        break
                    in_params = False
                    continue
                m = re.match(r"^  ([A-Za-z0-9_-]+):\s*(.*?)\s*$", line)
                if m:
                    params[m.group(1)] = m.group(2)
    need = ("fixed-at-utc", "tree_sha", "tree_fingerprint", "aggregate_script_sha256", "rounds", "arms")
    for k in need:
        if k not in params:
            raise AggError(f"PARAMS に {k} が無い")
        if "<" in params[k] or ">" in params[k]:
            raise AggError(f"PARAMS の {k} にプレースホルダが残っている")
    if self_sha is not None and params["aggregate_script_sha256"] != self_sha:
        raise AggError("aggregate_script_sha256 が自身の sha256 と不一致")
    if tuple(params["arms"].split(",")) != ARMS:
        raise AggError("arms が想定と不一致")
    try:
        params["rounds_int"] = int(params["rounds"])
    except ValueError as e:
        raise AggError("rounds が整数でない") from e
    if params["rounds_int"] < 1:
        raise AggError("rounds < 1")
    return params


def read_cell(path, arm, task, mode):
    if not os.path.isfile(path):
        raise AggError(f"欠損: {path}")
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise AggError(f"{path}:{n}: JSON 解釈不能") from e
    hit = [r for r in rows if isinstance(r, dict) and r.get("task") == task and r.get("mode") == mode]
    if len(hit) != 1:
        raise AggError(f"{path}: task={task} mode={mode} の行が {len(hit)} 件（1 件必須。欠損または重複）")
    r = hit[0]
    if r.get("framework") != "fandhe-ai":
        raise AggError(f"{path}: framework 不一致 {r.get('framework')!r}")
    if r.get("device") != "cpu" or r.get("size") != 64:
        raise AggError(f"{path}: device/size 不一致")
    med = r.get("median_s")
    if isinstance(med, bool) or not isinstance(med, (int, float)) or not math.isfinite(med) or med <= 0:
        raise AggError(f"{path}: median_s が不正 {med!r}")
    cs = r.get("checksum")
    if isinstance(cs, bool) or not isinstance(cs, (int, float)) or not math.isfinite(cs):
        raise AggError(f"{path}: checksum が不正 {cs!r}")
    return float(med), float(cs)


def read_gate(path):
    if not os.path.isfile(path):
        raise AggError(f"欠損: {path}")
    with open(path, encoding="utf-8") as f:
        first = f.read().split()
    if not first or first[0] not in ("PASS", "TIMEOUT", "INVALID"):
        raise AggError(f"{path}: gate 形式不正")
    return first[0] == "PASS"


def read_prereq(path, arm):
    """prereq/<arm>.log の probe 行を検査する。成立で True、不成立で False、欠損は例外。"""
    if not os.path.isfile(path):
        raise AggError(f"欠損: {path}")
    applied = {}
    aff = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.search(r"probe=infer_jpw1 path=(\w+) gate=(\w+) applied=(\d+)", line)
            if m:
                if m.group(1) in applied:
                    raise AggError(f"{path}: path={m.group(1)} が重複")
                applied[m.group(1)] = int(m.group(3))
            m = re.search(r"probe=gb10_affinity enabled=(\w+)", line)
            if m:
                aff = m.group(1)
    for p in ("infer_fresh", "infer_reuse", "train_fresh", "train_reuse"):
        if p not in applied:
            raise AggError(f"{path}: probe {p} が無い")
    if aff is None:
        raise AggError(f"{path}: gb10_affinity 行が無い")
    ok = aff == "false"
    if arm == "off":
        ok = ok and all(v == 0 for v in applied.values())
    else:
        ok = ok and applied["infer_fresh"] > 0 and applied["infer_reuse"] > 0
        ok = ok and applied["train_fresh"] == 0 and applied["train_reuse"] == 0
    return ok


def aggregate(logdir, params):
    n = params["rounds_int"]
    med = {}
    cks = {}
    gates = []
    for i in range(1, n + 1):
        rd = os.path.join(logdir, f"round{i}")
        passed = read_gate(os.path.join(rd, "gate.txt"))
        gates.append(passed)
        for arm in ARMS:
            for task, mode in CELLS:
                p = os.path.join(rd, "raw", f"{arm}-{task}-{mode}.jsonl")
                try:
                    m, c = read_cell(p, arm, task, mode)
                except AggError:
                    # ゲート不通過ラウンドは未計測でよい（正式系列から外れ undetermined になる）。
                    # 通過ラウンドの欠損・不正は従来どおり fail-closed で止める。
                    if passed:
                        raise
                    continue
                med[(i, arm, task, mode)] = m
                cks[(i, arm, task, mode)] = c
    prereq = {arm: read_prereq(os.path.join(logdir, "prereq", f"{arm}.log"), arm) for arm in ARMS}
    return med, cks, gates, prereq


def judge(med, cks, gates, prereq, n):
    ratios = {}
    for cell in CELLS:
        per = [med[(i, "inf-jpw1") + cell] / med[(i, "off") + cell] for i in range(1, n + 1)
               if (i, "inf-jpw1") + cell in med and (i, "off") + cell in med]
        ratios[cell] = (statistics.median(per) if per else float("nan"), per)
    checksum_ok = True
    for cell in CELLS:
        vals = {cks[(i, arm) + cell] for i in range(1, n + 1) for arm in ARMS if (i, arm) + cell in cks}
        if len(vals) > 1:
            checksum_ok = False
    complete = all((i, arm) + cell in med for i in range(1, n + 1) for arm in ARMS for cell in CELLS)
    formal = all(gates) and len(gates) == n and complete
    premise = all(prereq.values())
    if not checksum_ok:
        verdict = "FAIL"
    elif not formal or not premise:
        verdict = "undetermined"
    elif any(ratios[c][0] > 1.0 for c in TRAIN_CELLS) or any(ratios[c][0] >= 1.0 for c in INFER_CELLS):
        verdict = "REJECT"
    else:
        verdict = "ADOPT"
    return verdict, ratios, checksum_ok, formal, premise


def noise_band(per):
    """NB-RANGE5（記録のみ）: 5 ラウンドの run 別比の (max-min)/median。"""
    if not per:
        return float("nan")
    m = statistics.median(per)
    return (max(per) - min(per)) / m if m else float("nan")


def render(params, verdict, ratios, checksum_ok, formal, premise, logdir):
    lines = ["# #4161 aggregate", "", f"verdict: {verdict}", f"checksum_ok: {checksum_ok}",
             f"formal_series(gate {params['rounds_int']}/{params['rounds_int']}): {formal}",
             f"premise_ok: {premise}", "", "| cell | median ratio | per-round | wins | NB-RANGE5 |",
             "|---|---|---|---|---|"]
    tsv = ["task\tmode\tmedian_ratio\tper_round\twins\tnb_range5"]
    for cell, (mr, per) in ratios.items():
        wins = sum(1 for r in per if r < 1.0)
        nb = noise_band(per)
        pr = ",".join(f"{r:.4f}" for r in per)
        lines.append(f"| {cell[0]} {cell[1]} | {mr:.4f} | {pr} | {wins}/{len(per)} | {nb:.4f} |")
        tsv.append(f"{cell[0]}\t{cell[1]}\t{mr:.6f}\t{pr}\t{wins}\t{nb:.6f}")
    with open(os.path.join(logdir, "aggregate.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(logdir, "aggregate.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(tsv) + "\n")


def self_sha():
    with open(os.path.abspath(__file__), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run(logdir):
    params = parse_params(os.path.join(logdir, "RULE.txt"), self_sha())
    med, cks, gates, prereq = aggregate(logdir, params)
    v, ratios, cok, formal, premise = judge(med, cks, gates, prereq, params["rounds_int"])
    render(params, v, ratios, cok, formal, premise, logdir)
    print(v)
    return v


# ---- self-test ----
def _write_fixture(root, n, off, on, gates=None, ck_off=1.5, ck_on=1.5, skip=None, dup=None):
    for i in range(1, n + 1):
        rd = os.path.join(root, f"round{i}")
        os.makedirs(os.path.join(rd, "raw"), exist_ok=True)
        g = "PASS" if gates is None or gates[i - 1] else "TIMEOUT"
        with open(os.path.join(rd, "gate.txt"), "w") as f:
            f.write(g + " x\n")
        for arm, vals, ck in (("off", off, ck_off), ("inf-jpw1", on, ck_on)):
            for task, mode in CELLS:
                if skip == (i, arm, task, mode):
                    continue
                row = {"framework": "fandhe-ai", "task": task, "device": "cpu", "size": 64,
                       "mode": mode, "median_s": vals[(task, mode)], "checksum": ck}
                with open(os.path.join(rd, "raw", f"{arm}-{task}-{mode}.jsonl"), "w") as f:
                    f.write(json.dumps(row) + "\n")
                    if dup == (i, arm, task, mode):
                        f.write(json.dumps(row) + "\n")
    os.makedirs(os.path.join(root, "prereq"), exist_ok=True)
    for arm in ARMS:
        a = 0 if arm == "off" else 3
        with open(os.path.join(root, "prereq", f"{arm}.log"), "w") as f:
            for p, v in (("infer_fresh", a), ("infer_reuse", a), ("train_fresh", 0), ("train_reuse", 0)):
                f.write(f"probe=infer_jpw1 path={p} gate=x applied={v}\n")
            f.write("probe=gb10_affinity enabled=false\n")


def _verdict(**kw):
    n = 5
    base = {c: 1.0 for c in CELLS}
    on = kw.pop("on", {("infer", "fresh"): 0.9, ("infer", "reuse"): 0.9,
                       ("train", "fresh"): 1.0, ("train", "reuse"): 1.0})
    with tempfile.TemporaryDirectory() as d:
        _write_fixture(d, n, base, on, **kw)
        params = {"rounds_int": n}
        med, cks, gates, prereq = aggregate(d, params)
        return judge(med, cks, gates, prereq, n)[0]


def self_test():
    assert _verdict() == "ADOPT"
    bad = {("infer", "fresh"): 0.9, ("infer", "reuse"): 0.9, ("train", "fresh"): 1.01, ("train", "reuse"): 1.0}
    assert _verdict(on=bad) == "REJECT"
    assert _verdict(gates=[True, True, True, True, False]) == "undetermined"
    assert _verdict(ck_on=2.5) == "FAIL"
    for kw in ({"skip": (2, "off", "infer", "fresh")}, {"dup": (1, "inf-jpw1", "train", "reuse")}):
        try:
            _verdict(**kw)
        except AggError:
            pass
        else:
            raise AssertionError("例外が出なかった")
    # ゲート不通過ラウンドの raw 欠損は例外でなく undetermined（ADOPT 不可）
    n = 5
    with tempfile.TemporaryDirectory() as d:
        base = {c: 1.0 for c in CELLS}
        on = {("infer", "fresh"): 0.9, ("infer", "reuse"): 0.9, ("train", "fresh"): 1.0, ("train", "reuse"): 1.0}
        _write_fixture(d, n, base, on, gates=[True, True, True, True, False])
        for f in os.listdir(os.path.join(d, "round5", "raw")):
            os.remove(os.path.join(d, "round5", "raw", f))
        med, cks, gates, prereq = aggregate(d, {"rounds_int": n})
        assert judge(med, cks, gates, prereq, n)[0] == "undetermined"
    # 正常系: ハイフン付きキー（fixed-at-utc）を読める
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "RULE.txt")
        with open(p, "w") as f:
            f.write("PARAMS\n  fixed-at-utc: 2026-11-09T00:00:00Z\n  tree_sha: abc\n  tree_fingerprint: def\n"
                    "  aggregate_script_sha256: 123\n  rounds: 5\n  arms: off,inf-jpw1\n  machine: x\n")
        pr = parse_params(p)
        assert pr["fixed-at-utc"] == "2026-11-09T00:00:00Z" and pr["rounds_int"] == 5
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "RULE.txt")
        with open(p, "w") as f:
            f.write("PARAMS\n  fixed-at-utc: <x>\n")
        try:
            parse_params(p)
        except AggError:
            pass
        else:
            raise AssertionError("プレースホルダを拒否しなかった")
    print("self-test OK")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        self_test()
    elif len(sys.argv) == 2:
        try:
            run(sys.argv[1])
        except AggError as e:
            print(f"AGGREGATE_ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(__doc__)
        sys.exit(2)
