#!/usr/bin/env python3
"""イシュー #3180: Mac mini の `--phases` 5 run 中央値表・同期回数照合・同期 1 回の固定費を集計する。

土台は `../train-infer-phases-0.9.0-1980-1981/aggregate.py`（phase 表）。本版で足した点:
- 5 run 完備・欠損／非数値／重複／run 間フェーズ集合不一致を例外相当（exit 1）で拒否する（0 埋めしない）。
- `sync-fixed-cost/run{1..5}.log` の `SYNC_FIXED_COST_EMPTY_DISPATCH` 行（構造検査 cb+1,wait+1 付き）の中央値集計。
- `counters/*.log` から同期回数（encode／command_buffer／wait）を抽出し、設計 doc の 11/7/7（1 step）・
  5/3/3（backward 窓）との一致を判定する（不一致でも期待値を書き換えない。実測値をそのまま出す）。
- 同期による説明量（wait 回数 × 固定費）と step_total に対する比（Metal reuse。原因断定はしない）。

python3 標準ライブラリのみ。使い方（`python3 -I` で実行）:
  aggregate.py <dir> [--devices cpu,metal] [--machine "Apple M4 (Mac mini)"] > aggregate.md
  aggregate.py --self-test
"""
import argparse
import json
import math
import re
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

TOTAL = {"train_phases": "step_total", "infer_phases": "iter_total"}
EXPECT_STEP = (11, 7, 7)
EXPECT_BACKWARD = (5, 3, 3)

# RULE.txt §9: device／task／mode ごとの必須フェーズ集合（run1 との一致だけでは全 run 同時欠落を拒否できない）。
REQUIRED_PHASES = {
    ("cpu", "infer_phases", "fresh"): {"predict", "host_copy", "checksum", "iter_total"},
    ("cpu", "infer_phases", "reuse"): {"predict_resident", "host_copy", "checksum", "iter_total"},
    ("metal", "infer_phases", "fresh"): {"leaf_register", "forward", "to_tensor", "host_copy", "checksum", "iter_total"},
    ("metal", "infer_phases", "reuse"): {"predict_resident", "host_copy", "checksum", "iter_total"},
    **{
        (dev, "train_phases", "fresh"): {
            "tape_build", "leaf_register", "forward", "loss_readout", "backward",
            "param_readout", "host_sgd", "apply_params", "tape_drop", "step_total",
        }
        for dev in ("cpu", "metal")
    },
    **{
        (dev, "train_phases", "reuse"): {
            "tape_build", "leaf_register", "forward_resident", "loss_readout", "backward",
            "device_update", "tape_drop", "step_total",
        }
        for dev in ("cpu", "metal")
    },
}


class AggError(Exception):
    pass


def expected_cells(devices):
    return sorted(
        (dev, task, mode)
        for dev in devices
        for task in ("train_phases", "infer_phases")
        for mode in ("fresh", "reuse")
    )


def load_phases(d, devices):
    cells_expected = expected_cells(devices)
    vals = defaultdict(dict)  # key(dev,task,mode,phase) -> {run: median_s}
    order = {}
    phase_sets = {}
    for i in range(1, 6):
        f = d / f"run{i}.jsonl"
        if not f.exists():
            raise AggError(f"5 run 揃っていない（不足: {f.name}）")
        seen = set()
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task") not in TOTAL:
                continue
            key = (r["device"], r["task"], r["mode"], r["phase"])
            if key in seen:
                raise AggError(f"{f.name}:{ln} に重複行 {key}")
            seen.add(key)
            m = r.get("median_s")
            if isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m) or m < 0:
                raise AggError(f"{f.name}:{ln} の median_s が数値でない/負/非有限（NaN・Infinity）: {m!r}")
            vals[key][i] = float(m)
            order[key] = r["phase_index"]
        per_cell = defaultdict(set)
        for k in seen:
            per_cell[k[:3]].add(k[3])
        miss = [c for c in cells_expected if c not in per_cell]
        if miss:
            raise AggError(f"{f.name} に欠落セル {miss}（期待 {len(cells_expected)} セル）")
        extra = sorted(c for c in per_cell if c not in cells_expected)
        if extra:
            raise AggError(f"{f.name} に期待外のセル {extra}")
        for c, ph in per_cell.items():
            if TOTAL[c[1]] not in ph:
                raise AggError(f"{f.name} の {c} に合計フェーズ {TOTAL[c[1]]} がない")
        for c, ph in per_cell.items():
            req = REQUIRED_PHASES.get(c)
            if req is not None and ph != req:
                raise AggError(f"{f.name} の {c} が必須フェーズ集合と不一致（差分 {sorted(ph ^ req)}）")
        phase_sets[i] = {c: frozenset(ph) for c, ph in per_cell.items()}
    for i in range(2, 6):
        for c in cells_expected:
            if phase_sets[i][c] != phase_sets[1][c]:
                raise AggError(f"run{i} と run1 で {c} のフェーズ集合が不一致 {sorted(phase_sets[i][c] ^ phase_sets[1][c])}")
    for k, per_run in vals.items():
        if sorted(per_run) != [1, 2, 3, 4, 5]:
            raise AggError(f"{k} が 5 run 揃っていない（{sorted(per_run)}）")
    return vals, order, cells_expected


GATE_RE = re.compile(r"^(run[1-5](?:-syncfc)?) gate=(\S+) ")


def check_load_gate(d):
    """load_gate.log を検証し (全通過か, 説明文) を返す。記録欠損は拒否、未通過は参考系列扱い（RULE.txt §4）。"""
    f = d / "load_gate.log"
    if not f.exists():
        raise AggError(f"{f.name} が無い（負荷ゲート記録欠損）")
    status = {}
    failed = []
    for line in f.read_text(encoding="utf-8").splitlines():
        m = GATE_RE.match(line)
        if m:
            if m.group(1) in status:
                raise AggError(f"load_gate.log に {m.group(1)} の gate 記録が重複")
            status[m.group(1)] = m.group(2)
        elif " FAILED " in line or line.startswith("FAILED"):
            failed.append(line)
    expected = [f"run{i}{suf}" for i in range(1, 6) for suf in ("", "-syncfc")]
    missing = [k for k in expected if k not in status]
    if missing:
        raise AggError(f"load_gate.log に gate 記録がない: {missing}")
    bad = {k: status[k] for k in expected if status[k] != "pass"}
    if bad or failed:
        return False, f"負荷ゲート未通過/実行失敗あり（{bad or ''} {len(failed)} 件の FAILED）"
    return True, "負荷ゲート 10/10 pass（run 5 + 同期固定費 5）"


def render_phases(vals, order, cells, machine, gate=(True, "")):
    out = [f"# train／infer `--phases` 5 run 中央値（registry `fandhe-ai =0.11.0`・{machine}）\n"]
    if gate[0]:
        out.append(f"- 系列区分: **正式系列**（{gate[1]}）\n")
    else:
        out.append(f"- 系列区分: **参考系列（正式ではない。結論は参考扱い）** — {gate[1]}\n")
    medians = {}
    for cell in cells:
        dev, task, mode = cell
        phases = sorted((k for k in vals if k[:3] == cell), key=lambda k: order[k])
        total = statistics.median(vals[cell + (TOTAL[task],)].values())
        out.append(f"## {task} / {dev} / {mode}\n")
        out.append("| phase | 中央値 (µs) | min–max (µs) | 合計比 |")
        out.append("|---|---:|---|---:|")
        ranked = []
        for k in phases:
            runs = list(vals[k].values())
            med = statistics.median(runs)
            medians[k] = med
            out.append(
                f"| {k[3]} | {med*1e6:.1f} | {min(runs)*1e6:.1f}–{max(runs)*1e6:.1f} | {med/total*100:.1f}% |"
            )
            if k[3] != TOTAL[task]:
                ranked.append((med / total, k[3], med))
        ranked.sort(reverse=True)
        top = "・".join(f"{n}（{r*100:.1f}%・{m*1e6:.1f} µs）" for r, n, m in ranked[:3])
        covered = sum(m for _, _, m in ranked)
        out.append(f"\n- トップ 3: {top}")
        out.append(f"- フェーズ和／合計: {covered/total*100:.1f}%（差分は各フェーズ中央値の非加法性を含む）\n")
    return out, medians


SYNC_RE = re.compile(
    r"SYNC_FIXED_COST_EMPTY_DISPATCH trials=(\d+) warmup=(\d+) median_us=([0-9.]+) q1_us=([0-9.]+) q3_us=([0-9.]+) structure=cb\+1,wait\+1$"
)


def load_sync(d):
    meds = {}
    for i in range(1, 6):
        f = d / "sync-fixed-cost" / f"run{i}.log"
        if not f.exists():
            raise AggError(f"{f} が無い（5 run 揃っていない）")
        text = f.read_text(encoding="utf-8")
        hits = [m for m in (SYNC_RE.search(l.strip()) for l in text.splitlines()) if m]
        if len(hits) != 1:
            raise AggError(f"{f.name}: SYNC_FIXED_COST_EMPTY_DISPATCH 行が {len(hits)} 件（構造検査を通った行が 1 件でない）")
        if not re.search(r"test result: ok\. 1 passed", text):
            raise AggError(f"{f.name}: テストが ok で終わっていない（cb+1/wait+1 の assert 失敗の可能性）")
        meds[i] = float(hits[0].group(3))
    return meds


def parse_counters(d):
    res = {}
    f = d / "counters" / "batch_counters.log"
    if not f.exists():
        raise AggError(f"{f} が無い")
    t = f.read_text(encoding="utf-8")
    m = re.search(r"encode_delta=(\d+) command_buffer_delta=(\d+) wait_delta=(\d+)", t)
    if not m:
        raise AggError("batch_counters.log から encode/cb/wait を抽出できない")
    res["step"] = tuple(int(x) for x in m.groups())
    f = d / "counters" / "backward_dinput_phase.log"
    if not f.exists():
        raise AggError(f"{f} が無い")
    t = f.read_text(encoding="utf-8")
    lists = []
    for key in ("encode_deltas", "command_buffer_deltas", "wait_deltas"):
        m = re.search(key + r"=\[([0-9, ]+)\]", t)
        if not m:
            raise AggError(f"backward_dinput_phase.log から {key} を抽出できない")
        lists.append([int(x) for x in m.group(1).split(",")])
    if not (len(lists[0]) == len(lists[1]) == len(lists[2]) > 0):
        raise AggError("backward 窓の試行数が一致しない/0")
    res["backward_trials"] = list(zip(*lists))
    return res


def render_sync(meds, counters, medians, d):
    out = ["# 同期回数と同期 1 回の固定費（Mac mini）\n"]
    fc = statistics.median(meds.values())
    out.append("## 同期 1 回の固定費（空 dispatch の壁時間。cb+1/wait+1 の構造検査通過）\n")
    out.append("| run | 中央値 (µs) |")
    out.append("|---|---:|")
    for i in range(1, 6):
        out.append(f"| run{i} | {meds[i]:.3f} |")
    out.append(f"\n- 5 run 中央値: **{fc:.3f} µs**（min–max {min(meds.values()):.3f}–{max(meds.values()):.3f} µs）\n")
    out.append("## 同期回数（steady-state）\n")
    s = counters["step"]
    out.append(f"- 1 step 全体: encode/command_buffer/wait = {s[0]}/{s[1]}/{s[2]}（設計 doc の机上導出 11/7/7 と{'一致' if s == EXPECT_STEP else '**不一致**'}）")
    bt = counters["backward_trials"]
    uniq = sorted(set(bt))
    ok = uniq == [EXPECT_BACKWARD]
    out.append(
        f"- backward 窓（{len(bt)} 試行）: {', '.join('/'.join(map(str, u)) for u in uniq)}（期待 5/3/3 と{'全試行一致' if ok else '**不一致あり**'}）"
    )
    if ok:
        out.append(f"- 差し引き（forward＋loss＋update 窓）: {s[0]-5}/{s[1]-3}/{s[2]-3}")
    out.append("")
    out.append("## 同期による説明量（Metal reuse・参考。原因断定はしない）\n")
    wait = s[2]
    k = ("metal", "train_phases", "reuse", "step_total")
    if k in medians:
        total = medians[k]
        expl = wait * fc * 1e-6
        out.append(
            f"- wait {wait} 回 × {fc:.1f} µs = {expl*1e6:.1f} µs は、Mini の train Metal reuse step_total {total*1e6:.1f} µs の **{expl/total*100:.1f}%**"
        )
        out.append(f"- 未説明の残差: {(total-expl)*1e6:.1f} µs（{(1-expl/total)*100:.1f}%）")
    k2 = ("metal", "infer_phases", "reuse", "iter_total")
    ifc = d / "sync-fixed-cost" / "infer_fixed_cost_diag.log"
    if k in medians and k2 in medians and ifc.exists():
        m = re.search(r"wait_until_completed/iter min=(\d+) max=(\d+)", ifc.read_text(encoding="utf-8"))
        if m and m.group(1) == m.group(2):
            iw = int(m.group(1))
            if wait > iw:
                per = (medians[k] - medians[k2]) / (wait - iw)
                out.append(
                    f"- 外挿: (train reuse − infer reuse) / (wait {wait} − {iw}) = {per*1e6:.1f} µs／同期（実測固定費 {fc:.1f} µs と比較。infer の wait は診断テストの再現経路の値）"
                )
        else:
            out.append("- 外挿: infer の wait 回数が取れない/一定でないため算出しない")
    return out


def run(d, devices, machine):
    vals, order, cells = load_phases(d, devices)
    gate = check_load_gate(d)
    lines, medians = render_phases(vals, order, cells, machine, gate)
    meds = load_sync(d)
    counters = parse_counters(d)
    lines += render_sync(meds, counters, medians, d)
    return "\n".join(lines) + "\n"


# ---------------- self-test ----------------
def _write_fixture(d, drop=None, bad=None, gate="pass", allrun_drop=None):
    for i in range(1, 6):
        rows = []
        for dev in ("cpu", "metal"):
            for task, total in (("train_phases", "step_total"), ("infer_phases", "iter_total")):
                for mode in ("fresh", "reuse"):
                    names = sorted(REQUIRED_PHASES[(dev, task, mode)] - {total}) + [total]
                    for idx, ph in enumerate(names):
                        rows.append(
                            {"task": task, "device": dev, "mode": mode, "phase": ph, "phase_index": idx,
                             "median_s": 1e-4 * (idx + 1) * (1 + i / 100)}
                        )
        if allrun_drop:
            rows = [r for r in rows if not (r["device"] == "cpu" and r["task"] == "infer_phases" and r["phase"] == allrun_drop)]
        if drop == i:
            rows = [r for r in rows if not (r["device"] == "metal" and r["task"] == "infer_phases")]
        if bad == i:
            rows[0]["median_s"] = "nan?" if bad != 5 else float("inf")
        (d / f"run{i}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (d / "load_gate.log").write_text(
        "".join(
            f"run{i}{suf} gate={gate if (i, suf) == (3, '') else 'pass'} load1=1.00 waited_s=0 at=x\n"
            for i in range(1, 6)
            for suf in ("", "-syncfc")
        )
    )
    (d / "sync-fixed-cost").mkdir()
    for i in range(1, 6):
        (d / "sync-fixed-cost" / f"run{i}.log").write_text(
            f"test sync_fixed_cost_empty_dispatch ... SYNC_FIXED_COST_EMPTY_DISPATCH trials=200 warmup=20 median_us={290+i}.500 q1_us=280.000 q3_us=300.000 structure=cb+1,wait+1\n"
            "test result: ok. 1 passed; 0 failed\n"
        )
    (d / "sync-fixed-cost" / "infer_fixed_cost_diag.log").write_text("  batches_with_timestamps/iter min=1 max=1; wait_until_completed/iter min=2 max=2\n")
    (d / "counters").mkdir()
    (d / "counters" / "batch_counters.log").write_text("x encode_delta=11 command_buffer_delta=7 wait_delta=7 y\n")
    (d / "counters" / "backward_dinput_phase.log").write_text(
        "encode_deltas=[5, 5] command_buffer_deltas=[3, 3] wait_deltas=[3, 3]\n"
    )


def self_test():
    def expect_reject(label, **kw):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            _write_fixture(d, **{k: v for k, v in kw.items() if k in ("drop", "bad", "allrun_drop")})
            if kw.get("rm"):
                (d / kw["rm"]).unlink()
            try:
                run(d, ["cpu", "metal"], "test")
            except (AggError, ValueError, KeyError):
                return
            raise SystemExit(f"self-test 失敗: {label} が拒否されなかった")

    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        _write_fixture(d)
        out = run(d, ["cpu", "metal"], "test")
        assert "一致" in out and "7 回 × " in out and "外挿" in out, out
    expect_reject("run 欠損", rm="run3.jsonl")
    expect_reject("行（セル）欠損", drop=2)
    expect_reject("非数値", bad=4)
    expect_reject("正の無限大", bad=5)
    expect_reject("全 run 同一フェーズ欠落", allrun_drop="host_copy")
    expect_reject("load_gate.log 欠損", rm="load_gate.log")
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        _write_fixture(d, gate="timeout")
        out = run(d, ["cpu", "metal"], "test")
        assert "参考系列" in out and "正式系列" not in out, out
    expect_reject("sync ログ欠損", rm="sync-fixed-cost/run5.log")
    expect_reject("counters 欠損", rm="counters/batch_counters.log")
    print("self-test ok")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--devices", default="cpu,metal")
    ap.add_argument("--machine", default="Apple M4 (Mac mini)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.dir:
        print("error: dir が必要", file=sys.stderr)
        return 1
    devices = [x for x in a.devices.split(",") if x]
    try:
        sys.stdout.write(run(Path(a.dir), devices, a.machine))
    except AggError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
