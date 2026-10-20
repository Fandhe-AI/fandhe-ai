#!/usr/bin/env python3
"""イシュー #3182: Mac mini の Metal GEMM 計測窓の内訳（Layer A／Layer B／candle）を集計する。

規則の正は同ディレクトリの RULE.txt（事前登録の式 F1〜F7）。本スクリプトは式をそのまま出力し、
欠損・非数値・run／サイズの欠け・重複は 0 埋めせず AggError（exit 1）で拒否する。
python3 標準ライブラリのみ。使い方（`python3 -I` で実行）:
  aggregate.py <dir> > aggregate.md
  aggregate.py --self-test
入力（<dir>）:
  layerA-run{1..5}.jsonl   bench-fandhe gemm reuse --phases（task=gemm_phases。5 サイズ）
  window-reuse-run{i}.jsonl bench-fandhe gemm reuse（phases なし。task=gemm）
  fresh-run{i}.jsonl        bench-fandhe gemm fresh（task=gemm）
  candle-run{i}.jsonl       bench-candle gemm（task=gemm）＋ gemm-transfer-split（gemm_transfer_incl／excl）
  layerB-run{i}.log         診断テスト出力（[production] N=… ブロック・ROUTE_ACTUAL・DIAG_BITEQ 行）
  load_gate.log             負荷ゲート記録（run{i} gate=…／FAILED 行）
"""
import json
import math
import re
import statistics
import sys
import tempfile
from pathlib import Path

SIZES = (256, 512, 1024, 2048, 4096)
RUNS = (1, 2, 3, 4, 5)
A_PHASES = ("matmul", "to_tensor", "host_copy", "checksum", "iter_total")
B_LABELS = (
    "upload_a", "upload_b", "alloc_c", "encode", "commit_wait", "commit_wait.kernel_gpu",
    "commit_wait.commit_wait_minus_kernel_gpu", "readback", "host_copy",
)
DRIFT_FANDHE_4096_MS = 122.82  # framework-compare-0.11.0-macmini-remeasure（系列 B）
DRIFT_CANDLE_4096_MS = 71.19
DRIFT_LIMIT = 0.10

# M4 Max・fandhe-ai 0.6.0 の参考値（別機体・別版。算術には混ぜない）。出典:
# docs/perf/metal-gemm-reuse-phase-breakdown.md §3（Layer A 中央値の中央値）・§11.2（kernel_gpu／commit_wait）
M4MAX_060 = {
    1024: {"matmul": 2.039, "host_copy": 0.2397, "checksum": 0.5693, "iter_total": 2.859,
           "commit_wait": 1.2606, "kernel_gpu": 1.0267},
    2048: {"matmul": 5.549, "host_copy": 0.9196, "checksum": 2.2682, "iter_total": 8.622,
           "commit_wait": 3.8869, "kernel_gpu": 3.1849},
    4096: {"matmul": 34.800, "host_copy": 3.7191, "checksum": 9.6276, "iter_total": 47.944,
           "commit_wait": 15.7892, "kernel_gpu": 13.7051},
}


class AggError(Exception):
    pass


def num(v, what, signed=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or (v < 0 and not signed):
        raise AggError(f"非数値または負値: {what}={v!r}")
    return float(v)


def load_jsonl(path, framework):
    """(task, mode, size, phase) -> (median_s, checksum)。重複・不正行は拒否。"""
    if not path.is_file():
        raise AggError(f"欠損: {path.name}")
    out = {}
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            raise AggError(f"{path.name}:{ln} JSON 不正: {e}")
        if r.get("framework") != framework or r.get("device") != "metal":
            continue
        key = (r.get("task"), r.get("mode"), r.get("size"), r.get("phase"))
        if key in out:
            raise AggError(f"{path.name}:{ln} 重複: {key}")
        out[key] = (num(r.get("median_s"), f"{path.name}:{ln} median_s"),
                    num(r.get("checksum"), f"{path.name}:{ln} checksum", signed=True) if "checksum" in r else None)
    return out


def get(rows, key, name):
    if key not in rows:
        raise AggError(f"欠損: {name} {key}")
    return rows[key]


HDR = re.compile(r"\[production\] N=(\d+) requested_tile=.*? resolved_tile=(.*?) \(median over")
MET = re.compile(r"^\s+(\S+): median=([^\s]+) ms")
ACT = re.compile(r"ROUTE_ACTUAL N=(\d+) route=(Classic|SplitK)")
EQ = re.compile(r"DIAG_BITEQ N=(\d+) equal=(true|false)")


def parse_layer_b(path):
    """-> (blocks{n: {label: ms}}, resolved{n: str}, route{n: str}, biteq{n: bool})。"""
    if not path.is_file():
        raise AggError(f"欠損: {path.name}")
    blocks, resolved, route, biteq = {}, {}, {}, {}
    cur = None
    for line in path.read_text().splitlines():
        m = HDR.search(line)
        if m:
            cur = int(m.group(1))
            if cur in blocks:
                raise AggError(f"{path.name}: N={cur} のブロックが重複")
            blocks[cur] = {}
            resolved[cur] = m.group(2)
            continue
        m = ACT.search(line)
        if m:
            n = int(m.group(1))
            if n in route:
                raise AggError(f"{path.name}: ROUTE_ACTUAL N={n} 重複")
            route[n] = m.group(2)
            cur = None
            continue
        m = EQ.search(line)
        if m:
            n = int(m.group(1))
            if n in biteq:
                raise AggError(f"{path.name}: DIAG_BITEQ N={n} 重複")
            biteq[n] = m.group(2) == "true"
            cur = None
            continue
        m = MET.match(line)
        if m and cur is not None:
            try:
                v = float(m.group(2))
            except ValueError:
                raise AggError(f"{path.name}: 非数値 {m.group(1)}={m.group(2)}")
            num(v, f"{path.name} {m.group(1)}")
            if m.group(1) in blocks[cur]:
                raise AggError(f"{path.name}: N={cur} {m.group(1)} 重複")
            blocks[cur][m.group(1)] = v
    for n in SIZES:
        if n not in blocks:
            raise AggError(f"{path.name}: N={n} のブロック欠損")
        for lab in B_LABELS:
            if lab not in blocks[n]:
                raise AggError(f"{path.name}: N={n} の {lab} 欠損")
        if n not in route or n not in biteq:
            raise AggError(f"{path.name}: N={n} の ROUTE_ACTUAL／DIAG_BITEQ 欠損")
    return blocks, resolved, route, biteq


def med(xs):
    return statistics.median(xs)


def rng(xs):
    return f"{min(xs):.4f}–{max(xs):.4f}"


def read_gate(d):
    p = d / "load_gate.log"
    if not p.is_file():
        raise AggError("欠損: load_gate.log")
    gates, failed = {}, []
    for line in p.read_text().splitlines():
        m = re.match(r"run(\d) gate=(\S+) ", line)
        if m:
            gates[int(m.group(1))] = m.group(2)
        if "FAILED" in line:
            failed.append(line)
    for i in RUNS:
        if i not in gates:
            raise AggError(f"load_gate.log に run{i} の gate 記録が無い")
    if failed:
        raise AggError("計測失敗の記録あり: " + "; ".join(failed))
    return gates


def aggregate(d):
    d = Path(d)
    gates = read_gate(d)
    A, WR, FR, CF, CX, CI, B, RT, EQV, RES = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    for i in RUNS:
        la = load_jsonl(d / f"layerA-run{i}.jsonl", "fandhe-ai")
        wr = load_jsonl(d / f"window-reuse-run{i}.jsonl", "fandhe-ai")
        fr = load_jsonl(d / f"fresh-run{i}.jsonl", "fandhe-ai")
        ca = load_jsonl(d / f"candle-run{i}.jsonl", "candle")
        for n in SIZES:
            for ph in A_PHASES:
                v, ck = get(la, ("gemm_phases", "reuse", n, ph), f"layerA run{i} N={n}")
                A[(i, n, ph)] = v * 1e3
                if ph == "iter_total":
                    A[(i, n, "ck")] = ck
            WR[(i, n)], WRck = get(wr, ("gemm", "reuse", n, None), f"window-reuse run{i} N={n}")
            WR[(i, n)] *= 1e3
            A[(i, n, "wr_ck")] = WRck
            FR[(i, n)] = get(fr, ("gemm", "fresh", n, None), f"fresh run{i} N={n}")[0] * 1e3
            CF[(i, n)] = get(ca, ("gemm", "fresh", n, None), f"candle gemm run{i} N={n}")[0] * 1e3
            CX[(i, n)] = get(ca, ("gemm_transfer_excl", "fresh", n, None), f"candle excl run{i} N={n}")[0] * 1e3
            CI[(i, n)] = get(ca, ("gemm_transfer_incl", "fresh", n, None), f"candle incl run{i} N={n}")[0] * 1e3
        blocks, resolved, route, biteq = parse_layer_b(d / f"layerB-run{i}.log")
        for n in SIZES:
            for lab in B_LABELS:
                B[(i, n, lab)] = blocks[n][lab]
            RT[(i, n)], EQV[(i, n)], RES[(i, n)] = route[n], biteq[n], resolved[n]

    L = []
    p = L.append
    formal = all(g == "pass" for g in gates.values())
    p("# イシュー #3182 集計（aggregate.py 機械出力。RULE.txt の式 F1〜F7）\n")
    p(f"負荷ゲート: {', '.join(f'run{i}={gates[i]}' for i in RUNS)} → "
      + ("**5/5 pass（正式系列）**" if formal else "**部分通過（参考扱い）**") + "\n")

    # ---- 経路の前提 ----
    p("## 経路の前提（Layer B の代表性。RULE.txt 9）\n")
    p("| N | ROUTE_ACTUAL（5 run） | DIAG_BITEQ（5 run） | resolved_tile（run1） | Layer B 代表 |")
    p("|---|---|---|---|---|")
    rep = {}
    for n in SIZES:
        rs = sorted({RT[(i, n)] for i in RUNS})
        es = sorted({EQV[(i, n)] for i in RUNS})
        rep[n] = rs == ["Classic"] and es == [True]
        p(f"| {n} | {'/'.join(rs)} | {'/'.join(str(e).lower() for e in es)} | `{RES[(1, n)]}` | "
          + ("はい" if rep[n] else "**いいえ（非代表）**") + " |")
    p("")

    # ---- Layer A ----
    p("## Layer A（bench-fandhe 0.11.0 `gemm reuse --phases`。単位 ms）\n")
    p("| N | run | matmul | to_tensor | host_copy | checksum | iter_total |")
    p("|---|---|---|---|---|---|---|")
    for n in SIZES:
        for i in RUNS:
            p(f"| {n} | {i} | " + " | ".join(f"{A[(i, n, ph)]:.4f}" for ph in A_PHASES) + " |")
    p("")
    p("**F1: 中央値の中央値（min–max）**\n")
    p("| N | matmul | to_tensor | host_copy | checksum | iter_total |")
    p("|---|---|---|---|---|---|")
    AM = {}
    for n in SIZES:
        cells = []
        for ph in A_PHASES:
            xs = [A[(i, n, ph)] for i in RUNS]
            AM[(n, ph)] = med(xs)
            cells.append(f"{AM[(n, ph)]:.4f}（{rng(xs)}）")
        p(f"| {n} | " + " | ".join(cells) + " |")
    p("")

    # ---- Layer B ----
    p("## Layer B（backend-metal 診断テスト・HEAD ソース。単位 ms）\n")
    p("| N | run | upload_a | upload_b | alloc_c | encode | commit_wait | kernel_gpu | cw−kg | readback | host_copy | Σ（host_copy 除く） |")
    p("|---|---|---|---|---|---|---|---|---|---|---|---|")
    SUM = {}
    for n in SIZES:
        for i in RUNS:
            s = sum(B[(i, n, k)] for k in ("upload_a", "upload_b", "alloc_c", "encode", "commit_wait", "readback"))
            SUM[(i, n)] = s
            p(f"| {n} | {i} | " + " | ".join(f"{B[(i, n, k)]:.4f}" for k in (
                "upload_a", "upload_b", "alloc_c", "encode", "commit_wait", "commit_wait.kernel_gpu",
                "commit_wait.commit_wait_minus_kernel_gpu", "readback", "host_copy")) + f" | {s:.4f} |")
    p("")
    p("**F1: 中央値の中央値（min–max）**\n")
    p("| N | upload(a+b) | commit_wait | kernel_gpu | cw−kg | readback | host_copy | encode | alloc_c | Σ（run 合計の中央値） |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    BM = {}
    for n in SIZES:
        up = [B[(i, n, "upload_a")] + B[(i, n, "upload_b")] for i in RUNS]
        BM[(n, "upload")] = med(up)
        cells = [f"{med(up):.4f}（{rng(up)}）"]
        for k, lab in (("commit_wait", "commit_wait"), ("kernel_gpu", "commit_wait.kernel_gpu"),
                       ("cwkg", "commit_wait.commit_wait_minus_kernel_gpu"), ("readback", "readback"),
                       ("host_copy", "host_copy"), ("encode", "encode"), ("alloc_c", "alloc_c")):
            xs = [B[(i, n, lab)] for i in RUNS]
            BM[(n, k)] = med(xs)
            cells.append(f"{med(xs):.4f}（{rng(xs)}）")
        sx = [SUM[(i, n)] for i in RUNS]
        BM[(n, "sum")] = med(sx)
        cells.append(f"{med(sx):.4f}（{rng(sx)}）")
        p(f"| {n} | " + " | ".join(cells) + " |")
    p("")
    p("A の matmul と B の Σ の残差（未説明。原因は断定しない）:\n")
    p("| N | A matmul | B Σ | 残差（A−B） |")
    p("|---|---|---|---|")
    for n in SIZES:
        p(f"| {n} | {AM[(n, 'matmul')]:.4f} | {BM[(n, 'sum')]:.4f} | {AM[(n, 'matmul')] - BM[(n, 'sum')]:+.4f} |")
    p("")

    # ---- F2 ----
    p("## F2: checksum の ns／要素（A の checksum 区間中央値 / N²）\n")
    p("| N | checksum (ms) | ns／要素 |")
    p("|---|---|---|")
    for n in SIZES:
        p(f"| {n} | {AM[(n, 'checksum')]:.4f} | {AM[(n, 'checksum')] * 1e6 / (n * n):.4f} |")
    p("")

    # ---- F3 ----
    p("## F3: 割合表（分母は A の iter_total。A（0.11.0）と B（HEAD）は版が違う）\n")
    p("| N | kernel_gpu(B) | upload(B) | host_copy(A) | checksum(A) | to_tensor(A) | その他(B) | 未説明 | iter_total(A) ms | B 代表 |")
    p("|---|---|---|---|---|---|---|---|---|---|")
    F3 = {}
    for n in SIZES:
        other = [B[(i, n, "alloc_c")] + B[(i, n, "encode")]
                 + B[(i, n, "commit_wait.commit_wait_minus_kernel_gpu")] + B[(i, n, "readback")] for i in RUNS]
        parts = {
            "kernel_gpu": BM[(n, "kernel_gpu")], "upload": BM[(n, "upload")],
            "host_copy": AM[(n, "host_copy")], "checksum": AM[(n, "checksum")],
            "to_tensor": AM[(n, "to_tensor")], "other": med(other),
        }
        tot = AM[(n, "iter_total")]
        parts["unexplained"] = tot - sum(parts.values())
        F3[n] = parts
        p(f"| {n} | " + " | ".join(f"{parts[k]:.4f}（{parts[k] / tot * 100:.1f}%）" for k in (
            "kernel_gpu", "upload", "host_copy", "checksum", "to_tensor", "other", "unexplained"))
          + f" | {tot:.4f} | " + ("はい" if rep[n] else "いいえ") + " |")
    p("\n注: 各列は中央値のため合計は iter_total と一致しない。差は「未説明」に入る（吸収させない）。\n")

    # ---- F4 ----
    p("## F4: 窓合計（phases なし。単位 ms）\n")
    p("| N | fresh | reuse | reuse−fresh | 出力 MiB | (reuse−fresh)/MiB |")
    p("|---|---|---|---|---|---|")
    WRm, FRm = {}, {}
    for n in SIZES:
        WRm[n] = med([WR[(i, n)] for i in RUNS])
        FRm[n] = med([FR[(i, n)] for i in RUNS])
        mib = n * n * 4 / 2**20
        p(f"| {n} | {FRm[n]:.4f}（{rng([FR[(i, n)] for i in RUNS])}） | {WRm[n]:.4f}（{rng([WR[(i, n)] for i in RUNS])}） "
          f"| {WRm[n] - FRm[n]:+.4f} | {mib:.1f} | {(WRm[n] - FRm[n]) / mib:+.4f} |")
    p("")

    # ---- AC4 ----
    p("## AC4 照合（同一 run・同一 N の checksum: layerA vs window-reuse）\n")
    bad = [(i, n) for i in RUNS for n in SIZES if A[(i, n, "ck")] != A[(i, n, "wr_ck")]]
    p("全 25 点で一致（一致は bit 同一の証明ではない）\n" if not bad else f"**不一致: {bad}**\n")

    # ---- F5/F6/F7 ----
    p("## F5〜F7: candle との差とカーネル由来の割合\n")
    p("| N | fandhe reuse | candle fresh | G=差 | candle excl | (a) (kernel_gpu−excl)/G | (b) ((encode+commit_wait)−excl)/G | F7 固有(upload+host_copy) | F7/G | checksum(共通) | 注記 |")
    p("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in SIZES:
        cf = med([CF[(i, n)] for i in RUNS])
        cx = med([CX[(i, n)] for i in RUNS])
        g = WRm[n] - cf
        kg = BM[(n, "kernel_gpu")]
        ecw = BM[(n, "encode")] + BM[(n, "commit_wait")]
        own = BM[(n, "upload")] + AM[(n, "host_copy")]
        notes = []
        if not rep[n]:
            notes.append("B 非代表")
        if g <= 0:
            a = b = f7 = "算出不能（G<=0）"
        else:
            vals = []
            for v in ((kg - cx) / g, (ecw - cx) / g, own / g):
                vals.append(f"{v * 100:.1f}%" + ("（負値）" if v < 0 else "（1 超）" if v > 1 else ""))
            a, b, f7 = vals
        p(f"| {n} | {WRm[n]:.4f} | {cf:.4f} | {g:+.4f} | {cx:.4f} | {a} | {b} | {own:.4f} | {f7} | {AM[(n, 'checksum')]:.4f} | {'; '.join(notes)} |")
    p("\n限界: candle の transfer_excl は壁時計（matmul＋synchronize）、kernel_gpu は GPU タイムスタンプのため (a) はカーネル差を小さく見積もりやすい。"
      "M4 Max・0.6.0 の §11.4(b)（candle 13.17 TFLOPS・27.6%）は別機体・別版で、算術には混ぜない。\n")

    # ---- ドリフト ----
    p("## ドリフト検査（N=4096。系列 B との乖離）\n")
    cf4 = med([CF[(i, 4096)] for i in RUNS])
    for name, cur, ref in (("fandhe reuse", WRm[4096], DRIFT_FANDHE_4096_MS), ("candle fresh", cf4, DRIFT_CANDLE_4096_MS)):
        dev = (cur - ref) / ref
        p(f"- {name}: {cur:.2f} ms vs 系列 B {ref:.2f} ms → {dev * 100:+.1f}%" + ("（**10% 超**）" if abs(dev) > DRIFT_LIMIT else ""))
    p("")

    # ---- M4 Max 並置 ----
    p("## M4 Max・fandhe-ai 0.6.0 との並置（別機体・別版。混ぜない。出典: metal-gemm-reuse-phase-breakdown.md §3・§11.2）\n")
    p("| N | 項目 | M4 Max 0.6.0 | Mac mini（A: 0.11.0／B: HEAD） |")
    p("|---|---|---|---|")
    for n in (1024, 2048, 4096):
        ref = M4MAX_060[n]
        for k, cur in (("iter_total", AM[(n, "iter_total")]), ("matmul", AM[(n, "matmul")]),
                       ("host_copy", AM[(n, "host_copy")]), ("checksum", AM[(n, "checksum")]),
                       ("commit_wait", BM[(n, "commit_wait")]), ("kernel_gpu", BM[(n, "kernel_gpu")])):
            p(f"| {n} | {k} | {ref[k]:.4f} | {cur:.4f} |")
    p("")
    return "\n".join(L)


def _self_test():
    import shutil

    def build(d, runs=RUNS, break_kind=None):
        d.mkdir(parents=True, exist_ok=True)
        (d / "load_gate.log").write_text("".join(f"run{i} gate=pass load1=1.0 waited_s=0 at=x\n" for i in RUNS))
        for i in runs:
            la, wr, fr, ca, lb = [], [], [], [], []
            for n in SIZES:
                base = (n / 1024) ** 3 * 1e-3
                for idx, ph in enumerate(A_PHASES):
                    la.append({"framework": "fandhe-ai", "device": "metal", "task": "gemm_phases", "mode": "reuse",
                               "size": n, "phase": ph, "median_s": base * (idx + 1), "checksum": 1.5})
                wr.append({"framework": "fandhe-ai", "device": "metal", "task": "gemm", "mode": "reuse", "size": n,
                           "median_s": base * 6, "checksum": 1.5})
                fr.append({"framework": "fandhe-ai", "device": "metal", "task": "gemm", "mode": "fresh", "size": n,
                           "median_s": base * 5, "checksum": 1.5})
                for t, k in (("gemm", 3), ("gemm_transfer_excl", 2), ("gemm_transfer_incl", 4)):
                    ca.append({"framework": "candle", "device": "metal", "task": t, "mode": "fresh", "size": n,
                               "median_s": base * k, "checksum": 1.5})
                lb.append(f"test x ... ROUTE N={n} route=Classic(..) resolved_tile=T")
                lb.append(f"ROUTE_ACTUAL N={n} route=Classic(..)")
                lb.append(f"DIAG_BITEQ N={n} equal=true")
                lb.append(f"  [production] N={n} requested_tile=T resolved_tile=T {{ a: 1 }} (median over 20 trials, 20 warmup):")
                for lab in B_LABELS:
                    v = "nanx" if (break_kind == "nonnum" and lab == "encode" and n == 256) else f"{base * 1e3 * 0.1:.4f}"
                    lb.append(f"    {lab}: median={v} ms  q1=0 ms  q3=0 ms")
                lb.append("    sum of medians: 1.0000 ms")
            for name, rows in (("layerA", la), ("window-reuse", wr), ("fresh", fr), ("candle", ca)):
                (d / f"{name}-run{i}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (d / f"layerB-run{i}.log").write_text("\n".join(lb) + "\n")

    def expect_fail(label, fn):
        try:
            fn()
        except AggError:
            return
        raise AssertionError(f"{label}: AggError が出なかった")

    root = Path(tempfile.mkdtemp(prefix="agg3182-"))
    try:
        ok = root / "ok"
        build(ok)
        out = aggregate(ok)
        assert "F2" in out and "5/5 pass" in out and "G=差" in out
        short = root / "short"
        build(short, runs=(1, 2, 3, 4))
        expect_fail("5 run 未満", lambda: aggregate(short))
        nn = root / "nonnum"
        build(nn, break_kind="nonnum")
        expect_fail("非数値", lambda: aggregate(nn))
        miss = root / "miss"
        build(miss)
        p = miss / "layerA-run3.jsonl"
        p.write_text("".join(l + "\n" for l in p.read_text().splitlines() if '"size": 512' not in l))
        expect_fail("サイズ欠損", lambda: aggregate(miss))
        dup = root / "dup"
        build(dup)
        p = dup / "fresh-run2.jsonl"
        first = p.read_text().splitlines()[0]
        p.write_text(p.read_text() + first + "\n")
        expect_fail("重複", lambda: aggregate(dup))
        failed = root / "failed"
        build(failed)
        with open(failed / "load_gate.log", "a") as f:
            f.write("run2 layerB FAILED\n")
        expect_fail("FAILED 記録", lambda: aggregate(failed))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print("self-test OK")


def main(argv):
    if argv[1:] == ["--self-test"]:
        _self_test()
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        print(aggregate(argv[1]))
    except AggError as e:
        print(f"集計拒否: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
