#!/usr/bin/env python3
"""イシュー #2109 実測ログ集計（python3 標準ライブラリのみ。RULE.txt が判定規則の正）。

入力（`orchestrate.sh` の生成物）:
  - layerA-phases-N256.log / layerA-ac2-N256.log / candle-fresh-N256.log
    （bench の JSONL を `-- run i/N=256 --` 区切りで 5 run 分連結したもの）
  - layerB-run{1..5}.log（`DIAG2109 ...` 行。診断テストの出力）

fail-closed: run 数不足・phase 欠落/重複・壊れた JSON・checksum 不一致・
件数不一致・未マスクの絶対パス（/home/<user>）を検出したら
`LogIntegrityError` を送出し、正式な集計（Markdown）を一切出力せず
非ゼロ終了する。ログ未生成（実測前）は正常な中間状態として扱う。
`--self-test` は固定 fixture による自己検証（実機不要）。
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

N = 256
RUNS = 5
EXPECTED_CHECKSUM = "237.546660"
PHASES_A = ("matmul", "to_tensor", "host_copy", "checksum", "iter_total")
DIAG_RE = re.compile(
    r"^DIAG2109 n=(\d+) layer=(\w+) phase=(\w+) "
    r"median_us=([0-9.]+) q1_us=([0-9.]+) q3_us=([0-9.]+)$"
)
COUNTS_RE = re.compile(r"^DIAG2109 n=(\d+) counts (.+)$")
CKSUM_RE = re.compile(r"^DIAG2109 n=(\d+) checksum_bits=(0x[0-9a-f]{16})$")
UNMASKED_RE = re.compile(r"/home/[A-Za-z0-9_.-]+")


class LogIntegrityError(Exception):
    """完全性検査に失敗したことを表す（黙って除外せず非ゼロ終了する）。"""


def check_masked(text: str, name: str) -> None:
    m = UNMASKED_RE.search(text)
    if m:
        raise LogIntegrityError(f"{name}: 未マスクの絶対パス {m.group(0)!r}（RULE.txt 10）")


def parse_jsonl_runs(text: str, name: str, need: tuple[str, ...] | None) -> list[list[dict]]:
    """`-- run i/N=n --` 区切りの JSONL を run 単位のレコード列へ分割する。"""
    check_masked(text, name)
    runs: list[list[dict]] = []
    cur: list[dict] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            if cur is not None:
                runs.append(cur)
            cur = []
            continue
        if cur is None:
            raise LogIntegrityError(f"{name}: 区切り行より前にデータ行: {line!r}")
        if not line.startswith("{"):
            raise LogIntegrityError(f"{name}: 認識できない行: {line!r}")
        try:
            cur.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise LogIntegrityError(f"{name}: 壊れた JSON: {line!r}") from exc
    if cur is not None:
        runs.append(cur)
    if len(runs) != RUNS:
        raise LogIntegrityError(f"{name}: run 数が不一致（期待 {RUNS}・実際 {len(runs)}）")
    if need:
        for i, r in enumerate(runs, 1):
            got = [rec.get("phase") for rec in r]
            if sorted(got) != sorted(need):
                raise LogIntegrityError(f"{name}: run {i} の phase 集合が不正: {got}")
    return runs


def med(xs: list[float]) -> float:
    return statistics.median(xs)


def parse_layer_b(text: str, name: str) -> dict:
    """1 プロセス分の Layer B ログ → {(n, layer, phase): median_us, ...}。"""
    check_masked(text, name)
    out: dict = {"phases": {}, "checksum": {}, "counts": {}}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("DIAG2109 "):
            continue
        if m := DIAG_RE.match(line):
            key = (int(m[1]), m[2], m[3])
            if key in out["phases"]:
                raise LogIntegrityError(f"{name}: 区間が重複: {key}")
            out["phases"][key] = float(m[4])
        elif m := CKSUM_RE.match(line):
            out["checksum"][int(m[1])] = m[2]
        elif m := COUNTS_RE.match(line):
            out["counts"][int(m[1])] = dict(kv.split("=") for kv in m[2].split())
        elif " kernel=" in line:
            pass
        else:
            raise LogIntegrityError(f"{name}: 認識できない DIAG2109 行: {line!r}")
    if N not in out["checksum"]:
        raise LogIntegrityError(f"{name}: N={N} の checksum_bits が無い")
    if not any(k[0] == N for k in out["phases"]):
        raise LogIntegrityError(f"{name}: N={N} の区間が無い")
    return out


def expected_counts(n: int) -> dict:
    b = n * n * 4
    return {
        "driver_call_scopes": "1", "h2d_calls": "2", "h2d_bytes": str(2 * b),
        "pool_allocs": "1", "kernel_launches": "1", "d2h_calls": "1",
        "d2h_bytes": str(b), "stream_syncs": "1",
    }


def aggregate(layer_b_texts: list[str], phases_text: str, ac2_text: str, candle_text: str) -> str:
    if len(layer_b_texts) != RUNS:
        raise LogIntegrityError(f"Layer B の run 数が不一致（期待 {RUNS}・実際 {len(layer_b_texts)}）")
    bs = [parse_layer_b(t, f"layerB-run{i}") for i, t in enumerate(layer_b_texts, 1)]
    if len({b["checksum"][N] for b in bs}) != 1:
        raise LogIntegrityError("Layer B の checksum_bits が run 間で不一致（RULE.txt 2）")
    for i, b in enumerate(bs, 1):
        if b["counts"].get(N) != expected_counts(N):
            raise LogIntegrityError(f"layerB-run{i}: 件数が期待と不一致: {b['counts'].get(N)}")

    pa = parse_jsonl_runs(phases_text, "layerA-phases", PHASES_A)
    ac2 = parse_jsonl_runs(ac2_text, "layerA-ac2", None)
    cd = parse_jsonl_runs(candle_text, "candle-fresh", None)
    sums = {f"{rec['checksum']:.6f}" for r in pa for rec in r} | {
        f"{rec['checksum']:.6f}" for r in ac2 for rec in r
    }
    if sums != {EXPECTED_CHECKSUM}:
        raise LogIntegrityError(f"Layer A の checksum が {EXPECTED_CHECKSUM} と不一致: {sorted(sums)}")

    def a_med(phase: str) -> float:
        return med([next(x["median_s"] for x in r if x["phase"] == phase) for r in pa]) * 1e6

    fandhe_reuse = med([r[0]["median_s"] for r in ac2]) * 1e6
    candle = med([r[0]["median_s"] for r in cd]) * 1e6
    gap = fandhe_reuse - candle

    def b(layer: str, phase: str) -> float:
        return med([x["phases"][(N, layer, phase)] for x in bs])

    l0, l1, l2 = b("L0", "ops_total"), b("L1", "gemm_total"), b("L2", "l2_sum")
    rows = [
        ("H1 毎反復 H2D", b("L2", "h2d_a") + b("L2", "h2d_b")),
        ("H2 launch+同期の往復", b("L2", "launch_issue") + b("L2", "kernel_wait") - b("D", "dev_kernel_b2b")),
        ("H3 D2H readback", b("L2", "d2h")),
        ("H4a host dispatch (L0-L1)", l0 - l1),
        ("H4b host dispatch (L1-l2_sum)", l1 - l2),
        ("H5 facade/tape (matmul-L0)", a_med("matmul") - l0),
        ("H6 checksum", a_med("checksum")),
        ("補助 teardown", b("L2", "teardown")),
        ("補助 alloc_c", b("L2", "alloc_c")),
    ]
    lines = [
        f"## N={N} 集計（5 run 中央値, µs）", "",
        f"- fandhe reuse（AC-2）: {fandhe_reuse:.3f}／candle fresh: {candle:.3f}／gap: {gap:.3f}",
        f"- L0 ops_total: {l0:.3f}／L1 gemm_total: {l1:.3f}／L2 l2_sum: {l2:.3f}",
        f"- Layer A iter_total: {a_med('iter_total'):.3f}／matmul: {a_med('matmul'):.3f}",
        f"- floor tiny_roundtrip: {b('E', 'tiny_roundtrip'):.3f}／sync_idle: {b('E', 'sync_idle'):.3f}",
        f"- device idle 推定: {b('D', 'dev_kernel_seg') - b('D', 'dev_kernel_b2b'):.3f}"
        f"／host のみ: {max(0.0, l2 - b('D', 'dev_span')):.3f}", "",
        "| 仮説 | µs | L0 比 | gap 比 | 判定 |", "| --- | --- | --- | --- | --- |",
    ]
    for name, v in rows:
        share = v / gap if gap > 0 else float("nan")
        verdict = "支持" if gap > 0 and share >= 0.5 else "未確定"
        lines.append(f"| {name} | {v:.3f} | {v / l0 * 100:.1f}% | {share * 100:.1f}% | {verdict} |")
    lines.append("")
    return "\n".join(lines)


def _layer_b_fixture(cksum: str = "0x4062c00000000000") -> str:
    ph = {("L0", "ops_total"): 90, ("L1", "gemm_total"): 85, ("L2", "h2d_a"): 10,
          ("L2", "h2d_b"): 10, ("L2", "alloc_c"): 2, ("L2", "launch_issue"): 6,
          ("L2", "kernel_wait"): 20, ("L2", "d2h"): 15, ("L2", "teardown"): 8,
          ("L2", "l2_sum"): 71, ("D", "dev_kernel_b2b"): 8, ("D", "dev_kernel_seg"): 14,
          ("D", "dev_span"): 50, ("E", "tiny_roundtrip"): 12, ("E", "sync_idle"): 3}
    rows = [f"DIAG2109 n={N} layer={l} phase={p} median_us={v:.3f} q1_us={v:.3f} q3_us={v:.3f}"
            for (l, p), v in ph.items()]
    e = expected_counts(N)
    rows.append(f"DIAG2109 n={N} checksum_bits={cksum}")
    rows.append(f"DIAG2109 n={N} counts " + " ".join(f"{k}={v}" for k, v in e.items()))
    return "\n".join(rows) + "\n"


def _jsonl(vals: list[float], phases: tuple[str, ...] | None, cks: str = EXPECTED_CHECKSUM) -> str:
    out = []
    for i in range(RUNS):
        out.append(f"-- run {i + 1}/N={N} --")
        for j, ph in enumerate(phases or (None,)):
            rec = {"median_s": vals[j] * 1e-6, "checksum": float(cks)}
            if ph:
                rec["phase"] = ph
            out.append(json.dumps(rec))
    return "\n".join(out) + "\n"


def self_test() -> None:
    lb = [_layer_b_fixture()] * RUNS
    pa = _jsonl([70, 1, 1, 5, 92], PHASES_A)
    ac2 = _jsonl([92], None)
    cd = _jsonl([76], None)
    md = aggregate(lb, pa, ac2, cd)
    assert "gap: 16.000" in md, md

    def must_fail(fn, label: str) -> None:
        try:
            fn()
        except LogIntegrityError:
            return
        raise AssertionError(f"fail-closed でない: {label}")

    must_fail(lambda: aggregate(lb[:4], pa, ac2, cd), "Layer B run 数不足")
    must_fail(
        lambda: aggregate([_layer_b_fixture("0x4062c00000000001")] + lb[1:], pa, ac2, cd),
        "checksum_bits 不一致",
    )
    must_fail(lambda: aggregate(lb, pa, _jsonl([92], None, "1.000000"), cd), "Layer A checksum 不一致")
    must_fail(lambda: aggregate(lb, pa.replace('"matmul"', '"matmul_x"', 1), ac2, cd), "phase 不正")
    must_fail(lambda: aggregate(lb, pa + "{broken\n", ac2, cd), "壊れた JSON")
    must_fail(lambda: aggregate([lb[0] + "# /home/someone/x\n"] + lb[1:], pa, ac2, cd), "未マスク")
    bad = lb[0].replace("kernel_launches=1", "kernel_launches=2")
    must_fail(lambda: aggregate([bad] + lb[1:], pa, ac2, cd), "件数不一致")
    print("self-test OK")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log-dir", default=str(Path(__file__).parent))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    d = Path(args.log_dir)
    needed = [d / f"layerB-run{i}.log" for i in range(1, RUNS + 1)] + [
        d / f"layerA-phases-N{N}.log", d / f"layerA-ac2-N{N}.log", d / f"candle-fresh-N{N}.log"]
    if not all(p.exists() for p in needed):
        print("ログ未生成（実測前）。orchestrate.sh を GB10 で実行後に再実行すること。")
        return 0
    try:
        print(aggregate([p.read_text() for p in needed[:RUNS]], needed[RUNS].read_text(),
                        needed[RUNS + 1].read_text(), needed[RUNS + 2].read_text()))
    except LogIntegrityError as exc:
        print(f"LogIntegrityError: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
