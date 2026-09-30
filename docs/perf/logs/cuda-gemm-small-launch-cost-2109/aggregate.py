#!/usr/bin/env python3
"""イシュー #2109 実測ログ集計（python3 標準ライブラリのみ。RULE.txt が判定規則の正）。

入力（`orchestrate.sh` の生成物）:
  - layerA-phases-N256.log / layerA-ac2-N256.log / candle-fresh-N256.log
    （registry =0.9.0。ratio の分母）と head-phases-N256.log / head-ac2-N256.log
    （HEAD path-patch。H5 の正式な突合相手。RULE.txt 4）
    （bench の JSONL を `-- run i/N=256 --` 区切りで 5 run 分連結したもの）
  - layerB-run{1..5}.log（`DIAG2109 ...` 行。診断テストの出力）
  - counts-exact.log（`gemm_small_launch_counts_exact` の出力。RULE.txt 3 の hard 条件。
    `test result: ok. 1 passed; 0 failed` を含まなければ fail-closed）

fail-closed: run 数不足・phase 欠落/重複・壊れた JSON・checksum 不一致・
件数不一致・未マスクの絶対パス（/home/<user>）・load_gate.log の欠落や
書式不正を検出したら `LogIntegrityError` を送出し、正式な集計（Markdown）を
一切出力せず非ゼロ終了する。全ログ未生成（実測前）のときだけ正常な中間状態
として exit 0 とし、一部だけ欠けている場合は欠測として exit 非 0（RULE.txt 1）。
専有ゲート FAIL の run が 1 つでもある系列は「参考扱い」とし、仮説の
「支持」判定を出さない（RULE.txt 9）。
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


RUN_SEP_RE = re.compile(r"^--\s*run (\d+)/N=(\d+)\s*--$")


def parse_jsonl_runs(text: str, name: str, need: tuple[str, ...] | None,
                     expect: dict | None = None) -> list[list[dict]]:
    """`-- run i/N=n --` 区切りの JSONL を run 単位のレコード列へ分割する。

    区切り行は run 番号が 1..RUNS の連番・N=256 であること、各レコードは
    `expect`（task/device/size/mode の期待値）と一致することを検証する
    （RULE.txt 1・5。別条件のログを N=256 計測として集計しない）。
    """
    check_masked(text, name)
    runs: list[list[dict]] = []
    cur: list[dict] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--"):
            m = RUN_SEP_RE.match(line)
            if not m:
                raise LogIntegrityError(f"{name}: 区切り行の書式が不正: {line!r}")
            if int(m[2]) != N:
                raise LogIntegrityError(f"{name}: 区切り行の N が {N} でない: {line!r}")
            if cur is not None:
                runs.append(cur)
            if int(m[1]) != len(runs) + 1:
                raise LogIntegrityError(
                    f"{name}: run 番号が連番でない（期待 {len(runs) + 1}・実際 {m[1]}）: {line!r}")
            cur = []
            continue
        if cur is None:
            raise LogIntegrityError(f"{name}: 区切り行より前にデータ行: {line!r}")
        if not line.startswith("{"):
            raise LogIntegrityError(f"{name}: 認識できない行: {line!r}")
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LogIntegrityError(f"{name}: 壊れた JSON: {line!r}") from exc
        if not isinstance(rec, dict):
            raise LogIntegrityError(f"{name}: レコードがオブジェクトでない: {line!r}")
        for key, want in (expect or {}).items():
            if rec.get(key) != want:
                raise LogIntegrityError(
                    f"{name}: run {len(runs) + 1} のレコード {key} が期待と不一致"
                    f"（期待 {want!r}・実際 {rec.get(key)!r}）")
        cur.append(rec)
    if cur is not None:
        runs.append(cur)
    if len(runs) != RUNS:
        raise LogIntegrityError(f"{name}: run 数が不一致（期待 {RUNS}・実際 {len(runs)}）")
    for i, r in enumerate(runs, 1):
        # 各 run のレコード数は期待値ちょうど（phases なしは 1 件。重複行・余剰行を採用しない）
        want = len(need) if need else 1
        if len(r) != want:
            raise LogIntegrityError(f"{name}: run {i} のレコード数が不正（期待 {want}・実際 {len(r)}）")
        if need:
            got = [rec.get("phase") for rec in r]
            if sorted(got) != sorted(need):
                raise LogIntegrityError(f"{name}: run {i} の phase 集合が不正: {got}")
    return runs


def med(xs: list[float]) -> float:
    return statistics.median(xs)


LAYER_B_TEST = "gemm_small_launch_cost_diag_tests::gemm_small_launch_cost_diag"
# `--nocapture` ではテスト名行の後ろに標準出力が続くため、テスト名行は行頭一致のみで判定し、
# 成功は `test result: ok. 1 passed; 0 failed;` で確認する（失敗・パニックの痕跡も拒否する）。
LAYER_B_NAME_RE = re.compile(r"^test " + re.escape(LAYER_B_TEST) + r" \.\.\.", re.M)
LAYER_B_FAIL_RE = re.compile(r"(^test result: FAILED|\.\.\. FAILED|panicked at)", re.M)


def check_layer_b_success(text: str, name: str) -> None:
    """Layer B ログが診断テストの成功終了を示すことを検証する（fail-closed）。

    N=256 の行を出した後に N=512 でテストが失敗しても DIAG2109 行だけは揃うため、
    対象テストの実行行と成功した `test result` を必須にする（未完走の実測を採用しない）。
    """
    if (not LAYER_B_NAME_RE.search(text) or not TEST_RESULT_OK_RE.search(text)
            or LAYER_B_FAIL_RE.search(text)):
        raise LogIntegrityError(
            f"{name}: {LAYER_B_TEST} の成功（test result: ok. 1 passed）を確認できない")


def parse_layer_b(text: str, name: str) -> dict:
    """1 プロセス分の Layer B ログ → {(n, layer, phase): median_us, ...}。"""
    check_masked(text, name)
    check_layer_b_success(text, name)
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


COUNTS_TEST = "gemm_small_launch_cost_diag_tests::gemm_small_launch_counts_exact"
TEST_RESULT_OK_RE = re.compile(r"^test result: ok\. 1 passed; 0 failed; 0 ignored;", re.M)
TEST_LINE_OK_RE = re.compile(r"^test " + re.escape(COUNTS_TEST) + r" \.\.\. ok$", re.M)


def check_counts_exact(text: str) -> None:
    """counts-exact.log が件数厳密テストの成功を示すことを検証する（RULE.txt 3・fail-closed）。

    `cargo test` の出力に当該テストの `... ok` 行と `test result: ok. 1 passed; 0 failed;
    0 ignored;` の両方が必要（テスト未実施・失敗・0 件実行・別テストの成功では通さない）。
    """
    check_masked(text, "counts-exact")
    if not TEST_LINE_OK_RE.search(text) or not TEST_RESULT_OK_RE.search(text):
        raise LogIntegrityError(
            "counts-exact.log: gemm_small_launch_counts_exact の成功（ok / 1 passed）を確認できない"
            "（RULE.txt 3）")


GATE_RE = re.compile(r"^(\S+ run\d+) (PASS|FAIL\(参考扱い\)) load1=(\S+) gpu_util=(\S+)$")
GATE_LABELS = ("layerA-phases", "ac2", "head-phases", "head-ac2", "candle", "layerB")
GATE_EXPECTED = {f"{lab} run{i}" for lab in GATE_LABELS for i in range(1, RUNS + 1)}


def parse_gate(text: str) -> bool:
    """load_gate.log を検証し、全 run が専有ゲート PASS なら True（FAIL があれば False）。

    ゲート行は Layer A phases・AC-2・candle・Layer B × run1..5 の計 30 行と
    ラベル集合が完全一致することを要求する（欠落・重複・未知ラベルは fail-closed）。
    PASS 行は実測値が load1 < 1.0 かつ gpu_util == 0 であること、FAIL 行は
    その条件を実際に満たさないことを検証し、矛盾は LogIntegrityError とする
    （RULE.txt 9。文字列 PASS だけで専有を信用しない）。
    """
    check_masked(text, "load_gate")
    ok = True
    seen: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = GATE_RE.match(line)
        if not m:
            raise LogIntegrityError(f"load_gate: 認識できない行: {line!r}")
        label, verdict = m[1], m[2]
        try:
            load1 = float(m[3])
            util = int(m[4])
        except ValueError as exc:
            raise LogIntegrityError(f"load_gate: 実測値が数値でない: {line!r}") from exc
        meets = load1 < 1.0 and util == 0
        if verdict == "PASS" and not meets:
            raise LogIntegrityError(f"load_gate: PASS だが実測値が専有条件を満たさない: {line!r}")
        if verdict != "PASS" and meets:
            raise LogIntegrityError(f"load_gate: FAIL だが実測値が専有条件を満たす（矛盾）: {line!r}")
        seen.append(label)
        if verdict != "PASS":
            ok = False
    if len(seen) != len(set(seen)):
        raise LogIntegrityError("load_gate: ラベルが重複")
    if set(seen) != GATE_EXPECTED:
        missing = sorted(GATE_EXPECTED - set(seen))
        extra = sorted(set(seen) - GATE_EXPECTED)
        raise LogIntegrityError(f"load_gate: ラベル集合が期待と不一致（欠落 {missing}・余剰 {extra}）")
    return ok


def expected_counts(n: int) -> dict:
    b = n * n * 4
    return {
        "driver_call_scopes": "1", "h2d_calls": "2", "h2d_bytes": str(2 * b),
        "pool_allocs": "1", "kernel_launches": "1", "d2h_calls": "1",
        "d2h_bytes": str(b), "stream_syncs": "1",
    }


def aggregate(layer_b_texts: list[str], phases_text: str, ac2_text: str, candle_text: str,
              gate_text: str, head_phases_text: str, head_ac2_text: str,
              counts_exact_text: str, nsys_text: str | None = None) -> str:
    gate_ok = parse_gate(gate_text)
    check_counts_exact(counts_exact_text)
    if len(layer_b_texts) != RUNS:
        raise LogIntegrityError(f"Layer B の run 数が不一致（期待 {RUNS}・実際 {len(layer_b_texts)}）")
    bs = [parse_layer_b(t, f"layerB-run{i}") for i, t in enumerate(layer_b_texts, 1)]
    if len({b["checksum"][N] for b in bs}) != 1:
        raise LogIntegrityError("Layer B の checksum_bits が run 間で不一致（RULE.txt 2）")
    for i, b in enumerate(bs, 1):
        if b["counts"].get(N) != expected_counts(N):
            raise LogIntegrityError(f"layerB-run{i}: 件数が期待と不一致: {b['counts'].get(N)}")

    base = {"device": "cuda", "size": N}
    pa = parse_jsonl_runs(phases_text, "layerA-phases", PHASES_A,
                          {**base, "task": "gemm_phases", "mode": "reuse"})
    ac2 = parse_jsonl_runs(ac2_text, "layerA-ac2", None, {**base, "task": "gemm", "mode": "reuse"})
    hpa = parse_jsonl_runs(head_phases_text, "head-phases", PHASES_A,
                           {**base, "task": "gemm_phases", "mode": "reuse"})
    hac2 = parse_jsonl_runs(head_ac2_text, "head-ac2", None, {**base, "task": "gemm", "mode": "reuse"})
    cd = parse_jsonl_runs(candle_text, "candle-fresh", None,
                          {**base, "task": "gemm", "mode": "fresh"})
    sums = {f"{rec['checksum']:.6f}" for runs_ in (pa, ac2, hpa, hac2) for r in runs_ for rec in r}
    if sums != {EXPECTED_CHECKSUM}:
        raise LogIntegrityError(f"Layer A の checksum が {EXPECTED_CHECKSUM} と不一致: {sorted(sums)}")

    def a_runs(phase: str, src=pa) -> list[float]:
        return [next(x["median_s"] for x in r if x["phase"] == phase) * 1e6 for r in src]

    def a_med(phase: str, src=hpa) -> float:
        # 既定は HEAD path-patch 系列（gap・H5 と同一ツリー。registry 値は src=pa で参考併記のみ）。
        return med(a_runs(phase, src))

    fandhe_reuse = med([r[0]["median_s"] for r in ac2]) * 1e6
    head_reuse = med([r[0]["median_s"] for r in hac2]) * 1e6
    candle = med([r[0]["median_s"] for r in cd]) * 1e6
    # 帰属の分母は HEAD path-patch の reuse（Layer B・H5 と同一ツリー。RULE.txt 5）。
    # registry =0.9.0 との差は版差を含むため、参考値としてのみ併記する。
    gap = head_reuse - candle
    gap_registry = fandhe_reuse - candle

    def b_runs(layer: str, phase: str) -> list[float]:
        return [x["phases"][(N, layer, phase)] for x in bs]

    def b(layer: str, phase: str) -> float:
        return med(b_runs(layer, phase))

    def rng(xs: list[float]) -> str:
        return f"{min(xs):.3f}–{max(xs):.3f}"

    l0, l1, l2 = b("L0", "ops_total"), b("L1", "gemm_total"), b("L2", "l2_sum")
    def rr(*terms: tuple[int, list[float]]) -> list[float]:
        """run 単位の系列を要素ごとに符号付き加算して run 毎の導出値にする。"""
        return [sum(sg * xs[i] for sg, xs in terms) for i in range(RUNS)]

    l0r, l1r, l2r = b_runs("L0", "ops_total"), b_runs("L1", "gemm_total"), b_runs("L2", "l2_sum")
    rows = [
        ("H1 毎反復 H2D", rr((1, b_runs("L2", "h2d_a")), (1, b_runs("L2", "h2d_b")))),
        ("H2 launch+同期の往復（推定）", rr((1, b_runs("L2", "launch_issue")), (1, b_runs("L2S", "kernel_wait")),
                                  (-1, b_runs("D", "dev_kernel_b2b")))),
        ("H3 D2H readback", b_runs("L2S", "d2h")),
        ("H4a host dispatch (L0-L1)", rr((1, l0r), (-1, l1r))),
        ("H4b host dispatch (L1-l2_sum)", rr((1, l1r), (-1, l2r))),
        # H5 の突合相手は HEAD path-patch の Layer A（Layer B・L0 と同一ツリー。RULE.txt 4）。
        # registry =0.9.0 との差は v0.9.0..HEAD の版差を含むため H5 の帰属に使わない。
        ("H5 facade/tape (HEAD matmul-L0)", rr((1, a_runs("matmul", hpa)), (-1, l0r))),
        ("参考 registry =0.9.0 matmul-L0（版差を含む・帰属に使わない）", rr((1, a_runs("matmul")), (-1, l0r))),
        ("H6 checksum（記録のみ）", a_runs("checksum", hpa)),
        ("補助 teardown", b_runs("L2", "teardown")),
        ("補助 driver_scope（本番 with_driver_call 入退場）", b_runs("L2", "driver_scope")),
        ("補助 alloc_c", b_runs("L2", "alloc_c")),
        # RULE.txt 6: cudarc 内部（event・async alloc）費用の補助量。run 毎の差分から中央値と min–max を出す。
        ("補助 h2d_clone_drop − h2d_prealloc（cudarc 内部の確保・解放）",
         rr((1, b_runs("E", "h2d_clone_drop")), (-1, b_runs("E", "h2d_prealloc")))),
    ]
    lines = [
        f"## N={N} 集計（5 run 中央値, µs。括弧内は 5 run 間の min–max）", "",
        "" if gate_ok else "> 専有ゲート FAIL の run を含む系列のため **参考扱い**（RULE.txt 9）。"
        "仮説の「支持」判定は出さない。\n",
        f"- HEAD path-patch reuse（AC-2）: {head_reuse:.3f}／candle fresh: {candle:.3f}"
        f"／gap（HEAD − candle。帰属の分母・RULE.txt 5）: {gap:.3f}",
        f"- 参考: registry =0.9.0 reuse（AC-2）: {fandhe_reuse:.3f}／gap（registry − candle。版差を含み"
        f"帰属に使わない）: {gap_registry:.3f}",
        f"- HEAD path-patch / registry（record_only・RULE.txt 4）: "
        f"{head_reuse / fandhe_reuse:.3f}"
        + ("（非後退）" if head_reuse <= fandhe_reuse else "（要調査・別 issue）"),
        f"- L0 ops_total: {l0:.3f}／L1 gemm_total: {l1:.3f}／L2 l2_sum: {l2:.3f}",
        f"- Layer A（HEAD path-patch）iter_total: {a_med('iter_total'):.3f}／matmul: {a_med('matmul'):.3f}"
        f"（参考 registry =0.9.0: iter_total {a_med('iter_total', pa):.3f}／matmul {a_med('matmul', pa):.3f}）",
        f"- L0 ops_total の run 間 min–max: {rng(l0r)}",
        # RULE.txt 6 H5: reuse が fresh より遅い逆転の手がかり（HEAD reuse − candle fresh の符号）。
        f"- reuse が candle fresh より遅い逆転: {'あり' if head_reuse > candle else 'なし'}"
        f"（HEAD reuse − candle fresh = {head_reuse - candle:+.3f}）",
        "- nsys（任意）: " + ("欠測（未取得）" if nsys_text is None or nsys_text.lstrip().startswith("nsys 欠測")
                            else "取得済み（nsys-cuda-api.log を参照）"), "",
        "- 注: 各値の中央値は run 単位の値の 5 run 中央値。仮説の µs は run 毎に導出した値の中央値",
        "  （差・和は同一 run 内の区間から作る。H2 は kernel_wait と dev_kernel_b2b が別測定系列の推定を",
        "  組み合わせた値で、厳密な区間ではない）。", "",
        "### 床・launch overhead（RULE.txt 7。L0 比を併記）", "",
        "| 項目 | µs (min–max) | L0 比 |", "| --- | --- | --- |",
    ]
    tiny_r = b_runs("E", "tiny_roundtrip")
    idle_r = rr((1, b_runs("D", "dev_kernel_seg")), (-1, b_runs("D", "dev_kernel_b2b")))
    host_r = [max(0.0, l2r[i] - b_runs("D", "dev_span")[i]) for i in range(RUNS)]
    for name, xs in (("host→kernel 往復 (E tiny_roundtrip)", tiny_r),
                     ("device idle 推定 (dev_kernel_seg-b2b)", idle_r),
                     ("host のみ時間 (l2_sum-dev_span, 0 未満は 0)", host_r)):
        lines.append(f"| {name} | {med(xs):.3f} ({rng(xs)}) | {med(xs) / l0 * 100:.1f}% |")
    # RULE.txt 6 H2: floor（E tiny_roundtrip・sync_idle）との照合。H2 は推定値で床との比は目安。
    h2_r = rows[1][1]
    sync_r = b_runs("E", "sync_idle")
    lines += ["", "### H2 と床の照合（RULE.txt 6）", "",
              "| 項目 | µs (min–max) | H2 中央値との比 |", "| --- | --- | --- |"]
    for name, xs in (("H2 launch+同期の往復（推定）", h2_r), ("E tiny_roundtrip", tiny_r),
                     ("E sync_idle", sync_r)):
        ratio = f"{med(xs) / med(h2_r) * 100:.1f}%" if med(h2_r) > 0 else "n/a"
        lines.append(f"| {name} | {med(xs):.3f} ({rng(xs)}) | {ratio} |")
    lines += ["", "| 仮説 | µs (min–max) | L0 比 | gap 比 | 判定 |", "| --- | --- | --- | --- | --- |"]
    for name, xs in rows:
        v = med(xs)
        share = v / gap if gap > 0 else float("nan")
        if name.startswith(("H6", "参考", "補助")):
            verdict = "記録のみ"  # RULE.txt 6: H6・補助は記録のみ・registry 版差の行は帰属に使わない
        elif not gate_ok:
            verdict = "参考（専有ゲート FAIL）"
        else:
            verdict = "支持" if gap > 0 and share >= 0.5 else "未確定"
        lines.append(f"| {name} | {v:.3f} ({rng(xs)}) | {v / l0 * 100:.1f}% | {share * 100:.1f}% | {verdict} |")
    # RULE.txt 8 支配項: L2 各区間（teardown 含む）と H4・H5 のうち L0 比が最大のもの。
    # 主分母は L0。H5 は Layer A 側の値なので HEAD iter_total 比も併記する。
    iter_total = a_med("iter_total")
    cand = [(f"L2 {ph}", b_runs("L2", ph)) for ph in
            ("h2d_a", "h2d_b", "alloc_c", "launch_issue", "readback", "teardown", "driver_scope")]
    cand += [(n, xs) for n, xs in rows if n.startswith(("H4a", "H4b", "H5"))]
    ranked = sorted(cand, key=lambda c: med(c[1]), reverse=True)
    lines += ["", "### 支配項（RULE.txt 8。L0 比が最大の項目）", "",
              "| 順位 | 項目 | µs | L0 比 | iter_total 比 |", "| --- | --- | --- | --- | --- |"]
    for k, (name, xs) in enumerate(ranked, 1):
        lines.append(f"| {k} | {name} | {med(xs):.3f} | {med(xs) / l0 * 100:.1f}% "
                     f"| {med(xs) / iter_total * 100:.1f}% |")
    top_name, top_xs = ranked[0]
    lines += ["", f"- 支配項: **{top_name}**（{med(top_xs):.3f} µs／L0 比 {med(top_xs) / l0 * 100:.1f}%）"
              + ("" if gate_ok else "（専有ゲート FAIL のため参考）"), ""]
    return "\n".join(lines)


def _layer_b_fixture(cksum: str = "0x4062c00000000000") -> str:
    ph = {("L0", "ops_total"): 90, ("L1", "gemm_total"): 85, ("L2", "h2d_a"): 10,
          ("L2", "h2d_b"): 10, ("L2", "alloc_c"): 2, ("L2", "launch_issue"): 6,
          ("L2", "readback"): 30, ("L2S", "kernel_wait"): 20, ("L2S", "d2h"): 15,
          ("L2", "teardown"): 8,
          ("L2", "driver_scope"): 1, ("L2", "l2_sum"): 72, ("D", "dev_kernel_b2b"): 8, ("D", "dev_kernel_seg"): 14,
          ("D", "dev_span"): 50, ("E", "tiny_roundtrip"): 12, ("E", "sync_idle"): 3,
          ("E", "h2d_prealloc"): 4, ("E", "h2d_clone_drop"): 9}
    rows = [f"DIAG2109 n={N} layer={l} phase={p} median_us={v:.3f} q1_us={v:.3f} q3_us={v:.3f}"
            for (l, p), v in ph.items()]
    e = expected_counts(N)
    rows.append(f"DIAG2109 n={N} checksum_bits={cksum}")
    rows.append(f"DIAG2109 n={N} counts " + " ".join(f"{k}={v}" for k, v in e.items()))
    rows.append("test " + LAYER_B_TEST + " ... ok")
    rows.append("test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out")
    return "\n".join(rows) + "\n"


def _jsonl(vals: list[float], phases: tuple[str, ...] | None, cks: str = EXPECTED_CHECKSUM,
           task: str = "gemm", mode: str = "reuse") -> str:
    out = []
    for i in range(RUNS):
        out.append(f"-- run {i + 1}/N={N} --")
        for j, ph in enumerate(phases or (None,)):
            rec = {"task": task, "device": "cuda", "size": N, "mode": mode,
                   "median_s": vals[j] * 1e-6, "checksum": float(cks)}
            if ph:
                rec["phase"] = ph
            out.append(json.dumps(rec))
    return "\n".join(out) + "\n"


def self_test() -> None:
    lb = [_layer_b_fixture()] * RUNS
    hpa = _jsonl([70, 1, 1, 5, 92], PHASES_A, task="gemm_phases")
    hac2 = _jsonl([90], None)

    ce_ok = ("running 1 test\ntest " + COUNTS_TEST + " ... ok\n\n"
             "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")

    def agg(lb_, pa_, ac2_, cd_, g_, hpa_=None, hac2_=None, ce_=None):
        return aggregate(lb_, pa_, ac2_, cd_, g_, hpa if hpa_ is None else hpa_,
                         hac2 if hac2_ is None else hac2_, ce_ok if ce_ is None else ce_)

    pa = _jsonl([70, 1, 1, 5, 92], PHASES_A, task="gemm_phases")
    ac2 = _jsonl([92], None)
    cd = _jsonl([76], None, mode="fresh")
    gate = "".join(f"{lab} run{i} PASS load1=0.10 gpu_util=0\n"
                   for lab in GATE_LABELS for i in range(1, RUNS + 1))
    md = agg(lb, pa, ac2, cd, gate)
    assert "gap（HEAD − candle。帰属の分母・RULE.txt 5）: 14.000" in md and "支持" in md and "min–max" in md and "L0 比" in md, md
    gate_fail = gate.replace("PASS load1=0.10 gpu_util=0", "FAIL(参考扱い) load1=2.50 gpu_util=0", 1)
    md_ref = agg(lb, pa, ac2, cd, gate_fail)
    assert "参考扱い" in md_ref and "支持" not in md_ref.split("| 仮説")[1], md_ref

    def must_fail(fn, label: str) -> None:
        try:
            fn()
        except LogIntegrityError:
            return
        raise AssertionError(f"fail-closed でない: {label}")

    must_fail(lambda: agg(lb[:4], pa, ac2, cd, gate), "Layer B run 数不足")
    must_fail(
        lambda: agg([_layer_b_fixture("0x4062c00000000001")] + lb[1:], pa, ac2, cd, gate),
        "checksum_bits 不一致",
    )
    must_fail(lambda: agg(lb, pa, _jsonl([92], None, "1.000000"), cd, gate), "Layer A checksum 不一致")
    must_fail(lambda: agg(lb, pa.replace('"matmul"', '"matmul_x"', 1), ac2, cd, gate), "phase 不正")
    must_fail(lambda: agg(lb, pa + "{broken\n", ac2, cd, gate), "壊れた JSON")
    must_fail(lambda: agg([lb[0] + "# /home/someone/x\n"] + lb[1:], pa, ac2, cd, gate), "未マスク")
    tail_ok = "test result: ok. 1 passed; 0 failed"
    must_fail(lambda: agg([lb[0].replace(tail_ok, "test result: FAILED. 0 passed; 1 failed")] + lb[1:],
                          pa, ac2, cd, gate), "Layer B テスト失敗")
    must_fail(lambda: agg([lb[0].split("test result:")[0]] + lb[1:], pa, ac2, cd, gate),
              "Layer B test result 欠落")
    must_fail(lambda: agg([lb[0].replace("test " + LAYER_B_TEST, "test other")] + lb[1:],
                          pa, ac2, cd, gate), "Layer B 別テスト")
    bad = lb[0].replace("kernel_launches=1", "kernel_launches=2")
    must_fail(lambda: agg([bad] + lb[1:], pa, ac2, cd, gate), "件数不一致")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate.splitlines()[0]), "ゲート行不足")
    glines = gate.splitlines()
    must_fail(lambda: agg(lb, pa, ac2, cd, "\n".join([glines[0].replace("load1=0.10", "load1=3.00")] + glines[1:])),
              "PASS だが load1 >= 1.0")
    must_fail(lambda: agg(lb, pa, ac2, cd, "\n".join([glines[0].replace("gpu_util=0", "gpu_util=37")] + glines[1:])),
              "PASS だが gpu_util != 0")
    must_fail(lambda: agg(lb, pa, ac2, cd, "\n".join([glines[0].replace("PASS", "FAIL(参考扱い)")] + glines[1:])),
              "FAIL だが実測は専有条件を満たす")
    must_fail(lambda: agg(lb, pa, ac2, cd, "\n".join(glines[:-1] + [glines[0]])), "ゲートラベル重複・欠落")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate.replace("layerB run5", "layerB run6")), "ゲートラベル不一致")
    must_fail(lambda: agg(lb, pa.replace("N=256", "N=512", 1), ac2, cd, gate), "区切り行の N 不正")
    must_fail(lambda: agg(lb, pa.replace("run 2/", "run 3/", 1), ac2, cd, gate), "run 番号が連番でない")
    must_fail(lambda: agg(lb, pa, ac2.replace('"cuda"', '"metal"', 1), cd, gate), "device 不一致")
    must_fail(lambda: agg(lb, pa, ac2.replace('"size": 256', '"size": 512', 1), cd, gate), "size 不一致")
    must_fail(lambda: agg(lb, pa, ac2, cd.replace('"fresh"', '"reuse"', 1), gate), "mode 不一致")
    must_fail(lambda: agg(lb, pa, ac2.replace('"gemm"', '"train"', 1), cd, gate), "task 不一致")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate + "garbage\n"), "ゲート書式不正")
    # 余剰・重複レコード（RULE.txt 1・5）と HEAD ログの検証
    first = [ln for ln in ac2.splitlines() if ln.startswith("{")][0]
    must_fail(lambda: agg(lb, pa, ac2.replace("-- run 2/", first + "\n-- run 2/", 1), cd, gate),
              "AC-2 の重複レコード")
    first_c = [ln for ln in cd.splitlines() if ln.startswith("{")][0]
    must_fail(lambda: agg(lb, pa, ac2, cd.replace("-- run 2/", first_c + "\n-- run 2/", 1), gate),
              "candle の重複レコード")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate, ce_=""), "counts-exact.log が空")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate, ce_=ce_ok.replace("... ok", "... FAILED")
                          .replace("ok. 1 passed; 0 failed", "FAILED. 0 passed; 1 failed")),
              "counts-exact テスト失敗")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate, ce_=ce_ok.replace("1 passed", "0 passed")), "counts-exact 0 件実行")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate, ce_=ce_ok + "# /home/someone/x\n"), "counts-exact 未マスク")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate, hac2_=_jsonl([90], None, "1.000000")), "HEAD checksum 不一致")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate, hpa_=hpa.replace('"matmul"', '"matmul_x"', 1)), "HEAD phase 不正")
    must_fail(lambda: agg(lb, pa, ac2, cd, gate.replace("head-ac2 run1", "head-ac2 run9")), "HEAD ゲートラベル不一致")
    row = [ln for ln in md.splitlines() if ln.startswith("| H6")][0]
    assert row.endswith("| 記録のみ |"), row
    assert "H5 facade/tape (HEAD matmul-L0)" in md and "HEAD path-patch / registry" in md
    assert "支配項: **" in md and "H2 と床の照合" in md and "h2d_clone_drop − h2d_prealloc" in md, md
    # fixture: L0=90・L2 readback=30 が最大の L2 区間、H4a=5・H5=70-90=-20 → 支配項は L2 readback
    assert "支配項: **L2 readback**（30.000 µs／L0 比 33.3%）" in md, md
    aux = [ln for ln in md.splitlines() if ln.startswith("| 補助 h2d_clone_drop")][0]
    assert aux.endswith("| 記録のみ |"), aux
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
        d / f"layerA-phases-N{N}.log", d / f"layerA-ac2-N{N}.log", d / f"candle-fresh-N{N}.log",
        d / "load_gate.log", d / f"head-phases-N{N}.log", d / f"head-ac2-N{N}.log",
        d / "counts-exact.log"]
    present = [p.exists() for p in needed]
    if not any(present):
        print("ログ未生成（実測前）。orchestrate.sh を GB10 で実行後に再実行すること。")
        return 0
    if not all(present):
        missing = ", ".join(p.name for p, ok in zip(needed, present) if not ok)
        print(f"LogIntegrityError: 一部のログが欠測: {missing}（RULE.txt 1・fail-closed）",
              file=sys.stderr)
        return 1
    nsys = d / "nsys-cuda-api.log"
    try:
        print(aggregate([p.read_text() for p in needed[:RUNS]], needed[RUNS].read_text(),
                        needed[RUNS + 1].read_text(), needed[RUNS + 2].read_text(),
                        needed[RUNS + 3].read_text(), needed[RUNS + 4].read_text(),
                        needed[RUNS + 5].read_text(), needed[RUNS + 6].read_text(),
                        nsys.read_text() if nsys.exists() else None))
    except LogIntegrityError as exc:
        print(f"LogIntegrityError: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
