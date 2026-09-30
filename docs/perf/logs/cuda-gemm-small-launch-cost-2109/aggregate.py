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

fail-closed: 全レコード種別（run 区切り・phase・DIAG2109 区間/checksum/counts/kernel・
ゲート行・test 成功行）の重複（値が同一でも拒否。register_once に一本化）とキー集合の
過不足（require_exact_keys）・run 数不足・phase 欠落/重複・壊れた JSON・checksum 不一致・
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


def register_once(store: dict, key, value, where: str) -> None:
    """キー付きレコードの唯一の登録口。既出キーは上書き・無視せず LogIntegrityError にする。

    本ファイルのパーサはキー付きレコード（区間・checksum・件数・件数の各項目・ゲート行・
    run 内の phase・run 区切り）を必ずこの関数で登録する（RULE.txt 1・3。重複は連結・再実行の
    痕跡であり、最初の値・最後の値のどちらも採用しない）。
    """
    if key in store:
        raise LogIntegrityError(f"{where}: レコードが重複: {key!r}")
    store[key] = value


def require_exact_keys(got, want, where: str) -> None:
    """観測したキー集合が期待集合と過不足なく一致することを要求する（欠落・余剰・未知を拒否）。"""
    got_s, want_s = set(got), set(want)
    if got_s != want_s:
        raise LogIntegrityError(
            f"{where}: キー集合が期待と不一致（欠落 {sorted(want_s - got_s)}・余剰 {sorted(got_s - want_s)}）")


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
    cur_keys: dict = {}
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
            cur_keys = {}
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
        # median_s・checksum は集計が読む必須フィールド（欠落・非数値は KeyError にせず fail-closed）。
        for fld in ("median_s", "checksum"):
            v = rec.get(fld)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                raise LogIntegrityError(f"{name}: run {len(runs) + 1} のレコードに数値の {fld} が無い: {line!r}")
        # run 内のレコード識別子は phase（phases なしは単一レコード）。重複は register_once で拒否。
        if need is None and "phase" in rec:
            raise LogIntegrityError(f"{name}: phase を持たない系列に phase 付きレコード: {line!r}")
        register_once(cur_keys, rec.get("phase") if need else "<record>", True,
                      f"{name} run {len(runs) + 1}")
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
            require_exact_keys([rec.get("phase") for rec in r], need, f"{name} run {i} の phase")
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
    if (len(LAYER_B_NAME_RE.findall(text)) != 1 or len(TEST_RESULT_OK_RE.findall(text)) != 1
            or LAYER_B_FAIL_RE.search(text)):
        raise LogIntegrityError(
            f"{name}: {LAYER_B_TEST} の成功（test result: ok. 1 passed）を確認できない")


SIZES = (128, 256, 512)  # 診断テストの `SIZES`（gemm_small_launch_cost_diag_tests.rs）と一致させる
# 診断テストが 1 サイズあたり emit する (layer, phase) の完全集合（同ファイル run_size の emit 呼び出し）。
LAYER_B_PHASES = frozenset(
    [("L0", "ops_total"), ("L1", "gemm_total")]
    + [("L2", p) for p in ("h2d_a", "h2d_b", "alloc_c", "launch_issue", "readback", "teardown",
                           "driver_scope", "l2_sum")]
    + [("L2S", "kernel_wait"), ("L2S", "d2h")]
    + [("D", p) for p in ("dev_h2d_a", "dev_h2d_b", "dev_kernel_seg", "dev_d2h", "dev_span",
                          "dev_kernel_b2b")]
    + [("E", p) for p in ("sync_idle", "tiny_roundtrip", "event_create_drop", "h2d_prealloc",
                          "h2d_clone_drop")])
KERNEL_RE = re.compile(r"^DIAG2109 n=(\d+) kernel=(.+)$")
# 診断テストの出力一致検証（PR #2454 指摘）。全腕の出力を計測窓の外で参照と bit 一致確認した
# 実施回数の行。不一致はテスト側が panic するため成功ログに現れるのは実施済みの腕のみ。
VERIFY_RE = re.compile(r"^DIAG2109 n=(\d+) verify arm=(\w+) checks=(\d+) mode=bit_exact$")
# 診断テストの `VERIFY_ARMS`（gemm_small_launch_cost_diag_tests.rs）と一致させる。
VERIFY_ARMS = ("L0_ops", "L1_gemm", "L2", "L2S", "D_dev_trial", "D_b2b", "E_tiny",
               "E_h2d_prealloc", "E_h2d_clone", "count_run")


def parse_layer_b(text: str, name: str) -> dict:
    """1 プロセス分の Layer B ログ → phases/checksum/counts/kernel の辞書。

    DIAG2109 の全レコード種別（区間・checksum・counts・kernel）を register_once で登録し、
    同一キーの重複を拒否する。サイズは SIZES、区間は LAYER_B_PHASES と過不足なく一致すること
    （kernel 行のみ feature 依存で任意だが、あれば SIZES の各 n につき 1 回まで）。
    """
    check_masked(text, name)
    check_layer_b_success(text, name)
    phases: dict = {}
    checksum: dict = {}
    counts: dict = {}
    kernel: dict = {}
    verify: dict = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line.startswith("DIAG2109 "):
            continue
        if m := DIAG_RE.match(line):
            register_once(phases, (int(m[1]), m[2], m[3]), float(m[4]), f"{name} 区間")
        elif m := CKSUM_RE.match(line):
            register_once(checksum, int(m[1]), m[2], f"{name} checksum_bits")
        elif m := COUNTS_RE.match(line):
            items: dict = {}
            for kv in m[2].split():
                k, sep, v = kv.partition("=")
                if not sep or not k:
                    raise LogIntegrityError(f"{name}: counts 行の項目が k=v 形式でない: {line!r}")
                register_once(items, k, v, f"{name} counts n={m[1]}")
            register_once(counts, int(m[1]), items, f"{name} counts")
        elif m := VERIFY_RE.match(line):
            if int(m[3]) < 1:
                raise LogIntegrityError(f"{name}: 出力検証の実施回数が 0: {line!r}")
            register_once(verify, (int(m[1]), m[2]), int(m[3]), f"{name} 出力検証")
        elif m := KERNEL_RE.match(line):
            register_once(kernel, int(m[1]), m[2], f"{name} kernel")
        else:
            raise LogIntegrityError(f"{name}: 認識できない DIAG2109 行: {line!r}")
    require_exact_keys(checksum, SIZES, f"{name} checksum_bits の n")
    require_exact_keys(counts, SIZES, f"{name} counts の n")
    if not set(kernel) <= set(SIZES):
        raise LogIntegrityError(f"{name}: 未知の n の kernel 行: {sorted(set(kernel) - set(SIZES))}")
    require_exact_keys({(n, l, p) for (n, l, p) in phases},
                       {(n, l, p) for n in SIZES for (l, p) in LAYER_B_PHASES}, f"{name} 区間")
    for n, items in counts.items():
        require_exact_keys(items, expected_counts(n), f"{name} counts n={n} の項目")
    # 出力一致検証: 全サイズ・全腕で実施済みであること（欠落・余剰・未知の腕を拒否）。
    require_exact_keys(verify, {(n, a) for n in SIZES for a in VERIFY_ARMS}, f"{name} 出力検証")
    return {"phases": phases, "checksum": checksum, "counts": counts, "kernel": kernel,
            "verify": verify}


COUNTS_TEST = "gemm_small_launch_cost_diag_tests::gemm_small_launch_counts_exact"
TEST_RESULT_OK_RE = re.compile(r"^test result: ok\. 1 passed; 0 failed; 0 ignored;", re.M)
TEST_LINE_OK_RE = re.compile(r"^test " + re.escape(COUNTS_TEST) + r" \.\.\. ok$", re.M)


def check_counts_exact(text: str) -> None:
    """counts-exact.log が件数厳密テストの成功を示すことを検証する（RULE.txt 3・fail-closed）。

    `cargo test` の出力に当該テストの `... ok` 行と `test result: ok. 1 passed; 0 failed;
    0 ignored;` の両方が必要（テスト未実施・失敗・0 件実行・別テストの成功では通さない）。
    """
    check_masked(text, "counts-exact")
    # 成功行は各 1 回ちょうど（連結ログ・再実行の重複を通さない）。
    if len(TEST_LINE_OK_RE.findall(text)) != 1 or len(TEST_RESULT_OK_RE.findall(text)) != 1:
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
    seen: dict = {}
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
        register_once(seen, label, verdict, "load_gate")
        if verdict != "PASS":
            ok = False
    require_exact_keys(seen, GATE_EXPECTED, "load_gate ラベル")
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
              counts_exact_text: str, nsys_text: str | None = None,
              env_text: str | None = None) -> str:
    # 任意・補助入力（nsys・env_info）も必須ログと同じマスク検査を通す（RULE.txt 10）。
    check_optional_log(nsys_text, "nsys-cuda-api")
    check_optional_log(env_text, "env_info")
    gate_ok = parse_gate(gate_text)
    check_counts_exact(counts_exact_text)
    if len(layer_b_texts) != RUNS:
        raise LogIntegrityError(f"Layer B の run 数が不一致（期待 {RUNS}・実際 {len(layer_b_texts)}）")
    bs = [parse_layer_b(t, f"layerB-run{i}") for i, t in enumerate(layer_b_texts, 1)]
    for n in SIZES:
        if len({b["checksum"][n] for b in bs}) != 1:
            raise LogIntegrityError(f"Layer B の checksum_bits が run 間で不一致（n={n}。RULE.txt 2）")
    for i, b in enumerate(bs, 1):
        for n in SIZES:
            if b["counts"][n] != expected_counts(n):
                raise LogIntegrityError(f"layerB-run{i}: n={n} の件数が期待と不一致: {b['counts'][n]}")

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
          ("E", "h2d_prealloc"): 4, ("E", "h2d_clone_drop"): 9, ("E", "event_create_drop"): 5,
          ("D", "dev_h2d_a"): 3, ("D", "dev_h2d_b"): 3, ("D", "dev_d2h"): 4}
    rows = []
    for n in SIZES:
        rows += [f"DIAG2109 n={n} layer={l} phase={p} median_us={v:.3f} q1_us={v:.3f} q3_us={v:.3f}"
                 for (l, p), v in ph.items()]
        rows.append(f"DIAG2109 n={n} kernel=\"tiled\"")
        rows.append(f"DIAG2109 n={n} checksum_bits={cksum}")
        rows.append(f"DIAG2109 n={n} counts " + " ".join(f"{k}={v}" for k, v in expected_counts(n).items()))
        rows += [f"DIAG2109 n={n} verify arm={a} checks=250 mode=bit_exact" for a in VERIFY_ARMS]
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
    # 任意・補助入力（nsys・env_info）も同じマスク検査を通る（存在すれば内容を検査。正常は通る）。
    ok_nsys = "nsys 欠測（RUN_NSYS=1 未指定）\n"
    assert "欠測" in aggregate(lb, pa, ac2, cd, gate, hpa, hac2, ce_ok, ok_nsys, "host masked\n")
    assert "取得済み" in aggregate(lb, pa, ac2, cd, gate, hpa, hac2, ce_ok, "CUDA API Summary\n<home>/x\n")
    must_fail(lambda: aggregate(lb, pa, ac2, cd, gate, hpa, hac2, ce_ok, "CUDA API\n/home/someone/x\n"),
              "nsys 未マスク")
    must_fail(lambda: aggregate(lb, pa, ac2, cd, gate, hpa, hac2, ce_ok, ok_nsys + "/home/someone/x\n"),
              "nsys 欠測記録に未マスク")
    must_fail(lambda: aggregate(lb, pa, ac2, cd, gate, hpa, hac2, ce_ok, None, "path /home/someone/x\n"),
              "env_info 未マスク")

    # ---- 全レコード種別の重複・欠落・余剰（RULE.txt 1・3。値が同一でも重複は拒否）----
    def lb0(f):
        return lambda: agg([f(lb[0])] + lb[1:], pa, ac2, cd, gate)

    def dup_line(prefix: str):
        def f(t: str) -> str:
            ln = [x for x in t.splitlines() if x.startswith(prefix)][0]
            return t.replace(ln + "\n", ln + "\n" + ln + "\n", 1)
        return f

    def drop_line(prefix: str):
        return lambda t: "\n".join(x for x in t.splitlines() if not x.startswith(prefix)) + "\n"

    n_ = N
    must_fail(lb0(dup_line(f"DIAG2109 n={n_} layer=L2 phase=h2d_a ")), "Layer B 区間の重複")
    must_fail(lb0(dup_line(f"DIAG2109 n={n_} checksum_bits=")), "Layer B checksum_bits の重複（同値）")
    must_fail(lb0(dup_line(f"DIAG2109 n={n_} counts ")), "Layer B counts の重複（同値）")
    must_fail(lb0(dup_line(f"DIAG2109 n={n_} kernel=")), "Layer B kernel 行の重複")
    must_fail(lb0(dup_line(f"DIAG2109 n={n_} verify arm=L0_ops ")), "Layer B 出力検証行の重複（同値）")
    must_fail(lb0(drop_line(f"DIAG2109 n={n_} verify arm=L0_ops ")), "Layer B 出力検証（L0）の欠落")
    must_fail(lb0(drop_line("DIAG2109 n=512 verify arm=D_b2b ")), "Layer B 別サイズの出力検証欠落")
    must_fail(lb0(lambda t: t + f"DIAG2109 n={n_} verify arm=X checks=1 mode=bit_exact\n"),
              "Layer B 未知の腕の出力検証")
    must_fail(lb0(lambda t: t.replace("verify arm=L2 checks=250", "verify arm=L2 checks=0", 1)),
              "Layer B 出力検証の実施回数 0")
    must_fail(lb0(lambda t: t.replace("mode=bit_exact", "mode=loose", 1)), "Layer B 出力検証の mode 不正")
    must_fail(lb0(dup_line("test result:")), "Layer B test result の重複")
    must_fail(lb0(dup_line("test " + LAYER_B_TEST)), "Layer B テスト名行の重複")
    must_fail(lb0(drop_line(f"DIAG2109 n={n_} layer=L2 phase=h2d_a ")), "Layer B 区間の欠落")
    must_fail(lb0(drop_line("DIAG2109 n=128 layer=L0 ")), "Layer B 別サイズの区間欠落")
    must_fail(lb0(drop_line(f"DIAG2109 n={n_} checksum_bits=")), "Layer B checksum_bits の欠落")
    must_fail(lb0(drop_line("DIAG2109 n=512 checksum_bits=")), "Layer B 別サイズの checksum 欠落")
    must_fail(lb0(drop_line(f"DIAG2109 n={n_} counts ")), "Layer B counts の欠落")
    must_fail(lb0(lambda t: t + f"DIAG2109 n={n_} layer=L9 phase=x median_us=1.000 q1_us=1.000 q3_us=1.000\n"),
              "Layer B 未知の区間")
    must_fail(lb0(lambda t: t + "DIAG2109 n=1024 checksum_bits=0x4062c00000000000\n"), "Layer B 未知のサイズ")
    must_fail(lb0(lambda t: t.replace("h2d_calls=2 ", "h2d_calls=2 h2d_calls=2 ", 1)), "counts 行内の項目重複")
    must_fail(lb0(lambda t: t.replace(" h2d_calls=2", "", 1)), "counts の項目欠落")
    must_fail(lb0(lambda t: t.replace("stream_syncs=1", "stream_syncs=1 extra=0", 1)), "counts の未知項目")
    must_fail(lb0(lambda t: t.replace("pool_allocs=1", "pool_allocs", 1)), "counts の項目が k=v でない")
    # JSONL: phase レコードの重複・欠落・必須フィールド欠落
    p_first = [ln for ln in pa.splitlines() if ln.startswith("{")][0]
    must_fail(lambda: agg(lb, pa.replace("-- run 2/", p_first + "\n-- run 2/", 1), ac2, cd, gate),
              "phases の重複レコード（同値）")
    must_fail(lambda: agg(lb, pa.replace(p_first + "\n", "", 1), ac2, cd, gate), "phases のレコード欠落")
    must_fail(lambda: agg(lb, pa.replace('"median_s"', '"median_x"', 1), ac2, cd, gate), "median_s 欠落")
    must_fail(lambda: agg(lb, pa.replace('"checksum"', '"checksum_x"', 1), ac2, cd, gate), "checksum 欠落")
    must_fail(lambda: agg(lb, pa, ac2.replace('"reuse"', '"reuse", "phase": "matmul"', 1), cd, gate),
              "phase を持たない系列に phase 付きレコード")
    must_fail(lambda: agg(lb, pa, ac2.replace("-- run 2/", "-- run 1/", 1), cd, gate), "run 区切りの重複")
    must_fail(lambda: agg(lb, pa, ac2.replace("-- run 5/N=256 --\n", "", 1), cd, gate), "run 区切りの欠落")
    # load_gate: 同一行（同値）の重複
    must_fail(lambda: agg(lb, pa, ac2, cd, gate + glines[0] + "\n"), "ゲート行の重複（同値）")
    # counts-exact: 成功行の重複
    must_fail(lambda: agg(lb, pa, ac2, cd, gate, ce_=ce_ok + ce_ok), "counts-exact 成功行の重複")
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


def check_optional_log(text: str | None, name: str) -> None:
    """任意・補助ログ（nsys-cuda-api.log 等）も必須ログと同じマスク検査を通す（RULE.txt 10）。

    「欠測」記録のみの内容でも検査する。存在しない（None）場合のみ検査を省く。
    """
    if text is not None:
        check_masked(text, name)


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
    env = d / "env_info.txt"
    try:
        print(aggregate([p.read_text() for p in needed[:RUNS]], needed[RUNS].read_text(),
                        needed[RUNS + 1].read_text(), needed[RUNS + 2].read_text(),
                        needed[RUNS + 3].read_text(), needed[RUNS + 4].read_text(),
                        needed[RUNS + 5].read_text(), needed[RUNS + 6].read_text(),
                        nsys.read_text() if nsys.exists() else None,
                        env.read_text() if env.exists() else None))
    except LogIntegrityError as exc:
        print(f"LogIntegrityError: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
