#!/usr/bin/env python3
"""PyTorch GEMM の parity fail 要素を厳密真値と突合する（イシュー #1985。#1184 同型）。

## 位置づけ

`docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py`（framework-compare
と同一プロトコルの Python 版ハーネス）は PyTorch cpu N=4096 で
`parity_fail_count=1`（`max_abs 8.96e-5`・第 3 項 bound `3.05e-5`・rescued 643。
`results-dgx-py-0.8.0.jsonl`）を記録したが、bench_py.py は集計値しか出力せず
fail 要素の座標・値を残さない（Rust 側 `bench-common::parity::dump_parity_failures`
に相当する `PARITY_DUMP` 経路が無い）。本スクリプトは bench_py.py と同じ
入力（`Xorshift64Star`・固定シード `SEED_A`／`SEED_B`・`fill_vec`）と同じ
参照実装（k ごとに f64 で外積を加算し f32 へ 1 回丸める方式。
`bench_py.py::reference_gemm`）と同じ複合判定（`bench_py.py::parity`。
`bench-common/src/parity.rs::element_error` と同一式）を再現して fail 要素を
特定し、`parity_dump_truth.py`（イシュー #1184）の有理数演算（`Fraction`）で
厳密真値 `Σ_k A[row,k]·B[k,col]` を求め、`|actual−truth|`・`|ref−truth|` を
記録する。あわせて `c ∈ {0.5, 1.0, 1.5}` × 形 `{線形 K, √K}` の第 3 項
（`bound = c·u·K·S_A·S_B` ／ `c·u·√K·S_A·S_B`。`u = 2^-24`・`S_A`／`S_B` は
`parity.rs::ScaledAbsTolerance::from_inputs` と同じ **大域** `max|A|`／
`max|B|`）で救済されるか否かの表を JSON と Markdown で出力する。

**判定・tolerance の変更はしない**（本体 `RELATIVE_TOLERANCE`／
`ABSOLUTE_RESCUE_THRESHOLD`／`PARITY_SCALED_ABS_COEFF`・`bench-common::parity`・
`bench_py.py`・`compare_gemm_gate.py` はいずれも不変。本スクリプトは机上計算
の診断専用ツールであり、係数変更はイシュー #1985 の記録を受けたユーザー
承認事項として別途扱う）。

## 参照実装の再現方法（bench_py.py 準拠）

`bench_py.py::reference_gemm` は `c = float32(float64(c) + A64[:,k]·B64[k,:])`
を k 昇順に繰り返す。f32×f32 の積は f64 で正確なので、これは各 k で
「厳密な `a·b + acc` を 1 回 f32 へ丸める」FMA と、`a·b + acc` の f64 丸めが
f32 丸めと二重丸めになる稀な境界ケースを除いて bit 一致する。本スクリプトは
torch の f64 演算で同じ手順を再現し（`reference_gemm_torch`）、fail 要素に
ついては `parity_dump_truth.fma_sequential_f32_exact`（各ステップを有理数で
厳密計算し 1 回だけ丸める真の FMA 再現）とも突合して bit 一致有無
（`ref_fma_bit_match`）を記録する。不一致なら二重丸めケースとして事実を
残す（隠さない）。fail 要素の pass/fail が真の FMA 参照でも同じかも併記する。

## 呼び出し元

DGX Spark GB10 の Python 参照フレームワーク venv（`docs/perf/logs/
lowlayer-diagnosis-2026-09-12/scripts/dgx-venv.sh` が構築する
`~/work/.venv-bench`・PyTorch 2.14.0+cu130）で 1 回実行し、生成物を
`docs/perf/logs/parity-torch-cpu-truth-1985/` へ収納する。CI では
`--self-test` のみ（torch 不在なら torch 依存部は skip）。

使い方:
    python3 parity_torch_truth.py --n 4096 --device cpu --out truth-torch-cpu-4096.json
    python3 parity_torch_truth.py --self-test

依存は Python 標準ライブラリ ＋ torch のみ（numpy は不要。
`.claude/rules/deps-policy.md` の対象外だが新規依存を増やさない）。
`parity_dump_truth.py` 自体は変更せず `importlib` で読み込んで再利用する。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.util
import json
import math
import os
import platform
import struct
import sys
import time
from fractions import Fraction
from typing import Any

# `parity_dump_truth.py` を隣接ファイルとして importlib 経由で読み込む
# （`parity_tolerance_candidates.py` と同じ手順。`sys.modules` への登録を
# `exec_module` より前に行わないと dataclass デコレータの解決で
# `AttributeError` になる制約がある）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRUTH_PATH = os.path.join(_HERE, "parity_dump_truth.py")
_TRUTH_SPEC = importlib.util.spec_from_file_location("parity_dump_truth", _TRUTH_PATH)
if _TRUTH_SPEC is None or _TRUTH_SPEC.loader is None:
    raise ImportError(f"parity_dump_truth.py を読み込めない: {_TRUTH_PATH}")
parity_dump_truth = importlib.util.module_from_spec(_TRUTH_SPEC)
sys.modules[_TRUTH_SPEC.name] = parity_dump_truth
_TRUTH_SPEC.loader.exec_module(parity_dump_truth)

SEED_A = parity_dump_truth.SEED_A
SEED_B = parity_dump_truth.SEED_B

# 複合判定の定数（`bench-common/src/parity.rs::PARITY_REL_TOL`／`PARITY_ABS_TOL`／
# `PARITY_SCALED_ABS_COEFF`／`F32_UNIT_ROUNDOFF` と同値。正は本体側であり
# ここでは再現のためだけに再定義する。変更はユーザー承認必須）。
PARITY_REL_TOL = 1e-3
PARITY_ABS_TOL = 1e-5
PARITY_SCALED_ABS_COEFF = 0.5
F32_UNIT_ROUNDOFF = 2.0**-24

# 救済可否表の候補（イシュー #1985 受け入れ条件。現行係数 c=0.5 を含む）。
TABLE_COEFFS = (0.5, 1.0, 1.5)
TABLE_K_MODES = ("K", "sqrtK")


def f32_bits(x: float) -> int:
    """f32 値（Python float として f32 表現可能な値）の IEEE754 bit パターン。"""
    return struct.unpack("<I", struct.pack("<f", x))[0]


def round_to_f32(x: float) -> float:
    """f64 値を f32 へ 1 回丸める（round-half-even。`struct` の変換規則）。"""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def fill_vec_f32(seed: int, n: int) -> list[float]:
    """`bench-common::Xorshift64Star::fill_vec`（`bench_py.py::fill_vec` と同順）の再現。

    1 要素は `((x >> 40) as f32) / 2^24 - 0.5`。`x >> 40` は 24 bit 整数なので
    `k / 2^24 - 0.5` は f32（したがって Python の f64）で厳密表現でき、
    `parity_dump_truth.Xorshift64StarExact.next_element_exact` の `Fraction`
    と値が一致する（`--self-test` で bit 単位に検査する）。
    """
    gen = parity_dump_truth.Xorshift64StarExact(seed)
    out: list[float] = []
    append = out.append
    for _ in range(n):
        x = gen.next_u64()
        append((x >> 40) / 16777216.0 - 0.5)
    return out


def reference_gemm_pure(a: list[float], b: list[float], n: int) -> list[float]:
    """`bench_py.py::reference_gemm` の純 Python 逐語再現（小 N の自己テスト用）。

    各要素について k 昇順に `acc = f32(f64(acc) + a*b)`。`a*b` は f32×f32 の積
    なので f64 で正確（24 bit × 24 bit = 48 bit ≤ 53 bit）。N=4096 の全要素には
    使えない（O(N^3) 純ループ）ため、本番経路は `reference_gemm_torch` を使う。
    """
    c: list[float] = []
    for i in range(n):
        a_row = a[i * n : i * n + n]
        for j in range(n):
            acc = 0.0
            for k in range(n):
                acc = round_to_f32(acc + a_row[k] * b[k * n + j])
            c.append(acc)
    return c


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError:
        return None
    return torch


def reference_gemm_torch(torch: Any, a: list[float], b: list[float], n: int) -> Any:
    """`bench_py.py::reference_gemm` の torch 版（cpu・f64 演算）。

    `c = (c.double() + outer(A64[:,k], B64[k,:])).float()` を k 昇順に n 回
    繰り返す。numpy 版と同じく「f64 で加算 → f32 へ 1 回丸め」であり、
    要素ごとの演算順序（k 昇順）も同一。戻り値は `[n, n]` の f32 テンソル。
    """
    a64 = torch.tensor(a, dtype=torch.float64).view(n, n)
    b64 = torch.tensor(b, dtype=torch.float64).view(n, n)
    c = torch.zeros((n, n), dtype=torch.float32)
    for k in range(n):
        c = (c.double() + torch.outer(a64[:, k], b64[k, :])).float()
    return c


def scale_ab(a: list[float], b: list[float]) -> tuple[float, float]:
    """`parity.rs::ScaledAbsTolerance::from_inputs` と同じ大域 `max|A|`／`max|B|`。"""
    return max(abs(v) for v in a), max(abs(v) for v in b)


def scaled_abs_bound(n: int, s_a: float, s_b: float, c: float, k_mode: str, u: float = F32_UNIT_ROUNDOFF) -> float:
    """第 3 項の bound。`k_mode="K"` は `parity.rs::ScaledAbsTolerance::bound` と同式
    （`c=0.5` で現行値）、`"sqrtK"` は案 1′（`docs/spec-proposal-req2-candle-parity-tolerance.md`）。
    非有限・負のスケールは fail-closed で 0.0（本体と同じ）。"""
    if not (math.isfinite(s_a) and math.isfinite(s_b)) or s_a < 0.0 or s_b < 0.0:
        return 0.0
    kscale = float(n) if k_mode == "K" else math.sqrt(n)
    bound = c * u * kscale * s_a * s_b
    return bound if math.isfinite(bound) else 0.0


def element_judgement(actual: float, reference: float, bound: float) -> dict[str, Any]:
    """`bench_py.py::parity`（= `parity.rs::element_error`）の 1 要素版。"""
    xf = float(actual)
    yf = float(reference)
    diff = abs(xf - yf)
    scale = max(abs(xf), abs(yf), 1e-12)
    rel = diff / scale
    legacy = (rel < PARITY_REL_TOL) or (diff < PARITY_ABS_TOL)
    scaled = diff <= bound
    finite = math.isfinite(xf)
    ok = (legacy or scaled) and finite
    return {
        "abs": diff,
        "rel": rel,
        "legacy_pass": legacy,
        "scaled_abs_pass": scaled,
        "pass": ok,
        "rescued_by_scaled_abs": (not legacy) and scaled and finite,
    }


def find_fail_elements(actual: list[float], ref: list[float], n: int, bound: float) -> tuple[list[int], dict[str, Any]]:
    """全 N² 要素に複合判定を適用し fail idx 列と集計（bench_py.py の JSONL 列と同名）を返す。"""
    fails: list[int] = []
    max_abs = 0.0
    max_rel = 0.0
    rescued = 0
    for idx, (x, y) in enumerate(zip(actual, ref)):
        j = element_judgement(x, y, bound)
        if not j["pass"]:
            fails.append(idx)
        if j["rescued_by_scaled_abs"]:
            rescued += 1
        # 非有限要素は `parity.rs::compare_elementwise` と同じく INFINITY センチネル。
        max_abs = max(max_abs, j["abs"]) if math.isfinite(j["abs"]) else math.inf
        max_rel = max(max_rel, j["rel"]) if math.isfinite(j["rel"]) else math.inf
    stats = {
        "parity_total": len(actual),
        "parity_fail_count": len(fails),
        "parity_max_abs_err": max_abs,
        "parity_max_rel_err": max_rel,
        "parity_scaled_abs_bound": bound,
        "parity_scaled_abs_rescued": rescued,
    }
    return fails, stats


def analyze_fail(
    idx: int,
    n: int,
    actual: float,
    ref: float,
    a_row: list[Fraction],
    b_col: list[Fraction],
    s_a: float,
    s_b: float,
) -> dict[str, Any]:
    """fail 1 要素の真値突合・救済可否表（JSON 1 レコード）。"""
    row, col = divmod(idx, n)
    exact = sum((x * y for x, y in zip(a_row, b_col)), Fraction(0))
    exact_f = float(exact)
    acc64 = 0.0
    for x, y in zip(a_row, b_col):
        acc64 += float(x) * float(y)
    fma_f32, partials = parity_dump_truth.fma_sequential_f32_exact(a_row, b_col)
    fma_bits = f32_bits(fma_f32)
    ref_bits = f32_bits(ref)
    actual_bits = f32_bits(actual)
    current_bound = scaled_abs_bound(n, s_a, s_b, PARITY_SCALED_ABS_COEFF, "K")
    judge_ref = element_judgement(actual, ref, current_bound)
    judge_fma = element_judgement(actual, fma_f32, current_bound)
    max_partial = max((abs(float(p)) for p in partials), default=0.0)
    d = abs(float(actual) - float(ref))
    table = []
    for c in TABLE_COEFFS:
        for k_mode in TABLE_K_MODES:
            bound = scaled_abs_bound(n, s_a, s_b, c, k_mode)
            table.append(
                {
                    "c": c,
                    "k_mode": k_mode,
                    "bound": bound,
                    "rescued": d <= bound,
                    # 現行 K 形・c=0.5 は本体の第 3 項そのもの（`parity.rs`）。
                    "is_current_contract": (c == PARITY_SCALED_ABS_COEFF and k_mode == "K"),
                }
            )
    return {
        "idx": idx,
        "row": row,
        "col": col,
        "actual": float(actual),
        "actual_bits": f"0x{actual_bits:08x}",
        "ref": float(ref),
        "ref_bits": f"0x{ref_bits:08x}",
        "ref_fma_exact": float(fma_f32),
        "ref_fma_exact_bits": f"0x{fma_bits:08x}",
        "ref_fma_bit_match": fma_bits == ref_bits,
        "exact_truth": exact_f,
        "exact_truth_fraction": f"{exact.numerator}/{exact.denominator}",
        "f64_seq": acc64,
        "abs_actual_minus_truth": abs(float(actual) - exact_f),
        "abs_ref_minus_truth": abs(float(ref) - exact_f),
        "abs_ref_minus_actual": d,
        "rel": judge_ref["rel"],
        "max_abs_partial": max_partial,
        "sqrtK_ulp_max_partial": math.sqrt(n) * parity_dump_truth.ulp_f32(max_partial),
        "closer_to_truth": "actual" if abs(float(actual) - exact_f) < abs(float(ref) - exact_f) else ("ref" if abs(float(actual) - exact_f) > abs(float(ref) - exact_f) else "tie"),
        "pass_under_numpy_style_ref": judge_ref["pass"],
        "pass_under_exact_fma_ref": judge_fma["pass"],
        "rescue_table": table,
    }


def render_markdown(result: dict[str, Any]) -> str:
    n = result["n"]
    lines: list[str] = []
    lines.append(f"# PyTorch {result['device']} N={n} parity fail 要素の真値突合（イシュー #1985）")
    lines.append("")
    lines.append(
        "判定・tolerance 変更なし（机上計算のみ）。参照は bench_py.py と同じ「k 昇順に f64 加算 → f32 丸め」方式。"
        f" `S_A={result['scale_a']:.9g}`・`S_B={result['scale_b']:.9g}`・`u=2^-24`・現行 bound（c=0.5・線形 K）={result['stats']['parity_scaled_abs_bound']:.6e}"
    )
    lines.append("")
    st = result["stats"]
    lines.append("## 集計（bench_py.py の JSONL 列と同名）")
    lines.append("")
    lines.append("| total | fail_count | max_abs_err | max_rel_err | scaled_abs_bound | scaled_abs_rescued |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    lines.append(
        f"| {st['parity_total']} | {st['parity_fail_count']} | {st['parity_max_abs_err']:.6e} | "
        f"{st['parity_max_rel_err']:.6e} | {st['parity_scaled_abs_bound']:.6e} | {st['parity_scaled_abs_rescued']} |"
    )
    lines.append("")
    if not result["fails"]:
        lines.append("**fail なし**（複合判定で fail した要素は 0 件）。")
        lines.append("")
        return "\n".join(lines)
    lines.append("## fail 要素の真値突合")
    lines.append("")
    if result.get("fail_analysis_truncated"):
        lines.append(
            f"**注**: `--max-fails` により真値突合は fail 要素 {len(result['fails'])} 件（idx 昇順）に限定した"
            f"（全 fail idx は JSON の `fail_indices` に {len(result['fail_indices'])} 件記録）。"
        )
        lines.append("")
    lines.append(
        "| idx | row | col | actual (bits) | ref (bits) | ref_fma_bit_match | exact truth | \\|actual−truth\\| | \\|ref−truth\\| | \\|ref−actual\\| | 真値に近い側 | pass(真 FMA 参照) |"
    )
    lines.append("|---:|---:|---:|---|---|---|---:|---:|---:|---:|---|---|")
    for f in result["fails"]:
        lines.append(
            f"| {f['idx']} | {f['row']} | {f['col']} | {f['actual']:.9e} ({f['actual_bits']}) | "
            f"{f['ref']:.9e} ({f['ref_bits']}) | {f['ref_fma_bit_match']} | {f['exact_truth']:.9e} | "
            f"{f['abs_actual_minus_truth']:.3e} | {f['abs_ref_minus_truth']:.3e} | {f['abs_ref_minus_actual']:.3e} | "
            f"{f['closer_to_truth']} | {f['pass_under_exact_fma_ref']} |"
        )
    lines.append("")
    lines.append("## 救済可否表（`d=|ref−actual| <= bound`。u=2^-24・S_A·S_B は大域 max）")
    lines.append("")
    header = "| c | 形 | bound | " + " | ".join(f"idx={f['idx']}" for f in result["fails"]) + " |"
    lines.append(header)
    lines.append("|---:|---|---:|" + "|".join(["---"] * len(result["fails"])) + "|")
    for i, entry in enumerate(result["fails"][0]["rescue_table"]):
        cells = []
        for f in result["fails"]:
            e = f["rescue_table"][i]
            cells.append("救済" if e["rescued"] else "fail")
        mark = "（現行契約）" if entry["is_current_contract"] else ""
        lines.append(
            f"| {entry['c']} | {'線形 K' if entry['k_mode'] == 'K' else '√K'}{mark} | {entry['bound']:.6e} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def env_info(torch: Any) -> dict[str, Any]:
    """記録用の環境情報（内部ホスト名・ユーザー名・絶対パスは含めない）。"""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if torch is not None:
        info["torch"] = torch.__version__
        info["torch_num_threads"] = torch.get_num_threads()
        try:
            info["torch_parallel_info"] = torch.__config__.parallel_info()
        except Exception:  # noqa: BLE001 - 診断情報の取得失敗は記録に留める
            info["torch_parallel_info"] = "<unavailable>"
    return info


def run(n: int, device: str, out_path: str, max_fails: int | None) -> int:
    torch = _import_torch()
    if torch is None:
        print("ERROR: torch を import できない（本スクリプトの本番経路は torch 必須）", file=sys.stderr)
        return 2
    if device == "cuda" and not torch.cuda.is_available():
        print("ERROR: --device cuda が指定されたが torch.cuda.is_available() が False", file=sys.stderr)
        return 2
    dev = torch.device({"cuda": "cuda", "mps": "mps"}.get(device, "cpu"))

    t0 = time.time()
    a = fill_vec_f32(SEED_A, n * n)
    b = fill_vec_f32(SEED_B, n * n)
    t_fill = time.time() - t0
    s_a, s_b = scale_ab(a, b)
    bound = scaled_abs_bound(n, s_a, s_b, PARITY_SCALED_ABS_COEFF, "K")

    a_t = torch.tensor(a, dtype=torch.float32).view(n, n)
    b_t = torch.tensor(b, dtype=torch.float32).view(n, n)
    t0 = time.time()
    with torch.no_grad():
        c_t = torch.mm(a_t.to(dev), b_t.to(dev)).cpu()
    t_mm = time.time() - t0
    actual = c_t.reshape(-1).tolist()

    t0 = time.time()
    ref_t = reference_gemm_torch(torch, a, b, n)
    t_ref = time.time() - t0
    ref = ref_t.reshape(-1).tolist()

    fails, stats = find_fail_elements(actual, ref, n, bound)
    print(f"# n={n} device={device} fail_count={stats['parity_fail_count']} max_abs={stats['parity_max_abs_err']:.6e} "
          f"bound={bound:.6e} rescued={stats['parity_scaled_abs_rescued']} (fill {t_fill:.1f}s / mm {t_mm:.1f}s / ref {t_ref:.1f}s)")

    analyzed = fails
    truncated = False
    if max_fails is not None and len(fails) > max_fails:
        analyzed = fails[:max_fails]
        truncated = True
    fail_records: list[dict[str, Any]] = []
    if analyzed:
        needed_rows = {i // n for i in analyzed}
        needed_cols = {i % n for i in analyzed}
        a_rows = parity_dump_truth.extract_rows_exact(SEED_A, n, needed_rows)
        b_cols = parity_dump_truth.extract_cols_exact(SEED_B, n, needed_cols)
        for idx in analyzed:
            rec = analyze_fail(idx, n, actual[idx], ref[idx], a_rows[idx // n], b_cols[idx % n], s_a, s_b)
            # 自己整合: 厳密再現した A・B 行列と f32 経由の値が一致すること
            # （RNG 再現の崩れを fail-closed に検出）。
            if any(float(x) != a[(idx // n) * n + k] for k, x in enumerate(a_rows[idx // n])):
                print(f"ERROR: idx={idx} の A 行の厳密再現が f32 版と不一致", file=sys.stderr)
                return 1
            if any(float(y) != b[k * n + idx % n] for k, y in enumerate(b_cols[idx % n])):
                print(f"ERROR: idx={idx} の B 列の厳密再現が f32 版と不一致", file=sys.stderr)
                return 1
            fail_records.append(rec)
            print(
                f"  idx={idx} row={rec['row']} col={rec['col']} actual={rec['actual']:.9e} ref={rec['ref']:.9e} "
                f"truth={rec['exact_truth']:.9e} |actual-truth|={rec['abs_actual_minus_truth']:.3e} "
                f"|ref-truth|={rec['abs_ref_minus_truth']:.3e} ref_fma_bit_match={rec['ref_fma_bit_match']} "
                f"closer={rec['closer_to_truth']}"
            )
    else:
        print("fail なし")

    result = {
        "issue": 1985,
        "n": n,
        "device": device,
        "scale_a": s_a,
        "scale_b": s_b,
        "u": F32_UNIT_ROUNDOFF,
        "stats": stats,
        "fail_indices": fails,
        "fails": fail_records,
        "fail_analysis_truncated": truncated,
        "timing_s": {"fill_vec": t_fill, "torch_mm": t_mm, "reference": t_ref},
        "env": env_info(torch),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    md_path = os.path.splitext(out_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(result))
        f.write("\n")
    print(f"wrote {os.path.basename(out_path)} / {os.path.basename(md_path)}")
    # bench_py.py の参照と真の FMA 参照の bit 不一致は事実として記録するが
    # 終了コードは成功（診断結果の生成自体は完了しているため）。
    return 0


def self_test() -> int:
    """torch 不在でも実行できる部分（RNG・参照方式・判定式）を検査し、torch があれば
    torch 経路（f64 mm と厳密真値・torch 参照と純 Python 参照の bit 一致）も検査する。"""
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"[{'ok' if ok else 'FAIL'}] {name}{(' ' + detail) if detail else ''}")
        if not ok:
            failures += 1

    # (1) fill_vec_f32 と parity_dump_truth の厳密 Fraction 再現が bit 一致する。
    gen = parity_dump_truth.Xorshift64StarExact(SEED_A)
    exact = [gen.next_element_exact() for _ in range(2048)]
    mine = fill_vec_f32(SEED_A, 2048)
    check("fill_vec_f32 == Fraction 再現（2048 要素 bit 一致）", all(f32_bits(float(x)) == f32_bits(y) for x, y in zip(exact, mine)))
    check("fill_vec_f32 値域 [-0.5, 0.5)", all(-0.5 <= v < 0.5 for v in mine))

    # (2) N=8: 純 Python の bench_py 方式参照 と 厳密 FMA 再現 の bit 一致（全 64 要素）。
    n = 8
    a = fill_vec_f32(SEED_A, n * n)
    b = fill_vec_f32(SEED_B, n * n)
    ref_pure = reference_gemm_pure(a, b, n)
    a_rows = parity_dump_truth.extract_rows_exact(SEED_A, n, set(range(n)))
    b_cols = parity_dump_truth.extract_cols_exact(SEED_B, n, set(range(n)))
    mism = 0
    for i in range(n):
        for j in range(n):
            fma_f32, _ = parity_dump_truth.fma_sequential_f32_exact(a_rows[i], b_cols[j])
            if f32_bits(fma_f32) != f32_bits(ref_pure[i * n + j]):
                mism += 1
    check("N=8 bench_py 方式参照（純 Python）== 厳密 FMA 再現（bit 一致）", mism == 0, f"mismatch={mism}")

    # (3) 判定式: 既知値で legacy／scaled の分岐を確認。
    s_a, s_b = scale_ab(a, b)
    bound = scaled_abs_bound(4096, s_a, s_b, 0.5, "K")
    check("bound(4096, c=0.5, K) ≈ 3.05e-5", abs(bound - 0.5 * 2.0**-24 * 4096 * s_a * s_b) == 0.0 and 2.9e-5 < bound < 3.2e-5, f"bound={bound:.6e}")
    j = element_judgement(1.0, 1.0 + 2e-3, 0.0)
    check("rel 2e-3 → fail", not j["pass"])
    j = element_judgement(1.0, 1.0 + 5e-4, 0.0)
    check("rel 5e-4 → legacy pass", j["pass"] and not j["rescued_by_scaled_abs"])
    j = element_judgement(0.0, 2.5e-5, 3.0e-5)
    check("abs 2.5e-5 ≤ bound 3e-5 → scaled 救済", j["pass"] and j["rescued_by_scaled_abs"])
    check("NaN → fail", not element_judgement(float("nan"), 0.0, 1.0)["pass"])
    # 救済表の形（K 形は √K 形より緩い・c 単調）。
    tb = [scaled_abs_bound(4096, s_a, s_b, c, m) for c in TABLE_COEFFS for m in TABLE_K_MODES]
    check("bound 単調性（c 昇順・K > √K）", tb[0] < tb[2] < tb[4] and tb[1] < tb[3] < tb[5] and tb[0] > tb[1])

    torch = _import_torch()
    if torch is None:
        print("[skip] torch が import できないため torch 経路（f64 mm と厳密真値の一致・torch 参照の bit 一致）は skip")
    else:
        # (4) N=8: torch 参照（f64 加算方式）と純 Python 参照の bit 一致。
        ref_t = reference_gemm_torch(torch, a, b, n).reshape(-1).tolist()
        check("N=8 torch 参照 == 純 Python 参照（bit 一致）", all(f32_bits(x) == f32_bits(y) for x, y in zip(ref_t, ref_pure)))
        # (5) N=8: torch f64 mm と Fraction 厳密真値が f64 数 ulp 以内で一致
        # （f64 の 8 項和は厳密とは限らないため tolerance 検査）。
        a64 = torch.tensor(a, dtype=torch.float64).view(n, n)
        b64 = torch.tensor(b, dtype=torch.float64).view(n, n)
        mm64 = torch.mm(a64, b64).reshape(-1).tolist()
        worst = 0.0
        for i in range(n):
            for j_ in range(n):
                ex = float(sum((x * y for x, y in zip(a_rows[i], b_cols[j_])), Fraction(0)))
                worst = max(worst, abs(mm64[i * n + j_] - ex))
        check("N=8 torch f64 mm ≈ Fraction 厳密真値（|diff| ≤ 1e-14）", worst <= 1e-14, f"worst={worst:.3e}")
        # (6) N=8 で本番経路（f32 mm vs 参照）が動く（fail 数は 0 を期待するが記録のみ）。
        c_t = torch.mm(torch.tensor(a, dtype=torch.float32).view(n, n), torch.tensor(b, dtype=torch.float32).view(n, n))
        fails, stats = find_fail_elements(c_t.reshape(-1).tolist(), ref_pure, n, scaled_abs_bound(n, s_a, s_b, 0.5, "K"))
        check("N=8 torch f32 mm の複合判定が実行できる", stats["parity_total"] == 64, f"fail_count={len(fails)}")

    print(f"self-test: {'green' if failures == 0 else f'{failures} FAIL'}")
    return 0 if failures == 0 else 1


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"整数として解釈できません: {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"1 以上の整数を指定してください（受領: {parsed}）")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=4096, help="GEMM の正方行列一辺長（既定 4096。1..8192）")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="torch.mm を実行するデバイス（既定 cpu。参照実装は常に cpu f64）")
    parser.add_argument("--out", default=None, help="結果 JSON の出力先（Markdown は同名 .md を隣に書く）")
    parser.add_argument(
        "--max-fails",
        type=_positive_int,
        default=None,
        help="真値突合する fail 要素数の上限（fail 要素の idx 昇順。未指定なら全件。fail_indices は常に全件記録）",
    )
    parser.add_argument("--self-test", action="store_true", help="自己テスト（torch 不在なら torch 経路は skip）")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if args.n < 1 or args.n > 8192:
        print(f"ERROR: --n は 1..8192 の範囲で指定する（受領: {args.n}）", file=sys.stderr)
        return 2
    if args.out is None:
        print("ERROR: --out を指定する（--self-test 以外では必須）", file=sys.stderr)
        return 2
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(out_dir):
        print("ERROR: --out の親ディレクトリが存在しない", file=sys.stderr)
        return 2
    return run(args.n, args.device, args.out, args.max_fails)


if __name__ == "__main__":
    raise SystemExit(main())
