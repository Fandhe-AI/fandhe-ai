#!/usr/bin/env python3
"""GEMM 目標達成ゲート（CUDA: #1031／Metal: #1037／CPU: #1117）の 5 回計測
中央値集計ツール（イシュー #1142・#1147 で Metal 対応汎用化・#1148 で CPU
対応拡張）。

使い方:
    python3 compare_gemm_gate.py [--device {cuda,metal,cpu}] JSONL [JSONL ...] [--out FILE]

`run_gemm_gate.sh <device> <label>`（device 別 wrapper は
`run_gemm_gate_cuda.sh`／`run_gemm_gate_metal.sh`／`run_gemm_gate_cpu.sh`）
が出力する JSONL（`bench-fandhe gemm <device> <N> reuse`・`bench-candle gemm
<device> <N> fresh` を対象形状（cuda/metal: N=1024/2048/4096、cpu:
N=512/1024/2048。`_SIZES_BY_DEVICE`）それぞれ 5 回起動した結果。cpu のみ
`bench-fandhe gemm cpu <N> fresh` も同数起動され、判定に使わない参考列
として集計される）を読み、N ごとに fandhe-ai（reuse）vs candle（fresh。
candle は reuse 非対応のため fresh 固定）の `median_s` を run 間中央値で
集約し、CUDA は #1031・Metal は #1037・CPU は #1117 の受け入れ条件
（`fandhe_median_s <= candle_median_s`）を判定する。`--device` は既定
`cuda`（#1142 時点の呼び出し元との後方互換）。

`summarize.py --target candle` は同一入力ファイル内の 1 レコードのみを
拾う設計（1 ファイル = 1 環境の単発計測が前提）のため、5 回計測の run 間
中央値（`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」）には
非対応。本ツールはその欠落を埋める専用集計（`compare_ab.py` が train タスク
向けに担っている 5 回集計を gemm タスク向けに用意したもの）。

設計方針（`compare_ab.py`・`summarize.py` と同じ思想。tolerance 定数・
`checksums_match` は `checksum_contract.py` を単一真実源として共有する）:
- fail-closed: レコード件数が size ごとに fandhe-ai/candle 各ちょうど 5 件で
  ない（過不足いずれも判定不能。標本差し替え防止。イシュー #1166）・
  `median_s`/`checksum` が不正値（非正・NaN・Infinity 等）・要素単位検証
  （`parity_*` フィールド。イシュー #970）が `parity_fail_count > 0` また
  は `parity_total != size*size`・checksum が本体の数値一致契約（相対
  誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を外れる場合は、性能値を確定
  表示せず「判定不能」を明示し理由（run ごとの fail_count/max_abs/max_rel
  を含む）を出力する（A08: 壊れた計算の実行時間で達成判定しない）。
  N=2048 の candle 無効データ（イシュー #1142 R2）の再現条件記録は、この
  判定不能理由の詳細出力でまかなう（`bench-common::parity` への追加計装
  は行わない。既存 JSONL フィールドで十分診断できるため）。
- 捏造しない: 入力 JSONL 自体は変更しない。JSON parse 不能な行は理由付き
  でスキップし黙って無視しない。
- `--tf32` 行（イシュー #1042）は本ゲートの対象外として除外する（FP32
  目標値との混同を防ぐ。summarize.py の目標達成ゲートと同じ既定）。
- 複数ファイル指定時はファイルごとに独立集計する（`summarize.py` と同じ
  「ファイルをまたいだ突合は環境混同になるため行わない」方針）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys

_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "checksum_contract",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checksum_contract.py"),
)
_checksum_contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(_checksum_contract)
checksums_match = _checksum_contract.checksums_match

# #1031（cuda）／#1037（metal）／#1117（cpu。#1148 で cpu 拡張）の対象
# 形状（reuse candle 比再計測）。cpu は run_gemm_gate.sh 側の run 構成
# （N=512/1024/2048）に合わせる（イシュー #1148。cuda/metal は #1142/#1147
# から不変）。
_SIZES_BY_DEVICE = {
    "cuda": (1024, 2048, 4096),
    "metal": (1024, 2048, 4096),
    "cpu": (512, 1024, 2048),
}
# coding-rust.md「ベンチは 5 回計測の中央値」。
MIN_RECORDS = 5
# candle は gemm reuse モードに非対応（`bench-candle` は reuse 指定を
# 常に MEASURE_ERROR で fail-fast する仕様。README「計測プロトコル」節）
# のため、比較対象は常に fresh 固定（summarize.py `_pick_row_for_gate` の
# 「target 側は fresh フォールバック」規約と同じ）。
FANDHE_MODE = "reuse"
CANDLE_MODE = "fresh"


def _is_plain_number(v):
    """外部 JSONL 由来の値が bool・NaN・Infinity・非数値でないか検証する
    （`compare_ab.py` `_is_plain_number` と同一の判定基準）。"""
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return math.isfinite(v)


def _safe_positive(v):
    if not _is_plain_number(v):
        return None
    try:
        fv = float(v)
    except OverflowError:
        return None
    return fv if fv > 0 else None


def _safe_finite(v):
    if not _is_plain_number(v):
        return None
    try:
        return float(v)
    except OverflowError:
        return None


def _non_integral(v):
    if isinstance(v, bool):
        return True
    if isinstance(v, int):
        return False
    return float(v) != int(v)


def load_rows(path):
    """JSONL を読み、不正な行は理由付きで報告しスキップする（A08）。"""
    rows = []
    warnings = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.append(f"{path}:{lineno}: invalid JSON ({e}) — skipped")
                continue
            if not isinstance(obj, dict):
                warnings.append(
                    f"{path}:{lineno}: JSON object ではない（{type(obj).__name__}） — skipped"
                )
                continue
            # codex-review P0 指摘（PR #1166）: `tf32` は外部 JSONL 由来の値
            # であり、`r.get("tf32", False) is True` のみで判定すると `1`・
            # `"true"` 等の bool 以外の不正値は `is True` が常に False を
            # 返すため「tf32 ではない」と誤って FP32 ゲートへ混入する
            # fail-open の欠陥だった（summarize.py `load_rows` の同種検証
            # と同じ理由。イシュー #1042 codex-review P0 指摘の再発）。
            # キー欠損（`False` 扱い。互換規約）または厳密な `bool` 型で
            # あることをここで検証し、不正型の行は理由付きで丸ごとスキップ
            # する（fail-closed: `_matching_rows` の `is True` 判定へ到達
            # させず、FP32/TF32 いずれのゲートにも黙って含めない）。
            if "tf32" in obj and not isinstance(obj["tf32"], bool):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'tf32' フィールド型（bool を期待。"
                    f"実際: {obj['tf32']!r}） — skipped"
                )
                continue
            rows.append(obj)
    return rows, warnings


def _matching_rows(rows, framework, mode, size, device="cuda"):
    """`(framework, task=gemm, device, size, mode)` に一致する行を返す。

    `device` は既定 "cuda"（#1142 時点の呼び出し元との後方互換）。Metal
    ゲート（#1037・イシュー #1147）は呼び出し側から "metal" を渡す。
    `tf32:true` の行は除外する（FP32 目標値との混同防止。summarize.py と
    同じ既定。Metal は TF32 経路自体が存在しないため常に該当なし）。
    `mode` キー欠損は "fresh" 扱い（イシュー #925 互換規約）。
    """
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("framework") != framework:
            continue
        if r.get("task") != "gemm":
            continue
        if r.get("device") != device:
            continue
        if r.get("size") != size:
            continue
        if r.get("mode", "fresh") != mode:
            continue
        if r.get("tf32", False) is True:
            continue
        out.append(r)
    return out


def _parity_check(r, size):
    """1 行の要素単位検証結果（イシュー #970）を検証する。

    戻り値: `(ok, reason)`。`reason` は `ok=False` のときのみ非 None で、
    診断のため fail_count/total/max_abs/max_rel を含む（イシュー #1142
    R2: N=2048 candle 無効データの再現条件を判定不能理由として記録する）。
    summarize.py `parity_status` と同じ判定基準（値域・整数性・
    `parity_total == size*size` の完全一致）を、本ツール用に簡約したもの。
    """
    keys = (
        "parity_fail_count",
        "parity_total",
        "parity_max_abs_err",
        "parity_max_rel_err",
    )
    if any(k not in r for k in keys):
        return False, "parity フィールド欠損（旧形式または破損 JSONL）"
    fail_count, total, max_abs, max_rel = (r.get(k) for k in keys)
    if not all(_is_plain_number(v) for v in (fail_count, total, max_abs, max_rel)):
        return False, "parity フィールドが数値でない"
    if _non_integral(total) or _non_integral(fail_count):
        return False, "parity_total/parity_fail_count が整数でない"
    total = int(total)
    fail_count = int(fail_count)
    if total != size * size:
        return False, f"parity_total が期待要素数と不一致（{total} != {size * size}）"
    if fail_count < 0 or fail_count > total:
        return False, f"parity_fail_count が値域外（{fail_count}/{total}）"
    if max_abs < 0 or max_rel < 0:
        return False, "parity_max_abs_err/parity_max_rel_err が負"
    if fail_count > 0:
        return False, (
            f"要素誤差超過 fail={fail_count}/{total}, "
            f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"
        )
    return True, None


def _median(values):
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def evaluate_size(rows, size, device="cuda"):
    """1 size 分の判定結果を辞書で返す（`status`: "ok"/"undeterminable"）。

    device="cpu" のときのみ、結果に `fandhe_fresh_median_s`（と min/max）を
    追加しうる（イシュー #1148）。これは環境 10/11 の単発 fresh 計測との
    連続性を説明するための参考値であり、`fandhe-ai fresh` 行がちょうど
    `MIN_RECORDS` 件かつ要素単位検証・checksum とも正式判定と同じ検証を
    通った場合のみ付与する。`achieved`（正式判定。reuse vs candle fresh）
    には一切使わない — fresh 行の有無・値は判定結果を変えない。
    """
    fandhe_rows = _matching_rows(rows, "fandhe-ai", FANDHE_MODE, size, device)
    candle_rows = _matching_rows(rows, "candle", CANDLE_MODE, size, device)
    result = {"size": size, "fandhe_n": len(fandhe_rows), "candle_n": len(candle_rows)}

    # 件数は厳密に MIN_RECORDS 件と一致することを要求する（codex-review P1
    # 指摘・PR #1166: 6 件以上ある場合に無条件で `[-MIN_RECORDS:]` の末尾を
    # 採用する実装だと、不利な計測結果が出た後に有利な run を追記するだけで
    # 判定対象の標本を差し替えられてしまい、fail-closed の再現性契約
    # （coding-rust.md「ベンチは 5 回計測の中央値」・AGENTS.md「捏造しない」）
    # が成り立たない。本ツールの入力生成元 `run_gemm_gate_cuda.sh` は 1 回の
    # 起動ごとに `: > "$OUT"` で出力ファイルを新規作成し、N ごとに
    # fandhe-ai/candle を厳密に 5 回ずつ起動する設計（README「計測プロトコル」
    # 節）のため、正常な入力では過不足は生じない。改変不能な run/session ID
    # フィールドは JSONL スキーマに存在しないため、件数の完全一致検証を
    # 標本差し替え耐性の担保手段とする（過不足いずれも判定不能に倒す）。
    if len(fandhe_rows) != MIN_RECORDS:
        result["status"] = "undeterminable"
        result["reason"] = (
            f"fandhe-ai reuse レコード件数が {MIN_RECORDS} 件と不一致"
            f"（{len(fandhe_rows)} 件。標本差し替え防止のため過不足いずれも判定不能）"
        )
        return result
    if len(candle_rows) != MIN_RECORDS:
        result["status"] = "undeterminable"
        result["reason"] = (
            f"candle fresh レコード件数が {MIN_RECORDS} 件と不一致"
            f"（{len(candle_rows)} 件。標本差し替え防止のため過不足いずれも判定不能）"
        )
        return result

    diagnostics = []
    for label, run_rows in (("fandhe-ai", fandhe_rows), ("candle", candle_rows)):
        for i, r in enumerate(run_rows, start=1):
            ok, reason = _parity_check(r, size)
            diagnostics.append(
                {
                    "framework": label,
                    "run": i,
                    "ok": ok,
                    "reason": reason,
                    "fail_count": r.get("parity_fail_count"),
                    "total": r.get("parity_total"),
                    "max_abs": r.get("parity_max_abs_err"),
                    "max_rel": r.get("parity_max_rel_err"),
                }
            )
    result["diagnostics"] = diagnostics
    parity_failures = [d for d in diagnostics if not d["ok"]]
    if parity_failures:
        reasons = "; ".join(f"{d['framework']}#{d['run']}: {d['reason']}" for d in parity_failures)
        result["status"] = "undeterminable"
        result["reason"] = f"要素単位検証が無効（{reasons}）"
        return result

    fandhe_medians = [_safe_positive(r.get("median_s")) for r in fandhe_rows]
    candle_medians = [_safe_positive(r.get("median_s")) for r in candle_rows]
    if any(v is None for v in fandhe_medians):
        result["status"] = "undeterminable"
        result["reason"] = "fandhe-ai の median_s に不正値（非正・NaN・Infinity 等）を含む"
        return result
    if any(v is None for v in candle_medians):
        result["status"] = "undeterminable"
        result["reason"] = "candle の median_s に不正値（非正・NaN・Infinity 等）を含む"
        return result

    fandhe_checksums = [_safe_finite(r.get("checksum")) for r in fandhe_rows]
    candle_checksums = [_safe_finite(r.get("checksum")) for r in candle_rows]
    if any(v is None for v in fandhe_checksums) or any(v is None for v in candle_checksums):
        result["status"] = "undeterminable"
        result["reason"] = "checksum に不正値を含む"
        return result

    # GEMM は全フレームワークで同一入力（xorshift64* の同一シード）のため、
    # checksum は本体の数値一致契約内で一致するはずである（summarize.py
    # `gemm_checksum_reference` と同じ前提）。fandhe-ai 側 5 件を参照値とし、
    # candle 側も含め全件が複合判定を満たすことを要求する。
    ref = fandhe_checksums[0]
    for c in fandhe_checksums + candle_checksums:
        if not checksums_match(c, ref):
            result["status"] = "undeterminable"
            result["reason"] = (
                "checksum が fandhe-ai/candle 間で数値一致契約"
                "（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を外れる"
                f"（参照値 {ref}, 実測 {c}）"
            )
            return result

    fandhe_median = _median(fandhe_medians)
    candle_median = _median(candle_medians)
    result["status"] = "ok"
    result["fandhe_median_s"] = fandhe_median
    result["fandhe_min_s"] = min(fandhe_medians)
    result["fandhe_max_s"] = max(fandhe_medians)
    result["candle_median_s"] = candle_median
    result["candle_min_s"] = min(candle_medians)
    result["candle_max_s"] = max(candle_medians)
    result["ratio_candle_over_fandhe"] = candle_median / fandhe_median if fandhe_median > 0 else None
    result["gflops"] = (2.0 * size**3 / fandhe_median) / 1e9 if fandhe_median > 0 else None
    # #1031/#1037/#1117 の受け入れ条件: reuse の fandhe-ai が candle と同等
    # 以上（中央値で candle 以下の所要時間）。
    result["achieved"] = fandhe_median <= candle_median

    # 参考列（cpu のみ。イシュー #1148）: 環境 10/11 単発計測との連続性
    # 説明のため、fandhe-ai の fresh 中央値（5 件ちょうどかつ要素単位検証・
    # checksum とも正式判定と同じ検証を通った場合のみ）を付記する。判定
    # （achieved）には一切使わない — 正式契約は reuse vs candle fresh の
    # ままとし、fresh 行を判定母集団に混入させない（本関数 docstring・
    # README「GEMM ゲート 5 回計測」節参照）。
    if device == "cpu":
        fresh_rows = _matching_rows(rows, "fandhe-ai", "fresh", size, device)
        if len(fresh_rows) == MIN_RECORDS:
            fresh_diag_ok = all(_parity_check(r, size)[0] for r in fresh_rows)
            fresh_medians = [_safe_positive(r.get("median_s")) for r in fresh_rows]
            fresh_checksums = [_safe_finite(r.get("checksum")) for r in fresh_rows]
            if (
                fresh_diag_ok
                and all(v is not None for v in fresh_medians)
                and all(v is not None for v in fresh_checksums)
                and all(checksums_match(c, ref) for c in fresh_checksums)
            ):
                result["fandhe_fresh_median_s"] = _median(fresh_medians)
                result["fandhe_fresh_min_s"] = min(fresh_medians)
                result["fandhe_fresh_max_s"] = max(fresh_medians)
    return result


def _fmt_s(s):
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.1f} us"


_GATE_ISSUE_BY_DEVICE = {"cuda": "#1031", "metal": "#1037", "cpu": "#1117"}


def render(path, results, device="cuda"):
    lines = []
    gate_issue = _GATE_ISSUE_BY_DEVICE.get(device, "#1031")
    lines.append(f"## GEMM 目標達成ゲート（{gate_issue}・device={device}）: `{path}`")
    lines.append("")
    show_fresh_ref = device == "cpu"
    header = (
        "| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | "
        "candle/fandhe | GFLOP/s | 判定 |"
    )
    sep = "|---|---|---|---|---|---|"
    if show_fresh_ref:
        header = header[:-1] + "| fandhe-ai fresh median（参考。n） |"
        sep = sep + "---|"
    lines.append(header)
    lines.append(sep)
    for r in results:
        if r["status"] != "ok":
            row = f"| {r['size']} | - | - | - | - | 判定不能: {r['reason']} |"
            if show_fresh_ref:
                row += " - |"
            lines.append(row)
            continue
        verdict = "達成" if r["achieved"] else "未達"
        row = (
            f"| {r['size']} | {_fmt_s(r['fandhe_median_s'])} "
            f"({_fmt_s(r['fandhe_min_s'])}–{_fmt_s(r['fandhe_max_s'])}, n=5) | "
            f"{_fmt_s(r['candle_median_s'])} (n=5) | "
            f"{r['ratio_candle_over_fandhe']:.3f} | {r['gflops']:.2f} | {verdict} |"
        )
        if show_fresh_ref:
            if "fandhe_fresh_median_s" in r:
                row += (
                    f" {_fmt_s(r['fandhe_fresh_median_s'])} "
                    f"({_fmt_s(r['fandhe_fresh_min_s'])}–{_fmt_s(r['fandhe_fresh_max_s'])}, n=5) |"
                )
            else:
                row += " - |"
        lines.append(row)
    lines.append("")

    # 診断表（run ごとの要素単位検証。R2: N=2048 無効データの再現条件記録用）。
    for r in results:
        diags = r.get("diagnostics")
        if not diags:
            continue
        lines.append(f"### N={r['size']} 要素単位検証の run 別内訳")
        lines.append("")
        lines.append("| framework | run | fail_count/total | max_abs | max_rel | 判定 |")
        lines.append("|---|---|---|---|---|---|")
        for d in diags:
            fail_str = (
                f"{d['fail_count']}/{d['total']}"
                if d["fail_count"] is not None and d["total"] is not None
                else "?"
            )
            abs_str = f"{d['max_abs']:.6e}" if isinstance(d["max_abs"], (int, float)) else "null"
            rel_str = f"{d['max_rel']:.6e}" if isinstance(d["max_rel"], (int, float)) else "null"
            lines.append(
                f"| {d['framework']} | {d['run']} | {fail_str} | {abs_str} | {rel_str} | "
                f"{'ok' if d['ok'] else d['reason']} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("jsonl", nargs="+", help="run_gemm_gate.sh の出力 JSONL（複数可・独立集計）")
    parser.add_argument(
        "--device",
        choices=("cuda", "metal", "cpu"),
        default="cuda",
        help=(
            "集計対象の device（既定 cuda。#1142 との後方互換。Metal は #1037 ゲート・"
            "イシュー #1147。cpu は #1117 ゲート・イシュー #1148）"
        ),
    )
    parser.add_argument("--out", help="出力先ファイル（省略時は標準出力）")
    args = parser.parse_args(argv)

    exit_code = 0
    out_parts = []
    for path in args.jsonl:
        try:
            rows, warnings = load_rows(path)
        except OSError as e:
            print(f"error: {path} を読み込めない（{e}）", file=sys.stderr)
            exit_code = max(exit_code, 2)
            continue
        for w in warnings:
            print(f"warning: {w}", file=sys.stderr)
        results = [evaluate_size(rows, size, args.device) for size in _SIZES_BY_DEVICE[args.device]]
        if warnings:
            # codex-review P0 指摘（PR #1166）: `load_rows` が破損 JSON・非
            # object・不正な `tf32` 型の行を warnings として除外するのみで、
            # 従来はここで標準エラーへ表示するだけに留まり終了コードへ反映
            # されなかった。正常な 5 行さえ揃えば同一入力ファイルに破損行・
            # 不正 tf32 行が混在していてもゲートが exit code 0 になり得る
            # fail-open な欠陥（外部フォーマット検証失敗を受理する A03・
            # fail-closed な性能判定を fail-open にする A08 相当）だった。
            # warnings が 1 件でもある入力ファイルは、正常行のみで算出した
            # 各 size の判定を丸ごと判定不能へ上書きし「達成」扱いにしない
            # （fail-closed。#1031 のゲート達成判定を汚染された入力から
            # 確定させない）。
            for r in results:
                r["status"] = "undeterminable"
                r["reason"] = (
                    f"入力ファイルに不正行が {len(warnings)} 件あり判定不能"
                    "（破損 JSON・非 object・不正な 'tf32' 型のいずれか。"
                    "詳細は上記 warning 行を参照）"
                )
                r.pop("achieved", None)
        out_parts.append(render(path, results, args.device))
        for r in results:
            if r["status"] != "ok":
                print(f"undeterminable: {path} size={r['size']}（{r['reason']}）", file=sys.stderr)
                exit_code = max(exit_code, 3)
            elif not r["achieved"]:
                print(
                    f"未達: {path} size={r['size']}（fandhe-ai {r['fandhe_median_s']:.6f}s "
                    f"> candle {r['candle_median_s']:.6f}s）",
                    file=sys.stderr,
                )
                exit_code = max(exit_code, 3)

    text = "\n".join(out_parts)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
