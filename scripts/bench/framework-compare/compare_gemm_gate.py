#!/usr/bin/env python3
"""GEMM 目標達成ゲート（#1031）の 5 回計測中央値集計ツール（イシュー #1142）。

使い方:
    python3 compare_gemm_gate.py JSONL [JSONL ...] [--out FILE]

`run_gemm_gate_cuda.sh` が出力する JSONL（`bench-fandhe gemm cuda <N> reuse`・
`bench-candle gemm cuda <N> fresh` を N=1024/2048/4096 それぞれ 5 回起動した
結果）を読み、N ごとに fandhe-ai（reuse）vs candle（fresh。candle は reuse
非対応のため fresh 固定）の `median_s` を run 間中央値で集約し、#1031 の受け
入れ条件（`fandhe_median_s <= candle_median_s`）を判定する。

`summarize.py --target candle` は同一入力ファイル内の 1 レコードのみを
拾う設計（1 ファイル = 1 環境の単発計測が前提）のため、5 回計測の run 間
中央値（`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」）には
非対応。本ツールはその欠落を埋める専用集計（`compare_ab.py` が train タスク
向けに担っている 5 回集計を gemm タスク向けに用意したもの）。

設計方針（`compare_ab.py`・`summarize.py` と同じ思想。tolerance 定数・
`checksums_match` は `checksum_contract.py` を単一真実源として共有する）:
- fail-closed: レコード不足（size ごとに fandhe-ai/candle 各 5 件未満）・
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

# #1031 の対象形状（reuse candle 比再計測。イシュー #1142）。
SIZES = (1024, 2048, 4096)
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
            rows.append(obj)
    return rows, warnings


def _matching_rows(rows, framework, mode, size):
    """`(framework, task=gemm, device=cuda, size, mode)` に一致する行を返す。

    `tf32:true` の行は除外する（FP32 目標値との混同防止。summarize.py と
    同じ既定）。`mode` キー欠損は "fresh" 扱い（イシュー #925 互換規約）。
    """
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("framework") != framework:
            continue
        if r.get("task") != "gemm":
            continue
        if r.get("device") != "cuda":
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


def evaluate_size(rows, size):
    """1 size 分の判定結果を辞書で返す（`status`: "ok"/"undeterminable"）。"""
    fandhe_rows = _matching_rows(rows, "fandhe-ai", FANDHE_MODE, size)
    candle_rows = _matching_rows(rows, "candle", CANDLE_MODE, size)
    result = {"size": size, "fandhe_n": len(fandhe_rows), "candle_n": len(candle_rows)}

    if len(fandhe_rows) < MIN_RECORDS:
        result["status"] = "undeterminable"
        result["reason"] = f"fandhe-ai reuse レコード不足（{len(fandhe_rows)} 件 < {MIN_RECORDS} 件必要）"
        return result
    if len(candle_rows) < MIN_RECORDS:
        result["status"] = "undeterminable"
        result["reason"] = f"candle fresh レコード不足（{len(candle_rows)} 件 < {MIN_RECORDS} 件必要）"
        return result

    # 直近 MIN_RECORDS 件（末尾）を対象にする: run スクリプトが 5 回超を
    # 追記した場合でも最新の計測のみを判定に使う（append 運用の JSONL に
    # 過去分が混在しても古い記録を判定不能の原因にしない）。
    fandhe_rows = fandhe_rows[-MIN_RECORDS:]
    candle_rows = candle_rows[-MIN_RECORDS:]

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
    # #1031 の受け入れ条件: reuse の fandhe-ai が candle と同等以上（中央値
    # で candle 以下の所要時間）。
    result["achieved"] = fandhe_median <= candle_median
    return result


def _fmt_s(s):
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.1f} us"


def render(path, results):
    lines = []
    lines.append(f"## GEMM 目標達成ゲート（#1031）: `{path}`")
    lines.append("")
    lines.append(
        "| N | fandhe-ai reuse median (min–max, n) | candle fresh median (n) | "
        "candle/fandhe | GFLOP/s | 判定 |"
    )
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        if r["status"] != "ok":
            lines.append(
                f"| {r['size']} | - | - | - | - | 判定不能: {r['reason']} |"
            )
            continue
        verdict = "達成" if r["achieved"] else "未達"
        lines.append(
            f"| {r['size']} | {_fmt_s(r['fandhe_median_s'])} "
            f"({_fmt_s(r['fandhe_min_s'])}–{_fmt_s(r['fandhe_max_s'])}, n=5) | "
            f"{_fmt_s(r['candle_median_s'])} (n=5) | "
            f"{r['ratio_candle_over_fandhe']:.3f} | {r['gflops']:.2f} | {verdict} |"
        )
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
    parser.add_argument("jsonl", nargs="+", help="run_gemm_gate_cuda.sh の出力 JSONL（複数可・独立集計）")
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
        results = [evaluate_size(rows, size) for size in SIZES]
        out_parts.append(render(path, results))
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
