#!/usr/bin/env python3
"""都度同期廃止（#1011）の実践規模 A/B 比較ツール（イシュー #1083）。

使い方:
    python3 compare_ab.py BEFORE.jsonl AFTER.jsonl [--out FILE]

`run_ab_train_cuda.sh` が出力する 2 本の JSONL（`bench-fandhe train cuda 64`
を fresh/reuse 各 5 回起動した結果）を読み、`(mode)` ごとに before/after の
`median_s` を集約して Markdown 表を出力する。

設計方針（summarize.py と同じ思想を踏襲。二重管理を避けるため関数は
再実装するが、判定基準の定数〈CHECKSUM_ABS_TOL/CHECKSUM_REL_TOL〉と
複合判定〈checksums_match〉のロジックは summarize.py と同一の値・アルゴリズム
を用いる）:
- fail-closed: レコード不足（5 件未満）・`framework_version` が before/after
  で同一（比較にならない）・最終 loss（checksum）が本体の数値一致契約
  （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。coding-rust.md）を外れる
  場合は、性能値を確定表示せず「判定不能」を明示し、終了コードを非 0 にする
  （A08 データ整合性: 壊れた比較を性能改善として報告しない）。
- 捏造しない: 入力 JSONL 自体は変更しない。集計不能な行は理由付きでスキップし
  黙って無視しない。
- train_phases 行があれば phase 別の before/after 表も出す（診断用。§2 の
  同期点分析の裏付け）。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys

# summarize.py の CHECKSUM_ABS_TOL/CHECKSUM_REL_TOL と同一値（本体の数値一致
# 契約: 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。coding-rust.md）。
CHECKSUM_ABS_TOL = 1e-5
CHECKSUM_REL_TOL = 1e-3

# 本比較で対象にする mode（bench-fandhe train タスクが emit する 2 モード）。
MODES = ["fresh", "reuse"]

# 最小限必要なレコード数（coding-rust.md「ベンチは 5 回計測の中央値」）。
MIN_RECORDS = 5

# `run_all_cuda.sh`・`run_ab_train_cuda.sh` はいずれも train タスクを
# `size=64` 固定でのみ起動する（両スクリプト実測）。現状の JSONL は train
# レコードが size=64 のみで構成されるため実害はないが、将来 size の異なる
# train レコードが同一 JSONL に混在した場合に compare_mode() が size を
# またいで中央値を算出してしまう defensive gap（Review 指摘・#1083）を
# 塞ぐため、`_train_records` で size=64 に明示的に絞り込む。
TRAIN_SIZE = 64

_PHASE_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _is_plain_number(v):
    """外部 JSONL 由来の値が bool・NaN・Infinity・非数値でない「素の数値」か
    検証する（summarize.py `_is_plain_number` と同一の判定基準。二重管理を
    避けるためロジックのみ複製し、A03 の「外部入力の型を信頼しない」思想を
    本ツールでも独立に適用する）。
    """
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return math.isfinite(v)


def _safe_positive(v):
    """時間値（median_s 等）を検証する。有効なら float・無効なら None。"""
    if not _is_plain_number(v):
        return None
    try:
        fv = float(v)
    except OverflowError:
        return None
    return fv if fv > 0 else None


def _safe_finite(v):
    """checksum 等、正値制約のない数値を検証する。有効なら float・無効なら None。"""
    if not _is_plain_number(v):
        return None
    try:
        return float(v)
    except OverflowError:
        return None


def checksums_match(a, b):
    """本体の数値一致契約と同一の複合判定（相対誤差 1e-3 未満 または 絶対誤差
    1e-5 未満）。summarize.py `checksums_match` と同一アルゴリズム（対称な
    分母 `max(|a|, |b|, 1e-12)`）。
    """
    diff = abs(a - b)
    if diff < CHECKSUM_ABS_TOL:
        return True
    denom = max(abs(a), abs(b), 1e-12)
    return diff / denom < CHECKSUM_REL_TOL


def load_rows(path):
    """JSONL を読み、不正な行（JSON として parse 不能）は理由付きで報告し
    スキップする（A08: 捏造しない。呼び出し元は戻り値の 2 要素目〈警告文の
    リスト〉を表示・記録する）。
    """
    rows = []
    warnings = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                warnings.append(f"{path}:{lineno}: invalid JSON ({e}) — skipped")
    return rows, warnings


def _train_records(rows, mode):
    """`framework == fandhe-ai`・`task == train`・`device == cuda`・
    `size == TRAIN_SIZE`・指定 mode の行だけを、`row.get("mode", "fresh")`
    （mode キー欠損は fresh 扱い。summarize.py `load_rows` と同じ互換方針）
    で絞り込む。

    size 絞り込みは fail-closed（`r.get("size") != TRAIN_SIZE` は除外）とし、
    size の異なる train レコードが同一 JSONL に混在しても中央値を size を
    またいで算出しない（Review 指摘・#1083）。
    """
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("framework") != "fandhe-ai":
            continue
        if r.get("task") != "train":
            continue
        if r.get("device") != "cuda":
            continue
        if r.get("size") != TRAIN_SIZE:
            continue
        if r.get("mode", "fresh") != mode:
            continue
        out.append(r)
    return out


def _median(values):
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _fmt_ms(s):
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.1f} us"


def compare_mode(before_rows, after_rows, mode):
    """1 mode 分の before/after 比較結果を辞書で返す。

    フィールド:
      mode, status（"ok"/"undeterminable"）, reason（undeterminable のみ）,
      before_median_s, after_median_s, before_n, after_n, ratio,
      before_version, after_version, before_checksum, after_checksum.
    """
    before = _train_records(before_rows, mode)
    after = _train_records(after_rows, mode)
    result = {"mode": mode}

    if len(before) < MIN_RECORDS:
        result["status"] = "undeterminable"
        result["reason"] = (
            f"before レコード不足（{len(before)} 件 < {MIN_RECORDS} 件必要）"
        )
        return result
    if len(after) < MIN_RECORDS:
        result["status"] = "undeterminable"
        result["reason"] = (
            f"after レコード不足（{len(after)} 件 < {MIN_RECORDS} 件必要）"
        )
        return result

    before_versions = {r.get("version") for r in before}
    after_versions = {r.get("version") for r in after}
    if len(before_versions) != 1 or len(after_versions) != 1:
        result["status"] = "undeterminable"
        result["reason"] = "before/after の framework_version が単一値でない（混在入力）"
        return result
    before_version = next(iter(before_versions))
    after_version = next(iter(after_versions))
    if before_version == after_version:
        result["status"] = "undeterminable"
        result["reason"] = (
            f"before/after の framework_version が同一（{before_version!r}）"
            "— A/B 比較になっていない"
        )
        return result

    before_medians = [_safe_positive(r.get("median_s")) for r in before]
    after_medians = [_safe_positive(r.get("median_s")) for r in after]
    if any(v is None for v in before_medians):
        result["status"] = "undeterminable"
        result["reason"] = "before の median_s に不正値（非正・NaN・Infinity 等）を含む"
        return result
    if any(v is None for v in after_medians):
        result["status"] = "undeterminable"
        result["reason"] = "after の median_s に不正値（非正・NaN・Infinity 等）を含む"
        return result

    before_checksums = [_safe_finite(r.get("checksum")) for r in before]
    after_checksums = [_safe_finite(r.get("checksum")) for r in after]
    if any(v is None for v in before_checksums) or any(v is None for v in after_checksums):
        result["status"] = "undeterminable"
        result["reason"] = "checksum（最終 loss）に不正値を含む"
        return result

    # 最終 loss（checksum）は同一プロトコル・同一入力（同一シード）であれば
    # before/after で一致するはずである（reuse は DeviceParamStore の更新
    # 経路のみが変わり数値契約は変えない設計。#957/#958）。5 反復すべてが
    # 複合判定を満たすことを要求する（1 件でも外れれば判定不能）。
    before_checksum_ref = before_checksums[0]
    after_checksum_ref = after_checksums[0]
    for c in before_checksums + after_checksums:
        if not checksums_match(c, before_checksum_ref):
            result["status"] = "undeterminable"
            result["reason"] = (
                "checksum（最終 loss）が before/after 間で数値一致契約"
                "（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を外れる"
                f"（参照値 {before_checksum_ref}, 実測 {c}）"
            )
            return result

    before_median = _median(before_medians)
    after_median = _median(after_medians)

    result["status"] = "ok"
    result["before_median_s"] = before_median
    result["after_median_s"] = after_median
    result["before_min_s"] = min(before_medians)
    result["before_max_s"] = max(before_medians)
    result["after_min_s"] = min(after_medians)
    result["after_max_s"] = max(after_medians)
    result["before_n"] = len(before)
    result["after_n"] = len(after)
    result["ratio"] = after_median / before_median if before_median > 0 else None
    result["before_version"] = before_version
    result["after_version"] = after_version
    result["before_checksum"] = before_checksum_ref
    result["after_checksum"] = after_checksum_ref
    return result


def _valid_phase_row(r):
    """`train_phases` 行（診断用・イシュー #1009）として表示に使ってよいか。

    phase の中身（`tape_build` 等）は sub-ns 区間の計時分解能限界で
    `median_s == 0` になりうる（summarize.py `_safe_phase_time_s` と同じ
    理由）ため、ここでは「0 以上の有限値」のみを要求する（負値・NaN・
    Infinity・非数値は無効）。
    """
    if not isinstance(r, dict):
        return False
    if r.get("framework") != "fandhe-ai" or r.get("task") != "train_phases":
        return False
    if r.get("device") != "cuda":
        return False
    phase = r.get("phase")
    if not isinstance(phase, str) or not _PHASE_NAME_RE.match(phase):
        return False
    v = _safe_finite(r.get("median_s"))
    return v is not None and v >= 0


def compare_phases(before_rows, after_rows, mode):
    """`train_phases` 行（診断用。イシュー #1009）の phase 別 before/after を
    集約する。フェーズ計測は 1 回のみ（run_ab_train_cuda.sh 参照）のため
    5 回中央値の対象外— 参考値として before/after の値をそのまま並べる。
    """
    before = [
        r
        for r in before_rows
        if _valid_phase_row(r) and r.get("mode", "fresh") == mode
    ]
    after = [
        r for r in after_rows if _valid_phase_row(r) and r.get("mode", "fresh") == mode
    ]
    if not before or not after:
        return []

    before_by_phase = {r["phase"]: r for r in before}
    after_by_phase = {r["phase"]: r for r in after}
    phases = sorted(
        set(before_by_phase) & set(after_by_phase),
        key=lambda p: (
            before_by_phase[p].get("phase_index", 1 << 30),
            p,
        ),
    )
    rows_out = []
    for p in phases:
        b = _safe_finite(before_by_phase[p].get("median_s"))
        a = _safe_finite(after_by_phase[p].get("median_s"))
        if b is None or a is None or b < 0 or a < 0:
            continue
        rows_out.append(
            {
                "phase": p,
                "before_s": b,
                "after_s": a,
                "ratio": (a / b) if b > 0 else None,
            }
        )
    return rows_out


def render(results, phase_results_by_mode, before_path, after_path):
    lines = []
    lines.append("# 都度同期廃止（#1011）A/B 計測（イシュー #1083）")
    lines.append("")
    lines.append(f"- before: `{before_path}`")
    lines.append(f"- after: `{after_path}`")
    lines.append("")
    lines.append("## 1 step 総和（5 回計測の中央値）")
    lines.append("")
    lines.append(
        "| mode | before version | after version | before median | after median | "
        "after/before | 判定 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    any_undeterminable = False
    for r in results:
        if r["status"] != "ok":
            any_undeterminable = True
            lines.append(
                f"| {r['mode']} | - | - | - | - | - | 判定不能: {r['reason']} |"
            )
            continue
        lines.append(
            f"| {r['mode']} | {r['before_version']} | {r['after_version']} | "
            f"{_fmt_ms(r['before_median_s'])} (n={r['before_n']}) | "
            f"{_fmt_ms(r['after_median_s'])} (n={r['after_n']}) | "
            f"{r['ratio']:.3f} | ok |"
        )
    lines.append("")

    for mode, phase_rows in phase_results_by_mode.items():
        if not phase_rows:
            continue
        lines.append(f"## フェーズ分解（診断用・{mode}・単発計測）")
        lines.append("")
        lines.append("| phase | before | after | after/before |")
        lines.append("|---|---|---|---|")
        for pr in phase_rows:
            ratio = f"{pr['ratio']:.3f}" if pr["ratio"] is not None else "-"
            lines.append(
                f"| {pr['phase']} | {_fmt_ms(pr['before_s'])} | "
                f"{_fmt_ms(pr['after_s'])} | {ratio} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n", any_undeterminable


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", help="before（都度同期あり）の JSONL パス")
    parser.add_argument("after", help="after（都度同期廃止後）の JSONL パス")
    parser.add_argument("--out", help="出力先ファイル（省略時は標準出力）")
    args = parser.parse_args(argv)

    before_rows, before_warnings = load_rows(args.before)
    after_rows, after_warnings = load_rows(args.after)
    for w in before_warnings + after_warnings:
        print(f"warning: {w}", file=sys.stderr)

    results = [compare_mode(before_rows, after_rows, mode) for mode in MODES]
    phase_results_by_mode = {
        mode: compare_phases(before_rows, after_rows, mode) for mode in MODES
    }

    text, any_undeterminable = render(
        results, phase_results_by_mode, args.before, args.after
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text, end="")

    if any_undeterminable:
        for r in results:
            if r["status"] != "ok":
                print(f"undeterminable: mode={r['mode']}（{r['reason']}）", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
