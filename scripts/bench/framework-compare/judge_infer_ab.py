#!/usr/bin/env python3
"""イシュー #1580: Metal 推論 forward チェーン単一同期化の A/B 判定。

`run_ab_infer_chain_metal.sh` が出力する before/after 2 本の JSONL
（`--task infer`）を読み、`mode` ごとに `median_s`／`checksum` を突き合わせる。
`compare_gemm_ab.py` は `--task train` 限定で `--phases` を受け付ける設計の
ため、他イシューと共有する同ツールを本イシューの都合で変更せず（衝突回避）、
本ファイルを `--task infer` 専用の自己完結ツールとして新設する。

fail-closed 方針（security.md A08。`compare_gemm_gate.py` 等と同方針）:
- `framework != "fandhe-ai"` の行・`device != "metal"` の行・`task !=
  "infer"` の行は判定対象から除外する
- 各セル（`mode` ごと）が before/after とも「ちょうど `--rounds`（既定 5）
  件」でなければ「判定不能」
- `warmup`/`iters` が before/after で「欠落」または「非 int（bool を含む）」
  なら「判定不能」（両腕とも欠落すると `{None} == {None}` で一致扱いに
  なってしまう誤りを防ぐ。codex-review 指摘対応）。存在する場合は
  `warmup >= 0`・`iters > 0` を要求し、before/after 間で値集合が一致
  しなければ「判定不能」
- セル内のいずれかの行で `median_s` が欠落・非数値（bool を含む）・
  非有限（`inf`/`-inf`/`nan`）・非正値、または `checksum` が欠落・
  非数値（bool・文字列・配列等を含む）であれば「判定不能」（欠落を
  「一致」として扱わない。codex-review 指摘対応）

事前登録判定規則（イシュー #1580 計画）:
- 判定セル: `mode == "reuse"`。対照（非判定・参考記録のみ）: `mode ==
  "fresh"`
- 判定: 全 round 中央値の比（`after_median / before_median`）が
  `--threshold`（既定 1.00）以下、かつ checksum が全 round 完全一致
  （`==`）
  → ADOPT（`ratio <= threshold` かつ `checksum_exact_match`）
- いずれか不成立 → REJECT
- 判定セル（`reuse`）の計測行が「ちょうど `--rounds` 件」でない・
  warmup/iters 不一致 等で判定不能 → undetermined
"""

import argparse
import json
import math
import statistics
import sys


def _load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("framework") != "fandhe-ai":
                continue
            if obj.get("device") != "metal":
                continue
            if obj.get("task") != "infer":
                continue
            rows.append(obj)
    return rows


def _cell_rows(rows, mode):
    return [r for r in rows if r.get("mode") == mode]


def _judge_cell(before_rows, after_rows, rounds, threshold):
    """1 セル（`mode` 固定）の判定を行い `(verdict, detail_dict)` を返す。

    `verdict` は "ADOPT"／"REJECT"／"undetermined" のいずれか。

    fail-closed 方針（codex-review 指摘対応。security.md A08 と同方針）:
    `warmup`／`iters`／`median_s`／`checksum` のいずれかが欠落・不正な型
    （bool を含む）・規定範囲外（`warmup < 0`／`iters <= 0`／`median_s`
    が非有限〈`inf`/`-inf`/`nan`〉または非正値）の行が 1 件でもあれば、
    それだけで "undetermined" とする。とくに `checksum`／`warmup`／
    `iters` が両腕とも欠落している場合に `None == None` や `{None} ==
    {None}` で「完全一致」と誤判定しないことを保証する（欠落は「一致が
    確認できていない」であって「一致」ではない）
    """
    if len(before_rows) != rounds or len(after_rows) != rounds:
        return "undetermined", {
            "reason": f"expected exactly {rounds} rows each (before={len(before_rows)}, "
            f"after={len(after_rows)})"
        }

    all_rows = before_rows + after_rows
    for r in all_rows:
        warmup = r.get("warmup")
        if not isinstance(warmup, int) or isinstance(warmup, bool) or warmup < 0:
            return "undetermined", {
                "reason": f"missing or invalid warmup (expected non-negative int): {warmup!r}"
            }
        iters = r.get("iters")
        if not isinstance(iters, int) or isinstance(iters, bool) or iters <= 0:
            return "undetermined", {
                "reason": f"missing or invalid iters (expected positive int): {iters!r}"
            }
        median_s = r.get("median_s")
        if (
            not isinstance(median_s, (int, float))
            or isinstance(median_s, bool)
            or not math.isfinite(median_s)
            or median_s <= 0
        ):
            return "undetermined", {
                "reason": f"missing, non-finite, or non-positive median_s: {median_s!r}"
            }
        checksum = r.get("checksum")
        if (
            checksum is None
            or isinstance(checksum, bool)
            or not isinstance(checksum, (int, float))
        ):
            return "undetermined", {
                "reason": f"missing or non-numeric checksum: {checksum!r}"
            }

    before_warmup = {r["warmup"] for r in before_rows}
    after_warmup = {r["warmup"] for r in after_rows}
    before_iters = {r["iters"] for r in before_rows}
    after_iters = {r["iters"] for r in after_rows}
    if len(before_warmup) != 1 or len(after_warmup) != 1 or before_warmup != after_warmup:
        return "undetermined", {"reason": "warmup mismatch between before/after"}
    if len(before_iters) != 1 or len(after_iters) != 1 or before_iters != after_iters:
        return "undetermined", {"reason": "iters mismatch between before/after"}

    before_medians = [r["median_s"] for r in before_rows]
    after_medians = [r["median_s"] for r in after_rows]
    before_median = statistics.median(before_medians)
    after_median = statistics.median(after_medians)
    ratio = after_median / before_median

    before_checksums = [r["checksum"] for r in before_rows]
    after_checksums = [r["checksum"] for r in after_rows]
    checksum_exact_match = before_checksums == after_checksums

    detail = {
        "before_median_s": before_median,
        "after_median_s": after_median,
        "ratio": ratio,
        "checksum_exact_match": checksum_exact_match,
        "before_checksums": before_checksums,
        "after_checksums": after_checksums,
    }
    if ratio <= threshold and checksum_exact_match:
        return "ADOPT", detail
    return "REJECT", detail


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, help="before 腕の JSONL パス")
    parser.add_argument("--after", required=True, help="after 腕の JSONL パス")
    parser.add_argument("--rounds", type=int, default=5, help="判定セルの必須行数（既定 5）")
    parser.add_argument("--threshold", type=float, default=1.00, help="非後退判定の閾値（既定 1.00）")
    args = parser.parse_args(argv)

    before_rows = _load_rows(args.before)
    after_rows = _load_rows(args.after)

    lines = ["# infer chain metal A/B (issue #1580)", ""]
    judged_verdict = None

    for mode, is_judged in (("reuse", True), ("fresh", False)):
        before_cell = _cell_rows(before_rows, mode)
        after_cell = _cell_rows(after_rows, mode)
        verdict, detail = _judge_cell(before_cell, after_cell, args.rounds, args.threshold)
        label = "判定対象" if is_judged else "対照（非判定）"
        lines.append(f"## mode={mode} ({label})")
        lines.append(f"- verdict: {verdict}")
        for k, v in detail.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        if is_judged:
            judged_verdict = verdict

    print("\n".join(lines))
    # 終了コード契約（`compare_gemm_ab.py` と同型）: 0=ADOPT（非後退）・
    # 2=判定不能・3=REJECT（後退）。判定対象セル（reuse）のみで決める
    # （fresh は対照であり終了コードへ影響しない）。
    if judged_verdict == "ADOPT":
        return 0
    if judged_verdict == "undetermined":
        return 2
    return 3


if __name__ == "__main__":
    sys.exit(main())
