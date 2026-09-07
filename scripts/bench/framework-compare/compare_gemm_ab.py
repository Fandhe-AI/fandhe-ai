#!/usr/bin/env python3
"""イシュー #1306: Metal GEMM の before（正式系列 `fandhe-ai =0.7.0`）/
after（参考系列 HEAD。`crates/facade` への path patch）2 バイナリ A/B 比較。

`run_ab_gemm_metal.sh` が出力する 2 本の JSONL（`bench-fandhe --task gemm
--device metal` を N=512/1024/2048/4096 × fresh/reuse × 5 回起動した結果）
を読み、`(size, mode)` セルごとに before/after の `median_s` を集約して
Markdown 表を出力する。

`compare_ab.py` は `framework_version` が before/after で同一だと fail-closed
拒否する（同一バージョンの A/B は意味を持たないという前提）ため、before/after
とも `fandhe-ai =0.7.0` を名乗る本用途（before=registry・after=HEAD path
patch。バージョン文字列は変わらない）には流用できない。`compare_managed_ab.py`
は同一バイナリのフラグ（`managed`）切替専用で、2 本の異なるバイナリを別ファイル
として比較する構造を持たない。本ツールは両者の fail-closed 方針を踏襲しつつ、
2 ファイル入力・`(size, mode)` セルキー（device=metal・task=gemm 固定）に
特化する。

fail-closed 方針（security.md A08。`compare_managed_ab.py` と同方針）:
- `framework != "fandhe-ai"`・`task != "gemm"`・`device != "metal"`・
  `tf32:true`・`managed:true` の行は判定不能行として除外する
- 各セル before/after とも「ちょうど 5 件」でなければ「判定不能」
- `warmup`/`iters`/`version` が before/after で不一致なら「判定不能」
- checksum が複合判定（`checksum_contract.checksums_match`）を外れれば
  「判定不能」。本番カーネル・選択結果は不変のため bit 完全一致が期待値
  であり、完全一致（`==`）列を別途表示して複合判定 pass のみとの契約破れ
  を区別する
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _import_from_path(name, filename):
    spec = importlib.util.spec_from_file_location(name, _HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checksum_contract = _import_from_path("checksum_contract", "checksum_contract.py")

_VALID_MODES = frozenset({"fresh", "reuse"})
_VALID_SIZES = frozenset({512, 1024, 2048, 4096})

# 既定の非後退閾値（median 比 ratio = after/before が 1.05 以下なら非後退。
# guardrail「劣化中央値 5% 以内」の慣例値。閾値変更は本ツールの CLI 引数
# としてのみ与え、コード定数としては固定しない。判定に使う値は既定のまま
# docs へ記録する）。
DEFAULT_THRESHOLD = 1.05


def _valid_cell_identity(obj):
    """`_cell_key` がグループ化に使う `task`/`device`/`size`/`mode` の型・
    値域を検証する（`compare_managed_ab.py::_valid_cell_identity` と同方針。
    未検証のまま集約すると、これらのフィールドを欠いた行が単一の偽セルへ
    迂回して集約され、比較対象が不明なまま "ok" 判定になりうる）。
    """
    task = obj.get("task")
    if not isinstance(task, str) or task != "gemm":
        return False
    device = obj.get("device")
    if not isinstance(device, str) or device != "metal":
        return False
    mode = obj.get("mode", "fresh")
    if not isinstance(mode, str) or mode not in _VALID_MODES:
        return False
    size = obj.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size not in _VALID_SIZES:
        return False
    return True


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
            if obj.get("framework") != "fandhe-ai":
                warnings.append(
                    f"{path}:{lineno}: 'framework' が 'fandhe-ai' ではない"
                    f"（実際: {obj.get('framework')!r}） — skipped"
                )
                continue
            if "tf32" in obj and not isinstance(obj["tf32"], bool):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'tf32' フィールド型（bool を期待。"
                    f"実際: {obj['tf32']!r}） — skipped"
                )
                continue
            if obj.get("tf32", False) is True:
                warnings.append(
                    f"{path}:{lineno}: 'tf32:true' の行は本 A/B の対象外 — skipped"
                )
                continue
            if "managed" in obj and not isinstance(obj["managed"], bool):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'managed' フィールド型（bool を期待。"
                    f"実際: {obj['managed']!r}） — skipped"
                )
                continue
            if obj.get("managed", False) is True:
                warnings.append(
                    f"{path}:{lineno}: 'managed:true' の行は本 A/B の対象外 — skipped"
                )
                continue
            if not _valid_cell_identity(obj):
                warnings.append(
                    f"{path}:{lineno}: 不正または欠損した 'task'/'device'/'size'/"
                    f"'mode' フィールド（行: {obj!r}） — skipped"
                )
                continue
            rows.append(obj)
    return rows, warnings


def _cell_key(r):
    """`(size, mode)`。task=gemm・device=metal は本ツールの対象を固定する
    ため、`_valid_cell_identity` で検証済みでありセルキーには含めない。
    """
    return (r.get("size"), r.get("mode", "fresh"))


def group_by_cell(rows):
    cells = {}
    for r in rows:
        key = _cell_key(r)
        cells.setdefault(key, []).append(r)
    return cells


def _valid_field_value(field, v):
    if isinstance(v, bool):
        return False
    if field == "warmup":
        return isinstance(v, int) and v >= 0
    if field == "iters":
        return isinstance(v, int) and v >= 1
    if field == "version":
        return isinstance(v, str) and v != ""
    return True


def _median(values):
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def evaluate_cell(before_rows, after_rows, threshold):
    """1 セル分の before/after 行を突合し判定結果 dict を返す。

    戻り値のキー: `status`（"ok"|"undeterminable"）・`reason`（undeterminable
    のときのみ非 None）・`before_median_s`/`after_median_s`/`ratio`
    （= after_median_s / before_median_s。値 <= threshold は非後退）・
    `before_min_s`/`before_max_s`/`after_min_s`/`after_max_s`・
    `checksum_composite_match`/`checksum_exact_match`・`verdict`。
    """
    if len(before_rows) != 5 or len(after_rows) != 5:
        return {
            "status": "undeterminable",
            "reason": (
                f"before={len(before_rows)} 件・after={len(after_rows)} 件"
                "（各ちょうど 5 件を要求）"
            ),
        }
    for field in ("warmup", "iters", "version"):
        before_raw = [r.get(field) for r in before_rows]
        after_raw = [r.get(field) for r in after_rows]
        if None in before_raw or None in after_raw:
            return {
                "status": "undeterminable",
                "reason": f"'{field}' が before または after の行に欠損している",
            }
        invalid_values = [
            v for v in before_raw + after_raw if not _valid_field_value(field, v)
        ]
        if invalid_values:
            return {
                "status": "undeterminable",
                "reason": f"'{field}' に不正な値がある（{invalid_values!r}）",
            }
        before_values = set(before_raw)
        after_values = set(after_raw)
        if len(before_values) != 1 or len(after_values) != 1:
            return {
                "status": "undeterminable",
                "reason": f"'{field}' が before または after 内で不一致",
            }
        # `version` はいずれも "0.7.0" を名乗ることが前提（before=registry・
        # after=HEAD path patch。workspace.package.version 不変）。before/after
        # 間で異なる場合は前提が崩れているため判定不能とする。
        if before_values != after_values:
            return {
                "status": "undeterminable",
                "reason": (
                    f"'{field}' が before/after 間で不一致（before={before_values!r} "
                    f"after={after_values!r}）"
                ),
            }

    for label, group_rows in (("before", before_rows), ("after", after_rows)):
        for r in group_rows:
            v = r.get("median_s")
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return {
                    "status": "undeterminable",
                    "reason": f"{label} 側 median_s が数値ではない（{v!r}）",
                }
            if not math.isfinite(v) or v <= 0:
                return {
                    "status": "undeterminable",
                    "reason": f"{label} 側 median_s が有限正数ではない（{v!r}）",
                }
        parity = [r.get("parity_fail_count") for r in group_rows]
        for pf in parity:
            if pf is not None and (isinstance(pf, bool) or not isinstance(pf, int) or pf < 0):
                return {
                    "status": "undeterminable",
                    "reason": f"{label} 側 parity_fail_count が不正（{pf!r}）",
                }
            if isinstance(pf, int) and not isinstance(pf, bool) and pf > 0:
                return {
                    "status": "undeterminable",
                    "reason": f"{label} 側 parity_fail_count > 0（{pf!r}）",
                }

    before_median = _median([r["median_s"] for r in before_rows])
    after_median = _median([r["median_s"] for r in after_rows])
    if before_median <= 0:
        return {"status": "undeterminable", "reason": "before_median_s <= 0"}

    before_checksums = [r.get("checksum") for r in before_rows]
    after_checksums = [r.get("checksum") for r in after_rows]
    if any(
        c is None or isinstance(c, bool) or not isinstance(c, (int, float))
        for c in before_checksums + after_checksums
    ):
        return {
            "status": "undeterminable",
            "reason": "checksum 欠損または非数値の行あり",
        }
    ref_checksum = before_checksums[0]
    composite_ok = all(
        checksum_contract.checksums_match(ref_checksum, c)
        for c in before_checksums + after_checksums
    )
    exact_ok = all(c == ref_checksum for c in before_checksums + after_checksums)
    if not composite_ok:
        return {
            "status": "undeterminable",
            "reason": (
                f"checksum が複合判定を外れる（before={before_checksums!r} "
                f"after={after_checksums!r}）"
            ),
        }

    ratio = after_median / before_median
    before_min = min(r["median_s"] for r in before_rows)
    before_max = max(r["median_s"] for r in before_rows)
    spread = (before_max / before_min) if before_min > 0 else float("inf")
    verdict = "非後退" if ratio <= threshold else "後退"

    return {
        "status": "ok",
        "reason": None,
        "before_median_s": before_median,
        "after_median_s": after_median,
        "ratio": ratio,
        "before_min_s": before_min,
        "before_max_s": before_max,
        "after_min_s": min(r["median_s"] for r in after_rows),
        "after_max_s": max(r["median_s"] for r in after_rows),
        "checksum_composite_match": composite_ok,
        "checksum_exact_match": exact_ok,
        "before_spread": spread,
        "verdict": verdict,
    }


def _fmt_ms(s):
    if s is None:
        return "-"
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.1f} us"


def _all_expected_cells():
    """契約上の全 8 セル（`_VALID_SIZES` × `_VALID_MODES`）を昇順で返す。

    `render_markdown`／`main` の集計対象を `cells` に実在するキーだけに
    限定すると、before/after 双方から同一セルが欠落した場合に何も表示
    されず・`any_bad` 判定にも寄与しないまま終了コード 0 になりうる
    （codex-review P2・Cursor Bugbot Medium 指摘。イシュー #1306 の
    「全 8 セル非後退」契約に反する）。欠測セルを判定不能として明示する
    ため、実データに依らずこの固定集合を走査の基準にする。
    """
    return sorted(
        (size, mode) for size in _VALID_SIZES for mode in _VALID_MODES
    )


def render_markdown(cells, threshold):
    lines = []
    lines.append(
        "| size/mode | before median | after median | after/before | checksum | 判定 |"
    )
    lines.append("|---|---|---|---|---|---|")
    for key in _all_expected_cells():
        rows = cells.get(key, [])
        before_rows = [r for r in rows if not r.get("_is_after")]
        after_rows = [r for r in rows if r.get("_is_after")]
        if not rows:
            result = {
                "status": "undeterminable",
                "reason": "before/after 双方にこのセルの行がない（欠測セル）",
            }
        else:
            result = evaluate_cell(before_rows, after_rows, threshold)
        cell_label = "/".join(str(v) for v in key)
        if result["status"] != "ok":
            lines.append(
                f"| {cell_label} | - | - | - | - | 判定不能: {result['reason']} |"
            )
            continue
        checksum_label = (
            "完全一致"
            if result["checksum_exact_match"]
            else ("複合判定 ok" if result["checksum_composite_match"] else "不一致")
        )
        note = ""
        if result["before_spread"] > 1.5:
            note = "（判定注意: before spread > 1.5x・負荷ノイズの疑い）"
        lines.append(
            f"| {cell_label} | {_fmt_ms(result['before_median_s'])} "
            f"(min {_fmt_ms(result['before_min_s'])} / max {_fmt_ms(result['before_max_s'])}) | "
            f"{_fmt_ms(result['after_median_s'])} "
            f"(min {_fmt_ms(result['after_min_s'])} / max {_fmt_ms(result['after_max_s'])}) | "
            f"{result['ratio']:.4f} | {checksum_label} | {result['verdict']}{note} |"
        )
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Metal GEMM before/after (0.7.0 vs HEAD) A/B comparison (issue #1306)"
    )
    parser.add_argument("before", help="before (registry fandhe-ai =0.7.0) JSONL path")
    parser.add_argument("after", help="after (HEAD path-patched) JSONL path")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"non-regression ratio threshold (after/before; default {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args(argv[1:])

    before_rows, before_warnings = load_rows(args.before)
    after_rows, after_warnings = load_rows(args.after)
    warnings = before_warnings + after_warnings
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if warnings:
        print(
            "判定不能: 入力ファイルに不正行があり判定不能（詳細は上記 warning 行を参照）",
            file=sys.stderr,
        )
        return 2

    for r in before_rows:
        r["_is_after"] = False
    for r in after_rows:
        r["_is_after"] = True

    cells = {}
    for r in before_rows + after_rows:
        key = _cell_key(r)
        cells.setdefault(key, []).append(r)

    if not cells:
        print("判定不能: 入力ファイルに行がない", file=sys.stderr)
        return 2

    print(render_markdown(cells, args.threshold))

    any_bad = False
    for key in _all_expected_cells():
        rows = cells.get(key, [])
        if not rows:
            any_bad = True
            continue
        before_rows_k = [r for r in rows if not r.get("_is_after")]
        after_rows_k = [r for r in rows if r.get("_is_after")]
        result = evaluate_cell(before_rows_k, after_rows_k, args.threshold)
        if result["status"] != "ok" or result.get("verdict") != "非後退":
            any_bad = True
    return 3 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
