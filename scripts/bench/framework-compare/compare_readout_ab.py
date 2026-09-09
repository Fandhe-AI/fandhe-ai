#!/usr/bin/env python3
"""イシュー #1477: Metal 借用ビュー readout（legacy フォールバック）の
legacy/borrowed override（`--readout`）interleave 再計測の A/B 集計。

`run_ab_readout_metal.sh` が出力する 1 本の JSONL には、`readout` フィールド
（`bench-common::Record.readout`。`"legacy"`／`"borrowed"`）で legacy/borrowed
行が run 単位（奇数 run: legacy→borrowed・偶数 run: borrowed→legacy）に交互
配置で混在する。本ツールはこれを `(task, device, size, mode)` セルごとに
legacy/borrowed へ分離し、5 回計測中央値の比・checksum 一致（複合判定 +
完全一致の両方）を報告する。`compare_managed_ab.py`（同一バイナリ・
フラグ切替専用の A/B）と同型の設計で、判定対象を `readout` フィールドへ
差し替えたもの。

対象は device=metal・N ∈ {1024, 2048, 4096}（`docs/perf/
metal-gemm-candle-gate-remeasurement.md` §11/§14 の #1037 ゲート判定形状）・
mode ∈ {fresh, reuse} の 6 セルに限定する（事前宣言した判定規則。
イシュー #1477 計画 §2）。

fail-closed 方針（security.md A08。`compare_gemm_gate.py`/`compare_managed_
ab.py` と同方針）:
- `framework != "fandhe-ai"` の行、`device != "metal"` の行、`tf32:true`／
  `managed:true`／`device_checksum:true`／`graph` キーを持つ行は判定不能行
  として除外する（本ツールは「同一実装・同一バイナリで readout フラグの
  みを変える」A/B 比較契約に限定されるため）
- `readout` キーを持たない行（override 計測ではない通常計測行）は除外する
- 各セル legacy/borrowed とも「ちょうど 5 件」でなければ「判定不能」
- `warmup`/`iters`/`version` が legacy/borrowed で不一致なら「判定不能」
- checksum が複合判定（`checksum_contract.checksums_match`）を外れれば
  「判定不能」。加えて完全一致（`==`）列を別途表示する

ADOPT／REJECT／undetermined の判定規則（事前宣言。閾値の既定は 1.00 —
`compare_gemm_ab.py` 既定の 1.05 ではなく、`docs/perf/metal-gemm-candle-
gate-remeasurement.md` §13.5 が「#1438 判定木 (a) 全 N で after/before <= 1.00」
の再計測と位置づけていることを継承する）:
- 全 6 セルで `ratio <= threshold` かつ checksum 完全一致・parity 0 fail
  （`parity_fail_count` が全行で `0`。欠損は fail-closed に fail 扱い）
  → ADOPT（`readout_uses_borrowed_view` の Metal 分岐を除去し借用ビュー
  を既定化する根拠）
- 1 セルでも `ratio > threshold` または checksum 不一致・parity fail
  → REJECT（legacy フォールバックを維持する）
- 必須 6 セル（`(gemm, metal, size, mode, phase=None)`。[`required_cells`]）
  のうち 1 件でも計測行が存在しない、またはセル自体が「判定不能」
  （件数過不足・warmup/iters/version 不一致・checksum 複合判定外れ 等）
  → undetermined としてまとめて記録する（codex-review 指摘・PR #1493
  P1: 存在するセルのみで ADOPT を確定しない）
"""

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

# `_cell_key` がグループ化に使う識別フィールドの許容値（`compare_managed_
# ab.py` と同型）。
_VALID_TASKS = frozenset(
    {"gemm", "gemm_phases", "train", "train_phases", "infer", "infer_phases"}
)
_VALID_DEVICES = frozenset({"metal"})
_VALID_MODES = frozenset({"fresh", "reuse"})
# イシュー #1477 判定規則 §2 の対象形状（gate 3 サイズ）。
GATE_SIZES = frozenset({1024, 2048, 4096})


def _valid_cell_identity(obj, size_set=None):
    """`_cell_key` がグループ化キーへ使う `task`/`device`/`size`/`mode`/
    `phase` の型・値域を検証する（`compare_managed_ab.py::_valid_cell_
    identity` と同方針）。
    """
    task = obj.get("task")
    if not isinstance(task, str) or task not in _VALID_TASKS:
        return False
    device = obj.get("device")
    if not isinstance(device, str) or device not in _VALID_DEVICES:
        return False
    mode = obj.get("mode", "fresh")
    if not isinstance(mode, str) or mode not in _VALID_MODES:
        return False
    size = obj.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        return False
    if size_set is not None and size not in size_set:
        return False
    phase = obj.get("phase")
    if phase is not None and not isinstance(phase, str):
        return False
    return True


def load_rows(path, size_set=None):
    """JSONL を読み、不正な行は理由付きで報告しスキップする（A08）。

    `readout` フィールドは本ツールが分離キーとして使う必須フィールド
    （`"legacy"`／`"borrowed"` のいずれか）。それ以外（キー欠損・不正な
    値）の行は除外する。`framework`/`tf32`/`managed`/`device_checksum`/
    `graph` の検証は `compare_managed_ab.py` と同方針。
    """
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
                    f"{path}:{lineno}: 'tf32:true' の行は readout A/B の"
                    "対象外 — skipped"
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
                    f"{path}:{lineno}: 'managed:true' の行は readout A/B の"
                    "対象外 — skipped"
                )
                continue
            if "device_checksum" in obj and not isinstance(
                obj["device_checksum"], bool
            ):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'device_checksum' フィールド型（bool を"
                    f"期待。実際: {obj['device_checksum']!r}） — skipped"
                )
                continue
            if obj.get("device_checksum", False) is True:
                warnings.append(
                    f"{path}:{lineno}: 'device_checksum:true' の行は readout A/B の"
                    "対象外 — skipped"
                )
                continue
            if "graph" in obj and not isinstance(obj["graph"], str):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'graph' フィールド型（str を期待。"
                    f"実際: {obj['graph']!r}） — skipped"
                )
                continue
            if "graph" in obj:
                warnings.append(
                    f"{path}:{lineno}: 'graph' キーを持つ行は readout A/B の"
                    "対象外 — skipped"
                )
                continue
            readout = obj.get("readout")
            if readout not in ("legacy", "borrowed"):
                warnings.append(
                    f"{path}:{lineno}: 不正または欠損した 'readout' フィールド"
                    f"（'legacy'／'borrowed' を期待。実際: {readout!r}） — skipped"
                )
                continue
            if not _valid_cell_identity(obj, size_set=size_set):
                warnings.append(
                    f"{path}:{lineno}: 不正または欠損した 'task'/'device'/'size'/"
                    f"'mode' フィールド（行: {obj!r}） — skipped"
                )
                continue
            rows.append(obj)
    return rows, warnings


def _cell_key(r):
    """`(task, device, size, mode, phase)`。"""
    return (
        r.get("task"),
        r.get("device"),
        r.get("size"),
        r.get("mode", "fresh"),
        r.get("phase"),
    )


def split_legacy_borrowed(rows):
    """`(cell) -> {"legacy": [...], "borrowed": [...]}` へ分離する。"""
    cells = {}
    for r in rows:
        key = _cell_key(r)
        bucket = r.get("readout")
        cells.setdefault(key, {"legacy": [], "borrowed": []})[bucket].append(r)
    return cells


def required_cells(device, size_set):
    """イシュー #1477 計画 §2・`docs/perf/metal-gemm-candle-gate-
    remeasurement.md` §15.1 が要求する必須セル集合を返す（`task="gemm"`・
    `device`・`size_set` の各サイズ・`mode ∈ {"fresh", "reuse"}`・
    `phase=None` の直積。既定 `--sizes gate` では 3 サイズ × 2 モード =
    6 セル）。`cells` にこの集合の全キーが揃っていない限り、存在する
    セルのみで ADOPT／REJECT を確定してはならない（codex-review 指摘・
    PR #1493 P1: 欠損セルがあっても非後退の見かけになりうるため）。
    """
    return frozenset(
        ("gemm", device, size, mode, None)
        for size in size_set
        for mode in ("fresh", "reuse")
    )


def missing_required_cells(cells, device, size_set):
    """`cells` に必須セル（[`required_cells`]）のうち計測行が 1 件も
    存在しないものを、表示順（size 昇順・mode 順）で返す。
    """
    required = required_cells(device, size_set)
    present = set(cells.keys())
    missing = required - present
    return sorted(missing, key=lambda k: tuple(str(v) for v in k))


def _valid_field_value(field, v):
    """`warmup`／`iters`／`version` 1 値の型・値域を検証する
    （`compare_managed_ab.py::_valid_field_value` と同方針）。
    """
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


def evaluate_cell(legacy_rows, borrowed_rows):
    """1 セル分の legacy/borrowed 行を突合し判定結果 dict を返す。

    戻り値のキー: `status`（"ok"|"undeterminable"）・`reason`
    （undeterminable のときのみ非 None）・`legacy_median_s`/
    `borrowed_median_s`/`ratio`（= borrowed_median_s / legacy_median_s。
    値 <= 1.0 は borrowed が速い／同等であることを示す）・
    `legacy_min_s`/`legacy_max_s`/`borrowed_min_s`/`borrowed_max_s`・
    `checksum_composite_match`/`checksum_exact_match`。
    """
    if len(legacy_rows) != 5 or len(borrowed_rows) != 5:
        return {
            "status": "undeterminable",
            "reason": (
                f"legacy={len(legacy_rows)} 件・borrowed={len(borrowed_rows)} 件"
                "（各ちょうど 5 件を要求）"
            ),
        }
    for field in ("warmup", "iters", "version"):
        legacy_raw = [r.get(field) for r in legacy_rows]
        borrowed_raw = [r.get(field) for r in borrowed_rows]
        if None in legacy_raw or None in borrowed_raw:
            return {
                "status": "undeterminable",
                "reason": f"'{field}' が legacy または borrowed の行に欠損している",
            }
        invalid_values = [
            v for v in legacy_raw + borrowed_raw if not _valid_field_value(field, v)
        ]
        if invalid_values:
            return {
                "status": "undeterminable",
                "reason": f"'{field}' に不正な値がある（{invalid_values!r}）",
            }
        legacy_values = set(legacy_raw)
        borrowed_values = set(borrowed_raw)
        if len(legacy_values) != 1 or len(borrowed_values) != 1:
            return {
                "status": "undeterminable",
                "reason": f"'{field}' が legacy または borrowed 内で不一致",
            }
        if legacy_values != borrowed_values:
            return {
                "status": "undeterminable",
                "reason": (
                    f"'{field}' が legacy/borrowed 間で不一致"
                    f"（legacy={legacy_values!r} borrowed={borrowed_values!r}）"
                ),
            }

    for label, group_rows in (("legacy", legacy_rows), ("borrowed", borrowed_rows)):
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

    legacy_median = _median([r["median_s"] for r in legacy_rows])
    borrowed_median = _median([r["median_s"] for r in borrowed_rows])
    if legacy_median <= 0:
        return {"status": "undeterminable", "reason": "legacy_median_s <= 0"}

    legacy_checksums = [r.get("checksum") for r in legacy_rows]
    borrowed_checksums = [r.get("checksum") for r in borrowed_rows]
    if any(
        c is None or isinstance(c, bool) or not isinstance(c, (int, float))
        for c in legacy_checksums + borrowed_checksums
    ):
        return {
            "status": "undeterminable",
            "reason": "checksum 欠損または非数値の行あり",
        }
    ref_checksum = legacy_checksums[0]
    composite_ok = all(
        checksum_contract.checksums_match(ref_checksum, c)
        for c in legacy_checksums + borrowed_checksums
    )
    exact_ok = all(c == ref_checksum for c in legacy_checksums + borrowed_checksums)
    if not composite_ok:
        return {
            "status": "undeterminable",
            "reason": (
                f"checksum が複合判定を外れる（legacy={legacy_checksums!r} "
                f"borrowed={borrowed_checksums!r}）"
            ),
        }

    return {
        "status": "ok",
        "reason": None,
        "legacy_median_s": legacy_median,
        "borrowed_median_s": borrowed_median,
        "ratio": borrowed_median / legacy_median,
        "legacy_min_s": min(r["median_s"] for r in legacy_rows),
        "legacy_max_s": max(r["median_s"] for r in legacy_rows),
        "borrowed_min_s": min(r["median_s"] for r in borrowed_rows),
        "borrowed_max_s": max(r["median_s"] for r in borrowed_rows),
        "checksum_composite_match": composite_ok,
        "checksum_exact_match": exact_ok,
    }


def _fmt_ms(s):
    if s is None:
        return "-"
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.1f} us"


def render_markdown(cells, threshold, device, size_set):
    """セルごとの判定結果を Markdown 表として整形し、総合 verdict 行を
    末尾に付す（イシュー #1477 判定規則 §2）。必須セル（[`required_
    cells`]）のうち計測行が 1 件も無いものは「欠損（計測なし）」行として
    明示し、存在するセルのみで判定が確定しないようにする（codex-review
    指摘・PR #1493 P1）。
    """
    lines = []
    lines.append(
        "| cell (task/device/size/mode/phase) | legacy median | borrowed median | borrowed/legacy | checksum | 判定 |"
    )
    lines.append("|---|---|---|---|---|---|")
    results = {}
    for key in sorted(cells.keys(), key=lambda k: tuple(str(v) for v in k)):
        legacy_rows = cells[key]["legacy"]
        borrowed_rows = cells[key]["borrowed"]
        result = evaluate_cell(legacy_rows, borrowed_rows)
        results[key] = result
        cell_label = "/".join(str(v) for v in key)
        if result["status"] != "ok":
            lines.append(f"| {cell_label} | - | - | - | - | 判定不能: {result['reason']} |")
            continue
        checksum_label = (
            "完全一致"
            if result["checksum_exact_match"]
            else ("複合判定 ok" if result["checksum_composite_match"] else "不一致")
        )
        verdict = (
            "非後退"
            if result["ratio"] <= threshold and result["checksum_exact_match"]
            else "後退"
        )
        lines.append(
            f"| {cell_label} | {_fmt_ms(result['legacy_median_s'])} "
            f"(min {_fmt_ms(result['legacy_min_s'])} / max {_fmt_ms(result['legacy_max_s'])}) | "
            f"{_fmt_ms(result['borrowed_median_s'])} "
            f"(min {_fmt_ms(result['borrowed_min_s'])} / max {_fmt_ms(result['borrowed_max_s'])}) | "
            f"{result['ratio']:.4f} | {checksum_label} | {verdict} |"
        )

    missing = missing_required_cells(cells, device, size_set)
    for key in missing:
        cell_label = "/".join(str(v) for v in key)
        lines.append(f"| {cell_label} | - | - | - | - | 欠損（計測なし） |")

    lines.append("")
    lines.append(f"総合判定: {overall_verdict(cells, threshold, device, size_set)}")
    return "\n".join(lines)


def overall_verdict(cells, threshold, device, size_set):
    """イシュー #1477 判定規則 §2 の 3 択（ADOPT／REJECT／undetermined）を
    機械的に確定する。

    - 必須セル（[`required_cells`]。`gemm`/`device`/`size_set` 各サイズ/
      `{fresh, reuse}`/`phase=None` の直積）が 1 件でも欠損 → undetermined
      （存在するセルのみを見て非後退と誤判定しない。codex-review 指摘・
      PR #1493 P1）
    - いずれかの必須セルが「判定不能」→ undetermined
    - 全必須セル ok かつ全セルで `ratio <= threshold` かつ checksum 完全
      一致・parity 0 fail（`--parity-fail-count-key` 経由。P1: parity
      未検証は ADOPT 対象外） → ADOPT
    - それ以外（1 セルでも `ratio > threshold` または checksum 不一致・
      parity fail） → REJECT
    """
    if missing_required_cells(cells, device, size_set):
        return "undetermined（必須セル欠損あり。legacy フォールバックを維持する）"
    required = required_cells(device, size_set)
    results = [
        evaluate_cell(cells[key]["legacy"], cells[key]["borrowed"]) for key in required
    ]
    if any(r["status"] != "ok" for r in results):
        return "undetermined（判定不能セルあり。legacy フォールバックを維持する）"
    if any(_cell_has_parity_fail(cells[key]) for key in required):
        return "REJECT（parity fail の行があるため legacy フォールバックを維持する）"
    if all(
        r["ratio"] <= threshold and r["checksum_exact_match"] for r in results
    ):
        return "ADOPT（借用ビューへ切替。Metal 分岐を除去して既定化する）"
    return "REJECT（legacy フォールバックを維持する）"


def _cell_has_parity_fail(cell):
    """1 セル（legacy/borrowed 全行）のうち、`parity_fail_count`（`Record::
    to_json_line` が emit するフラットキー。`bench-common::Record.parity`
    の `fail_count`）が 0 以外の整数または欠損している行が 1 件でもあれば
    `True` を返す（codex-review 指摘・PR #1493 P1: `evaluate_cell` が
    parity を一切見ないため、checksum と性能さえ一致すれば parity fail
    が残っていても・parity が未検証〈`--device-checksum` 等で `parity`
    キー自体が省略される場合〉でも ADOPT になりうる問題の是正。REQ-2
    coding-rust.md「テスト・ベンチ」節の parity 0 fail 契約に倣い、欠損
    は fail-closed に「fail あり」として扱う。`fail_count` は非負のはずの
    件数フィールドであり負整数は不正値だが、`> 0` 判定のみでは負整数
    （例 `-1`）を素通りさせ「全行で parity 0 fail」契約を破る〈codex-review
    指摘・PR #1493 スレッド 2〉ため `!= 0` で判定し負整数も fail 扱いに
    する）。
    """
    for label in ("legacy", "borrowed"):
        for r in cell.get(label, []):
            fail_count = r.get("parity_fail_count")
            if (
                not isinstance(fail_count, int)
                or isinstance(fail_count, bool)
                or fail_count != 0
            ):
                return True
    return False


def main(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "イシュー #1477: Metal 借用ビュー readout の legacy/borrowed "
            "override interleave 計測を A/B 集計する"
        )
    )
    parser.add_argument("results_jsonl")
    parser.add_argument(
        "--device",
        default="metal",
        choices=sorted(_VALID_DEVICES),
        help="対象デバイス（既定 metal。本ツールは Metal 限定）",
    )
    parser.add_argument(
        "--sizes",
        default="gate",
        help=(
            "'gate'（既定。GATE_SIZES={1024,2048,4096}）または"
            " カンマ区切りの整数リスト"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.00,
        help=(
            "ADOPT 判定の ratio 上限（既定 1.00。判定規則の出典は"
            " docs/perf/metal-gemm-candle-gate-remeasurement.md §13.5）"
        ),
    )
    args = parser.parse_args(argv[1:])

    if args.sizes == "gate":
        size_set = GATE_SIZES
    else:
        try:
            size_set = frozenset(int(x) for x in args.sizes.split(","))
        except ValueError:
            print(f"不正な --sizes 値: {args.sizes!r}", file=sys.stderr)
            return 2

    rows, warnings = load_rows(args.results_jsonl, size_set=size_set)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if warnings:
        print(
            "判定不能: 入力ファイルに不正行があり判定不能（詳細は上記 warning 行を参照）",
            file=sys.stderr,
        )
        return 3
    rows = [r for r in rows if r.get("device") == args.device]
    cells = split_legacy_borrowed(rows)
    if not cells:
        print("判定不能: 入力ファイルに対象行がない", file=sys.stderr)
        return 3
    print(render_markdown(cells, args.threshold, args.device, size_set))
    # イシュー #1477 判定規則 §2: 必須セル（[`required_cells`]）が 1 件
    # でも欠損していれば「存在するセルのみで非後退」と誤判定してはいけない
    # （codex-review 指摘・PR #1493 P1）。missing 自体を判定不能の理由に
    # 含めたうえで、存在する必須セルの判定不能も従来どおり検出する。
    any_undeterminable = bool(
        missing_required_cells(cells, args.device, size_set)
    ) or any(
        evaluate_cell(cells[key]["legacy"], cells[key]["borrowed"])["status"] != "ok"
        for key in required_cells(args.device, size_set)
        if key in cells
    )
    return 3 if any_undeterminable else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
