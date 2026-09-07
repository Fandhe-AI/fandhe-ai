#!/usr/bin/env python3
"""イシュー #1350: CUDA Graph step capture（`--graph`）3 状態の A/B 集計。

`run_ab_graph_cuda.sh` が出力する 1 本の JSONL には、`graph` フィールド
（`bench-common::Record.graph`。キー欠損 = off・`"stream-only"`・`"on"`）で
off／stream-only／on の 3 状態行が交互に混在する。本ツールはこれを
`(task, device, size, mode, phase)` セルごとに 3 状態へ分離し、5 回計測
中央値の比（stream-only/off・on/off・on/stream-only）・checksum 一致
（複合判定 + 完全一致の両方）・launch 固定費の診断カウンタ
（`graph_captured`/`graph_replayed`/`graph_launches`/
`graph_sgd_kernel_launches`）の 5 run 内一致を報告する。

`compare_managed_ab.py`（off/on の 2 状態版）の 3 状態拡張。設計判断・
事前宣言した採否判定基準は `docs/perf/train-step-phase-breakdown.md` §16
（イシュー #1350）を参照。

fail-closed 方針（security.md A08。`compare_managed_ab.py` と同方針）:
- `framework != "fandhe-ai"` の行、`tf32:true`／`managed:true`／
  `device_checksum:true` の行は判定不能行として除外する（本ツールは
  「同一実装・同一バイナリで graph フラグのみを変える」A/B 比較契約に
  限定されるため）
- 各セル off/stream-only/on のうち計測された状態はいずれも「ちょうど
  5 件」でなければ「判定不能」（一部の状態のみ計測されたセル——例えば
  train fresh は off/on のみで stream-only は計測しない、gemm/infer は
  そもそも `--graph` 対象外——では、存在しない状態を要求せず、存在する
  状態同士だけで比較する）
- `warmup`/`iters`/`version` が状態間で不一致なら「判定不能」
- checksum が複合判定（`checksum_contract.checksums_match`）を外れれば
  「判定不能」。加えて完全一致（`==`）列を別途表示する
- `on` 状態の launch カウンタ（`graph_captured` 等）が 5 run 内で不一致
  なら「判定不能」（capture／replay 回数が非決定的というシグナルのため、
  黙って中央値だけ報告しない）
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

_VALID_TASKS = frozenset(
    {"gemm", "gemm_phases", "train", "train_phases", "infer", "infer_phases"}
)
_VALID_DEVICES = frozenset({"cpu", "metal", "cuda"})
_VALID_MODES = frozenset({"fresh", "reuse"})
_VALID_GRAPH_STATES = frozenset({"off", "stream-only", "on"})
_GRAPH_COUNTER_FIELDS = (
    "graph_captured",
    "graph_replayed",
    "graph_launches",
    "graph_sgd_kernel_launches",
)


def _valid_cell_identity(obj):
    """`_cell_key` がグループ化キーへ使う `task`/`device`/`size`/`mode`/
    `phase` の型・値域を検証する（`compare_managed_ab.py::
    _valid_cell_identity` と同型・同じ fail-open 対策）。
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
    phase = obj.get("phase")
    if phase is not None and not isinstance(phase, str):
        return False
    return True


def _graph_state(obj):
    """行の `graph` フィールドから 3 状態のいずれかを返す。不正な値は
    `None` を返し呼び出し元がスキップする（`_VALID_GRAPH_STATES` の
    allowlist。security.md A03）。
    """
    g = obj.get("graph")
    if g is None:
        return "off"
    if not isinstance(g, str) or g not in _VALID_GRAPH_STATES or g == "off":
        # "off" という文字列そのものは `Record.graph` の契約上 emit
        # されない（`Cli::graph` の allowlist が "on"／"stream-only" のみ
        # を受理するため）。万一 JSONL に紛れ込んでいたら不正値として扱う。
        return None
    return g


def load_rows(path):
    """JSONL を読み、不正な行は理由付きで報告しスキップする（A08）。

    `compare_managed_ab.py::load_rows` と同じ検証方針を 3 状態版へ拡張
    する。`tf32`/`managed`/`device_checksum` はいずれも別軸のフラグの
    ため `true` の行を除外する（graph A/B との取り違え防止）。
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
            skip_true_bool_fields = ("tf32", "managed", "device_checksum")
            skipped = False
            for field in skip_true_bool_fields:
                if field in obj and not isinstance(obj[field], bool):
                    warnings.append(
                        f"{path}:{lineno}: 不正な '{field}' フィールド型（bool を期待。"
                        f"実際: {obj[field]!r}） — skipped"
                    )
                    skipped = True
                    break
                if obj.get(field, False) is True:
                    warnings.append(
                        f"{path}:{lineno}: '{field}:true' の行は graph A/B の対象外"
                        " — skipped"
                    )
                    skipped = True
                    break
            if skipped:
                continue
            if "graph" in obj and not isinstance(obj["graph"], str):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'graph' フィールド型（str を期待。"
                    f"実際: {obj['graph']!r}） — skipped"
                )
                continue
            state = _graph_state(obj)
            if state is None:
                warnings.append(
                    f"{path}:{lineno}: 不明な 'graph' 値（{obj.get('graph')!r}） — skipped"
                )
                continue
            if not _valid_cell_identity(obj):
                warnings.append(
                    f"{path}:{lineno}: 不正または欠損した 'task'/'device'/'size'/"
                    f"'mode' フィールド（行: {obj!r}） — skipped"
                )
                continue
            obj["_graph_state"] = state
            rows.append(obj)
    return rows, warnings


def _cell_key(r):
    return (
        r.get("task"),
        r.get("device"),
        r.get("size"),
        r.get("mode", "fresh"),
        r.get("phase"),
    )


def split_states(rows):
    """`(cell) -> {"off": [...], "stream-only": [...], "on": [...]}`。"""
    cells = {}
    for r in rows:
        key = _cell_key(r)
        bucket = cells.setdefault(key, {"off": [], "stream-only": [], "on": []})
        bucket[r["_graph_state"]].append(r)
    return cells


def _valid_field_value(field, v):
    """`compare_managed_ab.py::_valid_field_value` と同型。"""
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


def _check_common_fields(groups):
    """複数状態（各 5 件）にわたり `warmup`/`iters`/`version` が一致する
    ことを検証する。`groups` は `{state: [rows...]}`（存在する状態のみ）。
    """
    for field in ("warmup", "iters", "version"):
        all_values_by_state = {}
        for state, group_rows in groups.items():
            raw = [r.get(field) for r in group_rows]
            if None in raw:
                return f"'{field}' が {state} の行に欠損している"
            invalid = [v for v in raw if not _valid_field_value(field, v)]
            if invalid:
                return f"'{field}' に不正な値がある（{state}: {invalid!r}）"
            values = set(raw)
            if len(values) != 1:
                return f"'{field}' が {state} 内で不一致"
            all_values_by_state[state] = values
        distinct = set()
        for values in all_values_by_state.values():
            distinct |= values
        if len(distinct) != 1:
            return f"'{field}' が状態間で不一致（{all_values_by_state!r}）"
    return None


def _check_checksums(groups):
    """全状態を横断した checksum 突合。戻り値は
    `(composite_ok, exact_ok, reason_if_undeterminable)`。

    codex-review 指摘（PR #1425・P1）: 以前は `checksums[0]`（先頭 1 件）
    のみを基準にした「星型」比較（他の全要素を ref とだけ突合）だった。
    `checksum_contract.checksums_match` は許容誤差つきの比較のため
    厳密には推移的ではなく（A〜ref・B〜ref が成立しても A〜B が成立する
    保証はない）、星型比較では non-off 状態同士（例: stream-only と on）
    の不一致を見逃しうる。3 状態（off/stream-only/on）は最大でも 15 件
    （5 run × 3 状態）程度のため、計算コストを気にせず全ペア突合へ
    変更し、off/stream-only/on の全ペアで一致していることを保証する。
    """
    all_rows = [r for rows in groups.values() for r in rows]
    checksums = [r.get("checksum") for r in all_rows]
    if any(c is None or isinstance(c, bool) or not isinstance(c, (int, float)) for c in checksums):
        return False, False, "checksum 欠損または非数値の行あり"
    composite_ok = all(
        checksum_contract.checksums_match(a, b)
        for i, a in enumerate(checksums)
        for b in checksums[i + 1 :]
    )
    exact_ok = all(a == b for i, a in enumerate(checksums) for b in checksums[i + 1 :])
    if not composite_ok:
        return False, False, f"checksum が複合判定を外れる（{checksums!r}）"
    return composite_ok, exact_ok, None


def _check_graph_counters(on_rows):
    """`on` 状態の launch カウンタが 5 run 内で一致するかを検証する。
    `on_rows` が空（このセルで on が計測されていない）場合は None を返す
    （検査対象なし）。フィールド欠損は「実測できていない」ことを示す
    fail-closed な判定不能扱いとする。
    """
    if not on_rows:
        return None
    counters = {}
    for field in _GRAPH_COUNTER_FIELDS:
        raw = [r.get(field) for r in on_rows]
        if any(v is None or isinstance(v, bool) or not isinstance(v, int) for v in raw):
            return {"ok": False, "reason": f"'{field}' が欠損または非整数（{raw!r}）"}
        values = set(raw)
        if len(values) != 1:
            return {"ok": False, "reason": f"'{field}' が on の 5 run 内で不一致（{raw!r}）"}
        counters[field] = raw[0]
    return {"ok": True, "reason": None, **counters}


def evaluate_cell(groups):
    """1 セル分の `{state: rows}`（存在する状態のみ・各ちょうど 5 件を
    要求）を突合し判定結果 dict を返す。
    """
    present = {s: rows for s, rows in groups.items() if rows}
    if len(present) < 2:
        return {
            "status": "undeterminable",
            "reason": f"比較対象の状態が 2 未満（存在する状態: {sorted(present.keys())!r}）",
        }
    for state, rows in present.items():
        if len(rows) != 5:
            return {
                "status": "undeterminable",
                "reason": f"{state} が {len(rows)} 件（ちょうど 5 件を要求）",
            }
    field_reason = _check_common_fields(present)
    if field_reason is not None:
        return {"status": "undeterminable", "reason": field_reason}

    for state, rows in present.items():
        for r in rows:
            v = r.get("median_s")
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                return {
                    "status": "undeterminable",
                    "reason": f"{state} 側 median_s が数値ではない（{v!r}）",
                }
            if not math.isfinite(v) or v <= 0:
                return {
                    "status": "undeterminable",
                    "reason": f"{state} 側 median_s が有限正数ではない（{v!r}）",
                }

    medians = {s: _median([r["median_s"] for r in rows]) for s, rows in present.items()}
    mins = {s: min(r["median_s"] for r in rows) for s, rows in present.items()}
    maxs = {s: max(r["median_s"] for r in rows) for s, rows in present.items()}

    composite_ok, exact_ok, checksum_reason = _check_checksums(present)
    if checksum_reason is not None:
        return {"status": "undeterminable", "reason": checksum_reason}

    graph_counters = _check_graph_counters(present.get("on", []))
    if graph_counters is not None and not graph_counters["ok"]:
        return {"status": "undeterminable", "reason": graph_counters["reason"]}

    ratios = {}
    if "off" in medians and "stream-only" in medians:
        ratios["stream_only_over_off"] = medians["stream-only"] / medians["off"]
    if "off" in medians and "on" in medians:
        ratios["on_over_off"] = medians["on"] / medians["off"]
    if "stream-only" in medians and "on" in medians:
        ratios["on_over_stream_only"] = medians["on"] / medians["stream-only"]

    return {
        "status": "ok",
        "reason": None,
        "medians": medians,
        "mins": mins,
        "maxs": maxs,
        "ratios": ratios,
        "checksum_composite_match": composite_ok,
        "checksum_exact_match": exact_ok,
        "graph_counters": graph_counters,
    }


def _fmt_ms(s):
    if s is None:
        return "-"
    if s >= 1.0:
        return f"{s:.3f} s"
    if s >= 1e-3:
        return f"{s * 1e3:.3f} ms"
    return f"{s * 1e6:.1f} us"


def render_markdown(cells):
    lines = []
    # 「判定」列は出さない: 事前宣言した採否基準（`docs/perf/train-step-
    # phase-breakdown.md` §16）は `step_total`（本ツールの train セル）と
    # `device_update`（train_phases セルの当該 phase）を横断した基準で、
    # 単一セルの比だけでは決められないため、ここでは各セルの実測値
    # （中央値・比・checksum・launch カウンタ）のみを機械的に報告し、
    # 採否の結論は docs 側で人間が書く。
    lines.append(
        "| cell (task/device/size/mode/phase) | off median | stream-only median | "
        "on median | stream-only/off | on/off | on/stream-only | checksum | "
        "on launch counters (captured/replayed/graph_launches/sgd_launches) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for key in sorted(cells.keys(), key=lambda k: tuple(str(v) for v in k)):
        groups = cells[key]
        result = evaluate_cell(groups)
        cell_label = "/".join(str(v) for v in key)
        if result["status"] != "ok":
            lines.append(
                f"| {cell_label} | - | - | - | - | - | - | - | 判定不能: {result['reason']} |"
            )
            continue
        medians = result["medians"]
        checksum_label = (
            "完全一致"
            if result["checksum_exact_match"]
            else ("複合判定 ok" if result["checksum_composite_match"] else "不一致")
        )
        counters = result["graph_counters"]
        counters_label = (
            "/".join(str(counters[f]) for f in _GRAPH_COUNTER_FIELDS)
            if counters is not None
            else "-"
        )
        ratios = result["ratios"]

        def _ratio_label(k):
            return f"{ratios[k]:.4f}" if k in ratios else "-"

        lines.append(
            f"| {cell_label} "
            f"| {_fmt_ms(medians.get('off'))} "
            f"| {_fmt_ms(medians.get('stream-only'))} "
            f"| {_fmt_ms(medians.get('on'))} "
            f"| {_ratio_label('stream_only_over_off')} "
            f"| {_ratio_label('on_over_off')} "
            f"| {_ratio_label('on_over_stream_only')} "
            f"| {checksum_label} "
            f"| {counters_label} |"
        )
    return "\n".join(lines)


def main(argv):
    if len(argv) != 2:
        print(f"usage: {argv[0]} <results.jsonl>", file=sys.stderr)
        return 2
    path = argv[1]
    rows, warnings = load_rows(path)
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    if warnings:
        print(
            "判定不能: 入力ファイルに不正行があり判定不能（詳細は上記 warning 行を参照）",
            file=sys.stderr,
        )
        return 3
    cells = split_states(rows)
    if not cells:
        print("判定不能: 入力ファイルに行がない", file=sys.stderr)
        return 3
    print(render_markdown(cells))
    any_undeterminable = any(
        evaluate_cell(v)["status"] != "ok" for v in cells.values()
    )
    return 3 if any_undeterminable else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
