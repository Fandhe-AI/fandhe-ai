#!/usr/bin/env python3
"""イシュー #1353: CUDA managed memory 配置（`--managed`）有無の A/B 集計。

`run_ab_managed_cuda.sh` が出力する 1 本の JSONL には、`managed` フィールド
（`bench-common::Record.managed`。キー欠損／`false` = off・`true` = on）で
off/on 行が交互に混在する。本ツールはこれを `(task, device, size, mode)`
セルごとに off/on へ分離し、5 回計測中央値の比・checksum 一致（複合判定 +
完全一致の両方）を報告する。

`compare_ab.py` は `framework_version` が before/after で同一であることを
fail-closed で拒否する（同一バージョンの A/B は before/after 比較として
意味を持たないという前提）ため、同一バイナリ・同一 `fandhe-ai` バージョンで
フラグのみを変える本用途には流用できない（`.claude/rules/deps-policy.md`
「同一バイナリ・off/on を run 単位で交互起動」の設計）。

fail-closed 方針（security.md A08。`compare_gemm_gate.py` と同方針）:
- 各セル off/on とも「ちょうど 5 件」でなければ「判定不能」
- `warmup`/`iters`/`version` が off/on で不一致なら「判定不能」
- checksum が複合判定（`checksum_contract.checksums_match`）を外れれば
  「判定不能」。加えて完全一致（`==`）列を別途表示する（#1352 の核心契約
  「配置に依らず bit 同一」の裏取りのため、複合判定 pass だけでは契約破れ
  を隠してしまう）
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

# `_cell_key` がグループ化に使う識別フィールドの許容値（イシュー #1353・
# github-actions レビュー指摘）。`run_ab_managed_cuda.sh` が起動する
# `bench-fandhe` の task（`--phases` 指定時は `<task>_phases`）・device
# （`summarize.py::DEVICE_ORDER` と同じ許容集合。本ツールは CUDA managed
# 配置専用のため `cuda` のみだが、将来の拡張余地を残し他デバイスも許容する）・
# mode の値をここで固定する。
_VALID_TASKS = frozenset(
    {"gemm", "gemm_phases", "train", "train_phases", "infer", "infer_phases"}
)
_VALID_DEVICES = frozenset({"cpu", "metal", "cuda"})
_VALID_MODES = frozenset({"fresh", "reuse"})


def _valid_cell_identity(obj):
    """`_cell_key` がグループ化キーへ使う `task`/`device`/`size`/`mode`/
    `phase` の型・値域を検証する（イシュー #1353・github-actions レビュー
    指摘）。

    これらを検証せずに読み込むと、正常な off/on 各 5 件からこれらの
    フィールドを削除した行でも `_cell_key` が `(None, None, None,
    "fresh", None)` のような単一セルへ迂回して集約され、比較対象が
    実際には不明であるにもかかわらず判定 "ok" となりうる（fail-open の
    おそれ。`managed` フィールド自体の型検証だけでは防げない）。

    `task`/`device`/`mode` は `frozenset` への `in`/`not in` 判定の前に
    必ず `isinstance(..., str)` で型を確認する（codex-review 指摘・
    イシュー #1353 3 巡目）。JSON の配列・オブジェクト（Python では
    list/dict）は非 hashable なため、型確認なしに `not in _VALID_TASKS`
    等の集合メンバーシップ判定へ直接渡すと、`False` を返す前に
    `TypeError` を送出してクラッシュする（`load_rows` は「不正な行は
    理由付きでスキップする」契約〈A08〉のため、判定不能な行が
    プロセス全体を落とすのは fail-open ではなく可用性の欠陥だが、
    いずれにせよここで通常の「不正行スキップ」経路へ倒す必要がある）。
    `phase` も `_cell_key` のグループ化キーに含まれるため同様に検証する
    （list/dict のまま `cells.setdefault(key, ...)` の辞書キーに使われると
    `split_off_on` 側で同種の `TypeError` を招く）。
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


def load_rows(path):
    """JSONL を読み、不正な行は理由付きで報告しスキップする（A08）。

    `managed` フィールドの型検証は summarize.py/compare_gemm_gate.py と同じ
    fail-closed 方針（bool 以外はスキップ）。`task`/`device`/`size`/`mode`
    （`_cell_key` が使う識別フィールド）も `_valid_cell_identity` で検証する
    （イシュー #1353・github-actions レビュー指摘。詳細は同関数 docstring）。
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
            if "managed" in obj and not isinstance(obj["managed"], bool):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'managed' フィールド型（bool を期待。"
                    f"実際: {obj['managed']!r}） — skipped"
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
    """`(task, device, size, mode)`。`phase`（`train_phases`/`gemm_phases`
    等の task 名で運ばれる）がある行は phase 別にも区別できるよう含める。
    """
    return (
        r.get("task"),
        r.get("device"),
        r.get("size"),
        r.get("mode", "fresh"),
        r.get("phase"),
    )


def split_off_on(rows):
    """`(cell) -> {"off": [...], "on": [...]}` へ分離する。"""
    cells = {}
    for r in rows:
        key = _cell_key(r)
        bucket = "on" if r.get("managed", False) is True else "off"
        cells.setdefault(key, {"off": [], "on": []})[bucket].append(r)
    return cells


def _valid_field_value(field, v):
    """`warmup`／`iters`／`version` 1 値の型・値域を検証する（イシュー
    #1353・github-actions レビュー指摘）。

    `bool` は Python では `int` のサブクラス（`True == 1`）のため、
    `isinstance(v, int)` だけでは `warmup=True` のような型混入を弾けない。
    明示的に `bool` を除外したうえで、`warmup` は 0 以上・`iters` は
    1 以上（0 回計測は無意味）の整数、`version` は非空文字列であることを
    要求する（fail-closed。`compare_gemm_gate.py` と同様の方針）。
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


def evaluate_cell(off_rows, on_rows):
    """1 セル分の off/on 行を突合し判定結果 dict を返す。

    戻り値のキー: `status`（"ok"|"undeterminable"）・`reason`（undeterminable
    のときのみ非 None）・`off_median_s`/`on_median_s`/`ratio`
    （= on_median_s / off_median_s。値 < 1.0 は on が速いことを示す）・
    `off_min_s`/`off_max_s`/`on_min_s`/`on_max_s`・
    `checksum_composite_match`/`checksum_exact_match`。
    """
    if len(off_rows) != 5 or len(on_rows) != 5:
        return {
            "status": "undeterminable",
            "reason": (
                f"off={len(off_rows)} 件・on={len(on_rows)} 件"
                "（各ちょうど 5 件を要求）"
            ),
        }
    for field in ("warmup", "iters", "version"):
        # イシュー #1353（github-actions レビュー指摘・2 巡目）: `bool` は
        # Python では `int` のサブクラスで `True == 1`／`hash(True) ==
        # hash(1)` が成立するため、型検証より先に `set` 化すると
        # `iters=[1, True, 1, 1, 1]` のような入力で `True` が同値の `1` に
        # 吸収されて集合から消え、後段の `_valid_field_value` が `bool`
        # 混入を検出できずに素通りしてしまう（fail-open のおそれ）。
        # 集合化する前に各行の生値（リストのまま。重複除去しない）を
        # 個別に検証する。
        off_raw = [r.get(field) for r in off_rows]
        on_raw = [r.get(field) for r in on_rows]
        # 全行欠損（`r.get(field)` が None）でも生値リストの一致判定は
        # 通ってしまう（codex-review 指摘）。欠損・不正値を明示的に
        # 判定不能として拒否する。
        if None in off_raw or None in on_raw:
            return {
                "status": "undeterminable",
                "reason": f"'{field}' が off または on の行に欠損している",
            }
        # イシュー #1353（github-actions レビュー指摘）: 型・値域を検証する。
        # 欠損チェック（上）と一致チェック（下）だけでは
        # `warmup=-1`／`iters=0`／`version=""` のような不正値でも off/on
        # 双方で揃ってさえいれば「一致」として素通りしてしまう（fail-open
        # のおそれ）。生値リストの各要素に対して明示的に妥当性を検証する。
        #
        # `invalid_values` は `set` ではなく `list` として構築する
        # （codex-review 指摘・イシュー #1353 3 巡目）: 外部 JSONL の
        # `warmup`/`iters`/`version` は任意の JSON 値になりうり、配列・
        # オブジェクト（Python の list/dict）は非 hashable。`_valid_field_value`
        # がそれらを「不正」と正しく判定しても、判定結果を `set` 内包表記へ
        # 集めようとした時点で当の値自体をハッシュ化しようとして
        # `TypeError` を送出し、「判定不能として拒否する」という設計契約
        # 通りに動かずクラッシュしてしまう。`list` は非 hashable な要素も
        # 保持できるため、ここでは順序付きリストのまま収集する。
        invalid_values = [
            v for v in off_raw + on_raw if not _valid_field_value(field, v)
        ]
        if invalid_values:
            return {
                "status": "undeterminable",
                "reason": f"'{field}' に不正な値がある（{invalid_values!r}）",
            }
        # 型・値域検証を通過した生値のみをここで集合化する（`bool`／`int`
        # の同値吸収問題は上記の生値検証で既に弾いているため、ここでの
        # 集合化は安全。加えて `_valid_field_value` が通過させる値は
        # int/str のみで list/dict は上で弾かれているため hashable）。
        off_values = set(off_raw)
        on_values = set(on_raw)
        if len(off_values) != 1 or len(on_values) != 1:
            return {
                "status": "undeterminable",
                "reason": f"'{field}' が off または on 内で不一致",
            }
        if off_values != on_values:
            return {
                "status": "undeterminable",
                "reason": (
                    f"'{field}' が off/on 間で不一致（off={off_values!r} "
                    f"on={on_values!r}）"
                ),
            }

    # 計測時間の有限性・正数性を off/on 両方について検証する（codex-review
    # 指摘: off 側のみ検査していると on 側の負数・Infinity/NaN を弾けない）。
    for label, group_rows in (("off", off_rows), ("on", on_rows)):
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

    off_medians = _median([r["median_s"] for r in off_rows])
    on_medians = _median([r["median_s"] for r in on_rows])
    if off_medians <= 0:
        return {"status": "undeterminable", "reason": "off_median_s <= 0"}

    off_checksums = [r.get("checksum") for r in off_rows]
    on_checksums = [r.get("checksum") for r in on_rows]
    # イシュー #1353（github-actions レビュー指摘）: `bool` は `int` の
    # サブクラス（`True == 1`）のため、`checksum` に `bool` が混入すると
    # `checksums_match`／`==` が数値として扱ってしまい「完全一致」と
    # 誤判定されうる。欠損（`None`）に加え非数値・`bool` も明示的に拒否
    # する（fail-closed）。
    if any(
        c is None or isinstance(c, bool) or not isinstance(c, (int, float))
        for c in off_checksums + on_checksums
    ):
        return {
            "status": "undeterminable",
            "reason": "checksum 欠損または非数値の行あり",
        }
    ref_checksum = off_checksums[0]
    composite_ok = all(
        checksum_contract.checksums_match(ref_checksum, c)
        for c in off_checksums + on_checksums
    )
    exact_ok = all(c == ref_checksum for c in off_checksums + on_checksums)
    if not composite_ok:
        return {
            "status": "undeterminable",
            "reason": (
                f"checksum が複合判定を外れる（off={off_checksums!r} "
                f"on={on_checksums!r}）"
            ),
        }

    return {
        "status": "ok",
        "reason": None,
        "off_median_s": off_medians,
        "on_median_s": on_medians,
        "ratio": on_medians / off_medians,
        "off_min_s": min(r["median_s"] for r in off_rows),
        "off_max_s": max(r["median_s"] for r in off_rows),
        "on_min_s": min(r["median_s"] for r in on_rows),
        "on_max_s": max(r["median_s"] for r in on_rows),
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


def render_markdown(cells):
    """セルごとの判定結果を Markdown 表として整形する。"""
    lines = []
    lines.append("| cell (task/device/size/mode/phase) | off median | on median | on/off | checksum | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    # 出力順序を安定させるため cell key でソートする（None を含みうるため
    # 文字列化してから比較する）。
    for key in sorted(cells.keys(), key=lambda k: tuple(str(v) for v in k)):
        off_rows = cells[key]["off"]
        on_rows = cells[key]["on"]
        result = evaluate_cell(off_rows, on_rows)
        cell_label = "/".join(str(v) for v in key)
        if result["status"] != "ok":
            lines.append(f"| {cell_label} | - | - | - | - | 判定不能: {result['reason']} |")
            continue
        checksum_label = (
            "完全一致"
            if result["checksum_exact_match"]
            else ("複合判定 ok" if result["checksum_composite_match"] else "不一致")
        )
        # ADOPT 候補の判定は時間比だけでなく checksum 完全一致も要求する
        # （codex-review 指摘: docs の採用条件「checksum 完全一致」との整合。
        # 複合判定 ok に留まる僅差〈例 off=1.0/on=1.000001〉を ADOPT 候補と
        # 表示してしまうと、採用条件を満たしていないのに満たしたかのように
        # 読める）。
        verdict = (
            "ADOPT 候補"
            if result["ratio"] <= 1.0 and result["checksum_exact_match"]
            else "後退（REJECT 方向）"
        )
        lines.append(
            f"| {cell_label} | {_fmt_ms(result['off_median_s'])} "
            f"(min {_fmt_ms(result['off_min_s'])} / max {_fmt_ms(result['off_max_s'])}) | "
            f"{_fmt_ms(result['on_median_s'])} "
            f"(min {_fmt_ms(result['on_min_s'])} / max {_fmt_ms(result['on_max_s'])}) | "
            f"{result['ratio']:.4f} | {checksum_label} | {verdict} |"
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
    cells = split_off_on(rows)
    if not cells:
        print("判定不能: 入力ファイルに行がない", file=sys.stderr)
        return 3
    print(render_markdown(cells))
    # 1 セルでも undeterminable があれば非 0 終了（fail-closed。呼び出し元が
    # 出力を読まずに成功と誤判定しないようにする）。
    any_undeterminable = any(
        evaluate_cell(v["off"], v["on"])["status"] != "ok" for v in cells.values()
    )
    return 3 if any_undeterminable else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
