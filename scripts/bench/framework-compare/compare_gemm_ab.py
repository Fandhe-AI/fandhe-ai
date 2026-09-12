#!/usr/bin/env python3
"""イシュー #1306: Metal GEMM の before（正式系列。承認ピン〈現行
`fandhe-ai =0.8.0`。#1487〉）/after（参考系列 HEAD。`crates/facade` への
path patch）2 バイナリ A/B 比較。

`run_ab_gemm_metal.sh` が出力する 2 本の JSONL（`bench-fandhe --task gemm
--device metal` を N=512/1024/2048/4096 × fresh/reuse × 5 回起動した結果）
を読み、`(size, mode)` セルごとに before/after の `median_s` を集約して
Markdown 表を出力する。

`--device`（既定 `metal`。後方互換）で `metal`／`cpu` を選択できる
（イシュー #1364）。`cpu` はセル集合が `{512,1024,2048}×{fresh,reuse}`
（`compare_gemm_gate.py --device cpu` と同じ N 集合。N=4096 は CPU GEMM
ゲート計測の対象外）。同一バイナリ（承認ピンの facade path patch を固定
し、環境変数 `RAYON_NUM_THREADS` の有無のみを切替える）の on/off 比較
にも本ツールを流用する（#1364「既定スレッド数限定」A/B）。
判定ロジック（threshold・checksum 複合判定・parity fail-closed）は
device に関わらず不変。

`compare_ab.py` は `framework_version` が before/after で同一だと fail-closed
拒否する（同一バージョンの A/B は意味を持たないという前提）ため、before/after
とも承認ピンのバージョン文字列を名乗る本用途（before=registry・after=HEAD
path patch。バージョン文字列は変わらない）には流用できない。`compare_managed_ab.py`
は同一バイナリのフラグ（`managed`）切替専用で、2 本の異なるバイナリを別ファイル
として比較する構造を持たない。本ツールは両者の fail-closed 方針を踏襲しつつ、
2 ファイル入力・`(size, mode)` セルキー（device=metal・task=gemm 固定）に
特化する。

`--task train`（既定 `gemm`。イシュー #1517）指定時は `bench-fandhe
--task train` が emit する単一形状（`size=BATCH=64`）の 2 セルを対象に
する（split-K 結線前後 A/B の train セル判定用。`run_ab_splitk_metal.sh`
が出力する JSONL を読む）。

fail-closed 方針（security.md A08。`compare_managed_ab.py` と同方針）:
- `framework != "fandhe-ai"`・`task != <--task の指定値>`・
  `device != "metal"`・`tf32:true`・`managed:true` の行は判定不能行と
  して除外する
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
import re
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

# device 別のセル集合（イシュー #1364）。`cpu` は `compare_gemm_gate.py
# --device cpu` と同じ N=512/1024/2048 のみ（N=4096 は CPU GEMM ゲート
# 計測の対象外。`run_gemm_gate_cpu.sh` が発行する N 集合と揃える）。
# `cuda` はイシュー #1337 で追加（`run_gemm_gate_cuda.sh` の対象 N
# 集合＝1024/2048/4096 のみ。metal のような 512 込みの独自 8 セル用途
# 〈#1306〉の前例は cuda に無いため、既定 = gate セル集合とする）。
_VALID_SIZES_BY_DEVICE = {
    "metal": frozenset({512, 1024, 2048, 4096}),
    "cpu": frozenset({512, 1024, 2048}),
    "cuda": frozenset({1024, 2048, 4096}),
}
_VALID_DEVICES = frozenset(_VALID_SIZES_BY_DEVICE)
DEFAULT_DEVICE = "metal"

# イシュー #1337: `run_gemm_gate.sh`（cuda/metal/cpu 共通ロジック）が発行
# する N 集合（`run_gemm_gate.sh` 冒頭コメント参照）。`--sizes gate` 指定時
# はこの集合をセル集合として使う（`metal` の既定 8 セル用途〈#1306〉とは
# 独立に、ゲート再計測専用の入力を扱うため）。`cpu`／`cuda` は元々この集合と
# 同一（`_VALID_SIZES_BY_DEVICE` 参照）だが、`metal` は 512 を含む 8 セルが
# 既定のため明示的に絞り込む必要がある。
_GATE_SIZES_BY_DEVICE = {
    "metal": frozenset({1024, 2048, 4096}),
    "cpu": frozenset({512, 1024, 2048}),
    "cuda": frozenset({1024, 2048, 4096}),
}

# イシュー #1517: `--task train` 用のセル集合。`bench-fandhe --task train`
# は `Record.size` に `BATCH`（`scripts/bench/framework-compare/
# bench-fandhe/src/main.rs` の `const BATCH: usize = 64;`）を emit する
# （train タスクは gemm と異なり N をスイープしない・単一形状のみ）。
# device に依らず単一値（現状 metal 限定用途だが、将来 cpu/cuda へ拡張
# しても train の形状定義は `bench-fandhe` 側の定数に従うため、device
# 別の分岐は設けない）。
_VALID_SIZES_TRAIN = frozenset({64})
DEFAULT_TASK = "gemm"
_VALID_TASKS = frozenset({"gemm", "train"})


def _size_set_for(device, sizes_arg, task=DEFAULT_TASK):
    """`--device`/`--sizes`/`--task` から実際に使うセルサイズ集合を決める。

    `sizes_arg` は `"full"`（既定・後方互換。`_VALID_SIZES_BY_DEVICE` を
    そのまま使う）または `"gate"`（`_GATE_SIZES_BY_DEVICE` へ絞り込む）。
    `task == "train"` の場合は `sizes_arg` を無視し `_VALID_SIZES_TRAIN`
    （単一形状）を返す（train タスクに "gate" の概念は存在しない）。
    """
    if task == "train":
        return _VALID_SIZES_TRAIN
    if sizes_arg == "gate":
        return _GATE_SIZES_BY_DEVICE[device]
    return _VALID_SIZES_BY_DEVICE[device]

# 既定の非後退閾値（median 比 ratio = after/before が 1.05 以下なら非後退。
# guardrail「劣化中央値 5% 以内」の慣例値。閾値変更は本ツールの CLI 引数
# としてのみ与え、コード定数としては固定しない。判定に使う値は既定のまま
# docs へ記録する）。
DEFAULT_THRESHOLD = 1.05


def _valid_cell_identity(obj, device, size_set=None, task=DEFAULT_TASK):
    """`_cell_key` がグループ化に使う `task`/`device`/`size`/`mode` の型・
    値域を検証する（`compare_managed_ab.py::_valid_cell_identity` と同方針。
    未検証のまま集約すると、これらのフィールドを欠いた行が単一の偽セルへ
    迂回して集約され、比較対象が不明なまま "ok" 判定になりうる）。

    `device`（`metal`／`cpu`／`cuda`）でセル集合の N を切り替える
    （イシュー #1364・#1337）。`size_set` 省略時は `_VALID_SIZES_BY_DEVICE
    [device]`（既定の `--sizes full`）を使う。`mode` フィールド自体の妥当性
    （`_VALID_MODES` 内か）はここで検証するが、`--modes` による絞り込み
    （例: `reuse` のみ）は呼び出し側（`load_rows`）が別途「無視して集約
    しない」形で扱う（不正行としては扱わない。#1337）。

    `task`（既定 `"gemm"`・後方互換）で行の `task` フィールド自体の期待値を
    切り替える（イシュー #1517「`--task train`」）。行が求める `task`
    以外（例: `--task train` 実行時に紛れ込んだ `task:"gemm"` 行や
    `bench-fandhe --phases` が emit する `task:"train_phases"` 行）は不正
    として除外する——`bench-fandhe` は `--phases` 実行時に `task:"train"`
    行を emit しない契約（`main.rs` 実測確認済み）のため実運用では混在
    しないが、本関数は入力の型・値域を機械的に検証する層であり前提を
    信用せず fail-closed に拒否する。
    """
    row_task = obj.get("task")
    if not isinstance(row_task, str) or row_task != task:
        return False
    row_device = obj.get("device")
    if not isinstance(row_device, str) or row_device != device:
        return False
    mode = obj.get("mode", "fresh")
    if not isinstance(mode, str) or mode not in _VALID_MODES:
        return False
    size = obj.get("size")
    valid_sizes = size_set if size_set is not None else _VALID_SIZES_BY_DEVICE[device]
    if isinstance(size, bool) or not isinstance(size, int) or size not in valid_sizes:
        return False
    return True


def load_rows(path, device=DEFAULT_DEVICE, size_set=None, modes=None, task=DEFAULT_TASK):
    """JSONL を読み、不正な行は理由付きで報告しスキップする（A08）。

    `size_set`（`--sizes` 由来。省略時 `_VALID_SIZES_BY_DEVICE[device]`）・
    `modes`（`--modes` 由来。省略時 `_VALID_MODES`＝両方）で絞り込む。
    `mode` フィールド自体が `_VALID_MODES` 外の不正値であれば従来どおり
    警告付きでスキップするが、値は正しいが `modes` に含まれない行
    （例: `--modes reuse` 指定時の `fresh` 行）は「期待セル外」として
    警告なしで無視する（gate 出力が参考記録として fresh 行を含む場合が
    あるため。イシュー #1337 README「借用ビュー readout」節）。
    """
    if modes is None:
        modes = _VALID_MODES
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
            # イシュー #1339: `device_checksum` も `tf32`／`managed` と同じ
            # 型検証・除外を適用する。
            if "device_checksum" in obj and not isinstance(obj["device_checksum"], bool):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'device_checksum' フィールド型（bool を"
                    f"期待。実際: {obj['device_checksum']!r}） — skipped"
                )
                continue
            if obj.get("device_checksum", False) is True:
                warnings.append(
                    f"{path}:{lineno}: 'device_checksum:true' の行は本 A/B の対象外 "
                    "— skipped"
                )
                continue
            # イシュー #1477: `readout`（Metal 借用ビュー readout の
            # legacy/borrowed override 計測）も `tf32`／`managed`／
            # `device_checksum` と同じ理由で型検証・除外する
            # （`compare_readout_ab.py` が専用の interleave A/B 比較を
            # 別途担う）。
            if "readout" in obj and not isinstance(obj["readout"], str):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'readout' フィールド型（str を期待。"
                    f"実際: {obj['readout']!r}） — skipped"
                )
                continue
            if "readout" in obj:
                warnings.append(
                    f"{path}:{lineno}: 'readout' キーを持つ行は本 A/B の対象外 "
                    "— skipped"
                )
                continue
            # イシュー #1545: `metal_split_k`（Metal GEMM split-K opt-in
            # 経路の runtime トグル A/B。`--metal-split-k`）は
            # `readout`／`graph`／`managed` 等とは異なり**本比較器の対象
            # から除外しない** — `run_ab_splitk_metal.sh` は
            # `--metal-split-k off`（before 相当）／`--metal-split-k on`
            # （after 相当）で分離した別ファイルをそれぞれ本比較器の
            # before/after 入力として渡す設計であり、本比較器はどのフラグ
            # で分岐したかを意識せず「2 つの JSONL 間の中央値比・checksum
            # 一致」だけを見る。型検証（`"on"`/`"off"` の str のみ許容）
            # のみ適用し、不正型・不正値の行は理由付きで丸ごとスキップする
            # （fail-closed。README「`--metal-split-k <on|off>`」節参照）。
            if "metal_split_k" in obj and (
                not isinstance(obj["metal_split_k"], str)
                or obj["metal_split_k"] not in ("on", "off")
            ):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'metal_split_k' フィールド型/値"
                    f"（'on'/'off' の str を期待。実際: "
                    f"{obj['metal_split_k']!r}） — skipped"
                )
                continue
            if not _valid_cell_identity(obj, device, size_set=size_set, task=task):
                warnings.append(
                    f"{path}:{lineno}: 不正または欠損した 'task'/'device'/'size'/"
                    f"'mode' フィールド（行: {obj!r}） — skipped"
                )
                continue
            if obj.get("mode", "fresh") not in modes:
                # `--modes` による意図的な絞り込み。不正行ではないため
                # warning を出さず静かに除外する（上記 docstring 参照）。
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
        # `version` はいずれも承認ピンのバージョン文字列（現行 "0.8.0"）を
        # 名乗ることが前提（before=registry・after=HEAD path patch。
        # workspace.package.version 不変）。before/after 間で異なる場合は
        # 前提が崩れているため判定不能とする。
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


def _all_expected_cells(device=DEFAULT_DEVICE, size_set=None, modes=None):
    """契約上の全セル（device 別 `size_set`〈省略時 `_VALID_SIZES_BY_DEVICE
    [device]`〉× `modes`〈省略時 `_VALID_MODES`。metal 既定は 8 セル・
    cpu は 6 セル〉）を昇順で返す。

    `render_markdown`／`main` の集計対象を `cells` に実在するキーだけに
    限定すると、before/after 双方から同一セルが欠落した場合に何も表示
    されず・`any_bad` 判定にも寄与しないまま終了コード 0 になりうる
    （codex-review P2・Cursor Bugbot Medium 指摘。イシュー #1306 の
    「全セル非後退」契約に反する）。欠測セルを判定不能として明示する
    ため、実データに依らずこの固定集合を走査の基準にする。

    `size_set`／`modes`（イシュー #1337「`--sizes`／`--modes`」）は
    ゲート由来入力（`run_gemm_gate.sh`。cuda/metal は reuse のみ発行）を
    reuse セルのみで判定する用途に使う。
    """
    if size_set is None:
        size_set = _VALID_SIZES_BY_DEVICE[device]
    if modes is None:
        modes = _VALID_MODES
    return sorted((size, mode) for size in size_set for mode in modes)


def per_run_ratios(before_rows, after_rows):
    """`before_rows`/`after_rows`（各ちょうど 5 件・append 順＝run 順という
    計測スクリプトの契約〈`run_ab_gemm_metal.sh`／`run_ab_splitk_metal.sh`
    が run 単位で交互起動し、各腕の出力ファイルへその順で 1 行ずつ
    追記する〉を前提に、run k 番目同士の `after_k/before_k` 比を 5 件
    返す。イシュー #1517 実装計画 §4 rule (b) の「run 単位ペアの run 内比
    が 5/5 run すべて `>1.00` なら結線維持不可」を人間（Mac セッション）
    が機械的に判定できるようにするための診断値（判定そのものは行わない
    ——終了コード・`verdict` には影響しない。`--per-run` 指定時のみ
    `render_markdown` が表へ追記する）。

    件数が 5 件ちょうどでない場合は `None`（`evaluate_cell` の「ちょうど
    5 件」契約と同じ理由で判定不能扱いとする）。
    """
    if len(before_rows) != 5 or len(after_rows) != 5:
        return None
    ratios = []
    for b, a in zip(before_rows, after_rows):
        bv = b.get("median_s")
        av = a.get("median_s")
        if (
            not isinstance(bv, (int, float))
            or isinstance(bv, bool)
            or not isinstance(av, (int, float))
            or isinstance(av, bool)
            or not math.isfinite(bv)
            or not math.isfinite(av)
            or bv <= 0
        ):
            return None
        ratios.append(av / bv)
    return ratios


def render_markdown(cells, threshold, device=DEFAULT_DEVICE, size_set=None, modes=None, per_run=False):
    """`cells` を Markdown 表として整形する。

    列は `columns` リストへ 1 列 1 要素で積んでから
    `"| " + " | ".join(columns) + " |"` で組み立てる（header・sep・各データ
    行のすべてで同一の組み立て方をする）。以前は文字列スライス
    （`header[:-2]` 等）と行末への直接連結（`f"...{tail}"`／
    `f"...{per_run_tail}"`）で `--per-run` 列を継ぎ足しており、
    区切り行のスライスが `---` の 1 文字を余分に削って列数がずれ、
    データ行は基本列の末尾 `|` と追加列の先頭 `|` が連結されて `||` の
    空列が生じていた（codex-review P2・Cursor Bugbot Low Severity
    指摘。イシュー #1517 PR #1531）。列リスト方式は header・sep・
    データ行の列数を機械的に一致させるため、この種のずれが再発しない。
    非 `--per-run`（既定）出力は本リファクタ前とバイト同一。
    """
    lines = []
    base_columns = ["size/mode", "before median", "after median", "after/before", "checksum", "判定"]
    per_run_columns = ["run 内比（5 run）", "符号一貫（全 run >1.00）"]
    columns = base_columns + per_run_columns if per_run else list(base_columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for key in _all_expected_cells(device, size_set=size_set, modes=modes):
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
            row = [cell_label, "-", "-", "-", "-", f"判定不能: {result['reason']}"]
            if per_run:
                row += ["-", "-"]
            lines.append("| " + " | ".join(row) + " |")
            continue
        checksum_label = (
            "完全一致"
            if result["checksum_exact_match"]
            else ("複合判定 ok" if result["checksum_composite_match"] else "不一致")
        )
        note = ""
        if result["before_spread"] > 1.5:
            note = "（判定注意: before spread > 1.5x・負荷ノイズの疑い）"
        row = [
            cell_label,
            f"{_fmt_ms(result['before_median_s'])} "
            f"(min {_fmt_ms(result['before_min_s'])} / max {_fmt_ms(result['before_max_s'])})",
            f"{_fmt_ms(result['after_median_s'])} "
            f"(min {_fmt_ms(result['after_min_s'])} / max {_fmt_ms(result['after_max_s'])})",
            f"{result['ratio']:.4f}",
            checksum_label,
            f"{result['verdict']}{note}",
        ]
        if per_run:
            ratios = per_run_ratios(before_rows, after_rows)
            if ratios is None:
                row += ["判定不能", "-"]
            else:
                sign_consistent = all(r > 1.0 for r in ratios)
                ratios_str = ", ".join(f"{r:.4f}" for r in ratios)
                row += [ratios_str, "はい" if sign_consistent else "いいえ"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


_PHASE_NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _valid_phase_row(obj, device):
    """`train_phases` 行（`bench-fandhe --task train --phases`。イシュー
    #1009）として `--phases` 診断表に使ってよいかを検証する。

    `compare_ab.py::_valid_phase_row` は `device == "cuda"` 固定（イシュー
    #1083 の CUDA 専用ツール）だが、本関数は呼び出し元が渡す `device`
    （`metal` 等）と突き合わせる（イシュー #1517「`train --phases` 診断表」。
    `compare_ab.compare_phases` を流用しない理由は同モジュール docstring
    参照）。
    """
    if not isinstance(obj, dict):
        return False
    if obj.get("framework") != "fandhe-ai" or obj.get("task") != "train_phases":
        return False
    if obj.get("device") != device:
        return False
    phase = obj.get("phase")
    if not isinstance(phase, str) or not _PHASE_NAME_RE.match(phase):
        return False
    v = obj.get("median_s")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        return False
    return True


def render_phases_table(before_rows, after_rows, device, mode):
    """`--phases` 診断表（1 回計測・参考値。判定には用いない）を Markdown
    で返す。`before_rows`/`after_rows` は別ファイル（`--phases` 引数）から
    読んだ生の JSON オブジェクトのリスト。一致する行が無ければ空文字列
    （呼び出し元が節ごと省略する）。
    """
    before = [r for r in before_rows if _valid_phase_row(r, device) and r.get("mode", "fresh") == mode]
    after = [r for r in after_rows if _valid_phase_row(r, device) and r.get("mode", "fresh") == mode]
    if not before or not after:
        return ""
    before_by_phase = {r["phase"]: r for r in before}
    after_by_phase = {r["phase"]: r for r in after}
    phases = sorted(
        set(before_by_phase) & set(after_by_phase),
        key=lambda p: (before_by_phase[p].get("phase_index", 1 << 30), p),
    )
    if not phases:
        return ""
    lines = [f"### フェーズ分解（診断用・{mode}・単発計測・非判定）", ""]
    lines.append("| phase | before | after | after/before |")
    lines.append("|---|---|---|---|")
    for p in phases:
        b = before_by_phase[p]["median_s"]
        a = after_by_phase[p]["median_s"]
        ratio = f"{(a / b):.3f}" if b > 0 else "-"
        lines.append(f"| {p} | {_fmt_ms(b)} | {_fmt_ms(a)} | {ratio} |")
    return "\n".join(lines)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Metal GEMM before/after (approved pin vs HEAD) A/B comparison (issue #1306)"
    )
    parser.add_argument("before", help="before JSONL path")
    parser.add_argument("after", help="after JSONL path")
    parser.add_argument(
        "--device",
        choices=sorted(_VALID_DEVICES),
        default=DEFAULT_DEVICE,
        help=(
            "device 別セル集合を選択する（既定 'metal'。後方互換。"
            "'cpu' は N=512/1024/2048 のみ。イシュー #1364）"
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"non-regression ratio threshold (after/before; default {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--sizes",
        choices=("full", "gate"),
        default="full",
        help=(
            "セルサイズ集合。'full'（既定・後方互換。device 別の "
            "_VALID_SIZES_BY_DEVICE）または 'gate'（run_gemm_gate.sh 由来入力用。"
            "metal は 512 を除いた 1024/2048/4096 のみに絞り込む。イシュー #1337）"
        ),
    )
    parser.add_argument(
        "--modes",
        default="fresh,reuse",
        help=(
            "判定対象のカンマ区切り mode 集合（既定 'fresh,reuse'＝両方。"
            "cuda/metal のゲート出力は reuse のみを発行するため "
            "'--modes reuse' で fresh 参考行をセル外扱いにできる。"
            "イシュー #1337）"
        ),
    )
    parser.add_argument(
        "--task",
        choices=sorted(_VALID_TASKS),
        default=DEFAULT_TASK,
        help=(
            "判定対象タスク（既定 'gemm'・後方互換。'train' は "
            "bench-fandhe --task train が emit する単一形状 "
            "（size=BATCH=64）のセルを対象にする。イシュー #1517。"
            "'train' 指定時 --sizes は無視する（train に gate の概念は"
            "ない）"
        ),
    )
    parser.add_argument(
        "--phases",
        nargs=2,
        metavar=("BEFORE_PHASES_JSONL", "AFTER_PHASES_JSONL"),
        default=None,
        help=(
            "`--task train` 限定の診断表（`bench-fandhe --task train "
            "--phases` が出す task:\"train_phases\" 行の before/after を "
            "1 回計測のまま並べる。判定には用いない参考値。"
            "イシュー #1517）"
        ),
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help=(
            "各セルへ run 単位（append 順＝run 順）の `after_k/before_k` "
            "比 5 件と「5 run 全てで比 > 1.00（符号一貫）」フラグを追加列と"
            "して表示する（既定 off・既定出力はバイト不変。判定〈終了"
            "コード・verdict〉には影響しない診断列。イシュー #1517 実装"
            "計画 §4 rule (b) の『run 内比が 5/5 run すべて > 1.00 なら"
            "結線維持不可』を人間が機械的に確認するための値）"
        ),
    )
    parser.add_argument(
        "--require-checksum-exact",
        action="store_true",
        help=(
            "既定 off（後方互換。既存呼び出しは `checksum_composite_match` "
            "のみで判定する契約を維持する）。指定時は各セルの "
            "`checksum_exact_match` が False の場合も、比が threshold 内で"
            "あっても non-regression 判定から除外し、終了コードへ反映する"
            "（イシュー #1560 codex-review [P1] 指摘: 事前登録規則が "
            "reuse セルの checksum 完全一致を必須としている呼び出し向けの"
            "明示 opt-in）"
        ),
    )
    args = parser.parse_args(argv[1:])
    if args.phases is not None and args.task != "train":
        print("ERROR: --phases は --task train と併用する場合のみ有効", file=sys.stderr)
        return 2

    modes = frozenset(m.strip() for m in args.modes.split(",") if m.strip())
    invalid_modes = modes - _VALID_MODES
    if not modes or invalid_modes:
        print(
            f"ERROR: --modes に不正な値が含まれる（invalid={sorted(invalid_modes)!r}・"
            f"許容値={sorted(_VALID_MODES)!r}）",
            file=sys.stderr,
        )
        return 2
    size_set = _size_set_for(args.device, args.sizes, task=args.task)

    before_rows, before_warnings = load_rows(
        args.before, args.device, size_set=size_set, modes=modes, task=args.task
    )
    after_rows, after_warnings = load_rows(
        args.after, args.device, size_set=size_set, modes=modes, task=args.task
    )
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

    print(
        render_markdown(
            cells, args.threshold, args.device, size_set=size_set, modes=modes, per_run=args.per_run
        )
    )

    if args.phases is not None:
        try:
            with open(args.phases[0], encoding="utf-8") as f:
                before_phase_rows = [json.loads(line) for line in f if line.strip()]
            with open(args.phases[1], encoding="utf-8") as f:
                after_phase_rows = [json.loads(line) for line in f if line.strip()]
        except (OSError, json.JSONDecodeError) as e:
            print(f"WARNING: --phases 入力の読み込みに失敗した（{e}） — 診断表を省略", file=sys.stderr)
        else:
            for mode in sorted(modes):
                table = render_phases_table(
                    before_phase_rows, after_phase_rows, args.device, mode
                )
                if table:
                    print()
                    print(table)

    any_bad = False
    for key in _all_expected_cells(args.device, size_set=size_set, modes=modes):
        rows = cells.get(key, [])
        if not rows:
            any_bad = True
            continue
        before_rows_k = [r for r in rows if not r.get("_is_after")]
        after_rows_k = [r for r in rows if r.get("_is_after")]
        result = evaluate_cell(before_rows_k, after_rows_k, args.threshold)
        if result["status"] != "ok" or result.get("verdict") != "非後退":
            any_bad = True
        elif args.require_checksum_exact and not result.get("checksum_exact_match", False):
            print(
                f"NG: {key} は checksum_exact_match=False のため "
                "--require-checksum-exact により後退相当として扱う",
                file=sys.stderr,
            )
            any_bad = True
    return 3 if any_bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
