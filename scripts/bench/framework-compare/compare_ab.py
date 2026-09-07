#!/usr/bin/env python3
"""都度同期廃止（#1011）の実践規模 A/B 比較ツール（イシュー #1083）。

使い方:
    python3 compare_ab.py BEFORE.jsonl AFTER.jsonl [--out FILE]

`run_ab_train_cuda.sh` が出力する 2 本の JSONL（`bench-fandhe train cuda 64`
を fresh/reuse 各 5 回起動した結果）を読み、`(mode)` ごとに before/after の
`median_s` を集約して Markdown 表を出力する。

設計方針（summarize.py と同じ思想を踏襲。判定基準の定数〈CHECKSUM_ABS_TOL/
CHECKSUM_REL_TOL〉と複合判定〈checksums_match〉は checksum_contract.py を
単一真実源として summarize.py と共有する）:
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
import importlib.util
import json
import math
import os
import re
import sys

# 数値一致契約（tolerance 定数・checksums_match）は checksum_contract.py を
# 単一真実源として参照する（codex-review P1 指摘・PR #1088: 分散定義の禁止）。
# sys.path に依存しないファイルパス指定 import（テストと同じ方式）。
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "checksum_contract",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "checksum_contract.py"),
)
_checksum_contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(_checksum_contract)
CHECKSUM_ABS_TOL = _checksum_contract.CHECKSUM_ABS_TOL
CHECKSUM_REL_TOL = _checksum_contract.CHECKSUM_REL_TOL
checksums_match = _checksum_contract.checksums_match

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

# `version`（framework_version）の許容文字集合（codex-review P0 指摘・PR #1088:
# 未検証の version を Markdown 表へそのまま連結すると `|` や改行を含む外部
# 入力で列・行を追加できてしまう。A03 の「外部フォーマットのパース検証」に
# 従い、crate バージョン文字列として想定される英数字・`.`・`-`・`+`・`_`
# のみを許容し、それ以外（`|`・改行・バッククォート等の Markdown 制御文字を
# 含む）は schema 不正として warning 化する）。
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


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


def _protocol_int(rows, key):
    """`rows` 全件が同一の正整数値を持つ計測プロトコル値（`warmup`／
    `iters`）を検証する（Review 指摘・#1083: プロトコル値の schema 検証が
    無く異なる計測プロトコル混在でも比較が成立してしまう fail-open を塞ぐ）。

    戻り値は `(value, error)` のタプル。妥当なら `(int値, None)`、不正なら
    `(None, エラー理由の文字列)`。
    """
    values = {r.get(key) for r in rows}
    if len(values) != 1:
        return None, "単一値でない（混在入力）"
    v = next(iter(values))
    if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
        return None, f"正整数でない（{v!r}）"
    return v, None


def _train_row_schema_error(r):
    """`framework == fandhe-ai` かつ `task == train` の行に対する必須フィールド
    の schema 検証（codex-review P0 指摘・PR #1088: 構文上有効な JSON だが
    必須フィールド欠損・型不正な行が `_train_records` で無検証のまま黙って
    除外され、他 mode に正常行が `MIN_RECORDS` 件以上残っていれば A/B 判定
    が成功扱いになってしまう fail-open を塞ぐ）。

    schema 適合なら None、不適合ならエラー理由の文字列を返す。
    """
    mode = r.get("mode", "fresh")
    if mode not in ("fresh", "reuse"):
        return f"mode が fresh/reuse のいずれでもない（{mode!r}）"
    # device／size は比較対象（cuda・TRAIN_SIZE）を決める必須フィールド
    # （codex-review P0 指摘・PR #1088）。値の一致検証（device=="cuda"・
    # size==TRAIN_SIZE）自体は `_train_records` が既に担い、他 device・
    # 他 size の正当な行（例: device="cpu" の別ベンチ行）を黙って除外する
    # 設計を保つ（compare_ab_test.py
    # test_other_framework_or_device_rows_are_excluded・
    # test_rows_with_different_size_are_excluded が回帰防止として固定）。
    # ここで検証するのは「型として schema に適合するか」のみ: device 欠損
    # （None 等）・size が bool／文字列等の型不正は、値の意味を問う以前に
    # レコードとして壊れているため warning 化して入力全体を判定不能にする。
    device = r.get("device")
    if not isinstance(device, str) or not device:
        return f"device が非空文字列でない（{device!r}）"
    size = r.get("size")
    if isinstance(size, bool) or not isinstance(size, int):
        return f"size が整数でない（{size!r}）"
    if not isinstance(r.get("version"), str) or not _VERSION_RE.match(r.get("version")):
        return (
            "version が許容文字集合（英数字・`.`・`-`・`+`・`_`）の"
            f"semver 形式でない（{r.get('version')!r}）"
        )
    if _safe_positive(r.get("median_s")) is None:
        return f"median_s が正の有限数でない（{r.get('median_s')!r}）"
    if _safe_finite(r.get("checksum")) is None:
        return f"checksum が有限数でない（{r.get('checksum')!r}）"
    for key in ("warmup", "iters"):
        v = r.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            return f"{key} が正整数でない（{v!r}）"
    return None


def load_rows(path):
    """JSONL を読み、不正な行は理由付きで報告しスキップする（A08: 捏造しない。
    呼び出し元は戻り値の 2 要素目〈警告文のリスト〉を表示・記録する）。

    不正行の判定は 3 段階（いずれも該当すれば warning を追加し `rows` へは
    含めない）:
      1. JSON として parse 不能
      2. parse 結果が JSON object（dict）でない（scalar・配列等。正当な
         レコードは常に object であるため schema 不正として扱う）
      3. `framework == fandhe-ai` かつ `task == train` の行で、
         `_train_row_schema_error` が必須フィールド欠損・型不正を検出した
         場合（codex-review P0 指摘・PR #1088）

    呼び出し元（`main`）は戻り値の warnings が非空の場合、残った正常行の
    件数に関わらず入力全体を判定不能（fail-closed・非 0 終了）として扱う
    契約とする（Review 指摘・#1083: 破損・切り詰められた外部 JSONL でも
    各 mode 5 件以上の正常行が残れば成功扱いになる fail-open を塞ぐ）。
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
            if obj.get("framework") == "fandhe-ai" and obj.get("task") == "train":
                err = _train_row_schema_error(obj)
                if err is not None:
                    warnings.append(
                        f"{path}:{lineno}: train record の schema が不正"
                        f"（{err}） — skipped"
                    )
                    continue
            # イシュー #1353: `managed`（`Record.managed`）は外部 JSONL 由来
            # の値のため、summarize.py/compare_gemm_gate.py と同じ
            # fail-closed 型検証を適用する。
            if "managed" in obj and not isinstance(obj["managed"], bool):
                warnings.append(
                    f"{path}:{lineno}: 不正な 'managed' フィールド型（bool を期待。"
                    f"実際: {obj['managed']!r}） — skipped"
                )
                continue
            rows.append(obj)
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
        # イシュー #1353: `managed:true`（CUDA managed memory 配置）行は
        # 本ツールの A/B 比較対象外（既定 device-only 配置とのゲート混同
        # 防止。managed 有無の A/B 自体は専用ツール `compare_managed_ab.py`
        # で行う）。
        if r.get("managed", False) is True:
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


def _path_markdown_error(path):
    """CLI から受け取った入力パスを Markdown レポートへ埋め込む前の fail-closed
    検証（codex-review P0 指摘・PR #1088: `before_path`／`after_path` をバック
    クォートで囲むだけで `render()` の Markdown へ連結すると、ファイル名に
    バッククォート・改行を含めて任意の見出し・表・指示文をレポートへ挿入できる。
    `--out` の生成物を perf 記録へ転記する運用では外部入力由来のパスが
    プロンプトインジェクションにも波及するため、A03 の「外部入力のパース検証」
    に従いエスケープではなく拒否で塞ぐ）。

    コードスパン（`` ` `` 囲み）を突き破れるバッククォートと、行構造を壊す
    改行を含む制御文字（C0 全域・DEL）を拒否する。正当な計測 JSONL のパスに
    これらが現れることはない。適合なら None、不適合なら理由の文字列を返す。
    """
    if not isinstance(path, str) or not path:
        return "パスが非空文字列でない"
    if "`" in path:
        return "パスにバッククォートを含む（Markdown コードスパンを突き破るため拒否）"
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in path):
        return "パスに改行等の制御文字を含む（Markdown の行構造を壊すため拒否）"
    return None


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
    if not isinstance(before_version, str) or not before_version:
        result["status"] = "undeterminable"
        result["reason"] = f"before の framework_version が非空文字列でない（{before_version!r}）"
        return result
    if not isinstance(after_version, str) or not after_version:
        result["status"] = "undeterminable"
        result["reason"] = f"after の framework_version が非空文字列でない（{after_version!r}）"
        return result
    if before_version == after_version:
        result["status"] = "undeterminable"
        result["reason"] = (
            f"before/after の framework_version が同一（{before_version!r}）"
            "— A/B 比較になっていない"
        )
        return result

    # 計測プロトコル値（warmup／iters）の schema 検証・before/after 一致検証
    # （Review 指摘・#1083）: 異なる計測プロトコル〈例: warmup/iters が違う〉
    # の記録同士を比較すると、性能差が実装変更由来なのか計測条件差由来なのか
    # 判別できなくなる。各値が before/after それぞれで単一の正整数であり、
    # かつ before/after 間で一致することを要求する。
    before_warmup, before_warmup_err = _protocol_int(before, "warmup")
    if before_warmup is None:
        result["status"] = "undeterminable"
        result["reason"] = f"before の warmup が{before_warmup_err}"
        return result
    after_warmup, after_warmup_err = _protocol_int(after, "warmup")
    if after_warmup is None:
        result["status"] = "undeterminable"
        result["reason"] = f"after の warmup が{after_warmup_err}"
        return result
    if before_warmup != after_warmup:
        result["status"] = "undeterminable"
        result["reason"] = (
            f"before/after の warmup が異なる（{before_warmup} != {after_warmup}）"
            "— 計測プロトコルが揃っていない"
        )
        return result

    before_iters, before_iters_err = _protocol_int(before, "iters")
    if before_iters is None:
        result["status"] = "undeterminable"
        result["reason"] = f"before の iters が{before_iters_err}"
        return result
    after_iters, after_iters_err = _protocol_int(after, "iters")
    if after_iters is None:
        result["status"] = "undeterminable"
        result["reason"] = f"after の iters が{after_iters_err}"
        return result
    if before_iters != after_iters:
        result["status"] = "undeterminable"
        result["reason"] = (
            f"before/after の iters が異なる（{before_iters} != {after_iters}）"
            "— 計測プロトコルが揃っていない"
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

    # render() より前に入力パス自体を検証する（codex-review P0 指摘・
    # PR #1088: 詳細は _path_markdown_error の docstring）。不正なパスは
    # レポートを一切生成せず非 0 終了する（fail-closed）。
    for label, path in (("before", args.before), ("after", args.after)):
        err = _path_markdown_error(path)
        if err is not None:
            print(f"error: {label} パスが不正: {err}", file=sys.stderr)
            return 2

    before_rows, before_warnings = load_rows(args.before)
    after_rows, after_warnings = load_rows(args.after)
    for w in before_warnings + after_warnings:
        print(f"warning: {w}", file=sys.stderr)

    # fail-closed（A08・Review 指摘・#1083）: 入力 JSONL に不正な行（JSON
    # parse 不能・schema 不正）が 1 行でもあれば、残った正常行だけで各 mode
    # 5 件以上を満たしていても「破損・切り詰められた外部入力を正常計測として
    # 確定表示する」ことを許さない。入力全体を判定不能扱いとし非 0 終了させる。
    #
    # この判定は render() 呼び出しより前に確定させ、`results` の各 mode の
    # status を undeterminable へ上書きしてから表を描画する（Cursor Bugbot
    # 指摘・PR #1088: has_invalid_lines の判定を render() 後に行うと、
    # Markdown 表の「判定」列には不正行検出前に確定した中央値・比率つきの
    # "ok" がそのまま残ってしまい、Phase D がその表をそのまま perf record
    # へコピーする経路で「判定不能」の情報が失われる fail-open になる）。
    has_invalid_lines = bool(before_warnings or after_warnings)

    results = [compare_mode(before_rows, after_rows, mode) for mode in MODES]
    if has_invalid_lines:
        invalid_reason = (
            "入力 JSONL に不正な行（JSON parse 不能／schema 不正）が"
            "含まれるため判定不能（fail-closed）"
        )
        for r in results:
            r["status"] = "undeterminable"
            r["reason"] = invalid_reason
        # フェーズ分解表（診断用）も同一入力由来のため、不正行検出時は
        # 実測値らしき数値を一切表に出さない（判定不能な入力からの診断値の
        # 独り歩きを防ぐ）。
        phase_results_by_mode = {mode: [] for mode in MODES}
    else:
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

    if has_invalid_lines:
        print(
            "undeterminable: 入力 JSONL に不正な行（JSON parse 不能）が"
            "含まれるため、正常行の件数に関わらず判定不能として扱う"
            "（fail-closed）",
            file=sys.stderr,
        )
    if any_undeterminable or has_invalid_lines:
        for r in results:
            if r["status"] != "ok":
                print(f"undeterminable: mode={r['mode']}（{r['reason']}）", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
