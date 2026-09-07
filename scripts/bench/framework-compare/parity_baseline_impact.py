#!/usr/bin/env python3
"""候補判定（スケール付き絶対誤差／ULP ベース）を fandhe-ai 本体側の parity
非後退契約（`ParityBaseline`/`assert_no_parity_regression`）へ適用した場合の
影響を机上で確認する（イシュー #1238。前段イシュー #1237 の引き継ぎ）。

## 位置づけ

#1237（`parity_tolerance_candidates.py`）は framework-compare の N=2048
candle 側 fail 4 要素（イシュー #1184 ダンプ）に対し、候補判定を現行複合判定
（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）へ OR 追加した場合の
fail 数を要素ダンプの実値から算出した。本スクリプトは同じ候補定義を、
fandhe-ai 本体側の parity 非後退契約（`crates/backend-cuda/tests/common/
parity_baseline.rs::BASELINES`。45 行。#491 系）へ適用した場合の影響を
机上で確認する。

**両者の重要な差**: #1237 は要素単位の fail ダンプ（fail した 2〜4 要素の
`d`・`max_ab` 等）を入力に持つため候補の救済可否を直接判定できたが、
`BASELINES` は行単位の集計値（`fail_count`・`baseline_mean_abs_diff_ceiling`・
`baseline_max_abs_diff_ceiling` 等）しか持たず、要素単位のダンプは存在しない。
よって本スクリプトは行単位の集計値から機械的に導ける範囲（no-op／全救済／
部分・未確定の 3 クラス分類と `fail_count` の値域）までしか判定できない
（詳細は `docs/perf/candle-parity-tolerance-baseline-impact.md` §3）。

## 入力分布の差（#1237 との重要な差その2）

`BASELINES` の入力は `bench_harness::rng::Xorshift64Star::fill_vec`/
`fill_vec_f16`（`next_f32` は `[-1, 1)` の一様分布）で生成される。#1237 の
対象データ（U[-0.5,0.5)、`M=max|A|・max|B| ≈ 0.25`）とはスケールが異なる
ため、#1237 の候補定義（`build_candidates_a()` の `m_mode="fixed0.25"`）を
そのまま転用すると本データセットのスケールを過小評価する。本スクリプトは
候補の係数（`c`・`eps_value`。`m_mode`/`k_mode="sqrtK"` の A-3 を除き
`k_mode="K"` の線形スケール）のみを `parity_tolerance_candidates.py` から
再利用し、`M` は `--scale-mode`（既定 `exact`）で各行ごとに実際の
`max|A|・max|B|` を計算するか、`upper-bound`（`M=1`。入力範囲 `[-1,1)` から
の事前上界）で近似する。**この `M` 導出方法は #1237 §7 引き継ぎ項目 1
（「A-2 の入力規模導出を実装する場合の方法」）への回答である**。

候補定義自体（`build_candidates_a()`/`build_candidates_b(k)`）は
`parity_tolerance_candidates.py` を importlib で読み込んで再利用し複製
しない。B-1（`max_partial` 基準）・B-3（`exact` 基準）は行単位の集計値
からは机上分類できない（実行時トレースまたは厳密真値が必要）ため、全行
「机上分類不能」として明示するのみで bound は計算しない。

## fail-closed 契約

- `BASELINES` 配列の構造的な行数（`ParityBaseline {` ブロック開始の出現数）
  と、フィールド抽出に成功した行数が一致しない場合は非ゼロ終了する
  （フィールド抽出漏れ・宣言スタイル変更のドリフト検出）
- 各行で `total == m * n`・`baseline_fail_count <= total` を検査し、
  不成立なら非ゼロ終了する
- **実測される行数は 45 である**（`ParityPath` 別内訳: WmmaTf32×9・
  WmmaTf32Opt×9・WmmaTf32Staged×8・MmaF16×1・WmmaF16×2・MmaTf32×11・
  MmaTf32VsWmmaStaged×2・SpecializedMmaF16×3 = 45。単純な
  `grep -c "ParityBaseline {"` はファイル中の `pub struct ParityBaseline {`
  定義自体の 1 件を含んで 46 と数えるため、行数そのものの指標には使えない
  ——本スクリプトはこの構造定義を含まない `BASELINES` 配列区間限定で
  カウントする）

## 依存

Python 3 標準ライブラリのみ（`argparse`・`importlib`・`math`・`os`・
`re`・`struct`・`sys`・`dataclasses`）。CI（`ci.yml` の `deps-forbidden`
ジョブ）では単体テスト（`parity_baseline_impact_test.py`）のみ実行し、実
ファイル（`crates/backend-cuda/tests/common/parity_baseline.rs`）に対する
計算は行わない（実行コストの都合。`--scale-mode exact` は 45 行中の重複
`(seed, m, k, n, dtype)` をメモ化しても約 1.2 億要素の生成を伴う）。

## 使い方

```bash
cd scripts/bench/framework-compare
python3 parity_baseline_impact.py \\
  --baselines ../../../crates/backend-cuda/tests/common/parity_baseline.rs
```
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from typing import Optional

# `parity_tolerance_candidates.py` を隣接ファイルとして importlib 経由で
# 読み込む（#1237 実装時の制約と同じ理由で `exec_module` より前に
# `sys.modules` へ登録する。同ファイル冒頭コメント参照）。候補定義の
# 二重管理を避けるため、候補パラメータ（`c`・`eps_value`）のみを本モジュール
# から再利用し、`M`（入力規模）は本スクリプト独自に導出する（上記docstring
# 「入力分布の差」参照）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_CAND_PATH = os.path.join(_HERE, "parity_tolerance_candidates.py")
_CAND_SPEC = importlib.util.spec_from_file_location(
    "parity_tolerance_candidates", _CAND_PATH
)
if _CAND_SPEC is None or _CAND_SPEC.loader is None:
    raise ImportError(f"parity_tolerance_candidates.py を読み込めない: {_CAND_PATH}")
parity_tolerance_candidates = importlib.util.module_from_spec(_CAND_SPEC)
sys.modules[_CAND_SPEC.name] = parity_tolerance_candidates
_CAND_SPEC.loader.exec_module(parity_tolerance_candidates)


# ---------------------------------------------------------------------------
# 既存救済（既存複合判定）の絶対誤差閾値を正本 `crates/backend-cpu/src/
# parity.rs::ABSOLUTE_RESCUE_THRESHOLD` から直接抽出する（ハードコード
# 分離・乖離検出のため。`summarize_test.py::_extract_f64_const` と同じ
# 抽出方式を用いる。codex-review 指摘・PR #1421 P1。値そのものの変更は
# `.claude/rules/coding-rust.md` によりユーザー承認必須のためこの
# スクリプトは読み取り専用）。
# ---------------------------------------------------------------------------

_BACKEND_CPU_PARITY_PATH = os.path.join(
    _HERE, "..", "..", "..", "crates", "backend-cpu", "src", "parity.rs"
)


def _extract_f64_const(source: str, name: str) -> float:
    """`pub const <name>: f64 = <value>;` 形式の宣言から数値を取り出す。

    正規表現クレート追加を避けるため stdlib `re` のみで済む簡易パーサー
    （本体の宣言スタイル固定が前提）。宣言が見つからない・数値化できない
    場合は fail-closed に例外を送出する。
    """
    match = re.search(rf"pub const {re.escape(name)}: f64 = ([^;]+);", source)
    if match is None:
        raise BaselineParseError(
            f"本体 parity.rs に `pub const {name}: f64 = ...;` の宣言が"
            "見つからない（宣言スタイルが変わった可能性）"
        )
    return float(match.group(1).strip())


def load_absolute_rescue_threshold(path: str = _BACKEND_CPU_PARITY_PATH) -> float:
    """正本 `ABSOLUTE_RESCUE_THRESHOLD`（既存複合判定の絶対誤差救済閾値）
    を本体ソースから読み取る。読み取り失敗も fail-closed（例外送出）。
    """
    try:
        with open(path, encoding="utf-8") as f:
            source = f.read()
    except OSError as err:
        raise BaselineParseError(
            f"本体 parity.rs を読めない（{path}）: {err}。パスがずれていないか確認すること"
        ) from err
    return _extract_f64_const(source, "ABSOLUTE_RESCUE_THRESHOLD")


# ---------------------------------------------------------------------------
# `Xorshift64Star`（`crates/bench-harness/src/rng.rs`）の Python 移植。
#
# baseline 行の入力（A・B）を生成した実際の PRNG と同一のアルゴリズムを
# 再現し、行ごとの実測入力規模（`max|A|`・`max|B|`）を求める。固定値照合は
# `parity_baseline_impact_test.py::XorshiftPortTest` を参照
# （出典コマンドは同テストの docstring に記す）。
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_XORSHIFT_MUL = 0x2545F4914F6CDD1D
_ZERO_SEED_SUBSTITUTE = 0x9E3779B97F4A7C15


def xorshift_seed_state(seed: int) -> int:
    """`Xorshift64Star::new` の 0 シード補正を再現する。"""
    return seed if seed != 0 else _ZERO_SEED_SUBSTITUTE


def bits_extreme_scan(state: int, count: int) -> tuple[int, int, int]:
    """`count` 個の `next_u64` を生成しつつ、そこから導かれる 24bit
    `bits`（`Xorshift64Star::next_f32` が仮数部として使う値。
    `next_u64() >> 40`）の最小・最大を求める。

    `next_f32 = |2*bits/2^24 - 1|` は `bits` の凸関数（絶対値関数の合成）
    であるため、集合全体の絶対値最大は集合内の `bits` の最小値・最大値の
    いずれかで達成される（凸関数は区間の端点で最大化される一般性質。
    `docs/perf/candle-parity-tolerance-baseline-impact.md` §3.2 に証明の
    要旨を記す）。よって全要素を保持せず min/max の 2 値だけを追跡すれば
    十分であり、O(count) 時間・O(1) 空間で `max|A|`・`max|B|` を求められる。

    戻り値: `(次の state, min_bits, max_bits)`。`count == 0` の場合は
    `(state, -1, -1)` を返す（呼び出し側で空区間を判別できるようにする）。
    """
    if count == 0:
        return state, -1, -1
    bmin = -1
    bmax = -1
    s = state
    for _ in range(count):
        x = s
        x ^= (x << 13) & _MASK64
        x ^= x >> 7
        x ^= (x << 17) & _MASK64
        s = x & _MASK64
        v = (s * _XORSHIFT_MUL) & _MASK64
        bits = (v >> 40) & 0xFFFFFF
        if bmin < 0 or bits < bmin:
            bmin = bits
        if bmax < 0 or bits > bmax:
            bmax = bits
    return s, bmin, bmax


def bits_to_f32(bits: int) -> float:
    """`Xorshift64Star::next_f32` の `bits -> [-1, 1)` 写像を再現する。"""
    unit = bits / float(1 << 24)
    return unit * 2.0 - 1.0


def round_f32_to_f16_abs(x: float) -> float:
    """f32 値を IEEE754 binary16（half）へ round-to-nearest-even で丸めた
    絶対値。`struct` の `'e'` フォーマット（Python 3.6+ 標準ライブラリの
    half-precision 対応）を使い、`half::f16::from_f32` と同じ丸め方式を
    再現する（`crates/bench-harness/src/rng.rs::fill_vec_f16` 参照）。
    """
    (rounded,) = struct.unpack("<e", struct.pack("<e", x))
    return abs(rounded)


# `ParityPath` ごとの入力 dtype（f32 か f16 か）。各テストファイルの
# `fill_vec`/`fill_vec_f16` 呼び出しを確認して確定した事実（#1238 実装時
# 実地確認。file:line は `docs/perf/candle-parity-tolerance-baseline-impact.md`
# §6 の表を参照）:
#   - f32（`fill_vec`）: WmmaTf32・WmmaTf32Opt・WmmaTf32Staged・MmaTf32・
#     MmaTf32VsWmmaStaged
#   - f16（`fill_vec_f16`）: MmaF16・WmmaF16・SpecializedMmaF16
F16_PATHS = frozenset({"MmaF16", "WmmaF16", "SpecializedMmaF16"})


_scale_cache: dict[tuple[int, int, int, int, bool], tuple[float, float]] = {}


def compute_scale(
    seed: int, m: int, k: int, n: int, is_f16: bool, mode: str
) -> tuple[float, float]:
    """1 行分の `(S_A, S_B) = (max|A|, max|B|)` を求める。

    `mode == "upper-bound"` では計算せず `(1.0, 1.0)`（入力範囲 `[-1,1)`
    からの事前上界）を返す。全救済判定には使えない（上界を過大評価する
    ため）ことに注意。`mode == "exact"`（既定）では実際に PRNG を
    `m*k + k*n` 回進めて厳密値を求める。同一 `(seed, m, k, n, is_f16)` は
    45 行中に複数回登場しうるため（例: `seed=0xBEEF, m=n=k=4096` は
    `WmmaTf32Opt`・`WmmaTf32Staged` の 2 行で共有）、プロセス内メモ化で
    重複計算を避ける。
    """
    if mode == "upper-bound":
        return 1.0, 1.0
    key = (seed, m, k, n, is_f16)
    cached = _scale_cache.get(key)
    if cached is not None:
        return cached

    state = xorshift_seed_state(seed)
    state, a_min, a_max = bits_extreme_scan(state, m * k)
    _state, b_min, b_max = bits_extreme_scan(state, k * n)

    def extreme_abs(bmin: int, bmax: int) -> float:
        if bmin < 0:
            return 0.0
        if is_f16:
            return max(
                round_f32_to_f16_abs(bits_to_f32(bmin)),
                round_f32_to_f16_abs(bits_to_f32(bmax)),
            )
        return max(abs(bits_to_f32(bmin)), abs(bits_to_f32(bmax)))

    s_a = extreme_abs(a_min, a_max)
    s_b = extreme_abs(b_min, b_max)
    _scale_cache[key] = (s_a, s_b)
    return s_a, s_b


# ---------------------------------------------------------------------------
# `BASELINES` 配列（`crates/backend-cuda/tests/common/parity_baseline.rs`）
# のパース。
# ---------------------------------------------------------------------------

_ARRAY_START = "pub static BASELINES: &[ParityBaseline] = &["
# ブロック境界の検出は `_FIELD_RE`（フィールド抽出）が前提とするフィールド
# 並び・空白書式から独立させる（codex-review 指摘・PR #1421 P2-3: 旧実装は
# `ParityBaseline {\n        path: ParityPath::` という `_FIELD_RE` とほぼ
# 同じリテラルを要求していたため、フィールド書式が変わると block_starts 側
# も同時に減り、構造的な行数とフィールド抽出成功数の不一致という
# fail-closed 契約自体が機能しなくなる欠陥があった）。`BASELINES` 配列
# 区間（`arr`）には `pub struct ParityBaseline {` という定義行は含まれない
# （区間は `pub static BASELINES: &[ParityBaseline] = &[` 以降なので構造体
# 定義より後ろから始まる）ため、`ParityBaseline {` という短いリテラルの
# 出現数だけで安全にブロック数を数えられる（実測: 45。本ファイル冒頭
# docstring「fail-closed 契約」参照）。
#
# **追加修正（codex-review 指摘・PR #1421 P2 再指摘）**: 上記の初回修正は
# `ParityBaseline` と `{` の間に単一スペースを要求する `ParityBaseline \{`
# のままだったため、`ParityBaseline\n    {` のような改行を挟む合法な書式
# ではブロック開始が検出漏れし、`_FIELD_RE` 側も同時に該当エントリを拾え
# なければ両者の件数が同じだけ減って一致してしまい、fail-closed 契約が
# 機能しない再発条件が残っていた。`\s*`（改行・タブを含む任意の空白 0 個
# 以上）へ緩め、`ParityBaseline` の直後に任意個の空白を挟んで `{` が続く
# 書式を独立に検出できるようにする。
_BLOCK_START_RE = re.compile(r"ParityBaseline\s*\{")

_FIELD_RE = re.compile(
    r"path: ParityPath::(?P<path>\w+),\s*"
    r'context: "(?P<context>[^"]*)",\s*'
    r"m: (?P<m>\d+),\s*"
    r"n: (?P<n>\d+),\s*"
    r"k: (?P<k>\d+),\s*"
    r"seed: (?P<seed>0x[0-9A-Fa-f]+|\d+),\s*"
    r"total: (?P<total_m>\d+) \* (?P<total_n>\d+),\s*"
    r"baseline_fail_count: (?P<fail>\d+),\s*"
    r"baseline_mean_abs_diff_ceiling: (?P<mean_ceil>[0-9.eE+-]+),\s*"
    r"(?:(?://[^\n]*\n\s*)*)"  # 行間の doc/コメント行（0 個以上）を許容する
    r"baseline_provenance_unconfirmed: (?P<prov>true|false),\s*"
    r"baseline_max_abs_diff_ceiling: (?P<max_abs>None|Some\([0-9.eE+-]+\)),\s*"
    r"baseline_max_rel_err_ceiling: (?P<max_rel>None|Some\([0-9.eE+-]+\)),\s*",
    re.S,
)


@dataclass(frozen=True)
class BaselineRow:
    path: str
    context: str
    m: int
    n: int
    k: int
    seed: int
    total: int
    fail_count: int
    mean_abs_diff_ceiling: float
    provenance_unconfirmed: bool
    max_abs_diff_ceiling: Optional[float]
    max_rel_err_ceiling: Optional[float]

    @property
    def is_f16(self) -> bool:
        return self.path in F16_PATHS


class BaselineParseError(RuntimeError):
    """`BASELINES` のパース・自己検査に失敗した場合の fail-closed 例外。"""


def _parse_option_f64(text: str) -> Optional[float]:
    if text == "None":
        return None
    m = re.fullmatch(r"Some\(([0-9.eE+-]+)\)", text)
    if not m:
        raise BaselineParseError(f"Option<f64> のパースに失敗: {text!r}")
    return float(m.group(1))


def parse_baselines(path: str) -> list[BaselineRow]:
    """`ParityBaseline` の Rust ソースから `BASELINES` 配列をパースする。

    fail-closed 契約（本ファイル冒頭docstring参照）: 構造的なブロック開始
    数とフィールド抽出成功数が一致しない、または各行の整合性検査
    （`total == m*n`・`fail_count <= total`）に失敗した場合は
    `BaselineParseError` を送出する。
    """
    if not os.path.isfile(path):
        raise BaselineParseError(f"--baselines のパスが存在しないか通常ファイルでない: {path!r}")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if _ARRAY_START not in text:
        raise BaselineParseError(
            f"`{_ARRAY_START}` が見つからない（宣言スタイルが変更された可能性）"
        )
    start = text.index(_ARRAY_START)
    end_marker = "\n];"
    if end_marker not in text[start:]:
        raise BaselineParseError("`BASELINES` 配列の終端 `];` が見つからない")
    end = text.index(end_marker, start)
    arr = text[start:end]

    block_starts = _BLOCK_START_RE.findall(arr)
    field_matches = list(_FIELD_RE.finditer(arr))

    if len(block_starts) != len(field_matches):
        raise BaselineParseError(
            f"構造的なブロック開始数（{len(block_starts)}）とフィールド抽出成功数"
            f"（{len(field_matches)}）が一致しない（宣言スタイル変更によるパース"
            "漏れの可能性。fail-closed で中断）"
        )
    if not field_matches:
        raise BaselineParseError("`BASELINES` 配列から 1 行もパースできなかった")

    rows: list[BaselineRow] = []
    for gm in field_matches:
        g = gm.groupdict()
        seed = int(g["seed"], 0)
        m_val = int(g["m"])
        n_val = int(g["n"])
        total_m = int(g["total_m"])
        total_n = int(g["total_n"])
        if total_m != m_val or total_n != n_val:
            raise BaselineParseError(
                f"context={g['context']!r}: total 式（{total_m} * {total_n}）が "
                f"m/n（{m_val}/{n_val}）と一致しない"
            )
        total = total_m * total_n
        fail_count = int(g["fail"])
        if fail_count > total:
            raise BaselineParseError(
                f"context={g['context']!r}: fail_count（{fail_count}）が total"
                f"（{total}）を超えている"
            )
        rows.append(
            BaselineRow(
                path=g["path"],
                context=g["context"],
                m=m_val,
                n=n_val,
                k=int(g["k"]),
                seed=seed,
                total=total,
                fail_count=fail_count,
                mean_abs_diff_ceiling=float(g["mean_ceil"]),
                provenance_unconfirmed=(g["prov"] == "true"),
                max_abs_diff_ceiling=_parse_option_f64(g["max_abs"]),
                max_rel_err_ceiling=_parse_option_f64(g["max_rel"]),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 候補ごとの机上分類（no-op／全救済／部分・未確定／机上分類不能）。
# `docs/perf/candle-parity-tolerance-baseline-impact.md` §3.3 の論理を実装
# する（doc とスクリプトで定義を共有し、二重管理しない）。
# ---------------------------------------------------------------------------

NO_OP = "no-op"
FULL_RESCUE = "全救済"
PARTIAL = "部分／未確定"
UNCLASSIFIABLE_CEILING = "分類不能（ceiling 未実測）"
UNCLASSIFIABLE_RUNTIME = "机上分類不能（実行時適用不可／要トレース）"

_ABS_RESCUE_THRESHOLD_CACHE: Optional[float] = None


def get_absolute_rescue_threshold() -> float:
    """`ABSOLUTE_RESCUE_THRESHOLD` をプロセス内で 1 回だけ本体ソースから
    読み取りキャッシュする（`classify` の既定閾値として使う）。
    """
    global _ABS_RESCUE_THRESHOLD_CACHE
    if _ABS_RESCUE_THRESHOLD_CACHE is None:
        _ABS_RESCUE_THRESHOLD_CACHE = load_absolute_rescue_threshold()
    return _ABS_RESCUE_THRESHOLD_CACHE


def classify(
    bound: float,
    ceiling: Optional[float],
    bound_is_upper: bool = False,
    threshold: Optional[float] = None,
) -> str:
    """`bound` に対する 1 行の 3 クラス分類。

    `threshold` は既存救済（既存複合判定）の絶対誤差閾値。省略時は正本
    `crates/backend-cpu/src/parity.rs::ABSOLUTE_RESCUE_THRESHOLD` から
    読み取った値（`get_absolute_rescue_threshold()`）を使う。値を直書き
    すると正本が変更された際に無断で乖離するため（codex-review 指摘・
    PR #1421 P1）、呼び出し側でのハードコードは避けること。

    - `bound < threshold`（厳密不等号）: 既存救済（`diff < threshold`）に
      完全に包含されるため `fail_count` は厳密に不変（no-op）。
      `bound == threshold` は境界の要素（`d == threshold`）が候補側でのみ
      救済されうるため no-op に含めない。この no-op 判定は `ceiling` の
      有無に関わらず確定できるため、`ceiling is None` の判定より**先に**
      行う（codex-review 指摘・PR #1421 P2-2: 旧実装は `ceiling is None`
      を先に判定していたため、ceiling 未実測かつ no-op の行
      〈`WmmaTf32Opt 512x512x512 seed=0x7A0` 等〉が誤って「分類不能」に
      なっていた）
    - `ceiling is None`: この行は `baseline_max_abs_diff_ceiling` が未実測
      （`BASELINES` 中 1 行のみ）ため全救済判定ができない
    - `ceiling <= bound`: ceiling は「表示桁の最終桁 +1」で切り上げた保守的
      な値のため、これを下回るなら行内の全要素が候補で救済される。
      ただし `bound_is_upper=True`（`--scale-mode upper-bound` 由来。`M=1`
      という入力規模の事前上界で計算した `bound`）の場合、`bound` 自体が
      実際の閾値を過大評価しているため「全救済」を確定できない
      （codex-review 指摘・PR #1421 P2-1）。この場合は「部分／未確定」へ
      落とす。no-op 判定は `bound` が過大評価であっても真の bound
      （<= 表示 bound）がさらに小さいだけなので `bound < threshold` から
      `真の bound < threshold` が導け、引き続き安全に確定できる
    - それ以外: 行内の一部要素のみ救済されるか、確定できない
      （要素単位ダンプが `BASELINES` には存在しないため）
    """
    if threshold is None:
        threshold = get_absolute_rescue_threshold()
    if bound < threshold:
        return NO_OP
    if ceiling is None:
        return UNCLASSIFIABLE_CEILING
    if ceiling <= bound:
        if bound_is_upper:
            return PARTIAL
        return FULL_RESCUE
    return PARTIAL


@dataclass(frozen=True)
class CandidateASpec:
    name: str
    c: float
    eps_label: str
    eps_value: float
    k_mode: str  # "K" | "sqrtK"

    def bound(self, k: int, m_scale: float) -> float:
        kscale = float(k) if self.k_mode == "K" else math.sqrt(k)
        return self.c * self.eps_value * kscale * m_scale


def build_candidate_a_specs() -> list[CandidateASpec]:
    """`parity_tolerance_candidates.build_candidates_a()` から `c`・
    `eps_value`・`k_mode` のみを抽出し、`m_mode`（`fixed0.25`/`actual`）は
    捨てて本スクリプト独自の `M`（`--scale-mode`）で統一評価する
    （候補パラメータの二重管理を避けつつ、入力分布の差 — 本 docstring
    「入力分布の差」節 — を吸収する）。`m_mode="actual"` の A-2 相当は
    `c=1, eps=2^-23, k_mode="K"` の A-1 エントリと本スクリプトの下では
    同一（重複除去済み）。
    """
    seen: set[tuple[float, str, str]] = set()
    specs: list[CandidateASpec] = []
    for cand in parity_tolerance_candidates.build_candidates_a():
        key = (cand.c, cand.eps_label, cand.k_mode)
        if key in seen:
            continue
        seen.add(key)
        specs.append(
            CandidateASpec(
                name=f"A c={cand.c} eps={cand.eps_label} {cand.k_mode}*M",
                c=cand.c,
                eps_label=cand.eps_label,
                eps_value=cand.eps_value,
                k_mode=cand.k_mode,
            )
        )
    return specs


def a4_bound(k: int, s_a: float, s_b: float) -> float:
    """A-4: `K * u * Σ|a_k・b_k|` の上界近似（`Σ|ab| <= K * S_A * S_B`）。
    緩すぎる参考値（`parity_tolerance_candidates.py::CandidateA4` 参照）。

    真の `Σ|ab|` は行内で要素依存であり `K * S_A * S_B` はその上界のため、
    この関数が返す `bound` は真の閾値以上（過大評価）。呼び出し側は
    `classify(..., bound_is_upper=True)` を渡し「全救済」の誤確定を防ぐ
    こと（codex-review 指摘・PR #1421 P2-1）。
    """
    u_f32 = 2.0**-24
    return float(k) * u_f32 * (float(k) * s_a * s_b)


def b2_bound(t: float, k: int, s_a: float, s_b: float) -> float:
    """B-2: `t * ulp(Σ|a_k・b_k|)` の上界近似（`Σ|ab|` を `K*S_A*S_B` で
    代表する。`parity_dump_truth.ulp_f32` を再利用）。

    `a4_bound` と同じ理由（`Σ|ab| <= K*S_A*S_B` の上界代用・`ulp` の単調性
    により真の bound 以上を返す）で、呼び出し側は
    `classify(..., bound_is_upper=True)` を渡すこと。
    """
    sum_abs_ab_upper = float(k) * s_a * s_b
    return t * parity_tolerance_candidates.parity_dump_truth.ulp_f32(sum_abs_ab_upper)


# ---------------------------------------------------------------------------
# Markdown 出力。
# ---------------------------------------------------------------------------


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(rows: list[BaselineRow], scale_mode: str) -> str:
    # `upper-bound` モード（M=1 の事前上界）では `bound` 自体が実際の閾値を
    # 過大評価するため、`classify` の「全救済」確定を「部分／未確定」へ
    # 落とす（codex-review 指摘・PR #1421 P2-1。docstring 参照）。
    bound_is_upper = scale_mode == "upper-bound"

    lines: list[str] = []
    lines.append("# candle-parity-tolerance-baseline-impact 生出力（イシュー #1238）")
    lines.append("")
    lines.append(
        f"`--scale-mode {scale_mode}`。行数: {len(rows)}（実測値。"
        "`ParityPath` 別内訳は本文 doc 参照）。"
    )
    lines.append("")

    # 行別スケール（S_A・S_B）表。
    lines.append("## 行別 入力規模（M = S_A・S_B）")
    lines.append("")
    lines.append("| path | context | m | n | k | seed | dtype | S_A | S_B | M=S_A*S_B |")
    lines.append("|---|---|---:|---:|---:|---|---|---:|---:|---:|")
    scales: dict[int, tuple[float, float]] = {}
    for i, row in enumerate(rows):
        s_a, s_b = compute_scale(row.seed, row.m, row.k, row.n, row.is_f16, scale_mode)
        scales[i] = (s_a, s_b)
        dtype = "f16" if row.is_f16 else "f32"
        lines.append(
            f"| {row.path} | {_escape_md_cell(row.context)} | {row.m} | {row.n} | {row.k} | "
            f"0x{row.seed:X} | {dtype} | {s_a:.6f} | {s_b:.6f} | {s_a * s_b:.6f} |"
        )
    lines.append("")

    # 候補 A（A-1 系列。A-2 は c=1,eps=2^-23 の A-1 エントリと同一という
    # 前提を doc 側に明記）。
    a_specs = build_candidate_a_specs()
    lines.append("## 候補 A（スケール付き絶対誤差。M は上表の行別値を使用）")
    lines.append("")
    lines.append("各セルは `bound/分類`（分類: no-op／全救済／部分／未確定／分類不能）。")
    lines.append("")
    lines.append("| 候補 | " + " | ".join(f"row{i}" for i in range(len(rows))) + " |")
    lines.append("|---|" + "|".join(["---"] * len(rows)) + "|")
    for spec in a_specs:
        cells = []
        for i, row in enumerate(rows):
            s_a, s_b = scales[i]
            bound = spec.bound(row.k, s_a * s_b)
            cls = classify(bound, row.max_abs_diff_ceiling, bound_is_upper=bound_is_upper)
            cells.append(f"{bound:.3e}/{cls}")
        lines.append(f"| {_escape_md_cell(spec.name)} | " + " | ".join(cells) + " |")
    lines.append("")

    # A-4（参考値のみ）。`a4_bound` は `Σ|ab| <= K*S_A*S_B` という上界近似
    # を経由するため、`--scale-mode` に関わらず常に真の閾値を過大評価する
    # （codex-review 指摘・PR #1421 P2-1。`bound_is_upper=True` を無条件で
    # 渡し「全救済」の誤確定を防ぐ）。
    lines.append("## 候補 A-4（緩すぎる参考値。K*u*(K*S_A*S_B) 上界）")
    lines.append("")
    lines.append("| row | bound | class |")
    lines.append("|---|---:|---|")
    for i, row in enumerate(rows):
        s_a, s_b = scales[i]
        bound = a4_bound(row.k, s_a, s_b)
        cls = classify(bound, row.max_abs_diff_ceiling, bound_is_upper=True)
        lines.append(f"| {i} ({_escape_md_cell(row.context)}) | {bound:.3e} | {cls} |")
    lines.append("")

    # 候補 B-2。`b2_bound` も `Σ|ab| <= K*S_A*S_B` の上界近似を経由するため
    # A-4 と同じ理由で常に `bound_is_upper=True`。
    lines.append("## 候補 B-2（ULP ベース。t*ulp(K*S_A*S_B) 上界）")
    lines.append("")
    lines.append("| t | row | bound | class |")
    lines.append("|---:|---|---:|---|")
    for t in (1.0, 2.0, 4.0):
        for i, row in enumerate(rows):
            s_a, s_b = scales[i]
            bound = b2_bound(t, row.k, s_a, s_b)
            cls = classify(bound, row.max_abs_diff_ceiling, bound_is_upper=True)
            lines.append(f"| {t:.0f} | {i} ({_escape_md_cell(row.context)}) | {bound:.3e} | {cls} |")
    lines.append("")

    lines.append("## 候補 B-1・B-3（机上分類不能）")
    lines.append("")
    lines.append(
        f"B-1（`max_partial` 基準）・B-3（`exact` 基準）は全 {len(rows)} 行とも "
        f"「{UNCLASSIFIABLE_RUNTIME}」（部分和トレースまたは厳密真値が本データ"
        "セットには存在しない）。"
    )
    lines.append("")

    # 集計。
    lines.append("## 集計（候補ごとの分類件数）")
    lines.append("")
    lines.append("| 候補 | no-op | 全救済 | 部分／未確定 | 分類不能（ceiling未実測） |")
    lines.append("|---|---:|---:|---:|---:|")
    for spec in a_specs:
        counts = {NO_OP: 0, FULL_RESCUE: 0, PARTIAL: 0, UNCLASSIFIABLE_CEILING: 0}
        for i, row in enumerate(rows):
            s_a, s_b = scales[i]
            bound = spec.bound(row.k, s_a * s_b)
            counts[classify(bound, row.max_abs_diff_ceiling, bound_is_upper=bound_is_upper)] += 1
        lines.append(
            f"| {_escape_md_cell(spec.name)} | {counts[NO_OP]} | {counts[FULL_RESCUE]} | "
            f"{counts[PARTIAL]} | {counts[UNCLASSIFIABLE_CEILING]} |"
        )
    lines.append("")

    # 契約 5 項目への影響（構造的事実。全候補・全行共通で不変）。
    # provenance 未確定件数はパース済み行から実測する（codex-review 指摘・
    # PR #1421 P2-5: 旧実装は「1 行のみ true」と固定文言で出力していたが、
    # 現在の `BASELINES` は 45 行全てが `baseline_provenance_unconfirmed:
    # false`〈ceiling が `None` の行を含む〉であり、固定文言は実データと
    # 乖離した別状態の記述だった）。
    unconfirmed_count = sum(1 for row in rows if row.provenance_unconfirmed)
    lines.append("## 契約 5 項目への影響（構造的事実。OR 追加の単調性による）")
    lines.append("")
    lines.append("| 契約検査項目（`assert_no_parity_regression`） | 影響 |")
    lines.append("|---|---|")
    lines.append(
        f"| provenance 確定（`baseline_provenance_unconfirmed=true` の行数: "
        f"{unconfirmed_count}/{len(rows)}） | 無影響（候補追加とは独立） |"
    )
    lines.append("| `total` 完全一致 | 無影響（要素数は不変） |")
    lines.append(
        "| `fail_count <= baseline_fail_count` | 単調非増加のため恒常成立 "
        "（OR 追加は fail→pass のみ生じ、pass→fail は生じない） |"
    )
    lines.append(
        "| `mean_abs_diff <= ceiling` | 無影響（全セル集計は bit 同一。"
        "候補は判定式のみを変え計算対象の値は変えない） |"
    )
    lines.append(
        "| `max_abs_diff`/`max_rel_err <= ceiling`（`Some` のみ） | 無影響（同上） |"
    )
    lines.append("")

    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baselines",
        default=os.path.join(
            _HERE, "..", "..", "..", "crates", "backend-cuda", "tests", "common", "parity_baseline.rs"
        ),
        help="ParityBaseline 定義ソースへのパス",
    )
    parser.add_argument(
        "--scale-mode",
        choices=("exact", "upper-bound"),
        default="exact",
        help=(
            "M（入力規模）の導出方法。exact: PRNG を実際に走らせ厳密な "
            "max|A|・max|B| を求める（既定）。upper-bound: 入力範囲 [-1,1) "
            "からの事前上界 M=1 を使う（no-op 判定のみに使える。全救済判定を "
            "過大評価しうるため使用しない）"
        ),
    )
    args = parser.parse_args(argv)

    try:
        rows = parse_baselines(args.baselines)
    except BaselineParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # `render_markdown` は内部の `classify()` 経由で `threshold` 省略時に
    # `get_absolute_rescue_threshold()`（正本 `crates/backend-cpu/src/
    # parity.rs` を実ファイル読み取り）を呼ぶため、`parse_baselines` とは
    # 別に `BaselineParseError` を送出しうる（正本ファイル欠落・宣言スタイル
    # 変更）。`parse_baselines` 同様 fail-closed で捕捉する（PR #1421
    # レビュー指摘）。
    try:
        output = render_markdown(rows, args.scale_mode)
    except BaselineParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
