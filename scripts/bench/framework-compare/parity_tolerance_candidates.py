#!/usr/bin/env python3
"""GEMM parity fail 要素に対する tolerance 判定候補の机上評価（イシュー #1237）。

## 位置づけ

`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5 は N=2048 で candle 側
CUDA/CPU GEMM 出力が現行複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5
未満。`crates/backend-cpu/src/parity.rs::RELATIVE_TOLERANCE`/
`ABSOLUTE_RESCUE_THRESHOLD`）を各 2 要素ずつ外れ「判定不能」になることを
記録し、`parity_dump_truth.py`（イシュー #1184）はその fail 要素が
「累積過程で膨らんだ部分和がキャンセレーションで縮小した通常の丸め誤差」
であることを突合済みである。

本スクリプトは、そのダンプ実値のみを入力に「現行複合判定へ候補判定を
OR 追加した場合に fail 4 要素のうち何件が救済されるか」を、係数・閾値の
候補ごとに機械的に算出する（`docs/candle-parity-tolerance-candidates.md`
§4 の元データ）。**契約変更そのものはユーザー承認事項**（イシュー #1241）
であり、本スクリプトは承認判断に使う定量根拠の算出に閉じる。

## 評価モデルの限界（doc 側にも転記する）

- ダンプには現行判定で fail した要素しか含まれないため、各候補は
  「現行判定への OR 追加（単調緩和）」としてのみ評価できる
  （`fail 数（候補）= 入力要素数 − 救済件数`）。「候補で置き換える」判定
  （現行 pass 要素が新たに fail に転じうるか）は本ダンプからは評価不能
- 対象は K=2048・正方・入力 U[-0.5,0.5)・固定シードの 1 条件のみ。他形状・
  他シードへの外挿は行わない
- `exact`（有理数演算による厳密真値）を使う指標は診断専用であり、実行時に
  `assert_parity` 側が使えるのは `d = |ref − actual|` と入力・部分和から
  導ける量だけである（`applicable_at_runtime` 列で区別する）

## 呼び出し元

`docs/perf/logs/candle-parity-tolerance-candidates-1237/candidates-2048.md`
の生成に 1 回実行する。CI では単体テスト（`parity_tolerance_candidates_test.py`）
のみを実行し、実ダンプに対する計算は実行しない（実機ダンプが入力のため）。

使い方:
    python3 parity_tolerance_candidates.py --n 2048 \\
        --dump cuda=../../../docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cuda-2048.txt \\
        --dump cpu=../../../docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-cpu-2048.txt

標準ライブラリのみに依存する（`parity_dump_truth.py` と同方針）。
`parity_dump_truth.py` 自体は変更せず `importlib` で読み込んで再利用する。
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable

# `parity_dump_truth.py` を隣接ファイルとして importlib 経由で読み込む。
# `sys.modules` への登録を `exec_module` より前に行う必要がある
# （`from __future__ import annotations` + `@dataclass` の組み合わせで、
# dataclass デコレータがモジュール自身を `sys.modules` 経由で参照解決
# しようとする経路があり、未登録だと `AttributeError` になる。イシュー
# #1237 実装時に実機確認済みの制約）。
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRUTH_PATH = os.path.join(_HERE, "parity_dump_truth.py")
_TRUTH_SPEC = importlib.util.spec_from_file_location("parity_dump_truth", _TRUTH_PATH)
if _TRUTH_SPEC is None or _TRUTH_SPEC.loader is None:
    raise ImportError(f"parity_dump_truth.py を読み込めない: {_TRUTH_PATH}")
parity_dump_truth = importlib.util.module_from_spec(_TRUTH_SPEC)
sys.modules[_TRUTH_SPEC.name] = parity_dump_truth
_TRUTH_SPEC.loader.exec_module(parity_dump_truth)


# ラベル allowlist（`--dump LABEL=PATH` の LABEL）。表の見出し・ファイル名に
# 外部由来文字列を無検証で埋め込まないための入力検証（security.md A03）。
_LABEL_RE = re.compile(r"^[a-z0-9_-]+$")


@dataclass(frozen=True)
class ElementMetrics:
    """1 件の fail 要素について、候補判定に必要な量をまとめたもの。

    `d`（実行時に `assert_parity` 側が計算できる量）と、`exact`（厳密真値。
    診断専用で実行時には使えない）を明確に区別する。診断専用の量は候補
    評価そのものには使わず、doc 側の参考列としてのみ出力する。
    """

    idx: int
    row: int
    col: int
    d: float  # |ref - actual|（実行時適用可能）
    d_ex: float  # |actual - exact|（診断専用）
    d_ref_ex: float  # |ref - exact|（診断専用）
    max_partial: float  # f32 FMA 逐次累積の部分和絶対値の最大（実行時にはトレースが要る）
    max_ab: float  # 実 max_k|a_k| * max_k|b_k|（実行時 1 パスで計算可能）
    sum_abs_ab: float  # Σ_k|a_k・b_k|（実行時 1 パスで計算可能）
    exact: float  # 厳密真値（診断専用）
    fma_bit_match: bool


def compute_metrics(rows: list["parity_dump_truth.ParityDumpRow"], n: int) -> list[ElementMetrics]:
    """`ParityDumpRow` 列から `ElementMetrics` 列を算出する。

    行・列の厳密抽出は `parity_dump_truth.extract_rows_exact`/
    `extract_cols_exact`（xorshift 状態のみ進めて不要位置は `Fraction` 化
    しない設計）をそのまま再利用する。
    """
    needed_rows = {r.row for r in rows}
    needed_cols = {r.col for r in rows}
    a_rows = parity_dump_truth.extract_rows_exact(parity_dump_truth.SEED_A, n, needed_rows)
    b_cols = parity_dump_truth.extract_cols_exact(parity_dump_truth.SEED_B, n, needed_cols)

    out: list[ElementMetrics] = []
    for rec in rows:
        a_row = a_rows[rec.row]
        b_col = b_cols[rec.col]

        exact = sum((a * b for a, b in zip(a_row, b_col)), Fraction(0))
        exact_f = float(exact)

        fma_f32, partials = parity_dump_truth.fma_sequential_f32_exact(a_row, b_col)
        fma_bits = int.from_bytes(
            __import__("struct").pack("<f", fma_f32), "little"
        )
        fma_bit_match = fma_bits == rec.ref_bits

        max_partial = max((abs(float(p)) for p in partials), default=0.0)

        a_floats = [float(a) for a in a_row]
        b_floats = [float(b) for b in b_col]
        max_ab = (max(abs(v) for v in a_floats) if a_floats else 0.0) * (
            max(abs(v) for v in b_floats) if b_floats else 0.0
        )
        sum_abs_ab = sum(abs(a * b) for a, b in zip(a_floats, b_floats))

        ref_f = rec.ref_f32
        actual_f = rec.actual_f32

        out.append(
            ElementMetrics(
                idx=rec.idx,
                row=rec.row,
                col=rec.col,
                d=abs(ref_f - actual_f),
                d_ex=abs(actual_f - exact_f),
                d_ref_ex=abs(ref_f - exact_f),
                max_partial=max_partial,
                max_ab=max_ab,
                sum_abs_ab=sum_abs_ab,
                exact=exact_f,
                fma_bit_match=fma_bit_match,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 候補判定の定義（計画 §4 の式）。
# ---------------------------------------------------------------------------

EPS_F32 = 2.0**-23  # machine epsilon
U_F32 = 2.0**-24  # unit roundoff


@dataclass(frozen=True)
class CandidateA:
    """スケール付き絶対誤差候補。`pass 条件: d <= bound(metric, k)`。"""

    name: str
    eps_label: str
    eps_value: float
    c: float
    k_mode: str  # "K" | "sqrtK"
    m_mode: str  # "fixed0.25" | "actual"

    def bound(self, metric: ElementMetrics, k: int) -> float:
        kscale = float(k) if self.k_mode == "K" else math.sqrt(k)
        m = 0.25 if self.m_mode == "fixed0.25" else metric.max_ab
        return self.c * self.eps_value * kscale * m


@dataclass(frozen=True)
class CandidateA4:
    """A-4: 古典的前進誤差上界 `bound = K * u * Σ|a_k・b_k|`（緩すぎる参考値）。"""

    name: str = "A-4 (K*u*sum|ab|)"

    def bound(self, metric: ElementMetrics, k: int) -> float:
        return float(k) * U_F32 * metric.sum_abs_ab


def build_candidates_a() -> list[CandidateA]:
    cands: list[CandidateA] = []
    for eps_label, eps_value in (("2^-23", 2.0**-23), ("2^-24", 2.0**-24)):
        for c in (0.125, 0.25, 0.5, 1.0, 2.0):
            cands.append(
                CandidateA(
                    name=f"A-1 c={c} eps={eps_label} K*0.25",
                    eps_label=eps_label,
                    eps_value=eps_value,
                    c=c,
                    k_mode="K",
                    m_mode="fixed0.25",
                )
            )
    # A-2: M を実 max|a|*max|b| に置換（代表として eps=2^-23, c=1 のみ）。
    cands.append(
        CandidateA(
            name="A-2 c=1 eps=2^-23 K*max_ab",
            eps_label="2^-23",
            eps_value=2.0**-23,
            c=1.0,
            k_mode="K",
            m_mode="actual",
        )
    )
    # A-3: √K スケール（救済しないことの記録用）。
    for eps_label, eps_value in (("2^-23", 2.0**-23),):
        for c in (1.0,):
            cands.append(
                CandidateA(
                    name=f"A-3 c={c} eps={eps_label} sqrtK*0.25",
                    eps_label=eps_label,
                    eps_value=eps_value,
                    c=c,
                    k_mode="sqrtK",
                    m_mode="fixed0.25",
                )
            )
    return cands


@dataclass(frozen=True)
class CandidateB:
    """ULP ベース候補。`pass 条件: err <= t * ulp(base(metric))`。"""

    name: str
    base_mode: str  # "max_partial" | "sum_abs_ab" | "exact"
    t: float

    def base_value(self, metric: ElementMetrics) -> float:
        if self.base_mode == "max_partial":
            return metric.max_partial
        if self.base_mode == "sum_abs_ab":
            return metric.sum_abs_ab
        if self.base_mode == "exact":
            return metric.exact
        raise ValueError(f"未知の base_mode: {self.base_mode}")

    def bound(self, metric: ElementMetrics) -> float:
        return self.t * parity_dump_truth.ulp_f32(self.base_value(metric))


def build_candidates_b(k: int) -> list[CandidateB]:
    cands: list[CandidateB] = []
    for t in (8, 16, 32, 48, 64):
        cands.append(CandidateB(name=f"B-1 t={t} max_partial", base_mode="max_partial", t=float(t)))
    for c in (1, 2):
        t = c * math.sqrt(k)
        cands.append(
            CandidateB(name=f"B-1 t={c}*sqrtK≈{t:.1f} max_partial", base_mode="max_partial", t=t)
        )
    for t in (1, 2, 4):
        cands.append(CandidateB(name=f"B-2 t={t} sum_abs_ab", base_mode="sum_abs_ab", t=float(t)))
    for t in (1e3, 1e4, 1e5):
        cands.append(CandidateB(name=f"B-3 t={t:.0e} exact", base_mode="exact", t=t))
    return cands


# ---------------------------------------------------------------------------
# 評価・Markdown 出力
# ---------------------------------------------------------------------------


def evaluate_a(metrics: list[ElementMetrics], candidate: CandidateA, k: int) -> int:
    """`d`（実行時適用可能な唯一の量）で救済件数を数える。"""
    return sum(1 for m in metrics if m.d <= candidate.bound(m, k))


def evaluate_a4(metrics: list[ElementMetrics], candidate: CandidateA4, k: int) -> int:
    return sum(1 for m in metrics if m.d <= candidate.bound(m, k))


def evaluate_b(metrics: list[ElementMetrics], candidate: CandidateB, use: str) -> int:
    """`use`: "d"（実行時適用可能）または "d_ex"（診断専用）。"""
    bound_fn = candidate.bound
    if use == "d":
        return sum(1 for m in metrics if m.d <= bound_fn(m))
    return sum(1 for m in metrics if m.d_ex <= bound_fn(m))


def render_markdown(
    metrics_by_label: dict[str, list[ElementMetrics]],
    k: int,
) -> str:
    lines: list[str] = []
    labels = list(metrics_by_label.keys())

    lines.append(f"# tolerance 判定候補 机上評価（K=N={k}）")
    lines.append("")
    lines.append(
        "評価モデルの限界: 各候補は現行複合判定への OR 追加としてのみ評価できる "
        "（`fail 数（候補）= 入力要素数 − 救済件数`。現行 pass 要素が新たに fail "
        "に転じうるかは本ダンプからは評価不能）。対象は K=2048・正方・入力 "
        "U[-0.5,0.5)・固定シードの 1 条件のみ。詳細はスクリプト冒頭 docstring を "
        "参照。"
    )
    lines.append("")

    # 要素別メトリクス表。
    lines.append("## 要素別メトリクス")
    lines.append("")
    lines.append(
        "| device | idx | row | col | d=\\|ref-actual\\| | d_ex=\\|actual-exact\\|（診断） | "
        "d_ref_ex=\\|ref-exact\\|（診断） | max\\|partial\\| | max_ab（実測） | Σ\\|ab\\| | fma_bit_match |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        for m in metrics_by_label[label]:
            lines.append(
                f"| {label} | {m.idx} | {m.row} | {m.col} | {m.d:.3e} | {m.d_ex:.3e} | "
                f"{m.d_ref_ex:.3e} | {m.max_partial:.3e} | {m.max_ab:.4f} | {m.sum_abs_ab:.3f} | "
                f"{m.fma_bit_match} |"
            )
    lines.append("")

    # 候補 A 表。
    lines.append("## 候補 A（スケール付き絶対誤差。`d <= bound` で救済）")
    lines.append("")
    header_cols = " | ".join(f"{label} fail" for label in labels)
    lines.append(f"| 候補 | bound(K={k}) | {header_cols} | 実行時適用可否 |")
    lines.append(f"|---|---:|{'|'.join(['---:'] * len(labels))}|---|")
    for cand in build_candidates_a():
        row_cells = []
        for label in labels:
            metrics = metrics_by_label[label]
            rescued = evaluate_a(metrics, cand, k)
            fail = len(metrics) - rescued
            row_cells.append(f"{fail}/{len(metrics)}")
        # bound は要素依存（A-2 の m_mode="actual"）の場合があるため代表値として
        # 各 label の最初の要素で bound を1つ例示する（表の可読性目的。厳密な
        # per-element bound は「要素別メトリクス」表の max_ab から手計算可能）。
        sample_metric = next(iter(metrics_by_label.values()))[0]
        bound_repr = f"{cand.bound(sample_metric, k):.3e}"
        lines.append(
            f"| {cand.name} | {bound_repr} | " + " | ".join(row_cells) + " | 適用可能 |"
        )
    a4 = CandidateA4()
    row_cells = []
    for label in labels:
        metrics = metrics_by_label[label]
        rescued = evaluate_a4(metrics, a4, k)
        fail = len(metrics) - rescued
        row_cells.append(f"{fail}/{len(metrics)}")
    sample_metric = next(iter(metrics_by_label.values()))[0]
    lines.append(
        f"| {a4.name} | {a4.bound(sample_metric, k):.3e} | " + " | ".join(row_cells) + " | 適用可能（緩すぎる参考値） |"
    )
    lines.append("")

    # K スイープ表（A-1 は K のみの関数）。
    lines.append("## K スイープ（A-1 の緩和上限。現行絶対閾値 1e-5 との比）")
    lines.append("")
    lines.append("| N=K | c=1 eps=2^-23 bound | 1e-5比 | c=1 eps=2^-24 bound | 1e-5比 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for kk in (512, 1024, 2048, 4096):
        b23 = 1.0 * (2.0**-23) * kk * 0.25
        b24 = 1.0 * (2.0**-24) * kk * 0.25
        lines.append(f"| {kk} | {b23:.3e} | {b23 / 1e-5:.2f}x | {b24:.3e} | {b24 / 1e-5:.2f}x |")
    lines.append("")

    # 候補 B 表。
    lines.append("## 候補 B（ULP ベース。`err <= t * ulp(base)` で救済）")
    lines.append("")
    header_cols_d = " | ".join(f"{label} fail(d基準)" for label in labels)
    header_cols_dex = " | ".join(f"{label} fail(d_ex基準,診断)" for label in labels)
    lines.append(f"| 候補 | {header_cols_d} | {header_cols_dex} | 実行時適用可否 | Phase 2 実装可否 |")
    lines.append(f"|---|{'|'.join(['---:'] * len(labels))}|{'|'.join(['---:'] * len(labels))}|---|---|")
    runtime_note = {
        "max_partial": ("実行時は部分和トレースが要る（要拡張）", "B-1 は部分和トレース追加実装が要る"),
        "sum_abs_ab": ("実行時 1 パス追加で計算可能", "B-2 は 1 パス追加で実装可能"),
        "exact": ("実行時には使えない（診断専用）", "B-3 は実装不可（真値は実行時に得られない）"),
    }
    for cand in build_candidates_b(k):
        cells_d = []
        cells_dex = []
        for label in labels:
            metrics = metrics_by_label[label]
            rescued_d = evaluate_b(metrics, cand, "d")
            rescued_dex = evaluate_b(metrics, cand, "d_ex")
            cells_d.append(f"{len(metrics) - rescued_d}/{len(metrics)}")
            cells_dex.append(f"{len(metrics) - rescued_dex}/{len(metrics)}")
        applicable, phase2 = runtime_note[cand.base_mode]
        lines.append(
            f"| {cand.name} | " + " | ".join(cells_d) + " | " + " | ".join(cells_dex)
            + f" | {applicable} | {phase2} |"
        )
    lines.append("")

    return "\n".join(lines)


def _parse_dump_arg(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--dump は LABEL=PATH 形式で指定する（受領: {value!r}）")
    label, path = value.split("=", 1)
    if not _LABEL_RE.match(label):
        raise argparse.ArgumentTypeError(
            f"--dump のラベルは [a-z0-9_-]+ のみ許可する（受領: {label!r}）"
        )
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"--dump のパスが存在しないか通常ファイルでない: {path!r}")
    return label, path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="GEMM の正方行列一辺長")
    parser.add_argument(
        "--dump",
        action="append",
        type=_parse_dump_arg,
        required=True,
        help="LABEL=PATH 形式で PARITY_DUMP ファイルを指定する（複数可）",
    )
    args = parser.parse_args(argv)

    if args.n < 1 or args.n > 8192:
        print(f"ERROR: --n は 1..8192 の範囲で指定する（受領: {args.n}）", file=sys.stderr)
        return 2

    n = args.n
    metrics_by_label: dict[str, list[ElementMetrics]] = {}
    had_error = False

    for label, path in args.dump:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        error_count = [0]
        unique_by_idx: dict[int, "parity_dump_truth.ParityDumpRow"] = {}
        for r in parity_dump_truth.parse_dump_lines(lines, n, error_count=error_count):
            if r.idx in unique_by_idx:
                continue
            unique_by_idx[r.idx] = r
        if error_count[0] > 0:
            print(f"ERROR: label={label} のダンプに不正行 {error_count[0]} 件を検出した", file=sys.stderr)
            had_error = True
        if not unique_by_idx:
            print(f"ERROR: label={label} のダンプから有効な fail 要素を抽出できなかった", file=sys.stderr)
            had_error = True
            continue
        rows = [unique_by_idx[idx] for idx in sorted(unique_by_idx)]
        metrics = compute_metrics(rows, n)
        for m in metrics:
            if not m.fma_bit_match:
                print(
                    f"ERROR: label={label} idx={m.idx} の f32 FMA 逐次再現が ref_bits と bit 不一致",
                    file=sys.stderr,
                )
                had_error = True
        metrics_by_label[label] = metrics

    if had_error or not metrics_by_label:
        print("ERROR: 入力ダンプの検証に失敗したため計算結果は出力しない", file=sys.stderr)
        return 1

    print(render_markdown(metrics_by_label, n))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
