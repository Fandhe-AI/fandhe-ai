#!/usr/bin/env python3
"""GEMM parity ダンプ（`PARITY_DUMP` 行）の fail 要素を厳密真値と突合する（イシュー #1184）。

## 位置づけ

`FRAMEWORK_COMPARE_PARITY_DUMP=1` で bench-candle/bench-fandhe/bench-burn の
stderr に出る `PARITY_DUMP idx=... ref_bits=... actual_bits=...` 行
（`bench-common/src/parity.rs::dump_parity_failures`。イシュー #1183）は
「参照実装（f32 FMA・k 昇順逐次）の値」と「フレームワーク実測値」しか
教えてくれず、**どちら側が厳密な数学的真値から離れているか**は分からない。
本スクリプトは `bench-common::Xorshift64Star`／`fill_vec`（`lib.rs`）を Python
で厳密再現して同じ A・B 入力行列を再構成し、fail 要素 (row, col) の
`Σ_k A[row,k]·B[k,col]` を有理数演算（`fractions.Fraction`）で誤差ゼロに
計算する。あわせて f32 FMA 逐次累積（k 昇順）を 1 ステップずつ厳密丸めで
再現し、`ref_bits` と bit 一致することを確認する（一致すれば「RNG 再現が
正しい」かつ「参照実装が契約どおり動いている」の直接証拠になる）。

## 入力データの誤差ゼロ表現

`fill_vec` の 1 要素は `((x >> 40) as f32) / 2^24 - 0.5` で、`x >> 40` は
24 bit 整数 `k`（0..2^24）。`k / 2^24` は 2 のべき分母の二進有理数なので
f32 として厳密表現でき、`Fraction(k, 2**24) - Fraction(1, 2)` で誤差なく
保持できる（浮動小数点を経由しない）。

## 呼び出し元

`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.3 の追記作業（本イシュー
#1184）で GB10 実機ダンプ（`docs/perf/logs/cuda-gemm-candle-parity-1184/`）を
突合するために 1 回実行する。CI では実行しない（実機ダンプが入力のため）。

使い方:
    python3 parity_dump_truth.py --n 2048 < parity-dump-cuda-2048.txt

標準ライブラリのみに依存する（`.claude/rules/deps-policy.md` の対象外だが
依存ゼロを維持する）。
"""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Iterator

# `dump_parity_failures` の出力書式（`parity.rs` の `writeln!` フォーマット
# 文字列）と 1 対 1 対応する厳密パーサ。想定外の行は無視ではなく警告して
# 打ち切る（`.claude/rules/security.md` A03: 外部入力を検証してから使う）。
_LINE_RE = re.compile(
    r"^PARITY_DUMP call=(?P<call>\d+) n=(?P<n>\d+) idx=(?P<idx>\d+) "
    r"row=(?P<row>\d+) col=(?P<col>\d+) "
    r"ref=(?P<ref>[^ ]+) ref_bits=0x(?P<ref_bits>[0-9a-fA-F]{8}) "
    r"actual=(?P<actual>[^ ]+) actual_bits=0x(?P<actual_bits>[0-9a-fA-F]{8}) "
    r"abs=(?P<abs>[^ ]+) rel=(?P<rel>[^ ]+)$"
)

# xorshift64* (Vigna 2016) の乗数。`bench-common/src/lib.rs::Xorshift64Star`
# と同一。64 bit wrapping で演算する。
_MASK64 = (1 << 64) - 1
_MULT = 0x2545_F491_4F6C_DD1D
_ZERO_SEED_REPLACEMENT = 0x9E37_79B9_7F4A_7C15

# `bench-common/src/lib.rs` の固定シード（GEMM 入力生成用）。
SEED_A = 0xA11CE
SEED_B = 0xB0B


@dataclass
class ParityDumpRow:
    """1 件の `PARITY_DUMP` 行（厳密パース済み）。"""

    call: int
    n: int
    idx: int
    row: int
    col: int
    ref_bits: int
    actual_bits: int
    dump_abs: float
    dump_rel: float

    @property
    def ref_f32(self) -> float:
        return struct.unpack("<f", struct.pack("<I", self.ref_bits))[0]

    @property
    def actual_f32(self) -> float:
        return struct.unpack("<f", struct.pack("<I", self.actual_bits))[0]


def parse_dump_lines(lines: Iterable[str], expected_n: int) -> Iterator[ParityDumpRow]:
    """`PARITY_DUMP` 行のみを厳密パースして yield する。

    `PARITY_DUMP_SUMMARY` 行・その他の stderr 出力は無視する（サマリは
    件数の突合に使うだけで真値計算には不要なため、本関数の対象外とし
    呼び出し元が別途 grep する）。`row*n+col == idx` と `n*n` 範囲内を
    検証し、想定外の行は標準エラーへ警告してスキップする（A03: 破損
    入力を無条件に信用しない）。
    """
    for lineno, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        if not line.startswith("PARITY_DUMP "):
            continue
        m = _LINE_RE.match(line)
        if m is None:
            print(f"WARN: line {lineno} は PARITY_DUMP 書式に一致せず無視: {line!r}", file=sys.stderr)
            continue
        n = int(m.group("n"))
        if n != expected_n:
            print(
                f"WARN: line {lineno} の n={n} が --n {expected_n} と不一致のため無視",
                file=sys.stderr,
            )
            continue
        idx = int(m.group("idx"))
        row = int(m.group("row"))
        col = int(m.group("col"))
        if row * n + col != idx or idx >= n * n:
            print(
                f"WARN: line {lineno} の row/col/idx が整合しない（row*n+col={row * n + col}, idx={idx}, n*n={n * n}）ため無視",
                file=sys.stderr,
            )
            continue
        yield ParityDumpRow(
            call=int(m.group("call")),
            n=n,
            idx=idx,
            row=row,
            col=col,
            ref_bits=int(m.group("ref_bits"), 16),
            actual_bits=int(m.group("actual_bits"), 16),
            dump_abs=float(m.group("abs")),
            dump_rel=float(m.group("rel")),
        )


class Xorshift64StarExact:
    """`bench-common::Xorshift64Star` の Python 厳密再現（整数演算のみ）。"""

    def __init__(self, seed: int) -> None:
        self.state = _ZERO_SEED_REPLACEMENT if seed == 0 else (seed & _MASK64)

    def next_u64(self) -> int:
        x = self.state
        x ^= (x >> 12)
        x &= _MASK64
        x ^= (x << 25) & _MASK64
        x &= _MASK64
        x ^= (x >> 27)
        x &= _MASK64
        self.state = x
        return (x * _MULT) & _MASK64

    def next_element_exact(self) -> Fraction:
        """`fill_vec` の 1 要素を厳密有理数として返す（f32 化を経由しない）。

        Rust 側は `((x >> 40) as f32) / 2^24 - 0.5`。`x >> 40` は 0..2^24 の
        整数 `k` で、`k / 2^24` は f32 として厳密表現可能なため、
        `Fraction(k, 2**24) - Fraction(1, 2)` で誤差ゼロに一致する。
        """
        x = self.next_u64()
        k = x >> 40
        return Fraction(k, 1 << 24) - Fraction(1, 2)

    def fill_vec_exact(self, n: int) -> list[Fraction]:
        return [self.next_element_exact() for _ in range(n)]


def round_half_even_to_f32(value: Fraction) -> float:
    """任意精度有理数 `value` を IEEE754 binary32（round-half-even）へ直接丸める。

    `Fraction -> float(f64) -> f32` の 2 段丸めは f64 の丸めで情報を失い
    二重丸め誤差を生みうるため避ける（本関数は分子・分母から直接 f32 の
    仮数・指数を決定し、一度だけ丸める）。ゼロ・非正規化数・通常数を扱う。
    無限大・NaN は本用途の入力（GEMM 部分和）では発生しない想定のため
    未対応とし、範囲外は `OverflowError` とする（fail-closed）。
    """
    if value == 0:
        return 0.0
    sign = -1.0 if value < 0 else 1.0
    mag = abs(value)

    # 2 進指数 e を求める: 2^e <= mag < 2^(e+1)
    e = mag.numerator.bit_length() - mag.denominator.bit_length()
    # bit_length の差は概算なので厳密化する。
    while Fraction(2) ** e > mag:
        e -= 1
    while Fraction(2) ** (e + 1) <= mag:
        e += 1

    # binary32: 指数バイアス 127、仮数 23 bit、正規化範囲 e in [-126, 127]。
    if e < -126:
        # 非正規化数域: 固定スケール 2^-149 の整数倍として丸める。
        scale = Fraction(2) ** -149
        exp_used = -149
    elif e > 127:
        raise OverflowError(f"f32 表現域を超える値: {value}")
    else:
        scale = Fraction(2) ** (e - 23)
        exp_used = e - 23

    ratio = mag / scale  # 丸め対象の整数域比率（round-half-even する）。
    q, r = divmod(ratio.numerator, ratio.denominator)
    twice_r = 2 * r
    if twice_r > ratio.denominator or (twice_r == ratio.denominator and q % 2 == 1):
        q += 1

    result = sign * q * (2.0**exp_used)
    return float(result)


def fma_sequential_f32_exact(a_row: list[Fraction], b_col: list[Fraction]) -> tuple[float, list[Fraction]]:
    """f32 FMA 逐次累積（k 昇順）を厳密丸めで再現する。

    `backend-cpu::parity::matmul_reference_fma` / `GemmReference` と同じ
    契約（各ステップで `acc = a*b + acc` を f32 丸め 1 回）を、各ステップの
    `a*b + acc` を有理数で厳密計算してから `round_half_even_to_f32` で
    1 回だけ丸めることで再現する（f32 演算を模倣する最も直接的な方法。
    `a`/`b` はいずれも `fill_vec_exact` の厳密表現なので `a*b` 自体には
    丸め誤差が入らない）。

    戻り値は最終 f32 値と、各ステップ後の厳密部分和（Fraction）の列
    （後段のキャンセレーション解析で使う）。
    """
    acc = Fraction(0)
    partials: list[Fraction] = []
    acc_f32 = 0.0
    for a, b in zip(a_row, b_col):
        acc = Fraction(acc_f32) * 1  # 直前ステップの f32 値を厳密有理数として引き継ぐ
        acc = a * b + acc
        acc_f32 = round_half_even_to_f32(acc)
        partials.append(Fraction(acc_f32))
    return acc_f32, partials


def ulp_f32(x: float) -> float:
    """`|x|` の f32 ulp（1 単位最終桁）。0 の場合は最小非正規化数を返す。"""
    ax = abs(x)
    if ax == 0.0:
        return 2.0**-149
    e = math.floor(math.log2(ax))
    e = max(e, -126)  # 非正規化数域は指数を -126 に固定
    return 2.0 ** (e - 23)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, help="GEMM の正方行列一辺長（--size と同じ値）")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=64,
        help="解析する PARITY_DUMP 行数の上限（同一 idx の重複行は最初の 1 件のみ解析するため通常は既定で十分）",
    )
    args = parser.parse_args(argv)

    if args.n < 1 or args.n > 8192:
        print(f"ERROR: --n は 1..8192 の範囲で指定する（受領: {args.n}）", file=sys.stderr)
        return 2

    n = args.n
    lines = sys.stdin.readlines()
    rows = list(parse_dump_lines(lines, n))
    if not rows:
        print("ERROR: PARITY_DUMP 行が 1 件も見つからなかった", file=sys.stderr)
        return 2
    if len(rows) > args.max_rows:
        print(
            f"WARN: PARITY_DUMP 行が {len(rows)} 件（--max-rows {args.max_rows} 超）。"
            "先頭 --max-rows 件のみでユニーク idx を抽出する",
            file=sys.stderr,
        )
        rows = rows[: args.max_rows]

    # 同一 idx が複数 call にわたって重複するため、ユニークな idx 集合を
    # 抽出する（値は決定的想定なので最初の出現を代表値として使う。
    # `PARITY_DUMP_SUMMARY` 側の call 数・fail_count は呼び出し元が別途
    # 突合する）。
    unique_by_idx: dict[int, ParityDumpRow] = {}
    for r in rows:
        unique_by_idx.setdefault(r.idx, r)

    print(f"# n={n} ユニーク fail idx 件数={len(unique_by_idx)} (解析対象 PARITY_DUMP 行 {len(rows)} 件)")

    # A・B を厳密再構成（fill_vec_exact は要素ごとの厳密有理数）。
    a_exact = Xorshift64StarExact(SEED_A).fill_vec_exact(n * n)
    b_exact = Xorshift64StarExact(SEED_B).fill_vec_exact(n * n)

    header = (
        f"{'idx':>10} {'row':>6} {'col':>6} {'exact':>14} {'f64_seq':>14} "
        f"{'ref':>14} {'actual':>14} {'|ref-exact|':>12} {'|actual-exact|':>14} "
        f"{'|ref-actual|':>12} {'max|partial|':>12} {'sqrtK*ulp':>12} {'fma_bit_match':>13}"
    )
    print(header)

    for idx in sorted(unique_by_idx):
        rec = unique_by_idx[idx]
        row, col = rec.row, rec.col

        a_row = a_exact[row * n : row * n + n]
        b_col = [b_exact[k * n + col] for k in range(n)]

        # 厳密真値（有理数の完全和。丸め誤差ゼロ）。
        exact = sum((a * b for a, b in zip(a_row, b_col)), Fraction(0))
        exact_f = float(exact)

        # f64 逐次和（k 昇順。Python の float は IEEE754 binary64）。
        acc64 = 0.0
        for a, b in zip(a_row, b_col):
            acc64 += float(a) * float(b)

        # f32 FMA 逐次累積の厳密再現（bit 一致検証つき）。
        fma_f32, partials = fma_sequential_f32_exact(a_row, b_col)
        fma_bits = struct.unpack("<I", struct.pack("<f", fma_f32))[0]
        fma_bit_match = fma_bits == rec.ref_bits

        ref_f = rec.ref_f32
        actual_f = rec.actual_f32

        err_ref = abs(ref_f - exact_f)
        err_actual = abs(actual_f - exact_f)
        err_ref_actual = abs(ref_f - actual_f)

        max_partial = max((abs(float(p)) for p in partials), default=0.0)
        sqrtk_ulp = (n**0.5) * ulp_f32(max_partial)

        print(
            f"{idx:>10} {row:>6} {col:>6} {exact_f:>14.6e} {acc64:>14.6e} "
            f"{ref_f:>14.6e} {actual_f:>14.6e} {err_ref:>12.3e} {err_actual:>14.3e} "
            f"{err_ref_actual:>12.3e} {max_partial:>12.3e} {sqrtk_ulp:>12.3e} {str(fma_bit_match):>13}"
        )

        if not fma_bit_match:
            print(
                f"  WARN: idx={idx} の f32 FMA 逐次再現が ref_bits と bit 不一致 "
                f"(再現 0x{fma_bits:08x} vs ダンプ 0x{rec.ref_bits:08x})。"
                "RNG 再現または丸め実装を見直すこと（記録のみ・処理は継続）。",
                file=sys.stderr,
            )

        # 自己整合チェック: |ref-actual| がダンプの abs= と一致するか（誤差 1e-9 以内)。
        if abs(err_ref_actual - rec.dump_abs) > 1e-9:
            print(
                f"  WARN: idx={idx} の |ref-actual|={err_ref_actual:.6e} がダンプの abs={rec.dump_abs:.6e} と不一致",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
