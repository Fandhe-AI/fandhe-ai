#!/usr/bin/env python3
"""`truth-N.txt`（`parity_dump_truth.py` 出力）を N ごとに集計して `truth-summary.md` を生成する（イシュー #1984）。

## 位置づけ

`docs/perf/logs/parity-burn-tf32-truth-1984/` に収納した burn cuda（TF32 経路）
GEMM の parity fail 要素ダンプ（`parity-dump-burn-cuda-N.txt`。call=39 の行のみ）を
`scripts/bench/framework-compare/parity_dump_truth.py --n N` で厳密真値と突合した
per-element 表（`truth-N.txt`。`.gz` 圧縮済みでも可）から、RULE.txt の記録事項

- (a) 件数・row 範囲（ダンプ件数と `summary-N.txt` の 40 call `fail_count` 一致）
- (b) `|actual−exact|` と `|ref−exact|` の min／中央値／max・真値に近い側の件数
- (c) `fma_bit_match`（参照実装の f32 FMA 逐次累積 bit 再現）が True の割合
- (d) 線形 K 形 `bound_K = 0.5・2^-11・K・0.25 = K/16384` と `max|ref−actual|` の比較
- (e) √K 形 `bound_√K = √K/16384` で救済される件数（`|ref−actual| <= bound_√K`）

を 1 つの Markdown 表にまとめる。(d)／(e) の `|ref−actual|` は `truth-N.txt` の 3 桁表示
ではなく `parity-dump-burn-cuda-N.txt` の `abs=` 全精度値を idx で結合して使う。判定・tolerance・判定式は変更しない（診断専用の
机上集計。採否判定を書かない）。(d)／(e) の bound は `parity_tolerance_candidates.py
--extra-eps 'tf32u2^-11=0.00048828125' --extra-c 0.5` の `EXTRA c=0.5 eps=tf32u2^-11
{K|sqrtK}*0.25` 行と同じ式であり、(e) の件数は同表の「総数 − fail 数」と一致するはず
（README で突合する）。

## 使い方

    cd docs/perf/logs/parity-burn-tf32-truth-1984
    python3 summarize_truth.py > truth-summary.md

Python 標準ライブラリのみに依存する（`gzip` は `.gz` 入力の読み出しにのみ使う）。
"""

from __future__ import annotations

import gzip
import io
import math
import os
import re
import statistics
import sys

SIZES = (256, 512, 1024, 2048, 4096)
HERE = os.path.dirname(os.path.abspath(__file__))

# RULE.txt の (c): u=2^-11（TF32 仮数部 10 bit の unit roundoff）・c=0.5・M=fixed0.25
U_TF32 = 2.0**-11
C = 0.5
M_FIXED = 0.25

_HEADER_RE = re.compile(
    r"^# n=(\d+) ユニーク fail idx 件数=(\d+) \(解析対象 PARITY_DUMP 行 (\d+) 件\)"
)
_SUMMARY_RE = re.compile(
    r"^PARITY_DUMP_SUMMARY call=(\d+) n=(\d+) fail_count=(\d+) dumped=(\d+) truncated=(\w+)"
)
_DUMP_RE = re.compile(r"^PARITY_DUMP call=(\d+) n=(\d+) idx=(\d+) .* abs=([0-9.eE+-]+) ")


def read_dump_abs(n: int) -> dict[int, float]:
    """`parity-dump-burn-cuda-N.txt`（call=39 の行のみ）から idx → `abs=|ref−actual|`（全精度）を返す。

    `truth-N.txt` の `|ref-actual|` 列は 3 桁有効数字の表示のため、bound 近傍の要素を
    (d)／(e) の `<=` 判定へ掛けると丸め由来の誤分類が起きる（N=512 で 2 件確認）。
    (d)／(e) は本関数の全精度値を使い、`parity_tolerance_candidates.py` と同じ入力で判定する。
    """
    out: dict[int, float] = {}
    with open_text(os.path.join(HERE, f"parity-dump-burn-cuda-{n}.txt")) as f:
        for line in f:
            m = _DUMP_RE.match(line)
            if not m:
                continue
            if int(m.group(2)) != n or m.group(1) != "39":
                raise ValueError(f"dump {n}: call／n が想定外: {line.rstrip()[:80]}")
            idx = int(m.group(3))
            if idx in out:
                raise ValueError(f"dump {n}: idx={idx} が重複")
            out[idx] = float(m.group(4))
    return out


def open_text(path: str) -> io.TextIOBase:
    """`path` または `path + ".gz"` のどちらか存在する方をテキストで開く。"""
    if os.path.exists(path):
        return open(path, encoding="utf-8")
    gz = path + ".gz"
    if os.path.exists(gz):
        return io.TextIOWrapper(gzip.open(gz, "rb"), encoding="utf-8")
    raise FileNotFoundError(f"{path} も {gz} も存在しない")


def read_summary(n: int) -> tuple[list[int], list[int], set[str]]:
    """`summary-N.txt` から call ごとの fail_count／dumped／truncated を返す。"""
    fails: list[int] = []
    dumped: list[int] = []
    truncated: set[str] = set()
    with open(os.path.join(HERE, f"summary-{n}.txt"), encoding="utf-8") as f:
        for line in f:
            m = _SUMMARY_RE.match(line.strip())
            if not m:
                continue
            if int(m.group(2)) != n:
                raise ValueError(f"summary-{n}.txt に n={m.group(2)} の行が混在")
            fails.append(int(m.group(3)))
            dumped.append(int(m.group(4)))
            truncated.add(m.group(5))
    return fails, dumped, truncated


def summarize(n: int) -> dict:
    """`truth-N.txt` を読み、(a)〜(e) の集計値を dict で返す。"""
    path = os.path.join(HERE, f"truth-{n}.txt")
    rows: list[int] = []
    idxs: list[int] = []
    d_actual: list[float] = []
    d_ref: list[float] = []
    fma_true = 0
    unique_count = dumped_lines = None
    with open_text(path) as f:
        for line in f:
            if line.startswith("#"):
                m = _HEADER_RE.match(line)
                if m:
                    if int(m.group(1)) != n:
                        raise ValueError(f"{path}: n 不一致 {m.group(1)} != {n}")
                    unique_count = int(m.group(2))
                    dumped_lines = int(m.group(3))
                continue
            parts = line.split()
            if not parts or parts[0] == "idx":
                continue
            # 列: idx row col exact f64_seq ref actual |ref-exact| |actual-exact|
            #     |ref-actual| max|partial| sqrtK*ulp fma_bit_match
            if len(parts) != 13:
                raise ValueError(f"{path}: 列数 {len(parts)} != 13: {line.rstrip()}")
            idxs.append(int(parts[0]))
            rows.append(int(parts[1]))
            d_ref.append(float(parts[7]))
            d_actual.append(float(parts[8]))
            if parts[12] == "True":
                fma_true += 1
            elif parts[12] != "False":
                raise ValueError(f"{path}: fma_bit_match の値が不正: {parts[12]}")
    count = len(rows)
    if unique_count is None or unique_count != count or dumped_lines != count:
        raise ValueError(
            f"{path}: ヘッダ件数（ユニーク {unique_count}／行 {dumped_lines}）と本文 {count} 行が不一致"
        )
    dump_abs = read_dump_abs(n)
    if set(dump_abs) != set(idxs):
        raise ValueError(f"{path}: truth の idx 集合と dump の idx 集合が不一致")
    d_ref_actual = [dump_abs[i] for i in idxs]
    closer_actual = sum(1 for a, r in zip(d_actual, d_ref) if a < r)
    closer_ref = sum(1 for a, r in zip(d_actual, d_ref) if r < a)
    ties = count - closer_actual - closer_ref
    k = n  # 正方 GEMM（K=N）
    bound_k = C * U_TF32 * k * M_FIXED
    bound_sqrtk = C * U_TF32 * math.sqrt(k) * M_FIXED
    max_d = max(d_ref_actual)
    rescued_k = sum(1 for d in d_ref_actual if d <= bound_k)
    rescued_sqrtk = sum(1 for d in d_ref_actual if d <= bound_sqrtk)
    fails, dumped, truncated = read_summary(n)
    return {
        "n": n,
        "count": count,
        "row_min": min(rows),
        "row_max": max(rows),
        "calls": len(fails),
        "fail_count_set": sorted(set(fails)),
        "dumped_set": sorted(set(dumped)),
        "truncated": sorted(truncated),
        "da": (min(d_actual), statistics.median(d_actual), max(d_actual)),
        "dr": (min(d_ref), statistics.median(d_ref), max(d_ref)),
        "closer_actual": closer_actual,
        "closer_ref": closer_ref,
        "ties": ties,
        "fma_true": fma_true,
        "bound_k": bound_k,
        "bound_sqrtk": bound_sqrtk,
        "max_d": max_d,
        "rescued_k": rescued_k,
        "rescued_sqrtk": rescued_sqrtk,
    }


def fmt3(x: float) -> str:
    return f"{x:.3e}"


def main() -> int:
    results = [summarize(n) for n in SIZES]
    out = sys.stdout
    print("# burn cuda（TF32）parity fail 要素の真値突合 集計（イシュー #1984）", file=out)
    print(file=out)
    print(
        "`truth-N.txt`（`parity_dump_truth.py --n N` 出力・call=39 の PARITY_DUMP 行のみ）の"
        " 機械集計。判定・tolerance・判定式は変更しない（採否判定なし）。"
        f" bound は u=2^-11・c={C}・M=fixed{M_FIXED}（線形 K: `K/16384`・√K: `√K/16384`）。"
        " N=2048／4096 は行優先先頭 4096 件のサンプルであり母集団の値ではない。",
        file=out,
    )
    print(file=out)
    print("## (a) 件数・row 範囲・40 call の決定性", file=out)
    print(file=out)
    print("| N | 解析行数 | row 範囲 | call 数 | fail_count（全 call の集合） | dumped | truncated |", file=out)
    print("|---:|---:|---|---:|---|---|---|", file=out)
    for r in results:
        print(
            f"| {r['n']} | {r['count']} | {r['row_min']}〜{r['row_max']} | {r['calls']} |"
            f" {', '.join(map(str, r['fail_count_set']))} |"
            f" {', '.join(map(str, r['dumped_set']))} | {', '.join(r['truncated'])} |",
            file=out,
        )
    print(file=out)
    print("## (b) 真値からの距離（min／中央値／max）と近い側の件数", file=out)
    print(file=out)
    print(
        "| N | \\|actual−exact\\| min／中央値／max | \\|ref−exact\\| min／中央値／max |"
        " actual が近い | ref が近い | 同値 |",
        file=out,
    )
    print("|---:|---|---|---:|---:|---:|", file=out)
    for r in results:
        da = "／".join(fmt3(v) for v in r["da"])
        dr = "／".join(fmt3(v) for v in r["dr"])
        print(
            f"| {r['n']} | {da} | {dr} | {r['closer_actual']} | {r['closer_ref']} | {r['ties']} |",
            file=out,
        )
    print(file=out)
    print("## (c) fma_bit_match（参照実装の f32 FMA 逐次累積 bit 再現）", file=out)
    print(file=out)
    print("| N | True 件数 | 割合 |", file=out)
    print("|---:|---:|---:|", file=out)
    for r in results:
        print(f"| {r['n']} | {r['fma_true']}/{r['count']} | {100.0 * r['fma_true'] / r['count']:.2f}% |", file=out)
    print(file=out)
    print("## (d)(e) u=2^-11・c=0.5 の第 3 項（`|ref−actual| <= bound`）", file=out)
    print(file=out)
    print(
        "| N | max\\|ref−actual\\|（解析行内） | 線形 K bound | max ≤ 線形 K bound | 線形 K 救済 |"
        " √K bound | √K 救済 |",
        file=out,
    )
    print("|---:|---:|---:|---|---:|---:|---:|", file=out)
    for r in results:
        print(
            f"| {r['n']} | {fmt3(r['max_d'])} | {fmt3(r['bound_k'])} |"
            f" {'yes' if r['max_d'] <= r['bound_k'] else 'no'} | {r['rescued_k']}/{r['count']} |"
            f" {fmt3(r['bound_sqrtk'])} | {r['rescued_sqrtk']}/{r['count']} |",
            file=out,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
