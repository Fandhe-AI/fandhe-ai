#!/usr/bin/env python3
"""イシュー #1692（CUDA `mse_loss_backward` ストリーム順序契約）のノイズ床集計。

`orchestrate.sh` が生成した `runs-1692/{before,after}_round{N}_lines.txt`
（`bench[<case>][<numel>].median_s=…` と `grad[<case>][<numel>].fold_bits=0x…`
行の抽出）と `runs-1692/rounds.log`（round ごとの起動順）を読み、
case×numel ごとに before／after 各 round の `median_s` の中央値・比
（after/before）・`fold_bits` の腕内・腕間一致を Markdown 表として
標準出力へ書く。python3 標準ライブラリのみ・決定的（時刻・絶対パスを
出力しない）。

判定規則の正は `docs/perf/cuda-mse-backward-stream-contract.md` §2。本件は
`crates/*/src` の MSE 経路に機能差分がないため ADOPT／REJECT の判定対象では
なく、本スクリプトも verdict を出力しない（ノイズ床・再現性の記録のみ）。

使い方:
    python3 aggregate.py <runs-dir> > aggregate.md
    python3 aggregate.py --self-test
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

BENCH_RE = re.compile(r"^bench\[(?P<case>[^\]]+)\]\[(?P<numel>\d+)\]\.median_s=(?P<v>[0-9.eE+-]+)$")
FOLD_RE = re.compile(r"^grad\[(?P<case>[^\]]+)\]\[(?P<numel>\d+)\]\.fold_bits=(?P<v>0x[0-9a-fA-F]+)$")
ROUND_RE = re.compile(r"^round=(?P<n>\d+) order=(?P<order>\S+) before_rc=(?P<brc>\d+) after_rc=(?P<arc>\d+)$")
ARMS = ("before", "after")


def parse_lines(text: str) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], str]]:
    bench: dict[tuple[str, int], float] = {}
    fold: dict[tuple[str, int], str] = {}
    for line in text.splitlines():
        line = line.strip()
        m = BENCH_RE.match(line)
        if m:
            key = (m["case"], int(m["numel"]))
            if key in bench:
                raise ValueError(f"duplicate bench line for {key}")
            bench[key] = float(m["v"])
            continue
        m = FOLD_RE.match(line)
        if m:
            key = (m["case"], int(m["numel"]))
            if key in fold:
                raise ValueError(f"duplicate fold_bits line for {key}")
            fold[key] = m["v"].lower()
    return bench, fold


def parse_rounds(text: str) -> list[dict[str, str]]:
    rounds = []
    for line in text.splitlines():
        m = ROUND_RE.match(line.strip())
        if m:
            rounds.append(m.groupdict())
    return rounds


def load(runs_dir: Path):
    rounds = parse_rounds((runs_dir / "rounds.log").read_text())
    if not rounds:
        raise ValueError("rounds.log に round 行がない")
    data: dict[str, list[tuple[dict, dict]]] = {arm: [] for arm in ARMS}
    for r in rounds:
        n = r["n"]
        for arm in ARMS:
            p = runs_dir / f"{arm}_round{n}_lines.txt"
            data[arm].append(parse_lines(p.read_text()))
    return rounds, data


def fmt_s(v: float) -> str:
    return f"{v:.9f}"


def render(rounds, data) -> str:
    keys = sorted({k for arm in ARMS for b, _ in data[arm] for k in b}, key=lambda k: (k[0] != "train_shape", k[0], k[1]))
    nrounds = len(rounds)
    out: list[str] = []
    out.append("# イシュー #1692 集計（`aggregate.py` 生成・決定的）")
    out.append("")
    out.append(f"- round 数: {nrounds}（起動順: " + ", ".join(f"r{r['n']}={r['order']}" for r in rounds) + "）")
    rcs = sorted({(r["brc"], r["arc"]) for r in rounds})
    out.append(f"- 終了コード (before_rc, after_rc) の集合: {rcs}")
    out.append("- 本表は ADOPT／REJECT 判定を含まない（§2 規則: MSE 経路に機能差分なし → ノイズ床・再現性の記録のみ）")
    out.append("")
    out.append("## 1. case×numel 別 median_s（各腕 5 round の中央値）と比")
    out.append("")
    out.append("| case | numel | before 中央値 [s] | before min–max [s] | after 中央値 [s] | after min–max [s] | ratio after/before | fold_bits 腕内一致 (before/after) | fold_bits 腕間一致 |")
    out.append("|------|-------|------------------|--------------------|-----------------|-------------------|--------------------|-------------------------------|-------------------|")
    summary = {}
    for case, numel in keys:
        vals = {arm: [b[(case, numel)] for b, _ in data[arm]] for arm in ARMS}
        folds = {arm: [f[(case, numel)] for _, f in data[arm]] for arm in ARMS}
        med = {arm: statistics.median(vals[arm]) for arm in ARMS}
        ratio = med["after"] / med["before"]
        within = {arm: len(set(folds[arm])) == 1 for arm in ARMS}
        across = within["before"] and within["after"] and folds["before"][0] == folds["after"][0]
        summary[(case, numel)] = (ratio, across, folds["before"][0] if within["before"] else None)
        out.append(
            f"| {case} | {numel} | {fmt_s(med['before'])} | {fmt_s(min(vals['before']))}–{fmt_s(max(vals['before']))} "
            f"| {fmt_s(med['after'])} | {fmt_s(min(vals['after']))}–{fmt_s(max(vals['after']))} | {ratio:.4f} "
            f"| {'一致' if within['before'] else '不一致'}/{'一致' if within['after'] else '不一致'} "
            f"| {'一致 (' + folds['before'][0] + ')' if across else '不一致'} |"
        )
    out.append("")
    ratios = [v[0] for v in summary.values()]
    out.append(f"- ratio の範囲（全 case）: {min(ratios):.4f}〜{max(ratios):.4f}")
    out.append(f"- fold_bits 腕内・腕間とも全 case 一致: {'はい' if all(v[1] for v in summary.values()) else 'いいえ'}")
    out.append("")
    out.append("## 2. round 別生値（median_s [s]・起動順付き）")
    out.append("")
    header = "| case | numel | " + " | ".join(f"r{r['n']} before" for r in rounds) + " | " + " | ".join(f"r{r['n']} after" for r in rounds) + " |"
    out.append(header)
    out.append("|" + "---|" * (2 + 2 * nrounds))
    out.append("| (起動順) | | " + " | ".join(r["order"] for r in rounds) + " | " + " | ".join(r["order"] for r in rounds) + " |")
    for case, numel in keys:
        row = [case, str(numel)]
        for arm in ARMS:
            row += [fmt_s(b[(case, numel)]) for b, _ in data[arm]]
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    out.append("## 3. round 別 after/before 比（同一 round 内・起動順との対応）")
    out.append("")
    out.append("| case | numel | " + " | ".join(f"r{r['n']} ({r['order']})" for r in rounds) + " |")
    out.append("|" + "---|" * (2 + nrounds))
    for case, numel in keys:
        row = [case, str(numel)]
        for i in range(nrounds):
            b = data["before"][i][0][(case, numel)]
            a = data["after"][i][0][(case, numel)]
            row.append(f"{a / b:.4f}")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    return "\n".join(out) + "\n"


def self_test() -> None:
    b, f = parse_lines("bench[train_shape][640].median_s=0.000017024\ngrad[train_shape][640].fold_bits=0x7FB3\ngrad[train_shape][640][0].bits=0x1\n")
    assert b == {("train_shape", 640): 0.000017024}, b
    assert f == {("train_shape", 640): "0x7fb3"}, f
    r = parse_rounds("round=1 order=before_first before_rc=0 after_rc=0\n")
    assert r == [{"n": "1", "order": "before_first", "brc": "0", "arc": "0"}], r
    assert statistics.median([3.0, 1.0, 2.0]) == 2.0
    print("self-test ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    rounds, data = load(Path(argv[1]))
    sys.stdout.write(render(rounds, data))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
