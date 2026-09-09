#!/usr/bin/env python3
"""split-K vs classic 経路の A/B 実行ログ（イシュー #1475）を集計する。

`crates/backend-metal/examples/gemm_splitk_ab_bench.rs` が各プロセス起動
（`runN.log`）へ出力する `splitk_ab` 行を解析し、形状（kind, m, n, k）ごとに
複数 run の speedup を集約して `docs/perf/metal-gemm-splitk-ab.md` §4 へ
転記する `aggregate.md` を生成する。

python3 標準ライブラリのみ（許容依存外の追加なし。`docs/perf/logs/
metal-gemm-splitk-shapes-1308/aggregate.py` と同型の設計）。

事前登録判定基準（イシュー #1475 コメント）:
- ADOPT: 対象 9 形状すべてで (i) run 内比の run 数中央値 >= 1.5 かつ
  (ii) 全 run で run 内比 > 1.0（符号一貫性）、かつ対照 3 形状すべてで
  中央値 >= 0.95。
- 上記いずれかが不成立なら REJECT。
- `verdict=undetermined` 行を検出した run があれば全体を undetermined とする。

使い方:
    python3 aggregate.py --self-test
    python3 aggregate.py run1.log run2.log ... > aggregate.md
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field

ADOPT_TARGET_MIN_SPEEDUP = 1.5
ADOPT_CONTROL_MIN_SPEEDUP = 0.95


@dataclass
class ShapeSamples:
    kind: str
    m: int
    n: int
    k: int
    speedups: list[float] = field(default_factory=list)
    # フロア計測（単一腕。`speedup=NA`）用に A 側 median_secs のみ集める。
    median_a_secs_list: list[float] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (self.kind, self.m, self.n, self.k)

    @property
    def median_speedup(self) -> float | None:
        if not self.speedups:
            return None
        return statistics.median(self.speedups)

    @property
    def median_a_secs(self) -> float | None:
        if not self.median_a_secs_list:
            return None
        return statistics.median(self.median_a_secs_list)

    @property
    def all_positive(self) -> bool:
        return all(s > 1.0 for s in self.speedups)


def parse_splitk_ab_line(line: str) -> dict[str, str] | None:
    """`splitk_ab ...` 行を key=value の dict へ分解する（純関数）。"""
    if not line.startswith("splitk_ab "):
        return None
    out: dict[str, str] = {}
    for tok in line.strip().split(" ")[1:]:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        out[k] = v
    return out


def parse_log(text: str) -> tuple[list[dict[str, str]], bool]:
    """1 run 分のログを解析し、(splitk_ab 行の dict 列, undetermined か) を返す。"""
    rows: list[dict[str, str]] = []
    undetermined = False
    for line in text.splitlines():
        if line.strip() == "verdict=undetermined":
            undetermined = True
        parsed = parse_splitk_ab_line(line)
        if parsed is not None:
            rows.append(parsed)
    return rows, undetermined


def aggregate(logs: list[str]) -> tuple[dict[tuple[str, int, int, int], ShapeSamples], bool]:
    """複数 run のログテキストを集約する。undetermined が 1 件でもあれば
    True を返し、その場合 shapes は参考値として扱う（判定には使わない）。
    """
    shapes: dict[tuple[str, int, int, int], ShapeSamples] = {}
    any_undetermined = False
    for text in logs:
        rows, undetermined = parse_log(text)
        if undetermined:
            any_undetermined = True
        for row in rows:
            kind = row.get("kind")
            if kind is None or "m" not in row or "n" not in row or "k" not in row:
                continue
            m, n, k = int(row["m"]), int(row["n"]), int(row["k"])
            key = (kind, m, n, k)
            if key not in shapes:
                shapes[key] = ShapeSamples(kind, m, n, k)
            speedup = row.get("speedup")
            if speedup is not None and speedup != "NA":
                shapes[key].speedups.append(float(speedup))
            median_a = row.get("median_a_secs")
            if median_a is not None and median_a != "NA":
                shapes[key].median_a_secs_list.append(float(median_a))
    return shapes, any_undetermined


def render_markdown(shapes: dict[tuple[str, int, int, int], ShapeSamples], any_undetermined: bool, n_runs: int) -> str:
    lines: list[str] = []
    lines.append(f"# split-K A/B 集計（{n_runs} run）\n")
    if any_undetermined:
        lines.append("**いずれかの run が `verdict=undetermined`（専有ゲート不成立）で終了した。"
                      "以下は参考値であり判定には使わない。**\n")

    for kind in ["target", "target_tile", "control", "control_forced", "floor"]:
        rows = [v for v in shapes.values() if v.kind == kind]
        if not rows:
            continue
        lines.append(f"\n## {kind}\n")
        lines.append("| m | n | k | n_runs | speedups | median_speedup | all_run_positive | median_a_secs |")
        lines.append("|---|---|---|--------|----------|-----------------|-------------------|----------------|")
        for s in sorted(rows, key=lambda r: (r.m, r.n, r.k)):
            speedups_str = ",".join(f"{v:.4f}" for v in s.speedups) or "NA"
            median = s.median_speedup
            median_str = f"{median:.4f}" if median is not None else "NA"
            median_a = s.median_a_secs
            median_a_str = f"{median_a:.6e}" if median_a is not None else "NA"
            lines.append(
                f"| {s.m} | {s.n} | {s.k} | {len(s.speedups) or len(s.median_a_secs_list)} | "
                f"{speedups_str} | {median_str} | {s.all_positive} | {median_a_str} |"
            )

    if not any_undetermined:
        target_rows = [v for v in shapes.values() if v.kind == "target"]
        control_rows = [v for v in shapes.values() if v.kind == "control"]
        target_ok = len(target_rows) == 9 and all(
            (t.median_speedup or 0.0) >= ADOPT_TARGET_MIN_SPEEDUP and t.all_positive
            for t in target_rows
        )
        control_ok = len(control_rows) == 3 and all(
            (c.median_speedup or 0.0) >= ADOPT_CONTROL_MIN_SPEEDUP for c in control_rows
        )
        verdict = "ADOPT" if (target_ok and control_ok) else "REJECT"
        lines.append(f"\n## verdict\n\n**{verdict}**"
                      f"（target_shapes={len(target_rows)}/9・target_ok={target_ok}・"
                      f"control_shapes={len(control_rows)}/3・control_ok={control_ok}）\n")
    else:
        lines.append("\n## verdict\n\n**undetermined**\n")

    return "\n".join(lines) + "\n"


def self_test() -> None:
    sample = (
        "splitk_ab kind=target m=32 n=32 k=2048 partitions=32 median_a_secs=2e-4 "
        "median_b_secs=1e-4 speedup=2.0000 spread_a=0.1 spread_b=0.1\n"
        "splitk_ab kind=control m=256 n=256 k=2048 partitions=NA median_a_secs=5e-4 "
        "median_b_secs=5e-4 speedup=1.0000 spread_a=0.1 spread_b=0.1\n"
        "splitk_ab kind=floor m=32 n=32 k=64 partitions=NA median_a_secs=8e-5 "
        "median_b_secs=NA speedup=NA spread_a=0.1 spread_b=NA\n"
    )
    rows, undetermined = parse_log(sample)
    assert len(rows) == 3, rows
    assert not undetermined

    shapes, any_undetermined = aggregate([sample, sample])
    assert not any_undetermined
    key = ("target", 32, 32, 2048)
    assert key in shapes
    assert shapes[key].speedups == [2.0, 2.0]
    assert shapes[key].median_speedup == 2.0
    assert shapes[key].all_positive

    undetermined_sample = "verdict=undetermined\n"
    _, undetermined2 = parse_log(undetermined_sample)
    assert undetermined2

    md = render_markdown(shapes, False, 2)
    assert "target" in md
    assert "floor" in md

    print("self-test OK", file=sys.stderr)


def main() -> None:
    if "--self-test" in sys.argv:
        self_test()
        return

    paths = [a for a in sys.argv[1:] if a != "--self-test"]
    if not paths:
        print("usage: aggregate.py run1.log run2.log ... [--self-test]", file=sys.stderr)
        sys.exit(1)

    logs = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            logs.append(f.read())

    shapes, any_undetermined = aggregate(logs)
    print(render_markdown(shapes, any_undetermined, len(paths)))


if __name__ == "__main__":
    main()
