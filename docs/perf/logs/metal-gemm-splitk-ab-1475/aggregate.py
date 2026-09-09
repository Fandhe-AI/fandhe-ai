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
from collections import Counter
from dataclasses import dataclass, field

ADOPT_TARGET_MIN_SPEEDUP = 1.5
ADOPT_CONTROL_MIN_SPEEDUP = 0.95

# `crates/backend-metal/examples/gemm_splitk_ab_bench.rs` の `target_shapes`／
# `control_shapes`（`TARGET_MN`／`CONTROL_MN`／`K_LIST`）と同一の期待形状
# 集合。各 run は対象 9・対照 3 の各キーをちょうど 1 行ずつ持つことを
# 前提とする（イシュー #1499 codex-review P2 指摘: 集約後の件数だけでは
# 「あるキーの欠落」と「別キーの重複」が相殺し得るため、run ごとの
# 期待形状集合との完全一致を検査する）。
_TARGET_MN = (32, 64, 128)
_CONTROL_MN = 256
_K_LIST = (2048, 4096, 8192)
EXPECTED_TARGET_KEYS = frozenset(
    ("target", mn, mn, k) for mn in _TARGET_MN for k in _K_LIST
)
EXPECTED_CONTROL_KEYS = frozenset(
    ("control", _CONTROL_MN, _CONTROL_MN, k) for k in _K_LIST
)


def check_run_shape_completeness(
    rows: list[dict[str, str]], run_index: int
) -> list[str]:
    """1 run 分の `splitk_ab` 行（target/control kind に限る）が、期待形状
    集合の各キーをちょうど 1 回ずつ含むことを検査する純関数。

    集約後の総件数（`len(speedups) == n_runs`）だけでは、ある run での
    欠落を別 run での重複が相殺してしまい検出できない（イシュー #1499
    codex-review P2 指摘）。本関数は run 単体のキー集合を期待集合と
    突き合わせるため、run をまたいだ相殺が起こらない。
    """
    counts: Counter[tuple[str, int, int, int]] = Counter()
    for row in rows:
        kind = row.get("kind")
        if kind not in ("target", "control"):
            continue
        if "m" not in row or "n" not in row or "k" not in row:
            continue
        key = (kind, int(row["m"]), int(row["n"]), int(row["k"]))
        counts[key] += 1

    violations: list[str] = []
    expected = EXPECTED_TARGET_KEYS | EXPECTED_CONTROL_KEYS
    for key in sorted(expected):
        n = counts.get(key, 0)
        if n != 1:
            violations.append(
                f"run{run_index}: kind={key[0]} m={key[1]} n={key[2]} k={key[3]} "
                f"の行数が {n}（期待 1）"
            )
    unexpected = set(counts) - expected
    for key in sorted(unexpected):
        violations.append(
            f"run{run_index}: kind={key[0]} m={key[1]} n={key[2]} k={key[3]} "
            f"は期待形状集合外（n={counts[key]}）"
        )
    return violations


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


def aggregate(
    logs: list[str],
) -> tuple[dict[tuple[str, int, int, int], ShapeSamples], bool, list[str]]:
    """複数 run のログテキストを集約する。undetermined が 1 件でもあれば
    2 番目の戻り値が True になり、その場合 shapes は参考値として扱う
    （判定には使わない）。3 番目の戻り値は run ごとの期待形状集合との
    不一致（欠落・重複・想定外キー）の説明文（空なら全 run が完全）で、
    1 件でもあれば 2 番目の戻り値も True にする（イシュー #1499
    codex-review P2 指摘: 集約後の総件数一致だけでは run をまたいだ
    欠落／重複の相殺を検出できないため、run 単位で
    `check_run_shape_completeness` を呼び、run の所属情報を保ったまま
    検査する）。
    """
    shapes: dict[tuple[str, int, int, int], ShapeSamples] = {}
    any_undetermined = False
    shape_violations: list[str] = []
    for run_index, text in enumerate(logs):
        rows, undetermined = parse_log(text)
        if undetermined:
            any_undetermined = True
        run_violations = check_run_shape_completeness(rows, run_index)
        if run_violations:
            shape_violations.extend(run_violations)
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
    if shape_violations:
        any_undetermined = True
    return shapes, any_undetermined, shape_violations


def render_markdown(
    shapes: dict[tuple[str, int, int, int], ShapeSamples],
    any_undetermined: bool,
    n_runs: int,
    shape_violations: list[str] | None = None,
) -> str:
    shape_violations = shape_violations or []
    lines: list[str] = []
    lines.append(f"# split-K A/B 集計（{n_runs} run）\n")
    if any_undetermined:
        lines.append("**いずれかの run が `verdict=undetermined`（専有ゲート不成立）"
                      "または run 単位の期待形状集合検査で不一致（欠落・重複・想定外キー）"
                      "を検出した。以下は参考値であり判定には使わない。**\n")
    if shape_violations:
        lines.append("\n### run 単位の形状不一致\n")
        for v in shape_violations:
            lines.append(f"- {v}")
        lines.append("")

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
        # 完全性検査（イシュー #1499 codex-review 指摘）: 形状の個数が
        # 9/3 揃っているだけでは、いずれかの run でその形状が欠落・
        # 重複していても検出できない（例: 1 run だけで 9 形状が揃えば
        # ADOPT を出力し得た）。各形状の `speedups` 件数が `n_runs`
        # （渡された run ログの本数）と一致することまで確認し、AGENTS.md
        # の「5 回計測の中央値・5/5 run」契約を機械的に担保する。
        target_complete = len(target_rows) == 9 and all(
            len(t.speedups) == n_runs for t in target_rows
        )
        control_complete = len(control_rows) == 3 and all(
            len(c.speedups) == n_runs for c in control_rows
        )
        if not (target_complete and control_complete):
            lines.append(
                "\n## verdict\n\n**undetermined**"
                "（run の完全性検査が不成立: 対象・対照形状のいずれかが"
                f" 全 {n_runs} run に揃っていない。"
                f"target_shapes={len(target_rows)}/9・target_complete={target_complete}・"
                f"control_shapes={len(control_rows)}/3・control_complete={control_complete}）\n"
            )
        else:
            target_ok = all(
                (t.median_speedup or 0.0) >= ADOPT_TARGET_MIN_SPEEDUP and t.all_positive
                for t in target_rows
            )
            control_ok = all(
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

    shapes, any_undetermined, violations = aggregate([sample, sample])
    # sample は対象 1・対照 1 形状のみ（本 self-test は行解析のプラミング
    # 確認が目的で、期待形状集合とは意図的に一致しない）。よって run 単位の
    # 期待形状集合検査（`check_run_shape_completeness`）は不一致を検出し
    # `any_undetermined` は True になる（後段で render_markdown の
    # `**undetermined**` 出力として再確認する）。
    assert any_undetermined
    assert violations
    key = ("target", 32, 32, 2048)
    assert key in shapes
    assert shapes[key].speedups == [2.0, 2.0]
    assert shapes[key].median_speedup == 2.0
    assert shapes[key].all_positive

    undetermined_sample = "verdict=undetermined\n"
    _, undetermined2 = parse_log(undetermined_sample)
    assert undetermined2

    md = render_markdown(shapes, any_undetermined, 2, violations)
    assert "target" in md
    assert "floor" in md
    # 対象 9・対照 3 形状が揃っていない（self-test サンプルは各 1 形状のみ）
    # ため、判定は undetermined になるはず（イシュー #1499 codex-review
    # 指摘: 形状個数だけでなく run ごとの揃いを検査する）。
    assert "verdict" in md
    assert "**undetermined**" in md

    # 完全性検査そのものの自己検証: 対象 9・対照 3 形状を 5 run 分すべて
    # 揃えれば ADOPT、いずれか 1 run でも 1 形状が欠落すれば undetermined
    # になることを確認する（AGENTS.md の 5/5 run 契約の機械的担保）。
    target_mn = [32, 64, 128]
    control_mn = 256
    k_list = [2048, 4096, 8192]

    def make_run(
        drop: tuple[str, int, int, int] | None,
        dup: tuple[str, int, int, int] | None = None,
    ) -> str:
        lines_: list[str] = []
        for mn in target_mn:
            for k in k_list:
                if drop == ("target", mn, mn, k):
                    continue
                line = (
                    f"splitk_ab kind=target m={mn} n={mn} k={k} partitions=4 "
                    "median_a_secs=2e-4 median_b_secs=1e-4 speedup=2.0000 "
                    "spread_a=0.1 spread_b=0.1"
                )
                lines_.append(line)
                if dup == ("target", mn, mn, k):
                    lines_.append(line)
        for k in k_list:
            if drop == ("control", control_mn, control_mn, k):
                continue
            line = (
                f"splitk_ab kind=control m={control_mn} n={control_mn} k={k} "
                "partitions=NA median_a_secs=5e-4 median_b_secs=5e-4 "
                "speedup=1.0000 spread_a=0.1 spread_b=0.1"
            )
            lines_.append(line)
            if dup == ("control", control_mn, control_mn, k):
                lines_.append(line)
        return "\n".join(lines_) + "\n"

    complete_runs = [make_run(None) for _ in range(5)]
    complete_shapes, complete_any_undetermined, complete_violations = aggregate(
        complete_runs
    )
    assert not complete_any_undetermined
    assert not complete_violations
    complete_md = render_markdown(
        complete_shapes, complete_any_undetermined, len(complete_runs), complete_violations
    )
    assert "**ADOPT**" in complete_md, complete_md

    # 5 run 中 1 run だけ対象形状の 1 つ（32,32,2048）が欠落 → 全体が
    # undetermined になるはず（欠落を無視して ADOPT を出力してはならない）。
    incomplete_runs = [make_run(None) for _ in range(4)] + [
        make_run(("target", 32, 32, 2048))
    ]
    incomplete_shapes, incomplete_any_undetermined, incomplete_violations = aggregate(
        incomplete_runs
    )
    assert incomplete_any_undetermined
    assert incomplete_violations
    incomplete_md = render_markdown(
        incomplete_shapes, incomplete_any_undetermined, len(incomplete_runs), incomplete_violations
    )
    assert "**undetermined**" in incomplete_md, incomplete_md
    assert "**ADOPT**" not in incomplete_md, incomplete_md
    assert "**REJECT**" not in incomplete_md, incomplete_md

    # イシュー #1499 codex-review P2 が指摘した具体的な相殺シナリオ:
    # 5 run 中 1 run で対象形状（32,32,2048）が欠落し、別の 1 run で同じ
    # 形状が重複する。総件数（len(speedups)）は 4+0+1+1+1=... ではなく
    # 5（欠落 run 0 件・重複 run 2 件・残り 3 run 各 1 件）と n_runs=5 に
    # 一致してしまうため、run の所属情報を捨てた集計だけでは検出できない
    # （旧実装はこのケースで ADOPT を誤って出力していた）。run 単位の
    # `check_run_shape_completeness` はこの相殺を許さず undetermined に
    # なることを確認する。
    offset_runs = [
        make_run(("target", 32, 32, 2048), None),
        make_run(None, ("target", 32, 32, 2048)),
    ] + [make_run(None) for _ in range(3)]
    assert len(offset_runs) == 5
    offset_shapes, offset_any_undetermined, offset_violations = aggregate(offset_runs)
    assert offset_any_undetermined
    assert offset_violations
    key = ("target", 32, 32, 2048)
    assert key in offset_shapes
    # 相殺により総件数だけは n_runs と一致してしまうことも確認しておく
    # （このケースを検出できるのが run 単位検査の価値であることの裏付け）。
    assert len(offset_shapes[key].speedups) == len(offset_runs)
    offset_md = render_markdown(
        offset_shapes, offset_any_undetermined, len(offset_runs), offset_violations
    )
    assert "**undetermined**" in offset_md, offset_md
    assert "**ADOPT**" not in offset_md, offset_md
    assert "**REJECT**" not in offset_md, offset_md

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

    shapes, any_undetermined, violations = aggregate(logs)
    print(render_markdown(shapes, any_undetermined, len(paths), violations))


if __name__ == "__main__":
    main()
