#!/usr/bin/env python3
"""イシュー #1308: `gemm_splitk_shapes_bench` 実測結果（`run{1..5}.log`）の集計。

python3 標準ライブラリのみで完結する（追加依存なし。`docs/perf/logs/
cpu-gemm-kc-sweep-1315/` 等の先例と同型の集計方式）。

各 run ログから `target=(...) control=(...) target_tflops=... control_tflops=...
target_over_control=... spread_target=... spread_control=...
target_actual_groups=... control_actual_groups=...` 行を抽出し、12 組
（TARGET_MN × TARGET_K）それぞれについて:

- 5 run 生値（target_tflops・control_tflops・target_over_control・spread_*）
- run 別 `target_over_control` の中央値と 5/5 run が `< 0.7` かどうか（符号
  一貫性。`docs/perf/metal-gemm-splitk-shapes.md` §3.3 の事前登録判定規則）
- 両群中央値の比（target 5 run 中央値 ÷ control 5 run 中央値。参考指標。
  PR #1389/#1392 codex-review 指摘の踏襲で「run 内対応比」の中央値と明確に
  区別する）
- §5 判定基準（条件1・条件2）の該当有無

を Markdown 表として `aggregate.md` へ出力する。

`--self-test` で行パース・中央値算出の回帰確認を行う（実機ログなしで実行
可能）。
"""

from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent

LINE_RE = re.compile(
    r"target=\((?P<tm>\d+),(?P<tn>\d+),(?P<tk>\d+)\) "
    r"control=\((?P<cm>\d+),(?P<cn>\d+),(?P<ck>\d+)\) "
    r"target_tflops=(?P<target_tflops>[\d.]+) "
    r"control_tflops=(?P<control_tflops>[\d.]+) "
    r"target_over_control=(?P<target_over_control>[\d.]+) "
    r"spread_target=(?P<spread_target>[\d.]+) "
    r"spread_control=(?P<spread_control>[\d.]+) "
    r"target_actual_groups=(?P<target_actual_groups>\d+) "
    r"control_actual_groups=(?P<control_actual_groups>\d+)"
)

# §3.3 事前登録: (32,32,*)・(64,64,*)・(128,128,*) は analytics 側
# actual_groups（16／4／16）が実機コア数 40 未満のため条件 2 該当。
# (256,256,*) は analytics actual_groups=64 が 40 以上のため条件 2 非該当
# （target_actual_groups 列そのものから機械判定する。resolved 値ではなく
# analytics 値を使うのは §3.3 の事前登録が analytics::analyze 基準で書かれた
# ためであり、本集計はその定義をそのまま踏襲する）。
CONDITION2_GROUPS_THRESHOLD = 40
CONDITION1_RATIO_THRESHOLD = 0.7
STABILITY_SPREAD_GATE = 0.05


def parse_log(path: Path) -> dict[tuple[int, int, int], dict]:
    """1 run ログから 12 組の測定行を抽出する。"""
    rows: dict[tuple[int, int, int], dict] = {}
    text = path.read_text()
    for m in LINE_RE.finditer(text):
        key = (int(m["tm"]), int(m["tn"]), int(m["tk"]))
        rows[key] = {
            "control": (int(m["cm"]), int(m["cn"]), int(m["ck"])),
            "target_tflops": float(m["target_tflops"]),
            "control_tflops": float(m["control_tflops"]),
            "target_over_control": float(m["target_over_control"]),
            "spread_target": float(m["spread_target"]),
            "spread_control": float(m["spread_control"]),
            "target_actual_groups": int(m["target_actual_groups"]),
            "control_actual_groups": int(m["control_actual_groups"]),
        }
    return rows


def aggregate(run_logs: list[Path]) -> str:
    per_run = [parse_log(p) for p in run_logs]
    keys = list(per_run[0].keys())
    for i, rows in enumerate(per_run[1:], start=2):
        if list(rows.keys()) != keys:
            raise ValueError(f"run{i}.log の組順序が run1.log と一致しない")

    lines = []
    lines.append("| target (M,N,K) | control (S,S,S) | target_over_control (5 run) | median | all<0.7 | spread max(t/c) | actual_groups(t) | 条件1 | 条件2 | 判定対象 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    qualifying = 0
    candidate = 0
    for key in keys:
        target_tflops = [r[key]["target_tflops"] for r in per_run]
        control_tflops = [r[key]["control_tflops"] for r in per_run]
        ratios = [r[key]["target_over_control"] for r in per_run]
        spreads_t = [r[key]["spread_target"] for r in per_run]
        spreads_c = [r[key]["spread_control"] for r in per_run]
        control = per_run[0][key]["control"]
        target_groups = per_run[0][key]["target_actual_groups"]

        median_ratio = statistics.median(ratios)
        all_below = all(r < CONDITION1_RATIO_THRESHOLD for r in ratios)
        spread_ok = all(s <= STABILITY_SPREAD_GATE for s in spreads_t + spreads_c)
        # 条件1: 中央値 <0.7 かつ 5/5 run 一貫（run 間で閾値を跨がない）
        cond1 = median_ratio < CONDITION1_RATIO_THRESHOLD and all_below
        cond2 = target_groups < CONDITION2_GROUPS_THRESHOLD

        is_candidate = cond2
        if is_candidate:
            candidate += 1
        if is_candidate and cond1:
            qualifying += 1

        alt_ratio = statistics.median(target_tflops) / statistics.median(control_tflops)
        ratios_str = ",".join(f"{r:.4f}" for r in ratios)

        lines.append(
            f"| {key} | {control} | {ratios_str} | {median_ratio:.4f} (alt={alt_ratio:.4f}) "
            f"| {'yes' if all_below else 'no'} | {'ok' if spread_ok else 'EXCEEDED'} "
            f"| {target_groups} | {'○' if cond1 else '×'} | {'○' if cond2 else '×'} "
            f"| {'candidate' if is_candidate else '-'} |"
        )

    lines.append("")
    total_candidates = candidate
    lines.append(
        f"条件2 該当（並列度不足の解析裏付け）候補: {total_candidates} / 12 点。"
        f"うち条件1（劣化率 <0.7・5/5 run 一貫）も成立: {qualifying} / {total_candidates}"
        f"（{'≥7' if qualifying >= 7 else '<7'}）。"
    )
    verdict = "採用検討推奨" if qualifying >= 7 else "不採用（現状維持）"
    lines.append(f"§3.3 事前登録規則による機械判定: **{verdict}**")

    return "\n".join(lines)


def self_test() -> None:
    sample = (
        "target=(64,64,4096) control=(256,256,256) target_tflops=0.0909 "
        "control_tflops=0.1998 target_over_control=0.4552 spread_target=0.7399 "
        "spread_control=0.4255 target_actual_groups=4 control_actual_groups=64\n"
    )
    tmp = LOG_DIR / "_self_test_sample.log"
    tmp.write_text(sample)
    try:
        rows = parse_log(tmp)
        key = (64, 64, 4096)
        assert key in rows, "行パース失敗"
        assert rows[key]["control"] == (256, 256, 256)
        assert abs(rows[key]["target_tflops"] - 0.0909) < 1e-9
        assert rows[key]["target_actual_groups"] == 4
        assert rows[key]["control_actual_groups"] == 64
        print("self-test OK")
    finally:
        tmp.unlink()


def main() -> None:
    if "--self-test" in sys.argv:
        self_test()
        return

    run_logs = sorted(LOG_DIR.glob("run[1-5].log"))
    if len(run_logs) != 5:
        print(
            f"run1.log〜run5.log が揃っていない（見つかった数: {len(run_logs)}）",
            file=sys.stderr,
        )
        sys.exit(1)

    md = aggregate(run_logs)
    out = LOG_DIR / "aggregate.md"
    out.write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
