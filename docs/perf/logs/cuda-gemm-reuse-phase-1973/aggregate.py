#!/usr/bin/env python3
"""イシュー #1973 実測ログ集計スクリプト（python3 標準ライブラリのみ）。

`orchestrate.sh` が生成する `layerA-phases-N{1024,2048,4096}.log`
（bench-fandhe の `--task gemm --mode reuse --phases` JSONL 出力を
5 run 分連結したもの）から、フェーズ別の 5 run 中央値と
`iter_total` に対する比率を算出し Markdown 表として出力する。
#1182（`docs/perf/cuda-gemm-reuse-phase-1182/`）の §3・§6 と同じ
集計方式（各 run の `median_s` を集め、run 間でさらに中央値を取る）。

`--self-test` で固定 fixture に対する自己検証を行う（実機不要。
Linux CI で実行可能）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PHASES = ("matmul", "to_tensor", "host_copy", "checksum", "iter_total")


def parse_phase_log(text: str) -> dict[str, list[float]]:
    """`layerA-phases-N*.log` のテキストから phase 別 median_s 列を
    抽出する。`-- run i/N=n --` の区切り行はコメントとして無視し、
    JSON 行のみを対象にする（fail-closed: JSON でない行・想定外の
    phase 名は無視する。行番号のずれで誤集計しないよう `phase` キー
    必須の行のみ扱う）。
    """
    out: dict[str, list[float]] = {p: [] for p in PHASES}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        phase = rec.get("phase")
        if phase in out and "median_s" in rec:
            out[phase].append(float(rec["median_s"]))
    return out


def median_of(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.median(values)


def render_table(n: int, phase_medians: dict[str, list[float]]) -> str:
    lines = [f"### N={n}", "", "| phase | 5 run 中央値 (ms) | iter_total 比 |", "| --- | --- | --- |"]
    iter_total_med = median_of(phase_medians.get("iter_total", []))
    for phase in PHASES:
        med = median_of(phase_medians.get(phase, []))
        if med is None:
            lines.append(f"| {phase} | (データなし) | - |")
            continue
        ratio = "-"
        if phase != "iter_total" and iter_total_med:
            ratio = f"{(med / iter_total_med) * 100:.1f}%"
        lines.append(f"| {phase} | {med * 1e3:.4f} | {ratio} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default=str(Path(__file__).parent),
        help="layerA-phases-N*.log を探すディレクトリ（既定: このスクリプトのあるディレクトリ）",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1024, 2048, 4096], help="対象 N 一覧"
    )
    parser.add_argument("--self-test", action="store_true", help="固定 fixture で自己検証する")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    log_dir = Path(args.log_dir)
    md_parts = ["# イシュー #1973 Layer A 集計（自動生成）", ""]
    any_found = False
    for n in args.sizes:
        log_path = log_dir / f"layerA-phases-N{n}.log"
        if not log_path.exists():
            md_parts.append(f"### N={n}\n\n(ログ未生成: {log_path.name})\n")
            continue
        any_found = True
        phase_medians = parse_phase_log(log_path.read_text())
        md_parts.append(render_table(n, phase_medians))

    print("\n".join(md_parts))
    if not any_found:
        print(
            "警告: 対象ログが 1 件も見つからなかった（未実測）。"
            " orchestrate.sh を実行してから再度呼ぶこと。",
            file=sys.stderr,
        )
    return 0


def run_self_test() -> int:
    """固定 fixture（2 run 分の JSON 行）で median 計算・比率算出の
    正しさを検証する。実機ログを必要としない。
    """
    fixture = "\n".join(
        [
            "-- run 1/N=1024 --",
            json.dumps({"phase": "matmul", "median_s": 0.001}),
            json.dumps({"phase": "to_tensor", "median_s": 0.0}),
            json.dumps({"phase": "host_copy", "median_s": 0.002}),
            json.dumps({"phase": "checksum", "median_s": 0.0005}),
            json.dumps({"phase": "iter_total", "median_s": 0.0035}),
            "-- run 2/N=1024 --",
            json.dumps({"phase": "matmul", "median_s": 0.002}),
            json.dumps({"phase": "to_tensor", "median_s": 0.0}),
            json.dumps({"phase": "host_copy", "median_s": 0.003}),
            json.dumps({"phase": "checksum", "median_s": 0.0007}),
            json.dumps({"phase": "iter_total", "median_s": 0.0057}),
        ]
    )
    parsed = parse_phase_log(fixture)
    assert parsed["matmul"] == [0.001, 0.002], parsed["matmul"]
    assert abs(median_of(parsed["matmul"]) - 0.0015) < 1e-12
    assert abs(median_of(parsed["iter_total"]) - 0.0046) < 1e-12

    table = render_table(1024, parsed)
    assert "N=1024" in table
    assert "matmul" in table

    # 未生成ログ（空辞書）でも例外を投げないこと
    empty_table = render_table(2048, {p: [] for p in PHASES})
    assert "データなし" in empty_table

    print("self-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
