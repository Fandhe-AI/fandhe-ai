#!/usr/bin/env python3
"""Layer B（gemm_reuse_phase_diag_cpu、イシュー #1301）5 run 集計。

`layerB-<node>-<arm>-run{1..5}.log`（イシュー #1292 の
`run_layerB_m4max.sh`／`run_layerB_dgx.sh` と同一フォーマット。各 run は
テスト自身が 20 trial の中央値を出す `median=... ms` 行を N=512/1024/2048
ごとに `alloc_c`／`kernel`／`ops_gemm` 等の区間名で出力する）を読み、5 run
分の `median=` 値からさらに中央値を取って on/off 比を Markdown 表にする
（coding-rust.md「ベンチは 5 回計測の中央値」に従い、プロセス起動 5 回の
中央値を最終値とする。標準ライブラリのみで完結させる。#1301）。

使い方:
  python3 aggregate_layer_b.py --node dgx --off-glob 'layerB-dgx-off-run*.log' \\
      --on-glob 'layerB-dgx-on-run*.log' > aggregate-dgx.md
"""
from __future__ import annotations

import argparse
import glob
import re
import statistics
import sys

LINE_RE = re.compile(
    r"^\s*(?P<phase>[A-Za-z_]+)(?:\s*\([^)]*\))?:\s*median=(?P<median>[0-9.]+)\s*ms"
)
# N=512 の行のみ cargo test の出力接頭辞
# （`test gemm_reuse_phase_diag_tests::gemm_reuse_phase_diag_cpu ...   `）が
# 先頭に付いた状態で出力される（N=1024/2048 は直前の N ブロック終端の後に
# 素の `  N=1024 (median over ...):` で出力される）。`^\s*N=` に固定すると
# N=512 の行がこの接頭辞と一致せず cur_n が設定されないまま alloc_c 等の
# 後続行が捨てられ、N=512 が集計表から欠落する（イシュー #1301
# codex-review 指摘）。行頭固定をやめ、行中のどこかに `N=<n> (median over`
# が現れれば検出する `.search` へ切り替える。
SIZE_RE = re.compile(r"N=(?P<n>\d+)\s*\(median over")

PHASES = ["alloc_c", "kernel", "tensor_wrap", "ops_gemm", "tape_matmul", "to_tensor", "host_copy", "checksum"]


def parse_log(path: str) -> dict[int, dict[str, float]]:
    """1 run のログを { N: { phase: median_ms } } へパースする。"""
    out: dict[int, dict[str, float]] = {}
    cur_n: int | None = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m_size = SIZE_RE.search(line)
            if m_size:
                cur_n = int(m_size.group("n"))
                out.setdefault(cur_n, {})
                continue
            m = LINE_RE.match(line)
            if m and cur_n is not None:
                phase = m.group("phase")
                if phase in PHASES:
                    out[cur_n][phase] = float(m.group("median"))
    return out


def aggregate(paths: list[str]) -> dict[int, dict[str, list[float]]]:
    """複数 run 分の parse_log 結果を { N: { phase: [values] } } へ束ねる。"""
    agg: dict[int, dict[str, list[float]]] = {}
    for p in paths:
        parsed = parse_log(p)
        for n, phases in parsed.items():
            agg.setdefault(n, {})
            for phase, val in phases.items():
                agg[n].setdefault(phase, []).append(val)
    return agg


def validate_run_counts(
    label: str, paths: list[str], expect_runs: int, agg: dict[int, dict[str, list[float]]]
) -> list[str]:
    """coding-rust.md「ベンチは 5 回計測の中央値」契約の機械検査。

    `--off-glob`／`--on-glob` が拾ったファイル数が `expect_runs`（既定 5）と
    一致すること（glob の書き方次第で on-clean 系列と汚染 run が混在した
    り、想定より多い／少ないファイルを拾っても無検証で「5 run medians」と
    銘打った表を出力してしまう問題を防ぐ。イシュー #1301 codex-review 指
    摘）に加え、集計後の各 (N, phase) セルのサンプル数がファイル数と一致
    すること（一部の run でパース失敗し欠損した場合の検出）を確認する。
    違反があれば理由の一覧を返す（空リストなら問題なし）。
    """
    errors: list[str] = []
    if len(paths) != expect_runs:
        errors.append(
            f"{label}: {len(paths)} 件のログが glob にマッチしたが "
            f"expect_runs={expect_runs} と不一致（対象ファイル: {paths}）"
        )
    for n, phases in agg.items():
        for phase in PHASES:
            vals = phases.get(phase, [])
            if len(vals) != len(paths):
                errors.append(
                    f"{label}: N={n} phase={phase} のサンプル数が {len(vals)} 件で"
                    f" 読み込んだログ数 {len(paths)} 件と不一致"
                    "（一部ログでパースに失敗した可能性）"
                )
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", required=True)
    ap.add_argument("--off-glob", required=True)
    ap.add_argument("--on-glob", required=True)
    ap.add_argument(
        "--expect-runs",
        type=int,
        default=5,
        help="each glob が拾うべきログ件数（coding-rust.md の 5 回計測契約。既定 5）",
    )
    args = ap.parse_args()

    off_paths = sorted(glob.glob(args.off_glob))
    on_paths = sorted(glob.glob(args.on_glob))
    if not off_paths or not on_paths:
        print(f"ERROR: off={len(off_paths)} on={len(on_paths)} 件（0 件は不可）", file=sys.stderr)
        return 1

    off_agg = aggregate(off_paths)
    on_agg = aggregate(on_paths)

    # 5 回計測契約・パース欠損の検証（イシュー #1301 codex-review 指摘）。
    # 違反時は「5 run medians」と称した表を無検証で出さず fail-closed で
    # エラー終了する。
    errors = validate_run_counts("off", off_paths, args.expect_runs, off_agg)
    errors += validate_run_counts("on", on_paths, args.expect_runs, on_agg)
    if errors:
        print("ERROR: run 数・サンプル数の検証に失敗しました:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"# Layer B 集計（node={args.node}。イシュー #1301）\n")
    print(f"off runs: {off_paths}")
    print(f"on runs: {on_paths}\n")

    sizes = sorted(set(off_agg.keys()) | set(on_agg.keys()))
    for n in sizes:
        print(f"## N={n}\n")
        print("| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |")
        print("|---|---|---|---|---|---|")
        for phase in PHASES:
            off_vals = off_agg.get(n, {}).get(phase, [])
            on_vals = on_agg.get(n, {}).get(phase, [])
            off_med = statistics.median(off_vals) if off_vals else float("nan")
            on_med = statistics.median(on_vals) if on_vals else float("nan")
            ratio = (on_med / off_med) if off_vals and on_vals and off_med != 0 else float("nan")
            print(
                f"| {phase} | {off_med:.4f} | {len(off_vals)} | {on_med:.4f} | {len(on_vals)} | {ratio:.4f} |"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
