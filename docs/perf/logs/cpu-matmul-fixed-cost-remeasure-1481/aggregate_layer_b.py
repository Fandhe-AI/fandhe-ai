#!/usr/bin/env python3
"""Layer B（gemm_reuse_phase_diag_cpu、イシュー #1481（#1301 と同型を再利用））5 run 集計。

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
# 後続行が捨てられ、N=512 が集計表から欠落する（イシュー #1481（#1301 と同型を再利用）
# codex-review 指摘）。行頭固定をやめ、行中のどこかに `N=<n> (median over`
# が現れれば検出する `.search` へ切り替える。
SIZE_RE = re.compile(r"N=(?P<n>\d+)\s*\(median over")

PHASES = ["alloc_c", "kernel", "tensor_wrap", "ops_gemm", "tape_matmul", "to_tensor", "host_copy", "checksum"]

# 本ハーネスが対象とする GEMM ゲート形状（`run_gemm_gate_cpu.sh`・
# `compare_gemm_gate.py --device cpu` と同一の N=512/1024/2048）。
# `validate_run_counts` は agg に実在するサイズしか検査できないため
# （5 ファイルすべてで同一サイズの解析に失敗すると、そのサイズは
# agg に一切現れず検査対象から漏れる。イシュー #1481（#1301 と同型を再利用） codex-review 指
# 摘）、このリストで「集計対象サイズ全体の欠落」を独立に検出する。
EXPECTED_SIZES = [512, 1024, 2048]


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
    銘打った表を出力してしまう問題を防ぐ。イシュー #1481（#1301 と同型を再利用） codex-review 指
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


def validate_expected_sizes(
    label: str, agg: dict[int, dict[str, list[float]]], expected_sizes: list[int]
) -> list[str]:
    """全 `PHASES` が集計から漏れているサイズ（＝5 ファイルすべてで
    そのサイズの解析に失敗し `agg` に一切現れなかったケース）を検出する。

    `validate_run_counts` は `agg.items()`（既に存在するサイズのみ）を
    走査するため、サイズそのものが丸ごと欠落した場合は検査をすり抜けて
    無検証で表から抜け落ちる（イシュー #1481（#1301 と同型を再利用） codex-review 指摘）。
    `EXPECTED_SIZES` と突き合わせることで、この経路を fail-closed に
    塞ぐ。違反があれば理由の一覧を返す（空リストなら問題なし）。
    """
    errors: list[str] = []
    for n in expected_sizes:
        if n not in agg:
            errors.append(
                f"{label}: N={n} が集計から完全に欠落（読み込んだ全ログで"
                "この N の解析に失敗した可能性。SIZE_RE がログ中の "
                "`N=<n> (median over` 行を検出できなかった疑いがある）"
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
    ap.add_argument(
        "--expect-sizes",
        type=str,
        default=",".join(str(n) for n in EXPECTED_SIZES),
        help="集計対象として必須の N（カンマ区切り。既定は GEMM ゲート形状 512,1024,2048）",
    )
    args = ap.parse_args()
    expected_sizes = [int(s) for s in args.expect_sizes.split(",") if s.strip()]

    off_paths = sorted(glob.glob(args.off_glob))
    on_paths = sorted(glob.glob(args.on_glob))
    if not off_paths or not on_paths:
        print(f"ERROR: off={len(off_paths)} on={len(on_paths)} 件（0 件は不可）", file=sys.stderr)
        return 1

    off_agg = aggregate(off_paths)
    on_agg = aggregate(on_paths)

    # 5 回計測契約・パース欠損の検証（イシュー #1481（#1301 と同型を再利用） codex-review 指摘）。
    # 違反時は「5 run medians」と称した表を無検証で出さず fail-closed で
    # エラー終了する。
    errors = validate_run_counts("off", off_paths, args.expect_runs, off_agg)
    errors += validate_run_counts("on", on_paths, args.expect_runs, on_agg)
    # 集計対象サイズ全体の欠落検証（5 ファイルすべてでの解析失敗を検出。
    # イシュー #1481（#1301 と同型を再利用） codex-review 指摘。上記 validate_run_counts は
    # agg に実在するサイズしか走査できないため、この検査で補完する）。
    errors += validate_expected_sizes("off", off_agg, expected_sizes)
    errors += validate_expected_sizes("on", on_agg, expected_sizes)
    if errors:
        print("ERROR: run 数・サンプル数の検証に失敗しました:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(f"# Layer B 集計（node={args.node}。イシュー #1481（#1301 と同型を再利用））\n")
    print(f"off runs: {off_paths}")
    print(f"on runs: {on_paths}\n")

    sizes = sorted(set(off_agg.keys()) | set(on_agg.keys()))
    for n in sizes:
        print(f"## N={n}\n")
        print("| phase | off median (of 5 run medians, ms) | off n | on median (ms) | on n | on/off 比 |")
        print("|---|---|---|---|---|---|")
        # PR #1501 codex-review P2 是正: 丸めなし raw コメント（後述）を
        # 各行の直後へインラインで出力すると、GitHub の Markdown テーブルは
        # `|` で始まらない行（コメント行）に遭遇した時点で終了してしまい、
        # それ以降の行（同じテーブルの残り phase 行）が表として描画されず
        # 生テキストのまま表示されてしまう（イシュー #1481 codex-review
        # 指摘。`aggregate-dgx.md`／`aggregate-m4max.md` の各 N セクション
        # で alloc_c 行の直後に表が壊れて見える形で実際に発生していた）。
        # `judge_rules.py::parse_layer_b_md` の `LAYER_B_RAW_RE` はコメント
        # 自身が `N=`／`phase=` を保持しており位置に依存しないため、
        # 全 phase 行を出し切ってからコメントをまとめて表の後ろへ出力する
        # よう変更する（機械可読性は不変・人間可読の表だけが直る）。
        raw_comments: list[str] = []
        for phase in PHASES:
            off_vals = off_agg.get(n, {}).get(phase, [])
            on_vals = on_agg.get(n, {}).get(phase, [])
            off_med = statistics.median(off_vals) if off_vals else float("nan")
            on_med = statistics.median(on_vals) if on_vals else float("nan")
            ratio = (on_med / off_med) if off_vals and on_vals and off_med != 0 else float("nan")
            print(
                f"| {phase} | {off_med:.4f} | {len(off_vals)} | {on_med:.4f} | {len(on_vals)} | {ratio:.4f} |"
            )
            # 上記テーブル列は人間可読性のための表示丸め（小数 4 桁）で
            # あり、`judge_rules.py` の規則 2（DGX N=2048 決定セル。
            # `RULE2_OPS_GEMM_THRESHOLD=1.00` を厳密な `<=`／`<` で判定
            # する）がこの丸め値をそのまま比較に使うと、真の比が
            # 1.00004 のように閾値をわずかに超えていても表示は 1.0000 に
            # 丸まり誤って通過（逆に 0.99996 は誤って棄却）しうる欠陥が
            # あった。`repr()` は Python の float が同じ 64 bit 表現へ
            # 丸め落ちなく再構成できる最短の文字列表現（round-trip 保証）
            # を返すため、このコメントを機械可読な形で保持し
            # `judge_rules.py::parse_layer_b_md` が丸めなしの値を優先して
            # 読み取れるようにする（表示用の `.4f` テーブル自体は変更
            # しない＝丸めは表示専用に限定）。`off_vals`／`on_vals` が空
            # （=ratio が nan）の場合は追加しない（`parse_layer_b_md` 側で
            # "nan" 文字列を特別扱いする必要を避けるフェイルセーフ）。
            if off_vals and on_vals and off_med != 0:
                raw_comments.append(
                    f"<!-- raw N={n} phase={phase} off_med={off_med!r} "
                    f"on_med={on_med!r} ratio={ratio!r} -->"
                )
        if raw_comments:
            print()
            for comment in raw_comments:
                print(comment)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
