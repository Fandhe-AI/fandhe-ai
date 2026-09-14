#!/usr/bin/env python3
"""イシュー #1590: `gemm_transposed_perf`（§3.2 の正式補助 A/B。5 プロセス
起動）のログを集計し、形状ごとの speedup 5 run 中央値表（markdown）を
出力する。

`run_ignored_tests.sh` が生成する
`docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/aux/
gemm_transposed_perf_run{1..5}.log` を入力とする。

python3 標準ライブラリのみ（他の集計スクリプト群と同方針。追加依存
なしで表記トリック迂回を遮断する必要はここでは薄いが、一貫性のため）。

行フォーマット（`crates/backend-cuda/tests/gemm_transposed_perf.rs` の
出力。`docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/
gemm_transposed_perf.log` で実測確認済み）:

    test nt_transposed_entry_vs_contiguous_across_shapes ... NT m=64 k=784 n=256 before_median_s=0.000439 after_median_s=0.000079 speedup=5.563x
    NT m=64 k=256 n=10 before_median_s=0.000038 after_median_s=0.000036 speedup=1.057x
    ...

1 つ目の形状行は `test ... ...` と同一行に続き、以降の形状行は単独行に
なる。`re.search`（行頭アンカーなし）でどちらの形も拾う。
"""
from __future__ import annotations

import argparse
import io
import re
import statistics
import sys
from pathlib import Path

SHAPE_RE = re.compile(
    r"(?P<pattern>NT|TN) m=(?P<m>\d+) k=(?P<k>\d+) n=(?P<n>\d+) "
    r"before_median_s=(?P<before>[0-9.]+) after_median_s=(?P<after>[0-9.]+) "
    r"speedup=(?P<speedup>[0-9.]+)x"
)

# `crates/backend-cuda/tests/gemm_transposed_perf.rs` の
# `nt_transposed_entry_vs_contiguous_across_shapes`／
# `tn_transposed_entry_vs_contiguous_across_shapes` が対象とする形状
# （各 4 形状 × NT/TN の 2 パターン = 8 形状。1 プロセス起動につき各
# 形状はちょうど 1 回だけ出力される）。正式な 5 プロセス起動中央値
# 集計の入力検証に使う（codex-review 指摘・PR #1812）。
EXPECTED_SHAPES: frozenset[tuple[str, int, int, int]] = frozenset(
    {
        ("NT", 64, 784, 256),
        ("NT", 64, 256, 10),
        ("NT", 1024, 1024, 1024),
        ("NT", 2048, 2048, 2048),
        ("TN", 64, 784, 256),
        ("TN", 64, 256, 10),
        ("TN", 1024, 1024, 1024),
        ("TN", 2048, 2048, 2048),
    }
)
EXPECTED_RUNS = 5


def parse_log(text: str) -> list[dict]:
    """1 プロセス起動分のログ本文から形状行をすべて抽出する。"""
    rows = []
    for line in text.splitlines():
        m = SHAPE_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "pattern": m.group("pattern"),
                "m": int(m.group("m")),
                "k": int(m.group("k")),
                "n": int(m.group("n")),
                "before_s": float(m.group("before")),
                "after_s": float(m.group("after")),
                "speedup": float(m.group("speedup")),
            }
        )
    return rows


def aggregate(run_texts: list[str]) -> list[dict]:
    """複数起動分のログをまとめ、形状（pattern, m, k, n）ごとに
    before/after/speedup の中央値を計算する。順序は最初に出現した順を
    保つ（`dict` のキー挿入順に依存。Python 3.7+ の仕様どおり）。"""
    by_shape: dict[tuple, dict] = {}
    for text in run_texts:
        for row in parse_log(text):
            key = (row["pattern"], row["m"], row["k"], row["n"])
            entry = by_shape.setdefault(
                key, {"before": [], "after": [], "speedup": []}
            )
            entry["before"].append(row["before_s"])
            entry["after"].append(row["after_s"])
            entry["speedup"].append(row["speedup"])
    out = []
    for (pattern, m, k, n), entry in by_shape.items():
        out.append(
            {
                "pattern": pattern,
                "m": m,
                "k": k,
                "n": n,
                "n_runs": len(entry["speedup"]),
                "before_median_s": statistics.median(entry["before"]),
                "after_median_s": statistics.median(entry["after"]),
                "speedup_median": statistics.median(entry["speedup"]),
            }
        )
    return out


class ValidationError(ValueError):
    """`validate_run_texts` が不完全・重複入力を検出した場合に送出する。"""


def validate_run_texts(
    run_texts: list[str], expected_runs: int = EXPECTED_RUNS
) -> None:
    """正式な 5 プロセス起動中央値集計としての完全性を検証する
    （codex-review 指摘・PR #1812）。中断ログ（形状行が 8 件未満）・
    重複指定（同一ログの二重読み込み・同一形状が同一起動内で複数回
    出現）から不完全な中央値を正式集計として出力しないよう、以下を
    fail-closed で検査する:

    - 起動数（入力ログ数）が `expected_runs` と一致すること
    - 各起動（各ログ）が `EXPECTED_SHAPES` の 8 形状をちょうど 1 件ずつ
      含むこと（不足・重複・未知形状のいずれも許容しない）

    違反時は `ValidationError` を送出する（呼び出し側で非 0 終了させる）。
    """
    if len(run_texts) != expected_runs:
        raise ValidationError(
            f"入力ログ数が {expected_runs} 件ではない（実際: {len(run_texts)} 件）。"
            "正式な 5 プロセス起動中央値集計には gemm_transposed_perf_run{1..5}.log の"
            "ちょうど 5 件を指定すること（中断・重複指定を検出するための検査）。"
        )
    for run_idx, text in enumerate(run_texts, start=1):
        rows = parse_log(text)
        seen: dict[tuple[str, int, int, int], int] = {}
        for row in rows:
            key = (row["pattern"], row["m"], row["k"], row["n"])
            seen[key] = seen.get(key, 0) + 1
        duplicates = {k: c for k, c in seen.items() if c > 1}
        if duplicates:
            raise ValidationError(
                f"run{run_idx}: 同一形状が複数回出現している（重複指定・ログ二重連結の"
                f"疑い）: {sorted(duplicates)}"
            )
        missing = EXPECTED_SHAPES - set(seen)
        if missing:
            raise ValidationError(
                f"run{run_idx}: 期待する 8 形状のうち {len(missing)} 件が不足している"
                f"（中断ログの疑い）: {sorted(missing)}"
            )
        unknown = set(seen) - EXPECTED_SHAPES
        if unknown:
            raise ValidationError(
                f"run{run_idx}: 未知の形状が含まれている"
                f"（`gemm_transposed_perf.rs` の形状定義とズレている疑い）: {sorted(unknown)}"
            )


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "| パターン | m | k | n | before 中央値 (s) | after 中央値 (s) | 倍率中央値 | n_runs |",
        "|----------|---|---|---|-------------------|-------------------|------------|--------|",
    ]
    for row in rows:
        lines.append(
            "| {pattern} | {m} | {k} | {n} | {before_median_s:.6f} | "
            "{after_median_s:.6f} | {speedup_median:.3f}x | {n_runs} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def _full_run_fixture(run_i: int) -> str:
    """`EXPECTED_SHAPES` の 8 形状すべてを含む 1 起動分のログ断片を
    生成する（`validate_run_texts` の正常系フィクスチャ）。"""
    lines = ["running 8 tests"]
    first = True
    for pattern, m, k, n in sorted(EXPECTED_SHAPES):
        prefix = (
            f"test {pattern.lower()}_transposed_entry_vs_contiguous_across_shapes ... "
            if first
            else ""
        )
        first = False
        lines.append(
            f"{prefix}{pattern} m={m} k={k} n={n} "
            f"before_median_s=0.000439 after_median_s=0.000079 speedup=5.5{run_i}3x"
        )
    lines.append("ok")
    return "\n".join(lines) + "\n"


def _self_test() -> int:
    # 実ログの断片を模したフィクスチャ（1 つ目の形状行が `test ...` と
    # 同一行に続く実フォーマットを含む。5 run 分・NT 1 形状のみで検証）。
    fixture_runs = [
        (
            "running 2 tests\n"
            "test nt_transposed_entry_vs_contiguous_across_shapes ... "
            "NT m=64 k=784 n=256 before_median_s=0.000439 after_median_s=0.000079 speedup=5.5{i}3x\n"
            "ok\n"
        ).format(i=i)
        for i in range(5)
    ]
    rows = aggregate(fixture_runs)
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["pattern"] == "NT"
    assert row["m"] == 64 and row["k"] == 784 and row["n"] == 256
    assert row["n_runs"] == 5
    expected_speedups = [5.5 + 0.001 * i + 0.03 for i in range(5)]
    # 上の speedup 文字列は "5.5{i}3x" -> 5.503, 5.513, 5.523, 5.533, 5.543
    expected_speedups = [5.503, 5.513, 5.523, 5.533, 5.543]
    assert abs(row["speedup_median"] - statistics.median(expected_speedups)) < 1e-9
    md = render_markdown(rows)
    assert "NT" in md and "5.523x" in md

    # `validate_run_texts` の検査（codex-review 指摘・PR #1812）。
    # 正常系: 8 形状 × 5 run が揃っていれば例外を送出しない。
    full_runs = [_full_run_fixture(i) for i in range(5)]
    validate_run_texts(full_runs)

    # 異常系 1: 起動数が 5 でない（中断・過不足）。
    try:
        validate_run_texts(full_runs[:4])
        raise AssertionError("expected ValidationError for run-count mismatch")
    except ValidationError:
        pass

    # 異常系 2: 1 run の形状が 1 件欠落している（中断ログの疑い）。
    truncated_runs = list(full_runs)
    truncated_lines = truncated_runs[0].splitlines()
    truncated_runs[0] = "\n".join(
        line for line in truncated_lines if "m=2048 k=2048 n=2048" not in line
    )
    try:
        validate_run_texts(truncated_runs)
        raise AssertionError("expected ValidationError for missing shape")
    except ValidationError:
        pass

    # 異常系 3: 1 run 内で同一形状が重複している（重複指定・二重連結の
    # 疑い）。
    duplicated_runs = list(full_runs)
    duplicated_runs[0] = duplicated_runs[0] + duplicated_runs[0]
    try:
        validate_run_texts(duplicated_runs)
        raise AssertionError("expected ValidationError for duplicated shape")
    except ValidationError:
        pass

    print("self-test OK")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs", nargs="*", type=Path, help="gemm_transposed_perf_run*.log のパス"
    )
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--expect-runs",
        type=int,
        default=EXPECTED_RUNS,
        help=f"正式集計として要求する起動数（既定 {EXPECTED_RUNS}）",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help=(
            "起動数・8 形状完全性の検査を無効化する（診断・部分ログの"
            "確認専用。正式な §3.2 補助 A/B 表の生成には使わないこと）"
        ),
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _self_test()

    if not args.logs:
        parser.error("少なくとも 1 つのログファイルを指定するか --self-test を使うこと")

    run_texts = []
    for path in args.logs:
        run_texts.append(path.read_text(encoding="utf-8"))

    if not args.skip_validation:
        try:
            validate_run_texts(run_texts, expected_runs=args.expect_runs)
        except ValidationError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    rows = aggregate(run_texts)
    if not rows:
        print("warning: 形状行が 1 件も抽出できなかった", file=sys.stderr)
    sys.stdout.write(render_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
