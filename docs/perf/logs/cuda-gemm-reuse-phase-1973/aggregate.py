#!/usr/bin/env python3
"""イシュー #1973 実測ログ集計スクリプト（python3 標準ライブラリのみ）。

`orchestrate.sh` が生成する `layerA-phases-N{1024,2048,4096}.log`
（bench-fandhe の `--task gemm --mode reuse --phases` JSONL 出力を
5 run 分連結したもの）から、フェーズ別の 5 run 中央値と
`iter_total` に対する比率を算出し Markdown 表として出力する。
#1182（`docs/perf/cuda-gemm-reuse-phase-1182/`）の §3・§6 と同じ
集計方式（各 run の `median_s` を集め、run 間でさらに中央値を取る）。

**完全性検査（fail-closed）**: 各対象サイズについて `-- run i/N=n --`
区切りで期待 run 数（既定 5）が揃っていること、各 run に全 phase
（`matmul`／`to_tensor`／`host_copy`／`checksum`／`iter_total`）が
重複なく含まれていることを検証する。壊れた JSON 行・欠落・重複・
run 数不足を検出した場合は、その行を黙って除外せず
`LogIntegrityError` を送出し、正式な集計値（Markdown 表）を一切
出力せず非ゼロ終了する（ログ未生成〈ファイル自体が存在しない〉は
実測前の正常な中間状態のため区別してエラー扱いしない）。

`--self-test` で固定 fixture（正常系・不完全 run 数・壊れた JSON・
phase 重複の各ケース）に対する自己検証を行う（実機不要。Linux CI
で実行可能）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

PHASES = ("matmul", "to_tensor", "host_copy", "checksum", "iter_total")


class LogIntegrityError(Exception):
    """ログの完全性検査（run 数・phase 網羅性・重複・JSON 妥当性）に
    失敗したことを表す。呼び出し側はこの例外を捕捉して正式な集計値を
    一切出力せず非ゼロ終了する（fail-closed。黙って除外しない）。
    """


def parse_phase_log(text: str, expected_runs: int = 5) -> dict[str, list[float]]:
    """`layerA-phases-N*.log` のテキストを `-- run i/N=n --` 区切りで
    run 単位のブロックへ分割し、各ブロックが全 phase を重複なく
    含むこと・ブロック総数が `expected_runs` と一致することを検証
    したうえで phase 別 median_s 列を返す。

    検証に失敗した場合は `LogIntegrityError` を送出する（フェーズ名の
    typo・`median_s` 欠落・壊れた JSON・run 内重複・run 数不一致を
    すべて対象とする）。
    """
    blocks: list[dict[str, float]] = []
    current: dict[str, float] = {}
    started = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("--"):
            # `-- run i/N=n --` 形式の区切り行。新しい run ブロックを開始する。
            if started:
                blocks.append(current)
            current = {}
            started = True
            continue
        if not started:
            raise LogIntegrityError(
                f"run 区切り行（'-- run i/N=n --'）より前にデータ行がある: {line!r}"
            )
        if not line.startswith("{"):
            raise LogIntegrityError(f"認識できない行形式（JSON でも区切り行でもない）: {line!r}")
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LogIntegrityError(f"壊れた JSON 行: {line!r} ({exc})") from exc
        phase = rec.get("phase")
        if phase not in PHASES:
            raise LogIntegrityError(f"未知または欠落した phase: {phase!r}（行: {line!r}）")
        if "median_s" not in rec:
            raise LogIntegrityError(f"median_s キーが欠落: {line!r}")
        if phase in current:
            raise LogIntegrityError(f"同一 run 内で phase '{phase}' が重複している: {line!r}")
        current[phase] = float(rec["median_s"])

    if started:
        blocks.append(current)

    if len(blocks) != expected_runs:
        raise LogIntegrityError(
            f"run 数が期待値と不一致（期待 {expected_runs}・実際 {len(blocks)}）"
        )

    for i, block in enumerate(blocks, start=1):
        missing = set(PHASES) - set(block.keys())
        if missing:
            raise LogIntegrityError(f"run {i} に phase が欠落している: {sorted(missing)}")

    out: dict[str, list[float]] = {p: [] for p in PHASES}
    for block in blocks:
        for p in PHASES:
            out[p].append(block[p])
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
    parser.add_argument(
        "--expected-runs", type=int, default=5, help="1 サイズあたりの期待 run 数（既定 5）"
    )
    parser.add_argument("--self-test", action="store_true", help="固定 fixture で自己検証する")
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    log_dir = Path(args.log_dir)
    md_parts = ["# イシュー #1973 Layer A 集計（自動生成）", ""]
    any_found = False
    errors: list[str] = []
    for n in args.sizes:
        log_path = log_dir / f"layerA-phases-N{n}.log"
        if not log_path.exists():
            md_parts.append(f"### N={n}\n\n(ログ未生成: {log_path.name})\n")
            continue
        any_found = True
        try:
            phase_medians = parse_phase_log(log_path.read_text(), args.expected_runs)
        except LogIntegrityError as exc:
            errors.append(f"N={n}（{log_path.name}）: {exc}")
            continue
        md_parts.append(render_table(n, phase_medians))

    if errors:
        # fail-closed: 完全性検査に失敗したログが 1 件でもあれば、
        # 正式な集計値（Markdown 表）を一切出力せず非ゼロ終了する
        # （黙って当該サイズだけ除外して残りを正常出力しない）。
        print("ERROR: ログの完全性検査に失敗した（正式な集計値は出力しない）", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("\n".join(md_parts))
    if not any_found:
        print(
            "警告: 対象ログが 1 件も見つからなかった（未実測）。"
            " orchestrate.sh を実行してから再度呼ぶこと。",
            file=sys.stderr,
        )
    return 0


def run_self_test() -> int:
    """固定 fixture で以下を検証する（実機ログ不要）:

    1. 正常系（2 run・全 phase 揃い）で median 計算・比率算出が正しいこと
    2. run 数不足（期待 3・実際 2）が `LogIntegrityError` になること
    3. run 内で phase が欠落している場合に `LogIntegrityError` になること
    4. 壊れた JSON 行（構文エラー）が `LogIntegrityError` になること
    5. 同一 run 内で phase が重複している場合に `LogIntegrityError` になること
    6. 未生成ログ（空辞書。ファイル自体が存在しない場合の呼び出し元処理と
       同じ入力）が例外を投げず `render_table` できること
    """

    def make_full_run() -> list[str]:
        return [
            json.dumps({"phase": "matmul", "median_s": 0.001}),
            json.dumps({"phase": "to_tensor", "median_s": 0.0}),
            json.dumps({"phase": "host_copy", "median_s": 0.002}),
            json.dumps({"phase": "checksum", "median_s": 0.0005}),
            json.dumps({"phase": "iter_total", "median_s": 0.0035}),
        ]

    # 1. 正常系: 2 run とも全 phase 揃い
    fixture_ok = "\n".join(
        ["-- run 1/N=1024 --", *make_full_run(), "-- run 2/N=1024 --", *make_full_run()]
    )
    parsed = parse_phase_log(fixture_ok, expected_runs=2)
    assert parsed["matmul"] == [0.001, 0.001], parsed["matmul"]
    assert abs(median_of(parsed["matmul"]) - 0.001) < 1e-12
    assert abs(median_of(parsed["iter_total"]) - 0.0035) < 1e-12
    table = render_table(1024, parsed)
    assert "N=1024" in table
    assert "matmul" in table

    # 2. run 数不足（期待 3 だが実際は 2 run しかない）
    try:
        parse_phase_log(fixture_ok, expected_runs=3)
        raise AssertionError("run 数不足が検出されなかった")
    except LogIntegrityError:
        pass

    # 3. run 内で phase が欠落（iter_total 行を除去）
    incomplete_run = [
        line for line in make_full_run() if json.loads(line)["phase"] != "iter_total"
    ]
    fixture_missing_phase = "\n".join(["-- run 1/N=1024 --", *incomplete_run])
    try:
        parse_phase_log(fixture_missing_phase, expected_runs=1)
        raise AssertionError("phase 欠落が検出されなかった")
    except LogIntegrityError:
        pass

    # 4. 壊れた JSON 行（末尾の閉じ括弧を欠落させる）
    fixture_broken_json = "\n".join(
        ["-- run 1/N=1024 --", '{"phase": "matmul", "median_s": 0.001', *make_full_run()[1:]]
    )
    try:
        parse_phase_log(fixture_broken_json, expected_runs=1)
        raise AssertionError("壊れた JSON 行が検出されなかった")
    except LogIntegrityError:
        pass

    # 5. 同一 run 内で phase が重複
    fixture_dup_phase = "\n".join(
        [
            "-- run 1/N=1024 --",
            *make_full_run(),
            json.dumps({"phase": "matmul", "median_s": 0.999}),
        ]
    )
    try:
        parse_phase_log(fixture_dup_phase, expected_runs=1)
        raise AssertionError("phase 重複が検出されなかった")
    except LogIntegrityError:
        pass

    # 6. 未生成ログ（空辞書）でも例外を投げないこと
    empty_table = render_table(2048, {p: [] for p in PHASES})
    assert "データなし" in empty_table

    print("self-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
