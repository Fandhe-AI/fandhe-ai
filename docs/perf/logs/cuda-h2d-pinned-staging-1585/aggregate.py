#!/usr/bin/env python3
"""イシュー #1585 Layer B（H2D 単体マイクロ A/B）集計（python3 標準
ライブラリのみ。`.claude/rules/deps-policy.md` の依存方針・既存 A/B
集計スクリプト群と同じ方針）。

`crates/backend-cuda/tests/pinned_h2d_upload_ab_1585.rs` の 5 プロセス
起動出力（`layer_b_run{1..5}.log`）から CSV 行
（`phase,n,bytes_mib,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms`）を
抽出し、`docs/perf/cuda-h2d-pinned-staging.md` §3 の事前登録規則
「H2D 単体（発行＋`synchronize`）で N ごとに 5 プロセス起動中央値の
`pinned_staged/pageable`。`< 1.00` を改善、`>= 1.00` を非改善として
記録」に従って `aggregate.md` を書き出す。

`docs/perf/logs/cuda-host-view-staging-1336/aggregate.py`
（D2H 側の対称）・`docs/perf/logs/metal-mse-backward-1691/aggregate.py`
（`--self-test` 付き集計の先例）と同型の設計:
事前登録した全セル（PHASES × SIZES_N）× 全 run（既定 5 run）の記録が
過不足なく揃っていることを検査してから集計する（fail-closed。1 件でも
欠落・重複があれば `undetermined` として非 0 終了する）。加えて各 run の
`median_ms` は中央値算出前に「有限かつ正数」（`math.isfinite` かつ
`> 0`）であることを検査し、`inf`／`nan`／0／負数が 1 件でもあれば同じく
`undetermined` として集計を中止する（PR #1907 codex-review P2 指摘。
`pageable` が `inf` で `pinned_staged` が有限正数の記録では比率が 0 と
なり「改善」と誤判定されるため、集計前に fail-closed で遮断する）。
"""

from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADER = "phase,n,bytes_mib,cold_ms,min_ms,q1_ms,median_ms,q3_ms,max_ms"
# `pinned_h2d_upload_ab_1585.rs::SIZES_N`（1024/2048/4096）と同一。
SIZES_N = ("1024", "2048", "4096")
# `pinned_h2d_upload_ab_1585.rs` が出力する 2 腕（`pageable`／
# `pinned_staged`）。D2H 側（#1336）の `before`／`after_pageable`／
# `after_pinned` の 3 系列とは異なり、本イシューは H2D 単体のため
# 2 系列のみ。
PHASES = ("pageable", "pinned_staged")
EXPECTED_KEYS = frozenset((phase, n) for phase in PHASES for n in SIZES_N)


class AggregateError(Exception):
    """集計前検証（欠落・重複セル・非有限／非正の計測値）に失敗した場合に
    送出する。"""


def validate_median_ms(log_path: Path, phase: str, n: str, raw: str) -> float:
    """CSV 行の `median_ms` を有限正数として検証し `float` で返す。

    `float()` で解釈できない文字列・`inf`／`nan`・0 以下はいずれも
    5 run 中央値と `pinned_staged/pageable` 比の前提を崩す（`inf` を含む
    系列の比は 0 や `nan` になり、判定列が意味を失う）ため、検出した
    時点で `AggregateError` を送出し集計全体を中止する（PR #1907
    codex-review P2 指摘の是正）。
    """
    try:
        value = float(raw)
    except ValueError as e:
        raise AggregateError(
            f"{log_path}: phase={phase} N={n} の median_ms を数値として解釈できません: {raw!r}"
        ) from e
    if not (math.isfinite(value) and value > 0.0):
        raise AggregateError(
            f"{log_path}: phase={phase} N={n} の median_ms が非有限または非正の値です: {raw!r}"
        )
    return value


def extract_rows(log_path: Path) -> list[list[str]]:
    rows = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        for phase in PHASES:
            if line.startswith(phase + ","):
                rows.append(line.split(","))
                break
    return rows


def validate_run_log(log_path: Path, rows: list[list[str]]) -> None:
    """1 run ログが (phase, n) の全組み合わせをちょうど 1 回ずつ持つことを
    検査する（`host_view_staging_readout_ab_1336/aggregate.py::
    validate_run_log` と同型。欠落・重複のいずれも 5 run 中央値の前提を
    崩すため、検出した時点で集計全体を中止する）。
    """
    seen: dict[tuple[str, str], int] = {}
    for row in rows:
        phase, n = row[0], row[1]
        seen[(phase, n)] = seen.get((phase, n), 0) + 1

    duplicated = {key: count for key, count in seen.items() if count > 1}
    if duplicated:
        raise AggregateError(
            f"{log_path}: 重複した (phase, n) 行を検出: {sorted(duplicated)}"
        )

    missing = EXPECTED_KEYS - seen.keys()
    if missing:
        raise AggregateError(f"{log_path}: 欠落した (phase, n) 行を検出: {sorted(missing)}")

    unexpected = seen.keys() - EXPECTED_KEYS
    if unexpected:
        raise AggregateError(f"{log_path}: 未知の (phase, n) 行を検出: {sorted(unexpected)}")


def run_aggregate(logdir: Path, n_runs: int = 5, out=sys.stdout) -> bool:
    """1 件でも検証エラーがあれば False を返す（`--self-test` から標準
    出力を汚さずに呼べるように出力先を引数化する。`metal-mse-backward-
    1691/aggregate.py::run_aggregate` と同型）。
    """
    run_logs = [logdir / f"layer_b_run{i}.log" for i in range(1, n_runs + 1)]

    by_key: dict[tuple[str, str], list[float]] = {}
    bytes_mib_by_n: dict[str, str] = {}

    for log_path in run_logs:
        if not log_path.exists():
            print(f"ERROR: missing run log: {log_path}", file=out)
            return False
        rows = extract_rows(log_path)
        try:
            validate_run_log(log_path, rows)
            for row in rows:
                phase, n, bytes_mib, _cold, _min, _q1, median_ms, _q3, _max = row
                # 中央値算出前に各 run の値を有限正数として検証する
                # （不正値が 1 件でもあれば集計を中止する）。
                value = validate_median_ms(log_path, phase, n, median_ms)
                by_key.setdefault((phase, n), []).append(value)
                bytes_mib_by_n[n] = bytes_mib
        except AggregateError as e:
            print(f"ERROR: {e}", file=out)
            return False

    lines = []
    lines.append("# pinned_h2d_upload_ab_1585 集計（5 run 中央値）")
    lines.append("")
    lines.append(
        "各セルは `median_ms`（1 run あたり 20/20 warmup+計測の中央値）を"
        f" {n_runs} プロセス起動ぶん集めた系列の中央値。`pageable` はフラグ"
        " OFF（`clone_htod` 直呼び経路）、`pinned_staged` はフラグ ON"
        "（`H2dStagingCache` 経由の pinned ステージング）。いずれも"
        "「upload 発行 → `stream.synchronize()`」を 1 反復として計測する"
        "（`docs/perf/cuda-h2d-pinned-staging.md` §3）。"
    )
    lines.append("")
    lines.append(
        "| N | bytes(MiB) | pageable median_ms (5run) | pinned_staged"
        " median_ms (5run) | pinned_staged/pageable | 判定 |"
    )
    lines.append("|---|---|---|---|---|---|")

    all_improved = True
    for n in SIZES_N:
        pageable_vals = by_key[("pageable", n)]
        pinned_vals = by_key[("pinned_staged", n)]
        if len(pageable_vals) != n_runs or len(pinned_vals) != n_runs:
            print(
                f"ERROR: N={n} のサンプル数が n_runs={n_runs} と一致しません "
                f"(pageable={len(pageable_vals)}, pinned_staged={len(pinned_vals)})",
                file=out,
            )
            return False
        pageable_med = statistics.median(pageable_vals)
        pinned_med = statistics.median(pinned_vals)
        # 規則: `pinned_staged/pageable` が `< 1.00` なら改善、`>= 1.00`
        # なら非改善として記録する。各 run の値は `validate_median_ms` で
        # 有限正数と検証済みのため中央値も有限正数であり、ここでの
        # `pageable_med > 0` ガードは多重防御（万一 `nan` になっても
        # Python の比較規則により NaN < 1.00 は False で非改善扱い）。
        ratio = pinned_med / pageable_med if pageable_med > 0 else float("nan")
        improved = ratio < 1.00
        if not improved:
            all_improved = False
        verdict = "改善" if improved else "非改善"
        lines.append(
            f"| {n} | {bytes_mib_by_n[n]} | {pageable_med:.4f}"
            f" (n={len(pageable_vals)}) | {pinned_med:.4f}"
            f" (n={len(pinned_vals)}) | {ratio:.3f}x | {verdict} |"
        )

    lines.append("")
    lines.append(
        "`pinned_staged/pageable < 1.00` を改善、`>= 1.00` を非改善として"
        "記録する規則（`docs/perf/cuda-h2d-pinned-staging.md` §3）。"
        "本規則は Layer B（改善根拠）単体の判定であり、Layer A"
        "（framework-compare 非後退）・ゲート A（`#[ignore]` 実機テスト"
        "全件 pass）と合わせて総合判断する。"
    )
    lines.append("")
    lines.append(f"全 N で改善（pinned_staged/pageable < 1.00）: {all_improved}")

    out_path = logdir / "aggregate.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_path, file=out)
    return True


def main(logdir: str, n_runs: int = 5) -> None:
    ok = run_aggregate(Path(logdir), n_runs)
    if not ok:
        sys.exit(1)


# --- self-test（実機ログなしで検証ロジック自体を確認する。fixture は
# 一時ディレクトリへ書き出して run_aggregate() を直接呼ぶ。
# `metal-mse-backward-1691/aggregate.py::self_test` と同型の設計） ---


def _write_run_log(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """`rows` は `(phase, n, median_ms)` のリスト。CSV 行の他フィールド
    （`bytes_mib`／`cold_ms`／`min_ms`／`q1_ms`／`q3_ms`／`max_ms`）は
    集計対象外のため固定のダミー値を埋める。"""
    lines = [HEADER]
    for phase, n, median_ms in rows:
        lines.append(f"{phase},{n},1.0000,1.0000,1.0000,1.0000,{median_ms},1.0000,1.0000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture_rows(pageable_ms: float, pinned_ms: float) -> list[tuple[str, str, str]]:
    rows = []
    for n in SIZES_N:
        rows.append(("pageable", n, f"{pageable_ms:.4f}"))
        rows.append(("pinned_staged", n, f"{pinned_ms:.4f}"))
    return rows


def self_test() -> None:
    import io
    import tempfile

    # --- fixture 1: 正常系（全セル・全 run 揃い・pinned_staged が改善）。
    with tempfile.TemporaryDirectory() as d:
        logdir = Path(d)
        for run in range(1, 6):
            _write_run_log(
                logdir / f"layer_b_run{run}.log",
                _fixture_rows(pageable_ms=10.0 + run * 0.1, pinned_ms=5.0 + run * 0.1),
            )
        buf = io.StringIO()
        ok = run_aggregate(logdir, 5, out=buf)
        assert ok is True, f"normal fixture should pass: {buf.getvalue()}"
        aggregate_md = (logdir / "aggregate.md").read_text(encoding="utf-8")
        assert "全 N で改善（pinned_staged/pageable < 1.00）: True" in aggregate_md

    # --- fixture 2: 非改善（pinned_staged が pageable と同じか遅い）。
    with tempfile.TemporaryDirectory() as d:
        logdir = Path(d)
        for run in range(1, 6):
            _write_run_log(
                logdir / f"layer_b_run{run}.log",
                _fixture_rows(pageable_ms=5.0, pinned_ms=6.0),
            )
        buf = io.StringIO()
        ok = run_aggregate(logdir, 5, out=buf)
        # 非改善でも集計自体は成功する（`run_aggregate` の戻り値は
        # 「集計が実行できたか」を表し「改善したか」ではない。改善可否は
        # `aggregate.md` 内の判定列・末尾サマリを見る）。
        assert ok is True, f"regression fixture should still aggregate: {buf.getvalue()}"
        aggregate_md = (logdir / "aggregate.md").read_text(encoding="utf-8")
        assert "全 N で改善（pinned_staged/pageable < 1.00）: False" in aggregate_md
        assert "非改善" in aggregate_md

    # --- fixture 3: run ログ欠落（5 run のうち 1 本が存在しない）。
    with tempfile.TemporaryDirectory() as d:
        logdir = Path(d)
        for run in range(1, 5):
            _write_run_log(
                logdir / f"layer_b_run{run}.log",
                _fixture_rows(pageable_ms=5.0, pinned_ms=4.0),
            )
        buf = io.StringIO()
        ok = run_aggregate(logdir, 5, out=buf)
        assert ok is False, "missing run log fixture should fail (undetermined)"
        assert "missing run log" in buf.getvalue()

    # --- fixture 4: 欠落セル（1 run で N=4096/pinned_staged が欠落）。
    with tempfile.TemporaryDirectory() as d:
        logdir = Path(d)
        for run in range(1, 6):
            rows = _fixture_rows(pageable_ms=5.0, pinned_ms=4.0)
            if run == 3:
                rows = [r for r in rows if not (r[0] == "pinned_staged" and r[1] == "4096")]
            _write_run_log(logdir / f"layer_b_run{run}.log", rows)
        buf = io.StringIO()
        ok = run_aggregate(logdir, 5, out=buf)
        assert ok is False, "missing-cell fixture should fail"
        assert "欠落した" in buf.getvalue()

    # --- fixture 5: 重複行（同一ログ内で同じ (phase, n) が 2 回出現）。
    with tempfile.TemporaryDirectory() as d:
        logdir = Path(d)
        for run in range(1, 6):
            rows = _fixture_rows(pageable_ms=5.0, pinned_ms=4.0)
            if run == 1:
                rows = rows + [rows[0]]
            _write_run_log(logdir / f"layer_b_run{run}.log", rows)
        buf = io.StringIO()
        ok = run_aggregate(logdir, 5, out=buf)
        assert ok is False, "duplicate-key fixture should fail"
        assert "重複した" in buf.getvalue()

    # --- fixture 6〜8: 非有限・非正の median_ms（PR #1907 codex-review P2）。
    # `pageable` が `inf`（比率が 0 となり「改善」と誤判定される入力）、
    # `pinned_staged` が負数、いずれかが 0 のそれぞれで集計が中止される
    # ことを確認する。
    for label, target_phase, bad_value in (
        ("inf", "pageable", "inf"),
        ("negative", "pinned_staged", "-1.0000"),
        ("zero", "pinned_staged", "0.0000"),
    ):
        with tempfile.TemporaryDirectory() as d:
            logdir = Path(d)
            for run in range(1, 6):
                rows = _fixture_rows(pageable_ms=5.0, pinned_ms=4.0)
                if run == 2:
                    rows = [
                        (phase, n, bad_value) if (phase == target_phase and n == "2048") else (phase, n, ms)
                        for phase, n, ms in rows
                    ]
                _write_run_log(logdir / f"layer_b_run{run}.log", rows)
            buf = io.StringIO()
            ok = run_aggregate(logdir, 5, out=buf)
            assert ok is False, f"{label} fixture should fail (undetermined): {buf.getvalue()}"
            assert "非有限または非正" in buf.getvalue(), buf.getvalue()
            assert "N=2048" in buf.getvalue(), buf.getvalue()
            assert not (logdir / "aggregate.md").exists(), f"{label}: aggregate.md must not be written"

    # --- fixture 9: 数値として解釈できない median_ms。
    with tempfile.TemporaryDirectory() as d:
        logdir = Path(d)
        for run in range(1, 6):
            rows = _fixture_rows(pageable_ms=5.0, pinned_ms=4.0)
            if run == 5:
                rows = [
                    (phase, n, "abc") if (phase == "pageable" and n == "1024") else (phase, n, ms)
                    for phase, n, ms in rows
                ]
            _write_run_log(logdir / f"layer_b_run{run}.log", rows)
        buf = io.StringIO()
        ok = run_aggregate(logdir, 5, out=buf)
        assert ok is False, "non-numeric fixture should fail (undetermined)"
        assert "数値として解釈できません" in buf.getvalue(), buf.getvalue()

    print(
        "self-test: all fixtures passed (normal / regression / missing-run / "
        "missing-cell / duplicate-row / inf / negative / zero / non-numeric)"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        arg_logdir = sys.argv[1] if len(sys.argv) > 1 else str(HERE)
        arg_n_runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        main(arg_logdir, arg_n_runs)
