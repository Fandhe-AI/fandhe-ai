#!/usr/bin/env python3
"""uptime-m4max.log（30 秒間隔ポーラの生ログ）から load average（1 分値）の
min／median／max・サンプル数を集計する（イシュー #1522）。

`orchestrate_m4max.sh` の `GEMM_GATE_LOAD_GATE_MODE=record_only` 経路は専有
ゲートを要件にしないため、計測中の共有負荷が実際にどう推移したかを数値で
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §24.11 へ記録する必要がある
（計画 §4 規則 4: record_only は計測条件の運用パラメータであって判定入力では
ないが、共有負荷下であったことは記録する）。追加依存なしの python3 標準
ライブラリのみで完結させる（`docs/perf/logs/` 配下の他集計スクリプトと同じ
方針）。

各行の形式（`orchestrate_m4max.sh` が書く）:
    poll <ISO8601> <uptime コマンドの生出力全文>

`uptime` の出力は環境によって "load average:" と "load averages:" の両方が
あるため、両方を許容する正規表現で 1 分値のみを抽出する。
"""
from __future__ import annotations

import re
import statistics
import sys

_LOAD1_RE = re.compile(r"load averages?:\s*([0-9.]+)[,\s]")


def extract_load1(line: str) -> float | None:
    """1 行から load average（1 分値）を抽出する。見つからなければ None。"""
    m = _LOAD1_RE.search(line)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def aggregate(lines: list[str]) -> dict:
    """行のリストから load1 の min／median／max／サンプル数／無効行数を集計する。"""
    values: list[float] = []
    invalid = 0
    for line in lines:
        if not line.strip():
            continue
        v = extract_load1(line)
        if v is None:
            invalid += 1
            continue
        values.append(v)
    if not values:
        return {"count": 0, "invalid": invalid, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "invalid": invalid,
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def render_markdown(agg: dict, source: str) -> str:
    if agg["count"] == 0:
        return (
            f"# load average 推移集計（{source}）\n\n"
            f"有効サンプルなし（無効行 {agg['invalid']} 件）。\n"
        )
    return (
        f"# load average 推移集計（{source}）\n\n"
        f"| サンプル数 | 無効行 | min | median | max |\n"
        f"|---|---|---|---|---|\n"
        f"| {agg['count']} | {agg['invalid']} | {agg['min']:.2f} | "
        f"{agg['median']:.2f} | {agg['max']:.2f} |\n"
    )


def _self_test() -> None:
    """固定入力に対する期待値を検証する（`--self-test`）。"""
    fixture = [
        "poll 2026-09-10T00:00:00Z 00:00:00 up 1 day,  1:00,  1 user,  load average: 1.00, 0.90, 0.80",
        "poll 2026-09-10T00:00:30Z 00:00:30 up 1 day,  1:00,  1 user,  load average: 3.00, 1.90, 1.80",
        "poll 2026-09-10T00:01:00Z 00:01:00 up 1 day,  1:01,  1 user,  load average: 2.00, 1.50, 1.40",
        "",  # blank line ignored
        "poll 2026-09-10T00:01:30Z malformed uptime output without load average token",
    ]
    agg = aggregate(fixture)
    assert agg["count"] == 3, agg
    assert agg["invalid"] == 1, agg
    assert agg["min"] == 1.00, agg
    assert agg["median"] == 2.00, agg
    assert agg["max"] == 3.00, agg

    # macOS 表記（"load averages:" 複数形）も許容することを確認する。
    macos_fixture = [
        "poll 2026-09-10T00:00:00Z 00:00  up 1 day, 1:00, 2 users, load averages: 0.50 0.40 0.30",
    ]
    agg2 = aggregate(macos_fixture)
    assert agg2["count"] == 1, agg2
    assert agg2["min"] == 0.50, agg2

    print("self-test OK")


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "--self-test":
        _self_test()
        return 0
    if len(argv) != 2:
        print(f"usage: {argv[0]} <uptime-m4max.log> | --self-test", file=sys.stderr)
        return 1
    path = argv[1]
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    agg = aggregate(lines)
    print(render_markdown(agg, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
