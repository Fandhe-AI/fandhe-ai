#!/usr/bin/env python3
"""イシュー #1478（`HOST_STAGING_KIND` の既定を `Pinned` へ切替）の
ゲート B 集計スクリプト。`ab-run1.log`〜`ab-run5.log`（`cargo test
host_view_staging_readout_ab -- --ignored --nocapture --test-threads=1`
の 5 回独立プロセス起動出力）から `before`／`after_pageable`／
`after_pinned` の中央値（median_ms 列）を抽出し、N ごとに 5 run の
中央値（representative value）を算出する。Python3 標準ライブラリのみ
（`docs/perf/logs/cuda-host-view-staging-1336/aggregate.py` と同型）。

`default_kind,<kind>` 行（`test host_view_staging_readout_ab ...
default_kind,Pinned` として出力される。`test ... ` prefix は cargo test
のランナーが書く未フラッシュのテスト名で、後続の println! 出力と
同一行に連結される）が全 5 run で `Pinned` であることも検証する
（本番既定コンストラクタが実際に `Pinned` へ解決されている自己証明。
イシュー #1478 AC-2）。
"""

import math
import statistics
import sys

SIZES = [1024, 2048, 4096]
PHASES = ["before", "after_pageable", "after_pinned"]
RUN_FILES = [f"ab-run{i}.log" for i in range(1, 6)]


def parse_runs():
    """各ファイル（1 run）につき `default_kind` 行がちょうど 1 件、
    各 phase × N 組の計測値がちょうど 1 件（有限かつ正の値）存在する
    ことを検証してから集計する。codex-review 指摘（PR #1494 スレッド
    `PRRT_kwDOTuUCJc6gi09t`）: 検証なしで中央値を算出すると、例えば
    ab-run5.log が `default_kind` 行の直後で途切れていても残り 4 回の
    値だけでゲート B を通過できてしまう（欠落を無言で許容する）ため、
    ファイル単位で欠落・重複を fail-closed に検出する（AGENTS.md の
    「ベンチは 5 回計測の中央値」・ゲート B 契約を保証する）。
    """
    data = {p: {n: [] for n in SIZES} for p in PHASES}
    default_kinds = []
    for fname in RUN_FILES:
        with open(fname) as f:
            text = f.read()
        file_default_kinds = []
        file_counts = {p: {n: 0 for n in SIZES} for p in PHASES}
        for line in text.splitlines():
            if "default_kind" in line:
                # "test host_view_staging_readout_ab ... default_kind,Pinned"
                file_default_kinds.append(line.rsplit(",", 1)[-1].strip())
            parts = line.split(",")
            if len(parts) >= 9 and parts[0] in PHASES:
                phase = parts[0]
                n = int(parts[1])
                median_ms = float(parts[6])
                if not math.isfinite(median_ms) or median_ms <= 0:
                    raise ValueError(
                        f"{fname}: {phase} N={n} の median_ms が有限かつ正の値ではない: {median_ms}"
                    )
                if n in data[phase]:
                    file_counts[phase][n] += 1
                data[phase].setdefault(n, [])
                data[phase][n].append(median_ms)

        if len(file_default_kinds) != 1:
            raise ValueError(
                f"{fname}: default_kind 行はちょうど 1 件である必要があるが "
                f"{len(file_default_kinds)} 件検出（欠落または重複。行途中で "
                f"ログが途切れている可能性がある）: {file_default_kinds}"
            )
        default_kinds.extend(file_default_kinds)

        for phase in PHASES:
            for n in SIZES:
                count = file_counts[phase][n]
                if count != 1:
                    raise ValueError(
                        f"{fname}: phase={phase} N={n} の計測値はちょうど 1 件で"
                        f"ある必要があるが {count} 件検出（欠落または重複。ログが"
                        f"途中で途切れている可能性がある）"
                    )
    return data, default_kinds


def main():
    data, default_kinds = parse_runs()

    print(f"default_kind occurrences (expect 5x Pinned): {default_kinds}")
    assert len(default_kinds) == 5, f"expected 5 default_kind lines, got {len(default_kinds)}"
    assert all(k == "Pinned" for k in default_kinds), (
        f"default_kind must be Pinned in all 5 runs (self-proof of production default): {default_kinds}"
    )

    print()
    header = f"{'N':>6} {'before_med_ms':>14} {'pageable_med_ms':>16} {'pinned_med_ms':>14} {'pinned/pageable':>16} {'gate_b(<=1.05)':>15}"
    print(header)
    gate_b_all_pass = True
    for n in SIZES:
        b = statistics.median(data["before"][n])
        pg = statistics.median(data["after_pageable"][n])
        pn = statistics.median(data["after_pinned"][n])
        ratio = pn / pg
        gate_ok = ratio <= 1.05
        gate_b_all_pass = gate_b_all_pass and gate_ok
        print(
            f"{n:>6} {b:>14.4f} {pg:>16.4f} {pn:>14.4f} {ratio:>16.4f} {('PASS' if gate_ok else 'FAIL'):>15}"
        )

    print()
    print(f"gate_b_all_pass = {gate_b_all_pass}")
    if not gate_b_all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
