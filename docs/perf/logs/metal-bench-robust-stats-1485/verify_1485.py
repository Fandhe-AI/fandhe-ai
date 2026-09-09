#!/usr/bin/env python3
"""イシュー #1485: phase1_only_run1.log の判定不変・キー順を機械検査する。
標準ライブラリのみ使用（他のログ集計スクリプトと同方針）。
"""
import re
import sys

EXPECTED_KEY_ORDER = [
    "size", "rounds", "spread", "gate", "within_gate",
    "median_secs", "min_secs", "min_round_idx", "max_secs", "max_round_idx",
    "round_medians_secs",
]
EXPECTED_AUX_ORDER = ["trimmed_spread_k1", "iqr_spread", "mad_spread"]


def parse_kv_line(line: str) -> dict:
    assert line.startswith("phase1_round_stats ")
    rest = line[len("phase1_round_stats "):].strip()
    # round_medians_secs= value contains commas but no spaces, so simple split on whitespace is safe
    tokens = rest.split(" ")
    d = {}
    order = []
    for tok in tokens:
        key, _, val = tok.partition("=")
        d[key] = val
        order.append(key)
    return d, order


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "phase1_only_run1.log"
    with open(path, encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f if l.startswith("phase1_round_stats ")]

    assert len(lines) == 5, f"expected 5 phase1_round_stats lines, got {len(lines)}"

    sizes = []
    all_ok = True
    for line in lines:
        d, order = parse_kv_line(line)
        sizes.append(d["size"])

        # (a) key order check: first 11 keys match EXPECTED_KEY_ORDER, then 3 aux keys
        first11 = order[:11]
        aux3 = order[11:14]
        if first11 != EXPECTED_KEY_ORDER:
            print(f"FAIL key order (base) at size={d['size']}: {first11}")
            all_ok = False
        if aux3 != EXPECTED_AUX_ORDER:
            print(f"FAIL key order (aux) at size={d['size']}: {aux3}")
            all_ok = False

        # (b) within_gate consistency check
        spread = float(d["spread"])
        gate = float(d["gate"])
        expected_within = spread <= gate
        actual_within = d["within_gate"] == "true"
        if expected_within != actual_within:
            print(
                f"FAIL within_gate mismatch at size={d['size']}: "
                f"spread={spread} gate={gate} expected={expected_within} actual={d['within_gate']}"
            )
            all_ok = False

        print(
            f"size={d['size']} spread={spread:.4e} gate={gate:.4e} "
            f"within_gate={d['within_gate']} (spread<=gate: {expected_within}) "
            f"trimmed_spread_k1={d['trimmed_spread_k1']} iqr_spread={d['iqr_spread']} "
            f"mad_spread={d['mad_spread']}"
        )

    assert sizes == ["256", "512", "1024", "2048", "4096"], f"unexpected size order: {sizes}"

    if all_ok:
        print("ALL CHECKS PASS: key order matches / within_gate == (spread <= gate) for all 5 rows")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
