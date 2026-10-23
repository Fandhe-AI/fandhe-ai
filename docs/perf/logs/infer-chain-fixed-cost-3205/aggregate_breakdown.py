#!/usr/bin/env python3
"""イシュー #3205: infer_chain_issue_breakdown_diag_3205 の 5 run を集計する（標準ライブラリのみ）。

使い方: python3 -I aggregate_breakdown.py <runN.jsonl を含むディレクトリ> > aggregate.md
        python3 -I aggregate_breakdown.py --self-test

厳格方針（codex 指摘の集計スクリプト規約）: 5 run が揃っていなければ例外、欠損・非数・負値は例外、
run 番号付きの表を出す。0 埋め・寛容パースをしない。`*_other_derived` は導出値で符号付き（負は許す）。
"""
import glob
import json
import os
import re
import statistics
import sys
import tempfile

PHASES = ["l1_issue", "l2_issue", "alloc_in", "drop_in", "alloc_h", "drop_h", "alloc_out", "drop_out"]
DERIVED = ["l1_other_derived", "l2_other_derived"]
RUNS = 5


def parse_run(path):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = re.search(r"BACKEND_JSON (\{.*\})", line)
            if not m:
                continue
            d = json.loads(m.group(1))
            ph = d["phase"]
            if ph in rows:
                raise ValueError(f"{path}: phase {ph} が重複")
            v = d["median_s"]
            if not isinstance(v, (int, float)) or v != v or (ph in PHASES and v < 0):
                raise ValueError(f"{path}: phase {ph} の median_s が不正: {v!r}")
            rows[ph] = v * 1e6
    for ph in PHASES + DERIVED:
        if ph not in rows:
            raise ValueError(f"{path}: phase {ph} が欠損")
    return rows


def aggregate(d):
    paths = [os.path.join(d, f"breakdown_run{i}.jsonl") for i in range(1, RUNS + 1)]
    for p in paths:
        if not os.path.isfile(p):
            raise ValueError(f"run ファイルが欠損: {p}")
    extra = [p for p in glob.glob(os.path.join(d, "breakdown_run*.jsonl")) if p not in paths]
    if extra:
        raise ValueError(f"想定外の run ファイル: {extra}")
    runs = [parse_run(p) for p in paths]
    out = ["| 区間 (us) | " + " | ".join(f"run{i}" for i in range(1, RUNS + 1)) + " | 中央値 |", "|---|" + "---:|" * (RUNS + 1)]
    med = {}
    for ph in PHASES + DERIVED:
        vals = [r[ph] for r in runs]
        med[ph] = statistics.median(vals)
        out.append(f"| {ph} | " + " | ".join(f"{v:.2f}" for v in vals) + f" | {med[ph]:.2f} |")
    # チェーン内の確保・解放（再利用で消える分）= alloc_h + alloc_out + drop_h（出力 drop は download 後で計時窓外）
    removable = med["alloc_h"] + med["alloc_out"] + med["drop_h"]
    total = med["l1_issue"] + med["l2_issue"]
    out.append("")
    out.append(f"- 層の発行の合計（l1_issue＋l2_issue）: {total:.2f} us")
    out.append(f"- 再利用で削れる見込み（alloc_h＋alloc_out＋drop_h の単独計測値）: {removable:.2f} us（{removable / total * 100:.0f}%）")
    out.append(f"- 残り（導出。driver 境界・lookup・起動発行・入力 drop の合算）: l1 {med['l1_other_derived']:.2f} / l2 {med['l2_other_derived']:.2f} us")
    return "\n".join(out) + "\n"


def self_test():
    def write(d, i, drop=None, val=1e-6):
        with open(os.path.join(d, f"breakdown_run{i}.jsonl"), "w") as fh:
            for ph in PHASES + DERIVED:
                if ph == drop:
                    continue
                fh.write(f'test x ... BACKEND_JSON {{"backend":"cuda","phase":"{ph}","n":1,"median_s":{val}}}\n')

    with tempfile.TemporaryDirectory() as d:
        for i in range(1, 6):
            write(d, i)
        assert "中央値" in aggregate(d)
    rejects = [
        ("run 欠損", lambda d: [write(d, i) for i in range(1, 5)]),
        ("phase 欠損", lambda d: [write(d, i, drop="alloc_h" if i == 3 else None) for i in range(1, 6)]),
        ("負値", lambda d: [write(d, i, val=-1e-6 if i == 2 else 1e-6) for i in range(1, 6)]),
        ("非数", lambda d: [write(d, i, val="NaN" if i == 2 else 1e-6) for i in range(1, 6)]),
    ]
    for name, mk in rejects:
        with tempfile.TemporaryDirectory() as d:
            mk(d)
            try:
                aggregate(d)
            except (ValueError, json.JSONDecodeError):
                continue
            raise AssertionError(f"拒否されるべき入力が通った: {name}")
    print("self-test ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        self_test()
    elif len(sys.argv) == 2:
        sys.stdout.write(aggregate(sys.argv[1]))
    else:
        sys.exit("usage: aggregate_breakdown.py <dir> | --self-test")
