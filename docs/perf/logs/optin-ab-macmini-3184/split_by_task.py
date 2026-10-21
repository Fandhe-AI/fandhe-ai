#!/usr/bin/env python3
"""JSONL を task 別に機械的に分割する前処理（イシュー #3184。判定ロジックには関与しない）。

役割: `run_ab_readback_metal.sh`／`run_ab_readout_metal.sh` の生 JSONL は gemm 行と infer 行が
同一ファイルに混在し、`compare_gemm_ab.py` は「指定 task 以外の行」を不正行として「判定不能」に倒す。
2112 README §3 の位置引数コマンドを実行可能にするため、行を task フィールドだけで振り分ける
（行の値・順序は不変。欠損・不正行は捨てず `.other.jsonl` へ退避して件数を表示する）。stdlib のみ。
使い方: python3 -I split_by_task.py <入力.jsonl> <出力ディレクトリ>  -> <stem>.<task>.jsonl
"""
import json
import os
import sys


def split(path, outdir):
    stem = os.path.basename(path)[: -len(".jsonl")]
    buckets = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                task = json.loads(line).get("task")
            except (ValueError, AttributeError):
                task = None
            key = task if isinstance(task, str) and task.isidentifier() else "other"
            buckets.setdefault(key, []).append(line if line.endswith("\n") else line + "\n")
    os.makedirs(outdir, exist_ok=True)
    for key, lines in sorted(buckets.items()):
        out = os.path.join(outdir, f"{stem}.{key}.jsonl")
        with open(out, "w", encoding="utf-8") as g:
            g.writelines(lines)
        print(f"{out}: {len(lines)} rows")
    return buckets


def main(argv):
    if len(argv) != 3 or not argv[1].endswith(".jsonl"):
        print(__doc__, file=sys.stderr)
        return 2
    split(argv[1], argv[2])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
