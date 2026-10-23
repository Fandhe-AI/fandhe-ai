#!/usr/bin/env python3
"""イシュー #3227: CUDA readback 既定反転の確認計測（run_confirm_readback_default_cuda.sh）の集計表を出す。

役割: before（env=pretouched・旧既定）／after（env 未設定・新既定）の JSONL（gemm・train・infer が
混在）を task ごとに分け、セル（task, mode, size）別に 5 run の median_s の中央値どうしの比
ratio = median(after)/median(before)、run 内比、checksum 一致を表にする。判定語（ADOPT／REJECT）は
出さない。ratio > 1.00 のセルは「後退セル」として末尾に全て併記する（RULE.txt）。
`compare_gemm_ab.py` は gemm の size 集合が固定（256／512 を含む集合を別名で持つ）で、混在 JSONL を
不正行として拒否するため、確認計測専用に本ツールを置く（`compare_gemm_ab.py` は変更しない）。

呼び出し元: run_confirm_readback_default_cuda.sh。単体でも `python3 -I summarize_readback_confirm.py
<before.jsonl> <after.jsonl>` で実行できる。終了コードは常に 0（記録のみ。欠損は表に明示する）。
"""
import json
import statistics
import sys

RUNS = 5

# RULE.txt の固定 14 セル（task, mode, size）。train/infer は bench-fandhe の出力どおり size=64。表示対象を入力キーの和集合だけにすると、
# 両腕から丸ごと欠落したセルが表に出ないため、この集合との和集合を表示対象にする。
EXPECTED_CELLS = (
    [("gemm", mode, size) for mode in ("reuse", "fresh") for size in (256, 512, 1024, 2048, 4096)]
    + [("train", mode, 64) for mode in ("fresh", "reuse")]
    + [("infer", mode, 64) for mode in ("fresh", "reuse")]
)


def load(path):
    cells = {}
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            key = (o.get("task", "gemm"), o.get("mode", "fresh"), o.get("size"))
            cells.setdefault(key, []).append((o.get("median_s"), o.get("checksum")))
    return cells, bad


def main(before_path, after_path):
    before, bad_b = load(before_path)
    after, bad_a = load(after_path)
    order = {"gemm": 0, "train": 1, "infer": 2}
    keys = sorted(set(before) | set(after) | set(EXPECTED_CELLS), key=lambda k: (order.get(k[0], 9), k[1], k[2] or 0))
    print("| task | mode | size | before median | after median | after/before | run 内比 | run 数 b/a | checksum |")
    print("|---|---|---|---|---|---|---|---|---|")
    regress = []
    for k in keys:
        b = before.get(k, [])
        a = after.get(k, [])
        bm = [x for x, _ in b]
        am = [x for x, _ in a]
        ok = len(b) == RUNS and len(a) == RUNS and all(isinstance(x, (int, float)) for x in bm + am)
        cs = {c for _, c in b + a}
        cs_txt = "データなし" if not cs else "完全一致" if len(cs) == 1 else "不一致 %r" % sorted(cs, key=str)
        if not ok:
            print("| %s | %s | %s | - | - | 欠損 | - | %d/%d | %s |" % (k[0], k[1], k[2], len(b), len(a), cs_txt))
            continue
        mb, ma = statistics.median(bm), statistics.median(am)
        ratio = ma / mb
        per = ", ".join("%.4f" % (x / y) for x, y in zip(am, bm))
        print("| %s | %s | %s | %.1f us | %.1f us | %.4f | %s | %d/%d | %s |"
              % (k[0], k[1], k[2], mb * 1e6, ma * 1e6, ratio, per, len(b), len(a), cs_txt))
        if ratio > 1.0:
            regress.append((k, ratio))
    print()
    print("不正 JSON 行: before=%d after=%d" % (bad_b, bad_a))
    print("後退セル（ratio > 1.00。全て併記）:")
    if not regress:
        print("  なし")
    for k, r in regress:
        print("  - %s %s size=%s ratio=%.4f" % (k[0], k[1], k[2], r))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: summarize_readback_confirm.py <before.jsonl> <after.jsonl>")
    main(sys.argv[1], sys.argv[2])
