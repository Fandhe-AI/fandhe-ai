#!/usr/bin/env python3
"""イシュー #3178: Mac mini CPU スレッド方針 A/B の集計（after/before 比・PyTorch 比・checksum 一致）。

役割: `run_ab_thread_policy_macmini_cpu.sh` が出力した before／after／torch の JSONL から、
RULE.txt が定める 12 セル（gemm N=256/512/1024/2048、train／infer size=64 の各 fresh/reuse）の
  (a) 5 round 中央値の after/before 比と round 別の比、(b) 5/5 round の一貫性、
  (c) PyTorch ÷ fandhe の比（同じ round 内の比の中央値。bench_py.py は fresh のみのため
      fandhe の fresh／reuse の両方を同じ torch 行と比べる）、(d) checksum の完全一致、
を Markdown で標準出力へ出す。判定（ADOPT 候補／REJECT／FAIL）の規則は RULE.txt が正で、
本ツールは RULE の「12 セル全部で中央値 ratio <= 1.00」を機械的に当てはめた結果を併記するだけ。

入力の厳格性: 欠損・重複・round 数が 5 でない・非数値・checksum 欠落は例外で停止し exit 非ゼロ
（寛容なパースはしない）。python3 標準ライブラリのみ。

使い方:
  aggregate.py <results-dir> <label>      # results-{before,after}-<label>-cpu-{gemm,train,infer}.jsonl と
                                          # results-torch-3178-*-<label>-cpu.jsonl を読む
  aggregate.py --self-test
"""
import glob
import json
import math
import os
import statistics
import sys
import tempfile

ROUNDS = 5
GEMM_SIZES = (256, 512, 1024, 2048)
MODES = ("fresh", "reuse")
CELLS = [("gemm", s, m) for s in GEMM_SIZES for m in MODES] + [
    (t, 64, m) for t in ("train", "infer") for m in MODES
]


class AggError(Exception):
    pass


def read_rows(path):
    if not os.path.isfile(path):
        raise AggError("ファイルがありません: %s" % path)
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as e:
                raise AggError("%s:%d JSON 不正: %s" % (path, i, e))
    return rows


def num(v, what):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        raise AggError("非数値の %s: %r" % (what, v))
    return float(v)


def by_cell(rows, label, torch=False):
    """セル (task,size,mode) -> 出現順の行リスト。round 番号は出現順（1 始まり）。"""
    out = {}
    for r in rows:
        task = r.get("task")
        size = r.get("size")
        mode = "fresh" if torch else r.get("mode", "fresh")
        out.setdefault((task, size, mode), []).append(r)
    return out


def load_arm(path, label):
    return by_cell(read_rows(path), label)


def cell_series(cells, key, field, what):
    if key not in cells:
        raise AggError("%s: セル %s がありません" % (what, key))
    rows = cells[key]
    if len(rows) != ROUNDS:
        raise AggError("%s: セル %s の記録は %d 件（%d 件を要求）" % (what, key, len(rows), ROUNDS))
    return [num(r.get(field), "%s %s.%s" % (what, key, field)) for r in rows]


def aggregate(before_by_task, after_by_task, torch_cells):
    out = []
    for key in CELLS:
        task, size, mode = key
        b = cell_series(before_by_task[task], key, "median_s", "before")
        a = cell_series(after_by_task[task], key, "median_s", "after")
        bc = cell_series(before_by_task[task], key, "checksum", "before")
        ac = cell_series(after_by_task[task], key, "checksum", "after")
        if any(x <= 0 for x in b + a):
            raise AggError("median_s が正でない: %s" % (key,))
        cks = set(bc) | set(ac)
        ck_ok = len(cks) == 1
        ratio_rounds = [x / y for x, y in zip(a, b)]
        ratio_med = statistics.median(a) / statistics.median(b)
        tkey = (task, size, "fresh")
        t = cell_series(torch_cells, tkey, "median_s", "torch")
        pt_before = statistics.median([ti / bi for ti, bi in zip(t, b)])
        pt_after = statistics.median([ti / ai for ti, ai in zip(t, a)])
        out.append(
            dict(
                key=key,
                ratio=ratio_med,
                ratio_rounds=ratio_rounds,
                consistent=sum(1 for r in ratio_rounds if r <= 1.00),
                ck_ok=ck_ok,
                torch_before=pt_before,
                torch_after=pt_after,
            )
        )
    return out


def verdict(res):
    if any(not r["ck_ok"] for r in res):
        return "FAIL（checksum 不一致）"
    if all(r["ratio"] <= 1.00 for r in res):
        return "ADOPT 候補（12 セル全部で中央値比 <= 1.00。ゲート・前提の成立は別途 RULE で確認）"
    return "REJECT（中央値比 > 1.00 のセルあり。ゲート・前提の成立は別途 RULE で確認）"


def render(res, title):
    lines = ["## %s" % title, ""]
    lines.append("| セル | after/before 中央値比 | round 別比 | <=1.00 の round | checksum | PyTorch÷fandhe(before) | PyTorch÷fandhe(after) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in res:
        t, s, m = r["key"]
        lines.append(
            "| %s %d %s | %.4f | %s | %d/5 | %s | %.3f | %.3f |"
            % (
                t,
                s,
                m,
                r["ratio"],
                " ".join("%.3f" % x for x in r["ratio_rounds"]),
                r["consistent"],
                "一致" if r["ck_ok"] else "**不一致**",
                r["torch_before"],
                r["torch_after"],
            )
        )
    lines += ["", "機械適用結果: %s" % verdict(res), ""]
    return "\n".join(lines)


def find_torch(d, label):
    hits = sorted(glob.glob(os.path.join(d, "results-torch-3178-*-%s-cpu.jsonl" % label)))
    if len(hits) != 1:
        raise AggError("torch の JSONL が 1 件ではありません: %r" % hits)
    return hits[0]


def run(d, label, title=None):
    before = {}
    after = {}
    for task in ("gemm", "train", "infer"):
        before[task] = load_arm(os.path.join(d, "results-before-%s-cpu-%s.jsonl" % (label, task)), label)
        after[task] = load_arm(os.path.join(d, "results-after-%s-cpu-%s.jsonl" % (label, task)), label)
    torch_cells = by_cell(read_rows(find_torch(d, label)), label, torch=True)
    res = aggregate(before, after, torch_cells)
    return render(res, title or label), res


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _fixture(d, label, after_scale=0.9, drop=False):
    for arm, scale in (("before", 1.0), ("after", after_scale)):
        for task in ("gemm", "train", "infer"):
            rows = []
            for rnd in range(ROUNDS):
                for t, s, m in CELLS:
                    if t != task:
                        continue
                    rows.append(dict(task=t, size=s, mode=m, device="cpu", median_s=0.01 * scale * (1 + rnd * 0.01), checksum=1.5))
            if drop and arm == "after" and task == "gemm":
                rows = rows[:-1]
            _write(os.path.join(d, "results-%s-%s-cpu-%s.jsonl" % (arm, label, task)), rows)
    trows = []
    for rnd in range(ROUNDS):
        for t, s, m in CELLS:
            if m == "fresh":
                trows.append(dict(framework="pytorch", task=t, size=s, device="cpu", median_s=0.02, checksum=1.0))
    _write(os.path.join(d, "results-torch-3178-big-core-%s-cpu.jsonl" % label), trows)


def self_test():
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, "t")
        text, res = run(d, "t")
        assert len(res) == 12 and "ADOPT" in text and all(r["consistent"] == 5 for r in res)
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, "t", after_scale=1.2)
        text, _ = run(d, "t")
        assert "REJECT" in text
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, "t", drop=True)
        try:
            run(d, "t")
        except AggError:
            pass
        else:
            raise AssertionError("欠損で停止しなかった")
    with tempfile.TemporaryDirectory() as d:
        _fixture(d, "t")
        p = os.path.join(d, "results-after-t-cpu-train.jsonl")
        rows = read_rows(p)
        rows[0]["checksum"] = 2.5
        _write(p, rows)
        text, _ = run(d, "t")
        assert "FAIL" in text
    print("self-test ok")


def main(argv):
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        text, _ = run(argv[1], argv[2])
    except AggError as e:
        print("aggregate error: %s" % e, file=sys.stderr)
        return 1
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
