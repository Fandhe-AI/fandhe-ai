#!/usr/bin/env python3
"""イシュー #2106: 独立 5 run の JSONL を RULE.txt の規則で集計する（標準ライブラリのみ）。

使い方: aggregate.py <dir>   （dir は run1.jsonl..run5.jsonl と load_gate_status.txt を持つ）
        aggregate.py --self-test
欠落セル・不正 JSON・n 不一致・checksum 不一致は fail-closed（exit 1）。判定閾値は
RULE.txt（実測前に固定）が正で、本スクリプトは変更しない。
"""
import json
import math
import os
import statistics
import sys

RUNS = 5
EXPECTED = {
    "insitu_direct": ["device_update"],
    "insitu_pretouch": ["device_update"],
    "standalone": [
        "alloc", "stage", "sgd_kernel", "sgd_compute_split", "apply_params_split",
        "sgd_kernel_zip", "sgd_kernel_xthread", "sgd_kernel_fixed",
    ],
}
# checksum_bits を持つべきセル（RULE.txt checksum 規則）
CHECKSUM_INSITU = [("insitu_direct", "device_update"), ("insitu_pretouch", "device_update")]
CHECKSUM_STANDALONE = [
    ("standalone", p)
    for p in ("sgd_kernel", "sgd_kernel_zip", "apply_params_split", "sgd_kernel_xthread")
]
EXPECTED_N = 80  # 計測 20 反復 x 4 ラウンド（テスト側 ROUNDS x ITERS_PER_ROUND）
GATE_FILE = "load_gate_status.txt"
ATTRIB_THRESHOLD = 0.5  # RULE.txt 帰属規則（事後に変えない）


class AggError(Exception):
    pass


def load_run(path):
    cells = {}
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key = (rec["arm"], rec["phase"])
                    rec["median_s"] = float(rec["median_s"])
                    n = rec["n"]
                except (ValueError, KeyError, TypeError) as e:
                    raise AggError(f"{path}:{i}: 不正な行 ({e})")
                if key[1] not in EXPECTED.get(key[0], []):
                    raise AggError(f"{path}:{i}: 想定外のセル {key}")
                if not math.isfinite(rec["median_s"]) or rec["median_s"] <= 0.0:
                    raise AggError(f"{path}:{i}: median_s が有限の正値でない {key} ({rec['median_s']})")
                if isinstance(n, bool) or n != EXPECTED_N:
                    raise AggError(f"{path}:{i}: 計測回数 n={n!r} が {EXPECTED_N} でない {key}")
                if key in cells:
                    raise AggError(f"{path}:{i}: セル重複 {key}")
                cells[key] = rec
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    for arm, phases in EXPECTED.items():
        for ph in phases:
            if (arm, ph) not in cells:
                raise AggError(f"{path}: セル欠落 {arm}/{ph}")
    return cells


def load_gate(d):
    """load_gate_status.txt を読み、(未通過 run 名一覧, record_only run 名一覧) を返す（欠落・不正は fail-closed）。

    record_only は共有環境（M4 Max）で負荷ゲートを通過できず記録のみとした run であり、
    ゲート通過済み（pass）の通常判定と区別して出力に明示する。"""
    path = os.path.join(d, GATE_FILE)
    try:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    seen = {}
    for ln in lines:
        parts = ln.split()
        if len(parts) != 2 or not parts[1].startswith("gate="):
            raise AggError(f"{path}: 不正な行 {ln!r}")
        status = parts[1][5:]
        if status not in ("pass", "unpassed", "record_only") or parts[0] in seen:
            raise AggError(f"{path}: 不正または重複した行 {ln!r}")
        seen[parts[0]] = status
    if set(seen) != {f"run{n}" for n in range(1, RUNS + 1)}:
        raise AggError(f"{path}: run1..run{RUNS} の全行が必要 ({sorted(seen)})")
    return (sorted(r for r, st in seen.items() if st == "unpassed"),
            sorted(r for r, st in seen.items() if st == "record_only"))


def med(runs, arm, ph):
    vals = [r[(arm, ph)]["median_s"] for r in runs]
    return statistics.median(vals), min(vals), max(vals)


def check_checksums(runs):
    """RULE.txt checksum 規則。run 間・腕間の不一致と欠落を fail-closed で検出する。"""
    for group in (CHECKSUM_INSITU, CHECKSUM_STANDALONE):
        seen = set()
        for key in group:
            for r in runs:
                v = r[key].get("checksum_bits")
                if not v:
                    raise AggError(f"checksum_bits 欠落: {key}")
                seen.add(v)
        if len(seen) != 1:
            raise AggError(f"checksum が run 間または腕間で不一致: {[k[1] for k in group]} {sorted(seen)}")


def us(x):
    return f"{x * 1e6:.1f}"


def aggregate(runs, unpassed=(), record_only=()):
    check_checksums(runs)
    out = []
    if record_only:
        out.append(f"【record_only（共有環境・負荷ゲート非適用）】run: {', '.join(record_only)}"
                   "（RULE.txt「m4max: record_only」。負荷ゲート通過済み〈pass〉の通常判定とは区別し、以下の集計は記録扱い）")
        out.append("")
    if unpassed:
        out.append(f"【参考扱い】GB10 負荷ゲート未通過の run: {', '.join(unpassed)}（RULE.txt ゲート節。通常判定として扱わない）")
        out.append("")
    out.append("| arm | phase | median(us) | min-max(us) | q1/q3 median(us) |")
    out.append("|---|---|---|---|---|")
    for arm, phases in EXPECTED.items():
        for ph in phases:
            m, lo, hi = med(runs, arm, ph)
            q1 = statistics.median([r[(arm, ph)]["q1_s"] for r in runs])
            q3 = statistics.median([r[(arm, ph)]["q3_s"] for r in runs])
            out.append(f"| {arm} | {ph} | {us(m)} | {us(lo)}-{us(hi)} | {us(q1)}/{us(q3)} |")
    out.append("")
    out.append("checksum: run 間・腕間一致 OK")

    t = med(runs, "insitu_direct", "device_update")[0]
    pre = med(runs, "insitu_pretouch", "device_update")[0]
    s = lambda ph: med(runs, "standalone", ph)[0]
    kernel = s("sgd_kernel")
    terms = {
        "H2 cache 状態(direct-pretouch)": t - pre,
        "H1 ループ形((kernel-fixed)-zip)": (kernel - s("sgd_kernel_fixed")) - s("sgd_kernel_zip"),
        "H3 ホスト確保(alloc+stage)": s("alloc") + s("stage"),
        "kernel 全体(sgd_kernel)": kernel,
        "H4 prologue+残差(pretouch-(kernel+alloc+stage))": pre - (kernel + s("alloc") + s("stage")),
    }
    out.append(f"T = insitu_direct/device_update = {us(t)} us（bench §17.6.2 の 277.5 us〈GB10〉は参考値）")
    out.append("")
    out.append("| 項 | 値(us) | T 比 | 判定 |")
    out.append("|---|---|---|---|")
    supported = []
    for name, v in terms.items():
        share = v / t
        ok = share >= ATTRIB_THRESHOLD
        if ok:
            supported.append(name)
        out.append(f"| {name} | {us(v)} | {share:.2f} | {'支持' if ok else '-'} |")
    out.append("")
    out.append("帰属: " + (", ".join(supported) if supported else "未確定（どの項も 50% 未満）"))
    split = s("sgd_compute_split") + s("apply_params_split")
    out.append(f"補助: apply_params_split / (compute+apply)_split = {s('apply_params_split') / split:.2f}")
    out.append(f"補助: sgd_kernel_xthread = {us(s('sgd_kernel_xthread'))} us（H2 の補助・別スレッド生成 Tensor 経由でありクロスコア書き込みは再現しない。判定に使わない）")
    return "\n".join(out)


def synth_line(arm, ph, med_s, chk=None):
    return {"arm": arm, "phase": ph, "median_s": med_s, "n": EXPECTED_N, "checksum_bits": chk,
            "q1_s": med_s * 0.9, "q3_s": med_s * 1.1, "min_s": med_s * 0.8, "max_s": med_s * 1.3}


def synth_cells():
    lines = []
    for arm, phases in EXPECTED.items():
        for ph in phases:
            t = {"device_update": 2.5e-4, "sgd_kernel": 1.5e-4, "sgd_kernel_zip": 0.5e-4, "sgd_kernel_fixed": 0.2e-4}.get(ph, 1e-5)
            if arm == "insitu_pretouch":
                t = 2.0e-4
            if ph == "sgd_kernel_zip":
                t = 0.5e-4
            chk = None
            if (arm, ph) in CHECKSUM_INSITU:
                chk = "00ff"
            if (arm, ph) in CHECKSUM_STANDALONE:
                chk = "abcd"
            lines.append(synth_line(arm, ph, t, chk))
    return lines


def write_gate(d, statuses):
    with open(os.path.join(d, GATE_FILE), "w", encoding="utf-8") as f:
        for n, st in enumerate(statuses, 1):
            f.write(f"run{n} gate={st}\n")


def write_run(path, mut=None):
    with open(path, "w", encoding="utf-8") as f:
        for rec in synth_cells():
            if mut:
                mut(rec)
            f.write(json.dumps(rec) + "\n")


def expect_fail(d, label):
    try:
        run_dir(d)
    except AggError:
        return
    raise SystemExit(f"self-test 失敗: {label} を検出できない")


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        for n in range(1, RUNS + 1):
            write_run(os.path.join(d, f"run{n}.jsonl"))
        write_gate(d, ["pass"] * RUNS)
        text = run_dir(d)
        assert "帰属" in text and "参考扱い" not in text and "H1 ループ形" in text, text
        assert "record_only" not in text, text
        write_gate(d, ["pass", "unpassed", "pass", "pass", "pass"])
        text = run_dir(d)
        assert "【参考扱い】" in text and "run2" in text, text
        write_gate(d, ["record_only"] * RUNS)
        text = run_dir(d)
        assert "【record_only" in text and "run1" in text and "run5" in text and "参考扱い" not in text, text
        os.remove(os.path.join(d, GATE_FILE))
        expect_fail(d, "ゲート状態ファイル欠落")
        write_gate(d, ["pass"] * RUNS)

        p = os.path.join(d, "run1.jsonl")
        for label, mut in (
            ("n=1", lambda r: r.__setitem__("n", 1)),
            ("median NaN", lambda r: r.__setitem__("median_s", float("nan"))),
            ("median 負", lambda r: r.__setitem__("median_s", -1e-4)),
            ("median 0", lambda r: r.__setitem__("median_s", 0.0)),
            ("想定外セル", lambda r: r.__setitem__("phase", "bogus")),
        ):
            done = []

            def once(rec, mut=mut, done=done):
                if not done:
                    done.append(1)
                    mut(rec)
            write_run(p, once)
            expect_fail(d, label)
        # 欠落セル
        write_run(p)
        lines = open(p, encoding="utf-8").read().splitlines()
        open(p, "w", encoding="utf-8").write("\n".join(lines[:-1]) + "\n")
        expect_fail(d, "欠落セル")
        open(p, "w", encoding="utf-8").write("{not json\n")
        expect_fail(d, "不正 JSON")
        # checksum: 腕間不一致（standalone の 1 腕）
        write_run(p, lambda r: r.__setitem__("checksum_bits", "dead")
                  if (r["arm"], r["phase"]) == ("standalone", "sgd_kernel_zip") else None)
        expect_fail(d, "standalone 腕間の checksum 不一致")
        # checksum: in-situ 腕間不一致
        write_run(p, lambda r: r.__setitem__("checksum_bits", "dead")
                  if (r["arm"], r["phase"]) == ("insitu_pretouch", "device_update") else None)
        expect_fail(d, "in-situ 腕間の checksum 不一致")
        # checksum 欠落
        write_run(p, lambda r: r.__setitem__("checksum_bits", None)
                  if (r["arm"], r["phase"]) == ("standalone", "sgd_kernel") else None)
        expect_fail(d, "checksum 欠落")
    print("self-test OK")


def run_dir(d):
    runs = [load_run(os.path.join(d, f"run{n}.jsonl")) for n in range(1, RUNS + 1)]
    unpassed, record_only = load_gate(d)
    return aggregate(runs, unpassed, record_only)


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        print(run_dir(argv[1]))
    except AggError as e:
        print(f"集計失敗（fail-closed）: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
