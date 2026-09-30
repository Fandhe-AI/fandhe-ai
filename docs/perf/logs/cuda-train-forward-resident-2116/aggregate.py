#!/usr/bin/env python3
"""イシュー #2116: 独立 5 run の JSONL を RULE.txt の規則で集計する（標準ライブラリのみ）。

使い方: aggregate.py <dir>   （dir は run{1..5}.facade.jsonl・run{1..5}.backend.jsonl と
                              load_gate_status.txt を持つ）
        aggregate.py --self-test
欠落セル・不正 JSON・n 不一致・非有限値・checksum 不一致・ゲート状態欠落は fail-closed
（exit 1）。判定閾値は RULE.txt（実測前に固定）が正で、本スクリプトは変更しない。
"""
import json
import math
import os
import statistics
import sys

RUNS = 5
BATCHES = [16, 64, 256, 1024]
BENCH_BATCH = 64
EXPECTED_N = 80  # 計測 20 反復 x 4 ラウンド（テスト側 ROUNDS x ITERS_PER_ROUND）
GATE_FILE = "load_gate_status.txt"
SUPPORT = 0.40  # RULE.txt: 支配項（事後に変えない）
CONTRIB = 0.20
FIDELITY_LO, FIDELITY_HI = 0.8, 1.2
P_THRESHOLD = 0.50
P3_RATIO = 1.50
REF_FORWARD_US = 155.9  # train-step-phase-breakdown.md §17.6.2（registry 0.9.0・参考）

NOSYNC_PHASES = ["device_handle", "mem_new", "h2d_upload", "alloc_c", "gemm_lookup",
                 "launch_issue", "d2h_download", "total"]
SYNC_PHASES = NOSYNC_PHASES + ["kernel_wait"]
READOUT_ARMS = ["readout_r0_mirror", "readout_r1_split", "readout_r2_pretouched",
                "readout_r3_cpu_control"]
R1_PHASES = ["param_readout", "param_contiguous", "param_to_vec", "grad_contiguous",
             "grad_to_vec", "noncontig_grad_count"]


def expected_facade():
    """(arm, phase, batch) の期待集合。"""
    s = set()
    for b in BATCHES:
        s.add(("public", "forward_resident", b))
        for ph in ("register", "l1_linear_relu", "l2_linear", "mse_loss"):
            s.add(("decomposed", ph, b))
        s.add(("paired", "residual", b))
    for arm in READOUT_ARMS:
        for ph in (R1_PHASES if arm == "readout_r1_split" else ["param_readout"]):
            s.add((arm, ph, BENCH_BATCH))
    return s


def expected_backend():
    """(arm, kind, phase, batch) の期待集合。"""
    s = set()
    for b in BATCHES:
        for kind in ("l1", "l2"):
            s.add(("prod", kind, "total", b))
            for ph in NOSYNC_PHASES:
                s.add(("nosync", kind, ph, b))
            for ph in SYNC_PHASES:
                s.add(("syncsplit", kind, ph, b))
        s.add(("prod", "relu", "total", b))
        s.add(("prod", "mse", "total", b))
        s.add(("whatif", "l1", "total", b))
    return s


# 中央値が 0 以下でもよいセル（差分・カウント）
NON_POSITIVE_OK_FACADE = {("paired", "residual"), ("readout_r1_split", "noncontig_grad_count")}


class AggError(Exception):
    pass


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def load_facade(path):
    cells, checksums = {}, {}
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    kind = rec.get("record", "cell")
                    arm, batch = rec["arm"], rec["batch"]
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    raise AggError(f"{path}:{i}: 不正な行 ({e})")
                if kind == "checksum":
                    bits = rec.get("bits")
                    if not isinstance(bits, str) or not bits or (arm, batch) in checksums:
                        raise AggError(f"{path}:{i}: 不正または重複した checksum 行 ({arm},{batch})")
                    checksums[(arm, batch)] = bits
                    continue
                phase = rec.get("phase")
                key = (arm, phase, batch)
                if key not in EXPECTED_FACADE:
                    raise AggError(f"{path}:{i}: 想定外のセル {key}")
                if key in cells:
                    raise AggError(f"{path}:{i}: セル重複 {key}")
                m, n = rec.get("median_s"), rec.get("n")
                if not _finite(m):
                    raise AggError(f"{path}:{i}: median_s が有限でない {key} ({m!r})")
                if (arm, phase) not in NON_POSITIVE_OK_FACADE and m <= 0.0:
                    raise AggError(f"{path}:{i}: median_s が正でない {key} ({m})")
                if isinstance(n, bool) or n != EXPECTED_N:
                    raise AggError(f"{path}:{i}: 計測回数 n={n!r} が {EXPECTED_N} でない {key}")
                for k in ("q1_s", "q3_s"):
                    if not _finite(rec.get(k)):
                        raise AggError(f"{path}:{i}: {k} が有限でない {key}")
                cells[key] = rec
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    for key in EXPECTED_FACADE:
        if key not in cells:
            raise AggError(f"{path}: セル欠落 {key}")
    need = {(a, b) for b in BATCHES for a in ("public", "decomposed")}
    need |= {(a, BENCH_BATCH) for a in READOUT_ARMS}
    if set(checksums) != need:
        raise AggError(f"{path}: checksum 行の集合が不正 (欠落 {sorted(need - set(checksums))})")
    return cells, checksums


def load_backend(path):
    cells, whatif = {}, {}
    try:
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    arm, kind, phase, batch = rec["arm"], rec["kind"], rec["phase"], rec["batch"]
                except (ValueError, KeyError, TypeError, AttributeError) as e:
                    raise AggError(f"{path}:{i}: 不正な行 ({e})")
                if phase == "bits_equal":
                    v = rec.get("value")
                    if not isinstance(v, bool) or (kind, batch) in whatif:
                        raise AggError(f"{path}:{i}: 不正または重複した bits_equal 行")
                    whatif[(kind, batch)] = v
                    continue
                key = (arm, kind, phase, batch)
                if key not in EXPECTED_BACKEND:
                    raise AggError(f"{path}:{i}: 想定外のセル {key}")
                if key in cells:
                    raise AggError(f"{path}:{i}: セル重複 {key}")
                m, n = rec.get("median_s"), rec.get("n")
                if not _finite(m) or m <= 0.0:
                    raise AggError(f"{path}:{i}: median_s が有限の正値でない {key} ({m!r})")
                if isinstance(n, bool) or n != EXPECTED_N:
                    raise AggError(f"{path}:{i}: 計測回数 n={n!r} が {EXPECTED_N} でない {key}")
                for k in ("q1_s", "q3_s"):
                    if not _finite(rec.get(k)):
                        raise AggError(f"{path}:{i}: {k} が有限でない {key}")
                cells[key] = rec
    except OSError as e:
        raise AggError(f"{path}: 読み込み不可 ({e})")
    for key in EXPECTED_BACKEND:
        if key not in cells:
            raise AggError(f"{path}: セル欠落 {key}")
    if set(whatif) != {("l1", b) for b in BATCHES}:
        raise AggError(f"{path}: bits_equal 行の欠落")
    return cells, whatif


EXPECTED_FACADE = expected_facade()
EXPECTED_BACKEND = expected_backend()


def load_gate(d):
    """load_gate_status.txt を読み、(未通過 run 一覧, record_only run 一覧) を返す（欠落・不正は fail-closed）。"""
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


def check_checksums(f_runs, b_runs):
    """RULE.txt checksum 規則。run 間・腕間の不一致と欠落を fail-closed で検出する。"""
    for b in BATCHES:
        vals = {c[("public", b)] for _, c in f_runs} | {c[("decomposed", b)] for _, c in f_runs}
        if len(vals) != 1:
            raise AggError(f"checksum 不一致: public/decomposed batch={b} {sorted(vals)}")
    vals = {c[(a, BENCH_BATCH)] for _, c in f_runs for a in READOUT_ARMS[:3]}
    if len(vals) != 1:
        raise AggError(f"checksum 不一致: readout r0/r1/r2 {sorted(vals)}")
    if len({c[(READOUT_ARMS[3], BENCH_BATCH)] for _, c in f_runs}) != 1:
        raise AggError("checksum 不一致: readout r3_cpu_control が run 間で異なる")
    for b in BATCHES:
        for kind in ("l1", "l2", "relu", "mse"):
            vals = []
            for cells, _ in b_runs:
                v = cells[("prod", kind, "total", b)].get("checksum_bits")
                if not v:
                    raise AggError(f"checksum_bits 欠落: prod/{kind}/total batch={b}")
                vals.append(v)
            if len(set(vals)) != 1:
                raise AggError(f"checksum 不一致: prod/{kind} batch={b} が run 間で異なる")


def med(runs, key):
    vals = [r[key]["median_s"] for r in runs]
    return statistics.median(vals), min(vals), max(vals)


def us(x):
    return f"{x * 1e6:.1f}"


def verdict(share):
    if share >= SUPPORT:
        return "支持（支配項）"
    if share >= CONTRIB:
        return "寄与あり"
    return "非支持"


def aggregate(f_runs, b_runs, unpassed=(), record_only=()):
    check_checksums(f_runs, b_runs)
    fc = [c for c, _ in f_runs]
    bc = [c for c, _ in b_runs]
    out = []
    if record_only:
        out.append(f"【record_only（GB10 以外・負荷ゲート非適用）】run: {', '.join(record_only)}"
                   "（RULE.txt「rtx3060 ほか: record_only」。GB10 の受け入れ判定とは別の参考系列）")
        out.append("")
    if unpassed:
        out.append(f"【参考扱い】GB10 負荷ゲート未通過の run: {', '.join(unpassed)}（RULE.txt ゲート節）")
        out.append("")
    out.append("checksum: run 間・腕間一致 OK")
    out.append("")

    # ---- forward_resident ----
    out.append("## forward_resident（上位。5 run 中央値。単位 us）")
    out.append("")
    out.append("| batch | arm/phase | median | min-max |")
    out.append("|---|---|---|---|")
    for b in BATCHES:
        for arm, ph in (("public", "forward_resident"), ("decomposed", "register"),
                        ("decomposed", "l1_linear_relu"), ("decomposed", "l2_linear"),
                        ("decomposed", "mse_loss"), ("paired", "residual")):
            m, lo, hi = med(fc, (arm, ph, b))
            out.append(f"| {b} | {arm}/{ph} | {us(m)} | {us(lo)}-{us(hi)} |")
    out.append("")

    for b in BATCHES:
        F = med(fc, ("public", "forward_resident", b))[0]
        g = lambda arm, kind, ph: med(bc, (arm, kind, ph, b))[0]
        fid = {k: g("nosync", k, "total") / g("prod", k, "total") for k in ("l1", "l2")}
        fid_bad = [k for k, v in fid.items() if not (FIDELITY_LO <= v <= FIDELITY_HI)]
        terms = {
            "H1 層境界のホスト往復": sum(g("nosync", k, "h2d_upload") + g("nosync", k, "d2h_download")
                                        for k in ("l1", "l2")),
            "H2 未融合 relu の往復": g("prod", "relu", "total"),
            "H3 呼び出しごとのデバイス確保": sum(g("nosync", k, "mem_new") + g("nosync", k, "alloc_c")
                                                for k in ("l1", "l2")),
            "H4 カーネル実行(syncsplit)": sum(g("syncsplit", k, "kernel_wait") for k in ("l1", "l2")),
            "H5 上位 API 固定費(register+残差)": med(fc, ("decomposed", "register", b))[0]
                                                 + med(fc, ("paired", "residual", b))[0],
            "H6 mse の往復": g("prod", "mse", "total"),
        }
        out.append(f"## batch={b}: F = public/forward_resident = {us(F)} us")
        out.append("")
        out.append("| 項 | 値(us) | F 比 | 判定 |")
        out.append("|---|---|---|---|")
        dominant = []
        for name, v in terms.items():
            share = v / F
            note = verdict(share)
            if fid_bad and name.startswith(("H1", "H3")):
                note += "（fidelity 外のため参考扱い）"
            elif share >= SUPPORT:
                dominant.append(name)
            out.append(f"| {name} | {us(v)} | {share:.2f} | {note} |")
        out.append("")
        out.append("支配項: " + (", ".join(dominant) if dominant else "分散（どの項も 0.40 未満）"))
        out.append("fidelity(nosync total / prod total): "
                   + ", ".join(f"{k}={v:.2f}" for k, v in fid.items())
                   + ("（0.8〜1.2 外あり: " + ",".join(fid_bad) + "）" if fid_bad else "（範囲内）"))
        for k in ("l1", "l2"):
            tot = g("nosync", k, "total")
            dom = [ph for ph in NOSYNC_PHASES[:-1] if g("nosync", k, ph) / tot >= SUPPORT]
            out.append(f"層 {k} の支配区間(nosync 内比 0.40 以上): " + (", ".join(dom) if dom else "分散"))
        out.append("補助: decomposed の F 比 l1={:.2f} l2={:.2f} mse={:.2f}".format(
            med(fc, ("decomposed", "l1_linear_relu", b))[0] / F,
            med(fc, ("decomposed", "l2_linear", b))[0] / F,
            med(fc, ("decomposed", "mse_loss", b))[0] / F))
        wi = g("whatif", "l1", "total")
        base = g("prod", "l1", "total") + g("prod", "relu", "total")
        bits = {bool(w[("l1", b)]) for _, w in b_runs}
        out.append(f"補助: what-if(融合 relu) total = {us(wi)} us / (prod l1 + prod relu = {us(base)} us)"
                   f" = {wi / base:.2f}（bits_equal={sorted(bits)}。record-only）")
        out.append("")

    F64 = med(fc, ("public", "forward_resident", BENCH_BATCH))[0]
    ratio = F64 / (REF_FORWARD_US * 1e-6)
    out.append(f"非後退 ratio（record_only）= public/forward_resident(batch 64) {us(F64)} us / "
               f"{REF_FORWARD_US} us = {ratio:.2f}"
               + ("（> 1.00: HEAD で後退と記録。スコープ外 Issue 候補）" if ratio > 1.0 else "（<= 1.00）")
               + "。バイナリ（HEAD 対 registry 0.9.0）・ハーネス（テスト対 bench-fandhe）が異なる。")
    out.append("")

    # ---- param_readout ----
    rb = BENCH_BATCH
    r = lambda arm, ph="param_readout": med(fc, (arm, ph, rb))[0]
    R0, R2, R3 = r("readout_r0_mirror"), r("readout_r2_pretouched"), r("readout_r3_cpu_control")
    out.append("## param_readout（batch 64。単位 us）")
    out.append("")
    out.append("| arm/phase | median | min-max |")
    out.append("|---|---|---|")
    for arm in READOUT_ARMS:
        for ph in (R1_PHASES if arm == "readout_r1_split" else ["param_readout"]):
            m, lo, hi = med(fc, (arm, ph, rb))
            if ph == "noncontig_grad_count":
                out.append(f"| {arm}/{ph} | {m:.1f}（個数） | {lo:.1f}-{hi:.1f} |")
            else:
                out.append(f"| {arm}/{ph} | {us(m)} | {us(lo)}-{us(hi)} |")
    out.append("")
    p1 = (R0 - R2) / R0
    nc = r("readout_r1_split", "noncontig_grad_count")
    p2 = r("readout_r1_split", "grad_contiguous") / R0
    p3 = R0 / R3
    out.append(f"P1 host alloc / page fault: (R0-R2)/R0 = {p1:.2f} -> {'支持' if p1 >= P_THRESHOLD else '非支持'}")
    out.append(f"P2 非 contiguous の gather: noncontig_grad_count={nc:.1f}・grad_contiguous/R0 = {p2:.2f} -> "
               f"{'支持' if (nc > 0 and p2 >= P_THRESHOLD) else '非支持'}")
    out.append(f"P3 CUDA 由来の差: R0/R3 = {p3:.2f} -> "
               f"{'CUDA 由来の差あり' if p3 > P3_RATIO else '差なし（<= 1.50）'}")
    out.append("P4 sync 待ち: 構造上 0（測定対象外。RULE.txt）")
    return "\n".join(out)


# ---- self-test ----

def synth_facade(bump=None):
    lines = []

    def cell(arm, ph, b, m):
        lines.append({"record": "cell", "arm": arm, "phase": ph, "batch": b, "median_s": m, "q1_s": m * 0.9,
                      "q3_s": m * 1.1, "min_s": m * 0.8, "max_s": m * 1.3, "n": EXPECTED_N})
    for b in BATCHES:
        cell("public", "forward_resident", b, 2.0e-4)
        cell("decomposed", "register", b, 1e-6)
        cell("decomposed", "l1_linear_relu", b, 1.2e-4)
        cell("decomposed", "l2_linear", b, 4e-5)
        cell("decomposed", "mse_loss", b, 3e-5)
        cell("paired", "residual", b, -2e-6)
        lines.append({"record": "checksum", "arm": "public", "batch": b, "bits": f"{b:016x}"})
        lines.append({"record": "checksum", "arm": "decomposed", "batch": b, "bits": f"{b:016x}"})
    for arm in READOUT_ARMS:
        for ph in (R1_PHASES if arm == "readout_r1_split" else ["param_readout"]):
            m = 0.0 if ph == "noncontig_grad_count" else 8e-5
            cell(arm, ph, BENCH_BATCH, m)
        lines.append({"record": "checksum", "arm": arm, "batch": BENCH_BATCH,
                      "bits": "cafe" if arm != "readout_r3_cpu_control" else "beef"})
    return lines


def synth_backend():
    lines = []

    def cell(arm, kind, ph, b, m, chk=None):
        lines.append({"arm": arm, "kind": kind, "phase": ph, "batch": b, "median_s": m, "q1_s": m * 0.9,
                      "q3_s": m * 1.1, "min_s": m * 0.8, "max_s": m * 1.3, "n": EXPECTED_N,
                      "checksum_bits": chk})
    for b in BATCHES:
        for kind in ("l1", "l2"):
            cell("prod", kind, "total", b, 7e-5, f"aa{kind}{b}")
            for ph in NOSYNC_PHASES:
                cell("nosync", kind, ph, b, 7e-5 if ph == "total" else 1e-5)
            for ph in SYNC_PHASES:
                cell("syncsplit", kind, ph, b, 8e-5 if ph == "total" else 1e-5)
        cell("prod", "relu", "total", b, 3e-5, f"bb{b}")
        cell("prod", "mse", "total", b, 2e-5, f"cc{b}")
        cell("whatif", "l1", "total", b, 6e-5)
        lines.append({"arm": "whatif", "kind": "l1", "batch": b, "phase": "bits_equal", "value": True})
    return lines


def dump(path, lines, mut=None):
    with open(path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(lines):
            rec = dict(rec)
            if mut:
                mut(i, rec)
            f.write(json.dumps(rec) + "\n")


def write_gate(d, statuses):
    with open(os.path.join(d, GATE_FILE), "w", encoding="utf-8") as f:
        for n, st in enumerate(statuses, 1):
            f.write(f"run{n} gate={st}\n")


def expect_fail(d, label):
    try:
        run_dir(d)
    except AggError:
        return
    raise SystemExit(f"self-test 失敗: {label} を検出できない")


def self_test():
    import tempfile
    fl, bl = synth_facade(), synth_backend()
    with tempfile.TemporaryDirectory() as d:
        def write_all():
            for n in range(1, RUNS + 1):
                dump(os.path.join(d, f"run{n}.facade.jsonl"), fl)
                dump(os.path.join(d, f"run{n}.backend.jsonl"), bl)
        write_all()
        write_gate(d, ["pass"] * RUNS)
        text = run_dir(d)
        assert "支配項" in text and "param_readout" in text and "非後退 ratio" in text, text
        assert "【参考扱い】" not in text and "【record_only" not in text, text
        write_gate(d, ["pass", "unpassed", "pass", "pass", "pass"])
        assert "【参考扱い】" in run_dir(d)
        write_gate(d, ["record_only"] * RUNS)
        assert "【record_only" in run_dir(d)
        os.remove(os.path.join(d, GATE_FILE))
        expect_fail(d, "ゲート状態ファイル欠落")
        write_gate(d, ["pass"] * RUNS)

        pf = os.path.join(d, "run1.facade.jsonl")
        pb = os.path.join(d, "run1.backend.jsonl")

        def first(pred, fn):
            done = []

            def mut(i, rec):
                if not done and pred(rec):
                    done.append(1)
                    fn(rec)
            return mut
        cellp = lambda r: r.get("record") == "cell" and r["arm"] == "public"
        for label, fn in (("n=1", lambda r: r.__setitem__("n", 1)),
                          ("median NaN", lambda r: r.__setitem__("median_s", float("nan"))),
                          ("median 0", lambda r: r.__setitem__("median_s", 0.0)),
                          ("想定外セル", lambda r: r.__setitem__("phase", "bogus"))):
            dump(pf, fl, first(cellp, fn))
            expect_fail(d, "facade " + label)
        dump(pf, fl)
        # residual の負値は許容される（差分セル）
        run_dir(d)
        dump(pf, fl, first(lambda r: r.get("record") == "checksum" and r["arm"] == "decomposed" and r["batch"] == 64,
                           lambda r: r.__setitem__("bits", "dead")))
        expect_fail(d, "public/decomposed の checksum 不一致")
        dump(pf, fl, first(lambda r: r.get("record") == "checksum" and r["arm"] == "readout_r2_pretouched",
                           lambda r: r.__setitem__("bits", "dead")))
        expect_fail(d, "readout 腕間の checksum 不一致")
        dump(pf, fl[:-1])
        expect_fail(d, "facade 行の欠落")
        with open(pf, "w", encoding="utf-8") as f:
            f.write("{not json\n")
        expect_fail(d, "facade 不正 JSON")
        dump(pf, fl)
        backend_prod = lambda r: r["arm"] == "prod" and r["kind"] == "l1" and r["batch"] == 16 and r["phase"] == "total"
        dump(pb, bl, first(backend_prod, lambda r: r.__setitem__("checksum_bits", "dead")))
        expect_fail(d, "backend checksum の run 間不一致")
        dump(pb, bl, first(backend_prod, lambda r: r.__setitem__("checksum_bits", None)))
        expect_fail(d, "backend checksum 欠落")
        dump(pb, bl, first(lambda r: r["arm"] == "nosync", lambda r: r.__setitem__("median_s", -1.0)))
        expect_fail(d, "backend 負の median")
        dump(pb, bl[:-1])
        expect_fail(d, "backend bits_equal 欠落")
        dump(pb, bl)
        # fidelity 外の注記が出る
        dump(pb, bl, lambda i, r: r.__setitem__("median_s", 1e-4) if (r["arm"] == "nosync" and r["phase"] == "total") else None)
        for n in range(2, RUNS + 1):
            dump(os.path.join(d, f"run{n}.backend.jsonl"), bl,
                 lambda i, r: r.__setitem__("median_s", 1e-4) if (r["arm"] == "nosync" and r["phase"] == "total") else None)
        assert "fidelity 外のため参考扱い" in run_dir(d)
    print("self-test OK")


def run_dir(d):
    f_runs = [load_facade(os.path.join(d, f"run{n}.facade.jsonl")) for n in range(1, RUNS + 1)]
    b_runs = [load_backend(os.path.join(d, f"run{n}.backend.jsonl")) for n in range(1, RUNS + 1)]
    unpassed, record_only = load_gate(d)
    return aggregate(f_runs, b_runs, unpassed, record_only)


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
