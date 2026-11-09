#!/usr/bin/env python3
"""judge_eager_release_ab.py の境界値テスト（0.95／1.00 境界・5/5 一貫性・checksum・件数・機体欠落・不正行）。"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_eager_release_ab as J  # noqa: E402


def make(profile, treated_ratio=0.9, other_ratio=0.99, checksum=1.0, drop=None, bad_line=None):
    """(before_path, after_path) を返す。1 ファイルに全セルを round 順で書く。"""
    d = tempfile.mkdtemp()
    bp, ap = os.path.join(d, "b.jsonl"), os.path.join(d, "a.jsonl")
    with open(bp, "w") as bf, open(ap, "w") as af:
        for rnd in range(5):
            for (t, s, m) in J.expected_cells(profile):
                if drop == (t, s, m) and rnd == 4:
                    continue
                tr = profile == "gb10" and t == "gemm" and m == "reuse" and s in J.TREATED_SIZES
                ratio = treated_ratio if tr else other_ratio
                base = {"framework": "fandhe-ai", "device": "cpu", "task": t, "mode": m,
                        "size": s, "warmup": 3, "iters": 10, "checksum": checksum}
                bf.write(json.dumps({**base, "median_s": 1.0}) + "\n")
                af.write(json.dumps({**base, "median_s": ratio}) + "\n")
        if bad_line:
            bf.write(bad_line + "\n")
    return bp, ap


RULE = {"gb10": "1.0", "macmini": "3.0"}


def make_gate(series="official", precheck="assert", bad_round=None, threshold=None, rounds=5,
              machine="macmini", rule=None):
    threshold = threshold or RULE[machine]
    rule = rule or RULE[machine]
    fd, path = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write(f"threshold={threshold} rule_threshold={rule} series={series} precheck={precheck}\n")
        for i in range(1, rounds + 1):
            ok = i != bad_round
            f.write(f"round{i} gate={'pass' if ok else 'timeout'} gate_ok={1 if ok else 0} "
                    f"threshold={threshold} load1=0.5\n")
    return path


def judge(machines=("gb10", "macmini"), gate=None, **kw):
    args = []
    for m in machines:
        b, a = make(m, **kw)
        g = make_gate(machine=m) if gate is None else (gate if gate == "missing" else make_gate(machine=m, **gate))
        args += [f"--{m}-before", b, f"--{m}-after", a]
        if g != "missing":
            args += [f"--{m}-gate-log", g]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = J.main(args)
    return rc, out.getvalue()


class T(unittest.TestCase):
    def test_adopt(self):
        self.assertEqual(judge()[0], 0)

    def test_boundary_095_adopts(self):
        self.assertEqual(judge(treated_ratio=0.95)[0], 0)

    def test_above_095_is_undetermined(self):
        self.assertEqual(judge(treated_ratio=0.96)[0], 2)

    def test_treated_above_1_rejects(self):
        self.assertEqual(judge(treated_ratio=1.01)[0], 1)

    def test_nontreated_exactly_1_ok_but_above_is_not_adopt(self):
        self.assertEqual(judge(other_ratio=1.0)[0], 0)
        rc, _ = judge(other_ratio=1.01)
        self.assertEqual(rc, 1)  # 全 round 後退のため REJECT

    def test_checksum_mismatch_rejects(self):
        b, a = make("gb10")
        with open(a, "a") as f:
            pass
        lines = open(a).read().splitlines()
        o = json.loads(lines[0])
        o["checksum"] = 2.0
        lines[0] = json.dumps(o)
        open(a, "w").write("\n".join(lines) + "\n")
        mb, ma = make("macmini")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = J.main(["--gb10-before", b, "--gb10-after", a, "--macmini-before", mb, "--macmini-after", ma,
                         "--gb10-gate-log", make_gate(machine="gb10"), "--macmini-gate-log", make_gate(machine="macmini")])
        self.assertEqual(rc, 1)

    def test_missing_machine_is_undetermined(self):
        self.assertEqual(judge(machines=("gb10",))[0], 2)

    def test_missing_row_fail_closed(self):
        self.assertEqual(judge(drop=("gemm", 256, "fresh"))[0], 2)

    def test_bad_line_fail_closed(self):
        self.assertEqual(judge(bad_line="not json")[0], 2)
        self.assertEqual(judge(bad_line="[1]")[0], 2)

    def test_record_only_never_adopts(self):
        args = []
        for m in ("gb10", "macmini"):
            b, a = make(m)
            args += [f"--{m}-before", b, f"--{m}-after", a, f"--{m}-gate-log", make_gate(machine=m)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(J.main(args + ["--record-only"]), 2)

    def test_record_only_regression_is_not_reject(self):
        args = []
        for m in ("gb10", "macmini"):
            b, a = make(m, other_ratio=1.01)
            args += [f"--{m}-before", b, f"--{m}-after", a, f"--{m}-gate-log", make_gate(machine=m)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = J.main(args + ["--record-only"])
        self.assertEqual(rc, 2)
        self.assertIn("verdict: undetermined", out.getvalue())
        self.assertNotIn("verdict: REJECT", out.getvalue())
        self.assertIn("参考", out.getvalue())

    def test_missing_gate_log_is_undetermined(self):
        rc, out = judge(gate="missing")
        self.assertEqual(rc, 2)
        self.assertIn("負荷ゲートログ", out)
        self.assertEqual(judge(gate="missing", treated_ratio=1.01)[0], 2)

    def test_reference_or_smoke_series_is_undetermined(self):
        for series in ("reference", "smoke"):
            self.assertEqual(judge(gate=dict(series=series))[0], 2)
        self.assertEqual(judge(gate=dict(precheck="report-only"))[0], 2)

    def test_gate_failed_round_is_undetermined(self):
        self.assertEqual(judge(gate=dict(bad_round=3))[0], 2)
        self.assertEqual(judge(gate=dict(rounds=4))[0], 2)
        self.assertEqual(judge(gate=dict(bad_round=3), treated_ratio=1.01)[0], 2)

    def test_gate_threshold_must_match_machine_rule(self):
        # GB10 の規則は 1.0。Mac mini 用ログ（threshold=3.0）を GB10 に渡しても受理しない。
        args = []
        for m in ("gb10", "macmini"):
            b, a = make(m)
            args += [f"--{m}-before", b, f"--{m}-after", a, f"--{m}-gate-log", make_gate(machine="macmini")]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = J.main(args)
        self.assertEqual(rc, 2)
        self.assertIn("規則値", out.getvalue())
        # 自己申告の rule_threshold を改ざんして threshold と揃えても、機体の規則値と違えば不受理。
        self.assertEqual(judge(gate=dict(threshold="3.0", rule="3.0"), machines=("gb10",))[0], 2)
        # 規則値より緩い閾値（macmini に 5.0）も不受理。
        self.assertEqual(judge(gate=dict(threshold="5.0", rule="5.0"), machines=("macmini",))[0], 2)


def make4158(treated_ratio=0.9, other_ratio=0.99, checksum=1.0, drop=None, extra_tag=None):
    """4158 のセル集合（診断経路セルを含む）を書き出した (before, after) を返す。"""
    d = tempfile.mkdtemp()
    bp, ap = os.path.join(d, "b.jsonl"), os.path.join(d, "a.jsonl")
    with open(bp, "w") as bf, open(ap, "w") as af:
        for rnd in range(5):
            for (t, s, m, diag) in J.expected_cells_4158():
                if drop == (t, s, m, diag) and rnd == 4:
                    continue
                tr = t == "gemm" and m == "reuse" and s in J.TREATED_SIZES and diag != ""
                ratio = treated_ratio if tr else other_ratio
                base = {"framework": "fandhe-ai", "device": "cpu", "task": t, "mode": m,
                        "size": s, "warmup": 3, "iters": 10, "checksum": checksum}
                if diag:
                    base["diag"] = extra_tag or diag
                bf.write(json.dumps({**base, "median_s": 1.0}) + "\n")
                af.write(json.dumps({**base, "median_s": ratio}) + "\n")
    return bp, ap


def make_gate4158(procs="0"):
    path = make_gate(machine="gb10")
    lines = open(path).read().splitlines()
    out = [lines[0]] + [ln + f" compute_procs={procs}" for ln in lines[1:]]
    open(path, "w").write("\n".join(out) + "\n")
    return path


def judge4158(gate=None, **kw):
    b, a = make4158(**kw)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = J.main(["--rule-set", "4158", "--gb10-before", b, "--gb10-after", a,
                     "--gb10-gate-log", gate or make_gate4158()])
    return rc, out.getvalue()


class T4158(unittest.TestCase):
    def test_adopt_single_machine(self):
        self.assertEqual(judge4158()[0], 0)

    def test_boundary_and_reject(self):
        self.assertEqual(judge4158(treated_ratio=0.95)[0], 0)
        self.assertEqual(judge4158(treated_ratio=0.96)[0], 2)
        self.assertEqual(judge4158(treated_ratio=1.01)[0], 1)

    def test_guard_cell_regression(self):
        self.assertEqual(judge4158(other_ratio=1.01)[0], 1)

    def test_checksum_and_missing_cell(self):
        self.assertEqual(judge4158(drop=("gemm", 4096, "reuse", J.DIAG_TAG))[0], 2)

    def test_unknown_diag_tag_fail_closed(self):
        self.assertEqual(judge4158(extra_tag="other")[0], 2)

    def test_requires_zero_compute_procs(self):
        rc, out = judge4158(gate=make_gate4158(procs="1"))
        self.assertEqual(rc, 2)
        self.assertIn("compute", out)
        # compute_procs 欠落（3226 形式のログ）も不受理。
        self.assertEqual(judge4158(gate=make_gate(machine="gb10"))[0], 2)

    def test_3226_rejects_diag_rows(self):
        b, a = make4158()
        with contextlib.redirect_stdout(io.StringIO()):
            rc = J.main(["--gb10-before", b, "--gb10-after", a, "--gb10-gate-log", make_gate(machine="gb10")])
        self.assertEqual(rc, 2)


def make4159(judged_ratio=0.99, aa_ratio=1.0, checksum=1.0, drop=None, extra_tag=None, aa_cell_ratios=None,
             judged_rounds=None, dup=False):
    """4159 のセル集合（17 セル）。judged_rounds は判定セルの round 別 ratio、dup は 1 行重複。"""
    d = tempfile.mkdtemp()
    bp, ap = os.path.join(d, "b.jsonl"), os.path.join(d, "a.jsonl")
    with open(bp, "w") as bf, open(ap, "w") as af:
        for rnd in range(5):
            for (t, s, m, diag) in J.expected_cells_4159():
                if drop == (t, s, m, diag) and rnd == 4:
                    continue
                if diag:
                    ratio = judged_rounds[rnd] if judged_rounds else judged_ratio
                else:
                    ratio = aa_ratio
                base = {"framework": "fandhe-ai", "device": "cpu", "task": t, "mode": m,
                        "size": s, "warmup": 3, "iters": 10, "checksum": checksum}
                if diag:
                    base["diag"] = extra_tag or diag
                bf.write(json.dumps({**base, "median_s": 1.0}) + "\n")
                af.write(json.dumps({**base, "median_s": ratio}) + "\n")
                if dup and rnd == 0 and diag:
                    af.write(json.dumps({**base, "median_s": ratio}) + "\n")
    return bp, ap


def judge4159(gate=None, extra=(), machine_gb10=False, **kw):
    b, a = make4159(**kw)
    args = ["--rule-set", "4159", "--macmini-before", b, "--macmini-after", a,
            "--macmini-gate-log", gate or make_gate(machine="macmini")]
    if machine_gb10:
        args += ["--gb10-before", b, "--gb10-after", a, "--gb10-gate-log", make_gate(machine="gb10")]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = J.main(args + list(extra))
    return rc, out.getvalue()


class T4159(unittest.TestCase):
    def test_nonregress_and_aa_median_regress_does_not_count(self):
        # A/A 参照セルが全 round 1.00 ちょうど・判定セル 0.99。A/A の中央値後退は判定に効かない。
        rc, out = judge4159(judged_ratio=0.99, aa_ratio=1.0)
        self.assertEqual(rc, 0)
        self.assertIn("verdict: NONREGRESS", out)

    def test_judged_boundary_exactly_1_is_nonregress(self):
        self.assertEqual(judge4159(judged_ratio=1.0)[0], 0)

    def test_judged_all_rounds_regress(self):
        rc, out = judge4159(judged_ratio=1.01)
        self.assertEqual(rc, 1)
        self.assertIn("verdict: REGRESS", out)

    def test_judged_median_above_1_not_all_rounds_is_undetermined(self):
        rc, _ = judge4159(judged_rounds=[1.02, 1.02, 1.02, 0.98, 0.98])
        self.assertEqual(rc, 2)

    def test_aa_cell_all_rounds_regress_is_undetermined_not_regress(self):
        rc, out = judge4159(judged_ratio=0.99, aa_ratio=1.02)
        self.assertEqual(rc, 2)
        self.assertNotIn("verdict: REGRESS", out)

    def test_checksum_mismatch_regresses(self):
        b, a = make4159()
        lines = open(a).read().splitlines()
        o = json.loads(lines[0])
        o["checksum"] = 2.0
        lines[0] = json.dumps(o)
        open(a, "w").write("\n".join(lines) + "\n")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = J.main(["--rule-set", "4159", "--macmini-before", b, "--macmini-after", a,
                         "--macmini-gate-log", make_gate(machine="macmini")])
        self.assertEqual(rc, 1)

    def test_fail_closed_inputs(self):
        self.assertEqual(judge4159(drop=("gemm", 256, "fresh", ""))[0], 2)
        self.assertEqual(judge4159(drop=("train", 64, "reuse", J.DIAG_TAG))[0], 2)
        self.assertEqual(judge4159(dup=True)[0], 2)
        self.assertEqual(judge4159(extra_tag="other")[0], 2)

    def test_no_4096_cell(self):
        self.assertFalse(any(c[1] == 4096 for c in J.expected_cells_4159()))
        self.assertEqual(len(J.expected_cells_4159()), 17)

    def test_gb10_input_rejected_and_4158_rejects_macmini(self):
        self.assertEqual(judge4159(machine_gb10=True)[0], 2)
        b, a = make4159()
        with contextlib.redirect_stdout(io.StringIO()):
            rc = J.main(["--rule-set", "4158", "--macmini-before", b, "--macmini-after", a,
                         "--macmini-gate-log", make_gate(machine="macmini")])
        self.assertEqual(rc, 2)

    def test_gate_log_conditions(self):
        for kw in (dict(series="reference"), dict(series="smoke"), dict(precheck="report-only"),
                   dict(bad_round=3), dict(rounds=4), dict(threshold="1.0", rule="1.0")):
            self.assertEqual(judge4159(gate=make_gate(machine="macmini", **kw))[0], 2, kw)

    def test_compute_procs_column_not_required(self):
        self.assertEqual(judge4159(gate=make_gate(machine="macmini"))[0], 0)

    def test_record_only_never_decides(self):
        self.assertEqual(judge4159(extra=["--record-only"])[0], 2)
        rc, out = judge4159(judged_ratio=1.01, extra=["--record-only"])
        self.assertEqual(rc, 2)
        self.assertNotIn("verdict: REGRESS", out)
        self.assertIn("参考", out)

    def test_no_adopt_label(self):
        self.assertNotIn("ADOPT", judge4159()[1])


if __name__ == "__main__":
    unittest.main()
