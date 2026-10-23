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


def make_gate(series="official", precheck="assert", bad_round=None, threshold="3.0", rounds=5):
    fd, path = tempfile.mkstemp(suffix=".log")
    with os.fdopen(fd, "w") as f:
        f.write(f"threshold={threshold} rule_threshold=3.0 series={series} precheck={precheck}\n")
        for i in range(1, rounds + 1):
            ok = i != bad_round
            f.write(f"round{i} gate={'pass' if ok else 'timeout'} gate_ok={1 if ok else 0} "
                    f"threshold={threshold} load1=0.5\n")
    return path


def judge(machines=("gb10", "macmini"), gate=None, **kw):
    args = []
    for m in machines:
        b, a = make(m, **kw)
        g = make_gate() if gate is None else gate
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
                         "--gb10-gate-log", make_gate(), "--macmini-gate-log", make_gate()])
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
            args += [f"--{m}-before", b, f"--{m}-after", a, f"--{m}-gate-log", make_gate()]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(J.main(args + ["--record-only"]), 2)

    def test_record_only_regression_is_not_reject(self):
        args = []
        for m in ("gb10", "macmini"):
            b, a = make(m, other_ratio=1.01)
            args += [f"--{m}-before", b, f"--{m}-after", a, f"--{m}-gate-log", make_gate()]
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
            self.assertEqual(judge(gate=make_gate(series=series))[0], 2)
        self.assertEqual(judge(gate=make_gate(precheck="report-only"))[0], 2)

    def test_gate_failed_round_is_undetermined(self):
        self.assertEqual(judge(gate=make_gate(bad_round=3))[0], 2)
        self.assertEqual(judge(gate=make_gate(rounds=4))[0], 2)
        self.assertEqual(judge(gate=make_gate(bad_round=3), treated_ratio=1.01)[0], 2)


if __name__ == "__main__":
    unittest.main()
