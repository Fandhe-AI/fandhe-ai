"""summarize_breakdown_resident_chain_metal.py の単体テスト（合成 fixture のみ。実機不要。イシュー #3612）。"""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location(
    "sbm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "summarize_breakdown_resident_chain_metal.py")
)
sbm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sbm)

LABEL = "t"


def row(task, arm, **kw):
    r = {"task": task, "arm": arm, "wait_max": 1, "total_us": 400.0, "encode_us": 10.0, "gpu_busy_us": 220.0,
         "host_fixed_us": 150.0, "readout_us": 5.0, "prep_us": 1.0, "command_buffers_max": 1}
    if task == "infer" and arm == "b1":
        r["bit_identical_to_b0"] = True
    r.update(kw)
    return r


def write(d, task, arm, rows):
    with open(os.path.join(d, f"results-3612-{task}-{arm}-{LABEL}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def build(d, mutate=None, gate_ok=5):
    for task in sbm.TASKS:
        for arm in ("b0", "b1"):
            rows = [row(task, arm) for _ in range(5)]
            if mutate:
                rows = mutate(task, arm, rows)
            write(d, task, arm, rows)
    with open(os.path.join(d, f"load_gate-3612-{LABEL}.log"), "w") as f:
        for r in range(1, 6):
            f.write(f"round{r} dev=metal gate={'pass' if r <= gate_ok else 'fail'} gate_ok=1 x\n")


def run(d, *extra):
    buf, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        code = sbm.main(["--dir", d, "--label", LABEL, *extra])
    return code, buf.getvalue(), err.getvalue()


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_ok_without_a0_is_undetermined_conclusion(self):
        build(self.d)
        code, out, _ = run(self.d)
        self.assertEqual(code, 0)
        self.assertIn("判定不能（N 未取得）", out)
        # train: gap1 = 400 - (1+10+220+150+5) = 14
        self.assertIn("gap1（B1 total − Σ区間。計時のすき間）: 14.000", out)

    def test_within_noise(self):
        build(self.d)
        code, out, _ = run(self.d, "--a0-train-us", "500,520,510,505,515", "--a0-infer-us", "300,320,310,305,315")
        self.assertEqual(code, 0)
        self.assertIn("|Δ| ≤ N", out)
        self.assertIn("gap2", out)

    def test_exceeds_noise_lists_upper_bounds(self):
        build(self.d)
        code, out, _ = run(self.d, "--a0-train-us", "500,500,500,500,500", "--a0-infer-us", "300,300,300,300,300")
        self.assertEqual(code, 0)
        self.assertIn("|Δ| > N", out)
        self.assertIn("振り分け不能", out)
        self.assertIn("gpu_busy_us = 220.000", out)

    def test_wait_violation(self):
        def m(task, arm, rows):
            if task == "train" and arm == "b0":
                rows[2]["wait_max"] = 7
            return rows
        build(self.d, m)
        code, out, _ = run(self.d)
        self.assertEqual(code, 1)
        self.assertIn("wait_max=7", out)

    def test_command_buffers_missing(self):
        def m(task, arm, rows):
            if task == "infer" and arm == "b0":
                del rows[0]["command_buffers_max"]
            return rows
        build(self.d, m)
        code, out, _ = run(self.d)
        self.assertEqual(code, 1)
        self.assertIn("command_buffers_max が欠落", out)

    def test_bit_mismatch(self):
        def m(task, arm, rows):
            if task == "infer" and arm == "b1":
                rows[0]["bit_identical_to_b0"] = False
            return rows
        build(self.d, m)
        self.assertEqual(run(self.d)[0], 1)

    def test_gate_not_passed(self):
        build(self.d, gate_ok=4)
        code, out, _ = run(self.d)
        self.assertEqual(code, 1)
        self.assertIn("負荷ゲート", out)

    def test_missing_round_rejected(self):
        build(self.d, lambda t, a, rows: rows[:4])
        self.assertEqual(run(self.d)[0], 2)

    def test_missing_value_rejected(self):
        def m(task, arm, rows):
            if task == "train" and arm == "b1":
                del rows[1]["gpu_busy_us"]
            return rows
        build(self.d, m)
        self.assertEqual(run(self.d)[0], 2)

    def test_nonnumeric_value_rejected(self):
        def m(task, arm, rows):
            if task == "train" and arm == "b1":
                rows[1]["readout_us"] = "NaN"
            return rows
        build(self.d, m)
        self.assertEqual(run(self.d)[0], 2)

    def test_bad_a0_length_and_label(self):
        build(self.d)
        self.assertEqual(run(self.d, "--a0-train-us", "1,2,3")[0], 2)
        self.assertEqual(sbm.main(["--dir", self.d, "--label", "../x"]), 2)

    def test_arm_mismatch_rejected(self):
        build(self.d, lambda t, a, rows: [dict(r, arm="b9") for r in rows] if (t, a) == ("infer", "b0") else rows)
        self.assertEqual(run(self.d)[0], 2)


if __name__ == "__main__":
    unittest.main()
