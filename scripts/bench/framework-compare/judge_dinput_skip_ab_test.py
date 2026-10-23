#!/usr/bin/env python3
"""judge_dinput_skip_ab.py の unittest（stdlib のみ。イシュー #3247）。

実行: cd scripts/bench/framework-compare && python3 -I judge_dinput_skip_ab_test.py
fixture は tempfile 内に生成し、リポジトリへは何も書かない。
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_dinput_skip_ab as J  # noqa: E402

LABEL = "t1"
RULE_TEXT = b"RULE test\n"
RULE_SHA = hashlib.sha256(RULE_TEXT).hexdigest()
DIG = "PARAM_DIGEST task=train mode={m} device={d} n_tensors=6 n_elems=100 fnv1a64=0x0123456789abcdef"


def write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def make_machine(d, profile, devices, ratios, checksum=0.5, digests=None, gate_ok=True,
                 rule_sha=RULE_SHA, nrows=5):
    """ratios: {(dev, mode): [ratio_r x5]}。before は 1.0ms 固定、after = ratio_r ms。"""
    gate = J.PROFILES[profile]["gate"]
    for dev in devices:
        gl = [f"series=official threshold={gate} rule_threshold={gate} precheck=assert"]
        for r in range(1, 6):
            ok = gate_ok or r != 3
            gl.append(f"round{r} dev={dev} gate={'pass' if ok else 'fail'} gate_ok={1 if ok else 0} threshold={gate}")
        write(f"{d}/load_gate-{LABEL}-{dev}.log", "\n".join(gl) + "\n")
        for arm in ("before", "after"):
            lines = []
            for mode in ("fresh", "reuse"):
                rs = ratios[(dev, mode)][:nrows]
                for r, ratio in enumerate(rs, 1):
                    ms = 1e-3 if arm == "before" else 1e-3 * ratio
                    cs = checksum
                    lines.append(json.dumps({"framework": "fandhe-ai", "device": dev, "task": "train",
                                             "mode": mode, "size": 64, "median_s": ms, "checksum": cs}))
            write(f"{d}/results-{arm}-{LABEL}-{dev}-train.jsonl", "\n".join(lines) + "\n")
            for mode in ("fresh", "reuse"):
                for r in range(1, 6):
                    line = (digests or {}).get((arm, dev, mode, r), DIG.format(m=mode, d=dev))
                    write(f"{d}/digest-{arm}-{LABEL}-{dev}-{mode}-r{r}.log", (line or "") + "\n")
    env = {
        "profile": profile, "rule_sha256": rule_sha, "precheck": "assert", "patch_sha256": "a" * 64,
        "rev_stamp_head_before": "b" * 40, "rev_stamp_head_after": "c" * 40, "rev_stamp_head_runner": "d" * 40,
    }
    if profile == "macmini":
        env["cpu_brand"] = "Apple M4"
    write(f"{d}/env_info-{LABEL}.txt", "\n".join(f"{k}={v}" for k, v in env.items()) + "\n")


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.rule = os.path.join(self.tmp.name, "RULE.txt")
        with open(self.rule, "wb") as fh:
            fh.write(RULE_TEXT)

    def cell(self, mode="reuse", **kw):
        d = os.path.join(self.tmp.name, "m")
        os.makedirs(d, exist_ok=True)
        ratios = kw.pop("ratios")
        make_machine(d, "gb10", ("cpu", "cuda"), {(dv, m): ratios for dv in ("cpu", "cuda") for m in ("fresh", "reuse")}, **kw)
        return J.judge_cell("gb10", "gb10", "cpu", mode, d, LABEL, RULE_SHA)

    def test_adopt(self):
        self.assertEqual(self.cell(ratios=[0.9] * 5)["verdict"], "ADOPT")

    def test_adopt_boundary_095(self):
        self.assertEqual(self.cell(ratios=[0.95] * 5)["verdict"], "ADOPT")

    def test_between_is_undetermined(self):
        self.assertEqual(self.cell(ratios=[0.97] * 5)["verdict"], "undetermined")

    def test_ratio_over_one_is_reject(self):
        self.assertEqual(self.cell(ratios=[1.02] * 5)["verdict"], "REJECT")

    def test_ratio_exactly_one_is_undetermined(self):
        self.assertEqual(self.cell(ratios=[1.0] * 5)["verdict"], "undetermined")

    def test_one_round_regression_blocks_adopt(self):
        c = self.cell(ratios=[0.9, 0.9, 0.9, 0.9, 1.0])
        self.assertEqual(c["verdict"], "undetermined")

    def test_all_rounds_regress_is_reject_even_if_median_ok(self):
        # 中央値比 1.00 ちょうどでも 5/5 round で > 1.00 なら不可だが、ここでは ratio 判定を超える例
        self.assertEqual(self.cell(ratios=[1.001] * 5)["verdict"], "REJECT")

    def test_guard_boundary(self):
        self.assertEqual(self.cell(mode="fresh", ratios=[1.03] * 5)["verdict"], "guard-fail")  # 5/5 後退
        self.assertEqual(self.cell(mode="fresh", ratios=[1.03, 1.03, 1.03, 1.03, 0.99])["verdict"], "guard-pass")
        self.assertEqual(self.cell(mode="fresh", ratios=[1.04, 1.04, 1.04, 1.04, 0.99])["verdict"], "guard-fail")

    def test_checksum_mismatch_rejects(self):
        d = os.path.join(self.tmp.name, "m")
        os.makedirs(d)
        make_machine(d, "gb10", ("cpu", "cuda"), {(dv, m): [0.9] * 5 for dv in ("cpu", "cuda") for m in ("fresh", "reuse")})
        p = f"{d}/results-after-{LABEL}-cpu-train.jsonl"
        text = open(p, encoding="utf-8").read().replace('"checksum": 0.5', '"checksum": 0.6', 1)
        write(p, text)
        c = J.judge_cell("gb10", "gb10", "cpu", "fresh", d, LABEL, RULE_SHA)
        self.assertEqual(c["verdict"], "REJECT")

    def test_digest_mismatch_rejects(self):
        bad = {("after", "cpu", "reuse", 2): DIG.format(m="reuse", d="cpu").replace("0123", "ffff")}
        self.assertEqual(self.cell(ratios=[0.9] * 5, digests=bad)["verdict"], "REJECT")

    def test_digest_missing_rejects(self):
        self.assertEqual(self.cell(ratios=[0.9] * 5, digests={("before", "cpu", "reuse", 1): ""})["verdict"], "REJECT")

    def test_gate_failure_is_undetermined(self):
        self.assertEqual(self.cell(ratios=[0.9] * 5, gate_ok=False)["verdict"], "undetermined")

    def test_rule_sha_mismatch_is_undetermined(self):
        c = self.cell(ratios=[0.9] * 5, rule_sha="0" * 64)
        self.assertEqual(c["verdict"], "undetermined")

    def test_short_runs_are_undetermined(self):
        self.assertEqual(self.cell(ratios=[0.9] * 5, nrows=4)["verdict"], "undetermined")

    def test_macmini_brand_must_be_m4(self):
        d = os.path.join(self.tmp.name, "mm")
        os.makedirs(d)
        make_machine(d, "macmini", ("cpu", "metal"), {(dv, m): [0.9] * 5 for dv in ("cpu", "metal") for m in ("fresh", "reuse")})
        p = f"{d}/env_info-{LABEL}.txt"
        write(p, open(p, encoding="utf-8").read().replace("Apple M4", "Apple M4 Max"))
        c = J.judge_cell("macmini", "macmini", "cpu", "reuse", d, LABEL, RULE_SHA)
        self.assertEqual(c["verdict"], "undetermined")

    def run_main(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = J.main(["--rule", self.rule, *extra])
        return rc, out.getvalue()

    def test_main_exit_codes_and_require_both(self):
        d1, d2 = os.path.join(self.tmp.name, "a"), os.path.join(self.tmp.name, "b")
        os.makedirs(d1)
        os.makedirs(d2)
        rr = lambda devs, v: {(dv, m): [v] * 5 for dv in devs for m in ("fresh", "reuse")}
        make_machine(d1, "macmini", ("cpu", "metal"), rr(("cpu", "metal"), 0.9))
        make_machine(d2, "gb10", ("cpu", "cuda"), rr(("cpu", "cuda"), 0.9))
        args = ["--machine", f"macmini={d1}:{LABEL}", "--machine", f"gb10={d2}:{LABEL}", "--require-both"]
        rc, out = self.run_main(*args)
        self.assertEqual(rc, 0, out)
        rc, _ = self.run_main("--machine", f"gb10={d2}:{LABEL}", "--require-both")
        self.assertEqual(rc, 2)

    def test_main_reject_gives_3(self):
        d2 = os.path.join(self.tmp.name, "b")
        os.makedirs(d2)
        make_machine(d2, "gb10", ("cpu", "cuda"), {(dv, m): [1.1] * 5 for dv in ("cpu", "cuda") for m in ("fresh", "reuse")})
        rc, _ = self.run_main("--machine", f"gb10={d2}:{LABEL}")
        self.assertEqual(rc, 3)

    def test_unsafe_label_rejected(self):
        rc, _ = self.run_main("--machine", "gb10=/tmp/x:bad|label")
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
