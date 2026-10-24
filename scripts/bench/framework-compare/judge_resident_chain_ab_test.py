#!/usr/bin/env python3
"""judge_resident_chain_ab.py（#3257）の単体テスト。fixture を一時ディレクトリへ合成し、
REJECT（checksum／digest）・INVALID（run 数・ゲート・RULE sha・patch sha）・IMPROVE/REGRESS/NEUTRAL を固定する。
実行: python3 -I scripts/bench/framework-compare/judge_resident_chain_ab_test.py"""
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import judge_resident_chain_ab as J  # noqa: E402

LABEL = "t"


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Fixture:
    def __init__(self, before_ms, after_ms, checksum_after=None, digest_after=None, drop_digest=False,
                 n_after=5, gate_fail=False, before_extra=None):
        self.root = tempfile.mkdtemp(prefix="j3257-")
        self.rule = os.path.join(self.root, "RULE.txt")
        self.rule_bytes = b"RULE test\n"
        with open(self.rule, "wb") as fh:
            fh.write(self.rule_bytes)
        self.patch_bytes = b"patch\n"
        with open(os.path.join(self.root, "bench-resident-chain.patch"), "wb") as fh:
            fh.write(self.patch_bytes)
        self.d = os.path.join(self.root, "gb10")
        os.makedirs(self.d)
        dev = "cpu"
        for task in J.TASKS:
            for arm, vals in (("before", before_ms), ("after", after_ms)):
                mode = J.ARM_MODE[arm]
                n = n_after if arm == "after" else 5
                cs = (checksum_after if (arm == "after" and checksum_after is not None) else 1.5)
                with open(f"{self.d}/results-{arm}-{LABEL}-{dev}-{task}.jsonl", "w") as fh:
                    for i in range(n):
                        fh.write(json.dumps({"framework": "fandhe-ai", "task": task, "device": dev, "size": 64,
                                             "mode": mode, "median_s": vals[i] / 1e3, "checksum": cs,
                                             **((before_extra or {}) if arm == "before" else {})}) + "\n")
                if task == "train":
                    for r in range(1, 6):
                        if arm == "after" and drop_digest and r == 3:
                            open(f"{self.d}/digest-{arm}-{LABEL}-{dev}-train-r{r}.log", "w").close()
                            continue
                        h = digest_after if (arm == "after" and digest_after) else "0x" + "a" * 16
                        with open(f"{self.d}/digest-{arm}-{LABEL}-{dev}-train-r{r}.log", "w") as fh:
                            fh.write(f"PARAM_DIGEST task=train mode={mode} device={dev} n_tensors=4 "
                                     f"n_elems=10 fnv1a64={h}\n")
        with open(f"{self.d}/env_info-{LABEL}.txt", "w") as fh:
            fh.write("profile=gb10\n" + f"rule_sha256={sha(self.rule_bytes)}\n"
                     + f"patch_sha256={sha(self.patch_bytes)}\n" + "precheck=assert\n"
                     + "rev_stamp_head_runner=" + "b" * 40 + "\n" + "bin_sha256=" + "c" * 64 + "\n")
        with open(f"{self.d}/load_gate-{LABEL}-{dev}.log", "w") as fh:
            fh.write("series=official threshold=1.0 rule_threshold=1.0 precheck=assert\n")
            for r in range(1, 6):
                g = "fail" if (gate_fail and r == 2) else "pass"
                fh.write(f"round{r} dev={dev} gate={g} gate_ok={'0' if g == 'fail' else '1'}\n")

    def judge(self):
        buf = io.StringIO()
        res = [J.judge_cell("gb10", "cpu", t, self.d, LABEL, sha(self.rule_bytes), sha(self.patch_bytes))
               for t in J.TASKS]
        with redirect_stdout(buf):
            print(J.render(res, sha(self.rule_bytes)))
        return res

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


B = [1.0, 1.1, 1.0, 0.9, 1.0]


class JudgeTest(unittest.TestCase):
    def run_fx(self, **kw):
        fx = Fixture(**kw)
        self.addCleanup(fx.close)
        return fx.judge()

    def test_improve(self):
        r = self.run_fx(before_ms=B, after_ms=[0.5, 0.6, 0.5, 0.5, 0.5])
        self.assertEqual([x["verdict"] for x in r], ["IMPROVE", "IMPROVE"])

    def test_regress(self):
        r = self.run_fx(before_ms=B, after_ms=[1.5, 1.6, 1.5, 1.5, 1.5])
        self.assertEqual(r[0]["verdict"], "REGRESS")

    def test_neutral(self):
        r = self.run_fx(before_ms=B, after_ms=[0.99, 1.1, 0.98, 0.9, 1.0])
        self.assertEqual(r[0]["verdict"], "NEUTRAL")

    def test_checksum_mismatch_rejects(self):
        r = self.run_fx(before_ms=B, after_ms=B, checksum_after=1.6)
        self.assertEqual([x["verdict"] for x in r], ["REJECT", "REJECT"])

    def test_digest_mismatch_rejects_train_only(self):
        r = self.run_fx(before_ms=B, after_ms=B, digest_after="0x" + "f" * 16)
        self.assertEqual(r[0]["verdict"], "REJECT")
        self.assertNotEqual(r[1]["verdict"], "REJECT")

    def test_missing_digest_rejects(self):
        r = self.run_fx(before_ms=B, after_ms=B, drop_digest=True)
        self.assertEqual(r[0]["verdict"], "REJECT")

    def test_before_tf32_on_invalid(self):
        r = self.run_fx(before_ms=B, after_ms=[0.5] * 5, before_extra={"tf32": True})
        self.assertEqual(r[0]["verdict"], "INVALID")

    def test_before_graph_on_invalid(self):
        r = self.run_fx(before_ms=B, after_ms=[0.5] * 5, before_extra={"graph": "on"})
        self.assertEqual(r[0]["verdict"], "INVALID")

    def test_before_tf32_false_ok(self):
        r = self.run_fx(before_ms=B, after_ms=[0.5] * 5, before_extra={"tf32": False})
        self.assertEqual(r[0]["verdict"], "IMPROVE")

    def test_four_runs_invalid(self):
        r = self.run_fx(before_ms=B, after_ms=B, n_after=4)
        self.assertEqual(r[0]["verdict"], "INVALID")

    def test_gate_fail_invalid(self):
        r = self.run_fx(before_ms=B, after_ms=[0.5] * 5, gate_fail=True)
        self.assertEqual(r[0]["verdict"], "INVALID")

    def test_rule_sha_mismatch_invalid(self):
        fx = Fixture(before_ms=B, after_ms=B)
        self.addCleanup(fx.close)
        r = J.judge_cell("gb10", "cpu", "infer", fx.d, LABEL, "0" * 64, sha(fx.patch_bytes))
        self.assertEqual(r["verdict"], "INVALID")

    def test_patch_sha_mismatch_invalid(self):
        fx = Fixture(before_ms=B, after_ms=B)
        self.addCleanup(fx.close)
        r = J.judge_cell("gb10", "cpu", "infer", fx.d, LABEL, sha(fx.rule_bytes), "0" * 64)
        self.assertEqual(r["verdict"], "INVALID")

    def test_main_exit_codes(self):
        fx = Fixture(before_ms=B, after_ms=B, checksum_after=1.6)
        self.addCleanup(fx.close)
        with redirect_stdout(io.StringIO()):
            rc = J.main(["--rule", fx.rule, "--machine", f"gb10={fx.d}:{LABEL}"])
        # cuda デバイスのログは無いので INVALID も混ざるが、REJECT が優先される。
        self.assertEqual(rc, 3)


if __name__ == "__main__":
    unittest.main()
