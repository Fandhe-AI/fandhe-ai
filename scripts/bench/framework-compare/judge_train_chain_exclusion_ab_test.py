"""judge_train_chain_exclusion_ab.py の単体テスト（イシュー #3596）。合成 fixture のみ（実機不要）。

役割: RULE.txt の判定（NON-REGRESSION／REGRESSION／UNDETERMINED／入力不正）が事前登録どおりに
機械適用されることを担保する。CI の deps-forbidden ジョブから `python3 -m unittest` で実行する。
ドリフトガード: RULE.txt の固定値（腕・前提ゲート件数）と判定ツール定数の一致を検査する。
専有ゲート定数（load1 閾値・待機上限）は計測条件でありガード対象外（RULE.txt [ドリフトガードの除外]）。
"""
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("judge", os.path.join(HERE, "judge_train_chain_exclusion_ab.py"))
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)

REAL_RULE = os.path.join(HERE, "..", "..", "..", "docs", "perf", "logs", "train-chain-exclusion-3596", "RULE.txt")
LABEL = "t"
PSHA = "a" * 64
BEFORE = "d" * 40
CREF = "e" * 40
RULE = f"RULE\n  patch_sha256: {PSHA}\n- before = {BEFORE}\n- after_crates_identical_to = {CREF}\n"
CK = 0.123


def wjson(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class Fixture:
    def __init__(self, root):
        self.d = root
        self.rule = os.path.join(root, "RULE.txt")
        with open(self.rule, "w") as fh:
            fh.write(RULE)

    def write(self, name, text):
        with open(os.path.join(self.d, name), "w") as fh:
            fh.write(text)

    def build(self, ratios=None, per=None):
        """既定: off 1.00・on 0.99 の非後退正例。"""
        ratios = ratios or {"off": 1.00, "on": 0.99}
        rule_sha = hashlib.sha256(RULE.encode()).hexdigest()
        for m in ("off", "on"):
            rs = (per or {}).get(m, [ratios[m]] * 5)
            base = [{"task": "train", "device": "cuda", "size": 64, "framework": "fandhe-ai", "mode": "chain",
                     "checksum": CK, "median_s": 1e-4} for _ in range(5)]
            wjson(f"{self.d}/results-excl0-{m}-{LABEL}-cuda-train.jsonl", base)
            wjson(f"{self.d}/results-excl1-{m}-{LABEL}-cuda-train.jsonl",
                  [dict(o, median_s=1e-4 * p) for o, p in zip(base, rs)])
        for arm in judge.ARMS:
            for r in range(1, 6):
                self.write(f"digest-{arm}-{LABEL}-cuda-train-r{r}.log",
                           f"PARAM_DIGEST task=train mode=chain device=cuda n_tensors=4 n_elems=9 fnv1a64=0x{'1' * 16}\n")
        self.write(f"env_info-{LABEL}.txt", "\n".join([
            "profile=gb10", "gpu=NVIDIA GB10", "uname=Linux 6 aarch64", f"rule_sha256={rule_sha}",
            f"patch_sha256={PSHA}", f"rev_stamp_head_before={BEFORE}", f"crates_identical_to={CREF}",
            "precheck=assert", "arms=two_trees_env_toggle", "rounds=5", "tf32=off",
            f"rev_stamp_head_after={'b' * 40}", f"rev_stamp_head_runner={'b' * 40}",
            f"bin_sha256_excl0={'c' * 64}", f"bin_sha256_excl1={'f' * 64}"]) + "\n")
        self.write(f"load_gate-{LABEL}-cuda.log",
                   "series=official threshold=1.0 rule_threshold=1.0 precheck=assert\n" +
                   "".join(f"round{i} dev=cuda gate=pass gate_ok=1 threshold=1.0 waited_s=0 load1=0.1\n" for i in range(1, 6)))
        self.write(f"skipped-train-chain-exclusion-{LABEL}.log", "")
        for name, n in judge.PREGATES.items():
            body = f"test result: ok. {n} passed; 0 failed; 0 ignored\nrc=0\n"
            if name in ("g4-off", "g4-on"):
                body = "BITS a 1\nBITS b 2\n" + body
            self.write(f"{name}-{LABEL}.log", body)
        self.write(f"bits-off-{LABEL}.txt", "BITS a 1\nBITS b 2\n")
        self.write(f"bits-on-{LABEL}.txt", "BITS a 1\nBITS b 2\n")


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.fx = Fixture(self._tmp.name)

    def run_judge(self):
        return judge.judge(self.fx.d, LABEL, self.fx.rule)

    def test_non_regression(self):
        self.fx.build()
        self.assertEqual(self.run_judge()[0], "NON-REGRESSION")

    def test_ratio_boundary_exactly_one_passes(self):
        self.fx.build({"off": 1.00, "on": 1.00})
        self.assertEqual(self.run_judge()[0], "NON-REGRESSION")

    def test_regression_when_all_rounds_slower(self):
        self.fx.build({"off": 1.05, "on": 0.99})
        v, _, _, reject, _ = self.run_judge()
        self.assertEqual(v, "REGRESSION")
        self.assertTrue(any("off" in r for r in reject))

    def test_undetermined_when_sign_not_consistent(self):
        self.fx.build({"off": 1.01, "on": 0.99}, per={"off": [1.05, 1.05, 1.05, 1.05, 0.90]})
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_checksum_mismatch_is_regression(self):
        self.fx.build()
        p = f"{self.fx.d}/results-excl1-on-{LABEL}-cuda-train.jsonl"
        rows = [json.loads(x) for x in open(p)]
        rows[2]["checksum"] = 9.9
        wjson(p, rows)
        self.assertEqual(self.run_judge()[0], "REGRESSION")

    def test_digest_missing_is_regression(self):
        self.fx.build()
        os.remove(f"{self.fx.d}/digest-excl1-off-{LABEL}-cuda-train-r3.log")
        self.assertEqual(self.run_judge()[0], "REGRESSION")

    def test_short_runs_undetermined(self):
        self.fx.build()
        p = f"{self.fx.d}/results-excl0-off-{LABEL}-cuda-train.jsonl"
        rows = [json.loads(x) for x in open(p)][:4]
        wjson(p, rows)
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_missing_results_undetermined(self):
        self.fx.build()
        os.remove(f"{self.fx.d}/results-excl1-on-{LABEL}-cuda-train.jsonl")
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_extra_line_is_bad_input(self):
        self.fx.build()
        with open(f"{self.fx.d}/results-excl0-off-{LABEL}-cuda-train.jsonl", "a") as fh:
            fh.write("not json\n")
        with self.assertRaises(judge.Bad):
            self.run_judge()

    def test_wrong_mode_is_bad_input(self):
        self.fx.build()
        p = f"{self.fx.d}/results-excl0-off-{LABEL}-cuda-train.jsonl"
        rows = [json.loads(x) for x in open(p)]
        rows[0]["mode"] = "reuse"
        wjson(p, rows)
        with self.assertRaises(judge.Bad):
            self.run_judge()

    def test_rule_sha_mismatch_undetermined(self):
        self.fx.build()
        with open(self.fx.rule, "a") as fh:
            fh.write("changed\n")
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_before_sha_mismatch_undetermined(self):
        self.fx.build()
        p = f"{self.fx.d}/env_info-{LABEL}.txt"
        t = open(p).read().replace(f"rev_stamp_head_before={BEFORE}", f"rev_stamp_head_before={'0' * 40}")
        open(p, "w").write(t)
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_same_binary_undetermined(self):
        self.fx.build()
        p = f"{self.fx.d}/env_info-{LABEL}.txt"
        t = open(p).read().replace("f" * 64, "c" * 64)
        open(p, "w").write(t)
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_non_gb10_undetermined(self):
        self.fx.build()
        p = f"{self.fx.d}/env_info-{LABEL}.txt"
        with open(p) as fh:
            t = fh.read().replace("NVIDIA GB10", "NVIDIA H100")
        with open(p, "w") as fh:
            fh.write(t)
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_gate_fail_undetermined(self):
        self.fx.build()
        p = f"{self.fx.d}/load_gate-{LABEL}-cuda.log"
        with open(p) as fh:
            t = fh.read().replace("gate=pass gate_ok=1", "gate=fail gate_ok=0", 1)
        with open(p, "w") as fh:
            fh.write(t)
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_g1_failure_is_regression(self):
        self.fx.build()
        self.fx.write(f"g1-unfenced-{LABEL}.log", "test result: FAILED. 0 passed; 1 failed;\nrc=101\n")
        self.assertEqual(self.run_judge()[0], "REGRESSION")

    def test_g1_missing_undetermined(self):
        self.fx.build()
        os.remove(f"{self.fx.d}/g1-unfenced-{LABEL}.log")
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_bits_mismatch_is_regression(self):
        self.fx.build()
        self.fx.write(f"g4-on-{LABEL}.log", "BITS a 1\nBITS b 3\ntest result: ok. 1 passed; 0 failed; 0 ignored\nrc=0\n")
        self.fx.write(f"bits-on-{LABEL}.txt", "BITS a 1\nBITS b 3\n")
        self.assertEqual(self.run_judge()[0], "REGRESSION")

    def test_skipped_nonempty_undetermined(self):
        self.fx.build()
        self.fx.write(f"skipped-train-chain-exclusion-{LABEL}.log", "failed\n")
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")

    def test_pregate_stop_mark_with_failure_is_regression(self):
        self.fx.build()
        self.fx.write(f"pregate-stop-{LABEL}.txt", "pregate g1 FAILED\n")
        self.fx.write(f"g1-unfenced-{LABEL}.log", "rc=101\n")
        self.assertEqual(self.run_judge()[0], "REGRESSION")

    def test_pregate_stop_mark_without_evidence_undetermined(self):
        self.fx.build()
        self.fx.write(f"pregate-stop-{LABEL}.txt", "claimed\n")
        self.assertEqual(self.run_judge()[0], "UNDETERMINED")


class RuleDriftGuardTest(unittest.TestCase):
    """RULE.txt の固定値と判定ツール定数の一致（専有ゲート定数は対象外）。"""

    def setUp(self):
        with open(REAL_RULE, encoding="utf-8") as fh:
            self.text = fh.read()

    def test_rule_parses(self):
        sha, rule = judge.read_rule(REAL_RULE)
        self.assertEqual(len(sha), 64)
        self.assertEqual(rule["patch"], "92d0a8d446790feedfb0c3392c76c3641ac7977a1a604e26f68f5c811ac3a819")
        self.assertEqual(rule["crates_ref"][:9], "d3f38f71b")
        self.assertEqual(rule["before"][:9], "f7fe6b4df")

    def test_arms_and_pregates_documented(self):
        for arm in judge.ARMS:
            self.assertIn(arm, self.text)
        for name, n in judge.PREGATES.items():
            self.assertRegex(self.text, rf"{re.escape(name)}.*{n} passed")

    def test_rounds_and_limit(self):
        self.assertEqual(judge.ROUNDS, 5)
        self.assertEqual(judge.LIMIT_RATIO, 1.00)
        self.assertIn("ratio <= 1.00", self.text)


if __name__ == "__main__":
    unittest.main()
