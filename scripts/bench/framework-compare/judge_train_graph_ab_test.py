"""judge_train_graph_ab.py の単体テスト（イシュー #3261）。合成 fixture のみ（実機不要）。

役割: RULE.txt の判定（ADOPT／REJECT／UNDETERMINED／入力不正）が事前登録どおりに機械適用されることを担保する。
CI の deps-forbidden ジョブから `python3 -m unittest` で実行する。
"""
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("judge", os.path.join(HERE, "judge_train_graph_ab.py"))
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)

LABEL = "t"
PSHA = "a" * 64
RULE = f"RULE\n  patch_sha256: {PSHA}\n"
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
        """既定: chain 0.90・reuse 0.99・fresh 0.99 の ADOPT 正例。"""
        ratios = ratios or {"chain": 0.90, "reuse": 0.99, "fresh": 0.99}
        rule_sha = hashlib.sha256(RULE.encode()).hexdigest()
        for cell, ratio in ratios.items():
            rs = (per or {}).get(cell, [ratio] * 5)
            off = [{"task": "train", "device": "cuda", "size": 64, "framework": "fandhe-ai", "mode": cell,
                    "checksum": CK, "median_s": 1e-4} for _ in range(5)]
            on = [dict(o, median_s=1e-4 * p) for o, p in zip(off, rs)]
            wjson(f"{self.d}/results-{cell}-off-{LABEL}-cuda-train.jsonl", off)
            wjson(f"{self.d}/results-{cell}-on-{LABEL}-cuda-train.jsonl", on)
            if cell in ("chain", "reuse"):
                for arm in ("off", "on"):
                    for r in range(1, 6):
                        self.write(f"digest-{cell}-{arm}-{LABEL}-cuda-train-r{r}.log",
                                   f"PARAM_DIGEST task=train mode={cell} device=cuda n_tensors=4 n_elems=9 fnv1a64=0x{'1' * 16}\n")
        for opp, fw in (("candle", "candle"), ("torch", "pytorch")):
            wjson(f"{self.d}/results-{opp}-{LABEL}-cuda-train.jsonl",
                  [{"task": "train", "device": "cuda", "size": 64, "framework": fw, "checksum": 1.0, "median_s": 2e-4}] * 5)
        self.write(f"env_info-{LABEL}.txt",
                   "\n".join(["profile=gb10", "gpu=NVIDIA GB10", "uname=Linux 6 aarch64", f"rule_sha256={rule_sha}",
                              f"patch_sha256={PSHA}", "precheck=assert", "arms=same_binary_env_toggle", "rounds=5",
                              "tf32=off", f"rev_stamp_head_runner={'b' * 40}", f"bin_sha256={'c' * 64}"]) + "\n")
        self.write(f"load_gate-{LABEL}-cuda.log",
                   "series=official threshold=1.0 rule_threshold=1.0 precheck=assert\n" +
                   "".join(f"round{i} dev=cuda gate=pass gate_ok=1 threshold=1.0 waited_s=0 load1=0.1\n" for i in range(1, 6)))
        self.write(f"skipped-train-graph-{LABEL}.log", "")
        for name, n in judge.PREGATES.items():
            body = f"test result: ok. {n} passed; 0 failed; 0 ignored\nrc=0\n"
            if name.startswith("r2-"):
                body = "BITS a 1\nBITS b 2\n" + body
            self.write(f"{name}-{LABEL}.log", body)
        self.write(f"bits-off-{LABEL}.txt", "BITS a 1\nBITS b 2\n")
        self.write(f"bits-on-{LABEL}.txt", "BITS a 1\nBITS b 2\n")


class JudgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fx = Fixture(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_judge(self):
        return judge.judge(self.tmp, LABEL, self.fx.rule)[0]

    def test_adopt(self):
        self.fx.build()
        self.assertEqual(self.run_judge(), "ADOPT")

    def test_chain_between_is_undetermined(self):
        self.fx.build({"chain": 0.96, "reuse": 0.99, "fresh": 0.99})
        self.assertEqual(self.run_judge(), "UNDETERMINED")

    def test_sign_flip_in_one_round_is_undetermined(self):
        self.fx.build(per={"chain": [0.9, 0.9, 0.9, 0.9, 1.01]})
        self.assertEqual(self.run_judge(), "UNDETERMINED")

    def test_chain_regress_rejects(self):
        self.fx.build({"chain": 1.05, "reuse": 0.99, "fresh": 0.99})
        self.assertEqual(self.run_judge(), "REJECT")

    def test_control_cell_regress_rejects(self):
        self.fx.build({"chain": 0.9, "reuse": 1.02, "fresh": 0.99})
        self.assertEqual(self.run_judge(), "REJECT")

    def test_checksum_mismatch_rejects(self):
        self.fx.build()
        p = f"{self.tmp}/results-chain-on-{LABEL}-cuda-train.jsonl"
        rows = [json.loads(x) for x in open(p)]
        rows[2]["checksum"] = 9.9
        wjson(p, rows)
        self.assertEqual(self.run_judge(), "REJECT")

    def test_missing_digest_rejects(self):
        self.fx.build()
        os.remove(f"{self.tmp}/digest-chain-on-{LABEL}-cuda-train-r3.log")
        self.assertEqual(self.run_judge(), "REJECT")

    def test_four_runs_undetermined(self):
        self.fx.build()
        p = f"{self.tmp}/results-fresh-on-{LABEL}-cuda-train.jsonl"
        rows = open(p).read().splitlines()[:4]
        open(p, "w").write("\n".join(rows) + "\n")
        self.assertEqual(self.run_judge(), "UNDETERMINED")

    def test_gate_fail_undetermined(self):
        self.fx.build()
        p = f"{self.tmp}/load_gate-{LABEL}-cuda.log"
        open(p, "w").write(open(p).read().replace("round3 dev=cuda gate=pass gate_ok=1", "round3 dev=cuda gate=fail gate_ok=0"))
        self.assertEqual(self.run_judge(), "UNDETERMINED")

    def test_rule_sha_mismatch_undetermined(self):
        self.fx.build()
        open(self.fx.rule, "a").write("changed\n")
        self.assertEqual(self.run_judge(), "UNDETERMINED")

    def test_r2_diff_rejects(self):
        self.fx.build()
        self.fx.write(f"bits-on-{LABEL}.txt", "BITS a 1\nBITS b 3\n")
        self.assertEqual(self.run_judge(), "REJECT")

    def test_r2_empty_bits_undetermined(self):
        self.fx.build()
        self.fx.write(f"bits-off-{LABEL}.txt", "")
        self.fx.write(f"bits-on-{LABEL}.txt", "")
        self.assertEqual(self.run_judge(), "UNDETERMINED")

    def test_r1_failure_rejects(self):
        self.fx.build()
        self.fx.write(f"r1-facade-{LABEL}.log", "test result: ok. 1 passed; 0 failed;\nrc=101\n")
        self.assertEqual(self.run_judge(), "REJECT")

    def test_r0_missing_undetermined(self):
        self.fx.build()
        os.remove(f"{self.tmp}/r0-3252-{LABEL}.log")
        self.assertEqual(self.run_judge(), "UNDETERMINED")

    def test_skipped_nonempty_undetermined(self):
        self.fx.build()
        self.fx.write(f"skipped-train-graph-{LABEL}.log", "task=train arm=torch round=2: boom\n")
        self.assertEqual(self.run_judge(), "UNDETERMINED")

    def test_bad_json_is_bad_input(self):
        self.fx.build()
        open(f"{self.tmp}/results-chain-off-{LABEL}-cuda-train.jsonl", "a").write("{not json\n")
        with self.assertRaises(judge.Bad):
            self.run_judge()

    def test_type_error_is_bad_input(self):
        self.fx.build()
        p = f"{self.tmp}/results-chain-off-{LABEL}-cuda-train.jsonl"
        rows = [json.loads(x) for x in open(p)]
        rows[0]["median_s"] = "fast"
        wjson(p, rows)
        with self.assertRaises(judge.Bad):
            self.run_judge()


if __name__ == "__main__":
    unittest.main()
