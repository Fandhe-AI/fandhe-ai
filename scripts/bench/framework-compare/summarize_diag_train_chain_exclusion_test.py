"""summarize_diag_train_chain_exclusion.py の単体テスト（イシュー #3596 r4）。合成 fixture のみ（実機不要）。

役割: DIAG-r4.txt の [読み方の規則]（A/A ノイズ幅 N・Delta_after・revert 腕の分類・結論の対応表）と、
欠損・不一致を寛容に読まない fail-closed 動作が事前登録どおりであることを担保する。
CI の deps-forbidden ジョブから `python3 -m unittest` で実行する。
ドリフトガード: 実 DIAG-r4.txt の腕の集合・回転表がツール定数と一致することを検査する
（専有ゲート定数は計測条件のためガード対象外。DIAG [ドリフトガードの除外]）。
"""
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("summ", os.path.join(HERE, "summarize_diag_train_chain_exclusion.py"))
summ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summ)

REAL_DIAG = os.path.join(HERE, "..", "..", "..", "docs", "perf", "logs", "train-chain-exclusion-3596", "DIAG-r4.txt")
LABEL = "t"
PSHA = "a" * 64
BEFORE = "d" * 40
CREF = "e" * 40
RV = {"rv-scopecall": "1" * 64, "rv-kindlookup": "2" * 64, "rv-graphlock": "3" * 64}
DIAG = (
    f"DIAG\n  patch_sha256: {PSHA}\n- before = {BEFORE}\n- after_crates_identical_to = {CREF}\n"
    + "".join(f"  {k}_sha256: {v}\n" for k, v in RV.items())
)
BASE = 1e-4


def wjson(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class Fixture:
    def __init__(self, root):
        self.d = root
        self.diag = os.path.join(root, "DIAG-r4.txt")
        with open(self.diag, "w") as fh:
            fh.write(DIAG)

    def write(self, name, text):
        with open(os.path.join(self.d, name), "w") as fh:
            fh.write(text)

    def build(self, rho=None):
        """rho: 腕 -> 5 round の aa0 比（既定は全腕 1.0）。aa0 は常に BASE。"""
        rho = rho or {}
        diag_sha = hashlib.sha256(DIAG.encode()).hexdigest()
        for arm in summ.ARMS:
            rs = rho.get(arm, [1.0] * 5)
            rows = [{"task": "train", "device": "cuda", "size": 64, "framework": "fandhe-ai", "mode": "chain",
                     "checksum": 0.5, "median_s": BASE * p} for p in rs]
            wjson(f"{self.d}/results-diag-{arm}-{LABEL}-cuda-train.jsonl", rows)
            for r in range(1, 6):
                self.write(f"digest-diag-{arm}-{LABEL}-cuda-train-r{r}.log",
                           f"PARAM_DIGEST task=train mode=chain device=cuda n_tensors=4 n_elems=9 fnv1a64=0x{'1' * 16}\n")
        env = ["profile=gb10", "gpu=NVIDIA GB10", "uname=Linux 6 aarch64", f"diag_sha256={diag_sha}",
               f"patch_sha256={PSHA}", f"rev_stamp_head_before={BEFORE}", f"crates_identical_to={CREF}",
               "precheck=assert", "rounds=5", "tf32=off"]
        env += [f"{k}_sha256={v}" for k, v in RV.items()]
        for i, arm in enumerate(summ.ARMS):
            same = "9" * 64 if arm in ("aa0", "aa1") else f"{i:x}" * 64
            env.append(f"bin_sha256_{arm}={same}")
        self.write(f"env_info-diag-{LABEL}.txt", "\n".join(env) + "\n")
        self.write(f"load_gate-diag-{LABEL}-cuda.log",
                   "series=official threshold=1.0 rule_threshold=1.0 precheck=assert\n" +
                   "".join(f"round{i} dev=cuda gate=pass gate_ok=1 threshold=1.0 waited_s=0 load1=0.1\n" for i in range(1, 6)))

    def run(self):
        return summ.summarize(self.diag, self.d, LABEL)


class SummarizeTest(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_aa_equivalent_is_case_i(self):
        self.fx.build({"aa1": [1.01, 0.99, 1.0, 1.0, 1.0], "excl1": [1.005] * 5})
        out = self.fx.run()
        self.assertIn("場合 = (i)", out)
        self.assertIn("コードに原因は見つからない", out)

    def test_kindlookup_is_cause_candidate(self):
        self.fx.build({
            "aa1": [1.001] * 5, "excl1": [1.01] * 5,
            "rv-kindlookup": [1.0005] * 5, "rv-scopecall": [1.01] * 5, "rv-graphlock": [1.01] * 5,
        })
        out = self.fx.run()
        self.assertIn("場合 = (ii)", out)
        self.assertIn("- rv-kindlookup: 原因候補", out)
        self.assertIn("resolved_stream_kind の mutex", out)

    def test_scopecall_only_is_cause_candidate(self):
        self.fx.build({
            "aa1": [1.001] * 5, "excl1": [1.01] * 5,
            "rv-scopecall": [1.0005] * 5, "rv-kindlookup": [1.01] * 5, "rv-graphlock": [1.01] * 5,
        })
        out = self.fx.run()
        self.assertIn("- rv-scopecall: 原因候補", out)
        self.assertIn("- rv-kindlookup: 無関係", out)
        self.assertIn("呼び出し・スコープ型", out)

    def test_placement_effect_when_controls_move(self):
        self.fx.build({
            "aa1": [1.001] * 5, "excl1": [1.01] * 5,
            "rv-scopecall": [1.01] * 5, "rv-kindlookup": [1.01] * 5,
            "rv-graphlock": [1.0] * 5, "excl0-rebuild": [1.01] * 5,
        })
        out = self.fx.run()
        self.assertIn("ビルド・配置の影響あり", out)
        self.assertIn("配置・ビルド起因", out)

    def test_controls_moved_reported_in_case_i(self):
        self.fx.build({"aa1": [1.001] * 5, "excl1": [1.0005] * 5, "excl0-rebuild": [1.02] * 5})
        out = self.fx.run()
        self.assertIn("場合 = (i)", out)
        self.assertIn("ビルド・配置の影響あり（併記）: excl0-rebuild", out)

    def test_controls_moved_reported_in_case_iii(self):
        self.fx.build({"aa1": [1.001] * 5, "excl1": [0.98] * 5, "rv-graphlock": [1.02] * 5})
        out = self.fx.run()
        self.assertIn("場合 = (iii)", out)
        self.assertIn("ビルド・配置の影響あり（併記）: rv-graphlock", out)

    def test_after_faster_stops(self):
        self.fx.build({"aa1": [1.001] * 5, "excl1": [0.98] * 5})
        self.assertIn("場合 = (iii)", self.fx.run())

    def test_missing_run_is_undetermined(self):
        self.fx.build()
        p = f"{self.fx.d}/results-diag-excl1-{LABEL}-cuda-train.jsonl"
        with open(p) as fh:
            lines = fh.readlines()
        with open(p, "w") as fh:
            fh.writelines(lines[:4])
        with self.assertRaises(summ.Undet):
            self.fx.run()

    def test_missing_digest_is_undetermined(self):
        self.fx.build()
        os.remove(f"{self.fx.d}/digest-diag-rv-graphlock-{LABEL}-cuda-train-r3.log")
        with self.assertRaises(summ.Undet):
            self.fx.run()

    def test_gate_fail_is_undetermined(self):
        self.fx.build()
        p = f"{self.fx.d}/load_gate-diag-{LABEL}-cuda.log"
        with open(p) as fh:
            text = fh.read()
        with open(p, "w") as fh:
            fh.write(text.replace("round3 dev=cuda gate=pass gate_ok=1", "round3 dev=cuda gate=fail gate_ok=0"))
        with self.assertRaises(summ.Undet):
            self.fx.run()

    def test_checksum_mismatch_is_undetermined(self):
        self.fx.build()
        p = f"{self.fx.d}/results-diag-excl1-{LABEL}-cuda-train.jsonl"
        with open(p) as fh:
            rows = [json.loads(ln) for ln in fh]
        rows[2]["checksum"] = 0.6
        wjson(p, rows)
        with self.assertRaises(summ.Undet):
            self.fx.run()

    def test_param_digest_mismatch_is_undetermined(self):
        self.fx.build()
        self.fx.write(f"digest-diag-excl1-{LABEL}-cuda-train-r2.log",
                      f"PARAM_DIGEST task=train mode=chain device=cuda n_tensors=4 n_elems=9 fnv1a64=0x{'2' * 16}\n")
        with self.assertRaises(summ.Undet):
            self.fx.run()

    def test_diag_sha_mismatch_is_undetermined(self):
        self.fx.build()
        with open(self.fx.diag, "a") as fh:
            fh.write("tamper\n")
        with self.assertRaises(summ.Undet):
            self.fx.run()

    def test_aa_binaries_must_match(self):
        self.fx.build()
        p = f"{self.fx.d}/env_info-diag-{LABEL}.txt"
        with open(p) as fh:
            text = fh.read()
        with open(p, "w") as fh:
            fh.write(text.replace(f"bin_sha256_aa1={'9' * 64}", f"bin_sha256_aa1={'8' * 64}"))
        with self.assertRaises(summ.Undet):
            self.fx.run()

    def test_main_exit_codes(self):
        self.fx.build({"aa1": [1.001] * 5, "excl1": [1.0005] * 5})
        argv = ["--diag", self.fx.diag, "--dir", self.fx.d, "--label", LABEL]
        self.assertEqual(summ.main(argv), 0)
        os.remove(f"{self.fx.d}/digest-diag-aa0-{LABEL}-cuda-train-r1.log")
        self.assertEqual(summ.main(argv), 4)
        self.assertEqual(summ.main(["--diag", self.fx.diag, "--dir", self.fx.d, "--label", "../x"]), 2)


class DriftGuardTest(unittest.TestCase):
    """実 DIAG-r4.txt の腕と回転表がツール定数と一致すること。"""

    def test_real_diag_matches_tool_constants(self):
        with open(REAL_DIAG, encoding="utf-8") as fh:
            text = fh.read()
        base = list(summ.ARMS)
        for r in range(1, 6):
            want = " ".join(base[((r - 1) + i) % 7] for i in range(7))
            self.assertIn(f"  r{r}: {want}\n", text)
        for rv in summ.RV_ARMS:
            self.assertTrue(re.search(rf"^  {rv}_sha256: [0-9a-f]{{64}}$", text, re.M), rv)


if __name__ == "__main__":
    unittest.main()
