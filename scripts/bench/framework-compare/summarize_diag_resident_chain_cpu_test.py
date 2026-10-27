"""summarize_diag_resident_chain_cpu.py の単体テスト（イシュー #3611）。合成 fixture のみ（実機不要）。

役割: DIAG.txt の [分類規則]（A/A ノイズ幅 N・D1・D2・P・分類 (i)〜(v)・境界値・round 内比の 4/5 条件）と、
欠損・不一致を寛容に読まない fail-closed 動作が事前登録どおりであることを担保する。
CI の deps-forbidden ジョブから `python3 -m unittest` で実行する。
ドリフトガード: 実 DIAG.txt から集計ツールが読む固定値（SHA・パッチ sha256）が取り出せ、腕の回転表がツール定数と一致することを
検査する（負荷ゲート定数は計測条件のためガード対象外。DIAG [ドリフトガードの除外]）。
"""
import hashlib
import importlib.util
import json
import os
import re
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("summ", os.path.join(HERE, "summarize_diag_resident_chain_cpu.py"))
summ = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summ)

DIAG_DIR = os.path.join(HERE, "..", "..", "..", "docs", "perf", "logs", "resident-chain-cpu-train-diag-3611")
REAL_DIAGS = [os.path.join(DIAG_DIR, n) for n in ("DIAG.txt", "DIAG-r2.txt")]
lspec = importlib.util.spec_from_file_location("lsym", os.path.join(HERE, "diag_layout_symbols.py"))
lsym = importlib.util.module_from_spec(lspec)
lspec.loader.exec_module(lsym)
LABEL = "t"
PSHA = "a" * 64
LSHA = "b" * 64
B1 = "d" * 40
B0 = "e" * 40
DIAG = (f"DIAG\n  patch_sha256: {PSHA}\n  layout_shift_sha256: {LSHA}\n- b1 = {B1}\n- oldcode = {B0}\n")
BASE = 1e-4
FLAT = [1.0] * 5


def wjson(path, rows):
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class Fixture:
    def __init__(self, root, profile="gb10"):
        self.d = root
        self.profile = profile
        self.diag = os.path.join(root, "DIAG.txt")
        with open(self.diag, "w") as fh:
            fh.write(DIAG)

    def write(self, name, text):
        with open(os.path.join(self.d, name), "w") as fh:
            fh.write(text)

    def build(self, rho=None):
        """rho: 腕 -> 5 round の aa0 比（既定は全腕 1.0）。aa0 は常に BASE。"""
        rho = rho or {}
        diag_sha = hashlib.sha256(DIAG.encode()).hexdigest()
        gate = 1.0 if self.profile == "gb10" else 3.0
        for arm in summ.ARMS:
            rs = rho.get(arm, FLAT)
            mode = summ.ARM_MODE[arm]
            rows = [{"task": "train", "device": "cpu", "size": 64, "framework": "fandhe-ai", "mode": mode,
                     "checksum": 0.5, "median_s": BASE * p} for p in rs]
            wjson(f"{self.d}/results-diag-{arm}-{LABEL}-cpu-train.jsonl", rows)
            for r in range(1, 6):
                self.write(f"digest-diag-{arm}-{LABEL}-cpu-train-r{r}.log",
                           f"PARAM_DIGEST task=train mode={mode} device=cpu n_tensors=4 n_elems=9 fnv1a64=0x{'1' * 16}\n")
        env = [f"profile={self.profile}", f"diag_sha256={diag_sha}", f"patch_sha256={PSHA}",
               f"layout_shift_sha256={LSHA}", f"rev_stamp_head_b1={B1}", f"rev_stamp_head_b0={B0}",
               "precheck=assert", "rounds=5", "tf32=off", "layout_symbol_diff=2"]
        env += (["gpu=NVIDIA GB10", "uname=Linux 6 aarch64"] if self.profile == "gb10" else ["cpu_brand=Apple M4", "uname=Darwin 25 arm64"])
        grp = {"B1": "1" * 64, "B2": "2" * 64, "B0": "3" * 64}
        for arm in summ.ARMS:
            env.append(f"bin_sha256_{arm}={grp[summ.BIN_GROUP[arm]]}")
        self.write(f"env_info-diag-{LABEL}.txt", "\n".join(env) + "\n")
        self.write(f"load_gate-diag-{LABEL}-cpu.log",
                   f"series=official threshold={gate} rule_threshold={gate} precheck=assert\n" +
                   "".join(f"round{i} dev=cpu gate=pass gate_ok=1 threshold={gate} waited_s=0 load1=0.1\n" for i in range(1, 6)))

    def run(self):
        return summ.summarize(self.diag, self.d, LABEL, self.profile)


def case_of(text):
    return re.search(r"分類 = \((\w+)\)", text).group(1)


def shifted(**kw):
    """aa1 に ±noise、他腕に指定倍率を与えた rho を返す。"""
    return kw


class SummarizeTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def classify(self, rho):
        self.fx.build(rho)
        return case_of(self.fx.run())

    def test_case_i_flat(self):
        self.assertEqual(self.classify({}), "i")

    def test_case_i_when_noise_exceeds_difference(self):
        # N = 0.10。chain が 3% 遅くても N より小さいので (i)（規則上の帰結）。
        rho = {"aa1": [1.10, 1.0, 1.0, 1.0, 1.0], "ch1": [1.03] * 5, "rb1": [1.03] * 5}
        self.assertEqual(self.classify(rho), "i")

    def test_case_ii_regress_both_trees(self):
        rho = {"aa1": [1.01, 1.0, 1.0, 1.0, 1.0], "ch1": [1.05] * 5, "rb1": [1.05] * 5}
        self.assertEqual(self.classify(rho), "ii")

    def test_case_ii_needs_four_of_five_rounds(self):
        # 代表値は N を超えるが、round 内比 > 1 が 3/5 round のみ → (ii) でなく (v)。
        rho = {"aa1": [1.01, 1.0, 1.0, 1.0, 1.0], "ch1": [1.2, 1.2, 1.2, 0.99, 0.99], "rb1": [1.2, 1.2, 1.2, 0.99, 0.99]}
        self.assertEqual(self.classify(rho), "v")

    def test_case_ii_four_rounds_ok(self):
        rho = {"aa1": [1.01, 1.0, 1.0, 1.0, 1.0], "ch1": [1.2, 1.2, 1.2, 1.2, 0.5], "rb1": [1.2, 1.2, 1.2, 1.2, 0.5]}
        self.assertEqual(self.classify(rho), "ii")

    def test_case_iii_layout_only_one_tree(self):
        # B1 では差が出ず（D1 = 0）、B2 の reuse が配置で動き（P 大）chain との差が出る。
        rho = {"aa1": [1.01, 1.0, 1.0, 1.0, 1.0], "rb0": [1.10] * 5, "rb1": [1.20] * 5}
        self.assertEqual(self.classify(rho), "iii")

    def test_case_iii_opposite_signs(self):
        rho = {"aa1": [1.01, 1.0, 1.0, 1.0, 1.0], "ch1": [1.06] * 5, "rb0": [1.10] * 5, "rb1": [1.0] * 5}
        self.assertEqual(self.classify(rho), "iii")

    def test_one_exceeds_but_layout_insensitive_is_v(self):
        rho = {"aa1": [1.01, 1.0, 1.0, 1.0, 1.0], "ch1": [1.06] * 5}
        self.assertEqual(self.classify(rho), "v")

    def test_case_iv_chain_faster(self):
        rho = {"aa1": [1.01, 1.0, 1.0, 1.0, 1.0], "ch1": [0.9] * 5, "rb1": [0.9] * 5}
        self.assertEqual(self.classify(rho), "iv")

    def test_boundary_equal_to_noise_is_i(self):
        # 厳密に |D1| = N（aa1 の rho も 5 run の中央値が 1.0 なので N は round 内比の最大偏差 0.25）。
        rho = {"aa1": [1.25, 1.0, 1.0, 1.0, 1.0], "ch1": [1.25] * 5, "rb1": [1.25] * 5}
        res = summ.classify({a: [BASE * x for x in rho.get(a, FLAT)] for a in summ.ARMS})
        self.assertEqual(res["N"], 0.25)
        self.assertEqual(res["D1"], 0.25)
        self.assertEqual(res["case"], "i")

    def test_b0_ratio_is_reported_not_classified(self):
        rho = {"o1": [1.5] * 5}
        self.fx.build(rho)
        out = self.fx.run()
        self.assertEqual(case_of(out), "i")
        self.assertIn("o1/o0 - 1 = +0.5000", out)

    def test_table_has_run_numbers(self):
        self.fx.build({})
        self.assertIn("r1..r5", self.fx.run())

    def test_macmini_profile(self):
        fx = Fixture(self._td.name, "macmini")
        fx.build({})
        self.assertEqual(case_of(fx.run()), "i")


class RejectTest(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.fx = Fixture(self._td.name)
        self.fx.build({})

    def tearDown(self):
        self._td.cleanup()

    def undet(self):
        with self.assertRaises(summ.Undet):
            self.fx.run()

    def test_missing_run_rejected(self):
        p = f"{self.fx.d}/results-diag-ch1-{LABEL}-cpu-train.jsonl"
        lines = open(p).read().splitlines()[:4]
        open(p, "w").write("\n".join(lines) + "\n")
        self.undet()

    def test_extra_run_rejected(self):
        p = f"{self.fx.d}/results-diag-o0-{LABEL}-cpu-train.jsonl"
        lines = open(p).read().splitlines()
        open(p, "w").write("\n".join(lines + lines[:1]) + "\n")
        self.undet()

    def test_checksum_mismatch_rejected(self):
        p = f"{self.fx.d}/results-diag-rb1-{LABEL}-cpu-train.jsonl"
        rows = [json.loads(x) for x in open(p).read().splitlines()]
        rows[2]["checksum"] = 0.6
        wjson(p, rows)
        self.undet()

    def test_digest_mismatch_rejected(self):
        self.fx.write(f"digest-diag-o1-{LABEL}-cpu-train-r3.log",
                      f"PARAM_DIGEST task=train mode=chain device=cpu n_tensors=4 n_elems=9 fnv1a64=0x{'2' * 16}\n")
        self.undet()

    def test_digest_missing_rejected(self):
        self.fx.write(f"digest-diag-aa1-{LABEL}-cpu-train-r1.log", "no digest\n")
        self.undet()

    def test_wrong_mode_rejected(self):
        p = f"{self.fx.d}/results-diag-ch1-{LABEL}-cpu-train.jsonl"
        rows = [json.loads(x) for x in open(p).read().splitlines()]
        rows[0]["mode"] = "reuse"
        wjson(p, rows)
        self.undet()

    def test_gate_failure_rejected(self):
        p = f"{self.fx.d}/load_gate-diag-{LABEL}-cpu.log"
        text = open(p).read().replace("round3 dev=cpu gate=pass gate_ok=1", "round3 dev=cpu gate=fail gate_ok=0")
        open(p, "w").write(text)
        self.undet()

    def test_gate_threshold_mismatch_rejected(self):
        p = f"{self.fx.d}/load_gate-diag-{LABEL}-cpu.log"
        text = open(p).read().replace("threshold=1.0 rule_threshold=1.0", "threshold=8.0 rule_threshold=8.0", 1)
        open(p, "w").write(text)
        self.undet()

    def test_diag_sha_mismatch_rejected(self):
        with open(self.fx.diag, "a") as fh:
            fh.write("tamper\n")
        self.undet()

    def test_patch_sha_mismatch_rejected(self):
        p = f"{self.fx.d}/env_info-diag-{LABEL}.txt"
        text = open(p).read().replace(f"layout_shift_sha256={LSHA}", f"layout_shift_sha256={'c' * 64}")
        open(p, "w").write(text)
        self.undet()

    def test_same_binary_for_layout_arms_rejected(self):
        p = f"{self.fx.d}/env_info-diag-{LABEL}.txt"
        text = open(p).read().replace("bin_sha256_rb0=" + "2" * 64, "bin_sha256_rb0=" + "1" * 64)
        open(p, "w").write(text)
        self.undet()

    def test_no_layout_symbol_diff_rejected(self):
        p = f"{self.fx.d}/env_info-diag-{LABEL}.txt"
        text = open(p).read().replace("layout_symbol_diff=2", "layout_symbol_diff=0")
        open(p, "w").write(text)
        self.undet()

    def test_macmini_wrong_brand_rejected(self):
        fx = Fixture(self._td.name, "macmini")
        fx.build({})
        p = f"{fx.d}/env_info-diag-{LABEL}.txt"
        text = open(p).read().replace("Apple M4", "Apple M4 Max")
        open(p, "w").write(text)
        with self.assertRaises(summ.Undet):
            fx.run()

    def test_nonfinite_median_rejected(self):
        p = f"{self.fx.d}/results-diag-aa0-{LABEL}-cpu-train.jsonl"
        text = open(p).read().replace('"median_s": 0.0001', '"median_s": 0', 1)
        open(p, "w").write(text)
        self.undet()

    def test_bad_diag_path(self):
        with self.assertRaises(summ.Bad):
            summ.summarize(os.path.join(self.fx.d, "nope.txt"), self.fx.d, LABEL, "gb10")


class DriftGuardTest(unittest.TestCase):
    """実 DIAG.txt（r1）・DIAG-r2.txt と集計ツール定数の整合（負荷ゲート定数は対象外）。"""

    def each(self):
        for path in REAL_DIAGS:
            with open(path, encoding="utf-8") as fh:
                yield path, fh.read()

    def test_fixed_values_readable(self):
        for _, text in self.each():
            for pat, what in ((r"^  patch_sha256: ([0-9a-f]{64})$", "patch"), (r"^  layout_shift_sha256: ([0-9a-f]{64})$", "layout"),
                              (r"^- b1 = ([0-9a-f]{40})$", "b1"), (r"^- oldcode = ([0-9a-f]{40})$", "oldcode")):
                summ.diag_field(text, pat, what)

    def test_patch_sha_matches_files(self):
        for path, text in self.each():
            base = os.path.dirname(path)
            for rel, pat in (("../resident-chain-ab-3257/bench-resident-chain.patch", r"^  patch_sha256: ([0-9a-f]{64})$"),
                             ("layout-shift.patch", r"^  layout_shift_sha256: ([0-9a-f]{64})$")):
                with open(os.path.join(base, rel), "rb") as fh:
                    self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), summ.diag_field(text, pat, rel))

    def test_rotation_matches_tool_arms(self):
        for _, text in self.each():
            rows = re.findall(r"^  r([1-5]): (.+)$", text, re.M)
            self.assertEqual(len(rows), 5)
            n = len(summ.ARMS)
            for r, order in rows:
                start = (int(r) - 1) % n
                want = [summ.ARMS[(start + i) % n] for i in range(n)]
                self.assertEqual(order.split(), want)

    def test_arm_list_in_diag(self):
        for _, text in self.each():
            for a in summ.ARMS:
                self.assertIn(f"- {a}: ", text)

    def test_r2_runner_and_helper_sha_match_files(self):
        _, text = list(self.each())[1]
        for name, pat in (("run_diag_resident_chain_cpu.sh", r"^  runner_sha256: ([0-9a-f]{64})$"),
                          ("diag_layout_symbols.py", r"^  layout_helper_sha256: ([0-9a-f]{64})$")):
            with open(os.path.join(HERE, name), "rb") as fh:
                self.assertEqual(hashlib.sha256(fh.read()).hexdigest(), summ.diag_field(text, pat, name))


def nm_text(path, sep=" "):
    """コミット済みの r1 ダンプ（`name addr`）を nm 生出力の形（`addr T name`）へ並べ替える。"""
    out = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh.read().splitlines():
            name, addr = ln.split()
            out.append(f"{addr} T {name}")
    return "\n".join(out) + "\n"


class LayoutSymbolsTest(unittest.TestCase):
    """r1 停止の回帰（v0 mangling）と fail-closed 条件。"""

    def r1(self, node):
        d = os.path.join(DIAG_DIR, node)
        a = lsym.parse_nm(nm_text(os.path.join(d, f"nm-diag-b1-3611-{node}.txt")), "B1")
        b = lsym.parse_nm(nm_text(os.path.join(d, f"nm-diag-b2-3611-{node}.txt")), "B2")
        return a, b

    def test_r1_dumps_macmini_diff_2(self):
        diff, keys, _ = lsym.compare(*self.r1("macmini"))
        self.assertEqual((diff, keys), (2, 2))

    def test_r1_dumps_gb10_diff_2(self):
        diff, keys, _ = lsym.compare(*self.r1("gb10"))
        self.assertEqual((diff, keys), (2, 2))

    def sym(self, addr, name):
        return f"{addr} T {name}\n"

    def test_legacy_mangling(self):
        def t(h, a):
            return (self.sym(a, f"_ZN9fandhe_ai6compat10Sequential19train_step_resident17h{h * 16}E")
                    + self.sym(a, f"_ZN9fandhe_ai6compat10Sequential16forward_resident17h{h * 16}E"))
        diff, keys, _ = lsym.compare(lsym.parse_nm(t("a", "10"), "B1"), lsym.parse_nm(t("b", "20"), "B2"))
        self.assertEqual((diff, keys), (2, 2))

    def test_same_address_is_diff_0(self):
        a, _ = self.r1("gb10")
        diff, _, _ = lsym.compare(a, a)
        self.assertEqual(diff, 0)

    def test_duplicate_key_rejected(self):
        t = (self.sym("1", "_RNvCsabc_9fandhe_ai19train_step_resident")
             + self.sym("2", "_RNvCsdef_9fandhe_ai19train_step_resident"))
        with self.assertRaises(lsym.Bad):
            lsym.parse_nm(t, "B1")

    def test_missing_required_symbol_rejected(self):
        a, b = self.r1("gb10")
        b = {k: v for k, v in b.items() if not k.endswith("16forward_resident")}
        a = {k: v for k, v in a.items() if not k.endswith("16forward_resident")}
        with self.assertRaises(lsym.Bad):
            lsym.compare(a, b)

    def test_no_common_key_rejected(self):
        a, b = self.r1("gb10")
        b = {k + "x": v for k, v in b.items()}
        with self.assertRaises(lsym.Bad):
            lsym.compare(a, b)

    def test_non_hex_address_rejected(self):
        with self.assertRaises(lsym.Bad):
            lsym.parse_nm("zz T _RNvCsabc_9fandhe_ai19train_step_resident\n", "B1")

    def test_cs_inside_identifier_not_normalized(self):
        # Cs…_ の直後が数字・u でない位置（識別子の途中）は潰さない。
        name = "_RNvCsabc_9fandhe_ai6Cslike_x19train_step_resident"
        self.assertEqual(lsym.normalize(name), "_RNvC9fandhe_ai6Cslike_x19train_step_resident")

    def test_one_underscore_extra_prefix_unified(self):
        self.assertEqual(lsym.normalize("__RNvCsabc_9x"), lsym.normalize("_RNvCsxyz_9x"))


class RunnerShaFieldTest(unittest.TestCase):
    """DIAG に runner／helper の sha 行があるときだけ env_info との一致を必須にする。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._td.cleanup()

    def build(self, with_diag_sha, with_env_sha, env_sha="9" * 64):
        extra = f"  runner_sha256: {'7' * 64}\n  layout_helper_sha256: {'8' * 64}\n" if with_diag_sha else ""
        global DIAG
        saved = DIAG
        DIAG = saved + extra
        try:
            fx = Fixture(self._td.name)
            fx.build({})
        finally:
            DIAG = saved
        if with_env_sha:
            with open(f"{fx.d}/env_info-diag-{LABEL}.txt", "a") as fh:
                fh.write(f"runner_sha256={env_sha}\nlayout_helper_sha256={'8' * 64}\n")
        return fx

    def test_without_diag_fields_passes(self):
        self.assertEqual(case_of(self.build(False, False).run()), "i")

    def test_env_mismatch_rejected(self):
        with self.assertRaises(summ.Undet):
            self.build(True, True).run()

    def test_env_missing_rejected(self):
        with self.assertRaises(summ.Undet):
            self.build(True, False).run()

    def test_env_match_passes(self):
        self.assertEqual(case_of(self.build(True, True, "7" * 64).run()), "i")


if __name__ == "__main__":
    unittest.main()
