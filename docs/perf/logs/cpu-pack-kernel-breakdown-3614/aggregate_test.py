"""aggregate.py（イシュー #3614 の fail-closed 集計）の回帰テスト。合成 fixture のみを使う。

ci.yml の既存 python unittest ステップ群と同一方針（標準ライブラリのみ・cargo 非依存）。
実行: python3 -m unittest docs/perf/logs/cpu-pack-kernel-breakdown-3614/aggregate_test.py
"""

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location("aggregate_3614", os.path.join(_HERE, "aggregate.py"))
agg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(agg)

TICK = 41.0


def make_line(arm, mode, shape, *, threads, sa, tj=100_000.0, calls=500, jobs=10, hash_="0x" + "ab" * 8,
              interval_ns=1000.0, wall=100_000):
    """tJob=tj・A pack の割合=sa の 1 行。B pack・kernel は残りを 1:3 で割る。"""
    trans, m, n, k = agg.SHAPE_DEFS[shape]
    base = (
        f"arm={arm} mode={mode} shape={shape} trans={trans} m={m} n={n} k={k} rayon_threads={threads} "
        f"calls={calls} wall_median_ns={wall} wall_mean_ns={wall}.0 bits_hash={hash_}"
    )
    if arm == "base":
        return base
    ta = tj * sa
    rest = tj * 0.9 - ta
    tb = rest * 0.25
    tk = rest * 0.75
    ints = int(max(1, (ta * calls) // interval_ns))
    return base + (
        f" a_pack_ns={int(ta * calls)} b_pack_ns={int(tb * calls)} kernel_ns={int(tk * calls)} "
        f"job_ns={int(tj * calls)} jobs={jobs} a_intervals={ints} b_intervals={ints} k_intervals={ints}"
    )


def write_machine(root, sa_by_round, *, mutate=None, tj_by_round=None):
    """5 round × 4 プロセスのログを書く。`mutate(r, arm, mode, shape, kwargs)` で個別に改変する。"""
    for r in agg.ROUNDS:
        for arm, mode in agg.PROCS:
            threads = 1 if mode == "t1" else 16
            lines = [f"arm={arm} mode={mode} timer_tick_ns={TICK} timer_pair_ns=57.3"]
            for s in agg.ALL_SHAPES:
                kw = dict(threads=threads, sa=sa_by_round[r - 1])
                if tj_by_round:
                    kw["tj"] = tj_by_round[r - 1]
                if mutate:
                    mutate(r, arm, mode, s, kw)
                lines.append(make_line(arm, mode, s, **kw))
            with open(os.path.join(root, f"round{r}-{arm}-{mode}.log"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")


class AggregateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def verdict(self):
        return agg.analyze(agg.load_machine(self.root), "native")["verdict"]

    def test_pack_main(self):
        write_machine(self.root, [0.6] * 5)
        self.assertEqual(self.verdict(), "PACK_MAIN")

    def test_not_pack_main(self):
        write_machine(self.root, [0.2] * 5)
        self.assertEqual(self.verdict(), "NOT_PACK_MAIN")

    def test_indeterminate(self):
        # 中央値は 0.5 未満だが 2 round が 0.5 以上 → どちらでもない
        write_machine(self.root, [0.6, 0.6, 0.4, 0.4, 0.4])
        self.assertEqual(self.verdict(), "INDETERMINATE")

    def test_boundary_exactly_half_and_four_of_five(self):
        self.assertEqual(agg.verdict_from([0.5] * 5), "PACK_MAIN")
        self.assertEqual(agg.verdict_from([0.6, 0.6, 0.6, 0.6, 0.1]), "PACK_MAIN")
        # 3/5 では PACK_MAIN にならない（中央値は 0.5 以上でも INDETERMINATE）
        self.assertEqual(agg.verdict_from([0.6, 0.6, 0.6, 0.1, 0.1]), "INDETERMINATE")
        self.assertEqual(agg.verdict_from([0.49, 0.49, 0.49, 0.49, 0.49]), "NOT_PACK_MAIN")
        # 1 round だけ 0.5 以上で中央値 < 0.5 は NOT_PACK_MAIN、2 round は INDETERMINATE
        self.assertEqual(agg.verdict_from([0.5, 0.1, 0.1, 0.1, 0.1]), "NOT_PACK_MAIN")
        self.assertEqual(agg.verdict_from([0.5, 0.5, 0.1, 0.1, 0.1]), "INDETERMINATE")

    def test_report_renders(self):
        write_machine(self.root, [0.6] * 5)
        text = "\n".join(agg.render_machine("m", agg.load_machine(self.root)))
        self.assertIn("**判定: PACK_MAIN**", text)
        self.assertIn("| 5 |", text)

    def test_hash_mismatch_base_vs_timed(self):
        def mut(r, arm, mode, s, kw):
            if r == 3 and arm == "timed" and mode == "native" and s == "l2_dx":
                kw["hash_"] = "0x" + "cd" * 8

        write_machine(self.root, [0.6] * 5, mutate=mut)
        with self.assertRaises(agg.AggregateError) as cm:
            agg.load_machine(self.root)
        self.assertIn("bits_hash", str(cm.exception))

    def test_hash_mismatch_across_rounds(self):
        def mut(r, arm, mode, s, kw):
            if r == 2 and mode == "t1" and s == "l1_fwd":
                kw["hash_"] = "0x" + "cd" * 8

        write_machine(self.root, [0.6] * 5, mutate=mut)
        with self.assertRaises(agg.AggregateError):
            agg.load_machine(self.root)

    def test_missing_round(self):
        write_machine(self.root, [0.6] * 5)
        os.remove(os.path.join(self.root, "round4-timed-t1.log"))
        with self.assertRaises(agg.AggregateError) as cm:
            agg.load_machine(self.root)
        self.assertIn("欠けている", str(cm.exception))

    def test_unavailable_marker(self):
        write_machine(self.root, [0.6] * 5)
        os.remove(os.path.join(self.root, "round1-base-native.log"))
        with open(os.path.join(self.root, "round1-base-native.log.unavailable"), "w", encoding="utf-8") as fh:
            fh.write("x")
        with self.assertRaises(agg.AggregateError) as cm:
            agg.load_machine(self.root)
        self.assertIn("unavailable", str(cm.exception))

    def test_parse_failures(self):
        write_machine(self.root, [0.6] * 5)
        path = os.path.join(self.root, "round1-timed-native.log")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("garbage-without-equals\n")
        with self.assertRaises(agg.AggregateError):
            agg.load_machine(self.root)

    def test_missing_key_and_non_numeric(self):
        with self.assertRaises(agg.AggregateError):
            agg.parse_line("a=1 a=2", "x")
        write_machine(self.root, [0.6] * 5)
        path = os.path.join(self.root, "round1-timed-native.log")
        text = open(path, encoding="utf-8").read().replace("jobs=10", "jobs=ten", 1)
        open(path, "w", encoding="utf-8").write(text)
        with self.assertRaises(agg.AggregateError):
            agg.load_machine(self.root)

    def test_non_finite_and_garbage_floats(self):
        # 校正値の nan・非正値、wall_mean_ns の garbage・nan は fail-closed（RULE §8）。
        cases = (
            ("round1-timed-native.log", f"timer_tick_ns={TICK}", "timer_tick_ns=nan"),
            ("round1-timed-native.log", f"timer_tick_ns={TICK}", "timer_tick_ns=0"),
            ("round1-timed-native.log", "timer_pair_ns=57.3", "timer_pair_ns=inf"),
            ("round1-base-native.log", "wall_mean_ns=100000.0", "wall_mean_ns=garbage"),
            ("round1-base-native.log", "wall_mean_ns=100000.0", "wall_mean_ns=nan"),
        )
        for name, old, new in cases:
            with self.subTest(new=new):
                write_machine(self.root, [0.6] * 5)
                path = os.path.join(self.root, name)
                text = open(path, encoding="utf-8").read()
                self.assertIn(old, text)
                open(path, "w", encoding="utf-8").write(text.replace(old, new, 1))
                with self.assertRaises(agg.AggregateError):
                    agg.load_machine(self.root)

    def test_rayon_threads_mismatch(self):
        def mut(r, arm, mode, s, kw):
            if mode == "t1" and r == 1 and arm == "base" and s == "l1_dw":
                kw["threads"] = 4

        write_machine(self.root, [0.6] * 5, mutate=mut)
        with self.assertRaises(agg.AggregateError) as cm:
            agg.load_machine(self.root)
        self.assertIn("rayon_threads", str(cm.exception))

    def test_native_arm_with_one_thread(self):
        def mut(r, arm, mode, s, kw):
            if mode == "native":
                kw["threads"] = 1

        write_machine(self.root, [0.6] * 5, mutate=mut)
        with self.assertRaises(agg.AggregateError):
            agg.load_machine(self.root)

    def test_shape_trans_dims_mismatch(self):
        # l1_dx の行が trans=nn・別寸法でも受理してはならない（RULE §2 の対応表と照合）
        write_machine(self.root, [0.6] * 5)
        path = os.path.join(self.root, "round1-base-native.log")
        with open(path, encoding="utf-8") as fh:
            txt = fh.read()
        bad = txt.replace("shape=l1_dx trans=nt m=64 n=784 k=256", "shape=l1_dx trans=nn m=64 n=256 k=784")
        self.assertNotEqual(txt, bad)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(bad)
        with self.assertRaises(agg.AggregateError) as cm:
            agg.load_machine(self.root)
        self.assertIn("l1_dx", str(cm.exception))

    def test_jobs_zero(self):
        def mut(r, arm, mode, s, kw):
            if arm == "timed" and r == 5 and s == "l1_dx":
                kw["jobs"] = 0

        write_machine(self.root, [0.6] * 5, mutate=mut)
        with self.assertRaises(agg.AggregateError) as cm:
            agg.load_machine(self.root)
        self.assertIn("jobs=0", str(cm.exception))

    def test_resolution_note(self):
        # A pack の平均区間 = interval_ns。10×tick(410 ns) 未満の形状だけに a_pack の注記が付く。
        def mut(r, arm, mode, s, kw):
            kw["interval_ns"] = 100.0 if s == "l2_dx" else 5000.0

        write_machine(self.root, [0.6] * 5, mutate=mut)
        a = agg.analyze(agg.load_machine(self.root), "native")
        self.assertTrue(any(n.startswith("a_pack") for n in a["res"]["l2_dx"]["resolution_notes"]))
        self.assertFalse(any(n.startswith("a_pack") for n in a["res"]["l1_fwd"]["resolution_notes"]))

    def test_two_level_split(self):
        self.assertIsNone(agg.detect_split([100.0, 110.0, 120.0, 130.0, 140.0]))
        lo, hi = agg.detect_split([100.0, 105.0, 300.0, 310.0, 320.0])
        self.assertEqual((lo, hi), (102.5, 310.0))
        # 全体では 1.5 倍以上でも隣接の飛びが 1.5 未満（連続的）なら分裂扱いにしない
        self.assertIsNone(agg.detect_split([100.0, 120.0, 145.0, 175.0, 210.0]))

    def test_two_level_note_in_report(self):
        write_machine(self.root, [0.6] * 5, tj_by_round=[100_000.0, 100_000.0, 300_000.0, 300_000.0, 300_000.0])
        text = "\n".join(agg.render_machine("m", agg.load_machine(self.root)))
        self.assertIn("2 水準", text)

    def test_main_exit_codes(self):
        write_machine(self.root, [0.6] * 5)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(agg.main(["aggregate.py", self.root]), 0)
        os.remove(os.path.join(self.root, "round2-base-native.log"))
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(agg.main(["aggregate.py", self.root]), 1)


if __name__ == "__main__":
    unittest.main()
