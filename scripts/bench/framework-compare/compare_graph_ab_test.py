#!/usr/bin/env python3
"""`compare_graph_ab.py` の単体テスト（イシュー #1350）。

`compare_managed_ab_test.py` と同じ方式（ファイルパス指定 import・
tempfile への合成 JSONL 書き出し）。CI（`ci.yml` の `deps-forbidden` ジョブ）
は `python3 -m unittest scripts/bench/framework-compare/compare_graph_ab_test.py`
で本ファイルを実行する。

検証観点:
- 正常系: off/stream-only/on 各 5 件から中央値・比率・checksum 一致列を
  算出する。2 状態のみ計測されたセル（train fresh の off/on 等）も判定
  できる。
- 件数過不足は判定不能。
- checksum が複合判定を外れる場合は判定不能。
- `on` の launch カウンタが 5 run 内で不一致なら判定不能。
- 不正な JSON 行・不正な `graph` フィールド型・不明な `graph` 値は理由
  付きで警告しスキップする（例外を送出しない）。
"""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))

_SPEC = importlib.util.spec_from_file_location(
    "compare_graph_ab", os.path.join(HERE, "compare_graph_ab.py")
)
compare_graph_ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_graph_ab)


def _rec(
    graph,
    median_s,
    checksum=1.23456,
    task="train",
    device="cuda",
    size=64,
    mode="reuse",
    phase=None,
    graph_counters=None,
):
    r = {
        "framework": "fandhe-ai",
        "version": "0.7.0",
        "task": task,
        "device": device,
        "size": size,
        "median_s": median_s,
        "q1_s": median_s,
        "q3_s": median_s,
        "checksum": checksum,
        "warmup": 20,
        "iters": 20,
        "mode": mode,
    }
    if phase is not None:
        r["phase"] = phase
    if graph is not None:
        r["graph"] = graph
    if graph_counters is not None:
        r["graph_captured"] = graph_counters[0]
        r["graph_replayed"] = graph_counters[1]
        r["graph_launches"] = graph_counters[2]
        r["graph_sgd_kernel_launches"] = graph_counters[3]
    return r


def _write_jsonl(rows):
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


def _five_on_rows(median=0.0009, captured=1, replayed=24, launches=25, sgd=1):
    return [
        _rec("on", median, graph_counters=(captured, replayed, launches, sgd))
        for _ in range(5)
    ]


class LoadRowsTest(unittest.TestCase):
    def test_valid_rows_are_loaded_and_state_tagged(self):
        rows = [_rec(None, 0.001), _rec("stream-only", 0.0011), _rec("on", 0.0009)]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_graph_ab.load_rows(path)
            self.assertEqual(len(loaded), 3)
            self.assertEqual(warnings, [])
            states = sorted(r["_graph_state"] for r in loaded)
            self.assertEqual(states, ["off", "on", "stream-only"])
        finally:
            os.unlink(path)

    def test_invalid_json_line_is_skipped_with_warning(self):
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ).name
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_rec(None, 0.001)) + "\n")
            f.write("{not valid json\n")
        try:
            loaded, warnings = compare_graph_ab.load_rows(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(warnings), 1)
            self.assertIn("invalid JSON", warnings[0])
        finally:
            os.unlink(path)

    def test_unknown_graph_value_is_skipped_with_warning(self):
        r = _rec(None, 0.001)
        r["graph"] = "off"  # Record 契約上 emit されない不正値
        path = _write_jsonl([r])
        try:
            loaded, warnings = compare_graph_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("graph", warnings[0])
        finally:
            os.unlink(path)

    def test_invalid_graph_type_is_skipped_with_warning(self):
        r = _rec(None, 0.001)
        r["graph"] = True
        path = _write_jsonl([r])
        try:
            loaded, warnings = compare_graph_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("graph", warnings[0])
        finally:
            os.unlink(path)

    def test_tf32_true_row_is_skipped_with_warning(self):
        r = _rec(None, 0.001)
        r["tf32"] = True
        path = _write_jsonl([r])
        try:
            loaded, warnings = compare_graph_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("tf32", warnings[0])
        finally:
            os.unlink(path)

    def test_managed_true_row_is_skipped_with_warning(self):
        r = _rec(None, 0.001)
        r["managed"] = True
        path = _write_jsonl([r])
        try:
            loaded, warnings = compare_graph_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("managed", warnings[0])
        finally:
            os.unlink(path)

    def test_other_framework_is_skipped_with_warning(self):
        r = _rec(None, 0.001)
        r["framework"] = "candle"
        path = _write_jsonl([r])
        try:
            loaded, warnings = compare_graph_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("framework", warnings[0])
        finally:
            os.unlink(path)


class EvaluateCellTest(unittest.TestCase):
    def test_three_state_ok(self):
        groups = {
            "off": [_rec(None, 0.001) for _ in range(5)],
            "stream-only": [_rec("stream-only", 0.0011) for _ in range(5)],
            "on": _five_on_rows(median=0.0009),
        }
        result = compare_graph_ab.evaluate_cell(groups)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["ratios"]["on_over_off"], 0.9, places=6)
        self.assertTrue(result["checksum_exact_match"])
        self.assertEqual(result["graph_counters"]["graph_captured"], 1)
        self.assertEqual(result["graph_counters"]["graph_replayed"], 24)

    def test_two_state_ok_off_on_only(self):
        # train fresh は stream-only を計測しない対照系列（実装計画 (c)）。
        groups = {
            "off": [_rec(None, 0.001) for _ in range(5)],
            "stream-only": [],
            "on": _five_on_rows(median=0.0009),
        }
        result = compare_graph_ab.evaluate_cell(groups)
        self.assertEqual(result["status"], "ok")
        self.assertIn("on_over_off", result["ratios"])
        self.assertNotIn("stream_only_over_off", result["ratios"])

    def test_single_state_is_undeterminable(self):
        groups = {"off": [_rec(None, 0.001) for _ in range(5)], "stream-only": [], "on": []}
        result = compare_graph_ab.evaluate_cell(groups)
        self.assertEqual(result["status"], "undeterminable")

    def test_wrong_count_is_undeterminable(self):
        groups = {
            "off": [_rec(None, 0.001) for _ in range(4)],
            "stream-only": [],
            "on": _five_on_rows(),
        }
        result = compare_graph_ab.evaluate_cell(groups)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("off", result["reason"])

    def test_checksum_mismatch_beyond_composite_is_undeterminable(self):
        off_rows = [_rec(None, 0.001, checksum=1.0) for _ in range(5)]
        on_rows = _five_on_rows()
        for r in on_rows:
            r["checksum"] = 999.0
        groups = {"off": off_rows, "stream-only": [], "on": on_rows}
        result = compare_graph_ab.evaluate_cell(groups)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("checksum", result["reason"])

    def test_checksum_composite_ok_but_not_exact_is_reported(self):
        off_rows = [_rec(None, 0.001, checksum=1.0) for _ in range(5)]
        on_rows = _five_on_rows()
        for r in on_rows:
            r["checksum"] = 1.0 + 1e-7  # 複合判定内・厳密には不一致
        groups = {"off": off_rows, "stream-only": [], "on": on_rows}
        result = compare_graph_ab.evaluate_cell(groups)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["checksum_composite_match"])
        self.assertFalse(result["checksum_exact_match"])

    def test_inconsistent_launch_counters_is_undeterminable(self):
        off_rows = [_rec(None, 0.001) for _ in range(5)]
        on_rows = _five_on_rows()
        on_rows[2]["graph_captured"] = 2  # 5 run 内で不一致
        groups = {"off": off_rows, "stream-only": [], "on": on_rows}
        result = compare_graph_ab.evaluate_cell(groups)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("graph_captured", result["reason"])

    def test_missing_launch_counters_is_undeterminable(self):
        off_rows = [_rec(None, 0.001) for _ in range(5)]
        on_rows = [_rec("on", 0.0009) for _ in range(5)]  # counters 未設定
        groups = {"off": off_rows, "stream-only": [], "on": on_rows}
        result = compare_graph_ab.evaluate_cell(groups)
        self.assertEqual(result["status"], "undeterminable")


class MainTest(unittest.TestCase):
    def test_main_ok_exit_code_and_table(self):
        rows = (
            [_rec(None, 0.001) for _ in range(5)]
            + [_rec("stream-only", 0.0011) for _ in range(5)]
            + _five_on_rows(median=0.0009)
        )
        path = _write_jsonl(rows)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_graph_ab.main(["compare_graph_ab.py", path])
            self.assertEqual(code, 0)
            self.assertIn("train/cuda/64/reuse", out.getvalue())
        finally:
            os.unlink(path)

    def test_main_undeterminable_exit_code(self):
        rows = [_rec(None, 0.001) for _ in range(3)]  # 件数不足
        path = _write_jsonl(rows)
        try:
            out = io.StringIO()
            err = io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = compare_graph_ab.main(["compare_graph_ab.py", path])
            self.assertEqual(code, 3)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
