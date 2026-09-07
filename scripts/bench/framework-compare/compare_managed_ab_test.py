#!/usr/bin/env python3
"""`compare_managed_ab.py` の単体テスト（イシュー #1353）。

`compare_ab_test.py` と同じ方式（ファイルパス指定 import・tempfile への
合成 JSONL 書き出し）。CI（`ci.yml` の `deps-forbidden` ジョブ）は
`python3 -m unittest scripts/bench/framework-compare/compare_managed_ab_test.py`
で本ファイルを実行する。

検証観点:
- 正常系: off/on 各 5 件から中央値・比率・checksum 一致列を算出する。
- 件数過不足（5 件未満／超過）は判定不能。
- checksum が複合判定を外れる場合は判定不能。checksum が完全一致でない
  （複合判定内だが厳密には異なる）場合は「複合判定 ok」として区別する。
- 不正な JSON 行・不正な `managed` フィールド型は理由付きで警告しスキップ
  する（例外を送出しない）。
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
    "compare_managed_ab", os.path.join(HERE, "compare_managed_ab.py")
)
compare_managed_ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_managed_ab)


def _rec(managed, median_s, checksum=1.23456, task="gemm", device="cuda", size=1024, mode="reuse"):
    r = {
        "framework": "fandhe-ai",
        "version": "0.1.0",
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
    if managed:
        r["managed"] = True
    return r


def _write_jsonl(rows):
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    )
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


class LoadRowsTest(unittest.TestCase):
    def test_valid_rows_are_loaded(self):
        rows = [_rec(False, 0.001), _rec(True, 0.0009)]
        path = _write_jsonl(rows)
        try:
            loaded, warnings = compare_managed_ab.load_rows(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(warnings, [])
        finally:
            os.unlink(path)

    def test_invalid_json_line_is_skipped_with_warning(self):
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ).name
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(_rec(False, 0.001)) + "\n")
            f.write("{not valid json\n")
        try:
            loaded, warnings = compare_managed_ab.load_rows(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(len(warnings), 1)
            self.assertIn("invalid JSON", warnings[0])
        finally:
            os.unlink(path)

    def test_non_object_line_is_skipped_with_warning(self):
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ).name
        with open(path, "w", encoding="utf-8") as f:
            f.write("[1, 2, 3]\n")
        try:
            loaded, warnings = compare_managed_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("JSON object ではない", warnings[0])
        finally:
            os.unlink(path)

    def test_invalid_managed_type_is_skipped_with_warning(self):
        r = _rec(False, 0.001)
        r["managed"] = "true"
        path = _write_jsonl([r])
        try:
            loaded, warnings = compare_managed_ab.load_rows(path)
            self.assertEqual(loaded, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("managed", warnings[0])
        finally:
            os.unlink(path)


class SplitOffOnTest(unittest.TestCase):
    def test_splits_by_managed_field(self):
        rows = [_rec(False, 0.001), _rec(True, 0.0009)]
        cells = compare_managed_ab.split_off_on(rows)
        key = ("gemm", "cuda", 1024, "reuse", None)
        self.assertEqual(len(cells[key]["off"]), 1)
        self.assertEqual(len(cells[key]["on"]), 1)

    def test_distinguishes_cells_by_size_and_mode(self):
        rows = [
            _rec(False, 0.001, size=1024, mode="reuse"),
            _rec(False, 0.001, size=2048, mode="reuse"),
            _rec(False, 0.001, size=1024, mode="fresh"),
        ]
        cells = compare_managed_ab.split_off_on(rows)
        self.assertEqual(len(cells), 3)


class EvaluateCellTest(unittest.TestCase):
    def test_ok_with_five_each_and_matching_checksum(self):
        off_rows = [_rec(False, 0.001 + i * 1e-6) for i in range(5)]
        on_rows = [_rec(True, 0.0009 + i * 1e-6) for i in range(5)]
        result = compare_managed_ab.evaluate_cell(off_rows, on_rows)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["checksum_exact_match"])
        self.assertTrue(result["checksum_composite_match"])
        self.assertLess(result["ratio"], 1.0)

    def test_undeterminable_when_off_count_is_not_five(self):
        off_rows = [_rec(False, 0.001) for _ in range(4)]
        on_rows = [_rec(True, 0.0009) for _ in range(5)]
        result = compare_managed_ab.evaluate_cell(off_rows, on_rows)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("5 件", result["reason"])

    def test_undeterminable_when_on_count_exceeds_five(self):
        off_rows = [_rec(False, 0.001) for _ in range(5)]
        on_rows = [_rec(True, 0.0009) for _ in range(6)]
        result = compare_managed_ab.evaluate_cell(off_rows, on_rows)
        self.assertEqual(result["status"], "undeterminable")

    def test_undeterminable_when_checksum_diverges_beyond_composite_tolerance(self):
        off_rows = [_rec(False, 0.001, checksum=1.0) for _ in range(5)]
        on_rows = [_rec(True, 0.0009, checksum=999.0) for _ in range(5)]
        result = compare_managed_ab.evaluate_cell(off_rows, on_rows)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("checksum", result["reason"])

    def test_composite_match_but_not_exact(self):
        # 複合判定の絶対誤差許容内（1e-5 未満）だが厳密には異なる値。
        off_rows = [_rec(False, 0.001, checksum=1.0) for _ in range(5)]
        on_rows = [_rec(True, 0.0009, checksum=1.0 + 1e-6) for _ in range(5)]
        result = compare_managed_ab.evaluate_cell(off_rows, on_rows)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["checksum_composite_match"])
        self.assertFalse(result["checksum_exact_match"])

    def test_undeterminable_when_warmup_mismatches_across_off_on(self):
        off_rows = [_rec(False, 0.001) for _ in range(5)]
        on_rows = [_rec(True, 0.0009) for _ in range(5)]
        for r in on_rows:
            r["warmup"] = 99
        result = compare_managed_ab.evaluate_cell(off_rows, on_rows)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("warmup", result["reason"])


class MainTest(unittest.TestCase):
    def test_main_returns_zero_on_all_ok_cells(self):
        rows = [_rec(False, 0.001 + i * 1e-6) for i in range(5)] + [
            _rec(True, 0.0009 + i * 1e-6) for i in range(5)
        ]
        path = _write_jsonl(rows)
        try:
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                code = compare_managed_ab.main(["prog", path])
            self.assertEqual(code, 0)
            self.assertIn("ADOPT 候補", buf_out.getvalue())
        finally:
            os.unlink(path)

    def test_main_returns_nonzero_when_any_cell_is_undeterminable(self):
        rows = [_rec(False, 0.001) for _ in range(3)] + [
            _rec(True, 0.0009) for _ in range(5)
        ]
        path = _write_jsonl(rows)
        try:
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                code = compare_managed_ab.main(["prog", path])
            self.assertEqual(code, 3)
            self.assertIn("判定不能", buf_out.getvalue())
        finally:
            os.unlink(path)

    def test_main_returns_nonzero_on_malformed_input(self):
        path = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ).name
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json\n")
        try:
            buf_out, buf_err = io.StringIO(), io.StringIO()
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                code = compare_managed_ab.main(["prog", path])
            self.assertEqual(code, 3)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
