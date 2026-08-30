#!/usr/bin/env python3
"""`compare_ab.py` の単体テスト（イシュー #1083）。

`unittest`（stdlib のみ、追加依存なし）で、合成 JSONL を tempfile に書き
`compare_ab.py` の関数を直接呼ぶ。CI（`ci.yml` の `deps-forbidden` ジョブ、
`summarize_test.py` の直後）は
`python3 -m unittest scripts/bench/framework-compare/compare_ab_test.py` で
本ファイルを実行する。

検証観点（実装計画の fail-closed 契約）:
- 正常系: 5 件ずつの fresh/reuse レコードから中央値・比率を算出する。
- レコード不足（5 件未満）は判定不能。
- before/after の framework_version が同一は判定不能（A/B になっていない）。
- checksum（最終 loss）が複合判定を外れる場合は判定不能。
- 不正な JSON 行は理由付きで警告しスキップする（例外を送出しない）。
"""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr

HERE = os.path.dirname(os.path.abspath(__file__))

# summarize_test.py と同じロード方式（sys.path を汚染しないファイルパス指定
# import）。
_SPEC = importlib.util.spec_from_file_location(
    "compare_ab", os.path.join(HERE, "compare_ab.py")
)
compare_ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_ab)


def _rec(version, median_s, mode, checksum=1.23456, task="train", device="cuda"):
    return {
        "framework": "fandhe-ai",
        "version": version,
        "task": task,
        "device": device,
        "size": 64,
        "median_s": median_s,
        "q1_s": median_s,
        "q3_s": median_s,
        "checksum": checksum,
        "warmup": 20,
        "iters": 80,
        "mode": mode,
    }


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            if isinstance(r, str):
                f.write(r + "\n")
            else:
                f.write(json.dumps(r) + "\n")


class CompareModeTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def _paths(self):
        return (
            os.path.join(self.tmpdir.name, "before.jsonl"),
            os.path.join(self.tmpdir.name, "after.jsonl"),
        )

    def test_ok_when_five_records_each_and_checksums_match(self):
        before_path, after_path = self._paths()
        _write_jsonl(
            before_path,
            [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)],
        )
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        before_rows, bw = compare_ab.load_rows(before_path)
        after_rows, aw = compare_ab.load_rows(after_path)
        self.assertEqual(bw, [])
        self.assertEqual(aw, [])
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["before_median_s"], 0.0122, places=6)
        self.assertAlmostEqual(result["after_median_s"], 0.0092, places=6)
        self.assertEqual(result["before_n"], 5)
        self.assertEqual(result["after_n"], 5)
        self.assertLess(result["ratio"], 1.0)

    def test_undeterminable_when_before_has_too_few_records(self):
        before_path, after_path = self._paths()
        _write_jsonl(
            before_path,
            [_rec("0.4.0", 0.012, "fresh") for _ in range(4)],
        )
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("レコード不足", result["reason"])

    def test_undeterminable_when_versions_are_identical(self):
        before_path, after_path = self._paths()
        _write_jsonl(
            before_path,
            [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)],
        )
        _write_jsonl(
            after_path,
            [_rec("0.4.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("同一", result["reason"])

    def test_undeterminable_when_checksums_diverge(self):
        before_path, after_path = self._paths()
        _write_jsonl(
            before_path,
            [
                _rec("0.4.0", 0.012 + i * 0.0001, "fresh", checksum=1.0)
                for i in range(5)
            ],
        )
        _write_jsonl(
            after_path,
            [
                _rec("0.5.0", 0.009 + i * 0.0001, "fresh", checksum=999.0)
                for i in range(5)
            ],
        )
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("checksum", result["reason"])

    def test_undeterminable_when_median_s_is_non_positive(self):
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012, "fresh") for _ in range(5)]
        recs[0]["median_s"] = -1.0
        _write_jsonl(before_path, recs)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("median_s", result["reason"])

    def test_invalid_json_line_is_skipped_with_warning_not_exception(self):
        before_path, after_path = self._paths()
        lines = [json.dumps(_rec("0.4.0", 0.012 + i * 0.0001, "fresh")) for i in range(5)]
        lines.insert(2, "{not valid json")
        _write_jsonl(before_path, lines)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        rows, warnings = compare_ab.load_rows(before_path)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("invalid JSON", warnings[0])

    def test_other_framework_or_device_rows_are_excluded(self):
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        recs.append(_rec("0.4.0", 0.001, "fresh"))
        recs[-1]["framework"] = "candle"
        recs.append(_rec("0.4.0", 0.001, "fresh", device="cpu"))
        _write_jsonl(before_path, recs)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["before_n"], 5)

    def test_rows_with_different_size_are_excluded(self):
        """Review 指摘（#1083）: size の異なる train レコードが同一 JSONL に
        混在しても、`_train_records` の TRAIN_SIZE 絞り込みにより中央値算出
        が size をまたがないことを確認する（defensive gap の回帰防止）。
        """
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        # size=64 以外の行を大量に混入させる。中央値算出に混ざれば
        # before_median_s がこの極端な値へ引っ張られて壊れる。
        other_size_recs = [_rec("0.4.0", 999.0, "fresh") for _ in range(5)]
        for r in other_size_recs:
            r["size"] = 128
        _write_jsonl(before_path, recs + other_size_recs)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["before_n"], 5)
        self.assertLess(result["before_median_s"], 1.0)


    def test_undeterminable_when_warmup_differs(self):
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        for r in recs:
            r["warmup"] = 20
        _write_jsonl(before_path, recs)
        after_recs = [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)]
        for r in after_recs:
            r["warmup"] = 5
        _write_jsonl(after_path, after_recs)
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("warmup", result["reason"])

    def test_undeterminable_when_iters_differs(self):
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        for r in recs:
            r["iters"] = 80
        _write_jsonl(before_path, recs)
        after_recs = [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)]
        for r in after_recs:
            r["iters"] = 10
        _write_jsonl(after_path, after_recs)
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("iters", result["reason"])

    def test_undeterminable_when_warmup_not_a_single_positive_int(self):
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        recs[0]["warmup"] = 0
        _write_jsonl(before_path, recs)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("warmup", result["reason"])

    def test_undeterminable_when_version_is_empty_string(self):
        before_path, after_path = self._paths()
        recs = [_rec("", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        _write_jsonl(before_path, recs)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("framework_version", result["reason"])
        self.assertIn("非空文字列", result["reason"])

    def test_undeterminable_when_version_is_none(self):
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        for r in recs:
            r["version"] = None
        _write_jsonl(before_path, recs)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        before_rows, _ = compare_ab.load_rows(before_path)
        after_rows, _ = compare_ab.load_rows(after_path)
        result = compare_ab.compare_mode(before_rows, after_rows, "fresh")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("framework_version", result["reason"])


class MainCliTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def test_main_returns_zero_for_ok_result(self):
        before_path = os.path.join(self.tmpdir.name, "before.jsonl")
        after_path = os.path.join(self.tmpdir.name, "after.jsonl")
        _write_jsonl(
            before_path,
            [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
            + [_rec("0.4.0", 0.010 + i * 0.0001, "reuse") for i in range(5)],
        )
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)]
            + [_rec("0.5.0", 0.006 + i * 0.0001, "reuse") for i in range(5)],
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = compare_ab.main([before_path, after_path])
        self.assertEqual(rc, 0)

    def test_main_returns_nonzero_when_one_mode_undeterminable(self):
        before_path = os.path.join(self.tmpdir.name, "before.jsonl")
        after_path = os.path.join(self.tmpdir.name, "after.jsonl")
        _write_jsonl(
            before_path,
            [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
            + [_rec("0.4.0", 0.010, "reuse") for _ in range(3)],
        )
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)]
            + [_rec("0.5.0", 0.006 + i * 0.0001, "reuse") for i in range(5)],
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = compare_ab.main([before_path, after_path])
        self.assertEqual(rc, 2)
        self.assertIn("undeterminable: mode=reuse", buf.getvalue())

    def test_main_returns_nonzero_when_input_has_invalid_line_even_with_enough_valid_rows(
        self,
    ):
        """Review 指摘（#1083）: 破損・切り詰められた外部 JSONL でも、各
        mode 5 件以上の正常行が残っていれば `main` が exit 0・性能値を返す
        fail-open の回帰防止。不正行が 1 行でもあれば、正常行が十分でも
        判定不能として非 0 終了する。
        """
        before_path = os.path.join(self.tmpdir.name, "before.jsonl")
        after_path = os.path.join(self.tmpdir.name, "after.jsonl")
        lines = [
            json.dumps(_rec("0.4.0", 0.012 + i * 0.0001, "fresh")) for i in range(5)
        ]
        lines += [
            json.dumps(_rec("0.4.0", 0.010 + i * 0.0001, "reuse")) for i in range(5)
        ]
        # 末尾を切り詰めたような不正行を 1 行混入させる。
        lines.append('{"framework": "fandhe-ai", "task": "train"')
        _write_jsonl(before_path, lines)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)]
            + [_rec("0.5.0", 0.006 + i * 0.0001, "reuse") for i in range(5)],
        )
        buf = io.StringIO()
        with redirect_stderr(buf):
            rc = compare_ab.main([before_path, after_path])
        self.assertEqual(rc, 2)
        self.assertIn("不正な行", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
