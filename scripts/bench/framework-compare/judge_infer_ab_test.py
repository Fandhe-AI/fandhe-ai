#!/usr/bin/env python3
"""`judge_infer_ab.py` の単体テスト（イシュー #1580）。

`compare_readout_ab_test.py` と同じ方式（ファイルパス指定 import・
tempfile への合成 JSONL 書き出し）。CI（`ci.yml` の `deps-forbidden`
ジョブ）は
`python3 -m unittest scripts/bench/framework-compare/judge_infer_ab_test.py`
で本ファイルを実行する。

検証観点:
- 正常系: `mode=reuse` の before/after 各 5 件から中央値・比率・checksum
  完全一致を算出し、非後退（`ratio <= 1.00`）かつ checksum 完全一致なら
  総合判定（終了コード 0）は ADOPT。
- reuse セルが後退（`ratio > 1.00`）すれば終了コードは 3（REJECT）。
- reuse セルの件数が不足していれば終了コードは 2（undetermined）。
- `mode=fresh`（対照）の後退は終了コードへ影響しない。
- `framework != "fandhe-ai"`／`device != "metal"`／`task != "infer"` の行は
  判定対象から除外する。
"""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))

_SPEC = importlib.util.spec_from_file_location(
    "judge_infer_ab", os.path.join(HERE, "judge_infer_ab.py")
)
judge_infer_ab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(judge_infer_ab)


def _rec(mode, median_s, checksum=1.5, warmup=20, iters=20, task="infer", device="metal",
         framework="fandhe-ai"):
    return {
        "framework": framework,
        "version": "0.0.0-test",
        "task": task,
        "device": device,
        "size": 64,
        "median_s": median_s,
        "q1_s": median_s,
        "q3_s": median_s,
        "checksum": checksum,
        "warmup": warmup,
        "iters": iters,
        "mode": mode,
    }


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class JudgeInferAbTest(unittest.TestCase):
    def _run(self, before_rows, after_rows, threshold=1.00, rounds=5):
        with tempfile.TemporaryDirectory() as tdir:
            before_path = os.path.join(tdir, "before.jsonl")
            after_path = os.path.join(tdir, "after.jsonl")
            _write_jsonl(before_path, before_rows)
            _write_jsonl(after_path, after_rows)
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = judge_infer_ab.main(
                    [
                        "--before",
                        before_path,
                        "--after",
                        after_path,
                        "--threshold",
                        str(threshold),
                        "--rounds",
                        str(rounds),
                    ]
                )
            return code, buf.getvalue()

    def test_adopt_when_reuse_non_regressed_and_checksum_exact(self):
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)] + [
            _rec("fresh", 0.020, checksum=2.5) for _ in range(5)
        ]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)] + [
            _rec("fresh", 0.020, checksum=2.5) for _ in range(5)
        ]
        code, out = self._run(before, after)
        self.assertEqual(code, 0, out)
        self.assertIn("ADOPT", out)

    def test_reject_when_reuse_regresses(self):
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
        after = [_rec("reuse", 0.020, checksum=1.5) for _ in range(5)]
        code, _out = self._run(before, after)
        self.assertEqual(code, 3)

    def test_reject_when_reuse_checksum_differs(self):
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=1.500001) for _ in range(5)]
        code, _out = self._run(before, after)
        self.assertEqual(code, 3)

    def test_undetermined_when_reuse_row_count_insufficient(self):
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(4)]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        code, _out = self._run(before, after)
        self.assertEqual(code, 2)

    def test_fresh_regression_does_not_affect_exit_code(self):
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)] + [
            _rec("fresh", 0.010, checksum=2.5) for _ in range(5)
        ]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)] + [
            _rec("fresh", 0.050, checksum=2.5) for _ in range(5)
        ]
        code, out = self._run(before, after)
        self.assertEqual(code, 0, out)

    def test_ignores_rows_from_other_frameworks_devices_or_tasks(self):
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)] + [
            _rec("reuse", 999.0, checksum=9.0, framework="candle"),
            _rec("reuse", 999.0, checksum=9.0, device="cuda"),
            _rec("reuse", 999.0, checksum=9.0, task="gemm"),
        ]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 0, out)

    def test_undetermined_when_checksum_missing_on_both_sides(self):
        """codex-review 指摘: `checksum` が両腕とも欠落（`None`）のとき
        `None == None` を「完全一致」と誤判定せず、undetermined に倒す。"""
        before = [_rec("reuse", 0.010, checksum=None) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=None) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)
        self.assertIn("missing or non-numeric checksum", out)

    def test_undetermined_when_checksum_key_absent(self):
        before_rows = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
        after_rows = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        for r in before_rows + after_rows:
            del r["checksum"]
        code, out = self._run(before_rows, after_rows)
        self.assertEqual(code, 2, out)
        self.assertIn("missing or non-numeric checksum", out)

    def test_undetermined_when_median_s_missing(self):
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        del before[0]["median_s"]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)
        self.assertIn("median_s", out)

    def test_undetermined_when_median_s_non_positive(self):
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
        after = [_rec("reuse", 0.0, checksum=1.5) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)

    def test_undetermined_when_median_s_is_infinite(self):
        """codex-review 指摘: `median_s` が +Infinity のとき ratio=0 に
        より非後退が成立してしまい ADOPT へ誤って倒れるのを防ぐ。"""
        before = [_rec("reuse", float("inf"), checksum=1.5) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)
        self.assertIn("median_s", out)

    def test_undetermined_when_median_s_is_nan(self):
        before = [_rec("reuse", float("nan"), checksum=1.5) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)

    def test_undetermined_when_checksum_is_string(self):
        """codex-review 指摘: `checksum` が None 以外の非数値（文字列・
        空配列等）でも「一致」として ADOPT に採用されてしまうのを防ぐ。"""
        before = [_rec("reuse", 0.010, checksum="1.5") for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum="1.5") for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)
        self.assertIn("checksum", out)

    def test_undetermined_when_checksum_is_empty_list(self):
        before = [_rec("reuse", 0.010, checksum=[]) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=[]) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)

    def test_undetermined_when_warmup_missing_on_both_sides(self):
        """codex-review 指摘: `warmup` が両腕とも欠落（`None`）のとき
        `{None} == {None}` を「一致」と誤判定せず undetermined に倒す。"""
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        for r in before + after:
            r["warmup"] = None
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)
        self.assertIn("warmup", out)

    def test_undetermined_when_iters_missing_on_both_sides(self):
        before = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        for r in before + after:
            del r["iters"]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)
        self.assertIn("iters", out)

    def test_undetermined_when_warmup_negative(self):
        before = [_rec("reuse", 0.010, checksum=1.5, warmup=-1) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)

    def test_undetermined_when_iters_non_positive(self):
        before = [_rec("reuse", 0.010, checksum=1.5, iters=0) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)


if __name__ == "__main__":
    unittest.main()
