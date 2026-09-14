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
- 空行以外の JSON 解析失敗（破損した計測行）、またはデコードには
  成功したがトップレベルが JSON object でない行（`null`／`[]`／数値等）
  が 1 件でも混在すれば、正常な行が `--rounds` 件残っていても判定全体が
  undetermined・終了コード 2 になる（空行のみは従来どおり読み飛ばして
  許容する）。
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
        self.assertIn("missing, non-numeric, or non-finite checksum", out)

    def test_undetermined_when_checksum_key_absent(self):
        before_rows = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
        after_rows = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
        for r in before_rows + after_rows:
            del r["checksum"]
        code, out = self._run(before_rows, after_rows)
        self.assertEqual(code, 2, out)
        self.assertIn("missing, non-numeric, or non-finite checksum", out)

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

    def test_undetermined_when_checksum_is_infinite_on_both_sides(self):
        """codex-review 指摘: checksum が両腕とも `Infinity` で埋まって
        いると、数値型検査だけでは一致（`inf == inf`）とみなされ、
        時間比が閾値以下なら ADOPT へ倒れてしまう。`math.isfinite` に
        より非有限値を undetermined へ倒す。"""
        before = [_rec("reuse", 0.010, checksum=float("inf")) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=float("inf")) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)
        self.assertIn("checksum", out)

    def test_undetermined_when_checksum_is_negative_infinite_on_both_sides(self):
        before = [_rec("reuse", 0.010, checksum=float("-inf")) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=float("-inf")) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)

    def test_undetermined_when_checksum_is_nan_on_both_sides(self):
        """`nan != nan` のため一致判定自体は元々成立しないが、fail-closed
        方針上は非有限値そのものを判定不能として明示的に拒否する。"""
        before = [_rec("reuse", 0.010, checksum=float("nan")) for _ in range(5)]
        after = [_rec("reuse", 0.008, checksum=float("nan")) for _ in range(5)]
        code, out = self._run(before, after)
        self.assertEqual(code, 2, out)

    def test_undetermined_when_checksum_is_out_of_range_literal_on_both_sides(self):
        """codex-review 指摘: `json.loads` は `1e400` のような float の
        表現範囲を超える数値リテラルを `inf` として受理するため、この
        経路からも非有限 checksum に到達しうる。JSONL へ直接書き込んで
        `json.dumps` を経由せず `1e400` リテラルを再現する。"""
        with tempfile.TemporaryDirectory() as tdir:
            before_path = os.path.join(tdir, "before.jsonl")
            after_path = os.path.join(tdir, "after.jsonl")
            base_rows = [_rec("reuse", 0.010) for _ in range(5)]
            with open(before_path, "w", encoding="utf-8") as f:
                for r in base_rows:
                    r = dict(r)
                    r["checksum"] = None  # placeholder; overwritten below via string replace
                    line = json.dumps(r).replace("null", "1e400", 1)
                    f.write(line + "\n")
            with open(after_path, "w", encoding="utf-8") as f:
                for r in base_rows:
                    r = dict(r)
                    r["checksum"] = None
                    line = json.dumps(r).replace("null", "1e400", 1)
                    f.write(line + "\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = judge_infer_ab.main(
                    ["--before", before_path, "--after", after_path]
                )
            self.assertEqual(code, 2, buf.getvalue())
            self.assertIn("checksum", buf.getvalue())

    def test_undetermined_when_malformed_json_line_mixed_in(self):
        """codex-review P0 指摘: 途中で切れた計測行（壊れた JSON）が
        混在していても、正常な reuse 行が `--rounds` 件残っていれば
        従来は黙って除外され ADOPT・終了コード 0 になり得た。壊れた行を
        検出したら判定全体を undetermined（終了コード 2）へ倒し、行番号
        付きの理由を報告する。"""
        with tempfile.TemporaryDirectory() as tdir:
            before_path = os.path.join(tdir, "before.jsonl")
            after_path = os.path.join(tdir, "after.jsonl")
            before_rows = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
            after_rows = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
            _write_jsonl(before_path, before_rows)
            _write_jsonl(after_path, after_rows)
            # 途中で切れた（末尾が欠落した）計測行を末尾に 1 行追記する。
            with open(before_path, "a", encoding="utf-8") as f:
                f.write('{"framework": "fandhe-ai", "task": "infer", "mode": "reu\n')

            buf = io.StringIO()
            with redirect_stdout(buf):
                code = judge_infer_ab.main(
                    ["--before", before_path, "--after", after_path]
                )
            self.assertEqual(code, 2, buf.getvalue())
            self.assertIn("undetermined", buf.getvalue())
            self.assertIn(f"{before_path}:6", buf.getvalue())

    def test_blank_lines_only_are_still_permitted(self):
        """空行のみ（データを持たない行）は従来どおり読み飛ばしてよく、
        malformed 判定の対象にはしない（P0 是正が空行除外の既存挙動まで
        破壊していないことの確認）。"""
        with tempfile.TemporaryDirectory() as tdir:
            before_path = os.path.join(tdir, "before.jsonl")
            after_path = os.path.join(tdir, "after.jsonl")
            before_rows = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
            after_rows = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
            _write_jsonl(before_path, before_rows)
            _write_jsonl(after_path, after_rows)
            with open(before_path, "a", encoding="utf-8") as f:
                f.write("\n\n   \n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = judge_infer_ab.main(
                    ["--before", before_path, "--after", after_path]
                )
            self.assertEqual(code, 0, buf.getvalue())
            self.assertIn("ADOPT", buf.getvalue())

    def _run_with_appended_raw_line(self, raw_line):
        """codex-review P2 指摘: JSONL の行が構文的には正しい JSON
        （`json.loads` 自体は成功する）でも、トップレベルが `null`／
        `[]`／数値等の非オブジェクト値だと、後続の `obj.get(...)` が
        `AttributeError` で未捕捉のまま異常終了しうる。以下の共通
        ヘルパーは、正常な reuse 行 5 件の末尾に `raw_line` を
        そのまま追記した before.jsonl で `main` を実行し、
        `(code, output)` を返す。"""
        with tempfile.TemporaryDirectory() as tdir:
            before_path = os.path.join(tdir, "before.jsonl")
            after_path = os.path.join(tdir, "after.jsonl")
            before_rows = [_rec("reuse", 0.010, checksum=1.5) for _ in range(5)]
            after_rows = [_rec("reuse", 0.008, checksum=1.5) for _ in range(5)]
            _write_jsonl(before_path, before_rows)
            _write_jsonl(after_path, after_rows)
            with open(before_path, "a", encoding="utf-8") as f:
                f.write(raw_line + "\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = judge_infer_ab.main(
                    ["--before", before_path, "--after", after_path]
                )
            return code, buf.getvalue(), before_path

    def test_undetermined_when_null_line_mixed_in(self):
        code, out, before_path = self._run_with_appended_raw_line("null")
        self.assertEqual(code, 2, out)
        self.assertIn("undetermined", out)
        self.assertIn(f"{before_path}:6", out)
        self.assertIn("NoneType", out)

    def test_undetermined_when_empty_array_line_mixed_in(self):
        code, out, before_path = self._run_with_appended_raw_line("[]")
        self.assertEqual(code, 2, out)
        self.assertIn("undetermined", out)
        self.assertIn(f"{before_path}:6", out)
        self.assertIn("list", out)

    def test_undetermined_when_numeric_line_mixed_in(self):
        code, out, before_path = self._run_with_appended_raw_line("42")
        self.assertEqual(code, 2, out)
        self.assertIn("undetermined", out)
        self.assertIn(f"{before_path}:6", out)
        self.assertIn("int", out)


if __name__ == "__main__":
    unittest.main()
