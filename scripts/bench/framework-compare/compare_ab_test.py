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
from contextlib import redirect_stderr, redirect_stdout

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
        # median_s=-1.0 は load_rows の schema 検証（Review 指摘・#1083）で
        # 既に除外されるため、5 件中 1 件欠けレコード不足として判定不能になる。
        self.assertIn("レコード不足", result["reason"])

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

    def test_schema_invalid_train_row_is_skipped_with_warning(self):
        """codex-review P0 指摘（PR #1088）: 構文上有効な JSON だが必須
        フィールド（median_s）を欠く train レコードが、他に正常行が
        MIN_RECORDS 件以上あっても無検証に黙って除外されないことを確認する
        （`load_rows` 段階で warning 付きスキップされる）。
        """
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        broken = _rec("0.4.0", 0.012, "fresh")
        del broken["median_s"]
        recs.append(broken)
        _write_jsonl(before_path, recs)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        rows, warnings = compare_ab.load_rows(before_path)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("schema", warnings[0])

    def test_schema_invalid_train_row_with_non_object_value_is_skipped(self):
        """JSON としては有効だが object でない行（scalar・配列等）は
        schema 不正としてスキップされることを確認する。"""
        before_path, after_path = self._paths()
        lines = [json.dumps(_rec("0.4.0", 0.012 + i * 0.0001, "fresh")) for i in range(5)]
        lines.append(json.dumps([1, 2, 3]))
        _write_jsonl(before_path, lines)
        rows, warnings = compare_ab.load_rows(before_path)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("JSON object", warnings[0])

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
        # warmup=0（正整数でない）は load_rows の schema 検証（Review
        # 指摘・#1083）で当該 1 件が既に除外されるため、5 件中 4 件のみ残り
        # レコード不足として判定不能になる。
        self.assertIn("レコード不足", result["reason"])

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
        # version="" は load_rows の schema 検証（Review 指摘・#1083）で
        # 全 5 件が除外されるため、レコード不足として判定不能になる。
        self.assertIn("レコード不足", result["reason"])

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
        # version=None は load_rows の schema 検証（Review 指摘・#1083）で
        # 全 5 件が除外されるため、レコード不足として判定不能になる。
        self.assertIn("レコード不足", result["reason"])

    def test_device_missing_train_row_is_skipped_with_warning(self):
        """codex-review P0 指摘（PR #1088）: device 欠損の train 行が
        warning にならず `_train_records` で黙って除外され、他に正常行が
        MIN_RECORDS 件以上残れば A/B 判定が成功扱いになってしまう fail-open
        を塞ぐ（`_train_row_schema_error` の device 型検証）。"""
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        broken = _rec("0.4.0", 0.012, "fresh")
        del broken["device"]
        recs.append(broken)
        _write_jsonl(before_path, recs)
        rows, warnings = compare_ab.load_rows(before_path)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("device", warnings[0])

    def test_size_wrong_type_train_row_is_skipped_with_warning(self):
        """codex-review P0 指摘（PR #1088）: size が文字列型（例:
        `"64"`）の train 行が warning にならず `_train_records` で黙って
        除外される fail-open を塞ぐ（`_train_row_schema_error` の size
        型検証。bool は int のサブクラスのため明示的に除外する）。"""
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        broken = _rec("0.4.0", 0.012, "fresh")
        broken["size"] = "64"
        recs.append(broken)
        _write_jsonl(before_path, recs)
        rows, warnings = compare_ab.load_rows(before_path)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("size", warnings[0])

    def test_main_returns_nonzero_for_device_size_corrupted_input_even_with_enough_valid_rows(
        self,
    ):
        """codex-review P0 指摘（PR #1088）の再現シナリオ全体: device 欠損・
        size 型不正の行が各 mode に混在しても、他の正常行が MIN_RECORDS 件
        以上残っているだけで A/B 判定を成功扱いにしない（fail-closed）こと
        を CLI 経由で確認する。"""
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        broken_device = _rec("0.4.0", 0.012, "fresh")
        del broken_device["device"]
        broken_size = _rec("0.4.0", 0.012, "fresh")
        broken_size["size"] = "64"
        recs.append(broken_device)
        recs.append(broken_size)
        _write_jsonl(before_path, recs)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)],
        )
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            rc = compare_ab.main([before_path, after_path])
        self.assertNotEqual(rc, 0)

    def test_version_with_pipe_char_train_row_is_skipped_with_warning(self):
        """codex-review P0 指摘（PR #1088）: version に Markdown 制御文字
        （`|`）を含む train 行が非空文字列検証だけでは通過し、Markdown 表
        へ未エスケープで連結されて列・行を追加できてしまう脆弱性を塞ぐ
        （`_train_row_schema_error` の `_VERSION_RE` allowlist 検証）。"""
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        broken = _rec("0.4.0 | injected |", 0.012, "fresh")
        recs.append(broken)
        _write_jsonl(before_path, recs)
        rows, warnings = compare_ab.load_rows(before_path)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("version", warnings[0])

    def test_version_with_newline_train_row_is_skipped_with_warning(self):
        """version に改行を含む行も allowlist 検証で warning 化される
        ことを確認する（Markdown 表の行追加を防ぐ）。"""
        before_path, after_path = self._paths()
        recs = [_rec("0.4.0", 0.012 + i * 0.0001, "fresh") for i in range(5)]
        broken = _rec("0.4.0\n| injected |", 0.012, "fresh")
        recs.append(broken)
        _write_jsonl(before_path, recs)
        rows, warnings = compare_ab.load_rows(before_path)
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(warnings), 1)
        self.assertIn("version", warnings[0])


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

    def test_main_table_shows_undeterminable_not_ok_when_input_has_invalid_line(
        self,
    ):
        """Cursor Bugbot 指摘（PR #1088）: `has_invalid_lines` の判定が
        render() の後で行われていたため、不正行を検出しても出力 Markdown
        表の「判定」列には不正行検出前に確定した中央値・比率つきの "ok" が
        残ってしまう fail-open の回帰防止。標準出力（Markdown 表）自体に
        "ok" が現れず、"判定不能" が現れることを検証する。
        """
        before_path = os.path.join(self.tmpdir.name, "before.jsonl")
        after_path = os.path.join(self.tmpdir.name, "after.jsonl")
        lines = [
            json.dumps(_rec("0.4.0", 0.012 + i * 0.0001, "fresh")) for i in range(5)
        ]
        lines += [
            json.dumps(_rec("0.4.0", 0.010 + i * 0.0001, "reuse")) for i in range(5)
        ]
        lines.append('{"framework": "fandhe-ai", "task": "train"')
        _write_jsonl(before_path, lines)
        _write_jsonl(
            after_path,
            [_rec("0.5.0", 0.009 + i * 0.0001, "fresh") for i in range(5)]
            + [_rec("0.5.0", 0.006 + i * 0.0001, "reuse") for i in range(5)],
        )
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            rc = compare_ab.main([before_path, after_path])
        self.assertEqual(rc, 2)
        table_text = out_buf.getvalue()
        self.assertNotIn("| ok |", table_text)
        self.assertIn("判定不能", table_text)



class PathMarkdownValidationTests(unittest.TestCase):
    """入力パスの Markdown インジェクション拒否（codex-review P0 指摘・
    PR #1088: バッククォート・改行を含むパスは render() へ到達させず
    非 0 終了する fail-closed 契約）の回帰テスト。"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        # 検証は load_rows より前に行われるため after 側は実在ファイルで
        # なくてよいが、正常パスとの組合せ確認用に最小の正常入力を用意する。
        self.ok_path = os.path.join(self.tmpdir.name, "ok.jsonl")
        _write_jsonl(self.ok_path, [_rec("0.4.0", 0.01, "fresh")])

    def _assert_rejected(self, bad_path, expect_reason):
        err_buf = io.StringIO()
        out_buf = io.StringIO()
        with redirect_stderr(err_buf), redirect_stdout(out_buf):
            rc = compare_ab.main([bad_path, self.ok_path])
        self.assertEqual(rc, 2)
        self.assertIn(expect_reason, err_buf.getvalue())
        # レポート（Markdown）は一切出力されない（不正パスを埋め込んだ
        # 生成物を作らない）。
        self.assertEqual(out_buf.getvalue(), "")

    def test_main_rejects_path_with_backtick(self):
        self._assert_rejected("evil`# 挿入見出し`.jsonl", "バッククォート")

    def test_main_rejects_path_with_newline(self):
        self._assert_rejected("evil\n# 挿入見出し.jsonl", "制御文字")

    def test_main_rejects_after_path_too(self):
        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc = compare_ab.main([self.ok_path, "b\rad.jsonl"])
        self.assertEqual(rc, 2)
        self.assertIn("after パスが不正", err_buf.getvalue())

    def test_path_markdown_error_accepts_normal_paths(self):
        self.assertIsNone(compare_ab._path_markdown_error(self.ok_path))
        self.assertIsNone(
            compare_ab._path_markdown_error("results/raw/results-dgx-0.4.0.jsonl")
        )


if __name__ == "__main__":
    unittest.main()
