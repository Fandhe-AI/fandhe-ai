#!/usr/bin/env python3
"""`compare_gemm_gate.py` の stdlib unittest（イシュー #1142）。

GPU 不要・合成 fixture のみで完結する（`summarize_test.py`・
`compare_ab_test.py` と同じ方針）。ゲート判定の正しさを担保する最小構成に
絞る（happy path・レコード不足・要素単位検証無効の 3 ケース。網羅ではなく
判定ロジックの正しさの検証が目的）。
"""

import importlib.util
import json
import os
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "compare_gemm_gate",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_gemm_gate.py"),
)
compare_gemm_gate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_gemm_gate)


def _row(framework, size, mode, median_s, checksum=1.0, fail_count=0):
    total = size * size
    return {
        "framework": framework,
        "task": "gemm",
        "device": "cuda",
        "size": size,
        "mode": mode,
        "median_s": median_s,
        "checksum": checksum,
        "parity_total": total,
        "parity_fail_count": fail_count,
        "parity_max_abs_err": 0.0 if fail_count == 0 else 1.0,
        "parity_max_rel_err": 0.0 if fail_count == 0 else 1.0,
    }


def _write_jsonl(rows):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return f.name


class EvaluateSizeTest(unittest.TestCase):
    def test_happy_path_achieved(self):
        rows = [_row("fandhe-ai", 1024, "reuse", 0.010) for _ in range(5)] + [
            _row("candle", 1024, "fresh", 0.020) for _ in range(5)
        ]
        result = compare_gemm_gate.evaluate_size(rows, 1024)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["achieved"])
        self.assertAlmostEqual(result["fandhe_median_s"], 0.010)
        self.assertAlmostEqual(result["candle_median_s"], 0.020)
        self.assertAlmostEqual(result["ratio_candle_over_fandhe"], 2.0)

    def test_insufficient_records_is_undeterminable(self):
        rows = [_row("fandhe-ai", 1024, "reuse", 0.010) for _ in range(4)] + [
            _row("candle", 1024, "fresh", 0.020) for _ in range(5)
        ]
        result = compare_gemm_gate.evaluate_size(rows, 1024)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("件数が", result["reason"])

    def test_excess_records_is_undeterminable(self):
        # codex-review P1 指摘（PR #1166）: 6 件以上ある場合に無条件で末尾
        # 5 件を採用すると、不利な run の後に有利な run を追記するだけで
        # 判定対象を差し替えられてしまう。件数の完全一致検証で過不足いずれも
        # 判定不能に倒すことを確認する。
        rows = [_row("fandhe-ai", 1024, "reuse", 0.010) for _ in range(6)] + [
            _row("candle", 1024, "fresh", 0.020) for _ in range(5)
        ]
        result = compare_gemm_gate.evaluate_size(rows, 1024)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("件数が", result["reason"])

    def test_parity_failure_is_undeterminable_with_reason(self):
        rows = [_row("fandhe-ai", 2048, "reuse", 0.010) for _ in range(4)] + [
            _row("fandhe-ai", 2048, "reuse", 0.010, fail_count=2)
        ] + [_row("candle", 2048, "fresh", 0.020) for _ in range(5)]
        result = compare_gemm_gate.evaluate_size(rows, 2048)
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("要素単位検証が無効", result["reason"])
        self.assertIn("fail=2/", result["reason"])


class LoadRowsTfz32Test(unittest.TestCase):
    def test_non_bool_tf32_row_is_skipped_with_warning(self):
        # codex-review P0 指摘（PR #1166）: `tf32` が bool 以外（`1`・
        # `"true"` 等）だと `r.get("tf32", False) is True` が常に False を
        # 返すため、そのまま通すと不正形式入力で FP32 ゲートへ混入できて
        # しまう。`load_rows` の時点で行ごと除外し警告することを確認する。
        good = _row("fandhe-ai", 1024, "reuse", 0.010)
        bad = dict(_row("fandhe-ai", 1024, "reuse", 0.011))
        bad["tf32"] = 1  # 不正型（bool ではない）
        path = _write_jsonl([good, bad])
        try:
            rows, warnings = compare_gemm_gate.load_rows(path)
        finally:
            os.unlink(path)
        self.assertEqual(len(rows), 1)
        self.assertTrue(any("tf32" in w for w in warnings))


class MainCliTest(unittest.TestCase):
    def test_main_exit_code_reflects_achievement(self):
        achieved_rows = [_row("fandhe-ai", 1024, "reuse", 0.010) for _ in range(5)] + [
            _row("candle", 1024, "fresh", 0.020) for _ in range(5)
        ]
        path = _write_jsonl(achieved_rows)
        try:
            self.assertEqual(compare_gemm_gate.main([path]), 3)  # size 2048/4096 未計測 → 判定不能
        finally:
            os.unlink(path)

    def test_main_fails_closed_when_input_has_invalid_lines(self):
        # codex-review P0 指摘（PR #1166）: `load_rows` が破損 JSON・非
        # object・不正な `tf32` 型の行を warnings として除外するのみで、
        # `main` はそれを標準エラーへ表示するだけで終了コードを失敗に
        # 変えなかった。全 size に正常な 5 行が揃っていれば、同じ入力
        # ファイルに不正行が混在していても exit code 0（達成扱い）に
        # なり得ていた fail-open な欠陥を再発防止する回帰テスト。
        rows = []
        for size in compare_gemm_gate.SIZES:
            rows += [_row("fandhe-ai", size, "reuse", 0.010) for _ in range(5)]
            rows += [_row("candle", size, "fresh", 0.020) for _ in range(5)]
        lines = [json.dumps(r) for r in rows]
        lines.append("{not valid json")  # 破損行（load_rows が warnings へ回収）
        path = _write_jsonl([])
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        try:
            exit_code = compare_gemm_gate.main([path])
            self.assertNotEqual(
                exit_code,
                0,
                "全 size 分の正常行が揃っていても、不正行混在の入力ファイルは"
                "非ゼロ終了しなければならない（fail-open 再発防止）",
            )
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
