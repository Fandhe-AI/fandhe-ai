#!/usr/bin/env python3
"""`compare_gemm_gate.py` の stdlib unittest（イシュー #1142・#1147）。

GPU 不要・合成 fixture のみで完結する（`summarize_test.py`・
`compare_ab_test.py` と同じ方針）。ゲート判定の正しさを担保する最小構成に
絞る（happy path・レコード不足・要素単位検証無効・`--device` 分離の各
ケース。網羅ではなく判定ロジックの正しさの検証が目的）。
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


def _row(framework, size, mode, median_s, checksum=1.0, fail_count=0, device="cuda"):
    total = size * size
    return {
        "framework": framework,
        "task": "gemm",
        "device": device,
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


class DeviceParamTest(unittest.TestCase):
    """`--device`（イシュー #1147・Metal 対応汎用化）の集計分離を検証する。"""

    def test_metal_device_rows_are_aggregated_when_selected(self):
        rows = [_row("fandhe-ai", 1024, "reuse", 0.010, device="metal") for _ in range(5)] + [
            _row("candle", 1024, "fresh", 0.020, device="metal") for _ in range(5)
        ]
        result = compare_gemm_gate.evaluate_size(rows, 1024, device="metal")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["achieved"])
        self.assertAlmostEqual(result["fandhe_median_s"], 0.010)
        self.assertAlmostEqual(result["candle_median_s"], 0.020)

    def test_device_metal_selected_but_input_is_cuda_is_undeterminable(self):
        # cuda 行しか無い入力へ --device metal を指定すると、device 不一致で
        # 全行が除外され件数不一致（0 件）により判定不能へ倒れることを確認
        # する（標本の取り違えを fail-closed で検出する）。
        rows = [_row("fandhe-ai", 1024, "reuse", 0.010, device="cuda") for _ in range(5)] + [
            _row("candle", 1024, "fresh", 0.020, device="cuda") for _ in range(5)
        ]
        result = compare_gemm_gate.evaluate_size(rows, 1024, device="metal")
        self.assertEqual(result["status"], "undeterminable")
        self.assertIn("件数が", result["reason"])

    def test_device_omitted_defaults_to_cuda(self):
        # 既定（device 省略）は従来どおり cuda 行のみを集計する（#1142 との
        # 後方互換）。
        rows = [_row("fandhe-ai", 1024, "reuse", 0.010) for _ in range(5)] + [
            _row("candle", 1024, "fresh", 0.020) for _ in range(5)
        ]
        result_default = compare_gemm_gate.evaluate_size(rows, 1024)
        result_explicit_cuda = compare_gemm_gate.evaluate_size(rows, 1024, device="cuda")
        self.assertEqual(result_default, result_explicit_cuda)
        self.assertEqual(result_default["status"], "ok")


class CpuDeviceTest(unittest.TestCase):
    """`--device cpu`（イシュー #1148）の集計分離・N=512 対応・fresh 参考列
    （判定に使わないこと）を検証する。"""

    def test_cpu_device_rows_are_aggregated_with_size_512(self):
        # cuda/metal に無い N=512 が cpu の対象形状に含まれることの確認
        # （`_SIZES_BY_DEVICE["cpu"]`）。
        rows = [_row("fandhe-ai", 512, "reuse", 0.010, device="cpu") for _ in range(5)] + [
            _row("candle", 512, "fresh", 0.020, device="cpu") for _ in range(5)
        ]
        result = compare_gemm_gate.evaluate_size(rows, 512, device="cpu")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["achieved"])
        self.assertAlmostEqual(result["fandhe_median_s"], 0.010)
        self.assertNotIn(512, compare_gemm_gate._SIZES_BY_DEVICE["cuda"])
        self.assertIn(512, compare_gemm_gate._SIZES_BY_DEVICE["cpu"])

    def test_cuda_rows_excluded_when_device_cpu_selected(self):
        # cuda 行が混在していても --device cpu 指定時は除外され、cpu 行
        # のみで判定される（標本混入防止）。
        rows = (
            [_row("fandhe-ai", 512, "reuse", 0.010, device="cpu") for _ in range(5)]
            + [_row("candle", 512, "fresh", 0.020, device="cpu") for _ in range(5)]
            + [_row("fandhe-ai", 512, "reuse", 0.999, device="cuda") for _ in range(5)]
            + [_row("candle", 512, "fresh", 0.999, device="cuda") for _ in range(5)]
        )
        result = compare_gemm_gate.evaluate_size(rows, 512, device="cpu")
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["fandhe_median_s"], 0.010)

    def test_fresh_reference_column_present_when_five_rows_and_does_not_affect_achieved(self):
        # fandhe-ai fresh 行が 5 件そろい要素単位検証・checksum とも正式
        # 判定と同じ検証を通る場合、参考列 `fandhe_fresh_median_s` が付与
        # されるが `achieved` の値は fresh 行の有無に関わらず変わらない
        # （正式判定は reuse vs candle fresh のまま）。
        base_rows = [_row("fandhe-ai", 512, "reuse", 0.010, device="cpu") for _ in range(5)] + [
            _row("candle", 512, "fresh", 0.020, device="cpu") for _ in range(5)
        ]
        result_without_fresh = compare_gemm_gate.evaluate_size(base_rows, 512, device="cpu")
        self.assertNotIn("fandhe_fresh_median_s", result_without_fresh)

        rows_with_fresh = base_rows + [
            _row("fandhe-ai", 512, "fresh", 0.015, device="cpu") for _ in range(5)
        ]
        result_with_fresh = compare_gemm_gate.evaluate_size(rows_with_fresh, 512, device="cpu")
        self.assertEqual(result_with_fresh["status"], "ok")
        self.assertIn("fandhe_fresh_median_s", result_with_fresh)
        self.assertAlmostEqual(result_with_fresh["fandhe_fresh_median_s"], 0.015)
        self.assertEqual(result_with_fresh["achieved"], result_without_fresh["achieved"])
        self.assertAlmostEqual(
            result_with_fresh["fandhe_median_s"], result_without_fresh["fandhe_median_s"]
        )

    def test_fresh_reference_column_absent_when_row_count_insufficient(self):
        # fresh 行が 4 件（不足）の場合は参考列を付けない（正式判定
        # `achieved` には影響しない）。
        base_rows = [_row("fandhe-ai", 512, "reuse", 0.010, device="cpu") for _ in range(5)] + [
            _row("candle", 512, "fresh", 0.020, device="cpu") for _ in range(5)
        ]
        rows_with_short_fresh = base_rows + [
            _row("fandhe-ai", 512, "fresh", 0.015, device="cpu") for _ in range(4)
        ]
        result = compare_gemm_gate.evaluate_size(rows_with_short_fresh, 512, device="cpu")
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("fandhe_fresh_median_s", result)

    def test_main_accepts_device_cpu_flag(self):
        rows = []
        for size in compare_gemm_gate._SIZES_BY_DEVICE["cpu"]:
            rows += [_row("fandhe-ai", size, "reuse", 0.010, device="cpu") for _ in range(5)]
            rows += [_row("candle", size, "fresh", 0.020, device="cpu") for _ in range(5)]
        path = _write_jsonl(rows)
        try:
            self.assertEqual(compare_gemm_gate.main(["--device", "cpu", path]), 0)
        finally:
            os.unlink(path)


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
        for size in compare_gemm_gate._SIZES_BY_DEVICE["cuda"]:
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
