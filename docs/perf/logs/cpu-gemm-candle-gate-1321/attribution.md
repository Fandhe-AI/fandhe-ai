# イシュー #1321 帰属根拠（v0.7.0 → HEAD `ced4d14` の CPU NN GEMM reuse 経路差分）

本ファイルは `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §22.2 の帰属表の算出根拠。
`git diff v0.7.0..HEAD --stat -- crates/backend-cpu/src crates/facade/src
crates/autodiff/src crates/tensor-core/src` は `diff_v0.7.0_ced4d14_cpu_path.txt` を参照。

## Phase 別施策と結線状態（origin/main `ced4d14` 時点。イシュー計画 §2 を実装時に再確認したもの）

| Phase | 施策 | 結線状態 | 根拠（grep 実測） |
|---|---|---|---|
| 1 | 出力並列ゼロ埋め `zeroed_output`（#1299/#1301） | **無効**（`GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS = usize::MAX`。#1448 で差し戻し） | `crates/backend-cpu/src/ops.rs:153` |
| 2 | `TwoDDynamic` 2D 動的分配（#1311/#1312/#1313） | **本番結線済み**（`TWO_D_DYNAMIC_PRODUCTION_ENABLED = true`・`TWO_D_JOBS_PER_WORKER = 2`） | `crates/backend-cpu/src/gemm_blis/mod.rs:2987,3008` |
| 2 | 既定スレッド数の大コア限定（#1363/#1364） | 無効（`BIG_CORE_LIMIT_ENABLED = false`。REJECT） | `crates/backend-cpu/src/thread_limit.rs:93` |
| 3 | 候補 3 KC 再スイープ（#1315） | REJECT（`KC=256` 維持・コード変更なし） | `docs/perf/cpu-gemm-candle-cpu-retune.md` §8.1 |
| 3 | 候補 1 laneq ベクトル転置（#1317/#1318） | REJECT（`#[cfg(test)]` 維持・本番未結線） | `docs/perf/cpu-gemm-b-laneq-vec-transpose.md` §7/§8 |
| 3 | 候補 2 prefetch（#1319） | docs のみ（unsafe 未着手） | `docs/perf/cpu-gemm-prefetch-decision.md` |
| — | 借用ビュー readout（#1337） | feature `host-view-readout` 既定 OFF | `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §15 |

## `ced4d14` PR タイトルと実装状態の不一致に関する注記

`ced4d14`（PR #1448）のコミット件名は「出力並列ゼロ埋め（#1299）を DGX/M4 Max 両実機実測で
有効化する」だが、実際にマージされた最終状態は `GEMM_OUTPUT_PARALLEL_ZERO_MIN_ELEMS =
usize::MAX`（**無効化のまま**）である。これは PR 内で `2 << 20` へ一度有効化した後、
codex-review 指摘（事前宣言した規則 4〈candle 比の非後退〉が緩和なしでは 6 セル中 3 セルで
不成立）を受けて同一 PR 内で `usize::MAX` へ差し戻したため（詳細は
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §20 全体、とくに末尾の差し戻し記述）。
コミット件名のみを読むと「有効化された」と誤読しうるため、本イシューの §22.2 帰属表は
コミット件名ではなく `crates/backend-cpu/src/ops.rs:153` の実コード状態を正として記録する。

## 結論

結線後 HEAD（`ced4d14`）の CPU NN GEMM reuse 本番経路と v0.7.0 の実質的な差は
**「`RowPanel` → `TwoDDynamic`」のみ**。他の差分（`autodiff::optim::device_store`・
`Var` 等）は VJP 転置入口（#1213）・デバイス常駐勾配更新（#1212）等、NN GEMM reuse の
主経路（`gemm_blis_parallel_with_transpose`）とは別経路（backward・update フェーズ）の
変更であり、`compare_gemm_gate.py` が計測する `Var::matmul`（forward GEMM）reuse 経路には
到達しない。
