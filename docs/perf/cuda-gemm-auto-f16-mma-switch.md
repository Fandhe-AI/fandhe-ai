# CudaGemmAuto::run_f16 の MatrixUnit 分岐 mma 優先化 前後比較（#1156・#1160）

イシュー #1156 の受け入れ条件 R5（ユーザー承認条件）「結線前後で同一プロトコル・
5 回計測中央値の比較を取り、後退確認時は結線しない」に対応する記録。設計の正は
`docs/dispatch-rules-design.md` §5.6。

## 状態: 実測完了・本番結線済み（イシュー #1160・GB10 実機実測 2026-09-04）

`crates/backend-cuda/src/gemm_auto.rs::MMA_PRIORITY_PRODUCTION_ENABLED` を
`true`（mma 優先。§5.6 の設計目標どおり `CudaMmaGemm → CudaWmmaGemm → Tiled`）
へ本番結線した。下記実測（転送込み auto 経路・独立 5 回起動・run-median）が
512/1024/2048/4096 いずれも §5「非後退の判定基準」を満たしたことによる。

**数値一致（parity）の実測記録は #1158 で GB10 実機確認済み**
（`docs/perf/cuda-parity-baseline.md` §12。切替前〈本番既定・WMMA 優先〉・
切替後〈ノード側限定フリップの mma 優先〉双方で 5 回実行同値・既存
baseline 行からの後退なしを確認）。本イシュー（#1160）でも `MMA_PRIORITY_
PRODUCTION_ENABLED = true` の HEAD で `cargo test --all-features --test gemm_auto
-- --ignored --nocapture --test-threads=1` を実行し、8 件中 7 件 pass・
`run_f16_matches_cpu_reference_across_aligned_shapes`（厳密ゼロ fail）・
`f16_matrix_unit_impl_reports_selected_implementation`（更新後の期待値。
mma_available()=true・整列形状で `Mma` を返すことを確認）を含めて green を
再確認した。唯一の FAILED は `run_f16_k4096_stress_non_regression_route_aware`
であり、選択経路 `MmaF16` の baseline ceiling が未承認（`None`）のため
fail-closed に panic する既知 red（`docs/perf/cuda-parity-baseline.md`
§12.4/§12.5・§12.6）で、本 PR ではこのテスト・`BASELINES` を変更しない
（承認・反映は別途）。直接経路の参照証跡
`cpu_cuda_mma_parity.rs::mma_f16_k4096_stress_non_regression` は green
（`fail_count` 等が #1158 §12.3 の記録値と一致）。

## 実測表（転送込み auto 経路・独立 5 回起動・run-median。GB10・2026-09-04）

`bash scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh` の出力（プロセス
5 回・形状ごとに `auto_f16_tflops_run_median`）。base は
`MMA_PRIORITY_PRODUCTION_ENABLED = false`（ノード側限定フリップ。#1156 以前
と同じ wmma 優先）、after は HEAD（同定数 `true`。mma 優先）。GPU アイドル・
他プロセス非混在（`nvidia-smi --query-compute-apps`・`utilization.gpu` を
各計測前後で確認済み）。

| dim | base run 値（5 回） | base run-median | after run 値（5 回） | after run-median | after/base |
|---|---|---|---|---|---|
| 512  | [2.3373, 2.3350, 2.3192, 2.3250, 2.3302] | 2.3302 | [4.0702, 4.0398, 4.0369, 4.0761, 4.0831] | 4.0702 | 1.747 |
| 1024 | [5.7046, 5.8007, 5.6896, 5.8037, 5.7975] | 5.7975 | [11.5845, 11.5160, 11.5576, 11.4795, 11.5516] | 11.5516 | 1.993 |
| 2048 | [4.6330, 4.6553, 4.6082, 4.6194, 4.6219] | 4.6219 | [21.5916, 21.6420, 21.5543, 21.6007, 21.5924] | 21.5924 | 4.672 |
| 4096 | [0.4586, 0.4475, 0.4415, 3.2734, 3.4152] | 0.4586 | [0.4433, 0.5120, 8.4268, 0.5131, 0.5219] | 0.5131 | 1.119 |

4096 は base・after とも #1130（`docs/perf/cuda-wmma-f16-perf-triage.md` §4.2
に記録済みの per-call アロケーション病態。未解決。イシュー #1130 のスコープ）
の影響で run 間に大きなばらつきがある（低い値の run が多数・稀に高い値の
run が混じる二峰性）。この病態は base 側にも存在するため mma 優先化自体が
原因ではなく、転送込み計測が病態を継承していることを示す（下記「4096 の
判定」参照）。

## 判定（§5「非後退の判定基準」の適用結果）

- **512/1024/2048**: after の run-median が base の run-median を明確に上回る
  （1.75〜4.67 倍）。一次判定を満たす
- **4096**: after の run-median（0.5131）は base の 5 run 範囲
  `[0.4415, 3.4152]` 内に収まっており、「病態支配下で同等・後退なし」の
  基準を満たす。判別証跡（病態の影響を受けないカーネル単体プロトコル）は
  `docs/perf/cuda-wmma-f16-perf-triage.md` §3.1・
  `tests/dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record`
  の実測（dim=4096: `mma_over_wmma` = 10.81 倍。本ドキュメント §「判別証跡」
  参照）が mma 優先化の妥当性を裏付ける
- **総合判定**: 全形状で非後退（後退なし）。#1156 R5 のユーザー承認条件を
  満たしたため `MMA_PRIORITY_PRODUCTION_ENABLED` を `true` へ本番結線した

### 判別証跡（カーネル単体・転送区間を含まないプロトコル）

`docs/perf/cuda-wmma-f16-perf-triage.md` §3.1（イシュー #1123・GB10 実機実測
2026-09-03。`launch → synchronize` のみの「カーネル単体」計測で per-call
アロケーション病態〈#1130〉の影響を受けない）:

| dim | wmma_f16_opt TFLOPS | mma_sync_f16 TFLOPS | mma/wmma |
|---|---|---|---|
| 2048 | 7.135 | 51.732 | 7.25 |
| 4096 | 4.664 | 50.411 | 10.81 |

`tensor_core_tflops_record`（本イシューで本番 f16 経路の計測へ追従させた
バージョン。§「tensor_core_tflops_record の実測」参照）の M=N=K=4096 実測
（`mma_sync_f16` 55.449 TFLOPS・`wmma_f16_opt` 4.522 TFLOPS）でも
`mma_over_wmma` ≈ 12.26 倍となり、同傾向を確認した。

## `tensor_core_tflops_record` の実測（本番 f16 経路への追従後）

`crates/backend-cuda/tests/tensor_core_real_device.rs::tensor_core_tflops_record`
（M=N=K=4096・カーネル単体プロトコル）を GB10 実機で実行（`--all-features`。
`internal-diagnostics` feature 込みの選択器整合性検査を含む）した結果、
**TF32 assert・f16 assert とも pass**（`test result: ok. 1 passed`）。

| path | TFLOPS | 備考 |
|---|---|---|
| tiled_f32（基準） | 10.013 | — |
| wmma_tf32_staged | 14.134 | TF32 assert 対象（pass。tiled f32 比 1.412 倍） |
| wmma_f16_opt（参考行。本番既定では主経路から外れる） | 4.522 | assert 対象外 |
| mma_sync_f16（本番 f16 経路。イシュー #1160） | 55.449 | f16 assert 対象（pass。tiled f32 比 5.538 倍） |

f16 assert は #1123 是正版（比較対象 `wmma_f16_opt`）で GB10 実測が **red**
（4.391〜4.496 TFLOPS < tiled f32 6.776〜6.790 TFLOPS）だったが、本イシューで
比較対象を「本番 f16 経路が実際に選ぶ実装」（`CudaGemmAuto::mma_available()
== true` のため `mma_sync_f16`）へ差し替えたことで **pass に転じた**。これは
イシュー #1131 の完了条件（`docs/perf/cuda-wmma-f16-perf-triage.md` §6）を
満たす。

## 実機実行手順（再現用）

転送は rsync 方式（`docs/real-hardware-verification-env.md`。ホスト名等はローカル
管理値を使い本ドキュメントには書かない）。

計測用バイナリは `examples/gemm_auto_f16_mma_switch_bench.rs`（`bench-harness::
run`・`MeasurementConfig::default`〈warmup 20・計測 20〉による `bench_run` を
プロセス起動ごとに各形状 1 回だけ実行し、`CudaGemmAuto::run_f16` を転送込み
で計測する）。独立 5 回起動・run-median の集約は wrapper スクリプト
`scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh` が担う（`cargo build`
で 1 回ビルドした後、生成された実行ファイルを外側のシェルから 5 回
fork+exec する。真に独立した OS プロセスとして 5 回実行することでプロセス
起動・クロック・熱条件の独立実行間ばらつきを反映する。イシュー #1160 是正:
`BIN` の探索先を `CARGO_TARGET_DIR`〈未設定時は cargo 既定の `target/`〉に
揃え、実機手順が指定する外部ビルド成果物ディレクトリでも解決できるように
した）。

```bash
bash scripts/bench/run_gemm_auto_f16_mma_switch_bench.sh
```

after（HEAD。`MMA_PRIORITY_PRODUCTION_ENABLED = true`）はそのままビルド・
実行すれば得られる。base（同定数を一時的に `false` へ書き換えたノード側
コピー。コミットしない）と比較する場合は、同じ手順を該当コピーで実行し
`auto_f16_tflops_run_median` を突き合わせる。

## 補足: 転送込み計測の位置づけ

`examples/gemm_mma_bench.rs::measure_mma_f16` は GPU 実行のみ（H2D/D2H を
計測区間外）を計測するのに対し、本ベンチ（`gemm_auto_f16_mma_switch_bench.rs`）
は `CudaGemmAuto::run_f16` の実利用経路（転送込み）をそのまま計測する。両者は
計測対象が異なるため TFLOPS 値を直接比較しない（`gemm_mma_bench.rs` 側の
既存注記と同じ整理）。
