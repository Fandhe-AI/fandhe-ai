# CUDA バックエンド Phase 4 全 `#[ignore]` テスト実機実測（2026-09-18・DGX Spark GB10）

本ディレクトリは、Phase 4 で追加した CUDA カーネル（#1948 argmax／argmin・
#1949 `log_softmax` backward・#1950 LayerNorm／RMSNorm backward）の個別
`#[ignore]` テストと、`crates/backend-cuda` 配下のすべての `#[ignore]` テストを
DGX Spark GB10（sm_121）で実行した結果を記録する。Metal 版の先例は
`docs/perf/logs/metal-realdevice-phase4-2026-09-18/README.md`。

実装コードは一切変更していない（記録・ログ・doc 追記のみ）。FAIL は是正せず
記録のみとし、tolerance／baseline／事前登録判定規則の事後緩和も行っていない。

- 環境: `env_info.txt`（`hostname: masked`。GB10・driver 580.173.02・CUDA 13.0・
  Ubuntu 24.04.4 aarch64・Linux 6.17.0-1031-nvidia・rustc 1.97.0。ログ内の
  ユーザー名・絶対パスは `<home>` へ置換済み）
- 転送元コミット: origin/main `536c56a8`（`feat(facade): ONNX export（OnnxModel::to_bytes／to_path）を facade へ公開する (#2026)`）
- 負荷: 開始時 load average 1.09・終了時 1.15（`uptime_before.txt`／`uptime_after.txt`）・
  GPU utilization 0%。専有（常駐プロセス 2 つは停止せずアイドルのまま同居。
  実名は記載しない）
- 時刻（UTC）: 個別テスト 01:47:40Z〜01:47:48Z・全群 01:47:48Z〜02:04:36Z・
  直列再実行 02:06:21Z〜02:06:25Z
- ビルド: 同期ツリー外 `CARGO_TARGET_DIR`・`--release`・全群は
  `make test-ignored-cuda` と同じ `--all-features`

## 実行コマンド

個別（#1948〜#1950。各ログは対象イシューのディレクトリへ収納）:

```sh
# #1948 → docs/perf/logs/cuda-argmax-argmin-1948/reduce_parity-argm.log
cargo test -p fandhe-ai-backend-cuda --release --test reduce_parity -- --ignored --nocapture argm
# #1949 → docs/perf/logs/cuda-log-softmax-backward-1949/{log_softmax_backward_parity,facade-softmax_backend_parity}.log
cargo test -p fandhe-ai-backend-cuda --release --test log_softmax_backward_parity -- --ignored --nocapture
cargo test -p fandhe-ai --release --test softmax_backend_parity -- --ignored --nocapture
# #1950 → docs/perf/logs/cuda-norm-backward-1950/{norm_backward_parity,facade-norm_backend_parity,rmsnorm_backward_parity,rmsnorm_parity,layer_norm_parity}.log
cargo test -p fandhe-ai-backend-cuda --release --test norm_backward_parity -- --ignored --nocapture
cargo test -p fandhe-ai --release --test norm_backend_parity -- --ignored --nocapture cuda_
cargo test -p fandhe-ai-backend-cuda --release --test rmsnorm_backward_parity -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test rmsnorm_parity -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test layer_norm_parity -- --ignored --nocapture
```

全群（Makefile `test-ignored-cuda` と同形＋`--no-fail-fast`。#1931 と同じ）:

```sh
# → full_ignored_cuda.log
cargo test -p fandhe-ai-backend-cuda --release --all-features --no-fail-fast -- --ignored --nocapture \
    --skip sgd_update_segment_captures_then_replays_bit_identically \
    --skip different_config_key_produces_a_different_segment_key
# → full_ignored_cuda_graph_capture.log
cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device -- \
    --ignored --nocapture --test-threads=1
```

直列再実行（全群で FAIL した既知外 3 件を `--test-threads=1` で単独実行。参考記録・規則外）:

```sh
# → serial-init_cost_diag.log
cargo test -p fandhe-ai-backend-cuda --release --all-features --lib -- --ignored --nocapture --test-threads=1 init_cost_diag_gemm_new_lru_cold_vs_warm
# → serial-managed_placement.log
cargo test -p fandhe-ai-backend-cuda --release --all-features --test managed_placement_real_device -- --ignored --nocapture --test-threads=1 managed_placement_does_not_leak_on_real_hardware
# → serial-wmma_tf32_staged.log
cargo test -p fandhe-ai-backend-cuda --release --all-features --test gemm_wmma_tf32_staged -- --ignored --nocapture --test-threads=1 wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096
```

## 1. 個別テスト（#1948〜#1950）

すべて 0 fail。各ディレクトリの README「実測結果」節に転記済み。

| イシュー | ログ | pass | fail |
|---|---|---|---|
| #1948 | `cuda-argmax-argmin-1948/reduce_parity-argm.log` | 5 | 0 |
| #1949 | `cuda-log-softmax-backward-1949/log_softmax_backward_parity.log` | 4 | 0 |
| #1949 | `cuda-log-softmax-backward-1949/facade-softmax_backend_parity.log` | 3 | 0 |
| #1950 | `cuda-norm-backward-1950/norm_backward_parity.log` | 5 | 0 |
| #1950 | `cuda-norm-backward-1950/facade-norm_backend_parity.log` | 4 | 0 |
| #1950 | `cuda-norm-backward-1950/rmsnorm_backward_parity.log`（非後退確認） | 2 | 0 |
| #1950 | `cuda-norm-backward-1950/rmsnorm_parity.log`（非後退確認） | 3 | 0 |
| #1950 | `cuda-norm-backward-1950/layer_norm_parity.log`（非後退確認） | 4 | 0 |

## 2. 全群（`full_ignored_cuda.log`＋`full_ignored_cuda_graph_capture.log`）

- **総スコア**: **330 pass／12 FAIL**（1 本目 328 pass・12 FAIL・exit=101、
  2 本目 graph_capture 2 pass・exit=0。FAIL 名一覧は `failed_names.txt`）
- 基準 #1931（2026-09-16・転送元 `cf062643`。
  `docs/perf/logs/cuda-reduce-nvrtc-infinity-1893/make-test-ignored-cuda.log`）:
  315 pass／13 FAIL

### 2.1 by-name 差分（基準 #1931 ログとの機械比較）

`test <name> ... ok|FAILED` 行（`--nocapture` で結果が後続行に分かれるものを含む）を
テスト名で突合した結果。python3 標準ライブラリの一時スクリプトで算出（収録なし）。

| 区分 | 件数 | テスト名 |
|---|---|---|
| 後退（前回 ok → 今回 FAILED） | **1** | `wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096`（`tests/gemm_wmma_tf32_staged.rs`。staged 4.280 対 tiled f32 4.437 TFLOPS。直列単独では ok〈§3〉） |
| 消失（前回に存在し今回結果なし） | 0 | — |
| 新規 ok（前回に存在しない。multiset） | 14 | #1948（5）: `argmax_and_argmin_match_cpu_reference_on_real_device`・`argmax_and_argmin_nan_and_infinity_semantics_match_cpu_on_real_device`・`argmax_and_argmin_match_cpu_reference_for_transposed_view_on_real_device`・`argmax_and_argmin_empty_reduction_semantics_match_cpu_on_real_device`・`argmax_and_argmin_are_run_to_run_deterministic_on_real_device`／#1949（4）: `log_softmax_backward_matches_cpu_across_shapes`・`backend_ops_log_softmax_backward_matches_cpu_reference_across_shapes`・`log_softmax_backward_cancelling_upstream_grad_matches_cpu`・`log_softmax_backward_is_run_to_run_bit_identical`／#1950（5）: `rmsnorm_backward_matches_cpu_across_shapes`〈`tests/norm_backward_parity.rs`。既存 `tests/rmsnorm_backward_parity.rs` の同名テストとは別バイナリで、テスト名の出現回数が 1 → 2 になる〉・`layer_norm_backward_matches_cpu_across_shapes`・`norm_backward_zero_element_contract`・`norm_backward_numerical_stability_and_determinism`・`norm_backward_cancelling_input_detects_reduction_order_regression` |
| 前回 FAILED → 今回 ok | 2 | `gemm::tests::wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096`（既知 #9。性能比較）・`small_shape_matrix_unit_has_no_floor_tflops_record`（#1931 では並列干渉 FAIL。今回は並列実行のまま ok） |
| 両方 FAILED | 11 | 既知 9（§2.2）＋`init_cost_diag_tests::init_cost_diag_gemm_new_lru_cold_vs_warm`＋`managed_placement_does_not_leak_on_real_hardware` |

数式の確認: 315（前回 ok）− 1（後退）＋ 14（新規 ok）＋ 2（前回 FAILED → ok）＝ 330。
FAIL は 11（両方 FAILED）＋ 1（後退）＝ 12。facade 側（`-p fandhe-ai`）の `#[ignore]`
テストは全群コマンドの対象外（個別 §1 のみ）。

### 2.2 FAIL 12 件の分類

| FAIL | 分類 |
|---|---|
| `mma_f16_k4096_stress`・`wmma_f16_k4096_stress`・`wmma_f16_opt_k4096_stress` | 既知（f16 Tensor Core K=4096 ストレス 3。`cuda-realdevice-phase2-2026-09-16/README.md` §3.2） |
| `mma_tf32x3_k4096_stress`・`mma_tf32x3_matches_reference_across_shapes`・`gemm_tf32x3_optin_on_matches_cpu_across_shapes`・`gemm_tf32_optin_on_matches_cpu_across_shapes`・`tensor_core_parity_record` | 既知（TF32／3×TF32 厳密ゼロ fail 不成立 5。同 §3.2） |
| `module_cache_wiring_tests::cuda_gemm_new_second_construction_reuses_module_cache` | 既知（並列干渉 1。同 §3.2。#1931 の `module_cache_wiring_serial.log` で直列 pass 済み。今回は直列再実行対象に含めていない） |
| `init_cost_diag_tests::init_cost_diag_gemm_new_lru_cold_vs_warm` | 既知 10 件の外。並列実行時に LRU warm の `CudaGemm::new` が cold 3.687 s に対し 6.386 s（`init_cost_diag_tests.rs:484`）。#1931 でも並列 FAIL・直列 pass。**今回も直列単独で ok**（cold 198.929 ms／warm 0.061 ms。§3） |
| `wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096` | 既知 10 件の外・**今回初の FAIL**（#1931 では並列でも ok）。staged 4.280 TFLOPS が tiled f32 4.437 TFLOPS を下回る性能比較（`gemm_wmma_tf32_staged.rs:244`）。**直列単独で ok**（§3） |
| `managed_placement_does_not_leak_on_real_hardware` | 既知 10 件の外。並列で `leaked=18100224 bytes`（約 17.3 MiB）> 8 MiB。#1931 でも並列 FAIL（11.96 MB）・直列 pass だったが、**今回は直列単独でも FAILED**（`leaked=45703168 bytes`・約 43.6 MiB。§3・§4） |

内訳: 既知 10 件のうち 9 件が再現（既知 #9 `gemm::tests::wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096` は今回 ok）＋
既知外 3 件（並列干渉で直列 ok 2・直列でも FAIL 1）＝ 12。

### 2.3 事前登録規則の字義判定

`cuda-reduce-nvrtc-infinity-1893/README.md`「(5) 全体非後退」の規則
「FAIL が既知 10 件（`cuda-realdevice-phase2-2026-09-16/README.md` §3.2）を超えないこと」
は **12 > 10 で字義どおりには不充足**として記録する（規則の事後緩和はしない。
#1931 の 13 > 10 と同じ扱い）。超過 3 件はいずれも数値不一致（parity）ではなく
時間・デバイス全体メモリ量に依存する計測系テストであるが、その事実は規則の
充足判定を変えない。直列再実行の結果は別欄（§3）に置き、規則充足の根拠にはしない。

Phase 4 の新規テスト 14 件（§2.1）はすべて ok。#1948〜#1950 の事前登録規則
（各 README）はいずれも充足。

## 3. 直列再実行（`serial-*.log`。参考記録・規則外）

| テスト | 結果 |
|---|---|
| `init_cost_diag_tests::init_cost_diag_gemm_new_lru_cold_vs_warm`（`--lib --test-threads=1`） | **ok**（cold 198.929 ms／warm 0.061 ms） |
| `wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096`（`--test gemm_wmma_tf32_staged --test-threads=1`） | **ok** |
| `managed_placement_does_not_leak_on_real_hardware`（`--test managed_placement_real_device --test-threads=1`） | 1 回目 **FAILED**（`leaked=45703168 bytes`。`managed_placement_real_device.rs:344`。`serial-managed_placement.log`）／2 回目 **ok**（2.45 s。`serial-managed_placement-rerun2.log`）／3 回目 **ok**（3.12 s。`serial-managed_placement-rerun3.log`） |

`managed_placement` の 2 回目・3 回目は 1 回目の FAIL を受けて同一条件（直列単独・
`--test-threads=1`）で追加実行したもの（UTC 2026-09-18T02:2xZ・load1 0.06・GPU util 0%・
parity 突合完了後）。**直列単独 3 回のうち 1 回 FAIL・2 回 ok** を事実として記録し、
1 回目の FAIL を差し替えない（§2 の総スコア 330 pass／12 FAIL・§2.3 の規則判定は不変）。
`module_cache_wiring_tests`（既知の並列干渉）は今回の直列対象に含めていない
（#1931 で直列 pass を記録済み）。

## 4. `managed_placement_does_not_leak_on_real_hardware` の分析（是正・閾値変更なし）

### 4.1 テストが測っているもの（`crates/backend-cuda/tests/managed_placement_real_device.rs:299-349`）

1. `set_managed_placement_enabled(true)` のガード下で `CudaMemory::alloc_zeroed(&[16 * 1024 * 1024])`
   （f32 64 MiB 相当）を 1 回ウォームアップして drop
2. `CudaDevice::context().mem_get_info()`（`cuMemGetInfo` 相当）で **デバイス全体の
   free bytes** を `free_before` として取得
3. 同サイズの managed `alloc_zeroed`／drop を 100 回反復（各回の確保成功を `expect`）
4. さらに 1 回確保・解放したあと `mem_get_info` を再取得し `free_after` とする
5. `leaked = free_before.saturating_sub(free_after)` が **8 MiB 未満**であることを
   `assert!`（ソースコメント: managed 配置の解放は同期 `cuMemFree`〈`UnifiedSlice::drop`〉
   のため全解放済みのはず、多少のドライバ内部フラグメンテーションを許容する緩い閾値）

つまり判定量は「本プロセスの確保・解放の差」ではなく「デバイス全体の free bytes の
前後差」であり、`saturating_sub` により free が増えた場合のみ 0 に丸められる（減った
場合は他要因の減少もそのまま `leaked` に計上される）。

### 4.2 観測事実

| 実測 | 実行形態 | `leaked` | 判定 |
|---|---|---|---|
| #1903（2026-09-16・phase2） | 並列 | （記録なし） | pass |
| #1931（2026-09-16） | 並列 | 11.96 MB | FAIL |
| #1931 | 直列単独 | — | pass |
| 本実測（2026-09-18） | 並列 | 18,100,224 B（約 17.3 MiB） | FAIL |
| 本実測 | 直列単独 1 回目（02:06Z） | 45,703,168 B（約 43.6 MiB） | FAIL |
| 本実測 | 直列単独 2 回目（02:2xZ・load1 0.06・GPU util 0%） | （閾値未満。ログに値なし） | ok（2.45 s） |
| 本実測 | 直列単独 3 回目（同上） | （閾値未満。ログに値なし） | ok（3.12 s） |

- `leaked` の値は run ごとに異なり、64 MiB（反復 1 回分）の整数倍でもない
- ループ内の 100 回＋1 回の `alloc_zeroed` はすべて成功しており（`expect` 非発火）、
  確保容量の枯渇は観測されていない
- 直列単独 3 回のうち 1 回目のみ FAIL・2 回目・3 回目は ok。#1931（直列 1 回・pass）
  とは 1 回目の結果が異なる。「同一条件で再現しない」ことを flaky と断定はしない
  （3 回中 1 回 FAIL という事実のみ記録）

### 4.3 仮説（未検証）

GB10 は CPU／GPU が物理メモリを共有する unified memory 構成であり、
`cuMemGetInfo` が返す free bytes は本プロセスの managed 確保だけでなく、同居する
他プロセス（本実測では常駐 2 プロセス。停止していない）やドライバ側の
キャッシュ・ページ移動の影響を受けうる。free bytes が本テストの外側の要因で
数十 MiB 単位で減少すれば、本プロセスにリークがなくても `leaked` が閾値 8 MiB を
超える。`leaked` の値が反復サイズの整数倍にならず run ごとにばらつく事実・確保が
一度も失敗していない事実はこの仮説と整合するが、**検証はしていない**（本プロセス
単独の managed 確保量を追跡する測定・常駐プロセスを停止した状態での再実行のいずれも
未実施）。#1931 では同じ常駐プロセスが同居した状態で直列 pass しており、本実測でも
同じ常駐プロセスが同居したまま直列 2 回目・3 回目は ok だったため、常駐の
有無だけでは説明できない（その時点の活動量が未知数）。同一プロセス内の
`UnifiedSlice::drop` 経路に実際のリークがある可能性も排除できていないため、
本記録では「直列単独 3 回のうち 1 回 FAIL（1 回目・`leaked=45703168`）・2 回 ok」の
事実のみを確定事項とし、原因帰属・是正・閾値変更（8 MiB）は行わない。

## 5. #1959（Adam／AdamW 常駐 step）について

#1959 の CUDA 分は **カーネル未実装**（`BackendOps::adam_step_device` は CUDA では
既定の `Unsupported`。`docs/perf/logs/adam-device-step-1959/README.md`）のため、
本実測に実測対象は存在しない。#1948〜#1950 と同列の「実測済み」には含めない。

## 6. 保存ファイル

- `full_ignored_cuda.log`・`full_ignored_cuda_graph_capture.log`（全群）
- `failed_names.txt`（全群の FAIL 名 12 件）
- `serial-init_cost_diag.log`・`serial-managed_placement.log`・`serial-wmma_tf32_staged.log`（直列再実行）
- `serial-managed_placement-rerun2.log`・`serial-managed_placement-rerun3.log`（`managed_placement` の直列単独 2 回目・3 回目。いずれも ok）
- `uptime_before.txt`・`uptime_after.txt`
- `env_info.txt`（`hostname: masked`。同内容を #1948〜#1950 の各ディレクトリにも複製）
- 個別ログは `docs/perf/logs/cuda-argmax-argmin-1948/`・`cuda-log-softmax-backward-1949/`・
  `cuda-norm-backward-1950/` へ収納

## 判定

- 個別（#1948〜#1950）: 全 pass（各 README の事前登録規則を充足）
- 全群: 330 pass／12 FAIL。事前登録規則「FAIL ≤ 既知 10」は字義どおり**不充足**
  （12 > 10）。by-name 後退 1 件（性能比較・直列 ok）・新規 ok 14 件・前回 FAIL → ok 2 件。
  `managed_placement` は直列単独 3 回のうち 1 回 FAIL・2 回 ok（原因未特定・是正なし・
  1 回目の FAIL は差し替えない）
