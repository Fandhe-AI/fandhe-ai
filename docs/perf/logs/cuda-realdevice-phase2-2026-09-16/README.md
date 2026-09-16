# CUDA（DGX Spark GB10・sm_121）実機 Phase 2 機能群 parity 実測（2026-09-16）

低レイヤー診断・機能網羅ツリー（ルート #1570）Phase 2 で「GB10 実機実測は未実施の
まま申し送り」となっていた `#[ignore]` テスト群を、DGX Spark GB10 ノード（転送元
コミット origin/main `3e43bbd0`。ノード側 `.rev-stamp` で確認済み）で一括実行した
記録。実装コードは一切変更していない（記録・ログ・doc 追記のみ）。tolerance／
baseline／判定規則の事後緩和も行っていない。

- 環境: `env_info.txt`（`hostname: masked`。GB10・driver 580.173.02・CUDA 13.0
  V13.0.88・rustc 1.97.0・Linux 6.17.0-1031-nvidia aarch64。内部ホスト名・ユーザー名・
  絶対パスはログ内でも `<host>`／`/home/<user>` へ置換済み）
- 負荷: 実行中 load average 約 1.0〜1.4・GPU utilization 0〜3%（ComfyUI／Kokoro
  常駐だがアイドル）。本ディレクトリの項目は parity・bit 一致・決定性の確認で
  負荷非依存
- ビルド: 同期ツリー外 `CARGO_TARGET_DIR`・`--release`・backend-cuda 側は
  `make test-ignored-cuda` と同じ `--all-features` で統一
- 実行スクリプト: `step1.sh`（全体非後退）・`step2.sh`（Phase 2 群。`summary.tsv`
  を生成）。いずれも `docs/real-hardware-verification-env.md` §4.4 の切り離し方式

## 1. 全体非後退（`make test-ignored-cuda` 相当）

`make-test-ignored-cuda.log`: 1 本目（`--all-features --no-fail-fast -- --ignored
--skip sgd_update_segment_captures_then_replays_bit_identically --skip
different_config_key_produces_a_different_segment_key`）と 2 本目
（`graph_capture_real_device --test-threads=1`）。`--no-fail-fast` は全バイナリの
一覧取得のために付加（Makefile 自体は変更しない）。

結果: **309 pass・18 FAIL**（1 本目 exit=101・2 本目 exit=0）。

| FAIL | 分類 |
|---|---|
| `reduce_parity.rs` 6 件（`sum`／`max`／`min`／empty／NaN・inf／決定性） | **新規 FAIL・共通原因 §3.1**（reduction カーネルの NVRTC コンパイルエラー） |
| `typed_ops_f16_parity::typed_f16_elementwise_and_reduction_match_cpu_backend_ops_rounded`・`typed_ops_bf16_parity::typed_ops_bf16_matches_across_shapes` | 同上（§3.1。`sum` へ委譲した時点で失敗） |
| `cpu_cuda_mma_parity::mma_f16_k4096_stress`・`cpu_cuda_wmma_parity::wmma_f16_k4096_stress`・`gemm_wmma_f16_opt::wmma_f16_opt_k4096_stress` | 既知（`docs/backend-cuda-real-device-testing.md` §4／§5.1・`docs/perf/cuda-parity-baseline.md`。fail_count 101／99／81 は記録値と一致） |
| `gemm_mma_tf32x3::mma_tf32x3_{matches_reference_across_shapes,k4096_stress}`・`gemm_tf32_optin::gemm_tf32x3_optin_on_matches_cpu_across_shapes` | 既知（#1356 で P1 不成立・baseline 不承認を確定済み。`docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md`） |
| `gemm_tf32_optin::gemm_tf32_optin_on_matches_cpu_across_shapes`・`tensor_core_real_device::tensor_core_parity_record` | 既知（TF32 512³ の厳密ゼロ fail 不成立。`docs/perf/cuda-parity-baseline.md` の形状二分方式の対象） |
| `gemm::tests::wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096` | 既知の性能比較テスト（staged 1.004 対 opt 1.067 TFLOPS。`docs/perf/cuda-parity-baseline.md` §2 記載） |
| `module_cache_wiring_tests::cuda_gemm_new_second_construction_reuses_module_cache` | 既知の並列干渉（`docs/real-hardware-verification-env.md` §4「module_cache_wiring_tests は直列実行」。lib テスト既定並列で evict 156 件） |

## 2. 高優先（Phase 2 機能 parity）結果一覧

pass 数は各ログ末尾の `test result:` 行の実測値。`cuda_` は同一バイナリに Metal
実機必須テストが混在するファイルのテスト名フィルタ。

### 2a. Conv（#1771・親 #1645／#1606）→ `docs/perf/logs/conv-realdevice-1771/cuda/`

| ログ（run1／run2 とも同結果） | pass | fail | 失敗テスト |
|---|---|---|---|
| `im2col_col2im_parity.log` | 4 | 0 | — |
| `conv2d_backend_parity.log` | 1 | 1 | `cuda_conv2d_backward_matches_cpu` |
| `conv1d_backend_parity.log` | 2 | 2 | `cuda_conv1d_backward_matches_cpu`・`cuda_conv1d_matches_manual_reshape_conv2d_bit_exact` |
| `nn_conv_backend_parity.log` | 5 | 1 | `cuda_sequential_conv1d_matches_manual_reshape_conv2d_bit_exact` |
| `check_determinism.log` | — | — | 規則 4) PASS（4 ログ群とも run-to-run bit 同一。pass したテストの出力行のみ比較） |

**verdict = FAIL（規則 2)・3) 未成立・§3.1 の共通原因）**。規則 1)・4) は成立。
conv 演算の数値不一致は未観測（forward 全 pass・`mse_loss` 経由の nn 層 backward pass）。

### 2b. BatchNorm（#1735）

| ログ | pass | fail | 判定 |
|---|---|---|---|
| `backend-cuda_batch_norm_parity.log` | 10 | 0 | PASS |
| `facade_batch_norm_backend_parity.log`（`cuda_` 4 件: train/infer forward・train backward・rank4） | 4 | 0 | PASS |

### 2c. Pooling（#1729・追従 PR #1888 配線済み）

| ログ | pass | fail | 判定 |
|---|---|---|---|
| `backend-cuda_pooling_real_device.log`（`--lib pooling::pooling_real_device_tests`） | 11 | 0 | PASS（値・索引 bit 完全一致） |
| `facade_pooling_backend_parity.log`（`cuda_` 3 件） | 3 | 0 | PASS |

### 2d. backend-cuda 側 parity 群

| ログ | pass | fail | 判定 |
|---|---|---|---|
| `backend-cuda_sort_topk_parity.log` | 3 | 0 | PASS |
| `backend-cuda_scan_parity.log` | 1 | 0 | PASS |
| `backend-cuda_gather_scatter_parity.log` | 3 | 0 | PASS |
| `backend-cuda_cast_parity.log` | 1 | 0 | PASS |
| `backend-cuda_cast_ops_contract.log` | 0 | 0 | **未実行**（`--ignored` に一致するテストなし。2 件 filtered out。GPU 非依存の契約テストで CI 側で常時実行） |
| `backend-cuda_unique_parity.log` | 1 | 0 | PASS |
| `backend-cuda_constant_pad_parity.log` | 1 | 0 | PASS |
| `backend-cuda_interpolate_parity.log` | 1 | 0 | PASS |
| `backend-cuda_gemm_batched_parity.log` | 2 | 0 | PASS |
| `backend-cuda_lib_gemm_batched.log`（`--lib -- --ignored gemm_batched`） | 0 | 0 | **未実行**（フィルタに一致する `#[ignore]` lib テストなし） |
| `backend-cuda_typed_ops_f16_parity.log` | 2 | 1 | `typed_f16_elementwise_and_reduction_match_cpu_backend_ops_rounded` が §3.1 で FAIL（gemm 等 2 件は pass） |
| `backend-cuda_typed_ops_bf16_parity.log` | 0 | 1 | `typed_ops_bf16_matches_across_shapes` が §3.1 で FAIL |
| `backend-cuda_typed_ops_f64_contract.log` | 0 | 0 | **未実行**（`#[ignore]` なし。GPU 非依存契約テスト） |
| `backend-cuda_rnn_cell_parity.log`（#1647。GB10 初実測） | 1 | 0 | PASS |
| `backend-cuda_rmsnorm_parity.log`／`rmsnorm_backward_parity.log`（#1596。GB10 初実測） | 3／2 | 0 | PASS |
| `backend-cuda_scalar_op_parity.log` | 1 | 0 | PASS |
| `backend-cuda_mse_parity.log`（`mse_matches_cpu_across_shapes`） | 1 | 0 | PASS |
| `backend-cuda_where_masked_fill_parity.log` | 1 | 0 | PASS |

### 2e. facade 側（`tape_for(Device::Cuda(0))` 経路・`cuda_` フィルタ）

| ログ | pass | fail | 判定 |
|---|---|---|---|
| `facade_sort_topk_backend_parity.log` | 2 | 0 | PASS |
| `facade_index_ops_backend_parity.log`（where／masked_fill／narrow／one_hot） | 4 | 0 | PASS |
| `facade_cast_backend_parity.log` | 1 | 0 | PASS |
| `facade_unique_backend_parity.log` | 1 | 0 | PASS |
| `facade_constant_pad_backend_parity.log` | 1 | 0 | PASS |
| `facade_interpolate_backend_parity.log` | 2 | 0 | PASS |
| `facade_batched_matmul_backend_parity.log` | 2 | 0 | PASS |
| `facade_scalar_ops_backend_parity.log` | 6 | 0 | PASS |
| `facade_scalar_unary_transcendental_backend_parity.log` | 3 | 0 | PASS |
| `facade_activation_gelu_softplus_backend_parity.log` | 3 | 0 | PASS |
| `facade_bce_backend_parity.log` | 1 | 0 | PASS |
| `facade_nll_kl_div_backend_parity.log` | 3 | 0 | PASS |
| `facade_huber_backend_parity.log` | 1 | 0 | PASS |
| `facade_dropout_backend_parity.log` | 1 | 0 | PASS |
| `facade_attention_backend_parity.log` | 1 | 1 | `cuda_sdpa_backward_dq_matches_cpu` §3.1 |
| `facade_mha_backend_parity.log` | 2 | 0 | PASS |
| `facade_compat_sequential_layers_backend_parity.log` | 3 | 0 | PASS |
| `facade_no_grad_detach_backend_parity.log` | 0 | 1 | `cuda_detach_weight_grad_matches_cpu` §3.1 |
| `facade_backward_accumulate_backend_parity.log` | 0 | 1 | `cuda_backward_accumulate_weight_grad_matches_cpu` §3.1 |
| `facade_norm_backend_parity.log` | 2 | 0 | PASS |
| `facade_var_norm_backend_parity.log` | 0 | 1 | `var`／`std` は `sum` reduction に帰着するため §3.1 で FAIL |
| `facade_device_transfer_backend_parity.log` | 1 | 0 | PASS |
| `facade_scan_ops_backend_parity.log` | 0 | 0 | **未実行**（`cuda_` フィルタに一致する `#[ignore]` テストなし） |
| `facade_device_enumeration.log`（常時実行） | 5 | 0 | PASS（CUDA デバイス列挙成立） |

## 3. FAIL の分析

### 3.1 CUDA reduction カーネルの NVRTC コンパイルエラー（新規・共通原因。計 16 テスト）

panic はすべて
`CudaUnavailable("nvrtc compile error: … options: [\"--gpu-architecture=compute_121\"], log: \"default_program(11): error: identifier \\\"INFINITY\\\" is undefined\\n      float acc = -INFINITY;`。

- 機構: `crates/backend-cuda/src/kernels_reduce.rs`（`sum`／`max`／`min` を 1 つの NVRTC
  プログラムに持つ）が `-INFINITY` マクロを使う（242・258・282・297・327 行付近）。
  NVRTC は `<math.h>` を暗黙に含めないため `INFINITY` が未定義となり、プログラム
  全体がコンパイルに失敗する。結果として `CudaBackendOps::sum`／`max`／`min` は GB10
  実機で常に `CudaUnavailable` を返す
- 導入元: `-INFINITY` は #1675（`581d5208`。CUDA sum／max reduce 実装。GB10 未実測の
  まま出荷）で導入され、#1830（`1a1bcd5a`。min）が同プログラムへ追加した。本セッション
  が GB10 での初実測であり、reduce プログラムは実機で一度もコンパイルに成功して
  いないと考えられる（`reduce_parity.rs` 全 6 件が同一エラー）
- 影響範囲（本セッションで確認）: `reduce_parity` 6 件・typed f16／bf16 の reduction
  2 件・facade の `var`／`std`（`var_norm_backend_parity`）1 件に加え、Metal 側と同型で
  「`Var::sum(None)` を loss とする backward テスト」7 件（conv 4・sdpa backward・
  detach・backward_accumulate。backend-cuda 側 `gather_scatter_parity`／`constant_pad_parity`／
  `interpolate_parity` は Metal 側と異なり tape backward テストを持たないため対象外）。`Var::sum`
  （`crates/autodiff/src/var.rs:918`）はホストフォールバックを持たないため
  エラーがそのまま伝播する
- 帰結: 対象演算の forward parity・`mse_loss` 経由 backward は pass しており、
  **演算自体の数値不一致は未観測**。FAIL（判定不能）のまま記録し規則の事後緩和は
  行わない。公開面への含意: `tape_for(Device::Cuda(0))` 上の `Var::sum`／`max`／
  `min`／`var`／`std` は GB10 実機で失敗する（CPU は成功・Metal は別理由〈reduction
  カーネル未実装〉で失敗）。是正はコード変更（例: `__int_as_float(0x7f800000)` や
  `-1.0f/0.0f` 等 NVRTC で解決できる定義への置換）であり本 PR 対象外。
  `out-of-scope-tracking.md` に従い切り出し先の起票可否はユーザー判断へ回す

### 3.2 既知 FAIL（10 件）

§1 の表のとおり、f16 Tensor Core K=4096 ストレス 3 件・TF32／3×TF32 の厳密ゼロ fail
不成立 5 件・staged 対 opt の性能比較 1 件・`module_cache_wiring_tests` の並列干渉
1 件はいずれも既存記録と一致し、非後退の範囲内。

## 4. 実測していない項目

- 中優先（#1689／#1560／#1692／#1585／#1590）は本 README の対象外（時間があれば別 PR）
- Stream-K／persistent／managed 等の確定済み項目は再計測していない
