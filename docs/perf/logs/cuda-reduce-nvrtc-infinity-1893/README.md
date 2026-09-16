# イシュー #1893 実測記録（reduce カーネルの NVRTC `INFINITY` 未定義是正）

## 位置づけ

#1893 の是正（`kernels_reduce.rs` の単位元を bit パターン直接構成へ置換）
後の DGX Spark GB10 実機再実測記録。**2026-09-16（UTC）に実測済み**
（イシュー #1931・親 #1930・ルート #1920。転送元コミット origin/main
`cf062643`。ノード側 `.rev-stamp` で確認済み）。実装コードは一切変更して
いない（記録・ログ・doc 追記のみ）。tolerance／baseline／判定規則の事後緩和
も行っていない。

- 環境: `env_info.txt`（`hostname: masked`。GB10・driver 580.173.02・CUDA 13.0
  V13.0.88・rustc 1.97.0・Linux 6.17.0-1031-nvidia aarch64。ログ内のユーザー名・
  絶対パスは `/home/<user>` へ置換済み）
- 負荷: 開始時 load average 0.26・GPU utilization 0%・終了時 load average 約 1.2
- ビルド: 同期ツリー外 `CARGO_TARGET_DIR`・`--release`・backend-cuda 側は
  `make test-ignored-cuda` と同じ `--all-features`
- 実行: 本 README「実行コマンド」(1)〜(5) を 1 本のスクリプトで切り離し実行
  （`run.log` に各コマンドと `test result` 行を記録）。加えて #1898 の CUDA 項目
  （`device_param_store_backend_parity -- --ignored on_cuda`。別イシューの記録）と
  既知の並列干渉 FAIL の直列再実行を同一セッションで実施

## 結果

### (1)〜(4) 対象 16 テスト（事前登録規則 1）

| ログ | pass | fail | 備考 |
|---|---|---|---|
| `reduce_parity-ignored.log` | 6 | 0 | `sum`／`max`／`min`／empty／NaN・inf／決定性。#1903 では 6 件すべて NVRTC コンパイルエラー |
| `typed_ops_f16_parity-ignored.log` | 3 | 0 | うち `typed_f16_elementwise_and_reduction_match_cpu_backend_ops_rounded` が対象（#1903 FAIL）。残り 2 件は同バイナリの他 `#[ignore]` |
| `typed_ops_bf16_parity-ignored.log` | 1 | 0 | `typed_ops_bf16_matches_across_shapes`（#1903 FAIL） |
| `var_norm_backend_parity-ignored.log` | 1 | 0 | facade `var`／`std`（`cuda_` フィルタ） |
| `conv1d_backend_parity-cuda-ignored.log` | 4 | 0 | `Var::sum(None)` を loss とする backward 込み |
| `conv2d_backend_parity-cuda-ignored.log` | 2 | 0 | 同上 |
| `nn_conv_backend_parity-cuda-ignored.log` | 6 | 0 | 同上（`compat::Sequential` 経由の backward 込み） |
| `attention_backend_parity-cuda-ignored.log` | 2 | 0 | sdpa backward 込み |
| `no_grad_detach_backend_parity-cuda-ignored.log` | 1 | 0 | |
| `backward_accumulate_backend_parity-cuda-ignored.log` | 1 | 0 | |

対象 16 件はすべて pass（**規則 1 充足**）。#1903 で判定不能だった backward・
bit 一致契約（conv 4・sdpa・detach・backward_accumulate・var／std）は本実測で
判定可能となり pass。

### (5) 全体非後退（`make-test-ignored-cuda.log`。事前登録規則 2）

`make test-ignored-cuda` 相当（#1903 `step1.sh` と同方式。`--no-fail-fast`・
graph_capture 2 件 `--skip` → `graph_capture_real_device --test-threads=1` 別実行）。

結果: **315 pass・13 FAIL**（1 本目 exit=101・2 本目 exit=0。#1903 は 309 pass・18 FAIL）。

FAIL 13 件の内訳:

| FAIL | 分類 |
|---|---|
| `mma_f16_k4096_stress`・`wmma_f16_k4096_stress`・`wmma_f16_opt_k4096_stress` | 既知（f16 Tensor Core K=4096 ストレス 3。#1903 §3.2） |
| `mma_tf32x3_k4096_stress`・`mma_tf32x3_matches_reference_across_shapes`・`gemm_tf32x3_optin_on_matches_cpu_across_shapes`・`gemm_tf32_optin_on_matches_cpu_across_shapes`・`tensor_core_parity_record` | 既知（TF32／3×TF32 厳密ゼロ fail 不成立 5。#1903 §3.2） |
| `gemm::tests::wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096` | 既知（staged 対 opt 性能比較 1。#1903 §3.2） |
| `module_cache_wiring_tests::cuda_gemm_new_second_construction_reuses_module_cache` | 既知（並列干渉 1。#1903 §3.2。`module_cache_wiring_serial.log` の直列再実行で 2/2 pass） |
| `init_cost_diag_tests::init_cost_diag_gemm_new_lru_cold_vs_warm` | **#1903 では pass・本実測で FAIL**（並列実行時に LRU warm の `CudaGemm::new` が cold 3.52 s に対し 11.62 s。下記「直列再実行」で pass） |
| `small_shape_matrix_unit_has_no_floor_tflops_record` | **#1903 では pass・本実測で FAIL**（転送のみ計測 10.0 ms がカーネル込み計測 76 µs を上回りプロトコル前提違反。下記「直列再実行」で pass） |
| `managed_placement_does_not_leak_on_real_hardware` | **#1903 では pass・本実測で FAIL**（`mem_get_info` の差 11.96 MB > 8 MiB 閾値。同一プロセス内で並走する他テストのデバイス確保が混入。下記「直列再実行」で pass） |

事前登録規則 2「FAIL が既知 10 件を超えないこと」は **13 > 10 で字義どおりには
不充足**として記録する（規則の事後緩和はしない）。ただし超過 3 件はいずれも
数値不一致ではなく、並列実行の時間・デバイス全体メモリ量への干渉に依存する
計測系テストであり、直列単独再実行では 3 件とも pass した（次節）。

by-name（multiset）差分（#1903 `make-test-ignored-cuda.log` の `test … ok` 名 309 件
との比較）:

- 後退（#1903 で ok・本実測で ok でない）: 3 件（上表の並列干渉 3 件）
- 新規 pass: 9 件（#1893 是正で解消した reduce 関連 8 件〈`reduce_parity` 6・typed
  f16／bf16 reduction 2〉＋ #1903 以降に追加された `pinned_h2d_upload_ab`〈#1907〉）

### 直列再実行（`parallel_interference_serial.log`。参考記録・規則外）

| コマンド | 結果 |
|---|---|
| `--lib -- --ignored --test-threads=1`（フィルタ指定が `--` の前に置かれたため lib の `#[ignore]` 72 件全部が直列実行された） | 72 pass・0 fail。`init_cost_diag_gemm_new_lru_cold_vs_warm` は cold 181.06 ms／warm 0.064 ms で pass。`module_cache_wiring_tests` 2 件・`wmma_tf32_staged_kernel_exceeds_opt_kernel_tflops_at_4096` も直列では pass |
| `--test dispatch_boundary -- --ignored --test-threads=1 small_shape_matrix_unit_has_no_floor_tflops_record` | 1 pass |
| `--test managed_placement_real_device -- --ignored --test-threads=1 managed_placement_does_not_leak_on_real_hardware` | 1 pass |

### 規則 3（tolerance／baseline 不変）

`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・`parity_baseline.rs` の
baseline は変更していない（本 PR の差分は本ディレクトリのログ・README・
#1903 README §3.1 追記・CLAUDE.md のみ）。

### #1898 CUDA 項目（別イシュー・参考）

`1898-device_param_store_backend_parity_cuda.log`: `on_cuda` 2 件 pass（イシュー #1898 コメントに記録済み）。

## 目的

`crates/backend-cuda/src/kernels_reduce.rs` の max／min 系 8 カーネル
（単位元として C マクロ `INFINITY`／`-INFINITY` を使用）が NVRTC で
コンパイルエラーになり（NVRTC は `<math.h>` を暗黙に含めないため。
`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md` §3.1）、
`CudaReduce::new` 全体が `Err` となって sum／max／min の 12 カーネル
すべてが `CudaUnavailable` を返していた不具合を、単位元を bit パターン
直接構成（`__uint_as_float(0xff800000u)`／`__uint_as_float(0x7f800000u)`）
へ置換することで是正した。本ディレクトリは是正後の GB10 実機再実測の
記録（上記「結果」）。

## 実行コマンド

```bash
# 1) reduce_parity（6 件）
cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test reduce_parity -- --ignored --nocapture \
  2>&1 | tee reduce_parity-ignored.log

# 2) typed f16／bf16 reduction parity
cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test typed_ops_f16_parity -- --ignored --nocapture \
  2>&1 | tee typed_ops_f16_parity-ignored.log
cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test typed_ops_bf16_parity -- --ignored --nocapture \
  2>&1 | tee typed_ops_bf16_parity-ignored.log

# 3) facade var_norm_backend_parity（cuda_ プレフィックスのみ）
cargo test -p fandhe-ai --release --test var_norm_backend_parity \
  -- --ignored --nocapture cuda_ \
  2>&1 | tee var_norm_backend_parity-ignored.log

# 4) Var::sum(None) を loss とする tape backward（7 件）
for t in conv1d_backend_parity conv2d_backend_parity nn_conv_backend_parity \
         attention_backend_parity no_grad_detach_backend_parity \
         backward_accumulate_backend_parity; do
  cargo test -p fandhe-ai --release --test "$t" -- --ignored --nocapture cuda_ \
    2>&1 | tee "${t}-cuda-ignored.log"
done

# 5) 全体非後退確認
make test-ignored-cuda 2>&1 | tee make-test-ignored-cuda.log
```

## 保存すべきログ

- `reduce_parity-ignored.log`
- `typed_ops_f16_parity-ignored.log`
- `typed_ops_bf16_parity-ignored.log`
- `var_norm_backend_parity-ignored.log`
- `conv1d_backend_parity-cuda-ignored.log`・`conv2d_backend_parity-cuda-ignored.log`・
  `nn_conv_backend_parity-cuda-ignored.log`・`attention_backend_parity-cuda-ignored.log`・
  `no_grad_detach_backend_parity-cuda-ignored.log`・`backward_accumulate_backend_parity-cuda-ignored.log`
- `make-test-ignored-cuda.log`
- `run.log`（(1)〜(4)・#1898 項目・直列再実行の各コマンドと `test result` 行）
- `module_cache_wiring_serial.log`（既知の並列干渉 FAIL の直列再実行）
- `parallel_interference_serial.log`（本実測で新たに FAIL した並列干渉 3 件の直列再実行）
- `1898-device_param_store_backend_parity_cuda.log`（#1898 CUDA 項目）
- `env_info.txt`（下記フォーマット。内部ホスト名は書かない）

## 事前登録判定規則

- 上記 (1)〜(4) の対象テスト 16 件すべてが pass すること。
- `make test-ignored-cuda` 相当の全体実行で、FAIL が既知 10 件
  （`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md` §3.2
  の既知分類：f16 Tensor Core K=4096 ストレス 3・TF32／3×TF32 厳密
  ゼロ fail 不成立 5・staged 対 opt 性能比較 1・
  `module_cache_wiring_tests` 並列干渉 1）を超えないこと。
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・
  `parity_baseline.rs` の baseline は一切変更しないこと（AC-4）。

## env_info.txt

実測値は `env_info.txt` を参照（`hostname: masked`・GB10・driver 580.173.02・
CUDA 13.0 V13.0.88・rustc 1.97.0・date 2026-09-16〈UTC〉・source_sha `cf062643`）。
