# TF32 opt-staged 段数（stages）スイープ 実測記録（#742）

イシュー #742「TF32 staged 段数スイープ example 新設 + 実機計測（stages 2..10）」の実装成果と実測記録。
`kernels_wmma_opt.rs::WMMA_TF32_STAGED_STAGES`（既定 3）の採否根拠を、段数×TFLOPS の実測表で確定させる
ことが目的。

## ステータス: 未実測・要実機実行

**本 PR 時点では DGX Spark GB10 実機への接続情報（`docs/real-hardware-verification-env.local.md`）が本
作業環境に存在せず、実機実行ができなかった。** 以下は実装したコード（example・単体テスト）とその
ローカル検証結果、および SMEM/occupancy の事前試算のみを記録し、TFLOPS 実測表は空欄のまま残す
（`docs/spec/` に対する分母分子突合の慣行〈`docs/perf/gemm-optimization-baseline.md`〉を守るため、実測
していない数値を記入しない）。`docs/real-hardware-verification-env.md` の手順（`.rev-stamp` → rsync →
SSH 実行 → `cargo run -p fandhe-ai-backend-cuda --example gemm_wmma_tf32_staged_stages_bench --release --features
internal-diagnostics` → 出力回収）に従って実機実行し、以下の表を埋めること。

## 背景・対象カーネル

- 対象: `kernels_wmma_opt.rs::WMMA_TF32_F32_STAGED_BODY`（cp.async 多段パイプライン。TF32 Tensor Core・
  `nvcuda::wmma` API）。既定タイル: `block_m=block_n=64`・`k_tile=16`・`WMMA_TF32_STAGED_STAGES=3`。
- 既定は本番経路（`gemm.rs::CudaGemm::run_wmma_tf32`／`launch_wmma_tf32` の 3 段フォールバック選択で
  staged が最優先）で使われる。
- GB10 実測 occupancy が低い（ncu 実測 16.6%）ことが分かっており、段数を増やして SMEM 常駐ステージを
  深くすることで改善余地があるかを実測で確認するのが本イシューの目的。

## 1. SMEM 制約の事前分析

既定タイル（`block_m=block_n=64`・`k_tile=16`）の SMEM 試算（`a_pad = k_tile + 4 = 20`・
`b_pad = block_n + 4 = 68`。`kernels_wmma_opt.rs:1439-1446` 実測値と一致。単体テスト
`wmma_tf32_staged_dyn_smem_bytes_matches_expected_values` で固定）:

- ステージあたり: `as_tile` 64×20×4B = 5,120B + `bs_tile` 16×68×4B = 4,352B = **9,472B/段**
- エピローグ `c_tile`: 64×64×4B = 16,384B
- 現行 static 宣言合計（stages=3）: `3*9,472 + 16,384` = 44,800B ≤ 48KiB
  （`MMA_STATIC_SMEM_LIMIT_BYTES` = 49,152B。`kernels_mma.rs:287`）

**したがって static `__shared__` のままでは stages=4 で 54,272B となり 48KiB を超え、既定タイルで
スイープ可能なのは 2..3 のみ**（`validate_wmma_tf32_staged_config` の SMEM 検査が fail-closed で拒否
する。単体テスト `validate_wmma_tf32_staged_config_accepts_default_and_rejects_smem_overflow` で
stages=8 の拒否を確認済み）。stages 2..10 帯の計測には動的共有メモリ（`extern __shared__`）+ opt-in
属性（`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`）を使う計測専用カーネル変種が必要となる。

c_tile を別領域に取ると stages=9 で `9,472×9+16,384 = 101,632B` > optin 予算 101,376B（GB10 実測。
`docs/perf/sm121-device-attributes.md` `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 101376`）となり 9..10 が
入らない。そこで **c_tile をステージバッファ先頭へエイリアス**する設計を採った
（`wmma_tf32_staged_dyn_smem_bytes` = `max(stages×9,472, 16,384)`）。stages=10 でも `94,720B ≤
101,376B` となり全帯域が計測可能になる。エイリアスの安全性は既存カーネル本体のエピローグ直前に
`cp.async.wait_group 0; __syncthreads();`（`kernels_wmma_opt.rs` 該当行）が既にあり、以降
as_tile/bs_tile はどのスレッドからも読まれないため成立する。

### 動的 SMEM 所要バイト数の期待値（単体テストで固定）

| stages | 所要バイト数（`max(stages×9,472, 16,384)`） |
|---|---|
| 2 | 18,944 |
| 3 | 28,416 |
| 4 | 37,888 |
| 5 | 47,360 |
| 6 | 56,832 |
| 7 | 66,304 |
| 8 | 75,776 |
| 9 | 85,248 |
| 10 | 94,720 |

いずれも GB10 実測 optin 予算 101,376B 以内。

## 2. 実装

- `kernels_wmma_opt.rs::WMMA_TF32_F32_STAGED_BODY` の SMEM 宣言部を `#if WMMA_TF32_STAGED_DYNAMIC_SMEM`
  プリプロセッサ分岐化（static 側は現行宣言をそのまま残す。本番経路は無変更）。
- `render_wmma_tf32_staged_dyn(cfg, optin_budget_bytes)` → `RenderedWmmaTf32StagedDynKernel` →
  `CompiledWmmaTf32StagedDynKernel`（compile 時に 48KiB 超のときのみ opt-in 属性設定、launch 時に
  `shared_mem_bytes` を指定）を追加。検証は既存 `validate_wmma_tf32_staged_config` と同項目
  （stages 範囲・warp タイル整合・cp.async 4 要素整列・スレッド数上限）+ SMEM を optin 予算に対して
  fail-closed 検査する `validate_wmma_tf32_staged_dyn_config`。static 側の検証関数は無変更。
- `device.rs::CudaDevice` に `shared_memory_per_block_optin()`・`shared_memory_per_multiprocessor()`
  アクセサを追加（`multiprocessor_count()` と同型の `Option<u32>` 返し・fail-soft 方針）。
- `lib.rs::diagnostics`（`internal-diagnostics` feature 配下）に上記の薄い re-export を追加。
- `examples/gemm_wmma_tf32_staged_stages_bench.rs`（新規。`required-features = ["internal-diagnostics"]`）:
  起動時に SM 数・SMEM optin・SMEM/SM を表示し、段数ごとの SMEM 所要・occupancy 上限概算（
  `floor(smem_per_sm / smem_bytes)` × 4 warp/block）を出力。stages 2..=10 × size ∈ {2048, 4096} を
  スイープし、各構成で先行して小形状（M=513・N=512・K=512。M を非倍数にしてエピローグ guarded store の
  境界分岐を踏ませる）の正しさ検査（統一複合判定: 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。緩和
  していない）を行ってから `bench-harness::protocol::run`（warmup/計測 20 回以上・中央値/Q1/Q3）で
  GPU 実行のみを計測。比較基準行として本番経路（static・`CudaGemm::launch_wmma_tf32`）を同条件で計測。
  optin 予算超過・コンパイル失敗・parity 不一致の構成は理由付きで SKIP/FAIL 表示し、スイープ全体は
  止めずに残りの段数の計測を継続する。

## 3. ローカル検証結果（CUDA 非搭載環境）

| ゲート | 結果 |
|---|---|
| `cargo fmt --all -- --check` | pass |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | pass（新規警告 0） |
| `cargo test --workspace --all-features` | pass（`backend-cuda` lib 394 passed。新規追加分含む） |
| `cargo build -p fandhe-ai-backend-cuda --example gemm_wmma_tf32_staged_stages_bench --features internal-diagnostics` | pass（CUDA toolkit 非搭載環境でもビルド成立。cudarc 動的ロード契約の維持確認） |
| `cargo deny check`（advisories/bans/licenses/sources） | pass（依存追加なし） |
| `scripts/check-forbidden-deps.sh lock Cargo.lock` | pass |

新規単体テスト（`kernels_wmma_opt.rs`。GPU 不要・CI 実行可能）:

- `render_wmma_tf32_staged_static_source_keeps_static_shared_declarations`: 既定（static）レンダーが
  `WMMA_TF32_STAGED_DYNAMIC_SMEM 0` を定義し、static `__shared__` 宣言テキストが変更されていないことを
  固定
- `render_wmma_tf32_staged_dyn_sets_dynamic_smem_define_and_stages` / `_is_deterministic_for_same_config`:
  dyn レンダーが `WMMA_TF32_STAGED_DYNAMIC_SMEM 1`・要求 stages・`extern __shared__` を正しく焼き込み、
  同一 config からの再現性を保つことを確認
- `wmma_tf32_staged_dyn_smem_bytes_matches_expected_values`: stages=3 → 28,416B・stages=10 → 94,720B
  の期待値固定（上記試算表と一致）
- `validate_wmma_tf32_staged_dyn_config_rejects_budget_just_below_requirement` /
  `_accepts_budget_exactly_at_requirement`: optin 予算の境界値検査
- `validate_wmma_tf32_staged_dyn_config_accepts_stages_beyond_static_smem_limit`: static 側が 48KiB
  超過で拒否する stages=8 が、dyn 側では GB10 実測 optin 予算（101,376B）内であれば受理されることを
  確認（本イシューの中核受入条件）

いずれも NVRTC（CUDA toolkit）非搭載のため、レンダー・検証・SMEM 計算の Rust 側ロジックのみを検証して
おり、実際の NVRTC コンパイル・GPU 実行・正しさ検査（parity）・性能計測は未実施（下記「未実施の検証」
参照）。

## 4. 実機計測結果（未実施）

以下は `docs/real-hardware-verification-env.md` の手順で GB10 実機実行後に埋める。

### デバイス属性

| 属性 | 値 |
|---|---|
| `MULTIPROCESSOR_COUNT` | 未実測 |
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | 未実測 |
| `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` | 未実測（`docs/perf/sm121-device-attributes.md` 実測値 101,376B と一致するはず） |

### 正しさ検査（stages 2..=10。M=513, N=512, K=512）

| stages | 結果（PASS/FAIL/SKIP・理由） |
|---|---|
| 2..10 | 未実測 |

### TFLOPS（stages × size。中央値。Q1/Q3 は example 生 CSV 出力を参照）

| stages | smem_bytes | blocks/SM 上限 | warps/SM 上限 | 2048 dyn TFLOPS | 2048 static(stages=3) TFLOPS | 2048 ratio | 4096 dyn TFLOPS | 4096 static(stages=3) TFLOPS | 4096 ratio |
|---|---|---|---|---|---|---|---|---|---|
| 2..10 | （上記試算表） | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

### `#[ignore]` 実機テストの非後退確認

| テスト | 結果 |
|---|---|
| `tests/gemm_wmma_tf32_staged.rs`（該当があれば） | 未実測 |
| `tests/parity_nonregression.rs` | 未実測 |

## 5. 採否判断（未実施・実機計測後に記入）

- **最良段数が 2..3 の場合**: `WMMA_TF32_STAGED_STAGES` 定数変更のみで既定へ反映可能。変更する場合は
  GB10 で parity 非後退（`parity_nonregression.rs` 全行 + staged 実機テスト）全 pass を確認してから
  別 PR で反映する。
- **最良段数が 4 以上の場合**: 既定経路の動的 SMEM 化（本番カーネルの構造変更・opt-in 依存の可用性
  分岐追加）が必要となり、本イシューの 4h 粒度を超えるためスコープ外とする。実測根拠を本ドキュメントに
  記録したうえでフォローアップ Issue の起票をユーザーへ提案する
  （`.claude/rules/out-of-scope-tracking.md`: 起票自体はユーザー承認が必要なため提案止まりとする）。

現時点では実機計測が未了のため、**`WMMA_TF32_STAGED_STAGES` の既定値（3）は変更していない**。

## 参照

- `crates/backend-cuda/src/kernels_wmma_opt.rs`（`WmmaTf32StagedKernelConfig`・
  `render_wmma_tf32_staged`／`_dyn`・`validate_wmma_tf32_staged_config`／`_dyn_config`・
  `wmma_tf32_staged_dyn_smem_bytes`）
- `crates/backend-cuda/src/device.rs`（`shared_memory_per_block_optin`・`shared_memory_per_multiprocessor`）
- `crates/backend-cuda/examples/gemm_wmma_tf32_staged_stages_bench.rs`
- `docs/perf/cuda-gemm-wmma-tf32-phase-b.md`（SMEM 予算試算の先行事例・既定タイル決定根拠）
- `docs/perf/sm121-device-attributes.md`（GB10 実測デバイス属性）
- `docs/perf/cuda-parity-baseline.md`（parity 非後退契約のベースライン）
- `docs/real-hardware-verification-env.md`（実機接続・転送・計測手順）
