# mma_f16 ブロックタイル拡大・ステージ数増候補の実測記録（#804）

親: GEMM 性能改善ツリー #479 → Phase 4 親 #789。イシュー #804「perf(backend-cuda):
mma_f16 ブロックタイル拡大とステージ数増」の実測記録。先行 #803（`docs/perf/
cuda-gemm-mma-warp-tile-register-budget.md`）の warp タイル拡大候補の実機実測が
「実行待ち」のまま引き継がれた状態で着手した。

## 状態: 未実測・実機実行待ち（Step F フォールバック）

本実装セッションでは以下 2 経路のいずれからも CUDA toolkit（`ptxas`/`nvrtc`
本体）へ到達できなかった:

1. **DGX Spark GB10 実機 SSH 経路**: `docs/real-hardware-verification-env.local.md`
   （実ホスト名を記す gitignore 対象ファイル）が本 worktree に存在せず、
   `CUDA_NODE` 環境変数・SSH config 上のノード alias も未設定のため到達不能
   （#803 の実測記録セッションと同一の制約。`docs/perf/
   cuda-gemm-mma-warp-tile-register-budget.md` §5 の到達試行記録参照）。
2. **ローカル toolkit 経由の静的 `ptxas` 実測経路**: `which ptxas nvcc` はいずれも
   未検出、`ldconfig -p` にも `libnvrtc`/`ptxas` 相当のエントリはなく、
   `libcuda.so.1`（driver stub）のみが存在した。

推定値で実測表を埋めず、**実装計画 Step F（実機不達時のフォールバック）**に
従い、以下のみを本イシューの成果物とする:

- 診断機構の拡張（`kernels_mma.rs::mma_f16_source_with_block_tile`。§2 参照）
- 本ドキュメントの机上候補表（§3）
- `examples/mma_ptx_dump.rs` への候補ダンプ追加

**本番カーネル定数（`MMA_BM`/`MMA_BN`/`MMA_STAGES`）・`swizzle.rs` の
`SWIZZLE_APPLY_MIN_M_BLOCKS`/`_N_BLOCKS`・`gemm_auto.rs` の予算 assert は
一切変更していない**。後続セッションが実機（または `ptxas`/`nvrtc` を持つ
ローカル環境）へ到達できた場合、以下の 2 経路のいずれかで §4 の実測表を
埋めればよい（本探索を再実施する必要はない）。

## 1. 背景

現行 `mma_f16` はブロックタイル `64x128`・`MMA_STAGES=3`・静的共有メモリ
41,472B に留まり、opt-in 動的共有メモリ容量（GB10 実測 101,376B。出典
`docs/perf/sm121-device-attributes.md`）の約 4 割にしか達していない。warp タイル
`32x16` は CUTLASS 標準 WarpShape `64x64` の 1/8 で、`ldmatrix` フラグメント
ロード比が高く smem→レジスタ帯域が律速になりうる構造上の課題を持つ
（Phase 4 診断 `docs/perf/cuda-gemm-bottleneck-diagnosis.md`）。ブロックタイル
拡大・ステージ数増でデータ再利用と Tensor Core 発行密度を上げ、M=N=K=4096 の
スループット改善を狙う。

CUTLASS 標準 ThreadblockShape `128x256x64`・`kStages=3` は、本カーネルの
パディング方式（`A_PAD=BK+8`・`B_PAD=BN+8`）込みで約 156KB となり GB10 の
opt-in 容量にも収まらないため、容量内の拡大候補から実測で選定する方針とした。

## 2. 診断機構

`crates/backend-cuda/src/kernels_mma.rs::mma_f16_source_with_block_tile(bm, bn,
bk, stages, warp_tiles_m, warp_tiles_n, launch_bounds, optin_budget_bytes)`
（診断専用。`internal-diagnostics` feature 限定で `lib.rs::diagnostics` 経由・
`examples/mma_ptx_dump.rs` から到達可能）が、`mma_f16_source_with_warp_tiles`
（#803・#822）と同じアンカー完全一致置換方式で `BM`/`BN`/`BK`/`STAGES`/
`A_PAD`/`B_PAD`/`WARPS_N`/`WARP_TILES_M`/`WARP_TILES_N` の `#define` を候補値へ
差し替え、`launch_bounds` 指定時はシグネチャへ `__launch_bounds__(v)` を付与
したソース文字列を返す。

共有メモリ予算は呼び出し元供給の `optin_budget_bytes`（`kernels_wmma_opt.rs::
validate_wmma_tf32_staged_dyn_config` と同じ「デバイス実測値を呼び出し元が渡す」
方針）に対して 2 段階で判定する:

- 静的予算（[`MMA_STATIC_SMEM_LIMIT_BYTES`]・48KiB）以下: 本番と同じ静的
  `__shared__` 配列宣言のまま候補ソースを返す。
- 静的予算超・`optin_budget_bytes` 以下: 静的 `__shared__` 配列宣言
  （`as_tile`/`bs_tile`）を `extern __shared__` バッファ上のポインタへ変換した
  候補ソースを返す。多次元添字構文（`as_tile[stage][row][col]`）はそのまま
  流用し、宣言 2 行の置換のみでインデックス計算・バンク位相設計を不変に保つ。
  **この経路は `nvrtc`/`ptxas` 実機での構文検証を一度も通過していない**（本節
  冒頭「状態」参照）。
- `optin_budget_bytes` 超: 机上除外として `CudaError::InvalidKernelConfig` を
  返す（実機到達を待たず判定できる）。

既定値（`(MMA_BM, MMA_BN, MMA_BK, MMA_STAGES, MMA_WARP_TILES_M,
MMA_WARP_TILES_N, None, MMA_SHARED_MEM_BYTES)`）を渡すと `mma_f16_source()` と
バイト一致することをユニットテスト
（`mma_f16_source_with_block_tile_default_matches_mma_f16_source`）で固定して
おり、本番経路への影響がないことを機械的に担保する。

## 3. 机上見積もり

### 3.1 候補表

`SMEM(bm,bn,bk,stages) = (bm*(bk+8) + bk*(bn+8)) * 2B * stages`。

| 候補 | 識別子（`mma_ptx_dump` ファイル名接頭辞） | BM/BN/BK | STAGES | warp タイル | warps/block | threads/block | SMEM | 予算区分 |
|------|------|----------|--------|------------|------------|---------------|------|----------|
| 現行（基準） | `mma_f16_base` | 64/128/32 | 3 | 2x2（32x16） | 2x8=16 | 512 | 41,472B | 静的（48KiB 以下） |
| ステージ増のみ | `bt64x128_s4` | 64/128/32 | 4 | 2x2（32x16） | 2x8=16 | 512 | 55,296B | opt-in（静的超・101,376B 以下） |
| タイル拡大 | `bt128x128_s3_wt2x4` | 128/128/32 | 3 | 2x4（32x32） | 4x4=16 | 512 | 56,832B | opt-in |
| タイル拡大+ | `bt128x256_s3_wt4x4` | 128/256/32 | 3 | 4x4（64x32） | 2x8=16 | 512 | 81,408B | opt-in |
| タイル拡大+ステージ増 | `bt128x256_s4` | 128/256/32 | 4 | 4x4（64x32） | 2x8=16 | 512 | 108,544B | opt-in（デバイス実測上限に対する条件付き除外。下記参照） |

`MMA_BK=32` を全候補で維持しているため `MMA_K_STEPS_PER_STAGE=2`
（`kWarpGemmIterations>=2` 相当の既存 assert）を各候補が満たす。BK=64 等の
拡大は SMEM 効率が悪化するため本イシューでは候補としない（机上比較のみ）。

`bt128x256_s4`（108,544B）は GB10 実測 opt-in 上限（101,376B）を超えるため
GB10 実機ではダンプされないが、固定除外ではない。`mma_ptx_dump` 実行時に
接続中デバイスの `optin_budget_bytes`（実測値）と比較し、`smem_bytes >
optin_budget_bytes` の場合のみ非致命的に除外する（codex-review P1 是正・
PR #831。`crates/backend-cuda/examples/mma_ptx_dump.rs` 該当コメント参照）。
GB10 より opt-in 上限が大きいデバイスでは本候補もダンプされ、8 ファイル
（4 候補 × launch_bounds なし/あり）が生成されうる。

### 3.2 occupancy 導出式（実測後に埋める）

`docs/perf/cuda-gemm-mma-warp-tile-register-budget.md` §3.2 と同じ導出式
（レジスタ制約 `floor(65536 / (regs/thread × threads/block))`・smem 制約
`floor(SM あたり smem 容量 / SMEM_BYTES)`・`warps/SM = blocks/SM ×
threads/block / 32`）を適用する。SM あたり総レジスタ数・smem 容量は
`docs/perf/sm121-device-attributes.md` の実測値を出典とする。

### 3.3 spill 判定

`ptxas -v` の `spill stores`/`spill loads` が 0 bytes かどうかを候補ごとに
記録する（#804 受け入れ条件「spill 0 維持」の分母になる）。

## 4. 実機実測結果

**実行待ち**（`docs/real-hardware-verification-env.md` の手順で DGX Spark GB10
（sm_121）実機へ到達し `mma_ptx_dump` example を実行、または `ptxas`/`nvrtc` を
持つローカル環境で同等のコンパイル・計測を行い、GB10（opt-in 上限
101,376B）では 3 候補 × launch_bounds なし/あり の 6 ファイル分（`bt128x256_s4`
は GB10 実測上限超過のため非致命的に除外されダンプされない。§3.1 参照）、
GB10 より opt-in 上限が大きいデバイスでは `bt128x256_s4` を含む 4 候補 ×
launch_bounds なし/あり の 8 ファイル分の `ptxas -v` を掛けて本節を埋める
こと。到達できない場合は推定で埋めず本記録のまま残す。`docs/perf/
cuda-gemm-mma-warp-tile-register-budget.md` §5 と同じ「実行待ち」記録方式）。

| 候補 | launch_bounds | registers/thread | spill stores (bytes) | spill loads (bytes) | blocks/SM（レジスタ制約） | blocks/SM（smem 制約） | warps/SM |
|------|---------------|-------------------|----------------------|----------------------|--------------------------|--------------------------|----------|
| ステージ増のみ | なし | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| ステージ増のみ | あり(512) | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| タイル拡大 | なし | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| タイル拡大 | あり(512) | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| タイル拡大+ | なし | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| タイル拡大+ | あり(512) | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| タイル拡大+ステージ増（opt-in 上限がより大きいデバイスのみ） | なし | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| タイル拡大+ステージ増（opt-in 上限がより大きいデバイスのみ） | あり(512) | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

## 5. 判断

実機実測が完了するまで採用構成は未確定。**動的 SMEM opt-in の起動側結線
（`CudaFunction::set_attribute`・`shared_mem_bytes`）・`extern __shared__`
変換済みソースの構文検証・本番定数変更・swizzle/`gemm_auto.rs` 追従・4096
ベンチ（5 回中央値）・parity 非後退契約の確認は、いずれも実機到達を前提とする
ため本イシューでは未実施**。判断が確定した際は `docs/cuda-tensor-core-design.md`
§16 へ記録する（`docs/perf/cuda-gemm-mma-block-tile.md` と `cuda-tensor-core-
design.md` の役割分担を踏襲）。

## 6. 引き継ぎ事項（次に実機到達できたセッションへ）

- §4 の実測表（`mma_ptx_dump` 実行 → GB10 では 6 ファイル分、GB10 より
  opt-in 上限が大きいデバイスでは `bt128x256_s4` を含む 8 ファイル分の
  `ptxas -v`）
- 採用構成（spill 0 かつ occupancy 最良の候補。spill が出る候補は除外根拠として
  その値を記録する）の決定
- 採用構成が opt-in（48KiB 超）の場合の本番起動側結線: `CudaFunction::
  set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)`・
  `LaunchConfig::shared_mem_bytes`・`device.rs::shared_memory_per_block_optin`
  による実行時予算検証・予算不足デバイスでの base 構成へのフォールバック
- `extern __shared__` 変換済みソースの実機コンパイル（構文検証）・
  `tests/gemm_mma.rs`/`cpu_cuda_mma_parity.rs`/`parity_nonregression.rs` の
  `#[ignore]` テスト実行
- `swizzle.rs::SWIZZLE_APPLY_MIN_M_BLOCKS`/`_N_BLOCKS`（`4096 / 新BM`・
  `4096 / 新BN` への再導出）・`gemm_auto.rs` の静的 SMEM 予算 assert の追従
- `gemm_mma_bench` による 512/1024/2048/4096 の 5 回中央値ベンチ・小サイズ
  非劣化確認（劣化時はサイズ条件付き選択の実装）
