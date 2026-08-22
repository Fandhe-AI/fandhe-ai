# mma_f16 ブロックタイル拡大・ステージ数増候補の実測記録（#804）

親: GEMM 性能改善ツリー #479 → Phase 4 親 #789。イシュー #804「perf(backend-cuda):
mma_f16 ブロックタイル拡大とステージ数増」の実測記録。先行 #803（`docs/perf/
cuda-gemm-mma-warp-tile-register-budget.md`）の warp タイル拡大候補の実機実測が
「実行待ち」のまま引き継がれた状態で着手した。

## 状態: 実機 A/B 実測完了（イシュー #840）。採否判断は #842 で確定（§7。**不採用**）。
`__launch_bounds__(512)` 境界値変種の実起動可否はイシュー #854 で実測済み
（§8。**実起動可能・parity FAIL のため不採用は維持**）。数値不一致の
原因はイシュー #855 で実行時観測により特定済み（§9。**`extern
__shared__` 変換にもタイル候補定数にも起因せず、production の base
カーネル自身が既に持つ既存の数値差に帰属する**。#840/#842/#854 の
「変換経路に共通の原因」という帰属は誤りだったと訂正）

#804（PR #831）は本ドキュメントを「未実測・実機実行待ち（Step F フォール
バック）」のまま残した。イシュー #840 は DGX Spark GB10（sm_121）実機へ
到達し、以下を実施した:

1. 候補を実際に NVRTC コンパイル・起動する A/B ランナー（`kernels_mma.rs::
   RenderedMmaF16BlockTileKernel`/`CompiledMmaF16BlockTileKernel`・
   `examples/gemm_mma_block_tile_bench.rs`。#742 の `RenderedWmmaTf32StagedDynKernel`
   と同型設計）を新設
2. 実機 parity ゲート（`tests/gemm_mma.rs`・`cpu_cuda_mma_parity.rs`・
   `parity_nonregression.rs`。debug/release 両プロファイル）で非後退を確認
3. A/B ランナーを 5 回プロセス起動し、候補×形状ごとの実測値を記録（§4・
   新設 §4.1「A/B 実測（5 回計測）」）
4. `mma_ptx_dump` example で候補 PTX をダンプし、ノード上 `ptxas -arch=sm_121
   -v` で registers/thread・spill を実測（§4 表を充足）

**結果を先取りすると、4 候補中 3 候補が実機で構造的な不備（数値不一致・
起動時リソース超過）を示し、残る 1 候補（`bt128x256_s4`）は机上除外のまま
だった。よって #840 時点で採用可能な候補は無い**（詳細根拠は §4・§4.1）。
**採否の最終判断・原因調査・是正・本番結線は後続 #842 のスコープ**とし、
本イシューでは実測記録に専念する。

**本番カーネル定数（`MMA_BM`/`MMA_BN`/`MMA_STAGES`）・`gemm_mma.rs` 本番
コンストラクタ・`swizzle.rs` の `SWIZZLE_APPLY_MIN_M_BLOCKS`/`_N_BLOCKS`・
`gemm_auto.rs` の予算 assert・tolerance 定数は一切変更していない**。

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
  **この経路はイシュー #840 で GB10 実機の NVRTC/ptxas 構文検証・起動検証を
  通過した**（4 候補すべてが NVRTC コンパイル・`ptxas -v` に成功。§4）。
  構文としては有効だが、`bt128x256_s3_wt4x4` は起動時リソース超過
  （§4.1「実測結果」参照）、`bt64x128_s4`/`bt128x128_s3_wt2x4` は数値一致
  fail であり、いずれも構文検証とは別の欠陥として残っている。
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

### 3.2 occupancy 導出式

`docs/perf/cuda-gemm-mma-warp-tile-register-budget.md` §3.2 と同じ導出式
（レジスタ制約 `floor(65536 / (regs/thread × threads/block))`・smem 制約
`floor(SM あたり smem 容量 / SMEM_BYTES)`・`warps/SM = blocks/SM ×
threads/block / 32`）を適用する。SM あたり総レジスタ数（`MAX_REGISTERS_
PER_MULTIPROCESSOR=65,536`）・SM あたり smem 容量（`MAX_SHARED_MEMORY_
PER_MULTIPROCESSOR=102,400 bytes`）は `docs/perf/sm121-device-attributes.md`
の実測値を出典とする。

### 3.3 spill 判定

`ptxas -v` の `spill stores`/`spill loads` が 0 bytes かどうかを候補ごとに
記録する（#804 受け入れ条件「spill 0 維持」の分母になる）。

## 4. 実機実測結果（イシュー #840・GB10・2026-08-22）

`examples/mma_ptx_dump.rs --out-dir /tmp/issue840-ptx` で候補 PTX をダンプし
（`device: optin_budget_bytes=101376`・`num_sms=48`。GB10 実測。`bt128x256_s4`
は §3.1 のとおり非致命的に desk-excluded、6 ファイルが生成された）、ノード上
`ptxas -arch=sm_121 -v` を実行して registers/thread・spill・occupancy 上限を
取得した。

| 候補 | launch_bounds | registers/thread | spill stores (bytes) | spill loads (bytes) | blocks/SM（レジスタ制約） | blocks/SM（smem 制約） | blocks/SM（採用値） | warps/SM |
|------|---------------|-------------------|----------------------|----------------------|--------------------------|--------------------------|------|----------|
| 現行（基準） `mma_f16_base` | なし | 58 | 0 | 0 | floor(65536/(58×512))=2 | floor(102400/41472)=2 | 2 | 32 |
| ステージ増のみ `bt64x128_s4` | なし | 54 | 0 | 0 | floor(65536/(54×512))=2 | floor(102400/55296)=1 | **1** | 16 |
| ステージ増のみ `bt64x128_s4` | あり(512) | 60 | 0 | 0 | floor(65536/(60×512))=2 | floor(102400/55296)=1 | **1** | 16 |
| タイル拡大 `bt128x128_s3_wt2x4` | なし | 82 | 0 | 0 | floor(65536/(82×512))=1 | floor(102400/56832)=1 | **1** | 16 |
| タイル拡大 `bt128x128_s3_wt2x4` | あり(512) | 92 | 0 | 0 | floor(65536/(92×512))=1 | floor(102400/56832)=1 | **1** | 16 |
| タイル拡大+ `bt128x256_s3_wt4x4` | なし | 130 | 0 | 0 | floor(65536/(130×512))=**0** | floor(102400/81408)=1 | **0**（起動不能） | 0 |
| タイル拡大+ `bt128x256_s3_wt4x4` | あり(512) | 128 | 0 | 0 | floor(65536/(128×512))=1（境界値 128×512=65,536） | floor(102400/81408)=1 | 1（**実起動可能と実測確認**。#854・下記 §8 参照） | 16 |
| タイル拡大+ステージ増 `bt128x256_s4` | — | — | — | — | — | — | 机上除外（108,544B > opt-in 101,376B） | — |

**spill は全候補 0（stores/loads とも）**。`bt128x256_s3_wt4x4`（`launch_bounds`
なし）は 130 registers/thread × 512 threads/block = 66,560 > `MAX_REGISTERS_
PER_MULTIPROCESSOR`（65,536）で **1 ブロック分のレジスタすら SM に収まらない**
（`blocks/SM（レジスタ制約）=0`）。これが §4.1 で観測した
`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`（"too many resources requested for
launch"）の定量的な根拠である。`__launch_bounds__(512)` 付き変種（128
registers/thread）は 128×512=65,536 とちょうど境界値に収まり `blocks/SM=1`
の計算上は起動可能だが、**本イシューの A/B 計測（§4.1）は `launch_bounds` を
付与しない構成のみを対象としたため未計測**（占有率ヒントなしでの本番同条件
比較を優先したため。実装計画「全候補 threads/block=512（launch_bounds は
付与しない）」参照）。境界値での実起動可否・実測は #842 のスコープとする。

`bt64x128_s4`／`bt128x128_s3_wt2x4` はいずれも `blocks/SM=1`（現行基準の
`2` より低い）で、たとえ数値一致 fail が解消されても **occupancy 面では現行
基準を下回る**ことが机上・実測の両方から確認できる（後述 §4.1 の性能比較は
数値不一致のため未実施だが、occupancy だけで見ても改善余地は薄い）。

## 4.1 A/B 実測（5 回計測。イシュー #840）

`cargo run -p backend-cuda --example gemm_mma_block_tile_bench --release
--features internal-diagnostics` を **5 回**プロセス起動した（計測前後の
GPU 占有状況: `nvidia-smi --query-gpu=utilization.gpu` は 5 run 通じて
`0 %`、`--query-compute-apps` は `comfyui-env`〈170MiB〉・`kokoro`〈870MiB〉の
アイドル常駐プロセスのみで、計測対象プロセスの競合なし）。

**実機 parity ゲート（性能値採用に先立ち実施。#807 契約）**: `cargo test -p
backend-cuda --features internal-diagnostics --test gemm_mma --test
cpu_cuda_mma_parity --test parity_nonregression -- --ignored
--test-threads=1` を debug/release 両プロファイルで実行。`mma_f16_
k4096_stress`（既知 fail・#389 §5.3。`fail_count=101/65536,
max_abs_diff=6.250e-2, max_rel_err=5.849e-1`）は debug/release で完全に
同一の統計値となり非後退を確認した。他の parity テスト・`parity_
baselines_do_not_regress` はいずれも pass。

**結果（5 run とも決定的に同一の結果。乱数シード `0xC0FFEE` 固定・入力競合
なしのため run 間の分岐なし）**:

| 候補 | 512 | 1024 | 2048 | 4096 | 判定 |
|------|-----|------|------|------|------|
| `mma_f16_base`（現行・比較基準） | 17.39 TFLOPS | 38.68 TFLOPS | 52.23 TFLOPS | 55.78 TFLOPS | 比較基準（5 run 中央値。ratio=1.0000） |
| `bt64x128_s4` | — | — | — | — | **FAIL**（parity mismatch vs CPU `f32::mul_add` 参照値。統一複合判定〈相対 1e-3 未満 or 絶対 1e-5 未満〉不通過。計測せず） |
| `bt128x128_s3_wt2x4` | — | — | — | — | **FAIL**（同上） |
| `bt128x256_s3_wt4x4` | — | — | — | — | **SKIP**（`CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: "too many resources requested for launch"`。§4 の register 予算超過が定量的根拠） |
| `bt128x256_s4` | — | — | — | — | 机上除外（108,544B > opt-in 101,376B。§3.1） |

`mma_f16_base` の 512/1024/2048/4096 各列は 5 run の中央値（TFLOPS。生値は
`/tmp/issue840-ab/run{1..5}.log`。512: [17.3857, 17.4581, 17.2074, 17.2417,
17.4218] → 中央値 17.3857、1024: [39.0622, 37.7653, 38.6795, 38.7129,
38.6120] → 中央値 38.6795、2048: [52.5597, 52.2858, 52.2045, 52.2299,
52.2045] → 中央値 52.2299、4096: [55.8658, 55.7754, 55.8291, 55.6349,
55.6999] → 中央値 55.7754。四捨五入して表に記載）。

**4 候補すべてが実機で不採用**（3 候補は実機不良〈数値不一致 2・起動失敗
1〉、1 候補は机上除外）だったため、**本イシュー時点で `mma_f16_base` を
上回る候補の実測は得られなかった**。

## 5. 判断

**イシュー #840 時点で採用可能な候補は無い**。4 候補すべてが実機 A/B で
不採用となった内訳:

- `bt64x128_s4`／`bt128x128_s3_wt2x4`: NVRTC コンパイル・起動には成功する
  が、カーネル出力が CPU 参照値と数値不一致（統一複合判定不通過）。
  `extern __shared__` 変換（§2「共有メモリ予算」節）が焼き込み済みの
  `as_tile`/`bs_tile` インデックス算術・バンク位相設計と整合しているかは
  §4.1 の FAIL のみでは特定できておらず、**原因調査は #842 のスコープ**
  とする
- `bt128x256_s3_wt4x4`（`launch_bounds` なし）: 130 registers/thread ×
  512 threads/block が per-SM レジスタ上限（65,536）を超え、1 ブロックも
  起動できない（§4）。`__launch_bounds__(512)` 付き変種は境界値
  （65,536 ちょうど）で理論上 1 block/SM に収まるが本イシューでは未計測
  （§4 参照）
- `bt128x256_s4`: 机上見積もり（108,544B）が GB10 実測 opt-in 上限
  （101,376B）を超え、実行時に非致命的除外（§3.1）

**動的 SMEM opt-in の起動側結線自体（`CudaFunction::set_attribute`・
`shared_mem_bytes`）は本イシューで実装・実機動作確認済み**（§4.1 の
`bt64x128_s4`/`bt128x128_s3_wt2x4`/`bt128x256_s3_wt4x4` はいずれも
opt-in 属性設定を経て起動を試み、実際にカーネルが実行された〈parity
mismatch は計算結果の不一致であり起動自体は成功している〉ため、結線
機構そのものは機能を確認済みである）。**本番定数変更・swizzle/
`gemm_auto.rs` 追従・採否判断は、#842 で数値不一致の原因調査・是正が
完了してから行う**。判断が確定した際は `docs/cuda-tensor-core-design.md`
§16 へ記録する（`docs/perf/cuda-gemm-mma-block-tile.md` と `cuda-tensor-core-
design.md` の役割分担を踏襲）。

## 6. 引き継ぎ事項（#842 へ）

- `bt64x128_s4`／`bt128x128_s3_wt2x4` の数値不一致原因調査（`extern
  __shared__` 変換〈§2〉のインデックス算術・バンク位相・cp.async 転送
  先アドレスのいずれかに実装齟齬がある可能性が高い。§4.1 の FAIL 詳細は
  ミスマッチ件数までは出力していないため、まず `within_tolerance` 判定を
  ミスマッチ件数・最大誤差付きで出力するよう `gemm_mma_block_tile_bench.rs`
  を拡張し、再現・切り分けを行うことを推奨する）
- `bt128x256_s3_wt4x4` の `__launch_bounds__(512)` 付き変種（128
  registers/thread。§4 参照）の実起動可否の実測（境界値 65,536 での
  実際の occupancy・spill・レジスタ再割当ての有無を確認する）→ **#854 で
  実測済み（起動成功・parity FAIL。下記 §8 参照）**
- 上記原因調査の結果、いずれかの候補が数値一致・起動成功に至った場合の
  再計測（512/1024/2048/4096・5 回中央値）・`mma_f16_base` との比較
- 採用構成が確定した場合の本番起動側結線: `CudaFunction::
  set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ...)`・
  `LaunchConfig::shared_mem_bytes`・`device.rs::shared_memory_per_block_optin`
  による実行時予算検証・予算不足デバイスでの base 構成へのフォールバック
  （opt-in 属性設定自体は #840 の A/B ランナーで実機動作確認済み。上記
  「判断」節参照）
- `swizzle.rs::SWIZZLE_APPLY_MIN_M_BLOCKS`/`_N_BLOCKS`（`4096 / 新BM`・
  `4096 / 新BN` への再導出）・`gemm_auto.rs` の静的 SMEM 予算 assert の追従
- `gemm_mma_bench` による 512/1024/2048/4096 の 5 回中央値ベンチ・小サイズ
  非劣化確認（劣化時はサイズ条件付き選択の実装）

## 7. #842 実装セッションの結果（採否判断の確定・実機再到達不可）

**状態: 不採用（現行 `mma_f16_base` 定数を維持）を確定。#789 は本判断を
もってスコープ再定義・完了クローズとする**（詳細は
`docs/cuda-tensor-core-design.md` §16・#789 クローズコメント参照）。

- **実機到達性**: 本セッションも本 worktree に `ptxas`/`nvcc`（CUDA
  toolkit 本体）が存在せず、`docs/real-hardware-verification-env.local.md`
  も未配置のため DGX Spark GB10 実機へ到達できなかった（§6 の
  「原因調査」「`__launch_bounds__(512)` 境界値実測」「再計測」は実行
  できていない）。よって本セッションの成果は §6 の 3 項目のうち
  bench 診断出力拡張（下記）のみに留まる。
- **bench 診断出力拡張（実施済み・CI 検証可能）**:
  `crates/backend-cuda/examples/gemm_mma_block_tile_bench.rs::
  candidate_parity_ok` の戻り値を bool から `ParityDiagnostics`
  （mismatch 件数・最大絶対誤差・最大相対誤差・初回不一致座標）へ拡張
  した。FAIL 時の標準出力に `mismatch_count=.../..., max_abs_diff=...,
  max_rel_err=..., first_mismatch=(row=.., col=..)` を追加出力する
  （実機再到達時に `bt64x128_s4`／`bt128x128_s3_wt2x4` の不一致規模を
  即座に把握できるようにする準備）。対称性維持のため
  `gemm_mma_tf32_block_tile_bench.rs` にも同型の `mismatch_diagnostics`
  関数を追加した（TF32 側は #839 の既知 correctness bug により恒常的に
  大量 mismatch となるが、bug 修正後の回帰確認・規模比較に使える）。
- **数値不一致の原因机上調査（実施・結論に至らず）**: `kernels_mma.rs`
  の `DYNAMIC_SMEM_REPLACEMENT`（`extern __shared__` 変換。2518 行付近）
  について、静的宣言 `as_tile[STAGES][BM][A_PAD]`/`bs_tile[STAGES][BK]
  [B_PAD]` と、`typedef` 配列型ポインタ経由の動的宣言との等価性を
  以下の観点で机上検証したが、いずれも整合しており明確な欠陥を特定
  できなかった:
  - `as_tile[stage][row][col]` の多次元添字は、`MmaAsTileT* as_tile`
    （`MmaAsTileT = __half[BM][A_PAD]`）でも静的宣言と同じアドレス
    計算式（`stage` オフセット = `sizeof(MmaAsTileT)`）になる
  - `bs_tile` の開始オフセット（`sizeof(MmaAsTileT) * STAGES`）は
    `as_tile` 全体（`STAGES` 段分）の直後であり、`smem_bytes` の
    机上見積もり式（§3.1 `SMEM(bm,bn,bk,stages)` 式）とバイト単位で
    一致する（手動バンプアロケータとして自己整合的）
  - 全候補（`BM`/`A_PAD`/`STAGES` の組み合わせ）で `sizeof(MmaAsTileT)
    * STAGES` は 16 の倍数（`bt64x128_s4`: 5,120*4=20,480、
    `bt128x128_s3_wt2x4`: 10,240*3=30,720 等）であり、`bs_tile` の
    16 バイトアライメントは `mma_dyn_smem` バッファの `__align__(16)`
    起点から保たれる（`mma_cp_async16`・`ldmatrix` の 16B 整列要件を
    崩さない）
  - `LOAD_A_STAGE_GROUP`/`LOAD_B_STAGE_GROUP`（cp.async 発行）・
    `LDSM_A_FRAG`/`LDSM_B_FRAG`（ldmatrix 発行）はいずれも
    `__cvta_generic_to_shared` でランタイムのポインタ値からシェアード
    アドレスへ変換しており、コンパイル時の型属性（`__align__(16)` が
    `MmaAsTileT`/`MmaBsTileT` 型自体には付与されていない点）に依存する
    箇所は見当たらなかった
  - `extern __shared__ unsigned char[]` を型付きポインタへ
    `reinterpret_cast` する手法自体は CUDA の動的共有メモリ確保の標準
    パターンであり、strict aliasing 由来の未定義動作は本件の原因と
    考えにくい
  - **結論**: 机上調査だけでは原因を特定できなかった。実機での
    `compute-sanitizer`（memcheck）・中間 SMEM ダンプ等、実行時観測を
    伴う切り分けが必要（次に実機到達できたセッションへ引き継ぐ）
- **`__launch_bounds__(512)` 付き `bt128x256_s3_wt4x4` 変種の実起動確認**:
  実機到達不可のため本セッションでは未実施（§6 記載のまま引き継ぎ）→
  **#854 で実測済み（下記 §8 参照）**
- **本番カーネル定数（`MMA_BM`/`MMA_BN`/`MMA_STAGES`）・`gemm_mma.rs`
  本番コンストラクタ・`swizzle.rs`・`gemm_auto.rs`・tolerance 定数は
  本セッションでも一切変更していない**（不採用判断に伴い変更不要）。

### 7.1 次セッション（後続イシュー）への引き継ぎ事項

本節は #842 実装セッション自身の内部での気づきであり、#842 自身へ
「引き継ぐ」ものではない（レビュー指摘。#842 実装計画時点で後続イシュー
番号は未採番のため、番号確定時に本節へ追記する）。後続イシュー番号:
**#854**（`__launch_bounds__(512)` 変種の実起動可否実測。実施済み。下記
§8）・**#855**（`extern __shared__` 変換による数値不一致の実行時観測
での原因特定。**実施済み・§9 参照。結論: 原因は変換にもタイル候補定数
にもなく、production の base カーネル自身が既に持つ既存の数値差に
帰属することを確認した**）。§6 の未消化項目（原因調査の実機切り分け・
`__launch_bounds__(512)` 変種実測・再計測・採用時の本番結線手順）の
うち `__launch_bounds__(512)` 変種実測は #854 で消化済み（下記 §8）。
不一致原因の実機切り分けは #855 で消化済み（下記 §9）。加えて:

- 拡張済みの bench 診断出力（mismatch 件数・最大誤差・初回不一致座標）
  を使い、`bt64x128_s4`／`bt128x128_s3_wt2x4` の不一致がタイル境界
  付近（境界検査ロジック関連）か、タイル内部全域（アドレス計算・
  バンク位相関連）かをまず切り分けることを推奨する
- `compute-sanitizer --tool memcheck`／`--tool racecheck` による
  `extern __shared__` 変換経路の実行時検証（境界外アクセス・
  レース検出）

## 8. #854 実測（`__launch_bounds__(512)` 変種の実起動可否。GB10・2026-08-22）

**状態: 実起動可能と確定（起動失敗ではない）。ただし parity FAIL のため
不採用は維持**。#807 契約（性能値採用に先立ち parity ゲートを通す）に
より TFLOPS 実測は行っていない（bench 自体が parity FAIL 時に測定を
スキップする設計。§2「候補が opt-in 予算内かを実測レイアウトから判定
する」に続く数値一致検査ステップ）。

### 8.1 実行環境・手順

`crates/backend-cuda/examples/gemm_mma_block_tile_bench.rs`
（`Candidate::launch_bounds: Option<u32>` を追加し、5 番目の候補
`bt128x256_s3_wt4x4_lb512`〈bm=128, bn=256, bk=32, stages=3, warp タイル
4x4, `launch_bounds: Some(512)`〉を追加。CSV へ `launch_bounds` 列を
追加）を DGX Spark GB10 実機（`optin_budget_bytes=101376` 実測）で
release ビルド・**5 回プロセス起動**した。

- **実機 parity ゲート（#807 契約。性能値採用に先立ち実施）**:
  `cargo test -p backend-cuda --features internal-diagnostics --test
  gemm_mma --test cpu_cuda_mma_parity --test parity_nonregression
  --no-fail-fast -- --ignored --test-threads=1` を debug/release 両
  プロファイルで実行。既知 fail `mma_f16_k4096_stress`
  （`fail_count=101/65536, max_abs_diff=6.250e-2, max_rel_err=5.849e-1`）
  は debug/release で完全に同一の統計値となり非後退を確認した（#840・
  §4.1 の記録値と一致）。他の parity テスト・`parity_baselines_do_not_
  regress` はいずれも pass
- **GPU 占有状況**（5 run 前後で `nvidia-smi --query-gpu=utilization.gpu`
  を確認）: `0 %`。`--query-compute-apps` は常駐プロセス（`comfyui-env`
  〈170MiB〉・`kokoro`〈870MiB〉）のみで計測対象プロセスとの競合なし
  （#840・§4.1 と同一プロトコル・同一常駐構成）

### 8.2 実測結果（5 run とも決定的に同一。乱数シード `0xC0FFEE` 固定）

```
bt128x256_s3_wt4x4_lb512: FAIL (parity mismatch vs CPU f32::mul_add reference; not measuring;
  mismatch_count=2/266240, max_abs_diff=1.562e-2, max_rel_err=6.818e-2,
  first_mismatch=(row=168, col=2))
```

比較のため同一 run 内の他候補も記載する（5 run 通じて全項目が完全に
同一）:

| 候補 | launch_bounds | 起動 | parity | 詳細 |
|------|---------------|------|--------|------|
| `bt64x128_s4` | なし | 成功 | **FAIL** | mismatch_count=2/266240, max_abs_diff=1.562e-2, max_rel_err=6.818e-2, first_mismatch=(row=168, col=2) |
| `bt128x128_s3_wt2x4` | なし | 成功 | **FAIL** | 同上（値まで完全一致） |
| `bt128x256_s3_wt4x4` | なし | **失敗** | — | `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES`（"too many resources requested for launch"）。§4 の 130 regs/thread×512=66,560>65,536 が定量的根拠（#840 と同一・非後退） |
| `bt128x256_s4` | — | — | — | 机上除外（108,544B > opt-in 101,376B。§3.1。非後退） |
| **`bt128x256_s3_wt4x4_lb512`（#854 追加）** | **あり(512)** | **成功** | **FAIL** | **mismatch_count=2/266240, max_abs_diff=1.562e-2, max_rel_err=6.818e-2, first_mismatch=(row=168, col=2)（`bt64x128_s4`／`bt128x128_s3_wt2x4` と完全一致）** |
| `mma_f16_base`（現行・比較基準） | なし | 成功 | pass | — |

**`bt128x256_s3_wt4x4_lb512` は `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` を
起こさず実際に起動・実行された**（§4 の机上見積もり「128 regs/thread ×
512 threads = 65,536 の境界値ちょうどで 1 block/SM に収まる」が実機で
裏付けられた。`launch_bounds` なしの兄弟候補 `bt128x256_s3_wt4x4`
〈130 regs/thread〉との対比で、`__launch_bounds__(512)` のレジスタ
削減ヒントが起動可否の分水嶺になっていることを実機実測で確認）。

一方で **parity は FAIL**（統一複合判定〈相対誤差 1e-3 未満 または
絶対誤差 1e-5 未満〉不通過）。`mismatch_count`／`max_abs_diff`／
`max_rel_err`／`first_mismatch` の 4 項目すべてが `bt64x128_s4`・
`bt128x128_s3_wt2x4`（いずれも §7 で原因未特定のまま #855 へ引き継いだ
`extern __shared__` 変換に起因する数値不一致候補）と**完全に一致**して
いる。3 候補はいずれも `MMA_BK=32`・`extern __shared__` 変換経路
（§2）を通る点が共通のため、`bt128x256_s3_wt4x4_lb512` の不一致も
同一の原因（`extern __shared__` 変換のインデックス算術・バンク位相・
cp.async 転送先アドレスのいずれか）を共有している可能性が高いが、
**本イシューのスコープ外**（原因調査は #855 に帰属。イシュー本文
「スコープ境界」参照）。

**#807 契約（parity ゲートが性能値採用に先立つ）により、本候補の
512/1024/2048/4096 TFLOPS は計測していない**（parity FAIL のため bench
が測定をスキップする設計どおり。§8.1）。

### 8.3 判断

**#847 で確定した不採用判断（現行 `MMA_BM=64`/`MMA_BN=128`/
`MMA_STAGES=3` を維持）は変わらず維持する**。§6・§7 で「未計測」
だった `__launch_bounds__(512)` 変種の実起動可否は「起動可能」と実測
確定したが、parity FAIL のため採用可能な候補が新たに得られたわけでは
ない。よって #789／#842／#847 の不採用判断・スコープはそのまま有効。

**本イシューで確定した事実**:

1. `bt128x256_s3_wt4x4` の `launch_bounds` なし版が
   `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` で起動不能（§4・#840 の
   非後退再確認）であるのに対し、`__launch_bounds__(512)` 付き版は
   実機で正常に起動する（境界値 65,536 での机上見積もりが実機で
   裏付けられた）
2. しかし `__launch_bounds__(512)` 付き版も `extern __shared__`
   変換経路を通るため、`bt64x128_s4`／`bt128x128_s3_wt2x4` と同一の
   数値不一致（4 項目完全一致）を抱えている
3. 本番カーネル定数（`MMA_BM`/`MMA_BN`/`MMA_STAGES`）・
   `gemm_mma.rs` 本番コンストラクタ・`swizzle.rs`・`gemm_auto.rs`・
   tolerance 定数は本イシューでも一切変更していない

**後続への引き継ぎ**: #855（`extern __shared__` 変換の数値不一致の
実行時観測での原因特定）の対象候補に `bt128x256_s3_wt4x4_lb512`
（本イシューで追加）を加えることを推奨する。3 候補（`bt64x128_s4`・
`bt128x128_s3_wt2x4`・`bt128x256_s3_wt4x4_lb512`）が完全に同一の
mismatch 統計値を示している事実は、原因が個々のタイル形状固有の
バグではなく `extern __shared__` 変換経路（§2）そのものに共通して
存在することを強く示唆する追加根拠であり、#855 の原因調査の優先度・
切り分け範囲の判断材料になる。

## 9. #855 実行時観測の結果（原因特定・GB10・2026-08-22）

**状態: 原因特定済み。修正なし（修正対象がない）。#840/#842/#854 の
「不一致は `extern __shared__` 変換経路に起因する」という帰属は誤り
だったと訂正する**。

### 9.1 実施内容

DGX Spark GB10 実機（`docs/real-hardware-verification-env.md` 記載の
node4。CUDA 13.0 + NVRTC・`compute-sanitizer`〈CUDA toolkit 同梱〉導入
済み）で、以下を実施した:

1. `kernels_mma.rs` へ診断専用の強制動的 SMEM 変換フラグ
   （`mma_f16_source_with_block_tile_forced_dynamic_smem`／
   `render_mma_f16_block_tile_forced_dynamic_smem`。静的予算以下の候補
   でも `extern __shared__` 変換を強制適用する）を追加し、production
   と完全同一パラメータ（BM=64/BN=128/BK=32/STAGES=3/warp2x2）を
   強制動的化した対照実験行 `mma_f16_base_dynsmem` を
   `gemm_mma_block_tile_bench.rs::CONTROL_CANDIDATES` に追加した。
2. `bt64x128_s4`（STAGES=4 のみ差）・`bt128x128_s3_wt2x4`（BM=128・
   warp2x4 のみ差）それぞれから BN を 64 へ縮小し、**静的予算内
   （`extern __shared__` 変換を経ない）で同じパラメータ差分を単独検証
   する**対照候補 `bt64x64_s4_static`（38,912B）・
   `bt128x64_s3_wt2x4_static`（44,544B）を追加した。
3. **決定打となった対照実験**: production と完全同一パラメータを
   **非強制**（source が `mma_f16_source()` とバイト一致・静的
   `__shared__` のまま）で診断専用のコンパイル・起動経路
   （`RenderedMmaF16BlockTileKernel::compile`／
   `CompiledMmaF16BlockTileKernel::launch_f16`。production の
   `CudaMmaGemm::launch_f16` とはコード上別経路）のみを通す対照候補
   `debug_default_via_diagnostics_path` を追加した。
4. `gemm_mma_block_tile_bench.rs` に production 直接検査行
   （`CudaMmaGemm::launch_f16` を診断ハーネスを一切経由せず直接呼び、
   同じ `CORRECTNESS_M/N/K`・同じ `a_ref`/`b_ref`・同じ `expected_f32`
   で parity 検査する。標準出力の `production_direct(no diagnostics
   path):` 行）を追加した。
5. `compute-sanitizer --tool memcheck ./gemm_mma_block_tile_bench` を
   実行し、全候補（`extern __shared__` 変種を含む）のメモリ安全性を
   検証した。

### 9.2 観測結果

全候補・対照行が**完全に同一**の parity 統計値で fail した:

| 行 | 種別 | `extern __shared__` 変換 | parity |
|----|------|---------------------------|--------|
| `mma_f16_base`（production・別コード経路） | 比較基準 | — | pass（本 bench は TFLOPS のみ計測・parity 未検査） |
| `production_direct(no diagnostics path)` | production 直接検査（本イシューで追加） | — | **FAIL**（`mismatch_count=2/266240, max_abs_diff=1.562e-2, max_rel_err=6.818e-2, first_mismatch=(168, 2)`） |
| `debug_default_via_diagnostics_path` | production と同一パラメータ・診断経路のみ差（本イシューで追加） | 非適用（静的のまま） | **FAIL**（上記と完全一致） |
| `mma_f16_base_dynsmem` | production と同一パラメータ・強制動的化（本イシューで追加） | 適用（強制） | **FAIL**（上記と完全一致） |
| `bt64x64_s4_static` | STAGES=4 単独・静的予算内（本イシューで追加） | 非適用（静的） | **FAIL**（上記と完全一致） |
| `bt128x64_s3_wt2x4_static` | BM=128/warp2x4 単独・静的予算内（本イシューで追加） | 非適用（静的） | **FAIL**（上記と完全一致） |
| `bt64x128_s4` | #840 既存候補 | 適用 | FAIL（既知。上記と完全一致） |
| `bt128x128_s3_wt2x4` | #840 既存候補 | 適用 | FAIL（既知。上記と完全一致） |
| `bt128x256_s3_wt4x4_lb512` | #854 既存候補 | 適用 | FAIL（既知。上記と完全一致） |

`compute-sanitizer --tool memcheck` は**全候補でメモリ安全性エラーを
検出しなかった**（境界外アクセス・未初期化読み出しのいずれもゼロ件）。
唯一の報告は `bt128x256_s3_wt4x4`（`launch_bounds` なし。#840/#854 で
既知のレジスタ不足）の `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` であり、
これはメモリ破壊ではなく起動不能（1 block/SM に収まらない）という
別事象である。

### 9.3 結論

**`debug_default_via_diagnostics_path`（production とバイト一致の
ソース・静的 `__shared__` のまま・診断コンパイル・起動経路のみ経由）が
`production_direct`（production 自身の `CudaMmaGemm::launch_f16`）と
全く同一の座標・同一値で fail したことが決定的な証拠である**。両者の
違いはコンパイル・起動コードパスのみであり、カーネルソースは完全に
同一（`mma_f16_source_with_block_tile_default_matches_mma_f16_source`
テストが保証）。したがって:

1. **`extern __shared__` 動的 SMEM 変換（`DYNAMIC_SMEM_REPLACEMENT`）は
   無罪である**。強制動的化（`mma_f16_base_dynsmem`）・非強制（静的の
   まま）のいずれでも同一の不一致が生じるため、変換の適用有無は結果に
   影響しない。
2. **候補のタイル定数（BM/BN/STAGES/warp タイル）も無罪である**。
   静的予算内に収まるよう縮小した対照候補（`bt64x64_s4_static`・
   `bt128x64_s3_wt2x4_static`。いずれも `extern __shared__` 変換を
   経ない）も同一の不一致を示すため、STAGES=4・BM=128・warp2x4 いずれの
   タイル変更も原因ではない。
3. **診断ハーネスの compile/launch コードパス自体も無罪である**。
   production 自身の `CudaMmaGemm::launch_f16`（診断ハーネスを一切
   経由しない）が同一の不一致を出すため、`RenderedMmaF16BlockTileKernel`
   ／`CompiledMmaF16BlockTileKernel` の実装に固有の欠陥でもない。
4. **真の原因**: production の base カーネル自身が、本 bench の正しさ
   検査データ（`CORRECTNESS_M=520` 非整列端・`CORRECTNESS_N=512`・
   `CORRECTNESS_K=512`・`SEED=0xC0FFEE`）に対して**既に**この 2 要素の
   数値差（`f16` 丸め後で相対誤差 6.818e-2・絶対誤差 1.562e-2 と、
   統一複合判定の閾値〈相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満〉を
   大きく超える）を持っている。#840/#842/#854 はこのデータ点で base
   カーネル自身がどうなるかを一度も検査しないまま、動的 SMEM 変換を
   通る候補群だけが fail する状況から「変換に共通の原因がある」と
   推定したため、誤って帰属した（本 bench が比較基準行 `mma_f16_base`
   に対して TFLOPS のみを計測し parity を検査してこなかったことが
   誤帰属の直接要因。§9.1 の `production_direct` 検査行はこの欠落を
   埋める恒久追加である）。

### 9.4 スコープ外事項（#855 のスコープ外）

**本イシュー（#855）は「`extern __shared__` 変換による数値不一致」の
原因調査であり、原因が変換にないと判明した以上、`mismatch_count=
2/266240` 自体の是正は本イシューのスコープ外である。** 加えて、
既存の `#[ignore]` テスト `crates/backend-cuda/tests/
cpu_cuda_mma_parity.rs::mma_f16_k4096_stress`（M=N=256, K=4096.
`CudaMmaGemm` 経由）が本パッチと**無関係に**（`main` ブランチ単体でも
再現。2026-08-22 GB10 実機で確認）
`fail_count=101/65536, max_abs_diff=6.250e-2, max_rel_err=5.849e-1` で
fail することを確認した。一方、同ファイルの `#[ignore]` なしスモーク・
`mma_f16_matches_reference_across_shapes`（12 形状の境界網羅テスト）は
pass する。

以上から、**production の base `mma.sync` カーネルには、特定のデータ・
形状の組み合わせ（K=4096 の大深度蓄積、または本イシューで見つかった
M=520/`SEED=0xC0FFEE` の組み合わせ）でのみ顕在化する、既存の・
`extern __shared__` 変換とは無関係な数値不一致が存在する**可能性が
高い。これは #804/#840/#842/#854/#855 のいずれのスコープにも含まれない
別課題であり、ガードレール・テスト許容誤差の緩和はユーザー承認必須
（`.claude/rules/security.md`・`coding-rust.md`）のため本セッションでは
一切変更していない。追跡 issue の起票要否は本イシューの成果物（PR）
本文でユーザーへ提起する（`.claude/rules/out-of-scope-tracking.md`）。

### 9.5 本イシューで変更したコード

- `crates/backend-cuda/src/kernels_mma.rs`: 診断専用の強制動的 SMEM
  変換フラグ（`mma_f16_source_with_block_tile_impl`／
  `render_mma_f16_block_tile_impl` の `force_dynamic_smem: bool`。
  既存 8 引数の公開シグネチャは無変更のまま
  `mma_f16_source_with_block_tile_forced_dynamic_smem`／
  `render_mma_f16_block_tile_forced_dynamic_smem` を追加）・
  `RenderedMmaF16BlockTileKernel`/`CompiledMmaF16BlockTileKernel` へ
  `uses_dynamic_smem: bool` フィールドを追加（`layout.
  needs_dynamic_smem()` ではなく本フィールドで起動側 opt-in 設定・
  `shared_mem_bytes` を決定する契約）。ソース検査テスト
  （`mma_f16_source_with_block_tile_forced_dynamic_smem_applies_
  transform_below_static_budget`・`render_mma_f16_block_tile_forced_
  dynamic_smem_marks_uses_dynamic_smem`）を追加。
- `crates/backend-cuda/src/lib.rs`: `diagnostics` モジュールへ
  `render_mma_f16_block_tile_forced_dynamic_smem` を再公開。
- `crates/backend-cuda/examples/gemm_mma_block_tile_bench.rs`:
  `Candidate::force_dynamic_smem` フィールド追加・`CONTROL_CANDIDATES`
  （§9.1 の 4 対照行）追加・production 直接検査行（§9.1）追加。
- **本番カーネル定数（`MMA_BM`/`MMA_BN`/`MMA_STAGES`）・
  `gemm_mma.rs` 本番コンストラクタ・`swizzle.rs`・`gemm_auto.rs`・
  tolerance 定数は本イシューでも一切変更していない**。
