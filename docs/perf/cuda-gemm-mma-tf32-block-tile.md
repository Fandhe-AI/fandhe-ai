# TF32 `mma.sync`(m16n8k8) ブロックタイル拡大候補（イシュー #806）

## 1. 位置づけ

親イシュー #479（GEMM 性能改善ツリー）→ Phase 4 親 #789 配下の #806
「perf(backend-cuda): TF32 タイル拡大（mma.sync 化後）」。TF32 生
`mma.sync` 経路（`CudaMmaTf32Gemm`。#801→PR #823）は現行ブロックタイル
64x64（BK=16・3 ステージ・静的 SMEM 28,416B）で、f16 経路（64x128）より
さらに小さく、M=N=K=4096 で対 PyTorch 比 52.0% に留まる（出典
`docs/perf/gemm-optimization-baseline.md`・`docs/perf/
cuda-gemm-bottleneck-diagnosis.md`）。f16 側のタイル拡大（#803・#804→
PR #831。`docs/perf/cuda-gemm-mma-block-tile-stages.md`）で確立した手法
（アンカー完全一致置換によるタイル変種ソース生成・静的/`extern
__shared__` 2 段予算判定・opt-in 動的 SMEM・机上候補表＋実機実測表の
分離）を TF32 経路へ展開する。

`CudaMmaTf32Gemm` は本番ディスパッチ非結線（`internal-diagnostics`
feature 限定公開。採否判断は #802 の実機引き継ぎ事項として
`docs/perf/cuda-gemm-mma-tf32-ab.md` §2 に記録済み）。本イシューのタイル
拡大もこの非結線 API／診断経路上で行い、本番 3 段選択（`gemm.rs`/
`ops.rs`/`gemm_auto.rs`）へは触れない。

## 2. 実機到達性

- **#806 実装セッション時点**: 直近セッション（#802/#804/#821 と同型）と
  同じく、本 worktree のローカル環境には `ptxas`/`nvcc`（CUDA toolkit 本体）
  が存在せず、`docs/real-hardware-verification-env.local.md`（gitignore
  対象）も未配置のため DGX Spark GB10 実機へ到達できなかった。したがって
  同セッション時点の本ドキュメントは **Step F フォールバック**
  （`docs/perf/cuda-gemm-mma-block-tile-stages.md` §6 と同型）として、机上
  候補表・診断機構・ダンプ手順の整備までを成果物とし、推定値で実測表を
  埋めなかった。
- **#841 実装セッション（2026-08-22 JST＝2026-08-21 UTC。実行時刻の検証
  可能な記録はコミット `3268d18`／`aa8f582`）**: `docs/
  real-hardware-verification-env.md` の手順に従い DGX Spark GB10 実機
  （実ホスト名は `docs/real-hardware-verification-env.local.md`〈Git
  管理外〉を参照）へ SSH 到達し、実行系（§5.1）を使って regs/spill 実測・
  数値一致テスト・5 run 計測を完了した（§7・§7.1・§8）。本ドキュメントは
  実測値で更新済み。

## 3. 現行 TF32 定数（基準）

`crates/backend-cuda/src/kernels_mma_tf32.rs`（origin/main 時点）:

| 項目 | 値 |
|------|-----|
| MMA shape | m16n8k8 |
| BM/BN/BK | 64/64/16 |
| STAGES | 3 |
| warp タイル | 2x4（実寸 32x32） |
| warps | 2x2 = 4 warp = 128 threads |
| A_PAD/B_PAD | BK+4=20／BN+4=68 |
| 静的 SMEM | `(64*20 + 16*68) * 4B * 3` = 28,416B |

## 4. 机上候補（SMEM 式 `(BM*(BK+4) + BK*(BN+4)) * 4B * STAGES`。
GB10 opt-in 実測上限 101,376B・静的上限 49,152B。出典
`docs/perf/sm121-device-attributes.md`）

| 候補 | BM/BN/BK | STAGES | warp タイル / warps (threads) | SMEM | 予算区分 |
|------|----------|--------|-------------------------------|------|----------|
| 現行（基準） | 64/64/16 | 3 | 2x4（32x32）/ 2x2 (128) | 28,416B | 静的 |
| ステージ増のみ | 64/64/16 | 4 | 同上 | 37,888B | 静的 |
| M 拡大 | 128/64/16 | 3 | 4x2（64x16）/ 2x4 (256) | 43,776B | 静的 |
| N 拡大 | 64/128/16 | 3 | 2x4（32x32）/ 2x4 (256) | 40,704B | 静的 |
| 両拡大 | 128/128/16 | 3 | 2x4（32x32）/ 4x4 (512) | 56,064B | opt-in |
| 両拡大+ステージ増 | 128/128/16 | 4 | 同上 | 74,752B | opt-in |
| BK 拡大 | 64/64/32 | 3 | 2x2（32x16）/ 2x4 (256) | 53,760B | opt-in |

各値は `crates/backend-cuda/src/kernels_mma_tf32.rs::tests` 内の
`mma_tf32_source_with_block_tile_*` 系ユニットテスト（CI 常時実行・CUDA
非搭載環境でも文字列レベルの整合を検査可能）で機械的にロックしている。

### 4.1 アキュムレータレジスタ収支の目安

`docs/perf/cuda-gemm-mma-warp-tile-register-budget.md` §3.2 の f16 版
導出式（1 warp あたりアキュムレータ regs/thread = `WARP_TILES_M *
WARP_TILES_N * 4`。m16n8 1 タイルあたり 4 f32/lane で f16/TF32 共通）を
そのまま適用する。上記候補はいずれも warp タイル実寸が 32x32〜64x16 の
範囲に収まり、アキュムレータ regs/thread は 8〜16（2x4/4x2/2x2 いずれも
8 タイル以下）で f16 版の 4x4（64x32・16 タイル）ほど大きくない。ただし
A/B フラグメントのレジスタ本数は TF32 側で異なりうる（ldmatrix.x4
b16 流用〈本ファイル参照元 `kernels_mma_tf32.rs` 冒頭コメント「命令
選定」〉のフラグメント本数は f16 と同じ 4 レジスタ/warp だが、B は
`.trans` 不使用の素の共有メモリロードで f16 版と異なる発行本数になる
点に注意）。**実機 `ptxas -v` の spill 0 実測を採用ゲートとする**（受け
入れ条件。本ドキュメントの数値は机上見積もりであり実装時に再導出して
本表を正とする）。

## 5. 診断機構

`kernels_mma_tf32.rs::mma_tf32_source_with_block_tile(bm, bn, bk, stages,
warp_tiles_m, warp_tiles_n, launch_bounds, optin_budget_bytes)`
（`kernels_mma.rs::mma_f16_source_with_block_tile`〈#804〉と同型のアンカー
完全一致置換方式を `MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_BK`/
`MMA_TF32_STAGES`/`MMA_TF32_WARP_TILES_M`/`_N`/`MMA_TF32_WARPS_N`/
`MMA_TF32_A_PAD`/`MMA_TF32_B_PAD` へ適用）を新設し、`internal-diagnostics`
feature 限定で `lib.rs::diagnostics` 経由・`examples/mma_tf32_ptx_dump.rs`
から到達可能にした。

f16 版との差分（本ファイル §3〜4 参照）:

- cp.async 転送粒度は f32 4 要素（16B）。`bm`/`bn` の倍数制約は 4（f16 版
  は 8 要素/16B のため 8）。
- `A_PAD`/`B_PAD` は `BK+4`/`BN+4`（f16 版は `BK+8`/`BN+8`。要素サイズが
  4B〈f32〉であるため）。
- SMEM 予算式の乗数は 4B/要素（f16 版は 2B）。
- `#define` 名前空間は `MMA_TF32_*` 接頭辞。

共有メモリ予算判定は 2 段階（`kernels_mma.rs::mma_f16_source_with_block_tile`
と同じ方針）:

- 静的予算（[`crate::kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES`]・48KiB）
  以下: 本番と同じ静的 `__shared__` 配列宣言のまま候補ソースを返す。
- 静的予算超・opt-in 予算以下: `as_tile`/`bs_tile` の静的宣言を
  `extern __shared__` バッファ上のポインタへ変換した候補ソースを返す
  （宣言 2 行のみの置換。多次元添字構文は本体側で無変更のまま流用）。
  **この経路は `nvrtc`/`ptxas` 実機での構文検証を経ていない**。
- opt-in 予算超: 机上除外として `CudaError::InvalidKernelConfig` を返す。

既定値 `(MMA_TF32_BM, MMA_TF32_BN, MMA_TF32_BK, MMA_TF32_STAGES,
MMA_TF32_WARP_TILES_M, MMA_TF32_WARP_TILES_N, None,
MMA_TF32_SHARED_MEM_BYTES)` は `mma_tf32_source()` とバイト一致すること
をユニットテストで固定しており、本番経路（`gemm_mma_tf32.rs`）への影響
がないことを機械的に担保する。

### 5.1 A/B ランナー（イシュー #841 で追加）

上記の候補ソース生成（`mma_tf32_source_with_block_tile`）を実際に NVRTC
コンパイル・起動して計測するための実行系を追加した
（`kernels_mma.rs::render_mma_f16_block_tile`/
`RenderedMmaF16BlockTileKernel`/`CompiledMmaF16BlockTileKernel`〈#840・
f16 版〉と同型設計）。

- `MmaTf32BlockTileLayout`（`derive_mma_tf32_block_tile_layout` が返す、
  `threads`/`a_pad`/`b_pad`/`smem_bytes`/`needs_dynamic_smem()` を含む
  レイアウト descriptor。SMEM 式・スレッド数導出式の単一の真実源）
- `RenderedMmaTf32BlockTileKernel`/`render_mma_tf32_block_tile(bm, bn,
  bk, stages, warp_tiles_m, warp_tiles_n, launch_bounds,
  optin_budget_bytes)`（候補ソース＋レイアウトを 1 個の descriptor に
  束ねる。`optin_budget_bytes` 超過時は `mma_tf32_source_with_block_tile`
  と同じ理由で `CudaError::InvalidKernelConfig` を返す）
- `RenderedMmaTf32BlockTileKernel::compile(device)` → NVRTC コンパイル・
  固定エントリポイント `"gemm_mma_tf32"` のロード・`needs_dynamic_smem()`
  時のみ `CudaFunction::set_attribute`
  （`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`）で opt-in 予算を
  設定し `CompiledMmaTf32BlockTileKernel` を返す
- `CompiledMmaTf32BlockTileKernel::launch_tf32(...)` →
  `CudaMmaTf32Gemm::launch_tf32` と同じ検証手順（`validate_gemm_dims`／
  `validate_output_len`／no-op 早期 return／
  `validate_mma_tf32_alignment`／grid y 上限検査／K タイル境界検査）＋
  `LaunchConfig.shared_mem_bytes` の設定

いずれも `internal-diagnostics` feature 限定で `lib.rs::diagnostics`
経由・`examples/gemm_mma_tf32_block_tile_bench.rs`（本イシューで追加。
§6.1）から到達可能にした。`crates/backend-cuda/src/kernels_mma_tf32.rs::
tests` に既定値一致・候補表 SMEM 実測値一致・opt-in 予算超過拒否・
`RenderedMmaTf32BlockTileKernel` のソース一致の 4 種のユニットテストを
追加済み（CI 常時実行・CUDA 非搭載環境でも成立）。

**`CudaMmaTf32Gemm` 自体は #839 で不採用（凍結）と確定済み**（本ドキュ
メント §1・`docs/perf/cuda-gemm-mma-tf32-ab.md` §2 参照）。A/B ランナー
追加・実機計測（本イシュー）はこの凍結判断を覆すものではなく、既知
correctness bug が未修正のまま実機計測を実施し結果を正直に記録する
（§7.1）。

## 6. ダンプ手順

```sh
cargo run -p backend-cuda --example mma_tf32_ptx_dump --release \
    --features internal-diagnostics -- --out-dir /tmp/mma-tf32-ptx-dump
```

DGX Spark GB10 実機（CUDA 13.0 toolkit）へ `.ptx` ファイルを転送し
（`docs/real-hardware-verification-env.md` の手順）、標準出力が提示する
`ptxas -arch=sm_121 -v <in>.ptx -o <out>.cubin` をそのまま実行して
レジスタ使用量・spill を確認する。

## 7. 実測表（DGX Spark GB10 実機・2026-08-22 JST＝2026-08-21 UTC・イシュー #841 実装セッション。実行時刻の検証可能な記録は§2・コミット `3268d18`／`aa8f582`）

`launch_bounds` は §5.1 記載どおり全候補付与なし（比較基準行と同条件）で計測。
regs/thread・spill は §6 手順で、比較基準行（`launch_bounds` なし 1 通りの
み）と 6 候補（`launch_bounds` あり/なし各 2 通り）の計 13 ソースを
`ptxas -arch=sm_121 -v` 実測した結果、**13 個全ソースで spill 0**（採用ゲート
条件を全ソースが満たす）。以下は `launch_bounds` なし版（計測に使った版）の
regs/thread を記載する。ms は `gemm_mma_tf32_block_tile_bench` 5 回プロセス
起動・候補×形状ごとの 5 run 中央値（TFLOPS）から `2*size^3/tflops` で逆算。
生ログは `/tmp/mma-tf32-block-tile-bench-841/run{1..5}.log`（DGX Spark GB10
ノード上・本 PR には含めない）。

**数値一致（正しさ）**: 全候補（比較基準行を含む）で `parity_cpu=false`
（CPU `f32::mul_add` 参照との複合判定 FAIL）。これは `CudaMmaTf32Gemm` 自体
の既知 correctness bug（#839）に起因し、タイル・ステージ変更が原因ではない
（§8 手順 2 の `#[ignore]` テストでも同一 FAIL パターンを実機確認済み。下記
「#[ignore] テスト実機実行結果」参照）。**したがって以下の TFLOPS・
対 PyTorch 列はすべて「参考値（採否判断に使用不可。#839 修正後に再計測が
必要）」である**（本ファイル冒頭コメント「`CudaMmaTf32Gemm` の既知
correctness bug」節・実装計画の運用方針を踏襲）。

| 候補 | launch_bounds | regs/thread | spill (bytes) | 512 (ms) | 1024 (ms) | 2048 (ms) | 4096 (ms) | 対 PyTorch (4096・参考値) |
|------|---------------|-------------|----------------|----------|-----------|-----------|-----------|--------------------|
| 現行（基準） | なし | 88 | 0 | 0.0323 | 0.1327 | 0.8998 | 12.1097 | 52.0%（既存記録値。基準） |
| ステージ増のみ | なし/128 | 87 | 0 | 0.0321 | 0.1590 | 1.0759 | 16.2084 | 38.9%（参考値） |
| M 拡大 | なし/256 | 93 | 0 | 0.0332 | 0.1341 | 0.8249 | 8.2293 | 76.5%（参考値） |
| N 拡大 | なし/256 | 89 | 0 | 0.0338 | 0.1339 | 0.8030 | 8.6895 | 72.5%（参考値） |
| 両拡大 | なし/512 | 87 | 0 | 0.0473 | 0.1661 | 0.9352 | 8.5342 | 73.8%（参考値） |
| 両拡大+ステージ増 | なし/512 | 80 | 0 | 0.0472 | 0.1641 | 0.9266 | 8.0094 | 78.6%（参考値） |
| BK 拡大 | なし/256 | 60 | 0 | 0.0405 | 0.2058 | 1.4465 | 20.4185 | 30.8%（参考値） |

「対 PyTorch (4096・参考値)」の導出: 基準行の 52.0%（既存記録値・出典
`docs/perf/gemm-optimization-baseline.md`）に、本セッションで実測した
「候補 TFLOPS(4096) ÷ 基準 TFLOPS(4096)」比（`gemm_mma_tf32_block_tile_bench`
出力の `ratio_vs_production_4096` 列・5 run 中央値）を乗じた値。PyTorch
参照値そのものを本セッションで再計測してはいない（`mma_tf32` 経路は本番
非結線の参考計測専用のため、`docs/perf/cuda-phase34-remeasurement.md` の
確定計測対象に含まれない。同ファイル §7 参照）。

`launch_bounds` あり版の regs/thread（参考。spill はいずれも 0）:
`bt64x64_s4_lb128`=94・`bt128x64_s3_wt4x2_lb256`=96・
`bt64x128_s3_wt2x4_lb256`=92・`bt128x128_s3_wt2x4_lb512`=90・
`bt128x128_s4_wt2x4_lb512`=92・`bt64x64x32_s3_wt2x2_lb256`=60。

### `#[ignore]` テスト実機実行結果（DGX Spark GB10・2026-08-22 JST＝2026-08-21 UTC）

`docs/perf/cuda-gemm-mma-tf32-ab.md` §3 の記録（#838・2026-08-22）と**同一
の FAIL パターン**であることを実機確認した（非後退）:

- `tests/gemm_mma_tf32.rs`（`--ignored`）: 4 本中 1 pass・3 FAIL
  （`mma_tf32_matches_reference_across_shapes` は `m=16 n=8 k=8` で
  `fail_count=128/128, max_abs_diff=3.699e0`。`mma_tf32_k4096_stress` は
  `fail_count=16768000/16777216, max_abs_diff=1.148e2`。いずれも
  `docs/perf/cuda-gemm-mma-tf32-ab.md:100-101` の記録値と完全一致）
- `tests/mma_tf32_vs_wmma_tf32_staged.rs`（`--ignored`）: 2 本中 2 FAIL
  （`mma_tf32_matches_wmma_tf32_staged_across_shapes` は `m=64 n=64 k=64` で
  `fail_count=4092/4096`。`_k4096_stress` は `fail_count=16767942/16777216`。
  `docs/perf/cuda-gemm-mma-tf32-ab.md:104-105` の記録値と完全一致）
- `tests/parity_nonregression.rs`（`--ignored`）: 1 本 pass

### 7.1 #841 実装セッションの実施範囲

- **完了**（本セッションまでの累積）: A/B ランナー（
  `kernels_mma_tf32.rs::RenderedMmaTf32BlockTileKernel`/
  `CompiledMmaTf32BlockTileKernel`/`render_mma_tf32_block_tile`）・計測
  バイナリ（`examples/gemm_mma_tf32_block_tile_bench.rs`）・CUDA 非搭載
  環境でも通るユニットテスト・全 4 種の実機実行（(1) 13 候補の
  `ptxas -arch=sm_121 -v` regs/thread・spill 実測〈全候補 spill 0〉、
  (2) `#[ignore]` 数値一致テスト 3 ファイルの実機実行〈既存 FAIL パターンと
  完全一致・非後退確認〉、(3) `gemm_mma_tf32_block_tile_bench` の 5 回
  プロセス起動・候補×形状ごとの中央値記録〈上表〉）を完了した。
- **受け入れ条件充足状況**: 実測記録（regs/spill・数値一致・5 run 中央値）
  は完了。ただし全候補が `CudaMmaTf32Gemm` の既知 correctness bug（#839）
  により数値一致 FAIL のため、TFLOPS 値は「参考値（採否判断に使用不可）」
  区分での記録に留まる（実装計画の運用方針どおり。捏造・推定値の記入は
  行っていない）。
- 採否判断・本番結線（`MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_STAGES` 等の
  更新）は #842 のスコープであり本イシューでは行わない（§8 手順 4・6）。
  #839（`CudaMmaTf32Gemm` correctness bug 修正）も別イシューのスコープ。

## 8. 実機到達後（2026-08-22 JST＝2026-08-21 UTC・イシュー #841）の実施記録・後続イシューへの引き継ぎ

0. 実行系（A/B ランナー・計測バイナリ・ユニットテスト）は #841 前半セッション
   で整備済み（§5.1）。本セッションで DGX Spark GB10 実機へ到達し、以下
   1〜3・5 を完了した（§7・§7.1 参照）。
1. **完了**: §6 の手順で比較基準行（`launch_bounds` なし 1 通りのみ）と
   6 候補（`launch_bounds` あり/なし各 2 通り）の計 13 ソースを `ptxas -v`
   実測。**全ソース spill 0**（§7 表・脚注）。
2. **完了**: `#[ignore]` 数値一致テスト（`tests/gemm_mma_tf32.rs`・
   `tests/mma_tf32_vs_wmma_tf32_staged.rs`）と `parity_nonregression` を実機
   実行。`CudaMmaTf32Gemm` 自体の既知 correctness bug（#839）により想定どお
   り FAIL（`gemm_mma_tf32.rs` 4 本中 3・`mma_tf32_vs_wmma_tf32_staged.rs`
   2 本中 2）だが、`docs/perf/cuda-gemm-mma-tf32-ab.md` §3 の既存記録値
   （fail_count・max_abs_diff 等）と完全一致する非後退を確認した
   （`parity_nonregression` は pass）。
3. **完了**: `gemm_mma_tf32_block_tile_bench` を 5 回プロセス起動し、候補×
   形状ごとに 5 run の中央値を §7 表・§7.1 へ記録した（生ログは DGX Spark
   GB10 ノード `/tmp/mma-tf32-block-tile-bench-841/run{1..5}.log`。数値一致
   FAIL のため候補値は「参考値（採否判断に使用不可）」と明記済み）。
4. **未着手（#842 のスコープ）**: `MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_BK`/
   `MMA_TF32_STAGES`/`MMA_TF32_WARP_TILES_M`/`_N` 定数の更新・
   `gemm_mma_tf32.rs` 側の opt-in 起動結線は、§7 の TFLOPS が全候補
   「参考値」区分（#839 修正待ち）である以上 #841 では判断材料が揃わない。
   #839（correctness bug 修正）→ 再計測 → #842（採否判断・本番結線）の順で
   進める必要がある。
5. **完了**: 実測値を本ドキュメント §7 へ記録した。
6. TF32 mma.sync 経路の本番 3 段選択への結線・採否判断自体は #842 の
   スコープであり、本イシューでは行わない（`docs/perf/
   cuda-gemm-mma-tf32-ab.md` §2 参照）。`CudaMmaTf32Gemm` の既知
   correctness bug の原因調査・修正は別イシュー（ユーザー承認待ち、
   `docs/perf/cuda-gemm-mma-tf32-ab.md` §6 記載）。

## 9. #842 の判断（判断材料なし・凍結維持）

**状態: 不採用（現行 `MMA_TF32_*` 定数を維持・凍結継続）を記録**。

- §7 の全候補 TFLOPS は `CudaMmaTf32Gemm` 自体の既知 correctness bug
  （#839。§7 冒頭「数値一致（正しさ）」参照）により「参考値（採否判断に
  使用不可）」区分のままであり、本イシューで新たに実測を追加しても
  この位置づけは変わらない。**判断材料（bug 修正後の再計測値）が存在
  しないため、タイル・ステージ拡大候補の採否自体を判断できる状態には
  ない**。
- `CudaMmaTf32Gemm` は #839 で凍結確定済み（`docs/cuda-tensor-core-
  design.md` §15.7）であり、本イシューはこの凍結判断を変更しない。
  `MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_BK`/`MMA_TF32_STAGES`/
  `MMA_TF32_WARP_TILES_M`/`_N`・`gemm_mma_tf32.rs` の起動結線は本
  イシューでも一切変更していない。
- **bench 診断出力拡張**: `examples/gemm_mma_tf32_block_tile_bench.rs`
  に `mismatch_diagnostics`（mismatch 件数・最大絶対/相対誤差・初回
  不一致座標を返す。`gemm_mma_block_tile_bench.rs::ParityDiagnostics`
  と同型設計）を追加し、`parity_cpu=false` 時の標準出力へ規模情報を
  追記した（#839 修正後の再計測時に FAIL パターンの回帰確認・規模比較
  に使える準備。低コストな対称性維持のための拡張であり、本判断の
  根拠には使っていない）。
- **再評価条件**（不変）: (a) `CudaMmaTf32Gemm` 自体の correctness bug
  修正、(b) 実機での数値一致 6 本 pass、(c) 修正後の本ブロックタイル
  拡大候補の再計測、の順で行う。correctness bug 修正の新規イシュー
  起票はユーザー承認必須のため本イシューでは行わず、PR 本文で提案に
  留める（`.claude/rules/out-of-scope-tracking.md`）。
