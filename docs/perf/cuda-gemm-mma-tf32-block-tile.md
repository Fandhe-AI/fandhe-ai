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

## 2. 実機到達性（イシュー #806 実装セッション時点）

直近セッション（#802/#804/#821 と同型）と同じく、本 worktree のローカル
環境には `ptxas`/`nvcc`（CUDA toolkit 本体）が存在せず、
`docs/real-hardware-verification-env.local.md`（gitignore 対象）も未配置
のため DGX Spark GB10 実機へ到達できなかった。したがって本ドキュメントは
**Step F フォールバック**（`docs/perf/cuda-gemm-mma-block-tile-stages.md`
§6 と同型）として、机上候補表・診断機構・ダンプ手順の整備までを成果物
とし、推定値で実測表を埋めない。

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

## 6. ダンプ手順

```sh
cargo run -p backend-cuda --example mma_tf32_ptx_dump --release \
    --features internal-diagnostics -- --out-dir /tmp/mma-tf32-ptx-dump
```

DGX Spark GB10 実機（CUDA 13.0 toolkit）へ `.ptx` ファイルを転送し
（`docs/real-hardware-verification-env.md` の手順）、標準出力が提示する
`ptxas -arch=sm_121 -v <in>.ptx -o <out>.cubin` をそのまま実行して
レジスタ使用量・spill を確認する。

## 7. 実測表（実行待ち）

| 候補 | launch_bounds | regs/thread | spill (bytes) | 512 (ms) | 1024 (ms) | 2048 (ms) | 4096 (ms) | 対 PyTorch (4096) |
|------|---------------|-------------|----------------|----------|-----------|-----------|-----------|--------------------|
| 現行（基準） | — | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 52.0%（既存記録値） |
| ステージ増のみ | なし/128 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| M 拡大 | なし/256 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| N 拡大 | なし/256 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 両拡大 | なし/512 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 両拡大+ステージ増 | なし/512 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| BK 拡大 | なし/256 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

本イシュー時点では実機到達不能のため上記は未実測のまま「実行待ち」と
明記する（推定値を記入しない。`docs/perf/cuda-parity-baseline.md` §2
検査項目に従う）。

## 8. 次に実機到達できたセッションへの引き継ぎ事項

1. §6 の手順で全候補（`launch_bounds` あり/なし各 2 通り）を `ptxas -v`
   実測し、spill 0 の候補のみ §7 表へ記録する。
2. `#[ignore]` 数値一致テスト（`tests/gemm_mma_tf32.rs`・
   `tests/mma_tf32_vs_wmma_tf32_staged.rs`）と `parity_nonregression` 系を
   実行し、既存の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5
   未満）で確認する。
3. 5 回計測中央値で 512〜4096 を計測し、4096 改善・2048 非劣化を確認
   する。
4. 合格候補があれば `MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_BK`/
   `MMA_TF32_STAGES`/`MMA_TF32_WARP_TILES_M`/`_N` 定数を更新する
   （`CudaMmaTf32Gemm` は本番非結線のため 3 段選択には触れない。opt-in
   構成が最良の場合は `gemm_mma_tf32.rs` 側の起動結線
   〈`set_attribute`・`shared_mem_bytes`・実行時予算検証〉も同 PR で
   実装する）。
5. 実測値を本ドキュメント §7・`docs/perf/cuda-parity-baseline.md` へ
   記録する。
6. TF32 mma.sync 経路の本番 3 段選択への結線・採否判断自体は #802 の
   スコープであり、本イシューでは行わない（`docs/perf/
   cuda-gemm-mma-tf32-ab.md` §2 参照）。
