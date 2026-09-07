# CUDA GEMM 3×TF32（split-single 法）の設計判断

イシュー #1355（親ツリー #1354・承認元 #1338。ユーザー承認 2026-09-06
「3×TF32 モード（FMA 契約の例外。既定 OFF・opt-in）は採用」）。

## 1. 背景・目的

単発 TF32（`crate::precision::CudaGemmPrecision::Tf32`。#1042）は `mma.sync`
オペランドを仮数 10bit へ丸めるため、CPU f32 参照実装に対し形状によって
非ゼロ fail が残る（`docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md`
等）。3×TF32（split-single 法。A・B オペランドを hi/lo の 2 語（各々 TF32
丸め済み）へ分割し、`mma.sync` を 3 回発行して累積する手法。CUTLASS
`mma_tensor_op_fast_f32`（`include/cutlass/gemm/warp/mma_tensor_op_fast_f32.h`）
と同型）は、Tensor Core 命令を経由しながら f32 相当の実効精度を得る既知の
手法である。

本イシューは**カーネル実装と opt-in 配線（既定 OFF・fail-closed）まで**を
スコープとする。誤差分布・純カーネル時間の GB10 実機実測と採否判断は #1356
が引き継ぐ。

## 2. 分割式（hi/lo 分割・レジスタ段）

```c
float v  = __uint_as_float(raw);
float hi = wmma::__float_to_tf32(v);
float lo = wmma::__float_to_tf32(v - hi);
```

- `hi` は `v` の TF32 丸め値（仮数上位 10bit）。
- `lo` は `v - hi`（f32 減算で計算する残差）を再度 TF32 丸めした値。
  `v - hi` は f32 の減算命令 1 回で計算するため厳密な数学的差ではなく
  f32 丸め誤差を含むが、`lo` 自体が誤差の補正項であるため実用上十分な
  精度が得られる（CUTLASS の同手法と同じ近似）。
- `hi`・`lo` はいずれも TF32 丸め済みの値のため、`mma.sync` の tf32
  オペランド（32bit レジスタに TF32 精度の値をそのまま格納する契約。
  `kernels_mma_tf32.rs` 冒頭コメント「TF32 丸め」参照）としてそのまま
  使える。

**分割位置（レジスタ段。smem ステージング時ではない）**: 単発 TF32 経路
（`kernels_mma_tf32.rs`）は `cp.async` で smem へ到着した直後に
`CONVERT_A_STAGE_GROUP`/`CONVERT_B_STAGE_GROUP` で smem 上の値を in-place
TF32 丸めする。3×TF32 では `lo` 成分（丸め誤差の補正項）が必要なため、
smem 上で hi へ丸めてしまうと `lo` を復元できない。よって 3×TF32 では
**smem には生の f32 をそのまま保持し、フラグメントロード直後（レジスタ段）
で hi/lo へ分割する**。

- A オペランド: `kernels_mma_tf32.rs::LDSM_A_FRAG` と同一の `ldmatrix.x4`
  b16 流用アドレッシングでレジスタへ運ぶ。ldmatrix はビットパターンを
  転置なしで再配置するのみの命令であるため、smem に生の f32 を置いても
  正しく元のビット列（`raw`）をレジスタへ運べる。運ばれた `raw` を直後に
  `__uint_as_float` → 分割する。
- B オペランド: `kernels_mma_tf32.rs::LDS_B_FRAG` と同一の直接 smem ロード
  （`.trans` ldmatrix 不使用）の直後に同じ分割を適用する。

## 3. smem 2 面分離案の不採用（静的 SMEM 予算超過）

hi/lo をそれぞれ独立した smem タイル（2 面）として持つ設計案も検討したが、
既存タイル構成（`MMA_TF32_BM=64・MMA_TF32_BN=64・MMA_TF32_BK=16・
STAGES=3`）での 1 面の静的 SMEM 使用量は 28,416B（
`kernels_mma_tf32::MMA_TF32_SHARED_MEM_BYTES`）であり、2 面では 56,832B と
なって静的 SMEM 上限（`kernels_mma::MMA_STATIC_SMEM_LIMIT_BYTES` = 48KiB =
49,152B）を超える。よって smem 分離案は不採用とし、上記のレジスタ段分割を
採用した（`kernels_mma_tf32x3.rs::MMA_TF32X3_SHARED_MEM_BYTES` は単発 TF32
と完全に同一のまま。const assert で固定）。

## 4. 3 回の累積順序

`(a_hi, b_lo) → (a_lo, b_hi) → (a_hi, b_hi)` の順で `mma.sync` を 3 回発行
し、いずれも同じアキュムレータ（`d[mi][nj][0..4]`）へ蓄積する。

設計方針の出典: CUTLASS `mma_tensor_op_fast_f32.h`（`cutlass::gemm::warp::
MmaTensorOpFastF32`）が採る 3-pass split-single 近似と同じ次数・同じ意図
（本イシューの実装セッションはリポジトリ外の CUTLASS ソースを直接参照
できる環境になかったため、具体的な行・変数名は引用せず設計思想のみを
出典として明記する）:

- 大きい寄与（`hi·hi`）を最後に累積し、小さい寄与 2 項（`hi·lo`・`lo·hi`）
  を先に累積する（`d = hi·hi + (hi·lo + lo·hi + d_prev)` の形にして
  `hi·hi` が最終的な有効桁を支配するようにする設計意図）。本カーネルは
  この意図に従い `(a_hi,b_lo) → (a_lo,b_hi) → (a_hi,b_hi)` の順で 3 回
  累積する。**先頭 2 項（`hi·lo`・`lo·hi`）の相互順序は浮動小数点加算が
  結合則を満たさないため数値上完全に無関係ではないが、いずれも `hi·hi`
  より 1 桁小さい同格の補正項であり「大きい寄与を最後に累積する」という
  設計意図には影響しない**（先頭 2 項の順序自体を CUTLASS の実装と
  bit-for-bit 一致させる主張はしない）。
- `lo·lo` 項（相対誤差 ~2^-22 オーダー。仮数 10bit の丸め誤差の 2 乗に
  相当）は同じ近似次数の判断により省略する。

`mma.sync` 命令文字列自体は `MMA_TF32X3_ISSUE` マクロ（ソース中 1 箇所の
定義）から 3 回呼び出す形にし、コピペ増殖の回帰をソース検査テスト
（`kernels_mma_tf32x3.rs::tests::
mma_tf32x3_source_issues_mma_sync_from_single_macro_site_called_three_times`）
でロックする。

## 5. f32 SIMT との bit 非一致（FMA 契約の例外）

3×TF32 の結果は f32 SIMT 参照実装と **bit 一致しない**（TF32 丸めを経由する
構造上、`lo·lo` 省略・累積順序も f32 SIMT の FMA チェーンとは異なるため）。
`.claude/rules/coding-rust.md` の FMA 契約統一節に「3×TF32 opt-in モード
（`CudaGemmPrecision::Tf32x3`。既定 OFF）は例外」と明記した（ユーザー承認
2026-09-06・#1338 コメント）。数値一致の**複合判定自体**（相対誤差 1e-3
未満 または 絶対誤差 1e-5 未満）は変更しない。

## 6. レジスタ増加の見積り（性能実測は #1356 のスコープ）

フラグメント配列を hi/lo で倍化するため（A: `a_frag[2][WARP_TILES_M][4]` →
`a_hi_frag[2][WARP_TILES_M][4]` + `a_lo_frag[2][WARP_TILES_M][4]`。B も同様
に倍化）、単発 TF32 経路よりレジスタ使用量が増える。occupancy への影響・
純カーネル時間は本イシューでは実測しない（#1356 のスコープ）。

## 7. fail-closed 契約・facade 到達経路

`docs/cuda-tf32-optin-api-decision.md` 追補節（イシュー #1355）を参照。要点:

- 既定 `Fp32Strict`。`fandhe_ai::set_cuda_gemm_precision(CudaGemmPrecision::
  Tf32x3)` で opt-in。
- opt-in 時にカーネル使用不能（cc<8.0・NVRTC コンパイル失敗・cp.async 16
  バイト整列制約不成立）なら `BackendError::KernelLaunchFailed`（
  `"3xTF32 gemm unavailable (fail-closed): …"`）を返し、FP32／単発 TF32 への
  黙示フォールバックはしない。
- 適用範囲は素の `CudaBackendOps::gemm` のみ（`gemm_bias_act`・
  `gemm_resident_*`・学習経路は対象外）。

## 8. #1356 実測記入欄の引き継ぎ先

イシュー #1356（本節を引き継いだ実装。誤差分布プローブ `--routes mma`
拡張・純カーネル時間ベンチ `gemm_tf32x3_kernel_time_bench` を追加）は
コード実装（`crates/backend-cuda/examples/wmma_tolerance_probe.rs`・
`crates/backend-cuda/examples/gemm_tf32x3_kernel_time_bench.rs`）まで
完了したが、**本エージェント実行環境に CUDA 実機（GB10）が接続されて
いなかったため実測値は未取得のまま**である（`docs/real-hardware-
verification-env.local.md` 不在。数値の推定・外挿・捏造はしない）。

誤差分布（8.1）・純カーネル時間（8.2）・GB10 実機実測記録（8.3）の
記入欄は本節に残さず、`docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md`
（実測記入欄のスケルトン・再現手順・集計スクリプトを含む独立ドキュメント）
へ一本化して引き継いだ。同ドキュメント §0 が実測状況を明記し、§11 に
3 択語彙（「opt-in 維持・推奨」／「opt-in 維持・条件付き推奨」／
「opt-in 維持・非推奨」）による採否判定欄（未確定のまま）を持つ。
CPU f32 参照実装との厳密ゼロ fail 成立可否
（`assert_parity`）が不成立の場合の baseline 非後退方式（`ParityPath`
追加）への再割り当ては、引き続きユーザー承認を得たうえで判断する（本
イシューでは baseline 行・`ParityPath` 変種を追加していない）。
