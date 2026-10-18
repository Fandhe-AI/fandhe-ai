# backend-rocm の BackendOps 被覆計画と実装 issue 分解案

- 親イシュー: #3150（Phase 7）。本イシュー: #3156
- 前提: #3155（HIPRTC 経路と f32 GEMM。`crates/backend-rocm`）、`docs/backend-rocm-hip-ffi-decision.md`、`docs/backend-rocm-hiprtc-gemm-decision.md`、`docs/backend-abstraction-amd-readiness-decision.md`、`docs/rocm-grade-up-conditions-v2-spec-proposal.md` §3b・§3e
- 基準コミット: `2e38c37c`（#3422 マージ後の `main`）
- 承認状態: ROCm 継続は D-B（#3151）に基づく。**本 doc の分解案の起票は所有者の承認待ちで未実施**。Could への格上げは宣言しない（条件 (f)' の評価材料を提供するのみ）。コード・依存・tolerance・baseline・FMA 契約・`docs/spec/` は変更しない（docs のみの変更）

## 1. 結論

- `BackendOps`（`crates/tensor-core/src/backend_ops.rs`）のメソッドは **144 件**。ROCm 方針の内訳は次のとおり。

| 方針 | 件数 | 意味 |
|---|---|---|
| 必須 | 9 | 既定実装がなく、`impl` 時点で書く（R0） |
| 既定継承 | 24 | trait の既定実装（合成・委譲・`None`）を使う。ROCm 側で何も書かない |
| 移植 | 63 | CUDA の NVRTC カーネルを HIPRTC へ移植する（案 R0〜R13） |
| Unsupported | 48 | 既定の `BackendError::Unsupported` のまま（§5） |

- 9 件の必須メソッドのうち `gemm` は #3155 の `RocmGemmF32` を結線する。残りは R0 でホスト経路（読み戻して CPU 参照へ委譲）とし、R2・R3 でカーネル化する
- 分解案は R0〜R13 の 14 件。実機が要るのは R3〜R6・R12・R13、要らないのは R0〜R2・R7〜R11
- wave64 の影響を受けるのは 8 warp 固定の縮約系（reduce・huber・bce・nll・kl_div・typed_f64）と shfl・マスクのみの系（§6）。横断 issue R1 で ROCm 用 `WarpGeometry` を先に作る

## 2. メソッド数の根拠

trait の終端で止める次のコマンドで数える（基準コミット `2e38c37c`）。

```sh
awk '/^pub trait BackendOps/{f=1;next} f&&/^}/{exit} f&&/^    fn /{n++} END{print n}' crates/tensor-core/src/backend_ops.rs
# => 144
```

- `docs/rocm-grade-up-conditions-v2-spec-proposal.md` §3b の 195 は、`exit` を持たない awk（`f&&/^    fn /` で trait 終端後も数え続ける）の値で、trait 後続の impl・テストの `fn` を含む粗い数え上げである（同じコマンドは `2e38c37c` では 238 を返す。195 は当時のファイル規模の値）。**正しい値は 144**。§3b 自体の訂正は本変更の範囲外（所有者判断）
- 再計測: 基準コミット以降は同じコマンドを実行し、表の行数と突き合わせる
- CUDA（`crates/backend-cuda/src/ops.rs` の `impl BackendOps for CudaBackendOps`）は 89 件、Metal（`crates/backend-metal/src/ops.rs`）は 90 件をオーバーライドする（CUDA のうち `linalg_svd`・`linalg_matrix_norm` は `Unsupported` を明示返却するだけ。`typed_ops_f64` の 8 演算は CUDA ではネイティブ実装済み〈`crates/backend-cuda/src/typed_f64.rs` の `impl TypedOps<f64> for CudaBackendOps`、#2060〉で、`Unsupported` ではない）

## 3. 方針の定義

| 方針 | 定義 |
|---|---|
| 必須 | 既定実装がない。ホスト経路か移植のどちらかで成立させる |
| 既定継承 | 既定実装をそのまま使う。`*_tracked` は非 tracked 版へ委譲するため、非 tracked 版の方針に従う |
| 移植 | f32 SIMT・FMA は `__builtin_fmaf` で明示・REQ-8 の手動境界チェックを維持して HIPRTC へ移植する |
| ホスト経路 | デバイス→ホストの読み戻し後に CPU 参照へ委譲する。必須メソッドを先に成立させる場合のみ。性能は保証しない |
| Unsupported | 既定のまま。autodiff 側の `*_with_fallback` がホストへフォールバックする前提 |

実機検証の要否の定義: wave 幅に依存する、または FP 縮約の順序が実機で決まるものは「要」。要素ごとの演算と添字計算だけのものは「不要」（ソース生成とホスト側モデルで検証できる）。「要」の実機実行は `#[ignore]` テストとして #3159 に送る。

## 4. 被覆表（144 件。Op は trait 内の宣言順）

CUDA 列は `crates/backend-cuda/src/` 配下。`*_tracked` の CUDA 列は「あり(明示委譲)」が CUDA で個別にオーバーライドされていることを示す。

| # | Op | 既定実装 | CUDA 実装（ファイル） | ROCm 方針 | wave64 影響 | 実機検証 | 案 |
|---|---|---|---|---|---|---|---|
| 1 | `device` | 必須 | ops.rs・device.rs | 必須(移植) | なし | 不要 | R0 |
| 2 | `memory_ops` | 既定=合成/委譲 | memory.rs・typed_f64.rs・typed_f16.rs・typed_bf16.rs・cast.rs | 既定継承(None。型付き経路は f32 SIMT 初期スコープ外) | - | - | - |
| 3 | `typed_ops_f64` | 既定=合成/委譲 | memory.rs・typed_f64.rs・typed_f16.rs・typed_bf16.rs・cast.rs | 既定継承(None。型付き経路は f32 SIMT 初期スコープ外) | - | - | - |
| 4 | `typed_ops_f16` | 既定=合成/委譲 | memory.rs・typed_f64.rs・typed_f16.rs・typed_bf16.rs・cast.rs | 既定継承(None。型付き経路は f32 SIMT 初期スコープ外) | - | - | - |
| 5 | `typed_ops_bf16` | 既定=合成/委譲 | memory.rs・typed_f64.rs・typed_f16.rs・typed_bf16.rs・cast.rs | 既定継承(None。型付き経路は f32 SIMT 初期スコープ外) | - | - | - |
| 6 | `cast_ops` | 既定=合成/委譲 | memory.rs・typed_f64.rs・typed_f16.rs・typed_bf16.rs・cast.rs | 既定継承(None。型付き経路は f32 SIMT 初期スコープ外) | - | - | - |
| 7 | `sgd_step_device` | 既定=Unsupported | sgd.rs・kernels_sgd.rs | 移植 | なし | 不要 | R9 |
| 8 | `sgd_step_device_tracked` | 既定=合成/委譲 | なし(既定) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 9 | `adam_step_device` | 既定=Unsupported | adam.rs・kernels_adam.rs | 移植 | なし | 不要 | R9 |
| 10 | `adam_step_device_tracked` | 既定=合成/委譲 | なし(既定) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 11 | `rmsprop_step_device` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 12 | `rmsprop_step_device_tracked` | 既定=合成/委譲 | なし(既定) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 13 | `adagrad_step_device` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 14 | `adagrad_step_device_tracked` | 既定=合成/委譲 | なし(既定) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 15 | `lamb_step_device` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 16 | `lamb_step_device_tracked` | 既定=合成/委譲 | なし(既定) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 17 | `captured_segment_key` | 既定=合成/委譲 | graph.rs（CUDA Graph capture） | 既定継承(None) | - | - | - |
| 18 | `run_captured_sgd_step_segment` | 既定=Unsupported | graph.rs（CUDA Graph capture） | Unsupported | - | - | - |
| 19 | `gemm` | 必須 | gemm.rs・kernels_tiled_pipeline*.rs（3×TF32 は gemm_mma_tf32x3.rs） | 必須(移植済 #3155 を結線) | なし(GEMM は WARP 非依存) | 不要 | R0 |
| 20 | `gemm_fp32_strict` | 既定=合成/委譲 | gemm.rs | 既定継承(gemm/default_gemm_batched に合成。CUDA の厳密版は TF32 経路の分離が目的) | なし | 不要 | R0 |
| 21 | `gemm_batched` | 既定=合成/委譲 | gemm.rs | 既定継承(gemm/default_gemm_batched に合成。CUDA の厳密版は TF32 経路の分離が目的) | なし | 不要 | R0 |
| 22 | `gemm_batched_fp32_strict` | 既定=合成/委譲 | gemm.rs | 既定継承(gemm/default_gemm_batched に合成。CUDA の厳密版は TF32 経路の分離が目的) | なし | 不要 | R0 |
| 23 | `gemm_fp32_strict_into` | 既定=Unsupported | gemm.rs（resident は memory.rs・transpose.rs も使用） | 移植(R0 の RocmGemmF32 に積む) | なし | 不要 | R11 |
| 24 | `gemm_fp32_strict_into_tracked` | 既定=合成/委譲 | あり(CUDA は明示委譲) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 25 | `gemm_fp32_strict_into_with_bias_reduce_tracked` | 既定=合成/委譲 | なし(既定) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 26 | `add` | 必須 | ops.rs→elementwise.rs・kernels_elementwise.rs | 必須(R0 でホスト経路→R2 で移植) | なし | 不要 | R0→R2 |
| 27 | `mul` | 必須 | ops.rs→elementwise.rs・kernels_elementwise.rs | 必須(R0 でホスト経路→R2 で移植) | なし | 不要 | R0→R2 |
| 28 | `relu` | 必須 | ops.rs→elementwise.rs・kernels_elementwise.rs | 必須(R0 でホスト経路→R2 で移植) | なし | 不要 | R0→R2 |
| 29 | `exp` | 必須 | ops.rs→elementwise.rs・kernels_elementwise.rs | 必須(R0 でホスト経路→R2 で移植) | なし | 不要 | R0→R2 |
| 30 | `tanh` | 必須 | ops.rs→elementwise.rs・kernels_elementwise.rs | 必須(R0 でホスト経路→R2 で移植) | なし | 不要 | R0→R2 |
| 31 | `scalar_unary` | 既定=Unsupported | kernels_scalar_op.rs | 移植 | なし | 不要 | R2 |
| 32 | `scalar_binary` | 既定=Unsupported | kernels_scalar_op.rs | 移植 | なし | 不要 | R2 |
| 33 | `sum` | 必須 | ops.rs→reduce.rs・kernels_reduce.rs | 必須(R0 でホスト経路→R3 で移植) | 8 warp 固定 | 要 | R0→R3 |
| 34 | `max` | 必須 | ops.rs→reduce.rs・kernels_reduce.rs | 必須(R0 でホスト経路→R3 で移植) | 8 warp 固定 | 要 | R0→R3 |
| 35 | `min` | 既定=Unsupported | reduce.rs・kernels_reduce.rs／arg_reduce.rs・kernels_arg_reduce.rs | 移植 | min=8 warp 固定／arg_reduce=なし | 要 | R3 |
| 36 | `argmax` | 既定=Unsupported | reduce.rs・kernels_reduce.rs／arg_reduce.rs・kernels_arg_reduce.rs | 移植 | min=8 warp 固定／arg_reduce=なし | 要 | R3 |
| 37 | `argmin` | 既定=Unsupported | reduce.rs・kernels_reduce.rs／arg_reduce.rs・kernels_arg_reduce.rs | 移植 | min=8 warp 固定／arg_reduce=なし | 要 | R3 |
| 38 | `mse_loss` | 既定=Unsupported | mse.rs・kernels_mse.rs | 移植 | 注入済(WARP_SIZE。ROCm は wave 幅 64 も受理する版へ) | 要 | R5 |
| 39 | `mse_loss_backward` | 既定=Unsupported | mse.rs・kernels_mse.rs | 移植 | 注入済(WARP_SIZE。ROCm は wave 幅 64 も受理する版へ) | 要 | R5 |
| 40 | `huber_loss` | 既定=Unsupported | huber.rs・kernels_huber.rs | 移植 | 8 warp 固定 | 要 | R5 |
| 41 | `huber_loss_backward` | 既定=Unsupported | huber.rs・kernels_huber.rs | 移植 | 8 warp 固定 | 要 | R5 |
| 42 | `bce_loss` | 既定=Unsupported | bce.rs・kernels_bce.rs | 移植 | 8 warp 固定 | 要 | R5 |
| 43 | `bce_loss_backward` | 既定=Unsupported | bce.rs・kernels_bce.rs | 移植 | 8 warp 固定 | 要 | R5 |
| 44 | `nll_loss` | 既定=Unsupported | nll.rs・kernels_nll.rs | 移植 | 8 warp 固定 | 要 | R5 |
| 45 | `nll_loss_backward` | 既定=Unsupported | nll.rs・kernels_nll.rs | 移植 | 8 warp 固定 | 要 | R5 |
| 46 | `kl_div_loss` | 既定=Unsupported | kl_div.rs・kernels_kl_div.rs | 移植 | 8 warp 固定 | 要 | R5 |
| 47 | `kl_div_loss_backward` | 既定=Unsupported | kl_div.rs・kernels_kl_div.rs | 移植 | 8 warp 固定 | 要 | R5 |
| 48 | `softmax` | 既定=Unsupported | softmax.rs・kernels_softmax.rs | 移植 | 注入済(WARP_SIZE) | 要 | R4 |
| 49 | `cumsum` | 既定=Unsupported | scan.rs・kernels_scan.rs | 移植 | なし | 不要 | R8 |
| 50 | `cumprod` | 既定=Unsupported | scan.rs・kernels_scan.rs | 移植 | なし | 不要 | R8 |
| 51 | `scan_cummax` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 52 | `scan_cummin` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 53 | `scan_logcumsumexp` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 54 | `pool3d_max` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 55 | `pool3d_avg` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 56 | `lrn_forward` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 57 | `weight_norm_forward` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 58 | `spectral_norm_forward` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 59 | `pad_modes_forward` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 60 | `stat_kthvalue` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 61 | `stat_median_dim` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 62 | `stat_median_all` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 63 | `stat_quantile` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 64 | `stat_nansum` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 65 | `stat_nanmean` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 66 | `binning_histc` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 67 | `binning_bincount` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 68 | `binning_bincount_weighted` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 69 | `binning_searchsorted` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 70 | `log_softmax` | 既定=Unsupported | softmax.rs・kernels_softmax.rs | 移植 | 注入済(WARP_SIZE) | 要 | R4 |
| 71 | `log_softmax_backward` | 既定=Unsupported | log_softmax_backward.rs・kernels_log_softmax_backward.rs | 移植 | shfl のみ | 要 | R4 |
| 72 | `concat` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 73 | `where_cond` | 既定=Unsupported | elementwise.rs・kernels_elementwise.rs | 移植 | なし | 不要 | R2 |
| 74 | `masked_fill` | 既定=Unsupported | elementwise.rs・kernels_elementwise.rs | 移植 | なし | 不要 | R2 |
| 75 | `gather` | 既定=Unsupported | gather_scatter.rs・kernels_gather_scatter.rs | 移植 | なし | 不要 | R8 |
| 76 | `pad` | 既定=Unsupported | constant_pad.rs・kernels_constant_pad.rs | 移植 | なし | 不要 | R7 |
| 77 | `im2col` | 既定=Unsupported | im2col.rs・kernels_im2col.rs | 移植 | なし | 不要 | R7 |
| 78 | `col2im` | 既定=Unsupported | im2col.rs・kernels_im2col.rs | 移植 | なし | 不要 | R7 |
| 79 | `conv2d` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 80 | `im2col3d` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 81 | `col2im3d` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 82 | `conv3d` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 83 | `max_pool2d` | 既定=Unsupported | pooling.rs・kernels_pooling.rs | 移植 | なし | 不要 | R7 |
| 84 | `avg_pool2d` | 既定=Unsupported | pooling.rs・kernels_pooling.rs | 移植 | なし | 不要 | R7 |
| 85 | `adaptive_avg_pool2d` | 既定=Unsupported | pooling.rs・kernels_pooling.rs | 移植 | なし | 不要 | R7 |
| 86 | `adaptive_max_pool2d` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 87 | `scatter` | 既定=Unsupported | gather_scatter.rs・kernels_gather_scatter.rs | 移植 | なし | 不要 | R8 |
| 88 | `indexed_scatter_reduce` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 89 | `interpolate` | 既定=Unsupported | interpolate.rs・kernels_interpolate.rs | 移植 | なし | 不要 | R8 |
| 90 | `sort` | 既定=Unsupported | sort.rs・kernels_sort.rs | 移植 | なし | 不要 | R8 |
| 91 | `topk` | 既定=Unsupported | sort.rs・kernels_sort.rs | 移植 | なし | 不要 | R8 |
| 92 | `one_hot` | 既定=Unsupported | gather_scatter.rs・kernels_gather_scatter.rs | 移植 | なし | 不要 | R8 |
| 93 | `unique` | 既定=Unsupported | unique.rs・kernels_unique.rs | 移植 | なし | 不要 | R8 |
| 94 | `unique_ext` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 95 | `gemm_bias_act` | 既定=合成/委譲 | gemm.rs（CUDA は専用化） | 既定継承(R11 で専用化を任意検討) | なし | 不要 | R11 |
| 96 | `gemm_resident_rhs` | 既定=Unsupported | gemm.rs（resident は memory.rs・transpose.rs も使用） | 移植(R0 の RocmGemmF32 に積む) | なし | 不要 | R11 |
| 97 | `gemm_resident_rhs_act` | 既定=合成/委譲 | gemm.rs（CUDA に個別オーバーライドなし） | 既定継承(gemm_resident_rhs＋relu に合成) | なし | 不要 | R11 |
| 98 | `gemm_resident_lhs` | 既定=Unsupported | gemm.rs（resident は memory.rs・transpose.rs も使用） | 移植(R0 の RocmGemmF32 に積む) | なし | 不要 | R11 |
| 99 | `linear_forward_device` | 既定=Unsupported | gemm.rs（resident は memory.rs・transpose.rs も使用） | 移植(R0 の RocmGemmF32 に積む) | なし | 不要 | R11 |
| 100 | `linear_forward_device_tracked` | 既定=合成/委譲 | なし(既定) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 101 | `paged_kv_append` | 既定=Unsupported | paged_kv.rs・kernels_paged_kv.rs | 移植 | shfl のみ(1 か所) | 要 | R12 |
| 102 | `paged_kv_append_tracked` | 既定=合成/委譲 | なし(既定) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 103 | `paged_attention_forward` | 既定=Unsupported | paged_kv.rs・kernels_paged_kv.rs | 移植 | shfl のみ(1 か所) | 要 | R12 |
| 104 | `paged_attention_forward_tracked` | 既定=合成/委譲 | なし(既定) | 既定継承(非 tracked 版へ委譲。非 tracked 版の方針に従う) | 非 tracked 版に従う | 非 tracked 版に従う | - |
| 105 | `linear_chain_forward_captured` | 既定=合成/委譲 | graph.rs・precision.rs（CUDA Graph capture） | 既定継承(None) | - | - | - |
| 106 | `binary_elementwise_device` | 既定=Unsupported | elementwise.rs・kernels_elementwise.rs | 移植 | なし | 不要 | R2 |
| 107 | `unary_elementwise_device` | 既定=Unsupported | elementwise.rs・kernels_elementwise.rs | 移植 | なし | 不要 | R2 |
| 108 | `run_fused` | 既定=Unsupported | fused_elementwise.rs・kernels_fused_elementwise.rs（RMSNorm 融合プランは rmsnorm.rs/softmax.rs 経由） | 移植(elementwise 範囲。RMSNorm 融合プランは R6 後) | elementwise=なし／RMSNorm 融合=注入済 | 不要 | R2 |
| 109 | `release_cached_device_memory` | 既定=合成/委譲 | pool.rs・context_cache.rs | 既定継承(ROCm にプールなし) | - | - | - |
| 110 | `device_memory_pool_stats` | 既定=合成/委譲 | pool.rs・context_cache.rs | 既定継承(ROCm にプールなし) | - | - | - |
| 111 | `gemm_checksum` | 既定=Unsupported | なし(CUDA も既定) | Unsupported | - | - | - |
| 112 | `rmsnorm` | 既定=Unsupported | rmsnorm.rs・kernels_rmsnorm.rs | 移植(f64 アキュムレータ契約) | 注入済(WARP_SIZE) | 要 | R6 |
| 113 | `layer_norm` | 既定=Unsupported | layer_norm.rs・kernels_layer_norm.rs | 移植(f64 アキュムレータ契約) | shfl のみ | 要 | R6 |
| 114 | `rmsnorm_backward` | 既定=Unsupported | norm_backward.rs・kernels_norm_backward.rs | 移植(f64 アキュムレータ契約) | shfl のみ | 要 | R6 |
| 115 | `layer_norm_backward` | 既定=Unsupported | norm_backward.rs・kernels_norm_backward.rs | 移植(f64 アキュムレータ契約) | shfl のみ | 要 | R6 |
| 116 | `batch_norm_train` | 既定=Unsupported | batch_norm.rs・kernels_batch_norm.rs | 移植(f64 アキュムレータ契約) | shfl のみ | 要 | R6 |
| 117 | `batch_norm_infer` | 既定=Unsupported | batch_norm.rs・kernels_batch_norm.rs | 移植(f64 アキュムレータ契約) | shfl のみ | 要 | R6 |
| 118 | `lstm_pointwise` | 既定=Unsupported | rnn_cell.rs・kernels_rnn_cell.rs | 移植 | なし | 不要 | R10 |
| 119 | `lstm_hidden_backward` | 既定=Unsupported | rnn_cell.rs・kernels_rnn_cell.rs | 移植 | なし | 不要 | R10 |
| 120 | `lstm_cell_backward` | 既定=Unsupported | rnn_cell.rs・kernels_rnn_cell.rs | 移植 | なし | 不要 | R10 |
| 121 | `gru_pointwise` | 既定=Unsupported | rnn_cell.rs・kernels_rnn_cell.rs | 移植 | なし | 不要 | R10 |
| 122 | `gru_backward` | 既定=Unsupported | rnn_cell.rs・kernels_rnn_cell.rs | 移植 | なし | 不要 | R10 |
| 123 | `linalg_inv` | 既定=Unsupported | linalg.rs・kernels_linalg.rs（内部 opt-in ゲート既定 OFF） | 移植(ゲート配下) | なし | 要 | R13 |
| 124 | `linalg_solve` | 既定=Unsupported | linalg.rs・kernels_linalg.rs（内部 opt-in ゲート既定 OFF） | 移植(ゲート配下) | なし | 要 | R13 |
| 125 | `linalg_det` | 既定=Unsupported | linalg.rs・kernels_linalg.rs（内部 opt-in ゲート既定 OFF） | 移植(ゲート配下) | なし | 要 | R13 |
| 126 | `linalg_cholesky` | 既定=Unsupported | linalg.rs・kernels_linalg_factor.rs（同ゲート） | 移植(ゲート配下) | なし | 要 | R13 |
| 127 | `linalg_qr` | 既定=Unsupported | linalg.rs・kernels_linalg_factor.rs（同ゲート） | 移植(ゲート配下) | なし | 要 | R13 |
| 128 | `linalg_svd` | 既定=Unsupported | ops.rs（CUDA も Unsupported を明示返却） | Unsupported | - | - | - |
| 129 | `linalg_matrix_norm` | 既定=Unsupported | ops.rs（CUDA も Unsupported を明示返却） | Unsupported | - | - | - |
| 130 | `linalg_eigh` | 既定=Unsupported | linalg.rs・kernels_linalg_factor.rs（同ゲート） | 移植(ゲート配下) | なし | 要 | R13 |
| 131 | `linalg_slogdet` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 132 | `linalg_pinv` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 133 | `linalg_lstsq` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 134 | `linalg_matrix_rank` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 135 | `fft_rfft` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 136 | `fft_irfft` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 137 | `fft_fft` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 138 | `fft_ifft` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 139 | `fft_stft` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 140 | `fft_istft` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 141 | `var` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 142 | `vector_norm` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 143 | `logsumexp` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |
| 144 | `vector_norm_p` | 既定=Unsupported | なし(CUDA も既定=Unsupported) | Unsupported | - | - | - |

## 5. Unsupported 確定群（初期スコープ外）

| 群 | 理由 |
|---|---|
| Tensor Core 相当（CUDA の `kernels_mma*`・`kernels_wmma*`・TF32×3） | ROCm の MFMA は命令体系が別物。f32 SIMT を初期スコープとする（MFMA は別 issue 候補。`backend-rocm-hiprtc-gemm-decision.md` §7）。ROCm には Tensor Core 経路を持ち込まず、`gemm` は f32 SIMT のみ |
| graph capture 系（`run_captured_sgd_step_segment`・`captured_segment_key`・`linear_chain_forward_captured`） | CUDA Graph 相当（HIP Graph）は初期スコープ外。既定の `Ok(None)` で呼び出し側が通常経路へ戻る |
| 型付き経路（`typed_ops_f64/f16/bf16`・`cast_ops`・`memory_ops`） | f32 SIMT 初期スコープ外。既定の `None`。CUDA の `typed_ops_f64` は #2060 でネイティブ実装済み（`typed_f64.rs`）だが、ROCm の型付き経路は初期スコープ外とする方針は不変 |
| CUDA も未実装のもの（fft 6 件・`linalg_slogdet/pinv/lstsq/matrix_rank/svd/matrix_norm`・stat 6 件・binning 4 件・`scan_cummax/cummin/logcumsumexp`・`pool3d_*`・`lrn`・`weight_norm`・`spectral_norm`・`pad_modes`・`concat`・`conv2d/3d`・`im2col3d`・`col2im3d`・`adaptive_max_pool2d`・`indexed_scatter_reduce`・`unique_ext`・`var`・`vector_norm*`・`logsumexp`・`rmsprop/adagrad/lamb`） | 突き合わせ先（CUDA 実装）がなく、REQ-2 の数値一致を評価できない。CUDA 側の実装が先 |
| `gemm_checksum` | CUDA も既定。検査用途で初期スコープ外 |

いずれも autodiff 側の `*_with_fallback` がホストへフォールバックする前提で、`BackendError::Unsupported` のまま残すことが契約上の正しい挙動である。

## 6. wave64 の影響

`crates/backend-cuda/src/kernels_*.rs` の grep 実測（`warp_sums[8]`・`lane < 8`・`__shfl`・`0xffffffff`・`WARP_SIZE`。基準コミット）。

| 区分 | カーネル | 状態 |
|---|---|---|
| `WARP_SIZE` 注入済み（#2126） | rmsnorm・softmax・mse | CUDA 側は width 32 のみ受理（`warp_geometry.rs` の `CUDA_SUPPORTED_WARP_WIDTH`）。ROCm 用は wave 幅 64 も受理する別実装が要る |
| **8 warp 固定**（`warp_sums[8]`・`lane < 8 ?`・`__shfl`＋`0xffffffff`） | reduce・typed_f64・nll・bce・kl_div・huber | 未注入。`WARPS_PER_BLOCK = block / wave 幅` から導く必要がある（`backend-abstraction-amd-readiness-decision.md` §2 (B')・§6c）。置換だけで済ませるとバッファ長・読み出し境界のずれで未初期化読みが起きる |
| shfl・マスクのみ（8 warp バッファなし） | norm_backward・layer_norm・batch_norm・log_softmax_backward・paged_kv（1 か所） | 初期 offset 16・フルマスク 32 bit の置換が要る |
| warp 幅に依存しない | elementwise・arg_reduce・scan・sort・unique・pooling・adam・sgd・transpose・im2col・gather_scatter・rnn_cell・constant_pad・interpolate など | スレッド数を 64 の倍数に揃える（`GemmF32Tile::new` と同じ検査） |
| ISA 固有 | `kernels_mma*`・`kernels_wmma*`・TF32×3 | Unsupported（§5） |

HIP の `__shfl_xor` はマスク引数を取らず、lane mask は 64 bit のため、CUDA 用の `WARP_SHFL_XOR`／`WARP_SYNC` マクロはそのまま流用できない（`backend-abstraction-amd-readiness-decision.md` §1・§2 (C)、`backend-rocm-hiprtc-gemm-decision.md` §4）。**R1** が ROCm 版 `WarpGeometry`（wave 幅 32/64 を受理し、`WARP_SIZE`・`WARP_HALF`・`WARPS_PER_BLOCK = block_dim / wave 幅`・HIP 用のシャッフル／同期マクロを、検証済みの数値と固定テンプレートだけで生成する）を作り、R3〜R6・R12 の前提とする。

## 7. 実装 issue の分解案（未起票）

全案に共通する受け入れ条件:

- HIPRTC ソースは静的テンプレートと検証済み数値の `#define` だけで作る（`GemmSource` と同じ newtype 方式）。コンパイルオプションは `GfxTarget` allowlist の固定リテラルだけ。外部入力の文字列をカーネルソースに入れない（A03）
- FMA は `__builtin_fmaf` で明示。REQ-8 の手動境界チェックを維持する
- 正規化統計・勾配の長軸縮約は `f64` アキュムレータ契約（`.claude/rules/coding-rust.md`）に従う
- tolerance・baseline・FMA 契約・依存ピンは変えない。新しい依存が要る案は別承認（本案には含まない）
- 新しい `unsafe` は理由コメントを付け、#3158 の監査表へ追記する
- ホスト側モデル（`cfg(test)`）で CPU 参照と照合する。「要」の実機実行は `#[ignore]` テストにして #3159 へ送る。parity が外れても許容誤差は緩めない
- 目安は 1 issue 2 時間。超える見込みなら分割する

| 案 | タイトル（案） | 主なメソッド | 依存 | 実機 |
|---|---|---|---|---|
| R0 | `feat(backend-rocm): RocmBackendOps の骨格と必須 9 メソッドを実装する` | `device`・`gemm`（`RocmGemmF32`）・`add`・`mul`・`relu`・`exp`・`tanh`・`sum`・`max`（ホスト経路）、既定継承の確認 | #3155 | 不要 |
| R1 | `feat(backend-rocm): wave 幅を注入する WarpGeometry を ROCm 用に作る` | wave32/64・`WARPS_PER_BLOCK` 導出・HIP 用シャッフルマクロ | R0 | 不要（wave64 の確認は #3159） |
| R2 | `feat(backend-rocm): elementwise 一式を移植する` | `scalar_unary/binary`・`binary/unary_elementwise_device`・`where_cond`・`masked_fill`・`run_fused`・`add/mul/relu/exp/tanh` のカーネル化 | R0 | 不要 |
| R3 | `feat(backend-rocm): 縮約カーネルを wave 幅対応で移植する` | `sum`・`max` のデバイス化、`min`・`argmax`・`argmin` | R1 | 要 |
| R4 | `feat(backend-rocm): softmax 系を移植する` | `softmax`・`log_softmax`・`log_softmax_backward` | R1 | 要 |
| R5 | `feat(backend-rocm): loss 5 種と backward を移植する` | mse・huber・bce・nll・kl_div（forward／backward 各 10 件）。2 時間を超える見込みなら R5a／R5b に分ける | R1 | 要 |
| R6 | `feat(backend-rocm): 正規化と backward を移植する` | `rmsnorm`・`layer_norm`・各 backward・`batch_norm_train/infer` | R1 | 要 |
| R7 | `feat(backend-rocm): im2col・col2im・pad・pooling を移植する` | `im2col`・`col2im`・`pad`・`max_pool2d`・`avg_pool2d`・`adaptive_avg_pool2d` | R0 | 不要 |
| R8 | `feat(backend-rocm): gather/scatter・sort・topk・unique・scan を移植する` | `gather`・`scatter`・`interpolate`・`one_hot`・`sort`・`topk`・`unique`・`cumsum`・`cumprod`。2 時間を超える見込みなら分ける | R0 | 不要 |
| R9 | `feat(backend-rocm): SGD/Adam のデバイス更新を移植する` | `sgd_step_device`・`adam_step_device` | R0 | 不要 |
| R10 | `feat(backend-rocm): LSTM/GRU の pointwise を移植する` | `lstm_*` 3 件・`gru_*` 2 件 | R0 | 不要 |
| R11 | `feat(backend-rocm): GEMM 派生（resident・into・linear_forward）を移植する` | `gemm_resident_rhs/lhs`・`gemm_fp32_strict_into`・`linear_forward_device`・`gemm_bias_act` 専用化（任意） | R0 | 不要 |
| R12 | `feat(backend-rocm): paged KV と paged attention を移植する` | `paged_kv_append`・`paged_attention_forward` | R1 | 要 |
| R13 | `feat(backend-rocm): linalg（inv/solve/det/cholesky/qr/eigh）を移植する` | CUDA が実装している 6 件（内部 opt-in ゲート既定 OFF を踏襲）。`svd`・`matrix_norm` は CUDA も Unsupported のため対象外。2 時間を超える見込みなら分割する | R0 | 要 |

対象外（Unsupported のまま）: MFMA 経路・f16/bf16・split-K・ディスクキャッシュ（`backend-rocm-hiprtc-gemm-decision.md` §7 の別 issue 候補）。

## 8. #3157 との境界（所有者確認事項）

`backend-rocm-hiprtc-gemm-decision.md` §7 は #3157 を「BackendOps・facade 統合」と書くが、#3157 本文は facade の `Device`・`tape_for`・`available_devices` だけを扱う。本計画は「`impl BackendOps for RocmBackendOps` の骨格は R0、#3157 は facade 側のみ」と**提案**する。所有者が確定済み（#3150 コメント 6248286174）。R0 は #3431 で実装。R0 では `backend-rocm` の `lib.rs`・`Cargo.toml` のコメントも実態に合わせて更新する。

## 9. 未確認事項

- HIP のシャッフル組み込み関数（64 bit マスク版）の仕様: R1（#3432）で確認済み。`__shfl_xor(var, laneMask, width)` はマスクなし、`__shfl_xor_sync(unsigned long long mask, ...)` は 64 bit マスク（出典: HIP C++ 言語拡張 Warp cross-lane functions、2026-10-18）。`__syncwarp` は記載なし。HIPRTC での実コンパイル・wave64 実機確認は #3159
- R6（#3437）の正規化カーネルで未確認: `__shfl_xor` の double オーバーロードの HIPRTC 実コンパイル、`#pragma clang fp contract(off)` 下での `__builtin_fma`（f64）の命令化、wave64 での正規化 parity の実測（いずれも #3159。`tests/norm_real_device.rs` の `#[ignore]` テストで確認する）
- デバイスからの gfx target 取得方法（#3157・#3159）
- wave64 実機（MI300X 想定。`rocm-cloud-amd-procurement-decision.md`）での縮約順序の差と parity の実測は #3159 まで未検証
- CUDA 列のファイル対応は `ops.rs` の呼び出しを基に付けた。R 案着手時に各 issue で再確認する
- R4（#3435）着手時の再確認: #71 `log_softmax_backward` は現行 CUDA で shfl を使わない（PR #1994 で butterfly を撤回済み。lane 0 が `i` 昇順に `double` 逐次和を取り `__shared__` ＋ `__syncthreads()` で公開）。wave 幅依存は列ループの `i += 32` のみで、ROCm 版は `i += WARP_SIZE` へ置換した（§6 の「shfl のみ」分類は実態と異なる）。softmax／log_softmax は 2 パス経路のみ移植した（onepass は動的 SMEM が要るが `launch.rs` は `sharedMemBytes = 0` 固定で口がない。数値契約は同じで性能のみの差）
- R12（#3443）着手時の再確認: #101 `paged_kv_append`・#103 `paged_attention_forward` の「shfl のみ(1 か所)」は現行 CUDA ソース（`kernels_paged_kv.rs`）と一致しない。カーネル本体は `__shfl` を使わず（`j` 軸は単一の連続ループ、max と和は thread 0 の逐次計算）、`__shfl` は CUDA 側テストの禁止リスト文字列にだけ現れる（§6 の grep がそれを拾ったと判断）。wave 幅への依存は block 寸法（attention 128・append 256。ともに 64 の倍数）だけで、`WarpGeometry` は不要。§4 の表自体は書き換えず本項に訂正を記す（表の改訂は所有者判断）

### R2（#3433）の実装記録

- **融合はインタプリタ方式**: CUDA の `run_fused` はプランごとに CUDA C を `format!` で生成するが、HIPRTC ソースは静的テンプレートのみ（A03）なので移植しない。静的カーネル `fused_ew_f32` を 1 本だけ用意し、プランは `[kind, a, b]` の `i32` 3 つ組をデバイスバッファとして渡す（`kernels_fused_elementwise.rs`）。allowlist（`Input`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`）の検証は HIP に触れる前にホストで行い、外れたら `Unsupported`。性能（レジスタ配列の動的添字）は未測定で #3159 で扱う。
- **opt-in ゲートは設けない**: CUDA の融合は `AtomicBool` ゲート（facade の Tier B 公開判断に結びつく）の背後にある。ROCm はどの候補リストにも登録していないため CPU 版と同じくゲートなしにした。facade へ露出する #3157 で揃えるかを所有者が判断する。
- **FMA**: 新設する 3 本の静的ソースすべての先頭に `#pragma clang fp contract(off)` を置く（HIP/clang 既定の暗黙 FMA 縮約を止め、縮約しない CPU 参照と丸めを揃える。`GeluTanh`・`Hardswish`・融合の「mul の後に add」が対象）。意図した FMA はなく `__builtin_fmaf` は使わない。`f64` アキュムレータ契約は縮約がないため該当しない。
- **デバイス常駐 API の到達性**: `binary_elementwise_device`／`unary_elementwise_device` は実装したが、`memory_ops()` は `None`（§4 の既定継承）のままなので crate の外から ROCm の `DeviceBuffer` を作れない。判別は `downcast_handle::<RocmBufferHandle>()` の型のみ（`device()` は #3157 まで `Device::Cpu` の仮置きで CPU 由来と区別できない）。被覆表内の不整合（#106・#107 は「移植」、#2 は `None`）として所有者の判断待ち。
- relu は CUDA と同じ `x > 0 ? x : 0`（NaN は 0。CPU の `f32::max` と `-0.0` の符号だけ異なりうる）。新しい `unsafe` はない（`KernelHost::run`・`TypedDeviceBuffer` の監査済み `unsafe` を再利用）。

### R10（#3441）の実装記録

- **対象**: `lstm_pointwise`・`lstm_hidden_backward`・`lstm_cell_backward`・`gru_pointwise`・`gru_backward` の 5 メソッド。HIPRTC の静的ソース 1 本・5 カーネル（`kernels_rnn_cell.rs`）、実行型 `RocmRnnCell`（`rnn_cell.rs`）、結線は子モジュール `ops/rnn_cell.rs`（`ops.rs` には 1 行の委譲だけ。兄弟 PR #3454・#3455 との衝突回避）。
- **FMA**: ソース先頭に `#pragma clang fp contract(off)` を置き、意図した FMA は CPU の `f32::mul_add` と同じ 3 か所（LSTM の `c`、GRU の `n` と `h`）だけ `__builtin_fmaf` で明示した。backward 3 本に FMA はなく字面どおりの順で評価する。シグモイドは CPU と同じ 2 分岐形（`__expf` 等の高速近似は不使用）。
- **`f64` アキュムレータ契約は該当しない**: 縮約（長軸の和）を持たない pointwise 演算のため。wave 幅にも依存せず（§6 の分類どおり）、実機は不要。
- **空入力**: `B == 0` または `hidden == 0` は HIP に触れず `CpuBackendOps` へ委譲し、CPU と同じ結果・エラーにする（R3 の空縮約と同じ前例）。
- **ROCm 非搭載**: 形状・引数の検査後に `DeviceUnavailable` を返し、`Unsupported` へは落とさない（A08）。
- **新しい `unsafe` はない**（`KernelHost::run`・`TypedDeviceBuffer` の監査済み `unsafe` を再利用。`docs/backend-rocm-unsafe-audit.md` §3.9）。
- **未確認**: HIPRTC での実コンパイルと `expf`／`tanhf` の CPU との libm parity は #3159。ホストモデル（`rnn_cell_model.rs`）の超越関数はホスト libm で、GPU libm の模擬ではない。

### R7（#3438）の実装記録

- **CUDA 再確認**: `backend-cuda` の `kernels_{im2col,constant_pad,pooling}.rs`・`{im2col,constant_pad,pooling}.rs`・`ops.rs` の 6 メソッドを移植元とした。ファイルは 1 対 1 で `backend-rocm/src/` に対応させ、結線の本体は子モジュール `ops/conv_pool.rs` に置き `ops.rs` からは 1 行で委譲する。
- **対象**: `pad`・`im2col`・`col2im`・`max_pool2d`・`avg_pool2d`・`adaptive_avg_pool2d`。いずれも wave 幅に依存しないため `WarpGeometry` は使わず、ブロック幅は `INDEX_OPS_BLOCK_DIM`（256。64 の倍数）。
- **FMA**: 6 カーネルとも float の積和を持たない（im2col・pad はコピー、max はコピーと比較、col2im・avg 系は `double` の加算と除算 1 回）ので `__builtin_fmaf` を使う箇所はなく、FMA 契約は該当なし。全ソースの先頭に `#pragma clang fp contract(off)` を置くのは、将来の編集で暗黙の FMA が入るのを止める予防策（R2 と同じ）。
- **f64 契約の適用先**: `col2im`・`avg_pool2d`・`adaptive_avg_pool2d` は `double` アキュムレータへ row-major 固定順（`kh` 外側・`kw` 内側）に逐次加算し、最後に 1 回だけ `float` へ落とす（avg 系は `(float)(acc / (double)divisor)`）。CPU 参照の `f64` 逐次和と演算順まで同じなので bit 完全一致を目標とする。HIP は `double` をネイティブに扱うので Metal の soft-f64 は不要。
- **追加した手動境界チェック（REQ-8）**: adaptive の窓端に `if (h_end > h_in) h_end = h_in;`・`if (w_end > w_in) w_end = w_in;` を入れた。CUDA は窓の式の性質に頼っていた箇所で、有効な入力では値は変わらない。
- **`Unsupported` の事前確定**: avg の `dilation != [1, 1]`（カーネルが dilation を持たない。CUDA PR #1888 と同じ）と、`i32` 上限超過（`SizeLimitExceeded` を `Unsupported` へ写す。CUDA の `map_im2col_error`／`map_pooling_error` と同じ）は、HIP に触れる前に確定させる。共有の `to_backend_error` は変えず、本族専用の写像 `map_conv_pool_error` を使う。内部契約違反（`InvalidOpArgs`）は `ShapeMismatch(ElementCountOverflow)` に写し、ホストフォールバックで覆い隠さない。
- **R8 の実行型は未結線のまま**: `gather`・`cumsum`・`interpolate` 等の実行型（#3439）は `RocmBackendOps` に結線されておらず、`device()` も `Device::Cpu` の仮置きのまま。R7 の範囲外の事実として記録するだけで、起票はユーザー承認後に行う。
- **実機確認の申し送り（#3159）**: HIPRTC の暗黙ヘッダで `isnan(float)` が使えるか（監査 doc §6 で確認済みなのは linalg の `fabs`・`isfinite` のみ）、gfx942 での `double` 加算のスループット、wave64 での block 256 の動作。parity はホストモデル（`conv_pool_model.rs`）と `CpuBackendOps` の `to_bits` 一致で検証しており、HIPRTC のコンパイル結果そのものの検証ではない。
- `pool3d_*`・`adaptive_max_pool2d`・`im2col3d`・`col2im3d`・`pad_modes_forward` は CUDA も未実装なので `Unsupported` のまま（§5）。新しい `unsafe` はない（`KernelHost::run`・`TypedDeviceBuffer` の監査済み `unsafe` を再利用）。

### R11（#3442）の実装記録

- **CUDA 再確認**: `backend-cuda/src/ops.rs` の `gemm_fp32_strict_into_impl`・`gemm_bias_act`・`gemm_resident_rhs`・`linear_forward_device`・`gemm_resident_lhs` と `kernels.rs` の `TILED_BIAS_ACT_F32`、CPU の同名メソッド（`backend-cpu/src/ops.rs`）を移植元・契約の基準とした。検査順序（ハンドル判別 → rank → shape → 退化形状）とエラー種別（`DeviceMismatch`／`ShapeMismatch`／`InvalidArgument`／`Unsupported`）は CUDA と揃えた。
- **カーネルは 1 本（`kernels_gemm_bias_act.rs`）**: 4 メソッドを `has_bias`／`act`／オフセット引数の違いだけで共用する。オペランドは「base ポインタ + `long long` の要素オフセット」で渡す。`TypedDeviceBuffer` は base ポインタしか持たず、ホスト側で生ポインタを足さないことで、起動の `Arg::Ptr` が `as_device_ptr` の値だけという K1 監査の前提を崩さない。主ループは R0 の GEMM と**バイト単位で同一**（テスト `main_loop_is_byte_identical_to_r0_gemm` で固定）。FMA は `__builtin_fmaf` で明示、CUDA の `acc += a*b` は持ち込まない。
- **REQ-8**: カーネル側は guarded load・`row < m`／`col < n` の書き込みガード・ガード内でのみ `bias[col]` を読む。ホスト側は `validate_gemm_dims`（i32 収容）・`operand_range`（4 オペランドすべての `offset + len <= バッファ長`。checked）・`launch_geometry` の grid 上限・`out_offset + m*n <= out.numel()` を起動前に確かめる（二重化）。
- **退化形状**: `k == 0` は `gemm_resident_rhs`／`linear_forward_device` で `InvalidArgument`（CUDA と同じ）、`gemm_fp32_strict_into` では範囲を 0 で上書き。`gemm_resident_lhs` の `q == 0` は 0 埋めの `[p, r]`（HIP を使わない）、空出力は HIP に触れない。
- **継承**: `gemm_resident_rhs_act`（`gemm_resident_rhs` → `relu`）・`*_tracked`・`gemm_fp32_strict_into_with_bias_reduce_tracked` は既定のまま。ROCm は呼び出しごとに同期するため遅延失敗トークンを要さない。
- **`gemm_bias_act` の専用化は見送り**: 任意項目で、既定の合成（`gemm` → `add` → `relu`）が正しく動く。融合は性能だけの差。follow-up 候補。
- **NT/TN 転置ビューの高速経路は移植しない**: CUDA の `memory.rs`・`transpose.rs` 相当は持たず、host の `contiguous()` で再配置する（数値は同じで性能だけの差）。follow-up 候補。
- **数値契約**: bias は 1 回の加算、ReLU は `x > 0 ? x : 0`（R2 と同じ）。よって `gemm` → `add` → `relu` の合成と bit 一致する。長軸縮約はなく `f64` アキュムレータ契約は対象外。tolerance・baseline は変えていない。
- **到達性**: `memory_ops()` は `None` のままなので crate の外から ROCm の `DeviceBuffer` を作れず、`DeviceParamStore` からはまだ到達しない（R2 と同じ制約。#3157 の所有者判断待ち）。
- **実機確認の申し送り（#3159）**: HIPRTC のコンパイル、`long long` オフセット引数の ABI、別実行型（`EW_CACHE` 等）が確保したバッファの相互利用（同一プロセス・デバイス 0 で有効な想定）、wave64 での block 256。parity はホストモデル（`gemm_model::simulate_with_epilogue`）で検証しており、HIPRTC の出力そのものの検証ではない。
- 新しい `unsafe` はない（`KernelHost::run_geometry`・`TypedDeviceBuffer` の監査済み `unsafe` を再利用）。

### R12（#3443）の実装記録

- **対象**: `paged_kv_append`・`paged_attention_forward` の 2 メソッド（`*_tracked` は trait 既定の委譲のまま）。HIPRTC の静的ソース 1 本・2 カーネル（`kernels_paged_kv.rs`。`#define` はブロック寸法 2 つだけ）、実行型 `RocmPagedKv`（`paged_kv.rs`。resident 版は `pub(crate)`、スライス版 `append`／`attention` を公開）、結線は子モジュール `ops/paged_kv.rs`（`ops.rs` には 1 行ずつの委譲と `mod` 宣言のみ。thread_local キャッシュも子モジュールに置き、兄弟 PR #3455・#3457 との衝突面を最小にした）。
- **shfl・`WarpGeometry`**: 上の §9 訂正のとおり不要。ブロック内の違反フラグ集約は CUDA の `__syncthreads_or` ではなく、共有フラグ `s_bad` への書き込み（全書き手が同じ値 1）＋ `__syncthreads()` の組にした。`__syncthreads_or` は同梱の amd-rocm スキルの HIP 資料に記載がなく HIPRTC での可用性を確認できないため、確認済みのプリミティブだけで書ける等価形を採った（挙動は同じ。違反時は出力へ NaN）。
- **FMA**: ソース先頭に `#pragma clang fp contract(off)`。意図した FMA は CPU の `f32::mul_add` と同じ 2 か所（score の `(q * scale)` と `k` の積和、加重和の `p * v`）だけ `__builtin_fmaf` で明示。CUDA の `__fmul_rn` は contract(off) 下で不要。
- **`f64` アキュムレータ契約は該当しない**: 正規化統計でも勾配の長軸縮約でもなく、CUDA・CPU 参照とも f32 の逐次和（max と和は thread 0 が `j` の論理順）。REQ-2 の統一複合判定をそのまま適用する（結合順序が単一の連続ループと異ならない）。
- **引数型**: `launch::ArgKind` に符号なし 32bit がなく、`PagedKvLayout::recheck` は `num_blocks < u32::MAX` しか保証しないため `int` にも収まらない。CUDA の `unsigned int num_blocks` を `long long num_blocks`（`Arg::I64`）に変え、カーネル内は `(long long)id >= num_blocks`（番兵 `0xFFFFFFFF` も拒否）で比較する。`launch.rs` は変更しない。
- **REQ-8**: append は `idx < total`・`r >= rows || e >= embed_dim`・`slot >= pool_rows`。attention は `b_ll >= batch`・`s_prev < 0 || limit > seq_len || limit > s_stride`・`slot_idx >= max_blocks_per_seq`・`id >= num_blocks`（score 段・加重和段の両方）。違反時は出力へ NaN。添字は `long long`、ベクトル化ロードなし。ホスト側は tensor-core の `validate_paged_*` → 結線のハンドル長検査 → `paged_kv.rs` の launch 直前の再検証 → カーネル内の三層。
- **作業バッファ**: score の作業バッファと出力は `new_zeroed`（未初期化読みを作らない多層防御。数値契約には影響しない）。
- **空入力**: 追記 0 行は HIP に触れず `Ok(())`、`out_numel == 0`（`query_rows == 0` と同値）は `RocmBufferHandle::Empty` の `[B, L_new, E]` を返す。検証違反では pool を書き換えない。
- **ROCm 非搭載**: `DeviceUnavailable`（`Unsupported` ではない。A08）。engine とハンドル種別の取り違えは `DeviceMismatch`。
- **到達性**: `memory_ops()` は `None` のままなので crate の外から ROCm の `DeviceBuffer` を作れない（R2 と同じ。#3157 待ち）。`device()` も仮置きの `Device::Cpu` のまま。
- **新しい `unsafe` はない**（`KernelHost::run_geometry`〈監査行 K1〉・`TypedDeviceBuffer`〈T1〜T3。`u32`・`f32` は既存〉の再利用。`docs/backend-rocm-unsafe-audit.md` §3.11）。
- **検証**: ホストモデル（`paged_kv_model.rs`）と `CpuBackendOps` の pool 全体・出力の `to_bits` 一致（block_size 1/4/16、batch 1/2、複数ステップ、非連続ブロック ID、番兵の未使用枠）、モデル単体の違反時 NaN、`tests/paged_kv_absent.rs`。
- **未確認（#3159）**: HIPRTC での実コンパイル（`__builtin_fmaf`・`expf`・`__uint_as_float`）、GPU `expf` と CPU の parity（`tests/paged_kv_real_device.rs`、ログ行 `ROCM_PARITY_PAGED_KV`）、wave64 での block 128 の動作。

## 10. 起票手順（未実施）

所有者の承認後に `update-issue-tree` で親 #3150 配下へ R0〜R13 を追加する。各 issue に `phase:7`・依存（`dependsOn`）・実機要否を書く。承認前は起票しない。本 doc の追加だけでは #3156 の受け入れ条件（起票）は未達のまま残る。
