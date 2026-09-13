//! `TypedOps<half::f16>` の CPU 実装（イシュー #1698・親 #1649・
//! `docs/backend-dtype-dispatch-design.md` §4.2「最小集合 8 演算」）。
//!
//! # 方針: 既存 f32 `BackendOps` カーネルをソフトウェア変換で再利用する
//!
//! `half` クレートによる **ソフトウェア変換**（f16 ⇔ f32 の値変換のみ。
//! aarch64 fp16 intrinsics 等のハードウェア命令は使わない。設計 §5・
//! 承認事項 5 の既定側）で、各演算を次の 3 段に分解する:
//!
//! 1. **昇格**: 入力 `Tensor<f16>` を [`Tensor::host_slice`]（contiguous
//!    なら借用・非 contiguous な view はここで 1 回だけ実体化）で読み出し、
//!    `f16::to_f32` で `Vec<f32>` へ変換してから `Tensor<f32>` を構築する
//! 2. **演算**: 既存 `<CpuBackendOps as BackendOps>::{gemm, add, mul, relu,
//!    exp, tanh, sum, max}` へそのまま委譲する。BLIS 並列 GEMM・rayon
//!    elementwise・`CHUNK` 固定順序 reduction・エラー変換をすべて継承する
//!    ため、本ファイルは f32 カーネル本体（`elementwise.rs`／`gemm.rs`／
//!    `gemm_blis/`／`reduction.rs`／`parity.rs`）に一切触れない
//! 3. **丸め**: 出力 `Tensor<f32>` を `f16::from_f32`（IEEE 754 最近接
//!    偶数丸め）で `Vec<f16>` へ変換し最終 `Tensor<f16>` を構築する
//!
//! この構成により、`TypedOps<f16>` の各メソッドは構造的に
//! `f16::from_f32(BackendOps::op(upcast(x)))`（要素ごと bit 一致）という
//! 不変条件を満たす（`tests` モジュールで確認）。f32 既存経路
//! （`elementwise.rs`／`gemm.rs`／`gemm_blis/**`／`reduction.rs`／
//! `parity.rs`）は本ファイル追加によって一切変更されない。
//!
//! # 数値・意味論上の明文化事項
//!
//! - 丸めは `half::f16::from_f32`（IEEE 754 最近接偶数丸め）。f16 表現
//!   範囲（`|x| <= 65504`）を超える f32 結果は ±inf、絶対値が小さい結果は
//!   非正規化数／0 へ落ちる（PyTorch `half` と同じ IEEE 挙動。`exp`・
//!   大形状 `gemm`／`sum` で到達しうる）
//! - `sum`（全縮約・軸縮約）は f32 経路（f64 アキュムレータ→f32 に 1 回
//!   downcast）の出力を f16 へ丸める。つまり「f16 入力を f32 へ昇格し
//!   f32 経路を通した結果を f16 へ 1 回丸める」という意味では丸めは 1 回
//!   だが、f32 経路自体の内部で f64→f32 の丸めが既に 1 回発生している
//!   （二段丸め。`reduction::sum` の既存契約を変更しないための帰結）
//! - `max` は f32 版と同じ NaN 非伝播（既知事項・本イシューのスコープ外）
//! - `add`／`mul` のブロードキャスト規則は f32 版
//!   （`fandhe_ai_tensor_core::elementwise_out_shape`。NumPy 互換）と同一
//! - 非 contiguous view（`transpose_2d`／`narrow`／`broadcast_to`）は
//!   `host_slice()` で実体化してから渡すため、f32 版の stride 読み高速
//!   経路・`gemm` の NT/TN 転置 fast path は経由しない（性能最適化は本
//!   イシューのスコープ外。`docs/backend-dtype-dispatch-design.md` §8）
//! - 一時 `Vec<f32>` の追加確保（入力ごと・出力 1 回）は許容する（性能
//!   最適化は本イシューの対象外）
//!
//! # スコープ外
//!
//! bf16（#1699）・CUDA（#1650）・Metal（#1651）・`Var`／`Tape`／VJP・
//! facade 公開面への昇格・aarch64 fp16 intrinsics による高速化・
//! `elementwise.rs`／`reduction.rs` の `T: Scalar` 汎用化・変換の rayon
//! 並列化・NT/TN 転置 fast path の f16 対応は対象外（`docs/backend-
//! dtype-dispatch-design.md` §7〜§8・親 #1649 の後続イシュー分担）。

use half::f16;

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};

use crate::ops::CpuBackendOps;

/// `Tensor<f16>` を `Tensor<f32>` へソフトウェア変換で昇格する。
///
/// [`Tensor::host_slice`] で読み出す（contiguous なら借用・非
/// contiguous な view は 1 回だけ実体化）ため、呼び出し元の非
/// contiguous view（`transpose_2d`／`narrow`／`broadcast_to` 等）を
/// そのまま渡してよい。shape 自体は `f16`→`f32` で変わらないため
/// `Tensor::new` の失敗は契約上到達しないはずだが、`Tensor` 実装の
/// 不変条件違反に対する fail-safe として型付きエラーで受ける
/// （`crate::ops::gemm_contiguity_fail_safe` と同じ位置づけ）。
fn upcast_f16(t: &Tensor<f16>) -> Result<Tensor<f32>, BackendError> {
    let promoted: Vec<f32> = t.host_slice().iter().map(|v| v.to_f32()).collect();
    Tensor::new(promoted, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "typed_f16::upcast_f16: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

/// `Tensor<f32>` を `Tensor<f16>` へ最近接偶数丸めで降格する。
///
/// [`upcast_f16`] と対になるヘルパー。丸めは `half::f16::from_f32`
/// （IEEE 754 最近接偶数丸め）。shape は不変のため `Tensor::new` の
/// 失敗は契約上到達しないはずだが、同じ理由で fail-safe を持つ。
fn downcast_f32(t: &Tensor<f32>) -> Result<Tensor<f16>, BackendError> {
    let rounded: Vec<f16> = t.host_slice().iter().map(|&v| f16::from_f32(v)).collect();
    Tensor::new(rounded, t.shape()).map_err(|e| {
        BackendError::KernelLaunchFailed(format!(
            "typed_f16::downcast_f32: shape 不変のはずの Tensor::new が失敗した: {e}"
        ))
    })
}

impl TypedOps<f16> for CpuBackendOps {
    /// f16 入力を f32 へ昇格し、既存 `BackendOps::gemm`（BLIS 並列 GEMM・
    /// `f32::mul_add` 累算契約）へ委譲してから f16 へ 1 回丸める。
    fn gemm(&self, a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let b32 = upcast_f16(b)?;
        let out32 = BackendOps::gemm(self, &a32, &b32)?;
        downcast_f32(&out32)
    }

    /// f16 入力を f32 へ昇格し、既存 `BackendOps::add`（ブロードキャスト
    /// 規則は f32 版と同一）へ委譲してから f16 へ丸める。
    fn add(&self, a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let b32 = upcast_f16(b)?;
        let out32 = BackendOps::add(self, &a32, &b32)?;
        downcast_f32(&out32)
    }

    /// `add` と同型。既存 `BackendOps::mul` へ委譲する。
    fn mul(&self, a: &Tensor<f16>, b: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let b32 = upcast_f16(b)?;
        let out32 = BackendOps::mul(self, &a32, &b32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::relu` へ委譲する。
    fn relu(&self, a: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::relu(self, &a32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::exp` へ委譲する。f16 表現範囲
    /// （`|x| <= 65504`）を超える結果は `f16::from_f32` により
    /// `f16::INFINITY`／`f16::NEG_INFINITY` へ丸められる（IEEE 754
    /// 挙動。モジュール doc「数値・意味論上の明文化事項」参照）。
    fn exp(&self, a: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::exp(self, &a32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::tanh` へ委譲する。
    fn tanh(&self, a: &Tensor<f16>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::tanh(self, &a32)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::sum`（`f64` アキュムレータ・`CHUNK` 固定順序）
    /// へ委譲する。二段丸め（f64→f32→f16）についてはモジュール doc
    /// 「数値・意味論上の明文化事項」参照。
    fn sum(&self, a: &Tensor<f16>, dim: Option<usize>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::sum(self, &a32, dim)?;
        downcast_f32(&out32)
    }

    /// 既存 `BackendOps::max` へ委譲する。NaN 非伝播は f32 版と同じ
    /// 既知事項（モジュール doc 参照）。
    fn max(&self, a: &Tensor<f16>, dim: Option<usize>) -> Result<Tensor<f16>, BackendError> {
        let a32 = upcast_f16(a)?;
        let out32 = BackendOps::max(self, &a32, dim)?;
        downcast_f32(&out32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f16> {
        let d: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
        Tensor::new(d, shape).unwrap()
    }

    fn f32v(t: &Tensor<f16>) -> Vec<f32> {
        t.host_slice().iter().map(|v| v.to_f32()).collect()
    }

    #[test]
    fn gemm_2x2_known_value_bit_exact() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]（f16 で正確に
        // 表現可能な整数値のみを使うため、参照値との比較は丸め誤差を
        // 考慮しない厳密一致でよい）。
        let ops = CpuBackendOps::new();
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let out = TypedOps::<f16>::gemm(&ops, &a, &b).unwrap();
        assert_eq!(f32v(&out), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn add_broadcast_matches_f32_reference() {
        let ops = CpuBackendOps::new();
        // [2,3] + [3]（行方向ブロードキャスト）。
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[10.0, 20.0, 30.0], &[3]);
        let out = TypedOps::<f16>::add(&ops, &a, &b).unwrap();
        assert_eq!(f32v(&out), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn mul_broadcast_matches_f32_reference() {
        let ops = CpuBackendOps::new();
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[2.0, 3.0, 4.0], &[3]);
        let out = TypedOps::<f16>::mul(&ops, &a, &b).unwrap();
        assert_eq!(f32v(&out), vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);
    }

    #[test]
    fn relu_negative_values_clamped_to_zero() {
        let ops = CpuBackendOps::new();
        let a = t(&[-1.0, 0.0, 2.5, -3.5], &[4]);
        let out = TypedOps::<f16>::relu(&ops, &a).unwrap();
        assert_eq!(f32v(&out), vec![0.0, 0.0, 2.5, 0.0]);
    }

    #[test]
    fn exp_zero_is_one() {
        let ops = CpuBackendOps::new();
        let a = t(&[0.0], &[1]);
        let out = TypedOps::<f16>::exp(&ops, &a).unwrap();
        assert_eq!(f32v(&out), vec![1.0]);
    }

    #[test]
    fn exp_large_input_saturates_to_f16_infinity() {
        // f16 の有限最大値は 65504。exp(12) ≈ 162754.79 は f16 表現範囲外
        // のため `f16::from_f32` により +inf へ丸められる（IEEE 754
        // 挙動。モジュール doc 参照）。`assert_parity`（有限差分判定）は
        // 使わず inf であることを直接確認する。
        let ops = CpuBackendOps::new();
        let a = t(&[12.0], &[1]);
        let out = TypedOps::<f16>::exp(&ops, &a).unwrap();
        let v = out.host_slice();
        assert!(v[0].is_infinite() && v[0] > f16::from_f32(0.0));
    }

    #[test]
    fn tanh_zero_is_zero() {
        let ops = CpuBackendOps::new();
        let a = t(&[0.0], &[1]);
        let out = TypedOps::<f16>::tanh(&ops, &a).unwrap();
        assert_eq!(f32v(&out), vec![0.0]);
    }

    #[test]
    fn sum_all_and_axis_match_hand_computed_values() {
        let ops = CpuBackendOps::new();
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let total = TypedOps::<f16>::sum(&ops, &a, None).unwrap();
        assert_eq!(f32v(&total), vec![10.0]);
        let axis0 = TypedOps::<f16>::sum(&ops, &a, Some(0)).unwrap();
        assert_eq!(f32v(&axis0), vec![4.0, 6.0]);
    }

    #[test]
    fn max_all_and_axis_match_hand_computed_values() {
        let ops = CpuBackendOps::new();
        let a = t(&[1.0, 5.0, 3.0, 2.0], &[2, 2]);
        let total = TypedOps::<f16>::max(&ops, &a, None).unwrap();
        assert_eq!(f32v(&total), vec![5.0]);
        let axis1 = TypedOps::<f16>::max(&ops, &a, Some(1)).unwrap();
        assert_eq!(f32v(&axis1), vec![5.0, 3.0]);
    }

    /// 構造的な不変条件: `TypedOps<f16>` の出力は
    /// `f16::from_f32(BackendOps::<f32> の出力)` と要素ごと bit 一致する
    /// （8 演算すべて）。設計上「f32 カーネルを再利用するラッパー」で
    /// あることの直接的な検証。
    #[test]
    fn typed_f16_output_matches_f32_backend_ops_rounded_for_all_eight_ops() {
        let ops = CpuBackendOps::new();
        let a16 = t(&[1.0, -2.0, 3.5, 0.0, 4.0, -5.5], &[2, 3]);
        let b16 = t(&[2.0, 0.5, -1.0, 3.0, 1.0, 2.0], &[2, 3]);
        let a32 = upcast_f16(&a16).unwrap();
        let b32 = upcast_f16(&b16).unwrap();

        let relu32 = BackendOps::relu(&ops, &a32).unwrap();
        let relu16 = TypedOps::<f16>::relu(&ops, &a16).unwrap();
        assert_eq!(f32v(&relu16), relu32.host_slice().to_vec());

        let exp32 = BackendOps::exp(&ops, &a32).unwrap();
        let exp16 = TypedOps::<f16>::exp(&ops, &a16).unwrap();
        assert_eq!(
            f32v(&exp16),
            exp32
                .host_slice()
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        );

        let tanh32 = BackendOps::tanh(&ops, &a32).unwrap();
        let tanh16 = TypedOps::<f16>::tanh(&ops, &a16).unwrap();
        assert_eq!(
            f32v(&tanh16),
            tanh32
                .host_slice()
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        );

        let add32 = BackendOps::add(&ops, &a32, &b32).unwrap();
        let add16 = TypedOps::<f16>::add(&ops, &a16, &b16).unwrap();
        assert_eq!(
            f32v(&add16),
            add32
                .host_slice()
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        );

        let mul32 = BackendOps::mul(&ops, &a32, &b32).unwrap();
        let mul16 = TypedOps::<f16>::mul(&ops, &a16, &b16).unwrap();
        assert_eq!(
            f32v(&mul16),
            mul32
                .host_slice()
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        );

        let sum32 = BackendOps::sum(&ops, &a32, None).unwrap();
        let sum16 = TypedOps::<f16>::sum(&ops, &a16, None).unwrap();
        assert_eq!(
            f32v(&sum16),
            sum32
                .host_slice()
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        );

        let max32 = BackendOps::max(&ops, &a32, Some(1)).unwrap();
        let max16 = TypedOps::<f16>::max(&ops, &a16, Some(1)).unwrap();
        assert_eq!(
            f32v(&max16),
            max32
                .host_slice()
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        );

        // gemm は正方形状で K=2（`a16` を [2,3] のまま流用できないため
        // 別途 2x2 を用意する）。
        let ga = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let gb = t(&[2.0, 0.0, 1.0, 3.0], &[2, 2]);
        let ga32 = upcast_f16(&ga).unwrap();
        let gb32 = upcast_f16(&gb).unwrap();
        let gemm32 = BackendOps::gemm(&ops, &ga32, &gb32).unwrap();
        let gemm16 = TypedOps::<f16>::gemm(&ops, &ga, &gb).unwrap();
        assert_eq!(
            f32v(&gemm16),
            gemm32
                .host_slice()
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn non_contiguous_transpose_view_matches_contiguous_equivalent() {
        let ops = CpuBackendOps::new();
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let a_t = a.transpose_2d().unwrap();
        let a_t_contig = a_t.contiguous();

        let sum_view = TypedOps::<f16>::sum(&ops, &a_t, None).unwrap();
        let sum_contig = TypedOps::<f16>::sum(&ops, &a_t_contig, None).unwrap();
        assert_eq!(f32v(&sum_view), f32v(&sum_contig));

        let relu_view = TypedOps::<f16>::relu(&ops, &a_t).unwrap();
        let relu_contig = TypedOps::<f16>::relu(&ops, &a_t_contig).unwrap();
        assert_eq!(f32v(&relu_view), f32v(&relu_contig));
    }

    #[test]
    fn shape_mismatch_returns_typed_error_not_panic() {
        let ops = CpuBackendOps::new();
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[1.0, 2.0, 3.0], &[3]);
        let err = TypedOps::<f16>::add(&ops, &a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn max_empty_reduction_returns_kernel_launch_failed() {
        let ops = CpuBackendOps::new();
        let empty = t(&[], &[0]);
        let err = TypedOps::<f16>::max(&ops, &empty, None).unwrap_err();
        assert!(matches!(err, BackendError::KernelLaunchFailed(_)));
    }

    #[test]
    fn sum_empty_tensor_is_zero() {
        let ops = CpuBackendOps::new();
        let empty = t(&[], &[0]);
        let out = TypedOps::<f16>::sum(&ops, &empty, None).unwrap();
        assert_eq!(f32v(&out), vec![0.0]);
    }

    #[test]
    fn single_element_gemm_add_relu() {
        let ops = CpuBackendOps::new();
        let a = t(&[3.0], &[1, 1]);
        let b = t(&[4.0], &[1, 1]);
        let gemm_out = TypedOps::<f16>::gemm(&ops, &a, &b).unwrap();
        assert_eq!(f32v(&gemm_out), vec![12.0]);
        let add_out = TypedOps::<f16>::add(&ops, &a, &b).unwrap();
        assert_eq!(f32v(&add_out), vec![7.0]);
        let relu_out = TypedOps::<f16>::relu(&ops, &t(&[-1.0], &[1, 1])).unwrap();
        assert_eq!(f32v(&relu_out), vec![0.0]);
    }
}
