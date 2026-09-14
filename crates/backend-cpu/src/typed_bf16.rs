//! `TypedOps<half::bf16>` の CPU 実装(イシュー #1699・親 #1649・
//! `docs/backend-dtype-dispatch-design.md` §4.2)。
//!
//! # 方針: f32 カーネル再利用(bf16 専用カーネルを書かない)
//!
//! bf16 は「性能」ではなく「メモリ帯域・省メモリ表現」のための dtype
//! であり(設計 §5)、数値契約自体が「f32 へ昇格して計算し、最後に
//! 1 回だけ丸めて bf16 へ戻す」ことを求める(設計 §6)。したがって
//! 本モジュールは次の 3 段の薄いラッパーに徹し、`elementwise`/
//! `gemm`/`gemm_blis`/`reduction`/`parity` の f32 カーネル本体を
//! **1 行も変更しない**:
//!
//! 1. [`promote`]: `Tensor<bf16>` → `Tensor<f32>`(bf16→f32 は完全表現
//!    可能で損失なし。`contiguous()` で non-contiguous view を吸収して
//!    から複製する)
//! 2. 既存 f32 カーネル(`ops::CpuBackendOps` 経由の `gemm`・
//!    `elementwise::{add,mul,relu,exp,tanh}`・`reduction::{sum,max}`)
//! 3. [`round_to_bf16`]: `Tensor<f32>` → `Tensor<bf16>`
//!    (`bf16::from_f32` は最近接偶数丸め。要素ごとに 1 回のみ適用)
//!
//! この構成により「f32 経路 bit 同一」(本イシューの受け入れ条件)は
//! `git diff --stat` で構造的に証明できる(`crate::elementwise`・
//! `crate::gemm`・`crate::gemm_blis`・`crate::reduction`・`crate::parity`・
//! `fandhe_ai_tensor_core` を変更しないため)。
//!
//! # 数値契約の要点(設計 §6。tolerance・baseline は不変)
//!
//! bf16 は仮数 8 bit(1 ulp ≈ 2^-8・半 ulp ≈ 2^-9)であり、
//! `parity::RELATIVE_TOLERANCE`(1e-3)より粗い。したがって「参照値も
//! 出力 dtype(bf16)へ丸めてから比較する」契約でなければ複合判定は
//! 意味を持たない(`1.0 + 2^-8` が f32 では区別できるが bf16 では
//! `1.0` へ丸められる、という設計 §6 の例を本モジュールの単体テストで
//! 固定する)。乱数入力を異なる累積順序の f32 経路(例: 本番 BLIS 経路
//! と逐次参照)へ通したうえで両側 bf16 丸めして比較する検証は行わない
//! (丸め境界を跨いで 1 bf16 ulp ずれうるため。`gemm` は本番 BLIS 経路
//! と `parity::matmul_reference_fma` が bit 完全一致する契約(
//! `tests/gemm_blis_parity.rs`)があるため、参照点をそちらに揃えれば
//! この問題を回避できる)。
//!
//! `sum` の f64 アキュムレータ契約は f32 カーネル([`crate::reduction`])
//! 自身が既に満たしている(`sum_slice` は `CHUNK` 単位で f64 累算し
//! 最後に 1 回 `as f32`)ため、本モジュールは f32 出力を bf16 へ 1 回
//! 丸めるのみで足りる(方式 (a)。f64 中間値へ直接アクセスして bf16 へ
//! 丸める方式 (b) は `reduction::CHUNK` 等の可視性拡張を要するため
//! 採らない)。
//!
//! # スコープ外
//!
//! `Var`/`Tape`/VJP・facade 公開面・`MemoryOps`/resident 系・
//! カーネル融合(`gemm_bias_act` 等)・bf16 専用 SIMD は対象外
//! (設計 §8。親 #1649 の分解方針どおり後続イシューへ引き継ぐ)。

use half::bf16;

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};

use crate::ops::CpuBackendOps;
use crate::{elementwise, reduction};

/// `Tensor<bf16>` を `Tensor<f32>` へ昇格する(bf16→f32 は完全表現可能・
/// 損失なし)。non-contiguous view(transpose 後 view 等)は
/// `contiguous()` で複製してから読み出す(`ops.rs` の gemm 系が
/// 同種の non-contiguous 吸収に使う `contiguous()` パターンと同じ)。
fn promote(t: &Tensor<bf16>) -> Result<Tensor<f32>, BackendError> {
    let owned = t.contiguous();
    let slice = owned.as_slice().ok_or_else(|| {
        BackendError::KernelLaunchFailed(
            "typed_bf16::promote: tensor not contiguous after contiguous()".to_string(),
        )
    })?;
    let data: Vec<f32> = slice.iter().map(|v| v.to_f32()).collect();
    Tensor::new(data, owned.shape()).map_err(BackendError::ShapeMismatch)
}

/// `Tensor<f32>` を `Tensor<bf16>` へ丸める(`bf16::from_f32` は
/// 最近接偶数丸め。要素ごとに 1 回のみ適用し、これが本モジュールの
/// 演算全体を通じて唯一の丸め点になる)。
fn round_to_bf16(t: &Tensor<f32>) -> Result<Tensor<bf16>, BackendError> {
    let owned = t.contiguous();
    let slice = owned.as_slice().ok_or_else(|| {
        BackendError::KernelLaunchFailed(
            "typed_bf16::round_to_bf16: tensor not contiguous after contiguous()".to_string(),
        )
    })?;
    let data: Vec<bf16> = slice.iter().map(|v| bf16::from_f32(*v)).collect();
    Tensor::new(data, owned.shape()).map_err(BackendError::ShapeMismatch)
}

/// 行列積。本番 `BackendOps::gemm`(f32 BLIS 経路)を昇格入力へ適用する。
pub(crate) fn gemm_bf16(
    ops: &CpuBackendOps,
    a: &Tensor<bf16>,
    b: &Tensor<bf16>,
) -> Result<Tensor<bf16>, BackendError> {
    let a32 = promote(a)?;
    let b32 = promote(b)?;
    let out32 = <CpuBackendOps as BackendOps>::gemm(ops, &a32, &b32)?;
    round_to_bf16(&out32)
}

/// 要素ごとの加算(ブロードキャスト規則は `elementwise::add` に従う)。
pub(crate) fn add_bf16(a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
    let a32 = promote(a)?;
    let b32 = promote(b)?;
    let out32 = elementwise::add(&a32, &b32).map_err(BackendError::ShapeMismatch)?;
    round_to_bf16(&out32)
}

/// 要素ごとの乗算。
pub(crate) fn mul_bf16(a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
    let a32 = promote(a)?;
    let b32 = promote(b)?;
    let out32 = elementwise::mul(&a32, &b32).map_err(BackendError::ShapeMismatch)?;
    round_to_bf16(&out32)
}

/// ReLU 活性化(クランプのみのため丸め損失なし)。
pub(crate) fn relu_bf16(a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
    let a32 = promote(a)?;
    let out32 = elementwise::relu(&a32).map_err(BackendError::ShapeMismatch)?;
    round_to_bf16(&out32)
}

/// 要素ごとの指数関数(`f32::exp` の結果を 1 回 bf16 へ丸める)。
pub(crate) fn exp_bf16(a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
    let a32 = promote(a)?;
    let out32 = elementwise::exp(&a32).map_err(BackendError::ShapeMismatch)?;
    round_to_bf16(&out32)
}

/// 要素ごとの双曲線正接。
pub(crate) fn tanh_bf16(a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
    let a32 = promote(a)?;
    let out32 = elementwise::tanh(&a32).map_err(BackendError::ShapeMismatch)?;
    round_to_bf16(&out32)
}

/// 総和縮約(`dim` が `None` の場合は全要素縮約)。f32 カーネル
/// (`reduction::sum`)内部の f64 アキュムレータ契約に乗る(モジュール
/// doc の方式 (a))。
pub(crate) fn sum_bf16(a: &Tensor<bf16>, dim: Option<usize>) -> Result<Tensor<bf16>, BackendError> {
    let a32 = promote(a)?;
    let out32 = reduction::sum(&a32, dim).map_err(crate::ops::reduce_error_to_backend_error)?;
    round_to_bf16(&out32)
}

/// 最大値縮約(`dim` が `None` の場合は全要素縮約)。入力が既に bf16
/// 表現可能値であるため縮約自体に丸め損失は生じない。
pub(crate) fn max_bf16(a: &Tensor<bf16>, dim: Option<usize>) -> Result<Tensor<bf16>, BackendError> {
    let a32 = promote(a)?;
    let out32 = reduction::max(&a32, dim).map_err(crate::ops::reduce_error_to_backend_error)?;
    round_to_bf16(&out32)
}

impl TypedOps<bf16> for CpuBackendOps {
    fn gemm(&self, a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        gemm_bf16(self, a, b)
    }

    fn add(&self, a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        add_bf16(a, b)
    }

    fn mul(&self, a: &Tensor<bf16>, b: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        mul_bf16(a, b)
    }

    fn relu(&self, a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        relu_bf16(a)
    }

    fn exp(&self, a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        exp_bf16(a)
    }

    fn tanh(&self, a: &Tensor<bf16>) -> Result<Tensor<bf16>, BackendError> {
        tanh_bf16(a)
    }

    fn sum(&self, a: &Tensor<bf16>, dim: Option<usize>) -> Result<Tensor<bf16>, BackendError> {
        sum_bf16(a, dim)
    }

    fn max(&self, a: &Tensor<bf16>, dim: Option<usize>) -> Result<Tensor<bf16>, BackendError> {
        max_bf16(a, dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::device::BackendError;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<bf16> {
        let d: Vec<bf16> = data.into_iter().map(bf16::from_f32).collect();
        Tensor::new(d, shape).unwrap()
    }

    fn to_f32_vec(t: &Tensor<bf16>) -> Vec<f32> {
        t.contiguous()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_f32())
            .collect()
    }

    #[test]
    fn gemm_2x2_matches_hand_computed_value() {
        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]](f32/bf16 とも
        // 正確に表現できる整数のため丸め損失なしで一致する)。
        let ops = CpuBackendOps;
        let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let out = <CpuBackendOps as TypedOps<bf16>>::gemm(&ops, &a, &b).unwrap();
        assert_eq!(to_f32_vec(&out), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_shape_mismatch_returns_shape_mismatch_error() {
        let ops = CpuBackendOps;
        let a = t(vec![1.0, 2.0], &[1, 2]);
        let b = t(vec![1.0, 2.0, 3.0], &[3, 1]);
        let err = <CpuBackendOps as TypedOps<bf16>>::gemm(&ops, &a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn gemm_zero_k_dimension_succeeds_with_zero_output() {
        let ops = CpuBackendOps;
        let a = t(vec![], &[2, 0]);
        let b = t(vec![], &[0, 3]);
        let out = <CpuBackendOps as TypedOps<bf16>>::gemm(&ops, &a, &b).unwrap();
        assert_eq!(to_f32_vec(&out), vec![0.0; 6]);
    }

    #[test]
    fn add_broadcasts_like_f32_add() {
        let ops = CpuBackendOps;
        let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(vec![10.0, 20.0], &[2]);
        let out = <CpuBackendOps as TypedOps<bf16>>::add(&ops, &a, &b).unwrap();
        assert_eq!(to_f32_vec(&out), vec![11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn mul_scalar_broadcast() {
        let ops = CpuBackendOps;
        let a = t(vec![1.0, 2.0, 3.0], &[3]);
        let b = t(vec![2.0], &[1]);
        let out = <CpuBackendOps as TypedOps<bf16>>::mul(&ops, &a, &b).unwrap();
        assert_eq!(to_f32_vec(&out), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn relu_clamps_negative_values_without_rounding_loss() {
        let ops = CpuBackendOps;
        let a = t(vec![-1.0, 0.0, 1.0, -2.5], &[4]);
        let out = <CpuBackendOps as TypedOps<bf16>>::relu(&ops, &a).unwrap();
        assert_eq!(to_f32_vec(&out), vec![0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn exp_matches_f32_exp_then_round_to_bf16() {
        let ops = CpuBackendOps;
        let a = t(vec![0.0, 1.0, 2.0], &[3]);
        let out = <CpuBackendOps as TypedOps<bf16>>::exp(&ops, &a).unwrap();
        let expected: Vec<bf16> = [0.0f32, 1.0, 2.0]
            .iter()
            .map(|v| bf16::from_f32(v.exp()))
            .collect();
        let actual: Vec<bf16> = out.contiguous().as_slice().unwrap().to_vec();
        assert_eq!(actual, expected);
    }

    #[test]
    fn tanh_matches_f32_tanh_then_round_to_bf16() {
        let ops = CpuBackendOps;
        let a = t(vec![-1.0, 0.0, 1.0], &[3]);
        let out = <CpuBackendOps as TypedOps<bf16>>::tanh(&ops, &a).unwrap();
        let expected: Vec<bf16> = [-1.0f32, 0.0, 1.0]
            .iter()
            .map(|v| bf16::from_f32(v.tanh()))
            .collect();
        let actual: Vec<bf16> = out.contiguous().as_slice().unwrap().to_vec();
        assert_eq!(actual, expected);
    }

    #[test]
    fn sum_full_reduction() {
        let ops = CpuBackendOps;
        let a = t(vec![1.0, 2.0, 3.0, 4.0], &[4]);
        let out = <CpuBackendOps as TypedOps<bf16>>::sum(&ops, &a, None).unwrap();
        assert_eq!(to_f32_vec(&out), vec![10.0]);
    }

    #[test]
    fn sum_axis_reduction() {
        let ops = CpuBackendOps;
        let a = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let out = <CpuBackendOps as TypedOps<bf16>>::sum(&ops, &a, Some(1)).unwrap();
        assert_eq!(to_f32_vec(&out), vec![3.0, 7.0]);
    }

    #[test]
    fn sum_empty_reduction_returns_zero() {
        // `reduction::sum` は空縮約でも 0 を返す契約(`max` とは異なる。
        // モジュール doc・reduction.rs 双方の既存契約どおり)。
        let ops = CpuBackendOps;
        let a = t(vec![], &[0]);
        let out = <CpuBackendOps as TypedOps<bf16>>::sum(&ops, &a, None).unwrap();
        assert_eq!(to_f32_vec(&out), vec![0.0]);
    }

    #[test]
    fn max_full_reduction() {
        let ops = CpuBackendOps;
        let a = t(vec![1.0, 5.0, 3.0, -2.0], &[4]);
        let out = <CpuBackendOps as TypedOps<bf16>>::max(&ops, &a, None).unwrap();
        assert_eq!(to_f32_vec(&out), vec![5.0]);
    }

    #[test]
    fn max_empty_reduction_returns_kernel_launch_failed() {
        let ops = CpuBackendOps;
        let a = t(vec![], &[0]);
        let err = <CpuBackendOps as TypedOps<bf16>>::max(&ops, &a, None).unwrap_err();
        assert!(matches!(err, BackendError::KernelLaunchFailed(_)));
    }

    #[test]
    fn max_dim_out_of_range_returns_shape_mismatch() {
        let ops = CpuBackendOps;
        let a = t(vec![1.0, 2.0], &[2]);
        let err = <CpuBackendOps as TypedOps<bf16>>::max(&ops, &a, Some(5)).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn promote_handles_non_contiguous_transpose_view() {
        // transpose_2d() で得た view(non-contiguous)を昇格した場合と、
        // 明示的に contiguous() した場合とで同じ gemm 結果になることを
        // 確認する(promote 内部の contiguous() 呼び出しが機能する証拠)。
        let ops = CpuBackendOps;
        let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let a_t = a.transpose_2d().unwrap();
        let rhs = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let out_via_view = <CpuBackendOps as TypedOps<bf16>>::gemm(&ops, &a_t, &rhs).unwrap();
        let out_via_contig =
            <CpuBackendOps as TypedOps<bf16>>::gemm(&ops, &a_t.contiguous(), &rhs).unwrap();
        assert_eq!(to_f32_vec(&out_via_view), to_f32_vec(&out_via_contig));
    }

    /// 設計 §6 の丸め例: `1.0 + 2^-8` は f32 では区別できるが
    /// (`1.00390625`)、bf16 へは tie→偶数丸めで `1.0`(`0x3F80`)へ
    /// 丸められる。「参照値も出力 dtype へ丸めてから比較する」契約の
    /// 根拠をコードで固定する(tolerance 定数自体は変更しない)。
    #[test]
    fn add_one_plus_two_pow_neg_eight_rounds_to_one_via_bf16_tie_to_even() {
        let ops = CpuBackendOps;
        let a = t(vec![1.0], &[1]);
        let eps = t(vec![2.0f32.powi(-8)], &[1]);
        let out = <CpuBackendOps as TypedOps<bf16>>::add(&ops, &a, &eps).unwrap();
        let out_bits = out.contiguous().as_slice().unwrap()[0].to_bits();
        assert_eq!(out_bits, bf16::from_f32(1.0).to_bits());

        // f32 上では 1.0 と区別できる値(1.00390625)だが、丸めていない
        // その参照値と比較すると REQ-2 複合判定は fail する
        // (「参照値も丸める」契約が必要、という根拠)。
        let unrounded_f32_ref = 1.0f32 + 2.0f32.powi(-8);
        let actual_f32 = out.contiguous().as_slice().unwrap()[0].to_f32();
        let report = crate::parity::compare(&[actual_f32], &[unrounded_f32_ref]).unwrap();
        assert!(report.fail_count > 0);

        // 一方、参照値自体も bf16 へ丸めてから比較すれば通る。
        let rounded_ref = bf16::from_f32(unrounded_f32_ref).to_f32();
        let report2 = crate::parity::compare(&[actual_f32], &[rounded_ref]).unwrap();
        assert_eq!(report2.fail_count, 0);
    }

    #[test]
    fn nan_and_inf_propagate_with_correct_class() {
        let ops = CpuBackendOps;
        let a = t(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY], &[3]);
        let b = t(vec![1.0, 1.0, 1.0], &[3]);
        let out = <CpuBackendOps as TypedOps<bf16>>::add(&ops, &a, &b).unwrap();
        let v = to_f32_vec(&out);
        assert!(v[0].is_nan());
        assert!(v[1].is_infinite() && v[1] > 0.0);
        assert!(v[2].is_infinite() && v[2] < 0.0);
    }

    #[test]
    fn backend_ops_dyn_dispatch_reaches_typed_ops_bf16_accessor() {
        let ops = CpuBackendOps;
        let dyn_ops: &dyn BackendOps = &ops;
        let typed = dyn_ops
            .typed_ops_bf16()
            .expect("typed_ops_bf16 should be Some for CpuBackendOps");
        let a = t(vec![1.0, 2.0], &[2]);
        let b = t(vec![3.0, 4.0], &[2]);
        let out = typed.add(&a, &b).unwrap();
        assert_eq!(to_f32_vec(&out), vec![4.0, 6.0]);
    }

    /// 小整数入力(結果が bf16 で正確に表現できる)では、`TypedOps<bf16>`
    /// と f32 `BackendOps` の対応 8 演算(代表として gemm/add)が bit
    /// 単位で一致する(accessor 結線・意味論一致の確認であり、丸め誤差を
    /// 伴う数値 parity の検証ではない)。
    #[test]
    fn small_integer_inputs_match_f32_backend_ops_bit_exactly() {
        let ops = CpuBackendOps;
        let a_bf16 = t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b_bf16 = t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let a_f32 = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b_f32 = Tensor::<f32>::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

        let gemm_bf16_out =
            <CpuBackendOps as TypedOps<bf16>>::gemm(&ops, &a_bf16, &b_bf16).unwrap();
        let gemm_f32_out = <CpuBackendOps as BackendOps>::gemm(&ops, &a_f32, &b_f32).unwrap();
        assert_eq!(
            to_f32_vec(&gemm_bf16_out),
            gemm_f32_out.contiguous().as_slice().unwrap().to_vec()
        );

        let add_bf16_out = <CpuBackendOps as TypedOps<bf16>>::add(&ops, &a_bf16, &b_bf16).unwrap();
        let add_f32_out = <CpuBackendOps as BackendOps>::add(&ops, &a_f32, &b_f32).unwrap();
        assert_eq!(
            to_f32_vec(&add_bf16_out),
            add_f32_out.contiguous().as_slice().unwrap().to_vec()
        );
    }
}
