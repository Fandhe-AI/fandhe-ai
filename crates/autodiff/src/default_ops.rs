//! `Tape::new()`／`Tape::default()`／`compat::Sequential::predict`
//! （無引数 `ops` 版）が使う naive CPU 参照実装（TASK-12.1d 追補・#164
//! codex-review 第 19 波以降の P1 是正）。
//!
//! **背景**: TASK-12.1d は `Tape::new(ops: Box<dyn BackendOps + Send>)`
//! を必須所有値化し、無引数 `Tape::new()`／`impl Default for Tape`／
//! 無引数 `Sequential::predict(&Tensor)` を削除する破壊的変更を行った
//! （`docs/public-api-design.md` §4.1「破壊的変更」節）。`autodiff` は
//! `tests/architecture_boundaries.rs` により具体バックエンドクレート
//! （`backend-cpu`／`backend-cuda`／`backend-metal`）への依存を機械検査で
//! 禁止されているため、この破壊自体は正当（同節「移行手順」参照）。
//!
//! しかし codex-review はこの破壊を無引数 API の完全撤去と読み、
//! 互換 API を別名で残すことを要求した（PR #403 第 19〜21 波）。本
//! モジュールはその要求に応える最小限の compat 経路である: `eval.rs`
//! （クレート非公開の naive 参照実装。forward 値計算と数式を共有し、
//! `test_support::TestOps` と同一の委譲先）へそのまま委譲する
//! `BackendOps` 実装を、`#[cfg(test)]` 限定ではなく本番ビルドに含める
//! 形で追加する。第 22 波の P1 是正で ops 必須版は `Tape::new_with_ops`
//! へ改名し、`Tape::new()` を出荷済みシグネチャどおりの無引数 compat
//! 入口として復元した（`crates/autodiff/src/tape.rs` 参照）。
//!
//! **性能特性**: 本実装は融合実行（`FusionPlan::from_ops` 等）を経ない
//! 逐次 naive 実装であり、`backend-cpu` の最適化カーネル（`rayon` 並列・
//! BLIS ブロッキング等）と同等の性能を持たない。`Tape::default()`／
//! `Sequential::predict(&Tensor)`（無引数 `ops` 版）は「デフォルト値でも
//! 動く」ことを保証する compat 経路に徹し、性能が必要な呼び出し元は
//! 引き続き `Tape::new_with_ops(ops)`／`Sequential::predict_with_ops(input, ops)`
//! へ最適化済み `BackendOps`（`facade` composition root が結線する
//! `backend-cpu`／`backend-cuda`／`backend-metal` 等）を明示的に渡す
//! （`docs/public-api-design.md` §4.1「承認済み内容と結線の実装場所」の
//! 既定バックエンド方針とは独立: 本モジュールは `facade` 未実装下での
//! 暫定 compat であり、`facade` 実装後もこのフォールバック経路自体は
//! 維持する）。

use tensor_core::{BackendError, BackendOps, Device, Tensor, broadcast_shape, matmul_out_shape};

/// `eval.rs` の naive 参照実装へ委譲する compat 用 `BackendOps`。
/// `gemm`/`add`/`mul`/`relu`/`exp`/`tanh`/`sum`/`max` はいずれも構造的に
/// 失敗しない（`eval.rs` 側が非 fallible なため）が、`BackendOps` の
/// 契約に合わせ `Result` で包む（`test_support::TestOps` と同型）。
pub(crate) struct NaiveOps;

impl BackendOps for NaiveOps {
    fn device(&self) -> Device {
        Device::Cpu
    }

    // `gemm`/`add`/`mul` は `eval.rs` の naive 実装（呼び出し元が shape を
    // 事前検証済みという契約。`eval.rs` モジュール冒頭コメント参照）へ
    // 委譲する。`Var::matmul`/`add`/`mul`（`var.rs`）経由の呼び出しでは
    // グラフ構築時点で `matmul_out_shape`/`broadcast_shape` 済みのため
    // 実害はないが、`BackendOps` trait 自体は shape 不一致で
    // `ShapeMismatch` を返す契約（`backend_ops.rs`）であり、
    // `CpuBackendOps`（`backend-cpu`）もこれを満たす。`NaiveOps` を trait
    // 経由で直接呼ぶ将来の呼び出し元とも contract を揃えるため、ここでも
    // 同じ検証を行ってから委譲する（codex-review 第 19〜21 波の P1 是正の
    // 副次確認として advisor 指摘・2026-08-08 追記）。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        matmul_out_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
        Ok(crate::eval::matmul(a, b))
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        broadcast_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
        Ok(crate::eval::add(a, b))
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        broadcast_shape(a.shape(), b.shape()).map_err(BackendError::ShapeMismatch)?;
        Ok(crate::eval::mul(a, b))
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(crate::eval::relu(a))
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(crate::eval::exp(a))
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(crate::eval::tanh(a))
    }

    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        let shape = a.shape().to_vec();
        let out_shape =
            tensor_core::reduce_out_shape(&shape, dim).map_err(BackendError::ShapeMismatch)?;
        Ok(crate::eval::sum(a, dim, &out_shape))
    }

    /// `max` は単位元を持たないため、縮約対象の要素数が 0 の場合は
    /// エラーを返す（`eval::max` 自身は `f32::NEG_INFINITY` を返す
    /// テスト向けの寛容な挙動だが、`backend-cpu::CpuBackendOps::max`
    /// （`reduction::ReduceError::EmptyReduction` → `KernelLaunchFailed`）
    /// と挙動を揃えるための追加検査。`Tape::default()`／
    /// `Sequential::predict` の compat 経路と最適化バックエンド経路とで
    /// 同一入力に対する挙動を分岐させない。`sum` は単位元 0 を持つため
    /// 本検査は不要）。
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        let shape = a.shape().to_vec();
        let out_shape =
            tensor_core::reduce_out_shape(&shape, dim).map_err(BackendError::ShapeMismatch)?;
        // `backend-cpu::reduction::max`（`crates/backend-cpu/src/
        // reduction.rs`）と同じ条件式: 縮約軸の長さが 0 かつ出力要素数が
        // 1 個以上（＝各出力位置の縮約グループが空）の場合のみエラーと
        // する。出力自体が空（他軸も 0）の場合は縮約グループが存在しない
        // ため vacuous に成功とする。
        let is_empty_reduction = match dim {
            None => a.numel() == 0,
            Some(axis) => {
                let axis_len = shape.get(axis).copied().unwrap_or(0);
                let outer: usize = shape[..axis].iter().product();
                let inner: usize = shape[axis + 1..].iter().product();
                axis_len == 0 && outer.saturating_mul(inner) > 0
            }
        };
        if is_empty_reduction {
            return Err(BackendError::KernelLaunchFailed(
                "max: empty reduction has no maximum element".to_string(),
            ));
        }
        Ok(crate::eval::max(a, dim, &out_shape))
    }
}

/// `Tape::default()`／`Sequential::predict(&Tensor)`（無引数 `ops` 版）が
/// 使う既定 `ops`（[`NaiveOps`]）を組み立てる。
pub(crate) fn naive_ops() -> Box<dyn BackendOps + Send> {
    Box::new(NaiveOps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_rejects_shape_mismatch_like_cpu_backend() {
        let a = Tensor::new(vec![1.0_f32; 6], &[2, 3]).unwrap();
        let b = Tensor::new(vec![1.0_f32; 6], &[2, 3]).unwrap();
        let err = NaiveOps.gemm(&a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn add_rejects_non_broadcastable_shape_mismatch() {
        let a = Tensor::new(vec![1.0_f32; 6], &[2, 3]).unwrap();
        let b = Tensor::new(vec![1.0_f32; 4], &[2, 2]).unwrap();
        let err = NaiveOps.add(&a, &b).unwrap_err();
        assert!(matches!(err, BackendError::ShapeMismatch(_)));
    }

    #[test]
    fn max_full_reduction_of_empty_tensor_errors_like_cpu_backend() {
        let empty = Tensor::new(Vec::<f32>::new(), &[0]).unwrap();
        let err = NaiveOps.max(&empty, None).unwrap_err();
        assert!(matches!(err, BackendError::KernelLaunchFailed(_)));
    }

    #[test]
    fn max_axis_reduction_over_zero_length_axis_errors_like_cpu_backend() {
        // shape [0, 3] を axis=0 で縮約: 出力は [3]（3 要素）だが各出力
        // 位置の縮約グループは 0 要素（`backend-cpu::reduction::max` と
        // 同じ「出力要素数 > 0 かつ軸長 0」の空縮約ケース）。
        let empty = Tensor::new(Vec::<f32>::new(), &[0, 3]).unwrap();
        let err = NaiveOps.max(&empty, Some(0)).unwrap_err();
        assert!(matches!(err, BackendError::KernelLaunchFailed(_)));
    }

    #[test]
    fn max_axis_reduction_with_vacuous_empty_output_succeeds() {
        // shape [0, 0] を axis=0 で縮約: 出力自体が 0 要素（vacuous）
        // なので縮約グループは実在せず、エラーにならない。
        let empty = Tensor::new(Vec::<f32>::new(), &[0, 0]).unwrap();
        let result = NaiveOps.max(&empty, Some(0)).unwrap();
        assert_eq!(result.numel(), 0);
    }

    #[test]
    fn sum_of_empty_tensor_returns_identity_zero() {
        // `sum` は単位元 0 を持つため `max` と異なりエラーにならない。
        let empty = Tensor::new(Vec::<f32>::new(), &[0]).unwrap();
        let result = NaiveOps.sum(&empty, None).unwrap();
        assert_eq!(result.get(&[]), Some(0.0));
    }
}
