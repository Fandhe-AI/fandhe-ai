//! `histc`・`bincount`・`searchsorted`・`bucketize` の自由関数（イシュー #2638・親
//! #2625「Phase 4」・ルート #2499）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::histc` 等の委譲メソッド）は未承認で、承認依頼は
//! #2677（公開自体は承認後の #2678・#2679）。本モジュールは内部クレート限定の入口で、
//! `Var`／`Tape` に inherent メソッドを足さない。保留は `crates/facade/src/lib.rs` の
//! `BinningOpsHoldDoctestGuard` と `crates/facade/tests/api_surface.rs` の否定ガードが
//! 機械的に固定する（`docs/autodiff-binning-ops-decision.md`）。
//!
//! **非微分**: 4 演算とも出力は整数のカウント・索引、または勾配を持たないヒストグラムで、
//! tape へノードを積まない（`push_eager`／`push_lazy` を呼ばない。出力は detached）。PyTorch
//! 2.14.0 の実測でも `histc`・`searchsorted`・`bucketize` は `requires_grad == False`、
//! `bincount(weights=…)` は `grad_fn=NotImplemented` で backward が例外になる
//! （`docs/autodiff-binning-ops-decision.md` §5）。
//!
//! **PyTorch 相当**:
//!
//! | 関数 | PyTorch 相当 | 出力 |
//! |---|---|---|
//! | [`histc`] | `torch.histc(x, bins, min, max)` | 長さ `bins` の `Tensor<f32>` |
//! | [`bincount`] | `torch.bincount(input, minlength=…)` | `Tensor<i32>`（PyTorch は int64） |
//! | [`bincount_weighted`] | `torch.bincount(input, weights, minlength)` | `Tensor<f32>` |
//! | [`searchsorted`] | `torch.searchsorted(seq, values, right=…)` | `values` と同 shape の `Tensor<i32>` |
//! | [`bucketize`] | `torch.bucketize(input, boundaries, right=…)` | `input` と同 shape の `Tensor<i32>` |
//!
//! `bincount` の入力は整数で `Var`（f32 のみ）では表せないため `Tensor<i32>` を取り、
//! `BackendOps` へ到達するために `tape` を明示引数にする。
//!
//! **経路**: ① 引数・形状・確保サイズ・索引上限の検査（実体化より前。
//! `fandhe_ai_tensor_core::binning`）→ ② 入力の実体化 → ③ `BackendOps::binning_*`
//! （`Unsupported` のときだけ共有ホストカーネル `binning::*_host` へフォールバックし、他の
//! エラーは伝播する。戻り値 shape も検証する）。
//!
//! **対象外**: `searchsorted` の `side`／`sorter`／`out_int32`・`out=`・整数 dtype 入力・
//! int64 索引・`torch.histogram`／`histogramdd`・重みへの勾配。

use fandhe_ai_tensor_core::binning::{self, SearchSortedLayout};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::tape::{Tape, materialize_fallible};
use crate::var::Var;

fn shape_mismatch(actual: &[usize], expected: &[usize]) -> AutodiffError {
    AutodiffError::Backend(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
        lhs: actual.to_vec(),
        rhs: expected.to_vec(),
    }))
}

fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

/// バックエンド結果の共通後半。`Unsupported` のときだけ `host` で計算し、他のエラーは伝播する。
fn resolve<T>(
    backend: Result<Tensor<T>, BackendError>,
    verify: impl FnOnce(&[usize]) -> Result<(), AutodiffError>,
    host: impl FnOnce() -> Result<Tensor<T>, AutodiffError>,
) -> Result<Tensor<T>, AutodiffError>
where
    T: fandhe_ai_tensor_core::Element,
{
    match backend {
        Ok(t) => {
            verify(t.shape())?;
            Ok(t)
        }
        Err(BackendError::Unsupported(_)) => host(),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// バックエンド出力が契約どおりの長さ（`expected` と完全一致）の 1 次元であることを検証する。
fn verify_vector_len(shape: &[usize], expected: usize) -> Result<(), AutodiffError> {
    match shape {
        [n] if *n == expected => Ok(()),
        _ => Err(shape_mismatch(shape, &[expected])),
    }
}

/// ヒストグラム（`torch.histc` 相当）。入力を平坦化し、`[min, max]` を `bins` 等分したビンの
/// 要素数（`f32`）を返す。
///
/// `min == max` のときは入力の最小・最大（なお等しければ ±1）を範囲とする。範囲外・NaN 要素は
/// 無視し、`x == max` は最終ビン。`bins == 0`・`min > max`・範囲が非有限は型付きエラー
/// （実体化・確保より前に検査できるものは確保前）。
pub fn histc<'t>(
    x: &Var<'t>,
    bins: usize,
    min: f32,
    max: f32,
) -> Result<Tensor<f32>, AutodiffError> {
    binning::histc_check(&x.shape(), bins, min, max)?;
    let input = materialize_one(x)?;
    let backend = x.tape().ops().binning_histc(&input, bins, min, max);
    resolve(
        backend,
        |shape| {
            if shape == [bins] {
                Ok(())
            } else {
                Err(shape_mismatch(shape, &[bins]))
            }
        },
        || {
            let v = binning::histc_host(&input.contiguous().host_slice(), bins, min, max)?;
            Tensor::new(v, &[bins]).map_err(AutodiffError::Shape)
        },
    )
}

/// 重みなしの度数カウント（`torch.bincount(input, minlength=…)` 相当）。
///
/// `input` は 1 次元・非負のみ。出力長は `max(max(input) + 1, minlength)`、空入力は長さ
/// `minlength` の零。カウントは `i32`（入力長が `i32::MAX` を超える場合は型付きエラー）。
/// `Var` を取らないため `BackendOps` へ到達する `tape` を明示する。
pub fn bincount(
    tape: &Tape,
    input: &Tensor<i32>,
    minlength: usize,
) -> Result<Tensor<i32>, AutodiffError> {
    binning::bincount_check(input.shape(), false)?;
    // 負値拒否と契約上の出力長の算出をバックエンド呼び出し前に行う（全バックエンド共通の検証）。
    let expected = binning::bincount_out_len(&input.contiguous().host_slice(), minlength)?;
    let backend = tape.ops().binning_bincount(input, minlength);
    resolve(
        backend,
        |shape| verify_vector_len(shape, expected),
        || {
            let v = binning::bincount_host(&input.contiguous().host_slice(), minlength)?;
            let n = v.len();
            Tensor::new(v, &[n]).map_err(AutodiffError::Shape)
        },
    )
}

/// 重み付きの度数カウント（`torch.bincount(input, weights, minlength)` 相当）。
///
/// ビンごとに `f64` アキュムレータへ入力順に加算して最後に 1 回 `f32` へ downcast する。
/// `weights` は 1 次元で `input` と同長（空入力は重みを見ずに長さ `minlength` の零）。
/// 出力は detached（重みへの勾配は流れない。PyTorch も backward が未実装）。`weights` が
/// `tape` と別の tape の `Var` なら [`AutodiffError::TapeMismatch`]。
pub fn bincount_weighted<'t>(
    tape: &'t Tape,
    input: &Tensor<i32>,
    weights: &Var<'t>,
    minlength: usize,
) -> Result<Tensor<f32>, AutodiffError> {
    if weights.tape_id() != tape.id {
        return Err(AutodiffError::TapeMismatch);
    }
    let n = binning::bincount_check(input.shape(), true)?;
    let expected = binning::bincount_out_len(&input.contiguous().host_slice(), minlength)?;
    if n == 0 {
        // 空入力は重みの rank・長さを見ず、実体化もせず minlength 個の零を返す（PyTorch 同様）。
        let zeros = binning::bincount_zeros_f32(expected)?;
        return Tensor::new(zeros, &[expected]).map_err(AutodiffError::Shape);
    }
    // 重みの実体化・バックエンド呼び出しより前に rank・長さ不一致を拒否する。
    let wshape = weights.shape();
    binning::bincount_check(&wshape, true)?;
    binning::bincount_weights_check(n, &wshape)?;
    let w = materialize_one(weights)?;
    let backend = tape.ops().binning_bincount_weighted(input, &w, minlength);
    resolve(
        backend,
        |shape| verify_vector_len(shape, expected),
        || {
            let v = binning::bincount_weighted_host(
                &input.contiguous().host_slice(),
                &w.contiguous().host_slice(),
                minlength,
            )?;
            let n = v.len();
            Tensor::new(v, &[n]).map_err(AutodiffError::Shape)
        },
    )
}

/// 共通後半。`layout` 検査済みの 2 つの `Var` を実体化して探索する。
fn search_resolved<'t>(
    seq: &Var<'t>,
    values: &Var<'t>,
    layout: &SearchSortedLayout,
    right: bool,
) -> Result<Tensor<i32>, AutodiffError> {
    let s = materialize_one(seq)?;
    let v = materialize_one(values)?;
    let backend = seq.tape().ops().binning_searchsorted(&s, &v, right);
    resolve(
        backend,
        |shape| {
            if shape == layout.out_shape() {
                Ok(())
            } else {
                Err(shape_mismatch(shape, layout.out_shape()))
            }
        },
        || {
            let out = binning::searchsorted_host(
                &s.contiguous().host_slice(),
                &v.contiguous().host_slice(),
                layout,
                right,
            )?;
            Tensor::new(out, layout.out_shape()).map_err(AutodiffError::Shape)
        },
    )
}

/// 昇順列への挿入位置（`torch.searchsorted(sorted_sequence, values, right=…)` 相当）。
///
/// `right = false` は下限（`seq[i-1] < v <= seq[i]`）、`true` は上限。`sorted_sequence` は
/// rank 1 なら `values` は任意 shape（0 次元可）、rank N ≥ 2 なら `values` は同 rank で
/// 先頭 N−1 軸が一致する。出力は `values` と同 shape の `i32` 索引。未ソート列は検査せず
/// （PyTorch と同じ）、結果は未規定だが決定的で常に `[0, 列長]` 内に収まる。2 つの `Var` が
/// 別 tape なら shape 検査より前に [`AutodiffError::TapeMismatch`]。
pub fn searchsorted<'t>(
    sorted_sequence: &Var<'t>,
    values: &Var<'t>,
    right: bool,
) -> Result<Tensor<i32>, AutodiffError> {
    sorted_sequence.check_same_tape(values)?;
    let layout = binning::searchsorted_layout(&sorted_sequence.shape(), &values.shape())?;
    search_resolved(sorted_sequence, values, &layout, right)
}

/// 境界値によるバケット化（`torch.bucketize(input, boundaries, right=…)` 相当）。
///
/// `boundaries` は 1 次元必須。`searchsorted(boundaries, input, right)` と同値（1 次元特例）。
/// 別 tape なら shape 検査より前に [`AutodiffError::TapeMismatch`]。
pub fn bucketize<'t>(
    input: &Var<'t>,
    boundaries: &Var<'t>,
    right: bool,
) -> Result<Tensor<i32>, AutodiffError> {
    input.check_same_tape(boundaries)?;
    let layout = binning::bucketize_layout(&boundaries.shape(), &input.shape())?;
    search_resolved(boundaries, input, &layout, right)
}
