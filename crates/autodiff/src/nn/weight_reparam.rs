//! Conv 系 6 層（`nn/conv.rs`）が共有する weight 再パラメータ化の補助（イシュー #3049。
//! 親 #3018。決定記録 `docs/facade-weight-reparam-wiring-decision.md` §5〜§10・§13 の
//! 「(a) 層内再パラメータ化」）。
//!
//! 役割: `Linear`（#3048。`nn/linear.rs` が同型の実装を直接持つ）と同じ状態機械を、rank 3〜5 の
//! Conv weight 向けに 1 箇所へ集約し、6 層で同一ロジックを複製しない。呼び出し元は
//! `conv.rs` の `impl_conv_reparam!` マクロ（各層の `parametrize_*`・`forward_bound`・
//! `set_parameter` が本モジュールを呼ぶ）のみ。有効重みの再構成は `Var::weight_norm`／
//! `Var::spectral_norm`（REQ-9 の薄いラッパー方針。迂回実装を作らない。`security.md` A08）へ一本化する。
//!
//! 命名上の注意: `crates/facade/src/lib.rs` のインベントリ検査が `fn weight_norm`／
//! `fn spectral_norm`／`fn norm_except_dim` の宣言位置を固定しているため、本モジュールの
//! 関数名はこれらと完全一致させない。

use std::cell::RefCell;

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::init::{SPECTRAL_U_SEED_SALT, SPECTRAL_V_SEED_SALT, derive_seed, uniform_init};
use crate::var::Var;
use crate::weight_reparam_ops::{
    SPECTRAL_NORM_INIT_POWER_ITERATIONS, SpectralNormState, norm_except_dim,
};

/// Conv 層の `weight` スロットの再パラメータ化状態。
///
/// - `WeightNorm`: `weight` スロットが `v`、ここが `g`（`dim` 以外が 1 の keepdim 形、
///   `dim = None` は rank 0）。
/// - `SpectralNorm`: `weight` スロットが `original`。状態は forward が `&self` のため
///   `RefCell`（`Linear` と同じ）。
pub(crate) enum WeightReparam {
    WeightNorm { g: Tensor<f32>, dim: Option<usize> },
    SpectralNorm { state: RefCell<SpectralNormState> },
}

/// weight_norm 状態を作る。`g = norm_except_dim(weight, dim)` のため適用直後の有効重みは元の重みと
/// 一致する。`dim >= rank` は `norm_except_dim` が型付きエラーで拒否する。層は変更しない純粋計算。
pub(crate) fn build_weight_norm_state(
    weight: &Tensor<f32>,
    dim: Option<usize>,
) -> Result<WeightReparam, AutodiffError> {
    let g = norm_except_dim(weight, dim)?;
    Ok(WeightReparam::WeightNorm { g, dim })
}

/// spectral_norm 状態を作る。`spectral_dim` は層から導出する行軸（Conv 系 0・ConvTranspose 系 1。
/// PyTorch と一致）。`u0`（長さ `shape[dim]`）・`v0`（長さ `numel / shape[dim]`）は自層の shape から
/// 純粋算術で導出し、`seed` から決定的に初期化したうえで `SPECTRAL_NORM_INIT_POWER_ITERATIONS` 回の
/// 予備反復を行う（PyTorch の初期化と同じ）。`n_power_iterations`・`eps` の検証は
/// `SpectralNormState::from_vectors` が行う。
pub(crate) fn build_spectral_norm_state(
    weight: &Tensor<f32>,
    spectral_dim: usize,
    n_power_iterations: usize,
    eps: f32,
    seed: u64,
) -> Result<WeightReparam, AutodiffError> {
    let shape = weight.shape();
    let rows = shape.get(spectral_dim).copied().unwrap_or(0);
    let numel: usize = shape.iter().product();
    if rows == 0 || numel == 0 {
        return Err(AutodiffError::Shape(ShapeError::ElementCountMismatch {
            expected: 1,
            actual: 0,
        }));
    }
    let cols = numel / rows;
    let bound = 1.0 / (cols.max(1) as f32).sqrt();
    let u0 = uniform_init(rows, bound, derive_seed(seed, SPECTRAL_U_SEED_SALT));
    let v0 = uniform_init(cols, bound, derive_seed(seed, SPECTRAL_V_SEED_SALT));
    let mut state =
        SpectralNormState::from_vectors(shape, spectral_dim, &u0, &v0, n_power_iterations, eps)?;
    state.power_iterate(weight, SPECTRAL_NORM_INIT_POWER_ITERATIONS)?;
    Ok(WeightReparam::SpectralNorm {
        state: RefCell::new(state),
    })
}

/// 有効重みを tape 上で再構成する。spectral_norm は更新後状態を第 2 要素で返す（呼び出し元が
/// forward 全体の成功後に [`commit_spectral_state`] でコミットする。失敗した forward では
/// 状態が進まない。決定記録 §8）。
pub(crate) fn effective_weight_var<'t>(
    reparam: &WeightReparam,
    raw: &Var<'t>,
    g: Option<&Var<'t>>,
    training: bool,
    label: &str,
) -> Result<(Var<'t>, Option<SpectralNormState>), AutodiffError> {
    match reparam {
        WeightReparam::WeightNorm { dim, .. } => {
            let g = g.ok_or_else(|| {
                AutodiffError::InvalidArgument(format!(
                    "{label}::forward_bound: weight_norm の g が bind されていない"
                ))
            })?;
            Ok((raw.weight_norm(g, *dim)?, None))
        }
        WeightReparam::SpectralNorm { state } => {
            let st = state.try_borrow().map_err(|_| reentrant_error(label))?;
            let mut next = st.clone();
            let w = raw.spectral_norm(&mut next, training)?;
            Ok((w, Some(next)))
        }
    }
}

/// [`effective_weight_var`] が返した更新後状態を層へコミットする（`try_borrow_mut` 失敗は再入として
/// `InvalidArgument`）。
pub(crate) fn commit_spectral_state(
    reparam: &WeightReparam,
    pending: Option<SpectralNormState>,
    label: &str,
) -> Result<(), AutodiffError> {
    if let (Some(next), WeightReparam::SpectralNorm { state }) = (pending, reparam) {
        let mut st = state.try_borrow_mut().map_err(|_| reentrant_error(label))?;
        *st = next;
    }
    Ok(())
}

/// 保存済みの spectral_norm `u`／`v` を bit のまま復元する（イシュー #3050。層の
/// `restore_spectral_norm_vectors` から呼ばれ、`facade::compat::model_io` の load が使う）。
///
/// 層の現在の `n_power_iterations`／`eps`／`dim` を保持し、`u`／`v` だけを差し替える。新状態を
/// 先に構築してから代入する（construct-before-assign）ため、検査失敗時は状態不変。spectral_norm
/// 未適用の層は `InvalidArgument`。
pub(crate) fn restore_spectral_vectors(
    state: Option<&RefCell<SpectralNormState>>,
    weight: &Tensor<f32>,
    u: &[f32],
    v: &[f32],
    label: &str,
) -> Result<(), AutodiffError> {
    let Some(state) = state else {
        return Err(AutodiffError::InvalidArgument(format!(
            "{label}::restore_spectral_norm_vectors: spectral_norm が適用されていない"
        )));
    };
    let mut st = state.try_borrow_mut().map_err(|_| reentrant_error(label))?;
    let next = SpectralNormState::from_exact_parts(
        weight.shape(),
        st.dim(),
        u,
        v,
        st.n_power_iterations(),
        st.eps(),
    )?;
    *st = next;
    Ok(())
}

fn reentrant_error(label: &str) -> AutodiffError {
    AutodiffError::InvalidArgument(format!(
        "{label}::forward_bound: spectral_norm 状態が使用中（再入）"
    ))
}

/// `named_parameters` が先頭に出すキー列（再パラメータ化済み）。`weight` の代わりに
/// weight_norm は `original0`（g）→`original1`（v）、spectral_norm は `original`。
pub(crate) fn reparam_weight_keys(reparam: &WeightReparam) -> &'static [&'static str] {
    match reparam {
        WeightReparam::WeightNorm { .. } => &[
            "parametrizations.weight.original0",
            "parametrizations.weight.original1",
        ],
        WeightReparam::SpectralNorm { .. } => &["parametrizations.weight.original"],
    }
}

/// shape 保存置換の共通処理。shape 不一致は `ShapeMismatch`。
pub(crate) fn replace_same_shape(
    slot: &mut Tensor<f32>,
    value: Tensor<f32>,
) -> Result<(), AutodiffError> {
    if value.shape() != slot.shape() {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: value.shape().to_vec(),
            rhs: slot.shape().to_vec(),
        }));
    }
    *slot = value;
    Ok(())
}

/// 再パラメータ化済み層の `set_parameter`。分解パラメータのキーを受理すると `Ok(None)`、`weight` は
/// 拒否する。それ以外のキー（`bias` 等）は `value` を `Ok(Some(value))` で呼び出し元へ返す。
pub(crate) fn set_reparam_parameter(
    reparam: &mut WeightReparam,
    weight: &mut Tensor<f32>,
    name: &str,
    value: Tensor<f32>,
    label: &str,
) -> Result<Option<Tensor<f32>>, AutodiffError> {
    match (reparam, name) {
        (WeightReparam::WeightNorm { g, .. }, "parametrizations.weight.original0") => {
            replace_same_shape(g, value)?;
            Ok(None)
        }
        (WeightReparam::WeightNorm { .. }, "parametrizations.weight.original1")
        | (WeightReparam::SpectralNorm { .. }, "parametrizations.weight.original") => {
            replace_same_shape(weight, value)?;
            Ok(None)
        }
        (_, "weight") => Err(AutodiffError::InvalidArgument(format!(
            "{label}::set_parameter: `weight` is parametrized; use the `parametrizations.weight.*` keys"
        ))),
        _ => Ok(Some(value)),
    }
}
