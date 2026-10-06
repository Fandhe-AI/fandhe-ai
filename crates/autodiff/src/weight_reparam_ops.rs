//! 重み再パラメータ化の自由関数（`weight_norm`・`norm_except_dim`・`spectral_norm`）と内部型
//! [`SpectralNormState`]（イシュー #2646・親 #2625「Phase 4」・ルート #2499）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::weight_norm`／`Var::spectral_norm` の委譲メソッドと
//! `SpectralNormState` の公開位置）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
//! 層化（`Linear`／`Conv` の重みへの parametrization 結線・`Sequential::add_*`・保存復元）は #2679 の
//! 対象で本イシューでは作らない。本モジュールは内部クレート限定の入口で、`Var` に inherent メソッドを
//! 足さない。保留は `crates/facade/src/lib.rs` の `LrnWeightReparamHoldDoctestGuard` と
//! `crates/facade/tests/api_surface.rs` の否定ガードが機械的に固定する
//! （`docs/autodiff-lrn-weight-reparam-decision.md`）。
//!
//! **PyTorch 相当**:
//!
//! | 入口 | PyTorch 相当 |
//! |---|---|
//! | [`weight_norm`] | `torch._weight_norm(v, g, dim)`（`w = v·(g/‖v‖)`。`dim = None` は `dim=-1`） |
//! | [`norm_except_dim`] | `torch.norm_except_dim(v, 2, dim)`（`g` の初期値。非微分） |
//! | [`spectral_norm`] | `parametrizations.spectral_norm` の `_SpectralNorm.forward`（`_power_method` を含む） |
//!
//! **経路**: ① 引数・形状・状態整合の検査（実体化・tape 操作・状態更新より前。エラー時に孤児ノードを
//! 残さない）→ ② 入力の実体化 → ③ `BackendOps::{weight_norm_forward, spectral_norm_forward}`
//! （`Unsupported` のときだけ共有ホストカーネル `fandhe_ai_tensor_core::weight_reparam` へフォール
//! バックし、他のエラーは伝播する。戻り shape も検証する）→ ④ 専用 `Op`（`Op::WeightNorm`／
//! `Op::SpectralNorm`）を積む。VJP は `grad.rs`。`spectral_norm` は更新後の `u`／`v` を**ローカルに**
//! 計算して forward に使い、forward が成功してから（`push_eager` の直前）状態へ書き戻す
//! （forward が失敗しても状態だけ進まない）。
//!
//! **既存 Op の合成にしない理由**: `v ⊙ (g/‖v‖)`・`W/σ` の合成では broadcast 付き `Op::Mul`／`Op::Div` の
//! VJP が純 `f32` 逐次和（`reduce_to_shape`）で縮約するため、`dg`／`dσ` の長軸縮約が `f64`
//! アキュムレータ契約（`.claude/rules/coding-rust.md`）に抵触する。専用 `Op` と共有ホストカーネルとした。
//!
//! **PyTorch との意図的な差分**: rank 1 の `spectral_norm`（PyTorch は `F.normalize` に縮退）・負の
//! `dim`・`u`／`v` の乱数初期化（呼び出し側が `power_iterate` で再現する）は対象外。詳細は決定記録 §5。

use fandhe_ai_tensor_core::weight_reparam::{self as wr, SpectralLayout};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// PyTorch の `_SpectralNorm.__init__` が乱数初期化後に行う予備反復回数。呼び出し側が乱数 `u`／`v` から
/// `SpectralNormState::power_iterate(weight, SPECTRAL_NORM_INIT_POWER_ITERATIONS)` を呼ぶことで
/// PyTorch の初期化を再現できる（本クレートは乱数初期化を持たない）。
pub const SPECTRAL_NORM_INIT_POWER_ITERATIONS: usize = 15;

fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

/// `torch.norm_except_dim(v, 2, dim)`。`dim = Some(d)` は `d` 以外の全軸の L2（出力は `d` 以外が 1 の
/// keepdim 形。`weight_norm` の `g` の初期値）、`None` はテンソル全体（出力は rank 0）。非微分・tape
/// 非依存。rank 0 の入力・`dim >= rank` は型付きエラー。
pub fn norm_except_dim(v: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, AutodiffError> {
    let layout = wr::norm_except_dim_layout(v.shape(), dim).map_err(AutodiffError::Shape)?;
    let data = wr::norm_except_dim_host(&v.contiguous().host_slice(), &layout)
        .map_err(AutodiffError::Shape)?;
    Tensor::new(data, layout.g_shape()).map_err(AutodiffError::Shape)
}

/// weight_norm（`w = v·(g/‖v‖)`。`‖v‖` は `dim` 以外の全軸の L2、`dim = None` はテンソル全体）。
///
/// `g` の shape は [`norm_except_dim`] の出力 shape と**完全一致のみ**受理する（`Some(dim)` は keepdim 形
/// 〈例 `[C, 1, 1, 1]`〉・`None` は rank 0）。検査順: 同一 tape → レイアウト（rank・`dim`・`g` shape・
/// 確保サイズ）→ 実体化 → バックエンド／ホスト → 戻り shape 再検証 → `push_eager`。`‖v‖ = 0` は拒否せず
/// 伝播する。
pub fn weight_norm<'t>(
    v: &Var<'t>,
    g: &Var<'t>,
    dim: Option<usize>,
) -> Result<Var<'t>, AutodiffError> {
    v.check_same_tape(g)?;
    let layout =
        wr::weight_norm_layout(&v.shape(), &g.shape(), dim).map_err(AutodiffError::Shape)?;
    let v_val = materialize_one(v)?;
    let g_val = materialize_one(g)?;
    let value = match v.tape().ops().weight_norm_forward(&v_val, &g_val, dim) {
        Ok(out) => {
            verify_shape(out.shape(), layout.shape())?;
            out
        }
        Err(BackendError::Unsupported(_)) => {
            let data = wr::weight_norm_host(
                &v_val.contiguous().host_slice(),
                &g_val.contiguous().host_slice(),
                &layout,
            )
            .map_err(AutodiffError::Shape)?;
            Tensor::new(data, layout.shape()).map_err(AutodiffError::Shape)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = v.tape().push_eager(
        Op::WeightNorm {
            v: v.node_id(),
            g: g.node_id(),
            dim,
        },
        value,
    );
    Ok(Var::from_raw(v.tape(), id))
}

/// spectral_norm の非追跡状態（`u`: `[h]`・`v`: `[w]`）と設定。`h = shape[dim]`・`w = numel / h`。
///
/// **最初から `#[non_exhaustive]`・フィールド非公開・アクセサのみ**（後日の公開承認で形を変えずに済む
/// ようにするため）。乱数初期化は持たない（`Generator` の公開形が承認待ちのため結合しない）。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralNormState {
    weight_shape: Vec<usize>,
    u: Vec<f32>,
    v: Vec<f32>,
    dim: usize,
    eps: f32,
    n_power_iterations: usize,
}

impl SpectralNormState {
    /// `u0`／`v0` を `x / max(‖x‖, eps)`（`F.normalize` と同じ。`+ eps` ではない）で正規化して保持する。
    /// 検査: rank 2 以上・`dim < rank`・要素数 0 拒否（`spectral_norm_layout`）、`n_power_iterations >= 1`、
    /// `eps` が有限かつ非負、`u0.len() == h`・`v0.len() == w`、`u0`／`v0` が有限。
    pub fn from_vectors(
        weight_shape: &[usize],
        dim: usize,
        u0: &[f32],
        v0: &[f32],
        n_power_iterations: usize,
        eps: f32,
    ) -> Result<Self, AutodiffError> {
        let layout = wr::spectral_norm_layout(weight_shape, dim).map_err(AutodiffError::Shape)?;
        if n_power_iterations == 0 {
            return Err(AutodiffError::InvalidArgument(
                "SpectralNormState::from_vectors: n_power_iterations は 1 以上である必要がある"
                    .into(),
            ));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "SpectralNormState::from_vectors: eps は有限かつ非負である必要がある（eps={eps}）"
            )));
        }
        for (name, vec, want) in [("u0", u0, layout.rows()), ("v0", v0, layout.cols())] {
            if vec.len() != want {
                return Err(AutodiffError::Shape(ShapeError::ElementCountMismatch {
                    expected: want,
                    actual: vec.len(),
                }));
            }
            if vec.iter().any(|x| !x.is_finite()) {
                return Err(AutodiffError::InvalidArgument(format!(
                    "SpectralNormState::from_vectors: {name} は有限である必要がある"
                )));
            }
        }
        Ok(Self {
            weight_shape: weight_shape.to_vec(),
            u: wr::normalize_vector_host(u0, eps),
            v: wr::normalize_vector_host(v0, eps),
            dim,
            eps,
            n_power_iterations,
        })
    }

    /// 重みの shape。
    pub fn weight_shape(&self) -> &[usize] {
        &self.weight_shape
    }

    /// 左特異ベクトルの推定値（長さ `h`）。
    pub fn u(&self) -> &[f32] {
        &self.u
    }

    /// 右特異ベクトルの推定値（長さ `w`）。
    pub fn v(&self) -> &[f32] {
        &self.v
    }

    /// `W_mat` の行軸 `dim`。
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// `F.normalize` の `eps`。
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// training の forward 1 回あたりの反復回数。
    pub fn n_power_iterations(&self) -> usize {
        self.n_power_iterations
    }

    fn layout_for(&self, weight_shape: &[usize]) -> Result<SpectralLayout, AutodiffError> {
        if weight_shape != self.weight_shape.as_slice() {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: weight_shape.to_vec(),
                rhs: self.weight_shape.clone(),
            }));
        }
        wr::spectral_norm_layout(weight_shape, self.dim).map_err(AutodiffError::Shape)
    }

    /// `u ← normalize(W v)` → `v ← normalize(Wᵀ u)` を `n` 回進める（**u が先**。tape に載せない）。
    /// 行列ベクトル積・ノルムは `f64` 蓄積・各反復の `u`／`v` は `f32` で保持する。`weight` の shape が
    /// 状態と一致しなければ型付きエラー（状態は変わらない）。
    pub fn power_iterate(&mut self, weight: &Tensor<f32>, n: usize) -> Result<(), AutodiffError> {
        let layout = self.layout_for(weight.shape())?;
        let mut u = self.u.clone();
        let mut v = self.v.clone();
        wr::spectral_power_iterate_host(
            &weight.contiguous().host_slice(),
            &layout,
            &mut u,
            &mut v,
            n,
            self.eps,
        )
        .map_err(AutodiffError::Shape)?;
        self.u = u;
        self.v = v;
        Ok(())
    }
}

/// spectral_norm（`out = W / σ`、`σ = uᵀ W_mat v`）。
///
/// `training = true` のときだけ `state.n_power_iterations()` 回の power iteration を行って状態を更新する
/// （`eval` は状態不変）。`u`／`v` は非追跡で、勾配は `weight` のみへ流れる（PyTorch の buffer clone と
/// 同じ）。検査順: `state` と `weight` の shape 整合（`dim`・rank・要素数 0 拒否を含む）→ 実体化 →
/// 反復（ローカル）→ バックエンド／ホスト → 戻り shape 再検証 → 状態書き戻し → `push_eager`。
/// いずれかの検査・forward が失敗した場合 `state` は変更されない。`σ = 0` は拒否せず伝播する。
pub fn spectral_norm<'t>(
    weight: &Var<'t>,
    state: &mut SpectralNormState,
    training: bool,
) -> Result<Var<'t>, AutodiffError> {
    let w_shape = weight.shape();
    let layout = state.layout_for(&w_shape)?;
    let w_val = materialize_one(weight)?;
    let (u_vec, v_vec) = if training {
        let mut u = state.u.clone();
        let mut v = state.v.clone();
        wr::spectral_power_iterate_host(
            &w_val.contiguous().host_slice(),
            &layout,
            &mut u,
            &mut v,
            state.n_power_iterations,
            state.eps,
        )
        .map_err(AutodiffError::Shape)?;
        (u, v)
    } else {
        (state.u.clone(), state.v.clone())
    };
    let u_t = Tensor::new(u_vec.clone(), &[layout.rows()]).map_err(AutodiffError::Shape)?;
    let v_t = Tensor::new(v_vec.clone(), &[layout.cols()]).map_err(AutodiffError::Shape)?;
    let value = match weight
        .tape()
        .ops()
        .spectral_norm_forward(&w_val, &u_t, &v_t, state.dim)
    {
        Ok(out) => {
            verify_shape(out.shape(), layout.shape())?;
            out
        }
        Err(BackendError::Unsupported(_)) => {
            let data =
                wr::spectral_norm_host(&w_val.contiguous().host_slice(), &layout, &u_vec, &v_vec)
                    .map_err(AutodiffError::Shape)?;
            Tensor::new(data, layout.shape()).map_err(AutodiffError::Shape)?
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    // forward が成功してから状態を進める（以降に失敗しうる処理を置かない）。
    if training {
        state.u = u_vec;
        state.v = v_vec;
    }
    let id = weight.tape().push_eager(
        Op::SpectralNorm {
            weight: weight.node_id(),
            u: u_t,
            v: v_t,
            dim: state.dim,
        },
        value,
    );
    Ok(Var::from_raw(weight.tape(), id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    #[test]
    fn weight_norm_hand_computed() {
        let tape = Tape::new();
        let v = tape.var(&t(vec![3.0, 4.0, 0.0, 2.0], &[2, 2]));
        let g = tape.var(&t(vec![10.0, 1.0], &[2, 1]));
        let w = weight_norm(&v, &g, Some(0)).unwrap();
        assert_eq!(
            w.to_tensor().host_slice().into_owned(),
            vec![6.0, 8.0, 0.0, 1.0]
        );
    }

    #[test]
    fn norm_except_dim_shapes() {
        let v = t((0..24).map(|i| i as f32).collect(), &[2, 3, 4]);
        assert_eq!(norm_except_dim(&v, Some(1)).unwrap().shape(), &[1, 3, 1]);
        let whole = norm_except_dim(&v, None).unwrap();
        assert!(whole.shape().is_empty());
        assert!(norm_except_dim(&v, Some(3)).is_err());
    }

    #[test]
    fn weight_norm_rejects_bad_g_without_orphan_nodes() {
        let tape = Tape::new();
        let v = tape.var(&t(vec![1.0; 4], &[2, 2]));
        let g_bad = tape.var(&t(vec![1.0; 2], &[2]));
        let before = tape.len();
        assert!(weight_norm(&v, &g_bad, Some(0)).is_err());
        assert_eq!(tape.len(), before);
        let other = Tape::new();
        let g_other = other.var(&t(vec![1.0; 2], &[2, 1]));
        assert!(matches!(
            weight_norm(&v, &g_other, Some(0)),
            Err(AutodiffError::TapeMismatch)
        ));
    }

    #[test]
    fn spectral_state_validation_and_training_vs_eval() {
        let w = t(vec![3.0, 0.0, 0.0, 1.0], &[2, 2]);
        assert!(
            SpectralNormState::from_vectors(&[2, 2], 0, &[1.0], &[1.0, 1.0], 1, 1e-12).is_err()
        );
        assert!(
            SpectralNormState::from_vectors(&[2, 2], 0, &[1.0, 1.0], &[1.0, 1.0], 0, 1e-12)
                .is_err()
        );
        assert!(
            SpectralNormState::from_vectors(&[2, 2], 0, &[f32::NAN, 1.0], &[1.0, 1.0], 1, 1e-12)
                .is_err()
        );
        let mut st =
            SpectralNormState::from_vectors(&[2, 2], 0, &[1.0, 1.0], &[1.0, 1.0], 20, 1e-12)
                .unwrap();
        let tape = Tape::new();
        let wv = tape.var(&w);
        let before = st.clone();
        let _ = spectral_norm(&wv, &mut st, false).unwrap();
        assert_eq!(st, before, "eval は状態不変");
        let out = spectral_norm(&wv, &mut st, true).unwrap();
        assert_ne!(st, before, "training は状態を更新する");
        let o = out.to_tensor().host_slice().into_owned();
        assert!((o[0] - 1.0).abs() < 1e-5, "{o:?}");
    }

    #[test]
    fn spectral_errors_do_not_advance_state() {
        let tape = Tape::new();
        let mut st =
            SpectralNormState::from_vectors(&[2, 2], 0, &[1.0, 1.0], &[1.0, 0.0], 3, 1e-12)
                .unwrap();
        let before = st.clone();
        let wrong = tape.var(&t(vec![1.0; 6], &[2, 3]));
        let n = tape.len();
        assert!(spectral_norm(&wrong, &mut st, true).is_err());
        assert_eq!(st, before);
        assert_eq!(tape.len(), n);
    }
}
