//! `eigh`・`slogdet`・`pinv`・`matrix_rank`・`lstsq` の 5 線形代数演算
//! （イシュー #2150・親 #2131「Tier 2 追加分」）。
//!
//! **facade 非公開（意図的）**: `crate::reduce_ops`／`crate::matrix_ops`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。イシュー
//! #2150 本文は facade 公開面（`Var::eigh` 等の委譲メソッド）を承認
//! 事項として明示するため、承認が取れるまでは自由関数として `Var` の
//! 外に置き到達不能にする（`docs/autodiff-linalg-ops-decision.md`
//! §0）。承認後は `Var::eigh` 等の薄い委譲メソッドを追加し、facade
//! 側の保留ガード（`crates/facade/src/lib.rs::
//! VarLinalgOpsHoldDoctestGuard`）を撤去する。
//!
//! **PyTorch 相当**（詳細は `docs/autodiff-linalg-ops-decision.md` §1）:
//!
//! | 演算 | PyTorch 相当 | 出力 |
//! |---|---|---|
//! | [`eigh`] | `torch.linalg.eigh(UPLO='L')` | [`EighVars`]（`eigenvalues: [n]`・`eigenvectors: [n,n]`） |
//! | [`slogdet`] | `torch.linalg.slogdet` | [`SlogdetVars`]（`sign: []`・`logabsdet: []`） |
//! | [`pinv`] | `torch.linalg.pinv` | `Var`（`[n,m]`） |
//! | [`matrix_rank`] | `torch.linalg.matrix_rank` | `Var`（`[]`。非負整数を表す f32・**勾配ゼロ**） |
//! | [`lstsq`] | `torch.linalg.lstsq` | `Var`（`[n,k]`。最小ノルム解のみ。residuals／rank／singular_values は返さない） |
//!
//! **新規 `Op` の有無**: 5 演算とも専用 `Op`（`crate::tape::Op::
//! EighValues`／`EighVectors`／`SlogdetSign`／`SlogdetLogAbsDet`／
//! `Pinv`／`Lstsq`／`MatrixRank`）を持つ。合成（`svd → gt/where →
//! ...`）は使わない——既存の `eval::linalg::svd_vjp` はどのコタンジェ
//! ントが `Some` でも上流がゼロでも `|σ_j²-σ_i²| < 1e-9` を無条件に
//! `InvalidArgument` として拒否するため、rank 落ちの入力（`pinv`／
//! `matrix_rank` の主要な用途）で backward が失敗してしまう
//! （`docs/autodiff-linalg-ops-decision.md` §2「判断 2」）。
//!
//! **数値契約**（詳細は `docs/autodiff-linalg-ops-decision.md` §3）:
//! - 内部精度は `eval::linalg`（`f64` アキュムレータ）と同じ。
//! - `rcond`（`pinv`／`matrix_rank`／`lstsq` の引数）は**アルゴリズムの
//!   引数であり REQ-2 の tolerance ではない**（`eval::linalg::
//!   resolve_rcond` doc）。`eigh` の固有値縮退判定閾値
//!   （`eval::linalg::EIGH_DEGENERATE_RTOL`）も同じ扱い。
//! - `eigh`・`slogdet`・`pinv`・`lstsq` の各入口は非有限入力
//!   （`NaN`／`Inf`）を検査し `AutodiffError::InvalidArgument` とする
//!   （PyTorch は非有限を伝播またはエラーにする。差分として
//!   `docs/autodiff-linalg-ops-decision.md` §1 に記録）。
//!
//! **確保前のバイト数上限検査**: 5 入口すべてが冒頭で `checked_bytes_
//! for::<f32>`（入力 shape。`f64` 換算は `Mat` 内部の実装詳細だが、
//! 同じ要素数を確保する以上入力 shape の検査で足りる——`reduce_ops::
//! ensure_alloc_fits_f32` と同じ考え方）を呼び、あらゆる分岐・実体化
//! よりも前に確保不能な巨大 shape を拒否する。

use fandhe_ai_tensor_core::{BackendError, EighFactors, ShapeError, SlogdetFactors, Tensor};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::eval;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// [`eigh`] の戻り値（多出力。`Var::qr`〈`var.rs::QrVars`〉と同型）。
/// `Copy` にはしない——`Var<'t>` 自体が `Copy` ではないため
/// （`QrVars`／`SvdVars` も非 `Copy`）。
#[derive(Debug, Clone)]
pub struct EighVars<'t> {
    /// `[n]`（昇順・同値は安定順）。
    pub eigenvalues: Var<'t>,
    /// `[n, n]`（列が固有ベクトル）。
    pub eigenvectors: Var<'t>,
}

/// [`slogdet`] の戻り値（多出力）。
#[derive(Debug, Clone)]
pub struct SlogdetVars<'t> {
    /// `[]`（`-1.0`／`0.0`／`1.0`）。
    pub sign: Var<'t>,
    /// `[]`（`ln|det A|`。特異なら `-inf`）。
    pub logabsdet: Var<'t>,
}

/// バックエンド実装の戻り値 shape を検証する（`var.rs::verify_shape`
/// と同型の複製。`reduce_ops::verify_shape` と同じ理由——`var.rs` 側は
/// `pub(crate)` 化されていない `fn` のため呼べない）。
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

/// CPU 本番経路の `BackendError` を `AutodiffError` へ写像する
/// （`var.rs::unify_backend_error` と同型の複製）。
fn unify_backend_error(err: BackendError) -> AutodiffError {
    match err {
        BackendError::InvalidArgument(msg) => AutodiffError::InvalidArgument(msg),
        other => AutodiffError::Backend(other),
    }
}

/// 本モジュールの全公開入口が冒頭で呼ぶ、唯一の確保前バイト数上限
/// 検査ヘルパ（`reduce_ops::ensure_alloc_fits_f32` と同じ規律）。
fn ensure_alloc_fits_f32(shapes: &[&[usize]]) -> Result<(), AutodiffError> {
    for shape in shapes {
        checked_bytes_for::<f32>(shape)?;
    }
    Ok(())
}

fn require_rank2(shape: &[usize], op_name: &str) -> Result<(usize, usize), AutodiffError> {
    if shape.len() != 2 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: shape.len(),
        }));
    }
    let _ = op_name;
    Ok((shape[0], shape[1]))
}

fn require_square(shape: &[usize], op_name: &str) -> Result<usize, AutodiffError> {
    let (m, n) = require_rank2(shape, op_name)?;
    if m != n {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: 正方行列（[n,n]）が必要（形状 {shape:?}）"
        )));
    }
    Ok(m)
}

/// `x` を層 1 で実体化した `Tensor<f32>` を返す（`reduce_ops::
/// materialize_one` と同型）。
fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

/// `x`／`rcond`（渡された場合）が有限であることを検査する（PyTorch
/// との差分「非有限入力は `InvalidArgument`」。モジュール doc 参照）。
fn require_finite(t: &Tensor<f32>, op_name: &str) -> Result<(), AutodiffError> {
    if t.host_slice().iter().any(|v| !v.is_finite()) {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: 非有限入力（NaN／Inf）は受け付けない"
        )));
    }
    Ok(())
}

/// 対称固有値分解（`A: [n,n]`〈対称。下三角のみ読む〉→
/// [`EighVars`]）。イシュー #2150。
///
/// **確保前のバイト数上限検査**: 関数冒頭・非有限検査より前。
///
/// **非有限入力**: `AutodiffError::InvalidArgument`（モジュール doc）。
pub fn eigh<'t>(x: &Var<'t>) -> Result<EighVars<'t>, AutodiffError> {
    let shape = x.shape();
    ensure_alloc_fits_f32(&[&shape])?;
    let n = require_square(&shape, "linalg_ops::eigh")?;
    let a_val = materialize_one(x)?;
    require_finite(&a_val, "linalg_ops::eigh")?;
    let (values_val, vectors_val) = match x.tape().ops().linalg_eigh(&a_val) {
        Ok(EighFactors {
            eigenvalues,
            eigenvectors,
        }) => {
            verify_shape(eigenvalues.shape(), &[n])?;
            verify_shape(eigenvectors.shape(), &[n, n])?;
            (eigenvalues, eigenvectors)
        }
        Err(BackendError::Unsupported(_)) => eval::linalg::eigh(&a_val)?,
        Err(other) => return Err(unify_backend_error(other)),
    };
    let values_id = x.tape().push_eager(
        Op::EighValues {
            input: x.node_id(),
            vectors: vectors_val.clone(),
        },
        values_val.clone(),
    );
    let vectors_id = x.tape().push_eager(
        Op::EighVectors {
            input: x.node_id(),
            values: values_val,
        },
        vectors_val,
    );
    Ok(EighVars {
        eigenvalues: Var::from_raw(x.tape(), values_id),
        eigenvectors: Var::from_raw(x.tape(), vectors_id),
    })
}

/// 符号付き log 行列式（`A: [n,n]` → [`SlogdetVars`]）。イシュー
/// #2150。特異行列は forward で `(0, -inf)`（エラーにしない。
/// `eval::linalg::slogdet` doc 参照）。`SlogdetSign` の勾配は常にゼロ。
///
/// **確保前のバイト数上限検査**: 関数冒頭・非有限検査より前。
///
/// **非有限入力**: `AutodiffError::InvalidArgument`。
pub fn slogdet<'t>(x: &Var<'t>) -> Result<SlogdetVars<'t>, AutodiffError> {
    let shape = x.shape();
    ensure_alloc_fits_f32(&[&shape])?;
    require_square(&shape, "linalg_ops::slogdet")?;
    let a_val = materialize_one(x)?;
    require_finite(&a_val, "linalg_ops::slogdet")?;
    let (sign_val, logabsdet_val) = match x.tape().ops().linalg_slogdet(&a_val) {
        Ok(SlogdetFactors { sign, logabsdet }) => {
            verify_shape(sign.shape(), &[])?;
            verify_shape(logabsdet.shape(), &[])?;
            (sign, logabsdet)
        }
        Err(BackendError::Unsupported(_)) => eval::linalg::slogdet(&a_val),
        Err(other) => return Err(unify_backend_error(other)),
    };
    let sign_id = x
        .tape()
        .push_eager(Op::SlogdetSign { input: x.node_id() }, sign_val);
    let logabsdet_id = x
        .tape()
        .push_eager(Op::SlogdetLogAbsDet { input: x.node_id() }, logabsdet_val);
    Ok(SlogdetVars {
        sign: Var::from_raw(x.tape(), sign_id),
        logabsdet: Var::from_raw(x.tape(), logabsdet_id),
    })
}

/// `rcond` の有限性・非負性を検査する（`eval::linalg::resolve_rcond`
/// と同じ契約を呼び出し前に確定させる。`AutodiffError` へ写像するため
/// にここで一度検査してから `eval::linalg::resolve_rcond`〈`Result<_,
/// AutodiffError>` を返す〉へそのまま渡す）。
fn check_rcond(rcond: Option<f32>, op_name: &str) -> Result<(), AutodiffError> {
    if let Some(r) = rcond
        && (!r.is_finite() || r < 0.0)
    {
        return Err(AutodiffError::InvalidArgument(format!(
            "{op_name}: rcond は有限かつ非負である必要がある、got {r}"
        )));
    }
    Ok(())
}

/// Moore–Penrose 擬似逆行列（`A: [m,n]` → `Var`（`[n,m]`)）。イシュー
/// #2150。`rcond`（`None` は `max(m,n)・f32::EPSILON`）を下回る特異値は
/// 0 として扱う。
///
/// **確保前のバイト数上限検査**: 関数冒頭・非有限検査より前。
///
/// **非有限入力**: `AutodiffError::InvalidArgument`。
pub fn pinv<'t>(x: &Var<'t>, rcond: Option<f32>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    check_rcond(rcond, "linalg_ops::pinv")?;
    let (m, n) = require_rank2(&shape, "linalg_ops::pinv")?;
    let out_shape = [n, m];
    ensure_alloc_fits_f32(&[&shape, &out_shape])?;
    let a_val = materialize_one(x)?;
    require_finite(&a_val, "linalg_ops::pinv")?;
    let value = match x.tape().ops().linalg_pinv(&a_val, rcond) {
        Ok(v) => {
            verify_shape(v.shape(), &out_shape)?;
            v
        }
        Err(BackendError::Unsupported(_)) => eval::linalg::pinv(&a_val, rcond)?,
        Err(other) => return Err(unify_backend_error(other)),
    };
    let resolved_rcond = eval::linalg::resolve_rcond(rcond, m, n)? as f32;
    let id = x.tape().push_eager(
        Op::Pinv {
            input: x.node_id(),
            rcond: resolved_rcond,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

/// 行列のランク（`A: [m,n]` → `Var`（`[]`。非負整数を表す f32。
/// **勾配ゼロ**。イシュー #2150）。`rcond` を上回る特異値の個数。
///
/// **確保前のバイト数上限検査**: 関数冒頭。
pub fn matrix_rank<'t>(x: &Var<'t>, rcond: Option<f32>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    check_rcond(rcond, "linalg_ops::matrix_rank")?;
    let (m, n) = require_rank2(&shape, "linalg_ops::matrix_rank")?;
    ensure_alloc_fits_f32(&[&shape])?;
    let a_val = materialize_one(x)?;
    require_finite(&a_val, "linalg_ops::matrix_rank")?;
    let value = match x.tape().ops().linalg_matrix_rank(&a_val, rcond) {
        Ok(v) => {
            verify_shape(v.shape(), &[])?;
            v
        }
        Err(BackendError::Unsupported(_)) => eval::linalg::matrix_rank(&a_val, rcond)?,
        Err(other) => return Err(unify_backend_error(other)),
    };
    let resolved_rcond = eval::linalg::resolve_rcond(rcond, m, n)? as f32;
    let id = x.tape().push_eager(
        Op::MatrixRank {
            input: x.node_id(),
            rcond: resolved_rcond,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

/// 最小二乗解（`A: [m,n]`・`B: [m,k]` → `Var`（`[n,k]`)）。イシュー
/// #2150。rank 落ち・非正方でも `A⁺ B`（最小ノルム解）を返す
/// （PyTorch 既定 driver の residuals／rank／singular_values は返さない。
/// `docs/autodiff-linalg-ops-decision.md` §8「対象外」）。
///
/// **確保前のバイト数上限検査**: 関数冒頭・非有限検査より前。
///
/// **非有限入力**: `AutodiffError::InvalidArgument`（`a`／`b` 両方）。
pub fn lstsq<'t>(a: &Var<'t>, b: &Var<'t>, rcond: Option<f32>) -> Result<Var<'t>, AutodiffError> {
    if !std::ptr::eq(a.tape(), b.tape()) {
        return Err(AutodiffError::InvalidArgument(
            "linalg_ops::lstsq: a と b は同一の Tape 上の Var である必要がある".into(),
        ));
    }
    check_rcond(rcond, "linalg_ops::lstsq")?;
    let a_shape = a.shape();
    let b_shape = b.shape();
    let (m, n) = require_rank2(&a_shape, "linalg_ops::lstsq")?;
    let (bm, k) = require_rank2(&b_shape, "linalg_ops::lstsq")?;
    if bm != m {
        return Err(AutodiffError::InvalidArgument(format!(
            "linalg_ops::lstsq: a の行数 {m} と b の行数 {bm} が一致しない"
        )));
    }
    let out_shape = [n, k];
    ensure_alloc_fits_f32(&[&a_shape, &b_shape, &out_shape])?;
    let a_val = materialize_one(a)?;
    let b_val = materialize_one(b)?;
    require_finite(&a_val, "linalg_ops::lstsq")?;
    require_finite(&b_val, "linalg_ops::lstsq")?;
    let value = match a.tape().ops().linalg_lstsq(&a_val, &b_val, rcond) {
        Ok(v) => {
            verify_shape(v.shape(), &out_shape)?;
            v
        }
        Err(BackendError::Unsupported(_)) => eval::linalg::lstsq(&a_val, &b_val, rcond)?,
        Err(other) => return Err(unify_backend_error(other)),
    };
    let resolved_rcond = eval::linalg::resolve_rcond(rcond, m, n)? as f32;
    let id = a.tape().push_eager(
        Op::Lstsq {
            a: a.node_id(),
            b: b.node_id(),
            rcond: resolved_rcond,
        },
        value,
    );
    Ok(Var::from_raw(a.tape(), id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    #[test]
    fn eigh_symmetric_2x2() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![2.0, 1.0, 1.0, 2.0], &[2, 2]));
        let out = eigh(&x).unwrap();
        let evals = out.eigenvalues.to_tensor().host_slice().into_owned();
        assert!((evals[0] - 1.0).abs() < 1e-4);
        assert!((evals[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn eigh_reads_lower_triangle_only() {
        let tape = Tape::new();
        // 上三角にゴミ値を入れても結果が不変（下三角のみ読む契約）。
        let x = tape.var(&t(vec![2.0, 999.0, 1.0, 2.0], &[2, 2]));
        let out = eigh(&x).unwrap();
        let evals = out.eigenvalues.to_tensor().host_slice().into_owned();
        assert!((evals[0] - 1.0).abs() < 1e-4);
        assert!((evals[1] - 3.0).abs() < 1e-4);
    }

    #[test]
    fn eigh_reconstructs_a() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![4.0, 1.0, 1.0, 3.0], &[2, 2]));
        let out = eigh(&x).unwrap();
        let evals = out.eigenvalues.to_tensor().host_slice().into_owned();
        let vecs = out.eigenvectors.to_tensor().host_slice().into_owned();
        // V diag(λ) Vᵀ ≈ A を検算する。
        let v = |r: usize, c: usize| vecs[r * 2 + c];
        for r in 0..2 {
            for c in 0..2 {
                let recon: f32 = (0..2).map(|k| v(r, k) * evals[k] * v(c, k)).sum();
                let expected = if r == c {
                    if r == 0 { 4.0 } else { 3.0 }
                } else {
                    1.0
                };
                assert!(
                    (recon - expected).abs() < 1e-3,
                    "reconstruction mismatch at ({r},{c}): {recon} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn eigh_empty_matrix() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 0]));
        let out = eigh(&x).unwrap();
        assert_eq!(out.eigenvalues.to_tensor().shape(), &[0]);
        assert_eq!(out.eigenvectors.to_tensor().shape(), &[0, 0]);
    }

    #[test]
    fn eigh_non_square_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        assert!(eigh(&x).is_err());
    }

    #[test]
    fn eigh_gradient_matches_finite_difference_on_eigenvalues() {
        let eps = 1e-3f32;
        // 対称パラメタ化: `A = (B + Bᵀ)/2` の `B` に関して検算する
        // （`docs/autodiff-linalg-ops-decision.md` §4「eigh」）。
        let eval_sum_fn = |b: &[f32]| -> f32 {
            let a00 = b[0];
            let a01 = (b[1] + b[2]) / 2.0;
            let a11 = b[3];
            let tape = Tape::new();
            let x = tape.var(&t(vec![a00, a01, a01, a11], &[2, 2]));
            let out = eigh(&x).unwrap();
            out.eigenvalues.to_tensor().host_slice()[0]
                + out.eigenvalues.to_tensor().host_slice()[1]
        };
        let base = vec![2.0f32, 0.3, 0.3, 3.0];
        for i in 0..4 {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_sum_fn(&plus) - eval_sum_fn(&minus)) / (2.0 * eps);
            // trace(A) = sum(eigenvalues) = a00 + a11 = b[0] + b[3] は
            // 非対角パラメタ（b[1]・b[2]）に依存しない。
            let expected = if i == 0 || i == 3 { 1.0 } else { 0.0 };
            assert!(
                (numeric - expected).abs() < 1e-2,
                "i={i} numeric={numeric} expected={expected}"
            );
        }
    }

    /// `eigenvectors` 出力がグラフ上は損失へ到達する（`Op::EighVectors`
    /// の VJP が呼ばれ `g_vectors` が `Some` になる）が、その値が
    /// （ゼロ重みとの `mul` 等により）全要素厳密ゼロの場合、固有値が
    /// 縮退（本テストは単位行列で `λ0 = λ1 = 1`）していても
    /// `gA = V diag(gL) Vᵀ` は well-defined なため成功すべき（PR #2268
    /// codex-review〈P2〉指摘: `g_vectors` が `Some` というだけで縮退
    /// 判定してしまうと誤って `InvalidArgument` になっていた）。
    #[test]
    fn eigh_backward_with_zero_vector_grad_and_degenerate_eigenvalues_does_not_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
        let out = eigh(&x).unwrap();
        let zero = tape.var(&t(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]));
        let zero_evecs_contrib = out.eigenvectors.mul(&zero).unwrap().sum(None).unwrap();
        let loss = out
            .eigenvalues
            .sum(None)
            .unwrap()
            .add(&zero_evecs_contrib)
            .unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        // gA = V diag(1,1) Vᵀ = V Vᵀ = I（下三角へ集約: 対角のみ 1）。
        assert!((da[0] - 1.0).abs() < 1e-4);
        assert_eq!(da[1], 0.0);
        assert!(da[2].abs() < 1e-4);
        assert!((da[3] - 1.0).abs() < 1e-4);
    }

    /// `eigh_gradient_matches_finite_difference_on_eigenvalues` は `B` を
    /// 対称化してから forward するため、`Op::EighValues`／`Vectors` の
    /// VJP が実際に返す勾配テンソル（下三角のみ非ゼロという forward の
    /// 契約に対応するはず）は検証していない。本テストは
    /// `Tape::backward` が返す `[n,n]` 勾配テンソルそのものを、下三角
    /// 単一要素だけを直接摂動する有限差分と突合する（PR #2268
    /// codex-review〈P1〉指摘: 対称構成のまま返すと上三角へ誤った
    /// 非ゼロ勾配が漏れ下三角非対角勾配が半分になっていた）。
    #[test]
    fn eigh_gradient_is_zero_on_upper_triangle_and_matches_lower_triangle_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![2.0f32, 0.0, 0.7, 3.0]; // [a00, a01(上三角), a10(下三角), a11]
        let weights = [1.0f32, 3.0f32]; // 異なる重みで非対角成分を非ゼロにする
        let eval_fn = |a10: f32| -> f32 {
            let mut data = base.clone();
            data[2] = a10;
            let tape = Tape::new();
            let x = tape.var(&t(data, &[2, 2]));
            let out = eigh(&x).unwrap();
            let evals = out.eigenvalues.to_tensor().host_slice().into_owned();
            weights[0] * evals[0] + weights[1] * evals[1]
        };

        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[2, 2]));
        let out = eigh(&x).unwrap();
        let w = tape.var(&t(weights.to_vec(), &[2]));
        let loss = out.eigenvalues.mul(&w).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&x).unwrap().unwrap().host_slice().into_owned();

        // 上三角（対角除く）は forward で読まれないため厳密ゼロ。
        assert_eq!(
            da[1], 0.0,
            "上三角勾配は forward が読まないためゼロであるべき"
        );

        let numeric = (eval_fn(base[2] + eps) - eval_fn(base[2] - eps)) / (2.0 * eps);
        assert!(
            (numeric - da[2]).abs() < 5e-2,
            "下三角 a10 の数値微分と解析勾配が一致しない: numeric={numeric} analytic={}",
            da[2]
        );
    }

    #[test]
    fn slogdet_basic() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]));
        let out = slogdet(&x).unwrap();
        assert_eq!(out.sign.to_tensor().host_slice()[0], 1.0);
        assert!((out.logabsdet.to_tensor().host_slice()[0] - 6.0f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn slogdet_negative_determinant() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]));
        let out = slogdet(&x).unwrap();
        assert_eq!(out.sign.to_tensor().host_slice()[0], -1.0);
    }

    #[test]
    fn slogdet_singular_matrix() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]));
        let out = slogdet(&x).unwrap();
        assert_eq!(out.sign.to_tensor().host_slice()[0], 0.0);
        assert_eq!(out.logabsdet.to_tensor().host_slice()[0], f32::NEG_INFINITY);
    }

    #[test]
    fn slogdet_empty_matrix() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 0]));
        let out = slogdet(&x).unwrap();
        assert_eq!(out.sign.to_tensor().host_slice()[0], 1.0);
        assert_eq!(out.logabsdet.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn slogdet_sign_gradient_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![2.0, 0.0, 0.0, 3.0], &[2, 2]));
        let out = slogdet(&x).unwrap();
        let grads = tape.backward(&out.sign).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn slogdet_logabsdet_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![2.0f32, 0.3, 0.1, 3.0];
        let eval_fn = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[2, 2]));
            slogdet(&x).unwrap().logabsdet.to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[2, 2]));
        let out = slogdet(&x).unwrap();
        let grads = tape.backward(&out.logabsdet).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..4 {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus) - eval_fn(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    #[test]
    fn pinv_full_rank_square_matches_inverse() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]));
        let p = pinv(&x, None).unwrap();
        let inv = x.inv().unwrap();
        let p_data = p.to_tensor().host_slice().into_owned();
        let inv_data = inv.to_tensor().host_slice().into_owned();
        for (a, b) in p_data.iter().zip(inv_data.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn pinv_rank_deficient_matches_known_value() {
        // A = [[1,2],[2,4]]（rank 1）。A+ = A^T / 25（Aᵀ A のフロベニウス
        // ノルム二乗で正規化した既知解）。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]));
        let p = pinv(&x, None).unwrap();
        let data = p.to_tensor().host_slice().into_owned();
        let expected = [1.0 / 25.0, 2.0 / 25.0, 2.0 / 25.0, 4.0 / 25.0];
        for (a, b) in data.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn pinv_moore_penrose_conditions_tall() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]));
        let p = pinv(&x, None).unwrap();
        // A P A ≈ A の検算。
        let a_mat = x.to_tensor();
        let p_mat = p.to_tensor();
        let ap = x.matmul(&p).unwrap();
        let apa = ap.matmul(&x).unwrap().to_tensor();
        let a_data = a_mat.host_slice().into_owned();
        let apa_data = apa.host_slice().into_owned();
        for (a, b) in a_data.iter().zip(apa_data.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        let _ = p_mat;
    }

    #[test]
    fn pinv_gradient_matches_finite_difference_full_rank() {
        let eps = 1e-3f32;
        let base = vec![4.0f32, 7.0, 2.0, 6.0];
        let eval_fn = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[2, 2]));
            let p = pinv(&x, None).unwrap();
            p.to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[2, 2]));
        let p = pinv(&x, None).unwrap();
        let grads = tape
            .backward(&p.narrow(0, 0, 1).unwrap().narrow(1, 0, 1).unwrap())
            .unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..4 {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus) - eval_fn(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 5e-2,
                "i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    /// `eval::linalg::lstsq` の `m==0` 早期リターンは `n * k_cols` の
    /// 乗算を経て出力バッファを確保するが、`m` が非ゼロを要求しない
    /// ため `n`・`k_cols` を巨大にすると `checked_mul` なしでは
    /// `usize` overflow（debug panic／release wrap）しうる（PR #2268
    /// codex-review〈Bugbot〉指摘。`crates/backend-cpu/src/linalg.rs`
    /// の `BackendOps` 経路にも同型の検査を追加済み）。
    #[test]
    fn lstsq_eval_rejects_overflowing_output_shape_instead_of_panicking() {
        let a = t(vec![], &[0, usize::MAX]);
        let b = t(vec![], &[0, 2]);
        let result = eval::linalg::lstsq(&a, &b, None);
        assert!(
            matches!(result, Err(AutodiffError::InvalidArgument(_))),
            "巨大な出力形状は panic ではなく型付きエラーで拒否すべき: {result:?}"
        );
    }

    #[test]
    fn pinv_empty_matrix() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 3]));
        let p = pinv(&x, None).unwrap();
        assert_eq!(p.to_tensor().shape(), &[3, 0]);
    }

    #[test]
    fn matrix_rank_full_rank() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
        let r = matrix_rank(&x, None).unwrap();
        assert_eq!(r.to_tensor().host_slice()[0], 2.0);
    }

    #[test]
    fn matrix_rank_deficient() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]));
        let r = matrix_rank(&x, None).unwrap();
        assert_eq!(r.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn matrix_rank_zero_matrix() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 0.0, 0.0, 0.0], &[2, 2]));
        let r = matrix_rank(&x, None).unwrap();
        assert_eq!(r.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn matrix_rank_gradient_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
        let r = matrix_rank(&x, None).unwrap();
        let grads = tape.backward(&r).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn lstsq_overdetermined_matches_known_solution() {
        // A = [[1,0],[0,1],[1,1]], b = [1,2,4] の最小二乗解。
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]));
        let b = tape.var(&t(vec![1.0, 2.0, 4.0], &[3, 1]));
        let x = lstsq(&a, &b, None).unwrap();
        // 正規方程式 AᵀA x = Aᵀb を解いた既知値で検算。
        // AᵀA = [[2,1],[1,2]]、Aᵀb = [5,6] → x = [4/3, 7/3]。
        let data = x.to_tensor().host_slice().into_owned();
        assert!((data[0] - 4.0 / 3.0).abs() < 1e-3);
        assert!((data[1] - 7.0 / 3.0).abs() < 1e-3);
    }

    #[test]
    fn lstsq_rank_deficient_matches_pinv() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]));
        let b = tape.var(&t(vec![1.0, 2.0], &[2, 1]));
        let x = lstsq(&a, &b, None).unwrap();
        let p = pinv(&a, None).unwrap();
        let expected = p.matmul(&b).unwrap();
        let data = x.to_tensor().host_slice().into_owned();
        let expected_data = expected.to_tensor().host_slice().into_owned();
        for (v, e) in data.iter().zip(expected_data.iter()) {
            assert!((v - e).abs() < 1e-3, "{v} vs {e}");
        }
    }

    #[test]
    fn lstsq_row_mismatch_is_error() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
        let b = tape.var(&t(vec![1.0, 2.0, 3.0], &[3, 1]));
        assert!(lstsq(&a, &b, None).is_err());
    }

    #[test]
    fn lstsq_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let a_base = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0];
        let b_base = vec![1.0f32, 2.0, 4.0];
        let eval_fn = |a_data: &[f32]| -> f32 {
            let tape = Tape::new();
            let a = tape.var(&t(a_data.to_vec(), &[3, 2]));
            let b = tape.var(&t(b_base.clone(), &[3, 1]));
            let x = lstsq(&a, &b, None).unwrap();
            x.to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let a = tape.var(&t(a_base.clone(), &[3, 2]));
        let b = tape.var(&t(b_base.clone(), &[3, 1]));
        let x = lstsq(&a, &b, None).unwrap();
        let grads = tape
            .backward(&x.narrow(0, 0, 1).unwrap().narrow(1, 0, 1).unwrap())
            .unwrap();
        let da = grads.get(&a).unwrap().unwrap().host_slice().into_owned();
        for i in 0..6 {
            let mut plus = a_base.clone();
            plus[i] += eps;
            let mut minus = a_base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus) - eval_fn(&minus)) / (2.0 * eps);
            assert!(
                (numeric - da[i]).abs() < 5e-2,
                "i={i} numeric={numeric} analytic={}",
                da[i]
            );
        }
    }

    #[test]
    fn rcond_rejects_negative_and_non_finite() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
        for r in [-1.0f32, f32::NAN, f32::INFINITY] {
            assert!(pinv(&x, Some(r)).is_err());
            assert!(matrix_rank(&x, Some(r)).is_err());
        }
    }

    #[test]
    fn non_finite_input_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![f32::NAN, 0.0, 0.0, 1.0], &[2, 2]));
        assert!(eigh(&x).is_err());
        assert!(slogdet(&x).is_err());
        assert!(pinv(&x, None).is_err());
        assert!(matrix_rank(&x, None).is_err());
    }
}
