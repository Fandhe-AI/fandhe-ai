//! `prod`・`logsumexp`・`any`・`all`・`norm_p`（p-ノルム）の 5 縮約
//! （イシュー #2147・親 #2131「5-B 演算」）。
//!
//! **facade 非公開（意図的）**: `crates/autodiff/src/matrix_ops.rs`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。イシュー
//! #2147 本文は facade 公開面（`Var::prod` 等の委譲メソッド）を承認
//! 事項として明示し、親 #2131 はこのツリーに限り「設計判断記録 →
//! 承認 → 実装」の 2 段階を定めるため、承認が取れるまでは自由関数と
//! して `Var` の外に置き到達不能にする（`docs/autodiff-reduce-ops-
//! decision.md` §0）。承認後は `Var::prod` 等の薄い委譲メソッドを追加
//! し、facade 側の保留ガード（`crates/facade/src/lib.rs::
//! VarReduceOpsHoldDoctestGuard`）を撤去する。
//!
//! **PyTorch 相当・出力型**（詳細は `docs/autodiff-reduce-ops-
//! decision.md` §1 の表を参照）:
//!
//! | 演算 | PyTorch 相当 | 出力 | 微分 |
//! |---|---|---|---|
//! | [`prod`] | `torch.prod` | f32 | 可 |
//! | [`logsumexp`] | `torch.logsumexp` | f32 | 可 |
//! | [`any`] | `torch.any` | **f32 の 0.0／1.0 マスク** | 勾配ゼロ |
//! | [`all`] | `torch.all` | **f32 の 0.0／1.0 マスク** | 勾配ゼロ |
//! | [`norm_p`] | `torch.linalg.vector_norm(ord=p)` | f32 | 可 |
//!
//! `any`／`all` の **bool 出力版**は `crate::bool_ops`（イシュー #2141）
//! の対象であり本モジュールの対象外——本モジュールが返すのは既存の
//! `Var::gt` 等と同じ f32 の 0.0／1.0 マスク（勾配ゼロの tape ノード）。
//!
//! **新規 `Op` の有無**:
//! - [`prod`]・[`any`]・[`all`] は既存 `Op` の合成のみ（新規 `Op`
//!   なし）: `prod` は `cumprod（Op::Cumprod）→ narrow（Op::Narrow）→
//!   squeeze（reshape へ委譲）`、`any`／`all` は `ne（Op::ScalarBinary）
//!   → max／min（Op::Max／Op::Min）`。
//! - [`logsumexp`]・[`norm_p`] は専用 `Op`（`crate::tape::Op::
//!   LogSumExp`／`Op::PNorm`）を追加する。`sum`／`exp`／`log` や
//!   `max`／`pow`／`sum` の素朴な合成では、全要素が `-inf`（または
//!   `+inf`）の lane・overflow を起こす `p` で `NaN` が出るため
//!   （`docs/autodiff-reduce-ops-decision.md` §2.4・§2.5）。
//!
//! **数値契約**（詳細は `docs/autodiff-reduce-ops-decision.md` §3）:
//! - `prod`: `Op::Cumprod`（イシュー #1731）の forward は `f64`
//!   アキュムレータで計算し 1 回だけ `f32` へ落とすため、零要素を含む
//!   場合でも正確。VJP は除算を使わない厳密形（排他 prefix 積）。
//! - `logsumexp`：`m = max(x)`（`f64`。非有限なら安定化シフトを `0` に
//!   切り替える）→ `Σ exp(x_i − m)` を `f64` で蓄積 → `ln(acc) + m`。
//! - `norm_p`：`mx = max|x_i|` を括り出す overflow-safe なスケール形
//!   （`mx · (Σ (|x_i|/mx)^p)^(1/p)`）。
//! - `any`／`all`：出力値は厳密に `0.0` か `1.0` のみで縮約順序に依存
//!   しないため 3 バックエンドで **bit 完全一致**する。
//!
//! **PyTorch との差分**（`docs/autodiff-reduce-ops-decision.md` §4）:
//! - `logsumexp` の `y == -inf`（縮約対象が全て `-inf`）lane の勾配は
//!   `0`（PyTorch は `NaN`）。`NaN` 勾配で学習を汚染しない安全側の判断。
//! - `norm_p` の `p` は有限かつ正のみ許容（`0`／負／`±inf`／`NaN` は
//!   `AutodiffError::InvalidArgument`）。PyTorch は `inf`／`0`／負の
//!   `p` も受け付けるが、inf ノルムの勾配分配方式が本リポでは未定
//!   （`Var::max` の先勝ち VJP のみ・均等分配は別イシュー）のため見送る。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: `prod` の
//! 空縮約（`n == 0`）は単位元 `1.0` の定数葉を `narrow` 呼び出し前に
//! 返す（`narrow(n-1)` の underflow 回避）。`any`／`all` の空縮約は
//! `max`／`min` が単位元を持たずエラーになるため、`any(∅) = 0.0`・
//! `all(∅) = 1.0` を定数葉で明示的に返す（PyTorch と同じ規約）。
//! `norm_p` の `p` は有限性・正値を dispatch 前に検査する
//! （`nn/norm.rs::validate_eps` と同じ fail-closed 規律）。本番経路で
//! `unwrap()`／`expect()` は使わない。

use fandhe_ai_tensor_core::{BackendError, Tensor, reduce_out_shape};

use crate::error::AutodiffError;
use crate::eval;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// `dim` に沿った縮約対象の要素数（`Var::mean`／`Var::norm` と同じ
/// 「`dim=None` は全要素数・`dim=Some(axis)` は `shape[axis]`」規約）。
fn reduce_axis_len(shape: &[usize], dim: Option<usize>) -> usize {
    match dim {
        None => shape.iter().product(),
        Some(axis) => shape[axis],
    }
}

/// バックエンド実装の戻り値 shape を検証する（`var.rs::verify_shape`
/// と同型。`pub(crate)` 化を避け本モジュール内に複製する——
/// `var.rs` 側は `fn`〈非 `pub(crate)`〉のため呼べない）。
fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

/// CPU 本番経路の `BackendError` を `AutodiffError` へ写像する
/// （`var.rs::unify_backend_error` と同型の複製。理由は
/// [`verify_shape`] と同じ）。
fn unify_backend_error(err: BackendError) -> AutodiffError {
    match err {
        BackendError::InvalidArgument(msg) => AutodiffError::InvalidArgument(msg),
        other => AutodiffError::Backend(other),
    }
}

/// `x` を層 1 で実体化した `Tensor<f32>` を返す（`bool_ops::
/// materialize_pair` と同じ「`nodes` の `RefCell` 借用をこのブロック内に
/// 閉じ込め、返す前に解放する」パターン）。
fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

/// 縮約対象の要素数 `n` に沿った累積積（`torch.prod` 相当。イシュー
/// #2147）。`dim: None` は全軸縮約（先に `reshape([numel])` してから
/// `cumprod(0)` を取る）。
///
/// `Var::cumprod`（`Op::Cumprod`。イシュー #1731）の forward は `f64`
/// アキュムレータで計算し 1 回だけ `f32` へ落とすため、零要素を含む
/// 入力でも正確。VJP は除算を使わない厳密形（排他 prefix 積 ×
/// 後ろ向き Horner 型再帰）のため、零要素が 0 個・1 個・2 個以上の
/// いずれでも正しい。
///
/// **空縮約（`n == 0`）は単位元 `1.0`**（PyTorch と同じ）を、出力
/// shape に合わせた定数葉として返す（`narrow(n-1)` の underflow を
/// 避けるため合成より前に分岐する）。
pub fn prod<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    let n = reduce_axis_len(&shape, dim);
    if n == 0 {
        let numel: usize = out_shape.iter().product();
        let value = Tensor::new(vec![1.0f32; numel], &out_shape).map_err(AutodiffError::Shape)?;
        let id = x.tape().push_leaf(value, false);
        return Ok(Var::from_raw(x.tape(), id));
    }
    match dim {
        None => {
            // `reshape` は非 contiguous view を `ShapeError::
            // NonContiguousReshape` で拒否する（`Var::reshape` の契約。
            // `var.rs`）。呼び出し元が転置・narrow 等の非 contiguous
            // view を渡しうるため、`dim: Some` 分岐と同様に `reshape`
            // 前に `contiguous()` で実体化する。
            let flat = x.contiguous()?.reshape(&[n])?;
            let cp = flat.cumprod(0)?;
            let last = cp.narrow(0, n - 1, 1)?;
            last.squeeze(Some(0))
        }
        Some(axis) => {
            let cp = x.cumprod(axis)?;
            let last = cp.narrow(axis, n - 1, 1)?;
            // `narrow` は `axis` が末尾軸でない限り非 contiguous な
            // stride view を返しうる。`squeeze`（`Var::reshape` への
            // 委譲）は contiguous 入力を要求する（`ShapeError::
            // NonContiguousReshape`）ため、`squeeze` 前に
            // `contiguous()` で実体化する（`matrix_ops::
            // diag_2d_to_1d` の `narrow → gather → squeeze` は
            // `gather` が暗黙に実体化するため同じ問題を踏まないが、
            // `prod` は `gather` を経由しないため明示的に挟む必要が
            // ある）。
            last.contiguous()?.squeeze(Some(axis))
        }
    }
}

/// log-sum-exp（`torch.logsumexp(dim)` 相当。イシュー #2147）。
/// `dim: None` は全軸縮約（スカラー）。専用 `Op::LogSumExp`
/// （`crate::tape::Op`）を直接構築する（`Var::norm`〈`var.rs`〉と同じ
/// フォールバック契約: `BackendOps::logsumexp` → `Unsupported` の
/// ときのみ `eval::logsumexp_along` へ切り替える）。
///
/// **空縮約（`n == 0`）は [`AutodiffError::InvalidArgument`]**
/// （`-inf` を黙って返さない安全側の判断。`Var::norm` と同じ方針）。
pub fn logsumexp<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    let n = reduce_axis_len(&shape, dim);
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "reduce_ops::logsumexp: 縮約対象の要素数が 0（dim={dim:?}）"
        )));
    }
    let input_val = materialize_one(x)?;
    let value = match x.tape().ops().logsumexp(&input_val, dim) {
        Ok(v) => {
            verify_shape(v.shape(), &out_shape)?;
            v
        }
        Err(BackendError::Unsupported(_)) => eval::logsumexp_along(&input_val, dim, &out_shape),
        Err(other) => return Err(unify_backend_error(other)),
    };
    let id = x.tape().push_eager(
        Op::LogSumExp {
            input: x.node_id(),
            dim,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

/// `dim` 軸のいずれかが非ゼロなら `1.0`、それ以外は `0.0`（`torch.any`
/// 相当。イシュー #2147）。**出力は f32 の 0.0／1.0 マスク**（bool 版は
/// `crate::bool_ops` の対象。モジュール doc 参照）。`NaN != 0` は真の
/// ため `NaN` は真として扱う（PyTorch と一致）。`-0.0` は偽。
///
/// `x.ne(&zero) → max(dim)` の合成（新規 `Op` なし）。出力値は厳密に
/// `0.0`／`1.0` のみのため 3 バックエンドで bit 完全一致する。勾配は
/// `ne`（`ScalarBinaryOp::Ne`）の VJP がゼロを返すため、合成しただけで
/// 自動的に勾配ゼロの tape ノードになる。
///
/// **空縮約（`n == 0`）は `0.0`**（PyTorch と同じ。`max` は単位元を
/// 持たずエラーになるため、合成より前に定数葉で返す）。
pub fn any<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    let n = reduce_axis_len(&shape, dim);
    if n == 0 {
        let numel: usize = out_shape.iter().product();
        let value = Tensor::new(vec![0.0f32; numel], &out_shape).map_err(AutodiffError::Shape)?;
        let id = x.tape().push_leaf(value, false);
        return Ok(Var::from_raw(x.tape(), id));
    }
    let zero_val = Tensor::scalar(0.0f32);
    let zero_id = x.tape().push_leaf(zero_val, false);
    let zero = Var::from_raw(x.tape(), zero_id);
    x.ne(&zero)?.max(dim)
}

/// `dim` 軸の全要素が非ゼロなら `1.0`、それ以外は `0.0`（`torch.all`
/// 相当。イシュー #2147）。[`any`] と対称（`x.ne(&zero) → min(dim)`）。
///
/// **空縮約（`n == 0`）は `1.0`**（PyTorch と同じ。[`any`] と対称）。
pub fn all<'t>(x: &Var<'t>, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    let n = reduce_axis_len(&shape, dim);
    if n == 0 {
        let numel: usize = out_shape.iter().product();
        let value = Tensor::new(vec![1.0f32; numel], &out_shape).map_err(AutodiffError::Shape)?;
        let id = x.tape().push_leaf(value, false);
        return Ok(Var::from_raw(x.tape(), id));
    }
    let zero_val = Tensor::scalar(0.0f32);
    let zero_id = x.tape().push_leaf(zero_val, false);
    let zero = Var::from_raw(x.tape(), zero_id);
    x.ne(&zero)?.min(dim)
}

/// p-ノルム（`torch.linalg.vector_norm(ord=p)` 相当。イシュー #2147）。
/// `dim: None` は全軸縮約（スカラー）。
///
/// `p` は有限かつ正のみ許容する（`NaN`・`±inf`・`0`・負の値は
/// [`AutodiffError::InvalidArgument`]。モジュール doc「PyTorch との
/// 差分」参照）。`p == 1.0`／`p == 2.0` は既存の
/// `Var::norm_l1`／`norm_l2`（`pub(crate)` の `Var::norm` 経由）へ
/// 委譲し、`norm_p(x, 2.0, d)` と `norm_l2(d)` が **bit 同一**になる。
///
/// それ以外の `p` は専用 `Op::PNorm` を直接構築する（`Var::norm` と
/// 同じフォールバック契約: `BackendOps::vector_norm_p` →
/// `Unsupported` のときのみ `eval::vector_norm_p_along` へ切り替え）。
///
/// **空縮約（`n == 0`）は [`AutodiffError::InvalidArgument`]**
/// （`Var::norm` と同じ方針）。
pub fn norm_p<'t>(x: &Var<'t>, p: f32, dim: Option<usize>) -> Result<Var<'t>, AutodiffError> {
    if !p.is_finite() || p <= 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "reduce_ops::norm_p: p は有限かつ正である必要がある、got {p}"
        )));
    }
    if p == 1.0 {
        return x.norm_l1(dim);
    }
    if p == 2.0 {
        return x.norm_l2(dim);
    }
    let shape = x.shape();
    let out_shape = reduce_out_shape(&shape, dim)?;
    let n = reduce_axis_len(&shape, dim);
    if n == 0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "reduce_ops::norm_p: 縮約対象の要素数が 0（dim={dim:?}）"
        )));
    }
    let input_val = materialize_one(x)?;
    let value = match x.tape().ops().vector_norm_p(&input_val, p, dim) {
        Ok(v) => {
            verify_shape(v.shape(), &out_shape)?;
            v
        }
        Err(BackendError::Unsupported(_)) => {
            eval::vector_norm_p_along(&input_val, p, dim, &out_shape)
        }
        Err(other) => return Err(unify_backend_error(other)),
    };
    let id = x.tape().push_eager(
        Op::PNorm {
            input: x.node_id(),
            p,
            dim,
        },
        value,
    );
    Ok(Var::from_raw(x.tape(), id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    // --- prod ---

    #[test]
    fn prod_basic() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[4]));
        let out = prod(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 24.0);
    }

    #[test]
    fn prod_with_one_zero_element() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 3.0, 4.0], &[4]));
        let out = prod(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn prod_with_two_zero_elements() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 0.0, 4.0], &[4]));
        let out = prod(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn prod_dim_axis() {
        let tape = Tape::new();
        // [[1,2],[3,4]] -> dim=0: [3, 8]
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let out = prod(&x, Some(0)).unwrap();
        assert_eq!(out.to_tensor().shape(), &[2]);
        assert_eq!(out.to_tensor().host_slice().into_owned(), vec![3.0, 8.0]);
    }

    #[test]
    fn prod_empty_reduction_is_one() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        let out = prod(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn prod_dim_none_accepts_non_contiguous_input() {
        // `transpose` は非 contiguous な stride view を返す（イシュー
        // #2147 codex-review 指摘: `dim: None` 分岐が `reshape` 前の
        // `contiguous()` を欠き `ShapeError::NonContiguousReshape` で
        // 落ちていた）。転置後の `prod(None)` が全要素積を返せることを
        // 検証する。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let xt = x.transpose(0, 1).unwrap();
        let out = prod(&xt, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 24.0);
    }

    #[test]
    fn prod_gradient_with_zero_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![2.0f32, 0.0, 3.0];
        let eval_fn = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[3]));
            let y = prod(&x, None).unwrap();
            y.to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3]));
        let y = prod(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus) - eval_fn(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "prod 勾配の有限差分検算が乖離: i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    // --- logsumexp ---

    #[test]
    fn logsumexp_basic_matches_naive() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let out = logsumexp(&x, None).unwrap();
        let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln() as f32;
        assert!((out.to_tensor().host_slice()[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn logsumexp_large_values_does_not_overflow() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1e30, 1e30], &[2]));
        let out = logsumexp(&x, None).unwrap();
        let v = out.to_tensor().host_slice()[0];
        assert!(v.is_finite());
        // 2 要素とも 1e30 の logsumexp は 1e30 + ln(2) に極めて近い。
        assert!((v - 1e30).abs() < 1.0);
    }

    #[test]
    fn logsumexp_all_neg_inf_returns_neg_inf_and_zero_grad() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![f32::NEG_INFINITY, f32::NEG_INFINITY], &[2]));
        let out = logsumexp(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], f32::NEG_INFINITY);
        let grads = tape.backward(&out).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dx.host_slice().into_owned(), vec![0.0, 0.0]);
    }

    #[test]
    fn logsumexp_with_pos_inf_is_pos_inf() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, f32::INFINITY], &[2]));
        let out = logsumexp(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], f32::INFINITY);
    }

    #[test]
    fn logsumexp_with_pos_inf_gradient_distributes_to_inf_elements() {
        // codex-review 指摘（イシュー #2147）: `+inf` を含む
        // `logsumexp` の勾配が `inf - inf = NaN` になっていた。修正後は
        // `+inf` 要素へ上流勾配を均等分配し、有限要素は 0 になる。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, f32::INFINITY, 5.0, f32::INFINITY], &[4]));
        let out = logsumexp(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], f32::INFINITY);
        let grads = tape.backward(&out).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 0.5, 0.0, 0.5]);
    }

    #[test]
    fn logsumexp_nan_propagates() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, f32::NAN], &[2]));
        let out = logsumexp(&x, None).unwrap();
        assert!(out.to_tensor().host_slice()[0].is_nan());
    }

    #[test]
    fn logsumexp_empty_reduction_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        assert!(matches!(
            logsumexp(&x, None),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn logsumexp_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![0.5f32, -1.2, 2.3];
        let eval_fn = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[3]));
            let y = logsumexp(&x, None).unwrap();
            y.to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3]));
        let y = logsumexp(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus) - eval_fn(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "logsumexp 勾配の有限差分検算が乖離: i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    // --- any / all ---

    #[test]
    fn any_true_when_one_nonzero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 0.0, 3.0], &[3]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn any_false_when_all_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 0.0, 0.0], &[3]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn any_negative_zero_is_false() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![-0.0, -0.0], &[2]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn any_nan_is_true() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, f32::NAN], &[2]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn any_empty_reduction_is_false() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        let out = any(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn any_dim_axis() {
        let tape = Tape::new();
        // [[0,1],[0,0]] -> any(dim=1) = [1, 0]
        let x = tape.var(&t(vec![0.0, 1.0, 0.0, 0.0], &[2, 2]));
        let out = any(&x, Some(1)).unwrap();
        assert_eq!(out.to_tensor().host_slice().into_owned(), vec![1.0, 0.0]);
    }

    #[test]
    fn any_gradient_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let y = any(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dx.host_slice().into_owned(), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn all_true_when_all_nonzero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let out = all(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn all_false_when_one_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 0.0, 3.0], &[3]));
        let out = all(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn all_empty_reduction_is_true() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        let out = all(&x, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 1.0);
    }

    #[test]
    fn all_gradient_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let y = all(&x, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dx.host_slice().into_owned(), vec![0.0, 0.0, 0.0]);
    }

    // --- norm_p ---

    #[test]
    fn norm_p_matches_l1_for_p_one() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![-1.0, 2.0, -3.0], &[3]));
        let a = norm_p(&x, 1.0, None).unwrap();
        let b = x.norm_l1(None).unwrap();
        assert_eq!(
            a.to_tensor().host_slice()[0].to_bits(),
            b.to_tensor().host_slice()[0].to_bits()
        );
    }

    #[test]
    fn norm_p_matches_l2_for_p_two() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![3.0, 4.0], &[2]));
        let a = norm_p(&x, 2.0, None).unwrap();
        let b = x.norm_l2(None).unwrap();
        assert_eq!(
            a.to_tensor().host_slice()[0].to_bits(),
            b.to_tensor().host_slice()[0].to_bits()
        );
        assert_eq!(a.to_tensor().host_slice()[0], 5.0);
    }

    #[test]
    fn norm_p_three_matches_naive() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 2.0], &[3]));
        let out = norm_p(&x, 3.0, None).unwrap();
        let expected = (1.0f64 + 8.0 + 8.0).powf(1.0 / 3.0) as f32;
        assert!((out.to_tensor().host_slice()[0] - expected).abs() < 1e-4);
    }

    #[test]
    fn norm_p_large_p_does_not_overflow() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1e38, 1.0], &[2]));
        let out = norm_p(&x, 50.0, None).unwrap();
        assert!(out.to_tensor().host_slice()[0].is_finite());
    }

    #[test]
    fn norm_p_zero_vector_is_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 0.0, 0.0], &[3]));
        let out = norm_p(&x, 3.0, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], 0.0);
    }

    #[test]
    fn norm_p_rejects_invalid_p() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        for p in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                matches!(norm_p(&x, p, None), Err(AutodiffError::InvalidArgument(_))),
                "p={p} は拒否されるはず"
            );
        }
    }

    #[test]
    fn norm_p_empty_reduction_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0]));
        assert!(matches!(
            norm_p(&x, 3.0, None),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn norm_p_zero_element_with_p_less_than_one_has_zero_grad_at_zero() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 3.0], &[2]));
        let y = norm_p(&x, 0.5, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert!(dx.host_slice()[0].is_finite());
        assert_eq!(dx.host_slice()[0], 0.0);
    }

    #[test]
    fn norm_p_with_inf_gradient_distributes_to_inf_elements() {
        // codex-review 指摘（イシュー #2147）: `±inf` を含む p-norm の
        // VJP が `inf / inf = NaN` になっていた。修正後は `±inf` 要素へ
        // 符号付きで上流勾配を均等分配し、有限要素は 0 になる。
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, f32::INFINITY, 5.0, f32::NEG_INFINITY], &[4]));
        let out = norm_p(&x, 3.0, None).unwrap();
        assert_eq!(out.to_tensor().host_slice()[0], f32::INFINITY);
        let grads = tape.backward(&out).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_eq!(dx, vec![0.0, 0.5, 0.0, -0.5]);
    }

    #[test]
    fn norm_p_gradient_matches_finite_difference() {
        let eps = 1e-3f32;
        let base = vec![1.5f32, -2.5, 3.5];
        let eval_fn = |data: &[f32]| -> f32 {
            let tape = Tape::new();
            let x = tape.var(&t(data.to_vec(), &[3]));
            let y = norm_p(&x, 3.0, None).unwrap();
            y.to_tensor().host_slice()[0]
        };
        let tape = Tape::new();
        let x = tape.var(&t(base.clone(), &[3]));
        let y = norm_p(&x, 3.0, None).unwrap();
        let grads = tape.backward(&y).unwrap();
        let dx = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        for i in 0..base.len() {
            let mut plus = base.clone();
            plus[i] += eps;
            let mut minus = base.clone();
            minus[i] -= eps;
            let numeric = (eval_fn(&plus) - eval_fn(&minus)) / (2.0 * eps);
            assert!(
                (numeric - dx[i]).abs() < 1e-2,
                "norm_p 勾配の有限差分検算が乖離: i={i} numeric={numeric} analytic={}",
                dx[i]
            );
        }
    }

    // --- 共通: dim 範囲外 ---

    #[test]
    fn out_of_range_dim_is_error() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0], &[2]));
        assert!(prod(&x, Some(5)).is_err());
        assert!(logsumexp(&x, Some(5)).is_err());
        assert!(any(&x, Some(5)).is_err());
        assert!(all(&x, Some(5)).is_err());
        assert!(norm_p(&x, 3.0, Some(5)).is_err());
    }
}
