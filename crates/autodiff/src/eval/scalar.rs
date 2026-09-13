//! `ScalarUnaryOp`／`ScalarBinaryOp`（`tensor-core::scalar_op`。イシュー
//! #1634）のホスト参照 forward・VJP 係数。
//!
//! `crate::eval`（`NaiveOps`／`TestOps` compat 経路の forward 参照実装、
//! および `BackendOps::scalar_unary`／`scalar_binary` が `Unsupported`
//! を返した場合のフォールバック先）が使う。`eval/linalg.rs` と同じ
//! 「`crate::eval` の子モジュールとして分離する」配置方針
//! （`lib.rs` の `mod eval;` は変更しない）。
//!
//! forward 数式は `ScalarUnaryOp::apply`／`ScalarBinaryOp::apply`
//! （`tensor-core::scalar_op`）へ委譲する（`crate::scalar_op` モジュール
//! doc「forward 数式の単一情報源」参照。`eval.rs` 側に数式を複製しない）。
//! 本モジュールが独自に持つのは走査ロジック（`super::broadcast_binary`
//! への委譲・`dense_vec`/`build_tensor` による shape 復元）と、VJP
//! （backward）側の導関数のみ。

use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarUnaryOp, Tensor};

use super::{build_tensor, dense_vec};

/// [`ScalarUnaryOp`] の forward（shape 不変の要素ごとの map）。
///
/// `#[allow(dead_code)]`: 呼び出し元 `grad::scalar_unary_with_fallback`
/// と同じ理由（公開 API 面の配線は #1593／#1595）・同じ撤去条件
/// （`tape::Op::ScalarUnary` doc 参照）。
#[allow(dead_code)]
pub(crate) fn unary(input: &Tensor<f32>, op: ScalarUnaryOp) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    let data = dense_vec(input);
    let out: Vec<f32> = data.into_iter().map(|x| op.apply(x)).collect();
    build_tensor(out, &shape)
}

/// [`ScalarBinaryOp`] の forward（NumPy 互換ブロードキャスト。shape 検査
/// は呼び出し元が済ませている前提。`super::broadcast_binary` と同じ
/// 契約）。
///
/// `#[allow(dead_code)]`: [`unary`] と同じ理由・同じ撤去条件。
#[allow(dead_code)]
pub(crate) fn binary(lhs: &Tensor<f32>, rhs: &Tensor<f32>, op: ScalarBinaryOp) -> Tensor<f32> {
    super::broadcast_binary(lhs, rhs, move |a, b| op.apply(a, b))
}

// ---------------------------------------------------------------------
// VJP 係数（backward 側）。`x`（入力値）・`y`（forward 出力値。再計算
// を避けるための forward 記録値）を受け取り、`upstream` に乗じる係数
// テンソルを返す（`grad.rs::vjp_elementwise_mul` へ渡す想定）。
//
// 数値規約（`docs/scalar-op-dispatch-design.md` §3.4 が正）:
// - `Relu`／`Abs` の劣勾配は `x == 0` で `0`。
// - `Clamp` は範囲外で勾配 `0`、境界上（`x == min`／`x == max`）は `1`
//   を通す（PyTorch の `min <= x <= max` 判定と同じ）。`min > max` は
//   常に `0`（forward が常に `max` を返す定数関数のため）。
// - `Maximum`／`Minimum` の tie（`a == b`）は `0.5`／`0.5` に分配
//   （PyTorch 準拠。`Var::max`〈縮約〉の先勝ち規約とは別演算）。
//   **`a`／`b` のいずれかが `NaN` の場合はタイ分割ではなく `(0.0, 0.0)`**
//   （`Clamp` の NaN 規約と統一。PR #1686 codex-review／Bugbot 指摘の
//   是正・`docs/scalar-op-dispatch-design.md` 参照）。
// - `Pow`（binary）の `db`（`∂/∂b[a^b] = a^b・ln(a)`）は `a == 0` の
//   場合 `0` にマスクする（PyTorch のマスク規約。`ln(0) = -inf` に
//   `y = 0` が掛かり `NaN` になるのを避ける）。`da`
//   （`∂/∂a[a^b] = b・a^(b-1)`）も `b == 0` の場合 `0` にマスクする
//   （`a == 0` かつ `b == 0` で `0・inf = NaN` になるのを避ける。
//   unary `PowScalar` の `exponent == 0` マスクと同型）。
// ---------------------------------------------------------------------

/// [`ScalarUnaryOp`] の VJP 係数 `d/dx[op(x)]`。`y` は forward 出力値
/// （`Exp`/`Tanh`/`Sigmoid`/`Elu` が再計算を避けて再利用する）。
pub(crate) fn unary_grad_factor(op: ScalarUnaryOp, x: f32, y: f32) -> f32 {
    match op {
        ScalarUnaryOp::Neg => -1.0,
        ScalarUnaryOp::Abs => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        ScalarUnaryOp::Sqrt => 0.5 / y,
        ScalarUnaryOp::Log => 1.0 / x,
        ScalarUnaryOp::Log2 => 1.0 / (x * std::f32::consts::LN_2),
        ScalarUnaryOp::Log10 => 1.0 / (x * std::f32::consts::LN_10),
        ScalarUnaryOp::Sin => x.cos(),
        ScalarUnaryOp::Cos => -x.sin(),
        ScalarUnaryOp::Tan => {
            let c = x.cos();
            1.0 / (c * c)
        }
        ScalarUnaryOp::Relu => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        ScalarUnaryOp::Exp => y,
        ScalarUnaryOp::Tanh => 1.0 - y * y,
        ScalarUnaryOp::Sigmoid => y * (1.0 - y),
        ScalarUnaryOp::Gelu => fandhe_ai_tensor_core::scalar_op::gelu_erf_grad(x),
        ScalarUnaryOp::GeluTanh => fandhe_ai_tensor_core::scalar_op::gelu_tanh_grad(x),
        ScalarUnaryOp::Silu => fandhe_ai_tensor_core::scalar_op::silu_grad(x),
        ScalarUnaryOp::Hardswish => fandhe_ai_tensor_core::scalar_op::hardswish_grad(x),
        ScalarUnaryOp::LeakyRelu { negative_slope } => {
            if x >= 0.0 {
                1.0
            } else {
                negative_slope
            }
        }
        ScalarUnaryOp::Elu { alpha } => {
            if x > 0.0 {
                1.0
            } else {
                // forward: y = alpha * (exp(x) - 1) なので
                // dy/dx = alpha * exp(x) = y + alpha。
                y + alpha
            }
        }
        ScalarUnaryOp::Softplus { beta, threshold } => {
            if x * beta > threshold {
                1.0
            } else {
                fandhe_ai_tensor_core::scalar_op::sigmoid_scalar(beta * x)
            }
        }
        ScalarUnaryOp::Clamp { min, max } => {
            if x.is_nan() || min > max || x < min || x > max {
                0.0
            } else {
                1.0
            }
        }
        ScalarUnaryOp::PowScalar { exponent } => {
            // `exponent == 0.0` は forward が定数関数（`x^0 = 1`）に
            // なるため勾配は常に 0。ガードなしだと `x == 0.0` かつ
            // `exponent == 0.0` で `0.0 * 0.0.powf(-1.0)` = `0.0 * inf`
            // = `NaN` になる（PR #1686 codex-review 指摘 P2。
            // `docs/scalar-op-dispatch-design.md`「PR #1686 codex-review／
            // Bugbot 指摘の是正」参照）。
            if exponent == 0.0 {
                0.0
            } else {
                exponent * x.powf(exponent - 1.0)
            }
        }
        // `ScalarUnaryOp` は `#[non_exhaustive]`（`tensor-core` 側で
        // 将来 variant を追加できるようにするため）で、crate 境界を
        // またぐ match は列挙済み variant のみでは非網羅と判定される。
        // 本イシュー（#1634）が定義した variant は上記で尽くしており、
        // ここに到達するのは `tensor-core` 側へ新 variant が追加され
        // かつ本モジュールの導関数実装が追従していない場合のみ
        // （コンパイルは通るがテストで検知される契約違反）。
        _ => unreachable!(
            "eval::scalar::unary_grad_factor: 未対応の ScalarUnaryOp variant {op:?}（\
             tensor-core 側に新 variant が追加され本モジュールの導関数実装が \
             追従していない）"
        ),
    }
}

/// [`ScalarBinaryOp`] の VJP 係数 `(da, db)`
/// （`d/da[op(a,b)]`・`d/db[op(a,b)]`）。`y` は forward 出力値。
pub(crate) fn binary_partials(op: ScalarBinaryOp, a: f32, b: f32, y: f32) -> (f32, f32) {
    match op {
        ScalarBinaryOp::Add => (1.0, 1.0),
        ScalarBinaryOp::Sub => (1.0, -1.0),
        ScalarBinaryOp::Mul => (b, a),
        // `db = -a / b^2` を `-(a / b) / b` へ変形する（PR #1686
        // codex-review 指摘 P2）。`b * b` を先に計算すると
        // `a == b == 1e-30` で `b*b` が 0 へ underflow して `-inf`、
        // `a == b == 1e20` で `b*b` が `inf` へ overflow して `-0.0`
        // になり、いずれも数学的に有限な値（`-1/b`）から懸け離れる。
        // 変形後の式は非 NaN 入力では既存式と同値だが overflow/underflow
        // 耐性が高い。
        ScalarBinaryOp::Div => (1.0 / b, -(a / b) / b),
        ScalarBinaryOp::Pow => {
            // `b == 0.0` は forward が定数関数（`a^0 = 1`）になるため
            // da は常に 0。ガードなしだと `a == 0.0` かつ `b == 0.0` で
            // `0.0 * 0.0.powf(-1.0)` = `0.0 * inf` = `NaN` になる（unary
            // `PowScalar` と同型の bug。PR #1686 codex-review 指摘 P2）。
            let da = if b == 0.0 { 0.0 } else { b * a.powf(b - 1.0) };
            let db = if a == 0.0 { 0.0 } else { y * a.ln() };
            (da, db)
        }
        // `a`／`b` のいずれかが `NaN` のとき、IEEE 754 比較
        // （`>`／`<`）はすべて `false` になり `else` 節（タイ分割
        // `0.5`/`0.5`）へ落ちてしまう（Bugbot 指摘）。forward
        // （`tensor_core::scalar_op::nan_propagating_max`/`_min`）は
        // `NaN` を明示伝播しており、`Clamp` が `NaN` 入力で勾配を
        // ゼロにする規約（本モジュール doc「数値規約」）と揃え、
        // `Maximum`/`Minimum` も `NaN` 入力では両入力の勾配をゼロに
        // する（`docs/scalar-op-dispatch-design.md`「PR #1686
        // codex-review／Bugbot 指摘の是正」で規約として明記）。
        ScalarBinaryOp::Maximum => {
            if a.is_nan() || b.is_nan() {
                (0.0, 0.0)
            } else if a > b {
                (1.0, 0.0)
            } else if a < b {
                (0.0, 1.0)
            } else {
                (0.5, 0.5)
            }
        }
        ScalarBinaryOp::Minimum => {
            if a.is_nan() || b.is_nan() {
                (0.0, 0.0)
            } else if a < b {
                (1.0, 0.0)
            } else if a > b {
                (0.0, 1.0)
            } else {
                (0.5, 0.5)
            }
        }
        // 比較演算は出力が離散値（0.0/1.0）で入力に対し区分定数（傾き
        // 0）のため、両入力への寄与はゼロ勾配（§3.4 数値規約。寄与を
        // 省略せず明示的に `(0.0, 0.0)` を返す）。
        ScalarBinaryOp::Gt
        | ScalarBinaryOp::Ge
        | ScalarBinaryOp::Lt
        | ScalarBinaryOp::Le
        | ScalarBinaryOp::Eq
        | ScalarBinaryOp::Ne => (0.0, 0.0),
        // `unary_grad_factor` と同じ理由（`ScalarBinaryOp` も
        // `#[non_exhaustive]`）。
        _ => unreachable!(
            "eval::scalar::binary_partials: 未対応の ScalarBinaryOp variant {op:?}（\
             tensor-core 側に新 variant が追加され本モジュールの導関数実装が \
             追従していない）"
        ),
    }
}

/// [`unary_grad_factor`] を shape 全体（`x`／`y` は同一 shape。単項の
/// ため broadcast なし）へ適用したテンソル版。`grad.rs::vjp` の
/// `Op::ScalarUnary` 分岐が呼ぶ。
pub(crate) fn unary_grad_factors(
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    op: ScalarUnaryOp,
) -> Tensor<f32> {
    let shape = x.shape().to_vec();
    let x_data = dense_vec(x);
    let y_data = dense_vec(y);
    debug_assert_eq!(
        x_data.len(),
        y_data.len(),
        "unary_grad_factors: x/y の要素数が一致しない（shape 不変の \
         unary forward 契約が破れている）"
    );
    let out: Vec<f32> = x_data
        .iter()
        .zip(y_data.iter())
        .map(|(&xv, &yv)| unary_grad_factor(op, xv, yv))
        .collect();
    build_tensor(out, &shape)
}

/// [`binary_partials`] を shape 全体へ適用し `(da, db)` の 2 テンソルを
/// 返す（`lhs`／`rhs` を [`Tensor::broadcast_with`] で `out_value` と
/// 同じ共通 shape へ揃えてから要素ごとに計算する。`grad.rs::vjp` の
/// `Op::ScalarBinary` 分岐が呼ぶ。呼び出し元は `reduce_bias_grad`
/// 等で元の `a`/`b` shape へ縮約する契約——本関数は `out_value` と
/// 同じ broadcast 後 shape のまま返す）。
pub(crate) fn binary_grad_factors(
    lhs: &Tensor<f32>,
    rhs: &Tensor<f32>,
    out_value: &Tensor<f32>,
    op: ScalarBinaryOp,
) -> (Tensor<f32>, Tensor<f32>) {
    let (blhs, brhs) = match lhs.broadcast_with(rhs) {
        Ok(pair) => pair,
        Err(_) => {
            debug_assert!(
                false,
                "binary_grad_factors: 呼び出し元の broadcast_shape 検査済み前提が崩れた"
            );
            return (lhs.clone(), rhs.clone());
        }
    };
    let shape = blhs.shape().to_vec();
    let lhs_data = dense_vec(&blhs);
    let rhs_data = dense_vec(&brhs);
    let y_data = dense_vec(out_value);
    debug_assert_eq!(
        y_data.len(),
        lhs_data.len(),
        "binary_grad_factors: out_value の要素数が broadcast 後の \
         lhs/rhs と一致しない"
    );
    let mut da = Vec::with_capacity(lhs_data.len());
    let mut db = Vec::with_capacity(lhs_data.len());
    for ((&a, &b), &y) in lhs_data.iter().zip(rhs_data.iter()).zip(y_data.iter()) {
        let (pa, pb) = binary_partials(op, a, b, y);
        da.push(pa);
        db.push(pb);
    }
    (build_tensor(da, &shape), build_tensor(db, &shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::build_tensor;

    #[test]
    fn unary_forward_matches_apply() {
        let t = build_tensor(vec![1.0, -2.0, 3.0], &[3]);
        let out = unary(&t, ScalarUnaryOp::Relu);
        assert_eq!(out.get(&[0]), Some(1.0));
        assert_eq!(out.get(&[1]), Some(0.0));
        assert_eq!(out.get(&[2]), Some(3.0));
    }

    #[test]
    fn binary_forward_matches_apply_with_broadcast() {
        let a = build_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = build_tensor(vec![10.0, 20.0], &[2]);
        let out = binary(&a, &b, ScalarBinaryOp::Add);
        assert_eq!(out.get(&[0, 0]), Some(11.0));
        assert_eq!(out.get(&[0, 1]), Some(22.0));
        assert_eq!(out.get(&[1, 0]), Some(13.0));
        assert_eq!(out.get(&[1, 1]), Some(24.0));
    }

    #[test]
    fn maximum_minimum_tie_splits_half() {
        let (da, db) = binary_partials(ScalarBinaryOp::Maximum, 2.0, 2.0, 2.0);
        assert_eq!((da, db), (0.5, 0.5));
        let (da, db) = binary_partials(ScalarBinaryOp::Minimum, 2.0, 2.0, 2.0);
        assert_eq!((da, db), (0.5, 0.5));
    }

    #[test]
    fn pow_db_masked_at_zero_base() {
        let (_, db) = binary_partials(ScalarBinaryOp::Pow, 0.0, 2.0, 0.0);
        assert_eq!(db, 0.0);
    }

    #[test]
    fn comparison_partials_are_zero() {
        assert_eq!(
            binary_partials(ScalarBinaryOp::Gt, 1.0, 2.0, 0.0),
            (0.0, 0.0)
        );
    }
}
