//! `softmin`・`tanhshrink`・`threshold`・`rrelu`（と固定 noise 版の
//! `rrelu_with_noise`）の 4 活性化演算（イシュー #2650・親 #2648・
//! Phase 親 #2625）。
//!
//! **新規 `Op` はゼロ（受け入れ条件）**: いずれも既存の `Var::neg`・
//! `Var::softmax`・`Var::sub`・`Var::tanh`・`Var::masked_fill`・`Var::mul`
//! と `Tape::var_no_grad`（勾配を取らない葉）の合成のみで構成する。CPU・
//! CUDA・Metal の全バックエンドに経路があり（既定 `Unsupported` は
//! ホスト参照実装へフォールバック）、専用カーネルなしで到達可能。
//! `crates/backend-*`・`crates/tensor-core` は変更しない。
//!
//! **公開形**: `compat::Sequential::add_*` は #2679 で facade へ公開済み
//! （`docs/autodiff-softmin-threshold-ops-decision.md` §7・§12）。本モジュール・
//! `nn::softmin_threshold` の層は facade から再エクスポートしない。`Var` メソッド化は
//! #2678 の担当で、未承認経路は facade の `SoftminThresholdOpsHoldDoctestGuard` と
//! `api_surface.rs` の否定ガードが機械固定する。
//!
//! **数値契約**（詳細は決定記録 §3）:
//! - `softmin`: `softmax(-x)`。超越関数を含むため forward／backward とも
//!   REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
//! - `tanhshrink`: `x - tanh(x)`。同じく REQ-2 判定。0 近傍は桁落ち
//!   （真値は約 `x^3 / 3`）するが絶対誤差側で収まる。
//! - `threshold`・`rrelu_with_noise`・`rrelu`（推論）: 選択と IEEE 乗算 1
//!   回のみのため forward／backward とも bit 完全一致を見込む。
//!
//! **PyTorch との既知の差分**（決定記録 §5）:
//! - RReLU 学習時の乱数列は PyTorch と一致しない（本クレートの
//!   Xorshift64\*）。全要素分を引く点も PyTorch CPU（`x <= 0` の要素のみ
//!   引く）と異なる。
//! - `lower > upper`・非有限は推論時にも拒否する。
//! - `softmin` は負の `dim` を受け付けない（`usize`）。
//! - 学習時 noise は `[lower, upper]` へ clamp する（丸めで範囲外へ出ない
//!   保証）。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: `dim`
//! （`softmax` の軸検査へ委譲）・`lower`／`upper`（有限・`lower <= upper`）・
//! `noise` の shape は、tape 操作・RNG 消費・メモリ確保の前にすべて検査し、
//! 違反は型付きエラーで fail-closed に拒否する（エラー時に tape へ孤児
//! ノードを残さない）。本番経路で `unwrap()`／`expect()` は使わない。
//!
//! **公開状況（イシュー #2678）**: 承認形どおり公開済み: `Var::{softmin,tanhshrink,threshold,rrelu}`（`rrelu_with_noise` は公開しない）。本モジュール自体は facade から再エクスポートしない。
//! 上の「未承認」「保留」「承認依頼は #2677」の記述は #2677 時点のもので、承認形の公開は #2678 で行った
//! （ルート #2499 の承認コメント issuecomment-6033824965・`docs/compat-api-scope.md` §5.1）。

use fandhe_ai_tensor_core::{ShapeError, Tensor, rng};

use crate::activation_ops::build_value_mask;
use crate::error::AutodiffError;
use crate::var::Var;

/// Softmin（PyTorch `F.softmin(input, dim)` 相当）: `softmax(-x, dim)`。
///
/// 軸検査は [`Var::softmax`] に委譲する（範囲外は型付きエラー）。
/// `-x` の `neg` は軸検査より先に tape へノードを積むが、`dim` 範囲外は
/// 先に `x.shape()` で弾いて孤児ノードを残さない。
pub fn softmin<'t>(x: &Var<'t>, dim: usize) -> Result<Var<'t>, AutodiffError> {
    let rank = x.shape().len();
    if dim >= rank {
        return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: dim,
            rank,
        }));
    }
    x.neg()?.softmax(dim)
}

/// Tanhshrink（PyTorch `F.tanhshrink` 相当）: `x - tanh(x)`。
///
/// 0 近傍では桁落ちする（真値は約 `x^3 / 3`）が、絶対誤差側の判定
/// （1e-5 未満）で収まる。
pub fn tanhshrink<'t>(x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    x.sub(&x.tanh())
}

/// Threshold（PyTorch `F.threshold(input, threshold, value)` 相当）:
/// `x <= threshold` なら `value`、それ以外は `x`。
///
/// 述語は `v <= threshold`（`!(v > threshold)` ではない）で、`NaN` は
/// 偽側＝素通しになり勾配も通る（PyTorch 2.14.0 実測。決定記録 §5）。
/// 置換された位置の勾配は 0（`Var::masked_fill` の VJP）。`threshold`／
/// `value` は検証せず IEEE のまま扱う（`leaky_relu` の `negative_slope`
/// と同じ規約）。
pub fn threshold<'t>(x: &Var<'t>, threshold: f32, value: f32) -> Result<Var<'t>, AutodiffError> {
    let mask = build_value_mask(x, |v| v <= threshold)?;
    x.masked_fill(&mask, value)
}

/// `lower`／`upper` の検査（有限・`lower <= upper`）。学習時・推論時とも
/// tape 操作・RNG 消費より前に呼ぶ。
fn check_rrelu_bounds(lower: f32, upper: f32) -> Result<(), AutodiffError> {
    if !lower.is_finite() || !upper.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "softmin_threshold_ops::rrelu: lower/upper must be finite, got lower={lower}, upper={upper}"
        )));
    }
    if lower > upper {
        return Err(AutodiffError::InvalidArgument(format!(
            "softmin_threshold_ops::rrelu: lower must not exceed upper, got lower={lower}, upper={upper}"
        )));
    }
    Ok(())
}

/// RReLU を固定 noise で適用する: `x * noise`（`noise` は勾配を取らない
/// 定数葉）。
///
/// `noise` の各要素は正領域で `1.0`、負領域で傾きを持つ（[`rrelu`] が
/// 構築する）。shape は `x` と一致が必須で、不一致は
/// `AutodiffError::Shape`。forward・backward とも IEEE 乗算 1 回のため
/// bit 完全一致する。
pub fn rrelu_with_noise<'t>(x: &Var<'t>, noise: &Tensor<f32>) -> Result<Var<'t>, AutodiffError> {
    let x_shape = x.shape();
    if noise.shape() != x_shape.as_slice() {
        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
            lhs: x_shape,
            rhs: noise.shape().to_vec(),
        }));
    }
    let n = x.tape().var_no_grad(noise);
    x.mul(&n)
}

/// RReLU（PyTorch `F.rrelu(input, lower, upper, training)` 相当）。
///
/// - 学習時: `rng::rand(shape)` を全要素分 1 回引き（データ非依存の
///   消費量。`dropout_mask` と同型）、`r = lower + (upper - lower) * u`
///   を `f64` で計算して `f32` へ落とし `[lower, upper]` へ clamp する。
///   `x <= 0` の位置は `r`、それ以外（`NaN` 含む）は `1.0`。
/// - 推論時: RNG を消費しない。傾きは `(lower + upper) / 2`。`x > 0` は
///   `1.0`、それ以外は傾き。
///
/// 推論を `Var::leaky_relu` へ委譲しないのは、既存 `LeakyRelu` が
/// `x >= 0` を恒等側とし `x == 0` の勾配が 1 になって PyTorch
/// （`x == 0` の勾配は傾き）と食い違うため。定数 noise の単一路で境界の
/// 扱いを学習時と揃える（決定記録 §5）。
pub fn rrelu<'t>(
    x: &Var<'t>,
    lower: f32,
    upper: f32,
    training: bool,
) -> Result<Var<'t>, AutodiffError> {
    check_rrelu_bounds(lower, upper)?;
    let shape = x.shape();
    let noise_data: Vec<f32> = if training {
        let uniform = rng::rand(&shape).map_err(AutodiffError::Shape)?;
        let u = uniform.host_slice();
        let view = x.host_view();
        view.iter()
            .zip(u.iter())
            .map(|(&v, &u)| {
                if v <= 0.0 {
                    let r = f64::from(lower) + (f64::from(upper) - f64::from(lower)) * f64::from(u);
                    (r as f32).clamp(lower, upper)
                } else {
                    1.0
                }
            })
            .collect()
    } else {
        let slope = ((f64::from(lower) + f64::from(upper)) / 2.0) as f32;
        let view = x.host_view();
        view.iter()
            .map(|&v| if v > 0.0 { 1.0 } else { slope })
            .collect()
    };
    let noise = Tensor::new(noise_data, &shape).map_err(AutodiffError::Shape)?;
    rrelu_with_noise(x, &noise)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).unwrap()
    }

    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    /// `sum(y)` を損失とした入力勾配。
    fn grad_of<'t>(tape: &'t Tape, x: &Var<'t>, y: &Var<'t>) -> Vec<f32> {
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        grads.get(x).unwrap().unwrap().host_slice().into_owned()
    }

    #[test]
    fn softmin_forward_matches_hand_computation_and_rejects_bad_axis() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.0, 1.0, 2.0], &[3]));
        let y = vals(&softmin(&x, 0).unwrap());
        let e: Vec<f64> = [0.0f64, -1.0, -2.0].iter().map(|v| v.exp()).collect();
        let s: f64 = e.iter().sum();
        for (a, b) in y.iter().zip(&e) {
            assert!((f64::from(*a) - b / s).abs() < 1e-6);
        }
        let before = tape.len();
        assert!(softmin(&x, 1).is_err());
        assert_eq!(tape.len(), before, "エラー時に孤児ノードを残さない");
    }

    #[test]
    fn tanhshrink_matches_central_difference() {
        let tape = Tape::new();
        let xs = [-2.0f32, -0.3, 0.0, 0.8, 5.0];
        let x = tape.var(&t(xs.to_vec(), &[5]));
        let y = tanhshrink(&x).unwrap();
        let g = grad_of(&tape, &x, &y);
        let f = |v: f64| v - v.tanh();
        for (i, &v) in xs.iter().enumerate() {
            let v = f64::from(v);
            let h = 1e-5;
            let num = (f(v + h) - f(v - h)) / (2.0 * h);
            assert!((f64::from(g[i]) - num).abs() < 1e-5, "i={i}");
        }
    }

    #[test]
    fn threshold_replaces_at_boundary_with_zero_grad_and_passes_nan() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![0.5, 0.6, -1.0, f32::NAN, 2.0], &[5]));
        let y = threshold(&x, 0.5, 7.0).unwrap();
        let out = vals(&y);
        assert_eq!(&out[..3], &[7.0, 0.6, 7.0]);
        assert!(out[3].is_nan());
        assert_eq!(out[4], 2.0);
        assert_eq!(grad_of(&tape, &x, &y), vec![0.0, 1.0, 0.0, 1.0, 1.0]);
    }

    /// `value` が NaN／±inf のとき、置換位置は `value` を bit のまま持ち勾配 0、非置換位置は素通し
    /// （PyTorch 2.14.0 実測: 置換位置の出力は `value`・勾配 0、`x = 1` は出力 1・勾配 1）。
    /// JSON は NaN／inf を `params` に持てず fixture へ入れられないため単体テストで固定する
    /// （決定記録 §6）。
    #[test]
    fn threshold_value_nan_and_inf_is_written_bit_for_bit_with_zero_grad() {
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let tape = Tape::new();
            let x = tape.var(&t(vec![-1.0, 0.0, 1.0], &[3]));
            let y = threshold(&x, 0.0, value).unwrap();
            let out = vals(&y);
            assert_eq!(out[0].to_bits(), value.to_bits(), "value={value}");
            assert_eq!(out[1].to_bits(), value.to_bits(), "value={value}");
            assert_eq!(out[2], 1.0);
            assert_eq!(grad_of(&tape, &x, &y), vec![0.0, 0.0, 1.0], "value={value}");
        }
    }

    #[test]
    fn threshold_preserves_rank0_shape() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![3.0], &[]));
        let y = threshold(&x, 1.0, 0.0).unwrap();
        assert_eq!(y.to_tensor().shape(), &[] as &[usize]);
    }

    #[test]
    fn rrelu_eval_uses_mean_slope_and_pytorch_boundary_grad() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![2.0, 0.0, -2.0], &[3]));
        let y = rrelu(&x, 0.1, 0.3, false).unwrap();
        let slope = ((0.1f64 + 0.3f64) / 2.0) as f32;
        assert_eq!(vals(&y), vec![2.0, 0.0 * slope, -2.0 * slope]);
        // x == 0 の勾配は傾き（PyTorch。`leaky_relu` の x >= 0 恒等側とは異なる）
        assert_eq!(grad_of(&tape, &x, &y), vec![1.0, slope, slope]);
    }

    #[test]
    fn rrelu_rejects_invalid_bounds_without_touching_tape() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, -1.0], &[2]));
        let before = tape.len();
        for (lo, hi) in [
            (0.5, 0.1),
            (f32::NAN, 0.1),
            (0.1, f32::NAN),
            (0.1, f32::INFINITY),
        ] {
            for training in [true, false] {
                assert!(rrelu(&x, lo, hi, training).is_err());
            }
        }
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn rrelu_training_structure_and_determinism_when_bounds_equal() {
        let tape = Tape::new();
        let xs = vec![1.5, -2.0, 0.0, -0.5, 3.0, -4.0];
        let x = tape.var(&t(xs.clone(), &[2, 3]));
        let y = rrelu(&x, 0.25, 0.25, true).unwrap();
        let out = vals(&y);
        for (i, &v) in xs.iter().enumerate() {
            let expect = if v > 0.0 { v } else { v * 0.25 };
            assert_eq!(out[i].to_bits(), expect.to_bits());
        }
        // 範囲を持つ場合: 正領域は恒等・負領域の比は [lower, upper] 内
        let y2 = rrelu(&x, 0.1, 0.4, true).unwrap();
        let o2 = vals(&y2);
        for (i, &v) in xs.iter().enumerate() {
            if v > 0.0 {
                assert_eq!(o2[i], v);
            } else if v < 0.0 {
                let r = o2[i] / v;
                assert!((0.1..=0.4).contains(&r), "ratio {r}");
            }
        }
    }

    #[test]
    fn rrelu_with_noise_rejects_shape_mismatch_and_grad_equals_noise() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, -1.0, 2.0], &[3]));
        let before = tape.len();
        assert!(rrelu_with_noise(&x, &t(vec![1.0; 4], &[4])).is_err());
        assert!(rrelu_with_noise(&x, &t(vec![1.0; 3], &[1, 3])).is_err());
        assert_eq!(tape.len(), before);
        let noise = t(vec![1.0, 0.37, 1.0], &[3]);
        let y = rrelu_with_noise(&x, &noise).unwrap();
        let g = grad_of(&tape, &x, &y);
        for (a, b) in g.iter().zip(noise.host_slice().iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
}
