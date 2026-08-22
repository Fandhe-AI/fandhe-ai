//! ONNX `Relu`／`Sigmoid`（TASK-7.2c）に加え `Erf`（TASK-7.3c・#84）の要素ごと活性化
//! オペ。いずれも属性を持たない純粋な要素ごと写像であり、入力の shape をそのまま維持する。
//!
//! `Erf` は GELU（`0.5 * x * (1 + erf(x / sqrt(2)))`）の構成要素として Attention 系
//! （TASK-7.3c）で必要になる。Rust std に `erf` は存在せず、許容依存 8 区分
//! （`.claude/rules/deps-policy.md`）に数学関数クレート（libm 等）は無いため自前近似で
//! 実装する（依存追加はユーザー承認必須のため行わない）。

use fandhe_ai_tensor_core::Tensor;

use super::error::OpError;

/// 要素ごとに `f` を適用した新しいテンソルを返す共通実装。非 contiguous な入力
/// （transpose/narrow 後の view 等）は `contiguous()` で実体化してから走査する。
fn map_elementwise(
    op: &'static str,
    x: &Tensor<f32>,
    f: impl Fn(f32) -> f32,
) -> Result<Tensor<f32>, OpError> {
    let xc = x.contiguous();
    let slice = xc.as_slice().ok_or(OpError::NonContiguousInternal(op))?;
    let data: Vec<f32> = slice.iter().map(|&v| f(v)).collect();
    Tensor::new(data, xc.shape()).map_err(OpError::from)
}

/// NaN を伝播する `max`。`f32::max` は NaN 入力を暗黙に非 NaN 側へ潰す
/// （IEEE 754 の `maxNum` 系挙動）ため、`Relu` にそのまま使うと ONNX Runtime・
/// `fandhe_ai_autodiff::eval::relu`（`crates/autodiff/src/eval.rs`）の NaN 伝播動作と
/// 不整合になり、バックエンド数値一致検証で上流の数値破壊（NaN）が
/// 隠蔽されてしまう。両者の意味論を揃えるため同じ判定を用いる。
fn nan_propagating_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// `Relu(x) = max(x, 0)`。NaN 入力は `nan_propagating_max` により NaN のまま返す
/// （`fandhe_ai_autodiff::eval::relu` と同じ意味論。ONNX Runtime とも整合する）。
pub fn relu(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_elementwise("Relu", x, |v| nan_propagating_max(v, 0.0))
}

/// `Sigmoid(x) = 1 / (1 + exp(-x))`。
pub fn sigmoid(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_elementwise("Sigmoid", x, |v| 1.0 / (1.0 + (-v).exp()))
}

/// 誤差関数 `erf(x)` の有理多項式近似（Abramowitz & Stegun 7.1.26。最大絶対誤差
/// 1.5e-7）。`erf` は奇関数（`erf(-x) = -erf(x)`）であるため負値は絶対値で計算してから
/// 符号を戻す。f32 精度・バックエンド間数値一致の複合判定（相対誤差 1e-3 未満 または
/// 絶対誤差 1e-5 未満。`.claude/rules/coding-rust.md`）に対し、本近似の誤差上界
/// 1.5e-7 は十分小さく後続の GELU 計算へ悪影響を与えない。
fn erf_approx(x: f32) -> f32 {
    // 定数は原典（Abramowitz & Stegun, Handbook of Mathematical Functions, 7.1.26）の
    // 係数をそのまま用いる。
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_74;
    const A3: f32 = 1.421_413_8;
    const A4: f32 = -1.453_152_1;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / P.mul_add(ax, 1.0);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let y = 1.0 - poly * (-ax * ax).exp();
    sign * y
}

/// `Erf(x)`（誤差関数）。GELU（`0.5 * x * (1 + erf(x / sqrt(2)))`）の構成要素。
pub fn erf(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    map_elementwise("Erf", x, erf_approx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relu_clamps_negative_to_zero() {
        let x = Tensor::<f32>::new(vec![-2.0, -0.5, 0.0, 1.5, 3.0], &[5]).unwrap();
        let y = relu(&x).unwrap();
        assert_eq!(y.shape(), &[5]);
        let expected = [0.0, 0.0, 0.0, 1.5, 3.0];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(y.get(&[i]).unwrap(), e);
        }
    }

    #[test]
    fn relu_propagates_nan() {
        // `f32::max` は NaN を暗黙に 0.0 へ潰すため NaN 非伝播バグの回帰確認
        // （fandhe_ai_autodiff::eval::relu・ONNX Runtime との整合。#272 レビュー指摘）。
        let x = Tensor::<f32>::new(vec![f32::NAN, -1.0, 2.0], &[3]).unwrap();
        let y = relu(&x).unwrap();
        assert!(y.get(&[0]).unwrap().is_nan());
        assert_eq!(y.get(&[1]).unwrap(), 0.0);
        assert_eq!(y.get(&[2]).unwrap(), 2.0);
    }

    #[test]
    fn relu_on_non_contiguous_view() {
        let t = Tensor::<f32>::new(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        let y = relu(&tt).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.get(&[0, 0]).unwrap(), 0.0);
        assert_eq!(y.get(&[1, 0]).unwrap(), 2.0);
    }

    #[test]
    fn sigmoid_known_values() {
        let x = Tensor::<f32>::new(vec![0.0, 1.0, -1.0], &[3]).unwrap();
        let y = sigmoid(&x).unwrap();
        let tol = 1e-6;
        assert!((y.get(&[0]).unwrap() - 0.5).abs() < tol);
        assert!((y.get(&[1]).unwrap() - 0.731_058_6).abs() < 1e-5);
        assert!((y.get(&[2]).unwrap() - 0.268_941_4).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_near_boundary_does_not_collapse_to_exact_zero_or_one() {
        // f32 では `1.0 + exp(-x)` が `x >= 17` 付近から厳密に 1.0 へ丸まる
        // （`exp(-17) ≈ 4.14e-8` は 1.0 の ULP `2^-24 ≈ 5.96e-8` を下回るため）。
        // そのため「収縮しないこと」を実際に検証できる上限は ±16 が境界（実測値は
        // 本テスト直上のコメントではなく `sigmoid_saturates_to_correctly_rounded_extremes`
        // の説明を参照）。ここでは素朴な実装（`inf/inf` の NaN 化等）が本来の閾値より
        // 手前で誤って 0/1 に潰れていないかを固定化する。
        let x = Tensor::<f32>::new(vec![16.0, -16.0], &[2]).unwrap();
        let y = sigmoid(&x).unwrap();
        let pos = y.get(&[0]).unwrap();
        let neg = y.get(&[1]).unwrap();
        assert!(pos.is_finite() && (0.0..=1.0).contains(&pos));
        assert!(neg.is_finite() && (0.0..=1.0).contains(&neg));
        assert_ne!(
            pos, 1.0,
            "sigmoid(16.0) collapsed to exact 1.0 unexpectedly"
        );
        assert_ne!(
            neg, 0.0,
            "sigmoid(-16.0) collapsed to exact 0.0 unexpectedly"
        );
        assert!(pos > 0.999);
        assert!(neg < 0.001);
    }

    #[test]
    fn sigmoid_saturates_to_correctly_rounded_extremes() {
        // ±88／±1000 は f32 の丸め誤差の範囲内で厳密に 0.0/1.0 へ収束するのが
        // *正しい* 丸め結果である（バグではない）。真値 `1 - exp(-88) ≈ 1 - 6.05e-39`
        // は 1.0 の ULP（`2^-24 ≈ 5.96e-8`）より近く、f32 では 1.0 と区別不能。
        // 同様に `-1000` 側は `exp(1000)` が f32 上限（`f32::MAX ≈ 3.4e38`）を超えて
        // `inf` へオーバーフローし `1/(1+inf) = 0.0` となる。逆に `-88` は
        // `exp(88) ≈ 1.65e38` がまだ有限（非正規化数）に収まるため厳密には潰れない。
        // 本テストはこれら「収束するケース／しないケース」を実測値に基づき固定化し、
        // 弱い範囲チェックのみで実質何も検証しない状態（#297 Bugbot 指摘）を防ぐ。
        let x = Tensor::<f32>::new(vec![88.0, -88.0, 1000.0, -1000.0], &[4]).unwrap();
        let y = sigmoid(&x).unwrap();
        for i in 0..4 {
            let v = y.get(&[i]).unwrap();
            assert!(
                v.is_finite(),
                "sigmoid output must stay finite at index {i}: {v}"
            );
            assert!(
                (0.0..=1.0).contains(&v),
                "sigmoid output out of [0,1] at index {i}: {v}"
            );
        }
        // +88／+1000: 真値との差が 1.0 の ULP 未満のため厳密に 1.0 へ丸まる（正しい丸め）。
        assert_eq!(y.get(&[0]).unwrap(), 1.0);
        assert_eq!(y.get(&[2]).unwrap(), 1.0);
        // -88: `exp(88)` はまだ f32 で有限（非正規化数）のため厳密な 0.0 には潰れない。
        assert_ne!(y.get(&[1]).unwrap(), 0.0);
        assert!(y.get(&[1]).unwrap() < 0.001);
        // -1000: `exp(1000)` が f32 上限を超えてオーバーフローするため厳密に 0.0。
        assert_eq!(y.get(&[3]).unwrap(), 0.0);
    }

    #[test]
    fn erf_known_values() {
        let x = Tensor::<f32>::new(vec![0.0, 1.0, -1.0, 0.5, 2.0], &[5]).unwrap();
        let y = erf(&x).unwrap();
        let tol = 1e-6;
        assert!((y.get(&[0]).unwrap() - 0.0).abs() < tol);
        assert!((y.get(&[1]).unwrap() - 0.842_700_8).abs() < tol);
        assert!((y.get(&[2]).unwrap() - (-0.842_700_8)).abs() < tol);
        assert!((y.get(&[3]).unwrap() - 0.520_499_9).abs() < tol);
        assert!((y.get(&[4]).unwrap() - 0.995_322_3).abs() < tol);
    }

    #[test]
    fn erf_asymptotic_values_converge_to_one() {
        let x = Tensor::<f32>::new(vec![10.0, -10.0], &[2]).unwrap();
        let y = erf(&x).unwrap();
        assert!((y.get(&[0]).unwrap() - 1.0).abs() < 1e-7);
        assert!((y.get(&[1]).unwrap() - (-1.0)).abs() < 1e-7);
    }

    #[test]
    fn erf_is_odd_function() {
        let x = Tensor::<f32>::new(vec![0.3, 1.7, 2.5], &[3]).unwrap();
        let neg_x = Tensor::<f32>::new(vec![-0.3, -1.7, -2.5], &[3]).unwrap();
        let y = erf(&x).unwrap();
        let y_neg = erf(&neg_x).unwrap();
        for i in 0..3 {
            assert!((y.get(&[i]).unwrap() + y_neg.get(&[i]).unwrap()).abs() < 1e-6);
        }
    }

    #[test]
    fn erf_on_non_contiguous_view() {
        let t = Tensor::<f32>::new(vec![0.0, 1.0, -1.0, 0.5], &[2, 2]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        let y = erf(&tt).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        // tt = [[0.0, -1.0], [1.0, 0.5]]（transpose 後の論理配置）
        assert!((y.get(&[0, 0]).unwrap() - 0.0).abs() < 1e-6);
        assert!((y.get(&[1, 0]).unwrap() - 0.842_700_8).abs() < 1e-6);
    }
}
