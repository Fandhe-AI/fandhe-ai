//! ONNX `GlobalAveragePool`（イシュー #2200・親 #2185）オペ。
//!
//! 空間軸（`X.shape()[2..]`）全体を平均で縮約し、出力 shape は
//! `[N, C, 1, …, 1]`（rank は入力と同じ。ONNX 仕様の定義どおり）。
//! Issue 本文は「reduce（全軸・keepdims=false）への分解」を示唆するが、
//! ONNX `GlobalAveragePool` 仕様は空間軸のみを縮約し `N`／`C` 軸は保持
//! するため、受入基準（出力 shape が ONNX spec と一致）を優先し仕様どおり
//! 実装する（`ops/mod.rs` の「入力テンソル＋属性 → 出力テンソル」の
//! 単体演算方針にも従う直接ノード実装。イシュー #2200 実装計画「Issue
//! 記述と実装方針が食い違う点」節）。
//!
//! 数値契約（`backend-cpu::pooling::adaptive_avg_pool2d` の
//! `output_size=[1,1]` 経路と同一の走査順・同一の `f64` 逐次加算。窓が
//! 常に空間全体を覆うため両者は同値になる）: 各 `(n, c)` について空間
//! 要素を row-major 順に `acc: f64 += f64::from(v)` で加算し、
//! `(acc / count as f64) as f32` を返す。`count = spatial`。

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use super::error::OpError;

/// `GlobalAveragePool(X)` を計算する。`X` は rank 3 以上（`[N, C, d0, …]`）
/// を要求する（ONNX 仕様上 rank 2 以下は無効）。
///
/// 検証順序:
/// 1. `x.rank() < 3` なら [`OpError::RankMismatch`]
/// 2. `N`／`C` が 0 なら部分積を計算する前に空の出力を返す
/// 3. 空間要素数積（`spatial`）のオーバーフローは `checked_mul` で検査
/// 4. `spatial == 0`（`N・C > 0`）は 0 除算で `NaN` を静かに生成するため
///    [`OpError::EmptyNormalizedSet`] で拒否する（`LayerNormalization` の
///    正規化集合 0 件と同じ理由・同じ variant を再利用する）
pub fn global_average_pool(x: &Tensor<f32>) -> Result<Tensor<f32>, OpError> {
    if x.rank() < 3 {
        return Err(OpError::RankMismatch {
            op: "GlobalAveragePool",
            expected: 3,
            actual: x.rank(),
        });
    }

    let shape = x.shape();
    let (n, c) = (shape[0], shape[1]);
    let mut out_shape = shape.to_vec();
    for d in &mut out_shape[2..] {
        *d = 1;
    }

    if n == 0 || c == 0 {
        return Tensor::new(Vec::new(), &out_shape).map_err(OpError::from);
    }

    let spatial = shape[2..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(OpError::Shape(ShapeError::ElementCountOverflow))?;

    if spatial == 0 {
        return Err(OpError::EmptyNormalizedSet {
            op: "GlobalAveragePool",
            axis: 2,
        });
    }

    let xc = x.contiguous();
    let x_slice = xc
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("GlobalAveragePool(X)"))?;

    let mut out = vec![0f32; n * c];
    for ni in 0..n {
        for ch in 0..c {
            let base = (ni * c + ch) * spatial;
            let mut acc: f64 = 0.0;
            for sp in 0..spatial {
                acc += f64::from(x_slice[base + sp]);
            }
            out[ni * c + ch] = (acc / spatial as f64) as f32;
        }
    }

    Tensor::new(out, &out_shape).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank4_averages_spatial_axes() {
        // N=1, C=2, H=2, W=2. channel0=[1,2,3,4] mean=2.5, channel1=[5,6,7,8] mean=6.5
        let x = Tensor::<f32>::new((1..=8).map(|v| v as f32).collect(), &[1, 2, 2, 2]).unwrap();
        let y = global_average_pool(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2, 1, 1]);
        assert_eq!(y.get(&[0, 0, 0, 0]).unwrap(), 2.5);
        assert_eq!(y.get(&[0, 1, 0, 0]).unwrap(), 6.5);
    }

    #[test]
    fn rank3_averages_single_spatial_axis() {
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], &[1, 2, 3]).unwrap();
        let y = global_average_pool(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2, 1]);
        assert_eq!(y.get(&[0, 0, 0]).unwrap(), 2.0);
        assert_eq!(y.get(&[0, 1, 0]).unwrap(), 20.0);
    }

    #[test]
    fn rank5_supported() {
        let x = Tensor::<f32>::new((0..16).map(|v| v as f32).collect(), &[1, 2, 2, 2, 2]).unwrap();
        let y = global_average_pool(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2, 1, 1, 1]);
    }

    #[test]
    fn rank2_rejected() {
        let x = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = global_average_pool(&x).unwrap_err();
        assert!(matches!(
            err,
            OpError::RankMismatch {
                op: "GlobalAveragePool",
                expected: 3,
                actual: 2,
            }
        ));
    }

    #[test]
    fn zero_spatial_rejected() {
        let x = Tensor::<f32>::zeros(&[1, 2, 0, 3]).unwrap();
        let err = global_average_pool(&x).unwrap_err();
        assert!(matches!(
            err,
            OpError::EmptyNormalizedSet {
                op: "GlobalAveragePool",
                axis: 2,
            }
        ));
    }

    #[test]
    fn zero_batch_yields_empty_output() {
        let x = Tensor::<f32>::new(Vec::new(), &[0, 2, 3, 3]).unwrap();
        let y = global_average_pool(&x).unwrap();
        assert_eq!(y.shape(), &[0, 2, 1, 1]);
        assert!(y.contiguous().as_slice().unwrap().is_empty());
    }
}
