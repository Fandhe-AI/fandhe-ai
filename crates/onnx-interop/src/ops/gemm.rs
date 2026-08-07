//! ONNX `Gemm`（General Matrix Multiplication）オペ（TASK-7.2c）。
//!
//! `Y = alpha * (A' @ B') + beta * C`（`A'`/`B'` は `trans_a`/`trans_b` 適用後の 2 次元
//! テンソル。`C` は `Y` の shape へブロードキャスト可能な任意 shape で省略可）。
//! CPU 参照実装は丸め方針（FMA 契約）統一方針（`.claude/rules/coding-rust.md`）に従い
//! 内積の累積に `f32::mul_add` を用いる。

use tensor_core::Tensor;

use super::error::OpError;

/// `Gemm` の属性。ONNX proto の `AttributeProto` から後続タスク（デコード層との結線）で
/// 変換される想定であり、本モジュールは decode 層に依存しないプレーンな構造体で受け取る。
#[derive(Debug, Clone, Copy)]
pub struct GemmAttrs {
    pub alpha: f32,
    pub beta: f32,
    pub trans_a: bool,
    pub trans_b: bool,
}

impl Default for GemmAttrs {
    fn default() -> Self {
        GemmAttrs {
            alpha: 1.0,
            beta: 1.0,
            trans_a: false,
            trans_b: false,
        }
    }
}

/// `Gemm` を計算する。`a`／`b` は 2 次元のみ受け付ける（rank 不一致は
/// [`OpError::RankMismatch`]）。`c` を渡す場合は出力 shape `[m, n]` へブロードキャスト
/// 可能でなければならない（不可の場合 [`OpError::Shape`]）。
pub fn gemm(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    c: Option<&Tensor<f32>>,
    attrs: &GemmAttrs,
) -> Result<Tensor<f32>, OpError> {
    if a.rank() != 2 {
        return Err(OpError::RankMismatch {
            op: "Gemm(A)",
            expected: 2,
            actual: a.rank(),
        });
    }
    if b.rank() != 2 {
        return Err(OpError::RankMismatch {
            op: "Gemm(B)",
            expected: 2,
            actual: b.rank(),
        });
    }

    // `trans_a`/`trans_b`: transpose は zero-copy な stride 入替に留まるため、内積計算前に
    // `contiguous()` で行優先バッファへ実体化する（`as_slice` が連続領域を要求するため）。
    let a_eff = if attrs.trans_a {
        a.transpose(0, 1)?.contiguous()
    } else {
        a.contiguous()
    };
    let b_eff = if attrs.trans_b {
        b.transpose(0, 1)?.contiguous()
    } else {
        b.contiguous()
    };

    let (m, k) = (a_eff.shape()[0], a_eff.shape()[1]);
    let (k2, n) = (b_eff.shape()[0], b_eff.shape()[1]);
    if k != k2 {
        return Err(OpError::GemmDimMismatch {
            a: a_eff.shape().to_vec(),
            b: b_eff.shape().to_vec(),
        });
    }

    let a_slice = a_eff
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Gemm(A)"))?;
    let b_slice = b_eff
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("Gemm(B)"))?;

    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0f32;
            for p in 0..k {
                acc = a_slice[i * k + p].mul_add(b_slice[p * n + j], acc);
            }
            out[i * n + j] = attrs.alpha * acc;
        }
    }

    if let Some(c) = c {
        let c_b = c.broadcast_to(&[m, n])?.contiguous();
        let c_slice = c_b
            .as_slice()
            .ok_or(OpError::NonContiguousInternal("Gemm(C)"))?;
        for (o, &cv) in out.iter_mut().zip(c_slice.iter()) {
            *o += attrs.beta * cv;
        }
    }

    Tensor::new(out, &[m, n]).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_matmul_no_bias() {
        // A: [2,3], B: [3,2] -> Y: [2,2]（PyTorch/NumPy の A @ B と一致することを想定値で確認）。
        let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::<f32>::new(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        let y = gemm(&a, &b, None, &GemmAttrs::default()).unwrap();
        assert_eq!(y.shape(), &[2, 2]);
        // [1,2,3]·[7,9,11] = 58, [1,2,3]·[8,10,12] = 64
        // [4,5,6]·[7,9,11] = 139, [4,5,6]·[8,10,12] = 154
        assert_eq!(y.get(&[0, 0]).unwrap(), 58.0);
        assert_eq!(y.get(&[0, 1]).unwrap(), 64.0);
        assert_eq!(y.get(&[1, 0]).unwrap(), 139.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 154.0);
    }

    #[test]
    fn alpha_beta_and_bias_broadcast() {
        let a = Tensor::<f32>::new(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap(); // identity
        let b = Tensor::<f32>::new(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let c = Tensor::<f32>::new(vec![1.0, 1.0], &[2]).unwrap(); // ブロードキャスト bias
        let attrs = GemmAttrs {
            alpha: 2.0,
            beta: 0.5,
            trans_a: false,
            trans_b: false,
        };
        let y = gemm(&a, &b, Some(&c), &attrs).unwrap();
        // alpha * (A@B) = 2*B, + beta * c(broadcast) = +0.5
        assert_eq!(y.get(&[0, 0]).unwrap(), 2.0 * 2.0 + 0.5);
        assert_eq!(y.get(&[0, 1]).unwrap(), 2.0 * 3.0 + 0.5);
        assert_eq!(y.get(&[1, 0]).unwrap(), 2.0 * 4.0 + 0.5);
        assert_eq!(y.get(&[1, 1]).unwrap(), 2.0 * 5.0 + 0.5);
    }

    #[test]
    fn trans_a_and_trans_b() {
        // A: [3,2] (trans_a=true 適用後 [2,3])、B: [2,3] (trans_b=true 適用後 [3,2])
        let a = Tensor::<f32>::new(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[3, 2]).unwrap(); // A^T = [[1,2,3],[4,5,6]]
        let b = Tensor::<f32>::new(vec![7.0, 9.0, 11.0, 8.0, 10.0, 12.0], &[2, 3]).unwrap(); // B^T = [[7,8],[9,10],[11,12]]
        let attrs = GemmAttrs {
            alpha: 1.0,
            beta: 1.0,
            trans_a: true,
            trans_b: true,
        };
        let y = gemm(&a, &b, None, &attrs).unwrap();
        // 期待値は plain_matmul_no_bias と同じ A@B（A=[[1,2,3],[4,5,6]], B=[[7,8],[9,10],[11,12]]）。
        assert_eq!(y.get(&[0, 0]).unwrap(), 58.0);
        assert_eq!(y.get(&[1, 1]).unwrap(), 154.0);
    }

    #[test]
    fn rank_mismatch_rejected() {
        let a = Tensor::<f32>::zeros(&[3]).unwrap();
        let b = Tensor::<f32>::zeros(&[3, 2]).unwrap();
        let err = gemm(&a, &b, None, &GemmAttrs::default()).unwrap_err();
        assert!(matches!(
            err,
            OpError::RankMismatch {
                expected: 2,
                actual: 1,
                ..
            }
        ));
    }

    #[test]
    fn inner_dim_mismatch_rejected() {
        let a = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<f32>::zeros(&[4, 2]).unwrap();
        let err = gemm(&a, &b, None, &GemmAttrs::default()).unwrap_err();
        assert!(matches!(err, OpError::GemmDimMismatch { .. }));
    }
}
