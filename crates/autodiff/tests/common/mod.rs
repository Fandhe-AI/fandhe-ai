//! 統合テスト共通の naive `BackendOps` フィクスチャ（TASK-12.1d・#164）。
//!
//! `crates/autodiff/tests/*.rs` は `autodiff` の公開 API のみを経由する
//! 別クレート扱いのため、クレート非公開の `eval.rs`（`src/test_support.rs`
//! が委譲する参照実装）を再利用できない。本モジュールは同じ意味論
//! （FMA 契約〈`f32::mul_add`〉・NumPy 互換ブロードキャスト）を独立に
//! 実装する（`tensor-core` の `pub` API のみを使用。`backend-cpu` 等の
//! 具体バックエンドクレートへは依存しない——`autodiff` の統合テストが
//! `backend-cpu` に依存すると `autodiff` 自身が具体バックエンドへ依存
//! しないという設計上の不変条件〈`docs/fusion-graph-design.md` §3.4〉の
//! 検証と矛盾するため）。
//!
//! 各テストファイルは `mod common;` で読み込み、`common::naive_ops()` を
//! `Tape::new_with_ops(...)` へ渡す。

#![allow(dead_code)] // テストファイルごとに使う関数が異なるため。

use tensor_core::{BackendError, BackendOps, Device, ShapeError, Tensor, reduce_out_shape};

pub struct NaiveOps;

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    let c = t.contiguous();
    c.as_slice().map(|s| s.to_vec()).unwrap_or_default()
}

fn build(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    // `Tensor::from_shape_fill` は `checked_numel` によるオーバーフロー
    // 検査を経る `Result` を返す（PR #403 codex-review P1 是正）。本
    // フィクスチャは shape 検査済みの出力のみを渡す契約のため実運用では
    // `Err` に到達しないが、テストコードでも `expect` の理由を明記する
    // （`.claude/rules/code-comment-style.md`）。
    Tensor::from_shape_fill(shape, |i| data.get(i).copied().unwrap_or(0.0)).expect(
        "build: 呼び出し元は shape 検査済みの出力のみを渡す契約（フィクスチャ内部不変条件）",
    )
}

fn broadcast_binary(
    lhs: &Tensor<f32>,
    rhs: &Tensor<f32>,
    op: impl Fn(f32, f32) -> f32,
) -> Result<Tensor<f32>, BackendError> {
    let (blhs, brhs) = lhs
        .broadcast_with(rhs)
        .map_err(BackendError::ShapeMismatch)?;
    let shape = blhs.shape().to_vec();
    let lhs_data = dense(&blhs);
    let rhs_data = dense(&brhs);
    let out: Vec<f32> = lhs_data
        .iter()
        .zip(rhs_data.iter())
        .map(|(&a, &b)| op(a, b))
        .collect();
    Ok(build(out, &shape))
}

fn unary(input: &Tensor<f32>, op: impl Fn(f32) -> f32) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    let data = dense(input);
    let out: Vec<f32> = data.into_iter().map(op).collect();
    build(out, &shape)
}

fn nan_propagating_max(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

fn reduce_axis(
    input: &Tensor<f32>,
    axis: usize,
    init: f32,
    op: impl Fn(f32, f32) -> f32,
) -> Vec<f32> {
    let shape = input.shape();
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let data = dense(input);
    let mut out = vec![init; outer * inner];
    for o in 0..outer {
        for a in 0..axis_len {
            for i in 0..inner {
                let src = (o * axis_len + a) * inner + i;
                let dst = o * inner + i;
                out[dst] = op(out[dst], data[src]);
            }
        }
    }
    out
}

impl BackendOps for NaiveOps {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        if a.shape().len() != 2 || b.shape().len() != 2 || a.shape()[1] != b.shape()[0] {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: a.shape().len(),
            }));
        }
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        let a_data = dense(a);
        let b_data = dense(b);
        let mut out = vec![0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0f32;
                for p in 0..k {
                    acc = a_data[i * k + p].mul_add(b_data[p * n + j], acc);
                }
                out[i * n + j] = acc;
            }
        }
        Ok(build(out, &[m, n]))
    }

    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        broadcast_binary(a, b, |x, y| x + y)
    }

    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        broadcast_binary(a, b, |x, y| x * y)
    }

    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(unary(a, |v| nan_propagating_max(v, 0.0)))
    }

    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(unary(a, f32::exp))
    }

    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Ok(unary(a, f32::tanh))
    }

    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = reduce_out_shape(a.shape(), dim).map_err(BackendError::ShapeMismatch)?;
        match dim {
            None => {
                let total: f32 = dense(a).into_iter().sum();
                Ok(build(vec![total], &out_shape))
            }
            Some(axis) => Ok(build(reduce_axis(a, axis, 0.0, |x, y| x + y), &out_shape)),
        }
    }

    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        let out_shape = reduce_out_shape(a.shape(), dim).map_err(BackendError::ShapeMismatch)?;
        match dim {
            None => {
                let m = dense(a)
                    .into_iter()
                    .fold(f32::NEG_INFINITY, nan_propagating_max);
                Ok(build(vec![m], &out_shape))
            }
            Some(axis) => Ok(build(
                reduce_axis(a, axis, f32::NEG_INFINITY, nan_propagating_max),
                &out_shape,
            )),
        }
    }
}

pub fn naive_ops() -> Box<dyn BackendOps + Send> {
    Box::new(NaiveOps)
}
