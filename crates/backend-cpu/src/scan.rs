//! 累積和／累積積カーネル（`torch.cumsum`／`torch.cumprod` 相当。
//! イシュー #1731）。
//!
//! [`fandhe_ai_tensor_core::BackendOps::cumsum`]／[`BackendOps::cumprod`]
//! （`ops.rs`）の CPU 実装本体。呼び出し元（`ops.rs`）が `dim` を
//! [`fandhe_ai_tensor_core::reduce_out_shape`] で再検査してから本モジュール
//! へ委譲する契約のため、本モジュール自身も同じ検査を独立に行う
//! （fail-closed 境界の二重化。`.claude/rules/security.md` A08。
//! `gather_scatter.rs` と同じ設計判断: `Var` を経由せず本モジュールを
//! 直接呼ぶ経路でも範囲外 `dim` による誤動作・panic を防ぐ）。
//!
//! **数値契約**（[`fandhe_ai_tensor_core::BackendOps::cumsum`]／
//! `cumprod` doc と同一。`.claude/rules/coding-rust.md` の f64
//! アキュムレータ方針を forward の scan へ拡張したもの）: `dim` 以外の
//! 軸の組（lane）ごとに `f64` アキュムレータを保持し、`dim` の添字
//! 昇順に逐次計算する。各ステップの出力はその時点のアキュムレータを
//! `f32` へ downcast したスナップショットであり、次ステップは
//! downcast 後の `f32` を読み戻さない。`crates/autodiff/src/eval.rs::
//! cumsum_along`／`cumprod_along`（`BackendOps` が `Unsupported` を
//! 返したときのホストフォールバック）と**同一アルゴリズム**であり、
//! 出力は `to_bits` で完全一致する契約とする（`tests/scan_parity.rs`
//! で検証）。
//!
//! `Tensor::contiguous()` で実体化してから `as_slice()`（常に `Some`
//! を返す契約。`ops.rs::gemm_contiguity_fail_safe` 近傍のコメント
//! 参照）で読むため、strided（transpose view 等）入力も正しく扱える。
//! 並列化（rayon）は行わない単一スレッド逐次実装（lane 間は独立の
//! ため将来並列化しても bit 同一は保たれるが、本 issue のスコープ外。
//! `.claude/rules/out-of-scope-tracking.md` 対象）。`unsafe`／
//! `unwrap`／`expect` は使わない（`.claude/rules/coding-rust.md`）。

use fandhe_ai_tensor_core::{ShapeError, Tensor, reduce_out_shape};

/// [`fandhe_ai_tensor_core::BackendOps::cumsum`] の CPU 実装本体。
/// `dim` を [`reduce_out_shape`] で独立に再検査してから逐次走査する。
pub(crate) fn cumsum(x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, ShapeError> {
    reduce_out_shape(x.shape(), Some(dim))?;
    let shape = x.shape().to_vec();
    // 要素数ゼロ（shape のいずれかの次元が 0）のとき、`shape[..dim]`／
    // `shape[dim+1..]` の部分積は無関係な次元を含みうり usize
    // オーバーフローしうる（`eval::cumsum_along` 冒頭と同じ理由）ため、
    // outer/axis_len/inner を計算する前に空出力へ早期 return する。
    if shape.contains(&0) {
        return Tensor::new(Vec::new(), &shape);
    }
    let x_c = x.contiguous();
    let data = x_c.as_slice().unwrap_or(&[]);
    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc: f64 = 0.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                acc += data[idx] as f64;
                out[idx] = acc as f32;
            }
        }
    }
    Tensor::new(out, &shape)
}

/// [`fandhe_ai_tensor_core::BackendOps::cumprod`] の CPU 実装本体。
/// [`cumsum`] と同じ検査・lane 構造だが、アキュムレータは `1.0` から
/// 開始し積を蓄積する。
pub(crate) fn cumprod(x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, ShapeError> {
    reduce_out_shape(x.shape(), Some(dim))?;
    let shape = x.shape().to_vec();
    if shape.contains(&0) {
        return Tensor::new(Vec::new(), &shape);
    }
    let x_c = x.contiguous();
    let data = x_c.as_slice().unwrap_or(&[]);
    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc: f64 = 1.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                acc *= data[idx] as f64;
                out[idx] = acc as f32;
            }
        }
    }
    Tensor::new(out, &shape)
}
