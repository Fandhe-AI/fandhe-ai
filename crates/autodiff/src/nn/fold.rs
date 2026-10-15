//! `nn::Unfold`／`nn::Fold`（PyTorch `torch.nn.Unfold`／`torch.nn.Fold` 相当。イシュー #3039・親 #3018）。
//!
//! 役割: `Var::unfold`／`Var::fold`（実体は `crate::fold_ops` の自由関数。#2645・#2851）を
//! 構成引数ごと保持する**無状態**の薄い層ラッパー。新規 `Op`／`BackendOps`／VJP／カーネルは無い。
//! 公開形は `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §3（#3025）・
//! `docs/autodiff-fold-unfold-decision.md` §12・§14〜16 に従い、引数順は
//! `kernel_size, stride, padding, dilation`（`Fold` は先頭に `output_size`）で PyTorch の
//! `dilation, padding, stride` 順とは異なる。
//!
//! 呼び出し文脈: `Module` 実装（`nn/module.rs`）経由で `Sequential` 等から呼ばれる。facade の
//! `Sequential::add_fold`／`add_unfold` は #3040 で公開済み（保存は #3041 で kind `unfold`／`fold` として対応済み）。
//!
//! 構築時検査: `new` は `Var::unfold`／`fold` と同じ `Conv2dParams::new`（0 の kernel／stride／
//! dilation と `2*padding` オーバーフローを拒否）を再利用し、検査基準が食い違う別経路を作らない。
//! `Fold` の `output_size` の 0 は forward 時の `fold_out_shape` と同じ `ShapeMismatch` で
//! 前倒し拒否する。入力 shape に依存する不正（窓超過・`K`／`L` 不整合・rank）は forward 時に
//! 孤児ノード無しの型付きエラーで拒否される。

use fandhe_ai_tensor_core::{Conv2dParams, ShapeError};

use crate::error::AutodiffError;
use crate::var::Var;

/// `Conv2dParams::new` による引数検査のみを行う（検証済みの値は呼び出し側が保持する）。
fn validate(
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<(), AutodiffError> {
    Conv2dParams::new(kernel_size, stride, padding, dilation, 1)
        .map(|_| ())
        .map_err(AutodiffError::Backend)
}

/// PyTorch `torch.nn.Unfold` 相当。`[N, C, H, W]` を `[N, C·kH·kW, L]` へ展開する
/// [`Var::unfold`] の層ラッパー。バッチなし入力は非対応。
#[derive(Debug, Clone, PartialEq)]
pub struct Unfold {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
}

impl Unfold {
    /// 不正引数（0 の kernel／stride／dilation・`2*padding` オーバーフロー）は型付きエラー。
    pub fn new(
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self, AutodiffError> {
        validate(kernel_size, stride, padding, dilation)?;
        Ok(Self {
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// [`Var::unfold`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.unfold(self.kernel_size, self.stride, self.padding, self.dilation)
    }
}

/// PyTorch `torch.nn.Fold` 相当。`[N, C·kH·kW, L]` を `[N, C, H, W]` へ畳み戻す
/// （重なる窓は加算）[`Var::fold`] の層ラッパー。
#[derive(Debug, Clone, PartialEq)]
pub struct Fold {
    output_size: [usize; 2],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
}

impl Fold {
    /// 不正引数（`output_size` の 0・0 の kernel／stride／dilation・`2*padding` オーバーフロー）は
    /// 型付きエラー。
    pub fn new(
        output_size: [usize; 2],
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self, AutodiffError> {
        validate(kernel_size, stride, padding, dilation)?;
        let [h, w] = output_size;
        if h == 0 || w == 0 {
            // `tensor_core::fold::fold_out_shape` の forward 時拒否と同じ形。
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: vec![h, w],
                rhs: vec![1, 1],
            }));
        }
        Ok(Self {
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// [`Var::fold`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.fold(
            self.output_size,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
        )
    }
}
