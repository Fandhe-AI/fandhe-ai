//! `Hardshrink`・`Softshrink` の nn 層（イシュー #3817・親 #3813。設計記録
//! `docs/nn-shrink-bilinear-channel-shuffle-decision.md` §1）。
//!
//! 各層の `forward` は `crate::shrink_ops` の自由関数への委譲で、`Module` 実装も本ファイルに置く
//! （`nn/softmin_threshold.rs` と同型）。`lambd` は構築時に検査する（有限かつ `>= 0`。保存ファイル
//! 経由の値も `facade::compat::Sequential::add_*` を通って同じ検査を受ける）。
//!
//! `forward_host` は未提供（`supports_forward_host() == false`）: tape 経由 forward との bit 一致を
//! 検証していない経路を主張しないため。本モジュールは facade から再エクスポートしない。

use crate::error::AutodiffError;
use crate::nn::Module;
use crate::shrink_ops as ops;
use crate::tape::Tape;
use crate::var::Var;

/// PyTorch `nn.Hardshrink(lambd)` 相当。PyTorch の既定値 0.5 は呼び出し側が明示する。
#[derive(Debug, Clone, Copy)]
pub struct Hardshrink {
    lambd: f32,
}

impl Hardshrink {
    /// `lambd`（有限かつ `>= 0`）を指定して構築する。違反は `InvalidArgument`。
    pub fn new(lambd: f32) -> Result<Self, AutodiffError> {
        ops::validate_lambd("Hardshrink::new", lambd)?;
        Ok(Self { lambd })
    }

    /// 構築済みの `lambd`。
    pub fn lambd(&self) -> f32 {
        self.lambd
    }

    /// [`ops::hardshrink`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        ops::hardshrink(input, self.lambd)
    }
}

impl Module for Hardshrink {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Hardshrink::forward(self, input)
    }

    /// モジュール doc 参照: `forward_host` 未提供のため `false`。
    fn supports_forward_host(&self) -> bool {
        false
    }
}

/// PyTorch `nn.Softshrink(lambd)` 相当。
#[derive(Debug, Clone, Copy)]
pub struct Softshrink {
    lambd: f32,
}

impl Softshrink {
    /// `lambd`（有限かつ `>= 0`）を指定して構築する。違反は `InvalidArgument`。
    pub fn new(lambd: f32) -> Result<Self, AutodiffError> {
        ops::validate_lambd("Softshrink::new", lambd)?;
        Ok(Self { lambd })
    }

    /// 構築済みの `lambd`。
    pub fn lambd(&self) -> f32 {
        self.lambd
    }

    /// [`ops::softshrink`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        ops::softshrink(input, self.lambd)
    }
}

impl Module for Softshrink {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Softshrink::forward(self, input)
    }

    /// モジュール doc 参照: `forward_host` 未提供のため `false`。
    fn supports_forward_host(&self) -> bool {
        false
    }
}
