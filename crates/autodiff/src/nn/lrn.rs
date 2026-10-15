//! `nn::LocalResponseNorm`（PyTorch `torch.nn.LocalResponseNorm` 相当。イシュー #3043・親 #3018）。
//!
//! 役割: `Var::local_response_norm`（実体は `crate::lrn_ops::local_response_norm`。#2646・#2851）を
//! 構成引数ごと保持する**無状態**の薄い層ラッパー。新規 `Op`／`BackendOps`／VJP／カーネルは無い。
//! 公開形は `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §3（#3025）の LocalResponseNorm 行に
//! 従い、引数順は `size, alpha, beta, k`。PyTorch の既定値は設けず全引数を明示する。
//!
//! 呼び出し文脈: `Module` 実装（`nn/module.rs`）経由で `Sequential` 等から呼ばれる。facade の
//! `Sequential::add_local_response_norm`・保存 kind・常駐経路の拒否は後続 issue の範囲。
//!
//! 構築時検査: `new` は `Var` 経路と同じ `LrnParams::new` と同一のエラー写像で `size == 0`・
//! 非有限 `alpha`／`beta`／`k` を前倒し拒否し、判定基準が食い違う別経路を作らない。rank・確保サイズは
//! 入力 shape 依存のため forward 時に孤児ノード無しの型付きエラーで拒否される。

use fandhe_ai_tensor_core::BackendError;
use fandhe_ai_tensor_core::lrn::LrnParams;

use crate::error::AutodiffError;
use crate::var::Var;

/// PyTorch `torch.nn.LocalResponseNorm` 相当。チャネル軸の局所応答正規化
/// （[`Var::local_response_norm`]）の層ラッパー。入力 rank は 3 以上。
#[derive(Debug, Clone, PartialEq)]
pub struct LocalResponseNorm {
    size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
}

impl LocalResponseNorm {
    /// `size == 0`・非有限の `alpha`／`beta`／`k` は型付きエラー（`AutodiffError::InvalidArgument`）。
    /// `size > C` は PyTorch と同じく受理する（窓は手動クリップ）。
    pub fn new(size: usize, alpha: f32, beta: f32, k: f32) -> Result<Self, AutodiffError> {
        // `lrn_ops::local_response_norm` 冒頭と同一の写像。
        LrnParams::new(size, alpha, beta, k).map_err(|e| match e {
            BackendError::InvalidArgument(msg) => AutodiffError::InvalidArgument(msg),
            other => AutodiffError::Backend(other),
        })?;
        Ok(Self {
            size,
            alpha,
            beta,
            k,
        })
    }

    /// 窓サイズ。
    pub fn size(&self) -> usize {
        self.size
    }

    /// `alpha`。
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// `beta`。
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// `k`。
    pub fn k(&self) -> f32 {
        self.k
    }

    /// [`Var::local_response_norm`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.local_response_norm(self.size, self.alpha, self.beta, self.k)
    }
}
