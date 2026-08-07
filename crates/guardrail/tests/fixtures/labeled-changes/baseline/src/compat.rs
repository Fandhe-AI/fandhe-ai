//! v1（PoC-1〜3）の互換 API 層を v2 自作コア（`autodiff::nn::Linear`）の
//! 上に再構築したもの。`np.array(...)` 風のテンソル生成関数と、Keras
//! `Sequential` 風のレイヤー積み上げ API を `autodiff::Var`/`Tape` の
//! 上に薄くラップする（REQ-9「互換 API 層は自作コアの上の薄いラッパーに
//! 徹する」と同じ精神を、本評価データセット内でも踏襲する）。
//!
//! TASK-4.2a 検証題材のうち G3（`add_linear` の公開シグネチャ破壊）・
//! G4（大量ドキュメントコメント追加）の変更対象。

use autodiff::AutodiffError;
use autodiff::Tape;
use autodiff::Var;
use autodiff::nn::Linear;
use tensor_core::{ShapeError, Tensor};

use crate::activations;

/// numpy 風のテンソル生成: `np::array(data, [rows, cols])`。
pub fn array(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<f32>, ShapeError> {
    Tensor::new(data, shape)
}

/// Sequential 内で扱うレイヤー種別（互換 API 層の最小語彙）。
enum LayerKind {
    Linear(Box<Linear>),
    Relu,
    /// 機能追加題材: 新設した LeakyReLU レイヤー。
    LeakyRelu(f64),
}

/// Keras `Sequential` 風にレイヤーを積み上げる薄いビルダー。
pub struct Sequential {
    layers: Vec<LayerKind>,
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Sequential { layers: vec![] }
    }

    /// `model.add_linear(in_features, out_features, seed)` のような
    /// Python 慣習寄りの命名。
    pub fn add_linear(
        mut self,
        n_in: usize,
        n_out: usize,
        seed: u64,
    ) -> Result<Self, AutodiffError> {
        self.layers.push(LayerKind::Linear(Box::new(Linear::new(
            n_in, n_out, true, seed,
        )?)));
        Ok(self)
    }

    pub fn add_relu(mut self) -> Self {
        self.layers.push(LayerKind::Relu);
        self
    }

    /// 機能追加題材: 小さな機能追加要求として新設した LeakyReLU レイヤー
    /// の追加 API。
    pub fn add_leaky_relu(mut self, negative_slope: f64) -> Self {
        self.layers.push(LayerKind::LeakyRelu(negative_slope));
        self
    }

    pub fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let mut out = *x;
        for layer in &self.layers {
            out = match layer {
                LayerKind::Linear(l) => l.bind(tape).forward(&out)?,
                LayerKind::Relu => activations::relu(&out),
                LayerKind::LeakyRelu(slope) => activations::leaky_relu(tape, &out, *slope),
            };
        }
        Ok(out)
    }
}
