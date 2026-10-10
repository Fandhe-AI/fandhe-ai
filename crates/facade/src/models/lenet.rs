//! 参照モデル `LeNet`（PyTorch の定番 CNN。Conv2d 2 層 + Dense 2 層版。イシュー #2201 の
//! examples 実装を #2974 で `fandhe_ai::models` として公開したもの）の実体。
//!
//! 公開パスは `models/mod.rs` の `pub use` のみ（本モジュール自体は非公開）。`Mlp` と同じ
//! 設計で、`compat::Sequential::add_*` だけを組み合わせた薄いラッパー。保存・読み込みは
//! [`LeNet::sequential`] 経由で既存の `compat::save_model`／`compat::load_model` を使う。
//!
//! # PyTorch 参照定義（古典 LeNet-5 の fc 3 層版とは異なる。`docs/reference-models-decision.md`）
//!
//! ```text
//! conv1 = Conv2d(1, 6, 5)                  # 28 -> 24 (padding 0)
//! relu -> max_pool2d(2)                    # 24 -> 12
//! conv2 = Conv2d(6, 16, 5)                 # 12 -> 8
//! relu -> max_pool2d(2)                    # 8  -> 4
//! flatten(1)                               # 16*4*4 = 256
//! fc1 = Linear(256, 120) -> relu
//! fc2 = Linear(120, num_classes)
//! ```
//!
//! `Conv2d.weight` は PyTorch と同一 shape、`Linear.weight` は `[in, out]` で PyTorch の
//! `[out, in]` とは転置の関係。PyTorch 対応表は examples 側に置き、公開面には含めない。

use crate::compat::Sequential;
use crate::{AutodiffError, Tape, Tensor, Var};

/// PyTorch 定番 LeNet（Conv2d 2 層 + Dense 2 層版）の参照実装
/// （`compat::Sequential` のラッパー。`Mlp` と同じ設計）。
pub struct LeNet {
    model: Sequential,
    num_classes: usize,
}

impl LeNet {
    /// `num_classes` 分類の LeNet を構築する（本モジュール doc の PyTorch 参照定義固定。
    /// 入力は `[N, 1, 28, 28]` 前提）。`seed` は全 `Conv2d`／`Linear` 層の初期化の基準値
    /// （層ごとに `wrapping_add` でずらす）。`Mlp` と異なり既定シードは持たず、常に
    /// 呼び出し元が指定する。
    ///
    /// # Errors
    ///
    /// `num_classes == 0` の場合は `AutodiffError::InvalidArgument`。
    pub fn new(num_classes: usize, seed: u64) -> Result<Self, AutodiffError> {
        if num_classes == 0 {
            return Err(AutodiffError::InvalidArgument(
                "LeNet::new: num_classes は 0 より大きい必要がある".to_string(),
            ));
        }

        let model = Sequential::new()
            // index 0: conv1（1 -> 6, kernel 5, stride 1, padding 0, dilation 1, groups 1）。
            .add_conv2d(1, 6, [5, 5], [1, 1], [0, 0], [1, 1], 1, seed)?
            .add_relu() // index 1
            .add_max_pool2d([2, 2], None, [0, 0], [1, 1])? // index 2
            // index 3: conv2（6 -> 16）。
            .add_conv2d(
                6,
                16,
                [5, 5],
                [1, 1],
                [0, 0],
                [1, 1],
                1,
                seed.wrapping_add(1),
            )?
            .add_relu() // index 4
            .add_max_pool2d([2, 2], None, [0, 0], [1, 1])? // index 5
            .add_flatten(1, 3) // index 6: [N, 16, 4, 4] -> [N, 256]
            // index 7: fc1（256 -> 120）。
            .add_linear(16 * 4 * 4, 120, seed.wrapping_add(2))?
            .add_relu() // index 8
            // index 9: fc2（120 -> num_classes）。
            .add_linear(120, num_classes, seed.wrapping_add(3))?;

        Ok(LeNet { model, num_classes })
    }

    /// 外部 `Tape` 上で forward を計算する（`Mlp::forward` と同型）。
    pub fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.model.forward(tape, x)
    }

    /// 推論の入口（`Mlp::predict` と同型）。
    pub fn predict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        self.model.predict(x)
    }

    /// 内部 `Sequential` への参照（`compile`／`fit`／`evaluate`／`save_model` 等用）。
    pub fn sequential(&self) -> &Sequential {
        &self.model
    }

    /// [`LeNet::sequential`] の可変版。
    ///
    /// # 契約
    ///
    /// `Mlp::sequential_mut` と同じ契約: 層構成そのものの差し替えは想定しない
    /// （差し替えると examples 側の PyTorch 対応表関数が `Err` を返す）。モード切替・
    /// 学習系 API の呼び出しに限定して使うこと。
    pub fn sequential_mut(&mut self) -> &mut Sequential {
        &mut self.model
    }

    /// 構成値（分類クラス数）を返す。
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}
