//! 参照モデル定義: `LeNet`（PyTorch の定番 CNN。イシュー #2201・親
//! #2190）。
//!
//! ## 位置づけ
//!
//! `crates/facade/examples/models/mlp.rs`（`Mlp`）と同じ理由・同じ制約で、
//! 本ファイルも**単独で完結する**（他ファイルへの `super::`／`crate::`
//! 参照なし）。`crates/facade/examples/reference_models.rs`（runnable
//! example）と `crates/facade/tests/example_lenet_mnist.rs`（`#[path]`
//! 取り込みの統合テスト）の 2 箇所から読み込まれる。
//!
//! `facade` の公開面（`crates/facade/src/`）は変更していない
//! （`mlp.rs` モジュール doc「位置づけ（重要）」節と同じ保留理由）。
//!
//! ## PyTorch 参照定義（Conv2d 2 層・Dense 2 層版。古典 LeNet-5 の fc 3
//! 層版とは異なる。イシュー #2201 の受け入れ条件「Conv2d 2 層 + Dense 2
//! 層」に合わせた選択。`docs/reference-models-decision.md` 参照）
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
//! ## 重みレイアウトの契約
//!
//! `nn::Conv2d.weight` は `[out, in/groups, kH, kW]`
//! （`crates/autodiff/src/nn/conv.rs`）で PyTorch `nn.Conv2d.weight` と
//! **同一 shape**（転置不要）。`nn::Linear.weight` は
//! `[in_features, out_features]`（`mlp.rs` の `Mlp` と同じ転置契約）で
//! PyTorch `nn.Linear.weight` の `[out, in]` とは**転置の関係**。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

/// PyTorch チェックポイントとの層ごとの対応 1 件（`mlp.rs` の
/// `MlpParamMap` と同型。単独完結の方針のため型自体は個別に定義する）。
#[derive(Debug, Clone)]
pub struct LeNetParamMap {
    pub fandhe_key: String,
    pub pytorch_key: String,
    pub fandhe_shape: Vec<usize>,
    pub pytorch_shape: Vec<usize>,
    /// `true` の場合 fandhe → PyTorch で `weight` の転置が必要
    /// （`Linear` 系の 2 層のみ `true`。`Conv2d` 系は shape が同一のため
    /// `false`）。
    pub transpose: bool,
}

/// PyTorch 定番 LeNet（Conv2d 2 層 + Dense 2 層版）の参照実装
/// （`compat::Sequential` のラッパー。`mlp.rs` の `Mlp` と同じ設計）。
pub struct LeNet {
    model: Sequential,
    num_classes: usize,
}

impl LeNet {
    /// `num_classes` 分類の LeNet を構築する（本モジュール doc の
    /// PyTorch 参照定義固定。入力は `[N, 1, 28, 28]` 前提）。`seed` は
    /// 全 `Conv2d`／`Linear` 層の初期化に使う基準値（層ごとに
    /// `wrapping_add` でずらす。`mlp.rs` の `Mlp` と同じ方式だが、
    /// `Mlp::new`／`with_seed` の 2 段構えとは異なり LeNet は既定シードを
    /// 持たず常に呼び出し元が指定する）。
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

    /// 外部 `Tape` 上で forward を計算する（`mlp.rs` の `Mlp::forward`
    /// と同型）。
    pub fn forward<'t>(&self, tape: &'t Tape, x: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.model.forward(tape, x)
    }

    /// 推論の入口（`mlp.rs` の `Mlp::predict` と同型）。
    pub fn predict(&self, x: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
        self.model.predict(x)
    }

    /// 内部 `Sequential` への参照（`compile`／`fit`／`evaluate` 等用）。
    pub fn sequential(&self) -> &Sequential {
        &self.model
    }

    /// [`LeNet::sequential`] の可変版。
    ///
    /// # 契約（重要）
    ///
    /// `mlp.rs` の `Mlp::sequential_mut` と同じ契約: 本メソッド経由で
    /// 内部 `Sequential` の層構成を差し替えると
    /// [`LeNet::pytorch_param_map`] の対応表と実パラメータが不整合に
    /// なりうる（codex-review 指摘・イシュー #2201 PR #2320）。
    /// [`LeNet::pytorch_param_map`] は呼び出しのたびに
    /// `named_parameters()` と突き合わせて検証するため、差し替え後の
    /// 呼び出しは `Err` になる。本メソッドは `compile`・`train`・`eval`
    /// 等のモード切替・学習系 API 呼び出し用に限定して使うこと。
    pub fn sequential_mut(&mut self) -> &mut Sequential {
        &mut self.model
    }

    /// 構成値（分類クラス数）を返す（example・テストの表示用）。
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// PyTorch 参照定義との層ごとの重み対応表（AC5）。`fandhe_shape` は
    /// [`LeNet::new`] の構成値（本モジュール doc 固定の PyTorch 参照
    /// 定義）から直接書き下ろす（`named_parameters()` からの逆算はしない。
    /// `mlp.rs` の `Mlp::pytorch_param_map` と同じ非トートロジー方針）。
    /// そのうえで `named_parameters()` と突き合わせ、キー集合・shape が
    /// 完全一致することを検証する（[`LeNet::sequential_mut`] 経由で
    /// 内部構成が差し替えられていた場合に不整合を検出するため。
    /// codex-review 指摘・イシュー #2201 PR #2320）。
    pub fn pytorch_param_map(&self) -> Result<Vec<LeNetParamMap>, AutodiffError> {
        let c = self.num_classes;
        let out = vec![
            LeNetParamMap {
                fandhe_key: "0.weight".to_string(),
                pytorch_key: "conv1.weight".to_string(),
                fandhe_shape: vec![6, 1, 5, 5],
                pytorch_shape: vec![6, 1, 5, 5],
                transpose: false,
            },
            LeNetParamMap {
                fandhe_key: "0.bias".to_string(),
                pytorch_key: "conv1.bias".to_string(),
                fandhe_shape: vec![6],
                pytorch_shape: vec![6],
                transpose: false,
            },
            LeNetParamMap {
                fandhe_key: "3.weight".to_string(),
                pytorch_key: "conv2.weight".to_string(),
                fandhe_shape: vec![16, 6, 5, 5],
                pytorch_shape: vec![16, 6, 5, 5],
                transpose: false,
            },
            LeNetParamMap {
                fandhe_key: "3.bias".to_string(),
                pytorch_key: "conv2.bias".to_string(),
                fandhe_shape: vec![16],
                pytorch_shape: vec![16],
                transpose: false,
            },
            LeNetParamMap {
                fandhe_key: "7.weight".to_string(),
                pytorch_key: "fc1.weight".to_string(),
                fandhe_shape: vec![16 * 4 * 4, 120],
                pytorch_shape: vec![120, 16 * 4 * 4],
                transpose: true,
            },
            LeNetParamMap {
                fandhe_key: "7.bias".to_string(),
                pytorch_key: "fc1.bias".to_string(),
                fandhe_shape: vec![120],
                pytorch_shape: vec![120],
                transpose: false,
            },
            LeNetParamMap {
                fandhe_key: "9.weight".to_string(),
                pytorch_key: "fc2.weight".to_string(),
                fandhe_shape: vec![120, c],
                pytorch_shape: vec![c, 120],
                transpose: true,
            },
            LeNetParamMap {
                fandhe_key: "9.bias".to_string(),
                pytorch_key: "fc2.bias".to_string(),
                fandhe_shape: vec![c],
                pytorch_shape: vec![c],
                transpose: false,
            },
        ];

        let actual = self.model.named_parameters();
        if actual.len() != out.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "LeNet::pytorch_param_map: 対応表のエントリ数（{}）が実\
                 パラメータ数（{}）と一致しない（sequential_mut() 経由で\
                 内部構成が差し替えられた可能性がある）",
                out.len(),
                actual.len()
            )));
        }
        for entry in &out {
            let found = actual
                .iter()
                .find(|(key, _)| *key == entry.fandhe_key)
                .ok_or_else(|| {
                    AutodiffError::InvalidArgument(format!(
                        "LeNet::pytorch_param_map: キー '{}' が実パラメータに\
                         存在しない（sequential_mut() 経由で内部構成が\
                         差し替えられた可能性がある）",
                        entry.fandhe_key
                    ))
                })?;
            if found.1.shape() != entry.fandhe_shape.as_slice() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "LeNet::pytorch_param_map: キー '{}' の shape が対応表\
                     （{:?}）と実パラメータ（{:?}）で不一致\
                     （sequential_mut() 経由で内部構成が差し替えられた\
                     可能性がある）",
                    entry.fandhe_key,
                    entry.fandhe_shape,
                    found.1.shape()
                )));
            }
        }

        Ok(out)
    }
}
