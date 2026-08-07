//! 全結合層（Linear/Dense）。`nn` モジュール（TASK-9.1a・#91）の第 1 分割。
//!
//! `docs/spec/05-tasks.md` TASK-9.1（REQ-9・M3）に基づき、自作テンソル
//! （`tensor-core`）・自作 autodiff（`Tape`/`Var`）の上に PyTorch
//! `nn.Linear` 相当の層を再実装する。参照実装は PoC-v2-2 の MLP
//! （`docs/spec/03-poc/poc-v2-2-autodiff/code/rust/src/mlp.rs`）の
//! `x.matmul(w) → add(bias)` 構成。
//!
//! **`Tape` ライフサイクルとの関係**（`tape.rs` の「学習ループでの運用」
//! 節参照）: `Tape` はステップごとに生成・破棄される前提のため、
//! パラメータの永続的な値（`Tensor<f32>`）を保持する `Linear` 本体と、
//! 特定ステップのテープへ登録した後の `Var` を保持する `LinearVars` を
//! 分離する。呼び出し元は毎ステップ `Linear::bind(&tape)` で
//! `LinearVars` を作り直し、`forward` を呼ぶ。

use tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::init::uniform_init;
use crate::tape::Tape;
use crate::var::Var;

/// 全結合層のパラメータ本体。`weight` は `[in_features, out_features]`
/// （PoC-v2-2 の `x.matmul(w)` 慣習。転置は持たない）、`bias` は
/// `Some` の場合 `[out_features]`。
pub struct Linear {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
}

impl Linear {
    /// 決定的シードで `U(-1/√in_features, 1/√in_features)` の一様初期化
    /// を行う（PyTorch `nn.Linear` 既定初期化と同じ有効範囲。
    /// `nn/init.rs` 参照）。`bias: true` で `[out_features]` の bias を
    /// 同じ範囲・後続シードで初期化する。
    ///
    /// `in_features == 0` または `out_features == 0` は
    /// `ShapeError::ElementCountOverflow` ではなく実行時に構築できる
    /// 空テンソル（`tensor-core` はサイズ 0 軸を妥当な shape として扱う。
    /// `ops_shape.rs` の `matmul_zero_size_axis` 参照）だが、
    /// `1/√0` は非有限（inf）になるため、本関数は入口で
    /// `ShapeError::ElementCountOverflow` 相当ではなく明示的に
    /// `RankMismatch` とは別の失敗要因が必要になる。ここでは
    /// `in_features == 0` を `ShapeError::AxisOutOfRange { axis: 0,
    /// rank: 0 }` として弾く（他に適合する既存 variant がないため、
    /// 「0 番目の軸に有効な次元がない」の意味で転用する）。
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Linear, AutodiffError> {
        if in_features == 0 {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: 0,
                rank: 0,
            }));
        }
        let bound = 1.0 / (in_features as f32).sqrt();
        let weight_data = uniform_init(in_features * out_features, bound, seed);
        let weight = Tensor::new(weight_data, &[in_features, out_features])?;
        let bias = if bias {
            // weight と異なるシード系列にするため seed を 1 ずらす
            // （同一乱数系列の使い回しによる重みとバイアスの相関を避ける）。
            let bias_data = uniform_init(out_features, bound, seed.wrapping_add(1));
            Some(Tensor::new(bias_data, &[out_features])?)
        } else {
            None
        };
        Ok(Linear { weight, bias })
    }

    /// 明示的な重み・バイアスから構築する（テスト・将来の safetensors
    /// ロード経路（REQ-7 系。本イシューではスコープ外）向けの入口）。
    /// `weight` は rank 2、`bias` を渡す場合は rank 1 かつ
    /// `bias.shape() == [weight.shape()[1]]` を要求する（A03: 外部由来
    /// パラメータを計算前に検証する契約。`.claude/rules/security.md`）。
    pub fn from_parameters(
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
    ) -> Result<Linear, AutodiffError> {
        if weight.rank() != 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: weight.rank(),
            }));
        }
        if let Some(ref b) = bias {
            if b.rank() != 1 {
                return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                    expected: 1,
                    actual: b.rank(),
                }));
            }
            let out_features = weight.shape()[1];
            if b.shape() != [out_features] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![out_features],
                }));
            }
        }
        Ok(Linear { weight, bias })
    }

    /// このステップの `tape` へ `weight`/`bias` を葉ノードとして登録し、
    /// `forward` を呼べる `LinearVars` を返す。`Tape::var`（`tape.rs`）を
    /// 経由するため、返る `Var` はこの `tape` に属する（クロステープ
    /// 検査の対象になる）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> LinearVars<'t> {
        let weight = tape.var(&self.weight);
        let bias = self.bias.as_ref().map(|b| tape.var(b));
        LinearVars { weight, bias }
    }

    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.bias.as_ref()
    }
}

/// `Linear::bind` が返す、1 ステップ分のテープに登録済みパラメータ。
/// `weight`/`bias` を公開する理由: `Tape::backward` 後に
/// `Gradients::get(&vars.weight)` で勾配を取り出すのは呼び出し側
/// （optimizer。#192・本イシューではスコープ外）の責務であり、
/// `LinearVars` 自身は勾配更新 API を持たない。
pub struct LinearVars<'t> {
    pub weight: Var<'t>,
    pub bias: Option<Var<'t>>,
}

impl<'t> LinearVars<'t> {
    /// `y = input.matmul(weight) (+ bias)`。`input` は `[batch,
    /// in_features]`（2 次元。`Var::matmul` の rank 制約に従う）を
    /// 想定し、出力は `[batch, out_features]`。bias 加算は
    /// `Var::add` の broadcast（`[batch, out_features]` + `[out_features]`）
    /// に委ねるため、bias 勾配の batch 軸縮約は既存の `reduce_to_shape`
    /// 機構（`grad.rs`）でそのまま成立する（TASK-9.1a 計画 §2 参照）。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let y = input.matmul(&self.weight)?;
        match &self.bias {
            Some(bias) => y.add(bias),
            None => Ok(y),
        }
    }
}
