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

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::nn::init::{BIAS_SEED_SALT, WEIGHT_SEED_SALT, derive_seed, uniform_init};
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
    /// 同じ範囲・独立した導出シードで初期化する（weight/bias のシード
    /// 導出は `nn/init.rs::derive_seed` を参照。「同じ呼び出しシードから
    /// 2 系統を作る」設計上、単純な線形オフセットでは連番呼び出しシード
    /// で層を重ねる使い方の際に系列が衝突しうるため、ビットミキシング
    /// で独立させている）。
    ///
    /// `out_features == 0` は実行時に構築できる空テンソル（`tensor-core`
    /// はサイズ 0 軸を妥当な shape として扱う。`ops_shape.rs` の
    /// `matmul_zero_size_axis` 参照）としてそのまま受理する。一方
    /// `in_features == 0` は `bound = 1/√in_features` が非有限（inf）に
    /// なるため、テンソル生成に進む前に引数として弾く。この失敗は
    /// 「生成済み・生成中のテンソルの shape 不整合」ではなく「コンス
    /// トラクタ引数がそもそも構築不可能」という性質のため、
    /// `tensor-core::ShapeError` の既存 variant（`RankMismatch` 等）は
    /// いずれも意味的に適合せず、`AutodiffError::InvalidArgument` で
    /// 表現する（`error.rs` の doc 参照。review 指摘 #91: 当初
    /// `ShapeError::AxisOutOfRange` へ転用していたが撤回した）。
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Linear, AutodiffError> {
        if in_features == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Linear::new: in_features must be > 0 (1/sqrt(in_features) would be non-finite)"
                    .to_string(),
            ));
        }
        let bound = 1.0 / (in_features as f32).sqrt();
        let weight_seed = derive_seed(seed, WEIGHT_SEED_SALT);
        let weight_data = uniform_init(in_features * out_features, bound, weight_seed);
        let weight = Tensor::new(weight_data, &[in_features, out_features])?;
        let bias = if bias {
            let bias_seed = derive_seed(seed, BIAS_SEED_SALT);
            let bias_data = uniform_init(out_features, bound, bias_seed);
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
        // `weight.shape()[0]`（in_features）が 0 の場合、`tensor-core::ops_shape::matmul_out_shape`
        // は `lhs[1]==rhs[0]==0` を zero-K パスとして妥当な shape 扱いで許容してしまうため、
        // forward はエラーにならず全要素 0.0 の出力を静かに返す（review 指摘 #91）。
        // `Linear::new` が同条件を `AutodiffError::InvalidArgument` で明示的に拒否しているのに
        // 対し、safetensors ロード等の外部由来パラメータ入口である `from_parameters` がこの
        // 検証を欠くと、壊れた／欠損した checkpoint（shape が `[0, N]` に縮退したもの）を
        // エラーにせず読み込み、学習・推論が常時ゼロ出力のまま進行しうる（A03: 外部由来
        // パラメータを計算前に検証する契約。`.claude/rules/security.md`）。`out_features == 0`
        // は妥当な shape として引き続き許容する（`new` と対称。docstring 参照）。
        if weight.shape()[0] == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Linear::from_parameters: weight.shape()[0] (in_features) must be > 0 \
                 (zero-K matmul would silently produce an all-zero output)"
                    .to_string(),
            ));
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

    /// [`Self::forward`] の epilogue 融合版（イシュー #1044・`docs/
    /// kernel-fusion.md` §2.2「学習経路への結線」）。`y = act(input.matmul(
    /// weight) (+ bias))` を `Var::linear_act`（`var.rs`）経由で 1 ノード
    /// （`crate::tape::Op::LinearAct`。非公開のためコードスパン表記で
    /// 参照しリンク化しない）として記録し、`BackendOps::
    /// gemm_bias_act`（epilogue 融合カーネル。CPU／CUDA／Metal とも
    /// オーバーライド済み）へ直接委ねる。
    ///
    /// 唯一の呼び出し元は `fandhe_ai_facade::compat::sequential::
    /// Sequential`（次層が `ReLU` かを先読みし、その場合のみ
    /// `Activation::Relu` を渡して `ReLU` 層自体のノード追加をスキップ
    /// する。次層が `ReLU` でなければ `Activation::None` で bias のみ
    /// 融合する）。`Var::linear_act` が bias の broadcast 可否を
    /// `Var::add` と同じ判定で検査するため、本メソッド自体は追加の
    /// shape 検査を持たない。
    pub fn forward_with_activation(
        &self,
        input: &Var<'t>,
        act: fandhe_ai_tensor_core::Activation,
    ) -> Result<Var<'t>, AutodiffError> {
        input.linear_act(&self.weight, self.bias.as_ref(), act)
    }
}
