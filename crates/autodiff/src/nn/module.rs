//! 共通 `Module` trait（TASK-9.2a・#95）。
//!
//! `Sequential`（当時 `fandhe_ai_autodiff::compat::sequential`。TASK-9.4・#411 で
//! `fandhe_ai::compat::sequential` へ移設）がレイヤーの列を
//! `Vec<Box<dyn Module>>` として保持し、種類の異なる `nn` の部品
//! （`Linear`・活性化関数）を統一シグネチャで呼べるようにするための
//! 最小 trait。`nn/mod.rs` が「共通 `Module` trait の定義は
//! `compat::Sequential` 設計時に確定する」としていた点を本イシューで
//! 確定する。
//!
//! シグネチャに `tape: &'t Tape` を含める理由: `Linear` は
//! `Linear::bind(&tape)` でそのステップの葉ノードを毎回登録してから
//! でないと forward できない（`nn/linear.rs` の `Tape` ライフサイクル
//! 節参照）。活性化関数側は `tape` を使わないが、`Box<dyn Module>` を
//! 均一に扱うため同じ引数を受け取る。この「毎呼び出しで葉ノードを
//! 登録し直す」契約は推論・1 ステップ forward 用である。学習（勾配
//! 取得・パラメータ更新）は #294（`compat::Sequential::bind`・
//! `compat/sequential.rs` の `SequentialVars`）で対応済み: 下記
//! `as_linear`/`as_linear_mut` が `Sequential` から学習可能パラメータ
//! （`Linear`）を層順に取り出すためのダウンキャストフックを提供する。

use crate::error::AutodiffError;
use crate::eval;
use crate::nn::activation::{
    Elu, Gelu, GeluTanh, Hardswish, LeakyRelu, LogSoftmax, Relu, Sigmoid, Silu, Softmax, Softplus,
    Tanh,
};
use crate::nn::batch_norm::{
    BATCH_NORM_1D_RANKS, BATCH_NORM_2D_RANKS, BatchNorm1d, BatchNorm2d, BatchNormCore,
};
use crate::nn::linear::Linear;
use crate::nn::norm::{LayerNorm, RmsNorm};
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, ShapeError, Tensor, batch_norm_layout, broadcast_shape,
    gemm_out_shape, reduce_out_shape, require_same_shape, row_norm_layout,
};

/// [`Module::named_parameters`] の実装が、子 `Module`（`Linear` 等）を
/// 内包する複合層（`MultiheadAttention`・`Rnn`／`Lstm`／`Gru`）で名前へ
/// 接頭辞（`"q_proj."` 等）を連結するための共通ヘルパー（イシュー
/// #1758）。重複実装を避けるため `module.rs` 側に置き、`attention.rs`・
/// `rnn.rs` から使う。
pub(crate) fn prefixed<'a>(
    prefix: &str,
    inner: Vec<(String, &'a Tensor<f32>)>,
) -> Vec<(String, &'a Tensor<f32>)> {
    inner
        .into_iter()
        .map(|(name, tensor)| (format!("{prefix}.{name}"), tensor))
        .collect()
}

/// `nn` の部品（層・活性化関数）に共通の forward シグネチャ。
pub trait Module {
    /// このステップの `tape` 上で 1 回分の forward を計算する。
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError>;

    /// `tape`（葉ノード登録・演算記録）を経由せず、`ops` を直接呼んで
    /// 1 回分の forward をホスト常駐 `Tensor` で計算する（イシュー
    /// #1028・`docs/inference-forward-fixed-cost-design.md` §3.1「段階
    /// A」）。推論専用の tape 不要経路（`compat::Sequential::predict`
    /// が呼ぶ）が、[`Self::forward`] と同じ演算列・同じ丸め（bit
    /// 完全一致）を保ちつつ、`Tape::var` の葉クローン（`Linear` の
    /// `weight`/`bias` を毎呼び出しで clone する固定費。`nn/linear.rs`
    /// の `Linear::bind` 参照）とノード記録のアロケーションを回避する
    /// ために追加した。
    ///
    /// # デフォルト実装
    ///
    /// `fandhe-ai-autodiff` は crates.io 公開クレートであり
    /// （`docs/crates-io-naming-decision.md`）、本メソッドは非破壊拡張
    /// （デフォルトメソッド追加。外部実装者の既存 `impl Module` を壊さ
    /// ない）とする。既定は [`BackendError::Unsupported`] を返す
    /// fail-safe（本クレート内 15 実装〈`Linear`・`Relu`・`Sigmoid`・
    /// `Tanh`・`RmsNorm`・`LayerNorm`・`Softmax`・`LogSoftmax`・`Gelu`・
    /// `GeluTanh`・`Softplus`・`Silu`・`Hardswish`・`LeakyRelu`・`Elu`
    /// （イシュー #1714）〉はいずれも
    /// このデフォルトを
    /// オーバーライドする。呼び出し元
    /// が独自の `Module` 実装をこの経路で使う場合、`Unsupported` を
    /// フォールバックの合図として扱うこと）。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        _input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Err(AutodiffError::Backend(BackendError::Unsupported(
            "Module::forward_host: default fail-safe (tape-free forward not implemented for \
             this Module)"
                .into(),
        )))
    }

    /// 学習可能パラメータを持つ層（現状 `Linear` のみ）への読み取り
    /// アクセスフック。既定実装は `None`（活性化関数など無状態の層は
    /// オーバーライドしない）。`std::any::Any` による動的ダウンキャスト
    /// ではなくこの明示的フックを選ぶ理由: `compat` 層が対象とするレイヤー
    /// 集合は `docs/compat-api-scope.md` §1 で 3 種（Linear・
    /// ReLU/Sigmoid/Tanh）に閉じており、種類を増やすたびに `Any` の
    /// ダウンキャスト先を推測する曖昧さを避け、対応する層がここに
    /// 列挙されているかどうかで閉集合であることをコードとして保つため
    /// （#294。呼び出し元は `compat::Sequential::bind`/
    /// `trainable_parameters`/`apply_parameters`）。
    fn as_linear(&self) -> Option<&Linear> {
        None
    }

    /// [`Module::as_linear`] の可変版。`compat::Sequential::apply_parameters`
    /// が optimizer 更新後の `Tensor<f32>` を層へ書き戻す入口として使う。
    fn as_linear_mut(&mut self) -> Option<&mut Linear> {
        None
    }

    /// この層が `ReLU` かどうか（イシュー #1044・`docs/kernel-fusion.md`
    /// §2.2「学習経路への結線」）。`as_linear` と同じ明示列挙方式
    /// （`docs/compat-api-scope.md` §1 の閉集合維持。`Any` ダウンキャスト
    /// は使わない）で、`fandhe_ai_facade::compat::sequential::Sequential`
    /// が forward 中に「次層が `ReLU` か」を先読みし、`Linear` 層を
    /// `LinearVars::forward_with_activation(input, Activation::Relu)`
    /// （1 ノード・1 カーネル起動）へ結線して `ReLU` 層自体をスキップ
    /// するかどうかを判定する。既定は `false`（`Linear`／
    /// `Sigmoid`／`Tanh` はオーバーライドしない。`Sigmoid`／`Tanh` は
    /// `BackendOps::gemm_bias_act` の `Activation` に対応する variant を
    /// 持たないため融合対象外）。
    fn as_relu(&self) -> bool {
        false
    }

    /// `train()`／`eval()`（PyTorch `Module.training` 相当。イシュー
    /// #1758・`docs/spec/04-requirements.md` REQ-9 2026-09-12 追記
    /// Tier 1「Module の train／eval」）。**既定は no-op**。
    ///
    /// # 契約（無状態モジュールはモードを保持しない）
    ///
    /// 本クレート内実装（`Linear`・活性化関数群・`RmsNorm`／
    /// `LayerNorm`・`Softmax`／`LogSoftmax`・`MultiheadAttention`・
    /// `Rnn`／`Lstm`／`Gru`）はいずれも train／eval で挙動が変わらない
    /// ため、このデフォルト（no-op）のままオーバーライドしない。
    /// **モードの正はコンテナ**（`fandhe_ai_facade::compat::sequential::
    /// Sequential`・`crate::nn::container::ModuleList`／`Sequential`
    /// 〈イシュー #1759 で実装済み〉）**が保持するフラグ**である。今後
    /// Dropout（#1603）・
    /// BatchNorm（#1608 配下）等のモード依存層を追加する際は、本
    /// メソッドと [`Module::training`] の両方を必ずオーバーライドし、
    /// 自層のフィールド（例: `Cell<bool>`）へ実際に保持すること
    /// （既定のまま放置すると、コンテナ側の `set_training` 呼び出しが
    /// 当該層へ伝播しても無視されてしまう）。
    fn set_training(&mut self, _training: bool) {}

    /// 現在のモード。**既定 `true`**（PyTorch `Module.training` の
    /// 初期値と揃える）。[`Module::set_training`] と同じ契約
    /// （無状態モジュールは保持しない・モード依存層は必ずオーバーライド
    /// する）に従う。
    fn training(&self) -> bool {
        true
    }

    /// この層（および子を持つ場合は子を含む）が公開する学習可能
    /// パラメータの「名前, 参照」列（PyTorch `Module.named_parameters()`
    /// 相当。イシュー #1758）。
    ///
    /// # 命名契約
    ///
    /// 名前の正は PyTorch の packed 命名（`in_proj_weight`・
    /// `weight_ih_l0` 等）ではなく、**本クレートの struct フィールド名／
    /// accessor 名**とする（`docs/compat-api-scope.md` §1.2 該当行・
    /// イシュー #1616〈state_dict〉が直列化する compat 契約の一部と
    /// なるため、実装ごとに一貫させる）。列挙順は「登録順＝
    /// weight → bias」（[`fandhe_ai_facade::compat::sequential::
    /// Sequential::trainable_parameters`] と共通の順序契約。同 struct
    /// を参照）。`Option` パラメータは `Some` のときのみ列挙する。
    /// 既定は空 `Vec`（無状態モジュール向け）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        Vec::new()
    }
}

/// `Linear::bind(tape)` で当該ステップの葉ノードを登録してから
/// `LinearVars::forward` を呼ぶ（`nn/linear.rs` 参照）。
impl Module for Linear {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    fn as_linear(&self) -> Option<&Linear> {
        Some(self)
    }

    fn as_linear_mut(&mut self) -> Option<&mut Linear> {
        Some(self)
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（常に）→ `bias`（`Some` の場合のみ）の順。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = vec![("weight".to_string(), self.weight())];
        if let Some(bias) = self.bias() {
            out.push(("bias".to_string(), bias));
        }
        out
    }

    /// [`Module::forward`]（`Linear::bind(tape).forward(input)`。
    /// `LinearVars::forward` が `input.matmul(&weight)` → `.add(&bias)`
    /// と非融合合成する）と **同一の演算列**（`ops.gemm` → `ops.add`）を
    /// 直接呼ぶ。融合カーネル（`ops.gemm_bias_act`）を使わない理由は
    /// bit-exactness 契約（`Module::forward_host` doc 参照）: 融合
    /// epilogue はカーネル内 tiling 次第で加算順序が変わりうるため、
    /// 旧経路と厳密に同じ累積順序を保証できるのは非融合合成のみ。
    /// この bit-exactness 契約は本 trait メソッド（汎用 `&dyn BackendOps`
    /// 向け）に対するもので、CUDA／Metal の融合オーバーライドは非融合
    /// 合成との bit 一致が未保証のため今後もここを融合へ切り替えない。
    ///
    /// **CPU 固定経路の例外（イシュー #1218・`docs/perf/
    /// cpu-infer-predict-profile.md`）**: CPU 融合カーネル
    /// （CPU バックエンドクレートの `CpuBackendOps::gemm_bias_act`）は epilogue
    /// を GEMM 完了後に適用するため非融合合成と bit 完全一致することが
    /// `crates/backend-cpu/tests/gemm_epilogue_parity.rs` で確認済み。
    /// `fandhe_ai_facade::compat::sequential::Sequential::predict` の
    /// CPU 固定 tape 不要経路はこの事実を根拠に `Linear::
    /// forward_host_with_activation`（`nn/linear.rs`。本メソッドとは別の
    /// inherent メソッド）を Linear→ReLU 限定で使う。本 trait メソッド
    /// 自体は汎用バックエンド向けのまま変更しない。
    ///
    /// **エラー型の一致契約（review 指摘）**: `Var::matmul`/`add`（tape
    /// 経路。`var.rs`）は shape 不整合を `gemm_out_shape`/
    /// `broadcast_shape` で `ops.gemm`/`ops.add` 呼び出し**前**に検査し
    /// `AutodiffError::Shape` として返す。本メソッド（tape 不要経路）が
    /// この事前検査を省いて `ops.gemm`/`ops.add` の `?` に任せると、同じ
    /// shape 不整合が `BackendError::ShapeMismatch` 経由の
    /// `AutodiffError::Backend` として返り、`compat::Sequential::predict`
    /// のフォールバック判定対象外の経路で `AutodiffError` の variant が
    /// 呼び出し元から見て変わってしまう（旧経路と新経路で同じ入力が
    /// 異なるエラー variant を返す）。それを避けるため、tape 経路と
    /// 同じ関数で同じ順序に事前検査してから `ops` を呼ぶ。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        gemm_out_shape(input.shape(), self.weight().shape())?;
        let y = ops.gemm(input, self.weight())?;
        match self.bias() {
            Some(bias) => {
                broadcast_shape(y.shape(), bias.shape())?;
                Ok(ops.add(&y, bias)?)
            }
            None => Ok(y),
        }
    }
}

/// `Relu::forward`（shape 不変の単項演算のため構造的に失敗しえない）を
/// `Result` へ包むだけの委譲。`tape` は使わない（`nn/activation.rs`
/// 参照）。
impl Module for Relu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Relu::forward(self, input))
    }

    fn as_relu(&self) -> bool {
        true
    }

    /// `Var::relu()`（`nn::activation::Relu::forward`）が呼ぶ
    /// `tape.ops().relu(...)` と同一のディスパッチ（`ops.relu`）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(ops.relu(input)?)
    }
}

/// `Sigmoid::forward` への委譲。`Relu` の実装と同じ理由で `tape` は
/// 使わない。
impl Module for Sigmoid {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Sigmoid::forward(self, input))
    }

    /// `Var::sigmoid()` は `BackendOps` ディスパッチを経由せず
    /// `eval::sigmoid`（ホスト直接計算）を呼ぶ（`var.rs` 参照）。
    /// bit-exactness のため同じ `eval::sigmoid` を直接呼ぶ。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(eval::sigmoid(input))
    }
}

/// `Tanh::forward` への委譲。`Relu` の実装と同じ理由で `tape` は
/// 使わない。
impl Module for Tanh {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Ok(Tanh::forward(self, input))
    }

    /// `Var::tanh()` も `Sigmoid` と同様 `eval::tanh`（ホスト直接計算）を
    /// 呼ぶ。bit-exactness のため同じ関数を直接呼ぶ。
    fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        Ok(eval::tanh(input))
    }
}

/// `Gelu::forward` への委譲（イシュー #1713）。`Relu` と異なり `Var::gelu`
/// の eager dispatch 契約が型付きエラーを返しうるため `Softmax` と同じ
/// fallible 契約。
impl Module for Gelu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Gelu::forward(self, input)
    }

    /// `Var::gelu()`（`scalar_unary` 経由）と同一のディスパッチ規律
    /// （`grad::scalar_unary_with_fallback`: バックエンド実装 →
    /// `Unsupported` のときのみホスト参照実装へフォールバック・戻り値
    /// shape 検証）を tape 不要経路で再現する。`Var::scalar_unary` が
    /// 呼ぶ同一関数をそのまま呼ぶため bit-exactness・判定迂回なしが
    /// 機構的に保証される。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(
            ops,
            fandhe_ai_tensor_core::ScalarUnaryOp::Gelu,
            input,
        )
    }
}

/// `GeluTanh::forward` への委譲（イシュー #1713）。[`Gelu`] と同じ
/// fallible 契約・`forward_host` ディスパッチ規律。
impl Module for GeluTanh {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        GeluTanh::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(
            ops,
            fandhe_ai_tensor_core::ScalarUnaryOp::GeluTanh,
            input,
        )
    }
}

/// `Softplus::forward` への委譲（イシュー #1713）。`beta`／`threshold`
/// の検査は [`Softplus::new`] が構築時に済ませているため、`forward`／
/// `forward_host` 自体は形状不整合以外では失敗しない。
impl Module for Softplus {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Softplus::forward(self, input)
    }

    /// [`Gelu::forward_host`] と同じディスパッチ規律。`self.beta()`／
    /// `self.threshold()`（クレート内アクセサ。`nn/activation.rs`
    /// 参照）で構築済みの検査済み値を読み出す。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(
            ops,
            fandhe_ai_tensor_core::ScalarUnaryOp::Softplus {
                beta: self.beta(),
                threshold: self.threshold(),
            },
            input,
        )
    }
}

/// `Silu::forward` への委譲（イシュー #1714）。`Softmax` と異なり
/// `forward` 自体は shape 検査で失敗しないが、`Var::scalar_unary` の
/// eager 実体化契約（バックエンド dispatch が型付きエラーを返しうる）
/// により戻り値は `Result`（`Softmax`／`RmsNorm` と同じく `?` ではなく
/// そのまま返す）。
impl Module for Silu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Silu::forward(self, input)
    }

    /// `Var::silu()`（`var.rs`）と同じディスパッチ（`grad::
    /// scalar_unary_with_fallback`）を `tape` 不要経路で再現する
    /// （`Var::scalar_unary` が呼ぶのと同一関数のため bit-exactness が
    /// 構造的に成立する。`.claude/rules/security.md` A08「判定迂回経路
    /// を作らない」規律）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(ops, self.op(), input)
    }
}

/// `Hardswish::forward` への委譲（イシュー #1714）。[`Silu`] と同じ
/// `forward_host` ディスパッチ規律。
impl Module for Hardswish {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Hardswish::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(ops, self.op(), input)
    }
}

/// `LeakyRelu::forward` への委譲（イシュー #1714）。[`Silu`] と同じ
/// `forward_host` ディスパッチ規律。
impl Module for LeakyRelu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        LeakyRelu::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(ops, self.op(), input)
    }
}

/// `Elu::forward` への委譲（イシュー #1714）。[`Silu`] と同じ
/// `forward_host` ディスパッチ規律。
impl Module for Elu {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Elu::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        crate::grad::scalar_unary_with_fallback(ops, self.op(), input)
    }
}

/// `RmsNorm::bind(tape).forward(input)` への委譲（イシュー #1596）。
/// `Linear` と異なり `forward` 自体は fallible（`eps` 検査・`row_norm_
/// layout` の shape 検査で失敗しうる）ため、戻り値をそのまま返す。
impl Module for RmsNorm {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（`Some` の場合のみ。affine なし構成は空）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.weight()
            .map(|w| vec![("weight".to_string(), w)])
            .unwrap_or_default()
    }

    /// `Var::rms_norm`（`var.rs`）と同じディスパッチ規律を `tape` 不要
    /// 経路（`ops` を直接受け取る）で再現する: `row_norm_layout` で
    /// `hidden` を導出し `weight` の shape を検査してから `ops.rmsnorm`
    /// を試み、`Unsupported` のときのみ `eval::rmsnorm_rows` へ
    /// フォールバックする（`Var::rms_norm` と同じ判定迂回を作らない
    /// 規律。`.claude/rules/security.md` A08）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (rows, hidden) = row_norm_layout(input.shape())?;
        if let Some(w) = self.weight() {
            require_same_shape(w.shape(), &[hidden])?;
        }
        let value = match ops.rmsnorm(input, self.weight(), self.eps()) {
            Ok(v) => v,
            Err(BackendError::Unsupported(_)) => {
                let w_dense = self.weight().map(eval::dense_vec);
                eval::rmsnorm_rows(input, w_dense.as_deref(), self.eps(), rows, hidden)
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        // `Var::rms_norm`（`var.rs`）と同じバックエンド契約検証
        // （`BackendOps::rmsnorm` doc「戻り値の shape は入力 `x` と
        // 恒等」）を tape 不要経路でも行う。省略すると `forward_host`
        // 経由の推論のみ不整合 shape を素通りさせてしまい、`Var::
        // rms_norm` と `forward_host` とで判定基準が食い違う
        // （`.claude/rules/security.md` A08 の判定迂回経路になる）。
        if value.shape() != input.shape() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: input.shape().to_vec(),
                },
            )));
        }
        Ok(value)
    }
}

/// `LayerNorm::bind(tape).forward(input)` への委譲（イシュー #1596）。
/// `RmsNorm` と同じ fallible 契約・`forward_host` ディスパッチ規律。
impl Module for LayerNorm {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（`Some` の場合）→ `bias`（`Some` の場合）の順。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        let mut out = Vec::new();
        if let Some(w) = self.weight() {
            out.push(("weight".to_string(), w));
        }
        if let Some(b) = self.bias() {
            out.push(("bias".to_string(), b));
        }
        out
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let (rows, hidden) = row_norm_layout(input.shape())?;
        if let Some(w) = self.weight() {
            require_same_shape(w.shape(), &[hidden])?;
        }
        if let Some(b) = self.bias() {
            require_same_shape(b.shape(), &[hidden])?;
        }
        let value = match ops.layer_norm(input, self.weight(), self.bias(), self.eps()) {
            Ok(v) => v,
            Err(BackendError::Unsupported(_)) => {
                let w_dense = self.weight().map(eval::dense_vec);
                let b_dense = self.bias().map(eval::dense_vec);
                eval::layer_norm_rows(
                    input,
                    w_dense.as_deref(),
                    b_dense.as_deref(),
                    self.eps(),
                    rows,
                    hidden,
                )
            }
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        // `RmsNorm::forward_host`（直上）と同じバックエンド契約検証。
        if value.shape() != input.shape() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: input.shape().to_vec(),
                },
            )));
        }
        Ok(value)
    }
}

/// `BatchNorm1d`／`BatchNorm2d::forward_host`（両者共通）が呼ぶ
/// tape 不要経路の forward 本体（イシュー #1732・親 #1608）。`nn::
/// batch_norm::BatchNormVars::forward`（tape 経路）と同じ判定規律
/// （バックエンド → `Unsupported` のときのみホスト参照実装）を
/// `ops` 直接呼び出しで再現する（`RmsNorm::forward_host` と同じ
/// 理由）。train モードは呼ぶたび必ず running stats を更新する
/// （`BatchNormVars::forward` と同じ契約。`nn::batch_norm` モジュール
/// doc comment「running stats 更新契約」参照）。
fn batch_norm_forward_host(
    core: &BatchNormCore,
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    let (n, c, spatial) = batch_norm_layout(input.shape())?;
    if let Some(w) = core.weight() {
        require_same_shape(w.shape(), &[c])?;
    }
    if let Some(b) = core.bias() {
        require_same_shape(b.shape(), &[c])?;
    }
    if core.training() {
        let m = n
            .checked_mul(spatial)
            .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
        if m <= 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "BatchNorm::forward_host: train モードはチャネルごとの要素数 \
                 M=n*spatial が 1 以下を許容しない（got n={n}, spatial={spatial}, M={m}）"
            )));
        }
        let out = crate::grad::batch_norm_train_with_fallback(
            ops,
            input,
            core.weight(),
            core.bias(),
            core.eps(),
            n,
            c,
            spatial,
        )?;
        core.update_running_stats(&out.batch_mean, &out.batch_var, m);
        Ok(out.output)
    } else {
        let running_mean = core.running_mean_ref();
        let running_var = core.running_var_ref();
        crate::grad::batch_norm_infer_with_fallback(
            ops,
            input,
            &running_mean,
            &running_var,
            core.weight(),
            core.bias(),
            core.eps(),
            n,
            c,
            spatial,
        )
    }
}

/// `BatchNorm1d::bind(tape).forward(input)` への委譲（イシュー
/// #1732・親 #1608）。本クレート内で初めて train／eval でモードにより
/// 挙動が変わる層のため `set_training`／`training` を明示
/// オーバーライドする（`Module::set_training` trait doc の「今後
/// BatchNorm 等を追加する際は必ずオーバーライドすること」を実装する。
/// `nn::batch_norm` モジュール doc comment 参照）。
impl Module for BatchNorm1d {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    fn set_training(&mut self, training: bool) {
        self.core.set_training(training);
    }

    fn training(&self) -> bool {
        self.core.training()
    }

    /// 命名契約（`Module::named_parameters` doc §「命名契約」）:
    /// `weight`（`Some` の場合）→ `bias`（`Some` の場合）の順。running
    /// stats は buffer であり学習可能パラメータではないため含めない
    /// （`nn::batch_norm` モジュール doc comment「`BatchNormCore` の
    /// 可視性」節参照）。
    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.core.named_parameters()
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let rank = input.shape().len();
        if !BATCH_NORM_1D_RANKS.contains(&rank) {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: BATCH_NORM_1D_RANKS[0],
                actual: rank,
            }));
        }
        batch_norm_forward_host(&self.core, ops, input)
    }
}

/// `BatchNorm2d::bind(tape).forward(input)` への委譲（イシュー
/// #1732・親 #1608）。[`Module for BatchNorm1d`](trait.Module.html)
/// と同じ理由・同じ構造だが rank 限定契約のみ異なる（rank 4 のみ）。
impl Module for BatchNorm2d {
    fn forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        self.bind(tape).forward(input)
    }

    fn set_training(&mut self, training: bool) {
        self.core.set_training(training);
    }

    fn training(&self) -> bool {
        self.core.training()
    }

    fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
        self.core.named_parameters()
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let rank = input.shape().len();
        if !BATCH_NORM_2D_RANKS.contains(&rank) {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: BATCH_NORM_2D_RANKS[0],
                actual: rank,
            }));
        }
        batch_norm_forward_host(&self.core, ops, input)
    }
}

/// `Softmax::forward` への委譲（イシュー #1594）。`Relu`/`Sigmoid`/
/// `Tanh` と異なり `forward` 自体が `dim` の軸範囲検査により失敗しうる
/// ため（fallible）、`?` で伝播するだけの `Relu` と違い戻り値をそのまま
/// 返す。
impl Module for Softmax {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        Softmax::forward(self, input)
    }

    /// `Var::softmax()`（`var.rs`）と同じディスパッチ規律を `tape` 不要
    /// 経路（`ops` を直接受け取る）で再現する: `dim` を [`reduce_out_shape`]
    /// で事前検査してから `ops.softmax` を試み、`Unsupported` のときのみ
    /// `eval::softmax_along` へフォールバックする（`Var::softmax` と
    /// 同じ判定迂回を作らない規律。`.claude/rules/security.md` A08）。
    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let dim = self.dim();
        reduce_out_shape(input.shape(), Some(dim))?;
        let value = match ops.softmax(input, dim) {
            Ok(v) => v,
            Err(BackendError::Unsupported(_)) => eval::softmax_along(input, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        // `Var::softmax`（`var.rs`）と同じバックエンド契約検証
        // （`BackendOps::softmax` doc「戻り値 shape は入力と恒等」）を
        // tape 不要経路でも行う。ここを省略すると `forward_host`
        // 経由の推論のみ不整合 shape を素通りさせてしまい、`Var::
        // softmax` と `forward_host` とで判定基準が食い違う
        // （`.claude/rules/security.md` A08 の判定迂回経路になる）。
        if value.shape() != input.shape() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: input.shape().to_vec(),
                },
            )));
        }
        Ok(value)
    }
}

/// `LogSoftmax::forward` への委譲（イシュー #1594）。`Softmax` と同じ
/// fallible 契約・`forward_host` ディスパッチ規律。
impl Module for LogSoftmax {
    fn forward<'t>(&self, _tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        LogSoftmax::forward(self, input)
    }

    fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let dim = self.dim();
        reduce_out_shape(input.shape(), Some(dim))?;
        let value = match ops.log_softmax(input, dim) {
            Ok(v) => v,
            Err(BackendError::Unsupported(_)) => eval::log_softmax_along(input, dim),
            Err(other) => return Err(AutodiffError::Backend(other)),
        };
        // `Softmax::forward_host`（直上）と同じバックエンド契約検証。
        if value.shape() != input.shape() {
            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                    lhs: value.shape().to_vec(),
                    rhs: input.shape().to_vec(),
                },
            )));
        }
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    //! `Module::forward` が既存の直接呼び出し（`Linear::bind().forward()`・
    //! `Relu::forward()` 等）と同一の値・テープ記録を返すことを検証する
    //! （「薄いラッパー性」の担保）。

    use super::*;
    use crate::eval::dense_vec;
    use fandhe_ai_tensor_core::Tensor;

    #[test]
    fn linear_module_forward_matches_bind_forward() {
        let linear = Linear::new(3, 2, true, 42).expect("seed=42 は有効な構築引数");
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![1.0, 2.0, 3.0], &[1, 3]).unwrap());

        let via_module = <Linear as Module>::forward(&linear, &tape, &x).unwrap();
        let via_direct = linear.bind(&tape).forward(&x).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn relu_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Relu as Module>::forward(&Relu, &tape, &x).unwrap();
        let via_direct = Relu.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn sigmoid_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Sigmoid as Module>::forward(&Sigmoid, &tape, &x).unwrap();
        let via_direct = Sigmoid.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn tanh_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let via_module = <Tanh as Module>::forward(&Tanh, &tape, &x).unwrap();
        let via_direct = Tanh.forward(&x);

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn softmax_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap());
        let softmax = Softmax::new(1);

        let via_module = <Softmax as Module>::forward(&softmax, &tape, &x).unwrap();
        let via_direct = softmax.forward(&x).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    #[test]
    fn log_softmax_module_forward_matches_direct_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap());
        let log_softmax = LogSoftmax::new(1);

        let via_module = <LogSoftmax as Module>::forward(&log_softmax, &tape, &x).unwrap();
        let via_direct = log_softmax.forward(&x).unwrap();

        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_direct.to_tensor())
        );
    }

    /// `Module::forward_host`（tape 不要経路）と `Module::forward`
    /// （tape 経路）が同一値を返すことを確認する（`Linear` 等の既存
    /// 契約と同じ bit-exactness 期待。イシュー #1594）。
    #[test]
    fn softmax_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap();
        let softmax = Softmax::new(1);

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <Softmax as Module>::forward(&softmax, &tape, &xv).unwrap();

        let via_host = softmax.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn log_softmax_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap();
        let log_softmax = LogSoftmax::new(1);

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <LogSoftmax as Module>::forward(&log_softmax, &tape, &xv).unwrap();

        let via_host = log_softmax.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn softmax_forward_host_rejects_axis_out_of_range() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0], &[2]).unwrap();
        let softmax = Softmax::new(5);

        let result = softmax.forward_host(test_ops().as_ref(), &x);

        assert!(matches!(
            result,
            Err(AutodiffError::Shape(
                fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { axis: 5, rank: 1 }
            ))
        ));
    }

    /// Cursor Bugbot 指摘（PR #1664）の回帰検証用モック: `softmax`／
    /// `log_softmax` が入力と異なる shape を返す不正なバックエンドを
    /// 模す（他のメソッドは非到達のため `unreachable!` でよい）。
    /// `Var::softmax`（`var.rs`）はこの契約違反を `ShapeMismatch` で
    /// 拒否するが、`Module::forward_host`（tape 不要推論経路）が同じ
    /// 検証を省略していると不整合 shape を素通りさせてしまう
    /// （`.claude/rules/security.md` A08 の判定迂回経路になる）。
    struct WrongShapeOps;

    impl BackendOps for WrongShapeOps {
        fn device(&self) -> fandhe_ai_tensor_core::Device {
            fandhe_ai_tensor_core::Device::Cpu
        }
        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは gemm は呼ばれない")
        }
        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは add は呼ばれない")
        }
        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは mul は呼ばれない")
        }
        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは relu は呼ばれない")
        }
        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは exp は呼ばれない")
        }
        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは tanh は呼ばれない")
        }
        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは sum は呼ばれない")
        }
        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("本テストでは max は呼ばれない")
        }
        fn softmax(&self, x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
            // 入力 shape をそのまま返さず要素数を減らした shape で返す
            // ことで、契約違反（`BackendOps::softmax` doc「戻り値 shape
            // は入力と恒等」）を意図的に起こす。
            let numel: usize = x.shape().iter().product();
            Ok(Tensor::new(vec![0.0f32; numel], &[numel]).unwrap())
        }
        fn log_softmax(&self, x: &Tensor<f32>, _dim: usize) -> Result<Tensor<f32>, BackendError> {
            let numel: usize = x.shape().iter().product();
            Ok(Tensor::new(vec![0.0f32; numel], &[numel]).unwrap())
        }
    }

    #[test]
    fn softmax_forward_host_rejects_wrong_shape_from_backend() {
        let x = Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap();
        let softmax = Softmax::new(1);

        let result = softmax.forward_host(&WrongShapeOps, &x);

        assert!(matches!(
            result,
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
    }

    #[test]
    fn log_softmax_forward_host_rejects_wrong_shape_from_backend() {
        let x = Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap();
        let log_softmax = LogSoftmax::new(1);

        let result = log_softmax.forward_host(&WrongShapeOps, &x);

        assert!(matches!(
            result,
            Err(AutodiffError::Backend(BackendError::ShapeMismatch(_)))
        ));
    }

    /// `Silu`／`Hardswish`／`LeakyRelu`／`Elu` の `forward_host`（tape
    /// 不要経路）が `Module::forward`（tape 経路）と bit 完全一致する
    /// ことを検証する（イシュー #1714。`softmax_forward_host_matches_
    /// tape_forward` と同型——両経路とも同一の `grad::
    /// scalar_unary_with_fallback` を呼ぶため構造的に bit-exactness が
    /// 成立する）。
    #[test]
    fn silu_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0], &[2]).unwrap();
        let silu = Silu;

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <Silu as Module>::forward(&silu, &tape, &xv).unwrap();
        let via_host = silu.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn hardswish_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-4.0, 4.0], &[2]).unwrap();
        let hardswish = Hardswish;

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <Hardswish as Module>::forward(&hardswish, &tape, &xv).unwrap();
        let via_host = hardswish.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn leaky_relu_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0], &[2]).unwrap();
        let leaky_relu = LeakyRelu::new(0.2);

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <LeakyRelu as Module>::forward(&leaky_relu, &tape, &xv).unwrap();
        let via_host = leaky_relu.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }

    #[test]
    fn elu_forward_host_matches_tape_forward() {
        use crate::test_support::test_ops;

        let x = Tensor::new(vec![-1.0, 2.0], &[2]).unwrap();
        let elu = Elu::new(1.3);

        let tape = Tape::new_with_ops(test_ops());
        let xv = tape.var(&x);
        let via_tape = <Elu as Module>::forward(&elu, &tape, &xv).unwrap();
        let via_host = elu.forward_host(test_ops().as_ref(), &x).unwrap();

        assert_eq!(dense_vec(&via_tape.to_tensor()), dense_vec(&via_host));
    }
}
