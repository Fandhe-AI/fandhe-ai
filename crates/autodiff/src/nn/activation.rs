//! 基本活性化関数群（TASK-9.1b・#92）。
//!
//! `docs/spec/05-tasks.md` TASK-9.1（基本 NN モジュール〈Linear・活性化〉
//! の自作コア上再実装）のうち活性化部分を担当する。各構造体はフィール
//! ドを持たないユニット構造体で、`forward` は対応する `Var`
//! （`crate::var`）の演算メソッドを呼ぶだけの薄いラッパーに徹する
//! （REQ-9「互換 API 層は自作コアの上の薄いラッパーに徹する」）。
//!
//! **想定呼び出し元**: `compat::Sequential`（TASK-9.2・#94/#95）が
//! レイヤーの並びの一要素としてこれらを `forward` 経由で呼ぶ想定。
//! 本イシュー時点では共通 `Module` trait は未定義のため、各構造体は
//! 個別に `forward(&self, input: &Var<'t>) -> Var<'t>` を公開する
//! （trait 統一は Linear・#91 と合わせて #94/#95 側で設計する）。
//!
//! 当初のスコープは ReLU・Sigmoid・Tanh の 3 種に限定していた
//! （CrossEntropy 損失〈#191〉は log-softmax → NLL を個別オペ合成せず
//! 1 個の融合オペ〈`tape::Op::CrossEntropyLoss`〉として実装したため、
//! 独立した Softmax プリミティブは当時追加していなかった。`nn/loss.rs`
//! 冒頭 doc 参照）。イシュー #1594 で既存の行カーネル（`BackendOps::
//! softmax`／`log_softmax`）へ接続する独立した [`Softmax`]／
//! [`LogSoftmax`] を追加した（`CrossEntropyLoss` の内部 log-softmax
//! 〈`eval::softmax_along`〉はこれとは別実装のまま不変）。イシュー
//! #1713 で [`Gelu`]／[`GeluTanh`]／[`Softplus`] を追加した（`Var::gelu`／
//! `gelu_tanh`／`softplus` の薄いラッパー）。イシュー #1714 で
//! [`Silu`]／[`Hardswish`]／[`LeakyRelu`]／[`Elu`]（いずれも
//! `crate::var::Var` の `ScalarUnaryOp` 汎用 dispatch。#1592／#1634 が
//! 敷いた基盤への薄いラッパー）を追加した。イシュー #2146 で
//! [`Mish`]／[`Hardtanh`]／[`Relu6`]／[`PRelu`]／[`Glu`] を追加した
//! （`crate::activation_ops` の自由関数への薄いラッパー。`activation_ops`
//! は facade 非公開の内部専用モジュールのため、本 5 層自体も
//! `nn::activation` 経由でのみ到達可能——facade 公開は承認待ち。
//! `crate::activation_ops` モジュール doc 参照）。さらなる追加活性化は
//! 必要になった時点の後続イシューに委ねる。

use crate::error::AutodiffError;
use crate::tape::Tape;
use crate::var::Var;
use fandhe_ai_tensor_core::{ScalarUnaryOp, ShapeError, Tensor};

/// ReLU（`max(x, 0)`）。`Var::relu` の薄いラッパー。
#[derive(Debug, Default, Clone, Copy)]
pub struct Relu;

impl Relu {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Var<'t> {
        input.relu()
    }
}

/// シグモイド（`1 / (1 + exp(-x))`）。`Var::sigmoid` の薄いラッパー。
#[derive(Debug, Default, Clone, Copy)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Var<'t> {
        input.sigmoid()
    }
}

/// 双曲線正接（`tanh(x)`）。`Var::tanh` の薄いラッパー。
#[derive(Debug, Default, Clone, Copy)]
pub struct Tanh;

impl Tanh {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Var<'t> {
        input.tanh()
    }
}

/// GELU（誤差関数版）。`Var::gelu` の薄いラッパー（イシュー #1713）。
/// `Relu`/`Sigmoid`/`Tanh` と異なり `forward` は `Result` を返す
/// （`Var::gelu` の eager dispatch 契約が型付きエラーを返しうるため。
/// `Softmax`／`LogSoftmax` と同型）。
#[derive(Debug, Default, Clone, Copy)]
pub struct Gelu;

impl Gelu {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.gelu()
    }
}

/// GELU（tanh 近似版）。`Var::gelu_tanh` の薄いラッパー（イシュー
/// #1713）。[`Gelu`] と同じ fallible 契約。
#[derive(Debug, Default, Clone, Copy)]
pub struct GeluTanh;

impl GeluTanh {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.gelu_tanh()
    }
}

/// Softplus。`Var::softplus(beta, threshold)` の薄いラッパー（イシュー
/// #1713）。`Softmax`/`LogSoftmax` と同じ「`dim` を保持するフィールド」
/// の設計を踏襲し、`beta`／`threshold` を保持する。構築時に `Var::
/// softplus` と同じ検査（有限かつ `beta > 0`・`threshold` 有限）を行う
/// ため [`Softplus::new`] 自体が `Result` を返す（層構築の時点で早期に
/// 弾く。`nn/norm.rs::validate_eps` と同じ規律）。
#[derive(Debug, Clone, Copy)]
pub struct Softplus {
    beta: f32,
    threshold: f32,
}

impl Softplus {
    /// PyTorch `nn.Softplus` の既定値（`beta=1.0`・`threshold=20.0`）。
    pub fn new(beta: f32, threshold: f32) -> Result<Self, AutodiffError> {
        if !beta.is_finite() || beta <= 0.0 {
            return Err(AutodiffError::InvalidArgument(format!(
                "Softplus::new: beta must be finite and positive, got {beta}"
            )));
        }
        if !threshold.is_finite() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Softplus::new: threshold must be finite, got {threshold}"
            )));
        }
        Ok(Self { beta, threshold })
    }

    /// `self.beta`／`self.threshold` を用いて `Var::softplus` へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.softplus(self.beta, self.threshold)
    }

    /// `nn/module.rs::Module::forward_host` の `Softplus` 実装が
    /// `beta`／`threshold` を読み出すためのクレート内アクセサ
    /// （[`Softmax::dim`] と同じ理由。フィールド自体は非公開のまま）。
    pub(crate) fn beta(&self) -> f32 {
        self.beta
    }

    pub(crate) fn threshold(&self) -> f32 {
        self.threshold
    }
}

impl Default for Softplus {
    /// PyTorch `nn.Softplus` の既定値（`beta=1.0`・`threshold=20.0`）。
    /// `new` の検査（有限性・符号）を通る既知の定数のため、本番経路
    /// panic 禁止規約（`expect`／`unwrap` を避ける）に従いフィールドを
    /// 直接構築する。
    fn default() -> Self {
        Self {
            beta: 1.0,
            threshold: 20.0,
        }
    }
}

/// SiLU／Swish（`x * sigmoid(x)`）。`Var::silu` の薄いラッパー
/// （イシュー #1714）。`Relu`/`Sigmoid`/`Tanh` と異なり `forward` は
/// `Result` を返す（`Var::scalar_unary` の eager 実体化契約により
/// 構造的に失敗しうるため。`Softmax`／`Sqrt` 系と同型）。
#[derive(Debug, Default, Clone, Copy)]
pub struct Silu;

impl Silu {
    /// `input` に SiLU（`x * sigmoid(x)`）を適用する。`Var::silu` への
    /// 薄い委譲（バックエンド dispatch は同メソッドの契約に従う）。
    /// `input` と同じ shape・dtype の `Var` を返す。CPU／CUDA／Metal
    /// 専用カーネル未到達時はホスト参照実装（`ScalarUnaryOp::apply`）
    /// への eager フォールバックが走りうるため、その実体化に失敗した
    /// 場合に限り `Err` を返す（`Softmax`／`Sqrt` 系と同型の契約）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.silu()
    }

    /// `nn/module.rs::Module::forward_host` が `scalar_unary_with_
    /// fallback` へ渡す kind を読み出すためのクレート内アクセサ
    /// （`Softmax::dim` と同型の理由）。
    pub(crate) fn op(&self) -> ScalarUnaryOp {
        ScalarUnaryOp::Silu
    }
}

/// Hardswish（`x * clamp(x + 3, 0, 6) / 6`）。`Var::hardswish` の薄い
/// ラッパー（イシュー #1714）。
#[derive(Debug, Default, Clone, Copy)]
pub struct Hardswish;

impl Hardswish {
    /// `input` に Hardswish（`x * clamp(x + 3, 0, 6) / 6`）を適用する。
    /// `Var::hardswish` への薄い委譲。[`Silu::forward`] と同じ shape・
    /// エラー契約（eager フォールバック実体化に失敗した場合のみ
    /// `Err`）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.hardswish()
    }

    /// [`Silu::op`] と同じ理由のクレート内アクセサ。
    pub(crate) fn op(&self) -> ScalarUnaryOp {
        ScalarUnaryOp::Hardswish
    }
}

/// Leaky ReLU（`x >= 0` なら `x`、それ以外は `negative_slope * x`）。
/// `Var::leaky_relu` の薄いラッパー（イシュー #1714）。`Softmax`／
/// `LogSoftmax` と同じく `negative_slope` を保持するフィールドを持つ。
#[derive(Debug, Clone, Copy)]
pub struct LeakyRelu {
    negative_slope: f32,
}

impl LeakyRelu {
    /// `negative_slope`（負領域の傾き）を指定して構築する。
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }

    /// `input` に Leaky ReLU（`x >= 0` なら `x`、それ以外は
    /// `self.negative_slope * x`）を適用する。`Var::leaky_relu` への
    /// 薄い委譲。[`Silu::forward`] と同じ shape・エラー契約（eager
    /// フォールバック実体化に失敗した場合のみ `Err`）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.leaky_relu(self.negative_slope)
    }

    /// [`Silu::op`] と同じ理由のクレート内アクセサ。
    pub(crate) fn op(&self) -> ScalarUnaryOp {
        ScalarUnaryOp::LeakyRelu {
            negative_slope: self.negative_slope,
        }
    }
}

impl Default for LeakyRelu {
    /// PyTorch `torch.nn.LeakyReLU` の既定 `negative_slope=0.01` と揃える。
    fn default() -> Self {
        Self::new(0.01)
    }
}

/// ELU（`x > 0` なら `x`、それ以外は `alpha * (exp(x) - 1)`）。
/// `Var::elu` の薄いラッパー（イシュー #1714）。[`LeakyRelu`] と同じ
/// フィールド保持・`Default` 契約。
#[derive(Debug, Clone, Copy)]
pub struct Elu {
    alpha: f32,
}

impl Elu {
    /// `alpha` を指定して構築する。
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// `input` に ELU（`x > 0` なら `x`、それ以外は
    /// `self.alpha * (exp(x) - 1)` 相当）を適用する。`Var::elu` への
    /// 薄い委譲。[`Silu::forward`] と同じ shape・エラー契約（eager
    /// フォールバック実体化に失敗した場合のみ `Err`）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.elu(self.alpha)
    }

    /// [`Silu::op`] と同じ理由のクレート内アクセサ。
    pub(crate) fn op(&self) -> ScalarUnaryOp {
        ScalarUnaryOp::Elu { alpha: self.alpha }
    }
}

impl Default for Elu {
    /// PyTorch `torch.nn.ELU` の既定 `alpha=1.0` と揃える。
    fn default() -> Self {
        Self::new(1.0)
    }
}

/// 行方向 softmax。`Var::softmax(dim)` の薄いラッパー（イシュー
/// #1594）。`Relu`/`Sigmoid`/`Tanh` と異なり `dim` を保持するフィールド
/// を持ち、`forward` は `dim` の軸範囲検査（`Var::softmax` 内部）により
/// `Result` を返す（構造的に失敗しうる）。
#[derive(Debug, Clone, Copy)]
pub struct Softmax {
    dim: usize,
}

impl Softmax {
    /// `dim`（softmax を適用する軸）を指定して構築する。
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// `self.dim` 軸に沿って softmax を適用する（`Var::softmax` へ委譲）。
    /// 戻り値 shape は `input` と同一（正規化は形状を変えない）。
    /// `self.dim >= input.rank()`（rank 0 を含む）は
    /// `AutodiffError::Shape` で fail-closed に拒否する。バックエンド
    /// 側 `Unsupported`（非最終軸）はホスト参照実装へ透過的に
    /// フォールバックし、それ以外のバックエンドエラーはそのまま
    /// 伝播する（`Var::softmax` の契約をそのまま引き継ぐ）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.softmax(self.dim)
    }

    /// `nn/module.rs::Module::forward_host` の `Softmax` 実装が `dim` を
    /// 読み出すためのアクセサ（`dim` フィールド自体はカプセル化のため
    /// 非公開のまま）。イシュー #2076（親 #2034）で `onnx-interop::
    /// onnx::export_nn` が `Module::as_softmax` 経由で取得した
    /// `&Softmax` から `ExportOp::Softmax { axis }` の `axis` を組み立て
    /// るために公開へ変更した（`crate` 外の別クレートから呼ばれるため
    /// `pub(crate)` では届かない）。facade は `nn::activation::Softmax`
    /// を再エクスポートしないため、本変更は facade の公開面を拡張しない。
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// 行方向 log_softmax。`Var::log_softmax(dim)` の薄いラッパー（イシュー
/// #1594）。[`Softmax`] と同じ `dim` 保持・fallible 契約。
#[derive(Debug, Clone, Copy)]
pub struct LogSoftmax {
    dim: usize,
}

impl LogSoftmax {
    /// `dim`（log_softmax を適用する軸）を指定して構築する。
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// `self.dim` 軸に沿って log_softmax を適用する
    /// （`Var::log_softmax` へ委譲）。shape 不変・`self.dim` 範囲外の
    /// 拒否（`AutodiffError::Shape`）は [`Softmax::forward`] と同一。
    /// バックエンドが対応しない軸・演算（CUDA／Metal は現時点で
    /// log_softmax を最終軸含め未実装のため常にホストへフォール
    /// バックする）は `Unsupported` のときのみ透過的にホスト参照実装
    /// へ切り替わり、それ以外のバックエンドエラーは伝播する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.log_softmax(self.dim)
    }

    /// [`Softmax::dim`] と同じ理由のクレート内アクセサ。
    pub(crate) fn dim(&self) -> usize {
        self.dim
    }
}

/// Mish（`x * tanh(softplus(x))`）。`crate::activation_ops::mish` への
/// 薄いラッパー（イシュー #2146）。`Relu`/`Sigmoid`/`Tanh` と異なり
/// `forward` は `Result` を返す（`softplus`／`tanh`／`mul` の合成が
/// eager 実体化契約を持つため。[`Silu`] と同型の契約）。
#[derive(Debug, Default, Clone, Copy)]
pub struct Mish;

impl Mish {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        crate::activation_ops::mish(input)
    }
}

/// Hardtanh（`min < x < max` の開区間では `x` を素通し、それ以外は
/// `clamp(x, min, max)`）。`crate::activation_ops::hardtanh` への薄い
/// ラッパー（イシュー #2146）。数値・境界勾配の契約は
/// `crate::activation_ops::hardtanh` doc を参照。
#[derive(Debug, Clone, Copy)]
pub struct Hardtanh {
    min_val: f32,
    max_val: f32,
}

impl Hardtanh {
    /// `min_val`／`max_val` を指定して構築する。`crate::activation_ops::
    /// hardtanh` と同じ検査（有限〈`NaN` 拒否〉・`min_val < max_val`）を
    /// 構築時に行う（`nn/norm.rs::validate_eps` と同じ「層構築の時点で
    /// 早期に弾く」規律）。
    pub fn new(min_val: f32, max_val: f32) -> Result<Self, AutodiffError> {
        if min_val.is_nan() || max_val.is_nan() {
            return Err(AutodiffError::InvalidArgument(format!(
                "Hardtanh::new: min_val/max_val must not be NaN, got min_val={min_val}, \
                 max_val={max_val}"
            )));
        }
        if min_val >= max_val {
            return Err(AutodiffError::InvalidArgument(format!(
                "Hardtanh::new: min_val must be less than max_val, got min_val={min_val}, \
                 max_val={max_val}"
            )));
        }
        Ok(Self { min_val, max_val })
    }

    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        crate::activation_ops::hardtanh(input, self.min_val, self.max_val)
    }

    /// `nn/module.rs::Module::forward_host` が `scalar_unary_with_
    /// fallback` へ渡す `ScalarUnaryOp::Clamp { min, max }` を組み立てる
    /// ためのクレート内アクセサ（`Silu::op` と同型の理由）。
    /// `crate::activation_ops::hardtanh` の forward が `clamp` と bit
    /// 完全一致するため（同関数 doc 参照）、`forward_host` はこの
    /// `Clamp` 経路をそのまま使ってよい。
    pub(crate) fn op(&self) -> ScalarUnaryOp {
        ScalarUnaryOp::Clamp {
            min: self.min_val,
            max: self.max_val,
        }
    }
}

impl Default for Hardtanh {
    /// PyTorch `nn.Hardtanh` の既定値（`min_val=-1.0`・`max_val=1.0`）。
    /// `new` の検査を通る既知の定数のため、本番経路 panic 禁止規約に
    /// 従いフィールドを直接構築する（[`Softplus::default`] と同型）。
    fn default() -> Self {
        Self {
            min_val: -1.0,
            max_val: 1.0,
        }
    }
}

/// ReLU6（`hardtanh(x, 0.0, 6.0)`）。`crate::activation_ops::relu6` への
/// 薄いラッパー（イシュー #2146）。数値・境界勾配の契約は
/// [`Hardtanh`]／`crate::activation_ops::hardtanh` doc を参照。
#[derive(Debug, Default, Clone, Copy)]
pub struct Relu6;

impl Relu6 {
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        crate::activation_ops::relu6(input)
    }

    /// [`Hardtanh::op`] と同じ理由のクレート内アクセサ（`min=0.0`・
    /// `max=6.0` 固定）。
    pub(crate) fn op(&self) -> ScalarUnaryOp {
        ScalarUnaryOp::Clamp { min: 0.0, max: 6.0 }
    }
}

/// GLU（Gated Linear Unit。軸 `dim` に沿って前後半へ分割し
/// `a * sigmoid(b)` を返す）。`crate::activation_ops::glu` への薄い
/// ラッパー（イシュー #2146）。PyTorch の既定 `dim=-1` と異なり、本
/// クレートの慣例に従い負の軸番号は受け付けない（呼び出し側が明示的な
/// 軸番号を指定する。`crate::activation_ops` モジュール doc「PyTorch
/// との既知の差分」参照）。
#[derive(Debug, Clone, Copy)]
pub struct Glu {
    dim: usize,
}

impl Glu {
    /// `dim`（分割・ゲートを適用する軸）を指定して構築する。
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        crate::activation_ops::glu(input, self.dim)
    }
}

/// PReLU（`x > 0 ? x : weight * x`。チャネルごとの傾きを学習する唯一の
/// 「状態を持つ活性化」。イシュー #2146）。本体（`weight` を永続保持
/// する層パラメータ）→ `bind(&tape)` で `Var` 化した `PReluVars`
/// という分離パターンは `RmsNorm`／`RmsNormVars`（`nn/norm.rs`）と同型
/// （モジュール冒頭 doc・`nn/norm.rs` 冒頭 doc 参照。`Tape` はステップ
/// ごとに生成・破棄される前提のため）。
#[derive(Debug)]
pub struct PRelu {
    weight: Tensor<f32>,
    /// 層別 `requires_grad` 凍結フラグ（`nn::Linear`／`RmsNorm` と同型。
    /// イシュー #2137 の横展開）。既定 `true`。
    requires_grad: bool,
}

impl PRelu {
    /// `num_parameters` 個のチャネルを持つ `weight` を `init`（PyTorch
    /// `nn.PReLU` 既定は `0.25`）で初期化する。`num_parameters == 0` は
    /// `AutodiffError::InvalidArgument` で拒否する（`crate::
    /// activation_ops::prelu` の `C >= 1` 契約と同じ理由。層構築の時点
    /// で早期に弾く）。
    pub fn new(num_parameters: usize, init: f32) -> Result<Self, AutodiffError> {
        if num_parameters == 0 {
            return Err(AutodiffError::InvalidArgument(
                "PRelu::new: num_parameters must be at least 1".into(),
            ));
        }
        let weight = Tensor::new(vec![init; num_parameters], &[num_parameters])
            .map_err(AutodiffError::Shape)?;
        Ok(Self {
            weight,
            requires_grad: true,
        })
    }

    /// 明示的な `weight` から構築する（safetensors ロード等向けの入口。
    /// `RmsNorm::from_parameters` と同じ位置付け）。rank 1・非空を要求
    /// する（A03: 外部由来パラメータを計算前に検証する契約。
    /// `.claude/rules/security.md`）。
    pub fn from_parameters(weight: Tensor<f32>) -> Result<Self, AutodiffError> {
        if weight.rank() != 1 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 1,
                actual: weight.rank(),
            }));
        }
        if weight.shape()[0] == 0 {
            return Err(AutodiffError::InvalidArgument(
                "PRelu::from_parameters: weight must have at least 1 element".into(),
            ));
        }
        Ok(Self {
            weight,
            requires_grad: true,
        })
    }

    /// `weight` パラメータ（shape `[num_parameters]`）。
    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    /// [`crate::nn::module::Module::set_requires_grad`]（`PRelu` 実装。
    /// `module.rs` 参照）の本体。
    pub(crate) fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// [`crate::nn::module::Module::requires_grad`]（`PRelu` 実装）の
    /// 本体。
    pub(crate) fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// [`crate::nn::module::Module::set_parameter`]（`PRelu` 実装。
    /// `module.rs` 参照）の本体。`"weight"` のみ受理し、shape 保存置換
    /// のみを許す（`RmsNorm::set_parameter` と同じ契約。イシュー
    /// #1752）。
    pub(crate) fn set_parameter(
        &mut self,
        name: &str,
        value: Tensor<f32>,
    ) -> Result<(), AutodiffError> {
        match name {
            "weight" => {
                if value.shape() != self.weight.shape() {
                    return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                        lhs: value.shape().to_vec(),
                        rhs: self.weight.shape().to_vec(),
                    }));
                }
                self.weight = value;
                Ok(())
            }
            _ => Err(AutodiffError::InvalidArgument(format!(
                "PRelu::set_parameter: no parameter named `{name}`"
            ))),
        }
    }

    /// このステップの `tape` へ `weight` を葉ノードとして登録し、
    /// `forward` を呼べる `PReluVars` を返す（`RmsNorm::bind` と同じ
    /// 理由）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> PReluVars<'t> {
        let weight = tape.var_with_requires_grad(&self.weight, self.requires_grad);
        PReluVars { weight }
    }
}

/// `PRelu::bind` が返す、1 ステップ分のテープに登録済みパラメータ。
/// `weight` を公開する理由は [`crate::nn::linear::LinearVars`] と同じ
/// （`Tape::backward` 後に `Gradients::get(&vars.weight)` で `dweight`
/// を取得する。呼び出し側の責務）。
pub struct PReluVars<'t> {
    pub weight: Var<'t>,
}

impl<'t> PReluVars<'t> {
    /// `crate::activation_ops::prelu(input, &self.weight)` への委譲。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        crate::activation_ops::prelu(input, &self.weight)
    }
}

#[cfg(test)]
mod tests {
    //! `nn::activation` 各構造体の `forward` が、対応する `Var` メソッド
    //! 直接呼び出しと同一の値・テープ記録を返すことを検証する
    //! （「薄いラッパー性」の担保。イシュー #92 実装計画 §5）。

    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;

    #[test]
    fn relu_forward_matches_var_relu() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Relu.forward(&x);
        let via_var = x.relu();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        // `Tensor` は意図的に `PartialEq` を derive しないため
        // （`tensor-core::Tensor` のドキュメント参照）、稠密化した
        // データ列で値の一致を検証する。
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn sigmoid_forward_matches_var_sigmoid() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Sigmoid.forward(&x);
        let via_var = x.sigmoid();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        // `Tensor` は意図的に `PartialEq` を derive しないため
        // （`tensor-core::Tensor` のドキュメント参照）、稠密化した
        // データ列で値の一致を検証する。
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn tanh_forward_matches_var_tanh() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Tanh.forward(&x);
        let via_var = x.tanh();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        // `Tensor` は意図的に `PartialEq` を derive しないため
        // （`tensor-core::Tensor` のドキュメント参照）、稠密化した
        // データ列で値の一致を検証する。
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn gelu_forward_matches_var_gelu() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Gelu.forward(&x).unwrap();
        let via_var = x.gelu().unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn gelu_tanh_forward_matches_var_gelu_tanh() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = GeluTanh.forward(&x).unwrap();
        let via_var = x.gelu_tanh().unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn softplus_forward_matches_var_softplus() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Softplus::new(1.0, 20.0).unwrap().forward(&x).unwrap();
        let via_var = x.softplus(1.0, 20.0).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn softplus_new_rejects_non_positive_or_non_finite_beta_and_non_finite_threshold() {
        assert!(Softplus::new(0.0, 20.0).is_err());
        assert!(Softplus::new(-1.0, 20.0).is_err());
        assert!(Softplus::new(f32::NAN, 20.0).is_err());
        assert!(Softplus::new(f32::INFINITY, 20.0).is_err());
        assert!(Softplus::new(1.0, f32::NAN).is_err());
        assert!(Softplus::new(1.0, f32::INFINITY).is_err());
        assert!(Softplus::new(1.0, 20.0).is_ok());
    }

    #[test]
    fn softplus_default_matches_pytorch_defaults() {
        let s = Softplus::default();
        assert_eq!(s.beta(), 1.0);
        assert_eq!(s.threshold(), 20.0);
    }

    #[test]
    fn softmax_forward_matches_var_softmax() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap());
        let before = tape.len();

        let via_module = Softmax::new(1).forward(&x).unwrap();
        let via_var = x.softmax(1).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn softmax_forward_rejects_axis_out_of_range() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());

        let result = Softmax::new(5).forward(&x);

        assert!(matches!(
            result,
            Err(crate::error::AutodiffError::Shape(
                fandhe_ai_tensor_core::ShapeError::AxisOutOfRange { axis: 5, rank: 1 }
            ))
        ));
    }

    #[test]
    fn log_softmax_forward_matches_var_log_softmax() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape
            .var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0, 0.5, 1.0], &[2, 2]).unwrap());
        let before = tape.len();

        let via_module = LogSoftmax::new(1).forward(&x).unwrap();
        let via_var = x.log_softmax(1).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn silu_forward_matches_var_silu() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Silu.forward(&x).unwrap();
        let via_var = x.silu().unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn hardswish_forward_matches_var_hardswish() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-4.0, 4.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Hardswish.forward(&x).unwrap();
        let via_var = x.hardswish().unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn leaky_relu_forward_matches_var_leaky_relu() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = LeakyRelu::new(0.2).forward(&x).unwrap();
        let via_var = x.leaky_relu(0.2).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn leaky_relu_default_matches_pytorch_negative_slope() {
        assert_eq!(LeakyRelu::default().negative_slope, 0.01);
    }

    #[test]
    fn elu_forward_matches_var_elu() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![-1.0, 2.0], &[2]).unwrap());
        let before = tape.len();

        let via_module = Elu::new(1.3).forward(&x).unwrap();
        let via_var = x.elu(1.3).unwrap();

        assert_eq!(
            tape.len(),
            before + 2,
            "forward 呼び出しごとに 1 ノード追記"
        );
        assert_eq!(
            dense_vec(&via_module.to_tensor()),
            dense_vec(&via_var.to_tensor())
        );
    }

    #[test]
    fn elu_default_matches_pytorch_alpha() {
        assert_eq!(Elu::default().alpha, 1.0);
    }
}
