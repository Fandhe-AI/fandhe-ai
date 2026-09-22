//! 畳み込み層（Conv1d／Conv2d／ConvTranspose2d。イシュー #1770・
//! #2067・親 #1645）。
//!
//! `Var::conv2d`／`Var::conv1d`（#1764・#1765）・`Var::
//! conv_transpose2d`（#2067。いずれも `var.rs`）・`BackendOps::
//! im2col`／`col2im`／`conv2d`（既定 `Unsupported`。CPU 実装 #1642・
//! CUDA #1643・Metal #1644）を薄くラップし、PyTorch `nn.Conv2d`／
//! `nn.Conv1d`／`nn.ConvTranspose2d` 相当の層として `nn::Linear`
//! （`linear.rs`）と同じ「本体（`Conv2d`）／テープ登録済みハンドル
//! （`Conv2dVars`）」分離設計に従う（`Tape` はステップごとに生成・
//! 破棄される前提。`linear.rs` モジュール doc「`Tape` ライフサイクル
//! との関係」節を参照）。
//!
//! `docs/conv-ops-design.md` §8「`nn` 配線案」（Conv1d／Conv2d）・§15
//! （ConvTranspose2d）の実装であり、`ConvTranspose2d` は新規 `Op`
//! （`Op::ConvTranspose2d`）・VJP を追加した（#2067。既存 `Op::
//! Conv2d` は変更しない）が新規 `BackendOps` メソッドは追加しない
//! （既存 `im2col`／`col2im`／`gemm_batched` の合成。設計 doc §15
//! 「承認事項」節。`BackendOps::conv_transpose2d` override フックは
//! 追加しない）。

use fandhe_ai_tensor_core::{
    BackendOps, Conv2dParams, ShapeError, Tensor, conv_transpose2d_out_shape, conv2d_out_shape,
};

use crate::error::AutodiffError;
use crate::grad::{conv_transpose2d_with_fallback, conv2d_with_fallback};
use crate::nn::init::{BIAS_SEED_SALT, WEIGHT_SEED_SALT, derive_seed, try_uniform_init};
use crate::tape::Tape;
use crate::var::Var;

/// [`try_uniform_init`] の `Err`（`TryReserveError`）を
/// [`AutodiffError::InvalidArgument`] へ変換する重み初期化共通ヘルパー
/// （`nn::rnn::checked_uniform_init` と同型。イシュー #1770
/// codex-review P1 指摘: `checked_mul` による `usize` オーバーフロー
/// 検査だけでは `Vec<f32>` の `isize::MAX` バイト制限を検査できず、
/// `uniform_init` 内の `collect()` が capacity overflow で panic
/// しうる。本番経路 panic 禁止。`.claude/rules/coding-rust.md`）。
/// `field_name` はエラーメッセージにどのパラメータ（`weight`／
/// `bias`）の確保に失敗したかを残すためのラベル。
fn checked_uniform_init(
    len: usize,
    bound: f32,
    seed: u64,
    field_name: &str,
) -> Result<Vec<f32>, AutodiffError> {
    try_uniform_init(len, bound, seed).map_err(|err| {
        AutodiffError::InvalidArgument(format!(
            "{field_name}: len={len} 要素分のバッファを確保できません: {err}"
        ))
    })
}

/// Conv2d 層のパラメータ本体。`weight` は `[out_channels, in_channels /
/// groups, kH, kW]`（PyTorch `nn.Conv2d.weight` と同じレイアウト。
/// `nn::Linear` の `[in, out]` 慣習とは異なる点に注意）、`bias` は
/// `Some` の場合 `[out_channels]`。ハイパーパラメータ（`stride`／
/// `padding`／`dilation`／`groups`）は forward のたびに引数として
/// 渡し直さずに済むよう層側で保持する（`Var::conv2d` 自体は無状態の
/// メソッドであり、これらを引数に取る）。
pub struct Conv2d {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl Conv2d {
    /// 決定的シードで `U(-1/√fan_in, 1/√fan_in)`（`fan_in = (in_channels
    /// / groups) · kH · kW`。PyTorch `nn.Conv2d` 既定初期化と同じ有効
    /// 範囲）の一様初期化を行う（`nn::Linear::new` と同型。`derive_seed`
    /// による weight／bias の独立シード導出も同じ）。
    ///
    /// # 検査順序
    ///
    /// ① `Conv2dParams::new`（`kernel_size`／`stride`／`dilation`／
    /// `groups` の 0・`2·padding` オーバーフローを拒否。`BackendError`
    /// を `AutodiffError::Backend` へ委譲）→ ② `in_channels == 0` を
    /// `AutodiffError::InvalidArgument` で拒否（`bound` が非有限になる
    /// ため。`nn::Linear::new` と同じ理由）→ ③ `in_channels % groups`／
    /// `out_channels % groups`／`out_channels < groups`（forward 時に
    /// `conv2d_out_shape` が拒否する条件を構築時へ前倒しする。A03:
    /// 外部由来ではなく呼び出し引数の検査だが、早期に弾くことで
    /// 「構築は成功したが forward が常に失敗する」層を作らせない）。
    #[allow(clippy::too_many_arguments)] // PyTorch `nn.Conv2d` の全引数を受理する必要があるため（`Var::conv1d` の同種 allow・doc comment 方針を踏襲）。
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Conv2d, AutodiffError> {
        let params = Conv2dParams::new(kernel_size, stride, padding, dilation, groups)
            .map_err(AutodiffError::Backend)?;
        if in_channels == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Conv2d::new: in_channels must be > 0 (1/sqrt(fan_in) would be non-finite)"
                    .to_string(),
            ));
        }
        if !in_channels.is_multiple_of(groups) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Conv2d::new: in_channels ({in_channels}) must be divisible by groups ({groups})"
            )));
        }
        if !out_channels.is_multiple_of(groups) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Conv2d::new: out_channels ({out_channels}) must be divisible by groups \
                 ({groups})"
            )));
        }
        if out_channels < groups {
            return Err(AutodiffError::InvalidArgument(format!(
                "Conv2d::new: out_channels ({out_channels}) must be >= groups ({groups})"
            )));
        }
        let cin_g = in_channels / groups;
        let [kh, kw] = kernel_size;
        let fan_in = cin_g
            .checked_mul(kh)
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "Conv2d::new: fan_in (in_channels/groups * kH * kW) overflows usize"
                        .to_string(),
                )
            })?;
        if fan_in == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Conv2d::new: fan_in must be > 0 (1/sqrt(fan_in) would be non-finite)".to_string(),
            ));
        }
        let bound = 1.0 / (fan_in as f32).sqrt();

        let weight_seed = derive_seed(seed, WEIGHT_SEED_SALT);
        let weight_numel = out_channels
            .checked_mul(cin_g)
            .and_then(|v| v.checked_mul(kh))
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "Conv2d::new: weight element count overflows usize".to_string(),
                )
            })?;
        let weight_data =
            checked_uniform_init(weight_numel, bound, weight_seed, "Conv2d::new: weight")?;
        let weight = Tensor::new(weight_data, &[out_channels, cin_g, kh, kw])?;

        let bias = if bias {
            let bias_seed = derive_seed(seed, BIAS_SEED_SALT);
            let bias_data =
                checked_uniform_init(out_channels, bound, bias_seed, "Conv2d::new: bias")?;
            Some(Tensor::new(bias_data, &[out_channels])?)
        } else {
            None
        };

        Ok(Conv2d {
            weight,
            bias,
            stride: params.stride(),
            padding: params.padding(),
            dilation: params.dilation(),
            groups: params.groups(),
        })
    }

    /// 明示的な重み・バイアス・ハイパーパラメータから構築する
    /// （`state_dict`／`apply_parameters` の書き戻し先・テスト向け入口。
    /// `nn::Linear::from_parameters` と同型）。
    ///
    /// `weight` は rank 4・`weight.shape()[1] == 0`（`in_channels/groups`
    /// が 0）を拒否する（`Linear::from_parameters` が `weight.shape()[0]
    /// == 0` を拒否する理由〈zero-K matmul が全 0 出力を静かに返す〉と
    /// 対称。Conv2d では `weight.shape()[1]` が im2col 後の畳み込み和を
    /// 取る軸に対応する）。`stride`／`padding`／`dilation`／`groups` は
    /// `weight.shape()[2..4]` から導いた `kernel_size` とあわせて
    /// `Conv2dParams::new` で検査する。`groups` に対する `out_channels`／
    /// `in_channels`（`weight.shape()[1] * groups`）の整合検査は
    /// [`Conv2d::new`] と同じ内容を行う（`load_state_dict`／
    /// `apply_parameters` 由来の壊れた重みを入口で弾く。A03）。
    #[allow(clippy::too_many_arguments)] // Conv2d の全ハイパーパラメータを引数として受理する必要があるため。
    pub fn from_parameters(
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Conv2d, AutodiffError> {
        if weight.rank() != 4 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 4,
                actual: weight.rank(),
            }));
        }
        let weight_shape = weight.shape().to_vec();
        let cin_g = weight_shape[1];
        if cin_g == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Conv2d::from_parameters: weight.shape()[1] (in_channels/groups) must be > 0"
                    .to_string(),
            ));
        }
        let kernel_size = [weight_shape[2], weight_shape[3]];
        let params = Conv2dParams::new(kernel_size, stride, padding, dilation, groups)
            .map_err(AutodiffError::Backend)?;

        let out_channels = weight_shape[0];
        let in_channels = cin_g.checked_mul(groups).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "Conv2d::from_parameters: in_channels (weight.shape()[1] * groups) overflows \
                 usize"
                    .to_string(),
            )
        })?;
        if !in_channels.is_multiple_of(groups)
            || !out_channels.is_multiple_of(groups)
            || out_channels < groups
        {
            // `in_channels % groups` はここでは常に 0（`in_channels = cin_g * groups`
            // の構成のため）だが、`Conv2d::new` と同じ検査を明示的に
            // 揃えておく（将来 `cin_g`/`groups` の導出方法が変わった際の
            // 保険。到達しない分岐だが本番経路 panic を避ける fail-closed
            // 方針〈`.claude/rules/coding-rust.md`〉に従いエラーで表現する）。
            return Err(AutodiffError::InvalidArgument(format!(
                "Conv2d::from_parameters: in_channels ({in_channels}) / out_channels \
                 ({out_channels}) not consistent with groups ({groups})"
            )));
        }

        if let Some(ref b) = bias {
            if b.rank() != 1 {
                return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                    expected: 1,
                    actual: b.rank(),
                }));
            }
            if b.shape() != [out_channels] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![out_channels],
                }));
            }
        }

        Ok(Conv2d {
            weight,
            bias,
            stride: params.stride(),
            padding: params.padding(),
            dilation: params.dilation(),
            groups: params.groups(),
        })
    }

    /// このステップの `tape` へ `weight`/`bias` を葉ノードとして登録し、
    /// `forward` を呼べる `Conv2dVars` を返す（`nn::Linear::bind` と
    /// 同型）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> Conv2dVars<'t> {
        let weight = tape.var(&self.weight);
        let bias = self.bias.as_ref().map(|b| tape.var(b));
        Conv2dVars {
            weight,
            bias,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        }
    }

    /// 重み `[out_channels, in_channels / groups, kH, kW]`
    /// （PyTorch `nn.Conv2d.weight` と同じレイアウト）。
    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    /// バイアス `[out_channels]`。[`Conv2d::new`]／
    /// [`Conv2d::from_parameters`] に `bias: false`／`bias: None` を
    /// 渡した場合は `None`。
    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.bias.as_ref()
    }

    /// 空間軸 `[stride_h, stride_w]`（PyTorch `nn.Conv2d` と同じ軸順）。
    pub fn stride(&self) -> [usize; 2] {
        self.stride
    }

    /// 空間軸 `[padding_h, padding_w]`（軸順は [`Conv2d::stride`] と
    /// 同じ）。
    pub fn padding(&self) -> [usize; 2] {
        self.padding
    }

    /// 空間軸 `[dilation_h, dilation_w]`（軸順は [`Conv2d::stride`] と
    /// 同じ）。
    pub fn dilation(&self) -> [usize; 2] {
        self.dilation
    }

    /// グループ数（`groups == 1` が通常の畳み込み、`groups ==
    /// in_channels` が depthwise 畳み込みに相当）。
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// [`crate::nn::module::Module::set_parameter`]（`Conv2d` 実装）の
    /// 本体。`"weight"`（常に）／`"bias"`（`Some` の場合のみ）を受理
    /// する（`nn::Linear::set_parameter` と同型。shape 保存置換のみ）。
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
            "bias" => match &mut self.bias {
                Some(current) => {
                    if value.shape() != current.shape() {
                        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                            lhs: value.shape().to_vec(),
                            rhs: current.shape().to_vec(),
                        }));
                    }
                    *current = value;
                    Ok(())
                }
                None => Err(AutodiffError::InvalidArgument(format!(
                    "Conv2d::set_parameter: no parameter named `{name}` (this layer has no bias)"
                ))),
            },
            _ => Err(AutodiffError::InvalidArgument(format!(
                "Conv2d::set_parameter: no parameter named `{name}`"
            ))),
        }
    }

    /// [`crate::nn::module::Module::forward_host`]（`Conv2d` 実装）の
    /// 本体。`Var::conv2d`（`var.rs`）と**同じ検査順序**
    /// （①rank→②`Conv2dParams::new`→③`conv2d_out_shape`→④bias shape）
    /// で事前検査してから、`Var::conv2d` が呼ぶのと同じ `grad::
    /// conv2d_with_fallback` を直接呼ぶ（`nn::Linear::forward_host` の
    /// エラー型一致契約と同じ理由：tape 経路と tape 不要経路が同じ
    /// shape 不整合に対して同じ `AutodiffError` variant を返すように
    /// する）。
    pub fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let in_shape = input.shape();
        if in_shape.len() != 4 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 4,
                actual: in_shape.len(),
            }));
        }
        let weight_shape = self.weight.shape();
        let kernel_size = [weight_shape[2], weight_shape[3]];
        let params = Conv2dParams::new(
            kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        .map_err(AutodiffError::Backend)?;
        let out_shape =
            conv2d_out_shape(in_shape, weight_shape, &params).map_err(AutodiffError::Shape)?;
        if let Some(ref bias) = self.bias {
            let cout = weight_shape[0];
            if bias.shape() != [cout] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: bias.shape().to_vec(),
                    rhs: vec![cout],
                }));
            }
        }
        conv2d_with_fallback(
            ops,
            input,
            &self.weight,
            self.bias.as_ref(),
            &params,
            &out_shape,
        )
    }
}

/// `Conv2d::bind` が返す、1 ステップ分のテープに登録済みパラメータ
/// （`nn::LinearVars` と同型）。
pub struct Conv2dVars<'t> {
    /// `Conv2d::weight`（`[out_channels, in_channels / groups, kH,
    /// kW]`）をテープへ登録した `Var`。
    pub weight: Var<'t>,
    /// `Conv2d::bias`（`[out_channels]`）をテープへ登録した `Var`。
    /// 元の `Conv2d` が `bias: None` の場合は `None`。
    pub bias: Option<Var<'t>>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl<'t> Conv2dVars<'t> {
    /// 空間軸 `[stride_h, stride_w]`（[`Conv2d::stride`] と同じ）。
    pub fn stride(&self) -> [usize; 2] {
        self.stride
    }

    /// 空間軸 `[padding_h, padding_w]`（[`Conv2d::padding`] と同じ）。
    pub fn padding(&self) -> [usize; 2] {
        self.padding
    }

    /// 空間軸 `[dilation_h, dilation_w]`（[`Conv2d::dilation`] と
    /// 同じ）。
    pub fn dilation(&self) -> [usize; 2] {
        self.dilation
    }

    /// グループ数（[`Conv2d::groups`] と同じ）。
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// `y = input.conv2d(weight, bias, stride, padding, dilation,
    /// groups)`（`Var::conv2d` への薄い委譲。追加の shape 検査は
    /// `Var::conv2d` 自身に任せる）。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.conv2d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }
}

/// ConvTranspose2d 層のパラメータ本体（イシュー #2067）。`weight` は
/// `[in_channels, out_channels / groups, kH, kW]`（PyTorch
/// `nn.ConvTranspose2d.weight` と同じレイアウト。[`Conv2d`] の
/// `[out_channels, in_channels/groups, kH, kW]` と先頭 2 軸が逆）、
/// `bias` は `Some` の場合 `[out_channels]`。
pub struct ConvTranspose2d {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    stride: [usize; 2],
    padding: [usize; 2],
    output_padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl ConvTranspose2d {
    /// 決定的シードで一様初期化する（[`Conv2d::new`] と同型）。
    ///
    /// **初期化 bound は PyTorch `nn.init._calculate_fan_in_and_fan_out`
    /// に従い `fan_in = weight.size(1)·kH·kW`（`Cout_g·kH·kW`）を使う**
    /// （`torch/nn/init.py` 確認日: 2026-09-19。`ConvTranspose2d.weight`
    /// は `[Cin, Cout_g, kH, kW]` のため `size(1) = Cout_g` が
    /// `_calculate_fan_in_and_fan_out` の「transposed 判定」分岐が返す
    /// `fan_in` に対応する。[`Conv2d::new`] の `fan_in = Cin_g·kH·kW`
    /// をそのまま流用しない——weight のレイアウトが先頭 2 軸で逆の
    /// ため fan_in の定義も入れ替わる）。
    ///
    /// # 検査順序
    ///
    /// ① [`Conv2dParams::new`] → ② `output_padding[i] < stride[i]`
    /// （[`crate::var::Var::conv_transpose2d`] と同じ意図的な PyTorch
    /// 非互換ゲート）→ ③ `in_channels == 0` 拒否 → ④
    /// `in_channels`／`out_channels` の `groups` 整合（[`Conv2d::new`]
    /// と同型）。
    #[allow(clippy::too_many_arguments)] // PyTorch `nn.ConvTranspose2d` の全引数を受理する必要があるため。
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        output_padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
        bias: bool,
        seed: u64,
    ) -> Result<ConvTranspose2d, AutodiffError> {
        let params = Conv2dParams::new(kernel_size, stride, padding, dilation, groups)
            .map_err(AutodiffError::Backend)?;
        if output_padding[0] >= stride[0] || output_padding[1] >= stride[1] {
            return Err(AutodiffError::InvalidArgument(format!(
                "ConvTranspose2d::new: output_padding ({output_padding:?}) must be < stride \
                 ({stride:?}) on each axis"
            )));
        }
        if in_channels == 0 {
            return Err(AutodiffError::InvalidArgument(
                "ConvTranspose2d::new: in_channels must be > 0 (1/sqrt(fan_in) would be \
                 non-finite)"
                    .to_string(),
            ));
        }
        if !in_channels.is_multiple_of(groups) {
            return Err(AutodiffError::InvalidArgument(format!(
                "ConvTranspose2d::new: in_channels ({in_channels}) must be divisible by groups \
                 ({groups})"
            )));
        }
        if !out_channels.is_multiple_of(groups) {
            return Err(AutodiffError::InvalidArgument(format!(
                "ConvTranspose2d::new: out_channels ({out_channels}) must be divisible by \
                 groups ({groups})"
            )));
        }
        if out_channels < groups {
            return Err(AutodiffError::InvalidArgument(format!(
                "ConvTranspose2d::new: out_channels ({out_channels}) must be >= groups \
                 ({groups})"
            )));
        }
        let cout_g = out_channels / groups;
        let [kh, kw] = kernel_size;
        // fan_in = Cout_g * kH * kW（本関数 doc の PyTorch 準拠 fan_in）。
        let fan_in = cout_g
            .checked_mul(kh)
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "ConvTranspose2d::new: fan_in (out_channels/groups * kH * kW) overflows \
                     usize"
                        .to_string(),
                )
            })?;
        if fan_in == 0 {
            return Err(AutodiffError::InvalidArgument(
                "ConvTranspose2d::new: fan_in must be > 0 (1/sqrt(fan_in) would be non-finite)"
                    .to_string(),
            ));
        }
        let bound = 1.0 / (fan_in as f32).sqrt();

        let weight_seed = derive_seed(seed, WEIGHT_SEED_SALT);
        let weight_numel = in_channels
            .checked_mul(cout_g)
            .and_then(|v| v.checked_mul(kh))
            .and_then(|v| v.checked_mul(kw))
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "ConvTranspose2d::new: weight element count overflows usize".to_string(),
                )
            })?;
        let weight_data = checked_uniform_init(
            weight_numel,
            bound,
            weight_seed,
            "ConvTranspose2d::new: weight",
        )?;
        let weight = Tensor::new(weight_data, &[in_channels, cout_g, kh, kw])?;

        let bias = if bias {
            let bias_seed = derive_seed(seed, BIAS_SEED_SALT);
            let bias_data =
                checked_uniform_init(out_channels, bound, bias_seed, "ConvTranspose2d::new: bias")?;
            Some(Tensor::new(bias_data, &[out_channels])?)
        } else {
            None
        };

        Ok(ConvTranspose2d {
            weight,
            bias,
            stride: params.stride(),
            padding: params.padding(),
            output_padding,
            dilation: params.dilation(),
            groups: params.groups(),
        })
    }

    /// 明示的な重み・バイアス・ハイパーパラメータから構築する
    /// （[`Conv2d::from_parameters`] と同型。`state_dict`／
    /// `apply_parameters` の書き戻し先・テスト向け入口）。
    ///
    /// `weight` は rank 4・`weight.shape()[1] == 0`（`out_channels/
    /// groups` が 0）を拒否する。`out_channels = weight.shape()[1] *
    /// groups`（[`Conv2d::from_parameters`] の `in_channels` 導出と
    /// 対称。`ConvTranspose2d` は weight の軸 0 が `in_channels` 其の
    /// ものであるため `in_channels = weight.shape()[0]`）。
    #[allow(clippy::too_many_arguments)] // ConvTranspose2d の全ハイパーパラメータを引数として受理する必要があるため。
    pub fn from_parameters(
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
        stride: [usize; 2],
        padding: [usize; 2],
        output_padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<ConvTranspose2d, AutodiffError> {
        if weight.rank() != 4 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 4,
                actual: weight.rank(),
            }));
        }
        let weight_shape = weight.shape().to_vec();
        let cout_g = weight_shape[1];
        if cout_g == 0 {
            return Err(AutodiffError::InvalidArgument(
                "ConvTranspose2d::from_parameters: weight.shape()[1] (out_channels/groups) must \
                 be > 0"
                    .to_string(),
            ));
        }
        let kernel_size = [weight_shape[2], weight_shape[3]];
        let params = Conv2dParams::new(kernel_size, stride, padding, dilation, groups)
            .map_err(AutodiffError::Backend)?;
        if output_padding[0] >= stride[0] || output_padding[1] >= stride[1] {
            return Err(AutodiffError::InvalidArgument(format!(
                "ConvTranspose2d::from_parameters: output_padding ({output_padding:?}) must be \
                 < stride ({stride:?}) on each axis"
            )));
        }

        let in_channels = weight_shape[0];
        if !in_channels.is_multiple_of(groups) {
            return Err(AutodiffError::InvalidArgument(format!(
                "ConvTranspose2d::from_parameters: in_channels ({in_channels}) must be \
                 divisible by groups ({groups})"
            )));
        }
        let out_channels = cout_g.checked_mul(groups).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "ConvTranspose2d::from_parameters: out_channels (weight.shape()[1] * groups) \
                 overflows usize"
                    .to_string(),
            )
        })?;
        if out_channels < groups {
            return Err(AutodiffError::InvalidArgument(format!(
                "ConvTranspose2d::from_parameters: out_channels ({out_channels}) must be >= \
                 groups ({groups})"
            )));
        }

        if let Some(ref b) = bias {
            if b.rank() != 1 {
                return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                    expected: 1,
                    actual: b.rank(),
                }));
            }
            if b.shape() != [out_channels] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![out_channels],
                }));
            }
        }

        Ok(ConvTranspose2d {
            weight,
            bias,
            stride: params.stride(),
            padding: params.padding(),
            output_padding,
            dilation: params.dilation(),
            groups: params.groups(),
        })
    }

    /// このステップの `tape` へ `weight`/`bias` を葉ノードとして登録し、
    /// `forward` を呼べる `ConvTranspose2dVars` を返す（[`Conv2d::bind`]
    /// と同型）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> ConvTranspose2dVars<'t> {
        let weight = tape.var(&self.weight);
        let bias = self.bias.as_ref().map(|b| tape.var(b));
        ConvTranspose2dVars {
            weight,
            bias,
            stride: self.stride,
            padding: self.padding,
            output_padding: self.output_padding,
            dilation: self.dilation,
            groups: self.groups,
        }
    }

    /// 重み `[in_channels, out_channels / groups, kH, kW]`
    /// （PyTorch `nn.ConvTranspose2d.weight` と同じレイアウト）。
    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    /// バイアス `[out_channels]`。
    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.bias.as_ref()
    }

    /// 空間軸 `[stride_h, stride_w]`。
    pub fn stride(&self) -> [usize; 2] {
        self.stride
    }

    /// 空間軸 `[padding_h, padding_w]`。
    pub fn padding(&self) -> [usize; 2] {
        self.padding
    }

    /// 空間軸 `[output_padding_h, output_padding_w]`（各軸
    /// `< stride` を満たす。`Var::conv_transpose2d` doc 参照）。
    pub fn output_padding(&self) -> [usize; 2] {
        self.output_padding
    }

    /// 空間軸 `[dilation_h, dilation_w]`。
    pub fn dilation(&self) -> [usize; 2] {
        self.dilation
    }

    /// グループ数。
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// [`crate::nn::module::Module::set_parameter`]（`ConvTranspose2d`
    /// 実装）の本体（[`Conv2d::set_parameter`] と同型）。
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
            "bias" => match &mut self.bias {
                Some(current) => {
                    if value.shape() != current.shape() {
                        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                            lhs: value.shape().to_vec(),
                            rhs: current.shape().to_vec(),
                        }));
                    }
                    *current = value;
                    Ok(())
                }
                None => Err(AutodiffError::InvalidArgument(format!(
                    "ConvTranspose2d::set_parameter: no parameter named `{name}` (this layer \
                     has no bias)"
                ))),
            },
            _ => Err(AutodiffError::InvalidArgument(format!(
                "ConvTranspose2d::set_parameter: no parameter named `{name}`"
            ))),
        }
    }

    /// [`crate::nn::module::Module::forward_host`]（`ConvTranspose2d`
    /// 実装）の本体。[`crate::var::Var::conv_transpose2d`] と**同じ
    /// 検査順序**で事前検査してから、同じ `grad::
    /// conv_transpose2d_with_fallback` を直接呼ぶ（[`Conv2d::forward_host`]
    /// と同じエラー型一致契約）。
    pub fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let in_shape = input.shape();
        if in_shape.len() != 4 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 4,
                actual: in_shape.len(),
            }));
        }
        let weight_shape = self.weight.shape();
        let kernel_size = [weight_shape[2], weight_shape[3]];
        let params = Conv2dParams::new(
            kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        .map_err(AutodiffError::Backend)?;
        if self.output_padding[0] >= self.stride[0] || self.output_padding[1] >= self.stride[1] {
            return Err(AutodiffError::InvalidArgument(format!(
                "ConvTranspose2d::forward_host: output_padding ({:?}) must be < stride ({:?}) \
                 on each axis",
                self.output_padding, self.stride
            )));
        }
        let out_shape =
            conv_transpose2d_out_shape(in_shape, weight_shape, &params, self.output_padding)
                .map_err(AutodiffError::Shape)?;
        if let Some(ref bias) = self.bias {
            let cout = out_shape[1];
            if bias.shape() != [cout] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: bias.shape().to_vec(),
                    rhs: vec![cout],
                }));
            }
        }
        conv_transpose2d_with_fallback(
            ops,
            input,
            &self.weight,
            self.bias.as_ref(),
            &params,
            &out_shape,
        )
    }
}

/// [`ConvTranspose2d::bind`] が返す、1 ステップ分のテープに登録済み
/// パラメータ（[`Conv2dVars`] と同型）。
pub struct ConvTranspose2dVars<'t> {
    /// [`ConvTranspose2d::weight`] をテープへ登録した `Var`。
    pub weight: Var<'t>,
    /// [`ConvTranspose2d::bias`] をテープへ登録した `Var`。元の
    /// `ConvTranspose2d` が `bias: None` の場合は `None`。
    pub bias: Option<Var<'t>>,
    stride: [usize; 2],
    padding: [usize; 2],
    output_padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
}

impl<'t> ConvTranspose2dVars<'t> {
    /// 空間軸 `[stride_h, stride_w]`（[`ConvTranspose2d::stride`] と
    /// 同じ）。
    pub fn stride(&self) -> [usize; 2] {
        self.stride
    }

    /// 空間軸 `[padding_h, padding_w]`（[`ConvTranspose2d::padding`]
    /// と同じ）。
    pub fn padding(&self) -> [usize; 2] {
        self.padding
    }

    /// 空間軸 `[output_padding_h, output_padding_w]`
    /// （[`ConvTranspose2d::output_padding`] と同じ）。
    pub fn output_padding(&self) -> [usize; 2] {
        self.output_padding
    }

    /// 空間軸 `[dilation_h, dilation_w]`（[`ConvTranspose2d::
    /// dilation`] と同じ）。
    pub fn dilation(&self) -> [usize; 2] {
        self.dilation
    }

    /// グループ数（[`ConvTranspose2d::groups`] と同じ）。
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// `y = input.conv_transpose2d(weight, bias, stride, padding,
    /// output_padding, dilation, groups)`（`Var::conv_transpose2d`
    /// への薄い委譲。追加の shape 検査は `Var::conv_transpose2d`
    /// 自身に任せる）。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.conv_transpose2d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )
    }
}

/// Conv1d 層のパラメータ本体。`weight` は `[out_channels, in_channels /
/// groups, k]`（rank 3。PyTorch `nn.Conv1d.weight` と同じレイアウト）、
/// `bias` は `Some` の場合 `[out_channels]`。
///
/// `Var::conv1d`（#1765・`var.rs`）は呼び出しのたびに `weight`／`input`
/// を `[*, *, 1, *]` へ reshape してから `Var::conv2d` へ委譲する方式
/// だが、本層は `Conv2d` を内部に保持しない——`weight` を rank 3 の
/// まま格納する（`weight()`／`named_parameters` が `&Tensor<f32>` を
/// 返す既存契約〈`nn::Linear`／`nn::Conv2d` と同型〉を保つため。rank 4
/// への reshape は「`Conv2d` を内部保持し都度 reshape して取り出す」
/// 設計だと `Tensor<f32>` を毎回新規構築せねば返せず `&Tensor<f32>`
/// を返せない——`compat::Sequential::trainable_parameters` 等の
/// `Vec<&Tensor<f32>>` 契約と整合しないため採らない）。`forward`／
/// `forward_host` の内部でのみ一時的に rank 4 へ reshape する。
pub struct Conv1d {
    weight: Tensor<f32>,
    bias: Option<Tensor<f32>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl Conv1d {
    /// `nn::Conv2d::new` と同型（`kernel_size`／`stride`／`padding`／
    /// `dilation` をスカラーのまま `Conv2dParams::new([1, k], [1,
    /// stride], [0, padding], [1, dilation], groups)`〈`Var::conv1d` と
    /// 同じ変換〉で検査する）。`fan_in = (in_channels/groups) · k`。
    #[allow(clippy::too_many_arguments)] // PyTorch `nn.Conv1d` の全引数を受理する必要があるため。
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bias: bool,
        seed: u64,
    ) -> Result<Conv1d, AutodiffError> {
        let params = Conv2dParams::new(
            [1, kernel_size],
            [1, stride],
            [0, padding],
            [1, dilation],
            groups,
        )
        .map_err(AutodiffError::Backend)?;
        if in_channels == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Conv1d::new: in_channels must be > 0 (1/sqrt(fan_in) would be non-finite)"
                    .to_string(),
            ));
        }
        if !in_channels.is_multiple_of(groups) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Conv1d::new: in_channels ({in_channels}) must be divisible by groups ({groups})"
            )));
        }
        if !out_channels.is_multiple_of(groups) {
            return Err(AutodiffError::InvalidArgument(format!(
                "Conv1d::new: out_channels ({out_channels}) must be divisible by groups \
                 ({groups})"
            )));
        }
        if out_channels < groups {
            return Err(AutodiffError::InvalidArgument(format!(
                "Conv1d::new: out_channels ({out_channels}) must be >= groups ({groups})"
            )));
        }
        let cin_g = in_channels / groups;
        let fan_in = cin_g.checked_mul(kernel_size).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "Conv1d::new: fan_in (in_channels/groups * k) overflows usize".to_string(),
            )
        })?;
        if fan_in == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Conv1d::new: fan_in must be > 0 (1/sqrt(fan_in) would be non-finite)".to_string(),
            ));
        }
        let bound = 1.0 / (fan_in as f32).sqrt();

        let weight_seed = derive_seed(seed, WEIGHT_SEED_SALT);
        let weight_numel = out_channels
            .checked_mul(cin_g)
            .and_then(|v| v.checked_mul(kernel_size))
            .ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "Conv1d::new: weight element count overflows usize".to_string(),
                )
            })?;
        let weight_data =
            checked_uniform_init(weight_numel, bound, weight_seed, "Conv1d::new: weight")?;
        let weight = Tensor::new(weight_data, &[out_channels, cin_g, kernel_size])?;

        let bias = if bias {
            let bias_seed = derive_seed(seed, BIAS_SEED_SALT);
            let bias_data =
                checked_uniform_init(out_channels, bound, bias_seed, "Conv1d::new: bias")?;
            Some(Tensor::new(bias_data, &[out_channels])?)
        } else {
            None
        };

        Ok(Conv1d {
            weight,
            bias,
            stride: params.stride()[1],
            padding: params.padding()[1],
            dilation: params.dilation()[1],
            groups: params.groups(),
        })
    }

    /// 明示的な重み・バイアス・ハイパーパラメータから構築する
    /// （`nn::Conv2d::from_parameters` と同型。`weight` は rank 3）。
    pub fn from_parameters(
        weight: Tensor<f32>,
        bias: Option<Tensor<f32>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Conv1d, AutodiffError> {
        if weight.rank() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: weight.rank(),
            }));
        }
        let weight_shape = weight.shape().to_vec();
        let cin_g = weight_shape[1];
        if cin_g == 0 {
            return Err(AutodiffError::InvalidArgument(
                "Conv1d::from_parameters: weight.shape()[1] (in_channels/groups) must be > 0"
                    .to_string(),
            ));
        }
        let kernel_size = weight_shape[2];
        let params = Conv2dParams::new(
            [1, kernel_size],
            [1, stride],
            [0, padding],
            [1, dilation],
            groups,
        )
        .map_err(AutodiffError::Backend)?;

        let out_channels = weight_shape[0];
        let in_channels = cin_g.checked_mul(groups).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "Conv1d::from_parameters: in_channels (weight.shape()[1] * groups) overflows \
                 usize"
                    .to_string(),
            )
        })?;
        if !in_channels.is_multiple_of(groups)
            || !out_channels.is_multiple_of(groups)
            || out_channels < groups
        {
            // `Conv2d::from_parameters` と同じ理由の防御的検査（現状
            // 到達しない分岐だが fail-closed 方針に従いエラーで表現する。
            // `.claude/rules/coding-rust.md` 本番経路 panic 禁止方針）。
            return Err(AutodiffError::InvalidArgument(format!(
                "Conv1d::from_parameters: in_channels ({in_channels}) / out_channels \
                 ({out_channels}) not consistent with groups ({groups})"
            )));
        }

        if let Some(ref b) = bias {
            if b.rank() != 1 {
                return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                    expected: 1,
                    actual: b.rank(),
                }));
            }
            if b.shape() != [out_channels] {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: b.shape().to_vec(),
                    rhs: vec![out_channels],
                }));
            }
        }

        Ok(Conv1d {
            weight,
            bias,
            stride: params.stride()[1],
            padding: params.padding()[1],
            dilation: params.dilation()[1],
            groups: params.groups(),
        })
    }

    /// このステップの `tape` へ `weight`/`bias` を葉ノードとして登録する
    /// （`weight` は rank 3 のまま——`Var::conv1d` 自身が内部で reshape
    /// する契約のため、ここで reshape する必要はない）。
    pub fn bind<'t>(&self, tape: &'t Tape) -> Conv1dVars<'t> {
        let weight = tape.var(&self.weight);
        let bias = self.bias.as_ref().map(|b| tape.var(b));
        Conv1dVars {
            weight,
            bias,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        }
    }

    /// 重み `[out_channels, in_channels / groups, k]`（PyTorch
    /// `nn.Conv1d.weight` と同じレイアウト）。
    pub fn weight(&self) -> &Tensor<f32> {
        &self.weight
    }

    /// バイアス `[out_channels]`。[`Conv1d::new`]／
    /// [`Conv1d::from_parameters`] に `bias: false`／`bias: None` を
    /// 渡した場合は `None`。
    pub fn bias(&self) -> Option<&Tensor<f32>> {
        self.bias.as_ref()
    }

    /// 空間軸方向のストライド。
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// 空間軸方向のパディング。
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// 空間軸方向のダイレーション。
    pub fn dilation(&self) -> usize {
        self.dilation
    }

    /// グループ数（`groups == 1` が通常の畳み込み、`groups ==
    /// in_channels` が depthwise 畳み込みに相当）。
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// [`crate::nn::module::Module::set_parameter`]（`Conv1d` 実装）の
    /// 本体（`nn::Conv2d::set_parameter` と同型。shape 保存置換のみ）。
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
            "bias" => match &mut self.bias {
                Some(current) => {
                    if value.shape() != current.shape() {
                        return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                            lhs: value.shape().to_vec(),
                            rhs: current.shape().to_vec(),
                        }));
                    }
                    *current = value;
                    Ok(())
                }
                None => Err(AutodiffError::InvalidArgument(format!(
                    "Conv1d::set_parameter: no parameter named `{name}` (this layer has no bias)"
                ))),
            },
            _ => Err(AutodiffError::InvalidArgument(format!(
                "Conv1d::set_parameter: no parameter named `{name}`"
            ))),
        }
    }

    /// [`crate::nn::module::Module::forward_host`]（`Conv1d` 実装）の
    /// 本体。`Var::conv1d` と**同じ検査順序・同じ演算列**（①rank →
    /// ② `Conv2dParams::new` → ③ 4 次元 `conv2d_out_shape` → ④ bias
    /// shape → ⑤ `contiguous` → 4 次元 reshape → `grad::
    /// conv2d_with_fallback` → 3 次元 reshape）をホスト `Tensor` で
    /// 再現する（`nn::Conv2d::forward_host` doc「エラー型の一致契約」と
    /// 同じ理由）。
    pub fn forward_host(
        &self,
        ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let in_shape = input.shape();
        if in_shape.len() != 3 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 3,
                actual: in_shape.len(),
            }));
        }
        let weight_shape = self.weight.shape();
        let (cout, cin_g, k) = (weight_shape[0], weight_shape[1], weight_shape[2]);

        let params = Conv2dParams::new(
            [1, k],
            [1, self.stride],
            [0, self.padding],
            [1, self.dilation],
            self.groups,
        )
        .map_err(AutodiffError::Backend)?;

        let (n, cin, l) = (in_shape[0], in_shape[1], in_shape[2]);
        let in_shape_4d = vec![n, cin, 1, l];
        let weight_shape_4d = vec![cout, cin_g, 1, k];
        let out_shape_4d = conv2d_out_shape(&in_shape_4d, &weight_shape_4d, &params)
            .map_err(AutodiffError::Shape)?;
        let lout = out_shape_4d[3];

        if let Some(ref bias) = self.bias
            && bias.shape() != [cout]
        {
            return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                lhs: bias.shape().to_vec(),
                rhs: vec![cout],
            }));
        }

        let input4 = input
            .contiguous()
            .reshape(&in_shape_4d)
            .map_err(AutodiffError::Shape)?;
        let weight4 = self
            .weight
            .contiguous()
            .reshape(&weight_shape_4d)
            .map_err(AutodiffError::Shape)?;
        let out4 = conv2d_with_fallback(
            ops,
            &input4,
            &weight4,
            self.bias.as_ref(),
            &params,
            &out_shape_4d,
        )?;
        out4.reshape(&[n, cout, lout]).map_err(AutodiffError::Shape)
    }
}

/// `Conv1d::bind` が返す、1 ステップ分のテープに登録済みパラメータ。
pub struct Conv1dVars<'t> {
    /// `Conv1d::weight`（`[out_channels, in_channels / groups, k]`）
    /// をテープへ登録した `Var`。
    pub weight: Var<'t>,
    /// `Conv1d::bias`（`[out_channels]`）をテープへ登録した `Var`。
    /// 元の `Conv1d` が `bias: None` の場合は `None`。
    pub bias: Option<Var<'t>>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl<'t> Conv1dVars<'t> {
    /// 空間軸方向のストライド（[`Conv1d::stride`] と同じ）。
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// 空間軸方向のパディング（[`Conv1d::padding`] と同じ）。
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// 空間軸方向のダイレーション（[`Conv1d::dilation`] と同じ）。
    pub fn dilation(&self) -> usize {
        self.dilation
    }

    /// グループ数（[`Conv1d::groups`] と同じ）。
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// `Var::conv1d` への薄い委譲。
    pub fn forward(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.conv1d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
    }
}
