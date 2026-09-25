//! MaxPool／AvgPool／AdaptiveAvgPool（1d／2d。PyTorch `nn.MaxPool2d`
//! 等相当。イシュー #1728・設計 `docs/pooling-ops-design.md` §9）。
//!
//! いずれも**無状態**（`named_parameters` は空・`set_parameter` は
//! `Module` の既定 `Err`。学習可能パラメータを持たない空間演算）。
//! コンストラクタ `new(...)` は [`fandhe_ai_tensor_core::
//! Pool2dParams::new`] 相当の検査を前倒しし `Result` を返す
//! （`nn::Conv2d::new` と同型。「構築は成功したが forward が常に
//! 失敗する」層を作らせない）。`ceil_mode` はコンストラクタに含めない
//! ——v1 は `false` 固定（設計 doc §3・§11 スコープ外）のため。
//!
//! `Module::forward`／[`fandhe_ai_tensor_core::BackendOps`] 直呼びの
//! `forward_host` の 2 経路は `Var::max_pool2d`／`avg_pool2d`／
//! `adaptive_avg_pool2d` と同じ検査順序・同じフォールバック規律
//! （`grad::*_with_fallback`）を踏み、bit 完全一致を保つ
//! （`impl Module for MaxPool2d` 等は `nn/module.rs` に置く既存慣習に
//! 従う）。`MaxPool1d`／`MaxPool2d` の `Module::forward` は
//! `(values, index)` のうち `values` のみを返す（索引が必要な場合は
//! [`MaxPool2d::forward`]／[`MaxPool1d::forward`] を直接呼ぶ）。

use crate::adaptive_max_pool_ops;
use crate::error::AutodiffError;
use crate::var::Var;
use fandhe_ai_tensor_core::{Pool2dParams, ShapeError, Tensor};

/// PyTorch `torch.nn.MaxPool2d` 相当（`ceil_mode=false` 固定）。
#[derive(Debug, Clone, PartialEq)]
pub struct MaxPool2d {
    params: Pool2dParams,
}

impl MaxPool2d {
    /// `kernel_size`／`stride`（省略時 `kernel_size` と同じ。
    /// PyTorch 既定）／`padding`／`dilation` から構築する。
    /// [`Pool2dParams::new`] の検査（カーネル 0 拒否・padding
    /// 上限〈`padding <= kernel_size/2`〉等）を前倒しして行う。
    pub fn new(
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self, AutodiffError> {
        let params = Pool2dParams::new(kernel_size, stride, padding, dilation)
            .map_err(AutodiffError::Backend)?;
        Ok(Self { params })
    }

    /// クレート内アクセサ（`nn/module.rs::impl Module for MaxPool2d`
    /// が構築済みの検査済みパラメータを読み出すため。フィールド自体は
    /// 非公開のまま）。
    pub(crate) fn params(&self) -> &Pool2dParams {
        &self.params
    }

    /// `self.params` を用いて [`Var::max_pool2d`] へ委譲する。
    /// `NCHW` 入力を受け取り、`(values, index)` を返す
    /// （`index` は先勝ち決定的タイ規則で選ばれた入力位置の
    /// 平坦化添字。backward に必要な場合は呼び出し側が保持する）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
        let [kh, kw] = self.params.kernel_size();
        let [sh, sw] = self.params.stride();
        let [ph, pw] = self.params.padding();
        let [dh, dw] = self.params.dilation();
        input.max_pool2d([kh, kw], Some([sh, sw]), [ph, pw], [dh, dw], false)
    }
}

/// PyTorch `torch.nn.MaxPool1d` 相当（`ceil_mode=false` 固定）。
/// [`MaxPool2d`] を `H` 軸固定（`kernel=1`・`stride=1`・`padding=0`・
/// `dilation=1`）で保持する薄いラッパー（[`Var::max_pool1d`] と同型）。
#[derive(Debug, Clone, PartialEq)]
pub struct MaxPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl MaxPool1d {
    /// `kernel_size`／`stride`（省略時 `kernel_size` と同じ）／
    /// `padding`／`dilation` から構築する。内部で `[1, k]` 形の
    /// [`Pool2dParams`] を構築し検査を前倒しする（`H` 軸は常に
    /// `kernel=1`・`stride=1`・`padding=0`・`dilation=1` 固定）。
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        dilation: usize,
    ) -> Result<Self, AutodiffError> {
        // `[1, k]` 形の `Pool2dParams` で早期検査する（構築時点の検査
        // 前倒し。`Var::max_pool1d` 自身の rank 検査は forward 呼び出し
        // 時にのみ判明するため対象外）。
        let stride2 = stride.map(|s| [1, s]);
        Pool2dParams::new([1, kernel_size], stride2, [0, padding], [1, dilation])
            .map_err(AutodiffError::Backend)?;
        Ok(Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding,
            dilation,
        })
    }

    /// `self` の各パラメータを用いて [`Var::max_pool1d`] へ委譲
    /// する。`NCL` 入力を受け取り、`(values, index)` を返す
    /// （`index` の意味は [`MaxPool2d::forward`] と同じ）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
        input.max_pool1d(
            self.kernel_size,
            Some(self.stride),
            self.padding,
            self.dilation,
            false,
        )
    }

    /// `nn/module.rs::impl Module for MaxPool1d::forward_host` が
    /// `[1, k]` 形の `Pool2dParams` を再構築するためのクレート内
    /// アクセサ群（`Softplus::beta`／`threshold` と同じ理由）。
    pub(crate) fn kernel_size_2d(&self) -> [usize; 2] {
        [1, self.kernel_size]
    }

    pub(crate) fn stride_2d(&self) -> [usize; 2] {
        [1, self.stride]
    }

    pub(crate) fn padding_2d(&self) -> [usize; 2] {
        [0, self.padding]
    }

    pub(crate) fn dilation_2d(&self) -> [usize; 2] {
        [1, self.dilation]
    }
}

/// PyTorch `torch.nn.AvgPool2d` 相当（`ceil_mode=false` 固定）。
#[derive(Debug, Clone, PartialEq)]
pub struct AvgPool2d {
    params: Pool2dParams,
    count_include_pad: bool,
}

impl AvgPool2d {
    /// `kernel_size`／`stride`（省略時 `kernel_size` と同じ）／
    /// `padding`／`count_include_pad`（PyTorch `nn.AvgPool2d`
    /// 既定は `true`。padding 領域を分母に含めるかどうか）から
    /// 構築する。`dilation=[1,1]` 固定（設計 doc §2）。
    pub fn new(
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> Result<Self, AutodiffError> {
        // Avg 系は `dilation=[1,1]` 固定（設計 doc §2）。
        let params = Pool2dParams::new(kernel_size, stride, padding, [1, 1])
            .map_err(AutodiffError::Backend)?;
        Ok(Self {
            params,
            count_include_pad,
        })
    }

    pub(crate) fn params(&self) -> &Pool2dParams {
        &self.params
    }

    pub(crate) fn count_include_pad(&self) -> bool {
        self.count_include_pad
    }

    /// `self.params`／`self.count_include_pad` を用いて
    /// [`Var::avg_pool2d`] へ委譲する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let [kh, kw] = self.params.kernel_size();
        let [sh, sw] = self.params.stride();
        let [ph, pw] = self.params.padding();
        input.avg_pool2d(
            [kh, kw],
            Some([sh, sw]),
            [ph, pw],
            false,
            self.count_include_pad,
        )
    }
}

/// PyTorch `torch.nn.AvgPool1d` 相当（`ceil_mode=false` 固定）。
/// [`AvgPool2d`] を `H` 軸固定で保持する薄いラッパー。
#[derive(Debug, Clone, PartialEq)]
pub struct AvgPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
    count_include_pad: bool,
}

impl AvgPool1d {
    /// `kernel_size`／`stride`（省略時 `kernel_size` と同じ）／
    /// `padding`／`count_include_pad` から構築する。内部で
    /// `[1, k]` 形の [`Pool2dParams`] を構築し検査を前倒しする。
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
        count_include_pad: bool,
    ) -> Result<Self, AutodiffError> {
        let stride2 = stride.map(|s| [1, s]);
        Pool2dParams::new([1, kernel_size], stride2, [0, padding], [1, 1])
            .map_err(AutodiffError::Backend)?;
        Ok(Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding,
            count_include_pad,
        })
    }

    /// `self` の各パラメータを用いて [`Var::avg_pool1d`] へ委譲
    /// する。`NCL` 入力を受け取り出力を返す（無状態のため勾配は
    /// 常に `f64` 縮約契約で決定的に求まる）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.avg_pool1d(
            self.kernel_size,
            Some(self.stride),
            self.padding,
            false,
            self.count_include_pad,
        )
    }

    /// `nn/module.rs::impl Module for AvgPool1d::forward_host` が
    /// `[1, k]` 形の `Pool2dParams` を再構築するためのクレート内
    /// アクセサ群。
    pub(crate) fn kernel_size_2d(&self) -> [usize; 2] {
        [1, self.kernel_size]
    }

    pub(crate) fn stride_2d(&self) -> [usize; 2] {
        [1, self.stride]
    }

    pub(crate) fn padding_2d(&self) -> [usize; 2] {
        [0, self.padding]
    }

    pub(crate) fn count_include_pad(&self) -> bool {
        self.count_include_pad
    }
}

/// PyTorch `torch.nn.AdaptiveAvgPool2d` 相当。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveAvgPool2d {
    output_size: [usize; 2],
}

impl AdaptiveAvgPool2d {
    /// 出力空間サイズ `output_size`（`[out_h, out_w]`）から構築
    /// する。各軸が `0` の場合は `AutodiffError::InvalidArgument`
    /// を返す（PyTorch は `output_size=0` を許さないため）。
    pub fn new(output_size: [usize; 2]) -> Result<Self, AutodiffError> {
        if output_size[0] == 0 || output_size[1] == 0 {
            return Err(AutodiffError::InvalidArgument(
                "AdaptiveAvgPool2d::new: output_size の各軸は 1 以上である必要がある".into(),
            ));
        }
        Ok(Self { output_size })
    }

    pub(crate) fn output_size(&self) -> [usize; 2] {
        self.output_size
    }

    /// `self.output_size` を用いて [`Var::adaptive_avg_pool2d`]
    /// へ委譲する。`NCHW` 入力を受け取り `[N, C, out_h, out_w]`
    /// の出力を返す（窓は [`fandhe_ai_tensor_core::adaptive_window`]
    /// の重なり許容規則で決まる）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.adaptive_avg_pool2d(self.output_size)
    }
}

/// PyTorch `torch.nn.AdaptiveAvgPool1d` 相当。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveAvgPool1d {
    output_size: usize,
}

impl AdaptiveAvgPool1d {
    /// 出力長 `output_size` から構築する。`0` の場合は
    /// `AutodiffError::InvalidArgument` を返す。
    pub fn new(output_size: usize) -> Result<Self, AutodiffError> {
        if output_size == 0 {
            return Err(AutodiffError::InvalidArgument(
                "AdaptiveAvgPool1d::new: output_size は 1 以上である必要がある".into(),
            ));
        }
        Ok(Self { output_size })
    }

    /// `self.output_size` を用いて [`Var::adaptive_avg_pool1d`]
    /// へ委譲する。`NCL` 入力を受け取り `[N, C, output_size]`
    /// の出力を返す（窓決定規則は [`AdaptiveAvgPool2d::forward`]
    /// と同じ。`W` 軸固定で `H` 軸〈`out_h=1`〉を通すラッパー）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        input.adaptive_avg_pool1d(self.output_size)
    }

    /// `nn/module.rs::impl Module for AdaptiveAvgPool1d::forward_host`
    /// が `[1, o]` 形の `output_size` を再構築するためのクレート内
    /// アクセサ。
    pub(crate) fn output_size_1d(&self) -> usize {
        self.output_size
    }
}

/// PyTorch `torch.nn.AdaptiveMaxPool2d` 相当（イシュー #2160）。
/// **内部クレート限定**（facade 未公開。`crate::adaptive_max_pool_ops`
/// モジュール doc §承認事項を参照。`Var::adaptive_max_pool2d` の
/// 公開委譲メソッド・facade `compat::Sequential::
/// add_adaptive_max_pool2d` は未承認のため追加しない）。
///
/// [`AdaptiveAvgPool2d`] と異なり `forward` は `(values, index)` を
/// 返す（[`MaxPool2d::forward`] と同型。索引は `(n,c)` 平面内 flat
/// 添字 `h·W+w`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveMaxPool2d {
    output_size: [usize; 2],
}

impl AdaptiveMaxPool2d {
    /// 出力空間サイズ `output_size`（`[out_h, out_w]`）から構築
    /// する。各軸が `0` の場合は `AutodiffError::InvalidArgument`
    /// を返す（[`AdaptiveAvgPool2d::new`] と同型）。
    pub fn new(output_size: [usize; 2]) -> Result<Self, AutodiffError> {
        if output_size[0] == 0 || output_size[1] == 0 {
            return Err(AutodiffError::InvalidArgument(
                "AdaptiveMaxPool2d::new: output_size の各軸は 1 以上である必要がある".into(),
            ));
        }
        Ok(Self { output_size })
    }

    pub(crate) fn output_size(&self) -> [usize; 2] {
        self.output_size
    }

    /// `self.output_size` を用いて
    /// `crate::adaptive_max_pool_ops::adaptive_max_pool2d` へ委譲
    /// する（非公開項目のため intra-doc link にしない）。`NCHW`
    /// 入力を受け取り `(values, index)` を返す（窓は
    /// [`fandhe_ai_tensor_core::adaptive_window`] の重なり許容規則で
    /// 決まる。タイ規則・NaN 規則は [`MaxPool2d::forward`] と同一）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
        adaptive_max_pool_ops::adaptive_max_pool2d(input, self.output_size)
    }
}

/// PyTorch `torch.nn.AdaptiveMaxPool1d` 相当（イシュー #2160）。
/// [`AdaptiveMaxPool2d`] を `H` 軸固定（`output_size[0]=1`）で保持
/// する薄いラッパー（[`AdaptiveAvgPool1d`] と同型）。**内部クレート
/// 限定**（[`AdaptiveMaxPool2d`] の doc 参照）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveMaxPool1d {
    output_size: usize,
}

impl AdaptiveMaxPool1d {
    /// 出力長 `output_size` から構築する。`0` の場合は
    /// `AutodiffError::InvalidArgument` を返す。
    pub fn new(output_size: usize) -> Result<Self, AutodiffError> {
        if output_size == 0 {
            return Err(AutodiffError::InvalidArgument(
                "AdaptiveMaxPool1d::new: output_size は 1 以上である必要がある".into(),
            ));
        }
        Ok(Self { output_size })
    }

    /// `self.output_size` を用いて
    /// `crate::adaptive_max_pool_ops::adaptive_max_pool1d` へ委譲
    /// する（非公開項目のため intra-doc link にしない）。`NCL`
    /// 入力を受け取り `(values, index)` を返す（窓決定
    /// 規則は [`AdaptiveMaxPool2d::forward`] と同じ）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
        adaptive_max_pool_ops::adaptive_max_pool1d(input, self.output_size)
    }

    /// `nn/module.rs::impl Module for AdaptiveMaxPool1d::forward_host`
    /// が `[1, o]` 形の `output_size` を再構築するためのクレート内
    /// アクセサ。
    pub(crate) fn output_size_1d(&self) -> usize {
        self.output_size
    }
}

/// GlobalPool の集約方式（イシュー #2160）。ONNX `GlobalAveragePool`／
/// `GlobalMaxPool`、Keras `GlobalAveragePooling*`／`GlobalMaxPooling*`
/// 相当を単一の型 [`GlobalPool`] へ集約するための variant。
/// `#[non_exhaustive]` により将来の variant 追加（例: `GlobalLp`）を
/// 破壊的変更なしで行えるようにする（`docs/compat-api-scope.md` の
/// 非破壊拡張方針）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GlobalPoolMode {
    /// 空間軸全体の平均（[`AdaptiveAvgPool2d`]／`AdaptiveAvgPool1d`
    /// の `output_size=1` に委譲）。
    Avg,
    /// 空間軸全体の最大（[`AdaptiveMaxPool2d`] の `output_size=1` に
    /// 委譲。`Op::MaxPool2d` を経由するため索引は破棄する）。
    Max,
}

/// 空間軸全体を単一値へ縮約する層（Keras `GlobalAveragePooling2D`／
/// `GlobalMaxPooling2D`、ONNX `GlobalAveragePool`／`GlobalMaxPool`
/// 相当。イシュー #2160・設計 `docs/pooling-ops-design.md` §11）。
///
/// rank 3（`[N, C, L]`）・rank 4（`[N, C, H, W]`）のいずれも受理し、
/// [`Self::forward`] が rank で分岐する（`AdaptiveAvgPool1d`／`2d` の
/// 使い分けと同型）。`keepdims`: `true` なら ONNX `Global*Pool` 互換で
/// 空間軸を `1` のまま残す（`[N,C,1]`／`[N,C,1,1]`）・`false` なら
/// Keras 既定互換で空間軸を潰す（`[N,C]`）。
///
/// `Max` の場合、[`AdaptiveMaxPool2d::forward`] が返す索引は
/// [`Self::forward`] の戻り値には含まれない（値のみが必要な用途を
/// 主眼とするため。索引が必要な場合は [`AdaptiveMaxPool2d::forward`]
/// を直接呼ぶこと）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlobalPool {
    mode: GlobalPoolMode,
    keepdims: bool,
}

impl GlobalPool {
    /// 集約方式 `mode` と `keepdims` から構築する。引数検査は不要
    /// （`output_size` を伴わないため [`AdaptiveAvgPool2d::new`] 等と
    /// 異なり `Result` を返さない）。
    pub fn new(mode: GlobalPoolMode, keepdims: bool) -> Self {
        Self { mode, keepdims }
    }

    pub(crate) fn mode(&self) -> GlobalPoolMode {
        self.mode
    }

    pub(crate) fn keepdims(&self) -> bool {
        self.keepdims
    }

    /// `self.mode`／`self.keepdims` に従って空間軸全体を縮約する。
    /// rank 4（`NCHW`）は [`AdaptiveAvgPool2d::forward`]／
    /// [`AdaptiveMaxPool2d::forward`] の `output_size=[1,1]` に、
    /// rank 3（`NCL`）は `AdaptiveAvgPool1d`／[`AdaptiveMaxPool1d`]
    /// の `output_size=1` に委譲する。rank がそれ以外の場合は
    /// `AutodiffError::Shape(ShapeError::RankMismatch)` を返す
    /// （`expected` は `4`。rank 3 と rank 4 のどちらでもない契約
    /// 違反を型付きエラーで拒否する。REQ-8・A08）。
    ///
    /// `keepdims=false` の reshape は **縮約が確定した後にのみ**行う
    /// （検査を reshape より前に完了させる `Var::adaptive_avg_pool1d`
    /// 等の規律とは逆方向だが、ここでの reshape は「出力の後処理」
    /// であり孤立 view ノードのリスクがある「入力の前処理」ではない
    /// ため対象外）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let in_shape = input.shape();
        let reduced = match in_shape.len() {
            4 => {
                let (n, c) = (in_shape[0], in_shape[1]);
                let reduced = match self.mode {
                    GlobalPoolMode::Avg => input.adaptive_avg_pool2d([1, 1])?,
                    GlobalPoolMode::Max => {
                        adaptive_max_pool_ops::adaptive_max_pool2d(input, [1, 1])?.0
                    }
                };
                if self.keepdims {
                    reduced
                } else {
                    reduced.reshape(&[n, c])?
                }
            }
            3 => {
                let (n, c) = (in_shape[0], in_shape[1]);
                let reduced = match self.mode {
                    GlobalPoolMode::Avg => input.adaptive_avg_pool1d(1)?,
                    GlobalPoolMode::Max => adaptive_max_pool_ops::adaptive_max_pool1d(input, 1)?.0,
                };
                if self.keepdims {
                    reduced
                } else {
                    reduced.reshape(&[n, c])?
                }
            }
            actual => {
                return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                    expected: 4,
                    actual,
                }));
            }
        };
        Ok(reduced)
    }
}
