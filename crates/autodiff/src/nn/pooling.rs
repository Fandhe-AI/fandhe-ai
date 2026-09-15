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

use crate::error::AutodiffError;
use crate::var::Var;
use fandhe_ai_tensor_core::{Pool2dParams, Tensor};

/// PyTorch `torch.nn.MaxPool2d` 相当（`ceil_mode=false` 固定）。
#[derive(Debug, Clone, PartialEq)]
pub struct MaxPool2d {
    params: Pool2dParams,
}

impl MaxPool2d {
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
    pub fn new(output_size: usize) -> Result<Self, AutodiffError> {
        if output_size == 0 {
            return Err(AutodiffError::InvalidArgument(
                "AdaptiveAvgPool1d::new: output_size は 1 以上である必要がある".into(),
            ));
        }
        Ok(Self { output_size })
    }

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
