//! `nn::MaxUnpool1d`／`MaxUnpool2d`／`MaxUnpool3d`（PyTorch `nn.MaxUnpool{1,2,3}d` 相当。イシュー #3035・
//! 親 #3018・`docs/autodiff-conv-transpose3d-max-unpool-decision.md` §18）。
//!
//! 公開形は `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §3・§6・§12 を正とする。
//! 入力が「値」と「索引 `Tensor<i32>`」の 2 つであり、単入力・単出力の `Module::forward`
//! に載らないため、**`Module` は実装せず固有メソッド `forward` を持つ無状態の型**とする
//! （`Module::as_*` フックも設けない。`Sequential` へも載せない）。
//!
//! 実体は `Var::max_unpool{1,2,3}d`（`crate::max_unpool_ops`）への 1 行委譲で、新規 `Op`／`BackendOps`／
//! カーネルは追加しない。forward 値・入力勾配は直呼びと bit 一致し、重複索引の勾配規則も
//! 同決定記録 §12.2(a) のまま変わらない。構築時に 0 次元とオーバーフローを前倒し検査し、
//! `padding` と `kernel_size` の大小は入力長に依存するため forward 側（`max_unpool_layout`）の
//! `Shape` エラーに任せる。

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;
use crate::var::Var;

/// `kernel_size`・`stride`・`padding` の構築時検査（0 次元・`2 * padding` のオーバーフロー）。
/// `max_unpool_layout` の同検査を前倒しし、構築成功なのに forward が常に失敗する層を避ける。
fn validate_new(
    name: &str,
    kernel_size: &[usize],
    stride: Option<&[usize]>,
    padding: &[usize],
) -> Result<(), AutodiffError> {
    if kernel_size.contains(&0) {
        return Err(AutodiffError::InvalidArgument(format!(
            "{name}: kernel_size の各軸は 1 以上が必要（{kernel_size:?}）"
        )));
    }
    if let Some(s) = stride
        && s.contains(&0)
    {
        return Err(AutodiffError::InvalidArgument(format!(
            "{name}: stride の各軸は 1 以上が必要（{s:?}）"
        )));
    }
    if padding.iter().any(|p| p.checked_mul(2).is_none()) {
        return Err(AutodiffError::InvalidArgument(format!(
            "{name}: padding の 2 倍が usize を超える（{padding:?}）"
        )));
    }
    Ok(())
}

/// 1D MaxUnpool 層（`nn.MaxUnpool1d` 相当）。入力 `[N, C, L]` と索引から `[N, C, L_out]` を復元する。
#[derive(Debug, Clone, PartialEq)]
pub struct MaxUnpool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxUnpool1d {
    /// `stride = None` は `kernel_size`。0 次元・`padding` の 2 倍オーバーフローは `InvalidArgument`。
    pub fn new(
        kernel_size: usize,
        stride: Option<usize>,
        padding: usize,
    ) -> Result<Self, AutodiffError> {
        validate_new(
            "MaxUnpool1d::new",
            &[kernel_size],
            stride.as_ref().map(std::slice::from_ref),
            &[padding],
        )?;
        Ok(Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding,
        })
    }

    /// カーネル長。
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// 解決済みストライド。
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// パディング。
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// `Var::max_unpool1d` へ 1 行委譲する。`indices` は `Var::max_pool1d` が返す索引。
    pub fn forward<'t>(
        &self,
        input: &Var<'t>,
        indices: &Tensor<i32>,
        output_size: Option<usize>,
    ) -> Result<Var<'t>, AutodiffError> {
        input.max_unpool1d(
            indices,
            self.kernel_size,
            Some(self.stride),
            self.padding,
            output_size,
        )
    }
}

/// 2D MaxUnpool 層（`nn.MaxUnpool2d` 相当）。入力 `[N, C, H, W]`。
#[derive(Debug, Clone, PartialEq)]
pub struct MaxUnpool2d {
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl MaxUnpool2d {
    /// `stride = None` は `kernel_size`。検査は [`MaxUnpool1d::new`] と同じ。
    pub fn new(
        kernel_size: [usize; 2],
        stride: Option<[usize; 2]>,
        padding: [usize; 2],
    ) -> Result<Self, AutodiffError> {
        validate_new(
            "MaxUnpool2d::new",
            &kernel_size,
            stride.as_ref().map(|s| s.as_slice()),
            &padding,
        )?;
        Ok(Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding,
        })
    }

    /// カーネルサイズ。
    pub fn kernel_size(&self) -> [usize; 2] {
        self.kernel_size
    }

    /// 解決済みストライド。
    pub fn stride(&self) -> [usize; 2] {
        self.stride
    }

    /// パディング。
    pub fn padding(&self) -> [usize; 2] {
        self.padding
    }

    /// `Var::max_unpool2d` へ 1 行委譲する。
    pub fn forward<'t>(
        &self,
        input: &Var<'t>,
        indices: &Tensor<i32>,
        output_size: Option<[usize; 2]>,
    ) -> Result<Var<'t>, AutodiffError> {
        input.max_unpool2d(
            indices,
            self.kernel_size,
            Some(self.stride),
            self.padding,
            output_size,
        )
    }
}

/// 3D MaxUnpool 層（`nn.MaxUnpool3d` 相当）。入力 `[N, C, D, H, W]`。
#[derive(Debug, Clone, PartialEq)]
pub struct MaxUnpool3d {
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
}

impl MaxUnpool3d {
    /// `stride = None` は `kernel_size`。検査は [`MaxUnpool1d::new`] と同じ。
    pub fn new(
        kernel_size: [usize; 3],
        stride: Option<[usize; 3]>,
        padding: [usize; 3],
    ) -> Result<Self, AutodiffError> {
        validate_new(
            "MaxUnpool3d::new",
            &kernel_size,
            stride.as_ref().map(|s| s.as_slice()),
            &padding,
        )?;
        Ok(Self {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
            padding,
        })
    }

    /// カーネルサイズ。
    pub fn kernel_size(&self) -> [usize; 3] {
        self.kernel_size
    }

    /// 解決済みストライド。
    pub fn stride(&self) -> [usize; 3] {
        self.stride
    }

    /// パディング。
    pub fn padding(&self) -> [usize; 3] {
        self.padding
    }

    /// `Var::max_unpool3d` へ 1 行委譲する。
    pub fn forward<'t>(
        &self,
        input: &Var<'t>,
        indices: &Tensor<i32>,
        output_size: Option<[usize; 3]>,
    ) -> Result<Var<'t>, AutodiffError> {
        input.max_unpool3d(
            indices,
            self.kernel_size,
            Some(self.stride),
            self.padding,
            output_size,
        )
    }
}
