//! 指数移動平均（EMA）の facade 公開ラッパー（イシュー #2560・親 #2558。
//! 決定記録 `docs/autodiff-ema-decision.md` §10.2 (a)(b)）。
//!
//! 役割: 内部クレートの [`fandhe_ai_autodiff::nn::ExponentialMovingAverage`]
//! （`crates/autodiff/src/nn/ema.rs`）を 1 フィールドで保持し、各メソッドを
//! 同名の内部メソッドへ 1 行で委譲する薄いラッパー。素の再エクスポートを採らない
//! 理由は、内部型の `from_module`／`apply` 等が内部クレートの `nn::Module` を
//! 引数に取り、公開シグネチャへ内部 trait が露出するため。本型の
//! `from_module`／`update_from_module`／`apply`／`restore` は facade の
//! [`crate::nn::Module`] を受け、その `named_parameters`／`state_dict`／
//! `load_state_dict` 経由で委譲する。
//!
//! 公開パスは `fandhe_ai::optim::ExponentialMovingAverage` のみ
//! （`optim.rs` が `pub use` する。本ファイルの `mod optim_ema;` は private）。
//! `compat::Sequential::fit_with_callbacks` への結線は
//! [`crate::compat::Callback::Ema`]（[`crate::compat::EmaCallback`]）が担う。
//!
//! # 数値契約
//!
//! 更新式は `shadow = f32::mul_add(decay, shadow, (1 - decay) * param)`
//! （内部実装と同一。`.claude/rules/coding-rust.md` の FMA 契約）。PyTorch
//! `AveragedModel` の `lerp` 形とは丸めが異なるため bit 一致は主張しない。
//! 非有限値は伝播する。
//!
//! # スコープ
//!
//! 対象は学習可能パラメータのみ（`BatchNorm` の running buffer は対象外）。
//! デバイス常駐経路（`DeviceParamStore`）の重みは追従しない（常駐経路と併用すると
//! shadow が stale 化する）。ホスト `Tensor<f32>` のみを扱い、GPU 固有の数値経路は
//! 持たない。

use std::collections::HashMap;

use crate::nn::Module;
use crate::{AutodiffError, Tensor};

/// パラメータの指数移動平均（shadow copy）を保持する facade の公開型。
///
/// ```
/// use fandhe_ai::Tensor;
/// use fandhe_ai::optim::ExponentialMovingAverage;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let w0 = Tensor::new(vec![1.0f32, 2.0], &[2])?;
/// let mut ema = ExponentialMovingAverage::new(0.5, &[&w0])?;
/// let w1 = Tensor::new(vec![3.0f32, 4.0], &[2])?;
/// ema.update(&[&w1])?;
/// // shadow = 0.5 * w0 + 0.5 * w1
/// let shadow = ema.shadow_parameters()[0].contiguous();
/// assert_eq!(shadow.as_slice().unwrap(), &[2.0, 3.0]);
/// assert_eq!(ema.num_updates(), 1);
///
/// // decay は有限かつ [0, 1]。範囲外は型付きエラー。
/// assert!(ExponentialMovingAverage::new(1.5, &[&w0]).is_err());
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct ExponentialMovingAverage {
    inner: fandhe_ai_autodiff::nn::ExponentialMovingAverage,
}

impl ExponentialMovingAverage {
    /// `params`（位置対応。登録名は `"0"`, `"1"`, …）から構築する。
    ///
    /// # Errors
    /// `decay` が非有限または `[0, 1]` 外のとき [`AutodiffError::InvalidArgument`]。
    pub fn new(decay: f32, params: &[&Tensor<f32>]) -> Result<Self, AutodiffError> {
        Ok(Self {
            inner: fandhe_ai_autodiff::nn::ExponentialMovingAverage::new(decay, params)?,
        })
    }

    /// 名前付きパラメータ列から構築する（`Sequential::named_parameters()` の戻り値を
    /// そのまま渡せる）。名前重複・不正 `decay` は [`AutodiffError::InvalidArgument`]。
    pub fn from_named(
        decay: f32,
        named: Vec<(String, &Tensor<f32>)>,
    ) -> Result<Self, AutodiffError> {
        Ok(Self {
            inner: fandhe_ai_autodiff::nn::ExponentialMovingAverage::from_named(decay, named)?,
        })
    }

    /// facade の [`Module`] の `named_parameters()` から構築する。
    pub fn from_module(decay: f32, module: &dyn Module) -> Result<Self, AutodiffError> {
        Self::from_named(decay, module.named_parameters())
    }

    /// 構築時に検証済みの `decay`。
    pub fn decay(&self) -> f32 {
        self.inner.decay()
    }

    /// これまでの `update` 系の成功回数。
    pub fn num_updates(&self) -> u64 {
        self.inner.num_updates()
    }

    /// 位置対応で shadow を更新する。個数不一致は [`AutodiffError::InvalidArgument`]、
    /// shape 不一致は [`AutodiffError::Shape`]。いずれも shadow を変更しない（two-pass 検証）。
    pub fn update(&mut self, params: &[&Tensor<f32>]) -> Result<(), AutodiffError> {
        self.inner.update(params)
    }

    /// 名前付きで shadow を更新する。名前集合の不一致は [`AutodiffError::InvalidArgument`]、
    /// shape 不一致は [`AutodiffError::Shape`]。いずれも shadow を変更しない（two-pass 検証）。
    pub fn update_named(
        &mut self,
        named: Vec<(String, &Tensor<f32>)>,
    ) -> Result<(), AutodiffError> {
        self.inner.update_named(named)
    }

    /// facade の [`Module`] の `named_parameters()` で shadow を更新する。
    pub fn update_from_module(&mut self, module: &dyn Module) -> Result<(), AutodiffError> {
        self.update_named(module.named_parameters())
    }

    /// 登録名 `name` の shadow（無ければ `None`）。
    pub fn shadow(&self, name: &str) -> Option<&Tensor<f32>> {
        self.inner.shadow(name)
    }

    /// 登録順の shadow 参照列。
    pub fn shadow_parameters(&self) -> Vec<&Tensor<f32>> {
        self.inner.shadow_parameters()
    }

    /// shadow のキー付きコピー（`Module::load_state_dict` へそのまま渡せる）。
    pub fn shadow_state_dict(&self) -> HashMap<String, Tensor<f32>> {
        self.inner.shadow_state_dict()
    }

    /// `model` の重みを shadow へ一時的に差し替え、差し替え前の `state_dict()` を返す
    /// （[`Module::load_state_dict`] のアトミック性契約を継承する）。
    /// 評価後は [`Self::restore`] へ戻り値を渡して元の重みへ戻す。
    pub fn apply(
        &self,
        model: &mut dyn Module,
    ) -> Result<HashMap<String, Tensor<f32>>, AutodiffError> {
        let backup = model.state_dict();
        model.load_state_dict(self.inner.shadow_state_dict())?;
        Ok(backup)
    }

    /// [`Self::apply`] が返した退避値で `model` を元の重みへ戻す。
    pub fn restore(
        model: &mut dyn Module,
        backup: HashMap<String, Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        model.load_state_dict(backup)
    }
}
