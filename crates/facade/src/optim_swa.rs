//! SWA（確率的重み平均）の重み平均器の facade 公開ラッパー（イシュー #2679・親 #2625。
//! 決定記録 `docs/autodiff-swa-decision.md` §7）。
//!
//! 役割: 内部クレートの [`fandhe_ai_autodiff::nn::AveragedModel`]
//! （`crates/autodiff/src/nn/swa.rs`）を 1 フィールドで保持し、各メソッドを同名の
//! 内部メソッドへ 1 行で委譲する薄いラッパー（`optim_ema.rs` と同型。決定記録 §7 が
//! 「EMA §10.2 (b) と同型」と定めている）。素の再エクスポートを採らない理由は、
//! 内部型の `from_module`／`update_from_module`／`apply`／`restore` が内部クレートの
//! `nn::Module` を引数に取り、公開シグネチャへ内部 trait が露出する（REQ-12）ため。
//! 本型のこれら 4 メソッドは facade の [`crate::nn::Module`] を受け、その
//! `named_parameters`／`state_dict`／`load_state_dict` 経由で委譲する。
//!
//! 公開パスは `fandhe_ai::optim::AveragedModel` のみ（`optim.rs` が `pub use` する。
//! 本ファイルの `mod optim_swa;` は private）。学習率側の `SwaLr`／`SwaAnneal` は
//! `optim.rs` が内部クレートから素の再エクスポートする。`fit` への結線
//! （`FitConfig` の `use_swa` 等）は承認されておらず存在しない。
//!
//! # 数値契約
//!
//! 内部実装と同一（`n_averaged == 0` は複製、以降は `f32::mul_add` の lerp 形。
//! PyTorch との bit 一致は主張しない）。非有限値は伝播する。
//!
//! # スコープ
//!
//! 対象は学習可能パラメータのみ（`BatchNorm` の running buffer は対象外）。
//! ホスト `Tensor<f32>` のみを扱い、GPU 固有の数値経路は持たない。

use std::collections::HashMap;

use crate::nn::Module;
use crate::{AutodiffError, Tensor};

/// パラメータの等重み平均（shadow copy）を保持する facade の公開型。
///
/// ```
/// use fandhe_ai::Tensor;
/// use fandhe_ai::optim::AveragedModel;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let w0 = Tensor::new(vec![1.0f32, 2.0], &[2])?;
/// let mut swa = AveragedModel::new(&[&w0])?;
/// let w1 = Tensor::new(vec![3.0f32, 4.0], &[2])?;
/// swa.update(&[&w1])?;
/// // 平均値は初回更新の値の複製ではなく、構築時の値と等重みで平均される。
/// assert_eq!(swa.n_averaged(), 1);
/// assert_eq!(swa.averaged_parameters().len(), 1);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct AveragedModel {
    inner: fandhe_ai_autodiff::nn::AveragedModel,
}

impl AveragedModel {
    /// `params`（位置対応。登録名は `"0"`, `"1"`, …）から構築する。
    ///
    /// # Errors
    /// 内部実装の型付きエラー（[`AutodiffError`]）をそのまま返す。
    pub fn new(params: &[&Tensor<f32>]) -> Result<Self, AutodiffError> {
        Ok(Self {
            inner: fandhe_ai_autodiff::nn::AveragedModel::new(params)?,
        })
    }

    /// 名前付きパラメータ列から構築する（`Sequential::named_parameters()` の戻り値を
    /// そのまま渡せる）。名前重複は [`AutodiffError::InvalidArgument`]。
    pub fn from_named(named: Vec<(String, &Tensor<f32>)>) -> Result<Self, AutodiffError> {
        Ok(Self {
            inner: fandhe_ai_autodiff::nn::AveragedModel::from_named(named)?,
        })
    }

    /// facade の [`Module`] の `named_parameters()` から構築する。
    pub fn from_module(module: &dyn Module) -> Result<Self, AutodiffError> {
        Self::from_named(module.named_parameters())
    }

    /// これまでの `update` 系の成功回数。
    pub fn n_averaged(&self) -> u64 {
        self.inner.n_averaged()
    }

    /// 位置対応で平均を更新する。失敗時は状態を変更しない（two-pass 検証）。
    pub fn update(&mut self, params: &[&Tensor<f32>]) -> Result<(), AutodiffError> {
        self.inner.update(params)
    }

    /// 名前付きで平均を更新する。失敗時は状態を変更しない（two-pass 検証）。
    pub fn update_named(
        &mut self,
        named: Vec<(String, &Tensor<f32>)>,
    ) -> Result<(), AutodiffError> {
        self.inner.update_named(named)
    }

    /// facade の [`Module`] の `named_parameters()` で平均を更新する。
    pub fn update_from_module(&mut self, module: &dyn Module) -> Result<(), AutodiffError> {
        self.update_named(module.named_parameters())
    }

    /// 登録名 `name` の平均パラメータ（無ければ `None`）。
    pub fn averaged(&self, name: &str) -> Option<&Tensor<f32>> {
        self.inner.averaged(name)
    }

    /// 登録順の平均パラメータ参照列。
    pub fn averaged_parameters(&self) -> Vec<&Tensor<f32>> {
        self.inner.averaged_parameters()
    }

    /// 平均パラメータのキー付きコピー（`Module::load_state_dict` へそのまま渡せる）。
    pub fn averaged_state_dict(&self) -> HashMap<String, Tensor<f32>> {
        self.inner.averaged_state_dict()
    }

    /// `model` の重みを平均値へ一時的に差し替え、差し替え前の `state_dict()` を返す
    /// （[`Module::load_state_dict`] のアトミック性契約を継承する）。
    /// 評価後は [`Self::restore`] へ戻り値を渡して元の重みへ戻す。
    pub fn apply(
        &self,
        model: &mut dyn Module,
    ) -> Result<HashMap<String, Tensor<f32>>, AutodiffError> {
        let backup = model.state_dict();
        model.load_state_dict(self.inner.averaged_state_dict())?;
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
