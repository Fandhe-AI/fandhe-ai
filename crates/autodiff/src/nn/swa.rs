//! 等重みの重み平均（SWA の `AveragedModel` 相当。PyTorch
//! `torch.optim.swa_utils.AveragedModel` の既定 `avg_fn`〈等重み平均〉。
//! イシュー #2658・親 #2657）。
//!
//! # 役割・呼び出し文脈
//!
//! 学習の後半で各 epoch（または一定間隔）のパラメータを等重みで平均し、
//! 平坦な極小へ寄せた重み（SWA 重み）を得る。[`AveragedModel`] は平均値の
//! shadow copy を保持するだけの値型で、[`crate::nn::Module`] は実装せず
//! forward も持たない。EMA（[`crate::nn::ExponentialMovingAverage`]。
//! イシュー #2179）と同じ API 形（`new`／`from_named`／`from_module`／
//! `update*`／`apply`／`restore`）にそろえてあり、呼び出し元も同じ 2 系統:
//!
//! - 本クレート内の手動学習ループ（`crates/autodiff/tests/nn_swa.rs`）:
//!   [`crate::nn::Module`] を実装する型に対し
//!   [`AveragedModel::from_module`]／[`AveragedModel::update_from_module`]／
//!   [`AveragedModel::apply`]／[`AveragedModel::restore`] を使う。
//! - facade（`fandhe-ai`）の `compat::Sequential`
//!   （`crates/facade/tests/compat_sequential_swa_manual.rs`）:
//!   `Module` を実装しないため、公開済みの `named_parameters()`／
//!   `state_dict()`／`load_state_dict()` と [`AveragedModel::from_named`]／
//!   [`AveragedModel::update_named`]／[`AveragedModel::averaged_state_dict`]
//!   を結線する。
//!
//! 学習率側（`SWALR` 相当）は `nn/optim/lr_scheduler.rs` の `SwaLr` が担う。
//! facade への公開・`fit` 統合は未承認のため本イシューでは行わない
//! （`docs/autodiff-swa-decision.md` §7。保留の機械的固定は
//! `crates/facade/src/lib.rs::SwaHoldDoctestGuard`）。
//!
//! # EMA との関係
//!
//! EMA は「`decay` 固定の指数平滑」、本型は「`n_averaged` に応じて重みが
//! `1/(n+1)` へ減る等重み平均」であり更新式が異なる。EMA を `avg_fn`
//! 差し替え式へ一般化せず、`ema.rs` は変更しない（名前集合・shape の
//! two-pass 検証は本ファイルに独立して持つ）。
//!
//! # 数値契約
//!
//! `n_averaged == 0` の更新は渡された値の複製（bit 完全一致。NaN payload
//! も保存）。それ以降は `w = 1.0 / ((n_averaged + 1) as f32)` として
//! `avg[i] = f32::mul_add(param[i] - avg[i], w, avg[i])`（lerp 形。
//! `ema.rs`・`RmsProp::step` と同じ `f32::mul_add` の house style。
//! `.claude/rules/coding-rust.md` の CPU 参照実装 FMA 契約に整合）。
//!
//! PyTorch の CPU 既定経路は `avg + (param - avg) / (n + 1)`（除算形）
//! であり、数式は等価だが演算順が異なり丸めが一致しないため **bit 一致は
//! 主張しない**（判定は統一複合判定「相対 1e-3 未満 または 絶対 1e-5
//! 未満」のみ）。`f64` アキュムレータ契約は適用対象外: 1 回の演算内の
//! 長軸縮約ではなく状態を持つ逐次更新であり、PyTorch と同じ `f32` 逐次形に
//! 合わせる。非有限値は特別扱いせず伝播させる。
//!
//! # 検証と原子性
//!
//! パス 1 で名前の重複・欠落・余剰（昇順列挙）・shape 一致・
//! `n_averaged.checked_add(1)` を全件検査し、パス 2 は新テンソルを全件
//! 一時 `Vec` に作り終えてから差し替える。途中で失敗しても状態を一切
//! 変えない。`apply`／`restore` は [`Module::load_state_dict`] へ委譲し
//! その原子性契約を継承する。
//!
//! # スコープ
//!
//! 対象は [`Module::named_parameters`] のみ（BatchNorm の running stats は
//! 対象外。PyTorch `use_buffers=False` 既定と同じ）。`update_bn`・カスタム
//! `avg_fn`／`multi_avg_fn`・`DeviceParamStore` 結線・GPU カーネルは対象外
//! （ホスト `Tensor<f32>` のみ。CUDA／Metal 固有の数値経路を持たない）。

use std::collections::{HashMap, HashSet};

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;
use crate::nn::module::Module;

/// パラメータの等重み平均（shadow copy）を保持する。モジュール doc の
/// 「役割・呼び出し文脈」「数値契約」「スコープ」節を参照。
#[derive(Debug)]
pub struct AveragedModel {
    /// 登録順（`update`／`averaged_parameters` の位置対応で走査する）。
    /// `avg_vars`（`HashMap`）は走査順を保証しないため、順序が意味を持つ
    /// 操作は必ず本フィールド経由で走査する。
    names: Vec<String>,
    avg_vars: HashMap<String, Tensor<f32>>,
    n_averaged: u64,
}

impl AveragedModel {
    /// `params`（位置対応。登録名は `"0"`, `"1"`, … の連番）から構築する。
    /// `n_averaged` は 0 から始まる。
    pub fn new(params: &[&Tensor<f32>]) -> Result<Self, AutodiffError> {
        let named: Vec<(String, &Tensor<f32>)> = params
            .iter()
            .enumerate()
            .map(|(i, tensor)| (i.to_string(), *tensor))
            .collect();
        Self::from_named(named)
    }

    /// 名前付きパラメータ列から構築する（`compat::Sequential::
    /// named_parameters()` 等をそのまま渡せる）。名前の重複は
    /// [`AutodiffError::InvalidArgument`]。
    pub fn from_named(named: Vec<(String, &Tensor<f32>)>) -> Result<Self, AutodiffError> {
        let mut names = Vec::with_capacity(named.len());
        let mut avg_vars = HashMap::with_capacity(named.len());
        for (name, tensor) in named {
            if avg_vars.contains_key(&name) {
                return Err(AutodiffError::InvalidArgument(format!(
                    "AveragedModel::from_named: duplicate parameter name `{name}`"
                )));
            }
            names.push(name.clone());
            avg_vars.insert(name, tensor.clone());
        }
        Ok(AveragedModel {
            names,
            avg_vars,
            n_averaged: 0,
        })
    }

    /// `module.named_parameters()`（[`Module`] trait 経由）から構築する。
    pub fn from_module(module: &dyn Module) -> Result<Self, AutodiffError> {
        Self::from_named(module.named_parameters())
    }

    /// これまでに平均へ取り込んだ回数（PyTorch `n_averaged` 相当）。
    pub fn n_averaged(&self) -> u64 {
        self.n_averaged
    }

    /// 登録順で位置対応する `params` で平均を更新する。要素数が登録時と
    /// 異なる場合は [`AutodiffError::InvalidArgument`]。
    #[doc(alias = "update_parameters")]
    pub fn update(&mut self, params: &[&Tensor<f32>]) -> Result<(), AutodiffError> {
        if params.len() != self.names.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "AveragedModel::update: expected {} parameters (registration order), got {}",
                self.names.len(),
                params.len()
            )));
        }
        let named: Vec<(String, &Tensor<f32>)> = self
            .names
            .iter()
            .cloned()
            .zip(params.iter().copied())
            .collect();
        self.update_named(named)
    }

    /// 名前付きパラメータ列で平均を更新する。名前集合は登録名集合と完全
    /// 一致していなければならない（欠落・余剰は昇順列挙で
    /// [`AutodiffError::InvalidArgument`]、shape 不一致は
    /// [`AutodiffError::Shape`]）。1 件でも検証に失敗したら状態
    /// （平均値・`n_averaged`）を一切変更しない。
    pub fn update_named(
        &mut self,
        named: Vec<(String, &Tensor<f32>)>,
    ) -> Result<(), AutodiffError> {
        // パス 1（検証のみ・無変更）。
        let mut provided: HashMap<String, &Tensor<f32>> = HashMap::with_capacity(named.len());
        for (name, tensor) in named {
            if provided.insert(name.clone(), tensor).is_some() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "AveragedModel::update_named: duplicate parameter name `{name}`"
                )));
            }
        }
        let expected: HashSet<&str> = self.names.iter().map(String::as_str).collect();
        let provided_names: HashSet<&str> = provided.keys().map(String::as_str).collect();

        let mut missing: Vec<&str> = expected.difference(&provided_names).copied().collect();
        missing.sort_unstable();
        if !missing.is_empty() {
            return Err(AutodiffError::InvalidArgument(format!(
                "AveragedModel::update_named: missing keys: {missing:?}"
            )));
        }
        let mut unexpected: Vec<&str> = provided_names.difference(&expected).copied().collect();
        unexpected.sort_unstable();
        if !unexpected.is_empty() {
            return Err(AutodiffError::InvalidArgument(format!(
                "AveragedModel::update_named: unexpected keys: {unexpected:?}"
            )));
        }
        for name in &self.names {
            if let Some(tensor) = provided.get(name.as_str())
                && let Some(avg) = self.avg_vars.get(name.as_str())
                && tensor.shape() != avg.shape()
            {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: avg.shape().to_vec(),
                    rhs: tensor.shape().to_vec(),
                }));
            }
        }
        let next_n = self.n_averaged.checked_add(1).ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "AveragedModel::update_named: n_averaged counter overflow".to_string(),
            )
        })?;

        // パス 2（適用）。新テンソルを全件作り終えてから差し替える。
        let first = self.n_averaged == 0;
        // `n_averaged` は u64。`as f32` は 2^24 超で丸まるが、`w` は 1/(n+1) の
        // 近似値でありその領域では更新量が f32 の分解能を下回る。
        let w = 1.0f32 / ((self.n_averaged as f32) + 1.0);
        let mut staged: Vec<(&String, Tensor<f32>)> = Vec::with_capacity(self.names.len());
        for name in &self.names {
            let (Some(tensor), Some(avg)) = (
                provided.get(name.as_str()),
                self.avg_vars.get(name.as_str()),
            ) else {
                continue;
            };
            let avg_data = dense_vec_ref(avg);
            let param_data = dense_vec_ref(tensor);
            let new_avg: Vec<f32> = if first {
                param_data.iter().copied().collect()
            } else {
                (0..avg_data.len())
                    .map(|i| f32::mul_add(param_data[i] - avg_data[i], w, avg_data[i]))
                    .collect()
            };
            let shape = avg.shape().to_vec();
            drop(avg_data);
            drop(param_data);
            staged.push((name, Tensor::new(new_avg, &shape)?));
        }
        for (name, tensor) in staged {
            self.avg_vars.insert(name.clone(), tensor);
        }
        self.n_averaged = next_n;
        Ok(())
    }

    /// `module.named_parameters()`（[`Module`] trait 経由）で平均を更新する。
    pub fn update_from_module(&mut self, module: &dyn Module) -> Result<(), AutodiffError> {
        self.update_named(module.named_parameters())
    }

    /// 名前 `name` に対応する平均パラメータへの参照。未登録名は `None`。
    pub fn averaged(&self, name: &str) -> Option<&Tensor<f32>> {
        self.avg_vars.get(name)
    }

    /// 登録順で並べた平均パラメータの参照列。
    pub fn averaged_parameters(&self) -> Vec<&Tensor<f32>> {
        self.names
            .iter()
            .filter_map(|name| self.avg_vars.get(name.as_str()))
            .collect()
    }

    /// 平均パラメータの `{名前: 値}` マップ（[`Module::state_dict`] と同じ
    /// キー形式）。各値は `clone` される。
    pub fn averaged_state_dict(&self) -> HashMap<String, Tensor<f32>> {
        self.avg_vars.clone()
    }

    /// `model` の現在の重みを [`Self::averaged_state_dict`] へ一時的に
    /// 差し替える（[`Module::load_state_dict`] へ委譲し、その原子性契約を
    /// 継承する）。戻り値は差し替え前の `model.state_dict()`
    /// （[`Self::restore`] へ渡す退避値）。
    pub fn apply(
        &self,
        model: &mut dyn Module,
    ) -> Result<HashMap<String, Tensor<f32>>, AutodiffError> {
        let backup = model.state_dict();
        model.load_state_dict(self.averaged_state_dict())?;
        Ok(backup)
    }

    /// [`Self::apply`] が返した退避値 `backup` で `model` を元の重みへ戻す。
    pub fn restore(
        model: &mut dyn Module,
        backup: HashMap<String, Tensor<f32>>,
    ) -> Result<(), AutodiffError> {
        model.load_state_dict(backup)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
    }

    fn vals(avg: &AveragedModel, name: &str) -> Vec<f32> {
        avg.averaged(name).unwrap().as_slice().unwrap().to_vec()
    }

    #[test]
    fn new_clones_initial_params_and_starts_at_zero() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let p1 = t(vec![3.0], &[1]);
        let avg = AveragedModel::new(&[&p0, &p1]).unwrap();
        assert_eq!(vals(&avg, "0"), vec![1.0, 2.0]);
        assert_eq!(vals(&avg, "1"), vec![3.0]);
        assert_eq!(avg.n_averaged(), 0);
        assert_eq!(avg.averaged_parameters().len(), 2);
    }

    #[test]
    fn first_update_copies_param_bit_exact() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let mut avg = AveragedModel::new(&[&p0]).unwrap();
        let nan_payload = f32::from_bits(0x7fc0_1234);
        let p = t(vec![-0.0, nan_payload], &[2]);
        avg.update(&[&p]).unwrap();
        let got = vals(&avg, "0");
        assert_eq!(got[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(got[1].to_bits(), nan_payload.to_bits());
        assert_eq!(avg.n_averaged(), 1);
    }

    #[test]
    fn later_updates_match_closed_form_bit_exact() {
        let p0 = t(vec![0.0], &[1]);
        let mut avg = AveragedModel::new(&[&p0]).unwrap();
        let mut expected = 0.0f32;
        for step in 1..=5u32 {
            let param = t(vec![step as f32 * 0.7], &[1]);
            avg.update(&[&param]).unwrap();
            if step == 1 {
                expected = step as f32 * 0.7;
            } else {
                let w = 1.0f32 / (step as f32);
                expected = f32::mul_add(step as f32 * 0.7 - expected, w, expected);
            }
            assert_eq!(vals(&avg, "0"), vec![expected]);
        }
        assert_eq!(avg.n_averaged(), 5);
    }

    #[test]
    fn constant_input_keeps_average_unchanged() {
        let p0 = t(vec![0.3, -1.25], &[2]);
        let mut avg = AveragedModel::new(&[&p0]).unwrap();
        let p = t(vec![0.5, 2.0], &[2]);
        for _ in 0..8 {
            avg.update(&[&p]).unwrap();
        }
        assert_eq!(vals(&avg, "0"), vec![0.5, 2.0]);
    }

    #[test]
    fn non_finite_values_propagate() {
        let p0 = t(vec![1.0, 1.0], &[2]);
        let mut avg = AveragedModel::new(&[&p0]).unwrap();
        avg.update(&[&t(vec![1.0, 2.0], &[2])]).unwrap();
        avg.update(&[&t(vec![f32::INFINITY, f32::NAN], &[2])])
            .unwrap();
        let got = vals(&avg, "0");
        assert!(got[0].is_infinite());
        assert!(got[1].is_nan());
    }

    #[test]
    fn invalid_updates_leave_state_unchanged() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let p1 = t(vec![3.0], &[1]);
        let mut avg = AveragedModel::new(&[&p0, &p1]).unwrap();
        avg.update(&[&p0, &p1]).unwrap();
        let before0 = vals(&avg, "0");
        let before1 = vals(&avg, "1");

        // 要素数不一致
        assert!(avg.update(&[&p0]).is_err());
        // 余剰・欠落・重複
        let other = t(vec![9.0, 9.0], &[2]);
        assert!(
            avg.update_named(vec![
                ("0".into(), &other),
                ("1".into(), &p1),
                ("2".into(), &p1)
            ])
            .is_err()
        );
        assert!(avg.update_named(vec![("0".into(), &other)]).is_err());
        assert!(
            avg.update_named(vec![("0".into(), &other), ("0".into(), &other)])
                .is_err()
        );
        // shape 不一致（"0" は有効だが "1" が不正 → 部分更新されない）
        let bad = t(vec![1.0, 2.0], &[2]);
        let err = avg
            .update_named(vec![("0".into(), &other), ("1".into(), &bad)])
            .unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));

        assert_eq!(vals(&avg, "0"), before0);
        assert_eq!(vals(&avg, "1"), before1);
        assert_eq!(avg.n_averaged(), 1);
    }

    #[test]
    fn from_named_rejects_duplicate_names() {
        let p = t(vec![1.0], &[1]);
        assert!(AveragedModel::from_named(vec![("a".into(), &p), ("a".into(), &p)]).is_err());
    }

    #[test]
    fn empty_parameter_list_is_accepted() {
        let mut avg = AveragedModel::new(&[]).unwrap();
        avg.update(&[]).unwrap();
        assert_eq!(avg.n_averaged(), 1);
        assert!(avg.averaged_parameters().is_empty());
    }
}
