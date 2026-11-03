//! 指数移動平均（EMA。PyTorch `torch.optim.swa_utils.AveragedModel`
//! （`avg_fn` に EMA 式を渡した用法）／Keras 3
//! `EMAOverlay`（`ema_momentum`）相当。イシュー #2179・親 #2131
//! 「PyTorch／TF 置き換えの API 網羅」）。
//!
//! # 役割・呼び出し文脈
//!
//! 学習中にパラメータの複製（shadow copy）を保持し、各 step 後に
//! [`ExponentialMovingAverage::update`]（または名前付き版
//! [`ExponentialMovingAverage::update_named`]／
//! [`ExponentialMovingAverage::update_from_module`]）で shadow を
//! 指数平滑更新する。推論・評価時は [`ExponentialMovingAverage::apply`]
//! で shadow 重みへ一時的に差し替え、終わったら
//! [`ExponentialMovingAverage::restore`] で退避しておいた元の重みへ
//! 戻す。呼び出し元は主に 2 系統:
//!
//! - 本クレート内の手動学習ループ（`crates/autodiff/tests/nn_ema.rs`）:
//!   [`crate::nn::Module`] trait を実装する型（本クレート内 `nn::
//!   Sequential` を含む）に対し [`ExponentialMovingAverage::from_module`]／
//!   [`ExponentialMovingAverage::update_from_module`]／
//!   [`ExponentialMovingAverage::apply`]／[`ExponentialMovingAverage::
//!   restore`] を直接使う。
//! - facade（`fandhe-ai` crates.io 公開クレート）の `compat::Sequential`
//!   （`crates/facade/src/compat/sequential.rs`）: `compat::Sequential` は
//!   本クレートの [`Module`] trait を実装しないため `&mut dyn Module` を
//!   渡せない。代わりに公開済みの `named_parameters()`／
//!   `state_dict()`／`load_state_dict()` と
//!   [`ExponentialMovingAverage::from_named`]／[`ExponentialMovingAverage::
//!   update_named`]／[`ExponentialMovingAverage::shadow_state_dict`] を
//!   結線して同等の効果を得る（`crates/facade/tests/
//!   compat_sequential_ema_manual.rs`）。
//!
//! facade への公開はイシュー #2560（親 #2558）で実施済み: 内部 `nn::Module` を
//! 露出させないため素の再エクスポートではなく facade 独自の薄いラッパー
//! `fandhe_ai::optim::ExponentialMovingAverage`（`crates/facade/src/optim_ema.rs`）
//! として公開し、`fit` への結線は `compat::Callback::Ema`（`FitConfig` へは
//! フィールドを足さない）が担う（`docs/autodiff-ema-decision.md` §10.2・§13）。
//!
//! # 数値契約
//!
//! 更新式は `shadow[i] = decay * shadow[i] + (1 - decay) * param[i]`
//! （`f32::mul_add(decay, shadow[i], one_minus_decay * param[i])`。
//! `nn/optim/rmsprop.rs::RmsProp::step` の
//! `f32::mul_add(alpha, square_avg, one_minus_alpha * g * g)` と同じ
//! house style。`.claude/rules/coding-rust.md` の CPU 参照実装 FMA 契約
//! 〈`f32::mul_add`〉に整合）。Keras 3 `ema_momentum * average +
//! (1 - ema_momentum) * var` と同型だが、**PyTorch
//! `AveragedModel`（既定の `avg_fn` は `lerp` 形
//! `averaged_param + (1 - decay) * (param - averaged_param)`）とは
//! 丸めが異なるため bit 一致は主張しない**（数式としては等価だが
//! 浮動小数演算順序が異なり、有限精度では一致しない）。
//!
//! 非有限値（`NaN`／`inf`）は特別扱いせず伝播させる。`decay == 0.0`／
//! `decay == 1.0` の境界であっても、相手側（`param[i]` または
//! `shadow[i]`）が非有限なら `0.0 * inf = NaN` 等の IEEE 754 規則が
//! そのまま適用される。
//!
//! 初期化は構築時パラメータの `clone`（Keras／TF ExponentialMovingAverage
//! と同じ）。`num_updates`（[`ExponentialMovingAverage::num_updates`]）は
//! 呼び出し回数のカウンタのみで、TensorFlow の `num_updates` による
//! decay ウォームアップ（`min(decay, (1 + n) / (10 + n))`）は opt-in
//! （既定 OFF）の [`ExponentialMovingAverage::with_warmup`] として
//! 提供する（イシュー #3833。`docs/autodiff-ema-decision.md` §18・§20）。
//! n は更新の**前**の `num_updates`（初回は 0）で、実効 decay は
//! `min(decay, ((1 + n) as f64 / (10 + n) as f64) as f32)`。OFF の経路は
//! 従来と bit 単位で同一（実効 decay が `self.decay` そのもの）。
//!
//! # スコープ
//!
//! 対象は [`Module::named_parameters`]（学習可能パラメータ）のみ。
//! `BatchNorm1d`／`BatchNorm2d` の `running_mean`／`running_var` は
//! `RefCell` 越しに保持される buffer であり `named_parameters()` には
//! 含まれないため EMA の対象外（PyTorch
//! `AveragedModel(use_buffers=False)` 既定・Keras（trainable variables
//! のみ）と同じ）。デバイス常駐経路（`DeviceParamStore`）への結線・
//! GPU カーネルは対象外（ホスト `Tensor<f32>` のみを扱う。CUDA／Metal
//! 固有の数値経路を持たないため実機 parity テストは不要）。SWA
//! （Stochastic Weight Averaging）はスコープ外。

use std::collections::{HashMap, HashSet};

use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::eval::dense_vec_ref;
use crate::nn::module::Module;

/// `decay` が有限かつ `[0.0, 1.0]` に収まっているかを検証する
/// （[`ExponentialMovingAverage::new`]／[`ExponentialMovingAverage::
/// from_named`] 共通の入口）。
fn validate_decay(decay: f32) -> Result<(), AutodiffError> {
    if !(decay.is_finite() && (0.0..=1.0).contains(&decay)) {
        return Err(AutodiffError::InvalidArgument(format!(
            "ExponentialMovingAverage: decay must be finite and in [0.0, 1.0], got {decay}"
        )));
    }
    Ok(())
}

/// パラメータの指数移動平均（shadow copy）を保持する。型 doc（モジュール
/// doc）の「役割・呼び出し文脈」「数値契約」「スコープ」節を参照。
#[derive(Debug)]
pub struct ExponentialMovingAverage {
    decay: f32,
    /// 登録順（`update`／`shadow_parameters` の位置対応で走査する）。
    /// `shadow_vars`（`HashMap`）は走査順を保証しないため、順序が
    /// 意味を持つ操作は必ず本フィールド経由で走査する。
    names: Vec<String>,
    shadow_vars: HashMap<String, Tensor<f32>>,
    num_updates: u64,
    /// 減衰率ウォームアップの有効フラグ（既定 `false`。イシュー #3833）。
    warmup: bool,
}

impl ExponentialMovingAverage {
    /// `params`（位置対応。登録名は `"0"`, `"1"`, … の連番）から構築
    /// する。`update` はこの登録順で位置対応する（Issue #2179 の
    /// 受け入れ基準が想定する `update(&mut self, params: &[&Tensor<f32>])`
    /// と対応させるための最小構成）。
    pub fn new(decay: f32, params: &[&Tensor<f32>]) -> Result<Self, AutodiffError> {
        let named: Vec<(String, &Tensor<f32>)> = params
            .iter()
            .enumerate()
            .map(|(i, tensor)| (i.to_string(), *tensor))
            .collect();
        Self::from_named(decay, named)
    }

    /// 名前付きパラメータ列から構築する（`compat::Sequential::
    /// named_parameters()` 等の `Vec<(String, &Tensor<f32>)>` をそのまま
    /// 渡せる）。名前の重複は [`AutodiffError::InvalidArgument`]。
    pub fn from_named(
        decay: f32,
        named: Vec<(String, &Tensor<f32>)>,
    ) -> Result<Self, AutodiffError> {
        validate_decay(decay)?;
        let mut names = Vec::with_capacity(named.len());
        let mut shadow_vars = HashMap::with_capacity(named.len());
        for (name, tensor) in named {
            if shadow_vars.contains_key(&name) {
                return Err(AutodiffError::InvalidArgument(format!(
                    "ExponentialMovingAverage::from_named: duplicate parameter name `{name}`"
                )));
            }
            names.push(name.clone());
            shadow_vars.insert(name, tensor.clone());
        }
        Ok(ExponentialMovingAverage {
            decay,
            names,
            shadow_vars,
            num_updates: 0,
            warmup: false,
        })
    }

    /// 減衰率ウォームアップ（TensorFlow 方式）の有効・無効を設定する
    /// （既定 OFF。イシュー #3833、決定記録 §18・§20）。
    ///
    /// ON のとき、更新 1 回ごとの実効 decay は
    /// `min(decay, (1 + n) / (10 + n))`（n は更新前の `num_updates`、
    /// 初回は 0 なので `min(decay, 0.1)`）。比は f64 で 1 回割り f32 へ
    /// 1 回だけ丸める。`decay <= 0.1` では常に OFF と同じ。`decay == 1.0`
    /// は ON にすると shadow が動く。途中で切り替えた場合は次の更新から
    /// 効き、n は続く。OFF の経路は従来と bit 同一。
    pub fn with_warmup(mut self, enabled: bool) -> Self {
        self.warmup = enabled;
        self
    }

    /// 減衰率ウォームアップが有効か。
    pub fn warmup(&self) -> bool {
        self.warmup
    }

    /// 今回の更新で使う実効 decay。OFF では `self.decay` をそのまま返し
    /// 比の計算も `min` も走らない（既定経路の bit 同一性を守る）。
    fn effective_decay(&self) -> f32 {
        if !self.warmup {
            return self.decay;
        }
        let n = self.num_updates;
        let ratio = (n.saturating_add(1) as f64 / n.saturating_add(10) as f64) as f32;
        self.decay.min(ratio)
    }

    /// `module.named_parameters()`（[`Module`] trait 経由）から構築する。
    /// 本クレート内の手動学習ループ向け（モジュール doc「役割・呼び出し
    /// 文脈」節参照）。
    pub fn from_module(decay: f32, module: &dyn Module) -> Result<Self, AutodiffError> {
        Self::from_named(decay, module.named_parameters())
    }

    /// 構築時に検証済みの `decay`。
    pub fn decay(&self) -> f32 {
        self.decay
    }

    /// 実行済み `update`／`update_named`／`update_from_module` の
    /// 合計回数（減衰率ウォームアップの n はこの値）。
    pub fn num_updates(&self) -> u64 {
        self.num_updates
    }

    /// 登録順（構築時の順序）で位置対応する `params` で shadow を更新
    /// する。要素数が登録時と異なる場合は [`AutodiffError::
    /// InvalidArgument`]。
    pub fn update(&mut self, params: &[&Tensor<f32>]) -> Result<(), AutodiffError> {
        if params.len() != self.names.len() {
            return Err(AutodiffError::InvalidArgument(format!(
                "ExponentialMovingAverage::update: expected {} parameters (registration \
                 order), got {}",
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

    /// 名前付きパラメータ列で shadow を更新する。名前集合は構築時の
    /// 登録名集合と完全一致していなければならない（欠落・余剰キーは
    /// 昇順で列挙し [`AutodiffError::InvalidArgument`]。[`Module::
    /// load_state_dict`] と同型の two-pass 検証: shape 不一致を含め
    /// 1 件でも検証に失敗したら shadow を一切変更しない）。
    pub fn update_named(
        &mut self,
        named: Vec<(String, &Tensor<f32>)>,
    ) -> Result<(), AutodiffError> {
        // パス 1（検証のみ・無変更）。
        let mut provided: HashMap<String, &Tensor<f32>> = HashMap::with_capacity(named.len());
        for (name, tensor) in named {
            if provided.insert(name.clone(), tensor).is_some() {
                return Err(AutodiffError::InvalidArgument(format!(
                    "ExponentialMovingAverage::update_named: duplicate parameter name `{name}`"
                )));
            }
        }
        let expected: HashSet<&str> = self.names.iter().map(String::as_str).collect();
        let provided_names: HashSet<&str> = provided.keys().map(String::as_str).collect();

        let mut missing: Vec<&str> = expected.difference(&provided_names).copied().collect();
        missing.sort_unstable();
        if !missing.is_empty() {
            return Err(AutodiffError::InvalidArgument(format!(
                "ExponentialMovingAverage::update_named: missing keys: {missing:?}"
            )));
        }
        let mut unexpected: Vec<&str> = provided_names.difference(&expected).copied().collect();
        unexpected.sort_unstable();
        if !unexpected.is_empty() {
            return Err(AutodiffError::InvalidArgument(format!(
                "ExponentialMovingAverage::update_named: unexpected keys: {unexpected:?}"
            )));
        }
        for name in &self.names {
            // 直上の集合検査でキーは必ず存在する。
            if let Some(tensor) = provided.get(name.as_str())
                && let Some(shadow) = self.shadow_vars.get(name.as_str())
                && tensor.shape() != shadow.shape()
            {
                return Err(AutodiffError::Shape(ShapeError::ShapeMismatch {
                    lhs: shadow.shape().to_vec(),
                    rhs: tensor.shape().to_vec(),
                }));
            }
        }

        // パス 2（適用）。`shadow[i] = decay*shadow[i] +
        // (1-decay)*param[i]`（モジュール doc「数値契約」節参照）。
        let decay = self.effective_decay();
        let one_minus_decay = 1.0 - decay;
        for name in &self.names {
            let Some(tensor) = provided.get(name.as_str()) else {
                continue;
            };
            let Some(shadow) = self.shadow_vars.get(name.as_str()) else {
                continue;
            };
            let shadow_data = dense_vec_ref(shadow);
            let param_data = dense_vec_ref(tensor);
            let mut new_shadow = Vec::with_capacity(shadow_data.len());
            for i in 0..shadow_data.len() {
                new_shadow.push(f32::mul_add(
                    decay,
                    shadow_data[i],
                    one_minus_decay * param_data[i],
                ));
            }
            let shape = shadow.shape().to_vec();
            drop(shadow_data);
            drop(param_data);
            let new_tensor = Tensor::new(new_shadow, &shape)?;
            self.shadow_vars.insert(name.clone(), new_tensor);
        }
        self.num_updates += 1;
        Ok(())
    }

    /// `module.named_parameters()`（[`Module`] trait 経由）で shadow を
    /// 更新する（[`Self::from_module`] と対の呼び出し）。
    pub fn update_from_module(&mut self, module: &dyn Module) -> Result<(), AutodiffError> {
        self.update_named(module.named_parameters())
    }

    /// 名前 `name` に対応する shadow パラメータへの参照。未登録名は
    /// `None`。
    pub fn shadow(&self, name: &str) -> Option<&Tensor<f32>> {
        self.shadow_vars.get(name)
    }

    /// 登録順（構築時の順序）で並べた shadow パラメータの参照列。
    /// [`crate::nn::optim`] の optimizer `step()` 等、位置対応の API へ
    /// そのまま渡せる。
    pub fn shadow_parameters(&self) -> Vec<&Tensor<f32>> {
        self.names
            .iter()
            .filter_map(|name| self.shadow_vars.get(name.as_str()))
            .collect()
    }

    /// shadow パラメータの `{名前: 値}` マップ（[`Module::state_dict`]
    /// と同じキー形式。[`Module::load_state_dict`]／`compat::Sequential::
    /// load_state_dict` へそのまま渡せる）。各値は `clone` される。
    pub fn shadow_state_dict(&self) -> HashMap<String, Tensor<f32>> {
        self.shadow_vars.clone()
    }

    /// `model` の現在の重みを [`Self::shadow_state_dict`] へ一時的に
    /// 差し替える（[`Module::load_state_dict`] へ委譲するため、その
    /// アトミック性契約〈two-pass 検証＋ベストエフォート・
    /// ロールバック〉をそのまま継承する）。戻り値は差し替え前の
    /// `model.state_dict()`（[`Self::restore`] へ渡す退避値）。
    pub fn apply(
        &self,
        model: &mut dyn Module,
    ) -> Result<HashMap<String, Tensor<f32>>, AutodiffError> {
        let backup = model.state_dict();
        model.load_state_dict(self.shadow_state_dict())?;
        Ok(backup)
    }

    /// [`Self::apply`] が返した退避値 `backup` で `model` を元の重みへ
    /// 戻す（[`Module::load_state_dict`] への委譲）。
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

    #[test]
    fn new_clones_initial_params_as_shadow() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let p1 = t(vec![3.0], &[1]);
        let ema = ExponentialMovingAverage::new(0.9, &[&p0, &p1]).unwrap();
        assert_eq!(
            ema.shadow("0").unwrap().as_slice().unwrap().to_vec(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            ema.shadow("1").unwrap().as_slice().unwrap().to_vec(),
            vec![3.0]
        );
        assert_eq!(ema.num_updates(), 0);
    }

    #[test]
    fn update_matches_closed_form_mul_add_bit_exact() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let mut ema = ExponentialMovingAverage::new(0.9, &[&p0]).unwrap();

        let new_p0 = t(vec![2.0, 4.0], &[2]);
        ema.update(&[&new_p0]).unwrap();

        let expected: Vec<f32> = [1.0f32, 2.0]
            .iter()
            .zip([2.0f32, 4.0].iter())
            .map(|(&s, &p)| f32::mul_add(0.9, s, 0.1 * p))
            .collect();
        assert_eq!(
            ema.shadow("0").unwrap().as_slice().unwrap().to_vec(),
            expected
        );
        assert_eq!(ema.num_updates(), 1);
    }

    #[test]
    fn update_sequence_matches_closed_form_bit_exact() {
        let p0 = t(vec![0.0], &[1]);
        let mut ema = ExponentialMovingAverage::new(0.5, &[&p0]).unwrap();
        let mut expected = 0.0f32;
        for step in 1..=5u32 {
            let param = t(vec![step as f32], &[1]);
            ema.update(&[&param]).unwrap();
            expected = f32::mul_add(0.5, expected, 0.5 * step as f32);
            assert_eq!(
                ema.shadow("0").unwrap().as_slice().unwrap().to_vec(),
                vec![expected]
            );
        }
        assert_eq!(ema.num_updates(), 5);
    }

    #[test]
    fn decay_zero_shadow_becomes_latest_param() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let mut ema = ExponentialMovingAverage::new(0.0, &[&p0]).unwrap();
        let new_p0 = t(vec![9.0, -9.0], &[2]);
        ema.update(&[&new_p0]).unwrap();
        assert_eq!(
            ema.shadow("0").unwrap().as_slice().unwrap().to_vec(),
            vec![9.0, -9.0]
        );
    }

    #[test]
    fn decay_one_shadow_is_unchanged() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let mut ema = ExponentialMovingAverage::new(1.0, &[&p0]).unwrap();
        let new_p0 = t(vec![9.0, -9.0], &[2]);
        ema.update(&[&new_p0]).unwrap();
        assert_eq!(
            ema.shadow("0").unwrap().as_slice().unwrap().to_vec(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn new_rejects_invalid_decay() {
        let p0 = t(vec![1.0], &[1]);
        for bad in [-0.1f32, 1.1, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = ExponentialMovingAverage::new(bad, &[&p0]).unwrap_err();
            assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        }
    }

    #[test]
    fn from_named_rejects_duplicate_names() {
        let p0 = t(vec![1.0], &[1]);
        let p1 = t(vec![2.0], &[1]);
        let err =
            ExponentialMovingAverage::from_named(0.9, vec![("w".into(), &p0), ("w".into(), &p1)])
                .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn update_rejects_element_count_mismatch() {
        let p0 = t(vec![1.0], &[1]);
        let p1 = t(vec![2.0], &[1]);
        let mut ema = ExponentialMovingAverage::new(0.9, &[&p0]).unwrap();
        let err = ema.update(&[&p0, &p1]).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        // 検証失敗時は shadow が変化しない（two-pass）。
        assert_eq!(
            ema.shadow("0").unwrap().as_slice().unwrap().to_vec(),
            vec![1.0]
        );
        assert_eq!(ema.num_updates(), 0);
    }

    #[test]
    fn update_named_rejects_shape_mismatch_without_mutating_shadow() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let mut ema = ExponentialMovingAverage::new(0.9, &[&p0]).unwrap();
        let bad = t(vec![1.0, 2.0, 3.0], &[3]);
        let err = ema.update_named(vec![("0".to_string(), &bad)]).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ShapeMismatch { .. })
        ));
        assert_eq!(
            ema.shadow("0").unwrap().as_slice().unwrap().to_vec(),
            vec![1.0, 2.0]
        );
        assert_eq!(ema.num_updates(), 0);
    }

    #[test]
    fn update_named_rejects_missing_and_unexpected_keys() {
        let p0 = t(vec![1.0], &[1]);
        let p1 = t(vec![2.0], &[1]);
        let mut ema = ExponentialMovingAverage::from_named(
            0.9,
            vec![("a".to_string(), &p0), ("b".to_string(), &p1)],
        )
        .unwrap();

        let new_p0 = t(vec![10.0], &[1]);
        let err = ema
            .update_named(vec![("a".to_string(), &new_p0)])
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert_eq!(
            ema.shadow("a").unwrap().as_slice().unwrap().to_vec(),
            vec![1.0]
        );

        let stray = t(vec![0.0], &[1]);
        let err = ema
            .update_named(vec![
                ("a".to_string(), &new_p0),
                ("b".to_string(), &p1),
                ("c".to_string(), &stray),
            ])
            .unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
        assert_eq!(
            ema.shadow("a").unwrap().as_slice().unwrap().to_vec(),
            vec![1.0]
        );
        assert_eq!(ema.num_updates(), 0);
    }

    #[test]
    fn shadow_parameters_and_state_dict_follow_registration_order() {
        let p0 = t(vec![1.0], &[1]);
        let p1 = t(vec![2.0], &[1]);
        let ema = ExponentialMovingAverage::from_named(
            0.9,
            vec![("second".to_string(), &p1), ("first".to_string(), &p0)],
        )
        .unwrap();
        let params = ema.shadow_parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].as_slice().unwrap().to_vec(), vec![2.0]);
        assert_eq!(params[1].as_slice().unwrap().to_vec(), vec![1.0]);

        let dict = ema.shadow_state_dict();
        assert_eq!(dict.len(), 2);
        assert_eq!(dict["first"].as_slice().unwrap().to_vec(), vec![1.0]);
        assert_eq!(dict["second"].as_slice().unwrap().to_vec(), vec![2.0]);
    }

    #[test]
    fn empty_params_is_a_no_op() {
        let mut ema = ExponentialMovingAverage::new(0.9, &[]).unwrap();
        ema.update(&[]).unwrap();
        assert_eq!(ema.num_updates(), 1);
        assert!(ema.shadow_parameters().is_empty());
    }

    #[test]
    fn deterministic_repeated_runs_are_bit_identical() {
        let p0 = t(vec![1.0, -2.5, 3.75], &[3]);
        let updates: Vec<Tensor<f32>> = (0..4)
            .map(|i| t(vec![i as f32, -i as f32, i as f32 * 0.5], &[3]))
            .collect();

        let run = |p0: &Tensor<f32>, updates: &[Tensor<f32>]| -> Vec<f32> {
            let mut ema = ExponentialMovingAverage::new(0.8, &[p0]).unwrap();
            for u in updates {
                ema.update(&[u]).unwrap();
            }
            ema.shadow("0").unwrap().as_slice().unwrap().to_vec()
        };

        assert_eq!(run(&p0, &updates), run(&p0, &updates));
    }
    // --- 減衰率ウォームアップ（イシュー #3833）。判定はすべて bit 一致。 ---

    fn bits_of(ema: &ExponentialMovingAverage, name: &str) -> Vec<u32> {
        ema.shadow(name)
            .unwrap()
            .as_slice()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// 独立に計算した期待の実効 decay。
    fn expected_decay(decay: f32, n: u64) -> f32 {
        let r = ((1 + n) as f64 / (10 + n) as f64) as f32;
        decay.min(r)
    }

    #[test]
    fn warmup_off_is_bit_identical_to_default() {
        let p0 = t(vec![1.0, -2.0, 0.5], &[3]);
        let mut a = ExponentialMovingAverage::new(0.9, &[&p0])
            .unwrap()
            .with_warmup(false);
        let mut b = ExponentialMovingAverage::new(0.9, &[&p0]).unwrap();
        for i in 0..6 {
            let u = t(vec![i as f32, 1.5 * i as f32, -0.25], &[3]);
            a.update(&[&u]).unwrap();
            b.update(&[&u]).unwrap();
            assert_eq!(bits_of(&a, "0"), bits_of(&b, "0"));
        }
        assert!(!a.warmup() && !b.warmup());
    }

    #[test]
    fn warmup_first_update_uses_n_zero() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let mut ema = ExponentialMovingAverage::new(0.999, &[&p0])
            .unwrap()
            .with_warmup(true);
        let u = t(vec![3.0, 5.0], &[2]);
        ema.update(&[&u]).unwrap();
        let eff = 0.1f32;
        assert_eq!(expected_decay(0.999, 0), eff);
        let expected: Vec<u32> = [(1.0f32, 3.0f32), (2.0, 5.0)]
            .iter()
            .map(|&(s, p)| f32::mul_add(eff, s, (1.0 - eff) * p).to_bits())
            .collect();
        assert_eq!(bits_of(&ema, "0"), expected);
    }

    #[test]
    fn warmup_sequence_matches_closed_form_bit_exact() {
        let p0 = t(vec![0.5], &[1]);
        let mut ema = ExponentialMovingAverage::new(0.999, &[&p0])
            .unwrap()
            .with_warmup(true);
        let mut expected = 0.5f32;
        for n in 0..10u64 {
            let param = (n + 1) as f32;
            ema.update(&[&t(vec![param], &[1])]).unwrap();
            let eff = expected_decay(0.999, n);
            expected = f32::mul_add(eff, expected, (1.0 - eff) * param);
            assert_eq!(bits_of(&ema, "0"), vec![expected.to_bits()]);
        }
        assert_eq!(ema.num_updates(), 10);
    }

    #[test]
    fn warmup_reaches_configured_decay() {
        // decay 0.5: n = 8 で 9/18 = 0.5 に達し、以降の実効 decay は OFF と同じ。
        let p0 = t(vec![0.0], &[1]);
        let mut on = ExponentialMovingAverage::new(0.5, &[&p0])
            .unwrap()
            .with_warmup(true);
        let mut off = ExponentialMovingAverage::new(0.5, &[&p0]).unwrap();
        for _ in 0..8 {
            let u = t(vec![1.0], &[1]);
            on.update(&[&u]).unwrap();
            off.update(&[&u]).unwrap();
        }
        assert_eq!(expected_decay(0.5, 8), 0.5);
        let u = t(vec![2.0], &[1]);
        for ema in [&mut on, &mut off] {
            let s = ema.shadow("0").unwrap().as_slice().unwrap()[0];
            ema.update(&[&u]).unwrap();
            let e = f32::mul_add(0.5, s, 0.5 * 2.0);
            assert_eq!(bits_of(ema, "0"), vec![e.to_bits()]);
        }
    }

    #[test]
    fn warmup_boundaries_decay_zero_small_and_one() {
        let p0 = t(vec![1.0], &[1]);
        let u = t(vec![9.0], &[1]);
        // decay <= 0.1 は常に OFF と同じ。
        for d in [0.0f32, 0.05, 0.1] {
            let mut on = ExponentialMovingAverage::new(d, &[&p0])
                .unwrap()
                .with_warmup(true);
            let mut off = ExponentialMovingAverage::new(d, &[&p0]).unwrap();
            for _ in 0..5 {
                on.update(&[&u]).unwrap();
                off.update(&[&u]).unwrap();
                assert_eq!(bits_of(&on, "0"), bits_of(&off, "0"));
            }
        }
        // decay == 0.0 は最新の param になる。
        let mut zero = ExponentialMovingAverage::new(0.0, &[&p0])
            .unwrap()
            .with_warmup(true);
        zero.update(&[&u]).unwrap();
        assert_eq!(bits_of(&zero, "0"), vec![9.0f32.to_bits()]);
        // decay == 1.0: OFF では動かず、ON では shadow が動く。
        let mut one_off = ExponentialMovingAverage::new(1.0, &[&p0]).unwrap();
        let mut one_on = ExponentialMovingAverage::new(1.0, &[&p0])
            .unwrap()
            .with_warmup(true);
        one_off.update(&[&u]).unwrap();
        one_on.update(&[&u]).unwrap();
        assert_eq!(bits_of(&one_off, "0"), vec![1.0f32.to_bits()]);
        assert_ne!(bits_of(&one_on, "0"), vec![1.0f32.to_bits()]);
    }

    #[test]
    fn warmup_rejected_update_keeps_counter_and_shadow() {
        let p0 = t(vec![1.0, 2.0], &[2]);
        let mut ema = ExponentialMovingAverage::new(0.999, &[&p0])
            .unwrap()
            .with_warmup(true);
        ema.update(&[&t(vec![3.0, 4.0], &[2])]).unwrap();
        let before = bits_of(&ema, "0");
        assert!(ema.update(&[&t(vec![1.0], &[1])]).is_err());
        assert!(ema.update(&[]).is_err());
        assert_eq!(ema.num_updates(), 1);
        assert_eq!(bits_of(&ema, "0"), before);
        // 次に成功する更新は n = 1 を使う。
        let u = t(vec![5.0, 6.0], &[2]);
        ema.update(&[&u]).unwrap();
        let eff = expected_decay(0.999, 1);
        let s = f32::from_bits(before[0]);
        let expected = f32::mul_add(eff, s, (1.0 - eff) * 5.0);
        assert_eq!(bits_of(&ema, "0")[0], expected.to_bits());
    }

    #[test]
    fn warmup_toggle_mid_sequence_continues_counter() {
        let p0 = t(vec![0.0], &[1]);
        let mut ema = ExponentialMovingAverage::new(0.999, &[&p0]).unwrap();
        let u = t(vec![1.0], &[1]);
        ema.update(&[&u]).unwrap();
        ema.update(&[&u]).unwrap();
        let s = ema.shadow("0").unwrap().as_slice().unwrap()[0];
        let mut ema = ema.with_warmup(true);
        ema.update(&[&u]).unwrap();
        // 途中で ON にすると次の更新から効き、n は 2 のまま続く。
        let eff = expected_decay(0.999, 2);
        let expected = f32::mul_add(eff, s, (1.0 - eff) * 1.0);
        assert_eq!(bits_of(&ema, "0"), vec![expected.to_bits()]);
        assert_eq!(ema.num_updates(), 3);
    }
}
