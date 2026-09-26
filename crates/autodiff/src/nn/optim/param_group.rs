//! param groups（層別学習率・weight decay。イシュー #2173・親 #2131）。
//!
//! `torch.optim.Optimizer.param_groups` 相当の機能を、既存 optimizer
//! （[`super::AdamW`]・[`super::Adam`]・[`super::RmsProp`]・
//! [`super::Adagrad`]・[`super::Lamb`]・`crate::optim::Sgd`）へ後付けする。
//! [`ParamGroup`] はパラメータ集合（スロット添字列）ごとの `lr`／
//! `weight_decay` の上書き値を表す純データ型で、[`ParamGroupStep`] は
//! 各 optimizer が `step_with_groups` を実装するための trait である。
//!
//! **facade 非公開**（現時点）: 親イシュー #2131 は「facade 公開面の
//! 拡張は設計判断記録 → 承認 → 実装の 2 段」と定めており、本イシュー・
//! 親イシューのいずれにも所有者の承認コメントがない。このため本モジュール
//! は内部クレート（`fandhe_ai_autodiff`）限定で実装し、`crates/facade/
//! src/optim.rs`（名前指定の再エクスポート）からは一切参照しない。
//! `crates/facade/src/lib.rs::ParamGroupsHoldDoctestGuard` と
//! `crates/facade/tests/api_surface.rs` の否定ガードが、facade がこの
//! 名前を再エクスポート・宣言しないことを機械的に固定する（承認事項の
//! 詳細は `docs/autodiff-param-groups-decision.md` §5 を参照）。
//!
//! # スロット添字方式
//!
//! [`ParamGroup::params`] は Tensor 参照ではなく**スロット添字**
//! （`usize`）の列である。呼び出し元が `step_with_groups` の
//! `params`／`grads` に渡す列の位置（`Sequential::trainable_parameters`／
//! `SequentialVars::trainable_grads` と同じ「位置対応契約」。
//! `crates/facade/src/optim.rs` 冒頭 doc 参照）を指す。
//!
//! # 既定スロットの扱い（重要な仕様）
//!
//! どのグループにも属さないスロットは、optimizer の現在の config
//! （`config.lr`／`config.weight_decay`）をそのまま使う。これは黙示の
//! フォールバックではなく仕様として固定する契約であり、これにより
//! `groups = &[]` を渡した `step_with_groups` は既存 `step()` と
//! **bit 完全一致**する（`crates/autodiff/tests/nn_optim_param_groups.rs`
//! が固定する）。
//!
//! `set_lr`（LR scheduler 結線用。各 optimizer の doc 参照）は従来
//! どおり optimizer の config（既定グループ）の `lr` だけを書き換える。
//! グループの `lr` は絶対値であり `set_lr` の影響は受けない。
//!
//! # 検証規則（fail-closed。状態変更の前に全件を検証する）
//!
//! [`resolve_slot_hparams`] が一括して検証する:
//! - グループの `params` が空 → `InvalidArgument`
//! - スロット添字が `n_slots` の範囲外 → `InvalidArgument`
//! - 同一添字が同一グループ内、またはグループ間で重複 →
//!   `InvalidArgument`（PyTorch の「some parameters appear in more than
//!   one parameter group」と同じ）
//! - `lr`／`weight_decay` が非有限、または負値 → `InvalidArgument`
//!
//! いずれの検査も、呼び出し元 optimizer の状態（`m`／`v`／velocity 等）
//! を一切変更する前に完了する（各 `ParamGroupStep` 実装は
//! `params.len() == grads.len()` の検証 → [`resolve_slot_hparams`] →
//! 状態変更を伴う `step_with_slot_hparams` 系メソッドの順で呼ぶ）。
//!
//! # 追加しないもの（スコープ外。`docs/autodiff-param-groups-decision.md`
//! §6 参照）
//!
//! - Tensor 以外のパラメータ型を含むグループ
//! - optimizer 内部状態（`m`／`v`／velocity・`step_count`）のグループ
//!   追従。状態は従来どおり単一のまま
//! - optimizer 固有のハイパーパラメータ（`beta`／`eps`／`momentum`／
//!   `nesterov`／`lr_decay`／trust ratio 等）のグループ上書き
//! - `crate::optim::device_store::DeviceParamStore` 常駐経路の group 対応

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;

use super::adagrad::Adagrad;
use super::adam::Adam;
use super::adamw::AdamW;
use super::lamb::Lamb;
use super::rmsprop::RmsProp;
use crate::optim::Sgd;

/// パラメータグループ（層別学習率・weight decay）。
///
/// `params` はスロット添字列（モジュール doc「スロット添字方式」節）、
/// `lr`／`weight_decay` はこのグループに属するスロットへ適用する絶対値
/// （optimizer の config 値を上書きする）。
///
/// `#[non_exhaustive]` によりクレート外からは [`ParamGroup::new`] でのみ
/// 構築できる（将来のフィールド追加——momentum 等——が非破壊になる）。
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ParamGroup {
    /// このグループに属するパラメータのスロット添字列。
    pub params: Vec<usize>,
    /// このグループへ適用する学習率（絶対値）。
    pub lr: f32,
    /// このグループへ適用する weight decay（絶対値）。
    pub weight_decay: f32,
}

impl ParamGroup {
    /// `params`（スロット添字列）・`lr`・`weight_decay` からグループを
    /// 構築する。検証（範囲・重複・有限性）は `resolve_slot_hparams`
    /// （内部専用ヘルパー。呼び出し元 optimizer の `n_slots` が判明する
    /// `step_with_groups` 呼び出し時点）で一括して行うため、本
    /// コンストラクタ自体はフィールドを保持するだけで失敗しない。
    pub fn new(params: Vec<usize>, lr: f32, weight_decay: f32) -> ParamGroup {
        ParamGroup {
            params,
            lr,
            weight_decay,
        }
    }
}

/// 1 スロットへ実際に適用する `lr`／`weight_decay`（[`resolve_slot_hparams`]
/// の出力。グループ未所属スロットは optimizer の既定 config 値になる）。
/// facade からは到達不能（`nn/optim/mod.rs` で `pub(crate) use` のみ）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct SlotHparams {
    pub(crate) lr: f32,
    pub(crate) weight_decay: f32,
}

/// `groups` を検証し、`n_slots` 件の [`SlotHparams`] へ解決する。
///
/// モジュール doc「検証規則」節の全項目を、呼び出し元 optimizer の状態を
/// 変更する前に検査する（fail-closed。`who` はエラーメッセージに含める
/// optimizer 名）。検証を通過した場合、戻り値の `i` 番目はスロット `i`
/// に適用する `lr`／`weight_decay`（グループ未所属なら
/// `default_lr`／`default_wd`）。
pub(crate) fn resolve_slot_hparams(
    groups: &[ParamGroup],
    n_slots: usize,
    default_lr: f32,
    default_wd: f32,
    who: &str,
) -> Result<Vec<SlotHparams>, AutodiffError> {
    let mut assigned: Vec<Option<usize>> = vec![None; n_slots];

    for (group_idx, group) in groups.iter().enumerate() {
        if group.params.is_empty() {
            return Err(AutodiffError::InvalidArgument(format!(
                "{who}::step_with_groups: param_groups[{group_idx}].params is empty"
            )));
        }
        if !(group.lr.is_finite() && group.lr >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "{who}::step_with_groups: param_groups[{group_idx}].lr must be finite and \
                 >= 0.0, got {}",
                group.lr
            )));
        }
        if !(group.weight_decay.is_finite() && group.weight_decay >= 0.0) {
            return Err(AutodiffError::InvalidArgument(format!(
                "{who}::step_with_groups: param_groups[{group_idx}].weight_decay must be \
                 finite and >= 0.0, got {}",
                group.weight_decay
            )));
        }
        for &slot in &group.params {
            if slot >= n_slots {
                return Err(AutodiffError::InvalidArgument(format!(
                    "{who}::step_with_groups: param_groups[{group_idx}] references slot \
                     index {slot}, but only {n_slots} slot(s) were passed to step_with_groups"
                )));
            }
            match assigned[slot] {
                Some(prev_group_idx) => {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "{who}::step_with_groups: slot index {slot} appears in both \
                         param_groups[{prev_group_idx}] and param_groups[{group_idx}] (a \
                         slot may belong to at most one group)"
                    )));
                }
                None => assigned[slot] = Some(group_idx),
            }
        }
    }

    Ok((0..n_slots)
        .map(|slot| match assigned[slot] {
            Some(group_idx) => SlotHparams {
                lr: groups[group_idx].lr,
                weight_decay: groups[group_idx].weight_decay,
            },
            None => SlotHparams {
                lr: default_lr,
                weight_decay: default_wd,
            },
        })
        .collect())
}

/// optimizer が [`ParamGroup`] 単位で 1 step を実行するための trait。
///
/// シグネチャは `crate::optim::Sgd::step` と同じ 2 スライス形
/// （`params`・`grads`）に統一する（`crates/facade/src/optim.rs` の
/// `OptimizerState::step` と同じ形のため、承認後の `compile()` 結線が
/// 容易になる。`docs/autodiff-param-groups-decision.md` §2 参照）。
///
/// `groups` が空スライスのとき、実装は既存 `step()`（tuple 列を受け取る
/// optimizer は内部で `params`／`grads` を zip する）と**bit 完全一致**
/// する契約（モジュール doc「既定スロットの扱い」節）。
pub trait ParamGroupStep {
    /// `params[i]`／`grads[i]` を対応づけて 1 step 実行し、更新後
    /// テンソル列を `params` と同順で返す。
    ///
    /// # Errors
    ///
    /// - `params.len() != grads.len()` → `InvalidArgument`
    /// - `groups` の検証違反（モジュール doc「検証規則」節） →
    ///   `InvalidArgument`（状態変更前に検出する）
    /// - 各 optimizer 固有の shape 検証エラー
    fn step_with_groups(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
        groups: &[ParamGroup],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError>;
}

/// `params`／`grads`（2 スライス形）を `(param, grad)` タプル列
/// （tuple 形。`AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb::step` が
/// 受け取る形）へ変換する共通ヘルパー。所有権を取らず参照のみを
/// 詰め替えるため追加コピーは発生しない。
fn zip_params_grads<'a>(
    params: &[&'a Tensor<f32>],
    grads: &[&'a Tensor<f32>],
) -> Vec<(&'a Tensor<f32>, &'a Tensor<f32>)> {
    params.iter().copied().zip(grads.iter().copied()).collect()
}

/// `params.len() != grads.len()` の検証を各 `ParamGroupStep` 実装で
/// 共通化するヘルパー（`who` はエラーメッセージ用の optimizer 名）。
fn check_len_matches(
    who: &str,
    params: &[&Tensor<f32>],
    grads: &[&Tensor<f32>],
) -> Result<(), AutodiffError> {
    if params.len() != grads.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "{who}::step_with_groups: params.len() ({}) != grads.len() ({})",
            params.len(),
            grads.len()
        )));
    }
    Ok(())
}

impl ParamGroupStep for AdamW {
    fn step_with_groups(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
        groups: &[ParamGroup],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        check_len_matches("AdamW", params, grads)?;
        let hparams = resolve_slot_hparams(
            groups,
            params.len(),
            self.config().lr,
            self.config().weight_decay,
            "AdamW",
        )?;
        let pairs = zip_params_grads(params, grads);
        self.step_with_slot_hparams(&pairs, &hparams)
    }
}

impl ParamGroupStep for Adam {
    fn step_with_groups(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
        groups: &[ParamGroup],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        check_len_matches("Adam", params, grads)?;
        let hparams = resolve_slot_hparams(
            groups,
            params.len(),
            self.config().lr,
            self.config().weight_decay,
            "Adam",
        )?;
        let pairs = zip_params_grads(params, grads);
        self.step_with_slot_hparams(&pairs, &hparams)
    }
}

impl ParamGroupStep for RmsProp {
    fn step_with_groups(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
        groups: &[ParamGroup],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        check_len_matches("RmsProp", params, grads)?;
        let hparams = resolve_slot_hparams(
            groups,
            params.len(),
            self.config().lr,
            self.config().weight_decay,
            "RmsProp",
        )?;
        let pairs = zip_params_grads(params, grads);
        self.step_with_slot_hparams(&pairs, &hparams)
    }
}

impl ParamGroupStep for Adagrad {
    fn step_with_groups(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
        groups: &[ParamGroup],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        check_len_matches("Adagrad", params, grads)?;
        let hparams = resolve_slot_hparams(
            groups,
            params.len(),
            self.config().lr,
            self.config().weight_decay,
            "Adagrad",
        )?;
        let pairs = zip_params_grads(params, grads);
        self.step_with_slot_hparams(&pairs, &hparams)
    }
}

impl ParamGroupStep for Lamb {
    fn step_with_groups(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
        groups: &[ParamGroup],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        check_len_matches("Lamb", params, grads)?;
        let hparams = resolve_slot_hparams(
            groups,
            params.len(),
            self.config().lr,
            self.config().weight_decay,
            "Lamb",
        )?;
        let pairs = zip_params_grads(params, grads);
        self.step_with_slot_hparams(&pairs, &hparams)
    }
}

impl ParamGroupStep for Sgd {
    fn step_with_groups(
        &mut self,
        params: &[&Tensor<f32>],
        grads: &[&Tensor<f32>],
        groups: &[ParamGroup],
    ) -> Result<Vec<Tensor<f32>>, AutodiffError> {
        check_len_matches("Sgd", params, grads)?;
        let hparams = resolve_slot_hparams(
            groups,
            params.len(),
            self.config().lr,
            self.config().weight_decay,
            "Sgd",
        )?;
        self.step_with_slot_hparams(params, grads, &hparams)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_slot_hparams_empty_groups_uses_defaults_for_all_slots() {
        let result = resolve_slot_hparams(&[], 3, 0.1, 0.2, "Test").unwrap();
        assert_eq!(result.len(), 3);
        for hp in &result {
            assert_eq!(hp.lr, 0.1);
            assert_eq!(hp.weight_decay, 0.2);
        }
    }

    #[test]
    fn resolve_slot_hparams_overrides_only_assigned_slots() {
        let groups = vec![ParamGroup::new(vec![1], 0.5, 0.0)];
        let result = resolve_slot_hparams(&groups, 3, 0.1, 0.2, "Test").unwrap();
        assert_eq!(result[0].lr, 0.1);
        assert_eq!(result[0].weight_decay, 0.2);
        assert_eq!(result[1].lr, 0.5);
        assert_eq!(result[1].weight_decay, 0.0);
        assert_eq!(result[2].lr, 0.1);
        assert_eq!(result[2].weight_decay, 0.2);
    }

    #[test]
    fn resolve_slot_hparams_rejects_empty_group_params() {
        let groups = vec![ParamGroup::new(vec![], 0.5, 0.0)];
        let err = resolve_slot_hparams(&groups, 3, 0.1, 0.2, "Test").unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn resolve_slot_hparams_rejects_out_of_range_index() {
        let groups = vec![ParamGroup::new(vec![5], 0.5, 0.0)];
        let err = resolve_slot_hparams(&groups, 3, 0.1, 0.2, "Test").unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn resolve_slot_hparams_rejects_duplicate_within_group() {
        let groups = vec![ParamGroup::new(vec![0, 0], 0.5, 0.0)];
        let err = resolve_slot_hparams(&groups, 3, 0.1, 0.2, "Test").unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn resolve_slot_hparams_rejects_duplicate_across_groups() {
        let groups = vec![
            ParamGroup::new(vec![0], 0.5, 0.0),
            ParamGroup::new(vec![0], 0.1, 0.0),
        ];
        let err = resolve_slot_hparams(&groups, 3, 0.1, 0.2, "Test").unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn resolve_slot_hparams_rejects_negative_lr() {
        let groups = vec![ParamGroup::new(vec![0], -0.1, 0.0)];
        let err = resolve_slot_hparams(&groups, 3, 0.1, 0.2, "Test").unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn resolve_slot_hparams_rejects_nan_weight_decay() {
        let groups = vec![ParamGroup::new(vec![0], 0.1, f32::NAN)];
        let err = resolve_slot_hparams(&groups, 3, 0.1, 0.2, "Test").unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    #[test]
    fn resolve_slot_hparams_rejects_non_finite_lr() {
        let groups = vec![ParamGroup::new(vec![0], f32::INFINITY, 0.0)];
        let err = resolve_slot_hparams(&groups, 3, 0.1, 0.2, "Test").unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }
}
