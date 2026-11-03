//! `DeviceParamStore` の常駐更新を param groups に対応させる内部実装（イシュー
//! #3837・設計 `docs/autodiff-param-groups-decision.md` §19。承認: #3769 の
//! 所有者回答コメント 論点 4）。
//!
//! # 役割と呼び出し元
//! - 入口 [`device_param_store_with_slot_groups`]: group 付きで store を構築する。
//!   `optim::resident_param_groups` 経由でだけ到達できる（`DeviceParamStore` の
//!   inherent `pub fn` を増やすと facade の型再エクスポートで公開面になるため、
//!   `optim::resident_input` と同じく `pub mod` 側のフリー関数にする。facade の
//!   入口 `Sequential::init_device_param_store_with_param_groups` は #3838 が持つ）。
//! - 各 `step*`（`device_store.rs`）が、groups ありのときだけ本モジュールの
//!   `*_group_segments` で区間テーブルを作り、`BackendOps::*_step_device_grouped`
//!   へ 1 回の起動で渡す。groups なし（`slot_groups == None`）は既存コードの通り道を
//!   変えない（既定経路の bit 不変は構造で守る）。
//!
//! # 意味
//! group の `lr`／`weight_decay` は絶対値で、各 step の config の `lr` には追従しない
//! （group 外スロットだけ config から毎回解決する。§19.0）。group の変更は store の
//! 作り直しでだけ行う（setter は設けない）。導出スカラーの式・順序はホストの
//! `step_with_slot_hparams` と同じ（Adam の `step_size`、AdamW の `decay_factor`、
//! Adagrad の `clr`、Lamb の `step_size`）。
//!
//! # 非対応（fail-closed）
//! AMP・distributed・`train_step_resident` のチェーン経路は groups ありを拒否する
//! （それぞれ `amp.rs`／`distributed.rs`／`train_chain.rs`）。capability
//! （[`BackendOps::supports_grouped_optimizer_step`]）が `false` のバックエンド
//! （ROCm 等）は構築時に `Unsupported`。

use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{
    AdagradStepSegment, AdamStepKind, AdamStepSegment, LambStepSegment, RmsPropStepSegment,
    SgdStepSegment, Tensor,
};

use crate::nn::optim::param_group::{ParamGroup, resolve_slot_hparams};
use crate::tape::Tape;

use super::DeviceParamStore;

/// スロットごとの group 割り当て（`None` は group 外 = 各 step の config から解決）。
#[derive(Debug, Clone)]
pub(super) struct SlotGroupTable {
    slots: Vec<Option<(f32, f32)>>,
}

impl SlotGroupTable {
    /// スロット `i` の `(lr, weight_decay)`。group 外は渡された既定値。
    fn hp(&self, i: usize, default_lr: f32, default_wd: f32) -> (f32, f32) {
        self.slots
            .get(i)
            .copied()
            .flatten()
            .unwrap_or((default_lr, default_wd))
    }
}

/// group 付きで [`DeviceParamStore`] を構築する（内部入口。#3838 の facade 入口が呼ぶ）。
///
/// `groups` が空なら [`DeviceParamStore::new`] と完全に同じ store を返す。検証
/// （空・範囲外・重複・非有限・負）と capability の検査は upload より前に行い、
/// エラーでは store を作らない。検証エラーは `InvalidArgument`、capability 不足は
/// `Unsupported`。
pub fn device_param_store_with_slot_groups(
    tape: &Tape,
    params: &[&Tensor<f32>],
    groups: &[ParamGroup],
) -> Result<DeviceParamStore, BackendError> {
    if groups.is_empty() {
        return DeviceParamStore::new(tape, params);
    }
    // 既定値は構築時に分からないため NaN を番兵にし、未割り当てスロットを判別する
    // （group の lr／wd は有限・非負に検証済みなので NaN になりえない）。
    let resolved =
        resolve_slot_hparams(groups, params.len(), f32::NAN, f32::NAN, "DeviceParamStore")
            .map_err(|e| BackendError::InvalidArgument(e.to_string()))?;
    if !tape.ops().supports_grouped_optimizer_step() {
        return Err(BackendError::Unsupported(
            "DeviceParamStore: this backend does not support param groups for resident \
             updates (supports_grouped_optimizer_step() == false)"
                .to_string(),
        ));
    }
    let table = SlotGroupTable {
        slots: resolved
            .iter()
            .map(|h| (!h.lr.is_nan()).then_some((h.lr, h.weight_decay)))
            .collect(),
    };
    let mut store = DeviceParamStore::new(tape, params)?;
    store.slot_groups = Some(table);
    Ok(store)
}

/// 各 `step*` が非有限の導出値を検出したときの共通エラー（カーネル起動前に返す）。
fn non_finite_derived(who: &str, what: &str) -> BackendError {
    BackendError::InvalidArgument(format!(
        "DeviceParamStore::{who}: derived {what} is not finite for a param group; \
         refusing to launch the kernel"
    ))
}

impl DeviceParamStore {
    /// groups あり構築か。
    pub(super) fn has_slot_groups(&self) -> bool {
        self.slot_groups.is_some()
    }

    /// groups ありのとき、隣り合う同値 `(lr, weight_decay)` を統合した区間列
    /// `(numel, lr, wd)` を返す。統合は `to_bits()` の完全一致だけ（`-0.0 == 0.0`
    /// で統合するとホストと符号つきゼロが食い違うため）。`numel == 0` は捨てる。
    fn lr_wd_runs(&self, default_lr: f32, default_wd: f32) -> Option<Vec<(usize, f32, f32)>> {
        let table = self.slot_groups.as_ref()?;
        let mut out: Vec<(usize, f32, f32)> = Vec::new();
        for (i, l) in self.layout.iter().enumerate() {
            if l.numel == 0 {
                continue;
            }
            let (lr, wd) = table.hp(i, default_lr, default_wd);
            match out.last_mut() {
                Some(last)
                    if last.1.to_bits() == lr.to_bits() && last.2.to_bits() == wd.to_bits() =>
                {
                    last.0 += l.numel;
                }
                _ => out.push((l.numel, lr, wd)),
            }
        }
        Some(out)
    }

    /// SGD の区間列（config の `lr`／`weight_decay` が group 外スロットの値）。
    pub(super) fn sgd_group_segments(&self, lr: f32, wd: f32) -> Option<Vec<SgdStepSegment>> {
        let runs = self.lr_wd_runs(lr, wd)?;
        Some(
            runs.into_iter()
                .map(|(numel, lr, weight_decay)| SgdStepSegment {
                    numel,
                    lr,
                    weight_decay,
                })
                .collect(),
        )
    }

    /// Adam／AdamW の区間列。`step_size = (lr as f64 / bias_correction1) as f32`、
    /// `decay_factor = 1.0 - lr * weight_decay`（ホスト `step_with_slot_hparams` と同式）。
    pub(super) fn adam_group_segments(
        &self,
        kind: AdamStepKind,
        lr: f32,
        wd: f32,
        bias_correction1: f64,
    ) -> Result<Option<Vec<AdamStepSegment>>, BackendError> {
        let Some(runs) = self.lr_wd_runs(lr, wd) else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(runs.len());
        for (numel, slr, swd) in runs {
            let step_size = (slr as f64 / bias_correction1) as f32;
            let decay_factor = 1.0 - slr * swd;
            if !step_size.is_finite() {
                return Err(non_finite_derived("step_adam", "step_size"));
            }
            if matches!(kind, AdamStepKind::Decoupled) && !decay_factor.is_finite() {
                return Err(non_finite_derived("step_adamw", "decay_factor"));
            }
            out.push(AdamStepSegment {
                numel,
                step_size,
                weight_decay: swd,
                decay_factor,
            });
        }
        Ok(Some(out))
    }

    /// RmsProp の区間列。
    pub(super) fn rmsprop_group_segments(
        &self,
        lr: f32,
        wd: f32,
    ) -> Option<Vec<RmsPropStepSegment>> {
        let runs = self.lr_wd_runs(lr, wd)?;
        Some(
            runs.into_iter()
                .map(|(numel, lr, weight_decay)| RmsPropStepSegment {
                    numel,
                    lr,
                    weight_decay,
                })
                .collect(),
        )
    }

    /// Adagrad の区間列。`clr = (lr as f64 / (1.0 + (step-1) as f64 * lr_decay as f64)) as f32`。
    pub(super) fn adagrad_group_segments(
        &self,
        lr: f32,
        wd: f32,
        step: u64,
        lr_decay: f32,
    ) -> Result<Option<Vec<AdagradStepSegment>>, BackendError> {
        let Some(runs) = self.lr_wd_runs(lr, wd) else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(runs.len());
        for (numel, slr, swd) in runs {
            let clr = (slr as f64 / (1.0 + (step - 1) as f64 * lr_decay as f64)) as f32;
            if !clr.is_finite() {
                return Err(non_finite_derived("step_adagrad", "clr"));
            }
            out.push(AdagradStepSegment {
                numel,
                clr,
                weight_decay: swd,
            });
        }
        Ok(Some(out))
    }

    /// LAMB の層ごとの行（統合しない。trust ratio は層単位の縮約のため）。
    pub(super) fn lamb_group_segments(
        &self,
        lr: f32,
        wd: f32,
        bias_correction1: f64,
    ) -> Result<Option<Vec<LambStepSegment>>, BackendError> {
        let Some(table) = self.slot_groups.as_ref() else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(self.layout.len());
        for (i, l) in self.layout.iter().enumerate() {
            let (slr, swd) = table.hp(i, lr, wd);
            let step_size = (slr as f64 / bias_correction1) as f32;
            if !step_size.is_finite() {
                return Err(non_finite_derived("step_lamb", "step_size"));
            }
            out.push(LambStepSegment {
                numel: l.numel,
                lr: slr,
                step_size,
                weight_decay: swd,
            });
        }
        Ok(Some(out))
    }
}
