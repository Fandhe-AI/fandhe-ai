//! `save_model`／`load_model` の compile 状態（manifest の `compiled` 節と safetensors の
//! `optimizer.` 名前空間。イシュー #2372・親 #2362）。
//!
//! 役割: `Sequential::compile`／`compile_with_amp` の状態（loss 種別・optimizer 種別と
//! **現在の** config・optimizer 内部状態・AMP の `GradScaler` 状態）を、親モジュール
//! （`model_io.rs`）の手書き厳格 JSON パーサ・レンダラの上で往復させる。形式の正本は
//! `docs/compat-model-io-decision.md` §4・§5・§11・§13.5。
//!
//! 責務境界: optimizer 内部状態の取り出し・書き戻しと種別マーカー照合は
//! `Sequential::snapshot_compiled`／`restore_compiled`（`training.rs`。内部クレートの
//! `OptimizerStateDict`・`grad_scaler_from_state` を呼ぶ）が担い、本モジュールは manifest の
//! 描画・パース（文字列 allowlist・固定キー集合・正準 f32）と、manifest／safetensors の
//! キー集合・shape の照合だけを行う。値の範囲検証（`lr` の正値・`scale` の非正規化数・
//! `growth_tracker < growth_interval` 等）は各コンストラクタへ委ね、検証の迂回経路を作らない。
//!
//! param groups（#3835）は任意キー `compiled.param_groups`（CSR 形。groups が空ならキーなし。
//! `format_version` は据え置き。旧リーダーは未知キーで fail-closed。正は
//! `docs/autodiff-param-groups-decision.md` §18）。
//!
//! 呼び出し元: `model_io.rs` の `prepare_save`・`render_manifest`・`verify_round_trip`・
//! `parse_manifest`・`load_from_dir_with_limits`。
//!
//! `Lbfgs`（#2373）は `kind: "lbfgs"` として保存・復元する（AMP 併用は対象外）。manifest の
//! optimizer object には `history_len`（履歴ペア件数）を追加し、`config` は `LbfgsConfig` の
//! 全 8 フィールド（`max_eval` は `null` か整数・`line_search` は `"none"`／`"strong_wolfe"`）を持つ。
//! 履歴件数は非信頼値のため、固定上限 `MAX_LBFGS_HISTORY`（親モジュール。65536。2026-09-29
//! ユーザー承認）で `config.history_size` と `history_len` を挟み、load では safetensors の
//! 実キー数と照合する（`check_lbfgs_history`。`history_len` を確保量の根拠にしない）。

use std::collections::HashMap;

use super::super::training::{AmpDType, AmpSnapshot, CompiledSnapshot, Loss, Optimizer};
use super::{
    Json, MAX_LBFGS_HISTORY, ModelIoError, Params, as_arr, as_f32, as_str, as_u64, as_usize, clip,
    exact_fields, manifest_error,
};
use crate::Tensor;
use crate::optim::ParamGroup;
use crate::optim::{
    AdadeltaConfig, AdagradConfig, AdamConfig, AdamWConfig, AdamaxConfig, GradScalerConfig,
    LambConfig, LbfgsConfig, RmsPropConfig, SgdConfig,
};
// facade は `LbfgsLineSearch` を再エクスポートしない（承認範囲外）。ここは private use で、
// manifest 文字列との相互変換にだけ使う（`api_surface` の公開面検査の対象外）。
use fandhe_ai_autodiff::nn::optim::LbfgsLineSearch;

/// safetensors 内の optimizer 状態キーの接頭辞（`Sequential::state_dict` のキーは
/// `{層番号}.{名前}` で、この接頭辞から始まることはない）。
pub(super) const OPTIMIZER_PREFIX: &str = "optimizer.";

/// manifest の `compiled` 節の中身（描画・パース・照合の共通表現）。
pub(super) struct CompiledMeta {
    pub(super) loss: Loss,
    pub(super) optimizer: Optimizer,
    /// safetensors 内の完全キー（`optimizer.` 接頭辞付き）。昇順・重複なし。
    pub(super) state_keys: Vec<String>,
    pub(super) amp: Option<AmpSnapshot>,
    /// `Lbfgs` の履歴ペア件数（`Lbfgs` のときだけ `Some`。manifest の `history_len`）。
    pub(super) lbfgs_history_len: Option<usize>,
    /// param groups（#3835。空 = `param_groups` キーなし。形式は決定記録 §18）。
    pub(super) param_groups: Vec<ParamGroup>,
}

/// optimizer config の現在 lr（`from_snapshot` の base_lr 照合用）。
fn optimizer_lr(o: &Optimizer) -> f32 {
    match o {
        Optimizer::Sgd(c) => c.lr,
        Optimizer::AdamW(c) => c.lr,
        Optimizer::Adam(c) => c.lr,
        Optimizer::RmsProp(c) => c.lr,
        Optimizer::Adagrad(c) => c.lr,
        Optimizer::Lamb(c) => c.lr,
        Optimizer::Lbfgs(c) => c.lr,
        Optimizer::Adadelta(c) => c.lr,
        Optimizer::Adamax(c) => c.lr,
    }
}

/// `ParamGroup` の検証（autodiff の `resolve_slot_hparams` と同じ規則のうちスロット範囲以外。
/// 違反は保存不能な構成として `UnsupportedModel` 用の理由文を返す）。
fn validate_groups(groups: &[ParamGroup]) -> Result<(), String> {
    let mut seen: Vec<usize> = groups
        .iter()
        .flat_map(|g| g.params.iter().copied())
        .collect();
    for (i, g) in groups.iter().enumerate() {
        if g.params.is_empty() {
            return Err(format!("param_groups[{i}] の params が空です"));
        }
        for (name, v) in [("lr", g.lr), ("weight_decay", g.weight_decay)] {
            if !v.is_finite() || v < 0.0 {
                return Err(format!(
                    "param_groups[{i}].{name} が有限の非負値ではありません"
                ));
            }
        }
    }
    seen.sort_unstable();
    if seen.windows(2).any(|w| w[0] == w[1]) {
        return Err("param_groups にスロットの重複があります".to_string());
    }
    Ok(())
}

/// `param_groups` のスロット添字が訓練対象パラメータ数に収まることを確認する
/// （保存側 `prepare_save`・読込側 `load_from_dir_with_limits` の両方から呼ぶ。範囲外は `Mismatch`）。
pub(super) fn check_param_group_slots(
    groups: &[ParamGroup],
    n_params: usize,
) -> Result<(), ModelIoError> {
    for g in groups {
        if let Some(&slot) = g.params.iter().find(|&&s| s >= n_params) {
            return Err(ModelIoError::Mismatch {
                message: format!(
                    "param_groups がスロット添字 {slot} を参照していますが、訓練対象パラメータは {n_params} 個です"
                ),
            });
        }
    }
    Ok(())
}

/// `Lbfgs` の履歴に関する固定上限の検査（`what` は `TooLarge` の対象名）。
fn check_lbfgs_limit(n: usize, what: &'static str) -> Result<(), ModelIoError> {
    check_lbfgs_limit_with(n, MAX_LBFGS_HISTORY, what)
}

fn check_lbfgs_limit_with(n: usize, limit: usize, what: &'static str) -> Result<(), ModelIoError> {
    if n > limit {
        return Err(ModelIoError::TooLarge {
            what,
            limit: limit as u64,
        });
    }
    Ok(())
}

impl CompiledMeta {
    /// snapshot から manifest 用の写しを作る。`Lbfgs` は履歴の固定上限・`line_search` の
    /// 既知 variant・AMP 非併用を確認する（保存できないものはここで拒否する）。
    pub(super) fn from_snapshot(snap: &CompiledSnapshot) -> Result<Self, ModelIoError> {
        if let Optimizer::Lbfgs(c) = &snap.optimizer {
            check_lbfgs_limit(c.history_size, "Lbfgs history_size")?;
            let Some(n) = snap.lbfgs_history_len else {
                return Err(ModelIoError::UnsupportedModel {
                    reason: "Lbfgs の履歴件数が取得できません".into(),
                });
            };
            check_lbfgs_limit(n, "Lbfgs 履歴件数")?;
            if snap.amp.is_some() {
                return Err(ModelIoError::UnsupportedModel {
                    reason: "Lbfgs と AMP の併用は保存できません".into(),
                });
            }
            if line_search_name(c.line_search).is_none() {
                return Err(ModelIoError::UnsupportedModel {
                    reason: "未対応の Lbfgs line_search です".into(),
                });
            }
        }
        if !snap.param_groups.is_empty() {
            if matches!(snap.optimizer, Optimizer::Lbfgs(_)) {
                return Err(ModelIoError::UnsupportedModel {
                    reason: "Lbfgs と param groups の併用は保存できません".into(),
                });
            }
            // groups は絶対値で保存する。LrSchedule で lr が変わっていると読込後の base_lr
            // （= 現在 lr）との倍率が失われ、続きの学習率が黙って変わるため拒否する（§18.12）。
            if snap.base_lr.to_bits() != optimizer_lr(&snap.optimizer).to_bits() {
                return Err(ModelIoError::UnsupportedModel {
                    reason: "LrSchedule で学習率が変わった後の param groups 付き compile 状態は保存できません"
                        .into(),
                });
            }
            validate_groups(&snap.param_groups)
                .map_err(|reason| ModelIoError::UnsupportedModel { reason })?;
        }
        let mut state_keys: Vec<String> = snap
            .optimizer_state
            .keys()
            .map(|k| format!("{OPTIMIZER_PREFIX}{k}"))
            .collect();
        state_keys.sort();
        Ok(CompiledMeta {
            loss: snap.loss,
            optimizer: snap.optimizer,
            state_keys,
            amp: snap.amp.as_ref().map(|a| AmpSnapshot {
                dtype: a.dtype,
                grad_scaler_config: a.grad_scaler_config,
                scale: a.scale,
                growth_tracker: a.growth_tracker,
            }),
            lbfgs_history_len: snap.lbfgs_history_len,
            param_groups: snap.param_groups.clone(),
        })
    }

    /// safetensors から取り出した optimizer 状態（接頭辞なしキー）と合わせて復元用の写しにする。
    pub(super) fn into_snapshot(
        self,
        optimizer_state: HashMap<String, Tensor<f32>>,
    ) -> CompiledSnapshot {
        CompiledSnapshot {
            loss: self.loss,
            optimizer: self.optimizer,
            optimizer_state,
            lbfgs_history_len: self.lbfgs_history_len,
            amp: self.amp,
            base_lr: optimizer_lr(&self.optimizer),
            param_groups: self.param_groups,
        }
    }
}

fn loss_name(loss: Loss) -> &'static str {
    match loss {
        Loss::Mse => "mse",
        Loss::CrossEntropy => "cross_entropy",
        Loss::L1 => "l1",
        Loss::Bce => "bce",
        Loss::BceWithLogits => "bce_with_logits",
        Loss::Nll => "nll",
        Loss::KlDiv => "kl_div",
        Loss::Huber => "huber",
        Loss::SmoothL1 => "smooth_l1",
    }
}

fn dtype_name(dtype: AmpDType) -> &'static str {
    match dtype {
        AmpDType::F16 => "f16",
        AmpDType::Bf16 => "bf16",
    }
}

/// optimizer の manifest 上の `kind`（文字列 allowlist の正。`parse_optimizer` と往復テストで一致を担保）。
fn optimizer_kind(o: &Optimizer) -> &'static str {
    match o {
        Optimizer::Sgd(_) => "sgd",
        Optimizer::AdamW(_) => "adamw",
        Optimizer::Adam(_) => "adam",
        Optimizer::RmsProp(_) => "rmsprop",
        Optimizer::Adagrad(_) => "adagrad",
        Optimizer::Lamb(_) => "lamb",
        Optimizer::Lbfgs(_) => "lbfgs",
        Optimizer::Adadelta(_) => "adadelta",
        Optimizer::Adamax(_) => "adamax",
    }
}

/// `line_search` の manifest 上の文字列（allowlist の正）。`#[non_exhaustive]` の
/// 未知 variant は `None`（保存側 `from_snapshot` が拒否する）。
fn line_search_name(ls: LbfgsLineSearch) -> Option<&'static str> {
    match ls {
        LbfgsLineSearch::None => Some("none"),
        LbfgsLineSearch::StrongWolfe => Some("strong_wolfe"),
        _ => None,
    }
}

fn real(key: &str, v: f32) -> (String, String) {
    (key.to_string(), format!("{v:?}"))
}

/// optimizer の config を全フィールド（キー順固定）で描画する素材にする。
fn config_fields(o: &Optimizer) -> Vec<(String, String)> {
    match o {
        Optimizer::Sgd(c) => vec![
            real("lr", c.lr),
            real("momentum", c.momentum),
            real("dampening", c.dampening),
            real("weight_decay", c.weight_decay),
            ("nesterov".into(), c.nesterov.to_string()),
        ],
        Optimizer::AdamW(c) => vec![
            real("lr", c.lr),
            real("beta1", c.beta1),
            real("beta2", c.beta2),
            real("eps", c.eps),
            real("weight_decay", c.weight_decay),
        ],
        Optimizer::Adam(c) => vec![
            real("lr", c.lr),
            real("beta1", c.beta1),
            real("beta2", c.beta2),
            real("eps", c.eps),
            real("weight_decay", c.weight_decay),
        ],
        Optimizer::Lamb(c) => vec![
            real("lr", c.lr),
            real("beta1", c.beta1),
            real("beta2", c.beta2),
            real("eps", c.eps),
            real("weight_decay", c.weight_decay),
        ],
        Optimizer::RmsProp(c) => vec![
            real("lr", c.lr),
            real("alpha", c.alpha),
            real("eps", c.eps),
            real("weight_decay", c.weight_decay),
            real("momentum", c.momentum),
            ("centered".into(), c.centered.to_string()),
        ],
        Optimizer::Adagrad(c) => vec![
            real("lr", c.lr),
            real("lr_decay", c.lr_decay),
            real("weight_decay", c.weight_decay),
            real("initial_accumulator_value", c.initial_accumulator_value),
            real("eps", c.eps),
        ],
        Optimizer::Adadelta(c) => vec![
            real("lr", c.lr),
            real("rho", c.rho),
            real("eps", c.eps),
            real("weight_decay", c.weight_decay),
        ],
        Optimizer::Adamax(c) => vec![
            real("lr", c.lr),
            real("beta1", c.beta1),
            real("beta2", c.beta2),
            real("eps", c.eps),
            real("weight_decay", c.weight_decay),
        ],
        Optimizer::Lbfgs(c) => vec![
            real("lr", c.lr),
            ("max_iter".into(), c.max_iter.to_string()),
            (
                "max_eval".into(),
                c.max_eval.map_or("null".to_string(), |n| n.to_string()),
            ),
            real("tolerance_grad", c.tolerance_grad),
            real("tolerance_change", c.tolerance_change),
            ("history_size".into(), c.history_size.to_string()),
            (
                "line_search".into(),
                // 未知 variant は `from_snapshot` が先に拒否する。到達しても読み戻しで拒否される値にする。
                format!(
                    "\"{}\"",
                    line_search_name(c.line_search).unwrap_or("unknown")
                ),
            ),
            ("line_search_steps".into(), c.line_search_steps.to_string()),
        ],
    }
}

fn render_object(fields: &[(String, String)]) -> String {
    let body: Vec<String> = fields.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
    format!("{{{}}}", body.join(","))
}

/// `compiled` 節を決定的な JSON 文字列にする（キー順固定・ASCII のみ。f32 は `{:?}`）。
/// 保存側の自己検証は、この文字列を保存する構成と読み戻した構成とで比較する。
pub(super) fn render_compiled(m: &CompiledMeta) -> String {
    let keys: Vec<String> = m.state_keys.iter().map(|k| format!("\"{k}\"")).collect();
    let amp = match &m.amp {
        None => "null".to_string(),
        Some(a) => {
            let c = &a.grad_scaler_config;
            let cfg = render_object(&[
                real("init_scale", c.init_scale),
                real("growth_factor", c.growth_factor),
                real("backoff_factor", c.backoff_factor),
                ("growth_interval".into(), c.growth_interval.to_string()),
            ]);
            format!(
                "{{\"dtype\":\"{}\",\"grad_scaler_config\":{cfg},\"scale\":{:?},\"growth_tracker\":{}}}",
                dtype_name(a.dtype),
                a.scale,
                a.growth_tracker
            )
        }
    };
    let history_len = match (&m.optimizer, m.lbfgs_history_len) {
        (Optimizer::Lbfgs(_), Some(n)) => format!(",\"history_len\":{n}"),
        _ => String::new(),
    };
    format!(
        "{{\"loss\":\"{}\",\"optimizer\":{{\"kind\":\"{}\",\"config\":{}{history_len}}},\"optimizer_state_keys\":[{}],\"amp\":{amp}{}}}",
        loss_name(m.loss),
        optimizer_kind(&m.optimizer),
        render_object(&config_fields(&m.optimizer)),
        keys.join(","),
        render_param_groups(&m.param_groups)
    )
}

/// groups が空なら空文字（キーを書かない＝従来の manifest と完全一致）。空でなければ
/// CSR 形 `,"param_groups":{"lr":[..],"weight_decay":[..],"offsets":[0,..,N],"slots":[..]}`。
fn render_param_groups(groups: &[ParamGroup]) -> String {
    if groups.is_empty() {
        return String::new();
    }
    let join = |it: Vec<String>| it.join(",");
    let lr = join(groups.iter().map(|g| format!("{:?}", g.lr)).collect());
    let wd = join(
        groups
            .iter()
            .map(|g| format!("{:?}", g.weight_decay))
            .collect(),
    );
    let mut offsets = vec!["0".to_string()];
    let mut end = 0usize;
    for g in groups {
        end += g.params.len();
        offsets.push(end.to_string());
    }
    let slots = join(
        groups
            .iter()
            .flat_map(|g| g.params.iter().map(|s| s.to_string()))
            .collect(),
    );
    format!(
        ",\"param_groups\":{{\"lr\":[{lr}],\"weight_decay\":[{wd}],\"offsets\":[{}],\"slots\":[{slots}]}}",
        join(offsets)
    )
}

const CONFIG_CTX: &str = "compiled.optimizer.config";
const ADAM_LIKE_KEYS: [&str; 5] = ["lr", "beta1", "beta2", "eps", "weight_decay"];

/// `optimizer` object（`kind` の allowlist と kind ごとの固定キー集合）を読む。
/// `kind` を先に読み、`lbfgs` だけ `history_len` を加えたキー集合で厳格に照合する。
/// 戻り値の `Option<usize>` は `Lbfgs` の `history_len`（それ以外は `None`）。
fn parse_optimizer(value: &Json) -> Result<(Optimizer, Option<usize>), ModelIoError> {
    let Json::Obj(entries) = value else {
        return Err(manifest_error(
            "compiled.optimizer は object である必要があります",
        ));
    };
    let kind_value = entries
        .iter()
        .find(|(k, _)| k == "kind")
        .map(|(_, v)| v)
        .ok_or_else(|| manifest_error("compiled.optimizer にキー kind がありません"))?;
    let kind = as_str(kind_value, "compiled.optimizer.kind")?;
    let keys: &[&str] = if kind == "lbfgs" {
        &["kind", "config", "history_len"]
    } else {
        &["kind", "config"]
    };
    let f = exact_fields(value, "compiled.optimizer", keys)?;
    if let Some(history_value) = f.get(2) {
        // 上限判定は config・キー照合より前（非信頼値で資源を決めない）。
        let history_len = as_usize(history_value, "compiled.optimizer.history_len")?;
        check_lbfgs_limit(history_len, "Lbfgs 履歴件数")?;
        return parse_lbfgs(f[1], history_len);
    }
    parse_non_lbfgs(kind, f[1]).map(|o| (o, None))
}

/// `kind: "lbfgs"` の config（8 キー）を読む。
fn parse_lbfgs(
    config: &Json,
    history_len: usize,
) -> Result<(Optimizer, Option<usize>), ModelIoError> {
    let p = Params::named(
        CONFIG_CTX,
        config,
        &[
            "lr",
            "max_iter",
            "max_eval",
            "tolerance_grad",
            "tolerance_change",
            "history_size",
            "line_search",
            "line_search_steps",
        ],
    )?;
    let history_size = p.usize("history_size")?;
    check_lbfgs_limit(history_size, "Lbfgs history_size")?;
    let line_search = match as_str(p.get("line_search")?, "line_search")? {
        "none" => LbfgsLineSearch::None,
        "strong_wolfe" => LbfgsLineSearch::StrongWolfe,
        other => {
            return Err(ModelIoError::UnsupportedModel {
                reason: format!("未対応の Lbfgs line_search {}", clip(other)),
            });
        }
    };
    let cfg = LbfgsConfig {
        lr: p.f32("lr")?,
        max_iter: p.usize("max_iter")?,
        max_eval: p.opt_usize("max_eval")?,
        tolerance_grad: p.f32("tolerance_grad")?,
        tolerance_change: p.f32("tolerance_change")?,
        history_size,
        line_search,
        line_search_steps: p.usize("line_search_steps")?,
    };
    Ok((Optimizer::Lbfgs(cfg), Some(history_len)))
}

/// `lbfgs` 以外の kind と config を読む（`kind` の allowlist）。
fn parse_non_lbfgs(kind: &str, config: &Json) -> Result<Optimizer, ModelIoError> {
    match kind {
        "sgd" => {
            let p = Params::named(
                CONFIG_CTX,
                config,
                &["lr", "momentum", "dampening", "weight_decay", "nesterov"],
            )?;
            Ok(Optimizer::Sgd(SgdConfig {
                lr: p.f32("lr")?,
                momentum: p.f32("momentum")?,
                dampening: p.f32("dampening")?,
                weight_decay: p.f32("weight_decay")?,
                nesterov: p.bool("nesterov")?,
            }))
        }
        "adamw" => {
            let p = Params::named(CONFIG_CTX, config, &ADAM_LIKE_KEYS)?;
            Ok(Optimizer::AdamW(AdamWConfig {
                lr: p.f32("lr")?,
                beta1: p.f32("beta1")?,
                beta2: p.f32("beta2")?,
                eps: p.f32("eps")?,
                weight_decay: p.f32("weight_decay")?,
            }))
        }
        "adam" => {
            let p = Params::named(CONFIG_CTX, config, &ADAM_LIKE_KEYS)?;
            Ok(Optimizer::Adam(AdamConfig {
                lr: p.f32("lr")?,
                beta1: p.f32("beta1")?,
                beta2: p.f32("beta2")?,
                eps: p.f32("eps")?,
                weight_decay: p.f32("weight_decay")?,
            }))
        }
        "lamb" => {
            let p = Params::named(CONFIG_CTX, config, &ADAM_LIKE_KEYS)?;
            Ok(Optimizer::Lamb(LambConfig {
                lr: p.f32("lr")?,
                beta1: p.f32("beta1")?,
                beta2: p.f32("beta2")?,
                eps: p.f32("eps")?,
                weight_decay: p.f32("weight_decay")?,
            }))
        }
        "rmsprop" => {
            let p = Params::named(
                CONFIG_CTX,
                config,
                &["lr", "alpha", "eps", "weight_decay", "momentum", "centered"],
            )?;
            Ok(Optimizer::RmsProp(RmsPropConfig {
                lr: p.f32("lr")?,
                alpha: p.f32("alpha")?,
                eps: p.f32("eps")?,
                weight_decay: p.f32("weight_decay")?,
                momentum: p.f32("momentum")?,
                centered: p.bool("centered")?,
            }))
        }
        "adagrad" => {
            let p = Params::named(
                CONFIG_CTX,
                config,
                &[
                    "lr",
                    "lr_decay",
                    "weight_decay",
                    "initial_accumulator_value",
                    "eps",
                ],
            )?;
            Ok(Optimizer::Adagrad(AdagradConfig {
                lr: p.f32("lr")?,
                lr_decay: p.f32("lr_decay")?,
                weight_decay: p.f32("weight_decay")?,
                initial_accumulator_value: p.f32("initial_accumulator_value")?,
                eps: p.f32("eps")?,
            }))
        }
        "adadelta" => {
            let p = Params::named(CONFIG_CTX, config, &["lr", "rho", "eps", "weight_decay"])?;
            Ok(Optimizer::Adadelta(AdadeltaConfig {
                lr: p.f32("lr")?,
                rho: p.f32("rho")?,
                eps: p.f32("eps")?,
                weight_decay: p.f32("weight_decay")?,
            }))
        }
        "adamax" => {
            let p = Params::named(CONFIG_CTX, config, &ADAM_LIKE_KEYS)?;
            Ok(Optimizer::Adamax(AdamaxConfig {
                lr: p.f32("lr")?,
                beta1: p.f32("beta1")?,
                beta2: p.f32("beta2")?,
                eps: p.f32("eps")?,
                weight_decay: p.f32("weight_decay")?,
            }))
        }
        other => Err(ModelIoError::UnsupportedModel {
            reason: format!("未対応の optimizer kind {}", clip(other)),
        }),
    }
}

fn parse_amp(value: &Json) -> Result<Option<AmpSnapshot>, ModelIoError> {
    if matches!(value, Json::Null) {
        return Ok(None);
    }
    let f = exact_fields(
        value,
        "compiled.amp",
        &["dtype", "grad_scaler_config", "scale", "growth_tracker"],
    )?;
    let dtype = match as_str(f[0], "compiled.amp.dtype")? {
        "f16" => AmpDType::F16,
        "bf16" => AmpDType::Bf16,
        other => {
            return Err(ModelIoError::UnsupportedModel {
                reason: format!("未対応の AMP dtype {}", clip(other)),
            });
        }
    };
    let p = Params::named(
        "compiled.amp.grad_scaler_config",
        f[1],
        &[
            "init_scale",
            "growth_factor",
            "backoff_factor",
            "growth_interval",
        ],
    )?;
    let grad_scaler_config = GradScalerConfig {
        init_scale: p.f32("init_scale")?,
        growth_factor: p.f32("growth_factor")?,
        backoff_factor: p.f32("backoff_factor")?,
        growth_interval: as_u64(p.get("growth_interval")?, "growth_interval")?,
    };
    Ok(Some(AmpSnapshot {
        dtype,
        grad_scaler_config,
        scale: as_f32(f[2], "compiled.amp.scale")?,
        growth_tracker: as_u64(f[3], "compiled.amp.growth_tracker")?,
    }))
}

/// manifest の `compiled` 値（`null` または object）を検証して読む。
pub(super) fn parse_compiled(value: &Json) -> Result<Option<CompiledMeta>, ModelIoError> {
    if matches!(value, Json::Null) {
        return Ok(None);
    }
    // param_groups は任意キー（#3835）。有無を先に見てキー集合を選ぶ（parse_optimizer と同型）。
    let has_groups = matches!(value, Json::Obj(e) if e.iter().any(|(k, _)| k == "param_groups"));
    let keys: &[&str] = if has_groups {
        &[
            "loss",
            "optimizer",
            "optimizer_state_keys",
            "amp",
            "param_groups",
        ]
    } else {
        &["loss", "optimizer", "optimizer_state_keys", "amp"]
    };
    let f = exact_fields(value, "compiled", keys)?;
    let loss = match as_str(f[0], "compiled.loss")? {
        "mse" => Loss::Mse,
        "cross_entropy" => Loss::CrossEntropy,
        "l1" => Loss::L1,
        "bce" => Loss::Bce,
        "bce_with_logits" => Loss::BceWithLogits,
        "nll" => Loss::Nll,
        "kl_div" => Loss::KlDiv,
        "huber" => Loss::Huber,
        "smooth_l1" => Loss::SmoothL1,
        other => {
            return Err(ModelIoError::UnsupportedModel {
                reason: format!("未対応の loss {}", clip(other)),
            });
        }
    };
    let (optimizer, lbfgs_history_len) = parse_optimizer(f[1])?;
    let mut state_keys: Vec<String> = Vec::new();
    for k in as_arr(f[2], "compiled.optimizer_state_keys")? {
        let key = as_str(k, "compiled.optimizer_state_keys[]")?;
        if !key.starts_with(OPTIMIZER_PREFIX) {
            return Err(manifest_error(format!(
                "compiled.optimizer_state_keys[] が {OPTIMIZER_PREFIX} で始まりません"
            )));
        }
        // 書き手は昇順・重複なしで出す。厳密増加を要求して重複・順序違いの改竄を拒否する。
        if state_keys.last().is_some_and(|prev| prev.as_str() >= key) {
            return Err(manifest_error(
                "compiled.optimizer_state_keys が昇順・重複なしではありません",
            ));
        }
        state_keys.push(key.to_string());
    }
    let amp = parse_amp(f[3])?;
    if lbfgs_history_len.is_some() && amp.is_some() {
        // `compile_with_amp` が禁じる組合せを manifest から作らせない。
        return Err(ModelIoError::UnsupportedModel {
            reason: "Lbfgs と AMP の併用は復元できません".into(),
        });
    }
    let param_groups = match f.get(4) {
        Some(v) => parse_param_groups(v)?,
        None => Vec::new(),
    };
    if !param_groups.is_empty() && lbfgs_history_len.is_some() {
        return Err(ModelIoError::UnsupportedModel {
            reason: "Lbfgs と param groups の併用は復元できません".into(),
        });
    }
    Ok(Some(CompiledMeta {
        loss,
        optimizer,
        state_keys,
        amp,
        lbfgs_history_len,
        param_groups,
    }))
}

/// `compiled.param_groups`（CSR 形。決定記録 §18.3）を検証して読む。確保量は
/// パース済み配列（`MAX_ARRAY_LEN` で抑止済み）の長さだけで決め、非信頼な値は使わない。
/// スロット範囲（`slot < n_params`）はモデル構築後に `check_param_group_slots` が見る。
fn parse_param_groups(value: &Json) -> Result<Vec<ParamGroup>, ModelIoError> {
    const CTX: &str = "compiled.param_groups";
    let f = exact_fields(value, CTX, &["lr", "weight_decay", "offsets", "slots"])?;
    let lr_arr = as_arr(f[0], "compiled.param_groups.lr")?;
    let wd_arr = as_arr(f[1], "compiled.param_groups.weight_decay")?;
    let off_arr = as_arr(f[2], "compiled.param_groups.offsets")?;
    let slot_arr = as_arr(f[3], "compiled.param_groups.slots")?;
    let g = lr_arr.len();
    if g == 0 || wd_arr.len() != g || off_arr.len() != g + 1 {
        return Err(manifest_error(
            "compiled.param_groups の lr／weight_decay／offsets の長さが不整合です",
        ));
    }
    let mut offsets = Vec::with_capacity(off_arr.len());
    for v in off_arr {
        offsets.push(as_usize(v, "compiled.param_groups.offsets[]")?);
    }
    if offsets[0] != 0 || offsets.windows(2).any(|w| w[0] >= w[1]) || offsets[g] != slot_arr.len() {
        return Err(manifest_error(
            "compiled.param_groups.offsets が 0 始まりの狭義増加で slots 長に一致する必要があります",
        ));
    }
    let mut slots = Vec::with_capacity(slot_arr.len());
    for v in slot_arr {
        slots.push(as_usize(v, "compiled.param_groups.slots[]")?);
    }
    let mut sorted = slots.clone();
    sorted.sort_unstable();
    if sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(manifest_error(
            "compiled.param_groups.slots にスロットの重複があります",
        ));
    }
    let nonneg = |v: &Json, ctx: &str| -> Result<f32, ModelIoError> {
        let x = as_f32(v, ctx)?;
        if x >= 0.0 {
            Ok(x)
        } else {
            Err(manifest_error(format!("{ctx} が負です")))
        }
    };
    let mut groups = Vec::with_capacity(g);
    for i in 0..g {
        let lr = nonneg(&lr_arr[i], "compiled.param_groups.lr[]")?;
        let wd = nonneg(&wd_arr[i], "compiled.param_groups.weight_decay[]")?;
        groups.push(ParamGroup::new(
            slots[offsets[i]..offsets[i + 1]].to_vec(),
            lr,
            wd,
        ));
    }
    Ok(groups)
}

/// safetensors のテンソル集合から optimizer 状態を取り分ける（接頭辞除去済みの map と、
/// manifest 照合用の完全キー昇順リストを返す）。`tensors` にはモデルのパラメータだけが残る。
pub(super) fn split_optimizer_tensors(
    tensors: &mut HashMap<String, Tensor<f32>>,
) -> (Vec<String>, HashMap<String, Tensor<f32>>) {
    let mut full_keys: Vec<String> = tensors
        .keys()
        .filter(|k| k.starts_with(OPTIMIZER_PREFIX))
        .cloned()
        .collect();
    full_keys.sort();
    let mut state = HashMap::with_capacity(full_keys.len());
    for key in &full_keys {
        if let Some(t) = tensors.remove(key) {
            state.insert(key[OPTIMIZER_PREFIX.len()..].to_string(), t);
        }
    }
    (full_keys, state)
}

/// `history.<i>.s|y` の正規表記（`01`・`+1` は不可）を `(添字, s か)` に読む。autodiff の
/// `parse_history_key` は private のため同じ規則を写す（非正規キーは autodiff が余剰キーとして拒否）。
fn parse_history_key(key: &str) -> Option<(usize, bool)> {
    let rest = key.strip_prefix("history.")?;
    let (seg, part) = rest.rsplit_once('.')?;
    let is_s = match part {
        "s" => true,
        "y" => false,
        _ => return None,
    };
    let idx: usize = seg.parse().ok()?;
    (idx.to_string() == seg).then_some((idx, is_s))
}

/// `Lbfgs` の履歴件数を、実際の safetensors キー（`history.{i}.s|y` の添字数）と照合する。
/// `history_len` は照合にだけ使い、確保量の根拠にしない。上限超過は `TooLarge`、
/// 件数の不一致（s と y の不揃いを含む）は `Mismatch`。`Lbfgs` 以外は何もしない。
pub(super) fn check_lbfgs_history(
    meta: &CompiledMeta,
    state: &HashMap<String, Tensor<f32>>,
) -> Result<(), ModelIoError> {
    check_lbfgs_history_with_limit(meta, state, MAX_LBFGS_HISTORY)
}

fn check_lbfgs_history_with_limit(
    meta: &CompiledMeta,
    state: &HashMap<String, Tensor<f32>>,
    limit: usize,
) -> Result<(), ModelIoError> {
    let Some(declared) = meta.lbfgs_history_len else {
        return Ok(());
    };
    let mut s_idx = std::collections::HashSet::new();
    let mut y_idx = std::collections::HashSet::new();
    for key in state.keys() {
        match parse_history_key(key) {
            Some((i, true)) => {
                s_idx.insert(i);
            }
            Some((i, false)) => {
                y_idx.insert(i);
            }
            None => {}
        }
    }
    let actual = s_idx.len().max(y_idx.len());
    check_lbfgs_limit_with(actual, limit, "Lbfgs 履歴件数")?;
    if s_idx.len() != y_idx.len() || actual != declared {
        return Err(ModelIoError::Mismatch {
            message: format!(
                "Lbfgs の history_len {declared} が safetensors の履歴キー数（s: {}・y: {}）と一致しません",
                s_idx.len(),
                y_idx.len()
            ),
        });
    }
    Ok(())
}

/// `state.<i>.<buf>` の各バッファがモデルのパラメータ `i` と同 shape で、添字集合が空か
/// `0..params.len()` と一致することを確認する。optimizer の `load_state_dict` はスロット内の
/// 整合しか見ないため、パラメータ数・shape の食い違い（load は成功するが最初の fit で落ちる
/// 状態）をここで fail-closed に拒否する。非正規表記の添字は optimizer 側が余剰キーとして拒否する。
pub(super) fn check_slot_shapes(
    state: &HashMap<String, Tensor<f32>>,
    params: &[&Tensor<f32>],
) -> Result<(), ModelIoError> {
    let mismatch = |message: String| ModelIoError::Mismatch { message };
    let mut seen = vec![false; params.len()];
    for (key, tensor) in state {
        let Some(rest) = key.strip_prefix("state.") else {
            continue;
        };
        let Some((idx, _buf)) = rest.split_once('.') else {
            continue;
        };
        let Ok(i) = idx.parse::<usize>() else {
            continue;
        };
        if i.to_string() != idx {
            continue;
        }
        let Some(param) = params.get(i) else {
            return Err(mismatch(format!(
                "optimizer 状態のスロット添字 {i} がモデルのパラメータ数 {} を超えています",
                params.len()
            )));
        };
        if tensor.shape() != param.shape() {
            return Err(mismatch(format!(
                "optimizer 状態 {} の shape がパラメータ {i} と一致しません",
                clip(key)
            )));
        }
        seen[i] = true;
    }
    let any = seen.iter().any(|s| *s);
    if any && !seen.iter().all(|s| *s) {
        return Err(mismatch(
            "optimizer 状態のスロット数がモデルのパラメータ数と一致しません".into(),
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::super::{Parser, prepare_save};
    use super::*;
    use crate::compat::{AmpConfig, FitConfig, Sequential};
    use crate::optim::LbfgsConfig;

    fn parse(text: &str) -> Result<Option<CompiledMeta>, ModelIoError> {
        let root = Parser {
            src: text.as_bytes(),
            pos: 0,
        }
        .document()?;
        parse_compiled(&root)
    }

    fn all_optimizers() -> Vec<Optimizer> {
        vec![
            Optimizer::Sgd(SgdConfig {
                lr: 0.1,
                momentum: 0.9,
                dampening: 0.0,
                weight_decay: 1e-4,
                nesterov: true,
            }),
            Optimizer::AdamW(AdamWConfig::default()),
            Optimizer::Adam(AdamConfig {
                lr: 0.002,
                beta1: 0.8,
                beta2: 0.99,
                eps: 1e-6,
                weight_decay: 0.0,
            }),
            Optimizer::RmsProp(RmsPropConfig {
                lr: 0.01,
                alpha: 0.95,
                eps: 1e-8,
                weight_decay: 0.0,
                momentum: 0.5,
                centered: true,
            }),
            Optimizer::Adagrad(AdagradConfig {
                lr: 0.1,
                lr_decay: 0.01,
                weight_decay: 0.0,
                initial_accumulator_value: 0.25,
                eps: 1e-10,
            }),
            Optimizer::Lamb(LambConfig {
                lr: 0.003,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-6,
                weight_decay: 0.01,
            }),
            Optimizer::Lbfgs(LbfgsConfig {
                lr: 0.1,
                max_iter: 4,
                max_eval: None,
                tolerance_grad: 1e-7,
                tolerance_change: 1e-9,
                history_size: 3,
                line_search: LbfgsLineSearch::None,
                line_search_steps: 25,
            }),
            Optimizer::Lbfgs(LbfgsConfig {
                lr: 0.5,
                max_iter: 2,
                max_eval: Some(7),
                tolerance_grad: 1e-6,
                tolerance_change: 1e-8,
                history_size: MAX_LBFGS_HISTORY,
                line_search: LbfgsLineSearch::StrongWolfe,
                line_search_steps: 9,
            }),
            // イシュー #3839。添字 6 を Lbfgs とする既存テストのため末尾に置く。
            Optimizer::Adadelta(AdadeltaConfig {
                lr: 0.9,
                rho: 0.8,
                eps: 1e-5,
                weight_decay: 0.01,
            }),
            Optimizer::Adamax(AdamaxConfig {
                lr: 0.003,
                beta1: 0.8,
                beta2: 0.99,
                eps: 1e-7,
                weight_decay: 0.01,
            }),
        ]
    }

    fn meta(optimizer: Optimizer, amp: Option<AmpSnapshot>) -> CompiledMeta {
        let lbfgs_history_len = matches!(optimizer, Optimizer::Lbfgs(_)).then_some(3);
        CompiledMeta {
            lbfgs_history_len,
            loss: Loss::CrossEntropy,
            optimizer,
            state_keys: vec![
                "optimizer.__optimizer__.x".into(),
                "optimizer.num_slots.u64_u16x4".into(),
            ],
            amp,
            param_groups: Vec::new(),
        }
    }

    fn sample_amp() -> AmpSnapshot {
        AmpSnapshot {
            dtype: AmpDType::Bf16,
            grad_scaler_config: GradScalerConfig {
                init_scale: 1.0e10,
                growth_factor: 2.0,
                backoff_factor: 0.25,
                growth_interval: 7,
            },
            scale: 12345.5,
            growth_tracker: 6,
        }
    }

    #[test]
    fn render_then_parse_round_trips_every_optimizer_and_amp() {
        for opt in all_optimizers() {
            for with_amp in [false, true] {
                if with_amp && matches!(opt, Optimizer::Lbfgs(_)) {
                    continue; // Lbfgs と AMP は併用不可（別テストで拒否を確認）
                }
                let m = meta(opt, with_amp.then(sample_amp));
                let text = render_compiled(&m);
                let parsed = parse(&text).expect("読めるはず").expect("object のはず");
                assert_eq!(render_compiled(&parsed), text);
                assert_eq!(parsed.optimizer, opt);
                assert_eq!(parsed.loss, Loss::CrossEntropy);
                assert_eq!(parsed.amp.is_some(), with_amp);
                if let Some(a) = parsed.amp {
                    let want = sample_amp();
                    assert_eq!(a.scale.to_bits(), want.scale.to_bits());
                    assert_eq!(a.growth_tracker, 6);
                    assert_eq!(a.grad_scaler_config, want.grad_scaler_config);
                }
            }
        }
        assert!(parse("null").expect("null").is_none());
    }

    #[test]
    fn parse_compiled_rejects_schema_violations() {
        let good = render_compiled(&meta(all_optimizers()[1], None));
        let is_manifest = |t: &str| matches!(parse(t), Err(ModelIoError::Manifest { .. }));
        let is_unsupported =
            |t: &str| matches!(parse(t), Err(ModelIoError::UnsupportedModel { .. }));
        assert!(parse(&good).is_ok());
        assert!(is_manifest(&good.replace("\"loss\":", "\"los\":")));
        assert!(is_manifest(
            &good.replace("\"amp\":null", "\"amp\":null,\"x\":1")
        ));
        assert!(is_manifest(&good.replace("\"eps\":", "\"e\":")));
        assert!(is_manifest(
            &good.replace("\"optimizer.num_slots.u64_u16x4\"", "\"num_slots\"")
        ));
        assert!(is_manifest(&good.replace(
            "\"optimizer_state_keys\":[",
            "\"optimizer_state_keys\":[1,"
        )));
        assert!(is_unsupported(&good.replace("cross_entropy", "x")));
        // adamw の config のまま lbfgs にすると config のキー集合・history_len が合わず Manifest。
        assert!(is_manifest(&good.replace("\"adamw\"", "\"lbfgs\"")));
        assert!(is_unsupported(&good.replace("\"adamw\"", "\"nadam\"")));
        // 重複キー
        let dup = good.replace(
            "\"optimizer.num_slots.u64_u16x4\"",
            "\"optimizer.__optimizer__.x\"",
        );
        assert!(is_manifest(&dup));
    }

    fn scaled_model(steps: usize, seed: u64) -> Sequential {
        let mut m = Sequential::new()
            .add_linear(2, 2, seed)
            .and_then(|m| m.add_relu().add_linear(2, 1, seed))
            .expect("構築");
        m.compile_with_amp(
            Optimizer::Adam(AdamConfig::default()),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16).grad_scaler(GradScalerConfig {
                init_scale: 3.0e38,
                growth_factor: 2.0,
                backoff_factor: 0.5,
                growth_interval: 3,
            }),
        )
        .expect("compile_with_amp");
        let x = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]).expect("x");
        let y = Tensor::new(vec![1.0, 0.0, 1.0], &[3, 1]).expect("y");
        for _ in 0..steps {
            m.fit(&x, &y, FitConfig::new(1, 3)).expect("fit");
        }
        m
    }

    /// `growth_tracker` を直接観測できる層で、保存経路の写しから復元した状態が
    /// （tracker を含めて）元の snapshot と完全一致する（AC2 の補強）。
    #[test]
    fn snapshot_restore_preserves_scale_and_growth_tracker_for_every_phase() {
        let mut seen_nonzero_tracker = false;
        for steps in 0..14 {
            let m = scaled_model(steps, 1);
            let snap = m.snapshot_compiled().expect("snapshot").expect("compiled");
            let a = snap.amp.as_ref().expect("amp");
            seen_nonzero_tracker |= a.growth_tracker != 0;
            let (scale, tracker) = (a.scale, a.growth_tracker);

            let meta = CompiledMeta::from_snapshot(&snap).expect("meta");
            let parsed = parse(&render_compiled(&meta))
                .expect("parse")
                .expect("some");
            let mut restored = Sequential::new()
                .add_linear(2, 2, 9)
                .and_then(|m| m.add_relu().add_linear(2, 1, 9))
                .expect("構築");
            restored
                .restore_compiled(parsed.into_snapshot(snap.optimizer_state))
                .expect("restore");
            let again = restored
                .snapshot_compiled()
                .expect("snapshot")
                .expect("compiled");
            let b = again.amp.expect("amp");
            assert_eq!(b.scale.to_bits(), scale.to_bits(), "steps={steps}");
            assert_eq!(b.growth_tracker, tracker, "steps={steps}");
        }
        assert!(
            seen_nonzero_tracker,
            "tracker が 0 以外になる位相を含むはず"
        );
    }

    #[test]
    fn prepare_save_accepts_lbfgs_and_reports_too_large_for_huge_optimizer_state() {
        use crate::optim::LbfgsConfig;
        let mut m = Sequential::new().add_linear(1, 1, 0).expect("構築");
        m.compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
            .expect("compile");
        assert!(prepare_save(&m).is_ok());
        // history_size が固定上限を 1 超えると保存できない（compile 自体は成功する）。
        let mut over = Sequential::new().add_linear(1, 1, 0).expect("構築");
        over.compile(
            Optimizer::Lbfgs(LbfgsConfig {
                history_size: MAX_LBFGS_HISTORY + 1,
                ..LbfgsConfig::default()
            }),
            Loss::Mse,
        )
        .expect("compile");
        assert!(matches!(
            prepare_save(&over),
            Err(ModelIoError::TooLarge {
                what: "Lbfgs history_size",
                limit: 65536
            })
        ));

        // 3000 層 × (weight, bias) × AdamW の (m, v) = 12000 個の状態キーは配列長上限を超える
        // （パラメータ 6000 個は上限内）。書き込み前に TooLarge で拒否される。
        let mut big = Sequential::new();
        for _ in 0..3000 {
            big = big.add_linear(1, 1, 0).expect("add_linear");
        }
        big.compile(Optimizer::AdamW(AdamWConfig::default()), Loss::Mse)
            .expect("compile");
        let x = Tensor::new(vec![0.5, 0.25], &[2, 1]).expect("x");
        let y = Tensor::new(vec![0.1, 0.2], &[2, 1]).expect("y");
        big.fit(&x, &y, FitConfig::new(1, 2)).expect("fit");
        assert!(matches!(
            prepare_save(&big),
            Err(ModelIoError::TooLarge { .. })
        ));
    }

    /// compile・fit 後に層を足すとパラメータ数が optimizer のスロット数を超える。
    /// load が Mismatch で拒否するため、保存側も書き込み前に同じ Mismatch で拒否する。
    #[test]
    fn prepare_save_rejects_optimizer_slots_inconsistent_with_parameters() {
        let mut m = Sequential::new().add_linear(2, 2, 1).expect("構築");
        m.compile(Optimizer::AdamW(AdamWConfig::default()), Loss::Mse)
            .expect("compile");
        let x = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]).expect("x");
        let y = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], &[2, 2]).expect("y");
        m.fit(&x, &y, FitConfig::new(1, 2)).expect("fit");
        assert!(prepare_save(&m).is_ok());
        let m = m.add_linear(2, 2, 2).expect("add_linear");
        assert!(matches!(
            prepare_save(&m),
            Err(ModelIoError::Mismatch { .. })
        ));
    }

    fn lbfgs_model(cfg: LbfgsConfig) -> Sequential {
        let mut m = Sequential::new()
            .add_linear(2, 2, 1)
            .and_then(|m| m.add_relu().add_linear(2, 1, 1))
            .expect("構築");
        m.compile(Optimizer::Lbfgs(cfg), Loss::Mse)
            .expect("compile");
        m
    }

    fn fit_full_batch(m: &mut Sequential, epochs: usize) -> Vec<u32> {
        let x = Tensor::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &[3, 2]).expect("x");
        let y = Tensor::new(vec![1.0, 0.0, 1.0], &[3, 1]).expect("y");
        let h = m.fit(&x, &y, FitConfig::new(epochs, 3)).expect("fit");
        h.loss.iter().map(|v| v.to_bits()).collect()
    }

    /// `StrongWolfe`（facade 未公開の variant）を含む Lbfgs が、snapshot → 描画 → パース →
    /// 復元を通って、続く fit と bit 一致する。
    #[test]
    fn strong_wolfe_lbfgs_round_trips_through_snapshot_and_restore() {
        let cfg = LbfgsConfig {
            lr: 0.5,
            max_iter: 3,
            max_eval: Some(8),
            history_size: 4,
            line_search: LbfgsLineSearch::StrongWolfe,
            ..LbfgsConfig::default()
        };
        let mut a = lbfgs_model(cfg);
        fit_full_batch(&mut a, 2);
        let snap = a.snapshot_compiled().expect("snapshot").expect("compiled");
        assert!(snap.lbfgs_history_len.is_some());
        let meta = CompiledMeta::from_snapshot(&snap).expect("meta");
        let text = render_compiled(&meta);
        assert!(text.contains("\"line_search\":\"strong_wolfe\""), "{text}");
        let parsed = parse(&text).expect("parse").expect("some");
        assert_eq!(parsed.optimizer, Optimizer::Lbfgs(cfg));
        check_lbfgs_history(&parsed, &snap.optimizer_state).expect("履歴件数");

        let mut b = Sequential::new()
            .add_linear(2, 2, 9)
            .and_then(|m| m.add_relu().add_linear(2, 1, 9))
            .expect("構築");
        b.load_state_dict(a.state_dict()).expect("params");
        b.restore_compiled(parsed.into_snapshot(snap.optimizer_state))
            .expect("restore");
        assert_eq!(fit_full_batch(&mut a, 2), fit_full_batch(&mut b, 2));
    }

    /// `restore_compiled` の多層防御: 履歴件数の有無・AMP 併用の不整合は変更なしで拒否する。
    #[test]
    fn restore_compiled_rejects_inconsistent_lbfgs_snapshots() {
        let mut a = lbfgs_model(LbfgsConfig::default());
        fit_full_batch(&mut a, 1);
        let mk = || a.snapshot_compiled().expect("snapshot").expect("compiled");
        let mut target = lbfgs_model(LbfgsConfig::default());

        let mut no_len = mk();
        no_len.lbfgs_history_len = None;
        assert!(target.restore_compiled(no_len).is_err());

        let mut stray_len = mk();
        stray_len.optimizer = Optimizer::Sgd(SgdConfig {
            lr: 0.1,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        });
        assert!(target.restore_compiled(stray_len).is_err());

        let mut with_amp = mk();
        with_amp.amp = Some(sample_amp());
        assert!(target.restore_compiled(with_amp).is_err());

        // 失敗しても既存の compile 状態は変わらない（Lbfgs のまま）。
        let after = target.snapshot_compiled().expect("snapshot").expect("some");
        assert!(matches!(after.optimizer, Optimizer::Lbfgs(_)));
    }

    fn lbfgs_meta(history_len: usize) -> CompiledMeta {
        let mut m = meta(all_optimizers()[6], None);
        m.lbfgs_history_len = Some(history_len);
        m
    }

    #[test]
    fn parse_enforces_lbfgs_limits_and_shape() {
        let good = render_compiled(&lbfgs_meta(3));
        assert!(parse(&good).is_ok());
        let tl = |t: String| matches!(parse(&t), Err(ModelIoError::TooLarge { limit: 65536, .. }));
        let is_manifest = |t: &str| matches!(parse(t), Err(ModelIoError::Manifest { .. }));
        // 境界: 65536 は受理・65537 は TooLarge（history_len・history_size とも）。
        assert!(parse(&good.replace("\"history_len\":3", "\"history_len\":65536")).is_ok());
        assert!(tl(
            good.replace("\"history_len\":3", "\"history_len\":65537")
        ));
        assert!(tl(
            good.replace("\"history_size\":3", "\"history_size\":65537")
        ));
        assert!(is_manifest(&good.replace(",\"history_len\":3", "")));
        assert!(is_manifest(
            &good.replace("\"history_len\":3", "\"history_len\":-1")
        ));
        assert!(is_manifest(
            &good.replace("\"line_search\":\"none\"", "\"line_search\":1")
        ));
        assert!(matches!(
            parse(&good.replace("\"none\"", "\"bogus\"")),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
        assert!(is_manifest(
            &good.replace("\"max_eval\":null", "\"max_eval\":1.5")
        ));
        assert!(is_manifest(
            &good.replace("\"max_eval\":null", "\"max_eval\":\"x\"")
        ));
        // 非 lbfgs に history_len を足すと未知キー。
        let adamw = render_compiled(&meta(all_optimizers()[1], None));
        assert!(is_manifest(
            &adamw.replace("\"config\":", "\"history_len\":0,\"config\":")
        ));
        // Lbfgs と AMP の併用（manifest 経由の迂回）は拒否する。
        let with_amp = good.replace(
            "\"amp\":null",
            "\"amp\":{\"dtype\":\"f16\",\"grad_scaler_config\":{\"init_scale\":1.0,\"growth_factor\":2.0,\"backoff_factor\":0.5,\"growth_interval\":3},\"scale\":1.0,\"growth_tracker\":0}",
        );
        assert!(matches!(
            parse(&with_amp),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
    }

    #[test]
    fn check_lbfgs_history_counts_canonical_keys_only() {
        let t = || Tensor::new(vec![0.0], &[1]).expect("t");
        let mut state = HashMap::new();
        for i in 0..3 {
            state.insert(format!("history.{i}.s"), t());
            state.insert(format!("history.{i}.y"), t());
        }
        // 非正規表記・無関係なキーは数えない。
        state.insert("history.01.s".into(), t());
        state.insert("history.+1.y".into(), t());
        state.insert("history.rho".into(), t());
        assert!(check_lbfgs_history(&lbfgs_meta(3), &state).is_ok());
        assert!(matches!(
            check_lbfgs_history(&lbfgs_meta(2), &state),
            Err(ModelIoError::Mismatch { .. })
        ));
        assert!(matches!(
            check_lbfgs_history(&lbfgs_meta(4), &state),
            Err(ModelIoError::Mismatch { .. })
        ));
        // s と y が不揃い。
        state.remove("history.2.y");
        assert!(matches!(
            check_lbfgs_history(&lbfgs_meta(3), &state),
            Err(ModelIoError::Mismatch { .. })
        ));
        // 上限: 境界ちょうどは受理・超過は TooLarge（限界値を差し替えて検証）。
        let mut m = lbfgs_meta(3);
        m.lbfgs_history_len = Some(2);
        let mut two = HashMap::new();
        for i in 0..2 {
            two.insert(format!("history.{i}.s"), t());
            two.insert(format!("history.{i}.y"), t());
        }
        assert!(check_lbfgs_history_with_limit(&m, &two, 2).is_ok());
        assert!(matches!(
            check_lbfgs_history_with_limit(&m, &two, 1),
            Err(ModelIoError::TooLarge { .. })
        ));
        // Lbfgs 以外は何も見ない。
        assert!(check_lbfgs_history(&meta(all_optimizers()[0], None), &two).is_ok());
    }

    fn groups_meta(groups: Vec<ParamGroup>) -> CompiledMeta {
        let mut m = meta(all_optimizers()[0], None);
        m.param_groups = groups;
        m
    }

    #[test]
    fn param_groups_render_parse_round_trip_and_empty_has_no_key() {
        // 空 groups: キーを書かず、固定文字列の末尾は従来どおり `"amp":null}`。
        let plain = render_compiled(&groups_meta(Vec::new()));
        assert!(!plain.contains("param_groups"));
        assert!(plain.ends_with("\"amp\":null}"));
        assert!(
            parse(&plain)
                .expect("parse")
                .expect("some")
                .param_groups
                .is_empty()
        );

        // 複数 group・group 内の非昇順 params・-0.0 を含む。
        let groups = vec![
            ParamGroup::new(vec![3, 0], 0.01, -0.0),
            ParamGroup::new(vec![2], 0.0, 0.5),
        ];
        let text = render_compiled(&groups_meta(groups.clone()));
        assert!(text.ends_with(
            ",\"param_groups\":{\"lr\":[0.01,0.0],\"weight_decay\":[-0.0,0.5],\"offsets\":[0,2,3],\"slots\":[3,0,2]}}"
        ));
        let back = parse(&text).expect("parse").expect("some");
        assert_eq!(back.param_groups, groups);
        assert_eq!(render_compiled(&back), text);
    }

    #[test]
    fn param_groups_parse_rejects_basic_malformations() {
        let good = render_compiled(&groups_meta(vec![
            ParamGroup::new(vec![0, 1], 0.1, 0.0),
            ParamGroup::new(vec![2], 0.2, 0.0),
        ]));
        let is_manifest = |t: &str| matches!(parse(t), Err(ModelIoError::Manifest { .. }));
        assert!(parse(&good).is_ok());
        // 未知キー・offsets の長さ不整合・重複スロット。
        assert!(is_manifest(&good.replace("\"slots\"", "\"slotz\"")));
        assert!(is_manifest(
            &good.replace("\"offsets\":[0,2,3]", "\"offsets\":[0,2]")
        ));
        assert!(is_manifest(
            &good.replace("\"slots\":[0,1,2]", "\"slots\":[0,1,1]")
        ));
    }

    #[test]
    fn param_groups_with_lbfgs_is_unsupported_both_ways() {
        let mut m = lbfgs_meta(3);
        m.param_groups = vec![ParamGroup::new(vec![0], 0.1, 0.0)];
        let text = render_compiled(&m);
        assert!(matches!(
            parse(&text),
            Err(ModelIoError::UnsupportedModel { .. })
        ));
    }

    #[test]
    fn check_param_group_slots_rejects_out_of_range_as_mismatch() {
        let g = vec![ParamGroup::new(vec![0, 3], 0.1, 0.0)];
        assert!(check_param_group_slots(&g, 4).is_ok());
        assert!(matches!(
            check_param_group_slots(&g, 3),
            Err(ModelIoError::Mismatch { .. })
        ));
        assert!(check_param_group_slots(&[], 0).is_ok());
    }

    #[test]
    fn check_slot_shapes_rejects_count_and_shape_mismatch() {
        let p0 = Tensor::new(vec![0.0; 4], &[2, 2]).expect("p0");
        let p1 = Tensor::new(vec![0.0; 2], &[2]).expect("p1");
        let params = [&p0, &p1];
        let t = |n: usize| Tensor::new(vec![0.0; n], &[n]).expect("t");
        let mut ok = HashMap::new();
        ok.insert(
            "state.0.m".to_string(),
            Tensor::new(vec![0.0; 4], &[2, 2]).expect("s"),
        );
        ok.insert("state.1.m".to_string(), t(2));
        assert!(check_slot_shapes(&ok, &params).is_ok());
        assert!(check_slot_shapes(&HashMap::new(), &params).is_ok());

        let mut short = ok.clone();
        short.remove("state.1.m");
        assert!(check_slot_shapes(&short, &params).is_err());
        let mut extra = ok.clone();
        extra.insert("state.2.m".to_string(), t(1));
        assert!(check_slot_shapes(&extra, &params).is_err());
        let mut wrong = ok.clone();
        wrong.insert("state.1.m".to_string(), t(3));
        assert!(check_slot_shapes(&wrong, &params).is_err());
    }
}
