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
//! 呼び出し元: `model_io.rs` の `prepare_save`・`render_manifest`・`verify_round_trip`・
//! `parse_manifest`・`load_from_dir_with_limits`。
//!
//! `Lbfgs`（#2373）は manifest の `kind` として認識するが、それまでは
//! `ModelIoError::UnsupportedModel` で拒否する（分岐点は `parse_optimizer` と
//! `CompiledMeta::from_snapshot` の 2 か所のみ）。

use std::collections::HashMap;

use super::super::training::{AmpDType, AmpSnapshot, CompiledSnapshot, Loss, Optimizer};
use super::{
    Json, ModelIoError, Params, as_arr, as_f32, as_str, as_u64, clip, exact_fields, manifest_error,
};
use crate::Tensor;
use crate::optim::{
    AdagradConfig, AdamConfig, AdamWConfig, GradScalerConfig, LambConfig, RmsPropConfig, SgdConfig,
};

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
}

impl CompiledMeta {
    /// snapshot から manifest 用の写しを作る。`Lbfgs` は #2373 まで拒否する。
    pub(super) fn from_snapshot(snap: &CompiledSnapshot) -> Result<Self, ModelIoError> {
        if matches!(snap.optimizer, Optimizer::Lbfgs(_)) {
            return Err(ModelIoError::UnsupportedModel {
                reason: "compile 済み Lbfgs の保存は未対応です（イシュー #2373）".into(),
            });
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
            amp: self.amp,
        }
    }
}

fn loss_name(loss: Loss) -> &'static str {
    match loss {
        Loss::Mse => "mse",
        Loss::CrossEntropy => "cross_entropy",
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
        // 保存側は `CompiledMeta::from_snapshot` が先に拒否する。到達しても空の config にする。
        Optimizer::Lbfgs(_) => Vec::new(),
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
    format!(
        "{{\"loss\":\"{}\",\"optimizer\":{{\"kind\":\"{}\",\"config\":{}}},\"optimizer_state_keys\":[{}],\"amp\":{amp}}}",
        loss_name(m.loss),
        optimizer_kind(&m.optimizer),
        render_object(&config_fields(&m.optimizer)),
        keys.join(",")
    )
}

const CONFIG_CTX: &str = "compiled.optimizer.config";
const ADAM_LIKE_KEYS: [&str; 5] = ["lr", "beta1", "beta2", "eps", "weight_decay"];

/// `optimizer` object（`kind` の allowlist と kind ごとの固定キー集合）を読む。
fn parse_optimizer(value: &Json) -> Result<Optimizer, ModelIoError> {
    let f = exact_fields(value, "compiled.optimizer", &["kind", "config"])?;
    let kind = as_str(f[0], "compiled.optimizer.kind")?;
    match kind {
        "sgd" => {
            let p = Params::named(
                CONFIG_CTX,
                f[1],
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
            let p = Params::named(CONFIG_CTX, f[1], &ADAM_LIKE_KEYS)?;
            Ok(Optimizer::AdamW(AdamWConfig {
                lr: p.f32("lr")?,
                beta1: p.f32("beta1")?,
                beta2: p.f32("beta2")?,
                eps: p.f32("eps")?,
                weight_decay: p.f32("weight_decay")?,
            }))
        }
        "adam" => {
            let p = Params::named(CONFIG_CTX, f[1], &ADAM_LIKE_KEYS)?;
            Ok(Optimizer::Adam(AdamConfig {
                lr: p.f32("lr")?,
                beta1: p.f32("beta1")?,
                beta2: p.f32("beta2")?,
                eps: p.f32("eps")?,
                weight_decay: p.f32("weight_decay")?,
            }))
        }
        "lamb" => {
            let p = Params::named(CONFIG_CTX, f[1], &ADAM_LIKE_KEYS)?;
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
                f[1],
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
                f[1],
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
        // #2373 で Lbfgs の保存・復元を実装するまで、認識はするが拒否する（分岐点）。
        "lbfgs" => Err(ModelIoError::UnsupportedModel {
            reason: "optimizer kind lbfgs は未対応です（イシュー #2373）".into(),
        }),
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
    let f = exact_fields(
        value,
        "compiled",
        &["loss", "optimizer", "optimizer_state_keys", "amp"],
    )?;
    let loss = match as_str(f[0], "compiled.loss")? {
        "mse" => Loss::Mse,
        "cross_entropy" => Loss::CrossEntropy,
        other => {
            return Err(ModelIoError::UnsupportedModel {
                reason: format!("未対応の loss {}", clip(other)),
            });
        }
    };
    let optimizer = parse_optimizer(f[1])?;
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
    Ok(Some(CompiledMeta {
        loss,
        optimizer,
        state_keys,
        amp: parse_amp(f[3])?,
    }))
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
        ]
    }

    fn meta(optimizer: Optimizer, amp: Option<AmpSnapshot>) -> CompiledMeta {
        CompiledMeta {
            loss: Loss::CrossEntropy,
            optimizer,
            state_keys: vec![
                "optimizer.__optimizer__.x".into(),
                "optimizer.num_slots.u64_u16x4".into(),
            ],
            amp,
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
        assert!(is_unsupported(&good.replace("\"adamw\"", "\"lbfgs\"")));
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
    fn prepare_save_rejects_lbfgs_and_reports_too_large_for_huge_optimizer_state() {
        use crate::optim::LbfgsConfig;
        let mut m = Sequential::new().add_linear(1, 1, 0).expect("構築");
        m.compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
            .expect("compile");
        assert!(matches!(
            prepare_save(&m),
            Err(ModelIoError::UnsupportedModel { .. })
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
