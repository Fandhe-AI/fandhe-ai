//! `save_model`／`load_model` の compile 状態（loss・optimizer・AMP）の往復の統合テスト
//! （イシュー #2372・親 #2362。受入基準 AC1〜AC3）。
//!
//! 役割: facade の公開 API だけで、6 optimizer × AMP 有無の複数 step → save → load → 次の step が
//! bit 一致すること（AC1）、backoff／growth を経た GradScaler の状態が bit 一致で復元されること
//! （AC2）、種別・config・状態キーの改竄が fail-closed に拒否されること（AC3）を検証する。
//! AC4（`grad_scaler_from_state` が公開面に現れないこと）は `tests/api_surface.rs` が固定する。
//!
//! 決定的にするため Dropout を含めず、`shuffle=false`（既定）で学習する。既存の
//! `compat_sequential_model_io.rs`（未 compile の基本ケース）とは別ファイルに置く。

#![cfg(unix)]

use std::path::Path;

use fandhe_ai::Tensor;
use fandhe_ai::compat::{
    AmpConfig, AmpDType, Callback, FitConfig, Loss, LrSchedule, ModelIoError, Optimizer,
    Sequential, load_model, save_model,
};
use fandhe_ai::interop::safetensors::{
    load_safetensors_f32_from_bytes, save_safetensors_f32_to_bytes,
};
use fandhe_ai::optim::{
    AdagradConfig, AdamConfig, AdamWConfig, GradScalerConfig, LambConfig, RmsPropConfig, SgdConfig,
    StepLr,
};

mod common;
use common::temp_dir::TempDirGuard;

const N: usize = 8;

fn data() -> (Tensor<f32>, Tensor<f32>) {
    let x: Vec<f32> = (0..N * 3)
        .map(|i| ((i as f32) * 0.41 + 0.3).sin())
        .collect();
    let y: Vec<f32> = (0..N * 2)
        .map(|i| ((i as f32) * 0.23 + 1.1).cos())
        .collect();
    (
        Tensor::new(x, &[N, 3]).expect("x"),
        Tensor::new(y, &[N, 2]).expect("y"),
    )
}

fn build() -> Sequential {
    Sequential::new()
        .add_linear(3, 4, 7)
        .and_then(|m| m.add_relu().add_linear(4, 2, 8))
        .expect("構築できるはず")
}

fn optimizers() -> Vec<(&'static str, Optimizer)> {
    vec![
        (
            "sgd",
            Optimizer::Sgd(SgdConfig {
                lr: 0.05,
                momentum: 0.9,
                dampening: 0.1,
                weight_decay: 0.01,
                nesterov: false,
            }),
        ),
        (
            "sgd_nesterov",
            Optimizer::Sgd(SgdConfig {
                lr: 0.05,
                momentum: 0.9,
                dampening: 0.0,
                weight_decay: 0.0,
                nesterov: true,
            }),
        ),
        ("adamw", Optimizer::AdamW(AdamWConfig::default())),
        (
            "adam",
            Optimizer::Adam(AdamConfig {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.99,
                eps: 1e-7,
                weight_decay: 0.001,
            }),
        ),
        (
            "rmsprop",
            Optimizer::RmsProp(RmsPropConfig {
                lr: 0.01,
                alpha: 0.9,
                eps: 1e-8,
                weight_decay: 0.0,
                momentum: 0.9,
                centered: true,
            }),
        ),
        (
            "adagrad",
            Optimizer::Adagrad(AdagradConfig {
                lr: 0.05,
                lr_decay: 0.01,
                weight_decay: 0.0,
                initial_accumulator_value: 0.1,
                eps: 1e-10,
            }),
        ),
        (
            "lamb",
            Optimizer::Lamb(LambConfig {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-6,
                weight_decay: 0.01,
            }),
        ),
    ]
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("連続のはず")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn assert_same_params(a: &Sequential, b: &Sequential, what: &str) {
    let (sa, sb) = (a.state_dict(), b.state_dict());
    assert_eq!(sa.len(), sb.len(), "{what}: 件数");
    for (k, v) in &sa {
        assert_eq!(bits(v), bits(&sb[k]), "{what}: {k} が bit 不一致");
    }
}

fn load(dir: &Path) -> Sequential {
    load_model(dir).unwrap_or_else(|e| panic!("load できるはず: {e}"))
}

fn load_err(dir: &Path) -> ModelIoError {
    match load_model(dir) {
        Ok(_) => panic!("load が成功してはいけない"),
        Err(e) => e,
    }
}

fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// AC1: 6 optimizer（Sgd は 2 構成）× AMP 有無で、複数 step → save → load → 次の step が bit 一致。
#[test]
fn every_optimizer_with_and_without_amp_resumes_bit_identically() {
    let (x, y) = data();
    for (name, opt) in optimizers() {
        for amp in [false, true] {
            let label = format!("{name} amp={amp}");
            let mut m = build();
            if amp {
                m.compile_with_amp(opt, Loss::Mse, AmpConfig::new(AmpDType::F16))
                    .expect("compile_with_amp");
            } else {
                m.compile(opt, Loss::Mse).expect("compile");
            }
            m.fit(&x, &y, FitConfig::new(3, 4)).expect("fit");

            let guard = TempDirGuard::new("ac1");
            let dir = guard.path().join("m");
            save_model(&m, &dir).unwrap_or_else(|e| panic!("{label}: save: {e}"));
            let mut m2 = load(&dir);
            assert!(m2.is_compiled(), "{label}");
            assert_eq!(
                m.amp_loss_scale().map(f32::to_bits),
                m2.amp_loss_scale().map(f32::to_bits)
            );
            assert_same_params(&m, &m2, &label);

            let h1 = m.fit(&x, &y, FitConfig::new(2, 4)).expect("fit 続き");
            let h2 = m2
                .fit(&x, &y, FitConfig::new(2, 4))
                .expect("fit 続き（復元後）");
            assert_eq!(f32_bits(&h1.loss), f32_bits(&h2.loss), "{label}: loss");
            assert_eq!(f32_bits(&h1.lr), f32_bits(&h2.lr), "{label}: lr");
            assert_same_params(&m, &m2, &label);
            assert_eq!(
                m.amp_loss_scale().map(f32::to_bits),
                m2.amp_loss_scale().map(f32::to_bits),
                "{label}: amp scale"
            );
        }
    }
}

/// AC1: 未 step（optimizer 状態が空）の compile 直後も往復できる。
#[test]
fn freshly_compiled_model_round_trips() {
    let (x, y) = data();
    for (name, opt) in optimizers() {
        let mut m = build();
        m.compile(opt, Loss::Mse).expect("compile");
        let guard = TempDirGuard::new("fresh");
        let dir = guard.path().join("m");
        save_model(&m, &dir).unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut m2 = load(&dir);
        let h1 = m.fit(&x, &y, FitConfig::new(2, 4)).expect("fit");
        let h2 = m2.fit(&x, &y, FitConfig::new(2, 4)).expect("fit");
        assert_eq!(f32_bits(&h1.loss), f32_bits(&h2.loss), "{name}");
        assert_same_params(&m, &m2, name);
    }
}

/// CrossEntropy の loss 種別が復元される（int32 ターゲットで fit できる）。
#[test]
fn cross_entropy_loss_kind_is_restored() {
    let (x, _) = data();
    let labels: Vec<i32> = (0..N as i32).map(|i| i % 2).collect();
    let y = Tensor::new(labels, &[N]).expect("labels");
    let mut m = build();
    m.compile(Optimizer::AdamW(AdamWConfig::default()), Loss::CrossEntropy)
        .expect("compile");
    m.fit(&x, &y, FitConfig::new(2, 4)).expect("fit");
    let guard = TempDirGuard::new("ce");
    let dir = guard.path().join("m");
    save_model(&m, &dir).expect("save");
    let mut m2 = load(&dir);
    let h1 = m.fit(&x, &y, FitConfig::new(1, 4)).expect("fit");
    let h2 = m2.fit(&x, &y, FitConfig::new(1, 4)).expect("fit");
    assert_eq!(f32_bits(&h1.loss), f32_bits(&h2.loss));
}

/// AC1: LR scheduler が書き換えた「現在の lr」が config として復元される。
#[test]
fn scheduled_lr_is_restored_as_current_config() {
    let (x, y) = data();
    for (name, opt, base) in [
        (
            "sgd",
            Optimizer::Sgd(SgdConfig {
                lr: 0.1,
                momentum: 0.9,
                dampening: 0.0,
                weight_decay: 0.0,
                nesterov: false,
            }),
            0.1_f32,
        ),
        ("adamw", Optimizer::AdamW(AdamWConfig::default()), 1e-3),
        (
            "adam",
            Optimizer::Adam(AdamConfig {
                lr: 0.02,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.0,
            }),
            0.02,
        ),
    ] {
        let mut m = build();
        m.compile(opt, Loss::Mse).expect("compile");
        let mut cbs = [Callback::LrSchedule(LrSchedule::per_epoch(
            StepLr::new(base, 1, 0.5).expect("StepLr"),
        ))];
        m.fit_with_callbacks(&x, &y, FitConfig::new(3, 4), None, &mut cbs)
            .expect("fit_with_callbacks");

        let guard = TempDirGuard::new("lr");
        let dir = guard.path().join("m");
        save_model(&m, &dir).expect("save");
        let mut m2 = load(&dir);
        let h1 = m.fit(&x, &y, FitConfig::new(1, 4)).expect("fit");
        let h2 = m2.fit(&x, &y, FitConfig::new(1, 4)).expect("fit");
        assert_ne!(
            h1.lr[0].to_bits(),
            base.to_bits(),
            "{name}: lr が scheduler で変わっているはず"
        );
        assert_eq!(f32_bits(&h1.lr), f32_bits(&h2.lr), "{name}: lr");
        assert_eq!(f32_bits(&h1.loss), f32_bits(&h2.loss), "{name}: loss");
        assert_same_params(&m, &m2, name);
    }
}

/// AC2: backoff・growth を繰り返して初期値と異なる（scale, growth_tracker）になった GradScaler が、
/// 保存する step 数によらず（tracker の全位相で）bit 一致で復元される。tracker は統合テストから
/// 直接観測できないため、復元後の scale 軌跡と重みの一致で担保する（tracker の直接照合は
/// `model_io.rs` の単体テスト）。
#[test]
fn grad_scaler_state_after_backoff_and_growth_round_trips() {
    let (x, y) = data();
    let cfg = GradScalerConfig {
        init_scale: 3.0e38,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 3,
    };
    let compile = |m: &mut Sequential| {
        m.compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(0.01)),
            Loss::Mse,
            AmpConfig::new(AmpDType::Bf16).grad_scaler(cfg),
        )
        .expect("compile_with_amp");
    };
    let one_step = FitConfig::new(1, N);

    let mut saw_backoff = false;
    let mut saw_growth = false;
    for pre_steps in 0..14 {
        let mut m = build();
        compile(&mut m);
        let mut prev = m.amp_loss_scale().expect("amp");
        for _ in 0..pre_steps {
            m.fit(&x, &y, one_step).expect("fit");
            let s = m.amp_loss_scale().expect("amp");
            saw_backoff |= s < prev;
            saw_growth |= s > prev;
            prev = s;
        }
        let guard = TempDirGuard::new("ac2");
        let dir = guard.path().join("m");
        save_model(&m, &dir).expect("save");
        let mut m2 = load(&dir);
        assert_eq!(
            m.amp_loss_scale().map(f32::to_bits),
            m2.amp_loss_scale().map(f32::to_bits)
        );
        for step in 0..8 {
            let h1 = m.fit(&x, &y, one_step).expect("fit");
            let h2 = m2.fit(&x, &y, one_step).expect("fit");
            let label = format!("pre={pre_steps} step={step}");
            assert_eq!(f32_bits(&h1.loss), f32_bits(&h2.loss), "{label}: loss");
            assert_eq!(
                m.amp_loss_scale().map(f32::to_bits),
                m2.amp_loss_scale().map(f32::to_bits),
                "{label}: scale"
            );
            assert_same_params(&m, &m2, &label);
        }
    }
    assert!(saw_backoff, "backoff が起きる構成のはず");
    assert!(saw_growth, "growth が起きる構成のはず");
}

// ---------------------------------------------------------------------
// AC3: 改竄の拒否
// ---------------------------------------------------------------------

fn compiled_dir(label: &str, amp: bool) -> (TempDirGuard, std::path::PathBuf) {
    let (x, y) = data();
    let mut m = build();
    let opt = Optimizer::AdamW(AdamWConfig {
        lr: 0.01,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
    });
    if amp {
        m.compile_with_amp(opt, Loss::Mse, AmpConfig::new(AmpDType::F16))
            .expect("compile_with_amp");
    } else {
        m.compile(opt, Loss::Mse).expect("compile");
    }
    m.fit(&x, &y, FitConfig::new(2, 4)).expect("fit");
    let guard = TempDirGuard::new(label);
    let dir = guard.path().join("m");
    save_model(&m, &dir).expect("save");
    (guard, dir)
}

fn read_manifest(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず")
}

fn write_manifest(dir: &Path, text: &str) {
    std::fs::write(dir.join("manifest.json"), text).expect("書けるはず");
}

/// `"key":<value>` の value（次の `,` または `}` まで）を置き換える。
fn set_field(text: &str, key: &str, new_value: &str) -> String {
    let needle = format!("\"{key}\":");
    let start = text.find(&needle).unwrap_or_else(|| panic!("{key} が無い")) + needle.len();
    let end = start + text[start..].find([',', '}']).expect("終端");
    format!("{}{}{}", &text[..start], new_value, &text[end..])
}

fn state_keys(text: &str) -> (usize, usize, Vec<String>) {
    let needle = "\"optimizer_state_keys\":[";
    let start = text.find(needle).expect("keys") + needle.len();
    let end = start + text[start..].find(']').expect("]");
    let keys = text[start..end].split(',').map(str::to_string).collect();
    (start, end, keys)
}

fn with_state_keys(text: &str, f: impl FnOnce(Vec<String>) -> Vec<String>) -> String {
    let (start, end, keys) = state_keys(text);
    format!("{}{}{}", &text[..start], f(keys).join(","), &text[end..])
}

/// safetensors を読み替えて書き戻し、manifest の `safetensors_bytes` も合わせる。
fn rewrite_safetensors(
    dir: &Path,
    edit: impl FnOnce(&mut std::collections::HashMap<String, Tensor<f32>>),
) {
    let text = read_manifest(dir);
    let name_key = "\"safetensors_file\":\"";
    let s = text.find(name_key).expect("file") + name_key.len();
    let name = &text[s..s + text[s..].find('"').expect("\"")];
    let path = dir.join(name);
    let mut map =
        load_safetensors_f32_from_bytes(&std::fs::read(&path).expect("read")).expect("st");
    edit(&mut map);
    let bytes = save_safetensors_f32_to_bytes(&map, None).expect("save st");
    std::fs::write(&path, &bytes).expect("write");
    let old = set_field(&text, "safetensors_bytes", &bytes.len().to_string());
    write_manifest(dir, &old);
}

#[test]
fn tampered_kind_is_rejected() {
    // adamw -> adam: config のキー集合は同じだが、safetensors の種別マーカーが不一致。
    let (_g, dir) = compiled_dir("kind", false);
    let t = read_manifest(&dir).replace("\"kind\":\"adamw\"", "\"kind\":\"adam\"");
    write_manifest(&dir, &t);
    assert!(matches!(load_err(&dir), ModelIoError::Autodiff(_)));

    // adam の config のまま lbfgs にすると config のキー集合と history_len が合わず Manifest
    // （Lbfgs 自体の往復・改竄は compat_sequential_model_io_lbfgs.rs）。
    let t = read_manifest(&dir).replace("\"kind\":\"adam\"", "\"kind\":\"lbfgs\"");
    write_manifest(&dir, &t);
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
    let t = read_manifest(&dir).replace("\"kind\":\"lbfgs\"", "\"kind\":\"adam\"");
    write_manifest(&dir, &t);

    for bad in ["nadam", "SGD", ""] {
        let t = read_manifest(&dir).replace("\"kind\":\"adam\"", &format!("\"kind\":\"{bad}\""));
        write_manifest(&dir, &t);
        assert!(
            matches!(load_err(&dir), ModelIoError::UnsupportedModel { .. }),
            "{bad}"
        );
    }
}

#[test]
fn tampered_loss_and_dtype_are_rejected() {
    let (_g, dir) = compiled_dir("loss", true);
    let good = read_manifest(&dir);
    write_manifest(
        &dir,
        &good.replace("\"loss\":\"mse\"", "\"loss\":\"huber\""),
    );
    assert!(matches!(
        load_err(&dir),
        ModelIoError::UnsupportedModel { .. }
    ));
    write_manifest(
        &dir,
        &good.replace("\"dtype\":\"f16\"", "\"dtype\":\"f64\""),
    );
    assert!(matches!(
        load_err(&dir),
        ModelIoError::UnsupportedModel { .. }
    ));
}

#[test]
fn tampered_config_is_rejected() {
    let (_g, dir) = compiled_dir("config", false);
    let good = read_manifest(&dir);
    // 未知キー・欠落キー
    write_manifest(&dir, &good.replace("\"eps\":", "\"epsilon\":"));
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
    write_manifest(&dir, &good.replace("\"lr\":0.01,", ""));
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
    // 非正準の f32 表記・整数の欄に Real／Real の欄に整数
    write_manifest(&dir, &set_field(&good, "lr", "0.010"));
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
    write_manifest(&dir, &set_field(&good, "lr", "1"));
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
    // 範囲外の値は optimizer のコンストラクタが拒否する
    write_manifest(&dir, &set_field(&good, "lr", "-0.5"));
    assert!(matches!(load_err(&dir), ModelIoError::Autodiff(_)));
    write_manifest(&dir, &set_field(&good, "beta1", "1.5"));
    assert!(matches!(load_err(&dir), ModelIoError::Autodiff(_)));
}

#[test]
fn tampered_state_keys_are_rejected() {
    let (_g, dir) = compiled_dir("keys", false);
    let good = read_manifest(&dir);
    // 欠落（末尾を 1 件削除）・余剰・重複・順序違い
    write_manifest(
        &dir,
        &with_state_keys(&good, |mut k| {
            k.pop();
            k
        }),
    );
    assert!(matches!(load_err(&dir), ModelIoError::Mismatch { .. }));
    write_manifest(
        &dir,
        &with_state_keys(&good, |mut k| {
            k.push("\"optimizer.zzz\"".into());
            k
        }),
    );
    assert!(matches!(load_err(&dir), ModelIoError::Mismatch { .. }));
    write_manifest(
        &dir,
        &with_state_keys(&good, |mut k| {
            let f = k[0].clone();
            k.insert(1, f);
            k
        }),
    );
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
    write_manifest(
        &dir,
        &with_state_keys(&good, |mut k| {
            k.reverse();
            k
        }),
    );
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
    write_manifest(
        &dir,
        &with_state_keys(&good, |mut k| {
            k[0] = "\"state.0.m\"".into();
            k
        }),
    );
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
}

#[test]
fn tampered_safetensors_optimizer_entries_are_rejected() {
    // 1 本追加
    let (_g, dir) = compiled_dir("st-add", false);
    rewrite_safetensors(&dir, |m| {
        m.insert(
            "optimizer.extra".into(),
            Tensor::new(vec![0.0], &[1]).expect("t"),
        );
    });
    assert!(matches!(load_err(&dir), ModelIoError::Mismatch { .. }));
    // 1 本削除
    let (_g2, dir2) = compiled_dir("st-del", false);
    rewrite_safetensors(&dir2, |m| {
        let k = m
            .keys()
            .find(|k| k.starts_with("optimizer.state."))
            .cloned()
            .expect("key");
        m.remove(&k);
    });
    assert!(matches!(load_err(&dir2), ModelIoError::Mismatch { .. }));
}

#[test]
fn optimizer_entries_without_compiled_are_rejected() {
    let (x, y) = data();
    let _ = (&x, &y);
    let guard = TempDirGuard::new("nocompiled");
    let dir = guard.path().join("m");
    save_model(&build(), &dir).expect("save");
    rewrite_safetensors(&dir, |m| {
        m.insert(
            "optimizer.num_slots.u64_u16x4".into(),
            Tensor::new(vec![0.0; 4], &[4]).expect("t"),
        );
    });
    assert!(matches!(load_err(&dir), ModelIoError::Mismatch { .. }));
}

#[test]
fn slot_shape_mismatch_with_parameters_is_rejected() {
    let (_g, dir) = compiled_dir("slot-shape", false);
    // スロット 0 の m・v を（スロット内では整合したまま）パラメータと異なる shape にする。
    rewrite_safetensors(&dir, |m| {
        for buf in ["m", "v"] {
            m.insert(
                format!("optimizer.state.0.{buf}"),
                Tensor::new(vec![0.0], &[1]).expect("t"),
            );
        }
    });
    assert!(matches!(load_err(&dir), ModelIoError::Mismatch { .. }));
}

#[test]
fn tampered_amp_state_is_rejected() {
    let (_g, dir) = compiled_dir("amp", true);
    let good = read_manifest(&dir);
    for bad in ["0.0", "-1.0", "1e-40"] {
        write_manifest(&dir, &set_field(&good, "scale", bad));
        assert!(
            matches!(load_err(&dir), ModelIoError::Autodiff(_)),
            "scale={bad}"
        );
    }
    // growth_tracker >= growth_interval（既定 2000）
    write_manifest(&dir, &set_field(&good, "growth_tracker", "2000"));
    assert!(matches!(load_err(&dir), ModelIoError::Autodiff(_)));
    // 非正準の scale
    write_manifest(&dir, &set_field(&good, "scale", "65536.00"));
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
    // amp を持つ節のキー欠落
    write_manifest(&dir, &good.replace("\"growth_tracker\"", "\"tracker\""));
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));
    // 改竄していない manifest は読める（テストの前提確認）
    write_manifest(&dir, &good);
    assert!(load_model(&dir).is_ok());
}
