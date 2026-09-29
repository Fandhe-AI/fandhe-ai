//! `fandhe_ai::compat::{save_model, load_model}` の BatchNorm running stats・train モード復元
//! （イシュー #2371・親 #2362）の統合テスト。facade の公開 API と `std` のみで検証する。
//!
//! - (A1) train モードで学習した BatchNorm1d／2d を含むモデルを save → load し、eval モードの
//!   forward が bit 一致する（stats が初期値のままなら出力が食い違うことも先に確認する）。
//! - (A2) `buffer_keys`・safetensors の buffer キー・shape の過不足／不一致を `Mismatch` で拒否する。
//! - `num_batches_tracked` は復元しない（load 後は 0 から再開）。forward の数値には影響しないため、
//!   load 直後に train を 1 回回した後の eval 出力も一致する。
//!
//! `compile` 済みモデルは #2372 まで保存できないため、学習は手動 SGD ステップ
//! （`compat_sequential_callbacks.rs` と同型）で行う。実機（CUDA／Metal）は不要。

#![cfg(unix)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use fandhe_ai::Tensor;
use fandhe_ai::compat::{ModelIoError, Sequential, load_model, save_model};
use fandhe_ai::interop::safetensors::{load_safetensors_f32, save_safetensors_f32_to_bytes};
use fandhe_ai::optim::{Sgd, SgdConfig};

mod common;
use common::temp_dir::TempDirGuard;

fn tensor(shape: &[usize], base: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37 + base).sin()).collect();
    Tensor::new(data, shape).expect("テンソルを作れるはず")
}

fn assert_bit_identical(a: &Tensor<f32>, b: &Tensor<f32>, what: &str) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    let (ca, cb) = (a.contiguous(), b.contiguous());
    let (sa, sb) = (
        ca.as_slice().expect("連続のはず"),
        cb.as_slice().expect("連続のはず"),
    );
    assert!(
        sa.iter()
            .zip(sb.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits()),
        "{what}: bit 不一致"
    );
}

/// 手動 SGD ステップ（bind → forward → mse_loss → backward → trainable_grads → step →
/// apply_parameters）。train モードの BatchNorm は forward ごとに running stats を更新する。
fn train_steps(model: &mut Sequential, x: &Tensor<f32>, y: &Tensor<f32>, steps: usize) {
    let mut sgd = Sgd::new(SgdConfig::new(0.05)).expect("構築できるはず");
    for _ in 0..steps {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x_var = tape.var(x);
            let y_var = tape.var_no_grad(y);
            let pred = bound.forward(&tape, &x_var).expect("forward できるはず");
            let loss = pred.mse_loss(&y_var).expect("loss を計算できるはず");
            let grads = tape.backward(&loss).expect("backward できるはず");
            let grad_refs = bound.trainable_grads(&grads).expect("勾配を取れるはず");
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).expect("step できるはず")
        };
        model.apply_parameters(updated).expect("反映できるはず");
    }
}

/// state_dict のパラメータのみを新規モデルへ流し込む（running stats は初期値のまま）。
fn params_only_clone(build: fn() -> Sequential, src: &Sequential) -> Sequential {
    let mut fresh = build();
    fresh
        .load_state_dict(src.state_dict())
        .expect("パラメータを流し込めるはず");
    fresh.eval();
    fresh
}

fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("読めるはず")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn only_safetensors(dir: &Path) -> PathBuf {
    let names: Vec<String> = entries(dir)
        .into_iter()
        .filter(|n| n.starts_with("model.") && n.ends_with(".safetensors"))
        .collect();
    assert_eq!(names.len(), 1);
    dir.join(&names[0])
}

fn build_bn1d_rank2() -> Sequential {
    Sequential::new()
        .add_linear(4, 6, 1)
        .and_then(|m| m.add_batch_norm1d(6, 1e-5, 0.1))
        .and_then(|m| m.add_relu().add_linear(6, 2, 2))
        .expect("構築できるはず")
}

fn build_bn1d_rank3() -> Sequential {
    Sequential::new()
        .add_conv1d(2, 3, 1, 1, 0, 1, 1, 3)
        .and_then(|m| m.add_batch_norm1d(3, 1e-5, 0.2))
        .expect("構築できるはず")
}

fn build_bn2d() -> Sequential {
    Sequential::new()
        .add_conv2d(1, 2, [1, 1], [1, 1], [0, 0], [1, 1], 1, 4)
        .and_then(|m| m.add_batch_norm2d(2, 1e-5, 0.1))
        .map(|m| m.add_flatten(1, 3))
        .and_then(|m| m.add_linear(8, 2, 5))
        .expect("構築できるはず")
}

/// 学習 → 保存 → 復元の往復を eval モードの bit 一致で検証する（A1 の共通本体）。
fn assert_round_trip(label: &str, build: fn() -> Sequential, x: &Tensor<f32>, y: &Tensor<f32>) {
    let mut model = build();
    train_steps(&mut model, x, y, 3);
    // train モードの predict も stats を更新する。
    model.predict(x).expect("predict できるはず");
    assert!(model.training());

    // 安全確認: パラメータだけを写したモデルは stats が初期値のため eval 出力が異なる
    // （＝ buffer を保存しなければこのテストが落ちる）。
    let params_only = params_only_clone(build, &model);
    let g = TempDirGuard::new(label);
    let dir = g.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");

    let stored = load_safetensors_f32(&only_safetensors(&dir)).expect("読めるはず");
    let has_nonzero_mean = stored
        .iter()
        .filter(|(k, _)| k.ends_with(".running_mean"))
        .any(|(_, t)| {
            t.contiguous()
                .as_slice()
                .expect("連続のはず")
                .iter()
                .any(|v| *v != 0.0)
        });
    assert!(has_nonzero_mean, "{label}: running_mean が更新されていない");

    let mut loaded = load_model(&dir).expect("復元できるはず");
    assert!(loaded.training(), "{label}: training フラグ");

    // 保存された buffer キーが manifest と safetensors の両方に揃っている。
    let bn_keys: Vec<&String> = stored
        .keys()
        .filter(|k| k.ends_with(".running_mean") || k.ends_with(".running_var"))
        .collect();
    assert!(!bn_keys.is_empty(), "{label}: buffer キーがない");

    model.eval();
    loaded.eval();
    let ya = model.predict(x).expect("predict できるはず");
    let yb = loaded.predict(x).expect("predict できるはず");
    assert_bit_identical(&ya, &yb, &format!("{label}: eval 出力"));
    let y_params_only = params_only.predict(x).expect("predict できるはず");
    let differs = ya
        .contiguous()
        .as_slice()
        .expect("連続のはず")
        .iter()
        .zip(y_params_only.contiguous().as_slice().expect("連続のはず"))
        .any(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        differs,
        "{label}: stats 初期値の eval 出力と同じでは buffer 欠落を検出できない"
    );
}

#[test]
fn bn1d_rank2_trained_model_round_trips_eval_bit_identically() {
    let x = tensor(&[8, 4], 0.3);
    let y = tensor(&[8, 2], 1.1);
    assert_round_trip("bn1d-rank2", build_bn1d_rank2, &x, &y);
}

#[test]
fn bn1d_rank3_and_bn2d_trained_models_round_trip() {
    let x3 = tensor(&[6, 2, 5], 0.2);
    let y3 = tensor(&[6, 3, 5], 0.9);
    assert_round_trip("bn1d-rank3", build_bn1d_rank3, &x3, &y3);
    let x4 = tensor(&[5, 1, 2, 2], 0.4);
    let y4 = tensor(&[5, 2], 0.6);
    assert_round_trip("bn2d", build_bn2d, &x4, &y4);
}

#[test]
fn eval_mode_saved_model_restores_eval_flag_and_stats() {
    let x = tensor(&[8, 4], 0.3);
    let y = tensor(&[8, 2], 1.1);
    let mut model = build_bn1d_rank2();
    train_steps(&mut model, &x, &y, 3);
    model.eval();
    let g = TempDirGuard::new("bn-eval");
    let dir = g.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert!(!loaded.training());
    let ya = model.predict(&x).expect("predict できるはず");
    let yb = loaded.predict(&x).expect("predict できるはず");
    assert_bit_identical(&ya, &yb, "eval 保存の出力");
}

#[test]
fn train_mode_continuation_matches_after_load() {
    // num_batches_tracked を復元しなくても、load 後の train 継続で stats 更新の数値は一致する。
    let x = tensor(&[8, 4], 0.3);
    let y = tensor(&[8, 2], 1.1);
    let mut model = build_bn1d_rank2();
    train_steps(&mut model, &x, &y, 2);
    let g = TempDirGuard::new("bn-continue");
    let dir = g.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let mut loaded = load_model(&dir).expect("復元できるはず");
    let probe = tensor(&[8, 4], 0.77);
    model.predict(&probe).expect("train predict");
    loaded.predict(&probe).expect("train predict");
    model.eval();
    loaded.eval();
    let ya = model.predict(&x).expect("predict できるはず");
    let yb = loaded.predict(&x).expect("predict できるはず");
    assert_bit_identical(&ya, &yb, "継続後の eval 出力");
}

// ---------------------------------------------------------------------
// A2: 拒否
// ---------------------------------------------------------------------

fn try_load(dir: &Path) -> Result<(), ModelIoError> {
    load_model(dir).map(|_| ())
}

fn saved_bn_dir(label: &str) -> (TempDirGuard, PathBuf) {
    let x = tensor(&[8, 4], 0.3);
    let y = tensor(&[8, 2], 1.1);
    let mut model = build_bn1d_rank2();
    train_steps(&mut model, &x, &y, 2);
    let g = TempDirGuard::new(label);
    let dir = g.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    (g, dir)
}

fn manifest_path(dir: &Path) -> PathBuf {
    dir.join("manifest.json")
}

const M1: &str = "{\"key\":\"1.running_mean\",\"shape\":[6]}";
const V1: &str = "{\"key\":\"1.running_var\",\"shape\":[6]}";

#[test]
fn manifest_buffer_keys_tampering_is_rejected() {
    let mismatch: Vec<(&str, String)> = vec![
        ("drop", V1.to_string()),
        ("add", format!("{M1},{V1},{M1}")),
        ("swap", format!("{V1},{M1}")),
        (
            "rename",
            format!("{{\"key\":\"1.running_meen\",\"shape\":[6]}},{V1}"),
        ),
        (
            "shape",
            format!("{{\"key\":\"1.running_mean\",\"shape\":[7]}},{V1}"),
        ),
    ];
    for (label, replacement) in mismatch {
        let (_g, dir) = saved_bn_dir(&format!("bn-tamper-{label}"));
        let before = entries(&dir);
        let path = manifest_path(&dir);
        let text = std::fs::read_to_string(&path).expect("読めるはず");
        let good = format!("{M1},{V1}");
        assert!(text.contains(&good), "{label}: 期待した buffer_keys がない");
        std::fs::write(&path, text.replace(&good, &replacement)).expect("書けるはず");
        let err = try_load(&dir).expect_err("改竄は拒否されるはず");
        assert!(
            matches!(err, ModelIoError::Mismatch { .. }),
            "{label}: {err}"
        );
        assert_eq!(entries(&dir), before, "{label}: dir が変わってはいけない");
    }

    let type_errors: Vec<(&str, String)> = vec![
        (
            "shape-type",
            format!("{{\"key\":\"1.running_mean\",\"shape\":[6.0]}},{V1}"),
        ),
        ("key-type", format!("{{\"key\":1,\"shape\":[6]}},{V1}")),
        (
            "unknown-field",
            format!("{{\"key\":\"1.running_mean\",\"shape\":[6],\"x\":1}},{V1}"),
        ),
    ];
    for (label, replacement) in type_errors {
        let (_g, dir) = saved_bn_dir(&format!("bn-type-{label}"));
        let path = manifest_path(&dir);
        let text = std::fs::read_to_string(&path).expect("読めるはず");
        std::fs::write(&path, text.replace(&format!("{M1},{V1}"), &replacement))
            .expect("書けるはず");
        let err = try_load(&dir).expect_err("型違いは拒否されるはず");
        assert!(
            matches!(err, ModelIoError::Manifest { .. }),
            "{label}: {err}"
        );
    }
}

/// safetensors を差し替え、manifest の `safetensors_bytes` を実長へ合わせる。
fn replace_safetensors(dir: &Path, tensors: &HashMap<String, Tensor<f32>>) {
    let st = only_safetensors(dir);
    let bytes = save_safetensors_f32_to_bytes(tensors, None).expect("書き出せるはず");
    let old_len = std::fs::metadata(&st).expect("stat できるはず").len();
    std::fs::write(&st, &bytes).expect("書けるはず");
    let path = manifest_path(dir);
    let text = std::fs::read_to_string(&path).expect("読めるはず");
    let old = format!("\"safetensors_bytes\":{old_len}");
    assert!(text.contains(&old));
    std::fs::write(
        &path,
        text.replace(&old, &format!("\"safetensors_bytes\":{}", bytes.len())),
    )
    .expect("書けるはず");
}

#[test]
fn safetensors_buffer_key_or_shape_mismatch_is_rejected() {
    type Tamper = fn(&mut HashMap<String, Tensor<f32>>);
    let cases: [(&str, Tamper); 3] = [
        ("missing", |m| {
            m.remove("1.running_var");
        }),
        ("extra", |m| {
            m.insert(
                "1.running_extra".into(),
                Tensor::new(vec![0.0; 6], &[6]).unwrap(),
            );
        }),
        ("shape", |m| {
            m.insert(
                "1.running_var".into(),
                Tensor::new(vec![1.0; 7], &[7]).unwrap(),
            );
        }),
    ];
    for (label, tamper) in cases {
        let (_g, dir) = saved_bn_dir(&format!("bn-st-{label}"));
        let before = entries(&dir);
        let mut tensors = load_safetensors_f32(&only_safetensors(&dir)).expect("読めるはず");
        tamper(&mut tensors);
        replace_safetensors(&dir, &tensors);
        let err = try_load(&dir).expect_err("不整合は拒否されるはず");
        assert!(
            matches!(err, ModelIoError::Mismatch { .. }),
            "{label}: {err}"
        );
        assert_eq!(entries(&dir), before, "{label}: dir が変わってはいけない");
    }
}

#[test]
fn resave_of_loaded_model_keeps_buffers_bit_identical() {
    let (_g, dir) = saved_bn_dir("bn-resave");
    let loaded = load_model(&dir).expect("復元できるはず");
    let g2 = TempDirGuard::new("bn-resave-2");
    let dir2 = g2.path().join("m");
    save_model(&loaded, &dir2).expect("再保存できるはず");
    let a = load_safetensors_f32(&only_safetensors(&dir)).expect("読めるはず");
    let b = load_safetensors_f32(&only_safetensors(&dir2)).expect("読めるはず");
    assert_eq!(a.len(), b.len());
    for (k, v) in &a {
        assert_bit_identical(v, &b[k], k);
    }
}
