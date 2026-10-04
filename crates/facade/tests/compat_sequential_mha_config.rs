//! `compat::Sequential::add_multihead_attention_with_config` と
//! `compat::MultiheadAttentionConfig` の facade 統合テスト
//! （イシュー #2530・ルート #2499 の一括承認。承認記録は
//! `docs/autodiff-mha-options-decision.md`）。
//!
//! 検証する契約:
//! - 既定 config は `add_multihead_attention` と同 seed で `state_dict`・`predict` が bit 同一
//! - `batch_first=false` は `[L,B,E]` 入力を `[B,L,E]` 経路と同じ重みで処理する
//!   （統一複合判定: 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。tolerance は不変）
//! - `bias=false` は学習（`compile`→`fit`）・`apply_parameters`・`trainable_grads` に結線される
//! - kdim/vdim 非既定・0 次元・割り切れない値は `InvalidArgument`（fail-closed）
//! - resident 経路は `Unsupported`・AMP（低精度 forward）の非既定 config は型付きエラー
//! - `save_model`→`load_model` の往復と、旧 kind `multihead_attention` の manifest 不変
//! - 新 kind のスキーマ違反は `ModelIoError::Manifest`

use fandhe_ai::compat::{
    AmpConfig, AmpDType, FitConfig, Loss, ModelIoError, MultiheadAttentionConfig, Optimizer,
    Sequential, load_model, save_model,
};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, BackendError, Tensor};
use fandhe_ai_backend_cpu::parity::compare;

mod common;
use common::temp_dir::TempDirGuard;

const E: usize = 4;
const H: usize = 2;
const SEED: u64 = 7;

fn data(n: usize, base: f32) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.37 + base).sin()).collect()
}

fn tensor(shape: &[usize], base: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(data(n, base), shape).expect("test fixture: shape とデータ長は一致")
}

fn flat(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().expect("連続のはず").to_vec()
}

fn assert_bit_identical(a: &Tensor<f32>, b: &Tensor<f32>, what: &str) {
    assert_eq!(a.shape(), b.shape(), "{what}: shape");
    let (fa, fb) = (flat(a), flat(b));
    for (x, y) in fa.iter().zip(fb.iter()) {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}");
    }
}

fn cfg() -> MultiheadAttentionConfig {
    MultiheadAttentionConfig::new(E, H)
}

fn build(c: MultiheadAttentionConfig) -> Sequential {
    Sequential::new()
        .add_multihead_attention_with_config(c, SEED)
        .expect("test fixture: 構築できる config")
}

#[test]
fn default_config_matches_add_multihead_attention_bit_identically() {
    let legacy = Sequential::new()
        .add_multihead_attention(E, H, SEED)
        .expect("構築できる");
    let via_cfg = build(cfg());
    let (sa, sb) = (legacy.state_dict(), via_cfg.state_dict());
    assert_eq!(sa.len(), sb.len());
    for (k, v) in &sa {
        assert_bit_identical(v, sb.get(k).expect("同じキー集合"), k);
    }
    let x = tensor(&[2, 3, E], 0.5);
    assert_bit_identical(
        &legacy.predict(&x).expect("predict"),
        &via_cfg.predict(&x).expect("predict"),
        "predict",
    );
}

#[test]
fn batch_first_false_matches_batch_first_true_on_transposed_input() {
    let (l, b) = (3usize, 2usize);
    // [L, B, E] と、同じ要素を [B, L, E] へ並べ替えた入力。
    let lbe = data(l * b * E, 1.5);
    let mut ble = vec![0.0f32; l * b * E];
    for li in 0..l {
        for bi in 0..b {
            for ei in 0..E {
                ble[(bi * l + li) * E + ei] = lbe[(li * b + bi) * E + ei];
            }
        }
    }
    let first_false = build(cfg().with_batch_first(false));
    let first_true = build(cfg());
    let out_lbe = flat(
        &first_false
            .predict(&Tensor::new(lbe, &[l, b, E]).expect("shape"))
            .expect("predict"),
    );
    let out_ble = flat(
        &first_true
            .predict(&Tensor::new(ble, &[b, l, E]).expect("shape"))
            .expect("predict"),
    );
    let mut back = vec![0.0f32; l * b * E];
    for li in 0..l {
        for bi in 0..b {
            for ei in 0..E {
                back[(li * b + bi) * E + ei] = out_ble[(bi * l + li) * E + ei];
            }
        }
    }
    let report = compare(&out_lbe, &back).expect("同長");
    assert_eq!(report.fail_count, 0, "統一複合判定に不合格: {report:?}");
}

#[test]
fn bias_false_is_wired_through_training_paths() {
    let mut model = Sequential::new()
        .add_multihead_attention_with_config(cfg().with_bias(false), SEED)
        .expect("構築")
        .add_flatten(1, 2)
        .add_linear(3 * E, 2, 9)
        .expect("構築");
    // MHA 4 weight + Linear（weight・bias）。
    assert_eq!(model.trainable_parameters().len(), 4 + 2);
    let before: Vec<Vec<f32>> = model
        .trainable_parameters()
        .iter()
        .map(|t| flat(t))
        .collect();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .expect("compile");
    let x = tensor(&[4, 3, E], 0.1);
    let y = tensor(&[4, 2], 0.9);
    let history = model
        .fit(&x, &y, FitConfig::new(2, 4))
        .expect("fit は成功する");
    assert!(history.loss.iter().all(|v| v.is_finite()));
    let after: Vec<Vec<f32>> = model
        .trainable_parameters()
        .iter()
        .map(|t| flat(t))
        .collect();
    assert_eq!(before.len(), after.len());
    assert!(
        before.iter().zip(after.iter()).any(|(a, b)| a != b),
        "fit 後にパラメータが更新される"
    );
    // apply_parameters は同数・同 shape の更新を受け付ける。
    let same: Vec<Tensor<f32>> = model
        .trainable_parameters()
        .iter()
        .map(|t| (*t).clone())
        .collect();
    model.apply_parameters(same).expect("apply_parameters");
    // 更新後も batch_first=true の predict が通る。
    assert_eq!(model.predict(&x).expect("predict").shape(), &[4, 2]);
}

#[test]
fn invalid_configs_are_rejected_with_typed_errors() {
    for bad in [
        cfg().with_kdim(E + 2),
        cfg().with_vdim(E + 2),
        MultiheadAttentionConfig::new(E, 0),
        MultiheadAttentionConfig::new(5, 2),
        MultiheadAttentionConfig::new(0, 1),
    ] {
        let err = Sequential::new()
            .add_multihead_attention_with_config(bad, SEED)
            .err()
            .expect("拒否される");
        assert!(matches!(err, AutodiffError::InvalidArgument(_)), "{err:?}");
    }
}

#[test]
fn resident_path_rejects_config_built_mha() {
    let model = build(cfg().with_bias(false));
    let tape = fandhe_ai::tape();
    let err = model.init_device_param_store(&tape).unwrap_err();
    assert!(matches!(err, BackendError::Unsupported(_)), "{err:?}");
}

#[test]
fn amp_with_non_default_config_fails_closed() {
    let mut model = Sequential::new()
        .add_multihead_attention_with_config(cfg().with_batch_first(false), SEED)
        .expect("構築")
        .add_flatten(1, 2)
        .add_linear(2 * E, 2, 9)
        .expect("構築");
    model
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(0.05)),
            Loss::Mse,
            AmpConfig::new(AmpDType::F16),
        )
        .expect("compile_with_amp");
    let x = tensor(&[3, 2, E], 0.3);
    let y = tensor(&[2, 2], 0.2);
    // batch_first=false の入力は [L, B, E] のため、Flatten 後の行数は B=2。
    let err = model.fit(&x, &y, FitConfig::new(1, 2)).err();
    assert!(
        matches!(err, Some(AutodiffError::InvalidArgument(_))),
        "非既定 config の低精度 forward は黙って成功せず型付きエラー: {err:?}"
    );
}

fn manifest_text(dir: &std::path::Path) -> String {
    std::fs::read_to_string(dir.join("manifest.json")).expect("manifest を読める")
}

#[test]
fn save_load_round_trips_bias_false_and_batch_first_false() {
    let x = tensor(&[3, 2, E], 0.7);
    for c in [cfg().with_bias(false), cfg().with_batch_first(false)] {
        let mut model = build(c);
        model.eval();
        let guard = TempDirGuard::new("mha-config");
        save_model(&model, guard.path()).expect("save");
        assert!(manifest_text(guard.path()).contains("\"multihead_attention_config\""));
        let loaded = load_model(guard.path()).expect("load");
        assert_bit_identical(
            &model.predict(&x).expect("predict"),
            &loaded.predict(&x).expect("predict"),
            "往復後の predict",
        );
    }
}

#[test]
fn legacy_add_multihead_attention_manifest_kind_is_unchanged() {
    let mut model = Sequential::new()
        .add_multihead_attention(E, H, SEED)
        .expect("構築");
    model.eval();
    let guard = TempDirGuard::new("mha-legacy");
    save_model(&model, guard.path()).expect("save");
    let text = manifest_text(guard.path());
    assert!(text.contains("\"multihead_attention\""));
    assert!(!text.contains("multihead_attention_config"));
}

#[test]
fn new_kind_schema_violations_are_manifest_errors() {
    let mut model = build(cfg().with_bias(false));
    model.eval();
    let guard = TempDirGuard::new("mha-schema");
    save_model(&model, guard.path()).expect("save");
    let good = manifest_text(guard.path());
    assert!(good.contains("\"bias\""), "bias キーが書かれる: {good}");
    for (label, from, to) in [
        ("型違い", "\"bias\":false", "\"bias\":1"),
        ("未知キー", "\"bias\":false", "\"bias\":false,\"kdim\":4"),
        ("欠落", "\"bias\":false,", ""),
    ] {
        // 整形差異に備え、空白なしの表記で一致しない場合は空白付きも試す。
        let bad = if good.contains(from) {
            good.replacen(from, to, 1)
        } else {
            good.replacen(&from.replace(':', ": "), &to.replace(':', ": "), 1)
        };
        assert_ne!(bad, good, "{label}: 改変できた");
        let g2 = TempDirGuard::new("mha-schema-bad");
        save_model(&model, g2.path()).expect("save");
        std::fs::write(g2.path().join("manifest.json"), bad).expect("書ける");
        let err = load_model(g2.path()).err().expect("拒否される");
        assert!(
            matches!(err, ModelIoError::Manifest { .. }),
            "{label}: {err:?}"
        );
    }
}
