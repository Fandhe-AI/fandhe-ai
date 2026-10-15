//! `parametrize_weight_norm`／`parametrize_spectral_norm` で再パラメータ化した層の
//! `save_model`／`load_model` 対応（イシュー #3050・親 #3018。決定記録
//! `docs/facade-weight-reparam-wiring-decision.md` §7・§10・§13・§16〜§18）の統合テスト。
//!
//! facade の公開 API と `std` のみで次の受入基準を固定する。
//!
//! - 14 kind（基底 7 種〈Linear・Conv1d／2d／3d・ConvTranspose1d／2d／3d〉× weight_norm／spectral_norm）
//!   の save → load で `state_dict`・`predict` 出力・`training` が bit 一致する
//!   （spectral_norm は train 中に状態を進めた後の保存。`u`／`v` は buffer として保存し再正規化しない）。
//!   `format_version` は 2 のまま。weight_norm は `dim = Some(0)`・`Some(1)`・`None`（rank 0 の `g`）を含む。
//! - 学習（`fit`）後の optimizer 往復で、復元後の次の学習が元モデルと bit 一致する。
//! - 学習テスト: 損失が減少し、spectral_norm の `u`／`v` は train でだけ進む（保存した safetensors の
//!   buffer を比較する公開 API のみの検証）。
//! - 改竄拒否（`kind`・`params`・`parameter_keys`・`buffer_keys`・safetensors の buffer）と旧形式（v1）での拒否。
//! - save 側 fail-closed（非有限な `u`／`v`・`n_power_iterations` の上限超過）は `dir` に何も作らない。
//!
//! 実機（CUDA／Metal）は不要（ホスト側の I/O と層の再構築のみ）。

#![cfg(unix)]

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use fandhe_ai::compat::{
    FitConfig, Loss, ModelIoError, Optimizer, Sequential, load_model, save_model,
};
use fandhe_ai::interop::safetensors::{load_safetensors_f32, save_safetensors_f32_to_bytes};
use fandhe_ai::optim::{AdamWConfig, SgdConfig};
use fandhe_ai::{AutodiffError, Tensor};

mod common;
use common::temp_dir::TempDirGuard;

const SEED: u64 = 0x5eed_3050;
const N_ITER: usize = 2;
const EPS: f32 = 1e-12;
const U_KEY: &str = "0.parametrizations.weight.0._u";
const V_KEY: &str = "0.parametrizations.weight.0._v";

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn ramp(shape: &[usize], scale: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    tensor(
        (0..n).map(|i| ((i as f32) * 0.37).sin() * scale).collect(),
        shape,
    )
}

fn assert_same_state(a: &Sequential, b: &Sequential, what: &str) {
    let (sa, sb) = (a.state_dict(), b.state_dict());
    assert_eq!(sa.len(), sb.len(), "{what}: キー数");
    for (k, v) in &sa {
        assert_eq!(bits(v), bits(&sb[k]), "{what}: {k} が bit 不一致");
    }
}

fn only_safetensors(dir: &Path) -> PathBuf {
    let names: Vec<String> = std::fs::read_dir(dir)
        .expect("読めるはず")
        .map(|e| {
            e.expect("エントリ")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .filter(|n| n.starts_with("model.") && n.ends_with(".safetensors"))
        .collect();
    assert_eq!(names.len(), 1, "{names:?}");
    dir.join(&names[0])
}

fn entries(dir: &Path) -> BTreeSet<String> {
    std::fs::read_dir(dir)
        .expect("読めるはず")
        .map(|e| {
            e.expect("エントリ")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect()
}

/// manifest 本文から `"kind":"..."` を拾う（kind は英小文字・数字・`_` のみのプログラム生成値）。
fn kinds_of(manifest: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let mut rest = manifest;
    while let Some(pos) = rest.find("\"kind\":\"") {
        rest = &rest[pos + "\"kind\":\"".len()..];
        let end = rest.find('"').expect("kind は閉じているはず");
        out.insert(rest[..end].to_string());
    }
    out
}

fn saved(model: &Sequential, label: &str) -> (TempDirGuard, PathBuf, String) {
    let guard = TempDirGuard::new(label);
    let dir = guard.path().join("m");
    save_model(model, &dir).unwrap_or_else(|e| panic!("{label}: 保存できるはず: {e}"));
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず");
    (guard, dir, text)
}

fn load_err(dir: &Path) -> ModelIoError {
    match load_model(dir) {
        Ok(_) => panic!("改竄・不整合は拒否されるはず"),
        Err(e) => e,
    }
}

// ---------------------------------------------------------------------
// (a) 14 kind の往復
// ---------------------------------------------------------------------

type BaseFn = fn() -> Sequential;

fn linear_base() -> Sequential {
    Sequential::new().add_linear(3, 4, SEED).unwrap()
}
fn conv1d_base() -> Sequential {
    Sequential::new()
        .add_conv1d(4, 6, 3, 2, 1, 1, 2, SEED)
        .unwrap()
}
fn conv2d_base() -> Sequential {
    Sequential::new()
        .add_conv2d(4, 6, [3, 2], [1, 2], [1, 0], [1, 1], 2, SEED)
        .unwrap()
}
fn conv3d_base() -> Sequential {
    Sequential::new()
        .add_conv3d(4, 6, [2, 2, 2], [1, 1, 1], [1, 0, 0], [1, 1, 1], 2, SEED)
        .unwrap()
}
fn conv_transpose1d_base() -> Sequential {
    Sequential::new()
        .add_conv_transpose1d(4, 6, 3, 2, 1, 1, 1, 2, SEED)
        .unwrap()
}
fn conv_transpose2d_base() -> Sequential {
    Sequential::new()
        .add_conv_transpose2d(4, 6, [3, 2], [2, 1], [1, 0], [1, 0], [1, 1], 2, SEED)
        .unwrap()
}
fn conv_transpose3d_base() -> Sequential {
    Sequential::new()
        .add_conv_transpose3d(
            4,
            6,
            [2, 2, 2],
            [2, 1, 1],
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            2,
            SEED,
        )
        .unwrap()
}

/// `(基底 kind, 基底の構築, 入力 shape)`。
const BASES: [(&str, BaseFn, &[usize]); 7] = [
    ("linear", linear_base, &[5, 3]),
    ("conv1d", conv1d_base, &[2, 4, 9]),
    ("conv2d", conv2d_base, &[2, 4, 6, 7]),
    ("conv3d", conv3d_base, &[2, 4, 3, 3, 3]),
    ("conv_transpose1d", conv_transpose1d_base, &[2, 4, 5]),
    ("conv_transpose2d", conv_transpose2d_base, &[2, 4, 5, 4]),
    ("conv_transpose3d", conv_transpose3d_base, &[2, 4, 3, 3, 3]),
];

#[derive(Clone, Copy, Debug)]
enum Variant {
    WeightNorm(Option<usize>),
    SpectralNorm,
}

fn reparam(base: BaseFn, v: Variant) -> Sequential {
    let m = base();
    match v {
        Variant::WeightNorm(dim) => m.parametrize_weight_norm(0, dim).unwrap(),
        Variant::SpectralNorm => m.parametrize_spectral_norm(0, N_ITER, EPS, SEED).unwrap(),
    }
}

fn expected_kind(base_kind: &str, v: Variant) -> String {
    match v {
        Variant::WeightNorm(_) => format!("{base_kind}_weight_norm"),
        Variant::SpectralNorm => format!("{base_kind}_spectral_norm"),
    }
}

#[test]
fn fourteen_kinds_round_trip_bit_identically_and_keep_format_version_2() {
    let mut seen = BTreeSet::new();
    let variants = [
        Variant::WeightNorm(Some(0)),
        Variant::WeightNorm(Some(1)),
        Variant::WeightNorm(None),
        Variant::SpectralNorm,
    ];
    for (base_kind, base, shape) in BASES {
        let x = ramp(shape, 1.0);
        for v in variants {
            let label = format!("{base_kind}-{v:?}");
            let model = reparam(base, v);
            // train 中に状態を進める（spectral_norm の u／v を初期値から離す。weight_norm は無関係）。
            for _ in 0..3 {
                model.predict(&x).expect("predict できるはず");
            }
            let (g, dir, text) = saved(&model, &format!("reparam-rt-{label}"));
            assert!(
                text.contains("\"format_version\":2"),
                "{label}: format_version は 2 のまま"
            );
            let kind = expected_kind(base_kind, v);
            let kinds = kinds_of(&text);
            assert!(kinds.contains(&kind), "{label}: {kinds:?}");
            seen.extend(kinds);

            let loaded = load_model(&dir).unwrap_or_else(|e| panic!("{label}: 復元できる: {e}"));
            assert_same_state(&model, &loaded, &label);
            assert_eq!(model.training(), loaded.training(), "{label}: training");
            // u／v を bit のまま復元していれば、train の predict が同じ状態から進み続けて一致する。
            for call in 0..3 {
                let ya = model.predict(&x).expect("predict できるはず");
                let yb = loaded.predict(&x).expect("predict できるはず");
                assert_eq!(bits(&ya), bits(&yb), "{label}: predict call={call}");
            }

            // 再保存しても buffer（spectral の u／v）を含む全テンソルが bit 一致する。
            let g2 = TempDirGuard::new(&format!("reparam-resave-{label}"));
            let dir2 = g2.path().join("m");
            let loaded2 = load_model(&dir).expect("復元できるはず");
            save_model(&loaded2, &dir2).expect("再保存できるはず");
            let a = load_safetensors_f32(&only_safetensors(&dir)).expect("読めるはず");
            let b = load_safetensors_f32(&only_safetensors(&dir2)).expect("読めるはず");
            assert_eq!(a.len(), b.len(), "{label}: 再保存のキー数");
            for (k, t) in &a {
                assert_eq!(bits(t), bits(&b[k]), "{label}: 再保存 {k}");
            }
            let has_uv = a.contains_key(U_KEY) && a.contains_key(V_KEY);
            assert_eq!(
                has_uv,
                matches!(v, Variant::SpectralNorm),
                "{label}: u／v は spectral_norm のときだけ保存する"
            );
            drop(g);
        }
    }
    let expected: BTreeSet<String> = BASES
        .iter()
        .flat_map(|(b, _, _)| [format!("{b}_weight_norm"), format!("{b}_spectral_norm")])
        .collect();
    assert_eq!(seen, expected, "14 kind すべてを網羅する");
    assert_eq!(expected.len(), 14);
}

#[test]
fn eval_mode_models_round_trip_and_spectral_state_stays_frozen() {
    for (base_kind, base, shape) in BASES {
        let x = ramp(shape, 1.0);
        for v in [Variant::WeightNorm(Some(0)), Variant::SpectralNorm] {
            let label = format!("{base_kind}-{v:?}-eval");
            let mut model = reparam(base, v);
            model.predict(&x).unwrap();
            model.eval();
            let (_g, dir, _) = saved(&model, &format!("reparam-eval-{label}"));
            let loaded = load_model(&dir).expect("復元できるはず");
            assert!(!loaded.training(), "{label}: eval のまま復元");
            for _ in 0..2 {
                assert_eq!(
                    bits(&model.predict(&x).unwrap()),
                    bits(&loaded.predict(&x).unwrap()),
                    "{label}"
                );
            }
        }
    }
}

#[test]
fn rank_zero_weight_norm_g_round_trips() {
    // `dim = None` の g は rank 0（決定記録 §14.1 の積み残し項目の実測）。manifest は `"shape":[]`。
    let model = linear_base().parametrize_weight_norm(0, None).unwrap();
    let (_g, dir, text) = saved(&model, "reparam-rank0");
    assert!(
        text.contains("\"key\":\"0.parametrizations.weight.original0\",\"shape\":[]"),
        "{text}"
    );
    let loaded = load_model(&dir).expect("復元できるはず");
    assert_same_state(&model, &loaded, "rank0");
    let g = &loaded.state_dict()["0.parametrizations.weight.original0"];
    assert!(g.shape().is_empty());
}

// ---------------------------------------------------------------------
// optimizer 往復・学習テスト
// ---------------------------------------------------------------------

/// Linear(weight_norm) → ReLU → Linear(spectral_norm)。
fn mlp_model() -> Sequential {
    Sequential::new()
        .add_linear(3, 6, SEED)
        .unwrap()
        .add_relu()
        .add_linear(6, 4, SEED + 1)
        .unwrap()
        .parametrize_weight_norm(0, Some(1))
        .unwrap()
        .parametrize_spectral_norm(2, N_ITER, EPS, SEED)
        .unwrap()
}

fn mlp_data() -> (Tensor<f32>, Tensor<f32>) {
    (ramp(&[10, 3], 1.0), ramp(&[10, 4], 0.5))
}

/// ConvTranspose2d(spectral_norm) → ReLU → Conv2d(weight_norm)。入力 `[2, 2, 4, 4]` → 出力 `[2, 2, 5, 5]`。
fn conv_model() -> Sequential {
    Sequential::new()
        .add_conv_transpose2d(2, 3, [2, 2], [1, 1], [0, 0], [0, 0], [1, 1], 1, SEED)
        .unwrap()
        .add_relu()
        .add_conv2d(3, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, SEED + 1)
        .unwrap()
        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
        .unwrap()
        .parametrize_weight_norm(2, Some(0))
        .unwrap()
}

#[test]
fn optimizer_round_trip_resumes_training_bit_identically() {
    let (x, y) = mlp_data();
    let mut model = mlp_model();
    model
        .compile(Optimizer::AdamW(AdamWConfig::default()), Loss::Mse)
        .unwrap();
    model.fit(&x, &y, FitConfig::new(3, 5)).unwrap();
    let (_g, dir, _) = saved(&model, "reparam-opt");
    let mut loaded = load_model(&dir).expect("復元できるはず");
    assert!(loaded.is_compiled());
    assert_same_state(&model, &loaded, "復元直後");
    let h1 = model.fit(&x, &y, FitConfig::new(2, 5)).unwrap();
    let h2 = loaded.fit(&x, &y, FitConfig::new(2, 5)).unwrap();
    let lb = |h: &[f32]| h.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(lb(&h1.loss), lb(&h2.loss), "復元後の次の学習の loss");
    assert_same_state(&model, &loaded, "続き学習後");
}

#[test]
fn training_decreases_loss_and_spectral_state_moves_only_in_train() {
    let (x, y) = mlp_data();
    let mut model = mlp_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
        .unwrap();
    let read_uv = |m: &Sequential, label: &str| -> (Vec<u32>, Vec<u32>) {
        let (_g, dir, _) = saved(m, label);
        let st = load_safetensors_f32(&only_safetensors(&dir)).expect("読めるはず");
        (
            bits(&st["2.parametrizations.weight.0._u"]),
            bits(&st["2.parametrizations.weight.0._v"]),
        )
    };
    let before = read_uv(&model, "reparam-train-0");
    let history = model.fit(&x, &y, FitConfig::new(8, 5)).unwrap();
    let (first, last) = (history.loss[0], *history.loss.last().unwrap());
    assert!(
        first.is_finite() && last.is_finite() && last < first,
        "loss は減少するはず {:?}",
        history.loss
    );
    let trained = read_uv(&model, "reparam-train-1");
    assert_ne!(before, trained, "train の forward で u／v が進む");
    // eval の forward では進まない（保存値が不変）。
    model.eval();
    for _ in 0..3 {
        model.predict(&x).unwrap();
    }
    let evaled = read_uv(&model, "reparam-train-2");
    assert_eq!(trained, evaled, "eval では u／v は不変");
}

#[test]
fn conv_family_trains_and_round_trips() {
    let (x, y) = (ramp(&[2, 2, 4, 4], 1.0), ramp(&[2, 2, 5, 5], 0.5));
    let mut model = conv_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.01)), Loss::Mse)
        .unwrap();
    let history = model.fit(&x, &y, FitConfig::new(6, 2)).unwrap();
    let (first, last) = (history.loss[0], *history.loss.last().unwrap());
    assert!(
        first.is_finite() && last.is_finite() && last < first,
        "loss は減少するはず {:?}",
        history.loss
    );
    let (_g, dir, text) = saved(&model, "reparam-conv-train");
    let kinds = kinds_of(&text);
    assert!(
        kinds.contains("conv_transpose2d_spectral_norm"),
        "{kinds:?}"
    );
    assert!(kinds.contains("conv2d_weight_norm"), "{kinds:?}");
    let mut loaded = load_model(&dir).expect("復元できるはず");
    assert_same_state(&model, &loaded, "conv 復元直後");
    let h1 = model.fit(&x, &y, FitConfig::new(2, 2)).unwrap();
    let h2 = loaded.fit(&x, &y, FitConfig::new(2, 2)).unwrap();
    let lb = |h: &[f32]| h.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(lb(&h1.loss), lb(&h2.loss));
    assert_same_state(&model, &loaded, "conv 続き学習後");
}

// ---------------------------------------------------------------------
// 改竄拒否
// ---------------------------------------------------------------------

/// 改竄ケースの共通ベース（Linear(weight_norm, dim=1) → ReLU → Linear(spectral_norm)）。
fn base_for_tamper() -> Sequential {
    Sequential::new()
        .add_linear(3, 4, 1)
        .unwrap()
        .add_relu()
        .add_linear(4, 2, 2)
        .unwrap()
        .parametrize_weight_norm(0, Some(1))
        .unwrap()
        .parametrize_spectral_norm(2, N_ITER, EPS, 3)
        .unwrap()
}

fn manifest_tampered(label: &str, from: &str, to: &str) -> ModelIoError {
    let (_g, dir, text) = saved(&base_for_tamper(), &format!("reparam-t-{label}"));
    assert!(text.contains(from), "{label}: 置換対象 {from} がある");
    let before = entries(&dir);
    std::fs::write(dir.join("manifest.json"), text.replacen(from, to, 1)).expect("書けるはず");
    let err = load_err(&dir);
    assert_eq!(entries(&dir), before, "{label}: dir を変更しない");
    err
}

#[test]
fn tampered_params_are_rejected_with_manifest_error() {
    let cases: [(&str, &str, &str); 24] = [
        ("dim-out-of-range", "\"dim\":1", "\"dim\":2"),
        ("dim-huge", "\"dim\":1", "\"dim\":9"),
        ("dim-string", "\"dim\":1", "\"dim\":\"1\""),
        ("dim-negative", "\"dim\":1", "\"dim\":-1"),
        ("dim-fractional", "\"dim\":1", "\"dim\":1.5"),
        ("dim-bool", "\"dim\":1", "\"dim\":true"),
        ("dim-missing", ",\"dim\":1}", "}"),
        (
            "extra-key",
            "\"params\":{\"in_features\":3,",
            "\"params\":{\"x\":1,\"in_features\":3,",
        ),
        ("spectral-n-zero", "\"2:1e-12\"", "\"0:1e-12\""),
        ("spectral-n-over-limit", "\"2:1e-12\"", "\"4097:1e-12\""),
        (
            "spectral-n-overflow",
            "\"2:1e-12\"",
            "\"99999999999999999999:1e-12\"",
        ),
        ("spectral-n-leading-zero", "\"2:1e-12\"", "\"02:1e-12\""),
        ("spectral-eps-noncanonical", "\"2:1e-12\"", "\"2:1E-12\""),
        ("spectral-eps-negative", "\"2:1e-12\"", "\"2:-1e-12\""),
        ("spectral-eps-nan", "\"2:1e-12\"", "\"2:NaN\""),
        ("spectral-eps-inf", "\"2:1e-12\"", "\"2:inf\""),
        ("spectral-no-colon", "\"2:1e-12\"", "\"2\""),
        ("spectral-empty-eps", "\"2:1e-12\"", "\"2:\""),
        ("spectral-extra-colon", "\"2:1e-12\"", "\"2:1e-12:3\""),
        (
            "spectral-number",
            "\"spectral\":\"2:1e-12\"",
            "\"spectral\":2",
        ),
        (
            "spectral-wrong-key",
            "\"spectral\":\"2:1e-12\"",
            "\"spectrum\":\"2:1e-12\"",
        ),
        (
            "kind-swap-wn-to-sn",
            "\"kind\":\"linear_weight_norm\"",
            "\"kind\":\"linear_spectral_norm\"",
        ),
        (
            "kind-swap-sn-to-wn",
            "\"kind\":\"linear_spectral_norm\"",
            "\"kind\":\"linear_weight_norm\"",
        ),
        (
            "kind-swap-base",
            "\"kind\":\"linear_weight_norm\"",
            "\"kind\":\"conv2d_weight_norm\"",
        ),
    ];
    for (label, from, to) in cases {
        let err = manifest_tampered(label, from, to);
        assert!(
            matches!(err, ModelIoError::Manifest { .. }),
            "{label}: {err}"
        );
    }
    // 未知 kind は `UnsupportedModel`。
    let err = manifest_tampered(
        "unknown-kind",
        "\"kind\":\"linear_weight_norm\"",
        "\"kind\":\"linear_weight_norm2\"",
    );
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err}"
    );
}

#[test]
fn tampered_key_lists_are_rejected_with_mismatch() {
    let cases: [(&str, &str, &str); 7] = [
        (
            "g-shape",
            "\"key\":\"0.parametrizations.weight.original0\",\"shape\":[1,4]",
            "\"key\":\"0.parametrizations.weight.original0\",\"shape\":[1,5]",
        ),
        (
            "g-shape-rank0",
            "\"key\":\"0.parametrizations.weight.original0\",\"shape\":[1,4]",
            "\"key\":\"0.parametrizations.weight.original0\",\"shape\":[]",
        ),
        (
            "original-renamed",
            "2.parametrizations.weight.original\"",
            "2.parametrizations.weight.original2\"",
        ),
        (
            "buffers-dropped",
            "\"buffer_keys\":[{\"key\":\"2.parametrizations.weight.0._u\",\"shape\":[2]},{\"key\":\"2.parametrizations.weight.0._v\",\"shape\":[4]}]",
            "\"buffer_keys\":[]",
        ),
        (
            "u-shape",
            "\"key\":\"2.parametrizations.weight.0._u\",\"shape\":[2]",
            "\"key\":\"2.parametrizations.weight.0._u\",\"shape\":[3]",
        ),
        (
            "u-renamed",
            "2.parametrizations.weight.0._u\"",
            "2.parametrizations.weight.0._w\"",
        ),
        (
            "v-shape",
            "\"key\":\"2.parametrizations.weight.0._v\",\"shape\":[4]",
            "\"key\":\"2.parametrizations.weight.0._v\",\"shape\":[5]",
        ),
    ];
    for (label, from, to) in cases {
        let err = manifest_tampered(label, from, to);
        assert!(
            matches!(err, ModelIoError::Mismatch { .. }),
            "{label}: {err}"
        );
    }
}

#[test]
fn legacy_format_version_1_never_accepts_reparametrized_layers() {
    // v1 には再パラメータ化 kind が存在し得ない。buffer_keys が揃っていても、空でも拒否する。
    let err = manifest_tampered(
        "v1-with-buffers",
        "\"format_version\":2",
        "\"format_version\":1",
    );
    assert!(matches!(err, ModelIoError::Mismatch { .. }), "{err}");

    let (_g, dir, text) = saved(&base_for_tamper(), "reparam-t-v1-empty");
    let a = text.find(",\"buffer_keys\":[").expect("buffer_keys がある");
    let b = text[a..]
        .find("],\"safetensors_file\"")
        .expect("終端がある")
        + a;
    let tampered = format!(
        "{}{}",
        &text[..a],
        text[b + 1..].replacen(
            ",\"safetensors_file\"",
            ",\"buffer_keys\":[],\"safetensors_file\"",
            1
        )
    )
    .replacen("\"format_version\":2", "\"format_version\":1", 1);
    std::fs::write(dir.join("manifest.json"), tampered).expect("書けるはず");
    let err = load_err(&dir);
    assert!(
        matches!(
            err,
            ModelIoError::Mismatch { .. } | ModelIoError::Manifest { .. }
        ),
        "{err}"
    );

    // weight_norm のみのモデルでも v1 は拒否する。
    let wn = linear_base().parametrize_weight_norm(0, Some(0)).unwrap();
    let (_g2, dir2, text2) = saved(&wn, "reparam-t-v1-wn");
    std::fs::write(
        dir2.join("manifest.json"),
        text2.replacen("\"format_version\":2", "\"format_version\":1", 1),
    )
    .expect("書けるはず");
    let err = load_err(&dir2);
    assert!(matches!(err, ModelIoError::Mismatch { .. }), "{err}");
}

fn replace_safetensors(dir: &Path, tensors: &HashMap<String, Tensor<f32>>) {
    let st = only_safetensors(dir);
    let bytes = save_safetensors_f32_to_bytes(tensors, None).expect("書き出せるはず");
    let old_len = std::fs::metadata(&st).expect("stat できるはず").len();
    std::fs::write(&st, &bytes).expect("書けるはず");
    let path = dir.join("manifest.json");
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
fn tampered_safetensors_buffers_are_rejected() {
    type Tamper = fn(&mut HashMap<String, Tensor<f32>>);
    let cases: [(&str, Tamper, bool); 6] = [
        (
            "u-missing",
            |m| {
                m.remove("2.parametrizations.weight.0._u");
            },
            true,
        ),
        (
            "v-missing",
            |m| {
                m.remove("2.parametrizations.weight.0._v");
            },
            true,
        ),
        (
            "extra-buffer",
            |m| {
                m.insert(
                    "2.parametrizations.weight.0._x".into(),
                    Tensor::new(vec![0.0; 2], &[2]).unwrap(),
                );
            },
            true,
        ),
        (
            "u-shape",
            |m| {
                m.insert(
                    "2.parametrizations.weight.0._u".into(),
                    Tensor::new(vec![0.5; 3], &[3]).unwrap(),
                );
            },
            true,
        ),
        // 形式は整合するが値が非有限: 型付きの層エラー（`Autodiff`）で拒否される。
        (
            "u-nan",
            |m| {
                m.insert(
                    "2.parametrizations.weight.0._u".into(),
                    Tensor::new(vec![f32::NAN, 0.5], &[2]).unwrap(),
                );
            },
            false,
        ),
        (
            "v-inf",
            |m| {
                m.insert(
                    "2.parametrizations.weight.0._v".into(),
                    Tensor::new(vec![f32::INFINITY, 0.0, 0.0, 0.0], &[4]).unwrap(),
                );
            },
            false,
        ),
    ];
    for (label, tamper, mismatch) in cases {
        let (_g, dir, _) = saved(&base_for_tamper(), &format!("reparam-st-{label}"));
        let mut tensors = load_safetensors_f32(&only_safetensors(&dir)).expect("読めるはず");
        tamper(&mut tensors);
        replace_safetensors(&dir, &tensors);
        let err = load_err(&dir);
        if mismatch {
            assert!(
                matches!(err, ModelIoError::Mismatch { .. }),
                "{label}: {err}"
            );
        } else {
            assert!(
                matches!(
                    err,
                    ModelIoError::Autodiff(AutodiffError::InvalidArgument(_))
                ),
                "{label}: {err}"
            );
        }
    }
}

// ---------------------------------------------------------------------
// save 側 fail-closed
// ---------------------------------------------------------------------

#[test]
fn non_finite_spectral_state_is_rejected_at_save_without_touching_dir() {
    // weight を NaN にして train の forward を通すと u／v が非有限になる。
    let model = linear_base()
        .parametrize_spectral_norm(0, N_ITER, EPS, SEED)
        .unwrap();
    let mut state = model.state_dict();
    let key = "0.parametrizations.weight.original";
    let shape = state[key].shape().to_vec();
    let n: usize = shape.iter().product();
    state.insert(key.into(), tensor(vec![f32::NAN; n], &shape));
    let mut model = model;
    model
        .load_state_dict(state)
        .expect("NaN の重みも設定できる");
    let _ = model.predict(&ramp(&[5, 3], 1.0));
    let guard = TempDirGuard::new("reparam-nonfinite");
    let dir = guard.path().join("m");
    let err = save_model(&model, &dir).unwrap_err();
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err:?}"
    );
    assert!(!dir.exists(), "拒否時は dir を作らない");
}

#[test]
fn n_power_iterations_over_limit_is_rejected_at_save() {
    let model = linear_base()
        .parametrize_spectral_norm(0, 4097, EPS, SEED)
        .unwrap();
    let guard = TempDirGuard::new("reparam-n-limit");
    let dir = guard.path().join("m");
    let err = save_model(&model, &dir).unwrap_err();
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err:?}"
    );
    assert!(!dir.exists());
    // 上限ちょうどは保存・復元できる。
    let ok = linear_base()
        .parametrize_spectral_norm(0, 4096, EPS, SEED)
        .unwrap();
    let (_g, dir, _) = saved(&ok, "reparam-n-limit-ok");
    load_model(&dir).expect("上限ちょうどは復元できるはず");
}

// ---------------------------------------------------------------------
// Functional モデル（ブロックが再パラメータ化層を含む場合）
// ---------------------------------------------------------------------

#[test]
fn functional_blocks_with_reparametrized_layers_round_trip() {
    use fandhe_ai::compat::{FunctionalBuilder, load_functional_model, save_functional_model};
    let build = || {
        let mut b = FunctionalBuilder::new();
        let a = b.input().unwrap();
        let block0 = Sequential::new()
            .add_linear(3, 4, 1)
            .unwrap()
            .parametrize_weight_norm(0, Some(1))
            .unwrap();
        let block1 = Sequential::new()
            .add_linear(4, 2, 2)
            .unwrap()
            .add_relu()
            .parametrize_spectral_norm(0, N_ITER, EPS, 3)
            .unwrap();
        let h = b.apply(block0, a).unwrap();
        let o = b.apply(block1, h).unwrap();
        b.build(&[a], &[o]).unwrap()
    };
    let model = build();
    let x = ramp(&[6, 3], 1.0);
    model.predict(&[&x]).unwrap();
    let guard = TempDirGuard::new("reparam-functional");
    let dir = guard.path().join("m");
    save_functional_model(&model, &dir).expect("保存できるはず");
    let loaded = load_functional_model(&dir).expect("復元できるはず");
    let (pa, pb) = (
        model.named_parameters().unwrap(),
        loaded.named_parameters().unwrap(),
    );
    assert_eq!(pa.len(), pb.len());
    for ((ka, ta), (kb, tb)) in pa.iter().zip(&pb) {
        assert_eq!(ka, kb);
        assert_eq!(bits(ta), bits(tb), "{ka}");
    }
    for call in 0..2 {
        let a = model.predict(&[&x]).unwrap();
        let b = loaded.predict(&[&x]).unwrap();
        assert_eq!(bits(&a[0]), bits(&b[0]), "functional predict call={call}");
    }
}
