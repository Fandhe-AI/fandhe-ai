//! `save_model`／`load_model` の `Lbfgs` compile 状態の往復の統合テスト
//! （イシュー #2373・親 #2362。受入基準 (a)〜(c) のうち (a)・(b)）。
//!
//! 役割: facade の公開 API だけで、履歴が `history_size` 未満の状態・到達済み（追い出しを含む）の
//! 状態のそれぞれで save → load → 続く `fit` が bit 一致すること（(a)）、`history_len` と実キー数の
//! 不一致・固定上限（65536）の超過・非有限値・不正な config・AMP 併用の manifest が fail-closed に
//! 拒否されること（(b)）を検証する。上限値は `MAX_LBFGS_HISTORY`（`model_io.rs`。65536。
//! 2026-09-29 ユーザー承認済み）で、定数は非公開のためここでは 65536／65537 の実値で境界を突く。
//!
//! `Lbfgs` はフルバッチ（`batch_size = N`）でしか使えない（`Optimizer::Lbfgs` の doc）。
//! `LbfgsLineSearch` は facade 未公開のため、`StrongWolfe` の往復は `compiled.rs` の単体テストが担う。
//! 既存の `compat_sequential_model_io_compiled.rs`（他 6 optimizer）とは別ファイルに置く。

#![cfg(unix)]

use std::collections::HashMap;
use std::path::Path;

use fandhe_ai::Tensor;
use fandhe_ai::compat::{
    FitConfig, Loss, ModelIoError, Optimizer, Sequential, load_model, save_model,
};
use fandhe_ai::interop::safetensors::{
    load_safetensors_f32_from_bytes, save_safetensors_f32_to_bytes,
};
use fandhe_ai::optim::LbfgsConfig;

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

fn config(history_size: usize, max_iter: usize) -> LbfgsConfig {
    LbfgsConfig {
        lr: 0.05,
        max_iter,
        history_size,
        ..LbfgsConfig::default()
    }
}

fn compiled(cfg: LbfgsConfig) -> Sequential {
    let mut m = build();
    m.compile(Optimizer::Lbfgs(cfg), Loss::Mse)
        .expect("compile");
    m
}

fn fit(m: &mut Sequential, epochs: usize) -> fandhe_ai::compat::History {
    let (x, y) = data();
    m.fit(&x, &y, FitConfig::new(epochs, N)).expect("fit")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("連続のはず")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
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

fn field_usize(text: &str, key: &str) -> usize {
    let needle = format!("\"{key}\":");
    let start = text.find(&needle).unwrap_or_else(|| panic!("{key} が無い")) + needle.len();
    let end = start + text[start..].find([',', '}']).expect("終端");
    text[start..end].parse().expect("整数のはず")
}

fn with_state_keys(text: &str, f: impl FnOnce(Vec<String>) -> Vec<String>) -> String {
    let needle = "\"optimizer_state_keys\":[";
    let start = text.find(needle).expect("keys") + needle.len();
    let end = start + text[start..].find(']').expect("]");
    let keys: Vec<String> = text[start..end].split(',').map(str::to_string).collect();
    format!("{}{}{}", &text[..start], f(keys).join(","), &text[end..])
}

/// safetensors を読み替えて書き戻し、manifest の `safetensors_bytes` も合わせる。
fn rewrite_safetensors(dir: &Path, edit: impl FnOnce(&mut HashMap<String, Tensor<f32>>)) {
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
    write_manifest(
        dir,
        &set_field(&text, "safetensors_bytes", &bytes.len().to_string()),
    );
}

/// 履歴が溜まった Lbfgs モデルを保存した dir（`history_size = 3`・3 epoch で到達済み）。
fn saved_dir(label: &str) -> (TempDirGuard, std::path::PathBuf) {
    let mut m = compiled(config(3, 4));
    fit(&mut m, 3);
    let guard = TempDirGuard::new(label);
    let dir = guard.path().join("m");
    save_model(&m, &dir).expect("save");
    (guard, dir)
}

fn entries(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .expect("読めるはず")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// (a): 保存 → 復元 → 続く fit が bit 一致。`expect_full` は履歴が `history_size` に到達済みか。
fn assert_resumes_bit_identically(label: &str, cfg: LbfgsConfig, epochs: usize, expect_full: bool) {
    let mut m = compiled(cfg);
    fit(&mut m, epochs);

    let guard = TempDirGuard::new("resume");
    let dir = guard.path().join("m");
    save_model(&m, &dir).unwrap_or_else(|e| panic!("{label}: save: {e}"));
    // Lbfgs の内部状態は公開 API から見えないため、manifest の値で履歴の状態を確認する。
    let text = read_manifest(&dir);
    let (len, size) = (
        field_usize(&text, "history_len"),
        field_usize(&text, "history_size"),
    );
    if expect_full {
        assert_eq!(len, size, "{label}: 履歴は到達済みのはず");
    } else {
        assert!(
            0 < len && len < size,
            "{label}: 履歴は未満のはず: {len}/{size}"
        );
    }

    let mut m2 = load(&dir);
    assert!(m2.is_compiled(), "{label}");
    assert_same_params(&m, &m2, label);
    let h1 = fit(&mut m, 2);
    let h2 = fit(&mut m2, 2);
    assert_eq!(f32_bits(&h1.loss), f32_bits(&h2.loss), "{label}: loss");
    assert_eq!(f32_bits(&h1.lr), f32_bits(&h2.lr), "{label}: lr");
    assert_same_params(&m, &m2, label);
}

#[test]
fn resumes_bit_identically_with_history_below_and_at_capacity() {
    assert_resumes_bit_identically("未満", config(50, 4), 1, false);
    // 到達済み（古いペアの追い出しを含む）。
    assert_resumes_bit_identically("到達済み", config(3, 4), 3, true);
}

#[test]
fn freshly_compiled_lbfgs_round_trips_with_config_fields() {
    let cfg = LbfgsConfig {
        lr: 0.07,
        max_iter: 3,
        max_eval: Some(7),
        tolerance_grad: 1e-6,
        tolerance_change: 1e-8,
        history_size: 5,
        line_search_steps: 11,
        ..LbfgsConfig::default()
    };
    let m = compiled(cfg);
    let guard = TempDirGuard::new("fresh");
    let dir = guard.path().join("m");
    save_model(&m, &dir).expect("save");
    let text = read_manifest(&dir);
    assert_eq!(field_usize(&text, "history_len"), 0);
    assert!(text.contains("\"max_eval\":7"), "{text}");
    assert!(text.contains("\"line_search\":\"none\""), "{text}");
    let mut m2 = load(&dir);
    let mut m1 = m;
    let h1 = fit(&mut m1, 2);
    let h2 = fit(&mut m2, 2);
    assert_eq!(f32_bits(&h1.loss), f32_bits(&h2.loss));
    assert_eq!(f32_bits(&h1.lr), f32_bits(&h2.lr));
    assert_same_params(&m1, &m2, "fresh");

    // fit 前に再保存した compiled 節は元と同一（復元した config・状態キーが全て一致）。
    let guard2 = TempDirGuard::new("resave");
    let dir2 = guard2.path().join("m");
    let mut fresh_model = load(&dir);
    save_model(&fresh_model, &dir2).expect("resave");
    let compiled_part = |t: &str| t[t.find("\"compiled\":").expect("compiled")..].to_string();
    assert_eq!(
        compiled_part(&read_manifest(&dir)),
        compiled_part(&read_manifest(&dir2))
    );
    fit(&mut fresh_model, 1);
}

// ---------------------------------------------------------------------
// 改竄の拒否（(b)）
// ---------------------------------------------------------------------

#[test]
fn history_len_mismatch_with_real_keys_is_rejected() {
    let (_g, dir) = saved_dir("len");
    let text = read_manifest(&dir);
    let len = field_usize(&text, "history_len");
    for bad in [len - 1, len + 1] {
        write_manifest(&dir, &set_field(&text, "history_len", &bad.to_string()));
        let e = load_err(&dir);
        assert!(matches!(e, ModelIoError::Mismatch { .. }), "{bad}: {e}");
    }
    write_manifest(&dir, &text);
    load(&dir);
}

#[test]
fn limits_are_enforced_at_65536() {
    let (_g, dir) = saved_dir("limit");
    let text = read_manifest(&dir);
    for (key, v) in [("history_len", 65537), ("history_size", 65537)] {
        write_manifest(&dir, &set_field(&text, key, &v.to_string()));
        assert!(
            matches!(load_err(&dir), ModelIoError::TooLarge { limit: 65536, .. }),
            "{key}"
        );
    }
    // 上限ちょうどは manifest の上限検査を通り、実キー数との不一致（history_len）で拒否される。
    write_manifest(&dir, &set_field(&text, "history_len", "65536"));
    assert!(matches!(load_err(&dir), ModelIoError::Mismatch { .. }));
    // history_size を履歴件数未満にすると autodiff が拒否する。
    write_manifest(&dir, &set_field(&text, "history_size", "2"));
    assert!(matches!(load_err(&dir), ModelIoError::Autodiff(_)));
    // history_size = 65536 は受理され、正常に load できる。
    write_manifest(&dir, &set_field(&text, "history_size", "65536"));
    load(&dir);
}

#[test]
fn tampered_config_fields_are_rejected() {
    let (_g, dir) = saved_dir("cfg");
    let text = read_manifest(&dir);
    assert!(text.contains("\"max_eval\":null"), "{text}");
    for bad in ["-1", "1.5", "\"x\""] {
        write_manifest(&dir, &set_field(&text, "max_eval", bad));
        assert!(
            matches!(load_err(&dir), ModelIoError::Manifest { .. }),
            "max_eval={bad}"
        );
    }
    write_manifest(&dir, &set_field(&text, "max_eval", "0"));
    assert!(matches!(load_err(&dir), ModelIoError::Autodiff(_)));

    write_manifest(&dir, &set_field(&text, "line_search", "\"bogus\""));
    assert!(matches!(
        load_err(&dir),
        ModelIoError::UnsupportedModel { .. }
    ));
    write_manifest(&dir, &set_field(&text, "line_search", "1"));
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));

    for bad in ["NaN", "1e39", "0.5000", "1"] {
        write_manifest(&dir, &set_field(&text, "lr", bad));
        assert!(
            matches!(load_err(&dir), ModelIoError::Manifest { .. }),
            "lr={bad}"
        );
    }
}

#[test]
fn non_finite_or_invalid_state_tensors_are_rejected() {
    for (key, value) in [
        ("optimizer.d", f32::NAN),
        ("optimizer.d", f32::INFINITY),
        ("optimizer.t", f32::NAN),
        ("optimizer.h_diag", f32::INFINITY),
        ("optimizer.history.rho", f32::NAN),
        ("optimizer.history.rho", 0.0),
    ] {
        let (_g, dir) = saved_dir("nan");
        rewrite_safetensors(&dir, |map| {
            let t = map.get_mut(key).unwrap_or_else(|| panic!("{key} が無い"));
            let shape = t.shape().to_vec();
            let n: usize = shape.iter().product();
            *t = Tensor::new(vec![value; n], &shape).expect("置換");
        });
        let e = load_err(&dir);
        assert!(matches!(e, ModelIoError::Autodiff(_)), "{key}={value}: {e}");
    }
}

#[test]
fn missing_history_key_is_rejected_even_when_manifest_lists_are_consistent() {
    let (_g, dir) = saved_dir("missing");
    let last = field_usize(&read_manifest(&dir), "history_len") - 1;
    let key = format!("optimizer.history.{last}.s");
    rewrite_safetensors(&dir, |map| {
        map.remove(&key).expect("キーがあるはず");
    });
    let quoted = format!("\"{key}\"");
    let t = with_state_keys(&read_manifest(&dir), |ks| {
        ks.into_iter().filter(|k| *k != quoted).collect()
    });
    write_manifest(&dir, &t);
    let e = load_err(&dir);
    assert!(matches!(e, ModelIoError::Mismatch { .. }), "{e}");
}

#[test]
fn lbfgs_with_amp_or_history_len_on_other_kinds_is_rejected() {
    let (_g, dir) = saved_dir("shape");
    let text = read_manifest(&dir);
    let amp = "{\"dtype\":\"f16\",\"grad_scaler_config\":{\"init_scale\":1.0,\"growth_factor\":2.0,\"backoff_factor\":0.5,\"growth_interval\":3},\"scale\":1.0,\"growth_tracker\":0}";
    write_manifest(
        &dir,
        &text.replace("\"amp\":null", &format!("\"amp\":{amp}")),
    );
    assert!(matches!(
        load_err(&dir),
        ModelIoError::UnsupportedModel { .. }
    ));

    // history_len を欠く lbfgs。
    let len = field_usize(&text, "history_len");
    write_manifest(&dir, &text.replace(&format!(",\"history_len\":{len}"), ""));
    assert!(matches!(load_err(&dir), ModelIoError::Manifest { .. }));

    // 非 lbfgs（adam）に history_len を足す。
    let (x, y) = data();
    let mut m = build();
    m.compile(Optimizer::Adam(Default::default()), Loss::Mse)
        .expect("compile");
    m.fit(&x, &y, FitConfig::new(1, N)).expect("fit");
    let guard = TempDirGuard::new("adam");
    let d2 = guard.path().join("m");
    save_model(&m, &d2).expect("save");
    let t2 = read_manifest(&d2);
    write_manifest(
        &d2,
        &t2.replace("\"config\":", "\"history_len\":0,\"config\":"),
    );
    assert!(matches!(load_err(&d2), ModelIoError::Manifest { .. }));
}

// ---------------------------------------------------------------------
// 保存側
// ---------------------------------------------------------------------

#[test]
fn history_size_over_limit_is_rejected_on_save_without_touching_dir() {
    let m = compiled(config(65537, 2));
    let guard = TempDirGuard::new("over");
    let fresh = guard.path().join("fresh");
    let err = save_model(&m, &fresh).expect_err("上限超過は拒否されるはず");
    assert!(
        matches!(err, ModelIoError::TooLarge { limit: 65536, .. }),
        "{err}"
    );
    assert!(!fresh.exists(), "存在しなかった dir は作られない");

    let existing = guard.path().join("existing");
    std::fs::create_dir(&existing).expect("作れるはず");
    std::fs::write(existing.join("keep.txt"), b"x").expect("書けるはず");
    let before = entries(&existing);
    let err = save_model(&m, &existing).expect_err("上限超過は拒否されるはず");
    assert!(matches!(err, ModelIoError::TooLarge { .. }), "{err}");
    assert_eq!(entries(&existing), before, "既存 dir のエントリ集合は不変");

    // 上限ちょうどは保存できる（compile と fit も可能。履歴キーは fit 前で 0 件）。
    let ok = compiled(config(65536, 2));
    save_model(&ok, guard.path().join("ok")).expect("65536 は受理");
}

#[test]
fn model_grown_after_fit_is_rejected_on_save() {
    let mut m = compiled(config(3, 4));
    fit(&mut m, 2);
    let m = m.add_linear(2, 2, 9).expect("add_linear");
    let guard = TempDirGuard::new("grown");
    let dir = guard.path().join("m");
    let err = save_model(&m, &dir).expect_err("復元できない構成は保存できないはず");
    assert!(matches!(err, ModelIoError::Mismatch { .. }), "{err}");
    assert!(!dir.exists());
}
