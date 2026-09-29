//! `fandhe_ai::compat::{save_model, load_model, ModelIoError}`（イシュー #2369・親 #2362）の
//! 統合テスト。facade の公開 API と `std` のみで、受入基準（bit 一致の往復・世代コミット・
//! 未対応モデルの fail-closed・非信頼入力の拒否）を検証する。
//!
//! 網羅的な衝突注入・TOCTOU・改竄行列は #2375・#2376 で追加する（本ファイルは基本ケース）。
//! ファイル名は `compat_sequential_model_io_manual.rs`（既存公開 API のみの往復土台）と
//! 区別するため接尾辞なし。

use std::path::Path;

use fandhe_ai::Tensor;
use fandhe_ai::compat::{ModelIoError, Sequential, load_model, save_model};

mod common;
use common::temp_dir::TempDirGuard;

fn tensor(rows: usize, cols: usize, base: f32) -> Tensor<f32> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) * 0.37 + base).sin())
        .collect();
    Tensor::new(data, &[rows, cols]).expect("テンソルを作れるはず")
}

/// Linear→ReLU→Linear→Sigmoid→Linear→Tanh（3 種の活性化を含む）。
fn build_mixed_model() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, 11)
        .and_then(|m| m.add_relu().add_linear(8, 6, 12))
        .and_then(|m| m.add_sigmoid().add_linear(6, 3, 13))
        .map(Sequential::add_tanh)
        .expect("構築できるはず")
}

/// 対応する活性化 7 種をすべて含む構成。
fn build_all_activations_model() -> Sequential {
    Sequential::new()
        .add_linear(5, 5, 21)
        .map(|m| {
            m.add_relu()
                .add_sigmoid()
                .add_tanh()
                .add_silu()
                .add_hardswish()
                .add_gelu()
                .add_gelu_tanh()
        })
        .and_then(|m| m.add_linear(5, 2, 22))
        .expect("構築できるはず")
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

fn assert_models_identical(original: &Sequential, loaded: &Sequential, input: &Tensor<f32>) {
    let (sa, sb) = (original.state_dict(), loaded.state_dict());
    assert_eq!(sa.len(), sb.len());
    for (k, v) in &sa {
        assert_bit_identical(v, &sb[k], k);
    }
    assert_eq!(original.training(), loaded.training(), "training フラグ");
    let ya = original.predict(input).expect("predict できるはず");
    let yb = loaded.predict(input).expect("predict できるはず");
    assert_bit_identical(&ya, &yb, "predict 出力");
}

/// `Sequential` は `Debug` を持たないため、`expect_err` で使えるよう成功値を捨てる。
#[cfg(unix)]
fn try_load(dir: impl AsRef<Path>) -> Result<(), ModelIoError> {
    load_model(dir).map(|_| ())
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

fn safetensors_entries(dir: &Path) -> Vec<String> {
    entries(dir)
        .into_iter()
        .filter(|n| n.starts_with("model.") && n.ends_with(".safetensors"))
        .collect()
}

#[cfg(unix)]
#[test]
fn roundtrip_linear_activation_bit_identical() {
    let guard = TempDirGuard::new("roundtrip");
    let dir = guard.path().join("model");
    for (label, model) in [
        ("mixed", build_mixed_model()),
        ("all-activations", build_all_activations_model()),
    ] {
        let in_features = if label == "mixed" { 4 } else { 5 };
        let input = tensor(3, in_features, 0.5);
        save_model(&model, &dir).unwrap_or_else(|e| panic!("{label}: 保存できるはず: {e}"));
        let loaded = load_model(&dir).unwrap_or_else(|e| panic!("{label}: 復元できるはず: {e}"));
        assert_models_identical(&model, &loaded, &input);
        std::fs::remove_dir_all(&dir).expect("掃除できるはず");
    }
}

#[cfg(unix)]
#[test]
fn roundtrip_preserves_eval_mode_flag() {
    let guard = TempDirGuard::new("eval-flag");
    let dir = guard.path().join("m");
    let mut model = build_mixed_model();
    model.eval();
    save_model(&model, &dir).expect("保存できるはず");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert!(!loaded.training());
}

#[cfg(unix)]
#[test]
fn resave_into_existing_dir_keeps_old_generation() {
    let guard = TempDirGuard::new("resave");
    let dir = guard.path().to_path_buf();
    let first = build_mixed_model();
    save_model(&first, &dir).expect("1 回目を保存できるはず");
    let gen1 = safetensors_entries(&dir);
    assert_eq!(gen1.len(), 1);

    let second = Sequential::new()
        .add_linear(4, 8, 99)
        .and_then(|m| m.add_relu().add_linear(8, 6, 98))
        .and_then(|m| m.add_sigmoid().add_linear(6, 3, 97))
        .map(Sequential::add_tanh)
        .expect("構築できるはず");
    save_model(&second, &dir).expect("2 回目を保存できるはず");
    let both = safetensors_entries(&dir);
    assert_eq!(both.len(), 2, "旧世代は削除されない");
    assert!(both.contains(&gen1[0]));
    assert!(
        !entries(&dir)
            .iter()
            .any(|n| n.starts_with(".manifest.json.tmp-")),
        "一時ファイルが残らない"
    );

    let loaded = load_model(&dir).expect("復元できるはず");
    assert_models_identical(&second, &loaded, &tensor(2, 4, 1.5));
}

/// 構成を記録できない利用者定義層（`add_module`）。保存対象外の例として使う。
struct CustomIdentity;
impl fandhe_ai::nn::Module for CustomIdentity {
    fn forward<'t>(
        &self,
        _tape: fandhe_ai::TapeRef<'t>,
        input: &fandhe_ai::Var<'t>,
    ) -> Result<fandhe_ai::Var<'t>, fandhe_ai::AutodiffError> {
        Ok(input.tanh())
    }
}

#[cfg(unix)]
#[test]
fn unsupported_layer_is_rejected_without_touching_dir() {
    let guard = TempDirGuard::new("unsupported");
    let dir = guard.path().join("never-created");
    // `add_module`（利用者定義層）は構成を記録できないため、30 層対応後も保存対象外。
    let with_linear = Sequential::new()
        .add_linear(2, 2, 1)
        .expect("構築できるはず")
        .add_module(CustomIdentity);
    let with_relu = Sequential::new().add_relu().add_module(CustomIdentity);
    for (label, model) in [("linear+module", with_linear), ("relu+module", with_relu)] {
        let err = save_model(&model, &dir).expect_err("未対応の層は拒否されるはず");
        assert!(
            matches!(err, ModelIoError::UnsupportedModel { .. }),
            "{label}: {err}"
        );
        assert!(!dir.exists(), "{label}: dir が作られてはいけない");
    }
}

#[cfg(unix)]
#[test]
fn compiled_lbfgs_model_is_rejected_without_touching_dir() {
    use fandhe_ai::compat::{Loss, Optimizer};
    use fandhe_ai::optim::LbfgsConfig;

    let guard = TempDirGuard::new("compiled");
    let mut model = build_mixed_model();
    model
        .compile(Optimizer::Lbfgs(LbfgsConfig::default()), Loss::Mse)
        .expect("compile できるはず");

    let fresh = guard.path().join("fresh");
    let err =
        save_model(&model, &fresh).expect_err("compile 済み Lbfgs は拒否されるはず（#2373 まで）");
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err}"
    );
    assert!(!fresh.exists(), "存在しなかった dir は作られない");

    let existing = guard.path().join("existing");
    std::fs::create_dir(&existing).expect("作れるはず");
    std::fs::write(existing.join("keep.txt"), b"x").expect("書けるはず");
    let before = entries(&existing);
    let err = save_model(&model, &existing)
        .expect_err("compile 済み Lbfgs は拒否されるはず（#2373 まで）");
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err}"
    );
    assert_eq!(entries(&existing), before, "既存 dir のエントリ集合は不変");
}

#[cfg(unix)]
fn saved_dir(label: &str) -> (TempDirGuard, std::path::PathBuf) {
    let guard = TempDirGuard::new(label);
    let dir = guard.path().join("m");
    save_model(&build_mixed_model(), &dir).expect("保存できるはず");
    (guard, dir)
}

#[cfg(unix)]
fn only_safetensors(dir: &Path) -> std::path::PathBuf {
    let names = safetensors_entries(dir);
    assert_eq!(names.len(), 1);
    dir.join(&names[0])
}

#[cfg(unix)]
fn assert_io_rejects(err: ModelIoError, what: &str) {
    match err {
        ModelIoError::Io(e) => assert!(
            e.kind() == std::io::ErrorKind::InvalidInput || e.raw_os_error().is_some(),
            "{what}: 想定外の I/O エラー種別 {e:?}"
        ),
        other => panic!("{what}: Io を期待したが {other}"),
    }
}

#[cfg(unix)]
#[test]
fn load_rejects_symlinked_manifest_and_safetensors() {
    let outside = TempDirGuard::new("outside");
    let (_g, dir) = saved_dir("symlinks");

    let manifest = dir.join("manifest.json");
    let real_manifest = outside.path().join("manifest.json");
    std::fs::rename(&manifest, &real_manifest).expect("移せるはず");
    std::os::unix::fs::symlink(&real_manifest, &manifest).expect("symlink を作れるはず");
    assert_io_rejects(
        try_load(&dir).expect_err("symlink の manifest は拒否されるはず"),
        "manifest",
    );
    std::fs::remove_file(&manifest).expect("消せるはず");
    std::fs::rename(&real_manifest, &manifest).expect("戻せるはず");
    load_model(&dir).expect("元に戻せば読めるはず");

    let st = only_safetensors(&dir);
    let real_st = outside.path().join("real.safetensors");
    std::fs::rename(&st, &real_st).expect("移せるはず");
    std::os::unix::fs::symlink(&real_st, &st).expect("symlink を作れるはず");
    assert_io_rejects(
        try_load(&dir).expect_err("symlink の safetensors は拒否されるはず"),
        "safetensors",
    );
}

/// 通常ファイルでない `manifest.json`／`model.<gen>.safetensors`（ディレクトリ・Unix ソケット）を
/// no-follow 手順の事前検査（`is_file()` の明示要求）で拒否する。ソケットのパスは `SUN_LEN`
/// （約 108 バイト）未満が必要で、`TMPDIR` が長い環境では作れないため、作れる場合だけ追加で検査する。
#[cfg(unix)]
#[test]
fn load_rejects_non_regular_manifest_and_safetensors() {
    use std::os::unix::net::UnixListener;

    let (_g, dir) = saved_dir("nr");
    let manifest = dir.join("manifest.json");
    let saved = dir.join("manifest.saved");
    std::fs::rename(&manifest, &saved).expect("移せるはず");
    std::fs::create_dir(&manifest).expect("ディレクトリを作れるはず");
    assert_io_rejects(
        try_load(&dir).expect_err("ディレクトリの manifest は拒否されるはず"),
        "manifest(dir)",
    );
    std::fs::remove_dir(&manifest).expect("消せるはず");
    if let Ok(_listener) = UnixListener::bind(&manifest) {
        assert_io_rejects(
            try_load(&dir).expect_err("ソケットの manifest は拒否されるはず"),
            "manifest(socket)",
        );
        std::fs::remove_file(&manifest).expect("消せるはず");
    }
    std::fs::rename(&saved, &manifest).expect("戻せるはず");
    try_load(&dir).expect("元に戻せば読めるはず");

    let st = only_safetensors(&dir);
    let kept = dir.join("kept.bin");
    std::fs::rename(&st, &kept).expect("移せるはず");
    std::fs::create_dir(&st).expect("ディレクトリを作れるはず");
    assert_io_rejects(
        try_load(&dir).expect_err("ディレクトリの safetensors は拒否されるはず"),
        "safetensors(dir)",
    );
    std::fs::remove_dir(&st).expect("消せるはず");
    if let Ok(_listener) = UnixListener::bind(&st) {
        assert_io_rejects(
            try_load(&dir).expect_err("ソケットの safetensors は拒否されるはず"),
            "safetensors(socket)",
        );
    }
}

#[cfg(unix)]
#[test]
fn load_rejects_oversized_manifest() {
    let (_g, dir) = saved_dir("oversized");
    // 固定上限（1 MiB）+ 1 バイトの manifest。中身を読む前に拒否されるはず。
    std::fs::write(dir.join("manifest.json"), vec![b' '; 1024 * 1024 + 1]).expect("書けるはず");
    let err = try_load(&dir).expect_err("上限超過は拒否されるはず");
    assert!(
        matches!(
            err,
            ModelIoError::TooLarge {
                what: "manifest.json",
                ..
            }
        ),
        "{err}"
    );
}

#[cfg(unix)]
#[test]
fn load_rejects_safetensors_bytes_mismatch() {
    let (_g, dir) = saved_dir("bytes-mismatch");
    let st = only_safetensors(&dir);
    let mut bytes = std::fs::read(&st).expect("読めるはず");
    bytes.push(0);
    std::fs::write(&st, bytes).expect("書けるはず");
    let err = try_load(&dir).expect_err("バイト数不一致は拒否されるはず");
    assert!(matches!(err, ModelIoError::Mismatch { .. }), "{err}");
}

#[cfg(unix)]
#[test]
fn load_rejects_bad_safetensors_file_pattern() {
    let (_g, dir) = saved_dir("bad-pattern");
    let manifest_path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&manifest_path).expect("読めるはず");
    let st_name = only_safetensors(&dir)
        .file_name()
        .expect("名前があるはず")
        .to_string_lossy()
        .into_owned();
    for bad in [
        "../x".to_string(),
        "model.ABCDEFABCDEFABCDEFABCDEFABCDEFAB.safetensors".to_string(),
        "sub/model.0123456789abcdef0123456789abcdef.safetensors".to_string(),
        "model.0123456789abcdef0123456789abcdef.safetensors\\..\\x".to_string(),
        "model.0123.safetensors".to_string(),
    ] {
        // `\` は JSON 文字列で許さないため、置換後の manifest 自体が拒否される場合も
        // `Manifest` になる（どちらの経路でも Manifest で fail-closed）。
        let tampered = original.replace(&st_name, &bad);
        std::fs::write(&manifest_path, tampered).expect("書けるはず");
        let err = try_load(&dir).expect_err("不正なファイル名は拒否されるはず");
        assert!(matches!(err, ModelIoError::Manifest { .. }), "{bad}: {err}");
    }
}

#[cfg(unix)]
#[test]
fn load_rejects_unknown_key_and_unsupported_manifest_features() {
    let (_g, dir) = saved_dir("tamper");
    let manifest_path = dir.join("manifest.json");
    let original = std::fs::read_to_string(&manifest_path).expect("読めるはず");

    std::fs::write(
        &manifest_path,
        original.replacen("\"training\"", "\"extra\":1,\"training\"", 1),
    )
    .expect("書けるはず");
    assert!(matches!(
        try_load(&dir).expect_err("未知キーは拒否されるはず"),
        ModelIoError::Manifest { .. }
    ));

    std::fs::write(
        &manifest_path,
        original.replace("\"compiled\":null", "\"compiled\":{}"),
    )
    .expect("書けるはず");
    // compiled 節は固定キー集合の object（`{}` はキー欠落）。
    assert!(matches!(
        try_load(&dir).expect_err("キーの欠けた compiled は拒否されるはず"),
        ModelIoError::Manifest { .. }
    ));
}

#[cfg(unix)]
#[test]
fn load_from_missing_dir_reports_io_error() {
    let guard = TempDirGuard::new("missing");
    let err = try_load(guard.path().join("nope")).expect_err("存在しない dir は Err のはず");
    assert!(
        matches!(&err, ModelIoError::Io(e) if e.kind() == std::io::ErrorKind::NotFound),
        "{err}"
    );
}

#[cfg(not(unix))]
#[test]
fn save_model_is_unsupported_on_non_unix() {
    let guard = TempDirGuard::new("non-unix");
    let dir = guard.path().join("m");
    let err = save_model(&build_mixed_model(), &dir).expect_err("非 unix は未対応のはず");
    assert!(
        matches!(&err, ModelIoError::Io(e) if e.kind() == std::io::ErrorKind::Unsupported),
        "{err}"
    );
    assert!(!dir.exists(), "dir に何も作られない");
}
