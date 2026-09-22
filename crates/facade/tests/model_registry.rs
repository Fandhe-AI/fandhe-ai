//! `fandhe_ai::model::ModelRegistry`（イシュー #2087・親 #2082）の
//! 統合テストを `fandhe_ai`（facade）と `std` のみを import して検証
//! する（`interop_safetensors_roundtrip.rs` と同じ流儀）。
//!
//! 対象契約（`crates/facade/src/model.rs` モジュール doc 参照）:
//! 1. `<cache_dir>/<name>/<version>/model.safetensors` レイアウトから
//!    `load` が state dict を bit 完全一致で往復させる
//! 2. `available_models` の列挙規則（決定的ソート・不完全エントリの
//!    除外・非承認名のスキップ）
//! 3. `name`・`version` の許可文字集合検証がファイルシステムへ触れる
//!    前に `InvalidComponent` を返す
//! 4. `ModelError` が `std::error::Error` として `source()`・`Display`
//!    を持つ

use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::save_safetensors_f32;
use fandhe_ai::model::{ModelError, ModelRegistry};
use std::fs;

/// テストごとに衝突しない一時ディレクトリ（プロセス ID + テスト名）を
/// 作る（`interop_safetensors_roundtrip.rs::temp_dir_for` と同型）。
fn temp_dir_for(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "fandhe-ai-model-registry-{}-{test_name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, /* seed = */ 7)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 8)
        .unwrap()
}

fn sample_input() -> Tensor<f32> {
    Tensor::new(vec![0.1_f32, -0.2, 0.3, -0.4], &[1, 4]).unwrap()
}

fn assert_tensor_bit_exact(a: &Tensor<f32>, b: &Tensor<f32>, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape 不一致");
    let a_bits: Vec<u32> = a
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let b_bits: Vec<u32> = b
        .contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(a_bits, b_bits, "{label}: 要素 bit 不一致");
}

/// レイアウト規定どおりに配置したモデルを `load` で読み戻すと state
/// dict が全キー・全要素 bit 完全一致し、別モデルへ `load_state_dict`
/// した後の推論出力も元モデルと bit 一致する。
#[test]
fn load_roundtrips_state_dict_bit_exact() {
    let root = temp_dir_for("load_roundtrips");
    let model = build_model();
    let sd = model.state_dict();

    let version_dir = root.join("mlp").join("v1");
    fs::create_dir_all(&version_dir).unwrap();
    save_safetensors_f32(&version_dir.join("model.safetensors"), &sd).unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    let loaded = registry.load("mlp", "v1").unwrap();

    assert_eq!(sd.len(), loaded.len());
    for (key, tensor) in &sd {
        let from_loaded = loaded
            .get(key.as_str())
            .unwrap_or_else(|| panic!("ロード結果に `{key}` が存在しない"));
        assert_tensor_bit_exact(tensor, from_loaded, key);
    }

    let mut other = Sequential::new()
        .add_linear(4, 8, /* seed = */ 99)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 100)
        .unwrap();
    other.load_state_dict(loaded).unwrap();

    let input = sample_input();
    let expected = model.predict(&input).unwrap();
    let actual = other.predict(&input).unwrap();
    assert_tensor_bit_exact(&expected, &actual, "predict 出力");

    fs::remove_dir_all(&root).unwrap();
}

/// `available_models` は完全なエントリ（`model.safetensors` が存在
/// する `<name>/<version>`）のみを name・version とも昇順で列挙し、
/// 不完全エントリ（ファイル欠落）・非承認名（空文字・隠しディレクト
/// リ）・無関係なファイルはスキップする。
#[test]
fn available_models_lists_only_complete_entries_sorted() {
    let root = temp_dir_for("available_models_complete");

    for (name, version) in [("b", "2"), ("b", "1"), ("a", "x")] {
        let dir = root.join(name).join(version);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("model.safetensors"), b"dummy").unwrap();
    }
    // `a/empty` はファイルなし（不完全エントリとして除外される）。
    fs::create_dir_all(root.join("a").join("empty")).unwrap();
    // ルート直下の通常ファイルは name 候補から除外される。
    fs::write(root.join("zz.txt"), b"not a dir").unwrap();
    // 隠しディレクトリは先頭 '.' 不可のため除外される。
    fs::create_dir_all(root.join(".hidden").join("v")).unwrap();
    fs::write(
        root.join(".hidden").join("v").join("model.safetensors"),
        b"x",
    )
    .unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    let models = registry.available_models();

    assert_eq!(
        models,
        vec![
            ("a".to_string(), vec!["x".to_string()]),
            ("b".to_string(), vec!["1".to_string(), "2".to_string()]),
        ]
    );

    fs::remove_dir_all(&root).unwrap();
}

/// ルートディレクトリが存在しない場合、`available_models` はエラー
/// ではなく空 `Vec`（レジストリが空とみなす契約）を返す。
#[test]
fn available_models_on_missing_root_is_empty() {
    let root = temp_dir_for("available_models_missing_root").join("does-not-exist");
    let registry = ModelRegistry::with_cache_dir(&root);
    assert_eq!(
        registry.available_models(),
        Vec::<(String, Vec<String>)>::new()
    );
}

/// 存在しない `name`／`version` の組み合わせは
/// `ModelError::NotFound { name, version }` を返す。
#[test]
fn load_missing_model_is_not_found() {
    let root = temp_dir_for("load_missing_model");
    let registry = ModelRegistry::with_cache_dir(&root);

    match registry.load("nope", "v1") {
        Err(ModelError::NotFound { name, version }) => {
            assert_eq!(name, "nope");
            assert_eq!(version, "v1");
        }
        other => panic!("NotFound を期待したが {other:?} だった"),
    }

    fs::remove_dir_all(&root).unwrap();
}

/// パストラバーサル・区切り文字・NUL・空文字等を含む `name`／
/// `version` はファイルシステムへ触れる前に `InvalidComponent` として
/// 拒否される（ルート自体を存在しないパスにしても `NotFound` に
/// フォールバックしないことで、FS アクセス前の拒否であることを確認
/// する。OWASP A03 対策）。
#[test]
fn invalid_components_are_rejected_before_fs_access() {
    let root = temp_dir_for("invalid_components").join("does-not-exist");
    let registry = ModelRegistry::with_cache_dir(&root);

    for value in ["..", "a/b", "a\\b", "", ".", ".hidden", "a b", "a\0b"] {
        match registry.load(value, "v1") {
            Err(ModelError::InvalidComponent { kind, value: v }) => {
                assert_eq!(kind, "name");
                assert_eq!(v, value);
            }
            other => panic!("value={value:?} で InvalidComponent を期待したが {other:?} だった"),
        }
    }
}

/// 壊れた safetensors バイト列は `ModelError::Load` として fail-closed
/// に拒否される（`crate::interop::safetensors` への委譲を確認する）。
#[test]
fn corrupted_file_is_reported_as_load_error() {
    let root = temp_dir_for("corrupted_file");
    let version_dir = root.join("broken").join("v1");
    fs::create_dir_all(&version_dir).unwrap();
    fs::write(
        version_dir.join("model.safetensors"),
        b"not a safetensors file",
    )
    .unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    match registry.load("broken", "v1") {
        Err(ModelError::Load(_)) => {}
        other => panic!("Load エラーを期待したが {other:?} だった"),
    }

    fs::remove_dir_all(&root).unwrap();
}

/// `cache_dir()` は `with_cache_dir` に渡したパスをそのまま返す。
#[test]
fn cache_dir_returns_configured_root() {
    let root = temp_dir_for("cache_dir_returns_configured_root");
    let registry = ModelRegistry::with_cache_dir(&root);
    assert_eq!(registry.cache_dir(), root.as_path());
    fs::remove_dir_all(&root).unwrap();
}

/// `ModelError` は `std::error::Error` を実装し、`Load`／`Io` は
/// `source()` を持ち、`Display` は空でない文字列を返す。
#[test]
fn model_error_is_std_error_with_source() {
    let root = temp_dir_for("model_error_is_std_error");
    let version_dir = root.join("broken").join("v1");
    fs::create_dir_all(&version_dir).unwrap();
    fs::write(version_dir.join("model.safetensors"), b"garbage").unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    let err = registry.load("broken", "v1").unwrap_err();
    assert!(!err.to_string().is_empty());
    let std_err: &dyn std::error::Error = &err;
    assert!(std_err.source().is_some(), "Load エラーの source が None");

    fs::remove_dir_all(&root).unwrap();
}

// ============================================================================
// シンボリックリンク経由のキャッシュルート脱出対策
// （codex-review 指摘・PR #2226。OWASP A03。`crates/facade/src/model.rs`
// モジュール doc「非信頼入力の扱い」節・`resolve_model_file` 参照）
// ============================================================================

/// キャッシュルート外に正当な safetensors ファイルを 1 つ配置し、その
/// バイト列を返す（シンボリックリンク脱出テストの「盗み見られては
/// いけない秘密ファイル」役）。
#[cfg(unix)]
fn write_outside_secret(test_name: &str) -> std::path::PathBuf {
    let outside_dir = std::env::temp_dir().join(format!(
        "fandhe-ai-model-registry-outside-{}-{test_name}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&outside_dir);
    fs::create_dir_all(&outside_dir).unwrap();
    let sd = build_model().state_dict();
    let secret_path = outside_dir.join("secret.safetensors");
    save_safetensors_f32(&secret_path, &sd).unwrap();
    secret_path
}

/// 葉ファイル（`model.safetensors`）自体がキャッシュルート外を指す
/// シンボリックリンクの場合、`load` はそれを追跡せず `NotFound` として
/// 拒否する（`std::fs::metadata`／`load_safetensors_f32` はリンクを
/// 追跡するため、`validate_component` の文字集合検証だけでは防げない
/// 脱出経路）。
#[cfg(unix)]
#[test]
fn load_rejects_symlinked_leaf_file_escaping_root() {
    let root = temp_dir_for("symlink_leaf_escape");
    let secret = write_outside_secret("symlink_leaf_escape");

    let version_dir = root.join("mlp").join("v1");
    fs::create_dir_all(&version_dir).unwrap();
    std::os::unix::fs::symlink(&secret, version_dir.join("model.safetensors")).unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    match registry.load("mlp", "v1") {
        Err(ModelError::NotFound { name, version }) => {
            assert_eq!(name, "mlp");
            assert_eq!(version, "v1");
        }
        other => panic!("シンボリックリンクの葉ファイルは NotFound を期待したが {other:?} だった"),
    }

    fs::remove_dir_all(&root).unwrap();
    fs::remove_dir_all(secret.parent().unwrap()).unwrap();
}

/// `<version>` ディレクトリ自体がキャッシュルート外（正当な
/// `model.safetensors` を含むディレクトリ）を指すシンボリックリンクの
/// 場合も `load` は追跡せず `NotFound` として拒否する。
#[cfg(unix)]
#[test]
fn load_rejects_symlinked_version_dir_escaping_root() {
    let root = temp_dir_for("symlink_version_escape");
    let outside_dir = std::env::temp_dir().join(format!(
        "fandhe-ai-model-registry-outside-version-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&outside_dir);
    fs::create_dir_all(&outside_dir).unwrap();
    let sd = build_model().state_dict();
    save_safetensors_f32(&outside_dir.join("model.safetensors"), &sd).unwrap();

    let name_dir = root.join("mlp");
    fs::create_dir_all(&name_dir).unwrap();
    std::os::unix::fs::symlink(&outside_dir, name_dir.join("v1")).unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    match registry.load("mlp", "v1") {
        Err(ModelError::NotFound { .. }) => {}
        other => panic!(
            "シンボリックリンクの version ディレクトリは NotFound を期待したが {other:?} だった"
        ),
    }

    fs::remove_dir_all(&root).unwrap();
    fs::remove_dir_all(&outside_dir).unwrap();
}

/// `<name>` ディレクトリ自体がキャッシュルート外を指すシンボリック
/// リンクの場合も `load` は追跡せず `NotFound` として拒否する（葉が
/// 通常ファイルでも中間段のリンクで脱出できないことを確認する）。
#[cfg(unix)]
#[test]
fn load_rejects_symlinked_name_dir_escaping_root() {
    let root = temp_dir_for("symlink_name_escape");
    let outside_dir = std::env::temp_dir().join(format!(
        "fandhe-ai-model-registry-outside-name-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&outside_dir);
    fs::create_dir_all(outside_dir.join("v1")).unwrap();
    let sd = build_model().state_dict();
    save_safetensors_f32(&outside_dir.join("v1").join("model.safetensors"), &sd).unwrap();

    fs::create_dir_all(&root).unwrap();
    std::os::unix::fs::symlink(&outside_dir, root.join("mlp")).unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    match registry.load("mlp", "v1") {
        Err(ModelError::NotFound { .. }) => {}
        other => panic!(
            "シンボリックリンクの name ディレクトリは NotFound を期待したが {other:?} だった"
        ),
    }

    fs::remove_dir_all(&root).unwrap();
    fs::remove_dir_all(&outside_dir).unwrap();
}

/// `available_models` は `load` が拒否する 3 種のシンボリックリンク
/// 脱出経路（葉ファイル・version ディレクトリ・name ディレクトリ）を
/// すべて列挙結果から除外する（`load` との一貫性保証。
/// `resolve_model_file` を両者が共有することの回帰検出）。
#[cfg(unix)]
#[test]
fn available_models_excludes_all_symlink_escape_variants() {
    let root = temp_dir_for("symlink_available_models_excludes");
    let outside_dir = std::env::temp_dir().join(format!(
        "fandhe-ai-model-registry-outside-avail-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&outside_dir);
    fs::create_dir_all(outside_dir.join("payload")).unwrap();
    let sd = build_model().state_dict();
    save_safetensors_f32(&outside_dir.join("payload").join("model.safetensors"), &sd).unwrap();

    // 正当な比較対象（除外されてはいけない）。
    let legit_dir = root.join("legit").join("v1");
    fs::create_dir_all(&legit_dir).unwrap();
    save_safetensors_f32(&legit_dir.join("model.safetensors"), &sd).unwrap();

    // 葉ファイルがリンク。
    let leaf_dir = root.join("leaf-escape").join("v1");
    fs::create_dir_all(&leaf_dir).unwrap();
    std::os::unix::fs::symlink(
        outside_dir.join("payload").join("model.safetensors"),
        leaf_dir.join("model.safetensors"),
    )
    .unwrap();

    // version ディレクトリがリンク。
    let version_name_dir = root.join("version-escape");
    fs::create_dir_all(&version_name_dir).unwrap();
    std::os::unix::fs::symlink(outside_dir.join("payload"), version_name_dir.join("v1")).unwrap();

    // name ディレクトリがリンク。
    std::os::unix::fs::symlink(outside_dir.join("payload"), root.join("name-escape")).unwrap();
    // ↑ `name-escape/v1` ではなく `name-escape` 自体が `payload` を
    // 指すため、実体側に `v1/model.safetensors` は無い（`payload`
    // 直下は `model.safetensors` のみ）。`name-escape` を name として
    // 直接使わせず `resolve_model_file` の name 段リンク拒否を検証
    // する目的のみで配置する。

    let registry = ModelRegistry::with_cache_dir(&root);
    let models = registry.available_models();

    assert_eq!(
        models,
        vec![("legit".to_string(), vec!["v1".to_string()])],
        "シンボリックリンク経由の脱出候補が列挙結果に混入した: {models:?}"
    );

    fs::remove_dir_all(&root).unwrap();
    fs::remove_dir_all(&outside_dir).unwrap();
}

/// キャッシュルート自体がシンボリックリンクであっても、リンク先の
/// レイアウトが正当であれば `load` は正しくロードできる（過剰拒否の
/// 検出。シンボリックリンク拒否は「レジストリ内部からの脱出」のみを
/// 対象とし、ルート自体の間接参照は妨げない）。
#[cfg(unix)]
#[test]
fn load_succeeds_when_root_itself_is_a_symlink() {
    let real_root = temp_dir_for("symlink_root_real");
    let version_dir = real_root.join("mlp").join("v1");
    fs::create_dir_all(&version_dir).unwrap();
    let sd = build_model().state_dict();
    save_safetensors_f32(&version_dir.join("model.safetensors"), &sd).unwrap();

    let link_root = std::env::temp_dir().join(format!(
        "fandhe-ai-model-registry-symlink-root-link-{}",
        std::process::id()
    ));
    let _ = fs::remove_file(&link_root);
    let _ = fs::remove_dir_all(&link_root);
    std::os::unix::fs::symlink(&real_root, &link_root).unwrap();

    let registry = ModelRegistry::with_cache_dir(&link_root);
    let loaded = registry.load("mlp", "v1").unwrap();
    assert_eq!(loaded.len(), sd.len());

    fs::remove_dir_all(&real_root).unwrap();
    fs::remove_file(&link_root).unwrap();
}
