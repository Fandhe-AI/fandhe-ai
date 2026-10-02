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

mod common;
use common::temp_dir::TempDirGuard;

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
    let guard = TempDirGuard::new("model-registry-load_roundtrips");
    let root = guard.path().to_path_buf();
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
}

/// `available_models` は完全なエントリ（`model.safetensors` が存在
/// する `<name>/<version>`）のみを name・version とも昇順で列挙し、
/// 不完全エントリ（ファイル欠落）・非承認名（空文字・隠しディレクト
/// リ）・無関係なファイルはスキップする。
#[test]
fn available_models_lists_only_complete_entries_sorted() {
    let guard = TempDirGuard::new("model-registry-available_models_complete");
    let root = guard.path().to_path_buf();

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
}

/// ルートディレクトリが存在しない場合、`available_models` はエラー
/// ではなく空 `Vec`（レジストリが空とみなす契約）を返す。
#[test]
fn available_models_on_missing_root_is_empty() {
    let guard = TempDirGuard::new("model-registry-available_models_missing_root");
    let root = guard.path().join("does-not-exist");
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
    let guard = TempDirGuard::new("model-registry-load_missing_model");
    let root = guard.path().to_path_buf();
    let registry = ModelRegistry::with_cache_dir(&root);

    match registry.load("nope", "v1") {
        Err(ModelError::NotFound { name, version }) => {
            assert_eq!(name, "nope");
            assert_eq!(version, "v1");
        }
        other => panic!("NotFound を期待したが {other:?} だった"),
    }
}

/// パストラバーサル・区切り文字・NUL・空文字等を含む `name`／
/// `version` はファイルシステムへ触れる前に `InvalidComponent` として
/// 拒否される（ルート自体を存在しないパスにしても `NotFound` に
/// フォールバックしないことで、FS アクセス前の拒否であることを確認
/// する。OWASP A03 対策）。
#[test]
fn invalid_components_are_rejected_before_fs_access() {
    let guard = TempDirGuard::new("model-registry-invalid_components");
    let root = guard.path().join("does-not-exist");
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
    let guard = TempDirGuard::new("model-registry-corrupted_file");
    let root = guard.path().to_path_buf();
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
}

/// `cache_dir()` は `with_cache_dir` に渡したパスをそのまま返す。
#[test]
fn cache_dir_returns_configured_root() {
    let guard = TempDirGuard::new("model-registry-cache_dir_returns_configured_root");
    let root = guard.path().to_path_buf();
    let registry = ModelRegistry::with_cache_dir(&root);
    assert_eq!(registry.cache_dir(), root.as_path());
}

/// `ModelError` は `std::error::Error` を実装し、`Load`／`Io` は
/// `source()` を持ち、`Display` は空でない文字列を返す。
#[test]
fn model_error_is_std_error_with_source() {
    let guard = TempDirGuard::new("model-registry-model_error_is_std_error");
    let root = guard.path().to_path_buf();
    let version_dir = root.join("broken").join("v1");
    fs::create_dir_all(&version_dir).unwrap();
    fs::write(version_dir.join("model.safetensors"), b"garbage").unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    let err = registry.load("broken", "v1").unwrap_err();
    assert!(!err.to_string().is_empty());
    let std_err: &dyn std::error::Error = &err;
    assert!(std_err.source().is_some(), "Load エラーの source が None");
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
fn write_outside_secret(test_name: &str) -> (TempDirGuard, std::path::PathBuf) {
    let outside = TempDirGuard::new(&format!("model-registry-outside-{test_name}"));
    let sd = build_model().state_dict();
    let secret_path = outside.path().join("secret.safetensors");
    save_safetensors_f32(&secret_path, &sd).unwrap();
    (outside, secret_path)
}

/// 葉ファイル（`model.safetensors`）自体がキャッシュルート外を指す
/// シンボリックリンクの場合、`load` はそれを追跡せず `NotFound` として
/// 拒否する（`std::fs::metadata`／`load_safetensors_f32` はリンクを
/// 追跡するため、`validate_component` の文字集合検証だけでは防げない
/// 脱出経路）。
#[cfg(unix)]
#[test]
fn load_rejects_symlinked_leaf_file_escaping_root() {
    let guard = TempDirGuard::new("model-registry-symlink_leaf_escape");
    let root = guard.path().to_path_buf();
    let (_outside_guard, secret) = write_outside_secret("symlink_leaf_escape");

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
}

/// `<version>` ディレクトリ自体がキャッシュルート外（正当な
/// `model.safetensors` を含むディレクトリ）を指すシンボリックリンクの
/// 場合も `load` は追跡せず `NotFound` として拒否する。
#[cfg(unix)]
#[test]
fn load_rejects_symlinked_version_dir_escaping_root() {
    let guard = TempDirGuard::new("model-registry-symlink_version_escape");
    let root = guard.path().to_path_buf();
    let outside = TempDirGuard::new("model-registry-outside-version");
    let outside_dir = outside.path().to_path_buf();
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
}

/// `<name>` ディレクトリ自体がキャッシュルート外を指すシンボリック
/// リンクの場合も `load` は追跡せず `NotFound` として拒否する（葉が
/// 通常ファイルでも中間段のリンクで脱出できないことを確認する）。
#[cfg(unix)]
#[test]
fn load_rejects_symlinked_name_dir_escaping_root() {
    let guard = TempDirGuard::new("model-registry-symlink_name_escape");
    let root = guard.path().to_path_buf();
    let outside = TempDirGuard::new("model-registry-outside-name");
    let outside_dir = outside.path().to_path_buf();
    fs::create_dir_all(outside_dir.join("v1")).unwrap();
    let sd = build_model().state_dict();
    save_safetensors_f32(&outside_dir.join("v1").join("model.safetensors"), &sd).unwrap();

    std::os::unix::fs::symlink(&outside_dir, root.join("mlp")).unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    match registry.load("mlp", "v1") {
        Err(ModelError::NotFound { .. }) => {}
        other => panic!(
            "シンボリックリンクの name ディレクトリは NotFound を期待したが {other:?} だった"
        ),
    }
}

/// `available_models` は `load` が拒否する 3 種のシンボリックリンク
/// 脱出経路（葉ファイル・version ディレクトリ・name ディレクトリ）を
/// すべて列挙結果から除外する（`load` との一貫性保証。
/// `resolve_model_file` を両者が共有することの回帰検出）。
#[cfg(unix)]
#[test]
fn available_models_excludes_all_symlink_escape_variants() {
    let guard = TempDirGuard::new("model-registry-symlink_available_models_excludes");
    let root = guard.path().to_path_buf();
    let outside = TempDirGuard::new("model-registry-outside-avail");
    let outside_dir = outside.path().to_path_buf();
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
}

/// キャッシュルート自体がシンボリックリンクであっても、リンク先の
/// レイアウトが正当であれば `load` は正しくロードできる（過剰拒否の
/// 検出。シンボリックリンク拒否は「レジストリ内部からの脱出」のみを
/// 対象とし、ルート自体の間接参照は妨げない）。
#[cfg(unix)]
#[test]
fn load_succeeds_when_root_itself_is_a_symlink() {
    let real_guard = TempDirGuard::new("model-registry-symlink_root_real");
    let real_root = real_guard.path().to_path_buf();
    let version_dir = real_root.join("mlp").join("v1");
    fs::create_dir_all(&version_dir).unwrap();
    let sd = build_model().state_dict();
    save_safetensors_f32(&version_dir.join("model.safetensors"), &sd).unwrap();

    // 親ディレクトリを一意名で排他作成し、その内側に symlink を張る。symlink 自体も
    // 既存パスには EEXIST で失敗するため、第三者が先に置いたリンクを上書き・再利用しない。
    let link_parent = TempDirGuard::new("model-registry-symlink-root-link");
    let link_root = link_parent.path().join("link");
    std::os::unix::fs::symlink(&real_root, &link_root).unwrap();

    let registry = ModelRegistry::with_cache_dir(&link_root);
    let loaded = registry.load("mlp", "v1").unwrap();
    assert_eq!(loaded.len(), sd.len());
}

// ============================================================================
// 葉が通常ファイルでない場合の拒否・検査後の差し替え（TOCTOU）対策
// （codex-review 指摘・PR #2226。`resolve_model_file` 参照）
// ============================================================================

/// Unix ソケットを葉に置くテスト専用の短パス・キャッシュルート（RAII）。
///
/// 共通ガード `TempDirGuard` の一意名は約 62 文字あり、macOS 既定の
/// `TMPDIR`（約 49 文字）では葉 `mlp/v1/model.safetensors` までで
/// `sun_path`（macOS 104 バイト・Linux 108 バイト）を超え `bind` が
/// `InvalidInput` で失敗する（イシュー #2483）。ここでは検査自体を
/// スキップせず、短い一意名 `fa-<pid>-<seq>-<label>` を `temp_dir()`、
/// 収まらなければ `/tmp` 直下に排他作成する。macOS の `/tmp` は
/// `/private/tmp` への symlink のため `canonicalize` 済みの実体パスを
/// 使う。長さを満たせない場合は黙って通さず panic で気づける。
#[cfg(unix)]
struct ShortSocketRoot {
    path: std::path::PathBuf,
}

#[cfg(unix)]
impl ShortSocketRoot {
    /// bind 対象に使える `sun_path` の上限（macOS の 104 から終端 NUL を除く）。
    const MAX_LEAF_BYTES: usize = 103;

    fn new(label: &str) -> Self {
        use std::os::unix::ffi::OsStrExt;
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);

        let mut worst = 0;
        for cand in [std::env::temp_dir(), std::path::PathBuf::from("/tmp")] {
            let Ok(base) = cand.canonicalize() else {
                continue;
            };
            for _ in 0..64 {
                let seq = SEQ.fetch_add(1, Ordering::Relaxed);
                let name = format!("fa-{}-{seq}-{label}", std::process::id());
                let path = base.join(name);
                let leaf = path.join("mlp").join("v1").join("model.safetensors");
                worst = worst.max(leaf.as_os_str().as_bytes().len());
                if leaf.as_os_str().as_bytes().len() > Self::MAX_LEAF_BYTES {
                    break;
                }
                match fs::create_dir(&path) {
                    Ok(()) => return Self { path },
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(e) => panic!("ソケット用ルートの作成に失敗: {path:?}: {e}"),
                }
            }
        }
        panic!(
            "葉パスが sun_path 上限 {} バイトに収まる作業ディレクトリを確保できない（最短でも {worst} バイト）",
            Self::MAX_LEAF_BYTES
        );
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }
}

/// 葉が実際に Unix ソケットであること（拒否検査の前提）を lstat で確認する。
#[cfg(unix)]
fn is_socket(p: &std::path::Path) -> bool {
    use std::os::unix::fs::FileTypeExt;
    fs::symlink_metadata(p).is_ok_and(|m| m.file_type().is_socket())
}

#[cfg(unix)]
impl Drop for ShortSocketRoot {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

/// 葉ファイル（`model.safetensors`）が Unix ドメインソケットの場合、
/// `load` は `is_dir() == false` だけでは通過してしまう非通常ファイル
/// を拒否し `NotFound` を返す（`must_be_dir == false` は
/// `is_file() == true` を明示要求する。codex-review 指摘・PR #2226。
/// FIFO と異なりソケットは `std` のみで `mkfifo` 相当なしに再現できる
/// ため `UnixListener::bind` を使う）。
#[cfg(unix)]
#[test]
fn load_rejects_non_regular_leaf_unix_socket() {
    // 短い専用ルート: UNIX ソケットパスは sun_path 上限未満が必要（#2483）。
    let guard = ShortSocketRoot::new("sock");
    let root = guard.path().to_path_buf();
    let version_dir = root.join("mlp").join("v1");
    fs::create_dir_all(&version_dir).unwrap();

    let leaf = version_dir.join("model.safetensors");
    let _listener = std::os::unix::net::UnixListener::bind(&leaf).unwrap();
    assert!(is_socket(&leaf), "葉が socket でなく検査が空洞化している");

    let registry = ModelRegistry::with_cache_dir(&root);
    match registry.load("mlp", "v1") {
        Err(ModelError::NotFound { name, version }) => {
            assert_eq!(name, "mlp");
            assert_eq!(version, "v1");
        }
        other => panic!("非通常ファイルの葉は NotFound を期待したが {other:?} だった"),
    }
}

/// `available_models` も同じ非通常ファイル（Unix ソケット）を葉に持つ
/// バージョンを列挙結果から除外する（`load` との一貫性保証）。
#[cfg(unix)]
#[test]
fn available_models_excludes_non_regular_leaf() {
    // 短い専用ルート: UNIX ソケットパスは sun_path 上限未満が必要（#2483）。
    let guard = ShortSocketRoot::new("avail-sock");
    let root = guard.path().to_path_buf();
    let version_dir = root.join("mlp").join("v1");
    fs::create_dir_all(&version_dir).unwrap();
    let _listener =
        std::os::unix::net::UnixListener::bind(version_dir.join("model.safetensors")).unwrap();
    assert!(
        is_socket(&version_dir.join("model.safetensors")),
        "葉が socket でなく検査が空洞化している"
    );

    let registry = ModelRegistry::with_cache_dir(&root);
    assert_eq!(
        registry.available_models(),
        Vec::<(String, Vec<String>)>::new()
    );
}

/// 検査（lstat）後・open 前に葉ファイルをキャッシュルート外へ
/// エスケープするシンボリックリンクへ差し替えても、`load` は
/// 差し替え後に実際に開かれた実体の識別子（fstat の dev／ino）が
/// 検査時点のものと一致しないことを検出し `NotFound` を返す
/// （TOCTOU 対策。真のレース条件は非決定的なため、ここでは検査後に
/// 差し替えが「既に完了している」状態を固定して防御の効果を確認する。
/// `O_NOFOLLOW` 自体もこのケースを ELOOP で拒否する）。
#[cfg(unix)]
#[test]
fn load_rejects_leaf_replaced_with_symlink_after_initial_write() {
    let guard = TempDirGuard::new("model-registry-toctou_leaf_replaced_with_symlink");
    let root = guard.path().to_path_buf();
    let (_outside_guard, secret) = write_outside_secret("toctou_leaf_replaced_with_symlink");

    let version_dir = root.join("mlp").join("v1");
    fs::create_dir_all(&version_dir).unwrap();
    let leaf = version_dir.join("model.safetensors");

    // 最初は正当な通常ファイルとして書き込む（検査時点のスナップ
    // ショットが通常ファイルであるケースを模す）。
    fs::write(&leaf, b"placeholder").unwrap();
    // その後、シンボリックリンクへ差し替える（本来ならこの差し替えは
    // resolve_model_file の検査と open の間で起きるレースだが、
    // ここでは差し替え「後」の状態を検証することで対策の効果を確認
    // する）。
    fs::remove_file(&leaf).unwrap();
    std::os::unix::fs::symlink(&secret, &leaf).unwrap();

    let registry = ModelRegistry::with_cache_dir(&root);
    match registry.load("mlp", "v1") {
        Err(ModelError::NotFound { .. }) => {}
        other => panic!("差し替え後の葉は NotFound を期待したが {other:?} だった"),
    }
}
