//! `fandhe_ai::compat::{save_model, load_model, ModelIoError}`（イシュー #2369・親 #2362）の
//! 統合テスト。facade の公開 API と `std` のみで、受入基準（bit 一致の往復・世代コミット・
//! 未対応モデルの fail-closed・非信頼入力の拒否）を検証する。
//!
//! 改竄行列は `compat_sequential_model_io_tamper.rs`（#2375）、衝突注入・TOCTOU は
//! `compat_sequential_model_io_fs_threats.rs`（#2376）が担う（本ファイルは基本ケースと、
//! #2374 の正常系 bit 一致行列）。
//!
//! `mod roundtrip_matrix`（#2374）は `docs/compat-model-io-decision.md` §6 の正常系を
//! 「アーキテクチャ（42 種の和集合・深い異種スタック・transformer）× 状態（eval・train 後の BN・
//! 6 optimizer × AMP・GradScaler 非初期・Lbfgs 履歴 未満／到達済み）」の直積で固定し、全セルで
//! 再保存したファイルのバイト一致を検査する。単一軸の深掘りは `compat_sequential_model_io_{layers,
//! batch_norm,compiled,lbfgs}.rs` が担う（対応表は同 doc §6.1）。
//! `compat_sequential_model_io_manual.rs`（既存公開 API のみの重みのみ往復）は `save_model` とは
//! 独立した契約のため統合せず残す。CUDA／Metal 実機 parity は対象外（ホスト側 I/O と
//! CPU 上の層再構築のみでカーネルを持たない）。
//! ファイル名は `compat_sequential_model_io_manual.rs` と区別するため接尾辞なし。

#[cfg(unix)]
use std::path::Path;

#[cfg(unix)]
use fandhe_ai::Tensor;
use fandhe_ai::compat::{ModelIoError, Sequential, load_model, save_model};

mod common;
use common::temp_dir::TempDirGuard;

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
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

#[cfg(unix)]
fn entries(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("読めるはず")
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

#[cfg(unix)]
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
#[cfg(unix)]
struct CustomIdentity;
#[cfg(unix)]
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

/// 非 unix では `load_model` も `Unsupported` で fail-closed する（`save_model` 側と対）。
/// `fs_guard::open_leaf_checked` は `symlink_metadata` を先に呼ぶため、manifest が無いと
/// `NotFound` になる。通常ファイルの `manifest.json` を先に置き、no-follow オープンの
/// 拒否まで到達させる。Linux CI では型検査のみで、実行は Windows 実機検証（#2393）。
#[cfg(not(unix))]
#[test]
fn load_model_is_unsupported_on_non_unix() {
    let guard = TempDirGuard::new("non-unix-load");
    let dir = guard.path().join("m");
    std::fs::create_dir_all(&dir).expect("作れるはず");
    std::fs::write(dir.join("manifest.json"), b"{}").expect("書けるはず");
    let err = load_model(&dir)
        .map(|_| ())
        .expect_err("非 unix は未対応のはず");
    assert!(
        matches!(&err, ModelIoError::Io(e) if e.kind() == std::io::ErrorKind::Unsupported),
        "{err}"
    );
}
// ---------------------------------------------------------------------
// 正常系 bit 一致行列（イシュー #2374）
// ---------------------------------------------------------------------

/// `docs/compat-model-io-decision.md` §6 の正常系を「アーキテクチャ軸 × 状態軸」の直積で固定する。
///
/// `compat_sequential_model_io_{layers,batch_norm,compiled,lbfgs}.rs` は各軸を単独で深掘りする。
/// 本モジュールはその軸をまたぐ組み合わせ（BN を含む深いスタックで optimizer＋AMP、TE を含む
/// 構成で Lbfgs 履歴、Sgd 以外での GradScaler 非初期状態 等）と、全セル共通の
/// **再保存不変条件**（load したモデルを別 dir へ再保存すると safetensors がバイト一致し、
/// manifest が `safetensors_file` の値を除いて一致する）を固定する。`growth_tracker` や
/// optimizer 内部状態のように公開 API から直接観測できない状態も、この不変条件で bit 一致を担保する。
///
/// 決定性: プロセス共有の RNG を避けるため、train モードで forward する構成には Dropout を入れない
/// （Dropout は eval セルの A-30 でのみ扱う）。学習は `shuffle=false`。
/// 非 unix では `save_model` が fail-closed のためモジュールごとゲートする（Windows cross clippy 対策）。
#[cfg(unix)]
mod roundtrip_matrix {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};

    use fandhe_ai::compat::{
        AmpConfig, AmpDType, FitConfig, Loss, Optimizer, Sequential, load_model, save_model,
    };
    use fandhe_ai::optim::{
        AdagradConfig, AdamConfig, AdamWConfig, GradScalerConfig, LambConfig, LbfgsConfig,
        RmsPropConfig, SgdConfig,
    };
    use fandhe_ai::{AutodiffError, GlobalPoolMode, InterpolateMode, Tensor};

    use crate::common::temp_dir::TempDirGuard;

    /// allowlist の 42 kind（`compat::Sequential` の `add_*` と 1 対 1）。
    const ALL_KINDS: [&str; 42] = [
        "linear",
        "relu",
        "sigmoid",
        "tanh",
        "silu",
        "hardswish",
        "gelu",
        "gelu_tanh",
        "leaky_relu",
        "elu",
        "softmax",
        "log_softmax",
        "softplus",
        "flatten",
        "dropout",
        "conv2d",
        "conv_transpose2d",
        "conv3d",
        "conv1d",
        "layer_norm",
        "rms_norm",
        "batch_norm1d",
        "batch_norm2d",
        "embedding",
        "multihead_attention",
        "transformer_encoder",
        "max_pool2d",
        "max_pool1d",
        "avg_pool2d",
        "avg_pool1d",
        "adaptive_avg_pool2d",
        "adaptive_avg_pool1d",
        "adaptive_max_pool2d",
        "adaptive_max_pool1d",
        "global_pool",
        "upsample",
        "zero_pad2d",
        "identity",
        "group_norm",
        "instance_norm",
        "pixel_shuffle",
        "pixel_unshuffle",
    ];

    type Built = Result<Sequential, AutodiffError>;

    /// 学習セルのバッチ数（N=4・バッチ 2 で 1 epoch 2 step）。
    const N: usize = 4;
    const OUT: usize = 3;

    fn tensor(shape: &[usize], base: f32) -> Tensor<f32> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.37 + base).sin()).collect();
        Tensor::new(data, shape).expect("テンソルを作れるはず")
    }

    /// embedding 用の整数 id（f32 で渡す契約）。
    fn ids(shape: &[usize], modulo: usize) -> Tensor<f32> {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i % modulo) as f32).collect();
        Tensor::new(data, shape).expect("テンソルを作れるはず")
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
        assert_eq!(sa.len(), sb.len(), "{what}: state_dict のキー数");
        for (k, v) in &sa {
            let other = sb
                .get(k)
                .unwrap_or_else(|| panic!("{what}: {k} が復元側にない"));
            assert_eq!(v.shape(), other.shape(), "{what}: {k} の shape");
            assert_eq!(bits(v), bits(other), "{what}: {k} が bit 不一致");
        }
    }

    // -- アーキテクチャ軸 ------------------------------------------------

    /// A-30（1/4）rank 2 入力 `[3, 6]`: 活性化・正規化・dropout・BatchNorm1d・softmax 系。
    fn mlp_model() -> Built {
        let mut m = Sequential::new()
            .add_linear(6, 8, 11)?
            .add_relu()
            .add_sigmoid()
            .add_tanh()
            .add_silu()
            .add_hardswish()
            .add_leaky_relu(0.1)
            .add_elu(1.25)
            .add_gelu()
            .add_gelu_tanh()
            .add_softplus(1.5, 20.0)?
            .add_dropout(0.3)?
            .add_layer_norm(8, 1e-5)?
            .add_rms_norm(8, 1e-6)?
            .add_batch_norm1d(8, 1e-5, 0.1)?
            .add_linear(8, 5, 12)?
            .add_softmax(1)
            .add_log_softmax(1);
        m.eval();
        Ok(m)
    }

    /// A-30（2/4）rank 4 入力 `[2, 2, 8, 8]`: conv2d・BatchNorm2d・2D pooling・flatten。
    fn cnn_model() -> Built {
        let mut m = Sequential::new()
            .add_conv2d(2, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 21)?
            .add_batch_norm2d(4, 1e-5, 0.1)?
            .add_relu()
            .add_group_norm(2, 1e-5)?
            .add_instance_norm(1e-5)?
            .add_conv_transpose2d(4, 4, [3, 3], [1, 1], [1, 1], [0, 0], [1, 1], 2, 24)?
            .add_zero_pad2d([1, 1, 1, 1])
            .add_upsample(vec![8, 8], InterpolateMode::Nearest)?
            .add_identity()
            .add_pixel_unshuffle(2)?
            .add_pixel_shuffle(2)?
            .add_max_pool2d([2, 2], None, [0, 0], [1, 1])?
            .add_conv2d(4, 4, [1, 1], [1, 1], [0, 0], [1, 1], 2, 22)?
            .add_avg_pool2d([2, 2], Some([1, 1]), [1, 1], true)?
            .add_adaptive_avg_pool2d([2, 2])?
            .add_adaptive_max_pool2d([2, 2])?
            .add_max_pool2d([2, 2], Some([1, 1]), [0, 0], [1, 1])?
            .add_avg_pool2d([1, 1], None, [0, 0], false)?
            .add_global_pool(GlobalPoolMode::Avg, true)
            .add_flatten(1, 3)
            .add_linear(4, 3, 23)?;
        m.eval();
        Ok(m)
    }

    /// rank 5 入力 `[2, 2, 3, 3, 3]`: conv3d（groups・stride・dilation 非自明）・flatten（イシュー #2524）。
    fn cnn3d_model() -> Built {
        let mut m = Sequential::new()
            .add_conv3d(2, 4, [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 2, 25)?
            .add_relu()
            .add_flatten(1, 4)
            .add_linear(32, 3, 26)?;
        m.eval();
        Ok(m)
    }

    /// A-30（3/4）rank 3 入力 `[2, 3, 10]`: conv1d・BatchNorm1d（rank 3）・1D pooling・flatten。
    fn cnn1d_model() -> Built {
        let mut m = Sequential::new()
            .add_conv1d(3, 4, 3, 1, 1, 1, 1, 31)?
            .add_batch_norm1d(4, 1e-5, 0.1)?
            .add_relu()
            .add_max_pool1d(2, None, 0, 1)?
            .add_avg_pool1d(2, Some(1), 1, true)?
            .add_max_pool1d(2, Some(2), 0, 1)?
            .add_avg_pool1d(1, None, 0, false)?
            .add_adaptive_avg_pool1d(2)?
            .add_adaptive_max_pool1d(2)?
            .add_global_pool(GlobalPoolMode::Max, false)
            .add_linear(4, 2, 32)?;
        m.eval();
        Ok(m)
    }

    /// A-30（4/4）整数 id 入力 `[2, 5]`: embedding・MHA・TE・layer_norm。
    fn sequence_model(padding_idx: Option<usize>) -> Built {
        let mut m = Sequential::new()
            .add_embedding(10, 8, padding_idx, 41)?
            .add_multihead_attention(8, 2, 42)?
            .add_transformer_encoder(8, 2, 16, 43)?
            .add_layer_norm(8, 1e-5)?
            .add_flatten(1, 2)
            .add_linear(40, 3, 44)?;
        m.eval();
        Ok(m)
    }

    /// A-deep: 深い異種スタック（conv・BN・pool・group conv・norm・活性化・BN1d）。入力 `[N, 2, 8, 8]`。
    /// AMP 対象層（Conv2d・Linear）と BN を含み、Dropout は含まない（決定性）。
    fn deep_model() -> Built {
        Sequential::new()
            .add_conv2d(2, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 61)?
            .add_batch_norm2d(4, 1e-5, 0.1)?
            .add_relu()
            .add_max_pool2d([2, 2], None, [0, 0], [1, 1])?
            .add_conv2d(4, 4, [1, 1], [1, 1], [0, 0], [1, 1], 2, 62)?
            .add_avg_pool2d([2, 2], Some([1, 1]), [1, 1], true)?
            .add_adaptive_avg_pool2d([2, 2])?
            .add_flatten(1, 3)
            .add_linear(16, 8, 63)?
            .add_batch_norm1d(8, 1e-5, 0.1)?
            .add_layer_norm(8, 1e-5)?
            .add_rms_norm(8, 1e-6)?
            .add_gelu()
            .add_silu()
            .add_linear(8, OUT, 64)
    }

    fn deep_input() -> Tensor<f32> {
        tensor(&[N, 2, 8, 8], 0.5)
    }

    /// A-tf: transformer 構成（TE×2・MHA・layer_norm）。入力は f32 の `[N, 5, 8]`。
    /// embedding は整数 id 入力の A-30 eval セルで扱う（`fit` 経路の対象外）。
    fn transformer_model() -> Built {
        Sequential::new()
            .add_transformer_encoder(8, 2, 16, 71)?
            .add_transformer_encoder(8, 2, 16, 72)?
            .add_multihead_attention(8, 2, 73)?
            .add_layer_norm(8, 1e-5)?
            .add_flatten(1, 2)
            .add_linear(40, OUT, 74)
    }

    fn transformer_input() -> Tensor<f32> {
        tensor(&[N, 5, 8], 1.5)
    }

    fn target() -> Tensor<f32> {
        tensor(&[N, OUT], 2.5)
    }

    // -- 状態軸 ------------------------------------------------------------

    /// 6 optimizer（`compat_sequential_model_io_compiled.rs` と同じ構成）。
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

    /// backoff と growth の両方が起きる GradScaler 設定（`compat_sequential_model_io_compiled.rs` で実証済み）。
    const INIT_SCALE: f32 = 3.0e38;

    fn scaler_config() -> GradScalerConfig {
        GradScalerConfig {
            init_scale: INIT_SCALE,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 3,
        }
    }

    // -- 共通ヘルパー ----------------------------------------------------------

    fn read_manifest(dir: &Path) -> String {
        std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず")
    }

    /// `"safetensors_file":"..."` の値だけを固定文字列へ置換する（世代名は保存ごとに異なるため）。
    /// 以降の `compiled` 節まで比較対象に残すため、切り捨てではなく値の置換にする。
    fn manifest_without_gen(text: &str) -> String {
        let key = "\"safetensors_file\":\"";
        let start = text.find(key).expect("safetensors_file があるはず") + key.len();
        let end = start + text[start..].find('"').expect("値は閉じているはず");
        format!("{}<gen>{}", &text[..start], &text[end..])
    }

    fn only_safetensors(dir: &Path) -> PathBuf {
        let names: Vec<String> = std::fs::read_dir(dir)
            .expect("読めるはず")
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("model.") && n.ends_with(".safetensors"))
            .collect();
        assert_eq!(names.len(), 1, "世代ファイルは 1 つのはず: {names:?}");
        dir.join(&names[0])
    }

    /// `manifest.json` から `"kind":"..."` を拾う（プログラム生成値のみの簡易走査）。
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

    /// 2 つのモデルを別 dir へ保存し、safetensors のバイト列と manifest（世代名を除く）が
    /// 完全一致することを確認する。
    fn assert_saved_identically(label: &str, a: &Sequential, b: &Sequential) {
        let guard = TempDirGuard::new("resave");
        let (da, db) = (guard.path().join("a"), guard.path().join("b"));
        save_model(a, &da).unwrap_or_else(|e| panic!("{label}: 保存できるはず: {e}"));
        save_model(b, &db).unwrap_or_else(|e| panic!("{label}: 保存できるはず: {e}"));
        let (ba, bb) = (
            std::fs::read(only_safetensors(&da)).expect("読めるはず"),
            std::fs::read(only_safetensors(&db)).expect("読めるはず"),
        );
        assert!(ba == bb, "{label}: safetensors のバイト列が不一致");
        assert_eq!(
            manifest_without_gen(&read_manifest(&da)),
            manifest_without_gen(&read_manifest(&db)),
            "{label}: manifest が不一致"
        );
    }

    /// 続けて学習する compile 済みセルの継続条件。
    struct Continuation {
        x: Tensor<f32>,
        y: Tensor<f32>,
        batch: usize,
    }

    /// 全セル共通の検査。保存 → load → (1) state_dict の bit 一致 (2) training／compile／AMP scale
    /// (3) 再保存不変条件（バイト一致） (4) 継続 fit の一致と継続後の再保存一致 (5) eval の predict 一致。
    /// 保存した manifest の `kind` 集合を返す。`model` は継続 fit・eval 化で変更される。
    fn check_cell(
        label: &str,
        model: &mut Sequential,
        input: &Tensor<f32>,
        cont: Option<&Continuation>,
    ) -> BTreeSet<String> {
        let guard = TempDirGuard::new("cell");
        let dir = guard.path().join("m");
        save_model(model, &dir).unwrap_or_else(|e| panic!("{label}: 保存できるはず: {e}"));
        let kinds = kinds_of(&read_manifest(&dir));
        let mut loaded =
            load_model(&dir).unwrap_or_else(|e| panic!("{label}: 復元できるはず: {e}"));

        assert_same_params(model, &loaded, label);
        assert_eq!(model.training(), loaded.training(), "{label}: training");
        assert_eq!(
            model.is_compiled(),
            loaded.is_compiled(),
            "{label}: compiled"
        );
        assert_eq!(
            model.amp_loss_scale().map(f32::to_bits),
            loaded.amp_loss_scale().map(f32::to_bits),
            "{label}: amp scale"
        );

        // 再保存不変条件: 永続化された状態（BN buffer・optimizer 状態・tracker）の全体が一致する。
        assert_saved_identically(&format!("{label}: 再保存"), model, &loaded);

        if let Some(c) = cont {
            assert!(model.is_compiled(), "{label}: 継続 fit は compile 済みのみ");
            let cfg = || FitConfig::new(2, c.batch);
            let h1 = model.fit(&c.x, &c.y, cfg()).expect("fit 続き");
            let h2 = loaded.fit(&c.x, &c.y, cfg()).expect("fit 続き（復元後）");
            assert_eq!(f32_bits(&h1.loss), f32_bits(&h2.loss), "{label}: loss");
            assert_eq!(f32_bits(&h1.lr), f32_bits(&h2.lr), "{label}: lr");
            assert_same_params(model, &loaded, &format!("{label}: 継続後"));
            assert_eq!(
                model.amp_loss_scale().map(f32::to_bits),
                loaded.amp_loss_scale().map(f32::to_bits),
                "{label}: 継続後の amp scale"
            );
            assert_saved_identically(&format!("{label}: 継続後の再保存"), model, &loaded);
        }

        // eval() は状態を変えるため最後に行う。
        model.eval();
        loaded.eval();
        let ya = model.predict(input).expect("predict できるはず");
        let yb = loaded.predict(input).expect("predict できるはず");
        assert_eq!(ya.shape(), yb.shape(), "{label}: predict shape");
        assert_eq!(bits(&ya), bits(&yb), "{label}: predict 出力");
        kinds
    }

    /// 保存して manifest を返す（Lbfgs の履歴長など、公開 API から見えない状態の確認用）。
    fn manifest_of(model: &Sequential) -> String {
        let guard = TempDirGuard::new("peek");
        let dir = guard.path().join("m");
        save_model(model, &dir).expect("保存できるはず");
        read_manifest(&dir)
    }

    fn field_usize(text: &str, key: &str) -> usize {
        let needle = format!("\"{key}\":");
        let start = text.find(&needle).unwrap_or_else(|| panic!("{key} が無い")) + needle.len();
        let end = start + text[start..].find([',', '}']).expect("終端");
        text[start..end].parse().expect("整数のはず")
    }

    /// running_mean が初期値 0 から動いている（BN buffer を保存しないと検出できる状態）。
    /// buffer は `state_dict` に含まれないため、保存した safetensors から読む。
    fn assert_bn_stats_moved(label: &str, model: &Sequential) {
        let guard = TempDirGuard::new("bn-peek");
        let dir = guard.path().join("m");
        save_model(model, &dir).expect("保存できるはず");
        let stored = fandhe_ai::interop::safetensors::load_safetensors_f32_from_bytes(
            &std::fs::read(only_safetensors(&dir)).expect("読めるはず"),
        )
        .expect("safetensors を読めるはず");
        let moved = stored
            .iter()
            .filter(|(k, _)| k.ends_with(".running_mean"))
            .any(|(_, t)| bits(t).iter().any(|b| f32::from_bits(*b) != 0.0));
        assert!(moved, "{label}: running_mean が更新されていない");
    }

    // -- テスト ---------------------------------------------------------------

    /// A-30 × eval: 42 種の kind の和集合（未 compile）。
    #[test]
    fn matrix_thirty_kinds_eval_round_trip_and_resave_identical() {
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let cases: Vec<(&str, Sequential, Tensor<f32>)> = vec![
            ("mlp", mlp_model().expect("構築"), tensor(&[3, 6], 0.5)),
            (
                "cnn",
                cnn_model().expect("構築"),
                tensor(&[2, 2, 8, 8], 1.5),
            ),
            (
                "cnn1d",
                cnn1d_model().expect("構築"),
                tensor(&[2, 3, 10], 2.5),
            ),
            (
                "cnn3d",
                cnn3d_model().expect("構築"),
                tensor(&[2, 2, 3, 3, 3], 3.5),
            ),
            (
                "seq-pad",
                sequence_model(Some(0)).expect("構築"),
                ids(&[2, 5], 10),
            ),
            (
                "seq-nopad",
                sequence_model(None).expect("構築"),
                ids(&[2, 5], 10),
            ),
        ];
        for (label, mut model, input) in cases {
            seen.extend(check_cell(label, &mut model, &input, None));
        }
        let expected: BTreeSet<String> = ALL_KINDS.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(seen, expected, "42 種の kind をすべて往復させたはず");
    }

    /// A-deep × {eval, train 後の BN}: train モードで predict を回し running stats を動かした状態。
    #[test]
    fn matrix_deep_stack_eval_and_trained_bn_round_trip() {
        let x = deep_input();
        let mut m = deep_model().expect("構築");
        m.eval();
        check_cell("deep-eval", &mut m, &x, None);

        // train モードの predict は BN の running stats を更新する。
        let mut m = deep_model().expect("構築");
        for _ in 0..3 {
            m.predict(&x).expect("train predict");
        }
        assert!(m.training());
        assert_bn_stats_moved("deep-bn", &m);
        let kinds = check_cell("deep-bn", &mut m, &x, None);
        assert!(kinds.contains("batch_norm2d") && kinds.contains("batch_norm1d"));
    }

    /// A-tf × eval。
    #[test]
    fn matrix_transformer_stack_eval_round_trip() {
        let mut m = transformer_model().expect("構築");
        m.eval();
        let kinds = check_cell("tf-eval", &mut m, &transformer_input(), None);
        assert!(kinds.contains("transformer_encoder") && kinds.contains("multihead_attention"));
    }

    /// 6 optimizer × AMP{なし, あり}。AMP ありは GradScaler を非初期状態にしてから保存し、
    /// dtype は F16／Bf16 を交互に割り当てる。
    fn optimizer_amp_cells(prefix: &str, build: fn() -> Built, x: &Tensor<f32>, check_bn: bool) {
        let y = target();
        for (i, (name, opt)) in optimizers().into_iter().enumerate() {
            for amp in [false, true] {
                let label = format!("{prefix}/{name}/amp={amp}");
                let mut m = build().expect("構築");
                if amp {
                    let dtype = if i % 2 == 0 {
                        AmpDType::F16
                    } else {
                        AmpDType::Bf16
                    };
                    m.compile_with_amp(
                        opt,
                        Loss::Mse,
                        AmpConfig::new(dtype).grad_scaler(scaler_config()),
                    )
                    .unwrap_or_else(|e| panic!("{label}: compile_with_amp: {e}"));
                } else {
                    m.compile(opt, Loss::Mse)
                        .unwrap_or_else(|e| panic!("{label}: compile: {e}"));
                }
                m.fit(x, &y, FitConfig::new(3, 2))
                    .unwrap_or_else(|e| panic!("{label}: fit: {e}"));
                if amp {
                    let scale = m.amp_loss_scale().expect("amp");
                    assert_ne!(
                        scale.to_bits(),
                        INIT_SCALE.to_bits(),
                        "{label}: GradScaler が初期状態のまま"
                    );
                }
                if check_bn {
                    assert_bn_stats_moved(&label, &m);
                }
                let cont = Continuation {
                    x: x.clone(),
                    y: y.clone(),
                    batch: 2,
                };
                check_cell(&label, &mut m, x, Some(&cont));
            }
        }
    }

    #[test]
    fn matrix_deep_stack_six_optimizers_with_and_without_amp() {
        optimizer_amp_cells("deep", deep_model, &deep_input(), true);
    }

    #[test]
    fn matrix_transformer_stack_six_optimizers_with_and_without_amp() {
        optimizer_amp_cells("tf", transformer_model, &transformer_input(), false);
    }

    /// Lbfgs（AMP なし・フルバッチ）: 履歴が `history_size` 未満と到達済み（追い出しを含む）。
    fn lbfgs_cells(prefix: &str, build: fn() -> Built, x: &Tensor<f32>) {
        let y = target();
        for (name, history_size, epochs, full) in [("below", 50, 1, false), ("full", 3, 3, true)] {
            let label = format!("{prefix}/lbfgs-{name}");
            let mut m = build().expect("構築");
            m.compile(
                Optimizer::Lbfgs(LbfgsConfig {
                    lr: 0.05,
                    max_iter: 4,
                    history_size,
                    ..LbfgsConfig::default()
                }),
                Loss::Mse,
            )
            .unwrap_or_else(|e| panic!("{label}: compile: {e}"));
            m.fit(x, &y, FitConfig::new(epochs, N))
                .unwrap_or_else(|e| panic!("{label}: fit: {e}"));
            // 履歴の状態は公開 API から見えないため manifest の値で確認する。
            let text = manifest_of(&m);
            let (len, size) = (
                field_usize(&text, "history_len"),
                field_usize(&text, "history_size"),
            );
            if full {
                assert_eq!(len, size, "{label}: 履歴は到達済みのはず");
            } else {
                assert!(
                    0 < len && len < size,
                    "{label}: 履歴は未満のはず: {len}/{size}"
                );
            }
            let cont = Continuation {
                x: x.clone(),
                y: y.clone(),
                batch: N,
            };
            check_cell(&label, &mut m, x, Some(&cont));
        }
    }

    #[test]
    fn matrix_deep_stack_lbfgs_history_below_and_at_capacity() {
        lbfgs_cells("deep", deep_model, &deep_input());
    }

    #[test]
    fn matrix_transformer_stack_lbfgs_history_below_and_at_capacity() {
        lbfgs_cells("tf", transformer_model, &transformer_input());
    }
}
