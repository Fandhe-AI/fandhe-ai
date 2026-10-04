//! `fandhe_ai::compat::{save_model, load_model}` の全 50 層対応（イシュー #2370・親 #2362）の
//! 統合テスト。facade の公開 API と `std` のみで次の受入基準を検証する。
//!
//! - (a) 51 種の `add_*` をそれぞれ 1 層以上含むモデルの save → load で、パラメータと
//!   `predict` 出力が bit 一致する（51 種の網羅は保存した manifest の `kind` 集合で検査する）。
//! - (b) 深い異種スタック（`add_transformer_encoder` を複数含む数十層）で bit 一致する。
//! - (c) `kind`・`params` の改竄（未知 kind・範囲外の値・キーの過不足・型違い）を拒否する。
//! - fail-closed: 層のモードがモデル全体と異なる・利用者定義層・構造上限超過のモデルは、`dir` に何も作らず型付きエラーで拒否する。
//!
//! 衝突注入・TOCTOU・網羅的な改竄行列は #2375・#2376。BN の running stats の保存・復元は #2371（`compat_sequential_model_io_batch_norm.rs`）。
//! 実機（CUDA／Metal）は不要（ホスト側の I/O と層の再構築のみ）。

#![cfg(unix)]

use std::collections::BTreeSet;
use std::path::Path;

use fandhe_ai::compat::{ModelIoError, Sequential, load_model, save_model};
use fandhe_ai::{AutodiffError, EmbeddingBagMode, GlobalPoolMode, InterpolateMode, Tensor};

mod common;
use common::temp_dir::TempDirGuard;

/// allowlist の 45 kind（`compat::Sequential` の `add_*` と 1 対 1）。
const ALL_KINDS: [&str; 51] = [
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
    "mish",
    "hardtanh",
    "relu6",
    "glu",
    "prelu",
    "flatten",
    "dropout",
    "dropout2d",
    "alpha_dropout",
    "conv2d",
    "conv_transpose2d",
    "conv3d",
    "conv1d",
    "layer_norm",
    "rms_norm",
    "batch_norm1d",
    "batch_norm2d",
    "embedding",
    "embedding_bag",
    "multihead_attention",
    "multihead_attention_config",
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

/// save → load し、`state_dict` の全キー・`training`・`predict` 出力の bit 一致を確認して
/// 保存した manifest の `kind` 集合を返す。
fn round_trip(label: &str, model: &Sequential, input: &Tensor<f32>) -> BTreeSet<String> {
    let guard = TempDirGuard::new(label);
    let dir = guard.path().join("m");
    save_model(model, &dir).unwrap_or_else(|e| panic!("{label}: 保存できるはず: {e}"));
    let loaded = load_model(&dir).unwrap_or_else(|e| panic!("{label}: 復元できるはず: {e}"));

    let (sa, sb) = (model.state_dict(), loaded.state_dict());
    assert_eq!(sa.len(), sb.len(), "{label}: キー数");
    for (k, v) in &sa {
        assert_bit_identical(v, &sb[k], &format!("{label}: {k}"));
    }
    assert_eq!(model.training(), loaded.training(), "{label}: training");
    let ya = model.predict(input).expect("predict できるはず");
    let yb = loaded.predict(input).expect("predict できるはず");
    assert_bit_identical(&ya, &yb, &format!("{label}: predict 出力"));

    let manifest = std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず");
    kinds_of(&manifest)
}

/// manifest 本文から `"kind":"..."` を拾う（JSON パーサを持たないテストの簡易走査。
/// kind は英小文字・数字・`_` のみのプログラム生成値）。
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

// ---------------------------------------------------------------------
// (a) 51 種の bit 一致
// ---------------------------------------------------------------------

/// rank 2 入力 `[3, 6]`: 活性化・正規化・dropout・BatchNorm1d（rank 2）・softmax 系。
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
        .add_mish()
        .add_hardtanh(-2.0, 2.0)?
        .add_relu6()
        .add_prelu(8, 0.2)?
        .add_dropout(0.3)?
        .add_alpha_dropout(0.2)?
        .add_layer_norm(8, 1e-5)?
        .add_rms_norm(8, 1e-6)?
        .add_batch_norm1d(8, 1e-5, 0.1)?
        .add_glu(1)
        .add_linear(4, 5, 12)?
        .add_softmax(1)
        .add_log_softmax(1);
    m.eval();
    Ok(m)
}

/// rank 4 入力 `[2, 2, 8, 8]`: conv2d・BatchNorm2d・2D pooling（stride の Some／None・
/// `count_include_pad` の真偽）・flatten。
fn cnn_model() -> Built {
    let mut m = Sequential::new()
        .add_conv2d(2, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, 21)?
        .add_batch_norm2d(4, 1e-5, 0.1)?
        .add_relu()
        .add_dropout2d(0.2)?
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

/// rank 3 入力 `[2, 3, 10]`: conv1d・BatchNorm1d（rank 3）・1D pooling（stride の Some／None・
/// `count_include_pad` の真偽）・flatten。
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

/// 整数 id 入力 `[2, 5]`: embedding（padding_idx あり）・MHA・TE・layer_norm。
fn sequence_model(padding_idx: Option<usize>) -> Built {
    let mut m = Sequential::new()
        .add_embedding(10, 8, padding_idx, 41)?
        .add_multihead_attention(8, 2, 42)?
        .add_multihead_attention_with_config(
            fandhe_ai::compat::MultiheadAttentionConfig::new(8, 2)
                .with_bias(false)
                .with_batch_first(false),
            45,
        )?
        .add_transformer_encoder(8, 2, 16, 43)?
        .add_layer_norm(8, 1e-5)?
        .add_flatten(1, 2)
        .add_linear(40, 3, 44)?;
    m.eval();
    Ok(m)
}

/// 整数 id 入力 `[2, 5]`: EmbeddingBag（mode・padding_idx の組み合わせ。イシュー #2528）。
fn bag_model(mode: EmbeddingBagMode, padding_idx: Option<usize>) -> Built {
    let mut m = Sequential::new()
        .add_embedding_bag(10, 8, mode, padding_idx, 45)?
        .add_linear(8, 3, 46)?;
    m.eval();
    Ok(m)
}

#[test]
fn all_thirty_layer_kinds_round_trip_bit_identically() {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    seen.extend(round_trip(
        "mlp",
        &mlp_model().expect("構築できるはず"),
        &tensor(&[3, 6], 0.5),
    ));
    seen.extend(round_trip(
        "cnn",
        &cnn_model().expect("構築できるはず"),
        &tensor(&[2, 2, 8, 8], 1.5),
    ));
    seen.extend(round_trip(
        "cnn1d",
        &cnn1d_model().expect("構築できるはず"),
        &tensor(&[2, 3, 10], 2.5),
    ));
    seen.extend(round_trip(
        "cnn3d",
        &cnn3d_model().expect("構築できるはず"),
        &tensor(&[2, 2, 3, 3, 3], 3.5),
    ));
    for padding_idx in [Some(0), None] {
        seen.extend(round_trip(
            "sequence",
            &sequence_model(padding_idx).expect("構築できるはず"),
            &ids(&[2, 5], 10),
        ));
    }
    for (mode, padding_idx) in [
        (EmbeddingBagMode::Sum, Some(0)),
        (EmbeddingBagMode::Mean, None),
        (EmbeddingBagMode::Max, None),
    ] {
        seen.extend(round_trip(
            "bag",
            &bag_model(mode, padding_idx).expect("構築できるはず"),
            &ids(&[2, 5], 10),
        ));
    }
    let expected: BTreeSet<String> = ALL_KINDS.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(seen, expected, "51 種の kind をすべて往復させたはず");
}

/// train モードのまま保存・復元しても `training` フラグが往復し、dropout を含む構成で
/// 全層のモードが一致していれば保存できる。
#[test]
fn train_mode_model_round_trips_with_flag() {
    let model = Sequential::new()
        .add_linear(4, 4, 1)
        .and_then(|m| m.add_dropout(0.5))
        .and_then(|m| m.add_batch_norm1d(4, 1e-5, 0.1))
        .expect("構築できるはず");
    assert!(model.training());
    let guard = TempDirGuard::new("train-mode");
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let loaded = load_model(&dir).expect("復元できるはず");
    assert!(loaded.training());
    for (k, v) in &model.state_dict() {
        assert_bit_identical(v, &loaded.state_dict()[k], k);
    }
}

/// f32 引数は正準形で往復する（`-0.0`・非正規化数・`f32::MAX`・指数表記）。
/// 再保存した manifest の params が最初と一致すること（f32 の bit 一致）で確認する。
#[test]
fn f32_params_round_trip_bit_exactly() {
    let mut model = Sequential::new()
        .add_leaky_relu(-0.0)
        .add_elu(1e-45)
        .add_leaky_relu(f32::MAX)
        .add_leaky_relu(1.0e-5)
        .add_leaky_relu(0.1);
    model.eval();
    let guard = TempDirGuard::new("f32-exact");
    let (d1, d2) = (guard.path().join("a"), guard.path().join("b"));
    save_model(&model, &d1).expect("保存できるはず");
    let loaded = load_model(&d1).expect("復元できるはず");
    save_model(&loaded, &d2).expect("再保存できるはず");
    let strip = |dir: &Path| {
        let text = std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず");
        let start = text.find("\"safetensors_file\"").expect("キーがあるはず");
        text[..start].to_string()
    };
    assert_eq!(strip(&d1), strip(&d2));
    let input = tensor(&[2, 3], 0.25);
    assert_bit_identical(
        &model.predict(&input).expect("predict"),
        &loaded.predict(&input).expect("predict"),
        "predict 出力",
    );
}

// ---------------------------------------------------------------------
// (b) 深い異種スタック
// ---------------------------------------------------------------------

#[test]
fn deep_heterogeneous_stack_with_transformer_encoders_round_trips() {
    let mut m = Sequential::new()
        .add_embedding(12, 8, Some(0), 51)
        .expect("構築できるはず");
    for block in 0..6u64 {
        m = m
            .add_transformer_encoder(8, 2, 16, 100 + block)
            .and_then(|m| m.add_layer_norm(8, 1e-5))
            .map(|m| m.add_gelu().add_silu())
            .and_then(|m| m.add_dropout(0.1))
            .map(|m| m.add_hardswish().add_tanh())
            .and_then(|m| m.add_rms_norm(8, 1e-6))
            .and_then(|m| m.add_multihead_attention(8, 4, 300 + block))
            .expect("構築できるはず");
    }
    let mut m = m
        .add_layer_norm(8, 1e-5)
        .map(|m| m.add_flatten(1, 2))
        .and_then(|m| m.add_linear(40, 4, 400))
        .expect("構築できるはず");
    m.eval();
    let kinds = round_trip("deep", &m, &ids(&[2, 5], 12));
    for k in ["transformer_encoder", "multihead_attention", "dropout"] {
        assert!(kinds.contains(k), "{k} を含むはず");
    }
}

// ---------------------------------------------------------------------
// (c) 改竄の拒否
// ---------------------------------------------------------------------

/// dropout・softplus・conv1d・max_pool1d・relu を含む小さなモデルを保存し、
/// `manifest.json` の本文を返す。
fn saved_for_tamper(label: &str) -> (TempDirGuard, std::path::PathBuf, String) {
    let mut model = Sequential::new()
        .add_linear(4, 4, 1)
        .and_then(|m| m.add_dropout(0.25))
        .and_then(|m| m.add_softplus(2.0, 20.0))
        .and_then(|m| m.add_conv1d(4, 4, 3, 1, 1, 1, 2, 2))
        .and_then(|m| m.add_max_pool1d(2, None, 0, 1))
        .expect("構築できるはず");
    model.eval();
    let guard = TempDirGuard::new(label);
    let dir = guard.path().join("m");
    save_model(&model, &dir).expect("保存できるはず");
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず");
    (guard, dir, text)
}

fn load_err(dir: &Path) -> ModelIoError {
    match load_model(dir) {
        Ok(_) => panic!("改竄された manifest は拒否されるはず"),
        Err(e) => e,
    }
}

/// `from` を `to` に置換した manifest で `load_model` し、エラーを返す
/// （置換対象が存在することを先に確認する）。
fn tampered_error(label: &str, from: &str, to: &str) -> ModelIoError {
    let (_guard, dir, text) = saved_for_tamper(label);
    assert!(
        text.contains(from),
        "{label}: 置換対象 {from} が manifest にある"
    );
    std::fs::write(dir.join("manifest.json"), text.replacen(from, to, 1)).expect("書けるはず");
    let before = entries(&dir);
    let err = load_err(&dir);
    assert_eq!(entries(&dir), before, "{label}: load は dir を変更しない");
    err
}

#[test]
fn tampered_kind_is_rejected() {
    let err = tampered_error(
        "kind-unknown",
        "\"kind\":\"dropout\"",
        "\"kind\":\"no_such_kind\"",
    );
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err}"
    );
    // 既知 kind へ差し替えても params が固定スキーマと合わなければ Manifest。
    let err = tampered_error("kind-swap", "\"kind\":\"dropout\"", "\"kind\":\"softmax\"");
    assert!(matches!(err, ModelIoError::Manifest { .. }), "{err}");
    // パラメータを持つ層をパラメータなしの層へ差し替えると parameter_keys と整合しない。
    let err = tampered_error(
        "kind-drops-params",
        "\"kind\":\"linear\",\"params\":{\"in_features\":4,\"out_features\":4}",
        "\"kind\":\"relu\",\"params\":{}",
    );
    assert!(
        matches!(
            err,
            ModelIoError::Mismatch { .. } | ModelIoError::Manifest { .. }
        ),
        "{err}"
    );
}

#[test]
fn tampered_param_keys_and_types_are_rejected() {
    for (label, from, to) in [
        ("extra-key", "\"p\":0.25", "\"p\":0.25,\"q\":1"),
        ("missing-key", "{\"p\":0.25}", "{}"),
        ("int-for-f32", "\"p\":0.25", "\"p\":1"),
        ("string-for-f32", "\"p\":0.25", "\"p\":\"0.25\""),
        ("non-canonical-f32", "\"p\":0.25", "\"p\":0.250"),
        ("non-finite-f32", "\"p\":0.25", "\"p\":1e39"),
        ("real-for-int", "\"groups\":2", "\"groups\":2.0"),
        ("negative-int", "\"kernel_size\":3", "\"kernel_size\":-3"),
        ("null-for-int", "\"padding\":1", "\"padding\":null"),
        ("groups-zero", "\"groups\":2", "\"groups\":0"),
        ("groups-indivisible", "\"groups\":2", "\"groups\":3"),
    ] {
        let err = tampered_error(label, from, to);
        assert!(
            matches!(err, ModelIoError::Manifest { .. }),
            "{label}: {err}"
        );
    }
}

#[test]
fn tampered_out_of_range_values_are_rejected_by_layer_constructors() {
    for (label, from, to) in [
        ("dropout-p", "\"p\":0.25", "\"p\":2.0"),
        ("softplus-beta", "\"beta\":2.0", "\"beta\":-1.0"),
        ("pool-kernel", "\"kernel_size\":2", "\"kernel_size\":0"),
    ] {
        let err = tampered_error(label, from, to);
        assert!(matches!(err, ModelIoError::Autodiff(_)), "{label}: {err}");
    }
}

#[test]
fn tampered_parameter_shape_is_rejected() {
    let err = tampered_error("shape", "\"shape\":[4,4]", "\"shape\":[4,5]");
    assert!(matches!(err, ModelIoError::Mismatch { .. }), "{err}");
}

// ---------------------------------------------------------------------
// fail-closed（保存側）
// ---------------------------------------------------------------------

fn assert_save_rejected(label: &str, model: &Sequential, expect_too_large: bool) {
    let guard = TempDirGuard::new(label);
    let dir = guard.path().join("never-created");
    let err = save_model(model, &dir).expect_err("拒否されるはず");
    if expect_too_large {
        assert!(
            matches!(err, ModelIoError::TooLarge { .. }),
            "{label}: {err}"
        );
    } else {
        assert!(
            matches!(err, ModelIoError::UnsupportedModel { .. }),
            "{label}: {err}"
        );
    }
    assert!(!dir.exists(), "{label}: dir が作られてはいけない");
}

#[test]
fn layer_mode_differing_from_model_is_rejected() {
    // eval() の後に積んだ dropout・BatchNorm は層だけ train のまま（`add_*` は push 後に
    // モードを同期しない）。load は全層をモデルのモードへ揃えるため、復元後にずれる。
    let mut model = Sequential::new()
        .add_linear(2, 2, 1)
        .expect("構築できるはず");
    model.eval();
    let with_dropout = model.add_dropout(0.5).expect("構築できるはず");
    assert_save_rejected("mode-dropout", &with_dropout, false);

    let mut model = Sequential::new()
        .add_linear(2, 2, 1)
        .expect("構築できるはず");
    model.eval();
    let with_bn = model
        .add_batch_norm1d(2, 1e-5, 0.1)
        .expect("構築できるはず");
    assert_save_rejected("mode-bn", &with_bn, false);
}

#[test]
fn non_finite_f32_argument_is_rejected() {
    assert_save_rejected(
        "nan-slope",
        &Sequential::new().add_leaky_relu(f32::NAN),
        false,
    );
    assert_save_rejected(
        "inf-alpha",
        &Sequential::new().add_elu(f32::INFINITY),
        false,
    );
}

#[test]
fn model_exceeding_manifest_structure_bound_is_rejected_before_writing() {
    // TE は 16 パラメータ/層のため、層数の上限（4096）より手前で parameter_keys が
    // JSON 配列の上限を超える。書けるのに読めない manifest を作らず、書き込み前に拒否する。
    let mut model = Sequential::new();
    for _ in 0..600 {
        model = model
            .add_transformer_encoder(2, 1, 1, 0)
            .expect("構築できるはず");
    }
    assert_save_rejected("too-many-keys", &model, true);
}
