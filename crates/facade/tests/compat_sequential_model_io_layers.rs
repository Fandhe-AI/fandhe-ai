//! `fandhe_ai::compat::{save_model, load_model}` の全 55 層対応（イシュー #2370・親 #2362）の
//! 統合テスト。facade の公開 API と `std` のみで次の受入基準を検証する。
//!
//! - (a) 55 種の `add_*` をそれぞれ 1 層以上含むモデルの save → load で、パラメータと
//!   `predict` 出力が bit 一致する（55 種の網羅は保存した manifest の `kind` 集合で検査する）。
//! - (b) 深い異種スタック（`add_transformer_encoder` を複数含む数十層）で bit 一致する。
//! - (c) `kind`・`params` の改竄（未知 kind・範囲外の値・キーの過不足・型違い）を拒否する。
//! - fail-closed: 層のモードがモデル全体と異なる・利用者定義層・構造上限超過のモデルは、`dir` に何も作らず型付きエラーで拒否する。
//!
//! 無引数の活性化 5 層（selu・softsign・hardsigmoid・log_sigmoid・tanhshrink。イシュー #3053）の往復・改竄拒否は末尾のテストが受け持つ（上記 53 種の配列は不変）。
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

/// allowlist の 55 kind（`compat::Sequential` の `add_*` と 1 対 1）。
const ALL_KINDS: [&str; 55] = [
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
    "transformer_decoder_layer",
    "transformer",
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
    "conv_transpose1d",
    "unflatten",
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
// (a) 55 種の bit 一致
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

/// イシュー #3055。入力 `[3, 6]`: linear → unflatten（`[3, 2, 4]`）→ conv_transpose1d
/// （groups・stride・padding・output_padding・dilation を非自明な値にする）→ flatten → linear。
/// 出力長は `(4-1)*2 - 2*1 + 2*(3-1) + 1 + 1 = 10`。
fn deconv1d_model() -> Built {
    let mut m = Sequential::new()
        .add_linear(6, 8, 61)?
        .add_unflatten(1, vec![2, 4])?
        .add_conv_transpose1d(2, 4, 3, 2, 1, 1, 2, 2, 62)?
        .add_flatten(1, 2)
        .add_linear(40, 2, 63)?;
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
        .add_transformer_decoder_layer(8, 2, 16, 47)?
        .add_transformer(
            fandhe_ai::nn::TransformerConfig::new(8, 2)
                .with_num_encoder_layers(1)
                .with_num_decoder_layers(1)
                .with_dim_feedforward(16)
                .with_eps(1e-3),
            48,
        )?
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
    seen.extend(round_trip(
        "deconv1d",
        &deconv1d_model().expect("構築できるはず"),
        &tensor(&[3, 6], 4.5),
    ));
    let expected: BTreeSet<String> = ALL_KINDS.iter().map(|s| (*s).to_string()).collect();
    assert_eq!(seen, expected, "55 種の kind をすべて往復させたはず");
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

// ---------------------------------------------------------------------
// 無引数の活性化 5 層（イシュー #3053。kind の `params` は常に `{}`）
// ---------------------------------------------------------------------

const NULLARY_KINDS: [&str; 5] = [
    "selu",
    "softsign",
    "hardsigmoid",
    "log_sigmoid",
    "tanhshrink",
];

fn nullary_model() -> Sequential {
    let mut m = Sequential::new()
        .add_linear(3, 4, 5)
        .expect("構築できるはず")
        .add_selu()
        .add_softsign()
        .add_hardsigmoid()
        .add_log_sigmoid()
        .add_tanhshrink()
        .add_linear(4, 2, 6)
        .expect("構築できるはず");
    m.eval();
    m
}

#[test]
fn nullary_activation_layers_round_trip_bit_identically() {
    // 正負・0 付近・大きい値で各活性化の分岐領域を通す。
    let input = Tensor::new(
        vec![
            -9.0, -1.5, 0.0, 0.25, 1.0, 6.5, -0.0, 3.0, -4.0, 0.5, -0.25, 12.0,
        ],
        &[4, 3],
    )
    .unwrap();
    let kinds = round_trip("nullary-acts", &nullary_model(), &input);
    for k in NULLARY_KINDS {
        assert!(kinds.contains(k), "kind {k} が manifest にある: {kinds:?}");
    }
}

#[test]
fn nullary_activation_tampered_params_are_rejected() {
    for kind in NULLARY_KINDS {
        let from = format!("\"kind\":\"{kind}\",\"params\":{{}}");
        let dir_guard = TempDirGuard::new(&format!("nullary-tamper-{kind}"));
        let dir = dir_guard.path().join("m");
        save_model(&nullary_model(), &dir).expect("保存できるはず");
        let text = std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず");
        assert!(text.contains(&from), "{kind}: 置換対象 {from} がある");
        // 余分キー・非 object（null・配列）はいずれも `Manifest` で拒否する（無引数のため
        // 「キー欠落」は成立せず、params の型違いがそれに相当する）。
        for bad in [r#"{"q":1}"#, "null", "[]"] {
            let to = format!("\"kind\":\"{kind}\",\"params\":{bad}");
            std::fs::write(dir.join("manifest.json"), text.replacen(&from, &to, 1))
                .expect("書けるはず");
            let err = load_err(&dir);
            assert!(
                matches!(err, ModelIoError::Manifest { .. }),
                "{kind} {bad}: {err}"
            );
        }
    }
}

// ---------------------------------------------------------------------
// イシュー #3055: conv_transpose1d・unflatten の保存・復元
// ---------------------------------------------------------------------

fn manifest_of(model: &Sequential, label: &str) -> (TempDirGuard, std::path::PathBuf, String) {
    let guard = TempDirGuard::new(label);
    let dir = guard.path().join("m");
    save_model(model, &dir).expect("保存できるはず");
    let text = std::fs::read_to_string(dir.join("manifest.json")).expect("読めるはず");
    (guard, dir, text)
}

fn deconv_manifest(label: &str) -> (TempDirGuard, std::path::PathBuf, String) {
    manifest_of(&deconv1d_model().expect("構築できるはず"), label)
}

fn deconv_tampered(label: &str, from: &str, to: &str) -> ModelIoError {
    let (_guard, dir, text) = deconv_manifest(label);
    assert!(text.contains(from), "{label}: 置換対象 {from} がある");
    std::fs::write(dir.join("manifest.json"), text.replacen(from, to, 1)).expect("書けるはず");
    match load_model(&dir) {
        Ok(_) => panic!("{label}: 改竄された manifest は拒否されるはず"),
        Err(e) => e,
    }
}

#[test]
fn conv_transpose1d_unflatten_restore_arguments_and_keep_format_version_2() {
    let model = deconv1d_model().expect("構築できるはず");
    let (_guard, dir, text) = manifest_of(&model, "deconv-args");
    assert!(
        text.contains("\"format_version\":2"),
        "format_version は 2 のまま"
    );
    assert!(text.contains(
        "\"kind\":\"conv_transpose1d\",\"params\":{\"in_channels\":2,\"out_channels\":4,\"kernel_size\":3,\"stride\":2,\"padding\":1,\"output_padding\":1,\"dilation\":2,\"groups\":2}"
    ), "{text}");
    assert!(text.contains("\"kind\":\"unflatten\",\"params\":{\"dim\":1,\"size_len\":2,\"size_0\":2,\"size_1\":4,\"size_2\":null"), "{text}");
    let loaded = load_model(&dir).expect("復元できるはず");
    let x = tensor(&[3, 6], 4.5);
    let y = loaded.predict(&x).expect("predict できるはず");
    assert_eq!(y.shape(), &[3, 2]);
    // 復元した conv_transpose1d の weight shape は `[in, out/groups, k]`。
    assert_eq!(loaded.state_dict()["2.weight"].shape(), &[2, 2, 3]);
}

#[test]
fn eleven_layer_subset_of_new_kinds_is_savable() {
    // #3053／#3054（活性化 9 層）は別イシュー。本イシューの 2 層が拒否されないことのみ確認する。
    let model = Sequential::new()
        .add_unflatten(1, vec![1, 6])
        .and_then(|m| m.add_conv_transpose1d(1, 1, 1, 1, 0, 0, 1, 1, 1))
        .expect("構築できるはず");
    let (_g, _d, text) = manifest_of(&model, "deconv-two");
    let kinds = kinds_of(&text);
    assert!(kinds.contains("conv_transpose1d") && kinds.contains("unflatten"));
}

#[test]
fn tampered_conv_transpose1d_and_unflatten_manifests_are_rejected() {
    for (label, from, to) in [
        ("ct-groups-zero", "\"groups\":2}", "\"groups\":0}"),
        ("ct-groups-indivisible", "\"groups\":2}", "\"groups\":3}"),
        (
            "ct-negative",
            "\"kernel_size\":3,\"stride\":2",
            "\"kernel_size\":-3,\"stride\":2",
        ),
        ("ct-real", "\"stride\":2", "\"stride\":2.0"),
        ("ct-missing-key", "\"output_padding\":1,", ""),
        (
            "ct-extra-key",
            "\"dilation\":2,",
            "\"dilation\":2,\"bias\":true,",
        ),
        ("uf-size-len-zero", "\"size_len\":2", "\"size_len\":0"),
        ("uf-size-len-nine", "\"size_len\":2", "\"size_len\":9"),
        ("uf-null-pack-short", "\"size_len\":2", "\"size_len\":3"),
        ("uf-null-pack-long", "\"size_len\":2", "\"size_len\":1"),
        ("uf-negative", "\"size_0\":2", "\"size_0\":-2"),
        ("uf-dim-real", "\"dim\":1,", "\"dim\":1.5,"),
    ] {
        let err = deconv_tampered(label, from, to);
        assert!(
            matches!(err, ModelIoError::Manifest { .. }),
            "{label}: {err}"
        );
    }
    // 意味検査は add_* に委ねる（Autodiff）。
    let label = "ct-output-padding-ge-stride";
    let err = deconv_tampered(label, "\"output_padding\":1", "\"output_padding\":2");
    assert!(matches!(err, ModelIoError::Autodiff(_)), "{label}: {err}");
}

#[test]
fn tampered_conv_transpose1d_weight_shape_is_rejected() {
    let (_g, dir, text) = deconv_manifest("ct-shape");
    assert!(text.contains("\"shape\":[2,2,3]"));
    std::fs::write(
        dir.join("manifest.json"),
        text.replacen("\"shape\":[2,2,3]", "\"shape\":[2,2,4]", 1),
    )
    .expect("書けるはず");
    assert!(matches!(
        load_err(&dir),
        ModelIoError::Mismatch { .. } | ModelIoError::Manifest { .. }
    ));
}

#[test]
fn unflatten_with_more_than_eight_axes_is_rejected_before_writing() {
    let model = Sequential::new()
        .add_unflatten(1, vec![1; 9])
        .expect("構築できるはず");
    assert_save_rejected("unflatten-9-axes", &model, false);
}
// ---------------------------------------------------------------------
// イシュー #3029: max_pool3d・avg_pool3d の保存・復元
// ---------------------------------------------------------------------
// 両層は無パラメータ層のため safetensors のパラメータキー過不足は成立せず、改竄の主対象は
// manifest の params（キー集合・型・範囲）になる。

/// rank 5 入力 `[2, 2, 5, 6, 6]`。conv3d → max_pool3d（stride Some・padding／dilation 非自明）→
/// avg_pool3d（stride None・`count_include_pad=false`）。
fn pool3d_model() -> Sequential {
    let mut m = Sequential::new()
        .add_conv3d(2, 4, [2, 2, 2], [1, 1, 1], [0, 0, 0], [1, 1, 1], 1, 29)
        .and_then(|m| m.add_max_pool3d([2, 3, 2], Some([1, 2, 1]), [1, 1, 0], [1, 1, 2]))
        .and_then(|m| m.add_avg_pool3d([2, 2, 2], None, [1, 0, 0], false))
        .expect("構築できるはず");
    m.eval();
    m
}

/// 全要素が異なる ramp 入力（max のタイブレークに依存させない）。
fn pool3d_input() -> Tensor<f32> {
    let shape = [2, 2, 5, 6, 6];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| (i as f32) * 0.013 - 1.7).collect(), &shape).unwrap()
}

#[test]
fn pool3d_layers_round_trip_bit_identically_and_keep_format_version_2() {
    let kinds = round_trip("pool3d", &pool3d_model(), &pool3d_input());
    for k in ["max_pool3d", "avg_pool3d"] {
        assert!(kinds.contains(k), "kind {k} が manifest にある: {kinds:?}");
    }
    let (_guard, _dir, text) = manifest_of(&pool3d_model(), "pool3d-manifest");
    assert!(
        text.contains("\"format_version\":2"),
        "format_version は 2 のまま"
    );
    assert!(
        text.contains("\"kind\":\"avg_pool3d\",\"params\":{")
            && text.contains("\"count_include_pad\":false"),
        "{text}"
    );
}

#[test]
fn pool3d_none_stride_and_count_include_pad_true_round_trip() {
    let mut m = Sequential::new()
        .add_max_pool3d([2, 2, 2], None, [0, 0, 0], [1, 1, 1])
        .and_then(|m| m.add_avg_pool3d([2, 2, 2], Some([1, 1, 1]), [0, 0, 0], true))
        .expect("構築できるはず");
    m.eval();
    let x = Tensor::new(
        (0..2 * 2 * 6 * 6 * 6).map(|i| (i as f32) * 0.01).collect(),
        &[2, 2, 6, 6, 6],
    )
    .unwrap();
    round_trip("pool3d-none-stride", &m, &x);
}

fn pool3d_tampered(label: &str, from: &str, to: &str) -> ModelIoError {
    let (_guard, dir, text) = manifest_of(&pool3d_model(), label);
    assert!(text.contains(from), "{label}: 置換対象 {from} がある");
    std::fs::write(dir.join("manifest.json"), text.replacen(from, to, 1)).expect("書けるはず");
    load_err(&dir)
}

#[test]
fn tampered_pool3d_manifests_are_rejected() {
    let max_head = "\"kind\":\"max_pool3d\",\"params\":{";
    // 構造違反（キー欠落・余分なキー・負値・小数・型違い・stride 部分 null）は `Manifest`。
    let manifest_cases = [
        ("missing-key", "\"kernel_size_d\":2,", ""),
        (
            "extra-key",
            max_head,
            "\"kind\":\"max_pool3d\",\"params\":{\"x\":1,",
        ),
        ("negative", "\"stride_d\":1", "\"stride_d\":-1"),
        ("fractional", "\"padding_w\":0", "\"padding_w\":0.5"),
        ("string", "\"dilation_h\":1", "\"dilation_h\":\"1\""),
        ("partial-null", "\"stride_h\":2", "\"stride_h\":null"),
        (
            "bool-for-int",
            "\"kernel_size_w\":2",
            "\"kernel_size_w\":true",
        ),
        (
            "int-for-bool",
            "\"count_include_pad\":false",
            "\"count_include_pad\":0",
        ),
    ];
    for (label, from, to) in manifest_cases {
        let err = pool3d_tampered(&format!("pool3d-t-{label}"), from, to);
        assert!(
            matches!(err, ModelIoError::Manifest { .. }),
            "{label}: {err}"
        );
    }
    // 意味違反は層コンストラクタの検査（`Autodiff`）。
    let autodiff_cases = [
        ("kernel-zero", "\"kernel_size_h\":3", "\"kernel_size_h\":0"),
        ("padding-over-half", "\"padding_d\":1", "\"padding_d\":2"),
    ];
    for (label, from, to) in autodiff_cases {
        let err = pool3d_tampered(&format!("pool3d-t-{label}"), from, to);
        assert!(matches!(err, ModelIoError::Autodiff(_)), "{label}: {err}");
    }
    // 未知 kind は `UnsupportedModel`。
    let err = pool3d_tampered(
        "pool3d-t-kind",
        "\"kind\":\"max_pool3d\"",
        "\"kind\":\"max_pool4d\"",
    );
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err}"
    );
}
// ---------------------------------------------------------------------
// イシュー #3033: conv_transpose3d の保存・復元
// ---------------------------------------------------------------------
// 三つ組（kernel_size・stride・padding・output_padding・dilation）は正準形の文字列 "d,h,w" 1 キーに
// 畳まれる（平坦形は 18 キーとなり object キー上限 16 を超えるため。決定記録 §3）。

/// rank 5 入力 `[1, 4, 2, 2, 2]`。conv_transpose3d（groups=2・全パラメータ非自明）→ relu →
/// conv_transpose3d（非対称の三つ組・groups=1）。
fn deconv3d_model() -> Sequential {
    let mut m = Sequential::new()
        .add_conv_transpose3d(
            4,
            6,
            [2, 3, 1],
            [2, 3, 1],
            [0, 1, 0],
            [1, 2, 0],
            [1, 2, 1],
            2,
            33,
        )
        .map(|m| m.add_relu())
        .and_then(|m| {
            m.add_conv_transpose3d(
                6,
                4,
                [1, 2, 3],
                [1, 1, 2],
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 1],
                1,
                34,
            )
        })
        .expect("構築できるはず");
    m.eval();
    m
}

fn deconv3d_input() -> Tensor<f32> {
    let shape = [1, 4, 2, 2, 2];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| (i as f32) * 0.07 - 0.5).collect(), &shape).unwrap()
}

#[test]
fn conv_transpose3d_round_trips_bit_identically_and_keeps_format_version_2() {
    let kinds = round_trip("deconv3d", &deconv3d_model(), &deconv3d_input());
    assert!(kinds.contains("conv_transpose3d"), "{kinds:?}");
    let (_guard, _dir, text) = manifest_of(&deconv3d_model(), "deconv3d-manifest");
    assert!(
        text.contains("\"format_version\":2"),
        "format_version は 2 のまま"
    );
    assert!(
        text.contains("\"kernel_size\":\"2,3,1\"") && text.contains("\"output_padding\":\"1,2,0\""),
        "{text}"
    );
}

fn deconv3d_tampered(label: &str, from: &str, to: &str) -> ModelIoError {
    let (_guard, dir, text) = manifest_of(&deconv3d_model(), label);
    assert!(text.contains(from), "{label}: 置換対象 {from} がある");
    std::fs::write(dir.join("manifest.json"), text.replacen(from, to, 1)).expect("書けるはず");
    load_err(&dir)
}

#[test]
fn tampered_conv_transpose3d_manifests_are_rejected() {
    let head = "\"kind\":\"conv_transpose3d\",\"params\":{";
    let manifest_cases = [
        ("missing-key", "\"dilation\":\"1,2,1\",", ""),
        (
            "extra-key",
            head,
            "\"kind\":\"conv_transpose3d\",\"params\":{\"x\":1,",
        ),
        (
            "negative",
            "\"kernel_size\":\"2,3,1\"",
            "\"kernel_size\":\"-2,3,1\"",
        ),
        (
            "fractional",
            "\"kernel_size\":\"2,3,1\"",
            "\"kernel_size\":\"2.5,3,1\"",
        ),
        ("int-for-string", "\"stride\":\"2,3,1\"", "\"stride\":2"),
        ("two-elements", "\"stride\":\"2,3,1\"", "\"stride\":\"2,3\""),
        (
            "four-elements",
            "\"stride\":\"2,3,1\"",
            "\"stride\":\"2,3,1,1\"",
        ),
        (
            "leading-zero",
            "\"stride\":\"2,3,1\"",
            "\"stride\":\"02,3,1\"",
        ),
        ("space", "\"stride\":\"2,3,1\"", "\"stride\":\"2, 3,1\""),
        ("sign", "\"stride\":\"2,3,1\"", "\"stride\":\"+2,3,1\""),
        (
            "overflow",
            "\"stride\":\"2,3,1\"",
            "\"stride\":\"18446744073709551616,3,1\"",
        ),
        ("groups-not-dividing", "\"groups\":2", "\"groups\":4"),
        ("groups-zero", "\"groups\":2", "\"groups\":0"),
    ];
    for (label, from, to) in manifest_cases {
        let err = deconv3d_tampered(&format!("deconv3d-t-{label}"), from, to);
        assert!(
            matches!(err, ModelIoError::Manifest { .. }),
            "{label}: {err}"
        );
    }
    // 意味違反（`output_padding >= stride`）は層コンストラクタの検査（`Autodiff`）。
    let err = deconv3d_tampered(
        "deconv3d-t-output-padding",
        "\"output_padding\":\"1,2,0\"",
        "\"output_padding\":\"2,2,0\"",
    );
    assert!(matches!(err, ModelIoError::Autodiff(_)), "{err}");
    // manifest の shape を変えると期待キー・shape と safetensors が食い違い `Mismatch`。
    let err = deconv3d_tampered(
        "deconv3d-t-shape",
        "\"kernel_size\":\"2,3,1\"",
        "\"kernel_size\":\"2,3,2\"",
    );
    assert!(matches!(err, ModelIoError::Mismatch { .. }), "{err}");
    // 未知 kind は `UnsupportedModel`。
    let err = deconv3d_tampered(
        "deconv3d-t-kind",
        "\"kind\":\"conv_transpose3d\"",
        "\"kind\":\"conv_transpose4d\"",
    );
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err}"
    );
}

// ---------------------------------------------------------------------
// イシュー #3041: unfold／fold の保存・復元
// ---------------------------------------------------------------------
// 無パラメータ層。平坦形（unfold 8 キー・fold 10 キー。object キー上限 16 以内。決定記録 §3）。

/// 入力 `[1, 2, 5, 5]`。unfold → fold（output_size = 入力の空間形状）→ relu。
fn fold_unfold_model() -> Sequential {
    let mut m = Sequential::new()
        .add_unfold([2, 3], [1, 2], [1, 0], [1, 1])
        .and_then(|m| m.add_fold([5, 5], [2, 3], [1, 2], [1, 0], [1, 1]))
        .map(|m| m.add_relu())
        .expect("構築できるはず");
    m.eval();
    m
}

fn fold_unfold_input() -> Tensor<f32> {
    let shape = [1, 2, 5, 5];
    let n: usize = shape.iter().product();
    Tensor::new((0..n).map(|i| (i as f32) * 0.05 - 0.6).collect(), &shape).unwrap()
}

#[test]
fn unfold_fold_round_trip_bit_identically_and_keep_format_version_2() {
    let kinds = round_trip("fold-unfold", &fold_unfold_model(), &fold_unfold_input());
    assert!(
        kinds.contains("unfold") && kinds.contains("fold"),
        "{kinds:?}"
    );
    let (_guard, _dir, text) = manifest_of(&fold_unfold_model(), "fold-unfold-manifest");
    assert!(
        text.contains("\"format_version\":2"),
        "format_version は 2 のまま"
    );
    assert!(
        text.contains("\"output_size_h\":5") && text.contains("\"dilation_w\":1"),
        "{text}"
    );
}

#[test]
fn unfold_only_and_fold_only_models_round_trip_independently() {
    let mut unfold = Sequential::new()
        .add_unfold([2, 3], [1, 2], [1, 0], [1, 2])
        .expect("構築できるはず");
    unfold.eval();
    let kinds = round_trip("unfold-only", &unfold, &fold_unfold_input());
    assert!(
        kinds.contains("unfold") && !kinds.contains("fold"),
        "{kinds:?}"
    );

    // unfold(K=[2,3], S=[1,2], P=[1,0], D=[1,1]) の列形状は [1, 12, 12]。
    let mut fold = Sequential::new()
        .add_fold([5, 5], [2, 3], [1, 2], [1, 0], [1, 1])
        .expect("構築できるはず");
    fold.eval();
    let cols = Tensor::new(
        (0..12 * 12).map(|i| (i as f32) * 0.01 - 0.3).collect(),
        &[1, 12, 12],
    )
    .unwrap();
    let kinds = round_trip("fold-only", &fold, &cols);
    assert!(
        kinds.contains("fold") && !kinds.contains("unfold"),
        "{kinds:?}"
    );
}

fn fold_tampered(label: &str, from: &str, to: &str) -> ModelIoError {
    let (_guard, dir, text) = manifest_of(&fold_unfold_model(), label);
    assert!(text.contains(from), "{label}: 置換対象 {from} がある");
    std::fs::write(dir.join("manifest.json"), text.replacen(from, to, 1)).expect("書けるはず");
    load_err(&dir)
}

#[test]
fn tampered_unfold_fold_manifests_are_rejected() {
    let manifest_cases = [
        ("unfold-missing-key", "\"dilation_w\":1}", "}"),
        (
            "unfold-extra-key",
            "\"kind\":\"unfold\",\"params\":{",
            "\"kind\":\"unfold\",\"params\":{\"x\":1,",
        ),
        (
            "fold-extra-key",
            "\"kind\":\"fold\",\"params\":{",
            "\"kind\":\"fold\",\"params\":{\"x\":1,",
        ),
        ("negative", "\"stride_h\":1", "\"stride_h\":-1"),
        ("fractional", "\"stride_h\":1", "\"stride_h\":1.5"),
        ("string", "\"stride_h\":1", "\"stride_h\":\"1\""),
        ("bool", "\"stride_h\":1", "\"stride_h\":true"),
        ("null", "\"stride_h\":1", "\"stride_h\":null"),
        (
            "overflow",
            "\"stride_h\":1",
            "\"stride_h\":18446744073709551616",
        ),
        ("fold-missing-output-size", "\"output_size_h\":5,", ""),
    ];
    for (label, from, to) in manifest_cases {
        let err = fold_tampered(&format!("fold-t-{label}"), from, to);
        assert!(
            matches!(err, ModelIoError::Manifest { .. }),
            "{label}: {err}"
        );
    }
    // 意味違反（0 次元）は層コンストラクタの検査（`Autodiff`）。
    for (label, from, to) in [
        ("kernel-zero", "\"kernel_size_h\":2", "\"kernel_size_h\":0"),
        ("stride-zero", "\"stride_w\":2", "\"stride_w\":0"),
        (
            "output-size-zero",
            "\"output_size_w\":5",
            "\"output_size_w\":0",
        ),
    ] {
        let err = fold_tampered(&format!("fold-t-{label}"), from, to);
        assert!(matches!(err, ModelIoError::Autodiff(_)), "{label}: {err}");
    }
    // 未知 kind は `UnsupportedModel`。
    let err = fold_tampered(
        "fold-t-kind",
        "\"kind\":\"unfold\"",
        "\"kind\":\"unfold3d\"",
    );
    assert!(
        matches!(err, ModelIoError::UnsupportedModel { .. }),
        "{err}"
    );
}
