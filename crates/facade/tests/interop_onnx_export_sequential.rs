//! `fandhe_ai::interop::onnx::OnnxModel::from_sequential`（イシュー
//! #2037・親 #2034）の facade 単独（`fandhe_ai` と `std` のみ import）
//! 統合テスト。
//!
//! **本ファイルは意図的に `fandhe_ai_onnx_interop`／`fandhe_ai_autodiff`
//! を import しない**（facade のみで学習済み `compat::Sequential` から
//! ONNX への export→import roundtrip が成立し、`Sequential::predict`
//! と bit 完全一致することを検証するため。内部クレート直接呼び出しと
//! の一致確認は `tests/interop_onnx_internal_parity.rs` を参照）。
//!
//! 比較はすべて bit 同一（f32 は `to_bits()`）。tolerance は使わない・
//! 導入もしない。
//!
//! グローバル RNG（[`fandhe_ai::manual_seed`]／[`fandhe_ai::randn`]）を
//! 使うため、本ファイル内のテスト同士は `cargo test` の既定並列実行で
//! 競合しうる。ファイル局所 `Mutex` で直列化する
//! （`crates/facade/tests/rng_tensor_generation.rs` と同型）。

use std::collections::HashMap;
use std::sync::Mutex;

use fandhe_ai::Tensor;
use fandhe_ai::compat::{FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::interop::onnx::{OnnxError, OnnxExportOptions, OnnxModel, OnnxValue};
use fandhe_ai::optim::SgdConfig;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

/// グローバル RNG から `[n, d]` 形状の一様乱数テンソルを生成する（毎回
/// `manual_seed` してから呼ぶことで再現性を確保する。呼び出し元が
/// `test_lock()` を保持していることが前提）。
fn gen_input(seed: u64, n: usize, d: usize) -> Tensor<f32> {
    fandhe_ai::manual_seed(seed);
    fandhe_ai::rand(&[n, d]).expect("test fixture: rand 生成に失敗")
}

fn run_single_output(model: &OnnxModel, input: &Tensor<f32>) -> Tensor<f32> {
    let mut feeds = HashMap::new();
    feeds.insert("input".to_string(), OnnxValue::F32(input.clone()));
    let outputs = model.run(feeds).expect("test fixture: run 成功");
    match outputs.into_iter().next() {
        Some((name, OnnxValue::F32(t))) => {
            assert_eq!(name, "output", "出力名は常に \"output\" のはず");
            t
        }
        other => panic!("OnnxValue::F32 を期待したが {other:?}"),
    }
}

fn assert_bit_exact(a: &Tensor<f32>, b: &Tensor<f32>, context: &str) {
    assert_eq!(a.shape(), b.shape(), "{context}: shape が一致しない");
    let a_slice = a.as_slice().expect("test fixture: contiguous のはず");
    let b_slice = b.as_slice().expect("test fixture: contiguous のはず");
    assert_eq!(
        a_slice.len(),
        b_slice.len(),
        "{context}: 要素数が一致しない"
    );
    assert!(
        !a_slice.is_empty(),
        "{context}: 空虚 pass 防止（出力要素数 > 0 のはず）"
    );
    for (i, (x, y)) in a_slice.iter().zip(b_slice.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{context}: index={i} で bit 不一致 (a={x}, b={y})"
        );
    }
}

/// a. 学習済み MLP の roundtrip: `from_sequential(&m).to_bytes()` →
///    `from_bytes` → `run` が `m.predict(&x)` と bit 完全一致する
///    （`Linear→ReLU` の融合経路 `gemm_bias_act` と非融合 `Gemm`＋`Relu`
///    の一致を facade 単独で確認するのが主眼）。
#[test]
fn trained_mlp_roundtrip_matches_predict_bit_exact() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    const D_IN: usize = 4;
    const D_HIDDEN: usize = 8;
    const D_OUT: usize = 2;
    const N: usize = 6;

    let x = gen_input(0x2037_0001, N, D_IN);
    let y = gen_input(0x2037_0002, N, D_OUT);

    let mut model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x2037_1111)
        .expect("test fixture: 層 1 の構築に失敗")
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2037_2222)
        .expect("test fixture: 層 2 の構築に失敗");

    model
        .compile(Optimizer::Sgd(SgdConfig::new(0.01)), Loss::Mse)
        .expect("test fixture: compile 成功");
    model
        .fit(&x, &y, FitConfig::new(3, N))
        .expect("test fixture: fit 成功");

    let x_test = gen_input(0x2037_0003, N, D_IN);
    let predicted = model.predict(&x_test).expect("test fixture: predict 成功");

    let exported = OnnxModel::from_sequential(&model).expect("from_sequential 成功");
    let bytes = exported
        .to_bytes(&OnnxExportOptions::default())
        .expect("to_bytes 成功");
    let reimported = OnnxModel::from_bytes(&bytes).expect("from_bytes(再 import) 成功");
    let onnx_out = run_single_output(&reimported, &x_test);

    assert_bit_exact(&predicted, &onnx_out, "predict vs onnx roundtrip");
}

/// b. ブロックタイル境界（KC=256）を跨ぐ形状での bit 一致（学習なし・
///    初期化のみ。`Linear(in_features>=512)→ReLU→Linear`）。
#[test]
fn large_shape_crossing_block_tile_boundary_matches_predict_bit_exact() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    const D_IN: usize = 600;
    const D_HIDDEN: usize = 300;
    const D_OUT: usize = 5;
    const N: usize = 3;

    let x = gen_input(0x2037_0010, N, D_IN);

    let model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x2037_3333)
        .expect("test fixture: 層 1 の構築に失敗")
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2037_4444)
        .expect("test fixture: 層 2 の構築に失敗");

    let predicted = model.predict(&x).expect("test fixture: predict 成功");

    let exported = OnnxModel::from_sequential(&model).expect("from_sequential 成功");
    let bytes = exported
        .to_bytes(&OnnxExportOptions::default())
        .expect("to_bytes 成功");
    let reimported = OnnxModel::from_bytes(&bytes).expect("from_bytes(再 import) 成功");
    let onnx_out = run_single_output(&reimported, &x);

    assert_bit_exact(
        &predicted,
        &onnx_out,
        "large shape predict vs onnx roundtrip",
    );
}

/// c. 決定性・不動点: 再構築しても `to_bytes` は同一バイト列・
///    `from_bytes(&b1).to_bytes() == b1`・`eval()` 前後で不変。
#[test]
fn from_sequential_to_bytes_is_deterministic_and_idempotent() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let mut model = Sequential::new()
        .add_linear(3, 5, 0x2037_5555)
        .expect("test fixture: 層構築に失敗")
        .add_relu()
        .add_linear(5, 2, 0x2037_6666)
        .expect("test fixture: 層構築に失敗");

    let opts = OnnxExportOptions::default();

    let b1 = OnnxModel::from_sequential(&model)
        .expect("from_sequential 成功(1回目)")
        .to_bytes(&opts)
        .expect("to_bytes 成功(1回目)");
    let b1_again = OnnxModel::from_sequential(&model)
        .expect("from_sequential 成功(再構築)")
        .to_bytes(&opts)
        .expect("to_bytes 成功(再構築)");
    assert_eq!(b1, b1_again, "再構築しても to_bytes は同一バイト列のはず");

    let roundtrip_bytes = OnnxModel::from_bytes(&b1)
        .expect("from_bytes 成功")
        .to_bytes(&opts)
        .expect("to_bytes 成功(roundtrip)");
    assert_eq!(
        roundtrip_bytes, b1,
        "from_bytes(&b1).to_bytes() は b1 と同一バイト列のはず（不動点）"
    );

    model.eval();
    let b1_after_eval = OnnxModel::from_sequential(&model)
        .expect("from_sequential 成功(eval後)")
        .to_bytes(&opts)
        .expect("to_bytes 成功(eval後)");
    assert_eq!(
        b1_after_eval, b1,
        "eval() 前後で to_bytes は不変のはず（対応層は train/eval で挙動が変わらない）"
    );
}

/// d. `Graph` 直接保持の等価性: `from_sequential(&m).run(feeds)` と
///    `from_bytes(&b1).run(feeds)` の出力が bit 一致する。
#[test]
fn from_sequential_direct_graph_matches_from_bytes_roundtrip_bit_exact() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    let model = Sequential::new()
        .add_linear(4, 6, 0x2037_7777)
        .expect("test fixture: 層構築に失敗")
        .add_relu()
        .add_linear(6, 3, 0x2037_8888)
        .expect("test fixture: 層構築に失敗");

    let x = gen_input(0x2037_0020, 2, 4);

    let direct = OnnxModel::from_sequential(&model).expect("from_sequential 成功");
    let bytes = direct
        .to_bytes(&OnnxExportOptions::default())
        .expect("to_bytes 成功");
    let via_bytes = OnnxModel::from_bytes(&bytes).expect("from_bytes 成功");

    let out_direct = run_single_output(&direct, &x);
    let out_via_bytes = run_single_output(&via_bytes, &x);

    assert_bit_exact(
        &out_direct,
        &out_via_bytes,
        "direct graph vs from_bytes roundtrip",
    );
}

/// e-1. 空の `Sequential`（層 0 個）は `OnnxError::InvalidModel` で
///      拒否される。
#[test]
fn empty_sequential_is_rejected_with_invalid_model() {
    let model = Sequential::new();
    let err = OnnxModel::from_sequential(&model).unwrap_err();
    assert!(
        matches!(err, OnnxError::InvalidModel { .. }),
        "空の Sequential は InvalidModel で拒否されるはず: {err:?}"
    );
}

/// e-2. `Sigmoid`／`Tanh` を含む `Sequential` は `layer_kind == "unknown"`
///      の `OnnxError::UnsupportedLayer`（該当層の index 付き）で拒否
///      される。
#[test]
fn sequential_with_sigmoid_is_rejected_with_unsupported_layer() {
    let model = Sequential::new()
        .add_linear(2, 2, 0x2037_9999)
        .expect("test fixture: 層構築に失敗")
        .add_sigmoid();

    let err = OnnxModel::from_sequential(&model).unwrap_err();
    match err {
        OnnxError::UnsupportedLayer { index, layer_kind } => {
            assert_eq!(index, 1, "Sigmoid は index=1（Linear の次）のはず");
            assert_eq!(
                layer_kind, "unknown",
                "Sigmoid は判別フック非対応のため unknown のはず"
            );
        }
        other => panic!("UnsupportedLayer を期待したが {other:?}"),
    }
}

/// e-3. `Conv2d` を含む `Sequential` は `layer_kind == "Conv2d"` の
///      `OnnxError::UnsupportedLayer` で拒否される（`as_conv2d` フック
///      による判別）。
#[test]
fn sequential_with_conv2d_is_rejected_with_unsupported_layer_conv2d() {
    let model = Sequential::new()
        .add_conv2d(1, 2, [3, 3], [1, 1], [0, 0], [1, 1], 1, 0x2037_aaaa)
        .expect("test fixture: 層構築に失敗");

    let err = OnnxModel::from_sequential(&model).unwrap_err();
    match err {
        OnnxError::UnsupportedLayer { index, layer_kind } => {
            assert_eq!(index, 0);
            assert_eq!(layer_kind, "Conv2d");
        }
        other => panic!("UnsupportedLayer(layer_kind=Conv2d) を期待したが {other:?}"),
    }
}

/// e-4. 対応層の後に非対応層が続く場合も `Graph` を一切構築せず `Err`
///      を返す（部分モデルが返らないことを `Result` が `Err` であること
///      自体で担保する）。
#[test]
fn unsupported_layer_after_supported_layers_is_still_rejected() {
    let model = Sequential::new()
        .add_linear(2, 4, 0x2037_bbbb)
        .expect("test fixture: 層構築に失敗")
        .add_relu()
        .add_linear(4, 2, 0x2037_cccc)
        .expect("test fixture: 層構築に失敗")
        .add_sigmoid();

    let err = OnnxModel::from_sequential(&model).unwrap_err();
    match err {
        OnnxError::UnsupportedLayer { index, .. } => {
            assert_eq!(
                index, 3,
                "Sigmoid は index=3（Linear,ReLU,Linear の次）のはず"
            );
        }
        other => panic!("UnsupportedLayer を期待したが {other:?}"),
    }
}

/// e-5. `from_sequential` で構築した `OnnxModel` の `to_path` が `Err`
///      を返す場合（存在しない親ディレクトリ）はファイルを作成しない
///      （既存 #2018 契約〈`OnnxModel::to_path`〉が `from_sequential`
///      経由の構築でも継承されることの end-to-end 確認。
///      `interop_onnx_export.rs::
///      to_path_with_nonexistent_parent_directory_returns_io_error`
///      と同型）。
#[test]
fn to_path_does_not_create_file_when_export_fails() {
    let model = Sequential::new()
        .add_linear(2, 2, 0x2037_dddd)
        .expect("test fixture: 層構築に失敗");

    let exported = OnnxModel::from_sequential(&model).expect("from_sequential 成功");

    let nested_path = std::env::temp_dir().join(format!(
        "fandhe-ai-onnx-export-sequential-test-{}-nonexistent-dir/model.onnx",
        std::process::id()
    ));
    assert!(
        !nested_path.parent().unwrap().exists(),
        "test fixture: 親ディレクトリが存在しないはず"
    );

    let err = exported
        .to_path(&nested_path, &OnnxExportOptions::default())
        .unwrap_err();
    assert!(
        matches!(err, OnnxError::Io(_)),
        "存在しない親ディレクトリへの to_path は OnnxError::Io を返すはず: {err:?}"
    );
    assert!(
        !nested_path.exists(),
        "失敗した to_path がファイルを作成してしまっている"
    );
}

/// e-6. `run` に誤った feed 名（`"x"`）を渡すと
///      `OnnxError::MissingFeed { input: "input" }` が返る（入力名契約
///      `"input"` の実効性確認）。
#[test]
fn run_with_wrong_feed_name_returns_missing_feed() {
    let model = Sequential::new()
        .add_linear(2, 2, 0x2037_eeee)
        .expect("test fixture: 層構築に失敗");

    let exported = OnnxModel::from_sequential(&model).expect("from_sequential 成功");

    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(vec![0.0, 0.0], &[1, 2]).unwrap()),
    );
    let err = exported.run(feeds).unwrap_err();
    match err {
        OnnxError::MissingFeed { input } => assert_eq!(input, "input"),
        other => panic!("MissingFeed を期待したが {other:?}"),
    }
}
