//! `fandhe_ai::interop::onnx`（`OnnxModel`／`OnnxValue`／`OnnxError`。
//! イシュー #2017）の facade 単独到達性テスト。
//!
//! **本ファイルは `fandhe_ai` と `std` のみを import する**（facade だけで
//! ONNX モデルを読み込み・実行できることの直接的な裏付け。
//! `docs/facade-onnx-import-exposure-decision.md` §6.1 の受け入れ条件）。
//! 内部クレート（`fandhe_ai_onnx_interop`）との出力突合は
//! `tests/interop_onnx_internal_parity.rs` が別途担う。
//!
//! fixture は本クレート外（`crates/onnx-interop/tests/fixtures/`）を
//! `CARGO_MANIFEST_DIR` 相対で直接参照し複製しない（workspace 内テスト
//! 前提。`cargo publish` の verify はテストを走らせないため公開を阻害
//! しない）。
//!
//! 判定式は REQ-7 事前固定基準 `abs_err / (|ref| + 1e-6) <= 1e-3`
//! （`crates/onnx-interop/tests/onnx_interp.rs` と同一。REQ-2 バックエンド
//! 間数値一致 OR 複合判定とは別指標であり混同しない）。

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai::Tensor;
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel, OnnxValue};

fn onnx_interop_fixture(rel: &str) -> PathBuf {
    PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../onnx-interop/tests/fixtures"
    ))
    .join(rel)
}

/// `crates/onnx-interop/tests/fixtures/onnx-reference/onnx_reference.json` の
/// 8 サンプルをテスト内定数として転記した値（PoC-v2-6 実測値。出典は
/// 同 fixture ファイル）。
const MODEL_ONNX_SAMPLES: [([f32; 2], f32); 8] = [
    ([0.0, 0.0], 1.143_025_6e-5),
    ([1.0, 0.0], 0.994_080_36),
    ([0.0, 1.0], 0.994_080_36),
    ([1.0, 1.0], 1.693_155_9e-6),
    ([0.3, 0.7], 0.312_604_4),
    ([-0.2, 1.1], 0.994_080_36),
    ([0.5, 0.5], 0.003_300_812_3),
    ([2.0, -1.0], 0.993_433_83),
];

#[test]
fn model_onnx_matches_reference_via_from_path_within_req7_tolerance() {
    let model =
        OnnxModel::from_path(onnx_interop_fixture("model.onnx")).expect("from_path は成功するはず");

    let mut max_rel_err = 0.0f32;
    for &(input, expected) in &MODEL_ONNX_SAMPLES {
        let mut feeds = HashMap::new();
        feeds.insert(
            "input".to_string(),
            OnnxValue::F32(Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap()),
        );
        let result = model.run(feeds).expect("run は成功するはず");
        let actual = match &result["output"] {
            OnnxValue::F32(t) => t.get(&[0, 0]).unwrap(),
            other => panic!("OnnxValue::F32 を期待したが {other:?}"),
        };
        let abs_err = (actual - expected).abs();
        let rel_err = abs_err / (expected.abs() + 1e-6);
        max_rel_err = max_rel_err.max(rel_err);
        assert!(
            rel_err <= 1e-3,
            "REQ-7 数値一致基準を超過: input={input:?} expected={expected} actual={actual} \
             abs_err={abs_err} rel_err={rel_err}"
        );
    }
    eprintln!("max_rel_err={max_rel_err} (threshold=1e-3)");
}

#[test]
fn model_onnx_matches_reference_via_from_bytes() {
    let bytes = std::fs::read(onnx_interop_fixture("model.onnx")).expect("fixture 読み込み失敗");
    let model = OnnxModel::from_bytes(&bytes).expect("from_bytes は成功するはず");

    let (input, expected) = MODEL_ONNX_SAMPLES[1];
    let mut feeds = HashMap::new();
    feeds.insert(
        "input".to_string(),
        OnnxValue::F32(Tensor::<f32>::new(input.to_vec(), &[1, 2]).unwrap()),
    );
    let result = model.run(feeds).expect("run は成功するはず");
    let actual = match &result["output"] {
        OnnxValue::F32(t) => t.get(&[0, 0]).unwrap(),
        other => panic!("OnnxValue::F32 を期待したが {other:?}"),
    };
    let rel_err = (actual - expected).abs() / (expected.abs() + 1e-6);
    assert!(rel_err <= 1e-3, "rel_err={rel_err} を超過");
}

/// `slice_repro.onnx`（動的境界 Slice パターン）が `from_path` → `run` で
/// 成功し、出力 shape・値が参照どおりであることを確認する（値は
/// `crates/onnx-interop/tests/fixtures/onnx-reference/
/// slice_repro_reference.json` からの転記）。
#[test]
fn slice_repro_onnx_end_to_end_via_from_path() {
    let model = OnnxModel::from_path(onnx_interop_fixture("slice_repro.onnx"))
        .expect("from_path は成功するはず");

    let input: [[f32; 6]; 5] = [
        [
            0.304_717_1,
            -1.039_984_1,
            0.750_451_2,
            0.940_564_7,
            -1.951_035_1,
            -1.302_179_5,
        ],
        [
            0.127_840_4,
            -0.316_242_6,
            -0.016_801_158,
            -0.853_043_9,
            0.879_398,
            0.777_791_9,
        ],
        [
            0.066_030_696,
            1.127_241_3,
            0.467_509_33,
            -0.859_292_45,
            0.368_750_78,
            -0.958_882_63,
        ],
        [
            0.878_450_3,
            -0.049_925_912,
            -0.184_862_36,
            -0.680_929_54,
            1.222_541_3,
            -0.154_529_48,
        ],
        [
            -0.428_327_83,
            -0.352_133_54,
            0.532_309_2,
            0.365_444_06,
            0.412_732_6,
            0.430_821,
        ],
    ];
    let flat: Vec<f32> = input.iter().flatten().copied().collect();
    let x = Tensor::<f32>::new(flat, &[5, 6]).unwrap();

    let mut feeds = HashMap::new();
    feeds.insert("x".to_string(), OnnxValue::F32(x));
    let result = model.run(feeds).expect("run は成功するはず");
    let y = match &result["output"] {
        OnnxValue::F32(t) => t,
        other => panic!("OnnxValue::F32 を期待したが {other:?}"),
    };
    assert_eq!(y.shape(), &[5, 4]);
    // slice_repro は先頭 4 列をそのまま切り出す（`slice_repro_reference.json`
    // の出力が入力の先頭 4 列と一致することをコメントに残す）。
    for (i, row) in input.iter().enumerate() {
        for (j, &expected) in row.iter().enumerate().take(4) {
            let actual = y.get(&[i, j]).unwrap();
            let rel_err = (actual - expected).abs() / (expected.abs() + 1e-6);
            assert!(
                rel_err <= 1e-3,
                "REQ-7 数値一致基準を超過: (i={i},j={j}) expected={expected} actual={actual}"
            );
        }
    }
}

// --- 負例（no-silent-skip 契約・OWASP A03: 不正入力の fail-closed 拒否） ---

#[test]
fn from_bytes_rejects_garbage_bytes_with_decode_error() {
    let garbage = [0x08u8, 0xffu8];
    let err = OnnxModel::from_bytes(&garbage).unwrap_err();
    assert!(
        matches!(err, OnnxError::Decode { .. }),
        "OnnxError::Decode を期待したが {err:?}"
    );
}

#[test]
fn from_bytes_rejects_empty_bytes_with_invalid_model() {
    // 空バイト列は有効な protobuf として decode 自体は成功するが、
    // `graph` フィールドが存在しないため `GraphError::NoGraph` ->
    // `OnnxError::InvalidModel` へ写像される。
    let err = OnnxModel::from_bytes(&[]).unwrap_err();
    assert!(
        matches!(err, OnnxError::InvalidModel { .. }),
        "OnnxError::InvalidModel を期待したが {err:?}"
    );
}

#[test]
fn from_path_rejects_nonexistent_path_with_io_error() {
    let err = OnnxModel::from_path("/nonexistent/path/does-not-exist.onnx").unwrap_err();
    assert!(
        matches!(err, OnnxError::Io(_)),
        "OnnxError::Io を期待したが {err:?}"
    );
}

#[test]
fn run_rejects_missing_required_feed() {
    let model = OnnxModel::from_path(onnx_interop_fixture("model.onnx")).unwrap();
    let err = model.run(HashMap::new()).unwrap_err();
    assert!(
        matches!(&err, OnnxError::MissingFeed { input } if input == "input"),
        "OnnxError::MissingFeed を期待したが {err:?}"
    );
}

#[test]
fn run_rejects_unknown_feed_name() {
    let model = OnnxModel::from_path(onnx_interop_fixture("model.onnx")).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert(
        "input".to_string(),
        OnnxValue::F32(Tensor::<f32>::zeros(&[1, 2]).unwrap()),
    );
    feeds.insert(
        "not_a_real_input".to_string(),
        OnnxValue::F32(Tensor::<f32>::zeros(&[1]).unwrap()),
    );
    let err = model.run(feeds).unwrap_err();
    assert!(
        matches!(&err, OnnxError::UnknownFeed { name } if name == "not_a_real_input"),
        "OnnxError::UnknownFeed を期待したが {err:?}"
    );
}

/// `OnnxError` が `#[non_exhaustive]` のため、ワイルドカード腕を用いた
/// `match` が facade 単独でコンパイルできること（名指し可能性）を
/// コンパイル時に固定する。
#[test]
fn onnx_error_is_matchable_via_facade_only() {
    fn classify(e: &OnnxError) -> &'static str {
        match e {
            OnnxError::Io(_) => "io",
            OnnxError::Decode { .. } => "decode",
            OnnxError::UnsupportedDataType { .. } => "unsupported_data_type",
            OnnxError::UnsupportedOp { .. } => "unsupported_op",
            OnnxError::MissingFeed { .. } => "missing_feed",
            OnnxError::UnknownFeed { .. } => "unknown_feed",
            OnnxError::InvalidModel { .. } => "invalid_model",
            OnnxError::Execution { .. } => "execution",
            _ => "unknown",
        }
    }
    let err = OnnxModel::from_bytes(&[0x08u8, 0xffu8]).unwrap_err();
    assert_eq!(classify(&err), "decode");
}
