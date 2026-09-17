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
    ([0.0, 0.0], 1.1430255653976928e-05),
    ([1.0, 0.0], 0.9940803647041321),
    ([0.0, 1.0], 0.9940803647041321),
    ([1.0, 1.0], 1.6931559230215498e-06),
    ([0.30000001192092896, 0.699999988079071], 0.3126043975353241),
    (
        [-0.20000000298023224, 1.100000023841858],
        0.9940803647041321,
    ),
    ([0.5, 0.5], 0.003300812328234315),
    ([2.0, -1.0], 0.9934338331222534),
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
            0.304717093706131,
            -1.039984107017517,
            0.7504512071609497,
            0.9405646920204163,
            -1.9510351419448853,
            -1.3021794557571411,
        ],
        [
            0.12784039974212646,
            -0.31624260544776917,
            -0.01680115796625614,
            -0.8530439138412476,
            0.879397988319397,
            0.7777919173240662,
        ],
        [
            0.06603069603443146,
            1.1272412538528442,
            0.46750932931900024,
            -0.8592924475669861,
            0.36875078082084656,
            -0.9588826298713684,
        ],
        [
            0.8784502744674683,
            -0.04992591217160225,
            -0.18486236035823822,
            -0.6809295415878296,
            1.222541332244873,
            -0.15452948212623596,
        ],
        [
            -0.4283278286457062,
            -0.35213354229927063,
            0.5323091745376587,
            0.3654440641403198,
            0.4127326011657715,
            0.4308210015296936,
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
    for i in 0..5 {
        for j in 0..4 {
            let expected = input[i][j];
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
