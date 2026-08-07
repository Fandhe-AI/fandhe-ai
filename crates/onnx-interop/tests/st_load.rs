//! `onnx-interop::st_load` の統合テスト（TASK-7.1a・#73・REQ-7）。
//!
//! イシュー #73 の受け入れ条件「PyTorch 保存の重みが自作テンソルとして
//! ロードできる」を、`tests/fixtures/pytorch-reference/`（PoC-v2-6 由来の
//! 固定 fixture。出自は同ディレクトリの README 参照）を用いて直接検証する。
//! CI（self-hosted）は `docs/spec`（submodule）を checkout しないため
//! （`crates/tensor-core/tests/tensor_views.rs` 冒頭コメント参照）、
//! 本ファイルは `docs/spec` 配下のいかなるファイルにも依存しない。

use std::collections::HashMap;
use std::path::Path;

use onnx_interop::st_load::{
    LoadError, load_safetensors_f32, load_safetensors_f32_from_bytes, require_keys,
};
use safetensors::Dtype;
use safetensors::tensor::{TensorView, serialize};
use serde::Deserialize;
use tensor_core::Tensor;

const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/pytorch-reference"
);

const WEIGHT_KEYS: [&str; 6] = [
    "fc1.weight",
    "fc1.bias",
    "fc2.weight",
    "fc2.bias",
    "fc3.weight",
    "fc3.bias",
];

#[derive(Deserialize)]
struct StReference {
    inputs: Vec<[f32; 2]>,
    outputs: Vec<f32>,
}

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(FIXTURE_DIR).join(name)
}

fn load_fixture_weights() -> HashMap<String, Tensor<f32>> {
    load_safetensors_f32(&fixture_path("model.safetensors")).expect("fixture のロードに失敗した")
}

// --- 受け入れ条件 1: PyTorch 保存の重みが自作テンソルとしてロードできる ---

#[test]
fn loads_pytorch_weights_with_expected_keys_and_shapes() {
    let map = load_fixture_weights();

    require_keys(&map, &WEIGHT_KEYS).expect("6 キーすべてが揃っているはず");

    let shapes_json = std::fs::read_to_string(fixture_path("weight_shapes.json")).unwrap();
    let shapes: HashMap<String, Vec<usize>> = serde_json::from_str(&shapes_json).unwrap();
    for key in WEIGHT_KEYS {
        let expected = &shapes[key];
        let actual = map[key].shape();
        assert_eq!(actual, expected.as_slice(), "shape mismatch for key {key}");
    }
}

// --- 明示転置: `[out,in]` -> `[in,out]`（REQ-7 の暗黙アダプタ禁止契約） ---

#[test]
fn transpose_2d_on_loaded_weight_matches_pytorch_shape_swap() {
    let map = load_fixture_weights();

    // fc1.weight は weight_shapes.json 上 [8, 2]（[out=8, in=2]）。
    let fc1_w = &map["fc1.weight"];
    assert_eq!(fc1_w.shape(), &[8, 2]);

    let fc1_w_t = fc1_w.transpose_2d().unwrap();
    assert_eq!(fc1_w_t.shape(), &[2, 8]);
    for i in 0..8 {
        for j in 0..2 {
            assert_eq!(
                fc1_w_t.get(&[j, i]).unwrap(),
                fc1_w.get(&[i, j]).unwrap(),
                "transpose_2d が値の対応を保っていない (i={i}, j={j})"
            );
        }
    }
}

// --- 数値一致（end-to-end）: ロード済み重みでの素朴 MLP forward ---
//
// 判定式は REQ-7 事前固定基準 `abs_err / (|ref| + 1e-6) <= 1e-3`
// （`.claude/rules/coding-rust.md`「許容誤差を単独で緩和しない」）。
// forward はテスト専用のローカル実装（PoC-v2-6 `mlp.rs` 相当を
// このテストファイル内に再実装したもの。onnx-interop は matmul 等の
// 演算カーネルへ依存しないため、Tensor::get() 経由の素朴な実装とする）。

fn linear(x: &[f32], w: &Tensor<f32>, b: &Tensor<f32>, in_dim: usize, out_dim: usize) -> Vec<f32> {
    // w は [in_dim, out_dim]（呼び出し側で transpose_2d 済み）。
    let w = w.contiguous();
    let mut out = vec![0.0f32; out_dim];
    for (o, out_val) in out.iter_mut().enumerate() {
        let mut acc = b.get(&[o]).unwrap();
        for (i, &x_val) in x.iter().enumerate().take(in_dim) {
            acc += x_val * w.get(&[i, o]).unwrap();
        }
        *out_val = acc;
    }
    out
}

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

fn sigmoid(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect()
}

fn mlp_forward(map: &HashMap<String, Tensor<f32>>, input: &[f32; 2]) -> f32 {
    let fc1_w = map["fc1.weight"].transpose_2d().unwrap();
    let fc2_w = map["fc2.weight"].transpose_2d().unwrap();
    let fc3_w = map["fc3.weight"].transpose_2d().unwrap();

    let h1 = relu(&linear(input, &fc1_w, &map["fc1.bias"], 2, 8));
    let h2 = relu(&linear(&h1, &fc2_w, &map["fc2.bias"], 8, 8));
    let out = sigmoid(&linear(&h2, &fc3_w, &map["fc3.bias"], 8, 1));
    out[0]
}

#[test]
fn forward_matches_pytorch_reference_within_req7_tolerance() {
    let map = load_fixture_weights();
    let reference_json = std::fs::read_to_string(fixture_path("st_reference.json")).unwrap();
    let reference: StReference = serde_json::from_str(&reference_json).unwrap();

    assert_eq!(reference.inputs.len(), reference.outputs.len());
    for (input, &expected) in reference.inputs.iter().zip(reference.outputs.iter()) {
        let actual = mlp_forward(&map, input);
        let abs_err = (actual - expected).abs();
        let rel_denom = expected.abs() + 1e-6;
        assert!(
            abs_err / rel_denom <= 1e-3,
            "REQ-7 数値一致基準を超過: input={input:?} expected={expected} actual={actual} abs_err={abs_err}"
        );
    }
}

// --- エラー系 ---

#[test]
fn missing_keys_reports_all_missing_keys_not_a_silent_skip() {
    let map = load_fixture_weights();
    let err = require_keys(&map, &["fc1.weight", "does.not.exist", "also.missing"]).unwrap_err();
    match err {
        LoadError::MissingKeys(keys) => {
            assert_eq!(
                keys,
                vec!["does.not.exist".to_string(), "also.missing".to_string()]
            );
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn corrupted_bytes_return_safetensors_format_error() {
    let bytes = b"this is not a valid safetensors file";
    let err = load_safetensors_f32_from_bytes(bytes).unwrap_err();
    assert!(matches!(err, LoadError::SafetensorsFormat(_)));
}

#[test]
fn unsupported_dtype_is_reported_not_silently_skipped() {
    // F16 テンソルをテスト内で生成する（fixture に F16 は含まれないため）。
    let data: Vec<u8> = vec![0u8; 2]; // f16 1 要素 = 2 バイト。
    let view = TensorView::new(Dtype::F16, vec![1], &data).unwrap();
    let bytes = serialize([("w".to_string(), view)], None).unwrap();

    let err = load_safetensors_f32_from_bytes(&bytes).unwrap_err();
    match err {
        LoadError::UnsupportedDtype { key, dtype } => {
            assert_eq!(key, "w");
            assert_eq!(dtype, "F16");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn truncated_data_is_rejected_before_conversion() {
    // F32 2 要素分（8 バイト）を要求する shape に対し、1 要素分（4 バイト）
    // しかデータ本体がない safetensors バイト列を手組みして検証する
    // （ヘッダの shape が要求する要素数積とデータ長を意図的に矛盾させる）。
    //
    // 検出箇所についての注記: safetensors 0.7.0 は `deserialize` 内部の
    // `Metadata::validate`（tensor.rs:595-632）で「各テンソルの
    // `data_offsets` 区間の長さ」と「shape の要素数積 × dtype サイズ」の
    // 整合を**必ず**検査するため、このバイト列は `st_load.rs` の
    // `DataLengthMismatch` 検査（3.）に到達する前に `SafeTensors::
    // deserialize` 自体が `TensorInvalidInfo` として弾く。すなわち
    // 「データ変換より前に長さ不整合を検出する」という安全側の性質は
    // safetensors のフォーマット検証自体によっても保証されており、
    // 本モジュールの `DataLengthMismatch` 検査はその上に重ねた
    // 多層防御（同じ不変条件を自クレート側でも直接検証する。
    // OWASP A03「外部入力を信頼しない」の趣旨）である。
    let good_view_data = [0u8; 8]; // shape [2] の正規データ（f32 2 要素）。
    let view = TensorView::new(Dtype::F32, vec![2], &good_view_data).unwrap();
    let mut bytes = serialize([("w".to_string(), view)], None).unwrap();

    // ヘッダ長（先頭 8 バイト、little-endian u64）を読み取り、その直後の
    // JSON ヘッダ文字列内 "data_offsets":[0,8] を [0,4] に書き換えて
    // 不整合を作り、データ本体も 4 バイトへ切り詰める。
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header_start = 8;
    let header_end = header_start + header_len;
    let header_str = std::str::from_utf8(&bytes[header_start..header_end]).unwrap();
    let patched = header_str.replace("[0,8]", "[0,4]");
    assert_ne!(
        header_str, patched,
        "対象パターンがヘッダ JSON に見つからない"
    );
    bytes.splice(header_start..header_end, patched.into_bytes());
    bytes.truncate(header_end + 4);

    let err = load_safetensors_f32_from_bytes(&bytes).unwrap_err();
    assert!(
        matches!(err, LoadError::SafetensorsFormat(_)),
        "unexpected error variant: {err:?}"
    );
}
