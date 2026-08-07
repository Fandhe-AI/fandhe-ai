//! `onnx-interop::st_save` の統合テスト（TASK-7.1c・#197・REQ-7）。
//!
//! イシュー #197 の受け入れ条件「PyTorch 側で読み込めるファイルが生成される」を、
//! CI（self-hosted・Python/PyTorch 非搭載）上での代理検証として次の 3 点で確認する。
//!
//! 1. PyTorch 生成 fixture（`tests/fixtures/pytorch-reference/model.safetensors`）を
//!    ロード → 本モジュールで再書き出し → 再ロードした際の全キー shape・**bit 一致**
//! 2. 生成バイト列のヘッダ構造（safetensors 仕様＝Python 実装と同一フォーマット）が
//!    `dtype`／`shape`／`data_offsets` を正しく持つこと
//! 3. `safetensors::tensor::SafeTensors::deserialize` による再デシリアライズ成功
//!
//! 実機 PyTorch での確認手順は `crates/onnx-interop/src/st_save.rs` 冒頭
//! ドキュメンテーションコメント「PyTorch 側での手動検証手順」を参照。

use std::collections::HashMap;
use std::path::Path;

use onnx_interop::st_load::load_safetensors_f32;
use onnx_interop::st_save::{save_safetensors_f32, save_safetensors_f32_to_bytes};
use safetensors::tensor::SafeTensors;
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

fn fixture_path(name: &str) -> std::path::PathBuf {
    Path::new(FIXTURE_DIR).join(name)
}

fn load_fixture_weights() -> HashMap<String, Tensor<f32>> {
    load_safetensors_f32(&fixture_path("model.safetensors")).expect("fixture のロードに失敗した")
}

/// `weight_shapes.json`（PyTorch 側が生成した独立の ground truth）から
/// 期待 shape を読み込む。`header_has_expected_dtype_shape_and_data_offsets`
/// が書き出し対象（`original`）からの自己参照ではなく、独立ソースと突き合わせて
/// shape 破壊バグを検知できるようにするため（レビュー指摘対応）。
fn load_expected_shapes() -> HashMap<String, Vec<usize>> {
    let json_str = std::fs::read_to_string(fixture_path("weight_shapes.json"))
        .expect("weight_shapes.json の読み込みに失敗した");
    let value: serde_json::Value =
        serde_json::from_str(&json_str).expect("weight_shapes.json が JSON として解析できない");
    let obj = value
        .as_object()
        .expect("weight_shapes.json がオブジェクトでない");
    obj.iter()
        .map(|(k, v)| {
            let shape: Vec<usize> = v
                .as_array()
                .unwrap_or_else(|| panic!("shape が配列でない (key={k})"))
                .iter()
                .map(|n| n.as_u64().unwrap() as usize)
                .collect();
            (k.clone(), shape)
        })
        .collect()
}

// --- 受け入れ条件（CI 代理検証 1）: PyTorch fixture との bit 一致ラウンドトリップ ---

#[test]
fn round_trips_pytorch_fixture_with_bit_exact_values() {
    let original = load_fixture_weights();

    let bytes = save_safetensors_f32_to_bytes(&original, None).expect("書き出しに失敗した");
    let reloaded =
        onnx_interop::st_load::load_safetensors_f32_from_bytes(&bytes).expect("再ロードに失敗した");

    for key in WEIGHT_KEYS {
        let orig_t = &original[key];
        let reloaded_t = &reloaded[key];
        assert_eq!(
            orig_t.shape(),
            reloaded_t.shape(),
            "shape mismatch for key {key}"
        );

        let orig_data = orig_t.contiguous();
        let reloaded_data = reloaded_t.contiguous();
        let orig_slice = orig_data.as_slice().unwrap();
        let reloaded_slice = reloaded_data.as_slice().unwrap();
        for (i, (a, b)) in orig_slice.iter().zip(reloaded_slice.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "bit mismatch for key {key} at index {i}"
            );
        }
    }
}

// --- 受け入れ条件（CI 代理検証 2）: ヘッダ構造検証（safetensors 仕様＝Python 実装と同一）---

#[test]
fn header_has_expected_dtype_shape_and_data_offsets() {
    let original = load_fixture_weights();
    let bytes = save_safetensors_f32_to_bytes(&original, None).expect("書き出しに失敗した");
    // 期待 shape は書き出し対象（original）自体からではなく、独立の ground truth
    // （PyTorch fixture 生成スクリプトの出力）から取得する。書き出し経路に shape
    // 破壊バグがあってもこのテストで検知できるようにするため（レビュー指摘対応。
    // `round_trips_pytorch_fixture_with_bit_exact_values` は bit 一致を別途検証）。
    let expected_shapes = load_expected_shapes();

    // 先頭 8 バイトは LE u64 のヘッダ長（safetensors 仕様）。
    assert!(bytes.len() >= 8, "バイト列がヘッダ長分すら存在しない");
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header_json = &bytes[8..8 + header_len];
    let header: serde_json::Value =
        serde_json::from_slice(header_json).expect("ヘッダが JSON として解析できない");

    for key in WEIGHT_KEYS {
        let entry = &header[key];
        assert_eq!(entry["dtype"], "F32", "dtype mismatch for key {key}");

        let expected_shape: Vec<usize> = expected_shapes
            .get(key)
            .unwrap_or_else(|| panic!("weight_shapes.json に key {key} が存在しない"))
            .clone();
        let actual_shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap_or_else(|| panic!("shape が配列でない (key={key})"))
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert_eq!(actual_shape, expected_shape, "shape mismatch for key {key}");

        let offsets = entry["data_offsets"].as_array().unwrap();
        let start = offsets[0].as_u64().unwrap();
        let end = offsets[1].as_u64().unwrap();
        let expected_len: u64 = expected_shape.iter().product::<usize>() as u64 * 4;
        assert_eq!(
            end - start,
            expected_len,
            "data_offsets 長が shape の要素数積 x 4 バイトと一致しない (key={key})"
        );
    }

    // safetensors 実装（Python 側と同一フォーマット系）自体でも再デシリアライズできること。
    SafeTensors::deserialize(&bytes).expect("safetensors としての再デシリアライズに失敗した");
}

// --- 決定性: 同一マップは常に同一バイト列を生成する（HashMap 順序非依存の検証） ---

#[test]
fn same_map_produces_identical_bytes_across_multiple_calls() {
    let original = load_fixture_weights();
    let bytes1 = save_safetensors_f32_to_bytes(&original, None).unwrap();
    let bytes2 = save_safetensors_f32_to_bytes(&original, None).unwrap();
    assert_eq!(bytes1, bytes2, "同一入力から異なるバイト列が生成された");
}

// --- 非 contiguous view: transpose_2d() 後の書き出しが論理値を保つ ---

#[test]
fn non_contiguous_transposed_view_round_trips_correctly() {
    let original = load_fixture_weights();
    let fc1_w = original["fc1.weight"].clone();
    let fc1_w_t = fc1_w.transpose_2d().unwrap();
    assert!(
        !fc1_w_t.is_contiguous(),
        "transpose_2d の結果が想定通り non-contiguous であることが前提"
    );

    let mut tensors = HashMap::new();
    tensors.insert("fc1.weight.T".to_string(), fc1_w_t.clone());

    let bytes = save_safetensors_f32_to_bytes(&tensors, None).unwrap();
    let reloaded = onnx_interop::st_load::load_safetensors_f32_from_bytes(&bytes).unwrap();
    let reloaded_t = &reloaded["fc1.weight.T"];

    assert_eq!(reloaded_t.shape(), fc1_w_t.shape());
    let expected = fc1_w_t.contiguous();
    let expected_slice = expected.as_slice().unwrap();
    let actual_slice = reloaded_t.as_slice().unwrap();
    for (a, b) in expected_slice.iter().zip(actual_slice.iter()) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

// --- 空マップ・0 要素テンソル ---

#[test]
fn empty_map_round_trips_without_error() {
    let tensors: HashMap<String, Tensor<f32>> = HashMap::new();
    let bytes = save_safetensors_f32_to_bytes(&tensors, None).unwrap();
    let reloaded = onnx_interop::st_load::load_safetensors_f32_from_bytes(&bytes).unwrap();
    assert!(reloaded.is_empty());
}

#[test]
fn zero_element_tensor_round_trips_without_error() {
    let mut tensors: HashMap<String, Tensor<f32>> = HashMap::new();
    tensors.insert("empty".to_string(), Tensor::new(vec![], &[0]).unwrap());
    let bytes = save_safetensors_f32_to_bytes(&tensors, None).unwrap();
    let reloaded = onnx_interop::st_load::load_safetensors_f32_from_bytes(&bytes).unwrap();
    assert_eq!(reloaded["empty"].shape(), &[0]);
}

// --- ファイル書き出し（save_safetensors_f32）: 一時ファイル + rename の経路 ---

#[test]
fn save_to_file_produces_loadable_file() {
    let original = load_fixture_weights();
    let tmp_dir =
        std::env::temp_dir().join(format!("onnx-interop-st-save-test-{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let out_path = tmp_dir.join("roundtrip.safetensors");

    save_safetensors_f32(&out_path, &original).expect("ファイル書き出しに失敗した");
    assert!(out_path.exists(), "書き出し先ファイルが存在しない");

    let reloaded = load_safetensors_f32(&out_path).expect("書き出したファイルの再ロードに失敗した");
    for key in WEIGHT_KEYS {
        assert_eq!(reloaded[key].shape(), original[key].shape());
    }

    // 一時ファイルが残っていないこと（rename 後に削除されている）。
    let leftover: Vec<_> = std::fs::read_dir(&tmp_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(
        leftover.is_empty(),
        "一時ファイルが残存している: {leftover:?}"
    );

    std::fs::remove_dir_all(&tmp_dir).ok();
}
