//! `fandhe_ai::interop::safetensors`（イシュー #2019）の save／load 往復
//! 契約テストを `fandhe_ai`（facade）と `std` のみを import して検証する
//! （`interop_onnx_import.rs` と同じ流儀）。
//!
//! 対象契約（`crates/facade/src/interop/safetensors.rs` モジュール doc
//! 参照）:
//! 1. bytes 往復・`compat::Sequential::state_dict`／`load_state_dict`
//!    往復・ファイル往復のいずれも bit 完全一致（暗黙アダプタなし）
//! 2. 不足キー・dtype 不一致・形状不一致は型付き `Err` で fail-closed に
//!    拒否（`crates/onnx-interop/src/st_load.rs` の REQ-7 契約を迂回・
//!    複製しない）
//! 3. 同一マップから 2 回生成したバイト列は完全一致（決定的出力）

use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::{
    LoadError, SaveError, load_safetensors_f32, load_safetensors_f32_from_bytes, require_keys,
    save_safetensors_f32, save_safetensors_f32_to_bytes,
};
use std::collections::HashMap;

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, /* seed = */ 42)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 43)
        .unwrap()
}

fn sample_input() -> Tensor<f32> {
    Tensor::new(vec![0.1_f32, -0.2, 0.3, -0.4], &[1, 4]).unwrap()
}

/// テストごとに衝突しない一時ディレクトリ（プロセス ID + テスト名）を
/// 作る。docs・ログへ絶対パスを書かない（呼び出し元がパスを表示しない
/// 限り安全）。
fn temp_dir_for(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "fandhe-ai-safetensors-roundtrip-{}-{test_name}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
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

// ---- 正常系 ----

/// `compat::Sequential::state_dict()` → bytes → `load_safetensors_f32_from_bytes`
/// が全キー・全要素 bit 完全一致する（特殊値混じりの手組みマップでも
/// 同様。暗黙アダプタなしの確認は #1752 側で別途検証済みのためここでは
/// safetensors 往復自体の bit 一致に焦点を当てる）。
#[test]
fn state_dict_bytes_roundtrip_is_bit_exact() {
    let model = build_model();
    let sd = model.state_dict();

    let bytes = save_safetensors_f32_to_bytes(&sd, None).unwrap();
    let loaded = load_safetensors_f32_from_bytes(&bytes).unwrap();

    assert_eq!(sd.len(), loaded.len());
    for (key, tensor) in &sd {
        let from_loaded = loaded
            .get(key.as_str())
            .unwrap_or_else(|| panic!("ロード結果に `{key}` が存在しない"));
        assert_tensor_bit_exact(tensor, from_loaded, key);
    }
}

/// 特殊値（NaN・±inf・-0.0・非正規化数）を含む手組みマップでも bytes
/// 往復が bit 完全一致する。
#[test]
fn special_float_values_survive_bytes_roundtrip_bit_exact() {
    let mut map = HashMap::new();
    map.insert(
        "special".to_string(),
        Tensor::new(
            vec![
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                -0.0_f32,
                0.0_f32,
                f32::from_bits(1), // 最小非正規化数
                1.0_f32,
                -1.0_f32,
            ],
            &[8],
        )
        .unwrap(),
    );

    let bytes = save_safetensors_f32_to_bytes(&map, None).unwrap();
    let loaded = load_safetensors_f32_from_bytes(&bytes).unwrap();

    let original = map.get("special").unwrap().as_slice().unwrap();
    let round_tripped = loaded.get("special").unwrap().as_slice().unwrap();
    assert_eq!(original.len(), round_tripped.len());
    for (a, b) in original.iter().zip(round_tripped.iter()) {
        // NaN は to_bits でペイロードまで比較する（fandhe-ai 側は
        // データ変換を単純な byte 列コピーで行うため NaN payload も
        // 保存される契約。`st_load.rs`／`st_save.rs` 参照）。
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// `Sequential` 往復: 異なるシードの同形モデル B へ
/// `load_state_dict(save→load したマップ)` した後、`named_parameters` が
/// A と bit 一致し、同一入力の `predict` 出力も bit 一致する。
#[test]
fn sequential_state_dict_roundtrip_via_safetensors_bytes_matches_predict_output() {
    let a = build_model();
    let mut b = Sequential::new()
        .add_linear(4, 8, /* seed = */ 999)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 1000)
        .unwrap();

    let bytes = save_safetensors_f32_to_bytes(&a.state_dict(), None).unwrap();
    let loaded = load_safetensors_f32_from_bytes(&bytes).unwrap();
    b.load_state_dict(loaded).unwrap();

    let named_a = a.named_parameters();
    let named_b = b.named_parameters();
    assert_eq!(named_a.len(), named_b.len());
    for ((name_a, tensor_a), (name_b, tensor_b)) in named_a.iter().zip(named_b.iter()) {
        assert_eq!(name_a, name_b);
        assert_tensor_bit_exact(tensor_a, tensor_b, name_a);
    }

    let x = sample_input();
    let out_a = a.predict(&x).unwrap();
    let out_b = b.predict(&x).unwrap();
    assert_tensor_bit_exact(&out_a, &out_b, "predict output");
}

/// ファイル往復（`save_safetensors_f32` → `load_safetensors_f32`）が
/// bit 完全一致し、保存後に一時ファイル（`.tmp.` を含むファイル名）が
/// 残らない（一時ファイル + rename 契約の維持確認）。
#[test]
fn file_roundtrip_is_bit_exact_and_leaves_no_tmp_file() {
    let dir = temp_dir_for("file-roundtrip");
    let path = dir.join("weights.safetensors");

    let model = build_model();
    let sd = model.state_dict();
    save_safetensors_f32(&path, &sd).unwrap();

    let loaded = load_safetensors_f32(&path).unwrap();
    assert_eq!(sd.len(), loaded.len());
    for (key, tensor) in &sd {
        assert_tensor_bit_exact(tensor, loaded.get(key.as_str()).unwrap(), key);
    }

    let leftover_tmp: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.contains(".tmp."))
        .collect();
    assert!(
        leftover_tmp.is_empty(),
        "一時ファイルが残存している: {leftover_tmp:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// 同一マップから 2 回生成したバイト列は完全一致する（キー昇順ソート
/// による決定的出力契約）。
#[test]
fn save_to_bytes_is_deterministic() {
    let model = build_model();
    let sd = model.state_dict();

    let bytes1 = save_safetensors_f32_to_bytes(&sd, None).unwrap();
    let bytes2 = save_safetensors_f32_to_bytes(&sd, None).unwrap();
    assert_eq!(bytes1, bytes2);
}

/// 暗黙アダプタなし: 保存 → ロードで `Linear` weight の shape が転置
/// されずそのまま戻る。
#[test]
fn no_implicit_transpose_on_roundtrip() {
    let model = build_model();
    let sd = model.state_dict();
    let weight_key = sd
        .keys()
        .find(|k| k.ends_with(".weight"))
        .cloned()
        .expect("weight キーが見つからない");
    let original_shape = sd.get(&weight_key).unwrap().shape().to_vec();

    let bytes = save_safetensors_f32_to_bytes(&sd, None).unwrap();
    let loaded = load_safetensors_f32_from_bytes(&bytes).unwrap();
    let loaded_shape = loaded.get(weight_key.as_str()).unwrap().shape().to_vec();

    assert_eq!(original_shape, loaded_shape);
}

// ---- 異常系（すべて型付き Err・fail-closed） ----

/// 不足キー: state_dict から 1 キーを除いて保存 → ロード後
/// (a) `require_keys` が `LoadError::MissingKeys` に不足全件を載せて
/// 返す
/// (b) `load_state_dict` が `Err` を返し、モデル B のパラメータが変化
/// していない（strict・アトミック契約）。
#[test]
fn missing_key_is_rejected_with_all_missing_keys_and_load_state_dict_is_atomic() {
    let a = build_model();
    let mut sd = a.state_dict();
    let removed_key = sd
        .keys()
        .find(|k| k.ends_with(".bias"))
        .cloned()
        .expect("bias キーが見つからない");
    sd.remove(&removed_key);

    let bytes = save_safetensors_f32_to_bytes(&sd, None).unwrap();
    let loaded = load_safetensors_f32_from_bytes(&bytes).unwrap();

    // require_keys は文字列スライス参照を取るため、名前を所有した
    // `String` 列を先に確保してから `&str` の集合へ変換する。
    let all_keys_owned: Vec<String> = a
        .named_parameters()
        .into_iter()
        .map(|(name, _)| name)
        .collect();
    let all_keys_ref: Vec<&str> = all_keys_owned.iter().map(String::as_str).collect();

    let err = require_keys(&loaded, &all_keys_ref).unwrap_err();
    match err {
        LoadError::MissingKeys(missing) => {
            assert_eq!(missing, vec![removed_key.clone()]);
        }
        other => panic!("MissingKeys ではない: {other:?}"),
    }

    let mut b = Sequential::new()
        .add_linear(4, 8, /* seed = */ 999)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, /* seed = */ 1000)
        .unwrap();
    let before = b.state_dict();
    let load_result = b.load_state_dict(loaded);
    assert!(
        load_result.is_err(),
        "不足キーがあるのに load_state_dict が成功した"
    );

    let after = b.state_dict();
    assert_eq!(before.len(), after.len());
    for (key, tensor) in &before {
        assert_tensor_bit_exact(tensor, after.get(key.as_str()).unwrap(), key);
    }
}

/// dtype 不一致: `save_safetensors_f32_to_bytes` が出力した F32 ヘッダ
/// の dtype 文字列をバイト単位で `I32` へ置換し（同じ 4 バイト幅のため
/// safetensors 側のレイアウト検査は通過し
/// `LoadError::UnsupportedDtype` に到達する）ロードが型付き `Err` を
/// 返すことを確認する。`safetensors` クレートへの直接依存は追加しない
/// （facade テストは `fandhe_ai`／`std` のみを import する契約）。
#[test]
fn dtype_mismatch_is_rejected_with_unsupported_dtype() {
    let mut map = HashMap::new();
    map.insert(
        "w".to_string(),
        Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap(),
    );
    let mut bytes = save_safetensors_f32_to_bytes(&map, None).unwrap();

    // safetensors ワイヤフォーマットは先頭 8 バイトが LE u64 のヘッダ長。
    // ヘッダ本体（JSON）はその直後に続く。ヘッダ領域内でのみ探索・置換
    // することで、データ本体に偶然 "F32" というバイト列が含まれていて
    // も誤って書き換えない。
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header_start = 8;
    let header_end = header_start + header_len;
    let header = &mut bytes[header_start..header_end];

    let needle = b"F32";
    let pos = header
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("ヘッダに \"F32\" が見つからない");
    header[pos..pos + needle.len()].copy_from_slice(b"I32");

    let err = load_safetensors_f32_from_bytes(&bytes).unwrap_err();
    match err {
        LoadError::UnsupportedDtype { key, dtype } => {
            assert_eq!(key, "w");
            assert!(dtype.contains("I32") || dtype.to_uppercase().contains("I32"));
        }
        other => panic!("UnsupportedDtype ではない: {other:?}"),
    }
}

/// 形状不一致: 形状の異なるモデルの state_dict をロード →
/// `load_state_dict` が `Err`・B のパラメータが変化していない。
#[test]
fn shape_mismatch_is_rejected_and_load_state_dict_is_atomic() {
    let wrong_shape_model = Sequential::new()
        .add_linear(4, 16, /* seed = */ 7)
        .unwrap()
        .add_relu()
        .add_linear(16, 2, /* seed = */ 8)
        .unwrap();
    let bytes = save_safetensors_f32_to_bytes(&wrong_shape_model.state_dict(), None).unwrap();
    let loaded = load_safetensors_f32_from_bytes(&bytes).unwrap();

    let mut b = build_model();
    let before = b.state_dict();
    let result = b.load_state_dict(loaded);
    assert!(
        result.is_err(),
        "形状不一致なのに load_state_dict が成功した"
    );

    let after = b.state_dict();
    for (key, tensor) in &before {
        assert_tensor_bit_exact(tensor, after.get(key.as_str()).unwrap(), key);
    }
}

/// 壊れたバイト列（ランダム・切り詰め）は
/// `LoadError::SafetensorsFormat` を返す。
#[test]
fn corrupted_bytes_are_rejected_with_safetensors_format_error() {
    let garbage = vec![0xffu8; 16];
    let err = load_safetensors_f32_from_bytes(&garbage).unwrap_err();
    assert!(matches!(err, LoadError::SafetensorsFormat(_)));

    let model = build_model();
    let bytes = save_safetensors_f32_to_bytes(&model.state_dict(), None).unwrap();
    let truncated = &bytes[..bytes.len() / 2];
    let err2 = load_safetensors_f32_from_bytes(truncated).unwrap_err();
    assert!(matches!(
        err2,
        LoadError::SafetensorsFormat(_) | LoadError::Io(_)
    ));
}

/// 存在しないパスは `LoadError::Io`。
#[test]
fn nonexistent_path_is_rejected_with_io_error() {
    let dir = temp_dir_for("nonexistent-path");
    let path = dir.join("does-not-exist.safetensors");
    let err = load_safetensors_f32(&path).unwrap_err();
    assert!(matches!(err, LoadError::Io(_)));
    let _ = std::fs::remove_dir_all(&dir);
}

/// `SaveError` 型自体が facade 経由で到達可能であることの空虚 pass
/// 防止（`save_safetensors_f32_to_bytes` の成功系はここまでの他テストで
/// 検証済みのため、ここでは型が使用可能であることのみ確認する）。
#[test]
fn save_error_type_is_usable_via_facade() {
    // SaveError の各 variant を名指しできることをコンパイル時に固定する
    // （実行時アサーションは他テストの異常系で代替済み）。
    let _f: fn(&SaveError) -> &'static str = |e| match e {
        SaveError::Io(_) => "io",
        SaveError::SafetensorsFormat(_) => "safetensors_format",
        SaveError::DataUnavailable { .. } => "data_unavailable",
        _ => "unknown",
    };
}
