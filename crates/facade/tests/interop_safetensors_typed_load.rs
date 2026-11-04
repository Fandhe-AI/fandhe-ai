//! facade 経由の safetensors dtype 混在読込（`load_safetensors_typed*`）の受入テスト
//! （イシュー #3845・`docs/facade-safetensors-exposure-decision.md` §13 読込側）。
//!
//! facade は `safetensors`／`half` に直接依存しないため、safetensors 形式
//! （ヘッダ長 u64 LE + ヘッダ JSON + データ）を手で組む。書出しを含む往復は #3846。

use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::{
    LoadError, SafetensorsDtype, SafetensorsTensor, load_safetensors_f32_from_bytes,
    load_safetensors_typed, load_safetensors_typed_from_bytes, require_keys_typed,
};

/// `(name, dtype 文字列, shape, データ)` から safetensors バイト列を組む（連続配置）。
fn build(items: &[(&str, &str, &[usize], Vec<u8>)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut data = Vec::new();
    for (i, (name, dtype, shape, bytes)) in items.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let shape_s: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            shape_s.join(","),
            data.len(),
            data.len() + bytes.len()
        ));
        data.extend_from_slice(bytes);
    }
    header.push('}');
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}

fn bf16_bits(v: f32) -> [u8; 2] {
    // bf16 は f32 の上位 16 bit（テスト値は丸め不要な値のみ使う）。
    ((v.to_bits() >> 16) as u16).to_le_bytes()
}

#[test]
fn mixed_dtype_file_loads_via_facade() {
    let bytes = build(&[
        (
            "a",
            "F64",
            &[2],
            [1.5f64, -2.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        ("b", "I64", &[1], 7i64.to_le_bytes().to_vec()),
        ("c", "BOOL", &[3], vec![1, 0, 1]),
        (
            "d",
            "BF16",
            &[2],
            [bf16_bits(0.5), bf16_bits(-4.0)].concat(),
        ),
    ]);
    let m = load_safetensors_typed_from_bytes(&bytes).unwrap();
    require_keys_typed(&m, &["a", "b", "c", "d"]).unwrap();
    match &m["a"] {
        SafetensorsTensor::F64(t) => assert_eq!(&*t.host_slice(), &[1.5, -2.0]),
        o => panic!("{o:?}"),
    }
    assert!(matches!(&m["b"], SafetensorsTensor::I64(t) if t.host_slice()[0] == 7));
    assert!(
        matches!(&m["c"], SafetensorsTensor::Bool(t) if *t.host_slice() == [true, false, true])
    );
    assert_eq!(m["d"].dtype(), SafetensorsDtype::BF16);
    assert_eq!(&*m["d"].to_f32().unwrap().host_slice(), &[0.5, -4.0]);

    // 既存の f32 ローダーは非 F32 を拒否したまま。
    assert!(matches!(
        load_safetensors_f32_from_bytes(&bytes).unwrap_err(),
        LoadError::UnsupportedDtype { .. }
    ));
}

#[test]
fn bf16_checkpoint_reaches_sequential_load_state_dict() {
    let model = Sequential::new().add_linear(2, 2, 1).unwrap();
    let state = model.state_dict();
    let mut items: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (k, t) in state.iter() {
        let b: Vec<u8> = t.host_slice().iter().flat_map(|v| bf16_bits(*v)).collect();
        items.push((k.clone(), t.shape().to_vec(), b));
    }
    let refs: Vec<(&str, &str, &[usize], Vec<u8>)> = items
        .iter()
        .map(|(k, s, b)| (k.as_str(), "BF16", s.as_slice(), b.clone()))
        .collect();
    let bytes = build(&refs);

    let typed = load_safetensors_typed_from_bytes(&bytes).unwrap();
    let mut f32_map = std::collections::HashMap::new();
    for (k, t) in &typed {
        assert_eq!(t.dtype(), SafetensorsDtype::BF16);
        f32_map.insert(k.clone(), t.to_f32().unwrap());
    }
    let mut other = Sequential::new().add_linear(2, 2, 2).unwrap();
    other.load_state_dict(f32_map).unwrap();
}

#[test]
fn invalid_bool_unsupported_dtype_and_missing_keys_are_rejected() {
    let bytes = build(&[("m", "BOOL", &[2], vec![1, 2])]);
    assert!(matches!(
        load_safetensors_typed_from_bytes(&bytes).unwrap_err(),
        LoadError::InvalidBoolValue {
            index: 1,
            value: 2,
            ..
        }
    ));
    let bytes = build(&[("u", "U8", &[1], vec![0])]);
    assert!(matches!(
        load_safetensors_typed_from_bytes(&bytes).unwrap_err(),
        LoadError::UnsupportedDtype { .. }
    ));
    let m = load_safetensors_typed_from_bytes(&build(&[("x", "I32", &[1], vec![0; 4])])).unwrap();
    match require_keys_typed(&m, &["y", "x", "z"]).unwrap_err() {
        LoadError::MissingKeys(k) => assert_eq!(k, vec!["y".to_string(), "z".to_string()]),
        e => panic!("{e:?}"),
    }
}

#[test]
fn missing_path_is_io_error() {
    let p = std::path::Path::new("/nonexistent-dir-3845/none.safetensors");
    assert!(matches!(
        load_safetensors_typed(p).unwrap_err(),
        LoadError::Io(_)
    ));
}
