//! facade 経由の dtype 混在書出（`save_safetensors_typed*`）の受入テスト
//! （イシュー #3846・`docs/facade-safetensors-exposure-decision.md` §13 / §15）。
//!
//! import は `fandhe_ai` と `std` のみ（内部クレートに依存しない）。ホスト上の
//! バイト変換のみで device 経路を通らないため実機依存テストはない。

use std::collections::HashMap;

use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::safetensors::{
    SafetensorsDtype, SafetensorsTensor, load_safetensors_typed, load_safetensors_typed_from_bytes,
    save_safetensors_f32_to_bytes, save_safetensors_typed, save_safetensors_typed_to_bytes,
};

fn mixed() -> HashMap<String, SafetensorsTensor> {
    let mut m = HashMap::new();
    m.insert(
        "w".to_string(),
        SafetensorsTensor::F64(Tensor::new(vec![1.5f64, -2.0], &[2]).unwrap()),
    );
    m.insert(
        "step".to_string(),
        SafetensorsTensor::I64(Tensor::new(vec![i64::MAX], &[1]).unwrap()),
    );
    m.insert(
        "mask".to_string(),
        SafetensorsTensor::Bool(Tensor::new(vec![true, false, true], &[3]).unwrap()),
    );
    m
}

fn check_mixed(out: &HashMap<String, SafetensorsTensor>) {
    assert_eq!(out.len(), 3);
    match (&out["w"], &out["step"], &out["mask"]) {
        (SafetensorsTensor::F64(w), SafetensorsTensor::I64(s), SafetensorsTensor::Bool(m)) => {
            assert_eq!(w.host_slice().to_vec(), vec![1.5, -2.0]);
            assert_eq!(s.host_slice().to_vec(), vec![i64::MAX]);
            assert_eq!(m.host_slice().to_vec(), vec![true, false, true]);
        }
        _ => panic!("dtype が保たれていない"),
    }
}

#[test]
fn typed_bytes_roundtrip_via_facade() {
    let bytes = save_safetensors_typed_to_bytes(&mixed(), None).unwrap();
    check_mixed(&load_safetensors_typed_from_bytes(&bytes).unwrap());
}

#[test]
fn typed_file_roundtrip_via_facade() {
    let dir = std::env::temp_dir().join(format!("fandhe-ai-st-typed-rt-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mixed.safetensors");
    let saved = save_safetensors_typed(&path, &mixed());
    let out = load_safetensors_typed(&path);
    std::fs::remove_dir_all(&dir).unwrap();
    saved.unwrap();
    check_mixed(&out.unwrap());
}

#[test]
fn bf16_checkpoint_written_via_facade_reaches_load_state_dict() {
    let model = Sequential::new().add_linear(3, 5, 11).unwrap();
    let typed: HashMap<String, SafetensorsTensor> = model
        .state_dict()
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                SafetensorsTensor::from_f32(v, SafetensorsDtype::BF16).unwrap(),
            )
        })
        .collect();
    let bytes = save_safetensors_typed_to_bytes(&typed, None).unwrap();
    let loaded = load_safetensors_typed_from_bytes(&bytes).unwrap();
    let mut f32s = HashMap::new();
    for (k, v) in &loaded {
        assert_eq!(v.dtype(), SafetensorsDtype::BF16);
        f32s.insert(k.clone(), v.to_f32().unwrap());
    }
    let mut other = Sequential::new().add_linear(3, 5, 99).unwrap();
    other.load_state_dict(f32s).unwrap();
}

#[test]
fn typed_f32_only_bytes_equal_legacy_via_facade() {
    let t = Tensor::new(vec![1.0f32, -0.0, 3.5], &[3]).unwrap();
    let legacy = HashMap::from([("x".to_string(), t.clone())]);
    let typed = HashMap::from([("x".to_string(), SafetensorsTensor::F32(t))]);
    assert_eq!(
        save_safetensors_typed_to_bytes(&typed, None).unwrap(),
        save_safetensors_f32_to_bytes(&legacy, None).unwrap()
    );
}
