//! `st_save` の dtype 混在書出（`save_safetensors_typed*`）の統合テスト
//! （イシュー #3846・`docs/facade-safetensors-exposure-decision.md` §13 案 C 書出側）。
//!
//! ホスト上のバイト変換のみで device 経路を通らないため実機依存テストはない。

mod support;

use std::collections::HashMap;

use fandhe_ai_onnx_interop::st_load::{
    SafetensorsDtype, SafetensorsTensor, load_safetensors_typed, load_safetensors_typed_from_bytes,
};
use fandhe_ai_onnx_interop::st_save::{
    save_safetensors_f32_to_bytes, save_safetensors_typed, save_safetensors_typed_to_bytes,
};
use fandhe_ai_tensor_core::Tensor;
use half::{bf16, f16};

fn mixed() -> HashMap<String, SafetensorsTensor> {
    let mut m = HashMap::new();
    m.insert(
        "f32".into(),
        SafetensorsTensor::F32(Tensor::new(vec![1.5f32, -2.25], &[2]).unwrap()),
    );
    m.insert(
        "f16".into(),
        SafetensorsTensor::F16(
            Tensor::new(vec![f16::from_f32(0.5), f16::from_f32(-3.0)], &[2]).unwrap(),
        ),
    );
    m.insert(
        "bf16".into(),
        SafetensorsTensor::BF16(
            Tensor::new(vec![bf16::from_f32(1.0), bf16::from_f32(-0.125)], &[1, 2]).unwrap(),
        ),
    );
    m.insert(
        "f64".into(),
        SafetensorsTensor::F64(Tensor::new(vec![std::f64::consts::PI, -1e300], &[2]).unwrap()),
    );
    m.insert(
        "i32".into(),
        SafetensorsTensor::I32(Tensor::new(vec![i32::MIN, i32::MAX], &[2]).unwrap()),
    );
    m.insert(
        "i64".into(),
        SafetensorsTensor::I64(Tensor::new(vec![i64::MIN, i64::MAX], &[2]).unwrap()),
    );
    m.insert(
        "bool".into(),
        SafetensorsTensor::Bool(Tensor::new(vec![true, false, true], &[3]).unwrap()),
    );
    m
}

/// 往復後の値を bit 単位で比較できる形（dtype タグ・shape・生ビット列）へ落とす。
fn bits(t: &SafetensorsTensor) -> (SafetensorsDtype, Vec<usize>, Vec<u64>) {
    let v: Vec<u64> = match t {
        SafetensorsTensor::F32(x) => x
            .host_slice()
            .iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        SafetensorsTensor::F64(x) => x.host_slice().iter().map(|v| v.to_bits()).collect(),
        SafetensorsTensor::F16(x) => x
            .host_slice()
            .iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        SafetensorsTensor::BF16(x) => x
            .host_slice()
            .iter()
            .map(|v| u64::from(v.to_bits()))
            .collect(),
        SafetensorsTensor::I32(x) => x.host_slice().iter().map(|v| *v as u64).collect(),
        SafetensorsTensor::I64(x) => x.host_slice().iter().map(|v| *v as u64).collect(),
        SafetensorsTensor::Bool(x) => x.host_slice().iter().map(|v| u64::from(*v)).collect(),
        _ => unreachable!("未知の dtype"),
    };
    (t.dtype(), t.shape().to_vec(), v)
}

#[test]
fn typed_roundtrip_is_bit_identical_per_dtype() {
    let src = mixed();
    let bytes = save_safetensors_typed_to_bytes(&src, None).unwrap();
    let out = load_safetensors_typed_from_bytes(&bytes).unwrap();
    assert_eq!(out.len(), 7);
    for (k, v) in &src {
        assert_eq!(bits(v), bits(&out[k]), "key={k}");
    }
}

#[test]
fn typed_roundtrip_preserves_special_values() {
    let nan32 = f32::from_bits(0x7fc0_1234);
    let nan64 = f64::from_bits(0x7ff8_0000_dead_beef);
    let mut m = HashMap::new();
    m.insert(
        "f32".into(),
        SafetensorsTensor::F32(
            Tensor::new(
                vec![
                    nan32,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    -0.0,
                    f32::from_bits(1),
                ],
                &[5],
            )
            .unwrap(),
        ),
    );
    m.insert(
        "f64".into(),
        SafetensorsTensor::F64(
            Tensor::new(vec![nan64, f64::INFINITY, -0.0, f64::from_bits(1)], &[4]).unwrap(),
        ),
    );
    m.insert(
        "f16".into(),
        SafetensorsTensor::F16(
            Tensor::new(
                vec![
                    f16::from_bits(0x7e01),
                    f16::INFINITY,
                    f16::NEG_ZERO,
                    f16::from_bits(1),
                ],
                &[4],
            )
            .unwrap(),
        ),
    );
    m.insert(
        "bf16".into(),
        SafetensorsTensor::BF16(
            Tensor::new(
                vec![
                    bf16::from_bits(0x7fc1),
                    bf16::NEG_INFINITY,
                    bf16::NEG_ZERO,
                    bf16::from_bits(1),
                ],
                &[4],
            )
            .unwrap(),
        ),
    );
    let bytes = save_safetensors_typed_to_bytes(&m, None).unwrap();
    let out = load_safetensors_typed_from_bytes(&bytes).unwrap();
    for (k, v) in &m {
        assert_eq!(bits(v), bits(&out[k]), "key={k}");
    }
}

#[test]
fn mixed_dtype_file_roundtrip_via_path() {
    let dir = support::temp_dir::TempDirGuard::new("st-typed-save-path");
    let path = dir.path().join("mixed.safetensors");
    let src = mixed();
    save_safetensors_typed(&path, &src).unwrap();
    let out = load_safetensors_typed(&path).unwrap();
    for (k, v) in &src {
        assert_eq!(bits(v), bits(&out[k]), "key={k}");
    }
}

#[test]
fn typed_save_header_dtype_strings_match_pytorch() {
    let src = mixed();
    let bytes = save_safetensors_typed_to_bytes(&src, None).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
    // ヘッダは JSON。キーごとに dtype 文字列が PyTorch 表記で現れることを確かめる。
    for (k, v) in &src {
        let at = header.find(&format!("\"{k}\":")).unwrap();
        let rest = &header[at..];
        let needle = format!("\"dtype\":\"{}\"", v.dtype().header_name());
        let end = rest.find('}').unwrap();
        assert!(rest[..end].contains(&needle), "key={k} header={header}");
    }
}

#[test]
fn typed_save_is_deterministic() {
    let a = save_safetensors_typed_to_bytes(&mixed(), None).unwrap();
    let b = save_safetensors_typed_to_bytes(&mixed(), None).unwrap();
    assert_eq!(a, b);
    // 挿入順の異なるマップでも同一バイト列になる。
    let mut items: Vec<_> = mixed().into_iter().collect();
    items.reverse();
    let rev: HashMap<String, SafetensorsTensor> = items.into_iter().collect();
    assert_eq!(a, save_safetensors_typed_to_bytes(&rev, None).unwrap());
}

#[test]
fn typed_save_handles_non_contiguous() {
    let base = Tensor::new(vec![1i64, 2, 3, 4, 5, 6], &[2, 3]).unwrap();
    let tr = base.transpose_2d().unwrap();
    let mut m = HashMap::new();
    m.insert("t".to_string(), SafetensorsTensor::I64(tr));
    let bytes = save_safetensors_typed_to_bytes(&m, None).unwrap();
    let out = load_safetensors_typed_from_bytes(&bytes).unwrap();
    match &out["t"] {
        SafetensorsTensor::I64(x) => {
            assert_eq!(x.shape(), &[3, 2]);
            assert_eq!(&*x.host_slice(), &[1, 4, 2, 5, 3, 6]);
        }
        other => panic!("unexpected dtype {:?}", other.dtype()),
    }
}

#[test]
fn typed_f32_only_matches_legacy_f32_bytes() {
    let mut typed = HashMap::new();
    let mut legacy = HashMap::new();
    for (k, vals) in [("a", vec![1.0f32, 2.0]), ("b", vec![-0.0, f32::NAN, 3.5])] {
        let t = Tensor::new(vals.clone(), &[vals.len()]).unwrap();
        legacy.insert(k.to_string(), t.clone());
        typed.insert(k.to_string(), SafetensorsTensor::F32(t));
    }
    let md = || Some(HashMap::from([("format".to_string(), "pt".to_string())]));
    let cases = [(HashMap::new(), HashMap::new()), (typed, legacy)];
    for (t, l) in &cases {
        for m in [None, md()] {
            assert_eq!(
                save_safetensors_typed_to_bytes(t, m.clone()).unwrap(),
                save_safetensors_f32_to_bytes(l, m).unwrap()
            );
        }
    }
}
