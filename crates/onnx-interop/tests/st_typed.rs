//! `st_load` の dtype 混在読込（`*_typed`）の統合テスト（イシュー #3845・
//! `docs/facade-safetensors-exposure-decision.md` §13 案 C 読込側）。
//!
//! 非 f32 のバイト列は `safetensors::tensor::serialize` で組む。ホスト上の
//! バイト変換のみで device 経路を通らないため実機依存テストはない。

use std::collections::HashMap;

use fandhe_ai_onnx_interop::st_load::{
    LoadError, SafetensorsDtype, SafetensorsTensor, load_safetensors_f32_from_bytes,
    load_safetensors_typed, load_safetensors_typed_from_bytes, require_keys_typed,
};
use fandhe_ai_tensor_core::Tensor;
use half::{bf16, f16};
use safetensors::Dtype;
use safetensors::tensor::{TensorView, serialize};

fn ser(items: Vec<(&str, Dtype, Vec<usize>, Vec<u8>)>) -> Vec<u8> {
    let views: Vec<(String, TensorView<'_>)> = items
        .iter()
        .map(|(n, d, s, b)| {
            (
                n.to_string(),
                TensorView::new(*d, s.clone(), b.as_slice()).unwrap(),
            )
        })
        .collect();
    serialize(views, None).unwrap()
}

fn le<T: Copy, const N: usize>(v: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    v.iter().flat_map(|x| f(*x)).collect()
}

#[test]
fn typed_load_is_bit_identical_per_dtype() {
    let f32s = [1.5f32, -2.25];
    let f16s = [f16::from_f32(0.5), f16::from_f32(-3.0)];
    let bf16s = [bf16::from_f32(1.0), bf16::from_f32(-0.125)];
    let f64s = [std::f64::consts::PI, -1e300];
    let i32s = [i32::MIN, i32::MAX];
    let i64s = [i64::MIN, i64::MAX];
    let bytes = ser(vec![
        ("f32", Dtype::F32, vec![2], le(&f32s, f32::to_le_bytes)),
        (
            "f16",
            Dtype::F16,
            vec![2],
            le(&f16s, |v| v.to_bits().to_le_bytes()),
        ),
        (
            "bf16",
            Dtype::BF16,
            vec![2],
            le(&bf16s, |v| v.to_bits().to_le_bytes()),
        ),
        ("f64", Dtype::F64, vec![2], le(&f64s, f64::to_le_bytes)),
        ("i32", Dtype::I32, vec![2], le(&i32s, i32::to_le_bytes)),
        ("i64", Dtype::I64, vec![2], le(&i64s, i64::to_le_bytes)),
        ("bool", Dtype::BOOL, vec![2], vec![0, 1]),
    ]);
    let m = load_safetensors_typed_from_bytes(&bytes).unwrap();
    assert_eq!(m.len(), 7);
    match &m["f32"] {
        SafetensorsTensor::F32(t) => assert_eq!(&*t.host_slice(), &f32s),
        o => panic!("{o:?}"),
    }
    match &m["f16"] {
        SafetensorsTensor::F16(t) => assert_eq!(&*t.host_slice(), &f16s),
        o => panic!("{o:?}"),
    }
    match &m["bf16"] {
        SafetensorsTensor::BF16(t) => assert_eq!(&*t.host_slice(), &bf16s),
        o => panic!("{o:?}"),
    }
    match &m["f64"] {
        SafetensorsTensor::F64(t) => assert_eq!(&*t.host_slice(), &f64s),
        o => panic!("{o:?}"),
    }
    match &m["i32"] {
        SafetensorsTensor::I32(t) => assert_eq!(&*t.host_slice(), &i32s),
        o => panic!("{o:?}"),
    }
    match &m["i64"] {
        SafetensorsTensor::I64(t) => assert_eq!(&*t.host_slice(), &i64s),
        o => panic!("{o:?}"),
    }
    match &m["bool"] {
        SafetensorsTensor::Bool(t) => assert_eq!(&*t.host_slice(), &[false, true]),
        o => panic!("{o:?}"),
    }
    for (k, d) in [
        ("f32", SafetensorsDtype::F32),
        ("f16", SafetensorsDtype::F16),
        ("bf16", SafetensorsDtype::BF16),
        ("f64", SafetensorsDtype::F64),
        ("i32", SafetensorsDtype::I32),
        ("i64", SafetensorsDtype::I64),
        ("bool", SafetensorsDtype::Bool),
    ] {
        assert_eq!(m[k].dtype(), d);
        assert_eq!(m[k].shape(), &[2]);
    }
}

#[test]
fn typed_load_preserves_special_values() {
    let f32_bits = [
        0x7fc0_1234u32,
        0x7f80_0000,
        0xff80_0000,
        0x8000_0000,
        0x0000_0001,
    ];
    let f64_bits = [
        0x7ff8_0000_0000_1234u64,
        0x7ff0_0000_0000_0000,
        0x8000_0000_0000_0000,
        1,
    ];
    let f16_bits = [0x7e01u16, 0x7c00, 0xfc00, 0x8000, 0x0001];
    let bf16_bits = [0x7fc1u16, 0x7f80, 0xff80, 0x8000, 0x0001];
    let bytes = ser(vec![
        ("a", Dtype::F32, vec![5], le(&f32_bits, u32::to_le_bytes)),
        ("b", Dtype::F64, vec![4], le(&f64_bits, u64::to_le_bytes)),
        ("c", Dtype::F16, vec![5], le(&f16_bits, u16::to_le_bytes)),
        ("d", Dtype::BF16, vec![5], le(&bf16_bits, u16::to_le_bytes)),
        ("scalar", Dtype::F32, vec![], 7.0f32.to_le_bytes().to_vec()),
        ("empty", Dtype::I64, vec![0, 3], vec![]),
    ]);
    let m = load_safetensors_typed_from_bytes(&bytes).unwrap();
    let SafetensorsTensor::F32(a) = &m["a"] else {
        panic!()
    };
    let got: Vec<u32> = a.host_slice().iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, f32_bits);
    let SafetensorsTensor::F64(b) = &m["b"] else {
        panic!()
    };
    let got: Vec<u64> = b.host_slice().iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, f64_bits);
    let SafetensorsTensor::F16(c) = &m["c"] else {
        panic!()
    };
    let got: Vec<u16> = c.host_slice().iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, f16_bits);
    let SafetensorsTensor::BF16(d) = &m["d"] else {
        panic!()
    };
    let got: Vec<u16> = d.host_slice().iter().map(|v| v.to_bits()).collect();
    assert_eq!(got, bf16_bits);
    assert_eq!(m["scalar"].shape(), &[] as &[usize]);
    assert_eq!(m["empty"].shape(), &[0, 3]);
}

#[test]
fn mixed_dtype_file_loads_in_one_pass() {
    let w = le(&[1.0f32, 2.0, 3.0, 4.0], f32::to_le_bytes);
    let bf = le(&[bf16::from_f32(0.5)], |v| v.to_bits().to_le_bytes());
    let steps = le(&[42i64], i64::to_le_bytes);
    let bytes = ser(vec![
        ("fc.weight", Dtype::F32, vec![2, 2], w),
        ("emb", Dtype::BF16, vec![1], bf),
        ("num_batches", Dtype::I64, vec![], steps),
        ("mask", Dtype::BOOL, vec![3], vec![1, 0, 1]),
    ]);
    let m = load_safetensors_typed_from_bytes(&bytes).unwrap();
    require_keys_typed(&m, &["fc.weight", "emb", "num_batches", "mask"]).unwrap();
    assert_eq!(m["fc.weight"].dtype(), SafetensorsDtype::F32);
    assert_eq!(m["emb"].dtype(), SafetensorsDtype::BF16);
    assert_eq!(m["num_batches"].dtype(), SafetensorsDtype::I64);
    assert_eq!(m["mask"].dtype(), SafetensorsDtype::Bool);
}

#[test]
fn header_name_matches_wire_dtype_string() {
    let cases = [
        (Dtype::F32, SafetensorsDtype::F32, 4),
        (Dtype::F16, SafetensorsDtype::F16, 2),
        (Dtype::BF16, SafetensorsDtype::BF16, 2),
        (Dtype::F64, SafetensorsDtype::F64, 8),
        (Dtype::I32, SafetensorsDtype::I32, 4),
        (Dtype::I64, SafetensorsDtype::I64, 8),
        (Dtype::BOOL, SafetensorsDtype::Bool, 1),
    ];
    for (wire, tag, width) in cases {
        let bytes = ser(vec![("t", wire, vec![1], vec![0u8; width])]);
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
        let needle = format!("\"dtype\":\"{}\"", tag.header_name());
        assert!(header.contains(&needle), "{header} / {needle}");
    }
}

#[test]
fn unsupported_dtypes_are_rejected_fail_closed() {
    for (d, width) in [
        (Dtype::U8, 1),
        (Dtype::I8, 1),
        (Dtype::U16, 2),
        (Dtype::F8_E4M3, 1),
        (Dtype::C64, 8),
        (Dtype::U64, 8),
    ] {
        let bytes = ser(vec![("t", d, vec![1], vec![0u8; width])]);
        let err = load_safetensors_typed_from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, LoadError::UnsupportedDtype { .. }), "{d:?}");
    }
}

#[test]
fn invalid_bool_byte_is_rejected() {
    let bytes = ser(vec![("m", Dtype::BOOL, vec![3], vec![0, 1, 2])]);
    match load_safetensors_typed_from_bytes(&bytes).unwrap_err() {
        LoadError::InvalidBoolValue { key, index, value } => {
            assert_eq!((key.as_str(), index, value), ("m", 2, 2));
        }
        e => panic!("{e:?}"),
    }
    let bytes = ser(vec![("m", Dtype::BOOL, vec![4], vec![0, 7, 9, 1])]);
    match load_safetensors_typed_from_bytes(&bytes).unwrap_err() {
        LoadError::InvalidBoolValue { index, value, .. } => assert_eq!((index, value), (1, 7)),
        e => panic!("{e:?}"),
    }
}

#[test]
fn typed_f32_arm_matches_legacy_loader_bits() {
    let vals = [f32::from_bits(0x7fc0_0001), -0.0, 1.0e-40, 3.5];
    let bytes = ser(vec![(
        "w",
        Dtype::F32,
        vec![2, 2],
        le(&vals, f32::to_le_bytes),
    )]);
    let legacy = load_safetensors_f32_from_bytes(&bytes).unwrap();
    let typed = load_safetensors_typed_from_bytes(&bytes).unwrap();
    let SafetensorsTensor::F32(t) = &typed["w"] else {
        panic!()
    };
    let a: Vec<u32> = legacy["w"]
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let b: Vec<u32> = t.host_slice().iter().map(|v| v.to_bits()).collect();
    assert_eq!(a, b);
    assert_eq!(legacy["w"].shape(), t.shape());
}

#[test]
fn legacy_f32_loader_still_rejects_non_f32() {
    let bytes = ser(vec![("t", Dtype::I64, vec![1], vec![0u8; 8])]);
    let err = load_safetensors_f32_from_bytes(&bytes).unwrap_err();
    assert!(matches!(err, LoadError::UnsupportedDtype { .. }));
}

#[test]
fn require_keys_typed_reports_all_missing() {
    let m: HashMap<String, SafetensorsTensor> = HashMap::new();
    match require_keys_typed(&m, &["b", "a"]).unwrap_err() {
        LoadError::MissingKeys(k) => assert_eq!(k, vec!["b".to_string(), "a".to_string()]),
        e => panic!("{e:?}"),
    }
}

#[test]
fn explicit_conversion_follows_cast_contract() {
    // f16 / bf16 -> f32 は完全表現。
    let h = Tensor::new(vec![f16::from_f32(0.1), f16::from_f32(-65504.0)], &[2]).unwrap();
    let f = SafetensorsTensor::F16(h.clone()).to_f32().unwrap();
    assert_eq!(f.host_slice()[0], f16::from_f32(0.1).to_f32());
    assert_eq!(f.host_slice()[1], -65504.0);
    let b = Tensor::new(vec![bf16::from_f32(1.5)], &[1]).unwrap();
    assert_eq!(
        SafetensorsTensor::BF16(b).to_f32().unwrap().host_slice()[0],
        1.5
    );

    // f32 -> f16 は最近接偶数丸め、範囲超過は inf。
    let src = Tensor::new(vec![0.1f32, 1.0e6, -1.0e6], &[3]).unwrap();
    let SafetensorsTensor::F16(h) =
        SafetensorsTensor::from_f32(&src, SafetensorsDtype::F16).unwrap()
    else {
        panic!()
    };
    assert_eq!(
        &*h.host_slice(),
        &[f16::from_f32(0.1), f16::INFINITY, f16::NEG_INFINITY]
    );
    let SafetensorsTensor::BF16(b) =
        SafetensorsTensor::from_f32(&src, SafetensorsDtype::BF16).unwrap()
    else {
        panic!()
    };
    assert_eq!(b.host_slice()[0], bf16::from_f32(0.1));

    // i64 / f64 -> f32 は非可逆の丸め（cast_to_f32 と一致）。
    let i = Tensor::new(vec![16_777_217i64], &[1]).unwrap();
    assert_eq!(
        SafetensorsTensor::I64(i).to_f32().unwrap().host_slice()[0],
        16_777_217i64 as f32
    );
    let d = Tensor::new(vec![0.1f64], &[1]).unwrap();
    assert_eq!(
        SafetensorsTensor::F64(d).to_f32().unwrap().host_slice()[0],
        0.1f64 as f32
    );

    // bool は 0/1、f32 -> bool は v != 0.0（NaN は true）。
    let bl = Tensor::new(vec![true, false], &[2]).unwrap();
    assert_eq!(
        &*SafetensorsTensor::Bool(bl).to_f32().unwrap().host_slice(),
        &[1.0, 0.0]
    );
    let s = Tensor::new(vec![0.0f32, -0.0, 2.0, f32::NAN], &[4]).unwrap();
    let SafetensorsTensor::Bool(bl) =
        SafetensorsTensor::from_f32(&s, SafetensorsDtype::Bool).unwrap()
    else {
        panic!()
    };
    assert_eq!(&*bl.host_slice(), &[false, false, true, true]);

    // F32 は恒等。
    let SafetensorsTensor::F32(id) =
        SafetensorsTensor::from_f32(&src, SafetensorsDtype::F32).unwrap()
    else {
        panic!()
    };
    assert_eq!(&*id.host_slice(), &*src.host_slice());
}

#[test]
fn corrupt_and_empty_bytes_are_typed_errors() {
    assert!(matches!(
        load_safetensors_typed_from_bytes(&[]).unwrap_err(),
        LoadError::SafetensorsFormat(_)
    ));
    assert!(matches!(
        load_safetensors_typed_from_bytes(&[0xff; 16]).unwrap_err(),
        LoadError::SafetensorsFormat(_)
    ));
    let mut bytes = ser(vec![("t", Dtype::F64, vec![2], vec![0u8; 16])]);
    bytes.truncate(bytes.len() - 3);
    assert!(load_safetensors_typed_from_bytes(&bytes).is_err());
}

#[test]
fn missing_path_is_io_error() {
    let p = std::path::Path::new("/nonexistent-dir-3845/none.safetensors");
    assert!(matches!(
        load_safetensors_typed(p).unwrap_err(),
        LoadError::Io(_)
    ));
}
