//! `fandhe_ai::interop::npy`（イシュー #2590・親 #2588）の npy／npz 往復
//! 契約テストを `fandhe_ai`（facade）と `std` のみを import して検証する
//! （`interop_safetensors_roundtrip.rs` と同じ流儀）。
//!
//! 対象契約（`crates/facade/src/interop/npy.rs` モジュール doc 参照）:
//! 1. npy／npz のファイル往復は shape・要素とも bit 完全一致
//!    （NaN・±0・subnormal を含む）
//! 2. `compat::Sequential::state_dict` をそのまま `save_npz` に渡せ、
//!    `load_npz` → `load_state_dict` で出力が bit 一致する
//! 3. 存在しないパス・不正 magic・非対応 dtype は型付き `Err` で fail-closed
//!    （検証は `tensor-core` 実装を継承し、facade で迂回・複製しない）

mod common;

use common::temp_dir::TempDirGuard;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::npy::{NpyError, load_npy, load_npz, save_npy, save_npz};
use std::collections::HashMap;

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.as_slice()
        .expect("ホスト常駐テンソル")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn assert_bit_exact(a: &Tensor<f32>, b: &Tensor<f32>, label: &str) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape 不一致");
    assert_eq!(bits(a), bits(b), "{label}: 要素が bit 一致しない");
}

fn special_values() -> Vec<f32> {
    vec![
        f32::NAN,
        0.0,
        -0.0,
        f32::MIN_POSITIVE / 4.0, // subnormal
        f32::INFINITY,
        f32::NEG_INFINITY,
        1.5,
        -2.25,
    ]
}

#[test]
fn npy_roundtrip_is_bit_exact_for_rank_0_1_2() {
    let dir = TempDirGuard::new("npy_roundtrip");
    let cases = [
        ("rank0", Tensor::new(vec![3.5_f32], &[]).unwrap()),
        ("rank1", Tensor::new(special_values(), &[8]).unwrap()),
        ("rank2", Tensor::new(special_values(), &[2, 4]).unwrap()),
    ];
    for (label, t) in &cases {
        let path = dir.path().join(format!("{label}.npy"));
        save_npy(t, &path).unwrap();
        let back = load_npy(&path).unwrap();
        assert_bit_exact(t, &back, label);
    }
}

#[test]
fn npz_roundtrip_preserves_keys_shapes_and_bits() {
    let dir = TempDirGuard::new("npz_roundtrip");
    let mut map: HashMap<String, Tensor<f32>> = HashMap::new();
    map.insert("a".into(), Tensor::new(special_values(), &[2, 4]).unwrap());
    map.insert(
        "b.weight".into(),
        Tensor::new(vec![1.0_f32, 2.0], &[2]).unwrap(),
    );
    map.insert("scalar".into(), Tensor::new(vec![-7.0_f32], &[]).unwrap());

    let path = dir.path().join("m.npz");
    save_npz(&map, &path).unwrap();
    let back = load_npz(&path).unwrap();

    let mut want: Vec<&String> = map.keys().collect();
    let mut got: Vec<&String> = back.keys().collect();
    want.sort();
    got.sort();
    assert_eq!(want, got, "キー集合が一致しない");
    for (k, v) in &map {
        assert_bit_exact(v, &back[k], k);
    }

    // 決定的出力: 同一マップから 2 回書いたファイルは完全一致。
    let path2 = dir.path().join("m2.npz");
    save_npz(&map, &path2).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&path2).unwrap()
    );
}

#[test]
fn sequential_state_dict_roundtrips_through_npz() {
    let build = |s1, s2| {
        Sequential::new()
            .add_linear(4, 8, s1)
            .unwrap()
            .add_relu()
            .add_linear(8, 2, s2)
            .unwrap()
    };
    let model = build(42, 43);
    let dir = TempDirGuard::new("npz_state_dict");
    let path = dir.path().join("state.npz");

    save_npz(&model.state_dict(), &path).unwrap();
    let loaded = load_npz(&path).unwrap();

    let mut other = build(1, 2);
    other.load_state_dict(loaded).unwrap();

    let x = Tensor::new(vec![0.1_f32, -0.2, 0.3, -0.4], &[1, 4]).unwrap();
    let a = model.predict(&x).unwrap();
    let b = other.predict(&x).unwrap();
    assert_bit_exact(&a, &b, "predict");
}

#[test]
fn missing_path_is_io_error() {
    let dir = TempDirGuard::new("npy_missing");
    let path = dir.path().join("absent.npy");
    assert!(matches!(load_npy(&path), Err(NpyError::Io(_))));
    assert!(matches!(
        load_npz(dir.path().join("absent.npz")),
        Err(NpyError::Io(_))
    ));
}

#[test]
fn invalid_magic_is_rejected() {
    let dir = TempDirGuard::new("npy_magic");
    let path = dir.path().join("bad.npy");
    std::fs::write(&path, b"not a npy file at all").unwrap();
    let err = load_npy(&path).unwrap_err();
    assert!(matches!(err, NpyError::InvalidMagic), "{err:?}");
}

#[test]
fn non_f32_dtype_is_rejected() {
    // v1.0 ヘッダ（総長を 64 の倍数に揃える）で `<f8` を宣言した 1 要素の npy。
    let mut header = String::from("{'descr': '<f8', 'fortran_order': False, 'shape': (1,), }");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&1.0_f64.to_le_bytes());

    let dir = TempDirGuard::new("npy_f8");
    let path = dir.path().join("f8.npy");
    std::fs::write(&path, &bytes).unwrap();
    let err = load_npy(&path).unwrap_err();
    assert!(
        matches!(err, NpyError::UnsupportedDtype { .. }),
        "非対応 dtype は UnsupportedDtype のはず: {err:?}"
    );
}
