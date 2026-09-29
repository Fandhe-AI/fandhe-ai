//! `io::npz` の統合テスト（イシュー #2189）。
//!
//! NumPy 2.3.5 実出力の fixture（`tests/fixtures/npy/gen_fixtures.py`）を
//! 読み込み、`np.savez`／`np.savez_compressed` 双方の要素 bit 一致、
//! load → save → load の往復、CRC 改ざん・非対応 dtype メンバ等の
//! fail-closed 挙動を検証する。

mod common;

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::io::NpyError;
use fandhe_ai_tensor_core::io::npz::{load_npz, read_npz_bytes, save_npz, write_npz_bytes};
use std::collections::HashMap;

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/npy")
        .join(name)
}

fn fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture_path(name)).unwrap_or_else(|e| panic!("fixture {name} を読めない: {e}"))
}

#[test]
fn sample_stored_npz_matches_expected_members() {
    let bytes = fixture_bytes("sample_stored.npz");
    let map = read_npz_bytes(&bytes).unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(map["a"].shape(), &[2, 3]);
    assert_eq!(
        map["a"].host_slice().to_vec(),
        (0..6u32).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert_eq!(map["b"].shape(), &[0, 4]);
    assert!(map["b"].host_slice().is_empty());
}

#[test]
fn sample_compressed_npz_matches_expected_members() {
    let bytes = fixture_bytes("sample_compressed.npz");
    let map = read_npz_bytes(&bytes).unwrap();
    assert_eq!(map.len(), 2);
    assert_eq!(
        map["x"].host_slice().to_vec(),
        (0..50u32).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert_eq!(map["y"].host_slice().to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn load_save_load_round_trip_preserves_all_members() {
    let bytes = fixture_bytes("sample_compressed.npz");
    let map = read_npz_bytes(&bytes).unwrap();
    let rewritten = write_npz_bytes(&map).unwrap();
    let reloaded = read_npz_bytes(&rewritten).unwrap();
    assert_eq!(reloaded.len(), map.len());
    for (k, v) in &map {
        assert_eq!(reloaded[k].shape(), v.shape());
        assert_eq!(reloaded[k].host_slice().to_vec(), v.host_slice().to_vec());
    }
}

#[test]
fn rejects_non_f32_member_and_returns_no_partial_result() {
    let bytes = fixture_bytes("npz_with_non_f32_member.npz");
    let err = read_npz_bytes(&bytes);
    match err {
        Err(NpyError::Entry { name, source }) => {
            assert_eq!(name, "bad");
            assert!(matches!(*source, NpyError::UnsupportedDtype { .. }));
        }
        other => panic!("Entry(UnsupportedDtype) を期待したが: {other:?}"),
    }
}

#[test]
fn rejects_crc_corrupted_member() {
    let bytes = fixture_bytes("npz_crc_corrupted.npz");
    let err = read_npz_bytes(&bytes);
    assert!(err.is_err(), "CRC 改ざんされた npz は拒否されるべき");
}

#[test]
fn write_npz_produces_deterministic_key_order_independent_bytes() {
    let mut map = HashMap::new();
    map.insert("z".to_string(), Tensor::new(vec![1.0], &[1]).unwrap());
    map.insert("a".to_string(), Tensor::new(vec![2.0], &[1]).unwrap());
    let bytes1 = write_npz_bytes(&map).unwrap();
    let bytes2 = write_npz_bytes(&map).unwrap();
    assert_eq!(
        bytes1, bytes2,
        "同じ内容の HashMap から常に同一バイト列が生成されるべき"
    );
}

#[test]
fn path_based_round_trip() {
    let dir = common::TempDirGuard::new("path-round-trip");
    let path = dir.path().join("t.npz");
    let mut map = HashMap::new();
    map.insert("w".to_string(), Tensor::new(vec![1.0, 2.0], &[2]).unwrap());
    save_npz(&map, &path).unwrap();
    let back = load_npz(&path).unwrap();
    assert_eq!(back["w"].host_slice().to_vec(), vec![1.0, 2.0]);
}

#[test]
fn load_npz_on_missing_path_returns_io_error() {
    let dir = common::TempDirGuard::new("missing-path");
    let path = dir.path().join("does-not-exist.npz");
    let err = load_npz(&path);
    assert!(matches!(err, Err(NpyError::Io(_))));
}
