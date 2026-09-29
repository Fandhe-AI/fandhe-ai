//! `io::npy` の統合テスト（イシュー #2189）。
//!
//! NumPy 2.3.5 実出力の fixture（`tests/fixtures/npy/gen_fixtures.py` で
//! 生成・コミット済み）を読み込み、shape・要素 bit の一致、および
//! 自前 `write_npy_bytes` の出力が `np.save` の C 順 `<f4` 出力と
//! バイト完全一致することを検証する。

mod common;

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::io::NpyError;
use fandhe_ai_tensor_core::io::npy::{load_npy, read_npy_bytes, save_npy, write_npy_bytes};

fn fixture_path(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/npy")
        .join(name)
}

fn fixture_bytes(name: &str) -> Vec<u8> {
    std::fs::read(fixture_path(name)).unwrap_or_else(|e| panic!("fixture {name} を読めない: {e}"))
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.host_slice().iter().map(|v| v.to_bits()).collect()
}

#[test]
fn c_order_2x3_matches_expected_bits_and_round_trips_bytes() {
    let bytes = fixture_bytes("c_order_2x3.npy");
    let t = read_npy_bytes(&bytes).unwrap();
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(
        bits(&t),
        (0..6u32).map(|i| (i as f32).to_bits()).collect::<Vec<_>>()
    );

    // `write_npy_bytes` の出力が `np.save` の実バイト列と完全一致する
    // ことを確認する（R4: NumPy 標準ツールとの bit 同一往復）。
    let written = write_npy_bytes(&t).unwrap();
    assert_eq!(
        written, bytes,
        "write_npy_bytes の出力が np.save のバイト列と一致しない"
    );
}

#[test]
fn rank1_round_trips_bytes() {
    let bytes = fixture_bytes("rank1.npy");
    let t = read_npy_bytes(&bytes).unwrap();
    assert_eq!(t.shape(), &[4]);
    let written = write_npy_bytes(&t).unwrap();
    assert_eq!(written, bytes);
}

#[test]
fn scalar_rank0_round_trips_bytes() {
    let bytes = fixture_bytes("scalar.npy");
    let t = read_npy_bytes(&bytes).unwrap();
    assert_eq!(t.shape(), &[] as &[usize]);
    assert_eq!(bits(&t), vec![3.5f32.to_bits()]);
    let written = write_npy_bytes(&t).unwrap();
    assert_eq!(written, bytes);
}

#[test]
fn empty_0x4_round_trips_bytes() {
    let bytes = fixture_bytes("empty_0x4.npy");
    let t = read_npy_bytes(&bytes).unwrap();
    assert_eq!(t.shape(), &[0, 4]);
    assert!(t.host_slice().is_empty());
    let written = write_npy_bytes(&t).unwrap();
    assert_eq!(written, bytes);
}

#[test]
fn fortran_order_matches_c_order_fixture() {
    let fortran_bytes = fixture_bytes("fortran_2x3.npy");
    let c_bytes = fixture_bytes("c_order_2x3.npy");
    let t_fortran = read_npy_bytes(&fortran_bytes).unwrap();
    let t_c = read_npy_bytes(&c_bytes).unwrap();
    assert_eq!(t_fortran.shape(), t_c.shape());
    assert_eq!(bits(&t_fortran), bits(&t_c));
    // load → save の結果が C 順 fixture とバイト一致する。
    let written = write_npy_bytes(&t_fortran).unwrap();
    assert_eq!(written, c_bytes);
}

#[test]
fn big_endian_matches_expected_values_and_save_produces_little_endian() {
    let bytes = fixture_bytes("big_endian_1d.npy");
    let t = read_npy_bytes(&bytes).unwrap();
    assert_eq!(t.host_slice().to_vec(), vec![1.0, -2.5, 3.25]);
    // 書き出しは常に `<f4`（リトルエンディアン）のため、big-endian の
    // 元 fixture とはバイト列が一致しない（値は一致する）。
    let written = write_npy_bytes(&t).unwrap();
    let back = read_npy_bytes(&written).unwrap();
    assert_eq!(back.host_slice().to_vec(), t.host_slice().to_vec());
}

#[test]
fn special_values_preserve_exact_bits() {
    let bytes = fixture_bytes("special_values.npy");
    let t = read_npy_bytes(&bytes).unwrap();
    let vals = t.host_slice();
    assert!(vals[0].is_nan());
    assert_eq!(vals[1], f32::INFINITY);
    assert_eq!(vals[2], f32::NEG_INFINITY);
    assert!(vals[3].is_sign_negative() && vals[3] == 0.0);
    assert!(vals[4] != 0.0 && vals[4].abs() < f32::MIN_POSITIVE); // 非正規化数
    assert_eq!(vals[5], f32::MAX);
    assert_eq!(vals[6], f32::MIN_POSITIVE);

    let written = write_npy_bytes(&t).unwrap();
    assert_eq!(written, bytes);
}

#[test]
fn nan_payload_variant_preserves_exact_bit_pattern() {
    let bytes = fixture_bytes("nan_payload_variant.npy");
    let t = read_npy_bytes(&bytes).unwrap();
    assert_eq!(bits(&t), vec![0x7FC0_0001u32]);
    let written = write_npy_bytes(&t).unwrap();
    assert_eq!(written, bytes);
}

#[test]
fn v2_header_reads_correctly() {
    let bytes = fixture_bytes("v2_header.npy");
    assert_eq!(bytes[6], 2, "fixture が v2.0 ヘッダであることの前提確認");
    let t = read_npy_bytes(&bytes).unwrap();
    assert_eq!(t.host_slice().to_vec(), vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn rejects_unsupported_dtypes() {
    for name in ["reject_f8.npy", "reject_i4.npy", "reject_f2.npy"] {
        let bytes = fixture_bytes(name);
        let err = read_npy_bytes(&bytes);
        assert!(
            matches!(err, Err(NpyError::UnsupportedDtype { .. })),
            "{name} は UnsupportedDtype で拒否されるべき: {err:?}"
        );
    }
}

#[test]
fn rejects_structured_and_object_dtype() {
    for name in ["reject_structured.npy", "reject_object.npy"] {
        let bytes = fixture_bytes(name);
        let err = read_npy_bytes(&bytes);
        assert!(err.is_err(), "{name} は読み込みエラーになるべき");
    }
}

#[test]
fn rejects_truncated_file() {
    let bytes = fixture_bytes("truncated.npy");
    let err = read_npy_bytes(&bytes);
    assert!(matches!(err, Err(NpyError::DataLengthMismatch { .. })));
}

#[test]
fn rejects_forged_header_length() {
    let bytes = fixture_bytes("forged_header_len.npy");
    let err = read_npy_bytes(&bytes);
    assert!(err.is_err(), "偽装ヘッダ長は拒否されるべき");
}

#[test]
fn non_contiguous_view_is_saved_in_c_order() {
    let t = Tensor::new((0..12u32).map(|i| i as f32).collect(), &[3, 4]).unwrap();
    let transposed = t.transpose_2d().unwrap();
    assert!(!transposed.is_contiguous());
    let bytes = write_npy_bytes(&transposed).unwrap();
    let back = read_npy_bytes(&bytes).unwrap();
    assert_eq!(back.shape(), &[4, 3]);
    assert_eq!(back.host_slice().to_vec(), transposed.host_slice().to_vec());
}

#[test]
fn path_based_round_trip() {
    let dir = common::TempDirGuard::new("path-round-trip");
    let path = dir.path().join("t.npy");
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    save_npy(&t, &path).unwrap();
    let back = load_npy(&path).unwrap();
    assert_eq!(back.shape(), t.shape());
    assert_eq!(back.host_slice().to_vec(), t.host_slice().to_vec());
}

#[test]
fn load_npy_on_missing_path_returns_io_error() {
    let dir = common::TempDirGuard::new("missing-path");
    let path = dir.path().join("does-not-exist.npy");
    let err = load_npy(&path);
    assert!(matches!(err, Err(NpyError::Io(_))));
}
