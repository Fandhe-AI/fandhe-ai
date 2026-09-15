//! イシュー #1751: `CastOps`（dtype 変換。6 方向。f64 2 方向は MSL
//! `double` 非対応のため対象外）の CPU-Metal 数値一致検証（CUDA 側
//! #1751 の Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`unique_parity.rs` と同方針。`#![cfg(target_os = "macos")]` に
//! より Linux CI ではコンパイル対象外になり、`#[ignore]` により通常の
//! `cargo test` からも除外される）。
//!
//! **契約は bit 完全一致**（算術を含まない変換のため。NaN のみ
//! payload がプラットフォーム依存のためクラス一致で比較する）。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test cast_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{CastOps, Tensor};

/// f32 fixture（`crates/backend-cuda/tests/cast_parity.rs::f32_fixture`
/// と同じケース集合。両バックエンドで同一 fixture を使うことで
/// 3 バックエンド横断の数値契約が揃っていることを間接的に裏付ける）。
fn f32_fixture() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.5,
        -2.5,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        1e-45,
        2147483648.0,
        2147483520.0,
        -2147483904.0,
        9223372036854775808.0,
        -9223372036854775808.0,
    ]
}

fn assert_f32_bits_eq(a: f32, b: f32, ctx: &str) {
    if a.is_nan() && b.is_nan() {
        return;
    }
    assert_eq!(a.to_bits(), b.to_bits(), "{ctx}: bit 不一致 (a={a}, b={b})");
}

fn run_all_directions(cpu: &CpuBackendOps, metal: &MetalBackendOps) {
    let fixture = f32_fixture();
    let f32_in = Tensor::new(fixture.clone(), &[fixture.len()]).expect("valid tensor");

    // f32 -> i32
    let cpu_i32 = CastOps::cast_f32_to_i32(cpu, &f32_in).expect("cpu f32->i32");
    let metal_i32 = CastOps::cast_f32_to_i32(metal, &f32_in).expect("metal f32->i32");
    assert_eq!(
        metal_i32.as_slice().expect("contiguous"),
        cpu_i32.as_slice().expect("contiguous"),
        "f32->i32 不一致"
    );

    // f32 -> i64
    let cpu_i64 = CastOps::cast_f32_to_i64(cpu, &f32_in).expect("cpu f32->i64");
    let metal_i64 = CastOps::cast_f32_to_i64(metal, &f32_in).expect("metal f32->i64");
    assert_eq!(
        metal_i64.as_slice().expect("contiguous"),
        cpu_i64.as_slice().expect("contiguous"),
        "f32->i64 不一致"
    );

    // f32 -> bool
    let cpu_bool = CastOps::cast_f32_to_bool(cpu, &f32_in).expect("cpu f32->bool");
    let metal_bool = CastOps::cast_f32_to_bool(metal, &f32_in).expect("metal f32->bool");
    assert_eq!(
        metal_bool.as_slice().expect("contiguous"),
        cpu_bool.as_slice().expect("contiguous"),
        "f32->bool 不一致"
    );

    // i32 -> f32
    let i32_in = Tensor::new(
        vec![0, 1, -1, i32::MAX, i32::MIN, 1 << 24, (1 << 24) + 1],
        &[7],
    )
    .expect("valid tensor");
    let cpu_from_i32 = CastOps::cast_i32_to_f32(cpu, &i32_in).expect("cpu i32->f32");
    let metal_from_i32 = CastOps::cast_i32_to_f32(metal, &i32_in).expect("metal i32->f32");
    for (i, (&a, &b)) in metal_from_i32
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(cpu_from_i32.as_slice().expect("contiguous").iter())
        .enumerate()
    {
        assert_f32_bits_eq(a, b, &format!("i32->f32[{i}]"));
    }

    // i64 -> f32（タイケース: 2^24+1・2^25+1・2^25+3 を含む。事前登録
    // 〈`docs/tensor-core-cast-design.md` §11〉: `float(long)` の RNE
    // が実機で不一致だった場合、この方向のみ Metal 側で
    // `Unsupported` へ戻す代替案があるが、本テストはその判定材料と
    // なる）。
    let i64_in = Tensor::new(
        vec![
            0,
            1,
            -1,
            i64::MAX,
            i64::MIN,
            (1i64 << 24) + 1,
            (1i64 << 25) + 1,
            (1i64 << 25) + 3,
        ],
        &[8],
    )
    .expect("valid tensor");
    let cpu_from_i64 = CastOps::cast_i64_to_f32(cpu, &i64_in).expect("cpu i64->f32");
    let metal_from_i64 = CastOps::cast_i64_to_f32(metal, &i64_in).expect("metal i64->f32");
    for (i, (&a, &b)) in metal_from_i64
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(cpu_from_i64.as_slice().expect("contiguous").iter())
        .enumerate()
    {
        assert_f32_bits_eq(a, b, &format!("i64->f32[{i}]"));
    }

    // bool -> f32
    let bool_in = Tensor::new(vec![true, false, true, true, false], &[5]).expect("valid tensor");
    let cpu_from_bool = CastOps::cast_bool_to_f32(cpu, &bool_in).expect("cpu bool->f32");
    let metal_from_bool = CastOps::cast_bool_to_f32(metal, &bool_in).expect("metal bool->f32");
    assert_eq!(
        metal_from_bool.as_slice().expect("contiguous"),
        cpu_from_bool.as_slice().expect("contiguous"),
        "bool->f32 不一致"
    );
}

/// f64 2 方向は MSL `double` 非対応のため Metal 側で恒久
/// `Unsupported` を返す契約を実機で確認する（`docs/tensor-core-
/// cast-design.md` §3.3「Metal は既定 `Unsupported`（オーバーライド
/// しない）」）。
fn assert_f64_directions_are_unsupported(metal: &MetalBackendOps) {
    let f32_in = Tensor::new(vec![1.0f32], &[1]).expect("valid tensor");
    assert!(matches!(
        CastOps::cast_f32_to_f64(metal, &f32_in),
        Err(BackendError::Unsupported(_))
    ));
    let f64_in = Tensor::new(vec![1.0f64], &[1]).expect("valid tensor");
    assert!(matches!(
        CastOps::cast_f64_to_f32(metal, &f64_in),
        Err(BackendError::Unsupported(_))
    ));
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn cast_parity_all_directions() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    run_all_directions(&cpu, &metal);
    assert_f64_directions_are_unsupported(&metal);

    // run-to-run 決定性。
    let fixture = f32_fixture();
    let f32_in = Tensor::new(fixture.clone(), &[fixture.len()]).expect("valid tensor");
    let out1 = CastOps::cast_f32_to_i32(&metal, &f32_in).expect("run1");
    let out2 = CastOps::cast_f32_to_i32(&metal, &f32_in).expect("run2");
    assert_eq!(
        out1.as_slice().expect("contiguous"),
        out2.as_slice().expect("contiguous"),
        "run-to-run で bit 同一のはず"
    );

    // 空テンソル: GPU 起動なしの早期 return 経路。
    let empty = Tensor::new(Vec::<f32>::new(), &[0]).expect("valid tensor");
    let empty_out = CastOps::cast_f32_to_i32(&metal, &empty).expect("metal cast(empty)");
    assert_eq!(empty_out.shape(), &[0]);

    // 非 contiguous view（transpose 済み）も正しく稠密化される。
    let base = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid tensor");
    let transposed = base.permute(&[1, 0]).expect("rank 2 permute always valid");
    let cpu_t = CastOps::cast_f32_to_i32(&cpu, &transposed).expect("cpu transposed");
    let metal_t = CastOps::cast_f32_to_i32(&metal, &transposed).expect("metal transposed");
    assert_eq!(metal_t.shape(), &[3, 2]);
    assert_eq!(
        metal_t.as_slice().expect("contiguous"),
        cpu_t.as_slice().expect("contiguous")
    );
}
