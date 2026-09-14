//! イシュー #1740: `BackendOps::cumsum`／`cumprod`（`torch.cumsum`／
//! `torch.cumprod` 相当。親イシュー #1731）の CPU-Metal 数値一致検証
//! （`crates/backend-cuda/tests/scan_parity.rs` の Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`unique_parity.rs` と同方針。`#![cfg(target_os = "macos")]` に
//! より Linux CI ではコンパイル対象外になり、`#[ignore]` により通常の
//! `cargo test` からも除外される）。
//!
//! **契約は bit 同一**（lane ごとの binary64 ソフトウェアエミュレー
//! ションアキュムレータ逐次計算・CPU 参照実装と bit 完全一致。
//! `fandhe_ai_tensor_core::BackendOps::cumsum`／`cumprod` doc 参照）。
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
//! cargo test -p fandhe-ai-backend-metal --release --test scan_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// `cumsum`／`cumprod` の CPU-Metal parity を bit 単位で確認する共通
/// ヘルパー。`shape`・`dim` を指定し、`f(&ops, x, dim)` の形で
/// `BackendOps::cumsum`／`cumprod` を呼ぶクロージャを受け取る
/// （`crates/backend-cuda/tests/scan_parity.rs::assert_scan_parity`
/// と同型）。
fn assert_scan_parity(
    cpu: &CpuBackendOps,
    metal: &MetalBackendOps,
    x: &Tensor<f32>,
    dim: usize,
    f: impl Fn(
        &dyn BackendOps,
        &Tensor<f32>,
        usize,
    ) -> Result<Tensor<f32>, fandhe_ai_tensor_core::device::BackendError>,
    label: &str,
) {
    let cpu_out = f(cpu, x, dim).expect("cpu scan always succeeds");
    let metal_out = f(metal, x, dim).expect("metal scan must succeed on Metal-equipped runner");

    assert_eq!(metal_out.shape(), cpu_out.shape(), "{label}: shape 不一致");
    assert_eq!(bits(&metal_out), bits(&cpu_out), "{label}: bit 不一致");

    // run-to-run 決定性。
    let metal_out2 = f(metal, x, dim).expect("metal scan run2");
    assert_eq!(
        bits(&metal_out2),
        bits(&metal_out),
        "{label}: run-to-run で bit 同一のはず"
    );
}

fn cumsum_fn(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    dim: usize,
) -> Result<Tensor<f32>, fandhe_ai_tensor_core::device::BackendError> {
    ops.cumsum(x, dim)
}

fn cumprod_fn(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    dim: usize,
) -> Result<Tensor<f32>, fandhe_ai_tensor_core::device::BackendError> {
    ops.cumprod(x, dim)
}

/// サイズ網羅（1-D／2-D／3-D 各軸・axis_len=1・transpose view・f64
/// アキュムレータ契約証明ベクトル・特殊値・空入力）を実機で確認する
/// （Apple Silicon。`crates/backend-cuda/tests/scan_parity.rs` の
/// 環境適応スモークと同じケース集合を実機必須形で網羅する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn scan_matches_cpu_across_shapes() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    // 1-D。
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("valid tensor");
    assert_scan_parity(&cpu, &metal, &x, 0, cumsum_fn, "cumsum 1-D");
    assert_scan_parity(&cpu, &metal, &x, 0, cumprod_fn, "cumprod 1-D");

    // 2-D・各軸。
    let x2 = Tensor::new(Xorshift64Star::new(4001).fill_vec(24), &[4usize, 6usize])
        .expect("valid tensor");
    assert_scan_parity(&cpu, &metal, &x2, 0, cumsum_fn, "cumsum 2-D dim0");
    assert_scan_parity(&cpu, &metal, &x2, 1, cumsum_fn, "cumsum 2-D dim1");
    assert_scan_parity(&cpu, &metal, &x2, 0, cumprod_fn, "cumprod 2-D dim0");
    assert_scan_parity(&cpu, &metal, &x2, 1, cumprod_fn, "cumprod 2-D dim1");

    // 3-D・中間軸。
    let x3 = Tensor::new(
        Xorshift64Star::new(4002).fill_vec(60),
        &[3usize, 4usize, 5usize],
    )
    .expect("valid tensor");
    assert_scan_parity(&cpu, &metal, &x3, 1, cumsum_fn, "cumsum 3-D dim1");
    assert_scan_parity(&cpu, &metal, &x3, 1, cumprod_fn, "cumprod 3-D dim1");

    // axis_len=1（scan は恒等写像）。
    let x_axis1 = Tensor::new(vec![5.0, 6.0, 7.0], &[3usize, 1usize]).expect("valid tensor");
    assert_scan_parity(&cpu, &metal, &x_axis1, 1, cumsum_fn, "cumsum axis_len=1");

    // transpose view 入力（非 contiguous）。
    let base = Tensor::new(Xorshift64Star::new(4003).fill_vec(12), &[3usize, 4usize])
        .expect("valid tensor");
    let transposed = base.transpose(0, 1).unwrap();
    assert_scan_parity(&cpu, &metal, &transposed, 0, cumsum_fn, "cumsum transposed");
    assert_scan_parity(
        &cpu,
        &metal,
        &transposed,
        1,
        cumprod_fn,
        "cumprod transposed",
    );

    // 契約証明ベクトル（f64 相当アキュムレータでなければ成立しない値）。
    let big = Tensor::new(vec![1e8, 1.0, -1e8], &[3]).expect("valid tensor");
    let cpu_big = cpu.cumsum(&big, 0).expect("cpu cumsum");
    let metal_big = metal.cumsum(&big, 0).expect("metal cumsum");
    assert_eq!(bits(&metal_big), bits(&cpu_big));
    let metal_big_vals = metal_big.contiguous().as_slice().unwrap().to_vec();
    assert_eq!(metal_big_vals, vec![1e8, 1e8, 1.0]);

    let tiny = Tensor::new(vec![1e-30, 1e-30, 1e30], &[3]).expect("valid tensor");
    let cpu_tiny = cpu.cumprod(&tiny, 0).expect("cpu cumprod");
    let metal_tiny = metal.cumprod(&tiny, 0).expect("metal cumprod");
    assert_eq!(bits(&metal_tiny), bits(&cpu_tiny));
    let metal_tiny_vals = metal_tiny.contiguous().as_slice().unwrap().to_vec();
    assert_ne!(
        metal_tiny_vals[2], 0.0,
        "f64 相当アキュムレータなら underflow しないはず"
    );

    // -0.0 先頭。
    let neg_zero = Tensor::new(vec![-0.0, 1.0], &[2]).expect("valid tensor");
    assert_scan_parity(&cpu, &metal, &neg_zero, 0, cumsum_fn, "cumsum -0.0 leading");

    // inf + -inf → NaN、0 * inf → NaN（NaN クラス一致で確認。
    // `soft_f64` の NaN 正規化方針は `scan_model` の単体テストで
    // 機械的に裏付け済み）。
    let inf_pair = Tensor::new(vec![f32::INFINITY, f32::NEG_INFINITY], &[2]).expect("valid tensor");
    let metal_inf_sum = metal.cumsum(&inf_pair, 0).expect("metal cumsum");
    let metal_inf_vals = metal_inf_sum.contiguous().as_slice().unwrap().to_vec();
    assert!(metal_inf_vals[1].is_nan(), "inf + -inf は NaN のはず");

    let zero_inf = Tensor::new(vec![0.0, f32::INFINITY], &[2]).expect("valid tensor");
    let metal_zero_inf = metal.cumprod(&zero_inf, 0).expect("metal cumprod");
    let metal_zero_inf_vals = metal_zero_inf.contiguous().as_slice().unwrap().to_vec();
    assert!(metal_zero_inf_vals[1].is_nan(), "0 * inf は NaN のはず");

    // 空出力。
    let empty = Tensor::new(Vec::new(), &[0usize, 3usize]).expect("valid tensor");
    let cpu_empty = cpu.cumsum(&empty, 0).expect("cpu cumsum(empty)");
    let metal_empty = metal.cumsum(&empty, 0).expect("metal cumsum(empty)");
    assert_eq!(cpu_empty.shape(), metal_empty.shape());
}

/// 大きめ lane 数／axis 長を実機で確認する（Apple Silicon。
/// `#[ignore]`。`crates/backend-cuda/tests/scan_parity.rs::
/// scan_matches_cpu_for_large_shapes` の Metal 対応版）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn scan_matches_cpu_for_large_shapes() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let x = Tensor::new(
        Xorshift64Star::new(5001).fill_vec(2000 * 300),
        &[2000usize, 300usize],
    )
    .expect("valid tensor");
    assert_scan_parity(&cpu, &metal, &x, 0, cumsum_fn, "cumsum large dim0");
    assert_scan_parity(&cpu, &metal, &x, 1, cumsum_fn, "cumsum large dim1");
    assert_scan_parity(&cpu, &metal, &x, 1, cumprod_fn, "cumprod large dim1");

    let x1d = Tensor::new(Xorshift64Star::new(5002).fill_vec(1 << 16), &[1usize << 16])
        .expect("valid tensor");
    assert_scan_parity(&cpu, &metal, &x1d, 0, cumsum_fn, "cumsum large 1-D");
}
