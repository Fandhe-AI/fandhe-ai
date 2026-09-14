//! イシュー #1734: `BackendOps::unique`（`torch.unique(input,
//! sorted=True)` の values のみ）の CPU-Metal 数値一致検証
//! （CUDA 側 #1734 の Metal 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`gather_scatter_parity.rs` と同方針。`#![cfg(target_os =
//! "macos")]` により Linux CI ではコンパイル対象外になり、`#[ignore]`
//! により通常の `cargo test` からも除外される）。
//!
//! unique は選択演算（丸めを伴わない）のため CPU-Metal 間で
//! **bit 完全一致**を検証する。
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
//! cargo test -p fandhe-ai-backend-metal --release --test unique_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn assert_unique_parity(seed: u64, numel: usize) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let x = Tensor::new(Xorshift64Star::new(seed).fill_vec(numel), &[numel]).expect("valid tensor");

    let cpu_out = cpu.unique(&x).expect("cpu unique always succeeds");
    let metal_out = metal
        .unique(&x)
        .expect("metal unique must succeed on Metal-equipped runner");

    let cpu_slice = cpu_out.as_slice().expect("contiguous");
    let metal_slice = metal_out.as_slice().expect("contiguous");
    assert_eq!(
        metal_slice.len(),
        cpu_slice.len(),
        "unique: 要素数が一致しない (numel={numel})"
    );
    for (i, (&a, &b)) in metal_slice.iter().zip(cpu_slice.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "unique: 要素 {i} が bit 一致しない (metal={a}, cpu={b}, numel={numel})"
        );
    }

    let metal_out2 = metal.unique(&x).expect("metal unique run2");
    assert_eq!(
        metal_out2.as_slice().expect("contiguous"),
        metal_slice,
        "unique: run-to-run で bit 同一のはず"
    );
}

/// サイズ網羅（2 のべき乗境界・非 2 のべき乗・大きめサイズ）・全同一値・
/// ±0／NaN 混在・空入力・単一要素を実機で確認する（Apple Silicon）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn unique_matches_cpu_across_shapes() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    for (i, &n) in [1usize, 2, 3, 4, 7, 8, 16, 255, 256, 257, 4097, 1 << 16]
        .iter()
        .enumerate()
    {
        assert_unique_parity(30000 + i as u64, n);
    }

    // 全同一値。
    let all_same = Tensor::new(vec![7.0; 50], &[50]).expect("valid tensor");
    let cpu_out = cpu.unique(&all_same).expect("cpu unique");
    let metal_out = metal.unique(&all_same).expect("metal unique");
    assert_eq!(
        metal_out.as_slice().expect("contiguous"),
        cpu_out.as_slice().expect("contiguous")
    );
    assert_eq!(metal_out.shape(), &[1]);

    // ±0／NaN 混在。
    let nan1 = f32::NAN;
    let nan2 = f32::from_bits(f32::NAN.to_bits() | 1);
    let special = Tensor::new(
        vec![0.0, -0.0, 1.0, nan1, nan2, f32::INFINITY, f32::NEG_INFINITY],
        &[7],
    )
    .expect("valid tensor");
    let cpu_special = cpu.unique(&special).expect("cpu unique");
    let metal_special = metal.unique(&special).expect("metal unique");
    let cpu_bits: Vec<u32> = cpu_special
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let metal_bits: Vec<u32> = metal_special
        .as_slice()
        .expect("contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert_eq!(metal_bits, cpu_bits);

    // 空入力。
    let empty = Tensor::new(Vec::new(), &[0]).expect("valid tensor");
    let cpu_empty = cpu.unique(&empty).expect("cpu unique(empty)");
    let metal_empty = metal.unique(&empty).expect("metal unique(empty)");
    assert_eq!(cpu_empty.shape(), &[0]);
    assert_eq!(metal_empty.shape(), &[0]);

    // 単一要素（GPU dispatch なしの早期 return 経路）。
    let single = Tensor::new(vec![42.0], &[1]).expect("valid tensor");
    let metal_single = metal.unique(&single).expect("metal unique(single)");
    assert_eq!(metal_single.as_slice().expect("contiguous"), &[42.0]);
}
