//! ROCm 実機（MI300X・gfx942）での f32 GEMM parity。#3159 で実機実行する（CI では実行しない）。
//! REQ-2 の統一複合判定のみ assert し、bit 一致は観測値として出力するだけにする
//! （HIP-Clang の contraction 挙動は実機で未確認のため）。

use fandhe_ai_backend_cpu::{assert_parity, gemm_naive};
use fandhe_ai_backend_rocm::{GfxTarget, RocmGemmF32};

fn data(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 で実行）"]
fn gemm_f32_matches_cpu_reference() {
    let g = RocmGemmF32::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    let shapes = [
        (1, 1, 1),
        (1, 1, 0),
        (64, 64, 16),
        (63, 65, 17),
        (130, 70, 33),
        (5, 129, 200),
        (128, 128, 64),
        (257, 255, 513),
    ];
    for (i, &(m, n, k)) in shapes.iter().enumerate() {
        let a = data(m * k, 1 + i as u64);
        let b = data(k * n, 100 + i as u64);
        let mut expected = vec![0.0f32; m * n];
        gemm_naive(&a, &b, &mut expected, m, n, k).unwrap();
        let got = g.run(&a, &b, m, n, k).expect("run");
        assert_parity(&format!("rocm {m}x{n}x{k}"), &got, &expected);
        let mismatches = got
            .iter()
            .zip(&expected)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        eprintln!("{m}x{n}x{k}: bit 不一致 {mismatches}/{}", m * n);
    }
}
