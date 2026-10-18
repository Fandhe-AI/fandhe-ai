//! ROCm 実機（MI300X・gfx942）での f32 GEMM を CPU 参照と突き合わせる実機 parity テスト（#3159）。
//!
//! - 役割: `RocmGemmF32::run` の出力を `fandhe_ai_backend_cpu::compare`（REQ-2 統一複合判定。
//!   相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。定数は変更しない）で判定する。
//! - 判定対象は REQ-2 の複合判定のみ。CPU との bit 不一致数と GPU 同士の決定性は
//!   観測値として出力するだけで assert しない（HIP-Clang の contraction 挙動は
//!   `docs/backend-rocm-hiprtc-gemm-decision.md` §2 の方針どおり実機で観測する）。
//! - 呼び出し文脈: CI では実行しない（`#[ignore]`）。事前登録の規則と実行手順は
//!   `docs/perf/logs/rocm-parity-3159/`（`RULE.txt`・`run_rocm_parity.sh`）を正とする。
//! - 形状は `RULE.txt` の `shapes` と同じ並びに保つこと。

use fandhe_ai_backend_cpu::{compare, gemm_naive};
use fandhe_ai_backend_rocm::{GfxTarget, RocmGemmF32};

/// `RULE.txt` の `shapes` と同一の並び。タイルは BM=BN=64・BK=16（`GemmF32Tile::DEFAULT`）。
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 1, 0),
    (64, 64, 16),
    (63, 65, 17),
    (130, 70, 33),
    (5, 129, 200),
    (128, 128, 64),
    (257, 255, 513),
    (128, 128, 32),
    (65, 64, 16),
    (64, 63, 16),
    (64, 64, 15),
    (64, 64, 17),
    (1, 257, 33),
    (257, 1, 33),
    (193, 129, 48),
    (64, 64, 4096),
    (33, 17, 4097),
    (128, 96, 8192),
    (0, 5, 3),
    (5, 0, 3),
];

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

fn bit_mismatches(x: &[f32], y: &[f32]) -> usize {
    x.iter()
        .zip(y)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count()
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 で実行）"]
fn gemm_f32_matches_cpu_reference() {
    let g = RocmGemmF32::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    let mut failing = Vec::new();
    for (i, &(m, n, k)) in SHAPES.iter().enumerate() {
        let a = data(m * k, 1 + i as u64);
        let b = data(k * n, 100 + i as u64);
        let mut expected = vec![0.0f32; m * n];
        gemm_naive(&a, &b, &mut expected, m, n, k).unwrap();
        let got = g
            .run(&a, &b, m, n, k)
            .unwrap_or_else(|e| panic!("run {m}x{n}x{k}: {e:?}"));
        let r = compare(&got, &expected).unwrap_or_else(|e| panic!("compare {m}x{n}x{k}: {e:?}"));
        let bits = bit_mismatches(&got, &expected);
        // 形状ごとの記録行（README の表へ転記する。ループ内では panic せず最後にまとめて落とす）。
        eprintln!(
            "ROCM_PARITY\t{m}x{n}x{k}\ttotal={}\tfail={}\tmax_abs={:e}\tmax_rel={:e}\tmean_abs={:e}\tp99_abs={:e}\tbit_mismatch={bits}",
            r.total, r.fail_count, r.max_abs_diff, r.max_rel_err, r.mean_abs_diff, r.p99_abs_diff
        );
        if r.fail_count > 0 {
            failing.push(format!("{m}x{n}x{k}(fail={}/{})", r.fail_count, r.total));
        }
    }
    assert!(
        failing.is_empty(),
        "REQ-2 統一複合判定に不合格の形状: {failing:?}"
    );
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 で実行）"]
fn gemm_f32_repeat_is_bit_identical_record_only() {
    let g = RocmGemmF32::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    // 決定性は記録だけ（assert しない）。同一プロセス内で 2 回実行して bit 比較する。
    for (m, n, k) in [(257usize, 255usize, 513usize), (128, 96, 8192)] {
        let a = data(m * k, 7);
        let b = data(k * n, 8);
        let r1 = g.run(&a, &b, m, n, k).expect("run 1");
        let r2 = g.run(&a, &b, m, n, k).expect("run 2");
        eprintln!(
            "ROCM_DETERMINISM\t{m}x{n}x{k}\tbit_mismatch={}",
            bit_mismatches(&r1, &r2)
        );
    }
}
