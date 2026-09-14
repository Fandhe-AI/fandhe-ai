//! イシュー #1734: `BackendOps::unique`（`torch.unique(input,
//! sorted=True)` の values のみ）の CPU-CUDA 数値一致検証。
//!
//! `gather_scatter_parity.rs` と同じ構成方針を踏襲する: 環境適応
//! スモーク（属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `BackendError::CudaUnavailable` を確認して panic しないことのみ
//! 検証）と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を
//! 分離する。
//!
//! **契約は bit 同一**（選択演算のため丸めを伴わない。
//! `fandhe_ai_tensor_core::BackendOps::unique` doc 参照）: CPU 参照
//! 実装（[`fandhe_ai_backend_cpu::CpuBackendOps`]）と CUDA 実装の出力は
//! `len` と各要素の `to_bits()` が完全一致する契約。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test unique_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn assert_unique_parity(seed: u64, numel: usize) {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let x = Tensor::new(Xorshift64Star::new(seed).fill_vec(numel), &[numel]).expect("valid tensor");

    let cpu_out = cpu.unique(&x).expect("cpu unique always succeeds");
    let cuda_out = cuda
        .unique(&x)
        .expect("cuda unique must succeed on CUDA-equipped test runner");

    let cpu_slice = cpu_out.as_slice().expect("contiguous");
    let cuda_slice = cuda_out.as_slice().expect("contiguous");
    assert_eq!(
        cuda_slice.len(),
        cpu_slice.len(),
        "unique: 要素数が一致しない (numel={numel})"
    );
    for (i, (&a, &b)) in cuda_slice.iter().zip(cpu_slice.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "unique: 要素 {i} が bit 一致しない (cuda={a}, cpu={b}, numel={numel})"
        );
    }

    // run-to-run 決定性: 同一入力で 2 回起動しても bit 同一。
    let cuda_out2 = cuda.unique(&x).expect("cuda unique run2");
    assert_eq!(
        cuda_out2.as_slice().expect("contiguous"),
        cuda_slice,
        "unique: run-to-run で bit 同一のはず"
    );
}

#[test]
fn unique_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    let x = Tensor::new(vec![3.0, 1.0, 2.0, 1.0, 3.0], &[5]).expect("valid tensor");

    match cuda.unique(&x) {
        Ok(_) => {
            assert_unique_parity(9101, 5);
            assert_unique_parity(9102, 100);
            assert_unique_parity(9103, 1000);

            // 全同一値: 重複除去が正しく 1 要素へ集約されることを実機で
            // 確認する。
            let all_same = Tensor::new(vec![7.0; 50], &[50]).expect("valid tensor");
            let cpu_out = cpu.unique(&all_same).expect("cpu unique");
            let cuda_out = cuda.unique(&all_same).expect("cuda unique");
            assert_eq!(
                cuda_out.as_slice().expect("contiguous"),
                cpu_out.as_slice().expect("contiguous")
            );
            assert_eq!(cuda_out.shape(), &[1]);

            // ±0／NaN 混在: totalOrder 契約（`-0.0`／`+0.0` の集約・NaN
            // 全保持）を実機で確認する。
            let nan1 = f32::NAN;
            let nan2 = f32::from_bits(f32::NAN.to_bits() | 1);
            let special = Tensor::new(
                vec![0.0, -0.0, 1.0, nan1, nan2, f32::INFINITY, f32::NEG_INFINITY],
                &[7],
            )
            .expect("valid tensor");
            let cpu_special = cpu.unique(&special).expect("cpu unique");
            let cuda_special = cuda.unique(&special).expect("cuda unique");
            let cpu_bits: Vec<u32> = cpu_special
                .as_slice()
                .expect("contiguous")
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let cuda_bits: Vec<u32> = cuda_special
                .as_slice()
                .expect("contiguous")
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(cuda_bits, cpu_bits);

            // 空入力: shape `[0]` を CPU・CUDA とも返す契約。
            let empty = Tensor::new(Vec::new(), &[0]).expect("valid tensor");
            let cpu_empty = cpu.unique(&empty).expect("cpu unique(empty)");
            let cuda_empty = cuda.unique(&empty).expect("cuda unique(empty)");
            assert_eq!(cpu_empty.shape(), &[0]);
            assert_eq!(cuda_empty.shape(), &[0]);

            // 単一要素: GPU 起動なしの早期 return 経路。
            let single = Tensor::new(vec![42.0], &[1]).expect("valid tensor");
            let cuda_single = cuda.unique(&single).expect("cuda unique(single)");
            assert_eq!(cuda_single.as_slice().expect("contiguous"), &[42.0]);
        }
        Err(fandhe_ai_tensor_core::device::BackendError::CudaUnavailable(_)) => {
            // CUDA 非搭載環境（通常 CI）。panic せず終了する
            // （`gather_scatter_parity.rs` と同じ環境適応方針）。
        }
        Err(other) => panic!("unexpected error on CUDA-equipped runner: {other}"),
    }
}

/// サイズ網羅（2 のべき乗境界・非 2 のべき乗・大きめサイズ）を実機で
/// 確認する（DGX Spark GB10 等。`#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn unique_matches_cpu_across_sizes() {
    for (i, &n) in [1usize, 2, 3, 4, 7, 8, 16, 255, 256, 257, 4097, 1 << 16]
        .iter()
        .enumerate()
    {
        assert_unique_parity(20000 + i as u64, n);
    }
}
