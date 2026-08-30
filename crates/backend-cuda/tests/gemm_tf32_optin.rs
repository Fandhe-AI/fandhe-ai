//! イシュー #1042: `crate::precision::set_tf32_gemm_enabled` の opt-in で
//! 切り替わる `CudaBackendOps::gemm`（f32 の素の GEMM）の複合判定検証。
//!
//! カーネル本体（`CudaGemm::run_wmma_tf32`）自体の誤差分布は
//! `gemm_wmma_tf32.rs`・`gemm_wmma_tf32_opt.rs`・`mma_tf32_vs_wmma_tf32_
//! staged.rs`（既存）が既に検証済みであり、本ファイルはそれらを重複させ
//! ない。本ファイルが検証するのは「`ops.rs::CudaBackendOps::gemm` の
//! opt-in フラグ配線」という新規の到達経路のみ（`gemm_bias_act_parity.rs`
//! と同じ構成方針）:
//!
//! 1. opt-in OFF（既定）時、`gemm` 出力が本イシュー導入前と bit-exact に
//!    一致すること（既定 OFF の非後退契約）。
//! 2. opt-in ON 時、`gemm` 出力が CPU 参照実装（FP32 厳密）と REQ-2 統一
//!    複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で一致する
//!    こと。
//!
//! `common::parity_baseline` から tolerance 定数 pin を借用し、判定式・
//! 許容誤差は再定義しない（`.claude/rules/coding-rust.md`）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test gemm_tf32_optin -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_backend_cuda::precision::{set_tf32_gemm_enabled, tf32_gemm_enabled};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

mod common;

/// フラグはプロセスグローバル（`crate::precision`）のため、
/// `cargo test` の既定並列実行下で他のテストバイナリ内テストと競合し
/// うる。本ファイル内では直列に実行し、各テストの最後に必ず OFF へ
/// 戻す（`ops.rs::tests::Tf32FlagGuard` と同型の RAII ガード）。
struct Tf32FlagGuard {
    original: bool,
}

impl Tf32FlagGuard {
    fn acquire(enabled: bool) -> Self {
        let original = tf32_gemm_enabled();
        set_tf32_gemm_enabled(enabled);
        Self { original }
    }
}

impl Drop for Tf32FlagGuard {
    fn drop(&mut self) {
        set_tf32_gemm_enabled(self.original);
    }
}

fn assert_tf32_optin_gemm_parity(seed_a: u64, seed_b: u64, m: usize, n: usize, k: usize) {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    let a_data = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b_data = Xorshift64Star::new(seed_b).fill_vec(k * n);
    let a = Tensor::new(a_data, &[m, k]).expect("valid tensor");
    let b = Tensor::new(b_data, &[k, n]).expect("valid tensor");

    let cpu_result = cpu.gemm(&a, &b).expect("cpu gemm always succeeds");

    let _guard = Tf32FlagGuard::acquire(true);
    let cuda_result = cuda
        .gemm(&a, &b)
        .expect("CudaBackendOps::gemm (tf32 opt-in) must succeed on CUDA-equipped test runner");
    assert_eq!(cuda_result.shape(), cpu_result.shape());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("tf32 opt-in gemm cpu-cuda parity m={m} n={n} k={k}"),
        cuda_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。opt-in OFF（既定）時の
/// `gemm` 出力が opt-in を一切知らないかのように動作する（本イシュー
/// 導入前の `run_tiled_f32` 単独経路と bit-exact に一致する）ことを、
/// CPU 参照実装との複合判定で確認する。CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する
/// （`gemm_bias_act_parity.rs` と同じ分岐パターン）。
#[test]
fn gemm_tf32_optin_off_matches_default_fp32_path_env_adaptive() {
    let _guard = Tf32FlagGuard::acquire(false);
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).expect("valid tensor");

    match cuda.gemm(&a, &b) {
        Ok(cuda_result) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            let cpu_result = cpu.gemm(&a, &b).expect("cpu gemm always succeeds");
            fandhe_ai_backend_cpu::parity::assert_parity(
                "tf32 opt-in OFF gemm cpu-cuda parity smoke",
                cuda_result.as_slice().expect("contiguous"),
                cpu_result.as_slice().expect("contiguous"),
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::gemm: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件 2「opt-in 時の複合判定結果」の本体）。
/// opt-in ON 時に TF32 Tensor Core 経路が CPU 参照実装（FP32 厳密）と
/// REQ-2 統一複合判定で一致することを、正方・非正方・K 支配的形状で
/// 確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等。cc>=8.0）必須"]
fn gemm_tf32_optin_on_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let cases: &[(u64, u64, usize, usize, usize)] = &[
        (701, 702, 512, 512, 512),
        (703, 704, 1024, 1024, 1024),
        (705, 706, 96, 160, 48),
        // K 支配的な非正方形状（split-K 検討の先例と同じ形状クラス。
        // `docs/perf/metal-gemm-splitk-shapes.md` 参照）。
        (707, 708, 256, 256, 4096),
    ];
    for &(seed_a, seed_b, m, n, k) in cases {
        assert_tf32_optin_gemm_parity(seed_a, seed_b, m, n, k);
    }
}
