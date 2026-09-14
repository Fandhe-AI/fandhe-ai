//! `CudaBackendOps::gemm_batched`／`gemm_batched_fp32_strict`（バッチ
//! 行列積の CUDA デバイス常駐バッチループ経路。イシュー #1716・親
//! #1715／#1600）の受け入れ条件対応テスト。
//!
//! `gemm_transposed_parity.rs`（#1214）・`gather_scatter_parity.rs`
//! （#1777）と同じ構成方針: 環境適応スモーク（属性なし。CUDA 非搭載
//! 環境では `BackendError::CudaUnavailable` を確認して早期 return する
//! のみ）と、実機必須の形状網羅（`#[ignore]`）を分離する。
//!
//! 数値契約: `run_tiled_f32_batched` の新設オーバーライドは、既定合成
//! 実装（per-batch `gemm_fp32_strict_impl` を呼ぶ
//! `fandhe_ai_tensor_core::gemm_batched_via_per_batch_gemm_fp32_strict`。
//! オーバーライド導入前の CUDA 既定経路そのもの）と **bit 完全一致**
//! する契約（`crates/backend-cuda/src/gemm.rs::run_tiled_f32_batched`
//! ドキュメンテーションコメント「数値契約（bit 同一）」参照）。CPU
//! 参照実装（[`fandhe_ai_backend_cpu::CpuBackendOps::gemm_batched`]）
//! とは REQ-2 統一複合判定（[`fandhe_ai_backend_cpu::parity::assert_parity`]）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test gemm_batched_parity -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, gemm_batched_via_per_batch_gemm_fp32_strict};

/// 実機依存テストのため決定的な軽量疑似乱数生成（外部依存追加を避ける。
/// `.claude/rules/deps-policy.md`）。`gemm_transposed_parity.rs` と同じ
/// xorshift 系の最小実装。
fn random_vec(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state % 2000) as f32 - 1000.0) / 1000.0
        })
        .collect()
}

fn tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    Tensor::new(random_vec(seed, numel), shape).expect("valid tensor")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

/// byte 単位の完全一致判定用（PR #1841 codex-review 指摘対応）。
/// `assert_eq!(Vec<f32>, ...)` は `f32` の `PartialEq` を経由するため
/// `+0.0` と `-0.0` を同一視してしまい、文書で謳う bit 完全一致
/// （`docs/perf/logs/cuda-gemm-batched-1716/README.md`「数値契約」）を
/// 検証できない。`f32::to_bits` へ変換してから比較することで符号ビット
/// を含む厳密な byte 単位比較にする。
fn bits_vec(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// 事前登録した形状網羅セット（`docs/perf/logs/cuda-gemm-batched-1716/
/// README.md`「事前登録判定規則」の「全 9 形状」。整列〈N=K=1024 の
/// 128×64 スロット到達条件を含む〉・非整列・broadcast（lhs／rhs／
/// 中間軸）・rank 4・退化形状〈k==0／m==0〉）を、bit 同一テスト
/// （`gemm_batched_fp32_strict_matches_default_per_batch_composition_bit_exact_on_real_device`）
/// と CPU 参照実装 parity テスト（`gemm_batched_matches_cpu_reference_on_real_device`）
/// の両方から共有する（PR #1841 codex-review 指摘: 両テストの形状
/// セットが乖離していると N=K=1024 のような別カーネル選択経路に
/// 到達する形状で CPU 参照実装との数値誤差を検出できない）。
///
/// `seed_offset` はテストごとに異なる乱数系列を使うための基準値
/// （各ケースは `seed_offset + 2*index`／`seed_offset + 2*index + 1`
/// を a／b の seed に使う）。形状セット自体は共通で、テスト間の
/// 独立性のため seed のみを分ける。
fn nine_shape_cases(seed_offset: u64) -> Vec<(u64, u64, Vec<usize>, Vec<usize>)> {
    let shapes: Vec<(Vec<usize>, Vec<usize>)> = vec![
        // 整列形状（n%4==0 && k%4==0。128×64 スロット到達条件〈N≥1024
        // かつ K≥1024〉も含む）。
        (vec![3, 64, 256], vec![3, 256, 256]),
        (vec![2, 1024, 1024], vec![2, 1024, 1024]),
        // 非整列形状（classic 経路）。
        (vec![3, 17, 65], vec![3, 65, 33]),
        // broadcast（lhs バッチ次元 1）。
        (vec![1, 5, 6], vec![4, 6, 7]),
        // broadcast（rhs バッチ次元 1）。
        (vec![4, 5, 6], vec![1, 6, 7]),
        // broadcast（中間軸）。rank 4。
        (vec![2, 1, 5, 6], vec![1, 3, 6, 7]),
        // rank 4（broadcast なし）。
        (vec![2, 3, 8, 9], vec![2, 3, 9, 10]),
        // 退化形状: k==0（全 0 出力）。
        (vec![2, 5, 0], vec![2, 0, 6]),
        // 退化形状: m==0（空出力）。
        (vec![2, 0, 5], vec![2, 5, 6]),
    ];

    shapes
        .into_iter()
        .enumerate()
        .map(|(i, (a_shape, b_shape))| {
            let seed_a = seed_offset + 2 * i as u64;
            let seed_b = seed_offset + 2 * i as u64 + 1;
            (seed_a, seed_b, a_shape, b_shape)
        })
        .collect()
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在なら
/// `BackendError::CudaUnavailable` を確認して早期 return する
/// （`gather_scatter_parity.rs::gather_scatter_parity_smoke_env_adaptive`
/// と同じ分岐パターン）。CUDA 実機が利用可能な場合は、小さいバッチ
/// 形状で `gemm_batched`／`gemm_batched_fp32_strict` が CPU 参照実装と
/// REQ-2 複合判定内で一致することまで確認する。
#[test]
fn gemm_batched_parity_smoke_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let cpu = CpuBackendOps::new();

    let a = tensor(1, &[3, 2, 4]);
    let b = tensor(2, &[3, 4, 5]);

    match cuda.gemm_batched(&a, &b) {
        Ok(cuda_out) => {
            let cpu_out = cpu
                .gemm_batched(&a, &b)
                .expect("cpu gemm_batched always succeeds for valid input");
            assert_eq!(cuda_out.shape(), cpu_out.shape());
            assert_parity(
                "gemm_batched smoke: CUDA vs CPU",
                &contiguous_slice(&cuda_out),
                &contiguous_slice(&cpu_out),
            );

            let cuda_strict = cuda
                .gemm_batched_fp32_strict(&a, &b)
                .expect("cuda gemm_batched_fp32_strict must succeed alongside gemm_batched");
            assert_parity(
                "gemm_batched_fp32_strict smoke: CUDA vs CPU",
                &contiguous_slice(&cuda_strict),
                &contiguous_slice(&cpu_out),
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for gemm_batched smoke: {other}"),
    }
}

/// `run_tiled_f32_batched`（新設バッチループ経路）が既定合成実装
/// （`gemm_batched_via_per_batch_gemm_fp32_strict`。導入前の CUDA 既定
/// 経路そのもの）と bit 完全一致することを、整列形状（cp.async
/// pipeline 経路到達）・非整列形状（classic 経路）・broadcast（lhs／
/// rhs 双方・中間軸）・rank 4・退化形状（`k==0`／`m==0`）で確認する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_batched_fp32_strict_matches_default_per_batch_composition_bit_exact_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda = CudaBackendOps::new(device.ordinal());

    let cases = nine_shape_cases(10);

    for (seed_a, seed_b, a_shape, b_shape) in cases {
        let a = tensor(seed_a, &a_shape);
        let b = tensor(seed_b, &b_shape);

        let batched = cuda.gemm_batched_fp32_strict(&a, &b).unwrap_or_else(|e| {
            panic!("gemm_batched_fp32_strict failed for a={a_shape:?}, b={b_shape:?}: {e}")
        });
        let via_per_batch = gemm_batched_via_per_batch_gemm_fp32_strict(&cuda, &a, &b)
            .unwrap_or_else(|e| {
                panic!(
                    "gemm_batched_via_per_batch_gemm_fp32_strict failed for a={a_shape:?}, \
                     b={b_shape:?}: {e}"
                )
            });

        assert_eq!(batched.shape(), via_per_batch.shape());
        assert_eq!(
            bits_vec(&contiguous_slice(&batched)),
            bits_vec(&contiguous_slice(&via_per_batch)),
            "run_tiled_f32_batched must be bit-exact with the default per-batch composition \
             for a={a_shape:?}, b={b_shape:?}"
        );

        // run-to-run 決定性: 同一入力で 2 回起動しても bit 同一。
        let batched2 = cuda
            .gemm_batched_fp32_strict(&a, &b)
            .expect("second invocation must also succeed");
        assert_eq!(
            bits_vec(&contiguous_slice(&batched)),
            bits_vec(&contiguous_slice(&batched2)),
            "run-to-run must be bit-exact for a={a_shape:?}, b={b_shape:?}"
        );
    }
}

/// 上記と同じ形状セット（`nine_shape_cases`。事前登録した全 9 形状。
/// N=K=1024 の別カーネル選択経路〈128×64 スロット到達条件〉を含む）
/// で CPU 参照実装との REQ-2 統一複合判定（tolerance 不変。CPU BLIS の
/// 累積順序は CUDA と異なりうるため bit 同一は約束しない）を確認する
/// （PR #1841 codex-review 指摘対応: 従来は 4 形状のみで N=K=1024 が
/// 未網羅だった）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn gemm_batched_matches_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda = CudaBackendOps::new(device.ordinal());
    let cpu = CpuBackendOps::new();

    let cases = nine_shape_cases(30);

    for (seed_a, seed_b, a_shape, b_shape) in cases {
        let a = tensor(seed_a, &a_shape);
        let b = tensor(seed_b, &b_shape);

        let cuda_out = cuda.gemm_batched(&a, &b).unwrap_or_else(|e| {
            panic!("cuda gemm_batched failed for a={a_shape:?}, b={b_shape:?}: {e}")
        });
        let cpu_out = cpu
            .gemm_batched(&a, &b)
            .expect("cpu gemm_batched always succeeds for valid input");
        assert_eq!(cuda_out.shape(), cpu_out.shape());
        assert_parity(
            &format!("gemm_batched: CUDA vs CPU (a={a_shape:?}, b={b_shape:?})"),
            &contiguous_slice(&cuda_out),
            &contiguous_slice(&cpu_out),
        );
    }
}
