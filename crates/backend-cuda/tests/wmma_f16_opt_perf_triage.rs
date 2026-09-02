//! `wmma_f16_opt` の性能外れ値（イシュー #1123）切り分け用の診断専用テスト。
//!
//! ## 位置づけ
//!
//! `tests/dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record`
//! （転送込み計測。`CudaWmmaGemm::run_f16`／`CudaMmaGemm::run_f16` は毎回
//! `clone_htod`×2・`alloc_zeros`・readback を行う）で GB10 実機実測
//! （2026-09-03）を取ったところ、dim=2048 で wmma_f16_opt 3.60 ms・
//! mma_sync_f16 0.80 ms（2 回とも同値で再現的）、dim=4096 では転送のみ
//! 計測 0.263 s／0.275 s に対し wmma 合算 0.30 s・mma 合算が 1 回目
//! 0.0185 s・2 回目 0.260 s という **二峰性** が観測された。この外れ値が
//! (a) カーネル本体の性能問題か、(b) 転送・アロケーション側（`cudarc`
//! の `cuMemAllocAsync` プールのトリムや unified memory のページ
//! マッピング等）に起因するプロトコル外要因かを切り分けるため、本
//! ファイルは以下を計測する:
//!
//! 1. **カーネル単体**（H2D/D2H・バッファ確保を計測外に置き、
//!    `launch_f16` + `stream.synchronize()` のみを計測ループに含める。
//!    `CudaWmmaGemm::upload_f16`／`alloc_output_f16`／`launch_f16`／
//!    `download_f16`／`synchronize`〈#1013 で追加の常駐バッファ API。
//!    `crates/backend-cuda/src/gemm_wmma.rs`〉、`CudaMmaGemm` の同名 API
//!    〈`gemm_mma.rs`〉を使う。同期契約は `docs/
//!    backend-cuda-async-execution-design.md`「launch → synchronize で
//!    囲む」に従う）
//! 2. **転送込み**（既存 `run_f16` と同一。比較用）
//! 3. **転送のみ**（`tests/dispatch_boundary.rs::transfer_only_measurement`
//!    と同型の実装を **連続 3 回** 計測し中央値を 3 つ並べる。二峰性が
//!    転送・アロケーション側由来かを可視化する）
//!
//! wmma_f16_opt については、opt カーネルを強制する経路（`launch_f16`。
//! opt 可用時は自動的に opt を選ぶ）に加え、基本版カーネルを強制する
//! [`fandhe_ai_backend_cuda::CudaWmmaGemm::launch_f16_basic`]（本イシュー
//! で追加した診断専用の最小 `pub` 入口。本番ディスパッチ〈`run_f16`／
//! `launch_f16` の opt 優先フォールバック〉には影響しない）でも計測し、
//! opt/basic の差を表示する。
//!
//! ## 診断専用であることの明記
//!
//! **本ファイルは実測記録・切り分け専用であり、性能下限（REQ-8）・
//! ディスパッチ規則・カーネル選択ロジックの受け入れ判定には使わない**。
//! 既存テスト・カーネル実装・複合判定の許容誤差はいずれも変更しない
//! （`.claude/rules/coding-rust.md`）。
//!
//! ## 実機前提
//!
//! `tests/dispatch_boundary.rs` と同一の前提（compute capability 8.0
//! 以降・NVRTC 搭載必須）の `#[ignore]` 分離テストであり、通常 CI
//! （self-hosted・CUDA toolkit 非搭載）では実行されない
//! （`.claude/rules/ci.md`「実機依存」）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --test wmma_f16_opt_perf_triage -- --ignored --nocapture
//! ```

use bench_harness::MeasurementConfig;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaMmaGemm, CudaWmmaGemm};
use half::f16;

/// 計測対象形状（イシュー #1123 の観測形状 2048/4096 に加え、外れ値の
/// 出現境界を探るため中間形状 512/768/1024/1536 も含める）。
const DIMS: [u32; 6] = [512, 768, 1024, 1536, 2048, 4096];

fn flops(dim: u32) -> f64 {
    2.0 * (dim as f64).powi(3)
}

/// ホスト⇔デバイス転送のみ（`clone_htod`／`alloc_zeros`／`synchronize`／
/// `clone_dtoh`。カーネル起動を含まない）を計測する
/// （`tests/dispatch_boundary.rs::transfer_only_measurement` と同一の
/// 実装。本ファイルは「診断専用テスト間でのヘルパ共有機構が本クレートに
/// 存在しない」という同ファイルの既知の重複方針をそのまま踏襲する）。
fn transfer_only_measurement(
    device: &CudaDevice,
    config: &MeasurementConfig,
    a: &[f16],
    b: &[f16],
    out_len: usize,
) -> bench_harness::Measurement {
    bench_harness::run(config, || {
        let a_dev = device
            .stream()
            .clone_htod(a)
            .expect("clone_htod must succeed on CUDA-equipped test runner");
        let b_dev = device
            .stream()
            .clone_htod(b)
            .expect("clone_htod must succeed on CUDA-equipped test runner");
        let c_dev = device
            .stream()
            .alloc_zeros::<f16>(out_len)
            .expect("alloc_zeros must succeed on CUDA-equipped test runner");
        device
            .stream()
            .synchronize()
            .expect("synchronize must succeed on CUDA-equipped test runner");
        let _c_host = device
            .stream()
            .clone_dtoh(&c_dev)
            .expect("clone_dtoh must succeed on CUDA-equipped test runner");
        drop(a_dev);
        drop(b_dev);
    })
    .expect("transfer-only measurement must satisfy TASK-8.1 protocol")
}

/// イシュー #1123 の外れ値切り分け実測本体。
///
/// dim ごとに以下を stdout へ 1 行の表として出力する:
/// `path, dim, kernel_only_ms, kernel_tflops, run_f16_ms, transfer_only_ms(x3)`
///
/// カーネル単体・転送込み・転送のみの 3 系統の相対関係を突き合わせる
/// ことで、外れ値がカーネル本体（launch→synchronize の区間）に起因する
/// のか、転送・アロケーション側（H2D/D2H・`cuMemAllocAsync` プールの
/// トリム等）に起因するのかを切り分ける（本ファイル冒頭コメント参照）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須。イシュー #1123 の診断専用テストで受け入れ判定には使わない"]
fn wmma_f16_opt_perf_triage_record() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    println!(
        "environment: name={:?} compute_capability={:?} arch={:?}",
        device.name(),
        device.compute_capability(),
        device.arch()
    );

    let wmma_gemm = CudaWmmaGemm::new(&device).expect("CudaWmmaGemm::new (WMMA f16) must succeed");
    assert!(
        wmma_gemm.wmma_f16_opt_available(),
        "WMMA f16 opt kernel must be available on this ignored test runner so that the \
         opt/basic split-out actually exercises both kernel variants (reason: {:?})",
        wmma_gemm.wmma_f16_opt_unavailable_reason()
    );
    // `CudaMmaGemm::new` は cc>=8.0 かつ NVRTC 搭載が必須
    // （`gemm_mma.rs::MIN_COMPUTE_CAPABILITY_MAJOR = 8`）。本テストの
    // `#[ignore]` 前提と一致するため、失敗時はテスト全体を失敗させる
    // （`tests/dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record`
    // と同じ判断）。
    let mma_gemm = CudaMmaGemm::new(&device).expect(
        "CudaMmaGemm::new must succeed on the ignored test runner (cc>=8.0・NVRTC 搭載が前提)",
    );

    let config = MeasurementConfig::new(20, 20).expect("20/20 must satisfy TASK-8.1 minimums");

    println!(
        "path,dim,kernel_only_ms,kernel_tflops,run_f16_ms,transfer_only_ms_1,transfer_only_ms_2,transfer_only_ms_3"
    );

    for &dim in &DIMS {
        let (m, n, k) = (dim, dim, dim);
        let out_len = (m as usize) * (n as usize);

        let mut rng = Xorshift64Star::new(0xD000 + u64::from(dim));
        let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
        let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

        // 転送のみ計測を連続 3 回取り、二峰性の可視化に用いる（本ファイル
        // 冒頭コメント「転送のみ」参照）。
        let transfer_1 = transfer_only_measurement(&device, &config, &a, &b, out_len);
        let transfer_2 = transfer_only_measurement(&device, &config, &a, &b, out_len);
        let transfer_3 = transfer_only_measurement(&device, &config, &a, &b, out_len);

        // --- wmma_f16_opt（opt カーネル優先。`launch_f16`）---
        {
            let (a_dev, b_dev) = wmma_gemm
                .upload_f16(&a, &b)
                .expect("upload_f16 must succeed on CUDA-equipped test runner");
            let mut c_dev = wmma_gemm
                .alloc_output_f16(m, n)
                .expect("alloc_output_f16 must succeed on CUDA-equipped test runner");
            let kernel_only = bench_harness::run(&config, || {
                wmma_gemm
                    .launch_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)
                    .expect("launch_f16 must succeed on CUDA-equipped test runner");
                wmma_gemm
                    .synchronize()
                    .expect("synchronize must succeed on CUDA-equipped test runner");
            })
            .expect("kernel-only measurement must satisfy TASK-8.1 protocol");
            let kernel_tflops = (flops(dim) / kernel_only.median_secs) / 1e12;

            let run_f16 = bench_harness::run(&config, || {
                let _ = wmma_gemm
                    .run_f16(&a, &b, m, n, k)
                    .expect("run_f16 must succeed on CUDA-equipped test runner");
            })
            .expect("run_f16 measurement must satisfy TASK-8.1 protocol");

            print_row(
                "wmma_f16_opt",
                dim,
                &kernel_only,
                kernel_tflops,
                &run_f16,
                &transfer_1,
                &transfer_2,
                &transfer_3,
            );
        }

        // --- wmma_f16_basic（opt を経由せず基本版カーネルを強制。
        // `launch_f16_basic`。本ファイル冒頭コメント「wmma_f16_opt に
        // ついては」参照）---
        {
            let (a_dev, b_dev) = wmma_gemm
                .upload_f16(&a, &b)
                .expect("upload_f16 must succeed on CUDA-equipped test runner");
            let mut c_dev = wmma_gemm
                .alloc_output_f16(m, n)
                .expect("alloc_output_f16 must succeed on CUDA-equipped test runner");
            let kernel_only = bench_harness::run(&config, || {
                wmma_gemm
                    .launch_f16_basic(&a_dev, &b_dev, &mut c_dev, m, n, k)
                    .expect("launch_f16_basic must succeed on CUDA-equipped test runner");
                wmma_gemm
                    .synchronize()
                    .expect("synchronize must succeed on CUDA-equipped test runner");
            })
            .expect("kernel-only (basic) measurement must satisfy TASK-8.1 protocol");
            let kernel_tflops = (flops(dim) / kernel_only.median_secs) / 1e12;

            // `run_f16` 相当の転送込み計測は基本版専用の入口がないため
            // 対象外とする（basic 経路は opt 可用性に関わらず起動できる
            // `launch_f16_basic` のみを比較対象とする。本テストの目的は
            // opt/basic のカーネル単体差の把握であり、転送込み合算値は
            // opt 経路の run_f16 出力のみで十分）。
            println!(
                "wmma_f16_basic,{dim},{:.4},{:.3},n/a,n/a,n/a,n/a",
                kernel_only.median_secs * 1000.0,
                kernel_tflops,
            );
        }

        // --- mma_sync_f16（`CudaMmaGemm`。`mma.sync`/`ldmatrix`/`cp.async`
        // パイプライン）---
        {
            let (a_dev, b_dev) = mma_gemm
                .upload_f16(&a, &b)
                .expect("upload_f16 must succeed on CUDA-equipped test runner");
            let mut c_dev = mma_gemm
                .alloc_output_f16(m, n)
                .expect("alloc_output_f16 must succeed on CUDA-equipped test runner");
            let kernel_only = bench_harness::run(&config, || {
                mma_gemm
                    .launch_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)
                    .expect("launch_f16 must succeed on CUDA-equipped test runner");
                mma_gemm
                    .synchronize()
                    .expect("synchronize must succeed on CUDA-equipped test runner");
            })
            .expect("kernel-only measurement must satisfy TASK-8.1 protocol");
            let kernel_tflops = (flops(dim) / kernel_only.median_secs) / 1e12;

            let run_f16 = bench_harness::run(&config, || {
                let _ = mma_gemm
                    .run_f16(&a, &b, m, n, k)
                    .expect("run_f16 must succeed on CUDA-equipped test runner");
            })
            .expect("run_f16 measurement must satisfy TASK-8.1 protocol");

            print_row(
                "mma_sync_f16",
                dim,
                &kernel_only,
                kernel_tflops,
                &run_f16,
                &transfer_1,
                &transfer_2,
                &transfer_3,
            );
        }
    }
}

/// 表 1 行分の出力（CSV 風。`path,dim,kernel_only_ms,kernel_tflops,
/// run_f16_ms,transfer_only_ms_1,transfer_only_ms_2,transfer_only_ms_3`）。
/// 複数箇所（opt/mma_sync）で同一書式を使うため関数へ切り出す
/// （wmma_f16_basic は転送込み・転送のみを持たないため専用の `println!`
/// を呼び出し側に残す）。
#[allow(clippy::too_many_arguments)]
fn print_row(
    path: &str,
    dim: u32,
    kernel_only: &bench_harness::Measurement,
    kernel_tflops: f64,
    run_f16: &bench_harness::Measurement,
    transfer_1: &bench_harness::Measurement,
    transfer_2: &bench_harness::Measurement,
    transfer_3: &bench_harness::Measurement,
) {
    println!(
        "{path},{dim},{:.4},{:.3},{:.4},{:.4},{:.4},{:.4}",
        kernel_only.median_secs * 1000.0,
        kernel_tflops,
        run_f16.median_secs * 1000.0,
        transfer_1.median_secs * 1000.0,
        transfer_2.median_secs * 1000.0,
        transfer_3.median_secs * 1000.0,
    );
}
