//! Tensor Core（WMMA TF32／WMMA f16）経路の実機実測・数値一致検証
//! （TASK-11.1e・#64）。
//!
//! ## 位置づけ
//!
//! TASK-11.1a〜d（#60〜#63）で WMMA(TF32)／WMMA(f16) の基本・opt カーネルと
//! 経路別 parity テスト（`tests/gemm_wmma_tf32.rs`・`tests/gemm_wmma_tf32_opt.rs`・
//! `tests/gemm_wmma.rs`・`tests/cpu_cuda_wmma_parity.rs`・`tests/gemm_wmma_f16_opt.rs`）
//! は整備済みだが、それらは経路単体の検証に閉じており「Tensor Core 経路が
//! tiled f32 基準に対して TFLOPS で優位である」ことと「複合判定通過」を
//! 1 回の実機実行で横断的に記録する導線が存在しなかった（#64 受け入れ条件
//! 「実機実測記録（TFLOPS・複合判定通過）が残されている」）。
//!
//! 本ファイルは以下 2 点を提供する:
//!
//! 1. [`tensor_core_tflops_record`]: tiled f32／WMMA TF32（opt）／WMMA f16
//!    （opt）3 経路の `bench_harness::protocol::run`（warmup 20・計測 20、
//!    正本 TASK-8.1 準拠）による計測を 1 テスト内で行い、
//!    `bench_harness::report::BenchReport::to_json` で構造化出力する
//!    （`--nocapture` 実行で `docs/perf/cuda-tensor-core-measurement.md` の
//!    記録テンプレートへ転記できる形式）。
//! 2. [`tensor_core_parity_record`]: TF32・f16 経路の複合判定
//!    （`backend_cpu::assert_parity`。REQ-2 統一複合判定「相対誤差 1e-3
//!    未満 または 絶対誤差 1e-5 未満」の唯一の実体）通過を記録用に明示
//!    出力する。判定式・閾値はここでローカル複製しない
//!    （`.claude/rules/coding-rust.md`）。
//!
//! ## 実機前提
//!
//! 両テストとも実機（DGX Spark GB10 等、compute capability 7.0 以降）必須の
//! `#[ignore]` 分離テストであり、通常 CI（self-hosted・CUDA toolkit 非搭載）
//! では実行されない（`.claude/rules/ci.md`「実機依存」）。CUDA デバイス・
//! opt カーネルが利用できない環境では `.expect` により失敗が顕在化する
//! 設計とし、実機以外での silent green を許さない（既存 `tests/
//! gemm_wmma_tf32_opt.rs` の `#[ignore]` テスト群と同じ規約）。
//!
//! opt カーネルの可用性は `wmma_tf32_opt_available`／`wmma_f16_opt_available`
//! で事前に断定してから計測する（PR #256 レビュー指摘「opt 経路の可用性を
//! 断定せず計測すると基本版へのサイレントフォールバックで green になり
//! うる」への対処。`gemm.rs::CudaGemm::wmma_tf32_opt_available`・
//! `gemm_wmma.rs::CudaWmmaGemm::wmma_f16_opt_available` ドキュメンテーション
//! コメント参照）。
//!
//! 実機で複合判定を外れた場合は許容誤差を緩和せず、閾値実測再評価の
//! イシュー #186（Tensor Core 経路の数値一致閾値の実測再評価）へ引き渡す
//! （`.claude/rules/security.md`「ガードレール閾値・テスト許容誤差の変更は
//! 必ず人間の承認を経る」）。

use backend_cuda::{CudaDevice, CudaGemm, CudaWmmaGemm};
use bench_harness::{BenchReport, MeasurementConfig};
use half::f16;

/// TFLOPS 実測記録の本体（#64 受け入れ条件の TFLOPS 実測分）。
///
/// M=N=K=4096（`tests/gemm_wmma_tf32_opt.rs::
/// wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096` と同一形状。PoC-v2-3
/// 参考値 1.832 TFLOPS と比較可能にする）で tiled f32・WMMA TF32・WMMA
/// f16 の 3 経路を計測し、TFLOPS 換算値と `BenchReport` の JSON を
/// `println!` で出力する。`--nocapture` での実行結果を
/// `docs/perf/cuda-tensor-core-measurement.md` の記録テンプレートへ
/// 転記する運用（本ファイル冒頭コメント参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 7.0 以降）必須。実測記録は docs/perf/cuda-tensor-core-measurement.md"]
fn tensor_core_tflops_record() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

    // 記録 doc の「計測環境」節へ転記する素材。
    println!(
        "environment: name={:?} compute_capability={:?} arch={:?}",
        device.name(),
        device.compute_capability(),
        device.arch()
    );

    let gemm = CudaGemm::new(&device).expect("CudaGemm::new (tiled/WMMA TF32) must succeed");
    assert!(
        gemm.wmma_tf32_opt_available(),
        "WMMA TF32 opt kernel must be available on this ignored test runner so that the \
         TFLOPS record actually exercises the optimized kernel rather than silently falling \
         back to the basic WMMA kernel (reason: {:?})",
        gemm.wmma_tf32_opt_unavailable_reason()
    );

    let wmma_gemm = CudaWmmaGemm::new(&device).expect("CudaWmmaGemm::new (WMMA f16) must succeed");
    assert!(
        wmma_gemm.wmma_f16_opt_available(),
        "WMMA f16 opt kernel must be available on this ignored test runner so that the \
         TFLOPS record actually exercises the optimized kernel rather than silently falling \
         back to the basic WMMA kernel (reason: {:?})",
        wmma_gemm.wmma_f16_opt_unavailable_reason()
    );

    let (m, n, k) = (4096u32, 4096u32, 4096u32);
    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let config = MeasurementConfig::new(20, 20).expect("20/20 must satisfy TASK-8.1 minimums");

    let mut rng_f32 = bench_harness::rng::Xorshift64Star::new(0xACE1);
    let a_f32 = rng_f32.fill_vec((m as usize) * (k as usize));
    let b_f32 = rng_f32.fill_vec((k as usize) * (n as usize));

    let mut rng_f16 = bench_harness::rng::Xorshift64Star::new(0xBEEF01);
    let a_f16: Vec<f16> = rng_f16.fill_vec_f16((m as usize) * (k as usize));
    let b_f16: Vec<f16> = rng_f16.fill_vec_f16((k as usize) * (n as usize));

    // tiled f32（基準経路。PoC-v2-3 参考値 1.832 TFLOPS）。
    let tiled_measurement = bench_harness::run(&config, || {
        let _ = gemm
            .run_tiled_f32(&a_f32, &b_f32, m, n, k)
            .expect("run_tiled_f32 must succeed on CUDA-equipped test runner");
    })
    .expect("tiled f32 measurement must satisfy TASK-8.1 protocol");
    let tiled_report =
        BenchReport::from_measurement("gemm_tiled_f32_4096", "cuda", &tiled_measurement).expect(
            "BenchReport::from_measurement must succeed for a protocol-conformant measurement",
        );
    let tiled_tflops = (flops / tiled_measurement.median_secs) / 1e12;
    println!(
        "tflops_record path=tiled_f32 tflops={tiled_tflops:.3} report={}",
        tiled_report
            .to_json()
            .expect("BenchReport::to_json must succeed for a validated report")
    );

    // WMMA TF32（opt。上記 assert で opt 経路の実行を保証済み）。
    let tf32_measurement = bench_harness::run(&config, || {
        let _ = gemm
            .run_wmma_tf32(&a_f32, &b_f32, m, n, k)
            .expect("run_wmma_tf32 must succeed on CUDA-equipped test runner");
    })
    .expect("WMMA TF32 measurement must satisfy TASK-8.1 protocol");
    let tf32_report =
        BenchReport::from_measurement("gemm_wmma_tf32_opt_4096", "cuda", &tf32_measurement).expect(
            "BenchReport::from_measurement must succeed for a protocol-conformant measurement",
        );
    let tf32_tflops = (flops / tf32_measurement.median_secs) / 1e12;
    println!(
        "tflops_record path=wmma_tf32_opt tflops={tf32_tflops:.3} report={}",
        tf32_report
            .to_json()
            .expect("BenchReport::to_json must succeed for a validated report")
    );

    // WMMA f16（opt。上記 assert で opt 経路の実行を保証済み）。
    let f16_measurement = bench_harness::run(&config, || {
        let _ = wmma_gemm
            .run_f16(&a_f16, &b_f16, m, n, k)
            .expect("run_f16 must succeed on CUDA-equipped test runner");
    })
    .expect("WMMA f16 measurement must satisfy TASK-8.1 protocol");
    let f16_report =
        BenchReport::from_measurement("gemm_wmma_f16_opt_4096", "cuda", &f16_measurement).expect(
            "BenchReport::from_measurement must succeed for a protocol-conformant measurement",
        );
    let f16_tflops = (flops / f16_measurement.median_secs) / 1e12;
    println!(
        "tflops_record path=wmma_f16_opt tflops={f16_tflops:.3} report={}",
        f16_report
            .to_json()
            .expect("BenchReport::to_json must succeed for a validated report")
    );

    // #64 受け入れ条件・既存 `wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096`
    // と同じ判断根拠: Tensor Core 経路は同一転送コストを負担する tiled f32
    // を上回ることを実機で確認する（相対比較。実機で外れた場合は緩和せず
    // #186 へ引き渡す。本ファイル冒頭コメント参照）。
    assert!(
        tf32_tflops > tiled_tflops,
        "WMMA TF32 opt（{tf32_tflops:.3} TFLOPS）が tiled f32（{tiled_tflops:.3} TFLOPS）を \
         上回りませんでした（受け入れ条件: PoC-v2-3 参考値 1.832 TFLOPS 超過）"
    );
    assert!(
        f16_tflops > tiled_tflops,
        "WMMA f16 opt（{f16_tflops:.3} TFLOPS）が tiled f32（{tiled_tflops:.3} TFLOPS）を \
         上回りませんでした（受け入れ条件: PoC-v2-3 参考値 1.832 TFLOPS 超過）"
    );
}

/// 複合判定通過の記録（#64 受け入れ条件の数値一致検証分）。
///
/// TF32 経路は `CudaGemm::run_wmma_tf32` と `backend_cpu::matmul_reference_fma`
/// を、f16 経路は `tests/cpu_cuda_wmma_parity.rs` の確立済み手順
/// （f16→f32 参照計算→f16 丸め→f32 化→`assert_parity`）を踏襲して比較する。
/// 形状は 512×512×512（CPU 参照計算が実機でも数秒以内に収まる規模。
/// `tests/gemm_wmma_tf32_opt.rs::wmma_tf32_opt_matches_reference_across_shapes`
/// の倍数境界形状の 1 つと同じ）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 7.0 以降）必須。実測記録は docs/perf/cuda-tensor-core-measurement.md"]
fn tensor_core_parity_record() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let (m, n, k) = (512u32, 512u32, 512u32);

    // TF32 経路。
    let gemm = CudaGemm::new(&device).expect("CudaGemm::new (tiled/WMMA TF32) must succeed");
    assert!(
        gemm.wmma_tf32_opt_available(),
        "WMMA TF32 opt kernel must be available on this ignored test runner (reason: {:?})",
        gemm.wmma_tf32_opt_unavailable_reason()
    );

    let mut rng_tf32 = bench_harness::rng::Xorshift64Star::new(0x7A0);
    let a_tf32 = rng_tf32.fill_vec((m as usize) * (k as usize));
    let b_tf32 = rng_tf32.fill_vec((k as usize) * (n as usize));
    let mut c_ref_tf32 = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::matmul_reference_fma(
        &a_tf32,
        &b_tf32,
        &mut c_ref_tf32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_gpu_tf32 = gemm
        .run_wmma_tf32(&a_tf32, &b_tf32, m, n, k)
        .expect("run_wmma_tf32 must succeed on CUDA-equipped test runner");
    // 判定式・閾値は `backend_cpu::assert_parity` に一本化する（ローカル
    // 複製しない。`.claude/rules/coding-rust.md`）。実機で外れた場合は
    // 緩和せず #186 へ引き渡す。
    backend_cpu::assert_parity(
        "tensor_core_parity_record tf32 512x512x512",
        &c_gpu_tf32,
        &c_ref_tf32,
    );
    println!(
        "parity_record path=wmma_tf32_opt shape=512x512x512 result=pass \
         (composite tolerance: relative<1e-3 or absolute<1e-5)"
    );

    // f16 経路（`tests/cpu_cuda_wmma_parity.rs::assert_wmma_f16_parity` と
    // 同じ量子化手順。カーネルのエピローグ store〈__float2half〉と同じ
    // 丸めを参照側にも適用する）。
    let wmma_gemm = CudaWmmaGemm::new(&device).expect("CudaWmmaGemm::new (WMMA f16) must succeed");
    assert!(
        wmma_gemm.wmma_f16_opt_available(),
        "WMMA f16 opt kernel must be available on this ignored test runner (reason: {:?})",
        wmma_gemm.wmma_f16_opt_unavailable_reason()
    );

    let mut rng_f16 = bench_harness::rng::Xorshift64Star::new(0xF160);
    let a_f16: Vec<f16> = rng_f16.fill_vec_f16((m as usize) * (k as usize));
    let b_f16: Vec<f16> = rng_f16.fill_vec_f16((k as usize) * (n as usize));
    let a_f32_from_f16: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32_from_f16: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32_from_f16 = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::matmul_reference_fma(
        &a_f32_from_f16,
        &b_f32_from_f16,
        &mut c_ref_f32_from_f16,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_ref_rounded: Vec<f32> = c_ref_f32_from_f16
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();
    let c_gpu_f16 = wmma_gemm
        .run_f16(&a_f16, &b_f16, m, n, k)
        .expect("run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32_from_f16: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();
    backend_cpu::assert_parity(
        "tensor_core_parity_record f16 512x512x512",
        &c_gpu_f32_from_f16,
        &c_ref_rounded,
    );
    println!(
        "parity_record path=wmma_f16_opt shape=512x512x512 result=pass \
         (composite tolerance: relative<1e-3 or absolute<1e-5)"
    );
}
