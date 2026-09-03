//! CUDA GEMM 経路選択（`fandhe_ai_tensor_core::dispatch::select_gemm_kernel`）の
//! 境界形状 実測再検証（TASK-11.2c・#69）。
//!
//! ## 位置づけ
//!
//! `docs/dispatch-rules-design.md` §3.2「CUDA との非対称設計の実体」は、
//! CUDA 側に Metal のような `min(M,N,K)` 形状下限を設けない設計を、
//! v1 参考値（GB10 実測で小形状 256 でも Tensor Core 経路が 1.4〜1.6 倍
//! 優位）のみを根拠に採用している。本ファイルはこの「形状下限なし」
//! 判断を **v2 自作カーネル上の実測**で裏付ける。あわせて、v1 の
//! 「GB10 世代での TMA 選好」を v2 設計ではディスパッチ条件に含めず
//! Tensor Core 経路内部のチューニング材料として扱う整理（同文書 §3.2
//! 「TMA の扱い」）を、`mma.sync`/`ldmatrix`/`cp.async` パイプライン
//! （#187・`CudaMmaGemm`）と基本 WMMA（`CudaWmmaGemm`）の大形状比較で
//! 裏付ける。
//!
//! 本ファイルは以下 2 点を提供する:
//!
//! 1. [`small_shape_matrix_unit_has_no_floor_tflops_record`] 関数: 小形状
//!    128/256/384/512 で WMMA f16（opt）対 tiled f16、WMMA TF32（opt）
//!    対 tiled f32 を `bench_harness::protocol::run`（warmup 20・計測 20、
//!    正本 TASK-8.1 準拠）で計測し、`BenchReport::to_json` で構造化出力
//!    する（「CUDA は形状下限を設けない」規則の実測根拠。小形状でも
//!    MatrixUnit（WMMA）が Tiled を下回らないかを記録する）。
//! 2. [`large_shape_mma_pipeline_vs_wmma_tflops_record`] 関数: 大形状
//!    2048/4096 で `mma.sync`/`ldmatrix`/`cp.async` パイプライン
//!    （`CudaMmaGemm`。v1 TMA 選好の v2 対応物）と基本 WMMA f16（opt）を
//!    比較し、両者とも `select_gemm_kernel`（第 1 層）上は同じ
//!    `KernelKind::MatrixUnit` に写像されることを踏まえ、「mma
//!    パイプラインの優劣は経路選択条件でなくカーネル内部チューニング」
//!    の整理を実測で裏付ける記録を残す。第 2 層（`CudaGemmAuto` 内部）の
//!    判定ロジックは #1156 で整列形状・cc>=8.0 時に `CudaMmaGemm` を
//!    優先する設計どおり実装済みだが、`gemm_auto::MMA_PRIORITY_PRODUCTION_
//!    ENABLED`（本番既定 `false`）でゲートされ本番では未有効化（#1160 の
//!    実測・承認待ち）。この整理と計測ロジック自体は変更しない（第 1 層の
//!    決定表が両者を区別しない事実に基づく整理のため）。
//!
//! ## 実機前提
//!
//! 両テストとも実機（DGX Spark GB10 等、compute capability 8.0 以降
//! ——`CudaMmaGemm` の下限がより厳しいため両テスト共通の下限とする）・
//! NVRTC 搭載が必須の `#[ignore]` 分離テストであり、通常 CI（GitHub ホステッド・
//! CUDA 実機なし）では実行されない（`.claude/rules/ci.md`「実機
//! 依存」）。opt カーネルの可用性は `wmma_f16_opt_available`／
//! `wmma_tf32_opt_available` で事前に断定してから計測する（PR #256
//! レビュー指摘「opt 経路の可用性を断定せず計測すると基本版への
//! サイレントフォールバックで green になりうる」への対処。`tests/
//! tensor_core_real_device.rs` と同じ規約）。
//!
//! ## 転送バイト数差の補正（[`small_shape_matrix_unit_has_no_floor_tflops_record`] 限定）
//!
//! `tests/tensor_core_real_device.rs` の `transfer_only_measurement` と
//! 同じ理由（f32 系は 4 byte/要素・f16 系は 2 byte/要素で転送バイト数が
//! 異なり、合算計測のまま比較すると転送コストの差が TFLOPS 比較を歪める。
//! PR #258 レビュー指摘「f16 benchmark rewards smaller transfers」）で、
//! [`small_shape_matrix_unit_has_no_floor_tflops_record`] は dtype ごとの
//! 転送のみ計測を差し引いた「計算のみ」の TFLOPS で比較する。
//!
//! [`large_shape_mma_pipeline_vs_wmma_tflops_record`] は本節の対象外
//! （イシュー #1123 是正）。同関数はいずれも f16 のみを扱うため転送
//! バイト数差の問題自体が生じない一方、GB10 実機実測（2026-09-03）で
//! `run_f16`（転送＋カーネル実行の合算計測）から転送のみ計測を差し引く
//! 旧プロトコルは、大容量バッファ（4096 で 1 個あたり 32 MB 超）の
//! per-call アロケーション＋転送が二峰性（0.263 s/0.275 s と大きく
//! 乖離）を示す既知病態（`docs/perf/cuda-wmma-f16-perf-triage.md`）により
//! 破綻することが判明した。このため同関数は
//! `CudaWmmaGemm::upload_f16`／`alloc_output_f16`／`launch_f16`／
//! `synchronize`（`CudaMmaGemm` も同名 API）による常駐バッファ計測
//! （H2D/D2H・バッファ確保を計測区間の外に置く「カーネル単体」計測）へ
//! 切り替え、転送のみ計測の差し引き・「転送のみ ≥ 合算」プロトコル
//! 整合性検査を行わない。
//!
//! ## 閾値・許容誤差の不変更
//!
//! 本ファイルは実測を記録するのみで、`CUDA_WMMA_MIN_CC`・複合判定の
//! 許容誤差はいずれも変更しない。CUDA の「形状下限なし」規則自体も
//! 本ファイルでは変更しない（実測が規則を裏付けない結果になった場合は
//! 緩和せず後続レビューへ引き渡す。`.claude/rules/coding-rust.md`）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use bench_harness::{BenchReport, Measurement, MeasurementConfig};
use fandhe_ai_backend_cuda::{CudaDevice, CudaGemm, CudaMmaGemm, CudaWmmaGemm};
use half::f16;

/// 小形状（`min(M,N,K)` 下限なし検証用。128/256/384/512。「小形状でも
/// MatrixUnit が有利」という v1 参考値を v2 自作カーネルで再確認する
/// 解像度として 128 刻みを採用する）。
const SMALL_DIMS: [u32; 4] = [128, 256, 384, 512];

/// 大形状（mma パイプライン対 WMMA 基本比較用。`tests/
/// tensor_core_real_device.rs` の 4096 に加え、パイプライン化の効果が
/// 出やすい中間点として 2048 も計測する）。
const LARGE_DIMS: [u32; 2] = [2048, 4096];

fn flops(dim: u32) -> f64 {
    2.0 * (dim as f64).powi(3)
}

/// ホスト⇔デバイス転送のみ（`clone_htod`／`alloc_zeros`／`synchronize`／
/// `clone_dtoh`。カーネル起動を含まない）を計測する
/// （`tests/tensor_core_real_device.rs::transfer_only_measurement` と
/// 同一の実装・同一の補正目的。本ファイル冒頭コメント「転送バイト数差の
/// 補正」参照。ローカル複製ではなくコピーになる点は、`backend-cuda` の
/// 統合テスト間でヘルパを共有する `tests/common/` 等の仕組みが本クレート
/// に存在しないための已知の重複であり、複合判定・閾値のような regulated
/// な値の複製ではないため許容する）。
fn transfer_only_measurement<T>(
    device: &CudaDevice,
    config: &MeasurementConfig,
    a: &[T],
    b: &[T],
    out_len: usize,
) -> Measurement
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
{
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
            .alloc_zeros::<T>(out_len)
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

/// 「CUDA は形状下限を設けない」規則（`docs/dispatch-rules-design.md`
/// §3.2）の実測根拠。小形状 128/256/384/512 で WMMA（opt）が tiled を
/// 下回らないかを記録する。
///
/// `CudaGemm`（naive/tiled/WMMA TF32）と `CudaWmmaGemm`（WMMA f16）の
/// 両方を使う。opt カーネルの可用性は事前に断定する（本ファイル冒頭
/// コメント参照）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須。実測記録は docs/perf/dispatch-boundary-measurement.md"]
fn small_shape_matrix_unit_has_no_floor_tflops_record() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
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
         small-shape record actually exercises the optimized kernel (reason: {:?})",
        gemm.wmma_tf32_opt_unavailable_reason()
    );
    let wmma_gemm = CudaWmmaGemm::new(&device).expect("CudaWmmaGemm::new (WMMA f16) must succeed");
    assert!(
        wmma_gemm.wmma_f16_opt_available(),
        "WMMA f16 opt kernel must be available on this ignored test runner (reason: {:?})",
        wmma_gemm.wmma_f16_opt_unavailable_reason()
    );

    let config = MeasurementConfig::new(20, 20).expect("20/20 must satisfy TASK-8.1 minimums");

    for &dim in &SMALL_DIMS {
        let (m, n, k) = (dim, dim, dim);
        let out_len = (m as usize) * (n as usize);

        let mut rng_f32 = Xorshift64Star::new(0xA000 + u64::from(dim));
        let a_f32 = rng_f32.fill_vec((m as usize) * (k as usize));
        let b_f32 = rng_f32.fill_vec((k as usize) * (n as usize));
        let mut rng_f16 = Xorshift64Star::new(0xB000 + u64::from(dim));
        let a_f16: Vec<f16> = rng_f16.fill_vec_f16((m as usize) * (k as usize));
        let b_f16: Vec<f16> = rng_f16.fill_vec_f16((k as usize) * (n as usize));

        let f32_transfer =
            transfer_only_measurement::<f32>(&device, &config, &a_f32, &b_f32, out_len);
        let f16_transfer =
            transfer_only_measurement::<f16>(&device, &config, &a_f16, &b_f16, out_len);

        // tiled f32 vs WMMA TF32(opt)。
        let tiled_f32_measurement = bench_harness::run(&config, || {
            let _ = gemm
                .run_tiled_f32(&a_f32, &b_f32, m, n, k)
                .expect("run_tiled_f32 must succeed on CUDA-equipped test runner");
        })
        .expect("tiled f32 measurement must satisfy TASK-8.1 protocol");
        let tf32_measurement = bench_harness::run(&config, || {
            let _ = gemm
                .run_wmma_tf32(&a_f32, &b_f32, m, n, k)
                .expect("run_wmma_tf32 must succeed on CUDA-equipped test runner");
        })
        .expect("WMMA TF32 measurement must satisfy TASK-8.1 protocol");
        let tiled_f32_compute = tiled_f32_measurement.median_secs - f32_transfer.median_secs;
        let tf32_compute = tf32_measurement.median_secs - f32_transfer.median_secs;
        // 転送のみ計測が合算計測を下回ること（=計算のみ時間が正）を保証する。
        // `tests/tensor_core_real_device.rs:235-239` と同じガードで、小形状
        // （dim=128 等）では計算時間が転送時間と同程度以下になりうるため
        // 負値・`inf` の TFLOPS が出うる（PR レビュー指摘）。
        assert!(
            tiled_f32_compute > 0.0 && tf32_compute > 0.0,
            "転送のみ計測（f32: {:.6}s）が合算計測（tiled_f32: {:.6}s, wmma_tf32: {:.6}s）を \
             下回りませんでした（dim={dim}）。計測がプロトコル前提（転送・カーネル実行の直列化）\
             を満たしていない可能性があります",
            f32_transfer.median_secs,
            tiled_f32_measurement.median_secs,
            tf32_measurement.median_secs,
        );
        let tiled_f32_tflops = (flops(dim) / tiled_f32_compute) / 1e12;
        let tf32_tflops = (flops(dim) / tf32_compute) / 1e12;

        // tiled f16 vs WMMA f16(opt)。
        let tiled_f16_measurement = bench_harness::run(&config, || {
            let _ = gemm
                .run_tiled_f16(&a_f16, &b_f16, m, n, k)
                .expect("run_tiled_f16 must succeed on CUDA-equipped test runner");
        })
        .expect("tiled f16 measurement must satisfy TASK-8.1 protocol");
        let wmma_f16_measurement = bench_harness::run(&config, || {
            let _ = wmma_gemm
                .run_f16(&a_f16, &b_f16, m, n, k)
                .expect("run_f16 must succeed on CUDA-equipped test runner");
        })
        .expect("WMMA f16 measurement must satisfy TASK-8.1 protocol");
        let tiled_f16_compute = tiled_f16_measurement.median_secs - f16_transfer.median_secs;
        let wmma_f16_compute = wmma_f16_measurement.median_secs - f16_transfer.median_secs;
        // 上記 f32 系と同じ理由の正値ガード（`tests/tensor_core_real_device.rs:235-239`）。
        assert!(
            tiled_f16_compute > 0.0 && wmma_f16_compute > 0.0,
            "転送のみ計測（f16: {:.6}s）が合算計測（tiled_f16: {:.6}s, wmma_f16: {:.6}s）を \
             下回りませんでした（dim={dim}）。計測がプロトコル前提（転送・カーネル実行の直列化）\
             を満たしていない可能性があります",
            f16_transfer.median_secs,
            tiled_f16_measurement.median_secs,
            wmma_f16_measurement.median_secs,
        );
        let tiled_f16_tflops = (flops(dim) / tiled_f16_compute) / 1e12;
        let wmma_f16_tflops = (flops(dim) / wmma_f16_compute) / 1e12;

        let tf32_report = BenchReport::from_measurement(
            format!("gemm_wmma_tf32_opt_dim{dim}"),
            "cuda",
            &tf32_measurement,
        )
        .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");
        let wmma_f16_report = BenchReport::from_measurement(
            format!("gemm_wmma_f16_opt_dim{dim}"),
            "cuda",
            &wmma_f16_measurement,
        )
        .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");

        println!("dispatch_boundary_record dim={dim} path=tiled_f32 tflops={tiled_f32_tflops:.3}");
        println!(
            "dispatch_boundary_record dim={dim} path=wmma_tf32_opt tflops={tf32_tflops:.3} \
             matrix_unit_over_tiled={:.3} report={}",
            tf32_tflops / tiled_f32_tflops,
            tf32_report
                .to_json()
                .expect("BenchReport::to_json must succeed for a validated report")
        );
        println!("dispatch_boundary_record dim={dim} path=tiled_f16 tflops={tiled_f16_tflops:.3}");
        println!(
            "dispatch_boundary_record dim={dim} path=wmma_f16_opt tflops={wmma_f16_tflops:.3} \
             matrix_unit_over_tiled={:.3} report={}",
            wmma_f16_tflops / tiled_f16_tflops,
            wmma_f16_report
                .to_json()
                .expect("BenchReport::to_json must succeed for a validated report")
        );
    }
}

/// v1「GB10 世代での TMA 選好」を v2 設計ではディスパッチ条件に含めず
/// Tensor Core 経路内部のチューニング材料として扱う整理
/// （`docs/dispatch-rules-design.md` §3.2「TMA の扱い」）の実測根拠。
/// 大形状 2048/4096 で `mma.sync`/`ldmatrix`/`cp.async` パイプライン
/// （`CudaMmaGemm`）と基本 WMMA f16（opt。`CudaWmmaGemm`）を比較する。
///
/// 両カーネルとも `select_gemm_kernel`（第 1 層）上は同じ
/// `KernelKind::MatrixUnit` に写像される。第 2 層（`CudaGemmAuto` 内部。
/// `crates/backend-cuda/src/gemm_auto.rs::select_f16_matrix_unit_impl`）の
/// 判定ロジックは #1156 で整列形状・cc>=8.0 時に `CudaMmaGemm` を優先する
/// 設計どおり実装済みだが、`gemm_auto::MMA_PRIORITY_PRODUCTION_ENABLED`
/// （本番既定 `false`）でゲートされ本番では未有効化（#1160 の実測・承認
/// 待ち）。本テストは経路選択の妥当性ではなく「パイプライン差はカーネル
/// 内部チューニング材料であり、第 1 層の分岐条件にはしない」という設計
/// 判断の実測裏付けに限定する（計測ロジック自体は変更しない）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須。実測記録は docs/perf/dispatch-boundary-measurement.md"]
fn large_shape_mma_pipeline_vs_wmma_tflops_record() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

    let wmma_gemm = CudaWmmaGemm::new(&device).expect("CudaWmmaGemm::new (WMMA f16) must succeed");
    assert!(
        wmma_gemm.wmma_f16_opt_available(),
        "WMMA f16 opt kernel must be available on this ignored test runner (reason: {:?})",
        wmma_gemm.wmma_f16_opt_unavailable_reason()
    );
    // `CudaMmaGemm::new` は cc>=8.0 かつ NVRTC 搭載が必須（`gemm_mma.rs`
    // `MIN_COMPUTE_CAPABILITY_MAJOR = 8`）。本テストの `#[ignore]` 前提
    // （cc>=8.0・NVRTC 搭載）と一致するため、失敗時はテスト全体を
    // 失敗させる（サイレントスキップしない。`tests/gemm_mma.rs` の
    // 環境適応テストとは異なり、本テストは実測記録専用のため前提を
    // 断定してよい）。
    let mma_gemm = CudaMmaGemm::new(&device).expect(
        "CudaMmaGemm::new must succeed on the ignored test runner (cc>=8.0・NVRTC 搭載が前提)",
    );

    let config = MeasurementConfig::new(20, 20).expect("20/20 must satisfy TASK-8.1 minimums");

    for &dim in &LARGE_DIMS {
        let (m, n, k) = (dim, dim, dim);

        let mut rng = Xorshift64Star::new(0xC000 + u64::from(dim));
        let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
        let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

        // カーネル単体（H2D/D2H・バッファ確保を計測区間の外に置く）計測。
        // イシュー #1123 是正: 旧プロトコル（`run_f16` 合算計測から
        // 転送のみ計測を差し引く）は大容量バッファの per-call
        // アロケーション＋転送の二峰性により破綻するため、常駐バッファ
        // API（`upload_f16`／`alloc_output_f16`／`launch_f16`／
        // `synchronize`）による「launch → synchronize」区間のみの計測へ
        // 切り替える（本ファイル冒頭コメント参照）。
        let (a_wmma_dev, b_wmma_dev) = wmma_gemm
            .upload_f16(&a, &b)
            .expect("upload_f16 must succeed on CUDA-equipped test runner");
        let mut c_wmma_dev = wmma_gemm
            .alloc_output_f16(m, n)
            .expect("alloc_output_f16 must succeed on CUDA-equipped test runner");
        let wmma_kernel_only = bench_harness::run(&config, || {
            wmma_gemm
                .launch_f16(&a_wmma_dev, &b_wmma_dev, &mut c_wmma_dev, m, n, k)
                .expect("WMMA launch_f16 must succeed on CUDA-equipped test runner");
            wmma_gemm
                .synchronize()
                .expect("synchronize must succeed on CUDA-equipped test runner");
        })
        .expect("WMMA f16 kernel-only measurement must satisfy TASK-8.1 protocol");

        let (a_mma_dev, b_mma_dev) = mma_gemm
            .upload_f16(&a, &b)
            .expect("upload_f16 must succeed on CUDA-equipped test runner");
        let mut c_mma_dev = mma_gemm
            .alloc_output_f16(m, n)
            .expect("alloc_output_f16 must succeed on CUDA-equipped test runner");
        let mma_kernel_only = bench_harness::run(&config, || {
            mma_gemm
                .launch_f16(&a_mma_dev, &b_mma_dev, &mut c_mma_dev, m, n, k)
                .expect("mma.sync launch_f16 must succeed on CUDA-equipped test runner");
            mma_gemm
                .synchronize()
                .expect("synchronize must succeed on CUDA-equipped test runner");
        })
        .expect("mma.sync pipeline kernel-only measurement must satisfy TASK-8.1 protocol");

        let wmma_tflops = (flops(dim) / wmma_kernel_only.median_secs) / 1e12;
        let mma_tflops = (flops(dim) / mma_kernel_only.median_secs) / 1e12;

        let wmma_report = BenchReport::from_measurement(
            format!("gemm_wmma_f16_opt_kernel_only_dim{dim}"),
            "cuda",
            &wmma_kernel_only,
        )
        .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");
        let mma_report = BenchReport::from_measurement(
            format!("gemm_mma_f16_kernel_only_dim{dim}"),
            "cuda",
            &mma_kernel_only,
        )
        .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");

        println!(
            "dispatch_boundary_record_kernel_only dim={dim} path=wmma_f16_opt tflops={wmma_tflops:.3} \
             report={}",
            wmma_report
                .to_json()
                .expect("BenchReport::to_json must succeed for a validated report")
        );
        println!(
            "dispatch_boundary_record_kernel_only dim={dim} path=mma_sync_f16 tflops={mma_tflops:.3} \
             mma_over_wmma={:.3} report={}",
            mma_tflops / wmma_tflops,
            mma_report
                .to_json()
                .expect("BenchReport::to_json must succeed for a validated report")
        );
    }
}
