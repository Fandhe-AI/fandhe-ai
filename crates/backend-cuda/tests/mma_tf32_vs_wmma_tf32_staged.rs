//! `CudaMmaTf32Gemm::run_tf32`（イシュー #801。生 `mma.sync`(m16n8k8) 経路）
//! と `CudaGemm::run_wmma_tf32`（既存 WMMA C++ API staged 経路。イシュー
//! #500）を、同一入力で `backend_cpu::assert_parity` により相互照合する。
//!
//! 受け入れ条件 3 項（「既存 wmma_tf32 staged と同一入力で数値一致」）の
//! 本体。staged 側を参照値として扱う（`assert_parity` の `expected` 引数）。
//! `CudaGemm::run_wmma_tf32` は起動前整列条件（`n%4==0 && k%4==0`）を
//! 満たす形状であれば `CudaGemm::new` 時点で staged カーネルのコンパイル・
//! ロードに成功している限り自動的に staged 経路を選ぶため（`gemm.rs` 3
//! 段選択方針）、本ファイルの対象形状はすべて 4 の倍数に揃える
//! （`tests/gemm_wmma_tf32_staged.rs` と同じ整列前提）。
//!
//! **重要（#852 実機再実測結果）**: `#[ignore]` 実機テストは #852 で DGX
//! Spark GB10 実機（driver 580.159.03・CUDA 13.0 V13.0.88）にて実行済み。
//! #839 時点の機能欠陥（`kernels_mma_tf32.rs::LDSM_A_FRAG` の A フラグ
//! メント象限マッピング誤り）修正後、両経路の乖離は
//! `mean_rel_err` オーダーで劇的に縮小した（512x512x512:
//! `fail_count=7/262144・mean_rel_err=4.8e-6`）が僅かに FAIL が残る。
//! この GPU-GPU 相互比較の残差は `mma_tf32` 単独の CPU 参照比較
//! （`mean_rel_err` は 200 倍大きい）より大幅に小さいが、両経路が共有する
//! TF32 丸め誤差成分は GPU-GPU 相互比較では相殺されうるため、この事実
//! だけから残存 FAIL を TF32 丸め誤差では説明できないと断定する根拠には
//! ならない。残存 FAIL の原因は TF32 丸め誤差・機能欠陥のいずれとも
//! 未確定のままである（`docs/perf/cuda-gemm-mma-tf32-ab.md` §8.4 に
//! 分析を記録）。
//!
//! **実機依存の分離**: `tests/gemm_wmma_tf32_staged.rs` と同じ分岐
//! パターン（環境適応スモークのみ通常 CI で実行、CUDA/NVRTC 非搭載・
//! staged カーネル未対応環境では早期 return で green）。

use backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaMmaTf32Gemm};

/// 決定的シードで A・B（f32）を生成し、`CudaGemm::run_wmma_tf32`
/// （staged 経路。整列条件を満たす形状であれば自動選択される）を参照値
/// として `CudaMmaTf32Gemm::run_tf32` の出力と `assert_parity` で照合
/// する。
fn assert_mma_tf32_matches_wmma_tf32_staged(
    mma: &CudaMmaTf32Gemm,
    wmma: &CudaGemm,
    context: &str,
    seed: u64,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let c_staged = wmma
        .run_wmma_tf32(&a, &b, m, n, k)
        .expect("CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test runner");
    let c_mma = mma.run_tf32(&a, &b, m, n, k).expect(
        "CudaMmaTf32Gemm::run_tf32 must succeed on a compute capability >= 8.0 test runner",
    );

    backend_cpu::assert_parity(context, &c_mma, &c_staged);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
/// `tests/gemm_wmma_tf32_staged.rs::wmma_tf32_staged_parity_smoke_env_adaptive`
/// と同じ分岐パターン。
#[test]
fn mma_tf32_matches_wmma_tf32_staged_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let wmma = match CudaGemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
    };
    if !wmma.wmma_tf32_staged_available() {
        // staged カーネル未対応環境（`<mma.h>` 未解決・cc<8.0 等）。
        // opt／基本版のみが利用可能な環境は本比較の対象外。
        return;
    }

    let mma = match CudaMmaTf32Gemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaTf32Gemm::new: {other}"),
    };

    // 64x64x64: 両経路のブロックタイル（staged 64x64・mma.sync 64x64）が
    // ちょうど 1 個ずつになる整列形状（4 の倍数）。
    assert_mma_tf32_matches_wmma_tf32_staged(&mma, &wmma, "smoke 64x64x64", 1, 64, 64, 64);
}

/// 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須の形状網羅
/// テスト。受け入れ条件 3 項の本体。#852 で実機再実測済み（本ファイル
/// 冒頭コメント「重要」参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn mma_tf32_matches_wmma_tf32_staged_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let wmma = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    assert!(
        wmma.wmma_tf32_staged_available(),
        "staged kernel must be available on this ignored test runner (reason: {:?})",
        wmma.wmma_tf32_staged_unavailable_reason()
    );
    let mma = CudaMmaTf32Gemm::new(&device).expect("TF32 mma.sync kernel compilation must succeed");

    let cases: &[(u32, u32, u32)] = &[
        (64, 64, 64),
        (128, 128, 128),
        (512, 512, 512),
        (60, 68, 36),
        (68, 60, 20),
        (96, 68, 72),
        (64, 96, 256),
        (4, 4, 4),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_mma_tf32_matches_wmma_tf32_staged(&mma, &wmma, &context, 6000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（M=N=K=4096。`tests/gemm_wmma_tf32_staged.rs` の
/// K4096 ストレスケースと揃える）。#852 で実機再実測済み。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn mma_tf32_matches_wmma_tf32_staged_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let wmma = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    assert!(
        wmma.wmma_tf32_staged_available(),
        "staged kernel must be available on this ignored test runner (reason: {:?})",
        wmma.wmma_tf32_staged_unavailable_reason()
    );
    let mma = CudaMmaTf32Gemm::new(&device).expect("TF32 mma.sync kernel compilation must succeed");
    assert_mma_tf32_matches_wmma_tf32_staged(&mma, &wmma, "K=4096 stress", 9002, 4096, 4096, 4096);
}
