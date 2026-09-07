//! 3×TF32（split-single 法）`mma.sync`(m16n8k8)/`ldmatrix`/`cp.async` GEMM
//! （`CudaMmaTf32x3Gemm`。イシュー #1355・親ツリー #1354・承認元 #1338）
//! の API 健全性・数値一致テスト。
//!
//! `tests/gemm_mma_tf32.rs`（単発 TF32 経路。#801）と同じ設計方針:
//! CUDA 搭載・非搭載どちらの環境でも green になる（初期化・エラー型の
//! 契約確認・環境適応スモークの複合判定）。
//!
//! **判定方式**: 本イシュー時点では baseline 行・`ParityPath` 変種を
//! 追加しない（tolerance 定数・baseline の追加はユーザー承認必須。
//! `.claude/rules/coding-rust.md` テスト・ベンチ節）。よって `#[ignore]`
//! 実機テストは `fandhe_ai_backend_cpu::assert_parity`（厳密ゼロ fail）
//! で判定する。3×TF32 は単発 TF32 より高精度だが f32 SIMT とは
//! bit 一致しない（`.claude/rules/coding-rust.md` FMA 契約統一節の
//! 明示的例外。`kernels_mma_tf32x3.rs` 冒頭コメント参照）ため、厳密
//! ゼロ fail が成立するかどうかは GB10 実機実測（#1356 が引き継ぐ）まで
//! 未確定であり、本ファイルの実機テストは「未実測」のまま `#[ignore]`
//! 分離する。
//!
//! **`internal-diagnostics` feature 依存（Bugbot 指摘対応・PR #1390
//! 再修正）**: `launch_tf32x3_c_raw`／`download_f32_raw` は
//! `internal-diagnostics` feature（既定 off）限定の診断専用入口である。
//! 旧稿はファイル単位（`Cargo.toml` の `required-features`）で本
//! feature を要求していたが、それだと本ファイルの他のテスト（no-CUDA
//! 契約テスト・環境適応スモークテスト）まで既定ビルド（`cargo build
//! --no-cuda`／`cargo test --workspace` 等の feature 未指定コマンド）
//! から丸ごとスキップされてしまう（Bugbot Medium 指摘）。よって本
//! ファイル自体は `required-features` を持たず、上記 2 関数を直接呼ぶ
//! `launch_tf32x3_zero_dim_shape_is_noop_or_zero_fills_without_launch`
//! 1 関数だけを `#[cfg(feature = "internal-diagnostics")]` で個別に
//! ゲートする（同関数の doc コメント参照。`cargo test -p
//! fandhe-ai-backend-cuda --test gemm_mma_tf32x3 --all-features` で
//! フル実行できる）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaMmaTf32x3Gemm};

/// `CudaMmaTf32x3Gemm::new` は CUDA 非搭載環境で panic せず型付き
/// エラーを返す（`tests/gemm_mma_tf32.rs::
/// new_does_not_panic_and_returns_typed_result` と同型）。
#[test]
fn new_does_not_panic_and_returns_typed_result() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    match CudaMmaTf32x3Gemm::new(&device) {
        Ok(_gemm) => {
            // CUDA + cc>=8.0 + NVRTC あり環境: 3×TF32 mma.sync カーネルの
            // コンパイルが成功した。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            assert!(!detail.is_empty());
            assert!(detail.contains("compute capability"));
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaTf32x3Gemm::new: {other}"),
    }
}

/// 起動前検証（fail-closed）の実機非依存契約テスト: 整列非対応形状
/// （`n`/`k` が 4 の倍数でない）を `run_tf32x3` が `InvalidShape` で
/// 拒否することを確認する（`tests/gemm_mma_tf32.rs::
/// run_tf32_rejects_misaligned_shape_without_launch` と同型）。
#[test]
fn run_tf32x3_rejects_misaligned_shape_without_launch() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { .. }) | Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };
    let gemm = match CudaMmaTf32x3Gemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { .. }) | Err(CudaError::TensorCoreUnsupported { .. }) => {
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaTf32x3Gemm::new: {other}"),
    };

    // n=9 は 4 の倍数でない（cp.async 16B = f32 4 要素整列制約違反）。
    // m=4, k=4 のため A は m*k=16 要素（k*n=36 要素にすると
    // validate_gemm_dims の長さ検証が先に InvalidShape で早期リターンし、
    // 意図した整列制約検証パスへ到達しない。codex-review 指摘・PR #1400）。
    let err = gemm
        .run_tf32x3(&[0.0; 4 * 4], &[0.0; 4 * 9], 4, 9, 4)
        .expect_err("misaligned n must be rejected before any kernel launch");
    assert!(matches!(err, CudaError::InvalidShape { .. }));

    // k=9 も同様。
    let err = gemm
        .run_tf32x3(&[0.0; 4 * 9], &[0.0; 9 * 4], 4, 4, 9)
        .expect_err("misaligned k must be rejected before any kernel launch");
    assert!(matches!(err, CudaError::InvalidShape { .. }));
}

/// 起動前検証（fail-closed）の実機非依存契約テスト: 分離された公開
/// 起動 API（`upload_f32` → `launch_tf32x3` → `download_f32`）でも
/// 非有限入力が `run_tf32x3` と同じく `NonFiniteInput` で拒否される
/// ことを確認する（codex-review 指摘・PR #1400。回帰: 修正前は
/// `run_tf32x3` にしか検証がなく、分離 API 経由では
/// `upload_f32` がそのまま `+inf` を含む A をデバイスへ転送できて
/// しまい、`launch_tf32x3` まで到達しえた）。
#[test]
fn upload_f32_rejects_non_finite_input_in_separated_api() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { .. }) | Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };
    let gemm = match CudaMmaTf32x3Gemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { .. }) | Err(CudaError::TensorCoreUnsupported { .. }) => {
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaTf32x3Gemm::new: {other}"),
    };

    // m=n=k=4 の整列済み形状で A を全て +inf にする（指摘の再現条件）。
    let a = [f32::INFINITY; 16];
    let b = [1.0f32; 16];
    let err = gemm
        .upload_f32(&a, &b)
        .expect_err("non-finite lhs must be rejected by upload_f32 before any H2D transfer");
    assert!(matches!(err, CudaError::NonFiniteInput { .. }));

    // rhs 側の非有限値も同様に拒否される。
    let a = [1.0f32; 16];
    let b = [f32::NAN; 16];
    let err = gemm
        .upload_f32(&a, &b)
        .expect_err("non-finite rhs must be rejected by upload_f32 before any H2D transfer");
    assert!(matches!(err, CudaError::NonFiniteInput { .. }));
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
/// CUDA 非搭載環境では `DriverUnavailable`／`NvrtcUnavailable`／
/// `TensorCoreUnsupported` の型のみ確認して早期 return する。CUDA+NVRTC
/// ありの環境では `run_tf32x3` が `Ok` で成功することのみ確認する
/// （数値一致判定は本ファイル冒頭コメントのとおり `#[ignore]` 実機
/// テストへ委ねる）。
#[test]
fn run_tf32x3_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let gemm = match CudaMmaTf32x3Gemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaTf32x3Gemm::new: {other}"),
    };

    // 64x64x64: ブロックタイル（MMA_TF32X3_BM=64/MMA_TF32X3_BN=64）
    // ちょうど 1 個・K タイル（MMA_TF32X3_BK=16）を 4 段跨ぐ最小規模の
    // 網羅形状。
    let a = vec![0.5f32; 64 * 64];
    let b = vec![0.25f32; 64 * 64];
    let c = gemm
        .run_tf32x3(&a, &b, 64, 64, 64)
        .expect("CudaMmaTf32x3Gemm::run_tf32x3 must succeed on a CUDA+NVRTC test runner");
    assert_eq!(c.len(), 64 * 64);
}

/// m==0／n==0／k==0 で `run_tf32x3` を呼んでも CUDA 起動そのものが
/// 発生せず、ゼロ次元形状の契約どおりの結果を返すことを実機で確認する
/// （`tests/gemm_mma_tf32.rs::mma_tf32_zero_dim_shape_returns_empty_
/// without_launch` と同型）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_tf32x3_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm =
        CudaMmaTf32x3Gemm::new(&device).expect("3xTF32 mma.sync kernel compilation must succeed");

    let c = gemm
        .run_tf32x3(&[], &[0.0f32; 16], 0, 4, 4)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_tf32x3(&[0.0f32; 16], &[], 4, 0, 4)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_tf32x3(&[], &[], 4, 4, 0)
        .expect("k==0 must zero-fill C, not fail as a driver launch error");
    assert_eq!(c, vec![0.0f32; 16]);
}

/// `launch_tf32x3`（直接起動 safe API）も `run_tf32x3` と同じ no-op
/// 形状契約を守ることを実機で確認する（`tests/gemm_mma_tf32.rs::
/// launch_tf32_zero_dim_shape_is_noop_or_zero_fills_without_launch` と
/// 同型）。
///
/// **`internal-diagnostics` feature 限定（PR #1390 マージ時是正）**:
/// 本テストは `device.stream()` を直接呼ぶ（下記コメント参照）。
/// `crate::device::CudaDevice::stream` は codex-review P0 指摘対応
/// （イシュー #1349）で既定ビルド（同 feature 無効）では `pub(crate)`
/// に絞られており、このテストファイル自体は他のテスト（環境適応
/// スモーク等）を通常 CI（feature 未指定）でも実行させるため
/// `required-features` によるファイル単位ゲートを使わない。かわりに
/// このテスト関数だけを `internal-diagnostics` feature（`cargo test
/// --workspace --all-features`。CI の test ジョブ・`make test` が使う
/// コマンド）限定でコンパイルする（`device.rs::CudaDevice::context`
/// doc コメント参照。他の diagnostics 専用テストファイルと同じ契約）。
#[cfg(feature = "internal-diagnostics")]
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn launch_tf32x3_zero_dim_shape_is_noop_or_zero_fills_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm =
        CudaMmaTf32x3Gemm::new(&device).expect("3xTF32 mma.sync kernel compilation must succeed");

    let inputs = gemm
        .upload_f32(&[], &[0.0f32; 16])
        .expect("upload_f32 must succeed");
    let mut c_dev = gemm
        .alloc_output_f32(0, 4)
        .expect("alloc_output_f32 must succeed");
    gemm.launch_tf32x3(&inputs, &mut c_dev, 0, 4, 4)
        .expect("launch_tf32x3 must succeed as a no-op for m==0");
    assert_eq!(gemm.download_f32(&c_dev).unwrap(), Vec::<f32>::new());

    let inputs = gemm
        .upload_f32(&[0.0f32; 16], &[])
        .expect("upload_f32 must succeed");
    let mut c_dev = gemm
        .alloc_output_f32(4, 0)
        .expect("alloc_output_f32 must succeed");
    gemm.launch_tf32x3(&inputs, &mut c_dev, 4, 0, 4)
        .expect("launch_tf32x3 must succeed as a no-op for n==0");
    assert_eq!(gemm.download_f32(&c_dev).unwrap(), Vec::<f32>::new());

    let inputs = gemm.upload_f32(&[], &[]).expect("upload_f32 must succeed");
    // c_dev を未初期化のゼロ以外の値で事前汚染し、k==0 の zero-fill 契約
    // が実際にゼロで上書きすることを確認する（`ValidatedTf32x3Inputs`
    // 経由に限定した `upload_f32` はもう任意バッファの生成に流用でき
    // ないため、ここは本経路が明示的に許容する公開 API
    // `device.stream().clone_htod()` を直接使う。codex-review 指摘・
    // PR #1400 スレッド PRRT_kwDOTuUCJc6f0YV_ が名指ししたのと同じ
    // 経路だが、ここではテスト用の C バッファ生成に限定して使用して
    // おり、`launch_tf32x3` へは A/B として渡らない）。
    let mut c_dev = device
        .stream()
        .clone_htod(&[9.0f32; 16])
        .expect("uploading a pre-populated c buffer must succeed");
    gemm.launch_tf32x3_c_raw(&inputs, &mut c_dev, 4, 4, 0)
        .expect("launch_tf32x3_c_raw must succeed and zero-fill c_dev for k==0");
    assert_eq!(gemm.download_f32_raw(&c_dev).unwrap(), vec![0.0f32; 16]);
}

/// 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須の形状網羅
/// テスト。数値一致は `fandhe_ai_backend_cpu::assert_parity`（厳密ゼロ
/// fail）で判定する（本ファイル冒頭コメント「判定方式」参照。baseline
/// 行は追加しない）。**本イシュー時点では GB10 未実測**（#1356 が実測・
/// 採否判断を引き継ぐ）。厳密ゼロ fail が成立しない場合は本テストが
/// FAIL するが、それ自体が #1356 の入力になる（本テストの合否を本 PR
/// の完了条件とはしない。実装計画 §5 参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn mma_tf32x3_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm =
        CudaMmaTf32x3Gemm::new(&device).expect("3xTF32 mma.sync kernel compilation must succeed");

    // 16x8x8（1 mma.sync 呼び出しちょうど）・64x64x64（ブロックタイル
    // ちょうど 1 個）・128x128x128・512x512x512・非正方 60x68x36。
    let shapes: &[(u32, u32, u32, u64)] = &[
        (16, 8, 8, 5001),
        (64, 64, 64, 5002),
        (128, 128, 128, 5003),
        (512, 512, 512, 5004),
        (60, 68, 36, 5005),
    ];

    for &(m, n, k, seed) in shapes {
        let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
        let a = rng.fill_vec((m as usize) * (k as usize));
        let b = rng.fill_vec((k as usize) * (n as usize));

        let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
        fandhe_ai_backend_cpu::matmul_reference_fma(
            &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
        )
        .expect("matmul_reference_fma shape validation must pass for well-formed input");

        let c_gpu = gemm.run_tf32x3(&a, &b, m, n, k).expect(
            "CudaMmaTf32x3Gemm::run_tf32x3 must succeed on a compute capability >= 8.0 test runner",
        );

        fandhe_ai_backend_cpu::assert_parity(
            &format!("mma_tf32x3 m={m} n={n} k={k}"),
            &c_gpu,
            &c_ref,
        );
    }
}

/// K 大のストレスケース（`tests/gemm_mma_tf32.rs::mma_tf32_k4096_stress`
/// と同じ形状。PoC-v2-3 の M=N=K=4096 と揃える）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn mma_tf32x3_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm =
        CudaMmaTf32x3Gemm::new(&device).expect("3xTF32 mma.sync kernel compilation must succeed");

    let (m, n, k) = (4096u32, 4096u32, 4096u32);
    let mut rng = bench_harness::rng::Xorshift64Star::new(9001);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed input");

    let c_gpu = gemm.run_tf32x3(&a, &b, m, n, k).expect(
        "CudaMmaTf32x3Gemm::run_tf32x3 must succeed on a compute capability >= 8.0 test runner",
    );

    fandhe_ai_backend_cpu::assert_parity("mma_tf32x3 k4096 stress", &c_gpu, &c_ref);
}
