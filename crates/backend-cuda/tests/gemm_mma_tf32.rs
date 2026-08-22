//! TF32 `mma.sync`(m16n8k8)/`ldmatrix`/`cp.async` GEMM（`CudaMmaTf32Gemm`。
//! イシュー #801）の API 健全性・数値一致回帰テスト。
//!
//! `tests/gemm_mma.rs`（f16 mma.sync 経路）・`tests/cpu_cuda_mma_parity.rs`
//! と同じ設計方針: CUDA 搭載・非搭載どちらの環境でも green になる
//! （初期化・エラー型の契約確認・環境適応スモークの複合判定）。本実装
//! セッションの実行環境は CUDA driver はあるが NVRTC が無く、
//! `CudaMmaTf32Gemm::new` は `CudaError::NvrtcUnavailable` を返す分岐を
//! 必ず通る（`kernels_mma_tf32.rs` 冒頭コメント「検証状態」参照）。
//!
//! **重要（#852 実機再実測結果）**: 本ファイルの `#[ignore]` 実機テストは
//! #852 で DGX Spark GB10 実機（driver 580.159.03・CUDA 13.0 V13.0.88）に
//! て実行済み。`mma_tf32_zero_dim_shape_returns_empty_without_launch`
//! （#853 で一時的に環境適応型〔`#[ignore]` なし〕へ変換したが、実機
//! でのみ再現する `InvalidShape` panic を検出できないため `#[ignore]` を
//! 復元済み。`.claude/rules/coding-rust.md`「実機依存テストは `#[ignore]`
//! で分離」の原則に従う）・`launch_tf32_zero_dim_shape_is_noop_or_zero_fills_without_launch`
//! は pass。`mma_tf32_matches_reference_across_shapes`・
//! `mma_tf32_k4096_stress` は #839 時点の機能欠陥（A フラグメント象限
//! マッピング誤り。`kernels_mma_tf32.rs::LDSM_A_FRAG` 参照）修正後も
//! FAIL が残る。この残存 FAIL の原因は TF32 丸め誤差・機能欠陥のいずれ
//! とも確定していない（`wmma_tf32` との GPU-GPU 相互一致誤差が CPU 参照
//! 比較より小さいことは、両経路が共有する TF32 丸め誤差成分の相殺でも
//! 説明でき、TF32 丸め誤差説への反証にはならない。
//! `docs/perf/cuda-gemm-mma-tf32-ab.md` §8.4 に実測ログ・訂正経緯を記録）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaMmaTf32Gemm};

/// 決定的シードで A・B（f32）を生成し、CPU 参照実装と `run_tf32` の出力を
/// `fandhe_ai_backend_cpu::assert_parity`（統一複合判定「相対誤差 1e-3 未満 または
/// 絶対誤差 1e-5 未満」の唯一の実体）で照合する
/// （`tests/gemm_wmma_tf32_staged.rs::assert_wmma_tf32_staged_parity` と
/// 同一手順）。
fn assert_mma_tf32_parity(
    gemm: &CudaMmaTf32Gemm,
    context: &str,
    seed: u64,
    m: u32,
    n: u32,
    k: u32,
) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");

    let c_gpu = gemm.run_tf32(&a, &b, m, n, k).expect(
        "CudaMmaTf32Gemm::run_tf32 must succeed on a compute capability >= 8.0 test runner",
    );

    fandhe_ai_backend_cpu::assert_parity(context, &c_gpu, &c_ref);
}

/// `CudaMmaTf32Gemm::new` は CUDA 非搭載環境で panic せず型付きエラーを
/// 返す（`tests/gemm_mma.rs::new_does_not_panic_and_returns_typed_result`
/// と同型）。
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

    match CudaMmaTf32Gemm::new(&device) {
        Ok(_gemm) => {
            // CUDA + cc>=8.0 + NVRTC あり環境: TF32 mma.sync カーネルの
            // コンパイルが成功した。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            assert!(!detail.is_empty());
            assert!(detail.contains("compute capability"));
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaTf32Gemm::new: {other}"),
    }
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
///
/// **注意**: この分岐は「コンパイル・起動が成功して複合判定を通過した」
/// ことのみを green の条件とする。`run_tf32` の `Err` はここでは早期
/// return せず `panic` させる（CUDA+NVRTC 非搭載の通常 CI では
/// `DriverUnavailable` 分岐で早期 return し green のまま。コンパイル・
/// 起動失敗を誤って parity 通過とみなす退行を防ぐため、CUDA+NVRTC
/// 環境に限っては厳格化する。`.claude/rules/coding-rust.md` テスト・
/// ベンチ節）。**#852 実機再実測（GB10）**: 64x64x64 形状でも原因未特定の
/// 欠陥に起因する FAIL が残る（`fail_count=666/4096`。
/// `docs/perf/cuda-gemm-mma-tf32-ab.md` §8.4）。本経路は本番未結線・
/// 凍結継続のため通常 CI（GPU 非搭載）には影響しないが、将来 GPU 搭載
/// CI／実機セッションでこのテストを走らせると FAIL する状態が残って
/// いる点に注意（凍結解除判断の一部として #835 系で扱う）。
#[test]
fn mma_tf32_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let gemm = match CudaMmaTf32Gemm::new(&device) {
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

    // 64x64x64: ブロックタイル（MMA_TF32_BM=64/MMA_TF32_BN=64）ちょうど 1
    // 個・K タイル（MMA_TF32_BK=16）を 4 段跨ぐ最小規模の網羅形状。
    assert_mma_tf32_parity(&gemm, "smoke 64x64x64", 1, 64, 64, 64);
}

/// `CudaMmaTf32Gemm::new` 経由の `TensorCoreUnsupported` が `gemm_mma.rs`
/// の f16 経路と同一の cc>=8.0 メッセージを持つことを実機に依存せず
/// 確認する（`check_min_compute_capability` を再利用しているため
/// 文言も共有される契約）。
#[test]
fn tensor_core_unsupported_display_mentions_compute_capability_8() {
    use std::error::Error;

    let err = CudaError::TensorCoreUnsupported {
        detail: "mma.sync/ldmatrix/cp.async path requires compute capability >= 8.0, \
                  but device reports 7.5"
            .to_string(),
    };
    let msg = err.to_string();
    assert!(!msg.is_empty());
    assert!(msg.contains("compute capability"));
    assert!(err.source().is_none());
}

/// m==0／n==0／k==0 で `run_tf32` を呼んでも CUDA 起動そのものが発生せず、
/// ゼロ次元形状の契約どおりの結果を返すことを実機で確認する
/// （`tests/gemm_mma.rs::mma_f16_zero_dim_shape_returns_empty_without_launch`
/// と同型）。
///
/// **#853 是正の再修正（codex-review P1 指摘対応）**: 一度は環境適応型
/// （`#[ignore]` を外し `DriverUnavailable`/`NvrtcUnavailable` 等で
/// 早期 return する形）へ変換したが、この形では通常 CI（CUDA 非搭載）で
/// 実際には何も検証せず素通りするだけになり、AGENTS.md・
/// `.claude/rules/coding-rust.md`「実機依存テストは `#[ignore]` で分離
/// する」規約に反する（実機依存テストを通常 CI 対象化した扱いになる）
/// うえ、実機上でも `Driver(_)`/`NvrtcUnavailable`/`TensorCoreUnsupported`
/// を成功扱いにしてしまいドライバ異常・NVRTC 構成不備を回帰として
/// 検出できない。よって `#[ignore]` を復元し、`mma_f16_zero_dim_shape_
/// returns_empty_without_launch` と同じ `expect` ベースの実機専用形へ
/// 戻す。#389 前例を踏まえたテスト側のバッファ長是正（`validate_gemm_
/// dims` の検証順序・契約自体は変更しない）は維持する。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_tf32_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm =
        CudaMmaTf32Gemm::new(&device).expect("TF32 mma.sync kernel compilation must succeed");

    // `validate_gemm_dims`（`gemm_mma_tf32.rs::run_tf32` が m==0/n==0 の
    // 早期 return より先に必ず呼ぶ）は no-op 形状でも a.len()==m*k・
    // b.len()==k*n の厳密一致を要求する（f16 版 `tests/gemm_mma.rs::
    // mma_f16_zero_dim_shape_returns_empty_without_launch` と同一契約・
    // #389 の教訓を踏襲）。よって m=0 の呼び出しは b（k*n=16 要素）を、
    // n=0 の呼び出しは a（m*k=16 要素）を満たす必要がある（#852 で是正。
    // `validate_gemm_dims` 側の検証順序・契約は変更しない）。
    let c = gemm
        .run_tf32(&[], &[0.0f32; 16], 0, 4, 4)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_tf32(&[0.0f32; 16], &[], 4, 0, 4)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_tf32(&[], &[], 4, 4, 0)
        .expect("k==0 must zero-fill C, not fail as a driver launch error");
    assert_eq!(c, vec![0.0f32; 16]);
}

/// `launch_tf32`（直接起動 safe API）も `run_tf32` と同じ no-op 形状契約
/// （PR #823 codex-review P1 是正）を守ることを実機で確認する:
/// `m==0 || n==0` はカーネル起動せず成功、`k==0` は `c_dev` を明示的に
/// ゼロ化してから成功する（`gemm_mma_tf32.rs::launch_tf32` ドキュメン
/// テーションコメント参照）。`run_tf32` は内部で `launch_tf32` を
/// 呼ばずに no-op を処理するため、この契約は `launch_tf32` を直接
/// 呼ばない限り検証できない。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn launch_tf32_zero_dim_shape_is_noop_or_zero_fills_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm =
        CudaMmaTf32Gemm::new(&device).expect("TF32 mma.sync kernel compilation must succeed");

    // m==0: グリッド次元がゼロになり起動自体が no-op。c_dev は 0 要素。
    // `launch_tf32` は no-op 早期 return 前に `validate_gemm_dims`
    // （a.len()==m*k・b.len()==k*n の厳密一致要求）を通すため、m==0 でも
    // b は k*n=16 要素必要（`tests/gemm_mma.rs::
    // mma_f16_zero_dim_shape_returns_empty_without_launch` と同じ契約。
    // PR #823 codex-review 指摘是正: 空バッファのままだと b の長さ不一致で
    // no-op 契約の検証前に `InvalidShape` になる）。
    let (a_dev, b_dev) = gemm
        .upload_f32(&[], &[0.0f32; 16])
        .expect("upload_f32 must succeed");
    let mut c_dev = gemm
        .alloc_output_f32(0, 4)
        .expect("alloc_output_f32 must succeed");
    gemm.launch_tf32(&a_dev, &b_dev, &mut c_dev, 0, 4, 4)
        .expect("launch_tf32 must succeed as a no-op for m==0");
    assert_eq!(gemm.download_f32(&c_dev).unwrap(), Vec::<f32>::new());

    // n==0: 同様。a は m*k=16 要素必要。
    let (a_dev, b_dev) = gemm
        .upload_f32(&[0.0f32; 16], &[])
        .expect("upload_f32 must succeed");
    let mut c_dev = gemm
        .alloc_output_f32(4, 0)
        .expect("alloc_output_f32 must succeed");
    gemm.launch_tf32(&a_dev, &b_dev, &mut c_dev, 4, 0, 4)
        .expect("launch_tf32 must succeed as a no-op for n==0");
    assert_eq!(gemm.download_f32(&c_dev).unwrap(), Vec::<f32>::new());

    // k==0: カーネルを起動せず c_dev を明示的にゼロ化する。非ゼロで
    // 初期化した c_dev を渡し、呼び出し前の残存内容が反映されないこと
    // （GEMM 契約: K ループが空でも結果はゼロ行列）を確認する。
    let (a_dev, b_dev) = gemm.upload_f32(&[], &[]).expect("upload_f32 must succeed");
    // `upload_f32` は A・B いずれも `clone_htod` を呼ぶだけの汎用 H2D
    // 転送であるため（`gemm_mma_tf32.rs::upload_f32` 参照）、c 用の
    // 事前非ゼロデータ転送にも同じ経路を流用してよい。第 1 戻り値
    // （本来は a_dev 用）を「呼び出し前に非ゼロ値が入った c_dev」として
    // 使う。
    let (mut c_dev, _unused) = gemm
        .upload_f32(&[9.0f32; 16], &[])
        .expect("uploading a pre-populated c buffer must succeed");
    gemm.launch_tf32(&a_dev, &b_dev, &mut c_dev, 4, 4, 0)
        .expect("launch_tf32 must succeed and zero-fill c_dev for k==0");
    assert_eq!(gemm.download_f32(&c_dev).unwrap(), vec![0.0f32; 16]);
}

/// 起動前検証（fail-closed）の実機非依存契約テスト: 整列非対応形状
/// （`n`/`k` が 4 の倍数でない）を `run_tf32` が `InvalidShape` で拒否
/// することを確認する。`CudaMmaTf32Gemm::new` の成功可否に依存しない
/// ため CUDA 非搭載環境でも到達しうる検証だが、`new` 自体が
/// `NvrtcUnavailable`/`TensorCoreUnsupported` を返す環境では検証対象の
/// `gemm` を構築できないため、その場合は早期 return する。
#[test]
fn run_tf32_rejects_misaligned_shape_without_launch() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { .. }) | Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };
    let gemm = match CudaMmaTf32Gemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { .. }) | Err(CudaError::TensorCoreUnsupported { .. }) => {
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaTf32Gemm::new: {other}"),
    };

    // n=9 は 4 の倍数でない（cp.async 16B = f32 4 要素整列制約違反）。
    let err = gemm
        .run_tf32(&[0.0; 4 * 9], &[0.0; 4 * 9], 4, 9, 4)
        .expect_err("misaligned n must be rejected before any kernel launch");
    assert!(matches!(err, CudaError::InvalidShape { .. }));

    // k=9 も同様。
    let err = gemm
        .run_tf32(&[0.0; 4 * 9], &[0.0; 9 * 4], 4, 4, 9)
        .expect_err("misaligned k must be rejected before any kernel launch");
    assert!(matches!(err, CudaError::InvalidShape { .. }));
}

/// 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須の形状網羅
/// テスト。受け入れ条件 1〜2 項（cp.async 多段パイプライン動作・数値
/// 一致）の本体。#852 で実機再実測済み（残存 FAIL は
/// `docs/perf/cuda-gemm-mma-tf32-ab.md` §8 参照）
/// （#802 が実機で実行・確認する。本ファイル冒頭コメント「重要」参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn mma_tf32_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm =
        CudaMmaTf32Gemm::new(&device).expect("TF32 mma.sync kernel compilation must succeed");

    let cases: &[(u32, u32, u32)] = &[
        (16, 8, 8),   // 1 mma.sync 呼び出しちょうど（M=16,N=8,K=8）
        (64, 64, 64), // ブロックタイルちょうど 1 個
        (128, 128, 128),
        (512, 512, 512),
        // ブロックタイル・K タイル非倍数だが cp.async 4 要素整列条件は
        // 満たす形状（60/68/36 等はいずれも 4 の倍数）。
        (60, 68, 36),
        (68, 60, 20),
        // M 端・K タイル端を踏む非正方形状（96×68×72）。
        (96, 68, 72),
        // 非正方。
        (64, 96, 256),
        // 極小（4 の倍数の最小非自明形状）。
        (4, 4, 4),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_mma_tf32_parity(&gemm, &context, 5000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（`tests/cpu_cuda_mma_parity.rs`
/// `mma_f16_k4096_stress` と同じ形状。PoC-v2-3 の M=N=K=4096 と揃える）。
/// #852 で実機再実測済み（本ファイル冒頭コメント「重要」参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn mma_tf32_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm =
        CudaMmaTf32Gemm::new(&device).expect("TF32 mma.sync kernel compilation must succeed");
    assert_mma_tf32_parity(&gemm, "K=4096 stress", 9001, 4096, 4096, 4096);
}
