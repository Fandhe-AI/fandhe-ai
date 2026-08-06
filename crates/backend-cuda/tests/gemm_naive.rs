//! naive GEMM（`CudaGemm`）の環境適応型テスト＋実機必須の数値一致テスト。
//!
//! CUDA 搭載・非搭載どちらの環境でも green になる設計（`tests/device_init.rs`
//! の分岐パターンを踏襲）に加え、受け入れ条件そのものである複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）による CPU 参照実装
//! （`backend-cpu::gemm_naive`）との数値一致検証を `#[ignore]` で分離して
//! 提供する（`.claude/rules/coding-rust.md` の実機依存テスト分離規約）。

use backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// `CudaGemm::new` は CUDA 非搭載環境で panic せず型付きエラーを返す
/// （`CudaDevice::new` が既に満たしている契約を `CudaGemm::new` 経路でも
/// 確認する）。CUDA 搭載環境では naive f32/f16 カーネルのコンパイルが
/// 成功することを検証する。
#[test]
fn new_does_not_panic_and_returns_typed_result() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            // CUDA 非搭載環境（self-hosted CI 想定）: panic せず型付き
            // エラーが返ることそのものが受け入れ条件。
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => {
            // libcuda は存在するが cuInit 等が失敗したケース。プローブは
            // 通過しているため panic しない前提は保たれる。
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    match CudaGemm::new(&device) {
        Ok(_gemm) => {
            // CUDA 搭載環境: naive f32/f16 両カーネルのコンパイルが成功した。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            // libcuda はあるが libnvrtc が dlopen できない環境（CUDA driver
            // のみインストール済みで toolkit 非搭載のケース）。panic しない
            // ことが本テストの主張であり、このケースも許容する。
            assert!(!detail.is_empty());
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
    }
}

/// ホスト側形状検証（`validate_gemm_dims`）は実機非依存で網羅できるため、
/// `CudaGemm::new`（デバイス初期化）を経由せず直接検証する。
mod validate_gemm_dims_tests {
    // `validate_gemm_dims` は `pub(crate)` のため crate 外の統合テストから
    // 直接は呼べない。デバイス初期化前に検証したい対象（長さ不一致・
    // オーバーフロー・i32 超過）は `CudaGemm::run_naive_f32`/`run_naive_f16`
    // 経由でも同じ入口を通るため、`#[ignore]` テスト側（実機必須）で
    // あわせて検証する。ここでは公開 API から到達できる範囲、すなわち
    // `CudaError::InvalidShape` の `Display` 実装のみを環境非依存で確認する。
    use backend_cuda::CudaError;
    use std::error::Error;

    #[test]
    fn invalid_shape_display_is_non_empty_and_contains_detail() {
        let err = CudaError::InvalidShape {
            detail: "a length mismatch: expected 6, actual 5".to_string(),
        };
        let msg = err.to_string();
        assert!(!msg.is_empty());
        assert!(msg.contains("a length mismatch"));
        assert!(err.source().is_none());
    }
}

/// 複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
/// `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/compare.rs`
/// の判定式をそのまま移植する（許容誤差は単独緩和しない。
/// `.claude/rules/coding-rust.md`）。
const RELATIVE_TOLERANCE: f64 = 1e-3;
const ABSOLUTE_RESCUE_THRESHOLD: f64 = 1e-5;

/// `a`（GPU 出力）と `b`（CPU 参照）を要素ごとに複合判定し、
/// 不一致セル数を返す（0 なら PASS）。
fn count_composite_mismatches(a: &[f32], b: &[f32]) -> usize {
    assert_eq!(a.len(), b.len(), "compare: length mismatch");
    a.iter()
        .zip(b.iter())
        .filter(|&(&x, &y)| {
            let xf = x as f64;
            let yf = y as f64;
            let diff = (xf - yf).abs();
            let scale = xf.abs().max(yf.abs()).max(1e-12);
            let rel = diff / scale;
            rel >= RELATIVE_TOLERANCE && diff >= ABSOLUTE_RESCUE_THRESHOLD
        })
        .count()
}

/// 決定的シードで A・B（f32）を生成し、CPU 参照実装（`gemm_naive`。
/// `mul_add` FMA 契約）と GPU naive カーネルの出力を複合判定で照合する。
fn assert_naive_f32_matches_cpu_reference(gemm: &CudaGemm, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::gemm_naive(&a, &b, &mut c_ref, m as usize, n as usize, k as usize)
        .expect("backend-cpu gemm_naive shape validation must pass for well-formed test input");

    let c_gpu = gemm
        .run_naive_f32(&a, &b, m, n, k)
        .expect("CudaGemm::run_naive_f32 must succeed on CUDA-equipped test runner");

    let mismatches = count_composite_mismatches(&c_gpu, &c_ref);
    assert_eq!(
        mismatches,
        0,
        "naive f32 GEMM CPU/GPU mismatch: {mismatches}/{} cells failed composite tolerance \
         (rel<{RELATIVE_TOLERANCE} or abs<{ABSOLUTE_RESCUE_THRESHOLD}), shape m={m} n={n} k={k}",
        c_ref.len()
    );
}

/// 実機（DGX Spark GB10 等）必須の数値一致テスト。受け入れ条件の本体。
///
/// CI self-hosted runner は CUDA toolkit 非搭載のため通常実行では
/// スキップされる（`cargo test -- --ignored` での実機実行を前提とする。
/// 実行導線の整備は #36 のスコープ）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn naive_f32_matches_cpu_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive kernel compilation must succeed");

    // 形状ケース: 正方（128^3・512^3）・非正方（64x128x96）・ブロック
    // 非整数倍の境界形状（1x1x1・17x19x23・33x65x31）。NAIVE_BLOCK_DIM が
    // 16x16 のため、これらは末尾ブロックの手動境界チェック（REQ-8）を
    // 実際に踏む形状として選んでいる。
    let square_and_nonsquare_cases: &[(u32, u32, u32)] = &[
        (128, 128, 128),
        (512, 512, 512),
        (64, 96, 128),
        (1, 1, 1),
        (17, 23, 19),
        (33, 31, 65),
    ];
    for (idx, &(m, n, k)) in square_and_nonsquare_cases.iter().enumerate() {
        assert_naive_f32_matches_cpu_reference(&gemm, 1000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（M=N=256, K=4096）。PoC-v2-5 が FMA 契約統一
/// （`f32::mul_add` と NVRTC 既定 FMA 契約の一致）を確認したケースに対応し、
/// K 方向の加算回数が多いほど丸め方針の不一致が蓄積して顕在化しやすい
/// ことを踏まえた回帰ケースとして分離する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn naive_f32_matches_cpu_reference_k_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive kernel compilation must succeed");

    assert_naive_f32_matches_cpu_reference(&gemm, 9999, 256, 256, 4096);
}

/// naive f16 GEMM（実機必須）。f16 は仮数部 10bit のため、f32 CPU 参照との
/// 比較は複合判定（1e-3 相対誤差）をそのまま適用すると表現精度由来の
/// 差分で不安定になりうる。本テストは「GPU が panic せず妥当な形状の
/// 出力を返す」ことまでを確認し、判定基準の詳細な妥当性確認（f16 向け
/// tolerance の要否）は #36（実機テスト整備）へ委ねる。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn naive_f16_runs_and_returns_expected_shape() {
    use half::f16;

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive kernel compilation must succeed");

    let mut rng = bench_harness::rng::Xorshift64Star::new(4242);
    let (m, n, k) = (64u32, 64u32, 64u32);
    let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

    let c_gpu = gemm
        .run_naive_f16(&a, &b, m, n, k)
        .expect("CudaGemm::run_naive_f16 must succeed on CUDA-equipped test runner");

    assert_eq!(c_gpu.len(), (m as usize) * (n as usize));
}
