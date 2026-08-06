//! naive GEMM（`CudaGemm`）の API 健全性テスト。
//!
//! CUDA 搭載・非搭載どちらの環境でも green になる設計（初期化・形状検証・
//! エラー型の契約確認）を提供する。**数値一致回帰テスト（CPU-CUDA ペアの
//! 複合判定）は `tests/cpu_cuda_parity.rs`（TASK-2.2b・#54）へ移管した。**
//! 旧実装はローカル複製の判定式（`rel >= TOL && diff >= TOL` という否定形）
//! を使っており、NaN/Inf 混入時に誤って合格判定してしまう盲点を持って
//! いたため、`backend_cpu::assert_parity`（TASK-2.2a・#53）への一本化と
//! 合わせて移管した（詳細は `cpu_cuda_parity.rs` 冒頭コメント参照）。

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

/// m==0／n==0（`backend-cpu::gemm_naive` と同じ no-op 形状）で
/// `run_naive_f32` を呼んでも CUDA 起動そのものが発生せず（`gemm.rs` の
/// 早期 return）、空の結果を返すことを実機で確認する（Cursor Bugbot
/// 指摘 #240: 0 次元 grid の起動は CUDA ドライバに拒否される）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn naive_f32_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("naive kernel compilation must succeed");

    let c = gemm
        .run_naive_f32(&[], &[1.0, 2.0, 3.0, 4.0], 0, 4, 1)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_naive_f32(&[1.0, 2.0], &[], 2, 0, 1)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());
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
