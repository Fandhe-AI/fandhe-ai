//! CPU-CUDA ペアの数値一致回帰テスト: f16 `mma.sync`/`ldmatrix`/`cp.async`
//! GEMM（TASK-11.1h・#187）。
//!
//! 受け入れ条件（#187 本文）「数値一致複合判定の通過」の本体。
//! `cpu_cuda_wmma_parity.rs`（#61）と同じく、判定は
//! `backend_cpu::assert_parity`（REQ-2 統一複合判定「相対誤差 1e-3 未満
//! または 絶対誤差 1e-5 未満」の唯一の実体）に一本化し、閾値・判定式を
//! ローカル複製しない。`cpu_cuda_wmma_parity.rs` 冒頭コメントが記す
//! 「f16 は複合判定の適用が実質的な許容誤差変更にあたるため対象外」
//! 方針の例外扱いは WMMA 経路に限定されており、本ファイル（mma 経路）が
//! 適用する根拠は #187 受け入れ条件そのもの（同ファイルの整理と同型）。
//!
//! # 参照実装との比較方法
//!
//! `cpu_cuda_wmma_parity.rs::assert_wmma_f16_parity` と同一手順:
//! f16→f32→`backend_cpu::matmul_reference_fma`→f16 丸め→f32 の経路で
//! 得た参照値と、カーネル出力（f16→f32）を `assert_parity` で照合する。
//!
//! # 実機依存の分離
//!
//! `cpu_cuda_wmma_parity.rs` と同じ方針: 環境適応スモークのみ通常 CI で
//! 実行し（CUDA 非搭載・NVRTC 非搭載・cc<8.0 環境では早期 return で
//! green）、形状網羅・K=4096 ストレスケースは `#[ignore]` で分離する。
//! 本経路は `n`/`k` が 8 の倍数であることを要求する（`kernels_mma.rs`
//! 冒頭コメント「整列制約」）ため、`cpu_cuda_wmma_parity.rs` の非倍数
//! エッジ形状（17×19×23 等）はそのまま流用できない。8 の倍数の
//! エッジ形状（40×24×72 等。ブロックタイル `MMA_BM=32`/`MMA_BN=64` の
//! 非倍数）で境界チェックの回帰対象とする。

use backend_cuda::{CudaDevice, CudaError, CudaMmaGemm};
use half::f16;

/// 決定的シードで A・B（f16）を生成し、参照値とカーネル出力を
/// `assert_parity` で照合する（本ファイル冒頭コメント「参照実装との
/// 比較方法」参照）。
fn assert_mma_f16_parity(gemm: &CudaMmaGemm, context: &str, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b_f16: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut c_ref_f32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_ref_rounded: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let c_gpu_f16 = gemm
        .run_f16(&a_f16, &b_f16, m, n, k)
        .expect("CudaMmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    backend_cpu::assert_parity(context, &c_gpu_f32, &c_ref_rounded);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
///
/// `tests/gemm_mma.rs::new_does_not_panic_and_returns_typed_result` と
/// 同じ分岐パターンで、CUDA 非搭載・NVRTC 非搭載・cc<8.0 のいずれの
/// 環境でも早期 return し green とする（本実装セッションの実行環境は
/// NVRTC 非搭載分岐を通る。`kernels_mma.rs` 冒頭「検証状態」参照）。
/// CUDA+toolkit+cc>=8.0 環境でのみ 16×8×16（1 warp が担当する
/// `MMA_M x MMA_N` タイルちょうど・K タイル境界を跨がない最小形状）で
/// `assert_parity` による複合判定を実施する。
#[test]
fn mma_f16_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let gemm = match CudaMmaGemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            // cc < 8.0 の実機。ディスパッチ規則（#66）が未実装の現段階
            // では tiled/WMMA 経路へのフォールバックは呼び出し元の責務。
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaMmaGemm::new: {other}"),
    };

    assert_mma_f16_parity(&gemm, "smoke 16x8x16", 1, 16, 8, 16);
}

/// 実機（compute capability 8.0 以上・NVRTC 搭載）必須の形状網羅テスト。
/// 受け入れ条件の本体。
///
/// タイル倍数形状（32/64/128）・8 の倍数の非タイル倍数エッジ形状
/// （REQ-8 手動境界検査の回帰対象。`MMA_BM=32`/`MMA_BN=64` の非倍数）を
/// 含む。すべて `n`/`k` が 8 の倍数（本経路の整列制約）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let cases: &[(u32, u32, u32)] = &[
        (32, 64, 32),
        (64, 128, 64),
        (128, 256, 128),
        (40, 24, 72),
        (100, 40, 88),
        (130, 72, 96),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_mma_f16_parity(&gemm, &context, 3000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（PoC-v2-5 準拠の積和蓄積検証。
/// `cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress` と同じ形状で mma
/// 経路の桁落ち耐性・3 ステージパイプラインの周回耐性を確認する）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    assert_mma_f16_parity(&gemm, "K4096 stress 256x256x4096", 9999, 256, 256, 4096);
}

/// WMMA 経路（`CudaWmmaGemm::run_f16`）との相互比較。同一入力に対し
/// mma／WMMA 双方が同じ複合判定基準で参照実装と一致することを確認し、
/// mma 経路固有の回帰（フラグメントレーンマッピング・累算順序の誤り等）
/// を検出しやすくする（`cpu_cuda_wmma_parity.rs::wmma_f16_cross_check_against_naive_f16`
/// と同種の相互比較テスト）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_cross_check_against_wmma_f16() {
    use backend_cuda::CudaWmmaGemm;

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let mma_gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");
    let wmma_gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    let (m, n, k) = (48u32, 64u32, 64u32);
    let mut rng = bench_harness::rng::Xorshift64Star::new(5252);
    let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

    let c_mma_f16 = mma_gemm
        .run_f16(&a, &b, m, n, k)
        .expect("CudaMmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_wmma_f16 = wmma_gemm
        .run_f16(&a, &b, m, n, k)
        .expect("CudaWmmaGemm::run_f16 must succeed on CUDA-equipped test runner");

    let c_mma_f32: Vec<f32> = c_mma_f16.iter().map(|x| x.to_f32()).collect();
    let c_wmma_f32: Vec<f32> = c_wmma_f16.iter().map(|x| x.to_f32()).collect();
    backend_cpu::assert_parity("mma vs wmma f16 cross-check", &c_mma_f32, &c_wmma_f32);
}
