//! CPU-CUDA ペアの数値一致回帰テスト: f16 WMMA GEMM（TASK-11.1b・#61）。
//!
//! 受け入れ条件（#61 本文）「f16 GEMM が複合判定で参照実装と一致する」の
//! 本体。`cpu_cuda_parity.rs`（naive f32。#54）と同じく、判定は
//! `fandhe_ai_backend_cpu::assert_parity`（REQ-2 統一複合判定「相対誤差 1e-3 未満
//! または 絶対誤差 1e-5 未満」の唯一の実体）に一本化し、閾値・判定式を
//! ローカル複製しない。
//!
//! # f16 適用の位置づけ（`cpu_cuda_parity.rs` の f16 対象外方針との関係）
//!
//! `cpu_cuda_parity.rs` 冒頭コメントは「f16 は複合判定の適用が実質的な
//! 許容誤差変更にあたるため対象外」としているが、これは naive f16 等
//! 一般の f16 経路に対する既定方針である。本ファイル（WMMA f16）は
//! その方針の下でユーザー承認済みとして扱われる明示的な例外であり、
//! 根拠は #61 の受け入れ条件そのもの:
//!
//! > f16 GEMM が複合判定で参照実装と一致する
//!
//! この一文は「WMMA f16 カーネルの出力が REQ-2 統一複合判定
//! （1e-3/1e-5）で参照実装と一致すること」をイシューの受け入れ条件として
//! 明示的に要求しており、実装 Agent による閾値適用範囲の独自拡大では
//! ない（`.claude/rules/delegation-impl.md`
//! 「実装 Agent にガードレール閾値・テスト許容誤差を緩和させない」は
//! 閾値定数そのものの変更を指し、本ファイルは閾値定数を一切変更せず
//! `assert_parity` の判定式をそのまま利用する）。
//!
//! **範囲の限定**: 本例外は WMMA f16 経路（`CudaWmmaGemm::run_f16`）
//! にのみ適用され、naive f16（`CudaGemm::run_naive_f16`）や他の f16
//! 経路を複合判定の対象に含めるものではない。それらは引き続き
//! `cpu_cuda_parity.rs` の対象外方針に従う。複合判定を f16 全般へ
//! 一般化すべきかの検討は #186（Tensor Core 経路の数値一致閾値の
//! 実測再評価）に委ねる。
//!
//! # 参照実装との比較方法（実装計画 3.3 節）
//!
//! 1. f16 入力を f32 化し `fandhe_ai_backend_cpu::matmul_reference_fma`（FMA 契約の
//!    唯一の参照点）で参照値を計算する。
//! 2. 参照値を f16 経由で丸める（カーネルの `__float2half` によるエピ
//!    ローグ store と同じ量子化をホスト側でも再現し、丸め方式の差では
//!    なく計算経路の差のみを判定対象にする）。
//! 3. GPU 出力（f16）・丸め済み参照値（f16→f32）の双方を f32 化して
//!    `assert_parity` へ渡す。
//!
//! # 実機依存の分離
//!
//! `cpu_cuda_parity.rs` と同じ方針: 環境適応スモークのみ通常 CI で実行し
//! （CUDA 非搭載環境では早期 return で green）、形状網羅・cancellation
//! を誘発しやすい非倍数エッジ形状・K=4096 ストレスケースは `#[ignore]`
//! で分離する。スモークは cancellation の影響を受けにくい小さく偶数
//! 倍数の形状（16×16×16。WMMA_TILE 1 個ぴったり）に限定し、実機で
//! 複合判定を外れた場合も緩和せず #186（閾値再評価）へ引き渡す
//! （`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの
//! 許容誤差を単独で緩和しない」）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaWmmaGemm};
use half::f16;

mod common;

/// 決定的シードで A・B（f16）を生成し、f16→f32→参照 matmul→f16 丸め→f32 の
/// 経路で得た参照値と WMMA カーネル出力（f16→f32）を `assert_parity` で
/// 照合する（本ファイル冒頭コメント「参照実装との比較方法」参照）。
fn assert_wmma_f16_parity(gemm: &CudaWmmaGemm, context: &str, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b_f16: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut c_ref_f32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    // カーネルのエピローグ store（__float2half）と同じ量子化を参照側にも
    // 適用し、計算経路の差のみを判定対象にする（本ファイル冒頭コメント参照）。
    let c_ref_rounded: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let c_gpu_f16 = gemm
        .run_f16(&a_f16, &b_f16, m, n, k)
        .expect("CudaWmmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    fandhe_ai_backend_cpu::assert_parity(context, &c_gpu_f32, &c_ref_rounded);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
///
/// `tests/gemm_wmma.rs::new_does_not_panic_and_returns_typed_result` と
/// 同じ分岐パターンで、CUDA 非搭載・NVRTC 非搭載・cc<7.0 のいずれの
/// 環境でも早期 return し green とする。CUDA+toolkit+WMMA 対応環境
/// でのみ 16×16×16（WMMA_TILE 1 個ぴったり・K タイル境界を跨がない
/// 最小形状）で `assert_parity` による複合判定を実施する。
#[test]
fn wmma_f16_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let gemm = match CudaWmmaGemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            // cc < 7.0 の実機（WMMA 非対応）。ディスパッチ規則（#66）が
            // 未実装の現段階では tiled 経路へのフォールバックは呼び出し元
            // の責務であり、本テストは型付きエラーの契約のみ確認する。
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaWmmaGemm::new: {other}"),
    };

    assert_wmma_f16_parity(&gemm, "smoke 16x16x16", 1, 16, 16, 16);
}

/// 実機（DGX Spark GB10 等）必須の形状網羅テスト。受け入れ条件の本体。
///
/// fragment 倍数形状（64/128/256）・非倍数エッジ形状（REQ-8 手動境界検査
/// の回帰対象。17×19×23・100×100×100・130×70×90）を含む。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    let cases: &[(u32, u32, u32)] = &[
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (17, 19, 23),
        (100, 100, 100),
        (130, 70, 90),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_wmma_f16_parity(&gemm, &context, 2000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（PoC-v2-5 準拠の積和蓄積検証。`cpu_cuda_parity.rs`
/// の naive f32 K=4096 ケースと同じ形状で WMMA 経路の桁落ち耐性を確認する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    assert_wmma_f16_parity(&gemm, "K4096 stress 256x256x4096", 8888, 256, 256, 4096);
}

/// `wmma_f16_k4096_stress`（256×256×4096・seed=8888）に対する**非後退
/// 監視の併設テスト**（イシュー #1106・GB10 全件洗い出し。
/// `tests/cpu_cuda_mma_parity.rs::mma_f16_k4096_stress_non_regression`
/// と同型のパターン）。
///
/// f16 K=4096 ストレスは既知の tail 超過（`docs/backend-cuda-real-device-testing.md`
/// §5.3。K 支配的な積和で REQ-2 統一複合判定〈相対誤差 1e-3 未満または
/// 絶対誤差 1e-5 未満〉をわずかに外れる要素が生じる）を持つため、
/// `wmma_f16_k4096_stress` 本体は `assert_parity`（green 必須。REQ-2
/// 受け入れ条件そのもの）を維持したまま、本テストは #491 で確立した
/// parity 非後退契約（`common::parity_baseline::assert_no_parity_regression`）
/// で「既知の不合格分布から悪化していないか」を別観点として監視する。
/// **`assert_parity` を置き換えるものではなく追加のゲートである**——
/// `wmma_f16_k4096_stress` が green になるまでは REQ-2 違反として扱う
/// （本体テストの failing が本来の状態を正しく表す。
/// `docs/perf/cuda-parity-baseline.md` §10.4 参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_k4096_stress_non_regression() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");

    let (m, n, k, seed) = (256u32, 256u32, 4096u32, 8888u64);
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b_f16: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));
    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
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
        .expect("CudaWmmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    let baseline = common::parity_baseline::BASELINES
        .iter()
        .find(|b| {
            b.path == common::parity_baseline::ParityPath::WmmaF16
                && b.m == m
                && b.n == n
                && b.k == k
                && b.seed == seed
        })
        .expect("wmma_f16 256x256x4096 seed=8888 baseline row must exist in fixture");
    let report = fandhe_ai_backend_cpu::compare(&c_gpu_f32, &c_ref_rounded)
        .expect("shape must match baseline fixture");
    common::parity_baseline::assert_no_parity_regression(
        "wmma_f16_k4096_stress_non_regression 256x256x4096",
        &report,
        baseline,
    );
}

/// naive f16（`CudaGemm::run_naive_f16`）との相互比較（実装計画 4 節）。
/// 同一入力に対し naive／WMMA 双方が同じ複合判定基準で参照実装と一致する
/// ことを確認し、WMMA 経路固有の回帰（fragment ロード順・累算順序の
/// 誤り等）を検出しやすくする。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn wmma_f16_cross_check_against_naive_f16() {
    use fandhe_ai_backend_cuda::CudaGemm;

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let wmma_gemm = CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");
    let naive_gemm = CudaGemm::new(&device).expect("naive kernel compilation must succeed");

    let (m, n, k) = (48u32, 48u32, 48u32);
    let mut rng = bench_harness::rng::Xorshift64Star::new(4242);
    let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));

    let c_wmma_f16 = wmma_gemm
        .run_f16(&a, &b, m, n, k)
        .expect("CudaWmmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_naive_f16 = naive_gemm
        .run_naive_f16(&a, &b, m, n, k)
        .expect("CudaGemm::run_naive_f16 must succeed on CUDA-equipped test runner");

    let c_wmma_f32: Vec<f32> = c_wmma_f16.iter().map(|x| x.to_f32()).collect();
    let c_naive_f32: Vec<f32> = c_naive_f16.iter().map(|x| x.to_f32()).collect();
    fandhe_ai_backend_cpu::assert_parity(
        "wmma vs naive f16 cross-check",
        &c_wmma_f32,
        &c_naive_f32,
    );
}
