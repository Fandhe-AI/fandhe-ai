//! CPU-CUDA ペアの数値一致回帰テスト: f16 `mma.sync`/`ldmatrix`/`cp.async`
//! GEMM（TASK-11.1h・#187）。
//!
//! 受け入れ条件（#187 本文）「数値一致複合判定の通過」の本体。
//! `cpu_cuda_wmma_parity.rs`（#61）と同じく、判定は
//! `fandhe_ai_backend_cpu::assert_parity`（REQ-2 統一複合判定「相対誤差 1e-3 未満
//! または 絶対誤差 1e-5 未満」の唯一の実体）に一本化し、閾値・判定式を
//! ローカル複製しない。`cpu_cuda_wmma_parity.rs` 冒頭コメントが記す
//! 「f16 は複合判定の適用が実質的な許容誤差変更にあたるため対象外」
//! 方針の例外扱いは WMMA 経路に限定されており、本ファイル（mma 経路）が
//! 適用する根拠は #187 受け入れ条件そのもの（同ファイルの整理と同型）。
//!
//! # 参照実装との比較方法
//!
//! `cpu_cuda_wmma_parity.rs::assert_wmma_f16_parity` と同一手順:
//! f16→f32→`fandhe_ai_backend_cpu::matmul_reference_fma`→f16 丸め→f32 の経路で
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
//! エッジ形状（40×24×72 等。ブロックタイル `MMA_BM=64`/`MMA_BN=128`
//! 〈#494 時点〉の非倍数）で境界チェックの回帰対象とする。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaMmaGemm};
use half::f16;

mod common;

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
        .expect("CudaMmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    fandhe_ai_backend_cpu::assert_parity(context, &c_gpu_f32, &c_ref_rounded);
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
/// 受け入れ条件の本体（#501）。
///
/// タイル倍数形状（32/64/128）・8 の倍数の非タイル倍数エッジ形状
/// （REQ-8 手動境界検査の回帰対象。`MMA_BM=64`/`MMA_BN=128`/`MMA_BK=32`
/// 〈#494 時点。`kernels_mma.rs::mma_tile_constants_pinned_for_shape_table_cross_reference`
/// が前提値をロックしている〉の非倍数）を含む。すべて `n`/`k` が 8 の
/// 倍数（本経路の整列制約。`gemm_mma.rs::validate_mma_alignment`）。
/// #492〜#494（Phase B: wait_group 段数可変化・2x2 レジスタブロッキング・
/// ブロックタイル拡大）変更後の境界処理回帰を検出する目的で 12 形状へ
/// 拡張した（#501）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let cases: &[(u32, u32, u32)] = &[
        // --- タイル倍数コントロール（BM=64/BN=128/BK=32 の倍数） ---
        (64, 128, 32),   // 1 ブロックタイルちょうど（M/N/K いずれもタイル境界に一致）
        (64, 128, 64),   // K が 2 ブロックタイル分（K タイル境界を跨ぐが端数なし）
        (128, 256, 128), // M/N が 2 ブロックタイル分（複数ブロック分割の基準形状）
        // --- 非倍数・端タイル（REQ-8 手動境界検査の回帰対象） ---
        (32, 64, 32),    // M/N がブロックタイル未満のサブタイル形状
        (40, 24, 72),    // M がタイル未満・N/K は非倍数（既存: #187 由来）
        (100, 40, 88),   // M がタイル+端数・N/K 非倍数（既存: #187 由来）
        (130, 72, 96),   // M が 2 タイル+端数・N/K 非倍数（既存: #187 由来）
        (65, 136, 40),   // 全次元がタイル+端数（M=BM+1・N=BN+8・K=BK+8）
        (63, 120, 24),   // 全次元がタイル未満の非倍数（M=BM-1・N=BN-8・K=BK-8）
        (200, 264, 104), // 複数ブロックタイル+全次元端数（M=3*BM+8・N=2*BN+8・K=3*BK+8）
        (8, 8, 8),       // 最小整列形状（1 warp のさらに一部のみ有効な極小ケース）
        (1, 136, 40),    // m=1 の極小行（guarded store の r1 側が全域無効になる境界）
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let context = format!("shape m={m} n={n} k={k}");
        assert_mma_f16_parity(&gemm, &context, 3000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（PoC-v2-5 準拠の積和蓄積検証。
/// `cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress` と同じ形状で mma
/// 経路の桁落ち耐性・3 ステージパイプラインの周回耐性を確認する）。
///
/// **PR #1115（イシュー #1106）codex-review P1 指摘対応**: 本テストを
/// `assert_mma_f16_parity`（REQ-2 統一複合判定）から
/// `common::parity_baseline::assert_no_parity_regression`（既知不合格
/// ベースライン許容の非後退判定）へ置換していた変更を revert した
/// （AGENTS.md「数値契約の片側変更」・`.claude/rules/coding-rust.md`
/// 「バックエンド間数値一致テストの許容誤差を単独で緩和しない」に
/// 抵触するとの指摘）。f16 K=4096 ストレスに既知の tail 超過
/// （`docs/backend-cuda-real-device-testing.md` §5.3。K 支配的な積和で
/// REQ-2 統一複合判定をわずかに外れる要素が生じる）があること自体は
/// 事実であり、その非後退監視は `mma_f16_k4096_stress_non_regression`
/// （本ファイル下部）へ**別テストとして併設**する（codex-review の
/// 提案どおり、元の受け入れ条件はここで維持したまま置き換えない）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    assert_mma_f16_parity(&gemm, "K4096 stress 256x256x4096", 9999, 256, 256, 4096);
}

/// `mma_f16_k4096_stress`（256×256×4096・seed=9999）に対する**非後退
/// 監視の併設テスト**（イシュー #1106・PR #1115 codex-review P1 指摘
/// 対応）。
///
/// f16 K=4096 ストレスは既知の tail 超過（`docs/backend-cuda-real-device-testing.md`
/// §5.3。K 支配的な積和で REQ-2 統一複合判定〈相対誤差 1e-3 未満または
/// 絶対誤差 1e-5 未満〉をわずかに外れる要素が生じる）を持つため、
/// `mma_f16_k4096_stress` 本体は `assert_parity`（green 必須。REQ-2
/// 受け入れ条件そのもの）を維持したまま、本テストは #491 で確立した
/// parity 非後退契約（`common::parity_baseline::assert_no_parity_regression`）
/// で「既知の不合格分布から悪化していないか」を別観点として監視する。
/// **`assert_parity` を置き換えるものではなく追加のゲートである**——
/// `mma_f16_k4096_stress` が green になるまでは REQ-2 違反として扱う
/// （本体テストの failing が本来の状態を正しく表す）。
///
/// 形状・シード（256×256×4096・seed=9999）は `common/parity_baseline.rs`
/// の `ParityPath::MmaF16` 行（GB10 実機実測で確定済み・
/// `baseline_provenance_unconfirmed: false`）と完全一致するため、新規
/// 実機測定は不要（判定式・tolerance 定数は変更しない）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_k4096_stress_non_regression() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let (m, n, k, seed) = (256u32, 256u32, 4096u32, 9999u64);
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
        .expect("CudaMmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    let baseline = common::parity_baseline::BASELINES
        .iter()
        .find(|b| {
            b.path == common::parity_baseline::ParityPath::MmaF16
                && b.m == m
                && b.n == n
                && b.k == k
                && b.seed == seed
        })
        .expect("mma_f16 256x256x4096 seed=9999 baseline row must exist in fixture");
    let report = fandhe_ai_backend_cpu::compare(&c_gpu_f32, &c_ref_rounded)
        .expect("shape must match baseline fixture");
    common::parity_baseline::assert_no_parity_regression(
        "mma_f16_k4096_stress_non_regression 256x256x4096",
        &report,
        baseline,
    );
}

/// WMMA 経路（`CudaWmmaGemm::run_f16`）との相互比較。同一入力に対し
/// mma／WMMA 双方が同じ複合判定基準で参照実装と一致することを確認し、
/// mma 経路固有の回帰（フラグメントレーンマッピング・累算順序の誤り等）
/// を検出しやすくする（`cpu_cuda_wmma_parity.rs::wmma_f16_cross_check_against_naive_f16`
/// と同種の相互比較テスト）。
#[test]
#[ignore = "CUDA 実機（compute capability 8.0 以上・NVRTC 搭載）必須"]
fn mma_f16_cross_check_against_wmma_f16() {
    use fandhe_ai_backend_cuda::CudaWmmaGemm;

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
    fandhe_ai_backend_cpu::assert_parity("mma vs wmma f16 cross-check", &c_mma_f32, &c_wmma_f32);
}
