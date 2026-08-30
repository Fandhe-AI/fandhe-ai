//! `gemm_variant_selection::CudaGemmF32VariantSelection`（イシュー #1035。
//! f32 GEMM の simple / double-buffer / split-K ヒューリスティック選択）の
//! 環境適応型テスト＋実機必須テスト。
//!
//! `tests/gemm_tiled.rs` と同じ構成（CUDA 搭載・非搭載どちらの環境でも
//! green になる環境適応型スモークテスト＋受け入れ条件そのものである
//! 数値一致・決定性テストを `#[ignore]` で分離）を踏襲する
//! （`.claude/rules/coding-rust.md` の実機依存テスト分離規約）。
//!
//! `CudaGemmF32VariantSelection`（`internal-diagnostics` feature 限定）を
//! 使うため、本ファイル自体も `Cargo.toml` の `[[test]]` セクションで
//! `required-features = ["internal-diagnostics"]` を指定している
//! （`specialized_mma_parity.rs` と同じ理由）。
//!
//! **本ランでの実施範囲**: `#[ignore]` テストは NVRTC 実行不能環境（実機
//! CUDA toolkit 非搭載）のため未実行のまま残す（実装計画 §8）。実機実測
//! （複合判定・決定性・境界形状網羅）は Mac / DGX Spark セッションで
//! `cargo test -p fandhe-ai-backend-cuda --features internal-diagnostics \
//! -- --ignored` により実施する。

use fandhe_ai_backend_cuda::gemm_variant_selection::CudaGemmF32VariantSelection;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError};

/// 決定的シードで A・B（f32）を生成する（`tests/gemm_tiled.rs` と同じ
/// 生成方法）。
fn gen_ab(seed: u64, m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    (rng.fill_vec(m * k), rng.fill_vec(k * n))
}

/// `CudaGemmF32VariantSelection::new` は CUDA 非搭載環境で panic せず
/// 型付きエラーを返す（`tests/gemm_tiled.rs::
/// new_compiles_tiled_kernels_or_returns_typed_error_without_panicking`
/// と同じ環境適応スモーク）。CUDA 搭載環境では base（Simple 経路。
/// naive/tiled 必須 4 カーネル）は必ずコンパイルに成功する契約
/// （`CudaGemm::new` に委譲。DoubleBuffer／SplitK はコンパイル失敗時
/// fail-soft のため `new` 自体は成功しうる）。
#[test]
fn new_builds_or_returns_typed_error_without_panicking() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => {
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    match CudaGemmF32VariantSelection::new(&device) {
        Ok(selection) => {
            // CUDA 搭載環境: base（Simple 経路）は必ず利用可能。DoubleBuffer／
            // SplitK は fail-soft のため可用性は問わない（`double_buffer_
            // available`/`split_k_available` の呼び出しが panic しないことのみ
            // 確認する）。
            let _ = selection.double_buffer_available();
            let _ = selection.split_k_available();
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(other) => {
            panic!("unexpected CudaError variant from CudaGemmF32VariantSelection::new: {other}")
        }
    }
}

/// `selected_variant` が実際の起動を伴わず呼べること（可観測性用 API の
/// スモーク。GPU 資源が無い場合でも `CudaGemmF32VariantSelection::new` が
/// 失敗するため到達しないが、到達した場合に panic しないことを確認する）。
#[test]
fn selected_variant_does_not_panic_when_device_available() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(_) => return,
    };
    let Ok(selection) = CudaGemmF32VariantSelection::new(&device) else {
        return;
    };
    let _ = selection.selected_variant(4096, 4096, 4096);
    let _ = selection.selected_variant(128, 128, 8192);
    let _ = selection.selected_variant(0, 0, 0);
}

/// 形状ごとに `run_f32` の結果を CPU 参照実装
/// （[`fandhe_ai_backend_cpu::matmul_reference_fma`]）と
/// [`fandhe_ai_backend_cpu::assert_parity`]（相対誤差 1e-3 未満 または
/// 絶対誤差 1e-5 未満の複合判定。`.claude/rules/coding-rust.md` の許容誤差
/// を変更しない）で照合する。DoubleBuffer／SplitK が実際に選ばれる形状
/// （アラインメント済み大形状・K 支配的非正方形状）と Simple に留まる
/// 形状（非整列・小 K）の両方を網羅する。
#[test]
#[ignore = "実機（CUDA/NVRTC 搭載）環境が必要。#1035 実装計画: 本ランは \
            NVRTC 実行不能環境のため未実行。Mac / DGX Spark セッションで \
            `cargo test -p fandhe-ai-backend-cuda --features \
            internal-diagnostics -- --ignored` として実施する"]
fn run_f32_matches_cpu_reference_across_variant_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available for ignored test");
    let selection =
        CudaGemmF32VariantSelection::new(&device).expect("variant selection handle must build");

    // (m, n, k, seed): DoubleBuffer 想定（大アラインメント形状）・
    // SplitK 想定（K 支配的非正方）・Simple 想定（非整列・小 K）・境界
    // （33/65/1000 等の非整列サイズ・m=n=128,k=8192 の K 支配形状）。
    let cases: &[(u32, u32, u32, u64)] = &[
        (4096, 4096, 4096, 1),
        (2048, 2048, 2048, 2),
        (128, 128, 8192, 3),
        (256, 256, 16384, 4),
        (1000, 1000, 1000, 5),
        (33, 65, 97, 6),
        (1, 1, 1, 7),
    ];

    for &(m, n, k, seed) in cases {
        let (a, b) = gen_ab(seed, m as usize, n as usize, k as usize);

        let actual = selection
            .run_f32(&a, &b, m, n, k)
            .unwrap_or_else(|e| panic!("run_f32 failed for m={m} n={n} k={k}: {e}"));

        let mut expected = vec![0.0f32; (m as usize) * (n as usize)];
        fandhe_ai_backend_cpu::matmul_reference_fma(
            &a,
            &b,
            &mut expected,
            m as usize,
            n as usize,
            k as usize,
        )
        .unwrap_or_else(|e| panic!("CPU reference failed for m={m} n={n} k={k}: {e:?}"));

        fandhe_ai_backend_cpu::assert_parity(
            &format!(
                "gemm_f32_variants m={m} n={n} k={k} variant={:?}",
                selection.selected_variant(m, n, k)
            ),
            &actual,
            &expected,
        );
    }
}

/// SplitK 経路の決定性（同一入力の 2 回実行が bit 一致すること。
/// atomics 不使用の設計〈`kernels_gemm_variants::SPLITK_PARTIAL_F32`／
/// `SPLITK_REDUCE_F32`〉の実機裏付け）。
#[test]
#[ignore = "実機（CUDA/NVRTC 搭載）環境が必要。上記 run_f32_matches_cpu_reference_across_variant_shapes と同じ理由"]
fn split_k_execution_is_bit_deterministic_across_repeated_runs() {
    let device = CudaDevice::new(0).expect("CUDA device must be available for ignored test");
    let selection =
        CudaGemmF32VariantSelection::new(&device).expect("variant selection handle must build");

    // K 支配的非正方形状（SplitK 選択を狙う。`gemm_variant.rs` の
    // ヒューリスティックが実際に SplitK を選ぶかは実機の SM 数実測に
    // 依存するため、選ばれなかった場合（Simple/DoubleBuffer）でも決定性
    // 自体は成立するはずであり、テストの主張は「選択された変種が何であれ
    // 2 回の実行が bit 一致する」こととする。
    let (m, n, k) = (128u32, 128u32, 8192u32);
    let (a, b) = gen_ab(42, m as usize, n as usize, k as usize);

    let first = selection
        .run_f32(&a, &b, m, n, k)
        .expect("first run_f32 must succeed");
    let second = selection
        .run_f32(&a, &b, m, n, k)
        .expect("second run_f32 must succeed");

    assert_eq!(
        first,
        second,
        "selected variant={:?} must produce bit-identical output across repeated runs",
        selection.selected_variant(m, n, k)
    );
}
