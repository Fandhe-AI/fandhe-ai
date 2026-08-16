//! GEMM 自動経路選択（`CudaGemmAuto`。TASK-11.2b・#68）の統合テスト。
//!
//! (a) 実機非依存: `CudaGemmAuto::new` が CUDA 非搭載環境で panic せず
//!     型付きエラーを返すこと（`gemm_naive.rs::new_does_not_panic_and_returns_typed_result`
//!     と同じ健全性契約）を確認する。
//! (b) `#[ignore]` 実機依存: `run_f32`／`run_f16` の出力が
//!     `backend_cpu::assert_parity`（REQ-2 統一複合判定）で参照実装と
//!     一致することを検証する。既存 `cpu_cuda_parity.rs`／
//!     `cpu_cuda_wmma_parity.rs` と同じ判定式・許容誤差をそのまま使い、
//!     許容誤差は一切変更しない（`.claude/rules/coding-rust.md`）。
//!
//! 決定表そのもの（HW・形状・dtype から `KernelKind` を選ぶ純関数）の
//! 網羅テストは `tensor-core` 側（`crates/tensor-core/src/dispatch.rs`
//! の `#[cfg(test)]`）が担当する。本ファイルは「選ばれた経路が実際に
//! 実行され、既存カーネルと同じ数値契約を満たすか」の統合検証に限定する。

use backend_cuda::{
    CudaDevice, CudaError, CudaGemmAuto, TileSelectionBasis, select_tile_config_for_device,
};
use half::f16;
use tensor_core::dispatch::{DType, GemmShape};

// `backend_cpu::matmul_reference_fma`／`assert_parity` はクレートルート
// で再エクスポートされている（`crates/backend-cpu/src/lib.rs`）。
// `cpu_cuda_parity.rs`／`cpu_cuda_wmma_parity.rs` と同じ呼び出し規約。

/// `CudaGemmAuto::new` は CUDA 非搭載環境で panic せず型付きエラーを
/// 返す（`CudaDevice::new` が既に満たしている契約を auto 入口経路でも
/// 確認する）。CUDA 搭載環境では naive／tiled／（cc 対応時）WMMA の
/// コンパイルが成功することを検証する。
#[test]
fn new_does_not_panic_and_returns_typed_result() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => {
            // libcuda は存在するが cuInit 等が失敗したケース。
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    match CudaGemmAuto::new(&device) {
        Ok(_auto) => {
            // CUDA 搭載環境: naive/tiled の構築成功（WMMA は cc 非対応・
            // コンパイル失敗時 None のまま保持されるため new 自体は失敗しない）。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemmAuto::new: {other}"),
    }
}

/// 実機依存: f32 自動経路（現決定表では常に tiled）が
/// `backend_cpu::matmul_reference_fma` と複合判定で一致することを検証する。
/// `cpu_cuda_parity.rs` と同じ小型スモーク形状を使う。
#[test]
#[ignore = "実機（CUDA ドライバ搭載環境）依存。README/Makefile の \
            test-ignored-cuda 経由で実行する"]
fn run_f32_matches_cpu_reference() {
    let device = CudaDevice::new(0).expect("CUDA device available in ignored test environment");
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new succeeds on real hardware");

    let (m, n, k) = (16u32, 16u32, 16u32);
    let a: Vec<f32> = (0..(m * k)).map(|i| (i % 7) as f32 * 0.1).collect();
    let b: Vec<f32> = (0..(k * n)).map(|i| (i % 5) as f32 * 0.2).collect();

    let gpu = auto
        .run_f32(&a, &b, m, n, k)
        .expect("run_f32 succeeds on real hardware");

    let mut reference = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::matmul_reference_fma(&a, &b, &mut reference, m as usize, n as usize, k as usize)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");

    backend_cpu::assert_parity("CudaGemmAuto::run_f32 vs CPU reference", &gpu, &reference);
}

/// 実機依存: f16 自動経路が（cc >= 7.0 環境では WMMA、それ以外では
/// tiled が）参照実装と複合判定で一致することを検証する。判定方法は
/// `cpu_cuda_wmma_parity.rs`（f16→f32→参照matmul→f16丸め→f32）と同一。
#[test]
#[ignore = "実機（CUDA ドライバ搭載環境）依存。README/Makefile の \
            test-ignored-cuda 経由で実行する"]
fn run_f16_matches_cpu_reference() {
    let device = CudaDevice::new(0).expect("CUDA device available in ignored test environment");
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new succeeds on real hardware");

    let (m, n, k) = (16u32, 16u32, 16u32);
    let a_f32: Vec<f32> = (0..(m * k)).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_f32: Vec<f32> = (0..(k * n)).map(|i| (i % 5) as f32 * 0.2).collect();
    let a: Vec<f16> = a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let b: Vec<f16> = b_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let gpu = auto
        .run_f16(&a, &b, m, n, k)
        .expect("run_f16 succeeds on real hardware");

    let mut reference_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut reference_f32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    // カーネルのエピローグ store（__float2half）と同じ量子化をホスト側
    // でも再現してから比較する（cpu_cuda_wmma_parity.rs と同一手順）。
    let reference_rounded: Vec<f32> = reference_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();
    let gpu_f32: Vec<f32> = gpu.iter().map(|&x| x.to_f32()).collect();

    backend_cpu::assert_parity(
        "CudaGemmAuto::run_f16 vs CPU reference",
        &gpu_f32,
        &reference_rounded,
    );
}

/// 実機依存: `select_tile_config_for_device`（Phase C-9b・イシュー #527）
/// がドライバ属性照会（SMEM 予算・SM 数）を実際に成功させ、`Ok` の
/// タイル選定を返すことを検証する。
///
/// `SM121_MEASURED_BANDWIDTH` が未実測（`None`）である現時点では選定
/// 根拠は必ず [`TileSelectionBasis::FixedTable`] になり、選定構成は
/// 実測裏付けのある現行本番構成（64/128/32・stages 3）と一致する
/// （`docs/perf/cuda-gemm-cost-model-selection.md` の実機比較手順が
/// 完了し帯域が `Some` 化された後は、この事前条件〈`FixedTable` 固定〉
/// も合わせて更新する）。
#[test]
#[ignore = "実機（CUDA ドライバ搭載環境）依存。README/Makefile の \
            test-ignored-cuda 経由で実行する"]
fn select_tile_config_for_device_succeeds_on_real_hardware() {
    let device = CudaDevice::new(0).expect("CUDA device available in ignored test environment");
    let shape = GemmShape::new(4096, 4096, 4096);

    let selection = select_tile_config_for_device(&device, shape, DType::F16)
        .expect("select_tile_config_for_device succeeds on real hardware");

    // SM121_MEASURED_BANDWIDTH が None の間は常に固定選定テーブルへ
    // フォールバックする（本モジュール doc・gemm_auto.rs
    // `select_tile_config` 参照）。
    assert_eq!(selection.basis(), TileSelectionBasis::FixedTable);
    assert_eq!(selection.candidate().block_m().get(), 64);
    assert_eq!(selection.candidate().block_n().get(), 128);
    assert_eq!(selection.candidate().block_k().get(), 32);
    assert_eq!(selection.candidate().stages().get(), 3);
}
