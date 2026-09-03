//! GEMM 自動経路選択（`CudaGemmAuto`。TASK-11.2b・#68）の統合テスト。
//!
//! (a) 実機非依存: `CudaGemmAuto::new` が CUDA 非搭載環境で panic せず
//!     型付きエラーを返すこと（`gemm_naive.rs::new_does_not_panic_and_returns_typed_result`
//!     と同じ健全性契約）を確認する。
//! (b) `#[ignore]` 実機依存: `run_f32`／`run_f16` の出力が
//!     `fandhe_ai_backend_cpu::assert_parity`（REQ-2 統一複合判定）で参照実装と
//!     一致することを検証する。既存 `cpu_cuda_parity.rs`／
//!     `cpu_cuda_wmma_parity.rs` と同じ判定式・許容誤差をそのまま使い、
//!     許容誤差は一切変更しない（`.claude/rules/coding-rust.md`）。
//!
//! 決定表そのもの（HW・形状・dtype から `KernelKind` を選ぶ純関数）の
//! 網羅テストは `tensor-core` 側（`crates/tensor-core/src/dispatch.rs`
//! の `#[cfg(test)]`）が担当する。本ファイルは「選ばれた経路が実際に
//! 実行され、既存カーネルと同じ数値契約を満たすか」の統合検証に限定する。

use fandhe_ai_backend_cuda::{
    CudaDevice, CudaError, CudaGemmAuto, TileSelectionBasis, select_tile_config_for_device,
};
use fandhe_ai_tensor_core::dispatch::{DType, GemmShape};
use half::f16;

// `fandhe_ai_backend_cpu::matmul_reference_fma`／`assert_parity` はクレートルート
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
            // CUDA 搭載環境: naive/tiled の構築成功（WMMA・mma
            // 〈#1152〉はいずれも cc 非対応・コンパイル失敗時 None の
            // まま保持され、new 自体は失敗しない。fail-soft。
            // `docs/dispatch-rules-design.md` §5.6 判定規則 1）。
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemmAuto::new: {other}"),
    }
}

/// 実機依存: f32 自動経路（現決定表では常に tiled）が
/// `fandhe_ai_backend_cpu::matmul_reference_fma` と複合判定で一致することを検証する。
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
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a,
        &b,
        &mut reference,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");

    fandhe_ai_backend_cpu::assert_parity(
        "CudaGemmAuto::run_f32 vs CPU reference",
        &gpu,
        &reference,
    );
}

/// 実機依存: f16 自動経路（16x16x16・整列形状）が参照実装と複合判定で
/// 一致することを検証する。判定方法は `cpu_cuda_wmma_parity.rs`
/// （f16→f32→参照matmul→f16丸め→f32）と同一。cc >= 8.0 環境では
/// `select_f16_matrix_unit_impl` が `Mma` を返すため `CudaMmaGemm`
/// 経路（#1156）を通り、cc 7.x では WMMA、それ未満では tiled を通る
/// （`docs/dispatch-rules-design.md` §5.6）。
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
    fandhe_ai_backend_cpu::matmul_reference_fma(
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

    fandhe_ai_backend_cpu::assert_parity(
        "CudaGemmAuto::run_f16 vs CPU reference",
        &gpu_f32,
        &reference_rounded,
    );
}

/// 実機依存: `CudaGemmAuto::new` が構築する `mma` フィールド（#1152）が
/// cc>=8.0 ゲートに従って fail-soft に構築されることを検証する
/// （`docs/dispatch-rules-design.md` §5.6 判定規則 1）。本テストは経路
/// 実行ではなく `mma_available`／`mma_unavailable_reason` の診断
/// アクセサのみを検証する（`run_f16` 側が実際に `mma` を優先して呼ぶ
/// ことの担保は `run_f16_matches_cpu_reference`・下記
/// `run_f16_misaligned_shape_falls_back_to_wmma_or_tiled`・
/// `f16_matrix_unit_impl_reports_selected_implementation` が担う。
/// #1156）。
#[test]
#[ignore = "実機（CUDA ドライバ搭載環境）依存。README/Makefile の \
            test-ignored-cuda 経由で実行する"]
fn mma_field_is_constructed_by_compute_capability_gate() {
    let device = CudaDevice::new(0).expect("CUDA device available in ignored test environment");
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new succeeds on real hardware");

    let (major, _minor) = device.compute_capability();
    if major >= 8 {
        assert!(
            auto.mma_available(),
            "cc {major}.x >= 8.0 のはずが mma が None（理由: {:?}）。\
             CudaMmaGemm::new の NVRTC コンパイルが失敗している可能性がある",
            auto.mma_unavailable_reason(),
        );
        assert!(
            auto.mma_unavailable_reason().is_none(),
            "mma_available() が true の場合 mma_unavailable_reason() は None のはず"
        );
    } else {
        assert!(
            !auto.mma_available(),
            "cc {major}.x < 8.0 のはずが mma が Some（cc ゲートが機能していない）"
        );
        let reason = auto
            .mma_unavailable_reason()
            .expect("mma_available() が false の場合 mma_unavailable_reason() は Some のはず");
        assert!(
            reason.contains("compute capability"),
            "cc ゲート起因の失敗理由には 'compute capability' を含むはず: {reason}"
        );
    }
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

/// 実機依存: 事前形状ゲート非充足（`n` が 8 の倍数でない）形状では
/// `select_f16_matrix_unit_impl` が `Wmma`（cc ゲート対応環境）または
/// `Tiled`（非対応環境）を返し、`run_f16` の出力が引き続き参照実装と
/// 複合判定で一致することを検証する（mma → wmma/tiled フォールバック
/// 経路の維持を実機で実証する。§5.6 判定規則 2・3・#1156）。
#[test]
#[ignore = "実機（CUDA ドライバ搭載環境）依存。README/Makefile の \
            test-ignored-cuda 経由で実行する"]
fn run_f16_misaligned_shape_falls_back_to_wmma_or_tiled() {
    let device = CudaDevice::new(0).expect("CUDA device available in ignored test environment");
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new succeeds on real hardware");

    // n=12 は 8 の倍数でないため validate_mma_alignment が Err を返し、
    // mma が構築済みでも mma 経路には入らない（wmma または tiled）。
    let (m, n, k) = (16u32, 12u32, 16u32);
    let a_f32: Vec<f32> = (0..(m * k)).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_f32: Vec<f32> = (0..(k * n)).map(|i| (i % 5) as f32 * 0.2).collect();
    let a: Vec<f16> = a_f32.iter().map(|&x| f16::from_f32(x)).collect();
    let b: Vec<f16> = b_f32.iter().map(|&x| f16::from_f32(x)).collect();

    let selected = auto.f16_matrix_unit_impl(m, n, k);
    assert_ne!(
        selected,
        fandhe_ai_backend_cuda::F16MatrixUnitImpl::Mma,
        "n=12 は非整列のため mma 経路が選ばれてはならない"
    );

    let gpu = auto
        .run_f16(&a, &b, m, n, k)
        .expect("run_f16 succeeds on real hardware");

    let mut reference_f32 = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut reference_f32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let reference_rounded: Vec<f32> = reference_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();
    let gpu_f32: Vec<f32> = gpu.iter().map(|&x| x.to_f32()).collect();

    fandhe_ai_backend_cpu::assert_parity(
        "CudaGemmAuto::run_f16 (misaligned n, wmma/tiled fallback) vs CPU reference",
        &gpu_f32,
        &reference_rounded,
    );
}

/// 実機依存: `f16_matrix_unit_impl`（診断アクセサ）が cc・整列形状に
/// 応じて期待どおりの `F16MatrixUnitImpl` を返すことを検証する
/// （純関数 `select_f16_matrix_unit_impl` 自体の網羅テストは
/// `gemm_auto.rs::f16_matrix_unit_impl_tests`〈GPU 非依存〉が担当。
/// 本テストは実機の `mma_available()`／`wmma` 構築結果と整合すること
/// の統合検証に限定する）。
#[test]
#[ignore = "実機（CUDA ドライバ搭載環境）依存。README/Makefile の \
            test-ignored-cuda 経由で実行する"]
fn f16_matrix_unit_impl_reports_selected_implementation() {
    let device = CudaDevice::new(0).expect("CUDA device available in ignored test environment");
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new succeeds on real hardware");

    let (major, _minor) = device.compute_capability();
    let aligned = auto.f16_matrix_unit_impl(16, 16, 16);
    if major >= 8 && auto.mma_available() {
        assert_eq!(
            aligned,
            fandhe_ai_backend_cuda::F16MatrixUnitImpl::Mma,
            "cc {major}.x >= 8.0 かつ mma 構築済みなら整列形状は Mma のはず"
        );
    } else {
        assert_ne!(
            aligned,
            fandhe_ai_backend_cuda::F16MatrixUnitImpl::Mma,
            "mma 未構築（cc 非対応または NVRTC 失敗）なら Mma は選ばれないはず"
        );
    }
}
