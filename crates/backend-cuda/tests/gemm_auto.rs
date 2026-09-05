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

// イシュー #1158（f16 経路切替後の数値一致・非後退確認）の 2 テストが
// `common::parity_baseline`（ベースライン fixture・非後退判定ユーティリティ）
// を使うため読み込む（`cpu_cuda_mma_parity.rs` 等と同じ `mod common;` パターン）。
mod common;

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
/// （f16→f32→参照matmul→f16丸め→f32）と同一。本番既定
/// （`gemm_auto::MMA_PRIORITY_PRODUCTION_ENABLED = true`。#1191 で
/// 有効化・GB10 green 確認済み。イシュー #1160 の性能 A/B は GB10 実機
/// 実測で非後退を確認済みで、K=4096 非後退ゲートの `MmaF16` baseline
/// ceiling も #1190〈PR #1207〉でユーザー承認・反映済み〈PR #1179
/// codex-review 指摘の解消〉）は `mma → wmma → tiled` の優先順位
/// （#1156 設計、`docs/dispatch-rules-design.md` §5.6）を有効化して
/// いるため、整列形状（16,16,16）は cc >= 8.0 では mma、cc 7.x では
/// WMMA、cc < 7.0 では tiled を通る。本テストはパスの選択そのもので
/// はなく最終出力の parity のみを検証するため、いずれの経路が選ばれ
/// ても意味は変わらない。
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
///
/// `F16MatrixUnitImpl`／`CudaGemmAuto::f16_matrix_unit_impl`（診断アクセサ）
/// を直接使うため、`internal-diagnostics` feature（既定 off）限定
/// （codex-review PR #1177 指摘の是正。`src/lib.rs`・`src/gemm_auto.rs` の
/// 同 feature ゲート済み re-export／可視性を参照。`cargo test --all-
/// features` でのみビルド・実行される。他のテスト関数はこの feature に
/// 依存しないため無指定でも引き続き実行される）。
#[cfg(feature = "internal-diagnostics")]
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

/// 実機依存: `f16_matrix_unit_impl`（診断アクセサ）が本番既定
/// （`gemm_auto::MMA_PRIORITY_PRODUCTION_ENABLED = true`。#1191 で
/// ceiling 反映後に有効化済み。イシュー #1160 の本番有効化を保留して
/// いた K=4096 非後退ゲートの `MmaF16` baseline ceiling は #1190 で
/// ユーザー承認・反映済み〈PR #1179 codex-review 指摘の解消〉）どおり、
/// `auto.mma_available()`（`mma` が cc ゲートを満たし構築済みか）に
/// 応じて整列形状の選択結果が変わることを検証する（純関数
/// `select_f16_matrix_unit_impl` 自体の `prefer_mma` 網羅テストは
/// `gemm_auto.rs::f16_matrix_unit_impl_tests`〈GPU 非依存〉が担当。本
/// テストは実機上で本番既定の統合的な選択結果を検証する）。
///
/// `F16MatrixUnitImpl`／`CudaGemmAuto::f16_matrix_unit_impl`（診断アクセサ）
/// を直接使うため、`internal-diagnostics` feature（既定 off）限定
/// （codex-review PR #1177 指摘の是正。上記テストと同じ理由）。
#[cfg(feature = "internal-diagnostics")]
#[test]
#[ignore = "実機（CUDA ドライバ搭載環境）依存。README/Makefile の \
            test-ignored-cuda 経由で実行する"]
fn f16_matrix_unit_impl_reports_selected_implementation() {
    let device = CudaDevice::new(0).expect("CUDA device available in ignored test environment");
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new succeeds on real hardware");

    let mma_available = auto.mma_available();

    // 本番既定（MMA_PRIORITY_PRODUCTION_ENABLED = true。#1191）では、
    // mma が cc ゲート（cc>=8.0）を満たし構築済みなら整列形状
    // （n%8==0 && k%8==0・m の grid 上限内）で Mma を選ぶ。cc<8.0・
    // NVRTC 失敗環境等で mma_available() が false なら Wmma／Tiled へ
    // 倒れる（`Mma` を無条件にハードコードしない）。
    let aligned = auto.f16_matrix_unit_impl(16, 16, 16);
    if mma_available {
        assert_eq!(
            aligned,
            fandhe_ai_backend_cuda::F16MatrixUnitImpl::Mma,
            "MMA_PRIORITY_PRODUCTION_ENABLED = true かつ \
             mma_available() = true のため、整列形状（16,16,16）では \
             Mma が選ばれるはず"
        );
    } else {
        assert_ne!(
            aligned,
            fandhe_ai_backend_cuda::F16MatrixUnitImpl::Mma,
            "mma_available() = false のため、整列形状（16,16,16）でも \
             Mma は選ばれないはず"
        );
    }

    // 非整列形状（n=12 は n%8!=0）では mma_available() の値に関わらず
    // 事前形状ゲート非充足のため Mma を選ばない（`validate_mma_alignment`
    // 契約）。
    let misaligned = auto.f16_matrix_unit_impl(16, 12, 16);
    assert_ne!(
        misaligned,
        fandhe_ai_backend_cuda::F16MatrixUnitImpl::Mma,
        "非整列形状（n=12）では mma_available()（{mma_available}）に \
         関わらず事前形状ゲート非充足のため Mma は選ばれないはず"
    );
}

/// イシュー #1158: `CudaGemmAuto::run_f16` 経由（auto 経路）で
/// `cpu_cuda_mma_parity.rs::mma_f16_matches_reference_across_shapes` と
/// **同一の形状・シード・入力生成手順**を使い、CPU 参照実装との
/// 厳密ゼロ fail（`assert_parity`。spec REQ-2「2026-09-02 追記」項目 1）
/// が auto 経路でも成立することを確認する。
///
/// mma 優先が有効な本番既定（`MMA_PRIORITY_PRODUCTION_ENABLED = true`。
/// #1191 で有効化・GB10 green 確認済み。PR #1179 codex-review 指摘は
/// #1190 の ceiling 反映で解消済み）では `run_f16` はこれらの整列形状
/// で `CudaMmaGemm::run_f16` を呼ぶ。直接経路（`cpu_cuda_mma_parity.rs`）・
/// WMMA 直接経路（`cpu_cuda_wmma_parity.rs`）双方がこの 12 形状で GB10
/// 実機ゼロ fail 済みのため、本テストはフリップ前後どちらの経路でも
/// green のまま維持される設計だった（`docs/perf/cuda-parity-baseline.md`
/// §12 実測記録）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn run_f16_matches_cpu_reference_across_aligned_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new succeeds on real hardware");

    // `cpu_cuda_mma_parity.rs::mma_f16_matches_reference_across_shapes` と
    // 完全に同じ形状・idx 順（シードは `3000 + idx`）。
    let cases: &[(u32, u32, u32)] = &[
        (64, 128, 32),
        (64, 128, 64),
        (128, 256, 128),
        (32, 64, 32),
        (40, 24, 72),
        (100, 40, 88),
        (130, 72, 96),
        (65, 136, 40),
        (63, 120, 24),
        (200, 264, 104),
        (8, 8, 8),
        (1, 136, 40),
    ];

    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        let seed = 3000 + idx as u64;
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

        let c_gpu_f16 = auto
            .run_f16(&a_f16, &b_f16, m, n, k)
            .unwrap_or_else(|err| panic!("shape (m={m}, n={n}, k={k}): run_f16 failed: {err}"));
        let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

        fandhe_ai_backend_cpu::assert_parity(
            &format!("CudaGemmAuto::run_f16 (aligned shape m={m} n={n} k={k}) vs CPU reference"),
            &c_gpu_f32,
            &c_ref_rounded,
        );
    }
}

/// イシュー #1158: `CudaGemmAuto::run_f16` の K=4096 ストレス形状
/// （256×256×4096）が、**その時点で実際に選ばれている実装
/// （[`fandhe_ai_backend_cuda::F16MatrixUnitImpl`]）に対応する既存
/// ベースライン行**（`common::parity_baseline::ParityPath::MmaF16`／
/// `WmmaF16`）から後退していないことを検査する「経路自覚型」の非後退
/// テスト。
///
/// f16 K=4096 ストレスは K 支配的な積和で REQ-2 統一複合判定をわずかに
/// 外れる既知の tail 超過を持つ（`docs/backend-cuda-real-device-testing.md`
/// §5.3）ため、`assert_parity`（厳密ゼロ fail）ではなく非後退方式
/// （spec REQ-2「2026-09-02 追記」項目 2）で判定する。
///
/// **選ばれた経路によって参照する baseline 行・入力生成シードを切り替える**
/// （`Mma` → `ParityPath::MmaF16` 行・seed=9999〈`cpu_cuda_mma_parity.rs::
/// mma_f16_k4096_stress_non_regression` と同一入力〉、`Wmma` →
/// `ParityPath::WmmaF16` 行・seed=8888〈`cpu_cuda_wmma_parity.rs::
/// wmma_f16_k4096_stress_non_regression` と同一入力〉）ことで、#1160 が
/// `MMA_PRIORITY_PRODUCTION_ENABLED` を `true` へ切り替えても本テストの
/// 期待値変更は不要になる設計とする。`Tiled` が選ばれるケースは GB10
/// 実機では到達しない契約（対応する baseline 行を持たないため
/// fail-closed に panic する）。
///
/// `F16MatrixUnitImpl`／`CudaGemmAuto::f16_matrix_unit_impl` を直接使う
/// ため、`internal-diagnostics` feature（既定 off）限定
/// （`f16_matrix_unit_impl_reports_selected_implementation` と同じ理由）。
#[cfg(feature = "internal-diagnostics")]
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn run_f16_k4096_stress_non_regression_route_aware() {
    use fandhe_ai_backend_cuda::{CudaWmmaGemm, F16MatrixUnitImpl};

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new succeeds on real hardware");

    let (m, n, k) = (256u32, 256u32, 4096u32);
    let selected = auto.f16_matrix_unit_impl(m, n, k);
    let (path, seed) = match selected {
        F16MatrixUnitImpl::Mma => (common::parity_baseline::ParityPath::MmaF16, 9999u64),
        F16MatrixUnitImpl::Wmma => {
            // codex-review P1 / Cursor Bugbot 指摘対応（PR #1178 review）:
            // `F16MatrixUnitImpl::Wmma` は WMMA 実装が選ばれたことのみを
            // 表し、`CudaWmmaGemm::run_f16` が opt カーネルを実際にロード
            // できたことは保証しない（opt 利用不能時は basic へ黙って
            // フォールバックする。`gemm_wmma.rs::CudaWmmaGemm::run_f16`）。
            // `WmmaF16` baseline は opt 経路の実効値（`ParityPath::WmmaF16`
            // ドキュメンテーションコメント参照）のため、フォールバック状態
            // のまま比較すると opt 経路の消失を fail-open に見逃す。
            // 既存 `cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress_non_regression`
            // と同じ契約で、opt が実際に選択可能であることを fail-closed に
            // 確認する（理由付きで検査不能を明示する）。
            let wmma_gemm =
                CudaWmmaGemm::new(&device).expect("WMMA kernel compilation must succeed");
            assert!(
                wmma_gemm.wmma_f16_opt_available(),
                "opt カーネルが利用不能なため CudaGemmAuto::run_f16 は \
                 CudaWmmaGemm::run_f16 経由で基本版へフォールバックします \
                 （baseline は opt 経路の記録値のため検査不能。理由: {:?}）",
                wmma_gemm.wmma_f16_opt_unavailable_reason()
            );
            (common::parity_baseline::ParityPath::WmmaF16, 8888u64)
        }
        F16MatrixUnitImpl::Tiled => panic!(
            "CudaGemmAuto::run_f16 の K=4096 ストレス形状（m={m}, n={n}, k={k}）で \
             F16MatrixUnitImpl::Tiled が選ばれました。この経路には対応する \
             parity baseline 行が存在しないため非後退判定できません（GB10 \
             実機では到達しない契約。mma_available()={}・wmma 構築有無を \
             確認してください）",
            auto.mma_available(),
        ),
    };

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

    let c_gpu_f16 = auto
        .run_f16(&a_f16, &b_f16, m, n, k)
        .expect("CudaGemmAuto::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    let baseline = common::parity_baseline::BASELINES
        .iter()
        .find(|b| b.path == path && b.m == m && b.n == n && b.k == k && b.seed == seed)
        .unwrap_or_else(|| {
            panic!(
                "{path} 256x256x4096 seed={seed} の baseline 行が \
                 fixture に存在しません（selected={selected:?}）"
            )
        });
    let report = fandhe_ai_backend_cpu::compare(&c_gpu_f32, &c_ref_rounded)
        .expect("shape must match baseline fixture");
    // codex-review 指摘への対応（PR #1178 review）: `assert_no_parity_regression`
    // は `baseline_max_abs_diff_ceiling`/`baseline_max_rel_err_ceiling` が
    // `None` の行を「未実測」として黙ってスキップする設計
    // （`assert_no_parity_regression` 本体のコメント参照）。この
    // route-aware ゲートは fail_count・mean_abs_diff だけでなく外れ値
    // （1 要素だけ誤差が急増する回帰）も見逃さないことを本番経路の受け入れ
    // 判定として要求するため、選ばれた baseline 行の両 ceiling が `Some`
    // であることを事前に fail-closed 検査する（`None` を黙って通さない）。
    // `MmaF16` 行の両 ceiling はイシュー #1190 でユーザー承認値
    // （#1131 コメント 2026-09-04・実測は
    // `docs/perf/cuda-parity-baseline.md` §12.3〜§12.4）を
    // `common::parity_baseline::BASELINES` へ反映済み。mma 優先時に
    // 本ゲートが GB10 実機で green になることは、
    // `MMA_PRIORITY_PRODUCTION_ENABLED` を `true` へ復帰させたイシュー
    // #1191（本番有効化）で実機実測・確認済み。
    assert!(
        baseline.baseline_max_abs_diff_ceiling.is_some()
            && baseline.baseline_max_rel_err_ceiling.is_some(),
        "run_f16_k4096_stress_non_regression_route_aware 256x256x4096 \
         (selected={selected:?}, path={path}): baseline_max_abs_diff_ceiling/\
         baseline_max_rel_err_ceiling が None です。この route-aware ゲートは \
         外れ値回帰の見逃しを避けるため両 ceiling の設定を必須とします。\
         docs/perf/cuda-parity-baseline.md §12.4/§12.5 に記録済みの実測値・\
         提案 ceiling をユーザー承認のうえ \
         common::parity_baseline::BASELINES の該当行へ反映してください \
         （baseline 値の変更は人間承認必須。\
         .claude/rules/coding-rust.md「テスト・ベンチ」節）。"
    );
    common::parity_baseline::assert_no_parity_regression(
        &format!(
            "run_f16_k4096_stress_non_regression_route_aware 256x256x4096 (selected={selected:?})"
        ),
        &report,
        baseline,
    );
}
