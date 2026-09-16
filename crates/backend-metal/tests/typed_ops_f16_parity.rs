//! `TypedOps<half::f16>` の Metal 統合テスト（イシュー #1705・親 #1651）。
//!
//! (a) 実機非依存（macOS 上で常時コンパイル・実行。GPU 不要）: accessor
//!     契約・shape 検証（デバイス到達前に拒否されること）を確認する。
//! (b) `#[ignore]` 実機依存（Apple Silicon）: `gemm` が
//!     `MetalGemm::dispatch_f16_auto_unverified` 直接呼び出しと bit
//!     完全一致すること（パススルー結線の直接検証）・6 演算
//!     （`add`／`mul`／`relu`／`exp`／`tanh`／`gemm` 参照比較）が CPU
//!     参照実装の f32 経路を f16 へ丸めた値と REQ-2 統一複合判定
//!     （`assert_parity`）で一致することを確認する。`sum`（イシュー
//!     #1896 で `reduce::MetalReduce` へ結線済み）は下記 `#[ignore]`
//!     `sum_matches_f32_backend_ops_rounded_bit_exact` が別途担う。
//!     `max` は Metal f32 reduction 未実装のため対象外（`crate::
//!     typed_f16` モジュール doc「`max` は `Unsupported` を継承する」
//!     参照）。
//!
//! `tests/gemm_f16_auto_parity.rs` と同じ判定基盤
//! （`fandhe_ai_backend_cpu::parity::assert_parity`。REQ-2 統一複合判定
//! 「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の唯一の実体。許容誤差は
//! 変更しない `.claude/rules/coding-rust.md`）を使う。

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};
use fandhe_ai_backend_metal::{MetalBackendOps, MetalContext, MetalGemm};
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor, TypedOps};
use half::f16;

fn f16_tensor(data: &[f32], shape: &[usize]) -> Tensor<f16> {
    let d: Vec<f16> = data.iter().map(|&v| f16::from_f32(v)).collect();
    Tensor::new(d, shape).unwrap()
}

fn f16_to_f32_vec(t: &Tensor<f16>) -> Vec<f32> {
    t.host_slice().iter().map(|v| v.to_f32()).collect()
}

// ---------------------------------------------------------------------
// (a) 実機非依存
// ---------------------------------------------------------------------

#[test]
fn typed_ops_f16_is_some_and_typed_ops_f64_is_none() {
    let ops = MetalBackendOps::new();
    assert!(BackendOps::typed_ops_f16(&ops).is_some());
    assert!(BackendOps::typed_ops_f64(&ops).is_none());
}

#[test]
fn gemm_rejects_shape_mismatch_before_touching_device() {
    let ops = MetalBackendOps::new();
    let a = f16_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
    let b = f16_tensor(&[1.0, 2.0], &[2, 1]);
    let err = TypedOps::<f16>::gemm(&ops, &a, &b).unwrap_err();
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}

#[test]
fn gemm_rejects_batched_rank3_input_before_touching_device() {
    // イシュー #1715 で `matmul_out_shape`（バッチ対応・rank>=3 許容）が
    // 一般化されたことに伴い、`TypedOps<f16>::gemm`（2 次元専用の GEMM
    // カーネル入口。`docs/backend-dtype-dispatch-design.md` 同型の他
    // バックエンド〈CPU/CUDA〉と同じ契約）はカーネル専用の 2 次元厳密版
    // `gemm_out_shape` で検証しなければならない（PR #1810 codex-review
    // 指摘）。`matmul_out_shape` のまま呼ぶとバッチ入力の先頭 2 軸を
    // 誤って m/k として使い、デバイスへ誤った行列寸法で到達しうる
    // （境界検査の後退）。rank=3 のバッチ入力が
    // `ShapeError::RankMismatch { expected: 2, .. }` で拒否されることを
    // 確認し、この後退を回帰検知する。
    let ops = MetalBackendOps::new();
    let a = f16_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 1, 2]);
    let b = f16_tensor(&[1.0, 2.0], &[2, 1]);
    let err = TypedOps::<f16>::gemm(&ops, &a, &b).unwrap_err();
    match err {
        BackendError::ShapeMismatch(ShapeError::RankMismatch { expected, actual }) => {
            assert_eq!(expected, 2);
            assert_eq!(actual, 3);
        }
        other => panic!("RankMismatch を期待したが {other:?} だった"),
    }
}

#[test]
fn add_mul_reject_shape_mismatch_before_touching_device() {
    let ops = MetalBackendOps::new();
    let a = f16_tensor(&[1.0, -2.0, 3.0], &[3]);
    let b = f16_tensor(&[3.0, 4.0], &[2]);
    let add_err = TypedOps::<f16>::add(&ops, &a, &b).unwrap_err();
    assert!(matches!(add_err, BackendError::ShapeMismatch(_)));
    let mul_err = TypedOps::<f16>::mul(&ops, &a, &b).unwrap_err();
    assert!(matches!(mul_err, BackendError::ShapeMismatch(_)));
}

/// `max` が Metal f32 reduction 未実装（`ops::MetalBackendOps::max`）
/// と同じ `Unsupported` を返すことを、デバイス構築すら経由せず確認する
/// （`crate::typed_f16` の 3 段構成〈昇格→委譲→丸め〉が委譲先の
/// `Unsupported` をそのまま伝播することの直接検証）。`sum`（イシュー
/// #1896 で結線済み）は `sum_matches_f32_backend_ops_rounded_bit_exact`
/// が担う。
#[test]
fn max_is_unsupported_matching_metal_f32_backend_ops() {
    let ops = MetalBackendOps::new();
    let a = f16_tensor(&[1.0, 5.0, 3.0, 2.0], &[2, 2]);
    for dim in [None, Some(0), Some(5)] {
        assert!(matches!(
            TypedOps::<f16>::max(&ops, &a, dim),
            Err(BackendError::Unsupported(_))
        ));
    }
}

/// `sum`（イシュー #1896 で `reduce::MetalReduce` へ結線済み）が
/// `f16::from_f32(BackendOps::sum(f32))` と要素ごと bit 一致すること・
/// 範囲外 `dim` は両者 `ShapeMismatch` になることを Metal 実機で検証
/// する（`crate::typed_f16` モジュール doc「`sum` は結線済み」参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn sum_matches_f32_backend_ops_rounded_bit_exact() {
    let ops = MetalBackendOps::new();
    let a16 = f16_tensor(&[1.0, 5.0, 3.0, 2.0], &[2, 2]);
    let a32: Vec<f32> = f16_to_f32_vec(&a16);
    let a32_t = Tensor::new(a32, a16.shape()).expect("tensor");

    for dim in [None, Some(0)] {
        let v16 = TypedOps::<f16>::sum(&ops, &a16, dim).expect("metal f16 sum");
        let v32 = BackendOps::sum(&ops, &a32_t, dim).expect("metal f32 sum");
        let rounded32: Vec<f32> = v32
            .host_slice()
            .iter()
            .map(|&x| f16::from_f32(x).to_f32())
            .collect();
        assert_eq!(f16_to_f32_vec(&v16), rounded32);
    }

    let err16 = TypedOps::<f16>::sum(&ops, &a16, Some(5));
    let err32 = BackendOps::sum(&ops, &a32_t, Some(5));
    assert!(matches!(err16, Err(BackendError::ShapeMismatch(_))));
    assert!(matches!(err32, Err(BackendError::ShapeMismatch(_))));
}

/// 零次元形状（`[0,3]×[3,2]`）は `gemm_out_shape`（numel 0 は合法）を
/// 通過してデバイスへ到達したうえで `validate_dims_f16` の
/// `ZeroDimension` により拒否される（`crate::typed_f16` 実装計画 §3
/// 決定 4 の既知事項）。panic せず型付きエラーで返ることを確認する
/// （デバイス到達が必要なため `#[ignore]`）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_zero_dim_shape_reaches_device_and_fails_without_panic() {
    let ops = MetalBackendOps::new();
    let a: Tensor<f16> = Tensor::new(Vec::new(), &[0, 3]).unwrap();
    let b = f16_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    let err = TypedOps::<f16>::gemm(&ops, &a, &b).unwrap_err();
    assert!(matches!(err, BackendError::KernelLaunchFailed(_)));
}

// ---------------------------------------------------------------------
// (b) `#[ignore]` 実機依存
// ---------------------------------------------------------------------

/// `TypedOps<f16>::gemm`（`crate::typed_f16`。`ops.rs` の accessor 経由）
/// が `MetalGemm::dispatch_f16_auto_unverified` の直接呼び出しと bit
/// 単位で完全一致することを確認する（結線がパススルーであることの
/// 直接検証。イシュー #1705 実装計画 §5.2 L1 gemm）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn typed_f16_gemm_is_bit_identical_to_dispatch_f16_auto_unverified() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 タイル化含む）");
    let ops = MetalBackendOps::new();

    for (seed, m, n, k) in [(600u64, 16usize, 16usize, 16usize), (601, 64, 128, 32)] {
        let mut rng = Xorshift64Star::new(seed);
        let a_f16: Vec<f16> = rng.fill_vec_f16(m * k);
        let b_f16: Vec<f16> = rng.fill_vec_f16(k * n);

        let direct = gemm
            .dispatch_f16_auto_unverified(&ctx, &a_f16, &b_f16, m, n, k)
            .expect("dispatch_f16_auto_unverified must succeed on Metal-equipped test runner");

        let a_tensor = Tensor::new(a_f16, &[m, k]).unwrap();
        let b_tensor = Tensor::new(b_f16, &[k, n]).unwrap();
        let via_typed_ops = TypedOps::<f16>::gemm(&ops, &a_tensor, &b_tensor)
            .expect("TypedOps::<f16>::gemm must succeed on Metal-equipped test runner");

        assert_eq!(
            f16_to_f32_vec(&via_typed_ops),
            direct.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
            "TypedOps<f16>::gemm must be bit-identical to dispatch_f16_auto_unverified for m={m} n={n} k={k}"
        );
    }
}

/// `TypedOps<f16>::gemm` の数値が CPU f32 参照実装（`matmul_reference_fma`）
/// を f16 へ丸めた値と REQ-2 統一複合判定で一致することを確認する
/// （`tests/gemm_f16_auto_parity.rs` と同一の判定基盤・入力生成規則。
/// `crates/backend-cuda/tests/typed_ops_f16_parity.rs` と同じ構成）。
///
/// 参照計算には CUDA 版と同じ理由で「f16 へ丸めた後 f32 へ復元した値」を
/// 使う（丸め前の入力をそのまま参照計算へ渡すと「入力の量子化誤差」と
/// 「GEMM 演算自体の誤差」が REQ-2 複合判定に混在し、Metal 側の演算経路
/// 〈f16→f32 昇格は行わず f16 のまま `dispatch_f16_auto_unverified` へ渡す〉
/// 自体の正しさを検証できなくなる）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn typed_f16_gemm_matches_cpu_reference_rounded() {
    let ops = MetalBackendOps::new();
    let (m, n, k) = (256usize, 192usize, 320usize);
    let mut rng = Xorshift64Star::new(602);
    let a_f16: Vec<f16> = rng.fill_vec_f16(m * k);
    let b_f16: Vec<f16> = rng.fill_vec_f16(k * n);

    let a_tensor = Tensor::new(a_f16.clone(), &[m, k]).unwrap();
    let b_tensor = Tensor::new(b_f16.clone(), &[k, n]).unwrap();
    let gpu = TypedOps::<f16>::gemm(&ops, &a_tensor, &b_tensor)
        .expect("TypedOps::<f16>::gemm must succeed on Metal-equipped test runner");

    let a_f32_rounded: Vec<f32> = a_f16.iter().map(|v| v.to_f32()).collect();
    let b_f32_rounded: Vec<f32> = b_f16.iter().map(|v| v.to_f32()).collect();
    let mut reference_f32 = vec![0.0f32; m * n];
    matmul_reference_fma(&a_f32_rounded, &b_f32_rounded, &mut reference_f32, m, n, k)
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let reference_rounded: Vec<f32> = reference_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    assert_parity(
        "TypedOps<f16>::gemm vs CPU reference",
        &f16_to_f32_vec(&gpu),
        &reference_rounded,
    );
}

/// `add`／`mul`／`relu`／`exp`／`tanh`（f16）が CPU `BackendOps`（f32）を
/// f16 へ丸めた値と一致することを確認する（broadcast・単項演算を含む。
/// `sum` は `sum_matches_f32_backend_ops_rounded_bit_exact` が別途担い、
/// `max` は Metal f32 reduction 未実装のため対象外）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn typed_f16_elementwise_matches_cpu_backend_ops_rounded() {
    let metal = MetalBackendOps::new();
    let cpu = CpuBackendOps::new();

    // [2,3] + [3]（行方向ブロードキャスト）。値は f16 exp/tanh が有限
    // 範囲に収まるよう小さめに抑える（U[-1,1) 程度）。
    let a32 = [0.5f32, -0.25, 0.75, -0.5, 0.25, -0.75];
    let b32 = [0.1f32, 0.2, 0.3];
    let a16 = f16_tensor(&a32, &[2, 3]);
    let b16 = f16_tensor(&b32, &[3]);

    // CPU 参照側は Metal 側が実際に使う値（f16 へ丸めた後の値。
    // `typed_f16::upcast_f16` が `f16::to_f32` で復元する値と同一）を
    // 使う（CUDA 版と同じ是正。丸め境界の混在を避ける）。
    let a32_rounded: Vec<f32> = a32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let b32_rounded: Vec<f32> = b32.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    let a32_full = Tensor::new(a32_rounded, &[2, 3]).unwrap();
    let b32_full = Tensor::new(b32_rounded, &[3]).unwrap();

    macro_rules! check_binary {
        ($name:literal, $op:ident) => {
            let metal_out = TypedOps::<f16>::$op(&metal, &a16, &b16)
                .unwrap_or_else(|e| panic!("metal {} failed: {e}", $name));
            let cpu_out = BackendOps::$op(&cpu, &a32_full, &b32_full)
                .unwrap_or_else(|e| panic!("cpu {} failed: {e}", $name));
            let cpu_rounded: Vec<f32> = cpu_out
                .host_slice()
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect();
            assert_parity(
                &format!("typed f16 {} vs cpu f32 rounded", $name),
                &f16_to_f32_vec(&metal_out),
                &cpu_rounded,
            );
        };
    }
    check_binary!("add", add);
    check_binary!("mul", mul);

    macro_rules! check_unary {
        ($name:literal, $op:ident) => {
            let metal_out = TypedOps::<f16>::$op(&metal, &a16)
                .unwrap_or_else(|e| panic!("metal {} failed: {e}", $name));
            let cpu_out = BackendOps::$op(&cpu, &a32_full)
                .unwrap_or_else(|e| panic!("cpu {} failed: {e}", $name));
            let cpu_rounded: Vec<f32> = cpu_out
                .host_slice()
                .iter()
                .map(|&v| f16::from_f32(v).to_f32())
                .collect();
            assert_parity(
                &format!("typed f16 {} vs cpu f32 rounded", $name),
                &f16_to_f32_vec(&metal_out),
                &cpu_rounded,
            );
        };
    }
    check_unary!("relu", relu);
    check_unary!("exp", exp);
    check_unary!("tanh", tanh);
}
