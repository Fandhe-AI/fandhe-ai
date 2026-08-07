//! TASK-1.9d（#47）: Metal 実機での `BackendOps` 経由数値一致検証。
//!
//! `cpu_metal_parity.rs`（TASK-2.2c・#55）は `MetalGemm::dispatch` を直接
//! 呼び出す形で CPU-Metal ペアの数値一致を検証しているが、本ファイルは
//! 抽象層 `tensor_core::backend_ops::MetalBackendOps`（TASK-1.9c・#46）が
//! 内部で使う `MetalGemm::dispatch_auto`（動的タイル選択。TASK-1.8c・#40）
//! を経由した場合にも同じ複合判定（REQ-2）が成立することを固定する。
//! 判定式・許容誤差は再定義せず `backend_cpu::parity` を唯一の参照とする
//! （`.claude/rules/coding-rust.md`）。
//!
//! `cpu_metal_parity.rs`（基準形状 512^3・K=4096 ストレス）とは異なる形状を
//! 選び、`tile.rs::select` の動的タイル選択境界
//! （`SMALL=64`・`LARGE=512`。`crate::tile` 参照）近傍を 1〜2 ケース含める。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`cpu_metal_parity.rs` と同方針。`#![cfg(target_os = "macos")]` により
//! Linux self-hosted CI ではコンパイル対象外になり、`#[ignore]` により
//! 通常の `cargo test` からも除外される）。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨。
//! `docs/backend-metal-real-device-testing.md` 参照）:
//!
//! ```sh
//! cargo test -p backend-metal --release -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use backend_cpu::CpuBackendOps;
use backend_metal::MetalBackendOps;
use bench_harness::rng::Xorshift64Star;
use tensor_core::device::{BackendError, Device};
use tensor_core::{BackendOps, Tensor};

/// `MetalBackendOps::gemm`（`dispatch_auto` 委譲）を CPU `BackendOps::gemm`
/// と複合判定で突き合わせる。
fn assert_backend_ops_gemm_parity(seed_a: u64, seed_b: u64, m: usize, n: usize, k: usize) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a_data = Xorshift64Star::new(seed_a).fill_vec(m * k);
    let b_data = Xorshift64Star::new(seed_b).fill_vec(k * n);
    let a = Tensor::new(a_data, &[m, k]).expect("valid tensor");
    let b = Tensor::new(b_data, &[k, n]).expect("valid tensor");

    let cpu_result = cpu.gemm(&a, &b).expect("cpu gemm always succeeds");
    let metal_result = metal
        .gemm(&a, &b)
        .expect("MetalBackendOps::gemm must succeed on Metal-equipped test runner");

    assert_eq!(metal_result.shape(), cpu_result.shape());
    backend_cpu::parity::assert_parity(
        &format!("BackendOps cpu-metal gemm parity m={m} n={n} k={k}"),
        metal_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );
}

/// 基準ケース（`tile.rs::select` の中形状経路。SMALL=64 以上・LARGE=512
/// 未満で正方形状 → `CANDIDATES[3]`〈32x32〉が選ばれる想定）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_gemm_matches_cpu_mid_square_shape() {
    assert_backend_ops_gemm_parity(501, 502, 96, 96, 96);
}

/// `tile.rs::select` の動的タイル選択境界（`SMALL=64`）近傍ケース:
/// `m/n/k` のいずれかが 64 未満のとき `SINGLE_SIMDGROUP_8X8` へ分岐する
/// （`tile.rs` 参照）。63（境界未満）・65（境界以上）を隣接させて選択境界の
/// 両側を実機で確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_gemm_matches_cpu_near_small_tile_threshold() {
    assert_backend_ops_gemm_parity(503, 504, 63, 63, 63);
    assert_backend_ops_gemm_parity(505, 506, 65, 65, 65);
}

/// `tile.rs::select` の縦長・横長分岐（`ASPECT_RATIO=2`）ケース。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_gemm_matches_cpu_tall_and_wide_shapes() {
    assert_backend_ops_gemm_parity(507, 508, 128, 64, 96); // 縦長（m >= 2n）
    assert_backend_ops_gemm_parity(509, 510, 64, 128, 96); // 横長（n >= 2m）
}

/// 非正方・非 2 冪境界（`cpu_metal_parity.rs` の基準形状 512^3 とは異なる
/// 素数近傍形状）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn backend_ops_gemm_matches_cpu_non_power_of_two_shape() {
    assert_backend_ops_gemm_parity(511, 512, 97, 71, 83);
}

/// elementwise・reduction の `Unsupported` 契約を検証する。
///
/// `MetalBackendOps::add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`
/// （`backend-metal/src/ops.rs`）はいずれも `MetalContext::new` を呼ばず
/// 即座に `BackendError::Unsupported` を返す実装（TASK-1.9c スコープ外の
/// 未実装カーネル用プレースホルダ）のため、本テストは Metal 実機・
/// デバイス初期化を一切必要としない。他の実機依存テストと同じファイルに
/// 置かれてはいるが（`MetalBackendOps` 自体が `cfg(target_os = "macos")`
/// 限定のため非 macOS 環境ではコンパイル対象に入らない）、実機依存では
/// ないため `#[ignore]` を付けない。macOS 上での通常の
/// `cargo test -p backend-metal`（`--ignored` なし）で毎回実行され、
/// `MetalBackendOps::gemm` 以外の 7 演算が `Unsupported` を返し続ける
/// ことを回帰的に固定する。
///
/// `backend_ops_dispatch.rs`（`backend-cpu/tests/`）は `ops_for` 経由の
/// GEMM ディスパッチのみを検証しており、Metal の elementwise・reduction
/// カバレッジは含まない（Cursor Bugbot 指摘・PR #264 レビュースレッド。
/// 旧コメントは誤って「テストと分離していない」と記述していたが、実際は
/// 「そもそもカバーしていない」が正確な記述である）。
#[test]
fn elementwise_and_reduction_remain_unsupported_without_device_init() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![1.0, -2.0, 3.0, -4.0], &[2, 2]).expect("valid tensor");
    let b = a.clone();

    assert!(matches!(
        metal.add(&a, &b),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        metal.mul(&a, &b),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(metal.relu(&a), Err(BackendError::Unsupported(_))));
    assert!(matches!(metal.exp(&a), Err(BackendError::Unsupported(_))));
    assert!(matches!(metal.tanh(&a), Err(BackendError::Unsupported(_))));
    assert!(matches!(
        metal.sum(&a, None),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        metal.max(&a, None),
        Err(BackendError::Unsupported(_))
    ));
}

/// `ops_for`（`tensor_core::backend_ops`）を介したディスパッチでも同じ
/// 数値一致が成立することを固定する（`Device::Metal` 選択の回帰保護）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn ops_for_selects_metal_backend_and_matches_cpu() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let ops: Vec<&dyn BackendOps> = vec![&cpu, &metal];

    let a_data = Xorshift64Star::new(513).fill_vec(80 * 80);
    let b_data = Xorshift64Star::new(514).fill_vec(80 * 80);
    let a = Tensor::new(a_data, &[80, 80]).expect("valid tensor");
    let b = Tensor::new(b_data, &[80, 80]).expect("valid tensor");

    let cpu_result = tensor_core::ops_for(&ops, Device::Cpu)
        .expect("cpu ops registered")
        .gemm(&a, &b)
        .expect("cpu gemm always succeeds");
    let metal_result = tensor_core::ops_for(&ops, Device::Metal)
        .expect("metal ops registered")
        .gemm(&a, &b)
        .expect("MetalBackendOps::gemm must succeed on Metal-equipped test runner");

    backend_cpu::parity::assert_parity(
        "ops_for-dispatched metal gemm vs cpu",
        metal_result.as_slice().expect("contiguous"),
        cpu_result.as_slice().expect("contiguous"),
    );
}
