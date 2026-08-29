//! Metal サイズクラス・プールアロケータ（イシュー #1021）の実機ロード
//! テスト。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（Linux）
//! では `#![cfg(target_os = "macos")]` によりコンパイル対象外になり、
//! `#[ignore]` により通常の `cargo test` からも除外される
//! （`tests/memory_roundtrip.rs`・`tests/context_cache_bench.rs` と
//! 同じ構成。`.claude/rules/coding-rust.md`）。
//!
//! `crate::pool`／`crate::context_cache::cached_allocator` はいずれも
//! `pub(crate)` のため、本ファイル（外部クレート扱いの統合テスト）から
//! 直接は到達できない。プールへの唯一の到達経路は公開 `BackendOps`
//! （[`MetalBackendOps`]）の `memory_ops()`・
//! `release_cached_device_memory()`・`device_memory_pool_stats()` の
//! 3 メソッドであり、本ファイルはこれらのみを使って検証する（利用者が
//! 実際に到達できる経路のみを検証する方が REQ-14 の受け入れ条件
//! 「利用者から到達できる」の裏付けとしても適切）。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --test pool_real_device -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `--test-threads=1` が必須の理由: `crate::context_cache::
//! cached_allocator`（本ファイルからは `MetalBackendOps` 経由でのみ触れる）
//! はプロセスワイド singleton（`context_cache_bench.rs` と同じ理由）
//! であり、統計スナップショット（`device_memory_pool_stats()`）を
//! 用いた `cached_bytes` の検証は他テストの並行確保・解放と競合すると
//! フレーキーになる。

#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::BackendOps;
use fandhe_ai_tensor_core::Tensor;

/// `MemoryOps::alloc_zeroed` で確保したプール経由バッファを `Drop` した
/// 後、`device_memory_pool_stats()` の `cached_bytes` がその確保に相当
/// する分だけ増加することを確認する（設計文書 §3.1「不変条件」:
/// `Drop` された貸出中ハンドルはフリーリストへ返却される。本テストの
/// 確保はコマンドバッファ dispatch を伴わないため `open`／`committed`
/// バッチが存在せず、返却は即時 `put()` 経路〈設計文書 §3.3「Metal」〉
/// を通る）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dropping_pooled_alloc_zeroed_returns_bytes_to_free_list() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal デバイス上で memory_ops が取得できるはず");

    // 解放前の基準値（他テストとの並行実行を避けるため単独プロセスで
    // 実行する前提だが、念のため差分で判定する）。
    let before = ops
        .device_memory_pool_stats()
        .expect("device_memory_pool_stats は Some を返すはず")
        .cached_bytes;

    {
        let buf = mem
            .alloc_zeroed(&[1024])
            .expect("alloc_zeroed は成功するはず");
        // 貸出中は同一クラスのバイト数がフリーリストに現れないはず。
        let during = ops.device_memory_pool_stats().unwrap().cached_bytes;
        assert!(
            during <= before,
            "貸出中はフリーリストへ計上されないはず（during={during}, before={before}）"
        );
        drop(buf);
    }

    let after = ops.device_memory_pool_stats().unwrap().cached_bytes;
    assert!(
        after > before,
        "Drop 後はフリーリストへ返却され cached_bytes が増加するはず（before={before}, after={after}）"
    );
}

/// 同一サイズクラスの確保を 2 回行うと、2 回目はフリーリストから再利用
/// され `reuse_count` が増加することを確認する（設計文書 §3.1
/// 「フィールド更新契約」`reuse_count` 行）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn second_alloc_of_same_class_reuses_freed_handle() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal デバイス上で memory_ops が取得できるはず");

    let before_reuse = ops
        .device_memory_pool_stats()
        .expect("device_memory_pool_stats は Some を返すはず")
        .reuse_count;

    let first = mem
        .alloc_zeroed(&[2048])
        .expect("1 回目の alloc_zeroed は成功するはず");
    drop(first);

    let second = mem
        .alloc_zeroed(&[2048])
        .expect("2 回目の alloc_zeroed は成功するはず");
    let tensor = mem
        .download(&second)
        .expect("download は成功するはず（再利用バッファの全要素 0 契約）");
    for &v in tensor.as_slice().unwrap() {
        assert_eq!(
            v, 0.0,
            "再利用ヒット後も alloc_zeroed の全要素 0 契約を満たすはず"
        );
    }
    drop(second);

    let after_reuse = ops.device_memory_pool_stats().unwrap().reuse_count;
    assert!(
        after_reuse > before_reuse,
        "同一クラスの 2 回目確保はフリーリスト再利用ヒットになるはず\
         （before={before_reuse}, after={after_reuse}）"
    );
}

/// `release_cached_device_memory()` 呼び出し後は `cached_bytes == 0`
/// になることを確認する（REQ-14 の解放 API 受け入れ条件。設計文書
/// §3.6 (2)「バックエンド別の該当フェーズ」表「Metal」列）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn release_cached_device_memory_empties_free_list() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal デバイス上で memory_ops が取得できるはず");

    let buf = mem
        .alloc_zeroed(&[4096])
        .expect("alloc_zeroed は成功するはず");
    drop(buf);

    let before_release = ops.device_memory_pool_stats().unwrap().cached_bytes;
    assert!(
        before_release > 0,
        "解放前はフリーリストにバイトが残っているはず"
    );

    ops.release_cached_device_memory()
        .expect("release_cached_device_memory は成功するはず");

    let after_release = ops.device_memory_pool_stats().unwrap().cached_bytes;
    assert_eq!(after_release, 0, "解放後は cached_bytes == 0 になるはず");
}

/// GEMM（プール経由 C バッファ）の数値一致がプール導入後も崩れて
/// いないことを確認する（`tests/cpu_metal_parity.rs` 等の既存複合判定
/// テストが引き続き green であることが主たる裏付けだが、本テストは
/// プール経由確保を直接使う `gemm` 呼び出し 1 本を独立に確認する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn gemm_via_pooled_output_buffer_matches_expected() {
    let ops = MetalBackendOps::new();
    let a = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let b = Tensor::<f32>::new(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();

    let out = ops.gemm(&a, &b).expect("gemm は成功するはず");
    let expected = [19.0, 22.0, 43.0, 50.0];
    for (actual, expected) in out.as_slice().unwrap().iter().zip(expected.iter()) {
        let diff = (actual - expected).abs();
        assert!(
            diff < 1e-5,
            "GEMM 結果が期待値と一致しないはず（actual={actual}, expected={expected}）"
        );
    }
}
