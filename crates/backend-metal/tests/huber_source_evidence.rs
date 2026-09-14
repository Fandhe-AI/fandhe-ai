//! イシュー #1739: `shaders/huber.metal` に REQ-8 境界検査・決定的
//! reduction（`simd_sum` + 固定順序の threadgroup 間結合・`atomic` 系
//! 不使用）・`copysign` 使用が実在することを機械検査する証跡テスト
//! （`mse_source_evidence.rs`／`bce_source_evidence.rs` と同型。
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI 上でも green になる）。

/// `crates/backend-metal/src/shaders/huber.metal` のソース全文。
const HUBER_METAL_SOURCE: &str = include_str!("../src/shaders/huber.metal");

/// REQ-8: `huber_partial_f32`・`huber_backward_f32` が手動境界チェックを
/// 維持していることをロックする。
#[test]
fn huber_metal_source_has_bound_checks() {
    assert!(
        HUBER_METAL_SOURCE.contains("idx < numel"),
        "huber_partial_f32 のループ境界検査 `idx < numel` が見つかりません"
    );
    assert!(
        HUBER_METAL_SOURCE.contains("if (idx < numel)"),
        "huber_backward_f32 の手動境界チェック `if (idx < numel)` が見つかりません"
    );
    assert!(
        HUBER_METAL_SOURCE.contains("idx < num_partials"),
        "huber_finalize_f32 のループ境界検査 `idx < num_partials` が見つかりません"
    );
}

/// 決定性: simdgroup 内総和は `simd_sum`（Metal 組み込み）を使い、
/// 非決定的な `atomic` 系命令（`atomic_fetch_add` 等）を一切使わない
/// ことをロックする（`kernels_huber.rs` と同じ「float atomicAdd を
/// 使わない」決定性契約の Metal 側証跡）。
#[test]
fn huber_metal_source_uses_simd_sum_without_atomics() {
    assert!(
        HUBER_METAL_SOURCE.contains("simd_sum"),
        "huber.metal に simdgroup 内総和命令 `simd_sum` が見つかりません"
    );
    assert!(
        !HUBER_METAL_SOURCE.to_lowercase().contains("atomic"),
        "huber.metal に atomic 系命令が含まれています（決定性契約違反）"
    );
}

/// `sign(d)` の統一契約（`.claude/rules/coding-rust.md`）: Metal 側は
/// 組み込み `copysign` を使う。
#[test]
fn huber_metal_source_uses_copysign() {
    assert!(
        HUBER_METAL_SOURCE.contains("copysign("),
        "huber.metal に `copysign(...)` 呼び出しが見つかりません"
    );
}

/// threadgroup 間結合が `threadgroup_barrier` による明示的な同期を
/// 経由していることをロックする（`mse_source_evidence.rs` と同型）。
#[test]
fn huber_metal_source_uses_threadgroup_barrier() {
    assert!(
        HUBER_METAL_SOURCE.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "huber.metal に threadgroup 間同期 `threadgroup_barrier` が見つかりません"
    );
}

/// REQ-8: `huber_partial_f32` の grid-stride ループ添字（`idx`／
/// `stride`）が `ulong` で宣言されていることをロックする
/// （`mse_source_evidence.rs::mse_metal_partial_grid_stride_loop_index_is_declared_ulong`
/// と同型・同じ理由）。
#[test]
fn huber_metal_partial_grid_stride_loop_index_is_declared_ulong() {
    assert!(
        HUBER_METAL_SOURCE.contains("ulong stride = (ulong)grid_size * (ulong)tg_size;"),
        "stride が ulong で宣言されていない"
    );
    assert!(
        HUBER_METAL_SOURCE.contains(
            "for (ulong idx = (ulong)tg_id * (ulong)tg_size + (ulong)tid; idx < numel; idx += stride)"
        ),
        "idx ループ添字が ulong で宣言されていない"
    );
    assert!(
        !HUBER_METAL_SOURCE.contains("for (uint idx = tg_id * tg_size + tid"),
        "grid-stride ループ添字が uint へ縮退している"
    );
}

/// 3 カーネルすべてが実在することをロックする（関数シグネチャの grep。
/// リネームや削除を検出する）。
#[test]
fn huber_metal_source_declares_all_three_kernels() {
    for needle in [
        "kernel void huber_partial_f32",
        "kernel void huber_finalize_f32",
        "kernel void huber_backward_f32",
    ] {
        assert!(
            HUBER_METAL_SOURCE.contains(needle),
            "huber.metal にカーネル宣言 `{needle}` が見つかりません"
        );
    }
}

/// `kind`／`delta` 引数が forward・backward 双方のカーネル引数リストに
/// 現れることをロックする（区分損失の分岐材料が欠落していないことの
/// 証跡）。
#[test]
fn huber_metal_source_declares_kind_and_delta_arguments() {
    assert!(
        HUBER_METAL_SOURCE.contains("constant uint& kind [[buffer(4)]]"),
        "kind 引数（buffer(4)）が見つかりません"
    );
    assert!(
        HUBER_METAL_SOURCE
            .matches("constant float& delta [[buffer(5)]]")
            .count()
            >= 2,
        "delta 引数（buffer(5)）が forward・backward 双方に見つかりません"
    );
}
