//! イシュー #1737: `shaders/bce.metal` に REQ-8 境界検査・決定的
//! reduction（`simd_sum` + 固定順序の threadgroup 間結合・`atomic` 系
//! 不使用）が実在することを機械検査する証跡テスト（`mse_source_evidence.rs`
//! と同型構成）。
//!
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI 上でも green になる（Metal 実機・数値一致を直接検証する
//! `#[ignore]` テストは `bce_parity.rs`）。

/// `crates/backend-metal/src/shaders/bce.metal` のソース全文。
const BCE_METAL_SOURCE: &str = include_str!("../src/shaders/bce.metal");

/// REQ-8: `bce_partial_f32`・`bce_backward_f32` が手動境界チェックを
/// 維持していることをロックする。
#[test]
fn bce_metal_source_has_bound_checks() {
    assert!(
        BCE_METAL_SOURCE.contains("idx < numel"),
        "bce_partial_f32 のループ境界検査 `idx < numel` が見つかりません"
    );
    assert!(
        BCE_METAL_SOURCE.contains("if (idx < numel)"),
        "bce_backward_f32 の手動境界チェック `if (idx < numel)` が見つかりません"
    );
    assert!(
        BCE_METAL_SOURCE.contains("idx < num_partials"),
        "bce_finalize_f32 のループ境界検査 `idx < num_partials` が見つかりません"
    );
}

/// 決定性: simdgroup 内総和は `simd_sum`（Metal 組み込み）を使い、
/// 非決定的な `atomic` 系命令を一切使わないことをロックする
/// （`kernels_bce.rs` と同じ「float atomicAdd を使わない」決定性契約の
/// Metal 側証跡）。
#[test]
fn bce_metal_source_uses_simd_sum_without_atomics() {
    assert!(
        BCE_METAL_SOURCE.contains("simd_sum"),
        "bce.metal に simdgroup 内総和命令 `simd_sum` が見つかりません"
    );
    assert!(
        !BCE_METAL_SOURCE.to_lowercase().contains("atomic"),
        "bce.metal に atomic 系命令が含まれています（決定性契約違反）"
    );
}

/// threadgroup 間結合が `threadgroup_barrier` による明示的な同期を
/// 経由していることをロックする。
#[test]
fn bce_metal_source_uses_threadgroup_barrier() {
    assert!(
        BCE_METAL_SOURCE.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "bce.metal に threadgroup 間同期 `threadgroup_barrier` が見つかりません"
    );
}

/// REQ-8: `bce_partial_f32` の grid-stride ループ添字（`idx`／`stride`）が
/// `ulong` で宣言されていることをロックする（`mse_source_evidence.rs`
/// と同じ理由。`numel` 近傍での unsigned wraparound 回避）。
#[test]
fn bce_metal_partial_grid_stride_loop_index_is_declared_ulong() {
    assert!(
        BCE_METAL_SOURCE.contains("ulong stride = (ulong)grid_size * (ulong)tg_size;"),
        "stride が ulong で宣言されていない"
    );
    assert!(
        BCE_METAL_SOURCE.contains(
            "for (ulong idx = (ulong)tg_id * (ulong)tg_size + (ulong)tid; idx < numel; idx += stride)"
        ),
        "idx ループ添字が ulong で宣言されていない"
    );
    assert!(
        !BCE_METAL_SOURCE.contains("for (uint idx = tg_id * tg_size + tid"),
        "grid-stride ループ添字が uint へ縮退している"
    );
}

/// 3 カーネルすべてが実在することをロックする（関数シグネチャの grep。
/// リネームや削除を検出する）。
#[test]
fn bce_metal_source_declares_all_three_kernels() {
    for needle in [
        "kernel void bce_partial_f32",
        "kernel void bce_finalize_f32",
        "kernel void bce_backward_f32",
    ] {
        assert!(
            BCE_METAL_SOURCE.contains(needle),
            "bce.metal にカーネル宣言 `{needle}` が見つかりません"
        );
    }
}

/// `bce_log1p_f32`（MSL に組み込みが無い `log1p` の自作版。桁落ち回避の
/// 数値安定形）が実在することをロックする（`docs/compat-api-scope.md`
/// §1.2 の数値安定形要件・`kernels_bce.rs::BCE_DEVICE_FUNCS` の
/// `log1pf` 対応物）。
#[test]
fn bce_metal_source_defines_log1p_helper() {
    assert!(
        BCE_METAL_SOURCE.contains("bce_log1p_f32"),
        "bce.metal に自作 log1p ヘルパ `bce_log1p_f32` が見つかりません"
    );
}
