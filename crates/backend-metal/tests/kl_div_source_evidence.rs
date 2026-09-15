//! イシュー #1738: `shaders/kl_div.metal` に REQ-8 境界検査・決定的
//! reduction（`simd_sum` + 固定順序の threadgroup 間結合・`atomic` 系
//! 不使用）が実在することを機械検査する証跡テスト
//! （`bce_source_evidence.rs`〈#1737〉と同型構成）。
//!
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI 上でも green になる（Metal 実機・数値一致を直接検証する
//! `#[ignore]` テストは `kl_div_parity.rs`）。

/// `crates/backend-metal/src/shaders/kl_div.metal` のソース全文。
const KL_DIV_METAL_SOURCE: &str = include_str!("../src/shaders/kl_div.metal");

/// REQ-8: `kl_div_partial_f32`・`kl_div_backward_f32` が手動境界
/// チェックを維持していることをロックする。
#[test]
fn kl_div_metal_source_has_bound_checks() {
    assert!(
        KL_DIV_METAL_SOURCE.contains("idx < numel"),
        "kl_div_partial_f32 のループ境界検査 `idx < numel` が見つかりません"
    );
    assert!(
        KL_DIV_METAL_SOURCE.contains("if (idx < numel)"),
        "kl_div_backward_f32 の手動境界チェック `if (idx < numel)` が見つかりません"
    );
    assert!(
        KL_DIV_METAL_SOURCE.contains("idx < num_partials"),
        "kl_div_finalize_f32 のループ境界検査 `idx < num_partials` が見つかりません"
    );
}

/// 決定性: simdgroup 内総和は `simd_sum`（Metal 組み込み）を使い、
/// 非決定的な `atomic` 系命令を一切使わないことをロックする。
#[test]
fn kl_div_metal_source_uses_simd_sum_without_atomics() {
    assert!(
        KL_DIV_METAL_SOURCE.contains("simd_sum"),
        "kl_div.metal に simdgroup 内総和命令 `simd_sum` が見つかりません"
    );
    assert!(
        !KL_DIV_METAL_SOURCE.to_lowercase().contains("atomic"),
        "kl_div.metal に atomic 系命令が含まれています（決定性契約違反）"
    );
}

/// threadgroup 間結合が `threadgroup_barrier` による明示的な同期を
/// 経由していることをロックする。
#[test]
fn kl_div_metal_source_uses_threadgroup_barrier() {
    assert!(
        KL_DIV_METAL_SOURCE.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "kl_div.metal に threadgroup 間同期 `threadgroup_barrier` が見つかりません"
    );
}

/// REQ-8: `kl_div_partial_f32` の grid-stride ループ添字（`idx`／
/// `stride`）が `ulong` で宣言されていることをロックする（`numel`
/// 近傍での unsigned wraparound 回避）。
#[test]
fn kl_div_metal_partial_grid_stride_loop_index_is_declared_ulong() {
    assert!(
        KL_DIV_METAL_SOURCE.contains("ulong stride = (ulong)grid_size * (ulong)tg_size;"),
        "stride が ulong で宣言されていない"
    );
    assert!(
        KL_DIV_METAL_SOURCE.contains(
            "for (ulong idx = (ulong)tg_id * (ulong)tg_size + (ulong)tid; idx < numel; idx += stride)"
        ),
        "idx ループ添字が ulong で宣言されていない"
    );
}

/// 3 カーネルすべてが実在することをロックする（関数シグネチャの grep。
/// リネームや削除を検出する）。
#[test]
fn kl_div_metal_source_declares_all_three_kernels() {
    for needle in [
        "kernel void kl_div_partial_f32",
        "kernel void kl_div_finalize_f32",
        "kernel void kl_div_backward_f32",
    ] {
        assert!(
            KL_DIV_METAL_SOURCE.contains(needle),
            "kl_div.metal にカーネル宣言 `{needle}` が見つかりません"
        );
    }
}

/// `target == 0` 分岐（`Probabilities` の `xlogy` 規約。forward の
/// `l = 0` 分岐）が実在することをロックする（`docs/compat-api-scope.md`
/// §1.2 参照）。
#[test]
fn kl_div_metal_source_handles_target_zero_branch() {
    assert!(
        KL_DIV_METAL_SOURCE.contains("target == 0.0f"),
        "kl_div.metal に target==0 分岐が見つかりません"
    );
}
