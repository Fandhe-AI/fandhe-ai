//! イシュー #1738: `shaders/nll.metal` に REQ-8 境界検査・決定的
//! reduction（`simd_sum` + 固定順序の threadgroup 間結合・`atomic` 系
//! 不使用）が実在することを機械検査する証跡テスト
//! （`bce_source_evidence.rs`〈#1737〉と同型構成）。
//!
//! `include_str!` によるビルド時文字列埋め込みへの contains 検査のみで
//! 完結するため、Metal 実機・`cfg(target_os = "macos")` を必要とせず
//! Linux CI 上でも green になる（Metal 実機・数値一致を直接検証する
//! `#[ignore]` テストは `nll_parity.rs`）。

/// `crates/backend-metal/src/shaders/nll.metal` のソース全文。
const NLL_METAL_SOURCE: &str = include_str!("../src/shaders/nll.metal");

/// REQ-8: `nll_partial_f32`・`nll_backward_f32` が手動境界チェックを
/// 維持していることをロックする。
#[test]
fn nll_metal_source_has_bound_checks() {
    assert!(
        NLL_METAL_SOURCE.contains("idx < n_samples"),
        "nll_partial_f32 のループ境界検査 `idx < n_samples` が見つかりません"
    );
    assert!(
        NLL_METAL_SOURCE.contains("if (idx < n_samples)"),
        "nll_backward_f32 の手動境界チェック `if (idx < n_samples)` が見つかりません"
    );
    assert!(
        NLL_METAL_SOURCE.contains("idx < num_partials"),
        "nll_finalize_f32 のループ境界検査 `idx < num_partials` が見つかりません"
    );
}

/// 決定性: simdgroup 内総和は `simd_sum`（Metal 組み込み）を使い、
/// 非決定的な `atomic` 系命令を一切使わないことをロックする。
#[test]
fn nll_metal_source_uses_simd_sum_without_atomics() {
    assert!(
        NLL_METAL_SOURCE.contains("simd_sum"),
        "nll.metal に simdgroup 内総和命令 `simd_sum` が見つかりません"
    );
    assert!(
        !NLL_METAL_SOURCE.to_lowercase().contains("atomic"),
        "nll.metal に atomic 系命令が含まれています（決定性契約違反）"
    );
}

/// threadgroup 間結合が `threadgroup_barrier` による明示的な同期を
/// 経由していることをロックする。
#[test]
fn nll_metal_source_uses_threadgroup_barrier() {
    assert!(
        NLL_METAL_SOURCE.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "nll.metal に threadgroup 間同期 `threadgroup_barrier` が見つかりません"
    );
}

/// REQ-8: `nll_partial_f32` の grid-stride ループ添字（`idx`／`stride`）
/// が `ulong` で宣言されていることをロックする（`n_samples` 近傍での
/// unsigned wraparound 回避）。
#[test]
fn nll_metal_partial_grid_stride_loop_index_is_declared_ulong() {
    assert!(
        NLL_METAL_SOURCE.contains("ulong stride = (ulong)grid_size * (ulong)tg_size;"),
        "stride が ulong で宣言されていない"
    );
    assert!(
        NLL_METAL_SOURCE.contains(
            "for (ulong idx = (ulong)tg_id * (ulong)tg_size + (ulong)tid; idx < n_samples; idx += stride)"
        ),
        "idx ループ添字が ulong で宣言されていない"
    );
}

/// 3 カーネルすべてが実在することをロックする（関数シグネチャの grep。
/// リネームや削除を検出する）。
#[test]
fn nll_metal_source_declares_all_three_kernels() {
    for needle in [
        "kernel void nll_partial_f32",
        "kernel void nll_finalize_f32",
        "kernel void nll_backward_f32",
    ] {
        assert!(
            NLL_METAL_SOURCE.contains(needle),
            "nll.metal にカーネル宣言 `{needle}` が見つかりません"
        );
    }
}

/// `targets` バッファが `int`（`device const int*`）で宣言されている
/// ことをロックする（`i32` 正解クラス添字の受け渡し契約）。
#[test]
fn nll_metal_source_declares_int_targets_buffer() {
    assert!(
        NLL_METAL_SOURCE.contains("device const int* targets"),
        "nll.metal に `device const int* targets` バッファ宣言が見つかりません"
    );
}

/// PR #1850 codex-review P0 是正 2 の証跡: `t`（`targets[idx]`）の
/// 手動境界検査（`0 <= t < num_classes`）がシェーダソース自身に
/// 含まれることを確認する。
#[test]
fn nll_metal_source_has_target_bound_checks() {
    assert!(
        NLL_METAL_SOURCE.contains("if (t < 0 || (uint)t >= num_classes)"),
        "nll_partial_f32 の target 範囲検査が見つかりません"
    );
    assert!(
        NLL_METAL_SOURCE.contains("if (t >= 0 && (uint)t < num_classes)"),
        "nll_backward_f32 の target 範囲検査が見つかりません"
    );
}
