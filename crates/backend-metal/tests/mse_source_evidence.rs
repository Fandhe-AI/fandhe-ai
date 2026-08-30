//! イシュー #1045: `shaders/mse.metal` に REQ-8 境界検査・決定的
//! reduction（`simd_sum` + 固定順序の threadgroup 間結合・`atomic` 系
//! 不使用）が実在することを機械検査する証跡テスト。
//!
//! `shader_source_evidence.rs`（gemm.metal の `simdgroup_matrix` 命令
//! 実在検査）と同じ位置づけ: `include_str!` によるビルド時文字列埋め込み
//! への contains 検査のみで完結するため、Metal 実機・
//! `cfg(target_os = "macos")` を必要とせず Linux CI 上でも green になる
//! （本リポジトリで Metal 実機・数値一致を直接検証できる `#[ignore]`
//! テスト〈`mse_parity.rs`〉と異なり、本テストは Linux 上での唯一の
//! CUDA 側 `kernels_mse.rs::tests` に対応する証跡）。

/// `crates/backend-metal/src/shaders/mse.metal` のソース全文。
const MSE_METAL_SOURCE: &str = include_str!("../src/shaders/mse.metal");

/// REQ-8: `mse_partial_f32`・`mse_backward_f32` が手動境界チェックを
/// 維持していることをロックする。
#[test]
fn mse_metal_source_has_bound_checks() {
    assert!(
        MSE_METAL_SOURCE.contains("idx < numel"),
        "mse_partial_f32 のループ境界検査 `idx < numel` が見つかりません"
    );
    assert!(
        MSE_METAL_SOURCE.contains("if (idx < numel)"),
        "mse_backward_f32 の手動境界チェック `if (idx < numel)` が見つかりません"
    );
    assert!(
        MSE_METAL_SOURCE.contains("idx < num_partials"),
        "mse_finalize_f32 のループ境界検査 `idx < num_partials` が見つかりません"
    );
}

/// 決定性: simdgroup 内総和は `simd_sum`（Metal 組み込み）、FMA 契約は
/// `fma`（forward の 2 乗和累積）を使い、非決定的な `atomic` 系命令
/// （`atomic_fetch_add` 等）を一切使わないことをロックする（`kernels_mse.rs`
/// と同じ「float atomicAdd を使わない」決定性契約の Metal 側証跡）。
#[test]
fn mse_metal_source_uses_simd_sum_and_fma_without_atomics() {
    assert!(
        MSE_METAL_SOURCE.contains("simd_sum"),
        "mse.metal に simdgroup 内総和命令 `simd_sum` が見つかりません"
    );
    assert!(
        MSE_METAL_SOURCE.contains("fma("),
        "mse.metal に FMA 契約統一の `fma(...)` 呼び出しが見つかりません"
    );
    assert!(
        !MSE_METAL_SOURCE.to_lowercase().contains("atomic"),
        "mse.metal に atomic 系命令が含まれています（決定性契約違反）"
    );
}

/// threadgroup 間結合が `threadgroup_barrier` による明示的な同期を
/// 経由していることをロックする（`simd_sum` 結果を `threadgroup` メモリ
/// 経由で結合する契約。バリアなしでは他 simdgroup の書き込みが可視化
/// される保証がない）。
#[test]
fn mse_metal_source_uses_threadgroup_barrier() {
    assert!(
        MSE_METAL_SOURCE.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"),
        "mse.metal に threadgroup 間同期 `threadgroup_barrier` が見つかりません"
    );
}

/// REQ-8: `mse_partial_f32` の grid-stride ループ添字（`idx`／`stride`）が
/// `ulong` で宣言されていることをロックする
/// （`softmax.metal`／`rmsnorm.metal` の `ulong row_base` と同じ理由。
/// `numel` は `u32::MAX`（`mse.rs::validate_mse_len`）まで許容するため、
/// `uint` 添字のままだと `numel` 近傍で unsigned wraparound により
/// `idx < numel` の境界チェックを迂回しうる。Bugbot 指摘）。
#[test]
fn mse_metal_partial_grid_stride_loop_index_is_declared_ulong() {
    assert!(
        MSE_METAL_SOURCE.contains("ulong stride = (ulong)grid_size * (ulong)tg_size;"),
        "stride が ulong で宣言されていない"
    );
    assert!(
        MSE_METAL_SOURCE.contains(
            "for (ulong idx = (ulong)tg_id * (ulong)tg_size + (ulong)tid; idx < numel; idx += stride)"
        ),
        "idx ループ添字が ulong で宣言されていない"
    );
    assert!(
        !MSE_METAL_SOURCE.contains("for (uint idx = tg_id * tg_size + tid"),
        "grid-stride ループ添字が uint へ縮退している"
    );
}

/// 3 カーネルすべてが実在することをロックする（関数シグネチャの grep。
/// リネームや削除を検出する）。
#[test]
fn mse_metal_source_declares_all_three_kernels() {
    for needle in [
        "kernel void mse_partial_f32",
        "kernel void mse_finalize_f32",
        "kernel void mse_backward_f32",
    ] {
        assert!(
            MSE_METAL_SOURCE.contains(needle),
            "mse.metal にカーネル宣言 `{needle}` が見つかりません"
        );
    }
}
