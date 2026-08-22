//! 融合 RMSNorm 順伝播・online softmax カーネル（`rmsnorm.rs`／`softmax.rs`）
//! が共有する cfg 非依存の純関数層（イシュー #604）。
//!
//! `objc2` 系 FFI に一切触れないため、[`crate::pad`]・[`crate::tile`] と
//! 同じ設計判断で `cfg(target_os = "macos")` を付けない（`lib.rs` 参照）。
//! Linux（CI・本実装環境）でも `cargo test -p fandhe-ai-backend-metal` で単体テスト
//! が回る。
//!
//! CUDA 側 `backend-cuda::rmsnorm`（イシュー #592・G-6）と同じ責務分割
//! （経路選択・persistent grid 導出・canonical 融合プラン照合・起動前
//! fail-closed 検証）を Metal 側の制約に合わせて実装する。**CUDA との
//! 決定的な設計差**: CUDA は `cuFuncSetAttribute` による opt-in 動的 SMEM
//! を使うのに対し、`MetalContext::dispatch_sync`（`context.rs`）は
//! `setThreadgroupMemoryLength` 相当の動的 threadgroup memory 設定 API を
//! 経由しないため、1 パス経路は MSL 側 `threadgroup float smem[N]`
//! （コンパイル時固定長。[`ONEPASS_MAX_HIDDEN`]）を使う。よって
//! [`derive_persistent_grid`] の `smem_bytes_per_group` は「実際に確保する
//! バイト数」（CUDA `hidden * 4`）ではなく「コンパイル時に宣言済みの固定
//! バイト数」（[`ONEPASS_SMEM_BYTES_PER_GROUP`]。宣言した configure
//! threadgroup memory は使用量に関わらず GPU が予約するため）を渡す。
//!
//! `lib.rs` で `pub(crate) mod row_kernel;`（`pub` にしない）としている。
//! 呼び出し元（`ops.rs`／`rmsnorm.rs`／`softmax.rs`）は macOS 限定の
//! ため、Linux 単体ビルド（`cargo build`／`cargo clippy` の非テスト
//! パス）では本モジュールの各項目が到達不能になり dead_code lint が
//! 誤検知する。`pub` へ広げて回避せず、non-macOS ビルドに限定した
//! 以下の `allow` で個別に抑制する（codex-review P1 指摘・PR #714。
//! `lib.rs` 側コメント参照）。
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use fandhe_ai_tensor_core::{DType, FusedOpKind, FusionPlan, RowFusionMeta};

/// 1 パス経路（threadgroup memory 常駐）が扱える最大行長（要素数）。
/// `shaders/rmsnorm.metal`／`shaders/softmax.metal` の
/// `threadgroup float smem[RMSNORM_ONEPASS_MAX_HIDDEN]` 等の固定長配列と
/// 一致させる単一の真実源（テスト
/// [`tests::onepass_max_hidden_matches_smem_budget`] が数値の一致を
/// ロックする。シェーダ側の定数を変更する場合は本値も同時に変更する）。
pub const ONEPASS_MAX_HIDDEN: usize = 4096;

/// 1 パス経路が宣言する threadgroup memory バイト数
/// （[`ONEPASS_MAX_HIDDEN`] * `size_of::<f32>()`）。Apple GPU の
/// threadgroup memory 上限は世代により異なるが、32KiB は Apple7 世代以降
/// で広く確保できる下限（`MTLDevice::maxThreadgroupMemoryLength` の
/// 一般的な下限値）であり、本値（16KiB）はそれに対し十分な余裕を持つ
/// （GEMM 側 `TileConfig::validate` が使う既定余白と同様、実機ごとの
/// `maxThreadgroupMemoryLength` 実測値との突合は行わず保守的な固定値を
/// 採用する設計判断）。
pub const ONEPASS_SMEM_BYTES_PER_GROUP: u64 = (ONEPASS_MAX_HIDDEN * 4) as u64;

/// persistent grid 導出における 1 コアあたりの同時常駐 threadgroup 数上限。
/// CUDA 側 `derive_persistent_grid_one_pass`／`derive_persistent_grid_two_pass`
/// と同じ経験的キャップ（16。TileKernels 参照実装由来。実測値ではなく
/// コード上の定数）を踏襲する。
pub const PERSISTENT_GRID_GROUPS_PER_CORE_CAP: u64 = 16;

/// `occupancy_params()` が `None`（GPU コア数取得不能）の場合の
/// フォールバック grid 数上限。実機値なしで「保守的な grid 数」へ縮退する
/// ための固定キャップ（実装計画 §4.4「fail-safe フォールバック」）。
pub const FALLBACK_GRID_CAP: u32 = 64;

/// [`select_route`] が返す経路選択（実装計画 §4.1「1 パス／2 パス経路」）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKernelRoute {
    /// `*_onepass` カーネル（threadgroup memory に行を常駐。device メモリの
    /// 再読なし）。
    OnePass,
    /// `*_twopass` カーネル（device メモリを再読。threadgroup memory 不使用）。
    TwoPass,
}

/// `hidden`（行長）が [`ONEPASS_MAX_HIDDEN`] に収まるかで経路を選ぶ純関数
/// （実機なしで単体テスト可能。CUDA `rmsnorm_route` の Metal 版）。
pub fn select_route(hidden: usize) -> RowKernelRoute {
    if hidden <= ONEPASS_MAX_HIDDEN {
        RowKernelRoute::OnePass
    } else {
        RowKernelRoute::TwoPass
    }
}

/// persistent threadgroup 方式の grid（threadgroup 数）を導出する純関数
/// （実装計画 §4.4。CUDA `derive_persistent_grid_one_pass`／
/// `derive_persistent_grid_two_pass` を 1 関数に統合した Metal 版）。
///
/// `groups_per_core = min(max_threadgroup_memory_bytes / smem_bytes_per_group,
/// CAP)`（`smem_bytes_per_group == 0` は 2 パス経路〈threadgroup memory
/// 制約なし〉を表し、CAP をそのまま使う）。
/// `grid = clamp(gpu_core_count * groups_per_core, 1, rows)`。
///
/// `rows == 0` は呼び出し元（[`crate::rmsnorm::MetalRmsNorm`]／
/// [`crate::softmax::MetalSoftmax`]）が早期 return する契約だが、
/// `clamp` パニックを避けるフェイルセーフとして 1 を返す（CUDA 版と同じ
/// 契約）。
pub fn derive_persistent_grid(
    gpu_core_count: u32,
    max_threadgroup_memory_bytes: u32,
    smem_bytes_per_group: u64,
    rows: u32,
) -> u32 {
    if rows == 0 {
        return 1;
    }
    let groups_per_core: u64 = if smem_bytes_per_group == 0 {
        PERSISTENT_GRID_GROUPS_PER_CORE_CAP
    } else {
        (max_threadgroup_memory_bytes as u64)
            .checked_div(smem_bytes_per_group)
            .map_or(PERSISTENT_GRID_GROUPS_PER_CORE_CAP, |v| {
                v.clamp(1, PERSISTENT_GRID_GROUPS_PER_CORE_CAP)
            })
    };
    let grid = (gpu_core_count as u64).saturating_mul(groups_per_core);
    grid.clamp(1, rows as u64) as u32
}

/// `MetalContext::occupancy_params()` が `None`（GPU コア数取得不能）の
/// 場合のフォールバック grid 導出（実装計画 §4.4「fail-safe フォール
/// バック」）。`rows` を [`FALLBACK_GRID_CAP`] で保守的にクランプするのみ
/// （threadgroup memory 予算は考慮しない。実機値が取れない状況での安全側
/// の縮退であり最適化しない）。
pub fn derive_persistent_grid_fallback(rows: u32) -> u32 {
    if rows == 0 {
        return 1;
    }
    rows.clamp(1, FALLBACK_GRID_CAP)
}

/// ホスト側検証（起動前・fail-closed）: `rows * hidden == x_len`（checked
/// 乗算）・`w_len == Some(hidden)`（`w` 指定時のみ）・`rows`／`hidden`／
/// `numel` が `i32::MAX`（カーネル引数の `uint rows`／`uint hidden` 契約。
/// 実引数型は `uint` だが `i32::MAX` を上限にする理由は CUDA 側
/// `validate_rmsnorm_launch` と同じくホスト側の符号付き演算・比較との
/// 整合を保つため）に収まること・`eps`（RMSNorm のみ。`None` は softmax
/// 呼び出しで eps 検証をスキップすることを表す）が有限かつ非負
/// （`is_finite() && eps >= 0.0`）であることを検証する
/// （`.claude/rules/security.md` A03。CUDA `validate_rmsnorm_launch` と
/// 同じ判断根拠: 負の `eps` は `sum(x^2) * inv_n + eps` を負化しうる）。
pub fn validate_row_kernel_launch(
    rows: usize,
    hidden: usize,
    x_len: usize,
    w_len: Option<usize>,
    eps: Option<f32>,
) -> Result<(), RowKernelValidationError> {
    if let Some(eps) = eps
        && (!eps.is_finite() || eps < 0.0)
    {
        return Err(RowKernelValidationError::InvalidEps { eps });
    }

    let numel = rows
        .checked_mul(hidden)
        .ok_or(RowKernelValidationError::DimOverflow { rows, hidden })?;
    if numel != x_len {
        return Err(RowKernelValidationError::XLenMismatch {
            expected: numel,
            actual: x_len,
        });
    }
    if let Some(wl) = w_len
        && wl != hidden
    {
        return Err(RowKernelValidationError::WLenMismatch {
            expected: hidden,
            actual: wl,
        });
    }
    if rows > i32::MAX as usize || hidden > i32::MAX as usize || numel > i32::MAX as usize {
        return Err(RowKernelValidationError::DimsExceedI32Max {
            rows,
            hidden,
            numel,
        });
    }
    Ok(())
}

/// [`validate_row_kernel_launch`] が返す fail-closed 検証エラー
/// （`rmsnorm.rs`／`softmax.rs` の両方が `crate::error::MetalError` へ
/// 変換する共通型）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RowKernelValidationError {
    InvalidEps {
        eps: f32,
    },
    DimOverflow {
        rows: usize,
        hidden: usize,
    },
    XLenMismatch {
        expected: usize,
        actual: usize,
    },
    WLenMismatch {
        expected: usize,
        actual: usize,
    },
    DimsExceedI32Max {
        rows: usize,
        hidden: usize,
        numel: usize,
    },
}

impl std::fmt::Display for RowKernelValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidEps { eps } => {
                write!(
                    f,
                    "row kernel eps must be finite and non-negative: eps={eps}"
                )
            }
            Self::DimOverflow { rows, hidden } => write!(
                f,
                "row kernel rows*hidden overflowed usize: rows={rows}, hidden={hidden}"
            ),
            Self::XLenMismatch { expected, actual } => write!(
                f,
                "row kernel x length mismatch: rows*hidden={expected}, x.len()={actual}"
            ),
            Self::WLenMismatch { expected, actual } => write!(
                f,
                "row kernel w length mismatch: hidden={expected}, w.len()={actual}"
            ),
            Self::DimsExceedI32Max {
                rows,
                hidden,
                numel,
            } => write!(
                f,
                "row kernel dims must fit in i32 (kernel argument type): rows={rows}, \
                 hidden={hidden}, numel={numel}"
            ),
        }
    }
}

impl std::error::Error for RowKernelValidationError {}

/// canonical RMSNorm 融合プラン（`x * rsqrt(sum(x^2))`。mean 化・eps・
/// weight を含まない）に厳密一致する `plan` から行長を取り出す。
///
/// `backend-cuda::rmsnorm::match_rmsnorm_plan` と同一のプラン形状照合
/// （6 op 列・leaf 1 個・`axis: None` のみ受理）を Metal バックエンド側で
/// 独立実装する（`fandhe_ai_tensor_core::fusion::graph`／`detect` は `tensor-core`
/// 内部限定の `pub(crate)` でありバックエンドクレートから参照できず、
/// またバックエンドクレート同士〈`backend-cuda`／`backend-metal`〉は
/// 相互依存しない設計〈`delegation-impl.md`〉のため、判定ロジック自体を
/// 共有せず型〈`FusionPlan`〉のみを共有する）。
pub fn match_rmsnorm_plan(plan: &FusionPlan) -> Option<usize> {
    if plan.leaf_count() != 1 {
        return None;
    }
    let ops: Vec<FusedOpKind> = plan.ops().collect();
    if ops.len() != 6 {
        return None;
    }
    let expect = [
        matches!(ops[0], FusedOpKind::Input { leaf_index: 0 }),
        matches!(ops[1], FusedOpKind::Mul { lhs: 0, rhs: 0 }),
        matches!(
            ops[2],
            FusedOpKind::Sum {
                input: 1,
                axis: None
            }
        ),
        matches!(ops[3], FusedOpKind::Rsqrt { input: 2 }),
        matches!(
            ops[4],
            FusedOpKind::Broadcast {
                input: 3,
                axis: None
            }
        ),
        matches!(ops[5], FusedOpKind::Mul { lhs: 4, rhs: 0 }),
    ];
    if expect.iter().any(|ok| !ok) {
        return None;
    }

    let row_fusion: &RowFusionMeta = plan.row_fusion()?;
    if row_fusion.axis().is_some() {
        return None;
    }
    Some(row_fusion.row_len())
}

/// canonical softmax 融合プラン（`exp(x - max(x)) / sum(exp(x - max(x)))`）
/// に厳密一致する `plan` から行長を取り出す（実装計画 §4.5「Step 2」）。
///
/// op 列（8 op・leaf 1 個）: `Input → Max{axis} → Broadcast{axis} →
/// Sub(0,2) → Exp(3) → Sum{axis}(4) → Broadcast{axis}(6) → Div(4,6)`。
/// 4 箇所の `axis` は全て同一値（`None` または「最終次元（contiguous
/// 行）」）でなければならない（`axis: Some(a)` で `a` が
/// `output_shape().len() - 1` と異なる場合は「行方向以外の縮約」であり
/// 対象外。実装計画 §4.5「`axis` が最終次元（contiguous 行）または
/// `None` のみを受理」）。CUDA 側 `backend-cuda` に softmax 融合カーネルは
/// まだ存在しない（#594・G-7 が OPEN）ため、[`match_rmsnorm_plan`] と
/// 異なり移植元は存在せず本実装が初出になる。
pub fn match_softmax_plan(plan: &FusionPlan) -> Option<usize> {
    if plan.leaf_count() != 1 {
        return None;
    }
    let ops: Vec<FusedOpKind> = plan.ops().collect();
    if ops.len() != 8 {
        return None;
    }
    if !matches!(ops[0], FusedOpKind::Input { leaf_index: 0 }) {
        return None;
    }
    let FusedOpKind::Max { input: 0, axis } = ops[1] else {
        return None;
    };
    if !matches!(ops[2], FusedOpKind::Broadcast { input: 1, axis: bc_axis } if bc_axis == axis) {
        return None;
    }
    if !matches!(ops[3], FusedOpKind::Sub { lhs: 0, rhs: 2 }) {
        return None;
    }
    if !matches!(ops[4], FusedOpKind::Exp { input: 3 }) {
        return None;
    }
    if !matches!(ops[5], FusedOpKind::Sum { input: 4, axis: sum_axis } if sum_axis == axis) {
        return None;
    }
    if !matches!(ops[6], FusedOpKind::Broadcast { input: 5, axis: bc_axis2 } if bc_axis2 == axis) {
        return None;
    }
    if !matches!(ops[7], FusedOpKind::Div { lhs: 4, rhs: 6 }) {
        return None;
    }

    if let Some(a) = axis {
        let rank = plan.output_shape().len();
        if rank == 0 || a != rank - 1 {
            return None;
        }
    }

    let row_fusion: &RowFusionMeta = plan.row_fusion()?;
    if row_fusion.axis() != axis {
        return None;
    }
    Some(row_fusion.row_len())
}

/// [`match_rmsnorm_plan`]／[`match_softmax_plan`] 一致後の共通ガード:
/// `plan.dtype() == DType::F32`（`crate::ops::MetalBackendOps::run_fused`
/// から呼ばれる。CUDA 側 codex-review 是正 PR #706 と同じ「起動前に
/// dtype を明示検証する」契約）。
pub fn plan_dtype_is_f32(plan: &FusionPlan) -> bool {
    plan.dtype() == DType::F32
}

/// online softmax の online max 初期値・範囲外レーンの sentinel（x
/// ドメイン。`log2(e)` 適用前の生値）を `shaders/softmax.metal` の
/// `SOFTMAX_NEG_FLT_MAX` と同じ値でテスト側にロックする参照値。
/// 実際の計算は MSL シェーダ側のみが行う（本コメント末尾の注記参照）ため、
/// 「単一の真実源」はシェーダ側の定義であり、本定数はそれをテストで
/// 検証するための Rust 側ミラーである（`#[cfg(test)]` 限定の理由）。
///
/// `f32::MIN`（== `-f32::MAX`。IEEE 754 単精度の最小有限値そのもの）を
/// 直接使う。マージンを設けない理由（PR #714 codex-review 是正。旧版は
/// `-(0.875 * f32::MAX)` のマージン付き sentinel を sum 側の暗黙除外にも
/// 流用していたが、入力が `-f32::MAX` 付近の有限値の場合に sentinel と
/// 実データが数値的に拮抗し範囲外レーンの寄与が sum に混入する欠陥が
/// あった）: sum への寄与は `shaders/softmax.metal` 側で `valid` フラグに
/// より明示的にゲートするため、この sentinel は「どんな有限入力より真に
/// 小さいか等しい」ことだけを保証すればよく、マージンは不要かつ有害
/// （マージンを設けると sentinel 自身が有効な入力レンジを狭める）。
///
/// online max・sum の計算自体は MSL シェーダ側（`shaders/softmax.metal`）
/// のみが行い、Rust 側に対応する CPU 実行経路は存在しないため、本定数の
/// 実際の消費者は本ファイル末尾 `#[cfg(test)] mod tests`（数値特性の
/// ロック）のみである。`#[cfg(test)]` を付けないと、この値を読む唯一の
/// 経路がテストビルドにしか存在しないため、テストを含まない通常の `lib`
/// ターゲットビルド（dead_code 判定はターゲットごと）で dead_code になる。
#[cfg(test)]
pub const SOFTMAX_NEG_FLT_MAX: f32 = f32::MIN;

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::DType;

    // --- select_route ---

    #[test]
    fn select_route_onepass_at_boundary() {
        assert_eq!(select_route(ONEPASS_MAX_HIDDEN), RowKernelRoute::OnePass);
    }

    #[test]
    fn select_route_twopass_over_boundary() {
        assert_eq!(
            select_route(ONEPASS_MAX_HIDDEN + 1),
            RowKernelRoute::TwoPass
        );
    }

    #[test]
    fn select_route_onepass_for_small_hidden() {
        assert_eq!(select_route(8), RowKernelRoute::OnePass);
    }

    #[test]
    fn select_route_onepass_for_zero_hidden() {
        assert_eq!(select_route(0), RowKernelRoute::OnePass);
    }

    // --- derive_persistent_grid ---

    #[test]
    fn derive_persistent_grid_clamps_groups_per_core_at_cap() {
        // max_threadgroup_memory_bytes が極端に大きい → groups_per_core は
        // CAP（16）でクランプ。
        let grid = derive_persistent_grid(4, u32::MAX, 1, 1_000_000);
        assert_eq!(grid, 4 * 16);
    }

    #[test]
    fn derive_persistent_grid_clamps_grid_at_rows() {
        let grid = derive_persistent_grid(u32::MAX, u32::MAX, 1, 10);
        assert_eq!(grid, 10);
    }

    #[test]
    fn derive_persistent_grid_returns_at_least_one() {
        let grid = derive_persistent_grid(0, 0, ONEPASS_SMEM_BYTES_PER_GROUP, 5);
        assert_eq!(grid, 1);
    }

    #[test]
    fn derive_persistent_grid_zero_smem_uses_cap() {
        // 2 パス経路（threadgroup memory 制約なし）は CAP（16）を使う。
        let grid = derive_persistent_grid(4, 0, 0, 1000);
        assert_eq!(grid, 4 * 16);
    }

    #[test]
    fn derive_persistent_grid_rows_zero_is_failsafe_one() {
        assert_eq!(derive_persistent_grid(4, 1024, 4, 0), 1);
    }

    #[test]
    fn derive_persistent_grid_uses_declared_smem_budget_not_actual_hidden() {
        // 固定長 threadgroup 配列（コンパイル時宣言）は使用量に関わらず
        // 予約されるため、`hidden` が小さくても常に
        // `ONEPASS_SMEM_BYTES_PER_GROUP` を基準に groups_per_core を
        // 算出する（モジュール冒頭コメント「CUDA との決定的な設計差」）。
        let max_tg_mem = 32 * 1024u32; // 32KiB（一般的な Apple GPU 下限）
        let grid = derive_persistent_grid(4, max_tg_mem, ONEPASS_SMEM_BYTES_PER_GROUP, 1_000_000);
        // groups_per_core = 32KiB / 16KiB = 2
        assert_eq!(grid, 4 * 2);
    }

    // --- derive_persistent_grid_fallback ---

    #[test]
    fn derive_persistent_grid_fallback_clamps_at_cap() {
        assert_eq!(
            derive_persistent_grid_fallback(1_000_000),
            FALLBACK_GRID_CAP
        );
    }

    #[test]
    fn derive_persistent_grid_fallback_rows_zero_is_one() {
        assert_eq!(derive_persistent_grid_fallback(0), 1);
    }

    #[test]
    fn derive_persistent_grid_fallback_passes_through_small_rows() {
        assert_eq!(derive_persistent_grid_fallback(3), 3);
    }

    // --- validate_row_kernel_launch ---

    #[test]
    fn validate_row_kernel_launch_accepts_matching_dims() {
        assert!(validate_row_kernel_launch(3, 8, 24, Some(8), Some(1e-5)).is_ok());
        assert!(validate_row_kernel_launch(3, 8, 24, None, None).is_ok());
    }

    #[test]
    fn validate_row_kernel_launch_rejects_x_len_mismatch() {
        let err = validate_row_kernel_launch(3, 8, 23, None, None).unwrap_err();
        assert!(matches!(err, RowKernelValidationError::XLenMismatch { .. }));
    }

    #[test]
    fn validate_row_kernel_launch_rejects_w_len_mismatch() {
        let err = validate_row_kernel_launch(3, 8, 24, Some(7), None).unwrap_err();
        assert!(matches!(err, RowKernelValidationError::WLenMismatch { .. }));
    }

    #[test]
    fn validate_row_kernel_launch_rejects_non_finite_eps() {
        let err = validate_row_kernel_launch(3, 8, 24, None, Some(f32::NAN)).unwrap_err();
        assert!(matches!(err, RowKernelValidationError::InvalidEps { .. }));
        let err = validate_row_kernel_launch(3, 8, 24, None, Some(f32::INFINITY)).unwrap_err();
        assert!(matches!(err, RowKernelValidationError::InvalidEps { .. }));
    }

    #[test]
    fn validate_row_kernel_launch_rejects_negative_eps() {
        let err = validate_row_kernel_launch(3, 8, 24, None, Some(-1e-5)).unwrap_err();
        assert!(matches!(err, RowKernelValidationError::InvalidEps { .. }));
    }

    #[test]
    fn validate_row_kernel_launch_accepts_zero_eps() {
        assert!(validate_row_kernel_launch(3, 8, 24, None, Some(0.0)).is_ok());
    }

    #[test]
    fn validate_row_kernel_launch_skips_eps_check_when_none() {
        // softmax 呼び出し（eps 概念自体を持たない）は `eps: None` で
        // 呼ばれ、eps 検証自体がスキップされる。
        assert!(validate_row_kernel_launch(3, 8, 24, None, None).is_ok());
    }

    #[test]
    fn validate_row_kernel_launch_rejects_dims_exceeding_i32_max() {
        let err =
            validate_row_kernel_launch(i32::MAX as usize + 1, 1, i32::MAX as usize + 1, None, None)
                .unwrap_err();
        assert!(matches!(
            err,
            RowKernelValidationError::DimsExceedI32Max { .. }
        ));
    }

    // --- match_rmsnorm_plan ---
    //
    // `fandhe_ai_tensor_core::fusion::graph`／`detect` は `tensor-core` 内部限定の
    // `pub(crate)` で `backend-metal` からは参照できないため、CUDA 側
    // テストと同様 `FusionPlan::from_ops`（`pub` + `#[doc(hidden)]`）で
    // 直接 canonical プランを組み立てる。

    fn build_canonical_rmsnorm_plan(hidden: usize) -> FusionPlan {
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Mul { lhs: 0, rhs: 0 },
            FusedOpKind::Sum {
                input: 1,
                axis: None,
            },
            FusedOpKind::Rsqrt { input: 2 },
            FusedOpKind::Broadcast {
                input: 3,
                axis: None,
            },
            FusedOpKind::Mul { lhs: 4, rhs: 0 },
        ];
        FusionPlan::from_ops(ops, vec![hidden], DType::F32, 1).unwrap()
    }

    #[test]
    fn match_rmsnorm_plan_accepts_canonical_plan() {
        let plan = build_canonical_rmsnorm_plan(8);
        assert_eq!(match_rmsnorm_plan(&plan), Some(8));
    }

    #[test]
    fn match_rmsnorm_plan_rejects_row_axis_variant() {
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Mul { lhs: 0, rhs: 0 },
            FusedOpKind::Sum {
                input: 1,
                axis: Some(1),
            },
            FusedOpKind::Rsqrt { input: 2 },
            FusedOpKind::Broadcast {
                input: 3,
                axis: Some(1),
            },
            FusedOpKind::Mul { lhs: 4, rhs: 0 },
        ];
        let plan = FusionPlan::from_ops(ops, vec![2, 8], DType::F32, 1).unwrap();
        assert_eq!(match_rmsnorm_plan(&plan), None);
    }

    #[test]
    fn match_rmsnorm_plan_rejects_softmax_shaped_plan() {
        assert_eq!(
            match_rmsnorm_plan(&build_canonical_softmax_plan_none_axis(8)),
            None
        );
    }

    // --- match_softmax_plan ---

    fn build_canonical_softmax_plan_none_axis(hidden: usize) -> FusionPlan {
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Max {
                input: 0,
                axis: None,
            },
            FusedOpKind::Broadcast {
                input: 1,
                axis: None,
            },
            FusedOpKind::Sub { lhs: 0, rhs: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Sum {
                input: 4,
                axis: None,
            },
            FusedOpKind::Broadcast {
                input: 5,
                axis: None,
            },
            FusedOpKind::Div { lhs: 4, rhs: 6 },
        ];
        FusionPlan::from_ops(ops, vec![hidden], DType::F32, 1).unwrap()
    }

    fn build_canonical_softmax_plan_last_axis(rows: usize, hidden: usize) -> FusionPlan {
        let axis = Some(1);
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Max { input: 0, axis },
            FusedOpKind::Broadcast { input: 1, axis },
            FusedOpKind::Sub { lhs: 0, rhs: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Sum { input: 4, axis },
            FusedOpKind::Broadcast { input: 5, axis },
            FusedOpKind::Div { lhs: 4, rhs: 6 },
        ];
        FusionPlan::from_ops(ops, vec![rows, hidden], DType::F32, 1).unwrap()
    }

    #[test]
    fn match_softmax_plan_accepts_none_axis_plan() {
        let plan = build_canonical_softmax_plan_none_axis(8);
        assert_eq!(match_softmax_plan(&plan), Some(8));
    }

    #[test]
    fn match_softmax_plan_accepts_last_axis_plan() {
        let plan = build_canonical_softmax_plan_last_axis(2, 8);
        assert_eq!(match_softmax_plan(&plan), Some(8));
    }

    #[test]
    fn match_softmax_plan_rejects_non_last_axis() {
        // rank 3・axis=0（最終次元ではない）は対象外。
        let axis = Some(0);
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Max { input: 0, axis },
            FusedOpKind::Broadcast { input: 1, axis },
            FusedOpKind::Sub { lhs: 0, rhs: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Sum { input: 4, axis },
            FusedOpKind::Broadcast { input: 5, axis },
            FusedOpKind::Div { lhs: 4, rhs: 6 },
        ];
        let plan = FusionPlan::from_ops(ops, vec![4, 2, 8], DType::F32, 1).unwrap();
        assert_eq!(match_softmax_plan(&plan), None);
    }

    #[test]
    fn match_softmax_plan_rejects_rmsnorm_shaped_plan() {
        let plan = build_canonical_rmsnorm_plan(8);
        assert_eq!(match_softmax_plan(&plan), None);
    }

    #[test]
    fn match_softmax_plan_rejects_mismatched_axis_across_ops() {
        // ops[1]（Max）と ops[5]（Sum）で axis が食い違う不正プランは
        // `FusionPlan::from_ops` 自体が構築エラーを返す可能性があるが、
        // 万一構築できても `match_softmax_plan` は None を返す契約を
        // 確認する（fail-closed の多層防御）。
        let ops = vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Max {
                input: 0,
                axis: Some(1),
            },
            FusedOpKind::Broadcast {
                input: 1,
                axis: Some(1),
            },
            FusedOpKind::Sub { lhs: 0, rhs: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Sum {
                input: 4,
                axis: None,
            },
            FusedOpKind::Broadcast {
                input: 5,
                axis: Some(1),
            },
            FusedOpKind::Div { lhs: 4, rhs: 6 },
        ];
        if let Ok(plan) = FusionPlan::from_ops(ops, vec![2, 8], DType::F32, 1) {
            assert_eq!(match_softmax_plan(&plan), None);
        }
    }

    // --- plan_dtype_is_f32 ---

    #[test]
    fn plan_dtype_is_f32_true_for_f32_plan() {
        let plan = build_canonical_rmsnorm_plan(8);
        assert!(plan_dtype_is_f32(&plan));
    }

    // --- SOFTMAX_NEG_FLT_MAX ---
    //
    // online max sentinel の数値検証（PR #714 codex-review 是正）。
    // f16 版は本タスクのスコープ外（`BackendOps` が f32 専用のため）。

    #[test]
    fn softmax_neg_flt_max_is_finite() {
        assert!(SOFTMAX_NEG_FLT_MAX.is_finite());
    }

    #[test]
    fn softmax_neg_flt_max_equals_negative_f32_max() {
        // `f32::MIN == -f32::MAX`（IEEE 754 単精度の最小有限値。負値の
        // 検証も兼ねる）。マージンなしの sentinel であることをロックする。
        assert_eq!(SOFTMAX_NEG_FLT_MAX, -f32::MAX);
    }

    #[test]
    fn softmax_neg_flt_max_survives_fma_without_producing_negative_infinity() {
        // FMA 経由（`fma(1.0, sentinel, 0.0)` 等の恒等演算）でも -inf を
        // 生まないことを確認する。
        let via_fma = 1.0f32.mul_add(SOFTMAX_NEG_FLT_MAX, 0.0);
        assert!(via_fma.is_finite());
        assert_eq!(via_fma, SOFTMAX_NEG_FLT_MAX);
    }

    #[test]
    fn softmax_neg_flt_max_is_less_than_or_equal_to_any_finite_f32() {
        // sentinel はどんな有限な `f32` 入力よりも真に小さいか等しく、
        // オンライン最大値更新 `m_new = max(m, chunk_max)` が初期状態
        // （sentinel）を実データで確実に上書きする（同値の場合も含め
        // 安全）ことを保証する。
        for &v in &[0.0f32, 1.0, -1.0, 1e4, -1e4, f32::MAX, -f32::MAX] {
            assert!(
                SOFTMAX_NEG_FLT_MAX <= v,
                "sentinel={SOFTMAX_NEG_FLT_MAX} は有限値 {v} 以下でなければならない"
            );
        }
    }
}
