//! CUDA f32 GEMM の形状別カーネル選択ヒューリスティック（イシュー #1035）。
//!
//! # このモジュールの役割
//!
//! 親ツリー #1029（GEMM カーネルの candle 超え）・#1007 の Phase 2。
//! 現行の f32 本番経路（`gemm_auto.rs::CudaGemmAuto::run_f32` →
//! `tensor_core::dispatch::select_gemm_kernel` → `CudaGemm::run_tiled_f32`
//! 固定）は `kernels::TILED_F32`（32x32 単段 smem タイル）1 種のみで、
//! grid が SM を埋めきれない小サイズ（N=256〜512 級。candle 実測
//! 423〜1,109 GFLOP/s @GB10。`docs/perf/cuda-gemm-kernel-vs-frameworks-
//! baseline.md` §3.2）で SM が遊ぶ問題に対処できない。
//!
//! 本モジュールは `docs/perf/cuda-gemm-cost-model-selection.md`（#527・
//! C-9b）の静的コストモデルの考え方（実測なしで決定的に候補を切り替える
//! 純関数設計・判定不能時は安全側〈= 現行実装〉へフォールバック）を踏襲し、
//! M/N/K 歪度・SM 数・アラインメントを入力に
//! [`GemmVariantKind::Simple`]（現行 `TILED_F32`）／
//! [`GemmVariantKind::DoubleBuffer`]（smem 2 面プリフェッチ）のいずれかを
//! 選ぶ [`select_f32_gemm_variant`] を提供する。
//!
//! # SplitK 撤退の判断（イシュー #1100）
//!
//! [`GemmVariantKind::SplitK`]（K 方向分割 + 決定的縮約。
//! `rmsnorm.rs::derive_dw_split` と同型の 2 段リダクション）は #1035 で
//! ヒューリスティック候補として追加したが、GB10 実機実測（#1031。
//! `docs/perf/cuda-gemm-f32-variant-selection.md` §1a）で REQ-2 複合判定
//! FAIL（m=n=128,k=8192,`num_splits=8`: fail_count=8/16384）を検出した。
//! イシュー #1100 の計画セッションでホスト側 Rust シミュレーション
//! （`tests/splitk_reorder_error_host_model.rs`）により、この不一致は
//! カーネルのバグではなく **K 方向を複数の部分和へ分割してから縮約する
//! という split-K の演算順序そのものが、CPU 参照実装（`backend-cpu::
//! matmul_reference_fma`。K 方向を分割しない単一の逐次 `mul_add` 連鎖）
//! と異なる丸め結果を生む**ことを確認した。精度を f32→f64 へ引き上げても
//! fail 数は減らず（差分の支配項は CPU 参照実装自身の丸め誤差であり、
//! 真値ゼロ近傍〈桁落ち〉要素では絶対誤差救済閾値を超える）、tolerance・
//! 参照実装の変更はユーザー承認必須（`.claude/rules/security.md` A08）の
//! ため本イシューでは行えない。よって **[`select_f32_gemm_variant`] は
//! SplitK を選択候補から撤退させ、K 支配的形状も
//! [`GemmVariantKind::Simple`]（`blocks < num_sms` のため
//! [`GemmVariantKind::DoubleBuffer`] の前提〈`blocks >= num_sms`〉も
//! 満たさない）へ倒す**（安全側＝現行本番既定と同一の数値契約）。
//!
//! [`GemmVariantKind::SplitK`] 型・[`derive_split_count`]・
//! [`validate_split_k_launch`]・カーネルソース
//! （[`crate::kernels_gemm_variants::SPLITK_PARTIAL_F32`]／
//! [`crate::kernels_gemm_variants::SPLITK_REDUCE_F32`]）自体は削除せず、
//! `internal-diagnostics` feature 限定の診断専用エントリ
//! （`gemm_variant_selection.rs::CudaGemmF32VariantSelection::
//! run_split_k_forced`）として保持する（split 順序を反映した parity
//! 契約の是非は spec 側の議論候補として引き継ぐ。`out-of-scope-tracking.md`
//! 準拠）。
//!
//! # 呼び出し元・呼び出し先の文脈
//!
//! - GPU 資源を必要としない純関数のみで構成する（受け入れ条件 (b)
//!   「選択ロジックのユニットテスト」を通常 CI（`#[ignore]` 不要）で
//!   充足するため）。
//! - 実際のカーネル起動・NVRTC コンパイルは呼び出し側
//!   （`gemm_variant_selection.rs::CudaGemmF32VariantSelection::new`。
//!   `internal-diagnostics` feature 限定の opt-in 経路）が担う。
//!   本モジュールはどのカーネルを使うべきかの「判定」のみを返し、
//!   実行はしない（A03 インジェクション対策と同じ関心の分離: 判定に
//!   使う入力は形状の数値のみで、外部文字列を一切埋め込まない）。
//! - **本番既定経路（`CudaGemm::new`・`run_tiled_f32`・
//!   `CudaGemmAuto::run_f32`）へは結線しない**（#1035 実装計画 §3・§8。
//!   実機実測〈全 N で candle 以上の判定・暫定閾値の補正〉とユーザー承認を
//!   経てから後続 Issue で判断する）。
//!
//! # 暫定閾値についての注記
//!
//! [`SPLITK_MIN_K`] 等の定数は実機実測前の**暫定値**である。GB10 実機での
//! A/B 計測後に補正することを想定しているが、`cost_model`（#527）と同じ
//! 方針で**補正は 1 回限りとし、実測を追わない補正ループは行わない**
//! （`docs/perf/cuda-gemm-f32-variant-selection.md` 参照）。

use crate::error::CudaError;

/// GEMM カーネル変種。[`select_f32_gemm_variant`] が選ぶのは `Simple`・
/// `DoubleBuffer` の 2 種のみ（`SplitK` はイシュー #1100 で選択候補から
/// 撤退した。本モジュール冒頭「SplitK 撤退の判断」参照）。
///
/// `Simple` は現行本番既定の `kernels::TILED_F32`（32x32 単段 smem タイル）
/// に対応する。`DoubleBuffer`・`SplitK` は `kernels_gemm_variants.rs` の
/// opt-in カーネルに対応する（`internal-diagnostics` feature 限定。本
/// モジュール自体は feature 非依存の純粋な判定ロジックであることに注意
/// ——判定結果を実際に起動できるかどうかは呼び出し側の feature ゲートと
/// コンパイル成否に依存する）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmVariantKind {
    /// 現行本番既定（`TILED_F32`）。判定不能・境界条件のフォールバック先。
    Simple,
    /// smem 2 面プリフェッチ（cp.async 不使用。#1033 の cp.async 多段
    /// パイプラインとはスコープを分離し、後から `Option` スロット注入で
    /// 差し替え可能な設計としている。実装計画 §3 の 3 番）。
    DoubleBuffer,
    /// K 方向 `num_splits` 分割 + 決定的縮約（atomics 不使用。
    /// `rmsnorm.rs` の dw split-K〈#597〉と同型）。**[`select_f32_gemm_
    /// variant`] は本バリアントを返さない**（イシュー #1100 で選択候補
    /// から撤退。カーネル・型自体は `run_split_k_forced`〈診断専用〉が
    /// 引き続き参照するため保持する）。
    SplitK {
        /// K 方向の分割数。常に 2 以上 [`SPLITK_MAX_SPLITS`] 以下の 2 冪。
        num_splits: u32,
    },
}

/// grid が SM 数を埋めきれないと見なす閾値の判定に使う、SplitK 選択の
/// 前提となる最小 K（K 歪度の基準）。**イシュー #1100 で
/// [`select_f32_gemm_variant`] の選択条件からは撤退済み**（本モジュール
/// 冒頭「SplitK 撤退の判断」参照）。`derive_split_count`・
/// `validate_split_k_launch`・`run_split_k_forced`（診断専用）の入力
/// レンジ検証定数として保持する。
pub const SPLITK_MIN_K: u32 = 1024;

/// split-K の 1 分割あたりの最低 K（1 分割が最低 1 タイル分の仕事量を
/// 持つことを保証する下限。`gemm.rs::TILE` と同じ 32 を踏襲）。
/// [`SPLITK_MIN_K`] と同じく診断専用経路でのみ参照される。
pub const SPLITK_MIN_K_PER_SPLIT: u32 = 32;

/// split-K の `num_splits` 上限（部分和バッファサイズの増大を抑える
/// 安全弁。`rmsnorm.rs::RMSNORM_DW_MAX_SPLIT`〈64〉と同系統だが、GEMM の
/// 部分和は `m * n` 要素とより大きいため保守的に 32 とする）。
/// [`SPLITK_MIN_K`] と同じく診断専用経路でのみ参照される。
pub const SPLITK_MAX_SPLITS: u32 = 32;

/// split-K の部分和バッファ（`num_splits * m * n * 4` bytes）が超えては
/// ならない上限（`rmsnorm.rs::RMSNORM_DW_SPLIT_PARTIAL_MAX_BYTES` と同型の
/// 安全弁）。[`SPLITK_MIN_K`] と同じく診断専用経路（`validate_split_k_
/// launch`・`run_split_k_forced`）でのみ参照される。
pub const SPLITK_PARTIAL_MAX_BYTES: u64 = 256 * 1024 * 1024;

/// DoubleBuffer が成立する最小 K（2 段プリフェッチには最低 2 タイル分の
/// K が要る。`TILE=32` を踏襲し `2 * 32 = 64`）。
pub const DOUBLE_BUFFER_MIN_K: u32 = 2 * 32;

/// GEMM タイル一辺（`gemm.rs::TILE`・`kernels::TILE` と同じ値を本モジュール
/// 内の grid 占有度計算に使う。値そのものの単一の真実源は `kernels::TILE`
/// であり、本定数は判定ロジック専用の複製であることを明記する
/// （`code-comment-style.md`: 陳腐化しやすい実装詳細の重複を避けるため、
/// 値変更時は両方を同時に見直す必要がある点をコメントで残す）。
const VARIANT_TILE: u32 = 32;

/// `ceil(m / VARIANT_TILE) * ceil(n / VARIANT_TILE)`（grid が生成する
/// スレッドブロック総数）を `u64` で計算する。`m`／`n` は `i32::MAX` 以下
/// であることが呼び出し前提（`gemm.rs::validate_gemm_dims` が別途保証）
/// だが、本関数はオーバーフロー安全のため `u64` 昇算を用いる。
fn num_blocks(m: u32, n: u32) -> u64 {
    let tiles_m = u64::from(m).div_ceil(u64::from(VARIANT_TILE));
    let tiles_n = u64::from(n).div_ceil(u64::from(VARIANT_TILE));
    tiles_m * tiles_n
}

/// `num_blocks * num_splits >= 2 * num_sms` を満たす最小の 2 冪
/// `num_splits` を `[2, SPLITK_MAX_SPLITS]` の範囲で導出する。
/// `num_blocks == 0`（`m`／`n` のいずれかが 0）の場合は分割しても
/// 埋める仕事がないため `1`（= 事実上 SplitK を選ばない）を返す
/// （呼び出し元 [`select_f32_gemm_variant`] は `num_splits == 1` を
/// SplitK 非該当として扱う）。
fn derive_split_count(num_blocks: u64, num_sms: u32, k: u32) -> u32 {
    if num_blocks == 0 || num_sms == 0 {
        return 1;
    }
    let target = 2 * u64::from(num_sms);
    let mut splits: u32 = 2;
    while splits < SPLITK_MAX_SPLITS && num_blocks * u64::from(splits) < target {
        splits = splits.saturating_mul(2);
    }
    let splits = splits.min(SPLITK_MAX_SPLITS);
    // 1 分割あたり最低 SPLITK_MIN_K_PER_SPLIT を確保できる上限まで
    // クランプする（分割しすぎて 1 タイル未満の仕事量になるのを防ぐ）。
    let max_by_k = (k / SPLITK_MIN_K_PER_SPLIT).max(1);
    splits.min(max_by_k).max(1)
}

/// SplitK（診断専用。イシュー #1100）を明示的に起動する前に、`m`/`n`/`k`・
/// `num_sms` から [`derive_split_count`] が導く「妥当な分割数」を計算する
/// クレート内公開ラッパー。`gemm_variant_selection.rs::
/// CudaGemmF32VariantSelection::recommend_split_count`（`internal-
/// diagnostics` feature 限定）が呼ぶ唯一の呼び出し元であり、選択候補から
/// 撤退した SplitK を `run_split_k_forced` で診断的に再現する際に
/// `derive_split_count`／[`SPLITK_MIN_K`]／`num_blocks` を到達可能に
/// 保つ（`select_f32_gemm_variant` 自体はこれらを呼ばなくなったため）。
///
/// # 戻り値の契約（イシュー #1100 codex-review 指摘で `Option<u32>` 化）
///
/// `derive_split_count` は分割の意義が薄い場合（`k < SPLITK_MIN_K`・
/// `num_blocks(m, n) == 0`・`num_sms == 0`・K が小さく
/// `SPLITK_MIN_K_PER_SPLIT` 制約で 1 分割にクランプされる場合を含む）に
/// `1`（= 事実上分割なし）を返しうるが、[`validate_split_k_launch`] は
/// `num_splits == 1` を常に拒否する（`[2, SPLITK_MAX_SPLITS]` のみ許容）。
/// 内部値が `1` のまま `Some(1)` を返すと、呼び出し元が戻り値をそのまま
/// `run_split_k_forced` へ渡す正規呼び出しが必ず `InvalidShape` になる
/// 契約不整合が生じるため、本関数は分割不能を `None` で明示し、`Some` の
/// 場合は必ず `[2, SPLITK_MAX_SPLITS]`（`validate_split_k_launch` が
/// 受理可能な範囲）の値であることを保証する。
pub(crate) fn recommend_split_count(m: u32, n: u32, k: u32, num_sms: u32) -> Option<u32> {
    if k < SPLITK_MIN_K {
        return None;
    }
    match derive_split_count(num_blocks(m, n), num_sms, k) {
        1 => None,
        splits => Some(splits),
    }
}

/// SplitK の部分和バッファサイズ（bytes）を `u64` の checked 演算で
/// 計算する。オーバーフロー時は `None` を返す（`.claude/rules/security.md`
/// A03: 確保前に検証する fail-closed 設計。`rmsnorm.rs` の cap 検査と
/// 同型）。
fn split_k_partial_bytes(num_splits: u32, m: u32, n: u32) -> Option<u64> {
    u64::from(num_splits)
        .checked_mul(u64::from(m))
        .and_then(|v| v.checked_mul(u64::from(n)))
        .and_then(|v| v.checked_mul(4))
}

/// M/N/K・SM 数・候補可用性から使うべき GEMM 変種を判定する（純関数・
/// 決定的・オーバーフロー安全）。
///
/// # SplitK を選択しない理由（イシュー #1100）
///
/// 本関数はかつて（#1035）K 支配的形状で [`GemmVariantKind::SplitK`] を
/// 選ぶ分岐を持っていたが、GB10 実機実測（#1031）で REQ-2 複合判定 FAIL
/// を検出し、イシュー #1100 の計画セッションでその原因が K 方向分割
/// 縮約による累積順序の再結合誤差（CPU 参照実装との丸め方式の非両立。
/// tolerance・参照実装の変更はユーザー承認必須のため是正不能）と判明した
/// ため、SplitK 分岐を撤退した（本モジュール冒頭「SplitK 撤退の判断」
/// 参照）。K 支配的形状（`blocks < num_sms`）は下記 DoubleBuffer の前提
/// （`blocks >= num_sms`）も満たさないため、自然に Simple へ落ちる。
///
/// # 引数
///
/// - `m`／`n`／`k`: GEMM の形状（`gemm.rs::validate_gemm_dims` 通過後の
///   正の値を想定するが、0 を含む任意の `u32` に対し panic しない）。
/// - `num_sms`: `device.multiprocessor_count()` の実測結果。`None` は
///   取得失敗を意味し、判定不能として常に [`GemmVariantKind::Simple`]
///   を返す（fail-safe。`swizzle::should_apply_swizzle` と同じ「SM 数が
///   取れない環境では高度な分岐をしない」方針）。
/// - `double_buffer_available`: 呼び出し側で `TILED_DB_F32` のコンパイル・
///   ロードが成功しているか（`gemm_variant_selection.rs` の
///   `Option<CudaFunction>` フィールドが `Some` か）。`false` の場合は
///   DoubleBuffer を選ばず Simple へフォールバックする（未コンパイルの
///   カーネルを選んでしまう事故を防ぐ）。
///
/// # 判定順序（実装計画 §3 のヒューリスティック仕様。#1100 で SplitK 分岐を撤退）
///
/// 1. **DoubleBuffer**: `num_blocks >= num_sms`（grid が SM を十分埋める）
///    かつ `k >= DOUBLE_BUFFER_MIN_K` かつ M/N/K が `VARIANT_TILE` の倍数
///    （アラインメント。整列時のみ利得を見込み、非整列は Simple のまま
///    ——正しさ自体はカーネル側の手動境界チェックで担保するため、ここでの
///    アラインメント判定は純粋に利得予測であり安全性の条件ではない）。
/// 2. それ以外・入力不能（`num_sms.is_none()`）: **Simple**。
pub fn select_f32_gemm_variant(
    m: u32,
    n: u32,
    k: u32,
    num_sms: Option<u32>,
    double_buffer_available: bool,
) -> GemmVariantKind {
    let Some(num_sms) = num_sms else {
        return GemmVariantKind::Simple;
    };
    if num_sms == 0 || m == 0 || n == 0 || k == 0 {
        return GemmVariantKind::Simple;
    }

    let blocks = num_blocks(m, n);

    let aligned = m.is_multiple_of(VARIANT_TILE)
        && n.is_multiple_of(VARIANT_TILE)
        && k.is_multiple_of(VARIANT_TILE);
    if double_buffer_available
        && blocks >= u64::from(num_sms)
        && k >= DOUBLE_BUFFER_MIN_K
        && aligned
    {
        return GemmVariantKind::DoubleBuffer;
    }

    GemmVariantKind::Simple
}

/// [`select_f32_gemm_variant`] の `SplitK` 分割数を、部分和バッファの
/// cap・整合性を検査したうえで呼び出し側（`gemm.rs`）が起動前に再検証
/// するための公開ヘルパ。`select_f32_gemm_variant` 自体が既に cap を
/// 検査済みだが、呼び出し側が [`GemmVariantKind::SplitK`] を保持したまま
/// 別経路（診断 CLI 等）から再利用する場合に備え、独立して呼べる
/// fail-closed 検査を提供する（`rmsnorm.rs::validate_dw_split_launch` と
/// 同じ「分岐に関わらず必ず検証する」設計）。
pub fn validate_split_k_launch(m: u32, n: u32, num_splits: u32) -> Result<(), CudaError> {
    if !(2..=SPLITK_MAX_SPLITS).contains(&num_splits) {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "gemm split-k num_splits must be within [2, {SPLITK_MAX_SPLITS}]: \
                 num_splits={num_splits}"
            ),
        });
    }
    let bytes = split_k_partial_bytes(num_splits, m, n).ok_or_else(|| CudaError::InvalidShape {
        detail: format!(
            "gemm split-k partial buffer size overflowed u64: num_splits={num_splits}, m={m}, n={n}"
        ),
    })?;
    if bytes > SPLITK_PARTIAL_MAX_BYTES {
        return Err(CudaError::InvalidShape {
            detail: format!(
                "gemm split-k partial buffer exceeds cap: bytes={bytes}, cap={SPLITK_PARTIAL_MAX_BYTES}"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- num_sms 判定不能・境界条件は常に Simple ---

    #[test]
    fn returns_simple_when_num_sms_unavailable() {
        assert_eq!(
            select_f32_gemm_variant(4096, 4096, 4096, None, true),
            GemmVariantKind::Simple
        );
    }

    #[test]
    fn returns_simple_for_zero_dims() {
        assert_eq!(
            select_f32_gemm_variant(0, 4096, 4096, Some(64), true),
            GemmVariantKind::Simple
        );
        assert_eq!(
            select_f32_gemm_variant(128, 0, 4096, Some(64), true),
            GemmVariantKind::Simple
        );
        assert_eq!(
            select_f32_gemm_variant(128, 128, 0, Some(64), true),
            GemmVariantKind::Simple
        );
    }

    #[test]
    fn returns_simple_when_num_sms_is_zero() {
        assert_eq!(
            select_f32_gemm_variant(128, 128, 4096, Some(0), true),
            GemmVariantKind::Simple
        );
    }

    // --- SplitK（イシュー #1100 で選択候補から撤退。K 支配的形状は
    // Simple へ倒れることを検証する） ---

    #[test]
    fn k_dominant_small_grid_shape_selects_simple_not_split_k() {
        // m=n=128 → num_blocks = 4*4 = 16。num_sms=64（GB10 級）なら
        // grid が全く埋まらない。k=8192 は m/n より十分大きい K 支配形状
        // だが、#1100 の撤退判断により SplitK は返らず、DoubleBuffer の
        // 前提（blocks >= num_sms）も満たさないため Simple になる
        // （GB10 実機実測で検出した複合判定 FAIL の根本原因は
        // `tests/splitk_reorder_error_host_model.rs` 参照）。
        let variant = select_f32_gemm_variant(128, 128, 8192, Some(64), true);
        assert_eq!(variant, GemmVariantKind::Simple);
    }

    #[test]
    fn does_not_select_split_k_when_k_below_threshold() {
        let variant = select_f32_gemm_variant(128, 128, 512, Some(64), true);
        // k=512 は SPLITK_MIN_K(1024) 未満だった旧条件でも非該当だった
        // 形状だが、撤退後は SplitK 自体がそもそも選ばれない。
        assert!(!matches!(variant, GemmVariantKind::SplitK { .. }));
    }

    #[test]
    fn does_not_select_split_k_when_k_not_dominant() {
        // grid は埋まらない（num_blocks=16 < num_sms=64）が k < max(m, n)
        // のため旧条件でも K 支配形状ではなかった。撤退後は SplitK が
        // そもそも選ばれない。
        let variant = select_f32_gemm_variant(128, 128, 100, Some(64), true);
        assert!(!matches!(variant, GemmVariantKind::SplitK { .. }));
    }

    #[test]
    fn derive_split_count_is_power_of_two_and_within_bounds() {
        for num_sms in [1u32, 4, 16, 64, 132] {
            for blocks in [1u64, 2, 8, 16] {
                let splits = derive_split_count(blocks, num_sms, 65536);
                assert!(splits >= 1);
                assert!(splits <= SPLITK_MAX_SPLITS);
                assert!(splits == 1 || splits.is_power_of_two());
            }
        }
    }

    #[test]
    fn derive_split_count_returns_one_for_zero_blocks_or_sms() {
        assert_eq!(derive_split_count(0, 64, 65536), 1);
        assert_eq!(derive_split_count(16, 0, 65536), 1);
    }

    #[test]
    fn derive_split_count_respects_min_k_per_split() {
        // k=64 では 1 分割あたり最低 32 の制約から最大 2 分割まで。
        let splits = derive_split_count(1, 132, 64);
        assert!(splits <= 2);
    }

    // --- recommend_split_count（イシュー #1100 codex-review 指摘: 戻り値
    // Option<u32> 化。`Some` は必ず validate_split_k_launch が受理する
    // [2, SPLITK_MAX_SPLITS] の範囲内であることを保証する契約） ---

    #[test]
    fn recommend_split_count_returns_none_when_k_below_min() {
        assert_eq!(recommend_split_count(128, 128, SPLITK_MIN_K - 1, 64), None);
    }

    #[test]
    fn recommend_split_count_returns_none_for_zero_dims_or_sms() {
        assert_eq!(recommend_split_count(0, 128, 65536, 64), None);
        assert_eq!(recommend_split_count(128, 128, 65536, 0), None);
    }

    #[test]
    fn recommend_split_count_none_matches_derive_split_count_one_exhaustively() {
        // recommend_split_count が Some を返す全ケースで derive_split_count
        // が 1（=分割なし）を返さないこと、逆に derive_split_count が 1 を
        // 返すケース（num_blocks==0 or num_sms==0。k>=SPLITK_MIN_K の下で
        // max_by_k は SPLITK_MIN_K/SPLITK_MIN_K_PER_SPLIT=32 以上のため
        // 1 にクランプされ得ない）では常に None を返すことを、
        // derive_split_count と recommend_split_count の内部呼び出しの
        // 一貫性として突き合わせる（Some の値が validate_split_k_launch
        // 非受理の 1 に戻ってしまう再発を防ぐ）。
        for (m, n, num_sms) in [(0u32, 128u32, 64u32), (128, 0, 64), (128, 128, 0)] {
            let k = SPLITK_MIN_K;
            assert_eq!(derive_split_count(num_blocks(m, n), num_sms, k), 1);
            assert_eq!(recommend_split_count(m, n, k, num_sms), None);
        }
    }

    #[test]
    fn recommend_split_count_some_is_always_validate_split_k_launch_acceptable() {
        // Some(num_splits) を返す場合は run_split_k_forced へそのまま
        // 渡しても validate_split_k_launch が受理する範囲であることを
        // 保証する（codex-review P2 指摘の再発防止）。
        let m = 4096;
        let n = 4096;
        let k = 65536;
        let num_sms = 64;
        let recommended =
            recommend_split_count(m, n, k, num_sms).expect("この形状では分割が推奨されるはず");
        assert!((2..=SPLITK_MAX_SPLITS).contains(&recommended));
        assert!(validate_split_k_launch(m, n, recommended).is_ok());
    }

    // --- DoubleBuffer ---

    #[test]
    fn selects_double_buffer_for_large_aligned_grid_filling_shape() {
        // m=n=4096（アラインメント OK）・num_blocks = 128*128 = 16384 は
        // num_sms=64 を大きく超える。k=4096 は K 支配ではなく m/n 以下。
        let variant = select_f32_gemm_variant(4096, 4096, 4096, Some(64), true);
        assert_eq!(variant, GemmVariantKind::DoubleBuffer);
    }

    #[test]
    fn does_not_select_double_buffer_when_unaligned() {
        // 1000 は 32 の倍数ではない（1000 % 32 == 8）。
        let variant = select_f32_gemm_variant(1000, 1000, 4096, Some(64), true);
        assert_eq!(variant, GemmVariantKind::Simple);
    }

    #[test]
    fn does_not_select_double_buffer_when_unavailable() {
        let variant = select_f32_gemm_variant(4096, 4096, 4096, Some(64), false);
        assert_eq!(variant, GemmVariantKind::Simple);
    }

    #[test]
    fn does_not_select_double_buffer_when_k_too_small() {
        // k=32 < DOUBLE_BUFFER_MIN_K(64) のため 2 段プリフェッチが成立しない。
        let variant = select_f32_gemm_variant(4096, 4096, 32, Some(64), true);
        assert_eq!(variant, GemmVariantKind::Simple);
    }

    // --- 決定性 ---

    #[test]
    fn selection_is_deterministic_across_repeated_calls() {
        let a = select_f32_gemm_variant(2048, 2048, 2048, Some(48), true);
        let b = select_f32_gemm_variant(2048, 2048, 2048, Some(48), true);
        assert_eq!(a, b);
    }

    // --- 巨大形状での非 panic（u64 昇算のオーバーフロー安全性） ---

    #[test]
    fn does_not_panic_for_near_i32_max_dims() {
        let m = i32::MAX as u32 - 1;
        let n = 128u32;
        let k = 65536u32;
        // panic しないことのみを検証する（結果は Simple 想定だが本質は
        // オーバーフロー安全性）。
        let _ = select_f32_gemm_variant(m, n, k, Some(64), true);
        let _ = select_f32_gemm_variant(n, m, k, Some(64), true);
    }

    // --- validate_split_k_launch ---

    #[test]
    fn validate_split_k_launch_rejects_out_of_range_num_splits() {
        assert!(validate_split_k_launch(128, 128, 1).is_err());
        assert!(validate_split_k_launch(128, 128, SPLITK_MAX_SPLITS + 1).is_err());
    }

    #[test]
    fn validate_split_k_launch_rejects_cap_exceeding_buffer() {
        assert!(validate_split_k_launch(8192, 8192, 8).is_err());
    }

    #[test]
    fn validate_split_k_launch_accepts_in_range_config() {
        assert!(validate_split_k_launch(128, 128, 4).is_ok());
    }
}
