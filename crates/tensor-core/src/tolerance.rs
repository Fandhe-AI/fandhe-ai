//! REQ-2 統一複合判定の許容誤差定数と判定関数の単一定義（共通層）。
//!
//! `backend-cpu::parity`（バックエンド間数値一致の検査）と
//! `autodiff::create_graph`（構築時の fail-closed 検査）が同じ定義を参照するための
//! 置き場である。`autodiff` → `backend-cpu` の依存は禁止のため、双方が依存する
//! `tensor-core` に置く。`backend-cpu::parity` は本モジュールの定数を再エクスポートする。

/// REQ-2 統一複合判定の相対誤差閾値（全ペア共通）。
///
/// **閾値の変更はユーザー承認必須**（`.claude/rules/coding-rust.md`
/// 「バックエンド間数値一致テストの許容誤差を単独で緩和しない」・
/// `.claude/rules/security.md` A08「ガードレール閾値・テスト許容誤差の
/// 変更は必ず人間の承認を経る」）。ポリシー除外リストのブラインドスポット
/// 対象であり、自己修復ループ等による自動緩和を許可しない。
pub const RELATIVE_TOLERANCE: f64 = 1e-3;

/// REQ-2 統一複合判定の絶対誤差救済閾値（0 近傍の相対誤差跳ね上がり対策。全ペア共通）。
///
/// 変更にはユーザー承認が必須（[`RELATIVE_TOLERANCE`] と同じ理由）。
pub const ABSOLUTE_RESCUE_THRESHOLD: f64 = 1e-5;

/// 有限な f32 の 2 値が複合判定（相対 [`RELATIVE_TOLERANCE`] 未満 または
/// 絶対 [`ABSOLUTE_RESCUE_THRESHOLD`] 未満）を満たすか。`reference` が 0 のときは相対誤差を
/// 無限大として扱い、絶対誤差側のみで判定する。非有限値の扱いは呼び出し側が決める。
#[must_use]
pub fn composite_close_f32(actual: f32, reference: f32) -> bool {
    let diff = (f64::from(actual) - f64::from(reference)).abs();
    let rel = diff / f64::from(reference).abs();
    rel < RELATIVE_TOLERANCE || diff < ABSOLUTE_RESCUE_THRESHOLD
}
