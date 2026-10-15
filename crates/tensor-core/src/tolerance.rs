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
/// 絶対 [`ABSOLUTE_RESCUE_THRESHOLD`] 未満）を満たすか。
///
/// 相対誤差の分母は `backend-cpu::parity::compare_pairs` と同一の
/// `max(|actual|, |reference|, 1e-12)`（定数は不変。判定の緩和ではなく既存判定との一致）。
/// 非有限値は diff/rel が NaN となり `<` が false のため不合格（compare_pairs と同じ）。
#[must_use]
pub fn composite_close_f32(actual: f32, reference: f32) -> bool {
    let diff = (f64::from(actual) - f64::from(reference)).abs();
    let scale = f64::from(actual)
        .abs()
        .max(f64::from(reference).abs())
        .max(1e-12);
    let rel = diff / scale;
    rel < RELATIVE_TOLERANCE || diff < ABSOLUTE_RESCUE_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denominator_uses_max_of_actual_and_reference() {
        // 分母が reference のみだと rel=1.0005e-3 で不合格になる入力（既存判定では合格）。
        assert!(composite_close_f32(1.001_000_5, 1.0));
        assert!(composite_close_f32(1.0, 1.001_000_5));
    }

    #[test]
    fn boundary_and_non_finite_cases() {
        assert!(!composite_close_f32(1.01, 1.0));
        assert!(composite_close_f32(0.0, 0.0));
        assert!(composite_close_f32(5e-6, 0.0));
        assert!(!composite_close_f32(2e-5, 0.0));
        assert!(!composite_close_f32(f32::NAN, 1.0));
        assert!(!composite_close_f32(f32::INFINITY, 1.0));
        assert!(!composite_close_f32(f32::INFINITY, f32::INFINITY));
    }

    #[test]
    fn matches_reference_formula_of_compare_pairs() {
        let vals = [
            0.0f32,
            1e-12,
            5e-6,
            2e-5,
            0.5,
            1.0,
            1.0005,
            1.001_000_5,
            1.002,
            -1.0,
            -1.001_000_5,
            1e20,
            1.0011e20,
        ];
        for &a in &vals {
            for &b in &vals {
                let (x, y) = (f64::from(a), f64::from(b));
                let diff = (x - y).abs();
                let rel = diff / x.abs().max(y.abs()).max(1e-12);
                let expect = rel < RELATIVE_TOLERANCE || diff < ABSOLUTE_RESCUE_THRESHOLD;
                assert_eq!(composite_close_f32(a, b), expect, "{a} vs {b}");
            }
        }
    }
}
