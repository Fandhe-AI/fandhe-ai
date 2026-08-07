//! 自己修復対象の種別軸（TASK-3.1a・イシュー #132・REQ-3）。
//!
//! `crate::runner::SelfRepairLoop` は種別を問わず同一のオーケストレーションで
//! 動作するが、実際の検出・修正生成ロジックは種別ごとに異なる（バグ修正・
//! 性能回帰・機能追加の検出／生成実装自体は本イシューのスコープ外。
//! `docs/spec/v1-assets-inventory.md` L17 の「改修して再利用」判定に基づき、
//! v1 `tools/self-repair/src/kind.rs` から制御構造のみを移植する）。
//! 本 enum はその判別子であり、[`crate::stages::Detector`] /
//! [`crate::stages::FixGenerator`] の実装が「どの種別を担当するか」を
//! 自己申告・検証するために使う。

/// AI 自己修復ループが扱う変更種別（PoC-2 の 3 題材に対応）。
///
/// - `BugFix`: PoC-2 題材 (a) 相当。既存挙動の欠陥修正。
/// - `PerfRegression`: PoC-2 題材 (b) 相当。ベンチ劣化の検出・改善。
/// - `FeatureAddition`: PoC-2 題材 (c) 相当。要件差分を埋める機能追加。
///
/// variant 追加時は [`crate::stages`] の各 trait 実装（種別別の検出・修正生成。
/// 本イシューのスコープ外）・`runner::SelfRepairLoop` の種別非依存性を保つ
/// テストを要見直しとする。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RepairKind {
    /// バグ修正（既存挙動の欠陥を検出・修正する）。
    BugFix,
    /// 性能回帰（ベンチ計測の劣化を検出・改善する）。
    PerfRegression,
    /// 機能追加（要件・受け入れ基準との差分を埋める）。
    FeatureAddition,
}

impl RepairKind {
    /// ログ・レポート表示用の機械可読識別子。
    ///
    /// 構造化ログ出力（TASK-3.4・イシュー #145）が [`crate::report::LoopReport`]
    /// をシリアライズする際、種別を安定した文字列として埋め込むために使う想定。
    pub fn as_machine_id(self) -> &'static str {
        match self {
            RepairKind::BugFix => "bug_fix",
            RepairKind::PerfRegression => "perf_regression",
            RepairKind::FeatureAddition => "feature_addition",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_id_is_stable_per_variant() {
        assert_eq!(RepairKind::BugFix.as_machine_id(), "bug_fix");
        assert_eq!(
            RepairKind::PerfRegression.as_machine_id(),
            "perf_regression"
        );
        assert_eq!(
            RepairKind::FeatureAddition.as_machine_id(),
            "feature_addition"
        );
    }
}
