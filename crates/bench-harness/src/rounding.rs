//! REQ-8 の性能下限丸め規則（バックエンド非依存の純粋関数）。
//!
//! 実測比率（対 PyTorch、パーセント値）から初期リリース段階の性能下限を導出する
//! 単調な丸め規則を提供する。`docs/spec/04-requirements.md` REQ-8 の規則を実装する:
//!
//! - 実測比率が 10% 以上の場合は 5% 刻みで切り下げる
//! - 実測比率が 10% 未満の場合は 1% 刻みで切り下げる（10% 未満の領域は 5% 刻みが粗すぎるため）
//! - 境界（10% ちょうど）は 10% 以上側（5% 刻み）を適用する
//!
//! v1 は「境界近傍でさらに 1 段安全側へ」という条件付き追加ステップを持ち、これが非単調性
//! （実測比率 16.9% → 15% だが 17.0% → 15% からさらに 1 段下げて 10% となる逆転。旧 #4）を
//! 生んでいた。v2 はこの条件付き追加ステップを廃止し、実測比率の大小のみで刻み幅を切り替える
//! 規則に統一したことで、実測比率に対し非減少（実測値が高いほど下限も高いか同値）であることが
//! 数学的に保証される（`docs/spec/04-requirements.md:171`）。
//!
//! 本モジュールは丸め規則そのもの（TASK-8.2b・本イシュー #153）のみを担う。段階的下限表
//! （CPU 5%・CUDA f32 10% 等）の保持と合否判定は利用側の TASK-8.2a（#152）が行う。
//! 将来のバックエンド・精度追加時も本規則を流用すること（`docs/spec/04-requirements.md:187`、
//! 個別のバックエンド・精度ごとに異なる丸め規則を設けない）。

use std::fmt;

/// 実測比率（パーセント値）の妥当性上限。
///
/// 実測比率は「対 PyTorch の所要時間比」であり、常識的には数百 % 程度に収まる
/// （実装が参照実装よりわずかに遅い場合でも高々数倍）。この上限を大きく超える値は
/// 丸め規則の入力として尤もらしくなく、参照実装側の計測時間がゼロに近い等の
/// 計測異常（レポート破損・ゼロ除算的な比率爆発）を疑うべき fail-closed 対象とする
/// （Review 指摘: 負値側のみ fail-closed で上限側が fail-open だった非対称性の解消）。
/// `docs/spec/04-requirements.md` REQ-8 の丸め例（最大 23.2%）に対し十分な余裕を持たせつつ、
/// `floor_lower_bound(1e12)` のような明らかな異常値を `u32` への飽和キャストで
/// 「もっともらしい」下限として黙って受理しないための閾値である。
const MAX_PLAUSIBLE_PERCENT: f64 = 1_000.0;

/// [`floor_lower_bound`] の入力検証エラー。
///
/// 実測比率は計測レポート（JSON 等の外部入力）由来となりうるため、
/// 本番経路で `unwrap()` / `expect()` を使わない方針（`.claude/rules/coding-rust.md`）に従い、
/// NaN・無限大・負値・異常な大値を fail-closed に拒否する。ガードレール（将来配線）が丸め結果を
/// 性能下限の合否判定に用いる以上、判定不能な入力を黙って通さないことが重要となる。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingError {
    /// NaN・無限大は大小比較・刻み幅判定が定義できないため判定不能（fail-closed）。
    NonFinite,
    /// 実測比率が負（計測異常。所要時間・スループット比が負になることはない）。
    Negative,
    /// 実測比率が妥当性上限（`MAX_PLAUSIBLE_PERCENT`、非公開定数）を超える（計測異常の可能性が高い）。
    TooLarge,
}

impl fmt::Display for RoundingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RoundingError::NonFinite => write!(f, "実測比率が NaN または無限大で判定不能"),
            RoundingError::Negative => write!(f, "実測比率が負であり計測異常の可能性がある"),
            RoundingError::TooLarge => write!(
                f,
                "実測比率が上限（{MAX_PLAUSIBLE_PERCENT}%）を超えており計測異常の可能性がある"
            ),
        }
    }
}

impl std::error::Error for RoundingError {}

/// 実測比率（パーセント値。例: 10.3）に REQ-8 の丸め規則を適用し、
/// 初期リリース段階の性能下限（パーセント整数値）を返す。
///
/// 入力はパーセント値の f64 とする（`docs/spec/04-requirements.md` REQ-8 の表記と一致させ、
/// 比率→% 変換に伴う浮動小数点誤差の混入箇所を呼び出し側の 1 箇所に限定する）。
/// 戻り値は規則上必ず 1% または 5% の整数倍になるため `u32` とし、下流（TASK-8.2a の
/// 下限表比較・回帰テスト）で浮動小数点の等値比較を避けられるようにする。
///
/// # 丸めの安全側判断
///
/// 刻み幅で除してから `floor` を取る操作は常に実測比率以下（安全側）に働くため、
/// 浮動小数点の表現誤差（例: `10.0 / 5.0` が `1.9999999...` 側に丸まる等）があっても
/// 下限を過大評価することはない。このため境界値へのイプシロン補正は行わない。
///
/// # Errors
///
/// - `measured_percent` が NaN または無限大の場合は [`RoundingError::NonFinite`]
/// - `measured_percent` が負の場合は [`RoundingError::Negative`]
/// - `measured_percent` が妥当性上限（`MAX_PLAUSIBLE_PERCENT`、非公開定数）を超える場合は
///   [`RoundingError::TooLarge`]
///   （計測異常により比率が爆発したケースを、`u32` への飽和キャストで「もっともらしい」
///   下限として黙って通さないための fail-closed 拒否。負値側の拒否と対称にする）
pub fn floor_lower_bound(measured_percent: f64) -> Result<u32, RoundingError> {
    if !measured_percent.is_finite() {
        return Err(RoundingError::NonFinite);
    }
    if measured_percent < 0.0 {
        return Err(RoundingError::Negative);
    }
    if measured_percent > MAX_PLAUSIBLE_PERCENT {
        return Err(RoundingError::TooLarge);
    }

    // 境界（10% ちょうど）は `>=` により 5% 刻み側（10% 以上側）を適用する
    // （`docs/spec/04-requirements.md:171`「境界（10%）ちょうどの場合は 10% 以上側を適用」）。
    let step = if measured_percent >= 10.0 { 5.0 } else { 1.0 };

    Ok(((measured_percent / step).floor() * step) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- 境界ケース（受け入れ条件の中核: 「丸め規則の境界ケーステストが green」） ---

    #[test]
    fn boundary_ten_percent_uses_five_percent_step() {
        // 10% ちょうどは 5% 刻み側（10% 以上側）を適用する。
        assert_eq!(floor_lower_bound(10.0), Ok(10));
    }

    #[test]
    fn just_below_ten_percent_uses_one_percent_step() {
        assert_eq!(floor_lower_bound(9.999), Ok(9));
    }

    #[test]
    fn five_percent_step_boundary() {
        assert_eq!(floor_lower_bound(14.999), Ok(10));
        assert_eq!(floor_lower_bound(15.0), Ok(15));
    }

    #[test]
    fn one_percent_step_boundary() {
        assert_eq!(floor_lower_bound(0.999), Ok(0));
        assert_eq!(floor_lower_bound(1.0), Ok(1));
        assert_eq!(floor_lower_bound(0.0), Ok(0));
    }

    // --- spec 実測値の再現（`docs/spec/04-requirements.md:179-182` の丸め例） ---

    #[test]
    fn spec_cpu_measured_ratio() {
        // CPU 対 PyTorch CPU（Apple M4 Max）: 5.3% は 10% 未満のため 1% 刻み切り下げ → 5%。
        assert_eq!(floor_lower_bound(5.3), Ok(5));
    }

    #[test]
    fn spec_cuda_f32_measured_ratio() {
        // CUDA f32 対 PyTorch CUDA（DGX Spark GB10）: 10.3% は 10% 以上のため 5% 刻み切り下げ → 10%。
        assert_eq!(floor_lower_bound(10.3), Ok(10));
    }

    #[test]
    fn spec_metal_f32_measured_ratio() {
        // Metal f32 対 PyTorch MPS（Apple M4 Max）: 23.2% は 10% 以上のため 5% 刻み切り下げ → 20%。
        assert_eq!(floor_lower_bound(23.2), Ok(20));
    }

    #[test]
    fn spec_cuda_f16_measured_ratio_footnote() {
        // CUDA f16 対 PyTorch f16（脚注値。下限は設定しないが丸め規則自体は他値と同様に適用可能）:
        // 1.9% は 10% 未満のため 1% 刻み切り下げ → 1%。
        assert_eq!(floor_lower_bound(1.9), Ok(1));
    }

    // --- 旧 #4 非単調性の解消確認 ---

    #[test]
    fn old_issue_4_nonmonotonicity_is_resolved() {
        // v1 規則では 16.9% → 15% だが 17.0% → 10%（さらに 1 段下げ）という逆転が生じていた。
        // v2 規則ではいずれも 5% 刻み切り下げのみで同値（15%）となり、逆転が起きない。
        assert_eq!(floor_lower_bound(16.9), Ok(15));
        assert_eq!(floor_lower_bound(17.0), Ok(15));
    }

    // --- 単調性（非減少）の網羅確認 ---

    #[test]
    fn floor_lower_bound_is_monotonically_non_decreasing() {
        // REQ-8 が主張する「実測比率が高いほど下限も高いか同値」という数学的性質の回帰テスト。
        // 0.00〜120.00%（100% 超も 5% 刻みで動作することを含む）を 0.01% 刻みで走査する。
        let mut prev = floor_lower_bound(0.0).expect("0.0 は非負・有限なので成功するはず");
        let mut percent = 0.0_f64;
        while percent <= 120.0 {
            let current = floor_lower_bound(percent).expect("非負・有限なので成功するはず");
            assert!(
                current >= prev,
                "非減少性違反: percent={percent} で prev={prev} > current={current}"
            );
            prev = current;
            percent += 0.01;
        }
    }

    // --- エラーケース（fail-closed） ---

    #[test]
    fn nan_is_rejected() {
        assert_eq!(floor_lower_bound(f64::NAN), Err(RoundingError::NonFinite));
    }

    #[test]
    fn infinity_is_rejected() {
        assert_eq!(
            floor_lower_bound(f64::INFINITY),
            Err(RoundingError::NonFinite)
        );
    }

    #[test]
    fn negative_is_rejected() {
        assert_eq!(floor_lower_bound(-0.1), Err(RoundingError::Negative));
    }

    #[test]
    fn too_large_is_rejected() {
        // Review 指摘: 参照実装側の計測時間がゼロに近い等の計測異常で比率が爆発した場合、
        // 上限チェックなしでは `(x/step).floor()*step` が `u32` へ飽和キャストされ
        // 「もっともらしい」下限を黙って返してしまっていた（floor_lower_bound(1e12) が
        // RoundingError にならず 4294967295 を返す非対称性）。上限超過を fail-closed で拒否する。
        assert_eq!(floor_lower_bound(1e12), Err(RoundingError::TooLarge));
        assert_eq!(
            floor_lower_bound(MAX_PLAUSIBLE_PERCENT + 0.1),
            Err(RoundingError::TooLarge)
        );
    }

    #[test]
    fn max_plausible_percent_boundary_is_accepted() {
        // 上限ちょうどは許容側（拒否しない）。
        assert_eq!(floor_lower_bound(MAX_PLAUSIBLE_PERCENT), Ok(1_000));
    }
}
