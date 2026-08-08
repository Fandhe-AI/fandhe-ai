//! REQ-8 段階的下限表（バックエンド×dtype×段階）のデータ化と、
//! 計測結果（[`crate::BenchReport`]）に対する自動合否判定（TASK-8.2a・イシュー #152）。
//!
//! `docs/spec/05-tasks.md` TASK-8.2 は下限表を `bench-harness` の計測結果に対する
//! 自動合否判定として実装するタスクであり、本モジュールはそのうち「下限表のデータ化＋
//! 合否判定ロジック」（TASK-8.2a）を担当する。丸め規則（実測比率 10% 以上は 5% 刻み、
//! 10% 未満は 1% 刻み）の共通ロジック実装は別イシュー（TASK-8.2b）の担当であり、
//! 本モジュールは spec 側で丸め規則適用済みの確定定数（[`floor_spec`]）を保持するのみで
//! 丸め処理そのものには依存しない（`.claude/rules/delegation-impl.md`: 同一ファイル
//! 並行編集の回避のため、本イシューは判定専用の本ファイルに閉じる）。
//!
//! [`crate::report`]（TASK-8.1c）が定義する [`crate::BenchReport`] は「値の意味づけ
//! （TFLOPS 換算・合否判定）は呼び出し側の責務」と明記しており（`report.rs` の
//! `BenchReport` ドキュメントコメント）、本モジュールがその呼び出し側にあたる。
//!
//! ## 下限表の出典
//!
//! `docs/spec/04-requirements.md` REQ-8「バックエンド別・対 PyTorch 性能下限
//! （2026-08-05 v2 段階的再設計）」の下表（2026-08-05 版）をそのまま転記する。
//! 各行のコメントに PoC 出典を併記する（`.claude/rules/code-comment-style.md`）。
//! spec 改定（例: #151 で提案中の最適化後下限一律引き上げ案）が確定した場合は
//! [`floor_spec`] の定数のみを更新すれば追従できる（判定ロジック自体は不変）。

use crate::report::BenchReport;
use crate::stats::BenchError;
use serde::{Deserialize, Serialize};

/// [`FloorJudgment`] の JSON スキーマバージョン。
///
/// `report::SCHEMA_VERSION` と同じ手法（`report.rs` 冒頭コメント参照）を踏襲し、
/// [`FloorJudgment::validate`] が未知バージョンを fail-closed で拒否する。
pub const THRESHOLD_SCHEMA_VERSION: &str = "1";

/// 判定対象のバックエンド×dtype（REQ-8 下限表の行キー）。
///
/// [`crate::BenchReport::backend`] は自由文字列（`report.rs`: 「列挙型としての固定化は
/// TASK-8.2 側の関心事」）だが、判定時は本 enum へ fail-closed で解決する。
/// [`judge`] は `own`/`pytorch` の `backend` 文字列が本 enum の期待値
/// （[`BackendDtype::expected_backend_str`]）と一致しない場合、Pass に倒さずエラーを返す。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendDtype {
    CpuF32,
    CudaF32,
    CudaF16,
    MetalF32,
    MetalF16,
}

impl BackendDtype {
    /// [`crate::BenchReport::backend`] に期待される文字列（バックエンド部のみ。dtype は
    /// `BenchReport` に独立フィールドがないため文字列比較の対象外とし、[`judge`] の
    /// 呼び出し側が正しい `BackendDtype` を明示的に選ぶ責務を負う）。
    fn expected_backend_str(self) -> &'static str {
        match self {
            BackendDtype::CpuF32 => "cpu",
            BackendDtype::CudaF32 | BackendDtype::CudaF16 => "cuda",
            BackendDtype::MetalF32 | BackendDtype::MetalF16 => "metal",
        }
    }
}

/// 段階（初期リリース／最適化後。REQ-8 下限表の列キー）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stage {
    InitialRelease,
    Optimized,
}

/// 下限の設定状態。
///
/// REQ-8 下限表は「下限を設定しない」（例: CUDA f16 初期リリース。tensor core 未使用の
/// スカラー実装同士の比較は指標として無意味）と「未設定」（例: Metal f16 初期リリース。
/// 単に未実測）という 2 種類の「下限なし」状態を持つが、いずれも判定結果を
/// `Verdict::NotApplicable`（Pass でも Fail でもない）として扱う点は共通のため、
/// 本 enum では区別せず `reason` に出典を記録するのみとする。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FloorSpec {
    /// 下限（PyTorch 比パーセント）が設定されている行。
    Ratio {
        percent: f64,
        /// tensor core 実装未完了等を前提とした暫定値かどうか
        /// （REQ-8: CUDA f32/f16 最適化後下限は「暫定目標」「実測で再確定すること」と明記）。
        provisional: bool,
    },
    /// 下限を設定しない行（理由付き）。
    NotSet { reason: &'static str },
}

/// REQ-8 段階的下限表（2026-08-05 版、`docs/spec/04-requirements.md` REQ-8）を返す。
///
/// 丸め規則そのものはこの関数の関心事ではなく、spec 側で丸め規則適用済みの確定値を
/// そのまま転記する（TASK-8.2a のスコープ境界。丸め規則の共通ロジック化は TASK-8.2b）。
pub fn floor_spec(backend_dtype: BackendDtype, stage: Stage) -> FloorSpec {
    use BackendDtype::{CpuF32, CudaF16, CudaF32, MetalF16, MetalF32};
    use Stage::{InitialRelease, Optimized};

    match (backend_dtype, stage) {
        // CPU 対 PyTorch CPU（Apple M4 Max）。実測 5.3%（PoC-v2-1、10% 未満のため 1% 刻み切り下げ）。
        (CpuF32, InitialRelease) => FloorSpec::Ratio {
            percent: 5.0,
            provisional: false,
        },
        // NEON intrinsics 適用時の実効効率見積もり（PyTorch 比 29〜40% 相当）に対し
        // AMX 非搭載を踏まえ安全側に設定した確定値（REQ-8。暫定値ではない）。
        (CpuF32, Optimized) => FloorSpec::Ratio {
            percent: 20.0,
            provisional: false,
        },
        // CUDA f32 対 PyTorch CUDA（DGX Spark GB10）。実測 10.3%（PoC-v2-3、10% 以上のため 5% 刻み切り下げ）。
        (CudaF32, InitialRelease) => FloorSpec::Ratio {
            percent: 10.0,
            provisional: false,
        },
        // tensor core（WMMA/mma）化前提の見積もりだが f32/f16 混同の外挿を避けた保守的な暫定値
        // （REQ-8: 「tensor core 実装完了後の実測で本値を再確定すること」）。
        (CudaF32, Optimized) => FloorSpec::Ratio {
            percent: 40.0,
            provisional: true,
        },
        // 実測 1.9%（PoC-v2-3）は tensor core 未使用のスカラー実装同士の比較であり、
        // 指標として無意味なため下限を設定しない（REQ-8 脚注「CUDA f16 の扱い」）。
        (CudaF16, InitialRelease) => FloorSpec::NotSet {
            reason: "tensor core 未実装のスカラー実装同士の比較（実測 1.9%）は指標として無意味なため下限を設定しない（REQ-8 脚注）",
        },
        // f32 と同一水準の見積もり（tensor core 前提。理由は CudaF32/Optimized と同じ）。
        (CudaF16, Optimized) => FloorSpec::Ratio {
            percent: 40.0,
            provisional: true,
        },
        // Metal f32 対 PyTorch MPS（Apple M4 Max）。実測 23.2%（PoC-v2-4、10% 以上のため 5% 刻み切り下げ）。
        (MetalF32, InitialRelease) => FloorSpec::Ratio {
            percent: 20.0,
            provisional: false,
        },
        // PoC-v2-4 の事前固定判定基準（30%）を最適化後段階の目標として据え置き（確定値）。
        (MetalF32, Optimized) => FloorSpec::Ratio {
            percent: 30.0,
            provisional: false,
        },
        // v2 PoC で未実測のため未設定（REQ-8）。
        (MetalF16, InitialRelease) => FloorSpec::NotSet {
            reason: "v2 PoC で未実測のため未設定（REQ-8）",
        },
        (MetalF16, Optimized) => FloorSpec::NotSet {
            reason: "自作カーネルでの f16 実測後に本規則（丸め規則）で設定する（REQ-8）",
        },
    }
}

/// 合否判定結果。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Pass,
    Fail,
    /// 下限が設定されていない行（[`FloorSpec::NotSet`]）に対する判定。
    /// 実測比率は記録するが Pass/Fail のいずれにも倒さない
    /// （REQ-8「実測値 1.9% を制約事項としてのみ記録する」に対応）。
    NotApplicable,
}

/// 合否判定結果（1 計測対象分）。
///
/// `guardrail`（依存方向は `guardrail` → `bench-harness`。`report.rs` 冒頭コメント参照）
/// からの参照可能性を担保するため、[`crate::BenchReport`] と同じ serde 対応 DTO パターン
/// （`to_json`/`from_json` が必ず [`Self::validate`] を通す）を踏襲する。
/// guardrail / self-repair クレート自体への実配線は本イシューのスコープ外（計画書 2 章）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloorJudgment {
    pub schema_version: String,
    pub backend_dtype: BackendDtype,
    pub stage: Stage,
    /// 下限（PyTorch 比パーセント）。[`Verdict::NotApplicable`] の場合は `None`。
    pub floor_percent: Option<f64>,
    /// 下限が暫定値かどうか（[`FloorSpec::Ratio::provisional`] をそのまま転記）。
    /// [`Verdict::NotApplicable`] の場合は `None`。
    pub floor_provisional: Option<bool>,
    /// 中央値ベースの実測比率（`pytorch.median_secs / own.median_secs * 100`）。
    pub measured_ratio_percent: f64,
    /// Q1/Q3 由来の比率レンジ下限（`pytorch.q1_secs / own.q3_secs * 100`）。
    /// 単一のパーセンタイル値のみでの合否判定を避けるため、証跡として必ず記録する
    /// （REQ-8: 「単一のパーセンタイル値のみでの合否判定は行わないこと」）。
    pub ratio_q1_percent: f64,
    /// Q1/Q3 由来の比率レンジ上限（`pytorch.q3_secs / own.q1_secs * 100`）。
    pub ratio_q3_percent: f64,
    pub verdict: Verdict,
}

impl FloorJudgment {
    /// [`Self::to_json`]/[`Self::from_json`] が必ず経由する fail-closed 検証
    /// （`.claude/rules/security.md` A08: 判定の迂回経路を作らない）。
    ///
    /// # Errors
    ///
    /// 検証に失敗した場合 `BenchError::ProtocolViolation`。
    pub fn validate(&self) -> Result<(), BenchError> {
        if self.schema_version != THRESHOLD_SCHEMA_VERSION {
            return Err(BenchError::ProtocolViolation(format!(
                "未知の schema_version: 期待値 {THRESHOLD_SCHEMA_VERSION:?}, 実際 {:?}",
                self.schema_version
            )));
        }
        if !self.measured_ratio_percent.is_finite()
            || !self.ratio_q1_percent.is_finite()
            || !self.ratio_q3_percent.is_finite()
        {
            return Err(BenchError::ProtocolViolation(
                "measured_ratio_percent / ratio_q1_percent / ratio_q3_percent のいずれかが非有限値"
                    .to_string(),
            ));
        }
        if !(self.ratio_q1_percent <= self.measured_ratio_percent
            && self.measured_ratio_percent <= self.ratio_q3_percent)
        {
            return Err(BenchError::ProtocolViolation(format!(
                "ratio_q1_percent <= measured_ratio_percent <= ratio_q3_percent を満たさない: \
                 q1={}, measured={}, q3={}",
                self.ratio_q1_percent, self.measured_ratio_percent, self.ratio_q3_percent
            )));
        }
        match (self.verdict, self.floor_percent, self.floor_provisional) {
            (Verdict::NotApplicable, None, None) => Ok(()),
            (Verdict::NotApplicable, _, _) => Err(BenchError::ProtocolViolation(
                "verdict が NotApplicable なのに floor_percent/floor_provisional が設定されている"
                    .to_string(),
            )),
            (Verdict::Pass | Verdict::Fail, Some(_), Some(_)) => Ok(()),
            (Verdict::Pass | Verdict::Fail, _, _) => Err(BenchError::ProtocolViolation(
                "verdict が Pass/Fail なのに floor_percent/floor_provisional が未設定".to_string(),
            )),
        }
    }

    /// JSON へシリアライズする（シリアライズ前に [`Self::validate`] を実行。
    /// `report.rs::BenchReport::to_json` と同じ理由で非有限 f64 の silent corruption を防ぐ）。
    ///
    /// # Errors
    ///
    /// [`Self::validate`] の失敗、または JSON エンコード失敗（後者も `BenchError::ProtocolViolation`）。
    pub fn to_json(&self) -> Result<String, BenchError> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|e| BenchError::ProtocolViolation(format!("JSON エンコード失敗: {e}")))
    }

    /// JSON からデシリアライズし、[`Self::validate`] を通してから返す
    /// （検証を経ずに生値へアクセスできる公開経路は設けない）。
    ///
    /// # Errors
    ///
    /// JSON デコード失敗、または [`Self::validate`] の失敗（いずれも `BenchError::ProtocolViolation`）。
    pub fn from_json(json: &str) -> Result<Self, BenchError> {
        let judgment: Self = serde_json::from_str(json)
            .map_err(|e| BenchError::ProtocolViolation(format!("JSON デコード失敗: {e}")))?;
        judgment.validate()?;
        Ok(judgment)
    }
}

/// 自作実装（`own`）・PyTorch 参照実装（`pytorch`）それぞれの [`BenchReport`] から
/// 実測比率を算出し、REQ-8 下限表（[`floor_spec`]）と突合して合否を返す。
///
/// `guardrail`（`docs/guardrail-self-repair-cli.md` 2.1 節）・`self-repair`（TASK-3.2）から
/// 「計測結果に対し合否が自動判定される」（本イシュー受け入れ条件）ための入口となる。
///
/// # 契約
///
/// - `own`/`pytorch` はいずれも `backend_dtype` の [`BackendDtype::expected_backend_str`] と
///   一致するバックエンド上で計測された [`BenchReport`] であること。不一致は
///   Pass に倒さずエラーとする（fail-closed。`.claude/rules/security.md` A08）。
/// - 比率＝スループット比＝所要時間の逆数比: `pytorch.median_secs / own.median_secs * 100`。
///   同一ワークロード・同一 dtype で計測された前提（呼び出し側の責務）。
/// - `own` 側の値（比率の分母）はゼロ除算・非有限伝播を防ぐため正の有限値を要求する。
///   `pytorch` 側（分子）は非負の有限値を要求する。
///
/// # Errors
///
/// - `own`/`pytorch` が [`BenchReport::validate`]（TASK-8.1 プロトコル遵守）を満たさない場合
/// - `own`/`pytorch` の `backend` が `backend_dtype` の期待値と一致しない場合（未知の組合せ）
/// - 比率計算に必要な値（`own.median_secs`・`own.q1_secs`・`pytorch.*`）が非有限・不正な符号の場合
pub fn judge(
    own: &BenchReport,
    pytorch: &BenchReport,
    backend_dtype: BackendDtype,
    stage: Stage,
) -> Result<FloorJudgment, BenchError> {
    // 検証を経ない判定経路を作らない（.claude/rules/security.md A08）。
    // BenchReport::validate は TASK-8.1 計測プロトコル遵守（warmup/iters 下限・有限性・
    // q1<=median<=q3 順序）を確認する。JSON 経由の外部入力は特にここで先に弾く（A03）。
    own.validate()?;
    pytorch.validate()?;

    // 未知の backend/backend_dtype 組合せは Pass に倒さずエラーとする（fail-closed）。
    let expected_backend = backend_dtype.expected_backend_str();
    if own.backend != expected_backend {
        return Err(BenchError::ProtocolViolation(format!(
            "own.backend({:?}) が backend_dtype の期待バックエンド({expected_backend:?})と不一致",
            own.backend
        )));
    }
    if pytorch.backend != expected_backend {
        return Err(BenchError::ProtocolViolation(format!(
            "pytorch.backend({:?}) が backend_dtype の期待バックエンド({expected_backend:?})と不一致",
            pytorch.backend
        )));
    }

    // 比率の分母（own 側）はゼロ除算・NaN/inf 伝播を防ぐため正の有限値を要求する。
    // BenchReport::validate は有限性のみを保証し正値までは保証しないため、ここで追加検証する
    // （q1_secs は理論上 0 になり得るため、q1 を分母に使う ratio_q3_percent の計算前に確認する）。
    if !(own.median_secs.is_finite() && own.median_secs > 0.0) {
        return Err(BenchError::ProtocolViolation(format!(
            "own.median_secs は正の有限値が必須（比率計算の分母）。実際: {}",
            own.median_secs
        )));
    }
    if !(own.q1_secs.is_finite() && own.q1_secs > 0.0) {
        return Err(BenchError::ProtocolViolation(format!(
            "own.q1_secs は正の有限値が必須（比率レンジ上限の分母）。実際: {}",
            own.q1_secs
        )));
    }
    // own.q3_secs はここまでの検証（BenchReport::validate の q1<=median<=q3 順序 かつ
    // own.median_secs > 0）から正の有限値であることが導かれるため、追加検証は不要。

    // 比率の分子（pytorch 側）は非負の有限値を要求する。
    if !(pytorch.median_secs.is_finite() && pytorch.median_secs >= 0.0) {
        return Err(BenchError::ProtocolViolation(format!(
            "pytorch.median_secs は非負の有限値が必須。実際: {}",
            pytorch.median_secs
        )));
    }
    if !(pytorch.q1_secs.is_finite()
        && pytorch.q1_secs >= 0.0
        && pytorch.q3_secs.is_finite()
        && pytorch.q3_secs >= 0.0)
    {
        return Err(BenchError::ProtocolViolation(
            "pytorch.q1_secs / pytorch.q3_secs は非負の有限値が必須".to_string(),
        ));
    }

    // スループット比＝所要時間の逆数比。own が速い（所要時間が短い）ほど比率は高くなる。
    let measured_ratio_percent = pytorch.median_secs / own.median_secs * 100.0;
    // Q1/Q3 由来の比率レンジ: own の計測ばらつきを「最良（q1）／最悪（q3）」の両極で反映する。
    // pytorch.q3/own.q1（分子最大・分母最小）は上限、pytorch.q1/own.q3（分子最小・分母最大）は
    // 下限となり、[`FloorJudgment::validate`] が検査する
    // `ratio_q1 <= measured_ratio <= ratio_q3` を常に満たす。
    let ratio_q3_percent = pytorch.q3_secs / own.q1_secs * 100.0;
    let ratio_q1_percent = pytorch.q1_secs / own.q3_secs * 100.0;

    let (floor_percent, floor_provisional, verdict) = match floor_spec(backend_dtype, stage) {
        FloorSpec::NotSet { .. } => (None, None, Verdict::NotApplicable),
        FloorSpec::Ratio {
            percent,
            provisional,
        } => {
            // 下限ちょうどは Pass（計画書 4.2 節: 「measured_ratio >= floor で Pass」）。
            let verdict = if measured_ratio_percent >= percent {
                Verdict::Pass
            } else {
                Verdict::Fail
            };
            (Some(percent), Some(provisional), verdict)
        }
    };

    let judgment = FloorJudgment {
        schema_version: THRESHOLD_SCHEMA_VERSION.to_string(),
        backend_dtype,
        stage,
        floor_percent,
        floor_provisional,
        measured_ratio_percent,
        ratio_q1_percent,
        ratio_q3_percent,
        verdict,
    };
    // 自前で組み立てた値だが、判定を迂回して不変条件を破った FloorJudgment を返さないよう
    // 防御的に検証する（BenchReport::from_measurement と同じ「構築経路全体で不変条件を統一する」方針）。
    judgment.validate()?;
    Ok(judgment)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 判定テスト専用の `BenchReport` を構築する（`from_measurement` を経由せず、
    /// 比率計算に必要な `median_secs`/`q1_secs`/`q3_secs` を直接指定する）。
    /// `samples_secs` は `BenchReport::validate` の `len() == iters` 制約を満たすため
    /// `median` で埋めた 20 要素とする（本テストの関心は比率計算であり分位点の一致は問わない）。
    fn report(backend: &str, median: f64, q1: f64, q3: f64) -> BenchReport {
        BenchReport {
            schema_version: crate::report::SCHEMA_VERSION.to_string(),
            name: "threshold_test".to_string(),
            backend: backend.to_string(),
            warmup: 20,
            iters: 20,
            median_secs: median,
            q1_secs: q1,
            q3_secs: q3,
            samples_secs: vec![median; 20],
        }
    }

    #[test]
    fn cpu_f32_initial_floor_exactly_at_5_percent_passes() {
        // own.median=1.0, pytorch.median=0.05 -> ratio = 0.05/1.0*100 = 5.0（下限ちょうど）。
        let own = report("cpu", 1.0, 0.9, 1.1);
        let pytorch = report("cpu", 0.05, 0.045, 0.055);
        let j = judge(&own, &pytorch, BackendDtype::CpuF32, Stage::InitialRelease)
            .expect("有効な入力のため成功するはず");
        assert_eq!(j.verdict, Verdict::Pass);
        assert_eq!(j.floor_percent, Some(5.0));
        assert!((j.measured_ratio_percent - 5.0).abs() < 1e-9);
    }

    #[test]
    fn cpu_f32_initial_floor_just_below_5_percent_fails() {
        let own = report("cpu", 1.0, 0.9, 1.1);
        let pytorch = report("cpu", 0.0499, 0.045, 0.055);
        let j = judge(&own, &pytorch, BackendDtype::CpuF32, Stage::InitialRelease).unwrap();
        assert_eq!(j.verdict, Verdict::Fail);
    }

    #[test]
    fn cuda_f32_initial_floor_10_percent_boundary() {
        let own = report("cuda", 1.0, 0.9, 1.1);

        let pytorch_pass = report("cuda", 0.10, 0.09, 0.11);
        let pass = judge(
            &own,
            &pytorch_pass,
            BackendDtype::CudaF32,
            Stage::InitialRelease,
        )
        .unwrap();
        assert_eq!(pass.verdict, Verdict::Pass);

        let pytorch_fail = report("cuda", 0.099, 0.09, 0.11);
        let fail = judge(
            &own,
            &pytorch_fail,
            BackendDtype::CudaF32,
            Stage::InitialRelease,
        )
        .unwrap();
        assert_eq!(fail.verdict, Verdict::Fail);
    }

    #[test]
    fn cuda_f16_initial_release_is_not_applicable() {
        // 実測 1.9% 相当（tensor core 未使用）でも Fail にならず NotApplicable として記録される。
        let own = report("cuda", 1.0, 0.9, 1.1);
        let pytorch = report("cuda", 0.019, 0.017, 0.021);
        let j = judge(&own, &pytorch, BackendDtype::CudaF16, Stage::InitialRelease).unwrap();
        assert_eq!(j.verdict, Verdict::NotApplicable);
        assert_eq!(j.floor_percent, None);
        assert_eq!(j.floor_provisional, None);
        // NotApplicable でも実測比率自体は記録される。
        assert!((j.measured_ratio_percent - 1.9).abs() < 1e-9);
    }

    #[test]
    fn metal_f16_both_stages_are_not_applicable() {
        let own = report("metal", 1.0, 0.9, 1.1);
        let pytorch = report("metal", 0.5, 0.45, 0.55);
        for stage in [Stage::InitialRelease, Stage::Optimized] {
            let j = judge(&own, &pytorch, BackendDtype::MetalF16, stage).unwrap();
            assert_eq!(j.verdict, Verdict::NotApplicable);
        }
    }

    #[test]
    fn zero_own_median_secs_is_rejected() {
        let own = report("cpu", 0.0, 0.0, 0.0);
        let pytorch = report("cpu", 0.05, 0.045, 0.055);
        let err = judge(&own, &pytorch, BackendDtype::CpuF32, Stage::InitialRelease)
            .expect_err("own.median_secs=0 はゼロ除算防止のため拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn nan_median_secs_is_rejected_via_bench_report_validate() {
        let mut own = report("cpu", 1.0, 0.9, 1.1);
        own.median_secs = f64::NAN;
        let pytorch = report("cpu", 0.05, 0.045, 0.055);
        let err = judge(&own, &pytorch, BackendDtype::CpuF32, Stage::InitialRelease)
            .expect_err("NaN を含む BenchReport は own.validate() の時点で拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn unknown_backend_string_does_not_fall_back_to_pass() {
        // backend_dtype=CpuF32 の期待値は "cpu" だが、実際の backend は未知の "tpu"。
        // Pass に倒れず必ずエラーになることを確認する（fail-closed）。
        let own = report("tpu", 1.0, 0.9, 1.1);
        let pytorch = report("tpu", 0.05, 0.045, 0.055);
        let err = judge(&own, &pytorch, BackendDtype::CpuF32, Stage::InitialRelease)
            .expect_err("backend 不一致は拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }

    #[test]
    fn json_round_trip_preserves_judgment() {
        let own = report("metal", 1.0, 0.9, 1.1);
        let pytorch = report("metal", 0.25, 0.2, 0.3);
        let j = judge(
            &own,
            &pytorch,
            BackendDtype::MetalF32,
            Stage::InitialRelease,
        )
        .unwrap();
        let json = j
            .to_json()
            .expect("有効な判定結果は JSON エンコードできるはず");
        let restored = FloorJudgment::from_json(&json).expect("往復後も検証を通るはず");
        assert_eq!(j, restored);
    }

    #[test]
    fn from_json_rejects_unknown_schema_version() {
        let own = report("metal", 1.0, 0.9, 1.1);
        let pytorch = report("metal", 0.25, 0.2, 0.3);
        let mut j = judge(
            &own,
            &pytorch,
            BackendDtype::MetalF32,
            Stage::InitialRelease,
        )
        .unwrap();
        j.schema_version = "999".to_string();
        let json = serde_json::to_string(&j).unwrap();
        let err =
            FloorJudgment::from_json(&json).expect_err("未知の schema_version は拒否されるはず");
        assert!(matches!(err, BenchError::ProtocolViolation(_)));
    }
}
