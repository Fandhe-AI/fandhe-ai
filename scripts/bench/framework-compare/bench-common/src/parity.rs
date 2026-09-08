//! GEMM 結果の要素単位検証（イシュー #970）。
//!
//! 従来の全要素和 checksum（`crate::validate_gemm_checksum`・
//! `summarize.py` の checksum 相互突合。イシュー #965）は「縮退（全ゼロ／
//! 非有限）」と「他フレームワークとの粗い一致」しか検出できず、和が偶然
//! 一致する破損（要素の入れ替わり・正負誤差の相殺）を見逃す。本モジュール
//! は本体の数値一致契約（`crates/backend-cpu/src/parity.rs` の複合判定
//! 「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）と同じ式・同じ閾値
//! （既存 2 条件は本体と bit 単位で不変）を使い、GEMM 結果を要素単位で
//! 参照実装と突合する。加えて、本モジュール限定（ハーネス限定）の承認済み
//! 第 3 救済項（スケール付き絶対誤差。[`ScaledAbsTolerance`]・
//! [`PARITY_SCALED_ABS_COEFF`]）を OR 追加できる。本体
//! `compare`／`assert_parity`／`ParityBaseline` の判定式にはこの項は
//! 存在せず、あくまで candle/Burn 側参照 GEMM のキャンセレーション由来の
//! 丸め誤差フロアを許容するためのハーネス側運用（イシュー #1241 承認記録
//! コメント・`docs/candle-parity-tolerance-contract-decision.md` §8）。
//!
//! # 参照実装の選択
//!
//! fandhe-ai 0.3.0（crates.io 公開版）の facade は parity API を公開して
//! おらず、candle/Burn を参照にするとバイナリ間で結果を受け渡す仕組みが
//! 別途必要になる。本モジュールは本体 `backend-cpu::parity::matmul_reference_fma`
//! と同じ FMA 契約（f32 `mul_add`・逐次 k 昇順）を持つ参照 GEMM
//! （[`GemmReference`]）を自前実装することで、各バイナリが自己完結で
//! 突合できるようにする。この契約は CPU 参照 vs GPU の K=4096 ストレス
//! ケースで fail 0 を確認済み（`cpu_cuda_parity.rs::naive_f32_k4096_stress_poc_v2_5`）
//! であり、同じ契約を使うことで本ツールの閾値の意味を本体と揃える
//! （f64 累積参照は「真値との差」という別の指標になり契約と整合しない
//! ため不採用。`scripts/bench/framework-compare/README.md` 参照）。
//!
//! # 呼び出し元
//!
//! `bench-fandhe`/`bench-candle`/`bench-burn` の `run_gemm`（gemm タスクの
//! みが対象。train/infer は fandhe-ai の重み初期化が candle/Burn と異なる
//! 設計のため比較不能で対象外）が、計測窓（`Instant::now()` 〜
//! `elapsed()`）の外で毎反復 [`GemmReference::verify`] を呼び、反復間の
//! worst-case を [`ParityStats::worst`] で集約して `Record::parity` に
//! 記録する。`summarize.py` はこれを読み、閾値超過を「無効」として
//! fail-closed に報告する。

use crate::BenchError;

/// 本体 `backend-cpu::parity::RELATIVE_TOLERANCE` と同値の相対誤差閾値。
///
/// **正は `crates/backend-cpu/src/parity.rs::RELATIVE_TOLERANCE`**。本
/// ワークスペースは独立 workspace（`.claude/rules/deps-policy.md` 第 9
/// 区分）のため本体クレートへ path 依存できず、値をここへ再定義する
/// 以外の選択肢がない。乖離は `tests::parity_tolerances_match_backend_cpu_contract`
/// が本体ソースを読んで機械照合し fail-closed に検出する（イシュー #970
/// codex-review 指摘・PR #978 P1）。
///
/// **変更はユーザー承認必須**（`.claude/rules/coding-rust.md`
/// 「バックエンド間数値一致テストの許容誤差を単独で緩和しない」・
/// `.claude/rules/security.md` A08）。ベンチの相互検証は本体の数値一致
/// 契約と同じ意味を持たせるため、閾値を独自に緩めない。
pub const PARITY_REL_TOL: f64 = 1e-3;

/// 本体 `backend-cpu::parity::ABSOLUTE_RESCUE_THRESHOLD` と同値の絶対誤差
/// 救済閾値（0 近傍の相対誤差跳ね上がり対策）。正・乖離検出・変更承認の
/// 方針は [`PARITY_REL_TOL`] と同じ。
pub const PARITY_ABS_TOL: f64 = 1e-5;

/// スケール付き絶対誤差救済項（候補 A-1・係数 `c=0.5`。`u=2^-24` 表記）の
/// 係数。判定式は `diff <= PARITY_SCALED_ABS_COEFF * F32_UNIT_ROUNDOFF *
/// K * S_A * S_B`（[`ScaledAbsTolerance::bound`]）。
///
/// **本項目は framework-compare ハーネス限定の承認済み契約**であり、本体
/// `crates/backend-cpu/src/parity.rs::compare`・
/// `crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline` の
/// 判定式には対応する項が**設計上存在しない**（イシュー #1241 承認記録
/// コメント〈2026-09-08〉・`docs/candle-parity-tolerance-contract-decision.md`
/// §8。スコープ 3「ハーネス限定」の承認内容）。ハーネスの
/// [`GemmReference::verify`] が candle/Burn の cuBLAS/oneDNN 系参照 GEMM を
/// N=2048 で比較する際、キャンセレーションにより最終値が縮小し
/// [`PARITY_REL_TOL`]／[`PARITY_ABS_TOL`] の両方を割る 2 要素が決定的に
/// 発生する（`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.3・§11・
/// `docs/perf/logs/cuda-gemm-candle-parity-1184/`）ため、比較対象側の
/// 丸め誤差フロアを許容する第 3 項を OR 追加する。fandhe-ai 側 GEMM は
/// 本項の救済なしに 0 fail のままであることをテストで固定する
/// （`bench-fandhe::tests::gemm_cpu_parity_zero_fail_without_scaled_rescue`）。
///
/// **変更はユーザー承認必須**（[`PARITY_REL_TOL`] と同じ方針。
/// `.claude/rules/coding-rust.md`・`.claude/rules/security.md` A08）。
pub const PARITY_SCALED_ABS_COEFF: f64 = 0.5;

/// f32 の unit roundoff（machine epsilon の半分。`2^-24`）。承認済み判定式
/// （[`PARITY_SCALED_ABS_COEFF`]）の `u` に対応する。`f32::EPSILON`
/// （`2^-23`）の半分として const 評価できるため、リテラルの二重管理を
/// 避けてここから導出する。
pub const F32_UNIT_ROUNDOFF: f64 = 1.0 / 16_777_216.0;

/// [`PARITY_SCALED_ABS_COEFF`]・[`F32_UNIT_ROUNDOFF`] による第 3 救済項
/// （スケール付き絶対誤差。候補 A-1）の適用パラメータ。
///
/// `bound = c・u・K・S_A・S_B`（`c`=[`PARITY_SCALED_ABS_COEFF`]・
/// `u`=[`F32_UNIT_ROUNDOFF`]・`K`=内積長・`S_A`/`S_B`=入力行列 A/B の
/// 絶対値の全体最大）。要素の複合判定 `pass ⇔ rel < PARITY_REL_TOL ∨
/// diff < PARITY_ABS_TOL ∨ diff <= bound` へ [`element_error`] が OR 追加
/// する（`docs/candle-parity-tolerance-contract-decision.md` §8「承認内容」
/// 2 行目と同一式）。
///
/// [`NONE`](Self::NONE)（`bound() == 0.0`）を渡すと第 3 項が事実上
/// 無効化され、既存 2 条件のみの判定（レガシー同値）になる
/// （`compare_elementwise` の既存呼び出し元・テストの後方互換に使う）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaledAbsTolerance {
    /// 内積長（正方 GEMM の一辺長 `n` と同値。[`GemmReference::compute`]
    /// が `n` をそのまま渡す）。
    pub k: usize,
    /// 入力行列 A の絶対値の全体最大（`max|A|`）。非有限要素を検出した
    /// 場合は `f64::INFINITY` センチネル（[`bound`](Self::bound) が
    /// fail-closed に 0.0 へ倒す）。
    pub scale_a: f64,
    /// 入力行列 B の絶対値の全体最大（`max|B|`）。`scale_a` と同じ
    /// センチネル運用。
    pub scale_b: f64,
}

impl ScaledAbsTolerance {
    /// 第 3 項を無効化する値（`bound() == 0.0`）。`diff <= 0.0` は
    /// `diff == 0.0`（既存 2 条件で必ず pass する場合のみ）に限られるため、
    /// 既存判定に対し無害（レガシー同値）。
    pub const NONE: Self = Self {
        k: 0,
        scale_a: 0.0,
        scale_b: 0.0,
    };

    /// GEMM 入力 `a`・`b`（flat・行優先。長さは呼び出し元
    /// [`GemmReference::compute`] が検証済み）から `S_A`・`S_B` を導出する。
    ///
    /// `f64::max` は NaN を含む片側を暗黙に捨てる（`x.max(NaN) == x`）ため、
    /// [`worst_f64`] と同じ回避策で非有限要素を検出したら `f64::INFINITY`
    /// センチネルへ固定する（本体 `max_nonfinite_aware` と同じ論点。
    /// [`bound`](Self::bound) 側で救済なし〈0.0〉へ倒すことで、
    /// `diff <= +Inf` による全要素救済という fail-closed 違反を構造的に
    /// 防ぐ。A08）。
    pub fn from_inputs(k: usize, a: &[f32], b: &[f32]) -> Self {
        Self {
            k,
            scale_a: max_abs_f64(a),
            scale_b: max_abs_f64(b),
        }
    }

    /// スケール付き絶対誤差の許容量 `c・u・K・S_A・S_B` を返す。
    /// `scale_a`／`scale_b` が非有限・負、または結果が非有限になる場合は
    /// 救済なし（`0.0`）に倒す fail-closed 設計（[`from_inputs`](Self::from_inputs)
    /// のドキュメント参照）。
    pub fn bound(&self) -> f64 {
        if !self.scale_a.is_finite() || !self.scale_b.is_finite() {
            return 0.0;
        }
        if self.scale_a < 0.0 || self.scale_b < 0.0 {
            return 0.0;
        }
        let bound = PARITY_SCALED_ABS_COEFF
            * F32_UNIT_ROUNDOFF
            * (self.k as f64)
            * self.scale_a
            * self.scale_b;
        if bound.is_finite() { bound } else { 0.0 }
    }
}

/// `values` の絶対値の全体最大。空スライスは `0.0`（`ScaledAbsTolerance`
/// の呼び出し元は `n >= 1` を [`GemmReference::compute`] で検証済みのため
/// 実質到達しないが、`pub` な `from_inputs` の契約として空入力を panic
/// させない。`f64::max` の NaN 捨て問題を避けるため [`worst_f64`] と同じ
/// センチネル方式を使う）。
fn max_abs_f64(values: &[f32]) -> f64 {
    let mut acc = 0.0f64;
    for &v in values {
        let av = (v as f64).abs();
        acc = worst_f64(acc, av);
    }
    acc
}

/// GEMM の要素単位検証結果。反復間の worst-case を保持する
/// （[`ParityStats::worst`]）ため、`fail_count`・`max_abs_err`・
/// `max_rel_err` は必ずしも同一反復由来ではない（診断指標としてはそれで
/// 十分: 「どれか 1 回でも壊れていたか」「どこまで悪化したか」が分かれば
/// 良く、どの反復かの追跡は本ツールのスコープ外）。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityStats {
    /// 比較した要素数（= N×N）。
    pub total: usize,
    /// 複合判定で不合格だった要素数。0 なら「無効」ではない。
    pub fail_count: usize,
    /// 全要素中の絶対誤差の最大値。非有限（NaN/Inf 混入）の要素があった
    /// 場合は `f64::INFINITY`（`crates/backend-cpu/src/parity.rs` の
    /// `max_fail_abs_diff` センチネルと同じ考え方）。
    pub max_abs_err: f64,
    /// 全要素中の相対誤差の最大値。非有限の要素があった場合は
    /// `f64::INFINITY`。
    pub max_rel_err: f64,
    /// 適用したスケール付き絶対誤差救済項（[`ScaledAbsTolerance`]）の
    /// bound（`ScaledAbsTolerance::bound`）。非有限入力を検出していた
    /// 場合は `f64::INFINITY` センチネル（`bound()` 自体は fail-closed で
    /// 0.0 を返すため、このフィールドは診断用に検出有無を残す目的）。
    pub scaled_abs_bound: f64,
    /// 既存 2 条件（[`PARITY_REL_TOL`]／[`PARITY_ABS_TOL`]）では fail だが
    /// スケール付き絶対誤差救済項（[`PARITY_SCALED_ABS_COEFF`]）で pass に
    /// 転じた要素数。0 なら本救済項に依存せず判定が確定している
    /// （`bench-fandhe` 側は本フィールドが常に 0 であることをテストで
    /// 固定し、救済項が本体側の回帰を隠す経路を遮断する）。
    pub scaled_abs_rescued: usize,
}

impl ParityStats {
    /// 反復間の worst-case 集約。各フィールドを独立に最大化する
    /// （`fail_count` の最大値と `max_abs_err` の最大値が異なる反復由来に
    /// なりうるが、「毎反復検証し、途中反復の破損を見逃さない」という
    /// 受け入れ条件を満たすための診断集約であり、単一反復の完全な再現を
    /// 目的としないため許容する）。
    pub fn worst(self, other: Self) -> Self {
        Self {
            total: self.total.max(other.total),
            fail_count: self.fail_count.max(other.fail_count),
            max_abs_err: worst_f64(self.max_abs_err, other.max_abs_err),
            max_rel_err: worst_f64(self.max_rel_err, other.max_rel_err),
            scaled_abs_bound: worst_f64(self.scaled_abs_bound, other.scaled_abs_bound),
            scaled_abs_rescued: self.scaled_abs_rescued.max(other.scaled_abs_rescued),
        }
    }
}

/// `f64::max` は NaN/Inf を含む片側を捨ててしまう（`x.max(NaN) == x`）ため、
/// どちらかが非有限なら結果を `INFINITY` センチネルへ固定する
/// （`crates/backend-cpu/src/parity.rs::compare` の `max_fail_abs_diff`
/// 集計と同じ回避策）。
fn worst_f64(a: f64, b: f64) -> f64 {
    if a.is_finite() && b.is_finite() {
        a.max(b)
    } else {
        f64::INFINITY
    }
}

/// 1 要素分の複合判定の中間値（絶対誤差・相対誤差・pass/fail）。
///
/// [`compare_elementwise`]（集計のみ）と [`dump_parity_failures`]（fail
/// 要素の値をダンプ。イシュー #1183）の双方が同じ式を使うための切り出し。
/// 式そのもの（分母 1e-12 下支え・NaN の fail 側倒し）は変更しない
/// （`crates/backend-cpu/src/parity.rs::compare` と同一方式）。
#[derive(Debug, Clone, Copy)]
struct ElementError {
    abs: f64,
    rel: f64,
    pass: bool,
    /// 既存 2 条件（`legacy_pass`）では fail だがスケール付き絶対誤差
    /// 救済項（`scaled_abs_bound`）で pass に転じた場合に `true`
    /// （[`ParityStats::scaled_abs_rescued`] の集計元）。
    rescued_by_scaled_abs: bool,
}

/// `scaled_abs_bound` は [`ScaledAbsTolerance::bound`] の結果（ハーネス
/// 限定の第 3 救済項。`0.0` なら実質無効でレガシー判定と完全同値）。
#[inline]
fn element_error(actual: f32, reference: f32, scaled_abs_bound: f64) -> ElementError {
    let xf = actual as f64;
    let yf = reference as f64;
    let diff = (xf - yf).abs();
    // 真値 0 近傍での相対誤差の跳ね上がりを避けるため、分母を 1e-12 で
    // 下支えする（本体 `parity::compare` と同一方式）。
    let scale = xf.abs().max(yf.abs()).max(1e-12);
    let rel = diff / scale;
    // NaN 混入時 `rel`/`diff` は NaN になり `<` 比較は常に false のため
    // fail 側に倒れる（本体 `parity::compare` と同じ安全側の挙動）。既存
    // 2 条件は本体契約と bit 単位で不変（`legacy_pass` として分離し、
    // ハーネス限定の第 3 項〈`scaled_abs_pass`〉を OR で追加する）。
    let legacy_pass = rel < PARITY_REL_TOL || diff < PARITY_ABS_TOL;
    // NaN diff は `<=` も false のため fail 側に倒れる（legacy と同じ
    // 安全側の挙動）。承認済み判定式は境界含む `<=`
    // （`docs/candle-parity-tolerance-contract-decision.md` §8）。
    let scaled_abs_pass = diff <= scaled_abs_bound;
    let pass = legacy_pass || scaled_abs_pass;
    ElementError {
        abs: diff,
        rel,
        pass,
        rescued_by_scaled_abs: !legacy_pass && scaled_abs_pass,
    }
}

/// 要素単位の複合判定。既存 2 条件（[`PARITY_REL_TOL`]／[`PARITY_ABS_TOL`]）
/// は `crates/backend-cpu/src/parity.rs::compare` と同じ式で不変。`tol`
/// （[`ScaledAbsTolerance`]）はハーネス限定の第 3 救済項で、
/// [`ScaledAbsTolerance::NONE`] を渡すとレガシー判定（既存 2 条件のみ）と
/// 完全同値になる。
/// `actual`・`reference` は同じ長さの flat データを想定し、長さ不一致は
/// 呼び出し誤りの早期検出として型付きエラーを返す。
pub fn compare_elementwise(
    actual: &[f32],
    reference: &[f32],
    tol: &ScaledAbsTolerance,
) -> Result<ParityStats, BenchError> {
    if actual.len() != reference.len() {
        return Err(BenchError::ParityLengthMismatch {
            expected: reference.len(),
            actual: actual.len(),
        });
    }

    let bound = tol.bound();
    let total = actual.len();
    let mut fail_count = 0usize;
    let mut max_abs_err = 0.0f64;
    let mut max_rel_err = 0.0f64;
    let mut scaled_abs_rescued = 0usize;

    for (&x, &y) in actual.iter().zip(reference.iter()) {
        let err = element_error(x, y, bound);
        if !err.pass {
            fail_count += 1;
        }
        if err.rescued_by_scaled_abs {
            scaled_abs_rescued += 1;
        }

        max_abs_err = if err.abs.is_finite() {
            max_abs_err.max(err.abs)
        } else {
            f64::INFINITY
        };
        max_rel_err = if err.rel.is_finite() {
            max_rel_err.max(err.rel)
        } else {
            f64::INFINITY
        };
    }

    // 非有限スケールを検出していた場合（`bound()` 自体は fail-closed で
    // 0.0 を返す）、診断値としては検出があったことを `INFINITY` で残す。
    let scaled_abs_bound_diag = if tol.scale_a.is_finite() && tol.scale_b.is_finite() {
        bound
    } else {
        f64::INFINITY
    };

    Ok(ParityStats {
        total,
        fail_count,
        max_abs_err,
        max_rel_err,
        scaled_abs_bound: scaled_abs_bound_diag,
        scaled_abs_rescued,
    })
}

/// `FRAMEWORK_COMPARE_PARITY_DUMP` 環境変数の契約名（イシュー #1183）。
///
/// `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.3 で「fail 要素の
/// 値（index・reference・実測値）を取得する診断計装」として検討され
/// 未実施だったものを、opt-in の環境変数として実装する。値は
/// [`ParityDumpConfig::parse`] の allowlist（未設定 / `""` / `"0"` で無効、
/// `"1"` で既定上限、正の整数文字列でその上限）のみを受理し、それ以外は
/// `BenchError::InvalidParityDumpEnv` で fail-fast する（security.md A03。
/// `BenchError::InvalidMode` と同じ思想）。ファイルパス等は受け取らず
/// 出力先は stderr 固定（パス注入面を増やさない）。
pub const PARITY_DUMP_ENV: &str = "FRAMEWORK_COMPARE_PARITY_DUMP";

/// [`PARITY_DUMP_ENV`] が `"1"` のときに使う既定の出力上限（1 回の
/// [`GemmReference::verify`] 呼び出しあたりの fail 要素数）。
pub const PARITY_DUMP_DEFAULT_LIMIT: usize = 64;

/// fail 要素ダンプの有効化設定（イシュー #1183）。`limit` は 1 回の
/// `verify` 呼び出しあたりの出力上限で、決定的に繰り返し fail する要素が
/// 反復ごとに再ダンプされてもログが際限なく膨らまないよう bound する
/// （`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.3 の N=2048
/// ケースは毎反復同じ 2 要素が fail する想定であり、これは反復間の
/// 非決定性も可視化するための仕様として維持する）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParityDumpConfig {
    pub limit: usize,
}

impl ParityDumpConfig {
    /// [`PARITY_DUMP_ENV`] の生値（`std::env::var_os` の結果を呼び出し元が
    /// UTF-8 変換したもの）を解釈する純関数。プロセス環境を直接読まない
    /// ため env 非依存にユニットテストできる。
    pub fn parse(value: Option<&str>) -> Result<Option<Self>, BenchError> {
        match value {
            None => Ok(None),
            Some("") | Some("0") => Ok(None),
            Some("1") => Ok(Some(Self {
                limit: PARITY_DUMP_DEFAULT_LIMIT,
            })),
            Some(other) => match other.parse::<usize>() {
                Ok(limit) if limit >= 1 => Ok(Some(Self { limit })),
                _ => Err(BenchError::InvalidParityDumpEnv {
                    value: other.to_string(),
                }),
            },
        }
    }

    /// プロセス環境から [`PARITY_DUMP_ENV`] を読んで [`parse`](Self::parse)
    /// する薄いラッパー。非 UTF-8 値は診断用の固定文字列へ丸めてから渡す
    /// （生のバイト列をエラーメッセージへ持ち込まない）。
    /// [`GemmReference::compute`] が warmup 前に 1 回だけ呼ぶ。
    pub fn from_env() -> Result<Option<Self>, BenchError> {
        match std::env::var_os(PARITY_DUMP_ENV) {
            None => Ok(None),
            Some(raw) => match raw.to_str() {
                Some(s) => Self::parse(Some(s)),
                None => Err(BenchError::InvalidParityDumpEnv {
                    value: "<non-utf8>".to_string(),
                }),
            },
        }
    }
}

/// [`dump_parity_failures`] の呼び出し結果サマリ（テスト・呼び出し元双方が
/// 出力量を検証できるよう明示的に返す）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DumpOutcome {
    /// 複合判定で fail だった要素数（`ParityStats::fail_count` と同値）。
    pub fail_count: usize,
    /// 実際に出力した要素数（`limit` で切り詰められうる）。
    pub dumped: usize,
    /// `fail_count > dumped` のとき `true`（上限超過で切り詰めた）。
    pub truncated: bool,
}

/// fail 要素の index・reference・実測値を `sink` へ書き出す（イシュー
/// #1183）。[`GemmReference::verify`] が複合判定 fail 時にのみ第 2 パスと
/// して呼ぶ（`compare_elementwise` の判定結果・集計は変更しない。本関数は
/// 診断出力の追加のみを担う）。
///
/// 出力は 1 fail 要素につき 1 行（`PARITY_DUMP ...`）+ 呼び出し末尾の
/// サマリ行（`PARITY_DUMP_SUMMARY ...`）。f32 の bit パターン
/// （`ref_bits`/`actual_bits`）を併記するのは、10 進表記だけでは
/// `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.3 の「0 近傍の
/// 丸め誤差」仮説を検証するのに不足するため。
///
/// `n` は正方行列の一辺長（`row = idx / n`・`col = idx % n`）。長さ不一致は
/// [`compare_elementwise`] と同じ `BenchError::ParityLengthMismatch` を返す。
/// 書き込み失敗は `unwrap`/`expect` せず `BenchError::Io` へ変換する
/// （coding-rust.md「本番経路で panic させない」）。
///
/// `n == 0` は `BenchError::InvalidShape` として拒否する。本関数は `pub` で
/// あり、呼び出し元が現状 [`GemmReference::compute`] 経由の `n >= 1` 保証
/// 済み値に限られるとしても、`pub` 関数の契約としてその保証は成立しない
/// ため、`n == 0` かつ fail 要素を含む非空入力で `idx / n` が panic するの
/// を防ぐ（AGENTS.md「本番経路の panic 禁止」。イシュー #1183 PR #1196
/// codex-review 指摘・P1）。あわせて `actual.len() == n*n` も検証し、
/// row/col 診断値の健全性（`idx` が `n×n` 平面に収まること）を保証する
/// （[`gemm_element_count`] と同じ `checked_mul` 経路。`n*n` が
/// `usize::MAX` を超える場合は `BenchError::SizeOverflow` を返す）。
pub fn dump_parity_failures(
    actual: &[f32],
    reference: &[f32],
    n: usize,
    tol: &ScaledAbsTolerance,
    cfg: ParityDumpConfig,
    call_index: usize,
    sink: &mut dyn std::io::Write,
) -> Result<DumpOutcome, BenchError> {
    if actual.len() != reference.len() {
        return Err(BenchError::ParityLengthMismatch {
            expected: reference.len(),
            actual: actual.len(),
        });
    }
    if n == 0 {
        return Err(BenchError::InvalidShape { n });
    }
    let expected = gemm_element_count(n)?;
    if actual.len() != expected {
        return Err(BenchError::ParityLengthMismatch {
            expected,
            actual: actual.len(),
        });
    }

    let bound = tol.bound();
    let to_io_err = |source: std::io::Error| BenchError::Io {
        path: "<stderr>".to_string(),
        source,
    };

    let mut fail_count = 0usize;
    let mut dumped = 0usize;
    for (idx, (&x, &y)) in actual.iter().zip(reference.iter()).enumerate() {
        // `tol` に基づく判定（`err.pass` が救済済みならダンプ対象から
        // 除外する。ダンプ対象 ⟺ `compare_elementwise` の `fail_count` に
        // 数えた要素、という不変条件を維持する）。
        let err = element_error(x, y, bound);
        if err.pass {
            continue;
        }
        fail_count += 1;
        if dumped >= cfg.limit {
            continue;
        }
        // `n >= 1`・`actual.len() == n*n` は本関数冒頭で検証済み。
        let row = idx / n;
        let col = idx % n;
        writeln!(
            sink,
            "PARITY_DUMP call={call_index} n={n} idx={idx} row={row} col={col} \
             ref={y:e} ref_bits=0x{:08x} actual={x:e} actual_bits=0x{:08x} abs={:e} rel={:e}",
            y.to_bits(),
            x.to_bits(),
            err.abs,
            err.rel,
        )
        .map_err(to_io_err)?;
        dumped += 1;
    }

    let truncated = fail_count > dumped;
    writeln!(
        sink,
        "PARITY_DUMP_SUMMARY call={call_index} n={n} fail_count={fail_count} dumped={dumped} truncated={truncated}"
    )
    .map_err(to_io_err)?;

    Ok(DumpOutcome {
        fail_count,
        dumped,
        truncated,
    })
}

/// `--size` 由来の未検証な `n` から GEMM 入力・参照値の要素数（= `n*n`）を
/// 検証して返す。
///
/// イシュー #970 codex-review 指摘（PR #978・P1）: [`GemmReference::compute`]
/// 側にのみ `checked_mul` 検証を置いても、3 つの呼び出し元
/// （`bench-burn`/`bench-candle`/`bench-fandhe`）はいずれもその呼び出しより
/// **先に** 未検証の `n` で `n * n`（`fill_vec` へ渡す要素数）を評価していた
/// ため、debug ビルドではその時点で乗算オーバーフロー panic、release
/// ビルドでも wrap した長さでベクタを確保・使用してしまい
/// `GemmReference::compute` の検証まで到達できなかった（本番経路で panic
/// させない契約違反。`.claude/rules/coding-rust.md`）。本関数を入力ベクタ
/// 生成の**前**に呼び、検証済みの要素数を `fill_vec` と
/// [`GemmReference::compute`]（後者は本関数を内部で呼び直すだけで二重に
/// `checked_mul` しない）の双方で共有することで、検証漏れの経路を作らない。
pub fn gemm_element_count(n: usize) -> Result<usize, BenchError> {
    n.checked_mul(n).ok_or(BenchError::SizeOverflow { n })
}

/// FMA 契約の参照 GEMM（`C = A @ B`。i-k-j ループ・f32 `mul_add`・逐次 k
/// 昇順）。各 `c[i][j]` の累積鎖は k 昇順・FMA のまま
/// （`backend-cpu::parity::matmul_reference_fma` と bit 完全一致する契約）
/// であり、i（行）方向は互いに独立なので `std::thread::scope` で行ブロック
/// 分割して並列化しても各要素の演算順序・bit パターンは変わらない
/// （`tests::compute_is_bit_identical_to_sequential_k_ascending` で固定）。
pub struct GemmReference {
    c: Vec<f32>,
    /// 正方行列の一辺長（[`dump_parity_failures`] の `row`/`col` 復元に使う。
    /// イシュー #1183）。
    n: usize,
    /// [`ParityDumpConfig::from_env`] を [`compute`](Self::compute) 内で 1 回
    /// だけ読んだ結果。`None` なら [`verify`](Self::verify) は従来どおり
    /// `compare_elementwise` のみを呼び、ダンプ処理へは一切入らない
    /// （環境変数未設定時の出力・判定・終了コード不変を構造的に保証する）。
    dump: Option<ParityDumpConfig>,
    /// [`verify`](Self::verify) の呼び出し回数（ダンプ出力の `call=` に
    /// 反映。`&self` のまま数えるため `Cell` を使う）。
    verify_calls: std::cell::Cell<usize>,
    /// [`ScaledAbsTolerance::from_inputs`]（`a`・`b`・`n` から 1 回だけ導出。
    /// [`compute`](Self::compute) 内で計算する）。[`verify_with_sink`] が
    /// 毎反復これを再利用し、`compare_elementwise`／`dump_parity_failures`
    /// の第 3 救済項へ渡す。
    tol: ScaledAbsTolerance,
}

impl GemmReference {
    /// `a`・`b` は行優先 flat 表現の N×N 行列。`n == 0` または長さが
    /// `n*n` と一致しない場合は型付きエラーを返す（本番経路で panic
    /// させない。`.claude/rules/coding-rust.md`）。
    ///
    /// [`PARITY_DUMP_ENV`] を warmup 反復の前にここで 1 回だけ読み、不正な
    /// 値は起動直後に型付きエラーとして返す（20 反復後ではなく fail-fast。
    /// イシュー #1183）。
    pub fn compute(n: usize, a: &[f32], b: &[f32]) -> Result<Self, BenchError> {
        let dump = ParityDumpConfig::from_env()?;
        if n == 0 {
            return Err(BenchError::InvalidShape { n });
        }
        // `n` は `--size` 由来の未検証な CLI 入力（`parse_cli` は
        // `usize::parse` のみで上限を課さない）。呼び出し元（3 バイナリの
        // `run_gemm`/`gemm_inputs`）は入力ベクタ生成前に
        // [`gemm_element_count`] を既に呼んでいるため、ここでの再検証は
        // 同じ検証を重複させるだけだが、`GemmReference::compute` 単体でも
        // 契約が閉じるよう同じ検証済み経路（`checked_mul`）を通す
        // （coding-rust.md「本番経路で panic させない」。イシュー #970
        // codex-review 指摘・P1）。
        let expected = gemm_element_count(n)?;
        if a.len() != expected {
            return Err(BenchError::ParityLengthMismatch {
                expected,
                actual: a.len(),
            });
        }
        if b.len() != expected {
            return Err(BenchError::ParityLengthMismatch {
                expected,
                actual: b.len(),
            });
        }
        // スケール付き絶対誤差救済項（第 3 項）のスケール `S_A`・`S_B` を
        // 参照 GEMM 計算前に確定する（`a`・`b` は不変のまま以降使い回す
        // ため、`verify` の毎反復ではなくここで 1 回だけ計算する）。
        let tol = ScaledAbsTolerance::from_inputs(n, a, b);

        let mut c = vec![0.0f32; expected];
        // `available_parallelism()` は `Result<NonZeroUsize>`。失敗時は
        // 直列実行へフォールバックする（`unwrap()` しない。coding-rust.md）。
        let workers = std::thread::available_parallelism()
            .map(|w| w.get())
            .unwrap_or(1)
            .min(n);
        let rows_per_chunk = n.div_ceil(workers.max(1));

        // `rows_per_chunk * n` は `chunks_mut` のチャンク長。`rows_per_chunk
        // <= n`（`workers >= 1` なので `n.div_ceil(workers.max(1)) <= n`）
        // かつ `n * n` は上で `checked_mul` 済みのため、この乗算は必ず
        // `expected` 以下に収まりオーバーフローしない。
        //
        // `std::thread::Scope::spawn` は OS のスレッド生成に失敗すると
        // panic する（イシュー #970 codex-review 指摘・P1）。並列度分の
        // スレッドを無条件生成すると、プロセス・スレッド数上限や一時的
        // リソース不足でベンチ CLI 全体が panic しうるため、
        // `std::thread::Builder::spawn_scoped` で `io::Result` として
        // 受け取り、失敗時は型付きエラーへ変換して呼び出し元へ返す
        // （coding-rust.md「本番経路で panic させない」）。
        std::thread::scope(|scope| -> Result<(), BenchError> {
            let mut handles = Vec::with_capacity(workers.max(1));
            for (chunk_idx, c_chunk) in c.chunks_mut(rows_per_chunk * n).enumerate() {
                let row_start = chunk_idx * rows_per_chunk;
                let spawned = std::thread::Builder::new().spawn_scoped(scope, move || {
                    let rows_in_chunk = c_chunk.len() / n;
                    for local_i in 0..rows_in_chunk {
                        let i = row_start + local_i;
                        // k を中間ループに置くことで、固定 (i, j) の累積は
                        // k=0,1,...,n-1 の順に行われる（逐次 k 昇順と bit
                        // 完全一致）。j を内側に置き連続メモリアクセス・
                        // ベクトル化の余地を残す。
                        for k in 0..n {
                            let a_ik = a[i * n + k];
                            let b_row = &b[k * n..k * n + n];
                            let c_row = &mut c_chunk[local_i * n..local_i * n + n];
                            for j in 0..n {
                                c_row[j] = a_ik.mul_add(b_row[j], c_row[j]);
                            }
                        }
                    }
                });
                match spawned {
                    Ok(handle) => handles.push(handle),
                    Err(source) => {
                        // 1 本でも `spawn_scoped` が失敗したら即座に打ち切り、
                        // 型付きエラーとして呼び出し元へ返す。すでに spawn
                        // 済みのスレッドは `scope` の終了（このクロージャを
                        // 抜ける際の暗黙 join）で待ち合わされるため、ここで
                        // 早期 return しても join 漏れは起きない。
                        return Err(BenchError::ThreadSpawnFailed { source });
                    }
                }
            }
            for handle in handles {
                // スレッド内クロージャは戻り値を返さず panic もしない
                // （境界検査済みのスライス演算のみ）ため `join()` は通常
                // 失敗しない。万一 panic した場合はここで再 panic させず、
                // 参照 GEMM 全体を型付きエラーとして扱う（本番経路で
                // panic させない。coding-rust.md）。
                if handle.join().is_err() {
                    return Err(BenchError::ThreadSpawnFailed {
                        source: std::io::Error::other("gemm reference worker thread panicked"),
                    });
                }
            }
            Ok(())
        })?;

        Ok(Self {
            c,
            n,
            dump,
            verify_calls: std::cell::Cell::new(0),
            tol,
        })
    }

    /// 参照 GEMM の結果（flat・行優先）。
    pub fn as_slice(&self) -> &[f32] {
        &self.c
    }

    /// [`verify`](Self::verify)／[`verify_with_sink`](Self::verify_with_sink)
    /// が使うスケール付き絶対誤差救済項のパラメータ（[`ScaledAbsTolerance`]）。
    /// `bench-fandhe` 側テストが `compare_elementwise` を
    /// [`ScaledAbsTolerance::NONE`] で明示的に呼び直す際に、実ベンチ入力
    /// 由来の `tol` と対比するためのアクセサ。
    pub fn scaled_abs_tolerance(&self) -> ScaledAbsTolerance {
        self.tol
    }

    /// テスト用ビルダー: `dump` 設定を差し替えた同値のコピーを返す
    /// （env 非依存にダンプ経路をユニットテストするため。イシュー #1183）。
    #[cfg(test)]
    fn with_dump(&self, dump: Option<ParityDumpConfig>) -> Self {
        Self {
            c: self.c.clone(),
            n: self.n,
            dump,
            verify_calls: std::cell::Cell::new(0),
            tol: self.tol,
        }
    }

    /// [`compare_elementwise`] の薄い便宜メソッド（呼び出し側が参照値を
    /// 都度 `as_slice()` する必要をなくす）。fail 要素があり
    /// [`ParityDumpConfig`] が有効化されている場合のみ、判定結果・戻り値を
    /// 変えずに stderr へ第 2 パスのダンプを行う（イシュー #1183。
    /// `dump == None`〈環境変数未設定〉のときは本メソッドは
    /// `compare_elementwise` を呼ぶだけで従来と完全に同じ挙動になる）。
    pub fn verify(&self, out: &[f32]) -> Result<ParityStats, BenchError> {
        let mut stderr = std::io::stderr();
        self.verify_with_sink(out, &mut stderr)
    }

    /// [`verify`](Self::verify) の内部実装。`sink` を注入できるようにして
    /// ユニットテストから stderr を経由せず検証できるようにする。
    fn verify_with_sink(
        &self,
        out: &[f32],
        sink: &mut dyn std::io::Write,
    ) -> Result<ParityStats, BenchError> {
        // 診断用カウンタ（`dump` 出力の `call=` ラベルにのみ使う）。
        // `usize::MAX` 到達は現実的な反復回数では起こり得ないが、
        // `+ 1` は overflow-checks 有効ビルドで本番経路 panic になりうる
        // ため（本体規約「本番経路で panic させない」。`.claude/rules/coding-rust.md`）、
        // `saturating_add` で飽和させる。診断ラベルが `usize::MAX` に
        // 貼り付いても判定結果（`stats`）には一切影響しない
        // （codex-review 指摘・イシュー #1183 PR #1196 P1）。
        let call_index = self.verify_calls.get().saturating_add(1);
        self.verify_calls.set(call_index);

        let stats = compare_elementwise(out, &self.c, &self.tol)?;

        if let Some(cfg) = self.dump
            && stats.fail_count > 0
        {
            dump_parity_failures(out, &self.c, self.n, &self.tol, cfg, call_index, sink)?;
        }

        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Xorshift64Star;

    /// `GemmReference::compute` と同じ演算順序（i-k-j・逐次 k 昇順・
    /// `mul_add`）の単純直列実装。並列版との bit 完全一致確認専用
    /// （本体を再利用すると自明な同語反復になるため独立実装する）。
    fn sequential_reference(n: usize, a: &[f32], b: &[f32]) -> Vec<f32> {
        let mut c = vec![0.0f32; n * n];
        for i in 0..n {
            for k in 0..n {
                let a_ik = a[i * n + k];
                for j in 0..n {
                    let idx = i * n + j;
                    c[idx] = a_ik.mul_add(b[k * n + j], c[idx]);
                }
            }
        }
        c
    }

    #[test]
    fn compute_matches_hand_computed_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] => C = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let r = GemmReference::compute(2, &a, &b).expect("compute");
        assert_eq!(r.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn compute_matches_hand_computed_3x3_identity() {
        let a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let r = GemmReference::compute(3, &a, &b).expect("compute");
        assert_eq!(r.as_slice(), b.as_slice());
    }

    #[test]
    fn compute_n1() {
        let r = GemmReference::compute(1, &[3.0], &[4.0]).expect("compute");
        assert_eq!(r.as_slice(), &[12.0]);
    }

    #[test]
    fn compute_zero_n_is_invalid_shape() {
        assert!(matches!(
            GemmReference::compute(0, &[], &[]),
            Err(BenchError::InvalidShape { n: 0 })
        ));
    }

    #[test]
    fn compute_length_mismatch_is_typed_error() {
        assert!(matches!(
            GemmReference::compute(2, &[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0, 4.0]),
            Err(BenchError::ParityLengthMismatch { .. })
        ));
        assert!(matches!(
            GemmReference::compute(2, &[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0]),
            Err(BenchError::ParityLengthMismatch { .. })
        ));
    }

    /// イシュー #970 codex-review 指摘（P1）の回帰テスト: `--size` 由来の
    /// 未検証な `n` に対し `n * n` が `usize` をオーバーフローする場合、
    /// panic（debug ビルドの overflow-checks・release ビルドの
    /// `chunks_mut(0)`）ではなく型付きエラーを返すことを確認する。
    #[test]
    fn compute_size_overflow_is_typed_error_not_panic() {
        let n = (usize::MAX / 2) + 2; // n*n は usize::MAX を必ず超える
        assert!(matches!(
            GemmReference::compute(n, &[], &[]),
            Err(BenchError::SizeOverflow { n: got }) if got == n
        ));
    }

    /// イシュー #970 codex-review 指摘（PR #978・P1）の回帰テスト:
    /// [`gemm_element_count`] は呼び出し元（3 バイナリの `run_gemm`/
    /// `gemm_inputs`）が入力ベクタ生成前に呼ぶ共通関数。巨大な `n` に対し
    /// panic せず `SizeOverflow` を返すことを、`GemmReference::compute` と
    /// 独立に確認する（入力ベクタ生成前の検証経路が抜け落ちていないかの
    /// 直接的な回帰テスト）。
    #[test]
    fn gemm_element_count_overflow_is_typed_error_not_panic() {
        let n = (usize::MAX / 2) + 2; // n*n は usize::MAX を必ず超える
        assert!(matches!(
            gemm_element_count(n),
            Err(BenchError::SizeOverflow { n: got }) if got == n
        ));
    }

    /// 非オーバーフロー時は `n*n` をそのまま返す（正常系の疎通確認）。
    #[test]
    fn gemm_element_count_normal_case() {
        assert_eq!(gemm_element_count(8).expect("no overflow"), 64);
    }

    /// n=17 は典型的なスレッド数（2/4/8/12 等）で割り切れず、行ブロック
    /// 端の処理（`chunks_mut`・`rows_in_chunk` の端数）を確実に経由する。
    #[test]
    fn compute_is_bit_identical_to_sequential_k_ascending() {
        let n = 17;
        // SEED_A/SEED_B とは異なるシードで独立性を保つ。
        let a = Xorshift64Star::new(0xF00D_0001).fill_vec(n * n);
        let b = Xorshift64Star::new(0xF00D_0002).fill_vec(n * n);
        let par = GemmReference::compute(n, &a, &b).expect("compute");
        let seq = sequential_reference(n, &a, &b);
        for (p, s) in par.as_slice().iter().zip(seq.iter()) {
            assert_eq!(p.to_bits(), s.to_bits());
        }
    }

    #[test]
    fn compare_elementwise_identical_is_all_pass() {
        let v = vec![1.0f32, -2.5, 0.0, 100.0];
        let stats = compare_elementwise(&v, &v, &ScaledAbsTolerance::NONE).expect("compare");
        assert_eq!(stats.fail_count, 0);
        assert_eq!(stats.total, v.len());
        assert_eq!(stats.max_abs_err, 0.0);
        assert_eq!(stats.max_rel_err, 0.0);
    }

    #[test]
    fn compare_elementwise_detects_one_percent_perturbation() {
        let reference = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut actual = reference.clone();
        actual[2] *= 1.01; // 1% 相対誤差。PARITY_REL_TOL(1e-3) を超える。
        let stats =
            compare_elementwise(&actual, &reference, &ScaledAbsTolerance::NONE).expect("compare");
        assert_eq!(stats.fail_count, 1);
        assert!(stats.max_rel_err > PARITY_REL_TOL);
        assert!((stats.max_abs_err - 0.03).abs() < 1e-6);
    }

    #[test]
    fn compare_elementwise_near_zero_diff_is_rescued_by_absolute_threshold() {
        // 相対誤差は跳ね上がるが絶対誤差 1e-7 < PARITY_ABS_TOL(1e-5) で救済。
        let reference = vec![1e-7f32];
        let actual = vec![2e-7f32];
        let stats =
            compare_elementwise(&actual, &reference, &ScaledAbsTolerance::NONE).expect("compare");
        assert_eq!(stats.fail_count, 0);
    }

    #[test]
    fn compare_elementwise_nan_is_fail_with_infinite_max_abs_err() {
        let reference = vec![1.0f32];
        let actual = vec![f32::NAN];
        let stats =
            compare_elementwise(&actual, &reference, &ScaledAbsTolerance::NONE).expect("compare");
        assert_eq!(stats.fail_count, 1);
        assert_eq!(stats.max_abs_err, f64::INFINITY);
        assert_eq!(stats.max_rel_err, f64::INFINITY);
    }

    #[test]
    fn compare_elementwise_length_mismatch_is_typed_error() {
        assert!(matches!(
            compare_elementwise(&[1.0, 2.0], &[1.0], &ScaledAbsTolerance::NONE),
            Err(BenchError::ParityLengthMismatch {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn worst_takes_max_of_each_field_independently() {
        let a = ParityStats {
            total: 100,
            fail_count: 0,
            max_abs_err: 1e-6,
            max_rel_err: 1e-7,
            scaled_abs_bound: 1e-9,
            scaled_abs_rescued: 0,
        };
        let b = ParityStats {
            total: 100,
            fail_count: 2,
            max_abs_err: 1e-8,
            max_rel_err: 1e-4,
            scaled_abs_bound: 1e-6,
            scaled_abs_rescued: 3,
        };
        let w = a.worst(b);
        assert_eq!(w.fail_count, 2);
        assert!((w.max_abs_err - 1e-6).abs() < 1e-12);
        assert!((w.max_rel_err - 1e-4).abs() < 1e-12);
        assert!((w.scaled_abs_bound - 1e-6).abs() < 1e-18);
        assert_eq!(w.scaled_abs_rescued, 3);
    }

    #[test]
    fn worst_propagates_infinite_sentinel() {
        let ok = ParityStats {
            total: 10,
            fail_count: 0,
            max_abs_err: 1e-6,
            max_rel_err: 1e-6,
            scaled_abs_bound: 1e-9,
            scaled_abs_rescued: 0,
        };
        let broken = ParityStats {
            total: 10,
            fail_count: 1,
            max_abs_err: f64::INFINITY,
            max_rel_err: f64::INFINITY,
            scaled_abs_bound: f64::INFINITY,
            scaled_abs_rescued: 0,
        };
        let w = ok.worst(broken);
        assert_eq!(w.max_abs_err, f64::INFINITY);
        assert_eq!(w.max_rel_err, f64::INFINITY);
        assert_eq!(w.scaled_abs_bound, f64::INFINITY);
    }

    // --- スケール付き絶対誤差救済項（候補 A-1・`c=0.5`・`u=2^-24`。
    // イシュー #1247・承認記録: #1241 承認コメント 2026-09-08・
    // `docs/candle-parity-tolerance-contract-decision.md` §8）------------

    /// 承認済み係数がハーネス限定の値のまま固定されていることのピン止め
    /// （本体には対応定数が存在しない設計。値の変更はユーザー承認必須）。
    #[test]
    fn scaled_abs_constants_pinned() {
        assert_eq!(PARITY_SCALED_ABS_COEFF, 0.5);
        // u = 2^-24 は f32::EPSILON（2^-23）の半分と同値（別導出との
        // 一致確認。#1238 §7.1 で盲点になった定数ピンの死角を、後続の
        // `scaled_abs_bound_formula_pinned` で判定式自体もピン止めする
        // ことで塞ぐ）。
        assert_eq!(F32_UNIT_ROUNDOFF, (f32::EPSILON as f64) / 2.0);
    }

    /// 判定式 `bound = c・u・K・S_A・S_B` 自体をピン止めする
    /// （`docs/candle-parity-tolerance-contract-decision.md` §9 で
    /// 「定数ピンは判定式を検査しない」と指摘された #1238 §7.1 の盲点への
    /// 対応）。3 つの `K` はいずれも 2 のべき乗の積のみで構成されるため
    /// f64 演算が丸め誤差なく厳密値と bit 一致する（`assert_eq!` で確認
    /// できる。K=2048/512/4096・`S_A・S_B=0.25` はいずれも 2 のべき乗）。
    #[test]
    fn scaled_abs_bound_formula_pinned() {
        let tol_2048 = ScaledAbsTolerance {
            k: 2048,
            scale_a: 0.5,
            scale_b: 0.5,
        };
        assert_eq!(tol_2048.bound(), 2f64.powi(-16)); // 1.52587890625e-5

        let tol_512 = ScaledAbsTolerance {
            k: 512,
            scale_a: 0.5,
            scale_b: 0.5,
        };
        assert_eq!(tol_512.bound(), 2f64.powi(-18)); // 3.814697265625e-6

        let tol_4096 = ScaledAbsTolerance {
            k: 4096,
            scale_a: 0.5,
            scale_b: 0.5,
        };
        assert_eq!(tol_4096.bound(), 2f64.powi(-15));
    }

    /// 境界は `<=`（包含的）。`diff == bound` ちょうどは pass、1 ulp 超えると
    /// fail に戻る。かつ既存 2 条件（[`PARITY_REL_TOL`]/[`PARITY_ABS_TOL`]）
    /// は本項の適用有無に関わらず不変であることを [`ScaledAbsTolerance::NONE`]
    /// との比較で確認する。
    #[test]
    fn scaled_abs_boundary_is_inclusive_and_legacy_conditions_unchanged() {
        // reference=0.0・actual=2^-16 は rel=1.0（跳ね上がり）・
        // diff=2^-16 > PARITY_ABS_TOL(1e-5) のため既存 2 条件は fail。
        let reference = 0.0f32;
        let bound = 2f64.powi(-16);
        let actual_at_bound = bound as f32;
        assert_eq!((actual_at_bound as f64 - reference as f64).abs(), bound);

        let tol = ScaledAbsTolerance {
            k: 2048,
            scale_a: 0.5,
            scale_b: 0.5,
        };
        assert_eq!(tol.bound(), bound);

        let err_at_bound = element_error(actual_at_bound, reference, tol.bound());
        assert!(!err_at_bound.abs.is_nan());
        assert!(
            err_at_bound.pass,
            "diff == bound はちょうど境界で pass する契約（`<=`）"
        );
        assert!(err_at_bound.rescued_by_scaled_abs);

        // 1 ulp 超えると fail に戻る。
        let actual_over_bound = f32::from_bits(actual_at_bound.to_bits() + 1);
        let err_over_bound = element_error(actual_over_bound, reference, tol.bound());
        assert!(!err_over_bound.pass);
        assert!(!err_over_bound.rescued_by_scaled_abs);

        // `NONE`（bound=0.0）では第 3 項が無効化され、既存 2 条件のみの
        // 判定（レガシー同値）になる。
        let legacy = element_error(actual_at_bound, reference, ScaledAbsTolerance::NONE.bound());
        assert!(!legacy.pass);
        assert!(!legacy.rescued_by_scaled_abs);
    }

    /// `scale_a`／`scale_b` に非有限値が混入していた場合、`bound()` は
    /// fail-closed に `0.0`（救済なし）へ倒れる（`diff <= +Inf` による
    /// 全要素救済という事故を構造的に防ぐ。A08）。
    #[test]
    fn scaled_abs_bound_is_fail_closed_on_nonfinite_scale() {
        let nan_scale = ScaledAbsTolerance {
            k: 2048,
            scale_a: f64::NAN,
            scale_b: 0.5,
        };
        assert_eq!(nan_scale.bound(), 0.0);

        let inf_scale = ScaledAbsTolerance {
            k: 2048,
            scale_a: f64::INFINITY,
            scale_b: 0.5,
        };
        assert_eq!(inf_scale.bound(), 0.0);

        // `from_inputs` は NaN/Inf 要素を検出すると `f64::INFINITY`
        // センチネルへ固定する（`max_abs_f64`/`worst_f64` 経由）。
        let tol = ScaledAbsTolerance::from_inputs(2, &[1.0, f32::NAN], &[1.0, 2.0]);
        assert_eq!(tol.scale_a, f64::INFINITY);
        assert_eq!(tol.bound(), 0.0);

        // NaN diff を含む要素は bound が大きくても救済されず fail のまま
        // （`element_error` の `<=` 比較は NaN に対し常に false）。
        let err = element_error(f32::NAN, 1.0, f64::INFINITY);
        assert!(!err.pass);
    }

    /// `docs/perf/logs/cuda-gemm-candle-parity-1184/` の N=2048 candle
    /// 実測 fail 要素（CUDA・CPU 各 2 要素）を bit パターンで固定し、
    /// 実ベンチ入力（`Xorshift64Star::new(SEED_A/SEED_B).fill_vec(2048*2048)`。
    /// `bench-fandhe::main::{SEED_A,SEED_B}` と同一シード）由来の
    /// `ScaledAbsTolerance` で候補要素が救済されることを固定する
    /// （承認済み契約の主目的の直接確認。イシュー #1184・#1247）。
    #[test]
    fn candidate_elements_from_1184_dump_pass_under_scaled_abs() {
        // `crate::SEED_A`/`crate::SEED_B`（`bench-fandhe`/`bench-candle`
        // 等が実際に使う xorshift64* シード。`lib.rs` 参照）と同一の値で
        // `S_A`・`S_B` を導出し、実ベンチ入力条件と一致させる。
        let n = 2048usize;
        let a = crate::Xorshift64Star::new(crate::SEED_A).fill_vec(n * n);
        let b = crate::Xorshift64Star::new(crate::SEED_B).fill_vec(n * n);
        let tol = ScaledAbsTolerance::from_inputs(n, &a, &b);

        // (ref_bits, actual_bits) の 4 要素（CUDA 2・CPU 2。
        // `docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-*-2048.txt`
        // から転記）。
        let candidates: [(u32, u32); 4] = [
            (0x3b0e24b9, 0x3b0d6800), // CUDA idx=13850
            (0x3c1689b0, 0x3c16b800), // CUDA idx=4130484
            (0x3c2289f5, 0x3c22bf00), // CPU idx=1372466
            (0x3bb0187b, 0x3bb07a00), // CPU idx=1633751
        ];
        let refs: Vec<f32> = candidates.iter().map(|&(r, _)| f32::from_bits(r)).collect();
        let actuals: Vec<f32> = candidates.iter().map(|&(_, a)| f32::from_bits(a)).collect();

        // 承認済み判定式（`tol` は実ベンチ入力由来）では 0 fail・4 要素
        // 救済。
        let stats = compare_elementwise(&actuals, &refs, &tol).expect("compare");
        assert_eq!(stats.fail_count, 0);
        assert_eq!(stats.scaled_abs_rescued, 4);

        // `NONE`（既存 2 条件のみ）では引き続き 4 要素とも fail する
        // （#1184 実測どおり。既存契約は不変であることの確認）。
        let legacy =
            compare_elementwise(&actuals, &refs, &ScaledAbsTolerance::NONE).expect("compare");
        assert_eq!(legacy.fail_count, 4);
        assert_eq!(legacy.scaled_abs_rescued, 0);
    }

    /// 同じ 4 要素を `K=512`（`bound = 3.814697265625e-6 < PARITY_ABS_TOL`
    /// 未満で実質 no-op）で比較すると救済されず fail のままであることを
    /// 確認する（`K` 依存の確認。線形 K 形式であることの直接的な検証）。
    #[test]
    fn non_candidate_elements_still_fail_with_small_k() {
        let tol = ScaledAbsTolerance {
            k: 512,
            scale_a: 0.5,
            scale_b: 0.5,
        };
        assert!(tol.bound() < PARITY_ABS_TOL);

        let refs = [
            f32::from_bits(0x3b0e24b9),
            f32::from_bits(0x3c1689b0),
            f32::from_bits(0x3c2289f5),
            f32::from_bits(0x3bb0187b),
        ];
        let actuals = [
            f32::from_bits(0x3b0d6800),
            f32::from_bits(0x3c16b800),
            f32::from_bits(0x3c22bf00),
            f32::from_bits(0x3bb07a00),
        ];
        let stats = compare_elementwise(&actuals, &refs, &tol).expect("compare");
        assert_eq!(stats.fail_count, 4);
        assert_eq!(stats.scaled_abs_rescued, 0);

        // 既存の 1% 摂動テスト（diff=0.03）は `K=2048` の bound
        // （1.52587890625e-5）よりはるかに大きいため、引き続き fail する
        // ことも合わせて確認する（大きな摂動が救済されないことの確認）。
        let reference = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut actual = reference.clone();
        actual[2] *= 1.01;
        let big_tol = ScaledAbsTolerance {
            k: 2048,
            scale_a: 0.5,
            scale_b: 0.5,
        };
        let stats2 = compare_elementwise(&actual, &reference, &big_tol).expect("compare");
        assert_eq!(stats2.fail_count, 1);
    }

    /// 救済された要素は `dump_parity_failures` の出力対象から除外される
    /// （ダンプ対象 ⟺ `compare_elementwise` の `fail_count` に数えた要素、
    /// という不変条件の確認。イシュー #1183 の既存契約を第 3 項導入後も
    /// 維持する）。
    #[test]
    fn dump_parity_failures_skips_rescued_elements() {
        let n = 2;
        // idx=0 は大きく摂動させ救済されない fail、idx=1..3 は 0 埋めで
        // pass、idx=3 を第 3 項でのみ救済される値にする。
        let reference = vec![1.0f32, 0.0, 0.0, 0.0f32];
        let bound = 2f64.powi(-10);
        let rescued_actual = bound as f32;
        let actual = vec![2.0f32, 0.0, 0.0, rescued_actual];

        // `k・c・u = bound` から `k` を逆算し、`tol.bound()` が `bound`
        // とほぼ一致する `ScaledAbsTolerance` を組む（scale は 1.0 固定）。
        let k_for_bound = (bound / (PARITY_SCALED_ABS_COEFF * F32_UNIT_ROUNDOFF)) as usize;
        let tol = ScaledAbsTolerance {
            k: k_for_bound,
            scale_a: 1.0,
            scale_b: 1.0,
        };
        assert!((tol.bound() - bound).abs() / bound < 1e-9);

        let stats = compare_elementwise(&actual, &reference, &tol).expect("compare");
        assert_eq!(stats.fail_count, 1);
        assert_eq!(stats.scaled_abs_rescued, 1);

        let mut sink = Vec::new();
        let outcome = dump_parity_failures(
            &actual,
            &reference,
            n,
            &tol,
            ParityDumpConfig { limit: 64 },
            1,
            &mut sink,
        )
        .expect("dump");

        // ダンプ対象は `fail_count`（救済後）と一致する 1 件のみ。
        assert_eq!(outcome.fail_count, 1);
        assert_eq!(outcome.dumped, 1);
        let text = String::from_utf8(sink).expect("utf8");
        let dump_lines: Vec<&str> = text
            .lines()
            .filter(|l| l.starts_with("PARITY_DUMP call="))
            .collect();
        assert_eq!(dump_lines.len(), 1);
        assert!(dump_lines[0].contains("idx=0"));
        assert!(
            !dump_lines.iter().any(|l| l.contains("idx=3")),
            "救済された要素（idx=3）はダンプされない"
        );
    }

    #[test]
    fn gemm_reference_verify_roundtrip_passes() {
        let n = 8;
        let a = Xorshift64Star::new(0xABCD_0001).fill_vec(n * n);
        let b = Xorshift64Star::new(0xABCD_0002).fill_vec(n * n);
        let reference = GemmReference::compute(n, &a, &b).expect("compute");
        let out = reference.as_slice().to_vec();
        let stats = reference.verify(&out).expect("verify");
        assert_eq!(stats.fail_count, 0);
    }

    // --- fail 要素ダンプ計装（イシュー #1183）------------------------------
    //
    // `std::env::set_var` を使わず（edition 2024 で `unsafe`・プロセス環境の
    // 書き換えはテスト間の順序依存を生む）、`ParityDumpConfig::parse` と
    // `GemmReference::with_dump`/`verify_with_sink` の純関数経路のみで
    // 検証する。

    #[test]
    fn parity_dump_config_parse_allowlist() {
        // 未設定・空文字・"0" は無効（既存挙動＝不変）。
        assert_eq!(ParityDumpConfig::parse(None).expect("parse"), None);
        assert_eq!(ParityDumpConfig::parse(Some("")).expect("parse"), None);
        assert_eq!(ParityDumpConfig::parse(Some("0")).expect("parse"), None);

        // "1" は既定上限。
        assert_eq!(
            ParityDumpConfig::parse(Some("1")).expect("parse"),
            Some(ParityDumpConfig {
                limit: PARITY_DUMP_DEFAULT_LIMIT
            })
        );

        // 正の整数はその値を上限にする。
        assert_eq!(
            ParityDumpConfig::parse(Some("16")).expect("parse"),
            Some(ParityDumpConfig { limit: 16 })
        );
    }

    #[test]
    fn parity_dump_config_parse_rejects_invalid_values() {
        for bad in ["abc", "-1", "0x10", " 1", "1.5"] {
            let err =
                ParityDumpConfig::parse(Some(bad)).expect_err("must be rejected by the allowlist");
            assert!(matches!(err, BenchError::InvalidParityDumpEnv { .. }));
            // MEASURE_ERROR: prefix を持つこと（既存の InvalidMode 等と同型の
            // fail-fast 診断メッセージであることの確認）。
            assert!(err.to_string().starts_with("MEASURE_ERROR:"));
        }
    }

    #[test]
    fn dump_parity_failures_reports_index_row_col_and_values() {
        // 既存の 1% 摂動 fixture（`compare_elementwise_detects_one_percent_perturbation`
        // と同じ入力）を 2x2 として扱い idx=2 が row=1,col=0 になることを確認する。
        let reference = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut actual = reference.clone();
        actual[2] *= 1.01;

        let mut sink = Vec::new();
        let outcome = dump_parity_failures(
            &actual,
            &reference,
            2,
            &ScaledAbsTolerance::NONE,
            ParityDumpConfig { limit: 64 },
            1,
            &mut sink,
        )
        .expect("dump");

        assert_eq!(outcome.fail_count, 1);
        assert_eq!(outcome.dumped, 1);
        assert!(!outcome.truncated);

        let text = String::from_utf8(sink).expect("utf8");
        let dump_line = text
            .lines()
            .find(|l| l.starts_with("PARITY_DUMP call="))
            .expect("dump line present");
        assert!(dump_line.contains("idx=2"));
        assert!(dump_line.contains("row=1"));
        assert!(dump_line.contains("col=0"));
        assert!(dump_line.contains(&format!("ref_bits=0x{:08x}", 3.0f32.to_bits())));
        assert!(dump_line.contains(&format!("actual_bits=0x{:08x}", actual[2].to_bits())));

        let summary_line = text
            .lines()
            .find(|l| l.starts_with("PARITY_DUMP_SUMMARY"))
            .expect("summary line present");
        assert!(summary_line.contains("fail_count=1"));
        assert!(summary_line.contains("dumped=1"));
        assert!(summary_line.contains("truncated=false"));
    }

    #[test]
    fn dump_parity_failures_dumped_matches_fail_count_within_limit() {
        let n = 6;
        let reference = Xorshift64Star::new(0x0BAD_0001).fill_vec(n * n);
        let mut actual = reference.clone();
        // ランダムに複数要素を大きく摂動させ、fail_count と dumped が一致
        // することを確認する（上限内）。
        for idx in [0usize, 5, 11, 20] {
            actual[idx] *= 1.5;
        }

        let stats =
            compare_elementwise(&actual, &reference, &ScaledAbsTolerance::NONE).expect("compare");
        let mut sink = Vec::new();
        let outcome = dump_parity_failures(
            &actual,
            &reference,
            n,
            &ScaledAbsTolerance::NONE,
            ParityDumpConfig { limit: 64 },
            1,
            &mut sink,
        )
        .expect("dump");

        assert_eq!(outcome.fail_count, stats.fail_count);
        assert_eq!(outcome.dumped, stats.fail_count);
        assert!(!outcome.truncated);
    }

    #[test]
    fn dump_parity_failures_truncates_at_limit() {
        // n*n = 9 要素（3x3）。`actual.len() == n*n` 検証（イシュー #1183
        // PR #1196 codex-review 指摘・P1）と整合させるため、要素数を n=3 の
        // 正方行列として揃える。
        let reference = vec![1.0f32; 9];
        let mut actual = reference.clone();
        for v in actual.iter_mut() {
            *v *= 2.0; // 全要素 fail させる。
        }

        let mut sink = Vec::new();
        let outcome = dump_parity_failures(
            &actual,
            &reference,
            3,
            &ScaledAbsTolerance::NONE,
            ParityDumpConfig { limit: 2 },
            1,
            &mut sink,
        )
        .expect("dump");

        assert_eq!(outcome.fail_count, 9);
        assert_eq!(outcome.dumped, 2);
        assert!(outcome.truncated);

        let text = String::from_utf8(sink).expect("utf8");
        let dump_lines = text
            .lines()
            .filter(|l| l.starts_with("PARITY_DUMP call="))
            .count();
        assert_eq!(dump_lines, 2);
        let summary_line = text
            .lines()
            .find(|l| l.starts_with("PARITY_DUMP_SUMMARY"))
            .expect("summary line present");
        assert!(summary_line.contains("truncated=true"));
    }

    #[test]
    fn dump_parity_failures_handles_nan_without_panicking() {
        let reference = vec![1.0f32];
        let actual = vec![f32::NAN];

        let mut sink = Vec::new();
        let outcome = dump_parity_failures(
            &actual,
            &reference,
            1,
            &ScaledAbsTolerance::NONE,
            ParityDumpConfig { limit: 64 },
            1,
            &mut sink,
        )
        .expect("dump must not panic on NaN");

        assert_eq!(outcome.fail_count, 1);
        assert_eq!(outcome.dumped, 1);
        let text = String::from_utf8(sink).expect("utf8");
        assert!(text.contains("abs=inf") || text.contains("abs=NaN"));
    }

    /// イシュー #1183 PR #1196 codex-review 指摘（P1）の回帰確認: `pub` 関数
    /// である `dump_parity_failures` は `n == 0` かつ fail 要素を含む
    /// 非空入力を渡されても `idx / n` で panic せず、型付きエラー
    /// `BenchError::InvalidShape` を返す。
    #[test]
    fn dump_parity_failures_rejects_zero_n_without_panicking() {
        let reference = vec![1.0f32];
        let actual = vec![2.0f32]; // fail 要素（reference と不一致）を含む非空入力。

        let mut sink = Vec::new();
        let err = dump_parity_failures(
            &actual,
            &reference,
            0,
            &ScaledAbsTolerance::NONE,
            ParityDumpConfig { limit: 64 },
            1,
            &mut sink,
        )
        .expect_err("n == 0 must be rejected as a typed error, not panic");

        assert!(matches!(err, BenchError::InvalidShape { n: 0 }));
        assert!(sink.is_empty());
    }

    /// `actual.len() != n*n` の不整合（`n` と実際の要素数が矛盾する呼び出し
    /// 誤り）も型付きエラーで拒否し、`idx` が `n×n` 平面をはみ出した
    /// row/col 診断値を書き出さないことを確認する。
    #[test]
    fn dump_parity_failures_rejects_length_inconsistent_with_n() {
        let reference = vec![1.0f32, 2.0, 3.0, 4.0]; // 4 要素（n=2 相当）。
        let actual = reference.clone();

        let mut sink = Vec::new();
        let err = dump_parity_failures(
            &actual,
            &reference,
            3, // n=3 なら n*n=9 で 4 要素と矛盾する。
            &ScaledAbsTolerance::NONE,
            ParityDumpConfig { limit: 64 },
            1,
            &mut sink,
        )
        .expect_err("length inconsistent with n must be rejected as a typed error");

        assert!(matches!(
            err,
            BenchError::ParityLengthMismatch {
                expected: 9,
                actual: 4
            }
        ));
        assert!(sink.is_empty());
    }

    #[test]
    fn gemm_reference_verify_without_dump_matches_compare_elementwise_and_writes_nothing() {
        // 受け入れ条件 1（環境変数未設定時は既存の判定結果が完全に不変で
        // あること）の構造的確認: `dump == None` のとき `verify_with_sink`
        // は `compare_elementwise` と同一の `ParityStats` を返し、sink には
        // 何も書かれない。
        let n = 4;
        let a = Xorshift64Star::new(0x1234_0001).fill_vec(n * n);
        let b = Xorshift64Star::new(0x1234_0002).fill_vec(n * n);
        let reference = GemmReference::compute(n, &a, &b).expect("compute");
        let mut out = reference.as_slice().to_vec();
        out[0] *= 1.5; // fail 要素を作る。

        // `verify_with_sink` は `self.tol`（実ベンチ入力 a/b から導出した
        // `ScaledAbsTolerance`）を使うため、比較対象の `expected` も同じ
        // `tol` で計算する（`NONE` にすると別の判定条件になり同一性を
        // 検証したことにならない）。
        let expected = compare_elementwise(
            &out,
            reference.as_slice(),
            &reference.scaled_abs_tolerance(),
        )
        .expect("compare");

        let no_dump = reference.with_dump(None);
        let mut sink = Vec::new();
        let stats = no_dump
            .verify_with_sink(&out, &mut sink)
            .expect("verify_with_sink");

        assert_eq!(stats, expected);
        assert!(
            sink.is_empty(),
            "sink must stay empty when dump is disabled"
        );
    }

    #[test]
    fn gemm_reference_verify_with_dump_writes_nothing_on_pass() {
        // 成功時（fail_count == 0）は dump が有効でも無出力（不要なログを
        // 出さない）。
        let n = 4;
        let a = Xorshift64Star::new(0x5678_0001).fill_vec(n * n);
        let b = Xorshift64Star::new(0x5678_0002).fill_vec(n * n);
        let reference = GemmReference::compute(n, &a, &b).expect("compute");
        let out = reference.as_slice().to_vec();

        let with_dump = reference.with_dump(Some(ParityDumpConfig { limit: 64 }));
        let mut sink = Vec::new();
        let stats = with_dump
            .verify_with_sink(&out, &mut sink)
            .expect("verify_with_sink");

        assert_eq!(stats.fail_count, 0);
        assert!(sink.is_empty(), "no dump expected when all elements pass");
    }

    #[test]
    fn gemm_reference_verify_with_dump_increments_call_index() {
        let n = 2;
        let a = Xorshift64Star::new(0x9999_0001).fill_vec(n * n);
        let b = Xorshift64Star::new(0x9999_0002).fill_vec(n * n);
        let reference = GemmReference::compute(n, &a, &b).expect("compute");
        let mut out = reference.as_slice().to_vec();
        out[0] *= 2.0;

        let with_dump = reference.with_dump(Some(ParityDumpConfig { limit: 64 }));

        let mut sink1 = Vec::new();
        with_dump
            .verify_with_sink(&out, &mut sink1)
            .expect("verify 1");
        let text1 = String::from_utf8(sink1).expect("utf8");
        assert!(text1.contains("call=1"));

        let mut sink2 = Vec::new();
        with_dump
            .verify_with_sink(&out, &mut sink2)
            .expect("verify 2");
        let text2 = String::from_utf8(sink2).expect("utf8");
        assert!(text2.contains("call=2"));
    }

    /// [`PARITY_REL_TOL`]/[`PARITY_ABS_TOL`] は本体 `backend-cpu::parity`
    /// の値を独立 workspace の制約下で再定義したもの（両定数のドキュメント
    /// コメント参照）。本体側だけが変更されるとここが静かに乖離し、ベンチの
    /// 合否判定・レポート表示（`summarize.py` の `CHECKSUM_*`/`PARITY_*`
    /// 経由）が本体の承認済み契約と食い違う（イシュー #970 codex-review
    /// 指摘・PR #978 P1）。単一の正へ集約できない（deps-policy.md 第 9
    /// 区分により本体クレートへ path 依存できない）ため、代わりに本体
    /// ソースを直接読んで数値を機械照合し、乖離を fail-closed（パース失敗
    /// も含め test failure）で検知する。
    #[test]
    fn parity_tolerances_match_backend_cpu_contract() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        // bench-common (scripts/bench/framework-compare/bench-common) から
        // リポジトリルートまでは 4 階層上（framework-compare → bench →
        // scripts → root）。
        let backend_cpu_parity_path =
            std::path::Path::new(manifest_dir).join("../../../../crates/backend-cpu/src/parity.rs");
        let source = std::fs::read_to_string(&backend_cpu_parity_path).unwrap_or_else(|err| {
            panic!(
                "本体 parity.rs を読めない（{}）: {err}。\
                 パスがずれていないか確認すること",
                backend_cpu_parity_path.display()
            )
        });

        let rel_tol = extract_f64_const(&source, "RELATIVE_TOLERANCE");
        let abs_tol = extract_f64_const(&source, "ABSOLUTE_RESCUE_THRESHOLD");

        assert_eq!(
            rel_tol, PARITY_REL_TOL,
            "PARITY_REL_TOL が backend-cpu::parity::RELATIVE_TOLERANCE から乖離している。\
             閾値の変更はユーザー承認必須（.claude/rules/coding-rust.md）"
        );
        assert_eq!(
            abs_tol, PARITY_ABS_TOL,
            "PARITY_ABS_TOL が backend-cpu::parity::ABSOLUTE_RESCUE_THRESHOLD から乖離している。\
             閾値の変更はユーザー承認必須（.claude/rules/coding-rust.md）"
        );
    }

    /// `pub const <name>: f64 = <value>;` 形式の宣言から数値を取り出す。
    /// 本体 `crates/backend-cpu/src/parity.rs` の宣言スタイル固定を前提に
    /// した簡易パーサー（正規表現クレートを追加しないため文字列走査で
    /// 十分。宣言が見つからない・数値化できない場合は fail-closed に
    /// panic する）。
    fn extract_f64_const(source: &str, name: &str) -> f64 {
        let needle = format!("pub const {name}: f64 = ");
        let start = source.find(&needle).unwrap_or_else(|| {
            panic!(
                "本体 parity.rs に `{needle}` の宣言が見つからない（宣言スタイルが変わった可能性）"
            )
        });
        let rest = &source[start + needle.len()..];
        let end = rest.find(';').unwrap_or_else(|| {
            panic!("本体 parity.rs の `{name}` 宣言に終端の `;` が見つからない")
        });
        rest[..end].trim().parse::<f64>().unwrap_or_else(|err| {
            panic!("本体 parity.rs の `{name}` 宣言値を f64 として解釈できない: {err}")
        })
    }
}
