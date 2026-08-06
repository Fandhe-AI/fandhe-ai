//! バックエンド間数値一致の統一複合判定ユーティリティ・FMA 契約参照点（TASK-2.2a・#53）。
//!
//! `backend-cpu` は「無条件で有効化される数値一致の参照点」（`crate` ルートの
//! ドキュメント参照）であり、本モジュールはその責務を 2 点に具体化する。
//!
//! 1. **複合判定** ([`compare`]・[`assert_parity`]): REQ-2 が定める統一複合判定
//!    「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」の唯一の実体。#54（CPU-CUDA
//!    ペア）・#55（CPU-Metal ペア）はいずれも本関数を呼んで判定する想定であり、
//!    `&[f32]` のみを引数に取るバックエンド非依存インターフェースとすることで
//!    全ペア共通利用を成立させる（受け入れ条件）。
//! 2. **FMA 契約の参照 matmul** ([`matmul_reference_fma`]): CPU 参照実装の丸め方針
//!    （`f32::mul_add`・逐次 k 昇順の演算順序固定）を GPU 側の既定 FMA 契約と
//!    揃える基準点（PoC-v2-5 の K=4096 ストレスケースで実測確認済み。
//!    `.claude/rules/coding-rust.md`）。[`crate::gemm`] の公開 GEMM 入口群は
//!    本関数と bit 完全一致する契約を `tests/fma_contract.rs` で固定する。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/compare.rs`
//! （判定ロジックは差分なしで移植。`assert` による panic を型付きエラー
//! （[`ParityError`]）に置き換え、テスト支援 API（[`assert_parity`]）を追加した
//! 点のみ productize している。coding-rust.md「本番経路で unwrap/expect を
//! 使わない」に対応するため）。

use crate::gemm::GemmError;

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

/// [`compare`]・[`assert_parity`] の入力検証エラー。
///
/// `#[non_exhaustive]`: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、後続タスクで検査項目が増えても
/// 呼び出し側の網羅的 match を破壊しないため（`gemm::GemmError` と同方針）。
#[non_exhaustive]
#[derive(Debug)]
pub enum ParityError {
    /// `a`・`b` の要素数が一致しない（shape 不一致の呼び出し誤りを早期検出する）。
    LengthMismatch { left: usize, right: usize },
}

impl std::fmt::Display for ParityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParityError::LengthMismatch { left, right } => {
                write!(
                    f,
                    "parity compare: length mismatch (left={left}, right={right})"
                )
            }
        }
    }
}

impl std::error::Error for ParityError {}

/// 複合判定結果と誤差分布（PoC-v2-5 `CompareReport` の移植）。
///
/// #54/#55 の失敗時診断・`evidence/compare_*.log` 相当のレポート転記に使う
/// 想定のため、判定結果（[`fail_count`](Self::fail_count)）に加え
/// 分布統計（max/mean・p50/p99/p99.9）を保持する。
#[derive(Debug, Clone, Copy)]
pub struct CompareReport {
    pub total: usize,
    pub fail_count: usize,
    pub max_abs_diff: f64,
    pub mean_abs_diff: f64,
    pub max_rel_err: f64,
    pub mean_rel_err: f64,
    pub p50_abs_diff: f64,
    pub p99_abs_diff: f64,
    pub p999_abs_diff: f64,
}

impl CompareReport {
    /// 判定基準（REQ-2）: fail セルが 1 つもないこと。
    pub fn passes(&self) -> bool {
        self.fail_count == 0
    }
}

/// REQ-2 統一複合判定。`a`・`b` は同じ shape の flat データを想定し、
/// 呼び出し元（#54・#55 のペアテスト）が長さ一致を保証できない場合は
/// [`ParityError::LengthMismatch`] を返す。
///
/// バックエンド非依存の `&[f32]` インターフェースのため、CPU-CUDA・
/// CPU-Metal・（将来の）CUDA-Metal のいずれのペアでも同一関数を呼べる
/// （受け入れ条件: 全ペア共通で成立する判定ユーティリティ）。
pub fn compare(a: &[f32], b: &[f32]) -> Result<CompareReport, ParityError> {
    if a.len() != b.len() {
        return Err(ParityError::LengthMismatch {
            left: a.len(),
            right: b.len(),
        });
    }

    let mut abs_diffs: Vec<f64> = Vec::with_capacity(a.len());
    let mut rel_errs: Vec<f64> = Vec::with_capacity(a.len());
    let mut fail_count = 0usize;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        let diff = (xf - yf).abs();
        // 真値 0 近傍での相対誤差の跳ね上がりを避けるため、分母を 1e-12 で
        // 下支えする（PoC-v2-1 `verify_against_rust.py` と同じ方式）。
        let scale = xf.abs().max(yf.abs()).max(1e-12);
        let rel = diff / scale;

        // 合格条件（REQ-2）を肯定形で先に判定し、その否定を fail とする
        // （`autodiff::poc_v2_2_parity::composite_close` と同方針）。
        // NaN vs 有限値・Inf vs 有限値等で `rel`/`diff` が NaN になる場合、
        // `<` 比較は IEEE 754 上つねに false になるため合格条件が成立せず
        // fail 側に倒れる。旧実装（`rel >= tol && diff >= tol` を fail と
        // する否定形）はこの NaN ケースで両辺 false となり誤って合格判定
        // してしまっていた（Cursor Bugbot 指摘・PR #239）。
        let pass = rel < RELATIVE_TOLERANCE || diff < ABSOLUTE_RESCUE_THRESHOLD;
        let fail = !pass;
        if fail {
            fail_count += 1;
        }

        abs_diffs.push(diff);
        rel_errs.push(rel);
    }

    let total = a.len();
    let max_abs_diff = abs_diffs.iter().cloned().fold(0.0f64, f64::max);
    let mean_abs_diff = abs_diffs.iter().sum::<f64>() / total as f64;
    let max_rel_err = rel_errs.iter().cloned().fold(0.0f64, f64::max);
    let mean_rel_err = rel_errs.iter().sum::<f64>() / total as f64;

    let mut sorted = abs_diffs.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let p50_abs_diff = percentile(&sorted, 0.50);
    let p99_abs_diff = percentile(&sorted, 0.99);
    let p999_abs_diff = percentile(&sorted, 0.999);

    Ok(CompareReport {
        total,
        fail_count,
        max_abs_diff,
        mean_abs_diff,
        max_rel_err,
        mean_rel_err,
        p50_abs_diff,
        p99_abs_diff,
        p999_abs_diff,
    })
}

/// テスト支援 API: [`compare`] を呼び、複合判定が FAIL の場合は
/// [`CompareReport`] を整形して panic する。
///
/// `#54`/`#55` のペアテストが「合否のみ判定し、失敗時に分布統計を
/// 併記したい」用途向けの薄いラッパー（本番経路では使わない。
/// テスト専用であることを明示するため `#[track_caller]` を付し、
/// panic 位置が呼び出し側テスト関数を指すようにする）。
///
/// # Panics
///
/// - `actual`・`expected` の長さが一致しない場合
/// - 複合判定が FAIL（`fail_count > 0`）の場合
#[track_caller]
pub fn assert_parity(context: &str, actual: &[f32], expected: &[f32]) {
    let report = match compare(actual, expected) {
        Ok(report) => report,
        Err(err) => panic!("{context}: {err}"),
    };
    assert!(
        report.passes(),
        "{context}: 複合判定 FAIL（fail_count={}/{}, max_abs_diff={:.3e}, \
         max_rel_err={:.3e}, mean_abs_diff={:.3e}, mean_rel_err={:.3e}, \
         p50_abs_diff={:.3e}, p99_abs_diff={:.3e}, p999_abs_diff={:.3e}）",
        report.fail_count,
        report.total,
        report.max_abs_diff,
        report.max_rel_err,
        report.mean_abs_diff,
        report.mean_rel_err,
        report.p50_abs_diff,
        report.p99_abs_diff,
        report.p999_abs_diff,
    );
}

/// 昇順ソート済み配列から最近傍法でパーセンタイルを取る（統計ライブラリを
/// 依存に加えないための最小実装。PoC-v2-5 と同じ「分布の傾向把握が目的で
/// 補間方式の厳密性までは要求しない」方針を踏襲する）。
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// FMA 契約の参照 matmul（`C += A @ B`。逐次・k 昇順の演算順序固定・`f32::mul_add`）。
///
/// [`crate::gemm`]・[`crate::gemm_blis`] の全公開 GEMM 入口（naive/blocked/
/// parallel/parallel_tuned/gemm_blis/gemm_blis_parallel）は本関数と bit
/// 完全一致する契約を持つ（`tests/fma_contract.rs`）。CPU-CUDA（#54）・
/// CPU-Metal（#55）ペアテストが GPU 側出力と比較する際の CPU 側基準点でも
/// あり、GPU 側の既定 FMA 契約（CUDA NVRTC・Metal
/// `simdgroup_multiply_accumulate`）と揃えるための唯一の参照実装として
/// 本モジュールに集約する（`gemm::gemm_naive` と実装ロジックは同一だが、
/// 「FMA 契約の参照点」という役割を明示する別名として独立させ、`gemm_naive`
/// 自体は #24 の 3 段階性能比較の参照点という別の役割を保ったまま変更しない
/// ため）。
///
/// `c` は呼び出し前にゼロ初期化されている前提（本関数は加算のみ行う）。
/// 形状検証は `gemm::validate_dims` を再利用し、`gemm::GemmError` をそのまま
/// 返す（検証ロジックの重複を避ける。`gemm_blis` 系が同じ関数を再利用する
/// 既存パターンと同じ判断）。
pub fn matmul_reference_fma(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    crate::gemm::validate_dims(a, b, c, m, n, k)?;

    for i in 0..m {
        let a_row = &a[i * k..i * k + k];
        let c_row = &mut c[i * n..i * n + n];
        for (p, &a_ip) in a_row.iter().enumerate() {
            let b_row = &b[p * n..p * n + n];
            for j in 0..n {
                c_row[j] = a_ip.mul_add(b_row[j], c_row[j]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- compare / CompareReport ---

    #[test]
    fn identical_arrays_pass_with_zero_diff() {
        let a = [1.0f32, 2.0, 3.0];
        let report = compare(&a, &a).unwrap();
        assert!(report.passes());
        assert_eq!(report.fail_count, 0);
        assert_eq!(report.max_abs_diff, 0.0);
    }

    #[test]
    fn small_relative_diff_within_tolerance_passes() {
        // 1e-3 未満の相対誤差 → 複合判定 PASS。
        let a = [1.0f32];
        let b = [1.0005f32];
        let report = compare(&a, &b).unwrap();
        assert!(report.passes());
    }

    #[test]
    fn near_zero_absolute_rescue_saves_large_relative_error() {
        // 真値がほぼ 0 のとき相対誤差は跳ね上がるが、絶対差が
        // ABSOLUTE_RESCUE_THRESHOLD 未満なら複合判定は PASS になる
        // （0 近傍セルの既知パターン。PoC-v2-3 の実績を踏襲した確認）。
        let a = [0.0f32];
        let b = [1e-6f32];
        let report = compare(&a, &b).unwrap();
        assert!(report.max_rel_err > RELATIVE_TOLERANCE);
        assert!(report.passes());
    }

    /// falsification test: 明らかな不一致を注入した場合に fail が検出される
    /// こと（比較ロジック自体が「常に PASS を返す」壊れ方をしていないことの
    /// 確認。PoC-v2-4 の前例を踏襲）。
    #[test]
    fn falsification_large_diff_is_detected() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 30.0]; // 3.0 vs 30.0: 相対誤差・絶対誤差とも大
        let report = compare(&a, &b).unwrap();
        assert!(!report.passes());
        assert_eq!(report.fail_count, 1);
    }

    #[test]
    fn falsification_nan_mismatch_is_detected() {
        // PR #239 Cursor Bugbot 指摘の回帰テスト: NaN vs 有限値は
        // `rel`/`diff` が NaN になり `<` 比較がつねに false になるため、
        // 合格条件不成立で fail 側に倒れなければならない
        // （旧実装の `>=` 否定形では両辺 false となり誤って合格していた）。
        let a = [1.0f32, f32::NAN, 3.0];
        let b = [1.0f32, 2.0, 3.0];
        let report = compare(&a, &b).unwrap();
        assert!(!report.passes());
        assert_eq!(report.fail_count, 1);
    }

    #[test]
    fn falsification_infinite_vs_finite_mismatch_is_detected() {
        // Inf vs 有限値も同様に diff/rel が非有限になり NaN 比較と
        // 同じ落とし穴を持つため、独立に回帰させる。
        let a = [1.0f32, f32::INFINITY, 3.0];
        let b = [1.0f32, 2.0, 3.0];
        let report = compare(&a, &b).unwrap();
        assert!(!report.passes());
        assert_eq!(report.fail_count, 1);
    }

    #[test]
    fn percentile_of_sorted_array_is_reasonable() {
        // sorted[i] = i+1（値 1..=100、100 要素、index 0..=99）。
        // idx = round((len-1)*p) の最近傍法のため、0.50 は index 50（=値 51）、
        // 0.99 は index 98（=値 99）になる。
        let sorted: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_eq!(percentile(&sorted, 0.50), 51.0);
        assert_eq!(percentile(&sorted, 0.99), 99.0);
    }

    #[test]
    fn compare_rejects_length_mismatch() {
        let a = [1.0f32, 2.0];
        let b = [1.0f32];
        let err = compare(&a, &b).unwrap_err();
        assert!(matches!(
            err,
            ParityError::LengthMismatch { left: 2, right: 1 }
        ));
    }

    #[test]
    fn assert_parity_passes_for_identical_arrays() {
        let a = [1.0f32, 2.0, 3.0];
        assert_parity("identical", &a, &a);
    }

    #[test]
    #[should_panic(expected = "複合判定 FAIL")]
    fn assert_parity_panics_on_large_diff() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [1.0f32, 2.0, 30.0];
        assert_parity("falsification", &a, &b);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn assert_parity_panics_on_length_mismatch() {
        let a = [1.0f32, 2.0];
        let b = [1.0f32];
        assert_parity("length-mismatch", &a, &b);
    }

    // --- matmul_reference_fma ---

    #[test]
    fn matmul_reference_fma_matches_hand_computed_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // A@B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        matmul_reference_fma(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn matmul_reference_fma_handles_non_square_boundary_shape() {
        // m=1（行ベクトル）× k=3 × n=1（列ベクトル）＝内積相当の境界形状。
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut c = vec![0.0; 1];
        matmul_reference_fma(&a, &b, &mut c, 1, 1, 3).unwrap();
        assert_eq!(c, vec![32.0]); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn matmul_reference_fma_rejects_a_len_mismatch() {
        let a = vec![1.0, 2.0, 3.0]; // m*k = 4 を期待
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        let err = matmul_reference_fma(&a, &b, &mut c, 2, 2, 2).unwrap_err();
        assert!(matches!(
            err,
            GemmError::ALenMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }
}
