//! GEMM 結果の要素単位検証（イシュー #970）。
//!
//! 従来の全要素和 checksum（`crate::validate_gemm_checksum`・
//! `summarize.py` の checksum 相互突合。イシュー #965）は「縮退（全ゼロ／
//! 非有限）」と「他フレームワークとの粗い一致」しか検出できず、和が偶然
//! 一致する破損（要素の入れ替わり・正負誤差の相殺）を見逃す。本モジュール
//! は本体の数値一致契約（`crates/backend-cpu/src/parity.rs` の複合判定
//! 「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）と同じ式・同じ閾値を
//! 使い、GEMM 結果を要素単位で参照実装と突合する。
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

/// 要素単位の複合判定（`crates/backend-cpu/src/parity.rs::compare` と同じ式）。
/// `actual`・`reference` は同じ長さの flat データを想定し、長さ不一致は
/// 呼び出し誤りの早期検出として型付きエラーを返す。
pub fn compare_elementwise(actual: &[f32], reference: &[f32]) -> Result<ParityStats, BenchError> {
    if actual.len() != reference.len() {
        return Err(BenchError::ParityLengthMismatch {
            expected: reference.len(),
            actual: actual.len(),
        });
    }

    let total = actual.len();
    let mut fail_count = 0usize;
    let mut max_abs_err = 0.0f64;
    let mut max_rel_err = 0.0f64;

    for (&x, &y) in actual.iter().zip(reference.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        let diff = (xf - yf).abs();
        // 真値 0 近傍での相対誤差の跳ね上がりを避けるため、分母を 1e-12 で
        // 下支えする（本体 `parity::compare` と同一方式）。
        let scale = xf.abs().max(yf.abs()).max(1e-12);
        let rel = diff / scale;

        // NaN 混入時 `rel`/`diff` は NaN になり `<` 比較は常に false のため
        // fail 側に倒れる（本体 `parity::compare` と同じ安全側の挙動）。
        let pass = rel < PARITY_REL_TOL || diff < PARITY_ABS_TOL;
        if !pass {
            fail_count += 1;
        }

        max_abs_err = if diff.is_finite() {
            max_abs_err.max(diff)
        } else {
            f64::INFINITY
        };
        max_rel_err = if rel.is_finite() {
            max_rel_err.max(rel)
        } else {
            f64::INFINITY
        };
    }

    Ok(ParityStats {
        total,
        fail_count,
        max_abs_err,
        max_rel_err,
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
}

impl GemmReference {
    /// `a`・`b` は行優先 flat 表現の N×N 行列。`n == 0` または長さが
    /// `n*n` と一致しない場合は型付きエラーを返す（本番経路で panic
    /// させない。`.claude/rules/coding-rust.md`）。
    pub fn compute(n: usize, a: &[f32], b: &[f32]) -> Result<Self, BenchError> {
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

        Ok(Self { c })
    }

    /// 参照 GEMM の結果（flat・行優先）。
    pub fn as_slice(&self) -> &[f32] {
        &self.c
    }

    /// [`compare_elementwise`] の薄い便宜メソッド（呼び出し側が参照値を
    /// 都度 `as_slice()` する必要をなくす）。
    pub fn verify(&self, out: &[f32]) -> Result<ParityStats, BenchError> {
        compare_elementwise(out, &self.c)
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
        let stats = compare_elementwise(&v, &v).expect("compare");
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
        let stats = compare_elementwise(&actual, &reference).expect("compare");
        assert_eq!(stats.fail_count, 1);
        assert!(stats.max_rel_err > PARITY_REL_TOL);
        assert!((stats.max_abs_err - 0.03).abs() < 1e-6);
    }

    #[test]
    fn compare_elementwise_near_zero_diff_is_rescued_by_absolute_threshold() {
        // 相対誤差は跳ね上がるが絶対誤差 1e-7 < PARITY_ABS_TOL(1e-5) で救済。
        let reference = vec![1e-7f32];
        let actual = vec![2e-7f32];
        let stats = compare_elementwise(&actual, &reference).expect("compare");
        assert_eq!(stats.fail_count, 0);
    }

    #[test]
    fn compare_elementwise_nan_is_fail_with_infinite_max_abs_err() {
        let reference = vec![1.0f32];
        let actual = vec![f32::NAN];
        let stats = compare_elementwise(&actual, &reference).expect("compare");
        assert_eq!(stats.fail_count, 1);
        assert_eq!(stats.max_abs_err, f64::INFINITY);
        assert_eq!(stats.max_rel_err, f64::INFINITY);
    }

    #[test]
    fn compare_elementwise_length_mismatch_is_typed_error() {
        assert!(matches!(
            compare_elementwise(&[1.0, 2.0], &[1.0]),
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
        };
        let b = ParityStats {
            total: 100,
            fail_count: 2,
            max_abs_err: 1e-8,
            max_rel_err: 1e-4,
        };
        let w = a.worst(b);
        assert_eq!(w.fail_count, 2);
        assert!((w.max_abs_err - 1e-6).abs() < 1e-12);
        assert!((w.max_rel_err - 1e-4).abs() < 1e-12);
    }

    #[test]
    fn worst_propagates_infinite_sentinel() {
        let ok = ParityStats {
            total: 10,
            fail_count: 0,
            max_abs_err: 1e-6,
            max_rel_err: 1e-6,
        };
        let broken = ParityStats {
            total: 10,
            fail_count: 1,
            max_abs_err: f64::INFINITY,
            max_rel_err: f64::INFINITY,
        };
        let w = ok.worst(broken);
        assert_eq!(w.max_abs_err, f64::INFINITY);
        assert_eq!(w.max_rel_err, f64::INFINITY);
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
