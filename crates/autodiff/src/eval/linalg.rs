//! 線形代数（inv／solve／det／qr／cholesky／svd）・matrix_norm のホスト
//! 参照実装（イシュー #1621・親イシュー #1573「Tier 2: 線形代数」・
//! `docs/spec/04-requirements.md` REQ-9 2026-09-12 追記）。
//!
//! `crate::eval`（`NaiveOps`／`TestOps` compat 経路の forward 参照実装）
//! と同じ役割の拡張だが、行数が大きいためファイルを分ける
//! （`crate::eval` モジュールの子モジュールとして `eval/linalg.rs` に
//! 置く。`lib.rs` の `mod eval;` は変更しない）。
//!
//! `crates/backend-cpu/src/linalg.rs`（本番 CPU 実装）は本モジュールと
//! **同一のアルゴリズム・同一の符号／ゲージ規約**で実装する
//! （`docs/autodiff-linalg-design.md` §3「eval と CPU の関係」）。ただし
//! 依存方向の制約（`autodiff` → `backend-cpu` の依存は作らない。
//! `.claude/rules/delegation-impl.md`）により、コードは意図的に複製する
//! （数式の実体を 2 か所に持つこと自体は avoid しないが、shape 検査等の
//! 「呼び出し元が保証すべき前提」は本モジュールに一元化しない——
//! `crate::eval` モジュール冒頭コメントの契約と同じく、本モジュールの
//! 関数は shape が既に整合していることを前提とする）。
//!
//! # 数値契約（`docs/autodiff-linalg-design.md` §3.5 が正）
//!
//! - **内部精度**: 分解・解法は `f64` で逐次固定順序に計算し、出力時に
//!   1 回だけ `f32` へ downcast する（正規化統計の `f64` アキュムレータ
//!   契約と同じ思想。matmul 系 FMA 契約は変更しない。
//!   `.claude/rules/coding-rust.md`）。
//! - **符号・ゲージ規約**: QR は `R` の対角を非負に正規化する
//!   （Householder 反射の符号をここで吸収する）。SVD は特異値を降順
//!   （同値は安定ソート）に並べ、各 `V` 列は最大絶対値成分（同値は
//!   最小添字）が正になるよう符号を正規化し、`U = A V / σ` で導出する
//!   （`σ == 0` の列は Gram–Schmidt で補完する）。Cholesky は下三角のみ
//!   を返す（上三角は 0）。
//! - **エラー分類**: 特異／非正定値／非収束は
//!   `AutodiffError::InvalidArgument`（`crate::error`）。本モジュールの
//!   関数は shape 検査を行わない契約（呼び出し元 `var.rs` の責務）だが、
//!   数値的な失敗（特異・非収束等）はここで検出し `Result` で返す。
//! - **空行列（`n=0`）**: `det` は空積 `1.0`。`inv`／`cholesky`／`qr`／
//!   `svd` は対応する空 shape のテンソルを返す。`solve` は `[0,k]`。

use fandhe_ai_tensor_core::{MatrixNormOrd, Tensor};

use crate::error::AutodiffError;

use super::{build_tensor, dense_vec};

/// `f64` 版の稠密行列（行優先）。分解アルゴリズムは全てこの内部表現を
/// 使い、入出力の境界でのみ `f32` `Tensor` と相互変換する（本ファイル
/// 冒頭「数値契約」参照）。
#[derive(Debug, Clone)]
struct Mat {
    data: Vec<f64>,
    rows: usize,
    cols: usize,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Mat {
        Mat {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    fn identity(n: usize) -> Mat {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    fn from_tensor(t: &Tensor<f32>) -> Mat {
        let shape = t.shape();
        debug_assert!(
            shape.len() == 2,
            "eval::linalg::Mat::from_tensor: 呼び出し元が rank-2 を検査済みの契約"
        );
        let rows = shape[0];
        let cols = shape[1];
        let data = dense_vec(t).into_iter().map(f64::from).collect();
        Mat { data, rows, cols }
    }

    fn to_tensor(&self) -> Tensor<f32> {
        let data: Vec<f32> = self.data.iter().map(|&v| v as f32).collect();
        build_tensor(data, &[self.rows, self.cols])
    }

    #[inline]
    fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    fn col(&self, c: usize) -> Vec<f64> {
        (0..self.rows).map(|r| self.get(r, c)).collect()
    }

    fn set_col(&mut self, c: usize, values: &[f64]) {
        for (r, &v) in values.iter().enumerate() {
            self.set(r, c, v);
        }
    }

    fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.set(c, r, self.get(r, c));
            }
        }
        out
    }

    /// `self @ other`（rank-2 の素朴な `f64` 逐次和。分解サイズ〈通常
    /// 小〜中規模〉が対象のため、`BackendOps::gemm` の並列 SIMD 実装を
    /// ここで再利用しない——本モジュールは `tensor-core` の `Tensor<f32>`
    /// 境界の外で完結する `f64` 内部計算という設計上の理由がある）。
    fn matmul(&self, other: &Mat) -> Mat {
        debug_assert_eq!(self.cols, other.rows);
        let mut out = Mat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self.get(i, k);
                if a_ik == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out.set(i, j, out.get(i, j) + a_ik * other.get(k, j));
                }
            }
        }
        out
    }
}

fn invalid(msg: impl Into<String>) -> AutodiffError {
    AutodiffError::InvalidArgument(msg.into())
}

// =====================================================================
// LU 分解（部分ピボット）。`inv`／`solve`／`det` の共通基盤。
// =====================================================================

/// 部分ピボット LU 分解の結果。`lu` は下三角（対角 1 は暗黙・非格納）と
/// 上三角を 1 個の行列に重ねて保持する（標準的な in-place LU 表現）。
/// `perm` は行の置換（`perm[i]` = 元の行番号）、`sign` は置換の符号
/// （行交換回数の偶奇）。
struct LuDecomp {
    lu: Mat,
    perm: Vec<usize>,
    sign: f64,
}

/// `A: [n,n]` の部分ピボット LU 分解。ピボットが厳密 `0.0`（数値的な
/// ほぼ特異ではなく厳密特異のみを検出する設計。設計文書 §3.5「特異
/// （ピボット厳密 0）」）の場合 `None` を返す。
fn lu_decompose(a: &Mat) -> Option<LuDecomp> {
    let n = a.rows;
    debug_assert_eq!(a.cols, n, "lu_decompose: 正方行列の契約");
    let mut lu = a.clone();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut sign = 1.0;

    for k in 0..n {
        // 部分ピボット選択（数値安定性のため列 k で絶対値最大の行を選ぶ。
        // 「厳密特異」の判定はピボット後の値が 0.0 かどうかで行う）。
        let mut max_row = k;
        let mut max_val = lu.get(k, k).abs();
        for i in (k + 1)..n {
            let v = lu.get(i, k).abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_row != k {
            for c in 0..n {
                let tmp = lu.get(k, c);
                lu.set(k, c, lu.get(max_row, c));
                lu.set(max_row, c, tmp);
            }
            perm.swap(k, max_row);
            sign = -sign;
        }
        let pivot = lu.get(k, k);
        if pivot == 0.0 {
            return None;
        }
        for i in (k + 1)..n {
            let factor = lu.get(i, k) / pivot;
            lu.set(i, k, factor);
            if factor == 0.0 {
                continue;
            }
            for c in (k + 1)..n {
                lu.set(i, c, lu.get(i, c) - factor * lu.get(k, c));
            }
        }
    }

    Some(LuDecomp { lu, perm, sign })
}

/// 前進代入（下三角・対角 1 が暗黙）: `L y = perm(b)` を解く。
fn forward_substitute_unit(lu: &LuDecomp, b: &[f64]) -> Vec<f64> {
    let n = lu.lu.rows;
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[lu.perm[i]];
        for (j, &yj) in y.iter().enumerate().take(i) {
            sum -= lu.lu.get(i, j) * yj;
        }
        y[i] = sum;
    }
    y
}

/// 後退代入（上三角）: `U x = y` を解く。
fn backward_substitute(lu: &LuDecomp, y: &[f64]) -> Vec<f64> {
    let n = lu.lu.rows;
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for (j, &xj) in x.iter().enumerate().skip(i + 1) {
            sum -= lu.lu.get(i, j) * xj;
        }
        x[i] = sum / lu.lu.get(i, i);
    }
    x
}

/// LU 分解済みの `A` で `A x = b`（`b` はベクトル）を解く。
fn lu_solve_vec(lu: &LuDecomp, b: &[f64]) -> Vec<f64> {
    let y = forward_substitute_unit(lu, b);
    backward_substitute(lu, &y)
}

/// LU 分解済みの `A` で `A X = B`（`B: [n,k]`）を解く（列ごとに
/// `lu_solve_vec`）。
fn lu_solve_mat(lu: &LuDecomp, b: &Mat) -> Mat {
    let mut x = Mat::zeros(b.rows, b.cols);
    for c in 0..b.cols {
        let bc = b.col(c);
        let xc = lu_solve_vec(lu, &bc);
        x.set_col(c, &xc);
    }
    x
}

/// `A^{-1}`（`A X = I` を解く）。VJP（[`inv_vjp`]）は forward が返す
/// `out_value`（`f32` 丸め済み記録値）を再利用せず、`a` から改めて
/// `f64` で計算し直す（[`inv_vjp`] doc 参照。codex-review 指摘・
/// 2026-09-13 是正。以前の本コメントは逆の記述だった）。
pub(crate) fn inv(a: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
    let mat = Mat::from_tensor(a);
    let n = mat.rows;
    if n == 0 {
        return Ok(build_tensor(Vec::new(), &[0, 0]));
    }
    let lu = lu_decompose(&mat)
        .ok_or_else(|| invalid("eval::linalg::inv: 行列が特異（ピボットが厳密 0）"))?;
    let identity = Mat::identity(n);
    let x = lu_solve_mat(&lu, &identity);
    Ok(x.to_tensor())
}

/// `A X = B` を解く。
pub(crate) fn solve(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
    let a_mat = Mat::from_tensor(a);
    let b_mat = Mat::from_tensor(b);
    let n = a_mat.rows;
    if n == 0 {
        return Ok(build_tensor(Vec::new(), &[0, b_mat.cols]));
    }
    let lu = lu_decompose(&a_mat)
        .ok_or_else(|| invalid("eval::linalg::solve: 係数行列が特異（ピボットが厳密 0）"))?;
    let x = lu_solve_mat(&lu, &b_mat);
    Ok(x.to_tensor())
}

/// `x = m · 2^e`（`0.5 <= |m| < 1`。`frexp` 相当）へ分解する。`0` と
/// 非有限（`NaN`／`Inf`）はそのまま素通しする（`e = 0`）。
/// `f64::to_bits`／`from_bits` によるビットフィールド抽出・再構成のみで
/// 構成し `unsafe` を使わない（IEEE 754 binary64 の指数・仮数フィールド
/// は安全に読み書きできる。非正規化数〈subnormal〉は `2^64` を掛けて
/// 正規化してから指数を補正する）。[`lu_diag_product`] が使う
/// （codex-review 指摘 PRRT_kwDOTuUCJc6hxiys の是正）。
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    let bits = x.to_bits();
    let exponent_field = (bits >> 52) & 0x7ff;
    if exponent_field == 0 {
        // 非正規化数: 2^64 倍して正規化してから指数を 64 引いて補正する
        // （最小の非正規化数 × 2^64 でも正規化数の範囲に収まるため、
        // この 1 回の再帰呼び出しで必ず正規化数の分岐へ入る）。
        let (m, e) = frexp(x * 2f64.powi(64));
        return (m, e - 64);
    }
    let sign_bit = bits & 0x8000_0000_0000_0000;
    let mantissa_bits = bits & 0x000f_ffff_ffff_ffff;
    // 指数フィールドを 1022（非バイアス指数 -1）へ差し替えると
    // `1.mantissa_bits * 2^-1` となり、仮数が `[0.5, 1)`（符号付きなら
    // `(-1, -0.5]` または `[0.5, 1)`）に収まる。
    let new_bits = sign_bit | (1022u64 << 52) | mantissa_bits;
    let m = f64::from_bits(new_bits);
    let e = exponent_field as i32 - 1022;
    (m, e)
}

/// `m · 2^e`（[`frexp`] の逆演算。`ldexp` 相当）。底が厳密に 2 の
/// べき乗であるため `powi` は表現範囲内で丸め誤差を追加しない
/// （`f64` の乗算は 2 のべき倍について常に正確）。`f64` の表現範囲
/// （約 `±1.8e308`）を超える場合は標準の浮動小数点挙動どおり
/// `Inf`／`0.0` を返す（真値が範囲外のときの正しい挙動）。
fn ldexp(m: f64, e: i32) -> f64 {
    m * 2f64.powi(e)
}

/// LU 対角成分の総積（置換符号込み）を仮数・指数分離（[`frexp`]／
/// [`ldexp`] 相当）で計算する。単純な `f64` 逐次積では、正負に極端な
/// スケールが混在する対角（例 `diag([1e30; 11], [1e-30; 11])`。真の
/// 行列式は約 `1.0`）で中間積が `f64` の表現範囲を超えて `Inf` になり、
/// 続く小スケール要素と乗算しても `Inf` のまま戻らない——本来有限な
/// 行列式・[`det_vjp`] の勾配が `Inf`／`NaN` になってしまう
/// （codex-review 指摘 PRRT_kwDOTuUCJc6hxiys）。log-abs 方式（`ln` の
/// 和を経由する方式）は不要な丸め誤差を追加するため採用せず、各対角
/// 要素を `m·2^e`（`0.5<=|m|<1`）へ分解し、仮数の積を毎回正規化しつつ
/// 指数を整数で加算する方式を採る。最終合成（`ldexp`）が `f64` の
/// 範囲を超える場合にのみ `Inf`／`0.0` を返す（真値が範囲外のときの
/// 正しい挙動）。`det`・`det_vjp` の両方が本関数を経由することで、
/// 行列式の中間値に依存する箇所すべてで同じオーバーフロー耐性を持つ。
fn lu_diag_product(lu: &LuDecomp) -> f64 {
    let n = lu.lu.rows;
    let mut mantissa = 1.0f64;
    let mut exponent: i64 = 0;
    for i in 0..n {
        let (m, e) = frexp(lu.lu.get(i, i));
        mantissa *= m;
        exponent += i64::from(e);
        // 仮数を毎回 `[0.5, 1)` 近傍へ正規化し直すことで、対角成分数が
        // 多い場合でも仮数自体のアンダーフロー（正規化を怠ると
        // `0.5^n` で消失しうる）を防ぐ。`0`／非有限は `frexp` が
        // そのまま素通しする設計のため、ここでの再正規化もそのまま
        // 伝播する。
        if mantissa != 0.0 && mantissa.is_finite() {
            let (m2, e2) = frexp(mantissa);
            mantissa = m2;
            exponent += i64::from(e2);
        }
    }
    mantissa *= lu.sign;
    let exponent = exponent.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
    ldexp(mantissa, exponent)
}

/// `det(A)`（LU 対角積 × 置換符号）。空行列は空積 `1.0`。特異行列は
/// `0.0`（エラーにしない。設計文書 §3.5・PyTorch `torch.linalg.det`
/// 挙動）。対角積は [`lu_diag_product`]（仮数・指数分離）で計算し、
/// 単純な `f64` 逐次積の中間オーバーフローを避ける。
pub(crate) fn det(a: &Tensor<f32>) -> Tensor<f32> {
    let mat = Mat::from_tensor(a);
    let n = mat.rows;
    if n == 0 {
        return build_tensor(vec![1.0], &[]);
    }
    let value = match lu_decompose(&mat) {
        None => 0.0,
        Some(lu) => lu_diag_product(&lu),
    };
    build_tensor(vec![value as f32], &[])
}

/// `A^{-T} b`（`f64` の [`Mat`] のまま返す）。`Op::Solve` の VJP
/// （`dB = A^{-T} g`）専用の内部実装。forward の `lu_decompose(a)` を
/// 再利用せず、`aᵀ` を明示的に分解する（転置行列の LU を都度計算する
/// のは非効率だが、分解サイズが小さい前提のため単純さを優先する。
/// 設計文書 §3.4「Solve」）。
///
/// [`solve_vjp`] は戻り値を `f32` へ downcast せず `dA = -dB Xᵀ` の
/// 計算にそのまま使い、両方の勾配を計算し終えてから 1 回だけ
/// downcast する（`det_vjp`／`fro` ノルム VJP と同じ内部精度契約
/// 〈`.claude/rules/coding-rust.md`〉。codex-review 指摘: `dB` を
/// 先に `f32` へ downcast すると、極端なスケール〈`A=[[1e-20]]`・
/// `B=[[1e-30]]`・`upstream=1e20`〉で `dB≈1e40` が `f32::INFINITY` に
/// 丸められ、本来有限の `dA≈-1e30` も `-Inf` へ伝播してしまう）。
fn solve_transposed_mat(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Mat, AutodiffError> {
    let a_mat = Mat::from_tensor(a);
    let a_t = a_mat.transpose();
    let b_mat = Mat::from_tensor(b);
    let n = a_t.rows;
    if n == 0 {
        return Ok(Mat::zeros(0, b_mat.cols));
    }
    let lu = lu_decompose(&a_t)
        .ok_or_else(|| invalid("eval::linalg::solve_transposed: 係数行列が特異"))?;
    Ok(lu_solve_mat(&lu, &b_mat))
}

// =====================================================================
// Cholesky 分解（Cholesky–Banachiewicz、下三角）。
// =====================================================================

/// `A = L Lᵀ`（`A` は対称正定値と仮定・下三角のみ読む）。非正定値
/// （対角が非有限または非正）は `InvalidArgument`。
pub(crate) fn cholesky(a: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
    let mat = Mat::from_tensor(a);
    let n = mat.rows;
    let mut l = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = mat.get(i, j);
            for k in 0..j {
                sum -= l.get(i, k) * l.get(j, k);
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return Err(invalid(
                        "eval::linalg::cholesky: 行列が対称正定値でない（対角が非有限または非正）",
                    ));
                }
                l.set(i, j, sum.sqrt());
            } else {
                let diag = l.get(j, j);
                l.set(i, j, sum / diag);
            }
        }
    }
    Ok(l.to_tensor())
}

// =====================================================================
// Householder QR（reduced）。
// =====================================================================

/// reduced QR（`A: [m,n]` → `Q: [m,k]`・`R: [k,n]`、`k = min(m,n)`）。
/// `R` の対角は非負に正規化する（設計文書 §3.5「符号・ゲージ規約」）。
pub(crate) fn qr(a: &Tensor<f32>) -> (Tensor<f32>, Tensor<f32>) {
    let mat = Mat::from_tensor(a);
    let (m, n) = (mat.rows, mat.cols);
    let k = m.min(n);
    if m == 0 || n == 0 {
        return (
            build_tensor(Vec::new(), &[m, k]),
            build_tensor(Vec::new(), &[k, n]),
        );
    }

    // Householder 反射を `r`（作業用に `A` を上書き）へ逐次適用する。
    // `Q`（= `H_0 H_1 ... H_{k-1}`）は `m×m` の単位行列を明示構築せず、
    // 正規化済み反射ベクトル `v`（列インデックス付き）のみを蓄積し、
    // reduced 形（`[m,k]`）を後段で逆順適用により直接構築する
    // （O(mk) メモリ。以前は `Mat::identity(m)` により `m×m` の `f64`
    // 行列を確保しており、`[100000,1]` のような縦長入力でメモリ枯渇を
    // 招いていた——`m×m` だけで約 80 GB。codex-review 指摘。
    // `docs/autodiff-linalg-design.md` §3.4「QR」）。
    let mut r = mat;
    let mut reflectors: Vec<(usize, Vec<f64>)> = Vec::with_capacity(k);

    for col in 0..k {
        // Householder ベクトル `v`（列 `col` の対角以下）を作る。
        let mut x = vec![0.0; m - col];
        for i in col..m {
            x[i - col] = r.get(i, col);
        }
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_x == 0.0 {
            // この列は既にゼロ以下三角化済み（rank 落ち）。反射不要
            // （`reflectors` へ何も積まない＝恒等反射として扱う）。
            continue;
        }
        // 数値安定性のため `alpha` の符号は `x[0]` と逆にする
        // （標準的な Householder 反射の選択）。
        let alpha = if x[0] >= 0.0 { -norm_x } else { norm_x };
        let mut v = x.clone();
        v[0] -= alpha;
        let norm_v: f64 = v.iter().map(|e| e * e).sum::<f64>().sqrt();
        if norm_v == 0.0 {
            continue;
        }
        for e in v.iter_mut() {
            *e /= norm_v;
        }

        // `r ← H r`（`H = I - 2 v vᵀ`、部分行列 `[col.., col..]` 以降に
        // のみ作用）。
        for c in col..n {
            let mut dot = 0.0;
            for (i, &vi) in v.iter().enumerate() {
                dot += vi * r.get(col + i, c);
            }
            if dot == 0.0 {
                continue;
            }
            for (i, &vi) in v.iter().enumerate() {
                let idx = col + i;
                r.set(idx, c, r.get(idx, c) - 2.0 * vi * dot);
            }
        }
        reflectors.push((col, v));
    }

    // reduced `Q`（`[m,k]`）を `E_k`（`I_m` の先頭 `k` 列）から出発し、
    // 蓄積した反射を**逆順**（`col` の大きい順）に適用して構築する:
    // `Q e_j = H_0 (H_1 (... (H_{k-1} e_j) ...))`（`Q = H_0 H_1 ...
    // H_{k-1}` の定義どおり、ベクトルへ右から順に作用させるには
    // 反射を逆順に適用する）。`col > j` の反射は `e_j`（`j` 列成分の
    // みが非零）の `[col, m)` 区間が全て 0 のため内積が 0 になり恒等
    // 変換となる（`if dot == 0.0 { continue }` が自然にスキップする）。
    let mut q_reduced = Mat::zeros(m, k);
    for i in 0..k {
        q_reduced.set(i, i, 1.0);
    }
    for (col, v) in reflectors.iter().rev() {
        let col = *col;
        for c in 0..k {
            let mut dot = 0.0;
            for (i, &vi) in v.iter().enumerate() {
                dot += vi * q_reduced.get(col + i, c);
            }
            if dot == 0.0 {
                continue;
            }
            for (i, &vi) in v.iter().enumerate() {
                let idx = col + i;
                q_reduced.set(idx, c, q_reduced.get(idx, c) - 2.0 * vi * dot);
            }
        }
    }

    // `R` の先頭 k 行へ切り出しつつ対角の符号を非負へ正規化する
    // （対応する `Q` 列の符号も反転して `Q R = A` を保つ）。
    let mut r_reduced = Mat::zeros(k, n);
    for row in 0..k {
        for c in 0..n {
            // 下三角部分（`R` の理論上ゼロになるべき成分。Householder の
            // 数値誤差で厳密 0 にならない場合があるため明示的に 0 とし、
            // parity・不変量テストの安定性を高める）。
            let v = if c < row { 0.0 } else { r.get(row, c) };
            r_reduced.set(row, c, v);
        }
    }
    for i in 0..k {
        let diag = r_reduced.get(i, i);
        if diag < 0.0 {
            for c in 0..n {
                r_reduced.set(i, c, -r_reduced.get(i, c));
            }
            for row in 0..m {
                q_reduced.set(row, i, -q_reduced.get(row, i));
            }
        }
    }

    (q_reduced.to_tensor(), r_reduced.to_tensor())
}

// =====================================================================
// 片側 Jacobi SVD（Hestenes 法）。
// =====================================================================

/// ベクトルの符号を「最大絶対値成分（同値は最小添字）が正」に
/// 正規化する（設計文書 §3.5 の SVD 符号規約。`svd` の Gram–Schmidt
/// 直交補完で生成する `σ == 0` 側の列にもこの規約を適用するために
/// 独立関数化した——補完前の符号正規化ループの後に上書きするため、
/// 適用しないと補完列だけ規約に従わない不変量違反になっていた
/// （codex-review 指摘）。
fn normalize_max_abs_sign(v: &mut [f64]) {
    let mut max_abs = 0.0;
    let mut max_idx = 0usize;
    for (idx, &x) in v.iter().enumerate() {
        if x.abs() > max_abs {
            max_abs = x.abs();
            max_idx = idx;
        }
    }
    if v[max_idx] < 0.0 {
        for x in v.iter_mut() {
            *x = -*x;
        }
    }
}

/// 片側 Jacobi SVD（[`jacobi_svd_tall`]）の列直交収束判定に使う相対
/// しきい値。`gamma.abs() <= JACOBI_EPS * sqrt(alpha*beta)` を満たす
/// 列ペアは既に十分直交とみなし回転をスキップする（絶対下限を持た
/// ない理由は [`jacobi_svd_tall`] 内のコメント参照。旧ローカル定数
/// をモジュール定数へ昇格し、収束判定という単一の意味に統一）。
const JACOBI_EPS: f64 = 1e-14;

/// `m >= n` の片側 Jacobi SVD（列直交化による古典的手法）。
/// `A` の列を回転で逐次直交化し、収束後の列ノルムが特異値になる。
/// 反復上限は 60 スイープ（実用上ほぼ全ての小〜中規模行列で収束する）。
fn jacobi_svd_tall(a: &Mat) -> Result<(Mat, Vec<f64>, Mat), AutodiffError> {
    let (m, n) = (a.rows, a.cols);
    let mut u = a.clone();
    let mut v = Mat::identity(n);
    const MAX_SWEEPS: usize = 60;

    for _sweep in 0..MAX_SWEEPS {
        let mut converged = true;
        for p in 0..n {
            for q in (p + 1)..n {
                let col_p = u.col(p);
                let col_q = u.col(q);
                let alpha: f64 = col_p.iter().map(|v| v * v).sum();
                let beta: f64 = col_q.iter().map(|v| v * v).sum();
                let gamma: f64 = col_p.iter().zip(col_q.iter()).map(|(a, b)| a * b).sum();

                // 収束判定は `alpha*beta` の平方根に対する相対しきい値のみ
                // で行う（絶対下限 `.max(EPS)` を持たない）。`alpha` また
                // は `beta` が 0（零列）のとき `gamma` も必ず 0 になる
                // （零ベクトルとの内積）ため `0.0 <= 0.0` で安全に収束
                // 判定できる。絶対下限があると `JACOBI_EPS*JACOBI_EPS = 1e-28` という
                // 入力スケール非依存の閾値が生じ、列ノルムが
                // 約 1e-15 スケールの小さい入力で非直交な列を誤って
                // 収束扱いし、誤った特異値・特異ベクトルを返していた
                // （codex-review 指摘）。
                if gamma.abs() <= JACOBI_EPS * (alpha * beta).sqrt() {
                    continue;
                }
                converged = false;

                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for row in 0..m {
                    let up = u.get(row, p);
                    let uq = u.get(row, q);
                    u.set(row, p, c * up - s * uq);
                    u.set(row, q, s * up + c * uq);
                }
                for row in 0..n {
                    let vp = v.get(row, p);
                    let vq = v.get(row, q);
                    v.set(row, p, c * vp - s * vq);
                    v.set(row, q, s * vp + c * vq);
                }
            }
        }
        if converged {
            let sigmas: Vec<f64> = (0..n)
                .map(|j| u.col(j).iter().map(|v| v * v).sum::<f64>().sqrt())
                .collect();
            return Ok((u, sigmas, v));
        }
    }
    Err(invalid(
        "eval::linalg::svd: 片側 Jacobi 法が反復上限（60 スイープ）内に収束しなかった",
    ))
}

/// [`svd`] の戻り値 `(U, S, Vh)`（clippy::type_complexity 回避の型
/// エイリアス）。
type SvdOutput = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

/// reduced SVD（`A: [m,n]` → `U: [m,k]`・`S: [k]`・`Vh: [k,n]`、
/// `k = min(m,n)`）。`m < n` は転置に適用して結果を入れ替える
/// （`Uᵀ = Vh`・`Vhᵀ = U` の関係を使う）。特異値は降順（同値は安定
/// ソート）、各 `V` 列は最大絶対値成分（同値は最小添字）が正になる
/// よう符号を正規化する（設計文書 §3.5）。
pub(crate) fn svd(a: &Tensor<f32>) -> Result<SvdOutput, AutodiffError> {
    let mat = Mat::from_tensor(a);
    let (m, n) = (mat.rows, mat.cols);
    let k = m.min(n);
    if k == 0 {
        return Ok((
            build_tensor(Vec::new(), &[m, k]),
            build_tensor(Vec::new(), &[k]),
            build_tensor(Vec::new(), &[k, n]),
        ));
    }

    // `m < n` は `Aᵀ`（`[n,m]`、`n >= m`）に `jacobi_svd_tall` を適用し
    // `U`/`V` を入れ替えて `A` 側の意味へ戻す: `Aᵀ = U' Σ' V'ᵀ` ならば
    // `A = V' Σ' U'ᵀ` なので、`A` の `U`（形状 `[m,k]`）は `V'`
    // （`jacobi_svd_tall` が返す正方 `k×k`）、`A` の `V`（形状 `[n,k]`）は
    // `U'`（非正規化列を持つ `[n,k]`）に対応する。
    let (u_full, sigmas, v_full) = if m >= n {
        let (u, s, v) = jacobi_svd_tall(&mat)?;
        (u, s, v)
    } else {
        let (u_prime, s, v_prime) = jacobi_svd_tall(&mat.transpose())?;
        (v_prime, s, u_prime)
    };

    // `v_full` を単位長へ正規化する（`m >= n` 分岐は `jacobi_svd_tall`
    // の `v` 戻り値が既に直交〈単位長〉のため実質 no-op。`m < n` 分岐は
    // `v_full = u_prime` が非正規化〈列ノルム = σ〉であるため、ここで
    // 割らないと後段の `U = A V / σ` 再導出が `σ` 倍ずれる）。
    let mut v_full = v_full;
    for j in 0..k {
        let col = v_full.col(j);
        let norm: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            for r in 0..v_full.rows {
                v_full.set(r, j, v_full.get(r, j) / norm);
            }
        }
    }

    // 降順ソート（同値は安定ソート＝元の添字順を保つ。`sort_by` は
    // 安定ソート）。`sigmas` は列ノルム（`v*v` の和の `sqrt`）のため
    // 理論上 NaN にはならないことが多いが（`k >= 2` では非有限入力が
    // `jacobi_svd_tall` の収束判定〈`gamma.abs() <= JACOBI_EPS * ...`〉
    // を常に偽にし 60 スイープ非収束の `Err` で弾かれる）、`k == 1`
    // （`min(m,n) == 1`）では列ペア走査（`q in (p+1)..n`）自体が
    // 一度も実行されないため非有限入力を拒否できず NaN が
    // 素通りしうる（`docs/autodiff-linalg-design.md` の「非有限入力
    // は演算ごとに異なり一様ではない」記述どおり。codex-review
    // 指摘）。いずれの場合も `partial_cmp(...).unwrap()`
    // （unwrap 禁止。`.claude/rules/security.md`「本番経路の panic
    // 禁止」）を避け `total_cmp`（IEEE 754-2008 totalOrder。NaN も
    // 全順序に含め panic しない）で防御的に比較する（codex-review
    // 指摘。`crates/backend-cpu/src/linalg.rs::svd` と同型の対応）。
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&i, &j| sigmas[j].total_cmp(&sigmas[i]));

    let mut s_sorted = vec![0.0; k];
    let mut v_sorted = Mat::zeros(v_full.rows, k);
    let mut u_sorted = Mat::zeros(u_full.rows, k);
    for (new_idx, &old_idx) in order.iter().enumerate() {
        s_sorted[new_idx] = sigmas[old_idx];
        v_sorted.set_col(new_idx, &v_full.col(old_idx));
        u_sorted.set_col(new_idx, &u_full.col(old_idx));
    }

    // 各 `V` 列の符号を「最大絶対値成分（同値は最小添字）が正」に
    // 正規化し、`U` の対応列も同じ符号反転で追従させる（`U Σ Vᵀ = A`
    // を保つ）。`σ ≈ 0` の列（rank 落ち）は「`jacobi_svd_tall` の作業
    // 行列（回転ではなく明示的に回転される側）に対応する列」が数値
    // 誤差でほぼゼロベクトルになりうる——`m >= n` 分岐では `U`
    // （`jacobi_svd_tall` の `u`）、`m < n` 分岐では `V`
    // （`jacobi_svd_tall` の `u` を入れ替えたもの。上の分岐コメント
    // 参照）がこれに当たり、対する `V`／`U`（`jacobi_svd_tall` の
    // 回転行列 `v`）は σ に関わらず厳密直交のまま。下記 2 つの
    // Gram–Schmidt 補完ループ（`U` 側・`V` 側）で両方を独立に
    // 直交補完し、どちらの分岐でも公開 `SvdFactors` の列直交契約
    // （設計文書 §3.5）を保証する（`A=[[0,0]]` のような横長・全ゼロ
    // 特異値ケースで `Vh` の行が非単位ノルムのまま残っていた
    // codex-review 指摘の修正。`crates/backend-cpu/src/linalg.rs::svd`
    // と同型の対応）。
    for (j, &sigma) in s_sorted.iter().enumerate() {
        let col = v_sorted.col(j);
        let mut max_abs = 0.0;
        let mut max_idx = 0usize;
        for (idx, &v) in col.iter().enumerate() {
            if v.abs() > max_abs {
                max_abs = v.abs();
                max_idx = idx;
            }
        }
        let sign = if col[max_idx] < 0.0 { -1.0 } else { 1.0 };
        if sign < 0.0 {
            for r in 0..v_sorted.rows {
                v_sorted.set(r, j, -v_sorted.get(r, j));
            }
            for r in 0..u_sorted.rows {
                u_sorted.set(r, j, -u_sorted.get(r, j));
            }
        }

        // `U` 列を自身の実測ノルムで単位長へ正規化する。以前は
        // `U = A V / σ` で再導出していたが、`m >= n` 分岐の `U`
        // （`jacobi_svd_tall` の作業行列列。列ノルム = σ）と
        // `m < n` 分岐の `U`（`jacobi_svd_tall` の回転行列 `v` 由来。
        // 既に単位長）とで意味が異なるにも関わらず両分岐へ同一の
        // 行列積 `A V` を適用していたため、rank 落ち・悪条件の
        // `A` では桁落ちが増幅され再構成した列が非直交になっていた
        // （例 `A=[[1,3],[2,6],[5,15]]` のような rank-1 行列。
        // codex-review 指摘）。列は `jacobi_svd_tall` の回転／作業行列
        // 演算のみで既に（ほぼ）直交に保たれているため、追加の行列積
        // を経由せず自身のノルムで割るだけで両分岐とも安定して単位長
        // 化できる（`m >= n` 分岐は実測ノルムが `σ` に一致し従来と
        // 同じ結果、`m < n` 分岐は実測ノルムが既に 1 のため実質 no-op）。
        if sigma > 0.0 {
            let col_norm: f64 = u_sorted.col(j).iter().map(|v| v * v).sum::<f64>().sqrt();
            if col_norm > 0.0 {
                for r in 0..u_sorted.rows {
                    let v = u_sorted.get(r, j);
                    u_sorted.set(r, j, v / col_norm);
                }
            }
        }
    }

    // `σ ≈ 0` の列に対応する `U` 列を、既に定めた列群に対し
    // 修正 Gram–Schmidt で直交補完する（決定的な標準基底ベクトルから
    // 出発し、常に同じ結果を再現する）。
    for (j, &sigma) in s_sorted.iter().enumerate() {
        if sigma > 0.0 {
            continue;
        }
        let m_rows = u_sorted.rows;
        let mut candidate = vec![0.0; m_rows];
        // 標準基底ベクトル `e_0, e_1, ...` を順に試し、既存列群と
        // 独立なものを採用する（決定的）。
        let mut chosen = false;
        for basis_idx in 0..m_rows {
            let mut w = vec![0.0; m_rows];
            w[basis_idx] = 1.0;
            for prev in 0..j {
                let prev_col = u_sorted.col(prev);
                let dot: f64 = w.iter().zip(prev_col.iter()).map(|(a, b)| a * b).sum();
                for (idx, wv) in w.iter_mut().enumerate() {
                    *wv -= dot * prev_col[idx];
                }
            }
            let norm: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-9 {
                for wv in w.iter_mut() {
                    *wv /= norm;
                }
                candidate = w;
                chosen = true;
                break;
            }
        }
        if chosen {
            normalize_max_abs_sign(&mut candidate);
            u_sorted.set_col(j, &candidate);
        }
    }

    // `σ ≈ 0` の列に対応する `V` 列も、同じ手続きで直交補完する（上の
    // コメント参照。`m < n` 分岐では `V` がゼロになりうる側のため
    // ここが本質的な修正、`m >= n` 分岐では `V` は既に厳密直交のため
    // no-op に近い——ただし σ=0 の項は `U Σ Vᵀ = A` の再構成に寄与
    // しないため、この補完で上書きしても結果は変わらない）。
    for (j, &sigma) in s_sorted.iter().enumerate() {
        if sigma > 0.0 {
            continue;
        }
        let v_rows = v_sorted.rows;
        let mut candidate = vec![0.0; v_rows];
        let mut chosen = false;
        for basis_idx in 0..v_rows {
            let mut w = vec![0.0; v_rows];
            w[basis_idx] = 1.0;
            for prev in 0..j {
                let prev_col = v_sorted.col(prev);
                let dot: f64 = w.iter().zip(prev_col.iter()).map(|(a, b)| a * b).sum();
                for (idx, wv) in w.iter_mut().enumerate() {
                    *wv -= dot * prev_col[idx];
                }
            }
            let norm: f64 = w.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-9 {
                for wv in w.iter_mut() {
                    *wv /= norm;
                }
                candidate = w;
                chosen = true;
                break;
            }
        }
        if chosen {
            normalize_max_abs_sign(&mut candidate);
            v_sorted.set_col(j, &candidate);
        }
    }

    Ok((
        u_sorted.to_tensor(),
        build_tensor(s_sorted.iter().map(|&v| v as f32).collect(), &[k]),
        v_sorted.transpose().to_tensor(),
    ))
}

// =====================================================================
// 行列ノルム。
// =====================================================================

/// `MatrixNormOrd::Fro`／`One`／`Inf` の 3 種（`Nuc`／`Spectral` は
/// `svd` を要するため `var.rs` 側で分岐する）。
pub(crate) fn matrix_norm_fro(a: &Tensor<f32>) -> Tensor<f32> {
    let sum_sq: f64 = dense_vec(a)
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    build_tensor(vec![sum_sq.sqrt() as f32], &[])
}

pub(crate) fn matrix_norm_one(a: &Tensor<f32>) -> Tensor<f32> {
    let mat = Mat::from_tensor(a);
    let mut max_sum = 0.0f64;
    let mut has_nan = false;
    for c in 0..mat.cols {
        let sum: f64 = mat.col(c).iter().map(|v| v.abs()).sum();
        // `sum` が `NaN` のとき `sum > max_sum` は常に偽（IEEE 754 の
        // 順序付き比較は NaN を含む比較を全て偽にする）ため、`NaN` を
        // 含む列が最大値の更新へ一切寄与せず、無視されたかのように
        // `max_sum` が有限値のまま返ってしまう（codex-review 指摘。
        // 数値異常が正常値 `0` へ変換され隠れる）。列走査とは独立に
        // `NaN` の有無を検出し、1 つでもあれば最終結果を `NaN` へ
        // 伝播させる。
        if sum.is_nan() {
            has_nan = true;
        }
        if sum > max_sum {
            max_sum = sum;
        }
    }
    let result = if has_nan { f64::NAN } else { max_sum };
    build_tensor(vec![result as f32], &[])
}

pub(crate) fn matrix_norm_inf(a: &Tensor<f32>) -> Tensor<f32> {
    let mat = Mat::from_tensor(a);
    let mut max_sum = 0.0f64;
    let mut has_nan = false;
    for r in 0..mat.rows {
        let mut sum = 0.0;
        for c in 0..mat.cols {
            sum += mat.get(r, c).abs();
        }
        // `matrix_norm_one` と同じ理由（NaN 比較は常に偽）で NaN の
        // 有無を独立に検出し最終結果へ伝播する（codex-review 指摘）。
        if sum.is_nan() {
            has_nan = true;
        }
        if sum > max_sum {
            max_sum = sum;
        }
    }
    let result = if has_nan { f64::NAN } else { max_sum };
    build_tensor(vec![result as f32], &[])
}

/// `MatrixNormOrd` の 5 種すべてを扱う `var.rs::Var::matrix_norm` の
/// フォールバック実装（イシュー #1621）。`MatrixNormOrd` は
/// `#[non_exhaustive]`（`tensor-core`）なので、autodiff クレート側の
/// `match` は将来 variant に備え `_` 分岐を持つ（`Activation` の
/// `grad.rs` 分岐と同方針。未知 variant は fail-closed に拒否する）。
pub(crate) fn matrix_norm(
    a: &Tensor<f32>,
    ord: MatrixNormOrd,
) -> Result<Tensor<f32>, AutodiffError> {
    match ord {
        MatrixNormOrd::Fro => Ok(matrix_norm_fro(a)),
        MatrixNormOrd::One => Ok(matrix_norm_one(a)),
        MatrixNormOrd::Inf => Ok(matrix_norm_inf(a)),
        MatrixNormOrd::Nuc => {
            let (_, s, _) = svd(a)?;
            let sum: f64 = dense_vec(&s).iter().map(|&v| f64::from(v)).sum();
            Ok(build_tensor(vec![sum as f32], &[]))
        }
        MatrixNormOrd::Spectral => {
            let (_, s, _) = svd(a)?;
            let max = dense_vec(&s).first().copied().unwrap_or(0.0);
            Ok(build_tensor(vec![max], &[]))
        }
        _ => Err(invalid(format!(
            "eval::linalg::matrix_norm: 未知の MatrixNormOrd variant（{ord:?}）"
        ))),
    }
}

// =====================================================================
// VJP（vector-Jacobian product）補助関数（イシュー #1621。`grad.rs` の
// `Op::Inv`／`Op::Solve`／`Op::Det`／`Op::Cholesky`／`Op::QrQ`／
// `Op::QrR`／`Op::SvdU`／`Op::SvdS`／`Op::SvdVh` から呼ばれる）。
// forward と同じく `f64` 内部計算で固定順序に計算する（本ファイル冒頭
// 「数値契約」参照）。行列積は `tensor-core::BackendOps::gemm_fp32_strict`
// を経由せずここで完結させる（分解サイズが小さい前提の参照実装として、
// 三角解法・特異値スケーリングと同じ `Mat` 型で完結させ精度を統一する
// ため。`docs/autodiff-linalg-design.md` §3.4）。
// =====================================================================

/// `Op::Inv` の VJP: `dA = -(X^T g X^T)`（`X = A^{-1}`）。`X` は
/// forward の `f32` 記録値を再利用せず、`a`（`Op::Inv` の入力）から
/// `f64` の `LuDecomp` で改めて計算する——forward が返す `out_value` は
/// 出力テンソルの契約上 `f32` へ downcast 済みのため、`A` の要素が
/// 極端に小さいスケール（例 `A=[[1e-39]]`）では `X=A^{-1}` が
/// `f32::INFINITY` へ丸められ、本来有限な勾配（`X` の要素が `1/A` に
/// 比例して小さいスケールの `upstream` と相殺し得る）が `Inf`／`NaN`
/// になってしまう（codex-review 指摘。[`det_vjp`] が forward の丸め
/// 済み記録値を再利用しないのと同じ理由）。forward が成功済み（`A`
/// が非特異）である前提のため、ここで `lu_decompose` が失敗するのは
/// forward 側の契約違反のみ。
pub(crate) fn inv_vjp(a: &Tensor<f32>, g: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError> {
    let a_mat = Mat::from_tensor(a);
    let n = a_mat.rows;
    if n == 0 {
        return Ok(build_tensor(Vec::new(), &[0, 0]));
    }
    let lu = lu_decompose(&a_mat)
        .ok_or_else(|| invalid("eval::linalg::inv_vjp: 行列が特異（forward 側の契約違反）"))?;
    let identity = Mat::identity(n);
    // `X = A^{-1}`（`f64` のまま。`f32` へ downcast しない）。
    let x_mat = lu_solve_mat(&lu, &identity);
    let g_mat = Mat::from_tensor(g);
    let xt = x_mat.transpose();
    let mut result = xt.matmul(&g_mat).matmul(&xt);
    for v in result.data.iter_mut() {
        *v = -*v;
    }
    Ok(result.to_tensor())
}

/// `Op::Solve` の VJP: `dB = A^{-T} g`、`dA = -dB Xᵀ`（`X = A^{-1} B`）。
/// `X` は forward の `f32` 記録値を再利用せず、`a`／`b`（`Op::Solve` の
/// 両入力）から `f64` の `LuDecomp` で改めて計算する（[`inv_vjp`] と
/// 同じ理由。例 `A=[[1e-10]]`・`B=[[1e30]]`・`upstream=[[1e-20]]` では
/// forward の `f32` 記録値 `X` が `f32::INFINITY` に丸められ、本来
/// 有限な `dA≈-1e30` が `-Inf` になっていた。codex-review 指摘）。
/// `dB` を `f32` へ downcast する前の `f64` 中間値
/// （[`solve_transposed_mat`]）を `dA` の計算にも使い、両方の勾配を
/// 計算し終えてから 1 回だけ `f32` へ downcast する（別の codex-review
/// 指摘の是正。理由は [`solve_transposed_mat`] のコメント参照）。
pub(crate) fn solve_vjp(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Result<(Tensor<f32>, Tensor<f32>), AutodiffError> {
    let db_mat = solve_transposed_mat(a, g)?;
    let a_mat = Mat::from_tensor(a);
    let b_mat = Mat::from_tensor(b);
    let n = a_mat.rows;
    let x_mat = if n == 0 {
        Mat::zeros(0, b_mat.cols)
    } else {
        let lu = lu_decompose(&a_mat).ok_or_else(|| {
            invalid("eval::linalg::solve_vjp: 係数行列が特異（forward 側の契約違反）")
        })?;
        // `X = A^{-1} B`（`f64` のまま。`f32` へ downcast しない）。
        lu_solve_mat(&lu, &b_mat)
    };
    let xt = x_mat.transpose();
    let mut da = db_mat.matmul(&xt);
    for v in da.data.iter_mut() {
        *v = -*v;
    }
    Ok((da.to_tensor(), db_mat.to_tensor()))
}

/// `Op::Det` の VJP: `dA = g · det(A) · A^{-T}`。`det(A) == 0`（特異）の
/// 場合は `lu_decompose` が `None` を返すため fail-closed に伝播する
/// （設計文書 §3.4「`Det` は … fail-closed」）。
///
/// forward の `det()` が返す `f32` 記録値（`out_value`）は受け取らず、
/// ここで `a` から改めて `f64` の `LuDecomp` を作り、行列式・逆行列の
/// 両方をその 1 回の分解から導出する（`inv(a)` を呼ぶと別途 LU 分解が
/// 走るため二重計算にもなる）。forward の記録値は出力テンソルの契約
/// 上 `f32` へ downcast 済みのため、`|det(A)|` が `f32` の表現範囲
/// （約 3.4e38）を超える入力（例 `diag(1e20, 1e20)` は `det = 1e40`）
/// では forward 出力自体は `f32::INFINITY` になるが、勾配
/// `dA = g · det(A) · A^{-T}` は `A^{-T}` の要素が `det(A)` に反比例
/// して小さくなるため有限になりうる。forward の丸め済み `f32` 値を
/// 乗算に使うと、その時点で無限大が伝播し正しい有限勾配が
/// `Inf`／`NaN` になってしまう（codex-review 指摘）。`f64` のまま
/// `det(A) · A^{-T}` を計算し、最後に 1 回だけ `f32` へ downcast する
/// ことでこれを避ける（`.claude/rules/coding-rust.md` の内部精度契約）。
///
/// `det(A)` 自体の中間値も、単純な `f64` 逐次積ではスケールが極端に
/// 混在する対角（例 `diag([1e30; 11], [1e-30; 11])`。真の行列式は
/// 約 `1.0`）で中間オーバーフローし `Inf`／`NaN` 勾配を生んでいた
/// （codex-review 指摘 PRRT_kwDOTuUCJc6hxiys）。[`det`] と同じ
/// [`lu_diag_product`]（仮数・指数分離）で計算し直す。
pub(crate) fn det_vjp(a: &Tensor<f32>, g_scalar: f32) -> Result<Tensor<f32>, AutodiffError> {
    let mat = Mat::from_tensor(a);
    let n = mat.rows;
    if n == 0 {
        // 空行列の `det` は空積 `1.0`（定数）のため勾配は自明に空。
        return Ok(build_tensor(Vec::new(), &[0, 0]));
    }
    let lu = lu_decompose(&mat)
        .ok_or_else(|| invalid("eval::linalg::det_vjp: 行列が特異（ピボットが厳密 0）"))?;
    let det_value = lu_diag_product(&lu);
    let identity = Mat::identity(n);
    let inv_mat = lu_solve_mat(&lu, &identity);
    let mut result = inv_mat.transpose();
    let scale = f64::from(g_scalar) * det_value;
    for v in result.data.iter_mut() {
        *v *= scale;
    }
    Ok(result.to_tensor())
}

/// `Op::Cholesky` の VJP: `Φ = tril(Lᵀ dL)`（対角 1/2）・
/// `S = L^{-T} Φ L^{-1}`・`dA = (S + Sᵀ)/2`（`L` = forward 記録値）。
pub(crate) fn cholesky_vjp(
    l: &Tensor<f32>,
    dl: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    let l_mat = Mat::from_tensor(l);
    let dl_mat = Mat::from_tensor(dl);
    let n = l_mat.rows;
    let lt = l_mat.transpose();
    let mut phi = lt.matmul(&dl_mat);
    for r in 0..n {
        for c in 0..n {
            match c.cmp(&r) {
                std::cmp::Ordering::Greater => phi.set(r, c, 0.0),
                std::cmp::Ordering::Equal => {
                    let v = phi.get(r, c);
                    phi.set(r, c, v * 0.5);
                }
                std::cmp::Ordering::Less => {}
            }
        }
    }
    let lt_lu = lu_decompose(&lt)
        .ok_or_else(|| invalid("eval::linalg::cholesky_vjp: Lᵀ が特異（forward 側の契約違反）"))?;
    // Y = L^{-T} Φ（`L^T Y = Φ` を解く）。
    let y = lu_solve_mat(&lt_lu, &phi);
    // S = Y L^{-1} を `Sᵀ = L^{-T} Yᵀ`（同じ `L^T` 分解を再利用）として
    // 解き、転置して戻す（`L^{-1}` を明示的に構築しない）。
    let s_t = lu_solve_mat(&lt_lu, &y.transpose());
    let s = s_t.transpose();
    let mut da = Mat::zeros(n, n);
    for r in 0..n {
        for c in 0..n {
            da.set(r, c, 0.5 * (s.get(r, c) + s.get(c, r)));
        }
    }
    Ok(da.to_tensor())
}

/// `Op::QrQ`／`Op::QrR` の VJP: `M = R dRᵀ − dQᵀ Q`・
/// `dA = (dQ + Q copyltu(M)) R^{-T}`（`copyltu` = 下三角を上三角へ複製
/// して対称化）。**`m >= n`（`k = n`）限定**（設計文書 §3.4「QrQ／QrR」・
/// `m < n` は `InvalidArgument`）。
pub(crate) fn qr_vjp(
    q: &Tensor<f32>,
    r: &Tensor<f32>,
    dq: &Tensor<f32>,
    dr: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    let q_mat = Mat::from_tensor(q);
    let r_mat = Mat::from_tensor(r);
    let m = q_mat.rows;
    let k = q_mat.cols;
    let n = r_mat.cols;
    if m < n {
        return Err(invalid(
            "eval::linalg::qr_vjp: m < n（wide 行列）の逆伝播は未対応（設計文書スコープ外）",
        ));
    }
    let dq_mat = Mat::from_tensor(dq);
    let dr_mat = Mat::from_tensor(dr);
    let r_drt = r_mat.matmul(&dr_mat.transpose());
    let dqt_q = dq_mat.transpose().matmul(&q_mat);
    let mut mm = Mat::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            mm.set(i, j, r_drt.get(i, j) - dqt_q.get(i, j));
        }
    }
    // copyltu(M): 下三角（対角含む）を上三角へ複製して対称化する。
    let mut sym = Mat::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            if i >= j {
                sym.set(i, j, mm.get(i, j));
            } else {
                sym.set(i, j, mm.get(j, i));
            }
        }
    }
    let q_sym = q_mat.matmul(&sym);
    let mut inner = Mat::zeros(m, k);
    for i in 0..m {
        for j in 0..k {
            inner.set(i, j, dq_mat.get(i, j) + q_sym.get(i, j));
        }
    }
    // `dA = inner @ R^{-T}` を求める。`Z = inner @ R^{-T}` は
    // `Z Rᵀ = inner` と同値であり、両辺を転置すると `R Zᵀ = innerᵀ` と
    // なるため、`R X = innerᵀ` を解いて `Z = Xᵀ` とする（`(A B)ᵀ = Bᵀ Aᵀ`
    // を踏まえた転置順序。`R^T` ではなく `R` 自身を分解する点に注意——
    // 逆順にすると `inner @ R^{-1}` を計算してしまう。`k == n`〈`m >= n`
    // 前提〉のため `R` は正方）。
    let r_lu = lu_decompose(&r_mat)
        .ok_or_else(|| invalid("eval::linalg::qr_vjp: R が特異（rank 落ち）"))?;
    let x = lu_solve_mat(&r_lu, &inner.transpose());
    Ok(x.transpose().to_tensor())
}

/// `Op::SvdU`／`Op::SvdS`／`Op::SvdVh` の VJP（Townsend 2016 の標準式。
/// `docs/autodiff-linalg-design.md` §3.4「SvdU／SvdS／SvdVh」）。
/// `du`／`ds`／`dvh` はコタンジェント（自ノード以外は `None`＝ゼロ
/// 寄与。多出力ノードの部分寄与設計。`tape::Op::QrQ` doc 参照）。
/// 特異値が近接／重複する場合は `F` 行列の分母が破綻するため
/// `InvalidArgument`（設計文書「近接／重複時は … `InvalidArgument`」）。
pub(crate) fn svd_vjp(
    u: &Tensor<f32>,
    s: &Tensor<f32>,
    vh: &Tensor<f32>,
    du: Option<&Tensor<f32>>,
    ds: Option<&Tensor<f32>>,
    dvh: Option<&Tensor<f32>>,
) -> Result<Tensor<f32>, AutodiffError> {
    let u_mat = Mat::from_tensor(u);
    let vh_mat = Mat::from_tensor(vh);
    let v_mat = vh_mat.transpose();
    let m = u_mat.rows;
    let k = u_mat.cols;
    let n = v_mat.rows;
    let s_vals: Vec<f64> = dense_vec(s).into_iter().map(f64::from).collect();

    let du_mat = du.map(Mat::from_tensor).unwrap_or_else(|| Mat::zeros(m, k));
    let dv_mat = dvh
        .map(|t| Mat::from_tensor(t).transpose())
        .unwrap_or_else(|| Mat::zeros(n, k));
    let ds_vals: Vec<f64> = ds
        .map(|t| dense_vec(t).into_iter().map(f64::from).collect())
        .unwrap_or_else(|| vec![0.0; k]);

    for i in 0..k {
        for j in 0..k {
            if i != j {
                let denom = s_vals[j] * s_vals[j] - s_vals[i] * s_vals[i];
                if denom.abs() < 1e-9 {
                    return Err(invalid(
                        "eval::linalg::svd_vjp: 特異値が近接／重複しているため勾配が未定義（F 行列の分母が破綻）",
                    ));
                }
            }
        }
    }
    let f = |i: usize, j: usize, s_vals: &[f64]| -> f64 {
        if i == j {
            0.0
        } else {
            1.0 / (s_vals[j] * s_vals[j] - s_vals[i] * s_vals[i])
        }
    };

    let utdu = u_mat.transpose().matmul(&du_mat);
    let vtdv = v_mat.transpose().matmul(&dv_mat);

    let mut j_mat = Mat::zeros(k, k);
    let mut k_mat = Mat::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            j_mat.set(i, j, (utdu.get(i, j) - utdu.get(j, i)) * f(i, j, &s_vals));
            k_mat.set(i, j, (vtdv.get(i, j) - vtdv.get(j, i)) * f(i, j, &s_vals));
        }
    }

    let mut inner = Mat::zeros(k, k);
    for i in 0..k {
        for j in 0..k {
            let mut val = j_mat.get(i, j) * s_vals[j] + s_vals[i] * k_mat.get(i, j);
            if i == j {
                val += ds_vals[i];
            }
            inner.set(i, j, val);
        }
    }
    let term1 = u_mat.matmul(&inner).matmul(&vh_mat);

    // term2 = (I_m - U Uᵀ) dU diag(S)^{-1} Vᵀ
    //       = dU diag(S)^{-1} Vᵀ − U (Uᵀ dU diag(S)^{-1}) Vᵀ
    // `U Uᵀ`（`m×m`）・`V Vᵀ`（`n×n`）を明示構築すると reduced SVD
    // （`k = min(m,n)` 列限定）の縮退次元 `m`／`n` が非常に大きい入力
    // （例 `[100000,1]`）でメモリを枯渇させる（`m×m` だけで約 80 GB。
    // codex-review 指摘）。積の結合順序を変え、常に `k×k` 以下の
    // 中間行列のみを経由するよう書き換える（数式としては同値。
    // `(I_m − U Uᵀ) X = X − U (Uᵀ X)` は行列積の結合則そのもの）。
    //
    // `1/σ_j`（`σ_j == 0` は 0）を事前計算して `i`/`j` 二重ループの
    // 内側添字アクセスを配列直接インデクスから外す
    // （clippy::needless_range_loop 回避。`term2`／`term3` で共通
    // 利用する）。forward `svd` の `σ ≈ 0` 判定（PR #1668 是正）と
    // 同じく厳密ゼロ比較とし、固定絶対閾値 `1e-12` を用いない——
    // 閾値があると `1e-13` 等の数値的に小さいが非ゼロな特異値の列が
    // 逆数 0 として扱われ、対応する勾配寄与が黙って消える長方形
    // 行列のケースがあった（codex-review 指摘）。
    let inv_s_vals: Vec<f64> = s_vals
        .iter()
        .map(|&sv| if sv != 0.0 { 1.0 / sv } else { 0.0 })
        .collect();
    // `du`／`dvh` が `None`（自ノード以外のコタンジェント）の場合は
    // `term2`／`term3` がゼロ寄与になることが `du_mat`／`dv_mat` の
    // `Mat::zeros` 初期化から自明なため、`m×n` の中間行列（`du_sinv_
    // vht`・`u_ut_du_sinv_vht` 等）を確保せず計算そのものを省略する
    // （codex-review 指摘: `svd().s.sum()` のように `du`／`dvh` が存在
    // しない場合も従来は実行されており、軽量な forward に続く
    // backward が不要な `O(mn)` 確保を重ねていた）。
    let term2 = if du.is_some() {
        let mut du_sinv = Mat::zeros(m, k);
        for i in 0..m {
            for (j, &inv_s) in inv_s_vals.iter().enumerate() {
                du_sinv.set(i, j, du_mat.get(i, j) * inv_s);
            }
        }
        let du_sinv_vht = du_sinv.matmul(&vh_mat); // m×n
        let ut_du_sinv = u_mat.transpose().matmul(&du_sinv); // k×k
        let u_ut_du_sinv_vht = u_mat.matmul(&ut_du_sinv).matmul(&vh_mat); // m×n
        let mut term2 = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                term2.set(i, j, du_sinv_vht.get(i, j) - u_ut_du_sinv_vht.get(i, j));
            }
        }
        term2
    } else {
        Mat::zeros(m, n)
    };

    // term3 = U diag(S)^{-1} dVᵀ (I_n - V Vᵀ)
    //       = U diag(S)^{-1} dVᵀ − U (diag(S)^{-1} dVᵀ V) Vᵀ
    let term3 = if dvh.is_some() {
        let dv_t = dv_mat.transpose();
        let mut sinv_dvt = Mat::zeros(k, n);
        for (i, &inv_s) in inv_s_vals.iter().enumerate() {
            for j in 0..n {
                sinv_dvt.set(i, j, inv_s * dv_t.get(i, j));
            }
        }
        let u_sinv_dvt = u_mat.matmul(&sinv_dvt); // m×n
        let sinv_dvt_v = sinv_dvt.matmul(&v_mat); // k×k
        let u_sinv_dvt_v_vht = u_mat.matmul(&sinv_dvt_v).matmul(&vh_mat); // m×n
        let mut term3 = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                term3.set(i, j, u_sinv_dvt.get(i, j) - u_sinv_dvt_v_vht.get(i, j));
            }
        }
        term3
    } else {
        Mat::zeros(m, n)
    };

    let mut da = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            da.set(i, j, term1.get(i, j) + term2.get(i, j) + term3.get(i, j));
        }
    }
    Ok(da.to_tensor())
}

/// `Op::MatrixNorm` の VJP（5 ord。設計文書 §3.4「MatrixNorm」）。
/// `a` は入力の forward 値、`g_scalar` は upstream（スカラー）。forward
/// が記録するノルムの `f32` 値は受け取らない（`Fro` 分岐参照: 大きい
/// スケールの入力で `f32` へ丸め済みの記録値を再利用すると `Inf` が
/// 混入するため、必要な統計量は都度 `a` から `f64` で計算し直す）。
pub(crate) fn matrix_norm_vjp(
    a: &Tensor<f32>,
    ord: MatrixNormOrd,
    g_scalar: f32,
) -> Result<Tensor<f32>, AutodiffError> {
    let mat = Mat::from_tensor(a);
    if mat.rows == 0 || mat.cols == 0 {
        // forward（`matrix_norm`）は空行列を受理し `0`（One/Inf/Spectral）
        // または `0`（Fro・Nuc も同様）を返す（`docs/autodiff-linalg-
        // design.md` §3.5「空行列」）。backward はこれに整合させ、
        // 要素を持たない入力形状の勾配（総和は自明に空）をそのまま
        // 返す——`One`／`Inf` の `best_col`／`best_row` 初期値 `0` や
        // `Spectral` の `svd` が返す `k=0` 特異ベクトルへの
        // `mat.get(r, 0)`／`get(0, c)` が、空バッファ（`data.len() ==
        // 0`）を踏み抜いて panic するのを避ける（codex-review 指摘。
        // `.claude/rules/security.md`「本番経路の panic 禁止」）。
        return Ok(Mat::zeros(mat.rows, mat.cols).to_tensor());
    }
    match ord {
        MatrixNormOrd::Fro => {
            // `out_value`（forward が返す `f32` 記録値）を再利用せず、
            // `mat`（`a` の `f64` 表現）から改めて二乗和を計算する。
            // forward `matrix_norm_fro` は `f64` で二乗和を取るものの
            // 最終的に `f32` へ downcast した値を `out_value` として
            // 記録するため、要素スケールが大きい入力（例 `3e38`）では
            // `norm` が `f32::MAX`（約 `3.4e38`）を超えて `f32::INFINITY`
            // に丸まる。`scale = g / norm` にその `Inf` を使うと勾配が
            // 本来有限な値であるにもかかわらず `0` になり異常が隠れる
            // （codex-review 指摘）。`f64` のまま二乗和・平方根を計算し
            // 最後に 1 回だけ downcast することでこれを避ける
            // （`.claude/rules/coding-rust.md` の内部精度契約）。
            let sum_sq: f64 = mat.data.iter().map(|v| v * v).sum();
            let norm = sum_sq.sqrt();
            let mut da = mat.clone();
            if norm.is_nan() {
                // NaN 入力（`sum_sq` が NaN）を「ゼロノルム」分岐（`else`
                // 節）で全勾配 0 にすり替えない。forward（`matrix_norm_
                // fro`）が `sum_sq.sqrt()` をそのまま返すことで NaN を
                // 維持しているのと対称に、backward も NaN を明示的に
                // 伝播する（`norm > 0.0` は NaN に対して常に偽になる
                // ため、この分岐を用意しないと NaN が黙って 0 として
                // 扱われ数値異常が隠れる。codex-review 指摘）。
                for v in da.data.iter_mut() {
                    *v = f64::NAN;
                }
            } else if norm > 0.0 {
                let scale = f64::from(g_scalar) / norm;
                for v in da.data.iter_mut() {
                    *v *= scale;
                }
            } else {
                for v in da.data.iter_mut() {
                    *v = 0.0;
                }
            }
            Ok(da.to_tensor())
        }
        MatrixNormOrd::One => {
            // 最大絶対列和の列（同値は最初の添字）に `g·sign(a)` を置く。
            let mut best_col = 0usize;
            let mut best_sum = f64::MIN;
            let mut has_nan = false;
            for c in 0..mat.cols {
                let sum: f64 = mat.col(c).iter().map(|v| v.abs()).sum();
                // forward（`matrix_norm_one`）と同じ理由（NaN 比較は
                // 常に偽）で、NaN を含む列が「最大列」選択から無条件に
                // 除外され、有限な勾配が返ってしまう（codex-review
                // 指摘）。列走査とは独立に NaN の有無を検出し、1 つでも
                // あれば勾配全体へ NaN を伝播する。
                if sum.is_nan() {
                    has_nan = true;
                }
                if sum > best_sum {
                    best_sum = sum;
                    best_col = c;
                }
            }
            let mut da = Mat::zeros(mat.rows, mat.cols);
            if has_nan {
                for v in da.data.iter_mut() {
                    *v = f64::NAN;
                }
                return Ok(da.to_tensor());
            }
            for r in 0..mat.rows {
                let v = mat.get(r, best_col);
                // `sign(0) == 0`（数学的な符号関数の慣例。`v > 0.0` /
                // `v < 0.0` の二分岐だけだと `v == 0.0` にも `+1` の
                // 符号が割り当たり、設計文書 §3.4 の `g·sign(A)` から
                // 乖離する（codex-review 指摘）。
                let sign = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                da.set(r, best_col, f64::from(g_scalar) * sign);
            }
            Ok(da.to_tensor())
        }
        MatrixNormOrd::Inf => {
            // 最大絶対行和の行（同値は最初の添字）に `g·sign(a)` を置く。
            let mut best_row = 0usize;
            let mut best_sum = f64::MIN;
            let mut has_nan = false;
            for r in 0..mat.rows {
                let mut sum = 0.0;
                for c in 0..mat.cols {
                    sum += mat.get(r, c).abs();
                }
                // `MatrixNormOrd::One` と同じ理由で NaN の有無を独立に
                // 検出し勾配全体へ伝播する（codex-review 指摘）。
                if sum.is_nan() {
                    has_nan = true;
                }
                if sum > best_sum {
                    best_sum = sum;
                    best_row = r;
                }
            }
            let mut da = Mat::zeros(mat.rows, mat.cols);
            if has_nan {
                for v in da.data.iter_mut() {
                    *v = f64::NAN;
                }
                return Ok(da.to_tensor());
            }
            for c in 0..mat.cols {
                let v = mat.get(best_row, c);
                // `MatrixNormOrd::One` と同じ理由で `sign(0) == 0` を
                // 明示する（codex-review 指摘）。
                let sign = if v > 0.0 {
                    1.0
                } else if v < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                da.set(best_row, c, f64::from(g_scalar) * sign);
            }
            Ok(da.to_tensor())
        }
        MatrixNormOrd::Nuc => {
            // dA = g · U Vᵀ（`U`／`Vh` は full ランクの特異ベクトル）。
            let (u, _, vh) = svd(a)?;
            let mut da = Mat::from_tensor(&u).matmul(&Mat::from_tensor(&vh));
            for v in da.data.iter_mut() {
                *v *= f64::from(g_scalar);
            }
            Ok(da.to_tensor())
        }
        MatrixNormOrd::Spectral => {
            // dA = g · u_0 v_0ᵀ（最大特異値に対応する特異ベクトル対）。
            let (u, _, vh) = svd(a)?;
            let u_mat = Mat::from_tensor(&u);
            let vh_mat = Mat::from_tensor(&vh);
            let mut da = Mat::zeros(mat.rows, mat.cols);
            for r in 0..mat.rows {
                let uv = u_mat.get(r, 0);
                for c in 0..mat.cols {
                    da.set(r, c, f64::from(g_scalar) * uv * vh_mat.get(0, c));
                }
            }
            Ok(da.to_tensor())
        }
        _ => Err(invalid(format!(
            "eval::linalg::matrix_norm_vjp: 未知の MatrixNormOrd variant（{ord:?}）"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &Tensor<f32>, b: &Tensor<f32>, tol: f32) {
        assert_eq!(
            a.shape(),
            b.shape(),
            "shape mismatch: {:?} vs {:?}",
            a.shape(),
            b.shape()
        );
        let av = dense_vec(a);
        let bv = dense_vec(b);
        for (x, y) in av.iter().zip(bv.iter()) {
            assert!((x - y).abs() <= tol, "{x} vs {y} (tol={tol})");
        }
    }

    fn mat_matmul(a: &Tensor<f32>, b: &Tensor<f32>) -> Tensor<f32> {
        super::super::matmul(a, b)
    }

    /// `values` を対角成分に持つ正方行列（非対角は 0）を組み立てる
    /// テスト補助（`lu_diag_product` のオーバーフロー回帰テストで使う。
    /// `backend-cpu` クレート `linalg.rs` の同名テスト補助と同一実装）。
    fn diag_tensor(values: &[f32]) -> Tensor<f32> {
        let n = values.len();
        let mut data = vec![0.0f32; n * n];
        for (i, v) in values.iter().enumerate() {
            data[i * n + i] = *v;
        }
        build_tensor(data, &[n, n])
    }

    #[test]
    fn inv_2x2_matches_known_solution() {
        // A = [[4, 7], [2, 6]] -> A^{-1} = 1/10 * [[6, -7], [-2, 4]]
        let a = build_tensor(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]);
        let inv_a = inv(&a).unwrap();
        let expected = build_tensor(vec![0.6, -0.7, -0.2, 0.4], &[2, 2]);
        approx_eq(&inv_a, &expected, 1e-5);
    }

    #[test]
    fn inv_times_a_is_identity() {
        let a = build_tensor(vec![1.0, 2.0, 3.0, 4.0, 6.0, 5.0, 7.0, 1.0, 9.0], &[3, 3]);
        let inv_a = inv(&a).unwrap();
        let product = mat_matmul(&a, &inv_a);
        let identity = build_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
        approx_eq(&product, &identity, 1e-4);
    }

    #[test]
    fn inv_singular_is_invalid_argument() {
        let a = build_tensor(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
        assert!(matches!(inv(&a), Err(AutodiffError::InvalidArgument(_))));
    }

    #[test]
    fn inv_empty_matrix_returns_empty() {
        let a = build_tensor(Vec::new(), &[0, 0]);
        let result = inv(&a).unwrap();
        assert_eq!(result.shape(), &[0, 0]);
    }

    #[test]
    fn solve_matches_inv_times_b() {
        let a = build_tensor(vec![3.0, 1.0, 1.0, 2.0], &[2, 2]);
        let b = build_tensor(vec![9.0, 8.0], &[2, 1]);
        let x = solve(&a, &b).unwrap();
        let expected = build_tensor(vec![2.0, 3.0], &[2, 1]);
        approx_eq(&x, &expected, 1e-4);
    }

    #[test]
    fn det_2x2_known_value() {
        let a = build_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let d = det(&a);
        approx_eq(&d, &build_tensor(vec![-2.0], &[]), 1e-5);
    }

    #[test]
    fn det_singular_is_zero() {
        let a = build_tensor(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]);
        let d = det(&a);
        approx_eq(&d, &build_tensor(vec![0.0], &[]), 1e-6);
    }

    #[test]
    fn det_empty_matrix_is_one() {
        let a = build_tensor(Vec::new(), &[0, 0]);
        let d = det(&a);
        approx_eq(&d, &build_tensor(vec![1.0], &[]), 1e-6);
    }

    /// codex-review 指摘 PRRT_kwDOTuUCJc6hxiys の回帰:
    /// `diag([1e30; 11], [1e-30; 11])` の真の行列式は約 `1.0` だが、
    /// LU 対角積の単純な `f64` 逐次積では 11 個の `1e30` を掛けた時点で
    /// `f64` の表現範囲（約 `1.8e308`）を超えて `Inf` になり、続く
    /// `1e-30` を掛けても `Inf` のまま戻らなかった。`lu_diag_product`
    /// （仮数・指数分離方式）へ切替後は中間オーバーフローを避け、
    /// 有限な値（約 `1.0`）を返すことを検証する。
    #[test]
    fn det_extreme_scale_diag_does_not_overflow() {
        let mut values = vec![1e30f32; 11];
        values.extend(std::iter::repeat_n(1e-30f32, 11));
        let a = diag_tensor(&values);
        let v = dense_vec(&det(&a))[0];
        assert!(v.is_finite(), "det must be finite: {v}");
        assert!((v - 1.0).abs() < 1e-2, "det should be approx 1.0: {v}");
    }

    /// 上記の対角順序を逆にした回帰（codex-review 指摘: 順序を逆に
    /// すると単純な逐次積では `0.0` になっていた）。順序に依らず
    /// 約 `1.0` を返すことを検証する。
    #[test]
    fn det_extreme_scale_diag_reversed_order_does_not_underflow() {
        let mut values = vec![1e-30f32; 11];
        values.extend(std::iter::repeat_n(1e30f32, 11));
        let a = diag_tensor(&values);
        let v = dense_vec(&det(&a))[0];
        assert!(v.is_finite(), "det must be finite: {v}");
        assert_ne!(v, 0.0, "det should not underflow to zero: {v}");
        assert!((v - 1.0).abs() < 1e-2, "det should be approx 1.0: {v}");
    }

    #[test]
    fn cholesky_reconstructs_spd_matrix() {
        // A = B Bᵀ + n I は必ず対称正定値。
        let b = build_tensor(vec![1.0, 0.5, 0.2, 1.5, 0.3, 0.7], &[3, 2]);
        let bt = b.transpose(0, 1).unwrap().contiguous();
        let bbt = mat_matmul(&b, &bt);
        let mut spd = dense_vec(&bbt);
        for i in 0..3 {
            spd[i * 3 + i] += 3.0;
        }
        let spd_t = build_tensor(spd, &[3, 3]);
        let l = cholesky(&spd_t).unwrap();
        let lt = l.transpose(0, 1).unwrap().contiguous();
        let reconstructed = mat_matmul(&l, &lt);
        approx_eq(&reconstructed, &spd_t, 1e-4);
    }

    #[test]
    fn cholesky_non_positive_definite_is_invalid() {
        let a = build_tensor(vec![1.0, 2.0, 2.0, 1.0], &[2, 2]);
        assert!(matches!(
            cholesky(&a),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }

    #[test]
    fn qr_reconstructs_input_and_is_orthonormal() {
        let a = build_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let (q, r) = qr(&a);
        assert_eq!(q.shape(), &[3, 2]);
        assert_eq!(r.shape(), &[2, 2]);
        let reconstructed = mat_matmul(&q, &r);
        approx_eq(&reconstructed, &a, 1e-4);

        let qt = q.transpose(0, 1).unwrap().contiguous();
        let qtq = mat_matmul(&qt, &q);
        let identity = build_tensor(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]);
        approx_eq(&qtq, &identity, 1e-4);
    }

    #[test]
    fn qr_r_diagonal_is_nonnegative() {
        let a = build_tensor(vec![-1.0, 2.0, 3.0, -4.0, 5.0, 6.0], &[3, 2]);
        let (_, r) = qr(&a);
        let r_data = dense_vec(&r);
        assert!(r_data[0] >= 0.0);
        assert!(r_data[3] >= 0.0);
    }

    /// codex-review 指摘の回帰: `qr` は以前 `Mat::identity(m)`（`m×m` の
    /// `f64` 行列）を明示構築しており、`[100000,1]` のような縦長入力
    /// （データ自体は約 400 KB）でも中間 `Q` だけで約 80 GB を要求し
    /// メモリ枯渇を招いた。O(mk) メモリの反射ベクトル蓄積方式へ是正
    /// 済みで、この形状でも数秒以内に完了することを確認する
    /// （`docs/autodiff-linalg-design.md` §3.4「QR」）。
    #[test]
    fn qr_tall_matrix_does_not_allocate_full_m_by_m_intermediate() {
        let m = 100_000usize;
        let data: Vec<f32> = (0..m).map(|i| 1.0 + (i % 7) as f32).collect();
        let a = build_tensor(data, &[m, 1]);
        let (q, r) = qr(&a);
        assert_eq!(q.shape(), &[m, 1]);
        assert_eq!(r.shape(), &[1, 1]);

        let q_data = dense_vec(&q);
        let norm: f64 = q_data.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!(
            (norm.sqrt() - 1.0).abs() < 1e-3,
            "Q 列が単位ノルムでない: norm={norm}"
        );

        // `Q R = A` を数点サンプルして確認する（`k=1` のため
        // `q[i] * r[0]` が対応する `a[i]` に一致するはず）。
        let r_data = dense_vec(&r);
        let a_data = dense_vec(&a);
        for &i in &[0usize, 1, m / 2, m - 1] {
            let reconstructed = q_data[i] * r_data[0];
            assert!(
                (reconstructed - a_data[i]).abs() < 1e-2,
                "行 {i}: 再構成 {reconstructed} が入力 {} と乖離",
                a_data[i]
            );
        }
    }

    #[test]
    fn svd_reconstructs_input() {
        // A = U diag(3,2,1) Vᵀ で意図的に構成した既知の可分行列。
        let u = build_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
        let s = [3.0f32, 2.0, 1.0];
        let vh = build_tensor(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
        let mut a_data = vec![0.0f32; 9];
        for i in 0..3 {
            a_data[i * 3 + i] = s[i];
        }
        let _ = (u, vh);
        let a = build_tensor(a_data, &[3, 3]);
        let (u_out, s_out, vh_out) = svd(&a).unwrap();
        assert_eq!(u_out.shape(), &[3, 3]);
        assert_eq!(s_out.shape(), &[3]);
        assert_eq!(vh_out.shape(), &[3, 3]);
        let s_data = dense_vec(&s_out);
        assert!(s_data[0] >= s_data[1] && s_data[1] >= s_data[2]);
        approx_eq(&s_out, &build_tensor(vec![3.0, 2.0, 1.0], &[3]), 1e-4);

        let u_s = {
            let mut out = vec![0.0f32; 9];
            for r in 0..3 {
                for c in 0..3 {
                    out[r * 3 + c] = dense_vec(&u_out)[r * 3 + c] * s_data[c];
                }
            }
            build_tensor(out, &[3, 3])
        };
        let reconstructed = mat_matmul(&u_s, &vh_out);
        approx_eq(&reconstructed, &a, 1e-3);
    }

    /// codex-review 指摘の回帰: Jacobi の収束判定に絶対下限
    /// `.max(EPS)` があると `JACOBI_EPS² = 1e-28` という入力スケール
    /// 非依存の閾値が生じ、列ノルムが約 1e-15 スケールの入力
    /// （`A = 1e-15 * [[1,1],[0,1]]`）で非直交な列を誤って収束扱いし、
    /// 誤った特異値（`[1.414e-15, 1e-15]`）を返していた。相対収束判定
    /// へ是正後は正しい特異値（黄金比由来の
    /// `[1.618...e-15, 0.618...e-15]`）を返す。
    #[test]
    fn svd_tiny_scale_singular_values_are_accurate() {
        let scale = 1e-15f32;
        let a = build_tensor(vec![scale, scale, 0.0, scale], &[2, 2]);
        let (_, s, _) = svd(&a).unwrap();
        let s_data = dense_vec(&s);
        let expected0 = ((3.0 + 5.0f64.sqrt()) / 2.0).sqrt() * 1e-15;
        let expected1 = ((3.0 - 5.0f64.sqrt()) / 2.0).sqrt() * 1e-15;
        let rel_err0 = (f64::from(s_data[0]) - expected0).abs() / expected0;
        let rel_err1 = (f64::from(s_data[1]) - expected1).abs() / expected1;
        assert!(
            rel_err0 < 1e-2,
            "第一特異値が期待値から乖離: {} vs {expected0}",
            s_data[0]
        );
        assert!(
            rel_err1 < 1e-2,
            "第二特異値が期待値から乖離: {} vs {expected1}",
            s_data[1]
        );
    }

    /// codex-review 指摘の回帰: 零特異値の `V` 列を Gram–Schmidt で
    /// 直交補完した後にも「最大絶対値成分が正」という符号規約（設計
    /// 文書 §3.5）が保たれることを確認する。`A=[[2,1],[0,0]]` は
    /// 第二特異値が厳密 0 で、補完前に一度符号正規化した `V` 列を
    /// 補完ループが無条件に上書きすると規約が崩れていた。
    #[test]
    fn svd_zero_sigma_gram_schmidt_column_respects_sign_convention() {
        let a = build_tensor(vec![2.0, 1.0, 0.0, 0.0], &[2, 2]);
        let (_, s, vh) = svd(&a).unwrap();
        let s_data = dense_vec(&s);
        assert!(
            s_data[1].abs() < 1e-6,
            "第二特異値は厳密 0 のはず: {s_data:?}"
        );
        let vh_data = dense_vec(&vh);
        // `Vh` の第 2 行（`vh_data[2]`,`vh_data[3]`）が零特異値に対応する
        // `V` 列（の転置）。最大絶対値成分が非負であるべき。
        let col = [vh_data[2], vh_data[3]];
        let max_abs_idx = if col[0].abs() >= col[1].abs() { 0 } else { 1 };
        assert!(
            col[max_abs_idx] >= 0.0,
            "補完後の V 列が符号規約（最大絶対値成分が正）に違反: {col:?}"
        );
    }

    #[test]
    fn svd_wide_matrix_reconstructs_input() {
        let a = build_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let (u, s, vh) = svd(&a).unwrap();
        assert_eq!(u.shape(), &[2, 2]);
        assert_eq!(s.shape(), &[2]);
        assert_eq!(vh.shape(), &[2, 3]);
        let s_data = dense_vec(&s);
        let u_s = {
            let mut out = vec![0.0f32; 4];
            let u_data = dense_vec(&u);
            for r in 0..2 {
                for c in 0..2 {
                    out[r * 2 + c] = u_data[r * 2 + c] * s_data[c];
                }
            }
            build_tensor(out, &[2, 2])
        };
        let reconstructed = mat_matmul(&u_s, &vh);
        approx_eq(&reconstructed, &a, 1e-3);
    }

    /// `σ` が固定閾値 `1e-12` 未満でも非ゼロならゼロ特異値として扱わず、
    /// 独立基底で上書きしないことを確認する（codex-review 指摘。修正前は
    /// `A=[[-1e-13]]` で `U`/`Vh` の符号・再構成が誤っていた——σ<=1e-12
    /// の非ゼロ特異値が固定絶対閾値でゼロ扱いされ Gram–Schmidt 補完の
    /// 独立基底へ置き換わり `U Σ Vᵀ = A` を満たさなくなる不具合。設計
    /// §3.5「σ==0 の列のみ補完」契約に合わせ、判定を厳密ゼロ比較へ変更
    /// した回帰）。
    #[test]
    fn svd_tiny_nonzero_singular_value_reconstructs_input() {
        let a = build_tensor(vec![-1e-13], &[1, 1]);
        let (u, s, vh) = svd(&a).unwrap();
        let s_data = dense_vec(&s);
        assert!(
            s_data[0] > 0.0,
            "非ゼロ特異値がゼロ扱いされている: {}",
            s_data[0]
        );
        let u_s = {
            let out = vec![dense_vec(&u)[0] * s_data[0]];
            build_tensor(out, &[1, 1])
        };
        let reconstructed = mat_matmul(&u_s, &vh);
        approx_eq(&reconstructed, &a, 1e-18);
    }

    #[test]
    fn matrix_norm_fro_known_value() {
        let a = build_tensor(vec![3.0, 4.0], &[1, 2]);
        approx_eq(&matrix_norm_fro(&a), &build_tensor(vec![5.0], &[]), 1e-5);
    }

    #[test]
    fn matrix_norm_one_and_inf_known_values() {
        let a = build_tensor(vec![1.0, -2.0, -3.0, 4.0], &[2, 2]);
        // 列和絶対値: col0 = |1|+|-3| = 4, col1 = |-2|+|4| = 6 -> One = 6
        approx_eq(&matrix_norm_one(&a), &build_tensor(vec![6.0], &[]), 1e-5);
        // 行和絶対値: row0 = |1|+|-2| = 3, row1 = |-3|+|4| = 7 -> Inf = 7
        approx_eq(&matrix_norm_inf(&a), &build_tensor(vec![7.0], &[]), 1e-5);
    }

    /// codex-review 指摘の回帰: `sum > max_sum` は NaN に対して常に偽
    /// （IEEE 754 の順序付き比較）のため、NaN を含む唯一の列・行が
    /// 最大値の更新へ一切寄与せず `[[NaN]]` のノルムが正常な `0` として
    /// 返ってしまっていた。
    #[test]
    fn matrix_norm_one_and_inf_propagate_nan() {
        let a = build_tensor(vec![f32::NAN], &[1, 1]);
        assert!(dense_vec(&matrix_norm_one(&a))[0].is_nan());
        assert!(dense_vec(&matrix_norm_inf(&a))[0].is_nan());
    }

    // =====================================================================
    // VJP 数値微分検証（イシュー #1621）。`grad.rs::tests` の grad-check
    // 専用定数（`H`/`TAU`/`REL_TOL`/`ABS_TOL`。#223 承認済み）と同一値を
    // ここでも再利用する（新たな許容誤差を導入しない。`.claude/rules/
    // delegation-impl.md`「テスト許容誤差の変更はユーザー承認必須」）。
    // =====================================================================

    const GRAD_H: f64 = 1e-3;
    const GRAD_TAU: f32 = 1e-4;
    const GRAD_REL_TOL: f32 = 1e-2;
    const GRAD_ABS_TOL: f32 = 1e-3;

    fn scalar_dot(a: &Tensor<f32>, s: &Tensor<f32>) -> f64 {
        dense_vec(a)
            .iter()
            .zip(dense_vec(s).iter())
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
        let a = dense_vec(analytic);
        let n = dense_vec(numeric);
        assert_eq!(
            a.len(),
            n.len(),
            "{label}: analytic/numeric の要素数が一致しない"
        );
        for (i, (&av, &nv)) in a.iter().zip(n.iter()).enumerate() {
            let diff = (av - nv).abs();
            let rel = diff / av.abs().max(nv.abs()).max(GRAD_TAU);
            assert!(
                rel <= GRAD_REL_TOL || diff <= GRAD_ABS_TOL,
                "{label}[{i}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
            );
        }
    }

    /// `a` の各要素を独立変数として中央差分する（`grad.rs::tests::
    /// numeric_grad_unary` と同型。対称行列前提の Cholesky のみ
    /// `numeric_grad_symmetric` を使う）。
    fn numeric_grad(a: &Tensor<f32>, loss: impl Fn(&Tensor<f32>) -> f64) -> Tensor<f32> {
        let shape = a.shape().to_vec();
        let mut data: Vec<f64> = dense_vec(a).into_iter().map(f64::from).collect();
        let mut grad = vec![0f32; data.len()];
        for i in 0..data.len() {
            let orig = data[i];
            data[i] = orig + GRAD_H;
            let lp = loss(&build_tensor(
                data.iter().map(|&v| v as f32).collect(),
                &shape,
            ));
            data[i] = orig - GRAD_H;
            let lm = loss(&build_tensor(
                data.iter().map(|&v| v as f32).collect(),
                &shape,
            ));
            data[i] = orig;
            grad[i] = ((lp - lm) / (2.0 * GRAD_H)) as f32;
        }
        build_tensor(grad, &shape)
    }

    /// Cholesky は対称行列（下三角のみ読む）を前提とするため、`(i,j)`・
    /// `(j,i)` を対で摂動する（forward が上三角を読まないため単独摂動
    /// では数値勾配が構造的に 0 になり、対称化した解析勾配
    /// `(S+Sᵀ)/2` と比較不能なため。設計文書 §3.5「Cholesky は下三角
    /// のみ」）。
    fn numeric_grad_symmetric(a: &Tensor<f32>, loss: impl Fn(&Tensor<f32>) -> f64) -> Tensor<f32> {
        let mat = Mat::from_tensor(a);
        let n = mat.rows;
        let mut data = mat.data.clone();
        let mut grad = Mat::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let idx1 = i * n + j;
                let idx2 = j * n + i;
                let orig1 = data[idx1];
                let orig2 = data[idx2];
                data[idx1] = orig1 + GRAD_H;
                data[idx2] = orig2 + GRAD_H;
                let ap = Mat {
                    data: data.clone(),
                    rows: n,
                    cols: n,
                }
                .to_tensor();
                let lp = loss(&ap);
                data[idx1] = orig1 - GRAD_H;
                data[idx2] = orig2 - GRAD_H;
                let am = Mat {
                    data: data.clone(),
                    rows: n,
                    cols: n,
                }
                .to_tensor();
                let lm = loss(&am);
                data[idx1] = orig1;
                data[idx2] = orig2;
                let d = (lp - lm) / (2.0 * GRAD_H);
                // `i != j` は `(i,j)`／`(j,i)` を同時に `+H` 摂動した
                // ため `d` は両変数分の合算感度（forward が上三角を
                // 読まないため実質 `dL/da[j,i]` のみ）。解析値
                // `(S+Sᵀ)/2` は対の各片方に半分ずつ割り当てる規約の
                // ため、比較対象もここで半分にする（対角 `i == j` は
                // 単一変数のため半分にしない）。
                let value = if i == j { d } else { d / 2.0 };
                grad.set(i, j, value);
                grad.set(j, i, value);
            }
        }
        grad.to_tensor()
    }

    #[test]
    fn inv_vjp_matches_numeric() {
        let a = build_tensor(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
        let s = build_tensor(vec![1.0, -0.5, 0.3, 2.0], &[2, 2]);
        let da = inv_vjp(&a, &s).unwrap();
        let numeric = numeric_grad(&a, |ap| scalar_dot(&inv(ap).unwrap(), &s));
        assert_grad_close("inv", &da, &numeric);
    }

    #[test]
    fn solve_vjp_matches_numeric() {
        let a = build_tensor(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
        let b = build_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = build_tensor(vec![1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let (da, db) = solve_vjp(&a, &b, &s).unwrap();
        let num_da = numeric_grad(&a, |ap| scalar_dot(&solve(ap, &b).unwrap(), &s));
        let num_db = numeric_grad(&b, |bp| scalar_dot(&solve(&a, bp).unwrap(), &s));
        assert_grad_close("solve dA", &da, &num_da);
        assert_grad_close("solve dB", &db, &num_db);
    }

    /// `dB` を早期に `f32` へ downcast すると overflow して有限な `dA`
    /// まで `-Inf` になっていた不具合の回帰（codex-review 指摘）。
    /// `A=[[1e-20]]`・`X=solve(A,B)`・`upstream=1e20` では
    /// `dB = A^{-T} g ≈ 1e40`（`f32::INFINITY`）だが、`dA = -dB Xᵀ` は
    /// `X ≈ 1e-10` オーダーのため有限（`≈ -1e30`）であるべき。
    #[test]
    fn solve_vjp_da_stays_finite_when_db_overflows_f32() {
        let a = build_tensor(vec![1e-20], &[1, 1]);
        let b = build_tensor(vec![1e-30], &[1, 1]);
        let g = build_tensor(vec![1e20], &[1, 1]);
        let (da, db) = solve_vjp(&a, &b, &g).unwrap();
        let db_val = dense_vec(&db)[0];
        assert!(
            db_val.is_infinite(),
            "この極端なスケールでは dB 自体は f32 表現域を超えるはず: {db_val}"
        );
        let da_val = dense_vec(&da)[0];
        assert!(
            da_val.is_finite(),
            "dB の overflow が dA まで伝播してはいけない: da={da_val}"
        );
    }

    /// codex-review 指摘の回帰: `solve_vjp` が forward の `f32` に丸めた
    /// `X`（`Op::Inv`／`Op::Solve` の記録値）を再利用していた場合、
    /// `A=[[1e-10]]`・`B=[[1e30]]` では `X = A^{-1} B = 1e40` が forward
    /// 時点で `f32::INFINITY` に丸められ、`upstream=[[1e-20]]` に対する
    /// `dA = -dB Xᵀ`（本来 `dB≈1e-10`・`dA≈-1e30` で両方有限）が `-Inf`
    /// になっていた。`b` から `f64` で `X` を再計算する修正後は
    /// `dA` が有限のまま。
    #[test]
    fn solve_vjp_da_stays_finite_when_x_would_overflow_f32() {
        let a = build_tensor(vec![1e-10], &[1, 1]);
        let b = build_tensor(vec![1e30], &[1, 1]);
        let g = build_tensor(vec![1e-20], &[1, 1]);
        let (da, db) = solve_vjp(&a, &b, &g).unwrap();
        let db_val = dense_vec(&db)[0];
        assert!(db_val.is_finite(), "dB は本来有限のはず: {db_val}");
        let da_val = dense_vec(&da)[0];
        assert!(
            da_val.is_finite(),
            "forward の f32 丸め済み X の再利用により dA が Inf 化してはいけない: da={da_val}"
        );
        assert!(
            (da_val - (-1e30)).abs() / 1e30 < 1e-3,
            "dA が期待値 -1e30 から乖離: da={da_val}"
        );
    }

    /// codex-review 指摘の回帰（[`solve_vjp_da_stays_finite_when_x_would_
    /// overflow_f32`] の `inv_vjp` 版）: `A=[[1e-39]]`（f32 劣正規化数の
    /// 表現域内）では `X = A^{-1} = 1e39` が forward 時点で `f32::
    /// INFINITY` に丸められ、`upstream=[[1e-40]]` に対する
    /// `dA = -(X g X)`（本来 `≈ -1e38` で有限）が `-Inf` になっていた。
    /// `a` から `f64` で `X` を再計算する修正後は `dA` が有限のまま。
    #[test]
    fn inv_vjp_stays_finite_when_x_would_overflow_f32() {
        let a = build_tensor(vec![1e-39], &[1, 1]);
        let g = build_tensor(vec![1e-40], &[1, 1]);
        let da = inv_vjp(&a, &g).unwrap();
        let da_val = dense_vec(&da)[0];
        assert!(
            da_val.is_finite(),
            "forward の f32 丸め済み X の再利用により dA が Inf 化してはいけない: da={da_val}"
        );
        assert!(
            (da_val - (-1e38)).abs() / 1e38 < 1e-3,
            "dA が期待値 -1e38 から乖離: da={da_val}"
        );
    }

    #[test]
    fn det_vjp_matches_numeric() {
        let a = build_tensor(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
        let g_scalar = 1.5f32;
        let da = det_vjp(&a, g_scalar).unwrap();
        let numeric = numeric_grad(&a, |ap| {
            f64::from(g_scalar) * f64::from(dense_vec(&det(ap))[0])
        });
        assert_grad_close("det", &da, &numeric);
    }

    #[test]
    fn det_vjp_large_scale_input_is_finite() {
        // 診断済み P2: forward `det()` の `f32` 記録値をそのまま乗算に
        // 使うと `diag(1e20, 1e20)`（`det = 1e40`）で `f32::INFINITY`
        // が伝播し `dA` が `Inf`／`NaN` になっていた（codex-review
        // 指摘）。`det_vjp` は `a` から `f64` で改めて分解するため、
        // `A^{-T}` の要素（約 `1e-20`）との積で有限勾配になることを
        // 検証する。
        let a = build_tensor(vec![1e20, 0.0, 0.0, 1e20], &[2, 2]);
        let da = det_vjp(&a, 1.0).unwrap();
        for v in dense_vec(&da) {
            assert!(
                v.is_finite(),
                "det_vjp large-scale grad must be finite: {v}"
            );
        }
    }

    /// codex-review 指摘 PRRT_kwDOTuUCJc6hxiys の回帰: `det_vjp` 内部の
    /// LU 対角積も `det` と同じ中間オーバーフローの影響を受け、
    /// `diag([1e30; 11], [1e-30; 11])`（真の行列式は約 `1.0`）で本来
    /// 有限な勾配が `Inf`／`NaN`／全ゼロになっていた。`det` と共有する
    /// `lu_diag_product`（仮数・指数分離方式）へ切替後は、対角行列
    /// `dA = diag(det(A)/d_i)`（非対角は 0）がすべて有限であることを
    /// 検証する。
    #[test]
    fn det_vjp_extreme_scale_diag_is_finite() {
        let mut values = vec![1e30f32; 11];
        values.extend(std::iter::repeat_n(1e-30f32, 11));
        let a = diag_tensor(&values);
        let da = det_vjp(&a, 1.0).unwrap();
        for v in dense_vec(&da) {
            assert!(
                v.is_finite(),
                "det_vjp grad must be finite for extreme-scale diag: {v}"
            );
        }
    }

    #[test]
    fn cholesky_vjp_matches_numeric() {
        let b = build_tensor(vec![1.0, 0.5, 0.2, 1.5, 0.3, 0.7], &[3, 2]);
        let bt = b.transpose(0, 1).unwrap().contiguous();
        let bbt = mat_matmul(&b, &bt);
        let mut spd = dense_vec(&bbt);
        for i in 0..3 {
            spd[i * 3 + i] += 3.0;
        }
        let a = build_tensor(spd, &[3, 3]);
        let s = build_tensor(vec![1.0, 0.0, 0.0, 0.3, -0.5, 0.0, 0.2, 0.4, 0.6], &[3, 3]);
        let l = cholesky(&a).unwrap();
        let da = cholesky_vjp(&l, &s).unwrap();
        let numeric = numeric_grad_symmetric(&a, |ap| scalar_dot(&cholesky(ap).unwrap(), &s));
        assert_grad_close("cholesky", &da, &numeric);
    }

    #[test]
    fn qr_vjp_matches_numeric() {
        // m >= n（設計文書スコープ）。
        let a = build_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 7.0], &[3, 2]);
        let sq = build_tensor(vec![1.0, -0.5, 0.3, 0.7, -0.2, 0.4], &[3, 2]);
        let sr = build_tensor(vec![0.5, -1.0, 0.0, 0.3], &[2, 2]);
        let (q, r) = qr(&a);
        let da = qr_vjp(&q, &r, &sq, &sr).unwrap();
        let numeric = numeric_grad(&a, |ap| {
            let (qp, rp) = qr(ap);
            scalar_dot(&qp, &sq) + scalar_dot(&rp, &sr)
        });
        assert_grad_close("qr combined", &da, &numeric);
    }

    #[test]
    fn svd_vjp_matches_numeric() {
        // U diag(3,2,1) Vᵀ 型の意図的に特異値を離した可分行列
        // （近接／重複回避。設計文書 §3.4「相異なる特異値の前提」）。
        let a = build_tensor(vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0], &[3, 3]);
        let su = build_tensor(
            vec![0.5, -0.3, 0.2, 0.1, 0.4, -0.2, 0.3, -0.1, 0.2],
            &[3, 3],
        );
        let ss = build_tensor(vec![1.0, -0.5, 0.3], &[3]);
        let svh = build_tensor(
            vec![0.2, -0.4, 0.1, 0.3, 0.2, -0.3, -0.1, 0.5, 0.4],
            &[3, 3],
        );
        let (u, s, vh) = svd(&a).unwrap();
        let da = svd_vjp(&u, &s, &vh, Some(&su), Some(&ss), Some(&svh)).unwrap();
        let numeric = numeric_grad(&a, |ap| {
            let (up, sp, vhp) = svd(ap).unwrap();
            scalar_dot(&up, &su) + scalar_dot(&sp, &ss) + scalar_dot(&vhp, &svh)
        });
        assert_grad_close("svd combined", &da, &numeric);
    }

    #[test]
    fn matrix_norm_fro_vjp_matches_numeric() {
        let a = build_tensor(vec![1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let g_scalar = 1.3f32;
        let da = matrix_norm_vjp(&a, MatrixNormOrd::Fro, g_scalar).unwrap();
        let numeric = numeric_grad(&a, |ap| {
            f64::from(g_scalar) * f64::from(dense_vec(&matrix_norm_fro(ap))[0])
        });
        assert_grad_close("matrix_norm fro", &da, &numeric);
    }

    #[test]
    fn matrix_norm_one_vjp_matches_numeric() {
        // 列和絶対値 col0=|2|+|-6|=8・col1=|-1|+|3|=4 と明確に差を
        // 付けた入力（タイ回避。旧フィクスチャ [1,-2,-5,4] は col0=6・
        // col1=6 で偶然タイし、numeric_grad が境界を跨いで不成立に
        // なることが判明したため変更）。
        let a = build_tensor(vec![2.0, -1.0, -6.0, 3.0], &[2, 2]);
        let g_scalar = 0.7f32;
        let da = matrix_norm_vjp(&a, MatrixNormOrd::One, g_scalar).unwrap();
        let numeric = numeric_grad(&a, |ap| {
            f64::from(g_scalar) * f64::from(dense_vec(&matrix_norm_one(ap))[0])
        });
        assert_grad_close("matrix_norm one", &da, &numeric);
    }

    /// codex-review 指摘の回帰: 符号判定が `v >= 0.0` のみだと、ゼロ
    /// 要素にも `+1` の符号（＝upstream と同じ勾配）が割り当たり、
    /// 設計文書 §3.4 の `g·sign(A)`（`sign(0) == 0`）から乖離する。
    /// `A=[[1],[0]]` の One ノルム勾配は `[[1],[0]]` であるべき
    /// （旧実装は `[[1],[1]]` を返していた）。
    #[test]
    fn matrix_norm_one_vjp_zero_element_has_zero_gradient() {
        let a = build_tensor(vec![1.0, 0.0], &[2, 1]);
        let da = matrix_norm_vjp(&a, MatrixNormOrd::One, 1.0).unwrap();
        approx_eq(&da, &build_tensor(vec![1.0, 0.0], &[2, 1]), 1e-6);
    }

    #[test]
    fn matrix_norm_inf_vjp_matches_numeric() {
        let a = build_tensor(vec![1.0, -2.0, -5.0, 4.0], &[2, 2]);
        let g_scalar = -0.9f32;
        let da = matrix_norm_vjp(&a, MatrixNormOrd::Inf, g_scalar).unwrap();
        let numeric = numeric_grad(&a, |ap| {
            f64::from(g_scalar) * f64::from(dense_vec(&matrix_norm_inf(ap))[0])
        });
        assert_grad_close("matrix_norm inf", &da, &numeric);
    }

    #[test]
    fn matrix_norm_nuc_vjp_matches_numeric() {
        let a = build_tensor(vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0], &[3, 3]);
        let g_scalar = 1.1f32;
        let norm_fn = |ap: &Tensor<f32>| -> f64 {
            let (_, s, _) = svd(ap).unwrap();
            dense_vec(&s).iter().map(|&v| f64::from(v)).sum()
        };
        let da = matrix_norm_vjp(&a, MatrixNormOrd::Nuc, g_scalar).unwrap();
        let numeric = numeric_grad(&a, |ap| f64::from(g_scalar) * norm_fn(ap));
        assert_grad_close("matrix_norm nuc", &da, &numeric);
    }

    #[test]
    fn matrix_norm_spectral_vjp_matches_numeric() {
        let a = build_tensor(vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0], &[3, 3]);
        let g_scalar = 0.6f32;
        let norm_fn = |ap: &Tensor<f32>| -> f64 {
            let (_, s, _) = svd(ap).unwrap();
            f64::from(dense_vec(&s)[0])
        };
        let da = matrix_norm_vjp(&a, MatrixNormOrd::Spectral, g_scalar).unwrap();
        let numeric = numeric_grad(&a, |ap| f64::from(g_scalar) * norm_fn(ap));
        assert_grad_close("matrix_norm spectral", &da, &numeric);
    }

    /// codex-review 指摘の回帰: `Fro` の逆伝播は `norm > 0.0`
    /// （NaN に対して常に偽）で「ゼロノルム」分岐へ落ち、forward
    /// （`[[NaN,1]]` は `matrix_norm_fro` が正しく NaN を返す）が
    /// 維持する数値異常を逆伝播で消して全勾配 0 にしていた。
    #[test]
    fn matrix_norm_fro_vjp_propagates_nan() {
        let a = build_tensor(vec![f32::NAN, 1.0], &[1, 2]);
        let da = matrix_norm_vjp(&a, MatrixNormOrd::Fro, 1.0).unwrap();
        for v in dense_vec(&da) {
            assert!(v.is_nan(), "Fro 勾配は全要素 NaN のはず: {v}");
        }
    }

    /// `One`／`Inf` も同じ理由（NaN を含む列・行が「最大列／行」選択
    /// から無条件に除外され、有限な勾配が返ってしまう）で NaN を
    /// 伝播すべきことの回帰。
    #[test]
    fn matrix_norm_one_and_inf_vjp_propagate_nan() {
        let a = build_tensor(vec![f32::NAN, 1.0], &[1, 2]);
        let da_one = matrix_norm_vjp(&a, MatrixNormOrd::One, 1.0).unwrap();
        for v in dense_vec(&da_one) {
            assert!(v.is_nan(), "One 勾配は全要素 NaN のはず: {v}");
        }
        let da_inf = matrix_norm_vjp(&a, MatrixNormOrd::Inf, 1.0).unwrap();
        for v in dense_vec(&da_inf) {
            assert!(v.is_nan(), "Inf 勾配は全要素 NaN のはず: {v}");
        }
    }

    /// rank-1（縦長・列が正確に比例）行列で `U` の全列が直交すること
    /// （単位ノルム・列内積が 0 に近いこと）を確認する（codex-review
    /// 指摘の回帰。`crates/backend-cpu/src/linalg.rs` と同型のテスト）。
    /// `U` 列を自身の実測ノルムで単位長へ正規化する（`U = A V / σ` の
    /// 行列積再導出を経由しない）ため、σ が厳密ゼロになるか丸め残差
    /// として残るかに関わらず列直交性が成立する。
    #[test]
    fn svd_rank_deficient_tall_matrix_u_columns_are_orthonormal() {
        let a = build_tensor(vec![1.0, 3.0, 2.0, 6.0, 5.0, 15.0], &[3, 2]);
        let (u, s, vh) = svd(&a).unwrap();
        assert_eq!(u.shape(), &[3, 2]);
        let s_data = dense_vec(&s);
        assert!(s_data[0] > 0.0, "第一特異値が非ゼロであるべき: {s_data:?}");
        let u_mat = Mat::from_tensor(&u);
        for j in 0..2 {
            let col = u_mat.col(j);
            let norm: f64 = col.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "U 列 {j} が単位ノルムでない: norm={norm}"
            );
        }
        let col0 = u_mat.col(0);
        let col1 = u_mat.col(1);
        let dot: f64 = col0.iter().zip(col1.iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-6, "U の列が直交していない: dot={dot}");
        let mut sigma = Mat::zeros(2, 2);
        sigma.set(0, 0, f64::from(s_data[0]));
        sigma.set(1, 1, f64::from(s_data[1]));
        let reconstructed = u_mat
            .matmul(&sigma)
            .matmul(&Mat::from_tensor(&vh))
            .to_tensor();
        approx_eq(&reconstructed, &a, 1e-3);
    }
}
