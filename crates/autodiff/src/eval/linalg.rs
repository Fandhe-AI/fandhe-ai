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

/// `A^{-1}`（`A X = I` を解く）。VJP（`grad.rs::Op::Inv`）が
/// `out_value`（= `A^{-1}`）を再利用できるよう、forward 値と同じ関数を
/// 使う。
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

/// `det(A)`（LU 対角積 × 置換符号）。空行列は空積 `1.0`。特異行列は
/// `0.0`（エラーにしない。設計文書 §3.5・PyTorch `torch.linalg.det`
/// 挙動）。
pub(crate) fn det(a: &Tensor<f32>) -> Tensor<f32> {
    let mat = Mat::from_tensor(a);
    let n = mat.rows;
    if n == 0 {
        return build_tensor(vec![1.0], &[]);
    }
    let value = match lu_decompose(&mat) {
        None => 0.0,
        Some(lu) => {
            let mut prod = lu.sign;
            for i in 0..n {
                prod *= lu.lu.get(i, i);
            }
            prod
        }
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

    // Householder 反射を `r`（作業用に `A` を上書き）へ逐次適用しつつ、
    // `Q = H_0 H_1 ... H_{k-1}` を明示的に蓄積する（k が小さい前提の
    // 参照実装として、反射の合成を都度フル行列積で行う単純な方式）。
    let mut r = mat;
    let mut q = Mat::identity(m);

    for col in 0..k {
        // Householder ベクトル `v`（列 `col` の対角以下）を作る。
        let mut x = vec![0.0; m - col];
        for i in col..m {
            x[i - col] = r.get(i, col);
        }
        let norm_x: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm_x == 0.0 {
            // この列は既にゼロ以下三角化済み（rank 落ち）。反射不要。
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
        // `q ← q H`（列方向に反射を右から合成。`Q` の列 `col..` にのみ
        // 作用する）。
        for row in 0..m {
            let mut dot = 0.0;
            for (i, &vi) in v.iter().enumerate() {
                dot += vi * q.get(row, col + i);
            }
            if dot == 0.0 {
                continue;
            }
            for (i, &vi) in v.iter().enumerate() {
                let idx = col + i;
                q.set(row, idx, q.get(row, idx) - 2.0 * vi * dot);
            }
        }
    }

    // reduced 形（`Q` の先頭 k 列・`R` の先頭 k 行）へ切り出しつつ、
    // `R` 対角の符号を非負へ正規化する（対応する `Q` 列の符号も反転して
    // `Q R = A` を保つ）。
    let mut q_reduced = Mat::zeros(m, k);
    for row in 0..m {
        for c in 0..k {
            q_reduced.set(row, c, q.get(row, c));
        }
    }
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
/// しきい値。[`svd`] の数値 rank 判定（丸め残差由来の微小特異値を
/// 厳密ゼロへ丸める `rank_tol`）もこの値の平方根を共有する——
/// Jacobi は `gamma.abs() <= JACOBI_EPS * sqrt(alpha*beta)` で収束を
/// 打ち切るため、収束後に残る非直交性・特異値誤差は理論上
/// `O(sqrt(JACOBI_EPS))` スケールになりうる（本来 0 の特異値が
/// `JACOBI_EPS` そのものの桁では丸められず残ってしまう。
/// codex-review 指摘: `A=[[1,0.3],[2,0.6],[5,1.5]]` のような
/// 丸め残差由来の rank 落ちで、この定数を共有しない独立の機械
/// epsilon ベースしきい値では検出できなかった）。
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

    // 数値的な rank 判定のための相対しきい値: 最大特異値 ×
    // `max(m,n)` × `sqrt(JACOBI_EPS)`。`jacobi_svd_tall` の収束打ち切り
    // （`JACOBI_EPS` 相対判定）により残る非直交性が `O(sqrt(JACOBI_EPS))`
    // スケールになりうるため機械 epsilon では小さすぎる。
    // `crates/backend-cpu/src/linalg.rs::svd` と同型の対応（codex-review
    // 指摘。丸め残差由来の微小非ゼロ特異値が「実測ノルムで単位長化」
    // 経路へ進み直交契約を満たせなくなるのを防ぐ。固定絶対閾値は
    // 使わない理由も同ファイルの対応コメントを参照）。
    let rank_tol = s_sorted[0] * (m.max(n) as f64) * JACOBI_EPS.sqrt();
    for sigma in s_sorted.iter_mut() {
        if *sigma <= rank_tol {
            *sigma = 0.0;
        }
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

/// `Op::Inv` の VJP: `dA = -(X^T g X^T)`（`X = A^{-1}` = forward 記録値）。
pub(crate) fn inv_vjp(x: &Tensor<f32>, g: &Tensor<f32>) -> Tensor<f32> {
    let x_mat = Mat::from_tensor(x);
    let g_mat = Mat::from_tensor(g);
    let xt = x_mat.transpose();
    let mut result = xt.matmul(&g_mat).matmul(&xt);
    for v in result.data.iter_mut() {
        *v = -*v;
    }
    result.to_tensor()
}

/// `Op::Solve` の VJP: `dB = A^{-T} g`、`dA = -dB Xᵀ`（`X` = forward
/// 記録値の解）。`dB` を `f32` へ downcast する前の `f64` 中間値
/// （[`solve_transposed_mat`]）を `dA` の計算にも使い、両方の勾配を
/// 計算し終えてから 1 回だけ `f32` へ downcast する（codex-review
/// 指摘の是正。理由は [`solve_transposed_mat`] のコメント参照）。
pub(crate) fn solve_vjp(
    a: &Tensor<f32>,
    x: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Result<(Tensor<f32>, Tensor<f32>), AutodiffError> {
    let db_mat = solve_transposed_mat(a, g)?;
    let x_mat = Mat::from_tensor(x);
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
pub(crate) fn det_vjp(a: &Tensor<f32>, g_scalar: f32) -> Result<Tensor<f32>, AutodiffError> {
    let mat = Mat::from_tensor(a);
    let n = mat.rows;
    if n == 0 {
        // 空行列の `det` は空積 `1.0`（定数）のため勾配は自明に空。
        return Ok(build_tensor(Vec::new(), &[0, 0]));
    }
    let lu = lu_decompose(&mat)
        .ok_or_else(|| invalid("eval::linalg::det_vjp: 行列が特異（ピボットが厳密 0）"))?;
    let mut det_value = lu.sign;
    for i in 0..n {
        det_value *= lu.lu.get(i, i);
    }
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

    // term3 = U diag(S)^{-1} dVᵀ (I_n - V Vᵀ)
    //       = U diag(S)^{-1} dVᵀ − U (diag(S)^{-1} dVᵀ V) Vᵀ
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
            if norm > 0.0 {
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
            for c in 0..mat.cols {
                let sum: f64 = mat.col(c).iter().map(|v| v.abs()).sum();
                if sum > best_sum {
                    best_sum = sum;
                    best_col = c;
                }
            }
            let mut da = Mat::zeros(mat.rows, mat.cols);
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
            for r in 0..mat.rows {
                let mut sum = 0.0;
                for c in 0..mat.cols {
                    sum += mat.get(r, c).abs();
                }
                if sum > best_sum {
                    best_sum = sum;
                    best_row = r;
                }
            }
            let mut da = Mat::zeros(mat.rows, mat.cols);
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
        let x = inv(&a).unwrap();
        let da = inv_vjp(&x, &s);
        let numeric = numeric_grad(&a, |ap| scalar_dot(&inv(ap).unwrap(), &s));
        assert_grad_close("inv", &da, &numeric);
    }

    #[test]
    fn solve_vjp_matches_numeric() {
        let a = build_tensor(vec![4.0, 1.0, 2.0, 3.0], &[2, 2]);
        let b = build_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = build_tensor(vec![1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let x = solve(&a, &b).unwrap();
        let (da, db) = solve_vjp(&a, &x, &s).unwrap();
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
        let x = solve(&a, &b).unwrap();
        let g = build_tensor(vec![1e20], &[1, 1]);
        let (da, db) = solve_vjp(&a, &x, &g).unwrap();
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
}
