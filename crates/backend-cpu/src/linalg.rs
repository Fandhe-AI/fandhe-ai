//! 線形代数（inv／solve／det／qr／cholesky／svd）・matrix_norm の CPU
//! 参照実装（イシュー #1621・親イシュー #1573「Tier 2: 線形代数」・
//! `docs/spec/04-requirements.md` REQ-9 2026-09-12 追記）。
//!
//! `fandhe_ai_autodiff::eval::linalg`（`autodiff` クレート内 `pub(crate)`
//! のホスト参照実装。`compat`／`NaiveOps` 経路が使う）と**同一の
//! アルゴリズム・同一の符号／ゲージ規約**で実装するが、依存方向の制約
//! （`autodiff` → `backend-cpu` の依存は作れる一方、逆の
//! `backend-cpu`（本クレート）が `autodiff` の非公開実装へ依存すること
//! はできない。`crates/autodiff/tests/architecture_boundaries.rs` が
//! 機械検査する不変条件）により、コードは意図的に複製する
//! （`docs/autodiff-linalg-design.md` §3「eval と CPU の関係」）。
//!
//! `ops.rs::CpuBackendOps` の `BackendOps::linalg_*` 実装が本モジュールへ
//! 委譲する薄いディスパッチ層（`gemm_blis`／`rmsnorm` と同じ構成方針。
//! モジュール冒頭コメント参照）。
//!
//! # 数値契約（`docs/autodiff-linalg-design.md` §3.5 が正。`eval::linalg`
//! と同一）
//!
//! - **内部精度**: 分解・解法は `f64` で逐次固定順序に計算し、出力時に
//!   1 回だけ `f32` へ downcast する（`.claude/rules/coding-rust.md`）。
//! - **符号・ゲージ規約**: QR は `R` の対角を非負に正規化する。SVD は
//!   特異値を降順（同値は安定ソート）に並べ、各 `V` 列は最大絶対値成分
//!   （同値は最小添字）が正になるよう符号を正規化し、`U = A V / σ` で
//!   導出する（`σ == 0` の列は Gram–Schmidt で補完する）。Cholesky は
//!   下三角のみを返す（上三角は 0）。
//! - **エラー分類**: 特異／非正定値／非収束は [`LinalgError`]。`ops.rs`
//!   がこれを `BackendError::InvalidArgument` へ変換する（`rmsnorm.rs`
//!   の `RmsNormError` と同じ「小さな enum」方針）。
//! - **空行列（`n=0`）**: `det` は空積 `1.0`。`inv`／`cholesky`／`qr`／
//!   `svd` は対応する空 shape のテンソルを返す。`solve` は `[0,k]`。

use fandhe_ai_tensor_core::{MatrixNormOrd, Tensor};

/// 本モジュールの型付きエラー（`rmsnorm::RmsNormError` と同じ「小さな
/// enum」方針）。特異／非正定値／非収束のいずれも `ops.rs` 側では
/// 一様に [`fandhe_ai_tensor_core::BackendError::InvalidArgument`] へ
/// 変換する契約（設計文書 §3.5「エラー分類」は分類名を要求せず
/// メッセージで区別すれば足りるため、`autodiff::error::AutodiffError::
/// InvalidArgument(String)` と同型の単一 variant とする）。
#[non_exhaustive]
#[derive(Debug)]
pub enum LinalgError {
    /// 特異行列・非正定値・非収束のいずれか（メッセージで区別する）。
    InvalidArgument(String),
}

impl std::fmt::Display for LinalgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinalgError::InvalidArgument(msg) => write!(f, "linalg: invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for LinalgError {}

/// テンソルを行優先連続バッファへ実体化する（`autodiff::eval::
/// dense_vec` と同じ役割。`Tensor::contiguous` は必ず `as_slice` が
/// `Some` を返す状態を作るため `unwrap_or_default` は理論上到達しない
/// フォールバックに留まる）。
fn dense_vec(tensor: &Tensor<f32>) -> Vec<f32> {
    let contiguous = tensor.contiguous();
    contiguous
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_default()
}

/// `Vec<f32>` + shape からテンソルを構築する（`autodiff::eval::
/// build_tensor` と同じ役割）。呼び出し元は shape とデータ長を事前に
/// 一致させる契約（本モジュール内部のみで完結する構築のため、通常この
/// 契約が破れることはない）が、万一の内部契約違反を `.expect()`／
/// 黙殺（別 shape の空テンソルへの差し替え）で握りつぶさず
/// `LinalgError` として呼び出し元へ伝播する（codex-review 指摘。
/// `.claude/rules/security.md`「本番経路の unwrap／expect 禁止」）。
fn build_tensor(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<f32>, LinalgError> {
    Tensor::new(data, shape).map_err(|e| {
        invalid(format!(
            "linalg::build_tensor: shape とデータ長の不一致（内部契約違反）: {e}"
        ))
    })
}
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
            "linalg::Mat::from_tensor: 呼び出し元が rank-2 を検査済みの契約"
        );
        let rows = shape[0];
        let cols = shape[1];
        let data = dense_vec(t).into_iter().map(f64::from).collect();
        Mat { data, rows, cols }
    }

    fn to_tensor(&self) -> Result<Tensor<f32>, LinalgError> {
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
    /// 本番経路（`qr`／`svd` 等）は `U = A V / σ` 型の行列積再導出を
    /// 廃し自身のノルムで正規化する方式へ切り替えたため（codex-review
    /// 指摘。rank 落ち行列での桁落ち増幅を回避）、現在はテスト
    /// （`QR = Q R`／`SVD = U Σ Vᵀ` の再構成検証）専用。`#[cfg(test)]`
    /// を付けず本番ビルドの dead_code 警告を許容するより、実際の
    /// 利用範囲を型で明示する。
    #[cfg(test)]
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

fn invalid(msg: impl Into<String>) -> LinalgError {
    LinalgError::InvalidArgument(msg.into())
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

/// `A^{-1}`（`A X = I` を解く）。VJP は本クレートに実装がなく
/// `crates/autodiff::eval::linalg::inv_vjp` が担う（`Op::Inv` の逆伝播は
/// バックエンドに依らずホスト参照実装へ集約する設計。`docs/
/// autodiff-linalg-design.md` §3.6）。`inv_vjp` は forward が返す
/// `out_value`（`f32` 丸め済み記録値）を再利用せず `a` から改めて
/// `f64` で計算し直す（codex-review 指摘・2026-09-13 是正。以前の
/// 本コメントは「VJP が out_value を再利用する」という逆の記述
/// だった）。
pub(crate) fn inv(a: &Tensor<f32>) -> Result<Tensor<f32>, LinalgError> {
    let mat = Mat::from_tensor(a);
    let n = mat.rows;
    if n == 0 {
        return build_tensor(Vec::new(), &[0, 0]);
    }
    let lu =
        lu_decompose(&mat).ok_or_else(|| invalid("linalg::inv: 行列が特異（ピボットが厳密 0）"))?;
    let identity = Mat::identity(n);
    let x = lu_solve_mat(&lu, &identity);
    x.to_tensor()
}

/// `A X = B` を解く。
pub(crate) fn solve(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, LinalgError> {
    let a_mat = Mat::from_tensor(a);
    let b_mat = Mat::from_tensor(b);
    let n = a_mat.rows;
    if n == 0 {
        return build_tensor(Vec::new(), &[0, b_mat.cols]);
    }
    let lu = lu_decompose(&a_mat)
        .ok_or_else(|| invalid("linalg::solve: 係数行列が特異（ピボットが厳密 0）"))?;
    let x = lu_solve_mat(&lu, &b_mat);
    x.to_tensor()
}

/// `x = m · 2^e`（`0.5 <= |m| < 1`。`frexp` 相当）へ分解する。`0` と
/// 非有限（`NaN`／`Inf`）はそのまま素通しする（`e = 0`）。
/// `f64::to_bits`／`from_bits` によるビットフィールド抽出・再構成のみで
/// 構成し `unsafe` を使わない（IEEE 754 binary64 の指数・仮数フィールド
/// は安全に読み書きできる。非正規化数〈subnormal〉は `2^64` を掛けて
/// 正規化してから指数を補正する）。[`lu_diag_product`] が使う
/// （`crates/autodiff/src/eval/linalg.rs::frexp` と同一実装。設計文書
/// §3「eval と CPU の関係」に沿った意図的複製。codex-review 指摘
/// PRRT_kwDOTuUCJc6hxiys の是正）。
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
/// 行列式が `Inf`／`NaN` になってしまう（codex-review 指摘
/// PRRT_kwDOTuUCJc6hxiys）。log-abs 方式（`ln` の和を経由する方式）は
/// 不要な丸め誤差を追加するため採用せず、各対角要素を `m·2^e`
/// （`0.5<=|m|<1`）へ分解し、仮数の積を毎回正規化しつつ指数を整数で
/// 加算する方式を採る。最終合成（`ldexp`）が `f64` の範囲を超える
/// 場合にのみ `Inf`／`0.0` を返す（真値が範囲外のときの正しい挙動）。
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
pub(crate) fn det(a: &Tensor<f32>) -> Result<Tensor<f32>, LinalgError> {
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
// =====================================================================
// Cholesky 分解（Cholesky–Banachiewicz、下三角）。
// =====================================================================

/// `A = L Lᵀ`（`A` は対称正定値と仮定・下三角のみ読む）。非正定値
/// （対角が非有限または非正）は `InvalidArgument`。
pub(crate) fn cholesky(a: &Tensor<f32>) -> Result<Tensor<f32>, LinalgError> {
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
                        "linalg::cholesky: 行列が対称正定値でない（対角が非有限または非正）",
                    ));
                }
                l.set(i, j, sum.sqrt());
            } else {
                let diag = l.get(j, j);
                l.set(i, j, sum / diag);
            }
        }
    }
    l.to_tensor()
}

// =====================================================================
// Householder QR（reduced）。
// =====================================================================

/// reduced QR（`A: [m,n]` → `Q: [m,k]`・`R: [k,n]`、`k = min(m,n)`）。
/// `R` の対角は非負に正規化する（設計文書 §3.5「符号・ゲージ規約」）。
pub(crate) fn qr(a: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>), LinalgError> {
    let mat = Mat::from_tensor(a);
    let (m, n) = (mat.rows, mat.cols);
    let k = m.min(n);
    if m == 0 || n == 0 {
        return Ok((
            build_tensor(Vec::new(), &[m, k])?,
            build_tensor(Vec::new(), &[k, n])?,
        ));
    }

    // Householder 反射を `r`（作業用に `A` を上書き）へ逐次適用する。
    // `Q`（= `H_0 H_1 ... H_{k-1}`）は `m×m` の単位行列を明示構築せず、
    // 正規化済み反射ベクトル `v`（列インデックス付き）のみを蓄積し、
    // reduced 形（`[m,k]`）を後段で逆順適用により直接構築する
    // （O(mk) メモリ。以前は `Mat::identity(m)` により `m×m` の `f64`
    // 行列を確保しており、`[100000,1]` のような縦長入力でメモリ枯渇を
    // 招いていた——`m×m` だけで約 80 GB。codex-review 指摘。
    // `docs/autodiff-linalg-design.md` §3.4「QR」。`crates/autodiff::
    // eval::linalg::qr` と同一アルゴリズム）。
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

    Ok((q_reduced.to_tensor()?, r_reduced.to_tensor()?))
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
fn jacobi_svd_tall(a: &Mat) -> Result<(Mat, Vec<f64>, Mat), LinalgError> {
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
        "linalg::svd: 片側 Jacobi 法が反復上限（60 スイープ）内に収束しなかった",
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
pub(crate) fn svd(a: &Tensor<f32>) -> Result<SvdOutput, LinalgError> {
    let mat = Mat::from_tensor(a);
    let (m, n) = (mat.rows, mat.cols);
    let k = m.min(n);
    if k == 0 {
        return Ok((
            build_tensor(Vec::new(), &[m, k])?,
            build_tensor(Vec::new(), &[k])?,
            build_tensor(Vec::new(), &[k, n])?,
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
    // 指摘）。
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
    // codex-review 指摘の修正）。
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
        u_sorted.to_tensor()?,
        build_tensor(s_sorted.iter().map(|&v| v as f32).collect(), &[k])?,
        v_sorted.transpose().to_tensor()?,
    ))
}

// =====================================================================
// 行列ノルム。
// =====================================================================

/// `MatrixNormOrd::Fro`／`One`／`Inf` の 3 種（`Nuc`／`Spectral` は
/// `svd` を要するため `var.rs` 側で分岐する）。
pub(crate) fn matrix_norm_fro(a: &Tensor<f32>) -> Result<Tensor<f32>, LinalgError> {
    let sum_sq: f64 = dense_vec(a)
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum();
    build_tensor(vec![sum_sq.sqrt() as f32], &[])
}

pub(crate) fn matrix_norm_one(a: &Tensor<f32>) -> Result<Tensor<f32>, LinalgError> {
    let mat = Mat::from_tensor(a);
    let mut max_sum = 0.0f64;
    let mut has_nan = false;
    for c in 0..mat.cols {
        let sum: f64 = mat.col(c).iter().map(|v| v.abs()).sum();
        // `sum` が `NaN` のとき `sum > max_sum` は常に偽（IEEE 754 の
        // 順序付き比較は NaN を含む比較を全て偽にする）ため、NaN を
        // 含む列が最大値の更新へ一切寄与せず無視されたかのように
        // `max_sum` が有限値のまま返ってしまう（codex-review 指摘。
        // 数値異常が正常値 `0` へ変換され隠れる）。列走査とは独立に
        // NaN の有無を検出し最終結果へ伝播させる（`eval::linalg::
        // matrix_norm_one` と同一方針）。
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

pub(crate) fn matrix_norm_inf(a: &Tensor<f32>) -> Result<Tensor<f32>, LinalgError> {
    let mat = Mat::from_tensor(a);
    let mut max_sum = 0.0f64;
    let mut has_nan = false;
    for r in 0..mat.rows {
        let mut sum = 0.0;
        for c in 0..mat.cols {
            sum += mat.get(r, c).abs();
        }
        // `matrix_norm_one` と同じ理由で NaN の有無を独立に検出し
        // 最終結果へ伝播する（codex-review 指摘）。
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
pub(crate) fn matrix_norm(a: &Tensor<f32>, ord: MatrixNormOrd) -> Result<Tensor<f32>, LinalgError> {
    match ord {
        MatrixNormOrd::Fro => matrix_norm_fro(a),
        MatrixNormOrd::One => matrix_norm_one(a),
        MatrixNormOrd::Inf => matrix_norm_inf(a),
        MatrixNormOrd::Nuc => {
            let (_, s, _) = svd(a)?;
            let sum: f64 = dense_vec(&s).iter().map(|&v| f64::from(v)).sum();
            build_tensor(vec![sum as f32], &[])
        }
        MatrixNormOrd::Spectral => {
            let (_, s, _) = svd(a)?;
            let max = dense_vec(&s).first().copied().unwrap_or(0.0);
            build_tensor(vec![max], &[])
        }
        _ => Err(invalid(format!(
            "linalg::matrix_norm: 未知の MatrixNormOrd variant（{ord:?}）"
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

    #[test]
    fn inv_2x2_matches_known_solution() {
        let a = build_tensor(vec![4.0, 7.0, 2.0, 6.0], &[2, 2]).unwrap();
        let inv_a = inv(&a).unwrap();
        let expected = build_tensor(vec![0.6, -0.7, -0.2, 0.4], &[2, 2]).unwrap();
        approx_eq(&inv_a, &expected, 1e-5);
    }

    #[test]
    fn inv_singular_is_invalid_argument() {
        let a = build_tensor(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
        assert!(matches!(inv(&a), Err(LinalgError::InvalidArgument(_))));
    }

    #[test]
    fn inv_empty_matrix_returns_empty() {
        let a = build_tensor(Vec::new(), &[0, 0]).unwrap();
        let result = inv(&a).unwrap();
        assert_eq!(result.shape(), &[0, 0]);
    }

    #[test]
    fn solve_matches_known_solution() {
        let a = build_tensor(vec![3.0, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        let b = build_tensor(vec![9.0, 8.0], &[2, 1]).unwrap();
        let x = solve(&a, &b).unwrap();
        approx_eq(&x, &build_tensor(vec![2.0, 3.0], &[2, 1]).unwrap(), 1e-4);
    }

    #[test]
    fn det_known_value_and_singular_is_zero() {
        let a = build_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        approx_eq(
            &det(&a).unwrap(),
            &build_tensor(vec![-2.0], &[]).unwrap(),
            1e-5,
        );
        let s = build_tensor(vec![1.0, 2.0, 2.0, 4.0], &[2, 2]).unwrap();
        approx_eq(
            &det(&s).unwrap(),
            &build_tensor(vec![0.0], &[]).unwrap(),
            1e-6,
        );
    }

    #[test]
    fn det_empty_matrix_is_one() {
        let a = build_tensor(Vec::new(), &[0, 0]).unwrap();
        approx_eq(
            &det(&a).unwrap(),
            &build_tensor(vec![1.0], &[]).unwrap(),
            1e-6,
        );
    }

    /// `values` を対角成分に持つ正方行列（非対角は 0）を組み立てる
    /// テスト補助（`lu_diag_product` のオーバーフロー回帰テストで使う。
    /// `autodiff` クレート `eval/linalg.rs` の同名テスト補助と同一
    /// 実装。設計文書 §3「eval と CPU の関係」に沿った意図的複製）。
    fn diag_tensor(values: &[f32]) -> Tensor<f32> {
        let n = values.len();
        let mut data = vec![0.0f32; n * n];
        for (i, v) in values.iter().enumerate() {
            data[i * n + i] = *v;
        }
        build_tensor(data, &[n, n]).unwrap()
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
        let v = dense_vec(&det(&a).unwrap())[0];
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
        let v = dense_vec(&det(&a).unwrap())[0];
        assert!(v.is_finite(), "det must be finite: {v}");
        assert_ne!(v, 0.0, "det should not underflow to zero: {v}");
        assert!((v - 1.0).abs() < 1e-2, "det should be approx 1.0: {v}");
    }

    #[test]
    fn cholesky_reconstructs_spd_matrix() {
        // A = [[4,2],[2,3]] は対称正定値。
        let a = build_tensor(vec![4.0, 2.0, 2.0, 3.0], &[2, 2]).unwrap();
        let l = cholesky(&a).unwrap();
        approx_eq(
            &l,
            &build_tensor(vec![2.0, 0.0, 1.0, 2f32.sqrt()], &[2, 2]).unwrap(),
            1e-4,
        );
    }

    #[test]
    fn cholesky_non_positive_definite_is_invalid() {
        let a = build_tensor(vec![1.0, 2.0, 2.0, 1.0], &[2, 2]).unwrap();
        assert!(matches!(cholesky(&a), Err(LinalgError::InvalidArgument(_))));
    }

    #[test]
    fn qr_reconstructs_input_and_is_orthonormal() {
        let a = build_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let (q, r) = qr(&a).unwrap();
        assert_eq!(q.shape(), &[3, 2]);
        assert_eq!(r.shape(), &[2, 2]);
        let reconstructed = Mat::from_tensor(&q)
            .matmul(&Mat::from_tensor(&r))
            .to_tensor()
            .unwrap();
        approx_eq(&reconstructed, &a, 1e-4);
    }

    /// codex-review 指摘の回帰（`crates/autodiff::eval::linalg` と同型。
    /// 同ドキュメント参照）: `[100000,1]` のような縦長入力で `m×m` の
    /// 中間行列（約 80 GB）を確保せず O(mk) メモリで完了することを
    /// 確認する。
    #[test]
    fn qr_tall_matrix_does_not_allocate_full_m_by_m_intermediate() {
        let m = 100_000usize;
        let data: Vec<f32> = (0..m).map(|i| 1.0 + (i % 7) as f32).collect();
        let a = build_tensor(data, &[m, 1]).unwrap();
        let (q, r) = qr(&a).unwrap();
        assert_eq!(q.shape(), &[m, 1]);
        assert_eq!(r.shape(), &[1, 1]);

        let q_data = dense_vec(&q);
        let norm: f64 = q_data.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        assert!(
            (norm.sqrt() - 1.0).abs() < 1e-3,
            "Q 列が単位ノルムでない: norm={norm}"
        );

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
    fn svd_reconstructs_square_input() {
        let mut a_data = vec![0.0f32; 9];
        let s = [3.0f32, 2.0, 1.0];
        for i in 0..3 {
            a_data[i * 3 + i] = s[i];
        }
        let a = build_tensor(a_data, &[3, 3]).unwrap();
        let (u, s_out, vh) = svd(&a).unwrap();
        assert_eq!(u.shape(), &[3, 3]);
        assert_eq!(s_out.shape(), &[3]);
        assert_eq!(vh.shape(), &[3, 3]);
        approx_eq(
            &s_out,
            &build_tensor(vec![3.0, 2.0, 1.0], &[3]).unwrap(),
            1e-4,
        );
    }

    #[test]
    fn svd_wide_matrix_reconstructs_input() {
        let a = build_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let (u, s, vh) = svd(&a).unwrap();
        let s_data = dense_vec(&s);
        let u_data = dense_vec(&u);
        let mut u_scaled = vec![0.0f32; 4];
        for r in 0..2 {
            for c in 0..2 {
                u_scaled[r * 2 + c] = u_data[r * 2 + c] * s_data[c];
            }
        }
        let reconstructed = Mat::from_tensor(&build_tensor(u_scaled, &[2, 2]).unwrap())
            .matmul(&Mat::from_tensor(&vh))
            .to_tensor()
            .unwrap();
        approx_eq(&reconstructed, &a, 1e-3);
    }

    /// `σ` が固定閾値 `1e-12` 未満でも非ゼロならゼロ特異値として扱わず、
    /// 独立基底で上書きしないことを確認する（`crates/autodiff::eval::linalg`
    /// と同型の回帰。codex-review 指摘の修正）。修正前は `A=[[-1e-13]]` で
    /// `U`/`Vh` の符号・再構成が誤っていた——σ<=1e-12 の非ゼロ特異値が
    /// 固定絶対閾値でゼロ扱いされ Gram–Schmidt 補完の独立基底へ置き換わり
    /// `U Σ Vᵀ = A` を満たさなくなる不具合。設計 §3.5「σ==0 の列のみ補完」
    /// 契約に合わせ、判定を厳密ゼロ比較へ変更した。
    #[test]
    fn svd_tiny_nonzero_singular_value_reconstructs_input() {
        let a = build_tensor(vec![-1e-13], &[1, 1]).unwrap();
        let (u, s, vh) = svd(&a).unwrap();
        let s_data = dense_vec(&s);
        assert!(
            s_data[0] > 0.0,
            "非ゼロ特異値がゼロ扱いされている: {}",
            s_data[0]
        );
        let u_scaled = vec![dense_vec(&u)[0] * s_data[0]];
        let reconstructed = Mat::from_tensor(&build_tensor(u_scaled, &[1, 1]).unwrap())
            .matmul(&Mat::from_tensor(&vh))
            .to_tensor()
            .unwrap();
        approx_eq(&reconstructed, &a, 1e-18);
    }

    #[test]
    fn matrix_norm_known_values() {
        let a = build_tensor(vec![3.0, 4.0], &[1, 2]).unwrap();
        approx_eq(
            &matrix_norm_fro(&a).unwrap(),
            &build_tensor(vec![5.0], &[]).unwrap(),
            1e-5,
        );

        let b = build_tensor(vec![2.0, -1.0, -6.0, 3.0], &[2, 2]).unwrap();
        approx_eq(
            &matrix_norm_one(&b).unwrap(),
            &build_tensor(vec![8.0], &[]).unwrap(),
            1e-5,
        );
        approx_eq(
            &matrix_norm_inf(&b).unwrap(),
            &build_tensor(vec![9.0], &[]).unwrap(),
            1e-5,
        );
    }

    /// codex-review 指摘の回帰（`crates/autodiff::eval::linalg` と同型）:
    /// `sum > max_sum` は NaN に対して常に偽のため、NaN を含む唯一の
    /// 列・行が最大値の更新へ一切寄与せず `[[NaN]]` のノルムが正常な
    /// `0` として返ってしまっていた。
    #[test]
    fn matrix_norm_one_and_inf_propagate_nan() {
        let a = build_tensor(vec![f32::NAN], &[1, 1]).unwrap();
        assert!(dense_vec(&matrix_norm_one(&a).unwrap())[0].is_nan());
        assert!(dense_vec(&matrix_norm_inf(&a).unwrap())[0].is_nan());
    }

    /// codex-review 指摘の回帰（`crates/autodiff::eval::linalg` と同型）:
    /// Jacobi の収束判定に絶対下限 `.max(EPS)` があると入力スケール
    /// 非依存の閾値が生じ、約 1e-15 スケールの入力で非直交な列を誤って
    /// 収束扱いし、誤った特異値を返していた。相対収束判定へ是正後は
    /// 正しい特異値（黄金比由来の `[1.618...e-15, 0.618...e-15]`）を
    /// 返す。
    #[test]
    fn svd_tiny_scale_singular_values_are_accurate() {
        let scale = 1e-15f32;
        let a = build_tensor(vec![scale, scale, 0.0, scale], &[2, 2]).unwrap();
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

    /// codex-review 指摘の回帰（`crates/autodiff::eval::linalg` と同型）:
    /// 零特異値の `V` 列を Gram–Schmidt で直交補完した後にも「最大絶対
    /// 値成分が正」という符号規約（設計文書 §3.5）が保たれることを
    /// 確認する。
    #[test]
    fn svd_zero_sigma_gram_schmidt_column_respects_sign_convention() {
        let a = build_tensor(vec![2.0, 1.0, 0.0, 0.0], &[2, 2]).unwrap();
        let (_, s, vh) = svd(&a).unwrap();
        let s_data = dense_vec(&s);
        assert!(
            s_data[1].abs() < 1e-6,
            "第二特異値は厳密 0 のはず: {s_data:?}"
        );
        let vh_data = dense_vec(&vh);
        let col = [vh_data[2], vh_data[3]];
        let max_abs_idx = if col[0].abs() >= col[1].abs() { 0 } else { 1 };
        assert!(
            col[max_abs_idx] >= 0.0,
            "補完後の V 列が符号規約（最大絶対値成分が正）に違反: {col:?}"
        );
    }

    /// 同一入力を 2 回計算しても bit 決定的（run-to-run 一致）である
    /// ことを確認する（設計文書 §3.5「同一入力に対し run-to-run で
    /// bit 決定的」・PoC-v2-2 のビット一致決定性方針）。
    #[test]
    fn svd_is_bit_deterministic_across_repeated_calls() {
        let a = build_tensor(vec![3.0, 0.3, 0.1, 0.2, 2.0, 0.2, 0.1, 0.15, 1.0], &[3, 3]).unwrap();
        let (u1, s1, vh1) = svd(&a).unwrap();
        let (u2, s2, vh2) = svd(&a).unwrap();
        assert_eq!(dense_vec(&u1), dense_vec(&u2));
        assert_eq!(dense_vec(&s1), dense_vec(&s2));
        assert_eq!(dense_vec(&vh1), dense_vec(&vh2));
    }

    /// 横長・全ゼロ特異値ケース（`A=[[0,0]]`）で `Vh` の行が単位ノルムに
    /// 直交補完されることを確認する（codex-review 指摘。P1 #5 の修正
    /// 回帰）。修正前は `V=[[0,0]]` のまま残り `Vh・Vhᵀ ≈ I` を満たさな
    /// かった。
    #[test]
    fn svd_wide_all_zero_gauge_v_is_orthonormal() {
        let a = build_tensor(vec![0.0, 0.0], &[1, 2]).unwrap();
        let (u, s, vh) = svd(&a).unwrap();
        assert_eq!(u.shape(), &[1, 1]);
        assert_eq!(s.shape(), &[1]);
        assert_eq!(vh.shape(), &[1, 2]);
        approx_eq(&s, &build_tensor(vec![0.0], &[1]).unwrap(), 1e-6);
        // `U` は常に単位ノルム（`m>=n` 側なので元々成立）。
        let u_data = dense_vec(&u);
        let u_norm: f32 = u_data.iter().map(|v| v * v).sum::<f32>().sqrt();
        approx_eq(
            &build_tensor(vec![u_norm], &[]).unwrap(),
            &build_tensor(vec![1.0], &[]).unwrap(),
            1e-5,
        );
        // 修正対象: `Vh` の行が単位ノルムであること（修正前は
        // `[0.0, 0.0]` のまま残り 0 になっていた）。
        let vh_data = dense_vec(&vh);
        let vh_norm: f32 = vh_data.iter().map(|v| v * v).sum::<f32>().sqrt();
        approx_eq(
            &build_tensor(vec![vh_norm], &[]).unwrap(),
            &build_tensor(vec![1.0], &[]).unwrap(),
            1e-5,
        );
    }

    /// `min(m,n) == 1` では Jacobi の列ペア走査自体が実行されないため
    /// 非有限入力を拒否できない（`docs/autodiff-linalg-design.md` の
    /// 「非有限入力は演算ごとに異なり一様ではない」記述の裏付け。
    /// codex-review 指摘。挙動を仕様として固定する回帰テスト——将来
    /// 誤って「常に拒否される」よう変更された場合にこのテストが
    /// 失敗して気づけるようにする）。
    #[test]
    fn svd_min_dim_one_does_not_reject_non_finite_input() {
        let a = build_tensor(vec![f32::NAN], &[1, 1]).unwrap();
        let result = svd(&a);
        assert!(
            result.is_ok(),
            "min(m,n)==1 は非有限入力を拒否しない設計のはずが Err になった: {result:?}"
        );
    }

    /// rank-1（縦長・列が正確に比例）行列で `U` の全列が直交すること
    /// （単位ノルム・列内積が 0 に近いこと）を確認する（codex-review
    /// 指摘の回帰）。修正前は `U = A V / σ` を行列積で再導出しており、
    /// Jacobi の丸め残差により厳密には非ゼロな第二特異値に対しこの
    /// 再導出が桁落ちを増幅し、第二列のノルムが約 6.78（本来 0 に
    /// 近い値）になる等 `SvdFactors` の列直交契約（設計文書 §3.5）を
    /// 満たさなかった。修正後は `U` 列を自身の実測ノルムで単位長へ
    /// 正規化する（行列積を経由しない）ため、σ が厳密ゼロになるか
    /// 丸め残差として残るかに関わらず列直交性が成立する——σ2 の値
    /// 自体は主張しない（片側 Jacobi は収束時に列内積を極小化する
    /// ため、正規化後の直交性はいずれの場合も成り立つ）。
    #[test]
    fn svd_rank_deficient_tall_matrix_u_columns_are_orthonormal() {
        let a = build_tensor(vec![1.0, 3.0, 2.0, 6.0, 5.0, 15.0], &[3, 2]).unwrap();
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
        // 再構成 `U Σ Vᵀ ≈ A` も維持されることを確認する。
        let mut sigma = Mat::zeros(2, 2);
        sigma.set(0, 0, f64::from(s_data[0]));
        sigma.set(1, 1, f64::from(s_data[1]));
        let reconstructed = u_mat
            .matmul(&sigma)
            .matmul(&Mat::from_tensor(&vh))
            .to_tensor()
            .unwrap();
        approx_eq(&reconstructed, &a, 1e-3);
    }

    /// 上記と同型だが、列が「正確な比例関係だが float 演算では第二
    /// 特異値が厳密に 0 にならない」ケース（`col1 = 0.3f32 * col0` は
    /// 成分ごとに丸め誤差を持ち、`1.5f32 != 5.0f32 * 0.3f32` である
    /// ため σ2 は数値的に非ゼロになりうる）。この場合でも `U` の列
    /// 直交性は成立し、かつ σ2 自体は非ゼロ（≈ 0 へ強制的に丸めて
    /// いない）ことを確認する——固定・相対いずれのしきい値でも
    /// 非ゼロな小さい特異値を誤ってゼロ化しない契約（Ypd/SCp 系の
    /// codex-review 指摘）の回帰。
    #[test]
    fn svd_near_rank_deficient_tall_matrix_does_not_zero_out_nonzero_sigma() {
        let a = build_tensor(vec![1.0, 0.3, 2.0, 0.6, 5.0, 1.5], &[3, 2]).unwrap();
        let (u, s, vh) = svd(&a).unwrap();
        let s_data = dense_vec(&s);
        assert!(s_data[0] > 0.0, "第一特異値が非ゼロであるべき: {s_data:?}");
        assert!(
            s_data[1] > 0.0,
            "float 丸め由来の非ゼロ第二特異値をゼロ化してはいけない: {s_data:?}"
        );
        let u_mat = Mat::from_tensor(&u);
        let col0 = u_mat.col(0);
        let col1 = u_mat.col(1);
        let norm0: f64 = col0.iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm1: f64 = col1.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            (norm0 - 1.0).abs() < 1e-4,
            "U 列 0 が単位ノルムでない: norm={norm0}"
        );
        assert!(
            (norm1 - 1.0).abs() < 1e-4,
            "U 列 1 が単位ノルムでない: norm={norm1}"
        );
        let dot: f64 = col0.iter().zip(col1.iter()).map(|(a, b)| a * b).sum();
        assert!(dot.abs() < 1e-6, "U の列が直交していない: dot={dot}");
        let mut sigma = Mat::zeros(2, 2);
        sigma.set(0, 0, f64::from(s_data[0]));
        sigma.set(1, 1, f64::from(s_data[1]));
        let reconstructed = u_mat
            .matmul(&sigma)
            .matmul(&Mat::from_tensor(&vh))
            .to_tensor()
            .unwrap();
        approx_eq(&reconstructed, &a, 1e-3);
    }
}
