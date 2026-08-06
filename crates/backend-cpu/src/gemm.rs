//! CPU GEMM カーネル 3 種（naive / blocked / parallel）。
//!
//! `backend-cpu` の数値一致の参照点となる GEMM 実装（REQ-2）。TASK-1.9
//! （#43・`BackendOps` トレイト導入）から `gemm_parallel` が最終形として
//! 呼ばれる想定で、3 関数を公開のまま残すのは #24（PoC-v2-1 比
//! naive/blocked 比 約 6〜8.5 倍改善の再現確認）が 3 段階比較を
//! 必要とするため（本クレートの雛形段階では未結線。呼び出し文脈は
//! `docs/spec/05-tasks.md` TASK-1.6・TASK-1.9 参照）。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/rust/src/gemm.rs`。
//! PoC は 3 段階（素朴実装 → キャッシュ/レジスタブロッキング → `rayon`
//! 並列化）を独立した関数として残し、各段の性能寄与を要因分解できる
//! ようにしている（PoC README「設計判断」節）。productize にあたり
//! 以下 2 点を変更した:
//!
//! 1. **FMA 契約統一**（REQ-2・`.claude/rules/coding-rust.md`）: PoC の
//!    内側ループは `c += a * b`（乗算・加算を別々に丸める）だったが、
//!    本実装は `f32::mul_add` に置き換え、GPU 側（CUDA NVRTC・Metal
//!    `simdgroup_multiply_accumulate`）の既定 FMA 契約と揃える
//!    （PoC-v2-5 の K=4096 ストレスケースで `mul_add` 化により GPU と
//!    完全一致〈fail_cells=0/262144〉を実測確認済み）。
//! 2. **型付きエラー化**（coding-rust.md「本番経路で unwrap/expect を
//!    使わない」）: PoC の `debug_assert` による shape 検査を廃し、
//!    公開入口で [`GemmError`] を返す検証に置き換えた。
//!
//! すべて `unsafe` を使わない安全な Rust で実装する（境界検査は
//! コンパイラの範囲検査＋入口の明示検証に委ね、最適化を理由に手動境界
//! チェックを省略しない。REQ-8・`.claude/rules/coding-rust.md`）。

use rayon::prelude::*;
use std::fmt;

/// キャッシュブロッキングの行方向ブロックサイズ（A のパネル高さ）。
///
/// PoC-v2-1 実測環境（Apple M4 Max。P コア L1D 128KiB・L2 はコアクラスタ
/// 共有 16MiB 級）で、A パネル（MC×KC×4B）が L2 に収まる範囲で複数水準を
/// 試し採用した値（選定根拠は PoC README「設計判断」節に記録。
/// 再チューニングは #24 のスコープ）。
const MC: usize = 128;
/// 縮約次元（K）のブロックサイズ。B パネル（KC×NC×4B）が L1/L2 に収まる値。
const KC: usize = 256;
/// 列方向ブロックサイズ（B のパネル幅）。
const NC: usize = 512;

/// GEMM カーネル公開入口の shape 検証エラー。
///
/// `#[non_exhaustive]` を付す理由: 公開 API 非破壊はガードレール条件
/// （`.claude/rules/security.md`）であり、後続タスクで検査項目が増えても
/// 呼び出し側の網羅的 match を破壊しないため（`tensor-core::ShapeError`
/// と同方針。`crates/tensor-core/src/error.rs` 参照）。共通化（`tensor-core`
/// への統合）は #25 以降で判断する。
#[non_exhaustive]
#[derive(Debug)]
pub enum GemmError {
    /// `a` の要素数が `m * k` と一致しない。
    ALenMismatch { expected: usize, actual: usize },
    /// `b` の要素数が `k * n` と一致しない。
    BLenMismatch { expected: usize, actual: usize },
    /// `c` の要素数が `m * n` と一致しない。
    CLenMismatch { expected: usize, actual: usize },
    /// `m`・`k`・`n` の積のいずれかが `usize` の範囲でオーバーフローする
    /// （`checked_mul` によりアクセス前に検出する。OWASP A03 観点。
    /// `.claude/rules/security.md`）。
    DimProductOverflow,
}

impl fmt::Display for GemmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GemmError::ALenMismatch { expected, actual } => {
                write!(f, "a length mismatch: expected {expected}, actual {actual}")
            }
            GemmError::BLenMismatch { expected, actual } => {
                write!(f, "b length mismatch: expected {expected}, actual {actual}")
            }
            GemmError::CLenMismatch { expected, actual } => {
                write!(f, "c length mismatch: expected {expected}, actual {actual}")
            }
            GemmError::DimProductOverflow => {
                write!(f, "m*k, k*n or m*n overflows usize")
            }
        }
    }
}

impl std::error::Error for GemmError {}

/// 公開入口共通の shape 検証。`m*k`／`k*n`／`m*n` を `checked_mul` で
/// 算出し、オーバーフローとスライス長不整合を本体アクセス前に拒否する
/// （REQ-8・security.md「外部フォーマットパースは長さ・形状の検証を
/// 先に行う」と同じ思想を GEMM カーネル入口に適用）。
fn validate_dims(
    a: &[f32],
    b: &[f32],
    c: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    let mk = m.checked_mul(k).ok_or(GemmError::DimProductOverflow)?;
    let kn = k.checked_mul(n).ok_or(GemmError::DimProductOverflow)?;
    let mn = m.checked_mul(n).ok_or(GemmError::DimProductOverflow)?;

    if a.len() != mk {
        return Err(GemmError::ALenMismatch {
            expected: mk,
            actual: a.len(),
        });
    }
    if b.len() != kn {
        return Err(GemmError::BLenMismatch {
            expected: kn,
            actual: b.len(),
        });
    }
    if c.len() != mn {
        return Err(GemmError::CLenMismatch {
            expected: mn,
            actual: c.len(),
        });
    }
    Ok(())
}

/// 素朴な 3 重ループ GEMM（`C += A @ B`、ikj 順）。参照実装として
/// blocked／parallel 版の正しさの基準になる（`tests/gemm_parity.rs`）。
///
/// `c` は呼び出し前にゼロ初期化されている前提（本関数は加算のみ行う）。
/// ikj 順を採るのは、内側 j ループが `b`/`c` の行を連続アクセスするため
/// キャッシュライン再利用の基本を満たすため（それ以上の最適化はしない
/// 「素朴版」としての位置づけ。PoC-v2-1 設計方針を踏襲）。
pub fn gemm_naive(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;

    for i in 0..m {
        let a_row = &a[i * k..i * k + k];
        let c_row = &mut c[i * n..i * n + n];
        for (p, &a_ip) in a_row.iter().enumerate() {
            let b_row = &b[p * n..p * n + n];
            for j in 0..n {
                // FMA 契約統一（REQ-2）: GPU 側の既定 FMA 契約と揃えるため
                // `mul_add` を用いる（PoC の `c += a * b` から変更した箇所）。
                c_row[j] = a_ip.mul_add(b_row[j], c_row[j]);
            }
        }
    }
    Ok(())
}

/// C の `[row_start, row_end) × [jc, jc+nc_len)` ブロックへ、K 範囲
/// `[pc, pc+kc_len)` ぶんの寄与を加算するマイクロカーネル。
///
/// **設計変更の経緯**（PoC-v2-1 README「設計判断」節）: 当初は 4×4 の
/// レジスタブロッキング（ローカル配列にアキュムレートしてから書き戻す
/// 方式）を実装したが、実測したところ naive 版より遅くなる逆転現象が
/// 生じた。原因は、内側ループを `p`（K 方向）を最外周にした構成に
/// したため、A の同一 p に対する複数行アクセスが `a_row_stride`
/// （= K 全体幅、数百〜数千要素）分のストライドアクセスになり、かつ
/// `.iter_mut().enumerate().take(n)` の境界検査でオートベクトル化が
/// 阻害されていたためと考えられる（具体的な原因切り分けは未実施で、
/// 対処として下記の構成に置き換えた）。本カーネルは naive と同じ
/// 「行 i を外側、K を中側、N を内側（連続アクセス）」の ikj 順を、
/// キャッシュブロック単位に限定して適用する構成に改め、naive が持つ
/// 良好なメモリアクセスパターンをブロック内でも維持する。
///
/// `#[allow(clippy::too_many_arguments)]`: ブロック境界（row/pc/kc_len/
/// jc/nc_len）を個別引数として持つことで呼び出し側の意図が明確になる
/// ため、構造体へのまとめ込みは行わない（内部関数限定。理由コメント
/// 必須のルール `.claude/rules/coding-rust.md` に対応）。
#[allow(clippy::too_many_arguments)]
fn kernel_block(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize, // C・B の全体行幅（連続バッファ内でのストライド）
    k: usize, // A の全体行幅
    row_start: usize,
    row_end: usize,
    pc: usize,
    kc_len: usize,
    jc: usize,
    nc_len: usize,
) {
    for i in row_start..row_end {
        let a_row = &a[i * k + pc..i * k + pc + kc_len];
        let c_row = &mut c[i * n + jc..i * n + jc + nc_len];
        for (p, &a_ip) in a_row.iter().enumerate() {
            let b_row = &b[(pc + p) * n + jc..(pc + p) * n + jc + nc_len];
            for (c_val, &b_val) in c_row.iter_mut().zip(b_row.iter()) {
                // FMA 契約統一（REQ-2）。PoC の `*c_val += a_ip * b_val` から変更。
                *c_val = a_ip.mul_add(b_val, *c_val);
            }
        }
    }
}

/// キャッシュブロッキング（MC/KC/NC）を適用した単一スレッド GEMM。
/// `gemm_parallel` はこの関数の内部ロジック（[`gemm_blocked_region`]）を
/// 行パネルごとに並列呼び出しすることで並列化する。
pub fn gemm_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    gemm_blocked_region(a, b, c, m, n, k, 0, m);
    Ok(())
}

/// `gemm_blocked` の本体。行範囲 `[row_start, row_end)` のみを計算する形に
/// 切り出してあるのは、`gemm_parallel` が行パネルごとに同じロジックを
/// 再利用できるようにするため（naive/blocked/parallel でカーネル自体は
/// 共通化し、並列化の有無だけを差分にする。PoC-v2-1 設計方針）。
/// 引数検証は呼び出し元の公開入口（`gemm_blocked`）で完了済みの前提。
#[allow(clippy::too_many_arguments)]
fn gemm_blocked_region(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    debug_assert!(row_end <= m);

    for jc in (0..n).step_by(NC) {
        let nc_len = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let kc_len = KC.min(k - pc);
            let mut ic = row_start;
            while ic < row_end {
                let mc_len = MC.min(row_end - ic);
                kernel_block(a, b, c, n, k, ic, ic + mc_len, pc, kc_len, jc, nc_len);
                ic += MC;
            }
        }
    }
}

/// `gemm_blocked` を `rayon` で行パネル並列化した版。`BackendOps::gemm`
/// （TASK-1.9・#43 で結線予定）の最終形カーネルとなる想定。
///
/// C を行方向にパネル分割し、各パネルを独立スレッドで
/// `gemm_blocked_region` に渡す。C の書き込み範囲がパネルごとに排他的な
/// ため、`par_chunks_mut` によるデータ競合のない安全な並列化が成立する
/// （PoC-v2-1 実証構成。`rayon` は許容依存 CPU 並列区分。
/// `.claude/rules/deps-policy.md`）。
pub fn gemm_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;

    // 論理コア数（P+E）ぶんのパネル数を確保することを優先し、各パネルの
    // 行数は「m を num_threads で割った切り上げ」で決める。パネル行数の
    // 下限を MC（128）に固定しないのは、例えば M=512・16 スレッドでは
    // 512/16=32 行のパネルが作れるにもかかわらず MC=128 に切り上げると
    // 4 パネルしか生成されず、16 コア中 4 コアしか稼働しない頭打ちが
    // 生じるため（`gemm_blocked_region_relative` はパネル内部で MC 単位の
    // ブロッキングを自前で行うため、パネル行数が MC 未満でも正しく動作
    // する。PoC-v2-1 実測で確立した方針）。
    let num_threads = rayon::current_num_threads().max(1);
    let panel_rows = m.div_ceil(num_threads).max(1);

    c.par_chunks_mut(panel_rows * n)
        .enumerate()
        .for_each(|(panel_idx, c_chunk)| {
            let row_start = panel_idx * panel_rows;
            let row_end = (row_start + c_chunk.len() / n).min(m);
            // c_chunk は c[row_start*n .. row_end*n] と同じメモリだが、
            // gemm_blocked_region は c 全体を絶対オフセットで参照するため、
            // ここではパネル先頭からの相対オフセットに合わせて呼び出す。
            gemm_blocked_region_relative(a, b, c_chunk, n, k, row_start, row_end);
        });
    Ok(())
}

/// `gemm_blocked_region` の相対オフセット版。`c_chunk` はパネル先頭が
/// `row_start` 行目に対応するスライスであるため、内部で絶対行番号との
/// ずれを吸収する。呼び出し元（`gemm_parallel`）で shape 検証済みの前提。
#[allow(clippy::too_many_arguments)]
fn gemm_blocked_region_relative(
    a: &[f32],
    b: &[f32],
    c_chunk: &mut [f32],
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    let mc_total = row_end - row_start;
    for jc in (0..n).step_by(NC) {
        let nc_len = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let kc_len = KC.min(k - pc);
            let mut ic = 0;
            while ic < mc_total {
                let mc_len = MC.min(mc_total - ic);
                // a は絶対行番号（row_start 起点）、c_chunk はパネル先頭を
                // 0 とした相対行番号でアクセスする必要があるため、
                // kernel_block を 2 回に分けず「a 側オフセット + c 側オフセット」
                // をそれぞれのスライスに事前適用してから共通ロジックを呼ぶ。
                kernel_block(
                    &a[row_start * k..],
                    b,
                    c_chunk,
                    n,
                    k,
                    ic,
                    ic + mc_len,
                    pc,
                    kc_len,
                    jc,
                    nc_len,
                );
                ic += MC;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 小規模ケースを手計算・既知値と照合する（naive の正しさの根拠。
    /// PoC-v2-1 テストベクタを移植）。
    #[test]
    fn gemm_naive_matches_hand_computed_2x2() {
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
        // A@B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        gemm_naive(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_naive_rejects_a_len_mismatch() {
        let a = vec![1.0, 2.0, 3.0]; // m*k = 4 を期待
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        let err = gemm_naive(&a, &b, &mut c, 2, 2, 2).unwrap_err();
        assert!(matches!(
            err,
            GemmError::ALenMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[test]
    fn gemm_naive_rejects_dim_product_overflow() {
        let a = vec![0.0f32; 1];
        let b = vec![0.0f32; 1];
        let mut c = vec![0.0f32; 1];
        let err = gemm_naive(&a, &b, &mut c, usize::MAX, 2, 2).unwrap_err();
        assert!(matches!(err, GemmError::DimProductOverflow));
    }
}
