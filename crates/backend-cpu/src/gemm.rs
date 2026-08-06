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
/// 試し採用した値（選定根拠は PoC README「設計判断」節に記録）。
///
/// #24 実測（`docs/perf/cpu-gemm-rayon-tuning.md`。QEMU Virtual CPU 12
/// 論理コア環境、M=N=K=2048 での MC/KC/NC 近傍水準の座標降下法スイープ）
/// では、この PoC 値の近傍に有意な改善が見られなかったため現行値を
/// 維持した（実測差が Q1〜Q3 幅の内側）。[`BlockSizes`] へパラメータ化した
/// のは、以降のスイープ実測を再コンパイルなしで `examples/gemm_bench.rs`
/// から行えるようにするためであり、既定値としては本 PoC 値を使い続ける。
const MC: usize = 128;
/// 縮約次元（K）のブロックサイズ。B パネル（KC×NC×4B）が L1/L2 に収まる値。
const KC: usize = 256;
/// 列方向ブロックサイズ（B のパネル幅）。
const NC: usize = 512;

/// キャッシュブロッキングのブロックサイズ 3 つ組（MC/KC/NC）。
///
/// `gemm_blocked`／`gemm_parallel` は既定値（[`BlockSizes::poc_v2_1_default`]。
/// 上記 `MC`/`KC`/`NC` 定数）で呼ばれる。本構造体でパラメータ化した目的は
/// #24（TASK-1.6d）のチューニングスイープを `examples/gemm_bench.rs` から
/// 再コンパイルなしで実行できるようにするためであり、`gemm_blocked_region`
/// の内部ロジック自体は変更していない（K 方向の加算順序は不変のため
/// `tests/gemm_parity.rs` の bit-exact 契約に影響しない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockSizes {
    pub mc: usize,
    pub kc: usize,
    pub nc: usize,
}

impl BlockSizes {
    /// PoC-v2-1 実測に基づく既定値（`MC`/`KC`/`NC` 定数と同一）。
    pub const fn poc_v2_1_default() -> Self {
        Self {
            mc: MC,
            kc: KC,
            nc: NC,
        }
    }
}

impl Default for BlockSizes {
    fn default() -> Self {
        Self::poc_v2_1_default()
    }
}

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
    // 単一スレッド実行は「パネルが 1 枚（行範囲 [0, m) 全体）だけの並列実行」
    // と等価なため、`gemm_parallel` と共通のブロッキング本体
    // （`gemm_blocked_region`）をそのまま呼ぶ（row_start=0 なので `a`・`c`
    // とも絶対オフセット＝相対オフセットが一致する）。以前は本関数専用の
    // 絶対オフセット版を別途持っていたが、パネル分割版とロジックが
    // 完全重複していたため統合した（issue #21 レビュー指摘の Low 項目）。
    gemm_blocked_region(a, b, c, n, k, 0, m, BlockSizes::poc_v2_1_default());
    Ok(())
}

/// `gemm_blocked`／`gemm_parallel` 共通のブロッキング本体。`c` はパネル
/// 先頭が `row_start` 行目に対応するスライス（`gemm_blocked` からは行列
/// 全体、`gemm_parallel` からは `par_chunks_mut` が返す行パネル）である
/// ことを想定し、`c` 内部では `row_start` からの相対オフセットでアクセス
/// する。`a` は関数内部で `row_start*k` を先頭にスライスし直すことで、
/// `c` と同じ相対オフセットが `a` にも成立するようにする（`gemm_blocked`
/// は `row_start=0`・`c`＝行列全体を渡すことで絶対オフセットのケースを
/// 兼ねる）。引数検証は呼び出し元の公開入口（`gemm_blocked`・
/// `gemm_parallel`）で完了済みの前提。
#[allow(clippy::too_many_arguments)]
fn gemm_blocked_region(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
    blocks: BlockSizes,
) {
    let mc_total = row_end - row_start;
    let a = &a[row_start * k..];

    for jc in (0..n).step_by(blocks.nc) {
        let nc_len = blocks.nc.min(n - jc);
        for pc in (0..k).step_by(blocks.kc) {
            let kc_len = blocks.kc.min(k - pc);
            let mut ic = 0;
            while ic < mc_total {
                let mc_len = blocks.mc.min(mc_total - ic);
                kernel_block(a, b, c, n, k, ic, ic + mc_len, pc, kc_len, jc, nc_len);
                ic += blocks.mc;
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
    // オーバーサブスクリプション係数 1（PoC-v2-1 実証構成のまま）・
    // 既定ブロックサイズで `gemm_parallel_tuned` を呼ぶ薄いラッパー。
    // #24 実測（`docs/perf/cpu-gemm-rayon-tuning.md`）で係数 2・4 との
    // 有意差が確認できなかったため、公開 API の既定挙動は変更していない。
    gemm_parallel_tuned(a, b, c, m, n, k, BlockSizes::poc_v2_1_default(), 1)
}

/// `gemm_parallel` のチューニング可能版。ブロックサイズ・パネル分割の
/// オーバーサブスクリプション係数を明示指定できる。
///
/// **想定呼び出し元**: `examples/gemm_bench.rs`（#24・TASK-1.6d の A/B 実測
/// スイープ）。`gemm_parallel` 自身もこの関数を既定パラメータで呼ぶ薄い
/// ラッパーであるため、チューニング用エントリポイントを追加しても
/// 本番経路のカーネル本体（`gemm_blocked_region`・`kernel_block`）は
/// 単一の実装のまま保たれる。
///
/// `oversubscription`（1 以上）は生成するパネル数を
/// `num_threads * oversubscription` に増やすことで、rayon の
/// work-stealing による負荷平準化の効果を実測するための係数
/// （PoC-v2-1 は係数 1 相当の「スレッド数と同数のパネル」構成のみを
/// 検証していた）。
///
/// `#[allow(clippy::too_many_arguments)]`: `blocks`・`oversubscription` を
/// 個別引数のまま追加したのは、`gemm_parallel` からの薄い委譲呼び出しで
/// 各引数の対応が一目で分かるようにするため（`kernel_block` の同種
/// `#[allow]` と同じ理由。`.claude/rules/coding-rust.md`）。
#[allow(clippy::too_many_arguments)]
pub fn gemm_parallel_tuned(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
    blocks: BlockSizes,
    oversubscription: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;

    // n == 0 は shape として合法（validate_dims は m*n=0・k*n=0 を許容し、
    // b・c は空スライスになる）だが、下記 par_chunks_mut(panel_rows * n) は
    // チャンクサイズが 0 だとパニックする。naive／blocked は該当形状で
    // ループが単に 0 回になる no-op として振る舞うため（`gemm_blocked_region`
    // の `for jc in (0..n).step_by(blocks.nc)` が空レンジになる）、3 実装間の
    // parity 契約（`tests/gemm_parity.rs`）を保つよう本関数も明示的に
    // no-op で返す。新規エラー種別を追加しないのは、この形状自体は
    // 不正ではなく他 2 実装が正常に受理しているため。
    if n == 0 {
        return Ok(());
    }

    // 論理コア数（P+E）× オーバーサブスクリプション係数ぶんのパネル数を
    // 確保することを優先し、各パネルの行数は「m をパネル数で割った切り
    // 上げ」で決める。パネル行数の下限を MC（128）に固定しないのは、
    // 例えば M=512・16 スレッドでは 512/16=32 行のパネルが作れるにも
    // かかわらず MC=128 に切り上げると 4 パネルしか生成されず、16 コア中
    // 4 コアしか稼働しない頭打ちが生じるため（`gemm_blocked_region` は
    // パネル内部で MC 単位のブロッキングを自前で行うため、パネル行数が
    // MC 未満でも正しく動作する。PoC-v2-1 実測で確立した方針）。
    let num_threads = rayon::current_num_threads().max(1);
    let panel_count = num_threads.saturating_mul(oversubscription.max(1)).max(1);
    let panel_rows = m.div_ceil(panel_count).max(1);

    c.par_chunks_mut(panel_rows * n)
        .enumerate()
        .for_each(|(panel_idx, c_chunk)| {
            let row_start = panel_idx * panel_rows;
            let row_end = (row_start + c_chunk.len() / n).min(m);
            // c_chunk はパネル先頭が row_start 行目に対応するスライス。
            // gemm_blocked_region が row_start を基準に a・c 双方の相対
            // オフセットを揃えるため、パネル境界をそのまま渡せる。
            gemm_blocked_region(a, b, c_chunk, n, k, row_start, row_end, blocks);
        });
    Ok(())
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
