//! BLIS/GotoBLAS2 5-loop model（jc→pc→ic→jr→ir）による GEMM（TASK-1.6f・#184）。
//!
//! [`crate::gemm`] の `gemm_naive`／`gemm_blocked`／`gemm_parallel`（TASK-1.6a）
//! は自動ベクトル化頼みのスカラーループで、PoC-v2-1 実測では対 PyTorch CPU 比
//! 5.3%（M=N=K=2048/4096 最小値）に留まり REQ-8 の CPU 最適化後下限（20%）に
//! 届かない。本モジュールは `std::arch` intrinsics（NEON／AVX2+FMA）による
//! マイクロカーネル・A/B packing（[`pack`]）・キャッシュ階層ブロッキング
//! （MC/KC/NC）を実装し、性能向上を狙う。
//!
//! ## 公開 API 非破壊（既存 3 関数は変更しない）
//!
//! [`crate::gemm::gemm_naive`]／`gemm_blocked`／`gemm_parallel` は #24
//! （TASK-1.6d・PoC-v2-1 比 3 段階性能確認）の参照点として変更しない
//! （公開 API 非破壊はガードレール条件・`.claude/rules/security.md`）。
//! 本モジュールは新規に [`gemm_blis`]／[`gemm_blis_parallel`] を追加する。
//!
//! ## bit 完全一致契約（REQ-2）
//!
//! [`microkernel`] の各カーネルは C 要素ごとの累積を p 昇順の FMA 連鎖で
//! 行い、レーン間縮約（split-k 等）を一切行わない設計とすることで、
//! `gemm_naive` と bit 完全一致が成立する（`tests/gemm_blis_parity.rs`）。
//! 累積順序を変える最適化（split-k・ゼロ初期化してからの後加算方式）は
//! 本契約を壊すため、将来追加する場合は数値一致テストの契約変更として
//! ユーザー承認事項である。
//!
//! ## 境界検査（REQ-8）
//!
//! 公開入口は [`crate::gemm::validate_dims`]（`checked_mul` によるオーバー
//! フロー検査・スライス長検査）を再利用する。packing・端タイルの C
//! 書き戻しは安全な slice 操作で行い、intrinsics のロード／ストアは
//! マイクロカーネル関数入口の `assert!` で長さを検査した直後の最小
//! `unsafe` ブロックに限定する（[`microkernel::neon`]／[`microkernel::avx2`]
//! 参照）。

pub mod microkernel;
mod pack;

use crate::gemm::{GemmError, validate_dims};
use microkernel::{MR, NR, kernel as run_microkernel};
use pack::{pack_a, pack_b};
use rayon::prelude::*;

/// キャッシュブロッキングの行方向ブロックサイズ（A のパネル高さ）。
///
/// [`crate::gemm`] の既存定数（PoC-v2-1 実測環境で選定）を起点値として
/// 踏襲する。マイクロカーネルの MR/NR は ISA ごとに異なる（[`microkernel`]
/// 参照）ため、本モジュール独自の定数として持つ（`gemm` モジュールの
/// MC/KC/NC とは独立にチューニング可能にする）。再チューニングは #24 の
/// スコープ。
const MC: usize = 128;
/// 縮約次元（K）のブロックサイズ。
const KC: usize = 256;
/// 列方向ブロックサイズ（B のパネル幅）。
const NC: usize = 512;

/// 単一スレッドの BLIS 5-loop GEMM（jc→pc→ic→jr→ir）。
///
/// `gemm_blis_parallel` はこの関数の内部ロジック（[`gemm_blis_region`]）を
/// 行パネルごとに並列呼び出しすることで並列化する（`gemm_blocked`／
/// `gemm_parallel` と同じ構成。`crate::gemm` 参照）。
pub fn gemm_blis(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;
    gemm_blis_region(a, b, c, n, k, 0, m);
    Ok(())
}

/// `gemm_blis` を `rayon` で行パネル並列化した版。
///
/// C を行方向にパネル分割し、各パネルを独立スレッドで [`gemm_blis_region`]
/// に渡す（`crate::gemm::gemm_parallel` と同じ並列化戦略。C の書き込み
/// 範囲がパネルごとに排他的なためデータ競合なし）。
pub fn gemm_blis_parallel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(), GemmError> {
    validate_dims(a, b, c, m, n, k)?;

    // n == 0 は shape として合法だが par_chunks_mut(panel_rows * n) は
    // チャンクサイズ 0 でパニックする（`crate::gemm::gemm_parallel` と
    // 同じ理由・同じ対処。3 実装間の parity 契約を保つため no-op で返す）。
    if n == 0 {
        return Ok(());
    }

    let num_threads = rayon::current_num_threads().max(1);
    let panel_rows = m.div_ceil(num_threads).max(1);

    c.par_chunks_mut(panel_rows * n)
        .enumerate()
        .for_each(|(panel_idx, c_chunk)| {
            let row_start = panel_idx * panel_rows;
            let row_end = (row_start + c_chunk.len() / n).min(m);
            gemm_blis_region(a, b, c_chunk, n, k, row_start, row_end);
        });
    Ok(())
}

/// `gemm_blis`／`gemm_blis_parallel` 共通の 5-loop 本体。`c` はパネル
/// 先頭が `row_start` 行目に対応するスライス（`crate::gemm::gemm_blocked_region`
/// と同じ相対オフセット規約）。引数検証は呼び出し元の公開入口で完了済み
/// の前提。
///
/// ループ順は jc（列ブロック）→ pc（K ブロック）→ ic（行ブロック）→
/// jr（NR 単位の列パネル）→ ir（MR 単位の行パネル）。B パネルは pc/jc
/// ブロックごとに 1 回だけ packing して ic ループ全体で再利用し、A パネルは
/// ic ブロックごとに 1 回だけ packing して jr ループ全体で再利用する
/// （BLIS/GotoBLAS2 の packing 再利用による bandwidth 削減。PoC-v2-1
/// README「設計判断」節と同じ狙い）。
fn gemm_blis_region(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    n: usize,
    k: usize,
    row_start: usize,
    row_end: usize,
) {
    let mc_total = row_end - row_start;
    let a = &a[row_start * k..];

    for jc in (0..n).step_by(NC) {
        let nc_len = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let kc_len = KC.min(k - pc);

            // B パネル packing: nc_len を NR 単位のブロックに分割し、各
            // ブロックを kc_len*NR 要素の連続領域として 1 本のバッファに
            // 詰める（ic ループ全体で使い回すため pc/jc ブロックごとに
            // 1 回のみ実行）。
            let nr_blocks = nc_len.div_ceil(NR);
            let mut b_panel = vec![0.0f32; nr_blocks * kc_len * NR];
            for jr_block in 0..nr_blocks {
                let jr = jr_block * NR;
                let nr_eff = NR.min(nc_len - jr);
                let bp = pack_b(b, n, pc, kc_len, jc + jr, NR, nr_eff);
                b_panel[jr_block * kc_len * NR..(jr_block + 1) * kc_len * NR].copy_from_slice(&bp);
            }

            let mut ic = 0;
            while ic < mc_total {
                let mc_len = MC.min(mc_total - ic);

                // A パネル packing: mc_len を MR 単位のブロックに分割
                // （jr ループ全体で使い回すため ic ブロックごとに 1 回のみ）。
                let mr_blocks = mc_len.div_ceil(MR);
                let mut a_panel = vec![0.0f32; mr_blocks * kc_len * MR];
                for ir_block in 0..mr_blocks {
                    let ir = ir_block * MR;
                    let mr_eff = MR.min(mc_len - ir);
                    let ap = pack_a(a, k, ic + ir, MR, mr_eff, pc, kc_len);
                    a_panel[ir_block * kc_len * MR..(ir_block + 1) * kc_len * MR]
                        .copy_from_slice(&ap);
                }

                for jr_block in 0..nr_blocks {
                    let jr = jr_block * NR;
                    let nr_eff = NR.min(nc_len - jr);
                    let bp_slice = &b_panel[jr_block * kc_len * NR..(jr_block + 1) * kc_len * NR];

                    for ir_block in 0..mr_blocks {
                        let ir = ir_block * MR;
                        let mr_eff = MR.min(mc_len - ir);
                        let ap_slice =
                            &a_panel[ir_block * kc_len * MR..(ir_block + 1) * kc_len * MR];

                        // C タイルの現在値をロード（複数 pc ブロックに
                        // またがる累積を成立させるため、ゼロ初期化せず
                        // 実際の現在値を読み込む）。padding レーン
                        // （mr_eff..MR, nr_eff..NR）はゼロのままでよい
                        // （書き戻し時に有効部のみコピーするため不使用）。
                        // `MR * NR` は const 式のためスタック配列で確保する
                        // （ir_block×jr_block のたびにヒープ確保しない。
                        // Review 指摘: M=N=K=2048 では ir/jr ループの
                        // 反復数が数十万に達し `Vec` 確保が無視できない
                        // オーバーヘッドになるため）。
                        let mut c_tile = [0.0f32; MR * NR];
                        let col_base = jc + jr;
                        for i in 0..mr_eff {
                            let src = &c[(ic + ir + i) * n + col_base
                                ..(ic + ir + i) * n + col_base + nr_eff];
                            c_tile[i * NR..i * NR + nr_eff].copy_from_slice(src);
                        }

                        run_microkernel(ap_slice, bp_slice, &mut c_tile, kc_len);

                        for i in 0..mr_eff {
                            let dst = &mut c[(ic + ir + i) * n + col_base
                                ..(ic + ir + i) * n + col_base + nr_eff];
                            dst.copy_from_slice(&c_tile[i * NR..i * NR + nr_eff]);
                        }
                    }
                }

                ic += MC;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_blis_matches_hand_computed_2x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        gemm_blis(&a, &b, &mut c, 2, 2, 2).unwrap();
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn gemm_blis_rejects_a_len_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut c = vec![0.0; 4];
        let err = gemm_blis(&a, &b, &mut c, 2, 2, 2).unwrap_err();
        assert!(matches!(
            err,
            GemmError::ALenMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    // MC/KC/NC 境界を跨ぐ多ブロック形状・並列版との bit 完全一致は
    // `bench_harness`（乱数生成）を要するため、lib 単体テスト
    // （本 `mod tests`）ではなく統合テスト `tests/gemm_blis_parity.rs`
    // 側に集約する。理由: `bench_harness` は `serde_json` を推移依存に
    // 持ち、lib 単体テストバイナリへ持ち込むと `reduction.rs` 側の
    // 無関係な `assert_eq!(&[usize], &[])` が `usize: PartialEq<_>` の
    // 複数実装（`core` と `serde_json::Value` 向け）で型推論あいまいに
    // なり `E0282/E0283` を起こす（同一バイナリにリンクされる依存の
    // trait 実装がクレート全体で可視になるため。実装時に実測確認済み）。
    // 統合テスト（`tests/`）は個別バイナリのためこの問題が生じない。

    #[test]
    fn gemm_blis_parallel_handles_zero_n_as_noop() {
        let a = vec![1.0f32; 4];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        assert!(gemm_blis_parallel(&a, &b, &mut c, 2, 0, 2).is_ok());
    }

    #[test]
    fn gemm_blis_handles_zero_dims() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];
        let mut c: Vec<f32> = vec![];
        assert!(gemm_blis(&a, &b, &mut c, 0, 0, 0).is_ok());
    }
}
