//! 全 arch 共通のスカラーフォールバックマイクロカーネル（MR=4×NR=4）。
//!
//! `unsafe` を使わない安全な Rust のみで実装する。x86_64 で AVX2+FMA が
//! コンパイル時に有効でない環境・NEON を持たない arch（`microkernel::mod`
//! の cfg 選択がここへフォールバックする条件は同モジュールのドキュメント
//! 参照）で `gemm_blis` の既定経路として使われる。intrinsics 版
//! （[`super::neon`]・[`super::avx2`]）と累積順序（p 昇順の `mul_add`
//! 連鎖）を完全に揃えることで、`gemm_naive` との bit 完全一致契約
//! （REQ-2・PoC-v2-5 の FMA 契約統一）を ISA 間で共有する。

/// マイクロカーネルタイルの行数（C の m 方向レーン数）。
pub const MR: usize = 4;
/// マイクロカーネルタイルの列数（C の n 方向レーン数）。
pub const NR: usize = 4;

// [`super::super::gemm_blis_region`] の C タイルスタックバッファは
// `MAX_TILE`（256 要素・AVX-512 の 8×32 が最大）固定長で確保するため、
// 全 ISA の MR*NR がこれを超えないことをコンパイル時に検査する（#185）。
const _: () = assert!(MR * NR <= 256);

/// `ap`（packed A、MR 行×kc_len、p-major）・`bp`（packed B、kc_len×NR、
/// p-major）から MR×NR の C タイル `c`（row-major、行ストライド `ldc`）へ
/// `kc_len` ぶんの寄与を加算する。
///
/// 呼び出し元（[`super::super::mod`] の 5-loop ドライバ）は本関数呼び出し
/// 前に `c` へ実際の C の現在値をロードし、呼び出し後に書き戻す
/// （複数の `pc` ブロックにまたがる累積を成立させるため）。#557 により
/// 完全タイルは C の実バッファへ `ldc = n` で直接読み書きし、端タイルは
/// 従来どおり `MAX_TILE` スタックバッファへ `ldc = NR` でコピー往復する
/// （[`super::Microkernel::run`] の `ldc` 契約参照）。
///
/// # 累積順序（bit 完全一致契約）
///
/// 各 `(i, j)` に対し `c[i*ldc+j] = a[p][i].mul_add(b[p][j], c[i*ldc+j])`
/// を `p` 昇順に適用する。この順序は [`crate::gemm::gemm_naive`] が
/// 内側ループで行う蓄積順序と同一であり、他の `(i, j)` との縮約
/// （split-k 等）を一切行わないため、`gemm_naive` と bit 完全一致が
/// 成立する（`tests/gemm_blis_parity.rs` の `assert_eq!` 契約）。`ldc` の
/// 導入で変わるのはロード/ストアのアドレッシングのみで演算値・順序は
/// 不変のため、この契約は `ldc` に依らず成立する。
///
/// # Panics
///
/// `ap.len() != MR * kc_len`／`bp.len() != kc_len * NR` であればパニック
/// する（呼び出し元のバグを早期検出する契約前提の検証。REQ-8 境界検査
/// 規約。packing 段の呼び出し元バグ検出であり、本 PR〈#691〉P1 対応の
/// スコープ外）。`ldc < NR`／`c.len() < (MR - 1) * ldc + NR`（本関数が
/// アクセスする最大オフセット `+1`）は [`super::TileBoundsError`] として
/// `Result::Err` を返す（#691 レビュー P1 再指摘: 本関数は
/// `backend_cpu::gemm_blis::microkernel` 経由で外部の `Microkernel`
/// 実装からも到達しうる公開入口のため panic させない）。
///
/// # 公開 API 非破壊（#691 レビュー指摘への対応）
///
/// #557 導入時に既存の `kernel(ap, bp, c_tile, kc_len)`（`ldc = NR` 固定）
/// を本関数へ改名・拡張したが、`backend_cpu::gemm_blis::microkernel` は
/// `pub mod` であり既存呼び出し元を壊すため（AGENTS.md「公開 API の
/// 破壊的変更は P1」）、従来シグネチャは [`kernel`] として残し、本関数へ
/// `ldc = NR` で委譲する薄い後方互換ラッパーとする。
pub fn kernel_with_ldc(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    assert_eq!(ap.len(), MR * kc_len, "packed A panel length mismatch");
    assert_eq!(bp.len(), kc_len * NR, "packed B panel length mismatch");
    super::check_c_tile_bounds(MR, NR, ldc, c.len())?;
    compute(ap, bp, c, ldc, kc_len);
    Ok(())
}

/// [`kernel_with_ldc`]／[`kernel`] 共通の演算本体（境界検査は呼び出し元の
/// 責務。#691 レビュー P1 再指摘 `PRRT_kwDOTuUCJc6ZrQZG` 対応: `Result` を
/// `panic!` へ変換する経路をなくすため、検査ロジックと演算を分離する）。
fn compute(ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
    for p in 0..kc_len {
        let a_p = &ap[p * MR..p * MR + MR];
        let b_p = &bp[p * NR..p * NR + NR];
        for i in 0..MR {
            let a_val = a_p[i];
            for j in 0..NR {
                c[i * ldc + j] = a_val.mul_add(b_p[j], c[i * ldc + j]);
            }
        }
    }
}

/// [`kernel_with_ldc`] の従来シグネチャ後方互換ラッパー（`ldc = NR` 固定・
/// 密パッキング契約）。新規呼び出し元は `ldc` を明示できる
/// [`kernel_with_ldc`] を使うこと。
///
/// ## 戻り値非破壊（#691 レビュー P1 再指摘への対応）
///
/// #557 対応の過程で本関数の戻り値を一時的に
/// `Result<(), super::TileBoundsError>` へ変更していたが、これは
/// `backend_cpu::gemm_blis::microkernel::scalar` が `pub mod` であるため
/// 「従来 `()` を返す本関数を関数ポインタ・末尾式で使う既存の外部
/// 呼び出し元」をコンパイル不能にする破壊的変更だった（codex-review・
/// GraphQL reviewThreads 双方の指摘。AGENTS.md 公開 API 非破壊規約）。
/// 本関数は従来どおり `()` を返す必須シグネチャへ戻す。`c_tile` の長さ
/// 契約違反はここでは検査せず（#691 レビュー P1 再指摘
/// `PRRT_kwDOTuUCJc6ZrQZG` 対応: `check_c_tile_bounds` の `Result` を
/// `panic!("{e}")` へ変換する経路を作らない）、[`compute`] 内の通常の
/// スライス添字アクセスに委ねる。これは #557 以前（`kernel` が唯一の
/// 実装で、密パッキング契約〈`c_tile.len() == MR*NR`〉は呼び出し元の
/// 責務だった頃）と観測可能な挙動が同一である（契約違反時は言語組み込み
/// の範囲外添字 panic になる。本関数が新規に panic 経路を追加している
/// わけではない）。`ldc` を選べる新設 API は [`kernel_with_ldc`]（`Result`
/// を返す）側にのみ存在する非対称な形とする。
pub fn kernel(ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
    assert_eq!(ap.len(), MR * kc_len, "packed A panel length mismatch");
    assert_eq!(bp.len(), kc_len * NR, "packed B panel length mismatch");
    compute(ap, bp, c_tile, NR, kc_len);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 手計算 2x2（MR/NR=4 タイルの左上 2x2 のみ使用・残りはゼロ）で
    /// FMA 累積が正しいことを確認する（packing 契約の単体検証）。
    #[test]
    fn kernel_matches_hand_computed_subset() {
        // A = [[1,2],[3,4]] (2x2 有効部、残り 2 行はゼロ padding)
        // B = [[5,6],[7,8]] (2x2 有効部、残り 2 列はゼロ padding)
        // p-major packing: ap[p*MR+i], bp[p*NR+j]
        let mut ap = vec![0.0f32; MR * 2];
        let mut bp = vec![0.0f32; 2 * NR];
        // p=0: a[:,0] = [1,3,0,0]（p=0 ブロックの先頭は ap[0]）
        ap[0] = 1.0;
        ap[1] = 3.0;
        // p=1: a[:,1] = [2,4,0,0]（p=1 ブロックの先頭は ap[MR]）
        ap[MR] = 2.0;
        ap[MR + 1] = 4.0;
        // p=0: b[0,:] = [5,6,0,0]（p=0 ブロックの先頭は bp[0]）
        bp[0] = 5.0;
        bp[1] = 6.0;
        // p=1: b[1,:] = [7,8,0,0]（p=1 ブロックの先頭は bp[NR]）
        bp[NR] = 7.0;
        bp[NR + 1] = 8.0;

        let mut c_tile = vec![0.0f32; MR * NR];
        kernel(&ap, &bp, &mut c_tile, 2);

        assert_eq!(c_tile[0], 19.0); // 1*5+2*7（行 0 の先頭は c_tile[0]）
        assert_eq!(c_tile[1], 22.0); // 1*6+2*8
        assert_eq!(c_tile[NR], 43.0); // 3*5+4*7（行 1 の先頭は c_tile[NR]）
        assert_eq!(c_tile[NR + 1], 50.0); // 3*6+4*8
    }

    #[test]
    #[should_panic(expected = "packed A panel length mismatch")]
    fn kernel_rejects_ap_length_mismatch() {
        let ap = vec![0.0f32; MR * 2 - 1];
        let bp = vec![0.0f32; 2 * NR];
        let mut c_tile = vec![0.0f32; MR * NR];
        kernel(&ap, &bp, &mut c_tile, 2);
    }

    /// #557: `ldc > NR`（完全タイル C 直接経路の想定）でも `ldc = NR`
    /// と bit 完全一致し、ギャップ列（`j in NR..ldc` に相当する隣接領域）
    /// を破壊しないことを検証する（回帰検査。§5 テスト計画 1・2）。
    #[test]
    fn kernel_with_larger_ldc_matches_tight_packing_and_preserves_gap() {
        let kc_len = 3;
        let ap = vec![
            1.0, 2.0, 3.0, 4.0, // p=0
            5.0, 6.0, 7.0, 8.0, // p=1
            9.0, 10.0, 11.0, 12.0, // p=2
        ];
        let bp = vec![
            0.1, 0.2, 0.3, 0.4, // p=0
            0.5, 0.6, 0.7, 0.8, // p=1
            0.9, 1.0, 1.1, 1.2, // p=2
        ];

        let mut c_tight = vec![0.0f32; MR * NR];
        kernel_with_ldc(&ap, &bp, &mut c_tight, NR, kc_len).unwrap();

        // ldc = NR + 3 のギャップ付きバッファ。ギャップ列は番兵値
        // （追跡しやすい負の値）で埋め、カーネル実行後も不変であることを
        // 確認する（直接ストアが隣接領域を破壊しない回帰検査）。
        let ldc = NR + 3;
        let sentinel = -999.0f32;
        let mut c_gapped = vec![sentinel; (MR - 1) * ldc + ldc];
        // タイル本体（j in 0..NR）は c_tight と同じ初期値（0.0）で揃え、
        // ギャップ列（j in NR..ldc）のみ番兵値のまま残す。
        for i in 0..MR {
            for j in 0..NR {
                c_gapped[i * ldc + j] = 0.0;
            }
        }
        kernel_with_ldc(&ap, &bp, &mut c_gapped, ldc, kc_len).unwrap();

        for i in 0..MR {
            for j in 0..NR {
                assert_eq!(
                    c_gapped[i * ldc + j],
                    c_tight[i * NR + j],
                    "ldc={ldc} 経路と ldc=NR 経路は bit 完全一致するはず（i={i}, j={j}）"
                );
            }
            for j in NR..ldc {
                assert_eq!(
                    c_gapped[i * ldc + j],
                    sentinel,
                    "ギャップ列（i={i}, j={j}）は直接ストアで破壊されてはならない"
                );
            }
        }
    }

    /// `ldc < NR` は panic ではなく `Result::Err` として返る（#691
    /// レビュー P1 再指摘への対応。従来は `should_panic` テストだった）。
    #[test]
    fn kernel_rejects_ldc_smaller_than_nr() {
        let ap = vec![0.0f32; MR * 2];
        let bp = vec![0.0f32; 2 * NR];
        let mut c_tile = vec![0.0f32; MR * NR];
        let err = kernel_with_ldc(&ap, &bp, &mut c_tile, NR - 1, 2).unwrap_err();
        assert_eq!(
            err,
            super::super::TileBoundsError::LdcTooSmall {
                ldc: NR - 1,
                nr: NR
            }
        );
    }

    /// `c` バッファ不足は panic ではなく `Result::Err` として返る（#691
    /// レビュー P1 再指摘への対応。従来は `should_panic` テストだった）。
    #[test]
    fn kernel_rejects_c_buffer_too_small_for_ldc() {
        let ap = vec![0.0f32; MR * 2];
        let bp = vec![0.0f32; 2 * NR];
        // ldc = NR + 1 なら必要長は (MR-1)*(NR+1)+NR だが、ここでは
        // 密パッキング（MR*NR）ぶんしか用意しない。
        let mut c_tile = vec![0.0f32; MR * NR];
        let err = kernel_with_ldc(&ap, &bp, &mut c_tile, NR + 1, 2).unwrap_err();
        assert_eq!(
            err,
            super::super::TileBoundsError::CBufferTooSmall {
                required: (MR - 1) * (NR + 1) + NR,
                actual: MR * NR,
            }
        );
    }
}
