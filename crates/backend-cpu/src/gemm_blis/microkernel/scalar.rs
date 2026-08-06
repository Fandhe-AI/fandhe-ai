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
/// p-major）から MR×NR の C タイル `c_tile`（row-major、ld=NR）へ
/// `kc_len` ぶんの寄与を加算する。
///
/// 呼び出し元（[`super::super::mod`] の 5-loop ドライバ）は本関数呼び出し
/// 前に `c_tile` へ実際の C の現在値をロードし、呼び出し後に書き戻す
/// （複数の `pc` ブロックにまたがる累積を成立させるため）。
///
/// # 累積順序（bit 完全一致契約）
///
/// 各 `(i, j)` に対し `c_tile[i][j] = a[p][i].mul_add(b[p][j], c_tile[i][j])`
/// を `p` 昇順に適用する。この順序は [`crate::gemm::gemm_naive`] が
/// 内側ループで行う蓄積順序と同一であり、他の `(i, j)` との縮約
/// （split-k 等）を一切行わないため、`gemm_naive` と bit 完全一致が
/// 成立する（`tests/gemm_blis_parity.rs` の `assert_eq!` 契約）。
///
/// # Panics
///
/// `ap.len() != MR * kc_len`／`bp.len() != kc_len * NR`／
/// `c_tile.len() != MR * NR` のいずれかであればパニックする
/// （呼び出し元のバグを早期検出する契約前提の検証。REQ-8 境界検査規約）。
pub fn kernel(ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
    assert_eq!(ap.len(), MR * kc_len, "packed A panel length mismatch");
    assert_eq!(bp.len(), kc_len * NR, "packed B panel length mismatch");
    assert_eq!(c_tile.len(), MR * NR, "C tile length mismatch");

    for p in 0..kc_len {
        let a_p = &ap[p * MR..p * MR + MR];
        let b_p = &bp[p * NR..p * NR + NR];
        for i in 0..MR {
            let a_val = a_p[i];
            for j in 0..NR {
                c_tile[i * NR + j] = a_val.mul_add(b_p[j], c_tile[i * NR + j]);
            }
        }
    }
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
}
