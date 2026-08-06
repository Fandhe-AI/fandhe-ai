//! aarch64 NEON マイクロカーネル（MR=8×NR=8、`vfmaq_f32`）。
//!
//! `cfg(target_arch = "aarch64")` 限定でコンパイルされる（NEON は aarch64
//! のベースライン ISA であり、Metal 実機（Apple M4 Max）・DGX Spark GB10
//! の Grace CPU 側いずれでも常時利用可能なため、[`super::avx2`] と異なり
//! `target_feature` によるコンパイル時分岐は不要）。x86_64 開発環境では
//! 実行検証できないため、`cargo check --target aarch64-unknown-linux-gnu`
//! によるコンパイル検証に留める（実機実行確認は `#[ignore]` テストへ委ねる。
//! `.claude/rules/coding-rust.md` 実機分離方針）。
//!
//! FMA 契約（REQ-2）: `vfmaq_f32` は IEEE-754 fused multiply-add であり、
//! [`super::scalar::kernel`]・[`super::avx2`] と丸めが同一になる
//! （PoC-v2-5 の K=4096 ストレスケースで GPU 側含め実測確認済みの契約）。

use std::arch::aarch64::{float32x4_t, vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

/// マイクロカーネルタイルの行数。
pub const MR: usize = 8;
/// マイクロカーネルタイルの列数（f32x4 レジスタ 2 本ぶん）。
pub const NR: usize = 8;

/// [`super::scalar::kernel`] と同一の累積契約（p 昇順・mul_add 連鎖）を
/// NEON `vfmaq_f32` で実装する。C タイルをレーンごとに独立したレジスタへ
/// ロードし、レーン間縮約を一切行わないため、`p` ごとの `a[p][i]・b[p][j]`
/// への乗算順序はスカラー版と bit 完全一致する。
///
/// # Panics
///
/// `ap.len() != MR * kc_len`／`bp.len() != kc_len * NR`／
/// `c_tile.len() != MR * NR` のいずれかであればパニックする（REQ-8
/// 境界検査規約: 呼び出し元契約を関数入口で明示検査し、以降の
/// `unsafe` ロード／ストアはこの検査済み長さの範囲内でのみ行う）。
pub fn kernel(ap: &[f32], bp: &[f32], c_tile: &mut [f32], kc_len: usize) {
    assert_eq!(ap.len(), MR * kc_len, "packed A panel length mismatch");
    assert_eq!(bp.len(), kc_len * NR, "packed B panel length mismatch");
    assert_eq!(c_tile.len(), MR * NR, "C tile length mismatch");

    // SAFETY: 直前の assert により ap は MR*kc_len 要素、bp は kc_len*NR
    // 要素、c_tile は MR*NR(=64) 要素ちょうどであることが保証されている。
    // 以下のロード／ストアはいずれもこの範囲内のオフセット（p*MR+i の
    // 最大値は (kc_len-1)*MR+MR-1 = ap.len()-1、c_tile も i*NR+4..i*NR+8 が
    // 最大 i=MR-1 でも c_tile.len() を超えない）に限定される。NEON は
    // aarch64 のベースライン機能であり実行時検出は不要（本モジュールが
    // `cfg(target_arch = "aarch64")` 限定コンパイルであることが前提）。
    unsafe {
        let mut acc: [[float32x4_t; 2]; MR] = std::array::from_fn(|i| {
            [
                vld1q_f32(c_tile[i * NR..].as_ptr()),
                vld1q_f32(c_tile[i * NR + 4..].as_ptr()),
            ]
        });

        for p in 0..kc_len {
            let b0 = vld1q_f32(bp[p * NR..].as_ptr());
            let b1 = vld1q_f32(bp[p * NR + 4..].as_ptr());
            for i in 0..MR {
                let a_val = ap[p * MR + i];
                let a_vec = vdupq_n_f32(a_val);
                acc[i][0] = vfmaq_f32(acc[i][0], a_vec, b0);
                acc[i][1] = vfmaq_f32(acc[i][1], a_vec, b1);
            }
        }

        for (i, acc_i) in acc.iter().enumerate() {
            vst1q_f32(c_tile[i * NR..].as_mut_ptr(), acc_i[0]);
            vst1q_f32(c_tile[i * NR + 4..].as_mut_ptr(), acc_i[1]);
        }
    }
}
