//! aarch64 NEON マイクロカーネル（MR=8×NR=8、`vfmaq_laneq_f32`）。
//!
//! `cfg(target_arch = "aarch64")` 限定でコンパイルされる（NEON は aarch64
//! のベースライン ISA であり、Metal 実機（Apple M4 Max）・DGX Spark GB10
//! の Grace CPU 側いずれでも常時利用可能なため、[`super::avx2`] と異なり
//! `target_feature` によるコンパイル時分岐は不要）。x86_64 開発環境では
//! 実行検証できないため、`cargo check --target aarch64-unknown-linux-gnu`
//! によるコンパイル検証に留める（実機実行確認は `#[ignore]` テストへ委ねる。
//! `.claude/rules/coding-rust.md` 実機分離方針）。
//!
//! A オペランドのロードはレーン選択 FMA（`vfmaq_laneq_f32`。単一命令
//! `FMLA v.4s, v.4s, v.s[lane]`）を用いる（イシュー #552）。旧実装は p
//! ごとに A の各行値をスカラー読み出し → `vdupq_n_f32` で明示 broadcast
//! → `vfmaq_f32` する 2 命令方式だったが、[`super::super::pack::pack_a`]
//! が A パネルを p-major・mr 方向連続（`dst[p * mr + i]`）で packing する
//! ため、p ごとに `vld1q_f32` 2 回（8 行ぶん一括）で A 値をロードし、
//! レーンを直接 FMA へ渡せる。broadcast 命令（DUP）とスカラーロードを
//! 排し k-step あたりの命令数を削減する（BLIS armv8a sgemm カーネル・
//! matrixmultiply の sgemm_kernel と同技法）。
//!
//! FMA 契約（REQ-2）: `vfmaq_laneq_f32(acc, b, a, LANE)` は
//! `acc + b * a[LANE]` の単一 fused multiply-add であり、`DUP` は
//! ブロードキャストのみで演算順序・丸めに影響しないため、旧
//! `vdupq_n_f32` + `vfmaq_f32` 方式と数学的に同一（`p` 昇順の FMA 連鎖・
//! 乗算順序は不変）。[`super::scalar::kernel`]・[`super::avx2`] と丸めが
//! 同一になる契約（PoC-v2-5 の K=4096 ストレスケースで GPU 側含め実測
//! 確認済み）も維持され、bit 完全一致は `gemm_blis_parity` テストで検証
//! する（aarch64 実行環境では NEON 経路が既定選択されるため実機実行時に
//! この経路の bit 一致が検査される）。

use std::arch::aarch64::{float32x4_t, vfmaq_laneq_f32, vld1q_f32, vst1q_f32};

/// マイクロカーネルタイルの行数。
pub const MR: usize = 8;
/// マイクロカーネルタイルの列数（f32x4 レジスタ 2 本ぶん）。
pub const NR: usize = 8;

// [`super::super::gemm_blis_region`] の C タイルスタックバッファは
// `MAX_TILE`（256 要素）固定長で確保するため、コンパイル時に検査する（#185）。
const _: () = assert!(MR * NR <= 256);

// k ループのレーン展開（`fma_row!` 8 回展開）は MR=8（a0/a1 の 2 レジスタ・
// レーン 0..3 固定）・NR=8（b0/b1 の 2 レジスタ固定）を前提にハードコード
// されている。旧実装（`for i in 0..MR`）と異なりこの前提は暗黙のため、
// MR/NR を変更した場合にコンパイルエラーで検知できるようにする（#552。
// 変更を怠ると `assert_eq!(ap.len(), MR * kc_len)` は通過したまま行の
// 一部が計算から欠落し、実行時パニックなしに誤った結果を返しうる）。
const _: () = assert!(MR == 8 && NR == 8);

/// [`super::scalar::kernel`] と同一の累積契約（p 昇順・mul_add 連鎖）を
/// NEON `vfmaq_laneq_f32`（レーン選択 FMA）で実装する。C タイルをレーンごとに独立したレジスタへ
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
    // 最大値は (kc_len-1)*MR+MR-1 = ap.len()-1、p*MR+4..p*MR+8 の最大も
    // 同様に ap.len() を超えない。c_tile も i*NR+4..i*NR+8 が最大
    // i=MR-1 でも c_tile.len() を超えない）に限定される。NEON は
    // aarch64 のベースライン機能であり実行時検出は不要（本モジュールが
    // `cfg(target_arch = "aarch64")` 限定コンパイルであることが前提）。
    unsafe {
        let mut acc: [[float32x4_t; 2]; MR] = std::array::from_fn(|i| {
            [
                vld1q_f32(c_tile[i * NR..].as_ptr()),
                vld1q_f32(c_tile[i * NR + 4..].as_ptr()),
            ]
        });

        // レーン対応表: pack_a（`dst[p * mr + i]`。p-major・mr 方向連続）
        // により、a0 は行 0..3（レーン k = 行 k）・a1 は行 4..7（レーン
        // k = 行 4+k）を保持する。`vfmaq_laneq_f32::<LANE>(acc, b, a)` は
        // `acc + b * a[LANE]` のため、行 i (<4) は `(a0, i)`、行 i (>=4)
        // は `(a1, i-4)` を参照する。行 i 昇順・`[0]` → `[1]` の順・p
        // 昇順の FMA 連鎖はいずれも旧 `vdupq_n_f32` 版と同一に保つ
        // （bit 完全一致契約の前提。冒頭モジュールコメント参照）。
        macro_rules! fma_row {
            ($acc_i:expr, $a:expr, $lane:literal, $b0:expr, $b1:expr) => {{
                $acc_i[0] = vfmaq_laneq_f32::<$lane>($acc_i[0], $b0, $a);
                $acc_i[1] = vfmaq_laneq_f32::<$lane>($acc_i[1], $b1, $a);
            }};
        }

        for p in 0..kc_len {
            let b0 = vld1q_f32(bp[p * NR..].as_ptr());
            let b1 = vld1q_f32(bp[p * NR + 4..].as_ptr());
            // A の 8 行ぶんを 2 回の vld1q_f32 で一括ロード（行 0..4 は
            // a0、行 4..8 は a1）。旧実装のスカラーロード 8 回＋
            // vdupq_n_f32（DUP）8 回を排除する（イシュー #552）。
            let a0 = vld1q_f32(ap[p * MR..].as_ptr());
            let a1 = vld1q_f32(ap[p * MR + 4..].as_ptr());

            fma_row!(acc[0], a0, 0, b0, b1);
            fma_row!(acc[1], a0, 1, b0, b1);
            fma_row!(acc[2], a0, 2, b0, b1);
            fma_row!(acc[3], a0, 3, b0, b1);
            fma_row!(acc[4], a1, 0, b0, b1);
            fma_row!(acc[5], a1, 1, b0, b1);
            fma_row!(acc[6], a1, 2, b0, b1);
            fma_row!(acc[7], a1, 3, b0, b1);
        }

        for (i, acc_i) in acc.iter().enumerate() {
            vst1q_f32(c_tile[i * NR..].as_mut_ptr(), acc_i[0]);
            vst1q_f32(c_tile[i * NR + 4..].as_mut_ptr(), acc_i[1]);
        }
    }
}
