//! x86_64 AVX-512F マイクロカーネル（MR=8×NR=32、`_mm512_fmadd_ps`）。
//!
//! #185（TASK-1.6g）で新規追加。[`super::avx2`] と同じ「モジュールは
//! `cfg(target_arch = "x86_64")` のみでコンパイルし `target_feature` では
//! ゲートしない」方針を踏襲する（既定ビルドでもテスト限定の実行時検出
//! ガード付き直接検証を可能にするため。[`avx2`] モジュールドキュメント
//! 参照）。
//!
//! レジスタ構成: acc は `__m512`（f32x16）2 本 × MR=8 行 = 16 本、B パネル
//! ロード 2 本、A ブロードキャスト 1 本の計 19 本で、AVX-512 の zmm
//! レジスタ数（32 本）に収まる保守的な構成を採る（レジスタチューニングは
//! #24 のスコープ）。
//!
//! FMA 契約（REQ-2）: `_mm512_fmadd_ps` は IEEE-754 fused multiply-add
//! であり、[`super::scalar`]・[`super::neon`]・[`super::avx2`] と丸めが
//! 同一になる（PoC-v2-5 の K=4096 ストレスケースで GPU 側含め実測確認済み
//! の契約。累積順序は p 昇順・レーン間縮約なしで各カーネル共通）。

use std::arch::x86_64::{
    __m512, _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_set1_ps, _mm512_storeu_ps,
};

/// マイクロカーネルタイルの行数。
pub const MR: usize = 8;
/// マイクロカーネルタイルの列数（`__m512`〈f32x16〉レジスタ 2 本ぶん）。
pub const NR: usize = 32;

// [`super::super::gemm_blis_region`] の C タイルスタックバッファは
// `MAX_TILE`（256 要素・本カーネルの 8×32 が最大値）固定長で確保するため、
// コンパイル時に検査する（#185）。
const _: () = assert!(MR * NR <= 256);

/// AVX-512F を用いる実装本体。`#[target_feature(enable = "avx512f")]`
/// が付くため呼び出しは常に `unsafe`（コンパイラ既定の安全弾）であり、
/// 呼び出し元が「実行 CPU が AVX-512F をサポートする」ことを保証する
/// 責務を負う（本番経路では [`super::Avx512Kernel::try_new`] が検出済み
/// の場合のみトークンを生成することでこの責務を果たす。テストでは
/// `is_x86_feature_detected!` ガード付き直接呼び出しで同じ責務を果たす）。
///
/// # `ldc` 契約（#557）
///
/// `c` は要素 `c[i*ldc+j]`（`i in 0..MR`・`j in 0..NR`）のみを読み書きする。
/// 完全タイル呼び出しでは `ldc = n`（C の実列数）で C バッファへ直接、
/// 端タイル呼び出しでは `ldc = NR` で密パッキングされたスタックバッファへ
/// アクセスする（[`super::Microkernel::run`] 契約と同一）。
///
/// # Panics
///
/// `ap.len() != MR * kc_len`／`bp.len() != kc_len * NR` であればパニック
/// する（REQ-8 境界検査規約: packing 段の呼び出し元バグ検出であり、
/// 本 PR〈#691〉P1 対応のスコープ外。呼び出し頻度はマイクロカーネル
/// 呼び出し 1 回につき 1 回のみで、内側の SIMD ループには一切挟まない）。
/// `ldc < NR`／`c.len() < (MR - 1) * ldc + NR` は [`super::TileBoundsError`]
/// として `Result::Err` を返す（#691 レビュー P1 再指摘: 本関数は外部の
/// `Microkernel` 実装からも到達しうる公開入口のため panic させない）。
///
/// # Safety
///
/// 呼び出し元は実行 CPU が AVX-512F 命令セットをサポートすることを
/// 保証しなければならない（[`super::Avx512Kernel::try_new`] による
/// 実行時検出、またはテストの `is_x86_feature_detected!` ガードのいずれか）。
///
/// # 公開 API 非破壊（#691 レビュー指摘への対応）
///
/// [`super::scalar::kernel_with_ldc`] のドキュメント参照。本モジュールも
/// 同じ理由で従来シグネチャを [`kernel_unchecked`] として残す。
#[target_feature(enable = "avx512f")]
pub unsafe fn kernel_unchecked_with_ldc(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    ldc: usize,
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    assert_eq!(ap.len(), MR * kc_len, "packed A panel length mismatch");
    assert_eq!(bp.len(), kc_len * NR, "packed B panel length mismatch");
    super::check_c_tile_bounds(MR, NR, ldc, c.len())?;

    // SAFETY: 直前の assert により ap は MR*kc_len 要素、bp は kc_len*NR
    // 要素、c は最大アクセスオフセット `(MR-1)*ldc+NR-1` を含む長さである
    // ことが保証されている。以下のロード／ストアはいずれもこの範囲内の
    // オフセットに限定される（p*NR+16..p*NR+32 の最大値は kc_len-1 でも
    // bp.len() を超えない。c も i*ldc+16..i*ldc+32 が最大 i=MR-1 でも
    // c.len() を超えない）。AVX-512F 命令の発行自体は、この関数の
    // `#[target_feature]` 契約により呼び出し元が実行 CPU の対応を
    // 保証している前提で健全（関数ドキュメントの `# Safety` 節参照）。
    unsafe {
        let mut acc: [[__m512; 2]; MR] = std::array::from_fn(|i| {
            [
                _mm512_loadu_ps(c[i * ldc..].as_ptr()),
                _mm512_loadu_ps(c[i * ldc + 16..].as_ptr()),
            ]
        });

        for p in 0..kc_len {
            let b0 = _mm512_loadu_ps(bp[p * NR..].as_ptr());
            let b1 = _mm512_loadu_ps(bp[p * NR + 16..].as_ptr());
            for i in 0..MR {
                let a_val = ap[p * MR + i];
                let a_vec = _mm512_set1_ps(a_val);
                acc[i][0] = _mm512_fmadd_ps(a_vec, b0, acc[i][0]);
                acc[i][1] = _mm512_fmadd_ps(a_vec, b1, acc[i][1]);
            }
        }

        for (i, acc_i) in acc.iter().enumerate() {
            _mm512_storeu_ps(c[i * ldc..].as_mut_ptr(), acc_i[0]);
            _mm512_storeu_ps(c[i * ldc + 16..].as_mut_ptr(), acc_i[1]);
        }
    }
    Ok(())
}

/// [`kernel_unchecked_with_ldc`] の従来シグネチャ後方互換ラッパー（unsafe。
/// `ldc = NR` 固定・密パッキング契約）。[`kernel_unchecked_with_ldc`] の
/// `# Safety` 契約をそのまま引き継ぐ。
///
/// # Safety
///
/// [`kernel_unchecked_with_ldc`] と同一（呼び出し元は実行 CPU が
/// AVX-512F をサポートすることを保証しなければならない）。
#[target_feature(enable = "avx512f")]
pub unsafe fn kernel_unchecked(
    ap: &[f32],
    bp: &[f32],
    c: &mut [f32],
    kc_len: usize,
) -> Result<(), super::TileBoundsError> {
    // SAFETY: 呼び出し元契約を本関数の `# Safety` 節としてそのまま
    // 引き継いでいる。
    unsafe { kernel_unchecked_with_ldc(ap, bp, c, NR, kc_len) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AVX-512F 実行時検出ガード付きで [`kernel_unchecked`] を直接呼び、
    /// 既定ビルド（RUSTFLAGS なしの CI）でもカーネル本体をテストできる
    /// ようにする（[`super::avx2`] の同種テストと同一パターン）。CI
    /// （QEMU 仮想 CPU）が AVX-512 非対応の場合は skip する。
    #[test]
    fn kernel_unchecked_matches_hand_computed_subset_when_avx512_available() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("AVX-512F 非対応環境のためスキップ");
            return;
        }

        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] を MR=8×NR=32 タイルの
        // 左上に配置し、残りをゼロ padding する（avx2 テストと同じ
        // 手計算ケースを流用。padding は m/n 方向のみで REQ-2 の bit
        // 一致契約を崩さない）。
        let kc_len = 2;
        let mut ap = vec![0.0f32; MR * kc_len];
        let mut bp = vec![0.0f32; kc_len * NR];
        ap[0] = 1.0;
        ap[1] = 3.0;
        ap[MR] = 2.0;
        ap[MR + 1] = 4.0;
        bp[0] = 5.0;
        bp[1] = 6.0;
        bp[NR] = 7.0;
        bp[NR + 1] = 8.0;

        let mut c_tile = vec![0.0f32; MR * NR];
        // SAFETY: 直前の is_x86_feature_detected! ガードにより実行 CPU が
        // AVX-512F をサポートすることを確認済み。
        unsafe {
            kernel_unchecked(&ap, &bp, &mut c_tile, kc_len).unwrap();
        }

        assert_eq!(c_tile[0], 19.0);
        assert_eq!(c_tile[1], 22.0);
        assert_eq!(c_tile[NR], 43.0);
        assert_eq!(c_tile[NR + 1], 50.0);
    }

    /// xorshift32 による疑似乱数ベクトル生成（テスト専用・本体非依存。
    /// `bench_harness` を持ち込まない理由は [`super::avx2`] の同名関数の
    /// ドキュメントコメント参照。lib 単体テストへ `serde_json` 推移依存を
    /// 持ち込むと同一バイナリにリンクされる `reduction.rs` 側の
    /// `assert_eq!` が型推論あいまいで E0282/E0283 を起こすため）。
    fn xorshift32_vec(seed: u32, len: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f64 / u32::MAX as f64) as f32
            })
            .collect()
    }

    /// AVX-512F カーネルとスカラー参照実装（`mul_add` 連鎖・AVX-512F と
    /// 同じ MR×NR タイル形状）の bit 一致（乱数ストレス）。FMA 契約
    /// （REQ-2）が ISA 間で統一されていることの直接的な根拠。
    #[test]
    fn kernel_unchecked_matches_mul_add_reference_bit_exact() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("AVX-512F 非対応環境のためスキップ");
            return;
        }

        let kc_len = 37; // KC 境界を跨がない適当な値（マイクロカーネル単体テストのため無関係）
        let ap = xorshift32_vec(0xA5F3_1234, MR * kc_len);
        let bp = xorshift32_vec(0xBEEF_0001, kc_len * NR);
        let c_init = xorshift32_vec(0x1357_9BDF, MR * NR);

        let mut c_ref = c_init.clone();
        for p in 0..kc_len {
            for i in 0..MR {
                let a_val = ap[p * MR + i];
                for j in 0..NR {
                    c_ref[i * NR + j] = a_val.mul_add(bp[p * NR + j], c_ref[i * NR + j]);
                }
            }
        }

        let mut c_avx512 = c_init;
        // SAFETY: 直前の is_x86_feature_detected! ガードにより健全。
        unsafe {
            kernel_unchecked(&ap, &bp, &mut c_avx512, kc_len).unwrap();
        }

        assert_eq!(
            c_ref, c_avx512,
            "AVX-512F カーネルは mul_add 参照実装と bit 完全一致するはず"
        );
    }

    /// #557: `ldc > NR`（完全タイル C 直接経路の想定）でも `ldc = NR`
    /// と bit 完全一致し、ギャップ列を破壊しないことを検証する（avx2.rs
    /// の同種テストと同一パターン）。
    #[test]
    fn kernel_unchecked_with_larger_ldc_matches_tight_packing_and_preserves_gap() {
        if !is_x86_feature_detected!("avx512f") {
            eprintln!("AVX-512F 非対応環境のためスキップ");
            return;
        }

        let kc_len = 5;
        let ap = xorshift32_vec(0xD0FF_EE01, MR * kc_len);
        let bp = xorshift32_vec(0xD0FF_EE02, kc_len * NR);
        let c_init = xorshift32_vec(0xD0FF_EE03, MR * NR);

        let mut c_tight = c_init.clone();
        // SAFETY: 冒頭の is_x86_feature_detected! ガードにより健全。
        unsafe {
            kernel_unchecked_with_ldc(&ap, &bp, &mut c_tight, NR, kc_len).unwrap();
        }

        let ldc = NR + 5;
        let sentinel = -777.0f32;
        let mut c_gapped = vec![sentinel; (MR - 1) * ldc + ldc];
        for i in 0..MR {
            c_gapped[i * ldc..i * ldc + NR].copy_from_slice(&c_init[i * NR..i * NR + NR]);
        }
        // SAFETY: 冒頭の is_x86_feature_detected! ガードにより健全。
        unsafe {
            kernel_unchecked_with_ldc(&ap, &bp, &mut c_gapped, ldc, kc_len).unwrap();
        }

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
}
