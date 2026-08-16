//! x86_64 AVX2+FMA マイクロカーネル（MR=6×NR=16、`_mm256_fmadd_ps`）。
//!
//! **モジュールは `cfg(target_arch = "x86_64")` のみでコンパイルし、
//! `target_feature = "avx2"` ではゲートしない**（レビュー指摘: モジュール
//! 単位で `target_feature` cfg するとデフォルトビルド〈RUSTFLAGS なし〉で
//! 本体が一切コンパイルされず、`is_x86_feature_detected!` によるテスト
//! 限定の実行時検証が不可能になるため）。実際に AVX2+FMA 命令を発行する
//! [`kernel_unchecked`] は `#[target_feature(enable = ...)]` で個別に
//! ゲートし、それを安全に呼べるかどうかは呼び出し側の条件（コンパイル時
//! cfg または実行時 `is_x86_feature_detected!`）に委ねる。
//!
//! #185（TASK-1.6g）で本番ディスパッチ経路（[`super::Avx2Kernel`]）が
//! 追加され、`kernel_unchecked` は `Avx2Kernel::try_new()` 経由の実行時
//! 検出済み呼び出しでも駆動される（テスト限定の直接呼び出しは維持）。
//!
//! FMA 契約（REQ-2）: `_mm256_fmadd_ps` は IEEE-754 fused multiply-add
//! であり、[`super::scalar::kernel`]・[`super::neon`] と丸めが同一になる
//! （PoC-v2-5 の K=4096 ストレスケースで GPU 側含め実測確認済みの契約）。

use std::arch::x86_64::{
    __m256, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps,
};

/// マイクロカーネルタイルの行数。
pub const MR: usize = 6;
/// マイクロカーネルタイルの列数（`__m256`〈f32x8〉レジスタ 2 本ぶん）。
pub const NR: usize = 16;

// [`super::super::gemm_blis_region`] の C タイルスタックバッファは
// `MAX_TILE`（256 要素）固定長で確保するため、コンパイル時に検査する（#185）。
const _: () = assert!(MR * NR <= 256);

/// AVX2+FMA を用いる実装本体。`#[target_feature(enable = "avx2,fma")]`
/// が付くため呼び出しは常に `unsafe`（コンパイラ既定の安全弾）であり、
/// 呼び出し元が「実行 CPU が AVX2+FMA をサポートする」ことを保証する
/// 責務を負う（本ファイル内の [`kernel`]〈コンパイル時 cfg 経由〉、または
/// `tests/gemm_blis_parity.rs` の `is_x86_feature_detected!` ガード付き
/// 直接呼び出しがその責務を果たす）。
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
/// `ap.len() != MR * kc_len`／`bp.len() != kc_len * NR`／`ldc < NR`／
/// `c.len() < (MR - 1) * ldc + NR` のいずれかであればパニックする（REQ-8
/// 境界検査規約: 最適化対象の関数であっても関数入口の明示検査は省略
/// しない。呼び出し頻度はマイクロカーネル呼び出し 1 回につき 1 回のみ
/// で、内側の SIMD ループには一切挟まない）。
///
/// # Safety
///
/// 呼び出し元は実行 CPU が AVX2・FMA 命令セットをサポートすることを
/// 保証しなければならない（コンパイル時 cfg または `is_x86_feature_detected!`
/// による実行時検出のいずれか）。
#[target_feature(enable = "avx2,fma")]
pub unsafe fn kernel_unchecked(ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
    assert_eq!(ap.len(), MR * kc_len, "packed A panel length mismatch");
    assert_eq!(bp.len(), kc_len * NR, "packed B panel length mismatch");
    assert!(ldc >= NR, "ldc must be at least NR");
    assert!(
        c.len()
            >= (MR - 1)
                .checked_mul(ldc)
                .and_then(|v| v.checked_add(NR))
                .expect("ldc*MR overflow"),
        "C tile buffer too small for MR*ldc access pattern"
    );

    // SAFETY: 直前の assert により ap は MR*kc_len 要素、bp は kc_len*NR
    // 要素、c は最大アクセスオフセット `(MR-1)*ldc+NR-1` を含む長さである
    // ことが保証されている。以下のロード／ストアはいずれもこの範囲内の
    // オフセットに限定される（p*NR+8..p*NR+16 の最大値は kc_len-1 でも
    // bp.len() を超えない。c も i*ldc+8..i*ldc+16 が最大 i=MR-1 でも
    // c.len() を超えない）。AVX2+FMA 命令の発行自体は、この関数の
    // `#[target_feature]` 契約により呼び出し元が実行 CPU の対応を
    // 保証している前提で健全（関数ドキュメントの `# Safety` 節参照）。
    unsafe {
        let mut acc: [[__m256; 2]; MR] = std::array::from_fn(|i| {
            [
                _mm256_loadu_ps(c[i * ldc..].as_ptr()),
                _mm256_loadu_ps(c[i * ldc + 8..].as_ptr()),
            ]
        });

        for p in 0..kc_len {
            let b0 = _mm256_loadu_ps(bp[p * NR..].as_ptr());
            let b1 = _mm256_loadu_ps(bp[p * NR + 8..].as_ptr());
            for i in 0..MR {
                let a_val = ap[p * MR + i];
                let a_vec = _mm256_set1_ps(a_val);
                acc[i][0] = _mm256_fmadd_ps(a_vec, b0, acc[i][0]);
                acc[i][1] = _mm256_fmadd_ps(a_vec, b1, acc[i][1]);
            }
        }

        for (i, acc_i) in acc.iter().enumerate() {
            _mm256_storeu_ps(c[i * ldc..].as_mut_ptr(), acc_i[0]);
            _mm256_storeu_ps(c[i * ldc + 8..].as_mut_ptr(), acc_i[1]);
        }
    }
}

/// [`kernel_unchecked`] の安全なラッパー。`cfg(target_feature = "avx2",
/// target_feature = "fma")` が成立する場合のみコンパイルされ、これは
/// ビルド時 `RUSTFLAGS="-C target-feature=+avx2,+fma"` 等でコンパイル
/// ターゲット CPU が AVX2+FMA を持つと明示された場合にのみ真になる。
/// `gemm_blis` の既定経路（[`super::super::microkernel`] の cfg 選択）が
/// この条件下でのみ本関数を選ぶため、`unsafe` 呼び出しの健全性は
/// コンパイル時に確定する。
#[cfg(all(target_feature = "avx2", target_feature = "fma"))]
pub fn kernel(ap: &[f32], bp: &[f32], c: &mut [f32], ldc: usize, kc_len: usize) {
    // SAFETY: この関数がコンパイルされている時点で `cfg(target_feature =
    // "avx2", target_feature = "fma")` が成立しており、ビルド対象 CPU は
    // AVX2+FMA をサポートすると明示されている（`kernel_unchecked` の
    // `# Safety` 契約を満たす）。
    unsafe { kernel_unchecked(ap, bp, c, ldc, kc_len) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AVX2+FMA 実行時検出ガード付きで [`kernel_unchecked`] を直接呼び、
    /// 既定ビルド（RUSTFLAGS なしの CI）でもカーネル本体をテストできる
    /// ようにする（TASK-1.6f 計画の「テスト限定の AVX2 直接テスト」。
    /// 本番ディスパッチ経路への組み込みは #185 のスコープ）。
    #[test]
    fn kernel_unchecked_matches_hand_computed_subset_when_avx2_available() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            // AVX2/FMA 非対応環境（本 CI の想定外）ではスキップする。
            // このガードなしに `kernel_unchecked` を呼ぶと SIGILL の
            // リスクがあるため必須（Safety 契約）。
            eprintln!("AVX2/FMA 非対応環境のためスキップ");
            return;
        }

        // A = [[1,2],[3,4]], B = [[5,6],[7,8]] を MR=6×NR=16 タイルの
        // 左上に配置し、残りをゼロ padding する（scalar 版と同じ手計算
        // ケースを流用。padding は m/n 方向のみで REQ-2 の bit 一致契約を
        // 崩さない）。
        let kc_len = 2;
        let mut ap = vec![0.0f32; MR * kc_len];
        let mut bp = vec![0.0f32; kc_len * NR];
        // p=0 ブロックの先頭は ap[0]、p=1 ブロックの先頭は ap[MR]。
        ap[0] = 1.0;
        ap[1] = 3.0;
        ap[MR] = 2.0;
        ap[MR + 1] = 4.0;
        // p=0 ブロックの先頭は bp[0]、p=1 ブロックの先頭は bp[NR]。
        bp[0] = 5.0;
        bp[1] = 6.0;
        bp[NR] = 7.0;
        bp[NR + 1] = 8.0;

        let mut c_tile = vec![0.0f32; MR * NR];
        // SAFETY: 直前の is_x86_feature_detected! ガードにより実行 CPU が
        // AVX2・FMA をサポートすることを確認済み。
        unsafe {
            kernel_unchecked(&ap, &bp, &mut c_tile, NR, kc_len);
        }

        // 行 0 の先頭は c_tile[0]、行 1 の先頭は c_tile[NR]。
        assert_eq!(c_tile[0], 19.0);
        assert_eq!(c_tile[1], 22.0);
        assert_eq!(c_tile[NR], 43.0);
        assert_eq!(c_tile[NR + 1], 50.0);
    }

    /// xorshift32 による疑似乱数ベクトル生成（テスト専用・本体非依存）。
    ///
    /// `bench_harness::rng::Xorshift64Star` を使わない理由: `bench_harness`
    /// は `serde_json` を推移依存に持ち、lib 単体テストバイナリ（本ファイル
    /// は `crates/backend-cpu/src/` 配下）へ持ち込むと、同一バイナリに
    /// リンクされる `reduction.rs` 側の無関係な `assert_eq!(&[usize], &[])`
    /// が `usize: PartialEq<_>` の複数実装（`core` と `serde_json::Value`
    /// 向け）で型推論あいまいになり `E0282/E0283` を起こす（実装時に実測
    /// 確認済み。`gemm_blis::mod` の `#[cfg(test)] mod tests` の同種の
    /// コメント参照）。統合テスト（`tests/gemm_blis_parity.rs`）は個別
    /// バイナリのためこの問題が生じず、`bench_harness::rng` をそのまま使う。
    fn xorshift32_vec(seed: u32, len: usize) -> Vec<f32> {
        let mut state = seed | 1; // 0 を避ける（xorshift は 0 で退化する）
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                // [0, 1) の範囲に正規化する。
                (state as f64 / u32::MAX as f64) as f32
            })
            .collect()
    }

    /// AVX2 カーネルとスカラー参照実装（`mul_add` 連鎖・AVX2 と同じ
    /// MR×NR タイル形状）の bit 一致（乱数ストレス）。
    /// FMA 契約（REQ-2）が ISA 間で統一されていることの直接的な根拠。
    ///
    /// [`super::scalar::kernel`] を直接比較対象にしない理由: scalar は
    /// MR=4×NR=4 固定であり AVX2（MR=6×NR=16）とタイル形状が異なるため、
    /// 同一入力を渡す比較が成立しない。代わりに AVX2 と同じ MR×NR で
    /// `f32::mul_add` を p 昇順に適用する参照実装をこのテスト内に持つ
    /// （[`super::scalar::kernel`] 本体と同一ロジックだが MR/NR だけが
    /// 異なる、テスト専用の写し）。
    #[test]
    fn kernel_unchecked_matches_mul_add_reference_bit_exact() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("AVX2/FMA 非対応環境のためスキップ");
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

        let mut c_avx2 = c_init;
        // SAFETY: 直前の is_x86_feature_detected! ガードにより健全。
        unsafe {
            kernel_unchecked(&ap, &bp, &mut c_avx2, NR, kc_len);
        }

        assert_eq!(
            c_ref, c_avx2,
            "AVX2 カーネルは mul_add 参照実装と bit 完全一致するはず"
        );
    }

    /// #557: `ldc > NR`（完全タイル C 直接経路の想定）でも `ldc = NR`
    /// と bit 完全一致し、ギャップ列を破壊しないことを検証する（scalar.rs
    /// の同種テストと同一パターン）。
    #[test]
    fn kernel_unchecked_with_larger_ldc_matches_tight_packing_and_preserves_gap() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            eprintln!("AVX2/FMA 非対応環境のためスキップ");
            return;
        }

        let kc_len = 5;
        let ap = xorshift32_vec(0xC0FF_EE01, MR * kc_len);
        let bp = xorshift32_vec(0xC0FF_EE02, kc_len * NR);
        let c_init = xorshift32_vec(0xC0FF_EE03, MR * NR);

        let mut c_tight = c_init.clone();
        // SAFETY: 冒頭の is_x86_feature_detected! ガードにより健全。
        unsafe {
            kernel_unchecked(&ap, &bp, &mut c_tight, NR, kc_len);
        }

        let ldc = NR + 5;
        let sentinel = -777.0f32;
        let mut c_gapped = vec![sentinel; (MR - 1) * ldc + ldc];
        for i in 0..MR {
            c_gapped[i * ldc..i * ldc + NR].copy_from_slice(&c_init[i * NR..i * NR + NR]);
        }
        // SAFETY: 冒頭の is_x86_feature_detected! ガードにより健全。
        unsafe {
            kernel_unchecked(&ap, &bp, &mut c_gapped, ldc, kc_len);
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
