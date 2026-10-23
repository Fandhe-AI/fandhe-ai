//! aarch64 SME m=64 専用カーネル（ZA0〜ZA3 の 4 タイルで 64×16 の C ブロックを持ち、
//! K 全長を 1 つの `smstart`〜`smstop` 区間で流す。イシュー #3235）。
//!
//! ## 役割と責務境界
//!
//! `super::sme`（MR=16×NR=16・ZA0 単一タイル・micro-call ごとに `smstart`/`smstop`）は
//! m=64 級の形状では固定費と 1 タイルの発行上限で頭打ちになる（#3175 実測）。本モジュールは
//! `docs/perf/cpu-gemm-sme-thin-m64-decision.md`（#3234 判定 (A)）に従い、64×16 の C を
//! ZA0〜ZA3 に常駐させて K 全長を 1 区間で処理する**別入口**を提供する。`MAX_TILE`（256 要素）を
//! 通らないため `super::Microkernel` は実装しない。呼び出し元は `gemm_blis/mod.rs` の
//! `sme_thin_m64_path`（既定 OFF の `SME_THIN_M64_ENABLED` と適格判定の両方が真のときだけ到達）で、
//! 実行スレッド上で `super::SmeKernel::run_thin_m64_block` を経由して呼ばれる。
//!
//! ## bit 完全一致契約（REQ-2・FMA 契約統一）
//!
//! `sme.rs` と同じ演算列を継承する: C の現在値を ZA へプリロードし、p 昇順に要素ごと 1 回の
//! `fmopa`（= 単一丸め FMA）を適用して書き戻す。K 全長を 1 回で流しても要素ごとの演算列は
//! 変わらず、`scalar_fallback`（`f32::mul_add`・p 昇順）と有限値入力で bit 一致する。
//! NaN は payload 選択規則が実装間で保証されないため、クラス一致のみを主張する。
//!
//! ## 安全性
//!
//! 新規 `unsafe asm!` は 1 ブロック（`compute_thin_m64`）。入口 `kernel_unchecked` は asm より前に
//! 全長とストライドを `checked_*` で検査し、違反は型付きエラーで返す（REQ-8・panic しない）。
//! 監査記録は `docs/cpu-gemm-sme-unsafe-audit.md` §13。

use std::arch::asm;

use super::TileBoundsError;

/// 本カーネルが 1 回で処理する C ブロックの行数（ZA タイル 4 枚 × SVL=512 bit の 16 行）。
pub const M: usize = 64;
/// C ブロックの列数（f32 ベクトル 1 本 = 16 要素）。
pub const NR: usize = 16;

const _: () = assert!(M == 4 * NR);

/// 長さ・ストライドの事前検査（REQ-8）。`kernel_unchecked`／`scalar_fallback` 共通。
fn check_bounds(
    ap_len: usize,
    bp_len: usize,
    b_stride: usize,
    c_len: usize,
    ldc: usize,
    k: usize,
) -> Result<(), TileBoundsError> {
    match M.checked_mul(k) {
        Some(expected) if expected == ap_len => {}
        _ => {
            return Err(TileBoundsError::PanelLengthMismatch {
                panel: "ap",
                actual: ap_len,
            });
        }
    }
    if b_stride < NR {
        return Err(TileBoundsError::LdcTooSmall {
            ldc: b_stride,
            nr: NR,
        });
    }
    if k > 0 {
        let need = (k - 1)
            .checked_mul(b_stride)
            .and_then(|v| v.checked_add(NR));
        match need {
            Some(need) if bp_len >= need => {}
            _ => {
                return Err(TileBoundsError::PanelLengthMismatch {
                    panel: "bp",
                    actual: bp_len,
                });
            }
        }
    }
    super::check_c_tile_bounds(M, NR, ldc, c_len)
}

/// SME 命令を発行する演算本体（非公開）。
///
/// # Safety
///
/// 呼び出し元は次を保証する。
/// - 実行 CPU が SME・`SME_F32F32` に対応し SVL=64 バイト（`SmeKernel::try_new` と実行スレッド上の
///   再確認）。
/// - 直前に `TPIDR2_EL0 == 0`（`super::sme::has_pending_lazy_za_save` が偽）を確認済み。
/// - `check_bounds` を通過済み: `ap.len() == 64*k`、`k>0` のとき `bp.len() >= (k-1)*b_stride+16`、
///   `b_stride >= 16`、`ldc >= 16`、`c.len() >= 63*ldc+16`。
unsafe fn compute_thin_m64(
    ap: &[f32],
    bp: &[f32],
    b_stride: usize,
    c: &mut [f32],
    ldc: usize,
    k: usize,
) {
    let c_ptr = c.as_mut_ptr();
    let a_ptr = ap.as_ptr();
    let b_ptr = bp.as_ptr();
    let ldc_bytes = ldc * size_of::<f32>();
    let b_stride_bytes = b_stride * size_of::<f32>();
    // タイル t（C 行 16t..16t+16）の先頭行ポインタ。オフセットは検査済み長の範囲内
    // （最大 63*ldc+16 要素）のため `wrapping_add` でも実質オーバーフローしない。
    let c0 = c_ptr;
    let c1 = c_ptr.wrapping_add(16 * ldc);
    let c2 = c_ptr.wrapping_add(32 * ldc);
    let c3 = c_ptr.wrapping_add(48 * ldc);
    let back_bytes = 16 * ldc_bytes;

    // SAFETY: 呼び出し元契約（本関数 `# Safety` 節）により SME 対応・SVL=64 バイト・
    // 保留中 lazy ZA save なし・長さ検査済み。各ロード／ストアの最大オフセット:
    // - ap: `a` は 1 反復ごとに +256 バイト。反復 p で [a, a+256) を読み、最大は
    //   64*(k-1)+64 = 64*k 要素 = ap.len()（z0〜z3 は `mul vl` = 64 バイト刻みで 4 本）。
    // - bp: `b` は 1 反復ごとに +b_stride_bytes。最大は (k-1)*b_stride+16 <= bp.len()。
    // - c: タイル t・行 r の位置は (16t+r)*ldc+16 <= 63*ldc+16 <= c.len()。
    //   プリロード後に各ポインタは 16 行ぶん進むので `back_bytes` で戻してからストアする。
    //
    // レジスタ・状態契約:
    // - ZA0〜ZA3 の**全タイルを破壊**する（`smstart` で ZA が初期化され、`smstop` で破棄される）。
    //   呼び出し元の ZA 内容は、保留中 lazy save が無い（TPIDR2_EL0==0）場合に限り失ってよい。
    // - `smstart`/`smstop` は同一ブロック内で対にする。ストリーミング中の SIMD は SME 系のみ。
    // - `smstart` は Z0-Z31/P0-P15/FFR を不定化するため v0-v31・p0-p15・ffr を全列挙する。
    // - `mova` のスライスインデックスは w12 限定なので `out("w12")` のみ宣言する。
    // - K は 64bit の X レジスタで扱う（`cbz`/`subs` の W 版は 2^32 超を切り詰めるため）。
    // - フラグを書き換えるので `preserves_flags` なし。メモリへアクセスするので `nomem` なし。
    // - 残留リスク L3（ストリーミング中のシグナル配送）は `sme.rs` から引き継ぐ。K 全長を 1 区間で
    //   流すためストリーミング区間は長くなるが、新しい種類のリスクではない（監査 §13）。
    unsafe {
        asm!(
            ".arch_extension sme",
            "smstart",
            "ptrue p0.s",
            // C プリロード: タイル t の行 r を za{t}h.s[r] へ。
            "mov w12, #0",
            "10:",
            "ld1w {{z0.s}}, p0/z, [{c0}]",
            "mova za0h.s[w12, #0], p0/m, z0.s",
            "ld1w {{z0.s}}, p0/z, [{c1}]",
            "mova za1h.s[w12, #0], p0/m, z0.s",
            "ld1w {{z0.s}}, p0/z, [{c2}]",
            "mova za2h.s[w12, #0], p0/m, z0.s",
            "ld1w {{z0.s}}, p0/z, [{c3}]",
            "mova za3h.s[w12, #0], p0/m, z0.s",
            "add {c0}, {c0}, {ldc_bytes}",
            "add {c1}, {c1}, {ldc_bytes}",
            "add {c2}, {c2}, {ldc_bytes}",
            "add {c3}, {c3}, {ldc_bytes}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lt 10b",
            // ストア用にポインタを 16 行ぶん戻す。
            "sub {c0}, {c0}, {back}",
            "sub {c1}, {c1}, {back}",
            "sub {c2}, {c2}, {back}",
            "sub {c3}, {c3}, {back}",
            // K ループ（k == 0 ならスキップ。K は X レジスタで判定）。
            "cbz {kc}, 12f",
            "11:",
            "ld1w {{z0.s}}, p0/z, [{a}]",
            "ld1w {{z1.s}}, p0/z, [{a}, #1, mul vl]",
            "ld1w {{z2.s}}, p0/z, [{a}, #2, mul vl]",
            "ld1w {{z3.s}}, p0/z, [{a}, #3, mul vl]",
            "ld1w {{z4.s}}, p0/z, [{b}]",
            "fmopa za0.s, p0/m, p0/m, z0.s, z4.s",
            "fmopa za1.s, p0/m, p0/m, z1.s, z4.s",
            "fmopa za2.s, p0/m, p0/m, z2.s, z4.s",
            "fmopa za3.s, p0/m, p0/m, z3.s, z4.s",
            "add {a}, {a}, #256",
            "add {b}, {b}, {b_stride}",
            "subs {kc}, {kc}, #1",
            "b.ne 11b",
            "12:",
            // ストア: プリロードと同じ順で書き戻す。
            "mov w12, #0",
            "13:",
            "mova z0.s, p0/m, za0h.s[w12, #0]",
            "st1w {{z0.s}}, p0, [{c0}]",
            "mova z0.s, p0/m, za1h.s[w12, #0]",
            "st1w {{z0.s}}, p0, [{c1}]",
            "mova z0.s, p0/m, za2h.s[w12, #0]",
            "st1w {{z0.s}}, p0, [{c2}]",
            "mova z0.s, p0/m, za3h.s[w12, #0]",
            "st1w {{z0.s}}, p0, [{c3}]",
            "add {c0}, {c0}, {ldc_bytes}",
            "add {c1}, {c1}, {ldc_bytes}",
            "add {c2}, {c2}, {ldc_bytes}",
            "add {c3}, {c3}, {ldc_bytes}",
            "add w12, w12, #1",
            "cmp w12, #16",
            "b.lt 13b",
            "smstop",
            c0 = inout(reg) c0 => _,
            c1 = inout(reg) c1 => _,
            c2 = inout(reg) c2 => _,
            c3 = inout(reg) c3 => _,
            a = inout(reg) a_ptr => _,
            b = inout(reg) b_ptr => _,
            kc = inout(reg) k => _,
            ldc_bytes = in(reg) ldc_bytes,
            back = in(reg) back_bytes,
            b_stride = in(reg) b_stride_bytes,
            out("w12") _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _, out("v4") _,
            out("v5") _, out("v6") _, out("v7") _, out("v8") _, out("v9") _,
            out("v10") _, out("v11") _, out("v12") _, out("v13") _, out("v14") _,
            out("v15") _, out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _, out("v24") _,
            out("v25") _, out("v26") _, out("v27") _, out("v28") _, out("v29") _,
            out("v30") _, out("v31") _,
            out("p0") _, out("p1") _, out("p2") _, out("p3") _, out("p4") _,
            out("p5") _, out("p6") _, out("p7") _, out("p8") _, out("p9") _,
            out("p10") _, out("p11") _, out("p12") _, out("p13") _, out("p14") _,
            out("p15") _,
            out("ffr") _,
            options(nostack),
        );
    }
}

/// 境界検査つきの SME 入口。`ap` は `[p][64]`（`ap[p*64+i]`）、`bp` は行 p が `bp[p*b_stride..][..16]`、
/// `c` は 64 行（行間隔 `ldc`）× 16 列。検査違反は型付きエラーで返し、asm には入らない。
///
/// 保留中の lazy ZA save（`TPIDR2_EL0 != 0`）を検出したときは `sme::kernel_unchecked_with_ldc` と
/// 同じ自己防御として ZA に触れず [`scalar_fallback`] へ切り替える（bit 一致）。
///
/// # Safety
///
/// 実行 CPU が SME・`SME_F32F32` に対応し SVL=64 バイトであること
/// （`SmeKernel::try_new` と実行スレッド上の再確認）。`TPIDR2_EL0` の確認は本関数が行う。
pub(crate) unsafe fn kernel_unchecked(
    ap: &[f32],
    bp: &[f32],
    b_stride: usize,
    c: &mut [f32],
    ldc: usize,
    k: usize,
) -> Result<(), TileBoundsError> {
    check_bounds(ap.len(), bp.len(), b_stride, c.len(), ldc, k)?;
    // SAFETY: 呼び出し元契約により SME 対応（TPIDR2_EL0 を読める）。
    if unsafe { super::sme::has_pending_lazy_za_save() } {
        scalar_fallback_unchecked(ap, bp, b_stride, c, ldc, k);
        return Ok(());
    }
    // SAFETY: `compute_thin_m64` の `# Safety` 契約: SME 対応は呼び出し元契約、
    // lazy save なしは直前に確認、長さは `check_bounds` で検査済み。
    unsafe { compute_thin_m64(ap, bp, b_stride, c, ldc, k) };
    Ok(())
}

/// `compute_thin_m64` と同一の演算列（各 `c[i][j]` を p 昇順に 1 回ずつ `mul_add`）を安全な Rust で
/// 再現する境界検査つきフォールバック。SME 非対応・SVL 不一致・保留中 lazy save のスレッドで使う。
pub(crate) fn scalar_fallback(
    ap: &[f32],
    bp: &[f32],
    b_stride: usize,
    c: &mut [f32],
    ldc: usize,
    k: usize,
) -> Result<(), TileBoundsError> {
    check_bounds(ap.len(), bp.len(), b_stride, c.len(), ldc, k)?;
    scalar_fallback_unchecked(ap, bp, b_stride, c, ldc, k);
    Ok(())
}

/// 検査済み入力前提の本体（`check_bounds` を通過した呼び出しだけが到達する）。
fn scalar_fallback_unchecked(
    ap: &[f32],
    bp: &[f32],
    b_stride: usize,
    c: &mut [f32],
    ldc: usize,
    k: usize,
) {
    for p in 0..k {
        for i in 0..M {
            let a_val = ap[p * M + i];
            for j in 0..NR {
                let idx = i * ldc + j;
                c[idx] = a_val.mul_add(bp[p * b_stride + j], c[idx]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gemm_blis::microkernel::SmeKernel;

    fn xorshift32_vec(seed: u32, len: usize) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f64 / u32::MAX as f64) as f32 - 0.5
            })
            .collect()
    }

    /// 単一連続 K ループの参照（`gemm_naive` と同じ演算列）。
    fn reference(
        ap: &[f32],
        bp: &[f32],
        b_stride: usize,
        c: &[f32],
        ldc: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut out = c.to_vec();
        for i in 0..M {
            for j in 0..NR {
                let mut acc = out[i * ldc + j];
                for p in 0..k {
                    acc = ap[p * M + i].mul_add(bp[p * b_stride + j], acc);
                }
                out[i * ldc + j] = acc;
            }
        }
        out
    }

    /// REQ-8: 長さ・ストライド違反は asm より前に `Err` になり panic しない（非 SME ホストでも安全）。
    #[test]
    fn kernel_unchecked_returns_err_instead_of_panicking_on_bad_bounds() {
        let k = 4usize;
        let ap = vec![0.0f32; M * k];
        let bp = vec![0.0f32; k * NR];
        let mut c = vec![0.0f32; M * NR];
        // SAFETY: 検査エラーは asm・TPIDR2 読み取りより前に返る。
        unsafe {
            assert!(kernel_unchecked(&ap[1..], &bp, NR, &mut c, NR, k).is_err());
            assert!(kernel_unchecked(&ap, &bp[1..], NR, &mut c, NR, k).is_err());
            assert!(kernel_unchecked(&ap, &bp, NR, &mut c[1..], NR, k).is_err());
            assert!(kernel_unchecked(&ap, &bp, NR, &mut c, NR - 1, k).is_err());
            assert!(kernel_unchecked(&ap, &bp, NR - 1, &mut c, NR, k).is_err());
            assert!(kernel_unchecked(&ap, &bp, NR, &mut c, NR, usize::MAX).is_err());
            assert!(kernel_unchecked(&ap, &bp, usize::MAX, &mut c, NR, k).is_err());
        }
    }

    #[test]
    fn scalar_fallback_rejects_bad_bounds_and_matches_reference() {
        let k = 9usize;
        let b_stride = 40usize;
        let ldc = 20usize;
        let ap = xorshift32_vec(1, M * k);
        let bp = xorshift32_vec(2, (k - 1) * b_stride + NR);
        let c0 = xorshift32_vec(3, 63 * ldc + NR);
        let mut c = c0.clone();
        scalar_fallback(&ap, &bp, b_stride, &mut c, ldc, k).unwrap();
        assert_eq!(c, reference(&ap, &bp, b_stride, &c0, ldc, k));
        assert!(scalar_fallback(&ap[1..], &bp, b_stride, &mut c, ldc, k).is_err());
    }

    /// k == 0 は C を変更しない。
    #[test]
    fn scalar_fallback_k0_is_noop() {
        let mut c = xorshift32_vec(5, M * NR);
        let before = c.clone();
        scalar_fallback(&[], &[], NR, &mut c, NR, 0).unwrap();
        assert_eq!(c, before);
    }

    fn run_hw(k: usize, b_stride: usize, ldc: usize, seed: u32) {
        let Some(_kernel) = SmeKernel::try_new() else {
            return;
        };
        let ap = xorshift32_vec(seed, M * k);
        let bp = xorshift32_vec(seed ^ 0x55, k.saturating_sub(1) * b_stride + NR);
        let c0 = xorshift32_vec(seed ^ 0xaa, 63 * ldc + NR);
        let mut c = c0.clone();
        // SAFETY: try_new() が Some（SME 対応・SVL=64）。
        unsafe { kernel_unchecked(&ap, &bp, b_stride, &mut c, ldc, k) }.unwrap();
        assert_eq!(c, reference(&ap, &bp, b_stride, &c0, ldc, k), "k={k}");
        let mut c2 = c0.clone();
        // SAFETY: 同上。
        unsafe { kernel_unchecked(&ap, &bp, b_stride, &mut c2, ldc, k) }.unwrap();
        assert_eq!(c, c2, "決定性 k={k}");
    }

    #[test]
    #[ignore = "SME 実機（M4 系）専用。#3235"]
    fn sme_thin_m64_kernel_matches_scalar_reference() {
        for &(k, bs, ldc) in &[
            (1, 16, 16),
            (7, 16, 16),
            (64, 16, 16),
            (33, 784, 16),
            (50, 32, 48),
        ] {
            run_hw(k, bs, ldc, 0x1234 + k as u32);
        }
        run_hw(0, 16, 16, 9);
    }

    #[test]
    #[ignore = "SME 実機（M4 系）専用。#3235"]
    fn sme_thin_m64_kernel_matches_scalar_reference_denormal() {
        let Some(_kernel) = SmeKernel::try_new() else {
            return;
        };
        let k = 8usize;
        let ap: Vec<f32> = (0..M * k)
            .map(|i| f32::from_bits((i as u32 % 7) + 1))
            .collect();
        let bp: Vec<f32> = (0..k * NR)
            .map(|i| f32::from_bits((i as u32 % 5) + 1))
            .collect();
        let c0 = vec![f32::from_bits(3); M * NR];
        let mut c = c0.clone();
        // SAFETY: try_new() が Some。
        unsafe { kernel_unchecked(&ap, &bp, NR, &mut c, NR, k) }.unwrap();
        assert_eq!(c, reference(&ap, &bp, NR, &c0, NR, k));
    }

    #[test]
    #[ignore = "SME 実機（M4 系）専用。#3235"]
    fn sme_thin_m64_kernel_matches_scalar_reference_denormal_result_from_normal_operands() {
        let Some(_kernel) = SmeKernel::try_new() else {
            return;
        };
        let k = 4usize;
        let ap = vec![1.0e-20f32; M * k];
        let bp = vec![1.0e-20f32; k * NR];
        let c0 = vec![0.0f32; M * NR];
        let mut c = c0.clone();
        // SAFETY: try_new() が Some。
        unsafe { kernel_unchecked(&ap, &bp, NR, &mut c, NR, k) }.unwrap();
        assert_eq!(c, reference(&ap, &bp, NR, &c0, NR, k));
    }

    #[test]
    #[ignore = "SME 実機（M4 系）専用。#3235"]
    fn sme_thin_m64_kernel_nan_input_does_not_panic() {
        let Some(_kernel) = SmeKernel::try_new() else {
            return;
        };
        let k = 3usize;
        let mut ap = xorshift32_vec(7, M * k);
        ap[5] = f32::NAN;
        let bp = xorshift32_vec(8, k * NR);
        let mut c = vec![0.0f32; M * NR];
        // SAFETY: try_new() が Some。
        unsafe { kernel_unchecked(&ap, &bp, NR, &mut c, NR, k) }.unwrap();
        assert!(c[5 * NR..][..NR].iter().all(|v| v.is_nan()));
    }
}
