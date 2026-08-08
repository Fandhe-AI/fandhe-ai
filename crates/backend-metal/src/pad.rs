//! `simdgroup_matrix`（8×8 ハードウェア行列演算命令）向け 8 の倍数パディング（TASK-1.8c・#40）。
//!
//! [`crate::gemm::MetalGemm::dispatch_variant`] が `GemmVariant::Simdgroup`
//! を選択した際、入力 A・B を 8 の倍数の実効次元へ 0 パディングしてから
//! Metal バッファへアップロードし、readback 後は [`unpad_matrix`] で元の
//! m×n 形状へ切り出す（呼び出し元にパディングの有無を隠蔽する。
//! `docs/spec/03-poc/poc-v2-4-metal-gemm/code/rust/src/metal_gemm.rs`
//! の `pad8`/`pad_matrix`/`GemmCase::read_result` の移植）。
//!
//! 本モジュールは `objc2` 系 FFI に一切触れない純粋関数のみで構成する
//! ため `cfg(target_os = "macos")` を付けず、Linux（CI・本実装環境）の
//! `cargo test -p backend-metal` でも単体テストが回るようにしてある
//! （`crate::gemm`・`crate::pipeline` 等の FFI 境界モジュールとは異なり
//! macOS 実機なしで検証できる部分を切り出した設計判断）。
//!
//! **0 パディングが数値に影響しない理由**（[`pad_matrix`] のコメント参照）:
//! `gemm_simdgroup` カーネルは内積を `simdgroup_multiply_accumulate` の
//! 逐次累積で計算する。パディングした行・列は値 0 で埋めており、
//! `0 * x = 0` の寄与は加算しても和を変えないため、k 方向のパディング
//! （実効 k のうち元の k を超える範囲）を含めて計算しても元の GEMM 結果と
//! bit 完全に一致する（REQ-2 の FMA 契約・複合判定に影響しない）。

/// `x` を 8 の倍数へ切り上げる（`simdgroup_float8x8` の 8×8 タイル制約）。
pub fn pad8(x: usize) -> usize {
    x.div_ceil(8) * 8
}

/// `src`（`rows`×`cols`、行優先）を `rows_eff`×`cols_eff` へ 0 パディングする。
///
/// サイズが既に一致する場合（`Naive`/`Tiled` 経路等パディング不要な variant）
/// は `src` をそのまま借用で返し複製しない（[`crate::gemm::MetalGemm::dispatch_variant`]
/// が variant を問わず本関数を呼べるようにしつつ、パディング不要な経路で
/// 毎ディスパッチ発生していた不要な全行列コピーを避けるための `Cow` 化）。
pub fn pad_matrix<'a>(
    src: &'a [f32],
    rows: usize,
    cols: usize,
    rows_eff: usize,
    cols_eff: usize,
) -> std::borrow::Cow<'a, [f32]> {
    if rows == rows_eff && cols == cols_eff {
        return std::borrow::Cow::Borrowed(src);
    }
    let mut out = vec![0.0f32; rows_eff * cols_eff];
    for r in 0..rows {
        out[r * cols_eff..r * cols_eff + cols].copy_from_slice(&src[r * cols..r * cols + cols]);
    }
    std::borrow::Cow::Owned(out)
}

/// [`pad_matrix`] の逆操作。`src`（`rows_eff`×`cols_eff`、行優先）から
/// 先頭 `rows`×`cols` を切り出す（末尾のパディング行・列を捨てる）。
///
/// [`crate::gemm::MetalGemm::dispatch_variant`] が Metal readback 後の
/// C バッファ（実効次元）を呼び出し元へ渡す元の m×n 形状へ戻すために呼ぶ。
/// `src` を値渡し（所有権移動）で受け取り、形状が一致するパディング不要
/// 経路（`Naive`/`Tiled`、および既にアラインメント済みの `Simdgroup`）では
/// `src` をそのまま返す（複製しない）。呼び出し元は readback で得た
/// 所有 `Vec` をそのまま渡すことで、ホスト側の結果メモリを一時的に倍化
/// させる不要な全体コピーを避ける（レビュー指摘 #246 対応）。
pub fn unpad_matrix(
    src: Vec<f32>,
    rows_eff: usize,
    cols_eff: usize,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    if rows_eff == rows && cols_eff == cols {
        return src;
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        out[r * cols..r * cols + cols].copy_from_slice(&src[r * cols_eff..r * cols_eff + cols]);
    }
    out
}

/// [`pad_matrix`] の f16 版（TASK-8.3b・#156）。
/// `crate::gemm::MetalGemm::dispatch_f16` が `gemm_simdgroup_f16`
/// （half 型統一 simdgroup タイル。`shaders/gemm.metal` 参照）向けに A・B を
/// 8 の倍数の実効次元へ 0 パディングする際に呼ぶ。0 パディングが数値に
/// 影響しない理由は [`pad_matrix`] のモジュールコメントと同じ（`half` の
/// `0.0` も `0 * x = 0` の寄与を持つ）。
pub fn pad_matrix_f16<'a>(
    src: &'a [half::f16],
    rows: usize,
    cols: usize,
    rows_eff: usize,
    cols_eff: usize,
) -> std::borrow::Cow<'a, [half::f16]> {
    if rows == rows_eff && cols == cols_eff {
        return std::borrow::Cow::Borrowed(src);
    }
    let mut out = vec![half::f16::from_f32(0.0); rows_eff * cols_eff];
    for r in 0..rows {
        out[r * cols_eff..r * cols_eff + cols].copy_from_slice(&src[r * cols..r * cols + cols]);
    }
    std::borrow::Cow::Owned(out)
}

/// [`unpad_matrix`] の f16 版（TASK-8.3b・#156）。[`crate::gemm::MetalGemm::dispatch_f16`]
/// が Metal readback 後の C バッファ（実効次元。half 型）を呼び出し元へ渡す
/// 元の m×n 形状へ戻すために呼ぶ。
pub fn unpad_matrix_f16(
    src: Vec<half::f16>,
    rows_eff: usize,
    cols_eff: usize,
    rows: usize,
    cols: usize,
) -> Vec<half::f16> {
    if rows_eff == rows && cols_eff == cols {
        return src;
    }
    let mut out = vec![half::f16::from_f32(0.0); rows * cols];
    for r in 0..rows {
        out[r * cols..r * cols + cols].copy_from_slice(&src[r * cols_eff..r * cols_eff + cols]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad8_rounds_up_to_multiple_of_eight() {
        assert_eq!(pad8(0), 0);
        assert_eq!(pad8(1), 8);
        assert_eq!(pad8(7), 8);
        assert_eq!(pad8(8), 8);
        assert_eq!(pad8(9), 16);
        assert_eq!(pad8(64), 64);
    }

    #[test]
    fn pad_matrix_passthrough_when_size_matches() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let out = pad_matrix(&src, 2, 2, 2, 2);
        assert_eq!(out, src);
    }

    #[test]
    fn pad_matrix_zero_pads_rows_and_cols() {
        // 2x3 -> 4x8（末尾は 0 埋め）。
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = pad_matrix(&src, 2, 3, 4, 8);
        assert_eq!(out.len(), 4 * 8);
        // 行 0: [1,2,3,0,0,0,0,0]
        assert_eq!(&out[0..8], &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // 行 1: [4,5,6,0,0,0,0,0]
        assert_eq!(&out[8..16], &[4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // 行 2・3（パディング行）は全 0。
        assert_eq!(&out[16..24], &[0.0; 8]);
        assert_eq!(&out[24..32], &[0.0; 8]);
    }

    #[test]
    fn unpad_matrix_passthrough_when_size_matches() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0];
        let ptr = src.as_ptr();
        let out = unpad_matrix(src, 2, 2, 2, 2);
        assert_eq!(out, vec![1.0f32, 2.0, 3.0, 4.0]);
        // 形状一致時は複製せず所有権が移動することを確認（レビュー指摘 #246 対応）。
        assert_eq!(out.as_ptr(), ptr);
    }

    #[test]
    fn unpad_matrix_is_inverse_of_pad_matrix() {
        let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let padded = pad_matrix(&src, 2, 3, 8, 8).into_owned();
        let unpadded = unpad_matrix(padded, 8, 8, 2, 3);
        assert_eq!(unpadded, src);
    }

    #[test]
    fn unpad_matrix_discards_padding_tail() {
        // 8x8 の全 1 行列から先頭 3x5 のみ切り出す。
        let padded = vec![1.0f32; 64];
        let out = unpad_matrix(padded, 8, 8, 3, 5);
        assert_eq!(out.len(), 15);
        assert!(out.iter().all(|&v| v == 1.0));
    }

    // --- f16 版（TASK-8.3b・#156。pad_matrix/unpad_matrix と同じ検証を型だけ差し替え） ---

    #[test]
    fn pad_matrix_f16_passthrough_when_size_matches() {
        let src = vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
        ];
        let out = pad_matrix_f16(&src, 2, 2, 2, 2);
        assert_eq!(out.as_ref(), src.as_slice());
    }

    #[test]
    fn pad_matrix_f16_zero_pads_rows_and_cols() {
        let src: Vec<half::f16> = (1..=6).map(|v| half::f16::from_f32(v as f32)).collect();
        let out = pad_matrix_f16(&src, 2, 3, 4, 8);
        assert_eq!(out.len(), 4 * 8);
        let zero = half::f16::from_f32(0.0);
        assert_eq!(
            &out[0..8],
            &[
                half::f16::from_f32(1.0),
                half::f16::from_f32(2.0),
                half::f16::from_f32(3.0),
                zero,
                zero,
                zero,
                zero,
                zero,
            ]
        );
        assert!(out[16..32].iter().all(|&v| v == zero));
    }

    #[test]
    fn unpad_matrix_f16_is_inverse_of_pad_matrix_f16() {
        let src: Vec<half::f16> = (1..=6).map(|v| half::f16::from_f32(v as f32)).collect();
        let padded = pad_matrix_f16(&src, 2, 3, 8, 8).into_owned();
        let unpadded = unpad_matrix_f16(padded, 8, 8, 2, 3);
        assert_eq!(unpadded, src);
    }

    #[test]
    fn unpad_matrix_f16_passthrough_preserves_ownership() {
        let src = vec![half::f16::from_f32(1.0), half::f16::from_f32(2.0)];
        let ptr = src.as_ptr();
        let out = unpad_matrix_f16(src, 1, 2, 1, 2);
        assert_eq!(out.as_ptr(), ptr);
    }
}
