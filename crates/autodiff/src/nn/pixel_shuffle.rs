//! PixelShuffle・PixelUnshuffle（`torch.nn.PixelShuffle`／
//! `torch.nn.PixelUnshuffle` 相当。イシュー #2162・親 #2131）。
//!
//! チャネル軸と空間軸の間で要素を並べ替えるだけの層であり、
//! [`Var::reshape`]（`var.rs`）・[`Var::permute`]・[`Var::contiguous`]
//! の合成のみで実装する（新規 `Op`／`BackendOps`／VJP／カーネルは
//! 追加しない）。VJP は reshape・permute（view）と `Op::Contiguous`
//! （恒等パススルー）の既存 VJP 合成で自動的に成り立つため
//! `grad.rs` は変更していない。
//!
//! PyTorch と同じ軸の並びを採用する:
//! `out[.., c, h*r+i, w*r+j] = in[.., c*r*r+i*r+j, h, w]`
//! （`PixelShuffle`。`PixelUnshuffle` はこの逆変換）。先頭のバッチ軸
//! （0 本以上）はそのまま素通りする（rank 3 以上の `[*, C, H, W]` を
//! 受理。rank 2 以下は拒否）。
//!
//! `CPU 先行・CUDA／Metal は `Unsupported` フォールバック`という
//! イシュー #2162 の契約との関係: `reshape`／`permute` は zero-copy
//! の view ノード、`Op::Contiguous` はホスト側で eager 実行される
//! ノードのため、CUDA／Metal の tape でも追加実装なしで到達できる
//! （#2159 の `Unflatten`／`Identity` と同じ判断。
//! `docs/autodiff-pixel-shuffle-decision.md` §2.1）。

use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::var::Var;

/// `in_shape`（`[B.., C*r*r, H, W]`）から `PixelShuffle` の出力 shape
/// （`[B.., C, H*r, W*r]`）を計算する（[`PixelShuffle::forward`]／
/// [`PixelShuffle::forward_host`] で共有し、判定基準が食い違う迂回
/// 経路を作らない。`.claude/rules/security.md` A08）。
///
/// 検査順序: ①rank が 3 未満なら [`ShapeError::RankMismatch`] →
/// ②`r*r` を `checked_mul` で計算しオーバーフローは
/// [`ShapeError::ElementCountOverflow`] → ③チャネル軸が `r*r` で
/// 割り切れることを [`ShapeError::ShapeMismatch`] で検査 → ④出力の
/// 空間軸 `H*r`／`W*r` も `checked_mul` で計算 → ⑤出力 shape を返す。
/// この順序により、tape を操作する前に shape の値だけで Err 経路が
/// 確定する（`Err` 経路で孤児ノードを残さない。#2159 `ConvTranspose1d`
/// と同じ規律）。
pub(crate) fn pixel_shuffle_out_shape(
    in_shape: &[usize],
    upscale_factor: usize,
) -> Result<Vec<usize>, ShapeError> {
    let rank = in_shape.len();
    if rank < 3 {
        return Err(ShapeError::RankMismatch {
            expected: 3,
            actual: rank,
        });
    }
    let r2 = upscale_factor
        .checked_mul(upscale_factor)
        .ok_or(ShapeError::ElementCountOverflow)?;
    let c_in = in_shape[rank - 3];
    if r2 == 0 || !c_in.is_multiple_of(r2) {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![c_in],
            rhs: vec![r2],
        });
    }
    let c_out = c_in / r2;
    let h = in_shape[rank - 2];
    let w = in_shape[rank - 1];
    let h_out = h
        .checked_mul(upscale_factor)
        .ok_or(ShapeError::ElementCountOverflow)?;
    let w_out = w
        .checked_mul(upscale_factor)
        .ok_or(ShapeError::ElementCountOverflow)?;

    let mut out = Vec::with_capacity(rank);
    out.extend_from_slice(&in_shape[..rank - 3]);
    out.extend_from_slice(&[c_out, h_out, w_out]);
    Ok(out)
}

/// `in_shape`（`[B.., C, H*r, W*r]`）から `PixelUnshuffle` の出力
/// shape（`[B.., C*r*r, H, W]`）を計算する（[`pixel_shuffle_out_shape`]
/// と同じ「forward／forward_host で共有」規律。検査順序:
/// ①rank が 3 未満なら [`ShapeError::RankMismatch`] → ②`r*r` を
/// `checked_mul`（オーバーフローは [`ShapeError::ElementCountOverflow`]）
/// → ③空間軸 `H`／`W` が `r` で割り切れることを
/// [`ShapeError::ShapeMismatch`] で検査 → ④出力チャネル `C*r*r` も
/// `checked_mul` で計算 → ⑤出力 shape を返す。
pub(crate) fn pixel_unshuffle_out_shape(
    in_shape: &[usize],
    downscale_factor: usize,
) -> Result<Vec<usize>, ShapeError> {
    let rank = in_shape.len();
    if rank < 3 {
        return Err(ShapeError::RankMismatch {
            expected: 3,
            actual: rank,
        });
    }
    let r2 = downscale_factor
        .checked_mul(downscale_factor)
        .ok_or(ShapeError::ElementCountOverflow)?;
    let h = in_shape[rank - 2];
    let w = in_shape[rank - 1];
    if downscale_factor == 0
        || !h.is_multiple_of(downscale_factor)
        || !w.is_multiple_of(downscale_factor)
    {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![h, w],
            rhs: vec![downscale_factor],
        });
    }
    let h_out = h / downscale_factor;
    let w_out = w / downscale_factor;
    let c_in = in_shape[rank - 3];
    let c_out = c_in
        .checked_mul(r2)
        .ok_or(ShapeError::ElementCountOverflow)?;

    let mut out = Vec::with_capacity(rank);
    out.extend_from_slice(&in_shape[..rank - 3]);
    out.extend_from_slice(&[c_out, h_out, w_out]);
    Ok(out)
}

/// チャネル軸を空間軸へ並べ替える層（`torch.nn.PixelShuffle` 相当）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelShuffle {
    upscale_factor: usize,
}

impl PixelShuffle {
    /// `upscale_factor` を指定して構築する。`0` は
    /// [`AutodiffError::InvalidArgument`]（`0` 除算・無意味な変換を
    /// 構築時点で拒否する。rank・整除性の検査は入力の shape に依存
    /// するため forward 時（`pixel_shuffle_out_shape` 内部）へ遅延
    /// させる——`Unflatten::new` と同じ遅延検査契約）。
    pub fn new(upscale_factor: usize) -> Result<Self, AutodiffError> {
        if upscale_factor == 0 {
            return Err(AutodiffError::InvalidArgument(
                "PixelShuffle::new: upscale_factor must not be 0".to_string(),
            ));
        }
        Ok(Self { upscale_factor })
    }

    /// アップスケール倍率。
    pub fn upscale_factor(&self) -> usize {
        self.upscale_factor
    }

    /// `pixel_shuffle_out_shape` で出力 shape を検査してから
    /// `contiguous → reshape → permute → contiguous → reshape` の
    /// 5 手順（`docs/autodiff-pixel-shuffle-decision.md` §2.1）を
    /// 積む。出力 shape の検査が先に完了しているため、後続の
    /// `reshape`／`permute` 呼び出しは既に整合が取れた shape 値のみ
    /// を扱い、`Err` 経路は shape 検査の時点で確定する。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let in_shape = input.shape();
        let out_shape = pixel_shuffle_out_shape(&in_shape, self.upscale_factor)
            .map_err(AutodiffError::Shape)?;
        let rank = in_shape.len();
        let nb = rank - 3;
        let r = self.upscale_factor;
        let c = out_shape[nb];
        let h = in_shape[rank - 2];
        let w = in_shape[rank - 1];

        let mut mid_shape: Vec<usize> = in_shape[..nb].to_vec();
        mid_shape.extend_from_slice(&[c, r, r, h, w]);

        let mut perm: Vec<usize> = (0..nb).collect();
        perm.extend_from_slice(&[nb, nb + 3, nb + 1, nb + 4, nb + 2]);

        let x = input.contiguous()?;
        let x = x.reshape(&mid_shape)?;
        let x = x.permute(&perm)?;
        let x = x.contiguous()?;
        x.reshape(&out_shape)
    }

    /// [`crate::nn::module::Module::forward_host`]（`PixelShuffle`
    /// 実装）の本体。[`Self::forward`] と同じ shape 計算・並べ替え
    /// 手順を tape 不要経路で再現する（純粋なコピーのため `forward`
    /// と bit 単位で一致する）。
    pub fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let in_shape = input.shape();
        let out_shape =
            pixel_shuffle_out_shape(in_shape, self.upscale_factor).map_err(AutodiffError::Shape)?;
        let rank = in_shape.len();
        let nb = rank - 3;
        let r = self.upscale_factor;
        let c = out_shape[nb];
        let h = in_shape[rank - 2];
        let w = in_shape[rank - 1];

        let mut mid_shape: Vec<usize> = in_shape[..nb].to_vec();
        mid_shape.extend_from_slice(&[c, r, r, h, w]);

        let mut perm: Vec<usize> = (0..nb).collect();
        perm.extend_from_slice(&[nb, nb + 3, nb + 1, nb + 4, nb + 2]);

        let x = input.contiguous();
        let x = x.reshape(&mid_shape).map_err(AutodiffError::Shape)?;
        let x = x.permute(&perm).map_err(AutodiffError::Shape)?;
        let x = x.contiguous();
        x.reshape(&out_shape).map_err(AutodiffError::Shape)
    }
}

/// 空間軸をチャネル軸へ並べ替える層（`torch.nn.PixelUnshuffle`
/// 相当。[`PixelShuffle`] の逆変換）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PixelUnshuffle {
    downscale_factor: usize,
}

impl PixelUnshuffle {
    /// `downscale_factor` を指定して構築する。`0` は
    /// [`AutodiffError::InvalidArgument`]（[`PixelShuffle::new`] と
    /// 同じ理由・同じ遅延検査契約）。
    pub fn new(downscale_factor: usize) -> Result<Self, AutodiffError> {
        if downscale_factor == 0 {
            return Err(AutodiffError::InvalidArgument(
                "PixelUnshuffle::new: downscale_factor must not be 0".to_string(),
            ));
        }
        Ok(Self { downscale_factor })
    }

    /// ダウンスケール倍率。
    pub fn downscale_factor(&self) -> usize {
        self.downscale_factor
    }

    /// `pixel_unshuffle_out_shape` で出力 shape を検査してから
    /// `contiguous → reshape → permute → contiguous → reshape` の
    /// 5 手順を積む（[`PixelShuffle::forward`] と対称の構成）。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        let in_shape = input.shape();
        let out_shape = pixel_unshuffle_out_shape(&in_shape, self.downscale_factor)
            .map_err(AutodiffError::Shape)?;
        let rank = in_shape.len();
        let nb = rank - 3;
        let r = self.downscale_factor;
        let c = in_shape[rank - 3];
        let h_out = out_shape[nb + 1];
        let w_out = out_shape[nb + 2];

        let mut mid_shape: Vec<usize> = in_shape[..nb].to_vec();
        mid_shape.extend_from_slice(&[c, h_out, r, w_out, r]);

        let mut perm: Vec<usize> = (0..nb).collect();
        perm.extend_from_slice(&[nb, nb + 2, nb + 4, nb + 1, nb + 3]);

        let x = input.contiguous()?;
        let x = x.reshape(&mid_shape)?;
        let x = x.permute(&perm)?;
        let x = x.contiguous()?;
        x.reshape(&out_shape)
    }

    /// [`crate::nn::module::Module::forward_host`]（`PixelUnshuffle`
    /// 実装）の本体。[`Self::forward`] と同じ shape 計算・並べ替え
    /// 手順を tape 不要経路で再現する（純粋なコピーのため `forward`
    /// と bit 単位で一致する）。
    pub fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let in_shape = input.shape();
        let out_shape = pixel_unshuffle_out_shape(in_shape, self.downscale_factor)
            .map_err(AutodiffError::Shape)?;
        let rank = in_shape.len();
        let nb = rank - 3;
        let r = self.downscale_factor;
        let c = in_shape[rank - 3];
        let h_out = out_shape[nb + 1];
        let w_out = out_shape[nb + 2];

        let mut mid_shape: Vec<usize> = in_shape[..nb].to_vec();
        mid_shape.extend_from_slice(&[c, h_out, r, w_out, r]);

        let mut perm: Vec<usize> = (0..nb).collect();
        perm.extend_from_slice(&[nb, nb + 2, nb + 4, nb + 1, nb + 3]);

        let x = input.contiguous();
        let x = x.reshape(&mid_shape).map_err(AutodiffError::Shape)?;
        let x = x.permute(&perm).map_err(AutodiffError::Shape)?;
        let x = x.contiguous();
        x.reshape(&out_shape).map_err(AutodiffError::Shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::dense_vec;
    use crate::tape::Tape;

    #[test]
    fn pixel_shuffle_new_rejects_zero() {
        assert!(PixelShuffle::new(0).is_err());
    }

    #[test]
    fn pixel_unshuffle_new_rejects_zero() {
        assert!(PixelUnshuffle::new(0).is_err());
    }

    #[test]
    fn pixel_shuffle_r2_hand_computed_example() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![0.0, 1.0, 2.0, 3.0], &[1, 4, 1, 1]).unwrap());

        let y = PixelShuffle::new(2).unwrap().forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_eq!(dense_vec(&y.to_tensor()), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn pixel_unshuffle_r2_hand_computed_example() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&Tensor::new(vec![0.0, 1.0, 2.0, 3.0], &[1, 1, 2, 2]).unwrap());

        let y = PixelUnshuffle::new(2).unwrap().forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, 1, 1]);
        assert_eq!(dense_vec(&y.to_tensor()), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn pixel_shuffle_r1_is_identity() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let input = Tensor::new((1..=24).map(|v| v as f32).collect(), &[2, 3, 2, 2]).unwrap();
        let x = tape.var(&input);

        let y = PixelShuffle::new(1).unwrap().forward(&x).unwrap();
        assert_eq!(y.shape(), x.shape());
        assert_eq!(dense_vec(&y.to_tensor()), dense_vec(&input));
    }

    #[test]
    fn pixel_shuffle_then_unshuffle_round_trips_bit_exact() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let input = Tensor::new((1..=144).map(|v| v as f32).collect(), &[2, 8, 3, 3]).unwrap();
        let x = tape.var(&input);

        let shuffled = PixelShuffle::new(2).unwrap().forward(&x).unwrap();
        assert_eq!(shuffled.shape(), &[2, 2, 6, 6]);
        let restored = PixelUnshuffle::new(2).unwrap().forward(&shuffled).unwrap();
        assert_eq!(restored.shape(), x.shape());
        assert_eq!(dense_vec(&restored.to_tensor()), dense_vec(&input));
    }

    /// 参照実装（素朴な添字式）と bit 単位で一致することを、バッチ軸
    /// なし（rank 3）とバッチ軸 2 本（rank 5）の両方で確認する。
    fn naive_pixel_shuffle(
        data: &[f32],
        batch: &[usize],
        c: usize,
        r: usize,
        h: usize,
        w: usize,
    ) -> Vec<f32> {
        let batch_numel: usize = batch.iter().product::<usize>().max(1);
        let mut out = vec![0.0f32; batch_numel * c * (h * r) * (w * r)];
        for b in 0..batch_numel {
            for cc in 0..c {
                for hh in 0..h {
                    for ww in 0..w {
                        for i in 0..r {
                            for j in 0..r {
                                let in_c = cc * r * r + i * r + j;
                                let in_idx = ((b * (c * r * r) + in_c) * h + hh) * w + ww;
                                let out_h = hh * r + i;
                                let out_w = ww * r + j;
                                let out_idx = ((b * c + cc) * (h * r) + out_h) * (w * r) + out_w;
                                out[out_idx] = data[in_idx];
                            }
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn pixel_shuffle_matches_naive_reference_rank3_no_batch() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let (c, r, h, w) = (2usize, 2usize, 2usize, 3usize);
        let data: Vec<f32> = (0..(c * r * r * h * w)).map(|v| v as f32).collect();
        let input = Tensor::new(data.clone(), &[c * r * r, h, w]).unwrap();
        let x = tape.var(&input);

        let y = PixelShuffle::new(r).unwrap().forward(&x).unwrap();
        assert_eq!(y.shape(), &[c, h * r, w * r]);
        assert_eq!(
            dense_vec(&y.to_tensor()),
            naive_pixel_shuffle(&data, &[], c, r, h, w)
        );
    }

    #[test]
    fn pixel_shuffle_matches_naive_reference_rank5_two_batch_axes() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let (b0, b1, c, r, h, w) = (2usize, 3usize, 2usize, 2usize, 2usize, 2usize);
        let data: Vec<f32> = (0..(b0 * b1 * c * r * r * h * w))
            .map(|v| v as f32)
            .collect();
        let input = Tensor::new(data.clone(), &[b0, b1, c * r * r, h, w]).unwrap();
        let x = tape.var(&input);

        let y = PixelShuffle::new(r).unwrap().forward(&x).unwrap();
        assert_eq!(y.shape(), &[b0, b1, c, h * r, w * r]);
        assert_eq!(
            dense_vec(&y.to_tensor()),
            naive_pixel_shuffle(&data, &[b0, b1], c, r, h, w)
        );
    }

    #[test]
    fn pixel_shuffle_out_shape_rejects_rank_below_3() {
        assert!(pixel_shuffle_out_shape(&[4, 4], 2).is_err());
    }

    #[test]
    fn pixel_shuffle_out_shape_rejects_non_divisible_channels() {
        assert!(pixel_shuffle_out_shape(&[3, 4, 4], 2).is_err());
    }

    #[test]
    fn pixel_shuffle_out_shape_rejects_checked_mul_overflow() {
        assert!(pixel_shuffle_out_shape(&[4, usize::MAX, 4], 2).is_err());
        assert!(pixel_shuffle_out_shape(&[4, 4, 4], usize::MAX).is_err());
    }

    #[test]
    fn pixel_unshuffle_out_shape_rejects_rank_below_3() {
        assert!(pixel_unshuffle_out_shape(&[4, 4], 2).is_err());
    }

    #[test]
    fn pixel_unshuffle_out_shape_rejects_non_divisible_spatial() {
        assert!(pixel_unshuffle_out_shape(&[1, 3, 4], 2).is_err());
        assert!(pixel_unshuffle_out_shape(&[1, 4, 3], 2).is_err());
    }

    #[test]
    fn pixel_unshuffle_out_shape_rejects_checked_mul_overflow() {
        assert!(pixel_unshuffle_out_shape(&[usize::MAX, 4, 4], 2).is_err());
    }

    #[test]
    fn pixel_shuffle_forward_host_matches_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let host_ops = crate::test_support::test_ops();
        let input = Tensor::new((1..=144).map(|v| v as f32).collect(), &[2, 8, 3, 3]).unwrap();
        let layer = PixelShuffle::new(2).unwrap();

        let via_host = layer.forward_host(host_ops.as_ref(), &input).unwrap();
        let via_tape = layer.forward(&tape.var(&input)).unwrap();

        assert_eq!(via_host.shape(), via_tape.shape());
        assert_eq!(dense_vec(&via_host), dense_vec(&via_tape.to_tensor()));
    }

    #[test]
    fn pixel_unshuffle_forward_host_matches_forward() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let host_ops = crate::test_support::test_ops();
        let input = Tensor::new((1..=144).map(|v| v as f32).collect(), &[2, 2, 6, 6]).unwrap();
        let layer = PixelUnshuffle::new(2).unwrap();

        let via_host = layer.forward_host(host_ops.as_ref(), &input).unwrap();
        let via_tape = layer.forward(&tape.var(&input)).unwrap();

        assert_eq!(via_host.shape(), via_tape.shape());
        assert_eq!(dense_vec(&via_host), dense_vec(&via_tape.to_tensor()));
    }

    #[test]
    fn pixel_shuffle_forward_accepts_non_contiguous_input_matching_contiguous_result() {
        let tape = Tape::new_with_ops(crate::test_support::test_ops());
        let input = Tensor::new((1..=144).map(|v| v as f32).collect(), &[2, 8, 3, 3]).unwrap();
        let x = tape.var(&input);
        // H == W == 3 のため軸 2・3 の入れ替えは shape を変えずに
        // 非 contiguous 化する。`Var::contiguous`（`forward` 冒頭）が
        // 明示コピーで吸収し、事前に `contiguous()` 済みの入力と同じ
        // 結果になることを確認する（`docs/autodiff-pixel-shuffle-
        // decision.md` §2.1）。
        let xt = x.permute(&[0, 1, 3, 2]).unwrap();
        assert!(!xt.to_tensor().is_contiguous());

        let layer = PixelShuffle::new(2).unwrap();
        let y = layer.forward(&xt).unwrap();
        let xt_contig = xt.contiguous().unwrap();
        let expected = layer.forward(&xt_contig).unwrap();

        assert_eq!(y.shape(), expected.shape());
        assert_eq!(dense_vec(&y.to_tensor()), dense_vec(&expected.to_tensor()));
    }
}
