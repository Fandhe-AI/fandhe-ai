//! ChannelShuffle（`torch.nn.ChannelShuffle`／`F.channel_shuffle` 相当。
//! イシュー #3822・設計 #3746・承認 #3768 コメント 6554564412）。
//!
//! チャネル軸（dim 1 固定）を `groups` 個のグループに分けて転置するだけの
//! 並べ替え層であり、[`Var::reshape`]・[`Var::permute`]・[`Var::contiguous`]
//! の合成のみで実装する（新規 `Op`／`BackendOps`／VJP／カーネル／`unsafe`／
//! 依存は追加しない。`grad.rs` も変更しない）。PixelShuffle
//! （`pixel_shuffle.rs`）と同型で、`CPU 先行・CUDA／Metal は追加実装なし`
//! の判断も同じ（`Contiguous` はホストで eager 実行される。
//! `docs/nn-shrink-bilinear-channel-shuffle-decision.md` §3）。
//!
//! 並び: `[N, C, *S] → [N, g, C/g, S] → permute(0,2,1,3) → [N, C, *S]`、
//! すなわち `out[:, j*g + i] = in[:, i*(C/g) + j]`。入力は rank 3 以上。
//! `Var::channel_shuffle`（`var.rs`）と層 [`ChannelShuffle`]
//! （`compat::Sequential::add_channel_shuffle` の実体。facade からは
//! 再エクスポートしない）が [`channel_shuffle_forward`] を共有する。

use fandhe_ai_tensor_core::{BackendOps, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::var::Var;

/// 中間 shape `[N, g, C/g, S]` の構成要素 `(n, c, s)` を検査付きで求める。
fn checked_dims(in_shape: &[usize], groups: usize) -> Result<(usize, usize, usize), ShapeError> {
    let rank = in_shape.len();
    if rank < 3 {
        return Err(ShapeError::RankMismatch {
            expected: 3,
            actual: rank,
        });
    }
    let c = in_shape[1];
    if groups == 0 || !c.is_multiple_of(groups) {
        return Err(ShapeError::ShapeMismatch {
            lhs: vec![c],
            rhs: vec![groups],
        });
    }
    let mut s: usize = 1;
    for &d in &in_shape[2..] {
        s = s.checked_mul(d).ok_or(ShapeError::ElementCountOverflow)?;
    }
    Ok((in_shape[0], c, s))
}

/// 出力 shape（入力と同じ）を検査して返す。[`ChannelShuffle::forward`]／
/// [`ChannelShuffle::forward_host`]／[`channel_shuffle_forward`] で共有し、
/// 判定基準が食い違う迂回経路を作らない（`.claude/rules/security.md` A08）。
///
/// 検査順序: ①rank 3 未満は [`ShapeError::RankMismatch`] → ②`groups == 0`
/// または C が `groups` で割り切れないとき [`ShapeError::ShapeMismatch`] →
/// ③空間軸の積を `checked_mul` で計算（溢れは
/// [`ShapeError::ElementCountOverflow`]）→ ④出力 shape を返す。tape を
/// 操作する前に Err が確定するため孤児ノードを残さない。
pub(crate) fn channel_shuffle_out_shape(
    in_shape: &[usize],
    groups: usize,
) -> Result<Vec<usize>, ShapeError> {
    checked_dims(in_shape, groups)?;
    Ok(in_shape.to_vec())
}

/// `ChannelShuffle::forward` と `Var::channel_shuffle` の共有本体。
/// `Var` メソッドは `new()` を経由しないため `groups == 0` の拒否
/// （`InvalidArgument`）をここで行い、層経路と判定基準を一本化する。
pub(crate) fn channel_shuffle_forward<'t>(
    input: &Var<'t>,
    groups: usize,
) -> Result<Var<'t>, AutodiffError> {
    if groups == 0 {
        return Err(AutodiffError::InvalidArgument(
            "ChannelShuffle::new: groups must not be 0".to_string(),
        ));
    }
    let in_shape = input.shape();
    let out_shape = channel_shuffle_out_shape(&in_shape, groups).map_err(AutodiffError::Shape)?;
    let (n, c, s) = checked_dims(&in_shape, groups).map_err(AutodiffError::Shape)?;
    let mid_shape = [n, groups, c / groups, s];

    let x = input.contiguous()?;
    let x = x.reshape(&mid_shape)?;
    let x = x.permute(&[0, 2, 1, 3])?;
    let x = x.contiguous()?;
    x.reshape(&out_shape)
}

/// チャネルをグループ間で転置する層（`torch.nn.ChannelShuffle` 相当）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelShuffle {
    groups: usize,
}

impl ChannelShuffle {
    /// `groups` を指定して構築する。`0` は [`AutodiffError::InvalidArgument`]。
    /// rank・整除性は入力 shape に依存するため forward 時へ遅延させる
    /// （`PixelShuffle::new` と同じ遅延検査契約）。
    pub fn new(groups: usize) -> Result<Self, AutodiffError> {
        if groups == 0 {
            return Err(AutodiffError::InvalidArgument(
                "ChannelShuffle::new: groups must not be 0".to_string(),
            ));
        }
        Ok(Self { groups })
    }

    /// グループ数。
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// 出力 shape を検査してから `contiguous → reshape → permute →
    /// contiguous → reshape` の 5 手順を積む。
    pub fn forward<'t>(&self, input: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
        channel_shuffle_forward(input, self.groups)
    }

    /// `Module::forward_host` の本体。[`Self::forward`] と同じ検査・手順を
    /// tape 不要経路で再現する（純粋なコピーのため bit 単位で一致する）。
    pub fn forward_host(
        &self,
        _ops: &dyn BackendOps,
        input: &Tensor<f32>,
    ) -> Result<Tensor<f32>, AutodiffError> {
        let in_shape = input.shape();
        let out_shape =
            channel_shuffle_out_shape(in_shape, self.groups).map_err(AutodiffError::Shape)?;
        let (n, c, s) = checked_dims(in_shape, self.groups).map_err(AutodiffError::Shape)?;
        let mid_shape = [n, self.groups, c / self.groups, s];

        let x = input.contiguous();
        let x = x.reshape(&mid_shape).map_err(AutodiffError::Shape)?;
        let x = x.permute(&[0, 2, 1, 3]).map_err(AutodiffError::Shape)?;
        let x = x.contiguous();
        x.reshape(&out_shape).map_err(AutodiffError::Shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_shape_check_order() {
        // ① rank
        assert!(matches!(
            channel_shuffle_out_shape(&[4, 4], 2),
            Err(ShapeError::RankMismatch {
                expected: 3,
                actual: 2
            })
        ));
        // ② groups == 0 / 非整除
        assert!(matches!(
            channel_shuffle_out_shape(&[1, 4, 2], 0),
            Err(ShapeError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            channel_shuffle_out_shape(&[1, 5, 2], 2),
            Err(ShapeError::ShapeMismatch { .. })
        ));
        // ③ 空間軸の積の溢れ
        assert!(matches!(
            channel_shuffle_out_shape(&[1, 2, usize::MAX, 2], 2),
            Err(ShapeError::ElementCountOverflow)
        ));
        // ④ 正常（出力 shape は入力と同じ）
        assert_eq!(
            channel_shuffle_out_shape(&[2, 6, 3, 3], 3).unwrap(),
            vec![2, 6, 3, 3]
        );
    }

    #[test]
    fn new_rejects_zero() {
        assert!(ChannelShuffle::new(0).is_err());
        assert_eq!(ChannelShuffle::new(3).unwrap().groups(), 3);
    }
}
