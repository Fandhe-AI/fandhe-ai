//! `pad` の非定数モード（reflect／replicate／circular）のホスト参照カーネルの
//! **単一情報源**（イシュー #2642・親 #2625。実装記録は
//! `docs/autodiff-pad-modes-decision.md`）。
//!
//! # 役割と呼び出し元
//!
//! - `autodiff::pad_ops::pad_with_mode`（`BackendOps::pad_modes_forward` が
//!   `Unsupported` のときのホストフォールバック）・`autodiff::grad`
//!   （`Op::PadMode` の VJP）・`backend-cpu` の
//!   `CpuBackendOps::pad_modes_forward` がいずれも本モジュールの関数を直接呼ぶ。
//!   添字写像・蓄積契約をクレート間で複製せず、乖離を構造的に排除する
//!   （`cumulative.rs` と同じ「共有可能な層に一本化する」方式）。
//! - CUDA／Metal の専用カーネルは本イシューの対象外で、`BackendOps` 既定の
//!   `Unsupported` から autodiff 側がこのホスト実装へフォールバックする。
//! - 定数埋めの既存 `BackendOps::pad`／`ops_shape::pad_out_shape` は変更しない
//!   （出力 shape の算出は `pad_out_shape` を再利用する）。
//!
//! # 数値契約
//!
//! 3 モードはすべて「軸ごとの添字写像」で、forward は算術を含まない純粋な
//! コピー（`out[o] = x[map(o)]`）。値は入力要素そのもので **bit 完全一致**
//! （`-0.0`・NaN payload・`±inf` も保存）。backward は添字が重複する
//! scatter-add（`d_x[map(o)] += g[o]`）で、入力要素ごとに `f64` アキュムレータへ
//! 出力の行優先順で蓄積し、最後に 1 回だけ `f32` へ downcast する
//! （`.claude/rules/coding-rust.md` の勾配長軸縮約の規定）。蓄積順が固定の
//! ため結果は run-to-run で bit 一致する。matmul 系 FMA 契約には触れない。
//!
//! # 境界検査（REQ-8・OWASP A03）
//!
//! [`pad_modes_layout`] が rank 一致・出力要素数／バイト数のオーバーフロー・
//! モード別の pad 上限を確保より前に検査し、型付きエラー（[`PadModeError`]）で
//! 拒否する。カーネルはスライス長を再検査し、入力アクセスは `get` による検査付き
//! （`unsafe`／`get_unchecked` は使わない）。

use crate::device::BackendError;
use crate::error::ShapeError;
use crate::ops_shape::pad_out_shape;

/// 非定数パディングのモード（`F.pad` の `mode`。定数埋めは既存 `Var::pad`）。
///
/// 将来のモード追加を非破壊にするため `#[non_exhaustive]`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PadMode {
    /// 端の要素を折り返す（端要素自体は繰り返さない）。単一反射で済む
    /// `before < len` かつ `after < len` が必要。
    Reflect,
    /// 端の要素を繰り返す。軸長が 1 以上であればよい。
    Replicate,
    /// 反対側から巻き込む。`before <= len` かつ `after <= len` が必要。
    Circular,
}

/// 形状・引数検査の失敗（型付き）。`autodiff` は `AutodiffError` へ、
/// バックエンドは [`BackendError`] へ写像する（`StatReduceError` と同じ置き方）。
#[derive(Debug, Clone, PartialEq)]
pub enum PadModeError {
    /// rank 不一致・要素数／バイト数オーバーフロー・スライス長不一致等の形状違反。
    Shape(ShapeError),
    /// モード別の pad 上限違反（軸番号・軸長・pad 幅を文言に含む）。
    InvalidArgument(String),
}

impl std::fmt::Display for PadModeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PadModeError::Shape(e) => write!(f, "pad_modes: shape error: {e:?}"),
            PadModeError::InvalidArgument(msg) => write!(f, "pad_modes: invalid argument: {msg}"),
        }
    }
}

impl std::error::Error for PadModeError {}

impl From<ShapeError> for PadModeError {
    fn from(err: ShapeError) -> Self {
        PadModeError::Shape(err)
    }
}

impl From<PadModeError> for BackendError {
    fn from(err: PadModeError) -> Self {
        match err {
            PadModeError::Shape(e) => BackendError::ShapeMismatch(e),
            PadModeError::InvalidArgument(msg) => BackendError::InvalidArgument(msg),
        }
    }
}

/// 解決済みのパディングレイアウト（[`pad_modes_layout`] の戻り値）。
///
/// フィールドは非公開で、[`pad_modes_layout`] だけが生成する。生成時の検査で
/// 整合性（モード別 pad 上限・要素数の `checked` 演算）が保証され、公開カーネルが
/// 外部から改変された値で範囲外アクセスしないようにする（REQ-8）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PadModesLayout {
    mode: PadMode,
    pads: Vec<(usize, usize)>,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    in_strides: Vec<usize>,
    in_numel: usize,
    out_numel: usize,
}

impl PadModesLayout {
    /// 入力 shape。
    pub fn in_shape(&self) -> &[usize] {
        &self.in_shape
    }

    /// 出力 shape。
    pub fn out_shape(&self) -> &[usize] {
        &self.out_shape
    }

    /// 入力の総要素数。
    pub fn in_numel(&self) -> usize {
        self.in_numel
    }

    /// 出力の総要素数。
    pub fn out_numel(&self) -> usize {
        self.out_numel
    }

    /// パディングモード。
    pub fn mode(&self) -> PadMode {
        self.mode
    }
}

/// 出力座標 `o`（軸長 `len`・前側 pad `before`）に対応する入力座標。
///
/// 事前条件（[`pad_modes_layout`] が保証）: パディングのある軸ではモード別の
/// pad 上限を満たす。パディングのない軸では `o < len`。符号付きキャストや
/// wrapping 演算に頼らず `usize` の分岐だけで書く。
fn map_axis(mode: PadMode, len: usize, before: usize, o: usize) -> usize {
    if o < before {
        match mode {
            PadMode::Reflect => before - o,
            PadMode::Replicate => 0,
            PadMode::Circular => o + len - before,
        }
    } else {
        let p = o - before;
        if p < len {
            p
        } else {
            match mode {
                PadMode::Reflect => 2 * (len - 1) - p,
                PadMode::Replicate => len - 1,
                PadMode::Circular => p - len,
            }
        }
    }
}

fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// shape・`pads`（先頭軸から順の `(before, after)`。`Var::pad` と同じ並び）・
/// モードを検査して [`PadModesLayout`] を返す。
///
/// 出力 shape は [`pad_out_shape`]（rank 一致・加算オーバーフロー・バイト数上限）
/// を再利用する。パディングのある軸（`(0, 0)` 以外）ごとに次を検査し、違反は
/// [`PadModeError::InvalidArgument`]:
///
/// | モード | 事前条件 |
/// |---|---|
/// | `Reflect` | `before < len` かつ `after < len` |
/// | `Replicate` | `len >= 1` |
/// | `Circular` | `before <= len` かつ `after <= len` |
///
/// `(0, 0)` の軸には制約を課さない（軸長 0 を許し、出力は空になる）。
pub fn pad_modes_layout(
    shape: &[usize],
    pads: &[(usize, usize)],
    mode: PadMode,
) -> Result<PadModesLayout, PadModeError> {
    let out_shape = pad_out_shape(shape, pads)?;
    for (axis, (&len, &(before, after))) in shape.iter().zip(pads.iter()).enumerate() {
        if before == 0 && after == 0 {
            continue;
        }
        let ok = match mode {
            PadMode::Reflect => before < len && after < len,
            PadMode::Replicate => len >= 1,
            PadMode::Circular => before <= len && after <= len,
        };
        if !ok {
            return Err(PadModeError::InvalidArgument(format!(
                "pad_modes: {mode:?} の pad が軸長の上限を超える（軸 {axis}・軸長 {len}・pad ({before}, {after})）"
            )));
        }
    }
    let in_numel = checked_numel(shape)?;
    let out_numel = checked_numel(&out_shape)?;
    let mut in_strides = vec![1usize; shape.len()];
    for a in (0..shape.len().saturating_sub(1)).rev() {
        in_strides[a] = in_strides[a + 1]
            .checked_mul(shape[a + 1])
            .ok_or(ShapeError::ElementCountOverflow)?;
    }
    Ok(PadModesLayout {
        mode,
        pads: pads.to_vec(),
        in_shape: shape.to_vec(),
        out_shape,
        in_strides,
        in_numel,
        out_numel,
    })
}

/// 出力要素を行優先順に走査し、各出力要素に対応する入力の平坦添字を `f` へ渡す。
///
/// 最終軸以外をオドメータで回し、行ごとに入力側の基準オフセットを求めてから最終軸の
/// 写像を適用する。出力が空なら `f` は呼ばれない。
fn walk_sources<F>(layout: &PadModesLayout, mut f: F) -> Result<(), PadModeError>
where
    F: FnMut(usize) -> Result<(), PadModeError>,
{
    if layout.out_numel == 0 {
        return Ok(());
    }
    let rank = layout.out_shape.len();
    if rank == 0 {
        return f(0);
    }
    let last = rank - 1;
    let mut coord = vec![0usize; last];
    loop {
        let mut base = 0usize;
        for (a, &c) in coord.iter().enumerate() {
            let src = map_axis(layout.mode, layout.in_shape[a], layout.pads[a].0, c);
            base += src * layout.in_strides[a];
        }
        for o in 0..layout.out_shape[last] {
            let src = map_axis(layout.mode, layout.in_shape[last], layout.pads[last].0, o);
            f(base + src * layout.in_strides[last])?;
        }
        let mut a = last;
        loop {
            if a == 0 {
                return Ok(());
            }
            a -= 1;
            coord[a] += 1;
            if coord[a] < layout.out_shape[a] {
                break;
            }
            coord[a] = 0;
        }
    }
}

fn check_len(actual: usize, expected: usize) -> Result<(), PadModeError> {
    if actual == expected {
        Ok(())
    } else {
        Err(PadModeError::Shape(ShapeError::ElementCountMismatch {
            expected,
            actual,
        }))
    }
}

/// forward のホスト参照実装。`out[o] = x[map(o)]`（算術なし・bit 完全一致）。
///
/// `x` は連続配置・行優先で `layout.in_numel()` 個。出力は `layout.out_numel()` 個
/// （確保は `try_reserve_exact` で、失敗は [`ShapeError::ElementCountOverflow`]）。
pub fn pad_modes_host(x: &[f32], layout: &PadModesLayout) -> Result<Vec<f32>, PadModeError> {
    check_len(x.len(), layout.in_numel)?;
    let mut out: Vec<f32> = Vec::new();
    out.try_reserve_exact(layout.out_numel)
        .map_err(|_| PadModeError::Shape(ShapeError::ElementCountOverflow))?;
    walk_sources(layout, |src| {
        let v = x
            .get(src)
            .ok_or(PadModeError::Shape(ShapeError::ElementCountMismatch {
                expected: layout.in_numel,
                actual: x.len(),
            }))?;
        out.push(*v);
        Ok(())
    })?;
    Ok(out)
}

/// [`pad_modes_host`] の VJP（入力勾配）。`d_x[map(o)] += g[o]`。
///
/// 入力要素ごとに `f64` アキュムレータへ出力の行優先順で蓄積し、最後に 1 回だけ
/// `f32` へ downcast する（添字重複する scatter-add の決定的契約）。
/// `upstream` は `layout.out_numel()` 個。
pub fn pad_modes_vjp_host(
    upstream: &[f32],
    layout: &PadModesLayout,
) -> Result<Vec<f32>, PadModeError> {
    check_len(upstream.len(), layout.out_numel)?;
    let mut acc: Vec<f64> = Vec::new();
    acc.try_reserve_exact(layout.in_numel)
        .map_err(|_| PadModeError::Shape(ShapeError::ElementCountOverflow))?;
    acc.resize(layout.in_numel, 0.0);
    let mut k = 0usize;
    walk_sources(layout, |src| {
        let slot =
            acc.get_mut(src)
                .ok_or(PadModeError::Shape(ShapeError::ElementCountMismatch {
                    expected: layout.in_numel,
                    actual: src,
                }))?;
        let g = upstream
            .get(k)
            .ok_or(PadModeError::Shape(ShapeError::ElementCountMismatch {
                expected: layout.out_numel,
                actual: upstream.len(),
            }))?;
        *slot += f64::from(*g);
        k += 1;
        Ok(())
    })?;
    Ok(acc.into_iter().map(|v| v as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fwd(x: &[f32], shape: &[usize], pads: &[(usize, usize)], mode: PadMode) -> Vec<f32> {
        let layout = pad_modes_layout(shape, pads, mode).unwrap();
        pad_modes_host(x, &layout).unwrap()
    }

    const X: [f32; 4] = [1.0, 2.0, 3.0, 4.0];

    #[test]
    fn forward_1d_hand_values() {
        let p = [(2, 3)];
        assert_eq!(
            fwd(&X, &[4], &p, PadMode::Reflect),
            [3., 2., 1., 2., 3., 4., 3., 2., 1.]
        );
        assert_eq!(
            fwd(&X, &[4], &p, PadMode::Replicate),
            [1., 1., 1., 2., 3., 4., 4., 4., 4.]
        );
        assert_eq!(
            fwd(&X, &[4], &p, PadMode::Circular),
            [3., 4., 1., 2., 3., 4., 1., 2., 3.]
        );
    }

    #[test]
    fn forward_replicate_exceeds_axis_len() {
        assert_eq!(fwd(&[7.0], &[1], &[(3, 2)], PadMode::Replicate), [7.0; 6]);
    }

    #[test]
    fn forward_boundary_exact() {
        // reflect は len-1・circular は len がちょうど上限。
        assert_eq!(
            fwd(&X, &[4], &[(3, 3)], PadMode::Reflect),
            [4., 3., 2., 1., 2., 3., 4., 3., 2., 1.]
        );
        assert_eq!(
            fwd(&X, &[4], &[(4, 4)], PadMode::Circular),
            [1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.]
        );
    }

    #[test]
    fn forward_2d_axes_independent() {
        // [[1,2],[3,4]] を (1,0)×(0,1) circular。
        let x = [1., 2., 3., 4.];
        assert_eq!(
            fwd(&x, &[2, 2], &[(1, 0), (0, 1)], PadMode::Circular),
            [3., 4., 3., 1., 2., 1., 3., 4., 3.]
        );
    }

    #[test]
    fn forward_preserves_bits() {
        let x = [f32::NAN, -0.0, f32::INFINITY];
        let out = fwd(&x, &[3], &[(1, 1)], PadMode::Replicate);
        assert_eq!(out[0].to_bits(), f32::NAN.to_bits());
        assert_eq!(out[2].to_bits(), (-0.0f32).to_bits());
        assert_eq!(out[4], f32::INFINITY);
    }

    #[test]
    fn rank0_and_zero_pads_are_identity() {
        assert_eq!(fwd(&[5.0], &[], &[], PadMode::Reflect), [5.0]);
        assert_eq!(fwd(&X, &[4], &[(0, 0)], PadMode::Circular), X);
    }

    #[test]
    fn zero_len_unpadded_axis_gives_empty() {
        let layout = pad_modes_layout(&[0, 3], &[(0, 0), (1, 1)], PadMode::Reflect).unwrap();
        assert_eq!(layout.out_numel(), 0);
        assert!(pad_modes_host(&[], &layout).unwrap().is_empty());
        assert!(pad_modes_vjp_host(&[], &layout).unwrap().is_empty());
    }

    #[test]
    fn precondition_violations_are_typed_errors() {
        let cases = [
            (PadMode::Reflect, 4usize, (4usize, 0usize)),
            (PadMode::Reflect, 4, (0, 5)),
            (PadMode::Circular, 4, (5, 0)),
            (PadMode::Circular, 4, (0, 5)),
            (PadMode::Replicate, 0, (1, 0)),
            (PadMode::Reflect, 0, (0, 1)),
        ];
        for (mode, len, p) in cases {
            let err = pad_modes_layout(&[len], &[p], mode).unwrap_err();
            assert!(
                matches!(err, PadModeError::InvalidArgument(_)),
                "{mode:?} {len} {p:?}: {err:?}"
            );
        }
    }

    #[test]
    fn rank_mismatch_and_overflow_are_shape_errors() {
        let e = pad_modes_layout(&[3, 3], &[(1, 1)], PadMode::Replicate).unwrap_err();
        assert!(matches!(
            e,
            PadModeError::Shape(ShapeError::RankMismatch { .. })
        ));
        let e = pad_modes_layout(&[1], &[(usize::MAX, 1)], PadMode::Replicate).unwrap_err();
        assert!(matches!(
            e,
            PadModeError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn slice_length_mismatch_is_rejected() {
        let layout = pad_modes_layout(&[4], &[(1, 1)], PadMode::Reflect).unwrap();
        assert!(pad_modes_host(&X[..3], &layout).is_err());
        assert!(pad_modes_vjp_host(&[0.0; 5], &layout).is_err());
    }

    #[test]
    fn vjp_hand_values() {
        // replicate (2,3): 先頭要素は 3 回、末尾要素は 4 回受け取る。
        let layout = pad_modes_layout(&[4], &[(2, 3)], PadMode::Replicate).unwrap();
        let g = [1.0f32; 9];
        assert_eq!(pad_modes_vjp_host(&g, &layout).unwrap(), [3., 1., 1., 4.]);
        // reflect (2,3): 添字 0→(o=2,4,8 の鏡像) など。出力 [3,2,1,2,3,4,3,2,1]。
        let layout = pad_modes_layout(&[4], &[(2, 3)], PadMode::Reflect).unwrap();
        assert_eq!(pad_modes_vjp_host(&g, &layout).unwrap(), [2., 3., 3., 1.]);
        // circular (2,3): 出力 [3,4,1,2,3,4,1,2,3]。
        let layout = pad_modes_layout(&[4], &[(2, 3)], PadMode::Circular).unwrap();
        assert_eq!(pad_modes_vjp_host(&g, &layout).unwrap(), [2., 2., 3., 2.]);
    }

    #[test]
    fn vjp_uses_f64_accumulator() {
        // replicate で同一入力要素へ 1e8, 1, -1e8 が流れる。f32 蓄積なら 0 になる。
        let layout = pad_modes_layout(&[1], &[(1, 1)], PadMode::Replicate).unwrap();
        let out = pad_modes_vjp_host(&[1e8, 1.0, -1e8], &layout).unwrap();
        assert_eq!(out, [1.0]);
    }

    #[test]
    fn vjp_is_deterministic() {
        let layout = pad_modes_layout(&[3, 4], &[(2, 1), (3, 3)], PadMode::Reflect).unwrap();
        let g: Vec<f32> = (0..layout.out_numel()).map(|i| (i as f32).sin()).collect();
        let a = pad_modes_vjp_host(&g, &layout).unwrap();
        let b = pad_modes_vjp_host(&g, &layout).unwrap();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn error_display_and_backend_mapping() {
        let e = PadModeError::InvalidArgument("x".into());
        assert!(e.to_string().contains("invalid argument"));
        assert!(matches!(
            BackendError::from(e),
            BackendError::InvalidArgument(_)
        ));
        let e = PadModeError::Shape(ShapeError::ElementCountOverflow);
        assert!(e.to_string().contains("shape error"));
        assert!(matches!(
            BackendError::from(e),
            BackendError::ShapeMismatch(_)
        ));
    }
}
