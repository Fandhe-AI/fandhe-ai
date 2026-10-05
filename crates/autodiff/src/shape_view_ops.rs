//! `unbind`・`movedim`・`swapaxes`・`tensor_split`・`meshgrid`・`rot90` の 6 種
//! 形状演算（イシュー #2639・親 #2625「Phase 4」・ルート #2499）。
//!
//! **新規 `Op`・`BackendOps` メソッド・VJP はゼロ**: いずれも値のコピーか
//! view だけの演算のため、既存の `Var::transpose`（`Op::Transpose`）・
//! `Var::permute`（`Op::Permute`）・`Var::narrow`（`Op::Narrow`）・
//! `Var::reshape`（`Op::Reshape`）・`Var::broadcast_to`（`Op::BroadcastTo`）・
//! `Var::contiguous`（`Op::Contiguous`。`pub(crate)`）と
//! [`crate::rearrange_ops::flip`] の合成だけで構成する（#2143 の
//! `rearrange_ops`・#2144 の `matrix_ops` と同じ方式）。直近の兄弟
//! （#2631〜#2637）は `tensor-core` へ共有カーネルと `BackendOps::*` を足したが、
//! 本イシューはカーネルが不要で、公開済みクレート `fandhe-ai-tensor-core` の
//! trait 面を無用に広げないため採らない。`crates/tensor-core`・`crates/backend-*`
//! は変更しない。
//!
//! **バックエンド到達性（受入基準 2）**: 合成先のうち、バックエンドを呼び出して
//! `Unsupported` ならホスト参照実装へ落ちる経路は次のとおり。
//! - `swapaxes`／`movedim`: view のみ。バックエンド呼び出しなし。
//! - `tensor_split`／`unbind`: forward は `Op::Narrow` の view（`unbind` は必要時に
//!   `Op::Contiguous` のホストコピー）。backward は `Op::Narrow` の VJP が
//!   `BackendOps::concat` を呼ぶ（3 バックエンドとも override しておらず、常に
//!   ホスト参照実装へフォールバックする）。
//! - `meshgrid`: view のみ。backward は `reduce_to_shape` の縮約。
//! - `rot90`: `flip`（`Var::index_select` → `gather`）。`gather`／`scatter` は
//!   CPU・CUDA・Metal が実カーネルを持つため、実機ではフォールバックにならず
//!   GPU の gather が走る（`docs/autodiff-shape-view-ops-decision.md` §2）。
//!
//! **公開形（未承認・保留）**: facade（`fandhe_ai`）への公開形は未承認で、承認依頼は
//! #2677・公開自体は承認後の #2678。推奨案は決定記録 §7（推奨案の記録であり承認記録
//! ではない）。保留中は `ShapeViewOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
//! `crates/facade/tests/api_surface.rs` の否定ガードが facade への漏出を拒否する。
//!
//! **数値契約**: forward はコピーのみで算術を含まず、3 バックエンド間で構造的に
//! bit 完全一致する（`NaN` の payload も保存される）。勾配は各入力要素への寄与が
//! 1 つだけで、`meshgrid` のみ `reduce_to_shape` の合算になる。勾配の比較は REQ-2
//! の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で行い、tolerance は
//! 変更しない。FMA 契約・`f64` アキュムレータ契約（`.claude/rules/coding-rust.md`）の
//! 新たな対象はない。
//!
//! **PyTorch 2.14.0 との差分（実測は `tests/fixtures/shape-view-pytorch-reference/`）**:
//! - 軸は非負の `usize` のみ。負の軸・負の境界添字は型として表現できず非対応。
//! - `tensor_split` は分割数と境界添字列を別関数（[`tensor_split`]・
//!   [`tensor_split_indices`]）に分ける。テンソル引数形は非対応。
//! - `unbind` は view ではなくコピーを返しうる（先頭軸以外のスライスは
//!   `Var::reshape` の contiguous 制約のため `Op::Contiguous` を経由する）。値は
//!   bit 一致。
//! - `rot90` の `k mod 4 == 0` は clone ではなく入力そのものを返す。
//! - `swapaxes` は rank 0 入力を軸範囲外として拒否する（PyTorch は許容）。
//! - 出力本数の確保前検査（下記）により、巨大な `sections` や軸長は PyTorch と違い
//!   型付きエラーになる。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: 外部から渡る軸・
//! `source`／`destination`・`sections`・境界添字・`k`・`dims`・入力リストは、
//! tape へノードを積む前にすべて検査する（引数起因のエラーで孤児ノードを残さない）。
//! 加算・乗算は `checked_*`、`k` は `rem_euclid` で扱う。出力本数は
//! `checked_output_count` で確保前に検査する。`sections` はテンソルの大きさと無関係に
//! 巨大になりえ、`unbind` の軸長も stride 0 の broadcast view では実体なしに巨大に
//! なりうるため、`Vec` の capacity overflow panic と tape ノードの過大確保 abort を
//! 防ぐ目的である。この見積もりは簿記コスト（`Var` と `TapeNode` の本体サイズ）の
//! 下限で、ノードが持つヒープ分は数えない。

use fandhe_ai_tensor_core::ShapeError;

use crate::error::AutodiffError;
use crate::rearrange_ops::{checked_axis_len_as_i32, checked_index_alloc_len, flip};
use crate::tape::TapeNode;
use crate::var::Var;

/// [`meshgrid`] の座標生成規約（`torch.meshgrid` の `indexing`）。
///
/// 既定は `Ij`（行列添字。出力 shape は入力順の `(n_0, …, n_{N-1})`）。`Xy` は
/// 入力が 2 本以上のときだけ先頭 2 軸を入れ替える（直交座標）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MeshgridIndexing {
    /// 行列添字（`indexing="ij"`）。
    #[default]
    Ij,
    /// 直交座標（`indexing="xy"`）。入力 2 本以上で先頭 2 軸を入れ替える。
    Xy,
}

fn axis_out_of_range(axis: usize, rank: usize) -> AutodiffError {
    AutodiffError::Shape(ShapeError::AxisOutOfRange { axis, rank })
}

fn checked_axis_in_range(axis: usize, rank: usize) -> Result<(), AutodiffError> {
    if axis >= rank {
        return Err(axis_out_of_range(axis, rank));
    }
    Ok(())
}

/// 出力 `Var` の本数 `count` が必要とする簿記コストを確保前に検査する。
///
/// 見積もりは「`count` ×（`Var` 本体 + 1 本あたり最大 `nodes_per_output` 個の
/// `TapeNode` 本体）」を 4 バイト単位へ換算し、`rearrange_ops::checked_index_alloc_len`
/// （1 GiB 上限）へ渡す（`Var::pad` が同じヘルパーを f32 出力へ転用した先例と同じ。
/// 新しい定数は作らない）。超過は `ShapeError::ElementCountOverflow`。
fn checked_output_count(count: usize, nodes_per_output: usize) -> Result<(), AutodiffError> {
    let overflow = || AutodiffError::Shape(ShapeError::ElementCountOverflow);
    let per_output = nodes_per_output
        .checked_mul(std::mem::size_of::<TapeNode>())
        .and_then(|b| b.checked_add(std::mem::size_of::<Var<'static>>()))
        .ok_or_else(overflow)?;
    let bytes = count.checked_mul(per_output).ok_or_else(overflow)?;
    checked_index_alloc_len(bytes.div_ceil(std::mem::size_of::<i32>()))
}

/// 出力 `Vec` を確保失敗で abort させず `try_reserve_exact` で確保する。
fn reserve_outputs<'t>(count: usize) -> Result<Vec<Var<'t>>, AutodiffError> {
    let mut out: Vec<Var<'t>> = Vec::new();
    out.try_reserve_exact(count)
        .map_err(|_| AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    Ok(out)
}

/// 2 軸を入れ替える view（`torch.swapaxes` 相当）。`Var::transpose` への委譲。
///
/// `axis0 == axis1` は恒等 view。範囲外は `ShapeError::AxisOutOfRange`。
pub fn swapaxes<'t>(x: &Var<'t>, axis0: usize, axis1: usize) -> Result<Var<'t>, AutodiffError> {
    x.transpose(axis0, axis1)
}

/// `source` の各軸を `destination` の位置へ移す view（`torch.movedim` 相当）。
///
/// 長さ不一致は `AutodiffError::InvalidArgument`、範囲外は `AxisOutOfRange`、
/// `source` 内または `destination` 内の重複は `DuplicateAxis`。順列は
/// `perm[destination[i]] = source[i]` とし、残りの位置へ残りの入力軸を元の順で
/// 埋めて `Var::permute` へ渡す。恒等順列（空指定を含む）はノードを積まず `*x`
/// を返す（`flip` の空 `dims` と同じ）。
pub fn movedim<'t>(
    x: &Var<'t>,
    source: &[usize],
    destination: &[usize],
) -> Result<Var<'t>, AutodiffError> {
    if source.len() != destination.len() {
        return Err(AutodiffError::InvalidArgument(format!(
            "shape_view_ops::movedim: source と destination は同じ長さが必要\
             （source.len()={}, destination.len()={}）",
            source.len(),
            destination.len()
        )));
    }
    let rank = x.shape().len();
    let mut src_seen = vec![false; rank];
    let mut dst_seen = vec![false; rank];
    for (&s, &d) in source.iter().zip(destination) {
        checked_axis_in_range(s, rank)?;
        checked_axis_in_range(d, rank)?;
        if src_seen[s] {
            return Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: s }));
        }
        if dst_seen[d] {
            return Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: d }));
        }
        src_seen[s] = true;
        dst_seen[d] = true;
    }
    // `usize::MAX` は「未割り当て」の目印（実在の軸番号は rank 未満）。
    let mut perm = vec![usize::MAX; rank];
    for (&s, &d) in source.iter().zip(destination) {
        perm[d] = s;
    }
    let mut rest = (0..rank).filter(|a| !src_seen[*a]);
    for slot in perm.iter_mut().filter(|p| **p == usize::MAX) {
        // 割り当て済みの軸数と未割り当ての位置数は一致するため `None` にならない。
        *slot = rest.next().ok_or_else(|| {
            AutodiffError::InvalidArgument(
                "shape_view_ops::movedim: 内部不変条件違反（残軸の不足）".to_string(),
            )
        })?;
    }
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return Ok(*x);
    }
    x.permute(&perm)
}

/// 軸 `dim` を `sections` 個へ均等に分ける（`torch.tensor_split(x, sections, dim)`）。
///
/// 常にちょうど `sections` 本を返す（`chunk` の `div_ceil` 方式とは異なる）。
/// `n = shape[dim]` として先頭 `n % sections` 本が長さ `n / sections + 1`、残りが
/// `n / sections`。`sections > n` のときは空片を含む。rank 0 と `dim >= rank` は
/// `AxisOutOfRange`、`sections == 0` は `InvalidArgument`。出力本数は確保前に検査する。
pub fn tensor_split<'t>(
    x: &Var<'t>,
    sections: usize,
    dim: usize,
) -> Result<Vec<Var<'t>>, AutodiffError> {
    let shape = x.shape();
    checked_axis_in_range(dim, shape.len())?;
    if sections == 0 {
        return Err(AutodiffError::InvalidArgument(
            "shape_view_ops::tensor_split: sections は 1 以上が必要".to_string(),
        ));
    }
    checked_output_count(sections, 1)?;
    let n = shape[dim];
    let base = n / sections;
    let extra = n % sections;
    let mut out = reserve_outputs(sections)?;
    let mut start = 0usize;
    for i in 0..sections {
        let len = base + usize::from(i < extra);
        out.push(x.narrow(dim, start, len)?);
        start += len;
    }
    Ok(out)
}

/// 軸 `dim` を境界添字列 `indices` で分ける（`torch.tensor_split(x, indices, dim)`）。
///
/// 切り出すのは `x[:i0]`・`x[i0:i1]`・…・`x[ik:]` の `indices.len() + 1` 本。各片は
/// `start = min(直前の境界, n)`・`end = min(境界, n)` とし、`end < start`（非単調な列）
/// は位置 `start` の長さ 0 の片になる。範囲外の値や非単調な列はエラーにしない。
/// `indices` が空なら全体の 1 片。rank 0 と `dim >= rank` は `AxisOutOfRange`。
pub fn tensor_split_indices<'t>(
    x: &Var<'t>,
    indices: &[usize],
    dim: usize,
) -> Result<Vec<Var<'t>>, AutodiffError> {
    let shape = x.shape();
    checked_axis_in_range(dim, shape.len())?;
    let count = indices
        .len()
        .checked_add(1)
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    checked_output_count(count, 1)?;
    let n = shape[dim];
    let mut out = reserve_outputs(count)?;
    let mut prev = 0usize;
    for piece in 0..count {
        let boundary = indices.get(piece).copied().unwrap_or(n);
        let start = prev.min(n);
        let end = boundary.min(n);
        out.push(x.narrow(dim, start, end.saturating_sub(start))?);
        prev = boundary;
    }
    Ok(out)
}

/// 軸 `dim` を取り除きながら長さ 1 ずつに分解する（`torch.unbind` 相当）。
///
/// `shape[dim]` 本（0 なら空の `Vec`）を返し、各片は軸 `dim` を除いた shape を持つ。
/// 各片は `narrow(dim, i, 1)` → 必要時 `Var::contiguous` → `reshape`（`Var::reshape` は
/// 非 contiguous を拒否するため）の合成で、先頭軸以外ではスライスごとにコピーする
/// （PyTorch は view を返す差分。値は bit 一致）。rank 0 と `dim >= rank` は
/// `AxisOutOfRange`。出力本数は確保前に検査する。
pub fn unbind<'t>(x: &Var<'t>, dim: usize) -> Result<Vec<Var<'t>>, AutodiffError> {
    let shape = x.shape();
    checked_axis_in_range(dim, shape.len())?;
    let n = shape[dim];
    let mut out_shape = shape;
    out_shape.remove(dim);
    // narrow・contiguous・reshape の最大 3 ノード。
    checked_output_count(n, 3)?;
    let mut out = reserve_outputs(n)?;
    // 全スライスをコピーした場合の総要素数（stride 0 の broadcast view では実体より
    // 巨大になりうる）。コピーが必要になった時点で 1 GiB 上限を確保前に検査する。
    let total_elems = out_shape
        .iter()
        .try_fold(n, |acc, &d| acc.checked_mul(d))
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    for i in 0..n {
        let view = x.narrow(dim, i, 1)?;
        let slice = match view.reshape(&out_shape) {
            Ok(v) => v,
            Err(AutodiffError::Shape(ShapeError::NonContiguousReshape)) => {
                checked_index_alloc_len(total_elems)?;
                view.contiguous()?.reshape(&out_shape)?
            }
            Err(e) => return Err(e),
        };
        out.push(slice);
    }
    Ok(out)
}

/// 1 次元（または 0 次元）の座標入力から N 次元格子を作る（`torch.meshgrid` 相当）。
///
/// 入力が空のときと rank 2 以上の入力があるときは `InvalidArgument`、異なる
/// `Tape` の入力は `TapeMismatch`。0 次元入力は長さ 1 として扱う。`Ij` の出力 shape は
/// `(n_0, …, n_{N-1})`、`Xy` は `N >= 2` のときだけ先頭 2 軸を入れ替える。各入力を
/// 変化軸だけ `n_k`・他が 1 の rank N へ `reshape` し、全体 shape へ `broadcast_to`
/// する（stride 0 の view）。全体 shape の要素数積は tape 操作の前に `checked_mul` で
/// 検査する（`ElementCountOverflow`）。非 contiguous な入力のコピーが必要なときだけ、
/// その直前に `checked_index_alloc_len` をかける（その検査の失敗時、先行入力で積んだ
/// ノードは tape に残るが結果へは影響しない）。shape が一致していて変換が不要な入力はノードを積まない。
pub fn meshgrid<'t>(
    tensors: &[Var<'t>],
    indexing: MeshgridIndexing,
) -> Result<Vec<Var<'t>>, AutodiffError> {
    let Some(first) = tensors.first() else {
        return Err(AutodiffError::InvalidArgument(
            "shape_view_ops::meshgrid: 入力が 1 個以上必要".to_string(),
        ));
    };
    let n_inputs = tensors.len();
    let mut lens: Vec<usize> = Vec::new();
    lens.try_reserve_exact(n_inputs)
        .map_err(|_| AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    for t in tensors {
        first.check_same_tape(t)?;
        let s = t.shape();
        if s.len() > 1 {
            return Err(AutodiffError::InvalidArgument(format!(
                "shape_view_ops::meshgrid: 入力は 0 次元または 1 次元のみ（rank={}）",
                s.len()
            )));
        }
        lens.push(s.first().copied().unwrap_or(1));
    }
    let swap = indexing == MeshgridIndexing::Xy && n_inputs >= 2;
    let axis_of = |k: usize| -> usize {
        match (swap, k) {
            (true, 0) => 1,
            (true, 1) => 0,
            _ => k,
        }
    };
    let mut full = lens.clone();
    if swap {
        full.swap(0, 1);
    }
    full.iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
    // reshape・broadcast_to に非 contiguous 入力の contiguous を加えた最大 3 ノード。
    checked_output_count(n_inputs, 3)?;

    let mut out = reserve_outputs(n_inputs)?;
    for (k, t) in tensors.iter().enumerate() {
        let n_k = lens[k];
        let mut expanded = vec![1usize; n_inputs];
        expanded[axis_of(k)] = n_k;
        let shaped = if t.shape() == expanded {
            *t
        } else {
            match t.reshape(&expanded) {
                Ok(v) => v,
                Err(AutodiffError::Shape(ShapeError::NonContiguousReshape)) => {
                    checked_index_alloc_len(n_k)?;
                    t.contiguous()?.reshape(&expanded)?
                }
                Err(e) => return Err(e),
            }
        };
        out.push(if expanded == full {
            shaped
        } else {
            shaped.broadcast_to(&full)?
        });
    }
    Ok(out)
}

/// `dims` が張る平面を 90 度 `k` 回回転する（`torch.rot90` 相当）。
///
/// `k.rem_euclid(4)` で分岐する: 0 はノードを積まず `*x`（PyTorch は clone する差分）、
/// 1 は `flip(dims[1])` → `transpose`、2 は `flip(dims[0], dims[1])`、3 は
/// `flip(dims[0])` → `transpose`。`dims[0] == dims[1]` は `DuplicateAxis`、範囲外
/// （rank 2 未満を含む）は `AxisOutOfRange`。`flip` の内部検査を反転する全軸について
/// `flip` 呼び出し前に先行実行し、2 軸反転の途中で失敗して孤児ノードを残さない。
pub fn rot90<'t>(x: &Var<'t>, k: isize, dims: [usize; 2]) -> Result<Var<'t>, AutodiffError> {
    let shape = x.shape();
    let rank = shape.len();
    checked_axis_in_range(dims[0], rank)?;
    checked_axis_in_range(dims[1], rank)?;
    if dims[0] == dims[1] {
        return Err(AutodiffError::Shape(ShapeError::DuplicateAxis {
            axis: dims[0],
        }));
    }
    let k4 = k.rem_euclid(4);
    let flips: &[usize] = match k4 {
        0 => return Ok(*x),
        1 => &[dims[1]],
        2 => &dims,
        _ => &[dims[0]],
    };
    for &d in flips {
        checked_axis_len_as_i32(shape[d])?;
        checked_index_alloc_len(shape[d])?;
    }
    let flipped = flip(x, flips)?;
    if k4 == 2 {
        Ok(flipped)
    } else {
        flipped.transpose(dims[0], dims[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::Tensor;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    fn seq(shape: &[usize]) -> Tensor<f32> {
        let n: usize = shape.iter().product();
        t((0..n).map(|i| i as f32).collect(), shape)
    }

    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    fn dims_of(v: &Var<'_>) -> Vec<usize> {
        v.to_tensor().shape().to_vec()
    }

    #[test]
    fn swapaxes_swaps_and_identity() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[2, 3]));
        let y = swapaxes(&x, 0, 1).unwrap();
        assert_eq!(dims_of(&y), vec![3, 2]);
        assert_eq!(vals(&y), vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
        let z = swapaxes(&x, 1, 1).unwrap();
        assert_eq!(dims_of(&z), vec![2, 3]);
        assert!(matches!(
            swapaxes(&x, 0, 2),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: 2,
                rank: 2
            }))
        ));
    }

    #[test]
    fn movedim_known_values_and_identity_pushes_no_node() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[2, 3, 4]));
        let y = movedim(&x, &[0], &[2]).unwrap();
        assert_eq!(dims_of(&y), vec![3, 4, 2]);
        let before = tape.len();
        let same = movedim(&x, &[1], &[1]).unwrap();
        let empty = movedim(&x, &[], &[]).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(vals(&same), vals(&x));
        assert_eq!(vals(&empty), vals(&x));
        let z = movedim(&x, &[2, 0], &[0, 1]).unwrap();
        assert_eq!(dims_of(&z), vec![4, 2, 3]);
    }

    #[test]
    fn movedim_errors_leave_no_node() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[2, 3, 4]));
        let before = tape.len();
        assert!(matches!(
            movedim(&x, &[0, 1], &[0]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            movedim(&x, &[0, 0], &[1, 2]),
            Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: 0 }))
        ));
        assert!(matches!(
            movedim(&x, &[0, 1], &[2, 2]),
            Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: 2 }))
        ));
        assert!(matches!(
            movedim(&x, &[3], &[0]),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn tensor_split_sections_distribution() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[7]));
        let parts = tensor_split(&x, 3, 0).unwrap();
        let lens: Vec<usize> = parts.iter().map(|p| dims_of(p)[0]).collect();
        assert_eq!(lens, vec![3, 2, 2]);
        assert_eq!(vals(&parts[1]), vec![3.0, 4.0]);
        let many = tensor_split(&x, 9, 0).unwrap();
        let lens: Vec<usize> = many.iter().map(|p| dims_of(p)[0]).collect();
        assert_eq!(lens, vec![1, 1, 1, 1, 1, 1, 1, 0, 0]);
    }

    #[test]
    fn tensor_split_errors_and_huge_sections_rejected() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[4]));
        let before = tape.len();
        assert!(matches!(
            tensor_split(&x, 0, 0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            tensor_split(&x, 2, 1),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
        // 巨大な sections は確保前に拒否される（panic・abort しない）。
        assert!(matches!(
            tensor_split(&x, usize::MAX, 0),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert!(matches!(
            tensor_split(&x, 1 << 40, 0),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert_eq!(tape.len(), before);
        let s = tape.var(&Tensor::scalar(1.0));
        assert!(matches!(
            tensor_split(&s, 1, 0),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: 0,
                rank: 0
            }))
        ));
    }

    #[test]
    fn tensor_split_indices_clamps_like_torch() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[10]));
        let lens = |idx: &[usize]| -> Vec<usize> {
            tensor_split_indices(&x, idx, 0)
                .unwrap()
                .iter()
                .map(|p| dims_of(p)[0])
                .collect()
        };
        assert_eq!(lens(&[]), vec![10]);
        assert_eq!(lens(&[5, 3]), vec![5, 0, 7]);
        assert_eq!(lens(&[100, 2]), vec![10, 0, 8]);
        assert_eq!(lens(&[7, 3, 9]), vec![7, 0, 6, 1]);
        assert_eq!(lens(&[12]), vec![10, 0]);
        let parts = tensor_split_indices(&x, &[5, 3], 0).unwrap();
        assert_eq!(
            vals(&parts[2]),
            (3..10).map(|i| i as f32).collect::<Vec<_>>()
        );
        assert!(matches!(
            tensor_split_indices(&x, &[1], 1),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
    }

    #[test]
    fn unbind_values_non_contiguous_and_empty() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[2, 3, 4]));
        let parts = unbind(&x, 1).unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(dims_of(&parts[1]), vec![2, 4]);
        assert_eq!(
            vals(&parts[1]),
            vec![4.0, 5.0, 6.0, 7.0, 16.0, 17.0, 18.0, 19.0]
        );
        let xt = x.transpose(0, 2).unwrap();
        let parts_t = unbind(&xt, 0).unwrap();
        assert_eq!(parts_t.len(), 4);
        assert_eq!(dims_of(&parts_t[0]), vec![3, 2]);
        let e = tape.var(&t(vec![], &[0, 3]));
        assert!(unbind(&e, 0).unwrap().is_empty());
        let s = tape.var(&Tensor::scalar(1.0));
        assert!(matches!(
            unbind(&s, 0),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
    }

    #[test]
    fn unbind_huge_broadcast_axis_is_rejected_before_allocation() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[1]));
        let big = x.broadcast_to(&[1 << 40]).unwrap();
        let before = tape.len();
        assert!(matches!(
            unbind(&big, 0),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn unbind_huge_slice_copy_is_rejected_before_allocation() {
        // stride 0 の broadcast view は実体なしに巨大なスライスを持てる。
        // 先頭軸以外の unbind は contiguous コピーが要るため確保前に弾く。
        let tape = Tape::new();
        let x = tape.var(&seq(&[2]));
        let big = x.broadcast_to(&[1 << 31, 2]).unwrap();
        assert!(matches!(
            unbind(&big, 1),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }

    #[test]
    fn meshgrid_ij_xy_shapes_and_values() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let b = tape.var(&t(vec![10.0, 20.0], &[2]));
        let ij = meshgrid(&[a, b], MeshgridIndexing::Ij).unwrap();
        assert_eq!(dims_of(&ij[0]), vec![3, 2]);
        assert_eq!(vals(&ij[0]), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        assert_eq!(vals(&ij[1]), vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0]);
        let xy = meshgrid(&[a, b], MeshgridIndexing::Xy).unwrap();
        assert_eq!(dims_of(&xy[0]), vec![2, 3]);
        assert_eq!(vals(&xy[0]), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(vals(&xy[1]), vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]);
        let one = meshgrid(&[a], MeshgridIndexing::Xy).unwrap();
        assert_eq!(dims_of(&one[0]), vec![3]);
    }

    #[test]
    fn meshgrid_scalar_non_contiguous_and_errors() {
        let tape = Tape::new();
        let s = tape.var(&Tensor::scalar(7.0));
        let b = tape.var(&t(vec![1.0, 2.0], &[2]));
        let g = meshgrid(&[s, b], MeshgridIndexing::Ij).unwrap();
        assert_eq!(dims_of(&g[0]), vec![1, 2]);
        // 非 contiguous な 1 次元入力（stride 0 の broadcast view）。
        let bv = tape.var(&Tensor::scalar(5.0)).reshape(&[1]).unwrap();
        let wide = bv.broadcast_to(&[3]).unwrap();
        let g2 = meshgrid(&[wide, b], MeshgridIndexing::Ij).unwrap();
        assert_eq!(vals(&g2[0]), vec![5.0; 6]);
        let m = tape.var(&seq(&[2, 2]));
        let before = tape.len();
        assert!(matches!(
            meshgrid(&[], MeshgridIndexing::Ij),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            meshgrid(&[m, b], MeshgridIndexing::Ij),
            Err(AutodiffError::InvalidArgument(_))
        ));
        let other = Tape::new();
        let o = other.var(&t(vec![1.0], &[1]));
        assert!(matches!(
            meshgrid(&[b, o], MeshgridIndexing::Ij),
            Err(AutodiffError::TapeMismatch)
        ));
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn meshgrid_overflowing_shape_is_rejected() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[1]));
        let a = x.broadcast_to(&[1 << 33]).unwrap();
        let b = x.broadcast_to(&[1 << 33]).unwrap();
        let c = x.broadcast_to(&[1 << 33]).unwrap();
        let before = tape.len();
        assert!(matches!(
            meshgrid(&[a, b, c], MeshgridIndexing::Ij),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn rot90_known_values_and_k_modulo() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        // [[1,2,3],[4,5,6]] を反時計回りに 90 度: [[3,6],[2,5],[1,4]]。
        let r1 = rot90(&x, 1, [0, 1]).unwrap();
        assert_eq!(dims_of(&r1), vec![3, 2]);
        assert_eq!(vals(&r1), vec![3.0, 6.0, 2.0, 5.0, 1.0, 4.0]);
        let r2 = rot90(&x, 2, [0, 1]).unwrap();
        assert_eq!(vals(&r2), vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let r3 = rot90(&x, 3, [0, 1]).unwrap();
        assert_eq!(vals(&r3), vec![4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);
        let rm1 = rot90(&x, -1, [0, 1]).unwrap();
        assert_eq!(vals(&rm1), vals(&r3));
        let before = tape.len();
        let r4 = rot90(&x, 4, [0, 1]).unwrap();
        assert_eq!(tape.len(), before);
        assert_eq!(vals(&r4), vals(&x));
        let rmin = rot90(&x, isize::MIN, [0, 1]).unwrap();
        assert_eq!(vals(&rmin), vals(&x));
    }

    #[test]
    fn rot90_errors_leave_no_node() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[2, 3]));
        let v = tape.var(&seq(&[3]));
        let before = tape.len();
        assert!(matches!(
            rot90(&x, 1, [0, 0]),
            Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: 0 }))
        ));
        assert!(matches!(
            rot90(&x, 1, [0, 2]),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
        assert!(matches!(
            rot90(&v, 1, [0, 1]),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn rot90_two_axis_flip_precheck_prevents_orphan_nodes() {
        // 反転軸の長さが確保上限を超える場合、flip を呼ぶ前に拒否される。
        let tape = Tape::new();
        let x = tape.var(&seq(&[1, 1]));
        let wide = x.broadcast_to(&[1, 1 << 29]).unwrap();
        let before = tape.len();
        assert!(matches!(
            rot90(&wide, 2, [0, 1]),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn backward_through_compositions_is_shape_correct() {
        let tape = Tape::new();
        let x = tape.var(&seq(&[2, 3]));
        let parts = unbind(&x, 1).unwrap();
        let mut loss = parts[0].sum(None).unwrap();
        for p in &parts[1..] {
            loss = loss.add(&p.sum(None).unwrap()).unwrap();
        }
        let grads = tape.backward(&loss).unwrap();
        let g = grads.get(&x).unwrap().expect("勾配あり").clone();
        assert_eq!(g.host_slice().into_owned(), vec![1.0; 6]);
    }
}
