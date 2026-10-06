//! `kron`・`tensordot`（`tensordot_axes`）・`cdist`・`cross` の 4 演算
//! （イシュー #2640・親 #2625「Phase 4」・ルート #2499）。
//!
//! **新規 `Op`・`BackendOps` メソッド・VJP はゼロ**: いずれも既存の
//! `Var::permute`（`Op::Permute`）・`Var::reshape`（`Op::Reshape`）・`Var::contiguous`
//! （`Op::Contiguous`。`pub(crate)`）・`Var::unsqueeze`・`Var::mul`／`Var::sub`
//! （broadcast 付き）・`Var::matmul`・[`crate::reduce_ops::norm_p`]・
//! [`crate::rearrange_ops::roll`] の合成だけで構成する（#2639 の `shape_view_ops`・
//! #2143 の `rearrange_ops`・#2144 の `matrix_ops` と同じ方式）。公開済みクレート
//! `fandhe-ai-tensor-core` の trait 面を無用に広げないため `crates/tensor-core`・
//! `crates/backend-*` は変更しない。
//!
//! **合成表**
//! - `kron`: rank を揃え（短い方の先頭へ 1 を補う）、`a` を `[a0,1,a1,1,…]`・`b` を
//!   `[1,b0,1,b1,…]` へ `reshape` して `Var::mul`（broadcast）し、`[a0·b0, a1·b1, …]`
//!   へ `reshape` する。
//! - `tensordot_axes`／`tensordot`: `a` を `permute(自由軸 ++ 縮約軸)`、`b` を
//!   `permute(縮約軸 ++ 自由軸)` して `[L, K]`・`[K, R]` へ `reshape` し、`Var::matmul`
//!   の後に自由軸の shape へ `reshape` する（縮約なしは `K = 1`）。恒等な `permute`／
//!   `reshape` はノードを積まないため、rank 2 同士の `tensordot(a, b, 1)` は
//!   `a.matmul(&b)` と tape ノード 1 個・bit 同一になる。
//! - `cdist`: `x1.unsqueeze` と `x2.unsqueeze` の差（`Var::sub`。batch 軸も broadcast）
//!   を `norm_p(…, 最終軸)` で縮約する。`torch.cdist(compute_mode=
//!   "donot_use_mm_for_euclid_dist")` 相当の総当たり形。
//! - `cross`: `c_i = a_{i+1} b_{i+2} − a_{i+2} b_{i+1}` を `roll` 4 回・`mul` 2 回・
//!   `sub` 1 回で組む。
//!
//! **バックエンド到達性（受入基準 2）**
//! - `kron`: forward が呼ぶ `mul` は `BackendOps` の必須メソッドで、3 バックエンドとも
//!   実カーネルを持つ。`Unsupported` からホスト参照実装へ落ちる経路は存在しない。
//! - `tensordot`: forward の `gemm`（backward は `gemm_fp32_strict`）も必須メソッドで
//!   同様。
//! - `cdist`: 減算（`scalar_binary`）と `vector_norm`／`vector_norm_p`。後者の override
//!   は CPU のみで、CUDA／Metal の実機では「減算は GPU・ノルムはホスト参照実装」の
//!   混在になる（`Unsupported` はホスト参照実装へ到達する）。
//! - `cross`: `gather`（`roll`）・`mul`・`scalar_binary`、backward で `scatter`。3
//!   バックエンドとも実カーネルを持つため、実機では GPU が走る（#2639 の `rot90` と
//!   同じ）。
//!
//! **公開形（未承認・保留）**: facade（`fandhe_ai`）への公開形は未承認で、承認依頼は
//! #2677・公開自体は承認後の #2678。推奨案は `docs/autodiff-tensor-product-ops-decision.md`
//! §7（推奨案の記録であり承認記録ではない）。保留中は `TensorProductOpsHoldDoctestGuard`
//! （`crates/facade/src/lib.rs`）と `crates/facade/tests/api_surface.rs` の否定ガードが
//! facade への漏出を拒否する。
//!
//! **数値契約**: `kron` の forward は乗算 1 回で 3 バックエンド間 bit 一致する。
//! `tensordot` は `Var::matmul` と同一の FMA 契約（CUDA TF32 opt-in の挙動も同じ）で、
//! 新しい丸め経路を作らない。`cdist` は `norm_p` の `f64` アキュムレータ契約を継承する。
//! `cross` は乗算 2 回・減算 1 回のみで縮約を持たない。PyTorch との比較は REQ-2 の統一
//! 複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で行い、tolerance は変更しない。
//!
//! **PyTorch 2.14.0 との差分（実測は `tests/fixtures/tensor-product-pytorch-reference/`）**:
//! - 軸は非負の `usize` のみ。
//! - `cdist` の `p` は有限かつ正のみ（`0`・`inf`・負・`NaN` は `InvalidArgument`）。
//!   `compute_mode` 引数は設けず、常に差分形の総当たり。メモリは中間
//!   `[…,P,R,M]` を確保する O(P·R·M)。
//! - `cross` は同形状のみ（broadcast 非対応）・`dim` 明示必須。
//! - `tensordot` の縮約軸長は完全一致のみ。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03・A04）**: 外部から渡る軸・
//! 軸リスト・`n`・`p`・shape は、tape へノードを積む前にすべて検査する（引数起因の
//! エラーで孤児ノードを残さない）。積は `checked_mul`、確保は `checked_bytes_for` で
//! 事前検査し、`broadcast_to` 由来で実体のない巨大 shape による capacity overflow
//! panic を防ぐ。

use fandhe_ai_tensor_core::{ShapeError, broadcast_shape};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::rearrange_ops::roll;
use crate::reduce_ops::norm_p;
use crate::var::Var;

fn overflow() -> AutodiffError {
    AutodiffError::Shape(ShapeError::ElementCountOverflow)
}

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::InvalidArgument(msg)
}

/// shape の要素数積を `checked_mul` で求める。
fn checked_numel(shape: &[usize]) -> Result<usize, AutodiffError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(overflow)
}

/// 恒等でなければ `Var::permute`、恒等ならノードを積まず `*x` を返す。
fn permute_if_needed<'t>(x: &Var<'t>, perm: &[usize]) -> Result<Var<'t>, AutodiffError> {
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return Ok(*x);
    }
    x.permute(perm)
}

/// contiguous 化してから `reshape` する。shape が同じなら `reshape` を呼ばない。
fn reshape_contiguous<'t>(x: &Var<'t>, shape: &[usize]) -> Result<Var<'t>, AutodiffError> {
    let c = x.contiguous()?;
    if c.shape() == shape {
        return Ok(c);
    }
    c.reshape(shape)
}

/// クロネッカー積（`torch.kron` 相当）。
///
/// rank が異なるときは短い方の shape の先頭へ 1 を補う。出力 shape は軸ごとの積
/// `[a0·b0, a1·b1, …]`。軸ごとの積と各中間 shape の確保量は tape 操作前に検査する
/// （`ElementCountOverflow`）。異なる `Tape` の入力は `TapeMismatch`。
pub fn kron<'t>(a: &Var<'t>, b: &Var<'t>) -> Result<Var<'t>, AutodiffError> {
    a.check_same_tape(b)?;
    let a_shape = a.shape();
    let b_shape = b.shape();
    let rank = a_shape.len().max(b_shape.len());
    let pad = |s: &[usize]| -> Vec<usize> {
        let mut v = vec![1usize; rank - s.len()];
        v.extend_from_slice(s);
        v
    };
    let a_p = pad(&a_shape);
    let b_p = pad(&b_shape);
    let mut a_view = Vec::with_capacity(rank * 2);
    let mut b_view = Vec::with_capacity(rank * 2);
    let mut mid = Vec::with_capacity(rank * 2);
    let mut out_shape = Vec::with_capacity(rank);
    for i in 0..rank {
        a_view.extend_from_slice(&[a_p[i], 1]);
        b_view.extend_from_slice(&[1, b_p[i]]);
        mid.extend_from_slice(&[a_p[i], b_p[i]]);
        out_shape.push(a_p[i].checked_mul(b_p[i]).ok_or_else(overflow)?);
    }
    checked_bytes_for::<f32>(&a_shape)?;
    checked_bytes_for::<f32>(&b_shape)?;
    checked_bytes_for::<f32>(&mid)?;
    checked_bytes_for::<f32>(&out_shape)?;
    let a_r = reshape_contiguous(a, &a_view)?;
    let b_r = reshape_contiguous(b, &b_view)?;
    let prod = a_r.mul(&b_r)?;
    reshape_contiguous(&prod, &out_shape)
}

/// `a` の末尾 `n` 軸と `b` の先頭 `n` 軸を縮約する（`torch.tensordot(a, b, dims=n)`）。
///
/// `n = 0` は外積（出力 shape は `a.shape ++ b.shape`）。`n` が `a`／`b` の rank を超える
/// と `InvalidArgument`。その他の検査は [`tensordot_axes`] と同じ。
pub fn tensordot<'t>(a: &Var<'t>, b: &Var<'t>, n: usize) -> Result<Var<'t>, AutodiffError> {
    a.check_same_tape(b)?;
    let ra = a.shape().len();
    let rb = b.shape().len();
    if n > ra || n > rb {
        return Err(invalid(format!(
            "tensor_product_ops::tensordot: n={n} が rank を超える（a の rank={ra}, b の rank={rb}）"
        )));
    }
    let dims_a: Vec<usize> = (ra - n..ra).collect();
    let dims_b: Vec<usize> = (0..n).collect();
    tensordot_axes(a, b, &dims_a, &dims_b)
}

/// 軸リストを明示して縮約する（`torch.tensordot(a, b, dims=(dims_a, dims_b))`）。
///
/// `dims_a[i]` と `dims_b[i]` が対になり、対の軸長は完全一致が必要（size 1 の
/// broadcast 縮約は非対応）。出力 shape は `a` の残り軸 ++ `b` の残り軸。
/// 長さ不一致・軸長不一致は `InvalidArgument`、範囲外は `AxisOutOfRange`、各リスト内の
/// 重複は `DuplicateAxis`。
pub fn tensordot_axes<'t>(
    a: &Var<'t>,
    b: &Var<'t>,
    dims_a: &[usize],
    dims_b: &[usize],
) -> Result<Var<'t>, AutodiffError> {
    a.check_same_tape(b)?;
    if dims_a.len() != dims_b.len() {
        return Err(invalid(format!(
            "tensor_product_ops::tensordot_axes: dims_a と dims_b は同じ長さが必要\
             （dims_a.len()={}, dims_b.len()={}）",
            dims_a.len(),
            dims_b.len()
        )));
    }
    let a_shape = a.shape();
    let b_shape = b.shape();
    let mut seen_a = vec![false; a_shape.len()];
    let mut seen_b = vec![false; b_shape.len()];
    for (&da, &db) in dims_a.iter().zip(dims_b) {
        if da >= a_shape.len() {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: da,
                rank: a_shape.len(),
            }));
        }
        if db >= b_shape.len() {
            return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
                axis: db,
                rank: b_shape.len(),
            }));
        }
        if seen_a[da] {
            return Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: da }));
        }
        if seen_b[db] {
            return Err(AutodiffError::Shape(ShapeError::DuplicateAxis { axis: db }));
        }
        seen_a[da] = true;
        seen_b[db] = true;
        if a_shape[da] != b_shape[db] {
            return Err(invalid(format!(
                "tensor_product_ops::tensordot_axes: 縮約軸の長さが不一致\
                 （a の軸 {da}={}, b の軸 {db}={}）",
                a_shape[da], b_shape[db]
            )));
        }
    }
    let free_a: Vec<usize> = (0..a_shape.len()).filter(|&i| !seen_a[i]).collect();
    let free_b: Vec<usize> = (0..b_shape.len()).filter(|&i| !seen_b[i]).collect();
    let mut perm_a = free_a.clone();
    perm_a.extend_from_slice(dims_a);
    let mut perm_b = dims_b.to_vec();
    perm_b.extend_from_slice(&free_b);
    let free_a_shape: Vec<usize> = free_a.iter().map(|&i| a_shape[i]).collect();
    let free_b_shape: Vec<usize> = free_b.iter().map(|&i| b_shape[i]).collect();
    let contracted: Vec<usize> = dims_a.iter().map(|&i| a_shape[i]).collect();
    let l = checked_numel(&free_a_shape)?;
    let k = checked_numel(&contracted)?;
    let r = checked_numel(&free_b_shape)?;
    let mut out_shape = free_a_shape;
    out_shape.extend_from_slice(&free_b_shape);
    checked_bytes_for::<f32>(&a_shape)?;
    checked_bytes_for::<f32>(&b_shape)?;
    checked_bytes_for::<f32>(&[l, k])?;
    checked_bytes_for::<f32>(&[k, r])?;
    checked_bytes_for::<f32>(&[l, r])?;
    checked_bytes_for::<f32>(&out_shape)?;
    let a_p = permute_if_needed(a, &perm_a)?;
    let a_2d = reshape_contiguous(&a_p, &[l, k])?;
    let b_p = permute_if_needed(b, &perm_b)?;
    let b_2d = reshape_contiguous(&b_p, &[k, r])?;
    let prod = a_2d.matmul(&b_2d)?;
    reshape_contiguous(&prod, &out_shape)
}

/// 2 点集合間の p-ノルム距離行列（`torch.cdist(x1, x2, p)` 相当）。
///
/// `x1: […, P, M]`・`x2: […, R, M]` から `[…, P, R]` を返す（batch 軸は NumPy 互換で
/// broadcast）。両入力は rank 2 以上（`RankMismatch`）、末尾 `M` は一致（`InvalidArgument`）、
/// `p` は有限かつ正（`InvalidArgument`）。中間 `[…, P, R, M]` を確保する O(P·R·M) 実装で、
/// 確保量は tape 操作前に検査する。`M == 0` の距離は 0。
pub fn cdist<'t>(x1: &Var<'t>, x2: &Var<'t>, p: f32) -> Result<Var<'t>, AutodiffError> {
    x1.check_same_tape(x2)?;
    let s1 = x1.shape();
    let s2 = x2.shape();
    for s in [&s1, &s2] {
        if s.len() < 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: s.len(),
            }));
        }
    }
    let m = s1[s1.len() - 1];
    if s2[s2.len() - 1] != m {
        return Err(invalid(format!(
            "tensor_product_ops::cdist: 末尾の特徴次元が不一致（x1={m}, x2={}）",
            s2[s2.len() - 1]
        )));
    }
    if !p.is_finite() || p <= 0.0 {
        return Err(invalid(format!(
            "tensor_product_ops::cdist: p は有限かつ正である必要がある、got {p}"
        )));
    }
    let mut u1 = s1.clone();
    u1.insert(s1.len() - 1, 1);
    let mut u2 = s2.clone();
    u2.insert(s2.len() - 2, 1);
    let diff_shape = broadcast_shape(&u1, &u2).map_err(AutodiffError::Shape)?;
    let last = diff_shape.len() - 1;
    let out_shape = diff_shape[..last].to_vec();
    checked_bytes_for::<f32>(&s1)?;
    checked_bytes_for::<f32>(&s2)?;
    checked_bytes_for::<f32>(&diff_shape)?;
    checked_bytes_for::<f32>(&out_shape)?;
    let a = x1.contiguous()?.unsqueeze(s1.len() - 1)?;
    let b = x2.contiguous()?.unsqueeze(s2.len() - 2)?;
    let diff = a.sub(&b)?;
    if m == 0 {
        // `norm_p` は縮約対象が空だと拒否するため、空軸の和（0）で距離 0 を返す。
        return diff.sum(Some(last));
    }
    norm_p(&diff, p, Some(last))
}

/// 3 次元ベクトルの外積（`torch.linalg.cross(a, b, dim=dim)` 相当）。
///
/// `a`／`b` は同 shape（broadcast 非対応）で `shape[dim] == 3`。`c_i = a_{i+1} b_{i+2} −
/// a_{i+2} b_{i+1}`（添字は 3 で巡回）。shape 不一致・軸長が 3 でないときは
/// `InvalidArgument`、`dim` 範囲外は `AxisOutOfRange`。
pub fn cross<'t>(a: &Var<'t>, b: &Var<'t>, dim: usize) -> Result<Var<'t>, AutodiffError> {
    a.check_same_tape(b)?;
    let shape = a.shape();
    if b.shape() != shape {
        return Err(invalid(format!(
            "tensor_product_ops::cross: a と b は同じ shape が必要（a={shape:?}, b={:?}）",
            b.shape()
        )));
    }
    if dim >= shape.len() {
        return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: dim,
            rank: shape.len(),
        }));
    }
    if shape[dim] != 3 {
        return Err(invalid(format!(
            "tensor_product_ops::cross: 軸 {dim} の長さは 3 が必要、got {}",
            shape[dim]
        )));
    }
    checked_bytes_for::<f32>(&shape)?;
    let a1 = roll(a, &[-1], &[dim])?;
    let a2 = roll(a, &[-2], &[dim])?;
    let b1 = roll(b, &[-1], &[dim])?;
    let b2 = roll(b, &[-2], &[dim])?;
    a1.mul(&b2)?.sub(&a2.mul(&b1)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::Tensor;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.to_tensor().host_slice().into_owned()
    }

    #[test]
    fn kron_known_values() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let b = tape.var(&t(vec![0.0, 5.0, 6.0, 7.0], &[2, 2]));
        let y = kron(&a, &b).unwrap();
        assert_eq!(y.to_tensor().shape(), &[4, 4]);
        assert_eq!(
            vals(&y),
            vec![
                0.0, 5.0, 0.0, 10.0, 6.0, 7.0, 12.0, 14.0, 0.0, 15.0, 0.0, 20.0, 18.0, 21.0, 24.0,
                28.0
            ]
        );
    }

    #[test]
    fn kron_rank_mismatch_and_scalar() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0], &[2]));
        let b = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let y = kron(&a, &b).unwrap();
        assert_eq!(y.to_tensor().shape(), &[2, 4]);
        assert_eq!(vals(&y), vec![1.0, 2.0, 2.0, 4.0, 3.0, 4.0, 6.0, 8.0]);
        let s = tape.var(&t(vec![3.0], &[]));
        let z = kron(&s, &a).unwrap();
        assert_eq!(z.to_tensor().shape(), &[2]);
        assert_eq!(vals(&z), vec![3.0, 6.0]);
        let ss = kron(&s, &s).unwrap();
        assert_eq!(ss.to_tensor().shape(), &[] as &[usize]);
        assert_eq!(vals(&ss), vec![9.0]);
    }

    #[test]
    fn kron_zero_length_axis() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![], &[0, 2]));
        let b = tape.var(&t(vec![1.0, 2.0], &[1, 2]));
        let y = kron(&a, &b).unwrap();
        assert_eq!(y.to_tensor().shape(), &[0, 4]);
    }

    #[test]
    fn tensordot_matches_matmul_with_one_node() {
        let tape = Tape::new();
        let a = tape.var(&t((0..6).map(|i| i as f32 + 0.5).collect(), &[2, 3]));
        let b = tape.var(&t((0..12).map(|i| i as f32 * 0.25).collect(), &[3, 4]));
        let before = tape.len();
        let y = tensordot(&a, &b, 1).unwrap();
        assert_eq!(tape.len() - before, 1);
        let m = a.matmul(&b).unwrap();
        let (yv, mv) = (vals(&y), vals(&m));
        assert_eq!(
            yv.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            mv.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn tensordot_outer_and_full_contraction() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0], &[2]));
        let b = tape.var(&t(vec![3.0, 4.0, 5.0], &[3]));
        let o = tensordot(&a, &b, 0).unwrap();
        assert_eq!(o.to_tensor().shape(), &[2, 3]);
        assert_eq!(vals(&o), vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]);
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let y = tape.var(&t(vec![1.0, 1.0, 2.0, 2.0], &[2, 2]));
        let s = tensordot(&x, &y, 2).unwrap();
        assert_eq!(s.to_tensor().shape(), &[] as &[usize]);
        assert_eq!(vals(&s), vec![1.0 + 2.0 + 6.0 + 8.0]);
    }

    #[test]
    fn tensordot_axes_non_adjacent_and_permuted() {
        let tape = Tape::new();
        let a = tape.var(&t((0..24).map(|i| i as f32).collect(), &[2, 3, 4]));
        let b = tape.var(&t((0..12).map(|i| (i as f32) - 3.0).collect(), &[4, 3]));
        // a の軸 (2, 1) と b の軸 (0, 1) を対にする。残りは a の軸 0。
        let y = tensordot_axes(&a, &b, &[2, 1], &[0, 1]).unwrap();
        assert_eq!(y.to_tensor().shape(), &[2]);
        let av = vals(&a);
        let bv = vals(&b);
        let mut expect = [0.0f32; 2];
        for (i, e) in expect.iter_mut().enumerate() {
            for j in 0..3 {
                for k in 0..4 {
                    *e += av[i * 12 + j * 4 + k] * bv[k * 3 + j];
                }
            }
        }
        for (g, e) in vals(&y).iter().zip(expect) {
            assert!((g - e).abs() < 1e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn tensordot_non_contiguous_input() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let xt = x.transpose(0, 1).unwrap(); // [3, 2]
        let b = tape.var(&t(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]));
        let y = tensordot(&xt, &b, 1).unwrap();
        assert_eq!(y.to_tensor().shape(), &[3, 2]);
        assert_eq!(vals(&y), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn tensordot_zero_contraction_length() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![], &[2, 0]));
        let b = tape.var(&t(vec![], &[0, 3]));
        let y = tensordot(&a, &b, 1).unwrap();
        assert_eq!(y.to_tensor().shape(), &[2, 3]);
        assert_eq!(vals(&y), vec![0.0; 6]);
    }

    #[test]
    fn cdist_known_values() {
        let tape = Tape::new();
        let x1 = tape.var(&t(vec![0.0, 0.0, 3.0, 4.0], &[2, 2]));
        let x2 = tape.var(&t(vec![0.0, 0.0, 6.0, 8.0, 3.0, 0.0], &[3, 2]));
        let d2 = cdist(&x1, &x2, 2.0).unwrap();
        assert_eq!(d2.to_tensor().shape(), &[2, 3]);
        let expect = [0.0, 10.0, 3.0, 5.0, 5.0, 4.0];
        for (g, e) in vals(&d2).iter().zip(expect) {
            assert!((g - e).abs() < 1e-5, "{g} vs {e}");
        }
        let d1 = cdist(&x1, &x2, 1.0).unwrap();
        let expect1 = [0.0, 14.0, 3.0, 7.0, 7.0, 4.0];
        for (g, e) in vals(&d1).iter().zip(expect1) {
            assert!((g - e).abs() < 1e-5, "{g} vs {e}");
        }
    }

    #[test]
    fn cdist_batch_broadcast_and_empty() {
        let tape = Tape::new();
        let x1 = tape.var(&t(vec![1.0; 2 * 3 * 2], &[2, 3, 2]));
        let x2 = tape.var(&t(vec![0.0; 4 * 2], &[4, 2]));
        let d = cdist(&x1, &x2, 3.0).unwrap();
        assert_eq!(d.to_tensor().shape(), &[2, 3, 4]);
        let e1 = tape.var(&t(vec![], &[0, 2]));
        let e = cdist(&e1, &x2, 2.0).unwrap();
        assert_eq!(e.to_tensor().shape(), &[0, 4]);
        let m0a = tape.var(&t(vec![], &[2, 0]));
        let m0b = tape.var(&t(vec![], &[3, 0]));
        let z = cdist(&m0a, &m0b, 2.0).unwrap();
        assert_eq!(z.to_tensor().shape(), &[2, 3]);
        assert_eq!(vals(&z), vec![0.0; 6]);
    }

    #[test]
    fn cross_known_values() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]));
        let b = tape.var(&t(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]));
        let c = cross(&a, &b, 1).unwrap();
        assert_eq!(vals(&c), vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        // 平行ベクトルは 0。
        let p = cross(&a, &a, 1).unwrap();
        assert_eq!(vals(&p), vec![0.0; 6]);
    }

    #[test]
    fn typed_errors_leave_no_orphan_nodes() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![0.0; 6], &[2, 3]));
        let v3 = tape.var(&t(vec![0.0; 3], &[3]));
        let a22 = tape.var(&t(vec![0.0; 4], &[2, 2]));
        let before = tape.len();
        assert!(matches!(
            tensordot(&a, &a, 3),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            tensordot_axes(&a, &a, &[0], &[0, 1]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            tensordot_axes(&a, &a, &[5], &[0]),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
        assert!(matches!(
            tensordot_axes(&a, &a, &[0, 0], &[0, 1]),
            Err(AutodiffError::Shape(ShapeError::DuplicateAxis { .. }))
        ));
        assert!(matches!(
            tensordot_axes(&a, &a, &[0], &[1]),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            cdist(&v3, &a, 2.0),
            Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
        ));
        assert!(matches!(
            cdist(&a, &a22, 2.0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        for p in [0.0f32, -1.0, f32::INFINITY, f32::NAN] {
            assert!(matches!(
                cdist(&a, &a, p),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn typed_errors_do_not_push_nodes() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![0.0; 6], &[2, 3]));
        let b = tape.var(&t(vec![0.0; 6], &[3, 2]));
        let c3 = tape.var(&t(vec![0.0; 6], &[2, 3]));
        let before = tape.len();
        assert!(cross(&a, &b, 1).is_err()); // shape 不一致
        assert!(cross(&a, &c3, 0).is_err()); // 軸長 2
        assert!(cross(&a, &c3, 2).is_err()); // 軸範囲外
        assert!(tensordot(&a, &a, 3).is_err());
        assert!(cdist(&a, &b, 2.0).is_err());
        assert!(cdist(&a, &a, 0.0).is_err());
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn huge_broadcast_view_is_rejected_before_allocation() {
        let tape = Tape::new();
        let s = tape.var(&t(vec![1.0], &[1, 1]));
        let big = s.broadcast_to(&[1 << 31, 1 << 31]).unwrap();
        let before = tape.len();
        assert!(kron(&big, &big).is_err());
        assert!(tensordot(&big, &big, 1).is_err());
        assert!(cdist(&big, &big, 2.0).is_err());
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn cross_tape_is_rejected() {
        let t1 = Tape::new();
        let t2 = Tape::new();
        let a = t1.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let b = t2.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        assert!(matches!(kron(&a, &b), Err(AutodiffError::TapeMismatch)));
        assert!(matches!(
            tensordot(&a, &b, 0),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(matches!(
            cdist(&a, &b, 2.0),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(matches!(cross(&a, &b, 0), Err(AutodiffError::TapeMismatch)));
    }

    #[test]
    fn cdist_identical_points_have_finite_gradient() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let d = cdist(&x, &x, 2.0).unwrap();
        let loss = d.sum(None).unwrap();
        let g = tape.backward(&loss).unwrap();
        let gx = g.get(&x).unwrap().expect("勾配");
        assert!(gx.host_slice().iter().all(|v| v.is_finite()));
    }
}
