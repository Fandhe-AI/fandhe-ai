//! `bilinear`（PyTorch `F.bilinear(input1, input2, weight, bias)`、`nn.Bilinear` の計算本体
//! 相当）の自由関数（イシュー #3818・親 #3813。設計記録
//! `docs/nn-shrink-bilinear-channel-shuffle-decision.md` §2）。
//!
//! **新規 `Op`・`BackendOps`・カーネル・`unsafe`・依存はゼロ**（設計 §2 の不変条件）。既存の
//! `Var::{permute, reshape, contiguous, matmul, unsqueeze, mul, sum, add}` の合成のみで、CPU・CUDA・
//! Metal の全バックエンドに既存経路で到達する。呼び出し元は `Var::bilinear`（1 行委譲）と
//! `nn::Bilinear` 層（autodiff 内部。facade には出さない）。本モジュール自体は facade から
//! 再エクスポートしない。
//!
//! **式**: `y[n, o] = Σ_{i,j} x1[n, i] · weight[o, i, j] · x2[n, j] (+ bias[o])`。
//!
//! **形状**: `x1 = [*, in1]`・`x2 = [*, in2]`（先頭の次元は完全一致。rank 2 以上で同 rank）、
//! `weight = [out, in1, in2]`、`bias = [out]`、出力 `[*, out]`。PyTorch は rank 1 入力も受けるが
//! 本実装は拒否する（差分）。
//!
//! **合成**: 先頭の次元の積を `N` として `x1f = [N, in1]`・`x2f = [N, in2]` へ畳み、
//! `W' = weight.permute([1,0,2]) → [in1, out·in2]` と `x1f.matmul(W')` → `[N, out, in2]` →
//! `x2f` を `unsqueeze(1)` して `mul`（broadcast）→ `sum(軸 2)` → `[N, out]`。rank 2 の
//! `MatMul`・`Permute`・`Contiguous`・`Reshape`・`Mul`・単一軸 `Sum`・`Add` はすべて
//! `create_graph` の対象（`docs/autodiff-higher-order-grad-decision.md`）で、二階が追加実装なしで通る。
//!
//! **数値契約**: 縮約は `Var::matmul` の FMA 契約と `sum` の既存契約をそのまま継承する。バックエンド間の
//! bit 一致は主張せず、PyTorch・バックエンド間とも REQ-2 統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満）で比較する。tolerance は変更しない。
//!
//! **境界検査（REQ-8・`.claude/rules/security.md` A03）**: tape・rank・各次元・bias shape の検査と
//! 要素数の `checked_mul` は、tape へノードを積む前にすべて行う（エラー時に孤児ノードを残さない）。

use fandhe_ai_tensor_core::ShapeError;

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::var::Var;

fn overflow() -> AutodiffError {
    AutodiffError::Shape(ShapeError::ElementCountOverflow)
}

fn mismatch(lhs: &[usize], rhs: &[usize]) -> AutodiffError {
    AutodiffError::Shape(ShapeError::ShapeMismatch {
        lhs: lhs.to_vec(),
        rhs: rhs.to_vec(),
    })
}

/// contiguous 化してから `reshape` する。shape が同じなら `reshape` を呼ばない。
fn reshape_contiguous<'t>(x: &Var<'t>, shape: &[usize]) -> Result<Var<'t>, AutodiffError> {
    let c = x.contiguous()?;
    if c.shape() == shape {
        return Ok(c);
    }
    c.reshape(shape)
}

/// 双線形変換（`F.bilinear(x1, x2, weight, bias)` 相当）。
///
/// 検査順: 同一 tape（`x2`・`weight`・`bias`）→ `weight` の rank 3 → `x1`・`x2` の rank 2 以上かつ
/// 同 rank → 先頭の次元の完全一致 → `x1[-1] == weight[1]`・`x2[-1] == weight[2]` → `bias == [out]`。
/// 違反は `TapeMismatch`／`Shape(RankMismatch | ShapeMismatch | ElementCountOverflow)`。
pub fn bilinear<'t>(
    x1: &Var<'t>,
    x2: &Var<'t>,
    weight: &Var<'t>,
    bias: Option<&Var<'t>>,
) -> Result<Var<'t>, AutodiffError> {
    x1.check_same_tape(x2)?;
    x1.check_same_tape(weight)?;
    if let Some(b) = bias {
        x1.check_same_tape(b)?;
    }
    let s1 = x1.shape();
    let s2 = x2.shape();
    let sw = weight.shape();
    if sw.len() != 3 {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 3,
            actual: sw.len(),
        }));
    }
    for s in [&s1, &s2] {
        if s.len() < 2 {
            return Err(AutodiffError::Shape(ShapeError::RankMismatch {
                expected: 2,
                actual: s.len(),
            }));
        }
    }
    if s1.len() != s2.len() {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: s1.len(),
            actual: s2.len(),
        }));
    }
    let rank = s1.len();
    if s1[..rank - 1] != s2[..rank - 1] {
        return Err(mismatch(&s1[..rank - 1], &s2[..rank - 1]));
    }
    let (out, in1, in2) = (sw[0], sw[1], sw[2]);
    if s1[rank - 1] != in1 {
        return Err(mismatch(&[s1[rank - 1]], &[in1]));
    }
    if s2[rank - 1] != in2 {
        return Err(mismatch(&[s2[rank - 1]], &[in2]));
    }
    if let Some(b) = bias {
        let sb = b.shape();
        if sb != [out] {
            return Err(mismatch(&sb, &[out]));
        }
    }
    let lead = &s1[..rank - 1];
    let n = lead
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(overflow)?;
    let mid = [n, out, in2];
    let w_flat = [in1, out.checked_mul(in2).ok_or_else(overflow)?];
    let mut out_shape = lead.to_vec();
    out_shape.push(out);
    // 中間 `[N, out, in2]` の確保量を tape 操作の前に検査する（要素数積の overflow も
    // `ElementCountOverflow` で返る）。
    checked_bytes_for::<f32>(&mid)?;
    // in2 == 0 だと mid の確保量が 0 になり巨大な出力を検出できないため、出力側も別に検査する
    checked_bytes_for::<f32>(&out_shape)?;
    checked_bytes_for::<f32>(&[n, in1])?;
    checked_bytes_for::<f32>(&[n, in2])?;

    let x1f = reshape_contiguous(x1, &[n, in1])?;
    let x2f = reshape_contiguous(x2, &[n, in2])?;
    let w = weight.permute(&[1, 0, 2])?;
    let w = reshape_contiguous(&w, &w_flat)?;
    let t = x1f.matmul(&w)?.reshape(&mid)?;
    let mut y = t.mul(&x2f.unsqueeze(1)?)?.sum(Some(2))?;
    if let Some(b) = bias {
        y = y.add(b)?;
    }
    if y.shape() == out_shape {
        return Ok(y);
    }
    y.reshape(&out_shape)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;
    use fandhe_ai_tensor_core::Tensor;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).unwrap()
    }

    type RefInputs<'a> = (&'a [f32], &'a [f32], &'a [f32], Option<&'a [f32]>);

    /// f64 の素朴な三重ループ参照。`x1 = [N, in1]`・`x2 = [N, in2]`・`w = [out, in1, in2]`。
    fn reference(
        (x1, x2, w, b): RefInputs<'_>,
        (n, in1, in2, out): (usize, usize, usize, usize),
    ) -> Vec<f64> {
        let mut y = vec![0.0f64; n * out];
        for r in 0..n {
            for o in 0..out {
                let mut acc = b.map_or(0.0, |b| b[o] as f64);
                for i in 0..in1 {
                    for j in 0..in2 {
                        acc += x1[r * in1 + i] as f64
                            * w[(o * in1 + i) * in2 + j] as f64
                            * x2[r * in2 + j] as f64;
                    }
                }
                y[r * out + o] = acc;
            }
        }
        y
    }

    fn close(a: f64, e: f64) -> bool {
        (a - e).abs() < 1e-5 || (a - e).abs() / e.abs() < 1e-3
    }

    #[test]
    fn forward_backward_hand_computed_without_bias() {
        // N=1, in1=2, in2=2, out=1: y = x1^T W x2。
        let tape = Tape::new();
        let x1 = tape.var(&t(vec![1.0, 2.0], &[1, 2]));
        let x2 = tape.var(&t(vec![3.0, 4.0], &[1, 2]));
        let w = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]));
        let y = bilinear(&x1, &x2, &w, None).unwrap();
        assert_eq!(y.shape(), vec![1, 1]);
        // 1*1*3 + 1*2*4 + 2*3*3 + 2*4*4 = 3 + 8 + 18 + 32 = 61
        assert_eq!(y.value().host_slice().to_vec(), vec![61.0]);
        let loss = y.sum(None).unwrap();
        let g = tape.backward(&loss).unwrap();
        // dx1 = W x2 = [1*3+2*4, 3*3+4*4]
        assert_eq!(
            g.get(&x1).unwrap().unwrap().host_slice().to_vec(),
            vec![11.0, 25.0]
        );
        // dx2 = W^T x1 = [1*1+3*2, 2*1+4*2]
        assert_eq!(
            g.get(&x2).unwrap().unwrap().host_slice().to_vec(),
            vec![7.0, 10.0]
        );
        // dW = x1 ⊗ x2
        assert_eq!(
            g.get(&w).unwrap().unwrap().host_slice().to_vec(),
            vec![3.0, 4.0, 6.0, 8.0]
        );
    }

    #[test]
    fn bias_and_leading_dims_hand_computed() {
        // 先頭の次元 [2, 2]、in1=in2=1、out=2: y[., o] = x1*w[o]*x2 + b[o]。
        let tape = Tape::new();
        let x1 = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2, 1]));
        let x2 = tape.var(&t(vec![1.0, 1.0, 2.0, 2.0], &[2, 2, 1]));
        let w = tape.var(&t(vec![10.0, 100.0], &[2, 1, 1]));
        let b = tape.var(&t(vec![0.5, -0.5], &[2]));
        let y = bilinear(&x1, &x2, &w, Some(&b)).unwrap();
        assert_eq!(y.shape(), vec![2, 2, 2]);
        assert_eq!(
            y.value().host_slice().to_vec(),
            vec![10.5, 99.5, 20.5, 199.5, 60.5, 599.5, 80.5, 799.5]
        );
        let loss = y.sum(None).unwrap();
        let g = tape.backward(&loss).unwrap();
        assert_eq!(
            g.get(&b).unwrap().unwrap().host_slice().to_vec(),
            vec![4.0, 4.0]
        );
        // dW[o] = Σ x1*x2 = 1 + 2 + 6 + 8 = 17
        assert_eq!(
            g.get(&w).unwrap().unwrap().host_slice().to_vec(),
            vec![17.0, 17.0]
        );
    }

    #[test]
    fn matches_f64_reference_and_central_difference() {
        let dims = (3usize, 4usize, 5usize, 2usize);
        let (n, in1, in2, out) = dims;
        let seq = |len: usize, k: f32| -> Vec<f32> {
            (0..len).map(|i| (i as f32 * k).sin() * 0.7).collect()
        };
        let a = seq(n * in1, 0.9);
        let c = seq(n * in2, 1.3);
        let wv = seq(out * in1 * in2, 0.4);
        let bv = seq(out, 2.1);
        let tape = Tape::new();
        let x1 = tape.var(&t(a.clone(), &[n, in1]));
        let x2 = tape.var(&t(c.clone(), &[n, in2]));
        let w = tape.var(&t(wv.clone(), &[out, in1, in2]));
        let b = tape.var(&t(bv.clone(), &[out]));
        let y = bilinear(&x1, &x2, &w, Some(&b)).unwrap();
        let want = reference((&a, &c, &wv, Some(&bv)), dims);
        for (g, e) in y.value().host_slice().iter().zip(&want) {
            assert!(close(*g as f64, *e), "{g} vs {e}");
        }
        // 中央差分: loss = Σ y のときの d loss / d w[k]。
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let gw = grads.get(&w).unwrap().unwrap().host_slice().to_vec();
        let h = 1e-2f32;
        for k in 0..wv.len() {
            let eval = |delta: f32| -> f64 {
                let mut w2 = wv.clone();
                w2[k] += delta;
                reference((&a, &c, &w2, Some(&bv)), dims).iter().sum()
            };
            let num = (eval(h) - eval(-h)) / (2.0 * h as f64);
            assert!(close(gw[k] as f64, num), "{} vs {num}", gw[k]);
        }
    }

    #[test]
    fn zero_size_axes_are_accepted() {
        let tape = Tape::new();
        // N == 0
        let x1 = tape.var(&t(vec![], &[0, 2]));
        let x2 = tape.var(&t(vec![], &[0, 3]));
        let w = tape.var(&t(vec![0.5; 4 * 2 * 3], &[4, 2, 3]));
        assert_eq!(bilinear(&x1, &x2, &w, None).unwrap().shape(), vec![0, 4]);
        // out == 0
        let x1 = tape.var(&t(vec![1.0; 4], &[2, 2]));
        let x2 = tape.var(&t(vec![1.0; 6], &[2, 3]));
        let w0 = tape.var(&t(vec![], &[0, 2, 3]));
        assert_eq!(bilinear(&x1, &x2, &w0, None).unwrap().shape(), vec![2, 0]);
        // in2 == 0 かつ巨大 out: 事前検査で型付きエラーになり panic しない
        let xe1 = tape.var(&t(vec![], &[1, 0]));
        let xe2 = tape.var(&t(vec![], &[1, 0]));
        let we = tape.var(&t(vec![], &[1usize << 61, 0, 0]));
        assert!(bilinear(&xe1, &xe2, &we, None).is_err());
        // in1 == 0: 縮約が空で bias だけが出る。
        let x1z = tape.var(&t(vec![], &[2, 0]));
        let wz = tape.var(&t(vec![], &[1, 0, 3]));
        let b = tape.var(&t(vec![0.25], &[1]));
        let y = bilinear(&x1z, &x2, &wz, Some(&b)).unwrap();
        assert_eq!(y.value().host_slice().to_vec(), vec![0.25, 0.25]);
        // in2 == 0
        let x2z = tape.var(&t(vec![], &[2, 0]));
        let wz2 = tape.var(&t(vec![], &[1, 2, 0]));
        let y = bilinear(&x1, &x2z, &wz2, Some(&b)).unwrap();
        assert_eq!(y.value().host_slice().to_vec(), vec![0.25, 0.25]);
    }

    #[test]
    fn invalid_inputs_are_rejected_without_orphan_nodes() {
        let tape = Tape::new();
        let other = Tape::new();
        let x1 = tape.var(&t(vec![1.0; 4], &[2, 2]));
        let x2 = tape.var(&t(vec![1.0; 6], &[2, 3]));
        let w = tape.var(&t(vec![1.0; 12], &[2, 2, 3]));
        let b = tape.var(&t(vec![0.0; 2], &[2]));
        let ox2 = other.var(&t(vec![1.0; 6], &[2, 3]));
        let ow = other.var(&t(vec![1.0; 12], &[2, 2, 3]));
        let ob = other.var(&t(vec![0.0; 2], &[2]));
        let w2 = tape.var(&t(vec![1.0; 6], &[2, 3]));
        let v1 = tape.var(&t(vec![1.0; 2], &[2]));
        let v2 = tape.var(&t(vec![1.0; 3], &[3]));
        let x2r3 = tape.var(&t(vec![1.0; 6], &[1, 2, 3]));
        let x2n = tape.var(&t(vec![1.0; 9], &[3, 3]));
        let x1bad = tape.var(&t(vec![1.0; 6], &[2, 3]));
        let x2bad = tape.var(&t(vec![1.0; 4], &[2, 2]));
        let bbad = tape.var(&t(vec![0.0; 3], &[3]));
        let before = tape.len();

        let rank = |r: Result<Var<'_>, AutodiffError>| {
            matches!(
                r,
                Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
            )
        };
        let shp = |r: Result<Var<'_>, AutodiffError>| {
            matches!(
                r,
                Err(AutodiffError::Shape(ShapeError::ShapeMismatch { .. }))
            )
        };
        // 別 tape（x2・weight・bias の各々）。
        assert!(matches!(
            bilinear(&x1, &ox2, &w, None),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(matches!(
            bilinear(&x1, &x2, &ow, None),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(matches!(
            bilinear(&x1, &x2, &w, Some(&ob)),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(rank(bilinear(&x1, &x2, &w2, None))); // weight rank
        assert!(rank(bilinear(&v1, &x2, &w, None))); // x1 rank 1
        assert!(rank(bilinear(&x1, &v2, &w, None))); // x2 rank 1
        assert!(rank(bilinear(&x1, &x2r3, &w, None))); // rank 不一致
        assert!(shp(bilinear(&x1, &x2n, &w, None))); // 先頭の次元
        assert!(shp(bilinear(&x1bad, &x2, &w, None))); // in1
        assert!(shp(bilinear(&x1, &x2bad, &w, None))); // in2
        assert!(shp(bilinear(&x1, &x2, &w, Some(&bbad)))); // bias
        assert_eq!(tape.len(), before, "検査失敗でノードを積まない");
        assert!(bilinear(&x1, &x2, &w, Some(&b)).is_ok());
    }
}
