//! `cosine_similarity`（PyTorch `F.cosine_similarity`）と `pairwise_distance`
//! （PyTorch `F.pairwise_distance`）の自由関数（イシュー #3821・親 #3813。設計記録
//! `docs/autodiff-similarity-distance-decision.md`。承認は #3768 の全論点そのまま承認）。
//!
//! 呼び出し元は `Var::cosine_similarity`／`Var::pairwise_distance`（1 行委譲）。本モジュール自体は
//! facade から再エクスポートしない。**新規 `Op` 2 種（`Op::CosineSimilarity`・`Op::PairwiseDistance`。
//! いずれも `pub(crate)`）を host の `f64` で計算する**（`BackendOps`・カーネル・`unsafe`・依存はゼロ）。
//! forward は `crate::eval`、VJP は `crate::grad`、距離の核は triplet と共有する
//! `eval::p_norm_f64`／`eval::p_norm_grad_f64`。triplet の既存経路は書き換えない。
//!
//! **式**
//! - cosine: `n_i = sqrt(Σ x_i²)`、`c_i = max(n_i, eps)`（NaN は伝播）、`s = Σ (x1/c1)(x2/c2)`。
//!   `eps` は各ノルムへ個別のクランプで、クランプは勾配に記録しない（straight-through。#3768 Q2 (a)）。
//! - pairwise: `d = ‖x1 − x2 + eps‖_p`（差は `f64` で取り `eps` を先に足す。triplet と同じ規約）。
//!
//! **PyTorch との既知の差分**: broadcast しない（同一 shape のみ。Q3）。`dim` は `usize`（負は不可）。
//! pairwise の `p` は有限かつ 1 以上（`0`・`inf`・`(0, 1)` を拒否。Q4）、`eps` は両演算とも有限かつ 0 以上
//! （PyTorch の pairwise は無検査。Q5）。`create_graph` は型付きエラーで拒否（Q9 (a)・別設計待ち）。
//!
//! **数値契約**: 長軸縮約は `f64` アキュムレータ（要素を先に `f64` へ昇格してから二乗・積算）で、最後に
//! 1 回だけ `f32` へ下げる。host 計算のみなのでバックエンド間は bit 一致で比べる（REQ-2 より強い判定）。
//! tolerance・baseline は変えない。
//!
//! **検査順（REQ-8・A03）**: ① 同一 tape → ② 同 shape → ③ rank >= 1 と軸範囲 → ④ 確保量（入力と縮約後の出力の双方）
//! （`checked_bytes_for`）→ ⑤ スカラー（`p`・`eps`）→ ⑥ 実体化 → ⑦ host 計算 → ⑧ tape へ push。
//! ⑥ の前にすべて済ませるため、エラー時に孤児ノードを残さない。

use fandhe_ai_tensor_core::{ShapeError, require_same_shape};

use crate::bool_ops::checked_bytes_for;
use crate::error::AutodiffError;
use crate::eval;
use crate::loss_ops::materialize_pair;
use crate::tape::Op;
use crate::var::Var;

/// 共通の前段検査（①〜④）。通れば入力 shape を返す。
fn check_pair(x1: &Var<'_>, x2: &Var<'_>) -> Result<Vec<usize>, AutodiffError> {
    x1.check_same_tape(x2)?;
    let shape = x1.shape();
    require_same_shape(&shape, &x2.shape())?;
    if shape.is_empty() {
        return Err(AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 1,
            actual: 0,
        }));
    }
    checked_bytes_for::<f32>(&shape)?;
    Ok(shape)
}

fn check_eps(name: &str, eps: f32) -> Result<(), AutodiffError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "{name}: eps は有限かつ 0 以上でなければならない（got {eps}）"
        )));
    }
    Ok(())
}

/// コサイン類似度（`F.cosine_similarity(x1, x2, dim, eps)` 相当）。出力 shape は `dim` を除いた shape。
///
/// 違反: 別 tape は `TapeMismatch`、shape 不一致は `Shape(ShapeMismatch)`、rank 0 は
/// `Shape(RankMismatch)`、`dim >= rank` は `Shape(AxisOutOfRange)`、`eps` が非有限または負は
/// `InvalidArgument`。空の縮約軸は値 0。
pub fn cosine_similarity<'t>(
    x1: &Var<'t>,
    x2: &Var<'t>,
    dim: usize,
    eps: f32,
) -> Result<Var<'t>, AutodiffError> {
    let shape = check_pair(x1, x2)?;
    if dim >= shape.len() {
        return Err(AutodiffError::Shape(ShapeError::AxisOutOfRange {
            axis: dim,
            rank: shape.len(),
        }));
    }
    check_eps("cosine_similarity", eps)?;
    // 縮約軸が長さ 0 だと入力は 0 要素で通るが出力は他軸の積になりうる。実体化前に出力側も検査する。
    let mut out_shape = shape;
    out_shape.remove(dim);
    checked_bytes_for::<f32>(&out_shape)?;
    let (a, b) = materialize_pair(x1, x2)?;
    let value = eval::cosine_similarity_forward(&a, &b, dim, eps);
    let id = x1.tape().push_eager(
        Op::CosineSimilarity {
            x1: x1.node_id(),
            x2: x2.node_id(),
            dim,
            eps,
        },
        value,
    );
    Ok(Var::from_raw(x1.tape(), id))
}

/// ペアワイズ距離（`F.pairwise_distance(x1, x2, p, eps, keepdim)` 相当）。最後の軸を縮約する。
///
/// 違反: 別 tape は `TapeMismatch`、shape 不一致は `Shape(ShapeMismatch)`、rank 0 は
/// `Shape(RankMismatch)`、`p` が非有限または 1 未満・`eps` が非有限または負は `InvalidArgument`。
pub fn pairwise_distance<'t>(
    x1: &Var<'t>,
    x2: &Var<'t>,
    p: f32,
    eps: f32,
    keepdim: bool,
) -> Result<Var<'t>, AutodiffError> {
    let shape = check_pair(x1, x2)?;
    // 末尾軸が長さ 0 でも出力（他軸の積）が巨大になりうるため、実体化前に出力側も検査する。
    checked_bytes_for::<f32>(&shape[..shape.len() - 1])?;
    if !p.is_finite() || p < 1.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "pairwise_distance: p は有限かつ 1 以上でなければならない（got {p}）"
        )));
    }
    check_eps("pairwise_distance", eps)?;
    let (a, b) = materialize_pair(x1, x2)?;
    let value = eval::pairwise_distance_forward(&a, &b, p, eps, keepdim);
    let id = x1.tape().push_eager(
        Op::PairwiseDistance {
            x1: x1.node_id(),
            x2: x2.node_id(),
            p,
            eps,
            keepdim,
        },
        value,
    );
    Ok(Var::from_raw(x1.tape(), id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;
    use crate::eval::{TripletDistanceParams, triplet_distance_stats};
    use fandhe_ai_tensor_core::Tensor;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).unwrap()
    }

    fn vals(v: &Var<'_>) -> Vec<f32> {
        v.value().host_slice().to_vec()
    }

    fn grads(tape: &Tape, loss: &Var<'_>, vs: &[&Var<'_>]) -> Vec<Vec<f32>> {
        let g = tape.backward(loss).unwrap();
        vs.iter()
            .map(|v| g.get(v).unwrap().unwrap().host_slice().to_vec())
            .collect()
    }

    #[test]
    fn zero_length_reduction_axis_with_huge_output_is_rejected() {
        let tape = Tape::new();
        let shape = [1usize << 61, 0];
        let a = tape.var(&t(vec![], &shape));
        let b = tape.var(&t(vec![], &shape));
        assert!(matches!(
            cosine_similarity(&a, &b, 1, 1e-8),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert!(matches!(
            pairwise_distance(&a, &b, 2.0, 1e-6, false),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
        assert!(matches!(
            pairwise_distance(&a, &b, 2.0, 1e-6, true),
            Err(AutodiffError::Shape(ShapeError::ElementCountOverflow))
        ));
    }

    #[test]
    fn cosine_hand_computed_and_grad() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![3.0, 4.0], &[2]));
        let b = tape.var(&t(vec![4.0, 3.0], &[2]));
        let s = cosine_similarity(&a, &b, 0, 1e-8).unwrap();
        assert_eq!(s.shape(), Vec::<usize>::new());
        // (3*4 + 4*3) / (5*5) = 24/25
        assert!((vals(&s)[0] - 0.96).abs() < 1e-7);
        let g = grads(&tape, &s, &[&a, &b]);
        // ds/da_k = b_k/25 - 0.96 * a_k / 25
        let want_a = [
            4.0 / 25.0 - 0.96 * 3.0 / 25.0,
            3.0 / 25.0 - 0.96 * 4.0 / 25.0,
        ];
        for (got, want) in g[0].iter().zip(want_a) {
            assert!((*got as f64 - want).abs() < 1e-7, "{got} vs {want}");
        }
    }

    #[test]
    fn cosine_middle_dim_and_shapes() {
        let tape = Tape::new();
        let data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin() + 0.1).collect();
        let a = tape.var(&t(data.clone(), &[2, 3, 4]));
        let b = tape.var(&t(data.iter().map(|v| v * 2.0).collect(), &[2, 3, 4]));
        for dim in 0..3 {
            let s = cosine_similarity(&a, &b, dim, 1e-8).unwrap();
            let mut want_shape = vec![2, 3, 4];
            want_shape.remove(dim);
            assert_eq!(s.shape(), want_shape);
            // b = 2a なのでどの軸でも類似度は 1。
            for v in vals(&s) {
                assert!((v - 1.0).abs() < 1e-6, "{v}");
            }
        }
    }

    #[test]
    fn cosine_zero_norm_eps_clamp_and_nan() {
        let tape = Tape::new();
        let z = tape.var(&t(vec![0.0, 0.0], &[2]));
        let o = tape.var(&t(vec![1.0, 2.0], &[2]));
        // ゼロノルムは eps にクランプされ類似度 0。
        assert_eq!(
            vals(&cosine_similarity(&z, &o, 0, 1e-8).unwrap()),
            vec![0.0]
        );
        // eps = 0 のゼロベクトルは 0/0 で NaN。
        assert!(vals(&cosine_similarity(&z, &o, 0, 0.0).unwrap())[0].is_nan());
        // NaN は伝播する。
        let n = tape.var(&t(vec![f32::NAN, 1.0], &[2]));
        assert!(vals(&cosine_similarity(&n, &o, 0, 1e-8).unwrap())[0].is_nan());
    }

    #[test]
    fn cosine_straight_through_gradient_for_small_norm() {
        // 0 < ||x1|| < eps: c1 = eps だが第 2 項を流す（クランプの真の微分ではない）。
        // eps = 1.2: n1 = 0.5 < eps、n2 = sqrt(2) ≈ 1.414 > eps（x2 側はクランプされない）。
        let tape = Tape::new();
        let a = tape.var(&t(vec![0.5, 0.0], &[2]));
        let b = tape.var(&t(vec![1.0, 1.0], &[2]));
        let eps = 1.2f32;
        let s = cosine_similarity(&a, &b, 0, eps).unwrap();
        let (n1, c1, c2) = (0.5f64, eps as f64, 2f64.sqrt());
        let s_val = 0.5 / (c1 * c2);
        assert!((vals(&s)[0] as f64 - s_val).abs() < 1e-6);
        let g = grads(&tape, &s, &[&a]);
        let want0 = 1.0 / (c1 * c2) - s_val * 0.5 / (c1 * n1);
        let want1 = 1.0 / (c1 * c2);
        assert!((g[0][0] as f64 - want0).abs() < 1e-6, "{:?}", g[0]);
        assert!((g[0][1] as f64 - want1).abs() < 1e-6, "{:?}", g[0]);
    }

    #[test]
    fn cosine_zero_vector_gradient() {
        // n1 == 0: 第 2 項は 0。dx1 = x2/(eps*c2)、相手側 dx2 は 0（s = 0）。
        let tape = Tape::new();
        let z = tape.var(&t(vec![0.0, 0.0], &[2]));
        let o = tape.var(&t(vec![3.0, 4.0], &[2]));
        let eps = 0.5f32;
        let s = cosine_similarity(&z, &o, 0, eps).unwrap();
        let g = grads(&tape, &s, &[&z, &o]);
        assert!((g[0][0] - 3.0 / (0.5 * 5.0)).abs() < 1e-6);
        assert!((g[0][1] - 4.0 / (0.5 * 5.0)).abs() < 1e-6);
        assert_eq!(g[1], vec![0.0, 0.0]);
    }

    #[test]
    fn cosine_empty_axis_is_zero() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![], &[2, 0]));
        let b = tape.var(&t(vec![], &[2, 0]));
        let s = cosine_similarity(&a, &b, 1, 1e-8).unwrap();
        assert_eq!(vals(&s), vec![0.0, 0.0]);
        let g = grads(&tape, &s.sum(None).unwrap(), &[&a]);
        assert!(g[0].is_empty());
    }

    #[test]
    fn pairwise_hand_computed_keepdim_and_grad() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![3.0, 4.0, 1.0, 1.0], &[2, 2]));
        let b = tape.var(&t(vec![0.0, 0.0, 1.0, 1.0], &[2, 2]));
        let d = pairwise_distance(&a, &b, 2.0, 0.0, false).unwrap();
        assert_eq!(d.shape(), vec![2]);
        assert_eq!(vals(&d), vec![5.0, 0.0]);
        let dk = pairwise_distance(&a, &b, 2.0, 0.0, true).unwrap();
        assert_eq!(dk.shape(), vec![2, 1]);
        let g = grads(&tape, &d.sum(None).unwrap(), &[&a, &b]);
        // 行 0: (3,4)/5。行 1: ノルム 0 なので勾配 0。
        assert!((g[0][0] - 0.6).abs() < 1e-7 && (g[0][1] - 0.8).abs() < 1e-7);
        assert_eq!(&g[0][2..], &[0.0, 0.0]);
        assert!((g[1][0] + 0.6).abs() < 1e-7 && (g[1][1] + 0.8).abs() < 1e-7);
    }

    #[test]
    fn pairwise_rank1_scalar_and_p1_and_inf_nan() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![1.0, -2.0, 3.0], &[3]));
        let b = tape.var(&t(vec![0.0, 0.0, 0.0], &[3]));
        let d = pairwise_distance(&a, &b, 1.0, 0.0, false).unwrap();
        assert_eq!(d.shape(), Vec::<usize>::new());
        assert_eq!(vals(&d), vec![6.0]);
        let dk = pairwise_distance(&a, &b, 1.0, 0.0, true).unwrap();
        assert_eq!(dk.shape(), vec![1]);
        let inf = tape.var(&t(vec![f32::INFINITY, 0.0, 0.0], &[3]));
        assert!(vals(&pairwise_distance(&inf, &b, 2.0, 0.0, false).unwrap())[0].is_infinite());
        let nan = tape.var(&t(vec![f32::NAN, 0.0, 0.0], &[3]));
        assert!(vals(&pairwise_distance(&nan, &b, 2.0, 0.0, false).unwrap())[0].is_nan());
    }

    #[test]
    fn pairwise_empty_axis_is_zero() {
        let tape = Tape::new();
        let a = tape.var(&t(vec![], &[3, 0]));
        let b = tape.var(&t(vec![], &[3, 0]));
        let d = pairwise_distance(&a, &b, 2.0, 1e-6, false).unwrap();
        assert_eq!(vals(&d), vec![0.0; 3]);
    }

    #[test]
    fn pairwise_matches_triplet_internal_distance_bitwise() {
        let n = 3usize;
        let dd = 5usize;
        let xa: Vec<f32> = (0..n * dd).map(|i| (i as f32 * 0.71).sin() * 3.0).collect();
        let xb: Vec<f32> = (0..n * dd).map(|i| (i as f32 * 1.13).cos() * 2.0).collect();
        for p in [1.0f32, 2.0, 3.5] {
            let eps = 1e-6f32;
            let tape = Tape::new();
            let a = tape.var(&t(xa.clone(), &[n, dd]));
            let b = tape.var(&t(xb.clone(), &[n, dd]));
            let d = vals(&pairwise_distance(&a, &b, p, eps, false).unwrap());
            let af: Vec<f64> = xa.iter().map(|&v| v as f64).collect();
            let bf: Vec<f64> = xb.iter().map(|&v| v as f64).collect();
            let params = TripletDistanceParams {
                p: p as f64,
                eps: eps as f64,
                swap: false,
            };
            for (r, got) in d.iter().enumerate() {
                let (st, ..) = triplet_distance_stats(&af, &bf, &af, r * dd, dd, &params);
                assert_eq!(got.to_bits(), (st.d_ap as f32).to_bits(), "p={p} r={r}");
            }
        }
    }

    #[test]
    fn rejects_invalid_arguments_without_orphan_nodes() {
        let tape = Tape::new();
        let other = Tape::new();
        let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let b = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let c = tape.var(&t(vec![1.0, 2.0, 3.0], &[3]));
        let s0 = tape.var(&t(vec![1.0], &[]));
        let foreign = other.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
        let before = tape.len();
        assert!(matches!(
            cosine_similarity(&a, &foreign, 1, 1e-8),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(matches!(
            pairwise_distance(&a, &foreign, 2.0, 0.0, false),
            Err(AutodiffError::TapeMismatch)
        ));
        assert!(cosine_similarity(&a, &c, 0, 1e-8).is_err());
        assert!(pairwise_distance(&a, &c, 2.0, 0.0, false).is_err());
        assert!(matches!(
            cosine_similarity(&s0, &s0, 0, 1e-8),
            Err(AutodiffError::Shape(ShapeError::RankMismatch { .. }))
        ));
        assert!(pairwise_distance(&s0, &s0, 2.0, 0.0, false).is_err());
        assert!(matches!(
            cosine_similarity(&a, &b, 2, 1e-8),
            Err(AutodiffError::Shape(ShapeError::AxisOutOfRange { .. }))
        ));
        for eps in [-1.0f32, f32::NAN, f32::INFINITY] {
            assert!(matches!(
                cosine_similarity(&a, &b, 1, eps),
                Err(AutodiffError::InvalidArgument(_))
            ));
            assert!(matches!(
                pairwise_distance(&a, &b, 2.0, eps, false),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
        for p in [0.0f32, 0.5, f32::INFINITY, f32::NAN, -1.0] {
            assert!(matches!(
                pairwise_distance(&a, &b, p, 0.0, false),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
        assert_eq!(tape.len(), before, "拒否の前後で tape のノード数が同じ");
    }
}
