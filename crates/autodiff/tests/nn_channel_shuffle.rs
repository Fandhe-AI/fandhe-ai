//! `nn::ChannelShuffle`／`Var::channel_shuffle`（イシュー #3822・設計 #3746）の
//! 受け入れ条件検証。純粋な並べ替えなので REQ-2 複合判定より強い bit 完全一致で比較する。

mod common;

use fandhe_ai_autodiff::nn::{ChannelShuffle, Module};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor
        .contiguous()
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_default()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// 素朴な添字式の参照: `out[n, j*g+i, s] = in[n, i*(C/g)+j, s]`。
fn naive(data: &[f32], n: usize, c: usize, s: usize, g: usize) -> Vec<f32> {
    let cg = c / g;
    let mut out = vec![0.0f32; data.len()];
    for b in 0..n {
        for i in 0..g {
            for j in 0..cg {
                for k in 0..s {
                    out[(b * c + j * g + i) * s + k] = data[(b * c + i * cg + j) * s + k];
                }
            }
        }
    }
    out
}

fn err_of<T>(result: Result<T, AutodiffError>) -> AutodiffError {
    match result {
        Err(err) => err,
        Ok(_) => panic!("expected Err"),
    }
}

#[test]
fn hand_computed_example() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![0.0, 1.0, 2.0, 3.0], &[1, 4, 1, 1]));
    let y = x.channel_shuffle(2).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 4, 1, 1]);
    assert_eq!(dense(&y.to_tensor()), vec![0.0, 2.0, 1.0, 3.0]);
}

#[test]
fn forward_matches_naive_for_rank_3_4_5() {
    for shape in [vec![2usize, 6, 5], vec![2, 6, 3, 2], vec![1, 6, 2, 2, 3]] {
        let numel: usize = shape.iter().product();
        let data: Vec<f32> = (0..numel).map(|v| v as f32 * 0.5 - 7.0).collect();
        let s: usize = shape[2..].iter().product();
        for g in [1usize, 2, 3, 6] {
            let tape = Tape::new_with_ops(common::naive_ops());
            let x = tape.var(&t(data.clone(), &shape));
            let y = x.channel_shuffle(g).unwrap();
            assert_eq!(y.to_tensor().shape(), shape.as_slice());
            let expected = naive(&data, shape[0], shape[1], s, g);
            assert_eq!(
                bits(&dense(&y.to_tensor())),
                bits(&expected),
                "shape={shape:?} g={g}"
            );
        }
    }
}

#[test]
fn groups_one_and_groups_c_are_identity() {
    let data: Vec<f32> = (0..2 * 6 * 4).map(|v| v as f32).collect();
    for g in [1usize, 6] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(data.clone(), &[2, 6, 2, 2]));
        let y = x.channel_shuffle(g).unwrap();
        assert_eq!(dense(&y.to_tensor()), data);
    }
}

#[test]
fn backward_is_inverse_permutation_of_upstream_grad() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((0..48).map(|v| v as f32).collect(), &[2, 6, 2, 2]));
    let gd: Vec<f32> = (0..48).map(|v| v as f32 * 0.25 + 1.0).collect();
    let g = tape.var(&t(gd.clone(), &[2, 6, 2, 2]));
    let y = x.channel_shuffle(2).unwrap();
    let loss = y.mul(&g).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x へ到達する");
    // groups=g の転置の逆は groups=C/g の転置。
    let expected = naive(&gd, 2, 6, 4, 3);
    assert_eq!(bits(&dense(dx)), bits(&expected));
}

#[test]
fn layer_forward_and_forward_host_are_bit_identical() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let input = t((0..72).map(|v| v as f32 * 0.3).collect(), &[2, 6, 3, 2]);
    let layer = ChannelShuffle::new(3).unwrap();
    let via_var = layer.forward(&tape.var(&input)).unwrap();
    let ops = common::naive_ops();
    let via_host = Module::forward_host(&layer, &*ops, &input).unwrap();
    assert_eq!(bits(&dense(&via_var.to_tensor())), bits(&dense(&via_host)));
}

#[test]
fn non_contiguous_input_matches_contiguous() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((0..72).map(|v| v as f32).collect(), &[2, 4, 3, 3]));
    let xt = x.permute(&[0, 1, 3, 2]).unwrap();
    assert!(!xt.to_tensor().is_contiguous());
    let a = xt.channel_shuffle(2).unwrap();
    let xc = tape.var(&t(dense(&xt.to_tensor()), xt.to_tensor().shape()));
    let b = xc.channel_shuffle(2).unwrap();
    assert_eq!(dense(&a.to_tensor()), dense(&b.to_tensor()));
}

#[test]
fn rejections_leave_no_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let rank2 = tape.var(&t(vec![1.0; 4], &[2, 2]));
    let nondiv = tape.var(&t(vec![1.0; 15], &[1, 5, 3]));
    let ok = tape.var(&t(vec![1.0; 12], &[1, 4, 3]));
    let before = tape.len();
    assert!(matches!(
        err_of(ok.channel_shuffle(0)),
        AutodiffError::InvalidArgument(_)
    ));
    assert!(matches!(
        err_of(rank2.channel_shuffle(2)),
        AutodiffError::Shape(_)
    ));
    assert!(matches!(
        err_of(nondiv.channel_shuffle(2)),
        AutodiffError::Shape(_)
    ));
    assert!(ChannelShuffle::new(0).is_err());
    assert_eq!(tape.len(), before, "孤児ノードが残っている");
}

#[test]
fn zero_sized_axes_pass_through_as_empty_tensors() {
    for shape in [vec![0usize, 4, 2], vec![2, 4, 0]] {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&t(vec![], &shape));
        let y = x.channel_shuffle(2).unwrap();
        assert_eq!(y.to_tensor().shape(), shape.as_slice());
    }
}

#[test]
fn central_difference_matches_backward() {
    let base: Vec<f32> = (0..24).map(|v| (v as f32 * 0.37).sin()).collect();
    let w: Vec<f32> = (0..24).map(|v| (v as f32 * 0.11).cos() + 1.5).collect();
    let loss_at = |x: &[f32]| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(&t(x.to_vec(), &[2, 4, 3]));
        let wv = tape.var_no_grad(&t(w.clone(), &[2, 4, 3]));
        let l = xv
            .channel_shuffle(2)
            .unwrap()
            .mul(&wv)
            .unwrap()
            .sum(None)
            .unwrap();
        dense(&l.to_tensor())[0] as f64
    };
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&t(base.clone(), &[2, 4, 3]));
    let wv = tape.var_no_grad(&t(w.clone(), &[2, 4, 3]));
    let l = xv
        .channel_shuffle(2)
        .unwrap()
        .mul(&wv)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&l).unwrap();
    let dx = grads.get(&xv).unwrap().unwrap();
    let flat = dense(dx);
    let h = 1e-2f32;
    for i in 0..24 {
        let (mut p, mut m) = (base.clone(), base.clone());
        p[i] += h;
        m[i] -= h;
        let num = (loss_at(&p) - loss_at(&m)) / (2.0 * h as f64);
        assert!(
            common::req2_close(flat[i] as f64, num),
            "i={i}: {} vs {num}",
            flat[i]
        );
    }
}
