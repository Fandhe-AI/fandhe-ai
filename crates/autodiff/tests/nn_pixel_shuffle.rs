//! `nn::PixelShuffle`／`nn::PixelUnshuffle`（イシュー #2162・親
//! #2131）の受け入れ条件検証。
//!
//! - shuffle してから unshuffle（逆順も）で元に戻ること（bit 完全
//!   一致）
//! - backward: `sum(shuffle(x) * g)` の x 勾配が `unshuffle(g)` と
//!   bit 単位で一致すること（unshuffle 側も対称に確認）
//! - 非 contiguous な入力（permute 後）を受け付け、contiguous 化した
//!   同じ値に対する結果と一致すること
//! - 拒否のすべての経路（rank・割り切れない・オーバーフロー）で
//!   `tape.len()` が変わらないこと（孤児ノードがない）
//! - `nn::Sequential`（Conv2d → PixelShuffle → PixelUnshuffle）への
//!   統合

mod common;

use fandhe_ai_autodiff::nn::{Conv2d, Module, PixelShuffle, PixelUnshuffle, Sequential, summary};
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

// --- 往復（bit 完全一致） ---

#[test]
fn pixel_shuffle_then_unshuffle_round_trips_bit_exact() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=144).map(|v| v as f32).collect(), &[2, 8, 3, 3]));

    let shuffled = PixelShuffle::new(2).unwrap().forward(&x).unwrap();
    let restored = PixelUnshuffle::new(2).unwrap().forward(&shuffled).unwrap();

    assert_eq!(restored.to_tensor().shape(), x.to_tensor().shape());
    assert_eq!(dense(&restored.to_tensor()), dense(&x.to_tensor()));
}

#[test]
fn pixel_unshuffle_then_shuffle_round_trips_bit_exact() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=144).map(|v| v as f32).collect(), &[2, 2, 6, 6]));

    let unshuffled = PixelUnshuffle::new(2).unwrap().forward(&x).unwrap();
    let restored = PixelShuffle::new(2).unwrap().forward(&unshuffled).unwrap();

    assert_eq!(restored.to_tensor().shape(), x.to_tensor().shape());
    assert_eq!(dense(&restored.to_tensor()), dense(&x.to_tensor()));
}

// --- backward（bit 完全一致） ---

#[test]
fn pixel_shuffle_backward_matches_pixel_unshuffle_of_upstream_grad() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=144).map(|v| v as f32).collect(), &[2, 8, 3, 3]));
    let g_data: Vec<f32> = (0..144).map(|v| (v as f32) * 0.5 + 1.0).collect();
    let g = tape.var(&t(g_data, &[2, 2, 6, 6]));

    let y = PixelShuffle::new(2).unwrap().forward(&x).unwrap();
    let loss = y.mul(&g).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x へ到達する");

    // sum(shuffle(x) * g) の dx は、PixelShuffle が純粋な並べ替え
    // （線形写像）であるため unshuffle(g) と一致する（並べ替えの
    // 転置は逆並べ替えそのもの）。
    let dx_expected = PixelUnshuffle::new(2).unwrap().forward(&g).unwrap();

    assert_eq!(dx.shape(), dx_expected.to_tensor().shape());
    assert_eq!(dense(dx), dense(&dx_expected.to_tensor()));
}

#[test]
fn pixel_unshuffle_backward_matches_pixel_shuffle_of_upstream_grad() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=144).map(|v| v as f32).collect(), &[2, 2, 6, 6]));
    let g_data: Vec<f32> = (0..144).map(|v| (v as f32) * 0.25 - 3.0).collect();
    let g = tape.var(&t(g_data, &[2, 8, 3, 3]));

    let y = PixelUnshuffle::new(2).unwrap().forward(&x).unwrap();
    let loss = y.mul(&g).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("x へ到達する");

    let dx_expected = PixelShuffle::new(2).unwrap().forward(&g).unwrap();

    assert_eq!(dx.shape(), dx_expected.to_tensor().shape());
    assert_eq!(dense(dx), dense(&dx_expected.to_tensor()));
}

// --- 非 contiguous 入力 ---

#[test]
fn pixel_shuffle_forward_accepts_non_contiguous_input() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t((1..=144).map(|v| v as f32).collect(), &[2, 8, 3, 3]));
    // H == W == 3 のため軸 2・3 の入れ替えは shape を変えずに非
    // contiguous 化する。
    let xt = x.permute(&[0, 1, 3, 2]).unwrap();
    assert!(!xt.to_tensor().is_contiguous());

    let layer = PixelShuffle::new(2).unwrap();
    let via_noncontig = layer.forward(&xt).unwrap();

    let xt_contig_data = dense(&xt.to_tensor());
    let xt_contig_shape = xt.to_tensor().shape().to_vec();
    let x_contig = tape.var(&t(xt_contig_data, &xt_contig_shape));
    let via_contig = layer.forward(&x_contig).unwrap();

    assert_eq!(
        via_noncontig.to_tensor().shape(),
        via_contig.to_tensor().shape()
    );
    assert_eq!(
        dense(&via_noncontig.to_tensor()),
        dense(&via_contig.to_tensor())
    );
}

// --- 拒否経路（孤児ノードを残さないこと） ---

fn err_of<T>(result: Result<T, AutodiffError>) -> AutodiffError {
    match result {
        Err(err) => err,
        Ok(_) => panic!("expected Err"),
    }
}

#[test]
fn pixel_shuffle_forward_rejects_rank_below_3_without_leaving_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let len_before = tape.len();

    let err = err_of(PixelShuffle::new(2).unwrap().forward(&x));
    assert!(matches!(err, AutodiffError::Shape(_)));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn pixel_shuffle_forward_rejects_non_divisible_channels_without_leaving_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; 3 * 4 * 4], &[3, 4, 4]));
    let len_before = tape.len();

    let err = err_of(PixelShuffle::new(2).unwrap().forward(&x));
    assert!(matches!(err, AutodiffError::Shape(_)));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn pixel_unshuffle_forward_rejects_non_divisible_spatial_without_leaving_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; 3 * 4], &[1, 3, 4]));
    let len_before = tape.len();

    let err = err_of(PixelUnshuffle::new(2).unwrap().forward(&x));
    assert!(matches!(err, AutodiffError::Shape(_)));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

#[test]
fn pixel_shuffle_forward_rejects_checked_mul_overflow_without_leaving_orphan_nodes() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(vec![1.0; 4], &[4, 1, 1]));
    let len_before = tape.len();

    // `upscale_factor` を極端に大きくして `H*r`／`W*r` の
    // `checked_mul` オーバーフローを誘発する。
    let err = err_of(PixelShuffle::new(usize::MAX / 2).unwrap().forward(&x));
    assert!(matches!(err, AutodiffError::Shape(_)));
    assert_eq!(tape.len(), len_before, "孤児ノードが残っている");
}

// --- nn::Sequential 統合 ---

#[test]
fn sequential_with_conv2d_pixel_shuffle_pixel_unshuffle_round_trips() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let mut seq = Sequential::new();
    // Conv2d(in=1, out=8, k=1) で C=1 → C*r*r=8（r=2）へ拡張してから
    // PixelShuffle・PixelUnshuffle を通す。
    seq.push(Box::new(
        Conv2d::new(1, 8, [1, 1], [1, 1], [0, 0], [1, 1], 1, true, 1).unwrap(),
    ));
    seq.push(Box::new(PixelShuffle::new(2).unwrap()));
    seq.push(Box::new(PixelUnshuffle::new(2).unwrap()));

    assert_eq!(
        seq.named_parameters().len(),
        2,
        "Conv2d の weight/bias のみ"
    );
    assert!(seq.parameter_count() > 0);

    let x = tape.var(&t((1..=18).map(|v| v as f32).collect(), &[2, 1, 3, 3]));
    let y = seq.forward(&tape, &x).unwrap();
    assert_eq!(y.to_tensor().shape(), &[2, 8, 3, 3]);

    let text = summary(&seq);
    assert!(!text.is_empty());
}
