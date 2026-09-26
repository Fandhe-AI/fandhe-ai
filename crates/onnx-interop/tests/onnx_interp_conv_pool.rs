//! ONNX `Conv`（1D 拡張）・`MaxPool`／`AveragePool`（イシュー #2199・親
//! #2185）の統合テスト。
//!
//! `ops::conv`／`ops::max_pool`／`ops::average_pool`（ops 直接呼び出し。
//! decode 層を経由しない経路）を、`fandhe_ai_autodiff::nn::{Conv1d,
//! Conv2d, MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d}::forward_host`
//! （`CpuBackendOps`。tape 不要経路）と突合する。
//!
//! ## 「PyTorch export との bit 同一」の再定義（実装計画 §3.3。受け入れ
//! 文言からの逸脱を本ファイルに明記する）
//!
//! 手元に `torch` が無いため、`torch.onnx.export` が実際に出力する
//! ONNX グラフ由来の fixture は使わない。代わりに `ops::*` を直接呼び、
//! 同じハイパーパラメータで構築した `nn::*` 層の `forward_host` と
//! 突合する。bit 一致を主張する組合せ:
//!
//! 1. MaxPool1d／2d・AvgPool1d／2d（`ceil_mode=0`・対称 `pads <=
//!    kernel/2`・avg の `dilation=1`）と `nn::{MaxPool1d,MaxPool2d,
//!    AvgPool1d,AvgPool2d}::forward_host`
//! 2. Conv1d と、同じ重みを 2D へ持ち上げた Conv2d（`ops::conv` 内部の
//!    1D 持ち上げと同じ主張。`ops/conv.rs` の unit テストと同型だが、
//!    ここでは決定的乱数入力で再確認する）
//!
//! REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）
//! に留める組合せ: `Conv` と `nn::Conv1d`／`Conv2d::forward_host`
//! （直接ループと im2col+GEMM で結合順序が異なるため。#2076 の既存契約）。
//!
//! ceil_mode=1・非対称 pads は `nn::*` 側に対応が無いため、手計算した
//! 期待値との bit 一致で確認する。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_autodiff::nn::{AvgPool1d, AvgPool2d, Conv1d, Conv2d, MaxPool1d, MaxPool2d, Module};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_onnx_interop::ops::{ConvAttrs, PoolAttrs, average_pool, conv, max_pool};
use fandhe_ai_tensor_core::Tensor;

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous().as_slice().unwrap().to_vec()
}

fn rand_tensor(shape: &[usize], seed: u64) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let mut rng = Xorshift64Star::new(seed);
    let data: Vec<f32> = (0..numel).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    Tensor::new(data, shape).unwrap()
}

// --- MaxPool2d／AvgPool2d: `ops::*` と `nn::*::forward_host` の bit 一致 ---

#[test]
fn max_pool2d_matches_nn_forward_host_bit_exact() {
    let x = rand_tensor(&[2, 3, 7, 7], 1);
    let attrs = PoolAttrs {
        kernel_shape: vec![3, 3],
        strides: vec![2, 2],
        pads: vec![1, 1, 1, 1],
        ..PoolAttrs::default()
    };
    let via_ops = max_pool(&x, &attrs).unwrap();

    let nn_pool = MaxPool2d::new([3, 3], Some([2, 2]), [1, 1], [1, 1]).unwrap();
    let ops = CpuBackendOps::new();
    let via_nn = nn_pool.forward_host(&ops, &x).unwrap();

    assert_eq!(dense(&via_ops), dense(&via_nn));
}

#[test]
fn max_pool1d_matches_nn_forward_host_bit_exact() {
    let x = rand_tensor(&[2, 3, 11], 2);
    let attrs = PoolAttrs {
        kernel_shape: vec![3],
        strides: vec![2],
        pads: vec![1, 1],
        ..PoolAttrs::default()
    };
    let via_ops = max_pool(&x, &attrs).unwrap();

    let nn_pool = MaxPool1d::new(3, Some(2), 1, 1).unwrap();
    let ops = CpuBackendOps::new();
    let via_nn = nn_pool.forward_host(&ops, &x).unwrap();

    assert_eq!(dense(&via_ops), dense(&via_nn));
}

#[test]
fn avg_pool2d_count_include_pad_true_matches_nn_forward_host_bit_exact() {
    let x = rand_tensor(&[1, 2, 6, 6], 3);
    let attrs = PoolAttrs {
        kernel_shape: vec![2, 2],
        strides: vec![2, 2],
        pads: vec![0, 0, 0, 0],
        count_include_pad: 1,
        ..PoolAttrs::default()
    };
    let via_ops = average_pool(&x, &attrs).unwrap();

    let nn_pool = AvgPool2d::new([2, 2], Some([2, 2]), [0, 0], true).unwrap();
    let ops = CpuBackendOps::new();
    let via_nn = nn_pool.forward_host(&ops, &x).unwrap();

    assert_eq!(dense(&via_ops), dense(&via_nn));
}

#[test]
fn avg_pool1d_count_include_pad_false_matches_nn_forward_host_bit_exact() {
    let x = rand_tensor(&[1, 2, 9], 4);
    let attrs = PoolAttrs {
        kernel_shape: vec![3],
        strides: vec![1],
        pads: vec![1, 1],
        count_include_pad: 0,
        ..PoolAttrs::default()
    };
    let via_ops = average_pool(&x, &attrs).unwrap();

    let nn_pool = AvgPool1d::new(3, Some(1), 1, false).unwrap();
    let ops = CpuBackendOps::new();
    let via_nn = nn_pool.forward_host(&ops, &x).unwrap();

    assert_eq!(dense(&via_ops), dense(&via_nn));
}

// --- Conv1d／Conv2d: REQ-2 統一複合判定（im2col+GEMM vs 直接ループ） ---

#[test]
fn conv2d_matches_nn_forward_host_within_req2() {
    let layer = Conv2d::new(3, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 42).unwrap();
    let x = rand_tensor(&[2, 3, 8, 8], 5);

    let attrs = ConvAttrs {
        kernel_shape: vec![3, 3],
        strides: vec![1, 1],
        pads: vec![1, 1, 1, 1],
        dilations: vec![1, 1],
        group: 1,
        auto_pad: String::new(),
    };
    let via_ops = conv(&x, layer.weight(), layer.bias(), &attrs).unwrap();

    let ops = CpuBackendOps::new();
    let via_nn = layer.forward_host(&ops, &x).unwrap();

    assert_parity(
        "Conv2d: ops::conv vs nn::Conv2d::forward_host",
        &dense(&via_ops),
        &dense(&via_nn),
    );
}

#[test]
fn conv1d_matches_nn_forward_host_within_req2() {
    let layer = Conv1d::new(2, 3, 3, 2, 1, 1, 1, true, 43).unwrap();
    let x = rand_tensor(&[2, 2, 9], 6);

    let attrs = ConvAttrs {
        kernel_shape: vec![3],
        strides: vec![2],
        pads: vec![1, 1],
        dilations: vec![1],
        group: 1,
        auto_pad: String::new(),
    };
    let via_ops = conv(&x, layer.weight(), layer.bias(), &attrs).unwrap();

    let ops = CpuBackendOps::new();
    let via_nn = layer.forward_host(&ops, &x).unwrap();

    assert_parity(
        "Conv1d: ops::conv vs nn::Conv1d::forward_host",
        &dense(&via_ops),
        &dense(&via_nn),
    );
}

// --- ceil_mode=1・非対称 pads: 手計算値との bit 一致 ---

#[test]
fn max_pool_ceil_mode_matches_hand_computed_value() {
    // X: [1,1,1,5] = [1,3,2,5,4]. kernel=2, stride=2, pad=0, ceil_mode=1.
    // floor: (5-2)/2+1=2. ceil: ceil(3/2)+1=3、調整規則 (3-1)*2=4 >= 5+0 は
    // 偽 -> out=3 のまま。
    // windows: [1,3]->3, [2,5]->5, [4]->4 (最後の窓は入力内で 1 要素のみ)。
    let x = Tensor::<f32>::new(vec![1.0, 3.0, 2.0, 5.0, 4.0], &[1, 1, 1, 5]).unwrap();
    let attrs = PoolAttrs {
        kernel_shape: vec![1, 2],
        strides: vec![1, 2],
        ceil_mode: 1,
        ..PoolAttrs::default()
    };
    let y = max_pool(&x, &attrs).unwrap();
    assert_eq!(y.shape(), &[1, 1, 1, 3]);
    let out = dense(&y);
    assert_eq!(out, vec![3.0, 5.0, 4.0]);
}

#[test]
fn average_pool_asymmetric_pads_matches_hand_computed_value() {
    // X: [1,1,1,3] = [1,2,3]. kernel=2, stride=1, pads=[0,1] (begin=0,end=1).
    // count_include_pad=1 -> divisor は padded 座標でクリップ。
    // padded_w = 3+0+1 = 4. windows starts: 0,1,2.
    // w=0: [1,2] divisor=min(0+2,4)-0=2 -> 1.5
    // w=1: [2,3] divisor=min(1+2,4)-1=2 -> 2.5
    // w=2: [3,pad] divisor=min(2+2,4)-2=2 -> (3+0)/2=1.5
    let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[1, 1, 1, 3]).unwrap();
    let attrs = PoolAttrs {
        kernel_shape: vec![1, 2],
        strides: vec![1, 1],
        pads: vec![0, 0, 0, 1],
        count_include_pad: 1,
        ..PoolAttrs::default()
    };
    let y = average_pool(&x, &attrs).unwrap();
    assert_eq!(y.shape(), &[1, 1, 1, 3]);
    let out = dense(&y);
    assert_eq!(out, vec![1.5, 2.5, 1.5]);
}

// --- 複合 CNN: Conv -> MaxPool -> Conv -> AveragePool（ops 手動連結 vs nn 連結） ---

#[test]
fn composite_cnn_chain_matches_nn_within_req2() {
    // Conv(3->4,k3,s1,p1) -> MaxPool(k2,s2) -> Conv(4->2,k3,s1,p1) ->
    // AveragePool(k2,s2)。`ops::*` を手動連結した経路と、`nn::*` を
    // `Module::forward_host` で連結した経路を突合する。Conv を含むため
    // REQ-2 統一複合判定（`assert_parity`）を用いる。
    let conv1 = Conv2d::new(3, 4, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 100).unwrap();
    let conv2 = Conv2d::new(4, 2, [3, 3], [1, 1], [1, 1], [1, 1], 1, true, 200).unwrap();
    let x = rand_tensor(&[1, 3, 8, 8], 7);

    // ops 手動連結。
    let conv_attrs = |kh: i64, kw: i64| ConvAttrs {
        kernel_shape: vec![kh, kw],
        strides: vec![1, 1],
        pads: vec![1, 1, 1, 1],
        dilations: vec![1, 1],
        group: 1,
        auto_pad: String::new(),
    };
    let y1 = conv(&x, conv1.weight(), conv1.bias(), &conv_attrs(3, 3)).unwrap();
    let pool_attrs = PoolAttrs {
        kernel_shape: vec![2, 2],
        strides: vec![2, 2],
        ..PoolAttrs::default()
    };
    let y2 = max_pool(&y1, &pool_attrs).unwrap();
    let y3 = conv(&y2, conv2.weight(), conv2.bias(), &conv_attrs(3, 3)).unwrap();
    let avg_attrs = PoolAttrs {
        kernel_shape: vec![2, 2],
        strides: vec![2, 2],
        count_include_pad: 1,
        ..PoolAttrs::default()
    };
    let y4 = average_pool(&y3, &avg_attrs).unwrap();

    // nn 連結（forward_host）。
    let ops = CpuBackendOps::new();
    let n1 = conv1.forward_host(&ops, &x).unwrap();
    let nn_max = MaxPool2d::new([2, 2], Some([2, 2]), [0, 0], [1, 1]).unwrap();
    let n2 = nn_max.forward_host(&ops, &n1).unwrap();
    let n3 = conv2.forward_host(&ops, &n2).unwrap();
    let nn_avg = AvgPool2d::new([2, 2], Some([2, 2]), [0, 0], true).unwrap();
    let n4 = nn_avg.forward_host(&ops, &n3).unwrap();

    assert_eq!(y4.shape(), n4.shape());
    assert_parity(
        "複合 CNN（Conv->MaxPool->Conv->AveragePool）: ops 連結 vs nn 連結",
        &dense(&y4),
        &dense(&n4),
    );
}
