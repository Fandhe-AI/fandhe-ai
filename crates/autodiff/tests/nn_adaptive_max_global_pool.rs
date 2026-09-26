//! `nn::{AdaptiveMaxPool2d, AdaptiveMaxPool1d, GlobalPool}`
//! （イシュー #2160・設計 `docs/pooling-ops-design.md` §11）の
//! 受け入れ条件検証。**内部クレート限定**（facade 未公開。
//! `crate::adaptive_max_pool_ops` モジュール doc 参照）のため、
//! `nn` 層経由でのみ到達する。
//!
//! - AdaptiveMaxPool2d が割り切れる形状で `MaxPool2d` と値・索引とも
//!   bit 一致（forward）。
//! - `output_size == 入力 shape` で恒等写像・索引が `0..H*W`。
//! - 重なり窓での勾配 `scatter_add` 契約（1 入力が 2 窓の勝者になる
//!   ケース）。
//! - 数値微分との突合（タイのない入力）。
//! - AdaptiveMaxPool1d が `AdaptiveMaxPool2d([1,o])` の reshape と
//!   bit 一致。
//! - GlobalPool(Avg) が `AdaptiveAvgPool2d([1,1])` と、GlobalPool(Max)
//!   の values が `AdaptiveMaxPool2d([1,1])` の values と bit 一致
//!   （rank 3／rank 4 両方）。`keepdims` の shape。
//! - 無効引数（`output_size=0`・rank 不一致）の拒否。
//! - `forward_host` ≡ `forward`（bit 一致）。
//! - `nn::Sequential` 組み込み・`named_parameters` 空・`is_pooling`。

mod common;

use fandhe_ai_autodiff::nn::{
    AdaptiveAvgPool2d, AdaptiveMaxPool1d, AdaptiveMaxPool2d, GlobalPool, GlobalPoolMode, MaxPool2d,
    Module, Sequential,
};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

const H: f64 = 1e-3;
const TAU: f64 = 1e-4;
const REL_TOL: f64 = 1e-2;
const ABS_TOL: f64 = 1e-3;

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

fn assert_bit_exact(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape が一致しない");
    let av = dense(a);
    let bv = dense(b);
    for (i, (&x, &y)) in av.iter().zip(bv.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}[{i}]: bit 不一致（a={x}, b={y}）"
        );
    }
}

fn assert_grad_close(label: &str, analytic: &[f32], numeric: &[f64]) {
    assert_eq!(analytic.len(), numeric.len(), "{label}: 要素数不一致");
    for (i, (&av, &nv)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let av64 = av as f64;
        let diff = (av64 - nv).abs();
        let rel = diff / av64.abs().max(nv.abs()).max(TAU);
        assert!(
            rel <= REL_TOL || diff <= ABS_TOL,
            "{label}[{i}]: analytic={av64} numeric={nv} diff={diff} rel={rel}"
        );
    }
}

// --- 1. AdaptiveMaxPool2d が割り切れる形状で MaxPool2d と bit 一致 ---

#[test]
fn adaptive_max_pool2d_divisible_matches_max_pool2d_values_and_index() {
    let x = t((1..=16).map(|v| v as f32).collect(), &[1, 1, 4, 4]);

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&x);
    let layer = AdaptiveMaxPool2d::new([2, 2]).unwrap();
    let (via_adaptive, idx_adaptive) = layer.forward(&xv_a).unwrap();

    let tape_m = Tape::new_with_ops(common::naive_ops());
    let xv_m = tape_m.var(&x);
    let mp = MaxPool2d::new([2, 2], None, [0, 0], [1, 1]).unwrap();
    let (via_max, idx_max) = mp.forward(&xv_m).unwrap();

    assert_bit_exact(
        "adaptive_max_pool2d vs max_pool2d(kernel=stride=2)",
        &via_adaptive.to_tensor(),
        &via_max.to_tensor(),
    );
    assert_eq!(dense_i32(&idx_adaptive), dense_i32(&idx_max));
}

fn dense_i32(t: &Tensor<i32>) -> Vec<i32> {
    t.contiguous()
        .as_slice()
        .map(|s| s.to_vec())
        .unwrap_or_default()
}

// --- 2. output_size == 入力 shape のとき恒等写像・索引が 0..H*W ---

#[test]
fn adaptive_max_pool2d_identity_when_output_size_equals_input() {
    let x = t((0..12).map(|v| v as f32 * 1.5).collect(), &[1, 1, 3, 4]);
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let layer = AdaptiveMaxPool2d::new([3, 4]).unwrap();
    let (values, index) = layer.forward(&xv).unwrap();

    assert_bit_exact("identity adaptive_max_pool2d", &values.to_tensor(), &x);
    let idx_data = dense_i32(&index);
    let expected: Vec<i32> = (0..12).collect();
    assert_eq!(idx_data, expected);
}

// --- 3. 重なり窓での勾配 scatter_add 契約 ---

#[test]
fn adaptive_max_pool2d_overlapping_window_backward_accumulates_scatter_add() {
    // in=5, out=3: adaptive_window により窓 [1,4)・[3,5) が重なり合う。
    // 両窓の勝者が同一入力位置（flat=3）を共有するよう構成する。
    let tape = Tape::new_with_ops(common::naive_ops());
    let data = vec![0.0, 1.0, 2.0, 9.0, 0.5];
    let x = t(data, &[1, 1, 5]);
    let xv = tape.var(&x);
    let layer = AdaptiveMaxPool1d::new(3).unwrap();
    let (y, idx) = layer.forward(&xv).unwrap();
    // 窓: [0,2)={0,1}->1@1, [1,4)={1,2,9}->9@3, [3,5)={9,0.5}->9@3
    // （窓 1・窓 2 とも勝者が同一入力位置 flat=3 を共有する）。
    assert_eq!(dense(&y.to_tensor()), vec![1.0, 9.0, 9.0]);
    assert_eq!(dense_i32(&idx), vec![1, 3, 3]);

    let upstream = t(vec![10.0, 100.0, 1000.0], &[1, 1, 3]);
    let uv = tape.var(&upstream);
    let loss = y.mul(&uv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().unwrap();
    // scatter_add: 共有位置3(flat=3)<-100+1000=1100、位置1<-10、他は0。
    assert_eq!(dense(dx), vec![0.0, 10.0, 0.0, 1100.0, 0.0]);
}

// --- 4. 数値微分との突合（タイのない入力） ---

#[test]
fn adaptive_max_pool2d_matches_numeric_gradient_no_ties() {
    let x = t(
        vec![
            1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0, 7.0, 6.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,
        ],
        &[1, 1, 4, 4],
    );
    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let layer = AdaptiveMaxPool2d::new([2, 2]).unwrap();
    let (y, _idx) = layer.forward(&xv).unwrap();
    let s = t(
        vec![1.0; y.to_tensor().shape().iter().product()],
        y.to_tensor().shape(),
    );
    let sv = tape.var(&s);
    let loss = y.mul(&sv).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&xv).unwrap().unwrap();

    let forward = |x: &Tensor<f32>| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let xv = tape.var(x);
        let layer = AdaptiveMaxPool2d::new([2, 2]).unwrap();
        let (y, _idx) = layer.forward(&xv).unwrap();
        dense(&y.to_tensor())
            .iter()
            .zip(dense(&s).iter())
            .map(|(&yv, &sv)| yv as f64 * sv as f64)
            .sum()
    };
    let shape = x.shape().to_vec();
    let mut data = dense(&x);
    let mut numeric = vec![0f64; data.len()];
    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = forward(&t(data.clone(), &shape));
        data[i] = (orig - H) as f32;
        let lm = forward(&t(data.clone(), &shape));
        data[i] = orig as f32;
        numeric[i] = (lp - lm) / (2.0 * H);
    }
    assert_grad_close("adaptive_max_pool2d(no ties) dX", &dense(dx), &numeric);
}

// --- 5. AdaptiveMaxPool1d が AdaptiveMaxPool2d([1,o]) の reshape と bit 一致 ---

#[test]
fn adaptive_max_pool1d_matches_2d_reshape() {
    let tape2d = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0], &[1, 1, 1, 7]);
    let x2 = tape2d.var(&x);
    let layer2d = AdaptiveMaxPool2d::new([1, 3]).unwrap();
    let (y2, idx2) = layer2d.forward(&x2).unwrap();

    let tape1d = Tape::new_with_ops(common::naive_ops());
    let x1_flat = t(vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0], &[1, 1, 7]);
    let x1 = tape1d.var(&x1_flat);
    let layer1d = AdaptiveMaxPool1d::new(3).unwrap();
    let (y1, idx1) = layer1d.forward(&x1).unwrap();

    // `y1`（`[N,C,L]`）と `y2`（`[N,C,1,L]`）は shape の rank が異なる
    // （1d は 2d の `H` 軸〈`out_h=1`〉を reshape で潰した最終形の
    // ため）。値・索引は要素順で比較する。
    assert_eq!(dense(&y1.to_tensor()), dense(&y2.to_tensor()));
    assert_eq!(dense_i32(&idx1), dense_i32(&idx2));
}

// --- 6. GlobalPool(Avg) が AdaptiveAvgPool2d([1,1]) と bit 一致。
//        GlobalPool(Max) の values が AdaptiveMaxPool2d([1,1]) と一致。
//        rank 3／rank 4・keepdims の shape。 ---

#[test]
fn global_pool_avg_matches_adaptive_avg_pool2d_rank4() {
    let x = t(
        (0..24).map(|v| v as f32 * 0.3 - 1.0).collect(),
        &[1, 2, 3, 4],
    );

    let tape_g = Tape::new_with_ops(common::naive_ops());
    let xv_g = tape_g.var(&x);
    let gp = GlobalPool::new(GlobalPoolMode::Avg, true);
    let via_global = gp.forward(&xv_g).unwrap();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&x);
    let ap = AdaptiveAvgPool2d::new([1, 1]).unwrap();
    let via_adaptive = ap.forward(&xv_a).unwrap();

    assert_bit_exact(
        "GlobalPool(Avg, keepdims=true) vs AdaptiveAvgPool2d([1,1])",
        &via_global.to_tensor(),
        &via_adaptive.to_tensor(),
    );
    assert_eq!(via_global.to_tensor().shape(), &[1, 2, 1, 1]);
}

#[test]
fn global_pool_max_values_match_adaptive_max_pool2d_rank4() {
    let x = t(
        (0..24).map(|v| (v as f32 * 1.7).sin()).collect(),
        &[1, 2, 3, 4],
    );

    let tape_g = Tape::new_with_ops(common::naive_ops());
    let xv_g = tape_g.var(&x);
    let gp = GlobalPool::new(GlobalPoolMode::Max, true);
    let via_global = gp.forward(&xv_g).unwrap();

    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xv_a = tape_a.var(&x);
    let ap = AdaptiveMaxPool2d::new([1, 1]).unwrap();
    let (via_adaptive, _idx) = ap.forward(&xv_a).unwrap();

    assert_bit_exact(
        "GlobalPool(Max, keepdims=true) values vs AdaptiveMaxPool2d([1,1]) values",
        &via_global.to_tensor(),
        &via_adaptive.to_tensor(),
    );
}

#[test]
fn global_pool_rank3_and_keepdims_shapes() {
    let x = t((0..12).map(|v| v as f32).collect(), &[2, 3, 2]);

    // keepdims=true, rank 3 -> [N, C, 1]
    let tape1 = Tape::new_with_ops(common::naive_ops());
    let xv1 = tape1.var(&x);
    let gp_keep = GlobalPool::new(GlobalPoolMode::Avg, true);
    let y_keep = gp_keep.forward(&xv1).unwrap();
    assert_eq!(y_keep.to_tensor().shape(), &[2, 3, 1]);

    // keepdims=false, rank 3 -> [N, C]
    let tape2 = Tape::new_with_ops(common::naive_ops());
    let xv2 = tape2.var(&x);
    let gp_flat = GlobalPool::new(GlobalPoolMode::Avg, false);
    let y_flat = gp_flat.forward(&xv2).unwrap();
    assert_eq!(y_flat.to_tensor().shape(), &[2, 3]);

    // keepdims=false, rank 4 -> [N, C]
    let x4 = t((0..24).map(|v| v as f32).collect(), &[1, 2, 3, 4]);
    let tape3 = Tape::new_with_ops(common::naive_ops());
    let xv3 = tape3.var(&x4);
    let gp_flat4 = GlobalPool::new(GlobalPoolMode::Max, false);
    let y_flat4 = gp_flat4.forward(&xv3).unwrap();
    assert_eq!(y_flat4.to_tensor().shape(), &[1, 2]);
}

// --- 7. 無効引数の拒否 ---

#[test]
fn adaptive_max_pool2d_new_rejects_output_size_zero() {
    assert!(AdaptiveMaxPool2d::new([0, 2]).is_err());
    assert!(AdaptiveMaxPool2d::new([2, 0]).is_err());
}

#[test]
fn adaptive_max_pool1d_new_rejects_output_size_zero() {
    assert!(AdaptiveMaxPool1d::new(0).is_err());
}

// PR #2280 レビュー指摘（codex-review・P2）: `L` が `i32::MAX` を
// 超える入力は要素数ゼロ（`N=0`）でも `adaptive_pool2d_out_shape` の
// 形状検査（rank・空間軸ゼロ・output_size>=1）だけでは弾けない
// （`h=1・w=L` は非ゼロのため）。索引用 `l <= i32::MAX` は
// `contiguous()?.reshape(...)` による view 作成より前に検査する契約
// （`adaptive_max_pool_ops::adaptive_max_pool1d` doc 参照）であり、
// view 作成後に `IndexRangeOverflow` を返してテープへ孤立ノードを
// 残さないことを確認する。
#[test]
fn adaptive_max_pool1d_rejects_index_overflow_before_reshape() {
    let huge_l = i32::MAX as usize + 1;
    let tape = Tape::new_with_ops(common::naive_ops());
    // N=0 のため data は空で構築でき、numel オーバーフローには
    // 当たらない（検査対象は L 自体の索引表現可能性）。
    let x = Tensor::<f32>::new(vec![], &[0, 1, huge_l])
        .expect("test fixture: N=0 のため空 data で shape と整合する");
    let xv = tape.var(&x);
    let layer = AdaptiveMaxPool1d::new(1).unwrap();
    let err = layer.forward(&xv).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::IndexRangeOverflow { index }) if index == huge_l
    ));
}

/// `forward_host`（host 経路）が `forward`（tape 経路）と同じ
/// `H·W <= i32::MAX` 索引範囲検査を行うことの回帰テスト
/// （codex-review・Cursor Bugbot 指摘・イシュー #2160・PR #2280）。
/// 空バッチ（`N=0`）かつ空間次元が `i32::MAX` 超という極端形状で、
/// 両経路とも `IndexRangeOverflow` を返し契約が一致することを検証する
/// （修正前は host 経路のみ成功していた）。
#[test]
fn adaptive_max_pool2d_forward_host_rejects_index_overflow() {
    let huge_hw = i32::MAX as usize + 1;
    let ops = common::naive_ops();
    // N=0 のため data は空で構築でき、numel オーバーフローには
    // 当たらない（検査対象は H·W 自体の索引表現可能性）。
    let x = Tensor::<f32>::new(vec![], &[0, 1, huge_hw, 1])
        .expect("test fixture: N=0 のため空 data で shape と整合する");
    let layer = AdaptiveMaxPool2d::new([1, 1]).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let tape_err = <AdaptiveMaxPool2d as Module>::forward(&layer, &tape, &xv).unwrap_err();
    let host_err = layer.forward_host(ops.as_ref(), &x).unwrap_err();

    assert!(matches!(
        tape_err,
        AutodiffError::Shape(ShapeError::IndexRangeOverflow { index }) if index == huge_hw
    ));
    assert!(matches!(
        host_err,
        AutodiffError::Shape(ShapeError::IndexRangeOverflow { index }) if index == huge_hw
    ));
}

/// [`adaptive_max_pool2d_forward_host_rejects_index_overflow`] の
/// `AdaptiveMaxPool1d` 版（`l <= i32::MAX` 検査。イシュー #2160）。
#[test]
fn adaptive_max_pool1d_forward_host_rejects_index_overflow() {
    let huge_l = i32::MAX as usize + 1;
    let ops = common::naive_ops();
    let x = Tensor::<f32>::new(vec![], &[0, 1, huge_l])
        .expect("test fixture: N=0 のため空 data で shape と整合する");
    let layer = AdaptiveMaxPool1d::new(1).unwrap();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let tape_err = <AdaptiveMaxPool1d as Module>::forward(&layer, &tape, &xv).unwrap_err();
    let host_err = layer.forward_host(ops.as_ref(), &x).unwrap_err();

    assert!(matches!(
        tape_err,
        AutodiffError::Shape(ShapeError::IndexRangeOverflow { index }) if index == huge_l
    ));
    assert!(matches!(
        host_err,
        AutodiffError::Shape(ShapeError::IndexRangeOverflow { index }) if index == huge_l
    ));
}

/// [`adaptive_max_pool2d_forward_host_rejects_index_overflow`] の
/// `GlobalPool(Max)` 版（rank 3・rank 4 の両方。イシュー #2160）。
/// `GlobalPoolMode::Avg` は索引を持たないため対象外。
#[test]
fn global_pool_max_forward_host_rejects_index_overflow_rank3_and_rank4() {
    let huge_hw = i32::MAX as usize + 1;
    let ops = common::naive_ops();
    let layer = GlobalPool::new(GlobalPoolMode::Max, true);

    // rank 4: N=0 のため空 data のまま H·W が i32::MAX 超。
    let x4 = Tensor::<f32>::new(vec![], &[0, 1, huge_hw, 1])
        .expect("test fixture: N=0 のため空 data で shape と整合する");
    let tape4 = Tape::new_with_ops(common::naive_ops());
    let xv4 = tape4.var(&x4);
    let tape_err4 = <GlobalPool as Module>::forward(&layer, &tape4, &xv4).unwrap_err();
    let host_err4 = layer.forward_host(ops.as_ref(), &x4).unwrap_err();
    assert!(matches!(
        tape_err4,
        AutodiffError::Shape(ShapeError::IndexRangeOverflow { index }) if index == huge_hw
    ));
    assert!(matches!(
        host_err4,
        AutodiffError::Shape(ShapeError::IndexRangeOverflow { index }) if index == huge_hw
    ));

    // rank 3: N=0 のため空 data のまま L が i32::MAX 超。
    let x3 = Tensor::<f32>::new(vec![], &[0, 1, huge_hw])
        .expect("test fixture: N=0 のため空 data で shape と整合する");
    let tape3 = Tape::new_with_ops(common::naive_ops());
    let xv3 = tape3.var(&x3);
    let tape_err3 = <GlobalPool as Module>::forward(&layer, &tape3, &xv3).unwrap_err();
    let host_err3 = layer.forward_host(ops.as_ref(), &x3).unwrap_err();
    assert!(matches!(
        tape_err3,
        AutodiffError::Shape(ShapeError::IndexRangeOverflow { index }) if index == huge_hw
    ));
    assert!(matches!(
        host_err3,
        AutodiffError::Shape(ShapeError::IndexRangeOverflow { index }) if index == huge_hw
    ));
}

#[test]
fn global_pool_forward_rejects_rank_mismatch() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = t(vec![1.0, 2.0], &[2]); // rank 1（3 とも 4 とも異なる）
    let xv = tape.var(&x);
    let gp = GlobalPool::new(GlobalPoolMode::Avg, true);
    let err = gp.forward(&xv).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 4,
            actual: 1
        })
    ));
}

// --- 8. forward_host ≡ forward（bit 一致） ---

#[test]
fn adaptive_max_pool2d_forward_host_matches_forward() {
    let x = t(
        (0..16).map(|v| (v as f32 * 2.3).cos()).collect(),
        &[1, 1, 4, 4],
    );
    let layer = AdaptiveMaxPool2d::new([2, 2]).unwrap();
    let ops = common::naive_ops();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = <AdaptiveMaxPool2d as Module>::forward(&layer, &tape, &xv).unwrap();
    let via_host = layer.forward_host(ops.as_ref(), &x).unwrap();

    assert_bit_exact(
        "adaptive_max_pool2d module vs forward_host",
        &via_module.to_tensor(),
        &via_host,
    );
}

#[test]
fn adaptive_max_pool1d_forward_host_matches_forward() {
    let x = t(vec![1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0], &[1, 1, 7]);
    let layer = AdaptiveMaxPool1d::new(3).unwrap();
    let ops = common::naive_ops();

    let tape = Tape::new_with_ops(common::naive_ops());
    let xv = tape.var(&x);
    let via_module = <AdaptiveMaxPool1d as Module>::forward(&layer, &tape, &xv).unwrap();
    let via_host = layer.forward_host(ops.as_ref(), &x).unwrap();

    assert_bit_exact(
        "adaptive_max_pool1d module vs forward_host",
        &via_module.to_tensor(),
        &via_host,
    );
}

#[test]
fn global_pool_forward_host_matches_forward_rank3_and_rank4() {
    let ops = common::naive_ops();

    for mode in [GlobalPoolMode::Avg, GlobalPoolMode::Max] {
        for keepdims in [true, false] {
            let layer = GlobalPool::new(mode, keepdims);

            let x4 = t((0..24).map(|v| v as f32 * 0.5).collect(), &[1, 2, 3, 4]);
            let tape4 = Tape::new_with_ops(common::naive_ops());
            let xv4 = tape4.var(&x4);
            let via_module4 = <GlobalPool as Module>::forward(&layer, &tape4, &xv4).unwrap();
            let via_host4 = layer.forward_host(ops.as_ref(), &x4).unwrap();
            assert_bit_exact(
                "GlobalPool rank4 module vs forward_host",
                &via_module4.to_tensor(),
                &via_host4,
            );

            let x3 = t((0..12).map(|v| v as f32 * 0.5).collect(), &[2, 3, 2]);
            let tape3 = Tape::new_with_ops(common::naive_ops());
            let xv3 = tape3.var(&x3);
            let via_module3 = <GlobalPool as Module>::forward(&layer, &tape3, &xv3).unwrap();
            let via_host3 = layer.forward_host(ops.as_ref(), &x3).unwrap();
            assert_bit_exact(
                "GlobalPool rank3 module vs forward_host",
                &via_module3.to_tensor(),
                &via_host3,
            );
        }
    }
}

// --- 9. nn::Sequential 組み込み・named_parameters 空・is_pooling ---

#[test]
fn adaptive_max_global_pool_layers_have_no_named_parameters_and_are_pooling() {
    let amp2d = AdaptiveMaxPool2d::new([2, 2]).unwrap();
    let amp1d = AdaptiveMaxPool1d::new(2).unwrap();
    let gp = GlobalPool::new(GlobalPoolMode::Avg, true);
    assert!(amp2d.named_parameters().is_empty());
    assert!(amp1d.named_parameters().is_empty());
    assert!(gp.named_parameters().is_empty());
    assert!(Module::is_pooling(&amp2d));
    assert!(Module::is_pooling(&amp1d));
    assert!(Module::is_pooling(&gp));
}

#[test]
fn global_pool_works_inside_sequential() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let mut seq = Sequential::new();
    seq.push(Box::new(GlobalPool::new(GlobalPoolMode::Max, false)));
    let x = t((0..24).map(|v| v as f32).collect(), &[1, 2, 3, 4]);
    let xv = tape.var(&x);
    let y = seq.forward(&tape, &xv).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 2]);
}
