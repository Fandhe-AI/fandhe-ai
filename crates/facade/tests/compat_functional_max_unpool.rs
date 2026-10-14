//! `compat::FunctionalBuilder::{max_pool{1,2,3}d_with_indices, max_unpool{1,2,3}d}`（イシュー #3036・親 #3018。
//! 決定記録 `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §3・§6・§12）の CPU 統合テスト。
//! `fandhe_ai` だけを import する。
//!
//! MaxUnpool は入力が「特徴量」と「`MaxPool` の索引」の 2 つで `Module::forward` の単入力に載らないため、
//! `Sequential` には載せず `FunctionalBuilder` 限定で公開する（記録 §6 案 A）。索引は pool ノードの副スロットを
//! `eval_graph` が保持する。常駐経路（`DeviceParamStore`）は Functional に入口が無く、本テストの対象外
//! （記録 §3・§12。`Sequential` に `add_max_unpool*` が無いことは `lib.rs` の保留プローブと `api_surface.rs` が固定）。
//!
//! 検査項目:
//! - forward が `Tape` 上の `Var::max_pool*`／`Var::max_unpool*` 直呼びと bit 一致する（1d／2d／3d）
//! - SegNet 型（pool → ブロック → unpool）・fan-out・重なり窓の学習 1 step が手組みの backward + SGD と一致する
//! - `predict`／`evaluate` が `forward` と整合する
//! - 拒否系が panic せず型付きエラーになる
//! - 保存→復元の往復は #3037 で実装済み（旧「保存拒否」の固定は往復成功の検証へ置換。詳細は `compat_functional_model_io.rs`）

use fandhe_ai::compat::{
    FitConfig, FunctionalBuilder, FunctionalModel, Loss, Optimizer, Sequential,
    load_functional_model, save_functional_model,
};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai::{AutodiffError, Tensor, tape};

mod common;
use common::temp_dir::TempDirGuard;

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn values(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .to_vec()
}

/// 決定的な擬似データ（任意 rank）。
fn det(shape: &[usize], salt: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(
        (0..n).map(|k| ((k as f32) * 0.37 + salt).sin()).collect(),
        shape,
    )
    .expect("test fixture: shape とデータ長は一致させている")
}

/// REQ-2 統一複合判定。既存の `assert_parity` を再利用する（tolerance は不変）。
fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    fandhe_ai_backend_cpu::assert_parity(label, actual, expected);
}

fn conv_block(cin: usize, cout: usize, seed: u64) -> Sequential {
    Sequential::new()
        .add_conv2d(cin, cout, [3, 3], [1, 1], [1, 1], [1, 1], 1, seed)
        .unwrap()
}

/// forward 一致ケースの `(kernel, stride, padding, dilation, output_size)`。
type Case1d = (usize, Option<usize>, usize, usize, Option<usize>);
type Case2d = (
    [usize; 2],
    Option<[usize; 2]>,
    [usize; 2],
    [usize; 2],
    Option<[usize; 2]>,
);
type Case3d = (
    [usize; 3],
    Option<[usize; 3]>,
    [usize; 3],
    [usize; 3],
    Option<[usize; 3]>,
);

// ---- forward bit 一致 ----

#[test]
fn forward_1d_matches_var_direct_bit_for_bit() {
    let x = det(&[2, 3, 9], 0.1);
    // (kernel, stride, padding, dilation, output_size)
    let cases: [Case1d; 3] = [
        (2, None, 0, 1, None),
        (3, Some(2), 1, 1, Some(9)),
        (2, Some(1), 0, 1, None),
    ];
    for (k, s, p, d, out_size) in cases {
        let mut b = FunctionalBuilder::new();
        let i = b.input().unwrap();
        let pool = b.max_pool1d_with_indices(i, k, s, p, d).unwrap();
        let up = b.max_unpool1d(pool, pool, k, s, p, out_size).unwrap();
        let model = b.build(&[i], &[up]).unwrap();
        let got = model.predict(&[&x]).unwrap();

        let t = tape();
        let v = t.var(&x);
        let (pooled, idx) = v.max_pool1d(k, s, p, d, false).unwrap();
        let want = pooled.max_unpool1d(&idx, k, s, p, out_size).unwrap();
        assert_eq!(bits(&got[0]), bits(&want.to_tensor()), "1d k={k} s={s:?}");
    }
}

#[test]
fn forward_2d_matches_var_direct_bit_for_bit() {
    let x = det(&[2, 2, 8, 8], 0.2);
    let cases: [Case2d; 3] = [
        ([2, 2], None, [0, 0], [1, 1], None),
        ([3, 3], Some([2, 2]), [1, 1], [1, 1], Some([8, 8])),
        ([2, 3], Some([1, 2]), [0, 1], [1, 1], Some([8, 8])),
    ];
    for (k, s, p, d, out_size) in cases {
        let mut b = FunctionalBuilder::new();
        let i = b.input().unwrap();
        let pool = b.max_pool2d_with_indices(i, k, s, p, d).unwrap();
        let up = b.max_unpool2d(pool, pool, k, s, p, out_size).unwrap();
        let model = b.build(&[i], &[up]).unwrap();
        let got = model.predict(&[&x]).unwrap();

        let t = tape();
        let v = t.var(&x);
        let (pooled, idx) = v.max_pool2d(k, s, p, d, false).unwrap();
        let want = pooled.max_unpool2d(&idx, k, s, p, out_size).unwrap();
        assert_eq!(bits(&got[0]), bits(&want.to_tensor()), "2d k={k:?} s={s:?}");
    }
}

#[test]
fn forward_3d_matches_var_direct_bit_for_bit() {
    let x = det(&[1, 2, 6, 6, 6], 0.3);
    let cases: [Case3d; 2] = [
        ([2, 2, 2], None, [0, 0, 0], [1, 1, 1], None),
        (
            [3, 3, 3],
            Some([2, 2, 2]),
            [1, 1, 1],
            [1, 1, 1],
            Some([6, 6, 6]),
        ),
    ];
    for (k, s, p, d, out_size) in cases {
        let mut b = FunctionalBuilder::new();
        let i = b.input().unwrap();
        let pool = b.max_pool3d_with_indices(i, k, s, p, d).unwrap();
        let up = b.max_unpool3d(pool, pool, k, s, p, out_size).unwrap();
        let model = b.build(&[i], &[up]).unwrap();
        let got = model.predict(&[&x]).unwrap();

        let t = tape();
        let v = t.var(&x);
        let (pooled, idx) = v.max_pool3d(k, s, p, d, false).unwrap();
        let want = pooled.max_unpool3d(&idx, k, s, p, out_size).unwrap();
        assert_eq!(bits(&got[0]), bits(&want.to_tensor()), "3d k={k:?} s={s:?}");
    }
}

// ---- SegNet 型・fan-out・学習 1 step ----

/// SegNet 型: `x → conv → pool(idx) → conv → unpool(feature=後段, indices_from=pool)`。
fn segnet(kernel: [usize; 2], stride: Option<[usize; 2]>, pad: [usize; 2]) -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let h = b.apply(conv_block(1, 2, 0xA1), x).unwrap();
    let pool = b
        .max_pool2d_with_indices(h, kernel, stride, pad, [1, 1])
        .unwrap();
    let m = b.apply(conv_block(2, 2, 0xB2), pool).unwrap();
    let up = b.max_unpool2d(m, pool, kernel, stride, pad, None).unwrap();
    b.build(&[x], &[up]).unwrap()
}

/// `segnet` と同じ計算を `Tape` 上で手組みし、backward + 手計算 SGD（`p - lr * g`）で 1 step 進める。
/// 戻り値は `(forward 出力, 更新後パラメータ, 損失)`。
fn segnet_reference_step(
    kernel: [usize; 2],
    stride: Option<[usize; 2]>,
    pad: [usize; 2],
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    lr: f32,
) -> (Tensor<f32>, Vec<Tensor<f32>>, f32) {
    let a = conv_block(1, 2, 0xA1);
    let c = conv_block(2, 2, 0xB2);
    let t = tape();
    let xv = t.var(x);
    let yv = t.var(y);
    let (out, loss, grads) = {
        let ba = a.bind(&t);
        let bc = c.bind(&t);
        let h = ba.forward(&t, &xv).unwrap();
        let (pooled, idx) = h.max_pool2d(kernel, stride, pad, [1, 1], false).unwrap();
        let m = bc.forward(&t, &pooled).unwrap();
        let up = m.max_unpool2d(&idx, kernel, stride, pad, None).unwrap();
        let loss = up.mse_loss(&yv).unwrap();
        let grads = t.backward(&loss).unwrap();
        let mut g: Vec<Tensor<f32>> = Vec::new();
        for x in ba.trainable_grads(&grads).unwrap() {
            g.push(x.clone());
        }
        for x in bc.trainable_grads(&grads).unwrap() {
            g.push(x.clone());
        }
        (up.to_tensor(), loss.to_tensor(), g)
    };
    let params: Vec<&Tensor<f32>> = a
        .trainable_parameters()
        .into_iter()
        .chain(c.trainable_parameters())
        .collect();
    assert_eq!(params.len(), grads.len());
    let updated = params
        .iter()
        .zip(&grads)
        .map(|(p, g)| {
            let pv = values(p);
            let gv = values(g);
            let nv: Vec<f32> = pv.iter().zip(&gv).map(|(p, g)| p - lr * g).collect();
            Tensor::new(nv, p.shape()).unwrap()
        })
        .collect();
    (out, updated, values(&loss)[0])
}

fn assert_segnet_train_step_matches(
    kernel: [usize; 2],
    stride: Option<[usize; 2]>,
    pad: [usize; 2],
    hw: usize,
    label: &str,
) {
    let x = det(&[2, 1, hw, hw], 0.4);
    let y = det(&[2, 2, hw, hw], 0.9);
    let lr = 0.05f32;

    let (ref_out, ref_params, ref_loss) = segnet_reference_step(kernel, stride, pad, &x, &y, lr);

    let mut model = segnet(kernel, stride, pad);
    model.eval();
    let fwd = model.predict(&[&x]).unwrap();
    assert_eq!(bits(&fwd[0]), bits(&ref_out), "{label}: forward");
    let eval_loss = model.evaluate(&[&x], &[&y], 2);
    assert!(eval_loss.is_err(), "未 compile の evaluate は拒否される");

    model
        .compile(Optimizer::Sgd(SgdConfig::new(lr)), Loss::Mse)
        .unwrap();
    let ev = model.evaluate(&[&x], &[&y], 2).unwrap();
    assert_close(&format!("{label}: loss"), &[ev], &[ref_loss]);
    model.fit(&[&x], &[&y], FitConfig::new(1, 2)).unwrap();
    let got: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();
    assert_eq!(got.len(), ref_params.len());
    for (i, (g, r)) in got.iter().zip(&ref_params).enumerate() {
        assert_close(&format!("{label}: param {i}"), &values(g), &values(r));
    }
}

#[test]
fn segnet_train_step_matches_handwritten_backward_sgd() {
    assert_segnet_train_step_matches([2, 2], None, [0, 0], 8, "segnet 2x2");
}

#[test]
fn overlapping_window_train_step_matches_handwritten_backward_sgd() {
    // stride < kernel の重なり窓（重複索引の勾配規則は決定記録 §12.2(a) のまま）。
    assert_segnet_train_step_matches([3, 3], Some([2, 2]), [1, 1], 7, "overlap 3x3/2");
}

#[test]
fn fan_out_pool_indices_can_feed_multiple_unpools() {
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let pool = b
        .max_pool2d_with_indices(x, [2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let u1 = b
        .max_unpool2d(pool, pool, [2, 2], None, [0, 0], None)
        .unwrap();
    let m = b.apply(Sequential::new().add_relu(), pool).unwrap();
    let u2 = b.max_unpool2d(m, pool, [2, 2], None, [0, 0], None).unwrap();
    let model = b.build(&[x], &[u1, u2]).unwrap();
    let input = det(&[1, 1, 4, 4], 0.6);
    let outs = model.predict(&[&input]).unwrap();
    assert_eq!(outs.len(), 2);

    let t = tape();
    let v = t.var(&input);
    let (pooled, idx) = v.max_pool2d([2, 2], None, [0, 0], [1, 1], false).unwrap();
    let w1 = pooled
        .max_unpool2d(&idx, [2, 2], None, [0, 0], None)
        .unwrap();
    let w2 = pooled
        .relu()
        .max_unpool2d(&idx, [2, 2], None, [0, 0], None)
        .unwrap();
    assert_eq!(bits(&outs[0]), bits(&w1.to_tensor()));
    assert_eq!(bits(&outs[1]), bits(&w2.to_tensor()));
}

#[test]
fn pool_contributing_only_through_indices_builds() {
    // pool の値は未使用で、索引だけが unpool へ渡るグラフ。`build` の到達性は `sources()` 経由で通る。
    // 特徴量は pool 後の shape（[N,1,2,2]）に揃えた別経路（pool を 2 回通す）から作る。
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let pool = b
        .max_pool2d_with_indices(x, [2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let side = b
        .max_pool2d_with_indices(x, [2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let up = b
        .max_unpool2d(side, pool, [2, 2], None, [0, 0], None)
        .unwrap();
    let model = b.build(&[x], &[up]).unwrap();
    let input = det(&[1, 1, 4, 4], 0.5);
    let got = model.predict(&[&input]).unwrap();
    assert_eq!(got[0].shape(), &[1, 1, 4, 4]);
}

// ---- 拒否系 ----

#[test]
fn builder_rejects_invalid_wiring_with_typed_errors() {
    let mut other = FunctionalBuilder::new();
    let foreign = other.input().unwrap();

    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let pool2 = b
        .max_pool2d_with_indices(x, [2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let blk = b.apply(Sequential::new().add_relu(), pool2).unwrap();

    // 他ビルダー由来
    assert!(
        b.max_pool2d_with_indices(foreign, [2, 2], None, [0, 0], [1, 1])
            .is_err()
    );
    assert!(
        b.max_unpool2d(foreign, pool2, [2, 2], None, [0, 0], None)
            .is_err()
    );
    assert!(
        b.max_unpool2d(blk, foreign, [2, 2], None, [0, 0], None)
            .is_err()
    );
    // indices_from が pool ノードでない（Input／Block）
    assert!(b.max_unpool2d(blk, x, [2, 2], None, [0, 0], None).is_err());
    assert!(
        b.max_unpool2d(blk, blk, [2, 2], None, [0, 0], None)
            .is_err()
    );
    // 次元不一致（2d pool に 1d／3d unpool）
    assert!(b.max_unpool1d(blk, pool2, 2, None, 0, None).is_err());
    assert!(
        b.max_unpool3d(blk, pool2, [2, 2, 2], None, [0, 0, 0], None)
            .is_err()
    );
    // 0 次元・padding の 2 倍オーバーフロー
    assert!(
        b.max_unpool2d(blk, pool2, [0, 2], None, [0, 0], None)
            .is_err()
    );
    assert!(
        b.max_unpool2d(blk, pool2, [2, 2], Some([0, 1]), [0, 0], None)
            .is_err()
    );
    assert!(
        b.max_unpool2d(blk, pool2, [2, 2], None, [usize::MAX, 0], None)
            .is_err()
    );
}

#[test]
fn forward_rejects_bad_indices_without_panicking() {
    // 索引 shape 不整合: 特徴量 [N,C,1,1]（k=4 の pool）に [N,C,2,2] の索引（k=2 の pool）を組み合わせる。
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let p2 = b
        .max_pool2d_with_indices(x, [2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let p4 = b
        .max_pool2d_with_indices(x, [4, 4], None, [0, 0], [1, 1])
        .unwrap();
    let bad = b.max_unpool2d(p4, p2, [4, 4], None, [0, 0], None).unwrap();
    let model = b.build(&[x], &[bad, p2]).unwrap();
    let input = det(&[1, 1, 4, 4], 0.1);
    assert!(model.predict(&[&input]).is_err());

    // 索引範囲外: 元の 4x4 の索引を output_size=[2, 2] の復元先へ書き込む。
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let p = b
        .max_pool2d_with_indices(x, [2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let up = b
        .max_unpool2d(p, p, [2, 2], None, [0, 0], Some([2, 2]))
        .unwrap();
    let model = b.build(&[x], &[up]).unwrap();
    assert!(model.predict(&[&input]).is_err());
}

#[test]
fn forward_rejects_input_from_another_tape() {
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let p = b.max_pool1d_with_indices(x, 2, None, 0, 1).unwrap();
    let up = b.max_unpool1d(p, p, 2, None, 0, None).unwrap();
    let model = b.build(&[x], &[up]).unwrap();
    let t1 = tape();
    let t2 = tape();
    let v = t2.var(&det(&[1, 1, 4], 0.0));
    assert!(matches!(
        model.forward(&t1, &[v]),
        Err(AutodiffError::TapeMismatch)
    ));
}

// ---- 保存→復元（#3037。旧「保存は fail-closed」の固定を往復成功へ置換） ----

#[test]
fn save_then_load_round_trips_max_pool_unpool_graph() {
    let mut model = segnet([2, 2], None, [0, 0]);
    model.eval();
    let guard = TempDirGuard::new("functional_max_unpool_save");
    let dir = guard.path().join("m");
    save_functional_model(&model, &dir).unwrap();
    let loaded = load_functional_model(&dir).unwrap();
    let x = det(&[1, 1, 4, 4], 0.3);
    let a = model.predict(&[&x]).unwrap();
    let b = loaded.predict(&[&x]).unwrap();
    assert_eq!(bits(&a[0]), bits(&b[0]));
}
