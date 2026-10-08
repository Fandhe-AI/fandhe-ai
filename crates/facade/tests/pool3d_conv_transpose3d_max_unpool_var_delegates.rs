//! `Var::max_pool3d`／`avg_pool3d`／`conv_transpose3d`／`max_unpool1d/2d/3d` の facade 利用テスト
//! （イシュー #2850 の公開形。`docs/autodiff-pool3d-ops-decision.md` §12.1・
//! `docs/autodiff-conv-transpose3d-max-unpool-decision.md` §12.1・`docs/compat-api-scope.md` §5.1 行 12・13。
//! 承認根拠は ルート #2499 の `issuecomment-6052732061`）。
//!
//! 利用者が見る面は `fandhe_ai` だけ（`AutodiffError`・`Tape`・`Tensor`・`Var`）。委譲メソッドの
//! forward 値・索引・backward（勾配）が、内部クレートの自由関数
//! （`fandhe_ai_autodiff::{pool3d_ops, conv_transpose3d_ops, max_unpool_ops}`。比較の参照としてのみ使う）と
//! bit 一致することを固定する。委譲先は同一コードのため新しい tolerance・baseline は設けない
//! （グローバル RNG は使わず、入力は固定値。形状は CI の test 枠に収まる小ささに保つ）。
//! 実機（CUDA／Metal）の一致は既存の `pool3d_ops_backend_parity.rs`・
//! `conv_transpose3d_max_unpool_backend_parity.rs`（`#[ignore]`。#2850 以降は `Var` メソッド経由）が担う。
//! 層化（`nn::*` 層型・`compat::Sequential::add_*`）は保留継続のため本テストの対象外。

// 公開シグネチャを fn ポインタ型で 1 引数ずつ固定するため、型が長くなるのは意図通り。
#![allow(clippy::type_complexity)]

use fandhe_ai::{AutodiffError, Tape, Tensor, Var};

type R<'t> = Result<Var<'t>, AutodiffError>;
type Idx = Tensor<i32>;

/// シグネチャ固定用（`Var<'t>` の `'t` を名指しするため関数で包む）。
fn sig_max_pool3d<'t>() -> fn(
    &Var<'t>,
    [usize; 3],
    Option<[usize; 3]>,
    [usize; 3],
    [usize; 3],
    bool,
) -> Result<(Var<'t>, Idx), AutodiffError> {
    Var::<'t>::max_pool3d
}
fn sig_avg_pool3d<'t>()
-> fn(&Var<'t>, [usize; 3], Option<[usize; 3]>, [usize; 3], bool, bool) -> R<'t> {
    Var::<'t>::avg_pool3d
}
fn sig_conv_transpose3d<'t>() -> fn(
    &Var<'t>,
    &Var<'t>,
    Option<&Var<'t>>,
    [usize; 3],
    [usize; 3],
    [usize; 3],
    [usize; 3],
    usize,
) -> R<'t> {
    Var::<'t>::conv_transpose3d
}
fn sig_max_unpool1d<'t>() -> fn(&Var<'t>, &Idx, usize, Option<usize>, usize, Option<usize>) -> R<'t>
{
    Var::<'t>::max_unpool1d
}
fn sig_max_unpool2d<'t>()
-> fn(&Var<'t>, &Idx, [usize; 2], Option<[usize; 2]>, [usize; 2], Option<[usize; 2]>) -> R<'t> {
    Var::<'t>::max_unpool2d
}
fn sig_max_unpool3d<'t>()
-> fn(&Var<'t>, &Idx, [usize; 3], Option<[usize; 3]>, [usize; 3], Option<[usize; 3]>) -> R<'t> {
    Var::<'t>::max_unpool3d
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// 決定的な非自明入力（符号と大きさが揺れる三角波。重複最大値を避ける微小項付き）。
fn wave(n: usize, step: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * step + offset).sin() * 2.0) + (i as f32) * 1e-3)
        .collect()
}

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("tensor")
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// 出力に固定の重み `g` を掛けて総和した損失の勾配（forward 出力・各入力の勾配）を取り出す。
fn run1<'t>(
    tape: &'t Tape,
    x: &Tensor<f32>,
    f: impl Fn(&Var<'t>) -> R<'t>,
) -> (Tensor<f32>, Tensor<f32>) {
    let xv = tape.var(x);
    let y = f(&xv).expect("op");
    let out = y.to_tensor();
    let g = tape.var(&t(wave(numel(out.shape()), 0.37, 0.4), out.shape()));
    let loss = y.mul(&g).expect("mul").sum(None).expect("sum");
    let grads = tape.backward(&loss).expect("backward");
    let dx = grads.get(&xv).expect("get").expect("grad").clone();
    (out, dx)
}

fn assert_same(a: &(Tensor<f32>, Tensor<f32>), b: &(Tensor<f32>, Tensor<f32>), what: &str) {
    assert_eq!(a.0.shape(), b.0.shape(), "{what}: forward shape");
    assert_eq!(bits(&a.0), bits(&b.0), "{what}: forward bit 一致");
    assert_eq!(bits(&a.1), bits(&b.1), "{what}: backward bit 一致");
}

const X3: [usize; 5] = [1, 2, 4, 4, 4];

#[test]
fn signatures_match_approved_form() {
    // 取得できること自体が、引数順・戻り値型の固定になる。
    let _ = sig_max_pool3d();
    let _ = sig_avg_pool3d();
    let _ = sig_conv_transpose3d();
    let _ = sig_max_unpool1d();
    let _ = sig_max_unpool2d();
    let _ = sig_max_unpool3d();
}

#[test]
fn max_pool3d_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::pool3d_ops::max_pool3d;
    let x = t(wave(numel(&X3), 0.29, 0.3), &X3);
    let tape = fandhe_ai::tape();
    let via_method = run1(&tape, &x, |v| {
        v.max_pool3d([2, 2, 2], None, [0, 0, 0], [1, 1, 1], false)
            .map(|(y, _)| y)
    });
    let via_free = run1(&tape, &x, |v| {
        max_pool3d(v, [2, 2, 2], None, [0, 0, 0], [1, 1, 1], false).map(|(y, _)| y)
    });
    assert_same(&via_method, &via_free, "max_pool3d");

    // padding・stride・dilation 付きでも索引が完全一致する。
    let xv = tape.var(&x);
    let (ym, im) = xv
        .max_pool3d([2, 2, 2], Some([1, 2, 1]), [1, 0, 1], [1, 1, 1], false)
        .expect("method");
    let (yf, i_f) =
        max_pool3d(&xv, [2, 2, 2], Some([1, 2, 1]), [1, 0, 1], [1, 1, 1], false).expect("free");
    assert_eq!(
        bits(&ym.to_tensor()),
        bits(&yf.to_tensor()),
        "max_pool3d 値"
    );
    assert_eq!(im.shape(), i_f.shape(), "max_pool3d 索引 shape");
    assert_eq!(
        im.contiguous().host_slice(),
        i_f.contiguous().host_slice(),
        "max_pool3d 索引"
    );
}

#[test]
fn avg_pool3d_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::pool3d_ops::avg_pool3d;
    let x = t(wave(numel(&X3), 0.31, 0.2), &X3);
    let tape = fandhe_ai::tape();
    for count_include_pad in [true, false] {
        let via_method = run1(&tape, &x, |v| {
            v.avg_pool3d(
                [2, 2, 2],
                Some([1, 1, 2]),
                [1, 0, 1],
                false,
                count_include_pad,
            )
        });
        let via_free = run1(&tape, &x, |v| {
            avg_pool3d(
                v,
                [2, 2, 2],
                Some([1, 1, 2]),
                [1, 0, 1],
                false,
                count_include_pad,
            )
        });
        assert_same(&via_method, &via_free, "avg_pool3d");
    }
}

#[test]
fn avg_pool3d_closed_form_value() {
    // [1,1,2,2,2] の値 1..=8 を 2×2×2 窓で平均 = 4.5。
    let x = t((1..=8).map(|v| v as f32).collect(), &[1, 1, 2, 2, 2]);
    let tape = fandhe_ai::tape();
    let y = tape
        .var(&x)
        .avg_pool3d([2, 2, 2], None, [0, 0, 0], false, true)
        .expect("avg");
    let out = y.to_tensor();
    assert_eq!(out.shape(), &[1, 1, 1, 1, 1]);
    assert_eq!(&*out.contiguous().host_slice(), &[4.5]);
}

#[test]
fn conv_transpose3d_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::conv_transpose3d_ops::conv_transpose3d;
    let xs = [1usize, 2, 2, 2, 3];
    let ws = [2usize, 3, 2, 2, 2];
    let x = t(wave(numel(&xs), 0.43, 1.0), &xs);
    let w = t(wave(numel(&ws), 0.61, 0.5), &ws);
    let b = t(vec![0.1, -0.2, 0.3], &[3]);
    let args = ([2, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], 1usize);

    let go = |via_method: bool| -> (Tensor<f32>, Tensor<f32>, Tensor<f32>, Tensor<f32>) {
        let tape = fandhe_ai::tape();
        let xv = tape.var(&x);
        let wv = tape.var(&w);
        let bv = tape.var(&b);
        let y = if via_method {
            xv.conv_transpose3d(&wv, Some(&bv), args.0, args.1, args.2, args.3, args.4)
        } else {
            conv_transpose3d(&xv, &wv, Some(&bv), args.0, args.1, args.2, args.3, args.4)
        }
        .expect("op");
        let out = y.to_tensor();
        let g = tape.var(&t(wave(numel(out.shape()), 0.53, 0.7), out.shape()));
        let loss = y.mul(&g).expect("mul").sum(None).expect("sum");
        let grads = tape.backward(&loss).expect("backward");
        (
            out,
            grads.get(&xv).expect("get").expect("dx").clone(),
            grads.get(&wv).expect("get").expect("dw").clone(),
            grads.get(&bv).expect("get").expect("db").clone(),
        )
    };
    let m = go(true);
    let f = go(false);
    assert_eq!(m.0.shape(), f.0.shape(), "conv_transpose3d forward shape");
    assert_eq!(bits(&m.0), bits(&f.0), "conv_transpose3d forward");
    assert_eq!(bits(&m.1), bits(&f.1), "conv_transpose3d d_input");
    assert_eq!(bits(&m.2), bits(&f.2), "conv_transpose3d d_weight");
    assert_eq!(bits(&m.3), bits(&f.3), "conv_transpose3d d_bias");
}

#[test]
fn conv_transpose3d_closed_form_value() {
    // 単一要素 2.0 に全 1 の 2×2×2 カーネルを転置畳み込み → 全要素 2.0（+ bias 0.5）。
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![2.0], &[1, 1, 1, 1, 1]));
    let w = tape.var(&t(vec![1.0; 8], &[1, 1, 2, 2, 2]));
    let b = tape.var(&t(vec![0.5], &[1]));
    let y = x
        .conv_transpose3d(&w, Some(&b), [1, 1, 1], [0, 0, 0], [0, 0, 0], [1, 1, 1], 1)
        .expect("ct3d")
        .to_tensor();
    assert_eq!(y.shape(), &[1, 1, 2, 2, 2]);
    assert_eq!(&*y.contiguous().host_slice(), &[2.5; 8]);
}

#[test]
fn max_unpool_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::max_unpool_ops::{max_unpool1d, max_unpool2d, max_unpool3d};
    let tape = fandhe_ai::tape();

    // 1d: 重複索引（平面 3 の 2 と 2）を含む。
    let i1 = Tensor::new(vec![1, 0, 5, 2, 4, 3, 1, 1, 5, 0, 2, 2], &[2, 2, 3]).expect("idx");
    let x1 = t(wave(12, 0.7, 0.1), &[2, 2, 3]);
    assert_same(
        &run1(&tape, &x1, |v| v.max_unpool1d(&i1, 2, None, 0, None)),
        &run1(&tape, &x1, |v| max_unpool1d(v, &i1, 2, None, 0, None)),
        "max_unpool1d",
    );

    // 2d: 出力平面 4×6 = 24。
    let i2 = Tensor::new(
        vec![2, 9, 16, 23, 6, 13, 0, 0, 5, 23, 11, 12],
        &[1, 2, 2, 3],
    )
    .expect("idx");
    let x2 = t(wave(12, 0.5, 0.3), &[1, 2, 2, 3]);
    assert_same(
        &run1(&tape, &x2, |v| {
            v.max_unpool2d(&i2, [2, 2], None, [0, 0], None)
        }),
        &run1(&tape, &x2, |v| {
            max_unpool2d(v, &i2, [2, 2], None, [0, 0], None)
        }),
        "max_unpool2d",
    );

    // 3d: 出力平面 4×4×4 = 64。
    let i3 = Tensor::new(vec![0, 63, 21, 42, 7, 35, 56, 14], &[1, 1, 2, 2, 2]).expect("idx");
    let x3 = t(wave(8, 0.9, 0.2), &[1, 1, 2, 2, 2]);
    assert_same(
        &run1(&tape, &x3, |v| {
            v.max_unpool3d(&i3, [2, 2, 2], None, [0, 0, 0], None)
        }),
        &run1(&tape, &x3, |v| {
            max_unpool3d(v, &i3, [2, 2, 2], None, [0, 0, 0], None)
        }),
        "max_unpool3d",
    );
}

#[test]
fn max_unpool1d_closed_form_placement() {
    // 値 [10, 20] を索引 [3, 0] へ散布 → [20, 0, 0, 10]。
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![10.0, 20.0], &[1, 1, 2]));
    let idx = Tensor::new(vec![3, 0], &[1, 1, 2]).expect("idx");
    let y = x
        .max_unpool1d(&idx, 2, None, 0, None)
        .expect("unpool")
        .to_tensor();
    assert_eq!(y.shape(), &[1, 1, 4]);
    assert_eq!(&*y.contiguous().host_slice(), &[20.0, 0.0, 0.0, 10.0]);
}

#[test]
fn pool_indices_round_trip_through_unpool_on_facade_only() {
    // `max_pool{1,2,3}d` の索引をそのまま `max_unpool{1,2,3}d` へ渡せる（facade だけで書ける）。
    let tape = fandhe_ai::tape();

    let x3 = tape.var(&t(wave(numel(&X3), 0.29, 0.3), &X3));
    let (p3, i3) = x3
        .max_pool3d([2, 2, 2], None, [0, 0, 0], [1, 1, 1], false)
        .expect("pool3d");
    let u3 = p3
        .max_unpool3d(&i3, [2, 2, 2], None, [0, 0, 0], None)
        .expect("unpool3d");
    assert_eq!(u3.to_tensor().shape(), &X3);

    let x2 = tape.var(&t(wave(2 * 4 * 4, 0.33, 0.1), &[1, 2, 4, 4]));
    let (p2, i2) = x2
        .max_pool2d([2, 2], None, [0, 0], [1, 1], false)
        .expect("pool2d");
    let u2 = p2
        .max_unpool2d(&i2, [2, 2], None, [0, 0], None)
        .expect("unpool2d");
    assert_eq!(u2.to_tensor().shape(), &[1, 2, 4, 4]);

    let x1 = tape.var(&t(wave(2 * 6, 0.41, 0.6), &[1, 2, 6]));
    let (p1, i1) = x1.max_pool1d(2, None, 0, 1, false).expect("pool1d");
    let u1 = p1.max_unpool1d(&i1, 2, None, 0, None).expect("unpool1d");
    assert_eq!(u1.to_tensor().shape(), &[1, 2, 6]);
}

#[test]
fn typed_errors_for_rejected_arguments() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(wave(numel(&X3), 0.29, 0.3), &X3));

    // ceil_mode = true は型付きエラー（両プーリング）。
    assert!(
        x.max_pool3d([2, 2, 2], None, [0, 0, 0], [1, 1, 1], true)
            .is_err()
    );
    assert!(
        x.avg_pool3d([2, 2, 2], None, [0, 0, 0], true, true)
            .is_err()
    );

    // output_padding >= stride は拒否。
    let xc = tape.var(&t(vec![1.0], &[1, 1, 1, 1, 1]));
    let w = tape.var(&t(vec![1.0; 8], &[1, 1, 2, 2, 2]));
    let r = xc.conv_transpose3d(&w, None, [1, 1, 1], [0, 0, 0], [1, 0, 0], [1, 1, 1], 1);
    assert!(r.is_err(), "output_padding >= stride は拒否される");

    // 別 tape の weight は TapeMismatch。
    let other = fandhe_ai::tape();
    let w_other = other.var(&t(vec![1.0; 8], &[1, 1, 2, 2, 2]));
    let r = xc.conv_transpose3d(
        &w_other,
        None,
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        1,
    );
    assert!(matches!(r, Err(AutodiffError::TapeMismatch)));

    // kernel = 0 は拒否（MaxUnpool）。
    let xu = tape.var(&t(vec![1.0, 2.0], &[1, 1, 2]));
    let idx = Tensor::new(vec![0, 1], &[1, 1, 2]).expect("idx");
    assert!(xu.max_unpool1d(&idx, 0, None, 0, None).is_err());
}

#[test]
fn zero_channel_is_accepted_and_yields_empty_output() {
    // 記録 §12.2(c): `C = 0` は受理して空出力（PyTorch は拒否。`N = 0` と一貫させた差分）。
    let tape = fandhe_ai::tape();
    let x = tape.var(&t(vec![], &[1, 0, 2, 2, 2]));
    let y = x
        .avg_pool3d([2, 2, 2], None, [0, 0, 0], false, true)
        .expect("C=0 avg_pool3d");
    assert_eq!(y.to_tensor().shape(), &[1, 0, 1, 1, 1]);

    let idx = Tensor::new(vec![], &[1, 0, 2, 2, 2]).expect("idx");
    let u = x
        .max_unpool3d(&idx, [2, 2, 2], None, [0, 0, 0], None)
        .expect("C=0 max_unpool3d");
    assert_eq!(u.to_tensor().shape(), &[1, 0, 4, 4, 4]);
}
