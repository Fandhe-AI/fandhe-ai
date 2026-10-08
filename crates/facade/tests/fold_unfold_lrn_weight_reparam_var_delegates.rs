//! `Var::unfold`／`fold`／`local_response_norm`／`weight_norm`／`spectral_norm` の facade 利用テスト
//! （イシュー #2851 の公開形。`docs/autodiff-fold-unfold-decision.md` §12.1・
//! `docs/autodiff-lrn-weight-reparam-decision.md` §12.1・`docs/compat-api-scope.md` §5.1 行 14・15。
//! 承認根拠は ルート #2499 の `issuecomment-6052732061`）。
//!
//! 利用者が見る面は `fandhe_ai` だけ（`AutodiffError`・`SpectralNormState`・`Tape`・`Tensor`・`Var`）。
//! 委譲メソッドの forward 値・勾配（・`spectral_norm` の更新後状態）が、内部クレートの自由関数
//! （`fandhe_ai_autodiff::{fold_ops, lrn_ops, weight_reparam_ops}`。比較の参照としてのみ使う）と
//! bit 一致することを固定する。委譲先は同一コードのため新しい tolerance・baseline は設けない
//! （グローバル RNG は使わず、入力は固定値。形状は CI の test 枠に収まる小ささに保つ）。
//! 実機（CUDA／Metal）の一致は既存の `fold_unfold_backend_parity.rs`・`lrn_weight_reparam_backend_parity.rs`
//! （`#[ignore]`。#2851 以降は `Var` メソッド経由）が担う。
//! 層化（`nn::*` 層型・`compat::Sequential::add_*`）・重み再パラメータ化の結線方式・`norm_except_dim` の
//! 公開は保留継続のため本テストの対象外（`g` の初期値は本テスト内で自前計算する）。

// 公開シグネチャを fn ポインタ型で 1 引数ずつ固定するため、型が長くなるのは意図通り。
#![allow(clippy::type_complexity)]

use fandhe_ai::{AutodiffError, SpectralNormState, Tape, Tensor, Var};

type R<'t> = Result<Var<'t>, AutodiffError>;

fn sig_unfold<'t>() -> fn(&Var<'t>, [usize; 2], [usize; 2], [usize; 2], [usize; 2]) -> R<'t> {
    Var::<'t>::unfold
}
fn sig_fold<'t>()
-> fn(&Var<'t>, [usize; 2], [usize; 2], [usize; 2], [usize; 2], [usize; 2]) -> R<'t> {
    Var::<'t>::fold
}
fn sig_lrn<'t>() -> fn(&Var<'t>, usize, f32, f32, f32) -> R<'t> {
    Var::<'t>::local_response_norm
}
fn sig_weight_norm<'t>() -> fn(&Var<'t>, &Var<'t>, Option<usize>) -> R<'t> {
    Var::<'t>::weight_norm
}
fn sig_spectral_norm<'t>() -> fn(&Var<'t>, &mut SpectralNormState, bool) -> R<'t> {
    Var::<'t>::spectral_norm
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn vec_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// 決定的な非自明入力（符号と大きさが揺れる三角波）。
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

/// 出力に固定の重みを掛けて総和した損失の勾配（forward 出力・入力の勾配）を取り出す。
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

#[test]
fn signatures_match_approved_form() {
    // 取得できること自体が、引数順・戻り値型の固定になる（`SpectralNormState` のルート再エクスポートの到達性も兼ねる）。
    let _ = sig_unfold();
    let _ = sig_fold();
    let _ = sig_lrn();
    let _ = sig_weight_norm();
    let _ = sig_spectral_norm();
}

#[test]
fn unfold_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::fold_ops::unfold;
    let tape = fandhe_ai::tape();
    // (入力 shape, kernel, stride, padding, dilation)。重なりあり・なし・非自明な padding／dilation／stride。
    let cases: [([usize; 4], [usize; 2], [usize; 2], [usize; 2], [usize; 2]); 3] = [
        ([1, 2, 4, 4], [2, 2], [1, 1], [0, 0], [1, 1]),
        ([2, 3, 6, 7], [2, 3], [1, 2], [1, 1], [2, 1]),
        ([1, 1, 5, 5], [3, 3], [3, 3], [0, 0], [1, 1]),
    ];
    for (shape, k, s, p, d) in cases {
        let x = t(wave(numel(&shape), 0.29, 0.3), &shape);
        let via_method = run1(&tape, &x, |v| v.unfold(k, s, p, d));
        let via_free = run1(&tape, &x, |v| unfold(v, k, s, p, d));
        assert_same(&via_method, &via_free, "unfold");
    }
}

#[test]
fn fold_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::fold_ops::fold;
    let tape = fandhe_ai::tape();
    // 重なる窓（kernel [2,3]・stride [1,2]・padding [1,1]・dilation [2,1]）。出力 [2,3,6,7] 向けの列は [2, 18, 24]。
    let (k, s, p, d, out) = ([2, 3], [1, 2], [1, 1], [2, 1], [6, 7]);
    let shape = [2usize, 18, 24];
    let x = t(wave(numel(&shape), 0.17, 0.5), &shape);
    let via_method = run1(&tape, &x, |v| v.fold(out, k, s, p, d));
    let via_free = run1(&tape, &x, |v| fold(v, out, k, s, p, d));
    assert_same(&via_method, &via_free, "fold");
    assert_eq!(via_method.0.shape(), &[2, 3, 6, 7]);
}

#[test]
fn unfold_window_order_and_fold_overlap_closed_form() {
    let tape = fandhe_ai::tape();
    let x = t((1..=9).map(|v| v as f32).collect(), &[1, 1, 3, 3]);
    let xv = tape.var(&x);
    let cols = xv.unfold([2, 2], [1, 1], [0, 0], [1, 1]).expect("unfold");
    let out = cols.to_tensor();
    // PyTorch 順 `[N, C·kH·kW, L]`: 行 = カーネル位置、列 = 窓位置。
    assert_eq!(out.shape(), &[1, 4, 4]);
    let got = out.contiguous().host_slice().to_vec();
    assert_eq!(
        got,
        vec![
            1.0, 2.0, 4.0, 5.0, // (0,0)
            2.0, 3.0, 5.0, 6.0, // (0,1)
            4.0, 5.0, 7.0, 8.0, // (1,0)
            5.0, 6.0, 8.0, 9.0, // (1,1)
        ]
    );
    // fold(unfold(x)) は各画素が含まれる窓の数だけ倍になる。
    let back = cols
        .fold([3, 3], [2, 2], [1, 1], [0, 0], [1, 1])
        .expect("fold")
        .to_tensor();
    let counts = [1.0f32, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    let want: Vec<f32> = (1..=9)
        .map(|v| v as f32)
        .zip(counts)
        .map(|(a, c)| a * c)
        .collect();
    assert_eq!(back.contiguous().host_slice().to_vec(), want);
}

#[test]
fn local_response_norm_matches_free_function_bit_for_bit() {
    use fandhe_ai_autodiff::lrn_ops::local_response_norm;
    let tape = fandhe_ai::tape();
    let shape = [2usize, 7, 3, 2];
    let x = t(wave(numel(&shape), 0.043, 2.0), &shape);
    // 奇数・偶数 size。
    for size in [3usize, 4] {
        let via_method = run1(&tape, &x, |v| v.local_response_norm(size, 0.3, 0.75, 1.5));
        let via_free = run1(&tape, &x, |v| local_response_norm(v, size, 0.3, 0.75, 1.5));
        assert_same(&via_method, &via_free, "local_response_norm");
    }
}

/// `dim` 軸以外の L2 ノルム（`dim = None` は rank 0）を keepdim 形で自前計算する（決定記録 §12.3 の `g` shape 規約）。
fn own_norm_g(v: &Tensor<f32>, dim: Option<usize>) -> Tensor<f32> {
    let shape = v.shape().to_vec();
    let data = v.contiguous().host_slice().to_vec();
    match dim {
        None => t(vec![data.iter().map(|x| x * x).sum::<f32>().sqrt()], &[]),
        Some(d) => {
            let n = shape[d];
            let inner: usize = shape[d + 1..].iter().product();
            let mut acc = vec![0.0f32; n];
            for (i, x) in data.iter().enumerate() {
                acc[(i / inner) % n] += x * x;
            }
            let mut gshape = vec![1usize; shape.len()];
            gshape[d] = n;
            t(acc.into_iter().map(f32::sqrt).collect(), &gshape)
        }
    }
}

#[test]
fn weight_norm_matches_free_function_bit_for_bit_and_identity_when_g_is_norm() {
    use fandhe_ai_autodiff::weight_reparam_ops::weight_norm;
    let tape = fandhe_ai::tape();
    let shape = [4usize, 3, 5];
    let v0 = t(wave(numel(&shape), 0.071, 1.0), &shape);
    for dim in [Some(0usize), None] {
        let g0 = own_norm_g(&v0, dim);
        let g_scaled = t(
            g0.contiguous()
                .host_slice()
                .iter()
                .map(|x| x * 1.3)
                .collect(),
            g0.shape(),
        );
        let up = t(wave(numel(&shape), 0.037, 0.9), &shape);
        let grads_of = |use_method: bool| {
            let v = tape.var(&v0);
            let g = tape.var(&g_scaled);
            let w = if use_method {
                v.weight_norm(&g, dim)
            } else {
                weight_norm(&v, &g, dim)
            }
            .expect("weight_norm");
            let out = w.to_tensor();
            let loss = w.mul(&tape.var(&up)).expect("mul").sum(None).expect("sum");
            let grads = tape.backward(&loss).expect("backward");
            let dv = grads.get(&v).expect("get").expect("dv").clone();
            let dg = grads.get(&g).expect("get").expect("dg").clone();
            (out, dv, dg)
        };
        let (m, f) = (grads_of(true), grads_of(false));
        assert_eq!(bits(&m.0), bits(&f.0), "weight_norm dim={dim:?}: forward");
        assert_eq!(bits(&m.1), bits(&f.1), "weight_norm dim={dim:?}: dv");
        assert_eq!(bits(&m.2), bits(&f.2), "weight_norm dim={dim:?}: dg");
    }

    // g = ‖v‖ なら w は v に（丸め誤差の範囲で）一致する。`g` は利用者が自前で用意する。
    let v = tape.var(&v0);
    let g = tape.var(&own_norm_g(&v0, Some(0)));
    let w = v.weight_norm(&g, Some(0)).expect("weight_norm").to_tensor();
    for (a, b) in w
        .contiguous()
        .host_slice()
        .iter()
        .zip(v0.host_slice().iter())
    {
        assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0), "{a} vs {b}");
    }
}

fn fresh_state(shape: &[usize]) -> SpectralNormState {
    SpectralNormState::from_vectors(
        shape,
        1,
        &[1.0, 0.5, -0.25, 0.75],
        &[0.3, -0.6, 0.9, 0.1, 0.2, -0.4],
        2,
        1e-12,
    )
    .expect("state")
}

#[test]
fn spectral_norm_matches_free_function_bit_for_bit_including_state() {
    use fandhe_ai_autodiff::weight_reparam_ops::spectral_norm;
    let tape = fandhe_ai::tape();
    let shape = [3usize, 4, 2];
    let w0 = t(wave(numel(&shape), 0.091, 1.2), &shape);
    let up = t(wave(numel(&shape), 0.029, 0.8), &shape);
    for training in [true, false] {
        let run = |use_method: bool| {
            let mut st = fresh_state(&shape);
            let w = tape.var(&w0);
            let y = if use_method {
                w.spectral_norm(&mut st, training)
            } else {
                spectral_norm(&w, &mut st, training)
            }
            .expect("spectral_norm");
            let out = y.to_tensor();
            let loss = y.mul(&tape.var(&up)).expect("mul").sum(None).expect("sum");
            let dw = tape
                .backward(&loss)
                .expect("backward")
                .get(&w)
                .expect("get")
                .expect("dw")
                .clone();
            (out, dw, st)
        };
        let (m, f) = (run(true), run(false));
        assert_eq!(
            bits(&m.0),
            bits(&f.0),
            "spectral_norm training={training}: forward"
        );
        assert_eq!(
            bits(&m.1),
            bits(&f.1),
            "spectral_norm training={training}: dw"
        );
        assert_eq!(
            vec_bits(m.2.u()),
            vec_bits(f.2.u()),
            "u training={training}"
        );
        assert_eq!(
            vec_bits(m.2.v()),
            vec_bits(f.2.v()),
            "v training={training}"
        );
        if !training {
            // training = false は状態を更新しない。
            let init = fresh_state(&shape);
            assert_eq!(vec_bits(m.2.u()), vec_bits(init.u()));
            assert_eq!(vec_bits(m.2.v()), vec_bits(init.v()));
        }
    }
}

#[test]
fn typed_errors_for_rejected_arguments() {
    let tape = fandhe_ai::tape();

    // unfold: rank 3（バッチなし）入力は非対応。
    let x3 = tape.var(&t(wave(2 * 4 * 4, 0.3, 0.1), &[2, 4, 4]));
    assert!(x3.unfold([2, 2], [1, 1], [0, 0], [1, 1]).is_err());
    // fold: `C·kH·kW` で割り切れない列数。
    let cols = tape.var(&t(wave(5 * 4, 0.3, 0.1), &[1, 5, 4]));
    assert!(cols.fold([3, 3], [2, 2], [1, 1], [0, 0], [1, 1]).is_err());

    // local_response_norm: size = 0・非有限パラメータ。
    let x4 = tape.var(&t(wave(2 * 3 * 2 * 2, 0.3, 0.1), &[2, 3, 2, 2]));
    assert!(x4.local_response_norm(0, 1e-4, 0.75, 1.0).is_err());
    assert!(x4.local_response_norm(3, f32::NAN, 0.75, 1.0).is_err());
    assert!(x4.local_response_norm(3, 1e-4, f32::INFINITY, 1.0).is_err());

    // weight_norm: g の shape 不一致・別 tape。
    let v = tape.var(&t(wave(4 * 3, 0.3, 0.1), &[4, 3]));
    let g_bad = tape.var(&t(vec![1.0; 3], &[3]));
    assert!(v.weight_norm(&g_bad, Some(0)).is_err());
    let other = fandhe_ai::tape();
    let g_other = other.var(&t(vec![1.0; 4], &[4, 1]));
    assert!(matches!(
        v.weight_norm(&g_other, Some(0)),
        Err(AutodiffError::TapeMismatch)
    ));

    // spectral_norm: 状態と重みの shape 不整合は型付きエラーで、状態は bit 不変。
    let mut st = fresh_state(&[3, 4, 2]);
    let before = (vec_bits(st.u()), vec_bits(st.v()));
    let w_bad = tape.var(&t(wave(3 * 4 * 3, 0.3, 0.1), &[3, 4, 3]));
    assert!(w_bad.spectral_norm(&mut st, true).is_err());
    assert_eq!((vec_bits(st.u()), vec_bits(st.v())), before);
}
