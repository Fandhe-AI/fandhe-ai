//! 承認された Phase 4 の演算・自動微分の facade 利用テスト（イシュー #2678。親 #2625・
//! ルート #2499。承認根拠は `issuecomment-6033824965`・`docs/compat-api-scope.md` §5.1）。
//!
//! 利用者が見る面（`fandhe_ai::` だけを import）から、公開した `Var` の委譲メソッドと
//! 型の再エクスポートが到達でき、シグネチャが承認形と一致し、閉形式か手計算で厳密に決まる
//! 値（と一部の勾配）が得られることを固定する。数値経路は内部クレートの既存実装のままで、
//! 新しい tolerance・baseline は設けない。実機（CUDA／Metal）の数値一致は既存の
//! `*_backend_parity.rs`（`#[ignore]`）が担う。

use fandhe_ai::{
    AutodiffError, FftNorm, MeshgridIndexing, PadMode, QuantileInterpolation, ScalarDType,
    ScatterReduceMode, Tape, Tensor, Var,
};

type R<'t> = Result<Var<'t>, AutodiffError>;
type RF = Result<Tensor<f32>, AutodiffError>;
type RI = Result<Tensor<i32>, AutodiffError>;

fn shp(v: &Var<'_>) -> Vec<usize> {
    v.to_tensor().shape().to_vec()
}

/// シグネチャ固定用（`Var<'t>` の `'t` を名指しするため関数で包む）。
fn sig_unary<'t>() -> [fn(&Var<'t>) -> R<'t>; 8] {
    [
        Var::<'t>::atan,
        Var::<'t>::asin,
        Var::<'t>::sinh,
        Var::<'t>::asinh,
        Var::<'t>::atanh,
        Var::<'t>::cosh,
        Var::<'t>::acos,
        Var::<'t>::acosh,
    ]
}
fn sig_atan2<'t>() -> fn(&Var<'t>, &Var<'t>) -> R<'t> {
    Var::<'t>::atan2
}
fn sig_isnan<'t>() -> fn(&Var<'t>) -> Result<Tensor<bool>, AutodiffError> {
    Var::<'t>::isnan
}
type Pair<'t> = Result<(Var<'t>, Tensor<i32>), AutodiffError>;
fn sig_cummax<'t>() -> fn(&Var<'t>, usize) -> Pair<'t> {
    Var::<'t>::cummax
}
fn sig_kth<'t>() -> fn(&Var<'t>, usize, usize) -> Pair<'t> {
    Var::<'t>::kthvalue
}
fn sig_histc<'t>() -> fn(&Var<'t>, usize, f32, f32) -> RF {
    Var::<'t>::histc
}
fn sig_grid<'t>() -> fn(&[Var<'t>], MeshgridIndexing) -> Result<Vec<Var<'t>>, AutodiffError> {
    Var::<'t>::meshgrid
}
fn sig_relu<'t>() -> fn(&Var<'t>, ScalarDType) -> R<'t> {
    Var::<'t>::relu_low_precision
}

fn t1(v: &[f32]) -> Tensor<f32> {
    Tensor::new(v.to_vec(), &[v.len()]).expect("tensor")
}

fn vals(v: &Var<'_>) -> Vec<f32> {
    v.to_tensor().host_slice().into_owned()
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() < 1e-5, "{actual:?} vs {expected:?}");
    }
}

#[test]
fn trig_methods_values_and_gradient() {
    let tape = fandhe_ai::tape();
    let z = tape.var(&t1(&[0.0]));
    let one = tape.var(&t1(&[1.0]));
    let unary = sig_unary();
    // atan/asin/sinh/asinh/atanh(0) = 0、cosh(0) = 1、acos(1) = 0、acosh(1) = 0。
    for (i, f) in unary.iter().enumerate() {
        let (x, expected) = match i {
            5 => (&z, 1.0),
            6 | 7 => (&one, 0.0),
            _ => (&z, 0.0),
        };
        close(&vals(&f(x).expect("unary")), &[expected]);
    }
    let atan2 = sig_atan2();
    close(
        &vals(&atan2(&one, &one).expect("atan2")),
        &[std::f32::consts::FRAC_PI_4],
    );
    // d/dx atan(x) = 1 / (1 + x^2) = 1（x = 0）。
    let loss = z.atan().expect("atan").sum(None).expect("sum");
    let grads = tape.backward(&loss).expect("backward");
    close(&vals_t(grads.get(&z).expect("get").expect("grad")), &[1.0]);
}

fn vals_t(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice().into_owned()
}

#[test]
fn nonfinite_methods() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t1(&[1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY]));
    let isnan = sig_isnan();
    assert_eq!(
        isnan(&x).expect("isnan").host_slice().into_owned(),
        [false, true, false, false]
    );
    assert_eq!(
        x.isinf().expect("isinf").host_slice().into_owned(),
        [false, false, true, true]
    );
    assert_eq!(
        x.isfinite().expect("isfinite").host_slice().into_owned(),
        [true, false, false, false]
    );
    let y = x
        .nan_to_num(Some(0.0), Some(9.0), Some(-9.0))
        .expect("nan_to_num");
    assert_eq!(vals(&y), [1.0, 0.0, 9.0, -9.0]);
}

#[test]
fn cumulative_methods() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t1(&[1.0, 3.0, 2.0]));
    let cummax = sig_cummax();
    let (v, i) = cummax(&x, 0).expect("cummax");
    assert_eq!(vals(&v), [1.0, 3.0, 3.0]);
    assert_eq!(i.host_slice().into_owned(), [0, 1, 1]);
    let (v, i) = x.cummin(0).expect("cummin");
    assert_eq!(vals(&v), [1.0, 1.0, 1.0]);
    assert_eq!(i.host_slice().into_owned(), [0, 0, 0]);
    let z = tape.var(&t1(&[0.0, 0.0]));
    close(
        &vals(&z.logcumsumexp(0).expect("logcumsumexp")),
        &[0.0, std::f32::consts::LN_2],
    );
}

#[test]
fn stat_reduce_methods() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t1(&[5.0, 1.0, 3.0, 2.0, 4.0]));
    assert_eq!(vals(&x.median(None).expect("median")), [3.0]);
    let (v, i) = x.median_with_indices(0).expect("median_with_indices");
    assert_eq!(vals(&v), [3.0]);
    assert_eq!(i.host_slice().into_owned(), [2]);
    // k は 1 始まり。2 番目に小さい値は 2.0（元の位置 3）。
    let kth = sig_kth();
    let (v, i) = kth(&x, 2, 0).expect("kthvalue");
    assert_eq!(vals(&v), [2.0]);
    assert_eq!(i.host_slice().into_owned(), [3]);
    let q = x
        .quantile(0.5, None, QuantileInterpolation::Linear)
        .expect("quantile");
    close(&vals(&q), &[3.0]);
    let n = tape.var(&t1(&[1.0, f32::NAN, 3.0]));
    close(&vals(&n.nanmean(None).expect("nanmean")), &[2.0]);
    close(&vals(&n.nansum(None).expect("nansum")), &[4.0]);
}

#[test]
fn binning_methods_and_tape_bincount() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t1(&[0.5, 1.5, 2.5, 3.5]));
    let histc = sig_histc();
    assert_eq!(
        histc(&x, 4, 0.0, 4.0)
            .expect("histc")
            .host_slice()
            .into_owned(),
        [1.0, 1.0, 1.0, 1.0]
    );
    let sorted = tape.var(&t1(&[1.0, 3.0, 5.0]));
    let values = tape.var(&t1(&[3.0]));
    assert_eq!(
        sorted
            .searchsorted(&values, false)
            .expect("searchsorted")
            .host_slice()
            .into_owned(),
        [1]
    );
    assert_eq!(
        sorted
            .searchsorted(&values, true)
            .expect("searchsorted right")
            .host_slice()
            .into_owned(),
        [2]
    );
    assert_eq!(
        values
            .bucketize(&sorted, false)
            .expect("bucketize")
            .host_slice()
            .into_owned(),
        [1]
    );
    // facade `Tape` の入口（`Var` を取らない演算）。
    let bincount: fn(&Tape, &Tensor<i32>, usize) -> RI = Tape::bincount;
    let idx = Tensor::new(vec![0_i32, 2, 2], &[3]).expect("idx");
    assert_eq!(
        bincount(&tape, &idx, 4)
            .expect("bincount")
            .host_slice()
            .into_owned(),
        [1, 0, 2, 0]
    );
    let w = tape.var(&t1(&[0.5, 1.0, 2.0]));
    let weighted = tape.bincount_weighted(&idx, &w, 0).expect("weighted");
    assert_eq!(weighted.host_slice().into_owned(), [0.5, 0.0, 3.0]);
}

#[test]
fn shape_view_methods() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("x"));
    let parts = x.unbind(0).expect("unbind");
    assert_eq!(parts.len(), 2);
    assert_eq!(vals(&parts[1]), [3.0, 4.0]);
    let split = x.tensor_split(2, 1).expect("tensor_split");
    assert_eq!(vals(&split[0]), [1.0, 3.0]);
    let split_idx = x
        .tensor_split_indices(&[1], 0)
        .expect("tensor_split_indices");
    assert_eq!(split_idx.len(), 2);
    assert_eq!(shp(&x.swapaxes(0, 1).expect("swapaxes")), [2, 2]);
    assert_eq!(
        vals(&x.swapaxes(0, 1).expect("swapaxes")),
        [1.0, 3.0, 2.0, 4.0]
    );
    assert_eq!(
        vals(&x.movedim(&[0], &[1]).expect("movedim")),
        [1.0, 3.0, 2.0, 4.0]
    );
    // 90 度回転（k = 1）。`torch.rot90([[1, 2], [3, 4]])` = `[[2, 4], [1, 3]]`。
    assert_eq!(
        vals(&x.rot90(1, [0, 1]).expect("rot90")),
        [2.0, 4.0, 1.0, 3.0]
    );
    let a = tape.var(&t1(&[1.0, 2.0]));
    let b = tape.var(&t1(&[3.0, 4.0, 5.0]));
    let grid = sig_grid();
    let g = grid(&[a, b], MeshgridIndexing::Ij).expect("meshgrid");
    assert_eq!(shp(&g[0]), [2, 3]);
    assert_eq!(vals(&g[0]), [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    assert_eq!(vals(&g[1]), [3.0, 4.0, 5.0, 3.0, 4.0, 5.0]);
}

#[test]
fn indexed_update_methods() {
    let tape = fandhe_ai::tape();
    let idx = Tensor::new(vec![0_i32, 0], &[2]).expect("idx");
    let x = tape.var(&t1(&[0.0, 0.0, 0.0]));
    let src = tape.var(&t1(&[1.0, 2.0]));
    assert_eq!(
        vals(&x.index_add(0, &idx, &src).expect("index_add")),
        [3.0, 0.0, 0.0]
    );
    let one = Tensor::new(vec![1_i32], &[1]).expect("idx");
    let s = tape.var(&t1(&[5.0]));
    assert_eq!(
        vals(&x.index_copy(0, &one, &s).expect("index_copy")),
        [0.0, 5.0, 0.0]
    );
    let mask = Tensor::new(vec![true, false, true], &[3]).expect("mask");
    let ms = tape.var(&t1(&[7.0, 8.0]));
    assert_eq!(
        vals(&x.masked_scatter(&mask, &ms).expect("masked_scatter")),
        [7.0, 0.0, 8.0]
    );
    let y = tape.var(&t1(&[0.0, 0.0]));
    let sidx = Tensor::new(vec![0_i32, 0, 1], &[3]).expect("idx");
    let ssrc = tape.var(&t1(&[1.0, 2.0, 3.0]));
    let out = y
        .scatter_reduce(0, &sidx, &ssrc, ScatterReduceMode::Sum, true)
        .expect("scatter_reduce");
    assert_eq!(vals(&out), [3.0, 3.0]);
}

#[test]
fn tensor_product_methods() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&t1(&[1.0, 2.0]));
    let b = tape.var(&t1(&[3.0, 4.0]));
    assert_eq!(vals(&a.kron(&b).expect("kron")), [3.0, 4.0, 6.0, 8.0]);
    assert_eq!(vals(&a.tensordot(&b, 1).expect("tensordot")), [11.0]);
    assert_eq!(
        vals(&a.tensordot_axes(&b, &[0], &[0]).expect("tensordot_axes")),
        [11.0]
    );
    let x1 = tape.var(&Tensor::new(vec![0.0_f32, 0.0], &[1, 2]).expect("x1"));
    let x2 = tape.var(&Tensor::new(vec![3.0_f32, 4.0], &[1, 2]).expect("x2"));
    close(&vals(&x1.cdist(&x2, 2.0).expect("cdist")), &[5.0]);
    let e1 = tape.var(&t1(&[1.0, 0.0, 0.0]));
    let e2 = tape.var(&t1(&[0.0, 1.0, 0.0]));
    assert_eq!(vals(&e1.cross(&e2, 0).expect("cross")), [0.0, 0.0, 1.0]);
}

#[test]
fn pad_with_mode_method() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t1(&[1.0, 2.0, 3.0]));
    let reflect = x
        .pad_with_mode(&[(1, 1)], PadMode::Reflect)
        .expect("reflect");
    assert_eq!(vals(&reflect), [2.0, 1.0, 2.0, 3.0, 2.0]);
    let replicate = x
        .pad_with_mode(&[(1, 1)], PadMode::Replicate)
        .expect("replicate");
    assert_eq!(vals(&replicate), [1.0, 1.0, 2.0, 3.0, 3.0]);
}

#[test]
fn activation_scalar_and_softmin_threshold_methods() {
    let tape = fandhe_ai::tape();
    let z = tape.var(&t1(&[0.0]));
    close(&vals(&z.selu().expect("selu")), &[0.0]);
    close(&vals(&z.celu(1.0).expect("celu")), &[0.0]);
    close(&vals(&z.hardsigmoid().expect("hardsigmoid")), &[0.5]);
    close(
        &vals(&z.log_sigmoid().expect("log_sigmoid")),
        &[-std::f32::consts::LN_2],
    );
    let one = tape.var(&t1(&[1.0]));
    close(&vals(&one.softsign().expect("softsign")), &[0.5]);
    close(&vals(&z.tanhshrink().expect("tanhshrink")), &[0.0]);
    let x = tape.var(&t1(&[1.0, 3.0]));
    assert_eq!(
        vals(&x.threshold(2.0, -1.0).expect("threshold")),
        [-1.0, 3.0]
    );
    close(&vals(&z.softmin(0).expect("softmin")), &[1.0]);
    let eq = tape.var(&t1(&[0.0, 0.0]));
    close(&vals(&eq.softmin(0).expect("softmin")), &[0.5, 0.5]);
    // 評価時（training = false）の RReLU は傾き (lower + upper) / 2 の固定 leaky ReLU。
    let r = tape.var(&t1(&[-10.0, 5.0]));
    close(
        &vals(&r.rrelu(0.1, 0.3, false).expect("rrelu")),
        &[-2.0, 5.0],
    );
}

#[test]
fn fft_methods_roundtrip() {
    let tape = fandhe_ai::tape();
    let x = tape.var(&t1(&[1.0, 0.0, 0.0, 0.0]));
    let spec = x.rfft(None, None, FftNorm::Backward).expect("rfft");
    assert_eq!(shp(&spec), [3, 2]);
    close(&vals(&spec), &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    let back = spec.irfft(Some(4), None, FftNorm::Backward).expect("irfft");
    close(&vals(&back), &[1.0, 0.0, 0.0, 0.0]);
    // 入力信号への勾配が流れる（Σ re(rfft(x)) = 3 x0 + Σ…）。形状だけ固定する。
    let loss = spec.sum(None).expect("sum");
    let grads = tape.backward(&loss).expect("backward");
    assert_eq!(grads.get(&x).expect("get").expect("grad").shape(), &[4]);
}

#[test]
fn low_precision_methods_return_value_or_typed_error() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&t1(&[1.0, -2.0]));
    let b = tape.var(&t1(&[3.0, 4.0]));
    // 既定 CPU テープでの対応状況に依存するため、成功なら値、非対応なら型付きエラーを許す
    // （`Unsupported` を握りつぶして f32 へフォールバックしないことが契約）。
    let relu = sig_relu();
    match relu(&a, ScalarDType::F16) {
        Ok(v) => assert_eq!(vals(&v), [1.0, 0.0]),
        Err(e) => assert!(
            matches!(
                e,
                AutodiffError::Backend(_) | AutodiffError::InvalidArgument(_)
            ),
            "{e:?}"
        ),
    }
    for r in [
        a.add_low_precision(&b, ScalarDType::F16),
        a.mul_low_precision(&b, ScalarDType::F16),
        a.exp_low_precision(ScalarDType::F16),
        a.tanh_low_precision(ScalarDType::F16),
    ] {
        match r {
            Ok(v) => assert_eq!(shp(&v), [2]),
            Err(e) => assert!(
                matches!(
                    e,
                    AutodiffError::Backend(_) | AutodiffError::InvalidArgument(_)
                ),
                "{e:?}"
            ),
        }
    }
    let m = tape.var(&Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).expect("m"));
    match m.matmul_low_precision(&m, ScalarDType::F16) {
        Ok(v) => assert_eq!(shp(&v), [2, 2]),
        Err(e) => assert!(
            matches!(
                e,
                AutodiffError::Backend(_) | AutodiffError::InvalidArgument(_)
            ),
            "{e:?}"
        ),
    }
}

#[test]
fn tape_jacobian_hessian_and_anomaly_signatures() {
    let jacobian: fn(&Tape, &Var<'_>, &Var<'_>) -> RF = Tape::jacobian;
    let hessian: fn(&Tape, &Var<'_>, &Var<'_>, &Tape) -> RF = Tape::hessian;
    let tape = fandhe_ai::tape();
    let x = tape.var(&t1(&[1.0, 2.0]));
    let y = x.mul(&x).expect("mul");
    let j = jacobian(&tape, &y, &x).expect("jacobian");
    assert_eq!(j.shape(), &[2, 2]);
    assert_eq!(vals_t(&j), [2.0, 0.0, 0.0, 4.0]);

    let child = fandhe_ai::tape();
    let loss = x
        .mul(&x)
        .expect("mul")
        .mul(&x)
        .expect("mul")
        .sum(None)
        .expect("sum");
    let h = hessian(&tape, &loss, &x, &child).expect("hessian");
    // loss = Σ x^3 のヘッセ行列は diag(6 x) = diag(6, 12)。
    assert_eq!(h.shape(), &[2, 2]);
    close(&vals_t(&h), &[6.0, 0.0, 0.0, 12.0]);

    // 非有限値を生んだ逆伝播は型付きエラー（ノード種別名のみでテンソル値を含まない）。
    let tape2 = fandhe_ai::tape();
    let z = tape2.var(&t1(&[0.0]));
    let bad = z.log().expect("log").sum(None).expect("sum");
    let err = tape2.backward_detect_anomaly(&bad).expect_err("anomaly");
    assert!(matches!(err, AutodiffError::Backward(_)), "{err:?}");
}
