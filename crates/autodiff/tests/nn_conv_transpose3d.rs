//! `nn::ConvTranspose3d`（イシュー #3031・親 #3030）の受け入れ条件検証（`tests/nn_conv3d.rs` と同型）。
//!
//! - 構築検査: `new`／`from_parameters` の不正引数が型付きエラーで拒否される。
//! - forward 不正引数: 型付きエラーかつ tape に孤児ノードを残さない。
//! - bit 一致: 層の forward（`Module::forward`・`ConvTranspose3dVars::forward`）が
//!   `Var::conv_transpose3d` 直呼びと forward 値・全勾配で完全一致する。
//! - tape 経路 ≡ `forward_host` が bit 一致する。
//! - 決定性・初期化 bound・パラメータ API・`as_conv_transpose3d` フック。

mod common;

use fandhe_ai_autodiff::nn::{Conv3d, ConvTranspose3d, Module};
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

fn bits(tensor: &Tensor<f32>) -> Vec<u32> {
    dense(tensor).iter().map(|v| v.to_bits()).collect()
}

/// `ConvTranspose3d` は `Debug` を実装しないため `unwrap_err` の代替とする。
fn err_of<T>(result: Result<T, AutodiffError>) -> AutodiffError {
    match result {
        Err(err) => err,
        Ok(_) => panic!("expected Err"),
    }
}

fn is_typed_err(err: &AutodiffError) -> bool {
    matches!(
        err,
        AutodiffError::InvalidArgument(_) | AutodiffError::Backend(_) | AutodiffError::Shape(_)
    )
}

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * scale - offset).collect()
}

// --- 1. 構築検査 ---

#[test]
fn new_rejects_invalid_arguments() {
    let one = [1, 1, 1];
    let zero = [0, 0, 0];
    // in_channels == 0
    assert!(is_typed_err(&err_of(ConvTranspose3d::new(
        0, 2, [1; 3], one, zero, zero, one, 1, true, 1
    ))));
    // in が groups で割り切れない
    assert!(is_typed_err(&err_of(ConvTranspose3d::new(
        3, 4, [1; 3], one, zero, zero, one, 2, true, 1
    ))));
    // out が groups で割り切れない
    assert!(is_typed_err(&err_of(ConvTranspose3d::new(
        4, 3, [1; 3], one, zero, zero, one, 2, true, 1
    ))));
    // out < groups
    assert!(is_typed_err(&err_of(ConvTranspose3d::new(
        2, 0, [1; 3], one, zero, zero, one, 2, true, 1
    ))));
    // stride / dilation / groups == 0
    assert!(is_typed_err(&err_of(ConvTranspose3d::new(
        2,
        2,
        [1; 3],
        [1, 0, 1],
        zero,
        zero,
        one,
        1,
        true,
        1
    ))));
    assert!(is_typed_err(&err_of(ConvTranspose3d::new(
        2,
        2,
        [1; 3],
        one,
        zero,
        zero,
        [1, 1, 0],
        1,
        true,
        1
    ))));
    assert!(is_typed_err(&err_of(ConvTranspose3d::new(
        2, 2, [1; 3], one, zero, zero, one, 0, true, 1
    ))));
    // kernel_size に 0 を含む
    assert!(is_typed_err(&err_of(ConvTranspose3d::new(
        2,
        2,
        [1, 0, 1],
        one,
        zero,
        zero,
        one,
        1,
        true,
        1
    ))));
    // output_padding >= stride（各軸）
    for axis in 0..3 {
        let mut op = [0, 0, 0];
        op[axis] = 1;
        let err = err_of(ConvTranspose3d::new(
            2, 2, [1; 3], one, zero, op, one, 1, true, 1,
        ));
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }
    // padding 過大（2·padding オーバーフロー）
    assert!(is_typed_err(&err_of(ConvTranspose3d::new(
        2,
        2,
        [1; 3],
        one,
        [usize::MAX, 0, 0],
        zero,
        one,
        1,
        true,
        1
    ))));
}

#[test]
fn from_parameters_rejects_invalid_arguments() {
    let one = [1, 1, 1];
    let zero = [0, 0, 0];
    let w = |shape: &[usize]| t(vec![0.0; shape.iter().product()], shape);
    // rank 不一致
    assert!(is_typed_err(&err_of(ConvTranspose3d::from_parameters(
        w(&[2, 2, 1, 1]),
        None,
        one,
        zero,
        zero,
        one,
        1
    ))));
    // shape[1] == 0
    assert!(is_typed_err(&err_of(ConvTranspose3d::from_parameters(
        w(&[2, 0, 1, 1, 1]),
        None,
        one,
        zero,
        zero,
        one,
        1
    ))));
    // shape[0] == 0
    assert!(is_typed_err(&err_of(ConvTranspose3d::from_parameters(
        w(&[0, 2, 1, 1, 1]),
        None,
        one,
        zero,
        zero,
        one,
        1
    ))));
    // bias shape 不一致（out_channels = 2 * groups(1) = 2 のところへ 3）
    assert!(is_typed_err(&err_of(ConvTranspose3d::from_parameters(
        w(&[2, 2, 1, 1, 1]),
        Some(w(&[3])),
        one,
        zero,
        zero,
        one,
        1
    ))));
    // bias rank 不一致
    assert!(is_typed_err(&err_of(ConvTranspose3d::from_parameters(
        w(&[2, 2, 1, 1, 1]),
        Some(w(&[2, 1])),
        one,
        zero,
        zero,
        one,
        1
    ))));
    // output_padding >= stride
    assert!(is_typed_err(&err_of(ConvTranspose3d::from_parameters(
        w(&[2, 2, 1, 1, 1]),
        None,
        one,
        zero,
        [0, 1, 0],
        one,
        1
    ))));
    // in が groups で割り切れない
    assert!(is_typed_err(&err_of(ConvTranspose3d::from_parameters(
        w(&[3, 2, 1, 1, 1]),
        None,
        one,
        zero,
        zero,
        one,
        2
    ))));
}

// --- 2. forward 不正引数（孤児ノードなし）---

#[test]
fn forward_rejects_invalid_input_without_orphan_nodes() {
    let layer =
        ConvTranspose3d::new(2, 3, [2, 2, 2], [1; 3], [0; 3], [0; 3], [1; 3], 1, true, 1).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = layer.bind(&tape);
    let before = tape.len();

    // rank 不一致
    let x_rank = tape.var(&t(vec![0.0; 2 * 2 * 2], &[2, 2, 2]));
    assert!(is_typed_err(&err_of(vars.forward(&x_rank))));
    // チャンネル不整合（Cin = 3 != 2）
    let x_ch = tape.var(&t(vec![0.0; 3 * 8], &[1, 3, 2, 2, 2]));
    assert!(is_typed_err(&err_of(vars.forward(&x_ch))));
    // 空間軸 0
    let x_zero = tape.var(&t(vec![], &[1, 2, 0, 2, 2]));
    assert!(is_typed_err(&err_of(vars.forward(&x_zero))));

    // 入力 3 つ登録分のみ増え、forward が積んだノードは無い。
    assert_eq!(tape.len(), before + 3);

    // forward_host も同じ入力を型付きエラーで拒否する。
    let ops = common::naive_ops();
    assert!(is_typed_err(&err_of(
        layer.forward_host(&*ops, &t(vec![0.0; 8], &[2, 2, 2]))
    )));
    assert!(is_typed_err(&err_of(
        layer.forward_host(&*ops, &t(vec![0.0; 24], &[1, 3, 2, 2, 2]))
    )));
}

// --- 3. 層 forward ≡ Var::conv_transpose3d 直呼び（forward 値・全勾配 bit 一致）---

struct Case {
    n: usize,
    cin: usize,
    cout: usize,
    k: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
    output_padding: [usize; 3],
    dilation: [usize; 3],
    groups: usize,
    bias: bool,
    spatial: [usize; 3],
}

fn assert_layer_matches_direct(c: &Case, seed: u64) {
    let layer = ConvTranspose3d::new(
        c.cin,
        c.cout,
        c.k,
        c.stride,
        c.padding,
        c.output_padding,
        c.dilation,
        c.groups,
        c.bias,
        seed,
    )
    .unwrap();
    let numel = c.n * c.cin * c.spatial.iter().product::<usize>();
    let x = t(
        ramp(numel, 0.013, 0.2),
        &[c.n, c.cin, c.spatial[0], c.spatial[1], c.spatial[2]],
    );

    // 経路 A: ConvTranspose3dVars::forward
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let xa = tape_a.var(&x);
    let va = layer.bind(&tape_a);
    let ya = va.forward(&xa).unwrap();
    let out_a = ya.to_tensor();

    // 経路 B: Var::conv_transpose3d 直呼び
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let xb = tape_b.var(&x);
    let wb = tape_b.var(layer.weight());
    let bb = layer.bias().map(|b| tape_b.var(b));
    let yb = xb
        .conv_transpose3d(
            &wb,
            bb.as_ref(),
            c.stride,
            c.padding,
            c.output_padding,
            c.dilation,
            c.groups,
        )
        .unwrap();
    let out_b = yb.to_tensor();

    // 経路 C: Module::forward
    let tape_c = Tape::new_with_ops(common::naive_ops());
    let xc = tape_c.var(&x);
    let out_c = Module::forward(&layer, &tape_c, &xc).unwrap().to_tensor();

    assert_eq!(out_a.shape(), out_b.shape());
    assert_eq!(bits(&out_a), bits(&out_b));
    assert_eq!(bits(&out_c), bits(&out_b));

    // 経路 D: forward_host（tape 不要経路）
    let ops = common::naive_ops();
    let out_d = layer.forward_host(&*ops, &x).unwrap();
    assert_eq!(out_d.shape(), out_b.shape());
    assert_eq!(bits(&out_d), bits(&out_b));

    if c.n == 0 {
        return;
    }
    let ga = tape_a.backward(&ya).unwrap();
    let gb = tape_b.backward(&yb).unwrap();
    assert_eq!(
        bits(ga.get(&xa).unwrap().unwrap()),
        bits(gb.get(&xb).unwrap().unwrap())
    );
    assert_eq!(
        bits(ga.get(&va.weight).unwrap().unwrap()),
        bits(gb.get(&wb).unwrap().unwrap())
    );
    match (&va.bias, &bb) {
        (Some(a), Some(b)) => assert_eq!(
            bits(ga.get(a).unwrap().unwrap()),
            bits(gb.get(b).unwrap().unwrap())
        ),
        (None, None) => {}
        _ => panic!("bias presence mismatch"),
    }
}

#[test]
fn layer_matches_direct_call_groups2_nontrivial_hyperparams() {
    assert_layer_matches_direct(
        &Case {
            n: 2,
            cin: 4,
            cout: 6,
            k: [2, 3, 2],
            stride: [2, 2, 1],
            padding: [1, 0, 0],
            output_padding: [1, 1, 0],
            dilation: [1, 2, 1],
            groups: 2,
            bias: true,
            spatial: [3, 3, 4],
        },
        7,
    );
}

#[test]
fn layer_matches_direct_call_groups1_no_bias() {
    assert_layer_matches_direct(
        &Case {
            n: 1,
            cin: 2,
            cout: 3,
            k: [2, 2, 2],
            stride: [1; 3],
            padding: [0; 3],
            output_padding: [0; 3],
            dilation: [1; 3],
            groups: 1,
            bias: false,
            spatial: [2, 3, 2],
        },
        11,
    );
}

#[test]
fn layer_matches_direct_call_empty_batch() {
    assert_layer_matches_direct(
        &Case {
            n: 0,
            cin: 2,
            cout: 2,
            k: [2, 2, 2],
            stride: [2; 3],
            padding: [0; 3],
            output_padding: [1; 3],
            dilation: [1; 3],
            groups: 1,
            bias: true,
            spatial: [2, 2, 2],
        },
        3,
    );
}

// --- 5. 決定性・初期化 bound ---

#[test]
fn same_seed_same_weights_and_no_bias_is_none() {
    let mk = |seed, bias| {
        ConvTranspose3d::new(
            2,
            4,
            [2, 2, 2],
            [1; 3],
            [0; 3],
            [0; 3],
            [1; 3],
            2,
            bias,
            seed,
        )
        .unwrap()
    };
    let a = mk(42, true);
    let b = mk(42, true);
    let c = mk(43, true);
    assert_eq!(bits(a.weight()), bits(b.weight()));
    assert_ne!(bits(a.weight()), bits(c.weight()));
    assert!(mk(1, false).bias().is_none());
    assert_eq!(a.weight().shape(), &[2, 2, 2, 2, 2]);
    assert_eq!(a.bias().unwrap().shape(), &[4]);
}

#[test]
fn init_bound_uses_cout_per_group_fan_in() {
    // fan_in = (out / groups) · kD · kH · kW = (6 / 2) · 2 · 3 · 1 = 18。
    let layer =
        ConvTranspose3d::new(4, 6, [2, 3, 1], [1; 3], [0; 3], [0; 3], [1; 3], 2, true, 9).unwrap();
    let bound = 1.0 / (18.0f32).sqrt();
    assert!(dense(layer.weight()).iter().all(|w| w.abs() <= bound));
    assert!(
        dense(layer.bias().unwrap())
            .iter()
            .all(|w| w.abs() <= bound)
    );
}

// --- 6. パラメータ API ---

#[test]
fn named_parameters_order_and_set_parameter_contract() {
    let mut layer =
        ConvTranspose3d::new(2, 2, [1; 3], [1; 3], [0; 3], [0; 3], [1; 3], 1, true, 1).unwrap();
    let names: Vec<String> = layer
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert_eq!(names, vec!["weight".to_string(), "bias".to_string()]);

    let new_w = t(vec![1.0; 4], &[2, 2, 1, 1, 1]);
    Module::set_parameter(&mut layer, "weight", new_w).unwrap();
    assert_eq!(dense(layer.weight()), vec![1.0; 4]);

    // shape 不一致・未知名
    assert!(is_typed_err(&err_of(Module::set_parameter(
        &mut layer,
        "weight",
        t(vec![0.0; 3], &[3])
    ))));
    assert!(is_typed_err(&err_of(Module::set_parameter(
        &mut layer,
        "gamma",
        t(vec![0.0; 2], &[2])
    ))));

    // bias 無し層
    let mut nobias =
        ConvTranspose3d::new(2, 2, [1; 3], [1; 3], [0; 3], [0; 3], [1; 3], 1, false, 1).unwrap();
    assert_eq!(nobias.named_parameters().len(), 1);
    assert!(is_typed_err(&err_of(Module::set_parameter(
        &mut nobias,
        "bias",
        t(vec![0.0; 2], &[2])
    ))));
}

#[test]
fn set_requires_grad_false_blocks_weight_grad() {
    let mut layer =
        ConvTranspose3d::new(2, 2, [2, 2, 2], [1; 3], [0; 3], [0; 3], [1; 3], 1, true, 1).unwrap();
    assert!(Module::requires_grad(&layer));
    Module::set_requires_grad(&mut layer, false).unwrap();
    assert!(!Module::requires_grad(&layer));

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&t(ramp(16, 0.1, 0.5), &[1, 2, 2, 2, 2]));
    let vars = layer.bind(&tape);
    let y = vars.forward(&x).unwrap();
    let grads = tape.backward(&y).unwrap();
    assert!(matches!(
        grads.get(&vars.weight),
        Err(AutodiffError::GradientTrackingDisabled)
    ));
    assert!(grads.get(&x).unwrap().is_some());
}

// --- 7. フック ---

#[test]
fn as_conv_transpose3d_hooks_are_some_only_for_this_layer() {
    let mut layer =
        ConvTranspose3d::new(2, 2, [1; 3], [1; 3], [0; 3], [0; 3], [1; 3], 1, true, 1).unwrap();
    assert!(layer.as_conv_transpose3d().is_some());
    assert!(layer.as_conv_transpose3d_mut().is_some());

    let mut conv = Conv3d::new(2, 2, [1; 3], [1; 3], [0; 3], [1; 3], 1, true, 1).unwrap();
    assert!(conv.as_conv_transpose3d().is_none());
    assert!(conv.as_conv_transpose3d_mut().is_none());
    assert!(conv.as_conv3d().is_some());
    assert!(layer.as_conv3d().is_none());
}
