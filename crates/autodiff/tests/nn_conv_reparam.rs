//! Conv 系 6 層（`Conv1d`／`Conv2d`／`Conv3d`／`ConvTranspose1d`／`ConvTranspose2d`／
//! `ConvTranspose3d`）の weight_norm／spectral_norm 再パラメータ化（イシュー #3049・親 #3018。
//! 決定記録 `docs/facade-weight-reparam-wiring-decision.md` §5〜§10・§13）の層単体テスト。
//! `nn_linear_reparam.rs`（#3048）と同型。
//!
//! facade の結合テスト（`crates/facade/tests/compat_sequential_parametrize_conv.rs`）が
//! `Sequential` 越しの契約を担うのに対し、ここでは層自身の契約を 6 層共通のマクロで固定する。
//!
//! - 軸の方針: `spectral_norm` の行軸は Conv 系 0・ConvTranspose 系 1（PyTorch と一致）。
//!   `weight_norm` の `dim` は層種に依らず利用者指定（決定記録 §13。イシュー本文の
//!   「ConvTranspose は dim=1」は spectral_norm にのみ当てはまる）
//! - 勾配一致: 層経路と、`Var::weight_norm`／`Var::spectral_norm` で有効重みを組んで同じ畳み込みへ
//!   渡す参照計算が bit 一致する
//! - 命名・`set_parameter`・失敗時の層不変・spectral 状態の進行契約・host 経路の拒否

mod common;

use fandhe_ai_autodiff::nn::{
    Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d, Module,
};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::{BackendError, Tensor};

const SEED: u64 = 0x3049;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn ramp(shape: &[usize], seed: f32) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    t(common::det_seq(n, seed), shape)
}

fn bits(x: &Tensor<f32>) -> Vec<u32> {
    x.contiguous()
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn assert_req2(actual: &Tensor<f32>, expected: &Tensor<f32>, what: &str) {
    let (a, e) = (actual.contiguous(), expected.contiguous());
    assert_eq!(a.shape(), e.shape(), "{what}: shape");
    for (i, (x, y)) in a.host_slice().iter().zip(e.host_slice().iter()).enumerate() {
        assert!(
            common::req2_close(f64::from(*x), f64::from(*y)),
            "{what}[{i}]: {x} vs {y}"
        );
    }
}

/// `weight` の `dim` 軸を先頭へ移した `[h, w]` 行列の最大特異値（ホスト f64 の独立 power iteration）。
/// `Var::spectral_norm` の行列化（`dim` 軸を行にした 2 次元化）の独立な参照実装。
fn host_sigma(weight: &Tensor<f32>, dim: usize) -> f64 {
    let shape = weight.shape().to_vec();
    let data = weight.contiguous().host_slice().to_vec();
    let h = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = shape[..dim].iter().product();
    let w = data.len() / h;
    // 行列要素 (r, c): r = dim 軸の添字、c = (outer 添字, inner 添字) の連結。
    let at = |r: usize, c: usize| -> f64 {
        let (o, i) = (c / inner, c % inner);
        f64::from(data[(o * h + r) * inner + i])
    };
    assert_eq!(outer * inner, w);
    let mut v: Vec<f64> = (0..w).map(|j| 1.0 + 0.01 * j as f64).collect();
    let mut sigma = 0.0;
    for _ in 0..500 {
        let u: Vec<f64> = (0..h)
            .map(|r| (0..w).map(|c| at(r, c) * v[c]).sum())
            .collect();
        let un = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        let u: Vec<f64> = u.iter().map(|x| x / un).collect();
        let nv: Vec<f64> = (0..w)
            .map(|c| (0..h).map(|r| at(r, c) * u[r]).sum())
            .collect();
        sigma = nv.iter().map(|x| x * x).sum::<f64>().sqrt();
        v = nv.iter().map(|x| x / sigma).collect();
    }
    sigma
}

macro_rules! conv_reparam_tests {
    (
        $modname:ident, $layer:ty, $make:expr, $input_shape:expr,
        spectral_dim = $sdim:expr, u_len = $ulen:expr, wn_dims = $wn_dims:expr
    ) => {
        mod $modname {
            use super::*;

            fn make() -> $layer {
                $make
            }

            fn input() -> Tensor<f32> {
                ramp(&$input_shape, 0.5)
            }

            fn names(layer: &$layer) -> Vec<String> {
                layer
                    .named_parameters()
                    .into_iter()
                    .map(|(n, _)| n)
                    .collect()
            }

            fn forward_bits(layer: &$layer, x: &Tensor<f32>) -> Result<Vec<u32>, AutodiffError> {
                let tape = Tape::new_with_ops(common::naive_ops());
                let xv = tape.var(x);
                let vars = layer.bind(&tape);
                let g = layer.bind_weight_norm_g(&tape);
                let y = layer.forward_bound(&vars, g.as_ref(), &xv)?;
                Ok(bits(&y.to_tensor()))
            }

            /// 層経路の (loss ビット列, [g の勾配], weight の勾配, bias の勾配)。
            fn layer_step(layer: &$layer, x: &Tensor<f32>) -> (Vec<u32>, Vec<Vec<u32>>) {
                let tape = Tape::new_with_ops(common::naive_ops());
                let xv = tape.var(x);
                let vars = layer.bind(&tape);
                let g = layer.bind_weight_norm_g(&tape);
                let y = layer.forward_bound(&vars, g.as_ref(), &xv).unwrap();
                let target = tape.var(&ramp(y.to_tensor().shape(), 2.0));
                let loss = y.mse_loss(&target).unwrap();
                let grads = tape.backward(&loss).unwrap();
                let mut out = Vec::new();
                if let Some(g) = &g {
                    out.push(bits(grads.get(g).unwrap().unwrap()));
                }
                out.push(bits(grads.get(&vars.weight).unwrap().unwrap()));
                if let Some(b) = &vars.bias {
                    out.push(bits(grads.get(b).unwrap().unwrap()));
                }
                (bits(&loss.to_tensor()), out)
            }

            #[test]
            fn weight_norm_initial_forward_matches_plain_layer_for_each_dim() {
                let x = input();
                let plain = make();
                let want = {
                    let tape = Tape::new_with_ops(common::naive_ops());
                    let xv = tape.var(&x);
                    plain.forward(&tape, &xv).unwrap().to_tensor()
                };
                for dim in $wn_dims {
                    let layer = make().with_weight_norm(dim).unwrap();
                    assert_eq!(layer.weight_norm_dim(), Some(dim));
                    let tape = Tape::new_with_ops(common::naive_ops());
                    let xv = tape.var(&x);
                    let got = layer.forward(&tape, &xv).unwrap().to_tensor();
                    assert_req2(&got, &want, &format!("dim={dim:?}"));
                }
            }

            #[test]
            fn weight_norm_grads_bit_match_var_reference() {
                let x = input();
                for dim in $wn_dims {
                    let layer = make().with_weight_norm(dim).unwrap();
                    let (loss_bits, grads) = layer_step(&layer, &x);

                    // 参照: 素の層の Vars の weight を Var::weight_norm で組んだ有効重みへ差し替える。
                    let plain = make();
                    let tape = Tape::new_with_ops(common::naive_ops());
                    let xv = tape.var(&x);
                    let mut vars = plain.bind(&tape);
                    let v = vars.weight;
                    let g = tape.var(layer.weight_norm_g().unwrap());
                    vars.weight = v.weight_norm(&g, dim).unwrap();
                    let y = vars.forward(&xv).unwrap();
                    let target = tape.var(&ramp(y.to_tensor().shape(), 2.0));
                    let loss = y.mse_loss(&target).unwrap();
                    let ref_grads = tape.backward(&loss).unwrap();
                    let mut want = vec![
                        bits(ref_grads.get(&g).unwrap().unwrap()),
                        bits(ref_grads.get(&v).unwrap().unwrap()),
                    ];
                    if let Some(b) = &vars.bias {
                        want.push(bits(ref_grads.get(b).unwrap().unwrap()));
                    }
                    assert_eq!(loss_bits, bits(&loss.to_tensor()), "loss dim={dim:?}");
                    assert_eq!(grads, want, "grads dim={dim:?}");
                }
            }

            #[test]
            fn spectral_axis_state_and_reference_match() {
                let x = input();
                let layer = make().with_spectral_norm(1, 1e-12, SEED).unwrap();
                let st = layer.spectral_norm_state().unwrap();
                assert_eq!(
                    st.dim(),
                    $sdim,
                    "行軸（PyTorch: Conv 系 0・ConvTranspose 系 1）"
                );
                assert_eq!(st.u().len(), $ulen, "u 長 = shape[dim]");
                let numel: usize = layer.weight().shape().iter().product();
                assert_eq!(st.v().len(), numel / $ulen, "v 長 = numel / shape[dim]");

                // 参照: 層経路の損失・勾配が Var::spectral_norm 直接経路と bit 一致し、状態も一致する。
                let (loss_bits, grads) = layer_step(&layer, &x);
                let plain = make();
                let mut ref_state = st.clone();
                let tape = Tape::new_with_ops(common::naive_ops());
                let xv = tape.var(&x);
                let mut vars = plain.bind(&tape);
                let raw = vars.weight;
                vars.weight = raw.spectral_norm(&mut ref_state, true).unwrap();
                let y = vars.forward(&xv).unwrap();
                let target = tape.var(&ramp(y.to_tensor().shape(), 2.0));
                let loss = y.mse_loss(&target).unwrap();
                let ref_grads = tape.backward(&loss).unwrap();
                let mut want = vec![bits(ref_grads.get(&raw).unwrap().unwrap())];
                if let Some(b) = &vars.bias {
                    want.push(bits(ref_grads.get(b).unwrap().unwrap()));
                }
                assert_eq!(loss_bits, bits(&loss.to_tensor()));
                assert_eq!(grads, want);
                assert_eq!(layer.spectral_norm_state().unwrap(), ref_state);
            }

            #[test]
            fn spectral_converged_sigma_uses_the_derived_row_axis() {
                // 収束後の有効重み W/σ の σ が、`dim` 軸を行にした独立なホスト参照と一致する
                // （ConvTranspose で dim=1 が出力チャネル側の行軸であることの検証）。
                let x = input();
                let layer = make().with_spectral_norm(1, 1e-12, SEED).unwrap();
                for _ in 0..300 {
                    forward_bits(&layer, &x).unwrap();
                }
                let sigma = host_sigma(layer.weight(), $sdim);
                let tape = Tape::new_with_ops(common::naive_ops());
                let raw = tape.var(layer.weight());
                let mut st = layer.spectral_norm_state().unwrap();
                let eff = raw.spectral_norm(&mut st, false).unwrap().to_tensor();
                let expected: Vec<f32> = layer
                    .weight()
                    .contiguous()
                    .host_slice()
                    .iter()
                    .map(|w| (f64::from(*w) / sigma) as f32)
                    .collect();
                let expected = t(expected, layer.weight().shape());
                for (i, (a, e)) in eff
                    .contiguous()
                    .host_slice()
                    .iter()
                    .zip(expected.host_slice().iter())
                    .enumerate()
                {
                    let diff = f64::from((a - e).abs());
                    assert!(
                        diff < 2e-3 * f64::from(e.abs()).max(1e-2),
                        "W/σ[{i}]: {a} vs {e}"
                    );
                }
            }

            #[test]
            fn named_parameters_order_for_each_parametrization() {
                assert_eq!(names(&make()), ["weight", "bias"]);
                assert_eq!(
                    names(&make().with_weight_norm(Some(0)).unwrap()),
                    [
                        "parametrizations.weight.original0",
                        "parametrizations.weight.original1",
                        "bias"
                    ]
                );
                assert_eq!(
                    names(&make().with_spectral_norm(1, 1e-12, SEED).unwrap()),
                    ["parametrizations.weight.original", "bias"]
                );
            }

            #[test]
            fn set_parameter_accepts_new_keys_only_and_keeps_state() {
                let mut layer = make().with_weight_norm(Some(0)).unwrap();
                let g_shape = layer.weight_norm_g().unwrap().shape().to_vec();
                let g_new = t(vec![2.0; g_shape.iter().product()], &g_shape);
                layer
                    .set_parameter("parametrizations.weight.original0", g_new)
                    .unwrap();
                assert!(
                    layer
                        .weight_norm_g()
                        .unwrap()
                        .host_slice()
                        .iter()
                        .all(|v| *v == 2.0)
                );
                let w_shape = layer.weight().shape().to_vec();
                let v_new = t(vec![0.5; w_shape.iter().product()], &w_shape);
                layer
                    .set_parameter("parametrizations.weight.original1", v_new)
                    .unwrap();
                assert!(layer.weight().host_slice().iter().all(|v| *v == 0.5));
                assert!(layer.is_parametrized(), "再パラメータ化状態は保持される");

                // `weight` キーは拒否、shape 不一致は拒否。
                assert!(matches!(
                    layer.set_parameter("weight", t(vec![0.0; 1], &[1])),
                    Err(AutodiffError::InvalidArgument(_))
                ));
                assert!(matches!(
                    layer.set_parameter("parametrizations.weight.original0", t(vec![1.0], &[1])),
                    Err(AutodiffError::Shape(_))
                ));
                // bias は従来どおり受理される。
                let b_shape = layer.bias().unwrap().shape().to_vec();
                layer
                    .set_parameter("bias", t(vec![0.25; b_shape.iter().product()], &b_shape))
                    .unwrap();

                // 非再パラメータ化の層は新キーを受理しない。
                let mut plain = make();
                assert!(matches!(
                    plain.set_parameter("parametrizations.weight.original0", t(vec![1.0], &[1])),
                    Err(AutodiffError::InvalidArgument(_))
                ));
            }

            #[test]
            fn failed_parametrize_leaves_layer_unchanged() {
                let mut layer = make();
                let rank = layer.weight().rank();
                assert!(layer.parametrize_weight_norm(Some(rank)).is_err());
                assert!(layer.parametrize_spectral_norm(0, 1e-12, SEED).is_err());
                assert!(
                    layer
                        .parametrize_spectral_norm(1, f32::INFINITY, SEED)
                        .is_err()
                );
                assert!(!layer.is_parametrized());
                assert_eq!(names(&layer), ["weight", "bias"]);

                layer.parametrize_weight_norm(None).unwrap();
                assert!(layer.parametrize_weight_norm(None).is_err());
                assert!(layer.parametrize_spectral_norm(1, 1e-12, SEED).is_err());
                assert_eq!(
                    layer.weight_norm_dim(),
                    Some(None),
                    "二重適用の拒否後も不変"
                );
            }

            #[test]
            fn spectral_state_advances_only_in_train_and_not_on_failed_forward() {
                let x = input();
                let mut layer = make().with_spectral_norm(1, 1e-12, SEED).unwrap();
                let s0 = layer.spectral_norm_state().unwrap();
                forward_bits(&layer, &x).unwrap();
                let s1 = layer.spectral_norm_state().unwrap();
                assert_ne!(s0, s1, "train の forward は状態を進める");

                // shape 不一致入力（rank 違い）の失敗 forward は状態を進めない。
                let before = layer.spectral_norm_state().unwrap();
                let bad = t(vec![0.0; 4], &[2, 2]);
                assert!(forward_bits(&layer, &bad).is_err());
                assert_eq!(before, layer.spectral_norm_state().unwrap());

                // eval: 状態は進まず、出力も再現する。
                layer.set_training(false);
                assert!(!layer.training());
                let frozen = layer.spectral_norm_state().unwrap();
                let y1 = forward_bits(&layer, &x).unwrap();
                let y2 = forward_bits(&layer, &x).unwrap();
                assert_eq!(frozen, layer.spectral_norm_state().unwrap());
                assert_eq!(y1, y2);

                // train へ戻すと再び進む。
                layer.set_training(true);
                forward_bits(&layer, &x).unwrap();
                assert_ne!(frozen, layer.spectral_norm_state().unwrap());
            }

            #[test]
            fn non_parametrized_forward_bound_is_bit_identical_to_forward() {
                let x = input();
                let layer = make();
                assert!(
                    layer.training(),
                    "非再パラメータ化の training() は常に true"
                );
                let mut layer2 = make();
                layer2.set_training(false);
                assert!(layer2.training(), "非再パラメータ化はモードを保持しない");
                let tape = Tape::new_with_ops(common::naive_ops());
                let xv = tape.var(&x);
                let vars = layer.bind(&tape);
                let want = bits(&vars.forward(&xv).unwrap().to_tensor());
                assert_eq!(forward_bits(&layer, &x).unwrap(), want);
            }

            #[test]
            fn host_paths_never_read_raw_weight_as_effective_weight() {
                let ops = common::naive_ops();
                let x = input();
                for layer in [
                    make().with_weight_norm(Some(0)).unwrap(),
                    make().with_spectral_norm(1, 1e-12, SEED).unwrap(),
                ] {
                    assert!(!Module::supports_forward_host(&layer));
                    assert!(matches!(
                        layer.forward_host(&*ops, &x),
                        Err(AutodiffError::Backend(BackendError::Unsupported(_)))
                    ));
                }
                let plain = make();
                assert!(Module::supports_forward_host(&plain));
                assert!(plain.forward_host(&*ops, &x).is_ok());
            }
        }
    };
}

conv_reparam_tests!(
    conv2d,
    Conv2d,
    Conv2d::new(4, 6, [3, 2], [1, 2], [1, 0], [1, 1], 2, true, SEED).unwrap(),
    [2, 4, 6, 7],
    spectral_dim = 0,
    u_len = 6,
    wn_dims = [None, Some(0), Some(1), Some(3)]
);
conv_reparam_tests!(
    conv_transpose2d,
    ConvTranspose2d,
    ConvTranspose2d::new(4, 6, [3, 2], [2, 1], [1, 0], [1, 0], [1, 1], 2, true, SEED).unwrap(),
    [2, 4, 5, 4],
    spectral_dim = 1,
    u_len = 3,
    wn_dims = [None, Some(0), Some(1), Some(3)]
);
conv_reparam_tests!(
    conv1d,
    Conv1d,
    Conv1d::new(4, 6, 3, 2, 1, 1, 2, true, SEED).unwrap(),
    [2, 4, 9],
    spectral_dim = 0,
    u_len = 6,
    wn_dims = [None, Some(0), Some(2)]
);
conv_reparam_tests!(
    conv_transpose1d,
    ConvTranspose1d,
    ConvTranspose1d::new(4, 6, 3, 2, 1, 1, 1, 2, true, SEED).unwrap(),
    [2, 4, 5],
    spectral_dim = 1,
    u_len = 3,
    wn_dims = [None, Some(0), Some(1), Some(2)]
);
conv_reparam_tests!(
    conv3d,
    Conv3d,
    Conv3d::new(
        4,
        6,
        [2, 2, 2],
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1],
        2,
        true,
        SEED
    )
    .unwrap(),
    [1, 4, 3, 4, 4],
    spectral_dim = 0,
    u_len = 6,
    wn_dims = [None, Some(0), Some(4)]
);
conv_reparam_tests!(
    conv_transpose3d,
    ConvTranspose3d,
    ConvTranspose3d::new(
        4,
        6,
        [2, 2, 2],
        [2, 1, 1],
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 1],
        2,
        true,
        SEED
    )
    .unwrap(),
    [1, 4, 2, 3, 3],
    spectral_dim = 1,
    u_len = 3,
    wn_dims = [None, Some(0), Some(1), Some(4)]
);
