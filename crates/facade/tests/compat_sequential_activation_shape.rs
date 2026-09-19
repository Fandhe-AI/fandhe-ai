//! `compat::Sequential::add_softmax`／`add_log_softmax`／`add_gelu`／
//! `add_gelu_tanh`／`add_softplus`／`add_flatten`（イシュー #2065・親
//! #2059）の facade 公開面を検証する統合テスト（`compat_sequential_
//! layers.rs` と同型。CPU のみで Linux 実行可能）。
//!
//! - `add_softplus` の無効引数拒否。
//! - 遅延検査: `Softmax::new`／`Flatten::new` 自体は infallible だが、
//!   構築後の `forward`／`predict` は入力 rank 不整合を
//!   `AutodiffError::Shape` で拒否する。
//! - `predict`（tape 不要経路）と `forward`（`fandhe_ai::tape()` 上）が
//!   bit 完全一致（混在モデル。`Linear → ReLU → Flatten → Softmax →
//!   Linear`）。
//! - `bind().forward` + `Tape::backward` で入力勾配が伝播する。
//! - 無状態層の確認: `trainable_parameters()`／`named_parameters()` が
//!   `Linear` 分のみを返す。
//! - 常駐経路の実地検証: 新規 6 種を含むモデルでも
//!   `init_device_param_store`／`predict_resident` が成功し `predict`
//!   と bit 完全一致する（Conv／Norm／Embedding／Attention／Pooling と
//!   異なり、これら 6 層は `contains_resident_unsupported_layer` の
//!   allowlist に含まれないため常駐経路が使える）。
//! - `Softplus` の PyTorch 既定値 `(1.0, 20.0)` との一致確認。

use fandhe_ai::compat::Sequential;
use fandhe_ai::{AutodiffError, Tensor};

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

const SEED1: u64 = 0x5555_6666;
const SEED2: u64 = 0x7777_8888;

// --- add_softplus の無効引数拒否 ---

#[test]
fn add_softplus_rejects_zero_beta() {
    let err = Sequential::new()
        .add_softplus(0.0, 20.0)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_softplus_rejects_nan_beta() {
    let err = Sequential::new()
        .add_softplus(f32::NAN, 20.0)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn add_softplus_rejects_non_finite_threshold() {
    let err = Sequential::new()
        .add_softplus(1.0, f32::INFINITY)
        .map(|_| ())
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// --- 遅延検査: 構築は成功するが forward 時に rank 不整合を拒否 ---

#[test]
fn add_softmax_dim_out_of_range_is_rejected_at_forward_time() {
    // `Softmax::new(5)` 自体は infallible（構築時には検査しない）。
    let model = Sequential::new().add_softmax(5);
    let x = tensor(vec![1.0, 2.0, 3.0], &[3]);
    let err = model.predict(&x).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn add_flatten_end_dim_out_of_range_is_rejected_at_forward_time() {
    let model = Sequential::new().add_flatten(2, 5);
    let x = tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let err = model.predict(&x).unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

// `predict`（`Flatten::forward_host` → `Tensor::reshape`。`tensor.rs`
// 参照）と `forward`（`Var::flatten` → `Var::reshape`。`var.rs` 参照）が
// 非 contiguous 入力（`transpose` 後の view）を **同一エラー**
// （`ShapeError::NonContiguousReshape`）で拒否することを確認する
// （判定迂回経路が生じていないことの直接検証。`Sequential::add_flatten`
// doc・`tensor-core::flatten_out_shape` doc の A08 記述の裏付け）。
#[test]
fn add_flatten_on_non_contiguous_input_is_rejected_identically_via_both_paths() {
    let base = tensor((0..6).map(|i| i as f32).collect(), &[2, 3]);
    let transposed = base
        .transpose(0, 1)
        .expect("test fixture: transpose(0,1) は shape [3,2] へ成功するはず");
    assert!(
        !transposed.is_contiguous(),
        "test fixture: transpose 後は非 contiguous のはず"
    );

    let model = Sequential::new().add_flatten(0, 1);

    let predict_err = model.predict(&transposed).unwrap_err();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&transposed);
    let forward_err = model.forward(&tape, &xv).unwrap_err();

    assert!(
        matches!(
            predict_err,
            AutodiffError::Shape(fandhe_ai::ShapeError::NonContiguousReshape)
        ),
        "predict（tape 不要経路）は NonContiguousReshape で拒否するはず: {predict_err:?}"
    );
    assert!(
        matches!(
            forward_err,
            AutodiffError::Shape(fandhe_ai::ShapeError::NonContiguousReshape)
        ),
        "forward（tape 経路）は NonContiguousReshape で拒否するはず: {forward_err:?}"
    );
}

// --- predict と forward（外部 tape）の bit 完全一致 ---

fn mixed_model_input() -> Tensor<f32> {
    // `nn::Linear::forward` は 2 次元厳密版 `gemm_out_shape` を経由する
    // ため（`nn/linear.rs::Linear::forward` doc）、rank 3 入力を直接
    // 渡せない——CNN 出力（`[N, C, H]` 等）を `Linear` へ渡す前に
    // `Flatten` で 2 次元へ潰すのが本層の主要ユースケース
    // （`Sequential::add_flatten` doc「典型例」参照）。
    tensor(
        (0..2 * 2 * 3).map(|i| i as f32 * 0.1 - 0.7).collect(),
        &[2, 2, 3],
    )
}

fn mixed_model(seed1: u64, seed2: u64) -> Sequential {
    // [N=2, C=2, H=3] → Flatten(1, 2) → [2, 6] → Linear(6, 4) → ReLU →
    // Softmax(1) → Linear(4, 2)。
    Sequential::new()
        .add_flatten(1, 2)
        .add_linear(6, 4, seed1)
        .unwrap()
        .add_relu()
        .add_softmax(1)
        .add_linear(4, 2, seed2)
        .unwrap()
}

#[test]
fn mixed_model_predict_matches_forward_bit_exact() {
    let model = mixed_model(SEED1, SEED2);
    let x = mixed_model_input();

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(predicted.shape(), &[2, 2]);
    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn gelu_gelu_tanh_softplus_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_linear(4, 4, SEED1)
        .unwrap()
        .add_gelu()
        .add_linear(4, 4, SEED2)
        .unwrap()
        .add_gelu_tanh()
        .add_softplus(1.0, 20.0)
        .unwrap();
    let x = tensor((0..3 * 4).map(|i| i as f32 * 0.2 - 0.5).collect(), &[3, 4]);

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

#[test]
fn log_softmax_predict_matches_forward_bit_exact() {
    let model = Sequential::new()
        .add_linear(4, 3, SEED1)
        .unwrap()
        .add_log_softmax(1);
    let x = tensor((0..2 * 4).map(|i| i as f32 * 0.3 - 0.6).collect(), &[2, 4]);

    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();

    assert_eq!(dense_vec(&predicted), dense_vec(&forwarded));
}

// --- bind().forward + Tape::backward: 入力勾配が伝播する ---

#[test]
fn mixed_model_forward_on_external_tape_reaches_backward() {
    let model = mixed_model(SEED1, SEED2);
    let tape = fandhe_ai::tape();
    let x = mixed_model_input();
    let input = tape.var(&x);
    let output = model.forward(&tape, &input).unwrap();
    let loss = output.sum(None).unwrap();

    let grads = tape.backward(&loss).unwrap();
    let input_grad = grads
        .get(&input)
        .unwrap()
        .expect("入力ノードは loss に寄与している");
    assert_eq!(input_grad.shape(), x.shape());
}

// --- 無状態層の確認: trainable_parameters／named_parameters は Linear 分のみ ---

#[test]
fn mixed_model_trainable_parameters_only_contains_linear_weights() {
    let model = mixed_model(SEED1, SEED2);
    let named = model.named_parameters();
    let trainable = model.trainable_parameters();

    // Linear(3,4,bias) + Linear(4,2,bias) = weight/bias × 2 層 = 4 件。
    // Flatten／Softmax は named_parameters／trainable_parameters へ
    // 一切寄与しない（無状態層）。
    assert_eq!(named.len(), 4);
    assert_eq!(trainable.len(), 4);
    assert_eq!(
        named
            .iter()
            .map(|(_, t)| t.shape().to_vec())
            .collect::<Vec<_>>(),
        trainable
            .iter()
            .map(|t| t.shape().to_vec())
            .collect::<Vec<_>>()
    );
}

// --- 常駐経路の実地検証 ---
//
// Conv／Norm／Embedding／Attention／Pooling と異なり、Softmax／
// LogSoftmax／Gelu／GeluTanh／Softplus／Flatten は
// `contains_resident_unsupported_layer` の allowlist（`as_conv2d` 等）
// に含まれないフックなし層のため、常駐経路（`init_device_param_
// store`／`predict_resident`）が構造的に使える（`forward_from_flat_
// leaves` の汎用 `layer.forward(tape, &current)` 分岐を経由する）。
// もしこの経路が失敗する場合は原因を特定して対処する
// （`Module::supports_forward_host` を安易に `false` へ倒し
// `predict` の tape-free 経路自体を無効化しない。CNN→Flatten→Linear は
// 本層の主要ユースケースであるため）。

#[test]
fn mixed_model_init_device_param_store_succeeds_with_flatten_and_softmax() {
    let model = mixed_model(SEED1, SEED2);
    let tape = fandhe_ai::tape();
    model
        .init_device_param_store(&tape)
        .expect("Flatten／Softmax は常駐経路ガードの対象外のため成功するはず");
}

#[test]
fn mixed_model_predict_resident_matches_predict() {
    let model = mixed_model(SEED1, SEED2);
    let x = mixed_model_input();

    let init_tape = fandhe_ai::tape();
    let store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let via_resident = model.predict_resident(&store, &x).unwrap();
    let via_predict = model.predict(&x).unwrap();

    assert_eq!(dense_vec(&via_resident), dense_vec(&via_predict));
}

// --- Softplus の PyTorch 既定値 (beta=1.0, threshold=20.0) 確認 ---
//
// `Softplus(x) = ln(1 + exp(x))`（`beta * x <= threshold` の通常域）。
// `x=0` の解析解 `ln(2) ≈ 0.6931471805599453` との一致を確認する
// （PyTorch `nn.Softplus()(torch.tensor(0.0))` も同値）。

#[test]
fn softplus_pytorch_defaults_match_analytic_value_at_zero() {
    let model = Sequential::new().add_softplus(1.0, 20.0).unwrap();
    let x = tensor(vec![0.0], &[1]);

    let predicted = model.predict(&x).unwrap();

    let ln2 = std::f32::consts::LN_2;
    let got = dense_vec(&predicted)[0];
    assert!(
        (got - ln2).abs() < 1e-6,
        "softplus(0) = ln(2) のはずが {got} だった"
    );
}
