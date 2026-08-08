//! CrossEntropy 損失（`Var::cross_entropy_loss`・`nn::loss::
//! CrossEntropyLoss`）の受け入れ条件検証（イシュー #191・親イシュー
//! #189）。
//!
//! **PyTorch 参照値の生成方法**: `torch==2.13.0+cpu`
//! （scratchpad venv 限定インストール。リポ・CI には torch を持ち込ま
//! ない。`.claude/rules/security.md` A06「参照値生成用 torch は
//! scratchpad venv 限定」・CI は Python に依存しない契約
//! 〈`tests/poc_v2_2_parity.rs` と同型〉）で
//! `torch.nn.functional.cross_entropy`（f32・`reduction='mean'/'sum'`）
//! の loss 値・`logits.grad` を出力し、本ファイルへ定数として埋め込む
//! （`tests/poc_v2_2_parity.rs` の evidence 埋め込み先例）。生成
//! スクリプト全文:
//!
//! ```python
//! import torch
//! import torch.nn.functional as F
//!
//! def run(name, logits_data, shape, targets_data, target_shape, class_dim):
//!     logits = torch.tensor(logits_data, dtype=torch.float32).reshape(shape)
//!     targets = torch.tensor(targets_data, dtype=torch.long).reshape(target_shape)
//!     for reduction in ("mean", "sum"):
//!         x = logits.clone().requires_grad_(True)
//!         x_moved = x.movedim(class_dim, 1) if class_dim != 1 else x
//!         loss = F.cross_entropy(x_moved, targets, reduction=reduction)
//!         loss.backward()
//!         print(name, reduction, loss.item(), x.grad.reshape(-1).tolist())
//!
//! run("case1_basic_4x3",
//!     [1.0, 2.0, 0.5, 0.1, -0.5, 2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
//!     [4, 3], [1, 2, 0, 1], [4], class_dim=1)
//! run("case2_class_dim1_2x3x4",
//!     [0.5, -1.0, 2.0, 0.3, 1.5, 0.2, -0.5, 1.0, -1.0, 0.5, 1.0, -0.3,
//!      0.1, 0.2, -0.3, 0.4, -0.5, 1.5, 0.0, -1.0, 2.0, -1.0, 0.5, 0.3],
//!     [2, 3, 4], [0, 2, 1, 0, 1, 2, 0, 1], [2, 4], class_dim=1)
//! run("case3_large_magnitude",
//!     [1000.0, 1000.1, 999.9, -1000.0, -999.9, -1000.2,
//!      10000.0, 9999.5, 10000.2],
//!     [3, 3], [1, 0, 2], [3], class_dim=1)
//! ```
//!
//! **判定基準**: 損失値・勾配とも承認済み複合判定「相対誤差 1e-3 未満
//! または絶対誤差 1e-5 未満」（`.claude/rules/coding-rust.md`。バック
//! エンド間数値一致判定を PyTorch 参照値との突合にもそのまま適用し、
//! 新規閾値は発明しない）。

// 下記の参照値定数は生成スクリプト（PyTorch）が出力した値をそのまま
// 埋め込んだ evidence であり、f32 精度への丸め桁で切り詰めると生成元
// との対応が読み取りにくくなる（`.claude/rules/coding-rust.md`
// 「`#[allow]` の安易な追加で黙らせない」の例外として、値の出典
// （evidence）忠実性を優先する理由をここに明記する）。
#![allow(clippy::excessive_precision)]

mod common;

use autodiff::{AutodiffError, Tape};
use tensor_core::Tensor;

use autodiff::nn::loss::{CrossEntropyLoss, Reduction};

// 承認済み複合判定（`.claude/rules/coding-rust.md`）: 相対誤差 1e-3
// 未満または絶対誤差 1e-5 未満。
const REL_TOL: f32 = 1e-3;
const ABS_TOL: f32 = 1e-5;

fn f32_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data.to_vec(), shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn i32_tensor(data: &[i32], shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data.to_vec(), shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense(tensor: &Tensor<f32>) -> Vec<f32> {
    tensor
        .contiguous()
        .as_slice()
        .expect("test fixture: contiguous() 後の as_slice() は Some のはず")
        .to_vec()
}

fn assert_close(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 要素数が一致しない");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel = diff / a.abs().max(e.abs()).max(1e-12);
        assert!(
            rel < REL_TOL || diff < ABS_TOL,
            "{label}[{i}]: actual={a} expected={e} diff={diff} rel={rel}"
        );
    }
}

fn assert_scalar_close(label: &str, actual: f32, expected: f32) {
    assert_close(label, &[actual], &[expected]);
}

// --- 1. PyTorch 参照値一致（受け入れ条件本体）: logits [4,3]・targets [4] ---

#[test]
fn matches_pytorch_reference_case1_basic_mean() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let logits = tape.var(&f32_tensor(
        &[1.0, 2.0, 0.5, 0.1, -0.5, 2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
        &[4, 3],
    ));
    let targets = i32_tensor(&[1, 2, 0, 1], &[4]);

    let loss = logits
        .cross_entropy_loss(&targets, 1, Reduction::Mean)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dlogits = grads
        .get(&logits)
        .unwrap()
        .expect("logits は loss に到達する");

    assert_scalar_close("case1 mean loss", dense(&loss.to_tensor())[0], 1.1219844818);
    assert_close(
        "case1 mean grad",
        &dense(dlogits),
        &[
            0.0578059703,
            -0.0928670764,
            0.0350610949,
            0.030359311,
            0.0166615453,
            -0.0470208526,
            -0.2274923623,
            0.0611821227,
            0.1663102508,
            0.1663102508,
            -0.1888178736,
            0.0225076452,
        ],
    );
}

#[test]
fn matches_pytorch_reference_case1_basic_sum() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let logits = tape.var(&f32_tensor(
        &[1.0, 2.0, 0.5, 0.1, -0.5, 2.0, -1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
        &[4, 3],
    ));
    let targets = i32_tensor(&[1, 2, 0, 1], &[4]);

    let loss = logits
        .cross_entropy_loss(&targets, 1, Reduction::Sum)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dlogits = grads
        .get(&logits)
        .unwrap()
        .expect("logits は loss に到達する");

    assert_scalar_close("case1 sum loss", dense(&loss.to_tensor())[0], 4.4879379272);
    assert_close(
        "case1 sum grad",
        &dense(dlogits),
        &[
            0.2312238812,
            -0.3714683056,
            0.1402443796,
            0.1214372441,
            0.066646181,
            -0.1880834103,
            -0.909969449,
            0.2447284907,
            0.665241003,
            0.665241003,
            -0.7552714944,
            0.0900305808,
        ],
    );
}

// --- 2. クラス次元指定: logits [2,3,4]・class_dim=1・targets [2,4] ---

#[test]
fn matches_pytorch_reference_case2_class_dim1_mean() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let logits = tape.var(&f32_tensor(
        &[
            0.5, -1.0, 2.0, 0.3, 1.5, 0.2, -0.5, 1.0, -1.0, 0.5, 1.0, -0.3, 0.1, 0.2, -0.3, 0.4,
            -0.5, 1.5, 0.0, -1.0, 2.0, -1.0, 0.5, 0.3,
        ],
        &[2, 3, 4],
    ));
    let targets = i32_tensor(&[0, 2, 1, 0, 1, 2, 0, 1], &[2, 4]);

    let loss = logits
        .cross_entropy_loss(&targets, 1, Reduction::Mean)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dlogits = grads
        .get(&logits)
        .unwrap()
        .expect("logits は loss に到達する");

    assert_scalar_close("case2 mean loss", dense(&loss.to_tensor())[0], 1.9234025478);
    assert_close(
        "case2 mean grad",
        &dense(dlogits),
        &[
            -0.0932854712,
            0.0142016327,
            0.0862090066,
            -0.0899129137,
            0.0862090141,
            0.0471510738,
            -0.1179235354,
            0.0706567094,
            0.007076466,
            -0.0613527074,
            0.0317145213,
            0.0192562025,
            0.0151796555,
            0.0251484253,
            -0.0976799801,
            0.0581007712,
            -0.1166692302,
            0.0922770202,
            0.0368781649,
            -0.1106725261,
            0.1014895737,
            -0.1174254417,
            0.0608018152,
            0.0525717549,
        ],
    );
}

#[test]
fn matches_pytorch_reference_case2_class_dim1_sum() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let logits = tape.var(&f32_tensor(
        &[
            0.5, -1.0, 2.0, 0.3, 1.5, 0.2, -0.5, 1.0, -1.0, 0.5, 1.0, -0.3, 0.1, 0.2, -0.3, 0.4,
            -0.5, 1.5, 0.0, -1.0, 2.0, -1.0, 0.5, 0.3,
        ],
        &[2, 3, 4],
    ));
    let targets = i32_tensor(&[0, 2, 1, 0, 1, 2, 0, 1], &[2, 4]);

    let loss = logits
        .cross_entropy_loss(&targets, 1, Reduction::Sum)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dlogits = grads
        .get(&logits)
        .unwrap()
        .expect("logits は loss に到達する");

    assert_scalar_close("case2 sum loss", dense(&loss.to_tensor())[0], 15.3872203827);
    assert_close(
        "case2 sum grad",
        &dense(dlogits),
        &[
            -0.7462837696,
            0.1136130616,
            0.6896720529,
            -0.7193033099,
            0.6896721125,
            0.3772085905,
            -0.9433882833,
            0.565253675,
            0.0566117279,
            -0.4908216596,
            0.2537161708,
            0.15404962,
            0.1214372441,
            0.201187402,
            -0.7814398408,
            0.4648061693,
            -0.9333538413,
            0.7382161617,
            0.2950253189,
            -0.8853802085,
            0.8119165897,
            -0.9394035339,
            0.4864145219,
            0.4205740392,
        ],
    );
}

// --- 3. log-sum-exp 安定性: 大振幅 logits（±1e3〜1e4）でも有限・参照値一致 ---

#[test]
fn stable_for_large_magnitude_logits() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let logits = tape.var(&f32_tensor(
        &[
            1000.0, 1000.1, 999.9, -1000.0, -999.9, -1000.2, 10000.0, 9999.5, 10000.2,
        ],
        &[3, 3],
    ));
    let targets = i32_tensor(&[1, 0, 2], &[3]);

    let loss_mean = logits
        .cross_entropy_loss(&targets, 1, Reduction::Mean)
        .unwrap();
    let loss_value = dense(&loss_mean.to_tensor())[0];
    assert!(
        loss_value.is_finite(),
        "大振幅 logits で loss が非有限になった（log-sum-exp 安定化の破綻）: {loss_value}"
    );
    assert_scalar_close("case3 mean loss", loss_value, 0.9714357853);

    let grads = tape.backward(&loss_mean).unwrap();
    let dlogits = grads
        .get(&logits)
        .unwrap()
        .expect("logits は loss に到達する");
    let grad_data = dense(dlogits);
    assert!(
        grad_data.iter().all(|g| g.is_finite()),
        "大振幅 logits で勾配が非有限になった: {grad_data:?}"
    );
    assert_close(
        "case3 mean grad",
        &grad_data,
        &[
            0.1107418388,
            -0.2109476775,
            0.1002058014,
            -0.2193289697,
            0.1259912252,
            0.093337737,
            0.1178617477,
            0.0714867711,
            -0.1893485487,
        ],
    );

    let tape_sum = Tape::new_with_ops(common::naive_ops());
    let logits_sum = tape_sum.var(&f32_tensor(
        &[
            1000.0, 1000.1, 999.9, -1000.0, -999.9, -1000.2, 10000.0, 9999.5, 10000.2,
        ],
        &[3, 3],
    ));
    let loss_sum = logits_sum
        .cross_entropy_loss(&targets, 1, Reduction::Sum)
        .unwrap();
    assert_scalar_close(
        "case3 sum loss",
        dense(&loss_sum.to_tensor())[0],
        2.9143073559,
    );
}

// --- 4a. grad check（f64 中央差分。`grad.rs` の grad-check テストと
//     同一パラメータ H/TAU/REL_TOL/ABS_TOL を再利用。ユーザー承認待ち
//     の新規閾値は導入しない: Issue #223） ---

const H: f64 = 1e-3;
const GC_TAU: f32 = 1e-4;
const GC_REL_TOL: f32 = 1e-2;
const GC_ABS_TOL: f32 = 1e-3;

fn numeric_grad_cross_entropy(
    logits: &Tensor<f32>,
    targets: &Tensor<i32>,
    class_dim: usize,
    reduction: Reduction,
) -> Vec<f32> {
    let shape = logits.shape().to_vec();
    let mut data: Vec<f32> = logits
        .contiguous()
        .as_slice()
        .expect("test fixture: contiguous")
        .to_vec();
    let mut grad = vec![0f32; data.len()];

    let eval_loss = |data: &[f32]| -> f64 {
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&f32_tensor(data, &shape));
        let loss = x.cross_entropy_loss(targets, class_dim, reduction).unwrap();
        dense(&loss.to_tensor())[0] as f64
    };

    for i in 0..data.len() {
        let orig = data[i] as f64;
        data[i] = (orig + H) as f32;
        let lp = eval_loss(&data);
        data[i] = (orig - H) as f32;
        let lm = eval_loss(&data);
        data[i] = orig as f32;
        grad[i] = ((lp - lm) / (2.0 * H)) as f32;
    }
    grad
}

#[test]
fn grad_matches_numeric_central_difference() {
    let logits = f32_tensor(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
    let targets = i32_tensor(&[2, 0], &[2]);

    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&logits);
    let loss = x.cross_entropy_loss(&targets, 1, Reduction::Mean).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let analytic = dense(grads.get(&x).unwrap().expect("logits は loss に到達する"));

    let numeric = numeric_grad_cross_entropy(&logits, &targets, 1, Reduction::Mean);

    for (i, (&a, &n)) in analytic.iter().zip(numeric.iter()).enumerate() {
        let diff = (a - n).abs();
        let rel = diff / a.abs().max(n.abs()).max(GC_TAU);
        assert!(
            rel <= GC_REL_TOL || diff <= GC_ABS_TOL,
            "grad_check[{i}]: analytic={a} numeric={n} diff={diff} rel={rel}"
        );
    }
}

// --- 5. 薄いラッパー性: nn::loss::CrossEntropyLoss::forward が
//     Var::cross_entropy_loss 直接呼び出しと同一値・同一ノード追記数 ---

#[test]
fn nn_loss_wrapper_matches_var_method_directly() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5, -1.0, 0.0, 1.0], &[2, 3]));
    let targets = i32_tensor(&[1, 2], &[2]);
    let before = tape.len();

    let module = CrossEntropyLoss {
        class_dim: 1,
        reduction: Reduction::Mean,
    };
    let via_module = module.forward(&x, &targets).unwrap();
    let via_var = x.cross_entropy_loss(&targets, 1, Reduction::Mean).unwrap();

    assert_eq!(
        tape.len(),
        before + 2,
        "forward 呼び出しごとに 1 ノード追記"
    );
    assert_eq!(dense(&via_module.to_tensor()), dense(&via_var.to_tensor()));
}

// --- 6. エラー経路（panic しない・variant 固定） ---

#[test]
fn errors_when_class_dim_out_of_range() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5, -1.0, 0.0, 1.0], &[2, 3]));
    let targets = i32_tensor(&[1, 2], &[2]);

    let err = x
        .cross_entropy_loss(&targets, 5, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn errors_when_targets_shape_mismatches() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5, -1.0, 0.0, 1.0], &[2, 3]));
    // 期待 shape は [2]（class_dim=1 を除去）だが [3] を渡す。
    let targets = i32_tensor(&[1, 2, 0], &[3]);

    let err = x
        .cross_entropy_loss(&targets, 1, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

#[test]
fn errors_when_target_index_out_of_range_high() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5, -1.0, 0.0, 1.0], &[2, 3]));
    // class_dim=1 のサイズは 3（0..3 が妥当）だが 3 を渡す（範囲外）。
    let targets = i32_tensor(&[3, 0], &[2]);

    let err = x
        .cross_entropy_loss(&targets, 1, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn errors_when_target_index_negative() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0, 2.0, 0.5, -1.0, 0.0, 1.0], &[2, 3]));
    let targets = i32_tensor(&[-1, 0], &[2]);

    let err = x
        .cross_entropy_loss(&targets, 1, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn errors_for_rank0_logits() {
    let tape = Tape::new_with_ops(common::naive_ops());
    let x = tape.var(&f32_tensor(&[1.0], &[]));
    let targets = i32_tensor(&[0], &[]);

    let err = x
        .cross_entropy_loss(&targets, 0, Reduction::Mean)
        .unwrap_err();
    assert!(matches!(err, AutodiffError::Shape(_)));
}

// --- 7. 合成 end-to-end: Linear → cross_entropy_loss → backward が
//     weight/bias/input へ勾配を返す（学習最小構成 #189 の目的線上） ---

#[test]
fn end_to_end_linear_then_cross_entropy_backward_reaches_all_params() {
    let linear = autodiff::nn::Linear::new(3, 4, true, 42).unwrap();
    let tape = Tape::new_with_ops(common::naive_ops());
    let vars = linear.bind(&tape);

    let input = tape.var(&f32_tensor(&[0.5, -1.0, 2.0, 1.0, 0.0, -0.5], &[2, 3]));
    let targets = i32_tensor(&[1, 3], &[2]);

    let logits = vars.forward(&input).unwrap();
    let loss = logits
        .cross_entropy_loss(&targets, 1, Reduction::Mean)
        .unwrap();

    let grads = tape.backward(&loss).unwrap();

    assert!(
        grads.get(&vars.weight).unwrap().is_some(),
        "weight に勾配が到達する"
    );
    assert!(
        grads
            .get(vars.bias.as_ref().expect("bias=true で構築した"))
            .unwrap()
            .is_some(),
        "bias に勾配が到達する"
    );
    assert!(
        grads.get(&input).unwrap().is_some(),
        "input に勾配が到達する"
    );

    let loss_value = dense(&loss.to_tensor())[0];
    assert!(loss_value.is_finite());
}
