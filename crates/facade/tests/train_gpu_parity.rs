//! NN 層・最適化・ONNX の学習の実機 parity テストの登録表と実機テスト入口
//! （イシュー #3778。親 #3772）。
//!
//! **役割**: 比較・記録の部品は `tests/common/train_gpu_parity.rs` にあり、本ファイルは
//! 登録表 `CASES`（族ごとの測定ケース）と、族ごとに 1 本ずつの `#[ignore]` 実機テスト、
//! および GPU を使わず CPU 同士で部品の健全性を見る CI 自己検査を持つ。
//!
//! **置き場が facade である理由**: 全バックエンドを合成する公開経路
//! （`fandhe_ai::tape_for`）をそのまま通すため（`create_graph_gpu_parity.rs` と同じ）。
//!
//! **族と Phase 10 の測定 issue**:
//! - `scaffold`: 雛形の自己検査用（実機の測定対象ではない）
//! - `nn`: NN 層の forward・backward（#3860 GB10・#3861 Mac mini）
//! - `optim`: param groups の常駐更新（#3862・#3863）
//! - `onnx`: ONNX の学習（#3864・#3865。facade 公開形 `OnnxModel::bind_autograd`
//!   〈#3842〉につないだ。11 op はホスト実行のため route は H）
//!
//! `nn`・`optim` の行は #4026 で登録した（それまでは登録 0 件で undetermined になり、
//! 実機テストが構造上 pass できなかった）。実機の parity は #3860〜#3863 で未実測。
//! nn 族は「forward → MSE → backward → ホスト SGD 1 step → 更新後の loss」の学習ステップ、
//! optim 族は param groups 付き常駐更新とホスト参照（`ParamGroupStep`）の比較。
//! 実機テストの名前は固定で、Phase 10 の issue と RULE.txt が参照する。ROCm は対象外
//! （#3779・#3853）。
//!
//! 実行（Phase 10。1 本ずつ別プロセス）:
//! ```sh
//! cargo test -p fandhe-ai --test train_gpu_parity -- --ignored --exact --nocapture \
//!   --test-threads=1 train_parity_nn_cuda_matches_cpu
//! ```
//!
//! 判定は `compare`（REQ-2）または構造上の bit 一致のみ。tolerance・baseline は不変。

#[path = "common/train_gpu_parity.rs"]
mod harness;

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::interop::onnx::{OnnxAutogradOptions, OnnxModel};
use fandhe_ai::optim::{AdamConfig, AdamWConfig};
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig, Var};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::nn::optim::{Adam, AdamW, ParamGroup, ParamGroupStep};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::Tensor;
use harness::{Case, Judge, Quantity, Route};

const FAMILIES: &[&str] = &["scaffold", "nn", "optim", "onnx"];

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;
const SGD_STEPS: usize = 20;
const SGD_LR: f32 = 0.02;
/// nn 族の学習ステップ（ホスト SGD 1 step）の学習率。
const NN_LR: f32 = 0.05;
/// nn 族の Var グラフ行の入力・パラメータ用の種。
const SEED_VAR: u64 = 0x00DD_BA11;
/// optim 族の学習 step 数。
const OPTIM_STEPS: usize = 20;

/// nn 族の行（全行 Req2・参照腕なし＝CPU の同一関数）。
macro_rules! nn_case {
    ($name:expr, $route:expr, $shape:expr, $run:expr) => {
        Case {
            name: $name,
            family: "nn",
            route: $route,
            judge: Judge::Req2,
            shape: $shape,
            run: $run,
            reference: None,
        }
    };
}

/// optim 族の行（全行 Req2・腕 B はホスト参照）。
macro_rules! optim_case {
    ($name:expr, $shape:expr, $run:expr, $reference:expr) => {
        Case {
            name: $name,
            family: "optim",
            route: Route::K,
            judge: Judge::Req2,
            shape: $shape,
            run: $run,
            reference: Some($reference),
        }
    };
}

/// 登録表。行の足し方は `docs/perf/logs/train-gpu-parity-template/README.md`。
const CASES: &[Case] = &[
    Case {
        name: "scaffold_mlp_forward_backward_s1",
        family: "scaffold",
        route: Route::K,
        judge: Judge::Req2,
        shape: "S1: batch=4 d_in=8 hidden=16 d_out=4 seed=0xC0FFEE",
        run: |d| mlp_forward_backward(d, 4, 8, 16, 4),
        reference: None,
    },
    Case {
        name: "scaffold_mlp_forward_backward_s2",
        family: "scaffold",
        route: Route::K,
        judge: Judge::Req2,
        shape: "S2: batch=5 d_in=13 hidden=17 d_out=7 seed=0xC0FFEE",
        run: |d| mlp_forward_backward(d, 5, 13, 17, 7),
        reference: None,
    },
    Case {
        name: "scaffold_sgd_resident_vs_host",
        family: "scaffold",
        route: Route::K,
        judge: Judge::Req2,
        shape: "S1: batch=4 d_in=8 hidden=16 d_out=4 steps=20 lr=0.02 seed=0xC0FFEE",
        run: sgd_resident,
        reference: Some(sgd_host),
    },
    // onnx 族（#3842）。公開経路 `OnnxModel::bind_autograd` だけを通す。11 op の計算は
    // ホストで実行されるため route は H（テンプレ README の規則 4）。
    Case {
        name: "onnx_from_sequential_forward_backward_s1",
        family: "onnx",
        route: Route::H,
        judge: Judge::Req2,
        shape: "S1: batch=4 d_in=8 hidden=16 d_out=4 seed=0xC0FFEE",
        run: |d| onnx_forward_backward(d, 4, 8, 16, 4),
        reference: None,
    },
    Case {
        name: "onnx_from_sequential_forward_backward_s2",
        family: "onnx",
        route: Route::H,
        judge: Judge::Req2,
        shape: "S2: batch=5 d_in=13 hidden=17 d_out=7 seed=0xC0FFEE",
        run: |d| onnx_forward_backward(d, 5, 13, 17, 7),
        reference: None,
    },
    // nn 族（#4026）。追跡表の「実機 parity」列が #4026 待ちの行（7・9〜15）を登録する。
    // 全行 Req2（骨格に Linear／Conv の GEMM が入るため。Bit は使わない）。route は各決定記録に従う。
    nn_case!(
        "nn_upsample_scale_factor_train_step_s1",
        Route::K,
        "S1: x=[2,2,6,6] conv2d(2,3,k3,p1) upsample([2.0,1.5],bilinear) linear seed=0xC0FFEE lr=0.05",
        |d| seq_train_step(d, build_upsample, &[2, 2, 6, 6], 324)
    ),
    nn_case!(
        "nn_upsample_scale_factor_train_step_s2",
        Route::K,
        "S2: x=[3,2,7,5] conv2d(2,3,k3,p1) upsample([2.0,1.5],bilinear) linear seed=0xC0FFEE lr=0.05",
        |d| seq_train_step(d, build_upsample, &[3, 2, 7, 5], 294)
    ),
    nn_case!(
        "nn_shrink_train_step_s1",
        Route::H,
        "S1: x=[2,2,6,6] conv2d(2,3,k3,p1) hardshrink(0.05) softshrink(0.05) linear lr=0.05",
        |d| seq_train_step(d, build_shrink, &[2, 2, 6, 6], 108)
    ),
    nn_case!(
        "nn_shrink_train_step_s2",
        Route::H,
        "S2: x=[3,2,7,5] conv2d(2,3,k3,p1) hardshrink(0.05) softshrink(0.05) linear lr=0.05",
        |d| seq_train_step(d, build_shrink, &[3, 2, 7, 5], 105)
    ),
    nn_case!(
        "nn_batch_norm3d_train_step_s1",
        Route::K,
        "S1: x=[2,2,4,4,4] conv3d(2,3,k3,p1) batch_norm3d(training) relu linear lr=0.05",
        |d| seq_train_step(d, build_bn3d, &[2, 2, 4, 4, 4], 192)
    ),
    nn_case!(
        "nn_batch_norm3d_train_step_s2",
        Route::K,
        "S2: x=[3,2,3,5,4] conv3d(2,3,k3,p1) batch_norm3d(training) relu linear lr=0.05",
        |d| seq_train_step(d, build_bn3d, &[3, 2, 3, 5, 4], 180)
    ),
    nn_case!(
        "nn_channel_shuffle_train_step_s1",
        Route::H,
        "S1: x=[2,2,6,6] conv2d(2,4) channel_shuffle(2) conv2d(4,4) linear lr=0.05",
        |d| seq_train_step(d, build_channel_shuffle, &[2, 2, 6, 6], 144)
    ),
    nn_case!(
        "nn_channel_shuffle_train_step_s2",
        Route::H,
        "S2: x=[3,2,7,5] conv2d(2,4) channel_shuffle(2) conv2d(4,4) linear lr=0.05",
        |d| seq_train_step(d, build_channel_shuffle, &[3, 2, 7, 5], 140)
    ),
    nn_case!(
        "nn_adaptive_pool3d_train_step_s1",
        Route::H,
        "S1: x=[2,2,4,4,4] conv3d(2,3) adaptive_avg_pool3d([3,3,3]) adaptive_max_pool3d([2,2,2]) linear",
        |d| seq_train_step(d, build_adaptive_pool3d, &[2, 2, 4, 4, 4], 24)
    ),
    nn_case!(
        "nn_adaptive_pool3d_train_step_s2",
        Route::H,
        "S2: x=[3,2,5,6,4] conv3d(2,3) adaptive_avg_pool3d([3,3,3]) adaptive_max_pool3d([2,2,2]) linear",
        |d| seq_train_step(d, build_adaptive_pool3d, &[3, 2, 5, 6, 4], 24)
    ),
    nn_case!(
        "nn_lp_pool_train_step_s1",
        Route::H,
        "S1: x=[2,2,6,6] conv2d(2,3) lp_pool2d(p=2,k=[2,2]) linear lr=0.05",
        |d| seq_train_step(d, build_lp_pool, &[2, 2, 6, 6], 27)
    ),
    nn_case!(
        "nn_lp_pool_train_step_s2",
        Route::H,
        "S2: x=[3,2,7,5] conv2d(2,3) lp_pool2d(p=2,k=[2,2]) linear lr=0.05",
        |d| seq_train_step(d, build_lp_pool, &[3, 2, 7, 5], 18)
    ),
    nn_case!(
        "nn_fractional_max_pool_train_step_s1",
        Route::H,
        "S1: x=[2,2,6,6] conv2d(2,3) fractional_max_pool2d(k=[2,2],out=[3,3]) manual_seed=0xF00D linear",
        |d| seq_train_step(d, build_fractional, &[2, 2, 6, 6], 27)
    ),
    nn_case!(
        "nn_fractional_max_pool_train_step_s2",
        Route::H,
        "S2: x=[3,2,7,5] conv2d(2,3) fractional_max_pool2d(k=[2,2],out=[4,3]) manual_seed=0xF00D linear",
        |d| seq_train_step(d, build_fractional_s2, &[3, 2, 7, 5], 36)
    ),
    nn_case!(
        "nn_bilinear_train_step_s1",
        Route::H,
        "S1: x1=[4,3] x2=[4,4] w=[5,3,4] b=[5] Var::bilinear MSE lr=0.05 seed=0x00DDBA11",
        |d| var_train_step(d, VarKind::Bilinear, [4, 3, 4, 5])
    ),
    nn_case!(
        "nn_bilinear_train_step_s2",
        Route::H,
        "S2: x1=[5,7] x2=[5,6] w=[3,7,6] b=[3] Var::bilinear MSE lr=0.05 seed=0x00DDBA11",
        |d| var_train_step(d, VarKind::Bilinear, [5, 7, 6, 3])
    ),
    nn_case!(
        "nn_similarity_distance_train_step_s1",
        Route::H,
        "S1: x=[4,6] W=[6,5] x2=[4,5] cosine_similarity+pairwise_distance MSE 和 lr=0.05",
        |d| var_train_step(d, VarKind::Similarity, [4, 6, 5, 0])
    ),
    nn_case!(
        "nn_similarity_distance_train_step_s2",
        Route::H,
        "S2: x=[5,13] W=[13,7] x2=[5,7] cosine_similarity+pairwise_distance MSE 和 lr=0.05",
        |d| var_train_step(d, VarKind::Similarity, [5, 13, 7, 0])
    ),
    // optim 族（#4026）。追跡表 22 行（#3837・#3838）。腕 A は param groups 付き常駐更新、
    // 腕 B はホスト参照（`ParamGroupStep::step_with_groups` + `apply_parameters`）。
    optim_case!(
        "optim_param_groups_resident_sgd_s1",
        "S1: batch=4 d_in=8 hidden=16 d_out=4 steps=20 sgd lr=0.02 momentum=0.9 nesterov groups=[0]/[1,2]",
        |d| optim_resident(d, OptKind::Sgd, [4, 8, 16, 4]),
        || optim_host(OptKind::Sgd, [4, 8, 16, 4])
    ),
    optim_case!(
        "optim_param_groups_resident_sgd_s2",
        "S2: batch=5 d_in=13 hidden=17 d_out=7 steps=20 sgd lr=0.02 momentum=0.9 nesterov groups=[0]/[1,2]",
        |d| optim_resident(d, OptKind::Sgd, [5, 13, 17, 7]),
        || optim_host(OptKind::Sgd, [5, 13, 17, 7])
    ),
    optim_case!(
        "optim_param_groups_resident_adam_s1",
        "S1: batch=4 d_in=8 hidden=16 d_out=4 steps=20 adam default groups=[0]/[1,2]",
        |d| optim_resident(d, OptKind::Adam, [4, 8, 16, 4]),
        || optim_host(OptKind::Adam, [4, 8, 16, 4])
    ),
    optim_case!(
        "optim_param_groups_resident_adam_s2",
        "S2: batch=5 d_in=13 hidden=17 d_out=7 steps=20 adam default groups=[0]/[1,2]",
        |d| optim_resident(d, OptKind::Adam, [5, 13, 17, 7]),
        || optim_host(OptKind::Adam, [5, 13, 17, 7])
    ),
    optim_case!(
        "optim_param_groups_resident_adamw_s1",
        "S1: batch=4 d_in=8 hidden=16 d_out=4 steps=20 adamw default groups=[0]/[1,2]",
        |d| optim_resident(d, OptKind::AdamW, [4, 8, 16, 4]),
        || optim_host(OptKind::AdamW, [4, 8, 16, 4])
    ),
    optim_case!(
        "optim_param_groups_resident_adamw_s2",
        "S2: batch=5 d_in=13 hidden=17 d_out=7 steps=20 adamw default groups=[0]/[1,2]",
        |d| optim_resident(d, OptKind::AdamW, [5, 13, 17, 7]),
        || optim_host(OptKind::AdamW, [5, 13, 17, 7])
    ),
];

fn tensor(data: Vec<f32>, shape: &[usize]) -> Result<Tensor<f32>, String> {
    Tensor::new(data, shape).map_err(|e| format!("テンソル構築に失敗: {e:?}"))
}

fn dense(t: &Tensor<f32>) -> Result<Vec<f32>, String> {
    t.contiguous()
        .as_slice()
        .map(<[f32]>::to_vec)
        .ok_or_else(|| "contiguous() 後に as_slice が None".to_string())
}

fn build_model(d_in: usize, hidden: usize, d_out: usize) -> Result<Sequential, String> {
    Sequential::new()
        .add_linear(d_in, hidden, SEED_L1)
        .map_err(|e| format!("{e:?}"))?
        .add_relu()
        .add_linear(hidden, d_out, SEED_L2)
        .map_err(|e| format!("{e:?}"))
}

fn regression_data(
    batch: usize,
    d_in: usize,
    d_out: usize,
) -> Result<(Tensor<f32>, Tensor<f32>), String> {
    let mut rng = Xorshift64Star::new(SEED_DATA);
    let x = rng.fill_vec(batch * d_in);
    let y = rng.fill_vec(batch * d_out);
    Ok((tensor(x, &[batch, d_in])?, tensor(y, &[batch, d_out])?))
}

/// forward 出力・MSE loss・各パラメータ勾配を返す（腕 A は `device`、腕 B は CPU）。
fn mlp_forward_backward(
    device: Device,
    batch: usize,
    d_in: usize,
    hidden: usize,
    d_out: usize,
) -> Result<Vec<Quantity>, String> {
    let model = build_model(d_in, hidden, d_out)?;
    let (x_data, y_data) = regression_data(batch, d_in, d_out)?;
    let tape = fandhe_ai::tape_for(device).map_err(|e| format!("tape_for 失敗: {e:?}"))?;
    let bound = model.bind(&tape);
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = bound.forward(&tape, &x).map_err(|e| format!("{e:?}"))?;
    let loss = MseLoss::new(Reduction::Mean)
        .forward(&pred, &y)
        .map_err(|e| format!("{e:?}"))?;
    let grads = tape.backward(&loss).map_err(|e| format!("{e:?}"))?;
    let grad_refs = bound
        .trainable_grads(&grads)
        .map_err(|e| format!("{e:?}"))?;

    let mut out = vec![
        Quantity {
            name: "forward",
            values: dense(&pred.to_tensor())?,
            require_nonzero: true,
        },
        Quantity {
            name: "loss",
            values: dense(&loss.to_tensor())?,
            require_nonzero: true,
        },
    ];
    const GRAD_NAMES: [&str; 4] = ["grad_w1", "grad_b1", "grad_w2", "grad_b2"];
    if grad_refs.len() != GRAD_NAMES.len() {
        return Err(format!("勾配の本数が想定外: {}", grad_refs.len()));
    }
    for (name, g) in GRAD_NAMES.iter().zip(grad_refs) {
        out.push(Quantity {
            name,
            values: dense(g)?,
            require_nonzero: true,
        });
    }
    Ok(out)
}

/// ONNX 学習の測定（#3842）。`from_sequential` → `bind_autograd` → `forward` → MSE →
/// `backward`。出力・損失・各 param の勾配（`params()` の名前順）を返す。
fn onnx_forward_backward(
    device: Device,
    batch: usize,
    d_in: usize,
    hidden: usize,
    d_out: usize,
) -> Result<Vec<Quantity>, String> {
    let seq = build_model(d_in, hidden, d_out)?;
    let model = OnnxModel::from_sequential(&seq).map_err(|e| format!("{e:?}"))?;
    let (x_data, y_data) = regression_data(batch, d_in, d_out)?;
    let tape = fandhe_ai::tape_for(device).map_err(|e| format!("tape_for 失敗: {e:?}"))?;
    let g = model
        .bind_autograd(&tape, &OnnxAutogradOptions::default())
        .map_err(|e| format!("{e:?}"))?;
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = g.forward(&x).map_err(|e| format!("{e:?}"))?;
    let loss = MseLoss::new(Reduction::Mean)
        .forward(&pred, &y)
        .map_err(|e| format!("{e:?}"))?;
    let grads = tape.backward(&loss).map_err(|e| format!("{e:?}"))?;

    let mut out = vec![
        Quantity {
            name: "forward",
            values: dense(&pred.to_tensor())?,
            require_nonzero: true,
        },
        Quantity {
            name: "loss",
            values: dense(&loss.to_tensor())?,
            require_nonzero: true,
        },
    ];
    // `params()` は名前順: 0.bias, 0.weight, 2.bias, 2.weight。
    const GRAD_NAMES: [&str; 4] = [
        "grad_0_bias",
        "grad_0_weight",
        "grad_2_bias",
        "grad_2_weight",
    ];
    let params = g.params();
    if params.len() != GRAD_NAMES.len() {
        return Err(format!("param の本数が想定外: {}", params.len()));
    }
    for (name, (pname, v)) in GRAD_NAMES.iter().zip(&params) {
        let grad = grads
            .get(v)
            .map_err(|e| format!("{e:?}"))?
            .ok_or_else(|| format!("{pname} の勾配がない"))?;
        out.push(Quantity {
            name,
            values: dense(grad)?,
            require_nonzero: true,
        });
    }
    Ok(out)
}

fn final_params(params: &[Tensor<f32>]) -> Result<Vec<Quantity>, String> {
    const NAMES: [&str; 4] = ["param_w1", "param_b1", "param_w2", "param_b2"];
    if params.len() != NAMES.len() {
        return Err(format!("パラメータ本数が想定外: {}", params.len()));
    }
    NAMES
        .iter()
        .zip(params)
        .map(|(name, p)| {
            Ok(Quantity {
                name,
                values: dense(p)?,
                require_nonzero: true,
            })
        })
        .collect()
}

/// デバイス常駐経路で `SGD_STEPS` 回学習した最終パラメータ。
fn sgd_resident(device: Device) -> Result<Vec<Quantity>, String> {
    let model = build_model(8, 16, 4)?;
    let (x_data, y_data) = regression_data(4, 8, 4)?;
    let init_tape = fandhe_ai::tape_for(device).map_err(|e| format!("tape_for 失敗: {e:?}"))?;
    let mut store = model
        .init_device_param_store(&init_tape)
        .map_err(|e| format!("{e:?}"))?;
    drop(init_tape);
    let config = FacadeSgdConfig::new(SGD_LR).with_momentum(0.9);
    for _ in 0..SGD_STEPS {
        let tape = fandhe_ai::tape_for(device).map_err(|e| format!("{e:?}"))?;
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model
            .forward_resident(&tape, &x, &mut store)
            .map_err(|e| format!("{e:?}"))?;
        let loss = MseLoss::new(Reduction::Mean)
            .forward(&pred, &y)
            .map_err(|e| format!("{e:?}"))?;
        let grads = tape
            .backward_device_param_store(&loss, &store)
            .map_err(|e| format!("{e:?}"))?;
        tape.step_device_param_store(&mut store, &grads, &config)
            .map_err(|e| format!("{e:?}"))?;
    }
    let final_tape = fandhe_ai::tape_for(device).map_err(|e| format!("{e:?}"))?;
    let params = final_tape
        .sync_device_param_store_to_host(&store)
        .map_err(|e| format!("{e:?}"))?;
    final_params(&params)
}

/// ホスト参照（`Sgd::step` + `apply_parameters`）。
fn sgd_host() -> Result<Vec<Quantity>, String> {
    let mut model = build_model(8, 16, 4)?;
    let (x_data, y_data) = regression_data(4, 8, 4)?;
    let mut sgd =
        Sgd::new(SgdConfig::new(SGD_LR).with_momentum(0.9)).map_err(|e| format!("{e:?}"))?;
    for _ in 0..SGD_STEPS {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);
            let pred = bound.forward(&tape, &x).map_err(|e| format!("{e:?}"))?;
            let loss = MseLoss::new(Reduction::Mean)
                .forward(&pred, &y)
                .map_err(|e| format!("{e:?}"))?;
            let grads = tape.backward(&loss).map_err(|e| format!("{e:?}"))?;
            let grad_refs = bound
                .trainable_grads(&grads)
                .map_err(|e| format!("{e:?}"))?;
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs)
                .map_err(|e| format!("{e:?}"))?
        };
        model
            .apply_parameters(updated)
            .map_err(|e| format!("{e:?}"))?;
    }
    let params: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();
    final_params(&params)
}

// ---- nn 族（#4026）: 学習ステップ ----

const NN_D_OUT: usize = 3;
const FRACTIONAL_SEED: u64 = 0xF00D;

fn ae<E: std::fmt::Debug>(e: E) -> String {
    format!("{e:?}")
}

fn conv2d_3(c_in: usize, c_out: usize, seed: u64, m: Sequential) -> Result<Sequential, String> {
    m.add_conv2d(c_in, c_out, [3, 3], [1, 1], [1, 1], [1, 1], 1, seed)
        .map_err(ae)
}

fn build_upsample(feat: usize) -> Result<Sequential, String> {
    conv2d_3(2, 3, SEED_L1, Sequential::new())?
        .add_upsample_scale_factor(
            vec![2.0, 1.5],
            fandhe_ai::InterpolateMode::Bilinear {
                align_corners: false,
            },
        )
        .map_err(ae)?
        .add_flatten(1, 3)
        .add_linear(feat, NN_D_OUT, SEED_L2)
        .map_err(ae)
}

fn build_shrink(feat: usize) -> Result<Sequential, String> {
    conv2d_3(2, 3, SEED_L1, Sequential::new())?
        .add_hardshrink(0.05)
        .map_err(ae)?
        .add_softshrink(0.05)
        .map_err(ae)?
        .add_flatten(1, 3)
        .add_linear(feat, NN_D_OUT, SEED_L2)
        .map_err(ae)
}

fn build_bn3d(feat: usize) -> Result<Sequential, String> {
    Sequential::new()
        .add_conv3d(2, 3, [3, 3, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1], 1, SEED_L1)
        .map_err(ae)?
        .add_batch_norm3d(3, 1e-5, 0.1)
        .map_err(ae)?
        .add_relu()
        .add_flatten(1, 4)
        .add_linear(feat, NN_D_OUT, SEED_L2)
        .map_err(ae)
}

fn build_channel_shuffle(feat: usize) -> Result<Sequential, String> {
    let m = conv2d_3(2, 4, SEED_L1, Sequential::new())?
        .add_channel_shuffle(2)
        .map_err(ae)?;
    conv2d_3(4, 4, SEED_L2, m)?
        .add_flatten(1, 3)
        .add_linear(feat, NN_D_OUT, SEED_L2)
        .map_err(ae)
}

fn build_adaptive_pool3d(feat: usize) -> Result<Sequential, String> {
    Sequential::new()
        .add_conv3d(2, 3, [3, 3, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1], 1, SEED_L1)
        .map_err(ae)?
        .add_adaptive_avg_pool3d([3, 3, 3])
        .map_err(ae)?
        .add_adaptive_max_pool3d([2, 2, 2])
        .map_err(ae)?
        .add_flatten(1, 4)
        .add_linear(feat, NN_D_OUT, SEED_L2)
        .map_err(ae)
}

fn build_lp_pool(feat: usize) -> Result<Sequential, String> {
    conv2d_3(2, 3, SEED_L1, Sequential::new())?
        .add_lp_pool2d(2.0, [2, 2], None)
        .map_err(ae)?
        .add_flatten(1, 3)
        .add_linear(feat, NN_D_OUT, SEED_L2)
        .map_err(ae)
}

fn build_fractional(feat: usize) -> Result<Sequential, String> {
    conv2d_3(2, 3, SEED_L1, Sequential::new())?
        .add_fractional_max_pool2d([2, 2], [3, 3])
        .map_err(ae)?
        .add_flatten(1, 3)
        .add_linear(feat, NN_D_OUT, SEED_L2)
        .map_err(ae)
}

fn build_fractional_s2(feat: usize) -> Result<Sequential, String> {
    conv2d_3(2, 3, SEED_L1, Sequential::new())?
        .add_fractional_max_pool2d([2, 2], [4, 3])
        .map_err(ae)?
        .add_flatten(1, 3)
        .add_linear(feat, NN_D_OUT, SEED_L2)
        .map_err(ae)
}

/// 勾配量の名前（`&'static str` 固定。本数が超えたら Err で fail-closed）。
const SLOT_GRAD_NAMES: [&str; 12] = [
    "grad_p0", "grad_p1", "grad_p2", "grad_p3", "grad_p4", "grad_p5", "grad_p6", "grad_p7",
    "grad_p8", "grad_p9", "grad_p10", "grad_p11",
];

fn nz(name: &'static str, values: Vec<f32>) -> Quantity {
    Quantity {
        name,
        values,
        require_nonzero: true,
    }
}

/// `Sequential` の学習ステップ: forward → MSE → backward → ホスト SGD 1 step → 更新後の
/// forward・loss。層単体ではなく学習ステップ全体を `device` と CPU で比べる。
/// 返す量は forward・loss・各勾配（スロット順）・更新後の loss（`loss_after_step`）。
fn seq_train_step(
    device: Device,
    build: fn(usize) -> Result<Sequential, String>,
    x_shape: &[usize],
    feat: usize,
) -> Result<Vec<Quantity>, String> {
    let mut model = build(feat)?;
    let batch = x_shape[0];
    let mut rng = Xorshift64Star::new(SEED_DATA);
    let x_data = tensor(rng.fill_vec(x_shape.iter().product()), x_shape)?;
    let y_data = tensor(rng.fill_vec(batch * NN_D_OUT), &[batch, NN_D_OUT])?;
    // FractionalMaxPool はプロセス共有 RNG を使う。各 forward の直前で種を固定する。
    fandhe_ai::manual_seed(FRACTIONAL_SEED);
    let (mut out, updated) = {
        let tape = fandhe_ai::tape_for(device).map_err(ae)?;
        let bound = model.bind(&tape);
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = bound.forward(&tape, &x).map_err(ae)?;
        let loss = MseLoss::new(Reduction::Mean)
            .forward(&pred, &y)
            .map_err(ae)?;
        let grads = tape.backward(&loss).map_err(ae)?;
        let grad_refs = bound.trainable_grads(&grads).map_err(ae)?;
        if grad_refs.len() > SLOT_GRAD_NAMES.len() {
            return Err(format!("勾配の本数が想定外: {}", grad_refs.len()));
        }
        let mut out = vec![
            nz("forward", dense(&pred.to_tensor())?),
            nz("loss", dense(&loss.to_tensor())?),
        ];
        for (name, g) in SLOT_GRAD_NAMES.iter().zip(&grad_refs) {
            out.push(nz(name, dense(g)?));
        }
        let mut sgd = Sgd::new(SgdConfig::new(NN_LR)).map_err(ae)?;
        let updated = sgd
            .step(&model.trainable_parameters(), &grad_refs)
            .map_err(ae)?;
        (out, updated)
    };
    model.apply_parameters(updated).map_err(ae)?;
    fandhe_ai::manual_seed(FRACTIONAL_SEED);
    let tape = fandhe_ai::tape_for(device).map_err(ae)?;
    let bound = model.bind(&tape);
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = bound.forward(&tape, &x).map_err(ae)?;
    let loss = MseLoss::new(Reduction::Mean)
        .forward(&pred, &y)
        .map_err(ae)?;
    out.push(nz("loss_after_step", dense(&loss.to_tensor())?));
    Ok(out)
}

/// `Var` グラフ行の種別。`Sequential` を介さず `Var` の演算を直接つなぐ行（追跡表 10・12・13）。
#[derive(Clone, Copy)]
enum VarKind {
    Bilinear,
    Similarity,
}

fn bilinear_loss<'t>(p: &[Var<'t>], d: &[Var<'t>]) -> Result<Var<'t>, String> {
    let out = d[0].bilinear(&d[1], &p[0], Some(&p[1])).map_err(ae)?;
    MseLoss::new(Reduction::Mean)
        .forward(&out, &d[2])
        .map_err(ae)
}

fn similarity_loss<'t>(p: &[Var<'t>], d: &[Var<'t>]) -> Result<Var<'t>, String> {
    let h = d[0].matmul(&p[0]).map_err(ae)?;
    let cos = h.cosine_similarity(&d[1], 1, 1e-8).map_err(ae)?;
    let dist = h.pairwise_distance(&d[1], 2.0, 1e-6, false).map_err(ae)?;
    let mse = MseLoss::new(Reduction::Mean);
    let l1 = mse.forward(&cos, &d[2]).map_err(ae)?;
    let l2 = mse.forward(&dist, &d[3]).map_err(ae)?;
    l1.add(&l2).map_err(ae)
}

/// `dims = [batch, a, b, c]`。Bilinear は x1=[batch,a] x2=[batch,b] w=[c,a,b] b=[c]、
/// Similarity は x=[batch,a] W=[a,b] x2=[batch,b]（c は未使用）。損失 → 勾配 → ホスト SGD
/// 1 step → 更新後の損失を返す。
fn var_train_step(
    device: Device,
    kind: VarKind,
    dims: [usize; 4],
) -> Result<Vec<Quantity>, String> {
    let [batch, a, b, c] = dims;
    let mut rng = Xorshift64Star::new(SEED_VAR);
    let (mut params, data): (Vec<Tensor<f32>>, Vec<Tensor<f32>>) = match kind {
        VarKind::Bilinear => (
            vec![
                tensor(rng.fill_vec(c * a * b), &[c, a, b])?,
                tensor(rng.fill_vec(c), &[c])?,
            ],
            vec![
                tensor(rng.fill_vec(batch * a), &[batch, a])?,
                tensor(rng.fill_vec(batch * b), &[batch, b])?,
                tensor(rng.fill_vec(batch * c), &[batch, c])?,
            ],
        ),
        VarKind::Similarity => (
            vec![tensor(rng.fill_vec(a * b), &[a, b])?],
            vec![
                tensor(rng.fill_vec(batch * a), &[batch, a])?,
                tensor(rng.fill_vec(batch * b), &[batch, b])?,
                tensor(rng.fill_vec(batch), &[batch])?,
                tensor(rng.fill_vec(batch), &[batch])?,
            ],
        ),
    };
    let loss_fn: for<'t> fn(&[Var<'t>], &[Var<'t>]) -> Result<Var<'t>, String> = match kind {
        VarKind::Bilinear => bilinear_loss,
        VarKind::Similarity => similarity_loss,
    };
    let mut out = Vec::new();
    for step in 0..2 {
        let tape = fandhe_ai::tape_for(device).map_err(ae)?;
        let pv: Vec<Var<'_>> = params.iter().map(|t| tape.var(t)).collect();
        let dv: Vec<Var<'_>> = data.iter().map(|t| tape.var(t)).collect();
        let loss = loss_fn(&pv, &dv)?;
        if step == 1 {
            out.push(nz("loss_after_step", dense(&loss.to_tensor())?));
            break;
        }
        out.push(nz("loss", dense(&loss.to_tensor())?));
        let grads = tape.backward(&loss).map_err(ae)?;
        let mut next = Vec::new();
        for (i, (p, v)) in params.iter().zip(&pv).enumerate() {
            let g = grads
                .get(v)
                .map_err(ae)?
                .ok_or_else(|| format!("param {i} の勾配がない"))?;
            let gd = dense(g)?;
            let upd: Vec<f32> = dense(p)?
                .iter()
                .zip(&gd)
                .map(|(w, g)| w - NN_LR * g)
                .collect();
            next.push(tensor(upd, p.shape())?);
            out.push(nz(SLOT_GRAD_NAMES[i], gd));
        }
        params = next;
    }
    Ok(out)
}

// ---- optim 族（#4026）: param groups 付き常駐更新 ----

#[derive(Clone, Copy)]
enum OptKind {
    Sgd,
    Adam,
    AdamW,
}

const PARAM_NAMES: [&str; 4] = ["param_w1", "param_b1", "param_w2", "param_b2"];

fn optim_groups() -> Vec<ParamGroup> {
    // スロット 3（b2）はどのグループにも入れず、optimizer 既定値で更新させる。
    vec![
        ParamGroup::new(vec![0], 5e-3, 0.05),
        ParamGroup::new(vec![1, 2], 2e-2, 0.0),
    ]
}

fn optim_sgd_config() -> SgdConfig {
    SgdConfig::new(SGD_LR)
        .with_momentum(0.9)
        .with_nesterov(true)
}

/// `dims = [batch, d_in, hidden, d_out]`。
fn optim_data(dims: [usize; 4]) -> Result<(Tensor<f32>, Tensor<f32>), String> {
    regression_data(dims[0], dims[1], dims[3])
}

fn optim_resident(device: Device, opt: OptKind, dims: [usize; 4]) -> Result<Vec<Quantity>, String> {
    let model = build_model(dims[1], dims[2], dims[3])?;
    let (x_data, y_data) = optim_data(dims)?;
    let store_tape = fandhe_ai::tape_for(device).map_err(ae)?;
    let mut store = model
        .init_device_param_store_with_param_groups(&store_tape, &optim_groups())
        .map_err(ae)?;
    drop(store_tape);
    for _ in 0..OPTIM_STEPS {
        let tape = fandhe_ai::tape_for(device).map_err(ae)?;
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store).map_err(ae)?;
        let loss = MseLoss::new(Reduction::Mean)
            .forward(&pred, &y)
            .map_err(ae)?;
        let grads = tape
            .backward_device_param_store(&loss, &store)
            .map_err(ae)?;
        match opt {
            OptKind::Sgd => tape.step_device_param_store(&mut store, &grads, &optim_sgd_config()),
            OptKind::Adam => tape.step_device_param_store_adam(
                &mut store,
                &grads,
                &fandhe_ai::optim::AdamConfig::default(),
            ),
            OptKind::AdamW => tape.step_device_param_store_adamw(
                &mut store,
                &grads,
                &fandhe_ai::optim::AdamWConfig::default(),
            ),
        }
        .map_err(ae)?;
    }
    let params = fandhe_ai::tape_for(device)
        .map_err(ae)?
        .sync_device_param_store_to_host(&store)
        .map_err(ae)?;
    named_params(&params)
}

fn named_params(params: &[Tensor<f32>]) -> Result<Vec<Quantity>, String> {
    if params.len() != PARAM_NAMES.len() {
        return Err(format!("パラメータ本数が想定外: {}", params.len()));
    }
    PARAM_NAMES
        .iter()
        .zip(params)
        .map(|(n, p)| Ok(nz(n, dense(p)?)))
        .collect()
}

/// ホスト参照: 同じ groups を `ParamGroupStep::step_with_groups` で適用する。
fn optim_host(opt: OptKind, dims: [usize; 4]) -> Result<Vec<Quantity>, String> {
    let mut model = build_model(dims[1], dims[2], dims[3])?;
    let (x_data, y_data) = optim_data(dims)?;
    let groups = optim_groups();
    let mut sgd = Sgd::new(optim_sgd_config()).map_err(ae)?;
    let mut adam = Adam::new(AdamConfig::default()).map_err(ae)?;
    let mut adamw = AdamW::new(AdamWConfig::default()).map_err(ae)?;
    for _ in 0..OPTIM_STEPS {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);
            let pred = bound.forward(&tape, &x).map_err(ae)?;
            let loss = MseLoss::new(Reduction::Mean)
                .forward(&pred, &y)
                .map_err(ae)?;
            let grads = tape.backward(&loss).map_err(ae)?;
            let grad_refs = bound.trainable_grads(&grads).map_err(ae)?;
            let param_refs = model.trainable_parameters();
            match opt {
                OptKind::Sgd => sgd.step_with_groups(&param_refs, &grad_refs, &groups),
                OptKind::Adam => adam.step_with_groups(&param_refs, &grad_refs, &groups),
                OptKind::AdamW => adamw.step_with_groups(&param_refs, &grad_refs, &groups),
            }
            .map_err(ae)?
        };
        model.apply_parameters(updated).map_err(ae)?;
    }
    let params: Vec<Tensor<f32>> = model.trainable_parameters().into_iter().cloned().collect();
    named_params(&params)
}

// ---- 実機テスト入口（族ごとに 1 本。--exact で 1 本ずつ別プロセス実行）----

macro_rules! cuda_family_test {
    ($name:ident, $family:expr) => {
        #[test]
        #[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
        fn $name() {
            harness::run_family(CASES, $family, Device::Cuda(0));
        }
    };
}

macro_rules! metal_family_test {
    ($name:ident, $family:expr) => {
        #[cfg(target_os = "macos")]
        #[test]
        #[ignore = "Metal 実機（Apple Silicon）が必要"]
        fn $name() {
            harness::run_family(CASES, $family, Device::Metal);
        }
    };
}

cuda_family_test!(train_parity_scaffold_cuda_matches_cpu, "scaffold");
cuda_family_test!(train_parity_nn_cuda_matches_cpu, "nn");
cuda_family_test!(train_parity_optim_cuda_matches_cpu, "optim");
cuda_family_test!(train_parity_onnx_cuda_matches_cpu, "onnx");
metal_family_test!(train_parity_scaffold_metal_matches_cpu, "scaffold");
metal_family_test!(train_parity_nn_metal_matches_cpu, "nn");
metal_family_test!(train_parity_optim_metal_matches_cpu, "optim");
metal_family_test!(train_parity_onnx_metal_matches_cpu, "onnx");

// ---- CI 自己検査（GPU 不使用）----

#[test]
fn registry_names_are_unique() {
    let mut names: Vec<_> = CASES.iter().map(|c| c.name).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(before, names.len(), "登録名が重複している");
}

#[test]
fn every_case_family_is_declared() {
    for c in CASES {
        assert!(
            FAMILIES.contains(&c.family),
            "未宣言の族: {} (case={})",
            c.family,
            c.name
        );
    }
}

#[test]
fn scaffold_family_passes_cpu_against_cpu() {
    harness::run_family(CASES, "scaffold", Device::Cpu);
}

#[test]
fn onnx_family_passes_cpu_against_cpu() {
    harness::run_family(CASES, "onnx", Device::Cpu);
}

#[test]
fn nn_family_passes_cpu_against_cpu() {
    harness::run_family(CASES, "nn", Device::Cpu);
}

#[test]
fn optim_family_passes_cpu_against_cpu() {
    harness::run_family(CASES, "optim", Device::Cpu);
}

/// 登録 0 件の族があると実機テストが構造上 pass できない（#4026 の不具合）ため、
/// 宣言した全族に 1 行以上あることを CI で縛る。
#[test]
fn every_declared_family_has_cases() {
    for f in FAMILIES {
        assert!(CASES.iter().any(|c| c.family == *f), "登録 0 件の族: {f}");
    }
}

#[test]
#[should_panic(expected = "undetermined")]
fn empty_family_is_undetermined() {
    // 本番 CASES から切り離した空の登録表で検証する（#3842 で onnx のケースを足しても影響しない）。
    harness::run_family(&[], "onnx", Device::Cpu);
}

fn q(name: &'static str, values: Vec<f32>, require_nonzero: bool) -> Quantity {
    Quantity {
        name,
        values,
        require_nonzero,
    }
}

fn dummy_run(_: Device) -> Result<Vec<Quantity>, String> {
    Err("故意の失敗".to_string())
}

fn synthetic_case(judge: Judge) -> Case {
    Case {
        name: "synthetic",
        family: "scaffold",
        route: Route::K,
        judge,
        shape: "-",
        run: dummy_run,
        reference: None,
    }
}

#[test]
fn harness_fails_closed_on_mismatch() {
    let req2 = synthetic_case(Judge::Req2);
    let bit = synthetic_case(Judge::Bit);

    // 名前の不一致・長さの不一致・全ゼロ
    assert!(
        harness::compare_quantities(
            &req2,
            "t",
            &[q("a", vec![1.0], false)],
            &[q("b", vec![1.0], false)]
        )
        .is_err()
    );
    assert!(
        harness::compare_quantities(
            &req2,
            "t",
            &[q("a", vec![1.0, 2.0], false)],
            &[q("a", vec![1.0], false)]
        )
        .is_err()
    );
    assert!(
        harness::compare_quantities(
            &req2,
            "t",
            &[q("a", vec![0.0, 0.0], true)],
            &[q("a", vec![0.0, 0.0], true)]
        )
        .is_err()
    );

    // 空の測定量列・空 values（total=0）
    assert!(harness::compare_quantities(&req2, "t", &[], &[]).is_err());
    assert!(
        harness::compare_quantities(
            &req2,
            "t",
            &[q("a", vec![], false)],
            &[q("a", vec![], false)]
        )
        .is_err()
    );

    // 大きくずらした値は Req2 で fail
    let recs = harness::compare_quantities(
        &req2,
        "t",
        &[q("a", vec![1.0, 2.0], true)],
        &[q("a", vec![5.0, 9.0], true)],
    )
    .unwrap();
    assert_eq!(recs[0].verdict, harness::Verdict::Fail);
    assert!(recs[0].report.fail_count > 0);

    // Bit: +0.0 と -0.0 を区別し、NaN 同士は一致
    let recs = harness::compare_quantities(
        &bit,
        "t",
        &[q("a", vec![0.0], false)],
        &[q("a", vec![-0.0], false)],
    )
    .unwrap();
    assert_eq!(recs[0].verdict, harness::Verdict::Fail);
    let recs = harness::compare_quantities(
        &bit,
        "t",
        &[q("a", vec![f32::NAN], false)],
        &[q("a", vec![f32::NAN], false)],
    )
    .unwrap();
    assert_eq!(recs[0].verdict, harness::Verdict::Pass);

    // run の Err は undetermined（Err）になる
    assert!(harness::run_case(&req2, Device::Cpu).is_err());
}
