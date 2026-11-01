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
//! - `onnx`: ONNX の学習（#3864・#3865。facade 公開形 #3842 が入ってからつなぐ。
//!   それまで 0 件で、実行すると undetermined で止まる）
//!
//! `nn`・`optim`・`onnx` の行は Phase 6〜8 の実装 issue が足す。実機テストの名前は
//! 本 PR で固定し、Phase 10 の issue と RULE.txt が参照する。ROCm は対象外
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
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::Tensor;
use harness::{Case, Judge, Quantity, Route};

const FAMILIES: &[&str] = &["scaffold", "nn", "optim", "onnx"];

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;
const SGD_STEPS: usize = 20;
const SGD_LR: f32 = 0.02;

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
#[should_panic(expected = "undetermined")]
fn empty_family_is_undetermined() {
    harness::run_family(CASES, "onnx", Device::Cpu);
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
