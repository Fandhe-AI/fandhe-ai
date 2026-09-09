//! イシュー #1479「resident `GradStaging` の重み勾配をホストへ読み出す
//! 公開 API を追加する」の受入基準（AC-R3・実 CPU カーネル版）を検証する
//! 統合テスト。
//!
//! `crates/autodiff/src/optim/device_store.rs::tests` のクレート内単体
//! テスト（`resident_grads_to_host_returns_weight_slot_and_none_for_host_slots`
//! 等。`MockDeviceOps` 経由）は配管（判定ロジック・非破壊契約）の検証に
//! 留まる——本ファイルは `fandhe_ai::tape()`（既定 CPU バックエンド）を
//! 使い、`Tape::resident_grads_to_host`／`Tape::param_grads_to_host`
//! （facade 公開 API。`crates/facade/src/lib.rs`）が返す weight 勾配を、
//! CPU 本番 GEMM カーネル全入口の共通契約点である
//! `fandhe_ai_backend_cpu::matmul_reference_fma`（`.claude/rules/
//! coding-rust.md` の FMA 契約参照実装）と bit 完全一致で突合する。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない
//! （`fandhe_ai::tape()` 既定 CPU 経由。CUDA／Metal の契約検証は
//! `crates/facade/tests/device_param_store_backend_parity.rs` の
//! `#[ignore]` テストを参照）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::SgdConfig as FacadeSgdConfig;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{BackendError, Var};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_OUT: usize = 4;
const D_HIDDEN: usize = 16;
const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L: u64 = 0x1234_5678;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

/// 単層モデル（活性化なし。weight 勾配を単一 GEMM に帰着させ
/// `matmul_reference_fma` と直接突合できるようにする）。
fn build_single_layer() -> Sequential {
    Sequential::new().add_linear(D_IN, D_OUT, SEED_L).unwrap()
}

fn build_two_layer() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// AC-R3 の本命: `resident_grads_to_host` の weight slot（実 CPU
/// カーネル経由）が `matmul_reference_fma`（`x^T @ d_pred`。`eval::matmul`
/// と同一の逐次・k 昇順・`f32::mul_add` 契約）と bit 完全一致すること、
/// bias slot は resident 未充填のため `None` であることを検証する。
#[test]
fn resident_grads_to_host_weight_matches_matmul_reference_fma() {
    let model = build_single_layer();
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();

    let strict = tape.resident_grads_to_host(&store, &grads).unwrap();
    assert_eq!(
        strict.len(),
        2,
        "単層 Linear は weight・bias の 2 パラメータ"
    );
    assert!(
        strict[1].is_none(),
        "bias は resident 経由で充填されないため None のはず"
    );
    let weight_grad = strict[0]
        .as_ref()
        .expect("weight は resident 経由で新鮮に充填されているはず");

    // d_pred = grads.get(&pred) から取得（`Var::mse_loss` の VJP が
    // `2 * (pred - target) / numel` を書き込む。イシュー #1219 の
    // nograd スキップは活性化入力限定でありここでは対象外）。
    let d_pred = grads.get(&pred).unwrap().unwrap().contiguous();
    let x_t = x_data.transpose(0, 1).unwrap().contiguous();

    let mut expected = vec![0.0f32; D_IN * D_OUT];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        x_t.as_slice().unwrap(),
        d_pred.as_slice().unwrap(),
        &mut expected,
        D_IN,
        D_OUT,
        BATCH,
    )
    .unwrap();

    assert_eq!(
        weight_grad.contiguous().as_slice().unwrap(),
        expected.as_slice(),
        "resident_grads_to_host の weight 勾配が matmul_reference_fma 参照実装と食い違う"
    );

    // 読み出し後も `step()`（`Tape::step_device_param_store`）が通常
    // どおり成功すること（非破壊契約）。
    tape.step_device_param_store(&mut store, &grads, &FacadeSgdConfig::new(0.1))
        .unwrap();
}

/// `param_grads_to_host`（統合版）の全 slot（weight・bias）が、
/// host-only 経路（`model.bind` → `tape.backward` →
/// `bound.trainable_grads`）と bit 完全一致することを検証する（2 層
/// モデル。`device_param_store_train.rs::build_model` と同型）。
#[test]
fn param_grads_to_host_matches_host_only_path_two_layer() {
    let device_model = build_two_layer();
    let init_tape = fandhe_ai::tape();
    let mut store = device_model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let device_grads_host = {
        let tape = fandhe_ai::tape();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = device_model
            .forward_resident(&tape, &x, &mut store)
            .unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        let unified = tape.param_grads_to_host(&store, &grads).unwrap();
        // 読み出し後も step が成功することも併せて確認する。
        tape.step_device_param_store(&mut store, &grads, &FacadeSgdConfig::new(0.1))
            .unwrap();
        unified
    };

    let host_model = build_two_layer();
    let host_grads_host = {
        let tape = fandhe_ai::tape();
        let bound = host_model.bind(&tape);
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = bound.forward(&tape, &x).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let grads = tape.backward(&loss).unwrap();
        bound
            .trainable_grads(&grads)
            .unwrap()
            .into_iter()
            .cloned()
            .collect::<Vec<Tensor<f32>>>()
    };

    assert_eq!(device_grads_host.len(), host_grads_host.len());
    for (i, (d, h)) in device_grads_host
        .iter()
        .zip(host_grads_host.iter())
        .enumerate()
    {
        assert_eq!(d.shape(), h.shape(), "param {i} shape mismatch");
        assert_eq!(
            d.contiguous().as_slice().unwrap(),
            h.contiguous().as_slice().unwrap(),
            "param_grads_to_host の parameter {i} 勾配が host-only 経路と食い違う"
        );
    }
}

/// 契約: `step()`（`Tape::step_device_param_store`）で `pending` を
/// 消費した後に読み出し系 API を呼ぶと `BackendError::InvalidArgument`
/// を返す（`device_store.rs::tests::grad_readout_after_step_is_invalid_
/// argument` の facade 版）。
#[test]
fn grad_readout_after_step_is_invalid_argument_via_facade() {
    let model = build_single_layer();
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
    let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
    let grads = tape.backward_device_param_store(&loss, &store).unwrap();
    tape.step_device_param_store(&mut store, &grads, &FacadeSgdConfig::new(0.1))
        .unwrap();

    let strict_err = tape.resident_grads_to_host(&store, &grads).unwrap_err();
    assert!(matches!(strict_err, BackendError::InvalidArgument(_)));
    let unified_err = tape.param_grads_to_host(&store, &grads).unwrap_err();
    assert!(matches!(unified_err, BackendError::InvalidArgument(_)));
}

/// 契約: 古い backward 由来の `Gradients`（新しい `register_resident_
/// params` 相当の再登録を挟んだ後）を渡すと、`param_grads_to_host` は
/// `MissingGradient` で fail-closed に拒否する（`device_store.rs::
/// tests::grad_readout_rejects_stale_gradients_from_an_earlier_backward`
/// の facade 版。単層モデルで forward_resident を 2 回呼ぶことで
/// 新しい pending 登録を発生させる）。
#[test]
fn grad_readout_rejects_stale_gradients_via_facade() {
    let model = build_single_layer();
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let tape = fandhe_ai::tape();
    let x1 = tape.var(&x_data);
    let y1 = tape.var(&y_data);
    let pred1 = model.forward_resident(&tape, &x1, &mut store).unwrap();
    let loss1 = MseLoss::new(Reduction::Mean).forward(&pred1, &y1).unwrap();
    let stale_grads = tape.backward_device_param_store(&loss1, &store).unwrap();
    tape.step_device_param_store(&mut store, &stale_grads, &FacadeSgdConfig::new(0.1))
        .unwrap();

    let (x_data2, y_data2) = gen_regression_data(SEED_DATA ^ 0xABCD);
    let x2 = tape.var(&x_data2);
    let y2 = tape.var(&y_data2);
    let pred2 = model.forward_resident(&tape, &x2, &mut store).unwrap();
    let loss2 = MseLoss::new(Reduction::Mean).forward(&pred2, &y2).unwrap();
    let _fresh_grads = tape.backward_device_param_store(&loss2, &store).unwrap();

    let err = tape.param_grads_to_host(&store, &stale_grads).unwrap_err();
    assert!(matches!(err, BackendError::MissingGradient(_)));
}

/// `Var::from_raw` 相当の到達不能性を回避しつつ、`d_pred` の取得元が
/// facade 公開面（`Gradients::get`）に閉じていることを確認する軽い
/// スモーク（`Var` の import が未使用にならないための明示参照。上記
/// テストは `grads.get(&pred)` を直接使うため実質的にはここで新規に
/// 検証する事項はない——`Var` 型が facade の公開再エクスポートに含まれ
/// 続けることをコンパイル時に固定する目的）。
#[allow(dead_code)]
fn assert_var_type_is_reexported(_v: Var<'_>) {}
