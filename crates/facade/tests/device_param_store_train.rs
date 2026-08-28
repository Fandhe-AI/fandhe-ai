//! イシュー #935（feat(autodiff): デバイス上パラメータ更新の実装）の
//! 受け入れ条件を `fandhe_ai::compat::Sequential` 経由の公開 API で検証
//! する統合テスト。
//!
//! `crates/facade/tests/compat_sequential_train.rs`（既存の
//! `Sgd::step`/`apply_parameters` 経由の学習ループ）と同じモデル・
//! データ生成・収束判定様式を踏襲しつつ、本ファイルは
//! `Sequential::init_device_param_store`／`forward_resident`／
//! `Tape::step_device_param_store`／`sync_device_param_store_to_host`
//! （デバイス常駐パラメータ更新経路）を検証する。
//!
//! **数値検証**: デバイス常駐経路（`sgd_step_device`。`f32::mul_add` を
//! 用いる）とホスト参照実装（`Sgd::step`。fixture parity のため
//! `mul_add` を使わない）は丸え手順が完全には同一でないため、統一複合
//! 判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」
//! （`.claude/rules/coding-rust.md`）で突合する（`crates/backend-cpu/
//! tests/sgd_device_parity.rs` と同じ判定基準）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない
//! （`fandhe_ai::tape()` 既定 CPU 経由）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::SgdConfig as FacadeSgdConfig;
use fandhe_ai::compat::Sequential;
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap()
}

/// `Sequential::init_device_param_store` → `forward_resident` →
/// `tape.backward` → `Tape::step_device_param_store` を `steps` 回繰り返し、
/// 各 step の loss と、最終的にホストへ同期したパラメータ列を返す。
fn train_with_device_param_store(steps: usize, lr: f32) -> (Vec<f32>, Vec<Tensor<f32>>) {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(lr);
    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let tape = fandhe_ai::tape();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        log.push(scalar(&loss.to_tensor()));

        let grads = tape.backward(&loss).unwrap();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
    }

    // 明示同期点（最終ステップ後に 1 回だけホストへダウンロードする。
    // 本イシューが排除する対象は「毎ステップの再アップロード」であり、
    // ステップごとの download はこのテストの検証目的〈loss ログ〉には
    // 不要なため、ループ内では行わない）。
    let final_tape = fandhe_ai::tape();
    let synced = final_tape.sync_device_param_store_to_host(&store).unwrap();
    (log, synced)
}

/// `compat_sequential_train.rs::train_with_sgd` と同型の
/// ホスト経由学習ループ（比較対象）。
fn train_with_host_sgd(model: &mut Sequential, steps: usize, lr: f32) -> Vec<f32> {
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut sgd = Sgd::new(SgdConfig::new(lr)).unwrap();
    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);

            let pred = bound.forward(&tape, &x).unwrap();
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
            log.push(scalar(&loss.to_tensor()));

            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();
    }

    log
}

#[test]
fn device_resident_training_loop_converges() {
    const STEPS: usize = 100;
    const LR: f32 = 0.05;

    let (log, _synced) = train_with_device_param_store(STEPS, LR);

    assert_eq!(log.len(), STEPS);
    let initial = log[0];
    let final_loss = *log.last().unwrap();
    assert!(
        final_loss < 0.5 * initial,
        "loss did not converge sufficiently: initial={initial} final={final_loss}"
    );
}

/// デバイス常駐経路とホスト経由経路が、同一初期化・同一データ・同一
/// ハイパーパラメータで統一複合判定の範囲内で一致することを検証する
/// （受け入れ条件: パラメータ更新の正しさ）。
#[test]
fn device_resident_matches_host_sgd_within_composite_tolerance() {
    const STEPS: usize = 20;
    const LR: f32 = 0.05;

    let (device_log, device_params) = train_with_device_param_store(STEPS, LR);

    let mut host_model = build_model();
    let host_log = train_with_host_sgd(&mut host_model, STEPS, LR);
    let host_params = host_model.trainable_parameters();

    assert_eq!(device_log.len(), host_log.len());
    for (i, (d, h)) in device_log.iter().zip(host_log.iter()).enumerate() {
        assert_close(*d, *h, &format!("loss at step {i}"));
    }

    assert_eq!(device_params.len(), host_params.len());
    for (i, (d, h)) in device_params.iter().zip(host_params.iter()).enumerate() {
        assert_eq!(d.shape(), h.shape(), "param {i} shape mismatch");
        let d_slice = d.contiguous();
        let h_slice = h.contiguous();
        let d_data = d_slice.as_slice().unwrap();
        let h_data = h_slice.as_slice().unwrap();
        for (j, (dv, hv)) in d_data.iter().zip(h_data.iter()).enumerate() {
            assert_close(*dv, *hv, &format!("param {i} element {j}"));
        }
    }
}

/// `predict_resident` が、学習を経てデバイス側で更新された後の
/// パラメータでも `sync_device_param_store_to_host` → `apply_parameters`
/// した独立モデルの `predict` と一致することを検証する（設計文書 §3.3c
/// 「既存経路再利用で parity を構造的に保証する」）。
///
/// 未学習モデルの読み取り一致だけでは `predict_resident` が
/// デバイス側パラメータを実際に読んでいる保証にならない（初期化直後は
/// デバイス側とホスト側が同一値のため、`predict` 側の実装だけを
/// 呼んでいても偶然一致しうる）ため、学習ループを経由させてから
/// 突合する（レビュー指摘対応）。
#[test]
fn predict_resident_matches_predict_after_training() {
    const STEPS: usize = 20;
    const LR: f32 = 0.05;

    let model = build_model();
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let config = FacadeSgdConfig::new(LR);
    for _ in 0..STEPS {
        let tape = fandhe_ai::tape();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let grads = tape.backward(&loss).unwrap();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
    }

    // デバイス側で学習した結果をホスト側の独立モデルへ同期し、通常の
    // `predict` 経路と比較する（`predict_resident` はこの同期を経由せず
    // デバイス側パラメータを直接読む）。
    let sync_tape = fandhe_ai::tape();
    let synced = sync_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(sync_tape);
    let mut host_model = build_model();
    host_model.apply_parameters(synced).unwrap();

    let (x_query, _) = gen_regression_data(SEED_DATA ^ 0xABCD);
    let via_resident = model.predict_resident(&store, &x_query).unwrap();
    let via_predict = host_model.predict(&x_query).unwrap();

    assert_eq!(via_resident.shape(), via_predict.shape());
    let a = via_resident.contiguous();
    let b = via_predict.contiguous();
    for (av, bv) in a.as_slice().unwrap().iter().zip(b.as_slice().unwrap()) {
        assert_close(*av, *bv, "predict_resident と sync 後 predict の突合");
    }
}

/// `forward_resident` が forward 失敗時に `pending` をロールバックする
/// ことを検証する（codex-review PR #954 P2 是正）。`register_resident_leaves`
/// で pending 状態へ遷移した後、入力 shape 不整合で
/// `forward_from_flat_vars` を失敗させ、その直後に呼んだ 2 回目の
/// `forward_resident`（正しい入力）が `PendingForwardUnconsumed` ではなく
/// 成功することを確認する（`abandon_pending_forward` によるロールバック
/// がなければ、この 2 回目呼び出しは pending 未消費のまま拒否される）。
#[test]
fn forward_resident_rolls_back_pending_on_forward_error() {
    let model = build_model();
    let init_tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    // 1 回目: D_IN と食い違う特徴次元の入力で forward を意図的に失敗させる。
    {
        let tape = fandhe_ai::tape();
        let bad_x = tensor(vec![0.0f32; BATCH * (D_IN + 1)], &[BATCH, D_IN + 1]);
        let bad_input = tape.var(&bad_x);
        model
            .forward_resident(&tape, &bad_input, &mut store)
            .expect_err("shape 不整合の入力なので forward は失敗するはず");
    }

    // 2 回目: 正しい shape の入力で forward_resident を呼び直す。
    // ロールバックされていなければ `BackendError::PendingForwardUnconsumed`
    // 由来のエラーで拒否される。
    let (x_data, _) = gen_regression_data(SEED_DATA);
    let tape = fandhe_ai::tape();
    let x = tape.var(&x_data);
    let pred = model
        .forward_resident(&tape, &x, &mut store)
        .expect("pending がロールバックされていれば 2 回目の forward_resident は成功する");
    assert_eq!(pred.to_tensor().shape(), &[BATCH, D_OUT]);
}
