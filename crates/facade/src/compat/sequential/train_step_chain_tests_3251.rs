//! イシュー #3251・#3254: 学習ステップ全体を encode-only で積む内部チェーン
//! （`DeviceParamStore::train_step_device_chain`）と、その公開入口
//! `Sequential::train_step_resident`・`ResidentLoss::read`（P5・P6）の facade 側テスト。
//!
//! # 役割
//! #3251 では生の autodiff `Tape` を渡す `#[cfg(test)]` 補助でチェーンを検証していた。#3254 で
//! 公開入口へ移したため、本モジュールのテストは公開入口越しにチェーンを Tape 経路と突き合わせ、
//! 範囲外構成の型付きエラー・第 2 段フォールバック（`Unsupported` 注入）も固定する。
//!
//! # テスト
//! - CPU（CI で実行。イシュー #3253）: CPU バックエンドの部品が入り、チェーンが Tape 経路と
//!   loss 列・全パラメータで bit 一致すること（step 2 以降は `_into` の再利用経路）。範囲外の構成
//!   （Sigmoid／Tanh・最終層 ReLU・空バッチ・別 store の入力）は型付きエラーで止まり store を
//!   変えないこと。CI の debug ビルドの負荷のため、784 幅は 1 構成に絞り残りは中サイズで確かめる
//! - Metal（`#[ignore]`・実機）: loss と更新後パラメータが Tape 経路と bit 一致すること
//!   （momentum 0／>0・2 層／3 層）、1 step の `waitUntilCompleted` が loss を読む 1 回だけ
//!   であること。受け入れ記録は `docs/perf/logs/train-step-chain-metal-3251/`（Mac mini）
//! - CUDA（`#[ignore]`・GB10 実機。イシュー #3252）: 同じ Tape 経路との bit 一致（`_into` 版の
//!   部品で再利用バッファへ書く経路を含む。step 数 4 で 2 step 目以降が再利用経路）。同期・確保・
//!   H2D の回数は backend-cuda の診断テスト（`train_step_chain_diag_tests_3252`）が数える。
//!   受け入れ記録は `docs/perf/logs/train-step-chain-cuda-3252/`

use super::*;
use fandhe_ai_autodiff::optim::SgdConfig;

/// 決定的な擬似乱数（`-0.5` シフト）。
fn xorshift_fill(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

/// `dims = [d0, d1, ..., dN]` から `Linear(d_i, d_{i+1})` を ReLU で挟んで並べた MLP を作る。
fn build_mlp(dims: &[usize]) -> Sequential {
    let mut model = Sequential::new();
    for (i, w) in dims.windows(2).enumerate() {
        if i > 0 {
            model = model.add_relu();
        }
        model = model.add_linear(w[0], w[1], 7000 + i as u64).unwrap();
    }
    model
}

fn host_tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(xorshift_fill(seed, n), shape).unwrap()
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

#[test]
fn cpu_chain_runs_and_returns_finite_loss() {
    let model = build_mlp(&[6, 8, 3]);
    let tape = crate::tape();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let x = tape
        .upload_resident(&store, &host_tensor(1, &[4, 6]))
        .unwrap();
    let y = tape
        .upload_resident(&store, &host_tensor(2, &[4, 3]))
        .unwrap();
    let config = SgdConfig::new(0.05);
    let loss = model
        .train_step_resident(&mut store, &x, &y, &config)
        .expect("CPU の部品が入ったのでチェーンは通る（#3253）");
    let v = loss.read(&mut store).unwrap();
    assert!(v.is_finite(), "loss={v}");
}

fn params_bits(tape: &Tape, store: &DeviceParamStore) -> Vec<Vec<u32>> {
    tape.sync_device_param_store_to_host(store)
        .unwrap()
        .iter()
        .map(bits)
        .collect()
}

/// 範囲外構成の拒否: 型付きエラーで止まり、params が bit 単位で不変であること。
fn assert_rejected_and_unchanged(model: &Sequential, x_shape: &[usize], y_shape: &[usize]) {
    let tape = crate::tape();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let before = params_bits(&tape, &store);
    let x = tape
        .upload_resident(&store, &host_tensor(1, x_shape))
        .unwrap();
    let y = tape
        .upload_resident(&store, &host_tensor(2, y_shape))
        .unwrap();
    let r = model.train_step_resident(&mut store, &x, &y, &SgdConfig::new(0.05));
    assert!(
        matches!(r, Err(AutodiffError::Backend(BackendError::Unsupported(_)))),
        "範囲外は Unsupported のはず: {r:?}"
    );
    assert_eq!(before, params_bits(&tape, &store), "拒否時は params 不変");
}

#[test]
fn cpu_chain_rejects_unsupported_activation_and_keeps_store() {
    let mut model = Sequential::new();
    model = model.add_linear(6, 8, 7000).unwrap().add_sigmoid();
    model = model.add_linear(8, 3, 7001).unwrap();
    assert_rejected_and_unchanged(&model, &[4, 6], &[4, 3]);
    let mut model = Sequential::new();
    model = model.add_linear(6, 8, 7000).unwrap().add_tanh();
    model = model.add_linear(8, 3, 7001).unwrap();
    assert_rejected_and_unchanged(&model, &[4, 6], &[4, 3]);
}

#[test]
fn cpu_chain_rejects_trailing_relu_and_keeps_store() {
    let model = build_mlp(&[6, 8, 3]).add_relu();
    assert_rejected_and_unchanged(&model, &[4, 6], &[4, 3]);
}

#[test]
fn cpu_chain_rejects_empty_batch_and_keeps_store() {
    let model = build_mlp(&[6, 8, 3]);
    assert_rejected_and_unchanged(&model, &[0, 6], &[0, 3]);
}

#[test]
fn cpu_chain_rejects_resident_input_of_another_store() {
    let model = build_mlp(&[6, 8, 3]);
    let tape = crate::tape();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let other = model.init_device_param_store(&tape).unwrap();
    let before = params_bits(&tape, &store);
    let x = tape
        .upload_resident(&other, &host_tensor(1, &[4, 6]))
        .unwrap();
    let y = tape
        .upload_resident(&other, &host_tensor(2, &[4, 3]))
        .unwrap();
    let r = model.train_step_resident(&mut store, &x, &y, &SgdConfig::new(0.05));
    assert!(
        matches!(
            r,
            Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
        ),
        "別 store の入力は InvalidArgument のはず: {r:?}"
    );
    assert_eq!(before, params_bits(&tape, &store), "拒否時は params 不変");
}

fn momentum_config() -> SgdConfig {
    SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        dampening: 0.1,
        weight_decay: 1e-3,
        nesterov: false,
    }
}

fn nesterov_config() -> SgdConfig {
    SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 1e-3,
        nesterov: true,
    }
}

/// 2 層（L1 の d_input は計算しない契約）。784 幅は debug ビルドで重いため plain の 1 構成に絞る。
#[test]
fn cpu_chain_matches_tape_path_bit_exact_two_layers() {
    compare_with_tape_path(Device::Cpu, &[784, 256, 10], 64, &SgdConfig::new(0.05), 4);
    for config in [momentum_config(), nesterov_config()] {
        compare_with_tape_path(Device::Cpu, &[96, 64, 10], 64, &config, 4);
    }
}

/// 3 層（L2 の d_input・ReLU マスクを通る）。
#[test]
fn cpu_chain_matches_tape_path_bit_exact_three_layers() {
    let config = SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
    };
    compare_with_tape_path(Device::Cpu, &[96, 64, 32, 10], 64, &config, 4);
    compare_with_tape_path(Device::Cpu, &[96, 64, 32, 10], 64, &SgdConfig::new(0.05), 3);
}

/// 端数形状と batch 1（bias 縮約の `k == 1` 分岐・-0.0 保持）。
#[test]
fn cpu_chain_matches_tape_path_bit_exact_odd_shapes_and_batch_one() {
    compare_with_tape_path(Device::Cpu, &[7, 5, 3], 3, &momentum_config(), 4);
    compare_with_tape_path(Device::Cpu, &[7, 5, 3], 1, &SgdConfig::new(0.05), 4);
    compare_with_tape_path(Device::Cpu, &[7, 5, 4, 3], 1, &momentum_config(), 3);
}

/// 最終層の出力幅 0（n == 0）は `mse_loss_scale` が inf になるため、型付きエラーで拒否し params を変えない。
#[test]
fn chain_rejects_zero_width_output() {
    let model = build_mlp(&[6, 8, 0]);
    let tape = crate::tape();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let before: Vec<Vec<u32>> = tape
        .sync_device_param_store_to_host(&store)
        .unwrap()
        .iter()
        .map(bits)
        .collect();
    let x = tape
        .upload_resident(&store, &host_tensor(1, &[4, 6]))
        .unwrap();
    let y = tape
        .upload_resident(&store, &host_tensor(2, &[4, 0]))
        .unwrap();
    let config = SgdConfig::new(0.05);
    match model.train_step_resident(&mut store, &x, &y, &config) {
        Err(AutodiffError::Backend(BackendError::Unsupported(msg))) => {
            assert!(msg.contains("n == 0"), "msg={msg}");
        }
        other => panic!("n == 0 は Unsupported のはず: {other:?}"),
    }
    let after: Vec<Vec<u32>> = tape
        .sync_device_param_store_to_host(&store)
        .unwrap()
        .iter()
        .map(bits)
        .collect();
    assert_eq!(before, after, "拒否時は params 不変");
}

use fandhe_ai_tensor_core::Device;

fn assert_bits_or_nan_class(actual: &[u32], expected: &[u32], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: len");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        let (fa, fe) = (f32::from_bits(*a), f32::from_bits(*e));
        if fe.is_nan() {
            assert!(fa.is_nan(), "{what} idx {i}: expected NaN, got {fa}");
        } else {
            assert_eq!(a, e, "{what} idx {i}: {fa} vs {fe}");
        }
    }
}

/// 同じ初期値の store を 2 つ作り、A はチェーン・B は Tape 経路で `steps` 回学習して
/// step ごとの loss と最終パラメータの bit 一致を確かめる。
fn compare_with_tape_path(
    device: Device,
    dims: &[usize],
    batch: usize,
    config: &SgdConfig,
    steps: usize,
) {
    let model = build_mlp(dims);
    let d_in = dims[0];
    let d_out = *dims.last().unwrap();
    let x = host_tensor(11, &[batch, d_in]);
    let y = host_tensor(12, &[batch, d_out]);

    let tape_a = crate::tape_for(device).expect("実機が必要");
    let mut store_a = model.init_device_param_store(&tape_a).unwrap();
    let mut store_b = model.init_device_param_store(&tape_a).unwrap();
    let x_res = tape_a.upload_resident(&store_a, &x).unwrap();
    let y_res = tape_a.upload_resident(&store_a, &y).unwrap();

    for step in 0..steps {
        let handle = model
            .train_step_resident(&mut store_a, &x_res, &y_res, config)
            .unwrap();
        let loss_a = handle.read(&mut store_a).unwrap();

        let tape_b = crate::tape_for(device).unwrap();
        let xv = tape_b.var_no_grad(&x);
        let yv = tape_b.var_no_grad(&y);
        let pred = model.forward_resident(&tape_b, &xv, &mut store_b).unwrap();
        let loss = pred.mse_loss(&yv).unwrap();
        let loss_b = loss.to_tensor().get(&[]).unwrap();
        let grads = tape_b.backward_device_param_store(&loss, &store_b).unwrap();
        tape_b
            .step_device_param_store(&mut store_b, &grads, config)
            .unwrap();

        assert_bits_or_nan_class(
            &[loss_a.to_bits()],
            &[loss_b.to_bits()],
            &format!("{dims:?} step {step} loss"),
        );
        let pa = tape_a.sync_device_param_store_to_host(&store_a).unwrap();
        let pb = tape_a.sync_device_param_store_to_host(&store_b).unwrap();
        assert_eq!(pa.len(), pb.len());
        for (i, (a, b)) in pa.iter().zip(&pb).enumerate() {
            assert_bits_or_nan_class(
                &bits(a),
                &bits(b),
                &format!("{dims:?} step {step} param {i}"),
            );
        }
    }
}

// ---- イシュー #3254: 第 2 段のフォールバック（`Unsupported` 注入）と第 1 段の拒否 ----

use fandhe_ai_autodiff::optim::resident_input::{ResidentInput, TrainStepLoss};
use std::cell::Cell;

thread_local! {
    /// 注入したチェーン実行関数が呼ばれた回数（テストスレッドごと）。
    static RUNNER_CALLS: Cell<usize> = const { Cell::new(0) };
}

type InjectedStep<'a, 't> = (
    &'a ResidentLeaf<'t>,
    Option<&'a ResidentLeaf<'t>>,
    Activation,
);

/// バックエンドが部品を持たない（最初の書き込み前の `Unsupported`）状況の注入。
fn unsupported_runner<'a, 't>(
    _store: &mut DeviceParamStore,
    _tape: &'t fandhe_ai_autodiff::Tape,
    _x: &ResidentInput,
    _y: &ResidentInput,
    _steps: &[InjectedStep<'a, 't>],
    _config: &SgdConfig,
) -> Result<TrainStepLoss, BackendError> {
    RUNNER_CALLS.with(|c| c.set(c.get() + 1));
    Err(BackendError::Unsupported("injected".to_string()))
}

/// `Unsupported` 以外の失敗（書き込み後の失敗に相当）の注入。
fn failing_runner<'a, 't>(
    _store: &mut DeviceParamStore,
    _tape: &'t fandhe_ai_autodiff::Tape,
    _x: &ResidentInput,
    _y: &ResidentInput,
    _steps: &[InjectedStep<'a, 't>],
    _config: &SgdConfig,
) -> Result<TrainStepLoss, BackendError> {
    RUNNER_CALLS.with(|c| c.set(c.get() + 1));
    Err(BackendError::KernelLaunchFailed("injected".to_string()))
}

fn runner_calls() -> usize {
    RUNNER_CALLS.with(|c| c.get())
}

/// 第 2 段: チェーンが `Unsupported` を返すと Tape 経路へ落ち、loss・全パラメータが Tape 経路と
/// bit 一致する。loss は同期なしで読め、古いハンドルは拒否される。
#[test]
fn cpu_fallback_on_unsupported_matches_tape_path_bit_exact() {
    let model = build_mlp(&[7, 5, 3]);
    let config = momentum_config();
    let x = host_tensor(11, &[3, 7]);
    let y = host_tensor(12, &[3, 3]);
    let tape = crate::tape();
    let mut store_a = model.init_device_param_store(&tape).unwrap();
    let mut store_b = model.init_device_param_store(&tape).unwrap();
    let x_res = tape.upload_resident(&store_a, &x).unwrap();
    let y_res = tape.upload_resident(&store_a, &y).unwrap();
    let before = runner_calls();
    let mut prev: Option<ResidentLoss> = None;
    for step in 0..3 {
        let handle = model
            .train_step_resident_with(&mut store_a, &x_res, &y_res, &config, unsupported_runner)
            .unwrap();
        let loss_a = handle.read(&mut store_a).unwrap();

        let xv = tape.var_no_grad(&x);
        let yv = tape.var_no_grad(&y);
        let pred = model.forward_resident(&tape, &xv, &mut store_b).unwrap();
        let loss = pred.mse_loss(&yv).unwrap();
        let loss_b = loss.to_tensor().get(&[]).unwrap();
        let grads = tape.backward_device_param_store(&loss, &store_b).unwrap();
        tape.step_device_param_store(&mut store_b, &grads, &config)
            .unwrap();
        assert_bits_or_nan_class(&[loss_a.to_bits()], &[loss_b.to_bits()], "fallback loss");
        let pa = tape.sync_device_param_store_to_host(&store_a).unwrap();
        let pb = tape.sync_device_param_store_to_host(&store_b).unwrap();
        for (i, (a, b)) in pa.iter().zip(&pb).enumerate() {
            assert_bits_or_nan_class(&bits(a), &bits(b), &format!("step {step} param {i}"));
        }
        if let Some(old) = prev.replace(handle) {
            let r = old.read(&mut store_a);
            assert!(
                matches!(
                    r,
                    Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
                ),
                "古いハンドルは拒否: {r:?}"
            );
        }
    }
    assert_eq!(runner_calls() - before, 3, "各 step でチェーンを先に試す");
}

/// `Unsupported` 以外の失敗はフォールバックせずそのまま返す。
#[test]
fn cpu_non_unsupported_chain_error_is_not_fallen_back() {
    let model = build_mlp(&[6, 8, 3]);
    let tape = crate::tape();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let before_params = params_bits(&tape, &store);
    let x = tape
        .upload_resident(&store, &host_tensor(1, &[4, 6]))
        .unwrap();
    let y = tape
        .upload_resident(&store, &host_tensor(2, &[4, 3]))
        .unwrap();
    let r =
        model.train_step_resident_with(&mut store, &x, &y, &SgdConfig::new(0.05), failing_runner);
    assert!(
        matches!(
            r,
            Err(AutodiffError::Backend(BackendError::KernelLaunchFailed(_)))
        ),
        "{r:?}"
    );
    assert_eq!(
        before_params,
        params_bits(&tape, &store),
        "従来経路は走らない"
    );
}

/// 第 1 段（範囲外）ではチェーンもフォールバックも走らず、型付きエラーで store を変えない。
#[test]
fn cpu_stage_one_rejections_never_reach_chain_or_fallback() {
    let sigmoid = Sequential::new()
        .add_linear(6, 8, 7000)
        .unwrap()
        .add_sigmoid()
        .add_linear(8, 3, 7001)
        .unwrap();
    let cases: Vec<(Sequential, Vec<usize>, Vec<usize>)> = vec![
        (sigmoid, vec![4, 6], vec![4, 3]),
        (Sequential::new(), vec![4, 6], vec![4, 3]),
        (build_mlp(&[6, 8, 3]).add_relu(), vec![4, 6], vec![4, 3]),
        (build_mlp(&[6, 8, 3]), vec![0, 6], vec![0, 3]),
    ];
    for (model, xs, ys) in cases {
        let tape = crate::tape();
        let mut store = model.init_device_param_store(&tape).unwrap();
        let before_params = params_bits(&tape, &store);
        let x = tape.upload_resident(&store, &host_tensor(1, &xs)).unwrap();
        let y = tape.upload_resident(&store, &host_tensor(2, &ys)).unwrap();
        let calls = runner_calls();
        let r = model.train_step_resident_with(
            &mut store,
            &x,
            &y,
            &SgdConfig::new(0.05),
            unsupported_runner,
        );
        assert!(
            matches!(r, Err(AutodiffError::Backend(BackendError::Unsupported(_)))),
            "{r:?}"
        );
        assert_eq!(runner_calls(), calls, "チェーンは呼ばれない");
        assert_eq!(before_params, params_bits(&tape, &store));
    }
}

/// 古い世代のハンドルは拒否されるが、拒否の後も store を使い続けられ最新のハンドルは読める。
#[test]
fn cpu_store_reusable_after_stale_handle_rejection() {
    let model = build_mlp(&[6, 8, 3]);
    let tape = crate::tape();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let x = tape
        .upload_resident(&store, &host_tensor(1, &[4, 6]))
        .unwrap();
    let y = tape
        .upload_resident(&store, &host_tensor(2, &[4, 3]))
        .unwrap();
    let config = SgdConfig::new(0.05);
    let h1 = model
        .train_step_resident(&mut store, &x, &y, &config)
        .unwrap();
    let h2 = model
        .train_step_resident(&mut store, &x, &y, &config)
        .unwrap();
    assert!(h1.read(&mut store).is_err(), "古い世代は拒否");
    assert!(h2.read(&mut store).unwrap().is_finite());
    let h3 = model
        .train_step_resident(&mut store, &x, &y, &config)
        .unwrap();
    assert!(h3.read(&mut store).unwrap().is_finite());
}
#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot;
    use std::sync::Mutex;

    /// Metal コンテキストと診断カウンタはプロセス共有のため、テストを直列化する。
    static METAL_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        METAL_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn metal_chain_matches_tape_path_bit_exact_two_layers() {
        let _g = lock();
        for config in [
            SgdConfig::new(0.05),
            SgdConfig {
                lr: 0.05,
                momentum: 0.9,
                dampening: 0.1,
                weight_decay: 1e-3,
                nesterov: false,
            },
            SgdConfig {
                lr: 0.05,
                momentum: 0.9,
                dampening: 0.0,
                weight_decay: 1e-3,
                nesterov: true,
            },
        ] {
            compare_with_tape_path(Device::Metal, &[784, 256, 10], 64, &config, 4);
        }
    }

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn metal_chain_matches_tape_path_bit_exact_three_layers() {
        let _g = lock();
        let config = SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        };
        compare_with_tape_path(Device::Metal, &[784, 128, 64, 10], 64, &config, 4);
        compare_with_tape_path(
            Device::Metal,
            &[784, 128, 64, 10],
            64,
            &SgdConfig::new(0.05),
            3,
        );
    }

    /// P7（イシュー #3254）: `wait_device_param_store` は D2H なしで `waitUntilCompleted` を 1 回だけ出し、
    /// その後の loss 読み出しは追加の待機を要しない（積んだ仕事は完了済み）。未実測（実機で未実行）。
    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn metal_wait_device_param_store_issues_exactly_one_wait() {
        let _g = lock();
        let model = build_mlp(&[784usize, 256, 10]);
        let tape = crate::tape_for(Device::Metal).expect("Metal 実機が必要");
        let mut store = model.init_device_param_store(&tape).unwrap();
        let x = tape
            .upload_resident(&store, &host_tensor(21, &[64, 784]))
            .unwrap();
        let y = tape
            .upload_resident(&store, &host_tensor(22, &[64, 10]))
            .unwrap();
        let config = SgdConfig::new(0.01);
        let h = model
            .train_step_resident(&mut store, &x, &y, &config)
            .unwrap();
        let s0 = __diagnostic_batch_counters_snapshot().unwrap();
        tape.wait_device_param_store(&store).unwrap();
        let s1 = __diagnostic_batch_counters_snapshot().unwrap();
        assert_eq!(s1.wait_until_completed - s0.wait_until_completed, 1);
        assert!(h.read(&mut store).unwrap().is_finite());
    }

    #[test]
    #[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
    fn metal_chain_step_issues_exactly_one_wait() {
        let _g = lock();
        let dims = [784usize, 256, 10];
        let model = build_mlp(&dims);
        let tape = crate::tape_for(Device::Metal).expect("Metal 実機が必要");
        let mut store = model.init_device_param_store(&tape).unwrap();
        let x = tape
            .upload_resident(&store, &host_tensor(21, &[64, 784]))
            .unwrap();
        let y = tape
            .upload_resident(&store, &host_tensor(22, &[64, 10]))
            .unwrap();
        let config = SgdConfig {
            lr: 0.01,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        };
        // パイプラインのコンパイルと pool を定常状態にする。
        for _ in 0..3 {
            let h = model
                .train_step_resident(&mut store, &x, &y, &config)
                .unwrap();
            h.read(&mut store).unwrap();
        }
        let s0 = __diagnostic_batch_counters_snapshot().unwrap();
        let handle = model
            .train_step_resident(&mut store, &x, &y, &config)
            .unwrap();
        let s1 = __diagnostic_batch_counters_snapshot().unwrap();
        let loss = handle.read(&mut store).unwrap();
        let s2 = __diagnostic_batch_counters_snapshot().unwrap();
        println!(
            "chain: encode={} command_buffers={} wait={} / read: wait={} loss={loss}",
            s1.encode_calls - s0.encode_calls,
            s1.command_buffers - s0.command_buffers,
            s1.wait_until_completed - s0.wait_until_completed,
            s2.wait_until_completed - s1.wait_until_completed,
        );
        assert!(loss.is_finite());
        assert_eq!(
            s1.encode_calls - s0.encode_calls,
            10,
            "A3(c): 学習ステップ全体の encode 回数は 10"
        );
        assert_eq!(
            s1.wait_until_completed - s0.wait_until_completed,
            0,
            "チェーン区間（encode のみ）は wait を出さない"
        );
        assert_eq!(
            s2.wait_until_completed - s0.wait_until_completed,
            1,
            "学習 1 step の wait は loss を読む 1 回だけ"
        );
    }
}

/// CUDA 実機（GB10）。部品の `_into_tracked` 版が再利用バッファへ書く経路（2 step 目以降）が、
/// 確保版と同じ bit を出す（Tape 経路と一致する）ことを確かめる（イシュー #3252）。
#[cfg(unix)]
mod cuda {
    use super::*;
    use std::sync::Mutex;

    /// CUDA コンテキストはプロセス共有のため、テストを直列化する。
    static CUDA_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        CUDA_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
    fn cuda_chain_matches_tape_path_bit_exact_two_layers() {
        let _g = lock();
        for config in [
            SgdConfig::new(0.05),
            SgdConfig {
                lr: 0.05,
                momentum: 0.9,
                dampening: 0.1,
                weight_decay: 1e-3,
                nesterov: false,
            },
            SgdConfig {
                lr: 0.05,
                momentum: 0.9,
                dampening: 0.0,
                weight_decay: 1e-3,
                nesterov: true,
            },
        ] {
            compare_with_tape_path(Device::Cuda(0), &[784, 256, 10], 64, &config, 4);
        }
    }

    /// P7（イシュー #3254）: `wait_device_param_store`（default stream の同期）が成功し、内部 tape を
    /// 作り直す `ResidentLoss::read` が同じ stream 順で正しい値を返すこと。同期・D2H の回数の実測は
    /// 未実施（backend-cuda の診断テストで別途記録する）。
    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
    fn cuda_wait_device_param_store_and_read_after_internal_tape_dropped() {
        let _g = lock();
        let model = build_mlp(&[784usize, 256, 10]);
        let tape = crate::tape_for(Device::Cuda(0)).expect("CUDA 実機が必要");
        let mut store = model.init_device_param_store(&tape).unwrap();
        let x = tape
            .upload_resident(&store, &host_tensor(21, &[64, 784]))
            .unwrap();
        let y = tape
            .upload_resident(&store, &host_tensor(22, &[64, 10]))
            .unwrap();
        let config = SgdConfig::new(0.01);
        let h = model
            .train_step_resident(&mut store, &x, &y, &config)
            .unwrap();
        tape.wait_device_param_store(&store).unwrap();
        assert!(h.read(&mut store).unwrap().is_finite());
    }

    #[test]
    #[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
    fn cuda_chain_matches_tape_path_bit_exact_three_layers() {
        let _g = lock();
        let config = SgdConfig {
            lr: 0.05,
            momentum: 0.9,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        };
        compare_with_tape_path(Device::Cuda(0), &[784, 128, 64, 10], 64, &config, 4);
        compare_with_tape_path(
            Device::Cuda(0),
            &[784, 128, 64, 10],
            64,
            &SgdConfig::new(0.05),
            3,
        );
    }
}
