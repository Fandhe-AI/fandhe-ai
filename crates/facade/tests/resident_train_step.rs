//! イシュー #3254: 常駐学習ステップの公開面（P1〜P7）をクレート外から公開 API だけで確かめる
//! CPU テスト（CI で実行）。
//!
//! # 役割
//! `facade` は唯一のサポートされる公開面のため、`Sequential::train_step_resident`・
//! `ResidentLoss::read`・`Tape::wait_device_param_store` を内部 API なしで使えることと、
//! 範囲外・古い世代・別 store の型付きエラー契約を固定する。
//!
//! イシュー #3255: 公開 API だけで常駐チェーンと Tape 経路（`forward_resident` → `mse_loss` →
//! `backward_device_param_store` → `step_device_param_store`）の loss・更新後パラメータの bit 一致
//! （NaN はクラス一致。tolerance は使わない）も本ファイルで固定する。注入による第 2 段フォールバックの
//! bit 一致は非公開の `train_step_resident_with` が要るためクレート内テスト
//! （`compat/sequential/train_step_chain_tests_3251.rs`）が担う。CPU の tape はすべて同一デバイスの
//! ため公開面で `TapeMismatch` は作れず、別デバイスの tape（`DeviceMismatch`）は `#[ignore]` の
//! 実機テストで扱う。CUDA・Metal の bit 一致も `#[ignore]`（実行・実測は #3256）。
//! 決定記録は `docs/compat-resident-train-step-decision.md` §4・§14・§15。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{AutodiffError, BackendError, Device, DeviceParamStore, SgdConfig, Tape, Tensor};
use std::sync::Mutex;

fn model() -> Sequential {
    Sequential::new()
        .add_linear(4, 8, 11)
        .unwrap()
        .add_relu()
        .add_linear(8, 2, 12)
        .unwrap()
}

fn data(len: usize, seed: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32 + seed) * 0.37).sin()).collect()
}

#[test]
fn one_step_loss_is_finite_and_wait_is_ok() {
    let tape = fandhe_ai::tape();
    let model = model();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let x = tape
        .upload_resident(&store, &Tensor::new(data(12, 1.0), &[3, 4]).unwrap())
        .unwrap();
    let y = tape
        .upload_resident(&store, &Tensor::new(data(6, 2.0), &[3, 2]).unwrap())
        .unwrap();
    let loss = model
        .train_step_resident(&mut store, &x, &y, &SgdConfig::new(0.05))
        .unwrap();
    tape.wait_device_param_store(&store).unwrap();
    assert!(loss.read(&mut store).unwrap().is_finite());
}

#[test]
fn stale_loss_handle_is_invalid_argument_and_new_handle_reads() {
    let tape = fandhe_ai::tape();
    let model = model();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let x = tape
        .upload_resident(&store, &Tensor::new(data(12, 1.0), &[3, 4]).unwrap())
        .unwrap();
    let y = tape
        .upload_resident(&store, &Tensor::new(data(6, 2.0), &[3, 2]).unwrap())
        .unwrap();
    let config = SgdConfig::new(0.05);
    let first = model
        .train_step_resident(&mut store, &x, &y, &config)
        .unwrap();
    // 2 回目の step の後、`train_step_resident` 内部の tape はすべて drop 済み。
    let second = model
        .train_step_resident(&mut store, &x, &y, &config)
        .unwrap();
    assert!(matches!(
        first.read(&mut store),
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    // 拒否の後も store は使い続けられる。
    assert!(second.read(&mut store).unwrap().is_finite());
    let third = model
        .train_step_resident(&mut store, &x, &y, &config)
        .unwrap();
    assert!(third.read(&mut store).unwrap().is_finite());
}

#[test]
fn handles_of_another_store_are_invalid_argument() {
    let tape = fandhe_ai::tape();
    let model = model();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let mut other = model.init_device_param_store(&tape).unwrap();
    let x_other = tape
        .upload_resident(&other, &Tensor::new(data(12, 1.0), &[3, 4]).unwrap())
        .unwrap();
    let y_other = tape
        .upload_resident(&other, &Tensor::new(data(6, 2.0), &[3, 2]).unwrap())
        .unwrap();
    let config = SgdConfig::new(0.05);
    // 別 store の入力。
    let r = model.train_step_resident(&mut store, &x_other, &y_other, &config);
    assert!(matches!(
        r,
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    // 別 store の loss。
    let loss_other = model
        .train_step_resident(&mut other, &x_other, &y_other, &config)
        .unwrap();
    assert!(matches!(
        loss_other.read(&mut store),
        Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
    ));
    assert!(loss_other.read(&mut other).unwrap().is_finite());
}

#[test]
fn out_of_scope_model_is_a_typed_error_and_store_is_reusable() {
    let tape = fandhe_ai::tape();
    let bad = Sequential::new()
        .add_linear(4, 8, 11)
        .unwrap()
        .add_sigmoid()
        .add_linear(8, 2, 12)
        .unwrap();
    let mut store = bad.init_device_param_store(&tape).unwrap();
    let x = tape
        .upload_resident(&store, &Tensor::new(data(12, 1.0), &[3, 4]).unwrap())
        .unwrap();
    let y = tape
        .upload_resident(&store, &Tensor::new(data(6, 2.0), &[3, 2]).unwrap())
        .unwrap();
    let r = bad.train_step_resident(&mut store, &x, &y, &SgdConfig::new(0.05));
    assert!(matches!(
        r,
        Err(AutodiffError::Backend(BackendError::Unsupported(_)))
    ));
    // 拒否は store を変えず、同じ store で推論・次の入口を呼べる。
    assert!(tape.wait_device_param_store(&store).is_ok());
}
// ---- イシュー #3255: 公開 API だけでの Tape 経路との bit 一致と範囲外契約 ----

fn host_tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(Xorshift64Star::new(seed).fill_vec(n), shape).unwrap()
}

fn build_mlp(dims: &[usize]) -> Sequential {
    let mut model = Sequential::new();
    for (i, w) in dims.windows(2).enumerate() {
        if i > 0 {
            model = model.add_relu();
        }
        model = model.add_linear(w[0], w[1], 9000 + i as u64).unwrap();
    }
    model
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は as_slice() が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn params_bits(tape: &Tape, store: &DeviceParamStore) -> Vec<Vec<u32>> {
    tape.sync_device_param_store_to_host(store)
        .unwrap()
        .iter()
        .map(bits)
        .collect()
}

/// 期待値が NaN なら NaN クラス一致、それ以外は bit 完全一致（tolerance は使わない）。
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

fn plain() -> SgdConfig {
    SgdConfig::new(0.05)
}

fn momentum() -> SgdConfig {
    SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        dampening: 0.1,
        weight_decay: 1e-3,
        nesterov: false,
    }
}

fn nesterov() -> SgdConfig {
    SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 1e-3,
        nesterov: true,
    }
}

/// Tape 経路で 1 step 進めて loss を返す（毎 step 新しい tape）。
fn tape_path_step(
    device: Device,
    model: &Sequential,
    store: &mut DeviceParamStore,
    x: &Tensor<f32>,
    y: &Tensor<f32>,
    config: &SgdConfig,
) -> f32 {
    let tape = fandhe_ai::tape_for(device).expect("デバイスが利用可能な前提");
    let xv = tape.var_no_grad(x);
    let yv = tape.var_no_grad(y);
    let pred = model.forward_resident(&tape, &xv, store).unwrap();
    let loss = pred.mse_loss(&yv).unwrap();
    let v = loss.to_tensor().get(&[]).unwrap();
    let grads = tape.backward_device_param_store(&loss, store).unwrap();
    tape.step_device_param_store(store, &grads, config).unwrap();
    v
}

fn compare_params(tape: &Tape, a: &DeviceParamStore, b: &DeviceParamStore, what: &str) {
    let pa = params_bits(tape, a);
    let pb = params_bits(tape, b);
    assert_eq!(pa.len(), pb.len(), "{what}: param count");
    for (i, (x, y)) in pa.iter().zip(&pb).enumerate() {
        assert_bits_or_nan_class(x, y, &format!("{what} param {i}"));
    }
}

/// チェーン（A）と Tape 経路（B）を同じ初期値から `steps` 回進め、loss と全パラメータを bit 比較する。
/// `read_every_step == false` なら途中の loss を読まず最後のハンドルだけ読む（同期を挟まない積み方）。
fn compare_with_tape_path_ext(
    device: Device,
    dims: &[usize],
    batch: usize,
    config: &SgdConfig,
    steps: usize,
    read_every_step: bool,
) {
    let model = build_mlp(dims);
    let x = host_tensor(21, &[batch, dims[0]]);
    let y = host_tensor(22, &[batch, *dims.last().unwrap()]);
    let tape_a = fandhe_ai::tape_for(device).expect("デバイスが利用可能な前提");
    let mut store_a = model.init_device_param_store(&tape_a).unwrap();
    let mut store_b = model.init_device_param_store(&tape_a).unwrap();
    let x_res = tape_a.upload_resident(&store_a, &x).unwrap();
    let y_res = tape_a.upload_resident(&store_a, &y).unwrap();

    let mut handles = Vec::new();
    let mut last_loss_b = 0.0f32;
    for step in 0..steps {
        let h = model
            .train_step_resident(&mut store_a, &x_res, &y_res, config)
            .unwrap();
        let lb = tape_path_step(device, &model, &mut store_b, &x, &y, config);
        if read_every_step {
            let la = h.read(&mut store_a).unwrap();
            assert_bits_or_nan_class(
                &[la.to_bits()],
                &[lb.to_bits()],
                &format!("{dims:?} step {step} loss"),
            );
            compare_params(
                &tape_a,
                &store_a,
                &store_b,
                &format!("{dims:?} step {step}"),
            );
        } else {
            last_loss_b = lb;
            handles.push(h);
        }
    }
    if !read_every_step {
        let last = handles.pop().unwrap();
        for stale in &handles {
            assert!(matches!(
                stale.read(&mut store_a),
                Err(AutodiffError::Backend(BackendError::InvalidArgument(_)))
            ));
        }
        let la = last.read(&mut store_a).unwrap();
        assert_bits_or_nan_class(
            &[la.to_bits()],
            &[last_loss_b.to_bits()],
            &format!("{dims:?} final loss"),
        );
        compare_params(&tape_a, &store_a, &store_b, &format!("{dims:?} final"));
    }
}

fn compare_with_tape_path(
    device: Device,
    dims: &[usize],
    batch: usize,
    config: &SgdConfig,
    steps: usize,
) {
    compare_with_tape_path_ext(device, dims, batch, config, steps, true);
}

#[test]
fn public_api_chain_matches_tape_path_bit_exact_two_layers() {
    for config in [plain(), momentum(), nesterov()] {
        compare_with_tape_path(Device::Cpu, &[96, 64, 10], 16, &config, 4);
    }
}

#[test]
fn public_api_chain_matches_tape_path_bit_exact_three_layers() {
    let m = SgdConfig {
        lr: 0.05,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
    };
    compare_with_tape_path(Device::Cpu, &[32, 24, 16, 4], 8, &m, 4);
    compare_with_tape_path(Device::Cpu, &[32, 24, 16, 4], 8, &plain(), 3);
}

#[test]
fn public_api_chain_matches_tape_path_bit_exact_odd_shapes_and_batch_one() {
    compare_with_tape_path(Device::Cpu, &[7, 5, 3], 3, &momentum(), 4);
    compare_with_tape_path(Device::Cpu, &[7, 5, 3], 1, &plain(), 4);
    compare_with_tape_path(Device::Cpu, &[7, 5, 4, 3], 1, &momentum(), 3);
}

#[test]
fn public_api_chain_unread_steps_then_final_params_match_tape_path() {
    compare_with_tape_path_ext(Device::Cpu, &[32, 24, 4], 8, &momentum(), 4, false);
}

/// 範囲外の構成: 期待する型付きエラーで止まり、params が bit 単位で不変で、store が使い続けられること。
fn assert_rejected_and_unchanged(
    model: &Sequential,
    x_shape: &[usize],
    y_shape: &[usize],
    config: &SgdConfig,
    check: impl Fn(&AutodiffError) -> bool,
) {
    let tape = fandhe_ai::tape();
    let mut store = model.init_device_param_store(&tape).unwrap();
    let before = params_bits(&tape, &store);
    let x = tape
        .upload_resident(&store, &host_tensor(1, x_shape))
        .unwrap();
    let y = tape
        .upload_resident(&store, &host_tensor(2, y_shape))
        .unwrap();
    match model.train_step_resident(&mut store, &x, &y, config) {
        Err(e) => assert!(check(&e), "想定外のエラー: {e:?}"),
        Ok(_) => panic!("範囲外は拒否されるはず"),
    }
    assert_eq!(before, params_bits(&tape, &store), "拒否時は params 不変");
    assert!(tape.wait_device_param_store(&store).is_ok());
}

fn is_unsupported(e: &AutodiffError) -> bool {
    matches!(e, AutodiffError::Backend(BackendError::Unsupported(_)))
}

fn is_invalid_argument(e: &AutodiffError) -> bool {
    matches!(e, AutodiffError::Backend(BackendError::InvalidArgument(_)))
}

/// 第 1 段の拒否はフォールバックせず `Unsupported` で止まる（注入によるフォールバックはクレート内テスト）。
#[test]
fn public_api_out_of_scope_models_are_unsupported_without_fallback() {
    let tanh = Sequential::new()
        .add_linear(6, 8, 1)
        .unwrap()
        .add_tanh()
        .add_linear(8, 3, 2)
        .unwrap();
    assert_rejected_and_unchanged(&tanh, &[4, 6], &[4, 3], &plain(), is_unsupported);
    let trailing = build_mlp(&[6, 8, 3]).add_relu();
    assert_rejected_and_unchanged(&trailing, &[4, 6], &[4, 3], &plain(), is_unsupported);
    let zero = build_mlp(&[6, 8, 0]);
    assert_rejected_and_unchanged(
        &zero,
        &[4, 6],
        &[4, 0],
        &plain(),
        |e| matches!(e, AutodiffError::Backend(BackendError::Unsupported(m)) if m.contains("n == 0")),
    );
    let m = build_mlp(&[6, 8, 3]);
    assert_rejected_and_unchanged(&m, &[0, 6], &[0, 3], &plain(), is_unsupported);
}

#[test]
fn public_api_invalid_sgd_config_is_rejected_and_keeps_store() {
    let m = build_mlp(&[6, 8, 3]);
    for lr in [f32::NAN, -0.1] {
        assert_rejected_and_unchanged(&m, &[4, 6], &[4, 3], &SgdConfig::new(lr), |e| {
            matches!(e, AutodiffError::Backend(_))
        });
    }
}

#[test]
fn public_api_shape_mismatch_is_typed_error_and_keeps_store() {
    let m = build_mlp(&[6, 8, 3]);
    // x の列数が第 1 層と合わない。
    assert_rejected_and_unchanged(&m, &[4, 5], &[4, 3], &plain(), |_| true);
    // x と y の行数が合わない。
    assert_rejected_and_unchanged(&m, &[4, 6], &[3, 3], &plain(), |_| true);
}

#[test]
fn public_api_changing_momentum_midway_is_rejected() {
    let m = build_mlp(&[6, 8, 3]);
    let tape = fandhe_ai::tape();
    let mut store = m.init_device_param_store(&tape).unwrap();
    let x = tape
        .upload_resident(&store, &host_tensor(1, &[4, 6]))
        .unwrap();
    let y = tape
        .upload_resident(&store, &host_tensor(2, &[4, 3]))
        .unwrap();
    m.train_step_resident(&mut store, &x, &y, &momentum())
        .unwrap();
    let before = params_bits(&tape, &store);
    match m.train_step_resident(&mut store, &x, &y, &plain()) {
        Err(e) => assert!(is_invalid_argument(&e), "{e:?}"),
        Ok(_) => panic!("momentum 構成の途中変更は拒否されるはず"),
    }
    assert_eq!(before, params_bits(&tape, &store));
}

#[test]
fn public_api_store_used_by_rmsprop_is_rejected() {
    let m = build_mlp(&[6, 8, 3]);
    let tape = fandhe_ai::tape();
    let mut store = m.init_device_param_store(&tape).unwrap();
    let xt = host_tensor(1, &[4, 6]);
    let yt = host_tensor(2, &[4, 3]);
    {
        let xv = tape.var_no_grad(&xt);
        let yv = tape.var_no_grad(&yt);
        let pred = m.forward_resident(&tape, &xv, &mut store).unwrap();
        let loss = pred.mse_loss(&yv).unwrap();
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        tape.step_device_param_store_rmsprop(
            &mut store,
            &grads,
            &fandhe_ai::optim::RmsPropConfig::default(),
        )
        .unwrap();
    }
    let before = params_bits(&tape, &store);
    let x = tape.upload_resident(&store, &xt).unwrap();
    let y = tape.upload_resident(&store, &yt).unwrap();
    match m.train_step_resident(&mut store, &x, &y, &plain()) {
        Err(e) => assert!(is_invalid_argument(&e), "{e:?}"),
        Ok(_) => panic!("RMSProp 使用後の store は拒否されるはず"),
    }
    assert_eq!(before, params_bits(&tape, &store));
}

#[test]
fn public_api_pending_forward_is_rejected() {
    let m = build_mlp(&[6, 8, 3]);
    let tape = fandhe_ai::tape();
    let mut store = m.init_device_param_store(&tape).unwrap();
    let xt = host_tensor(1, &[4, 6]);
    let xv = tape.var_no_grad(&xt);
    let _pending = m.forward_resident(&tape, &xv, &mut store).unwrap();
    let x = tape.upload_resident(&store, &xt).unwrap();
    let y = tape
        .upload_resident(&store, &host_tensor(2, &[4, 3]))
        .unwrap();
    let r = m.train_step_resident(&mut store, &x, &y, &plain());
    assert!(
        matches!(
            r,
            Err(AutodiffError::Backend(
                BackendError::PendingForwardUnconsumed
            ))
        ),
        "{:?}",
        r.err()
    );
}

/// ハンドルは store_id＋device に束縛され、tape インスタンスには束縛されない（決定記録 §4）。
#[test]
fn resident_tensor_from_another_tape_instance_is_accepted_and_bit_identical() {
    let m = build_mlp(&[6, 8, 3]);
    let tape1 = fandhe_ai::tape();
    let tape2 = fandhe_ai::tape();
    let mut s1 = m.init_device_param_store(&tape1).unwrap();
    let mut s2 = m.init_device_param_store(&tape1).unwrap();
    let (xt, yt) = (host_tensor(1, &[4, 6]), host_tensor(2, &[4, 3]));
    let x1 = tape1.upload_resident(&s1, &xt).unwrap();
    let y1 = tape1.upload_resident(&s1, &yt).unwrap();
    let x2 = tape2.upload_resident(&s2, &xt).unwrap();
    let y2 = tape2.upload_resident(&s2, &yt).unwrap();
    for _ in 0..3 {
        let l1 = m
            .train_step_resident(&mut s1, &x1, &y1, &momentum())
            .unwrap()
            .read(&mut s1)
            .unwrap();
        let l2 = m
            .train_step_resident(&mut s2, &x2, &y2, &momentum())
            .unwrap()
            .read(&mut s2)
            .unwrap();
        assert_eq!(l1.to_bits(), l2.to_bits());
    }
    tape2.wait_device_param_store(&s2).unwrap();
    compare_params(&tape1, &s1, &s2, "another tape");
}

#[test]
fn only_y_of_another_store_is_invalid_argument_and_keeps_store() {
    let m = build_mlp(&[6, 8, 3]);
    let tape = fandhe_ai::tape();
    let mut store = m.init_device_param_store(&tape).unwrap();
    let other = m.init_device_param_store(&tape).unwrap();
    let before = params_bits(&tape, &store);
    let x = tape
        .upload_resident(&store, &host_tensor(1, &[4, 6]))
        .unwrap();
    let y_other = tape
        .upload_resident(&other, &host_tensor(2, &[4, 3]))
        .unwrap();
    match m.train_step_resident(&mut store, &x, &y_other, &plain()) {
        Err(e) => assert!(is_invalid_argument(&e), "{e:?}"),
        Ok(_) => panic!("別 store の y は拒否されるはず"),
    }
    assert_eq!(before, params_bits(&tape, &store));
}

// ---- 実機（`#[ignore]`。実行・実測は #3256） ----

// プロセス共有のデバイスコンテキストを直列化する（poison は回復する）。
// 使用側（Metal 実機テスト）と同じ cfg。非 macOS では dead_code になるため。
#[cfg(target_os = "macos")]
static METAL_LOCK: Mutex<()> = Mutex::new(());
static CUDA_LOCK: Mutex<()> = Mutex::new(());

fn gpu_two_layers(device: Device) {
    for config in [plain(), momentum(), nesterov()] {
        compare_with_tape_path(device, &[784, 256, 10], 64, &config, 4);
    }
}

fn gpu_three_layers(device: Device) {
    for config in [plain(), momentum()] {
        compare_with_tape_path(device, &[784, 128, 64, 10], 64, &config, 4);
    }
}

fn gpu_unread(device: Device) {
    compare_with_tape_path_ext(device, &[784, 128, 10], 64, &momentum(), 4, false);
}

/// デバイス store を CPU tape で扱うと `DeviceMismatch`。デバイス入力を CPU store へ渡すと
/// store_id 検査が先に走り `InvalidArgument`。
fn gpu_cross_device(device: Device) {
    let m = build_mlp(&[6, 8, 3]);
    let dev_tape = fandhe_ai::tape_for(device).expect("デバイスが利用可能な前提");
    let cpu_tape = fandhe_ai::tape();
    let dev_store = m.init_device_param_store(&dev_tape).unwrap();
    assert!(matches!(
        cpu_tape.upload_resident(&dev_store, &host_tensor(1, &[4, 6])),
        Err(AutodiffError::Backend(BackendError::DeviceMismatch))
    ));
    assert!(matches!(
        cpu_tape.wait_device_param_store(&dev_store),
        Err(AutodiffError::Backend(BackendError::DeviceMismatch))
    ));
    let mut cpu_store = m.init_device_param_store(&cpu_tape).unwrap();
    let x = dev_tape
        .upload_resident(&dev_store, &host_tensor(1, &[4, 6]))
        .unwrap();
    let y = dev_tape
        .upload_resident(&dev_store, &host_tensor(2, &[4, 3]))
        .unwrap();
    let r = m.train_step_resident(&mut cpu_store, &x, &y, &plain());
    assert!(
        matches!(r, Err(ref e) if is_invalid_argument(e)),
        "{:?}",
        r.err()
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_public_api_chain_matches_tape_path_bit_exact_two_layers() {
    let _g = METAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    gpu_two_layers(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_public_api_chain_matches_tape_path_bit_exact_three_layers() {
    let _g = METAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    gpu_three_layers(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_public_api_unread_steps_then_final_params_match_tape_path() {
    let _g = METAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    gpu_unread(Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_upload_resident_with_cpu_tape_is_device_mismatch() {
    let _g = METAL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    gpu_cross_device(Device::Metal);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
fn cuda_public_api_chain_matches_tape_path_bit_exact_two_layers() {
    let _g = CUDA_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    gpu_two_layers(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
fn cuda_public_api_chain_matches_tape_path_bit_exact_three_layers() {
    let _g = CUDA_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    gpu_three_layers(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
fn cuda_public_api_unread_steps_then_final_params_match_tape_path() {
    let _g = CUDA_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    gpu_unread(Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
fn cuda_upload_resident_with_cpu_tape_is_device_mismatch() {
    let _g = CUDA_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    gpu_cross_device(Device::Cuda(0));
}

/// 精度モードを drop で元に戻す RAII ガード（`CUDA_LOCK` 保持中だけ使う）。
struct CudaPrecisionGuard(fandhe_ai::CudaGemmPrecision);

impl Drop for CudaPrecisionGuard {
    fn drop(&mut self) {
        fandhe_ai::set_cuda_gemm_precision(self.0);
    }
}

/// TF32 モードでも常駐チェーンが Tape 経路と bit 一致する。第 2 段フォールバックに入るかどうかは
/// backend-cuda 側の部品の挙動に依存するため、本テストはフォールバックの発生を主張せず結果の一致だけを固定する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）依存。CI では実行しない"]
fn cuda_tf32_mode_chain_matches_tape_path_bit_exact() {
    let _g = CUDA_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _restore = CudaPrecisionGuard(fandhe_ai::cuda_gemm_precision());
    fandhe_ai::set_cuda_gemm_precision(fandhe_ai::CudaGemmPrecision::Tf32);
    compare_with_tape_path(Device::Cuda(0), &[784, 256, 10], 64, &momentum(), 4);
}
