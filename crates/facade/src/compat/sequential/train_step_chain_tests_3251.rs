//! イシュー #3251: 学習ステップ全体を encode-only で積む内部チェーン
//! （`DeviceParamStore::train_step_device_chain`）の facade 側テスト。
//!
//! # 役割
//! 公開入口 `Sequential::train_step_resident`・`ResidentLoss`（P5・P6）は #3254 の担当で、
//! 本 PR では facade の公開面を増やさない。生の autodiff `Tape` は `Tape.0` が `pub(crate)` の
//! ため facade の利用者からは渡せない。そこで `Sequential` の層ガード・`build_device_chain_steps`
//! を使う補助を `#[cfg(test)]` のこのモジュールに置き、チェーンを Tape 経路と突き合わせる。
//! #3254 はこの補助の形を本番コードへ移す。
//!
//! # テスト
//! - CPU（CI で実行）: CPU バックエンドは部品が未実装（#3253）で `Unsupported`。store が不変で
//!   poison しないことを固定する（黙って別の計算をしない）。#3253 が入って `Ok` になったら
//!   loss の有限性を確かめる（期待値の更新はそのときの PR で行う）
//! - Metal（`#[ignore]`・実機）: loss と更新後パラメータが Tape 経路と bit 一致すること
//!   （momentum 0／>0・2 層／3 層）、1 step の `waitUntilCompleted` が loss を読む 1 回だけ
//!   であること。受け入れ記録は `docs/perf/logs/train-step-chain-metal-3251/`（Mac mini）

use super::*;
use fandhe_ai_autodiff::optim::SgdConfig;
use fandhe_ai_autodiff::optim::resident_input::TrainStepLoss;

/// #3254 が本番コードへ移す想定の補助。`Sequential` を層ガード越しにチェーンへ渡す。
fn chain_train_step(
    model: &Sequential,
    store: &mut DeviceParamStore,
    tape: &Tape,
    x: &ResidentTensor,
    y: &ResidentTensor,
    config: &SgdConfig,
) -> Result<TrainStepLoss, BackendError> {
    if model.contains_resident_unsupported_layer() {
        return Err(BackendError::Unsupported(
            "chain_train_step: unsupported layer for the device-resident path".to_string(),
        ));
    }
    model.reject_untracked_parametric_layer_resident("chain_train_step")?;
    let leaves = store.snapshot_resident_params(&tape.0)?;
    let steps = model.build_device_chain_steps(&leaves).ok_or_else(|| {
        BackendError::Unsupported(
            "chain_train_step: not a Linear(+ReLU) chain (no host fallback)".to_string(),
        )
    })?;
    store.train_step_device_chain(&tape.0, &x.0, &y.0, &steps, config)
}

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
fn cpu_chain_keeps_the_store_unchanged_until_cpu_parts_exist() {
    let model = build_mlp(&[6, 8, 3]);
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
        .upload_resident(&store, &host_tensor(2, &[4, 3]))
        .unwrap();
    let config = SgdConfig::new(0.05);
    match chain_train_step(&model, &mut store, &tape, &x, &y, &config) {
        // 部品が未実装の間（#3253 まで）は Unsupported で止まり、何も変えない。
        Err(BackendError::Unsupported(_)) => {
            let after: Vec<Vec<u32>> = tape
                .sync_device_param_store_to_host(&store)
                .unwrap()
                .iter()
                .map(bits)
                .collect();
            assert_eq!(before, after, "Unsupported のとき params は不変");
        }
        // #3253 で CPU の部品が入った後はチェーンが通る。loss が有限であることだけ確かめる。
        Ok(loss) => {
            let v = store.read_train_step_loss(&tape.0, &loss).unwrap();
            assert!(v.is_finite(), "loss={v}");
        }
        Err(e) => panic!("想定外のエラー: {e:?}"),
    }
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
    match chain_train_step(&model, &mut store, &tape, &x, &y, &config) {
        Err(BackendError::Unsupported(msg)) => {
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

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot;
    use fandhe_ai_tensor_core::Device;
    use std::sync::Mutex;

    /// Metal コンテキストと診断カウンタはプロセス共有のため、テストを直列化する。
    static METAL_LOCK: Mutex<()> = Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        METAL_LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

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
    fn compare_with_tape_path(dims: &[usize], batch: usize, config: &SgdConfig, steps: usize) {
        let model = build_mlp(dims);
        let d_in = dims[0];
        let d_out = *dims.last().unwrap();
        let x = host_tensor(11, &[batch, d_in]);
        let y = host_tensor(12, &[batch, d_out]);

        let tape_a = crate::tape_for(Device::Metal).expect("Metal 実機が必要");
        let mut store_a = model.init_device_param_store(&tape_a).unwrap();
        let mut store_b = model.init_device_param_store(&tape_a).unwrap();
        let x_res = tape_a.upload_resident(&store_a, &x).unwrap();
        let y_res = tape_a.upload_resident(&store_a, &y).unwrap();

        for step in 0..steps {
            let handle =
                chain_train_step(&model, &mut store_a, &tape_a, &x_res, &y_res, config).unwrap();
            let loss_a = store_a.read_train_step_loss(&tape_a.0, &handle).unwrap();

            let tape_b = crate::tape_for(Device::Metal).unwrap();
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
            compare_with_tape_path(&[784, 256, 10], 64, &config, 4);
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
        compare_with_tape_path(&[784, 128, 64, 10], 64, &config, 4);
        compare_with_tape_path(&[784, 128, 64, 10], 64, &SgdConfig::new(0.05), 3);
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
            let h = chain_train_step(&model, &mut store, &tape, &x, &y, &config).unwrap();
            store.read_train_step_loss(&tape.0, &h).unwrap();
        }
        let s0 = __diagnostic_batch_counters_snapshot().unwrap();
        let handle = chain_train_step(&model, &mut store, &tape, &x, &y, &config).unwrap();
        let s1 = __diagnostic_batch_counters_snapshot().unwrap();
        let loss = store.read_train_step_loss(&tape.0, &handle).unwrap();
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
