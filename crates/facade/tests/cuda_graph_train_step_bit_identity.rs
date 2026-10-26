//! イシュー #3260: 学習 step 全体の Graph capture（既定 ON〈#3594〉。環境変数
//! `FANDHE_AI_CUDA_GRAPH_TRAIN=0` で OFF）を、facade の公開入口 `Sequential::train_step_resident` 経由で
//! 実行したときの loss・更新後パラメータの bit 一致を検証する CUDA 実機テスト
//! （GB10 等。`#[ignore]`。決定記録 `docs/backend-cuda-graph-step-capture-design.md` §8）。
//!
//! # 2 つの検証
//! - `in_process_capture_matches_plain_chain_bit_exact`: **同一プロセス**で、capture あり
//!   （既定 ON）と、同じ初期値・同じ入力で `override_train_graph_for_scope(false)`
//!   にした capture なしを交互に 10 step 進め、各 step の loss とパラメータを bit 比較する。
//!   併せて `train_graph_stats` の差分で結線（組み合わせごとに capture 1・replay 8）を固定する
//!   （黙って非 capture に落ちても bit 一致は合格してしまうため必須）
//! - `dump_bits_for_two_process_diff`: `FANDHE_AI_CUDA_GRAPH_TRAIN=0`（OFF・Legacy stream）のプロセスと、
//!   環境変数なし（既定 ON）のプロセスで同じ列を標準出力へ出す。`BITS` 行を `diff` で
//!   完全一致させる（stream 種別が違う 2 つの構成の突合。#1350・#1480 と同じ方式）
//!
//! # 実行方法
//! stream 種別は ordinal ごとに sticky なので、各テストは `--exact` で単独・単スレッド実行する。
//!
//! ```sh
//! # 同一プロセス比較（既定 ON のプロセス。環境変数を付けない）
//! cargo test -p fandhe-ai --release \
//!   --test cuda_graph_train_step_bit_identity \
//!   -- --ignored --nocapture --exact in_process_capture_matches_plain_chain_bit_exact
//!
//! # 2 プロセス比較（OFF = `=0`、ON = 環境変数なしの出力を diff する）
//! FANDHE_AI_CUDA_GRAPH_TRAIN=0 cargo test -p fandhe-ai --release --test cuda_graph_train_step_bit_identity \
//!   -- --ignored --nocapture --exact dump_bits_for_two_process_diff | grep '^BITS' > off.txt
//! cargo test -p fandhe-ai --release \
//!   --test cuda_graph_train_step_bit_identity \
//!   -- --ignored --nocapture --exact dump_bits_for_two_process_diff | grep '^BITS' > on.txt
//! diff off.txt on.txt
//! ```
//!
//! `fandhe_ai_backend_cuda::graph` の `train_graph_stats`・`override_train_graph_for_scope` は
//! 診断・テスト専用で facade へは再公開しない（`api_surface.rs` が固定する）。facade の `Sequential`
//! は bias なしの Linear を作れないため、bias なし経路は公開面からは到達できない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, DeviceParamStore, SgdConfig, Tape, Tensor};
use fandhe_ai_backend_cuda::graph::{
    override_train_graph_for_scope, train_graph_enabled, train_graph_stats,
};

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

/// 検証する組み合わせ（名前・層の幅・SGD 設定）。
fn combos() -> Vec<(&'static str, Vec<usize>, SgdConfig)> {
    vec![
        ("2layer-plain", vec![784, 256, 10], plain()),
        ("2layer-momentum", vec![784, 256, 10], momentum()),
        ("2layer-nesterov", vec![784, 256, 10], nesterov()),
        ("3layer-plain", vec![784, 128, 64, 10], plain()),
        ("3layer-momentum", vec![784, 128, 64, 10], momentum()),
        ("3layer-nesterov", vec![784, 128, 64, 10], nesterov()),
    ]
}

const BATCH: usize = 64;
const STEPS: usize = 10;

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

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。既定 ON（`FANDHE_AI_CUDA_GRAPH_TRAIN=0` を付けず）で --exact 単独実行"]
fn in_process_capture_matches_plain_chain_bit_exact() {
    assert!(
        train_graph_enabled(),
        "`FANDHE_AI_CUDA_GRAPH_TRAIN=0` など OFF の値を付けずに起動すること（stream 種別は起動時に決まる）"
    );
    let before = train_graph_stats();
    let n_combos = combos().len() as u64;
    for (name, dims, config) in combos() {
        let model = build_mlp(&dims);
        let x = host_tensor(21, &[BATCH, dims[0]]);
        let y = host_tensor(22, &[BATCH, *dims.last().unwrap()]);
        let tape = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA device 0");
        let mut store_on = model.init_device_param_store(&tape).unwrap();
        let mut store_off = model.init_device_param_store(&tape).unwrap();
        let x_on = tape.upload_resident(&store_on, &x).unwrap();
        let y_on = tape.upload_resident(&store_on, &y).unwrap();
        let x_off = tape.upload_resident(&store_off, &x).unwrap();
        let y_off = tape.upload_resident(&store_off, &y).unwrap();
        for step in 0..STEPS {
            let h_on = {
                let _on = override_train_graph_for_scope(true);
                model
                    .train_step_resident(&mut store_on, &x_on, &y_on, &config)
                    .unwrap()
            };
            let h_off = {
                let _off = override_train_graph_for_scope(false);
                model
                    .train_step_resident(&mut store_off, &x_off, &y_off, &config)
                    .unwrap()
            };
            let l_on = h_on.read(&mut store_on).unwrap();
            let l_off = h_off.read(&mut store_off).unwrap();
            assert_bits_or_nan_class(
                &[l_on.to_bits()],
                &[l_off.to_bits()],
                &format!("{name} step {step} loss"),
            );
            let (p_on, p_off) = (
                params_bits(&tape, &store_on),
                params_bits(&tape, &store_off),
            );
            for (i, (a, b)) in p_on.iter().zip(&p_off).enumerate() {
                assert_bits_or_nan_class(a, b, &format!("{name} step {step} param {i}"));
            }
        }
    }
    let d = train_graph_stats();
    assert_eq!(
        d.captured - before.captured,
        n_combos,
        "組み合わせごとに capture は 1 回"
    );
    assert_eq!(
        d.replayed - before.replayed,
        n_combos * (STEPS as u64 - 2),
        "step 3 以降は replay（step 1 はウォームアップ、step 2 で capture）"
    );
    assert_eq!(d.fallbacks - before.fallbacks, 0);
}

/// 2 プロセス比較用: 環境変数だけで ON/OFF が決まる状態で全組み合わせを回し、各 step の loss と
/// 最終パラメータの bit を `BITS` 行で出す。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。`=0` の有無で 2 回 --exact 単独実行して diff"]
fn dump_bits_for_two_process_diff() {
    for (name, dims, config) in combos() {
        let model = build_mlp(&dims);
        let x = host_tensor(21, &[BATCH, dims[0]]);
        let y = host_tensor(22, &[BATCH, *dims.last().unwrap()]);
        let tape = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA device 0");
        let mut store = model.init_device_param_store(&tape).unwrap();
        let x_res = tape.upload_resident(&store, &x).unwrap();
        let y_res = tape.upload_resident(&store, &y).unwrap();
        for step in 0..STEPS {
            let h = model
                .train_step_resident(&mut store, &x_res, &y_res, &config)
                .unwrap();
            println!(
                "BITS {name} step {step} loss {:08x}",
                h.read(&mut store).unwrap().to_bits()
            );
        }
        for (i, p) in params_bits(&tape, &store).iter().enumerate() {
            let hash = p.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, v| {
                (h ^ u64::from(*v)).wrapping_mul(0x0000_0100_0000_01b3)
            });
            println!(
                "BITS {name} final param {i} len {} fnv {hash:016x}",
                p.len()
            );
        }
    }
    let s = train_graph_stats();
    println!(
        "STATS captured={} replayed={} fallbacks={}",
        s.captured, s.replayed, s.fallbacks
    );
}
