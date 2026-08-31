//! イシュー #1015（親ツリー）・#1017（実装）・`docs/backend-metal-
//! command-batching-design.md` §4 の Mac セッション記入用ベンチ。
//!
//! `scripts/bench/framework-compare/bench-fandhe/src/main.rs` の
//! `run_train`（fresh）／`run_train_reuse`（reuse）と**同一のモデル形状・
//! 乱数シード・step 数・warmup 数・計測プロトコル**（`TRAIN_STEPS=100`
//! ・先頭 20 step を warmup として捨て、残り 80 step の median/Q1/Q3 を
//! 取る）を、`fandhe-ai =0.4.0`（crates.io ピン）ではなく**本ワークツリー
//! の HEAD（`facade` crate への path 依存）**で再現する。
//!
//! **境界差の明記（設計文書 §4「比較の基準点」・タスク指示）**:
//! `scripts/bench/framework-compare/results/summary.md` 環境 5 の
//! metal train reuse 中央値 20.381 ms・#1015 イシュー本文が挙げる
//! 18.6 ms は `fandhe-ai =0.4.0`（2026-08-29 crates.io 公開版）を計測
//! したものであり、#1017（本イシュー）だけでなくそれ以降の全ての性能
//! 改善コミット（#1013・#1023・#1028・#1043〜#1047・#1078〜#1082 等）を
//! 累積した差分になる。本ファイルの計測値は「HEAD 時点の絶対値」であり
//! #1017 単独の delta ではない（#1017 単独の delta は
//! `crates/backend-metal/tests/command_batching_bench.rs` を参照）。
//!
//! `cfg(target_os = "macos")` ＋ 理由付き `#[ignore]` は既存の
//! `device_param_store_bench.rs` と同じ方針。
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture
//! ```

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig, Tensor};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};

/// `bench-fandhe/src/main.rs` の `BATCH`/`D_IN`/`D_HIDDEN`/`D_OUT` と同値
/// （MNIST 規模の 2 層 MLP）。
const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;

/// `bench-fandhe` の `SEED_X`/`SEED_Y`/`SEED_L1`/`SEED_L2` と同値
/// （`scripts/bench/framework-compare/bench-common/src/lib.rs`）。
const SEED_X: u64 = 0xDA7A_0001;
const SEED_Y: u64 = 0xDA7A_0002;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `bench-fandhe` の `TRAIN_STEPS`/`TRAIN_WARMUP`/`LR` と同値。
const TRAIN_STEPS: usize = 100;
const TRAIN_WARMUP: usize = 20;
const LR: f32 = 0.01;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn mlp_data() -> (Tensor<f32>, Tensor<f32>) {
    let x = Xorshift64Star::new(SEED_X).fill_vec(BATCH * D_IN);
    let y = Xorshift64Star::new(SEED_Y).fill_vec(BATCH * D_OUT);
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

/// `bench-fandhe::run_train`（fresh。ホスト経由 SGD）と同一手順の
/// per-step 所要秒（`TRAIN_WARMUP` を除いた `TRAIN_STEPS -
/// TRAIN_WARMUP` 本）を返す。
fn run_fresh(device: Device) -> Vec<f64> {
    let mut model = build_model();
    let (x_data, y_data) = mlp_data();
    let mut durations = Vec::with_capacity(TRAIN_STEPS);

    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        let tape = fandhe_ai::tape_for(device).unwrap();
        let bound = model.bind(&tape);
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = bound.forward(&tape, &x).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        // host readout（bench-fandhe と同じくループ内の同期点）。
        let _last_loss = loss
            .to_tensor()
            .get(&[])
            .expect("loss は shape [] スカラー");
        let grads = tape.backward(&loss).unwrap();
        let grad_refs = bound.trainable_grads(&grads).unwrap();
        let param_refs = model.trainable_parameters();
        let updated: Vec<Tensor<f32>> = param_refs
            .iter()
            .zip(grad_refs.iter())
            .map(|(param, grad)| {
                let p = param.contiguous().as_slice().unwrap().to_vec();
                let g = grad.contiguous().as_slice().unwrap().to_vec();
                let upd: Vec<f32> = p.iter().zip(g.iter()).map(|(p, g)| p - LR * g).collect();
                Tensor::from_slice(&upd, param.shape()).unwrap()
            })
            .collect();
        model.apply_parameters(updated).unwrap();
        durations.push(start.elapsed().as_secs_f64());
    }

    durations[TRAIN_WARMUP..].to_vec()
}

/// `bench-fandhe::run_train_reuse`（reuse。デバイス常駐 SGD）と同一手順
/// の per-step 所要秒（`TRAIN_WARMUP` を除いた本数）を返す。
fn run_reuse(device: Device) -> Vec<f64> {
    let model = build_model();
    let (x_data, y_data) = mlp_data();

    let init_tape = fandhe_ai::tape_for(device).unwrap();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let _ = init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(LR);
    let mut durations = Vec::with_capacity(TRAIN_STEPS);

    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        let tape = fandhe_ai::tape_for(device).unwrap();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let _last_loss = loss
            .to_tensor()
            .get(&[])
            .expect("loss は shape [] スカラー");
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
        durations.push(start.elapsed().as_secs_f64());
    }

    // 終端同期（`bench-fandhe::run_train_reuse` の終端 `sync_to_host` と
    // 同じ位置づけ。計測窓の外）。
    let final_tape = fandhe_ai::tape_for(device).unwrap();
    let _ = final_tape.sync_device_param_store_to_host(&store).unwrap();

    durations[TRAIN_WARMUP..].to_vec()
}

/// `device` について fresh/reuse の median/Q1/Q3（ミリ秒換算）を計測し
/// 標準出力へ記録する（record only。`.claude/rules/coding-rust.md`
/// 「ベンチは 5 回計測の中央値」— `bench-fandhe` と同じく、1 プロセス内
/// で `TRAIN_STEPS` 回の step を回し、先頭 warmup を除いた残り本数の
/// 中央値・Q1/Q3 を採用する方式〈`scripts/bench/framework-compare/
/// run_all.sh` が `bench-fandhe` バイナリ自体を複数回 re-run しない
/// のと同じ計測単位〉）。
fn bench_fresh_vs_reuse(device: Device, label: &str) {
    // warmup run（tape 初期化コスト等の結線コストを本計測から除く）。
    let _ = run_fresh(device);
    let _ = run_reuse(device);

    let fresh_secs = run_fresh(device);
    let reuse_secs = run_reuse(device);

    let fresh_q = median_q1_q3(&fresh_secs).expect("TRAIN_STEPS - TRAIN_WARMUP 個のサンプル");
    let reuse_q = median_q1_q3(&reuse_secs).expect("TRAIN_STEPS - TRAIN_WARMUP 個のサンプル");

    println!(
        "[mnist_scale_train_reuse_bench:{label}] \
         fresh_median_ms={:.3} (q1={:.3}, q3={:.3}) \
         reuse_median_ms={:.3} (q1={:.3}, q3={:.3}) \
         reuse_vs_fresh_x={:.3} — HEAD 絶対値。0.4.0 基準点との境界差は \
         本ファイル冒頭コメント参照。record only, non-gating",
        fresh_q.median * 1e3,
        fresh_q.q1 * 1e3,
        fresh_q.q3 * 1e3,
        reuse_q.median * 1e3,
        reuse_q.q1 * 1e3,
        reuse_q.q3 * 1e3,
        fresh_q.median / reuse_q.median.max(f64::EPSILON),
    );
}

/// Metal 実機（macOS）での fresh/reuse 計測。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mnist_scale_train_fresh_vs_reuse_metal() {
    bench_fresh_vs_reuse(Device::Metal, "metal");
}
