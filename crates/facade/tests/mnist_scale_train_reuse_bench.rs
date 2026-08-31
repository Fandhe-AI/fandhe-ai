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
//! **計測境界の一致（codex-review PR #1097 P1 是正）**: `bench-fandhe::
//! run_train`／`run_train_reuse` はプロセス単位で「`TRAIN_STEPS` 回の
//! step を 1 回だけ実行し、先頭 `TRAIN_WARMUP` 本を捨てた残り 80 本の
//! median/Q1/Q3 を取る」プロトコルであり、この 1 回の実行の外側に
//! 追加の全量 warmup 実行を挟まない。初版は比較対象にない
//! `run_fresh(device)`／`run_reuse(device)` の破棄呼び出しを本計測の
//! 前段に追加していたため、比較元と計測境界が異なっていた。本版は
//! その追加 warmup を削除し、`TRIALS`（5。`.claude/rules/coding-rust.md`
//! 「ベンチは 5 回計測の中央値」）回、比較元と同一の「1 実行 = 100 step
//! ・先頭 20 step 破棄」プロトコルを独立に繰り返し、各実行の median を
//! `TRIALS` 個集めた上でさらにその中央値を採用する（各実行内の先頭
//! 20 step が bench-fandhe と同じ役割の warmup を兼ねる。実行間で状態を
//! 共有しないため、実行 1 回目の warmup を実行 2 回目以降が使い回す
//! こともない）。
//!
//! **数値検証（codex-review PR #1097 P2 是正）**: `bench-fandhe`
//! （`run_train`／`run_train_reuse`）と同じく、最終 step の loss が
//! 有限であることを検証し、reuse では終端 `sync_to_host` が返す
//! パラメータの個数が `trainable_parameters().len()` と一致しかつ
//! 全要素が有限であることを検証する。いずれかが破れた場合は計測結果を
//! 採用せず `assert!`／`panic!` で即座に失敗させる（A08:
//! 壊れた学習結果を性能値として残さない。`bench-fandhe::run_train_reuse`
//! の `MEASURE_ERROR` 相当の判断を、`Result` を返さないテスト関数の
//! 形へ落とし込んだもの）。
//!
//! `#![cfg(target_os = "macos")]`（ファイル全体をゲート）は
//! `crates/backend-metal/tests/command_batching_bench.rs` と同じ方針
//! （非 macOS ビルドでの dead_code clippy 警告を避けるため。codex-review
//! PR #1097 clippy 指摘対応）。理由付き `#[ignore]` は既存の
//! `device_param_store_bench.rs` と同じ方針。
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test mnist_scale_train_reuse_bench -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

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

/// 5 回計測中央値方針（`.claude/rules/coding-rust.md`）。比較元
/// `bench-fandhe` はプロセスあたり 1 回のみ実行するため、本ファイルは
/// 比較元と同一プロトコルの実行を `TRIALS` 回独立に繰り返し、各実行の
/// median をさらに中央値化する（冒頭コメント参照）。
const TRIALS: usize = 5;

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
/// TRAIN_WARMUP` 本）を返す。`bench-fandhe` と同じく最終 step の loss の
/// 有限性を検証する（codex-review PR #1097 P2）。
fn run_fresh(device: Device) -> Vec<f64> {
    let mut model = build_model();
    let (x_data, y_data) = mlp_data();
    let mut durations = Vec::with_capacity(TRAIN_STEPS);
    let mut last_loss = 0.0f32;

    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        let tape = fandhe_ai::tape_for(device).unwrap();
        let bound = model.bind(&tape);
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = bound.forward(&tape, &x).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        // host readout（bench-fandhe と同じくループ内の同期点）。
        last_loss = loss
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

    assert!(
        last_loss.is_finite(),
        "MEASURE_ERROR: final loss not finite: {last_loss}"
    );

    durations[TRAIN_WARMUP..].to_vec()
}

/// `bench-fandhe::run_train_reuse`（reuse。デバイス常駐 SGD）と同一手順
/// の per-step 所要秒（`TRAIN_WARMUP` を除いた本数）を返す。`bench-fandhe`
/// と同じく最終 step の loss の有限性、終端同期後のパラメータ個数・
/// 全要素有限性を検証する（codex-review PR #1097 P2）。
fn run_reuse(device: Device) -> Vec<f64> {
    let model = build_model();
    let (x_data, y_data) = mlp_data();

    let init_tape = fandhe_ai::tape_for(device).unwrap();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let _ = init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(LR);
    let mut durations = Vec::with_capacity(TRAIN_STEPS);
    let mut last_loss = 0.0f32;

    for _ in 0..TRAIN_STEPS {
        let start = Instant::now();
        let tape = fandhe_ai::tape_for(device).unwrap();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        last_loss = loss
            .to_tensor()
            .get(&[])
            .expect("loss は shape [] スカラー");
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
        durations.push(start.elapsed().as_secs_f64());
    }

    assert!(
        last_loss.is_finite(),
        "MEASURE_ERROR: final loss not finite: {last_loss}"
    );

    // 終端同期（`bench-fandhe::run_train_reuse` の終端 `sync_to_host` と
    // 同じ位置づけ。計測窓の外）。個数・全要素有限性を検証する
    // （`bench-fandhe::run_train_reuse` の `MEASURE_ERROR` 相当）。
    let final_tape = fandhe_ai::tape_for(device).unwrap();
    let synced = final_tape.sync_device_param_store_to_host(&store).unwrap();
    let expected_len = model.trainable_parameters().len();
    assert_eq!(
        synced.len(),
        expected_len,
        "MEASURE_ERROR: sync_device_param_store_to_host returned {} tensors, expected {expected_len}",
        synced.len()
    );
    for t in &synced {
        let slice = t
            .contiguous()
            .as_slice()
            .expect("synced param as_slice() returned None")
            .to_vec();
        assert!(
            slice.iter().all(|v| v.is_finite()),
            "MEASURE_ERROR: synced parameter contains non-finite element"
        );
    }

    durations[TRAIN_WARMUP..].to_vec()
}

/// `device` について fresh/reuse の median/Q1/Q3（ミリ秒換算）を計測し
/// 標準出力へ記録する（record only。冒頭コメント「計測境界の一致」参照。
/// 比較元 `bench-fandhe` と同一境界の実行を `TRIALS` 回繰り返し、各実行の
/// median をさらに中央値化する）。
fn bench_fresh_vs_reuse(device: Device, label: &str) {
    let mut fresh_trial_medians = Vec::with_capacity(TRIALS);
    let mut reuse_trial_medians = Vec::with_capacity(TRIALS);

    for _ in 0..TRIALS {
        let fresh_secs = run_fresh(device);
        let fresh_trial_q =
            median_q1_q3(&fresh_secs).expect("TRAIN_STEPS - TRAIN_WARMUP 個のサンプル");
        fresh_trial_medians.push(fresh_trial_q.median);

        let reuse_secs = run_reuse(device);
        let reuse_trial_q =
            median_q1_q3(&reuse_secs).expect("TRAIN_STEPS - TRAIN_WARMUP 個のサンプル");
        reuse_trial_medians.push(reuse_trial_q.median);
    }

    let fresh_q = median_q1_q3(&fresh_trial_medians).expect("TRIALS 個の non-NaN サンプル");
    let reuse_q = median_q1_q3(&reuse_trial_medians).expect("TRIALS 個の non-NaN サンプル");

    println!(
        "[mnist_scale_train_reuse_bench:{label}] trials={TRIALS} \
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
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mnist_scale_train_fresh_vs_reuse_metal() {
    bench_fresh_vs_reuse(Device::Metal, "metal");
}
