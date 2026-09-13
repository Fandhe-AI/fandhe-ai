//! `Sequential::predict_resident` の単一同期チェーン化（イシュー #1580・
//! 設計 `docs/inference-chain-single-sync-design.md`）の facade 統合
//! テスト。`crates/backend-metal/tests/linear_forward_device_parity.rs`
//! と同じ構成方針（`cfg(target_os = "macos")` + 理由付き `#[ignore]`）。
//!
//! **公開 API のみで「旧経路」を手動再現する理由**: `Sequential::
//! forward_from_flat_leaves`（旧 per-op 経路の本体）は private メソッド
//! のため facade クレート外の本ファイルからは呼べない。代わりに
//! `DeviceParamStore::linear_forward_with_activation`（`fandhe_ai_autodiff`
//! の公開 API。`fandhe_ai::DeviceParamStore` として facade が再エクスポート
//! 済み）を手動連鎖させ、`Sequential::predict_resident`（新経路。単一
//! 同期チェーン優先＋フォールバック）と突き合わせる。
//!
//! `linear_forward_with_activation` は `&fandhe_ai_autodiff::Tape`（生の
//! 型）を要求するため、facade `Tape`（newtype。内部フィールドは
//! `pub(crate)` でクレート外から取り出せない）とは別に、
//! `fandhe_ai_autodiff::Tape::new_with_ops(Box::new(MetalBackendOps::new()))`
//! で独立の生 `Tape` を構築する（`crates/facade/tests/compat_sequential.rs`
//! 冒頭 doc・`crates/autodiff/src/optim/device_store.rs` の `tape1`/`tape2`
//! 独立構築テストと同型のパターン。同一 `DeviceParamStore` に対し複数の
//! `Tape` インスタンスから `snapshot_resident_params` を呼んでもよい
//! ——`DeviceParamStore::check_device` は `tape.ops().device()` の一致
//! のみを検査するため）。
//!
//! 実行コマンド（Apple Silicon 実機）:
//!
//! ```sh
//! cargo test -p fandhe-ai --test infer_device_chain_metal -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use std::sync::Mutex;
use std::time::Instant;

use bench_harness::median_q1_q3;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, DeviceParamStore, Tensor};
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::Activation;

/// `MetalContext` はプロセス単位のシングルトン（`fandhe_ai_backend_metal::
/// __diagnostic_batch_counters_snapshot` が読む `encode_calls`／
/// `wait_until_completed` カウンタもプロセス全体で共有）であり、本ファイル
/// の 4 テストは同一テストバイナリ内で cargo test の既定並列実行
/// （`--test-threads` 省略時は複数）により同時に走りうる。とくに
/// `predict_resident_device_chain_dispatch_counters_metal`（テスト (c)）は
/// `before`/`after` 2 回のスナップショット差分を厳密なディスパッチ回数
/// として検証するため、その計測区間中に他テスト（(a)/(b)/(d)）由来の
/// Metal dispatch が割り込むと誤検出・flaky 化する（codex-review 指摘）。
/// 本ファイル内の全テストが Metal 実行区間の前後でこのロックを取得する
/// ことで、同一プロセス内では常に直列実行され、この割り込みを構造的に
/// 排除する（`--test-threads=1` の運用依存にしない）。
static METAL_SINGLETON_LOCK: Mutex<()> = Mutex::new(());

const SEED1: u64 = 5001;
const SEED2: u64 = 5002;
const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;

/// `Xorshift64Star` 同等の決定的疑似乱数（`-0.5` シフト。`linear_forward_
/// device_parity.rs::xorshift_fill` と同一アルゴリズム）。
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

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED2)
        .unwrap()
}

fn build_input() -> Tensor<f32> {
    Tensor::new(xorshift_fill(0xaaaa_bbbb, BATCH * D_IN), &[BATCH, D_IN]).unwrap()
}

/// 「旧経路」を公開 API のみで手動再現する: `DeviceParamStore::
/// linear_forward_with_activation` を 2 層分手動連鎖させる（`Sequential::
/// forward_from_flat_leaves` が `Linear -> ReLU -> Linear` に対し行う
/// ものと同一の演算列）。
fn manual_predict_via_public_api(store: &DeviceParamStore, input: &Tensor<f32>) -> Tensor<f32> {
    let raw_tape = fandhe_ai_autodiff::Tape::new_with_ops(Box::new(MetalBackendOps::new()));
    let leaves = store.snapshot_resident_params(&raw_tape).unwrap();
    let input_var = raw_tape.var(input);
    let h1 = store
        .linear_forward_with_activation(
            &raw_tape,
            &input_var,
            &leaves[0],
            Some(&leaves[1]),
            Activation::Relu,
        )
        .unwrap();
    let out = store
        .linear_forward_with_activation(
            &raw_tape,
            &h1,
            &leaves[2],
            Some(&leaves[3]),
            Activation::None,
        )
        .unwrap();
    out.to_tensor()
}

/// (a) `predict_resident`（新経路。単一同期チェーン優先）が旧経路
/// （手動 per-op 連鎖）と bit 完全一致することを検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn predict_resident_device_chain_matches_manual_per_layer_chain_bit_exact_metal() {
    let _metal_singleton_guard = METAL_SINGLETON_LOCK.lock().unwrap();
    let model = build_model();
    let init_tape = fandhe_ai::tape_for(Device::Metal).unwrap();
    let store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let input = build_input();

    let via_new = model.predict_resident(&store, &input).unwrap();
    let via_old = manual_predict_via_public_api(&store, &input);

    assert_eq!(via_new.shape(), via_old.shape());
    let a = via_new.contiguous();
    let b = via_old.contiguous();
    assert_eq!(
        a.as_slice().unwrap(),
        b.as_slice().unwrap(),
        "predict_resident（単一同期チェーン優先）が手動 per-op チェーンと bit 完全一致しない"
    );
}

/// (b) `predict_resident` の出力が run-to-run で bit 同一であることを
/// 確認する（イシュー #1580 事前登録判定規則「run-to-run bit 同一」）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn predict_resident_device_chain_is_run_to_run_bit_identical_metal() {
    let _metal_singleton_guard = METAL_SINGLETON_LOCK.lock().unwrap();
    let model = build_model();
    let init_tape = fandhe_ai::tape_for(Device::Metal).unwrap();
    let store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let input = build_input();

    let first = model.predict_resident(&store, &input).unwrap();
    for i in 0..4 {
        let repeat = model.predict_resident(&store, &input).unwrap();
        assert_eq!(
            first.contiguous().as_slice().unwrap(),
            repeat.contiguous().as_slice().unwrap(),
            "predict_resident の出力が run {i} で run 0 と bit 同一でない"
        );
    }
}

/// (c) 2 層モデルの `predict_resident` 1 回あたりのディスパッチ数を
/// `fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot` で
/// 検証する（イシュー #1580 事前登録判定規則: `encode_calls` 差分 2・
/// `wait_until_completed` 差分 1）。2 層 `Linear -> ReLU -> Linear` は
/// `build_device_chain_steps` により 2 ステップ（1 層目は次層 ReLU と
/// 融合）へ平坦化され、各ステップが `linear_forward_device_tracked`
/// を 1 回 encode するため合計 2 回・チェーン末尾の `download` が
/// `synchronize`（`waitUntilCompleted`）を 1 回だけ発生させる。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn predict_resident_device_chain_dispatch_counters_metal() {
    let _metal_singleton_guard = METAL_SINGLETON_LOCK.lock().unwrap();
    let model = build_model();
    let init_tape = fandhe_ai::tape_for(Device::Metal).unwrap();
    let store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let input = build_input();

    // ウォームアップ（初回のみ発生しうる遅延初期化コストを除く）。
    let _ = model.predict_resident(&store, &input).unwrap();

    let before = fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot()
        .expect("singleton MetalContext must be available on real hardware");
    let _ = model.predict_resident(&store, &input).unwrap();
    let after = fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot()
        .expect("singleton MetalContext must be available on real hardware");

    assert_eq!(
        after.encode_calls - before.encode_calls,
        2,
        "2 層 Linear->ReLU->Linear の predict_resident 1 回は encode_calls を 2 増やすはず"
    );
    assert_eq!(
        after.wait_until_completed - before.wait_until_completed,
        1,
        "predict_resident 1 回はチェーン末尾の download で wait_until_completed を \
         1 回だけ増やすはず（層ごとの同期点が生じていないことの確認）"
    );
}

/// backend レベル before/after 記録用ベンチ（record-only。hard assert
/// なし。`linear_forward_device_parity.rs::linear_forward_device_bench_
/// metal` と同方針）。`Sequential::predict_resident`（新経路）と
/// `manual_predict_via_public_api`（旧経路。層ごとに `linear_forward_
/// with_activation` を呼び直す＝毎回 `snapshot_resident_params` から
/// やり直す per-op 経路の素朴な再現）の per-forward 時間を比較する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない（record only）"]
fn predict_resident_device_chain_bench_metal() {
    let _metal_singleton_guard = METAL_SINGLETON_LOCK.lock().unwrap();
    const WARMUP: usize = 20;
    const ITERS: usize = 20;
    const TRIALS: usize = 5;

    let model = build_model();
    let init_tape = fandhe_ai::tape_for(Device::Metal).unwrap();
    let store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let input = build_input();

    let run_old = || {
        std::hint::black_box(manual_predict_via_public_api(&store, &input));
    };
    let run_new = || {
        std::hint::black_box(model.predict_resident(&store, &input).unwrap());
    };

    for _ in 0..WARMUP {
        run_old();
        run_new();
    }

    let mut old_medians = Vec::with_capacity(TRIALS);
    let mut new_medians = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let mut old_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = Instant::now();
            run_old();
            old_samples.push(t0.elapsed().as_secs_f64());
        }
        old_medians.push(
            median_q1_q3(&old_samples)
                .expect("non-empty samples")
                .median,
        );

        let mut new_samples = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let t0 = Instant::now();
            run_new();
            new_samples.push(t0.elapsed().as_secs_f64());
        }
        new_medians.push(
            median_q1_q3(&new_samples)
                .expect("non-empty samples")
                .median,
        );
    }

    let old_median = median_q1_q3(&old_medians)
        .expect("non-empty samples")
        .median;
    let new_median = median_q1_q3(&new_medians)
        .expect("non-empty samples")
        .median;
    let speedup = old_median / new_median;
    println!(
        "[predict_resident_device_chain_bench:metal] old_median_s={old_median:.6} \
         new_median_s={new_median:.6} speedup_x={speedup:.3}"
    );
}
