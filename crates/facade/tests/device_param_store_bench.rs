//! イシュー #936 受け入れ条件 2（ベンチ非後退確認）: 旧経路
//! （`Sgd::step` + `Sequential::apply_parameters`。毎 step `Linear::bind`
//! が `weight`／`bias` をホストからバックエンドへ再アップロードする）と
//! 新経路（`Sequential::init_device_param_store` + 毎 step
//! `forward_resident` + `Tape::step_device_param_store`。パラメータは
//! 1 回アップロードしたデバイス常駐バッファを使い回す）の per-step 時間を
//! 比較する。
//!
//! **record only（hard assert しない）方針**: `crates/facade/tests/
//! tape_cuda_cache_bench.rs` と同じ方針を踏襲する。GPU クロック挙動・
//! 他プロセス競合等の環境揺らぎをタイミング値の hard assert に持ち込むと
//! flaky 化するため（`.claude/rules/coding-rust.md`「ベンチは 5 回計測の
//! 中央値」）。非後退の判定は `docs/perf/device-resident-update-bench.md`
//! への実測記録で人間が行う。
//!
//! **転送モデルの前提（PR #954 申し送り。設計文書 §3.3）**: 新経路が削減
//! するのは「param の毎 step 再アップロード」のみであり、
//! `register_resident_leaves` は毎 step D2H download を行い、GEMM は内部で
//! H2D upload を行う。よって新経路が旧経路より必ず高速であるとは限らない
//! （とくに CPU バックエンドでは upload/download コスト自体が小さいため、
//! 差が計測ノイズに埋もれうる）。
//!
//! **計測区間**: 1 step 全体（forward + backward + update。旧経路は
//! `bind` の再アップロードを含む）を主計測とし、参考として更新フェーズ
//! 単体（旧: `Sgd::step` + `apply_parameters`、新:
//! `step_device_param_store`）も計測し (b)（毎 step 再アップロードの
//! 削減効果）の要因分離を観察できるようにする。#931 系のタイムアップ
//! 初期化コスト (a) はどちらの経路も同一の `tape_for` 呼び出しコストを
//! 払うため、本ベンチは (a) の差ではなく (b) の差を主眼に置く（設計文書
//! §7「#936 への引き渡し事項」）。
//!
//! **#1023 追補**: `DeviceParamStore` の内部実装をパラメータ横断の単一
//! 連結バッファへ再構成し、更新フェーズ（grad upload・
//! `sgd_step_device_tracked` 起動）をパラメータ数に依らず 1 回／step へ
//! バッチ化した（`crates/autodiff/src/optim/device_store.rs` モジュール
//! 冒頭コメント参照）。本ベンチは呼び出し側 API（`init_device_param_store`／
//! `forward_resident`／`step_device_param_store`）を変更なしで計測する
//! ため、変更後の理論的なディスパッチ回数（更新フェーズ: 2N → 2）を
//! 自動的に反映する。Metal 実機・DGX Spark GB10 実機での再計測は
//! `docs/perf/device-resident-update-bench.md` §6 追補を参照。

use std::time::Instant;

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig, Tensor};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_autodiff::optim::{Sgd, SgdConfig};

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// 5 回計測中央値方針（`.claude/rules/coding-rust.md`）。
const TRIALS: usize = 5;
/// 1 trial あたりの step 数（平均 per-step 時間をこの本数で割って求める）。
const STEPS_PER_TRIAL: usize = 20;
const LR: f32 = 0.02;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
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

/// 旧経路（`bind` + forward + backward + `Sgd::step` + `apply_parameters`）
/// を `steps` 回回し、`(1 step 全体の平均秒, update フェーズ単体の平均秒)`
/// を返す。
fn run_legacy_path(device: Device, steps: usize) -> (f64, f64) {
    let mut model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut sgd = Sgd::new(SgdConfig::new(LR)).unwrap();

    let mut total_secs = 0.0f64;
    let mut update_secs = 0.0f64;

    for _ in 0..steps {
        let t_step = Instant::now();
        // `t_update` は `sgd.step` 直前に確定するが、経過時間の加算は
        // `model.apply_parameters` の後で行う。update フェーズの定義
        // （本ファイル冒頭コメント・#955 レビュー指摘）は「`Sgd::step` +
        // `apply_parameters`」であり、resident 側の `step_device_param_store`
        // （更新処理全体を計測）と対称な区間にするため。
        let (updated, t_update) = {
            let tape = fandhe_ai::tape_for(device).unwrap();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);
            let pred = bound.forward(&tape, &x).unwrap();
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();

            let t_update = Instant::now();
            let param_refs = model.trainable_parameters();
            let updated = sgd.step(&param_refs, &grad_refs).unwrap();
            (updated, t_update)
        };
        model.apply_parameters(updated).unwrap();
        update_secs += t_update.elapsed().as_secs_f64();
        total_secs += t_step.elapsed().as_secs_f64();
    }

    (total_secs / steps as f64, update_secs / steps as f64)
}

/// 新経路（`init_device_param_store` + 毎 step `forward_resident` +
/// `step_device_param_store`）を `steps` 回回し、`(1 step 全体の平均秒,
/// update フェーズ単体の平均秒)` を返す。
fn run_resident_path(device: Device, steps: usize) -> (f64, f64) {
    let model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    let init_tape = fandhe_ai::tape_for(device).unwrap();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(LR);

    let mut total_secs = 0.0f64;
    let mut update_secs = 0.0f64;

    for _ in 0..steps {
        let t_step = Instant::now();
        let tape = fandhe_ai::tape_for(device).unwrap();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let grads = tape.backward_device_param_store(&loss, &store).unwrap();

        let t_update = Instant::now();
        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();
        update_secs += t_update.elapsed().as_secs_f64();
        // イシュー #1013（カーネル起動直後の都度 `synchronize()` 除去）
        // 後は、この step のループ本体には明示的な完了待ちが一切残らない
        // （forward/backward/update いずれも非同期投入のみ）。`Instant`
        // による経過時間計測がホスト側のディスパッチ時間（enqueue コスト）
        // しか捉えず GPU 実行完了を反映しない見かけ上の高速化に化けるのを
        // 防ぐため、`sync_to_host`（readback 境界。`DeviceParamStore`
        // の実処理は `MemoryOps::download` 経由で維持される同期点）を
        // 明示的に呼び、旧経路が暗黙に持っていた「launch 直後の同期」と
        // 同等の完了保証をこの計測境界へ復元する。before/after 双方の
        // 計測を同一の本ファイルで取ることで比較可能性を保つ
        // （before は #1013 変更前のソースに対して本行を含めて計測する）。
        let _ = tape.sync_device_param_store_to_host(&store).unwrap();
        total_secs += t_step.elapsed().as_secs_f64();
    }

    (total_secs / steps as f64, update_secs / steps as f64)
}

/// `device` について旧経路 vs 新経路の per-step 中央値/Q1/Q3 を計測し
/// 標準出力へ記録する（record only。`label` はログ識別用）。
fn bench_legacy_vs_resident(device: Device, label: &str) {
    // warmup: 初回呼び出しの結線コスト（#931 系 tape 初期化コスト (a)）を
    // 両経路の本計測から除く。
    let _ = run_legacy_path(device, 1);
    let _ = run_resident_path(device, 1);

    let mut legacy_total = Vec::with_capacity(TRIALS);
    let mut legacy_update = Vec::with_capacity(TRIALS);
    let mut resident_total = Vec::with_capacity(TRIALS);
    let mut resident_update = Vec::with_capacity(TRIALS);

    for _ in 0..TRIALS {
        let (t, u) = run_legacy_path(device, STEPS_PER_TRIAL);
        legacy_total.push(t);
        legacy_update.push(u);
        let (t, u) = run_resident_path(device, STEPS_PER_TRIAL);
        resident_total.push(t);
        resident_update.push(u);
    }

    let legacy_total_q = median_q1_q3(&legacy_total).expect("TRIALS 個の non-NaN サンプル");
    let legacy_update_q = median_q1_q3(&legacy_update).expect("TRIALS 個の non-NaN サンプル");
    let resident_total_q = median_q1_q3(&resident_total).expect("TRIALS 個の non-NaN サンプル");
    let resident_update_q = median_q1_q3(&resident_update).expect("TRIALS 個の non-NaN サンプル");

    println!(
        "[device_param_store_bench:{label}] \
         legacy_total_median_s={:.9} (q1={:.9}, q3={:.9}) \
         resident_total_median_s={:.9} (q1={:.9}, q3={:.9}) \
         total_speedup_x={:.3} resident_faster={} | \
         legacy_update_median_s={:.9} resident_update_median_s={:.9} \
         update_speedup_x={:.3} \
         — record only, non-gating（本ファイル冒頭コメント参照）",
        legacy_total_q.median,
        legacy_total_q.q1,
        legacy_total_q.q3,
        resident_total_q.median,
        resident_total_q.q1,
        resident_total_q.q3,
        legacy_total_q.median / resident_total_q.median.max(f64::EPSILON),
        resident_total_q.median < legacy_total_q.median,
        legacy_update_q.median,
        resident_update_q.median,
        legacy_update_q.median / resident_update_q.median.max(f64::EPSILON),
    );
}

/// CPU バックエンドは常時利用可能なため通常テストとして実行する。
#[test]
fn legacy_vs_resident_per_step_cpu() {
    bench_legacy_vs_resident(Device::Cpu, "cpu");
}

/// Metal 実機（macOS）での旧経路 vs 新経路比較。
///
/// ```sh
/// cargo test -p fandhe-ai --release --test device_param_store_bench -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn legacy_vs_resident_per_step_metal() {
    bench_legacy_vs_resident(Device::Metal, "metal");
}

/// CUDA 実機（DGX Spark GB10 等）での旧経路 vs 新経路比較。
///
/// ```sh
/// cargo test -p fandhe-ai --release --test device_param_store_bench -- --ignored --nocapture
/// ```
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn legacy_vs_resident_per_step_cuda() {
    bench_legacy_vs_resident(Device::Cuda(0), "cuda");
}
