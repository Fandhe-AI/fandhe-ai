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
//! **直列化ガードで安全化済み（イシュー #1550。旧
//! `--test-threads=1` 必須運用からの変更）**:
//! [`mnist_scale_train_reuse_metal_batch_counters`] はプロセスワイド
//! singleton `MetalContext` の診断カウンタ
//! （`__diagnostic_batch_counters_snapshot`）を読む。既定の並列実行
//! （`cargo test` は `--test-threads` 未指定だとスレッドプールで
//! `#[test]` 関数を同時実行する）下では、同一バイナリ内の他テスト
//! （`mnist_scale_train_fresh_vs_reuse_metal`）が同じ singleton 経由で
//! `encode()` を呼ぶため、カウンタの before/after 差分が他テストの
//! dispatch で汚染されうる。両テストの冒頭で
//! `serialize_diagnostic_counter_tests()`（プロセス内 `static Mutex`
//! ガード）を取得して直列化するため、`--test-threads=1` なしの既定
//! 並列実行でも安全（`--test-threads=1` を付けてもロックにより無害）。
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

/// 本ファイル内の 2 つの `#[test]`（[`mnist_scale_train_fresh_vs_reuse_metal`]・
/// [`mnist_scale_train_reuse_metal_batch_counters`]）を直列化するロック
/// （イシュー #1550）。両テストともプロセスワイド singleton
/// `MetalContext` 経由で `encode()` を呼ぶため、`cargo test` の既定
/// 並列実行下では互いの dispatch がカウンタへ混入し、
/// `mnist_scale_train_reuse_metal_batch_counters` の before/after 差分が
/// 意図と無関係な理由で fail/pass しうる（旧「`--test-threads=1` 必須」
/// 運用の代替。`crates/backend-metal/tests/gemm_splitk_auto_wiring.rs::
/// RuntimeFlagGuard` と同型: `static` 内 `Mutex` を関数内に置くことで
/// 同一アドレスをプロセス内で共有し、lock poisoning は
/// `unwrap_or_else(|e| e.into_inner())` で握り潰して継続する）。
fn serialize_diagnostic_counter_tests() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

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
    let _guard = serialize_diagnostic_counter_tests();
    bench_fresh_vs_reuse(Device::Metal, "metal");
}

/// イシュー #1099 の受入条件 2: プール再利用時ゼロ埋めの無条件
/// `synchronize()` 除去（`crates/backend-metal/src/pool.rs::
/// MetalAllocator::alloc_inner`）により、steady-state の MLP reuse 学習
/// 1 step でコマンドバッファ生成数・`waitUntilCompleted()` 呼び出し数が
/// `docs/backend-metal-command-batching-design.md` §4.2 の before 実測
/// （9 / 9 / 9）から #1099 適用直後は **9 / 8 / 8** へ減ることを、
/// プロセスワイド singleton `MetalContext`（`ops::MetalBackendOps` の
/// 全演算メソッドが経由する唯一のコンテキスト）の診断カウンタ
/// （`fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot`。
/// `#[doc(hidden)]` のテスト・診断専用 API）差分で確認する。
///
/// **#1550 追記**: #1223（`docs/perf/train-backward-gemm-wiring.md`）で
/// `Op::LinearResident` の VJP（`d_weight`）が host CPU `eval::matmul`
/// から `BackendOps::gemm_fp32_strict` 経由の Metal GPU dispatch へ
/// 切り替わったことにより、1 step あたり L1・L2 各層の `d_weight` 計算
/// （2 回）が新たに Metal GPU 上で実行されるようになった（当時の Metal は
/// `fill_resident_weight_grad` 未実装〈#1212 で CUDA/Metal 双方とも
/// スコープ外に据え置き〉のため、計算結果はホストへ `download` される
/// 必要があり、この `download` が同期点となって現在の open バッチを
/// 都度終端していた）。この結果、steady-state 1 step のカウンタは
/// 一時的に **11 / 10 / 10**（encode / command_buffer / wait）へ変化した
/// （既存のバッチ化経路〈forward・SGD〉が分割されたのではなく、新規に
/// 追加された GPU dispatch 2 件がそれぞれ独立した同期境界を伴うため）。
///
/// **#1555 追記**: `MetalBackendOps::gemm_fp32_strict_into`（`BackendOps::
/// gemm_fp32_strict_into` の Metal 実装。`docs/device-resident-
/// update-design.md` 追補）を実装したことで、L1・L2 の `d_weight` は
/// `DeviceParamStore::fill_resident_weight_grad` 経由で resident 化され、
/// `MetalGemm::encode_strided_bias_act_prepared_with_c_offset`
/// （encode-only。`gemm_fp32_strict_into` doc「同期の回収」参照）で grad
/// staging バッファへ直接書き込まれるようになった——**個別の
/// `download` は発生しない**。ただし `step()` の `any_resident == true`
/// 分岐（`crates/autodiff/src/optim/device_store.rs`）は resident 経由で
/// ない勾配（本モデルの bias）を `MemoryOps::upload_into`
/// （`memory.rs::MetalMemory::upload_into`。同 #1555 で新規実装）で同じ
/// staging バッファへ書き込む必要があり、この `upload_into` は
/// `zero_fill` と同じ理由（GPU 側の未完了書き込みとの競合回避）で
/// **書き込み前に 1 回 `synchronize()` する**。L1・L2 双方の `d_weight`
/// encode が既にバッチへ積まれた後、bias 分の最初の `upload_into` 呼び
/// 出しがこの 1 回の同期でバッチをまとめて終端する（2 回目以降の
/// `upload_into` は既に空になったバッチに対する no-op 相当の
/// `synchronize()` のため追加の commit/wait を発生させない）。この結果、
/// steady-state 1 step のカウンタは **11 / 9 / 9**
/// （encode / command_buffer / wait）——#1223 直後の 11/10/10 から
/// command_buffer・wait のみ 1 ずつ減り、#1099 直後の 9/8/8 へは戻らない
/// （d_weight 計算自体の GPU dispatch 2 件〈encode_delta の +2〉は
/// 引き続き残るが、これらはもはや個別の同期境界を持たず、bias 用
/// `upload_into` の同期 1 回へ集約される）。`d_input` の GEMM（`Op::
/// LinearResident` の VJP のもう一方）は引き続き `BackendOps::gemm`
/// （`gemm_strided_nt_tn` → `dispatch_strided_bias_act_prepared` → 内部
/// `ctx.synchronize()` → `download`）経由のままであり、resident 化の
/// 対象外（回収は部分的）。
///
/// **#1563 追記**: `crates/autodiff/src/grad.rs` の `Op::LinearResident`
/// VJP で、encode-only の `fill_resident_weight_grad`（d_weight）呼び
/// 出しを、同期点を持つ `gemm_resident_lhs`（d_input）呼び出しより
/// **前**へ移した（両者は独立計算のため出力は bit 同一。
/// `docs/backend-metal-command-batching-design.md` §7.4）。この結果、
/// 各層の d_weight の GPU コマンドが同じ層の d_input の同期点へ合流
/// するようになり、L1 の d_weight（従来は窓終了時点で未 commit のまま
/// 残り、bias 分の `upload_into` の同期 1 回が肩代わりしていた）も
/// L1 の d_input の同期で完了するようになる。backward 終了時点で open
/// バッチが残らないため、`upload_into` の防御的 `synchronize()` は
/// `committed` が空で `waitUntilCompleted` を呼ばない no-op になる
/// （`crates/backend-metal/src/context.rs::synchronize_observed` の
/// 契約）。この結果、steady-state 1 step のカウンタは **11 / 8 / 8**
/// （encode / command_buffer / wait）——encode 総数は変わらず、
/// command_buffer・wait のみ #1555 の 9 からさらに 1 ずつ減る（d_input
/// 自身の同期境界 2 件〈L1・L2 各層〉は引き続き残る。回収は部分的。
/// `docs/backend-metal-command-batching-design.md` §7.4「回収しない
/// 結論」参照）。
///
/// **#1566 追記（是正版。round1 の記述は根拠のない推測を含んでいた
/// ため #1665 取り込み時に是正）**: bias 勾配（本モデルでは L1・L2
/// 双方の bias）を `ops::MetalBackendOps::gemm_fp32_strict_into_with_
/// bias_reduce_tracked`（weight と同一 `ctx.encode` 呼び出し内で
/// NT/TN の場合は encode-only、NN/TT・分類不能形状はホスト経由で書く。
/// トレイト doc 参照）へ結線した。**round1 は「L2（NT/TN）の bias は
/// encode-only に折り込まれる一方 L1（NN・分類不能）の bias は
/// upload_into を要する」と記述していたが、これは根拠のない誤りだった
/// **——`x_t = transpose2d(x_val)` は L1・L2 いずれも常に転置 view
/// （`x_val` が contiguous な限り）・`g`（upstream。L1 は ReLU マスク
/// 後、L2 は `MseLoss` の VJP 出力）も両層とも contiguous であり、
/// `layout::classify_2d` によるレイアウト判定は L1・L2 で対称（どちらも
/// NT パターン）になる。実際、上記「#1563 追記」自身の backward 区間
/// イベント列導出（L2 の `d_weight`・L1 の `d_weight` いずれも
/// 「encode-only」と扱う）がこの対称性を裏付けている。
///
/// この結果、L1・L2 とも weight の resident 書き込みと**同一の**
/// encode-only ディスパッチへ bias 縮約が折り込まれ（`gemm::
/// MetalGemm::encode_weight_and_bias_grad_with_offsets`）、両層とも
/// もはや個別の `MemoryOps::upload_into`（bias 用）を経由しない。
/// #1563 適用後（上記）の時点で、bias 用 `upload_into`（当時は
/// `step()` 内のホスト `Gradients` 経由。L1・L2 各 1 回、計 2 回）は
/// 既に**完全な no-op**（`committed` が空で `waitUntilCompleted` を
/// 呼ばない）になっていた（backward 終了時点で open バッチが残らない
/// ため）。本イシュー（#1566）はこの 2 回の no-op な `upload_into`
/// 呼び出し自体を発生させなくするが、**除去される呼び出しの cb・wait
/// 寄与が元々ゼロだったため**、steady-state 1 step のカウンタは
/// #1563 適用後の **11 / 8 / 8**（encode / command_buffer / wait）
/// から**変化しない**という結論になる（encode 総数も、bias 縮約自体は
/// 既存の weight encode-only ディスパッチ内に折り込まれるだけで新規の
/// `ctx.encode` 呼び出しを増やさないため不変）。
///
/// この結論は `docs/backend-metal-command-batching-design.md`
/// §7.4（#1563 の cb/wait 削減根拠）・§10.7〜§10.9（#1566／#1659 の
/// bias resident 化・数値方式統一の実装記録）から論理的に導出した
/// 机上の結論であり、Mac 実機セッションでの実測確認（`cargo test
/// -p fandhe-ai --release --test mnist_scale_train_reuse_bench
/// mnist_scale_train_reuse_metal_batch_counters -- --ignored
/// --nocapture`）はまだ行っていない。実測で不一致が判明した場合は
/// 本テストのアサーション値・本コメントを実測値に合わせて更新する
/// こと（事前登録規則の事後緩和ではなく、机上導出の誤りを実測で
/// 訂正する通常のフロー）。
///
/// warmup（`WARMUP` step）で MSL パイプライン初回コンパイル・プールの
/// フリーリスト充足を steady-state 化してから、その次の 1 step だけを
/// 計測窓に取る（`run_fresh`／`run_reuse` と同じモデル形状・シードだが、
/// 本テストはカウンタ差分のみを見るため `durations` は測らない）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mnist_scale_train_reuse_metal_batch_counters() {
    let _guard = serialize_diagnostic_counter_tests();
    let device = Device::Metal;
    let model = build_model();
    let (x_data, y_data) = mlp_data();

    let init_tape = fandhe_ai::tape_for(device).unwrap();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let _ = init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(LR);

    let run_step = |store: &mut fandhe_ai::DeviceParamStore| {
        let tape = fandhe_ai::tape_for(device).unwrap();
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model.forward_resident(&tape, &x, store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let last_loss = loss
            .to_tensor()
            .get(&[])
            .expect("loss は shape [] スカラー");
        assert!(
            last_loss.is_finite(),
            "MEASURE_ERROR: step loss not finite: {last_loss}"
        );
        let grads = tape.backward_device_param_store(&loss, store).unwrap();
        tape.step_device_param_store(store, &grads, &config)
            .unwrap();
    };

    // steady-state 到達（`TRAIN_WARMUP`＝20 と同じ考え方。#1017／#1099
    // いずれもプールのフリーリスト充足状態が前提のため、初回数 step は
    // フレッシュ確保が混じりカウンタが安定しない）。
    const WARMUP: usize = TRAIN_WARMUP;
    for _ in 0..WARMUP {
        run_step(&mut store);
    }

    let before = fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot()
        .expect("singleton MetalContext は実機で必ず取得できるはず");
    run_step(&mut store);
    let after = fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot()
        .expect("singleton MetalContext は実機で必ず取得できるはず");

    let encode_delta = after.encode_calls - before.encode_calls;
    let command_buffer_delta = after.command_buffers - before.command_buffers;
    let wait_delta = after.wait_until_completed - before.wait_until_completed;

    println!(
        "[mnist_scale_train_reuse_metal_batch_counters] steady-state 1 step: \
         encode_delta={encode_delta} command_buffer_delta={command_buffer_delta} \
         wait_delta={wait_delta} — before（#1099 適用前）は 9/9/9、#1099 適用直後は \
         9/8/8、#1223（VJP の Metal GPU dispatch 化）適用直後は 11/10/10、#1555 \
         （gemm_fp32_strict_into の Metal 実装）適用後は 11/9/9、#1563 \
         （d_weight encode を d_input の同期点より前へ移し層内で合流）適用後は \
         11/8/8、#1566（bias 勾配も同一 encode-only ディスパッチへ統合。 \
         机上導出では #1563 時点で bias 用 upload_into が既に no-op だった \
         ため 11/8/8 から不変と結論。本ファイル冒頭 doc comment「#1566 \
         追記（是正版）」参照）適用後も 11/8/8（見込み・Mac 実機未確認） \
         （`docs/backend-metal-command-batching-design.md` §4.2・§7「#1550」\
         「#1555」「#1563」節・§10.7〜§10.9「#1566」節）"
    );

    // #1563: `crates/autodiff/src/grad.rs` の `Op::LinearResident` VJP で
    // encode-only の d_weight（`fill_resident_weight_grad`）を、同期点を
    // 持つ d_input（`gemm_resident_lhs`）より前へ移した。両者は独立計算
    // のため出力は bit 同一のまま、同じ層の d_weight コマンドが d_input
    // の同期点へ合流するようになり、backward 終了時点で open バッチが
    // 残らなくなる。この結果 bias 分の `upload_into` の防御的
    // `synchronize()` が待つ対象を失い（`committed` が空で
    // `waitUntilCompleted` を呼ばない no-op）、command_buffer_delta・
    // wait_delta のみ #1555 の 9 からさらに 1 ずつ減って 8 になる
    // （encode 総数は 11 のまま不変。本ファイル冒頭 doc comment
    // 「#1563 追記」参照）。
    //
    // #1566（bias 勾配のデバイス常駐化。本ファイル冒頭 doc comment
    // 「#1566 追記（是正版）」参照）: L1・L2 とも bias 縮約が weight と
    // 同一の encode-only ディスパッチへ折り込まれ、bias 用
    // `upload_into` 呼び出し自体（計 2 回）が発生しなくなる。ただし
    // これらの呼び出しは #1563 適用後の時点で既に完全な no-op
    // （`committed` が空・cb 0・wait 0 の寄与）だったため、それらを
    // 除去しても encode/cb/wait のいずれも変化しない——11/8/8 が
    // #1566 適用後も不変という机上結論になる（`docs/backend-metal-
    // command-batching-design.md` §10.7〜§10.9）。
    assert_eq!(
        encode_delta, 11,
        "encode() 呼び出し総数（= dispatch 総数）は #1223（Op::LinearResident \
         の VJP が Metal GPU dispatch を追加）適用後から #1555・#1563・#1566 \
         適用後も変わらず 11 のはず（d_weight 計算 2 回の GPU dispatch 自体は \
         残り、#1566 の bias 縮約は既存ディスパッチへ折り込まれるだけで \
         新規 encode を増やさないため）"
    );
    assert_eq!(
        command_buffer_delta, 8,
        "コマンドバッファ生成数は #1563（d_weight encode を d_input の同期点より \
         前へ移し層内で合流。upload_into の防御的同期が no-op 化）適用後は \
         9→8 のはず（d_input 自身の同期境界 2 件〈L1・L2〉のみが残る）。#1566 \
         は既に no-op だった bias 用 upload_into を除去するのみのため \
         8 から不変のはず（机上導出。Mac 実機未確認）"
    );
    assert_eq!(
        wait_delta, 8,
        "waitUntilCompleted() 呼び出し数は #1563 適用後は 9→8 のはず \
         （command_buffer_delta と同じ理由）。#1566 適用後も同じ理由で \
         8 から不変のはず（机上導出。Mac 実機未確認）"
    );
}

/// `backward_device_param_store` 区間**限定**でカウンタ差分・壁時間を
/// 計測する診断テスト（イシュー #1562。`docs/backend-metal-command-
/// batching-design.md` §7.3）。既存
/// [`mnist_scale_train_reuse_metal_batch_counters`] は 1 step 全体
/// （forward + backward + SGD update）のカウンタ差分（11/9/9）を見るが、
/// 本テストは `d_input`（`Op::LinearResident` の VJP のうち resident 化
/// されていない側。§7.2「#1555」追記・冒頭 doc comment 参照）が
/// backward フェーズ単独にどれだけの同期境界・壁時間を占めるかを
/// 切り分けるために `tape.backward_device_param_store` の呼び出し前後
/// **だけ**でカウンタ・`Instant` 計測を取る。
///
/// # 事前登録仮説（イシュー #1562 codex-review 是正後・#1563 で更新）
///
/// 当初仮説は backward 区間を「L1・L2 の `d_weight`〈encode-only・
/// #1555〉+ `d_input`〈`gemm_resident_lhs` 経由・個別 `download` あり〉
/// の計 4 GPU dispatch」とだけ捉えていたが、backward の最初の VJP は
/// `Op::MseLoss`（`grad.rs`）であり、その Metal 実装
/// `ops.mse_loss_backward` → `run_mse_backward_f32`
/// （`crates/backend-metal/src/mse.rs`）は**それ自身の
/// `ctx.dispatch_sync`**（encode + 即時 `synchronize`）を持つ。
/// この分の GPU dispatch を見落としていた（Bugbot 指摘）。#1562 時点
/// （#1563 適用前）はこれに基づき encode_delta=5・command_buffer_delta=4
/// ・wait_delta=3 という訂正仮説を記録した（履歴として残す。以下は
/// #1563 適用後の更新版）。
///
/// **#1563 更新**: `crates/autodiff/src/grad.rs` の `Op::LinearResident`
/// VJP で、encode-only の d_weight（`fill_resident_weight_grad`）呼び
/// 出しを、同期点を持つ d_input（`gemm_resident_lhs`）呼び出しより前へ
/// 移した（両者は独立計算のため出力は bit 同一）。backward の VJP
/// 評価順（ノード生成の逆順: `MseLoss` → `L2 LinearResident` → `L1
/// LinearResident`）と `context.rs::encode`／`synchronize`
/// （`slots.open.is_none()` の時のみ新規コマンドバッファを生成し
/// `diag_command_buffers` を加算。`waitUntilCompleted` ごとに
/// `diag_wait_until_completed` を加算）の契約から、次のイベント列を
/// 導出する（`before` snapshot 直前の `loss.to_tensor().get(&[])` が
/// forward 側のバッチを既に flush・wait 済みのため、backward 開始時点
/// で `slots.open == None` を前提にできる）:
///
/// 1. `MseLoss` VJP の `dispatch_sync`: encode #1（新規 cb #1）→
///    synchronize（wait #1）
/// 2. L2 `d_weight`（encode-only。#1563 により d_input より先に呼ばれる）:
///    encode #2（新規 cb #2。同期しないため開いたまま残る）
/// 3. L2 `d_input`（`gemm_resident_lhs`）: encode #3（cb #2 が開いた
///    ままのため新規 cb を開かず同じバッチへ追加）→ synchronize
///    （wait #2。cb #2 を commit・待機——L2 の `d_weight` と `d_input`
///    が同一バッチに合流する）
/// 4. L1 `d_weight`（encode-only）: encode #4（新規 cb #3。同期しない
///    ため開いたまま残る）
/// 5. L1 `d_input`: encode #5（cb #3 が開いたままのため新規 cb を
///    開かず同じバッチへ追加）→ synchronize（wait #3。cb #3 を
///    commit・待機——L1 の `d_weight` と `d_input` が同一バッチに
///    合流する）
///
/// これに基づく更新仮説:
///
/// - `encode_delta = 5`（上記 #1〜#5。#1562 時点から不変）
/// - `command_buffer_delta = 3`（cb #1〜#3 の生成。#1562 時点の 4 から
///   1 減る——L1 の `d_weight` がもはや独立した cb を残さず L1 の
///   `d_input` の cb へ合流するため）
/// - `wait_delta = 3`（wait #1〜#3。窓終了時点で open バッチが残らない
///   ため、#1562 時点で「窓外」だった cb #4 の wait 相当も本窓内で
///   完了する）
///
/// この更新仮説は「backward 中の `materialize_fallible`（`pred_val`／
/// `x_val`／ReLU マスク用 `out_value`）がいずれも forward 時点で
/// キャッシュ済みの値を返し、新規デバイス同期を伴わない」という
/// 机上の前提に基づく未検証の仮説であり、実機実測での確認は本イシュー
/// の実測記入欄（`docs/backend-metal-command-batching-design.md`
/// §7.4）で行う。仮説との不一致でも assert では止めず（本テストは
/// 記録専用・non-gating）乖離をログへ残す。`step_device_param_store`
/// （SGD update）分のカウンタ・時間はこの計測窓の外（bias の
/// `upload_into` 同期はここに含まれない。冒頭 doc comment
/// 「#1564 のスコープ」参照）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn mnist_scale_train_reuse_metal_backward_dinput_phase() {
    let _guard = serialize_diagnostic_counter_tests();
    let device = Device::Metal;
    let model = build_model();
    let (x_data, y_data) = mlp_data();

    let init_tape = fandhe_ai::tape_for(device).unwrap();
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    let _ = init_tape.sync_device_param_store_to_host(&store).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(LR);

    // 1 step 分（forward → backward〈計測対象〉→ SGD update）を実行し、
    // backward 区間限定の壁時間（秒）・カウンタ差分（encode/cb/wait）を
    // 返す。forward・update 自体は計測窓の外（`Instant` 計測は
    // backward_device_param_store 呼び出しの前後のみ）。
    let run_step_measure_backward =
        |store: &mut fandhe_ai::DeviceParamStore| -> (f64, usize, usize, usize) {
            let tape = fandhe_ai::tape_for(device).unwrap();
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);
            let pred = model.forward_resident(&tape, &x, store).unwrap();
            let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
            let last_loss = loss
                .to_tensor()
                .get(&[])
                .expect("loss は shape [] スカラー");
            assert!(
                last_loss.is_finite(),
                "MEASURE_ERROR: step loss not finite: {last_loss}"
            );

            let before = fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot()
                .expect("singleton MetalContext は実機で必ず取得できるはず");
            let t0 = Instant::now();
            let grads = tape.backward_device_param_store(&loss, store).unwrap();
            let backward_secs = t0.elapsed().as_secs_f64();
            let after = fandhe_ai_backend_metal::__diagnostic_batch_counters_snapshot()
                .expect("singleton MetalContext は実機で必ず取得できるはず");

            tape.step_device_param_store(store, &grads, &config)
                .unwrap();

            (
                backward_secs,
                after.encode_calls - before.encode_calls,
                after.command_buffers - before.command_buffers,
                after.wait_until_completed - before.wait_until_completed,
            )
        };

    // steady-state 到達（`mnist_scale_train_reuse_metal_batch_counters`
    // と同じ考え方。プールのフリーリスト充足前は確保パスが混じり
    // カウンタ・時間が安定しない）。
    const WARMUP: usize = TRAIN_WARMUP;
    for _ in 0..WARMUP {
        run_step_measure_backward(&mut store);
    }

    let mut backward_secs = Vec::with_capacity(TRIALS);
    let mut encode_deltas = Vec::with_capacity(TRIALS);
    let mut command_buffer_deltas = Vec::with_capacity(TRIALS);
    let mut wait_deltas = Vec::with_capacity(TRIALS);
    for _ in 0..TRIALS {
        let (secs, encode_delta, command_buffer_delta, wait_delta) =
            run_step_measure_backward(&mut store);
        backward_secs.push(secs);
        encode_deltas.push(encode_delta);
        command_buffer_deltas.push(command_buffer_delta);
        wait_deltas.push(wait_delta);
    }

    let q = median_q1_q3(&backward_secs).expect("backward_secs の分位点計算に失敗した");

    println!(
        "[mnist_scale_train_reuse_metal_backward_dinput_phase] backward-only \
         median={:.6}ms q1={:.6}ms q3={:.6}ms (n={TRIALS}) \
         encode_deltas={encode_deltas:?} command_buffer_deltas={command_buffer_deltas:?} \
         wait_deltas={wait_deltas:?} — 事前登録仮説（#1562 codex-review 是正後・\
         #1563 で更新）: encode_delta=5（MseLoss VJP の dispatch_sync 1 回 + \
         d_input 2 回 + d_weight 2 回。不変）・command_buffer_delta=3（cb 生成は \
         MseLoss 1 + [d_weight(L2)+d_input(L2)合流] 1 + [d_weight(L1)+d_input(L1)\
         合流] 1。#1562 時点の 4 から 1 減る）・wait_delta=3（MseLoss 1 + \
         [d_weight(L2)+d_input(L2)合流] 1 + [d_weight(L1)+d_input(L1)合流] 1。\
         窓終了時点で open バッチが残らないため全て本窓内で wait 済み）。\
         record only, non-gating（`docs/backend-metal-command-batching-design.md` \
         §7.4）",
        q.median * 1e3,
        q.q1 * 1e3,
        q.q3 * 1e3,
    );

    for (i, ((&encode_delta, &command_buffer_delta), &wait_delta)) in encode_deltas
        .iter()
        .zip(command_buffer_deltas.iter())
        .zip(wait_deltas.iter())
        .enumerate()
    {
        if encode_delta != 5 || command_buffer_delta != 3 || wait_delta != 3 {
            println!(
                "[mnist_scale_train_reuse_metal_backward_dinput_phase] trial {i}: \
                 事前登録仮説から乖離（encode_delta={encode_delta} \
                 command_buffer_delta={command_buffer_delta} wait_delta={wait_delta}）。\
                 §7.3 記入時に原因を確認すること（record only のため assert では \
                 止めない）"
            );
        }
    }
}
