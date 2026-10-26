//! イシュー #3260: 学習 step 全体の Graph capture（`BackendOps::train_chain_captured` の CUDA 実装。
//! opt-in・既定 OFF）の GB10 実機テスト。capture あり／なしの bit 一致・fail-safe・再 capture・
//! 別スレッドとの衝突・replay 中の窓のカウンタを検証する
//! （決定記録 `docs/backend-cuda-graph-step-capture-design.md` §8）。
//!
//! # 配置理由
//! 診断カウンタ（`gemm.rs` の `GemmLaunchDiagCounters`。`cfg(test)` で有効。`pub(crate)`）へ
//! 到達する必要があるため、`train_step_chain_diag_tests_3252` と同じく `lib.rs` の兄弟モジュールに
//! 置く。カウンタはスレッドローカルなので、計測区間の呼び出しは同じスレッドで行う。
//!
//! # 実行方法（既定 feature・`--release`・単独・単スレッド）
//! stream 種別は ordinal ごとに sticky で、capture には created stream が要る。プロセス起動時に
//! 環境変数で opt-in し、最初の CUDA 初期化より前に決める。`internal-diagnostics` feature を付けると
//! stream が Legacy に固定され capture に届かないので、付けない（CI の `--all-features` では
//! `#[ignore]` のためそもそも走らない）。
//!
//! ```sh
//! FANDHE_AI_CUDA_GRAPH_TRAIN=1 cargo test -p fandhe-ai-backend-cuda --release --lib \
//!   train_step_graph_diag_tests_3260 -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! # 判定
//! - 比較は常に bit 完全一致（loss・全パラメータ。tolerance は使わない）。基準は同一プロセス内で
//!   `override_train_graph_for_scope(false)` にして走らせた別 store（同じ初期値・同じ入力）
//! - 結線の確認（黙って非 capture に落ちても bit 一致は合格してしまうため必須）として、
//!   `train_graph_stats` の差分（captured／replayed／fallbacks）も固定する

use std::sync::{Arc, Barrier};

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::optim::resident_input::{ResidentInput, TrainStepLoss};
use fandhe_ai_autodiff::optim::{DeviceParamStore, ResidentLeaf, SgdConfig};
use fandhe_ai_tensor_core::{Activation, BackendError, Tensor};

use crate::gemm::{gemm_launch_diag_reset, gemm_launch_diag_snapshot};
use crate::graph::{
    inject_train_capture_failure_for_scope, override_train_graph_for_scope, train_graph_stats,
    train_graph_stream_requested,
};
use crate::ops::CudaBackendOps;

const BATCH: usize = 64;
const DIMS: [usize; 3] = [784, 256, 10];

fn pattern(n: usize, salt: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + salt * 17) % 23) as f32 - 11.0) * 0.011)
        .collect()
}

fn tensor(salt: usize, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(pattern(n, salt), shape).expect("tensor")
}

fn config(lr: f32) -> SgdConfig {
    SgdConfig {
        lr,
        momentum: 0.9,
        dampening: 0.0,
        weight_decay: 1e-3,
        nesterov: false,
    }
}

/// 起動時に opt-in していないと created stream にならず、capture に届かないまま bit 一致だけが
/// 合格してしまう。そのため最初に確認する。
pub(super) fn require_opt_in_env() {
    assert!(
        train_graph_stream_requested(),
        "FANDHE_AI_CUDA_GRAPH_TRAIN=1 を付けて起動すること（stream 種別は起動時に決まる）"
    );
}

pub(super) struct Run {
    tape: Tape,
    store: DeviceParamStore,
    x: ResidentInput,
    y: ResidentInput,
}

/// 同じ初期値・同じ入力の MLP（784-256-10・batch 64）。
pub(super) fn new_run() -> Run {
    new_run_with(DIMS)
}

/// 層幅を指定できる版（GEMM 内部ワークスペースを拡張させる大きい store の構築用）。
/// 並行テストでは他スレッドの capture と衝突して `DeviceContextCaptureInProgress`（一過性）が
/// 返りうるため、store 構築と入力アップロードは `retry_busy` で上限つきに再試行する。
fn new_run_with(dims: [usize; 3]) -> Run {
    let tape = Tape::new_with_ops(Box::new(CudaBackendOps::new(0)));
    let params = [
        tensor(1, &[dims[0], dims[1]]),
        tensor(2, &[dims[1]]),
        tensor(3, &[dims[1], dims[2]]),
        tensor(4, &[dims[2]]),
    ];
    let refs: Vec<&Tensor<f32>> = params.iter().collect();
    let store = retry_busy(|| DeviceParamStore::new(&tape, &refs));
    let x = retry_busy(|| store.upload_resident_input(&tape, &tensor(5, &[BATCH, dims[0]])));
    let y = retry_busy(|| store.upload_resident_input(&tape, &tensor(6, &[BATCH, dims[2]])));
    Run { tape, store, x, y }
}

pub(super) fn step(run: &mut Run, lr: f32) -> Result<TrainStepLoss, BackendError> {
    let leaves: Vec<ResidentLeaf<'_>> = run.store.snapshot_resident_params(&run.tape)?;
    let steps = [
        (&leaves[0], Some(&leaves[1]), Activation::Relu),
        (&leaves[2], Some(&leaves[3]), Activation::None),
    ];
    run.store
        .train_step_device_chain(&run.tape, &run.x, &run.y, &steps, &config(lr))
}

/// 別スレッドの capture と衝突した読み出し・同期は一過性のエラーなので、上限つきで再試行する
/// （`read_train_step_loss`／`sync_to_host` の契約。poison はされない）。
pub(super) fn retry_busy<T>(mut f: impl FnMut() -> Result<T, BackendError>) -> T {
    for _ in 0..2000 {
        match f() {
            Err(BackendError::DeviceContextCaptureInProgress { .. }) => {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            other => return other.expect("retry_busy"),
        }
    }
    panic!("capture 衝突が解消しなかった");
}

/// store が poison されていないこと（`wait_device` は poison 済みなら `StorePoisoned` を返し、
/// 失敗トークンもここで検査する）。
pub(super) fn assert_not_poisoned(run: &Run) {
    retry_busy(|| run.store.wait_device(&run.tape));
}

pub(super) fn read_loss(run: &Run, h: &TrainStepLoss) -> f32 {
    retry_busy(|| run.store.read_train_step_loss(&run.tape, h))
}

pub(super) fn param_bits(run: &Run) -> Vec<Vec<u32>> {
    retry_busy(|| run.store.sync_to_host(&run.tape))
        .iter()
        .map(|t| {
            t.contiguous()
                .as_slice()
                .expect("contiguous")
                .iter()
                .map(|v| v.to_bits())
                .collect()
        })
        .collect()
}

/// 期待値が NaN なら NaN クラス一致、それ以外は bit 完全一致。
fn assert_bits(actual: &[u32], expected: &[u32], what: &str) {
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

/// 1 シナリオの結果（各 step の loss bit と、最後のパラメータ bit）。
pub(super) struct Outcome {
    pub(super) losses: Vec<u32>,
    pub(super) params: Vec<Vec<u32>>,
}

/// `steps` 回進める。`capture` は現スレッドの opt-in（override）、`reupload` は毎 step x／y を
/// upload し直す、`inject_at` はその step だけ capture 失敗を 1 回注入する。
pub(super) fn run_scenario(
    steps: usize,
    lr_of: impl Fn(usize) -> f32,
    capture: bool,
    reupload: bool,
    inject_at: Option<usize>,
) -> Outcome {
    let _opt = override_train_graph_for_scope(capture);
    let mut run = new_run();
    let mut losses = Vec::with_capacity(steps);
    for i in 0..steps {
        if reupload {
            run.x = run
                .store
                .upload_resident_input(&run.tape, &tensor(5, &[BATCH, DIMS[0]]))
                .expect("x");
            run.y = run
                .store
                .upload_resident_input(&run.tape, &tensor(6, &[BATCH, DIMS[2]]))
                .expect("y");
        }
        let _inj = (inject_at == Some(i)).then(|| inject_train_capture_failure_for_scope(1));
        let h = step(&mut run, lr_of(i)).unwrap_or_else(|e| panic!("step {i}: {e:?}"));
        drop(_inj);
        losses.push(read_loss(&run, &h).to_bits());
    }
    assert_not_poisoned(&run);
    Outcome {
        losses,
        params: param_bits(&run),
    }
}

fn assert_same(a: &Outcome, b: &Outcome, what: &str) {
    assert_bits(&a.losses, &b.losses, &format!("{what}: loss"));
    assert_eq!(a.params.len(), b.params.len(), "{what}: param count");
    for (i, (x, y)) in a.params.iter().zip(&b.params).enumerate() {
        assert_bits(x, y, &format!("{what}: param {i}"));
    }
}

/// loss／パラメータが参照とずれたとき、assert の前に原因切り分け用の `DIAG3596` 行を出す
/// （イシュー #3596。設計記録 §8.11.6）。ずれた loss が参照列のどの index と bit 一致するか
/// （= 他 step の値の取り違えか）と、パラメータがずれたかを示す。判定は呼び出し側の assert が行う。
pub(super) fn report_mismatch_3596(out: &Outcome, reference: &Outcome, what: &str) {
    let params_equal = out.params == reference.params;
    for (i, (a, e)) in out.losses.iter().zip(&reference.losses).enumerate() {
        if a != e {
            let matches: Vec<usize> = reference
                .losses
                .iter()
                .enumerate()
                .filter(|(_, r)| *r == a)
                .map(|(j, _)| j)
                .collect();
            eprintln!(
                "DIAG3596 {what}: loss idx {i} got {} expected {} equals-reference-idx {matches:?} params_equal={params_equal}",
                f32::from_bits(*a),
                f32::from_bits(*e),
            );
        }
    }
    if !params_equal {
        eprintln!("DIAG3596 {what}: params differ from reference");
    }
}

/// 10 step の基本: step 1 はウォームアップ、step 2 で capture、step 3〜10 は replay。
/// loss と全パラメータは非 capture と bit 一致（R2）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn capture_matches_plain_chain_bit_exact_and_wires_up() {
    require_opt_in_env();
    let plain = run_scenario(10, |_| 0.01, false, false, None);
    let before = train_graph_stats();
    let graph = run_scenario(10, |_| 0.01, true, false, None);
    let d = train_graph_stats();
    assert_eq!(d.captured - before.captured, 1, "capture は 1 回");
    assert_eq!(d.replayed - before.replayed, 8, "replay は 8 回");
    assert_eq!(
        d.graph_launches - before.graph_launches,
        9,
        "graph_launches == captured + replayed"
    );
    assert_eq!(d.fallbacks - before.fallbacks, 0);
    assert_same(&graph, &plain, "capture vs plain");
}

/// replay 中の step の窓で、同期・D2H・H2D・確保が 0。loss を読む区間は同期 1・D2H 1。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn replay_window_has_no_sync_alloc_or_h2d_and_loss_read_is_one_sync() {
    require_opt_in_env();
    let _opt = override_train_graph_for_scope(true);
    let mut run = new_run();
    for _ in 0..4 {
        let h = step(&mut run, 0.01).expect("warm");
        let _ = read_loss(&run, &h);
    }
    let stats = train_graph_stats();

    gemm_launch_diag_reset();
    let before = gemm_launch_diag_snapshot();
    let handle = step(&mut run, 0.01).expect("step");
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(
        train_graph_stats().replayed - stats.replayed,
        1,
        "窓の step は replay のはず"
    );
    assert_eq!(d.stream_syncs, 0, "窓内の stream 同期: {d:?}");
    assert_eq!(d.d2h_calls, 0, "窓内の D2H: {d:?}");
    assert_eq!(d.h2d_calls, 0, "窓内の H2D（upload_h2d_new）: {d:?}");
    assert_eq!(d.mem_uploads, 0, "窓内の H2D（MemoryOps::upload）: {d:?}");
    assert_eq!(d.mem_allocs, 0, "窓内のデバイス確保: {d:?}");

    let before = gemm_launch_diag_snapshot();
    let loss = read_loss(&run, &handle);
    let d_read = gemm_launch_diag_snapshot().since(before);
    assert!(loss.is_finite(), "loss={loss}");
    assert_eq!(d_read.stream_syncs, 1, "loss 読み出しの同期: {d_read:?}");
    assert_eq!(d_read.d2h_calls, 1, "loss 読み出しの D2H: {d_read:?}");
    println!("DIAG3260 window={d:?} read={d_read:?} loss={loss}");
}

/// capture の失敗（注入）では ordinal を poison せず、capture しない既存チェーンで同じ step を
/// 実行して bit 一致を保つ（R3）。失敗は `fallbacks` に数えられ、次の step で再 capture に成功する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn injected_capture_failure_falls_back_to_plain_chain_bit_exact() {
    require_opt_in_env();
    let plain = run_scenario(8, |_| 0.01, false, false, None);
    let before = train_graph_stats();
    // step index 1（2 step 目）が最初の capture 試行。
    let graph = run_scenario(8, |_| 0.01, true, false, Some(1));
    let d = train_graph_stats();
    assert_eq!(d.fallbacks - before.fallbacks, 1, "失敗は 1 回");
    assert_eq!(
        d.captured - before.captured,
        1,
        "次の step で capture に成功"
    );
    assert_eq!(d.replayed - before.replayed, 5, "残りは replay");
    assert_same(&graph, &plain, "injected failure vs plain");
}

/// 失敗が続くとスレッド・ordinal 単位で capture を止め、以後は既存チェーンで走り続ける
/// （結果は bit 一致のまま）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn repeated_capture_failures_stop_capturing_but_stay_bit_exact() {
    require_opt_in_env();
    let plain = run_scenario(8, |_| 0.01, false, false, None);
    let before = train_graph_stats();
    let graph = {
        let _opt = override_train_graph_for_scope(true);
        let mut run = new_run();
        let mut losses = Vec::new();
        for i in 0..8 {
            // 2 step 目以降すべてで失敗を注入する（抑止が働けば後半は capture を試さない）。
            let _inj = (i >= 1).then(|| inject_train_capture_failure_for_scope(1));
            let h = step(&mut run, 0.01).expect("step");
            losses.push(read_loss(&run, &h).to_bits());
        }
        assert_not_poisoned(&run);
        Outcome {
            losses,
            params: param_bits(&run),
        }
    };
    let d = train_graph_stats();
    assert_eq!(
        d.fallbacks - before.fallbacks,
        u64::from(crate::graph::TRAIN_GRAPH_MAX_CAPTURE_FAILURES),
        "失敗が上限に達したら capture を試さなくなる"
    );
    assert_eq!(d.captured - before.captured, 0);
    assert_same(&graph, &plain, "repeated failures vs plain");
}

/// x／y を毎 step upload し直す使い方（キーが外れうる）でも結果は bit 一致し、再 capture は
/// 抑止で打ち止めになる。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn reuploading_inputs_every_step_stays_bit_exact() {
    require_opt_in_env();
    let plain = run_scenario(10, |_| 0.01, false, true, None);
    let before = train_graph_stats();
    let graph = run_scenario(10, |_| 0.01, true, true, None);
    let d = train_graph_stats();
    println!(
        "DIAG3260 reupload captured={} replayed={} fallbacks={}",
        d.captured - before.captured,
        d.replayed - before.replayed,
        d.fallbacks - before.fallbacks
    );
    assert_same(&graph, &plain, "reupload vs plain");
}

/// lr を 1 回変えると焼き込み値が変わるので再 capture する（R5）。毎 step 変えると抑止で打ち止め。
/// いずれも bit 一致。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn lr_change_recaptures_and_stays_bit_exact() {
    require_opt_in_env();
    let step_lr = |i: usize| if i < 4 { 0.01 } else { 0.02 };
    let plain = run_scenario(8, step_lr, false, false, None);
    let before = train_graph_stats();
    let graph = run_scenario(8, step_lr, true, false, None);
    let d = train_graph_stats();
    assert_eq!(d.captured - before.captured, 2, "lr 変更で再 capture");
    assert_eq!(
        d.replayed - before.replayed,
        5,
        "step 3,4 と 6,7,8 が replay"
    );
    assert_same(&graph, &plain, "lr change vs plain");

    let every = |i: usize| 0.01 + 0.001 * i as f32;
    let plain = run_scenario(10, every, false, false, None);
    let before = train_graph_stats();
    let graph = run_scenario(10, every, true, false, None);
    let d = train_graph_stats();
    assert_eq!(
        d.captured - before.captured,
        u64::from(crate::graph::TRAIN_GRAPH_MAX_CONSECUTIVE_CAPTURES),
        "毎 step 変える場合は連続 capture の上限で止まる"
    );
    assert_eq!(d.replayed - before.replayed, 0);
    assert_same(&graph, &plain, "lr every step vs plain");
}

/// 2 スレッドがそれぞれ別の store で、同じデバイス・capture ON のまま回してもエラーなし
/// （`DeviceContextCaptureInProgress` を待機・再試行で吸収する。R6）。結果は単独実行と bit 一致。
///
/// 各スレッドは 1 step 目（ウォームアップ。非 capture の通常チェーン）を終えて柵で揃えてから
/// capture に入る。この柵は #3596 以前の「通常チェーンの途中で他スレッドの capture が始まると
/// store が poison される」制約の回避として入れたもので、柵なしで同じ状況を作る回帰テストは
/// `unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact`（排他は設計記録 §8.11）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn two_threads_capturing_concurrently_do_not_fail() {
    require_opt_in_env();
    const STEPS: usize = 20;
    let reference = run_scenario(STEPS, |_| 0.01, false, false, None);
    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let _opt = override_train_graph_for_scope(true);
                let mut run = new_run();
                let mut losses = Vec::new();
                let h = step(&mut run, 0.01).expect("step 0");
                losses.push(read_loss(&run, &h).to_bits());
                barrier.wait();
                for i in 1..STEPS {
                    let h = step(&mut run, 0.01).unwrap_or_else(|e| panic!("step {i}: {e:?}"));
                    losses.push(read_loss(&run, &h).to_bits());
                }
                assert_not_poisoned(&run);
                Outcome {
                    losses,
                    params: param_bits(&run),
                }
            })
        })
        .collect();
    for (n, h) in handles.into_iter().enumerate() {
        let out = h.join().expect("スレッドが panic しない");
        assert_same(&out, &reference, &format!("thread {n} vs single"));
    }
}

/// 柵を置かずに、同じ ordinal で capture するスレッドと非 capture のスレッドが並行して学習しても、
/// どちらの store も poison されず、全 step の loss と最終パラメータが単独実行と bit 一致する
/// （イシュー #3596。学習チェーンと capture の ordinal 単位の排他。設計記録 §8.11）。
///
/// 衝突を起こしやすくするため、capture 側は繰り返しごとに新しい store を作り（毎回、非 capture の
/// ウォームアップ step → capture の流れを通る）、非 capture 側も新しい store で通常チェーンを
/// 繰り返す。スレッド間に柵は置かない。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact() {
    require_opt_in_env();
    const STEPS: usize = 20;
    const REPEATS: usize = 30;
    let reference = run_scenario(STEPS, |_| 0.01, false, false, None);
    let handles: Vec<_> = [true, false]
        .into_iter()
        .map(|capture| {
            std::thread::spawn(move || {
                let _opt = override_train_graph_for_scope(capture);
                let mut outs = Vec::with_capacity(REPEATS);
                for _ in 0..REPEATS {
                    let mut run = new_run();
                    let mut losses = Vec::with_capacity(STEPS);
                    for i in 0..STEPS {
                        let h = step(&mut run, 0.01).unwrap_or_else(|e| panic!("step {i}: {e:?}"));
                        losses.push(read_loss(&run, &h).to_bits());
                    }
                    assert_not_poisoned(&run);
                    outs.push(Outcome {
                        losses,
                        params: param_bits(&run),
                    });
                }
                outs
            })
        })
        .collect();
    for (n, h) in handles.into_iter().enumerate() {
        for (r, out) in h
            .join()
            .expect("スレッドが panic しない")
            .iter()
            .enumerate()
        {
            report_mismatch_3596(out, &reference, &format!("thread {n} repeat {r}"));
            assert_same(out, &reference, &format!("thread {n} repeat {r} vs single"));
        }
    }
}

/// store A を capture → 同デバイスの大きい store B を実行してワークスペースを拡張 → A を再生、
/// の順で回しても A の loss・全パラメータが非 capture と bit 一致する（codex P0 の回帰。
/// 世代が進んだ旧 Graph は再生されず、既存チェーンで実行されたのち再 capture される）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn workspace_growth_by_other_store_invalidates_captured_graph() {
    require_opt_in_env();
    const BIG: [usize; 3] = [784, 2048, 10];
    let drive = |capture: bool| -> Outcome {
        let _opt = override_train_graph_for_scope(capture);
        let mut a = new_run();
        let mut b = new_run_with(BIG);
        let mut losses = Vec::new();
        let run_a = |a: &mut Run, n: usize, losses: &mut Vec<u32>| {
            for _ in 0..n {
                let h = step(a, 0.01).expect("step A");
                losses.push(read_loss(a, &h).to_bits());
            }
        };
        run_a(&mut a, 4, &mut losses);
        let epoch_of = || {
            let dev = crate::context_cache::cached_device(0).expect("device");
            crate::context_cache::cached_gemm(&dev)
                .expect("gemm")
                .workspace_epoch()
        };
        let epoch_before = epoch_of();
        // B の 1 step が共有ワークスペースを拡張する（A の Graph が焼き込んだ領域を解放）。
        let hb = step(&mut b, 0.01).expect("step B");
        let _ = read_loss(&b, &hb);
        if capture {
            // 無効化の検証が成立する前提: この区間でワークスペース世代が実際に進んだこと。
            assert!(
                epoch_of() > epoch_before,
                "B の step で共有ワークスペースの世代が進むこと（capture 側を先に実行する）"
            );
        }
        run_a(&mut a, 4, &mut losses);
        assert_not_poisoned(&a);
        Outcome {
            losses,
            params: param_bits(&a),
        }
    };
    // capture 側を先に実行する。非 capture 側を先に回すと共有 grow-only ワークスペースが BIG まで
    // 拡張済みになり、capture 側で世代が進まず無効化を検証できない（#3260 codex P2）。
    let before = train_graph_stats();
    let graph = drive(true);
    let d = train_graph_stats();
    let plain = drive(false);
    assert!(
        d.captured - before.captured >= 2,
        "拡張後に旧 Graph を再生せず再 capture する"
    );
    assert_same(&graph, &plain, "workspace growth vs plain");
}
