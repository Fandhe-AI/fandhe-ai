//! イシュー #3596: 学習 graph の replay と、同じ ordinal の非 capture 学習チェーンが共有の永続
//! ワークスペース（MSE の `partial`）を取り合わないことの GB10 診断テスト
//! （設計記録 `docs/backend-cuda-graph-step-capture-design.md` §8.11.6）。
//!
//! G1（`train_step_graph_diag_tests_3260::unfenced_chain_and_capture_threads_*`）は capture の開始が
//! 重なるため #3597 の排他にも依存する。本テストは capture の開始を柵で重ならないようにし、
//! 「replay」と「非 capture チェーン」の衝突だけを切り出す。#3597 より前のツリーでも poison 経路を
//! 通らずに実行できるため、原因が #3597 以前からあるかの切り分けに使う。
//!
//! # 実行方法（GB10・既定 feature・`--release`・単独）
//! ```sh
//! FANDHE_AI_CUDA_GRAPH_TRAIN=1 cargo test -p fandhe-ai-backend-cuda --release --lib \
//!   train_graph_shared_workspace_tests_3596 -- --ignored --test-threads=1 --nocapture
//! ```
//! 判定は bit 完全一致のみ（tolerance なし）。不一致時は assert の前に `DIAG3596` 行を出す。

use std::sync::{Arc, Barrier};

use super::train_step_graph_diag_tests_3260::{
    Outcome, assert_not_poisoned, new_run, param_bits, read_loss, report_mismatch_3596,
    require_opt_in_env, run_scenario, step,
};
use crate::graph::override_train_graph_for_scope;

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。FANDHE_AI_CUDA_GRAPH_TRAIN=1 で単独実行"]
fn replay_thread_and_plain_chain_thread_keep_losses_bit_exact() {
    require_opt_in_env();
    const REPLAY_STEPS: usize = 200;
    const PLAIN_STEPS: usize = 20;
    const PLAIN_REPEATS: usize = 30;
    let reference = run_scenario(REPLAY_STEPS, |_| 0.01, false, false, None);
    let barrier = Arc::new(Barrier::new(2));

    // thread 0: ウォームアップ → capture を柵の前に終え、柵の後は replay だけを続ける。
    let b0 = Arc::clone(&barrier);
    let replay = std::thread::spawn(move || {
        let _opt = override_train_graph_for_scope(true);
        let mut run = new_run();
        let mut losses = Vec::with_capacity(REPLAY_STEPS);
        for i in 0..REPLAY_STEPS {
            if i == 2 {
                b0.wait();
            }
            let h = step(&mut run, 0.01).unwrap_or_else(|e| panic!("step {i}: {e:?}"));
            losses.push(read_loss(&run, &h).to_bits());
        }
        assert_not_poisoned(&run);
        Outcome {
            losses,
            params: param_bits(&run),
        }
    });

    // thread 1: 柵の後に、新しい store で非 capture のチェーンを繰り返す。
    let b1 = Arc::clone(&barrier);
    let plain = std::thread::spawn(move || {
        let _opt = override_train_graph_for_scope(false);
        b1.wait();
        let mut outs = Vec::with_capacity(PLAIN_REPEATS);
        for _ in 0..PLAIN_REPEATS {
            let mut run = new_run();
            let mut losses = Vec::with_capacity(PLAIN_STEPS);
            for i in 0..PLAIN_STEPS {
                let h = step(&mut run, 0.01).unwrap_or_else(|e| panic!("step {i}: {e:?}"));
                losses.push(read_loss(&run, &h).to_bits());
            }
            assert_not_poisoned(&run);
            outs.push(losses);
        }
        outs
    });

    let replay_out = replay.join().expect("replay スレッドが panic しない");
    let plain_outs = plain.join().expect("plain スレッドが panic しない");
    let mut failed = false;
    // 非 capture 側は損失列のみを参照の先頭 PLAIN_STEPS 件と比較する（パラメータは store ごとに
    // 別で、ずれた場合は loss 側にも現れる。パラメータの診断は replay 側で出す）。
    let short = Outcome {
        losses: reference.losses[..PLAIN_STEPS].to_vec(),
        params: Vec::new(),
    };
    for (r, losses) in plain_outs.into_iter().enumerate() {
        let probe = Outcome {
            losses,
            params: Vec::new(),
        };
        report_mismatch_3596(&probe, &short, &format!("plain repeat {r}"));
        failed |= probe.losses != short.losses;
    }
    report_mismatch_3596(&replay_out, &reference, "replay thread");
    failed |= replay_out.losses != reference.losses || replay_out.params != reference.params;
    assert!(
        !failed,
        "replay と非 capture チェーンの並行で loss／パラメータが参照とずれた（DIAG3596 行を参照）"
    );
}
