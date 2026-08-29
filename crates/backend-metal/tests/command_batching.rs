//! `MetalContext` のコマンドバッファ共有バッチ（イシュー #1017・
//! `docs/backend-metal-command-batching-design.md`）の実機回帰テスト。
//!
//! `context.rs::MetalContext::encode`／`flush`／`synchronize` は
//! `pub(crate)`／内部限定のため、本テスト（クレート外の統合テスト）
//! からは直接呼べない。代わりに、実際にバッチを積む唯一の公開経路
//! である [`BackendOps::sgd_step_device_tracked`]（`ops.rs::
//! MetalBackendOps` のオーバーライド）を連続呼び出しし、間に
//! `download`（暗黙の `synchronize`）を挟まないことで「1 コマンド
//! バッファに複数 dispatch が積まれ、ホスト実体化まで待たれない」
//! 経路を exercise する。GPU fault のような決定的な実行時エラーは
//! 実機でも再現不能なため、失敗伝播（`batch_state::propagate_failure`）
//! 自体は `crates/tensor-core/src/dispatch_failure.rs`・
//! `crates/backend-metal/src/batch_state.rs`・
//! `crates/autodiff/src/optim/device_store.rs` の単体テスト（Linux で
//! 実行）で検証済みであり、本ファイルは「正しさ」（数値一致・実行順序）
//! のみを実機で確認する（PR 本文にも明記）。
//!
//! `cfg(target_os = "macos")` は `crates/backend-metal/tests/
//! sgd_device_parity.rs` と同じ理由（`MetalBackendOps` 自体が
//! `cfg(target_os = "macos")` ゲート）で付ける。各 `#[test]` は理由付き
//! `#[ignore]` で通常 CI（GitHub ホステッド ubuntu-latest。Metal 実機
//! 非搭載）から除外し、macOS 実機で明示指定したときのみ実行する:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{BackendOps, DispatchFailureCell, SgdStepConfig, Tensor};

/// REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
/// `.claude/rules/coding-rust.md`）。
fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

/// CPU 側で愚直に計算した vanilla SGD（momentum なし）の期待値。
/// `fandhe_ai_autodiff::optim::sgd::Sgd::step` と同じ更新式
/// （`param -= lr * grad`）だが、本テストはデバイス常駐更新の
/// バッチング自体（実行順序・複数ステップの累積）を検証するのが目的の
/// ため、依存を増やさず手計算で再現する。
fn vanilla_sgd_expected(init: &[f32], grads: &[Vec<f32>], lr: f32) -> Vec<f32> {
    let mut param = init.to_vec();
    for grad in grads {
        for (p, g) in param.iter_mut().zip(grad.iter()) {
            *p -= lr * g;
        }
    }
    param
}

/// イシュー #1017 受け入れ条件 1: 「同一バッチ内の複数 dispatch が
/// 発行順に実行される」ことを、同一パラメータへの 2 回連続
/// `sgd_step_device_tracked`（間に `download` を挟まない = 同一
/// コマンドバッファへ積まれる）の結果が、逐次適用した場合の期待値と
/// 一致することで確認する。順序が入れ替わったり、2 回目が 1 回目の
/// 結果を読まず初期値に対して適用されたりすると期待値からずれる。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn dependent_dispatches_in_one_batch_execute_in_order() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("MetalBackendOps must implement MemoryOps");

    let init = vec![1.0f32, -2.0, 0.5, 3.25];
    let mut param = mem
        .upload(&Tensor::new(init.clone(), &[4]).unwrap())
        .unwrap();

    let grad1 = vec![0.1f32, 0.2, -0.1, 0.05];
    let grad2 = vec![0.05f32, -0.1, 0.2, 0.1];
    let grad1_buf = mem
        .upload(&Tensor::new(grad1.clone(), &[4]).unwrap())
        .unwrap();
    let grad2_buf = mem
        .upload(&Tensor::new(grad2.clone(), &[4]).unwrap())
        .unwrap();

    let config = SgdStepConfig {
        lr: 0.1,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };
    let token = DispatchFailureCell::new();

    // 2 回連続で encode するだけで、ここでは待たない（download が
    // 唯一の同期点。モジュール冒頭コメント参照）。
    ops.sgd_step_device_tracked(&mut param, &grad1_buf, None, &config, &token)
        .expect("1st sgd_step_device_tracked must succeed");
    ops.sgd_step_device_tracked(&mut param, &grad2_buf, None, &config, &token)
        .expect("2nd sgd_step_device_tracked must succeed");

    let result = mem
        .download(&param)
        .expect("download must synchronize the batch");
    let expected = vanilla_sgd_expected(&init, &[grad1, grad2], config.lr);
    for (i, exp) in expected.iter().enumerate() {
        assert_close(result.get(&[i]).unwrap(), *exp, &format!("index {i}"));
    }
    assert!(
        !token.is_set(),
        "no runtime error expected on real hardware"
    );
}

/// イシュー #1017 受け入れ条件 2: 複数パラメータ × 複数ステップを
/// `sgd_step_device_tracked` で連続投入し（ステップ間で `download` しない
/// = 全ステップが同一バッチに積まれるか、上限到達時のみ自動 flush される）、
/// 最終結果を CPU 参照実装（`vanilla_sgd_expected`）と突合する
/// （REQ-2 統一複合判定）。`crates/backend-metal/tests/
/// sgd_device_parity.rs` と異なりステップごとの `download` を行わない
/// 点が本イシューの新規契約を検証する核心。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn sgd_batched_multi_param_matches_cpu_across_many_steps() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("MetalBackendOps must implement MemoryOps");

    const STEPS: usize = 100;
    let inits: [Vec<f32>; 3] = [vec![1.0, -2.0], vec![0.5, 3.25, -1.0], vec![2.0]];
    let mut params: Vec<_> = inits
        .iter()
        .map(|init| {
            mem.upload(&Tensor::new(init.clone(), &[init.len()]).unwrap())
                .unwrap()
        })
        .collect();

    let mut all_grads: Vec<Vec<Vec<f32>>> = inits.iter().map(|_| Vec::new()).collect();
    let token = DispatchFailureCell::new();
    let config = SgdStepConfig {
        lr: 0.01,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };

    for step in 0..STEPS {
        for (i, init) in inits.iter().enumerate() {
            let grad: Vec<f32> = (0..init.len())
                .map(|j| 0.01 * (step as f32 + 1.0) + 0.02 * j as f32)
                .collect();
            let grad_buf = mem
                .upload(&Tensor::new(grad.clone(), &[grad.len()]).unwrap())
                .unwrap();
            all_grads[i].push(grad);
            ops.sgd_step_device_tracked(&mut params[i], &grad_buf, None, &config, &token)
                .expect("sgd_step_device_tracked must succeed on real hardware");
        }
    }

    assert!(
        !token.is_set(),
        "no runtime error expected across {STEPS} steps on real hardware"
    );

    for (i, init) in inits.iter().enumerate() {
        let result = mem
            .download(&params[i])
            .expect("download must synchronize all batched steps");
        let expected = vanilla_sgd_expected(init, &all_grads[i], config.lr);
        for (j, exp) in expected.iter().enumerate() {
            assert_close(
                result.get(&[j]).unwrap(),
                *exp,
                &format!("param {i} index {j}"),
            );
        }
    }
}

/// イシュー #1017 の安全弁（`batch_state::MAX_DISPATCHES_PER_BATCH` =
/// 256）到達時の自動 flush が、結果の正しさを崩さないことを確認する。
/// 単体テスト（`batch_state.rs`）は純粋ロジック（カウンタ）のみを
/// 検証するため、本テストは実機で「実際に上限を跨いでも数値が壊れない」
/// ことまで確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn synchronize_after_auto_flush_at_limit() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("MetalBackendOps must implement MemoryOps");

    // `batch_state::MAX_DISPATCHES_PER_BATCH`（256）を跨ぐ回数。
    const STEPS: usize = 300;
    let init = vec![10.0f32];
    let mut param = mem
        .upload(&Tensor::new(init.clone(), &[1]).unwrap())
        .unwrap();

    let mut grads = Vec::with_capacity(STEPS);
    let token = DispatchFailureCell::new();
    let config = SgdStepConfig {
        lr: 0.01,
        momentum: 0.0,
        dampening: 0.0,
        weight_decay: 0.0,
        nesterov: false,
        is_first_step: true,
    };
    for step in 0..STEPS {
        let grad = vec![0.001 * (step as f32 + 1.0)];
        let grad_buf = mem
            .upload(&Tensor::new(grad.clone(), &[1]).unwrap())
            .unwrap();
        grads.push(grad);
        ops.sgd_step_device_tracked(&mut param, &grad_buf, None, &config, &token)
            .expect("sgd_step_device_tracked must succeed across the auto-flush boundary");
    }

    let result = mem
        .download(&param)
        .expect("download must synchronize both the auto-flushed and the final open batch");
    let expected = vanilla_sgd_expected(&init, &grads, config.lr);
    assert_close(
        result.get(&[0]).unwrap(),
        expected[0],
        "post auto-flush value",
    );
    assert!(!token.is_set());
}

/// イシュー #1017 の後方互換契約: 既存の `dispatch_sync` 呼び出し元
/// （`gemm.rs`／`elementwise.rs`／`rmsnorm.rs`／`softmax.rs`）は本イシューで
/// 一切変更していない。その回帰は既存スイート
/// （`cpu_metal_parity.rs`／`sgd_device_parity.rs`／`rmsnorm_parity.rs`／
/// `softmax_parity.rs` 等）が引き続き green であることで担保する
/// （設計文書 §6.1 (4)）。本ファイルは新規追加した `_tracked` 経路のみを
/// 対象とし、既存経路の重複検証は行わない。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない。ドキュメント目的の no-op"]
fn dispatch_sync_callers_unchanged_regression_is_covered_elsewhere() {
    // 意図的な no-op（モジュール冒頭コメント参照）。
}
