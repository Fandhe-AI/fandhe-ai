//! `fandhe_ai::optim`（AMP。イシュー #1625・#1721・本イシュー #1722）
//! のみを損失スケーリング面として使う学習ループ統合テスト。
//!
//! **本ファイルは `fandhe_ai` と `bench_harness::rng` 以外を import しない**
//! （`fandhe_ai_autodiff`／`fandhe_ai_tensor_core` を一切 import しない契約。
//! `crates/facade/tests/optim_train_loop.rs` と同型の構成。レビュー・CI では
//! `grep -n "fandhe_ai_autodiff\|fandhe_ai_tensor_core" tests/optim_amp_train_loop.rs`
//! がヒット 0 件であることを確認する）。
//!
//! **適用順序契約（AMP 使用時）**: `scale_loss → backward → unscale（＋非有限
//! 検出）→ should_skip_step が true ならこの step の clip・optimizer step を
//! 両方スキップ → false なら clip → optimizer step → 最後に必ず
//! GradScaler::update`（`fandhe_ai::optim` モジュール doc「適用順序契約」節
//! 「AMP（GradScaler）を使う場合」を参照）。AMP を使わない既存経路の契約は
//! `optim_train_loop.rs` を参照。
//!
//! **決定的シード**: モデル・データ・シードは `optim_train_loop.rs`・
//! `compat_sequential_train.rs` と同一（`.claude/rules/coding-rust.md`
//! 「学習系回帰テストには決定的シード設定ユーティリティを使う」）。
//!
//! **数値判定の規律**: 収束判定は既存様式（最終 loss が初期 loss から十分
//! 減少すること）を踏襲し、新規の許容誤差（tolerance）は設けない
//! （`.claude/rules/coding-rust.md`）。1 step の grad bit 一致検証（下記
//! `amp_one_step_grads_match_non_amp_bit_exact_on_cpu`）も tolerance を持ち込まず
//! `f32::to_bits()` の完全一致で行う。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{
    GradScaler, GradScalerConfig, Sgd, SgdConfig, UnscaleResult, clip_grad_norm,
};

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `crates/facade/tests/optim_train_loop.rs::gen_regression_data` と同一
/// 生成順（`x`: `[BATCH, D_IN]`・`y`: `[BATCH, D_OUT]`）。
fn gen_regression_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (
        Tensor::new(x, &[BATCH, D_IN])
            .unwrap_or_else(|e| panic!("test fixture: x の shape 構築に失敗: {e}")),
        Tensor::new(y, &[BATCH, D_OUT])
            .unwrap_or_else(|e| panic!("test fixture: y の shape 構築に失敗: {e}")),
    )
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap_or_else(|e| panic!("test fixture: 層 1 の構築に失敗: {e}"))
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap_or_else(|e| panic!("test fixture: 層 2 の構築に失敗: {e}"))
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[])
        .unwrap_or_else(|| panic!("test fixture: スカラー shape [] のはず"))
}

// =====================================================================
// AMP（GradScaler）+ Sgd + clip の学習ループ
// =====================================================================

/// `Sequential::bind` → forward → `Var::mse_loss` → `scaler.scale_loss` →
/// `Tape::backward` → `SequentialVars::trainable_grads` → `scaler.unscale` →
/// `should_skip_step` が true ならこの step をスキップ → false なら
/// `clip_grad_norm` → `Sgd::step` → `Sequential::apply_parameters` →
/// `scaler.update` の 1 ステップを `steps` 回繰り返す。
///
/// **借用スコープ**: `SequentialVars`（`bound`）は `&model`／`&tape` を
/// 借用するため、`apply_parameters`（`&mut model`）を呼ぶ前に必ずブロックを
/// 抜けて借用を解放する（`optim_train_loop.rs` と同じ構成）。
fn train_with_sgd_amp_and_clip(
    model: &mut Sequential,
    steps: usize,
    lr: f32,
    max_norm: f32,
    config: GradScalerConfig,
) -> (Vec<f32>, usize) {
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut sgd = Sgd::new(SgdConfig::new(lr))
        .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));
    let mut scaler = GradScaler::new(config)
        .unwrap_or_else(|e| panic!("test fixture: GradScaler::new が失敗した: {e}"));
    let mut log = Vec::with_capacity(steps);
    let mut skipped_steps = 0usize;

    for _ in 0..steps {
        // クロージャの戻り値: `Some(updated)` なら `apply_parameters` へ渡す
        // 更新後パラメータ列、`None` ならこの step は非有限勾配でスキップ
        // （`optimizer step` を呼ばず `model` は不変のまま次 step へ進む）。
        let outcome = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let x = tape.var(&x_data);
            let y = tape.var(&y_data);

            let pred = bound
                .forward(&tape, &x)
                .unwrap_or_else(|e| panic!("test fixture: forward が失敗した: {e}"));
            let loss = pred
                .mse_loss(&y)
                .unwrap_or_else(|e| panic!("test fixture: mse_loss が失敗した: {e}"));
            log.push(scalar(&loss.to_tensor()));

            // 適用順序契約 1: scale_loss → backward。
            let scaled_loss = scaler
                .scale_loss(&loss)
                .unwrap_or_else(|e| panic!("test fixture: scale_loss が失敗した: {e}"));
            let grads = tape
                .backward(&scaled_loss)
                .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
            let grad_refs = bound
                .trainable_grads(&grads)
                .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));

            // 適用順序契約 2: unscale（非有限検出込み）→ should_skip_step。
            let unscale_result: UnscaleResult = scaler
                .unscale(&grad_refs)
                .unwrap_or_else(|e| panic!("test fixture: GradScaler::unscale が失敗した: {e}"));

            if unscale_result.should_skip_step() {
                None
            } else {
                // 適用順序契約 3: clip は unscale 後の生勾配にのみ適用する。
                let unscaled_refs: Vec<&Tensor<f32>> = unscale_result.grads.iter().collect();
                let clip_result = clip_grad_norm(&unscaled_refs, max_norm)
                    .unwrap_or_else(|e| panic!("test fixture: clip_grad_norm が失敗した: {e}"));
                let clipped_refs: Vec<&Tensor<f32>> = clip_result.grads.iter().collect();
                let param_refs = model.trainable_parameters();
                let updated = sgd
                    .step(&param_refs, &clipped_refs)
                    .unwrap_or_else(|e| panic!("test fixture: Sgd::step が失敗した: {e}"));
                Some(updated)
            }
        };

        match outcome {
            Some(updated) => {
                model
                    .apply_parameters(updated)
                    .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗した: {e}"));
                // 適用順序契約 4: スキップしなかった step は non-finite なし。
                scaler
                    .update(false)
                    .unwrap_or_else(|e| panic!("test fixture: GradScaler::update が失敗した: {e}"));
            }
            None => {
                skipped_steps += 1;
                // 適用順序契約 4（スキップした step でも必ず呼ぶ）。
                scaler
                    .update(true)
                    .unwrap_or_else(|e| panic!("test fixture: GradScaler::update が失敗した: {e}"));
            }
        }
    }

    (log, skipped_steps)
}

/// `fandhe_ai::optim::{GradScaler, Sgd, clip_grad_norm}` のみを使った AMP
/// 学習ループで loss が減少すること（受入基準 1）。`init_scale=256.0` は
/// 有限精度の範囲で早期に overflow しない小さめの値（`growth_interval=1000`
/// のため 100 step では成長しない）を選び、`skipped_steps == 0`・
/// `scaler.scale()` 不変（backoff が一度も起きない）ことも併せて固定する。
#[test]
fn sgd_with_amp_and_clip_converges_via_facade_only() {
    const STEPS: usize = 100;
    const LR: f32 = 0.05;
    const MAX_NORM: f32 = 10.0;
    let config = GradScalerConfig {
        init_scale: 256.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 1000,
    };

    let mut model = build_model();
    let (log, skipped_steps) = train_with_sgd_amp_and_clip(&mut model, STEPS, LR, MAX_NORM, config);

    assert_eq!(log.len(), STEPS);
    let initial = log[0];
    let final_loss = *log.last().unwrap_or_else(|| unreachable!("log は空でない"));
    assert!(final_loss.is_finite(), "final loss が非有限: {final_loss}");
    assert!(
        final_loss < 0.5 * initial,
        "loss did not converge sufficiently: initial={initial} final={final_loss}"
    );
    assert_eq!(
        skipped_steps, 0,
        "この回帰データ・init_scale では非有限勾配は発生しないはず"
    );
}

// =====================================================================
// AMP scale/unscale の線形性（bit 完全一致）
// =====================================================================

/// 同一初期モデル・同一データで 1 step の (A) `scale_loss(2^8)` →
/// backward → `unscale_grads` と (B) 非スケール backward の
/// weight／bias 勾配が `f32::to_bits()` で完全一致することを固定する。
///
/// **根拠**: 2 のべき乗スケールは f32 の乗除・FMA・f64 蓄積のいずれとも
/// 丸めが可換（over／underflow を除く。`unscale_grads` doc 参照）。
/// `crates/autodiff/tests/nn_optim_amp.rs::scale_loss_backward_unscale_matches_unscaled_backward_bit_exact`
/// は `naive_ops` 経由のため、facade `tape()`（`CpuBackendOps`: BLIS
/// GEMM・rayon `mse_loss_backward`・`Op::LinearAct` 融合・`reduce_bias_grad`
/// f64 蓄積）で同じ性質が成立することを本テストで初めて固定する。
///
/// 不一致が出た場合は tolerance を持ち込まず（内部クレートも import
/// しない）、facade レベルで再現することの記録として扱う。
#[test]
fn amp_one_step_grads_match_non_amp_bit_exact_on_cpu() {
    const SCALE: f32 = 256.0; // 2^8
    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    // (A) scale_loss → backward → unscale_grads。
    let model_a = build_model();
    let scaled_bits: Vec<Vec<u32>> = {
        let tape = fandhe_ai::tape();
        let bound = model_a.bind(&tape);
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = bound
            .forward(&tape, &x)
            .unwrap_or_else(|e| panic!("test fixture: forward(A) が失敗した: {e}"));
        let loss = pred
            .mse_loss(&y)
            .unwrap_or_else(|e| panic!("test fixture: mse_loss(A) が失敗した: {e}"));
        let scaled_loss = fandhe_ai::optim::scale_loss(&loss, SCALE)
            .unwrap_or_else(|e| panic!("test fixture: scale_loss が失敗した: {e}"));
        let grads = tape
            .backward(&scaled_loss)
            .unwrap_or_else(|e| panic!("test fixture: backward(A) が失敗した: {e}"));
        let grad_refs = bound
            .trainable_grads(&grads)
            .unwrap_or_else(|e| panic!("test fixture: trainable_grads(A) が失敗した: {e}"));
        let unscale_result = fandhe_ai::optim::unscale_grads(&grad_refs, SCALE)
            .unwrap_or_else(|e| panic!("test fixture: unscale_grads が失敗した: {e}"));
        assert!(
            !unscale_result.found_non_finite,
            "test fixture: この形状・スケールで非有限勾配は発生しないはず"
        );
        unscale_result
            .grads
            .iter()
            .map(|t| t.host_slice().iter().map(|v| v.to_bits()).collect())
            .collect()
    };

    // (B) 非スケール backward。
    let model_b = build_model();
    let unscaled_bits: Vec<Vec<u32>> = {
        let tape = fandhe_ai::tape();
        let bound = model_b.bind(&tape);
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = bound
            .forward(&tape, &x)
            .unwrap_or_else(|e| panic!("test fixture: forward(B) が失敗した: {e}"));
        let loss = pred
            .mse_loss(&y)
            .unwrap_or_else(|e| panic!("test fixture: mse_loss(B) が失敗した: {e}"));
        let grads = tape
            .backward(&loss)
            .unwrap_or_else(|e| panic!("test fixture: backward(B) が失敗した: {e}"));
        let grad_refs = bound
            .trainable_grads(&grads)
            .unwrap_or_else(|e| panic!("test fixture: trainable_grads(B) が失敗した: {e}"));
        grad_refs
            .iter()
            .map(|t| t.host_slice().iter().map(|v| v.to_bits()).collect())
            .collect()
    };

    assert_eq!(
        scaled_bits.len(),
        unscaled_bits.len(),
        "test fixture: 勾配テンソルの本数が一致するはず（weight/bias × 2 層）"
    );
    for (i, (a, b)) in scaled_bits.iter().zip(unscaled_bits.iter()).enumerate() {
        assert_eq!(
            a, b,
            "勾配テンソル #{i}: scale_loss→backward→unscale と非スケール backward の\
             勾配が bit 単位で一致しない（2 のべき乗スケールの丸め可換性が facade 経路で崩れている）"
        );
    }
}

// =====================================================================
// 非有限勾配時の skip・backoff
// =====================================================================

/// `unscale` の結果が非有限を含む場合、`should_skip_step()` が `true` に
/// なり、呼び出し元が `clip_grad_norm`／optimizer step をスキップし
/// `scaler.update(true)` を呼ぶと scale が `backoff_factor` 倍されること
/// を固定する（適用順序契約「skip 判定は clip より前に行う」の効果）。
#[test]
fn amp_non_finite_step_is_skipped_and_scaler_backs_off() {
    let config = GradScalerConfig {
        init_scale: 256.0,
        growth_factor: 2.0,
        backoff_factor: 0.5,
        growth_interval: 1000,
    };
    let mut scaler = GradScaler::new(config)
        .unwrap_or_else(|e| panic!("test fixture: GradScaler::new が失敗した: {e}"));
    assert_eq!(scaler.scale(), 256.0);

    let n = 4usize;
    let inf_grad = Tensor::new(vec![f32::INFINITY; n], &[n])
        .unwrap_or_else(|e| panic!("test fixture: inf_grad の構築に失敗: {e}"));
    let unscale_result = scaler
        .unscale(&[&inf_grad])
        .unwrap_or_else(|e| panic!("test fixture: unscale が失敗した: {e}"));

    assert!(
        unscale_result.should_skip_step(),
        "非有限勾配は should_skip_step()=true になるはず"
    );

    // 適用順序契約: skip 時は clip_grad_norm／optimizer step を呼ばず、
    // 最後に scaler.update(true) のみを呼ぶ。
    scaler
        .update(true)
        .unwrap_or_else(|e| panic!("test fixture: GradScaler::update(true) が失敗した: {e}"));

    assert_eq!(
        scaler.scale(),
        128.0,
        "backoff 後の scale は init_scale * backoff_factor = 256.0 * 0.5 = 128.0 のはず"
    );
    assert_eq!(
        scaler.growth_tracker(),
        0,
        "backoff は growth_tracker を 0 へリセットするはず"
    );
}
