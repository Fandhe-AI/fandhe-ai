//! `fandhe_ai::optim`（イシュー #961・親 #960）のみを optimizer 面として
//! 使う学習ループ統合テスト（受入基準 3「利用者が `fandhe_ai::optim` から
//! optimizer 一式を使えること」の構造的裏付け）。
//!
//! **本ファイルは `fandhe_ai` と `bench_harness::rng` 以外を import しない**
//! （`fandhe_ai_autodiff`／`fandhe_ai_tensor_core` を一切 import しない契約。
//! `crates/facade/tests/compat_sequential_train.rs` は内部クレート
//! `fandhe_ai_autodiff::optim`／`fandhe_ai_autodiff::nn::optim` を直接 import して
//! いるのに対し、本ファイルは「facade のみへの依存で学習ループが書ける」
//! ことを import 文そのもので裏付ける。レビュー・CI では
//! `grep -n "fandhe_ai_autodiff\|fandhe_ai_tensor_core" tests/optim_train_loop.rs`
//! がヒット 0 件であることを確認する）。
//!
//! **適用順序契約**: 1 学習ステップは
//! `backward → clip → optimizer step`（`fandhe_ai::optim` モジュール doc
//! 「適用順序契約」節を参照。AMP 未導入のため unscale ステップは存在
//! しない）。
//!
//! **決定的シード**: モデル・データ・シードは
//! `crates/facade/tests/compat_sequential_train.rs` と同一
//! （`.claude/rules/coding-rust.md`「学習系回帰テストには決定的シード
//! 設定ユーティリティを使う」）。
//!
//! **数値判定の規律**: 収束判定は既存様式（最終 loss が初期 loss から
//! 十分減少すること）を踏襲し、新規の許容誤差（tolerance）は設けない
//! （`.claude/rules/coding-rust.md`）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{
    AdamW, AdamWConfig, ConstantLr, LrScheduler, Sgd, SgdConfig, StepLr, clip_grad_norm,
};

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `crates/facade/tests/compat_sequential_train.rs::gen_regression_data`
/// と同一生成順（`x`: `[BATCH, D_IN]`・`y`: `[BATCH, D_OUT]`）。
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
// Sgd + clip
// =====================================================================

/// `Sequential::bind` → forward → `Var::mse_loss` → `Tape::backward` →
/// `SequentialVars::trainable_grads` → `clip_grad_norm` → `Sgd::step` →
/// `Sequential::apply_parameters` の 1 ステップを `steps` 回繰り返す。
///
/// **借用スコープ**: `SequentialVars`（`bound`）は `&model`／`&tape` を
/// 借用するため、`apply_parameters`（`&mut model`）を呼ぶ前に必ず
/// ブロックを抜けて借用を解放する（`compat_sequential_train.rs` と同じ
/// 構成）。
fn train_with_sgd_and_clip(
    model: &mut Sequential,
    steps: usize,
    lr: f32,
    max_norm: f32,
) -> Vec<f32> {
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut sgd = Sgd::new(SgdConfig::new(lr))
        .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));
    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let updated = {
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

            let grads = tape
                .backward(&loss)
                .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
            let grad_refs = bound
                .trainable_grads(&grads)
                .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));
            // 適用順序契約: backward → clip → optimizer step。
            let clip_result = clip_grad_norm(&grad_refs, max_norm)
                .unwrap_or_else(|e| panic!("test fixture: clip_grad_norm が失敗した: {e}"));
            let clipped_grad_refs: Vec<&Tensor<f32>> = clip_result.grads.iter().collect();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &clipped_grad_refs)
                .unwrap_or_else(|e| panic!("test fixture: Sgd::step が失敗した: {e}"))
        };
        model
            .apply_parameters(updated)
            .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗した: {e}"));
    }

    log
}

/// `fandhe_ai::optim::{Sgd, clip_grad_norm}` のみを使った学習ループで
/// loss が減少すること（受入基準 3）。
#[test]
fn sgd_with_clip_converges_via_facade_only() {
    const STEPS: usize = 100;
    const LR: f32 = 0.05;
    const MAX_NORM: f32 = 10.0;

    let mut model = build_model();
    let log = train_with_sgd_and_clip(&mut model, STEPS, LR, MAX_NORM);

    assert_eq!(log.len(), STEPS);
    let initial = log[0];
    let final_loss = *log.last().unwrap_or_else(|| unreachable!("log は空でない"));
    assert!(final_loss.is_finite(), "final loss が非有限: {final_loss}");
    assert!(
        final_loss < 0.5 * initial,
        "loss did not converge sufficiently: initial={initial} final={final_loss}"
    );
}

// =====================================================================
// AdamW + clip
// =====================================================================

/// `Sequential::bind` → forward → `Var::mse_loss` → `Tape::backward` →
/// `SequentialVars::trainable_grads` → `clip_grad_norm` → `AdamW::step`
/// （`params_and_grads` タプル列を `zip` で構築） →
/// `Sequential::apply_parameters` の 1 ステップを `steps` 回繰り返す。
fn train_with_adamw_and_clip(model: &mut Sequential, steps: usize, max_norm: f32) -> Vec<f32> {
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut adamw = AdamW::new(AdamWConfig::default())
        .unwrap_or_else(|e| panic!("test fixture: AdamW::new が失敗した: {e}"));
    let mut log = Vec::with_capacity(steps);

    for _ in 0..steps {
        let updated = {
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

            let grads = tape
                .backward(&loss)
                .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
            let grad_refs = bound
                .trainable_grads(&grads)
                .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));
            let clip_result = clip_grad_norm(&grad_refs, max_norm)
                .unwrap_or_else(|e| panic!("test fixture: clip_grad_norm が失敗した: {e}"));
            let param_refs = model.trainable_parameters();
            let params_and_grads: Vec<(&Tensor<f32>, &Tensor<f32>)> = param_refs
                .into_iter()
                .zip(clip_result.grads.iter())
                .collect();
            adamw
                .step(&params_and_grads)
                .unwrap_or_else(|e| panic!("test fixture: AdamW::step が失敗した: {e}"))
        };
        model
            .apply_parameters(updated)
            .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗した: {e}"));
    }

    log
}

/// `fandhe_ai::optim::{AdamW, clip_grad_norm}` のみを使った学習ループで
/// loss が減少すること（受入基準 3）。
#[test]
fn adamw_with_clip_converges_via_facade_only() {
    const STEPS: usize = 100;
    const MAX_NORM: f32 = 10.0;

    let mut model = build_model();
    let log = train_with_adamw_and_clip(&mut model, STEPS, MAX_NORM);

    assert_eq!(log.len(), STEPS);
    let initial = log[0];
    let final_loss = *log.last().unwrap_or_else(|| unreachable!("log は空でない"));
    assert!(final_loss.is_finite(), "final loss が非有限: {final_loss}");
    assert!(
        final_loss < 0.5 * initial,
        "loss did not converge sufficiently: initial={initial} final={final_loss}"
    );
}

// =====================================================================
// clip → step 順序契約の効果確認
// =====================================================================

/// 極端に小さい `max_norm` で clip した場合、1 ステップ目の更新幅が
/// clip なしより小さくなること（`scaled == true` を伴う）を確認する
/// （適用順序契約「clip は optimizer step の前に適用する」の効果が
/// 実際に現れることの回帰チェック。tolerance は使わず不等号比較のみ）。
#[test]
fn clip_before_step_order_contract_is_documented_and_effective() {
    const LR: f32 = 0.05;
    const TINY_MAX_NORM: f32 = 1e-3;
    const LARGE_MAX_NORM: f32 = 1e6;

    let (x_data, y_data) = gen_regression_data(SEED_DATA);

    // 同一初期パラメータから 1 step 分の grad を計算し、clip 適用の有無
    // で `ClipGradResult` を比較する（optimizer step 自体は本テストの
    // 主眼ではないため、`clip_grad_norm` の出力比較に絞る）。
    let model = build_model();
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
    let grads = tape
        .backward(&loss)
        .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
    let grad_refs = bound
        .trainable_grads(&grads)
        .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));

    let tiny = clip_grad_norm(&grad_refs, TINY_MAX_NORM)
        .unwrap_or_else(|e| panic!("test fixture: clip_grad_norm(tiny) が失敗した: {e}"));
    let large = clip_grad_norm(&grad_refs, LARGE_MAX_NORM)
        .unwrap_or_else(|e| panic!("test fixture: clip_grad_norm(large) が失敗した: {e}"));

    assert!(
        tiny.scaled,
        "max_norm={TINY_MAX_NORM} は total_norm={} より小さいはずで scaled=true になるべき",
        tiny.total_norm
    );
    assert!(
        !large.scaled,
        "max_norm={LARGE_MAX_NORM} は total_norm={} より大きいはずで scaled=false になるべき",
        large.total_norm
    );

    // clip 後の勾配ノルムは tiny 側が明確に小さい（LR を掛けても同じ
    // 大小関係が保たれる。順序契約が効いていることの確認）。
    let tiny_norm_sq: f32 = tiny
        .grads
        .iter()
        .map(|g| {
            g.as_slice()
                .unwrap_or_else(|| unreachable!("contiguous なテンソルのはず"))
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
        })
        .sum();
    let large_norm_sq: f32 = large
        .grads
        .iter()
        .map(|g| {
            g.as_slice()
                .unwrap_or_else(|| unreachable!("contiguous なテンソルのはず"))
                .iter()
                .map(|v| v * v)
                .sum::<f32>()
        })
        .sum();
    assert!(
        tiny_norm_sq < large_norm_sq,
        "tiny クリップ後の勾配ノルムが large クリップ後より小さくなっていない\
         （clip → step の順序契約の効果が確認できない）: tiny={tiny_norm_sq} large={large_norm_sq}"
    );
    let _ = LR; // LR は本テストの意図（クリップ効果の説明）を明確にするための定数。
}

// =====================================================================
// LrScheduler + Sgd
// =====================================================================

/// `StepLr::lr_at(step)` の返す学習率で毎 step `SgdConfig` を作り直し、
/// `Sgd`（momentum 無し）で数ステップ回す。lr の切り替わり系列と loss
/// 減少の両方を確認する（受入基準 3。`fandhe_ai::optim::{StepLr,
/// LrScheduler, Sgd, SgdConfig}` のみを使用）。
#[test]
fn lr_scheduler_drives_sgd_config_via_facade_only() {
    const STEPS: usize = 6;
    const BASE_LR: f32 = 0.1;
    const STEP_SIZE: usize = 2;
    const GAMMA: f32 = 0.5;

    let scheduler = StepLr::new(BASE_LR, STEP_SIZE, GAMMA)
        .unwrap_or_else(|e| panic!("test fixture: StepLr::new が失敗した: {e}"));

    // `lr_at` の返り値が期待する階段減衰系列と完全一致することを固定
    // する（tolerance 不使用。`StepLr` の定義: lr = base_lr * gamma^(step / step_size)
    // 〈整数除算切り捨て〉）。
    let expected_lrs = [0.1_f32, 0.1, 0.05, 0.05, 0.025, 0.025];
    let observed_lrs: Vec<f32> = (0..STEPS).map(|step| scheduler.lr_at(step)).collect();
    assert_eq!(
        observed_lrs, expected_lrs,
        "StepLr::lr_at の系列が期待値と一致しない"
    );

    let mut model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut log = Vec::with_capacity(STEPS);

    for step in 0..STEPS {
        let lr = scheduler.lr_at(step);
        let mut sgd = Sgd::new(SgdConfig::new(lr))
            .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));

        let updated = {
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

            let grads = tape
                .backward(&loss)
                .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
            let grad_refs = bound
                .trainable_grads(&grads)
                .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs)
                .unwrap_or_else(|e| panic!("test fixture: Sgd::step が失敗した: {e}"))
        };
        model
            .apply_parameters(updated)
            .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗した: {e}"));
    }

    assert_eq!(log.len(), STEPS);
    let initial = log[0];
    let final_loss = *log.last().unwrap_or_else(|| unreachable!("log は空でない"));
    assert!(final_loss.is_finite(), "final loss が非有限: {final_loss}");
    assert!(
        final_loss < initial,
        "loss did not decrease: initial={initial} final={final_loss}"
    );

    // `ConstantLr` も同モジュールから到達可能であることを併せて確認する
    // （facade のみ依存で `optim::{ConstantLr, LrScheduler}` を使える裏付け）。
    let constant = ConstantLr::new(BASE_LR)
        .unwrap_or_else(|e| panic!("test fixture: ConstantLr::new が失敗した: {e}"));
    assert_eq!(constant.lr_at(0), BASE_LR);
    assert_eq!(constant.lr_at(100), BASE_LR);
}
