//! `fandhe_ai::optim::ReduceLrOnPlateau`（イシュー #1746・親 #1611）が
//! facade のみを通じて学習ループを駆動できることの統合テスト
//! （受入基準の読み替え (c)。`crates/facade/tests/optim_train_loop.rs`
//! と同じくモデル・データ・シードは共通のフィクスチャ様式を踏襲する）。
//!
//! **本ファイルは `fandhe_ai` と `bench_harness::rng` 以外を import しない**
//! （`optim_train_loop.rs` と同じ契約。facade のみへの依存で学習ループが
//! 書けることを import 文そのもので裏付ける）。
//!
//! **適用順序契約**: 1 学習ステップは `backward → clip → optimizer step`
//! （`fandhe_ai::optim` モジュール doc「適用順序契約」節「AMP を使わない
//! 場合」を参照。本ファイルは AMP 非使用の既存経路を踏襲する）。
//!
//! **決定的シード**: モデル・データ・シードは `optim_train_loop.rs` と
//! 同一（`.claude/rules/coding-rust.md`「学習系回帰テストには決定的
//! シード設定ユーティリティを使う」）。
//!
//! 実機（CUDA/Metal）非依存のため `#[ignore]` 分離は行わない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::Sequential;
use fandhe_ai::optim::{
    LrScheduler, ReduceLrOnPlateau, ReduceLrOnPlateauConfig, Sgd, SgdConfig, clip_grad_norm,
};

const BATCH: usize = 4;
const D_IN: usize = 8;
const D_HIDDEN: usize = 16;
const D_OUT: usize = 4;

const SEED_DATA: u64 = 0xC0FFEE;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `optim_train_loop.rs::gen_regression_data` と同一生成順
/// （`x`: `[BATCH, D_IN]`・`y`: `[BATCH, D_OUT]`）。
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

/// `ReduceLrOnPlateau::step(loss)` の返り値で毎 step `SgdConfig` を
/// 作り直し、`Sgd`（momentum 無し）で学習ループを回す。`backward → clip
/// → optimizer step` の適用順序契約（モジュール冒頭 doc）を固定する。
///
/// **borrow スコープ**: `SequentialVars`（`bound`）は `&model`／`&tape` を
/// 借用するため、`apply_parameters`（`&mut model`）を呼ぶ前に必ず
/// ブロックを抜けて借用を解放する（`optim_train_loop.rs` と同じ構成）。
#[test]
fn reduce_lr_on_plateau_drives_sgd_config_via_facade_only() {
    const STEPS: usize = 20;
    const BASE_LR: f32 = 0.1;
    const MAX_NORM: f32 = 1.0;

    // patience を小さく取り、収束が進むにつれ改善が鈍化した段階で
    // 減衰が発火しうる設定にする（tolerance を追加しない。
    // `.claude/rules/coding-rust.md`）。
    let config = ReduceLrOnPlateauConfig {
        patience: 2,
        ..ReduceLrOnPlateauConfig::default()
    };
    let mut scheduler = ReduceLrOnPlateau::new(BASE_LR, config)
        .unwrap_or_else(|e| panic!("test fixture: ReduceLrOnPlateau::new が失敗した: {e}"));

    let mut model = build_model();
    let (x_data, y_data) = gen_regression_data(SEED_DATA);
    let mut log = Vec::with_capacity(STEPS);
    let mut lr_log = Vec::with_capacity(STEPS);

    for _ in 0..STEPS {
        // `lr_at` は状態を進めない（現在値の読み出しのみ。モジュール
        // 冒頭 doc「stateless 契約に対する唯一の例外」節）ため、この
        // step で使う lr は前 step までに `step(metric)` で確定した値。
        let lr = scheduler.lr_at(0);
        lr_log.push(lr);
        let mut sgd = Sgd::new(SgdConfig::new(lr))
            .unwrap_or_else(|e| panic!("test fixture: Sgd::new が失敗した: {e}"));

        let loss_value = {
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
            let loss_value = scalar(&loss.to_tensor());
            log.push(loss_value);

            let grads = tape
                .backward(&loss)
                .unwrap_or_else(|e| panic!("test fixture: backward が失敗した: {e}"));
            let grad_refs = bound
                .trainable_grads(&grads)
                .unwrap_or_else(|e| panic!("test fixture: trainable_grads が失敗した: {e}"));

            // 適用順序契約: backward → clip → optimizer step。
            // `clip_grad_norm` は `grad_refs` を変更せず、クリップ後の
            // 勾配は戻り値 `ClipGradResult::grads` に入る
            // （`optim_train_loop.rs` と同じ契約）ため、`Sgd::step` には
            // 必ず `clip_result.grads` から作った参照列を渡す。
            let clip_result = clip_grad_norm(&grad_refs, MAX_NORM)
                .unwrap_or_else(|e| panic!("test fixture: clip_grad_norm が失敗した: {e}"));
            let clipped_grad_refs: Vec<&Tensor<f32>> = clip_result.grads.iter().collect();

            let param_refs = model.trainable_parameters();
            let updated = sgd
                .step(&param_refs, &clipped_grad_refs)
                .unwrap_or_else(|e| panic!("test fixture: Sgd::step が失敗した: {e}"));
            drop(bound);
            (updated, loss_value)
        };
        let (updated, loss_value) = loss_value;
        model
            .apply_parameters(updated)
            .unwrap_or_else(|e| panic!("test fixture: apply_parameters が失敗した: {e}"));

        // 検証指標（本テストでは学習 loss を代用）を観測して次 step の
        // lr を確定する。
        let _ = scheduler
            .step(loss_value)
            .unwrap_or_else(|e| panic!("test fixture: ReduceLrOnPlateau::step が失敗した: {e}"));
    }

    assert_eq!(log.len(), STEPS);
    let initial = log[0];
    let final_loss = *log.last().unwrap_or_else(|| unreachable!("log は空でない"));
    assert!(final_loss.is_finite(), "final loss が非有限: {final_loss}");
    assert!(
        final_loss < initial,
        "loss did not decrease: initial={initial} final={final_loss}"
    );

    // lr は単調非増加（減衰のみが起き上昇はしない）であることを固定
    // する（`ReduceLrOnPlateau` は下げるだけの契約）。
    assert_eq!(lr_log.len(), STEPS);
    for pair in lr_log.windows(2) {
        assert!(pair[1] <= pair[0], "lr は単調非増加のはず: {:?}", lr_log);
    }
    assert!(
        lr_log[0] == BASE_LR,
        "初回 step の lr は base_lr のはず: {}",
        lr_log[0]
    );

    // `&dyn LrScheduler` 経由でも到達可能であることの固定
    // （`lr_at` は状態を進めないだけで trait 実装自体は成立する）。
    let dyn_scheduler: &dyn LrScheduler = &scheduler;
    assert_eq!(dyn_scheduler.lr_at(999), scheduler.current_lr());
}
