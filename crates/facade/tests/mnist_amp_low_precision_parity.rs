//! AC-b（イシュー #1961・親 #1958。実装計画 §5.3）: MNIST 規模（784→
//! 256(ReLU)→10・batch 64）の学習で、低精度 AMP（`compile_with_amp`）の
//! loss 系列が f32（`compile`）の loss 系列と REQ-2 統一複合判定
//! （`fandhe_ai_backend_cpu::parity::compare`。定数・判定式は既存のまま）
//! 内であることを検証する事前登録判定。
//!
//! **事前登録判定規則**（結果を見る前に固定。`.claude/rules/out-of-
//! scope-tracking.md`・`coding-rust.md`「バックエンド間数値一致テストの
//! 許容誤差を単独で緩和しない」の精神に倣い、FAIL しても本ファイルの
//! シナリオ・判定式・tolerance は変更しない）:
//!
//! - シナリオ: モデル 784→256(ReLU)→10（`SEED_L1=0x1111_1111`・
//!   `SEED_L2=0x2222_2222`）、データ `x=[64,784]`（`Xorshift64Star
//!   (0xDA7A_0001)`）・`y=[64,10]`（`0xDA7A_0002`）、`Sgd(lr=0.01)`、
//!   `Loss::Mse`、`FitConfig::new(20, 64)`（shuffle なし・同一バッチ
//!   20 step）、`GradScalerConfig::default()`、CPU（`fandhe_ai::tape`）。
//! - 判定: `History.loss`（20 点）全点を f32 系列 vs AMP 系列で
//!   `compare`（統一複合判定）にかけ、`fail_count == 0`・両系列とも
//!   全点有限・`loss[19] < loss[0]` を assert する。
//! - 主判定 = F16（[`amp_f16_matches_f32_within_req2_composite_tolerance`]）。
//!   副判定 = Bf16（[`amp_bf16_matches_f32_within_req2_composite_tolerance`]）。
//! - backward は AMP 有無に関わらず常に f32 のため（`linear_forward_
//!   low_precision` doc「backward は常に f32」節）、GradScaler 結線
//!   自体の正しさは `compat_sequential_fit_amp.rs` の手動ループ bit
//!   一致テストが独立に証明する。したがって本テストが FAIL する場合、
//!   原因は低精度 forward の丸め（f16: 仮数 10bit／bf16: 仮数 7bit・
//!   784 項内積の丸めと 20 step の軌道差）に帰属するものとして記録
//!   する（この帰属自体は本ファイルの assert では検証しない・実測は
//!   `docs/autodiff-low-precision-linear-design.md` §7 へ記録する）。
//!
//! `fandhe_ai` のみでは `fandhe_ai_backend_cpu::parity::compare`
//! （REQ-2 統一複合判定の実体）へ到達できないため、本ファイルは
//! `fandhe_ai_backend_cpu`／`fandhe_ai_autodiff`／`fandhe_ai_tensor_core`
//! を直接 import する（実装計画 §3 の「内部クレート import 可」区分）。
//!
//! CPU 決定的のため 1 回の実行で確定する（5 run 中央値は実機性能実測
//! 向けの規則であり本判定には該当しない）。fit は CPU `tape()` 固定の
//! ため実機（CUDA／Metal）は対象外。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::{AmpConfig, AmpDType, FitConfig, Loss, Optimizer, Sequential};
use fandhe_ai::optim::SgdConfig;
use fandhe_ai_backend_cpu::parity::compare;

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const STEPS: usize = 20;
const LR: f32 = 0.01;

const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;
const SEED_X: u64 = 0xDA7A_0001;
const SEED_Y: u64 = 0xDA7A_0002;

fn gen_data() -> (Tensor<f32>, Tensor<f32>) {
    let x = Xorshift64Star::new(SEED_X).fill_vec(BATCH * D_IN);
    let y = Xorshift64Star::new(SEED_Y).fill_vec(BATCH * D_OUT);
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

fn run_f32(x: &Tensor<f32>, y: &Tensor<f32>) -> Vec<f32> {
    let mut model = build_model();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(LR)), Loss::Mse)
        .unwrap_or_else(|e| panic!("test fixture: compile が失敗した: {e}"));
    model
        .fit(x, y, FitConfig::new(STEPS, BATCH))
        .unwrap_or_else(|e| panic!("test fixture: fit(f32) が失敗した: {e}"))
        .loss
}

fn run_amp(x: &Tensor<f32>, y: &Tensor<f32>, dtype: AmpDType) -> Vec<f32> {
    let mut model = build_model();
    model
        .compile_with_amp(
            Optimizer::Sgd(SgdConfig::new(LR)),
            Loss::Mse,
            AmpConfig::new(dtype),
        )
        .unwrap_or_else(|e| panic!("test fixture: compile_with_amp が失敗した: {e}"));
    model
        .fit(x, y, FitConfig::new(STEPS, BATCH))
        .unwrap_or_else(|e| panic!("test fixture: fit(amp) が失敗した: {e}"))
        .loss
}

/// 事前登録判定規則（本ファイル冒頭コメント）を機械的に検証する共通
/// ヘルパー。`dtype_label` は panic メッセージへの表示用。
fn assert_amp_matches_f32_within_composite_tolerance(dtype: AmpDType, dtype_label: &str) {
    let (x, y) = gen_data();
    let loss_f32 = run_f32(&x, &y);
    let loss_amp = run_amp(&x, &y, dtype);

    assert_eq!(loss_f32.len(), STEPS);
    assert_eq!(loss_amp.len(), STEPS);
    assert!(
        loss_f32.iter().all(|v| v.is_finite()),
        "f32 系列に非有限値がある（dtype={dtype_label}）: {loss_f32:?}"
    );
    assert!(
        loss_amp.iter().all(|v| v.is_finite()),
        "AMP（{dtype_label}）系列に非有限値がある: {loss_amp:?}"
    );
    assert!(
        loss_f32[STEPS - 1] < loss_f32[0],
        "f32 系列が収束しない（dtype={dtype_label}）: first={} last={}",
        loss_f32[0],
        loss_f32[STEPS - 1]
    );
    assert!(
        loss_amp[STEPS - 1] < loss_amp[0],
        "AMP（{dtype_label}）系列が収束しない: first={} last={}",
        loss_amp[0],
        loss_amp[STEPS - 1]
    );

    let report = compare(&loss_f32, &loss_amp).unwrap_or_else(|e| {
        panic!("test fixture: compare（loss 長さ不一致のはず。dtype={dtype_label}）: {e}")
    });
    eprintln!(
        "AC-b（dtype={dtype_label}）: fail_count={}/{} max_abs_diff={:.6e} \
         mean_abs_diff={:.6e} max_rel_err={:.6e} f32={loss_f32:?} amp={loss_amp:?}",
        report.fail_count,
        report.total,
        report.max_abs_diff,
        report.mean_abs_diff,
        report.max_rel_err
    );
    assert_eq!(
        report.fail_count, 0,
        "AC-b 未達（dtype={dtype_label}）: {report:?}"
    );
}

/// 主判定（F16）。AC-b の合否はこのテストで決める。
#[test]
fn amp_f16_matches_f32_within_req2_composite_tolerance() {
    assert_amp_matches_f32_within_composite_tolerance(AmpDType::F16, "F16");
}

/// 副判定（Bf16）。主判定と同一シナリオ・同一判定式。
#[test]
fn amp_bf16_matches_f32_within_req2_composite_tolerance() {
    assert_amp_matches_f32_within_composite_tolerance(AmpDType::Bf16, "Bf16");
}
