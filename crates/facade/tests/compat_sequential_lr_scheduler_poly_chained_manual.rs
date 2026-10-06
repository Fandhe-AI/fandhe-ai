//! イシュー #2659（親 #2657）の `fandhe_ai_autodiff::nn::optim::{PolynomialLr,
//! ChainedScheduler}` facade 公開保留（`crates/facade/src/lib.rs::
//! LrSchedulerPolyChainedHoldDoctestGuard`）下での、公開 compat API
//! （`Sequential::fit_with_callbacks`・`LrSchedule::per_epoch`）と内部スケジューラの
//! 手動結線の契約テスト。
//!
//! **本ファイルは 2 型を内部クレートから直接 import する契約ファイル**であり
//! （`compat_sequential_swa_manual.rs` と同型の位置づけ）、facade 再エクスポートのみを
//! 使う契約のテストファイルへ混入させない。`LrScheduler` trait は facade
//! （`fandhe_ai::optim::LrScheduler`）と同一の型であり、`LrSchedule::per_epoch` へそのまま
//! 渡せる（`docs/autodiff-lr-scheduler-poly-chained-decision.md` §8）。
//!
//! 検査項目: `history.lr` が各 epoch の `lr_at(epoch)` と bit 一致すること。flaky になりうる
//! accuracy 比較は入れない。ホスト計算のみのため `#[ignore]` 分離は行わない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Tensor;
use fandhe_ai::compat::{Callback, FitConfig, Loss, LrSchedule, Optimizer, Sequential};
use fandhe_ai::optim::{ExponentialLr, LrScheduler, SgdConfig};
use fandhe_ai_autodiff::nn::optim::{ChainedScheduler, PolynomialLr};

const N: usize = 16;
const D_IN: usize = 4;
const D_OUT: usize = 2;
const EPOCHS: usize = 6;
const BASE_LR: f32 = 0.05;

fn data() -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(0xC0FFEE);
    let x = rng.fill_vec(N * D_IN);
    let y = rng.fill_vec(N * D_OUT);
    (
        Tensor::new(x, &[N, D_IN]).unwrap(),
        Tensor::new(y, &[N, D_OUT]).unwrap(),
    )
}

fn history_lr(sched: impl LrScheduler + 'static) -> Vec<f32> {
    let (x, y) = data();
    let mut model = Sequential::new()
        .add_linear(D_IN, 8, 0x1111)
        .unwrap()
        .add_relu()
        .add_linear(8, D_OUT, 0x2222)
        .unwrap();
    model
        .compile(Optimizer::Sgd(SgdConfig::new(BASE_LR)), Loss::Mse)
        .unwrap();
    let mut callbacks = [Callback::LrSchedule(LrSchedule::per_epoch(sched))];
    model
        .fit_with_callbacks(&x, &y, FitConfig::new(EPOCHS, N), None, &mut callbacks)
        .unwrap()
        .lr
}

#[test]
fn polynomial_lr_drives_fit_via_per_epoch() {
    let make = || PolynomialLr::new(BASE_LR, 4, 2.0).unwrap();
    let lrs = history_lr(make());
    let reference = make();
    assert_eq!(lrs.len(), EPOCHS);
    for (e, lr) in lrs.iter().enumerate() {
        assert_eq!(lr.to_bits(), reference.lr_at(e).to_bits(), "epoch {e}");
    }
    // total_iters 到達後は 0（set_lr は 0 を受理し学習が止まるだけ）。
    assert_eq!(lrs[4], 0.0);
}

#[test]
fn chained_scheduler_drives_fit_via_per_epoch() {
    let make = || {
        ChainedScheduler::new(
            BASE_LR,
            vec![
                Box::new(PolynomialLr::new(BASE_LR, 8, 1.0).unwrap()),
                Box::new(ExponentialLr::new(BASE_LR, 0.5).unwrap()),
            ],
        )
        .unwrap()
    };
    let lrs = history_lr(make());
    let reference = make();
    assert_eq!(lrs.len(), EPOCHS);
    for (e, lr) in lrs.iter().enumerate() {
        assert_eq!(lr.to_bits(), reference.lr_at(e).to_bits(), "epoch {e}");
    }
}
