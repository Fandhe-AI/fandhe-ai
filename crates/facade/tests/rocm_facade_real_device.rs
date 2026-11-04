//! facade 経由の ROCm 実機テスト（イシュー #3853・親 #3711。実測は Phase 10 の #3867）。
//!
//! **役割**: `fandhe_ai::available_devices`／`tape_for(Device::Rocm(0))` の結線と、facade 経由の
//! matmul・小さな MLP 1 ステップの CPU parity（REQ-2 統一複合判定 `compare`）を実機で確かめる。
//! 4 本とも `#[ignore]`（ROCm 搭載ホストでだけ意味がある）。テスト名は設計記録
//! `docs/backend-rocm-device-exposure-decision.md` §10.4 と `docs/real-hardware-verification-env.md`
//! §9.5 が固定しており、変更しない。
//!
//! **本 PR の時点では未実測**。Phase 10 の #3867 が `<rocm-node>` で実行して記録する。判定は
//! `compare` だけで、tolerance・baseline は追加も変更もしない（bit の不一致数などは観測値として
//! 出力するだけ）。
//!
//! 実行例（実機のみ。1 本ずつ別プロセス）:
//! ```sh
//! cargo test -p fandhe-ai --test rocm_facade_real_device -- --ignored --exact --nocapture \
//!   rocm_facade_matmul_parity_vs_cpu
//! ```

use fandhe_ai::Device;
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_backend_cpu::compare;
use fandhe_ai_tensor_core::Tensor;

/// 決定的な擬似データ（外部乱数に依存しない）。
fn data(n: usize, salt: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2_654_435_761).wrapping_add(salt) >> 8;
            (x % 2001) as f32 / 1000.0 - 1.0
        })
        .collect()
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .to_vec()
}

fn assert_compare(context: &str, rocm: &[f32], cpu: &[f32]) {
    let report = compare(rocm, cpu).expect("compare");
    println!(
        "{context}: fail_count={}/{} max_abs_diff={:.3e} max_rel_err={:.3e}",
        report.fail_count, report.total, report.max_abs_diff, report.max_rel_err
    );
    assert!(report.passes(), "{context}: REQ-2 統一複合判定に失敗");
}

#[test]
#[ignore = "ROCm 実機（gfx942）が必要。実測は Phase 10 の #3867"]
fn rocm_available_devices_lists_rocm0_and_tape_for_ok() {
    let devices = fandhe_ai::available_devices();
    assert!(devices.contains(&Device::Rocm(0)), "{devices:?}");
    assert_eq!(
        devices
            .iter()
            .filter(|d| matches!(d, Device::Rocm(_)))
            .count(),
        1
    );
    fandhe_ai::tape_for(Device::Rocm(0)).expect("tape_for(Rocm(0))");
}

#[test]
#[ignore = "ROCm 実機（gfx942）が必要。実測は Phase 10 の #3867"]
fn rocm_facade_matmul_parity_vs_cpu() {
    let (m, k, n) = (33, 47, 29);
    let a = Tensor::new(data(m * k, 1), &[m, k]).expect("a");
    let b = Tensor::new(data(k * n, 2), &[k, n]).expect("b");
    let run = |device: Option<Device>| {
        let tape = match device {
            Some(d) => fandhe_ai::tape_for(d).expect("tape_for"),
            None => fandhe_ai::tape(),
        };
        let y = tape.var(&a).matmul(&tape.var(&b)).expect("matmul");
        dense(&y.to_tensor())
    };
    assert_compare("matmul", &run(Some(Device::Rocm(0))), &run(None));
}

#[test]
#[ignore = "ROCm 実機（gfx942）が必要。実測は Phase 10 の #3867"]
fn rocm_facade_mlp_train_step_parity_vs_cpu() {
    let (batch, d_in, hidden, d_out) = (8, 16, 24, 4);
    let x = Tensor::new(data(batch * d_in, 3), &[batch, d_in]).expect("x");
    let w1 = Tensor::new(data(d_in * hidden, 4), &[d_in, hidden]).expect("w1");
    let w2 = Tensor::new(data(hidden * d_out, 5), &[hidden, d_out]).expect("w2");
    let y = Tensor::new(data(batch * d_out, 6), &[batch, d_out]).expect("y");
    let run = |device: Option<Device>| {
        let tape = match device {
            Some(d) => fandhe_ai::tape_for(d).expect("tape_for"),
            None => fandhe_ai::tape(),
        };
        let (vx, vw1, vw2, vy) = (tape.var(&x), tape.var(&w1), tape.var(&w2), tape.var(&y));
        let pred = vx
            .matmul(&vw1)
            .expect("l1")
            .relu()
            .matmul(&vw2)
            .expect("l2");
        let loss = MseLoss::new(Reduction::Mean)
            .forward(&pred, &vy)
            .expect("loss");
        let grads = tape.backward(&loss).expect("backward");
        let g1 = dense(grads.get(&vw1).expect("get").expect("grad w1"));
        let g2 = dense(grads.get(&vw2).expect("get").expect("grad w2"));
        (dense(&loss.to_tensor()), g1, g2)
    };
    let (rl, r1, r2) = run(Some(Device::Rocm(0)));
    let (cl, c1, c2) = run(None);
    assert_compare("loss", &rl, &cl);
    assert_compare("grad_w1", &r1, &c1);
    assert_compare("grad_w2", &r2, &c2);
}

#[test]
#[ignore = "ROCm 実機（gfx942）が必要。実測は Phase 10 の #3867"]
fn rocm_tape_for_out_of_range_ordinal_is_err() {
    for d in [Device::Rocm(1), Device::Rocm(usize::MAX)] {
        assert!(
            matches!(
                fandhe_ai::tape_for(d),
                Err(fandhe_ai::BackendError::DeviceUnavailable(_))
            ),
            "{d:?}"
        );
    }
}
