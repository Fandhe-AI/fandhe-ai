//! `compat::FunctionalBuilder` の `max_pool{2,3}d_with_indices`／`max_unpool{2,3}d`（イシュー #3036・#3037・
//! 親 #3018）の層（Functional グラフ）経路の CUDA／Metal parity テスト。
//! `compat_sequential_conv_transpose3d_backend_parity.rs`（#3033）と同型。
//!
//! MaxUnpool は `Sequential` に載せず `FunctionalBuilder` 限定の公開のため、`Sequential` 名ではなく本ファイル名
//! にした（ログ・申し送りの置き場は issue 指定どおり `docs/perf/logs/compat-sequential-max-unpool-3037/`）。
//! 演算単体の parity は `conv_transpose3d_max_unpool_backend_parity.rs` にあり、本ファイルは SegNet 型の
//! `FunctionalModel`（conv ブロック → pool → conv ブロック → unpool）の forward 出力と、`sum` 損失に対する
//! 入力勾配を、対象デバイスと CPU で REQ-2 統一複合判定（`assert_parity`。相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満）で比較する。パラメータ勾配は `FunctionalModel::bind` が非公開で公開面から取れないため
//! 比較対象に含めない。tolerance は変更しない。GPU カーネルの有無は演算単体の記録に従う（無い場合は
//! ホストフォールバック経路の確認になる）。CPU 側の正しさは `compat_functional_max_unpool.rs` が担う。
//! 実機実測は #3171・#3172 が担い、本 issue では未実測
//! （`docs/perf/logs/compat-sequential-max-unpool-3037/README.md`）。

use fandhe_ai::compat::{FunctionalBuilder, FunctionalModel, Sequential};
use fandhe_ai::{Device, Tensor};
use fandhe_ai_backend_cpu::parity::assert_parity;

fn ramp(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(
        (0..n).map(|i| ((i as f32) * 0.173).sin() * 0.9).collect(),
        shape,
    )
    .expect("test fixture: shape とデータ長は一致させている")
}

fn dense(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が Some")
        .to_vec()
}

fn model_2d() -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let conv = |cin, cout, seed| {
        Sequential::new()
            .add_conv2d(cin, cout, [3, 3], [1, 1], [1, 1], [1, 1], 1, seed)
            .expect("test fixture: 有効な引数")
    };
    let h = b.apply(conv(2, 3, 0x3037), x).unwrap();
    let pool = b
        .max_pool2d_with_indices(h, [2, 2], None, [0, 0], [1, 1])
        .unwrap();
    let m = b.apply(conv(3, 3, 0x3038), pool).unwrap();
    let up = b.max_unpool2d(m, pool, [2, 2], None, [0, 0], None).unwrap();
    b.build(&[x], &[up]).unwrap()
}

fn model_3d() -> FunctionalModel {
    let mut b = FunctionalBuilder::new();
    let x = b.input().unwrap();
    let conv = |cin, cout, seed| {
        Sequential::new()
            .add_conv3d(
                cin,
                cout,
                [3, 3, 3],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
                1,
                seed,
            )
            .expect("test fixture: 有効な引数")
    };
    let h = b.apply(conv(2, 3, 0x3037), x).unwrap();
    let pool = b
        .max_pool3d_with_indices(h, [2, 2, 2], None, [0, 0, 0], [1, 1, 1])
        .unwrap();
    let m = b.apply(conv(3, 3, 0x3038), pool).unwrap();
    let up = b
        .max_unpool3d(m, pool, [2, 2, 2], None, [0, 0, 0], None)
        .unwrap();
    b.build(&[x], &[up]).unwrap()
}

/// (forward 出力, 入力勾配) を `device` 上で計算する。
fn run(build: fn() -> FunctionalModel, shape: &[usize], device: Device) -> (Vec<f32>, Vec<f32>) {
    let model = build();
    let x = ramp(shape);
    let tape = fandhe_ai::tape_for(device)
        .expect("実機必須（本テストは #[ignore]。実行時は事前に到達確認する）");
    let xv = tape.var(&x);
    let outs = model.forward(&tape, &[xv]).unwrap();
    let y = outs[0];
    let out = dense(&y.to_tensor());
    let loss = y.sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let gx = dense(grads.get(&xv).unwrap().expect("入力へ勾配が届く"));
    (out, gx)
}

fn run_parity(label: &str, build: fn() -> FunctionalModel, shape: &[usize], device: Device) {
    let (cpu_out, cpu_grad) = run(build, shape, Device::Cpu);
    let (dev_out, dev_grad) = run(build, shape, device);
    assert_parity(
        &format!("compat::Functional({label}) forward device vs CPU"),
        &dev_out,
        &cpu_out,
    );
    assert_parity(
        &format!("compat::Functional({label}) input-grad device vs CPU"),
        &dev_grad,
        &cpu_grad,
    );
}

/// 実機なしでも CI で通る自己確認（CPU 対 CPU）。比較ハーネス（forward・入力勾配の取り出し）自体が
/// 壊れていないことと、2 回の計算が決定的であることを確認する。
#[test]
fn harness_is_deterministic_on_cpu() {
    run_parity("MaxUnpool2d", model_2d, &[2, 2, 6, 6], Device::Cpu);
    run_parity("MaxUnpool3d", model_3d, &[1, 2, 4, 4, 4], Device::Cpu);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実測は #3171 の申し送り先（GB10 セッション）"]
fn cuda_functional_max_unpool_2d_forward_backward_matches_cpu() {
    run_parity("MaxUnpool2d", model_2d, &[2, 2, 6, 6], Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。実測は #3171 の申し送り先（GB10 セッション）"]
fn cuda_functional_max_unpool_3d_forward_backward_matches_cpu() {
    run_parity("MaxUnpool3d", model_3d, &[1, 2, 4, 4, 4], Device::Cuda(0));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実測は #3172 の申し送り先（Mac セッション）"]
fn metal_functional_max_unpool_2d_forward_backward_matches_cpu() {
    run_parity("MaxUnpool2d", model_2d, &[2, 2, 6, 6], Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須。実測は #3172 の申し送り先（Mac セッション）"]
fn metal_functional_max_unpool_3d_forward_backward_matches_cpu() {
    run_parity("MaxUnpool3d", model_3d, &[1, 2, 4, 4, 4], Device::Metal);
}
