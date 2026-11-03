//! `Var::lp_pool1d`／`lp_pool2d`／`lp_pool3d`（イシュー #3825）のバックエンド間 parity（REQ-2）対応テスト。
//! `adaptive_pool3d_ops_backend_parity.rs`（#3823）と同型。層型は facade から再エクスポートしないため
//! `Var` 演算だけで組む。
//!
//! - 属性なし: `CpuBackendOps`（`fandhe_ai::tape()`）と `fandhe_ai_autodiff::Tape::new()`（`NaiveOps`）の
//!   forward・backward を REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で突き合わせる。
//!   合成先の各演算は共有ホストカーネルへ到達するため、確認するのは配線が結果を変えないこと。
//! - `#[ignore]`: `cuda_*`（`Device::Cuda(0)`）・`metal_*`（`cfg(target_os = "macos")`）で CPU tape と比較する。
//!   実機未実測のまま出荷し `docs/perf/logs/lp-pool-ops-3825/README.md` へ申し送る（tolerance は変更しない）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_tensor_core::Tensor;

trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

/// 正の入力（非整数 `p` でも NaN にならない）。
fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data: Vec<f32> = Xorshift64Star::new(seed)
        .fill_vec(numel)
        .into_iter()
        .map(|v| v.abs() + 0.25)
        .collect();
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

fn assert_parity(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    fandhe_ai_backend_cpu::parity::assert_parity(label, &contiguous_slice(a), &contiguous_slice(b));
}

struct Outputs {
    y: [Tensor<f32>; 3],
    dx: [Tensor<f32>; 3],
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        let x1 = tape.make_var(&leaf(1, &[2, 3, 8]));
        let x2 = tape.make_var(&leaf(2, &[2, 3, 6, 6]));
        let x3 = tape.make_var(&leaf(3, &[1, 2, 4, 4, 4]));
        let y1 = x1.lp_pool1d(2.0, 3, Some(2), false).unwrap();
        let y2 = x2.lp_pool2d(3.0, [2, 2], None, false).unwrap();
        let y3 = x3
            .lp_pool3d(1.5, [2, 2, 2], Some([1, 2, 2]), false)
            .unwrap();
        let w1 = tape.make_var(&leaf(4, &y1.to_tensor().shape().to_vec()));
        let w2 = tape.make_var(&leaf(5, &y2.to_tensor().shape().to_vec()));
        let w3 = tape.make_var(&leaf(6, &y3.to_tensor().shape().to_vec()));
        let loss = y1
            .mul(&w1)
            .unwrap()
            .sum(None)
            .unwrap()
            .add(&y2.mul(&w2).unwrap().sum(None).unwrap())
            .unwrap()
            .add(&y3.mul(&w3).unwrap().sum(None).unwrap())
            .unwrap();
        let grads = tape.backward(&loss).unwrap();
        Outputs {
            y: [y1.to_tensor(), y2.to_tensor(), y3.to_tensor()],
            dx: [
                grads.get(&x1).unwrap().expect("到達する").clone(),
                grads.get(&x2).unwrap().expect("到達する").clone(),
                grads.get(&x3).unwrap().expect("到達する").clone(),
            ],
        }
    }};
}

fn assert_outputs_match(label: &str, a: &Outputs, b: &Outputs) {
    for i in 0..3 {
        assert_parity(
            &format!("{label}: lp_pool{}d forward", i + 1),
            &a.y[i],
            &b.y[i],
        );
        assert_parity(
            &format!("{label}: lp_pool{}d dx", i + 1),
            &a.dx[i],
            &b.dx[i],
        );
    }
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = outputs_on!(fandhe_ai::tape());
    let naive = outputs_on!(fandhe_ai_autodiff::Tape::new());
    assert_outputs_match("CpuBackendOps vs NaiveOps", &cpu, &naive);
}

fn assert_device_matches_cpu(label: &str, device: Device) {
    let dev = outputs_on!(
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず")
    );
    let cpu = outputs_on!(fandhe_ai::tape_for(Device::Cpu).expect("CPU tape は常に構築できる"));
    assert_outputs_match(&format!("{label} vs CPU"), &dev, &cpu);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない（#3861）"]
fn metal_lp_pool_ops_match_cpu_reference() {
    assert_device_matches_cpu("metal", Device::Metal);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須（#3860）"]
fn cuda_lp_pool_ops_match_cpu_reference() {
    assert_device_matches_cpu("cuda", Device::Cuda(0));
}
