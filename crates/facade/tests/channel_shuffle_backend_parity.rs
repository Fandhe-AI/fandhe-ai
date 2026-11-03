//! `Var::channel_shuffle`（イシュー #3822）のバックエンド間 parity（REQ-2）対応テスト。
//! `pixel_shuffle_backend_parity.rs`（#2162）と同型。層型は facade から再エクスポートしないため
//! `Var::channel_shuffle` だけで組む。
//!
//! - 属性なし: `CpuBackendOps`（`fandhe_ai::tape()`）と `fandhe_ai_autodiff::Tape::new()`
//!   （`NaiveOps`）の forward・backward を **bit 完全一致**で突き合わせる（純粋なコピー・view の
//!   合成のみ。tolerance は変更しない）。
//! - `#[ignore]`: `cuda_*`（`Device::Cuda(0)`）・`metal_*`（`cfg(target_os = "macos")`・
//!   `Device::Metal`）で同じ検証を CPU と対称に置く。実機未実測のまま出荷し
//!   `docs/perf/logs/channel-shuffle-3822/README.md` へ申し送る。

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

const X_SHAPE: [usize; 4] = [2, 6, 3, 3];
const GROUPS: usize = 3;

fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len(), "{label}: 要素数が一致しない");
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        if a.is_nan() || e.is_nan() {
            assert!(
                a.is_nan() && e.is_nan(),
                "{label}: 要素 {i} が NaN クラス一致しない（actual={a}, expected={e}）"
            );
        } else {
            assert_eq!(
                a.to_bits(),
                e.to_bits(),
                "{label}: 要素 {i} が bit 一致しない（actual={a:?}, expected={e:?}）"
            );
        }
    }
}

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

fn print_fold_bits(label: &str, data: &[f32]) {
    let mut acc: u64 = 0xcbf29ce484222325;
    for &v in data.iter() {
        acc ^= v.to_bits() as u64;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    println!("{label}.fold_bits={acc:#018x}");
}

/// forward の出力を返す。
fn forward_on<T: VarSource>(tape: &T) -> Tensor<f32> {
    let x = tape.make_var(&leaf(1, &X_SHAPE));
    x.channel_shuffle(GROUPS).unwrap().to_tensor()
}

/// 位置ごとに異なる重みを掛けた損失の（出力, 入力勾配）。逆置換が正しく伝わることを検出できる。
fn weighted_grad_cpu() -> (Tensor<f32>, Tensor<f32>) {
    let tape = fandhe_ai::tape();
    let x = tape.make_var(&leaf(2, &X_SHAPE));
    let w = tape.make_var(&leaf(3, &X_SHAPE));
    let y = x.channel_shuffle(GROUPS).unwrap();
    let loss = y.mul(&w).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("到達する").clone();
    (y.to_tensor(), dx)
}

fn weighted_grad_naive() -> (Tensor<f32>, Tensor<f32>) {
    let tape = fandhe_ai_autodiff::Tape::new();
    let x = tape.make_var(&leaf(2, &X_SHAPE));
    let w = tape.make_var(&leaf(3, &X_SHAPE));
    let y = x.channel_shuffle(GROUPS).unwrap();
    let loss = y.mul(&w).unwrap().sum(None).unwrap();
    let grads = tape.backward(&loss).unwrap();
    let dx = grads.get(&x).unwrap().expect("到達する").clone();
    (y.to_tensor(), dx)
}

#[test]
fn cpu_channel_shuffle_forward_bit_exact_vs_naive() {
    let out_cpu = forward_on(&fandhe_ai::tape());
    let out_naive = forward_on(&fandhe_ai_autodiff::Tape::new());
    assert_bits_eq(
        "ChannelShuffle forward: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&out_cpu),
        &contiguous_slice(&out_naive),
    );
}

#[test]
fn cpu_channel_shuffle_backward_bit_exact_vs_naive() {
    let dx_cpu = weighted_grad_cpu().1;
    let dx_naive = weighted_grad_naive().1;
    assert_bits_eq(
        "ChannelShuffle backward（dx）: CpuBackendOps vs NaiveOps",
        &contiguous_slice(&dx_cpu),
        &contiguous_slice(&dx_naive),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。bit 完全一致契約） ---

fn forward_on_device(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    forward_on(&tape)
}

fn backward_on_device(device: Device) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let x = tape.make_var(&leaf(2, &X_SHAPE));
    let w = tape.make_var(&leaf(3, &X_SHAPE));
    let loss = x
        .channel_shuffle(GROUPS)
        .unwrap()
        .mul(&w)
        .unwrap()
        .sum(None)
        .unwrap();
    let grads = tape.backward(&loss).unwrap();
    grads.get(&x).unwrap().expect("到達する").clone()
}

fn assert_device_matches_cpu(label: &str, device: Device) {
    let dev_out = forward_on_device(device);
    let cpu_out = forward_on_device(Device::Cpu);
    assert_bits_eq(
        &format!("ChannelShuffle forward: {label} vs CPU"),
        &contiguous_slice(&dev_out),
        &contiguous_slice(&cpu_out),
    );
    print_fold_bits(
        &format!("{label}_channel_shuffle_forward[out]"),
        &contiguous_slice(&dev_out),
    );
}

fn assert_device_backward_matches_cpu(label: &str, device: Device) {
    let dev_dx = backward_on_device(device);
    let cpu_dx = backward_on_device(Device::Cpu);
    assert_bits_eq(
        &format!("ChannelShuffle backward（dx）: {label} vs CPU"),
        &contiguous_slice(&dev_dx),
        &contiguous_slice(&cpu_dx),
    );
    print_fold_bits(
        &format!("{label}_channel_shuffle_backward[dx]"),
        &contiguous_slice(&dev_dx),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_channel_shuffle_forward_matches_cpu() {
    assert_device_matches_cpu("metal", Device::Metal);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_channel_shuffle_backward_matches_cpu() {
    assert_device_backward_matches_cpu("metal", Device::Metal);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_channel_shuffle_forward_matches_cpu() {
    assert_device_matches_cpu("cuda", Device::Cuda(0));
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_channel_shuffle_backward_matches_cpu() {
    assert_device_backward_matches_cpu("cuda", Device::Cuda(0));
}
