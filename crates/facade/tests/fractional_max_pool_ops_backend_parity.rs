//! `Var::fractional_max_pool2d`／`Var::fractional_max_pool3d`（イシュー #3826。実体は
//! `fandhe_ai_autodiff::fractional_max_pool_ops`。`crates/autodiff/src/fractional_max_pool_ops.rs`
//! モジュール doc 参照）のバックエンド間 parity テスト（`adaptive_pool3d_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps::fractional_max_pool*_windows`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定 `Unsupported` → 共有ホストカーネルへ
//! フォールバック〉の突き合わせ）: 値・索引・入力勾配を **bit 一致**で検証する（tolerance は使わず新設もしない）。
//! 乱数標本はホストのグローバル RNG で決まるため、**各 tape の呼び出しの直前に同じシードで
//! `manual_seed` を呼んで同じ標本にする**。テストバイナリ内の `LOCK` で RNG を直列化する。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os = "macos")` 限定〉で
//! 同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント実行環境にないため未実測のまま
//! GB10／Mac セッション（#3860／#3861）へ申し送る
//! （`docs/perf/logs/fractional-max-pool-ops-3826/README.md`）。CUDA／Metal はいずれも専用カーネルを
//! 持たない（既定 `Unsupported`）ため、この比較は「窓表をホストで作りホストカーネルへフォールバックする
//! 経路が CPU tape と同じ結果になること」の確認であり、GPU カーネル自体の parity ではない。

use std::sync::{Mutex, MutexGuard};

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_tensor_core::Tensor;

static LOCK: Mutex<()> = Mutex::new(());

fn lock() -> MutexGuard<'static, ()> {
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

const SEED: u64 = 3826;
// 2d: H 9→5（alpha が整数にならない）・W 8→4。
const IN2: [usize; 4] = [2, 2, 9, 8];
const K2: [usize; 2] = [2, 2];
const OUT2: [usize; 2] = [5, 4];
// 3d: 7→4・6→3・9→4（いずれも out + k - 1 < in）。
const IN3: [usize; 5] = [1, 2, 7, 6, 9];
const K3: [usize; 3] = [2, 2, 2];
const OUT3: [usize; 3] = [4, 3, 4];

/// 窓内タイを含みうる決定的な入力（`7` と `11` は互いに素）。
fn input(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    t(
        (0..n)
            .map(|i| (((i * 7 + 3) % 11) as f32) * 0.4 - 2.0)
            .collect(),
        shape,
    )
}

fn weights(shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    t(
        (0..n).map(|i| ((i % 5) as f32) * 0.3 - 0.6).collect(),
        shape,
    )
}

struct Outputs {
    v2: Tensor<f32>,
    i2: Vec<i32>,
    d2: Tensor<f32>,
    v3: Tensor<f32>,
    i3: Vec<i32>,
    d3: Tensor<f32>,
}

fn indices(i: &Tensor<i32>) -> Vec<i32> {
    i.contiguous().host_slice().into_owned()
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        let x2 = tape.make_var(&input(&IN2));
        fandhe_ai::manual_seed(SEED);
        let (v2, i2) = x2.fractional_max_pool2d(K2, OUT2).unwrap();
        let w2 = tape.make_var(&weights(v2.to_tensor().shape()));
        let loss2 = v2.mul(&w2).unwrap().sum(None).unwrap();
        let d2 = tape
            .backward(&loss2)
            .unwrap()
            .get(&x2)
            .unwrap()
            .unwrap()
            .clone();
        let x3 = tape.make_var(&input(&IN3));
        fandhe_ai::manual_seed(SEED);
        let (v3, i3) = x3.fractional_max_pool3d(K3, OUT3).unwrap();
        let w3 = tape.make_var(&weights(v3.to_tensor().shape()));
        let loss3 = v3.mul(&w3).unwrap().sum(None).unwrap();
        let d3 = tape
            .backward(&loss3)
            .unwrap()
            .get(&x3)
            .unwrap()
            .unwrap()
            .clone();
        Outputs {
            v2: v2.to_tensor(),
            i2: indices(&i2),
            d2,
            v3: v3.to_tensor(),
            i3: indices(&i3),
            d3,
        }
    }};
}

fn cpu_outputs() -> Outputs {
    outputs_on!(fandhe_ai::tape())
}

fn naive_outputs() -> Outputs {
    outputs_on!(fandhe_ai_autodiff::Tape::new())
}

fn assert_bits_eq(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    for (i, (x, y)) in a.host_slice().iter().zip(b.host_slice().iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}]: {x} vs {y}");
    }
}

fn assert_outputs_match(label: &str, a: &Outputs, b: &Outputs) {
    assert_bits_eq(&format!("{label}: 2d 値"), &a.v2, &b.v2);
    assert_eq!(a.i2, b.i2, "{label}: 2d 索引");
    assert_bits_eq(&format!("{label}: 2d backward"), &a.d2, &b.d2);
    assert_bits_eq(&format!("{label}: 3d 値"), &a.v3, &b.v3);
    assert_eq!(a.i3, b.i3, "{label}: 3d 索引");
    assert_bits_eq(&format!("{label}: 3d backward"), &a.d3, &b.d3);
}

#[test]
fn cpu_matches_naive_reference() {
    let _g = lock();
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）: 出力要素数・索引範囲・勾配の総和が重みの総和に
    // 一致する（各出力の寄与は勝者 1 要素へ重みの分だけ scatter-add されるため）。
    assert_eq!(cpu.i2.len(), 2 * 2 * 5 * 4);
    assert_eq!(cpu.i3.len(), 2 * 4 * 3 * 4);
    assert!(cpu.i2.iter().all(|&i| (0..9 * 8).contains(&i)));
    assert!(cpu.i3.iter().all(|&i| (0..7 * 6 * 9).contains(&i)));
    for (d, shape) in [(&cpu.d2, vec![2, 2, 5, 4]), (&cpu.d3, vec![1, 2, 4, 3, 4])] {
        let wsum: f64 = weights(&shape)
            .host_slice()
            .iter()
            .map(|&v| f64::from(v))
            .sum();
        let gsum: f64 = d.host_slice().iter().map(|&v| f64::from(v)).sum();
        assert!(
            (gsum - wsum).abs() < 1e-3,
            "勾配総和 {gsum} vs 重み総和 {wsum}"
        );
    }
    // 値は入力要素と bit 一致（選択のみ）: 全出力値が入力に含まれる。
    let xs: Vec<u32> = input(&IN2)
        .host_slice()
        .iter()
        .map(|v| v.to_bits())
        .collect();
    assert!(
        cpu.v2
            .host_slice()
            .iter()
            .all(|v| xs.contains(&v.to_bits()))
    );
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッション（#3860／#3861）へ申し送る
// （`docs/perf/logs/fractional-max-pool-ops-3826/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let _g = lock();
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// 2 演算の forward・backward の CPU／Metal 実機比較（同じシード＝同じ標本）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/fractional-max-pool-ops-3826/README.md 参照"]
fn metal_fractional_max_pool_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// 2 演算の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較（同じシード＝同じ標本）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/fractional-max-pool-ops-3826/README.md 参照"]
fn cuda_fractional_max_pool_ops_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
