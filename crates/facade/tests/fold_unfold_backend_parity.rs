//! `Var::fold`／`Var::unfold`（イシュー #2645 で実装・#2851 で facade 公開。実体は
//! `fandhe_ai_autodiff::fold_ops::{fold, unfold}` への 1 行委譲で、本テストは #2851 以降 `Var` メソッド経由で
//! 同じ経路を測る）のバックエンド間 parity テスト
//! （`conv_transpose3d_max_unpool_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝
//! 既定 `Unsupported` → ホストフォールバック〉の突き合わせ）: forward・勾配はいずれも **bit 一致**。
//! `unfold`／`fold` の d_input は純コピー（`im2col`）か `f64` アキュムレータの `col2im` で、
//! `CpuBackendOps` のカーネルとホスト参照実装（`eval::im2col`／`col2im`）は走査順・アキュムレータが
//! 同一という既存契約（`BackendOps::col2im` の trait doc）に従う。一致しなければ判定を緩めず原因を調べる。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os = "macos")`
//! 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント実行環境にないため未実施の
//! まま GB10／Mac セッションへ申し送る（`docs/perf/logs/fold-unfold-2645/README.md`）。CUDA／Metal は
//! `im2col`／`col2im` を既に専用カーネルで override 済みで、本テストは新規 GPU カーネルの parity では
//! なく「既存カーネルを通る経路が CPU tape と bit 一致すること」の確認である。

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

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn wave(n: usize, freq: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * freq).sin() * amp).collect()
}

/// 非自明な設定（重なる窓・padding・dilation・軸ごとに異なる引数・N=2・C=3）。
const KERNEL: [usize; 2] = [2, 3];
const STRIDE: [usize; 2] = [1, 2];
const PADDING: [usize; 2] = [1, 1];
const DILATION: [usize; 2] = [2, 1];
const IMG: [usize; 4] = [2, 3, 6, 7];
const OUT_SIZE: [usize; 2] = [6, 7];

struct Outputs {
    unfold_out: Tensor<f32>,
    unfold_dx: Tensor<f32>,
    fold_out: Tensor<f32>,
    fold_dx: Tensor<f32>,
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        // --- unfold ---
        let x = tape.make_var(&t(wave(IMG.iter().product(), 0.043, 1.0), &IMG));
        let y = x.unfold(KERNEL, STRIDE, PADDING, DILATION).unwrap();
        let unfold_out = y.to_tensor();
        let n_out: usize = unfold_out.shape().iter().product();
        let g = tape.make_var(&t(wave(n_out, 0.053, 0.7), unfold_out.shape()));
        let loss = y.mul(&g).unwrap().sum(None).unwrap();
        let unfold_dx = tape
            .backward(&loss)
            .unwrap()
            .get(&x)
            .unwrap()
            .unwrap()
            .clone();

        // --- fold（unfold の出力 shape を入力に取る） ---
        let col_shape = unfold_out.shape().to_vec();
        let n_col: usize = col_shape.iter().product();
        let c = tape.make_var(&t(wave(n_col, 0.071, 1.0), &col_shape));
        let z = c.fold(OUT_SIZE, KERNEL, STRIDE, PADDING, DILATION).unwrap();
        let fold_out = z.to_tensor();
        let n_z: usize = fold_out.shape().iter().product();
        let gz = tape.make_var(&t(wave(n_z, 0.037, 0.9), fold_out.shape()));
        let loss = z.mul(&gz).unwrap().sum(None).unwrap();
        let fold_dx = tape
            .backward(&loss)
            .unwrap()
            .get(&c)
            .unwrap()
            .unwrap()
            .clone();

        Outputs {
            unfold_out,
            unfold_dx,
            fold_out,
            fold_dx,
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
    assert_bits_eq(
        &format!("{label}: unfold forward"),
        &a.unfold_out,
        &b.unfold_out,
    );
    assert_bits_eq(
        &format!("{label}: unfold d_input"),
        &a.unfold_dx,
        &b.unfold_dx,
    );
    assert_bits_eq(&format!("{label}: fold forward"), &a.fold_out, &b.fold_out);
    assert_bits_eq(&format!("{label}: fold d_input"), &a.fold_dx, &b.fold_dx);
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）。
    // H: (6+2-2*(2-1)-1)/1+1 = 6、W: (7+2-1*(3-1)-1)/2+1 = 4 → L = 24、K = 3*2*3 = 18。
    assert_eq!(cpu.unfold_out.shape(), &[2, 18, 24]);
    assert_eq!(cpu.unfold_dx.shape(), &IMG);
    assert_eq!(cpu.fold_out.shape(), &[2, 3, 6, 7]);
    assert_eq!(cpu.fold_dx.shape(), &[2, 18, 24]);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/fold-unfold-2645/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// Fold／Unfold の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/fold-unfold-2645/README.md 参照"]
fn metal_fold_unfold_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// Fold／Unfold の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/fold-unfold-2645/README.md 参照"]
fn cuda_fold_unfold_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
