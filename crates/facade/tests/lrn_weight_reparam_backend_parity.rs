//! `Var::local_response_norm`／`Var::weight_norm`／`Var::spectral_norm`（イシュー #2646 で実装・#2851 で
//! facade 公開。実体は `fandhe_ai_autodiff::{lrn_ops, weight_reparam_ops}` への 1 行委譲で、本テストは #2851 以降
//! `Var` メソッド経由で同じ経路を測る。`norm_except_dim` は非公開のため内部パスから import する）の
//! バックエンド間 parity テスト
//! （`pool3d_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝
//! 既定 `Unsupported` → 共有ホストカーネルへのフォールバック〉の突き合わせ）: forward・勾配はいずれも
//! **bit 一致**。`CpuBackendOps` の 3 フックと autodiff のフォールバックは同じ共有カーネル
//! （`fandhe_ai_tensor_core::{lrn, weight_reparam}`）を呼ぶだけで、VJP は両 tape とも共有カーネルの
//! ホスト実装を通るため。一致しなければ判定を緩めず原因を調べる。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os = "macos")`
//! 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント実行環境にないため未実施の
//! まま GB10／Mac セッションへ申し送る（`docs/perf/logs/lrn-weight-reparam-2646/README.md`）。CUDA／
//! Metal は新規 3 フックを override しない（既定 `Unsupported`）ため、本テストは新規 GPU カーネルの
//! parity ではなく「`Unsupported` → ホストフォールバック経路が CPU tape と bit 一致すること」の確認である。

use fandhe_ai::Device;
use fandhe_ai::SpectralNormState;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::weight_reparam_ops::norm_except_dim;
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

const LRN_SHAPE: [usize; 4] = [2, 7, 3, 2];
const WN_SHAPE: [usize; 3] = [4, 3, 5];
const SP_SHAPE: [usize; 3] = [3, 4, 2];

struct Outputs {
    lrn_out: Tensor<f32>,
    lrn_dx: Tensor<f32>,
    wn_out: Tensor<f32>,
    wn_dv: Tensor<f32>,
    wn_dg: Tensor<f32>,
    sp_out: Tensor<f32>,
    sp_dw: Tensor<f32>,
    sp_u: Vec<f32>,
    sp_v: Vec<f32>,
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        // --- LocalResponseNorm（偶数 size・非既定パラメータ） ---
        let n_lrn: usize = LRN_SHAPE.iter().product();
        let x = tape.make_var(&t(wave(n_lrn, 0.043, 2.0), &LRN_SHAPE));
        let y = x.local_response_norm(4, 0.3, 0.75, 1.5).unwrap();
        let lrn_out = y.to_tensor();
        let g = tape.make_var(&t(wave(n_lrn, 0.053, 0.7), &LRN_SHAPE));
        let loss = y.mul(&g).unwrap().sum(None).unwrap();
        let lrn_dx = tape
            .backward(&loss)
            .unwrap()
            .get(&x)
            .unwrap()
            .unwrap()
            .clone();

        // --- weight_norm（dim = 1・g は own×1.3） ---
        let n_wn: usize = WN_SHAPE.iter().product();
        let v0 = t(wave(n_wn, 0.071, 1.0), &WN_SHAPE);
        let own = norm_except_dim(&v0, Some(1)).unwrap();
        let g0 = t(
            own.contiguous()
                .host_slice()
                .iter()
                .map(|x| x * 1.3)
                .collect(),
            own.shape(),
        );
        let v = tape.make_var(&v0);
        let gv = tape.make_var(&g0);
        let w = v.weight_norm(&gv, Some(1)).unwrap();
        let wn_out = w.to_tensor();
        let up = tape.make_var(&t(wave(n_wn, 0.037, 0.9), &WN_SHAPE));
        let loss = w.mul(&up).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let wn_dv = grads.get(&v).unwrap().unwrap().clone();
        let wn_dg = grads.get(&gv).unwrap().unwrap().clone();

        // --- spectral_norm（training・dim = 1・n = 2） ---
        let n_sp: usize = SP_SHAPE.iter().product();
        let mut st = SpectralNormState::from_vectors(
            &SP_SHAPE,
            1,
            &[1.0, 0.5, -0.25, 0.75],
            &[0.3, -0.6, 0.9, 0.1, 0.2, -0.4],
            2,
            1e-12,
        )
        .unwrap();
        let sw = tape.make_var(&t(wave(n_sp, 0.091, 1.2), &SP_SHAPE));
        let sy = sw.spectral_norm(&mut st, true).unwrap();
        let sp_out = sy.to_tensor();
        let sup = tape.make_var(&t(wave(n_sp, 0.029, 0.8), &SP_SHAPE));
        let loss = sy.mul(&sup).unwrap().sum(None).unwrap();
        let sp_dw = tape
            .backward(&loss)
            .unwrap()
            .get(&sw)
            .unwrap()
            .unwrap()
            .clone();

        Outputs {
            lrn_out,
            lrn_dx,
            wn_out,
            wn_dv,
            wn_dg,
            sp_out,
            sp_dw,
            sp_u: st.u().to_vec(),
            sp_v: st.v().to_vec(),
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

fn assert_vec_bits_eq(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: len");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}]: {x} vs {y}");
    }
}

fn assert_outputs_match(label: &str, a: &Outputs, b: &Outputs) {
    assert_bits_eq(&format!("{label}: lrn forward"), &a.lrn_out, &b.lrn_out);
    assert_bits_eq(&format!("{label}: lrn d_input"), &a.lrn_dx, &b.lrn_dx);
    assert_bits_eq(
        &format!("{label}: weight_norm forward"),
        &a.wn_out,
        &b.wn_out,
    );
    assert_bits_eq(&format!("{label}: weight_norm dv"), &a.wn_dv, &b.wn_dv);
    assert_bits_eq(&format!("{label}: weight_norm dg"), &a.wn_dg, &b.wn_dg);
    assert_bits_eq(&format!("{label}: spectral forward"), &a.sp_out, &b.sp_out);
    assert_bits_eq(&format!("{label}: spectral dw"), &a.sp_dw, &b.sp_dw);
    assert_vec_bits_eq(&format!("{label}: spectral u"), &a.sp_u, &b.sp_u);
    assert_vec_bits_eq(&format!("{label}: spectral v"), &a.sp_v, &b.sp_v);
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）。
    assert_eq!(cpu.lrn_out.shape(), &LRN_SHAPE);
    assert_eq!(cpu.wn_dg.shape(), &[1, 3, 1]);
    assert_eq!(cpu.sp_out.shape(), &SP_SHAPE);
    assert!(cpu.lrn_out.host_slice().iter().all(|v| v.is_finite()));
    assert!(cpu.sp_out.host_slice().iter().all(|v| v.is_finite()));
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/lrn-weight-reparam-2646/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// LRN／weight_norm／spectral_norm の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/lrn-weight-reparam-2646/README.md 参照"]
fn metal_lrn_weight_reparam_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// LRN／weight_norm／spectral_norm の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/lrn-weight-reparam-2646/README.md 参照"]
fn cuda_lrn_weight_reparam_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
