//! `fandhe_ai_autodiff::fft_ops`（イシュー #2631・#2632・`rfft`／`irfft`／`fft`／`ifft`。facade
//! 非公開のため `fandhe_ai_autodiff::fft_ops::*` を直接 use する。
//! `crates/autodiff/src/fft_ops.rs` モジュール doc 参照）のバックエンド間
//! parity テスト（`linalg_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps::fft_*`〉と
//! `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝既定 `Unsupported` →
//! 共有ホストカーネルへフォールバック〉の突き合わせ）: forward・backward を
//! REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::assert_parity`）で検証
//! する。両経路は同じ共有カーネル（`fandhe_ai_tensor_core::fft`）を呼ぶため
//! 実質 bit 一致だが、判定は統一複合判定を用いる。CPU 本番経路の不正引数が
//! 型付きエラーで拒否されることも確認する。
//!
//! `#[ignore]`（`tape_for(Device::Metal)`〈`cfg(target_os = "macos")` 限定〉／
//! `tape_for(Device::Cuda(0))` で同じ経路を CPU tape と比較）: 実機（DGX Spark
//! GB10／Apple Silicon）への到達手段が本エージェント実行環境にないため未実施
//! のまま Mac／GB10 セッションへ申し送る
//! （`docs/perf/logs/fft-rfft-irfft-2631/README.md`）。CUDA／Metal はいずれも
//! FFT の GPU カーネルを持たないため（`BackendOps` 既定 `Unsupported`）、この
//! 比較は「ホストへのフォールバック経路が CPU tape と同じ結果になること」を
//! 確認するものであり、GPU カーネル自体の parity ではない。

use fandhe_ai::Device;
use fandhe_ai_autodiff::fft_ops::{fft, ifft, irfft, rfft};
use fandhe_ai_autodiff::{AutodiffError, Var};
use fandhe_ai_tensor_core::{FftNorm, Tensor};

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

fn assert_parity(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        a.host_slice().as_ref(),
        b.host_slice().as_ref(),
    );
}

fn seeded(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn rfft_fixture() -> Tensor<f32> {
    t(seeded(2 * 9, 7), &[2, 9])
}

fn irfft_fixture() -> Tensor<f32> {
    t(seeded(2 * 5 * 2, 8), &[2, 5, 2])
}

/// CPU tape（`CpuBackendOps::fft_*`）上の forward 出力と入力勾配（損失 `Σ y²`）。
fn grad_of_cpu(x_data: &Tensor<f32>, op: &str, norm: FftNorm) -> (Tensor<f32>, Tensor<f32>) {
    let tape = fandhe_ai::tape();
    let x = tape.make_var(x_data);
    let y = match op {
        "rfft" => rfft(&x, None, Some(1), norm).unwrap(),
        _ => irfft(&x, Some(8), Some(1), norm).unwrap(),
    };
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let dx = tape
        .backward(&loss)
        .unwrap()
        .get(&x)
        .unwrap()
        .unwrap()
        .clone();
    (y.to_tensor(), dx)
}

/// NaiveOps tape（共有ホストカーネルへフォールバック）上の同じ計算。
fn grad_of_naive(x_data: &Tensor<f32>, op: &str, norm: FftNorm) -> (Tensor<f32>, Tensor<f32>) {
    let tape = fandhe_ai_autodiff::Tape::new();
    let x = tape.make_var(x_data);
    let y = match op {
        "rfft" => rfft(&x, None, Some(1), norm).unwrap(),
        _ => irfft(&x, Some(8), Some(1), norm).unwrap(),
    };
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let dx = tape
        .backward(&loss)
        .unwrap()
        .get(&x)
        .unwrap()
        .unwrap()
        .clone();
    (y.to_tensor(), dx)
}

#[test]
fn cpu_rfft_forward_and_backward_match_naive_reference() {
    for norm in [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward] {
        let data = rfft_fixture();
        let (y_cpu, dx_cpu) = grad_of_cpu(&data, "rfft", norm);
        let (y_naive, dx_naive) = grad_of_naive(&data, "rfft", norm);
        assert_parity(
            &format!("rfft forward {norm:?}: cpu vs naive"),
            &y_cpu,
            &y_naive,
        );
        assert_parity(
            &format!("rfft backward {norm:?}: cpu vs naive"),
            &dx_cpu,
            &dx_naive,
        );
    }
}

#[test]
fn cpu_irfft_forward_and_backward_match_naive_reference() {
    for norm in [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward] {
        let data = irfft_fixture();
        let (y_cpu, dx_cpu) = grad_of_cpu(&data, "irfft", norm);
        let (y_naive, dx_naive) = grad_of_naive(&data, "irfft", norm);
        assert_parity(
            &format!("irfft forward {norm:?}: cpu vs naive"),
            &y_cpu,
            &y_naive,
        );
        assert_parity(
            &format!("irfft backward {norm:?}: cpu vs naive"),
            &dx_cpu,
            &dx_naive,
        );
    }
}

#[test]
fn cpu_invalid_arguments_are_typed_errors() {
    let tape = fandhe_ai::tape();
    let x = tape.make_var(&t(vec![1.0; 4], &[4]));
    assert!(matches!(
        rfft(&x, Some(0), None, FftNorm::Backward),
        Err(AutodiffError::InvalidArgument(_))
    ));
    assert!(matches!(
        irfft(&x, None, None, FftNorm::Backward),
        Err(AutodiffError::Shape(_))
    ));
}

// --- fft／ifft（c2c。イシュー #2632） ---

type C2cFn =
    for<'t> fn(&Var<'t>, Option<usize>, Option<usize>, FftNorm) -> Result<Var<'t>, AutodiffError>;

fn c2c_fixture() -> Tensor<f32> {
    t(seeded(2 * 6 * 2, 9), &[2, 6, 2])
}

/// 任意の tape 上の c2c forward 出力と入力勾配（損失 `Σ y²`。`n=8` でゼロ詰め）。
fn c2c_grad_on<T: VarSource + HasBackward>(
    tape: &T,
    op: C2cFn,
    norm: FftNorm,
) -> (Tensor<f32>, Tensor<f32>) {
    let data = c2c_fixture();
    let x = tape.make_var(&data);
    let y = op(&x, Some(8), Some(1), norm).unwrap();
    let loss = y.mul(&y).unwrap().sum(None).unwrap();
    let dx = tape.grad_of(&loss, &x);
    (y.to_tensor(), dx)
}

/// `backward` を両 Tape 型で共通に呼ぶための薄いトレイト。
trait HasBackward {
    fn grad_of(&self, loss: &Var<'_>, x: &Var<'_>) -> Tensor<f32>;
}

impl HasBackward for fandhe_ai::Tape {
    fn grad_of(&self, loss: &Var<'_>, x: &Var<'_>) -> Tensor<f32> {
        self.backward(loss)
            .unwrap()
            .get(x)
            .unwrap()
            .unwrap()
            .clone()
    }
}

impl HasBackward for fandhe_ai_autodiff::Tape {
    fn grad_of(&self, loss: &Var<'_>, x: &Var<'_>) -> Tensor<f32> {
        self.backward(loss)
            .unwrap()
            .get(x)
            .unwrap()
            .unwrap()
            .clone()
    }
}

#[test]
fn cpu_c2c_forward_and_backward_match_naive_reference() {
    for (name, op) in [("fft", fft as C2cFn), ("ifft", ifft as C2cFn)] {
        for norm in [FftNorm::Backward, FftNorm::Ortho, FftNorm::Forward] {
            let (y_cpu, dx_cpu) = c2c_grad_on(&fandhe_ai::tape(), op, norm);
            let (y_naive, dx_naive) = c2c_grad_on(&fandhe_ai_autodiff::Tape::new(), op, norm);
            assert_parity(
                &format!("{name} forward {norm:?}: cpu vs naive"),
                &y_cpu,
                &y_naive,
            );
            assert_parity(
                &format!("{name} backward {norm:?}: cpu vs naive"),
                &dx_cpu,
                &dx_naive,
            );
        }
    }
}

#[test]
fn cpu_c2c_invalid_arguments_are_typed_errors() {
    let tape = fandhe_ai::tape();
    let real = tape.make_var(&t(vec![1.0; 4], &[4]));
    let c = tape.make_var(&t(vec![1.0; 6], &[3, 2]));
    for op in [fft as C2cFn, ifft as C2cFn] {
        assert!(matches!(
            op(&real, None, None, FftNorm::Backward),
            Err(AutodiffError::Shape(_))
        ));
        assert!(matches!(
            op(&c, Some(0), None, FftNorm::Backward),
            Err(AutodiffError::InvalidArgument(_))
        ));
    }
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: Mac／DGX Spark GB10 実機セッションへ
// 申し送る（`docs/perf/logs/fft-rfft-irfft-2631/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    for (op, data) in [("rfft", rfft_fixture()), ("irfft", irfft_fixture())] {
        let (y_cpu, dx_cpu) = grad_of_cpu(&data, op, FftNorm::Ortho);
        let x = device_tape.make_var(&data);
        let y = match op {
            "rfft" => rfft(&x, None, Some(1), FftNorm::Ortho).unwrap(),
            _ => irfft(&x, Some(8), Some(1), FftNorm::Ortho).unwrap(),
        };
        let loss = y.mul(&y).unwrap().sum(None).unwrap();
        let dx = device_tape
            .backward(&loss)
            .unwrap()
            .get(&x)
            .unwrap()
            .unwrap()
            .clone();
        assert_parity(
            &format!("{op} forward: cpu vs {label}"),
            &y_cpu,
            &y.to_tensor(),
        );
        assert_parity(&format!("{op} backward: cpu vs {label}"), &dx_cpu, &dx);
    }
}

/// rfft／irfft の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/fft-rfft-irfft-2631/README.md 参照"]
fn metal_fft_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// rfft／irfft の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/fft-rfft-irfft-2631/README.md 参照"]
fn cuda_fft_matches_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}

fn assert_device_c2c_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    for (name, op) in [("fft", fft as C2cFn), ("ifft", ifft as C2cFn)] {
        let (y_cpu, dx_cpu) = c2c_grad_on(&fandhe_ai::tape(), op, FftNorm::Ortho);
        let (y_dev, dx_dev) = c2c_grad_on(&device_tape, op, FftNorm::Ortho);
        assert_parity(&format!("{name} forward: cpu vs {label}"), &y_cpu, &y_dev);
        assert_parity(
            &format!("{name} backward: cpu vs {label}"),
            &dx_cpu,
            &dx_dev,
        );
    }
}

/// fft／ifft の forward・backward の CPU／Metal 実機比較（#2632）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/fft-fft-ifft-2632/README.md 参照"]
fn metal_fft_c2c_matches_cpu_reference() {
    assert_device_c2c_matches_cpu(Device::Metal, "metal");
}

/// fft／ifft の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較（#2632）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/fft-fft-ifft-2632/README.md 参照"]
fn cuda_fft_c2c_matches_cpu_reference() {
    assert_device_c2c_matches_cpu(Device::Cuda(0), "cuda");
}
