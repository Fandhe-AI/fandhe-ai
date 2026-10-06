//! `fandhe_ai_autodiff::conv_transpose3d_ops`／`max_unpool_ops`（イシュー #2644。facade 非公開のため
//! `fandhe_ai_autodiff::*` を直接 use する。各モジュール doc 参照）のバックエンド間 parity テスト
//! （`pool3d_ops_backend_parity.rs` と同型）。
//!
//! 属性なし（`fandhe_ai::tape()`〈`CpuBackendOps`〉と `fandhe_ai_autodiff::Tape::new()`〈`NaiveOps`＝
//! 既定 `Unsupported` → ホストフォールバック〉の突き合わせ）:
//! - ConvTranspose3d の forward・`d_input`・`d_weight`・`d_bias` は REQ-2 統一複合判定
//!   （`fandhe_ai_backend_cpu::parity::assert_parity`）。GEMM の実装がバックエンドで異なる
//!   （`CpuBackendOps` の `gemm_batched` 対 naive の既定合成）ため bit 一致は求めない。
//! - MaxUnpool は算術を含まないコピー（forward）・gather×マスク（勾配）なので **bit 一致**
//!   （重複索引を含めて last-writer の決定的契約をバックエンド間で固定する）。
//!
//! `#[ignore]`（`tape_for(Device::Cuda(0))`／`tape_for(Device::Metal)`〈`cfg(target_os = "macos")`
//! 限定〉で同じ経路を CPU tape と比較）: 実機への到達手段が本エージェント実行環境にないため未実施の
//! まま GB10／Mac セッションへ申し送る（`docs/perf/logs/conv-transpose3d-max-unpool-2644/
//! README.md`）。ConvTranspose3d は `gemm_batched` がデバイス上で走り `im2col3d`／`col2im3d` は
//! `Unsupported` → ホスト、MaxUnpool は既存の GPU `scatter`／`gather` が走る。本テストは GPU
//! カーネル新設の parity ではなく「デバイス経路が CPU tape と同じ結果になること」の確認である。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::conv_transpose3d_ops::conv_transpose3d;
use fandhe_ai_autodiff::max_unpool_ops::{max_unpool1d, max_unpool2d, max_unpool3d};
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

fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
    Tensor::new(data, shape).expect("test fixture: shape 一致")
}

fn wave(n: usize, freq: f32, amp: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * freq).sin() * amp).collect()
}

// ConvTranspose3d: groups=2・stride/padding/output_padding/dilation すべて非自明。
const CT_X: [usize; 5] = [2, 4, 3, 2, 2];
const CT_W: [usize; 5] = [4, 3, 2, 2, 2];
const CT_STRIDE: [usize; 3] = [2, 1, 1];
const CT_PADDING: [usize; 3] = [1, 0, 0];
const CT_OP: [usize; 3] = [1, 0, 0];
const CT_DILATION: [usize; 3] = [1, 1, 2];
const CT_GROUPS: usize = 2;

struct Outputs {
    ct_out: Tensor<f32>,
    ct_dx: Tensor<f32>,
    ct_dw: Tensor<f32>,
    ct_db: Tensor<f32>,
    unpool: Vec<(Tensor<f32>, Tensor<f32>)>,
}

macro_rules! outputs_on {
    ($tape:expr) => {{
        let tape = $tape;
        // --- ConvTranspose3d ---
        let x = tape.make_var(&t(wave(CT_X.iter().product(), 0.043, 1.0), &CT_X));
        let w = tape.make_var(&t(wave(CT_W.iter().product(), 0.061, 0.5), &CT_W));
        let b = tape.make_var(&t(vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6], &[6]));
        let y = conv_transpose3d(
            &x,
            &w,
            Some(&b),
            CT_STRIDE,
            CT_PADDING,
            CT_OP,
            CT_DILATION,
            CT_GROUPS,
        )
        .unwrap();
        let ct_out = y.to_tensor();
        let n_out: usize = ct_out.shape().iter().product();
        let g = tape.make_var(&t(wave(n_out, 0.053, 0.7), ct_out.shape()));
        let loss = y.mul(&g).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let ct_dx = grads.get(&x).unwrap().unwrap().clone();
        let ct_dw = grads.get(&w).unwrap().unwrap().clone();
        let ct_db = grads.get(&b).unwrap().unwrap().clone();

        // --- MaxUnpool 1d／2d／3d（各 1 件は重複索引を含む） ---
        let mut unpool = Vec::new();
        macro_rules! unpool_case {
            ($x_shape:expr, $idx:expr, |$x:ident, $i:ident| $body:expr) => {{
                let n: usize = $x_shape.iter().product();
                let xv = tape.make_var(&t(wave(n, 0.37, 1.0), &$x_shape));
                let idx = ti($idx, &$x_shape);
                let y = {
                    let ($x, $i) = (&xv, &idx);
                    $body
                }
                .unwrap();
                let out = y.to_tensor();
                let m: usize = out.shape().iter().product();
                let gv = tape.make_var(&t(wave(m, 0.11, 0.9), out.shape()));
                let loss = y.mul(&gv).unwrap().sum(None).unwrap();
                let dx = tape
                    .backward(&loss)
                    .unwrap()
                    .get(&xv)
                    .unwrap()
                    .unwrap()
                    .clone();
                unpool.push((out, dx));
            }};
        }
        // 1d: [2,2,3] -> 出力平面 6。(n,c) 平面ごとに索引を持ち、平面 3 に重複（2 と 2）を含む。
        unpool_case!(
            [2, 2, 3],
            vec![1, 0, 5, 2, 4, 3, 1, 1, 5, 0, 2, 4],
            |x, i| max_unpool1d(x, i, 2, None, 0, None)
        );
        // 2d: [1,2,2,3] -> 出力平面 4×6=24。
        unpool_case!(
            [1, 2, 2, 3],
            vec![2, 9, 16, 23, 6, 13, 0, 0, 5, 23, 11, 12],
            |x, i| max_unpool2d(x, i, [2, 2], None, [0, 0], None)
        );
        // 3d: [1,1,2,2,2] -> 出力平面 4×4×4=64。
        unpool_case!(
            [1, 1, 2, 2, 2],
            vec![0, 63, 21, 42, 7, 35, 56, 14],
            |x, i| max_unpool3d(x, i, [2, 2, 2], None, [0, 0, 0], None)
        );
        Outputs {
            ct_out,
            ct_dx,
            ct_dw,
            ct_db,
            unpool,
        }
    }};
}

fn cpu_outputs() -> Outputs {
    outputs_on!(fandhe_ai::tape())
}

fn naive_outputs() -> Outputs {
    outputs_on!(fandhe_ai_autodiff::Tape::new())
}

fn assert_parity(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    fandhe_ai_backend_cpu::parity::assert_parity(
        label,
        a.host_slice().as_ref(),
        b.host_slice().as_ref(),
    );
}

fn assert_bits_eq(label: &str, a: &Tensor<f32>, b: &Tensor<f32>) {
    assert_eq!(a.shape(), b.shape(), "{label}: shape");
    for (i, (x, y)) in a.host_slice().iter().zip(b.host_slice().iter()).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}[{i}]: {x} vs {y}");
    }
}

fn assert_outputs_match(label: &str, a: &Outputs, b: &Outputs) {
    assert_parity(
        &format!("{label}: conv_transpose3d forward"),
        &a.ct_out,
        &b.ct_out,
    );
    assert_parity(
        &format!("{label}: conv_transpose3d d_input"),
        &a.ct_dx,
        &b.ct_dx,
    );
    assert_parity(
        &format!("{label}: conv_transpose3d d_weight"),
        &a.ct_dw,
        &b.ct_dw,
    );
    assert_parity(
        &format!("{label}: conv_transpose3d d_bias"),
        &a.ct_db,
        &b.ct_db,
    );
    assert_eq!(a.unpool.len(), b.unpool.len());
    for (i, ((ao, ag), (bo, bg))) in a.unpool.iter().zip(&b.unpool).enumerate() {
        assert_bits_eq(&format!("{label}: max_unpool case {i} forward"), ao, bo);
        assert_bits_eq(&format!("{label}: max_unpool case {i} backward"), ag, bg);
    }
}

#[test]
fn cpu_matches_naive_reference() {
    let cpu = cpu_outputs();
    let naive = naive_outputs();
    assert_outputs_match("cpu vs naive", &cpu, &naive);
    // 期待値の健全性（両者が同じ誤りで一致していないこと）。
    // D: (3-1)*2-2+1*(2-1)+1+1=5, H: 2+1=3, W: 1+2*1+1=4 -> [2, 6, 5, 3, 4]。
    assert_eq!(cpu.ct_out.shape(), &[2, 6, 5, 3, 4]);
    assert_eq!(cpu.ct_dx.shape(), &CT_X);
    assert_eq!(cpu.ct_dw.shape(), &CT_W);
    // 1d: 平面 2（入力 6..9・索引 [1, 1, 5]）は入力 6 と 7 が出力位置 1 を争い、後に書く 7 が勝つ。
    // 敗者（6）の勾配は 0、勝者（7）は非 0。
    let (out1d, dx1d) = &cpu.unpool[0];
    assert_eq!(out1d.shape(), &[2, 2, 6]);
    let dx = dx1d.host_slice();
    assert_eq!(dx[6], 0.0);
    assert_ne!(dx[7], 0.0);
    // 2d: 平面 1（入力 6..12・索引 [0, 0, 5, ...]）も先頭 2 要素が重複し、敗者（6）の勾配は 0。
    let (_, dx2d) = &cpu.unpool[1];
    assert_eq!(dx2d.host_slice()[6], 0.0);
}

// ---------------------------------------------------------------------
// 実機バックエンド（`#[ignore]`）: GB10／Mac 実機セッションへ申し送る
// （`docs/perf/logs/conv-transpose3d-max-unpool-2644/README.md`）。
// ---------------------------------------------------------------------

fn assert_device_matches_cpu(device: Device, label: &str) {
    let device_tape =
        fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let cpu = cpu_outputs();
    let dev = outputs_on!(device_tape);
    assert_outputs_match(&format!("cpu vs {label}"), &cpu, &dev);
}

/// ConvTranspose3d・MaxUnpool の forward・backward の CPU／Metal 実機比較。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機が必要。docs/perf/logs/conv-transpose3d-max-unpool-2644/README.md 参照"]
fn metal_conv_transpose3d_max_unpool_match_cpu_reference() {
    assert_device_matches_cpu(Device::Metal, "metal");
}

/// ConvTranspose3d・MaxUnpool の forward・backward の CPU／CUDA 実機（DGX Spark GB10）比較。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）が必要。docs/perf/logs/conv-transpose3d-max-unpool-2644/README.md 参照"]
fn cuda_conv_transpose3d_max_unpool_match_cpu_reference() {
    assert_device_matches_cpu(Device::Cuda(0), "cuda");
}
