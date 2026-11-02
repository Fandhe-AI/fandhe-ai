//! 短時間フーリエ変換 `Stft`／`Istft` の `create_graph` 対応（イシュー #3804・親 #3733・決定記録
//! `docs/autodiff-higher-order-grad-decision.md` §42・§63）。
//!
//! 親モジュール [`super`]（`create_graph.rs`）の `replay_op`（forward 再生）・`build_cgrads`
//! （1 階 VJP の子テープ記録）・`validate_ancestors`（入口の事前拒否）から呼ばれる。
//!
//! **方式（随伴 Op の対。§42.2 案 A）**: `Stft`・`Istft` は入力に対して実線形で、1 階 VJP の
//! ホスト関数（`fft::stft_vjp_host`・`fft::istft_vjp_host`）が `Sᵀ`・`Iᵀ` そのものである。
//! overlap-add を `Var` 演算で組むと O(T) 個のノードが要るため、随伴を内部 Op の対
//! `Op::StftAdjoint`（`Sᵀ g`）・`Op::IstftAdjoint`（`Iᵀ g`）として 1 ノードで積む。
//! - `Stft` の cgrad: `dx = StftAdjoint(g)`。`StftAdjoint` の cgrad: `dg = Stft(up)`。
//! - `Istft` の cgrad: `dX = IstftAdjoint(g)`。`IstftAdjoint` の cgrad: `dg = Istft(up)`。
//!
//! 値は 1 階 VJP と同一のホスト関数を同じ順序で呼ぶため、1 階（`cg.first_order()`）は bit 一致する。
//! `Stft`／`Istft` の再生は `fft_ops::*_from_resolved` を通り、子テープの backend override
//! （`fft_stft`／`fft_istft`）も経由する。随伴 Op はホスト計算のみ（GPU カーネルなし。ROCm を含め
//! 全バックエンドでホストへのフォールバック）。
//!
//! **数値契約**: 新しい数値判定（適用判定・S1）は置かない（§42.14 点 3）。`cg.grad` は 1 階と REQ-2
//! 統一複合判定で一致し、窓は非追跡（窓への勾配は流れない）。`StftPadMode` の未知 variant は
//! 検査せず、ホスト関数の対応範囲に従う（点 4）。保証外は三階以上・真の値が f32 を超える場合。
//!
//! **メモリ見積もり**: 追加ノードは 1 Op あたり 1 個（信号またはスペクトルと同サイズの f32 テンソル
//! 1 本）で、窓は `n_fft` 要素のクローン。確保サイズは `stft_layout`／`istft_layout` が検査する。

use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::fft::{self, IstftParams, StftParams};

use super::value_of;
use crate::error::AutodiffError;
use crate::tape::Op;
use crate::var::Var;

fn invalid(msg: String) -> AutodiffError {
    AutodiffError::Backward(format!("create_graph: {msg}"))
}

fn check_window_len(window: &Tensor<f32>, n_fft: usize, what: &str) -> Result<(), AutodiffError> {
    if window.shape() != [n_fft] {
        return Err(invalid(format!(
            "{what}: 窓 shape {:?} が [n_fft = {n_fft}] と一致しない",
            window.shape()
        )));
    }
    Ok(())
}

/// `Op::Stft` の事前検査（`validate_ancestors` から）。layout の再計算と窓長。
pub(super) fn check_stft(
    input_shape: &[usize],
    window: &Tensor<f32>,
    params: &StftParams,
) -> Result<(), AutodiffError> {
    fft::stft_layout(input_shape, params)?;
    check_window_len(window, params.n_fft(), "Stft")
}

/// `Op::Istft` の事前検査。layout・窓長・NOLA。
pub(super) fn check_istft(
    input_shape: &[usize],
    window: &Tensor<f32>,
    params: &IstftParams,
) -> Result<(), AutodiffError> {
    let layout = fft::istft_layout(input_shape, params)?;
    check_window_len(window, params.n_fft(), "Istft")?;
    fft::istft_check_nola(&window.host_slice(), &layout)?;
    Ok(())
}

/// `Op::StftAdjoint` の事前検査。入力（スペクトル）shape が own（信号）shape の layout と一致すること。
pub(super) fn check_stft_adjoint(
    input_shape: &[usize],
    own_shape: &[usize],
    window: &Tensor<f32>,
    params: &StftParams,
) -> Result<(), AutodiffError> {
    let layout = fft::stft_layout(own_shape, params)?;
    if layout.out_shape() != input_shape {
        return Err(invalid(format!(
            "StftAdjoint: 入力 shape {input_shape:?} が layout の {:?} と一致しない",
            layout.out_shape()
        )));
    }
    check_window_len(window, params.n_fft(), "StftAdjoint")
}

/// `Op::IstftAdjoint` の事前検査。入力（信号）shape が own（スペクトル）shape の layout と一致し、NOLA を満たすこと。
pub(super) fn check_istft_adjoint(
    input_shape: &[usize],
    own_shape: &[usize],
    window: &Tensor<f32>,
    params: &IstftParams,
) -> Result<(), AutodiffError> {
    let layout = fft::istft_layout(own_shape, params)?;
    if layout.out_shape() != input_shape {
        return Err(invalid(format!(
            "IstftAdjoint: 入力 shape {input_shape:?} が layout の {:?} と一致しない",
            layout.out_shape()
        )));
    }
    check_window_len(window, params.n_fft(), "IstftAdjoint")?;
    fft::istft_check_nola(&window.host_slice(), &layout)?;
    Ok(())
}

/// `Sᵀ g`（`g`: スペクトル shape → 信号 shape `signal_shape`）を子テープへ `push_eager` する。
/// 信号長はフレーム数から一意に決まらないため、呼び出し側が `signal_shape` を渡す。
pub(super) fn stft_adjoint_var<'c>(
    g: &Var<'c>,
    window: &Tensor<f32>,
    params: &StftParams,
    signal_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let layout = fft::stft_layout(signal_shape, params)?;
    if g.shape() != layout.out_shape() {
        return Err(invalid(format!(
            "stft_adjoint: g shape {:?} が layout の {:?} と一致しない",
            g.shape(),
            layout.out_shape()
        )));
    }
    let tape = g.tape();
    let g_val = value_of(tape, g)?;
    let data = fft::stft_vjp_host(&g_val.host_slice(), &window.host_slice(), &layout)?;
    let value = Tensor::new(data, signal_shape).map_err(AutodiffError::Shape)?;
    let id = tape.push_eager(
        Op::StftAdjoint {
            input: g.node_id(),
            window: window.clone(),
            params: params.clone(),
        },
        value,
    );
    Ok(Var::from_raw(tape, id))
}

/// `Iᵀ g`（`g`: 信号 shape → スペクトル shape `spectrum_shape`）を子テープへ `push_eager` する。
pub(super) fn istft_adjoint_var<'c>(
    g: &Var<'c>,
    window: &Tensor<f32>,
    params: &IstftParams,
    spectrum_shape: &[usize],
) -> Result<Var<'c>, AutodiffError> {
    let layout = fft::istft_layout(spectrum_shape, params)?;
    if g.shape() != layout.out_shape() {
        return Err(invalid(format!(
            "istft_adjoint: g shape {:?} が layout の {:?} と一致しない",
            g.shape(),
            layout.out_shape()
        )));
    }
    fft::istft_check_nola(&window.host_slice(), &layout)?;
    let tape = g.tape();
    let g_val = value_of(tape, g)?;
    let data = fft::istft_vjp_host(&g_val.host_slice(), &window.host_slice(), &layout)?;
    let value = Tensor::new(data, spectrum_shape).map_err(AutodiffError::Shape)?;
    let id = tape.push_eager(
        Op::IstftAdjoint {
            input: g.node_id(),
            window: window.clone(),
            params: params.clone(),
        },
        value,
    );
    Ok(Var::from_raw(tape, id))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft_ops::{istft_from_resolved, stft_from_resolved};
    use crate::tape::Tape;
    use fandhe_ai_tensor_core::StftPadMode;

    fn data(n: usize, phase: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 + phase) * 0.7).sin() + 0.1 * i as f32 - 0.2)
            .collect()
    }

    fn window(n: usize) -> Tensor<f32> {
        Tensor::new(
            (0..n)
                .map(|i| 0.55 - 0.5 * (std::f32::consts::TAU * i as f32 / n as f32).cos())
                .collect(),
            &[n],
        )
        .unwrap()
    }

    fn inner_prod(a: &Tensor<f32>, b: &Tensor<f32>) -> f64 {
        let (a, b) = (a.contiguous(), b.contiguous());
        a.as_slice()
            .unwrap()
            .iter()
            .zip(b.as_slice().unwrap())
            .map(|(&x, &y)| f64::from(x) * f64::from(y))
            .sum()
    }

    fn close(a: f64, b: f64) -> bool {
        let d = (a - b).abs();
        d < 1e-5 || d / a.abs().max(b.abs()).max(1e-12) < 1e-3
    }

    /// `<stft(x), y> = <x, StftAdjoint(y)>`（`Sᵀ` が `S` の随伴であること）。
    #[test]
    fn stft_family_adjoint_ops_are_mutually_adjoint() {
        // (shape, n_fft, hop, center, mode, normalized, onesided)
        let cases = [
            (
                vec![16usize],
                8usize,
                2usize,
                true,
                StftPadMode::Reflect,
                false,
                true,
            ),
            (vec![16], 8, 3, true, StftPadMode::Constant, true, true),
            (vec![15], 8, 3, false, StftPadMode::Reflect, false, false),
            (vec![2, 14], 6, 4, true, StftPadMode::Reflect, true, true),
        ];
        for (shape, n_fft, hop, center, mode, norm, one) in cases {
            let params = StftParams::new(n_fft, Some(hop), center, mode, norm, one).unwrap();
            let win = window(n_fft);
            let tape = Tape::new();
            let n: usize = shape.iter().product();
            let x = tape.var_no_grad(&Tensor::new(data(n, 1.0), &shape).unwrap());
            let sx = stft_from_resolved(&x, &win, &params).unwrap();
            let sshape = sx.shape();
            let m: usize = sshape.iter().product();
            let y = tape.var_no_grad(&Tensor::new(data(m, 3.0), &sshape).unwrap());
            let aty = stft_adjoint_var(&y, &win, &params, &shape).unwrap();
            let lhs = inner_prod(&sx.to_tensor(), &y.to_tensor());
            let rhs = inner_prod(&x.to_tensor(), &aty.to_tensor());
            assert!(close(lhs, rhs), "stft {shape:?} {mode:?}: {lhs} vs {rhs}");
            // 窓の shape 不整合は型付きで拒否される。
            assert!(check_stft(&shape, &window(n_fft + 1), &params).is_err());
            assert!(check_stft_adjoint(&sshape, &shape, &win, &params).is_ok());
        }
    }

    /// `<istft(X), y> = <X, IstftAdjoint(y)>`（`length` の短縮・延長、`onesided = Some(false)` を含む）。
    #[test]
    fn istft_family_adjoint_ops_are_mutually_adjoint() {
        // (spectrum shape, n_fft, hop, center, norm, onesided, length)
        let cases = [
            (vec![5usize, 5, 2], 8usize, 2usize, true, false, None, None),
            (vec![5, 5, 2], 8, 2, true, true, Some(true), Some(6)),
            (vec![5, 5, 2], 8, 2, true, false, None, Some(14)),
            (vec![8, 4, 2], 8, 2, false, false, Some(false), None),
            (vec![2, 4, 5, 2], 6, 3, false, false, None, None),
        ];
        for (shape, n_fft, hop, center, norm, one, length) in cases {
            let params =
                IstftParams::new(n_fft, Some(hop), None, center, norm, one, length).unwrap();
            let win = window(n_fft);
            let tape = Tape::new();
            let n: usize = shape.iter().product();
            let xs = tape.var_no_grad(&Tensor::new(data(n, 2.0), &shape).unwrap());
            let ix = istft_from_resolved(&xs, &win, &params).unwrap();
            let ishape = ix.shape();
            let m: usize = ishape.iter().product();
            let y = tape.var_no_grad(&Tensor::new(data(m, 4.0), &ishape).unwrap());
            let aty = istft_adjoint_var(&y, &win, &params, &shape).unwrap();
            let lhs = inner_prod(&ix.to_tensor(), &y.to_tensor());
            let rhs = inner_prod(&xs.to_tensor(), &aty.to_tensor());
            assert!(close(lhs, rhs), "istft {shape:?}: {lhs} vs {rhs}");
            assert!(check_istft_adjoint(&ishape, &shape, &win, &params).is_ok());
            // shape 不一致は拒否される。
            assert!(check_istft_adjoint(&shape, &shape, &win, &params).is_err());
        }
    }
}
