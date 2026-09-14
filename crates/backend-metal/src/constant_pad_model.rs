//! `shaders/constant_pad.metal::constant_pad_f32` の**ホスト側逐語
//! モデル**（イシュー #1756）。
//!
//! `crate::gather_scatter_model`（`soft_f64`・`layout`・`pad` も同様）と
//! 同じ設計判断で `objc2` 系 FFI に一切触れない純粋関数群のため cfg を
//! 付けず、Linux（本実装環境・CI）でも単体テストが回る。カーネル本体
//! （`cfg(target_os = "macos")` 限定の `constant_pad.rs`）は macOS 実機
//! なしにテストできないため、カーネルのアルゴリズムをここで Rust として
//! 逐語再現し、CPU 参照実装（`fandhe_ai_backend_cpu::CpuBackendOps`。
//! テストのみで依存する dev-dependency）との bit 一致を Linux 上で
//! 機械的に固定する。
//!
//! [`constant_pad_model`] はカーネルの添字計算（末尾軸から `%`／`/=` で
//! 剥がす方式。`shaders/constant_pad.metal` を変更した場合は本モジュール
//! も追従させること）と 1 対 1 対応する。

use fandhe_ai_tensor_core::ShapeError;

/// shape の要素数積を検査付きで計算する（オーバーフロー時は
/// `ShapeError::ElementCountOverflow`。`crate::gather_scatter_model::
/// checked_numel` と同型の独立複製）。
fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// `numel` がカーネル引数 `constant uint&`（32bit）へ収まることを検証
/// する（`crate::gather_scatter_model::validate_launch_len` と同型）。
fn validate_launch_len(numel: usize) -> Result<(), ShapeError> {
    if numel > u32::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    Ok(())
}

/// 行優先（row-major）ストライドを計算する（`crate::
/// gather_scatter_model::row_major_strides` と同型の独立複製）。
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// `MetalConstantPad::run_pad_f32`（`crate::constant_pad`）の起動前検査
/// をカーネル本体から切り離した純関数版（`crate::gather_scatter_model::
/// validate_gather_launch` と同じ理由で Linux 実行可能。イシュー #1799
/// 教訓の踏襲）。
///
/// 検査順序: ①各 shape 要素が `u32` に収まること（`in_shape`／
/// `out_shape` とも。カーネル引数 `constant uint*` は 32bit のため）→
/// ②[`fandhe_ai_tensor_core::pad_out_shape`]（rank 一致・出力要素数
/// オーバーフロー検査）→ ③出力要素数が 0 なら早期 `Ok(0)` → ④出力
/// 要素数が `u32` へ収まること → ⑤`in_shape` が空でなければ `input`
/// の実スライス長が `in_shape` の要素数積と一致すること。戻り値は
/// 出力要素数（`numel`。0 の場合はカーネル起動不要を呼び出し元へ
/// 伝える）。
pub fn validate_pad_launch(
    input: &[f32],
    in_shape: &[usize],
    pads: &[(usize, usize)],
    out_shape: &[usize],
) -> Result<usize, ShapeError> {
    for &d in in_shape.iter().chain(out_shape.iter()) {
        if d > u32::MAX as usize {
            return Err(ShapeError::ElementCountOverflow);
        }
    }
    let expected_out_shape = fandhe_ai_tensor_core::pad_out_shape(in_shape, pads)?;
    if expected_out_shape != out_shape {
        return Err(ShapeError::ShapeMismatch {
            lhs: expected_out_shape,
            rhs: out_shape.to_vec(),
        });
    }

    let numel = checked_numel(out_shape)?;
    if numel == 0 {
        return Ok(0);
    }
    validate_launch_len(numel)?;

    let in_is_empty = in_shape.contains(&0);
    if !in_is_empty {
        let in_numel = checked_numel(in_shape)?;
        if input.len() != in_numel {
            return Err(ShapeError::ElementCountMismatch {
                expected: in_numel,
                actual: input.len(),
            });
        }
    }
    Ok(numel)
}

/// `shaders/constant_pad.metal::constant_pad_f32` のホスト側逐語モデル
/// （イシュー #1756）。
///
/// 本関数は `pub` かつ `#[cfg(test)]` の外にあるため、
/// [`crate::gather_scatter_model::gather_model`] と同じ理由で入口に
/// [`validate_pad_launch`] を呼ぶ（本番経路で panic させない方針。
/// `.claude/rules/coding-rust.md`）。
pub fn constant_pad_model(
    input: &[f32],
    in_shape: &[usize],
    pads: &[(usize, usize)],
    out_shape: &[usize],
    value: f32,
) -> Result<Vec<f32>, ShapeError> {
    let numel = validate_pad_launch(input, in_shape, pads, out_shape)?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let in_is_empty = in_shape.contains(&0);
    if in_is_empty {
        return Ok(vec![value; numel]);
    }
    let in_strides = row_major_strides(in_shape);
    let rank = out_shape.len();
    let mut out = vec![0.0f32; numel];
    for (gid, out_slot) in out.iter_mut().enumerate() {
        let mut rem = gid;
        let mut in_flat: i64 = 0;
        let mut inside = true;
        for a in (0..rank).rev() {
            let axis_size = out_shape[a];
            let c = rem % axis_size;
            rem /= axis_size;
            let src_c = c as i64 - pads[a].0 as i64;
            if src_c < 0 || src_c as usize >= in_shape[a] {
                inside = false;
                break;
            }
            in_flat += src_c * in_strides[a] as i64;
        }
        *out_slot = if inside {
            input[in_flat as usize]
        } else {
            value
        };
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    fn cpu_pad(input: &Tensor<f32>, pads: &[(usize, usize)], value: f32) -> Vec<f32> {
        let ops = CpuBackendOps::new();
        let out = ops.pad(input, pads, value).expect("cpu pad succeeds");
        out.contiguous()
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default()
    }

    fn assert_bit_exact(label: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: element {i} bit mismatch (model={x}, cpu={y})"
            );
        }
    }

    #[test]
    fn constant_pad_model_matches_cpu_1d() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let pads = [(1usize, 2usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let model_out = constant_pad_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &pads,
            &out_shape,
            0.0,
        )
        .unwrap();
        let cpu_out = cpu_pad(&x, &pads, 0.0);
        assert_bit_exact("pad_1d", &model_out, &cpu_out);
    }

    #[test]
    fn constant_pad_model_matches_cpu_2d_both_axes() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let pads = [(1usize, 0usize), (0usize, 2usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let model_out = constant_pad_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &pads,
            &out_shape,
            -3.5,
        )
        .unwrap();
        let cpu_out = cpu_pad(&x, &pads, -3.5);
        assert_bit_exact("pad_2d", &model_out, &cpu_out);
    }

    #[test]
    fn constant_pad_model_matches_cpu_3d() {
        let x = Tensor::new((1..=24).map(|v| v as f32).collect(), &[2, 3, 4]).unwrap();
        let pads = [(1usize, 1usize), (0usize, 1usize), (2usize, 0usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let model_out = constant_pad_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &pads,
            &out_shape,
            9.0,
        )
        .unwrap();
        let cpu_out = cpu_pad(&x, &pads, 9.0);
        assert_bit_exact("pad_3d", &model_out, &cpu_out);
    }

    #[test]
    fn constant_pad_model_matches_cpu_empty_input_to_nonempty_output() {
        let x = Tensor::new(Vec::<f32>::new(), &[0, 2]).unwrap();
        let pads = [(1usize, 0usize), (0usize, 0usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let model_out = constant_pad_model(&[], x.shape(), &pads, &out_shape, 5.0).unwrap();
        let cpu_out = cpu_pad(&x, &pads, 5.0);
        assert_bit_exact("pad_empty_input", &model_out, &cpu_out);
    }

    #[test]
    fn constant_pad_model_matches_cpu_all_zero_pads() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let pads = [(0usize, 0usize), (0usize, 0usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let model_out = constant_pad_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &pads,
            &out_shape,
            0.0,
        )
        .unwrap();
        let cpu_out = cpu_pad(&x, &pads, 0.0);
        assert_bit_exact("pad_all_zero", &model_out, &cpu_out);
    }

    #[test]
    fn constant_pad_model_nan_value_class_matches_cpu() {
        let x = Tensor::new(vec![1.0f32], &[1]).unwrap();
        let pads = [(1usize, 1usize)];
        let out_shape = fandhe_ai_tensor_core::pad_out_shape(x.shape(), &pads).unwrap();
        let model_out = constant_pad_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &pads,
            &out_shape,
            f32::NAN,
        )
        .unwrap();
        let cpu_out = cpu_pad(&x, &pads, f32::NAN);
        assert_eq!(model_out.len(), cpu_out.len());
        for (a, b) in model_out.iter().zip(cpu_out.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "NaN class mismatch");
            } else {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    #[test]
    fn validate_pad_launch_rejects_rank_mismatch() {
        let err = validate_pad_launch(&[1.0, 2.0], &[2], &[(1, 0), (0, 0)], &[3, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::RankMismatch { .. }));
    }

    #[test]
    fn validate_pad_launch_rejects_input_len_mismatch() {
        let err = validate_pad_launch(&[1.0], &[2], &[(1, 0)], &[3]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::ElementCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }
}
