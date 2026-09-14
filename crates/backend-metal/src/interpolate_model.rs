//! `shaders/interpolate.metal::interpolate_nearest_f32` の**ホスト側
//! 逐語モデル**（イシュー #1757）。
//!
//! `crate::gather_scatter_model`（`soft_f64`・`layout` も同様）と同じ
//! 設計判断で `objc2` 系 FFI に一切触れない純粋関数群のため cfg を
//! 付けず、Linux（本実装環境・CI）でも単体テストが回る。カーネル本体
//! （`cfg(target_os = "macos")` 限定の `interpolate.rs`）は macOS 実機
//! なしにテストできないため、カーネルのアルゴリズムをここで Rust
//! として逐語再現し、CPU 参照実装（`fandhe_ai_backend_cpu::
//! CpuBackendOps`。テストのみで依存する dev-dependency）との bit 一致
//! を Linux 上で機械的に固定する。
//!
//! [`interpolate_nearest_model`] はカーネルの添字計算（末尾軸から
//! `%`／`/=` で剥がす方式）と 1 対 1 対応する。`shaders/
//! interpolate.metal` を変更した場合は本モジュールも追従させること。
//!
//! `gather_scatter_model.rs::validate_shapes_fit_u32`（`GS_MAX_RANK`
//! 固定長スタック配列方式の rank 上限検査）とは異なり、本モジュールは
//! `crate::gather_scatter_model` の rank 上限を継承しない
//! （interpolate カーネルは gather／scatter と違い座標配列を保持
//! しないため rank 上限を設ける理由がない。`shaders/interpolate.metal`
//! 冒頭コメント参照）。

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

/// `dst`（`out_size` の範囲）に対応する `in_size` 側の添字を返す
/// （`fandhe_ai_autodiff::eval::nearest_src_coord`・
/// `fandhe_ai_backend_cpu::interpolate::nearest_src_coord` と同一の
/// 添字式の独立実装——各クレートは互いに依存できないため重複実装
/// する。`.claude/rules/coding-rust.md`）。
fn nearest_src_coord(dst: usize, in_size: usize, out_size: usize) -> usize {
    if out_size == 0 || in_size == 0 {
        return 0;
    }
    let src = (dst * in_size) / out_size;
    src.min(in_size - 1)
}

/// `MetalInterpolate::run_nearest_f32`（`crate::interpolate`）の起動前
/// 検査をカーネル本体から切り離した純関数版（`crate::
/// gather_scatter_model::validate_gather_launch` と同じ理由で Linux
/// 実行可能。イシュー #1799 教訓の踏襲）。
///
/// 検査順序: ①各 shape 要素が `u32` に収まること（`in_shape`／
/// `out_shape` とも。カーネル引数 `constant uint*` は 32bit のため）→
/// ②[`fandhe_ai_tensor_core::interpolate_out_shape`]（rank・空間軸
/// サイズ・出力要素数オーバーフロー検査）→ ③出力要素数が 0 なら
/// 早期 `Ok(0)` → ④出力要素数が `u32` へ収まること → ⑤`input` の
/// 実スライス長が `in_shape` の要素数積と一致すること → ⑥`in_shape`
/// の行優先ストライド（`crate::interpolate::row_major_strides_u32` が
/// カーネル引数 `constant uint*` へ渡す値と同じ計算）が各軸とも `u32`
/// へ収まること。
///
/// ⑥は①（各次元単体の `u32` 収容）だけでは保証されない: 例えば
/// `in_shape=[2,65536,65536]` は各次元とも `u32::MAX` 未満だが、先頭軸
/// のストライド（`65536 * 65536 = 4294967296 = 2^32`）は `u32` に収まらず
/// `as u32` で `0` へ切り詰まる。切り詰まったストライドをカーネルへ
/// 渡すとバックエンド間数値一致契約（`.claude/rules/coding-rust.md`）
/// に反する誤読（2 番目以降の batch が先頭 batch を読み直す等）を
/// 引き起こすため、切り詰め前にここで検出し型付きエラーを返す
/// （イシュー #1834 codex-review P1 是正）。戻り値は出力要素数
/// （`numel`。0 の場合はカーネル起動不要を呼び出し元へ伝える）。
pub fn validate_interpolate_launch(
    input: &[f32],
    in_shape: &[usize],
    size: &[usize],
    out_shape: &[usize],
) -> Result<usize, ShapeError> {
    for &d in in_shape.iter().chain(out_shape.iter()) {
        if d > u32::MAX as usize {
            return Err(ShapeError::ElementCountOverflow);
        }
    }
    let expected_out_shape = fandhe_ai_tensor_core::interpolate_out_shape(in_shape, size)?;
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

    let in_numel = checked_numel(in_shape)?;
    if input.len() != in_numel {
        return Err(ShapeError::ElementCountMismatch {
            expected: in_numel,
            actual: input.len(),
        });
    }

    for &stride in row_major_strides(in_shape).iter() {
        if stride > u32::MAX as usize {
            return Err(ShapeError::ElementCountOverflow);
        }
    }

    Ok(numel)
}

/// `shaders/interpolate.metal::interpolate_nearest_f32` のホスト側
/// 逐語モデル（イシュー #1757）。
///
/// 本関数は `pub` かつ `#[cfg(test)]` の外にあるため、
/// [`crate::gather_scatter_model::gather_model`] と同じ理由で入口に
/// [`validate_interpolate_launch`] を呼ぶ（本番経路で panic させない
/// 方針。`.claude/rules/coding-rust.md`）。
///
/// 添字の蓄積（`in_flat`）は `u64` で行い、`shaders/interpolate.metal`
/// の `ulong` 演算契約と一致させる（イシュー #1834 codex-review P0
/// 是正。以前は `i64` を使っており、この符号付き／符号なしの差自体は
/// `usize`〈通常 64bit〉が実務上オーバーフローしないホスト環境では
/// 表面化しないが、カーネル側の整数契約〈`ulong` 採用の理由は
/// `interpolate.metal` 冒頭コメント参照〉とモデル側の整数契約を
/// 一致させることを本モジュール doc が要求しているため揃える）。
pub fn interpolate_nearest_model(
    input: &[f32],
    in_shape: &[usize],
    size: &[usize],
    out_shape: &[usize],
) -> Result<Vec<f32>, ShapeError> {
    let numel = validate_interpolate_launch(input, in_shape, size, out_shape)?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let rank = out_shape.len();
    let spatial_start = rank - size.len();
    let in_strides = row_major_strides(in_shape);

    let mut out = vec![0.0f32; numel];
    for (gid, out_slot) in out.iter_mut().enumerate() {
        let mut rem = gid;
        let mut in_flat: u64 = 0;
        for a in (0..rank).rev() {
            let axis_size = out_shape[a];
            let c = rem % axis_size;
            rem /= axis_size;
            let src_c = if a >= spatial_start {
                nearest_src_coord(c, in_shape[a], out_shape[a])
            } else {
                c
            };
            in_flat += src_c as u64 * in_strides[a] as u64;
        }
        *out_slot = input[in_flat as usize];
    }
    Ok(out)
}

/// `MetalInterpolate::run_bilinear_f32`（`crate::interpolate`）の
/// 起動前検査（イシュー #1762。[`validate_interpolate_launch`] の
/// bilinear 版）。`nearest` 版と異なり [`fandhe_ai_tensor_core::
/// interpolate_out_shape_for_mode`]（`Bilinear` mode）で
/// `size.len() == 2` の追加検査を経由する。
///
/// 検査順序: ①各 shape 要素が `u32` に収まること → ②`interpolate_
/// out_shape_for_mode`（rank・空間軸サイズ・出力要素数オーバーフロー・
/// `size.len()==2` を検査）→ ③出力要素数が 0 なら早期 `Ok(0)` →
/// ④出力要素数が `u32` へ収まること → ⑤`input` の実スライス長が
/// `in_shape` の要素数積と一致すること → ⑥`in_shape` の行優先
/// ストライドが各軸とも `u32` へ収まること（[`validate_interpolate_
/// launch`] の同種検査と同じ理由。イシュー #1834 codex-review P1
/// 是正の踏襲）。
pub fn validate_interpolate_bilinear_launch(
    input: &[f32],
    in_shape: &[usize],
    size: &[usize],
    out_shape: &[usize],
) -> Result<usize, ShapeError> {
    for &d in in_shape.iter().chain(out_shape.iter()) {
        if d > u32::MAX as usize {
            return Err(ShapeError::ElementCountOverflow);
        }
    }
    let mode = fandhe_ai_tensor_core::InterpolateMode::Bilinear {
        align_corners: false, // shape 検査は align_corners に依存しない。
    };
    let expected_out_shape =
        fandhe_ai_tensor_core::interpolate_out_shape_for_mode(in_shape, size, mode)?;
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

    let in_numel = checked_numel(in_shape)?;
    if input.len() != in_numel {
        return Err(ShapeError::ElementCountMismatch {
            expected: in_numel,
            actual: input.len(),
        });
    }

    for &stride in row_major_strides(in_shape).iter() {
        if stride > u32::MAX as usize {
            return Err(ShapeError::ElementCountOverflow);
        }
    }

    Ok(numel)
}

/// `shaders/interpolate.metal::interpolate_bilinear_f32` のホスト側
/// 逐語モデル（イシュー #1762）。`fma` を [`f32::mul_add`] で再現する
/// （カーネルの `fma(...)` と一致させる）。座標・重みは
/// `fandhe_ai_tensor_core::interpolate`（`bilinear_scale`／
/// `bilinear_src_coord`）の単一情報源を使う（forward のホスト参照
/// 実装・CPU ネイティブ実装と同じ式）。
pub fn interpolate_bilinear_model(
    input: &[f32],
    in_shape: &[usize],
    size: &[usize],
    out_shape: &[usize],
    align_corners: bool,
) -> Result<Vec<f32>, ShapeError> {
    let numel = validate_interpolate_bilinear_launch(input, in_shape, size, out_shape)?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let rank = out_shape.len();
    let h_axis = rank - 2;
    let w_axis = rank - 1;
    let in_strides = row_major_strides(in_shape);
    let stride_h = in_strides[h_axis];
    let stride_w = in_strides[w_axis];
    let in_h = in_shape[h_axis];
    let in_w = in_shape[w_axis];
    let out_h = out_shape[h_axis];
    let out_w = out_shape[w_axis];
    let scale_h = fandhe_ai_tensor_core::bilinear_scale(in_h, out_h, align_corners);
    let scale_w = fandhe_ai_tensor_core::bilinear_scale(in_w, out_w, align_corners);

    let mut out = vec![0.0f32; numel];
    for (gid, out_slot) in out.iter_mut().enumerate() {
        let mut rem = gid;
        let mut base_flat: u64 = 0;
        let mut cy = 0usize;
        let mut cx = 0usize;
        for a in (0..rank).rev() {
            let axis_size = out_shape[a];
            let c = rem % axis_size;
            rem /= axis_size;
            if a == w_axis {
                cx = c;
            } else if a == h_axis {
                cy = c;
            } else {
                base_flat += c as u64 * in_strides[a] as u64;
            }
        }
        let cyc = fandhe_ai_tensor_core::bilinear_src_coord(cy, in_h, scale_h, align_corners);
        let cxc = fandhe_ai_tensor_core::bilinear_src_coord(cx, in_w, scale_w, align_corners);
        let v00 = input[(base_flat
            + cyc.i0 as u64 * stride_h as u64
            + cxc.i0 as u64 * stride_w as u64) as usize];
        let v01 = input[(base_flat
            + cyc.i0 as u64 * stride_h as u64
            + cxc.i1 as u64 * stride_w as u64) as usize];
        let v10 = input[(base_flat
            + cyc.i1 as u64 * stride_h as u64
            + cxc.i0 as u64 * stride_w as u64) as usize];
        let v11 = input[(base_flat
            + cyc.i1 as u64 * stride_h as u64
            + cxc.i1 as u64 * stride_w as u64) as usize];
        *out_slot =
            fandhe_ai_tensor_core::bilinear_blend(v00, v01, v10, v11, cxc.lambda1, cyc.lambda1);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, InterpolateMode, Tensor};

    fn cpu_interpolate(input: &Tensor<f32>, size: &[usize]) -> Vec<f32> {
        let ops = CpuBackendOps::new();
        let out = ops
            .interpolate(input, size, InterpolateMode::Nearest)
            .expect("cpu interpolate succeeds");
        out.contiguous()
            .as_slice()
            .map(|s| s.to_vec())
            .unwrap_or_default()
    }

    fn cpu_interpolate_bilinear(
        input: &Tensor<f32>,
        size: &[usize],
        align_corners: bool,
    ) -> Vec<f32> {
        let ops = CpuBackendOps::new();
        let mode = InterpolateMode::Bilinear { align_corners };
        let out = ops
            .interpolate(input, size, mode)
            .expect("cpu interpolate bilinear succeeds");
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
    fn interpolate_nearest_model_matches_cpu_1d_upsample() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let size = [6usize];
        let out_shape = fandhe_ai_tensor_core::interpolate_out_shape(x.shape(), &size).unwrap();
        let model_out = interpolate_nearest_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &size,
            &out_shape,
        )
        .unwrap();
        let cpu_out = cpu_interpolate(&x, &size);
        assert_bit_exact("interpolate_1d_upsample", &model_out, &cpu_out);
    }

    #[test]
    fn interpolate_nearest_model_matches_cpu_1d_downsample_non_integer() {
        let x = Tensor::new((1..=8).map(|v| v as f32).collect(), &[8]).unwrap();
        let size = [3usize];
        let out_shape = fandhe_ai_tensor_core::interpolate_out_shape(x.shape(), &size).unwrap();
        let model_out = interpolate_nearest_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &size,
            &out_shape,
        )
        .unwrap();
        let cpu_out = cpu_interpolate(&x, &size);
        assert_bit_exact("interpolate_1d_downsample", &model_out, &cpu_out);
    }

    #[test]
    fn interpolate_nearest_model_matches_cpu_2d_leading_batch_axis() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        let size = [9usize];
        let out_shape = fandhe_ai_tensor_core::interpolate_out_shape(x.shape(), &size).unwrap();
        let model_out = interpolate_nearest_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &size,
            &out_shape,
        )
        .unwrap();
        let cpu_out = cpu_interpolate(&x, &size);
        assert_bit_exact("interpolate_2d_leading_batch", &model_out, &cpu_out);
    }

    #[test]
    fn interpolate_nearest_model_matches_cpu_identity_size() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let size = [4usize];
        let out_shape = fandhe_ai_tensor_core::interpolate_out_shape(x.shape(), &size).unwrap();
        let model_out = interpolate_nearest_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &size,
            &out_shape,
        )
        .unwrap();
        let cpu_out = cpu_interpolate(&x, &size);
        assert_bit_exact("interpolate_identity", &model_out, &cpu_out);
    }

    #[test]
    fn interpolate_nearest_model_nan_inf_class_matches_cpu() {
        let x = Tensor::new(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 1.0], &[4]).unwrap();
        let size = [8usize];
        let out_shape = fandhe_ai_tensor_core::interpolate_out_shape(x.shape(), &size).unwrap();
        let model_out = interpolate_nearest_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &size,
            &out_shape,
        )
        .unwrap();
        let cpu_out = cpu_interpolate(&x, &size);
        assert_eq!(model_out.len(), cpu_out.len());
        for (a, b) in model_out.iter().zip(cpu_out.iter()) {
            if a.is_nan() || b.is_nan() {
                assert!(a.is_nan() && b.is_nan(), "NaN class mismatch");
            } else {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    // --- bilinear（イシュー #1762） ---

    #[test]
    fn interpolate_bilinear_model_matches_cpu_upsample_align_corners_false() {
        let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let size = [5usize, 5];
        let mode = fandhe_ai_tensor_core::InterpolateMode::Bilinear {
            align_corners: false,
        };
        let out_shape =
            fandhe_ai_tensor_core::interpolate_out_shape_for_mode(x.shape(), &size, mode).unwrap();
        let model_out = interpolate_bilinear_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &size,
            &out_shape,
            false,
        )
        .unwrap();
        let cpu_out = cpu_interpolate_bilinear(&x, &size, false);
        assert_bit_exact("interpolate_bilinear_upsample", &model_out, &cpu_out);
    }

    #[test]
    fn interpolate_bilinear_model_matches_cpu_downsample_align_corners_true() {
        let x = Tensor::new((1..=25).map(|v| v as f32 * 0.37).collect(), &[5, 5]).unwrap();
        let size = [2usize, 3];
        let mode = fandhe_ai_tensor_core::InterpolateMode::Bilinear {
            align_corners: true,
        };
        let out_shape =
            fandhe_ai_tensor_core::interpolate_out_shape_for_mode(x.shape(), &size, mode).unwrap();
        let model_out = interpolate_bilinear_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &size,
            &out_shape,
            true,
        )
        .unwrap();
        let cpu_out = cpu_interpolate_bilinear(&x, &size, true);
        assert_bit_exact("interpolate_bilinear_downsample", &model_out, &cpu_out);
    }

    #[test]
    fn interpolate_bilinear_model_matches_cpu_leading_batch_axis() {
        let x = Tensor::new((1..=24).map(|v| v as f32 * 0.1).collect(), &[2, 3, 4]).unwrap();
        let size = [6usize, 5];
        let mode = fandhe_ai_tensor_core::InterpolateMode::Bilinear {
            align_corners: false,
        };
        let out_shape =
            fandhe_ai_tensor_core::interpolate_out_shape_for_mode(x.shape(), &size, mode).unwrap();
        let model_out = interpolate_bilinear_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &size,
            &out_shape,
            false,
        )
        .unwrap();
        let cpu_out = cpu_interpolate_bilinear(&x, &size, false);
        assert_bit_exact(
            "interpolate_bilinear_leading_batch_axis",
            &model_out,
            &cpu_out,
        );
    }

    #[test]
    fn interpolate_bilinear_model_degenerate_in_size_one_matches_cpu() {
        let x = Tensor::new(vec![1.0f32, 5.0], &[1, 2]).unwrap();
        let size = [3usize, 2];
        let mode = fandhe_ai_tensor_core::InterpolateMode::Bilinear {
            align_corners: false,
        };
        let out_shape =
            fandhe_ai_tensor_core::interpolate_out_shape_for_mode(x.shape(), &size, mode).unwrap();
        let model_out = interpolate_bilinear_model(
            x.contiguous().as_slice().unwrap(),
            x.shape(),
            &size,
            &out_shape,
            false,
        )
        .unwrap();
        let cpu_out = cpu_interpolate_bilinear(&x, &size, false);
        assert_bit_exact("interpolate_bilinear_degenerate", &model_out, &cpu_out);
    }

    #[test]
    fn validate_interpolate_bilinear_launch_rejects_size_len_other_than_two() {
        let err = validate_interpolate_bilinear_launch(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &[4], &[4])
            .unwrap_err();
        assert!(matches!(err, ShapeError::RankMismatch { .. }));
    }

    #[test]
    fn validate_interpolate_bilinear_launch_rejects_input_len_mismatch() {
        let err =
            validate_interpolate_bilinear_launch(&[1.0], &[2, 2], &[4, 4], &[4, 4]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::ElementCountMismatch {
                expected: 4,
                actual: 1
            }
        );
    }

    #[test]
    fn validate_interpolate_launch_rejects_rank_mismatch() {
        let err = validate_interpolate_launch(&[1.0, 2.0], &[2], &[3, 3], &[3, 3]).unwrap_err();
        assert!(matches!(err, ShapeError::RankMismatch { .. }));
    }

    // イシュー #1834 codex-review P1 是正の回帰テスト。

    #[test]
    fn row_major_strides_axis0_overflows_u32_for_review_example_shape() {
        // レビュー指摘の再現形状: `in_shape=[2,65536,65536]` は各次元
        // とも `u32::MAX` 未満だが、先頭軸のストライド
        // `65536 * 65536 = 2^32` は `u32` に収まらず `as u32` で `0`
        // へ切り詰まる（`validate_interpolate_launch` が是正前に
        // 見逃していたケース）。総要素数（約 172 億バイト）を実際に
        // 確保するテストは非現実的なため、ストライド計算自体
        // （`row_major_strides`。純粋関数・アロケーションなし）を
        // 直接検証する。
        let strides = row_major_strides(&[2, 65536, 65536]);
        assert_eq!(strides, vec![65536usize * 65536, 65536, 1]);
        assert!(strides[0] > u32::MAX as usize);
        assert_eq!(strides[0] as u32, 0, "u32 へ切り詰めると 0 になる");
    }

    #[test]
    fn validate_interpolate_launch_rejects_input_len_mismatch() {
        let err = validate_interpolate_launch(&[1.0], &[2], &[4], &[4]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::ElementCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    #[test]
    fn validate_interpolate_launch_rejects_zero_spatial_axis() {
        let err = validate_interpolate_launch(&[1.0, 2.0], &[2], &[0], &[0]).unwrap_err();
        assert!(matches!(err, ShapeError::ShapeMismatch { .. }));
    }
}
