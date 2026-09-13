//! `shaders/gather_scatter.metal` の 3 カーネル（`gather_f32`／
//! `scatter_overwrite_f32`／`scatter_add_f32`）の**ホスト側逐語モデル**
//! （イシュー #1778）。
//!
//! `crate::soft_f64`（`layout.rs`・`pad.rs` と同じく `cfg(target_os =
//! "macos")` を付けない設計判断）と同様に、本モジュールは `objc2` 系
//! FFI に一切触れない純粋関数群のため cfg を付けず、Linux（本実装環境・
//! CI）でも単体テストが回る。カーネル本体（`cfg(target_os = "macos")`
//! 限定の `gather_scatter.rs`）は macOS 実機なしにテストできないため、
//! カーネルのアルゴリズムをここで Rust として逐語再現し、CPU 参照実装
//! （`fandhe_ai_backend_cpu::CpuBackendOps`。テストのみで依存する
//! dev-dependency）との bit 一致を Linux 上で機械的に固定する。
//!
//! `gather_model`／`scatter_model` はカーネルの添字計算（`gs_unravel`／
//! `gs_ravel`。row-major）・`Add` の soft-f64 逐次加算
//! （`crate::soft_f64::{widen_f32_bits, add_f64_bits, narrow_f64_bits}`）
//! と 1 対 1 対応する。`gather_scatter.metal` を変更した場合は本モジュール
//! も追従させること。

use crate::soft_f64::{add_f64_bits, narrow_f64_bits, widen_f32_bits};
use fandhe_ai_tensor_core::{ScatterReduce, ShapeError};

/// `shaders/gather_scatter.metal::GS_MAX_RANK` と一致させる rank 上限。
/// カーネル側は `thread ulong coords[GS_MAX_RANK]` のスタック配列を
/// 使うため、これを超える rank は起動前に拒否する（ホスト側検査。
/// `gather_scatter.rs::MetalGatherScatter::run_gather_f32`／
/// `run_scatter_f32` が呼び出し前に本モジュールの検証関数を経由する）。
pub const GS_MAX_RANK: usize = 8;

/// `rank` が [`GS_MAX_RANK`] を超えないこと、各 shape 要素が `u32` に
/// 収まることを検証する（カーネル引数 `constant uint* shapes` は 32bit
/// のため。`elementwise.rs::validate_elementwise_len` と同じ理由。
/// OWASP A03・`.claude/rules/security.md`）。
pub fn validate_shapes_fit_u32(shapes: &[&[usize]]) -> Result<(), ShapeError> {
    for shape in shapes {
        if shape.len() > GS_MAX_RANK {
            return Err(ShapeError::RankMismatch {
                expected: GS_MAX_RANK,
                actual: shape.len(),
            });
        }
        for &d in shape.iter() {
            if d > u32::MAX as usize {
                return Err(ShapeError::ElementCountOverflow);
            }
        }
    }
    Ok(())
}

/// `index` の全要素が `[0, dim_size)` の範囲内であることを検証する
/// （`gather_scatter.rs` がカーネル起動前に行うホスト側検査。
/// `crates/backend-cpu/src/gather_scatter.rs` と同じ独立検査方針——
/// `Var` 側の検査と重複するが判定迂回経路を作らない。
/// `.claude/rules/security.md` A08）。
pub fn validate_index_range(index: &[i32], dim: usize, dim_size: usize) -> Result<(), ShapeError> {
    for &raw in index {
        if raw < 0 || (raw as usize) >= dim_size {
            return Err(ShapeError::IndexOutOfRange {
                dim,
                index: raw as i64,
                dim_size,
            });
        }
    }
    Ok(())
}

/// 行優先（row-major）ストライドを計算する。
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// 線形添字を `shape` 上で row-major に多次元添字へ展開する
/// （`shaders/gather_scatter.metal::gs_unravel` の逐語再現）。
fn unravel(mut idx: usize, shape: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let d = shape[axis];
        if d == 0 {
            out[axis] = 0;
            continue;
        }
        out[axis] = idx % d;
        idx /= d;
    }
    out
}

/// 多次元添字を row-major で線形添字へ畳み込む
/// （`shaders/gather_scatter.metal::gs_ravel` の逐語再現）。
fn ravel(coords: &[usize], strides: &[usize]) -> usize {
    coords
        .iter()
        .zip(strides.iter())
        .map(|(&c, &s)| c * s)
        .sum()
}

/// shape の要素数積を検査付きで計算する（オーバーフロー時は
/// `ShapeError::ElementCountOverflow`。`crates/backend-cpu/src/
/// gather_scatter.rs::checked_numel` と同型の独立複製）。
fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// `gather_f32` カーネルのホスト側逐語モデル（イシュー #1778）。
///
/// 呼び出し元（`gather_scatter.rs`）が shape 検査
/// （[`fandhe_ai_tensor_core::gather_out_shape`]）・[`validate_shapes_fit_u32`]・
/// [`validate_index_range`] を済ませてから渡す契約のため、本関数自身は
/// 値検査を行わない（カーネル同様に、範囲外添字は呼び出し元が事前に
/// 拒否している前提。防御的ガードはカーネル側にのみ持たせ、ホスト
/// モデルはカーネルの「正常系」の逐語再現に専念する）。
pub fn gather_model(
    input: &[f32],
    in_shape: &[usize],
    index: &[i32],
    index_shape: &[usize],
    dim: usize,
) -> Vec<f32> {
    let in_strides = row_major_strides(in_shape);
    let numel = index_shape.iter().product::<usize>();
    let mut out = vec![0.0f32; numel];
    for (gid, out_slot) in out.iter_mut().enumerate() {
        let mut coords = unravel(gid, index_shape);
        let dim_idx = index[gid] as usize;
        coords[dim] = dim_idx;
        let src_off = ravel(&coords, &in_strides);
        *out_slot = input[src_off];
    }
    out
}

/// `scatter_overwrite_f32`／`scatter_add_f32` カーネルのホスト側逐語
/// モデル（イシュー #1778）。出力定常方式（`shaders/gather_scatter.metal`
/// 冒頭コメント「CPU 参照実装との順序等価性」）を Rust で再現する。
///
/// `out_shape`（＝`input.shape()`）・`index_shape`（＝`src.shape()`）は
/// 呼び出し元が shape 検査（[`fandhe_ai_tensor_core::scatter_out_shape`]）・
/// [`validate_shapes_fit_u32`]・[`validate_index_range`] 済みで渡す契約。
pub fn scatter_model(
    input: &[f32],
    out_shape: &[usize],
    index: &[i32],
    index_shape: &[usize],
    src: &[f32],
    dim: usize,
    reduce: ScatterReduce,
) -> Result<Vec<f32>, ShapeError> {
    let rank = out_shape.len();
    let numel_out = checked_numel(out_shape)?;
    let index_strides = row_major_strides(index_shape);
    let mut out = vec![0.0f32; numel_out];

    for (gid, out_slot) in out.iter_mut().enumerate() {
        let pos = unravel(gid, out_shape);
        let mut in_range = true;
        for axis in 0..rank {
            if axis == dim {
                continue;
            }
            if pos[axis] >= index_shape[axis] {
                in_range = false;
                break;
            }
        }

        if !in_range {
            *out_slot = input[gid];
            continue;
        }

        match reduce {
            ScatterReduce::Add => {
                let mut acc = widen_f32_bits(input[gid].to_bits());
                let mut idx_coords = pos.clone();
                for j in 0..index_shape[dim] {
                    idx_coords[dim] = j;
                    let idx_off = ravel(&idx_coords, &index_strides);
                    let dim_idx = index[idx_off] as usize;
                    if dim_idx == pos[dim] {
                        acc = add_f64_bits(acc, widen_f32_bits(src[idx_off].to_bits()));
                    }
                }
                *out_slot = f32::from_bits(narrow_f64_bits(acc));
            }
            // `Overwrite`、および `ScatterReduce`（`#[non_exhaustive]`）の
            // 未知 variant は同じ「上書き」意味論へフォールバックする
            // （CPU 参照実装・カーネル側と同じ安全側の割り切り方針）。
            _ => {
                let mut result = input[gid];
                let mut idx_coords = pos.clone();
                for j in 0..index_shape[dim] {
                    idx_coords[dim] = j;
                    let idx_off = ravel(&idx_coords, &index_strides);
                    let dim_idx = index[idx_off] as usize;
                    if dim_idx == pos[dim] {
                        result = src[idx_off];
                    }
                }
                *out_slot = result;
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bench_harness::rng::Xorshift64Star;
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    fn i32_index(seed: u64, numel: usize, dim_size: usize) -> Vec<i32> {
        Xorshift64Star::new(seed)
            .fill_vec(numel)
            .into_iter()
            .map(|v| {
                // `v` は `[-1, 1)`。`[0, dim_size)` へ写像する。
                let unit = (v + 1.0) / 2.0;
                let idx = (unit * dim_size as f32) as usize;
                idx.min(dim_size.saturating_sub(1)) as i32
            })
            .collect()
    }

    fn assert_gather_matches_cpu(in_shape: &[usize], index_shape: &[usize], dim: usize, seed: u64) {
        let cpu = CpuBackendOps::new();
        let in_numel: usize = in_shape.iter().product();
        let idx_numel: usize = index_shape.iter().product();
        let input_data = Xorshift64Star::new(seed).fill_vec(in_numel);
        let index_data = i32_index(seed.wrapping_add(1), idx_numel, in_shape[dim]);

        let input = Tensor::new(input_data.clone(), in_shape).unwrap();
        let index = Tensor::<i32>::new(index_data.clone(), index_shape).unwrap();
        let cpu_out = cpu.gather(&input, dim, &index).unwrap();

        let model_out = gather_model(&input_data, in_shape, &index_data, index_shape, dim);
        assert_eq!(
            model_out.len(),
            cpu_out.as_slice().unwrap().len(),
            "gather_model と CPU 出力の要素数が不一致"
        );
        for (a, b) in model_out.iter().zip(cpu_out.as_slice().unwrap().iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "gather_model が CPU 参照実装と bit 不一致"
            );
        }
    }

    fn assert_scatter_matches_cpu(
        out_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
        reduce: ScatterReduce,
        seed: u64,
    ) {
        let cpu = CpuBackendOps::new();
        let out_numel: usize = out_shape.iter().product();
        let idx_numel: usize = index_shape.iter().product();
        let input_data = Xorshift64Star::new(seed).fill_vec(out_numel);
        let index_data = i32_index(seed.wrapping_add(1), idx_numel, out_shape[dim]);
        let src_data = Xorshift64Star::new(seed.wrapping_add(2)).fill_vec(idx_numel);

        let input = Tensor::new(input_data.clone(), out_shape).unwrap();
        let index = Tensor::<i32>::new(index_data.clone(), index_shape).unwrap();
        let src = Tensor::new(src_data.clone(), index_shape).unwrap();
        let cpu_out = cpu.scatter(&input, dim, &index, &src, reduce).unwrap();

        let model_out = scatter_model(
            &input_data,
            out_shape,
            &index_data,
            index_shape,
            &src_data,
            dim,
            reduce,
        )
        .unwrap();

        assert_eq!(model_out.len(), cpu_out.as_slice().unwrap().len());
        for (a, b) in model_out.iter().zip(cpu_out.as_slice().unwrap().iter()) {
            match reduce {
                ScatterReduce::Add => assert!(
                    crate::soft_f64::f32_bits_match(*a, *b),
                    "scatter_model(Add) が CPU 参照実装と不一致: {a} vs {b}"
                ),
                _ => assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "scatter_model(Overwrite) が CPU 参照実装と bit 不一致"
                ),
            }
        }
    }

    #[test]
    fn gather_matches_cpu_1d() {
        assert_gather_matches_cpu(&[5], &[8], 0, 1);
    }

    #[test]
    fn gather_matches_cpu_2d_each_dim() {
        assert_gather_matches_cpu(&[3, 4], &[3, 6], 1, 2);
        assert_gather_matches_cpu(&[3, 4], &[6, 4], 0, 3);
    }

    #[test]
    fn gather_matches_cpu_3d() {
        assert_gather_matches_cpu(&[2, 3, 4], &[2, 3, 7], 2, 4);
    }

    #[test]
    fn scatter_overwrite_matches_cpu_2d() {
        assert_scatter_matches_cpu(&[3, 4], &[3, 2], 1, ScatterReduce::Overwrite, 10);
        assert_scatter_matches_cpu(&[3, 4], &[2, 4], 0, ScatterReduce::Overwrite, 11);
    }

    #[test]
    fn scatter_add_matches_cpu_2d() {
        assert_scatter_matches_cpu(&[3, 4], &[3, 2], 1, ScatterReduce::Add, 12);
        assert_scatter_matches_cpu(&[3, 4], &[2, 4], 0, ScatterReduce::Add, 13);
    }

    #[test]
    fn scatter_matches_cpu_3d_add() {
        assert_scatter_matches_cpu(&[2, 3, 4], &[2, 3, 4], 2, ScatterReduce::Add, 14);
    }

    #[test]
    fn scatter_add_duplicate_indices_matches_cpu() {
        // 全 index が同一位置を指す（duplicate）ケースを固定 index で検証。
        let cpu = CpuBackendOps::new();
        let out_shape = [1usize, 3usize];
        let index_shape = [1usize, 3usize];
        let input_data = vec![10.0f32, 0.0, 0.0];
        let index_data = vec![0i32, 0, 0];
        let src_data = vec![1.0f32, 2.0, 3.0];

        let input = Tensor::new(input_data.clone(), &out_shape).unwrap();
        let index = Tensor::<i32>::new(index_data.clone(), &index_shape).unwrap();
        let src = Tensor::new(src_data.clone(), &index_shape).unwrap();
        let cpu_out = cpu
            .scatter(&input, 1, &index, &src, ScatterReduce::Add)
            .unwrap();

        let model_out = scatter_model(
            &input_data,
            &out_shape,
            &index_data,
            &index_shape,
            &src_data,
            1,
            ScatterReduce::Add,
        )
        .unwrap();
        assert_eq!(model_out, cpu_out.as_slice().unwrap());
        assert_eq!(model_out, vec![16.0, 0.0, 0.0]);
    }

    #[test]
    fn scatter_reduced_index_axis_passes_through_input() {
        // 非 dim 軸で index_shape < out_shape（縮小 index）のケース。
        assert_scatter_matches_cpu(&[4, 4], &[2, 3], 1, ScatterReduce::Overwrite, 20);
        assert_scatter_matches_cpu(&[4, 4], &[2, 3], 1, ScatterReduce::Add, 21);
    }

    /// 相殺列（`[2^48, 2^24, 1, -2^48, -2^24]`。ホスト `f64` 逐次和は
    /// `1`）を同一出力スロットへ集約し、CPU 参照実装（`f64`
    /// アキュムレータ）と bit 完全一致することを確認する
    /// （`.claude/rules/coding-rust.md`「勾配の長軸縮約」節・
    /// `soft_f64.rs` モジュール doc 参照）。
    #[test]
    fn scatter_add_cancelling_sequence_matches_cpu() {
        let cpu = CpuBackendOps::new();
        let out_shape = [1usize];
        let index_shape = [5usize];
        let input_data = vec![0.0f32];
        let index_data = vec![0i32; 5];
        let src_data = vec![
            2f32.powi(48),
            2f32.powi(24),
            1.0,
            -(2f32.powi(48)),
            -(2f32.powi(24)),
        ];

        let input = Tensor::new(input_data.clone(), &out_shape).unwrap();
        let index = Tensor::<i32>::new(index_data.clone(), &index_shape).unwrap();
        let src = Tensor::new(src_data.clone(), &index_shape).unwrap();
        let cpu_out = cpu
            .scatter(&input, 0, &index, &src, ScatterReduce::Add)
            .unwrap();

        let model_out = scatter_model(
            &input_data,
            &out_shape,
            &index_data,
            &index_shape,
            &src_data,
            0,
            ScatterReduce::Add,
        )
        .unwrap();
        assert_eq!(model_out, cpu_out.as_slice().unwrap());
        assert_eq!(model_out, vec![1.0]);
    }

    /// NaN／±inf 入力での `Add` クラス一致（NaN payload はハードウェア
    /// 依存のため `f32_bits_match` で比較）。
    #[test]
    fn scatter_add_nan_inf_matches_cpu_class() {
        let cpu = CpuBackendOps::new();
        let out_shape = [1usize];
        let index_shape = [3usize];
        let input_data = vec![f32::NAN];
        let index_data = vec![0i32; 3];
        let src_data = vec![f32::INFINITY, f32::NEG_INFINITY, 1.0];

        let input = Tensor::new(input_data.clone(), &out_shape).unwrap();
        let index = Tensor::<i32>::new(index_data.clone(), &index_shape).unwrap();
        let src = Tensor::new(src_data.clone(), &index_shape).unwrap();
        let cpu_out = cpu
            .scatter(&input, 0, &index, &src, ScatterReduce::Add)
            .unwrap();

        let model_out = scatter_model(
            &input_data,
            &out_shape,
            &index_data,
            &index_shape,
            &src_data,
            0,
            ScatterReduce::Add,
        )
        .unwrap();
        assert!(crate::soft_f64::f32_bits_match(
            model_out[0],
            cpu_out.as_slice().unwrap()[0]
        ));
    }

    #[test]
    fn validate_index_range_rejects_out_of_range_and_negative() {
        assert!(validate_index_range(&[0, 1, 2], 0, 3).is_ok());
        let err = validate_index_range(&[0, 5, 2], 1, 3).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 1,
                index: 5,
                dim_size: 3
            }
        );
        let err = validate_index_range(&[-1, 1], 0, 3).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 0,
                index: -1,
                dim_size: 3
            }
        );
    }

    #[test]
    fn validate_shapes_fit_u32_rejects_excess_rank() {
        let shape = vec![1usize; GS_MAX_RANK + 1];
        let err = validate_shapes_fit_u32(&[&shape]).unwrap_err();
        assert_eq!(
            err,
            ShapeError::RankMismatch {
                expected: GS_MAX_RANK,
                actual: GS_MAX_RANK + 1
            }
        );
    }

    #[test]
    fn validate_shapes_fit_u32_rejects_dim_exceeding_u32() {
        let shape = [u32::MAX as usize + 1];
        let err = validate_shapes_fit_u32(&[&shape]).unwrap_err();
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    #[test]
    fn validate_shapes_fit_u32_accepts_small_shapes() {
        assert!(validate_shapes_fit_u32(&[&[2, 3], &[4]]).is_ok());
    }
}
