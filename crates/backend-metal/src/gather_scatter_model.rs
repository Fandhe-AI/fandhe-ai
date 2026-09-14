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
///
/// `pub(crate)`: `crate::gather_scatter`（実カーネル起動 API）の
/// `MetalGatherScatter::run_gather_f32`／`run_scatter_f32` も、無検査の
/// `.iter().product()`（大きな shape でオーバーフローし本番経路で panic
/// しうる。`.claude/rules/coding-rust.md`）の代わりに本関数を再利用する
/// （codex-review 指摘。イシュー #1778）。
pub(crate) fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
}

/// `numel`（`index_shape`／`out_shape` の要素数積）がカーネル引数
/// `constant uint&`（32bit）へ収まることを検証する（`crate::
/// gather_scatter::validate_gather_scatter_len` の純関数版・同一契約。
/// `numel` が `u32::MAX` を超える形状は `gs_dispatch_sizes` が実際の
/// 要素数より少ないスレッドグループしかディスパッチせず、確保した出力
/// バッファの一部が未初期化のままホストへ返る。`checked_numel` 自体は
/// `usize` オーバーフローのみを検査し `u32` 収容は検査しないため、
/// 両者は独立の検査軸として両方必要。`crate::elementwise::
/// validate_elementwise_len` と同じ理由。OWASP A03）。
pub(crate) fn validate_launch_len(numel: usize) -> Result<(), ShapeError> {
    if numel > u32::MAX as usize {
        return Err(ShapeError::ElementCountOverflow);
    }
    Ok(())
}

/// `MetalGatherScatter::run_gather_f32`（`crate::gather_scatter`）の
/// 起動前検査をカーネル本体から切り離した純関数版（`objc2` FFI に
/// 触れないため Linux 実行可能。イシュー #1799 レビュー指摘: P0 の
/// スライス長検証・非 dim 軸 shape 整合検査は macOS 実機限定
/// （`#[ignore]`）の `tests/gather_scatter_parity.rs` でしか回帰確認
/// できておらず、Linux で走る CI では実質未検証だった）。
///
/// 検査順序・内容は `run_gather_f32` 本体と同一（`validate_shapes_fit_u32`
/// → [`fandhe_ai_tensor_core::gather_out_shape`]〈rank・非 `dim` 軸整合・
/// `dim` 範囲〉→ `checked_numel` による出力要素数計算〈0 なら早期
/// `Ok(0)`〉→ `validate_launch_len`〈`pub(crate)` のためコードスパン
/// 表記とする〉→ [`validate_index_range`] →
/// `input`／`index` の実スライス長検証）。戻り値は出力要素数
/// （`numel`。0 の場合はカーネル起動不要を呼び出し元へ伝える）。
pub fn validate_gather_launch(
    input: &[f32],
    in_shape: &[usize],
    index: &[i32],
    index_shape: &[usize],
    dim: usize,
) -> Result<usize, ShapeError> {
    validate_shapes_fit_u32(&[in_shape, index_shape])?;
    fandhe_ai_tensor_core::gather_out_shape(in_shape, index_shape, dim)?;

    let numel = checked_numel(index_shape)?;
    if numel == 0 {
        return Ok(0);
    }
    validate_launch_len(numel)?;

    let dim_size = in_shape[dim];
    validate_index_range(index, dim, dim_size)?;

    let in_numel = checked_numel(in_shape)?;
    if input.len() != in_numel {
        return Err(ShapeError::ElementCountMismatch {
            expected: in_numel,
            actual: input.len(),
        });
    }
    if index.len() != numel {
        return Err(ShapeError::ElementCountMismatch {
            expected: numel,
            actual: index.len(),
        });
    }
    Ok(numel)
}

/// [`validate_gather_launch`] の戻り値（scatter は gather と異なり
/// 出力要素数〈`numel_out`〉と `index`／`src` 側要素数〈`idx_numel`〉の
/// 2 値が必要なため、gather のような単一 `usize` 戻り値では表現できない）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScatterLaunch {
    /// 出力（＝`out_shape`）の要素数積。0 の場合は空配列を返す契約。
    pub numel_out: usize,
    /// `index`／`src`（＝`index_shape`）の要素数積。0（`numel_out` は
    /// 非 0）の場合は `input` の完全なパススルーを返す契約
    /// （`scatter_out_shape` の非 `dim` 軸契約に基づく。`crate::
    /// gather_scatter::run_scatter_f32` 冒頭コメント参照）。
    pub idx_numel: usize,
}

/// `MetalGatherScatter::run_scatter_f32`（`crate::gather_scatter`）の
/// 起動前検査をカーネル本体から切り離した純関数版（[`validate_gather_launch`]
/// と同じ理由で Linux 実行可能。イシュー #1799 レビュー指摘）。
///
/// 検査順序・内容は `run_scatter_f32` 本体と同一。`index` の値域検査
/// （[`validate_index_range`]）は `numel_out == 0` の早期 return
/// **より前**に行う（`out_shape[dim]` の取得自体は `scatter_out_shape`
/// が既に `dim < rank` を検査済みのため `numel_out` の値に関わらず
/// 安全。`validate_index_range` 自体は `index` の実要素を単純走査する
/// だけで `row_major_strides` 等のストライド計算を伴わないため、
/// P1〈`numel_out == 0` での `usize` オーバーフロー回避〉の対象外——
/// P1 が守っている「大きな `index_shape` に対する無検査ストライド積」
/// は本関数内で一切計算しない。この順序変更前は
/// `scatter_model(&[], &[0], &[0], &[1], &[1.0], 0, Add)` のように
/// `numel_out == 0`（空出力）だが `index` に範囲外添字が混じる入力を
/// 「成功」として受理していた——CPU 参照実装・`BackendOps` 契約
/// （範囲外添字は常に `IndexOutOfRange`）と不整合だった。Cursor
/// Bugbot・codex-review P2 指摘。イシュー #1799）。`numel_out == 0`
/// の場合は早期 `Ok(ScatterLaunch { numel_out: 0, idx_numel: 0 })`
/// （呼び出し元は空配列を返す）。`idx_numel == 0`（`numel_out` は非 0）
/// の場合は `index`／`src` の長さ検証を行わずに返す（呼び出し元は
/// `input` の完全なパススルーを返す。Cursor Bugbot・codex-review P2
/// 指摘〈別件・イシュー #1799 初回レビュー〉）。
pub fn validate_scatter_launch(
    input: &[f32],
    out_shape: &[usize],
    index: &[i32],
    index_shape: &[usize],
    src: &[f32],
    dim: usize,
) -> Result<ScatterLaunch, ShapeError> {
    validate_shapes_fit_u32(&[out_shape, index_shape])?;
    fandhe_ai_tensor_core::scatter_out_shape(out_shape, index_shape, index_shape, dim)?;

    let dim_size = out_shape[dim];
    validate_index_range(index, dim, dim_size)?;

    let numel_out = checked_numel(out_shape)?;
    if numel_out == 0 {
        return Ok(ScatterLaunch {
            numel_out: 0,
            idx_numel: 0,
        });
    }
    validate_launch_len(numel_out)?;

    if input.len() != numel_out {
        return Err(ShapeError::ElementCountMismatch {
            expected: numel_out,
            actual: input.len(),
        });
    }

    let idx_numel = checked_numel(index_shape)?;
    if idx_numel == 0 {
        return Ok(ScatterLaunch {
            numel_out,
            idx_numel: 0,
        });
    }

    if index.len() != idx_numel {
        return Err(ShapeError::ElementCountMismatch {
            expected: idx_numel,
            actual: index.len(),
        });
    }
    if src.len() != idx_numel {
        return Err(ShapeError::ElementCountMismatch {
            expected: idx_numel,
            actual: src.len(),
        });
    }

    Ok(ScatterLaunch {
        numel_out,
        idx_numel,
    })
}

/// `MetalGatherScatter::run_one_hot_f32`（`crate::gather_scatter`）の
/// 起動前検査をカーネル本体から切り離した純関数版（**非微分演算**。
/// [`validate_gather_launch`]・[`validate_scatter_launch`] と同じ
/// 理由で Linux 実行可能。イシュー #1755）。`gather`／`scatter` と
/// 異なり `one_hot_f32` カーネルは座標展開・ストライド配列（`shapes`
/// バッファ）を一切使わないため、[`validate_shapes_fit_u32`] は呼ばず
/// [`checked_numel`]／[`validate_launch_len`]（出力要素数の `u32`
/// 収容）・[`validate_index_range`]（`index` 値域）のみで検証する。
///
/// 検査順序: [`fandhe_ai_tensor_core::one_hot_out_shape`]（`num_classes
/// == 0` を含む shape 検査。`Err(ShapeError::IndexOutOfRange { dim:
/// index_shape.len(), index: 0, dim_size: 0 })` を返しうる）→
/// `checked_numel`（出力要素数。0 なら早期 `Ok(0)`）→
/// `validate_launch_len`（`u32` 収容。`num_classes` 自体は `row_numel`
/// が 1 以上の下で `numel`（`row_numel` と `num_classes` の積）が
/// `num_classes` 以上になるため、この検査が `num_classes` の `u32`
/// 収容も暗に保証する——`row_numel == 0` の場合は直前の早期 `Ok(0)`
/// で既に return 済み）
/// → [`validate_index_range`]（`dim` ラベルは `gather`／`scatter` と
/// 同じ「範囲外添字が属する軸の位置」の意味で `index_shape.len()` を
/// 渡す。`Var::one_hot`・CPU／CUDA 実装と同じ規約）→ `index` の実
/// スライス長検証。戻り値は出力要素数（`numel`。0 の場合はカーネル
/// 起動不要を呼び出し元へ伝える）。
pub fn validate_one_hot_launch(
    index: &[i32],
    index_shape: &[usize],
    num_classes: usize,
) -> Result<usize, ShapeError> {
    let out_shape = fandhe_ai_tensor_core::one_hot_out_shape(index_shape, num_classes)?;

    let numel = checked_numel(&out_shape)?;
    if numel == 0 {
        return Ok(0);
    }
    validate_launch_len(numel)?;

    validate_index_range(index, index_shape.len(), num_classes)?;

    let row_numel = checked_numel(index_shape)?;
    if index.len() != row_numel {
        return Err(ShapeError::ElementCountMismatch {
            expected: row_numel,
            actual: index.len(),
        });
    }
    Ok(numel)
}

/// `gather_f32` カーネルのホスト側逐語モデル（イシュー #1778）。
///
/// 本関数は `pub`（`crate::gather_scatter_model` はクレート公開
/// モジュール）かつ `#[cfg(test)]` の外にあるため、テスト以外からの
/// 直接呼び出しでも配列アクセスで panic しないよう、[`validate_gather_launch`]
/// （[`fandhe_ai_tensor_core::gather_out_shape`] による shape・rank・
/// `dim` 検査・`u32` 収容・`index` 値域・スライス長一致まで一括で行う。
/// `crate::gather_scatter::MetalGatherScatter::run_gather_f32` と同じ
/// 検査を共有する）を入口で呼ぶ（codex-review 指摘。イシュー #1799。
/// 以前は「呼び出し元が検査済みで渡す契約」としていたが、契約を
/// コメントで示すだけでは `.claude/rules/coding-rust.md`「本番経路で
/// panic 禁止」を満たさないと判断し、実際に検証する形へ変更した）。
pub fn gather_model(
    input: &[f32],
    in_shape: &[usize],
    index: &[i32],
    index_shape: &[usize],
    dim: usize,
) -> Result<Vec<f32>, ShapeError> {
    let numel = validate_gather_launch(input, in_shape, index, index_shape, dim)?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let in_strides = row_major_strides(in_shape);
    let mut out = vec![0.0f32; numel];
    for (gid, out_slot) in out.iter_mut().enumerate() {
        let mut coords = unravel(gid, index_shape);
        let dim_idx = index[gid] as usize;
        coords[dim] = dim_idx;
        let src_off = ravel(&coords, &in_strides);
        *out_slot = input[src_off];
    }
    Ok(out)
}

/// `scatter_overwrite_f32`／`scatter_add_f32` カーネルのホスト側逐語
/// モデル（イシュー #1778）。出力定常方式（`shaders/gather_scatter.metal`
/// 冒頭コメント「CPU 参照実装との順序等価性」）を Rust で再現する。
///
/// 本関数は `pub` かつ `#[cfg(test)]` の外にあるため、[`gather_model`]
/// と同じ理由で入口に [`validate_scatter_launch`]（`crate::
/// gather_scatter::MetalGatherScatter::run_scatter_f32` と同じ検査を
/// 共有する）を呼び、配列アクセスで panic しないようにする
/// （codex-review 指摘。イシュー #1799）。`numel_out == 0` は空配列を
/// 返す。`idx_numel == 0`（`index`／`src` が空）の場合は
/// `validate_scatter_launch` が `index`／`src` の値・長さを検証しない
/// ため以降の計算ループへそのまま入るが、下の集約ループ自体が
/// `index_shape[axis] == 0` を非 `in_range`（＝`input` パススルー）
/// として、または `index_shape[dim] == 0` を空ループとして自然に扱う
/// ため追加の早期 return は不要（`ScatterLaunch::idx_numel` はここでは
/// 使わない）。
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
    let ScatterLaunch { numel_out, .. } =
        validate_scatter_launch(input, out_shape, index, index_shape, src, dim)?;
    if numel_out == 0 {
        return Ok(Vec::new());
    }
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

/// `one_hot_f32` カーネルのホスト側逐語モデル（**非微分演算**。イシュー
/// #1755）。`shaders/gather_scatter.metal::one_hot_f32` と同じ
/// `row = gid / num_classes`・`c = gid % num_classes` の単純な整数
/// 除算・剰余のみ（座標展開・ストライドは使わない）。
///
/// 本関数は `pub` かつ `#[cfg(test)]` の外にあるため、[`gather_model`]
/// と同じ理由で入口に [`validate_one_hot_launch`] を呼び、配列
/// アクセスで panic しないようにする。`numel == 0` は空配列を返す。
pub fn one_hot_model(
    index: &[i32],
    index_shape: &[usize],
    num_classes: usize,
) -> Result<Vec<f32>, ShapeError> {
    let numel = validate_one_hot_launch(index, index_shape, num_classes)?;
    if numel == 0 {
        return Ok(Vec::new());
    }
    let mut out = vec![0.0f32; numel];
    for (gid, out_slot) in out.iter_mut().enumerate() {
        let row = gid / num_classes;
        let c = gid % num_classes;
        let index_val = index[row];
        *out_slot =
            if index_val >= 0 && (index_val as usize) < num_classes && index_val as usize == c {
                1.0
            } else {
                0.0
            };
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

        let model_out = gather_model(&input_data, in_shape, &index_data, index_shape, dim).unwrap();
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

    // 以下、イシュー #1799 レビュー指摘（P0／P1／P2）の Linux 実行可能な
    // 回帰テスト。`validate_gather_launch`／`validate_scatter_launch`・
    // `validate_launch_len` は `objc2` FFI に触れない純関数のため、
    // 従来 macOS 実機限定（`#[ignore]`）の `tests/gather_scatter_parity.rs`
    // でしか検証できていなかった検査ロジックをここで直接検証する。

    #[test]
    fn validate_launch_len_accepts_len_at_u32_max() {
        assert!(validate_launch_len(u32::MAX as usize).is_ok());
    }

    #[test]
    fn validate_launch_len_rejects_len_exceeding_u32_max() {
        let err = validate_launch_len(u32::MAX as usize + 1)
            .expect_err("u32::MAX を超える numel は拒否されるべき");
        assert_eq!(err, ShapeError::ElementCountOverflow);
    }

    /// codex-review P0 指摘の再現ケース: `input=[1.0]`・`in_shape=[1]`・
    /// `index=[-1]`・`index_shape=[1]`・`dim=0`。範囲外（負）の `index`
    /// が `validate_index_range` で拒否されることを確認する
    /// （`MetalGatherScatter::run_gather_f32` を `ops.rs` 経由せず直接
    /// 呼んだ場合の防御。PR #1799 レビュースレッド）。
    #[test]
    fn validate_gather_launch_rejects_negative_index() {
        let err = validate_gather_launch(&[1.0], &[1], &[-1], &[1], 0).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 0,
                index: -1,
                dim_size: 1
            }
        );
    }

    /// codex-review P0 指摘: `shape` 引数とスライスの実長が食い違う
    /// 直接呼び出し（`input` が `in_shape` の要素数積より短い）を拒否する。
    #[test]
    fn validate_gather_launch_rejects_input_len_mismatch() {
        let err = validate_gather_launch(&[1.0], &[2], &[0, 0], &[2], 0).unwrap_err();
        assert_eq!(
            err,
            ShapeError::ElementCountMismatch {
                expected: 2,
                actual: 1
            }
        );
    }

    /// codex-review P0 指摘（advisor 追補分。#4e982e74 相当）:
    /// rank・`dim` 自体は正しいが非 `dim` 軸の次元が食い違う入力
    /// （`in_shape=[2,3]`・`index_shape=[5,3]`・`dim=1`）を
    /// `gather_out_shape` 経由で拒否する（手書きの rank／dim 検査のみ
    /// では見逃していたケース）。
    #[test]
    fn validate_gather_launch_rejects_non_dim_axis_mismatch() {
        let input = vec![0.0f32; 6];
        let index = vec![0i32; 15];
        let err = validate_gather_launch(&input, &[2, 3], &index, &[5, 3], 1).unwrap_err();
        assert_eq!(
            err,
            ShapeError::ShapeMismatch {
                lhs: vec![2, 3],
                rhs: vec![5, 3],
            }
        );
    }

    /// codex-review P1 指摘の再現ケース: `shape=[0, u32::MAX, u32::MAX, 2]`・
    /// `dim=1`。空出力（`numel == 0`）の早期 return が `row_major_strides`
    /// 呼び出しより前に効き、`2 * u32::MAX * u32::MAX` の `usize`
    /// オーバーフロー panic（debug ビルドの overflow-checks 有効時）を
    /// 起こさず `Ok(0)` を返すことを確認する。
    #[test]
    fn validate_gather_launch_empty_output_skips_stride_overflow() {
        let big = u32::MAX as usize;
        let index_shape = [0usize, big, big, 2];
        let numel = validate_gather_launch(&[], &[0, big, big, 2], &[], &index_shape, 1).unwrap();
        assert_eq!(numel, 0);
    }

    /// codex-review P2 指摘の再現ケース: `input` の shape=[3]・
    /// `index`／`src` の shape=[0]・`dim=0`。`index_shape` の要素数積が
    /// 0（`out_shape` は非空）の scatter は `input` の完全な
    /// パススルーとして扱われるべきで、`idx_numel == 0` を呼び出し元へ
    /// 伝える。
    #[test]
    fn validate_scatter_launch_reports_empty_index_as_passthrough() {
        let input = vec![1.0f32, 2.0, 3.0];
        let launch =
            validate_scatter_launch(&input, &[3], &[], &[0], &[], 0).expect("有効な入力のはず");
        assert_eq!(
            launch,
            ScatterLaunch {
                numel_out: 3,
                idx_numel: 0,
            }
        );
    }

    /// codex-review P1 指摘の scatter 版: `out_shape=[0, u32::MAX,
    /// u32::MAX, 2]`・`dim=1`。空出力の早期 return が
    /// `row_major_strides(index_shape)` より前に効き overflow panic
    /// しないことを確認する。
    #[test]
    fn validate_scatter_launch_empty_output_skips_stride_overflow() {
        let big = u32::MAX as usize;
        let out_shape = [0usize, big, big, 2];
        let launch = validate_scatter_launch(&[], &out_shape, &[], &out_shape, &[], 1).unwrap();
        assert_eq!(
            launch,
            ScatterLaunch {
                numel_out: 0,
                idx_numel: 0,
            }
        );
    }

    /// `validate_scatter_launch` も `validate_gather_launch` と同様、
    /// rank・`dim` は正しいが非 `dim` 軸の次元が `out_shape` を超える
    /// （`scatter_out_shape` の `index_shape[axis] <= out_shape[axis]`
    /// 契約に反する）入力を拒否する。
    #[test]
    fn validate_scatter_launch_rejects_non_dim_axis_exceeding_out_shape() {
        let input = vec![0.0f32; 12];
        let index = vec![0i32; 15];
        let src = vec![0.0f32; 15];
        let err = validate_scatter_launch(&input, &[4, 3], &index, &[5, 3], &src, 1).unwrap_err();
        assert_eq!(
            err,
            ShapeError::ShapeMismatch {
                lhs: vec![4, 3],
                rhs: vec![5, 3],
            }
        );
    }

    // 以下、イシュー #1799 の追加 codex-review 指摘（新規 P1: 公開ホスト
    // モデル gather_model／scatter_model 自体が入力検証をせず配列へ
    // アクセスして panic しうる）の再現ケースをそのまま単体テスト化する。
    // 検証ロジック自体は validate_gather_launch／validate_scatter_launch
    // へ既に切り出し済みだが、`gather_model`／`scatter_model` が実際に
    // それらを入口で呼び Err を返すことをここで直接確認する（呼び出し元
    // 〈本モジュールの他テスト〉は正しい入力のみを渡すため、この経路は
    // 従来のテストでは通らない）。

    /// codex-review 指摘の再現ケースそのもの: `gather_model(&[1.0],
    /// &[1], &[-1], &[1], 0)` は、検証前は `input[src_off]` で範囲外
    /// 添字アクセスにより panic していた。
    #[test]
    fn gather_model_rejects_negative_index_instead_of_panicking() {
        let err = gather_model(&[1.0], &[1], &[-1], &[1], 0).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 0,
                index: -1,
                dim_size: 1
            }
        );
    }

    /// codex-review 指摘の再現ケースそのもの: `scatter_model(&[], &[1],
    /// &[0], &[1], &[1.0], 0, ScatterReduce::Add)` は、検証前は
    /// `input[gid]`（空スライスへの添字アクセス）で panic していた。
    #[test]
    fn scatter_model_rejects_input_len_mismatch_instead_of_panicking() {
        let err = scatter_model(&[], &[1], &[0], &[1], &[1.0], 0, ScatterReduce::Add).unwrap_err();
        assert_eq!(
            err,
            ShapeError::ElementCountMismatch {
                expected: 1,
                actual: 0
            }
        );
    }

    /// codex-review 新規指摘（P2・イシュー #1799 2 回目レビュー）の
    /// 再現ケースそのもの: `scatter_model(&[], &[0], &[0], &[1], &[1.0],
    /// 0, ScatterReduce::Add)` は `out_shape=[0]`（空出力）だが
    /// `index=[0]` が `dim_size=out_shape[0]=0` に対し範囲外
    /// （`[0, 0)` は空区間なので `0` すら不正）という入力。是正前は
    /// `numel_out == 0` の早期 return が `validate_index_range` より前に
    /// あったため、この範囲外添字を検証せず空配列成功として受理して
    /// いた。是正後は `IndexOutOfRange` を返す（CPU 参照実装・
    /// `BackendOps` 契約と一致）。
    #[test]
    fn scatter_model_validates_index_range_even_for_empty_output() {
        let err = scatter_model(&[], &[0], &[0], &[1], &[1.0], 0, ScatterReduce::Add).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 0,
                index: 0,
                dim_size: 0
            }
        );
    }

    /// [`validate_scatter_launch`] 単体でも同じ挙動になることを確認する
    /// （`run_scatter_f32` の直接呼び出しも同じ検査を共有するため）。
    #[test]
    fn validate_scatter_launch_validates_index_range_even_for_empty_output() {
        let err = validate_scatter_launch(&[], &[0], &[0], &[1], &[1.0], 0).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 0,
                index: 0,
                dim_size: 0
            }
        );
    }

    // --- one_hot（非微分演算。イシュー #1755） ---

    fn assert_one_hot_matches_cpu(index_shape: &[usize], num_classes: usize, seed: u64) {
        let cpu = CpuBackendOps::new();
        let idx_numel: usize = index_shape.iter().product();
        let index_data = i32_index(seed, idx_numel, num_classes);

        let index = Tensor::<i32>::new(index_data.clone(), index_shape).unwrap();
        let cpu_out = cpu.one_hot(&index, num_classes).unwrap();

        let model_out = one_hot_model(&index_data, index_shape, num_classes).unwrap();
        assert_eq!(
            model_out.len(),
            cpu_out.as_slice().unwrap().len(),
            "one_hot_model と CPU 出力の要素数が不一致"
        );
        for (a, b) in model_out.iter().zip(cpu_out.as_slice().unwrap().iter()) {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "one_hot_model が CPU 参照実装と bit 不一致"
            );
        }
    }

    #[test]
    fn one_hot_model_matches_cpu_1d() {
        assert_one_hot_matches_cpu(&[4], 3, 4001);
    }

    #[test]
    fn one_hot_model_matches_cpu_2d() {
        assert_one_hot_matches_cpu(&[2, 3], 5, 4002);
    }

    #[test]
    fn one_hot_model_matches_cpu_3d() {
        assert_one_hot_matches_cpu(&[2, 2, 2], 4, 4003);
    }

    #[test]
    fn one_hot_model_empty_index_returns_empty_output() {
        let out = one_hot_model(&[], &[0], 3).unwrap();
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn one_hot_model_rejects_out_of_range_index_instead_of_panicking() {
        let err = one_hot_model(&[3], &[1], 3).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 1,
                index: 3,
                dim_size: 3,
            }
        );
    }

    #[test]
    fn one_hot_model_rejects_negative_index_instead_of_panicking() {
        let err = one_hot_model(&[-1], &[1], 3).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 1,
                index: -1,
                dim_size: 3,
            }
        );
    }

    /// `num_classes == 0` は `one_hot_out_shape` が構造的に
    /// `Err(IndexOutOfRange { dim: index_shape.len(), index: 0,
    /// dim_size: 0 })` を返すため、`validate_one_hot_launch` もこの
    /// エラーをそのまま透過する（`ops_shape.rs::one_hot_out_shape` doc
    /// と同じ意味論）。
    #[test]
    fn validate_one_hot_launch_rejects_num_classes_zero() {
        let err = validate_one_hot_launch(&[0], &[1], 0).unwrap_err();
        assert_eq!(
            err,
            ShapeError::IndexOutOfRange {
                dim: 1,
                index: 0,
                dim_size: 0,
            }
        );
    }
}
