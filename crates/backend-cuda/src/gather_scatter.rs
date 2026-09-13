//! gather／scatter（`torch.gather`／`torch.scatter`／`torch.scatter_add`
//! 相当）の起動 API（NVRTC コンパイル・保持・実行。イシュー #1777・
//! 親イシュー #1638）。
//!
//! `reduce.rs::CudaReduce`・`elementwise.rs::CudaElementwise` と同じ
//! 構成方針を踏襲する: [`CudaGatherScatter::new`] が `CudaDevice` から
//! 3 カーネル（`kernels_gather_scatter.rs`）を NVRTC コンパイルして
//! 保持し、以降は `run_gather_f32`／`run_scatter_f32` へホスト側スライス
//! を渡すだけで GPU 実行できる。`ops.rs::CudaBackendOps::gather`／
//! `scatter` から `BackendOps` の実装として呼ばれる。
//!
//! **shape 検証の責務分担**（`backend-cpu::gather_scatter` モジュール
//! doc・`.claude/rules/security.md` A08 と同じ二重検査方針）: 呼び出し元
//! `ops.rs` が [`fandhe_ai_tensor_core::gather_out_shape`]／
//! [`fandhe_ai_tensor_core::scatter_out_shape`] で `input`／`index`／`src`
//! の rank・軸整合を再検査してから本モジュールへ委譲する契約のため、
//! 本モジュール自身は `rank`／`dim` の整合を `debug_assert!` でのみ
//! 確認する（release ビルドでは無効化される内部契約検証。CPU 側と同じ
//! 「本モジュールは呼び出し元の shape 正しさを信頼する」方針）。一方で、
//! `index`／`input`／`src` の **要素数**（`checked_numel` で導出した
//! 期待長との一致）は release ビルドでも独立に検証する——CPU の
//! `Tensor::get` は境界チェック付き安全アクセスのため shape 不整合が
//! あっても panic／OOB を起こさないが、本モジュールは生ポインタで GPU
//! へ転送するため、期待長と異なるホストスライスを `clone_htod` すると
//! デバイス側配列が短く、カーネルが `idx < numel` の範囲内で読んでも
//! 実際には確保長を超える読み出し（未定義動作）になりうる。この長さ
//! 検証は shape 検証の代替ではなく、GPU 転送前に必須の独立した安全策
//! である。
//!
//! `index` の値が `[0, dim_size)` 範囲内であることは呼び出し元
//! （`ops.rs::gather_dispatch`／`scatter_dispatch`。ホスト側で index を
//! 全走査して検証する）が保証する契約とする（`fandhe_ai_autodiff::
//! var::Var::gather`／`scatter`／`scatter_add` の forward 時点検証と
//! 合わせた二重検査。`kernels_gather_scatter.rs` のカーネル側にも REQ-8
//! の縦深防御として範囲外添字のフォールバックがある）。
//!
//! **決定的集約順序・精度契約**は
//! [`fandhe_ai_tensor_core::ScatterReduce`] doc・
//! `kernels_gather_scatter.rs` モジュール doc を正とする。CPU 参照実装
//! との bit 同一性の根拠（row-major 走査順の一致）も同モジュール doc に
//! 記載する。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use fandhe_ai_tensor_core::ScatterReduce;

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_gather_scatter::{self, GATHER_SCATTER_BLOCK_DIM};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;

/// `value` が `i32::MAX` に収まることを検証する（カーネル引数 `int` は
/// C の 32bit 符号付き整数のため。`reduce.rs::validate_i32_bound` と
/// 同じ理由の複製——モジュール間で `pub(crate)` 関数を共有すると
/// エラー型〈`InvalidReduceShape` 対 `InvalidGatherScatterShape`〉が
/// 混在してしまうため、`kernels_*`／起動 API のモジュールごとに専用の
/// 検証関数を持つ既存方針〈`elementwise.rs::validate_elementwise_len`
/// 等〉を踏襲する）。
fn validate_i32_bound(value: usize, name: &str) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::InvalidGatherScatterShape {
        detail: format!(
            "gather/scatter dimension must fit in i32 (kernel argument type): \
                          {name}={value}"
        ),
    })
}

/// shape の要素数積を `checked_mul` の畳み込みで検査する
/// （`backend-cpu::gather_scatter::checked_numel` と同型だが `pub(crate)`
/// でクレートを跨いで共有できないため複製する。空 shape チェックより前に
/// 無検査の `.iter().product()` を計算すると、末尾軸が 0 でも先行する軸
/// 同士の積が `usize` の範囲でオーバーフローし、overflow-checks 有効時
/// に panic しうるため——CPU 側と同じ理由）。
fn checked_numel(shape: &[usize]) -> Result<usize, CudaError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or_else(|| CudaError::InvalidGatherScatterShape {
            detail: "checked_numel: element count overflow".to_string(),
        })
}

/// `shape` の各軸サイズを `i32` 配列へ変換する（カーネル引数
/// `out_shape`／`index_shape` の H2D 転送用）。
pub(crate) fn shape_to_i32(shape: &[usize]) -> Result<Vec<i32>, CudaError> {
    shape
        .iter()
        .map(|&d| validate_i32_bound(d, "shape_dim"))
        .collect()
}

/// `shape` の行優先（C-order）ストライドを `i32` 配列として計算する
/// （`backend-cpu::gather_scatter::row_major_strides` と同じ定義を
/// `i32`・オーバーフロー検査付きで複製する）。
pub(crate) fn row_major_strides_i32(shape: &[usize]) -> Result<Vec<i32>, CudaError> {
    let rank = shape.len();
    let mut strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].checked_mul(shape[i + 1]).ok_or_else(|| {
            CudaError::InvalidGatherScatterShape {
                detail: "row_major_strides_i32: stride overflow".to_string(),
            }
        })?;
    }
    strides
        .iter()
        .map(|&s| validate_i32_bound(s, "stride"))
        .collect()
}

/// gather／scatter 3 カーネルのコンパイル済みハンドルを保持する。
pub struct CudaGatherScatter {
    stream: Arc<CudaStream>,
    /// `reduce.rs::CudaReduce::ordinal` と同じ役割（`Self::
    /// with_driver_call` が `context_cache::with_driver_call` を呼ぶ際の
    /// キー）。
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    gather_f32: CudaFunction,
    scatter_overwrite_f32: CudaFunction,
    scatter_add_f32: CudaFunction,
}

impl CudaGatherScatter {
    /// `device` 上で gather／scatter 3 カーネルを NVRTC コンパイルし
    /// 保持するハンドルを構築する（`reduce.rs::CudaReduce::new` と同一
    /// 手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();

        macro_rules! compile_and_load {
            ($src:expr, $name:literal) => {{
                let ptx = compile_ptx($src, arch)?;
                device.context().load_module(ptx)?.load_function($name)?
            }};
        }

        let gather_f32 = compile_and_load!(kernels_gather_scatter::GATHER_F32, "gather_f32");
        let scatter_overwrite_f32 = compile_and_load!(
            kernels_gather_scatter::SCATTER_OVERWRITE_F32,
            "scatter_overwrite_f32"
        );
        let scatter_add_f32 =
            compile_and_load!(kernels_gather_scatter::SCATTER_ADD_F32, "scatter_add_f32");

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            gather_f32,
            scatter_overwrite_f32,
            scatter_add_f32,
        })
    }

    /// `CudaGatherScatter` の driver 呼び出しを CUDA Graph capture 排他へ
    /// 参加させる共通ヘルパー（`reduce.rs::CudaReduce::with_driver_call`
    /// と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// `torch.gather` 相当（`kernels_gather_scatter.rs` モジュール doc
    /// 参照）。`out_shape` は `index` の shape（＝呼び出し元の
    /// `gather_out_shape` の戻り値）と一致する契約。
    pub fn run_gather_f32(
        &self,
        input: &[f32],
        index: &[i32],
        in_shape: &[usize],
        out_shape: &[usize],
        dim: usize,
    ) -> Result<Vec<f32>, CudaError> {
        let rank = in_shape.len();
        debug_assert_eq!(
            out_shape.len(),
            rank,
            "run_gather_f32: caller must pre-validate rank match via gather_out_shape"
        );
        debug_assert!(
            dim < rank,
            "run_gather_f32: caller must pre-validate dim via gather_out_shape"
        );

        let numel_out = checked_numel(out_shape)?;
        if index.len() != numel_out {
            return Err(CudaError::InvalidGatherScatterShape {
                detail: format!(
                    "gather: index.len()={} does not match out numel={numel_out}",
                    index.len()
                ),
            });
        }
        let numel_in = checked_numel(in_shape)?;
        if input.len() != numel_in {
            return Err(CudaError::InvalidGatherScatterShape {
                detail: format!(
                    "gather: input.len()={} does not match in numel={numel_in}",
                    input.len()
                ),
            });
        }
        if numel_out == 0 {
            return Ok(Vec::new());
        }
        validate_i32_bound(numel_out, "numel_out")?;
        let in_dim_size = in_shape[dim];

        let out_shape_i32 = shape_to_i32(out_shape)?;
        let in_strides_i32 = row_major_strides_i32(in_shape)?;
        let rank_i = validate_i32_bound(rank, "rank")?;
        let dim_i = validate_i32_bound(dim, "dim")?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;
        let in_dim_size_i = validate_i32_bound(in_dim_size, "in_dim_size")?;

        self.with_driver_call(|| {
            let in_dev = self.stream.clone_htod(input)?;
            let index_dev = self.stream.clone_htod(index)?;
            let out_shape_dev = self.stream.clone_htod(&out_shape_i32)?;
            let in_strides_dev = self.stream.clone_htod(&in_strides_i32)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(GATHER_SCATTER_BLOCK_DIM), 1, 1),
                block_dim: (GATHER_SCATTER_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `in_dev`／`index_dev` はそれぞれ `numel_in`／
            // `numel_out` 要素の H2D 済みバッファ（直上の長さ検証で
            // ホストスライスと一致することを確認済み）。`out_dev` は
            // `numel_out` 要素確保済みでカーネルは `idx < numel`
            // （REQ-8）を維持したまま各出力要素を 1 回だけ書く。
            // `out_shape_dev`／`in_strides_dev` は `rank` 要素の配列で
            // カーネル内ループも `rank` 回のみ走査するため範囲外読み出し
            // はない（`kernels_gather_scatter.rs::GATHER_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(&self.gather_f32)
                    .arg(&in_dev)
                    .arg(&index_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&out_shape_dev)
                    .arg(&in_strides_dev)
                    .arg(&rank_i)
                    .arg(&dim_i)
                    .arg(&numel_i)
                    .arg(&in_dim_size_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev.as_view())
        })
    }

    /// `torch.scatter`／`torch.scatter_add` 相当（`reduce` で分岐。
    /// `kernels_gather_scatter.rs` モジュール doc 参照）。`in_shape` は
    /// 出力 shape と恒等（scatter は shape を変えない）。
    #[allow(clippy::too_many_arguments)] // scatter の意味論（input／index／src／shape 2 種／dim／reduce）を 1 関数に集約する設計上の要求（`nvrtc.rs`・`kernels_mma_tf32.rs` と同方針）
    pub fn run_scatter_f32(
        &self,
        input: &[f32],
        index: &[i32],
        src: &[f32],
        in_shape: &[usize],
        index_shape: &[usize],
        dim: usize,
        reduce: ScatterReduce,
    ) -> Result<Vec<f32>, CudaError> {
        let rank = in_shape.len();
        debug_assert_eq!(
            index_shape.len(),
            rank,
            "run_scatter_f32: caller must pre-validate rank match via scatter_out_shape"
        );
        debug_assert!(
            dim < rank,
            "run_scatter_f32: caller must pre-validate dim via scatter_out_shape"
        );

        // `index_shape` の要素数は `in_shape` の内容（0 要素軸の有無）に
        // 依らず必ず検査する（`backend-cpu::gather_scatter::scatter` が
        // `index_numel` を `out_shape.contains(&0)` 判定より前に計算する
        // のと同じ順序。末尾軸が 0 で先行軸の積がオーバーフローする
        // 病的 shape でも `Err` を返す——CPU 側の既存テスト
        // `scatter_index_shape_leading_large_dims_trailing_zero_does_not_overflow`
        // と同じ想定挙動）。
        let numel_index = checked_numel(index_shape)?;
        if index.len() != numel_index || src.len() != numel_index {
            return Err(CudaError::InvalidGatherScatterShape {
                detail: format!(
                    "scatter: index.len()={}／src.len()={} does not match index numel={numel_index}",
                    index.len(),
                    src.len()
                ),
            });
        }

        if in_shape.contains(&0) {
            // 空出力早期リターン（`backend-cpu::gather_scatter::scatter`
            // モジュール doc の「空テンソルの早期リターン」節と同じ理由:
            // `checked_numel(in_shape)` がストライド計算前にオーバー
            // フローしうる病的 shape〈例: `[usize::MAX, 2, 0]`〉を
            // 避けるため、0 要素軸の有無を先に判定してから積を計算する）。
            return Ok(Vec::new());
        }
        let numel_out = checked_numel(in_shape)?;
        if input.len() != numel_out {
            return Err(CudaError::InvalidGatherScatterShape {
                detail: format!(
                    "scatter: input.len()={} does not match out numel={numel_out}",
                    input.len()
                ),
            });
        }
        validate_i32_bound(numel_out, "numel_out")?;

        if numel_index == 0 {
            // `index`／`src` が空（`scatter_out_shape` は非 `dim` 軸に
            // `idx_s <= in_s` のみを課すため、`dim` 以外の軸が 0 の
            // `index_shape` を許容する）: scatter の寄与がなく `input`
            // をそのまま返す（カーネル起動を回避）。
            return Ok(input.to_vec());
        }

        let out_shape_i32 = shape_to_i32(in_shape)?;
        let index_shape_i32 = shape_to_i32(index_shape)?;
        let index_strides_i32 = row_major_strides_i32(index_shape)?;
        let rank_i = validate_i32_bound(rank, "rank")?;
        let dim_i = validate_i32_bound(dim, "dim")?;
        let numel_i = validate_i32_bound(numel_out, "numel_out")?;

        // `ScatterReduce` は `#[non_exhaustive]`。未知 variant は
        // `Overwrite` へ安全側フォールバックする（`backend-cpu::
        // gather_scatter::scatter` と同じ割り切り方針。`debug_assert!`
        // で契約違反〈到達しないはず〉を明示する）。
        let func = match reduce {
            ScatterReduce::Add => &self.scatter_add_f32,
            reduce => {
                debug_assert!(
                    matches!(reduce, ScatterReduce::Overwrite),
                    "run_scatter_f32: unknown ScatterReduce variant fell back to Overwrite \
                     (contract violation)"
                );
                &self.scatter_overwrite_f32
            }
        };

        self.with_driver_call(|| {
            let input_dev = self.stream.clone_htod(input)?;
            let index_dev = self.stream.clone_htod(index)?;
            let src_dev = self.stream.clone_htod(src)?;
            let out_shape_dev = self.stream.clone_htod(&out_shape_i32)?;
            let index_shape_dev = self.stream.clone_htod(&index_shape_i32)?;
            let index_strides_dev = self.stream.clone_htod(&index_strides_i32)?;
            let mut out_dev = self.allocator.alloc_uninit_f32(numel_out)?;

            let cfg = LaunchConfig {
                grid_dim: ((numel_out as u32).div_ceil(GATHER_SCATTER_BLOCK_DIM), 1, 1),
                block_dim: (GATHER_SCATTER_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `input_dev` は `numel_out` 要素、`index_dev`／
            // `src_dev` は `numel_index` 要素（直上の長さ検証で確認済み）
            // の H2D 済みバッファ。`out_dev` は `numel_out` 要素確保済み
            // でカーネルは `idx < numel`（REQ-8）を維持したまま各出力
            // 要素を 1 回だけ書く。`index_shape`／`index_strides` の
            // カーネル内アクセスは `dim`／非 `dim` 軸いずれも `rank`
            // 要素の配列範囲内（`out_shape` を末尾軸から `rank` 回だけ
            // 剥がすループでのみ参照する。`kernels_gather_scatter.rs::
            // SCATTER_OVERWRITE_F32`／`SCATTER_ADD_F32` 参照）。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&input_dev)
                    .arg(&index_dev)
                    .arg(&src_dev)
                    .arg(&mut out_dev.as_view_mut())
                    .arg(&out_shape_dev)
                    .arg(&index_shape_dev)
                    .arg(&index_strides_dev)
                    .arg(&rank_i)
                    .arg(&dim_i)
                    .arg(&numel_i)
                    .launch(cfg)?;
            }
            readback(&self.stream, &out_dev.as_view())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_strides_i32_decomposes_middle_axis() {
        // shape [2, 3, 4] → strides [12, 4, 1]（`backend-cpu::
        // gather_scatter::row_major_strides` と同じ定義）。
        let strides = row_major_strides_i32(&[2, 3, 4]).unwrap();
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn row_major_strides_i32_handles_rank_one() {
        let strides = row_major_strides_i32(&[7]).unwrap();
        assert_eq!(strides, vec![1]);
    }

    #[test]
    fn shape_to_i32_converts_all_axes() {
        let out = shape_to_i32(&[2, 3, 4]).unwrap();
        assert_eq!(out, vec![2, 3, 4]);
    }

    #[test]
    fn checked_numel_accepts_matching_product() {
        assert_eq!(checked_numel(&[2, 3, 4]).unwrap(), 24);
    }

    #[test]
    fn checked_numel_rejects_overflow() {
        let err = checked_numel(&[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, CudaError::InvalidGatherScatterShape { .. }));
    }

    #[test]
    fn validate_i32_bound_rejects_exceeding_i32_max() {
        let err = validate_i32_bound(i32::MAX as usize + 1, "numel").unwrap_err();
        assert!(matches!(err, CudaError::InvalidGatherScatterShape { .. }));
    }

    #[test]
    fn validate_i32_bound_accepts_i32_max() {
        assert!(validate_i32_bound(i32::MAX as usize, "numel").is_ok());
    }

    /// `row_major_strides_i32` が `i32::MAX` を超えるストライドを
    /// オーバーフロー検査で拒否する（GEMM／reduce 系と同じ
    /// `checked_mul` 方針）。
    #[test]
    fn row_major_strides_i32_rejects_overflowing_stride() {
        let err = row_major_strides_i32(&[2, usize::MAX]).unwrap_err();
        assert!(matches!(err, CudaError::InvalidGatherScatterShape { .. }));
    }
}
