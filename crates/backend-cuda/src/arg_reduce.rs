//! `argmax`／`argmin`（`torch.argmax`／`torch.argmin` 相当）の CUDA
//! 起動 API（NVRTC コンパイル・保持・実行。イシュー #1948・親イシュー
//! #1947）。
//!
//! `reduce.rs::CudaReduce`（`sum`／`max`／`min`）とは別のハンドル
//! （[`CudaArgReduce`]）として分離する: `CudaReduce::new` は既に 12
//! カーネルを一括 NVRTC コンパイルしており（`init_cost_diag` 系テストの
//! 対象）、ここへ argmax/argmin 6 カーネンルを相乗りさせると `sum`／
//! `max`／`min` の初回呼び出しコストを後退させうるため（`scan.rs` が
//! `reduce.rs` と別ハンドルに分離しているのと同じ理由）。
//! `ops.rs::CudaBackendOps::argmax`／`argmin` から
//! `context_cache::cached_arg_reduce` 経由で呼ばれる。
//!
//! `validate_i32_bound`／`validate_axis_layout`（`i32::MAX` 収容・
//! `outer*axis_len*inner == a.len()` 検査）は `reduce.rs` の実装を
//! `pub(crate)` として再利用する（複製しない）。
//!
//! # 走査契約・NaN 判定・全軸縮約のチャンク分割方式
//!
//! `kernels_arg_reduce.rs` モジュール doc が正（本ファイルはその起動
//! パラメータ〈[`arg_all_plan`]〉のみを担う）。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_arg_reduce;
use crate::kernels_reduce::REDUCE_BLOCK_DIM;
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::reduce::{validate_axis_layout, validate_i32_bound};

/// 全軸縮約 1 段目の 1 チャンクが担当する最小要素数。値そのものは
/// 性能チューニング対象外（実測なしで決めた既定値）で、正当性
/// （タイの決定的な最小添字選択）はチャンクが `[0, numel)` を
/// 連続・昇順・重複なく被覆することにのみ依存する（`arg_all_plan` の
/// テスト参照）。
const ARG_MIN_CHUNK: usize = 4096;

/// 全軸縮約 1 段目のチャンク数上限（`REDUCE_BLOCK_DIM * 16`。finalize
/// 段が単一スレッドで `0..num_chunks` を逐次走査するため、上限を
/// 大きくしすぎると finalize 段が支配的になる。境界検査
/// （`arg_reduce.rs::run_argmax_all_f32` 等）が超過を防ぐわけではなく
/// 単に起動ブロック数の目安として使う）。
const ARG_MAX_CHUNKS: usize = (REDUCE_BLOCK_DIM as usize) * 16;

/// 全軸縮約 1 段目の起動パラメータ（`chunk_len`, `num_chunks`）を
/// 決定する純関数（`kernels_arg_reduce.rs` モジュール doc「全軸縮約の
/// 2 段構成」参照）。`numel == 0` は `(0, 0)`（呼び出し元は `numel == 0`
/// を [`CudaError::EmptyReduction`] として起動前に拒否する契約のため
/// 実質到達しない防御的分岐）。
///
/// 契約: `numel > 0` のとき `num_chunks >= 1`・`chunk_len >= 1`・
/// `chunk_len * num_chunks >= numel`（最終チャンクは `numel` で
/// クランプされるためこれで `[0, numel)` を過不足なく被覆する）・
/// `num_chunks <= ARG_MAX_CHUNKS`。
pub(crate) fn arg_all_plan(numel: usize) -> (usize, usize) {
    if numel == 0 {
        return (0, 0);
    }
    let num_chunks = numel.div_ceil(ARG_MIN_CHUNK).clamp(1, ARG_MAX_CHUNKS);
    let chunk_len = numel.div_ceil(num_chunks).max(1);
    // 上記 `chunk_len` は `num_chunks` からの逆算のため、余りにより
    // 実際に必要なチャンク数が `num_chunks` を下回りうる（空チャンクを
    // 生成しないための再計算。`num_chunks` を再度縮めるだけで
    // `ARG_MAX_CHUNKS` 上限には抵触しない）。
    let num_chunks = numel.div_ceil(chunk_len);
    (chunk_len, num_chunks)
}

/// argmax／argmin 6 カーネン（単一軸・全軸 2 段 × 2 演算）のコンパイル
/// 済みハンドルを保持する。
pub struct CudaArgReduce {
    stream: Arc<CudaStream>,
    /// `reduce.rs::CudaReduce::ordinal` と同じ役割。
    ordinal: usize,
    argmax_axis_f32: CudaFunction,
    argmax_all_partial_f32: CudaFunction,
    argmax_all_finalize_f32: CudaFunction,
    argmin_axis_f32: CudaFunction,
    argmin_all_partial_f32: CudaFunction,
    argmin_all_finalize_f32: CudaFunction,
}

impl CudaArgReduce {
    /// `device` 上で argmax/argmin 6 カーネンを NVRTC コンパイルし
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

        let argmax_axis_f32 = compile_and_load!(
            kernels_arg_reduce::REDUCE_ARGMAX_AXIS_F32,
            "reduce_argmax_axis_f32"
        );
        let argmax_all_partial_f32 = compile_and_load!(
            kernels_arg_reduce::REDUCE_ARGMAX_ALL_PARTIAL_F32,
            "reduce_argmax_all_partial_f32"
        );
        let argmax_all_finalize_f32 = compile_and_load!(
            kernels_arg_reduce::REDUCE_ARGMAX_ALL_FINALIZE_F32,
            "reduce_argmax_all_finalize_f32"
        );
        let argmin_axis_f32 = compile_and_load!(
            kernels_arg_reduce::REDUCE_ARGMIN_AXIS_F32,
            "reduce_argmin_axis_f32"
        );
        let argmin_all_partial_f32 = compile_and_load!(
            kernels_arg_reduce::REDUCE_ARGMIN_ALL_PARTIAL_F32,
            "reduce_argmin_all_partial_f32"
        );
        let argmin_all_finalize_f32 = compile_and_load!(
            kernels_arg_reduce::REDUCE_ARGMIN_ALL_FINALIZE_F32,
            "reduce_argmin_all_finalize_f32"
        );

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            argmax_axis_f32,
            argmax_all_partial_f32,
            argmax_all_finalize_f32,
            argmin_axis_f32,
            argmin_all_partial_f32,
            argmin_all_finalize_f32,
        })
    }

    /// `reduce.rs::CudaReduce::with_driver_call` と同じ設計（CUDA Graph
    /// capture 排他への参加）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// 全軸 `argmax`（イシュー #1948）。`numel == 0` は
    /// [`CudaError::EmptyReduction`]（`op: "argmax"`。`backend-cpu::
    /// reduction::argmax` と同一の意味論・呼び出し元 `ops.rs` が同一
    /// 文言へ写像）を返す。
    pub fn run_argmax_all_f32(&self, a: &[f32]) -> Result<i32, CudaError> {
        self.run_all_f32(
            a,
            "argmax",
            &self.argmax_all_partial_f32,
            &self.argmax_all_finalize_f32,
        )
    }

    /// 全軸 `argmin`（イシュー #1948）。[`Self::run_argmax_all_f32`] と
    /// 対称（`op: "argmin"`）。
    pub fn run_argmin_all_f32(&self, a: &[f32]) -> Result<i32, CudaError> {
        self.run_all_f32(
            a,
            "argmin",
            &self.argmin_all_partial_f32,
            &self.argmin_all_finalize_f32,
        )
    }

    /// [`Self::run_argmax_all_f32`]／[`Self::run_argmin_all_f32`] 共通の
    /// 2 段実行本体（`partial`／`finalize` カーネンのみが argmax/argmin
    /// で異なる。カーネン自体の比較演算子〈`>`／`<`〉が唯一の差分で
    /// あり、起動手順・バッファ確保・境界検査は完全共通のため関数を
    /// 分離せず引数化する）。
    fn run_all_f32(
        &self,
        a: &[f32],
        op: &'static str,
        partial_fn: &CudaFunction,
        finalize_fn: &CudaFunction,
    ) -> Result<i32, CudaError> {
        let numel = a.len();
        validate_i32_bound(numel, "numel")?;
        if numel == 0 {
            return Err(CudaError::EmptyReduction { op });
        }

        let (chunk_len, num_chunks) = arg_all_plan(numel);
        let chunk_len_i = validate_i32_bound(chunk_len, "chunk_len")?;
        let num_chunks_i = validate_i32_bound(num_chunks, "num_chunks")?;
        let numel_i = numel as i32;

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            // `pval`／`pidx` は `f32`／`PoolDtype` 未実装の `i32` を
            // 混在させるため両方とも `stream.alloc_zeros` で per-call
            // 直接確保する（`reduce.rs::run_sum_all_f32` の `f64` partial
            // と同じくプール非経由。`sort.rs::run_sort_f32` の `index_dev`
            // と同じ確保方式）。カーネルは `t < num_chunks` の全スレッドが
            // `pval[t]`／`pidx[t]` を必ず 1 回書くため、ゼロ初期化の内容は
            // 上書きされる。
            let mut pval_dev = self.stream.alloc_zeros::<f32>(num_chunks)?;
            let mut pidx_dev = self.stream.alloc_zeros::<i32>(num_chunks)?;

            let partial_cfg = LaunchConfig {
                grid_dim: ((num_chunks as u32).div_ceil(REDUCE_BLOCK_DIM), 1, 1),
                block_dim: (REDUCE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `a_dev` は `numel` 要素の H2D 済みバッファ、
            // `pval_dev`／`pidx_dev` は `num_chunks` 要素確保済みで
            // カーネルは `t < num_chunks`（REQ-8）を維持したまま各スレッド
            // が担当チャンク `[t*chunk_len, min((t+1)*chunk_len, numel))`
            // のみを読み `pval[t]`／`pidx[t]` を 1 回だけ書く
            // （`kernels_arg_reduce.rs` 参照）。
            unsafe {
                self.stream
                    .launch_builder(partial_fn)
                    .arg(&a_dev)
                    .arg(&mut pval_dev)
                    .arg(&mut pidx_dev)
                    .arg(&numel_i)
                    .arg(&chunk_len_i)
                    .arg(&num_chunks_i)
                    .launch(partial_cfg)?;
            }

            let mut out_dev = self.stream.alloc_zeros::<i32>(1)?;
            let finalize_cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `pval_dev`／`pidx_dev` は上記で `num_chunks` 要素
            // すべて書き込み済み、`out_dev` は単一ブロック・単一スレッド
            // （`blockIdx.x == 0 && threadIdx.x == 0`）が `out[0]` を必ず
            // 1 回書くため `alloc_zeros(1)` で確保した内容は上書きされる。
            unsafe {
                self.stream
                    .launch_builder(finalize_fn)
                    .arg(&pval_dev)
                    .arg(&pidx_dev)
                    .arg(&mut out_dev)
                    .arg(&num_chunks_i)
                    .launch(finalize_cfg)?;
            }

            let host: Vec<i32> = readback(&self.stream, &out_dev)?;
            Ok(host.first().copied().unwrap_or(0))
        })
    }

    /// 単一軸 `argmax`（イシュー #1948）。`axis_len == 0 && total_out >
    /// 0` は [`CudaError::EmptyReduction`]（`op: "argmax"`）を返す
    /// （`backend-cpu::reduction::argmax` と同一の意味論。`reduce.rs::
    /// CudaReduce::run_max_axis_f32` と同型のルーティング方針）。
    pub fn run_argmax_axis_f32(
        &self,
        a: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<i32>, CudaError> {
        self.run_axis_f32(a, outer, axis_len, inner, "argmax", &self.argmax_axis_f32)
    }

    /// 単一軸 `argmin`（イシュー #1948）。[`Self::run_argmax_axis_f32`]
    /// と対称（`op: "argmin"`）。
    pub fn run_argmin_axis_f32(
        &self,
        a: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
    ) -> Result<Vec<i32>, CudaError> {
        self.run_axis_f32(a, outer, axis_len, inner, "argmin", &self.argmin_axis_f32)
    }

    /// [`Self::run_argmax_axis_f32`]／[`Self::run_argmin_axis_f32`] 共通の
    /// 実行本体（カーネン自体の比較演算子のみが差分）。
    fn run_axis_f32(
        &self,
        a: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
        op: &'static str,
        func: &CudaFunction,
    ) -> Result<Vec<i32>, CudaError> {
        let total_out = validate_axis_layout(a.len(), outer, axis_len, inner)?;
        if axis_len == 0 {
            if total_out > 0 {
                return Err(CudaError::EmptyReduction { op });
            }
            return Ok(Vec::new());
        }
        if total_out == 0 {
            return Ok(Vec::new());
        }

        self.with_driver_call(|| {
            let a_dev = self.stream.clone_htod(a)?;
            let mut out_dev = self.stream.alloc_zeros::<i32>(total_out)?;

            let (outer_i, axis_len_i, inner_i) = (outer as i32, axis_len as i32, inner as i32);
            let cfg = LaunchConfig {
                grid_dim: ((total_out as u32).div_ceil(REDUCE_BLOCK_DIM), 1, 1),
                block_dim: (REDUCE_BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            };
            // SAFETY: `a_dev` は `outer*axis_len*inner` 要素の H2D 済み
            // バッファ、`out_dev` は `total_out` 要素確保済みでカーネルは
            // `idx < total`（REQ-8）を維持したまま各出力要素を 1 回だけ
            // 書く（`kernels_arg_reduce.rs` 参照）。
            unsafe {
                self.stream
                    .launch_builder(func)
                    .arg(&a_dev)
                    .arg(&mut out_dev)
                    .arg(&outer_i)
                    .arg(&axis_len_i)
                    .arg(&inner_i)
                    .launch(cfg)?;
            }

            readback(&self.stream, &out_dev)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::Tensor;

    // ------------------------------------------------------------------
    // `arg_all_plan` の網羅性検証
    // ------------------------------------------------------------------

    /// チャンク群が `[0, numel)` を重複・欠落なく被覆し、`num_chunks`
    /// が `ARG_MAX_CHUNKS` を超えないことを検証する（純関数のため GPU
    /// 不要）。
    fn assert_plan_covers_range(numel: usize) {
        let (chunk_len, num_chunks) = arg_all_plan(numel);
        assert!(num_chunks <= ARG_MAX_CHUNKS, "numel={numel}");
        if numel == 0 {
            assert_eq!((chunk_len, num_chunks), (0, 0));
            return;
        }
        assert!(chunk_len >= 1, "numel={numel}");
        assert!(num_chunks >= 1, "numel={numel}");

        let mut covered = 0usize;
        for t in 0..num_chunks {
            let start = t * chunk_len;
            let end = ((t + 1) * chunk_len).min(numel);
            assert_eq!(
                start, covered,
                "チャンク {t} が連続していない: numel={numel}"
            );
            assert!(end > start, "空チャンクが生成された: t={t} numel={numel}");
            covered = end;
        }
        assert_eq!(covered, numel, "被覆が numel に届いていない: numel={numel}");
    }

    #[test]
    fn arg_all_plan_covers_small_and_boundary_sizes() {
        for numel in [
            0usize,
            1,
            2,
            3,
            ARG_MIN_CHUNK - 1,
            ARG_MIN_CHUNK,
            ARG_MIN_CHUNK + 1,
        ] {
            assert_plan_covers_range(numel);
        }
    }

    #[test]
    fn arg_all_plan_covers_large_sizes_without_exceeding_max_chunks() {
        for numel in [
            ARG_MIN_CHUNK * ARG_MAX_CHUNKS - 1,
            ARG_MIN_CHUNK * ARG_MAX_CHUNKS,
            ARG_MIN_CHUNK * ARG_MAX_CHUNKS + 1,
            ARG_MIN_CHUNK * ARG_MAX_CHUNKS * 4 + 7,
        ] {
            assert_plan_covers_range(numel);
        }
    }

    #[test]
    fn arg_all_plan_is_deterministic_pure_function() {
        // 同一入力に対し常に同一出力を返す（GPU 起動計画の run-to-run
        // 決定性の前提）。
        for numel in [1usize, 100, 4096, 4097, 1_000_003] {
            assert_eq!(arg_all_plan(numel), arg_all_plan(numel));
        }
    }

    // ------------------------------------------------------------------
    // ホストモデル（カーネル逐語再現）vs CPU 参照実装の突合
    //
    // GPU を起動せず、`kernels_arg_reduce.rs` のカーネル本体が実装する
    // 走査規則（決定的タイ分割・NaN 無視・全軸縮約のチャンク分割＋
    // finalize 結合）を Rust でそのまま再現し（`arg_reduce_model_*`）、
    // `fandhe_ai_backend_cpu::reduction::{argmax, argmin}` をオラクルに
    // 突合する。GPU 実機実測（`#[ignore]`）は `tests/reduce_parity.rs`
    // が担う。
    // ------------------------------------------------------------------

    fn is_nan_bit_pattern(v: f32) -> bool {
        // カーネルの `(__float_as_uint(v) & 0x7fffffffu) > 0x7f800000u`
        // を Rust で逐語再現する。
        (v.to_bits() & 0x7fff_ffff) > 0x7f80_0000
    }

    /// `reduce_argmax_axis_f32`／`reduce_argmin_axis_f32` の逐語モデル。
    fn model_axis(
        a: &[f32],
        outer: usize,
        axis_len: usize,
        inner: usize,
        is_max: bool,
    ) -> Vec<i32> {
        let mut out = vec![0i32; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut has_best = false;
                let mut best_val = 0.0f32;
                let mut best_idx = 0i32;
                for a_idx in 0..axis_len {
                    let src = (o * axis_len + a_idx) * inner + i;
                    let v = a[src];
                    if is_nan_bit_pattern(v) {
                        continue;
                    }
                    let better = if is_max { v > best_val } else { v < best_val };
                    if !has_best || better {
                        has_best = true;
                        best_val = v;
                        best_idx = a_idx as i32;
                    }
                }
                out[o * inner + i] = best_idx;
            }
        }
        out
    }

    /// `reduce_arg{max,min}_all_partial_f32` → `_finalize_f32` の 2 段
    /// 全軸縮約の逐語モデル。
    fn model_all(a: &[f32], is_max: bool) -> i32 {
        let numel = a.len();
        if numel == 0 {
            return 0;
        }
        let (chunk_len, num_chunks) = arg_all_plan(numel);
        let mut pval = vec![0.0f32; num_chunks];
        let mut pidx = vec![-1i32; num_chunks];
        for t in 0..num_chunks {
            let start = t * chunk_len;
            let end = ((t + 1) * chunk_len).min(numel);
            let mut has_best = false;
            let mut best_val = 0.0f32;
            let mut best_idx = -1i32;
            for (i, &v) in a.iter().enumerate().take(end).skip(start) {
                if is_nan_bit_pattern(v) {
                    continue;
                }
                let better = if is_max { v > best_val } else { v < best_val };
                if !has_best || better {
                    has_best = true;
                    best_val = v;
                    best_idx = i as i32;
                }
            }
            pval[t] = best_val;
            pidx[t] = best_idx;
        }

        let mut has_best = false;
        let mut best_val = 0.0f32;
        let mut best_idx = 0i32;
        for c in 0..num_chunks {
            let idx = pidx[c];
            if idx < 0 {
                continue;
            }
            let v = pval[c];
            let better = if is_max { v > best_val } else { v < best_val };
            if !has_best || better {
                has_best = true;
                best_val = v;
                best_idx = idx;
            }
        }
        best_idx
    }

    fn xorshift_f32(state: &mut u64, lo: f32, hi: f32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        let u = (*state >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * u as f32
    }

    #[test]
    fn model_all_matches_cpu_reference_across_shapes_with_ties_and_nan() {
        let mut state = 0x1234_5678_9abc_def1u64;
        for numel in [1usize, 2, 7, 4096, 4097, 10_000] {
            // タイを多発させるため値域を狭くする。
            let mut data: Vec<f32> = (0..numel)
                .map(|_| xorshift_f32(&mut state, -3.0, 3.0).round())
                .collect();
            // 確率的に NaN を混入する。
            for v in data.iter_mut() {
                if xorshift_f32(&mut state, 0.0, 1.0) < 0.05 {
                    *v = f32::NAN;
                }
            }
            let t = Tensor::new(data.clone(), &[numel]).unwrap();

            let expect_max = fandhe_ai_backend_cpu::reduction::argmax(&t, None);
            let expect_min = fandhe_ai_backend_cpu::reduction::argmin(&t, None);

            match expect_max {
                Ok(exp) => {
                    let exp_idx = exp.as_slice().unwrap()[0];
                    assert_eq!(model_all(&data, true), exp_idx, "argmax numel={numel}");
                }
                Err(_) => unreachable!("numel>0 のため EmptyReduction は発生しない"),
            }
            match expect_min {
                Ok(exp) => {
                    let exp_idx = exp.as_slice().unwrap()[0];
                    assert_eq!(model_all(&data, false), exp_idx, "argmin numel={numel}");
                }
                Err(_) => unreachable!("numel>0 のため EmptyReduction は発生しない"),
            }
        }
    }

    #[test]
    fn model_all_matches_cpu_reference_for_all_nan_input() {
        let data = vec![f32::NAN; 10];
        assert_eq!(model_all(&data, true), 0);
        assert_eq!(model_all(&data, false), 0);
    }

    #[test]
    fn model_all_matches_cpu_reference_for_all_infinite_input() {
        let data_pos = vec![f32::INFINITY; 5000];
        let data_neg = vec![f32::NEG_INFINITY; 5000];
        assert_eq!(model_all(&data_pos, true), 0);
        assert_eq!(model_all(&data_neg, false), 0);
    }

    #[test]
    fn model_all_matches_cpu_reference_with_tie_spanning_chunk_boundary() {
        // 最大値がチャンク境界を跨いで複数出現する場合、最初の添字が
        // 選ばれることを確認する（`arg_all_plan` の `chunk_len` を
        // 意図的に跨ぐ位置へタイを配置）。
        let numel = ARG_MIN_CHUNK * 3;
        let (chunk_len, _) = arg_all_plan(numel);
        let mut data = vec![0.0f32; numel];
        // 2 番目のチャンク先頭とその手前・3 番目のチャンク先頭に同値の
        // 最大値を置く。
        let first_tie = chunk_len - 1;
        let second_tie = chunk_len;
        let third_tie = chunk_len * 2;
        data[first_tie] = 100.0;
        data[second_tie] = 100.0;
        data[third_tie] = 100.0;
        assert_eq!(model_all(&data, true), first_tie as i32);

        let mut data_min = vec![0.0f32; numel];
        data_min[first_tie] = -100.0;
        data_min[second_tie] = -100.0;
        data_min[third_tie] = -100.0;
        assert_eq!(model_all(&data_min, false), first_tie as i32);
    }

    #[test]
    fn model_axis_matches_cpu_reference_middle_axis_with_nan_and_ties() {
        let mut state = 0xdead_beef_1234_5678u64;
        let (outer, axis_len, inner) = (3usize, 17usize, 5usize);
        let numel = outer * axis_len * inner;
        let mut data: Vec<f32> = (0..numel)
            .map(|_| xorshift_f32(&mut state, -2.0, 2.0).round())
            .collect();
        for v in data.iter_mut() {
            if xorshift_f32(&mut state, 0.0, 1.0) < 0.1 {
                *v = f32::NAN;
            }
        }
        let t = Tensor::new(data.clone(), &[outer, axis_len, inner]).unwrap();

        let exp_max = fandhe_ai_backend_cpu::reduction::argmax(&t, Some(1)).unwrap();
        let exp_min = fandhe_ai_backend_cpu::reduction::argmin(&t, Some(1)).unwrap();

        assert_eq!(
            model_axis(&data, outer, axis_len, inner, true),
            exp_max.as_slice().unwrap()
        );
        assert_eq!(
            model_axis(&data, outer, axis_len, inner, false),
            exp_min.as_slice().unwrap()
        );
    }

    #[test]
    fn model_axis_matches_cpu_reference_last_axis_all_nan_row() {
        // 1 行だけ全要素 NaN（添字 0 が期待値）にした行方向縮約
        // （`inner == 1`）。
        let (outer, axis_len, inner) = (2usize, 6usize, 1usize);
        let mut data = vec![0.0f32; outer * axis_len * inner];
        for (i, v) in data.iter_mut().enumerate().take(axis_len) {
            *v = (i as f32) - 3.0;
        }
        for v in data.iter_mut().skip(axis_len).take(axis_len) {
            *v = f32::NAN;
        }
        let t = Tensor::new(data.clone(), &[outer, axis_len]).unwrap();
        let exp_max = fandhe_ai_backend_cpu::reduction::argmax(&t, Some(1)).unwrap();
        assert_eq!(
            model_axis(&data, outer, axis_len, inner, true),
            exp_max.as_slice().unwrap()
        );
        // 2 行目（全 NaN）の期待添字は 0。
        assert_eq!(exp_max.as_slice().unwrap()[1], 0);
    }
}
