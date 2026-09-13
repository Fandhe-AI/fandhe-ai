//! gather／scatter（`torch.gather`／`torch.scatter`／`torch.scatter_add`
//! 相当）の CUDA C カーネルソース（NVRTC 実行時コンパイル用の静的文字列。
//! イシュー #1777・親イシュー #1638）。
//!
//! `gather_scatter.rs`（呼び出し元）は本モジュールの 3 定数を `nvrtc::
//! compile_ptx` に渡し `CudaFunction` を得る。`kernels_reduce.rs`・
//! `kernels_scalar_op.rs` と同じ理由でソースを `nvcc` 事前コンパイルせず
//! 文字列のまま埋め込む（ビルド時に nvcc/CUDA ヘッダを一切要求しない。
//! 「CUDA toolkit 非搭載環境でも `cargo build --workspace` が成立する」
//! 契約を維持する。`.claude/rules/deps-policy.md`）。
//!
//! CPU 参照実装（`backend-cpu::gather_scatter`）と同じ意味論・決定的
//! 集約契約に従う（`fandhe_ai_tensor_core::ScatterReduce` doc が正）。
//! rank 可変の shape を固定長ローカル配列に頼らず扱うため、各カーネルは
//! 出力位置ごとの座標を「末尾軸から `%`／`/=` で剥がして即座に使う」
//! 一過性のスカラー変数（`rem`／`index_base`／`dim_coord`）だけで処理し、
//! 座標配列を一切保持しない（rank に上限を設けない設計）。
//!
//! # 3 カーネルの共通構造
//!
//! いずれも「出力位置 1 個 = 1 スレッド」で、`idx`（出力 flat 添字）を
//! `out_shape`（行優先の各軸サイズ配列。`rank` 要素）で末尾軸から
//! 剥がしながら必要な値を組み立てる。座標展開の添字演算はすべて
//! `long long`（`kernels_reduce.rs::REDUCE_SUM_LASTAXIS_F32` の
//! イシュー #1675 教訓と同じ理由。`i32::MAX` 近傍でのオーバーフロー
//! 回避）。
//!
//! - [`GATHER_F32`]（`gather_f32`）: `index[idx]`（`out_shape` ==
//!   `index.shape()` かつ両者とも contiguous なので直接読める。追加の
//!   index 用ストライド配列は不要）を読み、`out_shape` を末尾軸から
//!   剥がして得た各軸座標（`dim` 軸のみ `index[idx]` の値へ差し替え）を
//!   `in_strides`（`input.shape()` の行優先ストライド）へ畳み込んで
//!   `input` 側の読み出し位置を得る。範囲外添字（`ops.rs` が起動前に
//!   ホスト側で検査済みのため通常到達しないが、REQ-8 の縦深防御として
//!   カーネル内でも検査する）は `0.0f` を書く。
//! - [`SCATTER_OVERWRITE_F32`]／[`SCATTER_ADD_F32`]（`scatter_overwrite_f32`／
//!   `scatter_add_f32`）: 出力 shape は `input.shape()` と恒等のため、
//!   `out[idx]`／`input[idx]` は同一 flat 位置（ravel／unravel 不要。
//!   `out_shape` は座標展開の「各軸サイズ」としてのみ使い、ストライド
//!   計算は行わない）。`out_shape` を末尾軸から剥がしながら、`dim` 軸
//!   以外の各軸座標 `c` が `index_shape[a]` 未満かを検査し（超過なら
//!   その出力位置には scatter の寄与がなく `input[idx]` がそのまま残る
//!   —— `scatter_out_shape` が非 `dim` 軸に `idx_s <= in_s` のみを課す
//!   ことに対応）、範囲内の軸については `index_strides`（`index.shape()`
//!   の行優先ストライド）へ畳み込んだ `index_base` を蓄積する。`dim` 軸
//!   の座標は `dim_coord` として保持し、`j ∈ [0, index_shape[dim])` を
//!   昇順に走査して `index[index_base + j*index_strides[dim]] ==
//!   dim_coord` が真の要素を「上書き（最後に一致した値が残る）」または
//!   「`double` アキュムレータへ逐次加算」する（`ScatterReduce` doc の
//!   決定的集約契約。CPU 側の row-major 走査順と一致する根拠は
//!   `gather_scatter.rs` モジュール doc を参照）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! 全カーネルで `idx < numel` を維持する（グリッドがちょうど割り切れない
//! 場合の末尾ブロック対策）。ベクトル化ロード等の最適化は本イシューでは
//! 適用しない（`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（3 カーネル共通。
/// `kernels_reduce::REDUCE_BLOCK_DIM`・`kernels_elementwise::EW_BLOCK_DIM`
/// と同じ値・同じ理由）。
pub const GATHER_SCATTER_BLOCK_DIM: u32 = 256;

/// `torch.gather` 相当（本ファイル冒頭コメント参照）。
pub const GATHER_F32: &str = r#"
extern "C" __global__ void gather_f32(
    const float* __restrict__ in,
    const int* __restrict__ index,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ in_strides,
    int rank,
    int dim,
    int numel,
    int in_dim_size)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int index_val = index[idx];
        if (index_val >= 0 && index_val < in_dim_size) {
            long long rem = idx;
            long long in_flat = 0;
            for (int a = rank - 1; a >= 0; a--) {
                long long axis_size = (long long)out_shape[a];
                long long c = rem % axis_size;
                rem /= axis_size;
                long long coord = (a == dim) ? (long long)index_val : c;
                in_flat += coord * (long long)in_strides[a];
            }
            out[idx] = in[in_flat];
        } else {
            // 範囲外添字（呼び出し元 `ops.rs` が起動前にホスト側で検査
            // 済みのため通常到達しない。REQ-8 の縦深防御として境界外
            // 読み出しを回避し安全側の値を書く）。
            out[idx] = 0.0f;
        }
    }
}
"#;

/// `torch.scatter`（上書き）相当（本ファイル冒頭コメント参照）。
pub const SCATTER_OVERWRITE_F32: &str = r#"
extern "C" __global__ void scatter_overwrite_f32(
    const float* __restrict__ input,
    const int* __restrict__ index,
    const float* __restrict__ src,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ index_shape,
    const int* __restrict__ index_strides,
    int rank,
    int dim,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long index_base = 0;
        long long dim_coord = 0;
        bool in_range = true;
        for (int a = rank - 1; a >= 0; a--) {
            long long axis_size = (long long)out_shape[a];
            long long c = rem % axis_size;
            rem /= axis_size;
            if (a == dim) {
                dim_coord = c;
            } else if (c >= (long long)index_shape[a]) {
                in_range = false;
                break;
            } else {
                index_base += c * (long long)index_strides[a];
            }
        }

        float result = input[idx];
        if (in_range) {
            long long dim_len = (long long)index_shape[dim];
            long long dim_stride = (long long)index_strides[dim];
            for (long long j = 0; j < dim_len; j++) {
                long long p = index_base + j * dim_stride;
                if ((long long)index[p] == dim_coord) {
                    result = src[p];
                }
            }
        }
        out[idx] = result;
    }
}
"#;

/// `torch.scatter_add` 相当（`double` アキュムレータ・本ファイル冒頭
/// コメント「決定的集約契約」参照。`.claude/rules/coding-rust.md`
/// 「勾配の長軸縮約」節と同じ精度規律: 要素を `double` へ昇格してから
/// 逐次加算し、最後に 1 回だけ `(float)` へ downcast する）。
pub const SCATTER_ADD_F32: &str = r#"
extern "C" __global__ void scatter_add_f32(
    const float* __restrict__ input,
    const int* __restrict__ index,
    const float* __restrict__ src,
    float* __restrict__ out,
    const int* __restrict__ out_shape,
    const int* __restrict__ index_shape,
    const int* __restrict__ index_strides,
    int rank,
    int dim,
    int numel)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        long long rem = idx;
        long long index_base = 0;
        long long dim_coord = 0;
        bool in_range = true;
        for (int a = rank - 1; a >= 0; a--) {
            long long axis_size = (long long)out_shape[a];
            long long c = rem % axis_size;
            rem /= axis_size;
            if (a == dim) {
                dim_coord = c;
            } else if (c >= (long long)index_shape[a]) {
                in_range = false;
                break;
            } else {
                index_base += c * (long long)index_strides[a];
            }
        }

        double acc = (double)input[idx];
        if (in_range) {
            long long dim_len = (long long)index_shape[dim];
            long long dim_stride = (long long)index_strides[dim];
            for (long long j = 0; j < dim_len; j++) {
                long long p = index_base + j * dim_stride;
                if ((long long)index[p] == dim_coord) {
                    acc += (double)src[p];
                }
            }
        }
        out[idx] = (float)acc;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// 3 ソースいずれも REQ-8 の境界検査（`idx < numel`）を維持している
    /// ことの機械検証（`kernels_reduce.rs`／`kernels_scalar_op.rs` と
    /// 同型の文字列テスト）。
    #[test]
    fn all_kernels_include_bounds_check() {
        for src in [GATHER_F32, SCATTER_OVERWRITE_F32, SCATTER_ADD_F32] {
            assert!(
                src.contains("if (idx < numel)"),
                "kernel source must retain the idx < numel bounds check: {src}"
            );
        }
    }

    /// `scatter_add_f32` のみ `double` アキュムレータ・1 回限りの `(float)`
    /// downcast を持つ（`.claude/rules/coding-rust.md` の精度契約）。
    #[test]
    fn scatter_add_uses_double_accumulator_and_single_downcast() {
        assert!(SCATTER_ADD_F32.contains("double acc = (double)input[idx];"));
        assert!(SCATTER_ADD_F32.contains("acc += (double)src[p];"));
        assert!(SCATTER_ADD_F32.contains("out[idx] = (float)acc;"));
        // `scatter_overwrite_f32`／`gather_f32` は丸めを伴わない単純代入
        // のため `double` を使わない。
        assert!(!SCATTER_OVERWRITE_F32.contains("double"));
        assert!(!GATHER_F32.contains("double"));
    }

    /// gather は範囲外 index 値を（REQ-8 の縦深防御として）安全側の
    /// `0.0f` へ書く分岐を持つ。
    #[test]
    fn gather_has_out_of_range_fallback() {
        assert!(GATHER_F32.contains("out[idx] = 0.0f;"));
    }

    /// scatter 系 2 カーネルは非 `dim` 軸の範囲外座標（`scatter_out_shape`
    /// が許容する `idx_s <= in_s` の隙間）を `in_range = false` で検出し、
    /// `input[idx]`（恒等コピー）へフォールバックする。
    #[test]
    fn scatter_kernels_handle_out_of_range_non_dim_axis() {
        for src in [SCATTER_OVERWRITE_F32, SCATTER_ADD_F32] {
            assert!(src.contains("in_range = false"));
        }
    }

    /// 添字演算は `long long`（`i32::MAX` 近傍でのオーバーフロー回避。
    /// イシュー #1675 の教訓を踏襲）。
    #[test]
    fn all_kernels_use_long_long_for_flat_index_arithmetic() {
        for src in [GATHER_F32, SCATTER_OVERWRITE_F32, SCATTER_ADD_F32] {
            assert!(src.contains("long long idx"));
        }
    }
}
