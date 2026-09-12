//! 汎用 reduction（`sum`／`max`。全軸・単一軸）の CUDA C カーネルソース
//! （NVRTC 実行時コンパイル用の静的文字列。イシュー #1584・親イシュー
//! #1571）。
//!
//! `reduce.rs`（呼び出し元）は本モジュールの 8 定数を `nvrtc::
//! compile_ptx` に渡し `CudaFunction` を得る。`kernels_mse.rs` と同じ
//! 理由でソースを `nvcc` 事前コンパイルせず文字列のまま埋め込む（ビルド
//! 時に nvcc/CUDA ヘッダを一切要求しない。「CUDA toolkit 非搭載環境でも
//! `cargo build --workspace` が成立する」契約を維持する。`.claude/rules/
//! deps-policy.md`）。
//!
//! # 全軸縮約の 2 段構成（`kernels_mse.rs` と同型）
//!
//! `reduce_*_all_partial_f32` → `reduce_*_all_finalize_f32` の 2 段
//! reduction。ブロック数・`partial` バッファ長の決定契約（呼び出し元が
//! `min(ceil_div(numel, REDUCE_BLOCK_DIM), REDUCE_MAX_BLOCKS)` を用いる）
//! は `kernels_mse.rs` 冒頭コメントと同一。
//!
//! # 単一軸縮約（`reduce_*_axis_f32`／`reduce_*_lastaxis_f32`）
//!
//! `reduce.rs::reduce_axis_layout` が入力 shape・縮約軸から
//! `(outer, axis_len, inner)` を導出する（`backend-cpu::reduction::
//! axis_reduce` と同じ分解）。出力要素数は `outer * inner`。
//!
//! - `reduce_*_axis_f32`（`inner != 1` の汎用版）: 1 スレッドが 1 出力
//!   要素を担当し、縮約軸を `0..axis_len` の昇順で逐次累積する
//!   （`backend-cpu::reduction::axis_reduce` と同じ決定的順序）。
//! - `reduce_*_lastaxis_f32`（`inner == 1`。最終軸縮約の coalesced 版）:
//!   1 warp（32 lane）が 1 出力行を担当し、`cols`（`axis_len`）を
//!   stride 32 で分割して縮約したのち warp 内 butterfly で結合する
//!   （`kernels_rmsnorm.rs` の 1 CTA = 1 warp 方針の再利用）。`rows`
//!   （`outer`）が起動ブロック数を超える場合は persistent row loop で
//!   ブロックを使い回す。
//!
//! いずれも決定的な累積順序（`blockIdx.x` 昇順・`axis` 昇順・`lane`
//! stride 昇順）であり、bit 決定的（run-to-run で同一入力なら同一結果）。
//! `atomicAdd`／`atomicMax` は使わない（`kernels_mse.rs` と同じ非決定性
//! 回避の理由。`.claude/rules/coding-rust.md` の再現性要件）。
//!
//! # sum の `double` アキュムレータ契約（`.claude/rules/coding-rust.md`）
//!
//! sum 系カーネルは全て `double acc` で累積し、**最後に 1 回だけ**
//! `(float)` へ downcast する（`reduce_sum_all_finalize_f32`／
//! `reduce_sum_axis_f32`／`reduce_sum_lastaxis_f32` の書き出し箇所のみ）。
//! `reduce_sum_all_partial_f32` の `partial` バッファ自体も `double`
//! （ホスト側 `reduce.rs` が `f64` デバイスバッファとして確保し、
//! `reduce_sum_all_finalize_f32` がデバイス側で読み切る。ホストへの D2H
//! は最終 `f32` 出力のみ）。
//!
//! # max は厳密選択（丸めなし）
//!
//! max 系カーネルは `float acc`・`fmaxf` で累積する（丸めを伴わない
//! 厳密な値選択のため `f64` アキュムレータ契約の対象外。`fmaxf` は
//! `f32::max`（`backend-cpu::reduction::max_slice`）と同じ NaN 非伝播
//! （NaN を無視して他方を返す）意味論）。`Op::Max` の VJP（`grad.rs::
//! max_vjp`）は forward 記録値と入力の `==` 一致で argmax 位置を決める
//! ため、丸めを伴う値を返すと勾配が誤って 0 になる。単位元は
//! `-INFINITY`（空縮約は呼び出し元がカーネル起動前に拒否する契約。
//! `reduce.rs` 参照）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! 全カーネルで手動境界チェック（`idx < numel`／`idx < num_partials`／
//! `idx < total`／`row < rows`）を維持する。ベクトル化ロード等の最適化は
//! 本イシューでは適用しない（`.claude/rules/coding-rust.md`）。

/// 1 スレッドブロックあたりのスレッド数（全軸縮約・汎用軸縮約カーネル
/// 共通。`kernels_mse::MSE_BLOCK_DIM` と同じ値・同じ理由）。
pub const REDUCE_BLOCK_DIM: u32 = 256;

/// 全軸縮約 2 段目（`reduce_*_all_finalize_f32`）が単一ブロックで処理
/// しきれる `partial` の最大長（＝ 1 段目の起動ブロック数の上限）。
/// `kernels_mse::MSE_MAX_BLOCKS` と同じ値・同じ理由。
pub const REDUCE_MAX_BLOCKS: u32 = 1024;

/// `reduce_*_lastaxis_f32` の 1 ブロックあたりスレッド数（1 warp）。
pub const REDUCE_LASTAXIS_BLOCK_DIM: u32 = 32;

/// sum 全軸縮約 1 段目: 各ブロックが担当区間の `Σ in[i]` を `double`
/// アキュムレータで計算し `partial[blockIdx.x]` へ書く（本ファイル冒頭
/// コメント「sum の `double` アキュムレータ契約」参照）。
pub const REDUCE_SUM_ALL_PARTIAL_F32: &str = r#"
extern "C" __global__ void reduce_sum_all_partial_f32(
    const float* __restrict__ in,
    double* __restrict__ partial,
    int numel)
{
    __shared__ double warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    double acc = 0.0;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc += (double)in[idx];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        double block_sum = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_xor_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            partial[blockIdx.x] = block_sum;
        }
    }
}
"#;

/// sum 全軸縮約 2 段目: `partial`（`num_partials` 要素。`double`。1
/// ブロックのみで起動）を再度 `double` で総和し、**最後に 1 回だけ**
/// `(float)` へ downcast して `out[0]` へ書く。
pub const REDUCE_SUM_ALL_FINALIZE_F32: &str = r#"
extern "C" __global__ void reduce_sum_all_finalize_f32(
    const double* __restrict__ partial,
    float* __restrict__ out,
    int num_partials)
{
    __shared__ double warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    double acc = 0.0;
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        acc += partial[idx];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        warp_sums[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        double block_sum = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_xor_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            out[0] = (float)block_sum;
        }
    }
}
"#;

/// sum 単一軸縮約（汎用版。`inner != 1`）: 1 スレッド 1 出力要素
/// （`idx = o * inner + i`）が縮約軸を `0..axis_len` の昇順で `double`
/// アキュムレータで逐次累積し、書き出し時のみ `(float)` へ downcast
/// する。`outer`／`axis_len`／`inner` の積は呼び出し元（`reduce.rs`）が
/// `i32::MAX` 範囲内であることを検証済み。添字計算は `long long` で
/// 行い overflow を避ける（`outer*inner` が `i32` 範囲を超えない一方、
/// 中間の `o * axis_len * inner` は超えうるため）。
pub const REDUCE_SUM_AXIS_F32: &str = r#"
extern "C" __global__ void reduce_sum_axis_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        double acc = 0.0;
        for (int a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + (long long)a) * (long long)inner + i;
            acc += (double)in[src];
        }
        out[idx] = (float)acc;
    }
}
"#;

/// sum 単一軸縮約（`inner == 1`。最終軸縮約の coalesced 版）: 1 warp が
/// 1 出力行（`row`）を担当し、`cols`（`axis_len`）を stride 32 で分割
/// して `double` で部分和を求めたのち warp 内 butterfly で結合する。
/// `rows`（`outer`）が `gridDim.x` を超える場合は persistent row loop で
/// ブロックを使い回す（起動ブロック数は `reduce.rs` が
/// `min(rows, REDUCE_MAX_BLOCKS)` で決定）。
pub const REDUCE_SUM_LASTAXIS_F32: &str = r#"
extern "C" __global__ void reduce_sum_lastaxis_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int rows,
    int cols)
{
    int lane = threadIdx.x;
    long long row_stride = (long long)gridDim.x;
    for (long long row = (long long)blockIdx.x; row < rows; row += row_stride) {
        double acc = 0.0;
        for (int c = lane; c < cols; c += 32) {
            long long src = row * (long long)cols + (long long)c;
            acc += (double)in[src];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) {
            out[row] = (float)acc;
        }
    }
}
"#;

/// max 全軸縮約 1 段目: 各ブロックが担当区間の `max(in[i])` を `float`
/// アキュムレータ（`fmaxf`。本ファイル冒頭コメント「max は厳密選択」
/// 参照）で計算し `partial[blockIdx.x]` へ書く。単位元は `-INFINITY`。
pub const REDUCE_MAX_ALL_PARTIAL_F32: &str = r#"
extern "C" __global__ void reduce_max_all_partial_f32(
    const float* __restrict__ in,
    float* __restrict__ partial,
    int numel)
{
    __shared__ float warp_maxes[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float acc = -INFINITY;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc = fmaxf(acc, in[idx]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc = fmaxf(acc, __shfl_xor_sync(0xffffffff, acc, offset));
    }
    if (lane == 0) {
        warp_maxes[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = (lane < 8) ? warp_maxes[lane] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, offset));
        }
        if (lane == 0) {
            partial[blockIdx.x] = block_max;
        }
    }
}
"#;

/// max 全軸縮約 2 段目: `partial`（`num_partials` 要素。`float`。1
/// ブロックのみで起動）を再度 `fmaxf` で結合し `out[0]` へ書く。
pub const REDUCE_MAX_ALL_FINALIZE_F32: &str = r#"
extern "C" __global__ void reduce_max_all_finalize_f32(
    const float* __restrict__ partial,
    float* __restrict__ out,
    int num_partials)
{
    __shared__ float warp_maxes[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    float acc = -INFINITY;
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        acc = fmaxf(acc, partial[idx]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc = fmaxf(acc, __shfl_xor_sync(0xffffffff, acc, offset));
    }
    if (lane == 0) {
        warp_maxes[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_max = (lane < 8) ? warp_maxes[lane] : -INFINITY;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_xor_sync(0xffffffff, block_max, offset));
        }
        if (lane == 0) {
            out[0] = block_max;
        }
    }
}
"#;

/// max 単一軸縮約（汎用版。`inner != 1`）。`REDUCE_SUM_AXIS_F32` と同一
/// 構造だが `float`／`fmaxf` で累積する（丸めなし）。呼び出し元は
/// `axis_len == 0` の起動を行わない契約（空縮約は host 側で拒否済み。
/// `reduce.rs` 参照）。
pub const REDUCE_MAX_AXIS_F32: &str = r#"
extern "C" __global__ void reduce_max_axis_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        float acc = -INFINITY;
        for (int a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + (long long)a) * (long long)inner + i;
            acc = fmaxf(acc, in[src]);
        }
        out[idx] = acc;
    }
}
"#;

/// max 単一軸縮約（`inner == 1`。最終軸縮約の coalesced 版）。
/// `REDUCE_SUM_LASTAXIS_F32` と同一構造だが `float`／`fmaxf` で累積
/// する。
pub const REDUCE_MAX_LASTAXIS_F32: &str = r#"
extern "C" __global__ void reduce_max_lastaxis_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    int rows,
    int cols)
{
    int lane = threadIdx.x;
    long long row_stride = (long long)gridDim.x;
    for (long long row = (long long)blockIdx.x; row < rows; row += row_stride) {
        float acc = -INFINITY;
        for (int c = lane; c < cols; c += 32) {
            long long src = row * (long long)cols + (long long)c;
            acc = fmaxf(acc, in[src]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc = fmaxf(acc, __shfl_xor_sync(0xffffffff, acc, offset));
        }
        if (lane == 0) {
            out[row] = acc;
        }
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 境界検査・非 atomic 決定性の証跡（`kernels_mse.rs::tests` と
    /// 同型の文字列検査。実際に NVRTC でコンパイル可能かは実機依存だが、
    /// 境界検査・atomicAdd/atomicMax 不使用は文字列検査で機械的に固定
    /// できる）。
    #[test]
    fn reduce_all_partial_kernels_have_grid_stride_bound_check_and_no_atomics() {
        for src in [REDUCE_SUM_ALL_PARTIAL_F32, REDUCE_MAX_ALL_PARTIAL_F32] {
            assert!(src.contains("idx < numel"));
            assert!(!src.contains("atomicAdd"));
            assert!(!src.contains("atomicMax"));
            assert!(
                src.contains("long long stride = (long long)gridDim.x * blockDim.x;"),
                "stride が long long で宣言されていない: {src}"
            );
        }
    }

    #[test]
    fn reduce_all_finalize_kernels_have_bound_check_and_no_atomics() {
        for src in [REDUCE_SUM_ALL_FINALIZE_F32, REDUCE_MAX_ALL_FINALIZE_F32] {
            assert!(src.contains("idx < num_partials"));
            assert!(!src.contains("atomicAdd"));
            assert!(!src.contains("atomicMax"));
        }
    }

    #[test]
    fn reduce_axis_kernels_have_bound_check_and_long_long_indices() {
        for src in [REDUCE_SUM_AXIS_F32, REDUCE_MAX_AXIS_F32] {
            assert!(src.contains("idx < total"));
            assert!(src.contains("long long total = (long long)outer * (long long)inner;"));
            assert!(!src.contains("atomicAdd"));
            assert!(!src.contains("atomicMax"));
        }
    }

    #[test]
    fn reduce_lastaxis_kernels_have_row_bound_check() {
        for src in [REDUCE_SUM_LASTAXIS_F32, REDUCE_MAX_LASTAXIS_F32] {
            assert!(src.contains("row < rows"));
            assert!(!src.contains("atomicAdd"));
            assert!(!src.contains("atomicMax"));
        }
    }

    #[test]
    fn sum_kernels_use_double_accumulator() {
        assert!(REDUCE_SUM_ALL_PARTIAL_F32.contains("double acc"));
        assert!(REDUCE_SUM_ALL_PARTIAL_F32.contains("double* __restrict__ partial"));
        assert!(REDUCE_SUM_ALL_FINALIZE_F32.contains("double acc"));
        assert!(REDUCE_SUM_ALL_FINALIZE_F32.contains("(float)block_sum"));
        assert!(REDUCE_SUM_AXIS_F32.contains("double acc"));
        assert!(REDUCE_SUM_AXIS_F32.contains("out[idx] = (float)acc;"));
        assert!(REDUCE_SUM_LASTAXIS_F32.contains("double acc"));
        assert!(REDUCE_SUM_LASTAXIS_F32.contains("out[row] = (float)acc;"));
    }

    #[test]
    fn max_kernels_use_float_fmaxf_and_neg_infinity_identity() {
        for src in [
            REDUCE_MAX_ALL_PARTIAL_F32,
            REDUCE_MAX_ALL_FINALIZE_F32,
            REDUCE_MAX_AXIS_F32,
            REDUCE_MAX_LASTAXIS_F32,
        ] {
            assert!(src.contains("fmaxf"));
            assert!(src.contains("-INFINITY"));
            assert!(
                !src.contains("double"),
                "max カーネルに double が混入: {src}"
            );
        }
    }
}
