//! `TypedOps<f64>`（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／
//! `max`。最小集合 8 演算）の CUDA C カーネルソース（NVRTC 実行時コンパイル
//! 用の静的文字列。イシュー #2060・親 #1650）。
//!
//! `typed_f64.rs`（呼び出し元）は本モジュールの 12 定数を `nvrtc::
//! compile_ptx` に渡し `CudaFunction` を得る。`kernels.rs`／
//! `kernels_elementwise.rs`／`kernels_reduce.rs` と同じ理由でソースを
//! `nvcc` 事前コンパイルせず文字列のまま埋め込む（ビルド時に nvcc/CUDA
//! ヘッダを一切要求しない。「CUDA toolkit 非搭載環境でも `cargo build
//! --workspace` が成立する」契約を維持する。`.claude/rules/deps-policy.md`）。
//!
//! # 数値契約: bit 完全一致／REQ-2 統一複合判定の使い分け
//!
//! `crates/backend-cpu/src/typed_f64.rs` が意味論の正である。累積順序が
//! CPU 参照実装と構造的に一致する演算は **bit 完全一致**を目標とし、
//! 構造的に異なる演算（GPU 側の並列縮約木と CPU 側の `CHUNK` 単位分割が
//! 異なる順序で結合される全軸縮約）は **REQ-2 統一複合判定**（相対誤差
//! 1e-3 未満 または 絶対誤差 1e-5 未満。`.claude/rules/coding-rust.md`）を
//! 適用する（正しさの検証は `crates/backend-cuda/tests/
//! typed_ops_f64_parity.rs` の実機 `#[ignore]` テストが担う。本ファイル
//! 自体は文字列定数の定義のみ）。
//!
//! - **bit 完全一致を狙う**（累積順序が CPU と同一）:
//!   - `gemm_naive_f64`: CPU `gemm_row_parallel_f64` は各出力要素
//!     `c[i,j]` を縮約軸 `p` の昇順で `f64::mul_add` により逐次更新する
//!     （行 `i` は rayon で並列化するが、固定 `(i,j)` に対する `p` の
//!     累積順序自体は逐次と同一）。本カーネルは 1 スレッド = 1 出力要素で
//!     `p` を `0..k` の昇順に走査し `fma(double,double,double)`
//!     （IEEE 754 correctly-rounded FMA。`.claude/rules/coding-rust.md`
//!     の FMA 契約統一節を f64 へ拡張）で累積するため、同一の逐次順序に
//!     なる。
//!   - `ew_add_f64`／`ew_mul_f64`／`ew_relu_f64`: 順序非依存の純粋算術
//!     （加算・乗算・比較のみ）のため並列度に関わらず一致する。
//!   - `reduce_sum_axis_f64`（軸指定 `sum`）: CPU `axis_reduce_f64` は
//!     出力要素ごとに縮約軸を `0..axis_len` の昇順で `acc = acc + v`
//!     と単純加算する（`CHUNK` 分割は全軸縮約〈`sum_slice_f64`〉限定で
//!     軸指定には適用されない）。本カーネルは 1 スレッド = 1 出力要素で
//!     同じ昇順の単純加算を行うため一致する。
//!   - `reduce_max_axis_f64`（軸指定 `max`）: CPU `axis_reduce_f64` は
//!     `acc = f64::max(acc, v)` を同じ昇順で適用する。`f64::max` と
//!     CUDA `fmax(double,double)` はいずれも「一方が NaN ならもう一方を
//!     返す」IEEE 754 準拠の意味論（`kernels_reduce.rs` の `fmaxf` と同型）
//!     のため一致する。
//! - **REQ-2 複合判定を適用する**（累積順序が構造的に異なる）:
//!   - `ew_exp_f64`／`ew_tanh_f64`: デバイス側 libm（`exp`／`tanh` の
//!     double 版組み込み関数）と CPU 側 libm の丸め差
//!     （`kernels_elementwise.rs` の `expf`／`tanhf` と同型の既知事項）。
//!   - `reduce_sum_all_partial_f64`／`_finalize_f64`（全軸縮約 `sum`）:
//!     CPU は `CHUNK=4096` 単位のチャンク分割・チャンク間結合で結合する
//!     一方、GPU は warp シャッフルによる 2 段木構造で結合するため、
//!     浮動小数点加算の非結合性により累積順序が異なりうる
//!     （`kernels_reduce.rs::REDUCE_SUM_ALL_PARTIAL_F32` と同型の判断。
//!     ただし f64 の場合は `f32→f64` 昇格が不要なぶん CPU 側とアキュムレータ
//!     型自体は揃う）。
//!   - `reduce_max_all_partial_f64`／`_finalize_f64`（全軸縮約 `max`）:
//!     同上の理由（`CHUNK` 分割 対 2 段木）。
//!
//! # ブロードキャスト
//!
//! 二項カーネル（`ew_add_f64`／`ew_mul_f64`）は同一長の 1 次元化済み
//! バッファのみを扱う（ブロードキャスト非対応）。ブロードキャスト対応は
//! 呼び出し元（`typed_f64.rs`）が `Tensor::broadcast_with` →
//! `contiguous()` で同一 shape へ実体化してから本カーネルへ渡す契約
//! （`kernels_elementwise.rs` と同じ役割分担）。
//!
//! # NVRTC の `INFINITY` マクロ非対応（イシュー #1893）
//!
//! NVRTC は `<math.h>` を暗黙に含めないため `INFINITY`／`-INFINITY`
//! マクロは未定義識別子としてコンパイルエラーになる（DGX Spark GB10・
//! `compute_121` で実測確認済み）。`max` 系カーネルは単位元 −inf を
//! `#define NEG_INF_F64 (__longlong_as_double(0xfff0000000000000LL))`
//! （各カーネル文字列に自己完結する形で埋め込む。定数は各文字列ごと
//! 個別に NVRTC コンパイルされるため共通プレフィックスへは括り出さ
//! ない）という bit パターン直接構成で表現する。`__longlong_as_double` は
//! 同族の `__uint_as_float`（`kernels_reduce.rs`）・`__float_as_uint`
//! （`kernels_cast.rs`）が GB10 上のコンパイル・実行実績を持つのと同じ
//! 組み込み device function（include path 不要）である。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! 全カーネルで手動境界チェック（`row < m && col < n`／`idx < numel`／
//! `idx < num_partials`／`idx < total`）を維持する
//! （`.claude/rules/coding-rust.md`）。

/// elementwise／reduction 系カーネルの 1 次元ブロックあたりスレッド数
/// （`kernels_elementwise::EW_BLOCK_DIM`／`kernels_reduce::
/// REDUCE_BLOCK_DIM` と同じ値だが、本モジュールは f32 実装から独立した
/// 定数として保持する。理由は本モジュール冒頭コメント「意味論の正」節
/// 参照——将来 f32 側の値だけを変更したくなった場合に f64 側が無関係に
/// 追従してしまう結合を避ける）。
pub const TYPED_F64_BLOCK_DIM: u32 = 256;

/// GEMM カーネル起動 1 回あたりのブロック次元（`gemm.rs::
/// NAIVE_BLOCK_DIM` と同じ値。理由は [`TYPED_F64_BLOCK_DIM`] と同様）。
pub const TYPED_F64_GEMM_BLOCK_DIM: (u32, u32, u32) = (16, 16, 1);

/// 全軸縮約 2 段目（`reduce_sum_all_finalize_f64`／
/// `reduce_max_all_finalize_f64`）が単一ブロックで処理しきれる `partial`
/// の最大長（＝ 1 段目の起動ブロック数の上限）。
/// `kernels_reduce::REDUCE_MAX_BLOCKS` と同じ値・同じ理由。
pub const TYPED_F64_REDUCE_MAX_BLOCKS: u32 = 1024;

/// naive GEMM（f64）。1 スレッド = C の 1 要素。縮約軸 `p` を `0..k` の
/// 昇順に走査し `fma(double,double,double)` で累積する（本ファイル
/// 冒頭コメント「bit 完全一致を狙う」節参照。`f64::mul_add` と同じ
/// IEEE 754 correctly-rounded FMA）。
pub const GEMM_NAIVE_F64: &str = r#"
extern "C" __global__ void gemm_naive_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ c,
    int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        double acc = 0.0;
        for (int p = 0; p < k; ++p) {
            acc = fma(a[row * k + p], b[p * n + col], acc);
        }
        c[row * n + col] = acc;
    }
}
"#;

/// 二項加算 `out[i] = a[i] + b[i]`（f64）。
pub const EW_ADD_F64: &str = r#"
extern "C" __global__ void ew_add_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = a[idx] + b[idx];
    }
}
"#;

/// 二項乗算 `out[i] = a[i] * b[i]`（f64）。
pub const EW_MUL_F64: &str = r#"
extern "C" __global__ void ew_mul_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = a[idx] * b[idx];
    }
}
"#;

/// ReLU（`max(x, 0)`）。CPU 参照実装（`backend-cpu::typed_f64::
/// relu_slice_f64`。`x.max(0.0)`）と同じ比較演算子の向き（`x > 0.0`）を
/// 用いるため、NaN 入力時は `f64::max` と同様 NaN を無視し
/// `relu(NaN) == 0.0` を返す。
pub const EW_RELU_F64: &str = r#"
extern "C" __global__ void ew_relu_f64(
    const double* __restrict__ a,
    double* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        double x = a[idx];
        out[idx] = x > 0.0 ? x : 0.0;
    }
}
"#;

/// `exp(x)`（倍精度 `exp`。REQ-2 複合判定対象。本ファイル冒頭コメント
/// 参照）。
pub const EW_EXP_F64: &str = r#"
extern "C" __global__ void ew_exp_f64(
    const double* __restrict__ a,
    double* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = exp(a[idx]);
    }
}
"#;

/// `tanh(x)`（倍精度 `tanh`。REQ-2 複合判定対象）。
pub const EW_TANH_F64: &str = r#"
extern "C" __global__ void ew_tanh_f64(
    const double* __restrict__ a,
    double* __restrict__ out,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        out[idx] = tanh(a[idx]);
    }
}
"#;

/// sum 全軸縮約 1 段目（REQ-2 複合判定対象。本ファイル冒頭コメント
/// 参照）: 各ブロックが担当区間の `Σ in[i]` を `double` アキュムレータ
/// で計算し `partial[blockIdx.x]` へ書く。`kernels_reduce.rs::
/// REDUCE_SUM_ALL_PARTIAL_F32` と同一構造だが、入力自体が `double` の
/// ため `f32→f64` 昇格を経ない。
pub const REDUCE_SUM_ALL_PARTIAL_F64: &str = r#"
extern "C" __global__ void reduce_sum_all_partial_f64(
    const double* __restrict__ in,
    double* __restrict__ partial,
    int numel)
{
    __shared__ double warp_sums[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    double acc = 0.0;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc += in[idx];
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

/// sum 全軸縮約 2 段目（REQ-2 複合判定対象）: `partial`（`num_partials`
/// 要素。`double`。1 ブロックのみで起動）を再度 `double` で総和し
/// `out[0]` へ書く（f32 版と異なり downcast は発生しない）。
pub const REDUCE_SUM_ALL_FINALIZE_F64: &str = r#"
extern "C" __global__ void reduce_sum_all_finalize_f64(
    const double* __restrict__ partial,
    double* __restrict__ out,
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
            out[0] = block_sum;
        }
    }
}
"#;

/// sum 単一軸縮約（bit 完全一致を狙う。本ファイル冒頭コメント参照）:
/// 1 スレッドが 1 出力要素を担当し、縮約軸を `0..axis_len` の昇順で
/// `double` アキュムレータへ単純加算する（CPU 参照実装
/// `axis_reduce_f64` と同一順序）。`kernels_reduce.rs::
/// REDUCE_SUM_AXIS_F32` と同型（`long long` 添字による overflow 回避を
/// 含む）だが downcast は発生しない。
pub const REDUCE_SUM_AXIS_F64: &str = r#"
extern "C" __global__ void reduce_sum_axis_f64(
    const double* __restrict__ in,
    double* __restrict__ out,
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
        for (long long a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + a) * (long long)inner + i;
            acc += in[src];
        }
        out[idx] = acc;
    }
}
"#;

/// max 全軸縮約 1 段目（REQ-2 複合判定対象）: 各ブロックが担当区間の
/// `max(in[i])` を `double` アキュムレータ（`fmax`。厳密選択・丸めなし）
/// で計算し `partial[blockIdx.x]` へ書く。単位元は −inf
/// （`NEG_INF_F64`。本ファイル冒頭コメント「NVRTC の `INFINITY` マクロ
/// 非対応」参照）。
pub const REDUCE_MAX_ALL_PARTIAL_F64: &str = r#"
#define NEG_INF_F64 (__longlong_as_double(0xfff0000000000000LL))
extern "C" __global__ void reduce_max_all_partial_f64(
    const double* __restrict__ in,
    double* __restrict__ partial,
    int numel)
{
    __shared__ double warp_maxes[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    double acc = NEG_INF_F64;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += stride) {
        acc = fmax(acc, in[idx]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc = fmax(acc, __shfl_xor_sync(0xffffffff, acc, offset));
    }
    if (lane == 0) {
        warp_maxes[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        double block_max = (lane < 8) ? warp_maxes[lane] : NEG_INF_F64;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_max = fmax(block_max, __shfl_xor_sync(0xffffffff, block_max, offset));
        }
        if (lane == 0) {
            partial[blockIdx.x] = block_max;
        }
    }
}
"#;

/// max 全軸縮約 2 段目（REQ-2 複合判定対象）: `partial`（`num_partials`
/// 要素。`double`。1 ブロックのみで起動）を再度 `fmax` で結合し `out[0]`
/// へ書く。
pub const REDUCE_MAX_ALL_FINALIZE_F64: &str = r#"
#define NEG_INF_F64 (__longlong_as_double(0xfff0000000000000LL))
extern "C" __global__ void reduce_max_all_finalize_f64(
    const double* __restrict__ partial,
    double* __restrict__ out,
    int num_partials)
{
    __shared__ double warp_maxes[8];
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    double acc = NEG_INF_F64;
    for (int idx = threadIdx.x; idx < num_partials; idx += blockDim.x) {
        acc = fmax(acc, partial[idx]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc = fmax(acc, __shfl_xor_sync(0xffffffff, acc, offset));
    }
    if (lane == 0) {
        warp_maxes[warp_id] = acc;
    }
    __syncthreads();

    if (warp_id == 0) {
        double block_max = (lane < 8) ? warp_maxes[lane] : NEG_INF_F64;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_max = fmax(block_max, __shfl_xor_sync(0xffffffff, block_max, offset));
        }
        if (lane == 0) {
            out[0] = block_max;
        }
    }
}
"#;

/// max 単一軸縮約（bit 完全一致を狙う）: `REDUCE_SUM_AXIS_F64` と同一
/// 構造（`long long` 添字）だが `double`／`fmax` で累積する（丸めなし。
/// CPU 参照実装 `axis_reduce_f64` の `f64::max` 適用順序と一致する）。
pub const REDUCE_MAX_AXIS_F64: &str = r#"
#define NEG_INF_F64 (__longlong_as_double(0xfff0000000000000LL))
extern "C" __global__ void reduce_max_axis_f64(
    const double* __restrict__ in,
    double* __restrict__ out,
    int outer,
    int axis_len,
    int inner)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)outer * (long long)inner;
    if (idx < total) {
        long long o = idx / inner;
        long long i = idx % inner;
        double acc = NEG_INF_F64;
        for (long long a = 0; a < axis_len; a++) {
            long long src = (o * (long long)axis_len + a) * (long long)inner + i;
            acc = fmax(acc, in[src]);
        }
        out[idx] = acc;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-8 境界検査・非 atomic 決定性の証跡（`kernels_reduce.rs::tests`
    /// と同型の文字列検査）。
    #[test]
    fn gemm_kernel_has_bound_check_and_fma() {
        assert!(GEMM_NAIVE_F64.contains("row < m && col < n"));
        assert!(GEMM_NAIVE_F64.contains("fma("));
        assert!(GEMM_NAIVE_F64.contains("double acc"));
    }

    #[test]
    fn elementwise_kernels_have_bound_check() {
        for src in [EW_ADD_F64, EW_MUL_F64, EW_RELU_F64, EW_EXP_F64, EW_TANH_F64] {
            assert!(src.contains("idx < numel"));
        }
    }

    #[test]
    fn reduce_all_partial_kernels_have_grid_stride_bound_check_and_no_atomics() {
        for src in [REDUCE_SUM_ALL_PARTIAL_F64, REDUCE_MAX_ALL_PARTIAL_F64] {
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
        for src in [REDUCE_SUM_ALL_FINALIZE_F64, REDUCE_MAX_ALL_FINALIZE_F64] {
            assert!(src.contains("idx < num_partials"));
            assert!(!src.contains("atomicAdd"));
            assert!(!src.contains("atomicMax"));
        }
    }

    #[test]
    fn reduce_axis_kernels_have_bound_check_and_long_long_indices() {
        for src in [REDUCE_SUM_AXIS_F64, REDUCE_MAX_AXIS_F64] {
            assert!(src.contains("idx < total"));
            assert!(src.contains("long long total = (long long)outer * (long long)inner;"));
            assert!(!src.contains("atomicAdd"));
            assert!(!src.contains("atomicMax"));
        }
    }

    #[test]
    fn sum_kernels_use_double_accumulator_without_downcast() {
        assert!(REDUCE_SUM_ALL_PARTIAL_F64.contains("double acc"));
        assert!(REDUCE_SUM_ALL_PARTIAL_F64.contains("double* __restrict__ partial"));
        assert!(REDUCE_SUM_ALL_FINALIZE_F64.contains("double acc"));
        assert!(REDUCE_SUM_ALL_FINALIZE_F64.contains("out[0] = block_sum;"));
        assert!(!REDUCE_SUM_ALL_FINALIZE_F64.contains("(float)"));
        assert!(REDUCE_SUM_AXIS_F64.contains("double acc"));
        assert!(REDUCE_SUM_AXIS_F64.contains("out[idx] = acc;"));
        assert!(!REDUCE_SUM_AXIS_F64.contains("(float)"));
    }

    /// max カーネルが `fmax`（倍精度）・単位元 −inf（`NEG_INF_F64`。NVRTC
    /// は `<math.h>` を暗黙に含めないため C マクロ `INFINITY` は使えない。
    /// イシュー #1893）という契約を満たすことを固定する。
    /// `#define NEG_INF_F64 (__longlong_as_double(0xfff0000000000000LL))`
    /// が各カーネル文字列に自己完結していることも検査する（定数は各
    /// 文字列ごと個別に NVRTC コンパイルされるため共通プレフィックスへの
    /// 括り出しはできない）。`fmaxf`（単精度版）を誤って使っていないこと
    /// も確認する。
    #[test]
    fn max_kernels_use_double_fmax_and_neg_inf_bit_pattern_identity() {
        for src in [
            REDUCE_MAX_ALL_PARTIAL_F64,
            REDUCE_MAX_ALL_FINALIZE_F64,
            REDUCE_MAX_AXIS_F64,
        ] {
            assert!(src.contains("fmax("));
            assert!(!src.contains("fmaxf("), "単精度版 fmaxf が混入: {src}");
            assert!(
                src.contains("#define NEG_INF_F64 (__longlong_as_double(0xfff0000000000000LL))")
            );
            assert!(src.contains("double acc = NEG_INF_F64;"));
        }
    }

    /// カーネル 12 定数すべてが NVRTC 組み込みヘッダに含まれない
    /// `INFINITY` マクロを参照しないこと（逆戻り防止）・`#include` に
    /// 依存しないこと・`__FLT_MAX__`（イシュー #1101 で compute_121
    /// 未定義と実測済み）を使わないことを fail-closed に固定する
    /// （NVRTC は `<math.h>` を暗黙に含めない。イシュー #1893）。
    #[test]
    fn kernels_do_not_reference_nvrtc_undefined_infinity_macro() {
        for src in [
            GEMM_NAIVE_F64,
            EW_ADD_F64,
            EW_MUL_F64,
            EW_RELU_F64,
            EW_EXP_F64,
            EW_TANH_F64,
            REDUCE_SUM_ALL_PARTIAL_F64,
            REDUCE_SUM_ALL_FINALIZE_F64,
            REDUCE_SUM_AXIS_F64,
            REDUCE_MAX_ALL_PARTIAL_F64,
            REDUCE_MAX_ALL_FINALIZE_F64,
            REDUCE_MAX_AXIS_F64,
        ] {
            assert!(
                !src.contains("INFINITY"),
                "NVRTC 未定義の INFINITY マクロが残存: {src}"
            );
            assert!(!src.contains("#include"), "ヘッダ依存を追加しない: {src}");
            assert!(
                !src.contains("__FLT_MAX__"),
                "__FLT_MAX__ は compute_121 で未定義（#1101）: {src}"
            );
        }
    }
}
