//! デバイス常駐 KV（K-3）の CUDA C カーネルソース（NVRTC 実行時コンパイル用の
//! 静的文字列。イシュー #3111・親 #3108。設計 `docs/device-resident-kv-paged-
//! attention-design.md` §3〜§5）。
//!
//! `paged_kv.rs`（呼び出し元）が本モジュールの 2 定数を `nvrtc::compile_ptx` に渡して
//! `CudaFunction` を得る。ソースは定数文字列だけで、利用者の入力を連結しない
//! （OWASP A03）。
//!
//! # `paged_kv_append_f32`
//!
//! 1 スレッドが 1 要素（行 `r`・成分 `e`）を担当し grid-stride で回す。行 `r` の書き先は
//! スロット写像 `slot_mapping[r]`（= `block_id · block_size + pos % block_size`）。
//! 重複スロットはホスト検証（`tensor-core::validate_paged_kv_append`）で拒否済みなので
//! 書き込みは競合しない。
//!
//! # `paged_attention_f32` と縮約順序（REQ-2）
//!
//! 1 query 行 `(b, h, i)` に 1 ブロック（`PAGED_ATTENTION_BLOCK_DIM` スレッド）を割り当てる。
//!
//! 1. score: スレッド `t` が `j = t, t + T, …` の score を計算する。各 score は `q · scale`
//!    を `__fmul_rn` で先に作り、`Dh` 方向に `fmaf` で**逐次**累積する（CPU 参照と同順）。
//!    結果は作業バッファ `[B · H · L_new, s_stride]` へ書く
//! 2. max: lane 0 が `j` の論理順に逐次で求める
//! 3. exp: 並列に `expf(s − max)`。**和は lane 0 が `j` の論理順に逐次**で足す
//! 4. 加重和: スレッド `d` が出力成分 `d`（`Dh` が `T` を超えるときはストライド）を受け持ち、
//!    `j` を論理順に逐次たどって `o = fmaf(p, v, o)`（`p = w / sum`）で累積する
//!
//! したがって **`j` 軸（系列方向）は単一の連続ループ**で、`Dh` 軸も逐次の `fmaf` なので、
//! REQ-2 の「結合順序が単一の連続 K ループと異なるカーネル」には当たらず、統一複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま適用する。flash-decoding 型の
//! split・`j` 軸の並列リダクション・warp shuffle による和は入れない（入れると baseline
//! 方式の対象になるため、第 1 段階では採らない）。`expf` は参照の `f32::exp` との差を
//! 統一複合判定の範囲で扱う（`kernels_softmax.rs` の `exp2f` 契約は softmax 専用）。
//!
//! # REQ-8（カーネル境界検査規約）
//!
//! - append: `idx < total`（grid-stride 条件）に加え、`r < rows`・`e < embed_dim`・
//!   `slot < pool_rows` を手動で検査し、範囲外なら書かずに抜ける
//! - attention: ブロックテーブルの添字（`j / block_size < max_blocks_per_seq`）・
//!   ブロック ID（`< num_blocks`。番兵 `0xFFFFFFFF` はここで拒否）・`seq_len` の範囲
//!   （`L_new <= seq_len`・`limit <= s_stride`）を検査し、違反を見つけたら出力へ **NaN** を
//!   書いて抜ける（ホストで検査済みのため通常は到達しない多層防御。黙ってゼロにはしない）
//! - インデックス計算は 64bit（`long long`）。ベクトル化ロードは使わない

/// attention カーネルの 1 ブロックあたりのスレッド数。
pub const PAGED_ATTENTION_BLOCK_DIM: u32 = 128;

/// append カーネルの 1 ブロックあたりのスレッド数。
pub const PAGED_APPEND_BLOCK_DIM: u32 = 256;

/// append カーネルの grid 幅の上限（grid-stride で回すので十分な値）。
pub const PAGED_APPEND_MAX_BLOCKS: u32 = 65_535;

/// 新トークンの K／V をスロット写像どおりにプールへ書くカーネル。
pub const PAGED_KV_APPEND_F32: &str = r#"
extern "C" __global__ void paged_kv_append_f32(
    const float* __restrict__ k_new,
    const float* __restrict__ v_new,
    float* __restrict__ k_pool,
    float* __restrict__ v_pool,
    const unsigned int* __restrict__ slot_mapping,
    int rows,
    int embed_dim,
    long long pool_rows)
{
    long long total = (long long)rows * (long long)embed_dim;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        long long r = idx / embed_dim;
        long long e = idx - r * (long long)embed_dim;
        if (r >= rows || e >= embed_dim) {
            continue;
        }
        long long slot = (long long)slot_mapping[r];
        if (slot >= pool_rows) {
            continue;
        }
        long long dst = slot * (long long)embed_dim + e;
        k_pool[dst] = k_new[idx];
        v_pool[dst] = v_new[idx];
    }
}
"#;

/// ブロックを論理順に走査する paged attention カーネル（縮約順序はモジュール doc）。
pub const PAGED_ATTENTION_F32: &str = r#"
// NVRTC does not include <math.h> implicitly: no infinity / NaN macros are available
// (same reason as kernels_arg_reduce.rs and kernels_reduce.rs). Define by bit pattern.
#define PAGED_NEG_INF (__uint_as_float(0xff800000u))
#define PAGED_QNAN (__uint_as_float(0x7fc00000u))
extern "C" __global__ void paged_attention_f32(
    const float* __restrict__ q,
    const float* __restrict__ k_pool,
    const float* __restrict__ v_pool,
    const unsigned int* __restrict__ block_tables,
    const unsigned int* __restrict__ seq_lens,
    float* scores,
    float* __restrict__ out,
    int batch,
    int l_new,
    int num_heads,
    int head_dim,
    int embed_dim,
    int block_size,
    unsigned int num_blocks,
    int max_blocks_per_seq,
    int s_stride,
    float scale)
{
    __shared__ int s_bad;
    __shared__ float s_max;
    __shared__ float s_sum;

    long long qrow = (long long)blockIdx.x;
    long long i_ll = qrow % l_new;
    long long h_ll = (qrow / l_new) % num_heads;
    long long b_ll = qrow / ((long long)l_new * num_heads);
    if (b_ll >= batch) {
        return;
    }
    int i = (int)i_ll;
    int h = (int)h_ll;
    int b = (int)b_ll;

    int seq_len = (int)seq_lens[b];
    int s_prev = seq_len - l_new;
    int limit = s_prev + i + 1;
    long long q_off = ((long long)b * l_new + i) * (long long)embed_dim + (long long)h * head_dim;
    float* score_row = scores + qrow * (long long)s_stride;

    if (threadIdx.x == 0) {
        s_bad = 0;
        s_max = PAGED_NEG_INF;
        s_sum = 0.0f;
    }
    __syncthreads();

    // seq_len range violations are judged from the same values by all threads (uniform in the block).
    // Thread 0 alone writes the shared flag; the barrier publishes it (no concurrent writers).
    if (threadIdx.x == 0 && (s_prev < 0 || limit > seq_len || limit > s_stride)) {
        s_bad = 1;
    }
    __syncthreads();

    // 1. score: j in logical order, Dh axis accumulated sequentially with fmaf.
    // Per-thread violation flag (no shared write inside the loop); reduced block-wide below.
    int bad_local = 0;
    if (!s_bad) {
        for (int j = threadIdx.x; j < limit; j += blockDim.x) {
            int slot_idx = j / block_size;
            if (slot_idx >= max_blocks_per_seq) {
                bad_local = 1;
                continue;
            }
            unsigned int id = block_tables[(long long)b * max_blocks_per_seq + slot_idx];
            if (id >= num_blocks) {
                bad_local = 1;
                continue;
            }
            long long k_off = (((long long)id * block_size + (j % block_size)) * (long long)embed_dim)
                + (long long)h * head_dim;
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                acc = fmaf(__fmul_rn(q[q_off + d], scale), k_pool[k_off + d], acc);
            }
            score_row[j] = acc;
        }
    }
    // Block-wide OR of the per-thread flags (also acts as the barrier after the score writes).
    if (__syncthreads_or(bad_local | s_bad)) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            out[q_off + d] = PAGED_QNAN;
        }
        return;
    }

    // 2. max: thread 0 scans sequentially in logical order.
    if (threadIdx.x == 0) {
        float m = PAGED_NEG_INF;
        for (int j = 0; j < limit; ++j) {
            float s = score_row[j];
            if (s > m) {
                m = s;
            }
        }
        s_max = m;
    }
    __syncthreads();

    // 3. exp in parallel; the sum is taken by thread 0 sequentially in logical order.
    for (int j = threadIdx.x; j < limit; j += blockDim.x) {
        score_row[j] = expf(score_row[j] - s_max);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int j = 0; j < limit; ++j) {
            sum += score_row[j];
        }
        s_sum = sum;
    }
    __syncthreads();

    // 4. weighted sum: for each component d, j sequentially in logical order. The block table
    //    was checked in the score stage, but the manual bound checks are kept.
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float o = 0.0f;
        bool bad = false;
        for (int j = 0; j < limit; ++j) {
            int slot_idx = j / block_size;
            if (slot_idx >= max_blocks_per_seq) {
                bad = true;
                break;
            }
            unsigned int id = block_tables[(long long)b * max_blocks_per_seq + slot_idx];
            if (id >= num_blocks) {
                bad = true;
                break;
            }
            float p = score_row[j] / s_sum;
            long long v_off = (((long long)id * block_size + (j % block_size)) * (long long)embed_dim)
                + (long long)h * head_dim + d;
            o = fmaf(p, v_pool[v_off], o);
        }
        out[q_off + d] = bad ? PAGED_QNAN : o;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_kernel_keeps_manual_bound_checks() {
        let s = PAGED_KV_APPEND_F32;
        assert!(s.contains("idx < total"));
        assert!(s.contains("r >= rows || e >= embed_dim"));
        assert!(s.contains("slot >= pool_rows"));
        // 64bit インデックス。
        assert!(s.contains("long long dst = slot * (long long)embed_dim + e;"));
        assert!(!s.contains("atomicAdd"));
    }

    #[test]
    fn attention_kernel_keeps_manual_bound_checks_and_nan_on_violation() {
        let s = PAGED_ATTENTION_F32;
        assert!(s.contains("slot_idx >= max_blocks_per_seq"));
        assert!(s.contains("id >= num_blocks"));
        assert!(s.contains("s_prev < 0 || limit > seq_len || limit > s_stride"));
        assert!(s.contains("= PAGED_QNAN;"));
        assert!(s.contains("b_ll >= batch"));
        // NVRTC 未定義のマクロを使わない。
        assert!(!s.contains("INFINITY"));
        assert!(!s.contains(" NAN"));
        assert!(!s.contains("#include"));
    }

    #[test]
    fn attention_kernel_fixes_reduction_order() {
        let s = PAGED_ATTENTION_F32;
        // score・加重和は fmaf で逐次累積し、q · scale は __fmul_rn で先に作る。
        assert!(s.contains("fmaf(__fmul_rn(q[q_off + d], scale), k_pool[k_off + d], acc)"));
        assert!(s.contains("o = fmaf(p, v_pool[v_off], o);"));
        // 和は lane 0 が論理順に逐次（`threadIdx.x == 0` ブロック内の単一ループ）。
        assert!(s.contains("sum += score_row[j];"));
        // j 軸の並列リダクション・warp 内集約・アトミックは入れない。
        for banned in [
            "__shfl",
            "atomicAdd",
            "atomicMax",
            "__reduce",
            "cooperative_groups",
        ] {
            assert!(!s.contains(banned), "{banned} must not appear");
        }
    }

    #[test]
    fn attention_kernel_uses_64bit_offsets() {
        let s = PAGED_ATTENTION_F32;
        assert!(s.contains("long long q_off"));
        assert!(s.contains("long long k_off"));
        assert!(s.contains("long long v_off"));
        assert!(s.contains("qrow * (long long)s_stride"));
    }

    #[test]
    fn no_untrusted_text_is_interpolated() {
        // ソースは定数文字列のみ（format! を使わない）ので `{`/`}` の置換は発生しない。
        assert!(PAGED_ATTENTION_F32.contains("extern \"C\" __global__"));
        assert!(PAGED_KV_APPEND_F32.contains("extern \"C\" __global__"));
    }
}
