//! デバイス常駐 KV（K-3）の HIP ソース（`backend-cuda::kernels_paged_kv` の移植。イシュー #3443・
//! 分解案 R12。設計 `docs/device-resident-kv-paged-attention-design.md` §3〜§5）。
//!
//! 役割: HIPRTC へ渡すソースの唯一の生成元（[`render_paged_kv_source`]）。呼び出し元は `paged_kv.rs` の
//! `RocmPagedKv`。2 本のカーネル（append・attention）を 1 ソース（1 module）にまとめ、数値の `#define`
//! （ブロック寸法 2 つ）だけを前置する（A03: 静的テンプレート＋検証済み数値のみ。shape・scale・ブロック
//! テーブルは引数／デバイスバッファで渡しソースへ入れない）。
//!
//! # `paged_kv_append_f32`
//!
//! 1 スレッドが 1 要素（行 `r`・成分 `e`）を担当し grid-stride で回す。行 `r` の書き先はスロット写像
//! `slot_mapping[r]`（= `block_id · block_size + pos % block_size`）。重複スロットはホスト検証
//! （`tensor-core::validate_paged_kv_append`）で拒否済みなので書き込みは競合しない。
//!
//! # `paged_attention_f32` と縮約順序（REQ-2）
//!
//! 1 query 行 `(b, h, i)` に 1 block（[`PAGED_ATTENTION_BLOCK_DIM`] スレッド）を割り当てる。
//!
//! 1. score: スレッド `t` が `j = t, t + T, …` を受け持ち、`(q · scale)` を `Dh` 方向に `__builtin_fmaf` で
//!    **逐次**累積する（CPU 参照の `(qv * scale).mul_add(kv, acc)` と同順）。結果は作業バッファへ
//! 2. max: thread 0 が `j` の論理順に逐次で求める
//! 3. exp: 並列に `expf(s − max)`。**和は thread 0 が `j` の論理順に逐次**で足す
//! 4. 加重和: スレッド `d` が出力成分 `d` を受け持ち、`j` を論理順にたどって `o = fmaf(p, v, o)`
//!    （`p = w / sum`）で累積する
//!
//! `j` 軸は単一の連続ループ、`Dh` 軸も逐次の FMA なので、REQ-2 の「結合順序が単一の連続 K ループと異なる
//! カーネル」には当たらず、統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）をそのまま適用する。
//!
//! # wave 幅・shuffle について（被覆計画の訂正）
//!
//! 被覆計画 §4 #101・#103 の「shfl を使う 1 か所」は現行 CUDA ソースと一致しない。カーネル本体は
//! `__shfl` を使わず、`j` 軸の並列リダクションも持たない（`__shfl` は CUDA 側テストの禁止リストに現れる
//! だけ）。wave 幅への依存は block 寸法だけで、attention 128・append 256 はいずれも 64 の倍数のため
//! wave64（MI300X）・wave32 とも端数 wave を作らない。よって `WarpGeometry` は使わない。
//!
//! # FMA・`f64` 契約
//!
//! ソース先頭に `#pragma clang fp contract(off)` を置き、暗黙の FMA 縮約を止める（CPU 参照は縮約しない）。
//! 意図した FMA は CPU の `f32::mul_add` と同じ 2 か所（score・加重和）だけで `__builtin_fmaf` で明示する。
//! 本件は正規化統計でも勾配の長軸縮約でもなく、CUDA・CPU 参照とも f32 の逐次和なので、`f64`
//! アキュムレータ契約（coding-rust.md）は**該当しない**。
//!
//! # REQ-8
//!
//! - append: `idx < total`（grid-stride 条件）に加え `r < rows`・`e < embed_dim`・`slot < pool_rows` を手動で
//!   検査し、範囲外なら書かずに抜ける
//! - attention: ブロックテーブルの添字（`j / block_size < max_blocks_per_seq`）・ブロック ID
//!   （`< num_blocks`。番兵 `0xFFFFFFFF` はここで拒否）・`seq_len` の範囲（`L_new <= seq_len`・
//!   `limit <= s_stride`）を score 段・加重和段の両方で検査し、違反時は出力へ **NaN** を書く（ホストで
//!   検査済みのため通常は到達しない多層防御。黙ってゼロにはしない）
//! - インデックス計算は 64bit（`long long`）。ベクトル化ロードは使わない
//!
//! # 引数型
//!
//! `launch::ArgKind` に符号なし 32bit がないため、CUDA 版の `unsigned int num_blocks` は
//! `long long num_blocks`（`Arg::I64`）に変えた。`PagedKvLayout::recheck` は `num_blocks < u32::MAX` しか
//! 保証せず `int` には収まらないため、`int` にはしない。カーネル内の比較は `(long long)id >= num_blocks`
//! で、番兵 `0xFFFFFFFF` も確実に拒否される。

use crate::error::RocmError;
use crate::kernel_source::KernelSource;
use crate::launch::{ArgKind, INDEX_OPS_BLOCK_DIM, KernelSpec};

/// attention カーネルの 1 block あたりのスレッド数（64 の倍数）。
pub(crate) const PAGED_ATTENTION_BLOCK_DIM: u32 = 128;

/// append カーネルの grid 幅の上限（grid-stride で回すので十分な値）。
pub(crate) const PAGED_APPEND_MAX_BLOCKS: u32 = 65_535;

const _: () = assert!(PAGED_ATTENTION_BLOCK_DIM.is_multiple_of(64));

const P: ArgKind = ArgKind::Ptr;
const I: ArgKind = ArgKind::I32;
const L: ArgKind = ArgKind::I64;
const F: ArgKind = ArgKind::F32;

/// `(k_new, v_new, k_pool, v_pool, slot_mapping, rows, embed_dim, pool_rows)`。
const PAGED_KV_APPEND_ARGS: [ArgKind; 8] = [P, P, P, P, P, I, I, L];
/// `(q, k_pool, v_pool, block_tables, seq_lens, scores, out, batch, l_new, num_heads, head_dim,
/// embed_dim, block_size, num_blocks, max_blocks_per_seq, s_stride, scale)`。
const PAGED_ATTENTION_ARGS: [ArgKind; 17] = [P, P, P, P, P, P, P, I, I, I, I, I, I, L, I, I, F];

pub(crate) const PAGED_KV_APPEND_SPEC: KernelSpec = KernelSpec {
    name: "paged_kv_append_f32",
    args: &PAGED_KV_APPEND_ARGS,
};
pub(crate) const PAGED_ATTENTION_SPEC: KernelSpec = KernelSpec {
    name: "paged_attention_f32",
    args: &PAGED_ATTENTION_ARGS,
};

/// 静的テンプレート本体。`PAGED_APPEND_BLOCK_DIM`・`PAGED_ATTENTION_BLOCK_DIM` は前置する `#define`。
pub(crate) const PAGED_KV_BODY: &str = r#"#pragma clang fp contract(off)

// HIPRTC has no <math.h> by default: define infinity / NaN by bit pattern (same as kernels_reduce.rs).
#define PAGED_NEG_INF (__uint_as_float(0xff800000u))
#define PAGED_QNAN (__uint_as_float(0x7fc00000u))

extern "C" __global__ __launch_bounds__(PAGED_APPEND_BLOCK_DIM) void paged_kv_append_f32(
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

extern "C" __global__ __launch_bounds__(PAGED_ATTENTION_BLOCK_DIM) void paged_attention_f32(
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
    long long num_blocks,
    int max_blocks_per_seq,
    int s_stride,
    float scale)
{
    __shared__ int s_bad;
    // Per-thread violation flags (written only by the owning thread; aggregated by thread 0 after a barrier).
    __shared__ int s_bad_flags[PAGED_ATTENTION_BLOCK_DIM];
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

    // 1. score: j in logical order, Dh axis accumulated sequentially with an explicit FMA.
    // Each thread keeps its violation flag in its own shared slot; after a barrier thread 0 folds them
    // into s_bad (single writer, no data race). The barrier after that publishes it.
    int bad_local = 0;
    if (!s_bad) {
        for (int j = threadIdx.x; j < limit; j += blockDim.x) {
            int slot_idx = j / block_size;
            if (slot_idx >= max_blocks_per_seq) {
                bad_local = 1;
                continue;
            }
            unsigned int id = block_tables[(long long)b * max_blocks_per_seq + slot_idx];
            if ((long long)id >= num_blocks) {
                bad_local = 1;
                continue;
            }
            long long k_off = (((long long)id * block_size + (j % block_size)) * (long long)embed_dim)
                + (long long)h * head_dim;
            float acc = 0.0f;
            for (long long d = 0; d < head_dim; ++d) {
                acc = __builtin_fmaf(q[q_off + d] * scale, k_pool[k_off + d], acc);
            }
            score_row[j] = acc;
        }
    }
    s_bad_flags[threadIdx.x] = bad_local;
    // Also the barrier after the score writes (they are read by other threads below).
    __syncthreads();
    if (threadIdx.x == 0) {
        int any_bad = s_bad;
        for (int t = 0; t < (int)blockDim.x; ++t) {
            any_bad |= s_bad_flags[t];
        }
        s_bad = any_bad;
    }
    __syncthreads();
    if (s_bad) {
        for (long long d = threadIdx.x; d < head_dim; d += blockDim.x) {
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

    // 4. weighted sum: for each component d, j sequentially in logical order. The block table was
    //    checked in the score stage, but the manual bound checks are kept.
    for (long long d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float o = 0.0f;
        bool bad = false;
        for (int j = 0; j < limit; ++j) {
            int slot_idx = j / block_size;
            if (slot_idx >= max_blocks_per_seq) {
                bad = true;
                break;
            }
            unsigned int id = block_tables[(long long)b * max_blocks_per_seq + slot_idx];
            if ((long long)id >= num_blocks) {
                bad = true;
                break;
            }
            float p = score_row[j] / s_sum;
            long long v_off = (((long long)id * block_size + (j % block_size)) * (long long)embed_dim)
                + (long long)h * head_dim + d;
            o = __builtin_fmaf(p, v_pool[v_off], o);
        }
        out[q_off + d] = bad ? PAGED_QNAN : o;
    }
}
"#;

/// paged KV 2 カーネル用ソースを生成する（ブロック寸法の `#define` 2 つだけ前置）。
pub(crate) fn render_paged_kv_source() -> Result<KernelSource, RocmError> {
    KernelSource::render(
        &[
            ("PAGED_APPEND_BLOCK_DIM", INDEX_OPS_BLOCK_DIM),
            ("PAGED_ATTENTION_BLOCK_DIM", PAGED_ATTENTION_BLOCK_DIM),
        ],
        PAGED_KV_BODY,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_source::{CompileSource, kernel_param_kinds};

    #[test]
    fn spec_matches_source_signature() {
        for spec in [PAGED_KV_APPEND_SPEC, PAGED_ATTENTION_SPEC] {
            assert_eq!(kernel_param_kinds(PAGED_KV_BODY, spec.name), spec.args);
        }
        // num_blocks は u32 全域を受けるため long long（I64）。
        assert_eq!(PAGED_ATTENTION_SPEC.args[13], ArgKind::I64);
    }

    #[test]
    fn source_defines_only_block_dims_and_starts_with_contract_off() {
        let s = render_paged_kv_source().unwrap();
        let defines: Vec<_> = s
            .as_str()
            .lines()
            .filter(|l| l.starts_with("#define PAGED_APPEND") || l.starts_with("#define PAGED_ATT"))
            .collect();
        assert_eq!(
            defines,
            [
                "#define PAGED_APPEND_BLOCK_DIM 256",
                "#define PAGED_ATTENTION_BLOCK_DIM 128"
            ]
        );
        assert!(PAGED_KV_BODY.starts_with("#pragma clang fp contract(off)\n"));
        assert_eq!(s, render_paged_kv_source().unwrap());
    }

    #[test]
    fn append_kernel_keeps_manual_bound_checks() {
        let s = PAGED_KV_BODY;
        assert!(s.contains("idx < total"));
        assert!(s.contains("r >= rows || e >= embed_dim"));
        assert!(s.contains("slot >= pool_rows"));
        assert!(s.contains("long long dst = slot * (long long)embed_dim + e;"));
    }

    #[test]
    fn attention_kernel_keeps_manual_bound_checks_and_nan_on_violation() {
        let s = PAGED_KV_BODY;
        // score 段・加重和段の両方で検査する。
        assert_eq!(s.matches("slot_idx >= max_blocks_per_seq").count(), 2);
        assert_eq!(s.matches("(long long)id >= num_blocks").count(), 2);
        assert!(s.contains("s_prev < 0 || limit > seq_len || limit > s_stride"));
        assert!(s.contains("= PAGED_QNAN;"));
        assert!(s.contains("bad ? PAGED_QNAN : o"));
        assert!(s.contains("b_ll >= batch"));
    }

    #[test]
    fn fma_only_at_score_and_weighted_sum() {
        let s = PAGED_KV_BODY;
        assert!(s.contains("acc = __builtin_fmaf(q[q_off + d] * scale, k_pool[k_off + d], acc);"));
        assert!(s.contains("o = __builtin_fmaf(p, v_pool[v_off], o);"));
        assert_eq!(s.matches("__builtin_fmaf(").count(), 2);
        let rest = s.replace("__builtin_fmaf(", "");
        for banned in ["fmaf(", "fma(", "__fmul_rn", "__expf"] {
            assert!(!rest.contains(banned), "{banned}");
        }
        // 和は thread 0 が論理順に逐次。
        assert!(s.contains("sum += score_row[j];"));
    }

    #[test]
    fn no_cross_lane_reduction_or_unavailable_macros() {
        let s = PAGED_KV_BODY;
        for banned in [
            "__shfl",
            "atomicAdd",
            "atomicMax",
            "__reduce",
            "cooperative_groups",
            "warpSize",
            "__ballot",
            "__syncthreads_or",
            "#include",
            "INFINITY",
            " NAN",
        ] {
            assert!(!s.contains(banned), "{banned} must not appear");
        }
    }

    #[test]
    fn attention_kernel_uses_64bit_offsets() {
        let s = PAGED_KV_BODY;
        assert!(s.contains("long long q_off"));
        assert!(s.contains("long long k_off"));
        assert!(s.contains("long long v_off"));
        assert!(s.contains("qrow * (long long)s_stride"));
    }
}
