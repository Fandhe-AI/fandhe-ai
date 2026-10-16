// デバイス常駐 KV（K-3）の MSL カーネルソース（イシュー #3111・親 #3108。設計
// `docs/device-resident-kv-paged-attention-design.md` §3〜§5。CUDA 側
// `backend-cuda::kernels_paged_kv`〈同イシュー〉の Metal 対応版）。
//
// `crate::paged_kv`（macOS 限定の起動 API）が `include_str!` で本ファイルを取り込み、
// `MTLCompileOptions`（`MathMode::Safe`・`Precise`。`crate::pipeline::compile_options`）で
// 実行時コンパイルする。`crate::paged_kv_model` が本ファイルのソース不変条件
// （境界検査・`fma`・番兵拒否・禁止構文の不在）を Linux CI でも検査する。
//
// # paged_kv_append_f32
//
// 2 次元グリッド（x = 成分 e、y = 行 r）で 1 スレッドが 1 要素を担当する。行 r の書き先は
// スロット写像 slot_mapping[r]（= block_id * block_size + pos % block_size）。重複スロットは
// ホスト検証（tensor-core の validate_paged_kv_append）で拒否済みなので書き込みは競合しない。
//
// # paged_attention_f32 と縮約順序（REQ-2）
//
// 1 query 行 (b, h, i) に 1 スレッドグループを割り当てる。
//   1. score: スレッド t が j = t, t + T, ... の score を計算する。q * scale を先に作り、
//      Dh 方向に fma で逐次累積する（CPU 参照 f32::mul_add と同順）。結果は作業バッファ
//      [B * H * L_new, s_stride] へ書く
//   2. max: thread 0 が j の論理順に逐次で求める
//   3. exp: 並列に exp(s - max)。和は thread 0 が j の論理順に逐次で足す
//   4. 加重和: スレッド d が出力成分 d（Dh が T を超えるときはストライド）を受け持ち、
//      j を論理順に逐次たどって o = fma(p, v, o)（p = w / sum）で累積する
// j 軸（系列方向）は単一の連続ループ、Dh 軸も逐次の fma なので、REQ-2 の「結合順序が
// 単一の連続 K ループと異なるカーネル」には当たらず、統一複合判定（相対誤差 1e-3 未満
// または 絶対誤差 1e-5 未満）をそのまま適用する。SIMD-group 集約による j 軸の並列リダクション・
// flash-decoding 型の split は入れない（入れると baseline 方式の対象になるため）。
//
// # REQ-8（手動境界チェック）
//
// append: r < rows・e < embed_dim・slot < pool_rows を検査し、範囲外なら書かずに抜ける。
// attention: ブロックテーブルの添字（j / block_size < max_blocks_per_seq）・ブロック ID
// （< num_blocks。番兵 0xFFFFFFFF はここで拒否）・seq_len の範囲を検査し、違反を見つけたら
// 出力へ NaN を書いて抜ける（ホストで検査済みのため通常は到達しない多層防御。黙って
// ゼロにはしない）。オフセット計算は ulong。ベクトル化ロードは使わない。

#include <metal_stdlib>
using namespace metal;

#define PAGED_NEG_INF (as_type<float>(0xFF800000u))
#define PAGED_QNAN (as_type<float>(0x7FC00000u))

kernel void paged_kv_append_f32(
    device const float* k_new [[buffer(0)]],
    device const float* v_new [[buffer(1)]],
    device float* k_pool [[buffer(2)]],
    device float* v_pool [[buffer(3)]],
    device const uint* slot_mapping [[buffer(4)]],
    constant uint& rows [[buffer(5)]],
    constant uint& embed_dim [[buffer(6)]],
    constant ulong& pool_rows [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint e = gid.x;
    uint r = gid.y;
    if (r >= rows || e >= embed_dim) {
        return;
    }
    ulong slot = (ulong)slot_mapping[r];
    if (slot >= pool_rows) {
        return;
    }
    ulong src = (ulong)r * (ulong)embed_dim + (ulong)e;
    ulong dst = slot * (ulong)embed_dim + (ulong)e;
    k_pool[dst] = k_new[src];
    v_pool[dst] = v_new[src];
}

kernel void paged_attention_f32(
    device const float* q [[buffer(0)]],
    device const float* k_pool [[buffer(1)]],
    device const float* v_pool [[buffer(2)]],
    device const uint* block_tables [[buffer(3)]],
    device const uint* seq_lens [[buffer(4)]],
    device float* scores [[buffer(5)]],
    device float* out [[buffer(6)]],
    constant uint& batch [[buffer(7)]],
    constant uint& l_new [[buffer(8)]],
    constant uint& num_heads [[buffer(9)]],
    constant uint& head_dim [[buffer(10)]],
    constant uint& embed_dim [[buffer(11)]],
    constant uint& block_size [[buffer(12)]],
    constant uint& num_blocks [[buffer(13)]],
    constant uint& max_blocks_per_seq [[buffer(14)]],
    constant uint& s_stride [[buffer(15)]],
    constant float& scale [[buffer(16)]],
    uint tg [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    threadgroup int s_bad;
    threadgroup float s_max;
    threadgroup float s_sum;

    ulong qrow = (ulong)tg;
    ulong i_ll = qrow % (ulong)l_new;
    ulong h_ll = (qrow / (ulong)l_new) % (ulong)num_heads;
    ulong b_ll = qrow / ((ulong)l_new * (ulong)num_heads);
    if (b_ll >= (ulong)batch) {
        return;
    }
    uint i = (uint)i_ll;
    uint h = (uint)h_ll;
    uint b = (uint)b_ll;

    uint seq_len = seq_lens[b];
    int s_prev = (int)seq_len - (int)l_new;
    int limit_i = s_prev + (int)i + 1;
    ulong q_off = ((ulong)b * (ulong)l_new + (ulong)i) * (ulong)embed_dim + (ulong)h * (ulong)head_dim;
    device float* score_row = scores + qrow * (ulong)s_stride;

    if (tid == 0) {
        s_bad = 0;
        s_max = PAGED_NEG_INF;
        s_sum = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // seq_len の範囲違反は全スレッドが同じ値で判定する（グループ内で一様）。
    if (s_prev < 0 || limit_i < 1 || (uint)limit_i > seq_len || (uint)limit_i > s_stride) {
        s_bad = 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint limit = (uint)max(limit_i, 0);

    // 1. score（j を論理順に、Dh 方向は fma で逐次累積）。
    if (!s_bad) {
        for (uint j = tid; j < limit; j += tg_size) {
            uint slot_idx = j / block_size;
            if (slot_idx >= max_blocks_per_seq) {
                s_bad = 1;
                continue;
            }
            uint id = block_tables[(ulong)b * (ulong)max_blocks_per_seq + (ulong)slot_idx];
            if (id >= num_blocks) {
                s_bad = 1;
                continue;
            }
            ulong k_off = (((ulong)id * (ulong)block_size + (ulong)(j % block_size)) * (ulong)embed_dim)
                + (ulong)h * (ulong)head_dim;
            float acc = 0.0f;
            for (uint d = 0; d < head_dim; ++d) {
                acc = fma(q[q_off + d] * scale, k_pool[k_off + d], acc);
            }
            score_row[j] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

    if (s_bad) {
        for (uint d = tid; d < head_dim; d += tg_size) {
            out[q_off + d] = PAGED_QNAN;
        }
        return;
    }

    // 2. max（thread 0 が論理順に逐次）。
    if (tid == 0) {
        float m = PAGED_NEG_INF;
        for (uint j = 0; j < limit; ++j) {
            float s = score_row[j];
            if (s > m) {
                m = s;
            }
        }
        s_max = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. exp は並列、和は thread 0 が論理順に逐次。
    for (uint j = tid; j < limit; j += tg_size) {
        score_row[j] = exp(score_row[j] - s_max);
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (tid == 0) {
        float sum = 0.0f;
        for (uint j = 0; j < limit; ++j) {
            sum += score_row[j];
        }
        s_sum = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. 加重和（成分 d ごとに j を論理順に逐次）。ブロックテーブルは score 段で検査済みだが、
    //    手動境界チェックを維持する。
    for (uint d = tid; d < head_dim; d += tg_size) {
        float o = 0.0f;
        bool bad = false;
        for (uint j = 0; j < limit; ++j) {
            uint slot_idx = j / block_size;
            if (slot_idx >= max_blocks_per_seq) {
                bad = true;
                break;
            }
            uint id = block_tables[(ulong)b * (ulong)max_blocks_per_seq + (ulong)slot_idx];
            if (id >= num_blocks) {
                bad = true;
                break;
            }
            float p = score_row[j] / s_sum;
            ulong v_off = (((ulong)id * (ulong)block_size + (ulong)(j % block_size)) * (ulong)embed_dim)
                + (ulong)h * (ulong)head_dim + (ulong)d;
            o = fma(p, v_pool[v_off], o);
        }
        out[q_off + d] = bad ? PAGED_QNAN : o;
    }
}
