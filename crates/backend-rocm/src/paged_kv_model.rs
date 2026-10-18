//! paged KV 2 カーネルの CPU 上の逐語モデル（テスト専用。イシュー #3443・分解案 R12）。
//!
//! 役割: ROCm 非搭載の CI で `ops/paged_kv.rs` の結線（検証順序・ハンドル分岐・出力組み立て）と、
//! `kernels_paged_kv.rs` の添字・演算順（grid-stride の全スレッド列挙、1 block = 1 query 行、thread 0 の
//! 逐次 max／和、成分ごとの論理順加重和）を CPU 参照と照合するための模擬。カーネルと同じ手動境界
//! チェックを持ち、違反時は出力へ quiet NaN を書く（ops 経由では tensor-core の検証が先に拒否するため、
//! この経路はモデル単体のテストで確認する）。
//!
//! 演算は `(q * scale).mul_add(k, acc)`・`f32::exp`・`p.mul_add(v, o)`（カーネルの `__builtin_fmaf` と
//! `expf` に対応）。GPU の `expf` との bit 一致は主張せず、実機 parity は #3159。

use crate::kernels_paged_kv::PAGED_ATTENTION_BLOCK_DIM;
use crate::launch::INDEX_OPS_BLOCK_DIM;
use crate::paged_kv::AttentionLaunchParams;

/// `paged_kv_append_f32` の逐語モデル。`grid_blocks × INDEX_OPS_BLOCK_DIM` の全スレッドを列挙し、
/// grid-stride で回す。
#[allow(clippy::too_many_arguments)]
pub(crate) fn append(
    k_new: &[f32],
    v_new: &[f32],
    k_pool: &mut [f32],
    v_pool: &mut [f32],
    slot_mapping: &[u32],
    rows: i32,
    embed_dim: i32,
    pool_rows: i64,
    grid_blocks: u32,
) {
    let total = i64::from(rows) * i64::from(embed_dim);
    let block = i64::from(INDEX_OPS_BLOCK_DIM);
    let stride = i64::from(grid_blocks) * block;
    for blk in 0..i64::from(grid_blocks) {
        for t in 0..block {
            let mut idx = blk * block + t;
            while idx < total {
                let r = idx / i64::from(embed_dim);
                let e = idx - r * i64::from(embed_dim);
                if r < i64::from(rows) && e < i64::from(embed_dim) {
                    let slot = i64::from(slot_mapping[r as usize]);
                    if slot < pool_rows {
                        let dst = (slot * i64::from(embed_dim) + e) as usize;
                        k_pool[dst] = k_new[idx as usize];
                        v_pool[dst] = v_new[idx as usize];
                    }
                }
                idx += stride;
            }
        }
    }
}

/// `paged_attention_f32` の逐語モデル。`p.grid_blocks` 個の block を順に走らせ、出力 `[B·L_new·E]` を返す。
pub(crate) fn attention(
    q: &[f32],
    k_pool: &[f32],
    v_pool: &[f32],
    block_tables: &[u32],
    seq_lens: &[u32],
    p: &AttentionLaunchParams,
) -> Vec<f32> {
    let s_stride = p.s_stride as i64;
    let mut scores = vec![0.0f32; p.grid_blocks as usize * p.s_stride as usize];
    let mut out = vec![0.0f32; p.batch as usize * p.l_new as usize * p.embed_dim as usize];
    let threads = PAGED_ATTENTION_BLOCK_DIM as i64;
    let qnan = f32::from_bits(0x7fc0_0000);
    let (l_new, num_heads) = (i64::from(p.l_new), i64::from(p.num_heads));
    let (embed, head_dim) = (i64::from(p.embed_dim), i64::from(p.head_dim));
    let bs = i64::from(p.block_size);

    for qrow in 0..i64::from(p.grid_blocks) {
        let i = qrow % l_new;
        let h = (qrow / l_new) % num_heads;
        let b = qrow / (l_new * num_heads);
        if b >= i64::from(p.batch) {
            continue;
        }
        let seq_len = seq_lens[b as usize] as i32;
        let s_prev = seq_len - p.l_new;
        let limit = s_prev + i as i32 + 1;
        let q_off = ((b * l_new + i) * embed + h * head_dim) as usize;
        let row_base = (qrow * s_stride) as usize;
        let mut s_bad = s_prev < 0 || limit > seq_len || i64::from(limit) > s_stride;

        // 1. score（スレッド t が j = t, t+T, … を担当）。
        let mut bad_any = false;
        if !s_bad {
            for t in 0..threads {
                let mut j = t as i32;
                while j < limit {
                    let slot_idx = j / p.block_size;
                    if slot_idx >= p.max_blocks_per_seq {
                        bad_any = true;
                        j += threads as i32;
                        continue;
                    }
                    let id = block_tables
                        [(b * i64::from(p.max_blocks_per_seq)) as usize + slot_idx as usize];
                    if i64::from(id) >= p.num_blocks {
                        bad_any = true;
                        j += threads as i32;
                        continue;
                    }
                    let k_off = ((i64::from(id) * bs + i64::from(j % p.block_size)) * embed
                        + h * head_dim) as usize;
                    let mut acc = 0.0f32;
                    for d in 0..head_dim as usize {
                        acc = (q[q_off + d] * p.scale).mul_add(k_pool[k_off + d], acc);
                    }
                    scores[row_base + j as usize] = acc;
                    j += threads as i32;
                }
            }
            if bad_any {
                s_bad = true;
            }
        }
        if s_bad {
            for d in 0..head_dim as usize {
                out[q_off + d] = qnan;
            }
            continue;
        }

        // 2. max（thread 0 が論理順に逐次）。
        let mut m = f32::NEG_INFINITY;
        for j in 0..limit as usize {
            let s = scores[row_base + j];
            if s > m {
                m = s;
            }
        }
        // 3. exp（並列）→ 和（thread 0 が論理順に逐次）。
        for j in 0..limit as usize {
            scores[row_base + j] = (scores[row_base + j] - m).exp();
        }
        let mut sum = 0.0f32;
        for j in 0..limit as usize {
            sum += scores[row_base + j];
        }
        // 4. 加重和（成分 d ごとに j を論理順）。
        for d in 0..head_dim as usize {
            let mut o = 0.0f32;
            let mut bad = false;
            for j in 0..limit {
                let slot_idx = j / p.block_size;
                if slot_idx >= p.max_blocks_per_seq {
                    bad = true;
                    break;
                }
                let id = block_tables
                    [(b * i64::from(p.max_blocks_per_seq)) as usize + slot_idx as usize];
                if i64::from(id) >= p.num_blocks {
                    bad = true;
                    break;
                }
                let pj = scores[row_base + j as usize] / sum;
                let v_off = ((i64::from(id) * bs + i64::from(j % p.block_size)) * embed
                    + h * head_dim) as usize
                    + d;
                o = pj.mul_add(v_pool[v_off], o);
            }
            out[q_off + d] = if bad { qnan } else { o };
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_kv::attention_launch_params;
    use fandhe_ai_tensor_core::{PagedAttentionArgs, PagedAttentionPlan, PagedKvLayout};

    fn setup() -> (PagedAttentionArgs, PagedAttentionPlan) {
        // num_blocks=4, block_size=2, E=4, H=2(Dh=2), B=1, L_new=1, max_blocks_per_seq=3。
        let layout = PagedKvLayout::new(4, 2, 4).unwrap();
        let args = PagedAttentionArgs::new(layout, 1, 1, 2, 3, 0.5).unwrap();
        let plan = PagedAttentionPlan {
            out_numel: 4,
            s_max: 4,
            query_rows: 2,
            workspace_elems: 8,
        };
        (args, plan)
    }

    fn pools() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = 4 * 2 * 4;
        let f = |s: f32| {
            (0..n)
                .map(|i| (i as f32) * 0.05 * s - 0.3)
                .collect::<Vec<_>>()
        };
        (vec![0.25, -0.5, 0.75, 1.0], f(1.0), f(-0.7))
    }

    fn is_all_nan(v: &[f32]) -> bool {
        v.iter().all(|x| x.is_nan())
    }

    #[test]
    fn valid_tables_give_finite_output() {
        let (args, plan) = setup();
        let p = attention_launch_params(&args, &plan).unwrap();
        let (q, k, v) = pools();
        let out = attention(&q, &k, &v, &[0, 1, u32::MAX], &[4], &p);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn block_id_violations_write_nan() {
        let (args, plan) = setup();
        let p = attention_launch_params(&args, &plan).unwrap();
        let (q, k, v) = pools();
        // id >= num_blocks と番兵 u32::MAX が使用範囲（seq_len=4 → 2 枠）にある。
        for tables in [[0u32, 4, 0], [0, u32::MAX, 0], [u32::MAX, 0, 0]] {
            let out = attention(&q, &k, &v, &tables, &[4], &p);
            assert!(is_all_nan(&out), "{tables:?}");
        }
    }

    #[test]
    fn seq_len_violations_write_nan() {
        let (args, mut plan) = setup();
        let (q, k, v) = pools();
        // seq_len < L_new（s_prev < 0）。
        let p = attention_launch_params(&args, &plan).unwrap();
        assert!(is_all_nan(&attention(&q, &k, &v, &[0, 1, 2], &[0], &p)));
        // limit > s_stride（s_stride を小さくして作る）。
        plan.s_max = 3;
        let p = attention_launch_params(&args, &plan).unwrap();
        assert!(is_all_nan(&attention(&q, &k, &v, &[0, 1, 2], &[4], &p)));
        // max_blocks_per_seq 超過（seq_len=6 → 3 枠目まで必要だが 2 枠のテーブルに縮める）。
        let mut small = args;
        small.max_blocks_per_seq = 1;
        let plan = PagedAttentionPlan { s_max: 6, ..plan };
        let p = attention_launch_params(&small, &plan).unwrap();
        assert!(is_all_nan(&attention(&q, &k, &v, &[0], &[6], &p)));
    }

    #[test]
    fn append_skips_out_of_range_slot_and_writes_in_range() {
        let rows = 2;
        let e = 3;
        let k_new: Vec<f32> = (0..6).map(|i| i as f32 + 1.0).collect();
        let v_new: Vec<f32> = (0..6).map(|i| -(i as f32) - 1.0).collect();
        let mut kp = vec![0.0f32; 4 * e];
        let mut vp = vec![0.0f32; 4 * e];
        // 行 0 -> スロット 3（有効）、行 1 -> スロット 4（pool_rows=4 以上で無視）。
        append(
            &k_new,
            &v_new,
            &mut kp,
            &mut vp,
            &[3, 4],
            rows,
            e as i32,
            4,
            1,
        );
        assert_eq!(&kp[9..12], &k_new[0..3]);
        assert_eq!(&vp[9..12], &v_new[0..3]);
        assert!(kp[..9].iter().all(|x| *x == 0.0));
    }
}
