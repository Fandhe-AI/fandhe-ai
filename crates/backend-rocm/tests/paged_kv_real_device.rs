//! ROCm 実機（MI300X・gfx942）での paged KV append／paged attention を CPU 参照と突き合わせる実機
//! parity テスト（イシュー #3443。実測は #3159 段 B）。
//!
//! - 役割: `RocmPagedKv::append`／`attention` の出力を `CpuBackendOps::paged_kv_append`／
//!   `paged_attention_forward` と `fandhe_ai_backend_cpu::compare`（REQ-2 統一複合判定。相対誤差 1e-3
//!   未満 または 絶対誤差 1e-5 未満。定数は変更しない）で判定する。append は純コピーなので bit 一致を
//!   assert する。attention は GPU の `expf` の丸めが CPU と異なりうるため bit 一致は assert せず、観測値
//!   （`bit_mismatch`）として出力するだけにする。
//! - 呼び出し文脈: CI では実行しない（`#[ignore]`）。`run_rocm_parity.sh` はクレートの `--ignored` テストを
//!   全件実行するため本ファイルも自動で含まれる。記録は `docs/perf/logs/rocm-parity-3159/README.md` の
//!   「paged KV（#3443）」節。HIPRTC での実コンパイル（`__builtin_fmaf`・`expf`）と wave64 での block 128
//!   動作もここで初めて確認される。
//! - ログ行は `ROCM_PARITY_PAGED_KV` で始まる（GEMM の `ROCM_PARITY` 行と同形式）。

use fandhe_ai_backend_cpu::{CpuBackendOps, compare};
use fandhe_ai_backend_rocm::{GfxTarget, RocmPagedKv};
use fandhe_ai_tensor_core::{BackendOps, PagedAttentionArgs, PagedKvLayout, Tensor};

fn data(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

struct Case {
    block_size: usize,
    batch: usize,
    embed: usize,
    heads: usize,
    steps: &'static [usize],
}

/// 1 件ぶんの記録行を出力し、複合判定の fail 数を返す。
fn record(name: &str, got: &[f32], want: &[f32]) -> usize {
    let r = compare(got, want).unwrap_or_else(|e| panic!("compare {name}: {e:?}"));
    let bits = got
        .iter()
        .zip(want)
        .filter(|(p, q)| p.to_bits() != q.to_bits())
        .count();
    eprintln!(
        "ROCM_PARITY_PAGED_KV\t{name}\ttotal={}\tfail={}\tmax_abs={:e}\tmax_rel={:e}\tmean_abs={:e}\tp99_abs={:e}\tbit_mismatch={bits}",
        r.total, r.fail_count, r.max_abs_diff, r.max_rel_err, r.mean_abs_diff, r.p99_abs_diff
    );
    r.fail_count
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn run_case(pk: &RocmPagedKv, c: &Case) -> usize {
    let max_tokens: usize = c.steps.iter().sum();
    let max_blocks = max_tokens.div_ceil(c.block_size);
    let num_blocks = c.batch * max_blocks + 3;
    let layout = PagedKvLayout::new(num_blocks, c.block_size, c.embed).unwrap();
    let stride = (1..num_blocks)
        .rev()
        .find(|s| gcd(*s, num_blocks) == 1)
        .unwrap_or(1);
    let id_of = |b: usize, slot: usize| ((b * max_blocks + slot) * stride % num_blocks) as u32;

    let cpu = CpuBackendOps::new();
    let mem = cpu.memory_ops().unwrap();
    let pool_shape = [num_blocks, c.block_size, c.embed];
    let mut cpu_k = mem.alloc_zeroed(&pool_shape).unwrap();
    let mut cpu_v = mem.alloc_zeroed(&pool_shape).unwrap();
    let pool_elems = layout.pool_elems().unwrap();
    let mut gpu_k = vec![0.0f32; pool_elems];
    let mut gpu_v = vec![0.0f32; pool_elems];

    let scale = 1.0 / ((c.embed / c.heads) as f32).sqrt();
    let mut seq = 0usize;
    let mut fails = 0;
    for (si, &l) in c.steps.iter().enumerate() {
        let n = c.batch * l * c.embed;
        let seed = 10 + si as u64;
        let (k, v, q) = (data(n, seed), data(n, seed + 1000), data(n, seed + 2000));
        let shape = [c.batch, l, c.embed];
        let t = |d: &[f32]| Tensor::new(d.to_vec(), &shape).unwrap();
        let mut slots = Vec::new();
        for b in 0..c.batch {
            for i in 0..l {
                let pos = seq + i;
                slots.push(
                    id_of(b, pos / c.block_size) * c.block_size as u32
                        + (pos % c.block_size) as u32,
                );
            }
        }
        seq += l;
        let used = seq.div_ceil(c.block_size);
        let mut tables = Vec::new();
        for b in 0..c.batch {
            for slot in 0..max_blocks {
                tables.push(if slot < used {
                    id_of(b, slot)
                } else {
                    u32::MAX
                });
            }
        }
        let seq_lens = vec![seq as u32; c.batch];
        let args = PagedAttentionArgs::new(layout, c.batch, l, c.heads, max_blocks, scale).unwrap();

        let (ck, cv, cq) = (
            mem.upload(&t(&k)).unwrap(),
            mem.upload(&t(&v)).unwrap(),
            mem.upload(&t(&q)).unwrap(),
        );
        cpu.paged_kv_append(&ck, &cv, &mut cpu_k, &mut cpu_v, &slots, &layout)
            .unwrap();
        let want = cpu
            .paged_attention_forward(&cq, &cpu_k, &cpu_v, &tables, &seq_lens, &args)
            .unwrap();

        pk.append(&k, &v, &mut gpu_k, &mut gpu_v, &slots, &layout)
            .expect("append が成功すること");
        let got = pk
            .attention(&q, &gpu_k, &gpu_v, &tables, &seq_lens, &args)
            .expect("attention が成功すること");

        let name = format!(
            "bs={} B={} E={} H={} step={si} L_new={l}",
            c.block_size, c.batch, c.embed, c.heads
        );
        let flat = |t: Tensor<f32>| t.contiguous().as_slice().unwrap().to_vec();
        // append は純コピーなので bit 一致する。
        assert_eq!(
            bits(&gpu_k),
            bits(&flat(mem.download(&cpu_k).unwrap())),
            "{name}: K pool"
        );
        assert_eq!(
            bits(&gpu_v),
            bits(&flat(mem.download(&cpu_v).unwrap())),
            "{name}: V pool"
        );
        fails += record(&name, &got, &flat(mem.download(&want).unwrap()));
    }
    fails
}

#[test]
#[ignore = "ROCm 実機が必要（#3159 段 B で実行）"]
fn paged_kv_append_and_attention_match_cpu_reference() {
    let pk = RocmPagedKv::new(GfxTarget::Gfx942).expect("HIPRTC コンパイル・ロードに成功すること");
    let mut fails = 0;
    for &(bs, steps) in &[
        (1usize, &[3usize, 1, 1][..]),
        (4, &[3, 1, 1, 1, 1, 5]),
        (4, &[5, 1]),
        (16, &[15, 1, 1, 17]),
        (16, &[200, 1, 1]),
    ] {
        // head_dim 2・4・16・64（block 128 未満の端数 wave を含む）。
        for &(batch, embed, heads) in &[
            (1usize, 8usize, 4usize),
            (2, 16, 4),
            (2, 64, 4),
            (2, 128, 2),
        ] {
            fails += run_case(
                &pk,
                &Case {
                    block_size: bs,
                    batch,
                    embed,
                    heads,
                    steps,
                },
            );
        }
    }
    assert_eq!(fails, 0, "統一複合判定の fail 要素数");
}
