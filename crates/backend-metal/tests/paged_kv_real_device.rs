//! `MetalBackendOps::paged_kv_append`／`paged_attention_forward`（K-3。イシュー #3111・
//! `docs/device-resident-kv-paged-attention-design.md`）の Mac 実機テスト。
//!
//! `cfg(target_os = "macos")` + 理由付き `#[ignore]`（Apple Silicon 実機が必要）。CPU 参照
//! （`CpuBackendOps`。`autodiff` の参照と同順序であることは
//! `crates/backend-cpu/tests/paged_kv_backend_parity.rs` が CI で確認済み）との一致を、
//! REQ-2 の統一複合判定（`assert_parity`。tolerance 定数は変更しない）で検証する。MSL の
//! 実コンパイルもここ（`MetalPagedKv::new`）で初めて検証される（Linux では未検証）。
//! 形状マトリクスの網羅・判定方式の構成・実行手順の整備は #3113、実測は Phase 8（#3172）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test paged_kv_real_device -- --ignored --nocapture
//! ```
#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{
    BackendOps, DeviceBuffer, DispatchFailureCell, PagedAttentionArgs, PagedKvLayout, Tensor,
};

fn data(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn tensor(b: usize, l: usize, e: usize, seed: u64) -> Tensor<f32> {
    Tensor::new(data(b * l * e, seed), &[b, l, e]).unwrap()
}

/// 1 バックエンドぶんの試行（プール・ページテーブルをホストで管理する最小の owner）。
/// ブロック ID は逆順に払い出し、物理配置が論理順と一致しないことを確認する。
struct Run<'a> {
    ops: &'a dyn BackendOps,
    layout: PagedKvLayout,
    k_pool: DeviceBuffer<f32>,
    v_pool: DeviceBuffer<f32>,
    tables: Vec<Vec<u32>>,
    next: u32,
    seq_len: usize,
    batch: usize,
    heads: usize,
}

impl<'a> Run<'a> {
    fn new(ops: &'a dyn BackendOps, layout: PagedKvLayout, batch: usize, heads: usize) -> Run<'a> {
        let mem = ops.memory_ops().expect("MemoryOps");
        let shape = [layout.num_blocks, layout.block_size, layout.embed_dim];
        Run {
            ops,
            layout,
            k_pool: mem.alloc_zeroed(&shape).unwrap(),
            v_pool: mem.alloc_zeroed(&shape).unwrap(),
            tables: vec![Vec::new(); batch],
            next: 0,
            seq_len: 0,
            batch,
            heads,
        }
    }

    /// 追記して attention を返す（ホストへ戻す）。
    fn step(&mut self, l_new: usize, seed: u64) -> Vec<f32> {
        let e = self.layout.embed_dim;
        let bs = self.layout.block_size;
        let mem = self.ops.memory_ops().unwrap();
        let (k, v, q) = (
            tensor(self.batch, l_new, e, seed),
            tensor(self.batch, l_new, e, seed + 1000),
            tensor(self.batch, l_new, e, seed + 2000),
        );
        let mut slots = Vec::new();
        for b in 0..self.batch {
            for i in 0..l_new {
                let pos = self.seq_len + i;
                if pos.is_multiple_of(bs) {
                    let id = (self.layout.num_blocks as u32 - 1) - self.next;
                    self.next += 1;
                    self.tables[b].push(id);
                }
                slots.push(self.tables[b][pos / bs] * bs as u32 + (pos % bs) as u32);
            }
        }
        self.seq_len += l_new;
        let (kd, vd) = (mem.upload(&k).unwrap(), mem.upload(&v).unwrap());
        self.ops
            .paged_kv_append(
                &kd,
                &vd,
                &mut self.k_pool,
                &mut self.v_pool,
                &slots,
                &self.layout,
            )
            .unwrap();
        let mbps = self.tables.iter().map(Vec::len).max().unwrap();
        let mut flat = Vec::new();
        for row in &self.tables {
            flat.extend_from_slice(row);
            flat.resize(flat.len() + (mbps - row.len()), u32::MAX);
        }
        let lens = vec![self.seq_len as u32; self.batch];
        let scale = 1.0 / ((e / self.heads) as f32).sqrt();
        let args = PagedAttentionArgs::new(self.layout, self.batch, l_new, self.heads, mbps, scale)
            .unwrap();
        let qd = mem.upload(&q).unwrap();
        let out = self
            .ops
            .paged_attention_forward(&qd, &self.k_pool, &self.v_pool, &flat, &lens, &args)
            .unwrap();
        mem.download(&out)
            .unwrap()
            .contiguous()
            .as_slice()
            .unwrap()
            .to_vec()
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn paged_kv_matches_cpu_reference_on_metal() {
    let metal = MetalBackendOps::new();
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(bs, batch, e, heads, steps) in &[
        (1usize, 1usize, 8usize, 2usize, &[3usize, 1, 1][..]),
        (4, 2, 16, 2, &[5, 1, 1, 4]),
        (16, 1, 64, 4, &[17, 1, 1]),
        (16, 3, 48, 3, &[16, 1, 15]),
    ] {
        let layout = PagedKvLayout::new(64, bs, e).unwrap();
        let mut c = Run::new(&metal, layout, batch, heads);
        let mut r = Run::new(&cpu, layout, batch, heads);
        for (si, &l) in steps.iter().enumerate() {
            let got = c.step(l, 7 + si as u64);
            let want = r.step(l, 7 + si as u64);
            fandhe_ai_backend_cpu::assert_parity(
                &format!("bs={bs} B={batch} E={e} H={heads} step={si} L_new={l}"),
                &got,
                &want,
            );
        }
    }
}

/// seq_len 1 の attention は softmax の重みが 1 になるので出力は V 行そのもの（スモーク）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn single_token_attention_returns_v_row_on_metal() {
    let metal = MetalBackendOps::new();
    let layout = PagedKvLayout::new(4, 2, 8).unwrap();
    let mut run = Run::new(&metal, layout, 1, 2);
    let got = run.step(1, 3);
    let want = data(8, 3 + 1000);
    fandhe_ai_backend_cpu::assert_parity("single token", &got, &want);
}

/// tracked 版（encode のみ・待たない）も同じ結果になり、健全な実行では token が立たない。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn tracked_variants_match_and_leave_token_unset_on_metal() {
    let metal = MetalBackendOps::new();
    let mem = metal.memory_ops().expect("MemoryOps");
    let layout = PagedKvLayout::new(4, 2, 8).unwrap();
    let pool = [layout.num_blocks, layout.block_size, layout.embed_dim];
    let token = DispatchFailureCell::new();
    let (k, v, q) = (tensor(1, 3, 8, 1), tensor(1, 3, 8, 2), tensor(1, 3, 8, 3));
    let (kd, vd, qd) = (
        mem.upload(&k).unwrap(),
        mem.upload(&v).unwrap(),
        mem.upload(&q).unwrap(),
    );
    let mut kp = mem.alloc_zeroed(&pool).unwrap();
    let mut vp = mem.alloc_zeroed(&pool).unwrap();
    metal
        .paged_kv_append_tracked(&kd, &vd, &mut kp, &mut vp, &[0, 1, 2], &layout, &token)
        .unwrap();
    let args = PagedAttentionArgs::new(layout, 1, 3, 2, 2, 0.5).unwrap();
    let out = metal
        .paged_attention_forward_tracked(&qd, &kp, &vp, &[0, 1], &[3], &args, &token)
        .unwrap();
    let got = mem.download(&out).unwrap();
    assert!(!token.is_set());
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cmem = cpu.memory_ops().unwrap();
    let (ckd, cvd, cqd) = (
        cmem.upload(&k).unwrap(),
        cmem.upload(&v).unwrap(),
        cmem.upload(&q).unwrap(),
    );
    let mut ckp = cmem.alloc_zeroed(&pool).unwrap();
    let mut cvp = cmem.alloc_zeroed(&pool).unwrap();
    cpu.paged_kv_append(&ckd, &cvd, &mut ckp, &mut cvp, &[0, 1, 2], &layout)
        .unwrap();
    let want = cmem
        .download(
            &cpu.paged_attention_forward(&cqd, &ckp, &cvp, &[0, 1], &[3], &args)
                .unwrap(),
        )
        .unwrap();
    fandhe_ai_backend_cpu::assert_parity(
        "tracked",
        got.contiguous().as_slice().unwrap(),
        want.contiguous().as_slice().unwrap(),
    );
}
