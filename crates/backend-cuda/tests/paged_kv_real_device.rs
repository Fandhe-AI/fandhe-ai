//! `CudaBackendOps::paged_kv_append`／`paged_attention_forward`（K-3。イシュー #3111・
//! `docs/device-resident-kv-paged-attention-design.md`）の実機必須テスト。
//!
//! `#[ignore]` 分離（CUDA 実機 = DGX Spark GB10 が必要）。CPU 参照（`CpuBackendOps`。
//! `autodiff` の参照 `paged_attention_reference` と同順序であることは
//! `crates/backend-cpu/tests/paged_kv_backend_parity.rs` が CI で確認済み）との一致を、
//! REQ-2 の統一複合判定（`assert_parity`。tolerance 定数は変更しない）で検証する。
//! 形状マトリクスの網羅・判定方式の構成・実行手順の整備は #3113、実測は Phase 8（#3171）。
//! なお `backend-cuda` に `autodiff` の dev-dependency は足さない（設計 §11 が保留していた
//! 判断。ここでは `CpuBackendOps` を直接の参照にするため不要）。
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test paged_kv_real_device -- --ignored --nocapture
//! ```

mod common;

use common::paged_attention_judgement::{
    CASES, PagedAttentionCase, judge, judgement_for, metric_line,
};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::{BackendOps, DeviceBuffer, PagedAttentionArgs, PagedKvLayout, Tensor};

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

fn scaled_tensor(b: usize, l: usize, e: usize, seed: u64, scale: f32) -> Tensor<f32> {
    let d: Vec<f32> = data(b * l * e, seed)
        .into_iter()
        .map(|x| x * scale)
        .collect();
    Tensor::new(d, &[b, l, e]).unwrap()
}

/// 1 バックエンドぶんの試行（プール・ページテーブルをホストで管理する最小の owner）。
/// ブロック ID は逆順に払い出し、物理配置が論理順と一致しないことを確認する。
struct Run<'a> {
    ops: &'a dyn BackendOps,
    layout: PagedKvLayout,
    k_pool: DeviceBuffer<f32>,
    v_pool: DeviceBuffer<f32>,
    tables: Vec<Vec<u32>>,
    /// 空きブロック ID（末尾から払い出す。初期は num_blocks-1 から降順＝物理配置が論理順と逆）。
    free: Vec<u32>,
    input_scale: f32,
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
            free: (0..layout.num_blocks as u32).collect(),
            input_scale: 1.0,
            seq_len: 0,
            batch,
            heads,
        }
    }

    /// 長さ `new_len` へ巻き戻す（不要になったブロックは空きへ返す。プール内容は消さず
    /// `lens` のマスクで無効化する。facade の `PagedKvCache::rewind_to` と同じ意味）。
    fn rewind_to(&mut self, new_len: usize) {
        assert!(new_len <= self.seq_len);
        let keep = new_len.div_ceil(self.layout.block_size);
        for row in &mut self.tables {
            while row.len() > keep {
                self.free.push(row.pop().unwrap());
            }
        }
        self.seq_len = new_len;
    }

    /// 追記して attention を返す（ホストへ戻す）。
    fn step(&mut self, l_new: usize, seed: u64) -> Vec<f32> {
        let e = self.layout.embed_dim;
        let bs = self.layout.block_size;
        let mem = self.ops.memory_ops().unwrap();
        let (k, v, q) = (
            scaled_tensor(self.batch, l_new, e, seed, self.input_scale),
            scaled_tensor(self.batch, l_new, e, seed + 1000, self.input_scale),
            scaled_tensor(self.batch, l_new, e, seed + 2000, self.input_scale),
        );
        let mut slots = Vec::new();
        for b in 0..self.batch {
            for i in 0..l_new {
                let pos = self.seq_len + i;
                if pos.is_multiple_of(bs) {
                    let id = self.free.pop().expect("pool exhausted");
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
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn paged_kv_matches_cpu_reference_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda = CudaBackendOps::new(device.ordinal());
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();

    for &(bs, batch, e, heads, steps) in &[
        (1usize, 1usize, 8usize, 2usize, &[3usize, 1, 1][..]),
        (4, 2, 16, 2, &[5, 1, 1, 4]),
        (16, 1, 64, 4, &[17, 1, 1]),
        (16, 3, 48, 3, &[16, 1, 15]),
    ] {
        let layout = PagedKvLayout::new(64, bs, e).unwrap();
        let mut c = Run::new(&cuda, layout, batch, heads);
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
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn single_token_attention_returns_v_row_on_real_device() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let cuda = CudaBackendOps::new(device.ordinal());
    let layout = PagedKvLayout::new(4, 2, 8).unwrap();
    let mut run = Run::new(&cuda, layout, 1, 2);
    let got = run.step(1, 3);
    let want = data(8, 3 + 1000);
    fandhe_ai_backend_cpu::assert_parity("single token", &got, &want);
}

/// 1 ケースを実機と CPU 参照で同じ手順（追記・巻き戻し）で走らせ、ステップごとに判定する。
/// fail は `failures` へ集め、呼び出し側が最後に 1 回 panic する
/// （1 回の実行で全ケースの `PAGED_PARITY` 行をログに残すため）。
fn run_case(
    backend: &str,
    ops: &dyn BackendOps,
    cpu: &dyn BackendOps,
    c: &PagedAttentionCase,
    failures: &mut Vec<String>,
) {
    let layout = PagedKvLayout::new(
        common::paged_attention_judgement::pool_blocks(c),
        c.block_size,
        c.embed,
    )
    .unwrap();
    let mut dev = Run::new(ops, layout, c.batch, c.heads);
    let mut r = Run::new(cpu, layout, c.batch, c.heads);
    dev.input_scale = c.input_scale;
    r.input_scale = c.input_scale;
    for (si, &l) in c.steps.iter().enumerate() {
        let got = dev.step(l, 7 + si as u64);
        let want = r.step(l, 7 + si as u64);
        let report = fandhe_ai_backend_cpu::compare(&got, &want).expect("length");
        let j = judgement_for(c.id, si);
        println!("{}", metric_line(backend, c.id, si, &report, j));
        if let Err(e) = judge(&format!("{backend} {} step={si}", c.id), &report, j) {
            failures.push(e);
        }
        if let Some((_, new_len)) = c.rewind.filter(|&(a, _)| a == si) {
            dev.rewind_to(new_len);
            r.rewind_to(new_len);
        }
    }
}

/// 形状マトリクス全件の CPU 参照一致（既定は統一複合判定のゼロ fail。tolerance は不変）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。docs/perf/logs/paged-attention-parity-3113/README.md 参照"]
fn cuda_paged_attention_matrix_matches_cpu_reference() {
    let device =
        CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
    let ops = CudaBackendOps::new(device.ordinal());
    let cpu = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let mut failures = Vec::new();
    for c in CASES {
        run_case("cuda", &ops, &cpu, c, &mut failures);
    }
    assert!(
        failures.is_empty(),
        "{} 件のケースが判定を満たさない:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// 実機ログ側にも tolerance 定数の固定の証跡を残す（契約テストと重複するが実機回の記録用）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。docs/perf/logs/paged-attention-parity-3113/README.md 参照"]
fn cuda_paged_attention_tolerance_constants_pinned() {
    assert_eq!(fandhe_ai_backend_cpu::RELATIVE_TOLERANCE, 1e-3);
    assert_eq!(fandhe_ai_backend_cpu::ABSOLUTE_RESCUE_THRESHOLD, 1e-5);
}

/// 実機なしで動く自己検査: マトリクス駆動部（`Run`・巻き戻し・ブロック払い出し・各ケースの
/// 引数検証）が CPU 同士で破綻せず、判定が Strict で通ること。実機テストの配線ミスを CI で拾う。
#[test]
fn matrix_harness_runs_on_cpu_against_cpu() {
    let cpu_a = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_b = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let mut failures = Vec::new();
    for c in CASES {
        run_case("cpu-selfcheck", &cpu_a, &cpu_b, c, &mut failures);
    }
    assert!(failures.is_empty(), "{failures:?}");
}
