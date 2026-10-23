//! イシュー #3226: reuse モードの eager matmul 結果の自動解放（`autodiff::eager_release`・
//! 既定 OFF のゲート）を `CpuBackendOps` で検証する facade 側の結線テスト。
//!
//! - 解放 ON／OFF × host arena ON／OFF の 4 通りで、reuse ループ（同じ `Tape` で matmul を
//!   繰り返す）の全反復の出力が fresh（反復ごとに `Tape` を作り直す公開経路）と bit 一致する
//! - arena に NaN の番兵を詰めたバッファを返却しても、次の出力に残値が混ざらない
//!   （`take_zeroed_f32` の全要素ゼロ埋め契約。arena の hit が 1 以上で再利用経路を通ったことも確認）
//! - 解放済みノードを経由する backward・create_graph の結果がゲート OFF と bit 一致し、
//!   create_graph が `Ok` のままであること
//!
//! 判定規則・実機 A/B は `docs/perf/logs/cpu-gemm-reuse-release-3226/RULE.txt`。
//! 本番コード・tolerance・baseline は変更しない。

use fandhe_ai::Tensor;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::eager_release::override_enabled_for_scope as release_override;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::alloc::{
    clear_thread_arena, override_enabled_for_scope as arena_override, recycle_f32, reset_stats,
    stats, take_zeroed_f32,
};

const N: usize = 64;
const ITERS: usize = 40;

fn new_tape() -> Tape {
    Tape::new_with_ops(Box::new(CpuBackendOps::new()))
}

/// 決定的な擬似乱数（xorshift64*）。bench-harness に依存しない。
fn fill(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((r >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn mat(seed: u64) -> Tensor<f32> {
    Tensor::new(fill(seed, N * N), &[N, N]).expect("形状とデータ長は一致させている")
}

fn bits(t: &[f32]) -> Vec<u32> {
    t.iter().map(|x| x.to_bits()).collect()
}

/// fresh（公開経路）: 反復ごとに Tape を作り直した出力。
fn fresh_bits(a: &Tensor<f32>, b: &Tensor<f32>) -> Vec<u32> {
    let tape = new_tape();
    let c = tape.var(a).matmul(&tape.var(b)).expect("matmul");
    bits(&c.host_view())
}

#[test]
fn reuse_release_matches_fresh_bit_exact_all_gate_combinations() {
    let (a, b) = (mat(0xA11CE), mat(0xB0B));
    let expect = fresh_bits(&a, &b);
    for release in [false, true] {
        for arena in [false, true] {
            let _r = release_override(release);
            let _a = arena_override(arena);
            clear_thread_arena();
            let tape = new_tape();
            let (va, vb) = (tape.var(&a), tape.var(&b));
            for i in 0..ITERS {
                let c = va.matmul(&vb).expect("matmul");
                assert_eq!(
                    bits(&c.host_view()),
                    expect,
                    "release={release} arena={arena} iter={i}"
                );
            }
        }
    }
}

#[test]
fn recycled_buffer_residue_does_not_leak_into_results() {
    let (a, b) = (mat(0xA11CE), mat(0xB0B));
    let expect = fresh_bits(&a, &b);
    let _r = release_override(true);
    let _a = arena_override(true);
    clear_thread_arena();
    reset_stats();
    let tape = new_tape();
    let (va, vb) = (tape.var(&a), tape.var(&b));
    for i in 0..ITERS {
        // 出力と同じ長さのバッファを NaN で汚して arena へ返す。次の出力確保が
        // 再利用しても全要素ゼロ埋めされ、NaN が結果へ混ざらないこと。
        let mut v = take_zeroed_f32(N * N);
        assert!(v.iter().all(|x| x.to_bits() == 0), "iter={i}: ゼロ埋め契約");
        v.fill(f32::NAN);
        recycle_f32(v);
        let c = va.matmul(&vb).expect("matmul");
        assert_eq!(bits(&c.host_view()), expect, "iter={i}");
    }
    assert!(stats().hit >= 1, "arena の再利用経路を通っていない");
}

/// c1 は次の matmul の前に解放され（consumer なし）、後から loss で読まれる。
fn grads_and_values(release: bool) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (a, b) = (mat(1), mat(2));
    let _r = release_override(release);
    let tape = new_tape();
    let (va, vb) = (tape.var(&a), tape.var(&b));
    let c1 = va.matmul(&vb).expect("matmul");
    let c2 = va.matmul(&vb).expect("matmul");
    let loss = c1
        .sum(None)
        .expect("sum")
        .add(&c2.sum(None).expect("sum"))
        .expect("add");
    let c1_bits = bits(&c1.host_view());
    let g = tape.backward(&loss).expect("backward");
    let ga = bits(&g.get(&va).expect("get").expect("ga").host_slice());
    let gb = bits(&g.get(&vb).expect("get").expect("gb").host_slice());
    (c1_bits, ga, gb)
}

#[test]
fn backward_through_released_node_is_bit_identical() {
    assert_eq!(grads_and_values(false), grads_and_values(true));
}

#[test]
fn create_graph_through_released_node_stays_ok_and_bit_identical() {
    let run = |release: bool| {
        let (a, b) = (mat(3), mat(4));
        let _r = release_override(release);
        let tape = new_tape();
        let child = new_tape();
        let (va, vb) = (tape.var(&a), tape.var(&b));
        let c1 = va.matmul(&vb).expect("matmul");
        let c2 = va.matmul(&vb).expect("matmul");
        let loss = c1
            .sum(None)
            .expect("sum")
            .add(&c2.sum(None).expect("sum"))
            .expect("add");
        let cg = tape
            .backward_create_graph(&loss, &child)
            .expect("create_graph は解放済みノードを含んでも Ok のまま");
        bits(&cg.grad(&va).expect("grad").expect("ga").host_view())
    };
    assert_eq!(run(false), run(true));
}

#[test]
fn release_coexists_with_checkpoint_and_values_match() {
    let run = |release: bool| {
        let (a, b) = (mat(5), mat(6));
        let _r = release_override(release);
        let tape = new_tape();
        let (va, vb) = (tape.var(&a), tape.var(&b));
        let h = tape
            .checkpoint(|| {
                let t = va.matmul(&vb)?;
                Ok(t.sigmoid())
            })
            .expect("checkpoint");
        let extra = va.matmul(&vb).expect("matmul");
        let loss = h
            .sum(None)
            .expect("sum")
            .add(&extra.sum(None).expect("sum"))
            .expect("add");
        let g = tape.backward(&loss).expect("backward");
        bits(&g.get(&va).expect("get").expect("ga").host_slice())
    };
    assert_eq!(run(false), run(true));
}
