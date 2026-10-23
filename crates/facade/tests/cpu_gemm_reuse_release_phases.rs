//! イシュー #3226: 実機（GB10／Mac mini）用の record-only 診断（判定外・minflt 補助）。
//!
//! 腕 fresh／reuse／reuse_release／reuse_release_arena の 1 反復あたり中央値時間と minflt を
//! `RELEASE_DIAG_JSON` 行で出す。判定は `docs/perf/logs/cpu-gemm-reuse-release-3226/RULE.txt`
//! に従う A/B 集計で行い、本テストは何も assert しない（`#[ignore]`）。
//! 正しさ（bit 一致）は `cpu_gemm_reuse_release_diag.rs` が CI で固定する。
//! `GEMM_REUSE_MINFLT=1` の別 run でのみ minflt を読む（/proc 読み出しが計時を歪めるため）。

use std::time::Instant;

use fandhe_ai::Tensor;
use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::eager_release::override_enabled_for_scope as release_override;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::alloc::{
    clear_thread_arena, override_enabled_for_scope as arena_override,
};

const WARMUP: usize = 20;
const MEASURE: usize = 20;

fn new_tape() -> Tape {
    Tape::new_with_ops(Box::new(CpuBackendOps::new()))
}

/// 決定的な擬似乱数（xorshift64*）。
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

fn minflt() -> Option<u64> {
    if !std::env::var("GEMM_REUSE_MINFLT").is_ok_and(|v| v == "1") {
        return None;
    }
    let s = std::fs::read_to_string("/proc/self/stat").ok()?;
    // comm は空白・括弧を含みうるため最後の ')' 以降で分割する。
    s.rsplit_once(')')?
        .1
        .split_whitespace()
        .nth(7)?
        .parse()
        .ok()
}

fn checksum(v: &[f32]) -> f64 {
    v.iter().map(|&x| f64::from(x)).sum()
}

#[test]
#[ignore = "実機（GB10／Mac mini）用の record-only 診断"]
fn cpu_gemm_reuse_release_phases() {
    for n in [256usize, 512, 1024, 2048] {
        let a = Tensor::new(fill(0xA11CE, n * n), &[n, n]).expect("shape");
        let b = Tensor::new(fill(0xB0B, n * n), &[n, n]).expect("shape");
        for (arm, reuse, release, arena) in [
            ("fresh", false, false, false),
            ("reuse", true, false, false),
            ("reuse_release", true, true, false),
            ("reuse_release_arena", true, true, true),
        ] {
            let _r = release_override(release);
            let _a = arena_override(arena);
            clear_thread_arena();
            let shared = new_tape();
            let (sa, sb) = (shared.var(&a), shared.var(&b));
            let (mut secs, mut flts, mut sum) = (Vec::new(), Vec::new(), 0.0f64);
            for i in 0..WARMUP + MEASURE {
                let f0 = minflt();
                let t = Instant::now();
                let s = if reuse {
                    let c = sa.matmul(&sb).expect("matmul");
                    checksum(&c.host_view())
                } else {
                    let tape = new_tape();
                    let c = tape.var(&a).matmul(&tape.var(&b)).expect("matmul");
                    checksum(&c.host_view())
                };
                let dt = t.elapsed().as_secs_f64();
                if i >= WARMUP {
                    secs.push(dt);
                    if let (Some(x), Some(y)) = (f0, minflt()) {
                        flts.push(y.saturating_sub(x));
                    }
                    sum = s;
                }
            }
            secs.sort_by(|x, y| x.total_cmp(y));
            let med = secs[secs.len() / 2];
            let flt = if flts.is_empty() {
                -1.0
            } else {
                flts.iter().sum::<u64>() as f64 / flts.len() as f64
            };
            println!(
                "RELEASE_DIAG_JSON {{\"arm\":\"{arm}\",\"n\":{n},\"median_s\":{med:.9},\"minflt_per_iter\":{flt:.2},\"checksum_bits\":{}}}",
                sum.to_bits()
            );
        }
    }
}
