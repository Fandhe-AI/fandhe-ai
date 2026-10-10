//! INT8 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` の GB10 実機プローブ
//! （イシュー #2608。格上げ条件 (b) の INT8 側。`docs/int8-quant-grade-up-verification-plan.md` §3・§5）。
//!
//! # 役割と責務境界
//!
//! #2122 の sm_121 ISA プローブは INT8 MMA を `policy=accept_only`（受理段 S1〜S3 のみ）で
//! 記録済みである。本ファイルは欠けている実行段（S4 起動・S5 同期・S6 bit 一致）と
//! 単一命令ループの TOPS を、`ProbeSpec` を新規に組み立てて #2122 の共有ハーネス
//! （`sm121_isa_probe_common`）へ渡すことで記録する。#2122 の登録済み成果物
//! （`registry.rs`・`RULE.txt`・`aggregate.py`）は凍結のまま触らない。
//!
//! 判定規則・パラメータの正は `docs/perf/logs/int8-mma-probe-2608/RULE.txt`（実測前に固定）。
//! 本ファイルは量子化カーネルではなくテスト専用のプローブであり、本番経路
//! （`crates/*/src`）へは何も入れない。正本 spec の除外事項ゲートを迂回しない。
//!
//! # 新規 `unsafe` を持たない理由
//!
//! 起動の `unsafe`（SAFETY コメント付き）は共有ハーネス `runner.rs` に 1 箇所だけあり、
//! 本ファイルは `common::runner::run_pipeline`（`Launch::Plain`）を呼ぶだけである。
//! カーネルは全ロード・ストアを `LD`／`ST` マクロで境界チェックする（REQ-8）。
//!
//! # A03
//!
//! カーネルソースは `&'static str` の定数のみ。環境変数 `INT8_PROBE_ID`／
//! `INT8_PROBE_TARGET` は固定表・`TARGETS` ホワイトリストと照合するだけで、ソースや
//! NVRTC オプションへ連結しない（未知の値は panic）。
//!
//! 実行例: `INT8_PROBE_ID=int8.t1.ones INT8_PROBE_TARGET=compute_121 timeout 120
//! cargo test -p fandhe-ai-backend-cuda --all-features --release --test
//! int8_mma_probe_real_device -- --ignored --nocapture`。

#[path = "sm121_isa_probe_common/mod.rs"]
mod common;

use common::jsonl::Record;
use common::kernels_mma::MMA_S8_M16N8K32;
use common::registry::{Check, Outcome, ProbeSpec, compare_words};
use common::runner::{Cell, emit_cell, emit_env, run_control, run_pipeline, stage_archs};
use common::types::{Expect, Kind, Launch, Layout, Policy, Stage, Status, target_by_name};
use fandhe_ai_backend_cuda::CudaDevice;

// ------------------------------------------------------------ 事前登録パラメータ（RULE.txt と一致）

/// MMA 1 命令の演算数（2 × M × N × K）。
const OPS_PER_MMA: u64 = 2 * 16 * 8 * 32;
/// TOPS カーネルの独立アキュムレータ連鎖数。
const CHAINS: u32 = 4;
/// TOPS カーネルの反復数（A=B=全 1 のとき最終値 = ITERS × 32 = 2,097,152 < i32::MAX）。
const ITERS: u32 = 65536;
/// TOPS カーネルの block 数（GB10 の SM 数 48 × 8。実行時に SM 数を読まず固定する）。
const TOPS_GRID: u32 = 384;
/// TOPS カーネルが block あたり出力する語数（t_start lo/hi・t_end lo/hi・chain 0..3 の d0）。
const TOPS_WORDS_PER_BLOCK: usize = 8;
/// K=128（4 連鎖）の最悪値 127×127×128。
const K128_ABS: i32 = 127 * 127 * 128;

/// `mma.sync` 入出力 fragment の語数（lane あたり入力 10 語・出力 4 語）。
const IN_PER_LANE: usize = 10;
const OUT_PER_LANE: usize = 4;

// ------------------------------------------------------------ カーネル（K=128 連鎖・TOPS）

/// K=128 最悪値: `mma.sync` を 4 回連鎖（C←D）。a = `in[0]`・b = `in[1]` を全 lane で使用。
const MMA_S8_K128_CHAIN: &str = concat!(
    common::kernels_mma::pre!(),
    r#"
extern "C" __global__ void __launch_bounds__(32) mma_s8_k128_chain(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned a = LD(0u), b = LD(1u);
    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    #pragma unroll 1
    for (int s = 0; s < 4; ++s) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
            : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b));
    }
    ST(l * 4u, (unsigned)d0); ST(l * 4u + 1u, (unsigned)d1);
    ST(l * 4u + 2u, (unsigned)d2); ST(l * 4u + 3u, (unsigned)d3);
}
"#
);

/// TOPS: 各 warp が `CHAINS` 本の独立連鎖を `ITERS` 回回し、warp 0 の lane 0 が
/// `%globaltimer`（ns）の開始・終了と各連鎖の d0 を書く。定数は RULE.txt と一致させる。
const MMA_S8_TOPS: &str = concat!(
    common::kernels_mma::pre!(),
    r#"
#define ITERS 65536
extern "C" __global__ void __launch_bounds__(32) mma_s8_tops(
    const unsigned* __restrict__ in, int n_in, unsigned* __restrict__ out, int n)
{
    unsigned l = threadIdx.x;
    unsigned a = LD(0u), b = LD(1u);
    int x0 = 0, x1 = 0, x2 = 0, x3 = 0;
    int y0 = 0, y1 = 0, y2 = 0, y3 = 0;
    int z0 = 0, z1 = 0, z2 = 0, z3 = 0;
    int w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    #pragma unroll 1
    for (int s = 0; s < ITERS; ++s) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(x0), "+r"(x1), "+r"(x2), "+r"(x3) : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(y0), "+r"(y1), "+r"(y2), "+r"(y3) : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(z0), "+r"(z1), "+r"(z2), "+r"(z3) : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b));
        asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+r"(w0), "+r"(w1), "+r"(w2), "+r"(w3) : "r"(a), "r"(a), "r"(a), "r"(a), "r"(b), "r"(b));
    }
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    if (l == 0u) {
        unsigned o = blockIdx.x * 8u;
        ST(o, (unsigned)t0); ST(o + 1u, (unsigned)(t0 >> 32));
        ST(o + 2u, (unsigned)t1); ST(o + 3u, (unsigned)(t1 >> 32));
        ST(o + 4u, (unsigned)x0); ST(o + 5u, (unsigned)y0);
        ST(o + 6u, (unsigned)z0); ST(o + 7u, (unsigned)w0);
    }
}
"#
);

// ------------------------------------------------------------ 参照モデル（Tier 2）

/// 4 個の s8 をバイト列（下位バイトが要素 0）として 1 語へ詰める。
fn pack4(v: [i8; 4]) -> u32 {
    u32::from_le_bytes([v[0] as u8, v[1] as u8, v[2] as u8, v[3] as u8])
}

fn unpack4(w: u32) -> [i32; 4] {
    let b = w.to_le_bytes();
    [
        b[0] as i8 as i32,
        b[1] as i8 as i32,
        b[2] as i8 as i32,
        b[3] as i8 as i32,
    ]
}

/// Tier 2 の決定的パターン（RULE.txt に式を登録）。-128・127 の端値を含む。
fn pattern_a(r: usize, k: usize) -> i8 {
    match (r, k) {
        (0, 0) => -128,
        (1, 1) => 127,
        _ => (((r * 37 + k * 91 + 11) % 251) as i32 - 125) as i8,
    }
}

fn pattern_b(k: usize, c: usize) -> i8 {
    match (k, c) {
        (0, 0) => -128,
        (31, 7) => 127,
        _ => (((k * 53 + c * 29 + 7) % 247) as i32 - 123) as i8,
    }
}

fn pattern_c(r: usize, c: usize) -> i32 {
    (r as i32 * 1_000_003) - (c as i32 * 7_919) - 5_000_000
}

/// A[16×32]・B[32×8]・C[16×8] を PTX ISA の m16n8k32 `.s8` fragment 配置へ詰める
/// （lane×10 語: a0..a3, b0, b1, c0..c3）。
fn pack_fragments(
    a: &dyn Fn(usize, usize) -> i8,
    b: &dyn Fn(usize, usize) -> i8,
    c: &dyn Fn(usize, usize) -> i32,
) -> Vec<u32> {
    let mut v = vec![0u32; 32 * IN_PER_LANE];
    for lane in 0..32usize {
        let (g, t) = (lane >> 2, lane & 3);
        let base = lane * IN_PER_LANE;
        // a0: 行 g・列 t*4+i / a1: 行 g+8 / a2: 行 g・列 t*4+16+i / a3: 行 g+8
        let rows = [g, g + 8, g, g + 8];
        let col_off = [0, 0, 16, 16];
        for j in 0..4 {
            v[base + j] = pack4(std::array::from_fn(|i| a(rows[j], t * 4 + col_off[j] + i)));
        }
        // b0: k = t*4+i / b1: k = t*4+16+i、列 g
        v[base + 4] = pack4(std::array::from_fn(|i| b(t * 4 + i, g)));
        v[base + 5] = pack4(std::array::from_fn(|i| b(t * 4 + 16 + i, g)));
        // c0,c1: 行 g・列 t*2+{0,1} / c2,c3: 行 g+8
        v[base + 6] = c(g, t * 2) as u32;
        v[base + 7] = c(g, t * 2 + 1) as u32;
        v[base + 8] = c(g + 8, t * 2) as u32;
        v[base + 9] = c(g + 8, t * 2 + 1) as u32;
    }
    v
}

fn s8_m16n8k32_input() -> Vec<u32> {
    pack_fragments(&pattern_a, &pattern_b, &pattern_c)
}

/// fragment 入力語から行列を復元して素朴な i32 GEMM を行い、D を lane×4 語の fragment 配置へ戻す。
fn s8_m16n8k32_expected(input: &[u32]) -> Vec<u32> {
    let (mut a, mut b, mut c) = ([[0i32; 32]; 16], [[0i32; 8]; 32], [[0i32; 8]; 16]);
    for lane in 0..32usize {
        let (g, t) = (lane >> 2, lane & 3);
        let w = |j: usize| input.get(lane * IN_PER_LANE + j).copied().unwrap_or(0);
        let rows = [g, g + 8, g, g + 8];
        let col_off = [0, 0, 16, 16];
        for j in 0..4 {
            for (i, x) in unpack4(w(j)).into_iter().enumerate() {
                a[rows[j]][t * 4 + col_off[j] + i] = x;
            }
        }
        for (i, x) in unpack4(w(4)).into_iter().enumerate() {
            b[t * 4 + i][g] = x;
        }
        for (i, x) in unpack4(w(5)).into_iter().enumerate() {
            b[t * 4 + 16 + i][g] = x;
        }
        c[g][t * 2] = w(6) as i32;
        c[g][t * 2 + 1] = w(7) as i32;
        c[g + 8][t * 2] = w(8) as i32;
        c[g + 8][t * 2 + 1] = w(9) as i32;
    }
    let mut out = vec![0u32; 32 * OUT_PER_LANE];
    for lane in 0..32usize {
        let (g, t) = (lane >> 2, lane & 3);
        let pos = [
            (g, t * 2),
            (g, t * 2 + 1),
            (g + 8, t * 2),
            (g + 8, t * 2 + 1),
        ];
        for (j, (r, col)) in pos.into_iter().enumerate() {
            let mut acc = c[r][col];
            for k in 0..32 {
                acc = acc.wrapping_add(a[r][k].wrapping_mul(b[k][col]));
            }
            out[lane * OUT_PER_LANE + j] = acc as u32;
        }
    }
    out
}

// ------------------------------------------------------------ 入力・期待値・判定

fn ones_input() -> Vec<u32> {
    let mut v = vec![0x0101_0101u32; 32 * IN_PER_LANE];
    for lane in 0..32 {
        for j in 6..10 {
            v[lane * IN_PER_LANE + j] = 0;
        }
    }
    v
}

fn ones_expected(_input: &[u32]) -> Vec<u32> {
    vec![32u32; 32 * OUT_PER_LANE]
}

fn k128_pos_input() -> Vec<u32> {
    vec![0x7f7f_7f7f, 0x7f7f_7f7f]
}

fn k128_neg_input() -> Vec<u32> {
    vec![0x7f7f_7f7f, 0x8181_8181]
}

fn k128_pos_expected(_input: &[u32]) -> Vec<u32> {
    vec![K128_ABS as u32; 32 * OUT_PER_LANE]
}

fn k128_neg_expected(_input: &[u32]) -> Vec<u32> {
    vec![(-K128_ABS) as u32; 32 * OUT_PER_LANE]
}

fn tops_input() -> Vec<u32> {
    vec![0x0101_0101, 0x0101_0101]
}

/// TOPS プローブの判定: 全 block の 4 連鎖が `ITERS×32` に厳密一致し、時刻が単調であること。
/// TOPS は detail へ記録するだけでゲートにしない（RULE.txt）。
fn tops_check(_input: &[u32], out: &[u32]) -> Outcome {
    let expect_len = TOPS_GRID as usize * TOPS_WORDS_PER_BLOCK;
    if out.len() != expect_len {
        return Outcome::Mismatch {
            first: out.len().min(expect_len),
            count: 1,
            detail: format!("length expected={expect_len} got={}", out.len()),
        };
    }
    let want = ITERS * 32;
    let (mut min_start, mut max_end) = (u64::MAX, 0u64);
    for (b, blk) in out.chunks(TOPS_WORDS_PER_BLOCK).enumerate() {
        let t0 = u64::from(blk[0]) | (u64::from(blk[1]) << 32);
        let t1 = u64::from(blk[2]) | (u64::from(blk[3]) << 32);
        if t1 < t0 {
            return Outcome::Mismatch {
                first: b * TOPS_WORDS_PER_BLOCK,
                count: 1,
                detail: format!("block={b} globaltimer not monotonic t0={t0} t1={t1}"),
            };
        }
        if let Some(j) = blk[4..8].iter().position(|&x| x != want) {
            return Outcome::Mismatch {
                first: b * TOPS_WORDS_PER_BLOCK + 4 + j,
                count: 1,
                detail: format!(
                    "block={b} chain={j} expected=0x{want:08x} got=0x{:08x}",
                    blk[4 + j]
                ),
            };
        }
        min_start = min_start.min(t0);
        max_end = max_end.max(t1);
    }
    let elapsed_ns = max_end - min_start;
    if elapsed_ns == 0 {
        // 計時粒度で 0 になる場合は TOPS を算出しない（bit 一致のみ成立）。
        return Outcome::Match("tops=unmeasurable(elapsed_ns=0) bit_exact".to_string());
    }
    let ops = OPS_PER_MMA * u64::from(CHAINS) * u64::from(ITERS) * u64::from(TOPS_GRID);
    let tops = ops as f64 / elapsed_ns as f64 / 1e3;
    Outcome::Match(format!(
        "tops={tops:.3} elapsed_ns={elapsed_ns} ops={ops} chains={CHAINS} iters={ITERS} grid={TOPS_GRID} bit_exact"
    ))
}

// ------------------------------------------------------------ プローブ表

fn spec(
    id: &'static str,
    src: &'static str,
    symbol: &'static str,
    grid: u32,
    out_words: usize,
    make_input: fn() -> Vec<u32>,
    check: Check,
) -> ProbeSpec {
    ProbeSpec {
        id,
        ac: "INT8",
        clause: "R-INT8-MMA",
        home_arch: "sm_80",
        policy: Policy::Verify,
        layout: Layout::Unverified,
        expect: Expect::None,
        kind: Kind::Kernel,
        src,
        symbol,
        fixed_arch: None,
        block: 32,
        grid,
        cluster: 0,
        out_words,
        make_input,
        check,
        launch: Launch::Plain,
        tma: None,
    }
}

/// 事前登録した 5 プローブ（ID は RULE.txt の `PROBE:` 行と一致）。
fn probes() -> Vec<ProbeSpec> {
    let frag = 32 * OUT_PER_LANE;
    vec![
        spec(
            "int8.t1.ones",
            MMA_S8_M16N8K32,
            "mma_s8_m16n8k32",
            1,
            frag,
            ones_input,
            Check::Exact(ones_expected),
        ),
        spec(
            "int8.t2.pattern",
            MMA_S8_M16N8K32,
            "mma_s8_m16n8k32",
            1,
            frag,
            s8_m16n8k32_input,
            Check::Exact(s8_m16n8k32_expected),
        ),
        spec(
            "int8.k128.pos",
            MMA_S8_K128_CHAIN,
            "mma_s8_k128_chain",
            1,
            frag,
            k128_pos_input,
            Check::Exact(k128_pos_expected),
        ),
        spec(
            "int8.k128.neg",
            MMA_S8_K128_CHAIN,
            "mma_s8_k128_chain",
            1,
            frag,
            k128_neg_input,
            Check::Exact(k128_neg_expected),
        ),
        spec(
            "int8.tops",
            MMA_S8_TOPS,
            "mma_s8_tops",
            TOPS_GRID,
            TOPS_GRID as usize * TOPS_WORDS_PER_BLOCK,
            tops_input,
            Check::Custom(tops_check),
        ),
    ]
}

fn required_env(name: &str) -> String {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => v,
        _ => panic!("環境変数 {name} が未設定（1 プロセス 1 件を明示指定する契約）"),
    }
}

// ------------------------------------------------------------ 実機テスト（#[ignore]）

#[test]
#[ignore = "CUDA 実機（GB10。開発機スモークは compute_86）と NVRTC が必要。記録先: docs/perf/logs/int8-mma-probe-2608/"]
fn int8_mma_probe_exec_selected() {
    let id = required_env("INT8_PROBE_ID");
    let target_name = required_env("INT8_PROBE_TARGET");
    let probe = probes()
        .into_iter()
        .find(|p| p.id == id)
        .unwrap_or_else(|| panic!("未知の INT8_PROBE_ID: {id:?}"));
    let target = target_by_name(&target_name)
        .unwrap_or_else(|| panic!("未知の INT8_PROBE_TARGET: {target_name:?}"));

    let device = match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            println!("INT8_PROBE_NOTE device_init_failed: {e}");
            None
        }
    };
    emit_env("exec", probe.id, target.name, device.as_ref());

    let ctl = run_control(device.as_ref(), target);
    emit_cell("exec", probe.id, target.name, target.virt, &ctl);

    let (virt, real) = stage_archs(&probe, target);
    let mut cells = 1u64;
    let mut mismatches = 0u64;
    run_pipeline(device.as_ref(), &probe, target, &mut |cell: Cell| {
        let arch = if cell.stage == Stage::S2NvrtcCubin {
            real
        } else {
            virt
        };
        emit_cell("exec", probe.id, target.name, arch, &cell);
        cells += 1;
        if cell.status == Status::Mismatch {
            mismatches += 1;
        }
    });
    Record::new()
        .int("v", 1)
        .str("kind", "done")
        .str("phase", "exec")
        .str("probe", probe.id)
        .str("target", target.name)
        .int("cells", cells)
        .int("mismatch", mismatches)
        .emit();
    assert_eq!(
        mismatches, 0,
        "S6 で不一致を記録した（{id} on {target_name}）"
    );
}

// ------------------------------------------------------------ ホストのみのテスト（CI で実行）

#[test]
fn model_is_consistent_with_naive_gemm() {
    // pack → expected が、行列式から直接計算した素朴な GEMM と一致する（fragment 配置の往復整合）。
    let input = s8_m16n8k32_input();
    let got = s8_m16n8k32_expected(&input);
    for lane in 0..32usize {
        let (g, t) = (lane >> 2, lane & 3);
        let pos = [
            (g, t * 2),
            (g, t * 2 + 1),
            (g + 8, t * 2),
            (g + 8, t * 2 + 1),
        ];
        for (j, (r, c)) in pos.into_iter().enumerate() {
            let mut acc = pattern_c(r, c) as i64;
            for k in 0..32 {
                acc += i64::from(pattern_a(r, k)) * i64::from(pattern_b(k, c));
            }
            assert_eq!(
                got[lane * OUT_PER_LANE + j],
                acc as i32 as u32,
                "lane={lane} j={j}"
            );
        }
    }
    assert!(matches!(compare_words(&got, &got), Outcome::Match(_)));
}

#[test]
fn pattern_contains_extreme_values() {
    assert_eq!(pattern_a(0, 0), -128);
    assert_eq!(pattern_a(1, 1), 127);
    assert_eq!(pattern_b(0, 0), -128);
    assert_eq!(pattern_b(31, 7), 127);
    for r in 0..16 {
        for k in 0..32 {
            let _ = pattern_a(r, k);
        }
    }
}

#[test]
fn tier1_and_k128_expectations_are_arithmetically_exact() {
    let ones = ones_input();
    assert_eq!(ones_expected(&ones), s8_m16n8k32_expected(&ones));
    assert_eq!(K128_ABS, 2_064_512);
    // A=B=127 の 4 連鎖 = 4 × 32 × 127² と同値。
    assert_eq!(4 * 32 * 127 * 127, K128_ABS);
}

#[test]
fn registered_probe_ids_are_unique_and_verify() {
    let ps = probes();
    let mut ids: Vec<_> = ps.iter().map(|p| p.id).collect();
    ids.sort_unstable();
    ids.dedup();
    assert_eq!(ids.len(), 5);
    assert!(ps.iter().all(|p| p.policy == Policy::Verify));
}

#[test]
fn tops_check_accepts_synthetic_output() {
    let mut out = vec![0u32; TOPS_GRID as usize * TOPS_WORDS_PER_BLOCK];
    for blk in out.chunks_mut(TOPS_WORDS_PER_BLOCK) {
        blk[0] = 1_000;
        blk[2] = 1_000_000;
        blk[4..8].fill(ITERS * 32);
    }
    assert!(matches!(tops_check(&[], &out), Outcome::Match(d) if d.contains("tops=")));
    out[5] ^= 1;
    assert!(matches!(tops_check(&[], &out), Outcome::Mismatch { .. }));
}

/// verification-plan §5 境界テスト (II): K_safe = floor(i32::MAX / 127²)。
/// 本番の事前検証 assert は将来の実装イシュー側の責務で、ここは形式境界の算術登録のみ。
#[test]
fn k_safe_boundary_arithmetic() {
    let per = 127i64 * 127;
    let k_safe = i64::from(i32::MAX) / per;
    assert_eq!(k_safe, 133_144);
    assert!(k_safe * per <= i64::from(i32::MAX));
    assert!((k_safe + 1) * per > i64::from(i32::MAX));
}
