//! M4 無印（GPU 10 コア・Mac mini）厳密一致表の実機検証（イシュー #3206）。
//!
//! 対象: `tile::VerifiedTileTableDevice::M4Base` の表（`tile.rs` の `select_impl`）。
//! 本番 f32 入口（`MetalGemm::dispatch_auto`）が Mini で表を引くこと（AC1）、
//! 追加前の選択（`CANDIDATES[3]` 強制）との bit 一致または REQ-2 複合判定（AC3）、
//! 追加前後の A/B 時間（AC4。記録のみ）を出す。
//! すべて Metal 実機依存の `#[ignore]` テストで CI では実行しない。型検査は
//! `make check-cross-metal-tests` が担う。tolerance・baseline は不変。
//! 対象機体（10 コア・"Apple M4"）でなければ `SKIP not-m4-base` を出して return する
//! （実行側は `MINI_TABLE_ACTIVE` 行の存在を必須にして偽 pass を防ぐ）。

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::device::{probe_gpu_core_count, probe_soc_brand_string};
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::tile::{CANDIDATES, GemmRoute, TileConfig, VerifiedTileTableDevice};

const SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];

/// 期待する本番選択（N=256 は既定の `[3]`）。
fn expected_idx(n: usize) -> usize {
    match n {
        512 | 1024 => 1,
        2048 | 4096 => 2,
        _ => 3,
    }
}

fn bits_hash_fnv1a64(values: &[f32]) -> u64 {
    values.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, v| {
        v.to_bits().to_le_bytes().iter().fold(h, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    })
}

fn run_tiled(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    a: &[f32],
    b: &[f32],
    n: usize,
    cfg: TileConfig,
) -> Vec<f32> {
    let a_buf = MetalBuffer::new_with_data(ctx, a).expect("A アップロードに失敗した");
    let b_buf = MetalBuffer::new_with_data(ctx, b).expect("B アップロードに失敗した");
    let c_buf = MetalBuffer::new_zeroed(ctx, n * n).expect("C 確保に失敗した");
    gemm.dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, n, n, n, cfg)
        .expect("dispatch_tiled_prepared に失敗した");
    c_buf.read_to_vec()
}

/// 対象機体なら true。probe が (10, "Apple M4") なのに token が `M4Base`
/// でなければ fail-closed で panic する。
fn active(ctx: &MetalContext) -> bool {
    let cores = probe_gpu_core_count();
    let brand = probe_soc_brand_string();
    let dev = ctx.verified_tile_table_device();
    let kind = match dev {
        Some(VerifiedTileTableDevice::M4Max(_)) => "M4Max",
        Some(VerifiedTileTableDevice::M4Base(_)) => "M4Base",
        None => "none",
    };
    println!(
        "DEVICE gpu_cores={} soc=\"{}\" token={kind}",
        cores.map_or_else(|| "none".to_string(), |v| v.to_string()),
        brand.clone().unwrap_or_else(|| "none".to_string()),
    );
    let is_base = cores == Some(10) && brand.as_deref() == Some("Apple M4");
    if !is_base {
        println!("SKIP not-m4-base");
        return false;
    }
    assert!(
        matches!(dev, Some(VerifiedTileTableDevice::M4Base(_))),
        "10 コア・Apple M4 なのに M4Base token が得られない"
    );
    println!("MINI_TABLE_ACTIVE");
    true
}

#[test]
#[ignore = "Metal 実機（M4 無印 10 コア）依存。CI では実行しない"]
fn m4_base_table_selected_on_macmini() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    if !active(&ctx) {
        return;
    }
    let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    for n in SIZES {
        let (a, b) = gen_square_ab(0x3206_a000 ^ (n as u64), n);
        let (_out, route) = gemm
            .dispatch_auto_with_route(&ctx, &a, &b, n, n, n)
            .expect("dispatch_auto_with_route に失敗した");
        let want = CANDIDATES[expected_idx(n)];
        assert_eq!(route, GemmRoute::Classic(want), "N={n}");
        let resolved = gemm
            .resolve_tile_config(&ctx, want)
            .expect("解決に失敗した");
        assert_eq!(resolved, want, "N={n} fallback した");
        println!(
            "SELECT N={n} idx={} fallback=false ok=true",
            expected_idx(n)
        );
    }
}

#[test]
#[ignore = "Metal 実機（M4 無印 10 コア）依存。CI では実行しない"]
fn m4_base_table_bit_match_and_parity() {
    use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    if !active(&ctx) {
        return;
    }
    let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    for n in SIZES {
        let (a, b) = gen_square_ab(0x3206_a000 ^ (n as u64), n);
        let (after, _) = gemm
            .dispatch_auto_with_route(&ctx, &a, &b, n, n, n)
            .expect("dispatch_auto_with_route に失敗した");
        let before = run_tiled(&ctx, &gemm, &a, &b, n, CANDIDATES[3]);
        let bit = after.len() == before.len()
            && after
                .iter()
                .zip(&before)
                .all(|(x, y)| x.to_bits() == y.to_bits());
        let mut expected = vec![0.0f32; n * n];
        matmul_reference_fma(&a, &b, &mut expected, n, n, n).expect("CPU 参照の形状検証に失敗した");
        assert_parity(&format!("m4base after N={n}"), &after, &expected);
        assert_parity(&format!("m4base before N={n}"), &before, &expected);
        println!(
            "BITMATCH N={n} bit_identical={bit} hash_after={:016x} hash_before={:016x} parity_ok=true",
            bits_hash_fnv1a64(&after),
            bits_hash_fnv1a64(&before)
        );
    }
}

#[test]
#[ignore = "Metal 実機（M4 無印 10 コア）依存。記録のみ"]
fn m4_base_table_ab_timing() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    if !active(&ctx) {
        return;
    }
    let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    for n in SIZES {
        let (a, b) = gen_square_ab(0x3206_a000 ^ (n as u64), n);
        let arms = [CANDIDATES[3], CANDIDATES[expected_idx(n)]]; // 0=before 1=after
        let mut keep: Vec<Vec<Vec<f32>>> = vec![Vec::new(), Vec::new()];
        for _ in 0..WARMUP_TRIALS {
            for i in 0..2 {
                let _ = measure_one_phase_trial(&ctx, &gemm, &a, &b, n, arms[i], &mut keep[i]);
                keep[i].clear();
            }
        }
        let mut kg: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let mut wall: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        for trial in 0..MEASURED_TRIALS {
            for step in 0..2 {
                let i = (trial + step) % 2;
                let s = measure_one_phase_trial(&ctx, &gemm, &a, &b, n, arms[i], &mut keep[i]);
                keep[i].clear();
                kg[i].push(s.kernel_gpu_secs);
                wall[i].push(s.alloc_c_secs + s.encode_secs + s.commit_wait_secs);
            }
        }
        for (i, arm) in ["before", "after"].iter().enumerate() {
            println!(
                "AB N={n} arm={arm} kernel_gpu_median_ms={:.4} wall_sum_median_ms={:.4}",
                median_of(&kg[i]).median * 1e3,
                median_of(&wall[i]).median * 1e3
            );
        }
    }
}
