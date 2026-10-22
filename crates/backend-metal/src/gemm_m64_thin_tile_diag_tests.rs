//! イシュー #3208: m=64 級 GEMM 向け細タイル候補（`tile::M64_THIN_CANDIDATES`）の
//! 実機 bit 一致テスト（Metal 実機依存のため全件 `#[ignore]`。CI では実行しない）。
//!
//! 役割: 候補 T が現行 classic 構成（ゲート OFF の `select_for_verified_device`）と
//! K ループの結合順序まで同一で、出力が bit 一致することを fail-closed に確かめる。
//! 非 NaN は `f32::to_bits` 完全一致、NaN は NaN 同士のクラス一致。統一複合判定
//! （tolerance）は使わず、tolerance・baseline・FMA 契約は変えない。
//! 1 件でも不一致の候補は型 B（結線しない）へ移す判断材料になる
//! （`docs/perf/metal-gemm-m64-thin-tile-candidates.md`）。性能は計測しない（#3209）。
//!
//! - NN: `MetalGemm::dispatch_variant(SimdgroupTiled(T))` 対 `SimdgroupTiled(cur)`
//! - NT／TN: `dispatch_strided_tiled_prepared(cfg = T)` 対 `cfg = cur`（8 の倍数形状のみ）
//!
//! 実行: `cargo test -p fandhe-ai-backend-metal --release --lib
//! gemm_m64_thin_tile_diag_tests -- --ignored --nocapture --test-threads=1`

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::gemm::{GemmVariant, MetalGemm};
use crate::layout::classify_2d;
use crate::tile::{self, M64_THIN_CANDIDATES, TileConfig};

/// 決定的な xorshift64*（外部依存を増やさない）。極端値を数か所混ぜる。
fn fill_det(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed | 1;
    let mut v: Vec<f32> = (0..len)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            let r = s.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((r >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let extremes = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1.0e30,
        -1.0e-30,
        3.0e38,
    ];
    if len > 0 {
        for (i, e) in extremes.iter().enumerate() {
            let idx = (i * 7919 + seed as usize) % len;
            v[idx] = *e;
        }
    }
    v
}

fn bit_equal(a: &[f32], b: &[f32]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("len {} != {}", a.len(), b.len()));
    }
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let ok = if x.is_nan() || y.is_nan() {
            x.is_nan() && y.is_nan()
        } else {
            x.to_bits() == y.to_bits()
        };
        if !ok {
            return Err(format!("idx {i}: {x:?} vs {y:?}"));
        }
    }
    Ok(())
}

/// ゲート OFF の本番 classic 構成（比較基準）。候補と同一なら比較が自明になる
/// ため、その場合は別の基準（`CANDIDATES[3]`）へ切り替える。
fn current_cfg(ctx: &MetalContext, m: usize, n: usize, k: usize, t: TileConfig) -> TileConfig {
    let cur = tile::select_for_verified_device(m, n, k, ctx.verified_tile_table_device());
    if cur == t { tile::CANDIDATES[3] } else { cur }
}

const NN_SHAPES: &[(usize, usize, usize)] = &[
    (64, 256, 784),
    (64, 10, 256),
    (784, 64, 256),
    (256, 64, 10),
    (784, 256, 64),
    (256, 10, 64),
    (64, 250, 780),
    (64, 200, 300),
    (784, 60, 250),
    (72, 129, 67),
    (64, 130, 100),
];

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn m64_thin_candidates_nn_bit_identical_to_current() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM 構築に失敗した");
    let mut checked = 0usize;
    for &(m, n, k) in NN_SHAPES {
        let a = fill_det(11 + m as u64, m * k);
        let b = fill_det(23 + n as u64, k * n);
        for t in M64_THIN_CANDIDATES {
            let cur = current_cfg(&ctx, m, n, k, *t);
            let got = gemm
                .dispatch_variant(&ctx, GemmVariant::SimdgroupTiled(*t), &a, &b, m, n, k)
                .expect("候補の dispatch に失敗した");
            let want = gemm
                .dispatch_variant(&ctx, GemmVariant::SimdgroupTiled(cur), &a, &b, m, n, k)
                .expect("基準の dispatch に失敗した");
            if let Err(e) = bit_equal(&got, &want) {
                panic!("NN {m}x{n}x{k} {t:?} vs {cur:?}: {e}");
            }
            checked += 1;
        }
    }
    println!("NN bit 一致: {checked} 件 pass");
}

/// 転置 view の物理配置を作る（`logical` は rows×cols 行優先）。
fn transpose_dense(logical: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = logical[r * cols + c];
        }
    }
    out
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn m64_thin_candidates_strided_nt_tn_bit_identical_to_current() {
    let ctx = MetalContext::new().expect("Metal 初期化に失敗した");
    let gemm = MetalGemm::new(&ctx).expect("GEMM 構築に失敗した");
    // 8 の倍数形状（strided tiled の適格条件）。L1 d_input = NT、dW = TN の向き。
    let shapes: &[(usize, usize, usize)] = &[
        (784, 64, 256),
        (256, 64, 8),
        (784, 256, 64),
        (256, 8, 64),
        (64, 248, 776),
        (64, 200, 304),
    ];
    let mut checked = 0usize;
    for &(m, n, k) in shapes {
        let a_log = fill_det(31 + m as u64, m * k);
        let b_log = fill_det(47 + n as u64, k * n);
        for pattern in ["NT", "TN"] {
            let (a_phys, a_layout) = if pattern == "TN" {
                (
                    transpose_dense(&a_log, m, k),
                    classify_2d(&[m, k], &[1, m as isize]).expect("A 転置 view"),
                )
            } else {
                (
                    a_log.clone(),
                    classify_2d(&[m, k], &[k as isize, 1]).expect("A view"),
                )
            };
            let (b_phys, b_layout) = if pattern == "NT" {
                (
                    transpose_dense(&b_log, k, n),
                    classify_2d(&[k, n], &[1, k as isize]).expect("B 転置 view"),
                )
            } else {
                (
                    b_log.clone(),
                    classify_2d(&[k, n], &[n as isize, 1]).expect("B view"),
                )
            };
            let a_buf = MetalBuffer::new_with_data(&ctx, &a_phys).unwrap();
            let b_buf = MetalBuffer::new_with_data(&ctx, &b_phys).unwrap();
            for t in M64_THIN_CANDIDATES {
                let cur = current_cfg(&ctx, m, n, k, *t);
                let run = |cfg: TileConfig| {
                    let c_buf = MetalBuffer::new_zeroed(&ctx, m * n).unwrap();
                    let resolved = gemm
                        .dispatch_strided_tiled_prepared(
                            &ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k, cfg,
                        )
                        .expect("strided tiled の dispatch に失敗した");
                    (resolved, c_buf.read_to_vec())
                };
                let (rt, got) = run(*t);
                // fallback_chain の 8x8 退避で比較が自明合格するのを防ぐ。
                assert_eq!(rt, *t, "{pattern} {m}x{n}x{k}: 候補が解決されなかった");
                let (_, want) = run(cur);
                if let Err(e) = bit_equal(&got, &want) {
                    panic!("{pattern} {m}x{n}x{k} {t:?} vs {cur:?}: {e}");
                }
                checked += 1;
            }
        }
    }
    println!("NT/TN bit 一致: {checked} 件 pass");
}
