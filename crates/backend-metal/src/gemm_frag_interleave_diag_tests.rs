//! simdgroup 交互配置候補（`FRAG_INTERLEAVE_ENABLED`・function constant index 19。
//! イシュー #3211）の自己検証（bit 一致・CPU 参照 parity）。すべて Metal 実機
//! （Apple Silicon）依存の `#[ignore]` テストで CI では実行しない。型検査だけは
//! `make check-cross-metal-tests` が担う。
//!
//! # 位置づけ
//!
//! - 候補の定義・根拠・重複確認表: `docs/perf/metal-gemm-steel-candidates.md` §10
//! - 比較は常に ON（`MetalGemm::new_with_frag_interleave(ctx, true)`）と OFF
//!   （`MetalGemm::new`）の出力の全要素 `to_bits` 一致。各 8×8 出力ブロックの K 方向
//!   累算列は担当 simdgroup に依らず不変なので bit 一致が契約。NaN の payload を含め
//!   緩めない（不一致は実装不具合として扱う）。
//! - 性能 A/B は受け入れ条件外（後続 issue）。本ファイルは判定を伴う正しさの検査のみ。
//! - 雛形: `gemm_steel_candidate_diag_tests.rs`（#2110）。

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::layout::{MatrixLayout, TransposePattern};
use crate::tile::{self, GemmRoute};

const SIZES: [usize; 5] = [256, 512, 1024, 2048, 4096];

fn bits_of(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// OFF（`MetalGemm::new`）と ON の 2 インスタンス。ゲート値も確認する。
fn pair(ctx: &MetalContext) -> (MetalGemm, MetalGemm) {
    let off = MetalGemm::new(ctx).expect("OFF GEMM の構築に失敗した");
    let on = MetalGemm::new_with_frag_interleave(ctx, true).expect("ON GEMM の構築に失敗した");
    assert!(!off.frag_interleave_enabled());
    assert!(on.frag_interleave_enabled());
    (off, on)
}

fn run_tiled(
    ctx: &MetalContext,
    gemm: &MetalGemm,
    a: &[f32],
    b: &[f32],
    (m, n, k): (usize, usize, usize),
    cfg: tile::TileConfig,
) -> Vec<f32> {
    let a_buf = MetalBuffer::new_with_data(ctx, a).expect("A アップロードに失敗した");
    let b_buf = MetalBuffer::new_with_data(ctx, b).expect("B アップロードに失敗した");
    let c_buf = MetalBuffer::new_zeroed(ctx, m * n).expect("C 確保に失敗した");
    gemm.dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, m, n, k, cfg)
        .expect("dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
    c_buf.read_to_vec()
}

fn fill(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(0x3211);
    (rng.fill_vec(m * k), rng.fill_vec(k * n))
}

/// 1: 全 `CANDIDATES` × 正方 256〜4096・8 の倍数の端数形状・K 端数で ON/OFF が bit 一致。
/// `dispatch_tiled_prepared` は pad8 済み次元を前提とするため強制 cfg の形状は 8 の倍数。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_on_off_bit_match_all_candidates() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let (off, on) = pair(&ctx);
    let mut shapes: Vec<(usize, usize, usize)> = SIZES.iter().map(|&s| (s, s, s)).collect();
    shapes.extend([
        (104, 136, 72),
        (1032, 1048, 1032),
        (40, 24, 8),
        (512, 512, 520),
        (72, 72, 72),
    ]);
    for (i, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
        for gemm in [&off, &on] {
            let resolved = gemm
                .resolve_tile_config(&ctx, cfg)
                .expect("構成の解決に失敗した");
            assert_eq!(
                resolved, cfg,
                "index={i}: フォールバックが発生した（検証が空振りする）"
            );
        }
        for &(m, n, k) in &shapes {
            let (a, b) = fill(m, n, k);
            let off_c = run_tiled(&ctx, &off, &a, &b, (m, n, k), cfg);
            let on_c = run_tiled(&ctx, &on, &a, &b, (m, n, k), cfg);
            assert_eq!(
                bits_of(&off_c),
                bits_of(&on_c),
                "index={i} cfg={cfg:?} shape=({m},{n},{k}): FRAG_INTERLEAVE の違いで bit 一致しなかった"
            );
        }
    }
}

/// 2: 本番自動選択経路で、8 の倍数でない真の端数形状と正方 256〜4096 が bit 一致。
/// 非 split-K の Classic 経路であることを確認し、split-K 形状は no-op の煙テストにする。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_on_off_bit_match_dispatch_auto() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let (off, on) = pair(&ctx);
    let mut shapes: Vec<(usize, usize, usize)> = SIZES.iter().map(|&s| (s, s, s)).collect();
    shapes.extend([
        (129, 131, 133),
        (257, 250, 263),
        (1000, 1000, 1000),
        (1023, 1025, 1027),
    ]);
    let mut classic_count = 0usize;
    for &(m, n, k) in &shapes {
        let (a, b) = fill(m, n, k);
        let (off_c, off_route) = off
            .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
            .expect("OFF dispatch_auto_with_route に失敗した");
        let (on_c, on_route) = on
            .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
            .expect("ON dispatch_auto_with_route に失敗した");
        // ゲート値に依らず経路は同一（ホストは split-K パス 1 へ常に false を渡す）。
        assert_eq!(
            off_route, on_route,
            "shape=({m},{n},{k}): 経路が ON/OFF で異なる"
        );
        if matches!(off_route, GemmRoute::Classic(_)) {
            classic_count += 1;
        } else {
            eprintln!("shape=({m},{n},{k}) は split-K 経路（no-op 煙テストとして比較）");
        }
        assert_eq!(
            bits_of(&off_c),
            bits_of(&on_c),
            "shape=({m},{n},{k}): dispatch_auto で bit 一致しなかった"
        );
    }
    // 検証が空振りしないよう、非 split-K の Classic 形状が十分あることを要求する
    // （正方 256〜4096 と大きい端数形状は Classic）。
    assert!(
        classic_count >= SIZES.len(),
        "Classic 経路の形状が少なすぎる: {classic_count}"
    );
    // split-K 対象形状: ホスト側が no-op（false）を渡すため ON でも bit 一致する（煙テスト）。
    let (m, n, k) = (64, 64, 4096);
    let (a, b) = fill(m, n, k);
    let (off_c, route) = off
        .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
        .expect("OFF split-K 形状の dispatch に失敗した");
    let (on_c, _) = on
        .dispatch_auto_with_route(&ctx, &a, &b, m, n, k)
        .expect("ON split-K 形状の dispatch に失敗した");
    eprintln!("split-K smoke route={route:?}");
    assert_eq!(
        bits_of(&off_c),
        bits_of(&on_c),
        "split-K 形状で bit 一致しなかった"
    );
}

/// 3: 転置ロード（NT／TN／TT）× 主要候補で ON/OFF が bit 一致。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_on_off_bit_match_transposed() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let (off, on) = pair(&ctx);
    const SIZE: usize = 1024;
    for idx in [0usize, 1, 2, 3, 5] {
        let cfg = tile::CANDIDATES[idx];
        for (pattern, trans_a, trans_b) in [
            (TransposePattern::Nt, false, true),
            (TransposePattern::Tn, true, false),
            (TransposePattern::Tt, true, true),
        ] {
            let layout = |transposed| MatrixLayout {
                rows: SIZE,
                cols: SIZE,
                ld: SIZE,
                transposed,
            };
            let (a, b) = fill(SIZE, SIZE, SIZE);
            let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A アップロードに失敗した");
            let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B アップロードに失敗した");
            let mut outs = Vec::new();
            for gemm in [&off, &on] {
                let c_buf = MetalBuffer::new_zeroed(&ctx, SIZE * SIZE).expect("C 確保に失敗した");
                gemm.dispatch_strided_tiled_prepared(
                    &ctx,
                    &a_buf,
                    0,
                    layout(trans_a),
                    &b_buf,
                    0,
                    layout(trans_b),
                    &c_buf,
                    SIZE,
                    SIZE,
                    SIZE,
                    cfg,
                )
                .expect("dispatch_strided_tiled_prepared に失敗した");
                outs.push(bits_of(&c_buf.read_to_vec()));
            }
            assert_eq!(
                outs[0], outs[1],
                "index={idx} pattern={pattern:?}: 転置ロードで bit 一致しなかった"
            );
        }
    }
}

/// 4: NaN／±inf／inf×0 を含む入力でも bit 一致（緩めない）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_on_off_bit_match_nan_inf() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let (off, on) = pair(&ctx);
    for idx in [1usize, 2, 3] {
        let cfg = tile::CANDIDATES[idx];
        for &(m, n, k) in &[(128usize, 128usize, 128usize), (104, 136, 72)] {
            let (mut a, mut b) = fill(m, n, k);
            let specials = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0];
            for (j, &s) in specials.iter().enumerate() {
                let ai = (7 + 37 * j) % a.len();
                let bi = (11 + 53 * j) % b.len();
                a[ai] = s;
                b[bi] = s;
            }
            // inf × 0 が同じ内積に現れる組合せ。
            a[0] = f32::INFINITY;
            b[0] = 0.0;
            a[k + 1] = f32::NEG_INFINITY;
            b[n + 1] = 0.0;
            let off_c = run_tiled(&ctx, &off, &a, &b, (m, n, k), cfg);
            let on_c = run_tiled(&ctx, &on, &a, &b, (m, n, k), cfg);
            assert_eq!(
                bits_of(&off_c),
                bits_of(&on_c),
                "index={idx} shape=({m},{n},{k}): NaN/inf 入力で bit 一致しなかった"
            );
        }
    }
}

/// 5: 直交ゲート（FINE_BARRIER・UNROLL_LOAD）との併用でも ON/OFF が bit 一致。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_combinations_bit_match() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    for (unroll_load, fine_barrier) in [(false, false), (true, false), (false, true), (true, true)]
    {
        let off = MetalGemm::new_with_steel_candidate(&ctx, false, unroll_load, fine_barrier)
            .expect("OFF 併用 GEMM の構築に失敗した");
        let on = MetalGemm::new_with_steel_candidate(&ctx, false, unroll_load, fine_barrier)
            .expect("ON 併用 GEMM の構築に失敗した")
            .with_frag_interleave(true);
        for idx in [1usize, 2, 3] {
            let cfg = tile::CANDIDATES[idx];
            for &(m, n, k) in &[(512usize, 512usize, 512usize), (104, 136, 72)] {
                let (a, b) = fill(m, n, k);
                let off_c = run_tiled(&ctx, &off, &a, &b, (m, n, k), cfg);
                let on_c = run_tiled(&ctx, &on, &a, &b, (m, n, k), cfg);
                assert_eq!(
                    bits_of(&off_c),
                    bits_of(&on_c),
                    "unroll_load={unroll_load} fine_barrier={fine_barrier} index={idx} \
                     shape=({m},{n},{k}): 併用で bit 一致しなかった"
                );
            }
        }
    }
}

/// 6: ON の variant が CPU 参照（FMA 契約）と REQ-2 複合判定で一致（tolerance は不変）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_matches_cpu_reference() {
    use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let on = MetalGemm::new_with_frag_interleave(&ctx, true).expect("ON GEMM の構築に失敗した");
    let mut shapes: Vec<(usize, usize, usize)> = [256usize, 512, 1024, 2048]
        .iter()
        .map(|&s| (s, s, s))
        .collect();
    shapes.extend([(104, 136, 72), (1032, 1048, 1032)]);
    for idx in [1usize, 2, 3] {
        let cfg = tile::CANDIDATES[idx];
        for &(m, n, k) in &shapes {
            let (a, b) = fill(m, n, k);
            let out = run_tiled(&ctx, &on, &a, &b, (m, n, k), cfg);
            let mut expected = vec![0.0f32; m * n];
            matmul_reference_fma(&a, &b, &mut expected, m, n, k)
                .expect("CPU 参照実装の形状検証に失敗した");
            assert_parity(
                &format!("metal frag_interleave index={idx} shape=({m},{n},{k})"),
                &out,
                &expected,
            );
        }
    }
}

/// 7: ゲートの保持と、f16 系構成解決が影響を受けないこと（no-op 契約の煙テスト）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn frag_interleave_gate_is_forwarded() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let (off, on) = pair(&ctx);
    for g in [&off, &on] {
        g.resolve_tile_config_f16(&ctx, tile::CANDIDATES[0])
            .expect("f16 系の構成解決に失敗した");
    }
}
