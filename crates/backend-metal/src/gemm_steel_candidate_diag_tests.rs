//! candle／MLX steel 解析差分由来の Metal GEMM 候補（イシュー #2110）の
//! 自己検証（bit 一致・CPU 参照 parity）と kernel_gpu 5 run A/B 計測ハーネス。
//! すべて Metal 実機（Apple Silicon）依存の `#[ignore]` テストで、CI では実行しない。
//! 型検査だけは `make check-cross-metal-tests`（`cargo check --tests --target
//! aarch64-apple-darwin`）が担う。
//!
//! # 位置づけ
//!
//! - 候補の定義・根拠・除外表: `docs/perf/metal-gemm-steel-candidates.md`
//!   （出典 `docs/analysis/candle-metal-01.md` §5・§6、`docs/analysis/mlx-v0.32.2-steel-gemm.md`）
//! - 事前登録 arm 表: [`crate::tile::STEEL_ARMS`]（`base`／`T0U`／`LU`／`T0U-LU`／`T0U-LU-FB`）
//! - 判定規則（実測前固定）: `docs/perf/logs/metal-gemm-candidate-ab-2111/RULE.txt`。
//!   実機 5 run 計測・結線判断は #2111 のスコープで、本ファイルはその入力を出す。
//! - 雛形: `gemm_smem_swizzle_diag_tests.rs`（#1970）。`gemm_reuse_phase_diag_tests` の
//!   `pub(crate)` 面へ到達するため `lib.rs` のクレートルート兄弟モジュールとして配置する。
//!
//! # テスト構成
//!
//! - assert あり（前提ゲート。RULE.txt 1.）:
//!   `unroll_load_on_off_bit_match_all_candidates`／`_dispatch_auto`／`_transposed`、
//!   `steel_candidate_arms_match_cpu_reference`（REQ-2 `assert_parity`。tolerance は不変）
//! - assert なし（記録のみ）: `steel_candidate_kernel_gpu_ab_production_sizes`。`kernel_gpu`
//!   の大小・タイル形状が異なる arm 同士の bit 一致（E7 §13.4 と同じく契約外）は
//!   `aggregate.py` が判定する。
//!
//! # プロダクションコード不変
//!
//! 本番既定（`MetalGemm::new`・`tile::select*`・`dispatch_auto`）は無変更。

use crate::buffer::MetalBuffer;
use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::layout::{MatrixLayout, TransposePattern};
use crate::tile::{self, STEEL_ARMS, SteelArm};

/// 事前登録サイズ（RULE.txt。#2111 の採用条件は N=512/1024/2048/4096 の 4 形状）。
const SIZES: [usize; 4] = [512, 1024, 2048, 4096];

/// f32 出力の f64 逐次和（`gemm_smem_swizzle_diag_tests.rs::checksum_f64` と同型）。
fn checksum_f64(values: &[f32]) -> f64 {
    values.iter().fold(0.0f64, |acc, &v| acc + v as f64)
}

fn bits_of(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// 1 回の `dispatch_tiled_prepared` 実行結果（C 行列）。
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

/// 協調ロード unroll を ON/OFF した 2 インスタンス（他軸は本番既定）を返す。
fn load_pair(ctx: &MetalContext) -> (MetalGemm, MetalGemm) {
    let base = MetalGemm::new_with_steel_candidate(ctx, false, false, false)
        .expect("base GEMM の構築に失敗した");
    let head = MetalGemm::new_with_steel_candidate(ctx, false, true, false)
        .expect("unroll_load GEMM の構築に失敗した");
    assert!(!base.unroll_load_enabled());
    assert!(head.unroll_load_enabled());
    (base, head)
}

/// 1: `UNROLL_LOAD_ENABLED` の ON/OFF が、全 `CANDIDATES` × N=512〜4096・境界形状・
/// K 端数形状で出力を bit 単位で変えないこと（assert）。フォールバックで
/// 別構成に解決されていないこと（`resolved == cfg`）も確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn unroll_load_on_off_bit_match_all_candidates() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let (base, head) = load_pair(&ctx);

    // (m, n, k): 正方 4 形状 + 8 の倍数の小形状 + 大形状端数 + K 端数。
    let mut shapes: Vec<(usize, usize, usize)> = SIZES.iter().map(|&s| (s, s, s)).collect();
    shapes.extend([(104, 136, 72), (1032, 1048, 1032), (512, 512, 520)]);

    for (i, cfg) in tile::CANDIDATES.iter().copied().enumerate() {
        for &(m, n, k) in &shapes {
            for (name, gemm) in [("base", &base), ("head", &head)] {
                let resolved = gemm
                    .resolve_tile_config(&ctx, cfg)
                    .expect("構成の解決に失敗した");
                assert_eq!(
                    resolved, cfg,
                    "{name} index={i} shape=({m},{n},{k}): フォールバックが発生した（検証が空振りする）"
                );
            }
            let mut rng = bench_harness::rng::Xorshift64Star::new(0x2110);
            let a = rng.fill_vec(m * k);
            let b = rng.fill_vec(k * n);
            let base_c = run_tiled(&ctx, &base, &a, &b, (m, n, k), cfg);
            let head_c = run_tiled(&ctx, &head, &a, &b, (m, n, k), cfg);
            assert_eq!(
                bits_of(&base_c),
                bits_of(&head_c),
                "index={i} cfg={cfg:?} shape=({m},{n},{k}): UNROLL_LOAD の違いで出力が bit 単位で\
                 一致しなかった。shaders/gemm.metal の協調ロード if/else 複製の本体差異を確認すること。"
            );
        }
    }
}

/// 2: 本番自動選択経路 `dispatch_auto` で N=512〜4096 の bit 一致（assert）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn unroll_load_on_off_bit_match_dispatch_auto() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let (base, head) = load_pair(&ctx);
    for n in SIZES {
        let mut rng = bench_harness::rng::Xorshift64Star::new(0x2110);
        let a = rng.fill_vec(n * n);
        let b = rng.fill_vec(n * n);
        let base_out = base
            .dispatch_auto(&ctx, &a, &b, n, n, n)
            .expect("base dispatch_auto に失敗した");
        let head_out = head
            .dispatch_auto(&ctx, &a, &b, n, n, n)
            .expect("head dispatch_auto に失敗した");
        assert_eq!(
            bits_of(&base_out),
            bits_of(&head_out),
            "n={n}: dispatch_auto で UNROLL_LOAD の違いにより bit 一致しなかった"
        );
    }
}

/// 3: 転置ロード（NT／TN／TT）でも協調ロード unroll が bit 一致すること（assert）。
/// TRANS_A／TRANS_B 分岐の複製ブロックを被覆する（性能 A/B は対象外）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn unroll_load_on_off_bit_match_transposed() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let (base, head) = load_pair(&ctx);
    const SIZE: usize = 1024;
    for cfg in [
        tile::CANDIDATES[0],
        tile::CANDIDATES[3],
        tile::CANDIDATES[5],
    ] {
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
            let mut rng = bench_harness::rng::Xorshift64Star::new(0x2110);
            let a = rng.fill_vec(SIZE * SIZE);
            let b = rng.fill_vec(SIZE * SIZE);
            let a_buf = MetalBuffer::new_with_data(&ctx, &a).expect("A アップロードに失敗した");
            let b_buf = MetalBuffer::new_with_data(&ctx, &b).expect("B アップロードに失敗した");
            let mut outs = Vec::new();
            for gemm in [&base, &head] {
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
                "cfg={cfg:?} pattern={pattern:?}: 転置ロードで UNROLL_LOAD の違いにより bit 一致しなかった"
            );
        }
    }
}

/// 4: 実効ゲートの転送確認（instance フラグが保持されること）と、候補インスタンスの
/// f16 系パイプライン構築が影響を受けないこと（no-op 契約の煙テスト）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn unroll_load_effective_gate_is_forwarded() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let default = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した");
    assert!(
        !default.unroll_load_enabled(),
        "本番既定は unroll_load=false のはず"
    );
    for arm in STEEL_ARMS {
        let g = MetalGemm::new_with_steel_candidate(
            &ctx,
            arm.unroll_acc,
            arm.unroll_load,
            arm.fine_barrier,
        )
        .expect("arm GEMM の構築に失敗した");
        assert_eq!(g.unroll_load_enabled(), arm.unroll_load, "{}", arm.label);
        assert_eq!(g.unroll_acc_enabled_flag(), arm.unroll_acc, "{}", arm.label);
        assert_eq!(
            g.fine_barrier_enabled_flag(),
            arm.fine_barrier,
            "{}",
            arm.label
        );
        // f16 経路（`gemm_simdgroup_tiled_f16`）は本軸を参照しない。構築できること
        // （function constant 未参照でも失敗しない）を確認する。
        g.resolve_tile_config_f16(&ctx, tile::CANDIDATES[0])
            .expect("f16 系の構成解決に失敗した");
    }
}

/// 5: 全 arm × 4 N・境界形状が CPU 参照（FMA 契約）と REQ-2 複合判定で一致すること
/// （assert。tolerance は不変）。タイル形状が異なる arm 同士の bit 一致は契約外のため
/// ここでは検査しない。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn steel_candidate_arms_match_cpu_reference() {
    use fandhe_ai_backend_cpu::parity::{assert_parity, matmul_reference_fma};

    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let verified = ctx.verified_m4_max_gpu_core_count();
    let mut shapes: Vec<(usize, usize, usize)> = SIZES.iter().map(|&s| (s, s, s)).collect();
    shapes.push((1032, 1048, 1032));
    for arm in STEEL_ARMS {
        let gemm = MetalGemm::new_with_steel_candidate(
            &ctx,
            arm.unroll_acc,
            arm.unroll_load,
            arm.fine_barrier,
        )
        .expect("arm GEMM の構築に失敗した");
        for &(m, n, k) in &shapes {
            let cfg = arm.tile_for(m.max(n), verified);
            let resolved = gemm
                .resolve_tile_config(&ctx, cfg)
                .expect("構成の解決に失敗した");
            assert_eq!(
                resolved, cfg,
                "arm={} shape=({m},{n},{k}): フォールバック",
                arm.label
            );
            let mut rng = bench_harness::rng::Xorshift64Star::new(0x2110);
            let a = rng.fill_vec(m * k);
            let b = rng.fill_vec(k * n);
            let out = run_tiled(&ctx, &gemm, &a, &b, (m, n, k), cfg);
            let mut expected = vec![0.0f32; m * n];
            matmul_reference_fma(&a, &b, &mut expected, m, n, k)
                .expect("CPU 参照実装の形状検証に失敗した");
            assert_parity(
                &format!("metal steel arm={} shape=({m},{n},{k})", arm.label),
                &out,
                &expected,
            );
        }
    }
}

/// N=512/1024/2048/4096 で全 arm の `kernel_gpu`（GPU タイムスタンプ。#1276 の変種）を
/// trial ごとに交互（開始オフセット回転）で計測し、プロセス 1 回分の中央値・
/// `head_over_base_kernel_gpu` 比・checksum・bit 一致を出力する。5 プロセス起動・
/// 集計は `docs/perf/logs/metal-gemm-candidate-ab-2111/` の `orchestrate.sh`／
/// `aggregate.py` が行う。`kernel_gpu` の大小は assert しない（記録のみ）。
///
/// 出力行（`aggregate.py` の正規表現と一致させる。変更時は両方を更新する）:
/// - `N=<n> arm=<label> checksum=<f> bit_identical=<bool> same_kernel=<bool>`
/// - `N=<n> arm=<label> resolved_tile=<cfg> kernel_gpu_median_ms=<v> q1=<v> q3=<v>`
/// - `N=<n> arm=<label> head_over_base_kernel_gpu=<r>`
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn steel_candidate_kernel_gpu_ab_production_sizes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let verified = ctx.verified_m4_max_gpu_core_count();

    let gemms: Vec<MetalGemm> = STEEL_ARMS
        .iter()
        .map(|arm: &SteelArm| {
            MetalGemm::new_with_steel_candidate(
                &ctx,
                arm.unroll_acc,
                arm.unroll_load,
                arm.fine_barrier,
            )
            .expect("arm GEMM の構築に失敗した")
        })
        .collect();
    assert_eq!(STEEL_ARMS[0].label, "base");

    for n in SIZES {
        let cfgs: Vec<tile::TileConfig> =
            STEEL_ARMS.iter().map(|a| a.tile_for(n, verified)).collect();
        for (arm, cfg) in STEEL_ARMS.iter().zip(&cfgs) {
            assert!(
                cfg.staged,
                "N={n} arm={}: staged=false では協調ロード軸が空振りする",
                arm.label
            );
        }
        let (a, b) = gen_square_ab(0x2110_a000 ^ (n as u64), n);
        let num = STEEL_ARMS.len();

        // trial 0: bit 一致・checksum（abort しない。判定は aggregate.py）。
        let base_c = run_tiled(&ctx, &gemms[0], &a, &b, (n, n, n), cfgs[0]);
        let base_bits = bits_of(&base_c);
        for idx in 0..num {
            let out = if idx == 0 {
                base_c.clone()
            } else {
                run_tiled(&ctx, &gemms[idx], &a, &b, (n, n, n), cfgs[idx])
            };
            let same_kernel = STEEL_ARMS[idx].same_kernel_as_base(n, verified);
            println!(
                "N={n} arm={} checksum={:.6} bit_identical={} same_kernel={same_kernel}",
                STEEL_ARMS[idx].label,
                checksum_f64(&out),
                bits_of(&out) == base_bits,
            );
        }

        let mut keep_alives: Vec<Vec<Vec<f32>>> = (0..num)
            .map(|_| Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS))
            .collect();
        for _ in 0..WARMUP_TRIALS {
            for idx in 0..num {
                let _ = measure_one_phase_trial(
                    &ctx,
                    &gemms[idx],
                    &a,
                    &b,
                    n,
                    cfgs[idx],
                    &mut keep_alives[idx],
                );
            }
        }
        let mut kernel_gpu: Vec<Vec<f64>> = (0..num)
            .map(|_| Vec::with_capacity(MEASURED_TRIALS))
            .collect();
        for trial in 0..MEASURED_TRIALS {
            let offset = trial % num;
            for step in 0..num {
                let idx = (offset + step) % num;
                let sample = measure_one_phase_trial(
                    &ctx,
                    &gemms[idx],
                    &a,
                    &b,
                    n,
                    cfgs[idx],
                    &mut keep_alives[idx],
                );
                assert_eq!(
                    sample.resolved_cfg, cfgs[idx],
                    "N={n} trial={trial} arm={}: pipeline_for_tile フォールバックが発生した\
                     (requested={:?}, resolved={:?})。比較の前提が崩れるため中断する",
                    STEEL_ARMS[idx].label, cfgs[idx], sample.resolved_cfg
                );
                kernel_gpu[idx].push(sample.kernel_gpu_secs);
            }
        }
        for idx in 0..num {
            let q = median_of(&kernel_gpu[idx]);
            println!(
                "N={n} arm={} resolved_tile={:?} kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
                STEEL_ARMS[idx].label,
                cfgs[idx],
                q.median * 1e3,
                q.q1 * 1e3,
                q.q3 * 1e3
            );
        }
        let base_median = median_of(&kernel_gpu[0]).median;
        for idx in 0..num {
            let ratio = median_of(&kernel_gpu[idx]).median / base_median;
            println!(
                "N={n} arm={} head_over_base_kernel_gpu={ratio:.6}",
                STEEL_ARMS[idx].label
            );
        }
    }
}
