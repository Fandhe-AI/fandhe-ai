//! 協調ロードの threadgroup メモリ格納位置 XOR swizzle 軸（`tile::
//! SmemSwizzle`。イシュー #1970 で実装・Linux 実行可能な範囲で bit 一致を
//! 自己検証済み。実機 bit 一致は `gemm::tests::smem_swizzle_bit_match_*`）
//! の N=512/1024/2048/4096 純カーネル時間（GPU タイムスタンプ。イシュー
//! #1276 の `kernel_gpu` 変種）を M4 Max で 5 回計測し、有効性（`tile::
//! select` への組み込み可否・採用候補）を判定する診断テスト。
//!
//! `crates/backend-metal/src/gemm_coop_load_diag_tests.rs`
//! （E4 協調ロードレイアウト候補・イシュー #1300）を雛形とする。
//!
//! # 対象 arm（7 arm。事前登録・issue コメント固定）
//!
//! - `L0-P4-S0`（本番既定。base）
//! - `L0-P0-S0`（対照: パディングなし単独）
//! - `L0-P4-S1`／`L0-P0-S1`（「パディングの代わりに XOR」仮説の主 head）
//! - `L0-P8-S1`
//! - `L0-P4-S2`／`L0-P0-S2`
//!
//! `L1-*`（`CoopLoadLayout::RowStrided`）は #1300 で N=4096 符号一貫後退の
//! REJECT 済みのため本診断の性能 A/B arm からは除外する（bit 一致は
//! `gemm::tests::smem_swizzle_*` 側が全 6 候補を被覆する）。
//!
//! # bit 一致・checksum（trial 0 のみ）
//!
//! `kernel_gpu` 計測ループとは別に、各 N の trial 0 で全 arm の
//! `dispatch_tiled_prepared` 出力を base と `to_bits()` 比較し、
//! `bit_identical=<bool>` と checksum（f64 逐次和。`gemm_te_diag_tests.rs`
//! の `checksum_f64`／`bit_identical` と同型）を出力する。abort はせず
//! 集計側（オーケストレーション `aggregate.py`）で判定する。
//!
//! # 配置理由・実行時の注意・gating しない方針
//!
//! `gemm_coop_load_diag_tests.rs` と同じ（`MetalGemm::new_with_smem_
//! swizzle`・`gemm_reuse_phase_diag_tests` の `pub(crate)` 面へ到達する
//! ため `lib.rs` の兄弟モジュールとして配置・`--test-threads=1` 前提・
//! `kernel_gpu` の大小関係への `assert!` は行わない）。
//!
//! # プロダクションコード不変
//!
//! 本ファイルは `gemm.rs`／`tile.rs`／`shaders/gemm.metal` への変更を
//! 一切含まない（イシュー #1970 が既に追加済みの `#[cfg(test)] pub`／
//! `pub(crate)` 面のみを利用する）。`tile::select` の候補表・本番既定
//! （`MetalGemm::new` の `tile::SMEM_SWIZZLE`）への組み込みは本イシュー・
//! 本ファイルのスコープ外（実機〈Apple Silicon〉未実測のまま出荷。
//! `docs/perf/metal-gemm-n4096-kernel-gap.md` §該当節）。

use crate::context::MetalContext;
use crate::gemm::MetalGemm;
use crate::gemm_reuse_phase_diag_tests::{
    MEASURED_TRIALS, WARMUP_TRIALS, gen_square_ab, measure_one_phase_trial, median_of,
};
use crate::tile::{self, CoopLoadConfig, CoopLoadLayout, SmemSwizzle, TgpPad};
use objc2_metal::MTLDevice;

/// [`xor_swizzle_kernel_gpu_ab_production_sizes`] が対象とするサイズ
/// （`gemm_coop_load_diag_tests::SIZES` に N=512 を加えた 4 点。事前登録
/// 判定規則が N 別判定のため N=512 も計測対象に含める）。
const SIZES: [usize; 4] = [512, 1024, 2048, 4096];

/// 7 arm のラベル・`(CoopLoadConfig, SmemSwizzle)` 対応（事前登録・issue
/// コメント固定。index 0 = base = 本番既定）。
const ARM_LABELS: [&str; 7] = [
    "L0-P4-S0", "L0-P0-S0", "L0-P4-S1", "L0-P0-S1", "L0-P8-S1", "L0-P4-S2", "L0-P0-S2",
];

const ARM_CONFIGS: [(CoopLoadConfig, SmemSwizzle); 7] = [
    (tile::COOP_LOAD_CONFIG, SmemSwizzle::Off), // L0-P4-S0（base）
    (
        CoopLoadConfig {
            layout: CoopLoadLayout::RowLinear,
            pad: TgpPad::Zero,
        },
        SmemSwizzle::Off,
    ), // L0-P0-S0
    (tile::COOP_LOAD_CONFIG, SmemSwizzle::ATile), // L0-P4-S1
    (
        CoopLoadConfig {
            layout: CoopLoadLayout::RowLinear,
            pad: TgpPad::Zero,
        },
        SmemSwizzle::ATile,
    ), // L0-P0-S1
    (
        CoopLoadConfig {
            layout: CoopLoadLayout::RowLinear,
            pad: TgpPad::Eight,
        },
        SmemSwizzle::ATile,
    ), // L0-P8-S1
    (tile::COOP_LOAD_CONFIG, SmemSwizzle::BothTiles), // L0-P4-S2
    (
        CoopLoadConfig {
            layout: CoopLoadLayout::RowLinear,
            pad: TgpPad::Zero,
        },
        SmemSwizzle::BothTiles,
    ), // L0-P0-S2
];

/// f32 出力の f64 逐次和（`gemm_te_diag_tests.rs::checksum_f64` と同型。
/// bit 同一判定の副次的な指標として trial 0 出力へ付与する）。
fn checksum_f64(values: &[f32]) -> f64 {
    values.iter().fold(0.0f64, |acc, &v| acc + v as f64)
}

/// N=512/1024/2048/4096 で 7 arm の `kernel_gpu`（GPU タイムスタンプに
/// よる純カーネル専有時間。イシュー #1276）を trial ごとに交互（開始
/// オフセット回転）で計測し、5 プロセス起動の 1 回分として中央値・
/// `head_over_base_kernel_gpu` 比を出力する（複数プロセス起動・集計は
/// `docs/perf/logs/metal-gemm-smem-swizzle-ab-1970/` の `orchestrate.sh`／
/// `aggregate.py` が行う）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn xor_swizzle_kernel_gpu_ab_production_sizes() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

    // 7 arm すべての `MetalGemm` を先に構築する（`new_with_smem_swizzle` は
    // MSL パイプラインの遅延構築のみで確保コストは小さい。ファイル冒頭
    // 「実行時は必ず `--test-threads=1`」参照）。
    let gemms: Vec<MetalGemm> = ARM_CONFIGS
        .iter()
        .map(|&(coop_cfg, swizzle)| {
            MetalGemm::new_with_smem_swizzle(&ctx, coop_cfg, swizzle)
                .expect("swizzle arm GEMM パイプラインの構築に失敗した")
        })
        .collect();
    assert_eq!(
        gemms[0].smem_swizzle(),
        tile::SMEM_SWIZZLE,
        "index 0（L0-P4-S0）は本番既定と一致するはず"
    );
    assert_eq!(gemms[0].coop_load(), tile::COOP_LOAD_CONFIG);

    let max_shared_mem_bytes = ctx.device().maxThreadgroupMemoryLength() as u32;

    for n in SIZES {
        let base_cfg = tile::select_for_device(n, n, n, ctx.verified_m4_max_gpu_core_count());
        assert!(
            base_cfg.staged,
            "N={n}: 選択構成が staged=false だった。CoopLoadConfig::pad_elems は \
             !staged で常に 0 を返すため P0/P4/P8 が同一カーネルへ縮退し比較が \
             空振りする"
        );

        // arm ごとの共有メモリ事前フィルタ（`TGP_PAD=8` がデバイス上限を
        // 超える場合、`pipeline_for_tile` が base と異なる `resolved_cfg`
        // へフォールバックしうるため事前に除外する。
        // `gemm_coop_load_diag_tests.rs` と同じ設計）。
        let active_indices: Vec<usize> = (0..ARM_LABELS.len())
            .filter(|&i| {
                let (coop_cfg, _) = ARM_CONFIGS[i];
                let pad_elems = coop_cfg.pad_elems(base_cfg);
                base_cfg.shared_mem_bytes_for_pad(crate::layout::TransposePattern::Nn, pad_elems)
                    <= max_shared_mem_bytes
            })
            .collect();
        assert!(
            active_indices.contains(&0),
            "N={n}: base（L0-P4-S0）自体が共有メモリ事前フィルタで除外された。\
             本番既定が動作しない環境のため計測を中断する"
        );
        for (i, label) in ARM_LABELS.iter().enumerate() {
            if !active_indices.contains(&i) {
                println!(
                    "N={n}: arm={label} は共有メモリ事前フィルタで対象外\
                     （shared_mem_bytes_for_pad > maxThreadgroupMemoryLength）"
                );
            }
        }

        let (a, b) = gen_square_ab(0x1970_a000 ^ (n as u64), n);
        let num_active = active_indices.len();

        // trial 0 のみ: bit 一致・checksum を dispatch_tiled_prepared で
        // 直接検証する（`PhaseSample` は出力バッファを保持しないため、
        // kernel_gpu 計測ループとは独立に 1 回だけ実行する。ファイル冒頭
        // 「bit 一致・checksum」節）。
        {
            let base_c = {
                let a_buf = crate::buffer::MetalBuffer::new_with_data(&ctx, &a)
                    .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let b_buf = crate::buffer::MetalBuffer::new_with_data(&ctx, &b)
                    .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let c_buf = crate::buffer::MetalBuffer::new_zeroed(&ctx, n * n)
                    .expect("C バッファの確保に失敗した（実機でのみ実行する前提）");
                gemms[0]
                    .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &c_buf, n, n, n, base_cfg)
                    .expect("base dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
                c_buf.read_to_vec()
            };
            let base_bits: Vec<u32> = base_c.iter().map(|v| v.to_bits()).collect();
            let base_checksum = checksum_f64(&base_c);
            println!("N={n} arm=L0-P4-S0 checksum={base_checksum:.6}");

            for &idx in &active_indices {
                if idx == 0 {
                    continue;
                }
                let a_buf = crate::buffer::MetalBuffer::new_with_data(&ctx, &a)
                    .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let b_buf = crate::buffer::MetalBuffer::new_with_data(&ctx, &b)
                    .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
                let c_buf = crate::buffer::MetalBuffer::new_zeroed(&ctx, n * n)
                    .expect("C バッファの確保に失敗した（実機でのみ実行する前提）");
                gemms[idx]
                    .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &c_buf, n, n, n, base_cfg)
                    .expect("arm dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
                let out = c_buf.read_to_vec();
                let bits: Vec<u32> = out.iter().map(|v| v.to_bits()).collect();
                let checksum = checksum_f64(&out);
                let bit_identical = bits == base_bits;
                println!(
                    "N={n} arm={} checksum={checksum:.6} bit_identical={bit_identical}",
                    ARM_LABELS[idx]
                );
            }
        }

        // 候補ごとの keep_alive（N ごとのスコープで drop することで
        // ピークメモリを 1 N 分に抑える。`gemm_coop_load_diag_tests.rs`
        // と同じ考え方）。
        let mut keep_alives: Vec<Vec<Vec<f32>>> = (0..ARM_LABELS.len())
            .map(|_| Vec::with_capacity(WARMUP_TRIALS + MEASURED_TRIALS))
            .collect();

        for _ in 0..WARMUP_TRIALS {
            for &idx in &active_indices {
                let _ = measure_one_phase_trial(
                    &ctx,
                    &gemms[idx],
                    &a,
                    &b,
                    n,
                    base_cfg,
                    &mut keep_alives[idx],
                );
            }
        }

        let mut kernel_gpu: Vec<Vec<f64>> = (0..ARM_LABELS.len())
            .map(|_| Vec::with_capacity(MEASURED_TRIALS))
            .collect();
        let mut resolved: Vec<Option<tile::TileConfig>> = vec![None; ARM_LABELS.len()];

        for trial in 0..MEASURED_TRIALS {
            let offset = trial % num_active;
            for step in 0..num_active {
                let idx = active_indices[(offset + step) % num_active];
                let sample = measure_one_phase_trial(
                    &ctx,
                    &gemms[idx],
                    &a,
                    &b,
                    n,
                    base_cfg,
                    &mut keep_alives[idx],
                );
                assert_eq!(
                    sample.resolved_cfg, base_cfg,
                    "N={n} trial={trial} arm={}: pipeline_for_tile フォールバックが\
                     発生した(requested={base_cfg:?}, resolved={:?})。性能比較の前提が\
                     崩れるため中断する",
                    ARM_LABELS[idx], sample.resolved_cfg
                );
                kernel_gpu[idx].push(sample.kernel_gpu_secs);
                resolved[idx] = Some(sample.resolved_cfg);
            }
        }

        for &idx in &active_indices {
            let q = median_of(&kernel_gpu[idx]);
            println!(
                "N={n} arm={} smem_swizzle={:?} coop_load={:?} resolved_tile={:?} \
                 kernel_gpu_median_ms={:.4} q1={:.4} q3={:.4}",
                ARM_LABELS[idx],
                ARM_CONFIGS[idx].1,
                ARM_CONFIGS[idx].0,
                resolved[idx].expect("MEASURED_TRIALS > 0 のため必ず Some"),
                q.median * 1e3,
                q.q1 * 1e3,
                q.q3 * 1e3
            );
        }

        // base（L0-P4-S0。index 0）比。base は共有メモリ事前フィルタで
        // 必ず active（上記 assert 参照）なので分母は必ず存在する。
        let base_median = median_of(&kernel_gpu[0]).median;
        for &idx in &active_indices {
            let ratio = median_of(&kernel_gpu[idx]).median / base_median;
            println!(
                "N={n} arm={} head_over_base_kernel_gpu={ratio:.6}",
                ARM_LABELS[idx]
            );
        }
    }
}
