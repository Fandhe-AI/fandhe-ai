//! TASK-8.3c（#157）: CUDA 最適化後下限（暫定 40%）の再実測バイナリ。
//!
//! REQ-8 の CUDA f32/f16 最適化後下限 40% は「tensor core 実装完了後の
//! 実測で再確定する」という条件付きの**暫定値**である
//! （`docs/spec/04-requirements.md`「CUDA f32 対 PyTorch CUDA」行・
//! `docs/spec/05-tasks.md` TASK-8.3「CUDA f32/f16 の最適化後下限（暫定値
//! 40%）は Tensor Core 実装完了後の実測で再確定する」）。TASK-11.1 系
//! （#59〜#65・#186・#187）で tiled f32／WMMA(TF32) opt／WMMA f16 opt／
//! `mma.sync` f16 パイプラインが `backend-cuda` に揃ったため、本バイナリは
//! それらを PoC-v2-3 と同一形状（M=N=K=512/2048/4096）で再計測し、
//! `docs/spec/04-requirements.md` の丸め規則（実測比率 10% 以上は 5% 刻み
//! 切り下げ・10% 未満は 1% 刻み切り下げ・条件付き追加ステップなし）を
//! 適用した**候補下限値**を出力する。
//!
//! 下限の最終確定は人間の判断事項であり（TASK-8.3「担当: 共同（計測実行は
//! Claude Code、下限値の最終確定は人間）」）、本バイナリの出力は
//! `docs/perf/cuda-floor-remeasurement.md` の実測記録テンプレへ転記した
//! うえで #158（TASK-8.3d）に引き継ぐ。REQ-8 は「判定対象形状は演算律速域
//! （M=N=K=2048・4096）の実測比率の最小値」と定めているため、512 は
//! 参考値として出力するのみで候補下限値の算出には使わない。
//!
//! `crates/bench-harness` の TASK-8.2（下限判定モジュール。#151〜#153）は
//! 本バイナリ実装時点で未マージのため、丸め規則は本ファイル内に純関数
//! として最小実装する（[`floor_round`]）。TASK-8.2 モジュールが
//! マージされ次第、#158/#159 でそちらへ一本化する
//! （実装計画「丸め規則の実装」節。out-of-scope-tracking 対応）。
//!
//! `examples/` に置くのは、通常の `cargo test`／CI では実行されず
//! ビルド検証（`cargo build --workspace --all-targets`）のみが CI で走る
//! ようにするため（self-hosted runner をベンチ実行で占有しない。`ci.md`）。
//! `bench-harness` は既に `backend-cuda` の `dev-dependencies`
//! （`examples/gemm_mma_bench.rs` が使用）であり、本ファイルの追加に伴う
//! `Cargo.toml` の変更は不要（`deps-policy.md` ユーザー承認事項に該当しない）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example cuda_floor_bench --release
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載・cc 非対応環境では、各経路の初期化失敗を
//! 検出した時点でその経路をスキップし理由を表示する（`gemm_mma_bench.rs`
//! と同じ環境適応分岐）。実測値は
//! `docs/perf/cuda-floor-remeasurement.md` の記録テンプレへ転記する。

use backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaMmaGemm, CudaWmmaGemm};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use half::f16;

/// 決定的シード（`gemm_mma_bench.rs`・PoC-v2-3 と同一値。過去実測・他
/// バックエンドベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

/// 判定対象形状（REQ-8「判定対象形状は演算律速域〈M=N=K=2048・4096〉」）。
/// 512 は参考値としてのみ計測し、候補下限値の算出には使わない。
const JUDGED_SIZES: [usize; 2] = [2048, 4096];
const REFERENCE_ONLY_SIZE: usize = 512;

/// PoC-v2-3 実測の PyTorch CUDA 実効値（TFLOPS、5〜20 回中央値。
/// DGX Spark GB10・PyTorch 2.13.0+cu130）。
/// `docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md`「計測結果」節の
/// 2 表から転記した参照値であり、同一実機（GB10）での比較を前提とする
/// （REQ-8「いずれも同一ハードウェア上の同一バックエンド比較」）。
/// 本バイナリの実行環境が GB10 系でない場合は `main` 内の GPU 名検出で
/// 参考値扱いの警告を出す。
fn pytorch_f32_tflops(size: usize) -> f64 {
    match size {
        512 => 7.8803,
        2048 => 17.4241,
        4096 => 17.7774,
        _ => f64::NAN,
    }
}

fn pytorch_f16_tflops(size: usize) -> f64 {
    match size {
        512 => 17.1898,
        2048 => 91.2115,
        4096 => 97.6308,
        _ => f64::NAN,
    }
}

/// REQ-8 の丸め規則（v2 統一版）を実装する純関数。
///
/// `docs/spec/04-requirements.md`「丸め規則の統一（旧 #4 の解消）」節:
/// 実測比率（0.0〜1.0 のうち、ここでは % 表現の `f64` を受け取る）が
/// 10% 以上の場合は 5% 刻みで切り下げ、10% 未満の場合は 1% 刻みで切り
/// 下げる（条件付き追加ステップは廃止済み・非減少性が数学的に保証される
/// 規則）。境界（10%）ちょうどの場合は 10% 以上側（5% 刻み）を適用する。
///
/// TASK-8.2 の下限判定モジュール（#151〜#153）がマージされ次第、
/// こちらの実装は削除し公開 API へ委譲する（本ファイル冒頭コメント参照）。
fn floor_round(ratio_percent: f64) -> f64 {
    if !ratio_percent.is_finite() || ratio_percent < 0.0 {
        return 0.0;
    }
    let step = if ratio_percent >= 10.0 { 5.0 } else { 1.0 };
    (ratio_percent / step).floor() * step
}

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

fn measure_tiled_f32(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a = rng.fill_vec(size * size);
    let b = rng.fill_vec(size * size);

    let measurement = bench_run(config, || {
        gemm.run_tiled_f32(&a, &b, size as u32, size as u32, size as u32)
            .expect("tiled f32 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

/// f32 最良経路（WMMA(TF32) opt。共有メモリ・タイル最適化版が利用不能
/// な場合は `CudaGemm::run_wmma_tf32` 内部で基本版 WMMA(TF32) へ自動
/// フォールバックする。`gemm.rs::run_wmma_tf32` 冒頭ドキュメンテーション
/// コメント「TASK-11.1d（#63）フォールバック方針」参照）。
fn measure_wmma_tf32(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a = rng.fill_vec(size * size);
    let b = rng.fill_vec(size * size);

    let measurement = bench_run(config, || {
        gemm.run_wmma_tf32(&a, &b, size as u32, size as u32, size as u32)
            .expect("WMMA(TF32) GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

fn measure_wmma_f16(gemm: &CudaWmmaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f16> = rng.fill_vec_f16(size * size);
    let b: Vec<f16> = rng.fill_vec_f16(size * size);

    let measurement = bench_run(config, || {
        gemm.run_f16(&a, &b, size as u32, size as u32, size as u32)
            .expect("WMMA f16 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

/// `mma.sync` 経路のみ H2D/D2H 転送・出力バッファ確保を計測区間の外へ
/// 出し、GPU 実行（カーネル起動 + 同期）のみを計測する（PR #255 レビュー
/// 指摘への対処。`gemm_mma_bench.rs::measure_mma_f16` と同じ判断を踏襲。
/// tiled f32・WMMA(TF32)・WMMA f16 の 3 経路は転送込みで計測するため、
/// mma_over_* 比は厳密な apples-to-apples 比較ではなく mma 側に有利な
/// 方向へ偏る。`docs/perf/cuda-floor-remeasurement.md` へ転記する際は
/// この注記も一緒に残すこと）。
fn measure_mma_f16(gemm: &CudaMmaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f16> = rng.fill_vec_f16(size * size);
    let b: Vec<f16> = rng.fill_vec_f16(size * size);

    let (a_dev, b_dev) = gemm
        .upload_f16(&a, &b)
        .expect("mma.sync f16 upload must succeed on CUDA-equipped runner");
    let mut c_dev = gemm
        .alloc_output_f16(size as u32, size as u32)
        .expect("mma.sync f16 output allocation must succeed on CUDA-equipped runner");

    let measurement = bench_run(config, || {
        gemm.launch_f16(
            &a_dev,
            &b_dev,
            &mut c_dev,
            size as u32,
            size as u32,
            size as u32,
        )
        .expect("mma.sync f16 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

/// f32 最良値（`WMMA(TF32) opt` 実測。無ければ tiled、両方無ければ
/// `None`）を選ぶ。`gemm_auto.rs::CudaGemmAuto` は現状 f32 を常に tiled
/// へディスパッチする決定表を採用しているが（TF32 経路は #62/#186 の
/// 実測・承認まで既定採用を保留）、本バイナリは REQ-8 の「最適化後下限」
/// 候補算出が目的のため、既定ディスパッチとは独立に到達可能な最良経路
/// （WMMA(TF32) opt）を直接計測する。
fn best_f32(tiled: Option<f64>, wmma_tf32: Option<f64>) -> Option<f64> {
    wmma_tf32.or(tiled)
}

/// f16 最良値（`mma.sync` 実測。無ければ WMMA f16、両方無ければ
/// `None`）を選ぶ（`docs/cuda-tensor-core-design.md` の到達目標経路）。
fn best_f16(wmma: Option<f64>, mma: Option<f64>) -> Option<f64> {
    mma.or(wmma)
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda cuda_floor_bench: CUDA driver unavailable ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!("backend-cuda cuda_floor_bench: CudaDevice::new failed ({other}); skipping.");
            return;
        }
    };

    // REQ-8「いずれも同一ハードウェア上の同一バックエンド比較」。
    // PoC-v2-3 の PyTorch 参照値は DGX Spark GB10 実測のため、実行機が
    // GB10 系でない場合は比率が参考値に留まる旨を明示する
    // （実装計画 3.1「GPU 名が GB10 系でない場合は警告行を出力」）。
    let gpu_name = device.name().to_string();
    if !gpu_name.to_ascii_uppercase().contains("GB10") {
        println!(
            "WARNING: GPU name '{gpu_name}' does not match PoC-v2-3 measurement \
             machine (DGX Spark GB10). PyTorch reference ratios below are \
             REFERENCE VALUES ONLY, not a same-hardware comparison \
             (REQ-8 requires same-hardware comparison for floor confirmation)."
        );
    }
    println!(
        "device: name={gpu_name} compute_capability={:?}",
        device.compute_capability()
    );

    let tiled_gemm = match CudaGemm::new(&device) {
        Ok(g) => Some(g),
        Err(e) => {
            println!("tiled/WMMA(TF32) f32 kernel unavailable ({e}); f32 columns will be skipped.");
            None
        }
    };
    let wmma_gemm = match CudaWmmaGemm::new(&device) {
        Ok(g) => Some(g),
        Err(e) => {
            println!("WMMA f16 kernel unavailable ({e}); wmma_f16 column will be skipped.");
            None
        }
    };
    let mma_gemm = match CudaMmaGemm::new(&device) {
        Ok(g) => Some(g),
        Err(e) => {
            println!("mma.sync f16 kernel unavailable ({e}); mma_f16 column will be skipped.");
            None
        }
    };

    if tiled_gemm.is_none() && wmma_gemm.is_none() && mma_gemm.is_none() {
        println!(
            "backend-cuda cuda_floor_bench: no kernel path available in this environment \
             (NVRTC unavailable or device unsupported); nothing to measure. \
             See docs/perf/cuda-floor-remeasurement.md."
        );
        return;
    }

    // 判定対象形状（2048/4096）の f32/f16 最良経路比率の最小値を追跡する
    // （REQ-8「演算律速域〈M=N=K=2048・4096〉の実測比率の最小値を採る」）。
    let mut min_f32_ratio_percent: Option<f64> = None;
    let mut min_f16_ratio_percent: Option<f64> = None;

    for size in [REFERENCE_ONLY_SIZE, 2048, 4096] {
        let config = MeasurementConfig::default();

        let tiled = tiled_gemm
            .as_ref()
            .map(|g| measure_tiled_f32(g, size, &config));
        let wmma_tf32 = tiled_gemm
            .as_ref()
            .map(|g| measure_wmma_tf32(g, size, &config));
        let wmma_f16 = wmma_gemm
            .as_ref()
            .map(|g| measure_wmma_f16(g, size, &config));
        let mma_f16 = mma_gemm.as_ref().map(|g| measure_mma_f16(g, size, &config));

        let f32_best = best_f32(tiled, wmma_tf32);
        let f16_best = best_f16(wmma_f16, mma_f16);

        let fmt = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.4}"));
        let f32_ratio_percent = f32_best.map(|v| v / pytorch_f32_tflops(size) * 100.0);
        let f16_ratio_percent = f16_best.map(|v| v / pytorch_f16_tflops(size) * 100.0);
        let fmt_ratio = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.2}%"));

        println!(
            "size={size} tiled_f32_tflops={} wmma_tf32_tflops={} wmma_f16_tflops={} \
             mma_f16_tflops={} f32_best_over_pytorch={} f16_best_over_pytorch={} \
             (pytorch_f32={:.4} pytorch_f16={:.4}, PoC-v2-3)",
            fmt(tiled),
            fmt(wmma_tf32),
            fmt(wmma_f16),
            fmt(mma_f16),
            fmt_ratio(f32_ratio_percent),
            fmt_ratio(f16_ratio_percent),
            pytorch_f32_tflops(size),
            pytorch_f16_tflops(size),
        );

        if JUDGED_SIZES.contains(&size) {
            if let Some(r) = f32_ratio_percent {
                min_f32_ratio_percent = Some(min_f32_ratio_percent.map_or(r, |m: f64| m.min(r)));
            }
            if let Some(r) = f16_ratio_percent {
                min_f16_ratio_percent = Some(min_f16_ratio_percent.map_or(r, |m: f64| m.min(r)));
            }
        }
    }

    println!(
        "---\n\
         judged shapes (REQ-8): M=N=K in {JUDGED_SIZES:?} (size={REFERENCE_ONLY_SIZE} is \
         reference-only, excluded from candidate floor)"
    );
    match min_f32_ratio_percent {
        Some(r) => println!(
            "CUDA f32 candidate optimized floor (rounding rule applied to min ratio {r:.2}%) = {:.0}% \
             (current provisional REQ-8 value: 40%)",
            floor_round(r)
        ),
        None => println!(
            "CUDA f32 candidate optimized floor: n/a (no judged-size measurement available in this environment)"
        ),
    }
    match min_f16_ratio_percent {
        Some(r) => println!(
            "CUDA f16 candidate optimized floor (rounding rule applied to min ratio {r:.2}%) = {:.0}% \
             (current provisional REQ-8 value: 40%)",
            floor_round(r)
        ),
        None => println!(
            "CUDA f16 candidate optimized floor: n/a (no judged-size measurement available in this environment)"
        ),
    }
    println!(
        "NOTE: candidate floor values are NOT final. Final floor confirmation is a human \
         decision (TASK-8.3 担当: 共同（計測実行は Claude Code、下限値の最終確定は人間）). \
         Transcribe this output into docs/perf/cuda-floor-remeasurement.md and hand off to #158 \
         (TASK-8.3d)."
    );
}

#[cfg(test)]
mod tests {
    use super::floor_round;

    // `docs/spec/04-requirements.md`「丸め規則の統一（旧 #4 の解消）」節の
    // 例（10.3%→10%・26.6%→25%・境界 10%→10%）との突合。TASK-8.2 モジュール
    // （#151〜#153）が未マージのため、本ファイルへインライン実装した
    // 丸め規則の正しさをここで検証する（実装計画 §5「丸め規則をインライン
    // 実装した場合: 仕様例との突合をレビューで確認」）。
    #[test]
    fn floor_round_matches_spec_examples() {
        assert_eq!(floor_round(10.3), 10.0);
        assert_eq!(floor_round(26.6), 25.0);
        assert_eq!(floor_round(1.9), 1.0);
        assert_eq!(floor_round(10.0), 10.0);
        assert_eq!(floor_round(9.9999), 9.0);
        assert_eq!(floor_round(5.3), 5.0);
        assert_eq!(floor_round(23.2), 20.0);
    }

    #[test]
    fn floor_round_is_non_decreasing_across_the_10_percent_boundary() {
        // v1 の非単調性（16.9%→15%、17.0%→10% のような逆転）が v2 で
        // 解消されていることの回帰確認（`04-requirements.md` 該当節）。
        let mut prev = 0.0_f64;
        let mut r = 0.0_f64;
        while r <= 50.0 {
            let floored = floor_round(r);
            assert!(
                floored >= prev,
                "floor_round は非減少であるべき: r={r} floored={floored} prev={prev}"
            );
            prev = floored;
            r += 0.1;
        }
    }

    #[test]
    fn floor_round_rejects_non_finite_and_negative_input() {
        assert_eq!(floor_round(f64::NAN), 0.0);
        assert_eq!(floor_round(f64::INFINITY), 0.0);
        assert_eq!(floor_round(-5.0), 0.0);
    }
}
