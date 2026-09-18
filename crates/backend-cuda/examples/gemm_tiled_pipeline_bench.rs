//! tiled f32（本番既定経路）vs tiled pipeline（cp.async 3 stage・既定）vs
//! tiled pipeline（4 stage・オンデマンドコンパイル）の 3 経路比較ベンチ
//! 実測バイナリ（イシュー #1033）。
//!
//! 受け入れ条件「N=4096 での改善値の記録（5 回計測中央値）」の実測手段。
//! 計測コアは `bench-harness::run`（warmup 20 回以上・計測 20 回以上・
//! 中央値/Q1/Q3。TASK-8.1）を使う（`gemm_mma_bench.rs` と同じ計測コア。
//! `.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」の上位互換
//! として使う）。
//!
//! `examples/` に置くのは、通常の `cargo test`／CI では実行されず、
//! ビルド検証（`cargo build --workspace --all-targets`）のみが CI で走る
//! ようにするため（self-hosted runner をベンチ実行で占有しない。`ci.md`）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-cuda --example gemm_tiled_pipeline_bench --release
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載・cp.async 非対応（sm_80 未満）環境では、各
//! 経路の初期化失敗を検出した時点でその経路をスキップし理由を表示する
//! （`gemm_mma_bench.rs` の環境適応分岐と同じ判断。本実装セッションの
//! 実行環境が実際にこの経路を通る）。実測値は
//! `docs/perf/cuda-gemm-tiled-pipeline.md` の記録テンプレへ転記する。
//!
//! tiled pipeline 経路は `n`/`k` が 4 の倍数の形状のみ対応する
//! （`kernels_tiled_pipeline.rs` 冒頭コメント「整列制約」）。本ベンチの
//! 形状はすべてこの制約を満たす正方形状のみを使う。
//!
//! ## 計測区間の統一（codex-review P2／Cursor Bugbot 指摘。PR #1071 対応）
//!
//! 3 stage（既定）・4 stage（オンデマンドコンパイル）の両方を
//! `launch_tiled_pipeline_f32`（GPU 実行のみ。H2D/D2H・出力バッファ確保は
//! 計測区間外）で揃えて計測する。転送込みの `run_tiled_pipeline_f32`
//! （`tiled_f32` 本番経路と同じ計測区間）と GPU-only 経路
//! （`launch_tiled_pipeline_f32`）は測る対象が異なり、異なる区間の
//! TFLOPS を同じ比率へ混ぜると「転送有無の違い」が「stage 数増加による
//! 改善」として誤計上される（以前の実装の不具合）。そのため本ベンチは
//! 比較を 2 段に分ける:
//!
//! 1. 転送込み同士: `tiled_f32`（本番既定経路） vs `pipeline3`（既定
//!    3 stage・転送込み）。`pipeline3_over_tiled` はこの区間の比率。
//! 2. GPU-only 同士: `pipeline3_gpu_only`（既定 3 stage） vs
//!    `pipeline4_gpu_only`（4 stage）。`pipeline4_over_pipeline3_gpu_only`
//!    はこの区間の比率で、cp.async ステージ数の増加そのものの効果を表す。
//!
//! イシュー #1343: `pipeline128x64_gpu_only`（128×64×16・8×4 レジスタ
//! ブロック・A フラグメント XOR スウィズル。既定 3 stage）を上記 2. と
//! 同じ GPU-only 区間へ追加する。`pipeline128x64_over_pipeline3_gpu_only`
//! が演算密度 2 倍化の効果を表す一次データであり、GB10 実機での結線可否
//! 判断（兄弟イシュー #1344）の入力になる。本イシュー（#1343）自体は
//! 実測を行わない（`docs/perf/cuda-gemm-tiled-pipeline.md`「#1343
//! 128×64×16 候補の追加」節に未実測の旨を明記する）。
//!
//! イシュー #1976: TMA（cp.async.bulk.tensor）ロード経路 Stage 1
//! （`None`／`B64` swizzle。#1975・設計
//! `docs/backend-cuda-tma-gemm-load-design.md` §6 ゲート C）を
//! `pipeline3_gpu_only`／`pipeline128x64_gpu_only` と同じ GPU-only 区間
//! （[`measure_tiled_pipeline_gpu_only`] と同一の計測コア。H2D/D2H・
//! 出力バッファ確保は計測区間外）へ追加する。**テンソルマップ生成
//! （`gemm.rs::tma_tiled_pipeline::encode_tensor_map_2d_f32`。a_map／
//! b_map の 2 回・`device_ptr(stream)` 経由のホスト側 driver 接触を伴う）
//! は計測区間外**——本イシューで追加した
//! `CudaGemm::prepare_tiled_pipeline_tma_maps`（事前 encode。計測ループの
//! 外で 1 回だけ呼ぶ）・`CudaGemm::launch_tiled_pipeline_tma_f32_prepared`
//! （事前 encode 済みテンソルマップを使ったカーネル起動のみ。計測ループの
//! 内側で繰り返し呼ぶ）の 2 分割 API（`internal-diagnostics` ゲート）を
//! [`measure_tiled_pipeline_tma_gpu_only`] が使うことで、他の GPU-only 列
//! （`pipeline3_gpu_only`／`pipeline128x64_gpu_only`）と同じく「ロード
//! 命令の実行時間のみ」を計測する（テンソルマップ生成コストが計測対象へ
//! 混入する非対称性は解消済み）。既存の一括 API
//! `CudaGemm::launch_tiled_pipeline_tma_f32`（毎回内部で encode する版）
//! は挙動を変えず、上記 2 API への委譲として残る（呼び出し元が事前
//! encode を意識しない用途向け）。主判定 `tma_none_over_pipeline3_gpu_only`
//! （同一 64×64 タイル・同一 stage 数の cp.async 版との比較——タイル
//! 構成・段数を揃えた「ロード命令の違いのみ」の比較）で、
//! `tma_none_over_pipeline128x64_gpu_only`（タイル構成が異なるため
//! 参考値）を併記する。`B64` swizzle 腕は
//! `cpu_cuda_tiled_pipeline_tma_parity.rs` が「仮説段階」と明記する
//! 物理配置仮説の実機性能を記録するためのものであり、本イシューでも
//! `select_tiled_f32_kernel`／`CudaGemm::new` への本番結線は行わない
//! （`docs/backend-cuda-tma-gemm-load-design.md` §6「スコープ外」）。

use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run as bench_run};
use fandhe_ai_backend_cuda::{
    CudaDevice, CudaError, CudaGemm, TiledPipelineFunction, TmaSwizzleA, TmaTiledPipelineFunction,
};

/// 128×64×16 pipeline カーネル（イシュー #1343）の既定ステージ数。
/// `kernels_tiled_pipeline_128x64::TP128_DEFAULT_STAGES` と同値（本クレート
/// 内部定数は非公開のためベンチ側で値を複製する。`STAGE_3` と同じ判断）。
const STAGE_128X64_DEFAULT: u32 = 3;

/// 決定的シード（`gemm_mma_bench.rs` と同一値。過去 PoC・他ベンチと同じ
/// 入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

/// 既定（本番オブジェクトが保持する）ステージ数。GPU-only 比較用に
/// `CudaGemm::compile_tiled_pipeline_variant` で同一段数のハンドルを
/// 別途オンデマンドコンパイルする（`kernels_tiled_pipeline.rs::
/// TP_DEFAULT_STAGES` と同値。本クレート内部定数は非公開のためベンチ側で
/// 値を複製する）。
const STAGE_3: u32 = 3;

/// 4 stage 変種の比較対象ステージ数。
const STAGE_4: u32 = 4;

fn tflops(size: usize, median_secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / median_secs / 1e12
}

/// tiled f32（本番既定経路。転送込み）を計測する。イシュー #1137 以降、
/// 本ベンチが使う形状（すべて 4 の倍数の正方形状）では内部で cp.async
/// パイプライン版へ分岐するため、この列は「本番ディスパッチが実際に
/// 呼び出し元へ返す性能」を表す（`measure_tiled_f32_classic` が固定
/// classic 版のベースラインを別途提供する）。
fn measure_tiled_f32(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let measurement = bench_run(config, || {
        gemm.run_tiled_f32(&a, &b, size as u32, size as u32, size as u32)
            .expect("tiled f32 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

/// tiled f32 classic 版（`kernels::TILED_F32` 固定・パイプライン非経由。
/// イシュー #1137）を、`measure_tiled_f32` と同じ計測区間（転送込み）で
/// 計測する。`#1137` 本番結線の before 値・A/B ベースラインとして使う
/// （`CudaGemm::run_tiled_f32_classic`。診断専用 API）。
fn measure_tiled_f32_classic(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let measurement = bench_run(config, || {
        gemm.run_tiled_f32_classic(&a, &b, size as u32, size as u32, size as u32)
            .expect("tiled f32 classic GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

/// tiled pipeline（既定 3 stage。転送込み）を計測する。`tiled_f32` と
/// 同じ計測区間（H2D/D2H 込み）のため `pipeline3_over_tiled` の分子として
/// 使ってよい。
fn measure_tiled_pipeline_default(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> f64 {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let measurement = bench_run(config, || {
        gemm.run_tiled_pipeline_f32(&a, &b, size as u32, size as u32, size as u32)
            .expect("tiled pipeline GEMM must succeed on cp.async-capable runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops(size, measurement.median_secs)
}

/// tiled pipeline（任意ステージ数。オンデマンドコンパイル済みハンドル経由。
/// GPU 実行のみを計測——H2D/D2H・出力バッファ確保は計測区間外。
/// `gemm_mma_bench.rs::measure_mma_f16` と同じ計測方針）を計測する。
/// 3 stage・4 stage いずれの比較にもこの関数を使い、計測区間を揃える
/// （モジュールコメント「計測区間の統一」参照）。
fn measure_tiled_pipeline_gpu_only(
    gemm: &CudaGemm,
    func: &TiledPipelineFunction,
    size: usize,
    config: &MeasurementConfig,
) -> Result<f64, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(size as u32, size as u32)?;

    // `bench_run` のクロージャは `FnMut()` （非 fallible）契約のため、
    // 計測中の CUDA 起動失敗をここで捕捉し、計測終了後に `Err` として
    // 返す（`gemm_wmma_tf32_staged_stages_bench.rs::measure_dyn_staged`
    // と同じ理由・同じ契約）。
    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_tiled_pipeline_f32(
            func,
            &a_dev,
            &b_dev,
            &mut c_dev,
            size as u32,
            size as u32,
            size as u32,
        ) {
            first_err = Some(e);
            return;
        }
        if let Err(e) = gemm.synchronize() {
            first_err = Some(e);
        }
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(tflops(size, measurement.median_secs))
}

/// TMA 版（`CudaGemm::launch_tiled_pipeline_tma_f32`。イシュー #1976）を
/// GPU 実行のみで計測する。[`measure_tiled_pipeline_gpu_only`] と同一の
/// 計測コア（`bench_run`・warmup/計測回数は同じ `MeasurementConfig`・
/// H2D/D2H と出力バッファ確保は計測区間外）を使い、GPU-only 列同士の
/// 比較が「転送有無の違い」を含まないようにする（モジュールコメント
/// 「計測区間の統一」参照）。テンソルマップ生成（`encode_tensor_map_2d_
/// f32`。ホスト側 driver 接触・ストリーム待機を伴う）は
/// `CudaGemm::prepare_tiled_pipeline_tma_maps` で計測ループの**外**に
/// 1 回だけ行い、ループ内は `launch_tiled_pipeline_tma_f32_prepared` の
/// カーネル起動＋`synchronize` のみとする（cp.async 版の GPU-only 列と
/// 同じく「起動＋同期」だけを測る。事前 encode の bit 同一性は
/// `tiled_pipeline_tma_prepared_matches_one_shot_launch_bit_exact` で担保。
/// tensor map キャッシュ自体は #1975／#1976 のスコープ外のまま）。
fn measure_tiled_pipeline_tma_gpu_only(
    gemm: &CudaGemm,
    func: &TmaTiledPipelineFunction,
    size: usize,
    config: &MeasurementConfig,
) -> Result<f64, CudaError> {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f32> = rng.fill_vec(size * size);
    let b: Vec<f32> = rng.fill_vec(size * size);

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b)?;
    let mut c_dev = gemm.alloc_output_f32(size as u32, size as u32)?;

    // テンソルマップの事前 encode（`encode_tensor_map_2d_f32`。ストリーム
    // 順序上の待機・`SyncOnDrop` によるイベント記録を伴う driver 接触）は
    // 計測ループの外で 1 回だけ行う（イシュー #1976。他の GPU-only 列
    // 〈`measure_tiled_pipeline_gpu_only`〉と同じく「ロード命令の実行時間
    // のみ」を計測するため）。
    let maps = gemm.prepare_tiled_pipeline_tma_maps(
        func,
        &a_dev,
        &b_dev,
        (size as u32, size as u32, size as u32),
    )?;

    // `bench_run` のクロージャは `FnMut()`（非 fallible）契約のため、
    // 計測中の CUDA 起動失敗をここで捕捉し、計測終了後に `Err` として
    // 返す（`measure_tiled_pipeline_gpu_only` と同じ理由・同じ契約）。
    let mut first_err: Option<CudaError> = None;
    let measurement = bench_run(config, || {
        if first_err.is_some() {
            return;
        }
        if let Err(e) = gemm.launch_tiled_pipeline_tma_f32_prepared(func, &maps, &mut c_dev) {
            first_err = Some(e);
            return;
        }
        if let Err(e) = gemm.synchronize() {
            first_err = Some(e);
        }
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(tflops(size, measurement.median_secs))
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_tiled_pipeline_bench: CUDA driver unavailable ({detail}); \
                 skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_tiled_pipeline_bench: CudaDevice::new failed ({other}); \
                 skipping."
            );
            return;
        }
    };

    let gemm = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) => {
            println!(
                "backend-cuda gemm_tiled_pipeline_bench: CudaGemm::new failed ({e}); nothing \
                 to measure. See docs/perf/cuda-gemm-tiled-pipeline.md."
            );
            return;
        }
    };

    if !gemm.tiled_pipeline_available() {
        println!(
            "tiled pipeline kernel unavailable ({:?}); pipeline columns will be skipped. \
             tiled_f32 column is still measured below for reference.",
            gemm.tiled_pipeline_unavailable_reason()
        );
    }

    // 3 stage・4 stage いずれも GPU-only 比較用にオンデマンドコンパイルする
    // （本番オブジェクトの初期化コストには影響しない独立経路。
    // `kernels_tiled_pipeline.rs` 冒頭コメント「stages=4 版はベンチ用途に
    // 限りオンデマンドでコンパイルする」参照。3 stage 側も同一経路で
    // コンパイルし直すのは、GPU-only 計測区間を 4 stage 側と厳密に揃える
    // ため——`CudaGemm::new` が保持する既定ハンドルへは `&self` 経由でしか
    // 到達できず、safe API の型で GPU-only 用途に流用できないため）。
    let stage3_func = match CudaGemm::compile_tiled_pipeline_variant(&device, STAGE_3) {
        Ok(f) => Some(f),
        Err(e) => {
            println!(
                "tiled pipeline (stages={STAGE_3}) on-demand compilation failed ({e}); \
                 GPU-only stage3 column will be skipped."
            );
            None
        }
    };
    let stage4_func = match CudaGemm::compile_tiled_pipeline_variant(&device, STAGE_4) {
        Ok(f) => Some(f),
        Err(e) => {
            println!(
                "tiled pipeline (stages={STAGE_4}) compilation failed ({e}); stage4 column \
                 will be skipped."
            );
            None
        }
    };

    // イシュー #1343: 128×64×16 pipeline カーネル（opt-in・#1344 の GB10
    // 実機比較用導線）。`measure_tiled_pipeline_gpu_only` は
    // `TiledPipelineFunction` の内部タイル構成タグ（`TiledPipelineTile`）を
    // 見て自身の grid/block 構成を導出するため、64×64 版と同じ関数を
    // そのまま再利用できる（計測区間は GPU-only 同士で統一。モジュール
    // コメント「計測区間の統一」参照）。
    let pipeline128x64_func =
        match CudaGemm::compile_tiled_pipeline_128x64_variant(&device, STAGE_128X64_DEFAULT) {
            Ok(f) => Some(f),
            Err(e) => {
                println!(
                    "tiled pipeline 128x64 (stages={STAGE_128X64_DEFAULT}) compilation failed \
                     ({e}); pipeline128x64 column will be skipped."
                );
                None
            }
        };

    // イシュー #1976: TMA 版（`None`／`B64` swizzle）を GPU-only 比較用に
    // オンデマンドコンパイルする（本番オブジェクトの初期化コストには
    // 影響しない独立経路。上記 `stage3_func`／`pipeline128x64_func` と
    // 同じ判断）。compute capability 9.0 未満（TMA 前提）や NVRTC
    // コンパイル失敗時は `CudaError::TiledPipelineUnavailable` 等を返し、
    // その旨を表示して該当列を skip する。
    let tma_none_func =
        match CudaGemm::compile_tiled_pipeline_tma_variant(&device, TmaSwizzleA::None) {
            Ok(f) => Some(f),
            Err(e) => {
                println!(
                    "tiled pipeline TMA (swizzle=None) compilation failed ({e}); tma_none column \
                 will be skipped."
                );
                None
            }
        };
    let tma_b64_func = match CudaGemm::compile_tiled_pipeline_tma_variant(&device, TmaSwizzleA::B64)
    {
        Ok(f) => Some(f),
        Err(e) => {
            println!(
                "tiled pipeline TMA (swizzle=B64) compilation failed ({e}); tma_b64 column \
                 will be skipped."
            );
            None
        }
    };

    for size in [256usize, 512, 1024, 2048, 4096] {
        let config = MeasurementConfig::default();

        let tiled = measure_tiled_f32(&gemm, size, &config);
        let tiled_classic = measure_tiled_f32_classic(&gemm, size, &config);
        let pipeline3 = gemm
            .tiled_pipeline_available()
            .then(|| measure_tiled_pipeline_default(&gemm, size, &config));
        let pipeline3_gpu_only = stage3_func.as_ref().and_then(|func| {
            match measure_tiled_pipeline_gpu_only(&gemm, func, size, &config) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!("size={size}: stage3 GPU-only measurement failed ({e}); skipping.");
                    None
                }
            }
        });
        let pipeline4_gpu_only = stage4_func.as_ref().and_then(|func| {
            match measure_tiled_pipeline_gpu_only(&gemm, func, size, &config) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!("size={size}: stage4 GPU-only measurement failed ({e}); skipping.");
                    None
                }
            }
        });
        let pipeline128x64_gpu_only = pipeline128x64_func.as_ref().and_then(|func| {
            match measure_tiled_pipeline_gpu_only(&gemm, func, size, &config) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!(
                        "size={size}: pipeline128x64 GPU-only measurement failed ({e}); \
                         skipping."
                    );
                    None
                }
            }
        });
        // イシュー #1976: TMA 版（`None`／`B64` swizzle）の GPU-only 列。
        let tma_none_gpu_only = tma_none_func.as_ref().and_then(|func| {
            match measure_tiled_pipeline_tma_gpu_only(&gemm, func, size, &config) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!("size={size}: tma_none GPU-only measurement failed ({e}); skipping.");
                    None
                }
            }
        });
        let tma_b64_gpu_only = tma_b64_func.as_ref().and_then(|func| {
            match measure_tiled_pipeline_tma_gpu_only(&gemm, func, size, &config) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!("size={size}: tma_b64 GPU-only measurement failed ({e}); skipping.");
                    None
                }
            }
        });

        let fmt = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.4}"));
        // 転送込み同士（`tiled_f32` vs `pipeline3`）の比率。
        let ratio_over_tiled = |num: Option<f64>| match num {
            Some(n) if tiled != 0.0 => format!("{:.4}", n / tiled),
            _ => "n/a".to_string(),
        };
        // イシュー #1137: 本番ディスパッチ（`tiled`。整列形状ではパイプライン
        // 版へ分岐）と固定 classic 版（`tiled_classic`）の比率。#1137 の
        // 採否判断における「同一プロトコルの before/after」に相当する。
        let dispatch_over_classic = if tiled_classic != 0.0 {
            format!("{:.4}", tiled / tiled_classic)
        } else {
            "n/a".to_string()
        };
        // GPU-only 同士（`pipeline3_gpu_only` vs `pipeline4_gpu_only`）の
        // 比率。stage 数増加そのものの効果を転送有無の差から切り離して表す
        // （codex-review P2／Cursor Bugbot 指摘の是正対象）。
        let pipeline4_over_pipeline3_gpu_only = match (pipeline3_gpu_only, pipeline4_gpu_only) {
            (Some(p3), Some(p4)) if p3 != 0.0 => format!("{:.4}", p4 / p3),
            _ => "n/a".to_string(),
        };
        // イシュー #1343: 128×64 版（GPU-only）と既定 3 stage の 64×64 版
        // （GPU-only）の比率。演算密度 2 倍化そのものの効果を、#1344 が
        // GB10 実機で結線可否を判断するための一次データとして記録する
        // （本イシューでは未実測であることを README・docs 側に明記する）。
        let pipeline128x64_over_pipeline3_gpu_only =
            match (pipeline3_gpu_only, pipeline128x64_gpu_only) {
                (Some(p3), Some(p128)) if p3 != 0.0 => format!("{:.4}", p128 / p3),
                _ => "n/a".to_string(),
            };
        // イシュー #1976: TMA 版（GPU-only）と cp.async pipeline 版
        // （GPU-only）の比率。`tma_*_over_pipeline3_gpu_only` が主判定
        // （同一 64×64 タイル・同一 stage 数との比較。ロード命令の違いの
        // みを表す）、`tma_*_over_pipeline128x64_gpu_only` は参考値
        // （タイル構成が異なる）。
        let tma_none_over_pipeline3_gpu_only = match (pipeline3_gpu_only, tma_none_gpu_only) {
            (Some(p3), Some(tma)) if p3 != 0.0 => format!("{:.4}", tma / p3),
            _ => "n/a".to_string(),
        };
        let tma_none_over_pipeline128x64_gpu_only =
            match (pipeline128x64_gpu_only, tma_none_gpu_only) {
                (Some(p128), Some(tma)) if p128 != 0.0 => format!("{:.4}", tma / p128),
                _ => "n/a".to_string(),
            };
        let tma_b64_over_pipeline3_gpu_only = match (pipeline3_gpu_only, tma_b64_gpu_only) {
            (Some(p3), Some(tma)) if p3 != 0.0 => format!("{:.4}", tma / p3),
            _ => "n/a".to_string(),
        };
        let tma_b64_over_pipeline128x64_gpu_only = match (pipeline128x64_gpu_only, tma_b64_gpu_only)
        {
            (Some(p128), Some(tma)) if p128 != 0.0 => format!("{:.4}", tma / p128),
            _ => "n/a".to_string(),
        };

        println!(
            "size={size} tiled_f32_tflops={:.4} tiled_f32_classic_tflops={:.4} \
             dispatch_over_classic={} pipeline3_tflops={} \
             pipeline3_over_tiled={} | pipeline3_gpu_only_tflops={} \
             pipeline4_gpu_only_tflops={} pipeline4_over_pipeline3_gpu_only={} | \
             pipeline128x64_gpu_only_tflops={} pipeline128x64_over_pipeline3_gpu_only={} | \
             tma_none_gpu_only_tflops={} tma_none_over_pipeline3_gpu_only={} \
             tma_none_over_pipeline128x64_gpu_only={} | tma_b64_gpu_only_tflops={} \
             tma_b64_over_pipeline3_gpu_only={} tma_b64_over_pipeline128x64_gpu_only={}",
            tiled,
            tiled_classic,
            dispatch_over_classic,
            fmt(pipeline3),
            ratio_over_tiled(pipeline3),
            fmt(pipeline3_gpu_only),
            fmt(pipeline4_gpu_only),
            pipeline4_over_pipeline3_gpu_only,
            fmt(pipeline128x64_gpu_only),
            pipeline128x64_over_pipeline3_gpu_only,
            fmt(tma_none_gpu_only),
            tma_none_over_pipeline3_gpu_only,
            tma_none_over_pipeline128x64_gpu_only,
            fmt(tma_b64_gpu_only),
            tma_b64_over_pipeline3_gpu_only,
            tma_b64_over_pipeline128x64_gpu_only,
        );
    }
}
