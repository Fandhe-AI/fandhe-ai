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
//!
//! ## PyTorch 参照値の再計測（PR #349 codex-review 指摘 P1 対応）
//!
//! REQ-8「いずれも同一ハードウェア上の同一バックエンド比較」を満たすため、
//! `docs/spec/03-poc/poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py`
//! （または同一形状・同一プロトコルの再計測手段）を**本バイナリと同一
//! 実機で**実行し、得られた値を以下の環境変数で注入できる:
//!
//! - `CUDA_FLOOR_BENCH_PYTORCH_F32_{512,2048,4096}` / `_F16_{512,2048,4096}`:
//!   再計測した TFLOPS 値（正の有限浮動小数点数）。再計測は
//!   `gemm_bench_torch_cuda.py` と同一の計測境界（下記「計測境界の統一」
//!   節）で行うこと
//! - `CUDA_FLOOR_BENCH_PYTORCH_SOURCE`: 再計測値の出所を明示する文字列
//!   （例: `"poc-v2-3-cuda-gemm/code/pytorch/gemm_bench_torch_cuda.py 実行,
//!   2026-08-08, 同一 GB10 個体"`）。値の注入だけでは根拠不足のため、出所
//!   文字列が非空でない限り注入値は無視し組み込み固定値へフォールバックする
//!
//! 判定対象形状（2048/4096）の f32・f16 それぞれについて、上記env変数が
//! 両サイズとも有効な値を持つ場合のみ「同一実機再計測」を根拠とした
//! 正式な `candidate optimized floor` を出力する。未注入または一部のみ
//! 注入の場合、GPU 名が GB10 系であっても固定値（PoC-v2-3）は同一実機
//! 再計測の代替にならないため `candidate optimized floor: n/a` とし、
//! 参考比率のみを表示する（codex-review 指摘「GPU 名の部分一致では
//! 同一ハードウェア比較を保証できない」対応。GB10 名一致は WARNING 抑制
//! のみに用い、候補下限の許可条件には使わない）。
//!
//! ## 計測境界の統一（PR #349 codex-review 再指摘 P1 対応）
//!
//! `gemm_bench_torch_cuda.py`（実測確認済み。同スクリプト L36-51）は
//! 計測ループの**外側**で入力テンソルを GPU 上に生成し、ループ内では
//! `torch.cuda.synchronize()` → `torch.matmul(a, b)` →
//! `torch.cuda.synchronize()` のみを計測する。ホスト→デバイス転送・
//! 反復ごとの入力確保は計測区間に含まれない。
//!
//! 旧実装（本コメント追加前の版）は tiled f32・WMMA(TF32)・WMMA f16 の
//! 3 経路を H2D 転送＋出力バッファ確保＋カーネル実行＋D2H 回収込みで計測
//! しており、PyTorch 参照計測より広い区間を計測していた（codex-review
//! 再指摘 P1「計測範囲の不一致」）。本バイナリは 4 経路すべて（tiled f32・
//! WMMA(TF32)・WMMA f16・`mma.sync` f16）で PyTorch 参照計測と同じ「入力
//! 事前配置＋カーネル起動＋同期のみ」の境界に統一する（`measure_*` 各
//! 関数が `upload_*`／`alloc_output_*`／`launch_*` の分割 API
//! （`gemm.rs::CudaGemm::upload_f32`／`launch_tiled_f32`／
//! `launch_wmma_tf32`、`gemm_wmma.rs::CudaWmmaGemm::upload_f16`／
//! `launch_f16`、`gemm_mma.rs::CudaMmaGemm::upload_f16`／`launch_f16`）を
//! 使い、H2D/D2H・バッファ確保をループ外に出す）。4 経路の計測境界が
//! 揃ったことで、f16 candidate floor の算出根拠を `wmma_f16` と
//! `mma_f16` の実測比較（[`best_of`]）へ戻す（旧実装が計測範囲の不一致を
//! 理由に `mma_f16` を除外していた判断はこの統一により前提が失われた
//! ため撤回する。[`f16_candidate_floor_value`] ドキュメンテーション
//! コメント参照）。
//!
//! ## 中央値・Q1・Q3 の出力（PR #349 codex-review 指摘 P1 対応）
//!
//! `bench_harness::run` の計測プロトコル（TASK-8.1。warmup 20 回・計測
//! 20 回・中央値/Q1/Q3）が返す四分位値を破棄すると、実測のばらつき・
//! 再現性を記録・検証できなくなる。本バイナリの各経路別 TFLOPS 出力は
//! `<中央値>(q1=<Q1由来値>,q3=<Q3由来値>)` の形式で 3 値を並記する
//! （[`TflopsSample`] 参照。経路選択・候補下限の算出ロジック自体は
//! 引き続き中央値のみを根拠とする。変更は出力・記録範囲の拡張のみ）。
//! `docs/perf/cuda-floor-remeasurement.md`「経路×形状 TFLOPS 実測」表にも
//! 中央値・Q1・Q3 の記入欄がある。

use backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaMmaGemm, CudaWmmaGemm};
use bench_harness::rng::Xorshift64Star;
use bench_harness::{Measurement, MeasurementConfig, run as bench_run};
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
/// 2 表から転記した過去実測固定値。GPU 名一致だけでは同一実機比較を
/// 保証できないため（PR #349 codex-review 指摘 P1）、この固定値を根拠に
/// 正式な candidate floor は出さない。あくまで参考比率の分母として使う。
fn pytorch_f32_fixed(size: usize) -> f64 {
    match size {
        512 => 7.8803,
        2048 => 17.4241,
        4096 => 17.7774,
        _ => f64::NAN,
    }
}

fn pytorch_f16_fixed(size: usize) -> f64 {
    match size {
        512 => 17.1898,
        2048 => 91.2115,
        4096 => 97.6308,
        _ => f64::NAN,
    }
}

/// 同一実機再計測値の出所文字列（`CUDA_FLOOR_BENCH_PYTORCH_SOURCE`）。
/// 非空でない限り注入値は使わない（数値だけでは根拠不足。モジュール冒頭
/// ドキュメンテーションコメント「PyTorch 参照値の再計測」参照）。
fn measured_source() -> Option<String> {
    std::env::var("CUDA_FLOOR_BENCH_PYTORCH_SOURCE")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// `CUDA_FLOOR_BENCH_PYTORCH_{F32,F16}_{size}` を読み、正の有限値として
/// パースできればそれを返す。未設定なら `None`。設定されているのに
/// パース不能な場合は診断のためのフォールバックではなく警告を出す
/// （advisor 指摘: 無音フォールバックはゲートが閉じた理由を追いにくい）。
fn env_override(kind: &str, size: usize) -> Option<f64> {
    let key = format!("CUDA_FLOOR_BENCH_PYTORCH_{kind}_{size}");
    match std::env::var(&key) {
        Ok(raw) => match raw.trim().parse::<f64>() {
            Ok(v) if v.is_finite() && v > 0.0 => Some(v),
            _ => {
                println!(
                    "WARNING: env var {key}='{raw}' is set but not a positive finite number; \
                     ignoring override for this size."
                );
                None
            }
        },
        Err(_) => None,
    }
}

/// PyTorch 参照値と、それが「本バイナリと同一実機で今回再計測された値」
/// かどうかのフラグを返す。`measured` が `true` のときのみ REQ-8 の
/// 同一実機比較要件を満たす（`source` が非空の場合に限り env override を
/// 採用する。`measured_source` 参照）。
fn pytorch_f32_ref(size: usize, source: &Option<String>) -> (f64, bool) {
    if source.is_some()
        && let Some(v) = env_override("F32", size)
    {
        return (v, true);
    }
    (pytorch_f32_fixed(size), false)
}

fn pytorch_f16_ref(size: usize, source: &Option<String>) -> (f64, bool) {
    if source.is_some()
        && let Some(v) = env_override("F16", size)
    {
        return (v, true);
    }
    (pytorch_f16_fixed(size), false)
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

fn tflops(size: usize, secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / secs / 1e12
}

/// 単一形状・単一経路の計測結果を TFLOPS 換算した中央値・Q1・Q3。
///
/// `bench_harness::run`（TASK-8.1 計測プロトコル。warmup 20 回・計測 20 回・
/// 中央値/Q1/Q3）が返す `Measurement` の時間ドメイン値を `tflops` で
/// TFLOPS ドメインへ変換して保持する。中央値だけを抜き出して Q1/Q3 を
/// 捨てると実測のばらつき・再現性を記録・検証できなくなるため
/// （PR #349 codex-review 指摘 P1「Q1/Q3 を破棄しており実測記録の契約を
/// 満たせない」対応）、`main` の出力行・
/// `docs/perf/cuda-floor-remeasurement.md` への転記の両方でこの 3 値を
/// 保持したまま伝播させる。
///
/// `tflops` は時間について単調減少（時間が短いほど TFLOPS は大きい）ため、
/// 時間ドメインの Q1（下位 25%＝速い方のサンプル）を変換した値は
/// TFLOPS ドメインでは中央値より大きい側、Q3（上位 25%＝遅い方）を
/// 変換した値は小さい側になりうる。フィールド名は変換元
/// （`q1_secs`/`q3_secs`）に揃え、時間ドメインの由来を追いやすくする
/// （TFLOPS ドメインでの大小関係をフィールド名に含めて誤解を招かない
/// ため）。
#[derive(Clone, Copy, Debug, PartialEq)]
struct TflopsSample {
    median: f64,
    tflops_from_q1_secs: f64,
    tflops_from_q3_secs: f64,
}

/// `bench_harness::run` が返す時間ドメインの `Measurement` を
/// `TflopsSample`（TFLOPS ドメイン）へ変換する唯一の変換経路。
/// 4 経路すべての `measure_*` 関数がこの関数を経由することで、
/// 変換ロジックの重複（コピー漏れによる変換忘れ）を防ぐ
/// （advisor 指摘: 4 箇所へインライン展開すると回帰時の抜け漏れの
/// 温床になる）。
fn tflops_sample(size: usize, measurement: &Measurement) -> TflopsSample {
    TflopsSample {
        median: tflops(size, measurement.median_secs),
        tflops_from_q1_secs: tflops(size, measurement.q1_secs),
        tflops_from_q3_secs: tflops(size, measurement.q3_secs),
    }
}

/// tiled f32 経路を計測する。H2D 転送・出力バッファ確保はループ外で
/// 済ませ、`gemm.rs::CudaGemm::launch_tiled_f32`（GPU 実行 + 同期のみ）を
/// 計測対象とする（PyTorch 参照計測〈`gemm_bench_torch_cuda.py`〉と同一
/// の境界に揃える。モジュール冒頭ドキュメンテーションコメント「計測境界
/// の統一」参照）。
fn measure_tiled_f32(gemm: &CudaGemm, size: usize, config: &MeasurementConfig) -> TflopsSample {
    let mut rng = Xorshift64Star::new(SEED);
    let a = rng.fill_vec(size * size);
    let b = rng.fill_vec(size * size);
    let (m, n, k) = (size as u32, size as u32, size as u32);

    let (a_dev, b_dev) = gemm
        .upload_f32(&a, &b)
        .expect("tiled f32 upload must succeed on CUDA-equipped runner");
    let mut c_dev = gemm
        .alloc_output_f32(m, n)
        .expect("tiled f32 output allocation must succeed on CUDA-equipped runner");

    let measurement = bench_run(config, || {
        gemm.launch_tiled_f32(&a_dev, &b_dev, &mut c_dev, m, n, k)
            .expect("tiled f32 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops_sample(size, &measurement)
}

/// f32 最良経路（WMMA(TF32) opt。共有メモリ・タイル最適化版が利用不能
/// な場合は `CudaGemm::run_wmma_tf32` 内部で基本版 WMMA(TF32) へ自動
/// フォールバックする。`gemm.rs::run_wmma_tf32` 冒頭ドキュメンテーション
/// コメント「TASK-11.1d（#63）フォールバック方針」参照）。
///
/// `CudaGemm::new` の成功は tiled/naive カーネルの準備のみを保証し、
/// WMMA(TF32)（opt・基本版とも）はオプションである。`run_wmma_tf32` 呼び出し
/// 時に `CudaError::WmmaUnavailable`（cc<8.0 や `mma.h` 非搭載環境で両方
/// 使用不能）として表面化しうるほか、opt 版使用時は形状検証・カーネル
/// 起動由来の他エラーも起こりうる（`gemm.rs::run_wmma_tf32` L594-598・
/// L641-645 参照）。エラー種別を区別せず panic させると動作している tiled
/// 計測結果まで失われるため、まず 1 回 `run_wmma_tf32`（転送込みの通常
/// 経路）で probe 実行しエラー時は本計測を行わず `None` を返してこの経路
/// のみ skip する（PR #349 Bugbot 指摘 High「WMMA path panics on skip」
/// 対応。呼び出し側 `main` は `Option::and_then` で受け、tiled 側の出力は
/// 継続する）。probe が選ぶ経路（opt／基本版）は
/// `gemm.rs::CudaGemm::launch_wmma_tf32` が同一の判定式
/// （`wmma_tf32_opt.is_some()`）で選ぶ経路と一致するため、probe 通過後の
/// launch-only 計測が異なるカーネルを起動することはない
/// （`launch_wmma_tf32` ドキュメンテーションコメント参照）。
///
/// H2D 転送・出力バッファ確保はループ外で済ませ、`launch_wmma_tf32`
/// （GPU 実行 + 同期のみ）を計測対象とする（PyTorch 参照計測と同一の
/// 境界に揃える。モジュール冒頭ドキュメンテーションコメント「計測境界の
/// 統一」参照）。
fn measure_wmma_tf32(
    gemm: &CudaGemm,
    size: usize,
    config: &MeasurementConfig,
) -> Option<TflopsSample> {
    let mut rng = Xorshift64Star::new(SEED);
    let a = rng.fill_vec(size * size);
    let b = rng.fill_vec(size * size);
    let (m, n, k) = (size as u32, size as u32, size as u32);

    if let Err(e) = gemm.run_wmma_tf32(&a, &b, m, n, k) {
        println!("WMMA(TF32) unavailable for size={size} ({e}); wmma_tf32 skipped for this size.");
        return None;
    }

    let (a_dev, b_dev) = gemm.upload_f32(&a, &b).expect(
        "WMMA(TF32) upload must succeed on CUDA-equipped runner (availability probed above)",
    );
    let mut c_dev = gemm.alloc_output_f32(m, n).expect(
        "WMMA(TF32) output allocation must succeed on CUDA-equipped runner (availability probed above)",
    );

    let measurement = bench_run(config, || {
        gemm.launch_wmma_tf32(&a_dev, &b_dev, &mut c_dev, m, n, k)
            .expect(
                "WMMA(TF32) GEMM must succeed on CUDA-equipped runner (availability probed above)",
            );
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    Some(tflops_sample(size, &measurement))
}

/// H2D 転送・出力バッファ確保はループ外で済ませ、
/// `gemm_wmma.rs::CudaWmmaGemm::launch_f16`（GPU 実行 + 同期のみ）を
/// 計測対象とする（PyTorch 参照計測と同一の境界に揃える。モジュール冒頭
/// ドキュメンテーションコメント「計測境界の統一」参照）。
fn measure_wmma_f16(gemm: &CudaWmmaGemm, size: usize, config: &MeasurementConfig) -> TflopsSample {
    let mut rng = Xorshift64Star::new(SEED);
    let a: Vec<f16> = rng.fill_vec_f16(size * size);
    let b: Vec<f16> = rng.fill_vec_f16(size * size);
    let (m, n, k) = (size as u32, size as u32, size as u32);

    let (a_dev, b_dev) = gemm
        .upload_f16(&a, &b)
        .expect("WMMA f16 upload must succeed on CUDA-equipped runner");
    let mut c_dev = gemm
        .alloc_output_f16(m, n)
        .expect("WMMA f16 output allocation must succeed on CUDA-equipped runner");

    let measurement = bench_run(config, || {
        gemm.launch_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)
            .expect("WMMA f16 GEMM must succeed on CUDA-equipped runner");
    })
    .expect("MeasurementConfig::default satisfies the 20/20 lower bound");
    tflops_sample(size, &measurement)
}

/// H2D/D2H 転送・出力バッファ確保を計測区間の外へ出し、GPU 実行
/// （カーネル起動 + 同期）のみを計測する（PR #255 レビュー指摘への対処。
/// `gemm_mma_bench.rs::measure_mma_f16` と同じ判断を踏襲）。4 経路すべて
/// （tiled f32・WMMA(TF32)・WMMA f16・`mma.sync` f16）が同じ launch-only
/// 境界で計測されるため（モジュール冒頭ドキュメンテーションコメント
/// 「計測境界の統一」参照）、`mma_over_wmma_f16` 比は apples-to-apples の
/// 比較になる。
fn measure_mma_f16(gemm: &CudaMmaGemm, size: usize, config: &MeasurementConfig) -> TflopsSample {
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
    tflops_sample(size, &measurement)
}

/// 2 経路の実測値のうち大きい方（＝実際に速い方）を選ぶ。両方存在する
/// 場合は非有限値（NaN・Inf。`gemm.rs` 側の理論上ありえない返り値に対する
/// 防御）を除外したうえで最大値を採り、片方のみ存在する場合はそちらへ
/// フォールバックする。固定優先順位（旧実装）は計測環境によっては遅い
/// 経路を「最良」として候補下限へ反映するロジックバグだったため廃止
/// （PR #349 codex-review 指摘 P1「実測性能を比較せず固定優先順位で
/// 『最良値』を選んでいる」対応）。
///
/// 返り値の第 2 要素はどちらが選ばれたかのラベル（ログ・
/// `docs/perf/cuda-floor-remeasurement.md` への転記時の由来表示用）。
fn best_of(
    label_a: &'static str,
    a: Option<f64>,
    label_b: &'static str,
    b: Option<f64>,
) -> Option<(f64, &'static str)> {
    let a = a.filter(|v| v.is_finite());
    let b = b.filter(|v| v.is_finite());
    match (a, b) {
        (Some(x), Some(y)) => {
            if x >= y {
                Some((x, label_a))
            } else {
                Some((y, label_b))
            }
        }
        (Some(x), None) => Some((x, label_a)),
        (None, Some(y)) => Some((y, label_b)),
        (None, None) => None,
    }
}

/// f32 最良値（tiled と WMMA(TF32) opt のうち実測 TFLOPS が大きい方）を
/// 選ぶ。`gemm_auto.rs::CudaGemmAuto` は現状 f32 を常に tiled へ
/// ディスパッチする決定表を採用しているが（TF32 経路は #62/#186 の
/// 実測・承認まで既定採用を保留）、本バイナリは REQ-8 の「最適化後下限」
/// 候補算出が目的のため、既定ディスパッチとは独立に到達可能な最良経路を
/// 実測比較で直接選ぶ。
fn best_f32(tiled: Option<f64>, wmma_tf32: Option<f64>) -> Option<(f64, &'static str)> {
    best_of("tiled", tiled, "wmma_tf32", wmma_tf32)
}

/// f16 最良値（`wmma_f16` と `mma_f16` のうち実測 TFLOPS が大きい方）を
/// 選ぶ。`best_f32` の f16 側鏡写し。
///
/// 旧実装（本コメント追加前の版）は `measure_wmma_f16` が転送込み・
/// `measure_mma_f16` が launch-only という異なる計測範囲を理由に
/// `mma_f16` を candidate floor から除外していたが、`measure_wmma_f16`
/// を launch-only 化したことで両経路の計測境界が一致した（モジュール
/// 冒頭ドキュメンテーションコメント「計測境界の統一」参照。PR #349
/// codex-review 再指摘 P1 対応）。境界が揃った以上、`best_f32` と対称に
/// 実測比較で最良経路を選ぶ方が「実測性能を比較せず固定で片方を選ぶ」
/// ロジックバグ（`best_of` ドキュメンテーションコメント参照）を再導入
/// しない。
fn f16_candidate_floor_value(wmma: Option<f64>, mma: Option<f64>) -> Option<(f64, &'static str)> {
    best_of("wmma_f16", wmma, "mma_f16", mma)
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
    // 注意: この GPU 名一致判定は WARNING 表示の要否のみに使い、
    // 正式な candidate optimized floor の許可条件にはしない（GPU 名の
    // 部分一致では同一実機比較を保証できないため。PR #349 codex-review
    // 指摘 P1「PyTorch 参照値の扱い」対応。許可条件は
    // `measured_source` / `pytorch_f32_ref` / `pytorch_f16_ref` 参照）。
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

    // 同一実機再計測値の出所（非空文字列）が与えられていれば、判定対象
    // サイズについて env override を候補下限算出の根拠として使いうる
    // （モジュール冒頭ドキュメンテーションコメント「PyTorch 参照値の
    // 再計測」参照）。
    let measured_source = measured_source();
    match &measured_source {
        Some(s) => println!("pytorch reference provenance: measured this run ({s})"),
        None => println!(
            "pytorch reference provenance: PoC-v2-3 fixed values (no \
             CUDA_FLOOR_BENCH_PYTORCH_SOURCE provided; same-hardware re-measurement \
             not confirmed for this run)."
        ),
    }

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
    // `*_same_hardware` は判定対象サイズすべてが同一実機再計測値を根拠と
    // している場合のみ `true` を維持する（1 サイズでも固定値フォール
    // バックが混じれば正式な candidate floor を出さない。PR #349
    // codex-review 指摘 P1 対応）。
    // `*_judged_count` は比率が `Some`（非有限値等での欠測でない）だった
    // 判定対象サイズの個数を数える。`JUDGED_SIZES.len()` に満たない場合は
    // 一部形状が欠測したまま残りの形状だけで正式な candidate floor が
    // 確定してしまうため（PR #349 codex-review 指摘 P1 再指摘対応）、
    // `print_candidate_floor` には全形状が揃った場合のみ「確定可能」として
    // 渡す（`min_ratio_percent` はそのまま、`same_hardware` は
    // `judged_count == JUDGED_SIZES.len()` 条件を合成した値を渡す）。
    let mut min_f32_ratio_percent: Option<f64> = None;
    let mut min_f16_ratio_percent: Option<f64> = None;
    let mut f32_same_hardware = true;
    let mut f16_same_hardware = true;
    let mut f32_judged_count: usize = 0;
    let mut f16_judged_count: usize = 0;

    for size in std::iter::once(REFERENCE_ONLY_SIZE).chain(JUDGED_SIZES) {
        let config = MeasurementConfig::default();

        let tiled = tiled_gemm
            .as_ref()
            .map(|g| measure_tiled_f32(g, size, &config));
        // `measure_wmma_tf32` は `CudaError::WmmaUnavailable`（cc<8.0 や
        // `mma.h` 非搭載環境）を probe した場合 `None` を返す（`and_then`
        // でネストした `Option<Option<f64>>` を平坦化する。上記
        // `measure_wmma_tf32` ドキュメンテーションコメント参照。PR #349
        // Bugbot 指摘 High 対応）。
        let wmma_tf32 = tiled_gemm
            .as_ref()
            .and_then(|g| measure_wmma_tf32(g, size, &config));
        let wmma_f16 = wmma_gemm
            .as_ref()
            .map(|g| measure_wmma_f16(g, size, &config));
        let mma_f16 = mma_gemm.as_ref().map(|g| measure_mma_f16(g, size, &config));

        // 経路選択・候補下限の算出は中央値（`TflopsSample::median`）のみを
        // 根拠とする（従来どおり）。Q1/Q3 は選択には使わず、出力行・
        // ドキュメント転記用の付随情報として別途 `fmt_sample` で出す
        // （PR #349 codex-review 指摘 P1「Q1/Q3 を破棄しており実測記録の
        // 契約を満たせない」対応。選択ロジック自体は変更しない）。
        let f32_best = best_f32(tiled.map(|s| s.median), wmma_tf32.map(|s| s.median));
        // f16 candidate floor は `wmma_f16`・`mma_f16` の実測比較（両者とも
        // launch-only 計測。モジュール冒頭ドキュメンテーションコメント
        // 「計測境界の統一」参照）で最良経路を選ぶ（`f16_candidate_floor_value`
        // ドキュメンテーションコメント参照。PR #349 codex-review 再指摘 P1
        // 対応）。
        let f16_candidate =
            f16_candidate_floor_value(wmma_f16.map(|s| s.median), mma_f16.map(|s| s.median));

        let (pytorch_f32, f32_measured) = pytorch_f32_ref(size, &measured_source);
        let (pytorch_f16, f16_measured) = pytorch_f16_ref(size, &measured_source);

        // 中央値・時間ドメイン Q1/Q3 由来の TFLOPS 値を並記する
        // （`TflopsSample` ドキュメンテーションコメント参照。実測の
        // ばらつき・再現性を出力行・ドキュメント転記の両方で追跡可能に
        // する）。
        let fmt_sample = |v: Option<TflopsSample>| {
            v.map_or("n/a".to_string(), |s| {
                format!(
                    "{:.4}(q1={:.4},q3={:.4})",
                    s.median, s.tflops_from_q1_secs, s.tflops_from_q3_secs
                )
            })
        };
        let f32_ratio_percent = f32_best
            .map(|(v, _)| v / pytorch_f32 * 100.0)
            .filter(|r| r.is_finite());
        let f16_ratio_percent = f16_candidate
            .map(|(v, _)| v / pytorch_f16 * 100.0)
            .filter(|r| r.is_finite());
        let fmt_ratio = |v: Option<f64>| v.map_or("n/a".to_string(), |x| format!("{x:.2}%"));
        let fmt_path = |v: Option<(f64, &'static str)>| v.map_or("n/a", |(_, p)| p);
        // `mma_f16` は `f16_candidate` の算出（`best_of`）にも使われるため
        // （モジュール冒頭ドキュメンテーションコメント「計測境界の統一」
        // 参照）、この比は候補下限とは独立した補足情報として `wmma_f16`
        // との相対値を併記する。両経路とも launch-only 計測のため
        // apples-to-apples の比較である（中央値同士の比較。Q1/Q3 は選択・
        // 比較には使わず付随情報に留める）。
        let mma_over_wmma_f16_percent = match (mma_f16, wmma_f16) {
            (Some(m), Some(w))
                if w.median.is_finite() && w.median > 0.0 && m.median.is_finite() =>
            {
                Some(m.median / w.median * 100.0)
            }
            _ => None,
        };

        println!(
            "size={size} tiled_f32_tflops={} wmma_tf32_tflops={} wmma_f16_tflops={} \
             mma_f16_tflops={} f32_best_path={} f16_candidate_path={} f32_best_over_pytorch={} \
             f16_candidate_over_pytorch={} (pytorch_f32={:.4} pytorch_f16={:.4}, \
             f32_ref_measured={f32_measured} f16_ref_measured={f16_measured}, \
             mma_over_wmma_f16(apples-to-apples, launch-only, median-based)={})",
            fmt_sample(tiled),
            fmt_sample(wmma_tf32),
            fmt_sample(wmma_f16),
            fmt_sample(mma_f16),
            fmt_path(f32_best),
            fmt_path(f16_candidate),
            fmt_ratio(f32_ratio_percent),
            fmt_ratio(f16_ratio_percent),
            pytorch_f32,
            pytorch_f16,
            fmt_ratio(mma_over_wmma_f16_percent),
        );

        if JUDGED_SIZES.contains(&size) {
            if let Some(r) = f32_ratio_percent {
                min_f32_ratio_percent = Some(min_f32_ratio_percent.map_or(r, |m: f64| m.min(r)));
                f32_same_hardware &= f32_measured;
                f32_judged_count += 1;
            }
            if let Some(r) = f16_ratio_percent {
                min_f16_ratio_percent = Some(min_f16_ratio_percent.map_or(r, |m: f64| m.min(r)));
                f16_same_hardware &= f16_measured;
                f16_judged_count += 1;
            }
        }
    }

    println!(
        "---\n\
         judged shapes (REQ-8): M=N=K in {JUDGED_SIZES:?} (size={REFERENCE_ONLY_SIZE} is \
         reference-only, excluded from candidate floor)"
    );
    print_candidate_floor(
        "f32",
        min_f32_ratio_percent,
        f32_same_hardware,
        f32_judged_count,
    );
    print_candidate_floor(
        "f16",
        min_f16_ratio_percent,
        f16_same_hardware,
        f16_judged_count,
    );
    println!(
        "NOTE: candidate floor values are NOT final. Final floor confirmation is a human \
         decision (TASK-8.3 担当: 共同（計測実行は Claude Code、下限値の最終確定は人間）). \
         Transcribe this output into docs/perf/cuda-floor-remeasurement.md and hand off to #158 \
         (TASK-8.3d)."
    );
}

/// 判定対象形状（`JUDGED_SIZES`）すべての比率が揃い（`judged_count ==
/// JUDGED_SIZES.len()`）、かつ同一実機再計測値のみを根拠とする
/// （`same_hardware == true`）場合にのみ、正式な candidate floor を確定値
/// として返す。片方でも欠ける場合は `None`（`n/a` 扱い）とする（REQ-8
/// 「2048・4096 の実測比率の最小値」契約。PR #349 codex-review 指摘 P1
/// 対応。`print_candidate_floor` から呼ばれ、単体でも回帰テストする）。
fn confirmed_candidate_floor(
    min_ratio_percent: Option<f64>,
    same_hardware: bool,
    judged_count: usize,
) -> Option<f64> {
    if same_hardware && judged_count == JUDGED_SIZES.len() {
        min_ratio_percent
    } else {
        None
    }
}

/// 判定対象形状の最小比率から候補下限値を出力する。`confirmed_candidate_floor`
/// が `None` を返す場合（`same_hardware == false`、または `judged_count` が
/// `JUDGED_SIZES.len()` 未満）は、GPU 名が GB10 系であっても正式な
/// candidate floor を出さず `n/a` とし、参考比率のみ表示する（PR #349
/// codex-review 指摘 P1 対応。詳細はモジュール冒頭ドキュメンテーション
/// コメント「PyTorch 参照値の再計測」参照）。
fn print_candidate_floor(
    label: &str,
    min_ratio_percent: Option<f64>,
    same_hardware: bool,
    judged_count: usize,
) {
    let all_judged_sizes_present = judged_count == JUDGED_SIZES.len();
    match confirmed_candidate_floor(min_ratio_percent, same_hardware, judged_count) {
        Some(r) => println!(
            "CUDA {label} candidate optimized floor (rounding rule applied to min ratio \
             {r:.2}%) = {:.0}% (current provisional REQ-8 value: 40%)",
            floor_round(r)
        ),
        // 判定対象サイズが欠測している場合は、実機フォールバック起因の
        // n/a よりも「そもそも全形状を評価し切れていない」欠陥のほうが
        // 根本的なため、`same_hardware` の真偽によらずこの分岐を優先する。
        None if !all_judged_sizes_present => println!(
            "CUDA {label} candidate optimized floor: n/a (one or more judged sizes \
             {JUDGED_SIZES:?} were missing/non-finite in this run; reference-only min ratio over \
             measured judged sizes was {}, but REQ-8 requires the minimum ratio across ALL \
             judged sizes to confirm a candidate floor)",
            min_ratio_percent.map_or("n/a".to_string(), |r| format!("{r:.2}%"))
        ),
        None if !same_hardware => println!(
            "CUDA {label} candidate optimized floor: n/a (same-hardware re-measured PyTorch \
             baseline not confirmed for all judged sizes; reference-only ratio {} uses \
             PoC-v2-3 fixed values, see CUDA_FLOOR_BENCH_PYTORCH_SOURCE in module doc)",
            min_ratio_percent.map_or("n/a".to_string(), |r| format!("{r:.2}%"))
        ),
        None => println!(
            "CUDA {label} candidate optimized floor: n/a (no judged-size measurement available \
             in this environment)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        best_of, confirmed_candidate_floor, f16_candidate_floor_value, floor_round, tflops_sample,
    };
    use bench_harness::Measurement;

    // PR #349 codex-review 指摘 P1「Q1/Q3 を破棄しており実測記録の契約を
    // 満たせない」の回帰確認: `tflops_sample`（4 経路すべての `measure_*` が
    // 経由する唯一の変換経路。重複実装を防ぐための共通化）が
    // `Measurement` の時間ドメイン 3 値を漏れなく TFLOPS ドメインへ変換する
    // こと、かつ `tflops` が時間について単調減少写像であるため
    // `q1_secs < median_secs < q3_secs`（速い方が下位 25%）のとき
    // TFLOPS ドメインでは大小関係が反転すること（`TflopsSample`
    // ドキュメンテーションコメント参照）を確認する。
    #[test]
    fn tflops_sample_converts_all_three_quartiles_and_inverts_time_domain_ordering() {
        let measurement = Measurement {
            median_secs: 2.0,
            q1_secs: 1.0,
            q3_secs: 4.0,
            samples_secs: vec![1.0, 2.0, 4.0],
            warmup: 20,
            iters: 20,
        };
        let sample = tflops_sample(512, &measurement);
        // 時間ドメインでは q1_secs(1.0) < median_secs(2.0) < q3_secs(4.0)。
        // TFLOPS ドメインでは短い時間ほど高 TFLOPS のため大小関係が反転する。
        assert!(sample.tflops_from_q1_secs > sample.median);
        assert!(sample.median > sample.tflops_from_q3_secs);
        // 変換元 secs から独立に算出した期待値とも一致することを確認する
        // （`tflops` を経由した換算そのものが正しいことの担保）。
        let flops = 2.0 * 512.0_f64.powi(3);
        assert_eq!(sample.median, flops / measurement.median_secs / 1e12);
        assert_eq!(
            sample.tflops_from_q1_secs,
            flops / measurement.q1_secs / 1e12
        );
        assert_eq!(
            sample.tflops_from_q3_secs,
            flops / measurement.q3_secs / 1e12
        );
    }

    // PR #349 codex-review 再指摘 P1「PyTorch 参照計測は転送・反復ごとの
    // 確保を含まないのに tiled f32・WMMA(TF32)・WMMA f16 の 3 経路は含めて
    // 計測している」の回帰確認: `measure_wmma_f16`・`measure_mma_f16` が
    // どちらも launch-only 計測に統一された結果（モジュール冒頭
    // ドキュメンテーションコメント「計測境界の統一」参照）、
    // `f16_candidate_floor_value` は `best_f32` と対称に実測比較で最良
    // 経路を選ぶことを確認する（固定で `wmma_f16` のみを使っていた旧実装
    // からの回帰防止）。
    #[test]
    fn f16_candidate_floor_value_picks_larger_of_wmma_and_mma() {
        assert_eq!(
            f16_candidate_floor_value(Some(20.0), Some(10.0)),
            Some((20.0, "wmma_f16"))
        );
        assert_eq!(
            f16_candidate_floor_value(Some(10.0), Some(20.0)),
            Some((20.0, "mma_f16"))
        );
    }

    #[test]
    fn f16_candidate_floor_value_falls_back_to_the_only_available_path() {
        assert_eq!(
            f16_candidate_floor_value(Some(42.0), None),
            Some((42.0, "wmma_f16"))
        );
        assert_eq!(
            f16_candidate_floor_value(None, Some(7.0)),
            Some((7.0, "mma_f16"))
        );
        assert_eq!(f16_candidate_floor_value(None, None), None);
    }

    #[test]
    fn f16_candidate_floor_value_excludes_non_finite() {
        assert_eq!(
            f16_candidate_floor_value(Some(f64::NAN), Some(10.0)),
            Some((10.0, "mma_f16"))
        );
        assert_eq!(
            f16_candidate_floor_value(Some(10.0), Some(f64::INFINITY)),
            Some((10.0, "wmma_f16"))
        );
        assert_eq!(
            f16_candidate_floor_value(Some(f64::NAN), Some(f64::INFINITY)),
            None
        );
    }

    // PR #349 codex-review 再指摘 P1 の回帰確認: 判定対象形状（2048・4096）
    // の一部が欠測（比率が非有限値等で `None`）しても、残りの形状だけから
    // 正式な candidate floor が確定してしまわないこと。`JUDGED_SIZES.len()`
    // は 2 のため、`judged_count == 1`（片方欠測）は `None` を返す必要が
    // ある。
    #[test]
    fn confirmed_candidate_floor_requires_all_judged_sizes_present() {
        // 両サイズとも計測済み・同一実機根拠 → 確定できる。
        assert_eq!(confirmed_candidate_floor(Some(45.0), true, 2), Some(45.0));
        // 1 サイズのみ計測（もう 1 サイズは比率が非有限値等で欠測）
        // → 確定できない（本回帰の主眼）。
        assert_eq!(confirmed_candidate_floor(Some(45.0), true, 1), None);
        // 同一実機根拠が確認できない → 確定できない（既存 P1 対応の維持）。
        assert_eq!(confirmed_candidate_floor(Some(45.0), false, 2), None);
        // どのサイズも計測できていない → 確定できない。
        assert_eq!(confirmed_candidate_floor(None, true, 2), None);
    }

    // PR #349 codex-review 指摘 P1「実測性能を比較せず固定優先順位で
    // 『最良値』を選んでいる」の回帰確認。固定優先（旧実装は常に
    // label_b＝wmma_tf32/mma を優先）ではなく、実測 TFLOPS が大きい方を
    // 選ぶことを両方向で確認する。
    #[test]
    fn best_of_picks_larger_value_either_direction() {
        assert_eq!(best_of("a", Some(20.0), "b", Some(10.0)), Some((20.0, "a")));
        assert_eq!(best_of("a", Some(10.0), "b", Some(20.0)), Some((20.0, "b")));
    }

    #[test]
    fn best_of_falls_back_to_the_only_available_value() {
        assert_eq!(best_of("a", Some(5.0), "b", None), Some((5.0, "a")));
        assert_eq!(best_of("a", None, "b", Some(5.0)), Some((5.0, "b")));
    }

    #[test]
    fn best_of_returns_none_when_both_absent() {
        assert_eq!(best_of("a", None, "b", None), None);
    }

    #[test]
    fn best_of_excludes_non_finite_values() {
        // `gemm.rs` 側の理論上ありえない返り値（NaN/Inf）に対する防御。
        // 非有限値は「未計測」と同様に扱い、有限な側へフォールバックする。
        assert_eq!(
            best_of("a", Some(f64::NAN), "b", Some(10.0)),
            Some((10.0, "b"))
        );
        assert_eq!(
            best_of("a", Some(10.0), "b", Some(f64::INFINITY)),
            Some((10.0, "a"))
        );
        assert_eq!(best_of("a", Some(f64::NAN), "b", Some(f64::INFINITY)), None);
    }

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
