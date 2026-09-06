//! Metal 固定オーバーヘッド（約 5 ms。framework-compare 実測）のフェーズ別
//! 内訳診断バイナリ（イシュー #927。親 #921「Phase 1・診断タスク」・
//! トラッキング #920）。
//!
//! ## 背景・出典
//!
//! PR #915 のフレームワーク横並び実測
//! （`scripts/bench/framework-compare/results/summary.md` 「(a) GEMM」
//! Metal 節）で、fandhe-ai の Metal GEMM は N=256 中央値 5.441 ms・
//! N=512 中央値 5.724 ms とサイズにほぼ非依存の約 5 ms に張り付く
//! （candle は同条件 257.6 µs／519.0 µs）。MLP 推論 24.125 ms・学習
//! 48.845 ms/step（同 summary.md「(b)」「(c)」節）も、GEMM 呼び出し回数に
//! この固定費を乗じた値と桁が整合する。ベンチ対象の MLP
//! （`scripts/bench/framework-compare/bench-fandhe/src/main.rs`
//! `build_model`）は 784→256・256→10 の 2 層構成のため、推論
//! （forward 1 回）は各層 1 回ずつ計 **GEMM 2 回**の呼び出しを含む
//! （`Sequential::forward` → 各 `Var::matmul` 呼び出し）。学習は forward
//! 2 回に加え、`autodiff::grad::matmul_vjp`（`crates/autodiff/src/grad.rs`）
//! が `matmul` 1 回あたり `da`/`db` 2 回の GEMM を計算するため、backward
//! は 2 層 × 2 回 = 4 回。1 step 合計で **GEMM 6 回**（forward 2 +
//! backward 4）が発生する。
//!
//! 計測ハーネス（`scripts/bench/framework-compare/bench-fandhe/src/
//! main.rs`）の GEMM 計測窓は `tape.var`（テンソル生成）の**後**・
//! `matmul` 呼び出しの直前に `Instant::now()` を開始し、ホスト実体化
//! （checksum）で閉じる（同ファイル 78〜83 行目相当）。したがって固定費は
//! `tape_for`（デバイス選択・初期化）のコストではなく、演算メソッド
//! （このバックエンドでは [`fandhe_ai_tensor_core::BackendOps::gemm`]）
//! 呼び出しそのものの内部コストに由来する。
//!
//! `crate::ops::MetalBackendOps::gemm`（`src/ops.rs`）のドキュメンテーション
//! コメントが明記するとおり、`MetalContext`／`MetalGemm` は
//! **各メソッド呼び出し時に都度構築**される（デバイスハンドル常駐化は
//! TASK-1.9b 以降の未着地の最適化）。本 example は、この都度構築が
//! 具体的にどのフェーズ（デバイス取得・キュー生成・MSL 実行時コンパイル・
//! パイプライン構築・転送/実行/同期）に何 ms かかっているかを内訳分解する。
//!
//! ### 構造分析（コード調査で確定済みの仮説フェーズ分解）
//!
//! | フェーズ | 内容 | 出典 |
//! |---------|------|------|
//! | P1 | `MTLCreateSystemDefaultDevice` | `src/context.rs::MetalContext::new` |
//! | P2 | `newCommandQueue` | 同上 |
//! | P3 | `MetalContext::new`（P1+P2 に加え caps／occupancy 照会を含む合算） | `src/context.rs` |
//! | P4 | `newLibraryWithSource_options_error`（MSL 実行時コンパイル。初回/2回目以降で分離集計） | `src/pipeline.rs::compile_gemm_library` |
//! | P5 | `MetalGemm::new(&ctx)`（ライブラリコンパイル＋固定 5 パイプライン構築の合算） | `src/gemm.rs::MetalGemm::new_with_gates` |
//! | P6 | 都度構築 end-to-end（`MetalBackendOps::new()` + `BackendOps::gemm` を毎反復） | `src/ops.rs::MetalBackendOps::gemm` |
//! | P7 | 対照: 資源再利用（`MetalContext`/`MetalGemm` を 1 回構築して `dispatch_auto` を反復） | `src/gemm.rs::MetalGemm::dispatch_auto` |
//!
//! 導出値 `P6 − P7` はプロセス都度構築による固定費の実測値である。`P6`・`P7`
//! はいずれも A/B のアップロード・Metal バッファ確保・カーネル実行・同期・
//! C readback を反復ごとに行う（[`fandhe_ai_backend_metal::gemm::MetalGemm::
//! dispatch_auto`] 内で共通）ため、これらの転送・バッファ確保コストは
//! `P6 − P7` の減算で原則相殺される。`P3 + P5`（デバイス/キュー/caps/
//! occupancy 構築 + ライブラリ/パイプライン構築の合算）との突合で得られる
//! 残差は、転送・バッファ確保ではなく主に (1) tile 構成別特殊化パイプライン
//! がインスタンス単位の遅延キャッシュのため都度構築（P6）では毎回コールドに
//! なる一方、資源再利用（P7）ではキャッシュが温存される差、(2)
//! `BackendOps::gemm`（`src/ops.rs`）固有の処理（形状検証・contiguous 化・
//! `Tensor` 再構築）が P6 側にのみ含まれる差、に帰属させて見積もる。
//!
//! 参考ベースライン（既存実測との突合に使う）: `docs/perf/
//! startup-cost-measurement.md` の Metal 実測（Apple M4 Max）はプロセス
//! 初回（cold）で `device_init_secs` 中央値 約 35 ms・`first_kernel_secs`
//! 中央値 約 42 ms（同 doc「Metal 実測結果」節）。ただし同 doc「コールド／
//! ウォームの検証」節が明記するとおり、macOS はコンパイル済み Metal 関数を
//! **システムレベル**（MTLCompilerService・ユーザーごとキャッシュ）で保持し、
//! そのハーネスの cold/warm フラグはこのシステムキャッシュを一切制御・削除
//! しない。したがって同 doc の「cold」観測点は「本プロセスにとっての 1 回目」
//! を意味するに過ぎず、システムキャッシュが未温であることまでは保証しない。
//! 本 example の `P4 first`（`measure_p4_library_compile` が返す
//! `first_secs`）も同様の限界を持つ: warmup を経ない**プロセス内最初の** MSL
//! コンパイル呼び出しではあるが、直前に他プロセスが同一 `gemm.metal` 相当の
//! シェーダーをコンパイル済みであれば、システムキャッシュがすでに温存されて
//! いる可能性を否定できない。よって `P4 first` を「システムキャッシュ未温な
//! 真のコールド値」と断定せず、**「プロセス内初回（システムキャッシュ状態は
//! 未制御・未記録）」**として扱う。`P4 rest`（`config.iters` 回の 2 回目以降）・
//! P5・P6・P7 はいずれもプロセス内 2 回目以降（warmup 済み）の都度構築費で
//! あり、`P4 first` とは異なる母集団である点に注意する（本 doc「実測結果」節
//! では `P4 first` を上記の限界を明記したうえで記録し、`P4 rest`/P5/P6/P7 は
//! warm 側として突合する）。
//!
//! ## 実行方法
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example fixed_overhead_diagnosis --release -- --size=256,512
//! ```
//!
//! `--size` は既定 `256,512`（framework-compare の実測対象と同一サイズ）。
//! `--iters=<N>`（`N` は 20 以上。`bench_harness::MeasurementConfig::new`
//! が fail-closed で検証する。TASK-8.1 下限）で warmup・計測回数を
//! 両方 `N` へ引き上げられる（未指定時は既定 20/20）。
//!
//! visibility 制約の注意（`gemm_diagnosis.rs` と同じ制約）:
//! `crate::pipeline::compile_options`／`crate::pipeline::make_pipeline` は
//! `pub(crate)` のため example（クレート外）から直接呼べない。P4 の
//! MSL コンパイルオプション（`MTLCompileOptions`）は `pipeline.rs` の
//! `compile_options` と**同一設定**（`MathMode::Safe` +
//! `MathFloatingPointFunctions::Precise`。数値一致契約 REQ-2 の前提）を
//! 本 example 側へ複製する（複製である旨をコード側コメントにも明記）。
//! `crates/backend-metal/src/` の実装変更（可視性変更を含む）は行わない
//! （親 #921「実装変更を伴わない調査・計測・記録タスクのみ」）。

/// 非 macOS 環境向けスタブ。`objc2` 系 FFI は `cfg(target_os = "macos")`
/// 限定のため実測部分はコンパイル対象外（Linux CI の
/// `cargo build --workspace --all-targets`／`cargo clippy --all-targets`
/// をこの example も含めて通すためのプレースホルダ。`gemm_f32_prepared_bench.rs`
/// 等の既存 example と同じスタブ方針）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal fixed_overhead_diagnosis example requires macOS (Apple Silicon). \
         run it on macOS hardware: cargo run -p fandhe-ai-backend-metal --example fixed_overhead_diagnosis --release -- --size=256,512"
    );
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use std::time::Instant;

    use objc2::rc::Retained;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLCompileOptions, MTLCreateSystemDefaultDevice, MTLDevice, MTLMathFloatingPointFunctions,
        MTLMathMode,
    };

    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, Quartiles, median_q1_q3, run as bench_run};
    use fandhe_ai_backend_metal::{MetalBackendOps, MetalContext, MetalGemm};
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    /// 決定的シード（`gemm_bench.rs::SEED`・`gemm_diagnosis.rs::SEED` と同一値。
    /// 既存ベンチと同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// `shaders/gemm.metal` のソース。`crate::pipeline::GEMM_MSL_SRC` は
    /// `pub(crate)` ではなく非公開定数のため、同じソースを直接
    /// `include_str!` する（P4 が対象とするのは `newLibraryWithSource`
    /// 呼び出しそのもののコストであり、ソース内容自体は本番経路と同一で
    /// なければ比較対象として意味を持たない）。
    const GEMM_MSL_SRC: &str = include_str!("../src/shaders/gemm.metal");

    /// `crate::pipeline::compile_options`（`pub(crate)`）と同一設定を
    /// 複製する（モジュール冒頭コメント「visibility 制約の注意」参照）。
    /// `MathMode::Safe` + `MathFloatingPointFunctions::Precise` の明示は
    /// CPU 参照実装・CUDA 側の標準精度指定との数値一致契約（REQ-2）の前提であり、
    /// これを省いた既定値（`Fast`）でコンパイル時間を計測すると本番経路と
    /// 異なるコストを測ってしまう。
    fn compile_options_dup() -> Retained<MTLCompileOptions> {
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        options
    }

    /// `--size` の 1 辺が取りうる上限（要素数）。P6/P7（`measure_p6_rebuild_each_call`・
    /// `measure_p7_reused_dispatch`）は `size * size` 個の `f32` を持つ正方行列
    /// A・B を 2 枚（P7 はさらに C readback も同サイズ）確保するため、
    /// `size * size` 自体の `usize` オーバーフロー・および 1 枚あたりの確保
    /// バイト数の両方を解析時（`parse_sizes_value`）に fail-closed で検証する
    /// （OWASP A03「外部入力の検証」観点。実行時に `checked_mul` が失敗する
    /// 経路〈`elements_for`〉へ到達させない）。上限は 1 枚あたり 1 GiB
    /// （`f32` で `size` 上限 16384 相当）とし、診断用途で現実的に必要な
    /// サイズ（framework-compare 実測対象の 256/512 等）を十分に超える
    /// 一方、`--iters` と組み合わせた誤指定による OOM を防ぐ。
    const MAX_MATRIX_BYTES: usize = 1 << 30;

    /// `size * size` 個の `f32` 要素数を安全に計算する（`checked_mul`）。
    /// `parse_sizes_value` が解析時に `size` の上限（`MAX_MATRIX_BYTES` 相当）を
    /// 検証済みのため、ここでのオーバーフローは到達しない契約だが、
    /// 呼び出し側（P6/P7）が直接 `size * size` を書かない多層防御として
    /// `expect` で失敗を即座に検出する。
    fn elements_for(size: usize) -> usize {
        size.checked_mul(size)
            .expect("size は parse_sizes_value で検証済みのため size*size はオーバーフローしない")
    }

    /// `--size=<N>[,<N>...]` の値部分を解析する（許可リスト方式: カンマ区切りの
    /// 正の整数のみを受理し、それ以外は `Err` で拒否する。OWASP A03
    /// 「外部入力の検証」観点）。各値は `size * size` の整数オーバーフロー・
    /// および `f32` 換算での過大メモリ確保を防ぐため `MAX_MATRIX_BYTES` 上限
    /// まで検証する（P6/P7 が確保する正方行列の 1 枚あたりバイト数がこの上限を
    /// 超える `size` は拒否する）。呼び出し元（[`parse_args`]）が引数列全体を
    /// 一度だけ走査したうえで本関数へ値部分のみを渡す。
    fn parse_sizes_value(v: &str) -> Result<Vec<usize>, String> {
        let mut sizes = Vec::new();
        for part in v.split(',') {
            let n: usize = part
                .trim()
                .parse()
                .map_err(|_| format!("--size の値が不正: '{part}'"))?;
            if n == 0 {
                return Err("--size の各値は正数である必要がある".to_string());
            }
            let elems = n.checked_mul(n).ok_or_else(|| {
                format!("--size の値が大きすぎる（size*size がオーバーフローする）: '{n}'")
            })?;
            let bytes = elems
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| {
                    format!("--size の値が大きすぎる（バイト数計算がオーバーフローする）: '{n}'")
                })?;
            if bytes > MAX_MATRIX_BYTES {
                return Err(format!(
                    "--size の値 '{n}' は上限（1 枚あたり {MAX_MATRIX_BYTES} バイト）を超える"
                ));
            }
            sizes.push(n);
        }
        if sizes.is_empty() {
            return Err("--size に値が 1 つも指定されていない".to_string());
        }
        Ok(sizes)
    }

    /// 解析済みの CLI 引数（[`parse_args`] の戻り値）。
    struct CliArgs {
        sizes: Vec<usize>,
        config: MeasurementConfig,
    }

    /// `std::env::args()` を**一度だけ**走査して `--size=<N>[,<N>...]`・
    /// `--iters=<N>` を解析する（許可リスト方式。OWASP A03「外部入力の検証」
    /// 観点）。従来は `parse_sizes`／`resolve_measurement_config` がそれぞれ
    /// 独立に引数列を走査し、自分の対象外の引数を無条件で無視していたため、
    /// 未知の引数（例: typo `--szie=512`）が黙って既定値扱いになり、
    /// `--size` の重複指定もエラーにならない欠陥があった（イシュー #943
    /// レビュー指摘）。本関数は全引数を一度の走査で分類し、以下を
    /// fail-closed で拒否する:
    /// - `--size=`／`--iters=` のいずれの許可プレフィックスにも一致しない引数
    ///   （typo・未知フラグを含む）
    /// - 同一プレフィックスの重複指定（`--size=256 --size=512` 等）
    ///
    /// 未指定時は `--size` が `[256, 512]`（framework-compare の実測対象
    /// サイズと同一。モジュール冒頭コメント「背景・出典」参照）、`--iters` は
    /// [`MeasurementConfig::default`]（20/20）を使う。
    fn parse_args() -> Result<CliArgs, String> {
        let mut size_raw: Option<String> = None;
        let mut iters_raw: Option<String> = None;
        for arg in std::env::args().skip(1) {
            if let Some(v) = arg.strip_prefix("--size=") {
                if size_raw.is_some() {
                    return Err(format!("--size は複数回指定できない（重複指定）: '{arg}'"));
                }
                size_raw = Some(v.to_string());
            } else if let Some(v) = arg.strip_prefix("--iters=") {
                if iters_raw.is_some() {
                    return Err(format!("--iters は複数回指定できない（重複指定）: '{arg}'"));
                }
                iters_raw = Some(v.to_string());
            } else {
                return Err(format!(
                    "未知の引数: '{arg}'（許可される引数は --size=<N>[,<N>...] と --iters=<N> のみ）"
                ));
            }
        }

        let sizes = match size_raw {
            Some(v) => parse_sizes_value(&v)?,
            None => vec![256, 512],
        };
        let config = match iters_raw {
            Some(v) => {
                let n: usize = v
                    .trim()
                    .parse()
                    .map_err(|_| format!("--iters の値が不正: '{v}'"))?;
                MeasurementConfig::new(n, n).map_err(|e| e.to_string())?
            }
            None => MeasurementConfig::default(),
        };
        Ok(CliArgs { sizes, config })
    }

    /// ミリ秒単位の [`Quartiles`] を整形する共通ヘルパ（`Measurement`
    /// （秒単位）・独自計測双方の出力書式を揃える）。
    fn fmt_ms(label: &str, q: Quartiles) -> String {
        format!(
            "{label}: median={:.4}ms q1={:.4}ms q3={:.4}ms",
            q.median * 1e3,
            q.q1 * 1e3,
            q.q3 * 1e3
        )
    }

    /// P1: `MTLCreateSystemDefaultDevice` 単体の呼び出しコスト。
    fn measure_p1_device(config: &MeasurementConfig) -> Quartiles {
        let measurement = bench_run(config, || {
            let _device = MTLCreateSystemDefaultDevice();
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");
        Quartiles {
            median: measurement.median_secs,
            q1: measurement.q1_secs,
            q3: measurement.q3_secs,
        }
    }

    /// P2: `newCommandQueue` 単体の呼び出しコスト（P1 で 1 回だけ取得した
    /// デバイスを使い回し、キュー生成のみを反復計測する）。
    fn measure_p2_queue(
        config: &MeasurementConfig,
        device: &objc2::runtime::ProtocolObject<dyn MTLDevice>,
    ) -> Quartiles {
        let measurement = bench_run(config, || {
            let _queue = device.newCommandQueue();
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");
        Quartiles {
            median: measurement.median_secs,
            q1: measurement.q1_secs,
            q3: measurement.q3_secs,
        }
    }

    /// P3: `MetalContext::new()`（デバイス取得＋キュー生成＋caps／occupancy
    /// 照会の合算。`src/context.rs` 参照）の呼び出しコスト。
    fn measure_p3_context(config: &MeasurementConfig) -> Quartiles {
        let measurement = bench_run(config, || {
            let _ctx = MetalContext::new().expect("MetalContext::new に失敗した（実機前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");
        Quartiles {
            median: measurement.median_secs,
            q1: measurement.q1_secs,
            q3: measurement.q3_secs,
        }
    }

    /// P4: `newLibraryWithSource_options_error`（MSL 実行時コンパイル）の
    /// 呼び出しコスト。初回（プロセス内 1 回目。システム Metal コンパイラ
    /// キャッシュの温存状態はハーネス側から制御・記録していないため
    /// 「未温」と断定しない — モジュール冒頭コメント「参考ベースライン」節
    /// 参照）と 2 回目以降（`config.iters` 回。下限 20 回は
    /// `MeasurementConfig::new` 経由で検証済み）を分離集計する（モジュール
    /// 冒頭コメント「構造分析」表 P4 参照。warmup は行わない — warmup 自体が
    /// 「2 回目以降」の一部になってしまい初回と分離する目的に反するため）。
    fn measure_p4_library_compile(
        config: &MeasurementConfig,
        device: &objc2::runtime::ProtocolObject<dyn MTLDevice>,
    ) -> (f64, Quartiles) {
        let src = NSString::from_str(GEMM_MSL_SRC);

        let first_start = Instant::now();
        let _library = device
            .newLibraryWithSource_options_error(&src, Some(&compile_options_dup()))
            .expect("MSL ライブラリの初回コンパイルに失敗した（実機前提）");
        let first_secs = first_start.elapsed().as_secs_f64();

        let mut rest_secs = Vec::with_capacity(config.iters);
        for _ in 0..config.iters {
            let start = Instant::now();
            let _library = device
                .newLibraryWithSource_options_error(&src, Some(&compile_options_dup()))
                .expect("MSL ライブラリの再コンパイルに失敗した（実機前提）");
            rest_secs.push(start.elapsed().as_secs_f64());
        }
        let rest_quartiles =
            median_q1_q3(&rest_secs).expect("rest_secs は config.iters（>=20）個の非空サンプル");

        (first_secs, rest_quartiles)
    }

    /// P5: `MetalGemm::new(&ctx)`（ライブラリコンパイル＋固定 5 パイプライン
    /// 構築の合算。`src/gemm.rs` 参照）の呼び出しコスト。`ctx` は 1 回だけ
    /// 構築して使い回す（P5 が対象とするのは `MetalGemm::new` 自体のコスト
    /// であり `MetalContext::new` の再計測ではないため。P3 とは独立に計測
    /// する）。
    fn measure_p5_gemm_new(config: &MeasurementConfig, ctx: &MetalContext) -> Quartiles {
        let measurement = bench_run(config, || {
            let _gemm = MetalGemm::new(ctx).expect("MetalGemm::new に失敗した（実機前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");
        Quartiles {
            median: measurement.median_secs,
            q1: measurement.q1_secs,
            q3: measurement.q3_secs,
        }
    }

    /// P6: 都度構築 end-to-end。`MetalBackendOps::new()` + `BackendOps::gemm`
    /// を毎反復呼び出す（`src/ops.rs::MetalBackendOps::gemm` が
    /// `MetalContext::new` + `MetalGemm::new` を毎回構築する経路をそのまま
    /// 経由する。framework-compare の 1 反復と同等条件の再現。モジュール
    /// 冒頭コメント「背景・出典」参照）。
    fn measure_p6_rebuild_each_call(config: &MeasurementConfig, size: usize) -> Quartiles {
        let mut rng = Xorshift64Star::new(SEED);
        let elems = elements_for(size);
        let a_data = rng.fill_vec(elems);
        let b_data = rng.fill_vec(elems);
        let a = Tensor::new(a_data, &[size, size]).expect("Tensor::new(a) に失敗した");
        let b = Tensor::new(b_data, &[size, size]).expect("Tensor::new(b) に失敗した");

        let measurement = bench_run(config, || {
            let ops = MetalBackendOps::new();
            let _c = ops
                .gemm(&a, &b)
                .expect("MetalBackendOps::gemm に失敗した（実機前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");
        Quartiles {
            median: measurement.median_secs,
            q1: measurement.q1_secs,
            q3: measurement.q3_secs,
        }
    }

    /// P7: 対照実験（資源再利用）。`MetalContext`/`MetalGemm` を 1 回だけ
    /// 構築し `dispatch_auto` を反復する（`gemm_diagnosis.rs::
    /// wall_measurement`・`gemm_bench.rs::measure_auto` と同型。転送
    /// （A・B アップロード＋C readback）＋カーネル実行＋同期のみを含み、
    /// 都度構築コスト（P1〜P5）を含まない）。
    fn measure_p7_reused_dispatch(config: &MeasurementConfig, size: usize) -> Quartiles {
        let ctx = MetalContext::new().expect("MetalContext::new に失敗した（実機前提）");
        let gemm = MetalGemm::new(&ctx).expect("MetalGemm::new に失敗した（実機前提）");
        let mut rng = Xorshift64Star::new(SEED);
        let elems = elements_for(size);
        let a = rng.fill_vec(elems);
        let b = rng.fill_vec(elems);

        let measurement = bench_run(config, || {
            gemm.dispatch_auto(&ctx, &a, &b, size, size, size)
                .expect("MetalGemm::dispatch_auto に失敗した（実機前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");
        Quartiles {
            median: measurement.median_secs,
            q1: measurement.q1_secs,
            q3: measurement.q3_secs,
        }
    }

    pub fn main() {
        let CliArgs { sizes, config } = match parse_args() {
            Ok(args) => args,
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        };

        // P1・P2・P4 はサイズ非依存（GEMM 形状に依存しないフェーズ）のため
        // size ループの外で 1 度だけ計測する。P3・P5・P6・P7 は
        // `MetalBackendOps::gemm`／`dispatch_auto` の呼び出し自体を含むため
        // size ごとに計測する（P3・P5 は本来サイズに依存しないが、
        // ループ内で計測し直しても値は安定するはずという前提を突合検算
        // できるようにするため、あえて size ごとに再計測して比較対象の
        // 表を揃える）。
        println!("=== サイズ非依存フェーズ ===");
        let device = MTLCreateSystemDefaultDevice().expect("Metal デバイスが見つからない");
        let p1 = measure_p1_device(&config);
        println!("{}", fmt_ms("P1 device_create", p1));
        let p2 = measure_p2_queue(&config, &device);
        println!("{}", fmt_ms("P2 queue_create", p2));
        let (p4_first_secs, p4_rest) = measure_p4_library_compile(&config, &device);
        println!("P4 library_compile_first: {:.4}ms", p4_first_secs * 1e3);
        println!("{}", fmt_ms("P4 library_compile_rest", p4_rest));

        for size in sizes {
            println!("=== size={size} ===");
            let p3 = measure_p3_context(&config);
            println!("{}", fmt_ms("P3 context_new", p3));

            let ctx_for_p5 =
                MetalContext::new().expect("MetalContext::new に失敗した（P5 用。実機前提）");
            let p5 = measure_p5_gemm_new(&config, &ctx_for_p5);
            println!("{}", fmt_ms("P5 gemm_new", p5));

            let p6 = measure_p6_rebuild_each_call(&config, size);
            println!("{}", fmt_ms("P6 rebuild_each_call(end_to_end)", p6));

            let p7 = measure_p7_reused_dispatch(&config, size);
            println!("{}", fmt_ms("P7 reused_dispatch(control)", p7));

            let fixed_cost_ms = (p6.median - p7.median) * 1e3;
            let p3_plus_p5_ms = (p3.median + p5.median) * 1e3;
            println!(
                "derived: P6-P7(fixed_cost)={fixed_cost_ms:.4}ms P3+P5={p3_plus_p5_ms:.4}ms \
                 residual(fixed_cost-(P3+P5))={:.4}ms",
                fixed_cost_ms - p3_plus_p5_ms
            );
        }
    }
}
