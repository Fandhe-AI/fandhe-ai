//! 転置タイル variant 自動ルーティング候補（`MetalGemm::
//! dispatch_strided_tiled_prepared`）の性能 A/B 計測 example（イシュー
//! #1186）。
//!
//! #1138（PR #1167）で `gemm_simdgroup_tiled` に転置ロード（`TRANS_A`/
//! `TRANS_B`）を追加し、明示入口 `dispatch_strided_tiled_prepared` の
//! 正確性（NN 非後退ビット同一・NT/TN/TT parity）は実機で確認済みだが、
//! bias/act なし・適格な入力を classic strided 経路
//! （`dispatch_strided_bias_act_prepared`。`gemm_tiled_bias_act`）から
//! この新経路へ自動委譲する結線は性能 A/B 未計測のため見送られていた
//! （`docs/perf/metal-gemm-transpose-tiled.md` §5・§6）。本 example は
//! その A/B を埋め、結線可否（別イシュー #1187）の判断材料を作る。
//!
//! - **A（base）**: `MetalGemm::dispatch_strided_bias_act_prepared`
//!   （現状の本番経路。bias=None・act=false）
//! - **B（head）**: `MetalGemm::dispatch_strided_tiled_prepared`
//!   （`tile::select_for_device` が選ぶ構成を明示指定。`dispatch_auto` の
//!   本番既定経路と同じ構成選択ロジックを使う——#1187 の結線が渡す構成
//!   そのものを計測する）
//!
//! 対象は `gemm_transpose_tile_sweep.rs::shapes()` と同一の 10 形状 ×
//! NT/TN/TT（3 パターン）の計 30 セル。NN は本 Issue のスコープ外
//! （`dispatch_strided_tiled_prepared` の NN 経路は既に
//! `dispatch_tiled_prepared` とビット同一が確認済みで、NN 向け自動
//! ルーティング判断は本 Issue の対象ではない）。
//!
//! 計測境界は prepared（アップロード済みバッファ・A/B で共有。転送
//! 非計測）。A・B とも 1 セルにつき同一の物理バッファ（`transpose_dense`
//! で構築した転置済みレイアウト）を使い回す
//! （`gemm_transpose_tile_sweep.rs::measure_transposed` と同じ計測範囲の
//! 判断）。
//!
//! 計測プロトコルは `gemm_swizzle_ab_bench.rs` と同一
//! （`bench_harness::ab`。フェーズ 1: 安定性セルフチェック→フェーズ 2:
//! A/B。ROUNDS/COOLDOWN/MIN_WARMUP は実機の負荷状況に応じて増やす方向のみ
//! 調整しうる（既定値はコード側の定数を正とする。値を本コメントへ複製
//! しない——`docs/perf/metal-bench-noise-protocol.md` と独立に乖離するのを
//! 防ぐため）。interleaved・
//! `docs/perf/metal-bench-noise-protocol.md` 準拠）。
//!
//! 判断基準（Issue #1186 本文）: 「全形状 × NT/TN/TT で B/A（TFLOPS 比）
//! が 1.0 以上」を満たせば `verdict=route_ok`（結線可）、1 セルでも
//! 下回れば `verdict=route_ng`（結線不可）、安定性ゲート超過セルが
//! 残れば `verdict=undetermined`（判定不可）とし、閾値そのものは
//! コード側で緩めない（fail-closed）。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_bench.rs`
//! ドキュメンテーションコメント（同ディレクトリ）と同一。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release --features internal-diagnostics
//! ```
//!
//! 実行前後に `pmset -g therm` でサーマル状態を記録すること
//! （`docs/perf/metal-gemm-transpose-tiled.md` §5 実行手順参照）。
//! フェーズ 1 でいずれかのサイズが安定性ゲート
//! （`bench_harness::ab::STABILITY_SPREAD_GATE`）を超過した場合、
//! フェーズ 2（A/B 判定）には進まない（「判定不可」を出力して終了する。
//! 安全側判断: 判定を無効化して中断する方向のみ許す）。
//!
//! ## phase 1 のみ実行モード（イシュー #1249/#1251）
//!
//! フェーズ 1（安定性セルフチェック）は #1186/#1187 の計 8 試行で一度も
//! 安定性ゲートを満たせず、ログ上は 10 ラウンド中 1 ラウンドだけ落ち込む
//! 単発スパイク型であることが分かっている（`docs/perf/
//! metal-gemm-transpose-tiled.md` §5.2・§5.4）。#1249 はこの再現条件を
//! 排他環境／負荷環境で phase 1 のみを複数回実行して切り分けるため、
//! フェーズ 2（A/B 判定）へ進まず終了する `--phase1-only` モードを設ける。
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release --features internal-diagnostics -- --phase1-only
//! ```
//!
//! `--phase1-only` を指定すると、既定の phase 1 → phase 2 の流れは実行せず
//! 冒頭に `mode=phase1_only` を出力したうえでフェーズ 1 のみを実行し、
//! `verdict=not_evaluated (...)` を出力して終了する（`undetermined` とは
//! 区別する。フェーズ 1 が安定性ゲートを満たしたか否かに関わらず、常に
//! A/B 判定を実行しなかったことを示すため）。既定動作（引数なし）は不変。
//!
//! フェーズ 1 の各サイズについて、既存の
//! `size=… spread=… (…) round_tflops=…` 行（バイト単位で不変）の直後に、
//! 分布集計を機械的に grep できる 1 行 `phase1_round_stats` を追加出力する
//! （キー: `size`・`rounds`・`spread`・`gate`・`within_gate`・
//! `median_secs`・`min_secs`／`min_round_idx`・`max_secs`／`max_round_idx`・
//! `round_medians_secs`〈カンマ区切り〉・`trimmed_spread_k1`／
//! `iqr_spread`／`mad_spread`〈イシュー #1484。`bench_harness::ab::
//! StabilityResult::aux` のトリム済みレンジ・IQR・MAD ベースの補助
//! spread 統計量。**判定には使わない**レポート専用列。`trimmed` が
//! `None`〈ラウンド数がトリム後 2 要素未満〉のときは `NA`〉）。`min`／
//! `max` は**秒基準**・**0 始まり** index である点に注意: TFLOPS 換算では
//! 大小関係が逆転する（レイテンシ比と TFLOPS 比の取り違えは #540/#746 で
//! 一度発生した既知の落とし穴。本モジュール内 `b_over_a_tflops` の doc
//! comment 参照）ため、「落ち込んだラウンド」は常に `max_secs` 側で読む
//! こと。フェーズ 1 末尾には総括 1 行 `phase1_summary`（ゲート超過サイズ
//! の一覧）を既定モード・`--phase1-only` の両方で出力する。
//!
//! `--phase1-only` はプロセス内リピートに対応しない（`--repeat=N` 等は
//! 非対応）。#1253/#1255 の「複数回実行」は 1 回ごとに別プロセスで起動し、
//! run ごとに env_info・uptime を独立に取る運用を想定するため。
//!
//! ## GPU タイムスタンプ分離計測モード（イシュー #1259）
//!
//! フェーズ 1 の単発スパイクが **GPU 実行時間（純カーネル時間）** に
//! 乗るのか **host 側時間**（upload・alloc・encode・commit_wait・
//! readback）に乗るのかを切り分けるため、`--gpu-timestamps`（opt-in・
//! `--phase1-only` と併用可）を指定すると、フェーズ 1 の対照ワークロード
//! を `dispatch_auto` から計装版へ**置換**する（追加パスを走らせて
//! ROUNDS を倍増させない）。計装版は `dispatch_auto`
//! （`GemmVariant::SimdgroupTiled` 分岐）と同一組成
//! （upload → alloc → encode → commit_wait → readback）を公開 API
//! （[`fandhe_ai_backend_metal::MetalGemm::encode_tiled_prepared`]・
//! [`fandhe_ai_backend_metal::MetalContext::synchronize_with_gpu_
//! timestamps`]）で再現しつつ、`Instant` によるホスト側フェーズ内訳と
//! `MTLCommandBuffer::GPUStartTime`/`GPUEndTime`（`kernel_gpu`）を
//! 呼び出しごとに記録する。対象サイズ（256〜4096）は全て 8 の倍数の
//! ため `pad_matrix`/`unpad_matrix` は no-op・`c_buf` は `dispatch_auto`
//! と同じ専有確保（本 example の `MetalContext` はプロセスワイド
//! singleton ではないため `alloc_uninit_pooled` は元々 `new_zeroed`へ
//! フォールバックする）で、既定組成との実質差は (i) バッチラベル、
//! (ii) `encode` の resources 3 本 retain、(iii) タイムスタンプ取得 2 回、
//! (iv) 入力検証経路（`validate_dims` on slice →
//! `validate_prepared_inputs_f32`）のみ（いずれも無視できる固定費）。
//!
//! 既定（引数なし・`--gpu-timestamps` を指定しない）の壁時計判定・出力
//! （`phase1_round_stats`／`phase1_summary`／`verdict=` 等）は本モードの
//! 影響を一切受けない（バイト単位で不変）。フェーズ 2（A/B 判定）は
//! 非計装のまま（本イシューのスコープ外）。
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release --features internal-diagnostics -- --phase1-only --gpu-timestamps
//! ```
//!
//! opt-in 時は冒頭に `phase1_workload=gpu_timestamps` を出力し、各サイズ
//! について機械可読な `phase1_gpu_host_round`（ラウンド別）・
//! `phase1_gpu_host_stats`（サイズ別総括）の 2 行を追加出力する
//! （キーは各出力関数〈`format_gpu_host_round_line`／
//! `format_gpu_host_size_line`〉のドキュメンテーションコメント参照）。
//! `wall_minus_gpu`／`commit_wait_minus_gpu` はサンプルごとに差を取って
//! から中央値を計算する（`median(a) − median(b)` ではない）。
//! `MTLCommandBuffer` の不変条件（`batches.len()==1`・タイムスタンプ
//! `Some`・`0 ≤ kernel_gpu ≤ commit_wait ≤ wall`）違反は fail-closed で
//! 当該ラウンドを `valid=false`・関連する差分／spread を `NA` として
//! 報告し、既定の壁時計判定出力を失わせない。
//!
//! ## 実行前環境ガード（イシュー #1265）
//!
//! フェーズ 1（`MetalContext::new()` より前）・フェーズ 2 の各開始前に、
//! `bench_harness::env_guard`（イシュー #1264）のバックオフ再試行
//! （`RetryConfig`・`run_guard_with_retry`）を通した実行前チェックを行う。
//! `--max-load-avg=<f64>` 未指定時は **record_only**（判定なし・記録のみ）、
//! 指定時は **gated**（`EnvGuardConfig` による判定あり。不成立
//! 〈`Fail`〉ならバックオフ再試行し、上限到達で `verdict=undetermined` を
//! 出力して非ゼロ終了する）で動作する。判定結果は `env_info.txt` 準拠の
//! テキストブロックとして stdout へ常時出力し、`--env-info-out=<path>` を
//! 指定すると同じテキストを追記する（opt-in）。
//!
//! ```sh
//! # 記録のみ（判定なし）で GPU を初期化せず終了する短時間動作確認
//! cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release --features internal-diagnostics -- --guard-only
//!
//! # 閾値を明示して gated モードで実行（不成立時は自動でバックオフ再試行）
//! cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release --features internal-diagnostics -- --max-load-avg=8.0 --gpu-watch=Chrome
//! ```
//!
//! 閾値（`--max-load-avg`）に既定値はない（ユーザー承認事項。
//! `docs/perf/metal-bench-noise-protocol.md`「#1265 向けの提案閾値」参照）。
//! バックオフ再試行の待機定数（`macos_impl::GUARD_INITIAL_WAIT` 等）は
//! `--guard-max-attempts`／`--guard-wait-secs` で上書きできる。詳細な出力
//! フォーマット・再試行規定は同ドキュメント「7. ガード不成立時の再試行
//! 規定」を参照。ガード・待機は計測区間（`run_stability`／`run_ab`）の
//! 外側で完結し、計測プロトコル定数（`ROUNDS`・`STABILITY_SPREAD_GATE`
//! 等）には一切影響しない。
//!
//! ## MIN_WARMUP 増加試行モード（イシュー #1261）
//!
//! フェーズ 1 の単発スパイクが原因候補 (b)（ウォームアップ不足。GPU
//! クロック〈DVFS〉が計測開始直後にまだ定常状態へ達していない可能性）
//! に起因するかを切り分けるため、`--min-warmup-secs=<N>`（`--phase1-only`／
//! `--gpu-timestamps` と順序不問で併用可）を指定すると、既定
//! `MIN_WARMUP`（3 秒）の代わりに `N` 秒をフェーズ 1・フェーズ 2 双方の
//! `AbConfig` へ渡す。`N` は **3〜600 の整数秒**（既定値を下回る「減らす
//! 方向」の指定は拒否——原因候補 (b) の切り分けは増やす方向の試行に
//! 限定する `docs/perf/metal-gemm-transpose-tiled.md` §5.6 の計画上の
//! 制約）。片方のフェーズだけへ適用すると A/B 側と対照側で warmup が
//! 異なる非対称が生じるため、両方へ一貫適用する。
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release --features internal-diagnostics -- --phase1-only --gpu-timestamps --min-warmup-secs=9
//! ```
//!
//! 指定時のみ冒頭に `phase1_min_warmup_override_secs=N` を出力する（実効値
//! をログ自身に残すことで、e.g. `docs/perf/logs/
//! metal-gemm-transpose-route-ab-1242/1261-aggregate.py` がどの run が
//! どの MIN_WARMUP で計測されたかを再現できるようにする）。未指定時の
//! 出力（既定 `--phase1-only`／`--gpu-timestamps` の出力を含む）はバイト
//! 単位で不変。

/// `parse_args_from` の解析結果（イシュー #1251）。
///
/// macOS 実機の `macos_impl::main` と Linux CI の `#[cfg(test)]` ユニット
/// テストの双方から使われるため、`gemm_swizzle_ab_bench.rs::
/// SingleRunVerdict` と同型の cfg 分岐（非 macOS・非テストの `example`
/// ターゲット単体でのみ未使用になる誤検知）で `dead_code` を抑止する。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone, PartialEq, Default)]
struct CliArgs {
    /// `true` なら phase 1（安定性セルフチェック）のみ実行してフェーズ 2
    /// （A/B 判定）へ進まない。
    phase1_only: bool,
    /// `true` ならフェーズ 1 の対照ワークロードを `dispatch_auto` から
    /// GPU タイムスタンプ計装版へ置換する（イシュー #1259。`--phase1-only`
    /// と順序不問で併用可）。
    gpu_timestamps: bool,
    /// `--max-load-avg=<f64>`（イシュー #1265）。指定時のみ実行前
    /// 環境ガードを **gated**（`bench_harness::env_guard::EnvGuardConfig`
    /// による判定あり）にする。未指定なら **record_only**（記録のみ・
    /// 判定なし）。閾値の既定値はユーザー承認事項のためコードへ埋め込まない
    /// （`docs/perf/metal-bench-noise-protocol.md`「#1265 向けの提案閾値
    /// （未承認・記録のみ）」参照）。
    max_load_avg: Option<f64>,
    /// `--gpu-watch=<name>`（複数指定可。イシュー #1265）。
    /// `EnvGuardConfig::with_gpu_process_watchlist` へ渡す部分一致名。
    /// `max_load_avg` 指定時のみ有効（単独指定はエラー）。
    gpu_watch: Vec<String>,
    /// `--guard-max-attempts=<usize>`（イシュー #1265）。未指定時は
    /// `macos_impl::GUARD_MAX_ATTEMPTS`。
    guard_max_attempts: Option<usize>,
    /// `--guard-wait-secs=<f64>`（イシュー #1265）。未指定時は
    /// `macos_impl::GUARD_INITIAL_WAIT`。
    guard_wait_secs: Option<f64>,
    /// `--guard-only`（イシュー #1265）。ガード＋env_info 出力のみ行い
    /// GPU を初期化せず終了する（短時間動作確認・実行前チェック用）。
    guard_only: bool,
    /// `--env-info-out=<path>`（イシュー #1265）。env_info ブロックを
    /// 指定ファイルへ追記する opt-in（stdout 出力は常に行う）。
    env_info_out: Option<String>,
    /// `Some(n)` なら既定のウォームアップ下限（[`DEFAULT_MIN_WARMUP_SECS`]。
    /// 3 秒）の代わりに `n` 秒をフェーズ 1・フェーズ 2 双方の `AbConfig`
    /// へ渡す（イシュー #1261。ウォームアップ不足〈原因候補 (b)〉の
    /// 切り分け用。既定値を「減らす方向」の指定は受け付けない——
    /// [`parse_min_warmup_secs`] 参照）。
    min_warmup_secs: Option<u64>,
}

/// フェーズ 1・フェーズ 2 共通のウォームアップ下限（`macos_impl::
/// phase1_stability_selfcheck`／`phase2_route_ab` が `--min-warmup-secs`
/// 未指定時に使う値）の既定値（秒）の単一真実源（イシュー #1261）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
const DEFAULT_MIN_WARMUP_SECS: u64 = 3;

/// `--min-warmup-secs=<N>` 解析結果（[`CliArgs::min_warmup_secs`]）から
/// 実効 `Duration` を求める純関数（イシュー #1261）。`None`（未指定）なら
/// [`DEFAULT_MIN_WARMUP_SECS`] を返す。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn min_warmup_override_duration(min_warmup_secs: Option<u64>) -> std::time::Duration {
    std::time::Duration::from_secs(min_warmup_secs.unwrap_or(DEFAULT_MIN_WARMUP_SECS))
}

/// `std::env::args()` を**一度だけ**走査して CLI 引数を解析する
/// （`gemm_counter_workload.rs::parse_args`・
/// `fixed_overhead_diagnosis.rs::parse_args` と同型の一括走査＋未知引数・
/// 重複指定の fail-closed 拒否。OWASP A03 観点）。
///
/// `std::env::args` を直接読まず引数列 `I` を受け取る純関数にしているのは、
/// `std::env::args()` を関数内で直接読む実装は単体テストから差し替えられ
/// ない（`gemm_profile_target.rs` が指摘する同種の問題）ため。呼び出し元
/// （`macos_impl::main`）が `std::env::args().skip(1)` を渡す薄い呼び出しへ
/// 分離することで、Linux CI の `#[cfg(test)]` から引数列を注入して検証
/// できる。
///
/// 許可する引数（順序不問で併用可）:
/// - `--phase1-only`／`--gpu-timestamps`（値なしフラグ）
/// - `--max-load-avg=<f64>`（有限・正）・`--guard-max-attempts=<usize>`
///   （1 以上）・`--guard-wait-secs=<f64>`（有限・正）・`--gpu-watch=<name>`
///   （非空文字列。複数指定可）・`--guard-only`（値なしフラグ）・
///   `--env-info-out=<path>`（非空文字列）（イシュー #1265。
///   `RetryConfig`／`EnvGuardConfig` への実結線は `macos_impl::run_env_guard`）
/// - `--min-warmup-secs=<N>`（単一トークン。`=` 必須・値は 3〜600 の整数秒。
///   イシュー #1261。既定のウォームアップ下限 [`DEFAULT_MIN_WARMUP_SECS`] の
///   代わりに使う）
///
/// それ以外の引数・重複指定（`--gpu-watch` を除く）は `Err` で fail-closed
/// に拒否する（呼び出し元は `MetalContext::new` に到達する前にこの結果を
/// 検査し、不正引数なら GPU を触らずに終了する）。`--gpu-watch` は
/// `max_load_avg`（`--max-load-avg`）が指定されていない状態での単独指定を
/// エラーとする（gated モードでのみ意味を持つため）。数値引数は非数値・
/// 非有限・非正（`guard_max_attempts` は 0）を fail-closed に拒否する。
/// `--min-warmup-secs 9`（値をスペース区切りの別トークンで渡す形）は
/// 本関数が認識する形式ではないため未知の引数として拒否される（`=` 必須の
/// 設計）。プロセス内リピート（`--repeat=N`）・`--help` 等は意図的に非対応
/// （本 example ヘッダ doc comment 参照）。環境変数は使わない
/// （#1454 の引数方式に統一）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn parse_args_from<I: IntoIterator<Item = String>>(args: I) -> Result<CliArgs, String> {
    let mut out = CliArgs::default();
    let mut max_load_avg_seen = false;
    let mut guard_max_attempts_seen = false;
    let mut guard_wait_secs_seen = false;
    let mut env_info_out_seen = false;
    let mut min_warmup_secs: Option<u64> = None;
    for arg in args {
        if let Some(rest) = arg.strip_prefix("--max-load-avg=") {
            if max_load_avg_seen {
                return Err(format!(
                    "--max-load-avg は複数回指定できない（重複指定）: '{arg}'"
                ));
            }
            let value: f64 = rest
                .parse()
                .map_err(|_| format!("--max-load-avg の値が数値として解釈できない: '{arg}'"))?;
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "--max-load-avg は有限かつ正である必要がある: '{arg}'"
                ));
            }
            out.max_load_avg = Some(value);
            max_load_avg_seen = true;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--gpu-watch=") {
            if rest.is_empty() {
                return Err(format!("--gpu-watch の値が空文字列: '{arg}'"));
            }
            out.gpu_watch.push(rest.to_string());
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--guard-max-attempts=") {
            if guard_max_attempts_seen {
                return Err(format!(
                    "--guard-max-attempts は複数回指定できない（重複指定）: '{arg}'"
                ));
            }
            let value: usize = rest.parse().map_err(|_| {
                format!("--guard-max-attempts の値が非負整数として解釈できない: '{arg}'")
            })?;
            if value == 0 {
                return Err(format!(
                    "--guard-max-attempts は 1 以上である必要がある: '{arg}'"
                ));
            }
            out.guard_max_attempts = Some(value);
            guard_max_attempts_seen = true;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--guard-wait-secs=") {
            if guard_wait_secs_seen {
                return Err(format!(
                    "--guard-wait-secs は複数回指定できない（重複指定）: '{arg}'"
                ));
            }
            let value: f64 = rest
                .parse()
                .map_err(|_| format!("--guard-wait-secs の値が数値として解釈できない: '{arg}'"))?;
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "--guard-wait-secs は有限かつ正である必要がある: '{arg}'"
                ));
            }
            // `is_finite() && > 0.0` だけでは `Duration` の表現範囲
            // （最大 `Duration::MAX` ≒ 1.8e19 秒）を超える値（例:
            // `--guard-wait-secs=1e100`）を通してしまい、後段の
            // `run_env_guard` 内 `Duration::from_secs_f64` が panic する
            // 経路になっていた（Review 指摘・AGENTS.md「本番経路の panic
            // 禁止」。#1265）。ここで `Duration::try_from_secs_f64` により
            // 変換可能性そのものを検証し、範囲外なら CLI 引数エラーとして
            // undetermined ではなく明示的に拒否する。
            if std::time::Duration::try_from_secs_f64(value).is_err() {
                return Err(format!(
                    "--guard-wait-secs の値が Duration の表現範囲を超えている: '{arg}'"
                ));
            }
            out.guard_wait_secs = Some(value);
            guard_wait_secs_seen = true;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--env-info-out=") {
            if env_info_out_seen {
                return Err(format!(
                    "--env-info-out は複数回指定できない（重複指定）: '{arg}'"
                ));
            }
            if rest.is_empty() {
                return Err(format!("--env-info-out の値が空文字列: '{arg}'"));
            }
            out.env_info_out = Some(rest.to_string());
            env_info_out_seen = true;
            continue;
        }
        match arg.as_str() {
            "--phase1-only" => {
                if out.phase1_only {
                    return Err(format!(
                        "--phase1-only は複数回指定できない（重複指定）: '{arg}'"
                    ));
                }
                out.phase1_only = true;
            }
            "--gpu-timestamps" => {
                if out.gpu_timestamps {
                    return Err(format!(
                        "--gpu-timestamps は複数回指定できない（重複指定）: '{arg}'"
                    ));
                }
                out.gpu_timestamps = true;
            }
            "--guard-only" => {
                if out.guard_only {
                    return Err(format!(
                        "--guard-only は複数回指定できない（重複指定）: '{arg}'"
                    ));
                }
                out.guard_only = true;
            }
            _ if arg.starts_with("--min-warmup-secs=") => {
                if min_warmup_secs.is_some() {
                    return Err(format!(
                        "--min-warmup-secs は複数回指定できない（重複指定）: '{arg}'"
                    ));
                }
                min_warmup_secs = Some(parse_min_warmup_secs(&arg)?);
            }
            _ => {
                return Err(format!(
                    "未知の引数: '{arg}'（許可される引数は --phase1-only／--gpu-timestamps／\
                     --max-load-avg=<f64>／--gpu-watch=<name>／--guard-max-attempts=<usize>／\
                     --guard-wait-secs=<f64>／--guard-only／--env-info-out=<path>／\
                     --min-warmup-secs=<N> のみ）"
                ));
            }
        }
    }
    if !out.gpu_watch.is_empty() && out.max_load_avg.is_none() {
        return Err("--gpu-watch は --max-load-avg 指定時のみ有効（単独指定はエラー）".to_string());
    }
    out.min_warmup_secs = min_warmup_secs;
    Ok(out)
}

/// `--min-warmup-secs=<N>` の値部分（`arg` は `--min-warmup-secs=` で
/// 始まる前提）を解析し、`3 ≤ N ≤ 600` の範囲検証まで行う（イシュー
/// #1261）。下限 3 は既定値 [`DEFAULT_MIN_WARMUP_SECS`]（3 秒）を下回る
/// 「減らす方向」の指定を拒否するため（原因候補 (b) の切り分けは
/// ウォームアップを**増やす**方向の試行に限定する計画上の制約。
/// `docs/perf/metal-gemm-transpose-tiled.md` §5.6 参照）。上限 600 は
/// 誤指定（桁間違い等）による長時間占有事故を防ぐための安全弁。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn parse_min_warmup_secs(arg: &str) -> Result<u64, String> {
    let value = arg
        .strip_prefix("--min-warmup-secs=")
        .expect("呼び出し元が prefix 一致を確認済み");
    if value.is_empty() {
        return Err(format!("--min-warmup-secs の値が空: '{arg}'"));
    }
    let n: u64 = value
        .parse()
        .map_err(|_| format!("--min-warmup-secs の値が非数値: '{arg}'"))?;
    if !(3..=600).contains(&n) {
        return Err(format!(
            "--min-warmup-secs は 3〜600 の範囲でなければならない（既定値 3 秒を下回る指定・\
             600 秒超の指定はいずれも拒否）。指定値: '{arg}'"
        ));
    }
    Ok(n)
}

/// [`round_extrema`] の戻り値。`phase1_round_stats` 行の `min_secs`／
/// `max_secs` とその 0 始まり index を保持する（イシュー #1251）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq)]
struct RoundExtrema {
    min_secs: f64,
    min_round_idx: usize,
    max_secs: f64,
    max_round_idx: usize,
}

/// `round_medians_secs`（[`bench_harness::ab::StabilityResult::
/// round_medians_secs`]。秒単位）から最小・最大ラウンドを求める純関数。
///
/// **秒基準**で判定する点が契約: TFLOPS へ換算すると大小関係が逆転する
/// （レイテンシ比と TFLOPS 比の取り違えは #540/#746 で一度発生した既知の
/// 落とし穴。本モジュール内 `b_over_a_tflops` の doc comment と同じ設計
/// 判断で、集約はここへ 1 箇所に留める）ため、呼び出し元
/// （`phase1_stability_selfcheck`）は本関数の戻り値をそのまま
/// `phase1_round_stats` 行へ出力すればよい。
///
/// 同値タイは**最初に出現した** index を採る（決定的）。空スライスは
/// `None`（fail-closed。`run_stability` は `rounds >= 2` を保証する契約
/// 上ここへは到達しない想定だが、契約として明示的に扱う）。`NaN` は
/// `run_stability`（内部で `stats::relative_spread` を呼ぶ）が計測時点で
/// 既に `BenchError::NanSample` として拒否するため、本関数へは到達しない
/// 前提とする。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn round_extrema(round_medians_secs: &[f64]) -> Option<RoundExtrema> {
    if round_medians_secs.is_empty() {
        return None;
    }
    let (mut min_idx, mut max_idx) = (0usize, 0usize);
    for (i, &v) in round_medians_secs.iter().enumerate().skip(1) {
        if v < round_medians_secs[min_idx] {
            min_idx = i;
        }
        if v > round_medians_secs[max_idx] {
            max_idx = i;
        }
    }
    Some(RoundExtrema {
        min_secs: round_medians_secs[min_idx],
        min_round_idx: min_idx,
        max_secs: round_medians_secs[max_idx],
        max_round_idx: max_idx,
    })
}

/// [`bench_harness::ab::StabilityResult::aux`]（[`bench_harness::ab::
/// AuxiliarySpread`]。イシュー #1483）を `phase1_round_stats` 行の末尾
/// キー群（`trimmed_spread_k1=`／`iqr_spread=`／`mad_spread=`）へ整形する
/// 純関数（イシュー #1484）。
///
/// キー → フィールド対応（`AuxiliarySpread` doc comment と同じ意味を
/// 保つ。ここで新たな意味を作らない）:
/// - `trimmed_spread_k1` ← `aux.trimmed`（`_k1` 接尾辞は
///   [`bench_harness::ab::AUXILIARY_TRIM_PER_SIDE`] = 1 を表す。定数と
///   キー名がずれていないことは呼び出し側テストで
///   `assert_eq!(AUXILIARY_TRIM_PER_SIDE, 1)` により固定する）
/// - `iqr_spread` ← `aux.iqr_over_median`
/// - `mad_spread` ← `aux.mad2_over_median`（**2·MAD/median** である点は
///   キー名だけでは分からないため、この doc comment と
///   `docs/perf/metal-gemm-transpose-tiled.md` のキー一覧に明記する）
///
/// `trimmed` が `None`（ラウンド数がトリム後 2 要素未満）の場合は既存
/// sentinel `NA`（同 example の `phase1_gpu_host_stats` 行
/// `kernel_gpu_median_secs=NA` 等と同じ表記。`opt` 関数参照）を使う。新規
/// sentinel は導入しない。
///
/// **本関数の戻り値・呼び出し元は補助値を [`bench_harness::ab::
/// STABILITY_SPREAD_GATE`] 等の閾値と比較する判定へ転用してはならない**
/// （`AuxiliarySpread` doc comment・`.claude/rules/security.md` のガード
/// レール閾値単独緩和禁止と同じ理由）。あくまでレポート専用の追記。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn format_aux_spread_keys(aux: &bench_harness::ab::AuxiliarySpread) -> String {
    let trimmed_str = match aux.trimmed {
        Some(v) => format!("{v:.4e}"),
        None => "NA".to_string(),
    };
    format!(
        "trimmed_spread_k1={trimmed_str} iqr_spread={:.4e} mad_spread={:.4e}",
        aux.iqr_over_median, aux.mad2_over_median
    )
}

/// `phase1_round_stats` 行（機械可読な 1 行。`grep '^phase1_round_stats '`
/// で既存ログ〈`docs/perf/logs/metal-gemm-transpose-route-ab-1242/`〉と
/// 突合できる形式）を組み立てる純関数（イシュー #1484）。
///
/// 既存キー（`rounds`・`spread`・`gate`・`within_gate`・`median_secs`・
/// `min_secs`／`min_round_idx`・`max_secs`／`max_round_idx`・
/// `round_medians_secs`）の並び・書式は
/// [`phase1_stability_selfcheck`] が元々直書きしていた `println!` と
/// byte 単位で同一（イシュー #1249/#1251 の既存契約を維持）。
/// [`format_aux_spread_keys`] による 3 キー（`trimmed_spread_k1`／
/// `iqr_spread`／`mad_spread`。イシュー #1483 の `StabilityResult::aux`）を
/// **末尾に追記するのみ**で、既存キーの意味・順序・判定
/// （`within_gate = result.spread <= gate`。本関数は判定を行わず呼び出し
/// 元が計算済みの値をそのまま受け取る）は一切変更しない。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn format_phase1_round_stats_line(
    size: usize,
    result: &bench_harness::ab::StabilityResult,
    gate: f64,
    within_gate: bool,
    extrema: &RoundExtrema,
) -> String {
    let median_secs = bench_harness::median_q1_q3(&result.round_medians_secs)
        .expect("run_stability が返す round_medians_secs は非空・非 NaN のため成功する")
        .median;
    let round_medians_secs_str = result
        .round_medians_secs
        .iter()
        .map(|s| format!("{s:.6e}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "phase1_round_stats size={size} rounds={} spread={:.4e} gate={gate:.4e} \
         within_gate={within_gate} median_secs={median_secs:.6e} \
         min_secs={:.6e} min_round_idx={} max_secs={:.6e} max_round_idx={} \
         round_medians_secs={round_medians_secs_str} {}",
        result.round_medians_secs.len(),
        result.spread,
        extrema.min_secs,
        extrema.min_round_idx,
        extrema.max_secs,
        extrema.max_round_idx,
        format_aux_spread_keys(&result.aux),
    )
}

/// `env_guard_result=` 行の値を導出する純関数（イシュー #1265。Review 指摘
/// 対応: 以前は `args.max_load_avg` の有無だけで `pass`／`record_only` を
/// 決め打ちしていたため、`run_guard_with_retry` が `Undetermined`（load
/// average 取得不能等）でも `Ok` を返す契約〈`bench_harness::env_guard`
/// モジュール doc「取得不能は未判定・ブロック要因にしない」〉の下では、直前
/// の `env_guard_overall verdict=undetermined` と矛盾する `pass` を出力
/// しえた）。
///
/// - `gated == false`（`--max-load-avg` 未指定）: 判定を行わないため常に
///   `record_only`（`GuardRetryOutcome::record_only` の `overall` は
///   `Undetermined` 固定だが、これは「未判定」であり「取得不能」とは区別する）
/// - `gated == true`: `final_report.overall` をそのまま写像する。`Fail` は
///   `run_guard_with_retry` が `Err(EnvGuardExhausted)` を返すため `Ok`
///   経路では現れないが、panic 経路を作らず `fail` へ写像しておく
///   （fail-closed）
///
/// `macos_impl::main` から呼ばれる（フェーズ 1・2 の各ガード後）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn env_guard_result_label(gated: bool, overall: bench_harness::ab::GuardVerdict) -> &'static str {
    use bench_harness::ab::GuardVerdict;
    if !gated {
        return "record_only";
    }
    match overall {
        GuardVerdict::Pass => "pass",
        GuardVerdict::Undetermined => "undetermined",
        GuardVerdict::Fail => "fail",
    }
}

/// `--gpu-timestamps` opt-in 時、フェーズ 1 の計装クロージャが 1 回の
/// ワークロード呼び出しごとに記録する 1 サンプル（イシュー #1259）。
///
/// `closure_wall_secs` は upload〜readback（クロージャ本体の計測区間の
/// 内側）のみを覆う `Instant` 計測で、クロージャ末尾で暗黙に発生する
/// `a_buf`/`b_buf`/`c_buf` の解放（drop）時間を**含まない**（診断用の
/// 参考値に限る）。安定性判定（`bench_harness::protocol::run`）が実際に
/// 使う壁時計サンプル（`Measurement::samples_secs`。クロージャ呼び出し
/// 全体を `Instant` で挟むため drop 時間を含む）とは計測区間が異なり、
/// drop 時のスパイクが `closure_wall_secs` には反映されない
/// （codex-review 指摘。イシュー #1261）。このため
/// [`aggregate_gpu_host_round`] の `wall`／`wall_minus_gpu` 系集計は
/// `closure_wall_secs` ではなく呼び出し元が `Measurement::samples_secs`
/// から渡す `measured_wall_secs` を単一真実源として使う
/// （[`aggregate_gpu_host_round`] ドキュメンテーションコメント参照）。
/// `commit_wait_secs` は `MetalContext::synchronize_with_gpu_timestamps`
/// 呼び出し自体の `Instant` 計測（commit + `waitUntilCompleted` +
/// タイムスタンプ取得）、`kernel_gpu_secs` はその中で得られた
/// `GPUEndTime − GPUStartTime`（`BatchGpuTimestamps::kernel_gpu_secs`）。
/// `batches_len` はそのバッチに含まれていたディスパッチ数（1 個の GEMM
/// ディスパッチのみが載っていたことの検証に使う。`gemm_reuse_phase_
/// diag_tests.rs` の不変条件と同じ理由）。
///
/// `resolved_cfg` は [`MetalGemm::encode_tiled_prepared`] の**戻り値**
/// （`{cfg:?}` 形式）——呼び出し時に渡した要求構成（`tile::
/// select_for_device` の解決値）そのものではなく、`pipeline_for_tile` が
/// フォールバック解決を行った後に実際に実行したカーネルの構成を記録する
/// （codex-review 指摘。フォールバック発生時は要求構成と乖離しうるため、
/// 診断ラベルは常に実行結果側を出す。イシュー #1261）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone)]
struct GpuHostSample {
    closure_wall_secs: f64,
    upload_secs: f64,
    alloc_secs: f64,
    encode_secs: f64,
    commit_wait_secs: f64,
    kernel_gpu_secs: Option<f64>,
    readback_secs: f64,
    batches_len: usize,
    resolved_cfg: String,
}

/// `samples` のうち末尾 `iters` 件（測定対象サンプル）を返す。
///
/// `bench_harness::ab::run_stability_observed` のラウンド完了フックは
/// 「直前に積まれた `measurement.iters` 件が測定対象」という契約
/// （`run_stability_observed` ドキュメンテーションコメント参照）を
/// 提供するため、本関数はそれをそのままスライスへ変換する。`samples`
/// の長さが `iters` 未満の場合は `None`（fail-closed。warmup 呼び出し分
/// を誤って含めて集計してしまう事故を防ぐ。呼び出し元が到達すると
/// `run_stability_observed` 自体の契約違反を意味するため `expect` で
/// panic させる想定）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn measured_tail<T>(samples: &[T], iters: usize) -> Option<&[T]> {
    if samples.len() < iters {
        return None;
    }
    Some(&samples[samples.len() - iters..])
}

/// [`aggregate_gpu_host_round`] の戻り値。フェーズ 1 の 1 ラウンド分の
/// GPU/host 内訳中央値（イシュー #1259）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone, PartialEq)]
struct GpuHostRoundStats {
    round: usize,
    iters: usize,
    /// `true` なら `tail` の全サンプルが不変条件（`batches_len==1`・
    /// `kernel_gpu_secs` が `Some`・`0 ≤ kernel_gpu ≤ commit_wait ≤ wall`）
    /// を満たした（fail-closed。1 件でも違反すれば `false`）。
    valid: bool,
    kernel_gpu_median_secs: Option<f64>,
    /// 安定性判定（`bench_harness::protocol::run`）が使う壁時計サンプル
    /// （`Measurement::samples_secs`）の中央値。`GpuHostSample::
    /// closure_wall_secs` ではなく `aggregate_gpu_host_round` の
    /// `measured_wall_secs` 引数から計算する（同関数ドキュメンテーション
    /// コメント参照）。
    wall_median_secs: f64,
    /// `GpuHostSample::closure_wall_secs`（クロージャ内側のみ、バッファ
    /// 解放時間を含まない `Instant` 計測）の中央値。診断専用の参考値
    /// （`wall_median_secs` との差が大きければ、drop 時間が無視できない
    /// ことを示唆する）であり、`wall_minus_gpu` 系の判定には使わない
    /// （イシュー #1261）。
    closure_wall_median_secs: f64,
    /// `wall − kernel_gpu_secs`（`wall` は上記 `Measurement::samples_secs`
    /// 由来）を**サンプルごとに差を取ってから**中央値化した値
    /// （`median(wall) − median(kernel_gpu)` ではない。
    /// `gemm_reuse_phase_diag_tests.rs`〈PR #1371 レビュー教訓〉と同じ
    /// 理由）。
    wall_minus_gpu_median_secs: Option<f64>,
    commit_wait_median_secs: f64,
    /// `commit_wait_secs − kernel_gpu_secs` の同様の差分中央値。
    commit_wait_minus_gpu_median_secs: Option<f64>,
    upload_median_secs: f64,
    alloc_median_secs: f64,
    encode_median_secs: f64,
    readback_median_secs: f64,
    /// `GpuHostSample::resolved_cfg`（`encode_tiled_prepared` の戻り値。
    /// 実際に実行したカーネル構成）から求めた、このラウンドの構成ラベル。
    /// 全サンプルで一致していればその値、不一致なら `MIXED(...)`
    /// （`aggregate_gpu_host_round` ドキュメンテーションコメント参照。
    /// イシュー #1261）。
    resolved_cfg: String,
}

/// `tail`（[`measured_tail`] が返す、あるラウンドの測定対象サンプル列）
/// から [`GpuHostRoundStats`] を集計する純関数（イシュー #1259）。
///
/// `measured_wall_secs` は呼び出し元（`run_stability_gpu_host`）が
/// `bench_harness::protocol::run` の `Measurement::samples_secs` から
/// そのまま渡す、安定性判定（`protocol::run`）自身が使う壁時計サンプル
/// である。`tail`（同じ呼び出し列から生成された [`GpuHostSample`]）と
/// インデックスが 1:1 で対応する契約——`tail` は同一クロージャの呼び出し
/// 順に積まれ、`measured_wall_secs` も同じ呼び出し順で記録されるため
/// （`run_stability_gpu_host` ドキュメンテーションコメント参照）。
/// `GpuHostSample::closure_wall_secs`（クロージャ内側のみの `Instant`
/// 計測。バッファ〈`a_buf`/`b_buf`/`c_buf`〉解放時間を含まない）は
/// `wall`／`wall_minus_gpu` 系集計には**使わない**——解放時のスパイクが
/// 安定性判定の対象区間（`Measurement::samples_secs`）には乗るのに
/// `closure_wall_secs` には乗らず、`spread_wall`／`wall_minus_gpu` が
/// 実際のばらつきを過小評価してしまうため（codex-review 指摘。
/// イシュー #1261）。
///
/// `wall_minus_gpu`／`commit_wait_minus_gpu` はサンプルごとに差を取って
/// から中央値を計算する契約（[`GpuHostRoundStats`] フィールドドキュメント
/// 参照）。不変条件違反サンプルは `kernel_gpu`／差分系列の集計対象から
/// 除外し、1 件でも違反があれば `valid=false` として `kernel_gpu_median_
/// secs`・両差分中央値を `None` にする（fail-closed。host 側フェーズ
/// 内訳〈upload/alloc/encode/commit_wait/readback/wall〉自体は GPU
/// タイムスタンプに依存しないため、`valid=false` でも中央値を計算する）。
///
/// `resolved_cfg` は `tail` の各サンプルが持つ [`GpuHostSample::
/// resolved_cfg`]（`encode_tiled_prepared` の戻り値。実際に実行した
/// カーネル構成）から求める。同一ラウンド内の全サンプルで一致していれば
/// その値をそのまま使い、`pipeline_for_tile` のフォールバック挙動が
/// 呼び出しの途中で変化して一致しない場合は `MIXED(...)`（全値をカンマ
/// 区切りで列挙）として可視化する（fail-closed。要求構成のラベルへ
/// 黙って戻さない。codex-review 指摘。イシュー #1261）。
///
/// # Panics
///
/// `tail.len() != measured_wall_secs.len()` の場合（呼び出し元の契約
/// 違反。両者は同じ呼び出し列から生成されるため通常発生しない）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn aggregate_gpu_host_round(
    round: usize,
    iters: usize,
    tail: &[GpuHostSample],
    measured_wall_secs: &[f64],
) -> GpuHostRoundStats {
    assert_eq!(
        tail.len(),
        measured_wall_secs.len(),
        "tail と measured_wall_secs は同一呼び出し列から生成される契約のため \
         長さが一致するはず（run_stability_gpu_host 参照）"
    );

    let resolved_cfg = match tail.first() {
        Some(first) if tail.iter().all(|s| s.resolved_cfg == first.resolved_cfg) => {
            first.resolved_cfg.clone()
        }
        Some(_) => format!(
            "MIXED({})",
            tail.iter()
                .map(|s| s.resolved_cfg.as_str())
                .collect::<Vec<_>>()
                .join(",")
        ),
        None => "NA".to_string(),
    };

    let median = |xs: &[f64]| -> f64 {
        bench_harness::median_q1_q3(xs)
            .expect("tail は run_stability_observed 契約により非空のはず")
            .median
    };

    let wall: Vec<f64> = measured_wall_secs.to_vec();
    let closure_wall: Vec<f64> = tail.iter().map(|s| s.closure_wall_secs).collect();
    let commit_wait: Vec<f64> = tail.iter().map(|s| s.commit_wait_secs).collect();
    let upload: Vec<f64> = tail.iter().map(|s| s.upload_secs).collect();
    let alloc: Vec<f64> = tail.iter().map(|s| s.alloc_secs).collect();
    let encode: Vec<f64> = tail.iter().map(|s| s.encode_secs).collect();
    let readback: Vec<f64> = tail.iter().map(|s| s.readback_secs).collect();

    let mut valid = !tail.is_empty();
    let mut kernel_gpu_samples: Vec<f64> = Vec::with_capacity(tail.len());
    let mut wall_minus_gpu_samples: Vec<f64> = Vec::with_capacity(tail.len());
    let mut commit_wait_minus_gpu_samples: Vec<f64> = Vec::with_capacity(tail.len());
    for (idx, s) in tail.iter().enumerate() {
        if s.batches_len != 1 {
            valid = false;
            continue;
        }
        let Some(kernel_gpu) = s.kernel_gpu_secs else {
            valid = false;
            continue;
        };
        let wall_true = measured_wall_secs[idx];
        if !(kernel_gpu >= 0.0
            && kernel_gpu <= s.commit_wait_secs
            && s.commit_wait_secs <= wall_true)
        {
            valid = false;
            continue;
        }
        kernel_gpu_samples.push(kernel_gpu);
        wall_minus_gpu_samples.push(wall_true - kernel_gpu);
        commit_wait_minus_gpu_samples.push(s.commit_wait_secs - kernel_gpu);
    }

    let (kernel_gpu_median_secs, wall_minus_gpu_median_secs, commit_wait_minus_gpu_median_secs) =
        if valid && !kernel_gpu_samples.is_empty() {
            (
                Some(median(&kernel_gpu_samples)),
                Some(median(&wall_minus_gpu_samples)),
                Some(median(&commit_wait_minus_gpu_samples)),
            )
        } else {
            (None, None, None)
        };

    GpuHostRoundStats {
        round,
        iters,
        valid,
        kernel_gpu_median_secs,
        wall_median_secs: median(&wall),
        closure_wall_median_secs: median(&closure_wall),
        wall_minus_gpu_median_secs,
        commit_wait_median_secs: median(&commit_wait),
        commit_wait_minus_gpu_median_secs,
        upload_median_secs: median(&upload),
        alloc_median_secs: median(&alloc),
        encode_median_secs: median(&encode),
        readback_median_secs: median(&readback),
        resolved_cfg,
    }
}

/// `phase1_gpu_host_round` 行（機械可読・`grep '^phase1_gpu_host_round '`）
/// を組み立てる。値なし（`None`）のフィールドは `NA` を出力する（イシュー
/// #1259）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn format_gpu_host_round_line(stats: &GpuHostRoundStats, size: usize) -> String {
    let opt = |x: Option<f64>| {
        x.map(|v| format!("{v:.6e}"))
            .unwrap_or_else(|| "NA".to_string())
    };
    format!(
        "phase1_gpu_host_round size={size} round={} iters={} kernel_gpu_median_secs={} \
         wall_median_secs={:.6e} closure_wall_median_secs={:.6e} \
         wall_minus_gpu_median_secs={} commit_wait_median_secs={:.6e} \
         commit_wait_minus_gpu_median_secs={} upload_median_secs={:.6e} alloc_median_secs={:.6e} \
         encode_median_secs={:.6e} readback_median_secs={:.6e} resolved_cfg={} valid={}",
        stats.round,
        stats.iters,
        opt(stats.kernel_gpu_median_secs),
        stats.wall_median_secs,
        stats.closure_wall_median_secs,
        opt(stats.wall_minus_gpu_median_secs),
        stats.commit_wait_median_secs,
        opt(stats.commit_wait_minus_gpu_median_secs),
        stats.upload_median_secs,
        stats.alloc_median_secs,
        stats.encode_median_secs,
        stats.readback_median_secs,
        stats.resolved_cfg,
        stats.valid,
    )
}

/// [`aggregate_gpu_host_size`] の戻り値。フェーズ 1 の 1 サイズ分の
/// ラウンド間ばらつき総括（イシュー #1259）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone, PartialEq)]
struct GpuHostSizeStats {
    size: usize,
    rounds: usize,
    /// `valid=true` だったラウンド数。
    valid_rounds: usize,
    spread_kernel_gpu: Option<f64>,
    max_round_idx_kernel_gpu: Option<usize>,
    spread_wall: f64,
    max_round_idx_wall: usize,
    spread_wall_minus_gpu: Option<f64>,
    max_round_idx_wall_minus_gpu: Option<usize>,
    /// ラウンド別 `kernel_gpu_median_secs`（無効ラウンドは `None`）。
    kernel_gpu_round_medians_secs: Vec<Option<f64>>,
    /// ラウンド別 `wall_median_secs`（GPU タイムスタンプ非依存のため常に
    /// 値を持つ）。
    wall_round_medians_secs: Vec<f64>,
}

/// `rounds`（サイズ 1 個分の [`GpuHostRoundStats`] 列。ラウンド順）から
/// [`GpuHostSizeStats`] を集計する純関数（イシュー #1259）。
///
/// `spread_kernel_gpu`／`spread_wall_minus_gpu` とその `max_round_idx_*`
/// は、無効ラウンド（`valid=false`）が 1 つでも混じっていれば `None`
/// とする（元のラウンド index との対応関係が崩れる部分集合だけの
/// spread 計算はしない。fail-closed）。`spread_wall` は GPU タイムスタンプ
/// に依存しないため常に計算する（[`bench_harness::relative_spread`]・
/// [`round_extrema`] は `rounds` が非空である `run_stability` の契約
/// 〈`AbConfig::rounds >= 2`〉により常に成功する前提）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn aggregate_gpu_host_size(size: usize, rounds: &[GpuHostRoundStats]) -> GpuHostSizeStats {
    let wall_medians: Vec<f64> = rounds.iter().map(|r| r.wall_median_secs).collect();
    let kernel_gpu_medians: Vec<Option<f64>> = rounds
        .iter()
        .map(|r| {
            if r.valid {
                r.kernel_gpu_median_secs
            } else {
                None
            }
        })
        .collect();
    let wall_minus_gpu_medians: Vec<Option<f64>> = rounds
        .iter()
        .map(|r| {
            if r.valid {
                r.wall_minus_gpu_median_secs
            } else {
                None
            }
        })
        .collect();

    let valid_rounds = rounds.iter().filter(|r| r.valid).count();

    let spread_wall = bench_harness::relative_spread(&wall_medians)
        .expect("wall_medians は run_stability の契約〈rounds>=2・非 NaN〉により成功する");
    let max_round_idx_wall = round_extrema(&wall_medians)
        .expect("wall_medians は run_stability の契約〈rounds>=2〉により非空")
        .max_round_idx;

    // `Option` 系列を「1 個でも欠損があれば全体を諦める」方式で集約する
    // 共有ヘルパ（`spread_kernel_gpu`／`spread_wall_minus_gpu` の両方で
    // 同じロジックを使うため 1 箇所に集約する）。
    let spread_and_max_idx = |series: &[Option<f64>]| -> (Option<f64>, Option<usize>) {
        if series.is_empty() || series.iter().any(Option::is_none) {
            return (None, None);
        }
        let present: Vec<f64> = series
            .iter()
            .map(|x| x.expect("is_none 済み検査"))
            .collect();
        match (
            bench_harness::relative_spread(&present),
            round_extrema(&present),
        ) {
            (Ok(spread), Some(extrema)) => (Some(spread), Some(extrema.max_round_idx)),
            _ => (None, None),
        }
    };

    let (spread_kernel_gpu, max_round_idx_kernel_gpu) = spread_and_max_idx(&kernel_gpu_medians);
    let (spread_wall_minus_gpu, max_round_idx_wall_minus_gpu) =
        spread_and_max_idx(&wall_minus_gpu_medians);

    GpuHostSizeStats {
        size,
        rounds: rounds.len(),
        valid_rounds,
        spread_kernel_gpu,
        max_round_idx_kernel_gpu,
        spread_wall,
        max_round_idx_wall,
        spread_wall_minus_gpu,
        max_round_idx_wall_minus_gpu,
        kernel_gpu_round_medians_secs: kernel_gpu_medians,
        wall_round_medians_secs: wall_medians,
    }
}

/// `phase1_gpu_host_stats` 行（機械可読・`grep '^phase1_gpu_host_stats '`）
/// を組み立てる。値なし（`None`）のフィールドは `NA` を出力する（イシュー
/// #1259）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn format_gpu_host_size_line(stats: &GpuHostSizeStats) -> String {
    let opt_f64 = |x: Option<f64>| {
        x.map(|v| format!("{v:.4e}"))
            .unwrap_or_else(|| "NA".to_string())
    };
    let opt_usize = |x: Option<usize>| x.map(|v| v.to_string()).unwrap_or_else(|| "NA".to_string());
    let kernel_gpu_series = stats
        .kernel_gpu_round_medians_secs
        .iter()
        .map(|x| {
            x.map(|v| format!("{v:.6e}"))
                .unwrap_or_else(|| "NA".to_string())
        })
        .collect::<Vec<_>>()
        .join(",");
    let wall_series = stats
        .wall_round_medians_secs
        .iter()
        .map(|v| format!("{v:.6e}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "phase1_gpu_host_stats size={} rounds={} valid={} spread_kernel_gpu={} \
         max_round_idx_kernel_gpu={} spread_wall={:.4e} max_round_idx_wall={} \
         spread_wall_minus_gpu={} max_round_idx_wall_minus_gpu={} \
         kernel_gpu_round_medians_secs={kernel_gpu_series} wall_round_medians_secs={wall_series}",
        stats.size,
        stats.rounds,
        stats.valid_rounds,
        opt_f64(stats.spread_kernel_gpu),
        opt_usize(stats.max_round_idx_kernel_gpu),
        stats.spread_wall,
        stats.max_round_idx_wall,
        opt_f64(stats.spread_wall_minus_gpu),
        opt_usize(stats.max_round_idx_wall_minus_gpu),
    )
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use bench_harness::BenchError;
    use bench_harness::MeasurementConfig;
    use bench_harness::ab::{
        AbConfig, EnvGuardConfig, EnvSample, GuardRetryOutcome, RetryConfig, format_env_info_text,
        run_ab, run_guard_with_retry, run_stability, run_stability_observed,
    };
    use bench_harness::rng::Xorshift64Star;
    use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
    use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, tile};
    use std::cell::RefCell;
    use std::time::{Duration, Instant};
    // イシュー #1249/#1251: `--phase1-only` 引数解析・ラウンド別 min/max
    // 集計は macOS 依存部分を持たない top-level 純関数（本モジュール外）。
    // イシュー #1259: `--gpu-timestamps` の集計・出力も同様に top-level
    // 純関数（`GpuHostSample`・`GpuHostRoundStats`・`measured_tail`・
    // `aggregate_gpu_host_round`／`_size`・`format_gpu_host_round_line`／
    // `_size_line`）へ切り出してある。
    use super::{
        CliArgs, GpuHostRoundStats, GpuHostSample, aggregate_gpu_host_round,
        aggregate_gpu_host_size, format_gpu_host_round_line, format_gpu_host_size_line,
        format_phase1_round_stats_line, measured_tail, min_warmup_override_duration, round_extrema,
    };

    /// `gemm_transpose_tile_sweep.rs`・`gemm_bench.rs` と同一値（決定的
    /// シード。過去 PoC・CPU 実装ベンチと同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// フェーズ 1・フェーズ 2 共通のラウンド数・cooldown・時間ベース
    /// ウォームアップ下限（`gemm_swizzle_ab_bench.rs` の既定値
    /// 〈ROUNDS=6・COOLDOWN=2s・MIN_WARMUP=1s〉から増やす方向のみ調整
    /// 済み。実機実行時、他プロセスの並行 GPU 負荷（同一マシンで兄弟
    /// イシューの GPU 計測が並走。`uptime` 実測 load average 3〜8）で
    /// フェーズ 1 の安定性ゲートを繰り返し満たせなかったため段階的に
    /// 増やした（`docs/perf/metal-bench-noise-protocol.md` の調整手順に
    /// 従う。安全側判断: 判定閾値〈`STABILITY_SPREAD_GATE`〉自体は変更
    /// しない）。
    const ROUNDS: usize = 10;
    const COOLDOWN: Duration = Duration::from_secs(8);
    // ウォームアップ下限の既定値は [`super::DEFAULT_MIN_WARMUP_SECS`]
    // （単一真実源）を [`super::min_warmup_override_duration`] 経由で使う
    // （イシュー #1261。`--min-warmup-secs=<N>` 未指定時は同値になる）。
    // 本モジュール内に `MIN_WARMUP` 定数として複製すると
    // `phase1_stability_selfcheck`／`phase2_route_ab` の呼び出し側が両方
    // ともオーバーライド経由の呼び出しへ切り替わった結果 dead_code に
    // なるため、定数としては保持しない。

    // --- イシュー #1265: 実行前環境ガードのバックオフ再試行既定値 --------
    //
    // `ROUNDS`／`COOLDOWN`／`MIN_WARMUP` と同様、調整は増やす方向のみ許容
    // する（`docs/perf/metal-bench-noise-protocol.md`）。閾値
    // （`--max-load-avg`）自体の既定値はユーザー承認事項のため、ここには
    // 待機・再試行回数のみを置く（`bench_harness::env_guard::EnvGuardConfig`
    // は既定値・`Default` を持たない設計。同モジュール doc 参照）。
    const GUARD_INITIAL_WAIT: Duration = Duration::from_secs(30);
    const GUARD_GROWTH_FACTOR: f64 = 1.5;
    const GUARD_MAX_WAIT: Duration = Duration::from_secs(300);
    const GUARD_MAX_ATTEMPTS: usize = 10;

    /// フェーズ 1・フェーズ 2 の各開始前に呼ぶ実行前環境ガード（イシュー
    /// #1265）。`args.max_load_avg` 指定時は **gated**（`EnvGuardConfig`
    /// による判定・`Fail` ならバックオフ再試行）、未指定時は **record_only**
    /// （[`GuardRetryOutcome::record_only`]。判定なし・記録のみ）で動作する。
    /// `EnvGuardConfig::new`（閾値の有限・正値検査）は呼び出し元
    /// `parse_args_from` が同じ条件を検証済みのため通常到達せず、
    /// `RetryConfig::new` の `max_wait >= initial_wait` 制約は
    /// `--guard-wait-secs` が `GUARD_MAX_WAIT` を上回る指定（例:
    /// `--guard-wait-secs=400`）でも本関数が `max_wait` を `initial_wait`
    /// 以上へ持ち上げるため構築失敗しない。それでも `Err` になった場合は
    /// fail-closed に同じ `BenchError` 経路で呼び出し元へ伝播する。I/O・待機は
    /// すべて `bench_harness::env_guard` に委譲し、本関数は設定の組み立てと
    /// 呼び分けのみを行う。計測区間（`run_stability`／`run_ab`）の外側で
    /// 完結するため、`ab::STABILITY_SPREAD_GATE` 等の判定ロジックには
    /// 影響しない。
    fn run_env_guard(
        args: &CliArgs,
    ) -> Result<(GuardRetryOutcome, Option<EnvGuardConfig>), BenchError> {
        match args.max_load_avg {
            Some(max_load_avg) => {
                let config = EnvGuardConfig::new(max_load_avg)?
                    .with_gpu_process_watchlist(args.gpu_watch.clone());
                let initial_wait = args
                    .guard_wait_secs
                    .map(Duration::from_secs_f64)
                    .unwrap_or(GUARD_INITIAL_WAIT);
                let max_attempts = args.guard_max_attempts.unwrap_or(GUARD_MAX_ATTEMPTS);
                // `RetryConfig::new` は `max_wait >= initial_wait` を要求する
                // （`bench_harness::env_guard::RetryConfig::new` doc 参照）。
                // `--guard-wait-secs` は利用者指定で `GUARD_MAX_WAIT`
                // （既定 300s）を上回りうるため、cap を initial_wait 未満に
                // 落とさないよう max で持ち上げる（fail-closed に構築失敗させ
                // ないための単純な整合。`GUARD_MAX_WAIT` 自体の既定値は不変）。
                let max_wait = GUARD_MAX_WAIT.max(initial_wait);
                let retry =
                    RetryConfig::new(initial_wait, GUARD_GROWTH_FACTOR, max_wait, max_attempts)?;
                let outcome = run_guard_with_retry(&config, &retry)?;
                Ok((outcome, Some(config)))
            }
            None => Ok((GuardRetryOutcome::record_only(EnvSample::collect()), None)),
        }
    }

    /// [`run_env_guard`] の結果を env_info テキストへ変換し、stdout へ出力
    /// する（常時）とともに、`args.env_info_out` が指定されていれば同じ
    /// テキストを追記する（イシュー #1265・opt-in）。ファイル出力に失敗
    /// した場合は「記録できない実行を記録済みと誤認しない」ため fail-closed
    /// に stderr へ理由を出して exit(1) する（`.claude/rules/security.md`
    /// A01/A05 相当の慎重さ）。`OpenOptions::create(true).append(true)` で
    /// 開くため既存内容の上書き（truncate）は行わず末尾追記のみとなる。
    /// 一方、指定パスがシンボリックリンクであれば OS の通常の解決に従い
    /// リンク先へ追記する（シンボリックリンクの検出・拒否は行わない。出力先
    /// は利用者が `--env-info-out` で明示した docs/perf/logs 配下のログ
    /// ファイルを想定し、権限・パスの妥当性は利用者側の責務とする）。
    fn emit_env_info(
        label: &str,
        outcome: &GuardRetryOutcome,
        config: Option<&EnvGuardConfig>,
        args: &CliArgs,
    ) {
        let text = format_env_info_text(label, outcome, config);
        print!("{text}");
        if let Some(path) = &args.env_info_out {
            use std::io::Write;
            let result = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut f| f.write_all(text.as_bytes()));
            if let Err(e) = result {
                eprintln!("env_info_out への書き込みに失敗した（path={path}）: {e}");
                std::process::exit(1);
            }
        }
    }

    /// [`run_env_guard`] が `Err` を返した場合の共通処理（イシュー #1265）。
    /// `EnvGuardExhausted`（再試行上限到達）を含め、いずれの `BenchError`
    /// も `verdict=undetermined` として出力し非ゼロ終了する
    /// （フェーズ 1 不成立時〈既存経路〉と同じ `verdict=` grep 運用に揃える。
    /// GPU は未初期化のまま終了できる位置でのみ呼ぶ想定 — フェーズ 1 前は
    /// `MetalContext::new()` より前、フェーズ 2 前は既にコンテキスト保持
    /// 済みだが追加の GPU 操作は行わずに終了する）。
    ///
    /// 以前は stdout のみへ出力していたため、`--env-info-out` 指定時でも
    /// 再試行上限到達（exhausted）で終了した実行がログファイルへ一切
    /// 残らなかった（[`emit_env_info`] は `Ok` 経路の成功試行でしか呼ばれ
    /// ないため。Review 指摘。#1265）。`emit_env_info` と同じ
    /// `OpenOptions::create(true).append(true)` 方式で `label`／`args` を
    /// 受け取り、stdout と同一テキストをファイルへも追記することで
    /// 「記録できない実行を記録済みと誤認しない」という既存の fail-closed
    /// 方針（書き込み失敗時は stderr へ理由を出して exit(1)）を exhausted
    /// 経路にも一貫適用する。
    fn abort_on_guard_error(label: &str, err: &BenchError, args: &CliArgs) -> ! {
        // `env_guard_result=` は行頭キーとして出力する（フェーズ 1/2 成功時
        // の `env_guard_result={} attempts={}` 行〈1798/1883 行目〉と同じ
        // 位置づけに揃え、`grep '^env_guard_result='` で成功・失敗いずれの
        // 経路も一律に拾えるようにする。`env_guard_label` は同じ行の末尾
        // 側の付加情報として出す（Bugbot 指摘。#1265。合わせて
        // `crates/bench-harness/src/env_guard.rs::summarize_exhausted` の
        // 内訳キーを `verdict=` から `state=` へ変更し、後続の `verdict=`
        // 行との衝突〈同一行に `verdict=` が複数回出現し行頭 grep が
        // 崩れる〉を解消済み）。
        let text = match err {
            BenchError::EnvGuardExhausted { attempts, detail } => {
                format!(
                    "env_guard_result=exhausted attempts={attempts} env_guard_label={label}\n\
                     verdict=undetermined (環境ガード上限到達: {detail})\n"
                )
            }
            other => {
                format!(
                    "env_guard_result=error env_guard_label={label}\n\
                     verdict=undetermined (環境ガード設定エラー: {other})\n"
                )
            }
        };
        print!("{text}");
        if let Some(path) = &args.env_info_out {
            use std::io::Write;
            let result = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut f| f.write_all(text.as_bytes()));
            if let Err(e) = result {
                eprintln!("env_info_out への書き込みに失敗した（path={path}）: {e}");
                std::process::exit(1);
            }
        }
        std::process::exit(1);
    }

    fn tflops(m: usize, n: usize, k: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        flops / median_secs / 1e12
    }

    /// `AbResult` の a/b 中央値秒数から `b_over_a_tflops`（TFLOPS 比。
    /// B が A より速ければ 1.0 超）を計算する共有ヘルパ。
    /// `result.b_over_a_ratio` は実行時間の比（レイテンシ比）であり
    /// TFLOPS 比はその逆数になる——取り違えは #540/#746（PR #763）で
    /// 一度発生した既知の落とし穴のため、1 箇所に集約して
    /// フェーズ間で重複実装しない（`gemm_swizzle_ab_bench.rs::
    /// head_over_base_tflops_ratio` と同じ設計判断）。
    fn b_over_a_tflops(
        m: usize,
        n: usize,
        k: usize,
        median_a_secs: f64,
        median_b_secs: f64,
    ) -> f64 {
        tflops(m, n, k, median_b_secs) / tflops(m, n, k, median_a_secs)
    }

    /// `logical`（行優先の論理 `[rows, cols]`）から `[cols, rows]` 行優先の
    /// 転置済み物理バッファを作る（`gemm_transpose_tile_sweep.rs`・
    /// `tests/gemm_strided_parity.rs` と同一ロジックの複製。いずれも
    /// 別コンパイル単位のため共有できない）。
    fn transpose_dense(logical: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = logical[r * cols + c];
            }
        }
        out
    }

    /// A/B 対象形状（`gemm_transpose_tile_sweep.rs::shapes()` と同一。
    /// `tile::select_with_occupancy` の分岐クラスを代表する点）。
    fn shapes() -> Vec<(usize, usize, usize)> {
        vec![
            // 正方立方（#744 実測点の再現確認）
            (512, 512, 512),
            (1024, 1024, 1024),
            (2048, 2048, 2048),
            (4096, 4096, 4096),
            // K 未実測の正方出力
            (2048, 2048, 64),
            (2048, 2048, 512),
            // 準正方長方形（縦横比 < 2）
            (1536, 1024, 1024),
            (1024, 1536, 1536),
            // 縦長・横長（縦横比 >= 2）
            (4096, 1024, 1024),
            (1024, 4096, 1024),
        ]
    }

    /// `--gpu-timestamps` opt-in 時、`dispatch_auto`
    /// （`GemmVariant::SimdgroupTiled` 分岐）と同一組成
    /// （upload → alloc → encode → commit_wait → readback）を公開 API
    /// （[`MetalGemm::encode_tiled_prepared`]・
    /// [`MetalContext::synchronize_with_gpu_timestamps`]）で再現しつつ、
    /// `Instant` によるホスト側フェーズ内訳と GPU タイムスタンプ
    /// （`kernel_gpu`）を呼び出しごとに記録する（イシュー #1259。
    /// example ヘッダ doc comment「GPU タイムスタンプ分離計測モード」
    /// 参照）。
    ///
    /// 対象サイズ（256〜4096）は全て 8 の倍数のため `pad_matrix`/
    /// `unpad_matrix` は no-op・`c_buf` は `dispatch_auto` と同じ専有
    /// 確保（本 example の `MetalContext` はプロセスワイド singleton
    /// ではないため `MetalBuffer::alloc_uninit_pooled`〈`dispatch_auto`
    /// が内部で使う〉は元々 `new_zeroed` へフォールバックする。本関数は
    /// その `new_zeroed` を直接呼ぶ）。戻り値の [`bench_harness::ab::
    /// StabilityResult`]（`round_medians_secs`・`spread`）は
    /// `run_stability`（`dispatch_auto` 直呼び）と同じ壁時計 `Instant`
    /// 計測（`protocol::run`）から得るため、呼び出し元
    /// （`phase1_stability_selfcheck`）の判定ロジックは分岐に依らない。
    fn run_stability_gpu_host(
        ctx: &MetalContext,
        gemm: &MetalGemm,
        ab_config: &AbConfig,
        measurement_config: &MeasurementConfig,
        size: usize,
    ) -> bench_harness::ab::StabilityResult {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(size * size);
        let b = rng.fill_vec(size * size);
        // `tile::select_for_device` はループの外（ラウンド計測の外）で
        // 1 回だけ解決する——`dispatch_auto` も呼び出しごとに再解決するが
        // 決定的（`(m, n, k)` とデバイス情報のみに依存）なため、計測ループ
        // 内で毎回呼んでも呼ばなくても値は不変。この `cfg` は
        // `encode_tiled_prepared` への**要求**構成であり、`pipeline_for_tile`
        // がフォールバックする可能性があるため、診断ラベル
        // （`GpuHostSample::resolved_cfg`）は呼び出しごとの**戻り値**から
        // 別途記録する（codex-review 指摘。イシュー #1261）。
        let cfg = tile::select_for_device(size, size, size, ctx.verified_m4_max_gpu_core_count());

        let samples: RefCell<Vec<GpuHostSample>> = RefCell::new(Vec::new());
        let round_stats: RefCell<Vec<GpuHostRoundStats>> = RefCell::new(Vec::new());

        let mut workload = || {
            let wall_start = Instant::now();

            let t = Instant::now();
            let a_buf = MetalBuffer::new_with_data(ctx, &a)
                .expect("upload A に失敗した（実機でのみ実行する前提）");
            let b_buf = MetalBuffer::new_with_data(ctx, &b)
                .expect("upload B に失敗した（実機でのみ実行する前提）");
            let upload_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let c_buf = MetalBuffer::new_zeroed(ctx, size * size)
                .expect("alloc C に失敗した（実機でのみ実行する前提）");
            let alloc_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let resolved_cfg_actual = gemm
                .encode_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, size, size, size, cfg)
                .expect("encode_tiled_prepared に失敗した（実機でのみ実行する前提）");
            let encode_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let batches = ctx
                .synchronize_with_gpu_timestamps()
                .expect("synchronize_with_gpu_timestamps に失敗した（実機でのみ実行する前提）");
            let commit_wait_secs = t.elapsed().as_secs_f64();

            let t = Instant::now();
            let c = c_buf.read_to_vec();
            std::hint::black_box(&c);
            let readback_secs = t.elapsed().as_secs_f64();

            let closure_wall_secs = wall_start.elapsed().as_secs_f64();
            let kernel_gpu_secs = if batches.len() == 1 {
                batches[0].kernel_gpu_secs()
            } else {
                None
            };

            samples.borrow_mut().push(GpuHostSample {
                closure_wall_secs,
                upload_secs,
                alloc_secs,
                encode_secs,
                commit_wait_secs,
                kernel_gpu_secs,
                readback_secs,
                batches_len: batches.len(),
                resolved_cfg: format!("{resolved_cfg_actual:?}"),
            });
        };

        let result = run_stability_observed(
            ab_config,
            measurement_config,
            &mut workload,
            |round, measurement| {
                let all_samples = samples.borrow();
                let tail = measured_tail(all_samples.as_slice(), measurement.iters).expect(
                    "run_stability_observed の契約〈直前 iters 件が測定対象〉により \
                     samples の長さは常に iters 以上のはず",
                );
                // 安定性判定（`protocol::run`）自身が使う壁時計サンプルを
                // そのまま渡す——`tail`（`closure_wall_secs`）は drop 時間を
                // 含まないため使わない（`aggregate_gpu_host_round`
                // ドキュメンテーションコメント参照。イシュー #1261）。
                let stats = aggregate_gpu_host_round(
                    round,
                    measurement.iters,
                    tail,
                    measurement.samples_secs.as_slice(),
                );
                println!("{}", format_gpu_host_round_line(&stats, size));
                round_stats.borrow_mut().push(stats);
            },
        )
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        let size_stats = aggregate_gpu_host_size(size, &round_stats.borrow());
        println!("{}", format_gpu_host_size_line(&size_stats));

        result
    }

    /// フェーズ 1: 対照カーネルとして `dispatch_auto`（`dispatch_auto` の
    /// 本番既定経路と同一構成選択。`gemm_swizzle_ab_bench.rs::
    /// phase1_stability_selfcheck` と同一手法）を各サイズで
    /// [`run_stability`] 計測し、spread を出力する。
    ///
    /// `gpu_timestamps=true`（`--gpu-timestamps`。イシュー #1259）の場合、
    /// 対照ワークロードを [`run_stability_gpu_host`] （`dispatch_auto`
    /// と同一組成の計装版）へ**置換**する。壁時計判定（`result.spread`・
    /// `within_gate`・`phase1_round_stats`／`phase1_summary` 等の既存出力）
    /// は両分岐で完全に同一の計算経路（`run_stability_observed` の壁時計
    /// `Instant` 計測。`run_stability` はこれを no-op フックで呼ぶ薄い
    /// ラッパー）を通るため、以下の判定ロジック自体は分岐に依らず不変。
    fn phase1_stability_selfcheck(
        ctx: &MetalContext,
        gemm: &MetalGemm,
        gpu_timestamps: bool,
        min_warmup_secs: Option<u64>,
    ) -> bool {
        // 安定性ゲートの値自体は `bench_harness::ab::STABILITY_SPREAD_GATE`
        // を単一真実源とする（`docs/perf/metal-bench-noise-protocol.md` と
        // 同じ値を example 内に直接複製すると、閾値変更時にコードと文書が
        // 独立に乖離しうるため。`gemm_swizzle_ab_bench.rs` と同じ判断）。
        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;
        println!("--- フェーズ 1: 安定性セルフチェック（対照カーネル: dispatch_auto）---");

        // `--min-warmup-secs=<N>`（イシュー #1261）が指定されていれば既定
        // `MIN_WARMUP`（3 秒）の代わりに使う。フェーズ 1・フェーズ 2 の
        // 両方へ一貫適用する契約（`min_warmup_override_duration`
        // ドキュメンテーションコメント参照）に従い、本関数は呼び出し元
        // （`main`）から渡された値をそのまま使う。
        let effective_min_warmup = min_warmup_override_duration(min_warmup_secs);
        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, effective_min_warmup)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        let mut all_within_gate = true;
        let mut gate_exceeded_sizes: Vec<usize> = Vec::new();
        for size in [256usize, 512, 1024, 2048, 4096] {
            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let result = if gpu_timestamps {
                run_stability_gpu_host(ctx, gemm, &ab_config, &measurement_config, size)
            } else {
                run_stability(&ab_config, &measurement_config, || {
                    gemm.dispatch_auto(ctx, &a, &b, size, size, size)
                        .expect("Metal GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
                })
                .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない")
            };

            let within_gate = result.spread <= SPREAD_GATE;
            all_within_gate &= within_gate;
            if !within_gate {
                gate_exceeded_sizes.push(size);
            }

            let round_tflops: Vec<f64> = result
                .round_medians_secs
                .iter()
                .map(|&secs| tflops(size, size, size, secs))
                .collect();
            println!(
                "size={size} spread={:.4} ({}) round_tflops={round_tflops:.4?}",
                result.spread,
                if within_gate { "OK" } else { "NG: gate 超過" }
            );

            // イシュー #1249/#1251: サイズごとのラウンド別中央値・spread を
            // 機械可読な 1 行で追加出力する（`grep '^phase1_round_stats '`
            // で既存ログ〈docs/perf/logs/metal-gemm-transpose-route-ab-1186/・
            // -1187/〉と同じ突合ができるよう、上記の既存行はバイト単位で
            // 変更せず直後に追加するのみ）。`round_extrema` の契約どおり
            // min/max は秒基準・0 始まり index。イシュー #1484:
            // `StabilityResult::aux`（トリム済みレンジ／IQR／MAD ベースの
            // 補助 spread 統計量。判定には使わない）を行末へ追記する整形は
            // [`format_phase1_round_stats_line`] に集約する。
            if let Some(extrema) = round_extrema(&result.round_medians_secs) {
                println!(
                    "{}",
                    format_phase1_round_stats_line(
                        size,
                        &result,
                        SPREAD_GATE,
                        within_gate,
                        &extrema
                    )
                );
            }
        }

        let sizes_gate_exceeded_str = if gate_exceeded_sizes.is_empty() {
            "none".to_string()
        } else {
            gate_exceeded_sizes
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        println!(
            "phase1_summary sizes_measured=5 sizes_gate_exceeded={sizes_gate_exceeded_str} \
             all_within_gate={all_within_gate}"
        );

        if !all_within_gate {
            println!(
                "--- フェーズ 1 判定: 一部サイズが spread ≤{SPREAD_GATE:.2} 相当を満たさなかった。\
                 フェーズ 2（A/B 判定）はスキップする（安全側判断: 判定不可のまま採否を確定しない）。"
            );
        }
        all_within_gate
    }

    /// [`measure_route_ab_cell`] の成功時の戻り値。`spread_a`/`spread_b`
    /// を呼び出し元（`phase2_route_ab`）の総括判定へ伝える——本 example の
    /// ヘッダ doc comment が約束する「安定性ゲート超過セルが残れば
    /// `verdict=undetermined`」は、フェーズ 1 の対照カーネルだけでなく
    /// フェーズ 2 の A/B 計測自体のラウンド間ばらつきにも適用される契約
    /// のため、比だけでなく spread も呼び出し元へ返す必要がある。
    struct CellResult {
        b_over_a_tflops: f64,
        spread_a: f64,
        spread_b: f64,
        /// `tile::select_for_device` が要求した構成と
        /// `dispatch_strided_tiled_prepared` が実際にフォールバック解決
        /// した構成が一致したか。不一致（`false`）は「#1187 が渡す構成
        /// そのものを測る」というベンチ契約が満たせていないセルを意味する
        /// ため、呼び出し元（`phase2_route_ab`）は本フィールドを見て
        /// `route_ok`/`route_ng` の判定材料（`ratios`）に含めず
        /// `verdict=undetermined` へ倒す（fail-closed。ログ出力のみで
        /// `ratios` へ加算すると、要求構成が使えない場合でも別のフォール
        /// バック構成がたまたま高速なら `route_ok` になり得てしまう）。
        resolved_matches_requested: bool,
    }

    /// フェーズ 2 総括で安定性ゲート超過セルを記録する行（`Vec` の要素型が
    /// clippy `type_complexity` に触れるタプルにならないよう構造体化）。
    struct GateExceededCell {
        shape: (usize, usize, usize),
        pattern: &'static str,
        spread_a: f64,
        spread_b: f64,
    }

    /// 転置パターン（NT/TN/TT）1 種・1 形状について A（classic strided）/
    /// B（strided tiled variant）を [`run_ab`] で interleaved 計測する。
    /// 戻り値は B が `Err(StridedTiledIneligible)` を返した場合 `None`
    /// （fail-closed skip。黙って除外せず理由を出力する。10 形状は全て
    /// 8 整除・ld 4 整除のため通常は発生しない想定だが、契約として扱う）。
    /// B が成功しても要求構成とフォールバック解決後の構成が不一致な場合は
    /// `CellResult::resolved_matches_requested=false` を返し、呼び出し元
    /// が undetermined へ倒す判断材料とする（#1187 が渡す構成そのものを
    /// 測るというベンチ契約の fail-closed 担保）。
    ///
    /// `#[allow(clippy::too_many_arguments)]`:
    /// `dispatch_strided_bias_act_prepared`／`dispatch_strided_tiled_prepared`
    /// 自体が個別引数方式（構造体へまとめ込まない設計判断）のため、その
    /// 計測ラッパーである本関数も同じ形状の引数列を持つ
    /// （`gemm_transpose_tile_sweep.rs::measure_transposed` と同じ判断）。
    #[allow(clippy::too_many_arguments)]
    fn measure_route_ab_cell(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        pattern_label: &str,
        ab_config: &AbConfig,
        measurement_config: &MeasurementConfig,
    ) -> Option<CellResult> {
        let mut rng = Xorshift64Star::new(SEED);
        let a_logical = rng.fill_vec(m * k);
        let b_logical = rng.fill_vec(k * n);

        let (a_phys, a_layout): (Vec<f32>, MatrixLayout) = if trans_a {
            (
                transpose_dense(&a_logical, m, k),
                classify_2d(&[m, k], &[1, m as isize]).expect("転置 A view の分類に失敗した"),
            )
        } else {
            (
                a_logical,
                classify_2d(&[m, k], &[k as isize, 1]).expect("行優先 A view の分類に失敗した"),
            )
        };
        let (b_phys, b_layout): (Vec<f32>, MatrixLayout) = if trans_b {
            (
                transpose_dense(&b_logical, k, n),
                classify_2d(&[k, n], &[1, k as isize]).expect("転置 B view の分類に失敗した"),
            )
        } else {
            (
                b_logical,
                classify_2d(&[k, n], &[n as isize, 1]).expect("行優先 B view の分類に失敗した"),
            )
        };

        let a_buf = MetalBuffer::new_with_data(ctx, &a_phys)
            .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let b_buf = MetalBuffer::new_with_data(ctx, &b_phys)
            .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let c_buf = MetalBuffer::new_zeroed(ctx, m * n)
            .expect("C バッファの確保に失敗した（実機でのみ実行する前提）");

        // #1187 の結線が渡す構成そのものを計測する（`dispatch_auto` の
        // 本番既定経路と同一の選択ロジック）。
        let cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());

        // B 側の適格性を計測前に確認する（fail-closed skip）。
        let head_resolved = match gemm.dispatch_strided_tiled_prepared(
            ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k, cfg,
        ) {
            Ok(resolved) => resolved,
            Err(e) => {
                println!("shape=({m},{n},{k}) pattern={pattern_label} skipped reason={e}");
                return None;
            }
        };
        let resolved_matches_requested = head_resolved == cfg;

        // A 側（現状の本番経路）を計測前に 1 回実行して成立を確認する。
        gemm.dispatch_strided_bias_act_prepared(
            ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, None, false, &c_buf, m, n, k,
        )
        .expect("dispatch_strided_bias_act_prepared に失敗した（実機でのみ実行する前提）");

        let result = run_ab(
            ab_config,
            measurement_config,
            || {
                gemm.dispatch_strided_bias_act_prepared(
                    ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, None, false, &c_buf, m, n, k,
                )
                .expect("直前に成功した構成が計測ループ中に失敗することはない想定");
            },
            || {
                gemm.dispatch_strided_tiled_prepared(
                    ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, &c_buf, m, n, k, cfg,
                )
                .expect("直前に成功した構成が計測ループ中に失敗することはない想定");
            },
        )
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        let a_tflops: Vec<f64> = result
            .a_round_medians_secs
            .iter()
            .map(|&secs| tflops(m, n, k, secs))
            .collect();
        let b_tflops: Vec<f64> = result
            .b_round_medians_secs
            .iter()
            .map(|&secs| tflops(m, n, k, secs))
            .collect();

        let a_median_tflops = tflops(m, n, k, result.median_a_secs);
        let b_median_tflops = tflops(m, n, k, result.median_b_secs);
        let ratio = b_over_a_tflops(m, n, k, result.median_a_secs, result.median_b_secs);

        println!(
            "shape=({m},{n},{k}) pattern={pattern_label} cfg={}x{}x{}_wm{}wn{} \
             resolved_matches_requested={resolved_matches_requested} \
             a_median_tflops={a_median_tflops:.4} b_median_tflops={b_median_tflops:.4} \
             b_over_a_tflops={ratio:.4} spread_a={:.4} spread_b={:.4} \
             a_round_tflops={a_tflops:.4?} b_round_tflops={b_tflops:.4?}",
            cfg.bm, cfg.bn, cfg.bk, cfg.wm, cfg.wn, result.spread_a, result.spread_b,
        );

        Some(CellResult {
            b_over_a_tflops: ratio,
            spread_a: result.spread_a,
            spread_b: result.spread_b,
            resolved_matches_requested,
        })
    }

    /// フェーズ 2: 全 10 形状 × NT/TN/TT（計 30 セル）の A/B を計測し、
    /// 総括判定（`verdict`）を出力する。
    fn phase2_route_ab(ctx: &MetalContext, gemm: &MetalGemm, min_warmup_secs: Option<u64>) {
        println!(
            "--- フェーズ 2: 転置タイル variant ルーティング A/B（A=classic strided / B=strided tiled）---"
        );

        // フェーズ 1 と同じ実効値を使う（イシュー #1261。片方だけ変えると
        // A/B 側と対照側で warmup が異なる非対称が生じるため）。
        let effective_min_warmup = min_warmup_override_duration(min_warmup_secs);
        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, effective_min_warmup)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        // 安定性ゲートの値自体は `bench_harness::ab::STABILITY_SPREAD_GATE`
        // を単一真実源とする（`phase1_stability_selfcheck` と同じ判断）。
        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;

        let mut ratios: Vec<((usize, usize, usize), &'static str, f64)> = Vec::new();
        let mut skipped: Vec<((usize, usize, usize), &'static str)> = Vec::new();
        let mut gate_exceeded: Vec<GateExceededCell> = Vec::new();
        // 要求構成と実際にフォールバック解決された構成が不一致だったセル
        // （`CellResult::resolved_matches_requested=false`）。`ratios` へは
        // 加算せず undetermined 判定の材料にする（下記 codex-review 指摘
        // 対応。#1187 が渡す構成そのものを測るというベンチ契約の担保）。
        let mut resolution_mismatched: Vec<((usize, usize, usize), &'static str)> = Vec::new();

        for (m, n, k) in shapes() {
            for (trans_a, trans_b, label) in
                [(false, true, "NT"), (true, false, "TN"), (true, true, "TT")]
            {
                match measure_route_ab_cell(
                    gemm,
                    ctx,
                    m,
                    n,
                    k,
                    trans_a,
                    trans_b,
                    label,
                    &ab_config,
                    &measurement_config,
                ) {
                    Some(cell) => {
                        // 本 example のヘッダ doc comment（モジュール冒頭）が
                        // 約束する「安定性ゲート超過セルが残れば
                        // verdict=undetermined」契約: フェーズ 2 の各セル自体
                        // の spread も判定材料に含める（フェーズ 1 の対照
                        // カーネルだけを見ると、A/B 本計測自体がノイズで
                        // 揺れているセルを route_ok/route_ng へ fail-open で
                        // 倒してしまう）。
                        if cell.spread_a > SPREAD_GATE || cell.spread_b > SPREAD_GATE {
                            gate_exceeded.push(GateExceededCell {
                                shape: (m, n, k),
                                pattern: label,
                                spread_a: cell.spread_a,
                                spread_b: cell.spread_b,
                            });
                        }
                        if cell.resolved_matches_requested {
                            ratios.push(((m, n, k), label, cell.b_over_a_tflops));
                        } else {
                            // codex-review 指摘対応（PR #1198）: 要求構成
                            // （`tile::select_for_device`）と
                            // `dispatch_strided_tiled_prepared` が実際に
                            // フォールバック解決した構成が不一致のセルは
                            // 「#1187 が渡す構成そのものを測る」契約を満た
                            // さないため `ratios` へ加算せず undetermined
                            // へ倒す（route_ok/route_ng の判定対象から除外）。
                            resolution_mismatched.push(((m, n, k), label));
                        }
                    }
                    None => skipped.push(((m, n, k), label)),
                }
            }
        }

        let below_threshold: Vec<_> = ratios.iter().filter(|(_, _, ratio)| *ratio < 1.0).collect();

        let verdict = if !skipped.is_empty()
            || !gate_exceeded.is_empty()
            || !resolution_mismatched.is_empty()
        {
            "undetermined"
        } else if below_threshold.is_empty() {
            "route_ok"
        } else {
            "route_ng"
        };

        let min_ratio = ratios
            .iter()
            .min_by(|a, b| a.2.partial_cmp(&b.2).expect("TFLOPS 比は常に有限値"));

        println!("--- フェーズ 2 総括 ---");
        println!(
            "cells_measured={} cells_skipped={} cells_below_threshold={} cells_gate_exceeded={} \
             cells_resolution_mismatched={}",
            ratios.len(),
            skipped.len(),
            below_threshold.len(),
            gate_exceeded.len(),
            resolution_mismatched.len()
        );
        if let Some(((m, n, k), label, ratio)) = min_ratio {
            println!("min_b_over_a_tflops={ratio:.4} at shape=({m},{n},{k}) pattern={label}");
        }
        for ((m, n, k), label) in &skipped {
            println!("skipped_cell shape=({m},{n},{k}) pattern={label}");
        }
        for cell in &gate_exceeded {
            let (m, n, k) = cell.shape;
            let (label, spread_a, spread_b) = (cell.pattern, cell.spread_a, cell.spread_b);
            println!(
                "gate_exceeded_cell shape=({m},{n},{k}) pattern={label} spread_a={spread_a:.4} spread_b={spread_b:.4}"
            );
        }
        for ((m, n, k), label) in &resolution_mismatched {
            println!(
                "resolution_mismatched_cell shape=({m},{n},{k}) pattern={label} \
                 reason=head_resolved_config_differs_from_requested_config"
            );
        }
        for ((m, n, k), label, ratio) in &below_threshold {
            println!(
                "below_threshold_cell shape=({m},{n},{k}) pattern={label} b_over_a_tflops={ratio:.4}"
            );
        }
        println!(
            "verdict={verdict} ({})",
            match verdict {
                "route_ok" => {
                    "全形状 × NT/TN/TT で B/A(TFLOPS) >= 1.0 かつ全セル spread が \
                     gate 内。結線可（#1187 で dispatch_strided_bias_act_prepared \
                     への自動ルーティングを実装しうる）"
                }
                "route_ng" => {
                    "1 セル以上で B/A(TFLOPS) < 1.0（かつ全セル spread は gate 内）。\
                     全形状基準未達のため現状の判断基準では結線不可"
                }
                _ => {
                    "skip セル・spread gate 超過セル・resolution 不一致\
                     セルのいずれかが残っており判定不可（適格性ゲート不成立、\
                     ラウンド間ばらつきが大きく計測値を信頼できない、または\
                     要求構成と実際にフォールバック解決された構成が不一致の\
                     いずれか）"
                }
            }
        );
        println!(
            "--- 実測結果は docs/perf/metal-gemm-transpose-tiled.md §5 へ記録すること（本番経路・テストは無変更のまま）。"
        );
    }

    pub fn main() {
        // イシュー #1249/#1251: 引数解析は `MetalContext::new()`（GPU 初期化）
        // より前に行う。不正引数なら GPU を一切触らずに終了する
        // （`super::parse_args_from` の doc comment 参照。OWASP A03 観点）。
        let args = match super::parse_args_from(std::env::args().skip(1)) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("引数解析エラー: {e}");
                std::process::exit(1);
            }
        };

        // イシュー #1265: フェーズ 1 前の実行前環境ガード。`MetalContext::new()`
        // （GPU 初期化）より前に行い、gated モードで上限到達（`Fail` が
        // バックオフ再試行を使い切っても解消しない）した場合は GPU を
        // 一切触らずに終了する。
        let (phase1_guard_outcome, phase1_guard_config) = match run_env_guard(&args) {
            Ok(v) => v,
            Err(e) => abort_on_guard_error("phase1", &e, &args),
        };
        emit_env_info(
            "phase1",
            &phase1_guard_outcome,
            phase1_guard_config.as_ref(),
            &args,
        );
        println!(
            "env_guard_result={} attempts={}",
            super::env_guard_result_label(
                args.max_load_avg.is_some(),
                phase1_guard_outcome.final_report.overall,
            ),
            phase1_guard_outcome.attempts_used()
        );

        if args.guard_only {
            // イシュー #1265: ガード＋env_info 出力のみ行い GPU を初期化
            // せず終了する（短時間動作確認・実行前チェック用）。
            println!(
                "verdict=not_evaluated (--guard-only: 環境ガードの記録のみで GEMM 計測は未実行)"
            );
            return;
        }

        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        if args.phase1_only {
            println!("mode=phase1_only");
        }
        if args.gpu_timestamps {
            // イシュー #1259: フェーズ 1 の対照ワークロードが計装版
            // （`run_stability_gpu_host`）へ置換されていることを示す
            // マーカー。`size=…`／`phase1_round_stats` 行がこの組成で
            // 得た値であることを識別する。
            println!("phase1_workload=gpu_timestamps");
        }
        if let Some(n) = args.min_warmup_secs {
            // イシュー #1261: 原因候補 (b)（ウォームアップ不足）切り分け用。
            // 実効値をログ自身に残すことで、e.g. `1261-aggregate.py` が
            // どの run がどの MIN_WARMUP で計測されたかを再現できる。
            println!("phase1_min_warmup_override_secs={n}");
        }

        let phase1_ok =
            phase1_stability_selfcheck(&ctx, &gemm, args.gpu_timestamps, args.min_warmup_secs);

        if args.phase1_only {
            // イシュー #1249/#1251: `--phase1-only` はフェーズ 1 が安定性
            // ゲートを満たしたか否かに関わらず、常にフェーズ 2（A/B 判定）を
            // 実行しなかったことを示す `not_evaluated` を出力して終了する
            // （`undetermined`〈判定不可〉とは意味が異なるため使い分ける。
            // #1253/#1255 が `verdict=` grep で本モードのログを区別できる
            // ようにする）。
            println!(
                "verdict=not_evaluated (--phase1-only: フェーズ 2〈A/B 判定〉は未実行。\
                 #1249 の spread 分布記録用)"
            );
            return;
        }

        if !phase1_ok {
            // codex-review 指摘対応（PR #1198）: フェーズ 1 不成立での早期
            // return もフェーズ 2 総括（`phase2_route_ab`）と同じ
            // `verdict=undetermined` 行を出力する。全終了経路で判定形式を
            // 統一し、ログを機械的に `verdict=` grep するだけで判定を
            // 一意に読み取れるようにする（フェーズ 2 到達時のみ verdict
            // 行が出る非対称を解消する）。
            println!(
                "verdict=undetermined (フェーズ 1 の安定性セルフチェックで \
                 spread ≤gate 相当を満たさないサイズが残ったため、フェーズ 2\
                 （A/B 判定）を実行せず判定不可のまま終了する)"
            );
            return;
        }

        // イシュー #1265: フェーズ 2（A/B 判定）前の実行前環境ガード。
        // フェーズ 1 と同じ設定を再実測して再判定する（フェーズ 1 実行中に
        // 負荷が上昇した場合を検知するため。GPU コンテキストは既に構築
        // 済みだが、ここでは追加の GPU 操作は行わずガード結果のみで終了
        // 判断する）。
        let (phase2_guard_outcome, phase2_guard_config) = match run_env_guard(&args) {
            Ok(v) => v,
            Err(e) => abort_on_guard_error("phase2", &e, &args),
        };
        emit_env_info(
            "phase2",
            &phase2_guard_outcome,
            phase2_guard_config.as_ref(),
            &args,
        );
        println!(
            "env_guard_result={} attempts={}",
            super::env_guard_result_label(
                args.max_load_avg.is_some(),
                phase2_guard_outcome.final_report.overall,
            ),
            phase2_guard_outcome.attempts_used()
        );

        phase2_route_ab(&ctx, &gemm, args.min_warmup_secs);
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け stub（`gemm_bench.rs` と同じ理由。`objc2` 系は
/// `cfg(target_os = "macos")` 限定のため本クレートの GEMM 実装自体が
/// コンパイル対象外になる。Linux CI の `cargo build --workspace
/// --all-targets`／`cargo clippy --all-targets` をこの example も含めて
/// 通すための最小 main）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_transpose_route_ab_bench example requires macOS (Apple Silicon). \
         See docs/perf/metal-bench-noise-protocol.md and \
         docs/perf/metal-gemm-transpose-tiled.md for the real-hardware execution procedure."
    );
}

/// `parse_args_from`／`round_extrema`（イシュー #1251）の純関数ユニット
/// テスト。macOS 依存部分を一切持たないため Linux CI（`cargo test -p
/// fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench`。
/// `Cargo.toml` の `[[example]] test = true` により実行対象）でも走る。
#[cfg(test)]
mod cli_and_round_stats_tests {
    use super::{
        CliArgs, GpuHostRoundStats, GpuHostSample, RoundExtrema, aggregate_gpu_host_round,
        aggregate_gpu_host_size, env_guard_result_label, format_aux_spread_keys,
        format_gpu_host_round_line, format_gpu_host_size_line, format_phase1_round_stats_line,
        measured_tail, min_warmup_override_duration, parse_args_from, parse_min_warmup_secs,
        round_extrema,
    };
    use bench_harness::ab::{
        AUXILIARY_TRIM_PER_SIDE, AuxiliarySpread, GuardVerdict, StabilityResult,
    };

    fn args(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    /// イシュー #1265 Review 指摘対応: `env_guard_result=` は
    /// `final_report.overall` から導出し、gated かつ `Undetermined` で
    /// `pass` を出さない。
    #[test]
    fn env_guard_result_label_record_only_ignores_verdict() {
        for v in [
            GuardVerdict::Pass,
            GuardVerdict::Fail,
            GuardVerdict::Undetermined,
        ] {
            assert_eq!(env_guard_result_label(false, v), "record_only");
        }
    }

    #[test]
    fn env_guard_result_label_gated_maps_verdict() {
        assert_eq!(env_guard_result_label(true, GuardVerdict::Pass), "pass");
        assert_eq!(
            env_guard_result_label(true, GuardVerdict::Undetermined),
            "undetermined"
        );
        assert_eq!(env_guard_result_label(true, GuardVerdict::Fail), "fail");
    }

    #[test]
    fn parse_args_from_empty_defaults_to_phase1_only_false() {
        let parsed = parse_args_from(args(&[])).expect("空引数列は成功するはず");
        assert_eq!(
            parsed,
            CliArgs {
                phase1_only: false,
                gpu_timestamps: false,
                ..Default::default()
            }
        );
    }

    #[test]
    fn parse_args_from_phase1_only_flag_sets_true() {
        let parsed =
            parse_args_from(args(&["--phase1-only"])).expect("既知の単一引数は成功するはず");
        assert_eq!(
            parsed,
            CliArgs {
                phase1_only: true,
                gpu_timestamps: false,
                ..Default::default()
            }
        );
    }

    #[test]
    fn parse_args_from_duplicate_phase1_only_is_error() {
        let err = parse_args_from(args(&["--phase1-only", "--phase1-only"]))
            .expect_err("重複指定は fail-closed に拒否するはず");
        assert!(err.contains("複数回指定できない"));
    }

    #[test]
    fn parse_args_from_unknown_argument_is_error() {
        for unknown in ["--bogus", "phase1-only", "--phase1-only=1"] {
            let err = parse_args_from(args(&[unknown]))
                .expect_err("未知の引数は fail-closed に拒否するはず");
            assert!(err.contains("未知の引数"), "unknown={unknown} err={err}");
        }
    }

    #[test]
    fn parse_args_from_gpu_timestamps_flag_sets_true() {
        let parsed =
            parse_args_from(args(&["--gpu-timestamps"])).expect("既知の単一引数は成功するはず");
        assert_eq!(
            parsed,
            CliArgs {
                phase1_only: false,
                gpu_timestamps: true,
                ..Default::default()
            }
        );
    }

    #[test]
    fn parse_args_from_both_flags_combine_regardless_of_order() {
        // イシュー #1259: `--phase1-only`／`--gpu-timestamps` は順序不問で併用可。
        for combo in [
            ["--phase1-only", "--gpu-timestamps"],
            ["--gpu-timestamps", "--phase1-only"],
        ] {
            let parsed = parse_args_from(args(&combo)).expect("順序不問で併用できるはず");
            assert_eq!(
                parsed,
                CliArgs {
                    phase1_only: true,
                    gpu_timestamps: true,
                    ..Default::default()
                }
            );
        }
    }

    #[test]
    fn parse_args_from_duplicate_gpu_timestamps_is_error() {
        let err = parse_args_from(args(&["--gpu-timestamps", "--gpu-timestamps"]))
            .expect_err("重複指定は fail-closed に拒否するはず");
        assert!(err.contains("複数回指定できない"));
    }

    // --- イシュー #1265: 環境ガード CLI 引数 ----------------------------

    #[test]
    fn parse_args_from_max_load_avg_sets_gated_mode() {
        let parsed =
            parse_args_from(args(&["--max-load-avg=8.5"])).expect("有効な値は成功するはず");
        assert_eq!(parsed.max_load_avg, Some(8.5));
    }

    #[test]
    fn parse_args_from_max_load_avg_rejects_non_numeric() {
        let err = parse_args_from(args(&["--max-load-avg=abc"]))
            .expect_err("非数値は fail-closed に拒否するはず");
        assert!(err.contains("数値として解釈できない"));
    }

    #[test]
    fn parse_args_from_max_load_avg_rejects_zero_and_negative() {
        for v in ["0", "-1.0"] {
            let err = parse_args_from(args(&[&format!("--max-load-avg={v}")]))
                .expect_err("0・負値は fail-closed に拒否するはず");
            assert!(err.contains("有限かつ正"), "v={v} err={err}");
        }
    }

    #[test]
    fn parse_args_from_max_load_avg_duplicate_is_error() {
        let err = parse_args_from(args(&["--max-load-avg=4.0", "--max-load-avg=8.0"]))
            .expect_err("重複指定は fail-closed に拒否するはず");
        assert!(err.contains("複数回指定できない"));
    }

    #[test]
    fn parse_args_from_gpu_watch_alone_is_error() {
        let err = parse_args_from(args(&["--gpu-watch=Safari"]))
            .expect_err("--max-load-avg 未指定での単独指定はエラーのはず");
        assert!(err.contains("--max-load-avg 指定時のみ"));
    }

    #[test]
    fn parse_args_from_gpu_watch_with_max_load_avg_collects_multiple() {
        let parsed = parse_args_from(args(&[
            "--max-load-avg=4.0",
            "--gpu-watch=Safari",
            "--gpu-watch=Chrome",
        ]))
        .expect("--max-load-avg と併用時は成功するはず");
        assert_eq!(
            parsed.gpu_watch,
            vec!["Safari".to_string(), "Chrome".to_string()]
        );
    }

    #[test]
    fn parse_args_from_guard_max_attempts_rejects_zero() {
        let err = parse_args_from(args(&["--guard-max-attempts=0"]))
            .expect_err("0 は fail-closed に拒否するはず");
        assert!(err.contains("1 以上"));
    }

    #[test]
    fn parse_args_from_guard_wait_secs_rejects_non_positive() {
        let err = parse_args_from(args(&["--guard-wait-secs=0"]))
            .expect_err("0 は fail-closed に拒否するはず");
        assert!(err.contains("有限かつ正"));
    }

    /// Review 指摘（#1265・P2）の再現ケース: `is_finite() && > 0.0` は
    /// 通過するが `Duration` の表現範囲（最大約 1.8e19 秒）を大きく
    /// 超える値（`1e100`）は、以前は検証をすり抜けて後段
    /// `run_env_guard` 内 `Duration::from_secs_f64` の panic 経路に
    /// 到達しえた。CLI 引数検証の時点で明示的に拒否することを確認する。
    #[test]
    fn parse_args_from_guard_wait_secs_rejects_duration_overflow() {
        let err = parse_args_from(args(&["--guard-wait-secs=1e100"]))
            .expect_err("Duration の表現範囲外は fail-closed に拒否するはず");
        assert!(err.contains("Duration の表現範囲"));
    }

    #[test]
    fn parse_args_from_guard_only_flag_sets_true() {
        let parsed = parse_args_from(args(&["--guard-only"])).expect("成功するはず");
        assert!(parsed.guard_only);
    }

    #[test]
    fn parse_args_from_env_info_out_rejects_empty_value() {
        let err = parse_args_from(args(&["--env-info-out="]))
            .expect_err("空文字列は fail-closed に拒否するはず");
        assert!(err.contains("空文字列"));
    }

    #[test]
    fn parse_args_from_env_info_out_sets_path() {
        let parsed =
            parse_args_from(args(&["--env-info-out=/tmp/env_info.txt"])).expect("成功するはず");
        assert_eq!(parsed.env_info_out, Some("/tmp/env_info.txt".to_string()));
    }

    #[test]
    fn parse_args_from_guard_options_combine_with_existing_flags() {
        let parsed = parse_args_from(args(&[
            "--phase1-only",
            "--max-load-avg=4.0",
            "--guard-max-attempts=2",
            "--guard-wait-secs=1.0",
            "--guard-only",
        ]))
        .expect("併用は成功するはず");
        assert!(parsed.phase1_only);
        assert!(parsed.guard_only);
        assert_eq!(parsed.max_load_avg, Some(4.0));
        assert_eq!(parsed.guard_max_attempts, Some(2));
        assert_eq!(parsed.guard_wait_secs, Some(1.0));
    }

    // イシュー #1261: `--min-warmup-secs=<N>` の解析テスト群。

    #[test]
    fn parse_args_from_min_warmup_secs_accepts_boundary_and_typical_values() {
        for (raw, expected) in [
            ("--min-warmup-secs=3", 3u64),
            ("--min-warmup-secs=9", 9),
            ("--min-warmup-secs=600", 600),
        ] {
            let parsed = parse_args_from(args(&[raw]))
                .unwrap_or_else(|e| panic!("{raw} は成功するはず: {e}"));
            assert_eq!(
                parsed,
                CliArgs {
                    phase1_only: false,
                    gpu_timestamps: false,
                    min_warmup_secs: Some(expected),
                    ..Default::default()
                }
            );
        }
    }

    #[test]
    fn parse_args_from_min_warmup_secs_rejects_below_default() {
        // 既定値（3 秒）を下回る「減らす方向」の指定は拒否する
        // （原因候補 (b) 切り分けは増やす方向の試行に限定する計画上の制約）。
        for raw in ["--min-warmup-secs=2", "--min-warmup-secs=0"] {
            let err = parse_args_from(args(&[raw])).expect_err(&format!("{raw} は拒否されるはず"));
            assert!(err.contains("3〜600"), "raw={raw} err={err}");
        }
    }

    #[test]
    fn parse_args_from_min_warmup_secs_rejects_above_upper_bound() {
        let err = parse_args_from(args(&["--min-warmup-secs=601"]))
            .expect_err("上限 600 秒超は拒否されるはず");
        assert!(err.contains("3〜600"));
    }

    #[test]
    fn parse_args_from_min_warmup_secs_rejects_non_numeric_or_empty() {
        for raw in [
            "--min-warmup-secs=abc",
            "--min-warmup-secs=",
            "--min-warmup-secs=9.5",
            "--min-warmup-secs=-1",
        ] {
            let err = parse_args_from(args(&[raw])).expect_err(&format!("{raw} は拒否されるはず"));
            assert!(
                err.contains("非数値") || err.contains("値が空"),
                "raw={raw} err={err}"
            );
        }
    }

    #[test]
    fn parse_args_from_min_warmup_secs_rejects_duplicate() {
        let err = parse_args_from(args(&["--min-warmup-secs=9", "--min-warmup-secs=9"]))
            .expect_err("重複指定は fail-closed に拒否するはず");
        assert!(err.contains("複数回指定できない"));
    }

    #[test]
    fn parse_args_from_min_warmup_secs_space_separated_value_is_unknown_arg() {
        // `--min-warmup-secs 9`（値を別トークンで渡す形）は `=` 必須の
        // 本関数が認識しない形式のため、`9` 単独が未知の引数として
        // fail-closed に拒否される（意図した挙動。plan §3.2 参照）。
        let err = parse_args_from(args(&["--min-warmup-secs", "9"]))
            .expect_err("space 区切りの値指定は未知引数として拒否されるはず");
        assert!(err.contains("未知の引数"), "err={err}");
    }

    #[test]
    fn parse_args_from_min_warmup_secs_combines_with_other_flags_regardless_of_order() {
        for combo in [
            vec!["--phase1-only", "--gpu-timestamps", "--min-warmup-secs=9"],
            vec!["--min-warmup-secs=9", "--phase1-only", "--gpu-timestamps"],
        ] {
            let parsed = parse_args_from(args(&combo)).expect("順序不問で併用できるはず");
            assert_eq!(
                parsed,
                CliArgs {
                    phase1_only: true,
                    gpu_timestamps: true,
                    min_warmup_secs: Some(9),
                    ..Default::default()
                }
            );
        }
    }

    #[test]
    fn min_warmup_override_duration_defaults_when_none() {
        assert_eq!(
            min_warmup_override_duration(None),
            std::time::Duration::from_secs(super::DEFAULT_MIN_WARMUP_SECS)
        );
    }

    #[test]
    fn min_warmup_override_duration_uses_override_when_some() {
        assert_eq!(
            min_warmup_override_duration(Some(9)),
            std::time::Duration::from_secs(9)
        );
    }

    #[test]
    fn parse_min_warmup_secs_boundary_values_ok() {
        assert_eq!(
            parse_min_warmup_secs("--min-warmup-secs=3"),
            Ok(3),
            "下限 3 は受理されるはず"
        );
        assert_eq!(
            parse_min_warmup_secs("--min-warmup-secs=600"),
            Ok(600),
            "上限 600 は受理されるはず"
        );
    }

    /// [`GpuHostSample`] の共通ビルダ（テスト用）。デフォルトは
    /// `closure_wall=10・commit_wait=6・kernel_gpu=Some(4)・upload/alloc/
    /// encode/readback=1・batches_len=1・resolved_cfg="cfg"`（全不変条件を
    /// 満たす基準値）。`closure_wall_secs` はここでは `measured_wall_secs`
    /// （`protocol::run` 側の真の壁時計。[`aggregate_gpu_host_round`] が
    /// 集計に使う値）と同一値を渡す呼び出しが大半——両者が乖離する場合
    /// （drop 時のスパイクを模す場合）は個別テスト
    /// （`aggregate_gpu_host_round_uses_measured_wall_not_closure_wall`）
    /// で明示的に区別する。`resolved_cfg` を個別に変える場合は
    /// [`sample_with_cfg`] を使う。
    fn sample(
        closure_wall: f64,
        commit_wait: f64,
        kernel_gpu: Option<f64>,
        batches_len: usize,
    ) -> GpuHostSample {
        sample_with_cfg(closure_wall, commit_wait, kernel_gpu, batches_len, "cfg")
    }

    /// [`sample`] の `resolved_cfg` 指定版（イシュー #1261。フォールバック
    /// による構成乖離を模すテスト用）。
    fn sample_with_cfg(
        closure_wall: f64,
        commit_wait: f64,
        kernel_gpu: Option<f64>,
        batches_len: usize,
        resolved_cfg: &str,
    ) -> GpuHostSample {
        GpuHostSample {
            closure_wall_secs: closure_wall,
            upload_secs: 1.0,
            alloc_secs: 1.0,
            encode_secs: 1.0,
            commit_wait_secs: commit_wait,
            kernel_gpu_secs: kernel_gpu,
            readback_secs: 1.0,
            batches_len,
            resolved_cfg: resolved_cfg.to_string(),
        }
    }

    /// `tail` の `closure_wall_secs` をそのまま `measured_wall_secs`
    /// として使う（両者が一致するケース用のテストヘルパ）。
    fn walls_from_closure(tail: &[GpuHostSample]) -> Vec<f64> {
        tail.iter().map(|s| s.closure_wall_secs).collect()
    }

    #[test]
    fn measured_tail_returns_none_when_shorter_than_iters() {
        assert_eq!(measured_tail(&[1, 2, 3], 4), None);
    }

    #[test]
    fn measured_tail_returns_last_iters_elements() {
        assert_eq!(measured_tail(&[1, 2, 3, 4, 5], 3), Some(&[3, 4, 5][..]));
    }

    #[test]
    fn measured_tail_exact_length_returns_whole_slice() {
        assert_eq!(measured_tail(&[1, 2], 2), Some(&[1, 2][..]));
    }

    #[test]
    fn aggregate_gpu_host_round_valid_tail_computes_medians() {
        let tail = [
            sample(10.0, 6.0, Some(4.0), 1),
            sample(12.0, 7.0, Some(5.0), 1),
            sample(11.0, 6.5, Some(4.5), 1),
        ];
        let stats = aggregate_gpu_host_round(2, 3, &tail, &walls_from_closure(&tail));
        assert!(stats.valid);
        assert_eq!(stats.round, 2);
        assert_eq!(stats.iters, 3);
        assert_eq!(stats.kernel_gpu_median_secs, Some(4.5));
        assert_eq!(stats.wall_median_secs, 11.0);
        // wall_minus_gpu はサンプルごとの差（6.0, 7.0, 6.5）の中央値。
        assert_eq!(stats.wall_minus_gpu_median_secs, Some(6.5));
        assert_eq!(stats.commit_wait_median_secs, 6.5);
        assert_eq!(stats.commit_wait_minus_gpu_median_secs, Some(2.0));
        assert_eq!(stats.resolved_cfg, "cfg");
    }

    /// バッファ（`a_buf`/`b_buf`/`c_buf`）解放時のスパイクは
    /// `GpuHostSample::closure_wall_secs`（クロージャ内側のみの計測）には
    /// 乗らないが、安定性判定が使う `Measurement::samples_secs`
    /// （`measured_wall_secs`）には乗る——`aggregate_gpu_host_round` が
    /// 後者を単一真実源とすることを、両者が乖離するサンプルで検証する
    /// （codex-review 指摘の再発防止。イシュー #1261）。
    ///
    /// `measured_wall_secs` の中央値（16.0）が `closure_wall_secs` の
    /// 中央値（10.0）と異なる組み合わせを選ぶ（PR #1455 codex-review
    /// 指摘。旧版は `[10.0, 10.0, 16.0]` で両者の中央値が偶然一致し、
    /// `closure_wall_secs` への取り違えを検知できなかった）。
    #[test]
    fn aggregate_gpu_host_round_uses_measured_wall_not_closure_wall() {
        // `closure_wall_secs` は 3 ラウンドとも 10.0 で不変（drop 時間を
        // 含まないため一定に見える）だが、`measured_wall_secs`
        // （`protocol::run` 側の真の壁時計）は解放時のスパイクで
        // 10.0/16.0/16.0 と変動する——中央値・差分はこちらを反映すべき。
        let tail = [
            sample(10.0, 6.0, Some(4.0), 1),
            sample(10.0, 6.0, Some(4.0), 1),
            sample(10.0, 6.0, Some(4.0), 1),
        ];
        let measured_wall_secs = [10.0, 16.0, 16.0];
        let stats = aggregate_gpu_host_round(0, 3, &tail, &measured_wall_secs);
        assert!(stats.valid);
        // `closure_wall_secs` の中央値（10.0）ではなく `measured_wall_secs`
        // の中央値（16.0）を反映すべき——実装が誤って
        // `closure_wall_secs` を参照していれば 10.0 のまま不変となり
        // このアサーションで検知できる。
        // wall_minus_gpu はサンプルごとの差（6.0, 12.0, 12.0）の中央値
        // 12.0 になり、`closure_wall_secs` ベースの差（6.0,6.0,6.0
        // → 6.0）とは値そのものが異なることでも取り違えを検知できる。
        assert_eq!(stats.wall_median_secs, 16.0);
        assert_eq!(stats.wall_minus_gpu_median_secs, Some(12.0));
    }

    #[test]
    #[should_panic(expected = "長さが一致するはず")]
    fn aggregate_gpu_host_round_panics_on_length_mismatch() {
        let tail = [sample(10.0, 6.0, Some(4.0), 1)];
        let measured_wall_secs: [f64; 2] = [10.0, 11.0];
        let _ = aggregate_gpu_host_round(0, 1, &tail, &measured_wall_secs);
    }

    /// `resolved_cfg` は要求構成（呼び出しループの外で 1 回だけ解決した
    /// `tile::select_for_device` の値）ではなく、`GpuHostSample::
    /// resolved_cfg`（`encode_tiled_prepared` の戻り値。実際に実行した
    /// カーネル構成）から求める——全サンプルが同一の実行構成なら
    /// そのまま採用する（codex-review 指摘。イシュー #1261）。
    #[test]
    fn aggregate_gpu_host_round_resolved_cfg_from_actual_execution() {
        let tail = [
            sample_with_cfg(10.0, 6.0, Some(4.0), 1, "TileConfig { bm: 64, .. }"),
            sample_with_cfg(10.0, 6.0, Some(4.0), 1, "TileConfig { bm: 64, .. }"),
        ];
        let stats = aggregate_gpu_host_round(0, 2, &tail, &walls_from_closure(&tail));
        assert_eq!(stats.resolved_cfg, "TileConfig { bm: 64, .. }");
    }

    /// `pipeline_for_tile` のフォールバック挙動がラウンド内の呼び出しで
    /// 一致しない稀なケースを想定し、黙って先頭値へ丸めず `MIXED(...)`
    /// として可視化することを検証する（fail-closed。イシュー #1261）。
    #[test]
    fn aggregate_gpu_host_round_resolved_cfg_mixed_when_samples_disagree() {
        let tail = [
            sample_with_cfg(10.0, 6.0, Some(4.0), 1, "cfg_a"),
            sample_with_cfg(10.0, 6.0, Some(4.0), 1, "cfg_b"),
        ];
        let stats = aggregate_gpu_host_round(0, 2, &tail, &walls_from_closure(&tail));
        assert_eq!(stats.resolved_cfg, "MIXED(cfg_a,cfg_b)");
    }

    #[test]
    fn aggregate_gpu_host_round_invalid_when_kernel_gpu_missing() {
        let tail = [sample(10.0, 6.0, None, 1)];
        let stats = aggregate_gpu_host_round(0, 1, &tail, &walls_from_closure(&tail));
        assert!(!stats.valid);
        assert_eq!(stats.kernel_gpu_median_secs, None);
        assert_eq!(stats.wall_minus_gpu_median_secs, None);
        // host 側フェーズ内訳自体は GPU タイムスタンプ非依存のため計算される。
        assert_eq!(stats.wall_median_secs, 10.0);
    }

    #[test]
    fn aggregate_gpu_host_round_invalid_when_batches_len_not_one() {
        let tail = [sample(10.0, 6.0, Some(4.0), 2)];
        let stats = aggregate_gpu_host_round(0, 1, &tail, &walls_from_closure(&tail));
        assert!(!stats.valid);
    }

    #[test]
    fn aggregate_gpu_host_round_invalid_when_invariant_order_violated() {
        // kernel_gpu > commit_wait は `0 ≤ kernel_gpu ≤ commit_wait ≤ wall`
        // 不変条件違反。
        let tail = [sample(10.0, 6.0, Some(7.0), 1)];
        let stats = aggregate_gpu_host_round(0, 1, &tail, &walls_from_closure(&tail));
        assert!(!stats.valid);
    }

    #[test]
    fn format_gpu_host_round_line_starts_with_grep_key_and_includes_size() {
        let tail = [sample(10.0, 6.0, Some(4.0), 1)];
        let stats = aggregate_gpu_host_round(0, 1, &tail, &walls_from_closure(&tail));
        let line = format_gpu_host_round_line(&stats, 512);
        assert!(line.starts_with("phase1_gpu_host_round "));
        assert!(line.contains("size=512"));
        assert!(line.contains("valid=true"));
    }

    #[test]
    fn format_gpu_host_round_line_reports_na_for_invalid_round() {
        let tail = [sample(10.0, 6.0, None, 1)];
        let stats = aggregate_gpu_host_round(0, 1, &tail, &walls_from_closure(&tail));
        let line = format_gpu_host_round_line(&stats, 512);
        assert!(line.contains("kernel_gpu_median_secs=NA"));
        assert!(line.contains("valid=false"));
    }

    fn round_stats(kernel_gpu: Option<f64>, wall: f64) -> GpuHostRoundStats {
        GpuHostRoundStats {
            round: 0,
            iters: 20,
            valid: kernel_gpu.is_some(),
            kernel_gpu_median_secs: kernel_gpu,
            wall_median_secs: wall,
            closure_wall_median_secs: wall,
            wall_minus_gpu_median_secs: kernel_gpu.map(|k| wall - k),
            commit_wait_median_secs: wall / 2.0,
            commit_wait_minus_gpu_median_secs: kernel_gpu.map(|k| wall / 2.0 - k),
            upload_median_secs: 1.0,
            alloc_median_secs: 1.0,
            encode_median_secs: 1.0,
            readback_median_secs: 1.0,
            resolved_cfg: "cfg".to_string(),
        }
    }

    #[test]
    fn aggregate_gpu_host_size_all_valid_computes_spreads() {
        let rounds = [
            round_stats(Some(4.0), 10.0),
            round_stats(Some(5.0), 12.0),
            round_stats(Some(4.5), 11.0),
            round_stats(Some(4.2), 10.5),
        ];
        let stats = aggregate_gpu_host_size(512, &rounds);
        assert_eq!(stats.size, 512);
        assert_eq!(stats.rounds, 4);
        assert_eq!(stats.valid_rounds, 4);
        assert!(stats.spread_kernel_gpu.is_some());
        assert!(stats.spread_wall_minus_gpu.is_some());
        assert!(stats.spread_wall >= 0.0);
    }

    #[test]
    fn aggregate_gpu_host_size_one_invalid_round_makes_kernel_gpu_spread_na() {
        // 1 ラウンドでも無効（`valid=false`）なら kernel_gpu 系の spread は
        // 部分集合で計算せず `None`（fail-closed）。壁時計側の spread は
        // GPU タイムスタンプに依存しないため常に計算される。
        let rounds = [
            round_stats(Some(4.0), 10.0),
            round_stats(None, 12.0),
            round_stats(Some(4.5), 11.0),
            round_stats(Some(4.2), 10.5),
        ];
        let stats = aggregate_gpu_host_size(512, &rounds);
        assert_eq!(stats.spread_kernel_gpu, None);
        assert_eq!(stats.max_round_idx_kernel_gpu, None);
        assert_eq!(stats.spread_wall_minus_gpu, None);
        assert!(stats.spread_wall >= 0.0);
        assert_eq!(stats.valid_rounds, 3);
    }

    #[test]
    fn format_gpu_host_size_line_starts_with_grep_key_and_includes_na() {
        let rounds = [
            round_stats(Some(4.0), 10.0),
            round_stats(None, 12.0),
            round_stats(Some(4.5), 11.0),
            round_stats(Some(4.2), 10.5),
        ];
        let stats = aggregate_gpu_host_size(512, &rounds);
        let line = format_gpu_host_size_line(&stats);
        assert!(line.starts_with("phase1_gpu_host_stats "));
        assert!(line.contains("size=512"));
        assert!(line.contains("spread_kernel_gpu=NA"));
        assert!(line.contains("kernel_gpu_round_medians_secs="));
        assert!(line.contains(",NA,"));
    }

    #[test]
    fn round_extrema_empty_slice_is_none() {
        assert_eq!(round_extrema(&[]), None);
    }

    #[test]
    fn round_extrema_single_element_min_equals_max_at_index_zero() {
        let extrema = round_extrema(&[1.5]).expect("単一要素は Some を返すはず");
        assert_eq!(extrema.min_secs, 1.5);
        assert_eq!(extrema.min_round_idx, 0);
        assert_eq!(extrema.max_secs, 1.5);
        assert_eq!(extrema.max_round_idx, 0);
    }

    #[test]
    fn round_extrema_single_spike_finds_correct_index() {
        // 単発スパイク型（#1249 本文が指摘する実測パターン）: index 3 だけ
        // 突出して遅い（秒基準で大きい値）。
        let samples = [1.0, 1.1, 0.9, 5.0, 1.05, 0.95];
        let extrema = round_extrema(&samples).expect("非空スライスは Some を返すはず");
        assert_eq!(extrema.max_secs, 5.0);
        assert_eq!(extrema.max_round_idx, 3);
        assert_eq!(extrema.min_secs, 0.9);
        assert_eq!(extrema.min_round_idx, 2);
    }

    #[test]
    fn round_extrema_tie_picks_first_occurrence() {
        let samples = [3.0, 1.0, 3.0, 1.0];
        let extrema = round_extrema(&samples).expect("非空スライスは Some を返すはず");
        // 最大値 3.0 は index 0・2 に出現するが最初の出現（0）を採る。
        assert_eq!(extrema.max_secs, 3.0);
        assert_eq!(extrema.max_round_idx, 0);
        // 最小値 1.0 は index 1・3 に出現するが最初の出現（1）を採る。
        assert_eq!(extrema.min_secs, 1.0);
        assert_eq!(extrema.min_round_idx, 1);
    }

    /// イシュー #1484: `AUXILIARY_TRIM_PER_SIDE` は
    /// `format_aux_spread_keys` のキー名 `trimmed_spread_k1` の `_k1`
    /// 接尾辞が前提とする値（k=1）。定数が変われば接尾辞もずれるため、
    /// このテストで固定する（値そのものはイシュー #1483 で確定済み・
    /// 本イシューでは変更しない）。
    #[test]
    fn auxiliary_trim_per_side_is_one() {
        assert_eq!(AUXILIARY_TRIM_PER_SIDE, 1);
    }

    /// `docs/perf/logs/metal-gemm-transpose-route-ab-1242/1255-phase1_run1.log`
    /// の実ログ行（size=256）を固定し、`format_phase1_round_stats_line`
    /// の出力が **既存 prefix と byte 単位で一致する**ことを検証する
    /// （イシュー #1484 の受入条件: 既存キー・順序は不変）。`spread` は
    /// ログ値をそのまま使う（`.6e` 丸め済み中央値から `relative_spread`
    /// を再計算すると 5 桁目がずれうるため再計算しない）。
    #[test]
    fn format_phase1_round_stats_line_starts_with_existing_log_prefix() {
        let round_medians_secs = vec![
            3.694170e-4,
            2.224590e-4,
            2.312080e-4,
            2.332920e-4,
            2.381670e-4,
            2.632910e-4,
            2.470000e-4,
            3.650830e-4,
            2.260000e-4,
            2.398340e-4,
        ];
        let existing_prefix = "phase1_round_stats size=256 rounds=10 spread=6.1275e-1 \
             gate=5.0000e-2 within_gate=false median_secs=2.398340e-4 \
             min_secs=2.224590e-4 min_round_idx=1 max_secs=3.694170e-4 max_round_idx=0 \
             round_medians_secs=3.694170e-4,2.224590e-4,2.312080e-4,2.332920e-4,2.381670e-4,\
             2.632910e-4,2.470000e-4,3.650830e-4,2.260000e-4,2.398340e-4";
        let result = StabilityResult {
            round_medians_secs: round_medians_secs.clone(),
            spread: 6.1275e-1,
            aux: AuxiliarySpread {
                trimmed: Some(1.2345e-1),
                iqr_over_median: 2.3456e-1,
                mad2_over_median: 3.4567e-1,
            },
        };
        let extrema = round_extrema(&round_medians_secs).expect("非空スライスは Some を返すはず");
        let line = format_phase1_round_stats_line(256, &result, 5.0000e-2, false, &extrema);
        assert!(
            line.starts_with(existing_prefix),
            "既存キーの prefix が byte 単位で不変であること: {line}"
        );
        // 補助 3 キーは既存 prefix の直後にスペース区切りで続く。
        assert_eq!(
            line,
            format!(
                "{existing_prefix} trimmed_spread_k1=1.2345e-1 iqr_spread=2.3456e-1 \
                 mad_spread=3.4567e-1"
            )
        );
    }

    /// `aux.trimmed == None`（ラウンド数がトリム後 2 要素未満）のとき、
    /// `trimmed_spread_k1=NA`（既存 sentinel。新規 sentinel を導入しない
    /// 契約）になることを検証する（イシュー #1484）。
    #[test]
    fn format_aux_spread_keys_trimmed_none_renders_na_sentinel() {
        let aux = AuxiliarySpread {
            trimmed: None,
            iqr_over_median: 1.0e-1,
            mad2_over_median: 2.0e-1,
        };
        let keys = format_aux_spread_keys(&aux);
        assert_eq!(
            keys,
            "trimmed_spread_k1=NA iqr_spread=1.0000e-1 mad_spread=2.0000e-1"
        );
    }

    /// 末尾キーの順序（`trimmed_spread_k1` → `iqr_spread` → `mad_spread`）
    /// を固定する（`docs/perf/metal-gemm-transpose-tiled.md` のキー一覧と
    /// 同じ順序。イシュー #1484）。
    #[test]
    fn format_aux_spread_keys_key_order_is_trimmed_then_iqr_then_mad() {
        let aux = AuxiliarySpread {
            trimmed: Some(9.0e-1),
            iqr_over_median: 8.0e-1,
            mad2_over_median: 7.0e-1,
        };
        let keys = format_aux_spread_keys(&aux);
        let trimmed_pos = keys.find("trimmed_spread_k1=").expect("キーが存在するはず");
        let iqr_pos = keys.find("iqr_spread=").expect("キーが存在するはず");
        let mad_pos = keys.find("mad_spread=").expect("キーが存在するはず");
        assert!(trimmed_pos < iqr_pos);
        assert!(iqr_pos < mad_pos);
    }

    /// [`RoundExtrema`] が公開されている（別 example への複製を避けた設計
    /// 判断）ことの回帰確認。イシュー #1484。
    #[test]
    fn round_extrema_type_is_reusable_from_other_examples() {
        let extrema: RoundExtrema = round_extrema(&[1.0]).expect("非空スライスは Some を返すはず");
        assert_eq!(extrema.min_secs, 1.0);
    }
}
