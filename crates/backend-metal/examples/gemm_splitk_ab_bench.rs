//! split-K 2 パス GEMM（イシュー #1474）と現行 classic 経路の性能 A/B
//! （イシュー #1475）。
//!
//! `docs/perf/metal-gemm-splitk-shapes.md` §4／§6・`docs/backend-metal-
//! splitk-decision.md` §3 は、K 支配的非正方 9 形状
//! `(32,32,*)`／`(64,64,*)`／`(128,128,*)`（K ∈ {2048, 4096, 8192}）の
//! 劣化率が classic 経路の並列度不足によるという因果を**仮説**として
//! 留保していた（実行時間差のみの観測で split-K 経路自体の改善効果は
//! 未計測）。本 example はその因果を実測で更新する（`docs/perf/
//! metal-gemm-splitk-ab.md` §6）。
//!
//! ## 計測境界・腕の定義（事前登録。計測後に変更しない）
//!
//! prepared 境界（A・B・C バッファの確保・アップロードは計測外。計測対象は
//! encode ＋ コマンドバッファ完了待ち。readback 対象外。`gemm_splitk_
//! shapes_bench.rs` と同型）で、以下の腕を比較する（レイアウトは NN 固定。
//! NT/TN/TT の性能比較は本 example の対象外。#1474 で 4 パターンとも同一
//! 集計値と確認済み）:
//!
//! - **A（classic）**: [`MetalGemm::dispatch_strided_tiled_prepared`]
//!   （`cfg = tile::select_for_device(m, n, k, ...)`。split-K のフォール
//!   バック先と同一経路）
//! - **A′（classic・split-K タイル）**: 同じ classic 経路だが
//!   `cfg = tile::split_k_tile(m, n)`。B との差分から「タイル構成効果」
//!   （A′/A）を「K 分割効果」（B/A′）と分離する（判定には使わない。記録
//!   のみ。`docs/perf/metal-gemm-splitk-ab.md` §6）。
//! - **B（split-K）**: [`MetalGemm::dispatch_split_k_strided_prepared_with_plan`]
//!   に `tile::should_split_k(m, n, k)` の計画を明示的に渡す（`internal-
//!   diagnostics` feature 限定。`SPLIT_K_NUMERIC_CONTRACT_APPROVED` ゲート
//!   対象外）。戻り値が `SplitKRoute::Split` でなければ計測を中止する
//!   （フォールバックをデータ点として扱わない）。
//! - **B′（対照・classic）**: 対照 `(256,256,K)` は `should_split_k` が
//!   `max_groups` 条件で `None` を返すため、`None` を assert したうえで
//!   classic を dispatch する（選択関数の呼び出し費用込み。結線相当経路）。
//! - **C（対照・強制 split-K。参考のみ）**: `max_groups=u64::MAX` で
//!   強制した split-K。`max_groups=40` 境界の妥当性を記録するだけで
//!   ADOPT／REJECT には影響させない。
//! - **フロア（参考のみ）**: 各 `(M,N)` を `K=64` で classic dispatch した
//!   時間。dispatch 固定費のフロアの記録。
//!
//! **比の方向**: `speedup = median_secs(A) / median_secs(B)`（1 より大きい
//! ほど split-K が速い）。
//!
//! ## 判定規則（事前登録。イシュー #1475 コメント参照）
//!
//! - **ADOPT**: 対象 9 形状すべてで (i) 主指標（run 内比の 5 run 中央値）
//!   ≥ 1.5 かつ (ii) 5/5 run すべてで run 内比 > 1.0、かつ対照 3 形状
//!   すべてで主指標 ≥ 0.95。
//! - **REJECT**: 専有ゲート成立下で上記いずれかが不成立。
//! - **undetermined**: 専有ゲート（`--max-load-avg`）が規定回数で成立しない
//!   場合（1 回だけ記録して終了）。
//!
//! spread（`spread_a`／`spread_b`。レンジベース）は判定に使わない
//! （#1308 が同一形状・同一境界で spread 0.78〜1.44 を実測しており、
//! `STABILITY_SPREAD_GATE` を適用すると本形状群は構造的に undetermined
//! にしかならないため）。数値契約（各腕の run-to-run bit 同一）は計測
//! 妥当性の前提条件（フェーズ 0）とし、腕間の差は情報として記録するのみで
//! verdict の入力にしない（`tests/gemm_splitk_parity.rs` の既知 fail が
//! 再現することが期待値であり、fail の有無で ADOPT／REJECT を左右しない）。
//!
//! **ADOPT は性能上の判定に限る**: 本番結線（`SPLIT_K_NUMERIC_CONTRACT_
//! APPROVED` の切替）は REQ-2 判定方式の Metal f32 split-K への適用拡張
//! というユーザー承認が別途必要（#1476 のスコープ）。本 example・
//! `crates/backend-metal/src/` はいずれも変更しない（性能 A/B の実測・
//! 記録に限る）。
//!
//! ## 実行方法
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_splitk_ab_bench \
//!   --release --features internal-diagnostics -- --max-load-avg=4.0
//! ```
//!
//! `--max-load-avg=<f64>` 未指定時は環境ガードを行わず即座に計測へ進む
//! （記録のみ。本番実測では必ず指定する。閾値自体に既定値はない）。
//! `--self-check-only` はフェーズ 0（自己検証）のみ実行し終了する。
//! `--iters=<N>` は warmup・計測回数を引き上げる（未指定なら
//! `MeasurementConfig::default` = 20/20）。

/// 対象（K 支配的非正方。`M == N`）の `M`/`N` 候補。
const TARGET_MN: [usize; 3] = [32, 64, 128];
/// 対照（`should_split_k` が `max_groups` 条件で `None` を返す正方）の `M`/`N`。
const CONTROL_MN: usize = 256;
/// 対象・対照共通の `K` 候補。
const K_LIST: [usize; 3] = [2048, 4096, 8192];
/// フロア計測（dispatch 固定費の参考値）が使う `K`。
/// macOS（`macos_impl`）とテストからのみ参照されるため、非 macOS の通常
/// ビルド（clippy 含む）では未使用になる。`gemm_transpose_route_ab_bench.rs`
/// と同型の `cfg_attr` で dead_code を抑止する。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
const FLOOR_K: usize = 64;
/// フロア計測対象の `(M,N)`（対象 3 種 ＋ 対照）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
const FLOOR_MN: [usize; 4] = [32, 64, 128, 256];

/// `splitk_ab` 行の種別。
// KIND_TARGET/KIND_TARGET_TILE/KIND_CONTROL/KIND_CONTROL_FORCED は
// `macos_impl`（`cfg(target_os = "macos")`）内でのみ参照され、
// `#[cfg(test)] mod tests` からは（`format_ab_line` を文字列リテラル
// 引数で直接呼ぶため）参照されない。よって非 macOS ビルドでは
// `cfg(test)` の真偽に関わらず未使用になるため、`test` を条件に含めず
// `not(target_os = "macos")` のみで dead_code を抑止する
// （KIND_FLOOR は `format_floor_line` 経由でテストからも到達するため
// 対象外。イシュー #1499 codex-review 指摘対応）。
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const KIND_TARGET: &str = "target";
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const KIND_TARGET_TILE: &str = "target_tile";
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const KIND_CONTROL: &str = "control";
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
const KIND_CONTROL_FORCED: &str = "control_forced";
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
const KIND_FLOOR: &str = "floor";

/// A/B 双方が揃う行（`splitk_ab`）を機械可読 1 行へ整形する純関数。
/// `kind` は [`KIND_TARGET`] 等の固定文字列のみを想定（外部入力を
/// 埋め込まない。`.claude/rules/security.md` A03 対応）。
/// macOS（`macos_impl`）とテストからのみ呼ばれるため非 macOS の通常
/// ビルドでは未使用になる（`FLOOR_K` と同様の理由）。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[allow(clippy::too_many_arguments)]
fn format_ab_line(
    kind: &str,
    m: usize,
    n: usize,
    k: usize,
    partitions: Option<u32>,
    median_a_secs: f64,
    median_b_secs: f64,
    spread_a: f64,
    spread_b: f64,
) -> String {
    let speedup = median_a_secs / median_b_secs;
    let partitions_str = match partitions {
        Some(p) => p.to_string(),
        None => "NA".to_string(),
    };
    format!(
        "splitk_ab kind={kind} m={m} n={n} k={k} partitions={partitions_str} \
         median_a_secs={median_a_secs:.6e} median_b_secs={median_b_secs:.6e} \
         speedup={speedup:.4} spread_a={spread_a:.4e} spread_b={spread_b:.4e}"
    )
}

/// フロア計測（単一腕。`median_b_secs`／`speedup`／`spread_b` は `NA`）を
/// 整形する純関数。macOS（`macos_impl`）とテストからのみ呼ばれる。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn format_floor_line(m: usize, n: usize, k: usize, median_secs: f64, spread: f64) -> String {
    format!(
        "splitk_ab kind={KIND_FLOOR} m={m} n={n} k={k} partitions=NA \
         median_a_secs={median_secs:.6e} median_b_secs=NA speedup=NA spread_a={spread:.4e} \
         spread_b=NA"
    )
}

/// run-to-run bit 同一検証を `to_bits()` 経由で行う純関数
/// （`tests/gemm_swizzle_bit_match.rs::assert_bit_exact`・
/// `tests/gemm_splitk_bit_match.rs` と同じ理由: `&[f32]` の `==` は
/// IEEE 754 の `+0.0 == -0.0` を区別できず符号ビットの差異を見逃す。
/// `NaN` は `to_bits()` がビットパターンをそのまま返すため
/// `NaN != NaN` の数値比較特性に引きずられない。イシュー #1499
/// codex-review 指摘対応: フェーズ 0 の `classic_stable`／
/// `target_tile_stable`／`splitk_stable` 判定が素の `Vec<f32>` 比較
/// だったため本関数へ置き換える）。macOS（`macos_impl`）とテストから
/// のみ呼ばれる。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn bit_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// フェーズ 1（A/B）の対象 9 形状。
fn target_shapes() -> Vec<(usize, usize, usize)> {
    let mut out = Vec::with_capacity(TARGET_MN.len() * K_LIST.len());
    for &mn in &TARGET_MN {
        for &k in &K_LIST {
            out.push((mn, mn, k));
        }
    }
    out
}

/// フェーズ 1（A/B′・A/C）の対照 3 形状。
fn control_shapes() -> Vec<(usize, usize, usize)> {
    K_LIST
        .iter()
        .map(|&k| (CONTROL_MN, CONTROL_MN, k))
        .collect()
}

/// フロア計測対象（4 `(M,N)` × `FLOOR_K`）。macOS（`macos_impl`）と
/// テストからのみ呼ばれる。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn floor_shapes() -> Vec<(usize, usize, usize)> {
    FLOOR_MN.iter().map(|&mn| (mn, mn, FLOOR_K)).collect()
}

/// CLI 引数の解析結果（`gemm_transpose_route_ab_bench.rs::CliArgs` と同型の
/// 一括走査＋未知引数・重複指定の拒否）。macOS（`macos_impl`）とテスト
/// からのみ構築される。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct CliArgs {
    /// `--max-load-avg=<f64>`。指定時のみ環境ガードを **gated** で行う
    /// （未指定時は環境ガードを行わず即座に計測へ進む。既定値はユーザー
    /// 承認事項のため組み込まない）。
    max_load_avg: Option<f64>,
    /// `--guard-max-attempts=<usize>`（`--max-load-avg` 指定時のみ有効）。
    guard_max_attempts: Option<usize>,
    /// `--iters=<N>`（warmup・計測回数を引き上げる。未指定なら既定 20/20）。
    iters: Option<usize>,
    /// `--self-check-only`（フェーズ 0 のみ実行して終了する）。
    self_check_only: bool,
}

/// [`CliArgs`] を解析する純関数（`gemm_transpose_route_ab_bench.rs::
/// parse_args_from` と同型。未知引数・重複指定を `Err` で拒否する）。
/// macOS（`macos_impl`）とテストからのみ呼ばれる。
#[cfg_attr(not(any(target_os = "macos", test)), allow(dead_code))]
fn parse_args_from<I: IntoIterator<Item = String>>(args: I) -> Result<CliArgs, String> {
    let mut out = CliArgs::default();
    let mut max_load_avg_seen = false;
    let mut guard_max_attempts_seen = false;
    let mut iters_seen = false;
    let mut self_check_only_seen = false;

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
        } else if let Some(rest) = arg.strip_prefix("--guard-max-attempts=") {
            if guard_max_attempts_seen {
                return Err(format!(
                    "--guard-max-attempts は複数回指定できない（重複指定）: '{arg}'"
                ));
            }
            let value: usize = rest.parse().map_err(|_| {
                format!("--guard-max-attempts の値が正の整数として解釈できない: '{arg}'")
            })?;
            if value == 0 {
                return Err(format!(
                    "--guard-max-attempts は 1 以上である必要がある: '{arg}'"
                ));
            }
            out.guard_max_attempts = Some(value);
            guard_max_attempts_seen = true;
        } else if let Some(rest) = arg.strip_prefix("--iters=") {
            if iters_seen {
                return Err(format!("--iters は複数回指定できない（重複指定）: '{arg}'"));
            }
            let value: usize = rest
                .parse()
                .map_err(|_| format!("--iters の値が正の整数として解釈できない: '{arg}'"))?;
            out.iters = Some(value);
            iters_seen = true;
        } else if arg == "--self-check-only" {
            if self_check_only_seen {
                return Err("--self-check-only は複数回指定できない（重複指定）".to_string());
            }
            out.self_check_only = true;
            self_check_only_seen = true;
        } else {
            return Err(format!(
                "未知の引数: '{arg}'（既知の引数: --max-load-avg=<f64>／\
                 --guard-max-attempts=<usize>／--iters=<N>／--self-check-only）"
            ));
        }
    }

    if out.guard_max_attempts.is_some() && out.max_load_avg.is_none() {
        return Err(
            "--guard-max-attempts は --max-load-avg 指定時のみ有効（単独指定はエラー）".to_string(),
        );
    }

    Ok(out)
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use super::{
        CliArgs, KIND_CONTROL, KIND_CONTROL_FORCED, KIND_TARGET, KIND_TARGET_TILE, bit_equal,
        control_shapes, floor_shapes, format_ab_line, format_floor_line, parse_args_from,
        target_shapes,
    };
    use bench_harness::BenchError;
    use bench_harness::MeasurementConfig;
    use bench_harness::ab::{
        AbConfig, EnvGuardConfig, GuardRetryOutcome, RetryConfig, format_env_info_text, run_ab,
        run_guard_with_retry, run_stability,
    };
    use bench_harness::rng::Xorshift64Star;
    use fandhe_ai_backend_cpu::parity::compare;
    use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
    use fandhe_ai_backend_metal::tile::{self, SplitKParams, SplitKPlan};
    use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, SplitKRoute, TileConfig};
    use std::time::Duration;

    /// 決定的シード（`gemm_splitk_shapes_bench.rs::SEED` と同一値）。
    const SEED: u64 = 0xC0FFEE;

    /// ラウンド数・cooldown・時間ベースウォームアップ下限
    /// （`gemm_splitk_shapes_bench.rs` と同一値。#1475 計画 §3.3）。
    const ROUNDS: usize = 6;
    const COOLDOWN: Duration = Duration::from_secs(2);
    const MIN_WARMUP: Duration = Duration::from_secs(1);

    /// バックオフ再試行の待機定数（`gemm_transpose_route_ab_bench.rs::
    /// GUARD_*` と同一値。閾値〈`--max-load-avg`〉自体はユーザー承認事項の
    /// ため既定値を持たない）。
    const GUARD_INITIAL_WAIT: Duration = Duration::from_secs(30);
    const GUARD_GROWTH_FACTOR: f64 = 1.5;
    const GUARD_MAX_WAIT: Duration = Duration::from_secs(300);
    const GUARD_MAX_ATTEMPTS: usize = 10;

    /// `m×n×k` 用に確保・アップロード済みの prepared 入力一式（NN 固定）。
    struct PreparedShape {
        m: usize,
        n: usize,
        k: usize,
        a_buf: MetalBuffer,
        a_layout: MatrixLayout,
        b_buf: MetalBuffer,
        b_layout: MatrixLayout,
        c_buf: MetalBuffer,
    }

    fn prepare(
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        seed_offset: u64,
    ) -> PreparedShape {
        let mut rng = Xorshift64Star::new(SEED ^ seed_offset);
        let a: Vec<f32> = rng.fill_vec(m * k);
        let b: Vec<f32> = rng.fill_vec(k * n);

        let a_layout = classify_2d(&[m, k], &[k as isize, 1])
            .expect("NN A レイアウトの classify_2d に失敗した（対象形状は常に成立する前提）");
        let b_layout = classify_2d(&[k, n], &[n as isize, 1])
            .expect("NN B レイアウトの classify_2d に失敗した（対象形状は常に成立する前提）");

        let a_buf = MetalBuffer::new_with_data(ctx, &a)
            .expect("A バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");
        let b_buf = MetalBuffer::new_with_data(ctx, &b)
            .expect("B バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");
        let c_buf = MetalBuffer::new_zeroed(ctx, m * n)
            .expect("C バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");

        PreparedShape {
            m,
            n,
            k,
            a_buf,
            a_layout,
            b_buf,
            b_layout,
            c_buf,
        }
    }

    /// A（classic。`cfg` は呼び出し側が渡す構成。`select_for_device` の
    /// 選択構成〈A〉と `tile::split_k_tile`〈A′〉の双方に使う）。
    fn dispatch_classic(gemm: &MetalGemm, ctx: &MetalContext, p: &PreparedShape, cfg: TileConfig) {
        gemm.dispatch_strided_tiled_prepared(
            ctx, &p.a_buf, 0, p.a_layout, &p.b_buf, 0, p.b_layout, &p.c_buf, p.m, p.n, p.k, cfg,
        )
        .expect(
            "dispatch_strided_tiled_prepared（classic 腕）に失敗した（実機でのみ実行する前提）",
        );
    }

    /// B（split-K）。`route` が `SplitKRoute::Split` でなければフォール
    /// バックした（データ点として扱わない）ため即座に中止する。
    fn dispatch_split_k(gemm: &MetalGemm, ctx: &MetalContext, p: &PreparedShape, plan: SplitKPlan) {
        let route = gemm
            .dispatch_split_k_strided_prepared_with_plan(
                ctx, &p.a_buf, 0, p.a_layout, &p.b_buf, 0, p.b_layout, &p.c_buf, p.m, p.n, p.k,
                plan,
            )
            .expect(
                "dispatch_split_k_strided_prepared_with_plan（split-K 腕）に失敗した \
                 （実機でのみ実行する前提）",
            );
        if !matches!(route, SplitKRoute::Split { .. }) {
            eprintln!(
                "split-K 腕が classic 経路へフォールバックした（m={}, n={}, k={}, route={route:?}）。\
                 フォールバックをデータ点として扱わないため計測を中止する。",
                p.m, p.n, p.k
            );
            std::process::exit(2);
        }
    }

    fn resolve_measurement_config(args: &CliArgs) -> Result<MeasurementConfig, String> {
        match args.iters {
            Some(n) => MeasurementConfig::new(n, n).map_err(|e| e.to_string()),
            None => Ok(MeasurementConfig::default()),
        }
    }

    /// 環境ガード（`--max-load-avg` 指定時のみ **gated**。未指定時は
    /// 実行せず即座に計測へ進む。`gemm_transpose_route_ab_bench.rs::
    /// macos_impl::run_env_guard` と同型）。
    fn run_env_guard(args: &CliArgs) -> Result<Option<GuardRetryOutcome>, BenchError> {
        match args.max_load_avg {
            Some(max_load_avg) => {
                let config = EnvGuardConfig::new(max_load_avg)?;
                let max_attempts = args.guard_max_attempts.unwrap_or(GUARD_MAX_ATTEMPTS);
                let retry = RetryConfig::new(
                    GUARD_INITIAL_WAIT,
                    GUARD_GROWTH_FACTOR,
                    GUARD_MAX_WAIT,
                    max_attempts,
                )?;
                let outcome = run_guard_with_retry(&config, &retry)?;
                println!(
                    "{}",
                    format_env_info_text("gemm_splitk_ab_bench", &outcome, Some(&config))
                );
                Ok(Some(outcome))
            }
            None => Ok(None),
        }
    }

    /// フェーズ 0: 各腕（classic・split-K）の run-to-run bit 同一を自己
    /// 検証する（計測妥当性の前提条件。#1475 計画 §3.1）。腕間
    /// （classic vs split-K）の差は `fandhe_ai_backend_cpu::parity::compare`
    /// の集計値と f64 checksum を情報として出力するのみで、フェーズ 0 の
    /// 合否には使わない（`tests/gemm_splitk_parity.rs` の既知 fail が
    /// 再現することが期待値のため）。
    fn phase0_self_check(gemm: &MetalGemm, ctx: &MetalContext) {
        println!("== phase0 self-check ==");
        let mut all_ok = true;

        for &(m, n, k) in target_shapes().iter() {
            let plan = tile::should_split_k(m, n, k).unwrap_or_else(|| {
                panic!("should_split_k が None を返した（対象形状の前提が崩れている）: m={m}, n={n}, k={k}")
            });

            // classic（A）run-to-run bit 同一。
            let p1 = prepare(ctx, m, n, k, 1);
            let select_cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());
            dispatch_classic(gemm, ctx, &p1, select_cfg);
            let a_run1 = p1.c_buf.read_to_vec();
            dispatch_classic(gemm, ctx, &p1, select_cfg);
            let a_run2 = p1.c_buf.read_to_vec();
            let classic_stable = bit_equal(&a_run1, &a_run2);

            // A′（classic・split-K タイル）run-to-run bit 同一。
            let p1t = prepare(ctx, m, n, k, 1);
            let tile_cfg = tile::split_k_tile(m, n);
            dispatch_classic(gemm, ctx, &p1t, tile_cfg);
            let at_run1 = p1t.c_buf.read_to_vec();
            dispatch_classic(gemm, ctx, &p1t, tile_cfg);
            let at_run2 = p1t.c_buf.read_to_vec();
            let target_tile_stable = bit_equal(&at_run1, &at_run2);

            // split-K（B）run-to-run bit 同一。`p1`（classic）と同一
            // `seed_offset`（=1）で生成する: 以下の `a_vs_b_fail_count`
            // は同一入力に対する classic／split-K の出力差（丸め順序差
            // 起因と期待される既知 fail）を見る比較であり、異なる入力
            // 行列同士を比較すると解釈の前提が崩れる（イシュー #1499
            // codex-review・Cursor Bugbot 指摘）。
            let p2 = prepare(ctx, m, n, k, 1);
            dispatch_split_k(gemm, ctx, &p2, plan);
            let b_run1 = p2.c_buf.read_to_vec();
            dispatch_split_k(gemm, ctx, &p2, plan);
            let b_run2 = p2.c_buf.read_to_vec();
            let splitk_stable = bit_equal(&b_run1, &b_run2);

            let stable = classic_stable && target_tile_stable && splitk_stable;
            all_ok &= stable;

            let checksum_a: f64 = a_run1.iter().map(|&v| v as f64).sum();
            let checksum_at: f64 = at_run1.iter().map(|&v| v as f64).sum();
            let checksum_b: f64 = b_run1.iter().map(|&v| v as f64).sum();
            let cmp_ab = compare(&a_run1, &b_run1).ok();
            let cmp_a_at = compare(&a_run1, &at_run1).ok();

            println!(
                "phase0 kind={KIND_TARGET} m={m} n={n} k={k} classic_stable={classic_stable} \
                 target_tile_stable={target_tile_stable} splitk_stable={splitk_stable} \
                 checksum_a={checksum_a:.6e} checksum_at={checksum_at:.6e} \
                 checksum_b={checksum_b:.6e} a_vs_b_fail_count={} a_vs_at_fail_count={}",
                cmp_ab
                    .map(|r| r.fail_count.to_string())
                    .unwrap_or_else(|| "NA".to_string()),
                cmp_a_at
                    .map(|r| r.fail_count.to_string())
                    .unwrap_or_else(|| "NA".to_string()),
            );
        }

        for &(m, n, k) in control_shapes().iter() {
            assert!(
                tile::should_split_k(m, n, k).is_none(),
                "対照形状 (256,256,*) は should_split_k が None を返す前提が崩れている: \
                 m={m}, n={n}, k={k}"
            );
            let p = prepare(ctx, m, n, k, 3);
            let select_cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());
            dispatch_classic(gemm, ctx, &p, select_cfg);
            let run1 = p.c_buf.read_to_vec();
            dispatch_classic(gemm, ctx, &p, select_cfg);
            let run2 = p.c_buf.read_to_vec();
            let stable = bit_equal(&run1, &run2);
            all_ok &= stable;
            println!("phase0 kind={KIND_CONTROL} m={m} n={n} k={k} classic_stable={stable}");
        }

        if !all_ok {
            eprintln!(
                "phase0 self-check で run-to-run bit 不一致を検出した。計測妥当性の前提が \
                 崩れているため中止する。"
            );
            std::process::exit(3);
        }
        println!("phase0 self-check: 全形状で run-to-run bit 同一を確認した");
    }

    /// フェーズ 1（対象 9 形状。A=classic vs B=split-K）。
    fn phase1_target(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        ab_config: &AbConfig,
        measurement_config: &MeasurementConfig,
    ) {
        for &(m, n, k) in target_shapes().iter() {
            let plan = tile::should_split_k(m, n, k)
                .unwrap_or_else(|| panic!("should_split_k が None を返した: m={m}, n={n}, k={k}"));
            let p_a = prepare(ctx, m, n, k, 10);
            let p_b = prepare(ctx, m, n, k, 20);
            let select_cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());

            dispatch_classic(gemm, ctx, &p_a, select_cfg);
            dispatch_split_k(gemm, ctx, &p_b, plan);

            let result = run_ab(
                ab_config,
                measurement_config,
                || dispatch_classic(gemm, ctx, &p_a, select_cfg),
                || dispatch_split_k(gemm, ctx, &p_b, plan),
            )
            .expect("run_ab が失敗した（MeasurementConfig::default は下限を満たす）");

            println!(
                "{}",
                format_ab_line(
                    KIND_TARGET,
                    m,
                    n,
                    k,
                    Some(plan.partitions),
                    result.median_a_secs,
                    result.median_b_secs,
                    result.spread_a,
                    result.spread_b,
                )
            );
        }
    }

    /// フェーズ 1（対象 9 形状。A=classic〈select_for_device〉vs
    /// A′=classic〈split_k_tile〉。参考のみ・タイル構成効果の分離）。
    fn phase1_target_tile(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        ab_config: &AbConfig,
        measurement_config: &MeasurementConfig,
    ) {
        for &(m, n, k) in target_shapes().iter() {
            let p_a = prepare(ctx, m, n, k, 30);
            let p_at = prepare(ctx, m, n, k, 40);
            let select_cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());
            let tile_cfg = tile::split_k_tile(m, n);

            dispatch_classic(gemm, ctx, &p_a, select_cfg);
            dispatch_classic(gemm, ctx, &p_at, tile_cfg);

            let result = run_ab(
                ab_config,
                measurement_config,
                || dispatch_classic(gemm, ctx, &p_a, select_cfg),
                || dispatch_classic(gemm, ctx, &p_at, tile_cfg),
            )
            .expect("run_ab が失敗した（MeasurementConfig::default は下限を満たす）");

            println!(
                "{}",
                format_ab_line(
                    KIND_TARGET_TILE,
                    m,
                    n,
                    k,
                    None,
                    result.median_a_secs,
                    result.median_b_secs,
                    result.spread_a,
                    result.spread_b,
                )
            );
        }
    }

    /// フェーズ 1（対照 3 形状。A=classic vs B′=classic〈`should_split_k`
    /// が None を返すことを assert したうえで classic を dispatch。
    /// 結線相当経路）。
    fn phase1_control(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        ab_config: &AbConfig,
        measurement_config: &MeasurementConfig,
    ) {
        for &(m, n, k) in control_shapes().iter() {
            assert!(
                tile::should_split_k(m, n, k).is_none(),
                "対照形状の前提が崩れている: m={m}, n={n}, k={k}"
            );
            let p_a = prepare(ctx, m, n, k, 50);
            let p_b = prepare(ctx, m, n, k, 60);
            let select_cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());

            dispatch_classic(gemm, ctx, &p_a, select_cfg);
            dispatch_classic(gemm, ctx, &p_b, select_cfg);

            // B′ の計測クロージャは `should_split_k` を呼んでから classic
            // へ dispatch する（`結線相当経路` を名乗る以上、本番の
            // `dispatch_auto` 相当が毎回払う選択関数の呼び出し費用込みで
            // 計測する必要がある。呼ばないと A と全く同一の経路を測る
            // だけになり B′ の存在意義がなくなる。イシュー #1499
            // codex-review 指摘）。対照形状では `None` を返す前提は
            // ループ先頭の assert で確認済みのため正しさの検査自体は
            // `debug_assert` に留めるが、戻り値をそれだけに任せると
            // release ビルドの計測経路では `debug_assert!` が消え
            // `route` が未使用になり、最適化で `should_split_k` の呼び出し
            // ごと消去されて「選択関数の呼び出し費用込み」の計測契約が
            // release ビルドで保証されなくなる（イシュー #1499
            // codex-review 指摘）。`std::hint::black_box` で戻り値を
            // 不透明化して消去を防ぎ、debug/release いずれでも
            // `should_split_k` が確実に評価されるようにする。
            let result = run_ab(
                ab_config,
                measurement_config,
                || dispatch_classic(gemm, ctx, &p_a, select_cfg),
                || {
                    let route = std::hint::black_box(tile::should_split_k(
                        std::hint::black_box(m),
                        std::hint::black_box(n),
                        std::hint::black_box(k),
                    ));
                    debug_assert!(route.is_none(), "対照形状の前提が崩れている");
                    dispatch_classic(gemm, ctx, &p_b, select_cfg);
                },
            )
            .expect("run_ab が失敗した（MeasurementConfig::default は下限を満たす）");

            println!(
                "{}",
                format_ab_line(
                    KIND_CONTROL,
                    m,
                    n,
                    k,
                    None,
                    result.median_a_secs,
                    result.median_b_secs,
                    result.spread_a,
                    result.spread_b,
                )
            );
        }
    }

    /// フェーズ 1（対照 3 形状。A=classic vs C=強制 split-K〈`max_groups=
    /// u64::MAX`〉。参考のみ・`max_groups=40` 境界の妥当性記録）。
    fn phase1_control_forced(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        ab_config: &AbConfig,
        measurement_config: &MeasurementConfig,
    ) {
        let forced_params = SplitKParams {
            max_groups: u64::MAX,
            ..SplitKParams::MLX_CASE1_M4_MAX
        };
        for &(m, n, k) in control_shapes().iter() {
            let Some(plan) = tile::should_split_k_with(m, n, k, &forced_params) else {
                println!(
                    "phase1 kind={KIND_CONTROL_FORCED} m={m} n={n} k={k} skipped=true \
                     reason=should_split_k_with(max_groups=MAX)がNoneを返した"
                );
                continue;
            };
            let p_a = prepare(ctx, m, n, k, 70);
            let p_c = prepare(ctx, m, n, k, 80);
            let select_cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());

            dispatch_classic(gemm, ctx, &p_a, select_cfg);
            dispatch_split_k(gemm, ctx, &p_c, plan);

            let result = run_ab(
                ab_config,
                measurement_config,
                || dispatch_classic(gemm, ctx, &p_a, select_cfg),
                || dispatch_split_k(gemm, ctx, &p_c, plan),
            )
            .expect("run_ab が失敗した（MeasurementConfig::default は下限を満たす）");

            println!(
                "{}",
                format_ab_line(
                    KIND_CONTROL_FORCED,
                    m,
                    n,
                    k,
                    Some(plan.partitions),
                    result.median_a_secs,
                    result.median_b_secs,
                    result.spread_a,
                    result.spread_b,
                )
            );
        }
    }

    /// フロア計測（各 `(M,N)` を `FLOOR_K` で classic dispatch。単一腕・
    /// `run_stability` を使う。参考のみ）。
    fn phase1_floor(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        ab_config: &AbConfig,
        measurement_config: &MeasurementConfig,
    ) {
        for &(m, n, k) in floor_shapes().iter() {
            let p = prepare(ctx, m, n, k, 90);
            let select_cfg = tile::select_for_device(m, n, k, ctx.verified_m4_max_gpu_core_count());
            dispatch_classic(gemm, ctx, &p, select_cfg);

            let result = run_stability(ab_config, measurement_config, || {
                dispatch_classic(gemm, ctx, &p, select_cfg)
            })
            .expect("run_stability が失敗した（MeasurementConfig::default は下限を満たす）");

            let median = bench_harness::median_q1_q3(&result.round_medians_secs)
                .expect("run_stability が返す round_medians_secs は非空のため成功する")
                .median;
            println!("{}", format_floor_line(m, n, k, median, result.spread));
        }
    }

    pub fn main() {
        let args = match parse_args_from(std::env::args().skip(1)) {
            Ok(args) => args,
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        };
        let measurement_config = match resolve_measurement_config(&args) {
            Ok(config) => config,
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        };
        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");

        match run_env_guard(&args) {
            Ok(_) => {}
            Err(BenchError::EnvGuardExhausted { attempts, detail }) => {
                println!(
                    "env_guard_result=exhausted attempts={attempts} detail={detail}\nverdict=undetermined"
                );
                std::process::exit(1);
            }
            Err(e) => {
                eprintln!("env_guard の初期化に失敗した: {e}");
                std::process::exit(1);
            }
        }

        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        phase0_self_check(&gemm, &ctx);
        if args.self_check_only {
            return;
        }

        phase1_target(&gemm, &ctx, &ab_config, &measurement_config);
        phase1_target_tile(&gemm, &ctx, &ab_config, &measurement_config);
        phase1_control(&gemm, &ctx, &ab_config, &measurement_config);
        phase1_control_forced(&gemm, &ctx, &ab_config, &measurement_config);
        phase1_floor(&gemm, &ctx, &ab_config, &measurement_config);
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け: `should_split_k` の解析値のみ出力する
/// （`gemm_splitk_shapes_bench.rs` の非 macOS フォールバックと同型）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_splitk_ab_bench example: 性能 A/B 計測は macOS（Apple Silicon）\
         実機限定。以下は should_split_k の解析値（対象 9・対照 3 形状）。\n"
    );
    for &(m, n, k) in target_shapes().iter() {
        let plan = fandhe_ai_backend_metal::tile::should_split_k(m, n, k);
        println!("target m={m} n={n} k={k} should_split_k={plan:?}");
    }
    for &(m, n, k) in control_shapes().iter() {
        let plan = fandhe_ai_backend_metal::tile::should_split_k(m, n, k);
        println!("control m={m} n={n} k={k} should_split_k={plan:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_args_from_accepts_known_flags() {
        let args = parse_args_from(vec![
            "--max-load-avg=4.0".to_string(),
            "--guard-max-attempts=3".to_string(),
            "--iters=30".to_string(),
        ])
        .expect("既知の引数の組は成功する");
        assert_eq!(args.max_load_avg, Some(4.0));
        assert_eq!(args.guard_max_attempts, Some(3));
        assert_eq!(args.iters, Some(30));
        assert!(!args.self_check_only);
    }

    #[test]
    fn parse_args_from_self_check_only() {
        let args = parse_args_from(vec!["--self-check-only".to_string()])
            .expect("--self-check-only 単独指定は成功する");
        assert!(args.self_check_only);
    }

    #[test]
    fn parse_args_from_rejects_unknown_flag() {
        assert!(parse_args_from(vec!["--bogus".to_string()]).is_err());
    }

    #[test]
    fn parse_args_from_rejects_duplicate_max_load_avg() {
        assert!(
            parse_args_from(vec![
                "--max-load-avg=1.0".to_string(),
                "--max-load-avg=2.0".to_string(),
            ])
            .is_err()
        );
    }

    #[test]
    fn parse_args_from_rejects_non_finite_max_load_avg() {
        assert!(parse_args_from(vec!["--max-load-avg=nan".to_string()]).is_err());
        assert!(parse_args_from(vec!["--max-load-avg=-1.0".to_string()]).is_err());
        assert!(parse_args_from(vec!["--max-load-avg=0.0".to_string()]).is_err());
    }

    #[test]
    fn parse_args_from_rejects_guard_max_attempts_without_max_load_avg() {
        assert!(parse_args_from(vec!["--guard-max-attempts=3".to_string()]).is_err());
    }

    #[test]
    fn target_shapes_has_nine_entries() {
        let shapes = target_shapes();
        assert_eq!(shapes.len(), 9);
        for &(m, n, _) in &shapes {
            assert_eq!(m, n);
            assert!(TARGET_MN.contains(&m));
        }
    }

    #[test]
    fn control_shapes_has_three_entries_at_control_mn() {
        let shapes = control_shapes();
        assert_eq!(shapes.len(), 3);
        for &(m, n, _) in &shapes {
            assert_eq!((m, n), (CONTROL_MN, CONTROL_MN));
        }
    }

    #[test]
    fn floor_shapes_uses_floor_k() {
        let shapes = floor_shapes();
        assert_eq!(shapes.len(), 4);
        for &(_, _, k) in &shapes {
            assert_eq!(k, FLOOR_K);
        }
    }

    #[test]
    fn should_split_k_is_some_for_all_target_shapes() {
        for &(m, n, k) in target_shapes().iter() {
            assert!(
                fandhe_ai_backend_metal::tile::should_split_k(m, n, k).is_some(),
                "対象形状の前提が崩れている: m={m}, n={n}, k={k}"
            );
        }
    }

    #[test]
    fn should_split_k_is_none_for_all_control_shapes() {
        for &(m, n, k) in control_shapes().iter() {
            assert!(
                fandhe_ai_backend_metal::tile::should_split_k(m, n, k).is_none(),
                "対照形状の前提が崩れている: m={m}, n={n}, k={k}"
            );
        }
    }

    #[test]
    fn should_split_k_with_forced_max_groups_is_some_for_control_shapes() {
        let forced_params = fandhe_ai_backend_metal::tile::SplitKParams {
            max_groups: u64::MAX,
            ..fandhe_ai_backend_metal::tile::SplitKParams::MLX_CASE1_M4_MAX
        };
        for &(m, n, k) in control_shapes().iter() {
            assert!(
                fandhe_ai_backend_metal::tile::should_split_k_with(m, n, k, &forced_params)
                    .is_some(),
                "強制 split-K の前提が崩れている: m={m}, n={n}, k={k}"
            );
        }
    }

    #[test]
    fn format_ab_line_computes_speedup_as_a_over_b() {
        let line = format_ab_line("target", 32, 32, 2048, Some(4), 2.0e-3, 1.0e-3, 0.1, 0.2);
        assert!(line.contains("speedup=2.0000"));
        assert!(line.contains("kind=target"));
        assert!(line.contains("partitions=4"));
    }

    /// [`bit_equal`] の自己検証（イシュー #1499 codex-review 指摘対応）:
    /// `&[f32]` の `==` が見逃す `+0.0`／`-0.0` の符号ビット差異・長さ不一致
    /// を `bit_equal` が正しく検出することを確認する。
    #[test]
    fn bit_equal_distinguishes_signed_zero_and_detects_length_mismatch() {
        assert!(bit_equal(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]));
        // +0.0 と -0.0 は `==` では等しいが `to_bits()` では異なる。
        assert!(!bit_equal(&[0.0], &[-0.0]));
        // NaN は `to_bits()` が同一ビットパターンなら bit_equal で一致扱い
        // にする（`NaN != NaN` の数値比較特性に引きずられない）。
        let nan = f32::NAN;
        assert!(bit_equal(&[nan], &[nan]));
        assert!(!bit_equal(&[1.0, 2.0], &[1.0]));
    }

    #[test]
    fn format_ab_line_partitions_none_renders_na() {
        let line = format_ab_line("control", 256, 256, 2048, None, 1.0e-3, 1.0e-3, 0.0, 0.0);
        assert!(line.contains("partitions=NA"));
    }

    #[test]
    fn format_floor_line_renders_na_sentinels_for_b_side() {
        let line = format_floor_line(32, 32, 64, 1.0e-4, 0.05);
        assert!(line.contains("median_b_secs=NA"));
        assert!(line.contains("speedup=NA"));
        assert!(line.contains("spread_b=NA"));
        assert!(line.contains("kind=floor"));
    }
}
