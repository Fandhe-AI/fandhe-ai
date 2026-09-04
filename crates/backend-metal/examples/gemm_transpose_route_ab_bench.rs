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
//! cargo run -p fandhe-ai-backend-metal --example gemm_transpose_route_ab_bench --release
//! ```
//!
//! 実行前後に `pmset -g therm` でサーマル状態を記録すること
//! （`docs/perf/metal-gemm-transpose-tiled.md` §5 実行手順参照）。
//! フェーズ 1 でいずれかのサイズが安定性ゲート
//! （`bench_harness::ab::STABILITY_SPREAD_GATE`）を超過した場合、
//! フェーズ 2（A/B 判定）には進まない（「判定不可」を出力して終了する。
//! 安全側判断: 判定を無効化して中断する方向のみ許す）。

#[cfg(target_os = "macos")]
mod macos_impl {
    use bench_harness::MeasurementConfig;
    use bench_harness::ab::{AbConfig, run_ab, run_stability};
    use bench_harness::rng::Xorshift64Star;
    use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
    use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, tile};
    use std::time::Duration;

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
    const MIN_WARMUP: Duration = Duration::from_secs(3);

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

    /// フェーズ 1: 対照カーネルとして `dispatch_auto`（`dispatch_auto` の
    /// 本番既定経路と同一構成選択。`gemm_swizzle_ab_bench.rs::
    /// phase1_stability_selfcheck` と同一手法）を各サイズで
    /// [`run_stability`] 計測し、spread を出力する。
    fn phase1_stability_selfcheck(ctx: &MetalContext, gemm: &MetalGemm) -> bool {
        // 安定性ゲートの値自体は `bench_harness::ab::STABILITY_SPREAD_GATE`
        // を単一真実源とする（`docs/perf/metal-bench-noise-protocol.md` と
        // 同じ値を example 内に直接複製すると、閾値変更時にコードと文書が
        // 独立に乖離しうるため。`gemm_swizzle_ab_bench.rs` と同じ判断）。
        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;
        println!("--- フェーズ 1: 安定性セルフチェック（対照カーネル: dispatch_auto）---");

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        let mut all_within_gate = true;
        for size in [256usize, 512, 1024, 2048, 4096] {
            let mut rng = Xorshift64Star::new(SEED);
            let a = rng.fill_vec(size * size);
            let b = rng.fill_vec(size * size);

            let result = run_stability(&ab_config, &measurement_config, || {
                gemm.dispatch_auto(ctx, &a, &b, size, size, size)
                    .expect("Metal GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
            })
            .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

            let within_gate = result.spread <= SPREAD_GATE;
            all_within_gate &= within_gate;

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
        }

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
        })
    }

    /// フェーズ 2: 全 10 形状 × NT/TN/TT（計 30 セル）の A/B を計測し、
    /// 総括判定（`verdict`）を出力する。
    fn phase2_route_ab(ctx: &MetalContext, gemm: &MetalGemm) {
        println!(
            "--- フェーズ 2: 転置タイル variant ルーティング A/B（A=classic strided / B=strided tiled）---"
        );

        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");
        let measurement_config = MeasurementConfig::default();

        // 安定性ゲートの値自体は `bench_harness::ab::STABILITY_SPREAD_GATE`
        // を単一真実源とする（`phase1_stability_selfcheck` と同じ判断）。
        const SPREAD_GATE: f64 = bench_harness::ab::STABILITY_SPREAD_GATE;

        let mut ratios: Vec<((usize, usize, usize), &'static str, f64)> = Vec::new();
        let mut skipped: Vec<((usize, usize, usize), &'static str)> = Vec::new();
        let mut gate_exceeded: Vec<GateExceededCell> = Vec::new();

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
                        ratios.push(((m, n, k), label, cell.b_over_a_tflops));
                    }
                    None => skipped.push(((m, n, k), label)),
                }
            }
        }

        let below_threshold: Vec<_> = ratios.iter().filter(|(_, _, ratio)| *ratio < 1.0).collect();

        let verdict = if !skipped.is_empty() || !gate_exceeded.is_empty() {
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
            "cells_measured={} cells_skipped={} cells_below_threshold={} cells_gate_exceeded={}",
            ratios.len(),
            skipped.len(),
            below_threshold.len(),
            gate_exceeded.len()
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
                    "skip セルまたは spread gate 超過セルが残っており判定不可\
                     （適格性ゲート不成立、またはラウンド間ばらつきが大きく \
                     計測値を信頼できないセルがある）"
                }
            }
        );
        println!(
            "--- 実測結果は docs/perf/metal-gemm-transpose-tiled.md §5 へ記録すること（本番経路・テストは無変更のまま）。"
        );
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let phase1_ok = phase1_stability_selfcheck(&ctx, &gemm);
        if !phase1_ok {
            return;
        }

        phase2_route_ab(&ctx, &gemm);
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
