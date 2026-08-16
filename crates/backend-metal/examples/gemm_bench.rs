//! Metal GEMM 4 段（naive/tiled/simdgroup/dynamic-tile）性能実測バイナリ
//! （TASK-1.8c・#40 の naive/tiled/simdgroup 計測に、TASK-1.8f・#188 の
//! 動的タイル選択計測を追加した）。
//!
//! 受け入れ条件「simdgroup 版が naive 比で PoC-v2-4 相当の性能を示す」
//! （PoC-v2-4 実測: Apple M4 Max・size=4096 で naive 1.271 → tiled 2.123 →
//! simdgroup 3.134 TFLOPS、simdgroup/naive ≒ ×2.47）に加え、TASK-1.8f の
//! 受け入れ条件「動的タイル選択（`dispatch_auto`）が simdgroup 版比で
//! 性能向上を示す実測記録」の実測手段を兼ねる。計測コアは
//! `bench-harness::protocol::run`（warmup 20 回以上・計測 20 回以上・
//! 中央値/Q1/Q3。TASK-8.1）を使い、`.claude/rules/coding-rust.md`
//! 「ベンチは 5 回計測の中央値を採用」を満たす（`crates/backend-cpu/examples/gemm_bench.rs`
//! と同じ計測コアを再利用する判断）。
//!
//! `examples/` に置くのは、`dev-dependencies`（`bench-harness`）を利用
//! しつつ、通常の `cargo test`／CI では実行されず、ビルド検証
//! （`cargo build --workspace --all-targets`）のみが CI で走るようにする
//! ためである（self-hosted runner をベンチ実行で占有しない。`ci.md`）。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_bench --release
//! ```
//!
//! size=256/512/1024/2048/4096（正方）で naive/tiled/simdgroup/
//! dynamic-tile-auto を計測して auto/simdgroup・simdgroup/naive 比を出力し、
//! 続けて縦長・横長の非正方形状（`crate::tile::select` の分岐実測）、
//! 最後に候補構成（`GemmVariant::SimdgroupTiled`）ごとの明示比較を出力
//! する。実測値は `docs/perf/metal-gemm-dynamic-tile.md` の記録テンプレへ
//! 転記する（イシュー #188 計画「実機実測」節）。非 macOS 環境（本実装
//! 環境を含む）では `main` が説明を表示するだけの stub になり、
//! `cargo build`／`cargo clippy --all-targets` はビルド対象として通る
//! （Linux CI でもコンパイル検証できるようにするため）。

#[cfg(target_os = "macos")]
mod macos_impl {
    use backend_metal::{GemmVariant, MetalContext, MetalGemm, TileConfig, tile};
    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, run as bench_run};

    /// 決定的シード（`crates/backend-cpu/examples/gemm_bench.rs` と同一値。
    /// 過去 PoC・CPU 実装ベンチと同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    fn tflops(size: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (size as f64).powi(3);
        flops / median_secs / 1e12
    }

    /// `size×size×size` の正方 GEMM を `variant` で計測し、中央値 TFLOPS を
    /// 返す。`MetalGemm::dispatch_variant` は呼び出しごとに A・B のアップ
    /// ロード・readback を含む（`docs/spec/.../metal_gemm.rs` の
    /// `GemmCase::dispatch`〈アップロード済みバッファを使い回す設計〉とは
    /// 異なる計測範囲になる点に注意。本 productize 版はディスパッチ入口を
    /// 「1 回の呼び出しで完結する」設計にしたため〈`crate::gemm` 参照〉。
    /// 受け入れ条件の比較対象は simdgroup/naive の相対比であり、両者とも
    /// 同じ計測範囲で揃っているため相対比較としては妥当と判断する）。
    fn measure(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        variant: GemmVariant,
        size: usize,
        config: &MeasurementConfig,
    ) -> f64 {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(size * size);
        let b = rng.fill_vec(size * size);

        let measurement = bench_run(config, || {
            gemm.dispatch_variant(ctx, variant, &a, &b, size, size, size)
                .expect("Metal GEMM ディスパッチに失敗した（実機でのみ実行する前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        tflops(size, measurement.median_secs)
    }

    /// `m×n×k`（非正方対応）の GEMM を `variant` で計測し、中央値 TFLOPS を
    /// 返す（TASK-1.8f・#188。[`measure`] の正方形状専用版に対して形状を
    /// 個別指定できるようにした版）。
    fn measure_shape(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        variant: GemmVariant,
        m: usize,
        n: usize,
        k: usize,
        config: &MeasurementConfig,
    ) -> f64 {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(m * k);
        let b = rng.fill_vec(k * n);

        let measurement = bench_run(config, || {
            gemm.dispatch_variant(ctx, variant, &a, &b, m, n, k)
                .expect("Metal GEMM ディスパッチに失敗した（実機でのみ実行する前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        flops / measurement.median_secs / 1e12
    }

    /// 昇順ソート済みでない `f64` 列から中央値を求める（要素数が偶数の
    /// 場合は中央 2 要素の平均）。`.claude/rules/coding-rust.md` の
    /// 「ベンチは 5 回計測の中央値を採用」に合わせ、occupancy 判定組み込み
    /// 比較のラウンド交互方式（後述）で得た各ラウンドの計測値を集約する
    /// のに使う。
    fn median(mut values: Vec<f64>) -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).expect("TFLOPS は NaN にならない"));
        let n = values.len();
        if n % 2 == 1 {
            values[n / 2]
        } else {
            (values[n / 2 - 1] + values[n / 2]) / 2.0
        }
    }

    /// `dispatch_auto`（`crate::tile::select` による行列サイズ別自動選択。
    /// TASK-1.8f・#188）を計測する。
    fn measure_auto(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        config: &MeasurementConfig,
    ) -> f64 {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(m * k);
        let b = rng.fill_vec(k * n);

        let measurement = bench_run(config, || {
            gemm.dispatch_auto(ctx, &a, &b, m, n, k)
                .expect("Metal GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        flops / measurement.median_secs / 1e12
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        for size in [256usize, 512, 1024, 2048, 4096] {
            // naive は大サイズで所要時間が過大になりやすいが、
            // `MeasurementConfig` の下限（warmup・計測とも 20 回）を
            // `MeasurementConfig::default()` がすでに満たしているため、
            // これ以上計測回数を減らす調整はしていない（「5 回計測の
            // 中央値」下限〈coding-rust.md〉は満たしたまま）。
            let config = MeasurementConfig::default();

            let naive = measure(&gemm, &ctx, GemmVariant::Naive, size, &config);
            let tiled = measure(&gemm, &ctx, GemmVariant::Tiled, size, &config);
            let simdgroup = measure(&gemm, &ctx, GemmVariant::Simdgroup, size, &config);
            // TASK-1.8f（#188）: dispatch_auto（行列サイズ別自動タイル選択）
            // を simdgroup（1.8c 固定 8x8 タイル）と比較する（受け入れ条件
            // 「simdgroup 版比で性能向上を示す実測記録」）。
            let auto = measure_auto(&gemm, &ctx, size, size, size, &config);

            println!(
                "size={size} naive_tflops={naive:.4} tiled_tflops={tiled:.4} simdgroup_tflops={simdgroup:.4} dynamic_tile_auto_tflops={auto:.4} auto_over_simdgroup={:.4} simdgroup_over_naive={:.4}",
                auto / simdgroup,
                simdgroup / naive
            );
        }

        // 非正方形状（縦長・横長）: `crate::tile::select` の
        // tall/wide 分岐（`docs/perf/metal-gemm-dynamic-tile.md` に記録する
        // 選択閾値確定の実測対象）。
        for &(m, n, k) in &[
            (4096usize, 512usize, 512usize), // 縦長
            (512usize, 4096usize, 512usize), // 横長
        ] {
            let config = MeasurementConfig::default();
            let simdgroup = measure_shape(&gemm, &ctx, GemmVariant::Simdgroup, m, n, k, &config);
            let auto = measure_auto(&gemm, &ctx, m, n, k, &config);
            println!(
                "shape=({m}x{n}x{k}) simdgroup_tflops={simdgroup:.4} dynamic_tile_auto_tflops={auto:.4} auto_over_simdgroup={:.4}",
                auto / simdgroup
            );
        }

        // 候補構成の明示比較（`GemmVariant::SimdgroupTiled`。size=2048 固定
        // 形状で BM/BN/BK/WM/WN・staged 有無ごとの実測を残す）。
        let size = 2048usize;
        let config = MeasurementConfig::default();
        for (label, cfg) in [
            (
                "bm64_bn64_bk16_staged",
                TileConfig {
                    bm: 64,
                    bn: 64,
                    bk: 16,
                    wm: 2,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "bm32_bn32_bk16_staged",
                TileConfig {
                    bm: 32,
                    bn: 32,
                    bk: 16,
                    wm: 2,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "bm32_bn32_bk16_direct",
                TileConfig {
                    bm: 32,
                    bn: 32,
                    bk: 16,
                    wm: 2,
                    wn: 2,
                    staged: false,
                },
            ),
        ] {
            let tflops = measure(&gemm, &ctx, GemmVariant::SimdgroupTiled(cfg), size, &config);
            println!("size={size} candidate={label} tflops={tflops:.4}");
        }

        // occupancy 判定組み込み比較（イシュー #542。受け入れ条件 2:
        // 「size ∈ {512, 1024, 2048, 4096} で現行 select() 比の劣化がない
        // ことを実測確認する」）。
        //
        // 「旧」: `tile::select`（形状のみの判定。occupancy 縮退なし）が
        // 選ぶ構成を `SimdgroupTiled` へ明示指定してディスパッチする
        // （`dispatch_auto` の現行本番挙動と同一。`select_with_occupancy`
        // は M4 Max 実機での性能非劣化確認が未完了のため `dispatch_auto`
        // へは未適用〈codex-review P1・PR #684。`crate::gemm` モジュール
        // ドキュメンテーションコメント参照〉）。
        // 「新」: `tile::select_with_occupancy`（`ctx.occupancy_params()`
        // 経由の occupancy 縮退込み判定）が選ぶ構成を同じく `SimdgroupTiled`
        // へ明示指定してディスパッチする（`dispatch_auto` 経由ではなく
        // 直接 `select_with_occupancy` の結果を計測する。GPU コア数取得
        // 不能時は `tile::select` と同一構成へ fail-safe フォールバックする）。
        // 本比較で非劣化を確認できたら `dispatch_auto` を
        // `select_with_occupancy` 呼び出しへ切り替える（別 PR）。
        //
        // 選択された `TileConfig`（`bm`/`bn`）も出力し、`docs/perf/
        // metal-gemm-occupancy-select.md` の記録テンプレへ転記できるように
        // する。
        println!("--- occupancy 判定組み込み比較（旧: select / 新: select_with_occupancy）---");
        // ラウンド交互方式（codex-review 指摘・PR #684）: 旧→新の固定順で
        // 全 warmup・計測を終えてから他方を計測すると、GPU の DVFS・温度
        // 上昇によるクロックスロットリングが計測順序に系統的に乗り、
        // `new_over_old` 比が測定順序に左右されてしまう（受け入れ条件
        // 「select() 比の性能非劣化」の判定を汚染しうる）。これを避ける
        // ため、ROUNDS 回の独立ラウンドに分割し、ラウンドごとに旧→新／
        // 新→旧の順序を反転させながら交互計測する。旧・新それぞれの
        // ラウンド計測値（各ラウンドは `bench_run` 内部で warmup・計測とも
        // `MeasurementConfig` の下限を満たした上での中央値）をさらに束ね、
        // その中央値を最終値として `new_over_old` を求める。
        //
        // ROUNDS は偶数固定とする（codex-review・Cursor Bugbot 指摘・PR
        // #684 追加レビュー）: 奇数だと「偶数ラウンドは旧→新・奇数ラウンド
        // は新→旧」の反転規則の下で旧先頭ラウンドが 1 回多くなり
        // （ROUNDS=5 なら旧先頭 3 回・新先頭 2 回）、old-first バイアスを
        // 完全には相殺できず `new_over_old` を系統的に過小評価しうる。
        // 偶数化により旧先頭・新先頭が同数（ROUNDS=6 で 3 対 3）になり、
        // 順序バイアスをラウンド間で厳密に相殺する。
        const ROUNDS: usize = 6;
        for size in [512usize, 1024, 2048, 4096] {
            let config = MeasurementConfig::default();

            let old_cfg = tile::select(size, size, size);
            let new_cfg = tile::select_with_occupancy(size, size, size, ctx.occupancy_params());

            let mut old_samples = Vec::with_capacity(ROUNDS);
            let mut new_samples = Vec::with_capacity(ROUNDS);

            for round in 0..ROUNDS {
                // 偶数ラウンド（0-origin）は旧→新、奇数ラウンドは新→旧に
                // 反転し、順序バイアスをラウンド間で打ち消す。
                if round % 2 == 0 {
                    old_samples.push(measure(
                        &gemm,
                        &ctx,
                        GemmVariant::SimdgroupTiled(old_cfg),
                        size,
                        &config,
                    ));
                    new_samples.push(measure(
                        &gemm,
                        &ctx,
                        GemmVariant::SimdgroupTiled(new_cfg),
                        size,
                        &config,
                    ));
                } else {
                    new_samples.push(measure(
                        &gemm,
                        &ctx,
                        GemmVariant::SimdgroupTiled(new_cfg),
                        size,
                        &config,
                    ));
                    old_samples.push(measure(
                        &gemm,
                        &ctx,
                        GemmVariant::SimdgroupTiled(old_cfg),
                        size,
                        &config,
                    ));
                }
            }

            let old_tflops = median(old_samples);
            let new_tflops = median(new_samples);

            println!(
                "size={size} old_tile=({}x{}) old_tflops={old_tflops:.4} new_tile=({}x{}) new_tflops={new_tflops:.4} new_over_old={:.4} occupancy_params={:?}",
                old_cfg.bm,
                old_cfg.bn,
                new_cfg.bm,
                new_cfg.bn,
                new_tflops / old_tflops,
                ctx.occupancy_params(),
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け stub（`objc2` 系は `cfg(target_os = "macos")` 限定の
/// ため本クレートの GEMM 実装自体がコンパイル対象外になる。Linux CI の
/// `cargo build --workspace --all-targets`／`cargo clippy --all-targets`
/// をこの example も含めて通すための最小 main）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_bench example requires macOS (Apple Silicon). \
         run it on macOS hardware: cargo run -p backend-metal --example gemm_bench --release"
    );
}
