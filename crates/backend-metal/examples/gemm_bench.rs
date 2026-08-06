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
    use backend_metal::{GemmVariant, MetalContext, MetalGemm, TileConfig};
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
