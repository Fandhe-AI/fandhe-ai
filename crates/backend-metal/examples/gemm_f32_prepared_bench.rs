//! Metal f32 `SimdgroupTiled` GEMM の §4 準拠計測境界バイナリ（イシュー
//! #572・Phase F-2）。
//!
//! `docs/perf/gemm-optimization-baseline.md` §2 が「f32 側には
//! `dispatch_f16_prepared_unverified` と同型の §4 準拠 prepared 入口が
//! 存在しない」問題を指摘し、その解消（`MetalGemm::dispatch_tiled_prepared`
//! の追加）を #572 のスコープとした対応。本バイナリは同入口を使い、
//! `docs/performance-targets.md` §4 の計測プロトコル（warmup 20 回以上・
//! 計測 20 回以上の中央値、ホスト転送を伴わない完了待ちのみを計測）で
//! f32 TFLOPS を出力する。
//!
//! 相対値の分母は `scripts/bench/gemm_bench_torch_mps_f32.py`（同一実機上で
//! 別プロセスとして計測する PyTorch MPS f32 ベースライン。`torch.mm` +
//! `torch.mps.synchronize()` のみ計測）とし、本バイナリ自体は Rust 側の
//! TFLOPS のみを出力する（REQ-8 v2「同一ハードウェア上の PyTorch とのみ
//! 比較」方針。実測記録は `docs/perf/metal-floor-remeasurement.md` へ転記
//! する）。
//!
//! 既存 `gemm_bench.rs`（`dispatch_auto` 経由・転送込み境界。#381 比較系列）
//! は改変しない。本バイナリは計測境界を PyTorch 側スクリプトと揃えるための
//! 別系列であり、両者は独立に維持する。
//!
//! `tile::select(m, n, k)` で選んだ構成を [`MetalGemm::dispatch_tiled_prepared`]
//! へ渡し、実際に採用された構成（フォールバック解決後。`pipeline_for_tile`
//! ドキュメントコメント参照）をログへ含めることでフォールバック透明性を
//! 保つ（`dispatch_auto` を直接計測すると内部でフォールバックしても外側
//! からは判別できない問題を避ける）。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_f16_bench.rs`・
//! `gemm_bench.rs` と同一（self-hosted runner をベンチ実行で占有しない・
//! Linux CI でもビルド検証のみ通す）。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_f32_prepared_bench --release
//! ```
//!
//! 実行前に数値一致（`gemm_dynamic_tile_parity.rs` の
//! `dispatch_tiled_prepared_matches_cpu_reference` 系）を確認することを
//! 推奨する:
//!
//! ```sh
//! cargo test -p backend-metal --release -- --ignored --nocapture dispatch_tiled_prepared
//! ```

#[cfg(target_os = "macos")]
mod macos_impl {
    use backend_metal::pad::{pad_matrix, pad8};
    use backend_metal::tile;
    use backend_metal::{MetalBuffer, MetalContext, MetalGemm};
    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, run as bench_run};

    /// 決定的シード（`gemm_bench.rs::SEED`・`gemm_f16_bench.rs::SEED` と
    /// 同一値。PoC-v2 系・既存 bench と同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    fn tflops(m: usize, n: usize, k: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        flops / median_secs / 1e12
    }

    /// `m×n×k` の f32 `SimdgroupTiled` GEMM（`MetalGemm::dispatch_tiled_prepared`）
    /// を計測し、中央値 TFLOPS と実際に採用された [`tile::TileConfig`] の
    /// ラベルを返す。`gemm_f16_bench.rs::measure` の f32 版（ディスパッチ
    /// 入口・構成選択が異なるため独立実装）。
    ///
    /// パディング・バッファ確保／アップロードは計測ループの外で 1 回だけ
    /// 行い、計測対象はディスパッチ（エンコード＋コマンドバッファ完了待ち）
    /// のみとする。PyTorch 側（`gemm_bench_torch_mps_f32.py::measure`）が
    /// 入力をループ外でデバイス転送し、ループ内は `torch.mm` +
    /// `torch.mps.synchronize()` のみを計測するのと同一の同期境界に揃える
    /// ため。readback は本ベンチの出力には不要なため計測対象に含めない。
    /// 中央値・Q1・Q3（秒）を TFLOPS へ変換した 3 つ組。`docs/performance-targets.md`
    /// §4 が中央値に加え Q1/Q3 の記録を必須とする（REQ-8）ため、`measure` は
    /// `Measurement` の 3 フィールドすべてを保持したまま呼び出し元へ返す
    /// （codex-review #700 P1 指摘: 従来は `median_secs` のみを変換し
    /// `q1_secs`/`q3_secs` を破棄していた）。時間が短いほど TFLOPS が高いため、
    /// 秒の昇順（`q1_secs` <= `median_secs` <= `q3_secs`）をそのまま TFLOPS へ
    /// 変換すると大小関係が反転する。本構造体の `q1`/`q3` は変換元の秒を
    /// 入れ替えて算出することでこの反転を打ち消し（`q1` は `q3_secs` から、
    /// `q3` は `q1_secs` から）、TFLOPS 側でも昇順（q1_tflops <= median_tflops
    /// <= q3_tflops）を保つ（codex-review #700 P1 指摘の分位点入れ替え・
    /// コメント誤記修正の双方に対応）。
    struct TflopsQuartiles {
        median: f64,
        q1: f64,
        q3: f64,
    }

    fn measure(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        config: &MeasurementConfig,
    ) -> (TflopsQuartiles, tile::TileConfig) {
        let mut rng = Xorshift64Star::new(SEED);
        let a: Vec<f32> = rng.fill_vec(m * k);
        let b: Vec<f32> = rng.fill_vec(k * n);

        let cfg = tile::select(m, n, k);

        let (m_eff, n_eff, k_eff) = (pad8(m), pad8(n), pad8(k));
        let a_padded = pad_matrix(&a, m, k, m_eff, k_eff);
        let b_padded = pad_matrix(&b, k, n, k_eff, n_eff);

        let a_buf = MetalBuffer::new_with_data(ctx, &a_padded)
            .expect("A バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");
        let b_buf = MetalBuffer::new_with_data(ctx, &b_padded)
            .expect("B バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");
        let c_buf = MetalBuffer::new_zeroed(ctx, m_eff * n_eff)
            .expect("C バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");

        // 採用構成（フォールバック解決後）は計測ループの外で 1 回だけ確定
        // させる（`pipeline_for_tile` はキャッシュ済み構成に対しては軽量な
        // 参照返しのため、計測ループ内で毎回呼んでも計測対象の重い処理
        // ―エンコード＋コマンドバッファ完了待ち―には影響しないが、
        // ラベル取得はループ外で十分）。
        let resolved_cfg = gemm
            .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff, cfg)
            .expect("Metal f32 SimdgroupTiled GEMM ウォームアップディスパッチに失敗した（実機でのみ実行する前提）");

        let measurement = bench_run(config, || {
            gemm.dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff, cfg)
                .expect("Metal f32 SimdgroupTiled GEMM ディスパッチに失敗した（実機でのみ実行する前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        // TFLOPS は実行時間の逆数のため、秒の昇順（q1_secs <= median_secs <=
        // q3_secs）は TFLOPS の降順に反転する。TFLOPS の Q1（下位＝低速側）は
        // 時間の Q3（q3_secs）から、TFLOPS の Q3（上位＝高速側）は時間の Q1
        // （q1_secs）から算出し、ラベルを TFLOPS 側で昇順に保つ（Bugbot #231
        // の gemm_bench.rs / gemm_blis_perf.rs と同一対応）。
        let quartiles = TflopsQuartiles {
            median: tflops(m, n, k, measurement.median_secs),
            q1: tflops(m, n, k, measurement.q3_secs),
            q3: tflops(m, n, k, measurement.q1_secs),
        };
        (quartiles, resolved_cfg)
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        // REQ-8 の主指標は 2048/4096（512 は起動オーバーヘッド支配のため
        // 参考値。PoC-v2-4 先例・`gemm_f16_bench.rs` と同一形状帯）。
        for size in [512usize, 1024, 2048, 4096] {
            let config = MeasurementConfig::default();
            let (q, resolved_cfg) = measure(&gemm, &ctx, size, size, size, &config);
            println!(
                "size={size} metal_f32_simdgroup_tiled_tflops={:.4} q1_tflops={:.4} q3_tflops={:.4} resolved_tile_config={resolved_cfg:?}",
                q.median, q.q1, q.q3
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け stub（`gemm_bench.rs`・`gemm_f16_bench.rs` と同じ
/// 位置づけ。`objc2` 系は `cfg(target_os = "macos")` 限定のため本クレートの
/// GEMM 実装自体がコンパイル対象外になる。Linux CI の
/// `cargo build --workspace --all-targets`／`cargo clippy --all-targets`
/// をこの example も含めて通すための最小 main）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_f32_prepared_bench example requires macOS (Apple Silicon). \
         run it on macOS hardware: cargo run -p backend-metal --example gemm_f32_prepared_bench --release"
    );
}
