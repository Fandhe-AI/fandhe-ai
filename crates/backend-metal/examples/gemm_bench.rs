//! Metal GEMM 3 段（naive/tiled/simdgroup）性能実測バイナリ（TASK-1.8c・#40）。
//!
//! 受け入れ条件「simdgroup 版が naive 比で PoC-v2-4 相当の性能を示す」
//! （PoC-v2-4 実測: Apple M4 Max・size=4096 で naive 1.271 → tiled 2.123 →
//! simdgroup 3.134 TFLOPS、simdgroup/naive ≒ ×2.47）の実測手段。
//! 計測コアは `bench-harness::protocol::run`（warmup 20 回以上・計測
//! 20 回以上・中央値/Q1/Q3。TASK-8.1）を使い、`.claude/rules/coding-rust.md`
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
//! size=512/2048/4096 で naive/tiled/simdgroup を計測し、simdgroup/naive
//! 比を出力する。非 macOS 環境（本実装環境を含む）では `main` が説明を
//! 表示するだけの stub になり、`cargo build`／`cargo clippy --all-targets`
//! はビルド対象として通る（Linux CI でもコンパイル検証できるようにする
//! ため）。

#[cfg(target_os = "macos")]
mod macos_impl {
    use backend_metal::{GemmVariant, MetalContext, MetalGemm};
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

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        for size in [512usize, 2048, 4096] {
            // naive は大サイズで所要時間が過大になりやすいが、
            // `MeasurementConfig` の下限（warmup・計測とも 20 回）を
            // `MeasurementConfig::default()` がすでに満たしているため、
            // これ以上計測回数を減らす調整はしていない（「5 回計測の
            // 中央値」下限〈coding-rust.md〉は満たしたまま）。
            let config = MeasurementConfig::default();

            let naive = measure(&gemm, &ctx, GemmVariant::Naive, size, &config);
            let tiled = measure(&gemm, &ctx, GemmVariant::Tiled, size, &config);
            let simdgroup = measure(&gemm, &ctx, GemmVariant::Simdgroup, size, &config);

            println!(
                "size={size} naive_tflops={naive:.4} tiled_tflops={tiled:.4} simdgroup_tflops={simdgroup:.4} simdgroup_over_naive={:.4}",
                simdgroup / naive
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
