//! Metal f16 GEMM（`gemm_simdgroup_f16`）実測バイナリ（TASK-8.3b・#156）。
//!
//! REQ-8 性能下限表「Metal f16 対 PyTorch MPS f16」（`docs/spec/04-requirements.md:183`）
//! の唯一の未実測行に対応する。計測コアは `bench-harness::protocol::run`
//! （warmup 20 回以上・計測 20 回以上・中央値/Q1/Q3。TASK-8.1）を使い、
//! `crates/backend-metal/examples/gemm_bench.rs`（f32 4 段計測）と同じ
//! 計測コア・シードを再利用する（`.claude/rules/coding-rust.md`「ベンチは
//! 5 回計測の中央値を採用」を満たす）。
//!
//! 相対値の分母は `scripts/bench/gemm_bench_torch_mps_f16.py`（同一実機上で
//! 別プロセスとして計測する PyTorch MPS f16 ベースライン）とし、本バイナリ
//! 自体は Rust 側 `gemm_simdgroup_f16` の TFLOPS のみを出力する（REQ-8 v2
//! 「同一ハードウェア上の PyTorch とのみ比較」方針。実測記録は
//! `docs/perf/metal-f16-vs-mps-f16.md` へ転記する）。
//!
//! 計測区間は PyTorch 側（`gemm_bench_torch_mps_f16.py::measure` の
//! `torch.matmul` + `torch.mps.synchronize()` のみ。入力は測定ループの外で
//! デバイスへ転送済み）と揃え、パディング・バッファ確保／アップロード・
//! readback／アンパディングは計測ループの外（ウォームアップ側）で行う。
//! `MetalGemm::dispatch_f16_prepared`（`dispatch_f16` からエンコード＋
//! コマンドバッファ完了待ちのみを切り出した入口）を使う（PR #346 Bugbot
//! 指摘 2: 計測区間の不一致修正）。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_bench.rs` と
//! 同一（self-hosted runner をベンチ実行で占有しない・Linux CI でも
//! ビルド検証のみ通す）。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_f16_bench --release
//! ```
//!
//! 実行前に数値一致（`cpu_metal_f16_parity.rs`）を確認することを推奨する:
//!
//! ```sh
//! cargo test -p backend-metal --release -- --ignored --nocapture cpu_metal_f16_parity
//! ```

#[cfg(target_os = "macos")]
mod macos_impl {
    use backend_metal::pad::{pad_matrix_f16, pad8};
    use backend_metal::{MetalContext, MetalGemm, MetalHalfBuffer};
    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, run as bench_run};
    use half::f16;

    /// 決定的シード（`gemm_bench.rs::SEED` と同一値。PoC-v2 系・既存 bench
    /// と同じ入力分布に揃える。実装計画 §3.3）。
    const SEED: u64 = 0xC0FFEE;

    fn tflops(m: usize, n: usize, k: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        flops / median_secs / 1e12
    }

    /// `m×n×k` の f16 GEMM（`MetalGemm::dispatch_f16_prepared`）を計測し、
    /// 中央値 TFLOPS を返す。`gemm_bench.rs::measure_shape` の f16 版
    /// （ディスパッチ入口が異なるため独立実装）。
    ///
    /// パディング・バッファ確保／アップロードは計測ループの外で 1 回だけ
    /// 行い、計測対象はディスパッチ（エンコード＋コマンドバッファ完了待ち）
    /// のみとする。PyTorch 側（`gemm_bench_torch_mps_f16.py::measure`）が
    /// 入力をループ外でデバイス転送し、ループ内は `matmul` +
    /// `torch.mps.synchronize()` のみを計測するのと同一の同期境界に揃える
    /// ため（PR #346 Bugbot 指摘 2）。readback／アンパディングも本ベンチの
    /// 出力には不要なため計測対象に含めない。
    fn measure(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        config: &MeasurementConfig,
    ) -> f64 {
        let mut rng = Xorshift64Star::new(SEED);
        let a: Vec<f16> = rng.fill_vec_f16(m * k);
        let b: Vec<f16> = rng.fill_vec_f16(k * n);

        let (m_eff, n_eff, k_eff) = (pad8(m), pad8(n), pad8(k));
        let a_padded = pad_matrix_f16(&a, m, k, m_eff, k_eff);
        let b_padded = pad_matrix_f16(&b, k, n, k_eff, n_eff);

        let a_buf = MetalHalfBuffer::new_with_data(ctx, &a_padded)
            .expect("A バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");
        let b_buf = MetalHalfBuffer::new_with_data(ctx, &b_padded)
            .expect("B バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");
        let c_buf = MetalHalfBuffer::new_zeroed(ctx, m_eff * n_eff)
            .expect("C バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");

        let measurement = bench_run(config, || {
            gemm.dispatch_f16_prepared(ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff)
                .expect("Metal f16 GEMM ディスパッチに失敗した（実機でのみ実行する前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        tflops(m, n, k, measurement.median_secs)
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 含む）");

        // REQ-8 の主指標は 2048/4096（512 は起動オーバーヘッド支配のため
        // 参考値。PoC-v2-4 先例。実装計画 §3.3）。
        for size in [512usize, 1024, 2048, 4096] {
            let config = MeasurementConfig::default();
            let f16_tflops = measure(&gemm, &ctx, size, size, size, &config);
            println!("size={size} metal_f16_simdgroup_tflops={f16_tflops:.4}");
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け stub（`gemm_bench.rs` と同じ位置づけ。`objc2` 系は
/// `cfg(target_os = "macos")` 限定のため本クレートの GEMM 実装自体が
/// コンパイル対象外になる。Linux CI の `cargo build --workspace --all-targets`／
/// `cargo clippy --all-targets` をこの example も含めて通すための最小 main）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_f16_bench example requires macOS (Apple Silicon). \
         run it on macOS hardware: cargo run -p backend-metal --example gemm_f16_bench --release"
    );
}
