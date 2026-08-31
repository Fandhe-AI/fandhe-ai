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
//! `MetalGemm::dispatch_f16_prepared_unverified`（`dispatch_f16_unverified`
//! からエンコード＋コマンドバッファ完了待ちのみを切り出した入口）を使う
//! （PR #346 Bugbot 指摘 2: 計測区間の不一致修正）。`_unverified` suffix・
//! 精度未検証の理由は `crates/backend-metal/src/gemm.rs::
//! MetalGemm::dispatch_f16_unverified` のドキュメントコメント（PR #346
//! codex-review P1-2 指摘）を参照。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_bench.rs` と
//! 同一（self-hosted runner をベンチ実行で占有しない・Linux CI でも
//! ビルド検証のみ通す）。
//!
//! ## タイル化後経路の追加計測（イシュー #799）
//!
//! #796〜#798 で `gemm_simdgroup_f16`（非タイル・1 threadgroup 1 simdgroup
//! 8x8）を `gemm_simdgroup_tiled_f16`（タイル化・BM/BN/BK/WM/WN・ベクトル化
//! ロード込み）へ世代更新した。本ファイルは旧経路（`metal_f16_simdgroup_tflops=`
//! 行。回帰基線として維持）に加え、新経路を `tile::select(m, n, k)` が選ぶ
//! `TileConfig` で計測する `metal_f16_tiled_tflops=` 行を並記出力する。
//! `MetalGemm::dispatch_f16_tiled_prepared_unverified` は `pipeline_for_tile_f16`
//! がデバイス上限超過等でサイレントにフォールバックしうる（`gemm.rs`
//! ドキュメントコメント参照）ため、戻り値の resolved `TileConfig` を
//! `tile=` として併記し、実際に採用された構成を透明化する。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_f16_bench --release
//! ```
//!
//! 実行前に数値一致（`cpu_metal_f16_parity.rs`・タイル化後経路は
//! `cpu_metal_f16_tiled_parity.rs`・`gemm_f16_auto_parity.rs`）を確認する
//! ことを推奨する（イシュー #799 実装計画 §4「実機で計測可能な場合のみ」）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture cpu_metal_f16_parity
//! cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture cpu_metal_f16_tiled_parity
//! cargo test -p fandhe-ai-backend-metal --release -- --ignored --nocapture gemm_f16_auto_parity
//! ```

#[cfg(target_os = "macos")]
mod macos_impl {
    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, run as bench_run};
    use fandhe_ai_backend_metal::pad::{pad_matrix_f16, pad8};
    use fandhe_ai_backend_metal::tile::{self, TileConfig};
    use fandhe_ai_backend_metal::{MetalContext, MetalGemm, MetalHalfBuffer};
    use half::f16;

    /// 決定的シード（`gemm_bench.rs::SEED` と同一値。PoC-v2 系・既存 bench
    /// と同じ入力分布に揃える。実装計画 §3.3）。
    const SEED: u64 = 0xC0FFEE;

    fn tflops(m: usize, n: usize, k: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        flops / median_secs / 1e12
    }

    /// `m×n×k` の f16 GEMM（`MetalGemm::dispatch_f16_prepared_unverified`）を計測し、
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
    /// 中央値・Q1・Q3（秒）を TFLOPS へ変換した 3 つ組。`docs/performance-targets.md`
    /// §4 が中央値に加え Q1/Q3 の記録を必須とする（REQ-8）ため、`measure` は
    /// `Measurement` の 3 フィールドすべてを保持したまま呼び出し元へ返す
    /// （codex-review #700 P1 指摘の f32 側修正に揃え、f16 側にも同様に適用。
    /// `gemm_f32_prepared_bench.rs::TflopsQuartiles` と同型・独立定義）。時間が
    /// 短いほど TFLOPS が高いため、秒の昇順（`q1_secs` <= `median_secs` <=
    /// `q3_secs`）をそのまま TFLOPS へ変換すると大小関係が反転する。本構造体の
    /// `q1`/`q3` は変換元の秒を入れ替えて算出することでこの反転を打ち消し
    /// （`q1` は `q3_secs` から、`q3` は `q1_secs` から）、TFLOPS 側でも昇順
    /// （q1_tflops <= median_tflops <= q3_tflops）を保つ（codex-review #700
    /// P1 指摘の分位点入れ替え・コメント誤記修正の双方に対応）。
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
    ) -> TflopsQuartiles {
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
            gemm.dispatch_f16_prepared_unverified(ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff)
                .expect("Metal f16 GEMM ディスパッチに失敗した（実機でのみ実行する前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        // TFLOPS の Q1/Q3 は時間の Q3/Q1 から算出する（上のドキュメンテーション
        // コメント参照。Bugbot #231 の gemm_bench.rs / gemm_blis_perf.rs と同一対応）。
        TflopsQuartiles {
            median: tflops(m, n, k, measurement.median_secs),
            q1: tflops(m, n, k, measurement.q3_secs),
            q3: tflops(m, n, k, measurement.q1_secs),
        }
    }

    /// `m×n×k` の f16 GEMM をタイル化後経路
    /// （`MetalGemm::dispatch_f16_tiled_prepared_unverified`）で計測し、
    /// 中央値 TFLOPS と実際に採用された `TileConfig`（resolved。
    /// フォールバック透明性）を返す（イシュー #799）。`measure`（非タイル
    /// 経路）と同一の計測境界（パディング・バッファ確保／アップロードは
    /// 計測ループの外・計測対象はディスパッチのみ）・同一 `SEED`・同一
    /// 入力分布を用いる（新旧経路を同一条件で比較するため）。`cfg` は
    /// `tile::select(m, n, k)`（本番ディスパッチ `dispatch_auto` と同じ
    /// 選択関数。`tile.rs::select` ドキュメントコメント参照）が選ぶ。
    fn measure_tiled(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        config: &MeasurementConfig,
    ) -> (TflopsQuartiles, TileConfig) {
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

        let cfg = tile::select(m, n, k, ctx.occupancy_params().map(|p| p.gpu_core_count));
        // ループの外で一度 `dispatch_f16_tiled_prepared_unverified` を呼び、
        // resolved 構成の取得と `MetalGemm::tiled_f16_cache`（`gemm.rs`
        // `pipeline_for_tile_f16`）のウォームアップを行う。フォールバックが
        // 発生しなければ `cfg` と一致する。
        //
        // 計測ループ内でも同じ `dispatch_f16_tiled_prepared_unverified` を
        // 呼ぶため、`pipeline_for_tile_f16` のキャッシュヒット経路
        // （`borrow()` + `HashMap::get` + `Retained::clone`）は計測区間に
        // 残る。旧経路 `dispatch_f16_prepared_unverified` は事前構築済みの
        // `self.pipeline_simdgroup_f16` を直接使うためこのコストを含まず、
        // 新旧経路の計測区間には非対称なオーバーヘッド（キャッシュ照会 1 回
        // 分。フルディスパッチに比べ無視できる大きさと見込むが未検証）が
        // 残る。レビュー指摘（イシュー #799 PR review）を受けて明記する。
        // 真に対称な計測境界にするには新経路にも「呼び出し前にパイプライン
        // 確定済み」入口が必要（将来の改善候補。フォローアップ Issue で
        // 追跡）。
        let resolved = gemm
            .dispatch_f16_tiled_prepared_unverified(
                ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff, cfg,
            )
            .expect(
                "Metal f16 タイル化 GEMM ディスパッチ（resolved 構成取得用の事前 1 回）に失敗した",
            );

        let measurement = bench_run(config, || {
            gemm.dispatch_f16_tiled_prepared_unverified(
                ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff, cfg,
            )
            .expect("Metal f16 タイル化 GEMM ディスパッチに失敗した（実機でのみ実行する前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        (
            TflopsQuartiles {
                median: tflops(m, n, k, measurement.median_secs),
                q1: tflops(m, n, k, measurement.q3_secs),
                q3: tflops(m, n, k, measurement.q1_secs),
            },
            resolved,
        )
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した（f16 含む）");

        // REQ-8 の主指標は 2048/4096（512 は起動オーバーヘッド支配のため
        // 参考値。PoC-v2-4 先例。実装計画 §3.3）。
        for size in [512usize, 1024, 2048, 4096] {
            let config = MeasurementConfig::default();
            let q = measure(&gemm, &ctx, size, size, size, &config);
            println!(
                "size={size} metal_f16_simdgroup_tflops={:.4} q1_tflops={:.4} q3_tflops={:.4}",
                q.median, q.q1, q.q3
            );

            // タイル化後経路（イシュー #799）。旧経路行と並記し、
            // 対 MPS f16 比・改善幅を同一プロセス内で突合できるようにする。
            let (q_tiled, resolved) = measure_tiled(&gemm, &ctx, size, size, size, &config);
            println!(
                "size={size} metal_f16_tiled_tflops={:.4} q1_tflops={:.4} q3_tflops={:.4} \
                 tile={}x{}x{}_wm{}_wn{}_staged{}",
                q_tiled.median,
                q_tiled.q1,
                q_tiled.q3,
                resolved.bm,
                resolved.bn,
                resolved.bk,
                resolved.wm,
                resolved.wn,
                resolved.staged,
            );
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
         run it on macOS hardware: cargo run -p fandhe-ai-backend-metal --example gemm_f16_bench --release"
    );
}
