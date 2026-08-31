//! GPU counters（`xcrun xctrace record --instrument 'Metal GPU Counters'`）
//! 採取専用の size 固定・反復ディスパッチワークロード（イシュー #1103）。
//!
//! `docs/perf/metal-gemm-bottleneck-rediagnosis.md` §5.1 は「Metal System
//! Trace」テンプレート既定でカウンタプロファイルが無効（`counter-profile=0`）
//! だったため GPU counters（占有率・ALU/メモリ limiter 内訳）が未計測のまま
//! 残った。同 §5.1 はさらに、計測対象（`gemm_f32_prepared_bench`）が
//! 512〜4096 を連続実行するため 1 トレースに複数 size の区間が混在し、
//! `<row>` の `ref` 属性解決パーサ（未実装。同 doc §8 スコープ外）なしでは
//! size 別に紐付けられない、とも指摘している。
//!
//! 本 example はこの 2 点目を解消するための size 固定ワークロードで、
//! `--size=N`（1 size のみ）を反復（`--iters`）ディスパッチし続けるだけの
//! 最小構成にする。1 プロセス実行 = 1 trace = 1 size という対応にすることで、
//! xctrace 側のトレースをまたいだ紐付けが不要になる（`--xpath` 抽出結果を
//! そのままその size の値として扱える）。
//!
//! `dispatch_tiled_prepared`（`gemm_f32_prepared_bench.rs` と同一入口。
//! バッファのアップロードはループ外で 1 回のみ・計測対象はディスパッチ
//! のみという計測境界を踏襲）を呼ぶ。`shaders/gemm.metal`・
//! `crates/backend-metal/src/` は変更しない（診断タスクの原則。
//! `docs/perf/metal-gemm-bottleneck-rediagnosis.md` 冒頭コメント）。
//!
//! `bench_harness::MeasurementConfig` 系の統計（中央値/Q1/Q3）はここでは
//! 出力しない（xctrace が壁時計・GPU 実行区間を別途記録するため、本
//! ワークロード自身の役割はディスパッチを継続して発生させることに限る）。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo build -p fandhe-ai-backend-metal --example gemm_counter_workload --release
//! xcrun xctrace record --template 'Metal System Trace' \
//!   --instrument 'Metal GPU Counters' \
//!   --output /tmp/gemm_counters_1024.trace --launch -- \
//!   ./target/release/examples/gemm_counter_workload --size=1024 --iters=200
//! ```
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_f32_prepared_bench.rs`
//! と同一（`objc2` 系は `cfg(target_os = "macos")` 限定のため本クレートの
//! GEMM 実装自体がコンパイル対象外になる。Linux CI のビルド検証のみ通す）。

#[cfg(target_os = "macos")]
mod macos_impl {
    use bench_harness::rng::Xorshift64Star;
    use fandhe_ai_backend_metal::pad::{pad_matrix, pad8};
    use fandhe_ai_backend_metal::tile;
    use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm};

    /// 決定的シード（`gemm_f32_prepared_bench.rs::SEED` と同一値。
    /// 診断ワークロードの入力分布を既存 bench と揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// `--size`・`--iters` が取りうる範囲。`--size` は 1 辺の要素数で、
    /// `size * size` の `f32` 正方行列を A・B・C（パディング後）の 3 枚
    /// 確保するため、`fixed_overhead_diagnosis.rs::MAX_MATRIX_BYTES`
    /// （1 枚あたり 1 GiB）と同一の上限で fail-closed 検証する（OWASP A03
    /// 「外部入力の検証」観点。診断用途で現実的に必要なサイズ
    /// 〈本 doc の対象 1024/2048/4096〉を十分に超える一方、誤指定による
    /// OOM を防ぐ）。
    const MAX_MATRIX_BYTES: usize = 1 << 30;

    /// `--iters` の下限。0 反復は「ワークロードを何も発生させない」ため
    /// xctrace 採取の目的（GPU 実行区間の継続的な発生）を満たさず、
    /// 誤指定として拒否する。
    const MIN_ITERS: usize = 1;

    /// 解析済みの CLI 引数（[`parse_args`] の戻り値）。
    struct CliArgs {
        size: usize,
        iters: usize,
    }

    /// `--size=<N>` の値部分を解析する（許可リスト方式: 正の整数のみを
    /// 受理し、それ以外は `Err` で拒否する。`n * n` のオーバーフロー・
    /// 過大メモリ確保は `MAX_MATRIX_BYTES` 上限まで検証する。
    /// `fixed_overhead_diagnosis.rs::parse_sizes_value` と同型の検証だが、
    /// 本 example は size を 1 個のみ受け取る（1 プロセス = 1 trace = 1 size
    /// という対応を保つため、複数値は受理しない）。
    fn parse_size_value(v: &str) -> Result<usize, String> {
        let n: usize = v
            .trim()
            .parse()
            .map_err(|_| format!("--size の値が不正: '{v}'"))?;
        if n == 0 {
            return Err("--size は正数である必要がある".to_string());
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
        Ok(n)
    }

    /// `--iters=<N>` の値部分を解析する（[`MIN_ITERS`] 以上の正の整数のみ
    /// 受理）。
    fn parse_iters_value(v: &str) -> Result<usize, String> {
        let n: usize = v
            .trim()
            .parse()
            .map_err(|_| format!("--iters の値が不正: '{v}'"))?;
        if n < MIN_ITERS {
            return Err(format!(
                "--iters は {MIN_ITERS} 以上である必要がある: '{n}'"
            ));
        }
        Ok(n)
    }

    /// `std::env::args()` を一度だけ走査して `--size=<N>`・`--iters=<N>` を
    /// 解析する（`fixed_overhead_diagnosis.rs::parse_args` と同型の一括走査
    /// ＋重複指定・未知引数の fail-closed 拒否。OWASP A03 観点）。
    /// 未指定時の既定値: `--size` は 4096（本診断の対象サイズのうち最大）・
    /// `--iters` は 200（`docs/perf/metal-gemm-bottleneck-rediagnosis.md`
    /// §3.2 の `gemm_diagnosis` example と同一の反復数で、xctrace の採取窓
    /// を十分に埋める）。
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
                    "未知の引数: '{arg}'（許可される引数は --size=<N> と --iters=<N> のみ）"
                ));
            }
        }
        let size = match size_raw {
            Some(v) => parse_size_value(&v)?,
            None => 4096,
        };
        let iters = match iters_raw {
            Some(v) => parse_iters_value(&v)?,
            None => 200,
        };
        Ok(CliArgs { size, iters })
    }

    pub fn main() {
        let args = match parse_args() {
            Ok(a) => a,
            Err(e) => {
                eprintln!("引数解析エラー: {e}");
                std::process::exit(1);
            }
        };

        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        let (m, n, k) = (args.size, args.size, args.size);
        let mut rng = Xorshift64Star::new(SEED);
        let a: Vec<f32> = rng.fill_vec(m * k);
        let b: Vec<f32> = rng.fill_vec(k * n);

        let cfg = tile::select(m, n, k);
        let (m_eff, n_eff, k_eff) = (pad8(m), pad8(n), pad8(k));
        let a_padded = pad_matrix(&a, m, k, m_eff, k_eff);
        let b_padded = pad_matrix(&b, k, n, k_eff, n_eff);

        // バッファのアップロードはループ外で 1 回のみ（`gemm_f32_prepared_bench.rs`
        // と同一の計測境界。xctrace で観測したいのはディスパッチ〈エンコード＋
        // コマンドバッファ完了待ち〉の反復であり、転送は対象外）。
        let a_buf = MetalBuffer::new_with_data(&ctx, &a_padded)
            .expect("A バッファ確保（ループ外の事前準備）に失敗した（実機でのみ実行する前提）");
        let b_buf = MetalBuffer::new_with_data(&ctx, &b_padded)
            .expect("B バッファ確保（ループ外の事前準備）に失敗した（実機でのみ実行する前提）");
        let c_buf = MetalBuffer::new_zeroed(&ctx, m_eff * n_eff)
            .expect("C バッファ確保（ループ外の事前準備）に失敗した（実機でのみ実行する前提）");

        // ウォームアップディスパッチ（採用構成の確定・初回コンパイル等の
        // 固定費を、xctrace が採取する反復区間の外へ出す）。
        let resolved_cfg = gemm
            .dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff, cfg)
            .expect(
                "Metal f32 SimdgroupTiled GEMM ウォームアップディスパッチに失敗した（実機でのみ実行する前提）",
            );

        eprintln!(
            "gemm_counter_workload: size={} iters={} resolved_tile_config={:?}",
            args.size, args.iters, resolved_cfg
        );

        for _ in 0..args.iters {
            gemm.dispatch_tiled_prepared(&ctx, &a_buf, &b_buf, &c_buf, m_eff, n_eff, k_eff, cfg)
                .expect(
                    "Metal f32 SimdgroupTiled GEMM ディスパッチに失敗した（実機でのみ実行する前提）",
                );
        }

        eprintln!(
            "gemm_counter_workload: 完了（{} 回ディスパッチ）",
            args.iters
        );
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け stub（`gemm_f32_prepared_bench.rs` と同じ位置づけ。
/// `objc2` 系は `cfg(target_os = "macos")` 限定のため本クレートの GEMM
/// 実装自体がコンパイル対象外になる。Linux CI の
/// `cargo build --workspace --all-targets`／`cargo clippy --all-targets`
/// をこの example も含めて通すための最小 main）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_counter_workload example requires macOS (Apple Silicon). \
         run it on macOS hardware: cargo run -p fandhe-ai-backend-metal --example gemm_counter_workload --release -- --size=4096 --iters=200"
    );
}
