//! `tile::select` の形状帯域・転置パターン（NN/NT/TN/TT）別スイープ
//! example（イシュー #1039。親 #1037・兄弟 #1036/#1038/#1040 の後続）。
//!
//! - NN: `MetalGemm::dispatch_tiled_prepared`（転送非計測）で
//!   `crate::tile::CANDIDATES` 全 8 候補を明示指定して比較する
//!   （`gemm_tile_sweep.rs` と同じ計測境界・`resolved_matches_requested`
//!   検証を踏襲）。
//! - NT/TN/TT: シェーダ側にタイル variant を持たない strided classic
//!   tiled 経路（`MetalGemm::dispatch_strided_bias_act_prepared`。イシュー
//!   #1040）のみを計測する（bias なし・act なし）。NN 最良候補との差分を
//!   `docs/perf/metal-gemm-tile-table.md` へ記録し、`gemm_simdgroup_tiled`
//!   への転置ロード拡張（#1037 系へ引き継ぎ済み・本イシューはスコープ外）
//!   の要否判断材料にする。
//!
//! 形状セット（`tile::select_with_occupancy` の分岐クラスを代表する点）:
//! - 正方立方（`m==n==k`）: 512/1024/2048/4096（#744 実測点の再現確認）
//! - K 未実測の正方出力（`m==n`・`k!=m`）: (2048,2048,64)・(2048,2048,512)
//! - 準正方長方形（縦横比 < 2・`m!=n`）: (1536,1024,1024)・(1024,1536,1536)
//! - 縦長・横長（縦横比 >= 2）: (4096,1024,1024)・(1024,4096,1024)
//!
//! 計測プロトコルは `gemm_tile_sweep.rs` と同一（`bench_harness::protocol::run`
//! と `MeasurementConfig::default()`〈warmup 20・計測 20・中央値〉・決定的
//! シード `0xC0FFEE`。`docs/perf/metal-bench-noise-protocol.md` 準拠）。
//!
//! 出力形式:
//! `shape=(m,n,k) pattern=<NN|NT|TN|TT> candidate=<label> tflops=<中央値> resolved_matches_requested=<bool>`
//! （NT/TN/TT は `candidate=strided_classic_tiled`・
//! `resolved_matches_requested=true` 固定でカーネル構成の有無を揃える）。
//!
//! `examples/` に置く理由・非 macOS stub の位置づけは `gemm_tile_sweep.rs`
//! と同一（Linux CI はビルド検証のみ・self-hosted runner を占有しない）。
//!
//! ## 実機実行手順（macOS・Apple Silicon）
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-metal --example gemm_transpose_tile_sweep --release
//! ```

#[cfg(target_os = "macos")]
mod macos_impl {
    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, run as bench_run};
    use fandhe_ai_backend_metal::layout::{MatrixLayout, classify_2d};
    use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm, TileConfig};

    /// 決定的シード（`gemm_tile_sweep.rs`・`gemm_bench.rs` と同一値）。
    const SEED: u64 = 0xC0FFEE;

    fn tflops(m: usize, n: usize, k: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        flops / median_secs / 1e12
    }

    /// `crate::tile::CANDIDATES`（`pub(crate)`。examples/ は別コンパイル
    /// 単位のため参照不可）の値をラベル付きで複製する（`gemm_tile_sweep.rs`
    /// と同じ複製方式の判断）。配列順・値は `tile.rs` の `CANDIDATES` 定義
    /// と一致させる。`tile.rs` 側が変わった場合は本 example 側も追従が
    /// 必要。
    fn candidates() -> [(&'static str, TileConfig); 9] {
        [
            (
                "cand0_64x64_wm2wn2",
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
                "cand1_64x32_wm2wn2_tall",
                TileConfig {
                    bm: 64,
                    bn: 32,
                    bk: 16,
                    wm: 2,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "cand2_32x64_wm2wn2_wide",
                TileConfig {
                    bm: 32,
                    bn: 64,
                    bk: 16,
                    wm: 2,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "cand3_32x32_wm2wn2",
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
                "cand4_64x64_wm1wn2",
                TileConfig {
                    bm: 64,
                    bn: 64,
                    bk: 16,
                    wm: 1,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "cand5_64x32x32_wm2wn2",
                TileConfig {
                    bm: 64,
                    bn: 32,
                    bk: 32,
                    wm: 2,
                    wn: 2,
                    staged: true,
                },
            ),
            (
                "cand6_64x32x8_wm4wn1",
                TileConfig {
                    bm: 64,
                    bn: 32,
                    bk: 8,
                    wm: 4,
                    wn: 1,
                    staged: true,
                },
            ),
            (
                "cand7_single_simdgroup_8x8",
                TileConfig {
                    bm: 8,
                    bn: 8,
                    bk: 8,
                    wm: 1,
                    wn: 1,
                    staged: false,
                },
            ),
            // MLX steel classic 経路の未収録構成（イシュー #1143）:
            // `cand2_32x64_wm2wn2_wide` の 4 simdgroup 分担を wm1wn2（2
            // simdgroup）へ落とし、simdgroup あたりの acc タイル
            // （8x8 の 2x2=4 個ではなく 4x8＝acc_rows=4,acc_cols=8）を
            // 変える構成。`tile.rs::CANDIDATES` の index 8（末尾追加。
            // 既存 index 0〜7 は不変）に対応する。
            (
                "cand8_32x64x16_wm1wn2",
                TileConfig {
                    bm: 32,
                    bn: 64,
                    bk: 16,
                    wm: 1,
                    wn: 2,
                    staged: true,
                },
            ),
        ]
    }

    /// NN 経路: `CANDIDATES` 全候補を `dispatch_tiled_prepared` で明示指定
    /// して比較する（`gemm_tile_sweep.rs::measure_tiled_prepared` と同一の
    /// 計測境界。バッファはループ外で 1 回だけ確保・アップロードし、計測
    /// 対象はディスパッチのみに絞る）。
    fn measure_nn(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        config: &MeasurementConfig,
    ) {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(m * k);
        let b = rng.fill_vec(k * n);

        let a_buf = MetalBuffer::new_with_data(ctx, &a)
            .expect("A バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let b_buf = MetalBuffer::new_with_data(ctx, &b)
            .expect("B バッファのアップロードに失敗した（実機でのみ実行する前提）");
        let c_buf = MetalBuffer::new_zeroed(ctx, m * n)
            .expect("C バッファの確保に失敗した（実機でのみ実行する前提）");

        for (label, cfg) in candidates() {
            // 非タイル倍数境界（bm/bn/bk 非整除）の形状では `pipeline_for_tile`
            // が構成検証で拒否しうるため、候補ごとに事前検証して失敗時は
            // skip する（`TileConfig::validate` 相当の fail-closed。全候補が
            // 常に全形状で妥当とは限らない。#1038 で確立済みの前提）。
            let resolved_cfg = match gemm
                .dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, m, n, k, cfg)
            {
                Ok(resolved) => resolved,
                Err(e) => {
                    println!("shape=({m},{n},{k}) pattern=NN candidate={label} skipped reason={e}");
                    continue;
                }
            };

            let measurement = bench_run(config, || {
                gemm.dispatch_tiled_prepared(ctx, &a_buf, &b_buf, &c_buf, m, n, k, cfg)
                    .expect("直前に成功した構成が計測ループ中に失敗することはない想定");
            })
            .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

            println!(
                "shape=({m},{n},{k}) pattern=NN candidate={label} tflops={:.4} resolved_matches_requested={}",
                tflops(m, n, k, measurement.median_secs),
                resolved_cfg == cfg,
            );
        }
    }

    /// 転置パターン（NT/TN/TT）1 種について、strided classic tiled 経路
    /// （タイル variant なし。#1040 確定構成）を計測する。物理バッファは
    /// `tests/gemm_strided_parity.rs::transpose_dense` と同じ方式（転置
    /// 対象は列優先物理レイアウトを CPU 側で明示構築）で用意する。
    ///
    /// `#[allow(clippy::too_many_arguments)]`: `crate::gemm::MetalGemm::
    /// dispatch_strided_bias_act_prepared` 自体が個別引数方式（構造体へ
    /// まとめ込まない設計判断。同関数の doc comment 参照）のため、その
    /// 計測ラッパーである本関数も同じ形状の引数列をそのまま持つ。
    #[allow(clippy::too_many_arguments)]
    fn measure_transposed(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        trans_a: bool,
        trans_b: bool,
        pattern_label: &str,
        config: &MeasurementConfig,
    ) {
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

        gemm.dispatch_strided_bias_act_prepared(
            ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, None, false, &c_buf, m, n, k,
        )
        .expect("dispatch_strided_bias_act_prepared に失敗した（実機でのみ実行する前提）");

        let measurement = bench_run(config, || {
            gemm.dispatch_strided_bias_act_prepared(
                ctx, &a_buf, 0, a_layout, &b_buf, 0, b_layout, None, false, &c_buf, m, n, k,
            )
            .expect("直前に成功した構成が計測ループ中に失敗することはない想定");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        println!(
            "shape=({m},{n},{k}) pattern={pattern_label} candidate=strided_classic_tiled tflops={:.4} resolved_matches_requested=true",
            tflops(m, n, k, measurement.median_secs),
        );
    }

    /// `logical`（行優先の論理 `[rows, cols]`）から `[cols, rows]` 行優先の
    /// 転置済み物理バッファを作る（`tests/gemm_strided_parity.rs` と同一
    /// ロジックの複製。両ファイルとも別コンパイル単位のため共有できない）。
    fn transpose_dense(logical: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = logical[r * cols + c];
            }
        }
        out
    }

    /// スイープ対象形状（計画§4「形状セット」）。
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

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
        let config = MeasurementConfig::default();

        for (m, n, k) in shapes() {
            measure_nn(&gemm, &ctx, m, n, k, &config);
            for (trans_a, trans_b, label) in
                [(false, true, "NT"), (true, false, "TN"), (true, true, "TT")]
            {
                measure_transposed(&gemm, &ctx, m, n, k, trans_a, trans_b, label, &config);
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け stub（`gemm_tile_sweep.rs` と同一の位置づけ。`objc2`
/// 系は `cfg(target_os = "macos")` 限定のため本クレートの GEMM 実装自体が
/// コンパイル対象外になる。Linux CI の `cargo build --workspace
/// --all-targets`／`cargo clippy --all-targets` をこの example も含めて
/// 通すための最小 main）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_transpose_tile_sweep example requires macOS (Apple Silicon). \
         run it on macOS hardware: cargo run -p fandhe-ai-backend-metal --example gemm_transpose_tile_sweep --release"
    );
}
