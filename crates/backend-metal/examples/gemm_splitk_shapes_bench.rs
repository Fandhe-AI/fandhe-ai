//! Metal GEMM の split-K ディスパッチ分岐設計検討（イシュー #810）向け、
//! K 支配的非正方形状（M・N が小さく K が大きい GEMM）の劣化定量化ベンチ。
//!
//! `crate::tile::select`（`crate::gemm::MetalGemm::dispatch_auto` が使う本番
//! タイル選択）は tall／wide／正方立方／大形状の 4 分岐のみを持ち、K 方向は
//! `TileConfig::bk`（K ループ刻み）にしか反映されない（`tile.rs:788-793`）。
//! すなわち 1 threadgroup が K 全域を直列にループする構造であり、M・N が小さく
//! threadgroup 数（`ceil(M/bm)*ceil(N/bn)`）が GPU コア数に対して不足する形状
//! では、K がいくら大きくても並列度が上がらない。MLX は同種の形状
//! （`steel_gemm_splitk_axpby` 選択条件。`docs/backend-metal-splitk-decision.md`
//! §1 に SHA 固定で記録）で split-K 専用カーネルへ分岐し、K 方向も
//! threadgroup 分割の対象にする。
//!
//! 本イシューは**設計検討（調査・計測・記録）であり、`dispatch_auto`・
//! シェーダの本番経路変更は行わない**（`docs/backend-metal-splitk-decision.md`
//! §2 に採用時の設計方針のみを記録する）。`crates/backend-metal/src/`・
//! `shaders/gemm.metal` は一切変更しない（#487・#549 と同じ「実装変更を伴わ
//! ない調査・計測・記録タスク」の型）。
//!
//! ## 実行方法
//!
//! 解析値（[`analytics`] モジュール: `tile::select` の選択結果・threadgroup
//! 数・K ループ回数・arithmetic intensity・MLX split-K 選択域への該当有無）は
//! `objc2` 系 FFI に触れない純粋関数のため macOS 以外でも算出できる:
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_splitk_shapes_bench --release
//! ```
//!
//! 壁時計計測（`wall_ms`・`tflops_lower_bound`）は macOS 実機限定
//! （`MetalGemm::dispatch_auto` を `bench_harness::protocol::run` で計測する
//! `gemm_diagnosis.rs` と同型のフォールバック経路。同ファイルの「実測部分の
//! 設計」節参照。`crate::pipeline::make_pipeline_with_constants` は
//! `pub(crate)` で example から呼べないため、既存公開 API 経由の
//! end-to-end 壁時計計測に留め、転送時間の分離は試みない）。
//!
//! ## 対象形状・対照形状
//!
//! `docs/perf/metal-gemm-splitk-shapes.md`「対象形状群」節参照。要旨:
//! - **対象（K 支配的非正方）**: `(M,N) ∈ {32, 64, 128, 256}`（`M == N`）
//!   × `K ∈ {2048, 4096, 8192}` の代表点
//! - **対照（同程度 FLOPs の正方立方形状）**: [`analytics::matched_cube_side`]
//!   で `2*M*N*K` に最も近い `2*S^3`（`S` は 8 の倍数）を与える `S` を算出し、
//!   `(S,S,S)` の正方立方 GEMM と比較する。TFLOPS は各形状自身の実 FLOPs で
//!   正規化する比率指標のため、`S` の丸めによる FLOPs のわずかな差異は
//!   比較の妥当性を損なわない（出力に両者の FLOPs 比も併記する）

/// 並列度〈threadgroup 数〉・K ループ回数・MLX split-K 選択域該当判定の解析値。
/// `objc2` 系 FFI に触れない純粋関数のため `cfg(target_os = "macos")` を
/// 付けず、Linux（本実装環境・CI）でも算出できる（`gemm_diagnosis.rs::analytics`
/// と同じ設計判断）。
mod analytics {
    use backend_metal::TileConfig;

    /// MLX `steel_gemm_splitk_axpby` の非 NAX 選択条件（Case 1）を
    /// `mlx/backend/metal/matmul.cpp:913-935`（`ml-explore/mlx` コミット
    /// `a082cb91d5908e9d89a61a31ee90ee45875b8a1e`。`gh api
    /// repos/ml-explore/mlx/commits/main --jq '.sha'` で解決）の実測条件式を
    /// そのまま突合する診断専用の純粋関数。**本実装の `tile::select` へは
    /// 組み込まない**（本ファイル冒頭コメント・`docs/backend-metal-
    /// splitk-decision.md` §2 参照）。
    ///
    /// `min_tmn_threshold` は `devc`（`MTLDevice::architecture()` 名の末尾
    /// 文字）が `'s'`／`'d'`（Mac Studio／Mac Pro〈Duo〉系を指すと推測される。
    /// MLX ソース中に明示コメントはなく本ドキュメントでは断定しない）の
    /// 場合 2048、それ以外は 1024。本実装の実機検証環境（M4 Max。
    /// `docs/real-hardware-verification-env.md` §1）はいずれにも該当しない
    /// 想定のため、本関数は常に 1024 を使う（`docs/backend-metal-
    /// mlx-classic-nax-decision.md` と同様、確認できない事項は推定で断定
    /// しない方針に従い、この前提を明記する）。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct MlxSplitkDomain {
        /// `ceil(m/16)`（MLX 側変数名 `_tm`）。
        pub tm: u64,
        /// `ceil(n/16)`（MLX 側変数名 `_tn`）。
        pub tn: u64,
        /// `k/16`（切り捨て。MLX 側変数名 `_tk`）。
        pub tk: u64,
        /// `tm*tn <= 1024 && tk >= 8 && k >= max(m,n)` の Case 1 該当判定。
        pub in_case1_domain: bool,
    }

    /// [`MlxSplitkDomain`] を算出する（`matmul.cpp:913-916,921,925-926` の
    /// 式をそのまま突合。`min_tmn_threshold` は上記ドキュメント参照の前提で
    /// 1024 固定）。
    pub fn mlx_case1_domain(m: usize, n: usize, k: usize) -> MlxSplitkDomain {
        const MIN_TMN_THRESHOLD: u64 = 1024;

        let tm = (m as u64).div_ceil(16);
        let tn = (n as u64).div_ceil(16);
        let tk = (k as u64) / 16;
        let max_mn = m.max(n) as u64;

        let in_case1_domain = tm * tn <= MIN_TMN_THRESHOLD && tk >= 8 && (k as u64) >= max_mn;

        MlxSplitkDomain {
            tm,
            tn,
            tk,
            in_case1_domain,
        }
    }

    /// `2*m*n*k`（総 FLOPs）に最も近い `2*s^3` を与える、8 の倍数の `s` を
    /// 求める（対照の正方立方形状決定用。ドキュメント冒頭「対象形状・対照
    /// 形状」節参照）。`s=0` は無意味なため下限 8 で切り上げる。
    pub fn matched_cube_side(m: usize, n: usize, k: usize) -> usize {
        let total = (m as f64) * (n as f64) * (k as f64);
        let approx = total.cbrt();
        let rounded = (approx / 8.0).round() * 8.0;
        (rounded as usize).max(8)
    }

    /// `(m, n, k)` 1 形状分の解析値。
    #[derive(Debug, Clone, Copy)]
    pub struct ShapeAnalytics {
        pub m: usize,
        pub n: usize,
        pub k: usize,
        pub tile: TileConfig,
        /// 実際に発行される threadgroup 数（`ceil(m/bm) * ceil(n/bn)`。K 方向は
        /// 1 threadgroup が全域を直列ループするため次元に含まれない — これが
        /// split-K 非対応構造そのものを表す値である）。
        pub actual_groups: u64,
        /// 1 threadgroup が K 方向ループで通過するタイル数（`ceil(k/bk)`）。
        pub k_tile_count: u64,
        pub flops: u64,
        pub mlx_domain: MlxSplitkDomain,
    }

    fn ceil_div(a: usize, b: u32) -> u64 {
        (a as u64).div_ceil(b as u64)
    }

    /// [`ShapeAnalytics`] を算出する（`tile::select` の本番選択結果を前提に
    /// 並列度・K ループ回数・MLX 選択域該当を求める。`gemm_diagnosis.rs::
    /// analytics::analyze` と同じ設計だが正方形状に限定しない）。
    pub fn analyze(m: usize, n: usize, k: usize) -> ShapeAnalytics {
        let tile = backend_metal::tile::select(m, n, k);

        let groups_m = ceil_div(m, tile.bm);
        let groups_n = ceil_div(n, tile.bn);
        let actual_groups = groups_m * groups_n;
        let k_tile_count = ceil_div(k, tile.bk);

        let flops = 2 * (m as u64) * (n as u64) * (k as u64);
        let mlx_domain = mlx_case1_domain(m, n, k);

        ShapeAnalytics {
            m,
            n,
            k,
            tile,
            actual_groups,
            k_tile_count,
            flops,
            mlx_domain,
        }
    }
}

/// 対象形状群（K 支配的非正方。`M == N`）。ドキュメント冒頭「対象形状・
/// 対照形状」節・`docs/perf/metal-gemm-splitk-shapes.md`「対象形状群」節参照。
const TARGET_MN: [usize; 4] = [32, 64, 128, 256];
const TARGET_K: [usize; 3] = [2048, 4096, 8192];

/// 対象形状 1 件と、その対照（同程度 FLOPs の正方立方形状）をまとめた組。
struct ShapePair {
    target: (usize, usize, usize),
    control_side: usize,
}

fn shape_pairs() -> Vec<ShapePair> {
    let mut pairs = Vec::with_capacity(TARGET_MN.len() * TARGET_K.len());
    for &mn in &TARGET_MN {
        for &k in &TARGET_K {
            let control_side = analytics::matched_cube_side(mn, mn, k);
            pairs.push(ShapePair {
                target: (mn, mn, k),
                control_side,
            });
        }
    }
    pairs
}

/// 解析値 1 行分を stdout へ出力する（macOS・非 macOS 双方の `main` から
/// 共有。`label` は `target` か `control` かを区別する）。
fn print_analytics_line(label: &str, a: &analytics::ShapeAnalytics) {
    println!(
        "label={label} m={} n={} k={} tile={}x{}x{}({}x{}, staged={}) actual_groups={} \
         k_tile_count={} flops={} mlx_case1_domain(tm={},tn={},tk={})={}",
        a.m,
        a.n,
        a.k,
        a.tile.bm,
        a.tile.bn,
        a.tile.bk,
        a.tile.wm,
        a.tile.wn,
        a.tile.staged,
        a.actual_groups,
        a.k_tile_count,
        a.flops,
        a.mlx_domain.tm,
        a.mlx_domain.tn,
        a.mlx_domain.tk,
        a.mlx_domain.in_case1_domain,
    );
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use super::{analytics, print_analytics_line, shape_pairs};
    use backend_metal::{MetalContext, MetalGemm};
    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{Measurement, MeasurementConfig, run as bench_run};

    /// 決定的シード（`gemm_bench.rs::SEED`・`gemm_diagnosis.rs::SEED` と同一値。
    /// 既存ベンチと同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// `m×n×k` の `dispatch_auto` を計測する（`gemm_diagnosis.rs::
    /// wall_measurement` と同型）。呼び出しごとに A・B のアップロード・C の
    /// readback を含む end-to-end 壁時計計測であり、カーネル時間の下限のみ
    /// を与える（`gemm_diagnosis.rs` モジュールドキュメント「転送時間分離を
    /// 試みて撤回した経緯」節と同じ理由でこの example も分離を試みない）。
    fn wall_measurement(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        config: &MeasurementConfig,
    ) -> Measurement {
        let mut rng = Xorshift64Star::new(SEED);
        let a = rng.fill_vec(m * k);
        let b = rng.fill_vec(k * n);

        bench_run(config, || {
            gemm.dispatch_auto(ctx, &a, &b, m, n, k)
                .expect("Metal GEMM dispatch_auto に失敗した（実機でのみ実行する前提）");
        })
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない")
    }

    /// 1 形状分の壁時計 TFLOPS 下限を算出・出力する。
    fn measure_and_print(
        label: &str,
        gemm: &MetalGemm,
        ctx: &MetalContext,
        m: usize,
        n: usize,
        k: usize,
        config: &MeasurementConfig,
    ) {
        let a = analytics::analyze(m, n, k);
        print_analytics_line(label, &a);

        let measurement = wall_measurement(gemm, ctx, m, n, k, config);
        let wall = measurement.median_secs;
        // `tflops_lower_bound`: 転送時間は非負という不等式のみから導かれる
        // 健全な下限値（`gemm_diagnosis.rs` の同名指標と同一の設計判断）。
        let tflops_lower_bound = a.flops as f64 / wall / 1e12;

        println!(
            "label={label} m={m} n={n} k={k} wall_ms={:.4} wall_q1_ms={:.4} wall_q3_ms={:.4} \
             tflops_lower_bound={:.4}",
            wall * 1e3,
            measurement.q1_secs * 1e3,
            measurement.q3_secs * 1e3,
            tflops_lower_bound,
        );
    }

    /// `--iters=<N>` で warmup・計測回数を引き上げる（`gemm_diagnosis.rs::
    /// parse_iters_override` と同型。未指定なら `MeasurementConfig::default`
    /// = 20/20 のまま）。
    fn resolve_measurement_config() -> Result<MeasurementConfig, String> {
        for arg in std::env::args().skip(1) {
            if let Some(v) = arg.strip_prefix("--iters=") {
                let n: usize = v
                    .parse()
                    .map_err(|_| format!("--iters の値が不正: '{v}'"))?;
                return MeasurementConfig::new(n, n).map_err(|e| e.to_string());
            }
        }
        Ok(MeasurementConfig::default())
    }

    pub fn main() {
        let config = match resolve_measurement_config() {
            Ok(config) => config,
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        };

        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        // サーマルドリフト対策として対象・対照を interleave する
        // （`docs/perf/metal-bench-noise-protocol.md` の順序バイアス相殺の
        // 趣旨。A/B 採否判定ではないため `run_ab` までは不要）。
        for pair in shape_pairs() {
            let (m, n, k) = pair.target;
            let s = pair.control_side;
            measure_and_print("target", &gemm, &ctx, m, n, k, &config);
            measure_and_print("control", &gemm, &ctx, s, s, s, &config);
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け: 解析値のみ出力する（`gemm_diagnosis.rs` の非 macOS
/// フォールバックと同型）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_splitk_shapes_bench example: wall_ms/tflops_lower_bound measurement \
         requires macOS (Apple Silicon). Analytical values (tile::select 選択結果・threadgroup 数・ \
         K ループ回数・MLX split-K Case 1 選択域該当) below are computed on any platform.\n"
    );
    for pair in shape_pairs() {
        let (m, n, k) = pair.target;
        let s = pair.control_side;
        let target = analytics::analyze(m, n, k);
        let control = analytics::analyze(s, s, s);
        print_analytics_line("target", &target);
        print_analytics_line("control", &control);
        println!(
            "  flops_ratio(control/target)={:.4}",
            control.flops as f64 / target.flops as f64
        );
    }
}

#[cfg(test)]
mod tests {
    use super::analytics::*;

    #[test]
    fn matched_cube_side_is_multiple_of_eight() {
        for &(m, n, k) in &[(32, 32, 2048), (64, 64, 4096), (256, 256, 8192)] {
            let s = matched_cube_side(m, n, k);
            assert_eq!(s % 8, 0, "m={m} n={n} k={k} で s={s} が 8 の倍数でない");
            assert!(s >= 8);
        }
    }

    #[test]
    fn matched_cube_side_approximately_preserves_flops() {
        // 丸め誤差は許容するが、オーダーが大きく外れていないことを確認する
        // （TFLOPS 比較の妥当性の前提。ドキュメント冒頭「対象形状・対照形状」
        // 節参照）。
        for &(m, n, k) in &[(64, 64, 2048), (128, 128, 4096), (256, 256, 8192)] {
            let s = matched_cube_side(m, n, k);
            let target_flops = 2.0 * m as f64 * n as f64 * k as f64;
            let control_flops = 2.0 * (s as f64).powi(3);
            let ratio = control_flops / target_flops;
            assert!(
                (0.5..2.0).contains(&ratio),
                "m={m} n={n} k={k} s={s} で flops 比が想定範囲外: {ratio}"
            );
        }
    }

    #[test]
    fn mlx_case1_domain_matches_known_cases() {
        // MLX 実測条件（matmul.cpp:913-935。`docs/backend-metal-
        // splitk-decision.md` §1 参照）: tm*tn<=1024 && tk>=8 && k>=max(m,n)。
        // 本テストの対象形状群（TARGET_MN×TARGET_K）は全点が該当することを
        // 固定する（形状群自体が MLX 選択域を参照して確定されたものである
        // ことのリグレッション検知）。
        for &mn in &super::TARGET_MN {
            for &k in &super::TARGET_K {
                let d = mlx_case1_domain(mn, mn, k);
                assert!(
                    d.in_case1_domain,
                    "m=n={mn} k={k} が MLX Case 1 選択域外: {d:?}"
                );
            }
        }
    }

    #[test]
    fn mlx_case1_domain_excludes_large_square() {
        // 大きな正方立方（K が M/N と同程度）は tm*tn > 1024 か k < max(m,n)
        // のいずれかで域外になることの対照確認。
        let d = mlx_case1_domain(4096, 4096, 4096);
        assert!(!d.in_case1_domain);
    }
}
