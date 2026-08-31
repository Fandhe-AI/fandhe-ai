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
//! cargo run -p fandhe-ai-backend-metal --example gemm_splitk_shapes_bench --release
//! ```
//!
//! 実測（`tflops`）は macOS 実機限定。**計測境界は `dispatch_tiled_prepared`
//! （§4 準拠 prepared 入口。`crates/backend-metal/examples/
//! gemm_f32_prepared_bench.rs`〈#572〉と同型）を使い、A・B バッファの確保・
//! アップロードは計測ループの外で 1 回だけ行い、計測対象はエンコード＋
//! コマンドバッファ完了待ちのみに限定する**（readback も対象外）。
//!
//! この境界を選ぶ理由: 対象（K 支配的非正方。M・N が小さい）と対照（同程度
//! FLOPs の正方立方）は総 FLOPs をほぼ揃えていても A・B の転送要素数
//! （`m*k + k*n = 2*m*k`〈`M==N` のため〉対 `2*s^2`）は対象側が常に多く、
//! 対象 12 点中で比率（対象/対照）は 2.0〜6.5536 倍の範囲で変動する（最大は
//! `(32,32,8192)` 対 `(200,200,200)`: `2*32*8192=524288` 要素対
//! `2*200^2=80000` 要素で 6.5536 倍。最小は `(256,256,2048)` 対
//! `(512,512,512)` で 2.0 倍）。対象側が常により多くの転送を要する
//! 方向性のある差のため、`dispatch_auto`（アップロード・readback を含む
//! end-to-end 壁時計境界）で判定基準
//! （`docs/perf/metal-gemm-splitk-shapes.md` §5 の `target/control < 0.7`）を
//! 判定すると、この転送量差だけで閾値を跨ぎうる（実装の並列度不足とは
//! 無関係な要因が判定へ混入する。codex-review 指摘対応。#810 PR #829）。
//! `dispatch_tiled_prepared` はクレート内部（`crate::pipeline::
//! make_pipeline_with_constants` 等）へアクセスせず公開 API のみで
//! 転送を計測区間外へ切り離せるため、「本イシューは `crates/backend-metal/src/`
//! を変更しない」制約と両立する。
//!
//! **順序バイアス対策**: 対象→対照の固定順で計測すると、サーマル状態・GPU
//! クロック（DVFS）変動の系統誤差が一方だけに乗りうる（`bench_harness::ab`
//! モジュールドキュメント・`docs/perf/metal-bench-noise-protocol.md` 参照）。
//! 本 example は `bench_harness::ab::run_ab`（`crates/backend-metal/examples/
//! gemm_swizzle_ab_bench.rs` フェーズ 2 と同型）で対象（side A）・対照
//! （side B）をラウンドごとに順序反転（偶数ラウンドは対象先頭、奇数
//! ラウンドは対照先頭）した interleaved 計測にし、ドリフトをラウンド間で
//! 相殺する。
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
    use fandhe_ai_backend_metal::TileConfig;

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
        // `crate::device::probe_gpu_core_count`（イシュー #1039 の厳密一致
        // テーブルの機種ゲート）は `cfg(target_os = "macos")` 限定だが、本
        // 関数はモジュール冒頭コメントのとおり `objc2` 系 FFI に触れない
        // 純粋関数として Linux（CI）でも算出できる設計を保つ。よって機種
        // ゲートは常に `None`（M4 Max 実測テーブル不使用）で評価し、形状
        // クラス判定のみによる選択を解析対象とする（`tile::select`
        // ドキュメンテーションコメント参照）。
        let tile = fandhe_ai_backend_metal::tile::select(m, n, k, None);

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
    use bench_harness::MeasurementConfig;
    use bench_harness::ab::{AbConfig, run_ab};
    use bench_harness::rng::Xorshift64Star;
    use fandhe_ai_backend_metal::pad::{pad_matrix, pad8};
    use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalGemm};
    use std::time::Duration;

    /// 決定的シード（`gemm_bench.rs::SEED`・`gemm_diagnosis.rs::SEED` と同一値。
    /// 既存ベンチと同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// ラウンド数・cooldown・時間ベースウォームアップ下限
    /// （`gemm_swizzle_ab_bench.rs::{ROUNDS,COOLDOWN,MIN_WARMUP}` と同一値。
    /// #746 実装計画 §4.2 の初期値を踏襲する）。
    const ROUNDS: usize = 6;
    const COOLDOWN: Duration = Duration::from_secs(2);
    const MIN_WARMUP: Duration = Duration::from_secs(1);

    /// `m×n×k` 用に確保・アップロード済みの prepared 入力一式
    /// （`gemm_f32_prepared_bench.rs::measure` と同型。計測ループの外で
    /// 1 回だけ構築し、計測対象からパディング・アップロードを除外する）。
    struct PreparedShape {
        m_eff: usize,
        n_eff: usize,
        k_eff: usize,
        cfg: fandhe_ai_backend_metal::tile::TileConfig,
        a_buf: MetalBuffer,
        b_buf: MetalBuffer,
        c_buf: MetalBuffer,
    }

    fn prepare(ctx: &MetalContext, m: usize, n: usize, k: usize) -> PreparedShape {
        let mut rng = Xorshift64Star::new(SEED);
        let a: Vec<f32> = rng.fill_vec(m * k);
        let b: Vec<f32> = rng.fill_vec(k * n);

        let cfg = fandhe_ai_backend_metal::tile::select(
            m,
            n,
            k,
            ctx.occupancy_params().map(|p| p.gpu_core_count),
        );
        let (m_eff, n_eff, k_eff) = (pad8(m), pad8(n), pad8(k));
        let a_padded = pad_matrix(&a, m, k, m_eff, k_eff);
        let b_padded = pad_matrix(&b, k, n, k_eff, n_eff);

        let a_buf = MetalBuffer::new_with_data(ctx, &a_padded)
            .expect("A バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");
        let b_buf = MetalBuffer::new_with_data(ctx, &b_padded)
            .expect("B バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");
        let c_buf = MetalBuffer::new_zeroed(ctx, m_eff * n_eff)
            .expect("C バッファ確保（計測外の事前準備）に失敗した（実機でのみ実行する前提）");

        PreparedShape {
            m_eff,
            n_eff,
            k_eff,
            cfg,
            a_buf,
            b_buf,
            c_buf,
        }
    }

    fn dispatch_prepared(gemm: &MetalGemm, ctx: &MetalContext, p: &PreparedShape) {
        gemm.dispatch_tiled_prepared(
            ctx, &p.a_buf, &p.b_buf, &p.c_buf, p.m_eff, p.n_eff, p.k_eff, p.cfg,
        )
        .expect("Metal f32 SimdgroupTiled GEMM dispatch_tiled_prepared に失敗した（実機でのみ実行する前提）");
    }

    fn tflops(m: usize, n: usize, k: usize, median_secs: f64) -> f64 {
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        flops / median_secs / 1e12
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

    /// 対象（side A）・対照（side B）1 組を prepared 境界・`run_ab` の
    /// 順序反転 interleave で計測し、劣化率（§5 判定基準の分子/分母）を
    /// 出力する。
    fn measure_and_print_pair(
        gemm: &MetalGemm,
        ctx: &MetalContext,
        target: (usize, usize, usize),
        control_side: usize,
        ab_config: &AbConfig,
        measurement_config: &MeasurementConfig,
    ) {
        let (tm, tn, tk) = target;
        let s = control_side;

        let target_analytics = analytics::analyze(tm, tn, tk);
        let control_analytics = analytics::analyze(s, s, s);
        print_analytics_line("target", &target_analytics);
        print_analytics_line("control", &control_analytics);

        let target_prepared = prepare(ctx, tm, tn, tk);
        let control_prepared = prepare(ctx, s, s, s);

        // ウォームアップディスパッチ 1 回（`gemm_f32_prepared_bench.rs::measure`
        // と同じ狙い。`run_ab` 内の `extended_warmup` に先立って構成解決の
        // 副作用〈パイプラインキャッシュ〉を確定させる）。
        dispatch_prepared(gemm, ctx, &target_prepared);
        dispatch_prepared(gemm, ctx, &control_prepared);

        let result = run_ab(
            ab_config,
            measurement_config,
            || dispatch_prepared(gemm, ctx, &target_prepared),
            || dispatch_prepared(gemm, ctx, &control_prepared),
        )
        .expect("MeasurementConfig::default は下限（20/20）を満たすため失敗しない");

        let target_tflops = tflops(tm, tn, tk, result.median_a_secs);
        let control_tflops = tflops(s, s, s, result.median_b_secs);
        let degradation_ratio = target_tflops / control_tflops;

        println!(
            "target=({tm},{tn},{tk}) control=({s},{s},{s}) target_tflops={:.4} \
             control_tflops={:.4} target_over_control={:.4} spread_target={:.4} \
             spread_control={:.4} target_actual_groups={} control_actual_groups={}",
            target_tflops,
            control_tflops,
            degradation_ratio,
            result.spread_a,
            result.spread_b,
            target_analytics.actual_groups,
            control_analytics.actual_groups,
        );
    }

    pub fn main() {
        let measurement_config = match resolve_measurement_config() {
            Ok(config) => config,
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        };
        let ab_config = AbConfig::new(ROUNDS, COOLDOWN, MIN_WARMUP)
            .expect("ROUNDS は偶数固定のため AbConfig::new は失敗しない");

        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        for pair in shape_pairs() {
            measure_and_print_pair(
                &gemm,
                &ctx,
                pair.target,
                pair.control_side,
                &ab_config,
                &measurement_config,
            );
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
        "backend-metal gemm_splitk_shapes_bench example: tflops measurement (dispatch_tiled_prepared \
         boundary・run_ab interleaved) requires macOS (Apple Silicon). Analytical values \
         (tile::select 選択結果・threadgroup 数・K ループ回数・MLX split-K Case 1 選択域該当) \
         below are computed on any platform.\n"
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
