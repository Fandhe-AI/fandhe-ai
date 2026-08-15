//! Metal GEMM の size=1024 以降スループット頭打ち（#381 実測。
//! `docs/perf/metal-gemm-dynamic-tile.md`）を定量診断するバイナリ
//! （イシュー #487。親 #480「GEMM 最適化の計測前提確定・実機プローブ・
//! ボトルネック診断」の A-7）。
//!
//! 診断結果は Phase D（#530）の 2 タスクの優先順位確定に使う:
//! - **D-2（#533）**: staged 協調ロードの float4 ベクトル化
//! - **D-7（#541/#542）**: occupancy 目標算出のタイル選択への組み込み
//!
//! 本イシューは「実装変更を伴わない調査・計測・記録タスクのみ」
//! （親 #480 本文）のため `crates/backend-metal/src/`・`shaders/gemm.metal`
//! は一切変更しない。占有率・バリア回数・理論トラフィックの算出式は
//! [`analytics`] モジュールへローカル実装した（`crate::tile::select` の
//! 選択結果を読むだけの純粋関数。`tile.rs` への恒久実装は D-7a・#541 の
//! 受け入れ基準であり、ここで先取りすると重複実装になるため避けた）。
//!
//! ## 実行方法
//!
//! 解析値（threadgroup 数・バリア回数・理論トラフィック・arithmetic
//! intensity）は macOS 以外でも算出できる（`objc2` 系 FFI に触れない
//! 純粋関数のため。`cargo run -p backend-metal --example gemm_diagnosis`
//! で Linux でも実行できる）。カーネル実行時間・実効帯域・TFLOPS の実測
//! （`kernel_ms` 以降の列）は macOS 実機限定（下記「実測部分の設計」）。
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_diagnosis --release
//! ```
//!
//! 計測手順・記録テンプレート・判定基準は
//! `docs/perf/metal-gemm-bottleneck-diagnosis.md` を参照。
//!
//! ## 実測部分の設計（macOS 限定）
//!
//! `crate::pipeline::make_pipeline_with_constants` は `pub(crate)` であり
//! 本 example（クレート外）から呼べない。そのため独自パイプライン構築
//! による `MTLCommandBuffer::GPUStartTime`/`GPUEndTime` 直接採取（計画の
//! 第一候補）ではなく、既存公開 API（`MetalGemm::dispatch_auto`）を
//! `bench-harness::protocol::run` で壁時計計測するフォールバック経路
//! （計画 §3.1 のフォールバック節）を採用した。`dispatch_auto` は呼び出し
//! ごとに A・B のアップロード・C の readback を含む（`gemm_bench.rs` と
//! 同じ計測範囲）ため、そのままでは転送時間がカーネル時間に混入し、
//! 転送律速が「頭打ち」として誤診断される恐れがある（size=4096 で
//! A・B・C 合計約 192MiB のホストコピーを含む）。
//!
//! この混入を避けるため、同一 `(m, n)` で `k` を最小構成（8。
//! `simdgroup_load` の最小整除単位）にした参照計測を追加し、
//! `kernel_ms ≈ wall_ms(size) - wall_ms(size, k=8)` という差分近似で
//! カーネル純時間を推定する（転送量は `k` に依らずほぼ一定という仮定に
//! 基づく近似であり、正確な GPU タイムスタンプではない。この点を
//! doc・出力の両方に明記し「近似値」であることを利用者が誤読しない
//! ようにする）。

/// 実効帯域・occupancy 等の解析値。`objc2` 系 FFI に触れない純粋関数の
/// ため `cfg(target_os = "macos")` を付けず、Linux（本実装環境・CI）でも
/// コンパイル・実行できる（`crate::tile`・`crate::pad` と同じ設計判断）。
mod analytics {
    use backend_metal::TileConfig;

    /// M4 Max（実機検証環境。`docs/real-hardware-verification-env.md` §1）
    /// の GPU コア数。`docs/perf/metal-gemm-dynamic-tile.md:53` の実測記録
    /// （`sysctl -n hw.model` = `Mac16,6`・GPU コア 40）を出典とする。
    /// `MTLDevice` に公開の GPU コア数取得 API は存在しないため、本 example
    /// は解析対象を実機検証環境の固定値として扱う（他 Mac 実機での
    /// occupancy 判定に流用しないよう doc に明記する）。
    pub const M4_MAX_GPU_CORE_COUNT: u64 = 40;

    /// MFA（Metal FlashAttention）の occupancy 判定式
    /// `idealGroups = coreCount * multiplier`（FP32 系は経験則で 6 倍。
    /// イシュー #487 計画「occupancy 不足の判定」節が出発点として指定）。
    pub const MFA_IDEAL_GROUPS_MULTIPLIER: u64 = 6;

    /// `size×size×size` 正方 GEMM 1 件分の解析値
    /// （`gemm.metal` の staged 経路・`tile::select` の選択結果を前提に
    /// 算出。occupancy・バリア回数・理論トラフィックはすべて `k` を
    /// パディング前の実サイズとして計算するため、8 の倍数でない `size`
    /// を渡した場合はカーネル側パディング後の実効値と厳密には一致しない
    /// 点に注意。本診断が対象とする 4 サイズ〈512/1024/2048/4096〉は
    /// すべて 8 の倍数のため実運用上の乖離はない）。
    #[derive(Debug, Clone, Copy)]
    pub struct SizeAnalytics {
        pub size: usize,
        pub tile: TileConfig,
        /// 実際に発行される threadgroup 数（`ceil(size/bm) * ceil(size/bn)`）。
        pub actual_groups: u64,
        /// MFA 判定式によるコア飽和目標 threadgroup 数。
        pub ideal_groups: u64,
        /// 1 threadgroup が K 方向ループで通過する `threadgroup_barrier`
        /// 回数（staged 経路: `gemm.metal:427,441` の 2 回 × K タイル数。
        /// direct 経路〈`staged=false`〉はバリアなし）。
        pub barriers_per_tg: u64,
        /// A・B タイルの device→threadgroup 共有メモリロード総量
        /// （全 threadgroup・全 K タイル合計。`TileConfig::shared_mem_bytes`
        /// を K タイル数・threadgroup 数倍したもの。threadgroup 間・K タイル
        /// 間のキャッシュ再利用は考慮しない下限値であり、実効 arithmetic
        /// intensity の下限を与える）。
        pub load_bytes_total: u64,
        /// C の書き込み総量（`size*size*4` バイト）。
        pub store_bytes_total: u64,
        /// 総浮動小数点演算数（`2*size^3`）。
        pub flops: u64,
        /// `flops / (load_bytes_total + store_bytes_total)`
        /// （FLOP/byte。ロード・演算のどちらが律速かの一次指標）。
        pub arithmetic_intensity: f64,
    }

    fn ceil_div(a: usize, b: u32) -> u64 {
        (a as u64).div_ceil(b as u64)
    }

    /// [`SizeAnalytics`] を算出する。`tile::select` を正方形状
    /// （`m=n=k=size`）で呼び出し、その選択結果に基づいて
    /// occupancy・バリア回数・理論トラフィックを求める。
    pub fn analyze(size: usize) -> SizeAnalytics {
        let tile = backend_metal::tile::select(size, size, size);

        let groups_m = ceil_div(size, tile.bm);
        let groups_n = ceil_div(size, tile.bn);
        let actual_groups = groups_m * groups_n;
        let ideal_groups = M4_MAX_GPU_CORE_COUNT * MFA_IDEAL_GROUPS_MULTIPLIER;

        let k_tile_count = ceil_div(size, tile.bk);
        let barriers_per_tg = if tile.staged { 2 * k_tile_count } else { 0 };

        // `staged=false`（direct load）の場合も総ロード量自体は同一
        // （device メモリから同じ要素数を読む）と仮定し、
        // `shared_mem_bytes()` の代わりに直接算出する。
        let bytes_per_group_per_ktile =
            (tile.bm as u64 * tile.bk as u64 + tile.bk as u64 * tile.bn as u64) * 4;
        let load_bytes_total = actual_groups * k_tile_count * bytes_per_group_per_ktile;
        let store_bytes_total = (size as u64) * (size as u64) * 4;

        let flops = 2 * (size as u64).pow(3);
        let arithmetic_intensity = flops as f64 / (load_bytes_total + store_bytes_total) as f64;

        SizeAnalytics {
            size,
            tile,
            actual_groups,
            ideal_groups,
            barriers_per_tg,
            load_bytes_total,
            store_bytes_total,
            flops,
            arithmetic_intensity,
        }
    }
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use super::analytics::{self, SizeAnalytics};
    use backend_metal::{MetalContext, MetalGemm};
    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, run as bench_run};

    /// 決定的シード（`gemm_bench.rs::SEED` と同一値。過去 PoC・既存ベンチと
    /// 同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// `simdgroup_load` の最小整除単位（8）。転送時間の参照計測
    /// （`k` 最小化ベースライン）に使う `k` 値（モジュールドキュメント
    /// 「実測部分の設計」参照）。
    const MIN_K: usize = 8;

    /// `m×n×k` の `dispatch_auto` を計測し中央値秒を返す
    /// （`gemm_bench.rs::measure_auto` と同型。呼び出しごとに A・B の
    /// アップロード・C の readback を含む壁時計計測）。
    fn wall_secs(
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

        measurement.median_secs
    }

    pub fn main() {
        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
        let config = MeasurementConfig::default();

        for size in [512usize, 1024, 2048, 4096] {
            let a: SizeAnalytics = analytics::analyze(size);

            let wall = wall_secs(&gemm, &ctx, size, size, size, &config);
            // 転送オーバーヘッドの近似ベースライン（モジュールドキュメント
            // 「実測部分の設計」参照。`k` を最小化することでカーネル計算量を
            // 最小化しつつ、A・B・C の転送量はほぼそのままに保つ）。
            let transfer_baseline = wall_secs(&gemm, &ctx, size, size, MIN_K, &config);
            // 近似カーネル時間は負にならない範囲でクランプする（計測ノイズで
            // `wall < transfer_baseline` になり得るため）。
            let kernel_secs = (wall - transfer_baseline).max(0.0);

            let tflops = a.flops as f64 / kernel_secs / 1e12;
            // `eff_bw_gbs`: `load_bytes_total + store_bytes_total` を
            // 近似カーネル時間で割った実効帯域（GB/s）。M4 Max 公称帯域
            // 546GB/s（`docs/perf/metal-gemm-bottleneck-diagnosis.md`
            // 出典節）に対する比率は doc 側で算出する。
            let eff_bw_gbs = (a.load_bytes_total + a.store_bytes_total) as f64 / kernel_secs / 1e9;

            println!(
                "size={} tile={}x{}x{}({}x{}, staged={}) actual_groups={} ideal_groups={} \
                 barriers_per_tg={} arithmetic_intensity={:.4} wall_ms={:.4} \
                 transfer_baseline_ms={:.4} kernel_ms_approx={:.4} tflops_approx={:.4} \
                 eff_bw_gbs_approx={:.4}",
                a.size,
                a.tile.bm,
                a.tile.bn,
                a.tile.bk,
                a.tile.wm,
                a.tile.wn,
                a.tile.staged,
                a.actual_groups,
                a.ideal_groups,
                a.barriers_per_tg,
                a.arithmetic_intensity,
                wall * 1e3,
                transfer_baseline * 1e3,
                kernel_secs * 1e3,
                tflops,
                eff_bw_gbs,
            );
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos_impl::main();
}

/// 非 macOS 環境向け: 解析値のみ出力する（`objc2` 系は
/// `cfg(target_os = "macos")` 限定のため実測部分はコンパイル対象外だが、
/// [`analytics`] は純粋関数のため Linux でも算出できる。Linux CI の
/// `cargo build --workspace --all-targets`／`cargo clippy --all-targets`
/// をこの example も含めて通しつつ、`docs/perf/metal-gemm-bottleneck-diagnosis.md`
/// の事前計算表を再生成する手段としても使える）。
#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "backend-metal gemm_diagnosis example: kernel_ms/tflops/eff_bw measurement requires \
         macOS (Apple Silicon). Analytical values (occupancy / barriers / arithmetic intensity) \
         below are computed on any platform.\n"
    );
    for size in [512usize, 1024, 2048, 4096] {
        let a = analytics::analyze(size);
        println!(
            "size={} tile={}x{}x{}({}x{}, staged={}) actual_groups={} ideal_groups={} \
             barriers_per_tg={} load_bytes_total={} store_bytes_total={} flops={} \
             arithmetic_intensity={:.4}",
            a.size,
            a.tile.bm,
            a.tile.bn,
            a.tile.bk,
            a.tile.wm,
            a.tile.wn,
            a.tile.staged,
            a.actual_groups,
            a.ideal_groups,
            a.barriers_per_tg,
            a.load_bytes_total,
            a.store_bytes_total,
            a.flops,
            a.arithmetic_intensity,
        );
    }
}
