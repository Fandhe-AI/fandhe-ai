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
//! で Linux でも実行できる）。壁時計計測・TFLOPS/ロードスループット下限
//! （`wall_ms` 以降の列）は macOS 実機限定（下記「実測部分の設計」）。
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_diagnosis --release
//! ```
//!
//! ### GPU デバイスプロファイル（occupancy 判定式のパラメータ）
//!
//! `ideal_groups` の算出（[`analytics::DeviceProfile`]）は既定で M4 Max
//! 実機検証環境（`gpu_core_count=40`・`ideal_groups_multiplier=6`）を前提
//! とする。macOS 実行時は `sysctl -n hw.model` で実機モデルを検出し、
//! `Mac16,6`（M4 Max）以外では誤った occupancy 判定を避けるため実行を
//! **拒否する**（fail-closed。codex-review 指摘 P1。PR #649）。他デバイス
//! で診断したい場合は `--gpu-core-count`・`--ideal-groups-multiplier` を
//! 両方明示指定する:
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_diagnosis --release -- \
//!     --gpu-core-count=20 --ideal-groups-multiplier=6
//! ```
//!
//! 非 macOS（解析値のみ算出。実機検出手段がない）は CLI 引数の明示指定が
//! なければ M4 Max 既定プロファイルを使い、その旨を stderr へ警告出力する。
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
//! 同じ計測範囲）ため、`wall_ms` は純粋なカーネル時間ではなく
//! 「A・B アップロード＋カーネル実行＋C readback」の end-to-end 時間で
//! ある。
//!
//! ### 転送時間分離を試みて撤回した経緯（イシュー #487 PR #649）
//!
//! 当初は同一 `(m, n)` で `k` を小さくした点（`k=8` 単独、次に
//! `k=8`・`k=32` の 2 点線形外挿）を「転送時間ベースライン」として
//! `wall_ms(size)` から差し引き `kernel_ms_approx`・`tflops_approx`・
//! `eff_bw_gbs_approx` を算出していたが、`tile::select` は `k < 64`
//! （`SMALL` 閾値未満）で `SINGLE_SIMDGROUP_8X8`（`bm=bn=8`・
//! `staged=false`）を選ぶため、この参照点は `actual_groups =
//! ceil(size/8)^2` という実測対象（staged 64×64 タイル）とは全く異なる
//! 大量の threadgroup をディスパッチする。この参照点の壁時計時間は
//! A・B 転送だけでなく直接ロード形式カーネルの演算・ディスパッチ
//! オーバーヘッドを主として含み、しかもその演算量も `k` にほぼ比例して
//! 増えるため、2 点間の傾きを「転送レート」として `k = size` まで
//! 外挿すると staged カーネルとは無関係な演算時間を転送時間の名目で
//! 拡大して差し引くことになる。結果として `kernel_ms_approx` が
//! ゼロに近づき（ときに負になり `max(0.0)` 後の除算で `inf`
//! が出力される）、`tflops_approx`・`eff_bw_gbs_approx` の基礎が
//! 成立しなかった（cursor[bot] review id 4943646199・codex-review
//! 指摘・Cursor Bugbot 指摘。いずれも PR #649 で確認）。
//!
//! GPU timestamp 直接採取（`MTLCommandBuffer::GPUStartTime`/`GPUEndTime`）
//! または演算を伴わない同量転送のみの対照経路は、いずれも
//! `crate::pipeline::make_pipeline_with_constants` 等クレート内部への
//! アクセスを要するが、本イシュー（親 #480 本文）は「実装変更を伴わない
//! 調査・計測・記録タスクのみ」と明記されておりクレート内部
//! （`crates/backend-metal/src/`・`shaders/gemm.metal`）の変更はスコープ外
//! である。そのため本 example は転送・カーネルの分離を**試みない**方針へ
//! 変更した: `wall_ms` を size ごとの end-to-end 指標としてそのまま報告し、
//! そこから導出する性能指標は「`wall_secs ≥ kernel_secs` （転送時間は
//! 非負）」という不等式のみから成立する**下限値**
//! （`tflops_lower_bound`・`logical_load_gbs_lower_bound`。詳細は
//! [`macos_impl`] のコメント参照）に限定する。分離を諦めた代わりに、
//! ロード律速・occupancy 不足の判定は非 macOS でも算出できる
//! [`analytics`]（`arithmetic_intensity`・`actual_groups`/`ideal_groups`・
//! `barriers_per_tg`）を主に用いる方針へ
//! `docs/perf/metal-gemm-bottleneck-diagnosis.md` §5 も合わせて改訂した。

/// 実効帯域・occupancy 等の解析値。`objc2` 系 FFI に触れない純粋関数の
/// ため `cfg(target_os = "macos")` を付けず、Linux（本実装環境・CI）でも
/// コンパイル・実行できる（`crate::tile`・`crate::pad` と同じ設計判断）。
mod analytics {
    use backend_metal::TileConfig;

    /// GPU デバイスプロファイル（occupancy 判定式
    /// `idealGroups = gpu_core_count * ideal_groups_multiplier`
    /// （MFA〈Metal FlashAttention〉の FP32 系判定式。イシュー #487 計画
    /// 「occupancy 不足の判定」節が出発点として指定）のパラメータ）。
    ///
    /// `MTLDevice` に公開の GPU コア数取得 API は存在しないため、本
    /// example は解析対象を明示的なプロファイル値として扱う。既定値
    /// （[`Self::M4_MAX`]）を他 Mac 実機の occupancy 判定に無警告で流用
    /// しないよう、呼び出し側（`main`）で実機モデル検出による
    /// fail-closed 拒否・CLI 引数上書きを行う（モジュールドキュメント
    /// 「GPU デバイスプロファイル」節・codex-review 指摘 P1・PR #649）。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DeviceProfile {
        pub gpu_core_count: u64,
        pub ideal_groups_multiplier: u64,
    }

    impl DeviceProfile {
        /// M4 Max（実機検証環境。`docs/real-hardware-verification-env.md`
        /// §1）の既定プロファイル。`docs/perf/metal-gemm-dynamic-tile.md:53`
        /// の実測記録（`sysctl -n hw.model` = `Mac16,6`・GPU コア 40）と
        /// MFA 経験則の 6 倍を出典とする。
        pub const M4_MAX: DeviceProfile = DeviceProfile {
            gpu_core_count: 40,
            ideal_groups_multiplier: 6,
        };
    }

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
    pub fn analyze(size: usize, profile: DeviceProfile) -> SizeAnalytics {
        let tile = backend_metal::tile::select(size, size, size);

        let groups_m = ceil_div(size, tile.bm);
        let groups_n = ceil_div(size, tile.bn);
        let actual_groups = groups_m * groups_n;
        let ideal_groups = profile.gpu_core_count * profile.ideal_groups_multiplier;

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

/// `--gpu-core-count=<N>` `--ideal-groups-multiplier=<M>` を解析し、明示
/// 指定された [`analytics::DeviceProfile`] を返す（macOS・非 macOS 双方の
/// `main` から呼ばれる）。既定の M4 Max プロファイルを他デバイスへ無警告
/// で流用しないための CLI 上書き経路（モジュールドキュメント「GPU
/// デバイスプロファイル」節・codex-review 指摘 P1・PR #649）。
///
/// 誤った occupancy 判定を招く片方だけの指定は `Err` で拒否する
/// （fail-closed）。両方とも未指定なら `Ok(None)`（呼び出し側が既定
/// プロファイルの解決方法〈macOS: 実機モデル検出／非 macOS: M4 Max 既定 +
/// 警告〉を判断する）。
fn parse_device_profile_override() -> Result<Option<analytics::DeviceProfile>, String> {
    let mut gpu_core_count: Option<u64> = None;
    let mut ideal_groups_multiplier: Option<u64> = None;

    for arg in std::env::args().skip(1) {
        if let Some(v) = arg.strip_prefix("--gpu-core-count=") {
            gpu_core_count = Some(
                v.parse()
                    .map_err(|_| format!("--gpu-core-count の値が不正: '{v}'"))?,
            );
        } else if let Some(v) = arg.strip_prefix("--ideal-groups-multiplier=") {
            ideal_groups_multiplier = Some(
                v.parse()
                    .map_err(|_| format!("--ideal-groups-multiplier の値が不正: '{v}'"))?,
            );
        }
    }

    match (gpu_core_count, ideal_groups_multiplier) {
        (Some(gpu_core_count), Some(ideal_groups_multiplier)) => {
            Ok(Some(analytics::DeviceProfile {
                gpu_core_count,
                ideal_groups_multiplier,
            }))
        }
        (None, None) => Ok(None),
        _ => Err(
            "--gpu-core-count と --ideal-groups-multiplier は両方同時に指定する必要がある \
             （片方のみの指定は不正な occupancy 判定を招くため拒否する）"
                .to_string(),
        ),
    }
}

#[cfg(target_os = "macos")]
mod macos_impl {
    use super::analytics::{self, DeviceProfile, SizeAnalytics};
    use backend_metal::{MetalContext, MetalGemm};
    use bench_harness::rng::Xorshift64Star;
    use bench_harness::{MeasurementConfig, run as bench_run};

    /// 決定的シード（`gemm_bench.rs::SEED` と同一値。過去 PoC・既存ベンチと
    /// 同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

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

    /// 実行対象デバイスの [`DeviceProfile`] を解決する。CLI 明示指定
    /// （`super::parse_device_profile_override`）を最優先で使い（他
    /// デバイスでの意図的な実行を許可するため）、未指定の場合は
    /// `sysctl -n hw.model` で実機モデルを検出する。検出結果が本 example
    /// の前提実機（M4 Max・`Mac16,6`）と一致しない、または検出自体に
    /// 失敗した場合は、誤った occupancy 判定（他デバイスに M4 Max 用の
    /// `ideal_groups` を無警告で適用してしまう）を避けるため実行を拒否
    /// する（fail-closed。codex-review 指摘 P1・PR #649）。
    fn resolve_device_profile() -> Result<DeviceProfile, String> {
        if let Some(profile) = super::parse_device_profile_override()? {
            return Ok(profile);
        }

        const SUPPORTED_MODEL: &str = "Mac16,6"; // M4 Max（実機検証環境）

        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.model"])
            .output()
            .map_err(|e| format!("`sysctl -n hw.model` の実行に失敗した: {e}"))?;
        if !output.status.success() {
            return Err("`sysctl -n hw.model` がエラー終了した".to_string());
        }
        let model = String::from_utf8_lossy(&output.stdout).trim().to_string();

        if model == SUPPORTED_MODEL {
            Ok(DeviceProfile::M4_MAX)
        } else {
            Err(format!(
                "検出デバイス '{model}' は本 example が前提とする実機検証環境（M4 Max・\
                 {SUPPORTED_MODEL}）と一致しない。occupancy 判定式 \
                 (idealGroups = gpu_core_count * ideal_groups_multiplier) は機種依存のため、\
                 既定の M4 Max 用プロファイルをそのまま適用すると誤った診断結果になる。\
                 `--gpu-core-count=<N> --ideal-groups-multiplier=<M>` を明示指定して再実行すること。"
            ))
        }
    }

    pub fn main() {
        let profile = match resolve_device_profile() {
            Ok(profile) => profile,
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        };

        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");
        let config = MeasurementConfig::default();

        for size in [512usize, 1024, 2048, 4096] {
            let a: SizeAnalytics = analytics::analyze(size, profile);

            // `wall_secs` は A・B アップロード＋カーネル実行＋C readback を
            // 含む end-to-end 壁時計時間（モジュールドキュメント「転送時間
            // 分離を試みて撤回した経緯」参照。転送時間だけを差し引く手段は
            // クレート内部アクセスを要し本イシューのスコープ外のため、
            // 分離を試みず `wall_secs` をそのまま報告する）。
            let wall = wall_secs(&gemm, &ctx, size, size, size, &config);

            // `tflops_lower_bound`: 転送時間は非負（`wall_secs ≥
            // kernel_secs`）という不等式のみから導かれる健全な下限値。
            // 実際のカーネル TFLOPS はこれ以上（転送時間の分だけ高い）。
            // 分離を試みないため `_approx`（誤った精度感を与える名称）
            // ではなく `_lower_bound` と明示する。
            let tflops_lower_bound = a.flops as f64 / wall / 1e12;
            // `logical_load_gbs_lower_bound`: `load_bytes_total +
            // store_bytes_total`（`analytics::analyze` 参照。threadgroup
            // 間・K タイル間のキャッシュ再利用を考慮しない論理ロード量の
            // 下限値）を `wall_secs` で割った値。**DRAM 実効帯域ではない**
            // （キャッシュヒットにより実際の DRAM トラフィックはこれより
            // 少なく、逆に `wall_secs` が `kernel_secs` 以上であることから
            // 論理ロードスループットの下限でもある）。M4 Max 公称帯域
            // 546GB/s との比較には使わない（codex-review 指摘。PR #649）。
            let logical_load_gbs_lower_bound =
                (a.load_bytes_total + a.store_bytes_total) as f64 / wall / 1e9;

            println!(
                "size={} tile={}x{}x{}({}x{}, staged={}) actual_groups={} ideal_groups={} \
                 barriers_per_tg={} arithmetic_intensity={:.4} wall_ms={:.4} \
                 tflops_lower_bound={:.4} logical_load_gbs_lower_bound={:.4}",
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
                tflops_lower_bound,
                logical_load_gbs_lower_bound,
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
    // 非 macOS には `sysctl -n hw.model` 相当の実機検出手段がなく、この
    // 経路はそもそも実測（GPU 実行）を行わない解析専用パスのため、CLI
    // 明示指定がなければ M4 Max 既定プロファイルを使い、その旨を stderr
    // へ警告する（macOS 側の fail-closed 拒否とは異なり、実機ではない
    // ため誤った実機診断を招くリスクがない。codex-review 指摘 P1・PR #649）。
    let profile = match parse_device_profile_override() {
        Ok(Some(profile)) => profile,
        Ok(None) => {
            eprintln!(
                "--gpu-core-count/--ideal-groups-multiplier 未指定のため既定の M4 Max プロファイル \
                 (gpu_core_count=40, ideal_groups_multiplier=6) を使う。"
            );
            analytics::DeviceProfile::M4_MAX
        }
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    };

    println!(
        "backend-metal gemm_diagnosis example: wall_ms/tflops_lower_bound/logical_load_gbs_lower_bound \
         measurement requires macOS (Apple Silicon). Analytical values (occupancy / barriers / \
         arithmetic intensity) below are computed on any platform.\n"
    );
    for size in [512usize, 1024, 2048, 4096] {
        let a = analytics::analyze(size, profile);
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
