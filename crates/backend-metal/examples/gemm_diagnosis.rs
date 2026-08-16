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
//! は一切変更しない。並列度〈concurrency/saturation〉ヒューリスティック・
//! バリア回数・理論トラフィックの算出式は
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
//! ### GPU デバイスプロファイル（並列度ヒューリスティックのパラメータ）
//!
//! `ideal_groups` の算出（[`analytics::DeviceProfile`]）に用いるパラメータ
//! （`gpu_core_count`・`ideal_groups_multiplier`）は、macOS（実測を伴う
//! 経路）では `--gpu-core-count`・`--ideal-groups-multiplier` の**両方の
//! 明示指定を必須**とする。`MTLDevice` に公開の GPU コア数取得 API は
//! 存在せず、`sysctl -n hw.model` の機種識別子（例: `Mac16,6`）だけでは
//! 同一機種内の構成差異（binned 版等）まで保証できないため、機種識別子
//! からの自動判定は行わない（codex-review 指摘 P1・PR #649。誤った
//! occupancy 判定を機種一致の見た目だけで許してしまう問題への対応。
//! 未指定は fail-closed でエラー終了する）:
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_diagnosis --release -- \
//!     --gpu-core-count=40 --ideal-groups-multiplier=6
//! ```
//!
//! 非 macOS（解析値のみ算出。実機を対象としないため誤診断リスクがない）
//! は CLI 引数の明示指定がなければ M4 Max 既定プロファイルを使い、その旨を
//! stderr へ警告出力する。
//!
//! ### 計測回数（ioreg サンプリング窓を広げる `--iters`）
//!
//! size=512/1024/2048 は既定（`MeasurementConfig::default()` =
//! warmup 20 回・計測 20 回）だと 1 size あたりの総実行区間が 1 秒未満で
//! 終わることがあり、`docs/perf/metal-gemm-bottleneck-diagnosis.md` §2 の
//! ioreg 継続サンプリング（0.5 秒間隔）では 1 size の区間に収まるサンプルが
//! 0〜1 個しか取れず median/max の算出が意味を持たない（cursor[bot] 指摘
//! Medium・PR #649）。`--iters=<N>`（`N` は 20 以上。TASK-8.1 下限を
//! `MeasurementConfig::new` が fail-closed で検証する）で warmup・計測回数を
//! 両方 `N` へ引き上げ、区間を意図的に伸ばして使う:
//!
//! ```sh
//! cargo run -p backend-metal --example gemm_diagnosis --release -- \
//!     --gpu-core-count=40 --ideal-groups-multiplier=6 --iters=200
//! ```
//!
//! 未指定時は既定（20/20）のまま（4096 size は 137 GFLOP/回のため、既定を
//! 引き上げると全呼び出し側の実行時間が伸びてしまう。opt-in に限定する）。
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
//! ロード律速の仮説判定・並列度〈concurrency/saturation〉ヒューリス
//! ティックによる一次観察は非 macOS でも算出できる
//! [`analytics`]（`arithmetic_intensity`・`actual_groups`/`ideal_groups`・
//! `barriers_per_tg`。`ideal_groups` が真の occupancy を表さない点は
//! [`analytics::DeviceProfile`] のドキュメント参照）を主に用いる方針へ
//! `docs/perf/metal-gemm-bottleneck-diagnosis.md` §5 も合わせて改訂した。

/// 実効帯域・並列度〈concurrency/saturation〉ヒューリスティック等の解析値
/// （`ideal_groups` は真の occupancy を表さない。[`DeviceProfile`] のドキュメント
/// 参照）。`objc2` 系 FFI に触れない純粋関数の
/// ため `cfg(target_os = "macos")` を付けず、Linux（本実装環境・CI）でも
/// コンパイル・実行できる（`crate::tile`・`crate::pad` と同じ設計判断）。
mod analytics {
    use backend_metal::TileConfig;

    /// GPU デバイスプロファイル（並列度〈concurrency/saturation〉飽和度
    /// ヒューリスティックの算出式 `idealGroups = gpu_core_count *
    /// ideal_groups_multiplier`（MFA〈Metal FlashAttention〉の FP32 系
    /// 経験式。イシュー #487 計画「occupancy 不足の判定」節が出発点として
    /// 指定）のパラメータ）。
    ///
    /// **本ヒューリスティックの限界（codex-review 指摘。PR #649）**:
    /// `idealGroups` は「コアあたり `ideal_groups_multiplier` 個の
    /// threadgroup を発行すればコアを飽和させられる」という
    /// concurrency/saturation の proxy に過ぎず、レジスタ使用量・
    /// threadgroup memory 使用量・`threads-per-threadgroup` 上限といった
    /// 真の occupancy を決める資源制約を一切表さない。したがって
    /// `actual_groups`（[`SizeAnalytics::actual_groups`]）とこの値の比較
    /// だけから「occupancy が十分／過剰である」と確定的に結論づけることは
    /// できず、あくまで「発行 threadgroup 数が経験的な飽和目標に対して
    /// 多いか少ないか」という一次指標として扱う（`docs/perf/
    /// metal-gemm-bottleneck-diagnosis.md` §5.1 も同じ限定を明記）。
    ///
    /// `MTLDevice` に公開の GPU コア数取得 API は存在しないため、本
    /// example は解析対象を明示的なプロファイル値として扱う。既定値
    /// （[`Self::M4_MAX`]）を他 Mac 実機の判定に無警告で流用しないよう、
    /// 呼び出し側（`main`）で macOS 実行時は CLI 引数
    /// （`--gpu-core-count`・`--ideal-groups-multiplier`）の明示指定を
    /// 必須化する（機種識別子〈`sysctl -n hw.model`〉だけでは同一機種内の
    /// 構成差異を保証できないため自動判定は行わない。モジュール
    /// ドキュメント「GPU デバイスプロファイル」節・codex-review 指摘 P1・
    /// PR #649）。
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DeviceProfile {
        pub gpu_core_count: u64,
        pub ideal_groups_multiplier: u64,
    }

    impl DeviceProfile {
        /// M4 Max（実機検証環境。`docs/real-hardware-verification-env.md`
        /// §1）の既定プロファイル。`docs/perf/metal-gemm-dynamic-tile.md:53`
        /// の実測記録（`sysctl -n hw.model` = `Mac16,6`・GPU コア 40）を
        /// 出典とする。`ideal_groups_multiplier` は
        /// `backend_metal::tile::IDEAL_GROUPS_MULTIPLIER_F32`
        /// （MFA 経験則の f32 系係数）をそのまま参照し、診断経路と
        /// ライブラリ経路で係数値が食い違わないようにする（単一真実源。
        /// codex-review 指摘・PR #662）。
        pub const M4_MAX: DeviceProfile = DeviceProfile {
            gpu_core_count: 40,
            ideal_groups_multiplier: backend_metal::tile::IDEAL_GROUPS_MULTIPLIER_F32 as u64,
        };
    }

    /// `size×size×size` 正方 GEMM 1 件分の解析値
    /// （`gemm.metal` の staged 経路・`tile::select` の選択結果を前提に
    /// 算出。並列度〈concurrency/saturation〉ヒューリスティック・
    /// バリア回数・理論トラフィックはすべて `k` を
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
        /// MFA 経験式によるコア飽和目標 threadgroup 数。
        /// [`DeviceProfile`] のドキュメント「本ヒューリスティックの限界」
        /// 節参照 — レジスタ・threadgroup memory 等の資源制約を表さない
        /// concurrency/saturation の proxy であり、真の occupancy ではない。
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
    /// 並列度〈concurrency/saturation〉ヒューリスティック・バリア回数・
    /// 理論トラフィックを求める（真の occupancy ではない点は
    /// [`DeviceProfile`] のドキュメント参照）。
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
/// 誤った occupancy 判定を招く片方だけの指定・ゼロ値・
/// `ideal_groups = gpu_core_count * ideal_groups_multiplier` の乗算
/// オーバーフローは `Err` で拒否する（fail-closed。ゼロ値検証・
/// `checked_mul` は codex-review 指摘 P2・PR #649）。両方とも未指定なら
/// `Ok(None)`（呼び出し側が既定プロファイルの解決方法〈macOS: 明示指定
/// 必須・非 macOS: M4 Max 既定 + 警告〉を判断する）。
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
            if gpu_core_count == 0 || ideal_groups_multiplier == 0 {
                return Err(
                    "--gpu-core-count と --ideal-groups-multiplier はいずれも正数を \
                     指定する必要がある（ゼロは ideal_groups=0 という成立しない \
                     occupancy 基準を生むため拒否する）"
                        .to_string(),
                );
            }
            gpu_core_count
                .checked_mul(ideal_groups_multiplier)
                .ok_or_else(|| {
                    format!(
                        "--gpu-core-count={gpu_core_count} と \
                         --ideal-groups-multiplier={ideal_groups_multiplier} の積が \
                         u64 の範囲を超える（ideal_groups の算出でオーバーフローする）"
                    )
                })?;
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
    use bench_harness::{Measurement, MeasurementConfig, run as bench_run};

    /// 決定的シード（`gemm_bench.rs::SEED` と同一値。過去 PoC・既存ベンチと
    /// 同じ入力分布に揃える）。
    const SEED: u64 = 0xC0FFEE;

    /// UNIX epoch ミリ秒。`docs/perf/metal-gemm-bottleneck-diagnosis.md`
    /// §2 の GPU 使用率サンプリングループが記録するミリ秒精度タイムスタンプと
    /// 同じ単位に揃え、別ターミナルの ioreg ログと本 example の stdout を
    /// size 別に対応付けられるようにする。
    ///
    /// 以前は秒精度（`epoch_secs`）だったが、M4 Max 実測（`dispatch_auto`・
    /// `MeasurementConfig::default()` = warmup 20 回・計測 20 回）では
    /// size=512/1024/2048 の 1 size 分の総実行区間が 1 秒未満で終わることが
    /// あり、`phase=start`・`phase=result` は言うに及ばず**隣接する別 size
    /// の区間同士**まで同一エポック秒に丸められて衝突し、ioreg サンドイッチ
    /// 方式（開始〜終了マーカーで挟んだ範囲を size に紐付ける手法）が size
    /// 単位で使用率を紐付けられなくなる問題があった（cursor[bot] 指摘
    /// Medium・PR #649）。ミリ秒精度化はこの衝突を実務上の解像度まで縮小する
    /// （完全排除ではない — `--iters` で意図的に区間を伸ばした運用と組み合わせる
    /// 前提。`main` 内のコメント・`docs/perf/metal-gemm-bottleneck-diagnosis.md`
    /// §2「サンプル数下限」節参照）。
    /// システムクロックが UNIX epoch より前を返すことは実運用上あり得ないが、
    /// 診断用の付随情報のため `duration_since` 失敗時も panic させず `0` に
    /// フォールバックする。
    fn epoch_millis() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// `m×n×k` の `dispatch_auto` を計測し [`Measurement`]（中央値・Q1・Q3
    /// 秒。`bench_harness::protocol::run` の出力）をそのまま返す
    /// （`gemm_bench.rs::measure_auto` と同型。呼び出しごとに A・B の
    /// アップロード・C の readback を含む壁時計計測）。中央値のみを
    /// 返さず `Measurement` 全体を返すのは、計測手順（`docs/perf/
    /// metal-gemm-bottleneck-diagnosis.md` §1「中央値/Q1/Q3」）と出力・
    /// §4 記録表を整合させるため（q1/q3 を破棄すると分布のばらつきが
    /// 記録から失われる。codex-review 指摘。PR #649）。
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

    /// 実行対象デバイスの [`DeviceProfile`] を解決する。CLI 明示指定
    /// （`super::parse_device_profile_override`）を**必須**とする。
    ///
    /// 以前は `sysctl -n hw.model` の機種識別子（`Mac16,6` = M4 Max）が
    /// 一致すれば既定の [`DeviceProfile::M4_MAX`] を自動採用していたが、
    /// `MTLDevice` に公開の GPU コア数取得 API は存在せず、機種識別子
    /// だけでは同一機種内の構成差異（binned 版等）まで保証できないため
    /// 誤った occupancy 判定を許してしまう問題があった（codex-review
    /// 指摘 P1・PR #649）。よって自動判定は行わず、未指定は fail-closed
    /// でエラー終了する。
    fn resolve_device_profile() -> Result<DeviceProfile, String> {
        match super::parse_device_profile_override()? {
            Some(profile) => Ok(profile),
            None => Err(
                "GPU デバイスプロファイルの自動判定は行わない（機種識別子だけでは \
                 GPU コア数の一致を保証できないため。codex-review 指摘 P1・PR #649）。\
                 `--gpu-core-count=<N> --ideal-groups-multiplier=<M>` を明示指定して \
                 再実行すること（`docs/real-hardware-verification-env.md` §1 記載の \
                 実機検証環境〈M4 Max〉であれば `--gpu-core-count=40 \
                 --ideal-groups-multiplier=6`）。"
                    .to_string(),
            ),
        }
    }

    /// `--iters=<N>` で warmup・計測回数（`MeasurementConfig::{warmup,iters}`）
    /// を両方 `N` へ引き上げる。未指定なら [`MeasurementConfig::default`]
    /// （20/20。TASK-8.1 下限）を使う。
    ///
    /// `epoch_millis` 化（本関数の docs 参照）だけでは、ioreg 側の外部
    /// サンプリング間隔（`docs/perf/metal-gemm-bottleneck-diagnosis.md` §2 =
    /// 0.5 秒間隔）そのものは変わらないため、1 size あたりの実行区間が
    /// サンプリング間隔の数倍程度なければ「その size の区間に収まる ioreg
    /// サンプル」が 0〜1 個しか得られず median/max の算出が意味を持たない
    /// （cursor[bot] 指摘 Medium・PR #649）。`--iters` は
    /// `MeasurementConfig::new` を経由し TASK-8.1 の 20 回下限を下回る指定を
    /// fail-closed で拒否する（`warmup`・`iters` は `pub` フィールドのため
    /// 構造体リテラル経由だとこの下限検証をバイパスしうる。下限検証を
    /// 迂回しない）。既定を引き上げない（4096 size は 137 GFLOP/回であり、
    /// 既定 20/20 を超えて全呼び出し側の実行時間を伸ばさないよう opt-in に
    /// 限定する）。
    fn parse_iters_override() -> Result<Option<usize>, String> {
        for arg in std::env::args().skip(1) {
            if let Some(v) = arg.strip_prefix("--iters=") {
                let n: usize = v
                    .parse()
                    .map_err(|_| format!("--iters の値が不正: '{v}'"))?;
                return Ok(Some(n));
            }
        }
        Ok(None)
    }

    /// [`MeasurementConfig`] を解決する。`--iters` 未指定なら既定
    /// （20/20）、指定時は warmup・iters とも `N` へ引き上げる
    /// （`parse_iters_override` docs 参照。TASK-8.1 の 20 回下限は
    /// `MeasurementConfig::new` が fail-closed で検証する）。
    fn resolve_measurement_config() -> Result<MeasurementConfig, String> {
        match parse_iters_override()? {
            Some(n) => MeasurementConfig::new(n, n).map_err(|e| e.to_string()),
            None => Ok(MeasurementConfig::default()),
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
        let config = match resolve_measurement_config() {
            Ok(config) => config,
            Err(msg) => {
                eprintln!("{msg}");
                std::process::exit(1);
            }
        };

        let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
        let gemm = MetalGemm::new(&ctx).expect("GEMM パイプラインの構築に失敗した");

        for size in [512usize, 1024, 2048, 4096] {
            let a: SizeAnalytics = analytics::analyze(size, profile);

            // size 別開始マーカー（`epoch_millis`）。別ターミナルで並走させる
            // ioreg 継続サンプリング（`docs/perf/metal-gemm-bottleneck-diagnosis.md`
            // §2。同じくミリ秒精度でログを記録する）と付き合わせ、size ごとの
            // 実行区間を特定できるようにする（codex-review 指摘。PR #649）。
            // 秒精度（`epoch_secs`）からミリ秒精度（`epoch_millis`）へ変更した
            // 理由は `epoch_millis` の docs コメント参照（cursor[bot] 指摘
            // Medium・PR #649: 1 size 分の総実行区間が 1 秒未満だと秒精度では
            // 隣接 size の区間まで同一秒に丸められ紐付けられない）。
            println!("size={size} phase=start epoch_ms={}", epoch_millis());

            // `measurement.{median,q1,q3}_secs` は A・B アップロード＋
            // カーネル実行＋C readback を含む end-to-end 壁時計時間
            // （モジュールドキュメント「転送時間分離を試みて撤回した経緯」
            // 参照。転送時間だけを差し引く手段はクレート内部アクセスを要し
            // 本イシューのスコープ外のため、分離を試みず `wall_measurement`
            // が返す `Measurement`（中央値・Q1・Q3）をそのまま報告する。
            // 中央値のみを使い q1/q3 を破棄すると計測手順〈§1「中央値/
            // Q1/Q3」〉と出力・§4 記録表が不整合になる。codex-review 指摘。
            // PR #649）。
            let measurement = wall_measurement(&gemm, &ctx, size, size, size, &config);
            let wall = measurement.median_secs;

            // `tflops_lower_bound`: 転送時間は非負（`wall`（壁時計秒） ≥
            // kernel_secs`）という不等式のみから導かれる健全な下限値。
            // 実際のカーネル TFLOPS はこれ以上（転送時間の分だけ高い）。
            // 分離を試みないため `_approx`（誤った精度感を与える名称）
            // ではなく `_lower_bound` と明示する。中央値秒を基準に算出し、
            // Q1/Q3 秒はばらつきの記録用にそのまま別途出力する（下記
            // `println!`）。
            let tflops_lower_bound = a.flops as f64 / wall / 1e12;
            // `logical_load_gbs_lower_bound`: `load_bytes_total +
            // store_bytes_total`（`analytics::analyze` 参照。threadgroup
            // 間・K タイル間のキャッシュ再利用を考慮しない論理ロード量の
            // 下限値）を `wall`（壁時計秒）で割った値。**DRAM 実効帯域ではない**
            // （キャッシュヒットにより実際の DRAM トラフィックはこれより
            // 少なく、逆に `wall` が `kernel_secs` 以上であることから
            // 論理ロードスループットの下限でもある）。M4 Max 公称帯域
            // 546GB/s との比較には使わない（codex-review 指摘。PR #649）。
            let logical_load_gbs_lower_bound =
                (a.load_bytes_total + a.store_bytes_total) as f64 / wall / 1e9;

            println!(
                "size={} phase=result epoch_ms={} tile={}x{}x{}({}x{}, staged={}) \
                 actual_groups={} ideal_groups={} barriers_per_tg={} \
                 arithmetic_intensity={:.4} wall_ms={:.4} wall_q1_ms={:.4} wall_q3_ms={:.4} \
                 tflops_lower_bound={:.4} logical_load_gbs_lower_bound={:.4}",
                a.size,
                epoch_millis(),
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
                measurement.q1_secs * 1e3,
                measurement.q3_secs * 1e3,
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
         measurement requires macOS (Apple Silicon). Analytical values (actual_groups/ideal_groups \
         parallelism-saturation heuristic — not true occupancy; see analytics::DeviceProfile docs — \
         barriers / arithmetic intensity) below are computed on any platform.\n"
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
