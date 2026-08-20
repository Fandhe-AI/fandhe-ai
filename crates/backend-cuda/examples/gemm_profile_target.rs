//! イシュー #486（Phase A・A-6。親 #480・ルート #479）: M=N=K=4096 での
//! CUDA GEMM データ再利用崩壊を `nsight-compute`（ncu）で定量診断する
//! ためのプロファイル対象バイナリ。
//!
//! `docs/perf/cuda-floor-remeasurement.md` の実測記録により、CUDA GEMM の
//! f32 最良経路（WMMA(TF32) opt）は 2048→4096 で 6.2995→4.4824 TFLOPS
//! （約 29% 低下）、f16 の `mma.sync` 経路も 12.0214→11.4462 TFLOPS
//! （約 4.8% 低下）と、サイズ増加に伴う明確な性能低下が実測されている。
//! 本バイナリは、この低下の主因（L2 ミス／SMEM バンクコンフリクト／
//! occupancy／命令発行のいずれか）を ncu 実測で特定するための「単一経路・
//! 単一形状をカーネル起動のみ計測する」最小構成の実行対象であり、
//! カーネル本体（`kernels_wmma_opt.rs`／`kernels_mma.rs`）は一切変更しない
//! （実装計画 §6「スコープ外」）。
//!
//! `examples/` に置く理由は既存 `cuda_floor_bench.rs`／`gemm_mma_bench.rs`
//! と同じ: 通常の `cargo test`／CI では実行されず、
//! `cargo build --workspace --all-targets` によるビルド検証のみが CI で
//! 走るようにするため（self-hosted・ホステッド runner をベンチ実行で
//! 占有しない。`ci.md`）。加えて ncu はプロセス単位でカーネル起動を
//! プロファイルするため、`bench_harness::run`（warmup 20 回以上・計測 20
//! 回以上を強制する TASK-8.1 プロトコル）ではなく、起動回数を
//! `--warmup`／`--iters` で自分で制御できる素朴な手動ループを使う
//! （計測プロトコルとしての中央値／Q1/Q3 算出は本バイナリの目的では
//! ない。TFLOPS 出力はあくまで ncu 実測値との突合用の参考値）。
//! `backend-cpu`／`bench-harness` は既に `backend-cuda` の
//! `dev-dependencies`（`examples/cuda_floor_bench.rs` 等が使用）であり、
//! 本ファイルの追加に伴う外部依存の追加は不要（`deps-policy.md` ユーザー
//! 承認事項に該当しない）。ただし本ファイルが使う
//! `backend_cuda::diagnostics`（内部カーネルのタイル定数を返す診断専用
//! 関数群）は `internal-diagnostics` feature（既定 off）でのみコンパイル
//! されるため、`Cargo.toml` の `[[example]] required-features` で本
//! feature を要求する構成にしてある（PR #637 codex-review 指摘の是正:
//! 診断専用の内部値を既定ビルドの通常の公開 API 面から除外するため。
//! `lib.rs` の `diagnostics` モジュール冒頭コメント参照）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo build -p backend-cuda --example gemm_profile_target --release \
//!     --features internal-diagnostics
//! ncu --launch-skip <warmup 起動数> --launch-count <iters> \
//!     --metrics <確定メトリクス名, カンマ区切り> \
//!     ./target/release/examples/gemm_profile_target \
//!     --path wmma_tf32 --size 4096
//! ```
//!
//! `--launch-skip` は `<warmup 起動数>` をそのまま指定する（`+ 1` の調整は
//! **不要**。旧版は `gemm.alloc_output_f32`／`alloc_output_f16`
//! （`gemm.rs`／`gemm_wmma.rs`／`gemm_mma.rs` 各 `alloc_output_*`）が呼ぶ
//! cudarc の `alloc_zeros`（デバイス側ゼロクリア）を ncu の
//! `--launch-skip` が数える「プロファイル対象カーネル起動」に含まれる
//! ものと誤って前提していたが、`alloc_zeros` は内部で `cuMemsetD*Async`
//! 系のドライバ API（`cudarc::driver::safe::core::memset_zeros`。
//! `cuLaunchKernel` を経由しない別経路）を呼ぶ memset 操作であり、
//! ncu がプロファイルする「カーネル起動」の通し番号には含まれない
//! （PR #637 codex-review 指摘。誤った `+ 1` のまま `--launch-skip` を
//! 大きくすると、warmup 回数分だけ計測対象カーネル起動を余分にスキップ
//! してしまい、`--launch-count 5` に対し実際に計測されるのは 4 回になる
//! ため 6 条件すべての診断結果が不完全・誤りになる）。ただし ncu の
//! カーネル起動カウント仕様は cudarc バージョン・実機の compute
//! capability に依存しうるため、この前提を鵜呑みにせず実機の ncu 出力で
//! 対象カーネル名と採取回数が一致することを事前確認する
//! （`docs/perf/cuda-gemm-bottleneck-diagnosis.md` §3.3.1）。この起動
//! 回数はこのバイナリの実行時に `path=... size=... warmup=... iters=...`
//! の直後に `ncu --launch-skip <値>` として明示出力するので、手計算せず
//! その値を使う。
//!
//! `--path`（`wmma_tf32`｜`mma_f16`。必須）・`--size`（`1024`｜`2048`｜
//! `4096`。必須）は固定 allowlist との完全一致のみ受理する（`.claude/rules/
//! security.md` A03「外部入力の検証」。シェル呼び出し・文字列展開は行わ
//! ない）。`--iters`（既定 5）・`--warmup`（既定 2）は正の整数のみ受理する。
//! 採取手順・実測記録・主因分析は `docs/perf/cuda-gemm-bottleneck-diagnosis.md`
//! を参照。
//!
//! `--b-pad <N>`（イシュー #743。`--path wmma_tf32` 限定・任意）を指定
//! すると、`gemm.launch_wmma_tf32`（本番経路。固定 `WMMA_TF32_STAGED_B_PAD`
//! 既定値）の代わりに `backend_cuda::diagnostics::render_wmma_tf32_staged`
//! （**static** 共有メモリ変種。`WmmaTf32StagedKernelConfig { b_pad: N,
//! ..default_tf32_staged() }`）でコンパイル・起動する。static 変種は本番
//! カーネルと同一の `__shared__` 宣言・同一 occupancy を持つため、
//! `b_pad` の差分だけを ncu で切り分けられる（PR #769 Bugbot 指摘 review
//! id 4978031442 の是正: 旧実装は動的共有メモリ変種
//! `render_wmma_tf32_staged_dyn`〈`c_tile` を `as_tile`/`bs_tile` へ
//! エイリアスし約 29KiB・3 blocks/SM〉を使っており、本番の静的変種
//! 〈44.8〜45.6KiB・2 blocks/SM〉との比較に dyn/static の occupancy 差が
//! 交絡していた）。SMEM バンクコンフリクト対策候補（`docs/perf/
//! cuda-gemm-wmma-tf32-staged-bank-conflict.md` §3「採否基準」1「ncu で
//! 4096 の ld バンクコンフリクトが有意に減少していること」）を ncu で
//! 実測するための経路であり、本番カーネルソース（`kernels_wmma_opt.rs`）
//! 自体は変更しない（モジュール冒頭の「カーネル本体は一切変更しない」
//! 契約に対し、config 経由で既存の static レンダリング関数を選べるように
//! する点のみが差分）。未指定時（既定）は従来どおり本番経路のみを計測し、
//! 本フィールド追加による挙動変化はない。
//!
//! `render_wmma_tf32_staged`（static）は展開前に `validate_wmma_tf32_
//! staged_config` で静的共有メモリ予算（48KiB。`MMA_STATIC_SMEM_LIMIT_
//! BYTES`）を fail-closed 検査するため、`b_pad` が予算を超える場合は
//! `--b-pad` 自体を非 0 終了で拒否する。この場合、動的共有メモリ変種
//! （opt-in budget を要する `render_wmma_tf32_staged_dyn`）でしか計測
//! できない `b_pad` は本バイナリではなく `gemm_wmma_tf32_staged_stages_
//! bench.rs`〈#742〉側の専用計測コードを使う（本バイナリで dyn 変種へ
//! 静かにフォールバックすると、上記の occupancy 交絡が再発するため
//! 意図的にフォールバックしない）。
//!
//! `--b-pad` を `--path mma_f16` と併用した場合・値が
//! `validate_wmma_tf32_staged_padding` の制約（4 要素倍数・タイル幅以上・
//! 余剰 32 要素以下）を満たさない場合・静的 SMEM 予算を超える場合は、
//! いずれも fail-closed で非 0 終了する（1 番目は CLI 引数検証・
//! 2〜3 番目は `render_wmma_tf32_staged` の `Result::Err`）。
//!
//! `CudaDevice::new`／`CudaGemm::new`／`CudaMmaGemm::new`／
//! `--path wmma_tf32` 選択カーネル不在（`--b-pad` 未指定時は
//! `wmma_tf32_staged_available() == false`〈PR #769 codex-review P1
//! 指摘対応。「実行手順」節参照〉、`--path mma_f16` は mma.sync f16
//! カーネル不在）のいずれの失敗も、既定では
//! 理由を表示したうえで `panic!` を使わず `std::process::exit(1)`（非 0
//! 終了）する。`docs/perf/cuda-gemm-bottleneck-diagnosis.md` §3.3 の採取
//! ループが `set -o pipefail` で非 0 終了を検知する fail-closed 契約のため
//! （PR #637 codex-review 指摘）。CUDA 非搭載環境として意図的にスキップ
//! したい場合（手元検証時等）のみ `--allow-missing-driver` を明示指定する
//! と `CudaDevice::new` の `CudaError::DriverUnavailable` に限り終了コード
//! 0 でスキップする（`Args::allow_missing_driver` 参照。既定は `false`）。
//! CI の `cargo build --workspace --all-targets` はビルドのみなので
//! この実行時分岐は CI に影響しない。

use std::time::Instant;

use backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaMmaGemm, diagnostics};
use bench_harness::rng::Xorshift64Star;
use cudarc::driver::sys::CUdevice_attribute;
use half::f16;

/// 決定的シード（`cuda_floor_bench.rs`・`gemm_mma_bench.rs` と同一値。
/// 過去実測・他バックエンドベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

/// `Path::WmmaTf32`／`Path::MmaF16` の各分岐が warmup ループの直前に呼ぶ
/// `gemm.alloc_output_f32`／`alloc_output_f16`（`gemm.rs`／`gemm_wmma.rs`／
/// `gemm_mma.rs`）は cudarc の `alloc_zeros` を経由するが、内部で呼ぶのは
/// `cuMemsetD*Async` 系のドライバ API（`memset_zeros`）であり、
/// `cuLaunchKernel` を経由しない別経路のため ncu の `--launch-skip` が
/// 数える「プロファイル対象カーネル起動」には含まれない。よって
/// `--launch-skip` の算出に加算は不要（値は常に 0）。定数として残すのは
/// 過去（PR #637 時点）に「memset も 1 回のカーネル起動として数える」と
/// 誤って前提し `--launch-skip` を `warmup + 1` にしていた経緯を記録し、
/// 将来同じ誤りを繰り返さないためのコメント錨点として。モジュール冒頭
/// ドキュメンテーションコメント「実行手順」・
/// `docs/perf/cuda-gemm-bottleneck-diagnosis.md` §3.3／§3.3.1 参照。
const ALLOC_ZEROS_LAUNCHES: usize = 0;

/// 対象経路（CLI `--path` の allowlist）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    WmmaTf32,
    MmaF16,
}

impl Path {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "wmma_tf32" => Some(Self::WmmaTf32),
            "mma_f16" => Some(Self::MmaF16),
            _ => None,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::WmmaTf32 => "wmma_tf32",
            Self::MmaF16 => "mma_f16",
        }
    }
}

/// CLI から確定した実行設定。`--path`／`--size` は固定 allowlist との
/// 完全一致のみ受理し、シェル呼び出し・文字列展開は一切行わない
/// （`.claude/rules/security.md` A03 対応）。
struct Args {
    path: Path,
    size: u32,
    iters: usize,
    warmup: usize,
    /// `warmup + iters`（`checked_add` 済み）。`main` 側で未検査の再加算を
    /// せず、この検証済み値をそのまま消費させるためのフィールド
    /// （PR #637 codex-review 指摘: `parse_args` が唯一の `Args` 構築点
    /// であることに未来の変更が依存しないようにする）。
    total_launches: usize,
    /// `warmup + ALLOC_ZEROS_LAUNCHES`（`checked_add` 済み。
    /// `ALLOC_ZEROS_LAUNCHES` は常に 0 のため実質 `warmup` と同値だが、
    /// `--launch-skip` 算出の唯一の構築点を `parse_args` に保つ意図で
    /// フィールドとして残す）。`--launch-skip` 値の算出に使う（同上）。
    launch_skip: usize,
    /// `--b-pad` 指定値（イシュー #743。`--path wmma_tf32` 限定）。`None`
    /// （既定）は本番経路（`gemm.launch_wmma_tf32`）を、`Some(v)` は
    /// `WmmaTf32StagedKernelConfig { b_pad: v, .. }` の **static** 共有
    /// メモリ変種（本番と同一レイアウト・同一 occupancy）を起動する
    /// （モジュール冒頭ドキュメンテーションコメント参照）。
    b_pad: Option<u32>,
    /// `CudaDevice::new` が `CudaError::DriverUnavailable`（CUDA 非搭載
    /// 環境）を返した場合に終了コード 0 でスキップすることを明示的に
    /// 許可するフラグ。既定は `false`（非 0 終了）。`docs/perf/`
    /// `cuda-gemm-bottleneck-diagnosis.md` §3.3 の `set -o pipefail`
    /// 採取ループは非 0 終了で失敗検知する fail-closed 契約のため、
    /// 既定でドライバ不在を成功終了扱いにすると 6 条件すべてが
    /// カーネル未起動・空ログのまま正常終了として見逃されうる
    /// （PR #637 codex-review 指摘）。CI はこの example をビルドするのみ
    /// で実行しないため、CUDA 非搭載環境で意図的にスキップしたい場合
    /// （手元検証時等）だけこのフラグを明示指定させる。
    allow_missing_driver: bool,
}

const USAGE: &str = "usage: gemm_profile_target --path {wmma_tf32|mma_f16} --size {1024|2048|4096} [--iters N] [--warmup N] [--b-pad N (wmma_tf32 only)] [--allow-missing-driver]";

/// `--b-pad` は static 共有メモリ変種（`render_wmma_tf32_staged`。イシュー
/// #743）でのみ意味を持つ TF32 staged 固有のパラメータのため、
/// `--path mma_f16` との併用を拒否する（イシュー #743。`.claude/rules/
/// security.md` A03「外部入力の検証」。無視して静かに no-op にすると、
/// オペレーターが指定した候補値が計測に反映されない誤計測を fail-closed
/// に検知できない）。`parse_args` 本体は `std::env::args` を直接読むため
/// 単体テストから差し替えられない（`Path::parse` と同じ理由）が、この
/// 純粋な検査ロジックは分離することで直接テストできる。
fn validate_b_pad_requires_wmma_tf32(b_pad: Option<u32>, path: Option<Path>) -> Result<(), String> {
    if b_pad.is_some() && path != Some(Path::WmmaTf32) {
        return Err("--b-pad は --path wmma_tf32 の場合のみ指定できる".to_string());
    }
    Ok(())
}

/// `std::env::args` のみで CLI 引数をパースする（依存追加なし。実装計画
/// §3 Step 1「CLI 引数を `std::env::args` のみでパースする」）。
/// allowlist 完全一致以外・数値パース失敗・想定外形状はいずれも `Err` で
/// 拒否し、呼び出し側（`main`）が usage を表示して非 0 終了する
/// （fail-closed。`.claude/rules/security.md` A03）。
fn parse_args() -> Result<Args, String> {
    let mut path: Option<Path> = None;
    let mut size: Option<u32> = None;
    let mut iters: usize = 5;
    let mut warmup: usize = 2;
    let mut allow_missing_driver = false;
    let mut b_pad: Option<u32> = None;

    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--path" => {
                let v = it.next().ok_or("--path には値が必要")?;
                path = Some(Path::parse(&v).ok_or_else(|| {
                    format!("--path は 'wmma_tf32' または 'mma_f16' のみ受理する（指定値: '{v}'）")
                })?);
            }
            "--size" => {
                let v = it.next().ok_or("--size には値が必要")?;
                let parsed: u32 = v
                    .parse()
                    .map_err(|_| format!("--size は正の整数のみ受理する（指定値: '{v}'）"))?;
                if !matches!(parsed, 1024 | 2048 | 4096) {
                    return Err(format!(
                        "--size は 1024/2048/4096 のみ受理する（指定値: '{v}'）"
                    ));
                }
                size = Some(parsed);
            }
            "--iters" => {
                let v = it.next().ok_or("--iters には値が必要")?;
                iters = v
                    .parse::<usize>()
                    .map_err(|_| format!("--iters は正の整数のみ受理する（指定値: '{v}'）"))?;
                if iters == 0 {
                    return Err("--iters は 1 以上を指定する".to_string());
                }
            }
            "--warmup" => {
                let v = it.next().ok_or("--warmup には値が必要")?;
                warmup = v.parse::<usize>().map_err(|_| {
                    format!("--warmup は 0 以上の整数のみ受理する（指定値: '{v}'）")
                })?;
            }
            "--allow-missing-driver" => {
                allow_missing_driver = true;
            }
            "--b-pad" => {
                let v = it.next().ok_or("--b-pad には値が必要")?;
                b_pad = Some(
                    v.parse::<u32>()
                        .map_err(|_| format!("--b-pad は正の整数のみ受理する（指定値: '{v}'）"))?,
                );
            }
            other => return Err(format!("未知の引数: '{other}'")),
        }
    }

    validate_b_pad_requires_wmma_tf32(b_pad, path)?;

    // `--warmup`／`--iters` は usize へ変換できれば無制限に受理していたため、
    // `args.warmup + args.iters`（total_launches 算出）や
    // `args.warmup + ALLOC_ZEROS_LAUNCHES`（`--launch-skip` 値の算出）の
    // 未検査加算が極端な入力で debug build では panic、release build では
    // wrap して `--launch-skip` 等の表示・起動有無判定を不正にしうる
    // （PR #637 codex-review 指摘）。CLI 境界で `checked_add` により
    // オーバーフローを入力エラーとして拒否し、fail-closed に非 0 終了させる
    // （`.claude/rules/security.md` A03: 外部入力の検証）。
    let total_launches = warmup
        .checked_add(iters)
        .ok_or_else(|| "--warmup と --iters の合計が usize の範囲を超える".to_string())?;
    let launch_skip = warmup.checked_add(ALLOC_ZEROS_LAUNCHES).ok_or_else(|| {
        "--warmup が大きすぎて --launch-skip 値（+alloc_zeros memset 起動分）を算出できない"
            .to_string()
    })?;

    Ok(Args {
        path: path.ok_or("--path は必須")?,
        size: size.ok_or("--size は必須")?,
        iters,
        warmup,
        total_launches,
        launch_skip,
        b_pad,
        allow_missing_driver,
    })
}

fn tflops(size: u32, secs: f64) -> f64 {
    let flops = 2.0 * (size as f64).powi(3);
    flops / secs / 1e12
}

/// MFA（`GEMMDescriptor.swift:255-321`）流の occupancy 判定式
/// 「actualGroups = ceil(M/タイル) × ceil(N/タイル) vs idealGroups =
/// コア数 × 係数」を CUDA 向けに読み替えた「実測 occupancy 事前計算の
/// 材料」。ncu の `sm__warps_active.avg.pct_of_peak_sustained_active`
/// 実測値との突合用に、ブロック単位のタイル分割数と SM 数から求まる
/// blocks/SM 比を起動時に print する（実装計画 §3 Step 3）。
fn print_occupancy_estimate(path: Path, size: u32, sm_count: Option<u32>) {
    // タイル定数はハードコード転記ではなく `backend_cuda::diagnostics` の
    // 診断専用安定関数（`wmma_tf32_opt_block_tile`／`mma_f16_block_tile`。
    // `lib.rs` 参照）経由で取得する。カーネル側モジュール自体
    // （`mod kernels_wmma_opt`／`mod kernels_mma`）・生の内部定数は非公開の
    // ままとし（PR #637 codex-review 指摘: 生の定数を crate root へ直接
    // `pub use` すると内部実装詳細がそのまま公開 API 互換性の対象に
    // なってしまう）、`diagnostics` モジュールが唯一の公開境界となる。
    // カーネル側の値を変更しないという本イシューのスコープ
    // （実装計画 §6）を保ちつつ、出典側の値変更がコンパイル時に機械的に
    // ここへ反映される（手元転記だと出典との乖離を検知できず、occupancy
    // estimate が静かに誤った参考値を出し続けるおそれがあった）。
    let (block_m, block_n): (u32, u32) = match path {
        Path::WmmaTf32 => diagnostics::wmma_tf32_opt_block_tile(),
        Path::MmaF16 => diagnostics::mma_f16_block_tile(),
    };
    let actual_blocks = size.div_ceil(block_m) as u64 * size.div_ceil(block_n) as u64;
    match sm_count {
        Some(sm) if sm > 0 => {
            let blocks_per_sm = actual_blocks as f64 / sm as f64;
            println!(
                "occupancy estimate: path={} size={size} block_tile={block_m}x{block_n} \
                 actual_blocks={actual_blocks} sm_count={sm} blocks_per_sm={blocks_per_sm:.3} \
                 (MFA-derived actualGroups/idealGroups 読み替え。実測値は ncu \
                 sm__warps_active.avg.pct_of_peak_sustained_active と突合する)",
                path.as_str()
            );
        }
        _ => {
            println!(
                "occupancy estimate: path={} size={size} block_tile={block_m}x{block_n} \
                 actual_blocks={actual_blocks} sm_count=n/a（デバイス属性取得失敗のため \
                 blocks_per_sm は算出できない）",
                path.as_str()
            );
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            std::process::exit(1);
        }
    };

    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) if args.allow_missing_driver => {
            println!(
                "backend-cuda gemm_profile_target: CUDA driver unavailable ({detail}); \
                 skipping because --allow-missing-driver was specified."
            );
            return;
        }
        Err(CudaError::DriverUnavailable { detail }) => {
            // 既定では非 0 終了させる（fail-closed）。ここで終了コード 0 に
            // すると `docs/perf/cuda-gemm-bottleneck-diagnosis.md` §3.3 の
            // 採取ループ（`set -o pipefail` で非 0 終了を検知する契約）が
            // ドライバ不在による未起動を見逃し、6 条件すべてがカーネル
            // 未起動・空ログのまま正常終了扱いで通過してしまう
            // （PR #637 codex-review 指摘）。CI はこの example をビルド
            // するだけで実行しないため、実行時に成功終了させる必要はない。
            // CUDA 非搭載環境で意図的にスキップしたい場合は `--allow-
            // missing-driver` を明示指定させる（採取用の既定動作とは
            // 分離した opt-in）。
            eprintln!(
                "backend-cuda gemm_profile_target: CUDA driver unavailable ({detail}); \
                 aborting because the target kernel never launched. Pass \
                 --allow-missing-driver to skip intentionally in CUDA-less environments."
            );
            std::process::exit(1);
        }
        Err(other) => {
            // `DriverUnavailable`（CUDA 非搭載環境。上の 2 分岐参照。既定
            // では非 0 終了、`--allow-missing-driver` 指定時のみ skip）
            // 以外の `CudaDevice::new` 失敗（ドライバ不整合・コンテキスト
            // 生成失敗等）は「CUDA 実行環境自体が無い」わけではない
            // 異常系であり、`--allow-missing-driver` の対象外。対象
            // カーネルは一度も起動していない。ここで終了コード 0 にすると
            // `docs/perf/cuda-gemm-bottleneck-diagnosis.md` §3.3 の採取
            // ループ（`set -o pipefail` で非 0 終了を検知する fail-closed
            // 契約）がこの未起動を見逃し、6 条件すべてを空のまま成功扱いで
            // 通過してしまう（PR #637 codex-review 指摘）。`CudaGemm::new`
            // 失敗時と同じ理由で非 0 終了させる。
            eprintln!(
                "backend-cuda gemm_profile_target: CudaDevice::new failed ({other}); \
                 aborting because the target kernel never launched (this is not an \
                 environment-not-present skip)."
            );
            std::process::exit(1);
        }
    };
    println!(
        "device: name={} compute_capability={:?}",
        device.name(),
        device.compute_capability()
    );

    // SM 数（`docs/spec` A-2/#482 の実測が未完了でも、本バイナリ単独で
    // occupancy 概算材料を出せるようにする。実装計画 §3 Step 1「デバイス
    // 属性（SM あたり最大スレッド数・SMEM 容量・レジスタ数）」の SM 数部分。
    // 取得失敗時は `None` にフォールバックし occupancy 出力は n/a とする
    // （panic させない。`device.rs::compute_units` と同じ fail-soft 方針）。
    let sm_count = device
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .ok()
        .and_then(|v| u32::try_from(v).ok());
    print_occupancy_estimate(args.path, args.size, sm_count);

    println!(
        "path={} size={} warmup={} iters={}",
        args.path.as_str(),
        args.size,
        args.warmup,
        args.iters
    );
    // `alloc_output_f32`／`alloc_output_f16`（各 `Path` 分岐で warmup
    // ループ直前に 1 回呼ばれる）の `alloc_zeros` memset は `cuLaunchKernel`
    // を経由しないため `--launch-skip` に含めない（`ALLOC_ZEROS_LAUNCHES`
    // 定義コメント参照）。手計算による転記ミスを防ぐため、実行時にそのまま
    // 使える値を出力する。
    println!(
        "ncu --launch-skip {} --launch-count {} \
         (warmup={} + alloc_zeros memset launches={})",
        args.launch_skip, args.iters, args.warmup, ALLOC_ZEROS_LAUNCHES
    );

    let mut rng = Xorshift64Star::new(SEED);
    let (m, n, k) = (args.size, args.size, args.size);
    // `args.total_launches`（`warmup + iters`）が 0 になる分岐は存在しない
    // （レビュー指摘: 以前あった `if total_launches == 0 { ... return; }` は
    // 到達不能なデッドコードだった）。`parse_args` が `--iters == 0` を
    // 明示的に `Err` で拒否するため（本ファイル `parse_args` 参照）、
    // `total_launches = warmup + iters` は `warmup` の値に関わらず常に 1
    // 以上になる。この不変条件が将来の `parse_args` 変更で崩れないことを
    // ここで検査だけしておく（本番経路の値を分岐させない `debug_assert!`。
    // release ビルドでは最適化除去される）。
    debug_assert!(
        args.total_launches >= 1,
        "parse_args が --iters==0 を拒否する契約が崩れている"
    );

    match args.path {
        Path::WmmaTf32 => {
            let gemm = match CudaGemm::new(&device) {
                Ok(g) => g,
                Err(e) => {
                    // `CudaDevice::new` 成功後（＝実機で CUDA デバイスは確立
                    // 済み）に `CudaGemm::new` が失敗するのは NVRTC コンパイル
                    // 失敗・対象 compute capability 非対応等の異常系であり、
                    // 「CUDA 実行環境自体が無い」ケース（上の `CudaDevice::new`
                    // 失敗 skip）とは区別する。ここで exit 0 にすると
                    // `docs/perf/cuda-gemm-bottleneck-diagnosis.md` §3.3 の
                    // 採取ループ（`set -o pipefail` で非 0 終了を検知する契約）
                    // が対象カーネル未起動を見逃し、次の条件へ進んでしまう
                    // （PR #637 codex-review 指摘）。よって非 0 終了させる。
                    eprintln!(
                        "backend-cuda gemm_profile_target: tiled/WMMA(TF32) kernel unavailable ({e}); \
                         aborting because the target kernel never launched (this is not an \
                         environment-adaptive skip)."
                    );
                    std::process::exit(1);
                }
            };
            // 実測時に誤ってフォールバック版（基本 WMMA(TF32)）を
            // プロファイルする事故を防ぐため、opt カーネルの可用性を
            // 明示する（`cuda_floor_bench.rs` の先例と同じ判断）。
            //
            // `CudaGemm::launch_wmma_tf32` は opt カーネル未ロード時に基本
            // カーネルへ自動フォールバックし（両方未ロードの場合のみ
            // `CudaError::WmmaUnavailable` を返す。`gemm.rs::launch_wmma_tf32`
            // 参照）、本バイナリはこの経路には依存しない。「単一経路・単一
            // 形状のみを計測する」契約（モジュール冒頭ドキュメンテーション
            // コメント参照）上、opt カーネル不在時に基本カーネルへ黙って
            // フォールバックして計測を続けると、診断対象と異なるカーネルの
            // ncu 結果を正常計測として生成してしまう（PR #637 codex-review
            // 指摘）。
            //
            // ここでの終了コードは上の `CudaDevice::new`／`CudaGemm::new`
            // 失敗時の扱いと揃える: `CudaDevice::new` の
            // `CudaError::DriverUnavailable` は既定でも非 0 終了とし
            // （`--allow-missing-driver` を明示指定した場合のみ opt-in で
            // exit 0 スキップ）、それ以外（`DriverUnavailable` 以外の
            // `CudaDevice::new` エラー・`CudaGemm::new`／`CudaMmaGemm::new`
            // 失敗）は常に非 0 終了する。ここに到達するのは CUDA デバイス・
            // `CudaGemm::new` 自体は成立した上で opt カーネルの NVRTC
            // ロードのみが失敗した場合
            // （`wmma_tf32_opt_unavailable_reason()` が理由を保持している
            // ことからも NVRTC コンパイル失敗等の異常系であることが分かる）
            // であり、オペレーターは opt カーネルをプロファイルする意図で
            // このバイナリを実機（GPU が動く環境）で起動している。この場合
            // に exit 0 で「正常終了」に見せると、ncu 実行スクリプト側が
            // 失敗を検知できず基本カーネルの結果を opt カーネルの正常計測
            // として記録表へ転記してしまう（PR #637 codex-review 指摘の
            // 「実行手順もこの終了状態を検査しないため誤ったボトルネック分析
            // に進みうる」の直接原因）。よって非 0 終了させ、§3.3 の採取
            // ループ（`docs/perf/cuda-gemm-bottleneck-diagnosis.md`）側の
            // `set -o pipefail` と組み合わせて誤計測をループ内で検知
            // させる。
            let a = rng.fill_vec((m as usize) * (k as usize));
            let b = rng.fill_vec((k as usize) * (n as usize));
            let (a_dev, b_dev) = gemm
                .upload_f32(&a, &b)
                .expect("wmma_tf32 upload must succeed on CUDA-equipped runner");
            // `alloc_output_f32`（`gemm.rs`）は cudarc `alloc_zeros` 経由で
            // memset を行うが、`cuLaunchKernel` を経由しない別経路の
            // ドライバ API のため ncu の起動通し番号には含まれず
            // `--launch-skip` は `args.warmup` のみでよい
            // （`ALLOC_ZEROS_LAUNCHES` 定義コメント・実行時出力・
            // モジュール冒頭「実行手順」参照）。
            let mut c_dev = gemm
                .alloc_output_f32(m, n)
                .expect("wmma_tf32 output allocation must succeed on CUDA-equipped runner");

            // `--b-pad` 未指定（既定）は従来どおり本番経路
            // （`gemm.launch_wmma_tf32`。固定 `WMMA_TF32_STAGED_B_PAD`）を
            // 計測する。指定時（イシュー #743）は `render_wmma_tf32_staged`
            // 経由の **static** 共有メモリ変種をコンパイル・起動し、SMEM
            // バンクコンフリクト対策候補を ncu で実測できるようにする
            // （モジュール冒頭ドキュメンテーションコメント参照）。
            //
            // PR #769 Bugbot 指摘（Medium・review id 4978031442）の是正:
            // 旧実装は動的共有メモリ変種（`render_wmma_tf32_staged_dyn`。
            // `c_tile` を `as_tile`/`bs_tile` へエイリアスし約 29KiB・
            // 3 blocks/SM）を使っていたため、既定（本番・static・
            // 44.8〜45.6KiB・2 blocks/SM）との ncu 比較が `b_pad` の差と
            // dyn/static の occupancy 差を交絡していた。static 変種は
            // 本番と同一の `__shared__` 宣言・同一 occupancy のため、
            // `b_pad` のみを変数化して切り分けられる。`render_wmma_tf32_
            // staged` は `validate_wmma_tf32_staged_config` で 48KiB の
            // 静的 SMEM 予算を fail-closed 検査するため、予算を超える
            // `b_pad`（static では計測不能な候補）は dyn 変種へ静かに
            // フォールバックせずここで非 0 終了させる（フォールバックする
            // と上記の occupancy 交絡が再発するため、意図的にしない。
            // 予算超過の候補は `gemm_wmma_tf32_staged_stages_bench.rs`
            // 〈#742〉側の専用計測コードを使う）。
            if let Some(b_pad) = args.b_pad {
                let cfg = diagnostics::WmmaTf32StagedKernelConfig {
                    b_pad,
                    ..diagnostics::WmmaTf32StagedKernelConfig::default_tf32_staged()
                };
                let compiled = match diagnostics::render_wmma_tf32_staged(&cfg)
                    .and_then(|rendered| rendered.compile(&device))
                {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!(
                            "backend-cuda gemm_profile_target: --b-pad {b_pad} static-SMEM \
                             variant unavailable ({e}); aborting because the target kernel \
                             never launched (this is not an environment-adaptive skip). If \
                             this b_pad exceeds the 48KiB static shared memory budget, use \
                             gemm_wmma_tf32_staged_stages_bench.rs (#742) instead, which is \
                             purpose-built for the dynamic-SMEM variant."
                        );
                        std::process::exit(1);
                    }
                };
                println!(
                    "wmma_tf32 staged (static, b_pad={b_pad}) kernel: compiled (used for this \
                     run's launches; production-identical __shared__ layout/occupancy, so this \
                     result is directly comparable to a separate --b-pad-less run of this same \
                     binary)."
                );

                let stream = device.stream();
                for _ in 0..args.warmup {
                    compiled
                        .launch_tf32_staged(stream, &a_dev, &b_dev, &mut c_dev, m, n, k)
                        .expect(
                            "wmma_tf32 staged (static) warmup launch must succeed on \
                             CUDA-equipped runner",
                        );
                }
                let start = Instant::now();
                for _ in 0..args.iters {
                    compiled
                        .launch_tf32_staged(stream, &a_dev, &b_dev, &mut c_dev, m, n, k)
                        .expect(
                            "wmma_tf32 staged (static) measured launch must succeed on \
                             CUDA-equipped runner",
                        );
                }
                let elapsed = start.elapsed().as_secs_f64();
                let per_iter_secs = elapsed / args.iters as f64;
                println!(
                    "wall-clock (wmma_tf32 static b_pad={b_pad}, launch-only, {} iters): \
                     total={elapsed:.6}s per_iter={per_iter_secs:.6}s tflops={:.4} \
                     (ncu 実測値との突合用の参考値。ncu 実行中は計測区間にプロファイラの \
                     オーバーヘッドが乗るため単体実行時の数値とは一致しない)",
                    args.iters,
                    tflops(args.size, per_iter_secs)
                );
            } else {
                // 実測時に誤ってフォールバック版（opt／基本 WMMA(TF32)）を
                // プロファイルする事故を防ぐため、staged カーネルの可用性を
                // 明示する（`cuda_floor_bench.rs` の先例と同じ判断）。
                //
                // `CudaGemm::launch_wmma_tf32` は staged → opt → basic の
                // 3 段選択（`gemm.rs::launch_wmma_tf32` 参照。staged は
                // `wmma_tf32_staged_available() && wmma_tf32_staged_
                // alignment_ok(n, k)` の場合のみ選ばれる）。本バイナリの
                // `--size` は `{1024, 2048, 4096}` の allowlist に限定され
                // `m == n == k == size` かつ全て 4 の倍数のため
                // （`wmma_tf32_staged_alignment_ok` は `n%4==0 && k%4==0`
                // のみを見る。上の `(m, n, k)` 束縛参照）、alignment 条件は
                // 本バイナリでは常に成立する。よって `wmma_tf32_staged_
                // available()` のみを fail-closed 条件とすれば staged 選択
                // を保証できる。
                //
                // PR #769 codex-review P1 指摘（discussion_r3817849763）の
                // 是正: 旧実装は `wmma_tf32_opt_available()` のみを確認して
                // いたため、staged のコンパイルだけが失敗し opt が利用可能
                // な環境では、`--b-pad` 未指定側が opt カーネル（3a）へ
                // 静かにフォールバックする一方 `--b-pad` 指定側は常に
                // static staged 変種（3b）を使う非対称が生じ、両者の差分を
                // `b_pad` の効果と誤認しうる（`docs/perf/
                // cuda-gemm-wmma-tf32-staged-bank-conflict.md` の採否基準
                // へ誤った値が使われる恐れ）。staged 不在時は opt へ
                // フォールバックせず非 0 終了し、両側が同一の static
                // staged 実装で比較されることを保証する。
                //
                // ここでの終了コードは上の `CudaDevice::new`／
                // `CudaGemm::new` 失敗時の扱いと揃える: `CudaDevice::new`
                // の `CudaError::DriverUnavailable` は既定でも非 0 終了と
                // し（`--allow-missing-driver` を明示指定した場合のみ
                // opt-in で exit 0 スキップ）、それ以外
                // （`DriverUnavailable` 以外の `CudaDevice::new` エラー・
                // `CudaGemm::new`／`CudaMmaGemm::new` 失敗）は常に非 0
                // 終了する。ここに到達するのは CUDA デバイス・
                // `CudaGemm::new` 自体は成立した上で staged カーネルの
                // NVRTC ロードのみが失敗した場合
                // （`wmma_tf32_staged_unavailable_reason()` が理由を保持
                // していることからも NVRTC コンパイル失敗等の異常系で
                // あることが分かる）であり、オペレーターは staged カーネル
                // をプロファイルする意図でこのバイナリを実機（GPU が
                // 動く環境）で起動している。この場合に exit 0 で「正常
                // 終了」に見せると、ncu 実行スクリプト側が失敗を検知
                // できずフォールバックカーネルの結果を staged カーネルの
                // 正常計測として記録表へ転記してしまう（PR #637
                // codex-review 指摘の「実行手順もこの終了状態を検査しない
                // ため誤ったボトルネック分析に進みうる」と同型のリスク）。
                // よって非 0 終了させ、§3.3 の採取ループ
                // （`docs/perf/cuda-gemm-bottleneck-diagnosis.md`）側の
                // `set -o pipefail` と組み合わせて誤計測をループ内で
                // 検知させる。
                if gemm.wmma_tf32_staged_available() {
                    println!(
                        "wmma_tf32 staged kernel: AVAILABLE (used for this run's launches; \
                         same static __shared__ layout/occupancy as a --b-pad run of this \
                         binary)."
                    );
                } else {
                    eprintln!(
                        "backend-cuda gemm_profile_target: wmma_tf32 staged kernel unavailable \
                         ({}); aborting instead of falling back to the opt/basic (non-staged) \
                         WMMA(TF32) kernel, because ncu results for the fallback kernel would \
                         not be directly comparable to a --b-pad run (which always uses the \
                         static staged variant), and could be misattributed to b_pad's effect.",
                        gemm.wmma_tf32_staged_unavailable_reason()
                            .unwrap_or("unknown reason")
                    );
                    std::process::exit(1);
                }

                // ncu は `--launch-skip <warmup 起動数 + alloc_zeros memset
                // 起動数> --launch-count <iters>` でこのループ内のカーネル
                // 起動番号を直接指定してプロファイルする（モジュール冒頭
                // ドキュメンテーションコメント「実行手順」参照）。
                // `launch_wmma_tf32` は呼び出しごとに内部で
                // `stream.synchronize()` するため
                // （`gemm.rs::launch_wmma_tf32` 末尾参照）、ここでの追加
                // 同期は不要。
                for _ in 0..args.warmup {
                    gemm.launch_wmma_tf32(&a_dev, &b_dev, &mut c_dev, m, n, k)
                        .expect("wmma_tf32 warmup launch must succeed on CUDA-equipped runner");
                }
                let start = Instant::now();
                for _ in 0..args.iters {
                    gemm.launch_wmma_tf32(&a_dev, &b_dev, &mut c_dev, m, n, k)
                        .expect("wmma_tf32 measured launch must succeed on CUDA-equipped runner");
                }
                let elapsed = start.elapsed().as_secs_f64();
                let per_iter_secs = elapsed / args.iters as f64;
                println!(
                    "wall-clock (wmma_tf32, launch-only, {} iters): total={elapsed:.6}s \
                     per_iter={per_iter_secs:.6}s tflops={:.4} (ncu 実測値との突合用の参考値。\
                     ncu 実行中は計測区間にプロファイラのオーバーヘッドが乗るため単体実行時の \
                     数値とは一致しない)",
                    args.iters,
                    tflops(args.size, per_iter_secs)
                );
            }
        }
        Path::MmaF16 => {
            let gemm = match CudaMmaGemm::new(&device) {
                Ok(g) => g,
                Err(e) => {
                    // 上の `Path::WmmaTf32` 分岐と同じ理由（`CudaDevice::new`
                    // 成立後の `CudaMmaGemm::new` 失敗は NVRTC コンパイル失敗等
                    // の異常系。fail-closed 採取ループが検知できるよう非 0
                    // 終了させる。PR #637 codex-review 指摘）。
                    eprintln!(
                        "backend-cuda gemm_profile_target: mma.sync f16 kernel unavailable ({e}); \
                         aborting because the target kernel never launched (this is not an \
                         environment-adaptive skip)."
                    );
                    std::process::exit(1);
                }
            };

            let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
            let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));
            let (a_dev, b_dev) = gemm
                .upload_f16(&a, &b)
                .expect("mma_f16 upload must succeed on CUDA-equipped runner");
            // `alloc_output_f16`（`gemm_mma.rs`）も `alloc_zeros` 経由の
            // memset だが `cuLaunchKernel` を経由しないため ncu の起動
            // 通し番号に含まれない（`ALLOC_ZEROS_LAUNCHES` 定義コメント・
            // 上の `Path::WmmaTf32` 分岐と同じ理由）。
            let mut c_dev = gemm
                .alloc_output_f16(m, n)
                .expect("mma_f16 output allocation must succeed on CUDA-equipped runner");

            for _ in 0..args.warmup {
                gemm.launch_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)
                    .expect("mma_f16 warmup launch must succeed on CUDA-equipped runner");
            }
            let start = Instant::now();
            for _ in 0..args.iters {
                gemm.launch_f16(&a_dev, &b_dev, &mut c_dev, m, n, k)
                    .expect("mma_f16 measured launch must succeed on CUDA-equipped runner");
            }
            let elapsed = start.elapsed().as_secs_f64();
            let per_iter_secs = elapsed / args.iters as f64;
            println!(
                "wall-clock (mma_f16, launch-only, {} iters): total={elapsed:.6}s \
                 per_iter={per_iter_secs:.6}s tflops={:.4} (ncu 実測値との突合用の参考値。\
                 ncu 実行中は計測区間にプロファイラのオーバーヘッドが乗るため単体実行時の \
                 数値とは一致しない)",
                args.iters,
                tflops(args.size, per_iter_secs)
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Path, tflops, validate_b_pad_requires_wmma_tf32};

    // CLI 引数は固定 allowlist との完全一致のみ受理する
    // （`.claude/rules/security.md` A03「シェル呼び出しでユーザー入力を
    // 直接展開しない」・「外部入力の検証」）。`parse_args` 本体は
    // `std::env::args()` を直接読むため単体テストから差し替えられない
    // （テスト専用に `std::env::args` を注入可能にするリファクタリングは
    // 本イシューのスコープ外。カーネル側の入力検証と異なりプロファイル
    // 対象バイナリの CLI パースであるため、allowlist 判定の核心
    // ロジックである `Path::parse` を個別に直接検証する）。

    #[test]
    fn path_parse_accepts_only_allowlisted_values() {
        assert_eq!(Path::parse("wmma_tf32"), Some(Path::WmmaTf32));
        assert_eq!(Path::parse("mma_f16"), Some(Path::MmaF16));
        assert_eq!(Path::parse("wmma_f16"), None);
        assert_eq!(Path::parse(""), None);
        assert_eq!(Path::parse("wmma_tf32; rm -rf /"), None);
    }

    #[test]
    fn path_as_str_round_trips_through_parse() {
        for p in [Path::WmmaTf32, Path::MmaF16] {
            assert_eq!(Path::parse(p.as_str()), Some(p));
        }
    }

    #[test]
    fn tflops_matches_flop_count_definition() {
        // 2*size^3 FLOPs を secs で割って TFLOPS 換算する定義どおりの値。
        let size = 4096u32;
        let secs = 1.0;
        let expected = 2.0 * (size as f64).powi(3) / secs / 1e12;
        assert_eq!(tflops(size, secs), expected);
    }

    // イシュー #743: `--b-pad` は `--path wmma_tf32` 限定（動的 SMEM 診断
    // 変種は TF32 staged 専用のため）。

    #[test]
    fn b_pad_without_path_wmma_tf32_is_rejected() {
        assert!(validate_b_pad_requires_wmma_tf32(Some(72), None).is_err());
        assert!(validate_b_pad_requires_wmma_tf32(Some(72), Some(Path::MmaF16)).is_err());
    }

    #[test]
    fn b_pad_with_path_wmma_tf32_or_absent_is_accepted() {
        assert!(validate_b_pad_requires_wmma_tf32(Some(72), Some(Path::WmmaTf32)).is_ok());
        assert!(validate_b_pad_requires_wmma_tf32(None, Some(Path::MmaF16)).is_ok());
        assert!(validate_b_pad_requires_wmma_tf32(None, None).is_ok());
    }
}
