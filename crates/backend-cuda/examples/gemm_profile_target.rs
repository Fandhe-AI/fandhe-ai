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
//! 本ファイルの追加に伴う `Cargo.toml` の変更は不要
//! （`deps-policy.md` ユーザー承認事項に該当しない）。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo build -p backend-cuda --example gemm_profile_target --release
//! ncu --launch-skip <warmup 起動数> --launch-count <iters> \
//!     --metrics <確定メトリクス名, カンマ区切り> \
//!     ./target/release/examples/gemm_profile_target \
//!     --path wmma_tf32 --size 4096
//! ```
//!
//! `--path`（`wmma_tf32`｜`mma_f16`。必須）・`--size`（`1024`｜`2048`｜
//! `4096`。必須）は固定 allowlist との完全一致のみ受理する（`.claude/rules/
//! security.md` A03「外部入力の検証」。シェル呼び出し・文字列展開は行わ
//! ない）。`--iters`（既定 5）・`--warmup`（既定 2）は正の整数のみ受理する。
//! 採取手順・実測記録・主因分析は `docs/perf/cuda-gemm-bottleneck-diagnosis.md`
//! を参照。
//!
//! CUDA 非搭載・NVRTC 非搭載・cc 非対応環境では、`CudaDevice::new`／各
//! `*Gemm::new` の失敗を検出した時点で理由を表示して終了する（`panic!`
//! しない。`cuda_floor_bench.rs`・`gemm_mma_bench.rs` と同じ環境適応分岐。
//! CI の `cargo build --workspace --all-targets` はビルドのみなので
//! この実行時分岐は CI に影響しない）。

use std::time::Instant;

use backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaMmaGemm};
use bench_harness::rng::Xorshift64Star;
use cudarc::driver::sys::CUdevice_attribute;
use half::f16;

/// 決定的シード（`cuda_floor_bench.rs`・`gemm_mma_bench.rs` と同一値。
/// 過去実測・他バックエンドベンチと同じ入力分布に揃える）。
const SEED: u64 = 0xC0FFEE;

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
}

const USAGE: &str = "usage: gemm_profile_target --path {wmma_tf32|mma_f16} --size {1024|2048|4096} [--iters N] [--warmup N]";

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
            other => return Err(format!("未知の引数: '{other}'")),
        }
    }

    Ok(Args {
        path: path.ok_or("--path は必須")?,
        size: size.ok_or("--size は必須")?,
        iters,
        warmup,
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
    // タイル定数は非公開モジュール（`kernels_wmma_opt.rs`／
    // `kernels_mma.rs` は `mod`、`pub mod` ではないため crate 外から
    // 参照不能）の値をこの診断バイナリ専用にコピーしたもの。カーネル側の
    // 値を変更しないという本イシューのスコープ（実装計画 §6）上、
    // カーネル側モジュールの可視性は変更せずここに手元転記する。値の
    // 出典・整合はコメントの参照先行番号で追跡する（値が乖離した場合は
    // 出典側の変更漏れとして検知できるよう、変更時は必ず両者を同時に
    // 更新すること）。
    let (block_m, block_n): (u32, u32) = match path {
        // `kernels_wmma_opt.rs::WMMA_TF32_OPT_BLOCK_M`/`_N`（64×64）。
        Path::WmmaTf32 => (64, 64),
        // `kernels_mma.rs::MMA_BM`/`MMA_BN`（32×64）。
        Path::MmaF16 => (32, 64),
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
        Err(CudaError::DriverUnavailable { detail }) => {
            println!(
                "backend-cuda gemm_profile_target: CUDA driver unavailable ({detail}); skipping."
            );
            return;
        }
        Err(other) => {
            println!(
                "backend-cuda gemm_profile_target: CudaDevice::new failed ({other}); skipping."
            );
            return;
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

    let mut rng = Xorshift64Star::new(SEED);
    let (m, n, k) = (args.size, args.size, args.size);
    let total_launches = args.warmup + args.iters;
    if total_launches == 0 {
        println!("warmup=0 かつ iters=0 のため計測対象の起動がない。終了する。");
        return;
    }

    match args.path {
        Path::WmmaTf32 => {
            let gemm = match CudaGemm::new(&device) {
                Ok(g) => g,
                Err(e) => {
                    println!(
                        "backend-cuda gemm_profile_target: tiled/WMMA(TF32) kernel unavailable ({e}); skipping."
                    );
                    return;
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
            // 指摘）。加えて、もし opt・基本の両方が未ロードであれば
            // フォールバック先の `launch_wmma_tf32` 自体が
            // `CudaError::WmmaUnavailable` を返し、起動ループ側の `.expect()`
            // が panic していた（`CudaDevice::new`／`CudaMmaGemm::new` が
            // 徹底している fail-soft skip 方針に反する。PR #637 Cursor
            // Bugbot 指摘）。よって opt カーネル不在時はフォールバック
            // させず、起動ループへ入る前に非 0 終了せず早期 return して
            // skip する（`cuda_floor_bench.rs` の環境非対応スキップと同じ
            // 判断: panic ではなく理由を表示して終了する）。
            if gemm.wmma_tf32_opt_available() {
                println!("wmma_tf32 opt kernel: AVAILABLE (used for this run's launches).");
            } else {
                println!(
                    "backend-cuda gemm_profile_target: wmma_tf32 opt kernel unavailable ({}); \
                     skipping instead of falling back to the basic (non-optimized) WMMA(TF32) \
                     kernel, because ncu results for the fallback kernel would not represent the \
                     opt-kernel data-reuse characteristics under diagnosis.",
                    gemm.wmma_tf32_opt_unavailable_reason()
                        .unwrap_or("unknown reason")
                );
                return;
            }

            let a = rng.fill_vec((m as usize) * (k as usize));
            let b = rng.fill_vec((k as usize) * (n as usize));
            let (a_dev, b_dev) = gemm
                .upload_f32(&a, &b)
                .expect("wmma_tf32 upload must succeed on CUDA-equipped runner");
            let mut c_dev = gemm
                .alloc_output_f32(m, n)
                .expect("wmma_tf32 output allocation must succeed on CUDA-equipped runner");

            // ncu は `--launch-skip <warmup 起動数> --launch-count <iters>`
            // でこのループ内のカーネル起動番号を直接指定してプロファイル
            // する（モジュール冒頭ドキュメンテーションコメント「実行手順」
            // 参照）。`launch_wmma_tf32` は呼び出しごとに内部で
            // `stream.synchronize()` するため（`gemm.rs::launch_wmma_tf32`
            // 末尾参照）、ここでの追加同期は不要。
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
        Path::MmaF16 => {
            let gemm = match CudaMmaGemm::new(&device) {
                Ok(g) => g,
                Err(e) => {
                    println!(
                        "backend-cuda gemm_profile_target: mma.sync f16 kernel unavailable ({e}); skipping."
                    );
                    return;
                }
            };

            let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
            let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));
            let (a_dev, b_dev) = gemm
                .upload_f16(&a, &b)
                .expect("mma_f16 upload must succeed on CUDA-equipped runner");
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
    use super::{Path, tflops};

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
}
