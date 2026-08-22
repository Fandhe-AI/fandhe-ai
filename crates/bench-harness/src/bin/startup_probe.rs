//! 起動コスト計測 probe バイナリ（TASK-13.1a・イシュー #170）。
//!
//! `bench_harness::startup::run_phase`（親ハーネス。`crates/bench-harness/src/startup.rs`）が
//! `std::process::Command` で子プロセスとして spawn し、標準出力へ [`bench_harness::startup::
//! ProbeReport`] 相当の JSON を 1 行出力する。本バイナリ自身の `main()` 開始時刻を起点に、
//! (1) バックエンド初期化完了・(2) 初回 GEMM カーネル完了（同期込み）の 2 点を計測する
//! （「内部計測」。`startup` モジュール冒頭ドキュメント参照）。
//!
//! 引数は許可リスト方式（[`bench_harness::startup::StartupBackend::parse`]）で検証し、
//! 未知の値は即座に拒否する（OWASP A03。`.claude/rules/security.md`）。
//! 本番経路で `unwrap`/`expect` を使わない方針（`.claude/rules/coding-rust.md`）に従い、
//! すべての失敗を標準エラーへメッセージ出力したうえで非ゼロ終了コードで終了する。

use bench_harness::rng::Xorshift64Star;
use bench_harness::startup::{PROBE_SCHEMA_VERSION, ProbeReport, StartupBackend};
use fandhe_ai_tensor_core::{BackendOps, Tensor, matmul_out_shape};
use std::process::ExitCode;
use std::time::Instant;

/// 起動コスト計測用ワークロードの正方行列サイズ。
///
/// 起動コストが計測対象でありカーネル実行時間そのものは従属変数のため、
/// NVRTC コンパイル・カーネル起動が発生する程度の小規模で足りる（実装計画メモ参照）。
const GEMM_SIZE: usize = 256;

/// 決定的シード（`bench_harness::rng`。`.claude/rules/coding-rust.md`「学習系回帰テストには
/// 決定的シード設定ユーティリティを使う」）。起動コスト計測において入力値そのものは
/// 意味を持たないが、フレーキーな再現性劣化を避けるため固定する。
const RNG_SEED: u64 = 0x5354_4152_5455_5001; // "STARTUP" 由来の固定値

fn main() -> ExitCode {
    let process_start = Instant::now();

    let args: Vec<String> = std::env::args().collect();
    let backend_arg = match args.get(1) {
        Some(s) => s.as_str(),
        None => {
            eprintln!("使用法: startup_probe <cpu|cuda|metal>");
            return ExitCode::from(2);
        }
    };
    let backend = match StartupBackend::parse(backend_arg) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("引数エラー: {e}");
            return ExitCode::from(2);
        }
    };

    match run(backend, process_start) {
        Ok(report) => match report.to_json() {
            Ok(json) => {
                println!("{json}");
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("レポート JSON エンコード失敗: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("startup_probe 失敗: {e}");
            ExitCode::FAILURE
        }
    }
}

/// 入力テンソル（`GEMM_SIZE` 正方行列）を決定的シードから生成する。
fn make_input() -> Result<Tensor<f32>, String> {
    let mut rng = Xorshift64Star::new(RNG_SEED);
    let data = rng.fill_vec(GEMM_SIZE * GEMM_SIZE);
    Tensor::new(data, &[GEMM_SIZE, GEMM_SIZE]).map_err(|e| format!("入力テンソル生成失敗: {e:?}"))
}

/// バックエンド初期化 → 初回 GEMM 実行までを計測し [`ProbeReport`] を構築する。
///
/// `process_start` は `main()` 冒頭で取得した `Instant`（本関数呼び出し前の
/// 引数パース分のオーバーヘッドも計測に含める。起動コストの外部計測（親ハーネス側の
/// wall time）との差分を安定させるため、内部計測の起点も可能な限り `main()` 冒頭へ寄せる）。
fn run(backend: StartupBackend, process_start: Instant) -> Result<ProbeReport, String> {
    let a = make_input()?;
    let b = make_input()?;

    let (device_init_secs, first_kernel_secs) = match backend {
        StartupBackend::Cpu => run_cpu(process_start, &a, &b)?,
        StartupBackend::Cuda => run_cuda(process_start, &a, &b)?,
        StartupBackend::Metal => run_metal(process_start, &a, &b)?,
    };

    Ok(ProbeReport {
        schema_version: PROBE_SCHEMA_VERSION.to_string(),
        backend: backend.as_str().to_string(),
        device_init_secs,
        first_kernel_secs,
    })
}

/// CPU 経路: `CpuBackendOps` は driver 初期化を持たないため、
/// `device_init_secs` はハンドル構築コストのみを表す参照点になる
/// （`startup` モジュールドキュメントの [`ProbeReport::device_init_secs`] 注記参照）。
fn run_cpu(process_start: Instant, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<(f64, f64), String> {
    let ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let device_init_secs = process_start.elapsed().as_secs_f64();

    ops.gemm(a, b)
        .map_err(|e| format!("CPU gemm 失敗: {e:?}"))?;
    let first_kernel_secs = process_start.elapsed().as_secs_f64();

    Ok((device_init_secs, first_kernel_secs))
}

/// CUDA 経路: `CudaDevice::new` を明示的に呼び、実際の driver 初期化（動的ロード込み）
/// コストを `device_init_secs` に含める。
///
/// `CudaBackendOps::gemm`（`crates/backend-cuda/src/ops.rs`）は呼び出しのたびに内部で
/// `CudaDevice::new` を再度実行する契約（ハンドル常駐化は TASK-1.9b／1.9d 以降。
/// `ops.rs` 冒頭コメント参照）であり、これを経由すると `first_kernel_secs` の区間に
/// 「NVRTC コンパイル＋カーネル起動＋完了待ち」だけでなく二重目のフル device 初期化が
/// 混入し計測が歪む（レビュー指摘。#170）。そのため本関数では `CudaBackendOps::gemm` を
/// 使わず、`device_handle` で取得済みの `CudaDevice` を [`fandhe_ai_backend_cuda::CudaGemm::new`] に
/// 明示的に渡して GEMM を実行する（`fandhe_ai_backend_cuda::CudaGemm` は `pub`。`ops.rs` の
/// `CudaBackendOps::gemm` 実装と同じ手順を、device 再取得なしで踏襲する）。
///
/// 注意: `run_tiled_f32` はホスト側スライスを受け取り内部で `clone_htod`／`clone_dtoh`
/// を行う契約のため、これらの転送コストは `first_kernel_secs` の計測区間に含まれる
/// （`ProbeReport::first_kernel_secs` のドキュメント参照。PR #360 codex-review 指摘・
/// Medium「Host transfer included in kernel timing」）。
fn run_cuda(
    process_start: Instant,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<(f64, f64), String> {
    if !fandhe_ai_backend_cuda::CudaDevice::is_available() {
        return Err("CUDA driver が利用不可（is_available() == false）".to_string());
    }
    let device = fandhe_ai_backend_cuda::CudaDevice::new(0)
        .map_err(|e| format!("CudaDevice::new 失敗: {e}"))?;
    let device_init_secs = process_start.elapsed().as_secs_f64();

    let out_shape = matmul_out_shape(a.shape(), b.shape())
        .map_err(|e| format!("GEMM 出力形状の算出失敗: {e:?}"))?;
    let (m, k) = (a.shape()[0] as u32, a.shape()[1] as u32);
    let n = b.shape()[1] as u32;
    let a_owned = a.contiguous();
    let b_owned = b.contiguous();
    let a_slice = a_owned
        .as_slice()
        .ok_or_else(|| "CUDA gemm: lhs not contiguous".to_string())?;
    let b_slice = b_owned
        .as_slice()
        .ok_or_else(|| "CUDA gemm: rhs not contiguous".to_string())?;

    let gemm = fandhe_ai_backend_cuda::CudaGemm::new(&device)
        .map_err(|e| format!("CudaGemm::new 失敗: {e}"))?;
    let out = gemm
        .run_tiled_f32(a_slice, b_slice, m, n, k)
        .map_err(|e| format!("CUDA gemm 失敗: {e}"))?;
    Tensor::new(out, &out_shape).map_err(|e| format!("出力テンソル構築失敗: {e:?}"))?;
    let first_kernel_secs = process_start.elapsed().as_secs_f64();

    Ok((device_init_secs, first_kernel_secs))
}

/// Metal 経路: `cfg(target_os = "macos")` 限定（`.claude/rules/deps-policy.md`）。
/// 非 macOS ではビルド時に `backend-metal` クレート自体が利用できないため、
/// 実行時エラーとして明示的に拒否する（`backend-cuda` の「toolkit 非搭載でもビルド成立し
/// 実行時のみ拒否」とは異なり、Metal は OS 単位でビルド自体を cfg 分離する契約。
/// `crates/bench-harness/Cargo.toml` の `[target.'cfg(target_os = "macos")'.dev-dependencies]`
/// 参照）。
#[cfg(target_os = "macos")]
fn run_metal(
    process_start: Instant,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<(f64, f64), String> {
    // `MetalContext::new` は Metal デバイス・コマンドキューの取得までを担う
    // （`crates/backend-metal/src/context.rs`）。`MetalBackendOps::gemm` 内部でも
    // 都度構築されるため、`first_kernel_secs - device_init_secs` の区間には
    // 「カーネル実行」だけでなく `MetalContext` の再構築コストも含まれうる
    // （CUDA 経路と同型の既知の制約。#170 レビュー指摘。本 OS では
    // ビルド確認できないため CUDA 経路と異なりここでは計測経路自体は変更せず、
    // 区間の意味をコメントで正確化するに留める。真の解消は `MetalContext` の
    // ハンドル常駐化以降）。
    fandhe_ai_backend_metal::MetalContext::new()
        .map_err(|e| format!("MetalContext::new 失敗: {e:?}"))?;
    let device_init_secs = process_start.elapsed().as_secs_f64();

    let ops = fandhe_ai_backend_metal::MetalBackendOps::new();
    ops.gemm(a, b)
        .map_err(|e| format!("Metal gemm 失敗: {e:?}"))?;
    let first_kernel_secs = process_start.elapsed().as_secs_f64();

    Ok((device_init_secs, first_kernel_secs))
}

#[cfg(not(target_os = "macos"))]
fn run_metal(
    _process_start: Instant,
    _a: &Tensor<f32>,
    _b: &Tensor<f32>,
) -> Result<(f64, f64), String> {
    Err(
        "Metal バックエンドは macOS 限定（cfg(target_os = \"macos\")）のため本 OS では未対応"
            .to_string(),
    )
}
