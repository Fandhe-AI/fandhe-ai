//! イシュー #806: TF32 生 `mma.sync` 経路（`kernels_mma_tf32.rs`。#801→
//! PR #823）のブロックタイル拡大候補を、`examples/mma_ptx_dump.rs`
//! （f16 `mma.sync` 経路・#782/#803/#804）と同型の手順で NVRTC コンパイル
//! → PTX ダンプする診断バイナリ。
//!
//! ## 背景
//!
//! `mma_ptx_dump.rs` 冒頭コメント「背景」節と同じ制約（NVRTC に
//! `-Xptxas -v` 相当のログ出力経路がない・カーネルソース取得関数が
//! 非公開 `mod` の中にある・JIT キャッシュがディスクへ成果物を残さない）
//! が TF32 経路にも同様に当てはまる。DGX Spark GB10 実機の CUDA 13.0
//! toolkit で NVRTC → PTX ダンプ → オフライン `ptxas -arch=sm_121 -v` に
//! 掛けるのが最小の観測経路である点も同じ。
//!
//! `mma_ptx_dump.rs` を直接拡張せず本ファイルを独立させた理由: 同ファイル
//! は既に f16 経路の base/swizzle/warp タイル/ブロックタイル候補で
//! 554 行に達しており（イシュー #806 実装計画 3 節）、TF32 側は現時点で
//! warp タイル単独候補・swizzle 経路が未整備（本番非結線・#802 スコープ）
//! のためブロックタイル候補のみを扱う。両ファイルを分離することで
//! 各々の候補セット・命名規則が交錯しない。
//!
//! ## ブロックタイル拡大候補（実装計画 §4 候補表。
//! `docs/perf/cuda-gemm-mma-tf32-block-tile.md` §3 参照）
//!
//! 現行 TF32 定数（64x64x16・S3・warp2x4・28,416B・静的）に対し、
//! ステージ増・M/N 拡大・両拡大・両拡大+ステージ増・BK 拡大の 5 候補を
//! `__launch_bounds__`（なし／導出スレッド数）の 2 通りずつダンプする。
//! `diagnostics::mma_tf32_source_with_block_tile`（`kernels_mma_tf32.rs`
//! 側ドキュメンテーションコメント参照）が候補ごとに静的/動的共有メモリ・
//! opt-in 予算超過を判定する。opt-in 予算超過候補は
//! （`mma_ptx_dump.rs` の f16 版と同じ codex-review P1 是正方針・PR #831
//! に倣い）接続デバイスの実測 opt-in 上限との比較で非致命的に除外し、
//! 除外根拠（実測要求量・実測上限）を標準出力へ記録する。**本番カーネル
//! 定数（`MMA_TF32_BM`/`MMA_TF32_BN`/`MMA_TF32_STAGES` 等）は変更しない**
//! （実機到達不能のため #806 実装セッション時点では本番結線を行わず、
//! 診断機構整備のみに留めた。`docs/cuda-tensor-core-design.md` §15 参照）。
//! 動的 SMEM 化を伴う生成ソースは NVRTC/ptxas での実機構文検証を経ておらず、
//! 本番起動側の opt-in 結線（`CudaFunction::set_attribute`・
//! `shared_mem_bytes`）も未実装のままである。
//!
//! `internal-diagnostics` feature（既定 off）を要求する。本 example が使う
//! `backend_cuda::diagnostics::{mma_tf32_source, mma_tf32_source_with_block_tile,
//! mma_tf32_block_tile}` は非公開 `mod kernels_mma_tf32` への薄い診断用
//! ラッパーであり、既定ビルドの公開 API 面（`facade`）には出さない契約
//! （`mma_ptx_dump.rs` と同じ feature ゲート方針）。`mma_ptx_dump.rs` と
//! 同様、feature 未指定でもビルドが成立する no-op main を明示的に持つ
//! ため `required-features` は使わず、ファイル内を丸ごと
//! `#[cfg(feature = "internal-diagnostics")]`／`#[cfg(not(...))]` で分岐
//! する。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example mma_tf32_ptx_dump --release \
//!     --features internal-diagnostics -- --out-dir /tmp/mma-tf32-ptx-dump
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載環境では、初期化・コンパイル失敗を検出した
//! 時点で理由を表示し非 0 終了する（`mma_ptx_dump.rs` と同じ理由で CI
//! では実行されず、実機セッションでの手動実行が前提）。

#[cfg(feature = "internal-diagnostics")]
use backend_cuda::{CudaDevice, CudaError, compile_ptx, diagnostics};

#[cfg(feature = "internal-diagnostics")]
const USAGE: &str = "usage: mma_tf32_ptx_dump [--out-dir PATH]";

/// `std::env::args` のみで CLI 引数をパースする（依存追加なし。
/// `mma_ptx_dump.rs::parse_out_dir` と同じ方針）。
#[cfg(feature = "internal-diagnostics")]
fn parse_out_dir() -> Result<std::path::PathBuf, String> {
    let mut out_dir: Option<String> = None;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--out-dir" => {
                let v = it.next().ok_or("--out-dir には値が必要")?;
                out_dir = Some(v);
            }
            other => return Err(format!("未知の引数: '{other}'")),
        }
    }
    Ok(out_dir
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from("target/mma-tf32-ptx-dump")))
}

/// `mma_ptx_dump.rs::sm_arch_string` と同一（手順出力〈オペレーターが
/// オフラインの `ptxas -arch=...` へそのまま使う値〉専用。`compile_ptx`
/// へは渡さない）。
#[cfg(feature = "internal-diagnostics")]
fn sm_arch_string(device: &CudaDevice) -> String {
    let (major, minor) = device.compute_capability();
    format!("sm_{major}{minor}")
}

/// `mma_ptx_dump.rs::dump_ptx` と同一契約（symlink 経由の任意ファイル
/// 破壊を防ぐため `OpenOptions::create_new(true)` で新規作成のみを許す。
/// codex-review P0 是正・PR #784 イシュー #782 の教訓を踏襲）。
#[cfg(feature = "internal-diagnostics")]
fn dump_ptx(
    src: &str,
    nvrtc_arch: &str,
    out_path: &std::path::Path,
    label: &str,
) -> Result<(), String> {
    use std::io::Write as _;

    let ptx = compile_ptx(src, nvrtc_arch)
        .map_err(|e| format!("{label}: NVRTC コンパイルに失敗しました ({e})"))?;

    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out_path)
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                format!(
                    "{label}: {} は既に存在します（symlink の可能性があるため上書きしません）。\
                     削除してから再実行してください。",
                    out_path.display()
                )
            } else {
                format!(
                    "{label}: {} への書き出しに失敗しました ({e})",
                    out_path.display()
                )
            }
        })?;
    file.write_all(ptx.to_src().as_bytes()).map_err(|e| {
        format!(
            "{label}: {} への書き出しに失敗しました ({e})",
            out_path.display()
        )
    })?;
    println!(
        "{label}: wrote {} (nvrtc_arch={nvrtc_arch})",
        out_path.display()
    );
    Ok(())
}

/// `mma_ptx_dump.rs::shell_quote` と同一（コマンドインジェクション対策。
/// codex-review P0 是正・PR #784 イシュー #782 の教訓を踏襲）。
#[cfg(feature = "internal-diagnostics")]
fn shell_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

#[cfg(feature = "internal-diagnostics")]
fn main() {
    let out_dir = match parse_out_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            std::process::exit(1);
        }
    };

    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            eprintln!(
                "backend-cuda mma_tf32_ptx_dump: CUDA driver unavailable ({detail}); \
                 this diagnostic requires a CUDA-equipped runner with the CUDA 13.0 \
                 toolkit (ptxas)."
            );
            std::process::exit(1);
        }
        Err(other) => {
            eprintln!("backend-cuda mma_tf32_ptx_dump: CudaDevice::new failed ({other})");
            std::process::exit(1);
        }
    };

    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!(
            "backend-cuda mma_tf32_ptx_dump: failed to create output directory {} ({e})",
            out_dir.display()
        );
        std::process::exit(1);
    }

    // `mma_ptx_dump.rs` と同じ codex-review P0 是正（PR #784 イシュー
    // #782）: symlink 経由の `create_dir_all` 迂回を防ぐため実パスへ
    // 確定させてから使う。
    let out_dir = match std::fs::canonicalize(&out_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "backend-cuda mma_tf32_ptx_dump: failed to canonicalize output directory {} ({e})",
                out_dir.display()
            );
            std::process::exit(1);
        }
    };
    match out_dir.metadata() {
        Ok(m) if m.is_dir() => {}
        Ok(_) => {
            eprintln!(
                "backend-cuda mma_tf32_ptx_dump: output path {} resolves to a non-directory",
                out_dir.display()
            );
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!(
                "backend-cuda mma_tf32_ptx_dump: failed to stat output directory {} ({e})",
                out_dir.display()
            );
            std::process::exit(1);
        }
    }

    let nvrtc_arch = device.arch().to_string();
    let ptxas_arch = sm_arch_string(&device);

    println!(
        "device: name={} compute_capability={:?} nvrtc_arch={nvrtc_arch} ptxas_arch={ptxas_arch}",
        device.name(),
        device.compute_capability()
    );

    let mut ptxas_lines: Vec<String> = Vec::new();

    // base（既定構成。現行本番カーネルと同一ソース）。
    let base_path = out_dir.join("mma_tf32_base.ptx");
    if let Err(e) = dump_ptx(
        diagnostics::mma_tf32_source(),
        &nvrtc_arch,
        &base_path,
        "base",
    ) {
        eprintln!("backend-cuda mma_tf32_ptx_dump: {e}");
        std::process::exit(1);
    }
    {
        let path_q = shell_quote(&base_path.display().to_string());
        let cubin_q = shell_quote(&format!("{}.cubin", base_path.display()));
        ptxas_lines.push(format!("ptxas -arch={ptxas_arch} -v {path_q} -o {cubin_q}"));
    }

    // イシュー #806 実装計画 §4 候補表: opt-in 予算判定は `mma_ptx_dump.rs`
    // の f16 ブロックタイル候補ループ（#804・PR #831 codex-review P1 是正）
    // と同じく接続デバイスの実測値を使い、固定値へフォールバックしない
    // （`docs/perf/sm121-device-attributes.md` の教訓——他デバイスの実測値
    // を GB10 の値と誤扱いした過去の不備〈#758〉の再発防止）。
    let optin_budget_bytes = match device.shared_memory_per_block_optin() {
        Some(v) => v,
        None => {
            eprintln!(
                "backend-cuda mma_tf32_ptx_dump: device.shared_memory_per_block_optin() \
                 returned None; cannot derive the opt-in shared memory budget without a real \
                 device attribute value."
            );
            std::process::exit(1);
        }
    };
    println!("device: optin_budget_bytes={optin_budget_bytes}");

    for (label, bm, bn, bk, stages, warp_tiles_m, warp_tiles_n, threads) in [
        // ステージ増のみ（現行ブロックタイルのまま S3→S4。37,888B。静的
        // 48KiB 以下のため extern __shared__ 変種にはならない）。
        ("bt64x64_s4", 64u32, 64u32, 16u32, 4u32, 2u32, 4u32, 128u32),
        // M 拡大（128x64・warp4x2。43,776B。静的 48KiB 以下）。
        ("bt128x64_s3_wt4x2", 128, 64, 16, 3, 4, 2, 256),
        // N 拡大（64x128・warp2x4。40,704B。静的 48KiB 以下）。
        ("bt64x128_s3_wt2x4", 64, 128, 16, 3, 2, 4, 256),
        // 両拡大（128x128・warp2x4。56,064B。静的超過→opt-in）。
        ("bt128x128_s3_wt2x4", 128, 128, 16, 3, 2, 4, 512),
        // 両拡大+ステージ増（128x128・S4・warp2x4。机上見積もり
        // 74,752B。opt-in 予算との比較は下記ループが実測値で判定する。
        ("bt128x128_s4_wt2x4", 128, 128, 16, 4, 2, 4, 512),
        // BK 拡大（64x64x32・warp2x2。53,760B。静的超過→opt-in）。
        ("bt64x64x32_s3_wt2x2", 64, 64, 32, 3, 2, 2, 256),
    ] {
        for launch_bounds in [None, Some(threads)] {
            let file_label = match launch_bounds {
                None => label.to_string(),
                Some(v) => format!("{label}_lb{v}"),
            };
            let source = match diagnostics::mma_tf32_source_with_block_tile(
                bm,
                bn,
                bk,
                stages,
                warp_tiles_m,
                warp_tiles_n,
                launch_bounds,
                optin_budget_bytes,
            ) {
                Ok(src) => src,
                // opt-in 予算超過（`smem_bytes > optin_budget_bytes`。
                // `kernels_mma_tf32.rs::mma_tf32_source_with_block_tile`
                // 参照）は実測上限に対する正当な机上除外であり非致命的に
                // 扱う。それ以外の `InvalidKernelConfig`（引数検証エラー
                // 等）は候補表・導出ロジックの不整合を示すため fatal。
                Err(CudaError::InvalidKernelConfig { detail })
                    if detail.contains("exceeding the opt-in budget") =>
                {
                    println!("desk-excluded: {file_label} ({bm}x{bn}x{bk} S{stages}) {detail}");
                    continue;
                }
                Err(e) => {
                    eprintln!(
                        "backend-cuda mma_tf32_ptx_dump: mma_tf32_source_with_block_tile({bm}, \
                         {bn}, {bk}, {stages}, {warp_tiles_m}, {warp_tiles_n}, \
                         {launch_bounds:?}) failed ({e})"
                    );
                    std::process::exit(1);
                }
            };
            let path = out_dir.join(format!("mma_tf32_{file_label}.ptx"));
            if let Err(e) = dump_ptx(&source, &nvrtc_arch, &path, &file_label) {
                eprintln!("backend-cuda mma_tf32_ptx_dump: {e}");
                std::process::exit(1);
            }
            let path_q = shell_quote(&path.display().to_string());
            let cubin_q = shell_quote(&format!("{}.cubin", path.display()));
            ptxas_lines.push(format!("ptxas -arch={ptxas_arch} -v {path_q} -o {cubin_q}"));
        }
    }

    let ptxas_commands = ptxas_lines
        .iter()
        .map(|line| format!("\x20   {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    println!(
        "done. out_dir={} nvrtc_arch={nvrtc_arch} ptxas_arch={ptxas_arch}. Run the following to \
         inspect register/spill counts (docs/perf/cuda-gemm-mma-tf32-block-tile.md §4):\n{ptxas_commands}",
        out_dir.display(),
    );
}

/// `internal-diagnostics` feature 未指定時の no-op（本ファイル冒頭
/// コメント参照。`cargo build -p backend-cuda --example mma_tf32_ptx_dump`
/// が feature なしでもビルド成立することを保証する）。
#[cfg(not(feature = "internal-diagnostics"))]
fn main() {
    println!(
        "mma_tf32_ptx_dump: internal-diagnostics feature not enabled; this diagnostic requires \
         `cargo run -p backend-cuda --example mma_tf32_ptx_dump --features internal-diagnostics`."
    );
}
