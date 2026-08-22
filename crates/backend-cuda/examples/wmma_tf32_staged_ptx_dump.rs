//! イシュー #856: TF32 opt-staged（`kernels_wmma_opt.rs::
//! wmma_tf32_f32_staged_source`）threadblock swizzle 変種の
//! レジスタ・スピル差分を、`examples/mma_ptx_dump.rs`（f16 `mma.sync`
//! 経路・#782/#803/#804）と同型の手順で NVRTC コンパイル → PTX ダンプする
//! 診断バイナリ。
//!
//! ## 背景
//!
//! `mma_ptx_dump.rs` 冒頭コメント「背景」節と同じ制約（NVRTC に
//! `-Xptxas -v` 相当のログ出力経路がない・カーネルソース取得関数が非公開
//! `mod` の中にある・JIT キャッシュがディスクへ成果物を残さない）が TF32
//! opt-staged 経路にも同様に当てはまる。DGX Spark GB10 実機の CUDA 13.0
//! toolkit で NVRTC → PTX ダンプ → オフライン `ptxas -arch=sm_121 -v` に
//! 掛けるのが最小の観測経路である点も同じ。
//!
//! `docs/perf/cuda-gemm-swizzle-ab.md` §7.4.1（サイズ条件付き採用基準）は
//! `gemm.rs::CudaGemm::new`（本番既定コンストラクタ）への結線を「結線前
//! 必須確認」4 項目（bit 一致・parity 非後退・`cuda_floor_bench` 実測・
//! レジスタスピル確認）のクリアを条件としており、うちレジスタスピル確認の
//! 観測経路が本イシュー時点で TF32 staged 側に未整備だったため、本ファイル
//! で新設する（`mma_ptx_dump.rs` の base/swizzle ダンプ部分のみを TF32
//! staged 向けに移植した最小構成。f16 側が持つ warp タイル・ブロックタイル
//! 拡大候補ダンプは対象外——TF32 staged 側はそれらの候補が未整備のため）。
//!
//! `internal-diagnostics` feature（既定 off）を要求する。本 example が使う
//! `backend_cuda::diagnostics::{wmma_tf32_f32_staged_source,
//! wmma_tf32_f32_staged_source_with_swizzle,
//! wmma_tf32_staged_swizzle_group_width}` は非公開 `mod kernels_wmma_opt`／
//! `mod swizzle` への薄い診断用ラッパーであり、既定ビルドの公開 API 面
//! （`facade`）には出さない契約（`mma_ptx_dump.rs` と同じ feature ゲート
//! 方針）。`mma_ptx_dump.rs`・`mma_tf32_ptx_dump.rs` と同様、feature
//! 未指定でもビルドが成立する no-op main を明示的に持つため
//! `required-features` は使わず、ファイル内を丸ごと `#[cfg(feature =
//! "internal-diagnostics")]`／`#[cfg(not(...))]` で分岐する。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example wmma_tf32_staged_ptx_dump --release \
//!     --features internal-diagnostics -- --out-dir /tmp/wmma-tf32-staged-ptx-dump
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載環境では、初期化・コンパイル失敗を検出した
//! 時点で理由を表示し非 0 終了する（`mma_ptx_dump.rs` と同じ理由で CI
//! では実行されず、実機セッションでの手動実行が前提）。

#[cfg(feature = "internal-diagnostics")]
use backend_cuda::{CudaDevice, CudaError, compile_ptx, diagnostics};

#[cfg(feature = "internal-diagnostics")]
const USAGE: &str = "usage: wmma_tf32_staged_ptx_dump [--out-dir PATH]";

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
        .unwrap_or_else(|| std::path::PathBuf::from("target/wmma-tf32-staged-ptx-dump")))
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
                "backend-cuda wmma_tf32_staged_ptx_dump: CUDA driver unavailable ({detail}); \
                 this diagnostic requires a CUDA-equipped runner with the CUDA 13.0 \
                 toolkit (ptxas)."
            );
            std::process::exit(1);
        }
        Err(other) => {
            eprintln!("backend-cuda wmma_tf32_staged_ptx_dump: CudaDevice::new failed ({other})");
            std::process::exit(1);
        }
    };

    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!(
            "backend-cuda wmma_tf32_staged_ptx_dump: failed to create output directory {} ({e})",
            out_dir.display()
        );
        std::process::exit(1);
    }

    // `mma_ptx_dump.rs` と同じ codex-review P0 是正（PR #784 イシュー
    // #782）: symlink 経由の `create_dir_all` 迂回を防ぐため実パスへ確定
    // させてから使う。
    let out_dir = match std::fs::canonicalize(&out_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "backend-cuda wmma_tf32_staged_ptx_dump: failed to canonicalize output \
                 directory {} ({e})",
                out_dir.display()
            );
            std::process::exit(1);
        }
    };
    match out_dir.metadata() {
        Ok(m) if m.is_dir() => {}
        Ok(_) => {
            eprintln!(
                "backend-cuda wmma_tf32_staged_ptx_dump: output path {} resolves to a \
                 non-directory",
                out_dir.display()
            );
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!(
                "backend-cuda wmma_tf32_staged_ptx_dump: failed to stat output directory {} \
                 ({e})",
                out_dir.display()
            );
            std::process::exit(1);
        }
    }

    // NVRTC へ渡す arch は `device.arch()`（`compute_XY` 形式）をそのまま
    // 使う。本番経路（`gemm.rs::CudaGemm::new`）も同じ形式で `compile_ptx`
    // を呼ぶため、ここで異なる形式を渡すと本番と異なるコンパイル結果の
    // レジスタ使用量を観測してしまう（`mma_ptx_dump.rs` advisor 指摘の
    // 是正を踏襲）。`sm_arch_string` は ptxas 手順出力専用（`sm_XY` 形式）。
    let nvrtc_arch = device.arch().to_string();
    let ptxas_arch = sm_arch_string(&device);
    let num_sms = match device.multiprocessor_count() {
        Some(n) if n > 0 => n,
        _ => {
            // SM 数取得は実デバイス経由のみで行う（既知の固定値へ
            // フォールバックしない。`docs/perf/sm121-device-attributes.md`
            // の教訓——他デバイスの実測値を GB10 の値と誤扱いした過去の
            // 不備〈#758〉の再発防止）。取得失敗はエラー報告して非 0
            // 終了する。
            eprintln!(
                "backend-cuda wmma_tf32_staged_ptx_dump: device.multiprocessor_count() \
                 returned None or 0; cannot derive the dynamic swizzle group width without a \
                 real SM count."
            );
            std::process::exit(1);
        }
    };

    println!(
        "device: name={} compute_capability={:?} nvrtc_arch={nvrtc_arch} \
         ptxas_arch={ptxas_arch} num_sms={num_sms}",
        device.name(),
        device.compute_capability()
    );

    let base_path = out_dir.join("wmma_tf32_staged_base.ptx");
    if let Err(e) = dump_ptx(
        diagnostics::wmma_tf32_f32_staged_source(),
        &nvrtc_arch,
        &base_path,
        "base",
    ) {
        eprintln!("backend-cuda wmma_tf32_staged_ptx_dump: {e}");
        std::process::exit(1);
    }

    let group_width = diagnostics::wmma_tf32_staged_swizzle_group_width(num_sms);
    println!("swizzle: dynamic group_width={group_width} (derived from num_sms={num_sms})");

    let swizzle_source = match diagnostics::wmma_tf32_f32_staged_source_with_swizzle(group_width) {
        Ok(src) => src,
        Err(e) => {
            eprintln!(
                "backend-cuda wmma_tf32_staged_ptx_dump: \
                 wmma_tf32_f32_staged_source_with_swizzle({group_width}) failed ({e})"
            );
            std::process::exit(1);
        }
    };
    let swizzle_path = out_dir.join(format!("wmma_tf32_staged_swizzle_g{group_width}.ptx"));
    if let Err(e) = dump_ptx(&swizzle_source, &nvrtc_arch, &swizzle_path, "swizzle") {
        eprintln!("backend-cuda wmma_tf32_staged_ptx_dump: {e}");
        std::process::exit(1);
    }

    // `-o /dev/null` は toolkit バージョンによって ptxas のレジスタ割り当て
    // 完了前に打ち切られる可能性があるため使わず、スクラッチファイルへ
    // cubin を出力させる（`mma_ptx_dump.rs` advisor 指摘の是正を踏襲。
    // cubin 自体は破棄してよく `-v` の stderr ログのみが目的）。
    //
    // codex-review P0 是正（PR #784 イシュー #782）を踏襲: `base`/`swizzle`
    // は `--out-dir`（オペレーター入力）由来のパスを含むため、貼り付け
    // 実行を想定した手順文字列へ未クォートで埋め込むとシェルインジェク
    // ションが成立しうる（空白・`;`・`$()` を含むパス等）。`shell_quote`
    // で各パスを単一引用符クォートしたうえで埋め込む。
    let base_q = shell_quote(&base_path.display().to_string());
    let base_cubin_q = shell_quote(&format!("{}.cubin", base_path.display()));
    let swizzle_q = shell_quote(&swizzle_path.display().to_string());
    let swizzle_cubin_q = shell_quote(&format!("{}.cubin", swizzle_path.display()));

    let ptxas_lines = [
        format!("ptxas -arch={ptxas_arch} -v {base_q} -o {base_cubin_q}"),
        format!("ptxas -arch={ptxas_arch} -v {swizzle_q} -o {swizzle_cubin_q}"),
    ];

    let ptxas_commands = ptxas_lines
        .iter()
        .map(|line| format!("\x20   {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    println!(
        "done. out_dir={} nvrtc_arch={nvrtc_arch} ptxas_arch={ptxas_arch}. Run the following to \
         inspect register/spill counts (docs/perf/cuda-gemm-swizzle-ab.md §7):\n{ptxas_commands}",
        out_dir.display(),
    );
}

/// `internal-diagnostics` feature 未指定時の no-op（本ファイル冒頭コメント
/// 参照。`cargo build -p backend-cuda --example wmma_tf32_staged_ptx_dump`
/// が feature なしでもビルド成立することを保証する）。
#[cfg(not(feature = "internal-diagnostics"))]
fn main() {
    println!(
        "wmma_tf32_staged_ptx_dump: internal-diagnostics feature not enabled; this diagnostic \
         requires `cargo run -p backend-cuda --example wmma_tf32_staged_ptx_dump --features \
         internal-diagnostics`."
    );
}
