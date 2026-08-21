//! イシュー #782 codex-review P1 是正: 「レジスタスピル確認を完了して
//! から本番結線せよ」を解消するため、`mma_f16` カーネルのレジスタスピル
//! を実機で観測可能にする診断バイナリ。
//!
//! ## 背景（前回の実機セッションで確認済みの制約）
//!
//! 1. `nvrtc::compile_ptx`（`src/nvrtc.rs`）は NVRTC の `-Xptxas -v` 相当
//!    （レジスタ使用量ログ）のオプションを渡さない。cudarc 0.19.8 の
//!    `CompileOptions` にも NVRTC 側にもそのようなログ出力の直接経路は
//!    ない
//! 2. `kernels_mma.rs` のカーネルソース取得関数（[`mma_f16_source`]／
//!    [`mma_f16_source_with_swizzle`]）は非公開 `mod kernels_mma` の中に
//!    あり、crate 外（本 example を含む）から到達不能だった
//! 3. JIT キャッシュ（`module_cache.rs`）はコンパイル成果物をプロセス内
//!    LRU に保持するのみで、ディスクへ成果物（PTX・cubin）を残さない
//!
//! DGX Spark GB10 実機には CUDA 13.0 toolkit（`/usr/local/cuda-13.0/bin/
//! ptxas`）が入っているため、**NVRTC で PTX を生成してファイルへダンプし、
//! オフラインで `ptxas -arch=sm_121 -v` に掛ける**のが、上記制約下での
//! 最小の観測経路である（NVRTC 自体の呼び出しオプション拡張・JIT
//! キャッシュへの成果物永続化は本 example のスコープ外）。
//!
//! `mma_f16_source()`（base）と `mma_f16_source_with_swizzle(group_width)`
//! （L2 再利用スウィズル適用版。イシュー #499・#782。動的選択幅は
//! `diagnostics::mma_swizzle_group_width` が実デバイスの SM 数から導出
//! する）の 2 ソースを NVRTC でコンパイルし、`.ptx` としてディスクへ書き
//! 出す。ptxas 自体の実行・ログ読み取りは本 example のスコープ外とし
//! （NVRTC → PTX ダンプまでが本 example の責務。ptxas 呼び出しは
//! オペレーターが手動で行う）、具体的な手順は `docs/perf/
//! cuda-gemm-swizzle-ab.md` §3「レジスタスピル確認」に記載する。
//!
//! ## warp タイル拡大候補（イシュー #803 追加）
//!
//! 上記 base／swizzle に加え、warp タイル拡大候補（`docs/perf/
//! cuda-gemm-mma-warp-tile-register-budget.md` §3.1 候補表: 2x2 現行・
//! 2x4 案 A・4x2 案 B・4x4 案 C）× `__launch_bounds__`（なし／導出スレッド
//! 数で明示付与）の組み合わせを `diagnostics::mma_f16_source_with_warp_tiles`
//! （`kernels_mma.rs` 側ドキュメンテーションコメント参照）でダンプする。
//! **本番カーネル定数（`MMA_WARP_TILES_M`/`_N`）は変更しない**（本番結線は
//! 後続 #804 のスコープ）。
//!
//! `internal-diagnostics` feature（既定 off）を要求する。本 example が使う
//! `backend_cuda::diagnostics::{mma_f16_source, mma_f16_source_with_swizzle,
//! mma_f16_source_with_warp_tiles, mma_swizzle_group_width}` は非公開 `mod
//! kernels_mma`／`mod swizzle` への薄い診断用ラッパーであり、既定ビルドの
//! 公開 API 面（`facade`）には出さない契約（`lib.rs::diagnostics` モジュール
//! 冒頭コメント・`examples/gemm_mma_swizzle_bench.rs` と同じ feature ゲート
//! 方針を踏襲）。
//! `gemm_mma_swizzle_bench.rs` は `Cargo.toml` の `[[example]]
//! required-features` で本 feature を要求し、feature 未指定時は
//! `cargo build --workspace --all-targets`（自動ターゲット走査）から
//! 静かに除外される方式だが、本 example は「feature なしでもビルドは
//! 成立する no-op main を明示的に持つ」ことを要件としているため
//! `required-features` は使わず、ファイル内を丸ごと `#[cfg(feature =
//! "internal-diagnostics")]`／`#[cfg(not(...))]` で分岐する。これにより
//! `cargo build -p backend-cuda --example mma_ptx_dump`（feature 未指定を
//! 明示指定）でもビルドが成立する。
//!
//! ## 実行手順
//!
//! ```sh
//! cargo run -p backend-cuda --example mma_ptx_dump --release \
//!     --features internal-diagnostics -- --out-dir /tmp/mma-ptx-dump
//! ```
//!
//! CUDA 非搭載・NVRTC 非搭載環境では、初期化・コンパイル失敗を検出した
//! 時点で理由を表示し非 0 終了する（本バイナリは CI では実行されず
//! （`cargo build --workspace --all-targets` によるビルド検証のみ）、
//! 実機セッションでの手動実行が前提のため、`gemm_profile_target.rs` の
//! ような `--allow-missing-driver` opt-in は設けない。実行するからには
//! 実機到達を前提とする）。
//!
//! `mma.sync` 経路は cc<8.0 環境では利用できない（`kernels_mma.rs`
//! 冒頭コメント参照）。NVRTC コンパイル自体はターゲット `arch` 文字列を
//! 渡すだけで実行元デバイスの compute capability に依存しないが、
//! 生成される PTX の意味（実機で使われる命令セット）は `arch` に従う
//! ため、本 example は接続中のデバイス（`CudaDevice::new(0)`）の
//! `arch()` をそのまま使う。

#[cfg(feature = "internal-diagnostics")]
use backend_cuda::{CudaDevice, CudaError, compile_ptx, diagnostics};

#[cfg(feature = "internal-diagnostics")]
const USAGE: &str = "usage: mma_ptx_dump [--out-dir PATH]";

/// `std::env::args` のみで CLI 引数をパースする（依存追加なし。
/// `gemm_profile_target.rs::parse_args` と同じ方針）。
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
        .unwrap_or_else(|| std::path::PathBuf::from("target/mma-ptx-dump")))
}

/// `CudaDevice::arch()`（NVRTC 向け `compute_XY` 形式。`compile_ptx` の
/// 呼び出し契約）から `ptxas -arch` が要求する `sm_XY` 形式を導出する。
/// `device.rs::arch()` の内部表現に依存せず `compute_capability()`
/// （major, minor の整数タプル）から直接組み立てることで、`compute_XY`
/// プレフィックスの文字列置換に頼らない（`arch()` の内部フォーマットが
/// 将来変わっても壊れない）。**この文字列は `compile_ptx` へは渡さない**
/// （NVRTC は `compute_XY` 形式を要求する。`nvrtc.rs::compile_ptx` の
/// ドキュメンテーションコメント参照）。手順出力（オペレーターがオフラインの
/// `ptxas -arch=...` へそのまま使う値）専用。
#[cfg(feature = "internal-diagnostics")]
fn sm_arch_string(device: &CudaDevice) -> String {
    let (major, minor) = device.compute_capability();
    format!("sm_{major}{minor}")
}

/// NVRTC コンパイル結果（`Ptx`）を `.ptx` テキストとしてファイルへ書き出す。
/// `compile_ptx`（`nvrtc.rs`）はコンパイル成果物をプロセス内に保持する
/// だけでディスクへ残さない（本ファイル冒頭「背景」節 3）ため、ここで
/// 明示的に永続化する。
///
/// `nvrtc_arch` は `device.arch()`（`compute_XY` 形式）をそのまま渡すこと
/// （本番経路 `gemm_mma.rs::CudaMmaGemm::new` が `compile_ptx` を呼ぶ際と
/// 同一の引数形式で NVRTC を呼ぶ必要がある。`sm_arch_string` の `sm_XY`
/// 形式を誤って渡すと、レジスタ確認の対象が本番と異なるコンパイル結果に
/// なってしまう）。
/// `out_path` に PTX テキストを書き出す。codex-review P0 是正
/// （PR #784 イシュー #782）: `std::fs::write` は対象が symlink でも無条件
/// truncate するため、`--out-dir` 配下に攻撃者が仕込んだ symlink（例:
/// `mma_f16_base.ptx -> /etc/passwd`）経由でリンク先の任意ファイルを
/// 破壊できてしまう。`OpenOptions::create_new(true)` は対象パスが
/// 既存（symlink を含む）の場合に `AlreadyExists` で失敗し新規作成しか
/// 許さないため、既存ファイル・symlink への書き込み（上書き）を構造的に
/// 防げる。ファイル名は呼び出し元で固定名（`mma_f16_base.ptx` 等）のみを
/// 渡す契約を維持し、外部入力から合成しない。
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

/// シェルの単一引用符でパスをクォートする（貼り付け実行時の任意コマンド
/// 実行対策。codex-review P0 是正・PR #784 イシュー #782）。`--out-dir`
/// はオペレーター入力であり空白・`;`・`$()` 等を含みうるため、手順出力
/// （operator がそのまま端末に貼り付ける想定の `ptxas` コマンド例）へ
/// 未クォートで埋め込むとコマンドインジェクションが成立する。POSIX
/// シェルの単一引用符内では `'` 自身のみがエスケープを要し、
/// `'\''`（引用符を閉じる→エスケープした `'` を挟む→再度開く）で
/// 表現するのが標準的な手法。手動でのシェル往復確認済み（`/tmp/a b`・
/// `/tmp/x;id`・`/tmp/$(id)`・`/tmp/it's` の 4 ケースで `sh -c "printf %s
/// <quoted>"` が原文をバイト単位で再現することを確認。パスの実体は
/// `PathBuf::display()`（呼び出し元）由来のため、非 UTF-8 パスでは
/// `display()` 自体が置換文字 `U+FFFD` を挿入しうる点に注意（不正な
/// バイト列を安全に丸めるだけで、`'` を合成することはないためクォート
/// 安全性には影響しないが、生成されるコマンドが本来のファイルを指さなく
/// なりうる）。
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
                "backend-cuda mma_ptx_dump: CUDA driver unavailable ({detail}); \
                 this diagnostic requires a CUDA-equipped runner with the CUDA 13.0 \
                 toolkit (ptxas)."
            );
            std::process::exit(1);
        }
        Err(other) => {
            eprintln!("backend-cuda mma_ptx_dump: CudaDevice::new failed ({other})");
            std::process::exit(1);
        }
    };

    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!(
            "backend-cuda mma_ptx_dump: failed to create output directory {} ({e})",
            out_dir.display()
        );
        std::process::exit(1);
    }

    // codex-review P0 是正（PR #784 イシュー #782）: `--out-dir` が
    // symlink（またはその途中要素が symlink）の場合でも `create_dir_all`
    // はリンク先を辿って成功しうる。`canonicalize` でシンボリックリンクを
    // 解決した実パスへ確定させ、かつ解決先が実在するディレクトリで
    // あることを検査してから、以降のファイル書き出しに使う（この
    // 実パスに対して `dump_ptx` の `create_new` が効くようにする）。
    let out_dir = match std::fs::canonicalize(&out_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "backend-cuda mma_ptx_dump: failed to canonicalize output directory {} ({e})",
                out_dir.display()
            );
            std::process::exit(1);
        }
    };
    match out_dir.metadata() {
        Ok(m) if m.is_dir() => {}
        Ok(_) => {
            eprintln!(
                "backend-cuda mma_ptx_dump: output path {} resolves to a non-directory",
                out_dir.display()
            );
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!(
                "backend-cuda mma_ptx_dump: failed to stat output directory {} ({e})",
                out_dir.display()
            );
            std::process::exit(1);
        }
    }

    // NVRTC へ渡す arch は `device.arch()`（`compute_XY` 形式）をそのまま
    // 使う。本番経路（`gemm_mma.rs::CudaMmaGemm::new`）も同じ形式で
    // `compile_ptx` を呼ぶため、ここで異なる形式を渡すと本番と異なる
    // コンパイル結果のレジスタ使用量を観測してしまう（advisor 指摘の
    // 是正）。`sm_arch_string` は ptxas 手順出力専用（`sm_XY` 形式）。
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
                "backend-cuda mma_ptx_dump: device.multiprocessor_count() returned None or 0; \
                 cannot derive the dynamic swizzle group width without a real SM count."
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

    let base_path = out_dir.join("mma_f16_base.ptx");
    if let Err(e) = dump_ptx(
        diagnostics::mma_f16_source(),
        &nvrtc_arch,
        &base_path,
        "base",
    ) {
        eprintln!("backend-cuda mma_ptx_dump: {e}");
        std::process::exit(1);
    }

    let group_width = diagnostics::mma_swizzle_group_width(num_sms);
    println!("swizzle: dynamic group_width={group_width} (derived from num_sms={num_sms})");

    let swizzle_source = match diagnostics::mma_f16_source_with_swizzle(group_width) {
        Ok(src) => src,
        Err(e) => {
            eprintln!(
                "backend-cuda mma_ptx_dump: mma_f16_source_with_swizzle({group_width}) failed \
                 ({e})"
            );
            std::process::exit(1);
        }
    };
    let swizzle_path = out_dir.join(format!("mma_f16_swizzle_g{group_width}.ptx"));
    if let Err(e) = dump_ptx(&swizzle_source, &nvrtc_arch, &swizzle_path, "swizzle") {
        eprintln!("backend-cuda mma_ptx_dump: {e}");
        std::process::exit(1);
    }

    // `-o /dev/null` は toolkit バージョンによって ptxas のレジスタ割り当て
    // 完了前に打ち切られる可能性があるため使わず、スクラッチファイルへ
    // cubin を出力させる（advisor 指摘の是正。cubin 自体は破棄してよく
    // `-v` の stderr ログのみが目的）。
    //
    // codex-review P0 是正（PR #784 イシュー #782）: `base`/`swizzle` は
    // `--out-dir`（オペレーター入力）由来のパスを含むため、貼り付け実行を
    // 想定した手順文字列へ未クォートで埋め込むとシェルインジェクションが
    // 成立しうる（空白・`;`・`$()` を含むパス等）。`shell_quote` で各パスを
    // 単一引用符クォートしたうえで埋め込む。
    let base_q = shell_quote(&base_path.display().to_string());
    let base_cubin_q = shell_quote(&format!("{}.cubin", base_path.display()));
    let swizzle_q = shell_quote(&swizzle_path.display().to_string());
    let swizzle_cubin_q = shell_quote(&format!("{}.cubin", swizzle_path.display()));

    let mut ptxas_lines = vec![
        format!("ptxas -arch={ptxas_arch} -v {base_q} -o {base_cubin_q}"),
        format!("ptxas -arch={ptxas_arch} -v {swizzle_q} -o {swizzle_cubin_q}"),
    ];

    // イシュー #803: warp タイル拡大候補（本ファイル冒頭コメント「warp
    // タイル拡大候補」節）を launch_bounds なし/あり（値=導出スレッド数）の
    // 2 通りずつダンプする。`(warp_tiles_m, warp_tiles_n)` は
    // `docs/perf/cuda-gemm-mma-warp-tile-register-budget.md` §3.1 候補表と
    // 一致させる（2x2 現行を含む: 既定構成との差分比較の基準点として
    // 必要）。`threads`（`launch_bounds` に渡す導出スレッド数）は候補表の
    // 値をそのままハードコードする。`diagnostics::mma_f16_source_with_warp_tiles`
    // 自体がブロックスレッド数を warp タイル構成から導出し、`launch_bounds`
    // に `Some(v)` を渡した場合は `v` が導出値と不一致なら fail-closed で
    // `CudaError::InvalidKernelConfig` を返す（`kernels_mma.rs` 該当エラー
    // 分岐参照）ため、ここでハードコードした `threads` の値が誤っていれば
    // 本ダンプ自体が Err で失敗し検知される。候補表・導出ロジックの整合は
    // `kernels_mma.rs` 側ユニットテスト
    // （`mma_f16_source_with_warp_tiles_replaces_defines_for_each_candidate`）が
    // pin していることに依拠する。
    for (warp_tiles_m, warp_tiles_n, threads) in
        [(2u32, 2u32, 512u32), (2, 4, 256), (4, 2, 256), (4, 4, 128)]
    {
        for launch_bounds in [None, Some(threads)] {
            let label = match launch_bounds {
                None => format!("wt{warp_tiles_m}x{warp_tiles_n}"),
                Some(v) => format!("wt{warp_tiles_m}x{warp_tiles_n}_lb{v}"),
            };
            let source = match diagnostics::mma_f16_source_with_warp_tiles(
                warp_tiles_m,
                warp_tiles_n,
                launch_bounds,
            ) {
                Ok(src) => src,
                Err(e) => {
                    eprintln!(
                        "backend-cuda mma_ptx_dump: mma_f16_source_with_warp_tiles(\
                         {warp_tiles_m}, {warp_tiles_n}, {launch_bounds:?}) failed ({e})"
                    );
                    std::process::exit(1);
                }
            };
            let path = out_dir.join(format!("mma_f16_{label}.ptx"));
            if let Err(e) = dump_ptx(&source, &nvrtc_arch, &path, &label) {
                eprintln!("backend-cuda mma_ptx_dump: {e}");
                std::process::exit(1);
            }
            let path_q = shell_quote(&path.display().to_string());
            let cubin_q = shell_quote(&format!("{}.cubin", path.display()));
            ptxas_lines.push(format!("ptxas -arch={ptxas_arch} -v {path_q} -o {cubin_q}"));
        }
    }

    // 冒頭の `out_dir={}` は `key=value` 形式の情報表示行であり、下記の
    // `ptxas ...` コマンド行（オペレーターがそのまま端末へ貼り付けて実行する
    // 想定）とは異なり、シェルコマンドとして貼り付けられる文脈ではない
    // ため未クォートのままとする（クォートすると `out_dir='...'` のように
    // 値として読みづらくなるだけで安全性向上に寄与しない）。
    let ptxas_commands = ptxas_lines
        .iter()
        .map(|line| format!("\x20   {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    println!(
        "done. out_dir={} nvrtc_arch={nvrtc_arch} ptxas_arch={ptxas_arch} \
         group_width={group_width}. Run the following to inspect register/spill counts \
         (docs/perf/cuda-gemm-swizzle-ab.md §3・docs/perf/\
         cuda-gemm-mma-warp-tile-register-budget.md §4):\n{ptxas_commands}",
        out_dir.display(),
    );
}

/// `internal-diagnostics` feature 未指定時の no-op（本ファイル冒頭
/// コメント参照。`cargo build -p backend-cuda --example mma_ptx_dump`
/// が feature なしでもビルド成立することを保証する）。
#[cfg(not(feature = "internal-diagnostics"))]
fn main() {
    println!(
        "backend-cuda mma_ptx_dump: requires --features internal-diagnostics; \
         see this file's doc comment. Nothing to do."
    );
}
