//! Tensor Core（WMMA）経路の誤差分布実測ハーネス（TASK-11.1g・#186）。
//!
//! REQ-2 受け入れ基準「tensor core（WMMA/mma）化で TF32／f16 累算経路を
//! 導入する際は当該経路の数値一致閾値を実測に基づき再評価する」の実測
//! ステップを担う。TF32 経路（[`backend_cuda::CudaGemm::run_wmma_tf32`]。
//! `tests/gemm_wmma_tf32.rs`）・f16 WMMA 経路
//! （[`backend_cuda::CudaWmmaGemm::run_f16`]。`tests/cpu_cuda_wmma_parity.rs`）
//! それぞれについて、形状×シードごとに `backend_cpu::compare` の
//! `CompareReport`（誤差分布統計）を収集し、REQ-2 統一複合判定の閾値
//! （`backend_cpu::RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）に
//! 対する閾値マージンを算出して Markdown 表形式で stdout へ出力する
//! （`docs/perf/cuda-tensor-core-tolerance-evaluation.md` へ転記する
//! 想定）。
//!
//! **本ハーネスは閾値・判定式を一切変更しない**。`backend_cpu::parity`
//! の定数・`compare` 関数をそのまま import して使い、ローカル複製・
//! 緩和は行わない（`.claude/rules/coding-rust.md`「バックエンド間数値
//! 一致テストの許容誤差を単独で緩和しない」・`delegation-impl.md`
//! 「実装 Agent にガードレール閾値・テスト許容誤差を緩和させない」）。
//!
//! `examples/` に置くのは `gemm_bench.rs`（`crates/backend-cpu/examples/`）
//! と同じ理由: `dev-dependencies`（`bench-harness`・`backend-cpu`）を
//! 利用しつつ通常の `cargo test`／CI では実行されず、ビルド検証のみが
//! CI を通過させるためである（self-hosted runner をベンチ実行で占有
//! しない。`ci.md`）。
//!
//! # 使い方
//!
//! ```text
//! cargo build --release -p backend-cuda --example wmma_tolerance_probe
//! ./target/release/examples/wmma_tolerance_probe
//! ```
//!
//! CUDA driver／NVRTC 非搭載環境、または Tensor Core 非対応（compute
//! capability 7.0/8.0 未満）環境では、経路ごとに理由を表示して正常終了
//! する（`tests/gemm_wmma_tf32.rs`・`tests/cpu_cuda_wmma_parity.rs` と
//! 同じ環境適応の分岐パターン）。一方、shape 検証失敗・`compare` の長さ
//! 不一致・WMMA 起動時エラー等の**意図しない**計測エラーは行を出力した
//! うえで exit code 1 を返す（環境非対応の意図的スキップと区別する。
//! #186・PR #257 Codex Review 指摘対応）。
//!
//! CUDA toolkit 非搭載環境で NVRTC をプロセス限定プロビジョニングした
//! 場合、`LD_LIBRARY_PATH`・`CUDA_INCLUDE_PATH` は **`cargo build` 時だけ
//! でなく実行時（本バイナリの起動時）にも必要**である。NVRTC は
//! `CudaGemm::new`／`CudaWmmaGemm::new` 呼び出し時に実行時コンパイルを
//! 行う（動的ロードした `libnvrtc` を使う）ため、ビルド後に環境変数が
//! 失われた状態で実行すると NVRTC 利用不可と誤報告される
//! （`docs/perf/cuda-tensor-core-tolerance-evaluation.md` §1 の手順参照。
//! #186・PR #257 Codex Review 指摘対応）。

use backend_cpu::{
    ABSOLUTE_RESCUE_THRESHOLD, CompareReport, RELATIVE_TOLERANCE, compare, matmul_reference_fma,
};
use backend_cuda::{CudaDevice, CudaError, CudaGemm, CudaWmmaGemm};
use bench_harness::rng::Xorshift64Star;
use half::f16;

/// 形状セット（`tests/gemm_wmma_tf32.rs`・`tests/cpu_cuda_wmma_parity.rs` の
/// `#[ignore]` 形状網羅テストと同じ形状に、K スイープ（M=N=256 固定で
/// K=256/512/1024/4096）を追加し、桁落ち蓄積と K の関係を見る）。
///
/// **両 ignored parity suite の全形状を含む**（#186・PR #257 Codex Review
/// 指摘「probe の shape リストが TF32 の (1,1,1) と f16 の (17,19,23) を
/// 欠いており、両 ignored parity suite を網羅していると主張しているが実際は
/// 網羅していない」対応）:
/// - `tests/gemm_wmma_tf32.rs::wmma_tf32_matches_reference_across_shapes` の
///   `(1, 1, 1)`（K タイル未満の極小形状）を追加した。
/// - `tests/cpu_cuda_wmma_parity.rs::wmma_f16_matches_reference_across_shapes`
///   の `(17, 19, 23)`（m=17, n=19, k=23）を追加した。TF32 側の
///   `(17, 23, 19)`（m=17, n=23, k=19）とは n・k が入れ替わっており別形状
///   のため、いずれも残している。
///
/// **意図的な重複**: 「256x256x256 (block tile x8)」と「256x256x256
/// (K sweep base)」は m=n=k=256 で完全に同一の形状であり、シード導出式
/// （形状インデックスに依存しない `m` 由来のシード）により同一入力・同一
/// 結果になる。前者は既存 `#[ignore]` テストの形状網羅網（block tile 系列
/// の完成）としての意味、後者は K スイープ（256/512/1024/4096）の起点として
/// の意味を持たせるため、重複を承知のうえで両方残している
/// （`docs/perf/cuda-tensor-core-tolerance-evaluation.md` §2 の実測表では
/// 重複分をマージして 1 行に記録する）。
///
/// **K=512 スイープ点**（#186・PR #257 Codex Review 指摘「K 依存の単調傾向の
/// 主張が 256/1024/4096 の sweep と 512x512x512（M/N・シードも変わる）の
/// 結果を混在させている」対応）: 「256x256x512 (K sweep)」を追加し、
/// M=N=256 固定のまま K のみを 256→512→1024→4096 と揃えて比較できるように
/// した。既存の「512x512x512」行は M/N も変わる別条件（シード導出も異なる）
/// のため、K 単調性の主張には流用しない。
const SHAPES: &[(&str, u32, u32, u32)] = &[
    ("32x32x32 (block tile)", 32, 32, 32),
    ("64x64x64 (block tile x2)", 64, 64, 64),
    ("128x128x128 (block tile x4)", 128, 128, 128),
    ("256x256x256 (block tile x8)", 256, 256, 256),
    ("512x512x512 (block tile x16)", 512, 512, 512),
    ("1x1x1 (sub-K-tile, TF32 suite)", 1, 1, 1),
    ("17x23x19 (non-multiple edge, TF32 suite)", 17, 23, 19),
    ("17x19x23 (non-multiple edge, f16 suite)", 17, 19, 23),
    ("33x31x65 (non-multiple edge)", 33, 31, 65),
    ("100x100x100 (non-multiple edge)", 100, 100, 100),
    ("130x70x90 (non-multiple edge)", 130, 70, 90),
    ("64x96x128 (non-square)", 64, 96, 128),
    ("256x256x256 (K sweep base)", 256, 256, 256),
    ("256x256x512 (K sweep)", 256, 256, 512),
    ("256x256x1024 (K sweep)", 256, 256, 1024),
    ("256x256x4096 (K sweep, PoC-v2-5 stress)", 256, 256, 4096),
];

/// 各形状 5 シード（5 回計測の中央値方針〈coding-rust.md〉に整合させ、
/// 単一シードの偶然の一致・不一致に結論を左右されないようにする）。
const SEEDS: &[u64] = &[1, 2, 3, 4, 5];

fn margin(threshold: f64, observed: f64) -> String {
    if observed <= 0.0 {
        "inf".to_string()
    } else {
        format!("{:.2}x", threshold / observed)
    }
}

/// `report` の全項目を Markdown 表 1 行として出力する。
///
/// `CompareReport` の全フィールド（`p50`/`p99`/`p999_abs_diff`・
/// `max_fail_abs_diff` を含む）を出力に含める（#186・PR #257 Codex Review
/// 指摘「出力が p50_abs_diff / p99_abs_diff / p999_abs_diff を破棄しており、
/// 再実行で CompareReport 全フィールドが再現できるという説明と矛盾する」
/// 対応。ファイル冒頭の doc コメントが謳う「`CompareReport`（誤差分布統計）を
/// 収集」という説明を出力自体で満たす）。
fn report_row(context: &str, seed: u64, report: &CompareReport) {
    println!(
        "| {context} | {seed} | {}/{} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {:.3e} | {} | {} |",
        report.fail_count,
        report.total,
        report.max_abs_diff,
        report.mean_abs_diff,
        report.max_rel_err,
        report.mean_rel_err,
        report.p50_abs_diff,
        report.p99_abs_diff,
        report.p999_abs_diff,
        report.max_fail_abs_diff,
        margin(ABSOLUTE_RESCUE_THRESHOLD, report.max_abs_diff),
        margin(RELATIVE_TOLERANCE, report.max_rel_err),
    );
}

fn table_header() {
    println!(
        "| shape | seed | fail/total | max_abs_diff | mean_abs_diff | max_rel_err | mean_rel_err | p50_abs_diff | p99_abs_diff | p999_abs_diff | max_fail_abs_diff | abs margin (1e-5/max) | rel margin (1e-3/max) |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
}

/// エラー行を Markdown 表フォーマット（列数を `report_row` と揃える）で
/// 出力する。空欄列数は `table_header` の列数（13 列）に合わせている。
fn error_row(label: &str, seed: u64, reason: &str) {
    println!("| {label} | {seed} | ({reason}) | - | - | - | - | - | - | - | - | - | - |");
}

/// TF32 経路（[`CudaGemm::run_wmma_tf32`]）の誤差分布を形状×シードごとに
/// 計測する。CPU 参照は `matmul_reference_fma`（FMA 契約の唯一の参照点。
/// `tests/gemm_wmma_tf32.rs::assert_wmma_tf32_parity` と同じ比較方法）。
///
/// 戻り値 `true` は「意図しない計測エラー（shape 検証失敗・`compare` の
/// 長さ不一致・WMMA 起動時エラー）が 1 件以上発生した」ことを示す。
/// `CudaError::WmmaUnavailable`（Tensor Core 非対応環境）は意図的スキップ
/// であり `false` のまま `return` する（#186・PR #257 Codex Review 指摘
/// 「stress shape で allocation/launch/execution エラーが起きても行を
/// 出力するだけでプログラムは exit 0 のままになり、部分計測を完了と
/// 誤認しうる」対応。行は最後まで出力を続け、失敗有無だけを呼び出し元
/// `main` へ伝播する）。
fn probe_tf32(gemm: &CudaGemm) -> bool {
    println!("\n## TF32 (`CudaGemm::run_wmma_tf32`)\n");
    table_header();
    let mut had_unexpected_error = false;
    for &(label, m, n, k) in SHAPES {
        for &seed in SEEDS {
            let mut rng = Xorshift64Star::new(seed.wrapping_mul(1000).wrapping_add(m as u64));
            let a = rng.fill_vec((m as usize) * (k as usize));
            let b = rng.fill_vec((k as usize) * (n as usize));

            let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
            if matmul_reference_fma(&a, &b, &mut c_ref, m as usize, n as usize, k as usize).is_err()
            {
                // SHAPES は固定の well-formed 形状のみのため、ここに到達する
                // のは想定外（shape 定義のバグ）である。
                error_row(label, seed, "unexpected: shape validation error");
                had_unexpected_error = true;
                continue;
            }

            match gemm.run_wmma_tf32(&a, &b, m, n, k) {
                Ok(c_gpu) => match compare(&c_gpu, &c_ref) {
                    Ok(report) => report_row(label, seed, &report),
                    Err(err) => {
                        error_row(label, seed, &format!("unexpected: compare error: {err}"));
                        had_unexpected_error = true;
                    }
                },
                Err(CudaError::WmmaUnavailable { detail }) => {
                    println!("\n(TF32 WMMA unavailable: {detail})\n");
                    return had_unexpected_error;
                }
                Err(other) => {
                    error_row(label, seed, &format!("unexpected: run error: {other}"));
                    had_unexpected_error = true;
                }
            }
        }
    }
    had_unexpected_error
}

/// f16 WMMA 経路（[`CudaWmmaGemm::run_f16`]）の誤差分布を形状×シード
/// ごとに計測する。参照方法は `tests/cpu_cuda_wmma_parity.rs` 冒頭コメント
/// の 3 手順（f16→f32→参照 matmul→f16 丸め→f32）をそのまま踏襲する。
///
/// 戻り値の意味は [`probe_tf32`] と同じ（意図しないエラー発生の有無）。
fn probe_f16(gemm: &CudaWmmaGemm) -> bool {
    println!("\n## f16 WMMA (`CudaWmmaGemm::run_f16`)\n");
    table_header();
    let mut had_unexpected_error = false;
    for &(label, m, n, k) in SHAPES {
        for &seed in SEEDS {
            let mut rng = Xorshift64Star::new(seed.wrapping_mul(2000).wrapping_add(m as u64));
            let a_f16: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
            let b_f16: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));
            let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
            let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();

            let mut c_ref_f32 = vec![0.0f32; (m as usize) * (n as usize)];
            if matmul_reference_fma(
                &a_f32,
                &b_f32,
                &mut c_ref_f32,
                m as usize,
                n as usize,
                k as usize,
            )
            .is_err()
            {
                error_row(label, seed, "unexpected: shape validation error");
                had_unexpected_error = true;
                continue;
            }
            let c_ref_rounded: Vec<f32> = c_ref_f32
                .iter()
                .map(|&x| f16::from_f32(x).to_f32())
                .collect();

            match gemm.run_f16(&a_f16, &b_f16, m, n, k) {
                Ok(c_gpu_f16) => {
                    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();
                    match compare(&c_gpu_f32, &c_ref_rounded) {
                        Ok(report) => report_row(label, seed, &report),
                        Err(err) => {
                            error_row(label, seed, &format!("unexpected: compare error: {err}"));
                            had_unexpected_error = true;
                        }
                    }
                }
                Err(other) => {
                    error_row(label, seed, &format!("unexpected: run error: {other}"));
                    had_unexpected_error = true;
                }
            }
        }
    }
    had_unexpected_error
}

/// `main` の終了コード。CUDA driver／NVRTC 非搭載・Tensor Core 非対応
/// 環境での意図的スキップは exit 0 のまま維持し、それ以外の想定外エラー
/// （`CudaDevice::new`／`CudaGemm::new`／`CudaWmmaGemm::new` の想定外エラー、
/// または [`probe_tf32`]・[`probe_f16`] 内の想定外エラー）は exit 1 を返す
/// （#186・PR #257 Codex Review 指摘「stress shape で
/// allocation/launch/execution エラーが起きても行を出力するだけでプログラム
/// は exit 0 のままになり、部分計測を完了と誤認しうる」対応）。
fn main() -> std::process::ExitCode {
    use std::process::ExitCode;

    println!("# WMMA Tensor Core 経路 誤差分布実測（TASK-11.1g・#186）\n");
    println!(
        "閾値（REQ-2 統一複合判定・変更対象外）: RELATIVE_TOLERANCE={RELATIVE_TOLERANCE:e}, \
         ABSOLUTE_RESCUE_THRESHOLD={ABSOLUTE_RESCUE_THRESHOLD:e}\n"
    );

    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!("CUDA driver 非搭載環境のため計測をスキップします: {detail}");
            return ExitCode::SUCCESS;
        }
        Err(CudaError::Driver(err)) => {
            println!("CUDA driver 初期化に失敗したため計測をスキップします: {err:?}");
            return ExitCode::SUCCESS;
        }
        Err(other) => {
            println!("CudaDevice::new が想定外のエラーを返しました: {other}");
            return ExitCode::FAILURE;
        }
    };
    println!("device compute capability: {}", device.arch());

    let mut had_unexpected_error = false;

    match CudaGemm::new(&device) {
        Ok(gemm) => {
            if probe_tf32(&gemm) {
                had_unexpected_error = true;
            }
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            println!("\nNVRTC 非搭載環境のため TF32 経路の計測をスキップします: {detail}");
        }
        Err(other) => {
            println!("\nCudaGemm::new が想定外のエラーを返しました: {other}");
            had_unexpected_error = true;
        }
    }

    match CudaWmmaGemm::new(&device) {
        Ok(gemm) => {
            if probe_f16(&gemm) {
                had_unexpected_error = true;
            }
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            println!("\nNVRTC 非搭載環境のため f16 WMMA 経路の計測をスキップします: {detail}");
        }
        Err(CudaError::TensorCoreUnsupported { detail }) => {
            println!(
                "\ncompute capability 7.0 未満のため f16 WMMA 経路の計測をスキップします: {detail}"
            );
        }
        Err(other) => {
            println!("\nCudaWmmaGemm::new が想定外のエラーを返しました: {other}");
            had_unexpected_error = true;
        }
    }

    if had_unexpected_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
