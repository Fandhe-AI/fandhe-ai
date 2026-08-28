//! CUDA GEMM の固定初期化コスト（イシュー #926。フレームワーク横並びベンチ
//! `scripts/bench/framework-compare/`・PR #915 実測で fandhe-ai の CUDA GEMM
//! 中央値が行列サイズにほぼ非依存の約 440〜460 ms 帯へ張り付いた事象）の
//! フェーズ分解診断テスト。
//!
//! # なぜ `crates/backend-cuda/tests/`（integration test）ではなく本ファイル
//! （`lib.rs` 直下の兄弟モジュール）に置くか
//!
//! `kernels`／`kernels_wmma_opt`（非公開 `mod`。カーネルソース定数・関数は
//! `pub` だがモジュール自体は非公開）へアクセスするため、integration test
//! （crate 外部と同じ扱い）では到達できない。`lib.rs` の兄弟モジュールとして
//! 宣言すれば、Rust の可視性規則（非公開アイテムは定義モジュールおよびその
//! 子孫から可視）によりクレートルートの子孫は互いに `crate::kernels::…` の
//! ような完全パスで到達できる（`gemm.rs` 自身の `use crate::kernels;` と
//! 同じ扱い）。`jit_cache_bench_tests.rs`（イシュー #534）が `nvrtc.rs` の
//! 子モジュールとして同じ理由で配置されているのと同型の判断。
//!
//! # 計測対象と帰属の明確化（イシュー #926 本文の解釈）
//!
//! `scripts/bench/framework-compare/bench-fandhe/src/main.rs::run_gemm` は
//! 計測イテレーションごとに `facade::tape_for` → `tape.var` を行うが、
//! `Instant::now()` は **`matmul` 呼び出し直前**に取る。`tape_for`
//! （`resolve_ops` → デバイス存在プローブ + `CudaBackendOps::new`。
//! `ordinal` 保持のみで driver 非接触）自体は計測区間の**外**であり、
//! 計測区間に入るのは `matmul` 呼び出し時に発生する **遅延初期化**
//! （`ops::CudaBackendOps::device_handle` → `CudaDevice::new` と
//! `gemm::CudaGemm::new` を毎呼び出しごとに都度構築する設計。`ops.rs`
//! `CudaBackendOps` 構造体ドキュメンテーションコメント「TASK-1.9b の
//! ハンドル常駐が未着地」参照）である。イシュータイトルの
//! 「`tape_for` 初期化コスト」はこの遅延初期化を指すと解釈し、本ファイルは
//! その内訳（(p1) デバイス・コンテキスト生成／(p2) NVRTC source→PTX
//! コンパイル ×8／(p3) driver `load_module`/`load_function`（PTX→SASS
//! JIT）×8／(p4) 実 GEMM 実行）を個別計測する。
//!
//! # 8 カーネルの内訳
//!
//! `gemm::CudaGemm::new` が 1 回の構築で NVRTC コンパイル＋ロードする
//! 8 カーネル（naive f32/f16・tiled f32/f16・tiled_bias_act_f32 の 5 本は
//! `?` 早期 return に合流・WMMA TF32 基本/opt/staged の 3 本は失敗を
//! `Option` へ退避するフォールバック方式。`gemm.rs::CudaGemm::new`
//! ドキュメンテーションコメント参照）と全く同じソース・関数名の組を
//! ここでも使う（[`crate::gemm::kernel_specs`]。本番側と本ファイルが
//! 手作業複製で乖離しないよう、`gemm.rs` 側の単一の真実源をそのまま
//! 呼び出す。Review #945 P2 指摘）。DGX Spark GB10（sm_121・Blackwell
//! 系。compute capability 8.0 以上）を計測対象実機として想定し、WMMA TF32
//! 3 本もコンパイル成功を前提とする（本番同様の非致命扱いはしない。実機
//! 到達不能なローカル環境では `#[ignore]` によりこの前提の是非自体が
//! 検証されない）。
//!
//! # なぜドライバ側 JIT キャッシュ（`CUDA_CACHE_PATH`）の cold/warm を
//! 子プロセス分離せず単一条件で計測するか（実装計画からのスコープ縮小）
//!
//! `bench-harness::startup`（TASK-13.1a・#170）はプロセスレベルの
//! cold/warm 比較を子プロセスの環境変数（`Command::env("CUDA_CACHE_PATH",
//! …)`）で実現する（自プロセスの環境変数を書き換えない設計。
//! `crates/bench-harness/src/startup.rs` モジュール冒頭コメント参照）。
//! `nvrtc.rs::resolve_cache_root_impl` のドキュメンテーションコメントも
//! 同じ理由（edition 2024 で `std::env::set_var` が `unsafe` になり、
//! テストバイナリ全体で共有されるグローバル状態を書き換えると決定的な
//! 再現ができない）で「注入で決定化」パターンへの統一を明記している。
//! 本ファイルは軽量な in-process フェーズ計測に徹する方針のため、子
//! プロセス分離までは行わず、**プロセスのアンビエント `CUDA_CACHE_PATH`
//! （通常は実機ランナーの `~/.nv/ComputeCache`。実質的に「ウォーム」
//! 相当）1 条件のみ**を計測する。ドライバ側 JIT キャッシュの cold 条件
//! （新規空ディレクトリでの初回 JIT）を厳密に分離した計測は本イシューの
//! スコープ外とし、実装完了報告の `outOfScope` に記録する。
//!
//! この「子プロセス分離をしない」スコープ縮小は、**同一プロセス内での
//! `CudaDevice::new(0)` の反復呼び出しが本番のフレッシュプロセス相当には
//! ならない**という副作用も伴う（Review #945 指摘）。`cudarc` の
//! `CudaDevice::new` は device ordinal に対する CUDA の *primary context*
//! を取得する構成であり、同一プロセス・同一 ordinal では新規に構築した
//! ハンドルであっても同じ primary context に接続し、driver 側の
//! PTX→SASS JIT 結果（インメモリ／`CUDA_CACHE_PATH` ディスクキャッシュ
//! いずれも）を共有する。したがって [`measure_one_trial`] 内で
//! `device_for_gemm_new`（後述）を独立したハンドルとして構築しても、
//! 直前の診断ループ・過去の試行が同じ 8 カーネルを既に driver ロード
//! 済みであれば `gemm_new_secs`（(p3) 相当分）はウォーム JIT を計測する。
//! 同様に `MEASURED_TRIALS` の 2 回目以降は `load_secs`（(p3)）も同じ
//! 理由でウォーム計測になる。§6 の実測値を解釈する際は、(p3)・
//! `gemm_new_secs` を「本番のコールドプロセス初回呼び出しの上界」では
//! なく**下界（ウォーム条件下の値）**として扱う。
//!
//! # 実行時は必ず `--test-threads=1`（Review #945 指摘: GPU 上での競合）
//!
//! 本ファイルの 3 テスト（[`init_cost_diag_phase_breakdown`]・
//! [`init_cost_diag_e2e_matches_framework_compare_shape`]・
//! [`init_cost_diag_reused_handle_steady_state_reference`]）はいずれも
//! `CudaDevice::new(0)`（device 0）を使う。`cargo test` の既定テストハーネスは
//! 複数スレッドで `#[test]` 関数を並行実行するため、既定のまま `--ignored` で
//! まとめて起動すると 3 テストの `Instant` 計測区間が同一 GPU 上で競合し、
//! 本ファイルが記録するフェーズ計測（(p1)〜(p4)・e2e gemm・再利用ハンドル）に
//! カーネル起動待ち・SM 占有の競合が混入して値が歪みうる。§5.3
//! （`docs/perf/cuda-tape-init-cost-diagnosis.md`）の実行コマンドは
//! `--test-threads=1` を明示し、この 3 テストを直列実行する。
//!
//! # gating しない方針（Review #534 と同じ理由）
//!
//! 本ファイルの `#[test]` はすべて **実行が成功すること**（NVRTC
//! コンパイル・driver ロード・カーネル起動が例外なく完了すること）のみを
//! 検証条件とし、フェーズ間の大小関係・絶対値への `assert!` は行わない。
//! GPU クロック挙動・他プロセス競合等の環境揺らぎをタイミング値の hard
//! assert に持ち込むと実機ランナー上で flaky 化するため
//! （`jit_cache_bench_tests.rs` 冒頭コメント「hard assert を撤去した理由」
//! と同じ判断）。数値は `println!` に残し、`docs/perf/
//! cuda-tape-init-cost-diagnosis.md` へ転記する一次情報とする。

use std::time::Instant;

use fandhe_ai_tensor_core::{BackendOps, Tensor};

use bench_harness::{Quartiles, median_q1_q3, rng::Xorshift64Star};

use crate::device::CudaDevice;
use crate::gemm::{CudaGemm, kernel_specs};
use crate::nvrtc::compile_ptx;
use crate::ops::CudaBackendOps;

/// ウォームアップ試行数（統計に含めない。GPU クロック遷移・ドライバ内部
/// 状態の安定化目的。`bench-harness::startup::DEFAULT_STARTUP_TRIALS` の
/// 「5 回計測中央値」方針とは別に、本ファイルは実装計画 §4.1「warmup 3
/// trial + 計測 10 trial」に従う）。
const WARMUP_TRIALS: usize = 3;

/// 統計に採用する計測試行数（`.claude/rules/coding-rust.md`「ベンチは
/// 5 回計測の中央値を採用し」の下限 5 を超える 10 回とし、Q1/Q3 の分散も
/// より安定させる）。
const MEASURED_TRIALS: usize = 10;

/// 決定的シードで `n x n` の f32 正方行列 2 枚（A・B）を生成する
/// （`jit_cache_bench_tests::gen_ab` と同じ「決定的シード PRNG」方針。
/// `.claude/rules/coding-rust.md`）。
fn gen_square_ab(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let a = rng.fill_vec(n * n);
    let b = rng.fill_vec(n * n);
    (a, b)
}

/// 1 試行分の測定結果。
struct TrialSample {
    /// (p1) `CudaDevice::new` の所要時間。
    device_new_secs: f64,
    /// (p2) 8 カーネルそれぞれの NVRTC `compile_ptx` 所要時間
    /// （カーネル順は [`kernel_specs`] と対応）。
    compile_secs: [f64; 8],
    /// (p3) 8 カーネルそれぞれの `load_module` + `load_function` 所要時間。
    load_secs: [f64; 8],
    /// (p2)+(p3) を本番と同一の呼び出し列（`gemm::CudaGemm::new` 単体
    /// 呼び出し）で計測した合計値。上記の個別計測は診断用の別ハンドルで
    /// 行うため、本番呼び出し 1 回分の実測値をここに独立して残し、
    /// 個別計測の合算値との整合確認に使う。
    gemm_new_secs: f64,
    /// (p4) `run_tiled_f32`（N=1024）の所要時間。
    run_tiled_f32_secs: f64,
}

/// N=1024 の `run_tiled_f32` 入力（全試行で使い回す固定サイズ・固定シード
/// データ。`.claude/rules/coding-rust.md` の決定的シード方針）。
const P4_MATMUL_N: usize = 1024;

fn measure_one_trial() -> TrialSample {
    // (p1) デバイス・コンテキスト生成。
    let t_device = Instant::now();
    let device = CudaDevice::new(0)
        .expect("CUDA device must be available on the ignored diagnostic bench runner");
    let device_new_secs = t_device.elapsed().as_secs_f64();

    // (p2)/(p3) 8 カーネルを個別に compile_ptx → load_module/load_function
    // する診断専用の呼び出し列（`gemm::CudaGemm::new` 本体はこれと同じ
    // 手順を 1 関数にまとめて実行する。本ファイル冒頭「8 カーネルの
    // 内訳」参照）。
    let mut compile_secs = [0.0_f64; 8];
    let mut load_secs = [0.0_f64; 8];
    for (i, (label, source, func_name)) in kernel_specs().iter().enumerate() {
        let t_compile = Instant::now();
        let ptx = compile_ptx(source, device.arch()).unwrap_or_else(|e| {
            panic!(
                "NVRTC compile of production kernel source `{label}` must succeed \
                 on a compute-capability >= 8.0 real device (DGX Spark GB10 assumed): {e}"
            )
        });
        compile_secs[i] = t_compile.elapsed().as_secs_f64();

        let t_load = Instant::now();
        let _func = device
            .context()
            .load_module(ptx)
            .and_then(|module| module.load_function(func_name))
            .unwrap_or_else(|e| {
                panic!("driver load_module/load_function for `{label}` must succeed: {e:?}")
            });
        load_secs[i] = t_load.elapsed().as_secs_f64();
    }

    // 本番と同一の単一呼び出し（`CudaGemm::new`）での (p2)+(p3) 合計実測。
    //
    // Review #945 指摘: 本番経路（`ops.rs::CudaBackendOps::gemm`）は呼び出し
    // ごとに新規 `CudaDevice::new` を都度構築する設計（`ops.rs`
    // `CudaBackendOps` 構造体ドキュメンテーションコメント参照）であり、直前に
    // 同一デバイス上で同じ 8 カーネルを compile_ptx／load_module 済みという
    // 状態は本番には存在しない。上の診断ループで使った `device` をそのまま
    // 使うと同一ハンドル内に保持された module 常駐状態の恩恵を受け
    // `gemm_new_secs` が偏るため、ハンドルは独立した新規デバイスで計測する
    // （この新規デバイス生成自体の所要時間はどの集計にも含めない。あくまで
    // `gemm_new_secs` を汚染しないための隔離目的）。
    //
    // ただし本ファイル冒頭「なぜドライバ側 JIT キャッシュ…単一条件で計測
    // するか」節（Review #945 再指摘）のとおり、これはハンドルレベルの
    // module 常駐を排除するのみであり、**同一プロセス・同一 device ordinal
    // が接続する CUDA primary context・driver 側 JIT キャッシュまでは隔離
    // しない**。よって `gemm_new_secs` は本番のコールドプロセス初回呼び出し
    // と忠実に対応する値ではなく、ウォーム条件下の下界として扱う。
    let device_for_gemm_new = CudaDevice::new(0).expect(
        "CUDA device must be available on the ignored diagnostic bench runner \
         (isolated handle for gemm_new_secs measurement)",
    );
    let t_gemm_new = Instant::now();
    let gemm = CudaGemm::new(&device_for_gemm_new)
        .expect("CudaGemm::new must succeed given the manual compile/load pass above succeeded");
    let gemm_new_secs = t_gemm_new.elapsed().as_secs_f64();

    // (p4) 実 GEMM 実行（N=1024。ホスト→デバイス転送・カーネル起動・
    // 同期・デバイス→ホスト転送の合計。`run_tiled_f32` の契約どおり）。
    // 本番同様、直前に構築した `gemm`（`device_for_gemm_new` 由来）をそのまま
    // 使う。
    let (a, b) = gen_square_ab(0x926, P4_MATMUL_N);
    let n = P4_MATMUL_N as u32;
    let t_run = Instant::now();
    let out = gemm
        .run_tiled_f32(&a, &b, n, n, n)
        .expect("run_tiled_f32 must succeed for a well-formed N=1024 square GEMM");
    let run_tiled_f32_secs = t_run.elapsed().as_secs_f64();
    assert_eq!(
        out.len(),
        P4_MATMUL_N * P4_MATMUL_N,
        "run_tiled_f32 output length must equal N*N for a square GEMM"
    );

    TrialSample {
        device_new_secs,
        compile_secs,
        load_secs,
        gemm_new_secs,
        run_tiled_f32_secs,
    }
}

fn median_of(samples: &[f64]) -> Quartiles {
    median_q1_q3(samples)
        .expect("samples collected from successful trials must be non-empty and NaN-free")
}

fn print_quartiles_ms(label: &str, q: Quartiles) {
    println!(
        "  {label}: median={:.3} ms  q1={:.3} ms  q3={:.3} ms",
        q.median * 1e3,
        q.q1 * 1e3,
        q.q3 * 1e3
    );
}

/// 受け入れ条件 1（内訳の定量記録）・2（支配的要因の特定）の本体。
///
/// (p1) デバイス生成・(p2) NVRTC コンパイル ×8（カーネル別）・(p3) driver
/// ロード ×8（カーネル別）・本番呼び出し列での (p2)+(p3) 合計・(p4) 実
/// GEMM 実行を、ウォームアップ後 [`MEASURED_TRIALS`] 回計測し中央値・
/// Q1/Q3 を標準出力へ記録する。転記先は `docs/perf/
/// cuda-tape-init-cost-diagnosis.md`（イシュー #926）。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上。DGX Spark GB10 想定）必須。#926"]
fn init_cost_diag_phase_breakdown() {
    for _ in 0..WARMUP_TRIALS {
        let _ = measure_one_trial();
    }

    let mut device_new = Vec::with_capacity(MEASURED_TRIALS);
    let mut compile_by_kernel: [Vec<f64>; 8] = Default::default();
    let mut load_by_kernel: [Vec<f64>; 8] = Default::default();
    let mut gemm_new = Vec::with_capacity(MEASURED_TRIALS);
    let mut run_tiled_f32 = Vec::with_capacity(MEASURED_TRIALS);
    // (p1)+(p2 合計)+(p3 合計)+(p4) をトライアル単位で対にして合算した
    // サンプル列（`jit_cache_bench_tests.rs` の `cold_total_samples` と
    // 同じ理由: 中央値は線形演算ではないため、区間別中央値の和ではなく
    // トライアル単位の和にまず `median_q1_q3` を適用する必要がある）。
    let mut reconstructed_total = Vec::with_capacity(MEASURED_TRIALS);

    for _ in 0..MEASURED_TRIALS {
        let sample = measure_one_trial();
        device_new.push(sample.device_new_secs);
        for i in 0..8 {
            compile_by_kernel[i].push(sample.compile_secs[i]);
            load_by_kernel[i].push(sample.load_secs[i]);
        }
        gemm_new.push(sample.gemm_new_secs);
        run_tiled_f32.push(sample.run_tiled_f32_secs);
        reconstructed_total.push(
            sample.device_new_secs
                + sample.compile_secs.iter().sum::<f64>()
                + sample.load_secs.iter().sum::<f64>()
                + sample.run_tiled_f32_secs,
        );
    }

    println!("=== CUDA tape 初期化コスト フェーズ分解（イシュー #926） ===");
    print_quartiles_ms("(p1) CudaDevice::new", median_of(&device_new));

    for (i, (label, _, _)) in kernel_specs().iter().enumerate() {
        print_quartiles_ms(
            &format!("(p2) compile_ptx[{label}]"),
            median_of(&compile_by_kernel[i]),
        );
        print_quartiles_ms(
            &format!("(p3) load_module+load_function[{label}]"),
            median_of(&load_by_kernel[i]),
        );
    }

    print_quartiles_ms(
        "(p2+p3 本番同一呼び出し) CudaGemm::new",
        median_of(&gemm_new),
    );
    print_quartiles_ms("(p4) run_tiled_f32(N=1024)", median_of(&run_tiled_f32));
    print_quartiles_ms(
        "(p1+p2+p3+p4 再構成合計。診断用の個別計測経路)",
        median_of(&reconstructed_total),
    );
}

/// 受け入れ条件 1・2 の傍証（エンドツーエンド整合確認）。
///
/// `scripts/bench/framework-compare/bench-fandhe` と同じ本番 API 経路
/// （[`CudaBackendOps::gemm`]。`ops.rs` 参照）を N=256/1024/4096 で毎回
/// フレッシュハンドル計測し、(a) フレームワーク横並びベンチが観測した
/// 「N にほぼ非依存の 440〜460 ms 帯」がこの diagnostic 計測でも再現
/// すること（値の記録のみ。本ファイル冒頭「gating しない方針」参照）・
/// (b) [`init_cost_diag_phase_breakdown`] のフェーズ合計と同オーダーに
/// なることを確認する。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上。DGX Spark GB10 想定）必須。#926"]
fn init_cost_diag_e2e_matches_framework_compare_shape() {
    const SIZES: [usize; 3] = [256, 1024, 4096];
    let backend = CudaBackendOps::new(0);

    for _ in 0..WARMUP_TRIALS {
        let (a, b) = gen_square_ab(0x926_e2e, SIZES[0]);
        let n = SIZES[0];
        let a_t = Tensor::new(a, &[n, n]).expect("warmup lhs tensor construction must succeed");
        let b_t = Tensor::new(b, &[n, n]).expect("warmup rhs tensor construction must succeed");
        let _ = backend
            .gemm(&a_t, &b_t)
            .expect("warmup CudaBackendOps::gemm call must succeed");
    }

    println!("=== CUDA tape 初期化コスト e2e（本番 CudaBackendOps::gemm 経路）===");
    for n in SIZES {
        let mut samples = Vec::with_capacity(MEASURED_TRIALS);
        for _ in 0..MEASURED_TRIALS {
            let (a, b) = gen_square_ab(0x926_e2e ^ (n as u64), n);
            let a_t = Tensor::new(a, &[n, n])
                .unwrap_or_else(|e| panic!("lhs tensor construction must succeed for N={n}: {e}"));
            let b_t = Tensor::new(b, &[n, n])
                .unwrap_or_else(|e| panic!("rhs tensor construction must succeed for N={n}: {e}"));

            let t = Instant::now();
            let out = backend.gemm(&a_t, &b_t).unwrap_or_else(|e| {
                panic!("fresh-handle CudaBackendOps::gemm must succeed for N={n}: {e}")
            });
            samples.push(t.elapsed().as_secs_f64());

            assert_eq!(
                out.shape(),
                &[n, n],
                "gemm output shape must be [N, N] for a square GEMM"
            );
        }
        print_quartiles_ms(&format!("N={n}"), median_of(&samples));
    }
}

/// 受け入れ条件 3（Phase 2 対応方針への示唆）の検算に使う参照値。
///
/// 同一 `CudaGemm` ハンドルを 1 度だけ構築し、以降 [`MEASURED_TRIALS`] 回
/// `run_tiled_f32`（N=1024）を反復した場合の 1 回あたり時間を計測する。
/// これは「初期化を除いた下限」の参照値であり、[`init_cost_diag_phase_
/// breakdown`] の (p4) 単独計測とほぼ一致するはずである（(p4) は初回
/// 呼び出しのみを測るため、両者の差はカーネルキャッシュ・ドライバ内部
/// 状態のウォームアップ差に限られる）。#928（対 candle/Burn ベースライン
/// 比較）はこの参照値をそのまま使わず独自に再計測する（本テストは内訳
/// 帰属の検算専用。本ファイル冒頭コメント参照）。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上。DGX Spark GB10 想定）必須。#926"]
fn init_cost_diag_reused_handle_steady_state_reference() {
    let device = CudaDevice::new(0)
        .expect("CUDA device must be available on the ignored diagnostic bench runner");
    let gemm = CudaGemm::new(&device)
        .expect("CudaGemm::new must succeed on the ignored diagnostic bench runner");
    let n = P4_MATMUL_N as u32;
    let (a, b) = gen_square_ab(0x0926_1eed_u64, P4_MATMUL_N);

    for _ in 0..WARMUP_TRIALS {
        let _ = gemm
            .run_tiled_f32(&a, &b, n, n, n)
            .expect("warmup run_tiled_f32 call must succeed");
    }

    let mut samples = Vec::with_capacity(MEASURED_TRIALS);
    for _ in 0..MEASURED_TRIALS {
        let t = Instant::now();
        let out = gemm
            .run_tiled_f32(&a, &b, n, n, n)
            .expect("reused-handle run_tiled_f32 call must succeed");
        samples.push(t.elapsed().as_secs_f64());
        assert_eq!(
            out.len(),
            P4_MATMUL_N * P4_MATMUL_N,
            "run_tiled_f32 output length must equal N*N for a square GEMM"
        );
    }

    println!("=== CUDA GEMM 再利用ハンドルの定常状態 1 回あたりコスト（参照値）===");
    print_quartiles_ms(
        "reused-handle run_tiled_f32(N=1024) per-call",
        median_of(&samples),
    );
}
