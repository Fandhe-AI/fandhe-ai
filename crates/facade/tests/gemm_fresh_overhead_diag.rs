//! `scripts/bench/framework-compare/` の fresh／reuse GEMM プロトコル差分
//! を facade 公開 API で再現し、N=2048 のみに現れる約 166 ms の再現性
//! ある固定コスト（イシュー #956）の帰属をトグル変種で切り分ける実機
//! 専用診断ベンチ。
//!
//! # なぜ facade（本クレート）に置くか
//!
//! イシュー #956 が観測した固定コストは `tape_for` からの一連の結線
//! （composition root）+ `tape.var` + `matmul` + ホスト実体化 +
//! `Tape` drop という facade 公開 API 経由の区間全体で発生する
//! （`bench-fandhe/src/main.rs::run_gemm` がまさにこの経路）。
//! `crates/backend-cuda/src/fresh_overhead_diag_tests.rs`（Phase 1-A）は
//! `CudaGemm`／`context_cache` を直接使う下位レイヤのフェーズ分解を担い、
//! 本ファイルは「facade 経路のどの構成要素を変えると固定コストが消える
//! か」を切り分ける上位レイヤの診断を担う（`tape_cuda_cache_bench.rs`
//! と同型の役割分担）。
//!
//! # 帰属の明確化（実装計画 §2.1・§2.3。詳細は `fresh_overhead_diag_
//! tests.rs` 冒頭コメントを参照。ここでは facade 経路固有の要点のみ）
//!
//! `bench-fandhe/src/main.rs::run_gemm`（fresh）は 1 イテレーションごとに
//! 新規 `tape_for` → `tape.var`×2 → `matmul` → `to_tensor().contiguous()
//! .as_slice().to_vec()`（チェックサム。16 MiB 相当のホストコピーを
//! 追加で発生させる）→ `Tape` drop、という順で計測区間を構成する。
//! `Tape` drop 時、matmul の結果ノード（`TapeNode.value: OnceCell
//! <Tensor<f32>>`。`crates/autodiff/src/tape.rs`）が保持する C の
//! `Tensor<f32>`（N=2048 で 16 MiB の `Arc<Storage>`）はここで解放される
//! （`run_gemm_reuse` では `Tape` を使い回すため蓄積され解放されない）。
//!
//! # 変種（P0〜P4。実装計画 §4.1-B）
//!
//! - **P0 (fresh)**: `bench-fandhe::run_gemm` と同一の呼び出し列
//! - **P1 (reuse)**: `bench-fandhe::run_gemm_reuse` と同一の呼び出し列
//!   （P0 との checksum 一致を確認する唯一の hard assert）
//! - **P2 (fresh + keep C alive)**: fresh だが `to_tensor()` の
//!   `Tensor<f32>`（`Arc<Storage>` の安価な clone）をループ外の `Vec` に
//!   保持し、`Tape` drop によるホストバッファ解放を実質的に抑止する
//!   （H3 判別: 解放コスト自体が支配的なら P2 は P0 より速いはず）
//! - **P3 (fresh + tape drop 計測区間外)**: fresh だが `Tape` の drop を
//!   `Instant::elapsed()` 呼び出しの後（別の `Instant` で）行い、
//!   「本体（matmul + 実体化）」と「drop」のコストを分離する
//! - **P4 (fresh, checksum を as_slice 直読み)**: fresh だが checksum の
//!   `.to_vec()`（16 MiB の追加ホストコピー）を省き `as_slice()` を直接
//!   集計し、`to_vec()` 自体の寄与を分離する
//!
//! # 実行時は必ず `--test-threads=1`（`tape_cuda_cache_bench.rs` と同じ
//! 理由: 同一 GPU 上での複数テストスレッド競合を避ける）
//!
//! # gating しない方針
//!
//! タイミング値は `println!` のみで record only（`tape_cuda_cache_bench.rs`
//! と同じ方針）。唯一の hard assert は P0/P1 の checksum 一致（数値
//! 正しさの契約。`.claude/rules/coding-rust.md`「バックエンド間数値一致
//! テストの許容誤差を単独で緩和しない」とは別軸だが、同じ「正しさは
//! 緩めない」思想を踏襲する）。

use std::time::Instant;

use bench_harness::{median_q1_q3, rng::Xorshift64Star};
use fandhe_ai::{Device, Tensor, Var, tape_for};

/// `bench-fandhe/src/main.rs` の `WARMUP_ITERS`/`MEASURE_ITERS`
/// （`scripts/bench/framework-compare/bench-common/src/lib.rs`）と同じ
/// プロトコル定数。framework-compare 側は独立ワークスペース
/// （`.claude/rules/deps-policy.md` 第 9 区分）のため `bench-common` を
/// path 依存で取り込めず、値のみをここへ複製する（生成式は
/// `Xorshift64Star`〈`bench-harness`。本クレートの既存 dev-dependency〉を
/// 共有するため自前実装は不要）。
const WARMUP_ITERS: usize = 20;
const MEASURE_ITERS: usize = 20;

/// イシュー #956 本文が固定コストを観測した N=2048 を中心に、
/// サイズ非依存の (f1)(f2) 上界確認（N=1024）・非再現確認（N=4096）を
/// 含めた 3 点（`fresh_overhead_diag_tests.rs::SIZES` と同じ選定理由）。
const SIZES: [usize; 3] = [1024, 2048, 4096];

/// `bench-fandhe::gemm_inputs` と同じ決定的シード方式（シード値自体は
/// `bench-common::{SEED_A, SEED_B}` の値を継承しない独立採番。本ファイルの
/// 目的は framework-compare との厳密な数値再現ではなく、fresh/reuse の
/// 相対比較のため独立シードで問題ない）。
fn gemm_inputs(n: usize) -> (Tensor<f32>, Tensor<f32>) {
    let a = Xorshift64Star::new(0x0956_00a1).fill_vec(n * n);
    let b = Xorshift64Star::new(0x0956_00b1).fill_vec(n * n);
    (
        Tensor::new(a, &[n, n]).expect("A tensor construction must succeed"),
        Tensor::new(b, &[n, n]).expect("B tensor construction must succeed"),
    )
}

/// `bench-fandhe::checksum_var` と同一の定義（`to_tensor()` +
/// `contiguous().as_slice().to_vec()` の合計。P0/P1/P2/P3 が使う）。
fn checksum_var(v: &Var<'_>) -> f64 {
    let t = v.to_tensor();
    let slice = t
        .contiguous()
        .as_slice()
        .expect("as_slice() must return Some after contiguous()")
        .to_vec();
    slice.iter().map(|&x| x as f64).sum()
}

/// P4 専用: `to_vec()`（16 MiB 相当の追加ホストコピー）を省き、
/// `as_slice()` の借用のまま直接集計する。
fn checksum_var_no_to_vec(v: &Var<'_>) -> f64 {
    let t = v.to_tensor();
    let contiguous = t.contiguous();
    let slice = contiguous
        .as_slice()
        .expect("as_slice() must return Some after contiguous()");
    slice.iter().map(|&x| x as f64).sum()
}

fn make_cuda_tape() -> fandhe_ai::Tape {
    tape_for(Device::Cuda(0))
        .expect("CUDA driver 搭載環境（本テストは #[ignore] 実機専用）では成功するはず")
}

/// P0 (fresh): `bench-fandhe::run_gemm` の 1 イテレーションと同一。
fn measure_p0_fresh(a: &Tensor<f32>, b: &Tensor<f32>) -> (f64, f64) {
    let start = Instant::now();
    let tape = make_cuda_tape();
    let av = tape.var(a);
    let bv = tape.var(b);
    let c = av
        .matmul(&bv)
        .expect("matmul must succeed for a well-formed square GEMM");
    let checksum = checksum_var(&c);
    let elapsed = start.elapsed().as_secs_f64();
    drop(tape);
    (elapsed, checksum)
}

/// P2 (fresh + keep C alive): `Tape` drop 前に C の `Tensor<f32>`
/// （`Arc<Storage>` clone）を `keep_alive` へ退避し、`Tape` drop による
/// ホストバッファ解放を実質的に抑止する。
fn measure_p2_keep_alive(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    keep_alive: &mut Vec<Tensor<f32>>,
) -> (f64, f64) {
    let start = Instant::now();
    let tape = make_cuda_tape();
    let av = tape.var(a);
    let bv = tape.var(b);
    let c = av
        .matmul(&bv)
        .expect("matmul must succeed for a well-formed square GEMM");
    let c_tensor = c.to_tensor();
    let checksum = c_tensor
        .contiguous()
        .as_slice()
        .expect("as_slice() must return Some after contiguous()")
        .iter()
        .map(|&x| x as f64)
        .sum();
    keep_alive.push(c_tensor);
    let elapsed = start.elapsed().as_secs_f64();
    drop(tape);
    (elapsed, checksum)
}

/// P3 (fresh + tape drop 計測区間外): 本体（matmul + 実体化）の所要時間と
/// `Tape` drop の所要時間を別々の `Instant` で計測する。
fn measure_p3_drop_outside(a: &Tensor<f32>, b: &Tensor<f32>) -> (f64, f64, f64) {
    let start = Instant::now();
    let tape = make_cuda_tape();
    let av = tape.var(a);
    let bv = tape.var(b);
    let c = av
        .matmul(&bv)
        .expect("matmul must succeed for a well-formed square GEMM");
    let checksum = checksum_var(&c);
    let body_elapsed = start.elapsed().as_secs_f64();

    let drop_start = Instant::now();
    drop(tape);
    let drop_elapsed = drop_start.elapsed().as_secs_f64();

    (body_elapsed, drop_elapsed, checksum)
}

/// P4 (fresh, checksum を as_slice 直読み): `to_vec()` を省いた checksum
/// 計算のコストを分離する。
fn measure_p4_no_to_vec(a: &Tensor<f32>, b: &Tensor<f32>) -> (f64, f64) {
    let start = Instant::now();
    let tape = make_cuda_tape();
    let av = tape.var(a);
    let bv = tape.var(b);
    let c = av
        .matmul(&bv)
        .expect("matmul must succeed for a well-formed square GEMM");
    let checksum = checksum_var_no_to_vec(&c);
    let elapsed = start.elapsed().as_secs_f64();
    drop(tape);
    (elapsed, checksum)
}

/// P1 (reuse): `bench-fandhe::run_gemm_reuse` と同一の呼び出し列。
/// tape・葉 Var（A・B）を 1 回だけ構築し、以降 matmul のみを繰り返す。
fn run_p1_reuse(n: usize) -> (bench_harness::Quartiles, f64) {
    let (a_data, b_data) = gemm_inputs(n);
    let tape = make_cuda_tape();
    let a = tape.var(&a_data);
    let b = tape.var(&b_data);

    let one = || -> f64 {
        let start = Instant::now();
        let c = a
            .matmul(&b)
            .expect("matmul must succeed for a well-formed square GEMM");
        let _ = checksum_var(&c);
        start.elapsed().as_secs_f64()
    };

    for _ in 0..WARMUP_ITERS {
        one();
    }
    let mut samples = Vec::with_capacity(MEASURE_ITERS);
    for _ in 0..MEASURE_ITERS {
        samples.push(one());
    }
    // 最終 checksum を P0 との整合確認に使う（同一入力 A・B・matmul の
    // 数値結果は fresh/reuse で一致するはず）。
    let c = a
        .matmul(&b)
        .expect("matmul must succeed for a well-formed square GEMM");
    let last_checksum = checksum_var(&c);

    (
        median_q1_q3(&samples).expect("MEASURE_ITERS 個の non-NaN サンプルは quartiles を持つはず"),
        last_checksum,
    )
}

fn print_quartiles_ms(label: &str, q: bench_harness::Quartiles) {
    println!(
        "  {label}: median={:.3} ms  q1={:.3} ms  q3={:.3} ms",
        q.median * 1e3,
        q.q1 * 1e3,
        q.q3 * 1e3
    );
}

/// 受け入れ条件 1・2 本体（facade 経路トグル比較）。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上。DGX Spark GB10 想定）必須。#956"]
fn gemm_fresh_overhead_diag_toggle_comparison() {
    for &n in &SIZES {
        println!("=== facade GEMM fresh/reuse トグル比較: N={n} (イシュー #956) ===");
        let (a_data, b_data) = gemm_inputs(n);

        // --- P0 (fresh) ---
        for _ in 0..WARMUP_ITERS {
            let _ = measure_p0_fresh(&a_data, &b_data);
        }
        let mut p0_samples = Vec::with_capacity(MEASURE_ITERS);
        let mut p0_checksum = 0.0;
        for _ in 0..MEASURE_ITERS {
            let (secs, checksum) = measure_p0_fresh(&a_data, &b_data);
            p0_samples.push(secs);
            p0_checksum = checksum;
        }
        print_quartiles_ms(
            "P0 (fresh)",
            median_q1_q3(&p0_samples).expect("P0 samples must yield quartiles"),
        );

        // --- P1 (reuse) ---
        let (p1_q, p1_checksum) = run_p1_reuse(n);
        print_quartiles_ms("P1 (reuse)", p1_q);

        // 唯一の hard assert: fresh/reuse で同一入力・同一演算の数値結果が
        // 一致すること（`.claude/rules/coding-rust.md`「バックエンド間
        // 数値一致…を単独で緩和しない」と同じ「正しさは緩めない」方針を
        // fresh/reuse の切り替え軸にも適用する）。浮動小数点の和の順序
        // 依存性を許容するため、統一複合判定と同じ緩やかな許容誤差
        // （相対 1e-3 または絶対 1e-5。ただし checksum は N^2 要素の総和
        // のためスケールが大きく、ここでは相対誤差のみで判定する）を使う。
        let rel_diff = (p0_checksum - p1_checksum).abs() / p1_checksum.abs().max(1.0);
        assert!(
            rel_diff < 1e-3,
            "N={n}: fresh checksum ({p0_checksum}) と reuse checksum ({p1_checksum}) は \
             同一入力・同一演算のため一致するはず（相対誤差 {rel_diff} が 1e-3 以上）"
        );

        // --- P2 (fresh + keep C alive) ---
        let mut keep_alive: Vec<Tensor<f32>> = Vec::new();
        for _ in 0..WARMUP_ITERS {
            let _ = measure_p2_keep_alive(&a_data, &b_data, &mut keep_alive);
        }
        keep_alive.clear();
        let mut p2_samples = Vec::with_capacity(MEASURE_ITERS);
        for _ in 0..MEASURE_ITERS {
            let (secs, _checksum) = measure_p2_keep_alive(&a_data, &b_data, &mut keep_alive);
            p2_samples.push(secs);
        }
        print_quartiles_ms(
            "P2 (fresh + keep C alive)",
            median_q1_q3(&p2_samples).expect("P2 samples must yield quartiles"),
        );
        // MEASURE_ITERS 回分の C（N=4096 でも 64 MiB × 20 ≒ 1.25 GiB）を
        // 保持したまま次サイズへ進まないよう明示的に解放する
        // （`.claude/rules/security.md`「資源枯渇」節）。
        drop(keep_alive);

        // --- P3 (fresh + tape drop 計測区間外) ---
        for _ in 0..WARMUP_ITERS {
            let _ = measure_p3_drop_outside(&a_data, &b_data);
        }
        let mut p3_body = Vec::with_capacity(MEASURE_ITERS);
        let mut p3_drop = Vec::with_capacity(MEASURE_ITERS);
        for _ in 0..MEASURE_ITERS {
            let (body, drop_secs, _checksum) = measure_p3_drop_outside(&a_data, &b_data);
            p3_body.push(body);
            p3_drop.push(drop_secs);
        }
        print_quartiles_ms(
            "P3 body (fresh, tape drop 区間外)",
            median_q1_q3(&p3_body).expect("P3 body samples must yield quartiles"),
        );
        print_quartiles_ms(
            "P3 tape drop (単独)",
            median_q1_q3(&p3_drop).expect("P3 drop samples must yield quartiles"),
        );

        // --- P4 (fresh, checksum を as_slice 直読み) ---
        for _ in 0..WARMUP_ITERS {
            let _ = measure_p4_no_to_vec(&a_data, &b_data);
        }
        let mut p4_samples = Vec::with_capacity(MEASURE_ITERS);
        for _ in 0..MEASURE_ITERS {
            let (secs, _checksum) = measure_p4_no_to_vec(&a_data, &b_data);
            p4_samples.push(secs);
        }
        print_quartiles_ms(
            "P4 (fresh, checksum as_slice 直読み)",
            median_q1_q3(&p4_samples).expect("P4 samples must yield quartiles"),
        );
    }
}
