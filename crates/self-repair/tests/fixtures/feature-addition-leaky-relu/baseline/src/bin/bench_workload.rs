//! 候補 diff 直接実測（TASK-3.2a・イシュー #137）向けの決定的ベンチワークロード。
//!
//! `crate::verify_bench_direct::DirectBenchRunner` が baseline commit・候補
//! 適用済み作業木の双方でこの bin を `cargo build --release --bin
//! bench_workload` し、生成物を「1 回 exec するだけ」で外部タイミング計測する
//! （`crates/self-repair/src/verify_bench_direct.rs` モジュール冒頭ドキュメント
//! 参照）。**このファイル自体は
//! `crate::verify_bench_direct::DirectBenchSpec::workload_sources` によって
//! ピン留めされ、候補 diff が改変すると計測前に fail-closed で拒否される**
//! （ゲーミング防止。同モジュール §「ゲーミング防止」）。
//!
//! # ワークロード内容
//! `activations::relu`／`activations::sigmoid`（本 fixture crate の baseline
//! 実装）を forward + backward で反復する。決定的シード（固定入力・固定式。
//! `guardrail::determinism::DEFAULT_SEED = 42` を根拠値として参照——本 crate は
//! `[workspace]` テーブルで分離されており `guardrail` に依存しないため、
//! 値のみをここに再掲する。実装計画 #137 §3.4）。
//!
//! # 作業量
//! 1 プロセス実行あたり最低 10ms 以上の作業量を確保する（ノイズ耐性。PR #341
//! `synthetic_relu_workload` と同方針。実測して反復数を固定値へ調整済み）。
//! `std::hint::black_box` で最適化による消去を防ぐ。

use autodiff::Tape;
use self_repair_feature_addition_leaky_relu_baseline::activations;
use tensor_core::Tensor;

/// 決定的シード（`guardrail::determinism::DEFAULT_SEED` と同一値。根拠は
/// モジュール冒頭ドキュメント参照）。線形合同法もどきの単純な決定的疑似乱数で
/// 固定入力テンソルを構成する（外部乱数クレートへの依存を増やさないため。
/// `rand` は許容依存区分内だが本 bin は最小依存で完結させる）。
const SEED: u64 = 42;

/// 1 回の forward+backward で使う入力の要素数。
const ELEMENTS: usize = 4096;

/// 決定的疑似乱数列（xorshift64）から `[-2.0, 2.0)` 相当の `f32` 系列を生成する。
fn deterministic_inputs(count: usize) -> Vec<f32> {
    let mut state = SEED ^ 0x9E37_79B9_7F4A_7C15;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        // xorshift64（決定的・依存追加なし）。
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // 上位ビットを [0, 1) の f64 へ写し、[-2.0, 2.0) へ線形変換する。
        let unit = (state >> 11) as f64 / (1u64 << 53) as f64;
        values.push((unit * 4.0 - 2.0) as f32);
    }
    values
}

/// relu・sigmoid それぞれを forward + backward し、勾配の総和を返す
/// （`black_box` の消去対象として使う）。
fn run_once(inputs: &[f32]) -> f32 {
    let tensor = Tensor::new(inputs.to_vec(), &[inputs.len()])
        .expect("bench_workload: shape とデータ長は一致させている");
    let tape = Tape::new(Box::new(backend_cpu::CpuBackendOps::new()));

    let x_relu = tape.var(&tensor);
    let y_relu = activations::relu(&x_relu);
    let tape_relu_grad = tape
        .backward(&y_relu)
        .expect("bench_workload: relu backward は常に成功する");
    let relu_grad_sum: f32 = tape_relu_grad
        .get(&x_relu)
        .expect("bench_workload: x_relu は同一 tape 上のノード")
        .map(|g| {
            g.contiguous()
                .as_slice()
                .map(|s| s.iter().sum())
                .unwrap_or(0.0)
        })
        .unwrap_or(0.0);

    let x_sigmoid = tape.var(&tensor);
    let y_sigmoid = activations::sigmoid(&x_sigmoid);
    let tape_sigmoid_grad = tape
        .backward(&y_sigmoid)
        .expect("bench_workload: sigmoid backward は常に成功する");
    let sigmoid_grad_sum: f32 = tape_sigmoid_grad
        .get(&x_sigmoid)
        .expect("bench_workload: x_sigmoid は同一 tape 上のノード")
        .map(|g| {
            g.contiguous()
                .as_slice()
                .map(|s| s.iter().sum())
                .unwrap_or(0.0)
        })
        .unwrap_or(0.0);

    relu_grad_sum + sigmoid_grad_sum
}

fn main() {
    let inputs = deterministic_inputs(ELEMENTS);
    // 反復回数は 1 プロセス実行あたり 10ms 以上の作業量を確保するため
    // 実測調整した固定値（ノイズ耐性。モジュール冒頭ドキュメント参照）。
    let mut acc = 0.0f32;
    for _ in 0..4000u32 {
        acc = std::hint::black_box(acc + run_once(std::hint::black_box(&inputs)));
    }
    std::hint::black_box(acc);
}
