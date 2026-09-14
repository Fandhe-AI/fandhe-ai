//! イシュー #1707/#1708/#1709: `BackendOps::scalar_unary`／
//! `scalar_binary`（`Sqrt`／`Sub`／`Div`／`Pow`・`Neg`／`Abs`／`Log`／
//! `Log2`／`Log10`／`Sin`／`Cos`／`Tan`・比較演算 6 種〈`Gt`／`Ge`／
//! `Lt`／`Le`／`Eq`／`Ne`〉・`Clamp`）の CPU-Metal 数値一致検証（CUDA 側
//! `backend-cuda::tests::scalar_op_parity`〈#1700/#1701/#1702〉の Metal
//! 対応版）。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`where_masked_fill_parity.rs`〈#1637〉と同方針。`#![cfg(target_os =
//! "macos")]` により Linux CI ではコンパイル対象外になり、`#[ignore]`
//! により通常の `cargo test` からも除外される。Metal は Linux で
//! `MetalBackendOps` が存在しないため CUDA 側のような環境適応スモーク
//! テストは持てない）。
//!
//! 判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の
//! 参照とする（`.claude/rules/coding-rust.md`）。`Sqrt`／`Sub`／`Div`・
//! `Neg`／`Abs` は `metal::precise::sqrt`／`+ - * /`／符号ビット演算が
//! correctly rounded（または算術を含まない選択）であることにより IEEE
//! 754 丸めとなりホスト `f32` 演算と bit 同一になる想定（`crate::
//! scalar_op_source` モジュール doc「コンパイルオプションと数値契約」
//! 参照）のため、`assert_parity`（REQ-2 複合判定）に加え bit 同一
//! （より強い検証。`NaN` はクラス一致）も併記する。`Pow`・超越関数系
//! （`Log`／`Log2`／`Log10`／`Sin`／`Cos`／`Tan`。`metal::precise::` の
//! ulp 誤差がホスト側 libm と一致する保証がない）は `assert_parity`
//! のみで検証する（既存 `exp`／`tanh` と同じ扱い）。比較演算 6 種・
//! `Clamp` は算術を含まない純粋な選択・比較演算のため bit 同一
//! （`assert_eq!`）で検証する（イシュー #1709。CUDA 側
//! `kernels_scalar_op.rs::unary_expr` の `Clamp` 分岐・`binary_expr` の
//! 比較演算コメント参照）。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test scalar_op_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, ScalarBinaryOp, ScalarUnaryOp, Tensor};

/// `Sqrt`／`Log`／`Log2`／`Log10` 用に定義域外（負値・0 近傍）を避けた
/// 正の乱数（`[0.1, 2.1)`）を生成する（CUDA 側 `backend-cuda/tests/
/// scalar_op_parity.rs::positive_data` と同じオフセット方針）。
fn positive_data(rng: &mut Xorshift64Star, len: usize) -> Vec<f32> {
    rng.fill_vec(len)
        .into_iter()
        .map(|v| v.abs() + 0.1)
        .collect()
}

/// `Neg`／`Abs`／`Sin`／`Cos`／`Tan` 用の符号あり乱数（`[-1, 1)`）。
/// 正の値だけでは `Abs` が恒等になり検証にならず、`Tan` は
/// `positive_data` の範囲（`[0.1, 2.1)`）が `π/2` をまたぐため検証意図
/// が不明瞭になる（`[-1, 1)` は subnormal 非到達域でもあり `Neg`／`Abs`
/// の bit 同一検証における flush-to-zero 差異も避ける。`crate::
/// scalar_op_source` モジュール doc「コンパイルオプションと数値契約」
/// 参照）。
fn signed_data(rng: &mut Xorshift64Star, len: usize) -> Vec<f32> {
    rng.fill_vec(len)
}

/// [`assert_unary_parity`] が使う入力生成器の選択（kind の定義域に応じ
/// 呼び出し元が指定する）。
#[derive(Clone, Copy)]
enum Domain {
    Positive,
    Signed,
}

fn gen_data(domain: Domain, rng: &mut Xorshift64Star, len: usize) -> Vec<f32> {
    match domain {
        Domain::Positive => positive_data(rng, len),
        Domain::Signed => signed_data(rng, len),
    }
}

fn assert_unary_parity(
    op: ScalarUnaryOp,
    domain: Domain,
    seed: u64,
    shape: &[usize],
    bit_exact: bool,
) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a = Tensor::new(
        gen_data(domain, &mut Xorshift64Star::new(seed), numel),
        shape,
    )
    .expect("valid tensor");

    let cpu_result = cpu
        .scalar_unary(op, &a)
        .expect("cpu scalar_unary always succeeds for implemented kinds");
    let metal_result = metal
        .scalar_unary(op, &a)
        .expect("metal scalar_unary must succeed on Metal-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_unary({op:?}) cpu-metal parity shape={shape:?}"),
        metal_slice,
        cpu_slice,
    );
    if bit_exact {
        assert_eq!(
            metal_slice, cpu_slice,
            "scalar_unary({op:?}): IEEE 754 丸め契約により bit 同一のはず（shape={shape:?}）"
        );
    }
}

fn assert_binary_parity(
    op: ScalarBinaryOp,
    seed_a: u64,
    seed_b: u64,
    shape: &[usize],
    bit_exact: bool,
) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a = Tensor::new(
        positive_data(&mut Xorshift64Star::new(seed_a), numel),
        shape,
    )
    .expect("valid tensor");
    let b = Tensor::new(
        positive_data(&mut Xorshift64Star::new(seed_b), numel),
        shape,
    )
    .expect("valid tensor");

    let cpu_result = cpu
        .scalar_binary(op, &a, &b)
        .expect("cpu scalar_binary always succeeds for implemented kinds");
    let metal_result = metal
        .scalar_binary(op, &a, &b)
        .expect("metal scalar_binary must succeed on Metal-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_binary({op:?}) cpu-metal parity shape={shape:?}"),
        metal_slice,
        cpu_slice,
    );
    if bit_exact {
        assert_eq!(
            metal_slice, cpu_slice,
            "scalar_binary({op:?}): IEEE 754 丸め契約により bit 同一のはず（shape={shape:?}）"
        );
    }
}

/// broadcast 形状（`[m, n]` + `[n]`）での `scalar_binary` parity。
fn assert_binary_broadcast_parity(op: ScalarBinaryOp, seed_a: u64, seed_b: u64) {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let a = Tensor::new(positive_data(&mut Xorshift64Star::new(seed_a), 12), &[3, 4])
        .expect("valid tensor");
    let b = Tensor::new(positive_data(&mut Xorshift64Star::new(seed_b), 4), &[4])
        .expect("valid tensor");

    let cpu_result = cpu.scalar_binary(op, &a, &b).expect("cpu succeeds");
    let metal_result = metal.scalar_binary(op, &a, &b).expect("metal succeeds");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_binary({op:?}) broadcast cpu-metal parity"),
        metal_slice,
        cpu_slice,
    );
}

/// 比較演算（`Gt`／`Ge`／`Lt`／`Le`／`Eq`／`Ne`）用の `(a, b)` ペアを
/// 生成する。独立乱数 2 本では `Eq` が全 `0.0`・`Ne` が全 `1.0` になり
/// 検証にならないため、`a` から意図的に等値・やや大きい値・やや小さい
/// 値を混在させて派生させる（イシュー #1709。CUDA 側
/// `scalar_op_parity.rs::comparison_pair_data` と同型）。
fn comparison_pair_data(seed: u64, len: usize) -> (Vec<f32>, Vec<f32>) {
    let a = positive_data(&mut Xorshift64Star::new(seed), len);
    let b = a
        .iter()
        .enumerate()
        .map(|(i, &v)| match i % 3 {
            0 => v,        // 等値ケース
            1 => v + 0.25, // a < b
            _ => v - 0.25, // a > b
        })
        .collect();
    (a, b)
}

/// 比較演算 [`ScalarBinaryOp`]（`Gt`／`Ge`／`Lt`／`Le`／`Eq`／`Ne`）の
/// CPU-Metal parity を bit 同一（0.0/1.0 リテラルの純粋な比較演算の
/// ため。モジュール doc参照）で検証する。
fn assert_comparison_parity(op: ScalarBinaryOp, seed: u64, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    let (a_data, b_data) = comparison_pair_data(seed, numel);
    let a = Tensor::new(a_data, shape).expect("valid tensor");
    let b = Tensor::new(b_data, shape).expect("valid tensor");

    let cpu_result = cpu
        .scalar_binary(op, &a, &b)
        .expect("cpu scalar_binary always succeeds for implemented kinds");
    let metal_result = metal
        .scalar_binary(op, &a, &b)
        .expect("metal scalar_binary must succeed on Metal-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_binary({op:?}) cpu-metal parity shape={shape:?}"),
        metal_slice,
        cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "scalar_binary({op:?}): 比較演算は 0.0/1.0 の純粋な比較のため bit 同一のはず（shape={shape:?}）"
    );
}

/// [`ScalarUnaryOp::Clamp`] の CPU-Metal parity を bit 同一（算術を含ま
/// ない純粋な選択演算のため）で検証する。
fn assert_clamp_parity(min: f32, max: f32, seed: u64, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let op = ScalarUnaryOp::Clamp { min, max };

    let a = Tensor::new(positive_data(&mut Xorshift64Star::new(seed), numel), shape)
        .expect("valid tensor");

    let cpu_result = cpu
        .scalar_unary(op, &a)
        .expect("cpu scalar_unary always succeeds for implemented kinds");
    let metal_result = metal
        .scalar_unary(op, &a)
        .expect("metal scalar_unary must succeed on Metal-equipped test runner");

    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!("scalar_unary(Clamp{{min={min},max={max}}}) cpu-metal parity shape={shape:?}"),
        metal_slice,
        cpu_slice,
    );
    assert_eq!(
        metal_slice, cpu_slice,
        "scalar_unary(Clamp{{min={min},max={max}}}): 純粋な選択演算のため bit 同一のはず（shape={shape:?}）"
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体）。`Sqrt`（unary）・
/// `Sub`／`Div`（bit 同一）・`Pow`（REQ-2 のみ）に加え、超越関数系 8
/// kind（`Neg`／`Abs` は bit 同一、`Log`／`Log2`／`Log10`／`Sin`／
/// `Cos`／`Tan` は REQ-2 のみ）を、`PARALLEL_THRESHOLD`（CPU 側 rayon
/// 閾値）境界前後を含む形状で検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_op_matches_cpu_across_shapes() {
    let shapes: &[&[usize]] = &[
        &[1],
        &[4],
        &[3, 5],
        &[2, 3, 4],
        &[1 << 15], // PARALLEL_THRESHOLD ちょうど
        &[(1 << 15) - 1],
        &[(1 << 15) + 1],
    ];
    let mut seed = 11_000u64;
    for &shape in shapes {
        seed += 20;
        assert_unary_parity(ScalarUnaryOp::Sqrt, Domain::Positive, seed, shape, true);
        assert_binary_parity(ScalarBinaryOp::Sub, seed, seed + 1, shape, true);
        assert_binary_parity(ScalarBinaryOp::Div, seed, seed + 1, shape, true);
        assert_binary_parity(ScalarBinaryOp::Pow, seed, seed + 1, shape, false);

        assert_unary_parity(ScalarUnaryOp::Neg, Domain::Signed, seed + 2, shape, true);
        assert_unary_parity(ScalarUnaryOp::Abs, Domain::Signed, seed + 3, shape, true);
        assert_unary_parity(ScalarUnaryOp::Log, Domain::Positive, seed + 4, shape, false);
        assert_unary_parity(
            ScalarUnaryOp::Log2,
            Domain::Positive,
            seed + 5,
            shape,
            false,
        );
        assert_unary_parity(
            ScalarUnaryOp::Log10,
            Domain::Positive,
            seed + 6,
            shape,
            false,
        );
        assert_unary_parity(ScalarUnaryOp::Sin, Domain::Signed, seed + 7, shape, false);
        assert_unary_parity(ScalarUnaryOp::Cos, Domain::Signed, seed + 8, shape, false);
        assert_unary_parity(ScalarUnaryOp::Tan, Domain::Signed, seed + 9, shape, false);

        // 比較演算 6 種・`Clamp`（イシュー #1709）。
        for op in [
            ScalarBinaryOp::Gt,
            ScalarBinaryOp::Ge,
            ScalarBinaryOp::Lt,
            ScalarBinaryOp::Le,
            ScalarBinaryOp::Eq,
            ScalarBinaryOp::Ne,
        ] {
            assert_comparison_parity(op, seed + 10, shape);
        }
        assert_clamp_parity(0.5, 1.5, seed + 11, shape);
    }
}

/// broadcast 形状（`[3, 4]` + `[4]`）の比較演算 parity（イシュー #1709）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_comparison_broadcast_matches_cpu() {
    assert_binary_broadcast_parity(ScalarBinaryOp::Gt, 13_001, 13_002);
    assert_binary_broadcast_parity(ScalarBinaryOp::Eq, 13_003, 13_004);
}

/// broadcast 形状（`[3, 4]` + `[4]`）の parity。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_binary_broadcast_matches_cpu() {
    assert_binary_broadcast_parity(ScalarBinaryOp::Sub, 12_001, 12_002);
    assert_binary_broadcast_parity(ScalarBinaryOp::Div, 12_003, 12_004);
    assert_binary_broadcast_parity(ScalarBinaryOp::Pow, 12_005, 12_006);
}

/// `0.0`／`inf`／`NaN` 混入時の通過（クラス一致で比較。`NaN` の payload
/// は処理系依存のため bit 同一ではなくクラス一致で検証する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_op_handles_zero_inf_nan() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    // Sqrt: 0 は 0・負値は NaN・+inf は +inf。
    let a = Tensor::new(vec![0.0, -1.0, f32::INFINITY, 4.0], &[4]).expect("valid tensor");
    let cpu_out = cpu
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("cpu succeeds");
    let metal_out = metal
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("metal succeeds");
    for (i, (&c, &g)) in cpu_out
        .as_slice()
        .unwrap()
        .iter()
        .zip(metal_out.as_slice().unwrap().iter())
        .enumerate()
    {
        if c.is_nan() {
            assert!(g.is_nan(), "index {i}: expected NaN, got {g}");
        } else {
            assert_eq!(c.to_bits(), g.to_bits(), "index {i}: {c} != {g}");
        }
    }

    // Div: 0/0 は NaN・x/0（x!=0）は ±inf。
    let x = Tensor::new(vec![0.0, 1.0, -1.0, 4.0], &[4]).expect("valid tensor");
    let y = Tensor::new(vec![0.0, 0.0, 0.0, 2.0], &[4]).expect("valid tensor");
    let cpu_out = cpu
        .scalar_binary(ScalarBinaryOp::Div, &x, &y)
        .expect("cpu succeeds");
    let metal_out = metal
        .scalar_binary(ScalarBinaryOp::Div, &x, &y)
        .expect("metal succeeds");
    for (i, (&c, &g)) in cpu_out
        .as_slice()
        .unwrap()
        .iter()
        .zip(metal_out.as_slice().unwrap().iter())
        .enumerate()
    {
        if c.is_nan() {
            assert!(g.is_nan(), "index {i}: expected NaN, got {g}");
        } else {
            assert_eq!(c.to_bits(), g.to_bits(), "index {i}: {c} != {g}");
        }
    }
}

/// `0^0 == 1`（IEEE 754 `powf` 規約）を CPU-Metal 両方で確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn pow_zero_pow_zero_is_one() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![0.0f32], &[1]).expect("valid tensor");
    let b = Tensor::new(vec![0.0f32], &[1]).expect("valid tensor");
    let cpu_out = cpu
        .scalar_binary(ScalarBinaryOp::Pow, &a, &b)
        .expect("cpu succeeds");
    let metal_out = metal
        .scalar_binary(ScalarBinaryOp::Pow, &a, &b)
        .expect("metal succeeds");
    assert_eq!(cpu_out.as_slice().unwrap()[0], 1.0);
    assert_eq!(metal_out.as_slice().unwrap()[0], 1.0);
}

/// 負の底に非整数指数を与えると `NaN` を返す（実数域での定義域外）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn pow_negative_base_noninteger_exponent_is_nan() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![-2.0f32], &[1]).expect("valid tensor");
    let b = Tensor::new(vec![0.5f32], &[1]).expect("valid tensor");
    let out = metal
        .scalar_binary(ScalarBinaryOp::Pow, &a, &b)
        .expect("metal succeeds");
    assert!(out.as_slice().unwrap()[0].is_nan());
}

/// 空 shape（`numel == 0`）は空の結果を返す。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_op_empty_shape_returns_empty() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(Vec::<f32>::new(), &[0]).expect("valid tensor");
    let out = metal
        .scalar_unary(ScalarUnaryOp::Sqrt, &a)
        .expect("metal succeeds on empty input");
    assert_eq!(out.as_slice().unwrap().len(), 0);

    let b = Tensor::new(Vec::<f32>::new(), &[0]).expect("valid tensor");
    let out = metal
        .scalar_binary(ScalarBinaryOp::Sub, &a, &b)
        .expect("metal succeeds on empty input");
    assert_eq!(out.as_slice().unwrap().len(), 0);

    // #1708 の対象 kind（`Log`）も空 shape で空を返すことを確認する
    // （テンプレート機構が kind 非依存であることの追加確認）。
    let out = metal
        .scalar_unary(ScalarUnaryOp::Log, &a)
        .expect("metal succeeds on empty input");
    assert_eq!(out.as_slice().unwrap().len(), 0);

    // #1709 の対象 kind（比較演算・`Clamp`）も空 shape で空を返すことを
    // 確認する。
    let out = metal
        .scalar_binary(ScalarBinaryOp::Gt, &a, &b)
        .expect("metal succeeds on empty input (Gt)");
    assert_eq!(out.as_slice().unwrap().len(), 0);
    let out = metal
        .scalar_unary(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 }, &a)
        .expect("metal succeeds on empty input (Clamp)");
    assert_eq!(out.as_slice().unwrap().len(), 0);
}

/// #1708 の対象 8 kind: 特殊値（0／-0／inf／-inf／NaN）の伝播確認。
/// `Neg`／`Abs` は bit 同一（NaN はクラス一致）、超越関数系は有限入力
/// のみ `assert_parity`（`Pow` の NaN 混入時フィルタと同様）で検証する
/// （CUDA 側 `scalar_op_parity.rs` の同名セクションと同型）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn transcendental_scalar_op_edge_cases() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    // Neg: `0.0`/`-0.0` の符号反転を bit 単位で確認する（`assert_eq!`
    // を f32 スライスへ直接使うと `+0.0 == -0.0` が真になり符号反転の
    // 回帰を検出できないため `to_bits()` を使う）。
    let a = Tensor::new(vec![0.0, -0.0, 1.5, f32::INFINITY], &[4]).expect("valid tensor");
    let cpu_result = cpu.scalar_unary(ScalarUnaryOp::Neg, &a).expect("succeeds");
    let metal_result = metal
        .scalar_unary(ScalarUnaryOp::Neg, &a)
        .expect("succeeds");
    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    for (i, (&c, &g)) in cpu_slice.iter().zip(metal_slice.iter()).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "Neg NaN propagation mismatch at index {i}");
        } else {
            assert_eq!(
                c.to_bits(),
                g.to_bits(),
                "Neg cpu/metal bit mismatch at index {i}"
            );
        }
    }
    assert_eq!(cpu_slice[0].to_bits(), (-0.0_f32).to_bits());
    assert_eq!(cpu_slice[1].to_bits(), (0.0_f32).to_bits());

    // Abs: `-0.0` は `+0.0` へ、NaN はクラス一致で確認する。
    let a = Tensor::new(vec![-0.0, -1.5, f32::INFINITY, f32::NAN], &[4]).expect("valid tensor");
    let cpu_result = cpu.scalar_unary(ScalarUnaryOp::Abs, &a).expect("succeeds");
    let metal_result = metal
        .scalar_unary(ScalarUnaryOp::Abs, &a)
        .expect("succeeds");
    let cpu_slice = cpu_result.as_slice().expect("contiguous");
    let metal_slice = metal_result.as_slice().expect("contiguous");
    for (i, (&c, &g)) in cpu_slice.iter().zip(metal_slice.iter()).enumerate() {
        if c.is_nan() {
            assert!(g.is_nan(), "Abs NaN propagation mismatch at index {i}");
        } else {
            assert_eq!(
                c.to_bits(),
                g.to_bits(),
                "Abs cpu/metal bit mismatch at index {i}"
            );
        }
    }
    assert_eq!(cpu_slice[0].to_bits(), (0.0_f32).to_bits());

    // Log/Log2/Log10: `Log(0.0) == -inf`・`Log(負値) == NaN`・
    // `Log(1.0) == 0.0`・`Log(inf) == inf` を両バックエンドで確認する。
    for op in [
        ScalarUnaryOp::Log,
        ScalarUnaryOp::Log2,
        ScalarUnaryOp::Log10,
    ] {
        let a = Tensor::new(vec![0.0, -1.0, 1.0, f32::INFINITY], &[4]).expect("valid tensor");
        let cpu_result = cpu.scalar_unary(op, &a).expect("succeeds");
        let metal_result = metal.scalar_unary(op, &a).expect("succeeds");
        let cpu_slice = cpu_result.as_slice().expect("contiguous");
        let metal_slice = metal_result.as_slice().expect("contiguous");
        assert!(
            cpu_slice[0].is_infinite() && cpu_slice[0] < 0.0,
            "{op:?}(0.0) must be -inf on cpu"
        );
        assert!(
            metal_slice[0].is_infinite() && metal_slice[0] < 0.0,
            "{op:?}(0.0) must be -inf on metal"
        );
        assert!(cpu_slice[1].is_nan(), "{op:?}(-1.0) must be NaN on cpu");
        assert!(metal_slice[1].is_nan(), "{op:?}(-1.0) must be NaN on metal");
        assert_eq!(cpu_slice[2], 0.0, "{op:?}(1.0) must be 0.0 on cpu");
        assert_eq!(metal_slice[2], 0.0, "{op:?}(1.0) must be 0.0 on metal");
        assert!(
            cpu_slice[3].is_infinite() && cpu_slice[3] > 0.0,
            "{op:?}(inf) must be +inf on cpu"
        );
        assert!(
            metal_slice[3].is_infinite() && metal_slice[3] > 0.0,
            "{op:?}(inf) must be +inf on metal"
        );
        // 有限値（index 2 のみ）を assert_parity へ渡す（NaN／inf
        // 要素はクラス一致確認のみに留め数値比較には使わない）。
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("scalar_unary({op:?}) edge case finite element"),
            &metal_slice[2..3],
            &cpu_slice[2..3],
        );
    }

    // Sin/Cos/Tan: `0.0`／`-0.0` の厳密値・inf／NaN 入力の NaN 伝播を
    // 確認する。
    let a = Tensor::new(vec![0.0, -0.0, f32::INFINITY, f32::NAN], &[4]).expect("valid tensor");
    for (op, zero_expect) in [
        (ScalarUnaryOp::Sin, 0.0_f32),
        (ScalarUnaryOp::Cos, 1.0_f32),
        (ScalarUnaryOp::Tan, 0.0_f32),
    ] {
        let cpu_result = cpu.scalar_unary(op, &a).expect("succeeds");
        let metal_result = metal.scalar_unary(op, &a).expect("succeeds");
        let cpu_slice = cpu_result.as_slice().expect("contiguous");
        let metal_slice = metal_result.as_slice().expect("contiguous");
        assert_eq!(
            cpu_slice[0], zero_expect,
            "{op:?}(0.0) unexpected cpu value"
        );
        assert_eq!(
            metal_slice[0], zero_expect,
            "{op:?}(0.0) unexpected metal value"
        );
        for i in [2usize, 3] {
            assert!(
                cpu_slice[i].is_nan(),
                "{op:?} at index {i} must be NaN on cpu (inf/NaN input)"
            );
            assert!(
                metal_slice[i].is_nan(),
                "{op:?} at index {i} must be NaN on metal (inf/NaN input)"
            );
        }
        fandhe_ai_backend_cpu::parity::assert_parity(
            &format!("scalar_unary({op:?}) edge case finite elements"),
            &metal_slice[0..2],
            &cpu_slice[0..2],
        );
    }
}

/// 比較演算・`Clamp` の特殊値（`NaN`／`-0.0`／`inf`）伝播確認（イシュー
/// #1709。CUDA 側 `scalar_op_parity.rs` の同名セクションと同型）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn comparison_and_clamp_scalar_op_edge_cases() {
    let cpu = CpuBackendOps::new();
    let metal = MetalBackendOps::new();

    // 比較演算エッジ: `Eq(NaN, NaN) == 0.0`／`Ne(NaN, NaN) == 1.0`（NaN
    // は自身とも等しくない。IEEE 754）・`Eq(-0.0, 0.0) == 1.0`（符号付き
    // ゼロは数値として等しい）・`Ge(inf, inf) == 1.0` を CPU と bit
    // 同一で確認する。
    let a = Tensor::new(vec![f32::NAN, -0.0, f32::INFINITY, 1.0, f32::NAN], &[5])
        .expect("valid tensor");
    let b = Tensor::new(vec![f32::NAN, 0.0, f32::INFINITY, 2.0, 1.0], &[5]).expect("valid tensor");
    for op in [
        ScalarBinaryOp::Gt,
        ScalarBinaryOp::Ge,
        ScalarBinaryOp::Lt,
        ScalarBinaryOp::Le,
        ScalarBinaryOp::Eq,
        ScalarBinaryOp::Ne,
    ] {
        let cpu_result = cpu.scalar_binary(op, &a, &b).expect("cpu succeeds");
        let metal_result = metal.scalar_binary(op, &a, &b).expect("metal succeeds");
        let cpu_slice = cpu_result.as_slice().expect("contiguous");
        let metal_slice = metal_result.as_slice().expect("contiguous");
        assert_eq!(
            metal_slice, cpu_slice,
            "scalar_binary({op:?}) edge case (NaN/-0.0/inf) cpu/metal bit mismatch"
        );
    }
    assert_eq!(
        cpu.scalar_binary(ScalarBinaryOp::Eq, &a, &b)
            .expect("cpu succeeds")
            .as_slice()
            .expect("contiguous")[0],
        0.0,
        "Eq(NaN, NaN) must be 0.0 per IEEE 754"
    );
    assert_eq!(
        cpu.scalar_binary(ScalarBinaryOp::Ne, &a, &b)
            .expect("cpu succeeds")
            .as_slice()
            .expect("contiguous")[0],
        1.0,
        "Ne(NaN, NaN) must be 1.0 per IEEE 754"
    );
    assert_eq!(
        cpu.scalar_binary(ScalarBinaryOp::Eq, &a, &b)
            .expect("cpu succeeds")
            .as_slice()
            .expect("contiguous")[1],
        1.0,
        "Eq(-0.0, 0.0) must be 1.0 (signed zeros compare equal)"
    );

    // Clamp エッジ: `NaN`／`inf`／`-inf`／`-0.0` を含む入力・通常の
    // `Clamp{-1,1}` と `min > max`（常に `max` を返す）の
    // `Clamp{5.0, 1.0}` の両方を確認する。
    let clamp_input = Tensor::new(
        vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0],
        &[5],
    )
    .expect("valid tensor");
    for (min, max) in [(-1.0f32, 1.0f32), (5.0f32, 1.0f32), (0.0f32, 1.0f32)] {
        let op = ScalarUnaryOp::Clamp { min, max };
        let cpu_result = cpu.scalar_unary(op, &clamp_input).expect("cpu succeeds");
        let metal_result = metal
            .scalar_unary(op, &clamp_input)
            .expect("metal succeeds");
        let cpu_slice = cpu_result.as_slice().expect("contiguous");
        let metal_slice = metal_result.as_slice().expect("contiguous");
        for (i, (&c, &g)) in cpu_slice.iter().zip(metal_slice.iter()).enumerate() {
            if c.is_nan() {
                assert!(
                    g.is_nan(),
                    "Clamp{{min={min},max={max}}} NaN propagation mismatch at index {i}"
                );
            } else {
                assert_eq!(
                    c.to_bits(),
                    g.to_bits(),
                    "Clamp{{min={min},max={max}}} cpu/metal bit mismatch at index {i}"
                );
            }
        }
    }
    // `-0.0` の保持（`Clamp{0.0, 1.0}` は `x < min` の比較で `-0.0 < 0.0`
    // は false のため `-0.0` がそのまま通過する）を bit で確認する。
    let clamp_zero_result = cpu
        .scalar_unary(ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 }, &clamp_input)
        .expect("cpu succeeds");
    assert_eq!(
        clamp_zero_result.as_slice().expect("contiguous")[3].to_bits(),
        (-0.0f32).to_bits(),
        "Clamp{{0.0,1.0}} must preserve -0.0 sign bit"
    );
}

/// 未実装 kind（`Relu`／`Sigmoid`〈活性化系。いずれの sub issue にも
/// 含まれない〉・`Add`／`Maximum`〈いずれの sub issue にも含まれない〉）
/// は `BackendError::Unsupported` を返し、パニックしないことの回帰ガード
/// （`crate::scalar_op_source` モジュール doc「スコープ」参照。`Log`
/// 〈超越関数系。#1708〉・`Clamp`〈#1709〉は実装済みになったため番兵
/// から外した）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn unimplemented_kinds_return_unsupported() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).expect("valid tensor");
    let b = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).expect("valid tensor");

    // `Log`（#1708）・`Clamp`（#1709）は実装済みになったため、番兵 kind
    // をいずれの sub issue にも含まれない `Relu`／`Sigmoid`（活性化系）
    // へ付け替える。
    let err = metal
        .scalar_unary(ScalarUnaryOp::Relu, &a)
        .expect_err("Relu is not implemented by #1707/#1708/#1709");
    assert!(matches!(err, BackendError::Unsupported(_)));

    let err = metal
        .scalar_unary(ScalarUnaryOp::Sigmoid, &a)
        .expect_err("Sigmoid is not implemented by #1707/#1708/#1709");
    assert!(matches!(err, BackendError::Unsupported(_)));

    let err = metal
        .scalar_binary(ScalarBinaryOp::Add, &a, &b)
        .expect_err("Add is not implemented by #1707/#1708/#1709");
    assert!(matches!(err, BackendError::Unsupported(_)));

    let err = metal
        .scalar_binary(ScalarBinaryOp::Maximum, &a, &b)
        .expect_err("Maximum is not implemented by #1707/#1708/#1709");
    assert!(matches!(err, BackendError::Unsupported(_)));
}

/// 非ブロードキャスト可能形状（`[2, 2]` vs `[3]`）は
/// `BackendError::ShapeMismatch` を返す。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn scalar_binary_incompatible_shape_is_rejected() {
    let metal = MetalBackendOps::new();
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("valid tensor");
    let b = Tensor::new(vec![1.0, 2.0, 3.0], &[3]).expect("valid tensor");
    let err = metal
        .scalar_binary(ScalarBinaryOp::Sub, &a, &b)
        .expect_err("incompatible shapes must be rejected");
    assert!(matches!(err, BackendError::ShapeMismatch(_)));
}
