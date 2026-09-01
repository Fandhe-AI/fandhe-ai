//! SplitK（`gemm_variant::GemmVariantKind::SplitK`）の parity 失敗が
//! カーネルバグではなく K 方向分割縮約による累積順序の再結合誤差である
//! ことを、GPU を使わずホスト側 Rust で固定するための決定記録テスト
//! （イシュー #1100）。
//!
//! # 背景・このテストの役割
//!
//! `docs/perf/cuda-gemm-f32-variant-selection.md` §1a（#1031 実機実測）が
//! GB10 実機で報告した SplitK 経路の複合判定 FAIL（m=n=128・k=8192・
//! `num_splits=8`・seed=3・fail_count=8/16384・max_abs_diff=3.662e-4・
//! max_rel_err=1.090e-2）を、GPU カーネルと**同一の数式モデル**（split
//! ごとに `f32::mul_add` の K 昇順連鎖で部分和を作り、`s` 昇順の f32
//! 加算で縮約する。`kernels_gemm_variants::SPLITK_PARTIAL_F32`／
//! `SPLITK_REDUCE_F32` の演算順序をそのままホスト側で再現したもの）で
//! ホスト側 CPU 上に再現する。
//!
//! 3 指標（fail_count・max_abs_diff・max_rel_err）が GB10 実機レポートと
//! 一致することは、GPU カーネル自体（インデックス計算・境界チェック・
//! 縮約順序）にバグが無く、**K 方向を複数の部分和へ分割してから縮約する
//! という split-K の演算順序そのものが、CPU 参照実装
//! （[`fandhe_ai_backend_cpu::matmul_reference_fma`]。K 方向を分割せず
//! 逐次 k 昇順で `mul_add` する単一連鎖）と異なる丸め結果を生む**ことの
//! 根拠になる。イシュー #1100 の実装計画 §2.1 はこの一致を計画セッション
//! で確認済みであり、本テストはその根拠を CI で再現可能な形へ固定する
//! （実機接続なしで再現できることが本テストの目的であり、GPU 資源は
//! 一切不要）。
//!
//! # なぜ「精度を上げても解決しない」ことも合わせて固定するか
//!
//! 実装計画 §2.1 は f32 縮約→ f64 縮約 → f64 部分和+f64 縮約（ほぼ厳密値）
//! と精度を上げても fail 数が減らないことを確認している（差分の支配項は
//! CPU 参照実装自身の丸め誤差であり、真値ゼロ近傍〈桁落ち〉要素では
//! 絶対誤差救済 1e-5 を超える）。これにより「partial reduction の数値
//! 修正で parity を通す」ことが達成不能であるという撤退判断（#1100）の
//! 根拠を、単一の f32 モデルだけでなく複数の精度モデルで裏付ける。

use fandhe_ai_backend_cpu::{ABSOLUTE_RESCUE_THRESHOLD, RELATIVE_TOLERANCE, compare};

/// GB10 実機レポート（#1031・`docs/perf/cuda-gemm-f32-variant-selection.md`
/// §1a）と同一の形状・分割数・シード。
const M: usize = 128;
const N: usize = 128;
const K: usize = 8192;
const NUM_SPLITS: usize = 8;
const SEED: u64 = 3;

/// GB10 実機レポートが記録した複合判定の 3 指標（許容誤差なしの厳密一致を
/// 求める整数指標 [`GB10_FAIL_COUNT`] と、浮動小数点表示丸め〈`{:.3e}` =
/// 有効数字 4 桁〉に由来する許容差を持つ [`assert_matches_gb10_report`] の
/// 2 指標）。
const GB10_FAIL_COUNT: usize = 8;
const GB10_MAX_ABS_DIFF: f64 = 3.662e-4;
const GB10_MAX_REL_ERR: f64 = 1.090e-2;

/// GB10 実機レポート値との一致判定に使う相対許容誤差。実機レポートの
/// 数値は `assert_parity`（`crate::parity::assert_parity`）の
/// `{:.3e}` フォーマット（有効数字 4 桁）を人力転記した値のため、
/// 表示丸め分の差異を許容する（本テストの主張は「ビット完全一致」ではなく
/// 「同一の数式モデルが同一の破綻を再現する」こと）。
const GB10_REPORT_MATCH_RELATIVE_TOLERANCE: f64 = 5e-3;

fn assert_matches_gb10_report(label: &str, actual: f64, expected: f64) {
    let rel = ((actual - expected).abs()) / expected.abs().max(1e-12);
    assert!(
        rel < GB10_REPORT_MATCH_RELATIVE_TOLERANCE,
        "{label}: ホストモデルの実測値 {actual:.6e} が GB10 実機レポート値 \
         {expected:.6e} と一致しない（相対差 {rel:.3e} >= 許容 \
         {GB10_REPORT_MATCH_RELATIVE_TOLERANCE:.1e}）。#1100 撤退判断の \
         前提（split 順序の再結合誤差が実機と同一挙動であること）が崩れて \
         いないか確認する"
    );
}

/// [`fandhe_ai_backend_cpu::matmul_reference_fma`] と同じ決定的シード PRNG
/// で A（`m x k`）・B（`k x n`）を生成する（`tests/gemm_f32_variants.rs::
/// gen_ab` と同一の生成方法。GB10 実機テストと同じ入力を再現する）。
fn gen_ab(seed: u64, m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    (rng.fill_vec(m * k), rng.fill_vec(k * n))
}

/// split-K の縮約に使う精度（部分和・縮約それぞれ独立に選べる）。
///
/// 実装計画 §2.1 の 3 モデル（f32 部分和+f32 縮約〈現行カーネルと同一〉・
/// f32 部分和+f64 縮約・f64 部分和+f64 縮約〈ほぼ厳密値〉）を 1 つの
/// ジェネリック実装で表現する。
#[derive(Debug, Clone, Copy)]
enum Precision {
    F32,
    F64,
}

/// [`SPLITK_PARTIAL_F32`](fandhe_ai_backend_cuda) と同一の分割・境界規則
/// （`k_per_split = ceil(k / num_splits)`・末尾分割の空範囲は 0.0 のまま
/// 無条件出力）で split-K を計算し、CPU 参照実装
/// （[`fandhe_ai_backend_cpu::matmul_reference_fma`]）とビット互換の入力
/// 契約（行優先・`a[row*k+p]`・`b[p*n+col]`）でホスト側に再現する。
///
/// `partial_precision`: 各 split 内の K 昇順連鎖の精度（`F32` は
/// カーネルと同じ `f32::mul_add` 連鎖・`F64` はほぼ厳密値のための
/// `f64` 昇算）。
/// `reduce_precision`: split 方向（`s` 昇順）の縮約精度。
///
/// 戻り値は常に `f32` へ丸めて返す（`matmul_reference_fma` の出力型・
/// `compare` の入力型と揃える。丸めは最終縮約結果に対してのみ 1 回行い、
/// 精度モデルの違いは縮約過程の内部精度にのみ現れる）。
#[allow(clippy::too_many_arguments)]
fn splitk_host_model(
    a: &[f32],
    b: &[f32],
    m: usize,
    n: usize,
    k: usize,
    num_splits: usize,
    partial_precision: Precision,
    reduce_precision: Precision,
) -> Vec<f32> {
    let k_per_split = k.div_ceil(num_splits);

    // [num_splits, m, n] 形状の部分和（f64 で保持し、F32 モデルでは
    // 都度 f32 へ丸めた値を格納することで「f32 部分和」を表現する。
    // f64 で保持する理由はホスト側の一時領域を単純化するためであり、
    // F32 精度モデルの数値挙動は各要素を都度 `as f32 as f64` へ丸める
    // ことで完全に再現している）。
    let mut partial = vec![0.0f64; num_splits * m * n];

    for s in 0..num_splits {
        let k_start = s * k_per_split;
        let k_end = (k_start + k_per_split).min(k);
        for row in 0..m {
            for col in 0..n {
                let idx = s * m * n + row * n + col;
                if k_start >= k_end {
                    // 末尾の空分割: SPLITK_PARTIAL_F32 と同じく無条件で
                    // 0.0 を書く（早期 return せず acc=0.0 のまま出力）。
                    partial[idx] = 0.0;
                    continue;
                }
                match partial_precision {
                    Precision::F32 => {
                        let mut acc = 0.0f32;
                        for p in k_start..k_end {
                            acc = a[row * k + p].mul_add(b[p * n + col], acc);
                        }
                        partial[idx] = acc as f64;
                    }
                    Precision::F64 => {
                        let mut acc = 0.0f64;
                        for p in k_start..k_end {
                            acc = (a[row * k + p] as f64).mul_add(b[p * n + col] as f64, acc);
                        }
                        partial[idx] = acc;
                    }
                }
            }
        }
    }

    // SPLITK_REDUCE_F32 と同じ s 昇順の逐次加算（乗算を伴わない単純和の
    // ため mul_add ではなく通常の加算を使う。カーネル側も
    // `acc += c_partial[...]` であり FMA ではない）。
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let idx = row * n + col;
            match reduce_precision {
                Precision::F32 => {
                    let mut acc = 0.0f32;
                    for s in 0..num_splits {
                        acc += partial[s * m * n + idx] as f32;
                    }
                    out[idx] = acc;
                }
                Precision::F64 => {
                    let mut acc = 0.0f64;
                    for s in 0..num_splits {
                        acc += partial[s * m * n + idx];
                    }
                    out[idx] = acc as f32;
                }
            }
        }
    }
    out
}

/// f32 部分和 + f32 縮約（現行 `SPLITK_PARTIAL_F32`／`SPLITK_REDUCE_F32`
/// カーネルと同一の数式モデル）が GB10 実機レポートの 3 指標
/// （fail_count・max_abs_diff・max_rel_err）と一致すること。
///
/// `fail_count` は複合判定の整数カウントであり表示丸めの影響を受けない
/// ため厳密一致（`assert_eq!`）で検査し、`max_abs_diff`／`max_rel_err`
/// （実機レポートは `{:.3e}` 表示丸め値の人力転記）は
/// [`assert_matches_gb10_report`] の相対許容で検査する。
#[test]
fn f32_partial_f32_reduce_matches_gb10_report() {
    let (a, b) = gen_ab(SEED, M, N, K);

    let mut expected = vec![0.0f32; M * N];
    fandhe_ai_backend_cpu::matmul_reference_fma(&a, &b, &mut expected, M, N, K)
        .expect("CPU reference must succeed for well-formed dims");

    let actual = splitk_host_model(&a, &b, M, N, K, NUM_SPLITS, Precision::F32, Precision::F32);

    let report = compare(&actual, &expected).expect("length must match (same m*n)");

    assert_eq!(
        report.fail_count, GB10_FAIL_COUNT,
        "ホストモデルの fail_count が GB10 実機レポート（{GB10_FAIL_COUNT}）と \
         一致しない: actual={report:?}"
    );
    assert_matches_gb10_report("max_abs_diff", report.max_abs_diff, GB10_MAX_ABS_DIFF);
    assert_matches_gb10_report("max_rel_err", report.max_rel_err, GB10_MAX_REL_ERR);

    // この一致自体が「複合判定 FAIL」であることを明示する（撤退判断の
    // 前提: SplitK は現行の許容誤差〈変更不可〉を満たさない）。
    assert!(
        !report.passes(),
        "GB10 実機と同じ FAIL を再現するはずが複合判定 PASS になった \
         （relative_tolerance={RELATIVE_TOLERANCE:e}, \
         absolute_rescue_threshold={ABSOLUTE_RESCUE_THRESHOLD:e}）"
    );
}

/// 実装計画 §2.1 の追加検証: 縮約精度・部分和精度をいずれも f64
/// （ほぼ厳密値）まで引き上げても複合判定 FAIL が解消しないこと
/// （fail_count が 0 にならないこと）。
///
/// 差分の支配項が CPU 参照実装（`matmul_reference_fma`。K=8192 の逐次
/// f32 `mul_add` 連鎖）自身の丸め誤差であり、split-K 側の数値精度を
/// 上げても解消しないという撤退判断の核心を固定する（「partial
/// reduction の数値修正で parity を通す」ことが達成不能であることの
/// 根拠）。
#[test]
fn higher_precision_reduction_does_not_eliminate_fail_count() {
    let (a, b) = gen_ab(SEED, M, N, K);

    let mut expected = vec![0.0f32; M * N];
    fandhe_ai_backend_cpu::matmul_reference_fma(&a, &b, &mut expected, M, N, K)
        .expect("CPU reference must succeed for well-formed dims");

    let models: &[(&str, Precision, Precision)] = &[
        (
            "f32_partial+f32_reduce (現行カーネルと同一)",
            Precision::F32,
            Precision::F32,
        ),
        ("f32_partial+f64_reduce", Precision::F32, Precision::F64),
        (
            "f64_partial+f64_reduce (ほぼ厳密値)",
            Precision::F64,
            Precision::F64,
        ),
    ];

    for &(label, partial_precision, reduce_precision) in models {
        let actual = splitk_host_model(
            &a,
            &b,
            M,
            N,
            K,
            NUM_SPLITS,
            partial_precision,
            reduce_precision,
        );
        let report = compare(&actual, &expected).expect("length must match (same m*n)");
        assert!(
            report.fail_count > 0,
            "{label}: 精度を上げても fail_count が 0 のままなら「split \
             順序の再結合誤差」という撤退判断の前提が崩れる想定外の結果 \
             （report={report:?}）"
        );
    }
}
