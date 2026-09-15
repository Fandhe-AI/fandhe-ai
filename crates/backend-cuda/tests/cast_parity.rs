//! イシュー #1751: `CastOps`（dtype 変換。8 方向）の CPU-CUDA 数値
//! 一致検証。
//!
//! `unique_parity.rs` と同じ構成方針を踏襲する: 環境適応スモーク
//! （属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `BackendError::CudaUnavailable` を確認して panic しないことのみ
//! 検証）と、実機必須の網羅（`#[ignore]`。DGX Spark GB10 等）を分離
//! する。
//!
//! **契約は bit 完全一致**（算術を含まない変換のため。NaN のみ payload
//! がプラットフォーム依存のためクラス一致で比較する。
//! `fandhe_ai_tensor_core::cast` モジュール doc・`docs/tensor-core-
//! cast-design.md` 参照）。
//!
//! fixture は NaN・±inf・±0.0・f32 subnormal・`±2^31`／`±2^63` 近傍
//! （`i32`／`i64` 飽和境界）・`f64::MAX`／`MIN`・f64 subnormal を含む
//! （`docs/tensor-core-cast-design.md` §11 の事前登録どおり）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test cast_parity -- --ignored --nocapture
//! ```

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{CastOps, Tensor};

/// f32 fixture（8 方向の f32 側入出力すべてに使う共通ケース集合）。
fn f32_fixture() -> Vec<f32> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        1.5,
        -2.5,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,      // subnormal に近い最小正規化数
        1e-45,                  // subnormal
        2147483648.0,           // i32::MAX を超える境界（飽和）
        2147483520.0,           // i32::MAX 直下（非飽和）
        -2147483904.0,          // i32::MIN を下回る境界（飽和）
        9223372036854775808.0,  // i64::MAX を超える境界（飽和）
        -9223372036854775808.0, // ちょうど i64::MIN（非飽和・exact）
    ]
}

/// f32 の bit 完全一致（NaN のみクラス一致）を検証する。
fn assert_f32_bits_eq(a: f32, b: f32, ctx: &str) {
    if a.is_nan() && b.is_nan() {
        return;
    }
    assert_eq!(a.to_bits(), b.to_bits(), "{ctx}: bit 不一致 (a={a}, b={b})");
}

/// f64 の bit 完全一致（NaN のみクラス一致）を検証する。
fn assert_f64_bits_eq(a: f64, b: f64, ctx: &str) {
    if a.is_nan() && b.is_nan() {
        return;
    }
    assert_eq!(a.to_bits(), b.to_bits(), "{ctx}: bit 不一致 (a={a}, b={b})");
}

fn run_all_directions(cpu: &CpuBackendOps, cuda: &CudaBackendOps) {
    let f32_in = Tensor::new(f32_fixture(), &[f32_fixture().len()]).expect("valid tensor");

    // f32 -> f64
    let cpu_f64 = CastOps::cast_f32_to_f64(cpu, &f32_in).expect("cpu f32->f64");
    let cuda_f64 = CastOps::cast_f32_to_f64(cuda, &f32_in).expect("cuda f32->f64");
    for (i, (&a, &b)) in cuda_f64
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(cpu_f64.as_slice().expect("contiguous").iter())
        .enumerate()
    {
        assert_f64_bits_eq(a, b, &format!("f32->f64[{i}]"));
    }

    // f32 -> i32
    let cpu_i32 = CastOps::cast_f32_to_i32(cpu, &f32_in).expect("cpu f32->i32");
    let cuda_i32 = CastOps::cast_f32_to_i32(cuda, &f32_in).expect("cuda f32->i32");
    assert_eq!(
        cuda_i32.as_slice().expect("contiguous"),
        cpu_i32.as_slice().expect("contiguous"),
        "f32->i32 不一致"
    );

    // f32 -> i64
    let cpu_i64 = CastOps::cast_f32_to_i64(cpu, &f32_in).expect("cpu f32->i64");
    let cuda_i64 = CastOps::cast_f32_to_i64(cuda, &f32_in).expect("cuda f32->i64");
    assert_eq!(
        cuda_i64.as_slice().expect("contiguous"),
        cpu_i64.as_slice().expect("contiguous"),
        "f32->i64 不一致"
    );

    // f32 -> bool
    let cpu_bool = CastOps::cast_f32_to_bool(cpu, &f32_in).expect("cpu f32->bool");
    let cuda_bool = CastOps::cast_f32_to_bool(cuda, &f32_in).expect("cuda f32->bool");
    assert_eq!(
        cuda_bool.as_slice().expect("contiguous"),
        cpu_bool.as_slice().expect("contiguous"),
        "f32->bool 不一致"
    );

    // f64 -> f32
    let f64_in = Tensor::new(
        vec![
            0.0,
            -0.0,
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
            5e-324,                    // f64 subnormal（f32 では 0 へ丸まる）
            1e-40,                     // f64 -> f32 subnormal
            (1i64 << 24) as f64 + 1.0, // f32 仮数部で非可逆
        ],
        &[11],
    )
    .expect("valid tensor");
    let cpu_from_f64 = CastOps::cast_f64_to_f32(cpu, &f64_in).expect("cpu f64->f32");
    let cuda_from_f64 = CastOps::cast_f64_to_f32(cuda, &f64_in).expect("cuda f64->f32");
    for (i, (&a, &b)) in cuda_from_f64
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(cpu_from_f64.as_slice().expect("contiguous").iter())
        .enumerate()
    {
        assert_f32_bits_eq(a, b, &format!("f64->f32[{i}]"));
    }

    // i32 -> f32
    let i32_in = Tensor::new(
        vec![0, 1, -1, i32::MAX, i32::MIN, 1 << 24, (1 << 24) + 1],
        &[7],
    )
    .expect("valid tensor");
    let cpu_from_i32 = CastOps::cast_i32_to_f32(cpu, &i32_in).expect("cpu i32->f32");
    let cuda_from_i32 = CastOps::cast_i32_to_f32(cuda, &i32_in).expect("cuda i32->f32");
    for (i, (&a, &b)) in cuda_from_i32
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(cpu_from_i32.as_slice().expect("contiguous").iter())
        .enumerate()
    {
        assert_f32_bits_eq(a, b, &format!("i32->f32[{i}]"));
    }

    // i64 -> f32（タイケース: 2^24+1・2^25+1・2^25+3 を含む。Metal
    // 側実機判定の事前登録〈`docs/tensor-core-cast-design.md` §11〉と
    // 同じ fixture を CUDA 側でも共有する）。
    let i64_in = Tensor::new(
        vec![
            0,
            1,
            -1,
            i64::MAX,
            i64::MIN,
            (1i64 << 24) + 1,
            (1i64 << 25) + 1,
            (1i64 << 25) + 3,
        ],
        &[8],
    )
    .expect("valid tensor");
    let cpu_from_i64 = CastOps::cast_i64_to_f32(cpu, &i64_in).expect("cpu i64->f32");
    let cuda_from_i64 = CastOps::cast_i64_to_f32(cuda, &i64_in).expect("cuda i64->f32");
    for (i, (&a, &b)) in cuda_from_i64
        .as_slice()
        .expect("contiguous")
        .iter()
        .zip(cpu_from_i64.as_slice().expect("contiguous").iter())
        .enumerate()
    {
        assert_f32_bits_eq(a, b, &format!("i64->f32[{i}]"));
    }

    // bool -> f32
    let bool_in = Tensor::new(vec![true, false, true, true, false], &[5]).expect("valid tensor");
    let cpu_from_bool = CastOps::cast_bool_to_f32(cpu, &bool_in).expect("cpu bool->f32");
    let cuda_from_bool = CastOps::cast_bool_to_f32(cuda, &bool_in).expect("cuda bool->f32");
    assert_eq!(
        cuda_from_bool.as_slice().expect("contiguous"),
        cpu_from_bool.as_slice().expect("contiguous"),
        "bool->f32 不一致"
    );
}

#[test]
fn cast_parity_smoke_env_adaptive() {
    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    // accessor 到達性の確認に f32->i32 の最小ケースを使う（CUDA 非搭載
    // 環境かどうかの分岐材料。`unique_parity.rs` と同じ方針）。
    let probe = Tensor::new(vec![1.0f32], &[1]).expect("valid tensor");
    match CastOps::cast_f32_to_i32(&cuda, &probe) {
        Ok(_) => {
            run_all_directions(&cpu, &cuda);

            // run-to-run 決定性。
            let f32_in = Tensor::new(f32_fixture(), &[f32_fixture().len()]).expect("valid tensor");
            let out1 = CastOps::cast_f32_to_i32(&cuda, &f32_in).expect("run1");
            let out2 = CastOps::cast_f32_to_i32(&cuda, &f32_in).expect("run2");
            assert_eq!(
                out1.as_slice().expect("contiguous"),
                out2.as_slice().expect("contiguous"),
                "run-to-run で bit 同一のはず"
            );

            // 空テンソル: GPU 起動なしの早期 return 経路。
            let empty = Tensor::new(Vec::<f32>::new(), &[0]).expect("valid tensor");
            let empty_out = CastOps::cast_f32_to_i32(&cuda, &empty).expect("cuda cast(empty)");
            assert_eq!(empty_out.shape(), &[0]);

            // 非 contiguous view（transpose 済み）も正しく稠密化される。
            let base =
                Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).expect("valid tensor");
            let transposed = base.permute(&[1, 0]).expect("rank 2 permute always valid");
            let cpu_t = CastOps::cast_f32_to_i32(&cpu, &transposed).expect("cpu transposed");
            let cuda_t = CastOps::cast_f32_to_i32(&cuda, &transposed).expect("cuda transposed");
            assert_eq!(cuda_t.shape(), &[3, 2]);
            assert_eq!(
                cuda_t.as_slice().expect("contiguous"),
                cpu_t.as_slice().expect("contiguous")
            );
        }
        Err(BackendError::CudaUnavailable(_)) => {
            // CUDA 非搭載環境（通常 CI）。panic せず終了する
            // （`unique_parity.rs` と同じ環境適応方針）。
        }
        Err(other) => panic!("unexpected error on CUDA-equipped runner: {other}"),
    }
}

/// より広いランダム系列でのサイズ網羅を実機で確認する（DGX Spark
/// GB10 等。`#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cast_matches_cpu_across_sizes() {
    use bench_harness::rng::Xorshift64Star;

    let cpu = CpuBackendOps::new();
    let cuda = CudaBackendOps::new(0);

    for (i, &n) in [1usize, 2, 3, 7, 255, 256, 257, 4097, 1 << 16]
        .iter()
        .enumerate()
    {
        let data = Xorshift64Star::new(30000 + i as u64).fill_vec(n);
        let x = Tensor::new(data, &[n]).expect("valid tensor");

        let cpu_i32 = CastOps::cast_f32_to_i32(&cpu, &x).expect("cpu f32->i32");
        let cuda_i32 = CastOps::cast_f32_to_i32(&cuda, &x).expect("cuda f32->i32");
        assert_eq!(
            cuda_i32.as_slice().expect("contiguous"),
            cpu_i32.as_slice().expect("contiguous"),
            "f32->i32 不一致 (n={n})"
        );

        let cpu_f64 = CastOps::cast_f32_to_f64(&cpu, &x).expect("cpu f32->f64");
        let cuda_f64 = CastOps::cast_f32_to_f64(&cuda, &x).expect("cuda f32->f64");
        for (j, (&a, &b)) in cuda_f64
            .as_slice()
            .expect("contiguous")
            .iter()
            .zip(cpu_f64.as_slice().expect("contiguous").iter())
            .enumerate()
        {
            assert_f64_bits_eq(a, b, &format!("f32->f64[{j}] (n={n})"));
        }
    }
}
