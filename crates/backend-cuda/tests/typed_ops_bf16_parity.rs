//! CUDA `TypedOps<half::bf16>`（`crate::typed_bf16`。イシュー #1704）の
//! 数値一致回帰テスト。
//!
//! `docs/backend-dtype-dispatch-design.md` §6 の判定契約（f16／bf16 出力は
//! 参照値側も出力 dtype と同じ丸めを経てから比較する）を踏まえ、以下の
//! 三層構成で検証する:
//!
//! - **環境適応スモーク**（属性なし。通常 CI で実行）: `typed_ops_bf16()`
//!   accessor が `Some` を返すこと、CUDA 非搭載環境では各演算が
//!   `BackendError::CudaUnavailable` を panic なく返すことを確認する
//! - **層 1（`#[ignore]`・実機）**: 同一バックエンド内の構造的不変条件
//!   `TypedOps::<bf16>::op(x) == bf16::from_f32(BackendOps::op_f32(f32(x)))`
//!   を bit 完全一致で検証する（`gemm` のみ `gemm_fp32_strict` と比較）。
//!   run-to-run 決定性（2 回実行の bit 一致）も併せて確認する
//! - **層 2（`#[ignore]`・実機）**: CPU `BackendOps`（f32 参照実装）を
//!   bf16 へ丸めた値とのクロスバックエンド判定。丸め境界またぎによる
//!   誤判定（`typed_bf16.rs` モジュール doc 参照）を避けるため、入力を
//!   bf16 で正確に表現できる小整数・小 K（累算順序に依存しない）に
//!   限定する。`exp`／`tanh` はこの技法が使えないため対象外とする。
//!   （**注記**: CPU 側 `TypedOps<bf16>` はイシュー #1699・PR #1794 で
//!   別途実装中で本ファイル作成時点では未マージのため、参照値は CPU の
//!   既存 f32 `BackendOps` をホスト側で bf16 丸めして構築する。CPU 版が
//!   マージされ次第、`TypedOps::<bf16>::op` 同士の比較へ差し替え可能）
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test typed_ops_bf16_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cuda::CudaBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};
use half::bf16;

/// 決定的シードで `[-8, 8]` の整数値のみからなる bf16 テンソルを生成する。
///
/// bf16（仮数部 7bit）は `[-8, 8]` の整数を正確に表現でき、かつこの
/// レンジの GEMM 積・総和は小 K のもとで f32／f64 中間値も丸め誤差なく
/// 正確に表現できる（累算順序への非依存性が構造的に成立する）。層 2
/// のクロスバックエンド判定を丸め境界またぎの誤判定なしで成立させる
/// ための入力生成ヘルパー（`typed_bf16.rs` モジュール doc「数値・
/// 意味論上の明文化事項」・本ファイル冒頭コメント参照）。
fn small_int_bf16_tensor(seed: u64, shape: &[usize]) -> Tensor<bf16> {
    let numel: usize = shape.iter().product();
    let mut rng = Xorshift64Star::new(seed);
    let raw = rng.fill_vec(numel);
    let data: Vec<bf16> = raw
        .iter()
        .map(|&v| {
            // Xorshift64Star::fill_vec は [0, 1) の f32 を返す前提
            // （`bench_harness::rng` 参照）。[-8, 8] の整数へ量子化する。
            let scaled = (v * 17.0).floor() - 8.0;
            bf16::from_f32(scaled.clamp(-8.0, 8.0))
        })
        .collect();
    Tensor::new(data, shape).expect("valid tensor")
}

fn f32_of(t: &Tensor<bf16>) -> Vec<f32> {
    t.host_slice().iter().map(|v| v.to_f32()).collect()
}

fn to_f32_tensor(t: &Tensor<bf16>) -> Tensor<f32> {
    Tensor::new(f32_of(t), t.shape()).expect("valid tensor")
}

/// `Tensor<f32>` の内容を bf16 最近接偶数丸めで丸めた `Vec<f32>` を返す
/// （`downcast_f32` → `f32_of` を素朴に再実装したもの。参照値の構築に
/// 使うため `crate::typed_bf16` の実装コードは経由しない）。
fn round_via_bf16(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice()
        .iter()
        .map(|&v| bf16::from_f32(v).to_f32())
        .collect()
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。`backend_ops_real_device.rs
/// ::backend_ops_gemm_parity_smoke_env_adaptive` と同じ分岐パターン: CUDA
/// 非搭載環境では型付きエラー（`CudaUnavailable`）を確認して早期 return
/// する。
#[test]
fn typed_ops_bf16_accessor_and_unsupported_env_adaptive() {
    let cuda = CudaBackendOps::new(0);
    let ops: &dyn BackendOps = &cuda;
    let typed = ops
        .typed_ops_bf16()
        .expect("typed_ops_bf16 accessor must always return Some for CudaBackendOps");

    let a = small_int_bf16_tensor(9001, &[2, 2]);
    let b = small_int_bf16_tensor(9002, &[2, 2]);

    match TypedOps::<bf16>::gemm(typed, &a, &b) {
        Ok(_) => {
            // 実機（CUDA 搭載 CI ランナー）: 層 1／層 2 の網羅ケースまで
            // 実行する。
            run_layer1_structural_checks(&cuda);
            run_layer2_cross_backend_checks(&cuda);
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for TypedOps::<bf16>::gemm: {other}"),
    }
}

/// 層 1: 同一バックエンド内の構造的不変条件（bit 完全一致）。
fn run_layer1_structural_checks(cuda: &CudaBackendOps) {
    let typed: &dyn TypedOps<bf16> = cuda;

    let a = small_int_bf16_tensor(101, &[3, 4]);
    let b = small_int_bf16_tensor(102, &[4, 5]);

    // gemm: TypedOps::<bf16>::gemm と gemm_fp32_strict(f32(a), f32(b)) を
    // bf16 へ丸めた結果が bit 完全一致すること。
    let typed_gemm = TypedOps::<bf16>::gemm(typed, &a, &b).expect("cuda typed gemm succeeds");
    let a32 = to_f32_tensor(&a);
    let b32 = to_f32_tensor(&b);
    let ref32 = BackendOps::gemm_fp32_strict(cuda, &a32, &b32).expect("cuda f32 gemm succeeds");
    assert_eq!(
        f32_of(&typed_gemm),
        round_via_bf16(&ref32),
        "gemm bit mismatch"
    );

    // add/mul/relu: BackendOps 経由の f32 演算を bf16 へ丸めた結果と bit
    // 完全一致すること。
    let x = small_int_bf16_tensor(103, &[6]);
    let y = small_int_bf16_tensor(104, &[6]);
    let x32 = to_f32_tensor(&x);
    let y32 = to_f32_tensor(&y);

    let typed_add = TypedOps::<bf16>::add(typed, &x, &y).unwrap();
    let ref_add = BackendOps::add(cuda, &x32, &y32).unwrap();
    assert_eq!(
        f32_of(&typed_add),
        round_via_bf16(&ref_add),
        "add bit mismatch"
    );

    let typed_mul = TypedOps::<bf16>::mul(typed, &x, &y).unwrap();
    let ref_mul = BackendOps::mul(cuda, &x32, &y32).unwrap();
    assert_eq!(
        f32_of(&typed_mul),
        round_via_bf16(&ref_mul),
        "mul bit mismatch"
    );

    let typed_relu = TypedOps::<bf16>::relu(typed, &x).unwrap();
    let ref_relu = BackendOps::relu(cuda, &x32).unwrap();
    assert_eq!(
        f32_of(&typed_relu),
        round_via_bf16(&ref_relu),
        "relu bit mismatch"
    );

    // sum/max: 軸縮約。
    let s = small_int_bf16_tensor(105, &[3, 4]);
    let s32 = to_f32_tensor(&s);

    let typed_sum = TypedOps::<bf16>::sum(typed, &s, None).unwrap();
    let ref_sum = BackendOps::sum(cuda, &s32, None).unwrap();
    assert_eq!(
        f32_of(&typed_sum),
        round_via_bf16(&ref_sum),
        "sum bit mismatch"
    );

    let typed_max = TypedOps::<bf16>::max(typed, &s, Some(1)).unwrap();
    let ref_max = BackendOps::max(cuda, &s32, Some(1)).unwrap();
    assert_eq!(
        f32_of(&typed_max),
        round_via_bf16(&ref_max),
        "max bit mismatch"
    );

    // run-to-run 決定性: 2 回実行して bit 一致すること。
    let run1 = TypedOps::<bf16>::gemm(typed, &a, &b).unwrap();
    let run2 = TypedOps::<bf16>::gemm(typed, &a, &b).unwrap();
    assert_eq!(
        f32_of(&run1),
        f32_of(&run2),
        "gemm must be run-to-run deterministic"
    );
}

/// 層 2: CPU `BackendOps`（f32 参照実装）を bf16 丸めした値との
/// クロスバックエンド判定。小整数・小 K の入力に限定するため丸め境界
/// またぎが起きず、`fandhe_ai_backend_cpu::assert_parity` の複合判定
/// （1e-3/1e-5）は変更せずそのまま適用できる（`typed_bf16.rs` モジュール
/// doc 参照）。
fn run_layer2_cross_backend_checks(cuda: &CudaBackendOps) {
    let cpu = CpuBackendOps::new();
    let cuda_typed: &dyn TypedOps<bf16> = cuda;

    let a = small_int_bf16_tensor(201, &[8, 16]);
    let b = small_int_bf16_tensor(202, &[16, 8]);
    let a32 = to_f32_tensor(&a);
    let b32 = to_f32_tensor(&b);

    // gemm: 小整数・K=16（bf16 で正確に表現可能な整数域）。
    let cuda_gemm = TypedOps::<bf16>::gemm(cuda_typed, &a, &b).expect("cuda gemm succeeds");
    let cpu_gemm_f32 = BackendOps::gemm(&cpu, &a32, &b32).expect("cpu gemm succeeds");
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 cuda-cpu gemm parity",
        &f32_of(&cuda_gemm),
        &round_via_bf16(&cpu_gemm_f32),
    );

    let x = small_int_bf16_tensor(203, &[10]);
    let y = small_int_bf16_tensor(204, &[10]);
    let x32 = to_f32_tensor(&x);
    let y32 = to_f32_tensor(&y);

    let cuda_add = TypedOps::<bf16>::add(cuda_typed, &x, &y).unwrap();
    let cpu_add_f32 = BackendOps::add(&cpu, &x32, &y32).unwrap();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 cuda-cpu add parity",
        &f32_of(&cuda_add),
        &round_via_bf16(&cpu_add_f32),
    );

    let cuda_mul = TypedOps::<bf16>::mul(cuda_typed, &x, &y).unwrap();
    let cpu_mul_f32 = BackendOps::mul(&cpu, &x32, &y32).unwrap();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 cuda-cpu mul parity",
        &f32_of(&cuda_mul),
        &round_via_bf16(&cpu_mul_f32),
    );

    let cuda_relu = TypedOps::<bf16>::relu(cuda_typed, &x).unwrap();
    let cpu_relu_f32 = BackendOps::relu(&cpu, &x32).unwrap();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 cuda-cpu relu parity",
        &f32_of(&cuda_relu),
        &round_via_bf16(&cpu_relu_f32),
    );

    let s = small_int_bf16_tensor(205, &[4, 6]);
    let s32 = to_f32_tensor(&s);

    let cuda_sum = TypedOps::<bf16>::sum(cuda_typed, &s, None).unwrap();
    let cpu_sum_f32 = BackendOps::sum(&cpu, &s32, None).unwrap();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 cuda-cpu sum parity",
        &f32_of(&cuda_sum),
        &round_via_bf16(&cpu_sum_f32),
    );

    let cuda_max = TypedOps::<bf16>::max(cuda_typed, &s, Some(1)).unwrap();
    let cpu_max_f32 = BackendOps::max(&cpu, &s32, Some(1)).unwrap();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 cuda-cpu max parity",
        &f32_of(&cuda_max),
        &round_via_bf16(&cpu_max_f32),
    );
}

/// 実機必須の形状網羅（受け入れ条件の本体。層 1／層 2 双方を実行する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn typed_ops_bf16_matches_across_shapes() {
    let cuda = CudaBackendOps::new(0);
    match TypedOps::<bf16>::gemm(
        &cuda,
        &small_int_bf16_tensor(1, &[2, 2]),
        &small_int_bf16_tensor(2, &[2, 2]),
    ) {
        Ok(_) => {}
        Err(BackendError::CudaUnavailable(_)) => {
            panic!("この test は #[ignore] で実機専用: CUDA 非搭載環境では実行しないこと");
        }
        Err(other) => panic!("unexpected error: {other}"),
    }
    run_layer1_structural_checks(&cuda);
    run_layer2_cross_backend_checks(&cuda);
}

/// エラー経路: shape 不整合が型付きエラーとして返ることを確認する
/// （`BackendOps::gemm_fp32_strict` と同じ次元検査を継承している構造的
/// 不変条件。実機不要）。
#[test]
fn typed_ops_bf16_gemm_shape_mismatch_returns_typed_error() {
    let cuda = CudaBackendOps::new(0);
    let a = small_int_bf16_tensor(1, &[2, 3]);
    let b = small_int_bf16_tensor(2, &[4, 2]); // k mismatch: 3 != 4
    match TypedOps::<bf16>::gemm(&cuda, &a, &b) {
        Err(BackendError::CudaUnavailable(_)) => {
            // CUDA 非搭載環境: device_handle 前に到達する場合とそうで
            // ない場合があるため、この分岐も許容する。
        }
        Err(BackendError::ShapeMismatch(_)) => {}
        other => panic!("expected ShapeMismatch or CudaUnavailable, got {other:?}"),
    }
}
