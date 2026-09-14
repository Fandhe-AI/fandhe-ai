//! Metal `TypedOps<half::bf16>`（`crate::typed_bf16`。イシュー #1706）の
//! 数値一致回帰テスト。CUDA 側 `backend-cuda/tests/typed_ops_bf16_parity.rs`
//! （#1704）の Metal 対応版。
//!
//! `docs/backend-dtype-dispatch-design.md` §6 の判定契約（f16／bf16 出力は
//! 参照値側も出力 dtype と同じ丸めを経てから比較する）を踏まえ、以下の
//! 二層構成で検証する:
//!
//! - **層 1（同一バックエンド内の構造的不変条件・bit 完全一致）**:
//!   `TypedOps::<bf16>::op(x) == bf16::from_f32(BackendOps::op_f32(f32(x)))`
//!   を `gemm`（`gemm_fp32_strict` と比較）／`add`／`mul`／`relu`／`exp`／
//!   `tanh` の 6 演算について検証する。`sum`／`max` は Metal f32
//!   `BackendOps` が GPU カーネル未実装で常に `Unsupported` を返すため
//!   （`typed_bf16.rs` モジュール doc「`sum`／`max` は `Unsupported` を
//!   そのまま伝播する」参照）、bit 一致検証の対象外とし別テストで
//!   `Unsupported` 伝播のみを確認する
//! - **層 2（CPU `BackendOps` を bf16 丸めした値とのクロスバックエンド
//!   判定）**: 丸め境界またぎによる誤判定を避けるため、入力を bf16 で
//!   正確に表現できる小整数・小 K（累算順序に依存しない）に限定する
//!   （`gemm`／`add`／`mul`／`relu`）。`exp`／`tanh` はこの技法が使えない
//!   ため層 2 の対象外とし層 1 で検証する
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`scalar_op_parity.rs`〈#1707/#1708〉・`linear_forward_device_parity.rs`
//! と同方針。`#![cfg(target_os = "macos")]` により Linux CI ではコンパイル
//! 対象外になり、`#[ignore]` により通常の `cargo test` からも除外される。
//! Metal は Linux で `MetalBackendOps` が存在しないため CUDA 側のような
//! 環境適応スモークテストは持てない）。
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
//! cargo test -p fandhe-ai-backend-metal --release --test typed_ops_bf16_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::device::BackendError;
use fandhe_ai_tensor_core::{BackendOps, Tensor, TypedOps};
use half::bf16;

/// 決定的シードで `[-8, 8]` の整数値のみからなる bf16 テンソルを生成する。
///
/// bf16（仮数部 7bit）は `[-8, 8]` の整数を正確に表現でき、かつこの
/// レンジの GEMM 積・総和は小 K のもとで f32 中間値も丸め誤差なく正確に
/// 表現できる（累算順序への非依存性が構造的に成立する）。層 2 の
/// クロスバックエンド判定を丸め境界またぎの誤判定なしで成立させるための
/// 入力生成ヘルパー（CUDA 側 `typed_ops_bf16_parity.rs` と同一方針）。
fn small_int_bf16_tensor(seed: u64, shape: &[usize]) -> Tensor<bf16> {
    let numel: usize = shape.iter().product();
    let mut rng = Xorshift64Star::new(seed);
    let raw = rng.fill_vec(numel);
    let data: Vec<bf16> = raw
        .iter()
        .map(|&v| {
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

/// 層 1: 同一バックエンド内の構造的不変条件（bit 完全一致）。
fn run_layer1_structural_checks(metal: &MetalBackendOps) {
    let typed: &dyn TypedOps<bf16> = metal;

    let a = small_int_bf16_tensor(101, &[3, 4]);
    let b = small_int_bf16_tensor(102, &[4, 5]);

    // gemm: TypedOps::<bf16>::gemm と gemm_fp32_strict(f32(a), f32(b)) を
    // bf16 へ丸めた結果が bit 完全一致すること。
    let typed_gemm = TypedOps::<bf16>::gemm(typed, &a, &b).expect("metal typed gemm succeeds");
    let a32 = to_f32_tensor(&a);
    let b32 = to_f32_tensor(&b);
    let ref32 = BackendOps::gemm_fp32_strict(metal, &a32, &b32).expect("metal f32 gemm succeeds");
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
    let ref_add = BackendOps::add(metal, &x32, &y32).unwrap();
    assert_eq!(
        f32_of(&typed_add),
        round_via_bf16(&ref_add),
        "add bit mismatch"
    );

    let typed_mul = TypedOps::<bf16>::mul(typed, &x, &y).unwrap();
    let ref_mul = BackendOps::mul(metal, &x32, &y32).unwrap();
    assert_eq!(
        f32_of(&typed_mul),
        round_via_bf16(&ref_mul),
        "mul bit mismatch"
    );

    let typed_relu = TypedOps::<bf16>::relu(typed, &x).unwrap();
    let ref_relu = BackendOps::relu(metal, &x32).unwrap();
    assert_eq!(
        f32_of(&typed_relu),
        round_via_bf16(&ref_relu),
        "relu bit mismatch"
    );

    // exp/tanh: 同一バックエンド内比較のため丸め境界またぎの問題がなく
    // 検証できる（層 2 では小整数入力への限定という技法が使えず対象外
    // としているが、層 1 はこの限定を必要としない）。
    let typed_exp = TypedOps::<bf16>::exp(typed, &x).unwrap();
    let ref_exp = BackendOps::exp(metal, &x32).unwrap();
    assert_eq!(
        f32_of(&typed_exp),
        round_via_bf16(&ref_exp),
        "exp bit mismatch"
    );

    let typed_tanh = TypedOps::<bf16>::tanh(typed, &x).unwrap();
    let ref_tanh = BackendOps::tanh(metal, &x32).unwrap();
    assert_eq!(
        f32_of(&typed_tanh),
        round_via_bf16(&ref_tanh),
        "tanh bit mismatch"
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
fn run_layer2_cross_backend_checks(metal: &MetalBackendOps) {
    let cpu = CpuBackendOps::new();
    let metal_typed: &dyn TypedOps<bf16> = metal;

    let a = small_int_bf16_tensor(201, &[8, 16]);
    let b = small_int_bf16_tensor(202, &[16, 8]);
    let a32 = to_f32_tensor(&a);
    let b32 = to_f32_tensor(&b);

    // gemm: 小整数・K=16（bf16 で正確に表現可能な整数域）。
    let metal_gemm = TypedOps::<bf16>::gemm(metal_typed, &a, &b).expect("metal gemm succeeds");
    let cpu_gemm_f32 = BackendOps::gemm(&cpu, &a32, &b32).expect("cpu gemm succeeds");
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 metal-cpu gemm parity",
        &f32_of(&metal_gemm),
        &round_via_bf16(&cpu_gemm_f32),
    );

    let x = small_int_bf16_tensor(203, &[10]);
    let y = small_int_bf16_tensor(204, &[10]);
    let x32 = to_f32_tensor(&x);
    let y32 = to_f32_tensor(&y);

    let metal_add = TypedOps::<bf16>::add(metal_typed, &x, &y).unwrap();
    let cpu_add_f32 = BackendOps::add(&cpu, &x32, &y32).unwrap();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 metal-cpu add parity",
        &f32_of(&metal_add),
        &round_via_bf16(&cpu_add_f32),
    );

    let metal_mul = TypedOps::<bf16>::mul(metal_typed, &x, &y).unwrap();
    let cpu_mul_f32 = BackendOps::mul(&cpu, &x32, &y32).unwrap();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 metal-cpu mul parity",
        &f32_of(&metal_mul),
        &round_via_bf16(&cpu_mul_f32),
    );

    let metal_relu = TypedOps::<bf16>::relu(metal_typed, &x).unwrap();
    let cpu_relu_f32 = BackendOps::relu(&cpu, &x32).unwrap();
    fandhe_ai_backend_cpu::parity::assert_parity(
        "typed_ops_bf16 metal-cpu relu parity",
        &f32_of(&metal_relu),
        &round_via_bf16(&cpu_relu_f32),
    );
}

/// accessor が常に `Some` を返すこと（実機不要。`ops.rs::typed_ops_bf16`
/// が ZST 経由の無条件 `Some(self)` を返す設計であることの回帰確認）。
#[test]
fn typed_ops_bf16_accessor_returns_some_without_device_init() {
    let metal = MetalBackendOps::new();
    let ops: &dyn BackendOps = &metal;
    assert!(
        ops.typed_ops_bf16().is_some(),
        "typed_ops_bf16 accessor must always return Some for MetalBackendOps"
    );
}

/// `sum`／`max` が実機不要で `Unsupported` を返すこと（Metal f32
/// `BackendOps::sum`／`max` が GPU カーネル未実装のため。
/// `typed_bf16.rs` の同名ユニットテストと同じ確認だが、統合テスト側にも
/// 固定して回帰対象を明示する）。
#[test]
fn sum_and_max_remain_unsupported_without_device_init() {
    let metal = MetalBackendOps::new();
    let a = small_int_bf16_tensor(1, &[1, 2]);

    assert!(matches!(
        TypedOps::<bf16>::sum(&metal, &a, None),
        Err(BackendError::Unsupported(_))
    ));
    assert!(matches!(
        TypedOps::<bf16>::max(&metal, &a, None),
        Err(BackendError::Unsupported(_))
    ));
}

/// 実機必須の形状網羅（受け入れ条件の本体。層 1／層 2 双方を実行する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn typed_ops_bf16_matches_across_shapes() {
    let metal = MetalBackendOps::new();
    run_layer1_structural_checks(&metal);
    run_layer2_cross_backend_checks(&metal);
}

/// エラー経路: shape 不整合が型付きエラーとして返ることを確認する
/// （`BackendOps::gemm_fp32_strict` と同じ次元検査を継承している構造的
/// 不変条件）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn typed_ops_bf16_gemm_shape_mismatch_returns_typed_error() {
    let metal = MetalBackendOps::new();
    let a = small_int_bf16_tensor(1, &[2, 3]);
    let b = small_int_bf16_tensor(2, &[4, 2]); // k mismatch: 3 != 4
    match TypedOps::<bf16>::gemm(&metal, &a, &b) {
        Err(BackendError::ShapeMismatch(_)) => {}
        other => panic!("expected ShapeMismatch, got {other:?}"),
    }
}
