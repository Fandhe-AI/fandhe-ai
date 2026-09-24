//! `fandhe_ai_autodiff::f64_autograd`（イシュー #2195・親 #2142）の
//! バックエンド別 parity テスト（`nn_module_freeze_backend_parity.rs`・
//! `no_grad_detach_backend_parity.rs` と同型）。
//!
//! **facade 公開面（意図的な非変更）**: `f64_autograd` は facade へ
//! 再エクスポートされていない（`docs/autodiff-var-dtype-multiplexing-
//! design.md` §10 承認事項はいずれも未承認のまま）。本テストは内部
//! クレート `fandhe_ai_autodiff::f64_autograd`・具体バックエンドクレート
//! （`fandhe_ai_backend_cpu`／`fandhe_ai_backend_cuda`／
//! `fandhe_ai_backend_metal`）を facade の依存経由で直接使う
//! （`RawTape` 構成は `nn_module_freeze_backend_parity.rs` と同じ）ため、
//! facade の公開面（`api_surface.rs` の否定ガード）には影響しない。
//!
//! - 属性なし（CPU）: `RawTape::new_with_ops(Box::new(CpuBackendOps::
//!   new()))`（`typed_ops_f64` がネイティブ実装を返す。#1697）上の
//!   `add`／`mul`（ネイティブ経路）・`div`／`pow`（常にホスト経路）の
//!   forward・勾配が、`RawTape::new()`（`NaiveOps`。`typed_ops_f64` は
//!   既定 `None`。全演算ホスト経路）と **bit 完全一致**することを
//!   確認する。`add`／`mul` は算術結合順序が異なる別実装（CPU ネイティブ
//!   vs ホスト参照）のため一般には bit 一致が保証されないが、本テストの
//!   固定 shape（ブロードキャストなし・小要素数）では両実装とも同一の
//!   単純な逐次加算／乗算に帰着するため bit 完全一致する
//!   （`.claude/rules/coding-rust.md` の複合判定〈相対誤差／絶対誤差〉は
//!   本テストの対象外——本テストは「ネイティブ経路とホスト経路の bit
//!   一致」という、それより厳しい契約を検証する）。
//! - `#[ignore]`: `CudaBackendOps::new(0)`（`typed_ops_f64` がネイティブ
//!   実装を返す。#2060）／`MetalBackendOps::new()`（`cfg(target_os =
//!   "macos")` 限定。`typed_ops_f64` は MSL `double` 非対応のため常に
//!   `None`）の同経路。実機実測は本エージェント実行環境に到達手段が
//!   無いため未実施のまま（`docs/autodiff-var-dtype-multiplexing-
//!   design.md` §13 へ申し送り）。

use fandhe_ai_autodiff::Tape as RawTape;
use fandhe_ai_autodiff::f64_autograd::TapeF64;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

fn t(data: Vec<f64>, shape: &[usize]) -> Tensor<f64> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn bits(t: &Tensor<f64>) -> Vec<u64> {
    t.contiguous()
        .host_slice()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

/// `add`／`mul`（ネイティブ経路の有無で分岐しうる）・`div`／`pow`
/// （常にホスト経路）の forward・1 step backward が、`ops_factory` が
/// 構築するバックエンドと `NaiveOps`（ホスト経路固定）とで bit 完全
/// 一致することを検証する共通処理。
fn assert_f64_autograd_parity_with(ops_factory: impl Fn() -> Box<dyn BackendOps + Send>) {
    let native_tape = RawTape::new_with_ops(ops_factory());
    let host_tape = RawTape::new();

    let native_graph = TapeF64::new(&native_tape);
    let host_graph = TapeF64::new(&host_tape);

    let a_data = vec![2.0, 3.0, -1.5, 4.25];
    let b_data = vec![1.5, -2.0, 0.5, 2.0];

    let a_native = native_graph.var(&t(a_data.clone(), &[4]));
    let b_native = native_graph.var(&t(b_data.clone(), &[4]));
    let a_host = host_graph.var(&t(a_data, &[4]));
    let b_host = host_graph.var(&t(b_data, &[4]));

    // y = a*b + a/b + a^2（4 演算すべてを 1 グラフに含める。`pow` は
    // 境界〈a==0 等〉に触れないよう固定指数 2.0 を独立の変数として渡す）。
    let exponent_native = native_graph.var(&t(vec![2.0, 2.0, 2.0, 2.0], &[4]));
    let exponent_host = host_graph.var(&t(vec![2.0, 2.0, 2.0, 2.0], &[4]));

    let native_y = a_native
        .mul(&b_native)
        .unwrap()
        .add(&a_native.div(&b_native).unwrap())
        .unwrap()
        .add(&a_native.pow(&exponent_native).unwrap())
        .unwrap();
    let host_y = a_host
        .mul(&b_host)
        .unwrap()
        .add(&a_host.div(&b_host).unwrap())
        .unwrap()
        .add(&a_host.pow(&exponent_host).unwrap())
        .unwrap();

    assert_eq!(
        bits(&native_y.value()),
        bits(&host_y.value()),
        "forward 値がネイティブ経路とホスト経路で bit 一致しない"
    );

    let native_grads = native_graph.backward(&native_y).unwrap();
    let host_grads = host_graph.backward(&host_y).unwrap();

    let da_native = native_grads.get(&a_native).unwrap().unwrap();
    let da_host = host_grads.get(&a_host).unwrap().unwrap();
    assert_eq!(
        bits(da_native),
        bits(da_host),
        "a の勾配がネイティブ経路とホスト経路で bit 一致しない"
    );

    let db_native = native_grads.get(&b_native).unwrap().unwrap();
    let db_host = host_grads.get(&b_host).unwrap().unwrap();
    assert_eq!(
        bits(db_native),
        bits(db_host),
        "b の勾配がネイティブ経路とホスト経路で bit 一致しない"
    );
}

/// CPU 本番 ops（`CpuBackendOps`）上での f64 autograd parity（CI で実行）。
#[test]
fn cpu_f64_autograd_forward_and_grad_match_host_reference() {
    assert_f64_autograd_parity_with(|| Box::new(CpuBackendOps::new()));
}

// --- 実機横断（`#[ignore]`。CUDA／Metal） ---
//
// `nn_module_freeze_backend_parity.rs` と同じ理由で Metal 依存テストのみ
// `cfg(target_os = "macos")` でコンパイル自体を限定する。実機実測は
// 本エージェント実行環境に到達手段が無いため未実施のまま（PR 本文・
// `docs/autodiff-var-dtype-multiplexing-design.md` §13 へ申し送り）。

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_f64_autograd_forward_and_grad_match_host_reference() {
    assert_f64_autograd_parity_with(|| Box::new(fandhe_ai_backend_cuda::CudaBackendOps::new(0)));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_f64_autograd_forward_and_grad_match_host_reference() {
    // Metal の `typed_ops_f64` は MSL `double` 非対応のため常に `None`
    // を返し、常にホスト経路へ到達する（本関数はそのことを含めて
    // `NaiveOps` ベースの `host_graph` と一致することを確認する）。
    assert_f64_autograd_parity_with(|| Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()));
}
