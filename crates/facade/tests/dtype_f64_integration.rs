//! `fandhe_ai_autodiff::f64_autograd`（イシュー #2196・親 #2142）の
//! `matmul`・`sum`・`max` を facade 経由で到達できることを確認する
//! テスト（`var_f64_autograd_backend_parity.rs`〈イシュー #2195〉と同型。
//! 本ファイルは #2195 で導入済みの add／mul／div／pow 経路には触れず、
//! #2196 で追加した 4 演算に限定する）。
//!
//! **facade 公開面（意図的な非変更）**: `f64_autograd`（`TapeF64`／
//! `VarF64`）は facade へ再エクスポートされていない
//! （`docs/autodiff-var-dtype-multiplexing-design.md` §10 承認事項は
//! いずれも未承認のまま）。「facade 到達」とは、既存の公開 accessor
//! `fandhe_ai::tape()` → `Tape::typed_ops_f64()`（`TypedOps<f64>` の
//! capability accessor。#2195 で追加済み・本イシューで変更しない）
//! 経由で f64 の `gemm`／`sum`／`max` を計算できることを指す。
//! `facade::Tape` は内部フィールドが `pub(crate)` の newtype のため、
//! そこから `TapeF64` を構築することはできない——本テストは facade
//! accessor が返す `&dyn TypedOps<f64>` を直接使う経路（①）と、
//! `fandhe_ai_autodiff::f64_autograd::TapeF64`（内部クレート・
//! `fandhe_ai_backend_cpu::CpuBackendOps` を facade の依存経由で直接
//! 使う。`var_f64_autograd_backend_parity.rs` と同じ構成）を使う経路
//! （②）の両方が bit 完全一致することで、facade accessor が
//! `TapeF64`／`VarF64` の内部 dispatch（`native_or_host_gemm` 等）と
//! 同一のネイティブ実装へ到達していることを検証する。`facade` の
//! `src`／公開面は変更しないため、`api_surface.rs` は無改変で green の
//! はずである。

use fandhe_ai_autodiff::Tape as RawTape;
use fandhe_ai_autodiff::f64_autograd::TapeF64;
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_tensor_core::Tensor;

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

/// facade accessor（①）が CPU では `Some` を返し、`gemm`／
/// `sum(None)`／`sum(Some(1))`／`max(Some(0))` を直接計算できることを
/// 確認する（受け入れ条件 A5 の中核）。
#[test]
fn facade_typed_ops_f64_accessor_is_some_on_cpu_and_computes_gemm_sum_max() {
    let tape = fandhe_ai::tape();
    let ops = tape
        .typed_ops_f64()
        .expect("CPU は typed_ops_f64() が常に Some を返すはず（#1697）");

    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
    let c = ops
        .gemm(&a, &b)
        .expect("gemm はネイティブ経路で成功するはず");
    assert_eq!(c.host_slice().into_owned(), vec![4.0, 5.0, 10.0, 11.0]);

    let sum_all = ops.sum(&a, None).expect("sum(None)");
    assert_eq!(sum_all.host_slice().into_owned(), vec![21.0]);

    let sum_axis1 = ops.sum(&a, Some(1)).expect("sum(Some(1))");
    assert_eq!(sum_axis1.host_slice().into_owned(), vec![6.0, 15.0]);

    let max_axis0 = ops.max(&a, Some(0)).expect("max(Some(0))");
    assert_eq!(max_axis0.host_slice().into_owned(), vec![4.0, 5.0, 6.0]);
}

/// facade accessor（①）経由の `gemm`／`sum`／`max` の値が、`TapeF64`
/// （②。`CpuBackendOps` を直接結線）の forward 値と bit 完全一致する
/// ことを確認する。
#[test]
fn facade_typed_ops_f64_matches_tape_f64_native_forward() {
    let facade_tape = fandhe_ai::tape();
    let ops = facade_tape
        .typed_ops_f64()
        .expect("CPU は typed_ops_f64() が常に Some を返すはず");

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let a = t(a_data.clone(), &[2, 3]);
    let b = t(b_data.clone(), &[3, 2]);

    let gemm_via_facade = ops.gemm(&a, &b).expect("gemm");
    let sum_via_facade = ops.sum(&a, Some(1)).expect("sum");
    let max_via_facade = ops.max(&a, Some(0)).expect("max");

    let raw_tape = RawTape::new_with_ops(Box::new(CpuBackendOps::new()));
    let graph = TapeF64::new(&raw_tape);
    let av = graph.var(&t(a_data, &[2, 3]));
    let bv = graph.var(&t(b_data, &[3, 2]));
    let gemm_via_tapef64 = av.matmul(&bv).expect("matmul");
    let sum_via_tapef64 = av.sum(Some(1)).expect("sum");
    let max_via_tapef64 = av.max(Some(0)).expect("max");

    assert_eq!(
        bits(&gemm_via_facade),
        bits(&gemm_via_tapef64.value()),
        "gemm が facade accessor 経由と TapeF64 経由で bit 一致しない"
    );
    assert_eq!(
        bits(&sum_via_facade),
        bits(&sum_via_tapef64.value()),
        "sum が facade accessor 経由と TapeF64 経由で bit 一致しない"
    );
    assert_eq!(
        bits(&max_via_facade),
        bits(&max_via_tapef64.value()),
        "max が facade accessor 経由と TapeF64 経由で bit 一致しない"
    );
}

/// CPU ネイティブ（`CpuBackendOps`）対ホスト（`NaiveOps`）の複合
/// backward が bit 完全一致することを確認する（`var_f64_autograd_
/// backend_parity.rs` の add/mul/div/pow 版と対になる #2196 版）。
/// 全軸 `sum` を含む経路はモジュール doc「bit 一致の境界」（CHUNK=4096
/// 以下）に従い、固定入力の要素数を 4096 以下に保つ。
#[test]
fn cpu_native_matmul_sum_mean_max_backward_matches_host_reference() {
    let native_tape = RawTape::new_with_ops(Box::new(CpuBackendOps::new()));
    let host_tape = RawTape::new();

    let native_graph = TapeF64::new(&native_tape);
    let host_graph = TapeF64::new(&host_tape);

    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];

    let a_native = native_graph.var(&t(a_data.clone(), &[2, 3]));
    let b_native = native_graph.var(&t(b_data.clone(), &[3, 2]));
    let a_host = host_graph.var(&t(a_data, &[2, 3]));
    let b_host = host_graph.var(&t(b_data, &[3, 2]));

    // loss = sum(matmul(a,b)) + mean(max(a, dim=1))
    let native_c = a_native.matmul(&b_native).unwrap();
    let native_loss = native_c
        .sum(None)
        .unwrap()
        .add(&a_native.max(Some(1)).unwrap().mean(None).unwrap())
        .unwrap();
    let host_c = a_host.matmul(&b_host).unwrap();
    let host_loss = host_c
        .sum(None)
        .unwrap()
        .add(&a_host.max(Some(1)).unwrap().mean(None).unwrap())
        .unwrap();

    assert_eq!(
        bits(&native_loss.value()),
        bits(&host_loss.value()),
        "forward 値がネイティブ経路とホスト経路で bit 一致しない"
    );

    let native_grads = native_graph.backward(&native_loss).unwrap();
    let host_grads = host_graph.backward(&host_loss).unwrap();

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

// --- 実機横断（`#[ignore]`。CUDA／Metal） ---
//
// `var_f64_autograd_backend_parity.rs` と同じ理由で Metal 依存テストの
// みコンパイル自体を `cfg(target_os = "macos")` で限定する。実機実測は
// 本エージェント実行環境に到達手段が無いため未実施のまま
// （`docs/perf/logs/var-f64-gemm-reduction-2196/README.md` へ申し送る）。

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_typed_ops_f64_accessor_is_some_and_gemm_matches_parity() {
    let tape = fandhe_ai::tape_for(fandhe_ai::Device::Cuda(0))
        .expect("CUDA デバイス解決（実機のみ成功する）");
    let ops = tape
        .typed_ops_f64()
        .expect("CUDA は typed_ops_f64() が Some を返すはず（#2060）");
    let a = t(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = t(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
    let c = ops.gemm(&a, &b).expect("gemm");
    // 全軸 sum を含む経路は GPU の 2 段木縮約と結合順序が異なりうる
    // ため、REQ-2 統一複合判定（`assert_parity_f64`。定数は無変更）で
    // 比較する（`.claude/rules/coding-rust.md`「結合順序が単一の連続
    // K ループと異なるカーネルの parity テスト判定方式」）。
    let expected = [4.0f64, 5.0, 10.0, 11.0];
    fandhe_ai_backend_cpu::assert_parity_f64("cuda_f64_gemm_parity", &c.host_slice(), &expected);
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_typed_ops_f64_accessor_is_none_and_falls_back_to_host() {
    // Metal の `typed_ops_f64` は MSL `double` 非対応のため常に `None`
    // を返し、`TapeF64::matmul`／`sum`／`max` は常にホスト経路へ到達
    // する（本関数はそのことを含めて `NaiveOps` ベースの結果と一致
    // することを確認する）。
    let metal_tape =
        RawTape::new_with_ops(Box::new(fandhe_ai_backend_metal::MetalBackendOps::new()));
    assert!(
        fandhe_ai_tensor_core::BackendOps::typed_ops_f64(&*Box::new(
            fandhe_ai_backend_metal::MetalBackendOps::new()
        ))
        .is_none(),
        "Metal は typed_ops_f64() が常に None を返すはず"
    );
    let host_tape = RawTape::new();
    let metal_graph = TapeF64::new(&metal_tape);
    let host_graph = TapeF64::new(&host_tape);
    let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let a_metal = metal_graph.var(&t(a_data.clone(), &[2, 3]));
    let b_metal = metal_graph.var(&t(b_data.clone(), &[3, 2]));
    let a_host = host_graph.var(&t(a_data, &[2, 3]));
    let b_host = host_graph.var(&t(b_data, &[3, 2]));
    let metal_c = a_metal.matmul(&b_metal).unwrap();
    let host_c = a_host.matmul(&b_host).unwrap();
    assert_eq!(bits(&metal_c.value()), bits(&host_c.value()));
}
