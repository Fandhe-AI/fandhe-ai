//! composition root の結線検証（受入基準 1）。
//!
//! `fandhe_ai::tape()`（既定 CPU）・`fandhe_ai::tape_for(Device)`（明示指定）の
//! 両経路で構築した `Tape` 上で forward → backward が成立することを固定
//! する。CUDA 経路は実行環境適応型（driver 有無・選択可能デバイス数
//! いずれの組み合わせでも green）とし、CUDA 実機テストの `#[ignore]`
//! 分離方針（`.claude/rules/coding-rust.md`）とは別に、
//! `fandhe_ai_backend_cuda::CudaDeviceProvider::is_available()`（`enumerate` と
//! 同じ探索・除外ロジックを通し、選択可能デバイスが 1 件以上あることを
//! 条件にする `.claude/rules/coding-rust.md` 前提の強い判定。
//! `crates/backend-cuda/src/device.rs` の `is_available` doc 参照）で
//! 分岐する fail-safe な検証にとどめる（実機必須の性能計測等は含まない
//! ため）。`CudaDevice::is_available()`（`libcuda` の有無のみを見る弱い
//! 判定）は使わない: driver はあるが選択可能な GPU が 0 台のホストでは
//! 弱い判定が `true` を返す一方 `fandhe_ai::tape_for` は
//! `CudaDeviceProvider::select` 経由で `Err` を返すため、弱い判定に
//! 依拠すると本テストの `Ok` 分岐が誤って `panic` する（PR #423 Bugbot
//! 指摘 `385da921-0618-4dfb-8cfc-4ea9bb59411e`）。

use fandhe_ai::Device;
use fandhe_ai_backend_cuda::CudaDeviceProvider;
use fandhe_ai_tensor_core::Tensor;
use fandhe_ai_tensor_core::device::DeviceProvider;

fn sample_tensor() -> Tensor<f32> {
    Tensor::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("sample tensor は shape が一致する")
}

/// `fandhe_ai::tape()`（既定 CPU）上で forward・backward が成立する。
#[test]
fn default_tape_forward_backward_succeeds() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&sample_tensor());
    let b = tape.var(&sample_tensor());
    let sum = a.add(&b).expect("add は shape 一致で成功する");
    let loss = sum.sum(None).expect("sum は成功する");

    let grads = tape.backward(&loss).expect("backward は成功する");
    assert!(
        grads.get(&a).expect("tape 一致").is_some(),
        "a への勾配が伝播しているはず"
    );
}

/// `fandhe_ai::tape_for(Device::Cpu)` は常に `Ok` を返し、既定 `tape()` と
/// 同様に forward・backward が成立する。
#[test]
fn tape_for_cpu_succeeds_and_matches_default_semantics() {
    let tape = fandhe_ai::tape_for(Device::Cpu).expect("CPU は常に利用可能");
    let a = tape.var(&sample_tensor());
    let b = tape.var(&sample_tensor());
    let product = a.mul(&b).expect("mul は shape 一致で成功する");
    let loss = product.sum(None).expect("sum は成功する");

    let grads = tape.backward(&loss).expect("backward は成功する");
    assert!(grads.get(&a).expect("tape 一致").is_some());
}

/// `fandhe_ai::tape_for(Device::Cuda(0))`: 実行環境適応型。
/// `fandhe_ai_backend_cuda::CudaDeviceProvider::is_available()`（`enumerate` と
/// 同じロジックで選択可能デバイスの有無まで見る強い判定。モジュール
/// 冒頭コメント参照）が `true` の環境（CUDA driver 搭載・選択可能な
/// GPU が 1 台以上存在）では `Ok` を返すこと、`false` の環境（driver
/// 非搭載、または driver はあるが選択可能デバイスが 0 台。CI
/// self-hosted の既定）では panic せず
/// `Err(BackendError::CudaUnavailable(_))` を返すことを検証する（CI
/// self-hosted runner の CUDA 有無どちらでも green。イシュー #410
/// 実装計画 §3「tests/tape_construction.rs」）。
///
/// **意図的に forward 実行までは検証しない**: driver（`libcuda`）検出と
/// NVRTC（`libnvrtc`。カーネルコンパイルに必要）搭載は独立した環境
/// 前提であり、「driver は検出できるが NVRTC toolkit は未搭載」の
/// 環境が実在する（このためのフォールバックが `cudarc`
/// 動的ロード方式・`build-no-cuda-toolkit` CI ジョブの存在理由でもある。
/// `.claude/rules/ci.md`）。composition root（本クレート）の責務は
/// `Device` → `BackendOps` の**結線**であり、GPU カーネルの実行可否は
/// `backend-cuda` クレート自身の実機テストが担う（本テストのスコープ
/// 外）。
#[test]
fn tape_for_cuda_adapts_to_runtime_availability() {
    let result = fandhe_ai::tape_for(Device::Cuda(0));
    if CudaDeviceProvider::new().is_available() {
        let _tape = result.expect("CUDA driver 搭載環境では結線が成功するはず");
    } else {
        let err = result.expect_err("CUDA driver 非搭載環境では失敗するはず");
        assert!(
            matches!(
                err,
                fandhe_ai_tensor_core::BackendError::CudaUnavailable(_)
                    | fandhe_ai_tensor_core::BackendError::DeviceUnavailable(_)
            ),
            "CUDA 不在時のエラーは CudaUnavailable/DeviceUnavailable のいずれかのはず: {err:?}"
        );
    }
}

/// 範囲外 ordinal（`usize::MAX`）は `Err` になる（fail-fast。
/// panic しないことを固定する）。
#[test]
fn tape_for_cuda_out_of_range_ordinal_returns_err() {
    let result = fandhe_ai::tape_for(Device::Cuda(usize::MAX));
    assert!(
        result.is_err(),
        "範囲外 ordinal（usize::MAX）は Err を返すはず"
    );
}
