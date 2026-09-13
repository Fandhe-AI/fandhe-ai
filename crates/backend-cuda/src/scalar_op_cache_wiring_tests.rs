//! `context_cache::cached_scalar_unary_kernel`／`cached_scalar_binary_
//! kernel`（イシュー #1700。ペイロードあり kind の追加は #1702）が実際に
//! プロセス内キャッシュへ結線されていることを実機で検証する診断テスト。
//!
//! # なぜ `crates/backend-cuda/tests/`（integration test）ではなく本
//! ファイル（`lib.rs` 直下の兄弟モジュール）に置くか
//!
//! `context_cache`（非公開 `mod`）へアクセスするため、integration test
//! では到達できない。`module_cache_wiring_tests.rs`（イシュー #1024）と
//! 同じ理由でクレートルートの兄弟モジュールとして配置する。
//!
//! # 実行方法（実機。DGX Spark GB10 等 CUDA 搭載環境）
//!
//! ```text
//! cargo test -p fandhe-ai-backend-cuda --release --lib -- \
//!     --ignored --test-threads=1 scalar_op_cache
//! ```
//!
//! ローカル（libnvrtc 非搭載）では `context_cache::cached_device` が
//! `CudaError::DriverUnavailable`／`NvrtcUnavailable` 等を返すため、
//! `module_cache_wiring_tests.rs` と同じパターンで早期 return し誤って
//! fail しない。

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use fandhe_ai_tensor_core::{ScalarBinaryOp, ScalarUnaryOp};

/// `module_cache_wiring_tests.rs::is_environment_unavailable_error` と
/// 同じ許容エラー分岐（`DriverUnavailable`／`Driver`／`NvrtcUnavailable`）。
/// `true` を返した場合、呼び出し元は当該テストを即座に終了してよい
/// （CUDA 非搭載・libnvrtc 非搭載環境で誤って fail しないため）。
fn is_environment_unavailable_error(e: &CudaError) -> bool {
    matches!(
        e,
        CudaError::DriverUnavailable { .. }
            | CudaError::Driver(_)
            | CudaError::NvrtcUnavailable { .. }
    )
}

/// 同一 `context_cache::cached_device(0)` に対し `cached_scalar_unary_
/// kernel(&device, Sqrt)` を 2 回呼ぶと、2 回目は 1 回目と同一の
/// `Arc<CudaFunction>`（`Arc::ptr_eq`）を返す（2 回目が再コンパイルしない
/// ことの直接証拠。`cached_gemm`／`cached_elementwise` と同じ single-flight
/// eternal キャッシュ契約）。
#[test]
#[ignore = "実機（DGX Spark GB10 等の CUDA 搭載環境）専用。libnvrtc 必須"]
fn cached_scalar_unary_kernel_second_call_reuses_cache() {
    let device = match context_cache::cached_device(0) {
        Ok(dev) => dev,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from context_cache::cached_device: {e}"),
    };

    let first = context_cache::cached_scalar_unary_kernel(&device, ScalarUnaryOp::Sqrt)
        .expect("Sqrt is implemented and device is available")
        .expect("Sqrt must return Some(Arc<CudaFunction>)");
    let second = context_cache::cached_scalar_unary_kernel(&device, ScalarUnaryOp::Sqrt)
        .expect("2nd call must succeed given the 1st succeeded")
        .expect("Sqrt must return Some(Arc<CudaFunction>)");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "2nd cached_scalar_unary_kernel call must reuse the cached CudaFunction (no recompile)"
    );
}

/// [`cached_scalar_unary_kernel_second_call_reuses_cache`] の 2 項版
/// （`ScalarBinaryOp::Sub`）。
#[test]
#[ignore = "実機（DGX Spark GB10 等の CUDA 搭載環境）専用。libnvrtc 必須"]
fn cached_scalar_binary_kernel_second_call_reuses_cache() {
    let device = match context_cache::cached_device(0) {
        Ok(dev) => dev,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from context_cache::cached_device: {e}"),
    };

    let first = context_cache::cached_scalar_binary_kernel(&device, ScalarBinaryOp::Sub)
        .expect("Sub is implemented and device is available")
        .expect("Sub must return Some(Arc<CudaFunction>)");
    let second = context_cache::cached_scalar_binary_kernel(&device, ScalarBinaryOp::Sub)
        .expect("2nd call must succeed given the 1st succeeded")
        .expect("Sub must return Some(Arc<CudaFunction>)");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "2nd cached_scalar_binary_kernel call must reuse the cached CudaFunction (no recompile)"
    );
}

/// 未実装 kind（[`ScalarUnaryOp::Log`]）に対して `cached_scalar_unary_
/// kernel` が `Ok(None)`（`Err` ではない）を返すことを実機で確認する
/// （fail-closed 契約: 未実装 kind はキャッシュへ触れず呼び出し元へ
/// `None` を伝播し、`ops.rs::CudaBackendOps::scalar_unary` が
/// `BackendError::Unsupported` へ変換する。`kernels_scalar_op.rs` の
/// ホストのみユニットテストと同じ内容だが、こちらは実際の `CudaDevice`
/// を経由する結線を検証する点が異なる）。
#[test]
#[ignore = "実機（DGX Spark GB10 等の CUDA 搭載環境）専用。libnvrtc 必須"]
fn cached_scalar_unary_kernel_returns_none_for_unimplemented_kind() {
    let device = match context_cache::cached_device(0) {
        Ok(dev) => dev,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from context_cache::cached_device: {e}"),
    };

    let result = context_cache::cached_scalar_unary_kernel(&device, ScalarUnaryOp::Log)
        .expect("unimplemented kind must not error");
    assert!(
        result.is_none(),
        "unimplemented ScalarUnaryOp kind must return Ok(None), not Some(_)"
    );
}

/// 異なる `f32` payload（`Clamp{0.0,1.0}`／`Clamp{-5.0,5.0}`）で
/// `cached_scalar_unary_kernel` を呼んでも、NVRTC キャッシュキーが
/// `kind_name()` のみに依存し payload 値を含まない（`kernels_scalar_op.rs`
/// モジュール doc「スコープ」参照）ため、2 回目は同一 `Arc<CudaFunction>`
/// を返す（再コンパイルしないことの直接証拠。イシュー #1702）。
#[test]
#[ignore = "実機（DGX Spark GB10 等の CUDA 搭載環境）専用。libnvrtc 必須"]
fn cached_scalar_unary_kernel_clamp_payload_does_not_split_cache() {
    let device = match context_cache::cached_device(0) {
        Ok(dev) => dev,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from context_cache::cached_device: {e}"),
    };

    let first = context_cache::cached_scalar_unary_kernel(
        &device,
        ScalarUnaryOp::Clamp { min: 0.0, max: 1.0 },
    )
    .expect("Clamp is implemented and device is available")
    .expect("Clamp must return Some(Arc<CudaFunction>)");
    let second = context_cache::cached_scalar_unary_kernel(
        &device,
        ScalarUnaryOp::Clamp {
            min: -5.0,
            max: 5.0,
        },
    )
    .expect("2nd call (different payload) must succeed given the 1st succeeded")
    .expect("Clamp must return Some(Arc<CudaFunction>)");
    assert!(
        std::sync::Arc::ptr_eq(&first, &second),
        "different Clamp payload values must not split the NVRTC cache entry \
         (cache key is kind_name()-only, payload-independent)"
    );
}

/// 異なる `CudaDevice`（`context_cache::cached_device` を経由しない
/// `CudaDevice::new` の直接呼び出し。`ContextKey` が異なる）から呼んだ
/// 場合はキャッシュがヒットせず別エントリになる（`cached_gemm`／
/// `cached_elementwise` の既存挙動と同じであることの確認。
/// `module_cache_wiring_tests.rs` 冒頭コメント「ABA 耐性」節と同じ、
/// ctx 単位のキー分離）。
#[test]
#[ignore = "実機（DGX Spark GB10 等の CUDA 搭載環境）専用。libnvrtc 必須"]
fn cached_scalar_unary_kernel_distinct_devices_do_not_share_cache() {
    let device_a = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from CudaDevice::new: {e}"),
    };
    let device_b =
        CudaDevice::new(0).expect("2nd CudaDevice::new must succeed given the 1st succeeded");

    let func_a = context_cache::cached_scalar_unary_kernel(&device_a, ScalarUnaryOp::Sqrt)
        .expect("Sqrt is implemented")
        .expect("Sqrt must return Some(Arc<CudaFunction>)");
    let func_b = context_cache::cached_scalar_unary_kernel(&device_b, ScalarUnaryOp::Sqrt)
        .expect("Sqrt is implemented")
        .expect("Sqrt must return Some(Arc<CudaFunction>)");
    assert!(
        !std::sync::Arc::ptr_eq(&func_a, &func_b),
        "distinct CudaContext instances must not share the scalar-op kernel cache entry"
    );
}
