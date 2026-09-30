//! イシュー #2123（親 #2121）: GB10 の unified memory 関連デバイス属性を 1 行 1 属性の
//! `key=value` で出力するプローブ。
//!
//! ## 役割
//!
//! `docs/gb10-unified-memory-grace-cpu-consideration.md` §3 の prefetch／advise 契約と
//! pageable 直接アクセスの検討余地を、GB10 実機の属性値で分岐させるための材料を出す。
//! 推定値で埋めず、実機の出力を `docs/perf/logs/gb10-unified-memory-grace-2123/` へ
//! 回収する（判定規則は同ディレクトリの `RULE.txt`）。
//!
//! ## `device_attributes_dump` と分けた理由
//!
//! `device_attributes_dump`（#482）は SMEM／L2 帯域のコストモデル用で、別イシュー（#2122）が
//! 編集しうる。並列イシュー間の競合を避けるため本プローブは別ファイルとした。
//!
//! ## 前提・限界
//!
//! - `CudaDevice::context().attribute()`（safe API）のみを使い、新規 `unsafe`・依存はない。
//! - CUDA ドライバ不在（`DriverUnavailable`）のみ非 CUDA 環境としてスキップ（終了コード 0）する。
//!   それ以外の `CudaDevice::new` 失敗は計測失敗として stderr に出力し終了コード 1 で終える。
//! - 開発機（非 GB10）での出力は GB10 の値の代替にならない。
//! - 属性取得に失敗した項目は `key=error` として出力を継続する（`unwrap`/`expect` なし）。
//!
//! ## 実行
//!
//! ```sh
//! cargo run -p fandhe-ai-backend-cuda --release --features internal-diagnostics --example unified_memory_probe
//! ```

use cudarc::driver::sys::CUdevice_attribute as A;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError};

/// 属性を 1 行 `key=value`（失敗時 `key=error`）で出力する。
fn print_attr(device: &CudaDevice, key: &str, attr: A) {
    match device.context().attribute(attr) {
        Ok(v) => println!("{key}={v}"),
        Err(_) => println!("{key}=error"),
    }
}

fn main() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            println!("unified_memory_probe: CUDA driver unavailable ({detail}); skipping.");
            return;
        }
        Err(other) => {
            // ドライバ不在以外の失敗（デバイス初期化不能等）は GB10 計測の失敗であり、
            // 非 CUDA 環境のスキップと区別するため stderr 出力＋非 0 終了とする。
            eprintln!(
                "unified_memory_probe: CudaDevice::new failed ({other}); measurement failed."
            );
            std::process::exit(1);
        }
    };
    println!("device_name={}", device.name());
    print_attr(
        &device,
        "MANAGED_MEMORY",
        A::CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY,
    );
    print_attr(
        &device,
        "CONCURRENT_MANAGED_ACCESS",
        A::CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS,
    );
    print_attr(
        &device,
        "PAGEABLE_MEMORY_ACCESS",
        A::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,
    );
    print_attr(
        &device,
        "PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES",
        A::CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES,
    );
    print_attr(
        &device,
        "DIRECT_MANAGED_MEM_ACCESS_FROM_HOST",
        A::CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST,
    );
    print_attr(
        &device,
        "HOST_NATIVE_ATOMIC_SUPPORTED",
        A::CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,
    );
    print_attr(&device, "INTEGRATED", A::CU_DEVICE_ATTRIBUTE_INTEGRATED);
    print_attr(
        &device,
        "CAN_MAP_HOST_MEMORY",
        A::CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY,
    );
    print_attr(
        &device,
        "CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM",
        A::CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM,
    );
}
