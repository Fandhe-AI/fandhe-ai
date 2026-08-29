//! `gemm.rs::CudaGemm::new`（f32 GEMM 本番経路）が
//! `module_cache`／NVRTC ディスクキャッシュへ実際に結線されていることを
//! 実機で検証する診断テスト（イシュー #1024）。
//!
//! # なぜ `crates/backend-cuda/tests/`（integration test）ではなく本ファイル
//! （`lib.rs` 直下の兄弟モジュール）に置くか
//!
//! `context_cache`（非公開 `mod`）・`module_cache::KernelModuleCache`
//! （`pub(crate)`）へアクセスするため、integration test では到達できない。
//! `init_cost_diag_tests.rs`（イシュー #926）と同じ理由でクレートルートの
//! 兄弟モジュールとして配置する（同ファイル冒頭ドキュメンテーション
//! コメント参照）。
//!
//! # 実行方法（実機。DGX Spark GB10 セッション）
//!
//! ```text
//! cargo test -p fandhe-ai-backend-cuda --release --all-features -- \
//!     --ignored --nocapture --test-threads=1 module_cache_wiring
//! ```
//!
//! ローカル（libnvrtc 非搭載）では `CudaGemm::new` が
//! `CudaError::NvrtcUnavailable` を返すため、`tests/gemm_naive.rs` と同じ
//! パターンで早期 return し誤って fail しない（本ファイル各テスト参照）。

use crate::context_cache;
use crate::error::CudaError;
use crate::gemm::{CudaGemm, kernel_specs};
use crate::module_cache::KernelModuleCache;
use crate::nvrtc::{cache_entry_path, runtime_workspace_root};

/// `CudaGemm::new` 呼び出し前に早期 return すべきかを判定する。
/// `tests/gemm_naive.rs::new_does_not_panic_and_returns_typed_result` と
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

/// (a) 同一 `context_cache::cached_device` 上で `CudaGemm::new` を 2 回
/// 構築すると、2 回目は `KernelModuleCache` の `hit_count` が
/// [`kernel_specs`] の要素数（8）以上増加し、`miss_count` は増加しない
/// （1 回目のコンパイル・insert で埋まった LRU を 2 回目が再利用する）。
///
/// `context_cache::cached_device(0)` を使う理由: `Arc<CudaContext>` の
/// ポインタ同一性（`module_cache.rs::KernelModuleCache` キーの `ctx_id`）
/// を 2 回の `CudaGemm::new` 呼び出し間で一致させるため。素朴に
/// `CudaDevice::new(0)` を 2 回呼ぶと別 `Arc<CudaContext>` になり
/// LRU が意図的にヒットしない（本モジュール冒頭ドキュメンテーション
/// コメント「ABA 耐性」節と同じ、ctx 単位のキー分離）。
#[test]
#[ignore = "実機（DGX Spark GB10 等の CUDA 搭載環境）専用。libnvrtc 必須"]
fn cuda_gemm_new_second_construction_reuses_module_cache() {
    let device = match context_cache::cached_device(0) {
        Ok(dev) => dev,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from context_cache::cached_device: {e}"),
    };

    let cache = KernelModuleCache::global().expect("module cache must be constructible");
    let misses_before = cache.miss_count();

    let gemm1 = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from CudaGemm::new (1st): {e}"),
    };
    let misses_after_first = cache.miss_count();
    // 1 回目は全カーネルが LRU ミス（新規コンパイル）のはず。
    assert!(
        misses_after_first >= misses_before + kernel_specs().len() as u64,
        "1st CudaGemm::new must miss the module cache for every kernel_specs entry: \
         before={misses_before}, after={misses_after_first}"
    );

    let hits_before_second = cache.hit_count();
    let gemm2 = CudaGemm::new(&device)
        .expect("2nd CudaGemm::new must succeed given the 1st succeeded on the same device");
    let hits_after_second = cache.hit_count();
    let misses_after_second = cache.miss_count();

    assert!(
        hits_after_second >= hits_before_second + kernel_specs().len() as u64,
        "2nd CudaGemm::new on the same cached device must hit the module cache for every \
         kernel_specs entry: before={hits_before_second}, after={hits_after_second}"
    );
    assert_eq!(
        misses_after_second, misses_after_first,
        "2nd CudaGemm::new must not introduce new module cache misses"
    );

    // (b) 同一入力に対する 2 インスタンスの出力が一致することを、既存の
    // 数値一致複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で
    // 確認する。同一 module 由来（LRU ヒット）のため実質 bit 一致を期待
    // するが、判定は `.claude/rules/coding-rust.md` の既存契約に揃える。
    let n: u32 = 64;
    let (a, b) = {
        use bench_harness::rng::Xorshift64Star;
        let mut rng = Xorshift64Star::new(0x1024);
        (
            rng.fill_vec((n * n) as usize),
            rng.fill_vec((n * n) as usize),
        )
    };
    let out1 = gemm1
        .run_tiled_f32(&a, &b, n, n, n)
        .expect("gemm1.run_tiled_f32 must succeed");
    let out2 = gemm2
        .run_tiled_f32(&a, &b, n, n, n)
        .expect("gemm2.run_tiled_f32 must succeed");
    fandhe_ai_backend_cpu::assert_parity(
        "cuda_gemm_new module-cache reuse (tiled_f32)",
        &out1,
        &out2,
    );
}

/// (c) `CudaGemm::new` が 1 回目にコンパイルした各カーネルについて、
/// NVRTC ディスクキャッシュ（`nvrtc.rs::store_cache_entry`）へ実際に
/// エントリが保存されていることを確認する（module_cache（プロセス内
/// LRU）だけでなく、ディスク側の結線も検証する。イシュー #1024）。
#[test]
#[ignore = "実機（DGX Spark GB10 等の CUDA 搭載環境）専用。libnvrtc 必須"]
fn cuda_gemm_new_stores_disk_cache_entry_for_every_kernel() {
    let device = match context_cache::cached_device(0) {
        Ok(dev) => dev,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from context_cache::cached_device: {e}"),
    };
    let root = match runtime_workspace_root() {
        Ok(root) => root,
        Err(e) => {
            eprintln!("workspace_root 解決不能のためスキップ（縮退運転パスは別テストで担保）: {e}");
            return;
        }
    };

    if let Err(e) = CudaGemm::new(&device) {
        if is_environment_unavailable_error(&e) {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        panic!("unexpected CudaError from CudaGemm::new: {e}");
    }

    for spec in kernel_specs() {
        let descriptor = spec
            .descriptor()
            .expect("descriptor construction must succeed for a production kernel_specs entry");
        let compile_flags = vec![format!("--gpu-architecture={}", device.arch())];
        let key = crate::nvrtc::CudaKernelCacheKey::from_device(
            descriptor,
            &device,
            compile_flags,
            spec.source.to_owned(),
        )
        .expect("CudaKernelCacheKey::from_device must succeed on a real device");
        let path = cache_entry_path(&root, &key)
            .expect("cache_entry_path must resolve for a valid workspace_root/key pair");
        assert!(
            path.exists(),
            "disk cache entry for kernel `{}` must exist after CudaGemm::new: {path:?}",
            spec.label
        );
    }
}
