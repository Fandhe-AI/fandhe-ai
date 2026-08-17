//! JIT キャッシュ（`nvrtc.rs`）のヒット/ミス・並行コンパイル競合・
//! 破損検出の網羅的回帰テスト（イシュー #529・Phase C-10）。
//!
//! # なぜ `crates/backend-cuda/tests/`（integration test）ではなく本ファイル
//! （`nvrtc` モジュールの子モジュール）に置くか
//!
//! JIT キャッシュ API（[`super::store_cache_entry_in`]・
//! [`super::load_cache_entry_in`]・[`super::cache_entry_path_in`]・
//! [`super::validate_cache_entry`] 等）はいずれも `pub(crate)` にすら
//! 満たない module-private `fn` である（`nvrtc.rs` 各関数のドキュメン
//! テーションコメント参照: 「crate 外から呼び出す手段はない」設計が
//! 意図的に維持されている）。Rust の可視性規則上、非 `pub` アイテムは
//! 定義モジュール（`nvrtc`）とその子孫モジュールからしかアクセスできず、
//! `crates/backend-cuda/tests/` 配下の integration test はクレート外部と
//! 同じ扱い（`nvrtc` の子孫ではない）のため到達できない。可視性を
//! `pub(crate)` へ広げる変更は記録済みの設計判断（`docs/
//! cuda-jit-cache-design.md`）を変える承認事項になるため、本イシューの
//! スコープ（テスト追加のみ）では行わない。
//!
//! 代わりに本ファイルは `nvrtc.rs` の `mod tests`（C-3・#509）とは別の、
//! `nvrtc` 直下の兄弟モジュールとして `#[path]` 属性で配置する（宣言は
//! `nvrtc.rs` 末尾を参照）。`use super::*;` により `nvrtc` 本体の
//! 非 `pub` アイテム（`store_cache_entry_in`・`cache_entry_path_in`・
//! `validate_cache_entry`・`fs`・`Path`／`PathBuf` 等）へ直接到達できる。
//! `mod tests` の private ヘルパー（`sample_key` 等）は兄弟モジュールから
//! 見えないため、下記 `sample_descriptor`／`sample_source`／`sample_key`／
//! `fresh_temp_dir` は本ファイル内に独立して定義する（`mod tests` 側の
//! 同名ヘルパーと実装は意図的に同型だが、モジュール境界をまたいだ結合を
//! 避けるため複製する）。既存テスト（roundtrip・不在ミス・ptx 欠落ミス・
//! 空 source ミス・不正 UTF-8 ミス・store 二重書き吸収・破損置換・
//! 0 バイト残骸置換・非ディレクトリ占有置換・スレッド並行 store・
//! symlink／FIFO／サイズ超過拒否・`ensure_cache_root_in` 系）とは重複
//! しない観点のみを追加する（実装計画 §2「既存カバレッジ」節）。
//!
//! # `get_or_compile`: C-4（#511）想定導線のテストローカル・シミュレーション
//!
//! C-4（プロセス内 LRU・NVRTC 実結線）はイシュー #529 時点で未実装
//! （実機待ち）。[`get_or_compile`] は C-4 が結線する想定の
//! 「ミス→コンパイル→store→hit」プロトコルを、実際の NVRTC 呼び出し
//! （高コスト処理）の代わりに呼び出し元が渡すクロージャで代替して
//! 検証するためだけのテストヘルパーであり、本番コード（`nvrtc.rs` 本体）
//! には一切手を加えない。

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};

use tensor_core::dispatch::{DType, GemmShape};

use super::*;

/// `mod tests`（`nvrtc.rs`）の `sample_descriptor` と同型の合成
/// descriptor（`STATIC_MNK`・block 64/64/32・stages 2・F32）。
fn sample_descriptor() -> CudaKernelDescriptor {
    CudaKernelDescriptor::new_with_compiled_dims(
        "wmma_tf32_f32",
        GemmShape::new(4096, 4096, 4096),
        64,
        64,
        32,
        2,
        DType::F32,
        CompiledDims::STATIC_MNK,
    )
    .expect("valid descriptor parameters must not fail")
}

/// `mod tests` の `sample_source` と同型: 実テンプレート
/// （`kernels_mma::mma_f16_source`）をそのまま使う（合成のダミー文字列
/// ではなく実際にコンパイル対象となるソースで検証するため）。
fn sample_source() -> String {
    crate::kernels_mma::mma_f16_source().to_string()
}

/// `mod tests` の `sample_key` と同型の合成キー。
fn sample_key() -> CudaKernelCacheKey {
    CudaKernelCacheKey::new(
        sample_descriptor(),
        (8, 0),
        (12, 9),
        vec!["--gpu-architecture=compute_80".to_string()],
        sample_source(),
    )
}

/// `mod tests` の `fresh_temp_dir` と同型: テスト用に一意な一時
/// ディレクトリを払い出す（プロセス内 `AtomicU64` カウンタ＋PID で並行
/// テスト実行時の衝突を避ける）。呼び出し元がテスト末尾で
/// `remove_dir_all` して片付ける。
fn fresh_temp_dir(label: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust-ai-library-cache-test.c10.{label}.{}.{seq}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("failed to create test temp dir");
    dir
}

/// C-4（#511）が結線する想定の「ミス→コンパイル→store→hit」導線を
/// テストローカルに模したヘルパー。
///
/// - キャッシュヒット時: `compile_fn` を呼ばず、ヒットした
///   `(エントリパス, false, 保存済み PTX)` を返す（受け入れ基準 1:
///   2 回目のアクセスがコンパイルを起こさないこと、を直接表現する）。
/// - キャッシュミス時: `compile_fn`（NVRTC コンパイル相当）を 1 回呼び、
///   結果を [`store_cache_entry_in`] で保存してから
///   `(エントリパス, true, 権威ある PTX)` を返す。
///
/// `store_cache_entry_in` は「他 writer が同一キーへ先着していれば
/// 自分の書き込みを破棄し正常系として吸収する」契約（受け入れ基準 1）
/// のため、store 呼び出し自体が成功しても実際にディスクへ反映された
/// のが自分の `compile_fn` 結果とは限らない（並行レースに負けた場合は
/// 先着 writer の PTX が実体になる）。そのため store 後に必ず
/// [`load_cache_entry_in`] で再読込し、以降の呼び出し元がディスク上の
/// 実体と一致する PTX を観測できるようにする（[`concurrent_get_or_compile_same_key_from_threads`]
/// が「全レース参加者が単一の勝者 PTX へ収束する」ことを検証できるのは
/// この再読込があるため）。
fn get_or_compile(
    root: &Path,
    key: &CudaKernelCacheKey,
    src: &str,
    compile_fn: impl FnOnce() -> String,
) -> Result<(PathBuf, bool, String), CudaError> {
    if let Some(cached) = load_cache_entry_in(root, key, src)? {
        return Ok((cache_entry_path_in(root, key)?, false, cached.kernel_ptx));
    }
    let ptx = compile_fn();
    let dir = store_cache_entry_in(root, key, src, &ptx)?;
    let authoritative = load_cache_entry_in(root, key, src)?.ok_or_else(|| CudaError::CacheIo {
        detail: "cache entry vanished immediately after a successful store".to_string(),
    })?;
    Ok((dir, true, authoritative.kernel_ptx))
}

// ============================================================================
// AC1: ヒット/ミス — 同一キーでの 2 回目のアクセスがコンパイルを
// 起こさないこと（受け入れ基準 1）。
// ============================================================================

#[test]
fn second_access_with_same_key_does_not_recompile() {
    let root = fresh_temp_dir("c10-second-access");
    let key = sample_key();
    let src = sample_source();
    let compile_calls = AtomicUsize::new(0);

    let (dir1, compiled1, ptx1) = get_or_compile(&root, &key, &src, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// compiled ptx (1st compile)".to_string()
    })
    .expect("cold-cache access must succeed");
    assert!(compiled1, "cold cache must compile on first access");
    assert_eq!(compile_calls.load(Ordering::SeqCst), 1);

    let (dir2, compiled2, ptx2) = get_or_compile(&root, &key, &src, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// compiled ptx (must not be reached)".to_string()
    })
    .expect("warm-cache access must succeed");
    assert!(!compiled2, "warm cache hit must not invoke compile_fn");
    assert_eq!(
        compile_calls.load(Ordering::SeqCst),
        1,
        "compile_fn must not run a second time for the same key"
    );
    assert_eq!(dir1, dir2, "hit must resolve to the same entry directory");
    assert_eq!(
        ptx1, ptx2,
        "hit must return byte-identical PTX to what the first compile produced"
    );

    let _ = fs::remove_dir_all(&root);
}

/// 受け入れ基準 1 の境界網羅: descriptor（block_m・dtype・compiled_dims）・
/// compute capability・NVRTC バージョン・compile_flags のいずれかが
/// 異なれば別キャッシュキーとなり、それぞれ独立に「1 回だけ」
/// コンパイルされること（かつ 2 回目は各々ヒットすること）。
#[test]
fn distinct_keys_each_compile_once() {
    let root = fresh_temp_dir("c10-distinct-keys");
    let shape = GemmShape::new(4096, 4096, 4096);
    let src = sample_source();
    let base_flags = vec!["--gpu-architecture=compute_80".to_string()];

    let base_descriptor = || {
        CudaKernelDescriptor::new_with_compiled_dims(
            "wmma_tf32_f32",
            shape,
            64,
            64,
            32,
            2,
            DType::F32,
            CompiledDims::STATIC_MNK,
        )
        .expect("valid descriptor parameters must not fail")
    };

    let variants: Vec<(&str, CudaKernelCacheKey)> = vec![
        (
            "block_m_128",
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new_with_compiled_dims(
                    "wmma_tf32_f32",
                    shape,
                    128,
                    64,
                    32,
                    2,
                    DType::F32,
                    CompiledDims::STATIC_MNK,
                )
                .expect("valid descriptor"),
                (8, 0),
                (12, 9),
                base_flags.clone(),
                src.clone(),
            ),
        ),
        (
            "dtype_f16",
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new_with_compiled_dims(
                    "wmma_tf32_f32",
                    shape,
                    64,
                    64,
                    32,
                    2,
                    DType::F16,
                    CompiledDims::STATIC_MNK,
                )
                .expect("valid descriptor"),
                (8, 0),
                (12, 9),
                base_flags.clone(),
                src.clone(),
            ),
        ),
        (
            "compiled_dims_static_nk",
            CudaKernelCacheKey::new(
                CudaKernelDescriptor::new_with_compiled_dims(
                    "wmma_tf32_f32",
                    shape,
                    64,
                    64,
                    32,
                    2,
                    DType::F32,
                    CompiledDims::STATIC_NK,
                )
                .expect("valid descriptor"),
                (8, 0),
                (12, 9),
                base_flags.clone(),
                src.clone(),
            ),
        ),
        (
            "compute_capability_9_0",
            CudaKernelCacheKey::new(
                base_descriptor(),
                (9, 0),
                (12, 9),
                base_flags.clone(),
                src.clone(),
            ),
        ),
        (
            "nvrtc_version_13_0",
            CudaKernelCacheKey::new(
                base_descriptor(),
                (8, 0),
                (13, 0),
                base_flags.clone(),
                src.clone(),
            ),
        ),
        (
            "compile_flags_fast_math",
            CudaKernelCacheKey::new(
                base_descriptor(),
                (8, 0),
                (12, 9),
                vec![
                    "--gpu-architecture=compute_80".to_string(),
                    "--use_fast_math".to_string(),
                ],
                src.clone(),
            ),
        ),
    ];

    let mut seen_dirs = HashSet::new();
    for (label, key) in variants {
        let compile_calls = AtomicUsize::new(0);
        let (dir, compiled, _) = get_or_compile(&root, &key, &src, || {
            compile_calls.fetch_add(1, Ordering::SeqCst);
            format!("// ptx for variant {label}")
        })
        .unwrap_or_else(|e| panic!("variant {label} must compile on cold cache: {e:?}"));
        assert!(compiled, "variant {label} must be a cold-cache compile");
        assert_eq!(
            compile_calls.load(Ordering::SeqCst),
            1,
            "variant {label} must compile exactly once"
        );
        assert!(
            seen_dirs.insert(dir),
            "variant {label} must resolve to a cache entry directory distinct from all prior variants"
        );

        let (_, compiled_again, _) = get_or_compile(&root, &key, &src, || {
            panic!("variant {label}: second access to the same key must not recompile")
        })
        .unwrap_or_else(|e| panic!("variant {label} warm access must succeed: {e:?}"));
        assert!(
            !compiled_again,
            "variant {label} second access must be a hit"
        );
    }

    let _ = fs::remove_dir_all(&root);
}

/// 受け入れ基準 1 の end-to-end 確認（C-5・#514 のソース取り込み契約）:
/// 同一 descriptor・同一環境パラメータでもソースのみ変更すればミスとなり
/// 再コンパイルされる。
#[test]
fn source_change_invalidates_cache() {
    let root = fresh_temp_dir("c10-source-change");
    let flags = vec!["--gpu-architecture=compute_80".to_string()];
    let src_v1 = sample_source();
    let key_v1 = CudaKernelCacheKey::new(
        sample_descriptor(),
        (8, 0),
        (12, 9),
        flags.clone(),
        src_v1.clone(),
    );
    let compile_calls = AtomicUsize::new(0);

    let (_, compiled1, _) = get_or_compile(&root, &key_v1, &src_v1, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// ptx v1".to_string()
    })
    .expect("first compile must succeed");
    assert!(compiled1);

    let src_v2 = format!("{src_v1}\n// edited for C-10 regression test\n");
    let key_v2 =
        CudaKernelCacheKey::new(sample_descriptor(), (8, 0), (12, 9), flags, src_v2.clone());

    let (_, compiled2, ptx2) = get_or_compile(&root, &key_v2, &src_v2, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// ptx v2".to_string()
    })
    .expect("edited-source compile must succeed");
    assert!(
        compiled2,
        "editing the source with an otherwise identical descriptor must miss and recompile"
    );
    assert_eq!(compile_calls.load(Ordering::SeqCst), 2);
    assert_eq!(ptx2, "// ptx v2");

    let _ = fs::remove_dir_all(&root);
}

// ============================================================================
// AC2: 並行 N プロセス/スレッドが同一キーへ書き込む際の rename 競合が
// 正常系として吸収されること（受け入れ基準 1）。
// ============================================================================

#[test]
fn concurrent_get_or_compile_same_key_from_threads() {
    const N: usize = 8;
    let root = Arc::new(fresh_temp_dir("c10-concurrent-get-or-compile"));
    let key = Arc::new(sample_key());
    let src = Arc::new(sample_source());
    let compile_calls = Arc::new(AtomicUsize::new(0));
    let barrier = Arc::new(Barrier::new(N));

    let handles: Vec<_> = (0..N)
        .map(|i| {
            let root = Arc::clone(&root);
            let key = Arc::clone(&key);
            let src = Arc::clone(&src);
            let compile_calls = Arc::clone(&compile_calls);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                get_or_compile(&root, &key, &src, || {
                    compile_calls.fetch_add(1, Ordering::SeqCst);
                    format!("// ptx from racing thread {i}")
                })
            })
        })
        .collect();

    let mut ptxs = HashSet::new();
    for handle in handles {
        let (_, _, ptx) = handle
            .join()
            .expect("thread must not panic")
            .expect("get_or_compile must succeed under concurrent cold-cache racing");
        ptxs.insert(ptx);
    }
    assert_eq!(
        ptxs.len(),
        1,
        "all racing threads must observe a single winning PTX after the rename race settles"
    );
    assert!(
        compile_calls.load(Ordering::SeqCst) >= 1,
        "at least one racing thread must have compiled on the cold cache"
    );

    let entry_path = cache_entry_path_in(&root, &key).expect("must resolve entry path");
    assert!(
        validate_cache_entry(&entry_path),
        "entry must be valid (not left mid-race) after all threads join"
    );

    // warm 後: 全スレッドがヒットし compile_fn は一切呼ばれないこと。
    let warm_calls = Arc::new(AtomicUsize::new(0));
    let warm_barrier = Arc::new(Barrier::new(N));
    let warm_handles: Vec<_> = (0..N)
        .map(|_| {
            let root = Arc::clone(&root);
            let key = Arc::clone(&key);
            let src = Arc::clone(&src);
            let warm_calls = Arc::clone(&warm_calls);
            let warm_barrier = Arc::clone(&warm_barrier);
            std::thread::spawn(move || {
                warm_barrier.wait();
                get_or_compile(&root, &key, &src, || {
                    warm_calls.fetch_add(1, Ordering::SeqCst);
                    "must-not-be-called".to_string()
                })
            })
        })
        .collect();
    for handle in warm_handles {
        let (_, compiled, _) = handle
            .join()
            .expect("thread must not panic")
            .expect("warm concurrent access must succeed");
        assert!(
            !compiled,
            "warm cache hit must never call compile_fn under concurrency"
        );
    }
    assert_eq!(
        warm_calls.load(Ordering::SeqCst),
        0,
        "warm concurrent access must not invoke compile_fn at all"
    );

    let _ = fs::remove_dir_all(root.as_path());
}

#[test]
fn concurrent_store_different_keys_do_not_interfere() {
    const N: usize = 6;
    let root = Arc::new(fresh_temp_dir("c10-concurrent-distinct-keys"));
    let src = sample_source();
    let shape = GemmShape::new(4096, 4096, 4096);

    let keys: Vec<Arc<CudaKernelCacheKey>> = (0..N)
        .map(|i| {
            let descriptor = CudaKernelDescriptor::new_with_compiled_dims(
                "wmma_tf32_f32",
                shape,
                64 + (i as u32) * 16,
                64,
                32,
                2,
                DType::F32,
                CompiledDims::STATIC_MNK,
            )
            .expect("valid descriptor parameters must not fail");
            Arc::new(CudaKernelCacheKey::new(
                descriptor,
                (8, 0),
                (12, 9),
                vec!["--gpu-architecture=compute_80".to_string()],
                src.clone(),
            ))
        })
        .collect();

    let barrier = Arc::new(Barrier::new(N));
    let handles: Vec<_> = keys
        .iter()
        .enumerate()
        .map(|(i, key)| {
            let root = Arc::clone(&root);
            let key = Arc::clone(key);
            let src = src.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                store_cache_entry_in(&root, &key, &src, &format!("// ptx {i}"))
            })
        })
        .collect();

    for (i, handle) in handles.into_iter().enumerate() {
        handle
            .join()
            .unwrap_or_else(|_| panic!("thread {i} must not panic"))
            .unwrap_or_else(|e| panic!("independent-key store {i} must succeed: {e:?}"));
    }

    for (i, key) in keys.iter().enumerate() {
        let loaded = load_cache_entry_in(&root, key, &src)
            .expect("load must succeed")
            .unwrap_or_else(|| panic!("key {i} must be a hit"));
        assert_eq!(
            loaded.kernel_ptx,
            format!("// ptx {i}"),
            "entries for distinct keys must not cross-contaminate under concurrent store"
        );
    }

    let _ = fs::remove_dir_all(root.as_path());
}

/// 通常の `cargo test` 実行時は環境変数 `MP_CHILD_ROOT_ENV` が未設定の
/// ため no-op で pass する。
/// [`multiprocess_concurrent_store_absorbs_rename_race`] が
/// `std::env::current_exe()` を `--exact` 付きで再起動したときのみ、
/// 環境変数経由でキャッシュ root を受け取り実際に 1 回 store する
/// （実プロセス境界を跨いだ rename 競合を検証するための子プロセス役）。
const MP_CHILD_ROOT_ENV: &str = "RAI_C10_MP_CHILD_ROOT";
const MP_CHILD_INDEX_ENV: &str = "RAI_C10_MP_CHILD_INDEX";

/// [`multiprocess_concurrent_store_absorbs_rename_race`] の barrier
/// 実装で親・子が共有するファイルベース同期プロトコル（プロセス境界を
/// 跨ぐため `std::sync::Barrier` は使えず、キャッシュ root と同じ
/// ファイルシステム上に readiness マーカー／go シグナルを置く）。
///
/// - `mp_sync/ready.<index>`: 子プロセスが store 直前に自分の index で
///   作成する空ファイル。親はこれが N 個揃うまで待つ。
/// - `mp_sync/go`: 親が全 ready を確認した後に作成する空ファイル。
///   子プロセスはこれの出現をポーリングしてから store を呼ぶ。
///
/// この 2 段階同期がないと、先行して spawn されたプロセスが後続プロセス
/// の起動前に書き込みを完了し得るため、受け入れ基準が要求する rename
/// 競合（複数プロセスが cold cache を確認した状態から同時に store へ
/// 進む状況）を通らずにテストが成功してしまう（イシュー #529 コメント
/// `PRRT_kwDOTuUCJc6ZnY5S`・codex-review 指摘）。
mod mp_sync {
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    /// barrier 待ちの上限（CI 環境でのハング防止。通常はミリ秒単位で
    /// 全員が揃うため、この値に到達するのは既に異常系のみ）。
    const WAIT_TIMEOUT: Duration = Duration::from_secs(30);
    const POLL_INTERVAL: Duration = Duration::from_micros(100);

    pub(super) fn dir(root: &Path) -> PathBuf {
        root.join("mp_sync")
    }

    pub(super) fn ready_marker(root: &Path, index: usize) -> PathBuf {
        dir(root).join(format!("ready.{index}"))
    }

    pub(super) fn go_marker(root: &Path) -> PathBuf {
        dir(root).join("go")
    }

    /// 子プロセス側: 自分の ready マーカーを作成し、`go` マーカーが
    /// 現れるまでポーリングで待つ。
    pub(super) fn child_wait_for_go(root: &Path, index: usize) {
        std::fs::create_dir_all(dir(root)).expect("mp_sync dir must be creatable by child");
        std::fs::write(ready_marker(root, index), b"")
            .expect("child ready marker must be writable");

        let go = go_marker(root);
        let start = Instant::now();
        while !go.exists() {
            assert!(
                start.elapsed() < WAIT_TIMEOUT,
                "child {index} timed out waiting for the parent's go signal"
            );
            std::thread::sleep(POLL_INTERVAL);
        }
    }

    /// 親プロセス側: N 個の ready マーカーが揃うまで待ってから `go`
    /// マーカーを作成し、全子プロセスの同時 store 開始を解禁する。
    pub(super) fn parent_release_when_all_ready(root: &Path, total: usize) {
        let dir_path = dir(root);
        std::fs::create_dir_all(&dir_path).expect("mp_sync dir must be creatable by parent");

        let start = Instant::now();
        loop {
            let ready_count = (0..total)
                .filter(|&i| ready_marker(root, i).exists())
                .count();
            if ready_count == total {
                break;
            }
            assert!(
                start.elapsed() < WAIT_TIMEOUT,
                "parent timed out waiting for all {total} children to signal ready \
                 (observed {ready_count})"
            );
            std::thread::sleep(POLL_INTERVAL);
        }

        std::fs::write(go_marker(root), b"").expect("go marker must be writable by parent");
    }
}

#[test]
fn mp_store_child() {
    let Ok(root) = std::env::var(MP_CHILD_ROOT_ENV) else {
        return;
    };
    let index = std::env::var(MP_CHILD_INDEX_ENV).unwrap_or_default();
    let index_num: usize = index.parse().expect("child index must be a valid usize");
    let root = PathBuf::from(root);
    let key = sample_key();
    let src = sample_source();

    // 受け入れ基準 1 が要求する「rename 競合の経路を実際に通ること」を
    // 保証するため、ready マーカーを立てる前に cache が cold（未ヒット）
    // であることを確認する。barrier（`go` は全 N ready 揃うまで書かれ
    // ない）によりこの時点で cache が warm になっていることはないはず
    // だが、barrier が破綻した場合はここで検知して fail-closed に落ちる
    // （黙って「全プロセスが起動済み」を仮定した緩いテストへ後退しない
    // ため。イシュー #529 コメント `PRRT_kwDOTuUCJc6ZnY5S`・codex-review
    // 指摘対応）。
    let cold = load_cache_entry_in(&root, &key, &src).expect("child cold-cache load must succeed");
    assert!(
        cold.is_none(),
        "child {index} observed a warm cache entry before signaling ready; \
         the parent/child barrier failed to keep all processes on the cold-cache path"
    );

    // 全子プロセスが cold cache を確認した状態から同時に store へ進める
    // よう、親からの go シグナルを待ってから store する（barrier。上記
    // `mp_sync` モジュールのドキュメント参照）。
    mp_sync::child_wait_for_go(&root, index_num);

    store_cache_entry_in(&root, &key, &src, &format!("// mp ptx from child {index}"))
        .expect("child store_cache_entry_in must succeed");
}

/// 受け入れ基準 1: 並行 N **プロセス**（スレッドではなく実プロセス境界）
/// が同一キーへ同時に store しても rename 競合が正常系として吸収され、
/// 最終的に単一の有効なエントリへ収束すること。
///
/// `std::env::current_exe()`（本テストバイナリ自身のパス）を
/// `--exact mp_store_child` で N 回再帰起動する。シェルは経由せず
/// `std::process::Command` に固定引数のみを渡す（A03 インジェクション
/// 対策。`.claude/rules/security.md`）。
///
/// spawn する順序だけでは「全 N プロセスが cold cache を確認してから
/// 同時に store へ進む」状況を保証できない（先に起動した子が後続の
/// 起動前に書き込みを完了してしまうと、後続は既存の有効エントリを
/// 見るだけになり rename 競合の経路を一度も通らずテストが偽陽性で
/// 成功しうる。イシュー #529 コメント `PRRT_kwDOTuUCJc6ZnY5S`・
/// codex-review 指摘）。そのため `mp_sync` の barrier（子は ready
/// マーカーを立てて `go` を待ち、親は全 ready を確認してから `go` を
/// 書く）で全子プロセスの store 開始を同時刻へ揃える。
#[test]
fn multiprocess_concurrent_store_absorbs_rename_race() {
    const N: usize = 5;
    let root = fresh_temp_dir("c10-multiprocess");
    let exe = std::env::current_exe().expect("must resolve current test binary path");

    let mut children = Vec::with_capacity(N);
    for i in 0..N {
        let child = std::process::Command::new(&exe)
            .arg("nvrtc::jit_cache_regression_tests::mp_store_child")
            .arg("--exact")
            .arg("--test-threads=1")
            .env(MP_CHILD_ROOT_ENV, &root)
            .env(MP_CHILD_INDEX_ENV, i.to_string())
            .spawn()
            .expect("must spawn child test process");
        children.push(child);
    }

    // 全 N 子プロセスが ready マーカーを立てる（= cold cache 確認直前まで
    // 進んだ）ことを確認してから go シグナルを送り、store 開始を同時刻へ
    // 揃える（barrier。上記コメント参照）。
    mp_sync::parent_release_when_all_ready(&root, N);

    for (i, mut child) in children.into_iter().enumerate() {
        let status = child.wait().expect("must wait for child process");
        assert!(status.success(), "child process {i} must exit 0");
    }

    let key = sample_key();
    let src = sample_source();
    let loaded = load_cache_entry_in(&root, &key, &src)
        .expect("load must succeed")
        .expect(
            "one of the N child processes must have won the rename race and be visible as a hit",
        );
    assert!(loaded.kernel_ptx.starts_with("// mp ptx from child "));

    let _ = fs::remove_dir_all(&root);
}

// ============================================================================
// AC3: 破損キャッシュ（エントリ内ファイルの片方削除等）をミスとして
// 検出し、再コンパイル → 置換で回復すること（受け入れ基準 2）。
// ============================================================================

#[test]
fn corrupt_entry_missing_cu_recovers_via_recompile() {
    let root = fresh_temp_dir("c10-corrupt-missing-cu");
    let key = sample_key();
    let src = sample_source();
    let compile_calls = AtomicUsize::new(0);

    let (dir, compiled1, _) = get_or_compile(&root, &key, &src, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// ptx original".to_string()
    })
    .expect("initial compile must succeed");
    assert!(compiled1);

    fs::remove_file(dir.join(CACHE_ENTRY_SOURCE_FILE)).expect("must remove kernel.cu");
    assert!(
        !validate_cache_entry(&dir),
        "entry missing kernel.cu must be considered corrupt"
    );

    let (dir2, compiled2, ptx2) = get_or_compile(&root, &key, &src, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// ptx recompiled".to_string()
    })
    .expect("recovery compile must succeed");
    assert!(
        compiled2,
        "missing kernel.cu must be detected as a miss and trigger recompilation"
    );
    assert_eq!(compile_calls.load(Ordering::SeqCst), 2);
    assert_eq!(dir2, dir, "recovery must replace the same entry directory");
    assert!(validate_cache_entry(&dir2), "recovered entry must be valid");

    let (_, compiled3, ptx3) = get_or_compile(&root, &key, &src, || {
        panic!("post-recovery access must hit and must not recompile again")
    })
    .expect("post-recovery access must succeed");
    assert!(!compiled3);
    assert_eq!(ptx3, ptx2);

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn corrupt_entry_missing_ptx_recovers_via_recompile() {
    let root = fresh_temp_dir("c10-corrupt-missing-ptx");
    let key = sample_key();
    let src = sample_source();
    let compile_calls = AtomicUsize::new(0);

    let (dir, compiled1, _) = get_or_compile(&root, &key, &src, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// ptx original".to_string()
    })
    .expect("initial compile must succeed");
    assert!(compiled1);

    fs::remove_file(dir.join(CACHE_ENTRY_PTX_FILE)).expect("must remove kernel.ptx");
    assert!(
        !validate_cache_entry(&dir),
        "entry missing kernel.ptx must be considered corrupt"
    );

    let (dir2, compiled2, ptx2) = get_or_compile(&root, &key, &src, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// ptx recompiled".to_string()
    })
    .expect("recovery compile must succeed");
    assert!(
        compiled2,
        "missing kernel.ptx must be detected as a miss and trigger recompilation"
    );
    assert_eq!(compile_calls.load(Ordering::SeqCst), 2);
    assert_eq!(dir2, dir, "recovery must replace the same entry directory");
    assert!(validate_cache_entry(&dir2), "recovered entry must be valid");

    let (_, compiled3, ptx3) = get_or_compile(&root, &key, &src, || {
        panic!("post-recovery access must hit and must not recompile again")
    })
    .expect("post-recovery access must succeed");
    assert!(!compiled3);
    assert_eq!(ptx3, ptx2);

    let _ = fs::remove_dir_all(&root);
}

#[test]
fn corrupt_entry_empty_dir_recovers() {
    let root = fresh_temp_dir("c10-corrupt-empty-dir");
    let key = sample_key();
    let src = sample_source();
    let compile_calls = AtomicUsize::new(0);

    let (dir, compiled1, _) = get_or_compile(&root, &key, &src, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// ptx original".to_string()
    })
    .expect("initial compile must succeed");
    assert!(compiled1);

    fs::remove_file(dir.join(CACHE_ENTRY_SOURCE_FILE)).expect("must remove kernel.cu");
    fs::remove_file(dir.join(CACHE_ENTRY_PTX_FILE)).expect("must remove kernel.ptx");
    assert!(
        !validate_cache_entry(&dir),
        "entry with both files removed must be considered corrupt"
    );

    let (dir2, compiled2, _) = get_or_compile(&root, &key, &src, || {
        compile_calls.fetch_add(1, Ordering::SeqCst);
        "// ptx recompiled".to_string()
    })
    .expect("recovery compile must succeed");
    assert!(
        compiled2,
        "empty entry directory must be detected as a miss and trigger recompilation"
    );
    assert_eq!(compile_calls.load(Ordering::SeqCst), 2);
    assert_eq!(dir2, dir);
    assert!(validate_cache_entry(&dir2), "recovered entry must be valid");

    let _ = fs::remove_dir_all(&root);
}

/// 既存 [`super::tests::load_treats_empty_source_file_as_miss`] は
/// `kernel.cu` 側の非空検査を検証する。本テストは同じ観点を `kernel.ptx`
/// 側に対して追加し、非空検査が両ファイルへ対称に適用されることを
/// 明示する。
#[test]
fn zero_byte_ptx_is_treated_as_miss() {
    let root = fresh_temp_dir("c10-zero-byte-ptx");
    let key = sample_key();
    let src = sample_source();

    let dir = store_cache_entry_in(&root, &key, &src, "ptx-body").expect("store must succeed");
    fs::write(dir.join(CACHE_ENTRY_PTX_FILE), b"").expect("must truncate kernel.ptx to zero bytes");

    let loaded = load_cache_entry_in(&root, &key, &src).expect("load must succeed");
    assert!(
        loaded.is_none(),
        "zero-byte kernel.ptx must be treated as a miss"
    );
    assert!(
        !validate_cache_entry(&dir),
        "zero-byte kernel.ptx must also fail the store-side corruption check"
    );

    let _ = fs::remove_dir_all(&root);
}

/// クラッシュ残骸を模した `.tmp.<entry>.<pid>.<seq>` ディレクトリが
/// キャッシュルートに残存していても、無関係な store/load 経路は正常に
/// 動作すること（本番の一時ディレクトリ命名規約と同型の残骸を
/// 事前配置するのみで、掃除処理自体は本イシューのスコープ外）。
#[test]
fn stale_tmp_dirs_do_not_break_store_or_load() {
    let root = fresh_temp_dir("c10-stale-tmp");
    let stale_tmp = root.join(format!(".tmp.stale-entry.{}.999", std::process::id()));
    fs::create_dir_all(&stale_tmp).expect("must create stale tmp dir");
    fs::write(stale_tmp.join(CACHE_ENTRY_SOURCE_FILE), "stale").expect("must write stale file");

    let key = sample_key();
    let src = sample_source();
    store_cache_entry_in(&root, &key, &src, "ptx-body")
        .expect("store must succeed despite a pre-existing stale tmp directory");
    let loaded = load_cache_entry_in(&root, &key, &src)
        .expect("load must succeed")
        .expect("entry must be a hit despite a pre-existing stale tmp directory");
    assert_eq!(loaded.kernel_ptx, "ptx-body");
    assert!(
        stale_tmp.exists(),
        "pre-existing stale tmp dir must be left untouched by unrelated store/load calls"
    );

    let _ = fs::remove_dir_all(&stale_tmp);
    let _ = fs::remove_dir_all(&root);
}

// ============================================================================
// PR #703 codex-review P0 回帰テスト（イシュー #511）
//
// 1. `runtime_workspace_root` の境界解決を cwd 直接採用からマーカー探索
//    （`.git`／`[workspace]` Cargo.toml）へ変更した契約の回帰確認
//    （`find_workspace_root_from`）。
// 2. `load_cache_entry_at` のキャッシュエントリ権限検査（group／other
//    書き込み可能なエントリはミス扱いにする）の回帰確認
//    （`is_cache_entry_permission_untrusted`・`load_cache_entry_in` 経由）。
// ============================================================================

/// `.git` ディレクトリを持つ祖先が境界として検出されること。
#[test]
fn find_workspace_root_from_detects_git_ancestor() {
    let root = fresh_temp_dir("workspace-root-git");
    fs::create_dir_all(root.join(".git")).expect("must create .git marker dir");
    let nested = root.join("crates").join("backend-cuda").join("src");
    fs::create_dir_all(&nested).expect("must create nested descendant dir");

    let found = find_workspace_root_from(&nested)
        .expect("must find the .git-marked ancestor from a nested descendant");
    assert_eq!(
        found.canonicalize().expect("root must canonicalize"),
        root.canonicalize().expect("root must canonicalize")
    );

    let _ = fs::remove_dir_all(&root);
}

/// `[workspace]` を持つ `Cargo.toml` の祖先が境界として検出されること。
#[test]
fn find_workspace_root_from_detects_workspace_cargo_toml_ancestor() {
    let root = fresh_temp_dir("workspace-root-cargo-toml");
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/*\"]\n",
    )
    .expect("must write workspace Cargo.toml marker");
    let nested = root.join("crates").join("backend-cuda");
    fs::create_dir_all(&nested).expect("must create nested descendant dir");

    let found = find_workspace_root_from(&nested)
        .expect("must find the [workspace] Cargo.toml ancestor from a nested descendant");
    assert_eq!(
        found.canonicalize().expect("root must canonicalize"),
        root.canonicalize().expect("root must canonicalize")
    );

    let _ = fs::remove_dir_all(&root);
}

/// マーカーのない `Cargo.toml`（`[package]` のみ・workspace メンバー側の
/// クレート単体を模したもの）は境界として検出されないこと（メンバー
/// クレート自身を誤って workspace 境界と判定しないことの確認）。
#[test]
fn find_workspace_root_from_ignores_non_workspace_cargo_toml() {
    let root = fresh_temp_dir("workspace-root-member-only");
    fs::write(root.join("Cargo.toml"), "[package]\nname = \"member\"\n")
        .expect("must write non-workspace Cargo.toml");

    assert!(
        !has_workspace_root_marker(&root),
        "a [package]-only Cargo.toml must not be treated as a workspace boundary marker"
    );

    let _ = fs::remove_dir_all(&root);
}

/// group／other 書き込み可能なディレクトリに置かれた `.git` は境界
/// マーカーとして信頼されないこと（Cursor Bugbot 指摘〈Forgeable
/// workspace root markers〉対応の回帰確認: `/tmp` 等の共有祖先ディレクトリ
/// 配下に攻撃者が `.git` を仕込むだけで `workspace_root` を偽造できない
/// ことを検証する。所有 uid が異なるケースはテスト環境で別 uid の
/// プロセスを起動できないため権限ビットのみで検証するが、
/// `has_workspace_root_marker` は mode・uid いずれか一方でも信頼できない
/// 場合に拒否する実装のため、mode 側の検証だけでも「両方満たす場合のみ
/// 信頼する」契約の主要部分をカバーする）。
#[test]
fn has_workspace_root_marker_rejects_world_writable_git_ancestor() {
    use std::os::unix::fs::PermissionsExt;

    let root = fresh_temp_dir("workspace-root-forged-git");
    fs::create_dir_all(root.join(".git")).expect("must create .git marker dir");
    // 攻撃シナリオの模擬: 通常このディレクトリを所有するユーザー以外も
    // 書き込める共有ディレクトリ（例: `/tmp` 直下）を想定し、`root`
    // 自体を other 書き込み可能（`0o707`）へ緩める。中身（`.git`）は
    // 変更しない。
    fs::set_permissions(&root, fs::Permissions::from_mode(0o707))
        .expect("must relax workspace root dir permissions to other-writable");

    assert!(
        !has_workspace_root_marker(&root),
        "a world-writable ancestor must not be trusted as a workspace root marker \
         even if it contains a .git directory"
    );

    // 権限を戻してからでないと `remove_dir_all` の後始末で権限起因の
    // 失敗が起きうる環境があるため、テスト終了前に元へ戻す。
    let _ = fs::set_permissions(&root, fs::Permissions::from_mode(0o755));
    let _ = fs::remove_dir_all(&root);
}

/// マーカーがどの祖先にも存在しない場合は `None`（呼び出し元
/// `runtime_workspace_root` は `Err` へ変換し、ディスクキャッシュなし
/// 運転へ縮退する。境界不明を許容側で埋めない契約の回帰確認。イシュー
/// #511 PR #703 codex-review P0 指摘: 旧実装は cwd をそのまま境界として
/// 受理しており、この「不明なら拒否」という契約自体を持たなかった）。
#[test]
fn find_workspace_root_from_returns_none_without_any_marker() {
    // マーカーを一切置かない孤立した一時ディレクトリ木。祖先を root まで
    // 辿っても `.git`／`[workspace]` Cargo.toml のいずれにも遭遇しない
    // 前提（テスト環境の一時ディレクトリ配下にこれらが存在しないこと。
    // 既存の `resolve_cache_root` 系テストと同じ `std::env::temp_dir()`
    // 配下を使う前提を踏襲する）。
    let root = fresh_temp_dir("workspace-root-none");
    let nested = root.join("a").join("b").join("c");
    fs::create_dir_all(&nested).expect("must create nested descendant dir");

    assert!(
        find_workspace_root_from(&nested).is_none(),
        "an isolated temp dir tree with no .git/[workspace] marker must resolve to None, \
         not silently fall back to an unverified boundary"
    );

    let _ = fs::remove_dir_all(&root);
}

/// group 書き込み可能な `kernel.ptx` を持つ既存エントリは、`kernel.cu`
/// が要求元ソースと一致していてもミス扱い（`Ok(None)`）になり、NVRTC
/// 再コンパイルへフォールバックすること（PR #703 codex-review P0 指摘の
/// 再現条件: `kernel.cu` を維持したまま `kernel.ptx` だけを書き換える
/// 攻撃シナリオに対する権限ベースの防御を検証する）。
#[test]
fn load_cache_entry_rejects_group_writable_ptx_file() {
    use std::os::unix::fs::PermissionsExt;

    let root = fresh_temp_dir("c10-ptx-group-writable");
    let key = sample_key();
    let src = sample_source();
    store_cache_entry_in(&root, &key, &src, "authentic-ptx-body")
        .expect("initial store must succeed");

    let entry_dir = root.join(
        key.cache_entry_dir_name()
            .expect("cache_entry_dir_name must succeed for a valid key"),
    );
    let ptx_path = entry_dir.join(CACHE_ENTRY_PTX_FILE);
    // 攻撃シナリオの模擬: `kernel.cu` はそのまま、`kernel.ptx` のみ
    // group 書き込み可能（`0o660`。owner rw + group rw）へ権限を緩める。
    // ファイル内容自体は変更しない（内容の改竄検出ではなく権限境界の
    // 検証が目的のため）。
    fs::set_permissions(&ptx_path, fs::Permissions::from_mode(0o660))
        .expect("must relax kernel.ptx permissions to group-writable");

    let loaded = load_cache_entry_in(&root, &key, &src).expect("load call itself must not error");
    assert!(
        loaded.is_none(),
        "a group-writable kernel.ptx must be treated as untrusted (miss), even when \
         kernel.cu still matches the expected source byte-for-byte"
    );

    let _ = fs::remove_dir_all(&root);
}

/// other 書き込み可能なエントリディレクトリ自体もミス扱いになること
/// （ファイル個別の権限だけでなく、エントリディレクトリの権限も検査
/// 対象であることの確認）。
#[test]
fn load_cache_entry_rejects_other_writable_entry_dir() {
    use std::os::unix::fs::PermissionsExt;

    let root = fresh_temp_dir("c10-dir-other-writable");
    let key = sample_key();
    let src = sample_source();
    store_cache_entry_in(&root, &key, &src, "authentic-ptx-body")
        .expect("initial store must succeed");

    let entry_dir = root.join(
        key.cache_entry_dir_name()
            .expect("cache_entry_dir_name must succeed for a valid key"),
    );
    fs::set_permissions(&entry_dir, fs::Permissions::from_mode(0o707))
        .expect("must relax entry dir permissions to other-writable");

    let loaded = load_cache_entry_in(&root, &key, &src).expect("load call itself must not error");
    assert!(
        loaded.is_none(),
        "an other-writable entry directory must be treated as untrusted (miss)"
    );

    let _ = fs::remove_dir_all(&root);
}

/// 通常の権限（store 直後のデフォルト）ではこの新しい権限検査によって
/// 正当なヒットが壊れないことの回帰確認（他の roundtrip 系テストと
/// 独立に、権限検査追加そのものが既存の正常系を壊さないことを明示する）。
#[test]
fn load_cache_entry_accepts_default_permissions_from_store() {
    let root = fresh_temp_dir("c10-default-permissions-ok");
    let key = sample_key();
    let src = sample_source();
    store_cache_entry_in(&root, &key, &src, "authentic-ptx-body")
        .expect("initial store must succeed");

    let loaded = load_cache_entry_in(&root, &key, &src)
        .expect("load call itself must not error")
        .expect("default store-created permissions must remain trusted (hit)");
    assert_eq!(loaded.kernel_ptx, "authentic-ptx-body");

    let _ = fs::remove_dir_all(&root);
}
