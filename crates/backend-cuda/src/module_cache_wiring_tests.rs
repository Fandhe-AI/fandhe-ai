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
//!
//! # 実行時は必ず `--test-threads=1`（イシュー #1107・GB10 実測 2026-09-03）
//!
//! 本ファイルの実機テスト（とくに
//! [`cuda_gemm_new_second_construction_reuses_module_cache`]）は
//! プロセスワイド `static` の共有 LRU（`module_cache::KernelModuleCache::global`。
//! 既定容量 32）の hit/miss 増分を厳密検査する。GB10 実機実測では、
//! `cargo test --lib -- --ignored --test-threads=1`（本ファイル単体の
//! 直列実行）は pass する一方、`kernels_mma.rs`・`kernels_wmma_opt.rs` 等
//! shape 特化カーネル群を含む lib 全体の `#[ignore]` テスト（`--all-features`
//! 込みで 57 件。`--all-features` なしのローカル計測では 28 件）を
//! 既定の並列テストハーネスで実行すると、他テストが同じ共有 LRU へ
//! 並行 insert し容量超過で `CudaGemm::new` 1 回目が登録した一部
//! エントリが 2 回目の構築前に evict されるため、当該テストのみ flaky に
//! fail する（`init_cost_diag_tests.rs` 冒頭ドキュメンテーションコメント
//! 「実行時は必ず `--test-threads=1`」節と同型の注意。実装の欠陥ではなく
//! テストの隔離不足）。実行は必ず以下のように直列化する:
//!
//! ```text
//! cargo test -p fandhe-ai-backend-cuda --release --lib -- \
//!     --ignored --test-threads=1
//! ```

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

/// [`kernel_specs`] のうち先頭 5 件（naive f32/f16・tiled f32/f16・
/// tiled_bias_act_f32）は `CudaGemm::new` の `?` 早期 return に合流する
/// 必須カーネルであり、`CudaGemm::new` が `Ok` を返した時点で必ず
/// module_cache／NVRTC ディスクキャッシュへ insert 済みである
/// （`gemm.rs::kernel_specs` ドキュメンテーションコメント「順序は」節）。
const MANDATORY_KERNEL_COUNT: u64 = 5;

/// 与えた `CudaGemm` インスタンスについて、`new` 時点で実際にコンパイル・
/// ロードへ成功したカーネル数（[`MANDATORY_KERNEL_COUNT`] + 利用可能な
/// WMMA(TF32) 系フォールバックカーネルの数）を返す。
///
/// [`kernel_specs`] の要素数（8）を無条件の期待値として使わない理由
/// （イシュー #1024 レビュー指摘。Cursor Bugbot Low Severity）: 末尾 3 件
/// （wmma_tf32／wmma_tf32_opt／wmma_tf32_staged）は `compile_wmma_tf32`
/// 等が NVRTC コンパイル失敗を `Option::None` へ fail-soft する
/// フォールバック方式（`gemm.rs::CudaGemm::wmma_tf32` フィールド
/// ドキュメンテーションコメント参照）であり、`load_function_cached` の
/// 呼び出し自体は発生するが insert（cache へのコンパイル成果物登録）は
/// 成功時にしか起きない。TF32 WMMA が未対応（naive/tiled はビルドできる）
/// な CUDA デバイスでは、無条件に 8 を期待すると 1 回目の miss 数・2 回目
/// の hit 数の下限判定もディスクエントリ存在判定も過大な期待になり
/// 誤って fail する。
fn compiled_kernel_count(gemm: &CudaGemm) -> u64 {
    let optional_available_count = [
        gemm.wmma_tf32_available(),
        gemm.wmma_tf32_opt_available(),
        gemm.wmma_tf32_staged_available(),
    ]
    .into_iter()
    .filter(|available| *available)
    .count() as u64;
    // staged swizzle 変種（`wmma_tf32_staged_swizzle_descriptor` / kernel_name
    // "wmma_tf32_staged_swizzle"）は `kernel_specs` に含まれない別カーネル
    // エントリだが、`CudaGemm::new` から `load_function_cached` 経由で
    // ロードされる（`gemm.rs::CudaGemm::new` の swizzle 分岐参照）。
    // 成功時（`wmma_tf32_staged_swizzle_group_width()` が `Some`）はキャッシュへ
    // insert 済みのため、実際にコンパイル・ロードへ成功したカーネル数へ
    // 含めないと 1st/2nd 構築間の miss/hit 期待値がずれる
    // （イシュー #1024 レビュー指摘。Cursor Bugbot Low Severity。
    // PRRT_kwDOTuUCJc6da-Ta）。
    let swizzle_available_count = u64::from(gemm.wmma_tf32_staged_swizzle_group_width().is_some());
    MANDATORY_KERNEL_COUNT + optional_available_count + swizzle_available_count
}

/// [`compiled_kernel_count`] の否定側: TF32 WMMA 系 3 種のうち NVRTC
/// コンパイルに失敗し fail-soft した（`compile_wmma_tf32` 等が `Err` を
/// 返し `*_available()` が `false` になった）カーネル数を返す。
///
/// レビュー指摘対応（イシュー #1024・codex-review。PRRT_kwDOTuUCJc6da6Ul）:
/// `load_function_cached`（`module_cache.rs`）は `KernelModuleCache` への
/// insert（cache 登録）をコンパイル成功時にしか行わないため、fail-soft
/// する optional カーネルは呼び出すたびに 3 段フォールバックの 2 段目
/// （ディスクキャッシュ照合）まで到達したうえで NVRTC 直コンパイルに
/// 失敗する。この経路は `KernelModuleCache::get` の miss（`get` 自体は
/// ヒットしていないため実装上ミスとして計上済み。`module_cache.rs`
/// `load_function_cached` 冒頭「1 段目」参照）を毎回計上するが insert
/// されないため、2 回目の `CudaGemm::new` でも同じ数だけ miss が
/// 増える。したがって 2 回の構築間で `miss_count` の増分がゼロになる
/// のは全 optional カーネルが利用可能な環境のみであり、TF32 WMMA が
/// 未対応（fail-soft）なデバイスでは無条件のゼロ期待は誤って fail する
/// （同一デバイス上では NVRTC コンパイル結果が決定的なため、fail-soft
/// する集合は 1 回目・2 回目で一致する前提）。
fn unavailable_optional_kernel_count(gemm: &CudaGemm) -> u64 {
    let total_optional: u64 = 3;
    let available_count = [
        gemm.wmma_tf32_available(),
        gemm.wmma_tf32_opt_available(),
        gemm.wmma_tf32_staged_available(),
    ]
    .into_iter()
    .filter(|available| *available)
    .count() as u64;
    // staged swizzle 変種の fail-soft miss も加算する（レビュー指摘対応。
    // イシュー #1024・Cursor Bugbot Low Severity。PRRT_kwDOTuUCJc6da-Ta）:
    // `wmma_tf32_staged_swizzle_unavailable_reason()` が `Some` を返すのは
    // `CudaGemm::new` が実際に swizzle 変種の `load_function_cached` 呼び出し
    // を試みた（`wmma_tf32_staged` 本体のコンパイルに成功し `device.
    // multiprocessor_count()` も実測できた）にもかかわらず NVRTC コンパイル
    // に失敗した場合のみ（`gemm.rs::CudaGemm::wmma_tf32_staged_swizzle_unavailable_reason`
    // ドキュメンテーションコメント参照）。この場合、`load_function_cached`
    // は 1 段目（`KernelModuleCache::get`）で miss を計上するが insert
    // されないため、[`unavailable_optional_kernel_count`] の 3 種と同様
    // 2 回目の構築でも同じ miss が再度発生する。base 側が未対応・SM 数が
    // 取得できず試み自体が起きなかった場合（reason・group_width とも
    // `None`）は miss が発生しないため加算しない。
    let swizzle_repeated_miss =
        u64::from(gemm.wmma_tf32_staged_swizzle_unavailable_reason().is_some());
    (total_optional - available_count) + swizzle_repeated_miss
}

/// (a) 同一 `context_cache::cached_device` 上で `CudaGemm::new` を 2 回
/// 構築すると、2 回目は `KernelModuleCache` の `hit_count` が
/// [`compiled_kernel_count`]（実際にコンパイル・ロードへ成功したカーネル
/// 数。TF32 WMMA 系が fail-soft する環境では [`kernel_specs`] の要素数
/// （8）を下回りうる。イシュー #1024 レビュー指摘対応）以上増加し、
/// `miss_count` は増加しない（1 回目のコンパイル・insert で埋まった LRU
/// を 2 回目が再利用する）。
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
    // gemm1 構築中（数百 ms・9 回の NVRTC コンパイル）に他テストが共有 LRU
    // へ並行 insert し、gemm1 自身が insert したエントリを 2nd 構築前に
    // evict してしまう窓を検出するため、テスト開始時点から起点を取る
    // （`evicts_before_second`〈gemm2 構築直前〉だけでは gemm1 構築中の
    // evict を見逃す。イシュー #1107・advisor 指摘）。
    let evicts_before = cache.evict_count();

    let gemm1 = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from CudaGemm::new (1st): {e}"),
    };
    let misses_after_first = cache.miss_count();
    // 1 回目は実際にコンパイル・ロードへ成功したカーネルの数だけ LRU
    // ミス（新規コンパイル）が発生するはず。TF32 WMMA 系が fail-soft した
    // 分（`gemm1` の可用性で判定）は insert 自体が起きないため期待値から
    // 除外する（`compiled_kernel_count` ドキュメンテーションコメント参照）。
    let expected_kernel_count = compiled_kernel_count(&gemm1);
    assert!(
        misses_after_first >= misses_before + expected_kernel_count,
        "1st CudaGemm::new must miss the module cache for every successfully compiled \
         kernel_specs entry: before={misses_before}, after={misses_after_first}, \
         expected_kernel_count={expected_kernel_count}"
    );

    let hits_before_second = cache.hit_count();
    let evicts_before_second = cache.evict_count();
    let gemm2 = CudaGemm::new(&device)
        .expect("2nd CudaGemm::new must succeed given the 1st succeeded on the same device");
    let hits_after_second = cache.hit_count();
    let misses_after_second = cache.miss_count();
    let evicts_after_second = cache.evict_count();
    // 2nd 構築直前から完了までの evict 増分（狭い窓）。共有 LRU（プロセス
    // ワイド `static`）を汚染しうるのは他プロセスではなく同一プロセス内で
    // 並列実行される他の `#[ignore]` テストのみのため、0 でなければ
    // それらとの並列実行が疑われる（本ファイル冒頭「実行時は必ず
    // `--test-threads=1`」節参照）。
    let evicted_during_window = evicts_after_second.saturating_sub(evicts_before_second);
    // テスト開始（1st 構築前）から 2nd 構築完了までの evict 増分（広い窓）。
    // 1st 構築自体が数百 ms・9 回の NVRTC コンパイルを要するため、gemm1 が
    // 自身のエントリを insert している最中に他テストが割り込んで evict
    // する事象は狭い窓（`evicted_during_window`）では捕捉できない。
    // GB10 実測（before=9, after=15）の再現時に狭い窓が 0 でも広い窓が
    // 非ゼロであれば「gemm1 構築中の並行 evict」が原因と切り分けられる
    // （イシュー #1107。advisor 指摘で追加）。
    let evicted_since_start = evicts_after_second.saturating_sub(evicts_before);

    // 2 回目も同一デバイス上の構築のため、TF32 WMMA 系の可用性は 1 回目
    // （`gemm1`）と一致するはず（同一環境・同一 NVRTC コンパイル結果）。
    assert_eq!(
        compiled_kernel_count(&gemm2),
        expected_kernel_count,
        "2nd CudaGemm::new on the same device must compile the same set of kernels as the 1st \
         (WMMA(TF32) availability must not flap across constructions on the same device)"
    );
    assert!(
        hits_after_second >= hits_before_second + expected_kernel_count,
        "2nd CudaGemm::new on the same cached device must hit the module cache for every \
         successfully compiled kernel_specs entry: before={hits_before_second}, \
         after={hits_after_second}, expected_kernel_count={expected_kernel_count}, \
         evicted_during_window={evicted_during_window} (narrow: since just before the 2nd \
         construction), evicted_since_start={evicted_since_start} (wide: since before the 1st \
         construction — the 1st construction itself takes hundreds of ms across 9 NVRTC \
         compiles, so a concurrent test can evict gemm1's own entries before this narrow window \
         even starts). Either being non-zero means the shared static LRU was concurrently \
         polluted by another #[ignore] test in the same process; re-run with \
         `cargo test -p fandhe-ai-backend-cuda --release --lib -- --ignored --test-threads=1` \
         to serialize, per module_cache_wiring_tests.rs's top-of-file doc comment, \
         issue #1107, GB10 measurement 2026-09-03)"
    );
    // fail-soft する optional WMMA(TF32) カーネル（未対応デバイスでは
    // `unavailable_optional_kernel_count(&gemm2)` > 0）は insert
    // されないため（`unavailable_optional_kernel_count` ドキュメンテー
    // ションコメント参照）、2 回目の構築でも同じ数だけ miss が再度
    // 発生しうる。したがって「増分ゼロ」ではなく「増分は fail-soft
    // したカーネル数と一致する」を期待値とする（イシュー #1024
    // レビュー指摘対応。PRRT_kwDOTuUCJc6da6Ul）。
    let expected_repeated_miss_count = unavailable_optional_kernel_count(&gemm2);
    assert_eq!(
        misses_after_second,
        misses_after_first + expected_repeated_miss_count,
        "2nd CudaGemm::new must not introduce new module cache misses beyond the fail-soft \
         optional WMMA(TF32) kernels that never get inserted into the cache: \
         misses_after_first={misses_after_first}, misses_after_second={misses_after_second}, \
         expected_repeated_miss_count={expected_repeated_miss_count}, \
         evicted_during_window={evicted_during_window}, evicted_since_start={evicted_since_start} \
         (see previous assertion's message for the shared static LRU pollution diagnosis; \
         issue #1107)"
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

/// [`kernel_specs`] の index（0-origin）が [`GemmKernelSpec::label`] の
/// どの `Option` 可用性 getter に対応するかを返す。先頭 5 件（必須
/// カーネル）は `None` を返し、無条件に検査対象とする。末尾 3 件
/// （WMMA(TF32) 系フォールバック）は対応する可用性 getter の戻り値を
/// 返し、呼び出し元がコンパイル失敗（fail-soft）分をスキップできるように
/// する（`gemm.rs::kernel_specs` ドキュメンテーションコメント「順序は」
/// 節の固定順序に依存する。イシュー #1024 レビュー指摘対応）。
fn optional_kernel_availability(gemm: &CudaGemm, index: usize) -> Option<bool> {
    match index {
        0..=4 => None,
        5 => Some(gemm.wmma_tf32_available()),
        6 => Some(gemm.wmma_tf32_opt_available()),
        7 => Some(gemm.wmma_tf32_staged_available()),
        _ => unreachable!(
            "kernel_specs() has a fixed length of 8 (gemm.rs::kernel_specs return type); \
             an index outside 0..=7 indicates kernel_specs was extended without updating \
             this mapping"
        ),
    }
}

/// (c) `CudaGemm::new` が 1 回目にコンパイルした各カーネルについて、
/// NVRTC ディスクキャッシュ（`nvrtc.rs::store_cache_entry`）へ実際に
/// エントリが保存されていることを確認する（module_cache（プロセス内
/// LRU）だけでなく、ディスク側の結線も検証する。イシュー #1024）。
///
/// TF32 WMMA 系（[`kernel_specs`] index 5〜7）は NVRTC が拒否しうる
/// フォールバックカーネルであり、コンパイル失敗時はディスクキャッシュへ
/// 一切 insert されない（`compile_wmma_tf32` 等の fail-soft 方針。
/// `optional_kernel_availability` ドキュメンテーションコメント参照）。
/// このため naive/tiled はビルドできるが TF32 WMMA は未対応の CUDA
/// デバイスでは、無条件に 8 エントリ全件の存在を期待すると誤って fail
/// する（イシュー #1024 レビュー指摘。Cursor Bugbot Low Severity）。
/// 本テストは `CudaGemm::new` が返したインスタンスの可用性 getter で
/// 実際にコンパイルへ成功したカーネルだけを検査対象にする。
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

    let gemm = match CudaGemm::new(&device) {
        Ok(g) => g,
        Err(e) if is_environment_unavailable_error(&e) => {
            eprintln!("CUDA/NVRTC 非搭載環境のためスキップ: {e}");
            return;
        }
        Err(e) => panic!("unexpected CudaError from CudaGemm::new: {e}"),
    };

    for (index, spec) in kernel_specs().into_iter().enumerate() {
        // 必須カーネル（先頭 5 件）は None、WMMA(TF32) 系（末尾 3 件）は
        // `Some(available)` を返す。`Some(false)`（このデバイスでは
        // fail-soft によりコンパイル未成立）はディスクエントリが存在
        // しなくて正しいためスキップする。
        if optional_kernel_availability(&gemm, index) == Some(false) {
            eprintln!(
                "kernel `{}` は本デバイスで fail-soft によりコンパイル未成立のためスキップ: \
                 {:?}",
                spec.label,
                match index {
                    5 => gemm.wmma_tf32_unavailable_reason(),
                    6 => gemm.wmma_tf32_opt_unavailable_reason(),
                    7 => gemm.wmma_tf32_staged_unavailable_reason(),
                    _ => None,
                }
            );
            continue;
        }

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
