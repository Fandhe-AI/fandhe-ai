//! CUDA JIT キャッシュ（`nvrtc.rs`）の実機ベンチ: 初回コンパイル時間・
//! 2 回目ロード時間・スループットを計測する（イシュー #534・Phase C-12。
//! 親 #503「NVRTC カーネルの shape 特化・コンパイルキャッシュ・静的タイル
//! 選定」の効果実測）。
//!
//! # なぜ `crates/backend-cuda/tests/`（integration test）ではなく本ファイル
//! （`nvrtc` モジュールの子モジュール）に置くか
//!
//! [`jit_cache_regression_tests`]（イシュー #529・兄弟モジュール。
//! `nvrtc.rs` 末尾の `#[path]` 登録参照）と同じ理由: キャッシュ I/O API
//! （[`super::store_cache_entry_in`]・[`super::load_cache_entry_in`]・
//! [`super::compile_ptx`] 等）はいずれも module-private（`pub(crate)` にすら
//! 満たない）ため、crate 外部と同じ扱いの integration test からは到達
//! できない。`use super::*;` で `nvrtc` 本体の非 `pub` アイテムへ直接
//! 到達する。
//!
//! # 計測対象と「JIT キャッシュ導入前後」の対応付け
//!
//! - **初回コンパイル時間**（キャッシュミス: `compile_ptx` → `store_cache_entry_in`）
//!   ＝「導入前」相当（毎プロセスでコンパイルする従来コスト）
//! - **2 回目ロード時間**（キャッシュヒット: `load_cache_entry_in`）
//!   ＝「導入後」の再起動時コスト
//! - **スループット**: キャッシュ経由でロードした PTX とフレッシュ
//!   コンパイルした PTX から起動した GEMM が同一出力（bit 一致)・
//!   同等性能であることを記録し、キャッシュ導入による性能非後退を示す
//!
//! # 「コンパイル+store 対 warm load」の hard assert を撤去した理由（Review #534）
//!
//! [`jit_cache_bench_cold_compile_vs_warm_load_latency`] はかつて
//! `warm_q.median < cold_total_median` を `assert!` していたが、これは
//! §2.1（[`jit_cache_bench_module_load_and_throughput_parity`] 冒頭）の
//! TFLOPS 非 gating 判断と同じ「GPU クロック挙動・他プロセス競合等の
//! 環境揺らぎをタイミング値の hard assert に持ち込むと flaky 化する」
//! リスクに、TFLOPS と同様にさらされている。NVRTC の `source→PTX`
//! コンパイルはプロセス内キャッシュを持たない設計
//! （`docs/perf/startup-cost-measurement.md`）であり通常はコンパイル
//! 時間がディスク読み込みより桁違いに大きいため実運用上は成立しやすい
//! が、実機ランナー上でのディスク I/O 遅延・NVRTC 初期化揺らぎ等で
//! 逆転しうる余地は理論上残る。よってこの比較も TFLOPS と対称に
//! **記録のみに留め、gating の対象にしない**
//! （`speedup_x` として `println!` に残し `docs/perf/
//! cuda-jit-cache-benchmark.md` へ転記する一次情報とする）。
//!
//! 一方 [`measure_cold_warm_trial`] 内の
//! `assert_eq!(loaded.kernel_ptx, ptx_src, ...)`（ストアしたエントリを
//! 直後にロードした結果がコンパイル直後の PTX と byte 一致すること）は
//! **決定的で環境揺らぎの影響を受けないキャッシュ正当性の検証**であり
//! 撤去しない。この gating は維持されるため、キャッシュ I/O 自体の
//! 破損・不整合は引き続き本テストの失敗として検出される。
//!
//! # C-4（#511）未結線であることの位置づけ（重要）
//!
//! 本番 GEMM ディスパッチ経路（`gemm_auto.rs::CudaGemmAuto::run_f16`）への
//! 「ミス→コンパイル→store→hit」結線（プロセス内 LRU・C-4・#511）は本
//! イシュー時点で未実装（open）。よって本ベンチは実装計画が想定する
//! 「本番経路の前後比較」ではなく、キャッシュ I/O プリミティブを
//! [`jit_cache_regression_tests::get_or_compile`] と同型の直叩きで計測する
//! （実装計画 §2 の設計判断）。本番経路での実効果は C-4 結線後の再計測
//! 事項として PR 本文・引き継ぎに記録する。
//!
//! # なぜ全ケースで実プロダクションカーネルソース 1 種類（[`crate::kernels_mma::mma_f16_source`]）
//! を使い回すか
//!
//! `kernels_mma::RenderedMmaKernel` は生ソース文字列を外部（`kernels_mma`
//! モジュール外）へ返す公開メソッドを一切持たない設計（PR #643
//! codex-review 再々指摘〈P0〉: コンパイル対象ソースと `CudaFunction` を
//! 不可分に束ね、検証を経ないソース差し替え経路を型で塞ぐため）。
//! `jit_cache_bench_tests` は `nvrtc` の子孫であって `kernels_mma` の子孫
//! ではないため、shape 特化構成（`gemm_auto::specialized_mma_config`）が
//! 生成する形状ごとに異なるソース文字列をこの API から取得することは
//! できない（意図された不変条件であり、本イシューのスコープでこの
//! カプセル化を弱めない）。そのため本ベンチは
//! [`jit_cache_regression_tests::sample_source`] と同じ選択で、既定構成
//! （全次元 `Dynamic`）の実プロダクションカーネルソースを全ケース共通で
//! 使う。`descriptor`（[`super::CudaKernelDescriptor`]）の `shape`／
//! `CompiledDims` のみを変えてキャッシュキーの一意性を作り、複数の
//! 「キャッシュエントリ」を区別する目的に限定する。したがって
//! 「初回コンパイル vs 2 回目ロード」の時間差は NVRTC コンパイル対象
//! ソースの複雑度（shape 特化の有無）には依存せず、単一の実カーネル
//! ソースに対する測定であることをここに明記する
//! （`docs/perf/cuda-jit-cache-benchmark.md` にも同じ限定を記載する）。

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use cudarc::driver::PushKernelArg;
use half::f16;
use tensor_core::dispatch::{DType, GemmShape};

use bench_harness::{MeasurementConfig, median_q1_q3, rng::Xorshift64Star};

use super::*;

use crate::device::CudaDevice;
use crate::gemm::validate_gemm_dims;
use crate::gemm_mma::{
    check_min_compute_capability, validate_mma_alignment, validate_mma_grid_bounds,
};
use crate::kernels_mma::{MmaKernelConfig, mma_f16_source};

/// 実機必須ベンチの計測回数（実装計画 §4.1「各 descriptor につき 5 trial」・
/// REQ-8 の 5 回計測中央値方針。`bench_harness::protocol::run` の
/// warmup/iters 下限〈20 回〉とは別に、コンパイル・キャッシュ I/O 自体は
/// 1 回ごとのコストが大きいため、`protocol::run` ではなく本ファイル内で
/// [`median_q1_q3`] を直接使う独自ループとする）。
const TRIALS: usize = 5;

/// [`fresh_temp_dir`] が払い出した一時ディレクトリを `Drop` で確実に
/// 片付ける RAII ガード（Review #534 指摘: 各関数末尾でのみ
/// `fs::remove_dir_all` していたため、その間の `expect()`（十数箇所）で
/// panic すると `/tmp` 配下にディレクトリが残り、実機ランナーで繰り返し
/// 実行される性質上リークが蓄積しうる問題への対処）。`path()` で内側の
/// `PathBuf` を借用し、既存の `fresh_temp_dir` 呼び出し箇所を最小差分で
/// 置き換えられるようにする。
struct TempDirGuard(PathBuf);

impl TempDirGuard {
    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDirGuard {
    fn drop(&mut self) {
        // ベンチ用一時ディレクトリの片付けであり、CI／実機ランナー上の
        // 通常経路では既に空か削除済みのことが多い。削除失敗（他プロセス
        // との競合等）はベンチ自体の成否に関わらないため無視する
        // （`fs::remove_dir_all` の戻り値を握りつぶす既存方針を踏襲）。
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// `jit_cache_regression_tests::fresh_temp_dir` と同型: テスト・ベンチ用に
/// 一意な一時ディレクトリを払い出す（プロセス内 `AtomicU64` カウンタ＋PID
/// で並行実行時の衝突を避ける）。モジュール境界をまたいだ結合を避けるため
/// 独立して定義する（`jit_cache_regression_tests.rs` 冒頭コメント参照）。
/// 戻り値は [`TempDirGuard`] であり、呼び出し元スコープを抜けるときに
/// panic 経路も含めて自動的に片付けられる。
fn fresh_temp_dir(label: &str) -> TempDirGuard {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "rust-ai-library-cache-bench.c12.{label}.{}.{seq}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).expect("failed to create bench temp dir");
    TempDirGuard(dir)
}

/// 決定的シードで GEMM 入力（A・B、f16）を生成する（`specialized_mma_parity.rs::gen_ab`
/// と同じ生成方法）。
fn gen_ab(seed: u64, m: u32, n: u32, k: u32) -> (Vec<f16>, Vec<f16>) {
    let mut rng = Xorshift64Star::new(seed);
    let a: Vec<f16> = rng.fill_vec_f16((m as usize) * (k as usize));
    let b: Vec<f16> = rng.fill_vec_f16((k as usize) * (n as usize));
    (a, b)
}

/// ベンチ対象 1 ケース分の `CudaKernelDescriptor` を構築する。ブロック
/// タイル値（64/128/32）・段数（3）は `MmaKernelConfig::default()`
/// （現行本番構成。`kernels_mma.rs` `MMA_BM`/`MMA_BN`/`MMA_BK`/`MMA_STAGES`）
/// と揃える（実際にコンパイルするソースが `mma_f16_source()`＝この既定
/// 構成の展開結果であるため、descriptor 側の記述と実体を一致させる）。
fn bench_descriptor(
    label: &'static str,
    shape: GemmShape,
    compiled: CompiledDims,
) -> CudaKernelDescriptor {
    CudaKernelDescriptor::new_with_compiled_dims(label, shape, 64, 128, 32, 3, DType::F16, compiled)
        .expect("bench descriptor parameters must satisfy CudaKernelDescriptor invariants")
}

/// `device`・`descriptor`・実カーネルソースからキャッシュキーを構築する。
fn bench_key(
    device: &CudaDevice,
    descriptor: CudaKernelDescriptor,
    source: &str,
) -> CudaKernelCacheKey {
    CudaKernelCacheKey::from_device(
        descriptor,
        device,
        vec![format!("--gpu-architecture={}", device.arch())],
        source.to_string(),
    )
    .expect("CudaKernelCacheKey::from_device must succeed on a real, initialized CUDA device")
}

/// 1 descriptor・1 trial 分の「ミス→コンパイル→store→2 回目ロード」計測。
///
/// [`jit_cache_regression_tests::get_or_compile`] のミス経路（compile_fn
/// 呼び出し→store）と同じ手順を、実際の `compile_ptx`（NVRTC 実コンパイル）
/// を使って実行し区間別に計測する。trial ごとに新規キャッシュルートを
/// 使う（root を跨いだ状態の持ち越しを避け、各 trial を独立したコールド
/// スタートとして扱うため）。
struct ColdWarmSample {
    compile_secs: f64,
    store_secs: f64,
    warm_load_secs: f64,
}

fn measure_cold_warm_trial(
    device: &CudaDevice,
    key: &CudaKernelCacheKey,
    source: &str,
) -> ColdWarmSample {
    // `_root_guard` は関数を抜ける際（panic 経路含む）に一時ディレクトリを
    // 自動で片付ける（`TempDirGuard` の `Drop` 実装参照。Review #534
    // 指摘対応）。以降の `root` 参照はすべて `.path()` 経由。
    let _root_guard = fresh_temp_dir("cold-warm");
    let root = _root_guard.path();

    let t_compile = Instant::now();
    let ptx = compile_ptx(source, device.arch()).expect(
        "NVRTC compile of the production mma_f16 kernel source must succeed on the bench runner",
    );
    let compile_secs = t_compile.elapsed().as_secs_f64();
    let ptx_src = ptx.to_src();

    let t_store = Instant::now();
    store_cache_entry_in(root, key, source, &ptx_src)
        .expect("store_cache_entry_in must succeed against a fresh, writable temp root");
    let store_secs = t_store.elapsed().as_secs_f64();

    let t_load = Instant::now();
    let loaded = load_cache_entry_in(root, key, source)
        .expect("load_cache_entry_in must not error against the entry just stored")
        .expect("a cache hit is expected immediately after store_cache_entry_in in a single-writer trial");
    let warm_load_secs = t_load.elapsed().as_secs_f64();

    assert_eq!(
        loaded.kernel_ptx, ptx_src,
        "cache-loaded PTX must byte-match the freshly compiled PTX (single-writer trial, no race)"
    );

    ColdWarmSample {
        compile_secs,
        store_secs,
        warm_load_secs,
    }
}

/// 受け入れ基準 1（初回コンパイル時間・2 回目ロード時間を 5 回計測中央値
/// で記録する）の本体。`STATIC_MNK`（1024³・4096³）・`DYNAMIC_ALL`
/// （4096³。既定プリセット）の 3 descriptor を対象とする（実装計画
/// §4.1）。`descriptor` の shape/CompiledDims が変わってもコンパイル対象
/// ソースは共通（本ファイル冒頭コメント参照）だが、[`bench_key`] が
/// `descriptor` をハッシュへ含めるため各ケースは独立したキャッシュ
/// エントリとして計測される。
///
/// `docs/perf/startup-cost-measurement.md` の CUDA 実測（`CUDA_CACHE_PATH`
/// ドライバ側 JIT キャッシュ制御下の first_kernel cold/warm 差）とは別
/// レイヤの計測であり、本テストの標準出力は `docs/perf/
/// cuda-jit-cache-benchmark.md` へ転記する一次情報とする。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上）必須。#534"]
fn jit_cache_bench_cold_compile_vs_warm_load_latency() {
    let device =
        CudaDevice::new(0).expect("CUDA device must be available on the ignored bench runner");
    check_min_compute_capability(&device).expect(
        "compute capability must satisfy the mma.sync/ldmatrix/cp.async minimum for this bench",
    );

    let source = mma_f16_source();

    let cases: &[(&'static str, GemmShape, CompiledDims)] = &[
        (
            "c12-static-mnk-1024",
            GemmShape::new(1024, 1024, 1024),
            CompiledDims::STATIC_MNK,
        ),
        (
            "c12-static-mnk-4096",
            GemmShape::new(4096, 4096, 4096),
            CompiledDims::STATIC_MNK,
        ),
        (
            "c12-dynamic-all-4096",
            GemmShape::new(4096, 4096, 4096),
            CompiledDims::DYNAMIC_ALL,
        ),
    ];

    for &(label, shape, compiled) in cases {
        let descriptor = bench_descriptor(label, shape, compiled);
        let key = bench_key(&device, descriptor, source);

        let mut compile_samples = Vec::with_capacity(TRIALS);
        let mut store_samples = Vec::with_capacity(TRIALS);
        let mut warm_samples = Vec::with_capacity(TRIALS);
        // trial ごとの「compile + store」合計（cold 合計）を独立にサンプル
        // 化する（Review #534〈codex-review〉指摘: `compile_q.median +
        // store_q.median`〔区間別中央値の和〕は、compile と store が同一
        // trial で対になっているにもかかわらず「各 trial の cold 合計の
        // 中央値」と一般には一致しない。中央値は線形演算ではないため、
        // trial 単位で対になった値の和をまず作り、その配列へ
        // `median_q1_q3` を適用する必要がある）。
        let mut cold_total_samples = Vec::with_capacity(TRIALS);

        for _ in 0..TRIALS {
            let sample = measure_cold_warm_trial(&device, &key, source);
            compile_samples.push(sample.compile_secs);
            store_samples.push(sample.store_secs);
            warm_samples.push(sample.warm_load_secs);
            cold_total_samples.push(sample.compile_secs + sample.store_secs);
        }

        let compile_q = median_q1_q3(&compile_samples)
            .expect("5 non-NaN compile-time samples must yield quartiles");
        let store_q = median_q1_q3(&store_samples)
            .expect("5 non-NaN store-time samples must yield quartiles");
        let warm_q = median_q1_q3(&warm_samples)
            .expect("5 non-NaN warm-load-time samples must yield quartiles");
        let cold_total_q = median_q1_q3(&cold_total_samples)
            .expect("5 non-NaN cold-total-time samples must yield quartiles");
        let cold_total_median = cold_total_q.median;

        // `--nocapture` 実行時のみ観測される構造化出力（実装計画 §4.1）。
        // 実測値は `docs/perf/cuda-jit-cache-benchmark.md` へ転記する。
        // `warm_faster`／`speedup_x` は記録のみ（本ファイル冒頭ドキュメン
        // テーションコメント「hard assert を撤去した理由」参照）。GPU
        // クロック挙動・他プロセス競合等の環境揺らぎを hard assert に
        // 持ち込まないための safety-side な判断であり TFLOPS の非 gating
        // 方針（[`jit_cache_bench_module_load_and_throughput_parity`]）と
        // 対称にしている。
        println!(
            "[jit_cache_bench:cold_vs_warm] descriptor={label} \
             compile_median_s={:.6} (q1={:.6}, q3={:.6}) \
             store_median_s={:.6} (q1={:.6}, q3={:.6}) \
             cold_total_median_s={cold_total_median:.6} (q1={:.6}, q3={:.6}) \
             warm_load_median_s={:.6} (q1={:.6}, q3={:.6}) \
             speedup_x={:.2} warm_faster={} \
             — record only, non-gating (see module doc comment)",
            compile_q.median,
            compile_q.q1,
            compile_q.q3,
            store_q.median,
            store_q.q1,
            store_q.q3,
            cold_total_q.q1,
            cold_total_q.q3,
            warm_q.median,
            warm_q.q1,
            warm_q.q3,
            cold_total_median / warm_q.median.max(f64::EPSILON),
            warm_q.median < cold_total_median,
        );
    }
}

/// 受け入れ基準 1 の「スループット」節（実装計画 §4.2）: (a) フレッシュ
/// コンパイル PTX と (b) キャッシュ経由でロードした PTX それぞれの
/// モジュールロード時間（5 回計測中央値）を記録し、両者から起動した
/// GEMM（f16・4096³）が bit 一致することを検証する。TFLOPS は記録の
/// みに留め hard assert にしない（実装計画 §8「スループット差の flaky
/// 化」への安全側判断: 環境揺らぎを含む数値をゲート条件にしない）。
///
/// TFLOPS の計測区間は kernel launch + sync のみ（出力バッファ確保・
/// H2D/D2H を含まない）。正当性確認用の D2H は計測区間の外で 1 回だけ
/// 別途実行する（Review #534〈cursor[bot] Medium／codex-review〉指摘:
/// 転送・確保込みの end-to-end 値では兄弟ベンチ
/// `examples/gemm_mma_bench.rs::measure_mma_f16`（`launch_f16` のみを
/// 計測）や `gemm_mma.rs::CudaMmaGemm::launch_f16` と絶対値を比較できない
/// ため、計測境界を揃える）。
#[test]
#[ignore = "CUDA 実機（NVRTC 搭載・compute capability 8.0 以上）必須。#534"]
fn jit_cache_bench_module_load_and_throughput_parity() {
    let device =
        CudaDevice::new(0).expect("CUDA device must be available on the ignored bench runner");
    check_min_compute_capability(&device).expect(
        "compute capability must satisfy the mma.sync/ldmatrix/cp.async minimum for this bench",
    );

    let source = mma_f16_source();
    let (m, n, k) = (4096u32, 4096u32, 4096u32);
    let shape = GemmShape::new(m, n, k);
    let descriptor = bench_descriptor("c12-throughput-4096", shape, CompiledDims::DYNAMIC_ALL);
    let key = bench_key(&device, descriptor, source);

    let ptx_fresh = compile_ptx(source, device.arch()).expect(
        "NVRTC compile of the production mma_f16 kernel source must succeed on the bench runner",
    );

    // `_root_guard` は関数を抜ける際（panic 経路含む）に一時ディレクトリを
    // 自動で片付ける（`TempDirGuard` の `Drop` 実装参照。Review #534
    // 指摘対応）。以降の `root` 参照はすべて `.path()` 経由。
    let _root_guard = fresh_temp_dir("throughput");
    let root = _root_guard.path();
    store_cache_entry_in(root, &key, source, &ptx_fresh.to_src())
        .expect("store_cache_entry_in must succeed against a fresh, writable temp root");
    let cached = load_cache_entry_in(root, &key, source)
        .expect("load_cache_entry_in must not error against the entry just stored")
        .expect("a cache hit is expected immediately after store_cache_entry_in in a single-writer trial");
    assert_eq!(
        cached.kernel_ptx,
        ptx_fresh.to_src(),
        "cache-loaded PTX must byte-match the freshly compiled PTX before the module-load comparison"
    );
    let ptx_cached = Ptx::from_src(cached.kernel_ptx);

    // モジュールロード+シンボル解決時間（5 回計測中央値）。`load_module`
    // （`cuModuleLoadData`。PTX→SASS JIT はここで実行される）に加えて
    // `load_function("gemm_mma_f16")`（`cuModuleGetFunction`。シンボル
    // 解決）まで計測区間に含める。`load_module` はロードごとに独立した
    // `CUmodule` を作るため、同じ `Ptx` を複数回ロードしても直前のロード
    // 結果を再利用しない。区間を `load_module` のみに絞らないのは、
    // cudarc（`cudarc-0.19.8` `driver/safe/core.rs`）の実装上
    // `load_function` の `cuModuleGetFunction` コストが `load_module` と
    // 独立した別ステップであり、「起動可能な `CudaFunction` を得るまで」
    // の実測を優先するため（狭めると `load_function` の寄与が計測から
    // 漏れ、`docs/perf/cuda-jit-cache-benchmark.md` へ転記する一次データ
    // が実際の起動コストを過小評価してしまう）。
    let module_load_and_resolve_secs = |ptx: &Ptx| -> f64 {
        let t = Instant::now();
        let _func = device
            .context()
            .load_module(ptx.clone())
            .expect("load_module must succeed for a PTX this test just compiled/cache-loaded")
            .load_function("gemm_mma_f16")
            .expect("gemm_mma_f16 entry point must be present in the compiled module");
        t.elapsed().as_secs_f64()
    };

    let mut fresh_load_samples = Vec::with_capacity(TRIALS);
    let mut cached_load_samples = Vec::with_capacity(TRIALS);
    // ロード順を trial ごとに交互化する（Review #534〈codex-review〉指摘:
    // 常に fresh を先・cached を後の順で同一 context へロードすると、
    // ドライバの PTX→SASS キャッシュ・初回初期化コスト等の順序依存の
    // 一時的コストが一方（先にロードする側）にのみ乗りうる。
    // `ptx_fresh`／`ptx_cached` は上で byte 一致を assert 済みの同一内容
    // なので、この交互化は「同一入力の比較で順序バイアスを打ち消す」
    // ための決定的な措置であり、乱数は使わない
    // （`docs/spec` REQ-8 の決定的シード方針と同様、計測系は再現可能な
    // 手順を優先する）。奇数 trial では cached を先に測る。
    for i in 0..TRIALS {
        if i % 2 == 0 {
            fresh_load_samples.push(module_load_and_resolve_secs(&ptx_fresh));
            cached_load_samples.push(module_load_and_resolve_secs(&ptx_cached));
        } else {
            let cached_secs = module_load_and_resolve_secs(&ptx_cached);
            let fresh_secs = module_load_and_resolve_secs(&ptx_fresh);
            cached_load_samples.push(cached_secs);
            fresh_load_samples.push(fresh_secs);
        }
    }
    let fresh_load_q = median_q1_q3(&fresh_load_samples)
        .expect("5 non-NaN fresh-PTX module-load samples must yield quartiles");
    let cached_load_q = median_q1_q3(&cached_load_samples)
        .expect("5 non-NaN cached-PTX module-load samples must yield quartiles");

    // ラベルは `load_module`（PTX→SASS JIT）+ `load_function`
    // （シンボル解決）の合算区間であることを明示する（上の
    // `module_load_and_resolve_secs` コメント参照）。
    println!(
        "[jit_cache_bench:module_load] fresh_ptx_load_and_resolve_median_s={:.6} (q1={:.6}, q3={:.6}) \
         cached_ptx_load_and_resolve_median_s={:.6} (q1={:.6}, q3={:.6})",
        fresh_load_q.median,
        fresh_load_q.q1,
        fresh_load_q.q3,
        cached_load_q.median,
        cached_load_q.q1,
        cached_load_q.q3,
    );

    // スループット比較用に、それぞれの PTX から起動可能な `CudaFunction`
    // を 1 個ずつ確保する（上のループはロード時間の計測のみが目的で、
    // ロードした `CudaFunction` を保持しないため別途ロードし直す）。
    let func_fresh = device
        .context()
        .load_module(ptx_fresh.clone())
        .expect("load_module must succeed for the freshly compiled PTX")
        .load_function("gemm_mma_f16")
        .expect("gemm_mma_f16 entry point must be present in the freshly compiled module");
    let func_cached = device
        .context()
        .load_module(ptx_cached.clone())
        .expect("load_module must succeed for the cache-loaded PTX")
        .load_function("gemm_mma_f16")
        .expect("gemm_mma_f16 entry point must be present in the cache-loaded module");

    // 起動前検証: `MmaKernelConfig::default()` は `mma_f16_source()` の
    // 展開元 config（本ファイル冒頭コメント）と一致するため、その
    // `validate_launch_shape`／既存の `validate_mma_alignment`・
    // `validate_mma_grid_bounds`（`gemm_mma.rs`。判定ロジックを複製
    // しない）をそのまま再利用する。
    let cfg = MmaKernelConfig::default();
    cfg.validate_launch_shape(m, n, k)
        .expect("shape (4096,4096,4096) must satisfy the all-Dynamic default config's launch shape contract");
    validate_mma_alignment(n, k).expect("n・k=4096 are both multiples of 8 by construction");
    validate_mma_grid_bounds(m).expect("m=4096 is far below the CUDA grid_dim.y limit");

    let (a, b) = gen_ab(534, m, n, k);
    validate_gemm_dims(a.len(), b.len(), m, n, k)
        .expect("gen_ab must produce host buffers whose lengths match (m,n,k)");

    let stream = device.stream();
    let a_dev = stream
        .clone_htod(&a)
        .expect("H2D transfer of A must succeed on the bench runner");
    let b_dev = stream
        .clone_htod(&b)
        .expect("H2D transfer of B must succeed on the bench runner");
    let launch_config = cfg.launch_config(m, n);
    let (m_i, n_i, k_i) = (m as i32, n as i32, k as i32);

    // `func`（cache-loaded／fresh いずれの経路からロードしたかを問わず、
    // 起動引数の並び・個数は `kernels_mma.rs::CompiledMmaKernel::launch_f16`
    // と同一契約）を起動し完了を待つ（H2D/D2H・出力バッファ確保を含まない
    // 「GPU 実行のみ」の区間。`gemm_mma.rs::CudaMmaGemm::launch_f16` と
    // 同型の SAFETY 根拠・同じ計測境界とする — 兄弟ベンチ
    // `examples/gemm_mma_bench.rs::measure_mma_f16` が `upload_f16`／
    // `alloc_output_f16` を計測区間外で 1 回だけ呼び、`launch_f16` のみを
    // `bench_run` の反復対象にしているのと同じ設計）。D2H は本関数の外で
    // 一度だけ行い、TFLOPS の計測区間には含めない
    // （Review #534〈cursor[bot] Medium／codex-review〉指摘: `alloc_zeros`
    // と 4096² f16 出力の `clone_dtoh` を計測反復ごとにフルに行うと
    // end-to-end 値になり、`launch_f16`／`gemm_mma_bench` の kernel
    // launch + sync のみの絶対値と比較不能になる）。
    let launch_gemm = |func: &cudarc::driver::CudaFunction,
                       c_dev: &mut cudarc::driver::CudaSlice<f16>| {
        // SAFETY: 引数は a_dev/b_dev/c_dev（上で m*k/k*n/m*n 要素として
        // 確保・検証済み）と m_i/n_i/k_i の 5 個・型・個数が、上記
        // `validate_gemm_dims`／`validate_mma_alignment`／
        // `validate_mma_grid_bounds`／`cfg.validate_launch_shape` で
        // 検証済みの m/n/k と 1:1 対応する。grid/block は同じく検証済みの
        // m/n から `cfg.launch_config` が導出したもののみを使う。これは
        // `CompiledMmaKernel::launch_f16`（`kernels_mma.rs`）の SAFETY 根拠
        // と同型であり、本関数はそれを「キャッシュ経由でロードした
        // `CudaFunction` を安全に起動する」ためだけに手動で再現している
        // （`CompiledMmaKernel` はコンパイル経路を自身の `RenderedMmaKernel::
        // compile` に固定しており、外部から得た `Ptx`／`CudaFunction` を
        // 受け付ける構築経路を持たないため。本ファイル冒頭コメント参照）。
        // カーネル内の手動境界チェック（REQ-8）は fresh／cached いずれの
        // 経路でロードしても同一 PTX 内容（本テストが上で bit 一致を
        // assert 済み）のため同一に働く。カーネルのエピローグは C の
        // 各要素へガード付き store を行う（累積しない）ため、`c_dev` を
        // 計測反復間で使い回しても書き込み結果は反復ごとに独立する。
        unsafe {
            stream
                .launch_builder(func)
                .arg(&a_dev)
                .arg(&b_dev)
                .arg(c_dev)
                .arg(&m_i)
                .arg(&n_i)
                .arg(&k_i)
                .launch(launch_config)
                .expect("kernel launch must succeed for a shape validated just above");
        }
        stream
            .synchronize()
            .expect("stream synchronize must succeed on the bench runner");
    };

    // 正当性確認（bit 一致）専用の 1 回限りの実行: 出力バッファを新規
    // 確保し、起動+sync 後に D2H で回収する。TFLOPS 計測区間の外で行う
    // ため、この D2H は絶対 TFLOPS 値には影響しない。
    let run_gemm_and_download = |func: &cudarc::driver::CudaFunction| -> Vec<f16> {
        let mut c_dev = stream
            .alloc_zeros::<f16>((m as usize) * (n as usize))
            .expect("device output buffer allocation must succeed on the bench runner");
        launch_gemm(func, &mut c_dev);
        stream
            .clone_dtoh(&c_dev)
            .expect("D2H transfer of C must succeed on the bench runner")
    };

    let c_fresh = run_gemm_and_download(&func_fresh);
    let c_cached = run_gemm_and_download(&func_cached);
    assert_eq!(
        c_fresh, c_cached,
        "GEMM output launched from the freshly compiled PTX and from the cache-loaded PTX must \
         bit-match (both PTX byte-identical; identical instruction stream must yield identical \
         accumulation order)"
    );

    // スループット（TFLOPS）は記録のみに留め、hard assert の対象にしない
    // （本ファイル冒頭コメント・実装計画 §8「スループット差の flaky 化」
    // への安全側判断）。`bench_harness::protocol::run` の warmup/iters 下限
    // （20 回以上。TASK-8.1）をそのまま満たす既定 `MeasurementConfig` を使う。
    // 計測区間は `launch_gemm`（kernel launch + sync のみ）に限定し、
    // 出力バッファは `bench_run` の外で 1 回だけ確保して反復間で使い回す
    // （`examples/gemm_mma_bench.rs::measure_mma_f16` と同じ設計。上記
    // Review #534 指摘対応）。
    let measurement_cfg = MeasurementConfig::default();
    let flops = 2.0_f64 * f64::from(m) * f64::from(n) * f64::from(k);

    let mut c_dev_fresh = stream
        .alloc_zeros::<f16>((m as usize) * (n as usize))
        .expect("device output buffer allocation must succeed on the bench runner");
    let mut c_dev_cached = stream
        .alloc_zeros::<f16>((m as usize) * (n as usize))
        .expect("device output buffer allocation must succeed on the bench runner");

    let meas_fresh = bench_harness::run(&measurement_cfg, || {
        launch_gemm(&func_fresh, &mut c_dev_fresh);
    })
    .expect("protocol::run must succeed with the default (20/20) measurement config");
    let meas_cached = bench_harness::run(&measurement_cfg, || {
        launch_gemm(&func_cached, &mut c_dev_cached);
    })
    .expect("protocol::run must succeed with the default (20/20) measurement config");

    let tflops_fresh = flops / meas_fresh.median_secs / 1e12;
    let tflops_cached = flops / meas_cached.median_secs / 1e12;

    println!(
        "[jit_cache_bench:throughput] shape=(4096,4096,4096) \
         fresh_tflops={tflops_fresh:.2} (median_s={:.6}) \
         cached_tflops={tflops_cached:.2} (median_s={:.6}) \
         — kernel launch + sync only (excludes buffer alloc/D2H; comparable with \
         gemm_mma_bench::measure_mma_f16), record only, non-gating (see module doc comment)",
        meas_fresh.median_secs, meas_cached.median_secs,
    );

    // `root`（`_root_guard`）は関数終了時に `Drop` で自動的に片付けられる。
}
