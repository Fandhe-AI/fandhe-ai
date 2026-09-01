//! WMMA(TF32) opt（共有メモリ・タイル最適化版）GEMM の CPU-CUDA 数値一致
//! 回帰テスト（TASK-11.1d・#63）。
//!
//! `tests/gemm_wmma_tf32.rs`（#62）と同じ方針で、判定式・閾値は
//! `fandhe_ai_backend_cpu::assert_parity`（統一複合判定「相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満」の唯一の実体）に一本化し、ここでローカル複製しない
//! （`.claude/rules/coding-rust.md`）。
//!
//! **公開 API との関係**: `CudaGemm::run_wmma_tf32` は opt カーネル・
//! opt-staged カーネルが `CudaGemm::new` 時点でコンパイル・ロードに成功
//! していれば自動的にそちらを選ぶ 3 段選択（`gemm.rs::run_wmma_tf32`
//! ドキュメンテーションコメント参照。専用の切替 API は存在しない。REQ-11）。
//! 本ファイルは `run_wmma_tf32` が実機で実際に選んだ経路（staged が
//! 利用可能なら staged、そうでなければ opt、それも不可なら basic）が
//! CPU 参照実装と一致することを検証する「ルーティング後の実効経路」の
//! parity 検査であり、opt カーネル単独の形状網羅は担わない
//! （下記「イシュー #500 でのルーティング変更に関する注記」参照）。
//!
//! **イシュー #500 でのルーティング変更に関する注記（PR #678 codex-review
//! P1 指摘対応・PR #678 再指摘対応で移設完了）**: #500 で TF32 opt-staged
//! カーネルが追加され、`run_wmma_tf32` は staged カーネルが利用可能かつ
//! cp.async 16 バイト整列条件（`n%4==0 && k%4==0`）を満たす形状では staged
//! 経路を最優先で選ぶ。本ファイルの形状（64×64×64・128×128×128・
//! 512×512×512・512×512×4096 等）はいずれも 4 の倍数であるため、staged
//! カーネル実装済み環境の実機では実際には staged 経路を通り、opt カーネル
//! 固有のタイル境界を踏まなくなる。数値一致（parity）検査自体は「その時点
//! で選ばれた経路が CPU 参照実装と一致するか」を見るものであり実行経路に
//! 依らず有効だが（下記
//! `wmma_tf32_routed_path_matches_reference_across_shapes`／
//! `wmma_tf32_routed_path_k4096_stress` が担う）、opt カーネル**単独**の
//! 形状網羅・回帰検出はこのファイルでは保証できないため
//! `fandhe_ai_backend_cuda::gemm::tests::wmma_tf32_opt_kernel_matches_reference_across_shapes`・
//! `wmma_tf32_opt_kernel_k4096_stress`・
//! `wmma_tf32_opt_kernel_parity_does_not_regress`（いずれも `src/gemm.rs`。
//! private field 経由で 3 段選択を経由せず opt カーネルを強制実行する）へ
//! 移設した（`docs/perf/cuda-parity-baseline.md` §3 参照）。
//!
//! **実機依存の分離**: `tests/gemm_wmma_tf32.rs` と同じ分岐パターン
//! （環境適応スモークのみ通常 CI で実行、CUDA/NVRTC 非搭載環境では早期
//! return で green）。

use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaGemm};

/// 決定的シードで A・B（f32）を生成し、CPU 参照実装と `run_wmma_tf32`
/// （3 段選択で実機が実際に選んだ経路——staged・opt・basic のいずれか）の
/// 出力を [`fandhe_ai_backend_cpu::assert_parity`] で照合する。
fn assert_wmma_tf32_opt_parity(gemm: &CudaGemm, context: &str, seed: u64, m: u32, n: u32, k: u32) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");

    let c_gpu = gemm
        .run_wmma_tf32(&a, &b, m, n, k)
        .expect("CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test runner");

    fandhe_ai_backend_cpu::assert_parity(context, &c_gpu, &c_ref);
}

/// 環境適応型のスモークテスト（`#[ignore]` なし。通常 CI で実行）。
/// `tests/gemm_wmma_tf32.rs::wmma_tf32_parity_smoke_env_adaptive` と同じ
/// 分岐パターン。opt カーネルのブロックタイル 1 個ぶん（64×64×64）で
/// 複合判定を実施する（opt が未対応環境ではコンパイル失敗により基本版
/// へ自動フォールバックするため、このスモークは opt 専用の分岐を持たない
/// ——`run_wmma_tf32` のフォールバック方針どおり、どちらの経路でも
/// 複合判定は成立することを確認できればよい）。
#[test]
fn wmma_tf32_opt_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => return,
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    let gemm = match CudaGemm::new(&device) {
        Ok(gemm) => gemm,
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaGemm::new: {other}"),
    };

    match gemm.run_wmma_tf32(&[0.0; 4096], &[0.0; 4096], 64, 64, 64) {
        Ok(_) => assert_wmma_tf32_opt_parity(&gemm, "smoke 64x64x64", 1, 64, 64, 64),
        Err(CudaError::WmmaUnavailable { detail }) => {
            // opt・基本版とも使用不能な環境（`<mma.h>` 未解決・cc<8.0）。
            // naive/tiled は道連れにならない（`gemm.rs::CudaGemm::new`
            // ドキュメンテーションコメント参照）。
            assert!(!detail.is_empty());
        }
        Err(other) => panic!("unexpected CudaError variant from run_wmma_tf32: {other}"),
    }
}

/// 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須の形状網羅
/// テスト。`run_wmma_tf32`（公開 API・3 段選択）が実機で実際に選んだ経路
/// （staged 利用可能なら staged、そうでなければ opt、それも不可なら
/// basic）が CPU 参照実装と一致することを検証する。
///
/// **PR #678 codex-review P1 再指摘対応で改名**: 旧名
/// `wmma_tf32_opt_matches_reference_across_shapes` は、本テストの形状が
/// 4 の倍数（cp.async 整列条件を満たす）であるため実機では staged 経路を
/// 通ることを踏まえると誤解を招く名称だった（「opt を検査している」という
/// 誤った期待を持たせる）。opt カーネル**単独**の形状網羅は
/// `fandhe_ai_backend_cuda::gemm::tests::wmma_tf32_opt_kernel_matches_reference_across_shapes`
/// （`src/gemm.rs`）へ移設済み（本ファイル冒頭のモジュールコメント参照）。
/// 本テストは「実効経路の parity」を検査する別軸の検査として維持する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_routed_path_matches_reference_across_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    // PR #256 レビュー指摘対応の踏襲（PR #678 codex-review P1 再指摘対応で
    // 条件を修正・PR #678 codex-review Medium 再指摘対応でゲートを形状
    // ごとの判定へ強化）: `run_wmma_tf32` は staged・opt の両方が使用不能な
    // 環境で基本版へ自動フォールバックするため、この検証が WMMA 経路
    // （staged・opt のいずれか）を実際に踏んだことを保証するには、basic
    // へのフォールバックが起きていないことを事前に確認する必要がある。
    // `wmma_tf32_staged_available() || wmma_tf32_opt_available()` という
    // gemm 単位（形状非依存）のゲートは、staged が利用可能でも整列非対応
    // 形状（63×65×33・1×1×1 等）では staged 経路を選ばず、opt が未対応な
    // 環境ではその形状だけ静かに basic へフォールバックしても検出できない
    // （Weak routed-path availability gate 指摘）。
    // `wmma_tf32_routed_path_available(n, k)`（`run_wmma_tf32` の 3 段
    // 選択ロジックを形状ごとに再現。`gemm.rs` ドキュメンテーションコメント
    // 参照）で各ケースごとに WMMA 経路が実際に選ばれることを確認してから
    // parity を検証する。
    let cases: &[(u32, u32, u32)] = &[
        (64, 64, 64),
        (128, 128, 128),
        (512, 512, 512),
        (63, 65, 33),
        (65, 63, 17),
        (64, 96, 256),
        (1, 1, 1),
    ];
    for (idx, &(m, n, k)) in cases.iter().enumerate() {
        assert!(
            gemm.wmma_tf32_routed_path_available(n, k),
            "staged or opt WMMA(TF32) kernel must be available for shape m={m} n={n} k={k} on \
             this ignored test runner so that the routed path actually exercises a WMMA path \
             rather than silently falling back to the basic kernel (staged reason: {:?}, opt \
             reason: {:?})",
            gemm.wmma_tf32_staged_unavailable_reason(),
            gemm.wmma_tf32_opt_unavailable_reason()
        );
        let context = format!("routed path shape m={m} n={n} k={k}");
        assert_wmma_tf32_opt_parity(&gemm, &context, 3000 + idx as u64, m, n, k);
    }
}

/// K 大のストレスケース（受け入れ条件の性能比較対象と同一形状。
/// PoC-v2-3 の M=N=K=4096 と揃える）。上記
/// [`wmma_tf32_routed_path_matches_reference_across_shapes`] と同じ理由で
/// 改名（旧名 `wmma_tf32_opt_k4096_stress`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_routed_path_k4096_stress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    // PR #256 レビュー指摘対応の踏襲（上記
    // wmma_tf32_routed_path_matches_reference_across_shapes と同じ根拠。
    // PR #678 codex-review Medium 再指摘対応で形状ごとのゲートへ強化）。
    // 本テストの 2 形状はいずれも 4 の倍数（cp.async 整列条件を満たす）
    // だが、gemm 単位のゲートは形状変更時に静かに緩む余地を残すため、
    // `wmma_tf32_routed_path_available(n, k)` で形状ごとに検証する
    // （上記関数と同じ理由）。
    for &(m, n, k, seed, label) in &[
        (512u32, 512u32, 4096u32, 0xC0FFEEu64, "512x512x4096"),
        (4096, 4096, 4096, 0xBEEF, "4096x4096x4096"),
    ] {
        assert!(
            gemm.wmma_tf32_routed_path_available(n, k),
            "staged or opt WMMA(TF32) kernel must be available for shape m={m} n={n} k={k} on \
             this ignored test runner (staged reason: {:?}, opt reason: {:?})",
            gemm.wmma_tf32_staged_unavailable_reason(),
            gemm.wmma_tf32_opt_unavailable_reason()
        );
        assert_wmma_tf32_opt_parity(
            &gemm,
            &format!("routed path K4096 stress {label}"),
            seed,
            m,
            n,
            k,
        );
    }
}

/// **診断専用（イシュー #1106・GB10 全件洗い出し。PR 用の一時コード。
/// 修正確定後に削除する）。**
///
/// [`wmma_tf32_routed_path_matches_reference_across_shapes`]・
/// [`wmma_tf32_routed_path_k4096_stress`] は `assert_wmma_tf32_opt_parity`
/// （内部で `assert_parity` を呼ぶ厳密ゼロ fail 判定）を使うため、最初に
/// fail するケースで panic して以降のケースを実行しない。GB10 実機実測
/// （2026-09-01 の直列 `--ignored` 実行）では両テストとも先頭ケース
/// （64×64×64 seed=3000・512×512×4096 seed=0xC0FFEE）のみが実測済みで、
/// いずれも `ParityBaseline::BASELINES`（`ParityPath::WmmaTf32Opt`）の
/// 既存記録値と一致した（opt 強制経路・#995 の bit-identical 性を踏まえる
/// と、公開 API `run_wmma_tf32` が実際に選ぶ経路〈staged／opt〉でも同じ
/// 値になる可能性が高いが、本テストは実測でこれを確認する）。残り
/// 7 ケース（128×128×128・512×512×512・63×65×33・65×63×17・64×96×256・
/// 1×1×1・4096×4096×4096 seed=0xBEEF）は未計測のまま。本テストは同じ
/// 全ケース・同じシードを `run_wmma_tf32`（公開 API・3 段選択そのまま。
/// ルーティング検証の意図を保つため private 強制経路は使わない）経由で
/// `fandhe_ai_backend_cpu::compare`（panic しない集計 API）で走らせ、
/// `fail_count`/`total`・`max_abs_diff`・`max_rel_err`・`mean_abs_diff` を
/// 標準出力へ表として出す。**assert は一切行わない**（常に成功する。
/// 目的は数値の可視化のみ）。
#[test]
#[ignore = "診断専用（イシュー #1106）。CUDA 実機（DGX Spark GB10 等、\
            compute capability 8.0 以降）必須。修正確定後に削除する"]
fn wmma_tf32_routed_path_parity_diagnostic_dump_issue_1106() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    let mut cases: Vec<(String, u64, u32, u32, u32)> = [
        (64u32, 64u32, 64u32),
        (128, 128, 128),
        (512, 512, 512),
        (63, 65, 33),
        (65, 63, 17),
        (64, 96, 256),
        (1, 1, 1),
    ]
    .into_iter()
    .enumerate()
    .map(|(idx, (m, n, k))| {
        (
            format!("across_shapes[{idx}] m={m} n={n} k={k}"),
            3000 + idx as u64,
            m,
            n,
            k,
        )
    })
    .collect();
    cases.push((
        "k4096_stress 512x512x4096".to_string(),
        0xC0FFEE,
        512,
        512,
        4096,
    ));
    cases.push((
        "k4096_stress 4096x4096x4096".to_string(),
        0xBEEF,
        4096,
        4096,
        4096,
    ));

    println!(
        "{:<34} {:>10} {:>16} {:>12} {:>12} {:>12} {:>8}",
        "case", "seed", "fail/total", "max_abs", "max_rel", "mean_abs", "routed"
    );
    for (label, seed, m, n, k) in &cases {
        let (m, n, k) = (*m, *n, *k);
        assert!(
            gemm.wmma_tf32_routed_path_available(n, k),
            "staged or opt WMMA(TF32) kernel must be available for shape m={m} n={n} k={k} on \
             this ignored test runner (staged reason: {:?}, opt reason: {:?})",
            gemm.wmma_tf32_staged_unavailable_reason(),
            gemm.wmma_tf32_opt_unavailable_reason()
        );
        let routed_label = if gemm.wmma_tf32_routed_path_is_staged(n, k) {
            "staged"
        } else {
            "opt"
        };

        let mut rng = bench_harness::rng::Xorshift64Star::new(*seed);
        let a = rng.fill_vec((m as usize) * (k as usize));
        let b = rng.fill_vec((k as usize) * (n as usize));

        let mut c_ref = vec![0.0f32; (m as usize) * (n as usize)];
        fandhe_ai_backend_cpu::matmul_reference_fma(
            &a, &b, &mut c_ref, m as usize, n as usize, k as usize,
        )
        .expect("matmul_reference_fma shape validation must pass for well-formed test input");

        let c_gpu = gemm
            .run_wmma_tf32(&a, &b, m, n, k)
            .expect("CudaGemm::run_wmma_tf32 must succeed on this ignored test runner");

        let report = fandhe_ai_backend_cpu::compare(&c_gpu, &c_ref)
            .expect("gpu output and reference must have matching lengths");

        println!(
            "{label:<34} {seed:>#10x} {fail:>7}/{total:<8} {max_abs:>12.3e} {max_rel:>12.3e} \
             {mean_abs:>12.3e} {routed_label:>8}",
            fail = report.fail_count,
            total = report.total,
            max_abs = report.max_abs_diff,
            max_rel = report.max_rel_err,
            mean_abs = report.mean_abs_diff,
        );
    }
}

/// `k == 0`（`num_k_tiles == 0` 経路）で C が全 0 になることを確認する
/// （`tests/gemm_wmma_tf32.rs::wmma_tf32_zero_k_returns_all_zero` の opt
/// 経路版。opt/基本どちらが選ばれても早期 return の契約は共通
/// （`gemm.rs::run_wmma_tf32_opt_kernel`／`run_wmma_f32_kernel` 参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_opt_zero_k_returns_all_zero() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    let (m, n, k) = (4u32, 4u32, 0u32);
    let c = gemm
        .run_wmma_tf32(&[], &[], m, n, k)
        .expect("k==0 must be a valid no-accumulation shape, not a launch error");
    assert_eq!(c.len(), (m as usize) * (n as usize));
    assert!(c.iter().all(|&v| v == 0.0), "k==0 output must be all zero");
}

/// m==0／n==0 の no-op 形状（`tests/gemm_wmma_tf32.rs::
/// wmma_tf32_zero_dim_shape_returns_empty_without_launch` の opt 経路版）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn wmma_tf32_opt_zero_dim_shape_returns_empty_without_launch() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");

    let c = gemm
        .run_wmma_tf32(&[], &[1.0, 2.0, 3.0, 4.0], 0, 4, 1)
        .expect("m==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());

    let c = gemm
        .run_wmma_tf32(&[1.0, 2.0], &[], 2, 0, 1)
        .expect("n==0 must be treated as a no-op, not a driver launch error");
    assert!(c.is_empty());
}

/// #63 受け入れ条件の本体: opt 経路（`run_wmma_tf32`。opt カーネルが利用
/// 可能な環境では自動的にこちらが選ばれる）が tiled f32 実装（1.832
/// TFLOPS、PoC-v2-3、M=N=K=4096）を上回ることを、同一実行内で 5 回
/// 計測した中央値で確認する（`.claude/rules/coding-rust.md`「ベンチは
/// 5 回計測の中央値」）。デバイス常駐バッファへの転送は計測対象に含めず、
/// 「起動→`stream.synchronize()`」のみを計測区間とする方針は
/// `bench-harness` の同期方式・PoC-v2-3 と同じだが、本テストは
/// `CudaGemm` の公開 API（ホスト⇔デバイス転送込みの `run_*`）越しに実測
/// する簡易版であり、転送コストを含む分 TFLOPS はデバイス常駐計測より
/// 保守的な値になる（受け入れ条件は「tiled を上回る」ことであり、両経路
/// とも同じ転送コストを負担するため相対比較としては妥当）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須。実測記録は docs/perf/cuda-tensor-core-measurement.md"]
fn wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096() {
    use std::time::Instant;

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    // PR #256 レビュー指摘（chatgpt-codex-connector「Require the optimized
    // kernel in the optimized benchmark」）対応: `run_wmma_tf32` は opt
    // カーネルのコンパイル・ロードに失敗すると基本版へ自動フォールバック
    // するため、この事前チェックなしでは opt カーネルが一度も実行されない
    // まま本テスト（および parity テスト）が green になりうる。opt カーネル
    // の可用性をここで断定し、フォールバックを起こさせない
    // （`gemm.rs::CudaGemm::wmma_tf32_opt_available` ドキュメンテーション
    // コメント参照）。
    assert!(
        gemm.wmma_tf32_opt_available(),
        "opt kernel must be available on this ignored test runner so that the TFLOPS \
         comparison actually exercises the optimized kernel rather than silently falling \
         back to the basic WMMA kernel (reason: {:?})",
        gemm.wmma_tf32_opt_unavailable_reason()
    );

    let (m, n, k) = (4096u32, 4096u32, 4096u32);
    let mut rng = bench_harness::rng::Xorshift64Star::new(0xACE1);
    let a = rng.fill_vec((m as usize) * (k as usize));
    let b = rng.fill_vec((k as usize) * (n as usize));

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);

    let median_tflops = |run: &dyn Fn() -> Vec<f32>| -> f64 {
        // warmup（NVRTC JIT・クロック遷移の影響を計測から除外する）。
        let _ = run();
        let mut samples = Vec::with_capacity(5);
        for _ in 0..5 {
            let start = Instant::now();
            let _ = run();
            samples.push(start.elapsed().as_secs_f64());
        }
        samples.sort_by(|x, y| x.partial_cmp(y).expect("elapsed seconds must not be NaN"));
        let median = samples[samples.len() / 2];
        (flops / median) / 1e12
    };

    let tiled_tflops = median_tflops(&|| {
        gemm.run_tiled_f32(&a, &b, m, n, k)
            .expect("tiled f32 must succeed on CUDA-equipped test runner")
    });
    let opt_tflops = median_tflops(&|| {
        gemm.run_wmma_tf32(&a, &b, m, n, k)
            .expect("run_wmma_tf32 must succeed on CUDA-equipped test runner")
    });

    assert!(
        opt_tflops > tiled_tflops,
        "opt 経路（{opt_tflops:.3} TFLOPS）が tiled f32（{tiled_tflops:.3} TFLOPS）を \
         上回りませんでした（受け入れ条件: PoC-v2-3 参考値 1.832 TFLOPS 超過）"
    );
}
