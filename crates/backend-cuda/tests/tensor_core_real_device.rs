//! Tensor Core（WMMA TF32／WMMA f16）経路の実機実測・数値一致検証
//! （TASK-11.1e・#64）。
//!
//! ## 位置づけ
//!
//! TASK-11.1a〜d（#60〜#63）で WMMA(TF32)／WMMA(f16) の基本・opt カーネルと
//! 経路別 parity テスト（`tests/gemm_wmma_tf32.rs`・`tests/gemm_wmma_tf32_opt.rs`・
//! `tests/gemm_wmma.rs`・`tests/cpu_cuda_wmma_parity.rs`・`tests/gemm_wmma_f16_opt.rs`）
//! は整備済みだが、それらは経路単体の検証に閉じており「Tensor Core 経路が
//! tiled f32 基準に対して TFLOPS で優位である」ことと「複合判定通過」を
//! 1 回の実機実行で横断的に記録する導線が存在しなかった（#64 受け入れ条件
//! 「実機実測記録（TFLOPS・複合判定通過）が残されている」）。
//!
//! 本ファイルは以下 2 点を提供する:
//!
//! 1. [`tensor_core_tflops_record`]: tiled f32／WMMA TF32（opt）／WMMA f16
//!    （opt）3 経路の `bench_harness::protocol::run`（warmup 20・計測 20、
//!    正本 TASK-8.1 準拠）による計測を 1 テスト内で行い、
//!    `bench_harness::report::BenchReport::to_json` で構造化出力する
//!    （`--nocapture` 実行で `docs/perf/cuda-tensor-core-measurement.md` の
//!    記録テンプレートへ転記できる形式）。
//! 2. [`tensor_core_parity_record`]: TF32・f16 経路の複合判定
//!    （`fandhe_ai_backend_cpu::assert_parity`。REQ-2 統一複合判定「相対誤差 1e-3
//!    未満 または 絶対誤差 1e-5 未満」の唯一の実体）通過を記録用に明示
//!    出力する。判定式・閾値はここでローカル複製しない
//!    （`.claude/rules/coding-rust.md`）。
//!
//! ## 実機前提
//!
//! 両テストとも実機（DGX Spark GB10 等、compute capability 7.0 以降）必須の
//! `#[ignore]` 分離テストであり、通常 CI（GitHub ホステッド・CUDA 実機なし）
//! では実行されない（`.claude/rules/ci.md`「実機依存」）。CUDA デバイス・
//! opt カーネルが利用できない環境では `.expect` により失敗が顕在化する
//! 設計とし、実機以外での silent green を許さない（既存 `tests/
//! gemm_wmma_tf32_opt.rs` の `#[ignore]` テスト群と同じ規約）。
//!
//! opt カーネルの可用性は `wmma_tf32_opt_available`／`wmma_f16_opt_available`
//! で事前に断定してから計測する（PR #256 レビュー指摘「opt 経路の可用性を
//! 断定せず計測すると基本版へのサイレントフォールバックで green になり
//! うる」への対処。`gemm.rs::CudaGemm::wmma_tf32_opt_available`・
//! `gemm_wmma.rs::CudaWmmaGemm::wmma_f16_opt_available` ドキュメンテーション
//! コメント参照）。
//!
//! ## カーネル単体計測への切り替え（イシュー #1123 是正）
//!
//! 旧プロトコルは `run_tiled_f32`／`run_wmma_tf32`／`run_f16`（転送＋
//! カーネル実行の合算計測）から dtype 別の転送のみ計測を差し引いた
//! 「計算のみ」の時間で TFLOPS を比較していた（PR #258 レビュー指摘
//! 「f16 benchmark rewards smaller transfers」対応。f32 系は 4 byte/要素・
//! f16 系は 2 byte/要素で転送バイト数が異なるための補正）。GB10 実機実測
//! （2026-09-03）で、この減算プロトコルは大容量バッファ（4096 で 1 個
//! あたり 32 MB 超）の per-call アロケーション＋転送が二峰性（0.263 s／
//! 0.275 s と大きく乖離）を示す既知病態
//! （`docs/perf/cuda-wmma-f16-perf-triage.md`）により破綻することが
//! 判明した。
//!
//! 本ファイルは各経路の常駐バッファ API（`CudaGemm::upload_f32`／
//! `alloc_output_f32`／`launch_tiled_f32`／`launch_wmma_tf32`、
//! `CudaWmmaGemm::upload_f16`／`alloc_output_f16`／`launch_f16`、いずれも
//! `synchronize`）で H2D/D2H・バッファ確保を計測区間の外へ完全に排除した
//! 「カーネル単体」（`launch → synchronize` のみ）計測へ切り替える。
//! 転送バイト数差の補正（差し引き）は、転送そのものを計測区間に含めない
//! ことでより根本的に解消されるため、dtype 別の転送のみ計測・減算・
//! 「転送のみ ≥ 合算」プロトコル整合性検査は行わない。
//!
//! ## f16 経路の本番 auto 経路追従（イシュー #1160）
//!
//! #1160 で `CudaGemmAuto::run_f16` の `MatrixUnit` 分岐 mma 優先化
//! （`gemm_auto::select_f16_matrix_unit_impl`）自体は §5.6 設計どおり
//! 実装済みだが、mma 優先を有効化する `gemm_auto::
//! MMA_PRIORITY_PRODUCTION_ENABLED` は K=4096 非後退ゲートの `MmaF16`
//! baseline ceiling 未承認（PR #1179 codex-review 指摘）により
//! **`false`（wmma 優先・#1156 以前と同じ従来経路）のまま保留**して
//! いる（`gemm_auto.rs` 該当 docblock 参照）。本ファイルの f16 assert
//! （Tensor Core 経路が tiled f32 を上回ること）が比較する対象カーネル
//! は、本番が実際に選ぶ実装へ追従させる。選択器（`f16_matrix_unit_impl`
//! 診断アクセサ。`internal-diagnostics` feature 限定）が返す実装
//! （`Mma` なら `CudaMmaGemm`、`Wmma` なら `CudaWmmaGemm`（opt））の
//! カーネル単体計測を「本番 f16 経路」の代表値として assert に使う
//! （`gemm_auto::select_f16_matrix_unit_impl` の判定順序と同じ fail-safe
//! 優先順位）。同 feature が無効なビルドでは診断アクセサへ到達できない
//! ため、本番既定（`MMA_PRIORITY_PRODUCTION_ENABLED = false`）に基づき
//! `Wmma` を前提に固定する（実機実行は `make test-ignored-cuda` の
//! `--all-features` に限られるため、この固定値が実際に使われることは
//! ない）。診断アクセサが有効な場合はさらに、選択器が実際に選ぶ実装と
//! 本テストが計測する実装が食い違わないことを fail-closed に検査する
//! （選択器が選ばない経路を黙って計測し続けて本番経路との乖離を見逃す
//! false-green を防ぐ）。`wmma_f16_opt` は本番既定では現在も主経路
//! （`MMA_PRIORITY_PRODUCTION_ENABLED = true` へ復帰するまでの間）で
//! あり続けるため assert 対象のまま扱う（`docs/perf/
//! cuda-wmma-f16-perf-triage.md` §8「`wmma_f16_opt` の扱い」参照）。

use bench_harness::{BenchReport, MeasurementConfig};
use fandhe_ai_backend_cuda::{CudaDevice, CudaGemm, CudaGemmAuto, CudaMmaGemm, CudaWmmaGemm};
use half::f16;

/// TFLOPS 実測記録の本体（#64 受け入れ条件の TFLOPS 実測分）。
///
/// M=N=K=4096（`tests/gemm_wmma_tf32_opt.rs::
/// wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096` と同一形状。PoC-v2-3
/// 参考値 1.832 TFLOPS と比較可能にする）で tiled f32・WMMA TF32・WMMA
/// f16 の 3 経路を計測し、TFLOPS 換算値と `BenchReport` の JSON を
/// `println!` で出力する。`--nocapture` での実行結果を
/// `docs/perf/cuda-tensor-core-measurement.md` の記録テンプレートへ
/// 転記する運用（本ファイル冒頭コメント参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 7.0 以降）必須。実測記録は docs/perf/cuda-tensor-core-measurement.md"]
fn tensor_core_tflops_record() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");

    // 記録 doc の「計測環境」節へ転記する素材。
    println!(
        "environment: name={:?} compute_capability={:?} arch={:?}",
        device.name(),
        device.compute_capability(),
        device.arch()
    );

    let gemm = CudaGemm::new(&device).expect("CudaGemm::new (tiled/WMMA TF32) must succeed");
    assert!(
        gemm.wmma_tf32_opt_available(),
        "WMMA TF32 opt kernel must be available on this ignored test runner so that the \
         TFLOPS record actually exercises the optimized kernel rather than silently falling \
         back to the basic WMMA kernel (reason: {:?})",
        gemm.wmma_tf32_opt_unavailable_reason()
    );

    let wmma_gemm = CudaWmmaGemm::new(&device).expect("CudaWmmaGemm::new (WMMA f16) must succeed");
    assert!(
        wmma_gemm.wmma_f16_opt_available(),
        "WMMA f16 opt kernel must be available on this ignored test runner so that the \
         TFLOPS record actually exercises the optimized kernel rather than silently falling \
         back to the basic WMMA kernel (reason: {:?})",
        wmma_gemm.wmma_f16_opt_unavailable_reason()
    );

    let (m, n, k) = (4096u32, 4096u32, 4096u32);
    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let config = MeasurementConfig::new(20, 20).expect("20/20 must satisfy TASK-8.1 minimums");

    let mut rng_f32 = bench_harness::rng::Xorshift64Star::new(0xACE1);
    let a_f32 = rng_f32.fill_vec((m as usize) * (k as usize));
    let b_f32 = rng_f32.fill_vec((k as usize) * (n as usize));

    let mut rng_f16 = bench_harness::rng::Xorshift64Star::new(0xBEEF01);
    let a_f16: Vec<f16> = rng_f16.fill_vec_f16((m as usize) * (k as usize));
    let b_f16: Vec<f16> = rng_f16.fill_vec_f16((k as usize) * (n as usize));

    // カーネル単体（H2D/D2H・バッファ確保を計測区間の外に置く）計測。
    // イシュー #1123 是正（本ファイル冒頭コメント参照）: 常駐バッファ
    // API（`upload_f32`／`alloc_output_f32`／`launch_*`／`synchronize`）で
    // 「launch → synchronize」区間のみを計測する。転送バイト数差
    // （f32 4 byte/要素・f16 2 byte/要素）は転送そのものを計測区間の
    // 外に置くことで解消済みのため、dtype 別の転送のみ計測・減算は
    // 行わない。

    // tiled f32（基準経路。PoC-v2-3 参考値 1.832 TFLOPS）。
    let (a_f32_dev, b_f32_dev) = gemm
        .upload_f32(&a_f32, &b_f32)
        .expect("upload_f32 must succeed on CUDA-equipped test runner");
    let mut c_tiled_dev = gemm
        .alloc_output_f32(m, n)
        .expect("alloc_output_f32 must succeed on CUDA-equipped test runner");
    let tiled_kernel_only = bench_harness::run(&config, || {
        gemm.launch_tiled_f32(&a_f32_dev, &b_f32_dev, &mut c_tiled_dev, m, n, k)
            .expect("launch_tiled_f32 must succeed on CUDA-equipped test runner");
        gemm.synchronize()
            .expect("synchronize must succeed on CUDA-equipped test runner");
    })
    .expect("tiled f32 kernel-only measurement must satisfy TASK-8.1 protocol");
    let tiled_report = BenchReport::from_measurement(
        "gemm_tiled_f32_kernel_only_4096",
        "cuda",
        &tiled_kernel_only,
    )
    .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");
    let tiled_kernel_tflops = (flops / tiled_kernel_only.median_secs) / 1e12;
    println!(
        "tflops_record_kernel_only path=tiled_f32 tflops={tiled_kernel_tflops:.3} report={}",
        tiled_report
            .to_json()
            .expect("BenchReport::to_json must succeed for a validated report")
    );

    // WMMA TF32（opt もしくは opt-staged。上記 assert は opt 版の可用性の
    // みを保証しており、`launch_wmma_tf32` の 3 段選択（staged→opt→basic。
    // `gemm.rs::launch_wmma_tf32`／`run_wmma_tf32` ドキュメンテーション
    // コメント参照）は整列形状（`n%4==0 && k%4==0`）かつ staged 版が
    // 利用可能ならそちらを最優先で選ぶ。M=N=K=4096 は整列形状であるため
    // 実機では staged 経路が選ばれる可能性が高い。記録ラベルを固定文字列
    // にすると実際に計測したカーネルと食い違う（イシュー #1106・
    // PR #1115 codex-review P1 指摘対応）ため、`CudaGemm::
    // wmma_tf32_routed_path_is_staged` で実際に選択された経路を判定して
    // ラベルへ反映する（`launch_wmma_tf32` も `run_wmma_tf32` と同一の
    // 3 段選択ロジックを使うため、この判定は常駐 API 経由でも引き続き
    // 正確）。
    let tf32_path_label = if gemm.wmma_tf32_routed_path_is_staged(n, k) {
        "wmma_tf32_staged"
    } else {
        "wmma_tf32_opt"
    };
    let mut c_tf32_dev = gemm
        .alloc_output_f32(m, n)
        .expect("alloc_output_f32 must succeed on CUDA-equipped test runner");
    let tf32_kernel_only = bench_harness::run(&config, || {
        gemm.launch_wmma_tf32(&a_f32_dev, &b_f32_dev, &mut c_tf32_dev, m, n, k)
            .expect("launch_wmma_tf32 must succeed on CUDA-equipped test runner");
        gemm.synchronize()
            .expect("synchronize must succeed on CUDA-equipped test runner");
    })
    .expect("WMMA TF32 kernel-only measurement must satisfy TASK-8.1 protocol");
    let tf32_report = BenchReport::from_measurement(
        format!("gemm_{tf32_path_label}_kernel_only_4096"),
        "cuda",
        &tf32_kernel_only,
    )
    .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");
    let tf32_kernel_tflops = (flops / tf32_kernel_only.median_secs) / 1e12;
    println!(
        "tflops_record_kernel_only path={tf32_path_label} tflops={tf32_kernel_tflops:.3} report={}",
        tf32_report
            .to_json()
            .expect("BenchReport::to_json must succeed for a validated report")
    );

    // WMMA f16（opt。上記 assert で opt 経路の実行を保証済み）。イシュー
    // #1160 是正: 本番 f16 auto 経路が mma 優先へ結線された後は、この
    // 計測は「参考行」（assert 対象外）へ格下げする（下記「本番 f16
    // 経路」参照）。
    let (a_f16_dev, b_f16_dev) = wmma_gemm
        .upload_f16(&a_f16, &b_f16)
        .expect("upload_f16 must succeed on CUDA-equipped test runner");
    let mut c_f16_dev = wmma_gemm
        .alloc_output_f16(m, n)
        .expect("alloc_output_f16 must succeed on CUDA-equipped test runner");
    let wmma_f16_kernel_only = bench_harness::run(&config, || {
        wmma_gemm
            .launch_f16(&a_f16_dev, &b_f16_dev, &mut c_f16_dev, m, n, k)
            .expect("launch_f16 must succeed on CUDA-equipped test runner");
        wmma_gemm
            .synchronize()
            .expect("synchronize must succeed on CUDA-equipped test runner");
    })
    .expect("WMMA f16 kernel-only measurement must satisfy TASK-8.1 protocol");
    let wmma_f16_report = BenchReport::from_measurement(
        "gemm_wmma_f16_opt_kernel_only_4096",
        "cuda",
        &wmma_f16_kernel_only,
    )
    .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");
    let wmma_f16_kernel_tflops = (flops / wmma_f16_kernel_only.median_secs) / 1e12;
    println!(
        "tflops_record_kernel_only path=wmma_f16_opt(reference_only) \
         tflops={wmma_f16_kernel_tflops:.3} report={}",
        wmma_f16_report
            .to_json()
            .expect("BenchReport::to_json must succeed for a validated report")
    );

    // 本番 f16 経路（イシュー #1160）: `CudaGemmAuto::run_f16` が
    // M=N=K=4096（整列形状）で実際に選ぶ実装をカーネル単体計測し、この
    // 値を f16 assert の比較対象とする。本番選択結果そのもの
    // （`f16_matrix_unit_impl` 診断アクセサが返す `F16MatrixUnitImpl`）
    // から判定する（`auto.mma_available()` だけでは
    // `MMA_PRIORITY_PRODUCTION_ENABLED`〈本番既定 `false`〉を無視して
    // しまい、cc>=8.0 環境で実際は `Wmma` が選ばれるのに `Mma` を計測
    // した扱いになってしまう。PR #1179 codex-review 指摘）。診断アクセサ
    // は `internal-diagnostics` feature 限定のため、同 feature 無効時は
    // 本番既定（`false`＝`Wmma`）を前提に固定する（実機実行は
    // `make test-ignored-cuda` の `--all-features` に限られるため、この
    // 固定値が実際に使われることはない）。上記の `wmma_f16_kernel_only`
    // はこの分岐の結果を再利用せず独立に測るため、`Wmma` が選ばれた
    // 環境ではここで測り直さず上記の値をそのまま使う。
    // `internal-diagnostics` feature 無効時は `f16_matrix_unit_impl` 診断
    // アクセサへ到達できず `auto` を使わないため（`production_f16_uses_mma`
    // は本番既定 `false` に固定する下記 cfg 分岐）、その場合の unused
    // warning を許容する。
    #[cfg_attr(not(feature = "internal-diagnostics"), allow(unused_variables))]
    let auto = CudaGemmAuto::new(&device).expect("CudaGemmAuto::new must succeed");
    #[cfg(feature = "internal-diagnostics")]
    let production_f16_uses_mma = matches!(
        auto.f16_matrix_unit_impl(m, n, k),
        fandhe_ai_backend_cuda::F16MatrixUnitImpl::Mma
    );
    #[cfg(not(feature = "internal-diagnostics"))]
    let production_f16_uses_mma = false;

    let (production_f16_path_label, production_f16_kernel_tflops) = if production_f16_uses_mma {
        let mma_gemm = CudaMmaGemm::new(&device).expect(
            "CudaMmaGemm::new must succeed when f16_matrix_unit_impl() selects Mma \
             (cc>=8.0・NVRTC コンパイル成功・MMA_PRIORITY_PRODUCTION_ENABLED=true が前提)",
        );
        let (a_mma_dev, b_mma_dev) = mma_gemm
            .upload_f16(&a_f16, &b_f16)
            .expect("upload_f16 must succeed on CUDA-equipped test runner");
        let mut c_mma_dev = mma_gemm
            .alloc_output_f16(m, n)
            .expect("alloc_output_f16 must succeed on CUDA-equipped test runner");
        let mma_kernel_only = bench_harness::run(&config, || {
            mma_gemm
                .launch_f16(&a_mma_dev, &b_mma_dev, &mut c_mma_dev, m, n, k)
                .expect("mma.sync launch_f16 must succeed on CUDA-equipped test runner");
            mma_gemm
                .synchronize()
                .expect("synchronize must succeed on CUDA-equipped test runner");
        })
        .expect("mma.sync pipeline kernel-only measurement must satisfy TASK-8.1 protocol");
        let mma_report = BenchReport::from_measurement(
            "gemm_mma_f16_kernel_only_4096",
            "cuda",
            &mma_kernel_only,
        )
        .expect("BenchReport::from_measurement must succeed for a protocol-conformant measurement");
        let mma_kernel_tflops = (flops / mma_kernel_only.median_secs) / 1e12;
        println!(
            "tflops_record_kernel_only path=mma_sync_f16(production) tflops={mma_kernel_tflops:.3} \
             report={}",
            mma_report
                .to_json()
                .expect("BenchReport::to_json must succeed for a validated report")
        );
        ("mma_sync_f16", mma_kernel_tflops)
    } else {
        println!(
            "tflops_record_kernel_only path=wmma_f16_opt(production; \
             f16_matrix_unit_impl()=Wmma〈MMA_PRIORITY_PRODUCTION_ENABLED=false 保留中〉) \
             tflops={wmma_f16_kernel_tflops:.3}"
        );
        ("wmma_f16_opt", wmma_f16_kernel_tflops)
    };

    // 選択器（`f16_matrix_unit_impl` 診断アクセサ）が実際に選ぶ実装と、
    // 上記で計測した実装が食い違わないことを fail-closed に検査する
    // （`internal-diagnostics` feature 限定。選択器が選ばない経路を
    // 黙って計測し続け本番経路との乖離を見逃す false-green を防ぐ。
    // `make test-ignored-cuda` は `--all-features` のため GB10 実機
    // 実行では常にこの検査を通る。`production_f16_uses_mma` 自体も同じ
    // 選択器の呼び出しから導出しているため、ここでの再呼び出しは
    // 選択器が形状ごとに決定的であること・呼び出し間で結果が変わらない
    // ことを確認する回帰検査として機能する）。
    #[cfg(feature = "internal-diagnostics")]
    {
        let selected = auto.f16_matrix_unit_impl(m, n, k);
        let expected = if production_f16_uses_mma {
            fandhe_ai_backend_cuda::F16MatrixUnitImpl::Mma
        } else {
            fandhe_ai_backend_cuda::F16MatrixUnitImpl::Wmma
        };
        assert_eq!(
            selected, expected,
            "f16_matrix_unit_impl（選択器）が {selected:?} を返したが、本テストは \
             {expected:?}（本番選択結果から導出した production_f16_uses_mma=\
             {production_f16_uses_mma}）を計測している。tensor_core_tflops_record の \
             f16 assert は本番経路が選ぶ実装と一致しない量を比較してしまう"
        );
    }

    // #64 受け入れ条件・既存 `wmma_tf32_opt_exceeds_tiled_f32_tflops_at_4096`
    // と同じ判断根拠: Tensor Core 経路は tiled f32 を上回ることを実機で
    // 確認する（相対比較。転送区間を除いたカーネル単体 TFLOPS での比較へ
    // 切り替えた〈イシュー #1123 是正〉ため、判定式自体は変更しない
    // 〈緩和なし〉まま比較対象の量だけをカーネル単体 TFLOPS へ差し替える。
    // 本ファイル冒頭コメント参照）。
    //
    // TF32 経路: 実機で外れた場合は緩和せず #186 へ引き渡す。
    //
    // f16 経路: イシュー #1123 是正版（旧比較対象 `wmma_f16_opt`）では
    // GB10 実機実測（2026-09-03）で本 assert が red だった（wmma_f16_opt
    // カーネル単体 4.391〜4.496 TFLOPS が tiled f32 カーネル単体
    // 6.776〜6.790 TFLOPS を下回る）。イシュー #1160 で比較対象を
    // 「本番 f16 経路が実際に選ぶ実装」（`mma_sync_f16`。GB10 では
    // `mma_available()=true` のため `CudaMmaGemm`）へ差し替えたことで、
    // #1123 で確認済みの mma.sync パイプラインの優位性（tiled f32 比
    // 約 7〜11 倍。`docs/perf/cuda-wmma-f16-perf-triage.md` §3.1・
    // `dispatch_boundary.rs::large_shape_mma_pipeline_vs_wmma_tflops_record`）
    // により本 assert は pass に転じた（`docs/perf/
    // cuda-wmma-f16-perf-triage.md` §8 実測記録）。イシュー #1131 の
    // 完了条件（本 assert が pass すること）はこれで満たされた。
    assert!(
        tf32_kernel_tflops > tiled_kernel_tflops,
        "WMMA TF32 opt（カーネル単体 {tf32_kernel_tflops:.3} TFLOPS）が tiled f32（カーネル単体 \
         {tiled_kernel_tflops:.3} TFLOPS）を上回りませんでした（受け入れ条件: PoC-v2-3 \
         参考値 1.832 TFLOPS 超過。転送区間を除いたカーネル単体での比較）"
    );
    assert!(
        production_f16_kernel_tflops > tiled_kernel_tflops,
        "本番 f16 経路（{production_f16_path_label}・カーネル単体 \
         {production_f16_kernel_tflops:.3} TFLOPS）が tiled f32（カーネル単体 \
         {tiled_kernel_tflops:.3} TFLOPS）を上回りませんでした（受け入れ条件: PoC-v2-3 \
         参考値 1.832 TFLOPS 超過。転送区間を除いたカーネル単体での比較。比較対象は \
         `CudaGemmAuto::run_f16` が実際に選ぶ実装〈イシュー #1160〉。\
         docs/perf/cuda-wmma-f16-perf-triage.md §8 参照）"
    );
}

/// 複合判定通過の記録（#64 受け入れ条件の数値一致検証分）。
///
/// TF32 経路は `CudaGemm::run_wmma_tf32` と `fandhe_ai_backend_cpu::matmul_reference_fma`
/// を、f16 経路は `tests/cpu_cuda_wmma_parity.rs` の確立済み手順
/// （f16→f32 参照計算→f16 丸め→f32 化→`assert_parity`）を踏襲して比較する。
/// 形状は 512×512×512（CPU 参照計算が実機でも数秒以内に収まる規模。
/// `tests/gemm_wmma_tf32_opt.rs::wmma_tf32_opt_matches_reference_across_shapes`
/// の倍数境界形状の 1 つと同じ）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 7.0 以降）必須。実測記録は docs/perf/cuda-tensor-core-measurement.md"]
fn tensor_core_parity_record() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let (m, n, k) = (512u32, 512u32, 512u32);

    // TF32 経路。
    let gemm = CudaGemm::new(&device).expect("CudaGemm::new (tiled/WMMA TF32) must succeed");
    assert!(
        gemm.wmma_tf32_opt_available(),
        "WMMA TF32 opt kernel must be available on this ignored test runner (reason: {:?})",
        gemm.wmma_tf32_opt_unavailable_reason()
    );

    let mut rng_tf32 = bench_harness::rng::Xorshift64Star::new(0x7A0);
    let a_tf32 = rng_tf32.fill_vec((m as usize) * (k as usize));
    let b_tf32 = rng_tf32.fill_vec((k as usize) * (n as usize));
    let mut c_ref_tf32 = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a_tf32,
        &b_tf32,
        &mut c_ref_tf32,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_gpu_tf32 = gemm
        .run_wmma_tf32(&a_tf32, &b_tf32, m, n, k)
        .expect("run_wmma_tf32 must succeed on CUDA-equipped test runner");
    // 判定式・閾値は `fandhe_ai_backend_cpu::assert_parity` に一本化する（ローカル
    // 複製しない。`.claude/rules/coding-rust.md`）。実機で外れた場合は
    // 緩和せず #186 へ引き渡す。
    //
    // **PR #1115（イシュー #1106）codex-review P1 指摘対応**: 本テストを
    // `common::parity_baseline::assert_no_parity_regression`（既知不合格
    // ベースライン許容の非後退判定）へ置換していた変更を revert した
    // （AGENTS.md「数値契約の片側変更」・`.claude/rules/coding-rust.md`
    // 「バックエンド間数値一致テストの許容誤差を単独で緩和しない」に
    // 抵触するとの指摘）。TF32 経路が REQ-2 統一複合判定を最小形状から
    // 恒常的に満たさない既知状態（`docs/spec/04-requirements.md` REQ-2
    // の 2026-08-29 追記）にあること自体は事実だが、その非後退監視は
    // 本ファイルには併設しない（**PR #1115 codex-review 再指摘対応**:
    // 併設していた `tensor_core_parity_record_tf32_non_regression` は
    // `wmma_tf32_opt_available()` の確認後に公開 API `run_wmma_tf32` を
    // 呼んでいたが、整列形状（512×512×512。`n%4==0 && k%4==0`）では
    // `run_wmma_tf32` の 3 段選択が staged 経路を最優先で選ぶため
    // （`gemm.rs::run_wmma_tf32` ドキュメンテーションコメント参照）、
    // 実際には staged 経路の結果を `ParityPath::WmmaTf32Opt` の記録値に
    // 対して判定してしまっていた。`common/parity_baseline.rs` の
    // `ParityPath::WmmaTf32Opt` ドキュメンテーションコメントが明記する
    // とおり、opt カーネル単独の非後退監視は公開 API 経由では行わず
    // `fandhe_ai_backend_cuda::gemm::tests::wmma_tf32_opt_kernel_parity_does_not_regress`
    // （`src/gemm.rs`。private field 経由で 3 段選択を経由せず opt
    // カーネルを直接強制実行し、512×512×512 seed=0x7A0 行を含む全
    // `WmmaTf32Opt` 行を検査する）が既に正しく行っている。実際に選ばれた
    // staged 経路専用の非後退監視を追加するには 512×512×512 形状の
    // `ParityPath::WmmaTf32Staged` ベースライン行の新規実機実測が必要
    // だが未実施のため、推定値を書かず（`docs/perf/cuda-parity-baseline.md`
    // 「ベースライン更新規約」）このファイルでは追加しない。よって本ファイル
    // からは重複かつ誤判定だった非後退テストを削除し、`src/gemm.rs` 側の
    // 既存の正しい検査に一本化する）。
    fandhe_ai_backend_cpu::assert_parity(
        "tensor_core_parity_record tf32 512x512x512",
        &c_gpu_tf32,
        &c_ref_tf32,
    );
    // 実際に選択された経路（staged／opt）をラベルへ反映する（上記
    // `tf32_path_label` と同じ理由。イシュー #1106・PR #1115 codex-review
    // P1 指摘対応）。
    let tf32_record_path_label = if gemm.wmma_tf32_routed_path_is_staged(n, k) {
        "wmma_tf32_staged"
    } else {
        "wmma_tf32_opt"
    };
    println!(
        "parity_record path={tf32_record_path_label} shape=512x512x512 result=pass \
         (composite tolerance: relative<1e-3 or absolute<1e-5)"
    );

    // f16 経路（`tests/cpu_cuda_wmma_parity.rs::assert_wmma_f16_parity` と
    // 同じ量子化手順。カーネルのエピローグ store〈__float2half〉と同じ
    // 丸めを参照側にも適用する）。
    let wmma_gemm = CudaWmmaGemm::new(&device).expect("CudaWmmaGemm::new (WMMA f16) must succeed");
    assert!(
        wmma_gemm.wmma_f16_opt_available(),
        "WMMA f16 opt kernel must be available on this ignored test runner (reason: {:?})",
        wmma_gemm.wmma_f16_opt_unavailable_reason()
    );

    let mut rng_f16 = bench_harness::rng::Xorshift64Star::new(0xF160);
    let a_f16: Vec<f16> = rng_f16.fill_vec_f16((m as usize) * (k as usize));
    let b_f16: Vec<f16> = rng_f16.fill_vec_f16((k as usize) * (n as usize));
    let a_f32_from_f16: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32_from_f16: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32_from_f16 = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a_f32_from_f16,
        &b_f32_from_f16,
        &mut c_ref_f32_from_f16,
        m as usize,
        n as usize,
        k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed test input");
    let c_ref_rounded: Vec<f32> = c_ref_f32_from_f16
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();
    let c_gpu_f16 = wmma_gemm
        .run_f16(&a_f16, &b_f16, m, n, k)
        .expect("run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32_from_f16: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();
    fandhe_ai_backend_cpu::assert_parity(
        "tensor_core_parity_record f16 512x512x512",
        &c_gpu_f32_from_f16,
        &c_ref_rounded,
    );
    println!(
        "parity_record path=wmma_f16_opt shape=512x512x512 result=pass \
         (composite tolerance: relative<1e-3 or absolute<1e-5)"
    );
}
