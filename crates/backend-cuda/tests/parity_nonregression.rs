//! parity 非後退契約のテスト（イシュー #491・GEMM 性能改善ツリー B-0）。
//!
//! `wmma_tf32`・`wmma_tf32_opt`・`mma_f16` 系経路は REQ-2 統一複合判定で
//! 恒常 fail の既知状態にある（#186・`docs/backend-cuda-real-device-testing.md`
//! §5.3）。以降の Phase B/C カーネル改修は「parity green」ではなく
//! 「非後退」（fail 比率・平均絶対誤差が記録済みベースラインを上回らない）
//! で受け入れ判定する必要があり、本ファイルはその機械検査を提供する。
//!
//! # テスト構成
//!
//! - **通常 CI（`#[ignore]` なし・GPU 不要）**: [`common::parity_baseline`]
//!   の tolerance 定数 pin・fixture 自己整合性・検査関数自体の
//!   falsification（常に pass する壊れ方の防止）を検査する。GPU・CUDA
//!   非搭載環境（GitHub ホステッド runner 等）でも実行できる純粋なロジック
//!   検査のみで構成する。
//! - **実機必須（`#[ignore = "CUDA 実機必須"]`）**: fixture の各行について
//!   記録時と同一の (エントリポイント・形状・シード・データ生成・参照計算
//!   経路) で GPU 実行し、[`common::parity_baseline::assert_no_parity_regression`]
//!   で非後退を検査する。既存の `assert_parity` ベーステスト（最初の fail
//!   で panic）と異なり、fixture の全行を検査する（1 行の fail で残りの
//!   行の検査を打ち切らない。どの行が後退したかを特定可能にするため）。
//!
//! ## `ParityPath::WmmaTf32`（基本版）行の検査場所（PR #640 codex-review
//! P1 指摘対応・イシュー #491）
//!
//! `ParityPath::WmmaTf32`（基本版カーネル単独）の非後退検査は
//! `fandhe_ai_backend_cuda::gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`
//! （`crates/backend-cuda/src/gemm.rs` 内のライブラリ単体テスト。`cargo test
//! -p fandhe-ai-backend-cuda --lib -- --ignored` で実行）へ移設済みである。
//!
//! 以前は `run_wmma_tf32`（opt 可用なら常に opt を優先する公開 API）とは
//! 別に、基本版カーネルを強制実行するテスト専用エントリポイント
//! （`run_wmma_tf32_basic_for_test`）を `internal-testing` という通常の
//! Cargo feature 経由で `pub` 化して本ファイルから呼んでいた。Cargo
//! feature は依存グラフ全体で単一に統合されるため、downstream が自分の
//! `Cargo.toml` で当該 feature を明示指定すればこのメソッドへ到達でき、
//! `[dev-dependencies]` の自己参照だけでは外部からの有効化を防げない
//! （REQ-11「明示切替 API を提供しない」方針・公開 API 設計〈内部表現の
//! 非漏出〉に抵触。指摘対応でこの feature・メソッドは削除した）。
//!
//! `tests/*.rs` は独立クレート扱いのため公開 API しか呼べず、基本版
//! カーネル（private field）へ公開 API を増やさずに到達する唯一の方法は
//! ライブラリ自身の単体テストであるため、上記の通り移設した
//! （`gemm.rs` 内の該当テストのドキュメンテーションコメント参照）。
//!
//! ## `ParityPath::WmmaTf32Opt` 行の検査場所（PR #678 codex-review P1
//! 指摘対応・イシュー #500）
//!
//! 本ファイルの実機必須テストは `ParityPath::WmmaTf32Opt` を**対象としない**
//! （`ParityPath::WmmaTf32` と同じ理由で `gemm.rs` へ移設済み）。イシュー
//! #500 で TF32 opt-staged カーネル（cp.async 多段パイプライン）が追加され、
//! `run_wmma_tf32`（公開 API）は staged カーネルが利用可能かつ cp.async 16
//! バイト整列条件（`n%4==0 && k%4==0`）を満たす形状では staged 経路を
//! 最優先で選ぶようになった（`gemm.rs::run_wmma_tf32` 3 段選択）。既存
//! `WmmaTf32Opt` ベースライン行（512×512×512・64×64×64・512×512×4096）は
//! いずれも 4 の倍数形状のため、以前ここで `wmma_tf32_opt_available()` を
//! assert してから公開 `run_wmma_tf32` を呼んでいた検査は、#500 以降は
//! 実際には staged カーネルの実測になってしまい、opt カーネル自体の回帰を
//! 検出できなくなっていた（opt 非後退テストが黙って staged 経路へ
//! すり替わる。codex-review P1 指摘）。修正として、opt カーネル単独の検査は
//! `fandhe_ai_backend_cuda::gemm::tests::wmma_tf32_opt_kernel_parity_does_not_regress`
//! （`src/gemm.rs`。private field `wmma_tf32_opt`・private fn
//! `run_wmma_tf32_opt_kernel` へ同一モジュール内から直接アクセスし、
//! 3 段選択を経由せず opt カーネルを強制実行する）へ移設した。本ファイルは
//! 代わりに新設の `ParityPath::WmmaTf32Staged`（[`check_wmma_tf32_staged_baseline`]）
//! を検査する。staged は #500 以降の `run_wmma_tf32` が整列形状で実際に
//! 選ぶ経路であるため、公開 API 経由でも正しく強制実行できる。

mod common;

use common::parity_baseline::{
    BASELINES, ParityBaseline, ParityPath, assert_no_parity_regression,
    assert_tolerance_constants_pinned,
};
use fandhe_ai_backend_cuda::{CudaDevice, CudaGemm, CudaMmaGemm};
use half::f16;

// --- 通常 CI テスト（GPU 不要） ---

/// 受け入れ基準 3: tolerance 定数（`RELATIVE_TOLERANCE`/
/// `ABSOLUTE_RESCUE_THRESHOLD`）の無断変更を機械検知する。
#[test]
fn tolerance_constants_are_pinned() {
    assert_tolerance_constants_pinned();
}

/// fixture 自体の妥当性検査: 各行の `baseline_fail_count <= total`・
/// `total == m*n`・4 経路すべてに 1 行以上存在することを確認する。
/// fixture 値の入力ミス（転記ミス等）を CI で機械的に検出する。
#[test]
fn baseline_fixture_is_self_consistent() {
    assert!(!BASELINES.is_empty(), "BASELINES must not be empty");

    for b in BASELINES {
        let expected_total = (b.m as usize) * (b.n as usize);
        assert_eq!(
            b.total, expected_total,
            "{}: total は m*n と一致する必要があります（total={}, m*n={}）",
            b.context, b.total, expected_total
        );
        assert!(
            b.baseline_fail_count <= b.total,
            "{}: baseline_fail_count({}) が total({}) を超えています",
            b.context,
            b.baseline_fail_count,
            b.total
        );
        assert!(
            b.baseline_mean_abs_diff_ceiling >= 0.0 && b.baseline_mean_abs_diff_ceiling.is_finite(),
            "{}: baseline_mean_abs_diff_ceiling は有限の非負値である必要があります（値={}）",
            b.context,
            b.baseline_mean_abs_diff_ceiling
        );
    }

    for path in [
        ParityPath::WmmaTf32,
        ParityPath::WmmaTf32Opt,
        ParityPath::WmmaTf32Staged,
        ParityPath::MmaF16,
    ] {
        assert!(
            BASELINES.iter().any(|b| b.path == path),
            "経路 {path} に対応する baseline 行が 1 件も存在しません"
        );
    }
}

/// `baseline_provenance_unconfirmed` フィールドの誤用防止（codex-review P1
/// 指摘対応・イシュー #491。PR #678 codex-review P1 指摘対応で
/// `WmmaTf32Staged` を許容経路へ追加・フィールド名変更に追従）: このフラグは
/// 「記録値の provenance が確定していない」ケース専用であり、
/// `WmmaTf32Opt`・`MmaF16` の記録元は opt 可用性確認済み／基本-opt 分岐が
/// 存在しないためこの不確実性が原理的に生じない（`common/parity_baseline.rs`
/// 各行コメント参照）。このテストは全経路の行に
/// `baseline_provenance_unconfirmed: true` が誤って付与されていないことを
/// 固定し、非後退ゲートが意図せず広くスキップされる回帰を防ぐ（`WmmaTf32`
/// はイシュー #1106 の GB10 実機実測で基本版カーネル単独の確定測定が
/// 完了したため、他経路と同じく `false` 固定の対象へ含める。`WmmaTf32Opt`・
/// `WmmaTf32Staged`・`MmaF16` はそれぞれ #491／#726 の実機実測で確定済み。
/// いずれかの経路が再び unconfirmed へ戻る変更はゲートの弱体化であり
/// このテストが検出する）。
#[test]
fn baseline_provenance_unconfirmed_is_scoped_to_unmeasured_paths_only() {
    for b in BASELINES {
        assert!(
            !b.baseline_provenance_unconfirmed,
            "{}: 経路 {:?} で baseline_provenance_unconfirmed が true に \
             なっており、非後退ゲートが意図せずスキップされます \
             （全経路が確定済みである前提が崩れています）",
            b.context, b.path
        );
    }
}

/// 「非後退ゲートが実際に機能しているか」を経路ごとに可視化する
/// （codex-review P1 指摘対応の副作用への対策・イシュー #491）。
///
/// `WmmaTf32Opt`・`MmaF16` は provenance 不確実性が原理的に生じない経路
/// （`common/parity_baseline.rs` 各行コメント参照）、`WmmaTf32Staged` は
/// イシュー #726 の実機実測で確定済みの経路、`WmmaTf32`（基本版）は
/// イシュー #1106 の GB10 実機実測（基本版カーネル専用の単体テスト
/// `fandhe_ai_backend_cuda::gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`
/// の release 2 回実行で値の安定を確認）で確定済みの経路のため、全行が
/// enforced（`baseline_provenance_unconfirmed == false`）であることを
/// 固定する。0 件を green として固定するテストは置かない —— それ自体が
/// codex-review P1 指摘が問題視した「機能していないゲートを正常状態として
/// 固定する」パターンになるため）。
#[test]
fn wmma_tf32_opt_and_mma_f16_rows_are_fully_enforced() {
    for path in [
        ParityPath::WmmaTf32,
        ParityPath::WmmaTf32Opt,
        ParityPath::WmmaTf32Staged,
        ParityPath::MmaF16,
    ] {
        let total = BASELINES.iter().filter(|b| b.path == path).count();
        let enforced = BASELINES
            .iter()
            .filter(|b| b.path == path && !b.baseline_provenance_unconfirmed)
            .count();
        assert_eq!(
            enforced, total,
            "経路 {path}: baseline_provenance_unconfirmed=true の行が含まれています \
             （この経路は provenance 不確実性が生じないため全行 enforced である\
             べきです）"
        );
    }
}

/// fail-closed 契約の falsification テスト（codex-review P1 再指摘対応・
/// イシュー #491）: `baseline_provenance_unconfirmed: true` の行を
/// `assert_no_parity_regression` に渡すと、実測値の良否に関わらず
/// **必ず panic する**ことを固定する（黙って skip して pass する壊れ方の
/// 防止。`assert_no_parity_regression_panics_on_*` 3 兄弟と同方針）。
///
/// 実測 report 側は baseline を上回らない（=通常なら pass する）合成値に
/// あえて設定し、それでも provenance 未確定の一点だけで fail-closed に
/// panic することを検証する。
#[test]
#[should_panic(expected = "baseline_provenance_unconfirmed")]
fn assert_no_parity_regression_panics_on_unconfirmed_baseline() {
    let baseline = ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "synthetic unconfirmed",
        m: 2,
        n: 2,
        k: 2,
        seed: 1,
        total: 4,
        // fail_count・mean_abs_diff とも余裕を持って baseline 以下（通常なら
        // pass する実測値）にしても、unconfirmed の一点で panic すること
        // を確認する。
        baseline_fail_count: 4,
        baseline_mean_abs_diff_ceiling: 1.0,
        baseline_provenance_unconfirmed: true,
    };
    let a = vec![0.0f32; baseline.total];
    let b = vec![0.0f32; baseline.total];
    let report = fandhe_ai_backend_cpu::compare(&a, &b).expect("length must match");
    assert_no_parity_regression("synthetic unconfirmed", &report, &baseline);
}

/// 検査関数自体の falsification テスト（`.claude/rules/coding-rust.md`
/// 「本番経路で unwrap/expect を使わない」とは別軸の品質観点: 検査
/// ユーティリティが「常に pass する」壊れ方をしていないことを固定する。
/// `backend-cpu/src/parity.rs` の既存テストと同方針）。
///
/// ベースラインを人為的に上回る合成 `CompareReport` を与えたとき、
/// `assert_no_parity_regression` が panic することを確認する。
#[test]
#[should_panic(expected = "非後退契約 FAIL")]
fn assert_no_parity_regression_panics_on_fail_count_regression() {
    let baseline = ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "synthetic",
        m: 4,
        n: 4,
        k: 4,
        seed: 1,
        total: 16,
        baseline_fail_count: 2,
        baseline_mean_abs_diff_ceiling: 1e-4,
        baseline_provenance_unconfirmed: false,
    };
    // fail_count がベースライン(2)を上回る合成レポート。
    let a = vec![0.0f32; baseline.total];
    let mut b = vec![0.0f32; baseline.total];
    // 3 セルを大きく乖離させ、複合判定で fail させる
    // （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満のいずれも満たさない値）。
    b[0] = 1.0;
    b[1] = 1.0;
    b[2] = 1.0;
    let report = fandhe_ai_backend_cpu::compare(&a, &b).expect("length must match");
    assert_no_parity_regression("synthetic", &report, &baseline);
}

/// mean_abs_diff 側の非後退違反を単独で検知することを確認する
/// （fail_count は据え置き、mean_abs_diff のみベースライン超過とする
/// 合成ケース）。
#[test]
#[should_panic(expected = "非後退契約 FAIL")]
fn assert_no_parity_regression_panics_on_mean_abs_diff_regression() {
    let baseline = ParityBaseline {
        path: ParityPath::MmaF16,
        context: "synthetic mean_abs_diff",
        m: 2,
        n: 2,
        k: 2,
        seed: 1,
        total: 4,
        // fail_count は緩く設定し、mean_abs_diff 側のみで fail させる。
        baseline_fail_count: 4,
        baseline_mean_abs_diff_ceiling: 1e-6,
        baseline_provenance_unconfirmed: false,
    };
    let a = vec![0.0f32; baseline.total];
    // 絶対誤差救済閾値(1e-5)未満のため複合判定は pass するが、
    // mean_abs_diff(約 5e-6) は baseline_mean_abs_diff_ceiling(1e-6) を上回る。
    let b = vec![5e-6f32; baseline.total];
    let report = fandhe_ai_backend_cpu::compare(&a, &b).expect("length must match");
    assert_no_parity_regression("synthetic mean_abs_diff", &report, &baseline);
}

/// total 不一致（形状・比較対象のずれ）を fail-closed で検知することを
/// 確認する。
#[test]
#[should_panic(expected = "baseline")]
fn assert_no_parity_regression_panics_on_total_mismatch() {
    let baseline = ParityBaseline {
        path: ParityPath::WmmaTf32,
        context: "synthetic total mismatch",
        m: 4,
        n: 4,
        k: 4,
        seed: 1,
        total: 16,
        baseline_fail_count: 100,
        baseline_mean_abs_diff_ceiling: 1.0,
        baseline_provenance_unconfirmed: false,
    };
    // total が baseline(16) と異なる合成レポート。
    let a = vec![0.0f32; 9];
    let b = vec![0.0f32; 9];
    let report = fandhe_ai_backend_cpu::compare(&a, &b).expect("length must match");
    assert_no_parity_regression("synthetic total mismatch", &report, &baseline);
}

// --- 実機必須テスト（`#[ignore]`。CUDA 実機・compute capability 8.0 以降必須） ---

/// fixture の各行を記録時と同一の入力（seed・形状・生成手順）で再現し、
/// GPU 実行結果を非後退判定する。1 行の fail で残りの行の検査を打ち切ら
/// ない（複数行が同時に後退した場合でもすべて検出できるよう、最後に
/// まとめて assert する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等、compute capability 8.0 以降）必須"]
fn parity_baselines_do_not_regress() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let gemm = CudaGemm::new(&device).expect("WMMA(TF32) kernel compilation must succeed");
    let mma_gemm = CudaMmaGemm::new(&device).expect("mma kernel compilation must succeed");

    let mut failures: Vec<String> = Vec::new();

    for baseline in BASELINES {
        // `ParityPath::WmmaTf32`（基本版カーネル単独）・`ParityPath::WmmaTf32Opt`
        // 行はここでは検査しない。どちらも private field（`self.wmma_tf32`／
        // `self.wmma_tf32_opt`）へ直接アクセスしないと 3 段選択を経由せず
        // 強制実行できず、独立クレート扱いの本ファイルから公開 API を
        // 増やさずに到達する手段がないため、ライブラリ自身の単体テスト
        // （`fandhe_ai_backend_cuda::gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`・
        // `wmma_tf32_opt_kernel_parity_does_not_regress`。`src/gemm.rs`）へ
        // 検査自体を移設済み（本ファイル冒頭ドキュメンテーションコメント
        // 参照。PR #640・#678 codex-review P1 指摘対応）。
        // `ParityPath::WmmaF16` 行（イシュー #1106・GB10 全件洗い出しで
        // 追加。PR #1124 codex-review〈Cursor Bugbot Medium〉指摘で
        // `WmmaF16Opt` から統合。`ParityPath::WmmaF16` ドキュメンテーション
        // コメント参照）もここでは検査しない。それぞれの記録元ファイル
        // （`tests/cpu_cuda_wmma_parity.rs::wmma_f16_k4096_stress_non_regression`・
        // `tests/gemm_wmma_f16_opt.rs::wmma_f16_opt_k4096_stress_non_regression`）
        // が同一 baseline 行を直接検査済みのため、ここでの重複検査は
        // 行わない（`WmmaTf32`／`WmmaTf32Opt` の除外と同じ理由）。
        if baseline.path == ParityPath::WmmaTf32
            || baseline.path == ParityPath::WmmaTf32Opt
            || baseline.path == ParityPath::WmmaF16
        {
            continue;
        }
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match baseline.path {
                ParityPath::WmmaTf32Staged => {
                    check_wmma_tf32_staged_baseline(&gemm, baseline);
                }
                ParityPath::MmaF16 => {
                    check_mma_f16_baseline(&mma_gemm, baseline);
                }
                ParityPath::WmmaTf32 | ParityPath::WmmaTf32Opt | ParityPath::WmmaF16 => {
                    unreachable!("continue で除外済み")
                }
            }));
        if let Err(err) = result {
            let msg = err
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "panic (詳細不明)".to_string());
            failures.push(format!("{}: {msg}", baseline.context));
        }
    }

    assert!(
        failures.is_empty(),
        "parity 非後退契約 FAIL（{} 件）:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// `wmma_tf32_staged` 経路 1 行の非後退検査（`ParityPath::WmmaTf32Staged`
/// 専用。PR #678 codex-review P1 指摘対応・イシュー #500）。
///
/// `ParityPath::WmmaTf32Opt` はここでは検査しない（`gemm.rs` の内部テストへ
/// 移設済み。本ファイル冒頭ドキュメンテーションコメント参照）。この行は
/// 逆に公開 API（`run_wmma_tf32`）経由で正しく staged カーネルを強制実行
/// できる: fixture のこの経路の形状は cp.async 16 バイト整列条件
/// （`n%4==0 && k%4==0`）を満たすよう選定済みであり、staged カーネルが
/// 利用可能な環境では `run_wmma_tf32` の 3 段選択が必ず staged 経路を
/// 選ぶ（`gemm.rs::run_wmma_tf32` ドキュメンテーションコメント参照）。
///
/// 参照値は `fandhe_ai_backend_cpu::matmul_reference_fma`（非量子化。
/// `docs/backend-cuda-real-device-testing.md` §5.3 が確認済みの意図的設計を
/// 踏襲する）。事前に `wmma_tf32_staged_available()` を assert し、staged
/// カーネルが実際に利用可能な環境で計測していることを保証してから実行する
/// （既存テスト `gemm_wmma_tf32_staged.rs` と同じ考え方）。
fn check_wmma_tf32_staged_baseline(gemm: &CudaGemm, baseline: &ParityBaseline) {
    assert!(
        gemm.wmma_tf32_staged_available(),
        "{}: staged kernel must be available on this ignored test runner (reason: {:?})",
        baseline.context,
        gemm.wmma_tf32_staged_unavailable_reason()
    );

    let mut rng = bench_harness::rng::Xorshift64Star::new(baseline.seed);
    let a = rng.fill_vec((baseline.m as usize) * (baseline.k as usize));
    let b = rng.fill_vec((baseline.k as usize) * (baseline.n as usize));

    let mut c_ref = vec![0.0f32; (baseline.m as usize) * (baseline.n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a,
        &b,
        &mut c_ref,
        baseline.m as usize,
        baseline.n as usize,
        baseline.k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed baseline input");

    let c_gpu = gemm
        .run_wmma_tf32(&a, &b, baseline.m, baseline.n, baseline.k)
        .expect("CudaGemm::run_wmma_tf32 must succeed on a compute capability >= 8.0 test runner");

    let report =
        fandhe_ai_backend_cpu::compare(&c_gpu, &c_ref).expect("shape must match baseline fixture");

    assert_no_parity_regression(baseline.context, &report, baseline);
}

/// f16 経路（`mma_f16`）1 行の非後退検査。
///
/// 参照値は「f16→f32→`matmul_reference_fma`→f16 丸め→f32」の量子化込み
/// 経路（`tests/cpu_cuda_mma_parity.rs::assert_mma_f16_parity` と同一手順。
/// GPU 側エピローグ store の丸めを参照側にも反映させる）。
fn check_mma_f16_baseline(gemm: &CudaMmaGemm, baseline: &ParityBaseline) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(baseline.seed);
    let a_f16: Vec<f16> = rng.fill_vec_f16((baseline.m as usize) * (baseline.k as usize));
    let b_f16: Vec<f16> = rng.fill_vec_f16((baseline.k as usize) * (baseline.n as usize));

    let a_f32: Vec<f32> = a_f16.iter().map(|x| x.to_f32()).collect();
    let b_f32: Vec<f32> = b_f16.iter().map(|x| x.to_f32()).collect();
    let mut c_ref_f32 = vec![0.0f32; (baseline.m as usize) * (baseline.n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a_f32,
        &b_f32,
        &mut c_ref_f32,
        baseline.m as usize,
        baseline.n as usize,
        baseline.k as usize,
    )
    .expect("matmul_reference_fma shape validation must pass for well-formed baseline input");
    let c_ref_rounded: Vec<f32> = c_ref_f32
        .iter()
        .map(|&x| f16::from_f32(x).to_f32())
        .collect();

    let c_gpu_f16 = gemm
        .run_f16(&a_f16, &b_f16, baseline.m, baseline.n, baseline.k)
        .expect("CudaMmaGemm::run_f16 must succeed on CUDA-equipped test runner");
    let c_gpu_f32: Vec<f32> = c_gpu_f16.iter().map(|x| x.to_f32()).collect();

    let report = fandhe_ai_backend_cpu::compare(&c_gpu_f32, &c_ref_rounded)
        .expect("shape must match baseline fixture");
    assert_no_parity_regression(baseline.context, &report, baseline);
}
