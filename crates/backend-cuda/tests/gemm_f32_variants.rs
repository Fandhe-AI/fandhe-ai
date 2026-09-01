//! `gemm_variant_selection::CudaGemmF32VariantSelection`（イシュー #1035。
//! f32 GEMM の simple / double-buffer ヒューリスティック選択。SplitK は
//! イシュー #1100 で選択候補から撤退し `run_split_k_forced` 経由の診断
//! 専用に変更した）の環境適応型テスト＋実機必須テスト。
//!
//! `tests/gemm_tiled.rs` と同じ構成（CUDA 搭載・非搭載どちらの環境でも
//! green になる環境適応型スモークテスト＋受け入れ条件そのものである
//! 数値一致・決定性テストを `#[ignore]` で分離）を踏襲する
//! （`.claude/rules/coding-rust.md` の実機依存テスト分離規約）。
//!
//! `CudaGemmF32VariantSelection`（`internal-diagnostics` feature 限定）を
//! 使うため、本ファイル自体も `Cargo.toml` の `[[test]]` セクションで
//! `required-features = ["internal-diagnostics"]` を指定している
//! （`specialized_mma_parity.rs` と同じ理由）。
//!
//! **本ランでの実施範囲**: `#[ignore]` テストは NVRTC 実行不能環境（実機
//! CUDA toolkit 非搭載）のため未実行のまま残す。実機実測（複合判定・
//! 決定性・境界形状網羅）は Mac / DGX Spark セッションで
//! `cargo test -p fandhe-ai-backend-cuda --features internal-diagnostics \
//! -- --ignored` により実施する。

use fandhe_ai_backend_cuda::gemm_variant_selection::{
    CudaGemmF32VariantSelection, GemmVariantKind,
};
use fandhe_ai_backend_cuda::{CudaDevice, CudaError};

/// `gemm_variant.rs::VARIANT_TILE` の複製（本ファイル側の期待変種計算
/// 専用）。値そのものの単一の真実源は `kernels::TILE`／`gemm_variant.rs::
/// VARIANT_TILE` であり、本定数はテスト側の複製であることを明記する
/// （`gemm_variant.rs::VARIANT_TILE` 自体のコメントと同じ方針。値変更時は
/// 3 箇所を同時に見直す必要がある）。
const EXPECT_TILE: u32 = 32;

/// `gemm_variant::num_blocks`（非公開）と同じ計算をテスト側で複製する
/// （`grid` が生成するスレッドブロック総数。オーバーフロー安全のため
/// `u64` 昇算）。イシュー #1035 PR #1073 レビュー指摘: これまでのテストは
/// `selected_variant` の戻り値を実行結果のラベル付けにしか使っておらず、
/// DoubleBuffer 候補が実際に選ばれたかどうかを一切検証していなかった
/// （NVRTC コンパイル失敗による fail-soft フォールバックで Simple へ静かに
/// 落ちても成功してしまう）。本関数はそのギャップを埋めるための期待値
/// 計算に使う。
fn expected_blocks(m: u32, n: u32) -> u64 {
    let tiles_m = u64::from(m).div_ceil(u64::from(EXPECT_TILE));
    let tiles_n = u64::from(n).div_ceil(u64::from(EXPECT_TILE));
    tiles_m * tiles_n
}

/// 各テスト形状が「どのヒューリスティック分岐を狙って設計されたか」を表す
/// 分類（`run_f32_matches_cpu_reference_across_variant_shapes` のケース表と
/// 対応）。
#[derive(Debug, Clone, Copy)]
enum ExpectedCategory {
    /// 非整列・小 K・境界形状。`num_sms`（実機 SM 数）に依存せず常に
    /// Simple が選ばれるはずの形状（アラインメント不成立または
    /// `k < DOUBLE_BUFFER_MIN_K` により DoubleBuffer 分岐に入り得ない）。
    AlwaysSimple,
    /// アラインメント済み・K 十分・grid が現実的などの GPU の SM 数も
    /// 大きく上回る大形状。DoubleBuffer 自体の必要条件（`blocks >=
    /// num_sms` かつ `num_sms` が既知）は形状だけでは保証されない
    /// （イシュー #1035 PR #1073 Bugbot 指摘: `num_sms` が `None`〈実機
    /// SM 数取得失敗〉のときはヒューリスティックが常に Simple を返す
    /// ため、DoubleBuffer が利用可能でも Simple へ fail-soft する）。
    /// `assert_matches_heuristic` 側で `num_sms` と `blocks` の大小関係も
    /// 検証する。
    DoubleBufferIfAvailable,
    /// K 支配的非正方（grid が SM 数を埋めきれない可能性がある形状）。
    /// **イシュー #1100 で SplitK が選択候補から撤退**したため、本カテゴリ
    /// は「grid が SM を埋められない（`blocks < num_sms`）場合は
    /// DoubleBuffer の前提〈`blocks >= num_sms`〉も満たさないため常に
    /// Simple」という契約を検証する（`gemm_variant.rs` 冒頭「SplitK
    /// 撤退の判断」参照）。`blocks >= num_sms` を満たす形状（例:
    /// 256×256×16384）では従来どおり DoubleBuffer が選ばれうる。
    KDominant,
}

/// [`assert_matches_heuristic`] の戻り値。集計側（呼び出しループ）が
/// 「DoubleBuffer が可用時に実際に一度でも選ばれたか」を追跡するために
/// 使う（可用なのに全ケースで Simple へ落ちていれば、テスト形状の設計
/// 不足かヒューリスティックの回帰を示す）。
#[derive(Debug, Clone, Copy, Default)]
struct VariantExercised {
    double_buffer: bool,
}

/// `category` と実機の可用性・SM 数から、`selected_variant` が返すべき
/// 値をヒューリスティックの判定順序どおりに検証する（`gemm_variant.rs::
/// select_f32_gemm_variant` の判定順序をテスト側で最小限複製する。
/// `gemm_variant` モジュールは非公開のため対象関数を直接呼べず、独立した
/// 経路で期待値を導出することでテストの意義を保つ）。
fn assert_matches_heuristic(
    label: &str,
    category: ExpectedCategory,
    m: u32,
    n: u32,
    num_sms: Option<u32>,
    double_buffer_available: bool,
    actual: GemmVariantKind,
) -> VariantExercised {
    let mut exercised = VariantExercised::default();
    match category {
        ExpectedCategory::AlwaysSimple => {
            assert_eq!(
                actual,
                GemmVariantKind::Simple,
                "{label}: 非整列・小 K・境界形状はハードウェアに関わらず常に \
                 Simple を選ぶはずだが actual={actual:?} だった"
            );
        }
        ExpectedCategory::DoubleBufferIfAvailable | ExpectedCategory::KDominant => {
            // イシュー #1100 で SplitK 分岐を撤退した結果、
            // DoubleBufferIfAvailable と KDominant はいずれも「blocks と
            // num_sms の大小関係・アラインメントだけで DoubleBuffer か
            // Simple かが決まる」という同一の期待値ロジックに帰着する
            // （旧 KDominant は `blocks < num_sms` の場合に SplitK を
            // 期待していたが、撤退後はその条件でも DoubleBuffer の前提
            // 〈blocks >= num_sms〉を満たさないため Simple になる）。
            let blocks = expected_blocks(m, n);
            let grid_fills_sms = num_sms.is_some_and(|sms| blocks >= u64::from(sms));
            let expected = if double_buffer_available && grid_fills_sms {
                GemmVariantKind::DoubleBuffer
            } else {
                GemmVariantKind::Simple
            };
            assert_eq!(
                actual, expected,
                "{label}: blocks({blocks}) と num_sms({num_sms:?}) の関係・\
                 double_buffer_available={double_buffer_available} から \
                 actual={expected:?} を期待したが actual={actual:?} だった \
                 （fail-soft フォールバックが本来のカーネルを覆い隠して \
                 いないか確認する）"
            );
            exercised.double_buffer = matches!(actual, GemmVariantKind::DoubleBuffer);
        }
    }
    exercised
}

/// 決定的シードで A・B（f32）を生成する（`tests/gemm_tiled.rs` と同じ
/// 生成方法）。
fn gen_ab(seed: u64, m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = bench_harness::rng::Xorshift64Star::new(seed);
    (rng.fill_vec(m * k), rng.fill_vec(k * n))
}

/// `CudaGemmF32VariantSelection::new` は CUDA 非搭載環境で panic せず
/// 型付きエラーを返す（`tests/gemm_tiled.rs::
/// new_compiles_tiled_kernels_or_returns_typed_error_without_panicking`
/// と同じ環境適応スモーク）。CUDA 搭載環境では base（Simple 経路。
/// naive/tiled 必須 4 カーネル）は必ずコンパイルに成功する契約
/// （`CudaGemm::new` に委譲。DoubleBuffer／SplitK はコンパイル失敗時
/// fail-soft のため `new` 自体は成功しうる）。
#[test]
fn new_builds_or_returns_typed_error_without_panicking() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(CudaError::DriverUnavailable { detail }) => {
            assert!(!detail.is_empty(), "detail message must not be empty");
            return;
        }
        Err(CudaError::Driver(_)) => {
            return;
        }
        Err(other) => panic!("unexpected CudaError variant from CudaDevice::new: {other}"),
    };

    match CudaGemmF32VariantSelection::new(&device) {
        Ok(selection) => {
            // CUDA 搭載環境: base（Simple 経路）は必ず利用可能。DoubleBuffer／
            // SplitK は fail-soft のため可用性は問わない（`double_buffer_
            // available`/`split_k_available` の呼び出しが panic しないことのみ
            // 確認する）。
            let _ = selection.double_buffer_available();
            let _ = selection.split_k_available();
        }
        Err(CudaError::NvrtcUnavailable { detail }) => {
            assert!(!detail.is_empty());
        }
        Err(other) => {
            panic!("unexpected CudaError variant from CudaGemmF32VariantSelection::new: {other}")
        }
    }
}

/// `selected_variant` が実際の起動を伴わず呼べること（可観測性用 API の
/// スモーク。GPU 資源が無い場合でも `CudaGemmF32VariantSelection::new` が
/// 失敗するため到達しないが、到達した場合に panic しないことを確認する）。
/// **`selected_variant` は SplitK を返さない**（イシュー #1100）ため
/// `128, 128, 8192`（旧 SplitK 想定形状）を含めても panic しないことを
/// 確認する。
#[test]
fn selected_variant_does_not_panic_when_device_available() {
    let device = match CudaDevice::new(0) {
        Ok(dev) => dev,
        Err(_) => return,
    };
    let Ok(selection) = CudaGemmF32VariantSelection::new(&device) else {
        return;
    };
    let _ = selection.selected_variant(4096, 4096, 4096);
    let _ = selection.selected_variant(128, 128, 8192);
    let _ = selection.selected_variant(0, 0, 0);
}

/// 形状ごとに `run_f32` の結果を CPU 参照実装
/// （[`fandhe_ai_backend_cpu::matmul_reference_fma`]）と
/// [`fandhe_ai_backend_cpu::assert_parity`]（相対誤差 1e-3 未満 または
/// 絶対誤差 1e-5 未満の複合判定。`.claude/rules/coding-rust.md` の許容誤差
/// を変更しない）で照合する。DoubleBuffer が実際に選ばれる形状
/// （アラインメント済み大形状）と Simple に留まる形状（非整列・小 K・
/// grid が SM を埋められない K 支配的形状）の両方を網羅する。
#[test]
#[ignore = "実機（CUDA/NVRTC 搭載）環境が必要。本ランは NVRTC 実行不能環境 \
            のため未実行。Mac / DGX Spark セッションで `cargo test -p \
            fandhe-ai-backend-cuda --features internal-diagnostics \
            -- --ignored` として実施する"]
fn run_f32_matches_cpu_reference_across_variant_shapes() {
    let device = CudaDevice::new(0).expect("CUDA device must be available for ignored test");
    let selection =
        CudaGemmF32VariantSelection::new(&device).expect("variant selection handle must build");

    // (m, n, k, seed, category): DoubleBuffer 想定（大アラインメント形状）・
    // K 支配的非正方（撤退後は grid が SM を埋められなければ Simple・
    // 埋められれば DoubleBuffer）・Simple 想定（非整列・小 K）・境界
    // （33/65/1000 等の非整列サイズ・m=n=128,k=8192 の旧 SplitK 想定形状）。
    // category は `assert_matches_heuristic` が実際の選択を検証するために
    // 使う（イシュー #1035 PR #1073 レビュー指摘対応・#1100 で KDominant の
    // 期待値を更新: これまでは `selected_variant` の戻り値をログラベルに
    // しか使わず、DoubleBuffer が実際に選ばれたかを検証していなかった）。
    let cases: &[(u32, u32, u32, u64, ExpectedCategory)] = &[
        (
            4096,
            4096,
            4096,
            1,
            ExpectedCategory::DoubleBufferIfAvailable,
        ),
        (
            2048,
            2048,
            2048,
            2,
            ExpectedCategory::DoubleBufferIfAvailable,
        ),
        (128, 128, 8192, 3, ExpectedCategory::KDominant),
        (256, 256, 16384, 4, ExpectedCategory::KDominant),
        (1000, 1000, 1000, 5, ExpectedCategory::AlwaysSimple),
        (33, 65, 97, 6, ExpectedCategory::AlwaysSimple),
        (1, 1, 1, 7, ExpectedCategory::AlwaysSimple),
    ];

    let num_sms = selection.num_sms();
    let double_buffer_available = selection.double_buffer_available();
    let mut exercised = VariantExercised::default();

    for &(m, n, k, seed, category) in cases {
        let (a, b) = gen_ab(seed, m as usize, n as usize, k as usize);

        let variant = selection.selected_variant(m, n, k);
        let label = format!("gemm_f32_variants m={m} n={n} k={k} variant={variant:?}");

        let result = assert_matches_heuristic(
            &label,
            category,
            m,
            n,
            num_sms,
            double_buffer_available,
            variant,
        );
        exercised.double_buffer |= result.double_buffer;

        let actual = selection
            .run_f32(&a, &b, m, n, k)
            .unwrap_or_else(|e| panic!("run_f32 failed for m={m} n={n} k={k}: {e}"));

        let mut expected = vec![0.0f32; (m as usize) * (n as usize)];
        fandhe_ai_backend_cpu::matmul_reference_fma(
            &a,
            &b,
            &mut expected,
            m as usize,
            n as usize,
            k as usize,
        )
        .unwrap_or_else(|e| panic!("CPU reference failed for m={m} n={n} k={k}: {e:?}"));

        fandhe_ai_backend_cpu::assert_parity(&label, &actual, &expected);
    }

    // fail-soft フォールバック（NVRTC コンパイル失敗）自体は許容するが、
    // 「コンパイルには成功しているのに、テスト形状のどれも実際にその
    // カーネルを選ばなかった」場合は検証漏れなので失敗させる（イシュー
    // #1035 PR #1073 レビュー指摘のコア: 候補カーネルが実際に選択される
    // ことを最低 1 回は確認する）。
    if double_buffer_available {
        assert!(
            exercised.double_buffer,
            "double_buffer_available()=true だが、いずれの形状も実際には \
             DoubleBuffer を選ばなかった（テスト形状が不十分かヒューリス \
             ティックが回帰している）"
        );
    } else {
        eprintln!(
            "DoubleBuffer カーネル不可用（fail-soft フォールバックが有効）: \
             {:?}",
            selection.double_buffer_error()
        );
    }
}

/// SplitK 経路（診断専用。イシュー #1100）の複合判定・決定性。
///
/// `run_f32` からは到達しなくなった SplitK を `run_split_k_forced` で
/// 明示的に起動し、以下の 2 点を検証する:
///
/// 1. **決定性**（旧テストの主張の核）: 同一入力の 2 回実行が bit 一致
///    すること（atomics 不使用の設計〈`kernels_gemm_variants::
///    SPLITK_PARTIAL_F32`／`SPLITK_REDUCE_F32`〉の実機裏付け）。
/// 2. **複合判定 FAIL の再現**: GB10 実機実測（#1031）で検出した parity
///    FAIL（`docs/perf/cuda-gemm-f32-variant-selection.md` §1a・
///    `tests/splitk_reorder_error_host_model.rs` のホストモデルが同じ
///    破綻を再現済み）が実機でも一貫して再現することを記録する
///    （**FAIL することを期待する**。カーネルが決定的に「同じ誤差」を
///    出し続けることの確認であり、tolerance を満たすことの確認ではない
///    ——tolerance を満たしてしまった場合は #1100 の前提〈split 順序の
///    再結合誤差〉が崩れているため、その旨を明示して失敗させる）。
#[test]
#[ignore = "実機（CUDA/NVRTC 搭載）環境が必要。上記 run_f32_matches_cpu_reference_across_variant_shapes と同じ理由"]
fn split_k_forced_execution_is_bit_deterministic_and_reproduces_gb10_fail() {
    let device = CudaDevice::new(0).expect("CUDA device must be available for ignored test");
    let selection =
        CudaGemmF32VariantSelection::new(&device).expect("variant selection handle must build");

    if !selection.split_k_available() {
        eprintln!(
            "SplitK カーネル不可用（fail-soft フォールバックが有効。本テスト \
             は検証できていない）: partial={:?} reduce={:?}",
            selection.split_k_partial_error(),
            selection.split_k_reduce_error()
        );
        return;
    }

    // GB10 実機レポート（#1031）・`tests/splitk_reorder_error_host_model.rs`
    // と同一の形状・分割数・シード。
    let (m, n, k, num_splits) = (128u32, 128u32, 8192u32, 8u32);
    let (a, b) = gen_ab(3, m as usize, n as usize, k as usize);

    let first = selection
        .run_split_k_forced(&a, &b, m, n, k, num_splits)
        .expect("first run_split_k_forced must succeed");
    let second = selection
        .run_split_k_forced(&a, &b, m, n, k, num_splits)
        .expect("second run_split_k_forced must succeed");
    assert_eq!(
        first, second,
        "SplitK（num_splits={num_splits}）must produce bit-identical output \
         across repeated runs"
    );

    let mut expected = vec![0.0f32; (m as usize) * (n as usize)];
    fandhe_ai_backend_cpu::matmul_reference_fma(
        &a,
        &b,
        &mut expected,
        m as usize,
        n as usize,
        k as usize,
    )
    .unwrap_or_else(|e| panic!("CPU reference failed for m={m} n={n} k={k}: {e:?}"));

    let report = fandhe_ai_backend_cpu::compare(&first, &expected)
        .unwrap_or_else(|e| panic!("compare failed (length mismatch): {e}"));
    assert!(
        !report.passes(),
        "SplitK（num_splits={num_splits}）が GB10 実機レポート（#1031）と \
         同じ複合判定 FAIL を再現するはずが PASS になった（report={report:?}）。\
         #1100 の撤退判断の前提（split 順序の再結合誤差は実機で常に \
         発生する）が崩れている可能性があるため、`gemm_variant.rs` 冒頭\
         「SplitK 撤退の判断」の再検討が必要"
    );
}
